from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as fn

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.anytouch.contact import explicit_contact_observation
from openpi.picf.anytouch.contact import rot6d
from openpi.picf.anytouch.contracts import AnyTouchFeatureBundle
from openpi.picf.anytouch.history import MultiSensorTactileClipBuffer
from openpi.picf.anytouch.wrapper import AnyTouch2TactileEncoder
from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import RuntimeMeta
from openpi.picf.core.config import PicfCoreConfig
from openpi.picf.core.contracts import PicfConditionedControlState
from openpi.picf.core.contracts import PicfControlState
from openpi.picf.core.contracts import PicfCoreOutput
from openpi.picf.core.contracts import PicfCoreState
from openpi.picf.core.contracts import PicfObservationAnchorState
from openpi.picf.core.contracts import PicfPosteriorAnchorState
from openpi.picf.core.contracts import PicfPredictionCache
from openpi.picf.core.contracts import PicfPredictiveState
from openpi.picf.core.contracts import PicfProjectiveGeometryState
from openpi.picf.core.contracts import PicfTaskReadoutState
from openpi.picf.core.contracts import PicfTokenFieldState
from openpi.picf.core.tactile_contact import contact_prob_with_hysteresis
from openpi.picf.core.tactile_contact import summarize_contact_context
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.fsdp_utils import call_module_forward_or_method
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.visual_expert import load_camera_model
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticFeatures
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.history import VisualClipBuffer


def _resolve_device(config: PicfCoreConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(config: PicfCoreConfig) -> torch.dtype:
    if config.dtype == "float16":
        return torch.float16
    if config.dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _to_tensor(
    value: torch.Tensor | np.ndarray | Sequence[float] | float | int,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype or value.dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _assert_index_tensor_bounds(indices: torch.Tensor, *, size: int, name: str) -> None:
    if indices.numel() == 0:
        return
    indices_long = indices.to(dtype=torch.long)
    min_idx = int(indices_long.min().item())
    max_idx = int(indices_long.max().item())
    if min_idx < 0 or max_idx >= int(size):
        raise RuntimeError(
            f"PICF index out of bounds for {name}: "
            f"valid=[0,{int(size) - 1}] got min={min_idx} max={max_idx} "
            f"shape={tuple(indices.shape)}"
        )


def _normalize_tensor(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def _clip_vector_norm(x: torch.Tensor, *, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1e-12), max=1.0)
    return x * scale


def _add_role_embedding(tokens: torch.Tensor, embedding: nn.Embedding, role_id: int) -> torch.Tensor:
    if tokens.shape[0] == 0:
        return tokens
    role = embedding.weight[int(role_id)].to(device=tokens.device, dtype=tokens.dtype)
    return tokens + role[None, :]


def _mean_query_state(tokens: torch.Tensor, *, num_query_tokens: int) -> torch.Tensor:
    count = max(int(num_query_tokens), 1)
    if tokens.shape[0] < count:
        raise RuntimeError(
            "PICF query-token contract violated: "
            f"requested {count} query tokens from tensor with shape={tuple(tokens.shape)}."
        )
    return tokens[-count:].mean(dim=0)


def _variance_from_logvar(logvar: torch.Tensor, *, min_var: float, max_var: float) -> torch.Tensor:
    if min_var <= 0.0:
        raise ValueError(f"Expected min_var > 0, got {min_var}.")
    if max_var < min_var:
        raise ValueError(f"Expected max_var >= min_var, got min_var={min_var} max_var={max_var}.")
    logvar_min = math.log(float(min_var))
    logvar_max = math.log(float(max_var))
    return torch.exp(torch.clamp(logvar, min=logvar_min, max=logvar_max))


def _fourier_features(x: torch.Tensor, *, scale: float, bands: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(*x.shape[:-1], x.shape[-1] * bands * 2, device=x.device, dtype=x.dtype)
    feats = []
    denom = max(scale, 1e-6)
    for k in range(bands):
        omega = (2.0**k) * math.pi / denom
        feats.append(torch.sin(omega * x))
        feats.append(torch.cos(omega * x))
    return torch.cat(feats, dim=-1)


def _point_pe(x: torch.Tensor, config: PicfCoreConfig) -> torch.Tensor:
    return torch.cat(
        [
            _fourier_features(x, scale=config.workspace_radius_m, bands=4),
            _fourier_features(x, scale=max(config.crop_radius_m / 4.0, 1e-6), bands=8),
        ],
        dim=-1,
    )


def _grid_index_to_norm(grid_index: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if grid_index.numel() == 0:
        return torch.zeros_like(grid_index)
    x = grid_index[..., 0]
    y = grid_index[..., 1]
    x_norm = (2.0 * x / max(width - 1, 1)) - 1.0
    y_norm = (2.0 * y / max(height - 1, 1)) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _point_proj_fourier(grid_coord: torch.Tensor, *, bands: int) -> torch.Tensor:
    if grid_coord.numel() == 0:
        return torch.zeros((*grid_coord.shape[:-1], grid_coord.shape[-1] * bands * 2), device=grid_coord.device, dtype=grid_coord.dtype)
    feats = []
    for k in range(bands):
        omega = (2.0**k) * math.pi
        feats.append(torch.sin(omega * grid_coord))
        feats.append(torch.cos(omega * grid_coord))
    return torch.cat(feats, dim=-1)


def _projective_candidate_radius_patches(*, sigma_proj: float, tau_proj: float) -> float:
    tau = min(max(float(tau_proj), 1e-6), 1.0 - 1e-6)
    sigma = max(float(sigma_proj), 1e-6)
    return math.sqrt(max(0.0, -2.0 * (sigma**2) * math.log(tau)))


def _sparse_projective_neighborhood_mask(
    point_proj_grid_index: torch.Tensor,
    visibility: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    radius_patches: float,
) -> torch.Tensor:
    mask = torch.zeros(
        (point_proj_grid_index.shape[0], grid_h * grid_w),
        device=point_proj_grid_index.device,
        dtype=torch.bool,
    )
    if mask.numel() == 0:
        return mask
    radius2 = float(radius_patches**2) + 1e-6
    for point_index in range(point_proj_grid_index.shape[0]):
        if not bool(visibility[point_index].item()):
            continue
        center_x = float(point_proj_grid_index[point_index, 0].item())
        center_y = float(point_proj_grid_index[point_index, 1].item())
        x0 = max(int(math.floor(center_x - radius_patches)), 0)
        x1 = min(int(math.ceil(center_x + radius_patches)), grid_w - 1)
        y0 = max(int(math.floor(center_y - radius_patches)), 0)
        y1 = min(int(math.ceil(center_y + radius_patches)), grid_h - 1)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                dx = center_x - float(xx)
                dy = center_y - float(yy)
                if (dx * dx) + (dy * dy) <= radius2:
                    mask[point_index, (yy * grid_w) + xx] = True
    return mask


def _geometry_pe(x: torch.Tensor, a: torch.Tensor, S: torch.Tensor, config: PicfCoreConfig) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((0, 0), device=x.device, dtype=x.dtype)
    eigvals = torch.linalg.eigvalsh(S)
    return torch.cat(
        [
            _point_pe(x, config),
            torch.log(torch.clamp(a, min=1e-6)),
            torch.log(torch.clamp(eigvals, min=1e-6)),
        ],
        dim=-1,
    )


def _diag_from_cov(cov: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(cov, dim1=-2, dim2=-1)


def _diag_embed(var: torch.Tensor) -> torch.Tensor:
    return torch.diag_embed(var)


def _extent_from_cov(S: torch.Tensor, config: PicfCoreConfig) -> torch.Tensor:
    if S.numel() == 0:
        return torch.zeros((S.shape[0], 3), device=S.device, dtype=S.dtype)
    eigvals = torch.linalg.eigvalsh(S)
    extent = 2.0 * torch.sqrt(torch.clamp(torch.flip(eigvals, dims=(-1,)), min=config.epsilon_ext_m2))
    a_min = _to_tensor(config.a_min_m, device=S.device, dtype=S.dtype)
    a_max = _to_tensor(config.a_max_m, device=S.device, dtype=S.dtype)
    return torch.clamp(extent, min=a_min, max=a_max)


def _weighted_cov(points: torch.Tensor, weights: torch.Tensor, center: torch.Tensor, config: PicfCoreConfig) -> torch.Tensor:
    if points.numel() == 0:
        eye = torch.eye(3, device=center.device, dtype=center.dtype)
        return eye[None, :].expand(center.shape[0], -1, -1) * config.epsilon_s
    centered = points[None, :, :] - center[:, None, :]
    cov = torch.einsum("nm,nmi,nmj->nij", weights, centered, centered)
    eye = torch.eye(3, device=points.device, dtype=points.dtype)[None, :, :]
    return cov + (config.epsilon_s * eye)


def _bilinear_sample_depth(depth_image: torch.Tensor, uv_pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if depth_image.numel() == 0 or uv_pixels.numel() == 0:
        empty = torch.zeros((uv_pixels.shape[0],), device=uv_pixels.device, dtype=uv_pixels.dtype)
        return empty, torch.zeros_like(empty, dtype=torch.bool)
    height, width = depth_image.shape
    x = uv_pixels[:, 0]
    y = uv_pixels[:, 1]
    finite = torch.isfinite(x) & torch.isfinite(y)
    in_bounds = finite & (x >= 0.0) & (x <= float(width - 1)) & (y >= 0.0) & (y <= float(height - 1))
    x0 = torch.floor(torch.clamp(x, 0.0, float(width - 1))).long()
    y0 = torch.floor(torch.clamp(y, 0.0, float(height - 1))).long()
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    dx = x - x0.to(dtype=uv_pixels.dtype)
    dy = y - y0.to(dtype=uv_pixels.dtype)
    wa = (1.0 - dx) * (1.0 - dy)
    wb = (1.0 - dx) * dy
    wc = dx * (1.0 - dy)
    wd = dx * dy
    sampled = (
        depth_image[y0, x0] * wa
        + depth_image[y1, x0] * wb
        + depth_image[y0, x1] * wc
        + depth_image[y1, x1] * wd
    )
    sampled_valid = in_bounds & torch.isfinite(sampled)
    return sampled, sampled_valid


def _fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[0] == 0 or count <= 0:
        return torch.zeros((0,), device=points.device, dtype=torch.long)
    if points.shape[0] <= count:
        return torch.arange(points.shape[0], device=points.device, dtype=torch.long)
    centroid = points.mean(dim=0)
    dists = torch.linalg.norm(points - centroid[None, :], dim=-1)
    first = int(torch.argmax(dists).item())
    chosen = [first]
    min_dist = torch.linalg.norm(points - points[first : first + 1], dim=-1)
    while len(chosen) < count:
        next_idx = int(torch.argmax(min_dist).item())
        chosen.append(next_idx)
        min_dist = torch.minimum(min_dist, torch.linalg.norm(points - points[next_idx : next_idx + 1], dim=-1))
    return torch.as_tensor(chosen, device=points.device, dtype=torch.long)


def _build_identity_frame_context(
    observation: PicfObservation,
    *,
    crop_radius_m: float,
    focus_centers_world: np.ndarray,
) -> PointFrameContext:
    assert observation.point_set is not None
    xyz_world = np.asarray(observation.point_set.xyz_world, dtype=np.float32)
    centers = np.asarray(focus_centers_world, dtype=np.float32).reshape(-1, 3)
    dists = np.linalg.norm(xyz_world[:, None, :] - centers[None, :, :], axis=-1).min(axis=1)
    keep = dists <= float(crop_radius_m)
    return PointFrameContext(
        grid_coord=np.asarray(observation.point_set.grid_coord[keep], dtype=np.int32),
        points_local=np.asarray(xyz_world[keep], dtype=np.float32),
        normals_local=np.asarray(observation.point_set.normal_world[keep], dtype=np.float32),
        colors=np.asarray(observation.point_set.rgb[keep], dtype=np.float32),
        local_mask=keep,
        world_to_local=np.eye(4, dtype=np.float32),
        G_t=np.asarray(observation.G_t, dtype=np.float32),
    )


def _focus_centers_world_from_observation(observation: PicfObservation) -> np.ndarray:
    if observation.G_t is None:
        raise ValueError("PICF focus center construction requires observation.G_t to be set.")
    centers = [np.asarray(observation.G_t[:3, 3], dtype=np.float32)]
    packet = observation.tactile
    if packet is not None:
        for sensor in packet.sensors:
            if not sensor.valid:
                continue
            sensor_pose_world = np.asarray(observation.G_t, dtype=np.float32) @ np.asarray(sensor.T_sens_to_wrist, dtype=np.float32)
            centers.append(np.asarray(sensor_pose_world[:3, 3], dtype=np.float32))
    return np.stack(centers, axis=0)


class ResidualMLP(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int, *, zero_init_last: bool = False):
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        if zero_init_last:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(fn.silu(self.fc1(x)))


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        del layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.shape[1] == 0:
            return x, None
        attn_in = self.norm1(x)
        attn_out, attn_weights = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=attn_bias,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights if return_attention else None


class TransformerStack(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int, *, activation_checkpointing: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(SelfAttentionBlock(hidden_dim, heads, 1) for _ in range(layers))
        self.activation_checkpointing = bool(activation_checkpointing)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if x.shape[1] == 0:
            return (x, None) if return_attention else x
        attn_maps = []
        for layer in self.layers:
            if (
                self.activation_checkpointing
                and self.training
                and not return_attention
                and bool(x.requires_grad)
            ):
                def _forward(layer_inputs: torch.Tensor) -> torch.Tensor:
                    layer_output, _ = layer(layer_inputs, attn_bias=attn_bias, return_attention=False)
                    return layer_output

                x = torch.utils.checkpoint.checkpoint(
                    _forward,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                attn = None
            else:
                x, attn = layer(x, attn_bias=attn_bias, return_attention=return_attention)
            if return_attention and attn is not None:
                attn_maps.append(attn.mean(dim=1))
        if return_attention:
            fused_attn = torch.stack(attn_maps, dim=0).mean(dim=0)[0] if attn_maps else None
            return x, fused_attn
        return x


class CrossAttentionRead(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, *, attn_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape[1] == 0:
            weights = torch.zeros((queries.shape[1], 0), device=queries.device, dtype=queries.dtype)
            return queries, weights
        output, weights = self.attn(
            queries,
            keys,
            keys,
            attn_mask=attn_bias,
            need_weights=True,
            average_attn_weights=False,
        )
        output = queries + output
        output = output + self.ff(self.norm(output))
        mean_weights = weights.mean(dim=1)[0]
        return output, mean_weights


class GatedCrossAttentionRead(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, heads: int, *, inner_dim: int, gate_init: float = 0.0):
        super().__init__()
        if inner_dim % heads != 0:
            raise ValueError(
                f"inner_dim must be divisible by heads for heterogeneous cross attention; "
                f"got inner_dim={inner_dim} heads={heads}."
            )
        self.query_norm = nn.LayerNorm(query_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(
            query_dim,
            heads,
            batch_first=True,
            kdim=kv_dim,
            vdim=kv_dim,
        )
        self.cross_gate = nn.Parameter(torch.tensor([float(gate_init)]))
        self.ff_norm = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, query_dim),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape[1] == 0:
            weights = torch.zeros((queries.shape[1], 0), device=queries.device, dtype=queries.dtype)
            return queries, weights
        query_in = self.query_norm(queries)
        key_in = self.kv_norm(keys)
        attn_out, weights = self.attn(
            query_in,
            key_in,
            key_in,
            attn_mask=attn_bias,
            need_weights=True,
            average_attn_weights=False,
        )
        # Keep the entire read block dormant until the semantic gate opens. If the
        # FF residual is left ungated, the block can learn a prompt-independent
        # query transform while semantic cross-attention remains effectively shut.
        gate = torch.tanh(self.cross_gate)
        output = queries + (gate * attn_out)
        output = output + (gate * self.ff(self.ff_norm(output)))
        mean_weights = weights.mean(dim=1)[0]
        return output, mean_weights


class LazyCrossAttentionRead(nn.Module):
    def __init__(self, query_dim: int, *, inner_dim: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, inner_dim)
        self.key_proj = nn.LazyLinear(inner_dim)
        self.value_proj = nn.LazyLinear(query_dim)
        self.cross_gate = nn.Parameter(torch.zeros(1))
        self.ff_norm = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, query_dim),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape[1] == 0:
            weights = torch.zeros((queries.shape[0], queries.shape[1], 0), device=queries.device, dtype=queries.dtype)
            return queries, weights
        query_in = self.query_proj(self.query_norm(queries))
        key_in = self.key_proj(fn.layer_norm(keys, (keys.shape[-1],)))
        value_in = self.value_proj(fn.layer_norm(keys, (keys.shape[-1],)))
        logits = torch.matmul(query_in, key_in.transpose(-2, -1)) / math.sqrt(max(query_in.shape[-1], 1))
        if attn_bias is not None:
            if attn_bias.ndim == 2:
                logits = logits + attn_bias[None, :, :]
            else:
                logits = logits + attn_bias
        weights = torch.softmax(logits, dim=-1)
        attn_out = torch.matmul(weights, value_in)
        gate = torch.tanh(self.cross_gate)
        output = queries + (gate * attn_out)
        output = output + (gate * self.ff(self.ff_norm(output)))
        return output, weights


class WorldLatentFusionStack(nn.Module):
    def __init__(
        self,
        world_dim: int,
        semantic_dim: int,
        heads: int,
        *,
        self_layers: int,
        cross_layers: int,
        cross_inner_dim: int,
    ):
        super().__init__()
        self.self_layers = nn.ModuleList(SelfAttentionBlock(world_dim, heads, 1) for _ in range(self_layers))
        self.cross_layers = nn.ModuleList(
            GatedCrossAttentionRead(world_dim, semantic_dim, heads, inner_dim=cross_inner_dim)
            for _ in range(cross_layers)
        )

    def forward(self, world_tokens: torch.Tensor, semantic_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = world_tokens[None, :]
        attention = None
        num_rounds = max(len(self.self_layers), len(self.cross_layers))
        for index in range(num_rounds):
            if index < len(self.self_layers):
                x, _ = self.self_layers[index](x)
            if index < len(self.cross_layers):
                x, attention = self.cross_layers[index](x, semantic_tokens[None, :])
        return x[0], attention


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        self.score = nn.LazyLinear(1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return torch.zeros((tokens.shape[0], tokens.shape[-1]), device=tokens.device, dtype=tokens.dtype)
        query = self.query[None, None, :].expand(tokens.shape[0], tokens.shape[1], -1)
        score = self.score(tokens + query).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        return torch.sum(weight[..., None] * tokens, dim=1)


class VoteHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.delta_mu = ResidualMLP(latent_dim, hidden_dim, zero_init_last=True)
        self.logvar = ResidualMLP(latent_dim, hidden_dim, zero_init_last=True)
        self.confidence = ResidualMLP(1, hidden_dim, zero_init_last=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.delta_mu(x), self.logvar(x), self.confidence(x).squeeze(-1)


class PaliGemmaSemanticWrapper:
    def summarize(
        self,
        *,
        outputs: Any | None = None,
        last_hidden_state: torch.Tensor | None = None,
        image_hidden_states: Sequence[torch.Tensor] | torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if outputs is not None:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            image_hidden_states = getattr(outputs, "image_hidden_states", None)
        if last_hidden_state is None or image_hidden_states is None:
            raise ValueError("Need PaliGemma outputs or explicit hidden states.")
        if isinstance(image_hidden_states, torch.Tensor):
            img_hidden = image_hidden_states[:, 0]
        else:
            img_hidden = image_hidden_states[0]
        if prompt_mask is None:
            txt = last_hidden_state.mean(dim=1)
        else:
            denom = torch.clamp(prompt_mask.sum(dim=1, keepdim=True), min=1)
            txt = (last_hidden_state * prompt_mask[..., None]).sum(dim=1) / denom
        img = img_hidden.mean(dim=1)
        return torch.cat([txt, img], dim=-1)


@dataclasses.dataclass(frozen=True)
class _SemanticContext:
    tokens: torch.Tensor
    prefix_tokens: torch.Tensor
    available: bool


@dataclasses.dataclass(frozen=True)
class _StepDenseMemory:
    point_payload: torch.Tensor
    visual_payload: torch.Tensor
    tactile_group_tokens: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class _ObservedStepState:
    runtime_meta: RuntimeMeta
    G_t: torch.Tensor
    token_field: PicfTokenFieldState
    dense_memory: _StepDenseMemory
    observation_anchors: PicfObservationAnchorState
    posterior: PicfPosteriorAnchorState
    current_targets: dict[str, torch.Tensor | None]
    availability: torch.Tensor
    innovation_token: torch.Tensor
    innovation_norm: torch.Tensor
    semantic: _SemanticContext
    proprio_token: torch.Tensor
    task_readout: PicfTaskReadoutState
    conditioned_control: PicfConditionedControlState
    control: PicfControlState
    last_prompt: str | None


class PicfFullCore(nn.Module):
    def __init__(
        self,
        pointcloud_builder: CalvinDepthToPicfPointCloud,
        *,
        config: PicfCoreConfig | None = None,
        local_frame: EndEffectorLocalFrame | None = None,
        point_feature_extractor: SonataPointFeatureExtractor | None = None,
        visual_config: VjepaVisualConfig | None = None,
        visual_encoder: Any | None = None,
        tactile_config: AnyTouchConfig | None = None,
        tactile_encoder: Any | None = None,
        semantic_wrapper: PaliGemmaSemanticWrapper | None = None,
    ):
        super().__init__()
        self.config = config or PicfCoreConfig()
        self.device = _resolve_device(self.config)
        self.dtype = _resolve_dtype(self.config)
        self.pointcloud_builder = pointcloud_builder
        self.local_frame = local_frame or EndEffectorLocalFrame()
        self.point_feature_extractor = point_feature_extractor
        self.visual_config = visual_config
        self.visual_encoder = visual_encoder
        self.semantic_wrapper = semantic_wrapper or PaliGemmaSemanticWrapper()
        self.tactile_config = tactile_config or AnyTouchConfig(
            device=self.config.device,
            dtype=self.config.dtype,
            num_frames=4,
            stride=2,
            allow_random_init=False,
            require_background=True,
        )
        self.tactile_encoder = tactile_encoder
        self.camera_model = None
        self.clip_buffer = None
        self.tactile_buffer = MultiSensorTactileClipBuffer(
            num_frames=self.tactile_config.num_frames,
            frame_stride=self.tactile_config.stride,
        )
        if self.visual_config is not None:
            if self.visual_encoder is None:
                from openpi.picf.vjepa.wrapper import Vjepa2VisualEncoder

                self.visual_encoder = Vjepa2VisualEncoder(self.visual_config)
            if self.visual_config.camera_json_path is not None:
                self.camera_model = load_camera_model(self.visual_config.camera_json_path, camera_name=self.visual_config.camera_name)
            self.clip_buffer = VisualClipBuffer(num_frames=self.visual_config.num_frames)
        if self.tactile_encoder is None:
            self.tactile_encoder = AnyTouch2TactileEncoder(self.tactile_config) if self.tactile_config is not None else None

        hidden_dim = self.config.hidden_dim
        semantic_trunk_dim = self.config.semantic_dim
        heads = self.config.attention_heads
        self.modality_embedding = nn.Embedding(4, hidden_dim)
        self.point_token_proj = nn.LazyLinear(hidden_dim)
        self.visual_token_proj = nn.LazyLinear(hidden_dim)
        self.tactile_token_proj = nn.LazyLinear(hidden_dim)
        self.point_align_proj = nn.LazyLinear(hidden_dim)
        self.visual_align_proj = nn.LazyLinear(hidden_dim)
        self.tactile_align_proj = nn.LazyLinear(hidden_dim)
        self.proprio_context_proj = nn.LazyLinear(hidden_dim)
        self.action_context_proj = nn.LazyLinear(hidden_dim)
        self.timing_context_proj = nn.LazyLinear(hidden_dim)
        self.contact_context_proj = nn.LazyLinear(hidden_dim)
        proj_coarse_dim = 2 * 4 * 2
        proj_fine_dim = 2 * 8 * 2
        self.null_proj_coarse = nn.Parameter(torch.zeros(proj_coarse_dim, device=self.device, dtype=self.dtype))
        self.null_proj_fine = nn.Parameter(torch.zeros(proj_fine_dim, device=self.device, dtype=self.dtype))
        self.projective_bias_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.token_fusion = TransformerStack(
            hidden_dim,
            heads,
            self.config.fusion_layers,
            activation_checkpointing=True,
        )

        self.obs_reader = CrossAttentionRead(hidden_dim, heads)
        self.obs_self = TransformerStack(hidden_dim, heads, 1, activation_checkpointing=True)

        self.prior_proj = nn.LazyLinear(self.config.future_hidden_dim)
        self.prior_delta_mu = ResidualMLP(self.config.latent_dim, self.config.future_hidden_dim, zero_init_last=True)
        self.prior_delta_logvar = ResidualMLP(self.config.latent_dim, self.config.future_hidden_dim, zero_init_last=True)
        self.prior_lstm = nn.LSTMCell(self.config.future_hidden_dim, self.config.posterior_hidden_dim)
        self.activity_head = ResidualMLP(1, hidden_dim)
        self.recycle_head = ResidualMLP(1, hidden_dim)
        self.residual_mu_head = ResidualMLP(self.config.latent_dim, hidden_dim)
        self.residual_logvar_head = ResidualMLP(self.config.latent_dim, hidden_dim)
        self.residual_h_head = ResidualMLP(self.config.posterior_hidden_dim, hidden_dim)
        self.residual_c_head = ResidualMLP(self.config.posterior_hidden_dim, hidden_dim)

        self.anchor_seed_proj = nn.LazyLinear(hidden_dim)
        self.anchor_reader = CrossAttentionRead(hidden_dim, heads)
        self.contact_head = ResidualMLP(1, hidden_dim)
        self.vote_heads = nn.ModuleList(VoteHead(self.config.latent_dim, hidden_dim) for _ in range(self.config.future_vote_heads))

        self.post_write_proj = nn.LazyLinear(hidden_dim)
        self.post_lstm = nn.LSTMCell(hidden_dim, self.config.posterior_hidden_dim)
        self.posterior_token_proj = nn.LazyLinear(hidden_dim)
        self.posterior_self = TransformerStack(
            hidden_dim,
            heads,
            self.config.posterior_layers,
            activation_checkpointing=True,
        )
        self.posterior_pool = AttentionPool(hidden_dim)

        self.semantic_prefix_proj = nn.Identity()
        self.proprio_proj = nn.LazyLinear(hidden_dim)
        self.action_cond_proj = nn.LazyLinear(hidden_dim)
        self.task_query_tokens = nn.Parameter(torch.zeros((self.config.task_local_queries, hidden_dim)))
        self.task_global_query_tokens = nn.Parameter(torch.zeros((self.config.task_global_queries, hidden_dim)))
        self.task_instruction_query_tokens = nn.Parameter(torch.zeros((self.config.task_instruction_queries, hidden_dim)))
        self.task_query_conditioner = GatedCrossAttentionRead(
            hidden_dim,
            semantic_trunk_dim,
            heads,
            inner_dim=max(self.config.semantic_cross_dim, hidden_dim),
            gate_init=1.0,
        )
        self.task_public_reader = CrossAttentionRead(hidden_dim, heads)
        self.task_visual_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.task_tactile_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.task_point_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.task_self = TransformerStack(
            hidden_dim,
            heads,
            self.config.task_self_layers,
            activation_checkpointing=True,
        )
        self.task_geom_proj = nn.LazyLinear(hidden_dim)
        self.posterior_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.global_post_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.innovation_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.proprio_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.task_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.task_global_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.instruction_to_control_proj = nn.LazyLinear(semantic_trunk_dim)
        self.control_role_embedding = nn.Embedding(8, semantic_trunk_dim)
        self.predictive_physical_role_embedding = nn.Embedding(4, hidden_dim)
        self.physical_pred_to_conditioned_proj = nn.LazyLinear(semantic_trunk_dim)
        self.predictive_conditioned_role_embedding = nn.Embedding(3, semantic_trunk_dim)
        self.control_query_tokens = nn.Parameter(torch.zeros((self.config.conditioned_control_queries, semantic_trunk_dim)))
        self.predictive_query_tokens = nn.Parameter(torch.zeros((self.config.conditioned_future_queries, semantic_trunk_dim)))
        self.pi_prefix_query_tokens = nn.Parameter(torch.zeros((self.config.pi_prefix_queries, semantic_trunk_dim)))
        self.pi_prefix_reader = CrossAttentionRead(semantic_trunk_dim, heads)
        self.future_condition_reader = CrossAttentionRead(semantic_trunk_dim, heads)
        self.predictive_world = TransformerStack(
            hidden_dim,
            heads,
            self.config.predictive_layers,
            activation_checkpointing=True,
        )
        self.predictive_semantic_world = TransformerStack(
            semantic_trunk_dim,
            heads,
            self.config.predictive_layers,
            activation_checkpointing=True,
        )
        self.predictive_pool = AttentionPool(hidden_dim)

        self.visual_latent_queries = nn.Parameter(torch.zeros((self.config.visual_latent_tokens, hidden_dim)))
        self.tactile_latent_queries = nn.Parameter(torch.zeros((self.config.tactile_latent_tokens, hidden_dim)))
        self.point_latent_queries = nn.Parameter(torch.zeros((self.config.point_latent_tokens, hidden_dim)))
        self.visual_latent_head = nn.Linear(hidden_dim, self.config.visual_latent_dim)
        self.visual_real_head = nn.Linear(hidden_dim, self.config.visual_real_dim)
        self.tactile_real_head = nn.Linear(hidden_dim, self.config.tactile_real_dim)
        self.point_real_head = nn.Linear(hidden_dim, self.config.point_real_dim)

        branch_dim = max(hidden_dim // 4, 32)
        self.visual_error_encoder = nn.LazyLinear(branch_dim)
        self.visual_real_error_encoder = nn.LazyLinear(branch_dim)
        self.tactile_error_encoder = nn.LazyLinear(branch_dim)
        self.point_error_encoder = nn.LazyLinear(branch_dim)
        self.innovation_proj = nn.LazyLinear(self.config.innovation_dim)
        self.innovation_token_proj = nn.Linear(self.config.innovation_dim, hidden_dim)
        self.visual_native_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.tactile_native_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.point_native_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.tactile_group_route_queries = nn.Parameter(torch.zeros((self.config.tactile_group_proposals, hidden_dim)))
        self.tactile_route_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.evidence_delta = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.evidence_delta[-1].weight)
        nn.init.zeros_(self.evidence_delta[-1].bias)
        self.evidence_gate = nn.LazyLinear(hidden_dim)

        self.control_world = TransformerStack(
            semantic_trunk_dim,
            heads,
            self.config.control_layers,
            activation_checkpointing=True,
        )
        self.control_state_proj = nn.Linear(semantic_trunk_dim, self.config.control_dim)
        self.predictive_state_proj = nn.Linear(semantic_trunk_dim, hidden_dim)
        self.to(device=self.device, dtype=self.dtype)

    def _build_runtime_meta(self, observation: PicfObservation, previous: RuntimeMeta | None) -> RuntimeMeta:
        meta = dataclasses.replace(observation.runtime_meta) if observation.runtime_meta is not None else (dataclasses.replace(previous) if previous is not None else RuntimeMeta())
        rgb = np.asarray(observation.rgb_static)
        visual_available = bool(rgb.size > 0 and np.isfinite(rgb).all())
        if visual_available:
            meta.t_v_last = float(observation.timestamp_s)
            meta.n_vis_upd = 1 if observation.reset_scaffold else (meta.n_vis_upd + 1)
        packet = observation.tactile
        tactile_available = bool(packet is not None and any(sensor.valid for sensor in packet.sensors))
        if tactile_available:
            meta.t_t_last = float(max(sensor.timestamp_s for sensor in packet.sensors if sensor.valid))
        point_ok = False
        if observation.point_set is not None and observation.point_set.frame_valid:
            rgb_ok = observation.point_set.rgb.ndim == 2 and observation.point_set.rgb.shape[1] == 3 and np.isfinite(observation.point_set.rgb).all()
            xyz_ok = observation.point_set.xyz_world.ndim == 2 and observation.point_set.xyz_world.shape[1] == 3 and np.isfinite(observation.point_set.xyz_world).all()
            point_ok = bool(rgb_ok and xyz_ok)
            if point_ok:
                meta.t_p_last = float(observation.timestamp_s)
                meta.t_rgb_last = float(observation.timestamp_s)
                meta.b_rgb_avail = True
                meta.rgb_proj_residual = 0.0
        if not point_ok:
            meta.b_rgb_avail = False
            meta.rgb_proj_residual = float("inf")
        meta.visual_available = visual_available
        meta.tactile_available = tactile_available and (float(observation.timestamp_s) - meta.t_t_last) <= self.config.tactile_stale_s
        meta.point_contract_ok = point_ok if self.config.pointcloud_requires_rgb else bool(observation.point_set is not None and observation.point_set.frame_valid)
        meta.v_pc_scaf = meta.point_contract_ok
        meta.v_rgb_p = meta.point_contract_ok
        sync_terms = []
        if meta.visual_available and meta.point_contract_ok:
            sync_terms.append(abs(meta.t_v_last - meta.t_p_last))
        if meta.tactile_available and meta.point_contract_ok:
            sync_terms.append(abs(meta.t_t_last - meta.t_p_last))
        meta.sync_valid = all(delta <= self.config.sync_tolerance_s for delta in sync_terms)
        meta.stale_scaffold_steps = 0 if meta.point_contract_ok else (0 if previous is None else previous.stale_scaffold_steps + 1)
        return meta

    def _point_subset(self, observation: PicfObservation) -> PointFrameContext:
        assert observation.point_set is not None
        focus_centers_world = _focus_centers_world_from_observation(observation)
        return _build_identity_frame_context(observation, crop_radius_m=self.config.crop_radius_m, focus_centers_world=focus_centers_world)

    def _extract_point_features(self, frame_context: PointFrameContext, override: torch.Tensor | np.ndarray | None) -> torch.Tensor:
        if override is not None:
            feature = _to_tensor(override, device=self.device, dtype=self.dtype)
            return feature if feature.ndim == 2 else feature.squeeze(0)
        if self.point_feature_extractor is None:
            return _to_tensor(frame_context.colors, device=self.device, dtype=self.dtype)
        encoded = call_module_forward_or_method(self.point_feature_extractor, "encode_local_context", frame_context)
        return _to_tensor(encoded.features, device=self.device, dtype=self.dtype)

    def _visual_map(self, observation: PicfObservation, override: torch.Tensor | np.ndarray | None, meta: RuntimeMeta) -> torch.Tensor | None:
        if override is not None:
            visual = _to_tensor(override, device=self.device, dtype=self.dtype)
            return visual if visual.ndim == 3 else visual.squeeze(0)
        if self.visual_encoder is None or self.clip_buffer is None or self.visual_config is None:
            return None
        rgb = np.asarray(observation.rgb_static)
        if meta.visual_available:
            self.clip_buffer.push(rgb, segment_id=int(observation.segment_id), reset=bool(observation.reset_scaffold))
        if not self.clip_buffer.has_frames:
            return None
        fmap = call_module_forward_or_method(self.visual_encoder, "encode_clip", self.clip_buffer.get_clip())
        return _to_tensor(fmap.current_map(use_last_two_mean=self.visual_config.use_last_two_mean), device=self.device, dtype=self.dtype)

    def _tactile_features(self, observation: PicfObservation, meta: RuntimeMeta) -> AnyTouchFeatureBundle | None:
        if self.tactile_encoder is None:
            return None
        if observation.reset_scaffold:
            self.tactile_buffer.reset(segment_id=int(observation.segment_id))
        packet = observation.tactile
        if packet is not None and any(sensor.valid for sensor in packet.sensors):
            self.tactile_buffer.push(packet, segment_id=int(observation.segment_id), reset=bool(observation.reset_scaffold))
        if not meta.tactile_available:
            return None
        sensor_names = [name for name in self.tactile_buffer.sensor_names if self.tactile_buffer.has_frames(name)]
        if not sensor_names:
            return None
        clips = {name: self.tactile_buffer.get_clip(name) for name in sensor_names}
        backgrounds = {name: self.tactile_buffer.background_for(name) for name in sensor_names}
        poses = {name: self.tactile_buffer.latest_pose(name) for name in sensor_names}
        return call_module_forward_or_method(
            self.tactile_encoder,
            "encode_sensor_clips",
            clips_by_sensor=clips,
            backgrounds_by_sensor=backgrounds,
            poses_by_sensor=poses,
        )

    def _zero_semantic_context(self) -> _SemanticContext:
        return _SemanticContext(
            tokens=torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype),
            prefix_tokens=torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype),
            available=False,
        )

    def _project_semantic_context(
        self,
        *,
        tokens_raw: torch.Tensor,
    ) -> _SemanticContext:
        if tokens_raw.ndim == 1:
            tokens_raw = tokens_raw[None, :]
        tokens_raw = _to_tensor(tokens_raw, device=self.device, dtype=self.dtype)
        if tokens_raw.shape[0] > 0 and int(tokens_raw.shape[-1]) != int(self.config.semantic_dim):
            raise RuntimeError(
                "Semantic token width mismatch. "
                f"Expected semantic_dim={self.config.semantic_dim}, got tokens shape={tuple(tokens_raw.shape)}."
            )
        semantic_tokens = (
            tokens_raw
            if tokens_raw.shape[0] > 0
            else torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype)
        )
        semantic_prefix_tokens = (
            self.semantic_prefix_proj(semantic_tokens)
            if semantic_tokens.shape[0] > 0
            else torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype)
        )
        return _SemanticContext(tokens=semantic_tokens, prefix_tokens=semantic_prefix_tokens, available=True)

    def _semantic_context(
        self,
        observation: PicfObservation,
        previous: PicfCoreState | None,
        semantic_override: Any | None,
    ) -> _SemanticContext:
        if not self.config.language_enabled:
            return self._zero_semantic_context()
        if semantic_override is None:
            return self._zero_semantic_context()
        if isinstance(semantic_override, PaliGemmaSemanticFeatures):
            return self._project_semantic_context(tokens_raw=semantic_override.tokens)
        if isinstance(semantic_override, torch.Tensor | np.ndarray):
            raw = _to_tensor(semantic_override, device=self.device, dtype=self.dtype)
            raw = raw if raw.ndim == 2 else raw[None, :]
            return self._project_semantic_context(tokens_raw=raw)
        if isinstance(semantic_override, dict):
            if "tokens" in semantic_override:
                tokens_raw = semantic_override["tokens"]
                return self._project_semantic_context(tokens_raw=tokens_raw)
            raise RuntimeError(
                "PICF semantic override contract now requires token-level semantic inputs. "
                "Pass PaliGemmaSemanticFeatures or a dict containing 'tokens'."
            )
        else:
            raise RuntimeError(
                "PICF semantic override contract now requires token-level semantic inputs. "
                "Pass PaliGemmaSemanticFeatures instead of summary-only outputs."
            )

    def _previous_action(self, previous: PicfCoreState | None) -> torch.Tensor:
        if previous is None:
            return torch.zeros((7,), device=self.device, dtype=self.dtype)
        executed = getattr(previous.predictive, "executed_action", None)
        if executed is None:
            executed = previous.predictive.action
        return executed.detach().to(device=self.device, dtype=self.dtype)

    def _executed_action(self, observation: PicfObservation, predicted_action: torch.Tensor) -> torch.Tensor:
        if observation.action is None:
            return predicted_action.detach()
        action = _to_tensor(observation.action, device=self.device, dtype=self.dtype).reshape(-1)
        if action.numel() < 7:
            action = fn.pad(action, (0, 7 - action.numel()))
        elif action.numel() > 7:
            action = action[:7]
        return action

    def _default_predictive_action(
        self,
        action_future: torch.Tensor | np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if action_future is None:
            return torch.zeros((7,), device=self.device, dtype=self.dtype), None
        action_tensor = _to_tensor(action_future, device=self.device, dtype=self.dtype)
        action_chunk = action_tensor if action_tensor.ndim > 1 else action_tensor[None, :]
        action = action_chunk[0]
        if action.numel() < 7:
            action = fn.pad(action, (0, 7 - action.numel()))
        elif action.numel() > 7:
            action = action[:7]
        return self._clip_action(action), action_chunk

    def _semantic_memory(
        self,
        semantic_tokens: torch.Tensor,
        *,
        dropout_prob: float,
    ) -> torch.Tensor:
        if semantic_tokens.shape[0] == 0 or dropout_prob <= 0.0 or not self.training:
            return semantic_tokens
        keep = torch.rand((semantic_tokens.shape[0],), device=semantic_tokens.device) >= float(dropout_prob)
        if not torch.any(keep):
            keep[int(torch.randint(semantic_tokens.shape[0], (1,), device=semantic_tokens.device).item())] = True
        scale = keep.to(dtype=semantic_tokens.dtype) / max(1.0 - float(dropout_prob), self.config.epsilon_a)
        return semantic_tokens * scale[:, None]

    def _semantic_prefix_tokens(
        self,
        semantic: _SemanticContext,
        *,
        dropout_prob: float,
    ) -> torch.Tensor:
        return self._semantic_memory(semantic.prefix_tokens, dropout_prob=dropout_prob)

    def _build_public_read_memory(self, token_field: PicfTokenFieldState) -> torch.Tensor:
        pieces = []
        if token_field.fused_tokens.shape[0] > 0:
            pieces.append(token_field.fused_tokens)
        if token_field.visual_tokens.shape[0] > 0:
            pieces.append(token_field.visual_tokens)
        if pieces:
            return torch.cat(pieces, dim=0)
        return torch.zeros((0, self.config.hidden_dim), device=self.device, dtype=self.dtype)

    def _build_task_readout(
        self,
        token_field: PicfTokenFieldState,
        dense_memory: _StepDenseMemory,
        semantic: _SemanticContext,
        proprio_token: torch.Tensor,
    ) -> PicfTaskReadoutState:
        del proprio_token
        query_tokens = torch.cat(
            [
                self.task_query_tokens.to(device=self.device, dtype=self.dtype),
                self.task_global_query_tokens.to(device=self.device, dtype=self.dtype),
                self.task_instruction_query_tokens.to(device=self.device, dtype=self.dtype),
            ],
            dim=0,
        )
        queries = query_tokens[None, :]
        semantic_attention = None
        if semantic.tokens.shape[0] > 0:
            queries, semantic_attention = self.task_query_conditioner(queries, semantic.tokens[None, :])
        public_read_memory = self._build_public_read_memory(token_field)
        public_attention = torch.zeros((queries.shape[1], public_read_memory.shape[0]), device=self.device, dtype=self.dtype)
        fused_attention = torch.zeros((queries.shape[1], token_field.fused_tokens.shape[0]), device=self.device, dtype=self.dtype)
        visual_public_attention = torch.zeros((queries.shape[1], token_field.visual_tokens.shape[0]), device=self.device, dtype=self.dtype)
        point_public_attention = torch.zeros((queries.shape[1], token_field.point_tokens.shape[0]), device=self.device, dtype=self.dtype)
        tactile_count = 0 if token_field.tactile_tokens_active is None else token_field.tactile_tokens_active.shape[0]
        tactile_public_attention = torch.zeros((queries.shape[1], tactile_count), device=self.device, dtype=self.dtype)
        if public_read_memory.shape[0] > 0:
            queries, public_attention = self.task_public_reader(queries, public_read_memory[None, :])
            fused_count = token_field.fused_tokens.shape[0]
            visual_count = token_field.visual_tokens.shape[0]
            fused_attention = public_attention[:, :fused_count]
            visual_public_attention = public_attention[:, fused_count : fused_count + visual_count]
            point_count = token_field.point_tokens.shape[0]
            point_public_attention = fused_attention[:, :point_count]
            tactile_public_attention = fused_attention[:, point_count : point_count + tactile_count]

        visual_private_attention = torch.zeros((queries.shape[1], 0), device=self.device, dtype=self.dtype)
        if dense_memory.visual_payload.shape[0] > 0:
            visual_candidates, visual_bias = self._gather_topk_native_candidates(
                dense_memory.visual_payload,
                visual_public_attention,
                topk=self.config.task_visual_reread_topk,
            )
            queries, visual_private_attention = self.task_visual_reread(queries, visual_candidates, attn_bias=visual_bias)
            visual_private_attention = visual_private_attention[0]

        point_private_attention = torch.zeros((queries.shape[1], 0), device=self.device, dtype=self.dtype)
        if dense_memory.point_payload.shape[0] > 0 and token_field.point_tokens.shape[0] > 0:
            point_candidates, point_bias = self._gather_topk_native_candidates(
                dense_memory.point_payload,
                point_public_attention,
                topk=self.config.task_point_reread_topk,
            )
            queries, point_private_attention = self.task_point_reread(queries, point_candidates, attn_bias=point_bias)
            point_private_attention = point_private_attention[0]

        tactile_private_attention = torch.zeros((queries.shape[1], 0), device=self.device, dtype=self.dtype)
        if dense_memory.tactile_group_tokens and tactile_count > 0 and token_field.tactile_group_ids is not None:
            group_weights = torch.zeros(
                (queries.shape[1], len(dense_memory.tactile_group_tokens)),
                device=self.device,
                dtype=self.dtype,
            )
            group_weights.scatter_add_(
                1,
                token_field.tactile_group_ids[None, :].expand(queries.shape[1], -1),
                tactile_public_attention,
            )
            tactile_candidates, tactile_bias = self._gather_tactile_group_candidates(
                dense_memory.tactile_group_tokens,
                group_weights,
                top_groups=self.config.task_tactile_reread_groups,
            )
            queries, tactile_private_attention = self.task_tactile_reread(queries, tactile_candidates, attn_bias=tactile_bias)
            tactile_private_attention = tactile_private_attention[0]

        task_tokens = self.task_self(queries)[0]
        local_count = int(self.config.task_local_queries)
        global_count = int(self.config.task_global_queries)
        instruction_count = int(self.config.task_instruction_queries)
        local_tokens = task_tokens[:local_count]
        global_tokens = task_tokens[local_count : local_count + global_count]
        instruction_tokens = task_tokens[local_count + global_count : local_count + global_count + instruction_count]

        if token_field.point_tokens.shape[0] > 0 and local_count > 0:
            point_weights = point_public_attention[:local_count]
            denom = torch.clamp(point_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            point_weights = point_weights / denom
            x = point_weights @ token_field.point_positions
            S = _weighted_cov(token_field.point_positions, point_weights, x, self.config)
            a = _extent_from_cov(S, self.config)
            local_tokens = local_tokens + self.task_geom_proj(_geometry_pe(x, a, S, self.config))
        else:
            point_weights = torch.zeros((local_count, 0), device=self.device, dtype=self.dtype)
            x = torch.zeros((local_count, 3), device=self.device, dtype=self.dtype)
            S = _diag_embed(torch.full((local_count, 3), self.config.epsilon_s, device=self.device, dtype=self.dtype))
            a = _to_tensor(self.config.a_min_m, device=self.device, dtype=self.dtype)[None, :].expand(local_count, -1)

        conditioned_queries = torch.cat([local_tokens, global_tokens, instruction_tokens], dim=0)
        global_token = (
            global_tokens.mean(dim=0)
            if global_tokens.shape[0] > 0
            else torch.zeros((self.config.hidden_dim,), device=self.device, dtype=self.dtype)
        )
        return PicfTaskReadoutState(
            conditioned_queries=conditioned_queries,
            local_tokens=local_tokens,
            global_token=global_token,
            instruction_tokens=instruction_tokens,
            point_weights=point_weights,
            x=x,
            S=S,
            a=a,
            semantic_attention=semantic_attention,
            public_attention=public_attention,
            visual_public_attention=visual_public_attention,
            point_public_attention=point_public_attention,
            tactile_public_attention=tactile_public_attention,
            visual_private_attention=visual_private_attention,
            tactile_private_attention=tactile_private_attention,
            point_private_attention=point_private_attention,
        )

    def _build_conditioned_control_state(
        self,
        posterior: PicfPosteriorAnchorState,
        innovation_token: torch.Tensor,
        proprio_token: torch.Tensor,
        task_readout: PicfTaskReadoutState,
    ) -> PicfConditionedControlState:
        control_posterior_tokens = self.posterior_to_control_proj(posterior.tokens)
        control_global_post = self.global_post_to_control_proj(posterior.global_post[None, :])
        control_innovation_token = self.innovation_to_control_proj(innovation_token[None, :])
        control_proprio_token = self.proprio_to_control_proj(proprio_token[None, :])
        task_local_tokens = self.task_to_control_proj(task_readout.local_tokens)
        task_global_token = self.task_global_to_control_proj(task_readout.global_token[None, :])
        instruction_tokens = (
            self.instruction_to_control_proj(task_readout.instruction_tokens)
            if task_readout.instruction_tokens.shape[0] > 0
            else torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype)
        )
        base_tokens = torch.cat(
            [
                _add_role_embedding(control_posterior_tokens, self.control_role_embedding, 0),
                _add_role_embedding(control_global_post, self.control_role_embedding, 1),
                _add_role_embedding(control_innovation_token, self.control_role_embedding, 2),
                _add_role_embedding(control_proprio_token, self.control_role_embedding, 3),
            ],
            dim=0,
        )
        task_tokens = torch.cat(
            [
                _add_role_embedding(task_local_tokens, self.control_role_embedding, 4),
                _add_role_embedding(task_global_token, self.control_role_embedding, 5),
                _add_role_embedding(instruction_tokens, self.control_role_embedding, 6),
            ],
            dim=0,
        )
        conditioned_control_queries = self.control_query_tokens.to(device=self.device, dtype=self.dtype)
        control_prefix = torch.cat(
            [
                base_tokens,
                task_tokens,
                _add_role_embedding(conditioned_control_queries, self.control_role_embedding, 7),
            ],
            dim=0,
        )
        control_tokens = self.control_world(control_prefix[None, :])[0]
        query_state = _mean_query_state(control_tokens, num_query_tokens=self.control_query_tokens.shape[0])
        pi_queries = self.pi_prefix_query_tokens.to(device=self.device, dtype=self.dtype)[None, :]
        pi_prefix_tokens, _ = self.pi_prefix_reader(pi_queries, control_tokens[None, :])
        future_queries = self.predictive_query_tokens.to(device=self.device, dtype=self.dtype)[None, :]
        future_condition_tokens, _ = self.future_condition_reader(future_queries, control_tokens[None, :])
        return PicfConditionedControlState(
            base_tokens=base_tokens,
            task_tokens=task_tokens,
            tokens=control_tokens,
            query_state=query_state,
            pi_prefix_tokens=pi_prefix_tokens[0],
            future_condition_tokens=future_condition_tokens[0],
        )

    def _build_physical_predictive_basis(
        self,
        posterior: PicfPosteriorAnchorState,
        *,
        proprio_token: torch.Tensor,
        executed_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, PicfPredictionCache]:
        pred_tail = torch.stack(
            [
                posterior.global_post,
                proprio_token,
                self.action_cond_proj(executed_action[None, :])[0],
            ],
            dim=0,
        )
        pred_world_tokens = torch.cat(
            [
                _add_role_embedding(posterior.tokens, self.predictive_physical_role_embedding, 0),
                _add_role_embedding(pred_tail[0:1], self.predictive_physical_role_embedding, 1),
                _add_role_embedding(pred_tail[1:2], self.predictive_physical_role_embedding, 2),
                _add_role_embedding(pred_tail[2:3], self.predictive_physical_role_embedding, 3),
            ],
            dim=0,
        )
        physical_pred_tokens = self.predictive_world(pred_world_tokens[None, :])[0]
        physical_global_pred = self.predictive_pool(physical_pred_tokens[None, :])[0]
        physical_prediction_cache = self._prediction_cache_from_global(physical_global_pred)
        return physical_pred_tokens, physical_global_pred, physical_prediction_cache

    def _build_conditioned_predictive_cache(
        self,
        physical_pred_tokens: torch.Tensor,
        future_condition_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, PicfPredictionCache]:
        conditioned_physical_pred_tokens = self.physical_pred_to_conditioned_proj(physical_pred_tokens)
        pred_conditioned_tokens = torch.cat(
            [
                _add_role_embedding(conditioned_physical_pred_tokens, self.predictive_conditioned_role_embedding, 0),
                _add_role_embedding(future_condition_tokens, self.predictive_conditioned_role_embedding, 1),
            ],
            dim=0,
        )
        pred_tokens = self.predictive_semantic_world(pred_conditioned_tokens[None, :])[0]
        predictive_query_state = _mean_query_state(pred_tokens, num_query_tokens=max(future_condition_tokens.shape[0], 1))
        global_pred = self.predictive_state_proj(predictive_query_state)
        prediction_cache = self._prediction_cache_from_global(global_pred)
        return predictive_query_state, global_pred, prediction_cache

    def _prediction_cache_from_global(self, global_state: torch.Tensor) -> PicfPredictionCache:
        return PicfPredictionCache(
            visual_latent=self.visual_latent_head(global_state),
            visual_real=self.visual_real_head(global_state) if self.config.visual_real_enabled else None,
            tactile_real=self.tactile_real_head(global_state),
            point_real=self.point_real_head(global_state),
            availability=torch.as_tensor(
                [
                    1.0,
                    1.0 if self.config.visual_real_enabled else 0.0,
                    1.0,
                    1.0,
                ],
                device=self.device,
                dtype=self.dtype,
            ),
        )

    def _visual_latent_target(self, dense_memory: _StepDenseMemory) -> torch.Tensor | None:
        if dense_memory.visual_payload.numel() == 0:
            return None
        queries = self.visual_latent_queries.to(device=self.device, dtype=self.dtype)[None, :]
        payload = dense_memory.visual_payload[None, :]
        visual_latent, _ = self.visual_native_reread(queries, payload)
        return visual_latent[0].reshape(-1)

    def _tactile_latent_target(self, dense_memory: _StepDenseMemory) -> torch.Tensor | None:
        if not dense_memory.tactile_group_tokens:
            return None
        payload = torch.cat(tuple(token for token in dense_memory.tactile_group_tokens if token.numel() > 0), dim=0)
        if payload.numel() == 0:
            return None
        queries = self.tactile_latent_queries.to(device=self.device, dtype=self.dtype)[None, :]
        tactile_latent, _ = self.tactile_native_reread(queries, payload[None, :])
        return tactile_latent[0].reshape(-1)

    def _point_latent_target(self, dense_memory: _StepDenseMemory) -> torch.Tensor | None:
        if dense_memory.point_payload.numel() == 0:
            return None
        queries = self.point_latent_queries.to(device=self.device, dtype=self.dtype)[None, :]
        payload = dense_memory.point_payload[None, :]
        point_latent, _ = self.point_native_reread(queries, payload)
        return point_latent[0].reshape(-1)

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        clip = getattr(self.config, "action_output_clip", None)
        if clip is None:
            return action
        clip_value = float(clip)
        if clip_value <= 0.0:
            return action
        return torch.clamp(action, min=-clip_value, max=clip_value)

    def _encode_context_tokens(self, observation: PicfObservation, meta: RuntimeMeta, previous: PicfCoreState | None) -> torch.Tensor:
        proprio = np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1)
        action = self._previous_action(previous)
        timing = np.asarray(
            [
                float(observation.timestamp_s - meta.t_v_last) if meta.visual_available else self.config.visual_stale_s,
                float(observation.timestamp_s - meta.t_p_last) if meta.point_contract_ok else self.config.visual_stale_s,
                float(observation.timestamp_s - meta.t_t_last) if meta.tactile_available else self.config.tactile_stale_s,
                float(meta.visual_available),
                float(meta.point_contract_ok),
                float(meta.tactile_available),
                float(meta.sync_valid),
            ],
            dtype=np.float32,
        )
        tokens = [
            self.proprio_context_proj(_to_tensor(proprio[None, :], device=self.device, dtype=self.dtype)),
            self.action_context_proj(action[None, :]),
            self.timing_context_proj(_to_tensor(timing[None, :], device=self.device, dtype=self.dtype)),
        ]
        return torch.cat(tokens, dim=0)

    def _empty_projective_geometry(self) -> PicfProjectiveGeometryState:
        return PicfProjectiveGeometryState(
            point_proj_grid_norm=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
            point_proj_grid_index=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
            point_visibility=torch.zeros((0,), device=self.device, dtype=self.dtype),
            point_depth=torch.zeros((0,), device=self.device, dtype=self.dtype),
            point_depth_sample=torch.zeros((0,), device=self.device, dtype=self.dtype),
            point_depth_valid=torch.zeros((0,), device=self.device, dtype=torch.bool),
            visual_grid_norm=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
            visual_grid_index=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
            visual_pixel_centers=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
            visual_ray_world=torch.zeros((0, 3), device=self.device, dtype=self.dtype),
            camera_origin_world=torch.zeros((3,), device=self.device, dtype=self.dtype),
            projective_compatibility=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
            projective_candidate_mask=torch.zeros((0, 0), device=self.device, dtype=torch.bool),
            projective_attention_bias=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
        )

    def _projective_attention_bias(
        self,
        *,
        point_positions: torch.Tensor,
        point_proj_grid_index: torch.Tensor,
        point_depth: torch.Tensor,
        point_visibility: torch.Tensor,
        visual_grid_index: torch.Tensor,
        visual_ray_world: torch.Tensor,
        camera_origin_world: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        point_count = point_positions.shape[0]
        visual_count = visual_grid_index.shape[0]
        bias = torch.zeros((point_count, visual_count), device=self.device, dtype=self.dtype)
        if point_count == 0 or visual_count == 0 or not bool(candidate_mask.any()):
            return bias
        expected = (point_count, visual_count)
        actual = tuple(int(dim) for dim in candidate_mask.shape)
        if actual != expected:
            raise RuntimeError(
                "PICF projective-bias contract violated: candidate_mask shape mismatch. "
                f"expected={expected} got={actual}"
            )
        delta = point_proj_grid_index[:, None, :] - visual_grid_index[None, :, :]
        point_rays = _normalize_tensor(point_positions - camera_origin_world[None, :], eps=self.config.epsilon_residual)
        ray_align = torch.sum(point_rays[:, None, :] * visual_ray_world[None, :, :], dim=-1, keepdim=True)
        visibility = point_visibility[:, None, None]
        log_depth = torch.log(torch.clamp(point_depth[:, None, None], min=self.config.z_min_m)).expand(-1, visual_count, -1)
        features = torch.cat([delta, log_depth, ray_align, visibility.expand(-1, visual_count, -1)], dim=-1)
        candidate_weight = candidate_mask.to(dtype=self.dtype)
        flat_features = features.reshape(point_count * visual_count, 1, features.shape[-1])
        flat_values = self.projective_bias_head(flat_features)[:, 0, 0]
        dense_bias = (self.config.projective_bias_scale * torch.tanh(flat_values)).reshape(point_count, visual_count)
        return dense_bias * candidate_weight

    def _fusion_projective_bias(
        self,
        *,
        projective_geometry: PicfProjectiveGeometryState,
        point_count: int,
        visual_count: int,
        total_tokens: int,
    ) -> torch.Tensor | None:
        if point_count == 0 or visual_count == 0 or total_tokens == 0:
            return None
        pv_bias = projective_geometry.projective_attention_bias
        if pv_bias.numel() == 0:
            return None
        full_bias = torch.zeros((total_tokens, total_tokens), device=self.device, dtype=self.dtype)
        visual_slice = slice(point_count, point_count + visual_count)
        full_bias[:point_count, visual_slice] = pv_bias
        full_bias[visual_slice, :point_count] = pv_bias.T
        return full_bias

    def _build_projective_geometry(
        self,
        *,
        observation: PicfObservation,
        point_positions: torch.Tensor,
        visual_hw: tuple[int, int] | None,
    ) -> PicfProjectiveGeometryState:
        point_count = int(point_positions.shape[0])
        visual_count = int((visual_hw[0] * visual_hw[1]) if visual_hw is not None else 0)
        if visual_hw is None:
            return self._empty_projective_geometry()
        source_h, source_w = int(observation.rgb_static.shape[0]), int(observation.rgb_static.shape[1])
        grid_h, grid_w = visual_hw
        ys_idx, xs_idx = torch.meshgrid(
            torch.arange(grid_h, device=self.device, dtype=self.dtype),
            torch.arange(grid_w, device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        visual_grid_index = torch.stack([xs_idx, ys_idx], dim=-1).reshape(-1, 2)
        visual_grid_norm = _grid_index_to_norm(visual_grid_index, height=grid_h, width=grid_w)
        pixel_x = xs_idx * (float(source_w - 1) / max(grid_w - 1, 1))
        pixel_y = ys_idx * (float(source_h - 1) / max(grid_h - 1, 1))
        visual_pixel_centers = torch.stack([pixel_x, pixel_y], dim=-1).reshape(-1, 2)
        if self.camera_model is None:
            return PicfProjectiveGeometryState(
                point_proj_grid_norm=torch.zeros((point_count, 2), device=self.device, dtype=self.dtype),
                point_proj_grid_index=torch.zeros((point_count, 2), device=self.device, dtype=self.dtype),
                point_visibility=torch.zeros((point_count,), device=self.device, dtype=self.dtype),
                point_depth=torch.full((point_count,), self.config.z_min_m, device=self.device, dtype=self.dtype),
                point_depth_sample=torch.zeros((point_count,), device=self.device, dtype=self.dtype),
                point_depth_valid=torch.zeros((point_count,), device=self.device, dtype=torch.bool),
                visual_grid_norm=visual_grid_norm,
                visual_grid_index=visual_grid_index,
                visual_pixel_centers=visual_pixel_centers,
                visual_ray_world=torch.zeros((visual_count, 3), device=self.device, dtype=self.dtype),
                camera_origin_world=torch.zeros((3,), device=self.device, dtype=self.dtype),
                projective_compatibility=torch.zeros((point_count, visual_count), device=self.device, dtype=self.dtype),
                projective_candidate_mask=torch.zeros((point_count, visual_count), device=self.device, dtype=torch.bool),
                projective_attention_bias=torch.zeros((point_count, visual_count), device=self.device, dtype=self.dtype),
            )

        K = _to_tensor(self.camera_model.K, device=self.device, dtype=self.dtype)
        K_inv = torch.linalg.inv(K)
        rays_cam_h = torch.cat(
            [
                visual_pixel_centers,
                torch.ones((visual_pixel_centers.shape[0], 1), device=self.device, dtype=self.dtype),
            ],
            dim=-1,
        )
        rays_cam = (K_inv @ rays_cam_h.T).T
        rays_cam = _normalize_tensor(rays_cam, eps=self.config.epsilon_residual)

        W_T_C = _to_tensor(self.camera_model.W_T_C, device=self.device, dtype=self.dtype)
        C_T_W = _to_tensor(self.camera_model.C_T_W, device=self.device, dtype=self.dtype)
        camera_origin_world = W_T_C[:3, 3]
        visual_ray_world = (W_T_C[:3, :3] @ rays_cam.T).T
        visual_ray_world = _normalize_tensor(visual_ray_world, eps=self.config.epsilon_residual)

        if point_positions.shape[0] == 0:
            return PicfProjectiveGeometryState(
                point_proj_grid_norm=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
                point_proj_grid_index=torch.zeros((0, 2), device=self.device, dtype=self.dtype),
                point_visibility=torch.zeros((0,), device=self.device, dtype=self.dtype),
                point_depth=torch.zeros((0,), device=self.device, dtype=self.dtype),
                point_depth_sample=torch.zeros((0,), device=self.device, dtype=self.dtype),
                point_depth_valid=torch.zeros((0,), device=self.device, dtype=torch.bool),
                visual_grid_norm=visual_grid_norm,
                visual_grid_index=visual_grid_index,
                visual_pixel_centers=visual_pixel_centers,
                visual_ray_world=visual_ray_world,
                camera_origin_world=camera_origin_world,
                projective_compatibility=torch.zeros((0, visual_grid_index.shape[0]), device=self.device, dtype=self.dtype),
                projective_candidate_mask=torch.zeros((0, visual_grid_index.shape[0]), device=self.device, dtype=torch.bool),
                projective_attention_bias=torch.zeros((0, visual_grid_index.shape[0]), device=self.device, dtype=self.dtype),
            )

        homo = torch.cat([point_positions, torch.ones((point_positions.shape[0], 1), device=self.device, dtype=self.dtype)], dim=-1)
        points_cam = (C_T_W @ homo.T).T[:, :3]
        z_raw = points_cam[:, 2]
        z = torch.clamp(z_raw, min=self.config.z_min_m, max=self.config.z_max_m)
        uv = torch.zeros((point_positions.shape[0], 2), device=self.device, dtype=self.dtype)
        uv[:, 0] = (self.camera_model.fx * points_cam[:, 0] / torch.clamp(z_raw, min=self.config.z_min_m)) + self.camera_model.cx
        uv[:, 1] = (self.camera_model.fy * points_cam[:, 1] / torch.clamp(z_raw, min=self.config.z_min_m)) + self.camera_model.cy
        visibility = (
            (z_raw > 0.0)
            & torch.isfinite(uv[:, 0])
            & torch.isfinite(uv[:, 1])
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= float(source_w - 1))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= float(source_h - 1))
        )
        point_proj_grid_index = torch.zeros((point_positions.shape[0], 2), device=self.device, dtype=self.dtype)
        point_proj_grid_index[:, 0] = uv[:, 0] * (float(grid_w - 1) / max(source_w - 1, 1))
        point_proj_grid_index[:, 1] = uv[:, 1] * (float(grid_h - 1) / max(source_h - 1, 1))
        point_proj_grid_norm = _grid_index_to_norm(point_proj_grid_index, height=grid_h, width=grid_w)

        depth_sample = torch.zeros((point_positions.shape[0],), device=self.device, dtype=self.dtype)
        depth_valid = torch.zeros((point_positions.shape[0],), device=self.device, dtype=torch.bool)
        depth_image = np.asarray(observation.depth_static, dtype=np.float32)
        if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
            depth_image = depth_image[..., 0]
        if depth_image.ndim == 2 and depth_image.size > 0:
            depth_t = _to_tensor(depth_image, device=self.device, dtype=self.dtype)
            depth_sample, depth_valid = _bilinear_sample_depth(depth_t, uv)

        sigma_proj = max(float(self.config.sigma_proj_patches), 1e-6)
        delta = point_proj_grid_index[:, None, :] - visual_grid_index[None, :, :]
        proj_score = torch.exp(-torch.sum(delta**2, dim=-1) / (2.0 * (sigma_proj**2)))
        depth_factor = torch.ones_like(proj_score)
        valid_depth_rows = depth_valid & visibility
        if valid_depth_rows.any():
            depth_row_factor = torch.exp(
                -(((z - depth_sample)[:, None]) ** 2) / (2.0 * (self.config.tau_proj_depth_m**2))
            )
            depth_factor = torch.where(valid_depth_rows[:, None], depth_row_factor.expand_as(depth_factor), depth_factor)
        projective_compatibility = proj_score * depth_factor * visibility[:, None].to(dtype=self.dtype)
        projective_compatibility = torch.nan_to_num(projective_compatibility, nan=0.0, posinf=1.0, neginf=0.0)
        projective_compatibility = torch.clamp(projective_compatibility, min=0.0, max=1.0)
        sparse_neighborhood = _sparse_projective_neighborhood_mask(
            point_proj_grid_index,
            visibility,
            grid_h=grid_h,
            grid_w=grid_w,
            radius_patches=_projective_candidate_radius_patches(
                sigma_proj=self.config.sigma_proj_patches,
                tau_proj=self.config.tau_proj,
            ),
        )
        candidate_mask = sparse_neighborhood & (projective_compatibility > self.config.tau_proj)
        projective_attention_bias = self._projective_attention_bias(
            point_positions=point_positions,
            point_proj_grid_index=point_proj_grid_index,
            point_depth=z,
            point_visibility=visibility.to(dtype=self.dtype),
            visual_grid_index=visual_grid_index,
            visual_ray_world=visual_ray_world,
            camera_origin_world=camera_origin_world,
            candidate_mask=candidate_mask,
        )
        return PicfProjectiveGeometryState(
            point_proj_grid_norm=point_proj_grid_norm,
            point_proj_grid_index=point_proj_grid_index,
            point_visibility=visibility.to(dtype=self.dtype),
            point_depth=z,
            point_depth_sample=depth_sample,
            point_depth_valid=depth_valid,
            visual_grid_norm=visual_grid_norm,
            visual_grid_index=visual_grid_index,
            visual_pixel_centers=visual_pixel_centers,
            visual_ray_world=visual_ray_world,
            camera_origin_world=camera_origin_world,
            projective_compatibility=projective_compatibility,
            projective_candidate_mask=candidate_mask,
            projective_attention_bias=projective_attention_bias,
        )

    def _point_projection_features(self, projective_geometry: PicfProjectiveGeometryState, *, source_hw: tuple[int, int]) -> torch.Tensor:
        point_count = projective_geometry.point_proj_grid_index.shape[0]
        if point_count == 0:
            proj_dim = self.null_proj_coarse.numel() + self.null_proj_fine.numel() + 4
            return torch.zeros((0, proj_dim), device=self.device, dtype=self.dtype)
        visible = projective_geometry.point_visibility[:, None]
        coarse = _point_proj_fourier(projective_geometry.point_proj_grid_index, bands=4)
        fine = _point_proj_fourier(projective_geometry.point_proj_grid_index, bands=8)
        coarse = (visible * coarse) + ((1.0 - visible) * self.null_proj_coarse[None, :])
        fine = (visible * fine) + ((1.0 - visible) * self.null_proj_fine[None, :])
        log_depth = visible * torch.log(torch.clamp(projective_geometry.point_depth[:, None], min=self.config.z_min_m))
        if self.camera_model is None:
            intrinsics = torch.zeros((point_count, 2), device=self.device, dtype=self.dtype)
        else:
            source_h, source_w = source_hw
            intrinsics = torch.as_tensor(
                [
                    math.log(max(float(self.camera_model.fx) / max(source_w, 1), 1e-6)),
                    math.log(max(float(self.camera_model.fy) / max(source_h, 1), 1e-6)),
                ],
                device=self.device,
                dtype=self.dtype,
            )[None, :].expand(point_count, -1)
        return torch.cat([coarse, fine, log_depth, visible, intrinsics], dim=-1)

    def _visual_ray_features(self, projective_geometry: PicfProjectiveGeometryState, *, source_hw: tuple[int, int]) -> torch.Tensor:
        visual_count = projective_geometry.visual_grid_index.shape[0]
        if visual_count == 0:
            return torch.zeros((0, 9), device=self.device, dtype=self.dtype)
        if self.camera_model is None:
            intrinsics = torch.zeros((visual_count, 2), device=self.device, dtype=self.dtype)
        else:
            intrinsics = torch.as_tensor(
                [
                    math.log(max(float(self.camera_model.fx) / max(source_hw[1], 1), 1e-6)),
                    math.log(max(float(self.camera_model.fy) / max(source_hw[0], 1), 1e-6)),
                ],
                device=self.device,
                dtype=self.dtype,
            )[None, :].expand(visual_count, -1)
        origin = projective_geometry.camera_origin_world[None, :].expand(visual_count, -1)
        return torch.cat([projective_geometry.visual_grid_index, projective_geometry.visual_ray_world, origin, intrinsics], dim=-1)

    def _build_token_field(
        self,
        observation: PicfObservation,
        frame_context: PointFrameContext | None,
        point_features: torch.Tensor,
        visual_map: torch.Tensor | None,
        tactile_bundle: AnyTouchFeatureBundle | None,
        meta: RuntimeMeta,
        previous: PicfCoreState | None,
    ) -> tuple[PicfTokenFieldState, _StepDenseMemory]:
        hidden_dim = self.config.hidden_dim
        point_tokens = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        point_positions = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        point_align_embeddings = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        visual_align_embeddings = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        tactile_align_embeddings = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        tactile_tokens_all = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        tactile_tokens_active = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        tactile_positions_world = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        tactile_normals_world = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        tactile_contact_gate = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_contact_prob = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_anchor_mask = torch.zeros((0,), device=self.device, dtype=torch.bool)
        tactile_contact_score = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_contact_score_ema = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_group_ids = torch.zeros((0,), device=self.device, dtype=torch.long)
        visual_payload = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        point_payload = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        tactile_group_tokens: tuple[torch.Tensor, ...] = ()
        visual_hw: tuple[int, int] | None = None
        if visual_map is not None and visual_map.numel() > 0:
            visual_hw = (int(visual_map.shape[0]), int(visual_map.shape[1]))
        projective_geometry = self._build_projective_geometry(
            observation=observation,
            point_positions=_to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype) if frame_context is not None else point_positions,
            visual_hw=visual_hw,
        )
        if frame_context is not None:
            point_positions = _to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype)
            proj_features = self._point_projection_features(
                projective_geometry,
                source_hw=(int(observation.rgb_static.shape[0]), int(observation.rgb_static.shape[1])),
            )
            point_token_in = torch.cat(
                [
                    point_features,
                    _to_tensor(frame_context.colors, device=self.device, dtype=self.dtype),
                    _point_pe(point_positions, self.config),
                    proj_features,
                ],
                dim=-1,
            )
            point_payload = torch.cat(
                [
                    point_positions,
                    _to_tensor(frame_context.colors, device=self.device, dtype=self.dtype),
                    _point_pe(point_positions, self.config),
                    proj_features,
                ],
                dim=-1,
            )
            point_tokens = self.point_token_proj(point_token_in) + self.modality_embedding.weight[0][None, :]
            point_align_embeddings = _normalize_tensor(self.point_align_proj(point_tokens), eps=self.config.epsilon_residual)

        visual_tokens = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        if visual_map is not None and visual_map.numel() > 0:
            h, w, _ = visual_map.shape
            grid = projective_geometry.visual_grid_index
            if self.camera_model is None:
                cam_pose = torch.zeros((1, 9), device=self.device, dtype=self.dtype)
            else:
                cam = _to_tensor(self.camera_model.W_T_C, device=self.device, dtype=self.dtype)
                cam_pose = torch.cat([cam[:3, 3], rot6d(cam[:3, :3])], dim=-1)[None, :]
            flat_map = visual_map.reshape(-1, visual_map.shape[-1])
            visual_payload = flat_map
            ray_features = self._visual_ray_features(projective_geometry, source_hw=(int(observation.rgb_static.shape[0]), int(observation.rgb_static.shape[1])))
            visual_in = torch.cat([flat_map, grid, cam_pose.expand(flat_map.shape[0], -1), ray_features], dim=-1)
            visual_tokens = self.visual_token_proj(visual_in) + self.modality_embedding.weight[1][None, :]
            visual_align_embeddings = _normalize_tensor(self.visual_align_proj(visual_tokens), eps=self.config.epsilon_residual)

        tactile_tokens = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        if tactile_bundle is not None and tactile_bundle.sensors:
            encoded = []
            positions = []
            normals = []
            dense_tokens_all: list[torch.Tensor] = []
            sensor_names = sorted(tactile_bundle.sensors)
            for sensor_name in sensor_names:
                sensor = tactile_bundle.sensors[sensor_name]
                sensor_pose_world = _to_tensor(observation.G_t, device=self.device, dtype=self.dtype) @ sensor.T_sens_to_wrist.to(device=self.device, dtype=self.dtype)
                sensor_in = torch.cat(
                    [
                        sensor.pooled_feature.to(device=self.device, dtype=self.dtype),
                        tactile_bundle.global_feature.to(device=self.device, dtype=self.dtype),
                        sensor_pose_world[:3, 3],
                        rot6d(sensor_pose_world[:3, :3]),
                    ],
                    dim=-1,
                )
                encoded.append(sensor_in)
                positions.append(sensor_pose_world[:3, 3])
                normals.append(_normalize_tensor(sensor_pose_world[:3, 0], eps=self.config.epsilon_residual))
                dense_tokens_all.append(sensor.tokens.to(device=self.device, dtype=self.dtype))
            tactile_tokens_all = self.tactile_token_proj(torch.stack(encoded, dim=0)) + self.modality_embedding.weight[2][None, :]
            tactile_align_embeddings = _normalize_tensor(self.tactile_align_proj(tactile_tokens_all), eps=self.config.epsilon_residual)
            tactile_positions_world = torch.stack(positions, dim=0)
            tactile_normals_world = torch.stack(normals, dim=0)
            has_explicit_contact = (
                observation.force_vec is not None
                or observation.indent_depth_m is not None
                or observation.tactile_pressure is not None
            )
            if has_explicit_contact:
                contact_value = float(
                    explicit_contact_observation(
                        force_vec=_to_tensor(
                            observation.force_vec if observation.force_vec is not None else np.zeros((3,), dtype=np.float32),
                            device=self.device,
                            dtype=self.dtype,
                        ),
                        indent_depth_m=observation.indent_depth_m,
                        tactile_pressure=observation.tactile_pressure,
                        tau_force_n=self.config.tau_force_n,
                        tau_indent_m=self.config.tau_indent_m,
                        tau_tactile_pressure=self.config.tau_tactile_pressure,
                    )
                    or 0.0
                )
                tactile_contact_gate = torch.full(
                    (tactile_tokens_all.shape[0],),
                    contact_value,
                    device=self.device,
                    dtype=self.dtype,
                )
                tactile_contact_prob = tactile_contact_gate
                tactile_anchor_mask = tactile_contact_prob >= float(self.config.tactile_anchor_prob_on)
                tactile_contact_score = tactile_contact_prob
                tactile_contact_score_ema = tactile_contact_prob
            else:
                contact_scores = torch.as_tensor(
                    [
                        float(
                            max(
                                0.0,
                                tactile_bundle.sensors[sensor_name].contact_score
                                if getattr(tactile_bundle.sensors[sensor_name], "contact_score", 0.0) > 0.0
                                else tactile_bundle.sensors[sensor_name].pseudo_contact_score,
                            )
                        )
                        for sensor_name in sensor_names
                    ],
                    device=self.device,
                    dtype=self.dtype,
                )
                prev_score_ema = None if previous is None else previous.token_field.tactile_contact_score_ema
                prev_active = None
                if previous is not None:
                    prev_gate = previous.token_field.tactile_contact_gate
                    if prev_gate is not None and prev_gate.shape == contact_scores.shape:
                        # Hysteresis should resume from the previous contact-active
                        # state, not the stricter anchor/fusion gate.
                        prev_active = prev_gate > 0.0
                    else:
                        prev_anchor = previous.token_field.tactile_anchor_mask
                        if prev_anchor is not None and prev_anchor.shape == contact_scores.shape:
                            prev_active = prev_anchor
                tactile_contact_score_ema, tactile_contact_prob, tactile_contact_active = contact_prob_with_hysteresis(
                    contact_scores,
                    tau_on=float(max(self.config.tactile_contact_tau_on, self.config.tau_tactile_pseudo_contact)),
                    tau_off=float(max(self.config.tactile_contact_tau_off, 0.0)),
                    temperature=float(self.config.tactile_contact_temperature),
                    ema_beta=float(self.config.tactile_contact_ema_beta),
                    previous_score_ema=prev_score_ema,
                    previous_active=prev_active,
                )
                tactile_contact_score = contact_scores
                tactile_contact_gate = tactile_contact_active.to(dtype=self.dtype)
                tactile_anchor_mask = tactile_contact_prob >= float(self.config.tactile_anchor_prob_on)
            active_indices = torch.nonzero(tactile_anchor_mask, as_tuple=False).squeeze(-1)
            tactile_group_tokens = tuple(dense_tokens_all[int(index.item())] for index in active_indices)
            if active_indices.numel() > 0:
                proposal_tokens = []
                proposal_align = []
                proposal_positions = []
                proposal_normals = []
                proposal_group_ids = []
                for group_local_index, sensor_index in enumerate(active_indices.tolist()):
                    dense_group = dense_tokens_all[sensor_index]
                    base_token = tactile_tokens_all[sensor_index]
                    route_queries = (
                        self.tactile_group_route_queries.to(device=self.device, dtype=self.dtype)[None, :]
                        + base_token[None, None, :]
                    )
                    route_tokens, _ = self.tactile_route_reread(route_queries, dense_group[None, :])
                    route_tokens = route_tokens[0]
                    proposal_tokens.append(route_tokens)
                    proposal_align.append(_normalize_tensor(self.tactile_align_proj(route_tokens), eps=self.config.epsilon_residual))
                    proposal_positions.append(tactile_positions_world[sensor_index][None, :].expand(route_tokens.shape[0], -1))
                    proposal_normals.append(tactile_normals_world[sensor_index][None, :].expand(route_tokens.shape[0], -1))
                    proposal_group_ids.append(
                        torch.full(
                            (route_tokens.shape[0],),
                            int(group_local_index),
                            device=self.device,
                            dtype=torch.long,
                        )
                    )
                tactile_tokens_active = torch.cat(proposal_tokens, dim=0)
                tactile_align_embeddings = torch.cat(proposal_align, dim=0)
                tactile_positions_world = torch.cat(proposal_positions, dim=0)
                tactile_normals_world = torch.cat(proposal_normals, dim=0)
                tactile_group_ids = torch.cat(proposal_group_ids, dim=0)
            tactile_tokens = tactile_tokens_active

        context_tokens = self._encode_context_tokens(observation, meta, previous) + self.modality_embedding.weight[3][None, :]
        contact_context = self.contact_context_proj(
            summarize_contact_context(tactile_contact_prob, tactile_anchor_mask)[None, :]
        ) + self.modality_embedding.weight[3][None, :]
        context_tokens = torch.cat([context_tokens, contact_context], dim=0)
        all_tokens = torch.cat([point_tokens, tactile_tokens_active, context_tokens], dim=0)
        fusion_attention_mean = None
        if all_tokens.shape[0] > 0:
            fused, fusion_attention_mean = self.token_fusion(
                all_tokens[None, :],
                attn_bias=None,
                return_attention=True,
            )
            fused = fused[0]
        else:
            fused = all_tokens
        modality_ids = torch.cat(
            [
                torch.zeros((point_tokens.shape[0],), device=self.device, dtype=torch.long),
                torch.full((tactile_tokens_active.shape[0],), 2, device=self.device, dtype=torch.long),
                torch.full((context_tokens.shape[0],), 3, device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        token_field = PicfTokenFieldState(
            point_tokens=point_tokens,
            visual_tokens=visual_tokens,
            tactile_tokens=tactile_tokens,
            context_tokens=context_tokens,
            fused_tokens=fused,
            point_positions=point_positions,
            modality_ids=modality_ids,
            point_align_embeddings=point_align_embeddings,
            visual_align_embeddings=visual_align_embeddings,
            tactile_align_embeddings=tactile_align_embeddings,
            tactile_positions_world=tactile_positions_world,
            tactile_contact_gate=tactile_contact_gate,
            tactile_tokens_all=tactile_tokens_all,
            tactile_tokens_active=tactile_tokens_active,
            tactile_group_ids=tactile_group_ids,
            tactile_contact_prob=tactile_contact_prob,
            tactile_anchor_mask=tactile_anchor_mask,
            tactile_normals_world=tactile_normals_world,
            tactile_contact_score=tactile_contact_score,
            tactile_contact_score_ema=tactile_contact_score_ema,
            fusion_attention_mean=fusion_attention_mean,
            projective_geometry=projective_geometry,
        )
        dense_memory = _StepDenseMemory(
            point_payload=point_payload,
            visual_payload=visual_payload,
            tactile_group_tokens=tactile_group_tokens,
        )
        return token_field, dense_memory

    def _build_observation_anchors(
        self,
        token_field: PicfTokenFieldState,
        dense_memory: _StepDenseMemory | None = None,
    ) -> PicfObservationAnchorState:
        if dense_memory is None:
            dense_memory = _StepDenseMemory(
                point_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                visual_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                tactile_group_tokens=(),
            )
        n_obs = self.config.observation_anchors
        hidden_dim = self.config.hidden_dim
        point_count = token_field.point_tokens.shape[0]
        visual_count = dense_memory.visual_payload.shape[0]
        tactile_count = token_field.tactile_tokens.shape[0]
        seed_indices = torch.full((n_obs,), -1, device=self.device, dtype=torch.long)
        queries = torch.zeros((1, n_obs, hidden_dim), device=self.device, dtype=self.dtype)
        if point_count > 0:
            chosen = _fps_indices(token_field.point_positions, min(n_obs, point_count))
            if chosen.numel() > 0:
                min_idx = int(chosen.min().item())
                max_idx = int(chosen.max().item())
                if min_idx < 0 or max_idx >= point_count:
                    raise RuntimeError(
                        "PICF observation-anchor seed index out of bounds: "
                        f"valid=[0,{point_count - 1}] got min={min_idx} max={max_idx}"
                    )
            seed_indices[: chosen.shape[0]] = chosen
            queries[0, : chosen.shape[0]] = token_field.point_tokens[chosen]
        attn_public = torch.zeros((n_obs, token_field.fused_tokens.shape[0]), device=self.device, dtype=self.dtype)
        attn_visual = torch.zeros((n_obs, visual_count), device=self.device, dtype=self.dtype)
        for _ in range(max(self.config.query_rounds, 1)):
            if visual_count > 0:
                queries, visual_weights = self.visual_native_reread(queries, dense_memory.visual_payload[None, :])
                attn_visual = visual_weights[0]
            queries, attn_public = self.obs_reader(queries, token_field.fused_tokens[None, :])
        obs_tokens = self.obs_self(queries)[0]
        routing_mass_point = attn_public[:, :point_count]
        routing_mass_visual = attn_visual
        routing_mass_tactile_token = attn_public[:, point_count : point_count + tactile_count]
        if tactile_count > 0 and token_field.tactile_group_ids is not None and token_field.tactile_group_ids.numel() == tactile_count:
            tactile_group_count = len(dense_memory.tactile_group_tokens)
            routing_mass_tactile = torch.zeros((n_obs, tactile_group_count), device=self.device, dtype=self.dtype)
            routing_mass_tactile.scatter_add_(
                1,
                token_field.tactile_group_ids[None, :].expand(n_obs, -1),
                routing_mass_tactile_token,
            )
        else:
            routing_mass_tactile = routing_mass_tactile_token
        routing_support_point = routing_mass_point.sum(dim=0) if point_count > 0 else torch.zeros((0,), device=self.device, dtype=self.dtype)
        routing_support_visual = routing_mass_visual.sum(dim=0) if visual_count > 0 else torch.zeros((0,), device=self.device, dtype=self.dtype)
        routing_support_tactile = routing_mass_tactile.sum(dim=0) if tactile_count > 0 else torch.zeros((0,), device=self.device, dtype=self.dtype)
        routing_gate_point = (
            routing_support_point / torch.clamp(routing_support_point + self.config.tau_route_p, min=self.config.epsilon_a)
            if point_count > 0
            else torch.zeros((0,), device=self.device, dtype=self.dtype)
        )
        routing_gate_visual = (
            routing_support_visual / torch.clamp(routing_support_visual + self.config.tau_route_v, min=self.config.epsilon_a)
            if visual_count > 0
            else torch.zeros((0,), device=self.device, dtype=self.dtype)
        )
        routing_gate_tactile = (
            routing_support_tactile / torch.clamp(routing_support_tactile + self.config.tau_route_v, min=self.config.epsilon_a)
            if tactile_count > 0
            else torch.zeros((0,), device=self.device, dtype=self.dtype)
        )
        if point_count > 0:
            denom = torch.clamp(routing_mass_point.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            point_weights = routing_mass_point / denom
            x = point_weights @ token_field.point_positions
            S = _weighted_cov(token_field.point_positions, point_weights, x, self.config)
            a = _extent_from_cov(S, self.config)
        else:
            point_weights = torch.zeros((n_obs, 0), device=self.device, dtype=self.dtype)
            x = torch.zeros((n_obs, 3), device=self.device, dtype=self.dtype)
            S = _diag_embed(torch.full((n_obs, 3), self.config.epsilon_s, device=self.device, dtype=self.dtype))
            a = _to_tensor(self.config.a_min_m, device=self.device, dtype=self.dtype)[None, :].expand(n_obs, -1)
        return PicfObservationAnchorState(
            seed_indices=seed_indices,
            tokens=obs_tokens,
            point_weights=point_weights,
            routing_mass_point=routing_mass_point,
            routing_mass_visual=routing_mass_visual,
            routing_support_point=routing_support_point,
            routing_support_visual=routing_support_visual,
            routing_gate_point=routing_gate_point,
            routing_gate_visual=routing_gate_visual,
            x=x,
            S=S,
            a=a,
            routing_mass_tactile=routing_mass_tactile,
            routing_support_tactile=routing_support_tactile,
            routing_gate_tactile=routing_gate_tactile,
        )

    def _initial_persistent(self) -> tuple[torch.Tensor, ...]:
        k = self.config.persistent_anchors
        mu = torch.zeros((k, self.config.latent_dim), device=self.device, dtype=self.dtype)
        var = torch.full((k, self.config.latent_dim), self.config.sigma_reset**2, device=self.device, dtype=self.dtype)
        h = torch.zeros((k, self.config.posterior_hidden_dim), device=self.device, dtype=self.dtype)
        c = torch.zeros((k, self.config.posterior_hidden_dim), device=self.device, dtype=self.dtype)
        a = _to_tensor(self.config.a_min_m, device=self.device, dtype=self.dtype)[None, :].expand(k, -1).clone()
        S = _diag_embed(torch.clamp((a / 2.0) ** 2, min=self.config.epsilon_s))
        x = torch.zeros((k, 3), device=self.device, dtype=self.dtype)
        alpha = torch.full((k,), self.config.alpha_init, device=self.device, dtype=self.dtype)
        return h, c, mu, var, x, S, a, alpha

    def _current_prior(self, previous: PicfCoreState | None, observation: PicfObservation) -> tuple[torch.Tensor, ...]:
        if previous is None:
            return self._initial_persistent()
        prev = previous.posterior
        prev_var = _diag_from_cov(prev.Sigma)
        proprio = np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1)
        proprio_t = _to_tensor(proprio, device=self.device, dtype=self.dtype)[None, :].expand(prev.h.shape[0], -1)
        action_t = self._previous_action(previous)[None, :].expand(prev.h.shape[0], -1)
        prior_in = torch.cat(
            [
                prev.h,
                prev.mu,
                torch.log(torch.clamp(prev_var, min=self.config.sigma_min2)),
                _geometry_pe(prev.x, prev.a, prev.S, self.config),
                prev.alpha[:, None],
                proprio_t,
                action_t,
            ],
            dim=-1,
        )
        hidden = fn.silu(self.prior_proj(prior_in))
        h_prior, c_prior = self.prior_lstm(hidden, (prev.h, prev.c))
        mu_prior = prev.mu + self.prior_delta_mu(hidden)
        logvar_prior = torch.log(torch.clamp(prev_var, min=self.config.sigma_min2)) + self.prior_delta_logvar(hidden)
        var_prior = _variance_from_logvar(
            logvar_prior,
            min_var=self.config.sigma_min2,
            max_var=self.config.sigma_max2,
        )
        return h_prior, c_prior, mu_prior, var_prior, prev.x, prev.S, prev.a, prev.alpha

    def _sinkhorn_dustbin(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 0:
            return torch.zeros((logits.shape[0] + 1, 0), device=logits.device, dtype=logits.dtype)
        dustbin = torch.zeros((1, logits.shape[1]), device=logits.device, dtype=logits.dtype)
        scores = torch.cat([logits, dustbin], dim=0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0)
        row_target = torch.full((scores.shape[0],), 1.0 / scores.shape[0], device=logits.device, dtype=logits.dtype)
        col_target = torch.full((scores.shape[1],), 1.0 / max(scores.shape[1], 1), device=logits.device, dtype=logits.dtype)
        log_row_target = torch.log(torch.clamp(row_target, min=self.config.epsilon_a))
        log_col_target = torch.log(torch.clamp(col_target, min=self.config.epsilon_a))
        log_P = scores
        for _ in range(6):
            log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True) + log_row_target[:, None]
            log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True) + log_col_target[None, :]
        P = torch.exp(log_P)
        P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
        return P * float(scores.shape[1])

    def _binding_logits(
        self,
        h_prior: torch.Tensor,
        x_prior: torch.Tensor,
        S_prior: torch.Tensor,
        obs: PicfObservationAnchorState,
    ) -> torch.Tensor:
        if obs.tokens.shape[0] == 0:
            return torch.zeros((self.config.persistent_anchors, 0), device=self.device, dtype=self.dtype)
        h_norm = _normalize_tensor(h_prior, eps=self.config.epsilon_residual)
        o_norm = _normalize_tensor(obs.tokens, eps=self.config.epsilon_residual)
        hidden_score = h_norm @ o_norm.T
        delta = obs.x[None, :, :] - x_prior[:, None, :]
        S_diag = torch.diagonal(S_prior, dim1=-2, dim2=-1)
        maha = torch.sum((delta**2) / torch.clamp(S_diag[:, None, :] + (self.config.bind_sigma_m**2), min=self.config.epsilon_s), dim=-1)
        return (self.config.lambda_bind_hidden * hidden_score) - (self.config.lambda_bind_geom * maha)

    def _gather_topk_native_candidates(
        self,
        payload: torch.Tensor,
        weights: torch.Tensor,
        *,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = weights.shape[0]
        if payload.shape[0] == 0 or weights.shape[1] == 0 or topk <= 0:
            return (
                torch.zeros((k, 0, payload.shape[-1] if payload.ndim == 2 else 0), device=self.device, dtype=self.dtype),
                torch.zeros((k, 1, 0), device=self.device, dtype=self.dtype),
            )
        select_k = min(int(topk), int(weights.shape[1]))
        top_values, top_indices = torch.topk(weights, k=select_k, dim=-1)
        candidates = payload[top_indices]
        bias = torch.log(torch.clamp(top_values, min=self.config.epsilon_a))[:, None, :]
        return candidates, bias

    def _gather_tactile_group_candidates(
        self,
        dense_groups: tuple[torch.Tensor, ...],
        weights: torch.Tensor,
        *,
        top_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = weights.shape[0]
        if len(dense_groups) == 0 or weights.shape[1] == 0 or top_groups <= 0:
            return (
                torch.zeros((k, 0, 0), device=self.device, dtype=self.dtype),
                torch.zeros((k, 1, 0), device=self.device, dtype=self.dtype),
            )
        top_values, top_indices = torch.topk(weights, k=min(int(top_groups), len(dense_groups)), dim=-1)
        sequences: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        max_len = 0
        feature_dim = int(dense_groups[0].shape[-1])
        for anchor_index in range(k):
            pieces = []
            piece_biases = []
            for value, group_index in zip(top_values[anchor_index], top_indices[anchor_index]):
                dense = dense_groups[int(group_index.item())]
                pieces.append(dense)
                piece_biases.append(
                    torch.full(
                        (dense.shape[0],),
                        float(torch.log(torch.clamp(value, min=self.config.epsilon_a)).item()),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            seq = torch.cat(pieces, dim=0) if pieces else torch.zeros((0, feature_dim), device=self.device, dtype=self.dtype)
            bias = torch.cat(piece_biases, dim=0) if piece_biases else torch.zeros((0,), device=self.device, dtype=self.dtype)
            sequences.append(seq)
            biases.append(bias)
            max_len = max(max_len, int(seq.shape[0]))
        padded = torch.zeros((k, max_len, feature_dim), device=self.device, dtype=self.dtype)
        padded_bias = torch.full((k, 1, max_len), float("-inf"), device=self.device, dtype=self.dtype)
        for anchor_index, (seq, bias) in enumerate(zip(sequences, biases)):
            if seq.shape[0] == 0:
                continue
            padded[anchor_index, : seq.shape[0]] = seq
            padded_bias[anchor_index, 0, : bias.shape[0]] = bias
        return padded, padded_bias

    def _fuse_measurement_evidence(
        self,
        obs_evidence: torch.Tensor,
        visual_evidence: torch.Tensor,
        tactile_evidence: torch.Tensor,
        support_mass: torch.Tensor,
    ) -> torch.Tensor:
        has_visual = (visual_evidence.abs().sum(dim=-1, keepdim=True) > 0.0).to(dtype=self.dtype)
        has_tactile = (tactile_evidence.abs().sum(dim=-1, keepdim=True) > 0.0).to(dtype=self.dtype)
        joint_in = torch.cat(
            [
                obs_evidence,
                visual_evidence,
                tactile_evidence,
                support_mass[:, None],
                has_visual,
                has_tactile,
            ],
            dim=-1,
        )
        delta = self.evidence_delta(joint_in)
        gate = torch.sigmoid(self.evidence_gate(joint_in))
        return obs_evidence + (gate * delta)

    def _posterior_update(
        self,
        previous: PicfCoreState | None,
        observation: PicfObservation,
        obs_anchors: PicfObservationAnchorState,
        dense_memory: _StepDenseMemory | None = None,
    ) -> PicfPosteriorAnchorState:
        if dense_memory is None:
            dense_memory = _StepDenseMemory(
                point_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                visual_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                tactile_group_tokens=(),
            )
        h_prior, c_prior, mu_prior, var_prior, x_prior, S_prior, a_prior, alpha_prior = self._current_prior(previous, observation)
        bind_logits = self._binding_logits(h_prior, x_prior, S_prior, obs_anchors)
        binding_raw = self._sinkhorn_dustbin(bind_logits)
        support_raw = binding_raw[:-1]
        dustbin_raw = binding_raw[-1]
        support_mass_raw = support_raw.sum(dim=1)
        residual_summary = (
            torch.sum(dustbin_raw[:, None] * obs_anchors.tokens, dim=0)
            if obs_anchors.tokens.shape[0] > 0
            else torch.zeros((self.config.hidden_dim,), device=self.device, dtype=self.dtype)
        )
        alpha_in = torch.cat(
            [
                h_prior,
                support_mass_raw[:, None],
                var_prior.mean(dim=-1, keepdim=True),
                alpha_prior[:, None],
            ],
            dim=-1,
        )
        alpha = torch.sigmoid(self.activity_head(alpha_in)).squeeze(-1)
        recycle_in = torch.cat(
            [
                h_prior,
                support_mass_raw[:, None],
                var_prior.mean(dim=-1, keepdim=True),
                residual_summary[None, :].expand(h_prior.shape[0], -1),
                alpha_prior[:, None],
            ],
            dim=-1,
        )
        recycle = torch.sigmoid(self.recycle_head(recycle_in)).squeeze(-1)
        recycle_share = recycle / torch.clamp(1.0 + recycle.sum(), min=self.config.epsilon_a)
        binding_support = support_raw + (recycle_share[:, None] * dustbin_raw[None, :])
        dustbin_final = dustbin_raw / torch.clamp(1.0 + recycle.sum(), min=self.config.epsilon_a)
        binding = torch.cat([binding_support, dustbin_final[None, :]], dim=0)
        support_mass = binding_support.sum(dim=1)
        res_mu = self.residual_mu_head(residual_summary[None, :])[0]
        res_var = _variance_from_logvar(
            self.residual_logvar_head(residual_summary[None, :]),
            min_var=self.config.sigma_min2,
            max_var=self.config.sigma_max2,
        )[0]
        res_h = self.residual_h_head(residual_summary[None, :])[0]
        res_c = self.residual_c_head(residual_summary[None, :])[0]
        bar_h = (1.0 - recycle[:, None]) * h_prior + recycle[:, None] * res_h[None, :]
        bar_c = (1.0 - recycle[:, None]) * c_prior + recycle[:, None] * res_c[None, :]
        bar_mu = (1.0 - recycle[:, None]) * mu_prior + recycle[:, None] * res_mu[None, :]
        bar_var = (1.0 - recycle[:, None]) * var_prior + recycle[:, None] * res_var[None, :]
        anchor_seed = torch.cat(
            [
                bar_h,
                bar_mu,
                torch.log(torch.clamp(bar_var, min=self.config.sigma_min2)),
                _geometry_pe(x_prior, a_prior, S_prior, self.config),
                alpha_prior[:, None],
            ],
            dim=-1,
        )
        query = self.anchor_seed_proj(anchor_seed)[None, :]
        bias = None
        if obs_anchors.tokens.shape[0] > 0:
            bias = self.config.lambda_bind_prior * torch.log(torch.clamp(binding[:-1], min=self.config.epsilon_a))
        evidence_tokens, _ = self.anchor_reader(query, obs_anchors.tokens[None, :], attn_bias=bias)
        evidence_tokens = evidence_tokens[0]
        binding_cond = binding_support / torch.clamp(support_mass[:, None], min=self.config.epsilon_a)
        visual_evidence = torch.zeros_like(evidence_tokens)
        if dense_memory.visual_payload.numel() > 0 and obs_anchors.routing_mass_visual.shape[1] > 0:
            visual_weights = binding_cond @ obs_anchors.routing_mass_visual
            visual_candidates, visual_bias = self._gather_topk_native_candidates(
                dense_memory.visual_payload,
                visual_weights,
                topk=self.config.visual_reread_topk,
            )
            visual_read, _ = self.visual_native_reread(
                evidence_tokens[:, None, :],
                visual_candidates,
                attn_bias=visual_bias,
            )
            visual_evidence = visual_read[:, 0, :]
        tactile_evidence = torch.zeros_like(evidence_tokens)
        if (
            dense_memory.tactile_group_tokens
            and obs_anchors.routing_mass_tactile is not None
            and obs_anchors.routing_mass_tactile.shape[1] > 0
        ):
            tactile_weights = binding_cond @ obs_anchors.routing_mass_tactile
            tactile_candidates, tactile_bias = self._gather_tactile_group_candidates(
                dense_memory.tactile_group_tokens,
                tactile_weights,
                top_groups=self.config.tactile_reread_groups,
            )
            tactile_read, _ = self.tactile_native_reread(
                evidence_tokens[:, None, :],
                tactile_candidates,
                attn_bias=tactile_bias,
            )
            tactile_evidence = tactile_read[:, 0, :]
        evidence_tokens = self._fuse_measurement_evidence(
            evidence_tokens,
            visual_evidence,
            tactile_evidence,
            support_mass,
        )
        if obs_anchors.tokens.shape[0] > 0:
            denom = torch.clamp(support_mass[:, None], min=self.config.epsilon_a)
            x_obs = (binding[:-1] @ obs_anchors.x) / denom
            centered = obs_anchors.x[None, :, :] - x_obs[:, None, :]
            scatter = centered[..., :, None] * centered[..., None, :]
            second_moment = obs_anchors.S[None, :, :, :] + scatter
            S_obs = torch.einsum("in,inab->iab", binding[:-1], second_moment) / denom[:, :, None]
            valid = support_mass > self.config.epsilon_a
            x = torch.where(valid[:, None], x_obs, x_prior)
            S = torch.where(valid[:, None, None], S_obs, S_prior)
            a_obs = _extent_from_cov(S_obs, self.config)
            a = torch.where(valid[:, None], a_obs, a_prior)
        else:
            x = x_prior
            S = S_prior
            a = a_prior
        contact_prob = torch.sigmoid(self.contact_head(evidence_tokens)).squeeze(-1)
        vote_mu: list[torch.Tensor] = []
        vote_var: list[torch.Tensor] = []
        vote_gamma: list[torch.Tensor] = []
        for head in self.vote_heads:
            delta_mu, delta_logvar, gamma = head(evidence_tokens)
            vote_mu.append(bar_mu + delta_mu)
            vote_var.append(
                _variance_from_logvar(
                    delta_logvar,
                    min_var=self.config.sigma_min2,
                    max_var=self.config.sigma_max2,
                )
            )
            vote_gamma.append(gamma)
        vote_mu_t = torch.stack(vote_mu, dim=0)
        vote_var_t = torch.stack(vote_var, dim=0)
        vote_gamma_t = torch.stack(vote_gamma, dim=0)
        agreement = torch.zeros_like(vote_gamma_t)
        if vote_mu_t.shape[0] > 1:
            for r in range(vote_mu_t.shape[0]):
                penalties = []
                for s in range(vote_mu_t.shape[0]):
                    if r == s:
                        continue
                    penalties.append(self._sym_diag_kl(vote_mu_t[r], vote_var_t[r], vote_mu_t[s], vote_var_t[s]))
                agreement[r] = -torch.stack(penalties, dim=0).mean(dim=0)
        beta = torch.sigmoid(vote_gamma_t + agreement)
        lambda_prior = 1.0 / torch.clamp(bar_var, min=self.config.sigma_min2)
        eta_prior = lambda_prior * bar_mu
        lambda_meas = torch.sum(beta[:, :, None] / torch.clamp(vote_var_t, min=self.config.sigma_min2), dim=0)
        eta_meas = torch.sum(beta[:, :, None] * vote_mu_t / torch.clamp(vote_var_t, min=self.config.sigma_min2), dim=0)
        var_post = 1.0 / torch.clamp(lambda_prior + lambda_meas, min=self.config.epsilon_residual)
        mu_post = var_post * (eta_prior + eta_meas)
        write_in = torch.cat(
            [
                mu_post,
                torch.log(torch.clamp(var_post, min=self.config.sigma_min2)),
                _geometry_pe(x, a, S, self.config),
                alpha[:, None],
                contact_prob[:, None],
            ],
            dim=-1,
        )
        write_hidden = self.post_write_proj(write_in)
        h_post, c_post = self.post_lstm(write_hidden, (bar_h, bar_c))
        token_in = torch.cat(
            [
                h_post,
                mu_post,
                torch.log(torch.clamp(var_post, min=self.config.sigma_min2)),
                _geometry_pe(x, a, S, self.config),
                alpha[:, None],
                contact_prob[:, None],
            ],
            dim=-1,
        )
        tokens = self.posterior_token_proj(token_in)
        tokens = self.posterior_self(tokens[None, :])[0]
        global_post = self.posterior_pool(tokens[None, :])[0]
        return PicfPosteriorAnchorState(
            h=h_post,
            c=c_post,
            mu=mu_post,
            Sigma=_diag_embed(var_post),
            x=x,
            S=S,
            a=a,
            alpha=alpha,
            contact_prob=contact_prob,
            support_mass=support_mass,
            recycle_gate=recycle,
            binding=binding,
            evidence_tokens=evidence_tokens,
            tokens=tokens,
            global_post=global_post,
        )

    @staticmethod
    def _sym_diag_kl(mu_a: torch.Tensor, var_a: torch.Tensor, mu_b: torch.Tensor, var_b: torch.Tensor) -> torch.Tensor:
        kl_ab = 0.5 * (
            torch.sum(torch.log(torch.clamp(var_b, min=1e-6) / torch.clamp(var_a, min=1e-6)), dim=-1)
            + torch.sum((var_a + (mu_a - mu_b) ** 2) / torch.clamp(var_b, min=1e-6), dim=-1)
            - mu_a.shape[-1]
        )
        kl_ba = 0.5 * (
            torch.sum(torch.log(torch.clamp(var_a, min=1e-6) / torch.clamp(var_b, min=1e-6)), dim=-1)
            + torch.sum((var_b + (mu_a - mu_b) ** 2) / torch.clamp(var_a, min=1e-6), dim=-1)
            - mu_a.shape[-1]
        )
        return kl_ab + kl_ba

    def _current_targets(
        self,
        observation: PicfObservation,
        frame_context: PointFrameContext | None,
        visual_map: torch.Tensor | None,
        dense_memory: _StepDenseMemory,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        targets: dict[str, torch.Tensor | None] = {
            "visual_latent": None,
            "visual_real": None,
            "tactile_real": None,
            "point_real": None,
        }
        availability = torch.zeros((4,), device=self.device, dtype=self.dtype)
        if visual_map is not None and visual_map.numel() > 0:
            targets["visual_latent"] = self._visual_latent_target(dense_memory)
            availability[0] = 1.0
        rgb = _to_tensor(np.asarray(observation.rgb_static, dtype=np.float32) / 255.0, device=self.device, dtype=self.dtype)
        if rgb.numel() > 0 and self.config.visual_real_enabled:
            rgb_target = fn.adaptive_avg_pool2d(rgb.permute(2, 0, 1)[None, :], (self.config.visual_real_grid, self.config.visual_real_grid))[0]
            targets["visual_real"] = rgb_target.reshape(-1)
            availability[1] = 1.0
        packet = observation.tactile
        if packet is not None and packet.sensors:
            tactile_maps = []
            for sensor in packet.sensors:
                if not sensor.valid:
                    continue
                img = _to_tensor(np.asarray(sensor.rgb, dtype=np.float32) / 255.0, device=self.device, dtype=self.dtype)
                gray = img.mean(dim=-1, keepdim=True).permute(2, 0, 1)[None, :]
                pooled = fn.adaptive_avg_pool2d(gray, (self.config.tactile_real_grid, self.config.tactile_real_grid))[0, 0]
                tactile_maps.append(pooled.reshape(-1))
            if tactile_maps:
                tactile_base = torch.stack(tactile_maps, dim=0).mean(dim=0)
                tactile_latent = self._tactile_latent_target(dense_memory)
                contact_pose = _to_tensor(observation.contact_pose if observation.contact_pose is not None else np.eye(4, dtype=np.float32), device=self.device, dtype=self.dtype)
                pose_world = _to_tensor(observation.G_t, device=self.device, dtype=self.dtype) @ contact_pose
                force = _to_tensor(observation.force_vec if observation.force_vec is not None else np.zeros((3,), dtype=np.float32), device=self.device, dtype=self.dtype)
                force = force[:3] if force.numel() >= 3 else fn.pad(force, (0, 3 - force.numel()))
                aux = torch.as_tensor(
                    [
                        float(
                            explicit_contact_observation(
                                force_vec=force,
                                indent_depth_m=observation.indent_depth_m,
                                tactile_pressure=observation.tactile_pressure,
                                tau_force_n=self.config.tau_force_n,
                                tau_indent_m=self.config.tau_indent_m,
                                tau_tactile_pressure=self.config.tau_tactile_pressure,
                            )
                            or 0.0
                        ),
                        float(torch.linalg.norm(force).item()),
                        float(observation.indent_depth_m or 0.0),
                        float(observation.tactile_pressure or 0.0),
                        min(len(tactile_maps) / 4.0, 1.0),
                    ],
                    device=self.device,
                    dtype=self.dtype,
                )
                pose_aux = pose_world[:3, 3]
                aux_full = torch.cat([aux, pose_aux], dim=0)
                if tactile_latent is None:
                    tactile_latent = torch.zeros((self.config.tactile_latent_dim,), device=self.device, dtype=self.dtype)
                targets["tactile_real"] = torch.cat([tactile_latent, tactile_base, aux_full], dim=0)
                availability[2] = 1.0
        if frame_context is not None and frame_context.points_local.shape[0] > 0:
            points = _to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype)
            point_latent = self._point_latent_target(dense_memory)
            center = _to_tensor(observation.G_t[:3, 3], device=self.device, dtype=self.dtype)
            rel = torch.clamp((points - center[None, :]) / max(self.config.crop_radius_m, 1e-6), min=-0.999, max=0.999)
            grid = ((rel + 1.0) * 0.5 * self.config.point_real_grid).long()
            grid = torch.clamp(grid, min=0, max=self.config.point_real_grid - 1)
            _assert_index_tensor_bounds(grid[:, 0], size=self.config.point_real_grid, name="point_real.grid_x")
            _assert_index_tensor_bounds(grid[:, 1], size=self.config.point_real_grid, name="point_real.grid_y")
            _assert_index_tensor_bounds(grid[:, 2], size=self.config.point_real_grid, name="point_real.grid_z")
            occ = torch.zeros((self.config.point_real_grid, self.config.point_real_grid, self.config.point_real_grid), device=self.device, dtype=self.dtype)
            occ[grid[:, 0], grid[:, 1], grid[:, 2]] = 1.0
            if point_latent is None:
                point_latent = torch.zeros((self.config.point_latent_dim,), device=self.device, dtype=self.dtype)
            targets["point_real"] = torch.cat([point_latent, occ.reshape(-1)], dim=0)
            availability[3] = 1.0
        return targets, availability

    def extract_targets(
        self,
        observation: PicfObservation,
        *,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Expose the current-step target construction used by one-step future supervision.

        This is intentionally public so training/smoke code can supervise `g_t^{pred}` against
        `t+1` targets without reimplementing the target builder outside the core.
        """
        if observation.G_t is None:
            observation.G_t = self.local_frame.make_transform(observation.robot_obs)
        if observation.point_set is None:
            focus_centers_world = _focus_centers_world_from_observation(observation)
            observation.point_set = self.pointcloud_builder(
                {
                    "rgb_static": observation.rgb_static,
                    "depth_static": observation.depth_static,
                    "rgb_gripper": observation.rgb_gripper,
                    "depth_gripper": observation.depth_gripper,
                    "robot_obs": observation.robot_obs,
                    "focus_centers_world": focus_centers_world,
                    "focus_radius_m": self.config.crop_radius_m,
                }
            )
        meta = self._build_runtime_meta(observation, observation.runtime_meta)
        frame_context = self._point_subset(observation) if meta.point_contract_ok else None
        point_features = self._extract_point_features(frame_context, None) if frame_context is not None else torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        clip_snapshot = None
        if visual_map_override is None and self.clip_buffer is not None:
            clip_snapshot = self.clip_buffer.snapshot()
        try:
            visual_map = self._visual_map(observation, visual_map_override, meta)
        finally:
            if clip_snapshot is not None and self.clip_buffer is not None:
                self.clip_buffer.restore(clip_snapshot)
        tactile_bundle = self._tactile_features(observation, meta)
        point_payload = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        visual_payload = (
            visual_map.reshape(-1, visual_map.shape[-1])
            if visual_map is not None and visual_map.numel() > 0
            else torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        )
        if frame_context is not None:
            projective_geometry = self._build_projective_geometry(
                observation=observation,
                point_positions=_to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype),
                visual_hw=None if visual_map is None or visual_map.numel() == 0 else (int(visual_map.shape[0]), int(visual_map.shape[1])),
            )
            proj_features = self._point_projection_features(
                projective_geometry,
                source_hw=(int(observation.rgb_static.shape[0]), int(observation.rgb_static.shape[1])),
            )
            point_positions = _to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype)
            point_payload = torch.cat(
                [
                    point_positions,
                    _to_tensor(frame_context.colors, device=self.device, dtype=self.dtype),
                    _point_pe(point_positions, self.config),
                    proj_features,
                ],
                dim=-1,
            )
        tactile_group_tokens: tuple[torch.Tensor, ...] = ()
        if tactile_bundle is not None and tactile_bundle.sensors:
            tactile_group_tokens = tuple(
                sensor.tokens.to(device=self.device, dtype=self.dtype)
                for sensor in tactile_bundle.sensors.values()
                if sensor.tokens.numel() > 0
            )
        dense_memory = _StepDenseMemory(
            point_payload=point_payload,
            visual_payload=visual_payload,
            tactile_group_tokens=tactile_group_tokens,
        )
        return self._current_targets(observation, frame_context, visual_map, dense_memory)

    def _standardized_residual(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        residual = target - pred
        scale = torch.sqrt(torch.mean(residual**2) + self.config.epsilon_residual)
        return residual / scale

    def _innovation(
        self,
        previous: PicfCoreState | None,
        targets: dict[str, torch.Tensor | None],
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous is None:
            return torch.zeros((self.config.hidden_dim,), device=self.device, dtype=self.dtype), torch.zeros((4,), device=self.device, dtype=self.dtype)
        # Innovation must compare against the world-only predictive basis, not the
        # semantic-conditioned future readout.
        cache = previous.predictive.physical_prediction_cache
        branch_specs = (
            ("visual_latent", cache.visual_latent, self.visual_error_encoder, self.config.hidden_dim),
            ("visual_real", cache.visual_real, self.visual_real_error_encoder, self.config.visual_real_dim),
            ("tactile_real", cache.tactile_real, self.tactile_error_encoder, self.config.tactile_real_dim),
            ("point_real", cache.point_real, self.point_error_encoder, self.config.point_real_dim),
        )
        branch_feats = []
        norms = []
        for index, (name, pred, encoder, target_dim) in enumerate(branch_specs):
            target = targets[name]
            usable = bool(availability[index].item()) and pred is not None and bool(cache.availability[index].item())
            if target is not None:
                target_vec = target.reshape(-1)
            else:
                target_vec = torch.zeros((target_dim,), device=self.device, dtype=self.dtype)
            if pred is not None:
                pred_vec = pred.reshape(-1)
            else:
                pred_vec = torch.zeros((target_dim,), device=self.device, dtype=self.dtype)
            if pred_vec.numel() != target_dim:
                if pred_vec.numel() > target_dim:
                    pred_vec = pred_vec[:target_dim]
                else:
                    pred_vec = fn.pad(pred_vec, (0, target_dim - pred_vec.numel()))
            if target_vec.numel() != target_dim:
                if target_vec.numel() > target_dim:
                    target_vec = target_vec[:target_dim]
                else:
                    target_vec = fn.pad(target_vec, (0, target_dim - target_vec.numel()))
            if usable and target is not None:
                std = self._standardized_residual(target_vec, pred_vec)
                mask = torch.ones((1,), device=self.device, dtype=self.dtype)
            else:
                std = torch.zeros((target_dim,), device=self.device, dtype=self.dtype)
                mask = torch.zeros((1,), device=self.device, dtype=self.dtype)
            branch_input = torch.cat([target_vec, pred_vec, std], dim=0)[None, :]
            feat = fn.silu(encoder(branch_input))[0] * mask[0]
            branch_feats.append(feat)
            norms.append(torch.linalg.norm(std)[None] * mask)
        innovation_latent = self.innovation_proj(torch.cat([*branch_feats, availability], dim=0)[None, :])[0]
        innovation = self.innovation_token_proj(innovation_latent)
        return innovation, torch.cat(norms, dim=0)

    def observe_step(
        self,
        observation: PicfObservation,
        previous: PicfCoreState | None = None,
        *,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        semantic_override: torch.Tensor | np.ndarray | None = None,
    ) -> _ObservedStepState:
        if observation.G_t is None:
            observation.G_t = self.local_frame.make_transform(observation.robot_obs)
        if observation.point_set is None:
            focus_centers_world = _focus_centers_world_from_observation(observation)
            observation.point_set = self.pointcloud_builder(
                {
                    "rgb_static": observation.rgb_static,
                    "depth_static": observation.depth_static,
                    "rgb_gripper": observation.rgb_gripper,
                    "depth_gripper": observation.depth_gripper,
                    "robot_obs": observation.robot_obs,
                    "focus_centers_world": focus_centers_world,
                    "focus_radius_m": self.config.crop_radius_m,
                }
            )
        if observation.reset_scaffold:
            previous = None
        meta = self._build_runtime_meta(observation, previous.runtime_meta if previous is not None else None)
        if previous is None and not meta.point_contract_ok:
            raise RuntimeError("PICF core requires a valid xyzrgb point cloud on the first control step.")
        frame_context = self._point_subset(observation) if meta.point_contract_ok else None
        if previous is None and frame_context is not None and frame_context.points_local.shape[0] == 0:
            raise RuntimeError("PICF core requires non-empty local xyzrgb support on the first control step.")
        point_features = self._extract_point_features(frame_context, point_features_override) if frame_context is not None else torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        visual_map = self._visual_map(observation, visual_map_override, meta)
        tactile_bundle = self._tactile_features(observation, meta)
        semantic = self._semantic_context(observation, previous, semantic_override)
        token_field, dense_memory = self._build_token_field(observation, frame_context, point_features, visual_map, tactile_bundle, meta, previous)
        observation_anchors = self._build_observation_anchors(token_field, dense_memory)
        posterior = self._posterior_update(previous, observation, observation_anchors, dense_memory)
        current_targets, availability = self._current_targets(observation, frame_context, visual_map, dense_memory)
        innovation_token, innovation_norm = self._innovation(previous, current_targets, availability)
        proprio = _to_tensor(
            np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1),
            device=self.device,
            dtype=self.dtype,
        )
        proprio_token = self.proprio_proj(proprio[None, :])[0]
        task_readout = self._build_task_readout(token_field, dense_memory, semantic, proprio_token)
        conditioned_control = self._build_conditioned_control_state(posterior, innovation_token, proprio_token, task_readout)
        hold_reason = self._hold_reason(meta, posterior, innovation_token)
        return _ObservedStepState(
            runtime_meta=meta,
            G_t=_to_tensor(observation.G_t, device=self.device, dtype=self.dtype),
            token_field=token_field,
            dense_memory=dense_memory,
            observation_anchors=observation_anchors,
            posterior=posterior,
            current_targets=current_targets,
            availability=availability,
            innovation_token=innovation_token,
            innovation_norm=innovation_norm,
            semantic=semantic,
            proprio_token=proprio_token,
            task_readout=task_readout,
            conditioned_control=conditioned_control,
            control=PicfControlState(hold_reason=hold_reason),
            last_prompt=observation.prompt,
        )

    def _predictive_state(
        self,
        observation: PicfObservation,
        posterior: PicfPosteriorAnchorState,
        semantic: _SemanticContext,
        innovation_token: torch.Tensor,
        innovation_norm: torch.Tensor,
        targets_availability: torch.Tensor,
        action_future: torch.Tensor | np.ndarray | None,
        *,
        conditioned_control: PicfConditionedControlState,
        proprio_token: torch.Tensor,
    ) -> PicfPredictiveState:
        action, action_chunk = self._default_predictive_action(action_future)
        executed_action = self._executed_action(observation, action)
        physical_pred_tokens, physical_global_pred, physical_prediction_cache = self._build_physical_predictive_basis(
            posterior,
            proprio_token=proprio_token,
            executed_action=executed_action,
        )
        predictive_query_state, global_pred, prediction_cache = self._build_conditioned_predictive_cache(
            physical_pred_tokens,
            conditioned_control.future_condition_tokens,
        )
        pooled_state = self.control_state_proj(conditioned_control.query_state)
        return PicfPredictiveState(
            semantic_tokens=semantic.tokens,
            innovation_token=innovation_token,
            innovation_norm=innovation_norm,
            availability=targets_availability,
            physical_pred_tokens=physical_pred_tokens,
            control_tokens=conditioned_control.tokens,
            action_condition_tokens=conditioned_control.pi_prefix_tokens,
            control_query_state=conditioned_control.query_state,
            pooled_state=pooled_state,
            action=action,
            action_chunk=action_chunk,
            executed_action=executed_action,
            physical_global_pred=physical_global_pred,
            physical_prediction_cache=physical_prediction_cache,
            predictive_query_state=predictive_query_state,
            global_pred=global_pred,
            prediction_cache=prediction_cache,
        )

    def finalize_with_action(
        self,
        observation: PicfObservation,
        observed: _ObservedStepState,
        *,
        action_future: torch.Tensor | np.ndarray | None,
    ) -> PicfCoreOutput:
        predictive = self._predictive_state(
            observation,
            posterior=observed.posterior,
            semantic=observed.semantic,
            innovation_token=observed.innovation_token,
            innovation_norm=observed.innovation_norm,
            targets_availability=observed.availability,
            action_future=action_future,
            conditioned_control=observed.conditioned_control,
            proprio_token=observed.proprio_token,
        )
        state = PicfCoreState(
            runtime_meta=observed.runtime_meta,
            G_t=observed.G_t,
            token_field=observed.token_field,
            observation_anchors=observed.observation_anchors,
            posterior=observed.posterior,
            task_readout=observed.task_readout,
            conditioned_control=observed.conditioned_control,
            predictive=predictive,
            control=observed.control,
            last_prompt=observed.last_prompt,
        )
        debug = {
            "num_point_tokens": float(observed.token_field.point_tokens.shape[0]),
            "num_visual_tokens": float(observed.token_field.visual_tokens.shape[0]),
            "num_tactile_tokens": float(observed.token_field.tactile_tokens.shape[0]),
            "num_tactile_tokens_all": float(0 if observed.token_field.tactile_tokens_all is None else observed.token_field.tactile_tokens_all.shape[0]),
            "support_mass_mean": float(observed.posterior.support_mass.mean().item()),
            "active_alpha_sum": float(observed.posterior.alpha.sum().item()),
            "innovation_norm": float(torch.linalg.norm(observed.innovation_token).item()),
            "hold_triggered": 1.0 if observed.control.hold_reason is not None else 0.0,
        }
        if observed.token_field.tactile_contact_prob is not None and observed.token_field.tactile_contact_prob.numel() > 0:
            debug["tactile_contact_prob_mean"] = float(observed.token_field.tactile_contact_prob.mean().item())
        if observed.token_field.tactile_anchor_mask is not None and observed.token_field.tactile_anchor_mask.numel() > 0:
            debug["tactile_active_rate"] = float(observed.token_field.tactile_anchor_mask.to(dtype=self.dtype).mean().item())
        if observed.observation_anchors.routing_gate_point.numel() > 0:
            debug["mean_point_route_gate"] = float(observed.observation_anchors.routing_gate_point.mean().item())
            debug["mean_point_route_support"] = float(observed.observation_anchors.routing_support_point.mean().item())
        if observed.observation_anchors.routing_gate_visual.numel() > 0:
            debug["mean_visual_route_gate"] = float(observed.observation_anchors.routing_gate_visual.mean().item())
            debug["mean_visual_route_support"] = float(observed.observation_anchors.routing_support_visual.mean().item())
        if observed.token_field.projective_geometry is not None:
            geom = observed.token_field.projective_geometry
            num_edges = int(geom.projective_candidate_mask.sum().item()) if geom.projective_candidate_mask.numel() > 0 else 0
            total_edges = int(geom.projective_candidate_mask.numel())
            density = (float(num_edges) / float(total_edges)) if total_edges > 0 else 0.0
            debug["mean_point_visibility"] = float(geom.point_visibility.mean().item()) if geom.point_visibility.numel() > 0 else 0.0
            debug["projective_candidate_edges"] = float(num_edges)
            debug["projective_candidate_density"] = float(density)
        return PicfCoreOutput(state=state, debug=debug)

    def refresh_predictive_state_for_action(
        self,
        observation: PicfObservation,
        state: PicfCoreState,
        *,
        action_future: torch.Tensor | np.ndarray,
    ) -> PicfPredictiveState:
        observed = _ObservedStepState(
            runtime_meta=state.runtime_meta,
            G_t=state.G_t,
            token_field=state.token_field,
            dense_memory=_StepDenseMemory(
                point_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                visual_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                tactile_group_tokens=(),
            ),
            observation_anchors=state.observation_anchors,
            posterior=state.posterior,
            current_targets={},
            availability=state.predictive.availability,
            innovation_token=state.predictive.innovation_token,
            innovation_norm=state.predictive.innovation_norm,
            semantic=self._project_semantic_context(tokens_raw=state.predictive.semantic_tokens),
            proprio_token=self.proprio_proj(
                _to_tensor(
                    np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1),
                    device=self.device,
                    dtype=self.dtype,
                )[None, :]
            )[0],
            task_readout=state.task_readout,
            conditioned_control=state.conditioned_control,
            control=state.control,
            last_prompt=state.last_prompt,
        )
        return self.finalize_with_action(observation, observed, action_future=action_future).state.predictive

    def _hold_reason(self, meta: RuntimeMeta, posterior: PicfPosteriorAnchorState, innovation_token: torch.Tensor) -> str | None:
        if not meta.point_contract_ok:
            return "point_contract_violation"
        if not meta.sync_valid:
            return "sensor_sync_invalid"
        uncertainty = _diag_from_cov(posterior.Sigma).mean(dim=-1).mean()
        if float(uncertainty.item()) > self.config.hold_uncertainty_threshold:
            return "posterior_uncertainty_spike"
        if float(torch.linalg.norm(innovation_token).item()) > self.config.hold_innovation_threshold and float(posterior.alpha.sum().item()) < self.config.hold_activity_threshold:
            return "innovation_with_anchor_collapse"
        return None

    def step(
        self,
        observation: PicfObservation,
        previous: PicfCoreState | None = None,
        *,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        semantic_override: torch.Tensor | np.ndarray | None = None,
        action_future: torch.Tensor | np.ndarray | None = None,
    ) -> PicfCoreOutput:
        observed = self.observe_step(
            observation,
            previous=previous,
            point_features_override=point_features_override,
            visual_map_override=visual_map_override,
            semantic_override=semantic_override,
        )
        return self.finalize_with_action(observation, observed, action_future=action_future)

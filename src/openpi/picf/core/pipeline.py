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
from openpi.picf.core.contracts import PicfControlState
from openpi.picf.core.contracts import PicfCoreOutput
from openpi.picf.core.contracts import PicfCoreState
from openpi.picf.core.contracts import PicfObservationAnchorState
from openpi.picf.core.contracts import PicfPosteriorAnchorState
from openpi.picf.core.contracts import PicfPredictionCache
from openpi.picf.core.contracts import PicfPredictiveState
from openpi.picf.core.contracts import PicfProjectiveGeometryState
from openpi.picf.core.contracts import PicfTokenFieldState
from openpi.picf.frame_context import PointFrameContext
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


def _normalize_tensor(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def _clip_vector_norm(x: torch.Tensor, *, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1e-12), max=1.0)
    return x * scale


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
    focus_center_world: np.ndarray,
) -> PointFrameContext:
    assert observation.point_set is not None
    xyz_world = np.asarray(observation.point_set.xyz_world, dtype=np.float32)
    dists = np.linalg.norm(xyz_world - focus_center_world[None, :], axis=1)
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
    def __init__(self, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(SelfAttentionBlock(hidden_dim, heads, 1) for _ in range(layers))

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
    def __init__(self, query_dim: int, kv_dim: int, heads: int, *, inner_dim: int):
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
        self.cross_gate = nn.Parameter(torch.zeros(()))
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
        output = queries + (torch.tanh(self.cross_gate) * attn_out)
        output = output + self.ff(self.ff_norm(output))
        mean_weights = weights.mean(dim=1)[0]
        return output, mean_weights


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
    summary: torch.Tensor


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
        proj_coarse_dim = 2 * 4 * 2
        proj_fine_dim = 2 * 8 * 2
        self.null_proj_coarse = nn.Parameter(torch.zeros(proj_coarse_dim, device=self.device, dtype=self.dtype))
        self.null_proj_fine = nn.Parameter(torch.zeros(proj_fine_dim, device=self.device, dtype=self.dtype))
        self.projective_bias_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.token_fusion = TransformerStack(hidden_dim, heads, self.config.fusion_layers)

        self.obs_reader = CrossAttentionRead(hidden_dim, heads)
        self.obs_self = TransformerStack(hidden_dim, heads, 1)

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
        self.posterior_self = TransformerStack(hidden_dim, heads, self.config.posterior_layers)
        self.posterior_pool = AttentionPool(hidden_dim)

        self.semantic_summary_proj = nn.LazyLinear(hidden_dim)
        self.proprio_proj = nn.LazyLinear(hidden_dim)
        self.action_cond_proj = nn.LazyLinear(hidden_dim)
        self.predictive_world = TransformerStack(hidden_dim, heads, self.config.predictive_layers)
        self.predictive_semantic_reads = nn.ModuleList(
            GatedCrossAttentionRead(
                hidden_dim,
                self.config.semantic_dim,
                heads,
                inner_dim=self.config.semantic_cross_dim,
            )
            for _ in range(self.config.predictive_semantic_reads)
        )
        self.predictive_pool = AttentionPool(hidden_dim)

        self.visual_latent_target_proj = nn.LazyLinear(hidden_dim)
        self.visual_latent_head = nn.Linear(hidden_dim, hidden_dim)
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

        self.control_world = TransformerStack(hidden_dim, heads, self.config.control_layers)
        self.control_semantic_reads = nn.ModuleList(
            GatedCrossAttentionRead(
                hidden_dim,
                self.config.semantic_dim,
                heads,
                inner_dim=self.config.semantic_cross_dim,
            )
            for _ in range(self.config.control_semantic_reads)
        )
        self.control_pool = AttentionPool(hidden_dim)
        self.control_state_proj = nn.Linear(hidden_dim, self.config.control_dim)
        self.action_head = nn.Linear(self.config.control_dim, 7)
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
        focus_center_world = np.asarray(observation.G_t[:3, 3], dtype=np.float32)
        return _build_identity_frame_context(observation, crop_radius_m=self.config.crop_radius_m, focus_center_world=focus_center_world)

    def _extract_point_features(self, frame_context: PointFrameContext, override: torch.Tensor | np.ndarray | None) -> torch.Tensor:
        if override is not None:
            feature = _to_tensor(override, device=self.device, dtype=self.dtype)
            return feature if feature.ndim == 2 else feature.squeeze(0)
        if self.point_feature_extractor is None:
            return _to_tensor(frame_context.colors, device=self.device, dtype=self.dtype)
        encoded = self.point_feature_extractor.encode_local_context(frame_context)
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
        fmap = self.visual_encoder.encode_clip(self.clip_buffer.get_clip())
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
        return self.tactile_encoder.encode_sensor_clips(
            clips_by_sensor=clips,
            backgrounds_by_sensor=backgrounds,
            poses_by_sensor=poses,
        )

    def _zero_semantic_context(self) -> _SemanticContext:
        return _SemanticContext(
            tokens=torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype),
            summary=torch.zeros((1, self.config.hidden_dim), device=self.device, dtype=self.dtype),
        )

    def _project_semantic_context(
        self,
        *,
        tokens_raw: torch.Tensor,
        summary_raw: torch.Tensor,
    ) -> _SemanticContext:
        if tokens_raw.ndim == 1:
            tokens_raw = tokens_raw[None, :]
        if summary_raw.ndim == 1:
            summary_raw = summary_raw[None, :]
        tokens_raw = _to_tensor(tokens_raw, device=self.device, dtype=self.dtype)
        summary_raw = _to_tensor(summary_raw, device=self.device, dtype=self.dtype)
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
        semantic_summary = self.semantic_summary_proj(summary_raw)
        return _SemanticContext(tokens=semantic_tokens, summary=semantic_summary)

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
            return self._project_semantic_context(
                tokens_raw=semantic_override.tokens,
                summary_raw=semantic_override.summary,
            )
        if isinstance(semantic_override, torch.Tensor | np.ndarray):
            raw = _to_tensor(semantic_override, device=self.device, dtype=self.dtype)
            raw = raw if raw.ndim == 2 else raw[None, :]
            return self._project_semantic_context(
                tokens_raw=raw,
                summary_raw=raw.mean(dim=0, keepdim=True),
            )
        if isinstance(semantic_override, dict):
            if "tokens" in semantic_override:
                tokens_raw = semantic_override["tokens"]
                summary_raw = semantic_override.get("summary")
                if summary_raw is None:
                    summary_raw = _to_tensor(tokens_raw, device=self.device, dtype=self.dtype)
                    summary_raw = summary_raw.mean(dim=0, keepdim=True) if summary_raw.ndim == 2 else summary_raw[None, :]
                return self._project_semantic_context(tokens_raw=tokens_raw, summary_raw=summary_raw)
            raw = self.semantic_wrapper.summarize(**semantic_override)
        else:
            raw = self.semantic_wrapper.summarize(outputs=semantic_override)
        raw = _to_tensor(raw, device=self.device, dtype=self.dtype)
        raw = raw if raw.ndim == 2 else raw[None, :]
        return _SemanticContext(
            tokens=torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype),
            summary=self.semantic_summary_proj(raw),
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
        return semantic_tokens[keep]

    def _apply_semantic_reads(
        self,
        world_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        *,
        reads: nn.ModuleList,
        dropout_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = world_tokens[None, :]
        attention = None
        semantic_memory = self._semantic_memory(semantic_tokens, dropout_prob=dropout_prob)
        for layer in reads:
            x, attention = layer(x, semantic_memory[None, :])
        return x[0], attention

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

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        pos = _clip_vector_norm(action[..., :3], max_norm=self.config.max_action_pos_m)
        rot = _clip_vector_norm(action[..., 3:6], max_norm=self.config.max_action_rot_rad)
        grip = torch.clamp(action[..., 6:], min=-self.config.max_action_gripper, max=self.config.max_action_gripper)
        return torch.cat([pos, rot, grip], dim=-1)

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
        delta = point_proj_grid_index[:, None, :] - visual_grid_index[None, :, :]
        point_rays = _normalize_tensor(point_positions - camera_origin_world[None, :], eps=self.config.epsilon_residual)
        ray_align = torch.sum(point_rays[:, None, :] * visual_ray_world[None, :, :], dim=-1, keepdim=True)
        visibility = point_visibility[:, None, None]
        log_depth = torch.log(torch.clamp(point_depth[:, None, None], min=self.config.z_min_m)).expand(-1, visual_count, -1)
        features = torch.cat([delta, log_depth, ray_align, visibility.expand(-1, visual_count, -1)], dim=-1)
        values = self.projective_bias_head(features[candidate_mask][:, None, :])[:, 0, 0]
        bias[candidate_mask] = self.config.projective_bias_scale * torch.tanh(values)
        return bias

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
            depth_residual = z[valid_depth_rows, None] - depth_sample[valid_depth_rows, None]
            depth_factor[valid_depth_rows] = torch.exp(
                -(depth_residual**2) / (2.0 * (self.config.tau_proj_depth_m**2))
            )
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
    ) -> PicfTokenFieldState:
        hidden_dim = self.config.hidden_dim
        point_tokens = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        point_positions = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        point_align_embeddings = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        visual_align_embeddings = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        tactile_align_embeddings = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        tactile_positions_world = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        tactile_contact_gate = torch.zeros((0,), device=self.device, dtype=self.dtype)
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
            ray_features = self._visual_ray_features(projective_geometry, source_hw=(int(observation.rgb_static.shape[0]), int(observation.rgb_static.shape[1])))
            visual_in = torch.cat([flat_map, grid, cam_pose.expand(flat_map.shape[0], -1), ray_features], dim=-1)
            visual_tokens = self.visual_token_proj(visual_in) + self.modality_embedding.weight[1][None, :]
            visual_align_embeddings = _normalize_tensor(self.visual_align_proj(visual_tokens), eps=self.config.epsilon_residual)

        tactile_tokens = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        if tactile_bundle is not None and tactile_bundle.sensors:
            encoded = []
            positions = []
            for sensor_name in sorted(tactile_bundle.sensors):
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
            tactile_tokens = self.tactile_token_proj(torch.stack(encoded, dim=0)) + self.modality_embedding.weight[2][None, :]
            tactile_align_embeddings = _normalize_tensor(self.tactile_align_proj(tactile_tokens), eps=self.config.epsilon_residual)
            tactile_positions_world = torch.stack(positions, dim=0)
            has_explicit_contact = (
                observation.force_vec is not None
                or observation.indent_depth_m is not None
                or observation.tactile_pressure is not None
            )
            contact_value = (
                float(
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
                if has_explicit_contact
                else 1.0
            )
            tactile_contact_gate = torch.full(
                (tactile_tokens.shape[0],),
                contact_value,
                device=self.device,
                dtype=self.dtype,
            )

        context_tokens = self._encode_context_tokens(observation, meta, previous) + self.modality_embedding.weight[3][None, :]
        all_tokens = torch.cat([point_tokens, visual_tokens, tactile_tokens, context_tokens], dim=0)
        fusion_attention_mean = None
        if all_tokens.shape[0] > 0:
            fusion_bias = self._fusion_projective_bias(
                projective_geometry=projective_geometry,
                point_count=point_tokens.shape[0],
                visual_count=visual_tokens.shape[0],
                total_tokens=all_tokens.shape[0],
            )
            fused, fusion_attention_mean = self.token_fusion(
                all_tokens[None, :],
                attn_bias=fusion_bias,
                return_attention=True,
            )
            fused = fused[0]
        else:
            fused = all_tokens
        modality_ids = torch.cat(
            [
                torch.zeros((point_tokens.shape[0],), device=self.device, dtype=torch.long),
                torch.ones((visual_tokens.shape[0],), device=self.device, dtype=torch.long),
                torch.full((tactile_tokens.shape[0],), 2, device=self.device, dtype=torch.long),
                torch.full((context_tokens.shape[0],), 3, device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        return PicfTokenFieldState(
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
            fusion_attention_mean=fusion_attention_mean,
            projective_geometry=projective_geometry,
        )

    def _build_observation_anchors(self, token_field: PicfTokenFieldState) -> PicfObservationAnchorState:
        n_obs = self.config.observation_anchors
        hidden_dim = self.config.hidden_dim
        point_count = token_field.point_tokens.shape[0]
        visual_count = token_field.visual_tokens.shape[0]
        seed_indices = torch.full((n_obs,), -1, device=self.device, dtype=torch.long)
        queries = torch.zeros((1, n_obs, hidden_dim), device=self.device, dtype=self.dtype)
        if point_count > 0:
            chosen = _fps_indices(token_field.point_positions, min(n_obs, point_count))
            seed_indices[: chosen.shape[0]] = chosen
            queries[0, : chosen.shape[0]] = token_field.point_tokens[chosen]
        attn = torch.zeros((n_obs, token_field.fused_tokens.shape[0]), device=self.device, dtype=self.dtype)
        for _ in range(max(self.config.query_rounds, 1)):
            queries, attn = self.obs_reader(queries, token_field.fused_tokens[None, :])
        obs_tokens = self.obs_self(queries)[0]
        routing_mass_point = attn[:, :point_count]
        routing_mass_visual = attn[:, point_count : point_count + visual_count]
        routing_support_point = routing_mass_point.sum(dim=0) if point_count > 0 else torch.zeros((0,), device=self.device, dtype=self.dtype)
        routing_support_visual = routing_mass_visual.sum(dim=0) if visual_count > 0 else torch.zeros((0,), device=self.device, dtype=self.dtype)
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
        var_prior = torch.clamp(torch.exp(logvar_prior), min=self.config.sigma_min2, max=self.config.sigma_max2)
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

    def _posterior_update(
        self,
        previous: PicfCoreState | None,
        observation: PicfObservation,
        obs_anchors: PicfObservationAnchorState,
    ) -> PicfPosteriorAnchorState:
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
        res_var = torch.clamp(torch.exp(self.residual_logvar_head(residual_summary[None, :])), min=self.config.sigma_min2, max=self.config.sigma_max2)[0]
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
        if obs_anchors.tokens.shape[0] > 0:
            denom = torch.clamp(support_mass[:, None], min=self.config.epsilon_a)
            x_obs = (binding[:-1] @ obs_anchors.x) / denom
            centered = obs_anchors.x[None, :, :] - x_obs[:, None, :]
            scatter = centered[..., :, None] * centered[..., None, :]
            second_moment = obs_anchors.S[None, :, :, :] + scatter
            S_obs = torch.einsum("in,inab->iab", binding[:-1], second_moment) / denom[:, :, None]
            valid = support_mass > self.config.epsilon_a
            x = torch.where(valid[:, None], x_obs, x_prior)
            S = S_prior.clone()
            S[valid] = S_obs[valid]
            a = a_prior.clone()
            a[valid] = _extent_from_cov(S[valid], self.config)
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
            vote_var.append(torch.clamp(torch.exp(delta_logvar), min=self.config.sigma_min2, max=self.config.sigma_max2))
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
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        targets: dict[str, torch.Tensor | None] = {
            "visual_latent": None,
            "visual_real": None,
            "tactile_real": None,
            "point_real": None,
        }
        availability = torch.zeros((4,), device=self.device, dtype=self.dtype)
        if visual_map is not None and visual_map.numel() > 0:
            pooled = visual_map.mean(dim=(0, 1))
            targets["visual_latent"] = self.visual_latent_target_proj(pooled[None, :])[0]
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
                targets["tactile_real"] = torch.cat([tactile_base, aux_full], dim=0)
                availability[2] = 1.0
        if frame_context is not None and frame_context.points_local.shape[0] > 0:
            points = _to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype)
            center = _to_tensor(observation.G_t[:3, 3], device=self.device, dtype=self.dtype)
            rel = torch.clamp((points - center[None, :]) / max(self.config.crop_radius_m, 1e-6), min=-0.999, max=0.999)
            grid = ((rel + 1.0) * 0.5 * self.config.point_real_grid).long()
            grid = torch.clamp(grid, min=0, max=self.config.point_real_grid - 1)
            occ = torch.zeros((self.config.point_real_grid, self.config.point_real_grid, self.config.point_real_grid), device=self.device, dtype=self.dtype)
            occ[grid[:, 0], grid[:, 1], grid[:, 2]] = 1.0
            targets["point_real"] = occ.reshape(-1)
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
            observation.point_set = self.pointcloud_builder(
                {
                    "rgb_static": observation.rgb_static,
                    "depth_static": observation.depth_static,
                    "focus_center_world": np.asarray(observation.G_t[:3, 3], dtype=np.float32),
                    "focus_radius_m": self.config.crop_radius_m,
                }
            )
        meta = self._build_runtime_meta(observation, observation.runtime_meta)
        frame_context = self._point_subset(observation) if meta.point_contract_ok else None
        clip_snapshot = None
        if visual_map_override is None and self.clip_buffer is not None:
            clip_snapshot = self.clip_buffer.snapshot()
        try:
            visual_map = self._visual_map(observation, visual_map_override, meta)
        finally:
            if clip_snapshot is not None and self.clip_buffer is not None:
                self.clip_buffer.restore(clip_snapshot)
        return self._current_targets(observation, frame_context, visual_map)

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

    def _predictive_state(
        self,
        observation: PicfObservation,
        posterior: PicfPosteriorAnchorState,
        semantic: _SemanticContext,
        innovation_token: torch.Tensor,
        innovation_norm: torch.Tensor,
        targets_availability: torch.Tensor,
        action_future: torch.Tensor | np.ndarray | None,
    ) -> PicfPredictiveState:
        proprio = _to_tensor(
            np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1),
            device=self.device,
            dtype=self.dtype,
        )
        proprio_token = self.proprio_proj(proprio[None, :])[0]
        control_world_tokens = torch.cat(
            [
                posterior.tokens,
                innovation_token[None, :],
                proprio_token[None, :],
            ],
            dim=0,
        )
        control_world_tokens = self.control_world(control_world_tokens[None, :])[0]
        control_tokens, _ = self._apply_semantic_reads(
            control_world_tokens,
            semantic.tokens,
            reads=self.control_semantic_reads,
        )
        pooled_hidden = self.control_pool(control_tokens[None, :])[0]
        pooled_state = self.control_state_proj(pooled_hidden)
        action = self._clip_action(self.action_head(pooled_state))
        executed_action = self._executed_action(observation, action)

        if action_future is not None:
            future_action = _to_tensor(action_future, device=self.device, dtype=self.dtype)
            action_cond = future_action[0] if future_action.ndim > 1 else future_action
        else:
            action_cond = action
        pred_tail = torch.stack(
            [posterior.global_post, proprio_token, self.action_cond_proj(action_cond[None, :])[0]],
            dim=0,
        )
        pred_world_tokens = torch.cat([posterior.tokens, pred_tail], dim=0)
        # First produce a purely physical predictive basis. This is the only cache
        # allowed to feed the next-step innovation constructor.
        physical_pred_tokens = self.predictive_world(pred_world_tokens[None, :])[0]
        physical_global_pred = self.predictive_pool(physical_pred_tokens[None, :])[0]
        physical_prediction_cache = self._prediction_cache_from_global(physical_global_pred)
        # Semantic tokens act only as posterior-late conditioning memory for future
        # readout. They do not rewrite the physical predictive cache above.
        pred_tokens, _ = self._apply_semantic_reads(
            physical_pred_tokens,
            semantic.tokens,
            reads=self.predictive_semantic_reads,
            dropout_prob=self.config.predictive_semantic_dropout_prob,
        )
        global_pred = self.predictive_pool(pred_tokens[None, :])[0]
        prediction_cache = self._prediction_cache_from_global(global_pred)
        return PicfPredictiveState(
            semantic_tokens=semantic.tokens,
            semantic_summary=semantic.summary,
            innovation_token=innovation_token,
            innovation_norm=innovation_norm,
            availability=targets_availability,
            control_tokens=control_tokens,
            pooled_state=pooled_state,
            action=action,
            executed_action=executed_action,
            physical_global_pred=physical_global_pred,
            physical_prediction_cache=physical_prediction_cache,
            global_pred=global_pred,
            prediction_cache=prediction_cache,
        )

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
        if observation.G_t is None:
            observation.G_t = self.local_frame.make_transform(observation.robot_obs)
        if observation.point_set is None:
            observation.point_set = self.pointcloud_builder(
                {
                    "rgb_static": observation.rgb_static,
                    "depth_static": observation.depth_static,
                    "focus_center_world": np.asarray(observation.G_t[:3, 3], dtype=np.float32),
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
        token_field = self._build_token_field(observation, frame_context, point_features, visual_map, tactile_bundle, meta, previous)
        observation_anchors = self._build_observation_anchors(token_field)
        posterior = self._posterior_update(previous, observation, observation_anchors)
        current_targets, availability = self._current_targets(observation, frame_context, visual_map)
        innovation_token, innovation_norm = self._innovation(previous, current_targets, availability)
        predictive = self._predictive_state(
            observation,
            posterior,
            semantic,
            innovation_token,
            innovation_norm,
            availability,
            action_future,
        )
        hold_reason = self._hold_reason(meta, posterior, innovation_token)
        state = PicfCoreState(
            runtime_meta=meta,
            G_t=_to_tensor(observation.G_t, device=self.device, dtype=self.dtype),
            token_field=token_field,
            observation_anchors=observation_anchors,
            posterior=posterior,
            predictive=predictive,
            control=PicfControlState(hold_reason=hold_reason),
            last_prompt=observation.prompt,
        )
        debug = {
            "num_point_tokens": float(token_field.point_tokens.shape[0]),
            "num_visual_tokens": float(token_field.visual_tokens.shape[0]),
            "num_tactile_tokens": float(token_field.tactile_tokens.shape[0]),
            "support_mass_mean": float(posterior.support_mass.mean().item()),
            "active_alpha_sum": float(posterior.alpha.sum().item()),
            "innovation_norm": float(torch.linalg.norm(innovation_token).item()),
            "hold_triggered": 1.0 if hold_reason is not None else 0.0,
        }
        if observation_anchors.routing_gate_point.numel() > 0:
            debug["mean_point_route_gate"] = float(observation_anchors.routing_gate_point.mean().item())
            debug["mean_point_route_support"] = float(observation_anchors.routing_support_point.mean().item())
        if observation_anchors.routing_gate_visual.numel() > 0:
            debug["mean_visual_route_gate"] = float(observation_anchors.routing_gate_visual.mean().item())
            debug["mean_visual_route_support"] = float(observation_anchors.routing_support_visual.mean().item())
        if token_field.projective_geometry is not None:
            geom = token_field.projective_geometry
            num_edges = int(geom.projective_candidate_mask.sum().item()) if geom.projective_candidate_mask.numel() > 0 else 0
            total_edges = int(geom.projective_candidate_mask.numel())
            density = (float(num_edges) / float(total_edges)) if total_edges > 0 else 0.0
            debug["mean_point_visibility"] = float(geom.point_visibility.mean().item()) if geom.point_visibility.numel() > 0 else 0.0
            debug["projective_candidate_edges"] = float(num_edges)
            debug["projective_candidate_density"] = float(density)
        return PicfCoreOutput(state=state, debug=debug)

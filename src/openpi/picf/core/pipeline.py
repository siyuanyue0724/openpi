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
from openpi.picf.core.contracts import PicfAnchorPriorGraphState
from openpi.picf.core.contracts import PicfActiveProposalState
from openpi.picf.core.contracts import PicfCoreOutput
from openpi.picf.core.contracts import PicfCoreState
from openpi.picf.core.contracts import PicfEvidenceCacheState
from openpi.picf.core.contracts import PicfCacheReadState
from openpi.picf.core.contracts import PicfObservationAnchorState
from openpi.picf.core.contracts import PicfObjectExplanationState
from openpi.picf.core.contracts import PicfPosteriorAnchorState
from openpi.picf.core.contracts import PicfPredictionCache
from openpi.picf.core.contracts import PicfPreviousState
from openpi.picf.core.contracts import PicfPredictiveState
from openpi.picf.core.contracts import PicfProjectiveGeometryState
from openpi.picf.core.contracts import PicfPseudoProposalState
from openpi.picf.core.contracts import PicfRecurrentCarryState
from openpi.picf.core.contracts import PicfRecurrentPredictiveState
from openpi.picf.core.contracts import PicfTemporalVisualSupportState
from openpi.picf.core.contracts import PicfTrackletSupportState
from openpi.picf.core.contracts import PicfRecurrentTokenFieldState
from openpi.picf.core.contracts import PicfTaskReadoutState
from openpi.picf.core.contracts import PicfTokenFieldState
from openpi.picf.core.contracts import PicfVLGroundingState
from openpi.picf.core.tactile_contact import contact_prob_with_hysteresis
from openpi.picf.core.tactile_contact import summarize_contact_context
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.fsdp_utils import call_module_forward_or_method
from openpi.picf.geometry import transform_points
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


def _apply_tokenwise_in_chunks(
    x: torch.Tensor,
    fn_apply,
    *,
    chunk_size: int,
) -> torch.Tensor:
    chunk = int(chunk_size)
    if chunk <= 0 or x.ndim < 2 or x.shape[1] <= chunk:
        return fn_apply(x)
    outputs = []
    for start in range(0, int(x.shape[1]), chunk):
        end = min(start + chunk, int(x.shape[1]))
        outputs.append(fn_apply(x[:, start:end]))
    return torch.cat(outputs, dim=1)


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


def _weighted_anchor_modes(
    positions: torch.Tensor,
    weights: torch.Tensor,
    *,
    count: int,
    radius_m: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if positions.shape[0] == 0 or weights.numel() == 0 or count <= 0:
        return torch.zeros((0,), device=positions.device, dtype=torch.long)
    if weights.shape[0] != positions.shape[0]:
        raise RuntimeError(
            "VL weighted-anchor mode contract violated: "
            f"positions={tuple(positions.shape)} weights={tuple(weights.shape)}"
        )
    work = torch.clamp(weights.to(device=positions.device, dtype=positions.dtype), min=0.0).clone()
    chosen: list[int] = []
    radius = max(float(radius_m), eps)
    for _ in range(int(count)):
        max_value = torch.max(work) if work.numel() > 0 else work.new_tensor(0.0)
        if float(max_value.item()) <= eps:
            break
        idx = int(torch.argmax(work).item())
        chosen.append(idx)
        dist2 = torch.sum((positions - positions[idx][None, :]) ** 2, dim=-1)
        suppress = torch.exp(-dist2 / (2.0 * radius * radius))
        work = work * torch.clamp(1.0 - suppress, min=0.0, max=1.0)
    if not chosen:
        return torch.zeros((0,), device=positions.device, dtype=torch.long)
    return torch.as_tensor(chosen, device=positions.device, dtype=torch.long)


def _resize_flat_heatmap(
    heatmap: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    eps: float,
) -> torch.Tensor:
    heat = heatmap.reshape(1, 1, int(src_hw[0]), int(src_hw[1])).to(dtype=torch.float32)
    if tuple(src_hw) != tuple(dst_hw):
        heat = fn.interpolate(heat, size=(int(dst_hw[0]), int(dst_hw[1])), mode="bilinear", align_corners=False)
    flat = torch.clamp(heat.reshape(-1).to(device=heatmap.device, dtype=heatmap.dtype), min=0.0)
    return flat / torch.clamp(flat.sum(), min=eps)


def _map_pg_grid_values_to_visual_grid(
    values: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    view_transform: Any | None,
) -> torch.Tensor:
    """Map PaliGemma padded-image grid values back onto the PICF visual grid.

    PaliGemma receives images through resize-with-pad. A naive interpolation from
    the PaliGemma token grid to the V-JEPA/PICF grid treats padded pixels as real
    image content and can shift grounding toward borders. This helper samples the
    PaliGemma grid at the padded-image coordinates corresponding to each original
    camera pixel center used by PICF.
    """

    if view_transform is None:
        return _resize_flat_grid_values(values, src_hw=src_hw, dst_hw=dst_hw)
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return torch.zeros((max(dst_h * dst_w, 0),), device=values.device, dtype=values.dtype)

    try:
        original_h, original_w = tuple(int(v) for v in view_transform.original_hw)
        target_h, target_w = tuple(int(v) for v in view_transform.target_hw)
        pad_top = float(view_transform.pad_top)
        pad_left = float(view_transform.pad_left)
        scale_y = float(view_transform.scale_y)
        scale_x = float(view_transform.scale_x)
    except Exception:
        return _resize_flat_grid_values(values, src_hw=src_hw, dst_hw=dst_hw)
    if original_h <= 0 or original_w <= 0 or target_h <= 0 or target_w <= 0 or scale_y <= 0.0 or scale_x <= 0.0:
        return _resize_flat_grid_values(values, src_hw=src_hw, dst_hw=dst_hw)

    ys_idx, xs_idx = torch.meshgrid(
        torch.arange(dst_h, device=values.device, dtype=torch.float32),
        torch.arange(dst_w, device=values.device, dtype=torch.float32),
        indexing="ij",
    )
    x_orig = xs_idx * (float(original_w - 1) / max(dst_w - 1, 1))
    y_orig = ys_idx * (float(original_h - 1) / max(dst_h - 1, 1))

    # Match torch.interpolate(..., align_corners=False) pixel-center geometry.
    x_padded = (x_orig + 0.5) * scale_x - 0.5 + pad_left
    y_padded = (y_orig + 0.5) * scale_y - 0.5 + pad_top
    x_cell = (x_padded + 0.5) * (float(src_w) / float(target_w)) - 0.5
    y_cell = (y_padded + 0.5) * (float(src_h) / float(target_h)) - 0.5
    grid_x = ((x_cell + 0.5) * 2.0 / float(src_w)) - 1.0
    grid_y = ((y_cell + 0.5) * 2.0 / float(src_h)) - 1.0
    sample_grid = torch.stack([grid_x, grid_y], dim=-1)[None, :]

    grid_values = values.reshape(1, 1, src_h, src_w).to(device=values.device, dtype=torch.float32)
    sampled = fn.grid_sample(
        grid_values,
        sample_grid.to(device=values.device, dtype=torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.reshape(-1).to(device=values.device, dtype=values.dtype)


def _map_pg_heatmap_to_visual_grid(
    heatmap: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    view_transform: Any | None,
    eps: float,
) -> torch.Tensor:
    if view_transform is None:
        return _resize_flat_heatmap(heatmap, src_hw=src_hw, dst_hw=dst_hw, eps=eps)
    flat = _map_pg_grid_values_to_visual_grid(
        heatmap,
        src_hw=src_hw,
        dst_hw=dst_hw,
        view_transform=view_transform,
    )
    flat = torch.clamp(flat, min=0.0)
    return flat / torch.clamp(flat.sum(), min=eps)


def _resize_flat_grid_values(
    values: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    grid = values.reshape(1, 1, int(src_hw[0]), int(src_hw[1])).to(dtype=torch.float32)
    if tuple(src_hw) != tuple(dst_hw):
        grid = fn.interpolate(grid, size=(int(dst_hw[0]), int(dst_hw[1])), mode="bilinear", align_corners=False)
    return grid.reshape(-1).to(device=values.device, dtype=values.dtype)


def _point_prior_from_heatmap(
    projective_compatibility: torch.Tensor,
    heatmap: torch.Tensor,
    *,
    point_projectable_mask: torch.Tensor | None,
    min_visible_mass: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if projective_compatibility.ndim != 2:
        raise RuntimeError(f"Expected projective compatibility [P,V], got {tuple(projective_compatibility.shape)}")
    point_count, visual_count = projective_compatibility.shape
    if heatmap.shape != (visual_count,):
        raise RuntimeError(
            "VL heatmap/projective contract violated: "
            f"heatmap={tuple(heatmap.shape)} projective={tuple(projective_compatibility.shape)}"
        )
    if point_count == 0 or visual_count == 0:
        prior = projective_compatibility.new_zeros((point_count,))
        return prior, torch.tensor(False, device=projective_compatibility.device), projective_compatibility.new_zeros(())
    compat = torch.clamp(torch.nan_to_num(projective_compatibility, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    if point_projectable_mask is not None:
        if point_projectable_mask.shape != (point_count,):
            raise RuntimeError(
                "VL point-projectable mask contract violated: "
                f"mask={tuple(point_projectable_mask.shape)} point_count={point_count}"
            )
        compat = compat * point_projectable_mask.to(device=compat.device, dtype=compat.dtype)[:, None]
    heat = torch.clamp(heatmap.to(device=compat.device, dtype=compat.dtype), min=0.0)
    heat = heat / torch.clamp(heat.sum(), min=eps)
    column_mass = compat.sum(dim=0)
    visible_cols = column_mass > eps
    visible_mass = torch.sum(heat * visible_cols.to(dtype=heat.dtype))
    valid = visible_mass > float(min_visible_mass)
    if not bool(valid.item()):
        return compat.new_zeros((point_count,)), valid, visible_mass
    compat_col = compat / torch.clamp(column_mass[None, :], min=eps)
    prior = compat_col @ heat
    prior = torch.clamp(prior, min=0.0)
    prior = prior / torch.clamp(prior.sum(), min=eps)
    return prior, valid, visible_mass


def _normalize_rows(values: torch.Tensor, *, eps: float) -> torch.Tensor:
    values = torch.clamp(torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    if values.ndim == 1:
        return values / torch.clamp(values.sum(), min=eps)
    return values / torch.clamp(values.sum(dim=-1, keepdim=True), min=eps)


def _row_has_mass(values: torch.Tensor, *, eps: float) -> torch.Tensor:
    if values.ndim == 1:
        return values.sum()[None] > eps
    return values.sum(dim=-1) > eps


def _distribution_confidence(values: torch.Tensor | None, *, eps: float, floor: float = 0.0) -> torch.Tensor | None:
    if values is None:
        return None
    if values.ndim == 1:
        values = values[None, :]
    if values.numel() == 0:
        return torch.zeros((values.shape[0],), device=values.device, dtype=values.dtype)
    mass = torch.clamp(values.sum(dim=-1), min=0.0)
    valid = mass > eps
    probs = _normalize_rows(values, eps=eps)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=eps)), dim=-1)
    max_entropy = math.log(max(int(values.shape[-1]), 2))
    confidence = torch.clamp(1.0 - (entropy / max(max_entropy, eps)), min=0.0, max=1.0)
    if floor > 0.0:
        confidence = torch.where(valid, torch.clamp(confidence, min=float(floor)), confidence)
    return torch.where(valid, confidence, torch.zeros_like(confidence))


def _distribution_js(p: torch.Tensor, q: torch.Tensor, *, eps: float) -> torch.Tensor:
    p = _normalize_rows(p, eps=eps)
    q = _normalize_rows(q, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(torch.clamp(p, min=eps)) - torch.log(torch.clamp(m, min=eps))), dim=-1)
    kl_qm = torch.sum(q * (torch.log(torch.clamp(q, min=eps)) - torch.log(torch.clamp(m, min=eps))), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _frame_context_points_world(frame_context: PointFrameContext) -> np.ndarray:
    """Return row-aligned world-frame point positions for camera/tactile geometry."""
    points_local = np.asarray(frame_context.points_local, dtype=np.float32)
    points_world = getattr(frame_context, "points_world", None)
    if points_world is not None:
        points_world_arr = np.asarray(points_world, dtype=np.float32)
        if points_world_arr.shape == points_local.shape:
            return points_world_arr
    return transform_points(points_local, np.asarray(frame_context.G_t, dtype=np.float32))


def _numpy_fps_indices(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if points.shape[0] <= count:
        return np.arange(points.shape[0], dtype=np.int64)
    centroid = points.mean(axis=0, dtype=np.float32)
    dists = np.linalg.norm(points - centroid[None, :], axis=1)
    first = int(np.argmax(dists))
    chosen = [first]
    min_dist = np.linalg.norm(points - points[first : first + 1], axis=1)
    while len(chosen) < count:
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[next_idx : next_idx + 1], axis=1))
    return np.asarray(chosen, dtype=np.int64)


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
        pool_ids=np.zeros((int(keep.sum()),), dtype=np.int64),
        points_world=np.asarray(xyz_world[keep], dtype=np.float32),
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
    def __init__(self, hidden_dim: int, heads: int, layers: int, *, ff_chunk_size: int = 0):
        super().__init__()
        del layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff_chunk_size = int(ff_chunk_size)
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
        ff_in = self.norm2(x)
        x = x + _apply_tokenwise_in_chunks(ff_in, self.ff, chunk_size=self.ff_chunk_size)
        return x, attn_weights if return_attention else None


class TransformerStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        layers: int,
        *,
        activation_checkpointing: bool = False,
        ff_chunk_size: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            SelfAttentionBlock(hidden_dim, heads, 1, ff_chunk_size=ff_chunk_size) for _ in range(layers)
        )
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
        # PICF frequently batches token fields via lightweight views such as
        # `tokens[None, :]`, and FSDP can also hand stacks parameter-compatible
        # activations whose storage aliasing is not reliably exposed via
        # `tensor._base`. Residual attention blocks do not benefit from
        # preserving those aliases, so we materialize a fresh tensor once at
        # stack entry. This is mathematically exact and prevents autograd's
        # multi-view in-place checks from tripping inside LayerNorm/attention.
        x = x.clone()
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
    def __init__(self, hidden_dim: int, heads: int, *, ff_chunk_size: int = 0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff_chunk_size = int(ff_chunk_size)
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
        ff_in = self.norm(output)
        output = output + _apply_tokenwise_in_chunks(ff_in, self.ff, chunk_size=self.ff_chunk_size)
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
        self.ff_chunk_size = 0
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
        ff_in = self.ff_norm(output)
        output = output + (gate * _apply_tokenwise_in_chunks(ff_in, self.ff, chunk_size=self.ff_chunk_size))
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
        self.ff_chunk_size = 0
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
        ff_in = self.ff_norm(output)
        output = output + (gate * _apply_tokenwise_in_chunks(ff_in, self.ff, chunk_size=self.ff_chunk_size))
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
    summary: torch.Tensor | None = None
    image_tokens: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    image_token_ranges: tuple[tuple[int, int], ...] = ()
    image_grid_shapes: tuple[tuple[int, int], ...] = ()
    image_view_names: tuple[str, ...] = ()
    image_view_transforms: tuple[Any, ...] = ()


@dataclasses.dataclass(frozen=True)
class _StepDenseMemory:
    point_payload: torch.Tensor
    visual_payload: torch.Tensor
    tactile_group_tokens: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class _ObservedStepState:
    previous: PicfPreviousState | None
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
    vl_grounding: PicfVLGroundingState | None
    anchor_prior_graph: PicfAnchorPriorGraphState | None
    object_explanation: PicfObjectExplanationState | None
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
        self.clip_buffers: dict[str, VisualClipBuffer] = {}
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
            view_names = self._configured_vjepa_views()
            self.clip_buffers = {
                view_name: VisualClipBuffer(num_frames=self.visual_config.num_frames)
                for view_name in view_names
            }
            if "static" not in self.clip_buffers:
                self.clip_buffers["static"] = VisualClipBuffer(num_frames=self.visual_config.num_frames)
            self.clip_buffer = self.clip_buffers["static"]
        if self.tactile_encoder is None:
            self.tactile_encoder = AnyTouch2TactileEncoder(self.tactile_config) if self.tactile_config is not None else None

        hidden_dim = self.config.hidden_dim
        semantic_trunk_dim = self.config.semantic_dim
        heads = self.config.attention_heads
        self.modality_embedding = nn.Embedding(4, hidden_dim)
        self.point_token_proj = nn.LazyLinear(hidden_dim)
        self.visual_token_proj = nn.LazyLinear(hidden_dim)
        self.tracklet_token_proj = nn.LazyLinear(hidden_dim)
        self.proposal_token_proj = nn.LazyLinear(hidden_dim)
        self.tactile_token_proj = nn.LazyLinear(hidden_dim)
        self.tactile_patch_token_proj = nn.LazyLinear(hidden_dim)
        self.point_align_proj = nn.LazyLinear(hidden_dim)
        self.visual_align_proj = nn.LazyLinear(hidden_dim)
        self.tactile_align_proj = nn.LazyLinear(hidden_dim)
        self.binding_signature_proj = nn.LazyLinear(int(self.config.binding_signature_dim))
        binding_dim = max(int(self.config.binding_signature_dim), 1)
        binding_rank = max(int(getattr(self.config, "binding_low_rank_signature_rank", 16)), 1)
        self.binding_quadratic_diag = nn.Parameter(
            torch.full(
                (binding_dim,),
                math.sqrt(float(binding_dim)),
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.binding_low_rank_left = nn.Linear(binding_dim, binding_rank, bias=False, device=self.device, dtype=self.dtype)
        self.binding_low_rank_right = nn.Linear(binding_dim, binding_rank, bias=False, device=self.device, dtype=self.dtype)
        nn.init.orthogonal_(self.binding_low_rank_left.weight)
        with torch.no_grad():
            self.binding_low_rank_right.weight.copy_(self.binding_low_rank_left.weight)
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
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )

        self.obs_reader = CrossAttentionRead(
            hidden_dim,
            heads,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.obs_self = TransformerStack(
            hidden_dim,
            heads,
            1,
            activation_checkpointing=True,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )

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

        self.posterior_slot_hidden = nn.Parameter(
            torch.empty((self.config.persistent_anchors, self.config.posterior_hidden_dim), device=self.device, dtype=self.dtype)
        )
        self.posterior_slot_token = nn.Parameter(
            torch.empty((self.config.persistent_anchors, hidden_dim), device=self.device, dtype=self.dtype)
        )
        posterior_slot_std = float(self.config.posterior_slot_identity_std)
        if posterior_slot_std > 0.0:
            nn.init.normal_(self.posterior_slot_hidden, mean=0.0, std=posterior_slot_std)
            nn.init.normal_(self.posterior_slot_token, mean=0.0, std=posterior_slot_std)
        else:
            nn.init.zeros_(self.posterior_slot_hidden)
            nn.init.zeros_(self.posterior_slot_token)
        self.anchor_seed_proj = nn.LazyLinear(hidden_dim)
        self.anchor_reader = CrossAttentionRead(
            hidden_dim,
            heads,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
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
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.posterior_pool = AttentionPool(hidden_dim)

        self.semantic_prefix_proj = nn.Identity()
        self.proprio_proj = nn.LazyLinear(hidden_dim)
        self.action_cond_proj = nn.LazyLinear(hidden_dim)
        aqr_pg_grounding_enabled = bool(self.config.aqr_mapg_enabled) and bool(
            self.config.aqr_pg_grounding_enabled
        )
        shared_pg_grounding_enabled = (
            bool(self.config.vl_anchor_router_enabled)
            or bool(self.config.mapg_enabled)
            or aqr_pg_grounding_enabled
        )
        if shared_pg_grounding_enabled:
            self.vl_heatmap_head = nn.Sequential(
                nn.LazyLinear(int(self.config.vl_heatmap_hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(self.config.vl_heatmap_hidden_dim)),
                nn.Linear(int(self.config.vl_heatmap_hidden_dim), 3),
            )
            self.vl_anchor_token_proj = nn.LazyLinear(hidden_dim)
        else:
            self.vl_heatmap_head = None
            self.vl_anchor_token_proj = None
        if bool(self.config.vl_anchor_router_enabled):
            self.vl_task_point_gate_logit = nn.Parameter(
                torch.tensor([float(self.config.vl_task_point_gate_init)], device=self.device, dtype=self.dtype)
            )
            self.vl_obs_anchor_gate_logit = nn.Parameter(
                torch.tensor([float(self.config.vl_obs_anchor_gate_init)], device=self.device, dtype=self.dtype)
            )
            self.vl_posterior_bind_gate_logit = nn.Parameter(
                torch.tensor([float(self.config.vl_posterior_bind_gate_init)], device=self.device, dtype=self.dtype)
            )
        else:
            self.vl_task_point_gate_logit = None
            self.vl_obs_anchor_gate_logit = None
            self.vl_posterior_bind_gate_logit = None
        legacy_mapg_enabled = bool(self.config.mapg_enabled)
        graph_router_enabled = legacy_mapg_enabled or bool(self.config.aqr_mapg_enabled)
        if legacy_mapg_enabled:
            self.mapg_pg_proj = nn.LazyLinear(hidden_dim)
            self.mapg_visual_proj = nn.Linear(hidden_dim, hidden_dim)
            self.mapg_point_proj = nn.Linear(hidden_dim, hidden_dim)
            self.mapg_tactile_proj = nn.Linear(hidden_dim, hidden_dim)
            self.mapg_posterior_proj = nn.Linear(hidden_dim, hidden_dim)
            self.mapg_anchor_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 5 + 5, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.mapg_anchor_fusion[-1].weight)
            nn.init.zeros_(self.mapg_anchor_fusion[-1].bias)
            self.mapg_role_embedding = nn.Embedding(4, hidden_dim)
        else:
            self.mapg_pg_proj = None
            self.mapg_visual_proj = None
            self.mapg_point_proj = None
            self.mapg_tactile_proj = None
            self.mapg_posterior_proj = None
            self.mapg_anchor_fusion = None
            self.mapg_role_embedding = None
        if graph_router_enabled:
            self.mapg_task_visual_proj = nn.Linear(hidden_dim, hidden_dim)
            self.mapg_to_control_proj = nn.Linear(hidden_dim, semantic_trunk_dim)
            self.mapg_control_role_embedding = nn.Embedding(1, semantic_trunk_dim)
            obs_gate_init = float(self.config.aqr_obs_gate_init) if bool(self.config.aqr_mapg_enabled) else float(self.config.mapg_obs_gate_init)
            task_gate_init = float(self.config.aqr_task_gate_init) if bool(self.config.aqr_mapg_enabled) else float(self.config.mapg_task_gate_init)
            posterior_gate_init = float(self.config.aqr_posterior_gate_init) if bool(self.config.aqr_mapg_enabled) else float(self.config.mapg_posterior_gate_init)
            control_gate_init = float(self.config.aqr_control_gate_init) if bool(self.config.aqr_mapg_enabled) else float(self.config.mapg_control_gate_init)
            self.mapg_obs_gate_logit = nn.Parameter(
                torch.tensor([obs_gate_init], device=self.device, dtype=self.dtype)
            )
            self.mapg_task_gate_logit = nn.Parameter(
                torch.tensor([task_gate_init], device=self.device, dtype=self.dtype)
            )
            self.mapg_posterior_gate_logit = nn.Parameter(
                torch.tensor([posterior_gate_init], device=self.device, dtype=self.dtype)
            )
            self.mapg_control_gate_logit = nn.Parameter(
                torch.tensor([control_gate_init], device=self.device, dtype=self.dtype)
            )
        else:
            self.mapg_task_visual_proj = None
            self.mapg_to_control_proj = None
            self.mapg_control_role_embedding = None
            self.mapg_obs_gate_logit = None
            self.mapg_task_gate_logit = None
            self.mapg_posterior_gate_logit = None
            self.mapg_control_gate_logit = None
        if bool(self.config.aqr_mapg_enabled):
            self.aqr_physical_query_tokens = nn.Parameter(
                torch.empty((max(int(self.config.aqr_query_count_physical), 0), hidden_dim), device=self.device, dtype=self.dtype)
            )
            self.aqr_task_query_tokens = nn.Parameter(
                torch.empty((max(int(self.config.aqr_query_count_task), 0), hidden_dim), device=self.device, dtype=self.dtype)
            )
            nn.init.normal_(self.aqr_physical_query_tokens, mean=0.0, std=0.02)
            nn.init.normal_(self.aqr_task_query_tokens, mean=0.0, std=0.02)
            self.aqr_role_embedding = nn.Embedding(4, hidden_dim)
            self.aqr_type_embedding = nn.Embedding(2, hidden_dim)
            self.aqr_coverage_proj = nn.Linear(2, hidden_dim)
            self.aqr_proprio_proj = nn.LazyLinear(hidden_dim)
            self.aqr_posterior_summary_proj = nn.LazyLinear(hidden_dim)
            self.aqr_task_conditioner = GatedCrossAttentionRead(
                hidden_dim,
                semantic_trunk_dim,
                heads,
                inner_dim=max(self.config.semantic_cross_dim, hidden_dim),
                gate_init=1.0,
            )
            self.aqr_task_conditioner.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
            self.aqr_pg_image_reader = GatedCrossAttentionRead(
                hidden_dim,
                semantic_trunk_dim,
                heads,
                inner_dim=max(self.config.semantic_cross_dim, hidden_dim),
                gate_init=0.5,
            )
            self.aqr_pg_image_reader.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
            self.aqr_visual_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_temporal_visual_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_point_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_tactile_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_posterior_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_cache_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_tracklet_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_proposal_reader = CrossAttentionRead(
                hidden_dim,
                heads,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            self.aqr_query_self = TransformerStack(
                hidden_dim,
                heads,
                1,
                activation_checkpointing=True,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
            if bool(getattr(self.config, "vcap_enabled", False)):
                self.vcap_start_token = nn.Parameter(torch.empty((hidden_dim,), device=self.device, dtype=self.dtype))
                self.vcap_reserve_token = nn.Parameter(torch.empty((hidden_dim,), device=self.device, dtype=self.dtype))
                nn.init.normal_(self.vcap_start_token, mean=0.0, std=0.02)
                nn.init.normal_(self.vcap_reserve_token, mean=0.0, std=0.02)
                self.vcap_summary_proj = nn.Linear(hidden_dim, hidden_dim)
                self.vcap_decoder = nn.GRUCell(hidden_dim, hidden_dim)
                self.vcap_query_head = nn.Linear(hidden_dim, hidden_dim)
                self.vcap_address_head = nn.Linear(hidden_dim, hidden_dim)
                self.vcap_geometry_head = nn.Linear(hidden_dim, 3)
                self.vcap_role_head = nn.Linear(hidden_dim, 4)
                self.vcap_stop_head = nn.Linear(hidden_dim, 1)
                self.vcap_support_head = nn.Linear(hidden_dim, hidden_dim)
            else:
                self.vcap_start_token = None
                self.vcap_reserve_token = None
                self.vcap_summary_proj = None
                self.vcap_decoder = None
                self.vcap_query_head = None
                self.vcap_address_head = None
                self.vcap_geometry_head = None
                self.vcap_role_head = None
                self.vcap_stop_head = None
                self.vcap_support_head = None
        else:
            self.aqr_physical_query_tokens = None
            self.aqr_task_query_tokens = None
            self.aqr_role_embedding = None
            self.aqr_type_embedding = None
            self.aqr_coverage_proj = None
            self.aqr_proprio_proj = None
            self.aqr_posterior_summary_proj = None
            self.aqr_task_conditioner = None
            self.aqr_pg_image_reader = None
            self.aqr_visual_reader = None
            self.aqr_temporal_visual_reader = None
            self.aqr_point_reader = None
            self.aqr_tactile_reader = None
            self.aqr_posterior_reader = None
            self.aqr_cache_reader = None
            self.aqr_tracklet_reader = None
            self.aqr_proposal_reader = None
            self.aqr_query_self = None
            self.vcap_start_token = None
            self.vcap_reserve_token = None
            self.vcap_summary_proj = None
            self.vcap_decoder = None
            self.vcap_query_head = None
            self.vcap_address_head = None
            self.vcap_geometry_head = None
            self.vcap_role_head = None
            self.vcap_stop_head = None
            self.vcap_support_head = None
        self.temporal_visual_time_proj = nn.Linear(1, hidden_dim)
        self.temporal_visual_view_embedding = nn.Embedding(max(int(self.config.vjepa_max_views), 1), hidden_dim)
        self.task_query_tokens = nn.Parameter(torch.empty((self.config.task_local_queries, hidden_dim)))
        self.task_global_query_tokens = nn.Parameter(torch.empty((self.config.task_global_queries, hidden_dim)))
        self.task_instruction_query_tokens = nn.Parameter(torch.empty((self.config.task_instruction_queries, hidden_dim)))
        task_slot_std = float(self.config.task_slot_identity_std)
        if task_slot_std > 0.0:
            nn.init.normal_(self.task_query_tokens, mean=0.0, std=task_slot_std)
            nn.init.normal_(self.task_global_query_tokens, mean=0.0, std=task_slot_std)
            nn.init.normal_(self.task_instruction_query_tokens, mean=0.0, std=task_slot_std)
        else:
            nn.init.zeros_(self.task_query_tokens)
            nn.init.zeros_(self.task_global_query_tokens)
            nn.init.zeros_(self.task_instruction_query_tokens)
        self.task_query_conditioner = GatedCrossAttentionRead(
            hidden_dim,
            semantic_trunk_dim,
            heads,
            inner_dim=max(self.config.semantic_cross_dim, hidden_dim),
            gate_init=1.0,
        )
        self.task_query_conditioner.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.task_public_reader = CrossAttentionRead(
            hidden_dim,
            heads,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.task_visual_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.task_tactile_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.task_point_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.task_visual_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.task_tactile_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.task_point_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.task_self = TransformerStack(
            hidden_dim,
            heads,
            self.config.task_self_layers,
            activation_checkpointing=True,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
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
        self.pi_prefix_reader = CrossAttentionRead(
            semantic_trunk_dim,
            heads,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.future_condition_reader = CrossAttentionRead(
            semantic_trunk_dim,
            heads,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.predictive_world = TransformerStack(
            hidden_dim,
            heads,
            self.config.predictive_layers,
            activation_checkpointing=True,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.predictive_semantic_world = TransformerStack(
            semantic_trunk_dim,
            heads,
            self.config.predictive_layers,
            activation_checkpointing=True,
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.predictive_pool = AttentionPool(hidden_dim)

        self.visual_latent_queries = nn.Parameter(torch.zeros((self.config.visual_latent_tokens, hidden_dim)))
        self.tactile_latent_queries = nn.Parameter(torch.zeros((self.config.tactile_latent_tokens, hidden_dim)))
        self.point_latent_queries = nn.Parameter(torch.zeros((self.config.point_latent_tokens, hidden_dim)))
        self.visual_latent_head = nn.Linear(hidden_dim, self.config.visual_latent_dim)
        self.visual_real_head = nn.Linear(hidden_dim, self.config.visual_real_dim)
        self.tactile_real_head = nn.Linear(hidden_dim, self.config.tactile_real_dim)
        self.point_real_head = nn.Linear(hidden_dim, self.config.point_real_dim)
        self.slot_support_pred_head = nn.Linear(hidden_dim, 4)

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
        self.visual_native_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.tactile_native_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.point_native_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
        self.tactile_group_route_queries = nn.Parameter(torch.zeros((self.config.tactile_group_proposals, hidden_dim)))
        self.tactile_route_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)
        self.tactile_route_reread.ff_chunk_size = int(self.config.tokenwise_ff_chunk_size)
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
            ff_chunk_size=self.config.tokenwise_ff_chunk_size,
        )
        self.control_state_proj = nn.Linear(semantic_trunk_dim, self.config.control_dim)
        self.predictive_state_proj = nn.Linear(semantic_trunk_dim, hidden_dim)
        self.to(device=self.device, dtype=self.dtype)

    def _configured_vjepa_views(self) -> tuple[str, ...]:
        raw_views = tuple(str(view).strip() for view in getattr(self.config, "vjepa_views", ("static",)) if str(view).strip())
        if not bool(getattr(self.config, "vjepa_multiview_enabled", True)):
            raw_views = ("static",)
        views: list[str] = []
        for view in raw_views:
            if view not in views:
                views.append(view)
        if "static" not in views:
            views.insert(0, "static")
        return tuple(views)

    def _observation_rgb_for_view(self, observation: PicfObservation, view_name: str) -> np.ndarray | None:
        if view_name == "static":
            return np.asarray(observation.rgb_static)
        if view_name in ("gripper", "wrist"):
            return None if observation.rgb_gripper is None else np.asarray(observation.rgb_gripper)
        return None

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

    def _point_context_with_global_scene(self, observation: PicfObservation, local_context: PointFrameContext) -> PointFrameContext:
        """Return a unified point context with local effector points followed by global scene points.

        The physical/contact path still has explicit local points. Scene/object
        anchors and task slots need a separate global_scene_context candidate
        pool so their geometry is not forced to collapse into the gripper crop.
        """
        assert observation.point_set is not None
        cap = int(self.config.global_scene_point_cap)
        if cap <= 0:
            return local_context
        xyz_world = np.asarray(observation.point_set.xyz_world, dtype=np.float32)
        if xyz_world.shape[0] == 0:
            return local_context
        local_mask = np.asarray(local_context.local_mask, dtype=bool).reshape(-1)
        if local_mask.shape[0] != xyz_world.shape[0]:
            local_mask = np.zeros((xyz_world.shape[0],), dtype=bool)
        # The scene/object pool is sampled from the whole frame, not from the
        # complement of the effector crop. When the gripper is already near the
        # task object, excluding the local crop would incorrectly hide that
        # object from object-role anchors.
        selected_rel = _numpy_fps_indices(xyz_world, min(cap, int(xyz_world.shape[0])))
        scene_indices = selected_rel
        grid = np.concatenate(
            [
                np.asarray(local_context.grid_coord, dtype=np.int32),
                np.asarray(observation.point_set.grid_coord[scene_indices], dtype=np.int32),
            ],
            axis=0,
        )
        points = np.concatenate(
            [
                np.asarray(local_context.points_local, dtype=np.float32),
                np.asarray(observation.point_set.xyz_world[scene_indices], dtype=np.float32),
            ],
            axis=0,
        )
        points_world = np.concatenate(
            [
                _frame_context_points_world(local_context),
                np.asarray(observation.point_set.xyz_world[scene_indices], dtype=np.float32),
            ],
            axis=0,
        )
        normals = np.concatenate(
            [
                np.asarray(local_context.normals_local, dtype=np.float32),
                np.asarray(observation.point_set.normal_world[scene_indices], dtype=np.float32),
            ],
            axis=0,
        )
        colors = np.concatenate(
            [
                np.asarray(local_context.colors, dtype=np.float32),
                np.asarray(observation.point_set.rgb[scene_indices], dtype=np.float32),
            ],
            axis=0,
        )
        selected_mask = local_mask.copy()
        selected_mask[scene_indices] = True
        pool_ids = np.concatenate(
            [
                np.zeros((int(local_context.points_local.shape[0]),), dtype=np.int64),
                np.ones((int(scene_indices.shape[0]),), dtype=np.int64),
            ],
            axis=0,
        )
        return PointFrameContext(
            grid_coord=grid,
            points_local=points,
            normals_local=normals,
            colors=colors,
            local_mask=selected_mask,
            world_to_local=local_context.world_to_local,
            G_t=local_context.G_t,
            pool_ids=pool_ids,
            points_world=points_world,
        )

    def _extract_point_features(self, frame_context: PointFrameContext, override: torch.Tensor | np.ndarray | None) -> torch.Tensor:
        if override is not None:
            feature = _to_tensor(override, device=self.device, dtype=self.dtype)
            return feature if feature.ndim == 2 else feature.squeeze(0)
        if self.point_feature_extractor is None:
            return _to_tensor(frame_context.colors, device=self.device, dtype=self.dtype)
        encoded = call_module_forward_or_method(self.point_feature_extractor, "encode_local_context", frame_context)
        return _to_tensor(encoded.features, device=self.device, dtype=self.dtype)

    def _visual_maps(
        self,
        observation: PicfObservation,
        override: torch.Tensor | np.ndarray | None,
        meta: RuntimeMeta,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if override is not None:
            visual = _to_tensor(override, device=self.device, dtype=self.dtype)
            if visual.ndim == 5 and visual.shape[0] == 1:
                visual = visual.squeeze(0)
            if visual.ndim == 4:
                return visual[-1], visual
            visual_map = visual if visual.ndim == 3 else visual.squeeze(0)
            return visual_map, None
        if self.visual_encoder is None or self.visual_config is None or not self.clip_buffers:
            return None, None
        temporal_maps: list[torch.Tensor] = []
        current = None
        temporal_mode = str(self.config.aqr_vjepa_temporal_mode)
        recent_count = max(int(self.config.aqr_vjepa_temporal_tokens), 1)
        if temporal_mode == "last_only":
            recent_count = 1
        elif temporal_mode in ("last_two_tokens", "last_mean_delta"):
            recent_count = max(recent_count, 2)
        elif temporal_mode == "last4_tokens":
            recent_count = max(recent_count, 4)
        reference_shape: tuple[int, ...] | None = None
        reference_recent_count: int | None = None
        for view_name in self._configured_vjepa_views():
            buffer = self.clip_buffers.get(view_name)
            rgb = self._observation_rgb_for_view(observation, view_name)
            if buffer is None or rgb is None:
                continue
            rgb = np.asarray(rgb)
            view_available = bool(rgb.size > 0 and np.isfinite(rgb).all())
            if view_name == "static":
                view_available = view_available and bool(meta.visual_available)
            if view_available:
                buffer.push(rgb, segment_id=int(observation.segment_id), reset=bool(observation.reset_scaffold))
            if not buffer.has_frames:
                continue
            fmap = call_module_forward_or_method(self.visual_encoder, "encode_clip", buffer.get_clip())
            current_map = _to_tensor(
                fmap.current_map(use_last_two_mean=self.visual_config.use_last_two_mean),
                device=self.device,
                dtype=self.dtype,
            )
            if view_name == "static":
                current = current_map
            if temporal_mode != "disabled" and hasattr(fmap, "recent_maps"):
                recent = _to_tensor(fmap.recent_maps(n=recent_count), device=self.device, dtype=self.dtype)
                if recent.ndim == 3:
                    recent = recent[None, :]
                if recent.ndim != 4:
                    continue
                shape = tuple(int(dim) for dim in recent.shape[1:])
                if reference_shape is None:
                    reference_shape = shape
                    reference_recent_count = int(recent.shape[0])
                if shape != reference_shape:
                    # Do not force a mismatched wrist feature map into the static grid.
                    continue
                if reference_recent_count is not None and int(recent.shape[0]) != reference_recent_count:
                    continue
                temporal_maps.append(recent)
        if current is None:
            static_buffer = self.clip_buffers.get("static")
            if static_buffer is None or not static_buffer.has_frames:
                return None, None
            fmap = call_module_forward_or_method(self.visual_encoder, "encode_clip", static_buffer.get_clip())
            current = _to_tensor(fmap.current_map(use_last_two_mean=self.visual_config.use_last_two_mean), device=self.device, dtype=self.dtype)
        temporal = torch.stack(temporal_maps, dim=0) if len(temporal_maps) > 1 else (temporal_maps[0] if temporal_maps else None)
        return current, temporal

    def _visual_map(self, observation: PicfObservation, override: torch.Tensor | np.ndarray | None, meta: RuntimeMeta) -> torch.Tensor | None:
        visual_map, _ = self._visual_maps(observation, override, meta)
        return visual_map

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
        features: PaliGemmaSemanticFeatures | None = None,
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
        summary = None
        image_tokens = None
        text_tokens = None
        image_token_ranges: tuple[tuple[int, int], ...] = ()
        image_grid_shapes: tuple[tuple[int, int], ...] = ()
        image_view_names: tuple[str, ...] = ()
        image_view_transforms: tuple[Any, ...] = ()
        if features is not None:
            if features.summary is not None:
                summary = _to_tensor(features.summary, device=self.device, dtype=self.dtype)
                if summary.ndim > 1:
                    summary = summary.reshape(-1, summary.shape[-1]).mean(dim=0)
            if features.image_tokens is not None:
                image_tokens = _to_tensor(features.image_tokens, device=self.device, dtype=self.dtype)
            if features.text_tokens is not None:
                text_tokens = _to_tensor(features.text_tokens, device=self.device, dtype=self.dtype)
            image_token_ranges = tuple(features.image_token_ranges)
            image_grid_shapes = tuple(features.image_grid_shapes)
            image_view_names = tuple(features.image_view_names)
            image_view_transforms = tuple(features.image_view_transforms)
        return _SemanticContext(
            tokens=semantic_tokens,
            prefix_tokens=semantic_prefix_tokens,
            available=True,
            summary=summary,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            image_token_ranges=image_token_ranges,
            image_grid_shapes=image_grid_shapes,
            image_view_names=image_view_names,
            image_view_transforms=image_view_transforms,
        )

    def _semantic_context(
        self,
        observation: PicfObservation,
        previous: PicfPreviousState | None,
        semantic_override: Any | None,
    ) -> _SemanticContext:
        if not self.config.language_enabled:
            return self._zero_semantic_context()
        if semantic_override is None:
            return self._zero_semantic_context()
        if isinstance(semantic_override, PaliGemmaSemanticFeatures):
            return self._project_semantic_context(tokens_raw=semantic_override.tokens, features=semantic_override)
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

    def _build_vl_grounding(
        self,
        *,
        semantic: _SemanticContext,
        token_field: PicfTokenFieldState,
    ) -> PicfVLGroundingState | None:
        aqr_pg_grounding_enabled = bool(self.config.aqr_mapg_enabled) and bool(
            self.config.aqr_pg_grounding_enabled
        )
        if not (
            bool(self.config.vl_anchor_router_enabled)
            or bool(self.config.mapg_enabled)
            or aqr_pg_grounding_enabled
        ):
            return None
        if self.vl_heatmap_head is None or self.vl_anchor_token_proj is None:
            return None
        geometry = token_field.projective_geometry
        if (
            geometry is None
            or semantic.image_tokens is None
            or semantic.image_tokens.numel() == 0
            or not semantic.image_token_ranges
            or not semantic.image_grid_shapes
        ):
            return None

        view_names = tuple(semantic.image_view_names)
        view_index = 0
        requested_view = str(self.config.vl_grounding_view)
        if view_names and requested_view in view_names:
            view_index = view_names.index(requested_view)
        start, end = semantic.image_token_ranges[view_index]
        src_hw = semantic.image_grid_shapes[view_index]
        view_transform = (
            semantic.image_view_transforms[view_index]
            if view_index < len(semantic.image_view_transforms)
            else None
        )
        img_tokens = semantic.image_tokens[start:end]
        if img_tokens.shape[0] != int(src_hw[0] * src_hw[1]):
            raise RuntimeError(
                "VL grounding image-token/grid contract violated: "
                f"tokens={tuple(img_tokens.shape)} src_hw={src_hw}"
            )
        if semantic.text_tokens is not None and semantic.text_tokens.numel() > 0:
            text_summary = semantic.text_tokens.reshape(-1, semantic.text_tokens.shape[-1]).mean(dim=0)
        elif semantic.summary is not None:
            text_summary = semantic.summary.reshape(-1)
        elif semantic.tokens.numel() > 0:
            text_summary = semantic.tokens.mean(dim=0)
        else:
            return None

        visual_grid = geometry.visual_grid_index
        if visual_grid.numel() == 0:
            return None
        dst_w = int(torch.max(visual_grid[:, 0]).item()) + 1
        dst_h = int(torch.max(visual_grid[:, 1]).item()) + 1
        dst_hw = (dst_h, dst_w)

        text = text_summary.to(device=img_tokens.device, dtype=img_tokens.dtype)
        head_input = torch.cat([img_tokens, text[None, :].expand(img_tokens.shape[0], -1)], dim=-1)
        logits = self.vl_heatmap_head(head_input)
        temperature = max(float(self.config.vl_heatmap_temperature), 1e-6)
        task_logits_pg = logits[:, 0]
        eff_logits_pg = logits[:, 1]
        int_logits_pg = logits[:, 2]
        task_heat_pg = torch.softmax((task_logits_pg / temperature).float(), dim=0).to(dtype=self.dtype)
        eff_heat_pg = torch.softmax((eff_logits_pg / temperature).float(), dim=0).to(dtype=self.dtype)
        int_heat_pg = torch.softmax((int_logits_pg / temperature).float(), dim=0).to(dtype=self.dtype)
        task_heat = _map_pg_heatmap_to_visual_grid(
            task_heat_pg,
            src_hw=src_hw,
            dst_hw=dst_hw,
            view_transform=view_transform,
            eps=self.config.epsilon_a,
        )
        eff_heat = _map_pg_heatmap_to_visual_grid(
            eff_heat_pg,
            src_hw=src_hw,
            dst_hw=dst_hw,
            view_transform=view_transform,
            eps=self.config.epsilon_a,
        )
        int_heat = _map_pg_heatmap_to_visual_grid(
            int_heat_pg,
            src_hw=src_hw,
            dst_hw=dst_hw,
            view_transform=view_transform,
            eps=self.config.epsilon_a,
        )

        point_count = int(token_field.point_positions.shape[0])
        compat = geometry.projective_compatibility
        can_lift_to_point = bool(compat.shape == (point_count, int(task_heat.shape[0])) and compat.numel() > 0 and point_count > 0)
        if can_lift_to_point:
            projectable_mask = None
            if token_field.point_pool_ids is not None and token_field.point_pool_ids.numel() == point_count:
                projectable_mask = token_field.point_pool_ids.to(device=self.device) == 1
            # VL task/interaction lift is allowed to no-op. Do not use the global
            # fallback here: fallback rows are only for coverage seeding, not for
            # converting language-conditioned heatmaps into physical support.
            scene_projectable_mask = self._scene_point_candidate_mask(token_field, fallback_to_global=False)
            task_prior, task_valid, task_mass = _point_prior_from_heatmap(
                compat,
                task_heat,
                point_projectable_mask=scene_projectable_mask,
                min_visible_mass=float(self.config.vl_min_visible_mass),
                eps=self.config.epsilon_a,
            )
            eff_prior, eff_valid, eff_mass = _point_prior_from_heatmap(
                compat,
                eff_heat,
                point_projectable_mask=projectable_mask,
                min_visible_mass=float(self.config.vl_min_visible_mass),
                eps=self.config.epsilon_a,
            )
            int_prior, int_valid, int_mass = _point_prior_from_heatmap(
                compat,
                int_heat,
                point_projectable_mask=scene_projectable_mask,
                min_visible_mass=float(self.config.vl_min_visible_mass),
                eps=self.config.epsilon_a,
            )
        else:
            # MAPG can consume language-conditioned visual heatmaps without point
            # support. Point-centric router consumers remain disabled because the
            # lifted point priors are explicit no-ops.
            task_prior = torch.zeros((point_count,), device=self.device, dtype=self.dtype)
            eff_prior = torch.zeros((point_count,), device=self.device, dtype=self.dtype)
            int_prior = torch.zeros((point_count,), device=self.device, dtype=self.dtype)
            task_valid = torch.zeros((), device=self.device, dtype=torch.bool)
            eff_valid = torch.zeros((), device=self.device, dtype=torch.bool)
            int_valid = torch.zeros((), device=self.device, dtype=torch.bool)
            task_mass = torch.zeros((), device=self.device, dtype=self.dtype)
            eff_mass = torch.zeros((), device=self.device, dtype=self.dtype)
            int_mass = torch.zeros((), device=self.device, dtype=self.dtype)
        valid = task_valid | eff_valid | int_valid
        if not bool(valid.item()):
            hidden = int(self.config.hidden_dim)
            empty_cov = torch.zeros((0, 3, 3), device=self.device, dtype=self.dtype)
            return PicfVLGroundingState(
                task_heatmap_logits=_map_pg_grid_values_to_visual_grid(
                    task_logits_pg.detach().to(dtype=self.dtype),
                    src_hw=src_hw,
                    dst_hw=dst_hw,
                    view_transform=view_transform,
                ),
                effector_heatmap_logits=_map_pg_grid_values_to_visual_grid(
                    eff_logits_pg.detach().to(dtype=self.dtype),
                    src_hw=src_hw,
                    dst_hw=dst_hw,
                    view_transform=view_transform,
                ),
                interaction_heatmap_logits=_map_pg_grid_values_to_visual_grid(
                    int_logits_pg.detach().to(dtype=self.dtype),
                    src_hw=src_hw,
                    dst_hw=dst_hw,
                    view_transform=view_transform,
                ),
                task_heatmap=task_heat,
                effector_heatmap=eff_heat,
                interaction_heatmap=int_heat,
                task_point_prior=torch.zeros((point_count,), device=self.device, dtype=self.dtype),
                effector_point_prior=torch.zeros((point_count,), device=self.device, dtype=self.dtype),
                interaction_point_prior=torch.zeros((point_count,), device=self.device, dtype=self.dtype),
                anchor_point_priors=torch.zeros((0, point_count), device=self.device, dtype=self.dtype),
                anchor_x=torch.zeros((0, 3), device=self.device, dtype=self.dtype),
                anchor_S=empty_cov,
                anchor_tokens=torch.zeros((0, hidden), device=self.device, dtype=self.dtype),
                anchor_roles=torch.zeros((0,), device=self.device, dtype=torch.long),
                anchor_scores=torch.zeros((0,), device=self.device, dtype=self.dtype),
                visual_pixel_centers=geometry.visual_pixel_centers,
                valid=valid,
                confidence=torch.stack([task_mass, eff_mass, int_mass]).max(),
                task_pg_heatmap=task_heat_pg,
                effector_pg_heatmap=eff_heat_pg,
                interaction_pg_heatmap=int_heat_pg,
            )

        anchor_priors: list[torch.Tensor] = []
        anchor_roles: list[int] = []
        per_role_budget = max(int(self.config.vl_anchor_modes), 0)
        eff_budget = min(1, per_role_budget)
        remaining = max(per_role_budget - eff_budget, 0)
        task_budget = remaining // 2
        int_budget = remaining - task_budget
        role_specs = (
            (0, eff_prior, eff_budget),
            (1, task_prior, task_budget),
            (2, int_prior, int_budget),
        )
        point_positions_world = self._world_point_positions(token_field)
        for role, prior, count in role_specs:
            modes = _weighted_anchor_modes(
                point_positions_world,
                prior,
                count=count,
                radius_m=float(self.config.vl_anchor_nms_radius_m),
                eps=self.config.epsilon_a,
            )
            for idx in modes.tolist():
                center = point_positions_world[int(idx)]
                dist2 = torch.sum((point_positions_world - center[None, :]) ** 2, dim=-1)
                sigma = max(float(self.config.vl_anchor_local_sigma_m), self.config.epsilon_a)
                local = torch.exp(-dist2 / (2.0 * sigma * sigma)) * torch.clamp(prior, min=0.0)
                local = local / torch.clamp(local.sum(), min=self.config.epsilon_a)
                anchor_priors.append(local)
                anchor_roles.append(int(role))
        if anchor_priors:
            anchor_point_priors = torch.stack(anchor_priors, dim=0)
            anchor_x = anchor_point_priors @ point_positions_world
            anchor_S = _weighted_cov(point_positions_world, anchor_point_priors, anchor_x, self.config)
            anchor_tokens_raw = anchor_point_priors @ token_field.point_tokens
            anchor_tokens = self.vl_anchor_token_proj(anchor_tokens_raw)
            anchor_scores = torch.max(anchor_point_priors, dim=-1).values
            anchor_roles_t = torch.as_tensor(anchor_roles, device=self.device, dtype=torch.long)
        else:
            point_count = int(token_field.point_positions.shape[0])
            hidden = int(self.config.hidden_dim)
            anchor_point_priors = torch.zeros((0, point_count), device=self.device, dtype=self.dtype)
            anchor_x = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
            anchor_S = torch.zeros((0, 3, 3), device=self.device, dtype=self.dtype)
            anchor_tokens = torch.zeros((0, hidden), device=self.device, dtype=self.dtype)
            anchor_scores = torch.zeros((0,), device=self.device, dtype=self.dtype)
            anchor_roles_t = torch.zeros((0,), device=self.device, dtype=torch.long)

        return PicfVLGroundingState(
            task_heatmap_logits=_map_pg_grid_values_to_visual_grid(
                task_logits_pg.to(dtype=self.dtype),
                src_hw=src_hw,
                dst_hw=dst_hw,
                view_transform=view_transform,
            ),
            effector_heatmap_logits=_map_pg_grid_values_to_visual_grid(
                eff_logits_pg.to(dtype=self.dtype),
                src_hw=src_hw,
                dst_hw=dst_hw,
                view_transform=view_transform,
            ),
            interaction_heatmap_logits=_map_pg_grid_values_to_visual_grid(
                int_logits_pg.to(dtype=self.dtype),
                src_hw=src_hw,
                dst_hw=dst_hw,
                view_transform=view_transform,
            ),
            task_heatmap=task_heat,
            effector_heatmap=eff_heat,
            interaction_heatmap=int_heat,
            task_point_prior=task_prior,
            effector_point_prior=eff_prior,
            interaction_point_prior=int_prior,
            anchor_point_priors=anchor_point_priors,
            anchor_x=anchor_x,
            anchor_S=anchor_S,
            anchor_tokens=anchor_tokens,
            anchor_roles=anchor_roles_t,
            anchor_scores=anchor_scores,
            visual_pixel_centers=geometry.visual_pixel_centers,
            valid=valid,
            confidence=torch.stack([task_mass, eff_mass, int_mass]).max(),
            task_pg_heatmap=task_heat_pg,
            effector_pg_heatmap=eff_heat_pg,
            interaction_pg_heatmap=int_heat_pg,
        )

    def _vl_gate(self, logit: nn.Parameter | None, grounding: PicfVLGroundingState | None) -> torch.Tensor:
        if (
            logit is None
            or grounding is None
            or not bool(self.config.vl_anchor_router_enabled)
            or not bool(grounding.valid.item())
        ):
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return torch.sigmoid(logit.to(device=self.device, dtype=self.dtype))

    def _vl_centered_log_prior_bias(self, priors: torch.Tensor) -> torch.Tensor:
        if priors.numel() == 0:
            return priors
        prior = torch.clamp(priors.to(device=self.device, dtype=self.dtype), min=0.0)
        prior = prior / torch.clamp(prior.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        log_prior = torch.log(torch.clamp(prior, min=self.config.epsilon_a))
        centered = log_prior - log_prior.mean(dim=-1, keepdim=True)
        clip = max(float(self.config.vl_prior_bias_clip), 0.0)
        if clip > 0.0:
            centered = torch.clamp(centered, min=-clip, max=clip)
        return centered

    def _aqr_centered_log_prior_bias(self, priors: torch.Tensor) -> torch.Tensor:
        if priors.numel() == 0:
            return priors
        prior = torch.clamp(priors.to(device=self.device, dtype=self.dtype), min=0.0)
        prior = prior / torch.clamp(prior.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        log_prior = torch.log(torch.clamp(prior, min=self.config.epsilon_a))
        centered = log_prior - log_prior.mean(dim=-1, keepdim=True)
        clip = max(float(self.config.aqr_support_bias_clip), 0.0)
        if clip > 0.0:
            centered = torch.clamp(centered, min=-clip, max=clip)
        return centered

    def _aqr_ownership_priors_from_coords(
        self,
        *,
        roles: torch.Tensor,
        coords: torch.Tensor,
        source: torch.Tensor | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        count = int(roles.numel())
        support_count = int(coords.shape[0])
        if count == 0 or support_count == 0:
            return torch.zeros((count, support_count), device=self.device, dtype=self.dtype)
        coords_t = coords.to(device=self.device, dtype=self.dtype)
        uniform = torch.full((support_count,), 1.0 / max(support_count, 1), device=self.device, dtype=self.dtype)
        if source is None:
            source_t = uniform
        else:
            source_t = source.to(device=self.device, dtype=self.dtype)
            if source_t.ndim > 1:
                source_t = source_t.reshape(-1, support_count).mean(dim=0)
            source_t = _normalize_rows(source_t, eps=self.config.epsilon_a)
        priors = torch.zeros((count, support_count), device=self.device, dtype=self.dtype)
        sigma_v = max(
            float(self.config.mapg_visual_sigma_patches if sigma is None else sigma),
            self.config.epsilon_a,
        )
        roles_t = roles.to(device=self.device, dtype=torch.long)
        for role in torch.unique(roles_t).tolist():
            indices = torch.nonzero(roles_t == int(role), as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            mode_priors = self._mapg_mode_priors(
                source_t,
                coords_t,
                count=int(indices.numel()),
                sigma=sigma_v,
            )
            priors[indices] = mode_priors
        mix = float(getattr(self.config, "aqr_ownership_prior_uniform_mix", 0.0))
        if mix > 0.0:
            mix = max(0.0, min(mix, 1.0))
            priors = ((1.0 - mix) * priors) + (mix * uniform[None, :])
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _aqr_visual_ownership_bias(
        self,
        *,
        roles: torch.Tensor,
        visual_count: int,
        visual_grid_index: torch.Tensor,
        vl_grounding: PicfVLGroundingState | None,
    ) -> torch.Tensor | None:
        if (
            not bool(getattr(self.config, "aqr_ownership_prior_enabled", True))
            or float(getattr(self.config, "aqr_ownership_prior_weight", 0.0)) == 0.0
            or visual_count <= 0
            or roles.numel() == 0
            or visual_grid_index.numel() == 0
        ):
            return None
        priors = self._mapg_visual_seed_priors(
            vl_grounding,
            roles=roles,
            visual_count=visual_count,
            visual_grid_index=visual_grid_index.to(device=self.device, dtype=self.dtype),
        )
        mix = float(getattr(self.config, "aqr_ownership_prior_uniform_mix", 0.0))
        if mix > 0.0 and priors.numel() > 0:
            mix = max(0.0, min(mix, 1.0))
            uniform = torch.full_like(priors, 1.0 / max(int(priors.shape[-1]), 1))
            priors = ((1.0 - mix) * priors) + (mix * uniform)
        return float(self.config.aqr_ownership_prior_weight) * self._aqr_centered_log_prior_bias(priors)

    def _aqr_temporal_ownership_bias(
        self,
        *,
        roles: torch.Tensor,
        temporal: PicfTemporalVisualSupportState | None,
    ) -> torch.Tensor | None:
        if (
            temporal is None
            or not bool(getattr(self.config, "aqr_ownership_prior_enabled", True))
            or float(getattr(self.config, "aqr_ownership_temporal_prior_weight", 0.0)) == 0.0
            or roles.numel() == 0
            or temporal.tokens.numel() == 0
            or temporal.grid_index.numel() == 0
        ):
            return None
        grid = temporal.grid_index.to(device=self.device, dtype=self.dtype)
        grid_hw_max = int(temporal.grid_hw.max().item()) if temporal.grid_hw.numel() > 0 else 1
        scale = torch.as_tensor(float(max(grid_hw_max, 1)), device=self.device, dtype=self.dtype)
        view = temporal.view_ids.to(device=self.device, dtype=self.dtype)[:, None] * scale
        time = temporal.time_ids.to(device=self.device, dtype=self.dtype)[:, None] * (0.25 * scale)
        coords = torch.cat([grid, view, time], dim=-1)
        priors = self._aqr_ownership_priors_from_coords(roles=roles, coords=coords)
        return float(self.config.aqr_ownership_temporal_prior_weight) * self._aqr_centered_log_prior_bias(priors)

    def _aqr_point_ownership_bias(
        self,
        token_field: PicfTokenFieldState,
        roles: torch.Tensor,
    ) -> torch.Tensor | None:
        """Label-free object-core ownership prior over point evidence.

        Visual/temporal ownership breaks image-token symmetry, but posterior
        object files are corrected through point-derived geometry. Without a
        row-specific point prior, same-role scene slots can still read the same
        broad point mixture and only separate after posterior correction, which
        is too late for stable object-file ownership.
        """

        point_count = int(token_field.point_tokens.shape[0])
        if (
            point_count == 0
            or roles.numel() == 0
            or not bool(getattr(self.config, "aqr_ownership_prior_enabled", True))
            or float(getattr(self.config, "aqr_ownership_point_prior_weight", 0.0)) == 0.0
        ):
            return None
        positions = self._world_point_positions(token_field)
        if positions.shape != (point_count, 3):
            return None
        roles_t = roles.to(device=self.device, dtype=torch.long)
        pool_ids = self._point_pool_ids(token_field)
        local_mask = pool_ids == 0
        scene_mask = self._scene_point_candidate_mask(token_field, fallback_to_global=True)
        priors = torch.zeros((int(roles_t.numel()), point_count), device=self.device, dtype=self.dtype)
        sigma = max(
            float(getattr(self.config, "aqr_ownership_point_prior_sigma_m", 0.04)),
            float(self.config.epsilon_a),
        )
        for role_value in torch.unique(roles_t, sorted=True).tolist():
            row_indices = torch.nonzero(roles_t == int(role_value), as_tuple=False).squeeze(-1)
            if row_indices.numel() == 0:
                continue
            candidate_mask = local_mask if int(role_value) == 0 else scene_mask
            if not bool(candidate_mask.any().item()):
                candidate_mask = torch.ones((point_count,), device=self.device, dtype=torch.bool)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() == 0:
                continue
            local_priors = self._aqr_ownership_priors_from_coords(
                roles=roles_t.index_select(0, row_indices),
                coords=positions.index_select(0, candidate_indices),
                sigma=sigma,
            )
            priors[row_indices[:, None], candidate_indices[None, :]] = local_priors
        if not bool((priors.sum(dim=-1) > self.config.epsilon_a).any().item()):
            return None
        priors = _normalize_rows(priors, eps=self.config.epsilon_a)
        return float(self.config.aqr_ownership_point_prior_weight) * self._aqr_centered_log_prior_bias(priors)

    def _support_overlap_matrix(self, priors: torch.Tensor | None) -> torch.Tensor | None:
        if priors is None or priors.numel() == 0 or priors.shape[0] == 0 or priors.shape[1] == 0:
            return None
        p = torch.clamp(
            torch.nan_to_num(priors.to(device=self.device, dtype=self.dtype), nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        valid = p.sum(dim=-1) > self.config.epsilon_a
        if not bool(valid.any().item()):
            return None
        p = p / torch.clamp(p.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        overlap = p @ p.T
        diag = torch.clamp(torch.diag(overlap), min=self.config.epsilon_a)
        overlap = overlap / torch.sqrt(torch.clamp(diag[:, None] * diag[None, :], min=self.config.epsilon_a))
        valid_pair = valid[:, None] & valid[None, :]
        return torch.where(valid_pair, overlap, torch.zeros_like(overlap))

    def _object_core_overlap_matrix(
        self,
        visual_priors: torch.Tensor | None,
        *,
        point_priors: torch.Tensor | None = None,
        temporal_priors: torch.Tensor | None = None,
        pg_priors: torch.Tensor | None = None,
        proposal_priors: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        overlaps = [
            overlap
            for overlap in (
                self._support_overlap_matrix(visual_priors),
                self._support_overlap_matrix(point_priors),
                self._support_overlap_matrix(temporal_priors),
                self._support_overlap_matrix(pg_priors),
                self._support_overlap_matrix(proposal_priors),
            )
            if overlap is not None
        ]
        if not overlaps:
            return None
        if len(overlaps) == 1:
            return overlaps[0]
        # Redundancy means the same rows overlap in every available object-core
        # evidence space. A geometric mean keeps one discriminative modality from
        # being overruled by a diffuse visual support map.
        stacked = torch.stack(overlaps, dim=0)
        return torch.exp(torch.mean(torch.log(torch.clamp(stacked, min=self.config.epsilon_a)), dim=0))

    def _aqr_active_slot_mask(
        self,
        *,
        roles: torch.Tensor,
        visual_priors: torch.Tensor,
        point_priors: torch.Tensor | None = None,
        temporal_priors: torch.Tensor | None = None,
        pg_priors: torch.Tensor | None = None,
        proposal_priors: torch.Tensor | None = None,
        anchor_x: torch.Tensor | None = None,
        geometry_valid: torch.Tensor | None = None,
        anchor_scores: torch.Tensor,
        anchor_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Select distinct active anchors; redundant same-role anchors become dustbin candidates."""

        count = int(roles.numel())
        if count == 0:
            return torch.zeros((0,), device=self.device, dtype=self.dtype)
        if not bool(getattr(self.config, "aqr_active_slot_filter_enabled", True)):
            return torch.ones((count,), device=self.device, dtype=self.dtype)
        active = torch.zeros((count,), device=self.device, dtype=self.dtype)
        min_per_role = max(int(getattr(self.config, "aqr_active_slot_min_per_role", 1)), 0)
        max_per_role = max(int(getattr(self.config, "aqr_active_slot_max_per_role", 0)), 0)
        min_conf = max(float(getattr(self.config, "aqr_active_slot_min_confidence", 0.0)), 0.0)
        overlap_threshold = min(max(float(getattr(self.config, "aqr_active_slot_overlap_threshold", 1.0)), 0.0), 1.0)
        priors = torch.clamp(
            torch.nan_to_num(visual_priors.to(device=self.device, dtype=self.dtype), nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        if priors.numel() == 0:
            return torch.ones((count,), device=self.device, dtype=self.dtype)
        overlap = self._object_core_overlap_matrix(
            priors,
            point_priors=point_priors,
            temporal_priors=temporal_priors,
            pg_priors=pg_priors,
            proposal_priors=proposal_priors,
        )
        if overlap is None:
            overlap = torch.eye(count, device=self.device, dtype=self.dtype)
        duplicate_overlap = overlap.to(device=self.device, dtype=self.dtype).clone()
        if (
            bool(getattr(self.config, "aqr_active_slot_geometry_duplicate_enabled", True))
            and anchor_x is not None
            and anchor_x.numel() > 0
            and anchor_x.shape[0] == count
        ):
            centers = anchor_x.to(device=self.device, dtype=self.dtype)[:, :3]
            dist2 = torch.cdist(centers, centers) ** 2
            sigma = max(float(getattr(self.config, "aqr_active_slot_geometry_duplicate_sigma_m", 0.04)), self.config.epsilon_a)
            geom_overlap = torch.exp(-dist2 / (2.0 * sigma * sigma))
            if geometry_valid is not None and geometry_valid.numel() == count:
                valid = geometry_valid.to(device=self.device, dtype=torch.bool)
                geom_overlap = torch.where(valid[:, None] & valid[None, :], geom_overlap, torch.zeros_like(geom_overlap))
            geom_threshold = min(
                max(float(getattr(self.config, "aqr_active_slot_geometry_duplicate_threshold", 0.70)), 0.0),
                1.0,
            )
            geom_duplicate = torch.where(geom_overlap >= geom_threshold, geom_overlap, torch.zeros_like(geom_overlap))
            duplicate_overlap = torch.maximum(duplicate_overlap, geom_duplicate)
        score = torch.clamp(anchor_scores.to(device=self.device, dtype=self.dtype), min=0.0)
        confidence = torch.clamp(anchor_confidence.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        support_peak = priors.max(dim=-1).values
        # Deterministic tie-breaks matter here: when two same-role anchors are
        # near duplicates, keep the slightly more confident support owner and
        # demote the redundant candidate to dustbin.
        tie_break = torch.arange(count, device=self.device, dtype=self.dtype) * self.config.epsilon_a
        score = (score * torch.clamp(confidence, min=float(self.config.mapg_confidence_floor))) + support_peak - tie_break
        relative_threshold = min(
            max(float(getattr(self.config, "aqr_active_slot_relative_score_threshold", 0.0)), 0.0),
            1.0,
        )
        roles = roles.to(device=self.device, dtype=torch.long)
        for role_value in torch.unique(roles, sorted=True).tolist():
            role_indices = torch.nonzero(roles == int(role_value), as_tuple=False).squeeze(-1)
            if role_indices.numel() == 0:
                continue
            local_score = score.index_select(0, role_indices)
            order = torch.argsort(local_score, descending=True)
            kept: list[int] = []
            for local_rank in order.tolist():
                idx = int(role_indices[int(local_rank)].item())
                if max_per_role > 0 and len(kept) >= max_per_role:
                    continue
                if relative_threshold > 0.0 and len(kept) >= min_per_role:
                    best_score = torch.clamp(local_score.max(), min=self.config.epsilon_a)
                    relative_score = float((score[idx] / best_score).item())
                    if relative_score < relative_threshold:
                        continue
                if len(kept) >= min_per_role and float(score[idx].item()) < min_conf:
                    continue
                if kept:
                    kept_t = torch.as_tensor(kept, device=self.device, dtype=torch.long)
                    max_overlap = float(duplicate_overlap[idx, kept_t].max().item())
                    if max_overlap > overlap_threshold and len(kept) >= min_per_role:
                        continue
                kept.append(idx)
            if not kept and role_indices.numel() > 0 and min_per_role > 0:
                top = int(role_indices[int(order[0].item())].item())
                kept.append(top)
            if kept:
                active[torch.as_tensor(kept, device=self.device, dtype=torch.long)] = 1.0
        return active

    def _aqr_downstream_slot_weights(
        self,
        *,
        roles: torch.Tensor,
        visual_priors: torch.Tensor,
        active: torch.Tensor,
        point_priors: torch.Tensor | None = None,
        temporal_priors: torch.Tensor | None = None,
        pg_priors: torch.Tensor | None = None,
        proposal_priors: torch.Tensor | None = None,
        anchor_x: torch.Tensor | None = None,
        geometry_valid: torch.Tensor | None = None,
        anchor_scores: torch.Tensor,
        anchor_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Return action-visible graph weights for active/context/reserve anchors.

        AQR attention can read every typed memory token. This function only
        decides whether an anchor becomes action-prefix object evidence:

        * active anchors use weight 1.0;
        * context anchors use a small weight and preserve real scene objects;
        * duplicate/no-object reserve anchors use weight 0.0.

        This is intentionally not a hard foreground mask over the memories.
        Background remains available through task/global/semantic/visual
        readout, while duplicate fixed-capacity files do not masquerade as
        action-relevant objects.
        """

        count = int(roles.numel())
        if count == 0:
            return torch.zeros((0,), device=self.device, dtype=self.dtype)
        active = torch.clamp(active.to(device=self.device, dtype=self.dtype).reshape(-1)[:count], min=0.0, max=1.0)
        if active.numel() < count:
            active = fn.pad(active, (0, count - int(active.numel())), value=0.0)
        if not bool(getattr(self.config, "aqr_context_slot_enabled", True)):
            return active
        priors = torch.clamp(
            torch.nan_to_num(visual_priors.to(device=self.device, dtype=self.dtype), nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        if priors.numel() == 0 or priors.shape[0] != count:
            return active
        confidence = torch.clamp(anchor_confidence.to(device=self.device, dtype=self.dtype).reshape(-1)[:count], min=0.0, max=1.0)
        if confidence.numel() < count:
            confidence = fn.pad(confidence, (0, count - int(confidence.numel())), value=0.0)
        score = torch.clamp(anchor_scores.to(device=self.device, dtype=self.dtype).reshape(-1)[:count], min=0.0)
        if score.numel() < count:
            score = fn.pad(score, (0, count - int(score.numel())), value=0.0)
        support_peak = priors.max(dim=-1).values
        if proposal_priors is not None and proposal_priors.numel() > 0 and proposal_priors.shape[0] == count:
            proposal_peak = torch.clamp(
                proposal_priors.to(device=self.device, dtype=self.dtype),
                min=0.0,
            ).max(dim=-1).values
            support_peak = torch.maximum(support_peak, proposal_peak)
        object_score = (score * torch.clamp(confidence, min=float(self.config.mapg_confidence_floor))) + support_peak
        min_conf = max(float(getattr(self.config, "aqr_context_slot_min_confidence", 0.05)), 0.0)
        min_score = max(float(getattr(self.config, "aqr_context_slot_min_score", 0.01)), 0.0)
        context_candidate = (active < 0.5) & (confidence >= min_conf) & (object_score >= min_score)

        overlap = self._object_core_overlap_matrix(
            priors,
            point_priors=point_priors,
            temporal_priors=temporal_priors,
            pg_priors=pg_priors,
            proposal_priors=proposal_priors,
        )
        if overlap is None:
            overlap = torch.eye(count, device=self.device, dtype=self.dtype)
        duplicate_overlap = overlap.to(device=self.device, dtype=self.dtype).clone()
        if (
            bool(getattr(self.config, "aqr_active_slot_geometry_duplicate_enabled", True))
            and anchor_x is not None
            and anchor_x.numel() > 0
            and anchor_x.shape[0] == count
        ):
            centers = anchor_x.to(device=self.device, dtype=self.dtype)[:, :3]
            dist2 = torch.cdist(centers, centers) ** 2
            sigma = max(float(getattr(self.config, "aqr_active_slot_geometry_duplicate_sigma_m", 0.04)), self.config.epsilon_a)
            geom_overlap = torch.exp(-dist2 / (2.0 * sigma * sigma))
            if geometry_valid is not None and geometry_valid.numel() == count:
                valid = geometry_valid.to(device=self.device, dtype=torch.bool)
                geom_overlap = torch.where(valid[:, None] & valid[None, :], geom_overlap, torch.zeros_like(geom_overlap))
            duplicate_overlap = torch.maximum(duplicate_overlap, geom_overlap)

        active_bool = active >= 0.5
        if bool(active_bool.any().item()):
            max_to_active = duplicate_overlap[:, active_bool].max(dim=-1).values
            duplicate_threshold = min(
                max(float(getattr(self.config, "aqr_context_slot_duplicate_overlap_threshold", 0.75)), 0.0),
                1.0,
            )
            context_candidate = context_candidate & (max_to_active < duplicate_threshold)
        context_scale = min(max(float(getattr(self.config, "aqr_context_slot_weight", 0.15)), 0.0), 1.0)
        context = context_candidate.to(dtype=self.dtype) * context_scale
        return torch.clamp(torch.maximum(active, context), min=0.0, max=1.0)

    def _task_owner_query_rows(self, roles: torch.Tensor, query_types: torch.Tensor) -> torch.Tensor:
        """Rows that carry task-object semantics inside AQR."""

        if roles.numel() == 0 or query_types.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        roles = roles.to(device=self.device, dtype=torch.long).reshape(-1)
        query_types = query_types.to(device=self.device, dtype=torch.long).reshape(-1)
        task_rows = query_types == 1
        preferred = task_rows & (roles == 1)
        if bool(preferred.any().item()):
            return torch.nonzero(preferred, as_tuple=False).squeeze(-1)
        fallback = task_rows & (roles != 0)
        if bool(fallback.any().item()):
            return torch.nonzero(fallback, as_tuple=False).squeeze(-1)
        return torch.zeros((0,), device=self.device, dtype=torch.long)

    def _task_owner_physical_rows(self, roles: torch.Tensor, query_types: torch.Tensor) -> torch.Tensor:
        """Physical scene-object rows eligible for task ownership bias."""

        if roles.numel() == 0 or query_types.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        roles = roles.to(device=self.device, dtype=torch.long).reshape(-1)
        query_types = query_types.to(device=self.device, dtype=torch.long).reshape(-1)
        mask = (query_types == 0) & (roles == 1)
        if bool(mask.any().item()):
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)
        return torch.zeros((0,), device=self.device, dtype=torch.long)

    def _object_candidate_physical_rows(self, roles: torch.Tensor, query_types: torch.Tensor) -> torch.Tensor:
        """Physical rows eligible to explain sidecar object candidates.

        This is intentionally broader than task ownership: role 1 owns the
        object file, while role 2 carries the contact/interaction bridge needed
        for tactile evidence to attach to the same object. Role 0 effector rows
        are excluded so the gripper cannot become the object owner.
        """

        if roles.numel() == 0 or query_types.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        roles = roles.to(device=self.device, dtype=torch.long).reshape(-1)
        query_types = query_types.to(device=self.device, dtype=torch.long).reshape(-1)
        eligible_roles = tuple(int(role) for role in getattr(self.config, "object_candidate_eligible_roles", (1, 2)))
        mask = query_types == 0
        role_mask = torch.zeros_like(mask, dtype=torch.bool)
        for role in eligible_roles:
            if int(role) == 0:
                continue
            role_mask = role_mask | (roles == int(role))
        mask = mask & role_mask
        if bool(mask.any().item()):
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)
        return self._task_owner_physical_rows(roles, query_types)

    def _centered_log_bias(self, scores: torch.Tensor, *, weight: float) -> torch.Tensor | None:
        scores = torch.clamp(torch.nan_to_num(scores.to(device=self.device, dtype=self.dtype), nan=0.0), min=0.0)
        if scores.numel() == 0 or not bool((scores.sum() > self.config.epsilon_a).item()):
            return None
        prob = scores / torch.clamp(scores.sum(), min=self.config.epsilon_a)
        bias = torch.log(torch.clamp(prob, min=self.config.epsilon_a))
        bias = bias - bias.mean()
        clip = max(float(getattr(self.config, "binding_signature_score_clip", 4.0)), 1.0)
        return torch.clamp(bias, min=-clip, max=clip) * float(weight)

    def _task_owner_visual_prior(
        self,
        visual_priors: torch.Tensor | None,
        *,
        roles: torch.Tensor,
        query_types: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            not bool(getattr(self.config, "task_owner_bias_enabled", True))
            or visual_priors is None
            or visual_priors.numel() == 0
        ):
            return None
        rows = self._task_owner_query_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        priors = torch.clamp(visual_priors.to(device=self.device, dtype=self.dtype).index_select(0, rows), min=0.0)
        if priors.numel() == 0 or not bool((priors.sum() > self.config.epsilon_a).item()):
            return None
        prior = _normalize_rows(priors, eps=self.config.epsilon_a).mean(dim=0)
        if not bool((prior.sum() > self.config.epsilon_a).item()):
            return None
        return prior / torch.clamp(prior.sum(), min=self.config.epsilon_a)

    def _task_owner_visual_bias(
        self,
        visual_prior: torch.Tensor | None,
        *,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        visual_count: int,
    ) -> torch.Tensor | None:
        if visual_prior is None or int(visual_count) <= 0:
            return None
        # Task-object visual evidence should bias the object file row and the
        # contact/interaction bridge row.  The effector row remains excluded so
        # gripper context cannot become object ownership.
        rows = self._object_candidate_physical_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        bias_row = self._centered_log_bias(
            visual_prior[: int(visual_count)],
            weight=float(getattr(self.config, "task_owner_visual_bias_weight", 0.0)),
        )
        if bias_row is None or not bool((torch.abs(bias_row).sum() > self.config.epsilon_a).item()):
            return None
        out = torch.zeros((int(roles.numel()), int(visual_count)), device=self.device, dtype=self.dtype)
        out.index_copy_(0, rows, bias_row[None, :].expand(int(rows.numel()), -1))
        return out

    def _proposal_shape_quality(self, proposal: PicfPseudoProposalState | None) -> torch.Tensor | None:
        """Softly downweight sidecar fragments that are unlikely object proposals.

        Offline proposal sidecars can be useful high-recall evidence, but objectness alone
        can score wall panels, robot protrusions, or drawer edges.  This quality
        prior is deliberately soft: it calibrates proposal influence without
        deleting dense visual tokens or treating a box as a hard object label.
        """

        if proposal is None or proposal.boxes_xyxy.numel() == 0:
            return None
        boxes = torch.clamp(proposal.boxes_xyxy.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            return None
        x0 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y0 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x1 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y1 = torch.maximum(boxes[:, 1], boxes[:, 3])
        wh = torch.clamp(torch.stack((x1 - x0, y1 - y0), dim=-1), min=self.config.epsilon_a)
        area = torch.clamp(wh[:, 0] * wh[:, 1], min=0.0, max=1.0)
        aspect = torch.minimum(wh[:, 0] / torch.clamp(wh[:, 1], min=self.config.epsilon_a), wh[:, 1] / torch.clamp(wh[:, 0], min=self.config.epsilon_a))
        if not bool(getattr(self.config, "proposal_shape_quality_enabled", True)):
            quality = torch.ones_like(area)
        else:
            area_min = max(float(getattr(self.config, "proposal_shape_area_min", 0.002)), self.config.epsilon_a)
            area_max = max(float(getattr(self.config, "proposal_shape_area_max", 0.35)), area_min + self.config.epsilon_a)
            aspect_min = max(float(getattr(self.config, "proposal_shape_aspect_min", 0.20)), self.config.epsilon_a)
            low_tau = max(area_min * 0.5, self.config.epsilon_a)
            high_tau = max(area_max * 0.25, self.config.epsilon_a)
            aspect_tau = max(aspect_min * 0.5, self.config.epsilon_a)
            area_low = torch.sigmoid((area - area_min) / low_tau)
            area_high = torch.sigmoid((area_max - area) / high_tau)
            aspect_gate = torch.sigmoid((aspect - aspect_min) / aspect_tau)
            quality = area_low * area_high * aspect_gate
        if proposal.valid.numel() == quality.numel():
            quality = torch.where(proposal.valid.to(device=self.device, dtype=torch.bool), quality, torch.zeros_like(quality))
        return torch.clamp(quality, min=0.0, max=1.0)

    def _postprocess_task_owner_proposal_score(self, score: torch.Tensor) -> torch.Tensor | None:
        score = torch.clamp(score.to(device=self.device, dtype=self.dtype), min=0.0)
        if not bool((score.max() > self.config.epsilon_a).item()):
            return None
        score = score / torch.clamp(score.max(), min=self.config.epsilon_a)
        floor = max(float(getattr(self.config, "task_owner_proposal_score_floor", 0.05)), 0.0)
        if floor > 0.0:
            score = torch.where(score >= floor, score, torch.zeros_like(score))
        topk = int(getattr(self.config, "task_owner_proposal_topk", 0))
        if topk > 0 and score.numel() > topk:
            values, indices = torch.topk(score, k=min(topk, int(score.numel())), largest=True)
            sparse = torch.zeros_like(score)
            sparse.scatter_(0, indices, values)
            score = sparse
        if not bool((score.max() > self.config.epsilon_a).item()):
            return None
        return score / torch.clamp(score.max(), min=self.config.epsilon_a)

    def _proposal_scores_from_visual_prior(
        self,
        token_field: PicfTokenFieldState,
        visual_prior: torch.Tensor | None,
    ) -> torch.Tensor | None:
        proposal = token_field.proposal
        geom = token_field.projective_geometry
        if (
            proposal is None
            or geom is None
            or visual_prior is None
            or proposal.tokens.numel() == 0
            or proposal.boxes_xyxy.numel() == 0
            or geom.visual_grid_norm.numel() == 0
        ):
            return None
        visual_xy = geom.visual_grid_norm.to(device=self.device, dtype=self.dtype)
        if visual_xy.shape[0] != int(visual_prior.numel()):
            return None
        visual_xy01 = torch.clamp((visual_xy + 1.0) * 0.5, min=0.0, max=1.0)
        prior = torch.clamp(visual_prior.to(device=self.device, dtype=self.dtype), min=0.0)
        if not bool((prior.sum() > self.config.epsilon_a).item()):
            return None
        prior = prior / torch.clamp(prior.sum(), min=self.config.epsilon_a)

        boxes = torch.clamp(proposal.boxes_xyxy.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            return None
        x0 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y0 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x1 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y1 = torch.maximum(boxes[:, 1], boxes[:, 3])
        inside = (
            (visual_xy01[:, 0:1] >= x0[None, :])
            & (visual_xy01[:, 0:1] <= x1[None, :])
            & (visual_xy01[:, 1:2] >= y0[None, :])
            & (visual_xy01[:, 1:2] <= y1[None, :])
        ).to(dtype=self.dtype)
        if inside.numel() == 0:
            return None
        mass = (prior[:, None] * inside).sum(dim=0)
        coverage = torch.clamp(inside.mean(dim=0), min=self.config.epsilon_a)
        objectness = torch.clamp(proposal.objectness.to(device=self.device, dtype=self.dtype), min=0.0)
        power = max(float(getattr(self.config, "task_owner_proposal_objectness_power", 0.5)), 0.0)
        if objectness.numel() == mass.numel() and power > 0.0:
            mass = mass * torch.pow(torch.clamp(objectness, min=self.config.epsilon_a), power)
        shape_quality = self._proposal_shape_quality(proposal)
        if shape_quality is not None and shape_quality.numel() == mass.numel():
            mass = mass * shape_quality
        score = mass / torch.sqrt(coverage)
        if (
            bool(getattr(self.config, "task_owner_proposal_static_only", True))
            and proposal.view_ids.numel() == score.numel()
        ):
            score = torch.where(proposal.view_ids.to(device=self.device, dtype=torch.long) == 0, score, torch.zeros_like(score))
        if proposal.valid.numel() == score.numel():
            score = torch.where(proposal.valid.to(device=self.device, dtype=torch.bool), score, torch.zeros_like(score))
        if not bool((score.max() > self.config.epsilon_a).item()):
            return None
        return self._postprocess_task_owner_proposal_score(score)

    def _task_owner_proposal_bias(
        self,
        proposal_scores: torch.Tensor | None,
        *,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        proposal_count: int,
    ) -> torch.Tensor | None:
        if proposal_scores is None or int(proposal_count) <= 0:
            return None
        rows = self._object_candidate_physical_rows(roles, query_types)
        task_rows = self._task_owner_query_rows(roles, query_types)
        if task_rows.numel() > 0:
            rows = torch.unique(torch.cat([rows, task_rows], dim=0), sorted=True)
        if rows.numel() == 0:
            return None
        bias_row = self._centered_log_bias(
            proposal_scores[: int(proposal_count)],
            weight=float(getattr(self.config, "task_owner_proposal_bias_weight", 0.0)),
        )
        if bias_row is None or not bool((torch.abs(bias_row).sum() > self.config.epsilon_a).item()):
            return None
        out = torch.zeros((int(roles.numel()), int(proposal_count)), device=self.device, dtype=self.dtype)
        out.index_copy_(0, rows, bias_row[None, :].expand(int(rows.numel()), -1))
        return out

    def _proposal_to_point_matrix(
        self,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        """Project proposal boxes into a soft proposal-to-point transport matrix.

        The matrix is a weak geometric correspondence, not a segmentation label:
        each proposal row is normalized over visible projected points inside the
        soft box support.
        """

        proposal = token_field.proposal
        geom = token_field.projective_geometry
        if (
            proposal is None
            or geom is None
            or proposal.boxes_xyxy.numel() == 0
            or geom.point_proj_grid_norm.numel() == 0
        ):
            return None
        point_xy = geom.point_proj_grid_norm.to(device=self.device, dtype=self.dtype)
        if point_xy.ndim != 2 or point_xy.shape[-1] != 2:
            return None
        point_xy01 = torch.clamp((point_xy + 1.0) * 0.5, min=0.0, max=1.0)
        boxes = torch.clamp(proposal.boxes_xyxy.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        membership = None
        if proposal.mask_xy is not None and proposal.mask_weights is not None and proposal.mask_offsets is not None:
            mask_xy = torch.clamp(proposal.mask_xy.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
            mask_weights = torch.clamp(proposal.mask_weights.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0)
            mask_offsets = proposal.mask_offsets.to(device=self.device, dtype=torch.long).reshape(-1)
            proposal_count = int(boxes.shape[0])
            if mask_xy.numel() > 0 and mask_weights.numel() == mask_xy.shape[0] and mask_offsets.numel() >= proposal_count + 1:
                tau_mask = max(float(getattr(self.config, "proposal_mask_point_tau", 0.025)), self.config.epsilon_a)
                rows: list[torch.Tensor] = []
                for proposal_idx in range(proposal_count):
                    start = int(torch.clamp(mask_offsets[proposal_idx], min=0).item())
                    end = int(torch.clamp(mask_offsets[proposal_idx + 1], min=start).item())
                    end = min(end, int(mask_xy.shape[0]))
                    start = min(start, end)
                    if end <= start:
                        rows.append(torch.zeros((point_xy01.shape[0],), device=self.device, dtype=self.dtype))
                        continue
                    samples = mask_xy[start:end]
                    weights = mask_weights[start:end]
                    if not bool((weights.sum() > self.config.epsilon_a).item()):
                        rows.append(torch.zeros((point_xy01.shape[0],), device=self.device, dtype=self.dtype))
                        continue
                    diff = point_xy01[:, None, :] - samples[None, :, :]
                    kernel = torch.exp(-torch.sum(diff * diff, dim=-1) / max(2.0 * tau_mask * tau_mask, self.config.epsilon_a))
                    rows.append((kernel @ weights) / torch.clamp(weights.sum(), min=self.config.epsilon_a))
                membership = torch.stack(rows, dim=1) if rows else None
        if membership is None:
            x0 = torch.minimum(boxes[:, 0], boxes[:, 2])
            y0 = torch.minimum(boxes[:, 1], boxes[:, 3])
            x1 = torch.maximum(boxes[:, 0], boxes[:, 2])
            y1 = torch.maximum(boxes[:, 1], boxes[:, 3])
            tau = max(float(getattr(self.config, "proposal_point_bridge_edge_tau", 0.02)), self.config.epsilon_a)
            px = point_xy01[:, 0:1]
            py = point_xy01[:, 1:2]
            # Soft box membership avoids brittle one-pixel boundary effects while
            # preserving the proposal as a bounded measurement, not a hard mask.
            membership = (
                torch.sigmoid((px - x0[None, :]) / tau)
                * torch.sigmoid((x1[None, :] - px) / tau)
                * torch.sigmoid((py - y0[None, :]) / tau)
                * torch.sigmoid((y1[None, :] - py) / tau)
            )
        if geom.point_visibility.numel() == membership.shape[0]:
            membership = membership * torch.clamp(geom.point_visibility.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)[:, None]
        if geom.point_depth_valid.numel() == membership.shape[0]:
            membership = torch.where(
                geom.point_depth_valid.to(device=self.device, dtype=torch.bool)[:, None],
                membership,
                torch.zeros_like(membership),
            )
        if proposal.valid.numel() == membership.shape[1]:
            membership = torch.where(
                proposal.valid.to(device=self.device, dtype=torch.bool)[None, :],
                membership,
                torch.zeros_like(membership),
            )
        shape_quality = self._proposal_shape_quality(proposal)
        if shape_quality is not None and shape_quality.numel() == membership.shape[1]:
            membership = membership * shape_quality[None, :]
        if not bool((membership.sum() > self.config.epsilon_a).item()):
            return None
        return _normalize_rows(membership.T, eps=self.config.epsilon_a)

    def _proposal_priors_to_point_priors(
        self,
        proposal_priors: torch.Tensor | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        """Bridge anchor proposal reads into 3D point support through projection.

        Proposal boxes are frozen weak measurements, not object labels.  This
        bridge only says: if an anchor reads a proposal box, points whose static
        projection lies inside that box are plausible 3D support for the same
        measurement.  The later posterior update remains authoritative.
        """

        proposal = token_field.proposal
        proposal_to_point = self._proposal_to_point_matrix(token_field)
        if (
            proposal_priors is None
            or proposal_priors.numel() == 0
            or proposal is None
            or proposal_to_point is None
            or proposal_priors.shape[-1] != proposal_to_point.shape[0]
        ):
            return None
        point_priors = torch.clamp(proposal_priors.to(device=self.device, dtype=self.dtype), min=0.0) @ proposal_to_point
        if not bool((point_priors.sum() > self.config.epsilon_a).item()):
            return None
        return _normalize_rows(point_priors, eps=self.config.epsilon_a)

    def _task_owner_proposal_to_point_priors(
        self,
        *,
        task_owner_proposal_score: torch.Tensor | None,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        token_field: PicfTokenFieldState,
        row_count: int,
    ) -> torch.Tensor | None:
        """Use the task-owner proposal as a weak 3D measurement for scene rows.

        This is the missing transport leg when PaliGemma/proposal evidence
        identifies a task-object proposal but no physical AQR row has yet
        learned to attend to it.  The evidence is bounded, row-filtered to physical scene rows,
        and still competes with existing point/posterior measurements.
        """

        if task_owner_proposal_score is None or int(row_count) <= 0:
            return None
        proposal_to_point = self._proposal_to_point_matrix(token_field)
        if proposal_to_point is None or task_owner_proposal_score.numel() != proposal_to_point.shape[0]:
            return None
        rows = self._object_candidate_physical_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        rows = rows[rows < int(row_count)]
        if rows.numel() == 0:
            return None
        target = torch.clamp(task_owner_proposal_score.to(device=self.device, dtype=self.dtype), min=0.0)
        if not bool((target.sum() > self.config.epsilon_a).item()):
            return None
        point_prior = _normalize_rows(target[None, :] @ proposal_to_point, eps=self.config.epsilon_a).squeeze(0)
        if not bool((point_prior.sum() > self.config.epsilon_a).item()):
            return None
        out = torch.zeros((int(row_count), int(point_prior.numel())), device=self.device, dtype=self.dtype)
        out.index_copy_(0, rows, point_prior[None, :].expand(int(rows.numel()), -1))
        return out

    def _proposal_anchor_seed_transport(
        self,
        *,
        task_owner_proposal_score: torch.Tensor | None,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        token_field: PicfTokenFieldState,
        row_count: int,
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Map top task/contact proposals into physical reference rows.

        This is a reference-query transport step, not a hard object label.  It
        turns already-inspected proposal/mask evidence into bounded point priors
        for a small number of physical task-object rows so those rows can enter
        the normal AQR competition and posterior update with the right geometry.
        """

        if (
            not bool(getattr(self.config, "proposal_anchor_seed_enabled", False))
            or task_owner_proposal_score is None
            or int(row_count) <= 0
            or int(point_count) <= 0
        ):
            return None
        proposal = token_field.proposal
        proposal_to_point = self._proposal_to_point_matrix(token_field)
        if (
            proposal is None
            or proposal_to_point is None
            or proposal.tokens.numel() == 0
            or task_owner_proposal_score.numel() != proposal_to_point.shape[0]
            or int(proposal_to_point.shape[1]) != int(point_count)
        ):
            return None
        rows = self._object_candidate_physical_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        row_limit = min(int(rows.numel()), max(int(getattr(self.config, "proposal_anchor_seed_rows", 0)), 0))
        if row_limit <= 0:
            return None
        score = torch.clamp(task_owner_proposal_score.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0)
        if score.numel() != proposal_to_point.shape[0]:
            return None
        floor = max(float(getattr(self.config, "proposal_anchor_seed_score_floor", 0.05)), 0.0)
        if floor > 0.0:
            score = torch.where(score >= floor, score, torch.zeros_like(score))
        if proposal.valid.numel() == score.numel():
            score = torch.where(proposal.valid.to(device=self.device, dtype=torch.bool), score, torch.zeros_like(score))
        if not bool((score.max() > self.config.epsilon_a).item()):
            return None

        proposal_topk = min(row_limit, int(proposal_to_point.shape[0]))
        values, indices = torch.topk(score, k=proposal_topk, largest=True)
        valid = values > self.config.epsilon_a
        if not bool(valid.any().item()):
            return None
        indices = indices[valid]
        values = values[valid]
        if int(indices.numel()) < row_limit:
            repeat = row_limit - int(indices.numel())
            indices = torch.cat([indices, indices[:1].expand(repeat)], dim=0)
            values = torch.cat([values, values[:1].expand(repeat)], dim=0)
        selected_rows = rows[: int(indices.numel())]
        point_rows = torch.clamp(
            proposal_to_point.to(device=self.device, dtype=self.dtype).index_select(0, indices),
            min=0.0,
        )
        power = max(float(getattr(self.config, "proposal_anchor_seed_point_power", 1.0)), self.config.epsilon_a)
        if abs(power - 1.0) > 1.0e-6:
            point_rows = torch.pow(torch.clamp(point_rows, min=0.0), power)
        topk = int(getattr(self.config, "proposal_anchor_seed_point_topk", 0))
        if topk > 0 and point_rows.shape[-1] > topk:
            top_values, top_indices = torch.topk(point_rows, k=topk, dim=-1)
            sparse = torch.zeros_like(point_rows)
            sparse.scatter_(dim=-1, index=top_indices, src=top_values)
            point_rows = sparse
        point_rows = _normalize_rows(point_rows, eps=self.config.epsilon_a)
        if not bool((_row_has_mass(point_rows, eps=self.config.epsilon_a)).any().item()):
            return None

        seed_priors = torch.zeros((int(row_count), int(point_count)), device=self.device, dtype=self.dtype)
        seed_priors.index_copy_(0, selected_rows, point_rows[:, :point_count])
        assignment = torch.zeros((int(row_count), int(proposal_to_point.shape[0])), device=self.device, dtype=self.dtype)
        assignment[selected_rows, indices] = torch.clamp(values / torch.clamp(values.max(), min=self.config.epsilon_a), min=0.0, max=1.0)
        strength = torch.zeros((int(row_count),), device=self.device, dtype=self.dtype)
        strength.index_copy_(0, selected_rows, torch.clamp(values / torch.clamp(values.max(), min=self.config.epsilon_a), min=0.0, max=1.0))
        return seed_priors, assignment, strength

    def _proposal_object_candidate_assignment(
        self,
        *,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        token_field: PicfTokenFieldState,
        point_priors: torch.Tensor | None,
        proposal_priors: torch.Tensor | None,
        task_owner_proposal_score: torch.Tensor | None,
        proposal_anchor_seed_assignment: torch.Tensor | None,
        row_count: int,
        point_count: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None:
        """Assign inspected object candidates to physical slots.

        Sidecar masks are noisy measurements, not labels.  This method turns
        them into object candidates and lets physical scene slots compete to
        explain each candidate, with an explicit background residual absorbing
        invalid fragments.  The resulting assignment is then used as a bounded
        measurement prior for proposal/point routing; it never overwrites the
        posterior state.
        """

        if (
            not bool(getattr(self.config, "object_candidate_assignment_enabled", True))
            or token_field.proposal is None
            or int(row_count) <= 0
            or int(point_count) <= 0
        ):
            return None
        proposal = token_field.proposal
        proposal_to_point = self._proposal_to_point_matrix(token_field)
        if (
            proposal.tokens.numel() == 0
            or proposal_to_point is None
            or proposal_to_point.numel() == 0
            or int(proposal_to_point.shape[1]) != int(point_count)
        ):
            return None
        proposal_count = int(proposal_to_point.shape[0])
        rows = self._object_candidate_physical_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        rows = rows[rows < int(row_count)]
        if rows.numel() == 0:
            return None

        valid = torch.ones((proposal_count,), device=self.device, dtype=torch.bool)
        if proposal.valid.numel() == proposal_count:
            valid = valid & proposal.valid.to(device=self.device, dtype=torch.bool)
        shape_quality = self._proposal_shape_quality(proposal)
        if shape_quality is not None and shape_quality.numel() == proposal_count:
            min_quality = max(float(getattr(self.config, "object_candidate_min_shape_quality", 0.01)), 0.0)
            valid = valid & (shape_quality.to(device=self.device, dtype=self.dtype) >= min_quality)
        if not bool(valid.any().item()):
            return None

        slot_logits = torch.full(
            (int(row_count), proposal_count),
            -1.0e4,
            device=self.device,
            dtype=self.dtype,
        )
        row_score = torch.zeros((int(rows.numel()), proposal_count), device=self.device, dtype=self.dtype)
        row_specific = torch.zeros_like(row_score)
        support_floor = max(
            float(getattr(self.config, "object_candidate_row_support_floor", 1.0e-4)),
            self.config.epsilon_a,
        )

        def _add_positive_score(score: torch.Tensor | None, weight: float, *, row_specific_source: bool) -> None:
            """Add row/candidate evidence without treating weak sources as negatives.

            SlotAttention-style mask competition is additive in positive logits:
            a weak or missing measurement should be neutral, not a multiplicative
            veto.  We therefore normalize each candidate column over eligible
            physical rows and only use source presence for the row-specific
            guard.
            """

            nonlocal row_score, row_specific
            if score is None or score.numel() == 0 or weight == 0.0:
                return
            score = torch.clamp(score.to(device=self.device, dtype=self.dtype), min=0.0)
            if score.shape != row_score.shape:
                return
            good = score >= support_floor
            supported = torch.where(good, score, torch.zeros_like(score))
            col_max = torch.clamp(supported.max(dim=0, keepdim=True).values, min=self.config.epsilon_a)
            normalized = torch.where(col_max > self.config.epsilon_a, supported / col_max, torch.zeros_like(supported))
            row_score = row_score + (float(weight) * normalized)
            if row_specific_source:
                row_specific = torch.maximum(row_specific, good.to(dtype=self.dtype))

        candidate_quality = torch.zeros((proposal_count,), device=self.device, dtype=self.dtype)
        if proposal.objectness.numel() == proposal_count:
            objectness = torch.clamp(proposal.objectness.to(device=self.device, dtype=self.dtype), min=0.0)
            if bool((objectness.max() > self.config.epsilon_a).item()):
                candidate_quality = torch.maximum(
                    candidate_quality,
                    objectness / torch.clamp(objectness.max(), min=self.config.epsilon_a),
                )
        if shape_quality is not None and shape_quality.numel() == proposal_count:
            candidate_quality = candidate_quality * torch.clamp(
                shape_quality.to(device=self.device, dtype=self.dtype),
                min=0.0,
                max=1.0,
            )

        if (
            proposal_priors is not None
            and proposal_priors.numel() > 0
            and proposal_priors.shape[0] == int(row_count)
            and proposal_priors.shape[1] == proposal_count
        ):
            proposal_support = proposal_priors.index_select(0, rows)
            _add_positive_score(
                proposal_support,
                float(getattr(self.config, "object_candidate_proposal_weight", 0.75)),
                row_specific_source=True,
            )
            proposal_quality = torch.clamp(proposal_support.max(dim=0).values, min=0.0)
            if bool((proposal_quality.max() > self.config.epsilon_a).item()):
                candidate_quality = torch.maximum(
                    candidate_quality,
                    proposal_quality / torch.clamp(proposal_quality.max(), min=self.config.epsilon_a),
                )
        if point_priors is not None and point_priors.numel() > 0 and point_priors.shape[0] == int(row_count):
            if point_priors.shape[1] == int(point_count):
                point_overlap = torch.clamp(point_priors.index_select(0, rows), min=0.0) @ torch.clamp(
                    proposal_to_point.to(device=self.device, dtype=self.dtype),
                    min=0.0,
                ).T
                _add_positive_score(
                    point_overlap,
                    float(getattr(self.config, "object_candidate_point_weight", 1.0)),
                    row_specific_source=True,
                )
        if (
            proposal_anchor_seed_assignment is not None
            and proposal_anchor_seed_assignment.numel() > 0
            and proposal_anchor_seed_assignment.shape[0] == int(row_count)
            and proposal_anchor_seed_assignment.shape[1] == proposal_count
        ):
            seed = torch.clamp(proposal_anchor_seed_assignment.index_select(0, rows), min=0.0)
            seed_weight = float(getattr(self.config, "object_candidate_seed_weight", 1.25))
            if seed_weight != 0.0:
                _add_positive_score(seed, seed_weight, row_specific_source=True)
        if task_owner_proposal_score is not None and task_owner_proposal_score.numel() == proposal_count:
            owner = torch.clamp(task_owner_proposal_score.to(device=self.device, dtype=self.dtype), min=0.0)
            owner_weight = float(getattr(self.config, "object_candidate_task_owner_weight", 0.5))
            if owner_weight != 0.0 and bool((owner.max() > self.config.epsilon_a).item()):
                owner = owner / torch.clamp(owner.max(), min=self.config.epsilon_a)
                row_score = row_score + (owner_weight * owner[None, :])
                candidate_quality = torch.maximum(candidate_quality, owner)

        # Do not create arbitrary row symmetry from a task-level proposal alone.
        # At least one row-specific support source must connect a physical row
        # to a candidate, otherwise the background residual should absorb it.
        row_score = torch.where(row_specific > 0.0, row_score, torch.full_like(row_score, -1.0e4))
        slot_logits.index_copy_(0, rows, row_score)
        slot_logits = torch.where(valid[None, :], slot_logits, torch.full_like(slot_logits, -1.0e4))
        if not bool((slot_logits > -9999.0).any().item()):
            return None

        pre_topk_slot_logits = slot_logits.clone()
        max_rows_per_candidate = int(getattr(self.config, "object_candidate_max_rows_per_candidate", 1))
        if max_rows_per_candidate > 0 and max_rows_per_candidate < int(row_count):
            finite = slot_logits > -9999.0
            k = min(max_rows_per_candidate, int(row_count))
            top_values, top_indices = torch.topk(
                torch.where(finite, slot_logits, torch.full_like(slot_logits, -1.0e4)),
                k=k,
                dim=0,
            )
            keep = torch.zeros_like(finite)
            keep.scatter_(dim=0, index=top_indices, src=top_values > -9999.0)
            slot_logits = torch.where(keep & finite, slot_logits, torch.full_like(slot_logits, -1.0e4))

        temperature = max(float(getattr(self.config, "object_candidate_assignment_temperature", 0.35)), self.config.epsilon_a)
        bg_prior = max(float(getattr(self.config, "object_candidate_background_prior", 0.25)), self.config.epsilon_a)
        bg_quality_weight = max(float(getattr(self.config, "object_candidate_background_quality_weight", 2.0)), 0.0)
        candidate_quality = torch.where(valid, torch.clamp(candidate_quality, min=0.0, max=1.0), torch.zeros_like(candidate_quality))
        scaled = slot_logits / temperature
        bg_logit = torch.full((proposal_count,), math.log(bg_prior), device=self.device, dtype=self.dtype)
        bg_logit = bg_logit - (bg_quality_weight * candidate_quality)
        max_col = torch.maximum(scaled.max(dim=0).values, bg_logit)
        slot_exp = torch.exp(scaled - max_col[None, :])
        slot_exp = torch.where(valid[None, :], slot_exp, torch.zeros_like(slot_exp))
        bg_exp = torch.exp(bg_logit - max_col)
        bg_exp = torch.where(valid, bg_exp, torch.zeros_like(bg_exp))
        row_capacity = float(getattr(self.config, "object_candidate_row_capacity", 1.25))
        row_capacity_iters = int(getattr(self.config, "object_candidate_row_capacity_iters", 2))
        if row_capacity > 0.0 and row_capacity_iters > 0:
            cap = torch.as_tensor(row_capacity, device=self.device, dtype=self.dtype)
            for _ in range(row_capacity_iters):
                denom_i = torch.clamp(slot_exp.sum(dim=0) + bg_exp, min=self.config.epsilon_a)
                assignment_i = slot_exp / denom_i[None, :]
                row_mass = assignment_i.sum(dim=-1, keepdim=True)
                scale = torch.clamp(cap / torch.clamp(row_mass, min=self.config.epsilon_a), max=1.0)
                slot_exp = slot_exp * scale
        denom = torch.clamp(slot_exp.sum(dim=0) + bg_exp, min=self.config.epsilon_a)
        assignment = slot_exp / denom[None, :]
        background = bg_exp / denom
        coverage = assignment.sum(dim=0)
        row_strength = torch.clamp(assignment.sum(dim=-1), min=0.0, max=1.0)

        owner_assignment = torch.zeros_like(assignment)
        owner_point_priors = torch.zeros(
            (int(row_count), int(point_count)),
            device=self.device,
            dtype=self.dtype,
        )
        if bool(getattr(self.config, "object_candidate_owner_transport_enabled", True)):
            owner_roles = tuple(int(role) for role in getattr(self.config, "object_candidate_owner_roles", (1,)))
            roles_t = roles.to(device=self.device, dtype=torch.long).reshape(-1)
            query_types_t = query_types.to(device=self.device, dtype=torch.long).reshape(-1)
            owner_mask = query_types_t == 0
            role_mask = torch.zeros_like(owner_mask, dtype=torch.bool)
            for role in owner_roles:
                if int(role) == 0:
                    continue
                role_mask = role_mask | (roles_t == int(role))
            owner_mask = owner_mask & role_mask
            owner_rows = torch.nonzero(owner_mask, as_tuple=False).squeeze(-1)
            owner_rows = owner_rows[owner_rows < int(row_count)]
            if owner_rows.numel() > 0:
                owner_logits = pre_topk_slot_logits.index_select(0, owner_rows)
                finite_owner = owner_logits > -9999.0
                candidate_ok = valid & (coverage > self.config.epsilon_a)
                owner_logits = torch.where(
                    finite_owner & candidate_ok[None, :],
                    owner_logits,
                    torch.full_like(owner_logits, -1.0e4),
                )
                best_values, best_pos = owner_logits.max(dim=0)
                candidate_has_owner = best_values > -9999.0
                if bool(candidate_has_owner.any().item()):
                    candidate_idx = torch.nonzero(candidate_has_owner, as_tuple=False).squeeze(-1)
                    selected_owner_rows = owner_rows.index_select(0, best_pos.index_select(0, candidate_idx))
                    min_share = min(max(float(getattr(self.config, "object_candidate_owner_min_share", 0.65)), 0.0), 1.0)
                    owner_mass = torch.clamp(coverage.index_select(0, candidate_idx) * min_share, min=0.0, max=1.0)
                    current_owner_mass = assignment[selected_owner_rows, candidate_idx]
                    owner_mass = torch.maximum(owner_mass, current_owner_mass)
                    owner_assignment[selected_owner_rows, candidate_idx] = owner_mass
                    owner_point_priors = torch.clamp(owner_assignment, min=0.0) @ torch.clamp(
                        proposal_to_point.to(device=self.device, dtype=self.dtype),
                        min=0.0,
                    )
                    owner_rows_have = _row_has_mass(owner_point_priors, eps=self.config.epsilon_a)
                    if bool(owner_rows_have.any().item()):
                        owner_point_priors = torch.where(
                            owner_rows_have[:, None],
                            _normalize_rows(owner_point_priors, eps=self.config.epsilon_a),
                            torch.zeros_like(owner_point_priors),
                        )

        candidate_point_priors = torch.clamp(assignment, min=0.0) @ torch.clamp(
            proposal_to_point.to(device=self.device, dtype=self.dtype),
            min=0.0,
        )
        point_rows = _row_has_mass(candidate_point_priors, eps=self.config.epsilon_a)
        if bool(point_rows.any().item()):
            candidate_point_priors = torch.where(
                point_rows[:, None],
                _normalize_rows(candidate_point_priors, eps=self.config.epsilon_a),
                torch.zeros_like(candidate_point_priors),
            )
        if owner_point_priors.numel() > 0 and bool(_row_has_mass(owner_point_priors, eps=self.config.epsilon_a).any().item()):
            owner_mix = min(max(float(getattr(self.config, "object_candidate_owner_point_mix", 0.85)), 0.0), 1.0)
            if owner_mix > 0.0:
                owner_rows_have = _row_has_mass(owner_point_priors, eps=self.config.epsilon_a)
                mixed_owner = (
                    ((1.0 - owner_mix) * torch.clamp(candidate_point_priors, min=0.0))
                    + (owner_mix * torch.clamp(owner_point_priors, min=0.0))
                )
                candidate_point_priors = torch.where(
                    owner_rows_have[:, None],
                    _normalize_rows(mixed_owner, eps=self.config.epsilon_a),
                    candidate_point_priors,
                )

        candidate_proposal_priors = torch.where(
            row_strength[:, None] > self.config.epsilon_a,
            _normalize_rows(torch.clamp(assignment, min=0.0), eps=self.config.epsilon_a),
            torch.zeros_like(assignment),
        )
        if owner_assignment.numel() > 0 and bool(_row_has_mass(owner_assignment, eps=self.config.epsilon_a).any().item()):
            owner_rows_have = _row_has_mass(owner_assignment, eps=self.config.epsilon_a)
            mixed_owner_assignment = (
                (0.5 * torch.clamp(candidate_proposal_priors, min=0.0))
                + (0.5 * torch.clamp(owner_assignment, min=0.0))
            )
            candidate_proposal_priors = torch.where(
                owner_rows_have[:, None],
                _normalize_rows(mixed_owner_assignment, eps=self.config.epsilon_a),
                candidate_proposal_priors,
            )
        denom_rows = torch.sqrt(torch.clamp((assignment * assignment).sum(dim=-1, keepdim=True), min=self.config.epsilon_a))
        norm_assign = assignment / denom_rows
        duplicate = norm_assign @ norm_assign.T
        duplicate = duplicate.masked_fill(torch.eye(int(row_count), device=self.device, dtype=torch.bool), 0.0)
        return (
            assignment,
            owner_assignment,
            owner_point_priors,
            coverage,
            background,
            duplicate,
            candidate_point_priors,
            candidate_proposal_priors,
            row_strength,
        )

    def _task_owner_proposal_point_bias(
        self,
        *,
        task_owner_proposal_score: torch.Tensor | None,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        token_field: PicfTokenFieldState,
        point_count: int,
    ) -> torch.Tensor | None:
        """Point-reader likelihood bias induced by the task-owner proposal.

        This is stronger than post-read prior mixing: it lets the current AQR
        query compete over projected task-object points during point attention,
        so the resulting hidden state and point priors stay aligned.
        """

        if task_owner_proposal_score is None or int(point_count) <= 0:
            return None
        proposal_to_point = self._proposal_to_point_matrix(token_field)
        if proposal_to_point is None or task_owner_proposal_score.numel() != proposal_to_point.shape[0]:
            return None
        rows = self._object_candidate_physical_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        target = torch.clamp(task_owner_proposal_score.to(device=self.device, dtype=self.dtype), min=0.0)
        if not bool((target.sum() > self.config.epsilon_a).item()):
            return None
        point_scores = torch.clamp(target[None, :] @ proposal_to_point, min=0.0).squeeze(0)
        bias_row = self._centered_log_bias(
            point_scores,
            weight=float(getattr(self.config, "task_owner_proposal_point_bias_weight", 0.0)),
        )
        if bias_row is None:
            return None
        out = torch.zeros((int(roles.numel()), int(point_count)), device=self.device, dtype=self.dtype)
        out.index_copy_(0, rows, bias_row[None, :].expand(int(rows.numel()), -1))
        return out

    def _task_owner_anchor_score(
        self,
        *,
        proposal_priors: torch.Tensor | None,
        task_owner_proposal_score: torch.Tensor | None,
        visual_priors: torch.Tensor | None,
        task_owner_visual_prior: torch.Tensor | None,
        roles: torch.Tensor,
        query_types: torch.Tensor,
    ) -> torch.Tensor | None:
        """Row-wise task-object ownership evidence for active-slot selection."""

        row_count = int(roles.numel())
        if row_count == 0:
            return None
        score = torch.zeros((row_count,), device=self.device, dtype=self.dtype)
        if (
            proposal_priors is not None
            and task_owner_proposal_score is not None
            and proposal_priors.numel() > 0
            and task_owner_proposal_score.numel() == proposal_priors.shape[-1]
        ):
            prop = _normalize_rows(torch.clamp(proposal_priors.to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
            target = torch.clamp(task_owner_proposal_score.to(device=self.device, dtype=self.dtype), min=0.0)
            if bool((target.max() > self.config.epsilon_a).item()):
                target = target / torch.clamp(target.max(), min=self.config.epsilon_a)
                score = torch.maximum(score, prop @ target)
        if (
            visual_priors is not None
            and task_owner_visual_prior is not None
            and visual_priors.numel() > 0
            and task_owner_visual_prior.numel() == visual_priors.shape[-1]
        ):
            visual = _normalize_rows(torch.clamp(visual_priors.to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
            target_v = torch.clamp(task_owner_visual_prior.to(device=self.device, dtype=self.dtype), min=0.0)
            if bool((target_v.max() > self.config.epsilon_a).item()):
                target_v = target_v / torch.clamp(target_v.max(), min=self.config.epsilon_a)
                score = torch.maximum(score, visual @ target_v)
        rows = self._object_candidate_physical_rows(roles, query_types)
        if rows.numel() == 0:
            return None
        filtered = torch.zeros_like(score)
        filtered.index_copy_(0, rows, score.index_select(0, rows))
        if not bool((filtered.max() > self.config.epsilon_a).item()):
            return None
        return filtered

    def _vl_slot_point_priors(
        self,
        grounding: PicfVLGroundingState | None,
        slot_role_ids: torch.Tensor,
        *,
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_count = int(slot_role_ids.numel())
        priors = torch.zeros((slot_count, int(point_count)), device=self.device, dtype=self.dtype)
        valid_rows = torch.zeros((slot_count,), device=self.device, dtype=torch.bool)
        if (
            grounding is None
            or not bool(grounding.valid.item())
            or point_count <= 0
            or slot_count == 0
        ):
            return priors, valid_rows

        anchor_priors = grounding.anchor_point_priors.to(device=self.device, dtype=self.dtype)
        anchor_roles = grounding.anchor_roles.to(device=self.device, dtype=torch.long)

        def _fallback_for_role(role: int) -> torch.Tensor:
            if role == 0:
                return grounding.effector_point_prior.to(device=self.device, dtype=self.dtype)
            if role == 1:
                scene = grounding.task_point_prior.to(device=self.device, dtype=self.dtype)
                scene = scene + grounding.interaction_point_prior.to(device=self.device, dtype=self.dtype)
                return scene
            scene = grounding.task_point_prior.to(device=self.device, dtype=self.dtype)
            scene = scene + grounding.interaction_point_prior.to(device=self.device, dtype=self.dtype)
            return scene

        for role in torch.unique(slot_role_ids.to(device=self.device, dtype=torch.long)).tolist():
            role_int = int(role)
            slot_indices = torch.nonzero(slot_role_ids.to(device=self.device, dtype=torch.long) == role_int, as_tuple=False).squeeze(-1)
            if slot_indices.numel() == 0:
                continue
            if anchor_priors.numel() > 0 and anchor_roles.numel() == anchor_priors.shape[0]:
                if role_int == 0:
                    role_mask = anchor_roles == 0
                elif role_int == 1:
                    role_mask = (anchor_roles == 1) | (anchor_roles == 2)
                else:
                    role_mask = (anchor_roles == 1) | (anchor_roles == 2)
                candidates = anchor_priors[role_mask]
            else:
                candidates = torch.zeros((0, point_count), device=self.device, dtype=self.dtype)
            fallback = _fallback_for_role(role_int)
            fallback = torch.clamp(fallback, min=0.0)
            fallback = fallback / torch.clamp(fallback.sum(), min=self.config.epsilon_a)
            for local_index, slot_index in enumerate(slot_indices.tolist()):
                if candidates.shape[0] > 0:
                    prior = candidates[int(local_index) % int(candidates.shape[0])]
                else:
                    prior = fallback
                prior = torch.clamp(prior[:point_count], min=0.0)
                if prior.numel() < point_count:
                    prior = fn.pad(prior, (0, point_count - prior.numel()))
                mass = prior.sum()
                if bool((mass > self.config.epsilon_a).item()):
                    priors[int(slot_index)] = prior / torch.clamp(mass, min=self.config.epsilon_a)
                    valid_rows[int(slot_index)] = True
        return priors, valid_rows

    def _mapg_gate(self, logit: nn.Parameter | None, graph: PicfAnchorPriorGraphState | None) -> torch.Tensor:
        if logit is None or graph is None or not bool(graph.valid.item()):
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return torch.sigmoid(logit.reshape(()).to(device=self.device, dtype=self.dtype))

    def _mapg_anchor_roles(self, count: int) -> torch.Tensor:
        count = max(int(count), 0)
        if count == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        layout = str(getattr(self.config, "aqr_role_layout", "structured")).lower().replace("-", "_")
        if layout in {"object_only", "object"}:
            return torch.ones((count,), device=self.device, dtype=torch.long)
        if layout in {"no_effector", "object_contact_context"}:
            task = max(1, count // 2)
            interaction = max(1, (count - task) // 2) if count - task > 0 else 0
            coverage = max(count - task - interaction, 0)
            roles = ([1] * task) + ([2] * interaction) + ([3] * coverage)
            while len(roles) < count:
                roles.append(3)
            return torch.as_tensor(roles[:count], device=self.device, dtype=torch.long)
        effector = min(1, count)
        remaining = count - effector
        task = max(1, remaining // 3) if remaining > 0 else 0
        interaction = max(1, (remaining - task) // 2) if remaining - task > 0 else 0
        coverage = max(remaining - task - interaction, 0)
        roles = (
            ([0] * effector)
            + ([1] * task)
            + ([2] * interaction)
            + ([3] * coverage)
        )
        while len(roles) < count:
            roles.append(3)
        return torch.as_tensor(roles[:count], device=self.device, dtype=torch.long)

    def _aqr_coverage_codes(self, count: int) -> torch.Tensor:
        count = max(int(count), 0)
        if count == 0:
            return torch.zeros((0, 2), device=self.device, dtype=self.dtype)
        index = torch.arange(count, device=self.device, dtype=self.dtype)
        # Deterministic low-discrepancy query identities break same-role row
        # symmetry without introducing a separate keypoint extractor.
        phi = (math.sqrt(5.0) - 1.0) * 0.5
        x = torch.frac(index * phi)
        y = torch.frac(index * (phi * phi))
        return (torch.stack([x, y], dim=-1) * 2.0) - 1.0

    def _aqr_heatmap_confidence(self, heatmap: torch.Tensor | None) -> torch.Tensor:
        if heatmap is None or heatmap.numel() == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        probs = _normalize_rows(heatmap.reshape(-1).to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
        count = max(int(probs.numel()), 2)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=self.config.epsilon_a)))
        entropy_norm = entropy / torch.log(torch.as_tensor(float(count), device=self.device, dtype=self.dtype))
        peak_to_uniform = torch.max(probs) * float(count)
        entropy_gate = torch.sigmoid(12.0 * (float(self.config.aqr_pg_entropy_threshold) - entropy_norm))
        peak_gate = torch.sigmoid(2.0 * (peak_to_uniform - float(self.config.aqr_pg_peak_threshold)))
        return torch.clamp(entropy_gate * peak_gate, min=0.0, max=1.0)

    def _aqr_pg_visual_bias(
        self,
        vl_grounding: PicfVLGroundingState | None,
        *,
        roles: torch.Tensor,
        query_types: torch.Tensor,
        visual_count: int,
    ) -> torch.Tensor | None:
        if vl_grounding is None or visual_count == 0 or roles.numel() == 0:
            return None
        if not bool(self.config.aqr_pg_grounding_enabled) or float(self.config.aqr_pg_bias_weight) <= 0.0:
            return None
        bias = torch.zeros((int(roles.numel()), visual_count), device=self.device, dtype=self.dtype)
        for role_value, heatmap in (
            (0, vl_grounding.effector_heatmap),
            (1, vl_grounding.task_heatmap),
            (2, vl_grounding.interaction_heatmap),
            (3, None),
        ):
            rows = torch.nonzero((roles == int(role_value)) & (query_types == 1), as_tuple=False).squeeze(-1)
            if rows.numel() == 0 or heatmap is None or heatmap.numel() != visual_count:
                continue
            confidence = self._aqr_heatmap_confidence(heatmap)
            if not bool((confidence > self.config.epsilon_a).item()):
                continue
            prior = _normalize_rows(heatmap.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
            centered = torch.log(torch.clamp(prior, min=self.config.epsilon_a))
            centered = centered - centered.mean()
            centered = torch.clamp(centered, min=-float(self.config.aqr_support_bias_clip), max=float(self.config.aqr_support_bias_clip))
            bias[rows] = float(self.config.aqr_pg_bias_weight) * confidence * centered[None, :]
        return bias

    def _aqr_visual_grid_hw(self, token_field: PicfTokenFieldState, visual_count: int) -> tuple[int, int] | None:
        geometry = token_field.projective_geometry
        if geometry is not None and geometry.visual_grid_index.shape[0] == visual_count and visual_count > 0:
            grid = geometry.visual_grid_index
            width = int(torch.max(grid[:, 0]).item()) + 1
            height = int(torch.max(grid[:, 1]).item()) + 1
            if width > 0 and height > 0 and (width * height) == visual_count:
                return (height, width)
        side = int(round(math.sqrt(max(int(visual_count), 0))))
        if side > 0 and (side * side) == int(visual_count):
            return (side, side)
        return None

    def _aqr_pg_image_support_read(
        self,
        q: torch.Tensor,
        semantic: _SemanticContext,
        *,
        query_types: torch.Tensor,
        token_field: PicfTokenFieldState,
        visual_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if (
            not bool(self.config.aqr_pg_image_support_enabled)
            or self.aqr_pg_image_reader is None
            or visual_count == 0
            or semantic.image_tokens is None
            or semantic.image_tokens.numel() == 0
            or not semantic.image_token_ranges
            or not semantic.image_grid_shapes
        ):
            return q, None, None
        rows = torch.nonzero(query_types == 1, as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            return q, None, None
        token_slices: list[tuple[int, int, tuple[int, int], Any | None]] = []
        for index, (start, end) in enumerate(semantic.image_token_ranges):
            if end <= start or index >= len(semantic.image_grid_shapes):
                continue
            pg_hw = semantic.image_grid_shapes[index]
            if int(pg_hw[0]) * int(pg_hw[1]) != int(end - start):
                continue
            transform = semantic.image_view_transforms[index] if index < len(semantic.image_view_transforms) else None
            token_slices.append((int(start), int(end), pg_hw, transform))
        if not token_slices:
            return q, None, None
        pg_tokens = torch.cat(
            [
                semantic.image_tokens[start:end].to(device=self.device, dtype=self.dtype)
                for start, end, _pg_hw, _transform in token_slices
            ],
            dim=0,
        )
        visual_hw = self._aqr_visual_grid_hw(token_field, visual_count)
        if visual_hw is None:
            return q, None, None
        task_q, pg_weights = self.aqr_pg_image_reader(
            q[:, rows, :],
            pg_tokens[None, :],
        )
        updated = q.clone()
        updated[:, rows, :] = task_q
        pg_priors = torch.zeros((int(query_types.numel()), int(pg_tokens.shape[0])), device=self.device, dtype=self.dtype)
        pg_priors[rows] = self._aqr_competitive_support(pg_weights, eps=self.config.epsilon_a)
        bias = torch.zeros((int(query_types.numel()), visual_count), device=self.device, dtype=self.dtype)
        weight = max(float(self.config.aqr_pg_image_support_weight), 0.0)
        if weight <= 0.0:
            return updated, pg_priors, None
        offset = 0
        for start, end, pg_hw, transform in token_slices:
            length = int(end - start)
            local_weights = pg_priors[:, offset : offset + length]
            offset += length
            for row_idx in rows.tolist():
                support = _map_pg_heatmap_to_visual_grid(
                    local_weights[int(row_idx)].to(device=self.device, dtype=self.dtype),
                    src_hw=pg_hw,
                    dst_hw=visual_hw,
                    view_transform=transform,
                    eps=self.config.epsilon_a,
                )
                centered = torch.log(torch.clamp(support, min=self.config.epsilon_a))
                centered = centered - centered.mean()
                centered = torch.clamp(
                    centered,
                    min=-float(self.config.aqr_support_bias_clip),
                    max=float(self.config.aqr_support_bias_clip),
                )
                bias[int(row_idx)] = bias[int(row_idx)] + (weight * centered)
        return updated, pg_priors, bias

    def _aqr_point_bias(self, token_field: PicfTokenFieldState, roles: torch.Tensor) -> torch.Tensor | None:
        point_count = int(token_field.point_tokens.shape[0])
        if point_count == 0 or roles.numel() == 0:
            return None
        pool_ids = self._point_pool_ids(token_field)
        scene_mask = self._scene_point_candidate_mask(token_field, fallback_to_global=True)
        local_mask = pool_ids == 0
        bias = torch.zeros((int(roles.numel()), point_count), device=self.device, dtype=self.dtype)
        neg = torch.full_like(bias, -1.0e4)
        for row, role in enumerate(roles.tolist()):
            role_int = int(role)
            mask = local_mask if role_int == 0 else scene_mask
            if not bool(mask.any().item()):
                mask = torch.ones((point_count,), device=self.device, dtype=torch.bool)
            bias[row] = torch.where(mask, bias[row], neg[row])
        return bias

    def _aqr_posterior_bias(self, previous: PicfPreviousState | None, roles: torch.Tensor) -> torch.Tensor | None:
        if previous is None or roles.numel() == 0:
            return None
        post_count = int(previous.posterior.tokens.shape[0])
        if post_count == 0:
            return None
        post_roles = previous.posterior.role_ids
        if post_roles is None or post_roles.numel() != post_count:
            post_roles = self._posterior_role_ids()
        post_roles = post_roles.to(device=self.device, dtype=torch.long)
        bias = torch.zeros((int(roles.numel()), post_count), device=self.device, dtype=self.dtype)
        neg = torch.full_like(bias, -1.0e4)
        active_gate = self._posterior_file_active_gate(previous.posterior, count=post_count, dtype=self.dtype)
        active_mask = active_gate >= 0.5
        for row, role in enumerate(roles.tolist()):
            role_int = int(role)
            mask = (post_roles == 0) if role_int == 0 else (post_roles != 0)
            mask = mask & active_mask
            if not bool(mask.any().item()):
                # Backward-compatible fallback: if file competition marked no
                # active same-role file, allow role-compatible reserve state
                # rather than making the attention row all -inf.
                mask = (post_roles == 0) if role_int == 0 else (post_roles != 0)
            bias[row] = torch.where(mask, bias[row], neg[row])
        return bias

    def _empty_cache_read_state(self, *, hidden_dim: int | None = None) -> PicfCacheReadState:
        width = int(hidden_dim or self.config.hidden_dim)
        return PicfCacheReadState(
            tokens=torch.zeros((0, width), device=self.device, dtype=self.dtype),
            slot_address=torch.zeros((0, width), device=self.device, dtype=self.dtype),
            slot_content=torch.zeros((0, width), device=self.device, dtype=self.dtype),
            role_ids=torch.zeros((0,), device=self.device, dtype=torch.long),
            source_ids=torch.zeros((0,), device=self.device, dtype=torch.long),
            score=torch.zeros((0,), device=self.device, dtype=self.dtype),
            age=torch.zeros((0,), device=self.device, dtype=self.dtype),
            uncertainty=torch.zeros((0,), device=self.device, dtype=self.dtype),
            innovation=torch.zeros((0,), device=self.device, dtype=self.dtype),
            modality_validity=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
            valid=torch.zeros((0,), device=self.device, dtype=torch.bool),
        )

    def _innovation_risk_scalar(self, innovation_norm: torch.Tensor | None) -> torch.Tensor:
        if innovation_norm is None or innovation_norm.numel() == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        values = torch.clamp(innovation_norm.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0)
        if values.numel() == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return torch.linalg.norm(values) / math.sqrt(float(values.numel()))

    def _measurement_innovation_norm(
        self,
        x_prior: torch.Tensor,
        S_prior: torch.Tensor,
        obs: PicfObservationAnchorState,
    ) -> torch.Tensor:
        if obs.x.numel() == 0 or x_prior.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=self.dtype)
        obs_x = obs.x.to(device=self.device, dtype=self.dtype)
        obs_S = obs.S.to(device=self.device, dtype=self.dtype)
        prior_x = x_prior.to(device=self.device, dtype=self.dtype)
        prior_S = S_prior.to(device=self.device, dtype=self.dtype)
        prior_diag = torch.diagonal(prior_S, dim1=-2, dim2=-1)
        obs_diag = torch.diagonal(obs_S, dim1=-2, dim2=-1)
        delta = obs_x[None, :, :] - prior_x[:, None, :]
        scale = torch.clamp(prior_diag[:, None, :] + obs_diag[None, :, :] + (self.config.bind_sigma_m**2), min=self.config.epsilon_s)
        maha = torch.sum((delta**2) / scale, dim=-1)
        nearest = torch.min(maha, dim=-1).values
        return torch.sqrt(torch.clamp(nearest, min=0.0))

    def _physical_query_addresses(
        self,
        previous: PicfPreviousState | None,
        physical_count: int,
    ) -> torch.Tensor:
        base = self.aqr_physical_query_tokens[:physical_count].to(device=self.device, dtype=self.dtype)
        if previous is None or previous.posterior.slot_address is None or previous.posterior.slot_address.numel() == 0:
            return base
        posterior_address = previous.posterior.slot_address.to(device=self.device, dtype=self.dtype)
        width = min(int(base.shape[-1]), int(posterior_address.shape[-1]))
        count = min(int(base.shape[0]), int(posterior_address.shape[0]))
        if count <= 0 or width <= 0:
            return base
        out = base.clone()
        out[:count, :width] = posterior_address[:count, :width]
        return _normalize_tensor(out, eps=self.config.epsilon_residual)

    def _aqr_cache_query_addresses(
        self,
        previous: PicfPreviousState | None,
        physical_count: int,
        task_count: int,
    ) -> torch.Tensor:
        physical = self._physical_query_addresses(previous, physical_count)
        task = self.aqr_task_query_tokens[:task_count].to(device=self.device, dtype=self.dtype)
        return torch.cat([physical, task], dim=0)

    def _vcap_grad_scale(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        scale = float(scale)
        if scale >= 1.0:
            return tensor
        if scale <= 0.0:
            return tensor.detach()
        return tensor.detach() + (scale * (tensor - tensor.detach()))

    def _vcap_memory_summary(
        self,
        *,
        token_field: PicfTokenFieldState,
        previous: PicfPreviousState | None,
    ) -> torch.Tensor:
        """Summarize current evidence for active-proposal generation.

        VCAP is not allowed to prune or replace dense typed memory. This summary
        is only a compact conditioning vector for generating proposal query
        initializers before the normal AQR readers consume the full memories.
        """

        width = int(self.config.hidden_dim)
        parts: list[torch.Tensor] = []

        def _add(tokens: torch.Tensor | None) -> None:
            if tokens is None or tokens.numel() == 0:
                return
            tokens_t = tokens.to(device=self.device, dtype=self.dtype)
            if tokens_t.shape[-1] != width:
                return
            parts.append(tokens_t.reshape(-1, width).mean(dim=0))

        _add(token_field.visual_tokens)
        _add(token_field.point_tokens)
        _add(token_field.tactile_tokens)
        if token_field.temporal_visual is not None:
            _add(token_field.temporal_visual.tokens)
        if token_field.tracklet is not None:
            _add(token_field.tracklet.tokens)
        if token_field.proposal is not None:
            _add(token_field.proposal.tokens)
        if previous is not None and previous.posterior.tokens.numel() > 0:
            alpha = torch.clamp(previous.posterior.alpha.to(device=self.device, dtype=self.dtype), min=0.0)
            denom = torch.clamp(alpha.sum(), min=self.config.epsilon_a)
            parts.append((alpha[:, None] * previous.posterior.tokens.to(device=self.device, dtype=self.dtype)).sum(dim=0) / denom)
        if not parts:
            return torch.zeros((width,), device=self.device, dtype=self.dtype)
        return torch.stack(parts, dim=0).mean(dim=0)

    def _vcap_active_proposal_queries(
        self,
        *,
        base_queries: torch.Tensor,
        token_field: PicfTokenFieldState,
        previous: PicfPreviousState | None,
    ) -> tuple[torch.Tensor, PicfActiveProposalState | None, torch.Tensor | None]:
        """Generate padded variable-cardinality active proposal query initializers.

        The returned query tensor preserves the existing fixed AQR shape. VCAP
        controls only proposal/query initialization and active priors; posterior
        files remain the authoritative state after the normal matching/update.
        """

        physical_count = int(base_queries.shape[0])
        if (
            physical_count == 0
            or not bool(getattr(self.config, "vcap_enabled", False))
            or self.vcap_decoder is None
            or self.vcap_summary_proj is None
            or self.vcap_start_token is None
            or self.vcap_reserve_token is None
            or self.vcap_query_head is None
            or self.vcap_address_head is None
            or self.vcap_geometry_head is None
            or self.vcap_role_head is None
            or self.vcap_stop_head is None
            or self.vcap_support_head is None
        ):
            return base_queries, None, None

        max_active = min(max(int(getattr(self.config, "vcap_max_active", physical_count)), 0), physical_count)
        min_active = min(max(int(getattr(self.config, "vcap_min_active", 1)), 0), max_active)
        summary = self._vcap_memory_summary(token_field=token_field, previous=previous)
        hidden = torch.tanh(self.vcap_summary_proj(summary.reshape(1, -1)))[0]
        prev_token = self.vcap_start_token.to(device=self.device, dtype=self.dtype)
        reserve = self.vcap_reserve_token.to(device=self.device, dtype=self.dtype)
        survival = torch.ones((), device=self.device, dtype=self.dtype)
        generated: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        active_probs: list[torch.Tensor] = []
        role_logits: list[torch.Tensor] = []
        address_seed: list[torch.Tensor] = []
        geometry_seed: list[torch.Tensor] = []
        support_seed: list[torch.Tensor] = []
        for index in range(physical_count):
            hidden = self.vcap_decoder(prev_token.reshape(1, -1), hidden.reshape(1, -1))[0]
            delta = self.vcap_query_head(hidden)
            token = base_queries[index] + delta
            stop = self.vcap_stop_head(hidden).reshape(())
            stop_prob = torch.sigmoid(stop)
            if index < min_active:
                active = torch.ones((), device=self.device, dtype=self.dtype)
            elif index < max_active:
                active = survival * (1.0 - stop_prob)
            else:
                active = torch.zeros((), device=self.device, dtype=self.dtype)
            if index >= min_active - 1:
                survival = survival * (1.0 - stop_prob)
            generated.append(token)
            stop_logits.append(stop)
            active_probs.append(active)
            role_logits.append(self.vcap_role_head(hidden))
            address_seed.append(_normalize_tensor(self.vcap_address_head(hidden), eps=self.config.epsilon_residual))
            geometry_seed.append(self.vcap_geometry_head(hidden))
            support_seed.append(_normalize_tensor(self.vcap_support_head(hidden), eps=self.config.epsilon_residual))
            prev_token = token

        tokens = torch.stack(generated, dim=0)
        stop_t = torch.stack(stop_logits, dim=0)
        active_t = torch.clamp(torch.stack(active_probs, dim=0), min=0.0, max=1.0)
        threshold = min(max(float(getattr(self.config, "vcap_stop_threshold", 0.5)), 0.0), 1.0)
        active_hard = (active_t >= threshold).to(dtype=self.dtype)
        if min_active > 0:
            min_mask = torch.arange(physical_count, device=self.device) < int(min_active)
            active_hard = torch.where(min_mask, torch.ones_like(active_hard), active_hard)
        if max_active < physical_count:
            max_mask = torch.arange(physical_count, device=self.device) < int(max_active)
            active_hard = torch.where(max_mask, active_hard, torch.zeros_like(active_hard))
        active_forward = active_hard + (active_t - active_t.detach())
        support_t = torch.stack(support_seed, dim=0)
        if physical_count > 1:
            sim = torch.clamp(support_t @ support_t.T, min=0.0, max=1.0)
            pair_mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
            weighted_dup = sim * active_t[:, None] * active_t[None, :]
            duplicate_score = weighted_dup[pair_mask].mean() if bool(pair_mask.any().item()) else tokens.sum() * 0.0
        else:
            duplicate_score = tokens.sum() * 0.0
        count_cost = active_t.sum() / max(float(physical_count), 1.0)
        if previous is not None and previous.posterior.alpha.numel() > 0:
            prev_alpha = torch.clamp(previous.posterior.alpha.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
            target_count = torch.clamp(
                (prev_alpha > float(getattr(self.config, "posterior_file_competition_min_support", 0.02))).to(dtype=self.dtype).sum(),
                min=float(min_active),
                max=float(max_active),
            )
            continuity_cost = torch.abs(active_t.sum() - target_count) / max(float(physical_count), 1.0)
        else:
            continuity_cost = tokens.sum() * 0.0
        state = PicfActiveProposalState(
            tokens=tokens,
            stop_logits=stop_t,
            active_prob=active_t,
            role_logits=torch.stack(role_logits, dim=0),
            address_seed=torch.stack(address_seed, dim=0),
            geometry_seed=torch.stack(geometry_seed, dim=0),
            support_signature_seed=support_t,
            coverage_score=active_hard.detach(),
            duplicate_score=duplicate_score.reshape(()),
            valid=torch.tensor(True, device=self.device),
            unexplained_evidence=None,
            count_cost=count_cost.reshape(()),
            continuity_cost=continuity_cost.reshape(()),
        )
        action_scale = float(getattr(self.config, "vcap_action_grad_scale", 0.0))
        active_downstream = self._vcap_grad_scale(active_forward, action_scale)
        delta_downstream = self._vcap_grad_scale(tokens - base_queries, action_scale)
        reserve_downstream = self._vcap_grad_scale(reserve, action_scale)
        padded_queries = base_queries + (active_downstream[:, None] * delta_downstream) + ((1.0 - active_downstream)[:, None] * reserve_downstream[None, :])
        assignment = torch.eye(physical_count, device=self.device, dtype=self.dtype)
        return padded_queries, state, assignment

    def _finalize_vcap_proposal_state(
        self,
        state: PicfActiveProposalState | None,
        *,
        visual_priors: torch.Tensor,
        point_priors: torch.Tensor | None,
        temporal_priors: torch.Tensor | None,
        pg_priors: torch.Tensor | None,
        proposal_priors: torch.Tensor | None,
        physical_count: int,
    ) -> PicfActiveProposalState | None:
        if state is None or int(physical_count) <= 0:
            return state
        active = torch.clamp(state.active_prob[:physical_count].to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        evidence_rows: list[torch.Tensor] = []
        for priors in (visual_priors, point_priors, temporal_priors, pg_priors, proposal_priors):
            if priors is None or priors.numel() == 0 or priors.shape[0] < physical_count:
                continue
            p = _normalize_rows(torch.clamp(priors[:physical_count].to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
            importance = torch.clamp(p.max(dim=0).values, min=0.0)
            if bool((importance.sum() > self.config.epsilon_a).item()):
                covered = torch.clamp((active[:, None] * p).max(dim=0).values, min=0.0, max=1.0)
                evidence_rows.append((importance * (1.0 - covered)).sum() / torch.clamp(importance.sum(), min=self.config.epsilon_a))
        unexplained = torch.stack(evidence_rows).mean() if evidence_rows else state.tokens.sum() * 0.0
        return dataclasses.replace(state, unexplained_evidence=unexplained.reshape(()))

    def _binding_keys(self, tokens: torch.Tensor | None, *, center: bool = False) -> torch.Tensor:
        if tokens is None or tokens.numel() == 0:
            width = max(int(self.config.binding_signature_dim), 1)
            return torch.zeros((0, width), device=self.device, dtype=self.dtype)
        projected = self.binding_signature_proj(tokens.to(device=self.device, dtype=self.dtype))
        if center and bool(getattr(self.config, "binding_signature_centering_enabled", True)):
            min_tokens = max(int(getattr(self.config, "binding_signature_centering_min_tokens", 4)), 2)
            if int(projected.shape[0]) >= min_tokens:
                projected = projected - projected.mean(dim=0, keepdim=True)
        return _normalize_tensor(projected, eps=self.config.epsilon_residual)

    def _support_binding_signature(
        self,
        weights: torch.Tensor | None,
        tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if weights is None or tokens is None or weights.numel() == 0 or tokens.numel() == 0:
            return None
        if int(weights.shape[-1]) != int(tokens.shape[0]):
            return None
        normalized_weights = _normalize_rows(weights.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
        signature = normalized_weights @ self._binding_keys(tokens, center=True)
        return _normalize_tensor(signature, eps=self.config.epsilon_residual)

    def _binding_signature_quadratic_scores(
        self,
        prev_binding: torch.Tensor,
        obs_binding: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pairwise IsSameObject scores over the centered binding subspace.

        The diagonal term starts as an identity-equivalent quadratic score and
        can learn per-dimension reliability. The low-rank symmetric term follows
        the fixed-rank quadratic probe family used in object-binding audits,
        but it is gated inside posterior binding rather than used as a weak
        same-object loss.
        """

        if prev_binding.numel() == 0 or obs_binding.numel() == 0:
            return None, None
        width = min(
            int(prev_binding.shape[-1]),
            int(obs_binding.shape[-1]),
            int(self.binding_quadratic_diag.numel()),
            int(self.binding_low_rank_left.in_features),
        )
        if width <= 0:
            return None, None
        prev_norm = _normalize_tensor(prev_binding.to(device=self.device, dtype=self.dtype)[..., :width], eps=self.config.epsilon_residual)
        obs_norm = _normalize_tensor(obs_binding.to(device=self.device, dtype=self.dtype)[..., :width], eps=self.config.epsilon_residual)
        diag = self.binding_quadratic_diag.to(device=self.device, dtype=self.dtype)[:width]
        diag_score = (prev_norm * diag[None, :]) @ obs_norm.T / math.sqrt(float(width))

        low_rank_score: torch.Tensor | None = None
        if width == int(self.binding_low_rank_left.in_features):
            left_prev = self.binding_low_rank_left(prev_norm)
            right_prev = self.binding_low_rank_right(prev_norm)
            left_obs = self.binding_low_rank_left(obs_norm)
            right_obs = self.binding_low_rank_right(obs_norm)
            rank = max(int(left_prev.shape[-1]), 1)
            low_rank_score = 0.5 * (
                (left_prev @ right_obs.T) + (right_prev @ left_obs.T)
            ) / math.sqrt(float(rank))
        return diag_score, low_rank_score

    def _calibrate_pairwise_binding_score(self, score: torch.Tensor) -> torch.Tensor:
        """Convert raw IsSameObject scores into relative assignment logits.

        The object-binding probe code trains a BCE logit with learned scale and
        bias. Runtime PICF has no mask labels, so a raw positive cosine or
        quadratic common-mode must not be treated as object identity evidence.
        This calibration removes row/column common terms and only emits a
        binding logit when the pairwise matrix has real dispersion.
        """

        if score.numel() == 0:
            return score
        score = torch.nan_to_num(score.to(device=self.device, dtype=self.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        if not bool(getattr(self.config, "binding_signature_score_calibration_enabled", True)):
            return score
        if score.ndim != 2 or int(score.shape[0]) < 2 or int(score.shape[1]) < 2:
            return torch.zeros_like(score)
        mode = str(getattr(self.config, "binding_signature_score_calibration_mode", "double_center_zscore")).lower()
        if mode in {"double_center", "double_center_zscore", "double_center_std"}:
            centered = score - score.mean(dim=1, keepdim=True) - score.mean(dim=0, keepdim=True) + score.mean()
        elif mode in {"row_center", "row_zscore"}:
            centered = score - score.mean(dim=1, keepdim=True)
        elif mode in {"global_center", "zscore"}:
            centered = score - score.mean()
        else:
            centered = score - score.mean(dim=1, keepdim=True) - score.mean(dim=0, keepdim=True) + score.mean()
        if "zscore" in mode or "std" in mode:
            min_std = max(float(getattr(self.config, "binding_signature_score_min_std", 0.05)), float(self.config.epsilon_a))
            std = torch.std(centered, unbiased=False)
            if bool((std < min_std).item()):
                return torch.zeros_like(centered)
            centered = centered / std
        clip = float(getattr(self.config, "binding_signature_score_clip", 4.0))
        if clip > 0.0:
            centered = torch.clamp(centered, min=-clip, max=clip)
        return torch.nan_to_num(centered, nan=0.0, posinf=clip if clip > 0.0 else 0.0, neginf=-clip if clip > 0.0 else 0.0)

    def _previous_evidence_cache_tokens(
        self,
        previous: PicfPreviousState | None,
    ) -> PicfCacheReadState:
        if previous is None or not bool(self.config.evidence_cache_enabled):
            return self._empty_cache_read_state()
        cache = getattr(previous.predictive, "evidence_cache", None)
        if cache is None or cache.tokens.numel() == 0:
            return self._empty_cache_read_state()
        valid = cache.valid.to(device=self.device, dtype=torch.bool)
        age_all = cache.age.to(device=self.device, dtype=self.dtype)
        source_all = cache.source_ids.to(device=self.device, dtype=torch.long)
        # The newest posterior cache row is exactly previous.posterior.tokens,
        # which AQR already reads through a dedicated posterior branch. Keep the
        # cache as longer-horizon episodic evidence by skipping that duplicate
        # source and letting older rows provide history.
        immediate_posterior = (source_all == 1) & (age_all <= self.config.epsilon_a)
        valid = valid & ~immediate_posterior
        if valid.numel() == 0 or not bool(valid.any().item()):
            return self._empty_cache_read_state(hidden_dim=int(cache.tokens.shape[-1]))
        tokens = cache.tokens.to(device=self.device, dtype=self.dtype)[valid]
        slot_address = cache.slot_address.to(device=self.device, dtype=self.dtype)[valid]
        uncertainty = cache.uncertainty.to(device=self.device, dtype=self.dtype)[valid]
        age = age_all[valid]
        innovation = cache.innovation_at_write.to(device=self.device, dtype=self.dtype)[valid]
        source_ids = source_all[valid]
        role_ids = cache.role_ids.to(device=self.device, dtype=torch.long)[valid]
        modality_validity = cache.modality_validity.to(device=self.device, dtype=self.dtype)[valid]
        innovation_cost = float(self.config.evidence_cache_innovation_downweight) * torch.clamp(innovation, min=0.0)
        source_factor = torch.where(
            source_ids == 1,
            torch.ones_like(uncertainty),
            torch.full_like(uncertainty, 0.5),
        )
        score = source_factor / torch.clamp(1.0 + uncertainty + age + innovation_cost, min=self.config.epsilon_a)
        flat_tokens = tokens.reshape(-1, tokens.shape[-1])
        return PicfCacheReadState(
            tokens=flat_tokens,
            slot_address=slot_address.reshape(-1, slot_address.shape[-1]),
            slot_content=flat_tokens,
            role_ids=role_ids.reshape(-1),
            source_ids=source_ids.reshape(-1),
            score=score.reshape(-1),
            age=age.reshape(-1),
            uncertainty=uncertainty.reshape(-1),
            innovation=innovation.reshape(-1),
            modality_validity=modality_validity.reshape(-1, modality_validity.shape[-1]),
            valid=torch.ones((int(flat_tokens.shape[0]),), device=self.device, dtype=torch.bool),
        )

    def _aqr_competitive_support(self, weights: torch.Tensor, *, eps: float) -> torch.Tensor:
        if weights.numel() == 0 or weights.shape[0] == 0 or weights.shape[1] == 0:
            return weights
        local = torch.clamp(torch.nan_to_num(weights.to(device=self.device, dtype=self.dtype), nan=0.0), min=0.0)
        topk = min(int(local.shape[1]), max(32, int(local.shape[0]) * 4))
        if topk < int(local.shape[1]):
            _, top_indices = torch.topk(local, k=topk, dim=-1)
            mask = torch.zeros_like(local, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
            local = torch.where(mask, local, torch.zeros_like(local))
        local = local / torch.clamp(local.sum(dim=-1, keepdim=True), min=eps)
        temperature = max(float(self.config.aqr_sinkhorn_temperature), eps)
        if abs(temperature - 1.0) > 1e-6:
            local = torch.pow(torch.clamp(local, min=eps), 1.0 / temperature)
            local = local / torch.clamp(local.sum(dim=-1, keepdim=True), min=eps)
        active_cols = local.sum(dim=0) > eps
        target_col = active_cols.to(dtype=self.dtype) * (float(local.shape[0]) / max(int(active_cols.sum().item()), 1))
        for _ in range(max(int(self.config.aqr_sinkhorn_iters), 0)):
            local = local * (target_col / torch.clamp(local.sum(dim=0), min=eps))[None, :]
            local = local / torch.clamp(local.sum(dim=-1, keepdim=True), min=eps)
        return local

    def _aqr_same_role_support_competition(
        self,
        priors: torch.Tensor,
        *,
        roles: torch.Tensor,
        query_types: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        """Make same-role scene object files compete for evidence tokens.

        This is a measurement-routing constraint, not an auxiliary loss: if two
        same-role object files read the same support, the branch keeps only the
        row-specific relative advantage and then mixes it back as a small
        residual. Identical rows remain identical, so the term cannot invent
        evidence; it only amplifies weak object-specific differences already
        present in the typed support.
        """
        if not bool(getattr(self.config, "aqr_same_role_support_competition_enabled", False)):
            return priors
        if priors.numel() == 0 or priors.shape[0] < 2 or priors.shape[1] == 0:
            return priors
        weight = max(float(getattr(self.config, "aqr_same_role_support_competition_weight", 0.0)), 0.0)
        if weight <= 0.0:
            return priors
        weight = min(weight, 1.0)
        iters = max(int(getattr(self.config, "aqr_same_role_support_competition_iters", 1)), 1)
        local = _normalize_rows(
            torch.clamp(torch.nan_to_num(priors.to(device=self.device, dtype=self.dtype), nan=0.0), min=0.0),
            eps=eps,
        )
        roles_t = roles.to(device=self.device)
        if roles_t.numel() != int(local.shape[0]):
            return local
        eligible = roles_t != 0
        if bool(getattr(self.config, "aqr_same_role_support_competition_physical_only", True)) and query_types is not None:
            qt = query_types.to(device=self.device)
            if qt.numel() == int(local.shape[0]):
                eligible = eligible & (qt == 0)
        if int(eligible.sum().item()) < 2:
            return local
        out = local.clone()
        for role_value in torch.unique(roles_t[eligible]).tolist():
            row_mask = eligible & (roles_t == int(role_value))
            if int(row_mask.sum().item()) < 2:
                continue
            rows = torch.nonzero(row_mask, as_tuple=False).flatten()
            original = local.index_select(0, rows)
            exclusive = original
            valid_cols = original.sum(dim=0, keepdim=True) > eps
            if int(valid_cols.sum().item()) <= 1:
                continue
            for _ in range(iters):
                share = exclusive / torch.clamp(exclusive.sum(dim=0, keepdim=True), min=eps)
                share = torch.where(valid_cols, share, torch.zeros_like(share))
                exclusive = _normalize_rows(share, eps=eps)
            mixed = _normalize_rows(((1.0 - weight) * original) + (weight * exclusive), eps=eps)
            out = out.index_copy(0, rows, mixed)
        return out

    def _mapg_mode_priors(
        self,
        weights: torch.Tensor,
        coords: torch.Tensor,
        *,
        count: int,
        sigma: float,
    ) -> torch.Tensor:
        support_count = int(weights.shape[0])
        count = max(int(count), 0)
        if count == 0:
            return torch.zeros((0, support_count), device=self.device, dtype=self.dtype)
        if support_count == 0:
            return torch.zeros((count, 0), device=self.device, dtype=self.dtype)
        base = torch.clamp(weights.to(device=self.device, dtype=self.dtype), min=0.0)
        if not bool((base.sum() > self.config.epsilon_a).item()):
            base = torch.ones_like(base)
        base = base / torch.clamp(base.sum(), min=self.config.epsilon_a)
        coords_t = coords.to(device=self.device, dtype=self.dtype)
        entropy = -torch.sum(base * torch.log(torch.clamp(base, min=self.config.epsilon_a)))
        max_entropy = torch.log(torch.as_tensor(float(max(support_count, 2)), device=self.device, dtype=self.dtype))
        confidence = 1.0 - (entropy / torch.clamp(max_entropy, min=self.config.epsilon_a))
        flat_source = bool(((torch.max(base) - torch.min(base)) <= (10.0 * self.config.epsilon_a)).item())
        low_confidence = bool((confidence < float(self.config.mapg_mode_confidence_threshold)).item())
        if flat_source or low_confidence:
            # Coverage priors must not inherit torch.argmax's first-cell bias when the
            # evidence is uniform or too diffuse. Use geometry-only FPS over the
            # native support rather than treating the first cell as a semantic mode.
            modes = _fps_indices(coords_t, min(count, support_count))
        else:
            modes = _weighted_anchor_modes(
                coords_t,
                base,
                count=count,
                radius_m=max(float(sigma), self.config.epsilon_a),
                eps=self.config.epsilon_a,
            )
        priors = []
        sigma_v = max(float(sigma), self.config.epsilon_a)
        for idx in modes.tolist():
            center = coords_t[int(idx)]
            dist2 = torch.sum((coords_t - center[None, :]) ** 2, dim=-1)
            local = torch.exp(-dist2 / (2.0 * sigma_v * sigma_v)) * base
            priors.append(_normalize_rows(local, eps=self.config.epsilon_a))
        while len(priors) < count:
            priors.append(base)
        return torch.stack(priors[:count], dim=0)

    def _mapg_visual_seed_priors(
        self,
        vl_grounding: PicfVLGroundingState | None,
        *,
        roles: torch.Tensor,
        visual_count: int,
        visual_grid_index: torch.Tensor,
    ) -> torch.Tensor:
        if visual_count == 0 or roles.numel() == 0:
            return torch.zeros((int(roles.numel()), visual_count), device=self.device, dtype=self.dtype)
        uniform = torch.full((visual_count,), 1.0 / max(visual_count, 1), device=self.device, dtype=self.dtype)
        if vl_grounding is None:
            task_heat = eff_heat = int_heat = uniform
        else:
            task_heat = _normalize_rows(vl_grounding.task_heatmap.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
            eff_heat = _normalize_rows(vl_grounding.effector_heatmap.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
            int_heat = _normalize_rows(vl_grounding.interaction_heatmap.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
        sigma = max(float(self.config.mapg_visual_sigma_patches), self.config.epsilon_a)
        priors = torch.zeros((int(roles.numel()), visual_count), device=self.device, dtype=self.dtype)
        for role in torch.unique(roles).tolist():
            role_int = int(role)
            indices = torch.nonzero(roles == role_int, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            if role_int == 0:
                source = eff_heat
            elif role_int == 1:
                source = task_heat
            elif role_int == 2:
                source = int_heat
            else:
                source = uniform
            mode_priors = self._mapg_mode_priors(
                source,
                visual_grid_index,
                count=int(indices.numel()),
                sigma=sigma,
            )
            priors[indices] = mode_priors
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_pg_seed_priors(
        self,
        vl_grounding: PicfVLGroundingState | None,
        *,
        roles: torch.Tensor,
        pg_count: int,
        pg_hw: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if pg_count == 0 or pg_hw is None or roles.numel() == 0:
            return None
        grid_y, grid_x = torch.meshgrid(
            torch.arange(int(pg_hw[0]), device=self.device, dtype=self.dtype),
            torch.arange(int(pg_hw[1]), device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        uniform = torch.full((pg_count,), 1.0 / max(pg_count, 1), device=self.device, dtype=self.dtype)
        if vl_grounding is None:
            task_heat = eff_heat = int_heat = uniform
        else:
            task_heat = vl_grounding.task_pg_heatmap if vl_grounding.task_pg_heatmap is not None else uniform
            eff_heat = vl_grounding.effector_pg_heatmap if vl_grounding.effector_pg_heatmap is not None else uniform
            int_heat = vl_grounding.interaction_pg_heatmap if vl_grounding.interaction_pg_heatmap is not None else uniform
            task_heat = _normalize_rows(task_heat.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
            eff_heat = _normalize_rows(eff_heat.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
            int_heat = _normalize_rows(int_heat.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
        priors = torch.zeros((int(roles.numel()), pg_count), device=self.device, dtype=self.dtype)
        for role in torch.unique(roles).tolist():
            role_int = int(role)
            indices = torch.nonzero(roles == role_int, as_tuple=False).squeeze(-1)
            if role_int == 0:
                source = eff_heat
            elif role_int == 1:
                source = task_heat
            elif role_int == 2:
                source = int_heat
            else:
                source = uniform
            priors[indices] = self._mapg_mode_priors(
                source,
                coords,
                count=int(indices.numel()),
                sigma=max(float(self.config.mapg_visual_sigma_patches), self.config.epsilon_a),
            )
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_visual_to_point(
        self,
        visual_priors: torch.Tensor,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        geometry = token_field.projective_geometry
        if geometry is None or geometry.projective_compatibility.numel() == 0 or visual_priors.numel() == 0:
            return None
        compat = torch.clamp(torch.nan_to_num(geometry.projective_compatibility.to(device=self.device, dtype=self.dtype), nan=0.0), min=0.0)
        mask = token_field.point_projectable_mask
        if mask is not None and mask.shape == (compat.shape[0],):
            compat = compat * mask.to(device=self.device, dtype=self.dtype)[:, None]
        col_mass = compat.sum(dim=0, keepdim=True)
        compat_col = compat / torch.clamp(col_mass, min=self.config.epsilon_a)
        priors = visual_priors.to(device=self.device, dtype=self.dtype) @ compat_col.T
        valid = col_mass.reshape(-1) > self.config.epsilon_a
        visible_mass = visual_priors.to(device=self.device, dtype=self.dtype) @ valid.to(dtype=self.dtype)
        priors = torch.where(visible_mass[:, None] > float(self.config.vl_min_visible_mass), priors, torch.zeros_like(priors))
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_point_to_visual(
        self,
        point_priors: torch.Tensor | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        if point_priors is None:
            return None
        geometry = token_field.projective_geometry
        if geometry is None or geometry.projective_compatibility.numel() == 0 or point_priors.numel() == 0:
            return None
        compat = torch.clamp(torch.nan_to_num(geometry.projective_compatibility.to(device=self.device, dtype=self.dtype), nan=0.0), min=0.0)
        mask = token_field.point_projectable_mask
        if mask is not None and mask.shape == (compat.shape[0],):
            compat = compat * mask.to(device=self.device, dtype=self.dtype)[:, None]
        row_mass = compat.sum(dim=1, keepdim=True)
        compat_row = compat / torch.clamp(row_mass, min=self.config.epsilon_a)
        priors = point_priors.to(device=self.device, dtype=self.dtype) @ compat_row
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_tactile_seed_priors(self, token_field: PicfTokenFieldState, roles: torch.Tensor) -> torch.Tensor | None:
        tactile_count = int(token_field.tactile_tokens.shape[0])
        if tactile_count == 0 or roles.numel() == 0:
            return None
        prob = token_field.tactile_contact_prob
        group_ids = token_field.tactile_group_ids
        if prob is not None and prob.numel() > 0 and group_ids is not None and group_ids.numel() == tactile_count:
            if int(group_ids.max().item()) < int(prob.numel()):
                weights = prob.index_select(0, group_ids).to(device=self.device, dtype=self.dtype)
            else:
                weights = torch.ones((tactile_count,), device=self.device, dtype=self.dtype)
        elif prob is not None and prob.numel() == tactile_count:
            weights = prob.to(device=self.device, dtype=self.dtype)
        else:
            weights = torch.ones((tactile_count,), device=self.device, dtype=self.dtype)
        weights = _normalize_rows(weights, eps=self.config.epsilon_a)
        priors = torch.zeros((int(roles.numel()), tactile_count), device=self.device, dtype=self.dtype)
        attach_to_object = bool(getattr(self.config, "tactile_attach_to_object_owner", True))
        active_roles = (roles == 1) if attach_to_object else ((roles == 0) | (roles == 2))
        priors[active_roles] = weights[None, :]
        return priors

    def _mapg_tactile_to_point(
        self,
        tactile_priors: torch.Tensor | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        if tactile_priors is None or tactile_priors.numel() == 0:
            return None
        point_positions = self._world_point_positions(token_field)
        tactile_positions = token_field.tactile_positions_world
        if point_positions.numel() == 0 or tactile_positions.numel() == 0:
            return None
        sigma = max(float(self.config.mapg_tactile_sigma_m), self.config.epsilon_a)
        dist2 = torch.cdist(point_positions.to(dtype=self.dtype), tactile_positions.to(device=self.device, dtype=self.dtype)) ** 2
        kernel = torch.exp(-dist2 / (2.0 * sigma * sigma))
        kernel = kernel / torch.clamp(kernel.sum(dim=0, keepdim=True), min=self.config.epsilon_a)
        priors = tactile_priors.to(device=self.device, dtype=self.dtype) @ kernel.T
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_posterior_seed_priors(
        self,
        previous: PicfPreviousState | None,
        roles: torch.Tensor,
    ) -> torch.Tensor | None:
        if previous is None or roles.numel() == 0:
            return None
        post_roles = previous.posterior.role_ids
        if post_roles is None or post_roles.numel() != previous.posterior.tokens.shape[0]:
            post_roles = self._posterior_role_ids().to(device=self.device, dtype=torch.long)
        post_roles = post_roles.to(device=self.device, dtype=torch.long)
        alpha = torch.clamp(previous.posterior.alpha.to(device=self.device, dtype=self.dtype), min=0.0)
        post_count = int(previous.posterior.tokens.shape[0])
        priors = torch.zeros((int(roles.numel()), post_count), device=self.device, dtype=self.dtype)
        for index, role in enumerate(roles.tolist()):
            role_int = int(role)
            if role_int == 0:
                mask = post_roles == 0
            else:
                mask = post_roles != 0
            weights = torch.where(mask, alpha, torch.zeros_like(alpha))
            if bool((weights.sum() > self.config.epsilon_a).item()):
                priors[index] = weights / torch.clamp(weights.sum(), min=self.config.epsilon_a)
        return priors

    def _mapg_posterior_to_point(
        self,
        posterior_priors: torch.Tensor | None,
        previous: PicfPreviousState | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        if posterior_priors is None or previous is None or posterior_priors.numel() == 0:
            return None
        point_positions = self._world_point_positions(token_field)
        if point_positions.numel() == 0:
            return None
        sigma = max(float(self.config.mapg_posterior_sigma_m), self.config.epsilon_a)
        dist2 = torch.cdist(point_positions.to(dtype=self.dtype), previous.posterior.x.to(device=self.device, dtype=self.dtype)) ** 2
        kernel = torch.exp(-dist2 / (2.0 * sigma * sigma))
        kernel = kernel / torch.clamp(kernel.sum(dim=0, keepdim=True), min=self.config.epsilon_a)
        priors = posterior_priors.to(device=self.device, dtype=self.dtype) @ kernel.T
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_world_positions_to_visual(
        self,
        source_priors: torch.Tensor | None,
        source_positions_world: torch.Tensor | None,
        token_field: PicfTokenFieldState,
        *,
        sigma_m: float,
    ) -> torch.Tensor | None:
        geometry = token_field.projective_geometry
        if (
            source_priors is None
            or source_positions_world is None
            or source_priors.numel() == 0
            or source_positions_world.numel() == 0
            or geometry is None
            or geometry.visual_ray_world.numel() == 0
        ):
            return None
        positions = source_positions_world.to(device=self.device, dtype=self.dtype)
        rays = _normalize_tensor(geometry.visual_ray_world.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
        origin = geometry.camera_origin_world.to(device=self.device, dtype=self.dtype).reshape(-1, 3).mean(dim=0)
        if source_priors.shape[-1] != positions.shape[0] or rays.shape[0] == 0:
            return None
        vec = positions[:, None, :] - origin[None, None, :]
        depth = torch.sum(vec * rays[None, :, :], dim=-1)
        closest = depth[..., None] * rays[None, :, :]
        perp2 = torch.sum((vec - closest) ** 2, dim=-1)
        sigma = max(float(sigma_m), self.config.epsilon_a)
        kernel = torch.exp(-perp2 / (2.0 * sigma * sigma))
        kernel = torch.where(depth > float(self.config.z_min_m), kernel, torch.zeros_like(kernel))
        kernel = kernel / torch.clamp(kernel.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        priors = source_priors.to(device=self.device, dtype=self.dtype) @ kernel
        return _normalize_rows(priors, eps=self.config.epsilon_a)

    def _mapg_tactile_to_visual(
        self,
        tactile_priors: torch.Tensor | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        return self._mapg_world_positions_to_visual(
            tactile_priors,
            token_field.tactile_positions_world,
            token_field,
            sigma_m=max(float(self.config.mapg_tactile_sigma_m), self.config.epsilon_a),
        )

    def _mapg_posterior_to_visual(
        self,
        posterior_priors: torch.Tensor | None,
        previous: PicfPreviousState | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        positions = None if previous is None else previous.posterior.x
        return self._mapg_world_positions_to_visual(
            posterior_priors,
            positions,
            token_field,
            sigma_m=max(float(self.config.mapg_posterior_sigma_m), self.config.epsilon_a),
        )

    def _build_aqr_anchor_graph(
        self,
        *,
        semantic: _SemanticContext,
        token_field: PicfTokenFieldState,
        previous: PicfPreviousState | None,
        vl_grounding: PicfVLGroundingState | None,
        proprio_token: torch.Tensor | None = None,
    ) -> PicfAnchorPriorGraphState | None:
        if not bool(self.config.aqr_mapg_enabled):
            return None
        if self.aqr_physical_query_tokens is None or self.aqr_task_query_tokens is None:
            return None
        visual_count = int(token_field.visual_tokens.shape[0])
        temporal_count = 0 if token_field.temporal_visual is None else int(token_field.temporal_visual.tokens.shape[0])
        tracklet_count = 0 if token_field.tracklet is None else int(token_field.tracklet.tokens.shape[0])
        proposal_count = 0 if token_field.proposal is None else int(token_field.proposal.tokens.shape[0])
        point_count = int(token_field.point_tokens.shape[0])
        tactile_count = int(token_field.tactile_tokens.shape[0])
        post_count = 0 if previous is None else int(previous.posterior.tokens.shape[0])
        cache_read_state = self._previous_evidence_cache_tokens(previous)
        cache_tokens = cache_read_state.tokens
        cache_scores = cache_read_state.score if cache_read_state.score.numel() > 0 else None
        cache_roles = cache_read_state.role_ids if cache_read_state.role_ids.numel() > 0 else None
        cache_address = cache_read_state.slot_address if cache_read_state.slot_address.numel() > 0 else None
        cache_count = int(cache_tokens.shape[0])
        physical_count = max(int(self.config.aqr_query_count_physical), 0)
        task_count = max(int(self.config.aqr_query_count_task), 0)
        anchor_count = physical_count + task_count
        if anchor_count == 0 or visual_count == 0:
            roles = self._mapg_anchor_roles(anchor_count)
            return PicfAnchorPriorGraphState(
                pg_priors=None,
                visual_priors=torch.zeros((anchor_count, visual_count), device=self.device, dtype=self.dtype),
                point_priors=None,
                tactile_priors=None,
                posterior_priors=None,
                anchor_tokens=torch.zeros((anchor_count, self.config.hidden_dim), device=self.device, dtype=self.dtype),
                anchor_roles=roles,
                anchor_scores=torch.zeros((anchor_count,), device=self.device, dtype=self.dtype),
                anchor_confidence=torch.zeros((anchor_count,), device=self.device, dtype=self.dtype),
                anchor_x=None,
                anchor_S=None,
                geometry_valid=torch.zeros((anchor_count,), device=self.device, dtype=torch.bool),
                obs_slot_assignment=None,
                task_assignment=None,
                modality_confidence=torch.zeros((anchor_count, 10), device=self.device, dtype=self.dtype),
                valid=torch.tensor(False, device=self.device),
            )

        physical_roles = self._mapg_anchor_roles(physical_count)
        task_roles = self._mapg_anchor_roles(task_count)
        roles = torch.cat([physical_roles, task_roles], dim=0)
        query_types = torch.cat(
            [
                torch.zeros((physical_count,), device=self.device, dtype=torch.long),
                torch.ones((task_count,), device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        active_proposals: PicfActiveProposalState | None = None
        proposal_to_graph_assignment: torch.Tensor | None = None
        physical_queries = self.aqr_physical_query_tokens[:physical_count].to(device=self.device, dtype=self.dtype)
        physical_queries, active_proposals, proposal_to_graph_assignment = self._vcap_active_proposal_queries(
            base_queries=physical_queries,
            token_field=token_field,
            previous=previous,
        )
        queries = torch.cat(
            [
                physical_queries,
                self.aqr_task_query_tokens[:task_count].to(device=self.device, dtype=self.dtype),
            ],
            dim=0,
        )
        coverage = torch.cat(
            [
                self._aqr_coverage_codes(physical_count),
                self._aqr_coverage_codes(task_count),
            ],
            dim=0,
        )
        queries = queries + self.aqr_role_embedding(roles) + self.aqr_type_embedding(query_types) + self.aqr_coverage_proj(coverage)
        if proprio_token is not None and self.aqr_proprio_proj is not None:
            proprio_context = self.aqr_proprio_proj(proprio_token.reshape(1, -1))[0]
            queries = queries + proprio_context[None, :]
        if previous is not None and self.aqr_posterior_summary_proj is not None and previous.posterior.tokens.numel() > 0:
            post_alpha = torch.clamp(previous.posterior.alpha.to(device=self.device, dtype=self.dtype), min=0.0)
            denom = torch.clamp(post_alpha.sum(), min=self.config.epsilon_a)
            post_summary = (post_alpha[:, None] * previous.posterior.tokens.to(device=self.device, dtype=self.dtype)).sum(dim=0) / denom
            queries = queries + self.aqr_posterior_summary_proj(post_summary.reshape(1, -1))[0][None, :]
        if task_count > 0 and self.aqr_task_conditioner is not None and semantic.tokens.numel() > 0:
            task_slice = slice(physical_count, physical_count + task_count)
            task_queries, _ = self.aqr_task_conditioner(
                queries[task_slice][None, :],
                semantic.tokens.to(device=self.device, dtype=self.dtype)[None, :],
            )
            queries = torch.cat([queries[:physical_count], task_queries[0]], dim=0)

        visual_bias = self._aqr_pg_visual_bias(
            vl_grounding,
            roles=roles,
            query_types=query_types,
            visual_count=visual_count,
        )
        visual_grid_index = (
            token_field.projective_geometry.visual_grid_index
            if token_field.projective_geometry is not None
            and token_field.projective_geometry.visual_grid_index.shape[0] == visual_count
            else None
        )
        if visual_grid_index is not None:
            ownership_bias = self._aqr_visual_ownership_bias(
                roles=roles,
                visual_count=visual_count,
                visual_grid_index=visual_grid_index,
                vl_grounding=vl_grounding,
            )
            if ownership_bias is not None:
                visual_bias = ownership_bias if visual_bias is None else (visual_bias + ownership_bias)
        temporal_bias = self._aqr_temporal_ownership_bias(roles=roles, temporal=token_field.temporal_visual)
        point_bias = self._aqr_point_bias(token_field, roles)
        point_ownership_bias = self._aqr_point_ownership_bias(token_field, roles)
        if point_ownership_bias is not None:
            point_bias = point_ownership_bias if point_bias is None else (point_bias + point_ownership_bias)
        posterior_bias = self._aqr_posterior_bias(previous, roles)
        cache_bias = None
        if cache_scores is not None and cache_count > 0:
            cache_bias = torch.log(torch.clamp(cache_scores, min=self.config.epsilon_a))[None, :].expand(anchor_count, -1).clone()
            if cache_roles is not None and cache_roles.numel() == cache_count:
                neg = torch.full_like(cache_bias, -1.0e4)
                role_bonus = float(self.config.evidence_cache_role_weight)
                for row, role in enumerate(roles.tolist()):
                    role_int = int(role)
                    role_mask = (cache_roles == 0) if role_int == 0 else (cache_roles != 0)
                    if not bool(role_mask.any().item()):
                        role_mask = torch.ones((cache_count,), device=self.device, dtype=torch.bool)
                    cache_bias[row] = torch.where(role_mask, cache_bias[row] + role_bonus, neg[row])
            if cache_address is not None and cache_address.numel() > 0:
                query_address = self._aqr_cache_query_addresses(previous, physical_count, task_count)
                addr_score = _normalize_tensor(query_address, eps=self.config.epsilon_residual) @ _normalize_tensor(
                    cache_address.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_residual,
                ).T
                cache_bias = cache_bias + (float(self.config.evidence_cache_address_weight) * addr_score)
            if float(self.config.evidence_cache_content_weight) != 0.0:
                content_score = _normalize_tensor(queries, eps=self.config.epsilon_residual) @ _normalize_tensor(
                    cache_tokens.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_residual,
                ).T
                cache_bias = cache_bias + (float(self.config.evidence_cache_content_weight) * content_score)
        tactile_bias = None
        if tactile_count > 0:
            tactile_bias = torch.zeros((anchor_count, tactile_count), device=self.device, dtype=self.dtype)
            attach_to_object = bool(getattr(self.config, "tactile_attach_to_object_owner", True))
            tactile_roles = (roles == 1) if attach_to_object else ((roles == 0) | (roles == 2))
            tactile_bias = torch.where(tactile_roles[:, None], tactile_bias, torch.full_like(tactile_bias, -2.0))

        visual_priors = torch.zeros((anchor_count, visual_count), device=self.device, dtype=self.dtype)
        vjepa_temporal_priors = torch.zeros((anchor_count, temporal_count), device=self.device, dtype=self.dtype) if temporal_count > 0 else None
        pg_priors = None
        point_priors = torch.zeros((anchor_count, point_count), device=self.device, dtype=self.dtype) if point_count > 0 else None
        tactile_priors = torch.zeros((anchor_count, tactile_count), device=self.device, dtype=self.dtype) if tactile_count > 0 else None
        posterior_priors = torch.zeros((anchor_count, post_count), device=self.device, dtype=self.dtype) if post_count > 0 else None
        cache_priors = torch.zeros((anchor_count, cache_count), device=self.device, dtype=self.dtype) if cache_count > 0 else None
        tracklet_priors = torch.zeros((anchor_count, tracklet_count), device=self.device, dtype=self.dtype) if tracklet_count > 0 else None
        proposal_priors = torch.zeros((anchor_count, proposal_count), device=self.device, dtype=self.dtype) if proposal_count > 0 else None
        proposal_point_priors = None
        task_owner_point_priors = None
        proposal_anchor_seed_priors = None
        proposal_anchor_seed_assignment = None
        proposal_anchor_seed_strength = None
        object_candidate_assignment = None
        object_candidate_owner_assignment = None
        object_candidate_owner_point_priors = None
        object_candidate_coverage = None
        object_candidate_background = None
        object_candidate_duplicate_overlap = None
        object_candidate_row_strength = None
        task_owner_visual_prior = None
        task_owner_proposal_score = None
        task_owner_anchor_score = None
        local_priors = None
        local_token_indices = None
        local_source_ids = None
        rounds = max(int(self.config.aqr_query_rounds), 1)
        q = queries[None, :, :]
        for _ in range(rounds):
            round_visual_bias = visual_bias
            owner_visual_bias = self._task_owner_visual_bias(
                task_owner_visual_prior,
                roles=roles,
                query_types=query_types,
                visual_count=visual_count,
            )
            if owner_visual_bias is not None:
                round_visual_bias = owner_visual_bias if round_visual_bias is None else (round_visual_bias + owner_visual_bias)
            q, round_pg_priors, pg_image_bias = self._aqr_pg_image_support_read(
                q,
                semantic,
                query_types=query_types,
                token_field=token_field,
                visual_count=visual_count,
            )
            if round_pg_priors is not None:
                pg_priors = self._aqr_same_role_support_competition(
                    round_pg_priors,
                    roles=roles,
                    query_types=query_types,
                    eps=self.config.epsilon_a,
                )
            if pg_image_bias is not None:
                round_visual_bias = pg_image_bias if round_visual_bias is None else (round_visual_bias + pg_image_bias)
            if self.aqr_visual_reader is not None and visual_count > 0:
                q, visual_weights = self.aqr_visual_reader(
                    q,
                    token_field.visual_tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=round_visual_bias,
                )
                visual_priors = self._aqr_same_role_support_competition(
                    self._aqr_competitive_support(visual_weights, eps=self.config.epsilon_a),
                    roles=roles,
                    query_types=query_types,
                    eps=self.config.epsilon_a,
                )
                next_task_owner_visual_prior = self._task_owner_visual_prior(
                    visual_priors,
                    roles=roles,
                    query_types=query_types,
                )
                if next_task_owner_visual_prior is not None:
                    task_owner_visual_prior = next_task_owner_visual_prior
            if self.aqr_temporal_visual_reader is not None and token_field.temporal_visual is not None and temporal_count > 0:
                q, temporal_weights = self.aqr_temporal_visual_reader(
                    q,
                    token_field.temporal_visual.tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=temporal_bias,
                )
                vjepa_temporal_priors = self._aqr_same_role_support_competition(
                    self._aqr_competitive_support(temporal_weights, eps=self.config.epsilon_a),
                    roles=roles,
                    query_types=query_types,
                    eps=self.config.epsilon_a,
                )
            if self.aqr_point_reader is not None and point_count > 0:
                round_point_bias = point_bias
                owner_point_bias = None
                if token_field.proposal is not None and proposal_count > 0 and task_owner_visual_prior is not None:
                    owner_point_score = self._proposal_scores_from_visual_prior(token_field, task_owner_visual_prior)
                    if owner_point_score is not None:
                        task_owner_proposal_score = owner_point_score
                        owner_point_bias = self._task_owner_proposal_point_bias(
                            task_owner_proposal_score=owner_point_score,
                            roles=roles,
                            query_types=query_types,
                            token_field=token_field,
                            point_count=point_count,
                        )
                if owner_point_bias is not None:
                    round_point_bias = owner_point_bias if round_point_bias is None else (round_point_bias + owner_point_bias)
                q, point_weights = self.aqr_point_reader(
                    q,
                    token_field.point_tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=round_point_bias,
                )
                point_priors = self._aqr_same_role_support_competition(
                    self._aqr_competitive_support(point_weights, eps=self.config.epsilon_a),
                    roles=roles,
                    query_types=query_types,
                    eps=self.config.epsilon_a,
                )
            if self.aqr_tactile_reader is not None and tactile_count > 0:
                q, tactile_weights = self.aqr_tactile_reader(
                    q,
                    token_field.tactile_tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=tactile_bias,
                )
                tactile_priors = self._aqr_competitive_support(tactile_weights, eps=self.config.epsilon_a)
            if self.aqr_posterior_reader is not None and previous is not None and post_count > 0:
                q, posterior_weights = self.aqr_posterior_reader(
                    q,
                    previous.posterior.tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=posterior_bias,
                )
                posterior_priors = self._aqr_competitive_support(posterior_weights, eps=self.config.epsilon_a)
            if self.aqr_cache_reader is not None and cache_count > 0 and float(self.config.evidence_cache_read_weight) > 0.0:
                q_before_cache = q
                cache_read, cache_weights = self.aqr_cache_reader(
                    q,
                    cache_tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=cache_bias,
                )
                cache_scale = max(float(self.config.evidence_cache_read_weight), 0.0)
                q = q_before_cache + (cache_scale * (cache_read - q_before_cache))
                cache_priors = self._aqr_competitive_support(cache_weights, eps=self.config.epsilon_a)
            if self.aqr_tracklet_reader is not None and token_field.tracklet is not None and tracklet_count > 0:
                tracklet_bias = torch.log(
                    torch.clamp(token_field.tracklet.visibility * token_field.tracklet.confidence, min=self.config.epsilon_a)
                )[None, :].expand(anchor_count, -1)
                q_before_track = q
                track_read, track_weights = self.aqr_tracklet_reader(
                    q,
                    token_field.tracklet.tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=tracklet_bias,
                )
                q = q_before_track + (float(self.config.tracklet_read_weight) * (track_read - q_before_track))
                tracklet_priors = self._aqr_same_role_support_competition(
                    self._aqr_competitive_support(track_weights, eps=self.config.epsilon_a),
                    roles=roles,
                    query_types=query_types,
                    eps=self.config.epsilon_a,
                )
            if self.aqr_proposal_reader is not None and token_field.proposal is not None and proposal_count > 0:
                proposal_base_score = torch.clamp(token_field.proposal.objectness, min=self.config.epsilon_a)
                proposal_shape_quality = self._proposal_shape_quality(token_field.proposal)
                context_power = max(float(getattr(self.config, "proposal_context_quality_power", 0.5)), 0.0)
                if proposal_shape_quality is not None and proposal_shape_quality.numel() == proposal_base_score.numel() and context_power > 0.0:
                    proposal_base_score = proposal_base_score * torch.pow(
                        torch.clamp(proposal_shape_quality, min=self.config.epsilon_a),
                        context_power,
                    )
                proposal_bias = torch.log(
                    torch.clamp(proposal_base_score, min=self.config.epsilon_a)
                )[None, :].expand(anchor_count, -1)
                owner_proposal_score = self._proposal_scores_from_visual_prior(token_field, task_owner_visual_prior)
                owner_proposal_bias = self._task_owner_proposal_bias(
                    owner_proposal_score,
                    roles=roles,
                    query_types=query_types,
                    proposal_count=proposal_count,
                )
                if owner_proposal_bias is not None:
                    proposal_bias = proposal_bias + owner_proposal_bias
                    task_owner_proposal_score = owner_proposal_score
                q_before_prop = q
                prop_read, prop_weights = self.aqr_proposal_reader(
                    q,
                    token_field.proposal.tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=proposal_bias,
                )
                q = q_before_prop + (float(self.config.proposal_read_weight) * (prop_read - q_before_prop))
                proposal_priors = self._aqr_same_role_support_competition(
                    self._aqr_competitive_support(prop_weights, eps=self.config.epsilon_a),
                    roles=roles,
                    query_types=query_types,
                    eps=self.config.epsilon_a,
                )
            # Archived legacy path: this top-k reread is intentionally not part
            # of the production belief update unless both flags are explicitly
            # enabled. The maintained path relies on AQR supports plus normalized
            # posterior recycle gating; local reread remains only for ablations.
            local_refinement_active = (
                bool(getattr(self.config, "legacy_local_refinement_opt_in", False))
                and bool(self.config.local_refinement_enabled)
                and float(self.config.local_refinement_weight) != 0.0
                and int(self.config.local_refinement_topk) > 0
            )
            if local_refinement_active:
                local_vectors: list[torch.Tensor] = []
                local_masses: list[torch.Tensor] = []
                local_weight_rows: list[torch.Tensor] = []
                local_index_rows: list[torch.Tensor] = []
                local_source_rows: list[torch.Tensor] = []
                local_token_offset = 0

                def _add_local_component(
                    priors: torch.Tensor | None,
                    tokens: torch.Tensor | None,
                    source_id: int,
                    token_xy_norm: torch.Tensor | None = None,
                ) -> None:
                    nonlocal local_token_offset
                    if priors is None or tokens is None or priors.numel() == 0 or tokens.numel() == 0:
                        return
                    if priors.shape[-1] != tokens.shape[0]:
                        return
                    weights = _normalize_rows(torch.clamp(priors.to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
                    binding_weight = float(getattr(self.config, "local_refinement_binding_weight", 0.0))
                    if binding_weight != 0.0 and q.numel() > 0:
                        token_binding = self._binding_keys(tokens.to(device=self.device, dtype=self.dtype))
                        query_binding = self._binding_keys(q[0].to(device=self.device, dtype=self.dtype))
                        if (
                            query_binding.shape[0] == weights.shape[0]
                            and token_binding.shape[0] == weights.shape[-1]
                            and query_binding.shape[-1] == token_binding.shape[-1]
                        ):
                            binding_logits = query_binding @ token_binding.T
                            local_logits = torch.log(torch.clamp(weights, min=self.config.epsilon_a))
                            local_logits = local_logits + (binding_weight * binding_logits)
                            weights = torch.softmax(local_logits, dim=-1)
                    topk = min(max(int(self.config.local_refinement_topk), 1), int(weights.shape[-1]))
                    _, top_indices = torch.topk(weights, k=topk, dim=-1)
                    top_values = torch.gather(weights, dim=-1, index=top_indices)
                    gathered = tokens.to(device=self.device, dtype=self.dtype).index_select(0, top_indices.reshape(-1))
                    gathered = gathered.reshape(int(weights.shape[0]), topk, -1)
                    local_vectors.append((top_values[..., None] * gathered).sum(dim=1))
                    local_masses.append(top_values.sum(dim=-1))
                    local_weight_rows.append(top_values)
                    local_index_rows.append(top_indices.to(device=self.device, dtype=torch.long) + int(local_token_offset))
                    local_source_rows.append(torch.full_like(top_indices, int(source_id), dtype=torch.long, device=self.device))
                    local_token_offset += int(tokens.shape[0])

                visual_xy = (
                    token_field.projective_geometry.visual_grid_norm
                    if token_field.projective_geometry is not None
                    and token_field.projective_geometry.visual_grid_norm.shape == (visual_count, 2)
                    else None
                )
                temporal_xy = None
                if token_field.temporal_visual is not None and token_field.temporal_visual.grid_index.numel() > 0:
                    temporal_hw = token_field.temporal_visual.grid_hw.reshape(-1)
                    if temporal_hw.numel() >= 2:
                        temporal_xy = _grid_index_to_norm(
                            token_field.temporal_visual.grid_index.to(device=self.device, dtype=self.dtype),
                            height=int(temporal_hw[0].item()),
                            width=int(temporal_hw[1].item()),
                        )
                point_xy = (
                    token_field.projective_geometry.point_proj_grid_norm
                    if token_field.projective_geometry is not None
                    and token_field.projective_geometry.point_proj_grid_norm.shape == (point_count, 2)
                    else None
                )
                tracklet_xy = (
                    ((torch.clamp(token_field.tracklet.xy_norm.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0) * 2.0) - 1.0)
                    if token_field.tracklet is not None and token_field.tracklet.xy_norm.numel() > 0
                    else None
                )
                proposal_xy = (
                    ((torch.clamp(token_field.proposal.centers_xy.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0) * 2.0) - 1.0)
                    if token_field.proposal is not None and token_field.proposal.centers_xy.numel() > 0
                    else None
                )

                _add_local_component(visual_priors, token_field.visual_tokens, 1, visual_xy)
                _add_local_component(
                    vjepa_temporal_priors,
                    None if token_field.temporal_visual is None else token_field.temporal_visual.tokens,
                    2,
                    temporal_xy,
                )
                _add_local_component(point_priors, token_field.point_tokens, 3, point_xy)
                _add_local_component(tracklet_priors, None if token_field.tracklet is None else token_field.tracklet.tokens, 4, tracklet_xy)
                _add_local_component(proposal_priors, None if token_field.proposal is None else token_field.proposal.tokens, 5, proposal_xy)
                if local_vectors:
                    local_read = torch.stack(local_vectors, dim=0).sum(dim=0)
                    local_mass = torch.stack(local_masses, dim=0).sum(dim=0)
                    local_read = local_read / torch.clamp(local_mass[:, None], min=self.config.epsilon_a)
                    q = q + (float(self.config.local_refinement_weight) * (local_read[None, :, :] - q))
                    local_priors = _normalize_rows(torch.cat(local_weight_rows, dim=-1), eps=self.config.epsilon_a)
                    local_token_indices = torch.cat(local_index_rows, dim=-1) if local_index_rows else None
                    local_source_ids = torch.cat(local_source_rows, dim=-1) if local_source_rows else None
            if self.aqr_query_self is not None:
                q = self.aqr_query_self(q)

        if proposal_priors is not None and point_count > 0:
            proposal_point_priors = self._proposal_priors_to_point_priors(proposal_priors, token_field)
            if proposal_point_priors is not None:
                bridge_weight = min(max(float(getattr(self.config, "proposal_point_bridge_weight", 0.35)), 0.0), 1.0)
                if bridge_weight > 0.0:
                    if point_priors is not None and point_priors.numel() > 0 and point_priors.shape == proposal_point_priors.shape:
                        point_priors = _normalize_rows(
                            ((1.0 - bridge_weight) * torch.clamp(point_priors, min=0.0))
                            + (bridge_weight * torch.clamp(proposal_point_priors, min=0.0)),
                            eps=self.config.epsilon_a,
                        )
                    else:
                        point_priors = proposal_point_priors

        if task_owner_proposal_score is not None and point_count > 0:
            task_owner_point_priors = self._task_owner_proposal_to_point_priors(
                task_owner_proposal_score=task_owner_proposal_score,
                roles=roles,
                query_types=query_types,
                token_field=token_field,
                row_count=anchor_count,
            )
            if task_owner_point_priors is not None:
                owner_point_weight = min(
                    max(float(getattr(self.config, "task_owner_proposal_point_bridge_weight", 0.0)), 0.0),
                    1.0,
                )
                if owner_point_weight > 0.0:
                    if point_priors is not None and point_priors.numel() > 0 and point_priors.shape == task_owner_point_priors.shape:
                        point_priors = _normalize_rows(
                            ((1.0 - owner_point_weight) * torch.clamp(point_priors, min=0.0))
                            + (owner_point_weight * torch.clamp(task_owner_point_priors, min=0.0)),
                            eps=self.config.epsilon_a,
                        )
                    else:
                        point_priors = task_owner_point_priors

        if task_owner_proposal_score is not None and point_count > 0:
            proposal_anchor_seed = self._proposal_anchor_seed_transport(
                task_owner_proposal_score=task_owner_proposal_score,
                roles=roles,
                query_types=query_types,
                token_field=token_field,
                row_count=anchor_count,
                point_count=point_count,
            )
            if proposal_anchor_seed is not None:
                proposal_anchor_seed_priors, proposal_anchor_seed_assignment, proposal_anchor_seed_strength = proposal_anchor_seed
                seed_weight = min(
                    max(float(getattr(self.config, "proposal_anchor_seed_weight", 0.0)), 0.0),
                    1.0,
                )
                if seed_weight > 0.0:
                    seed_rows = _row_has_mass(proposal_anchor_seed_priors, eps=self.config.epsilon_a)
                    if point_priors is None or point_priors.numel() == 0 or point_priors.shape != proposal_anchor_seed_priors.shape:
                        point_priors = proposal_anchor_seed_priors
                    else:
                        row_mix = (seed_weight * torch.clamp(proposal_anchor_seed_strength, min=0.0, max=1.0))[:, None]
                        mixed = ((1.0 - row_mix) * torch.clamp(point_priors, min=0.0)) + (
                            row_mix * torch.clamp(proposal_anchor_seed_priors, min=0.0)
                        )
                        point_priors = torch.where(seed_rows[:, None], _normalize_rows(mixed, eps=self.config.epsilon_a), point_priors)
                token_weight = min(
                    max(float(getattr(self.config, "proposal_anchor_seed_token_weight", 0.0)), 0.0),
                    1.0,
                )
                if (
                    token_weight > 0.0
                    and proposal_anchor_seed_assignment is not None
                    and token_field.proposal is not None
                    and token_field.proposal.tokens.numel() > 0
                    and proposal_anchor_seed_assignment.shape[-1] == token_field.proposal.tokens.shape[0]
                ):
                    seed_tokens = proposal_anchor_seed_assignment @ token_field.proposal.tokens.to(device=self.device, dtype=self.dtype)
                    seed_rows = _row_has_mass(proposal_anchor_seed_assignment, eps=self.config.epsilon_a)
                    q_seeded = q[0].clone()
                    row_mix = (token_weight * torch.clamp(proposal_anchor_seed_strength, min=0.0, max=1.0))[:, None]
                    q_seeded = torch.where(
                        seed_rows[:, None],
                        q_seeded + (row_mix * (seed_tokens - q_seeded)),
                        q_seeded,
                    )
                    q = q_seeded[None, :, :]

        object_candidate = self._proposal_object_candidate_assignment(
            roles=roles,
            query_types=query_types,
            token_field=token_field,
            point_priors=point_priors,
            proposal_priors=proposal_priors,
            task_owner_proposal_score=task_owner_proposal_score,
            proposal_anchor_seed_assignment=proposal_anchor_seed_assignment,
            row_count=anchor_count,
            point_count=point_count,
        )
        if object_candidate is not None:
            (
                object_candidate_assignment,
                object_candidate_owner_assignment,
                object_candidate_owner_point_priors,
                object_candidate_coverage,
                object_candidate_background,
                object_candidate_duplicate_overlap,
                candidate_point_priors,
                candidate_proposal_priors,
                object_candidate_row_strength,
            ) = object_candidate
            candidate_point_mix = min(max(float(getattr(self.config, "object_candidate_point_mix", 0.5)), 0.0), 1.0)
            if (
                candidate_point_mix > 0.0
                and candidate_point_priors is not None
                and candidate_point_priors.numel() > 0
                and int(candidate_point_priors.shape[-1]) == point_count
            ):
                candidate_rows = _row_has_mass(candidate_point_priors, eps=self.config.epsilon_a)
                if point_priors is None or point_priors.numel() == 0 or point_priors.shape != candidate_point_priors.shape:
                    point_priors = candidate_point_priors
                else:
                    mixed = (
                        ((1.0 - candidate_point_mix) * torch.clamp(point_priors, min=0.0))
                        + (candidate_point_mix * torch.clamp(candidate_point_priors, min=0.0))
                    )
                    point_priors = torch.where(candidate_rows[:, None], _normalize_rows(mixed, eps=self.config.epsilon_a), point_priors)
            candidate_proposal_mix = min(max(float(getattr(self.config, "object_candidate_proposal_mix", 0.35)), 0.0), 1.0)
            if (
                candidate_proposal_mix > 0.0
                and candidate_proposal_priors is not None
                and candidate_proposal_priors.numel() > 0
                and proposal_priors is not None
                and proposal_priors.shape == candidate_proposal_priors.shape
            ):
                candidate_rows = _row_has_mass(candidate_proposal_priors, eps=self.config.epsilon_a)
                mixed = (
                    ((1.0 - candidate_proposal_mix) * torch.clamp(proposal_priors, min=0.0))
                    + (candidate_proposal_mix * torch.clamp(candidate_proposal_priors, min=0.0))
                )
                proposal_priors = torch.where(candidate_rows[:, None], _normalize_rows(mixed, eps=self.config.epsilon_a), proposal_priors)

        anchor_tokens = q[0]
        visual_conf = _distribution_confidence(visual_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        temporal_conf = _distribution_confidence(vjepa_temporal_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        pg_conf = _distribution_confidence(pg_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        point_conf = _distribution_confidence(point_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        tactile_conf = _distribution_confidence(tactile_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        post_conf = _distribution_confidence(posterior_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        cache_conf = _distribution_confidence(cache_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        tracklet_conf = _distribution_confidence(tracklet_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        proposal_conf = _distribution_confidence(proposal_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        zero_conf = torch.zeros((anchor_count,), device=self.device, dtype=self.dtype)
        modality_conf = torch.stack(
            [
                zero_conf,
                zero_conf if visual_conf is None else visual_conf,
                zero_conf if temporal_conf is None else temporal_conf,
                zero_conf if pg_conf is None else pg_conf,
                zero_conf if point_conf is None else point_conf,
                zero_conf if tactile_conf is None else tactile_conf,
                zero_conf if post_conf is None else post_conf,
                zero_conf if cache_conf is None else cache_conf,
                zero_conf if tracklet_conf is None else tracklet_conf,
                zero_conf if proposal_conf is None else proposal_conf,
            ],
            dim=-1,
        )
        anchor_scores = torch.max(visual_priors, dim=-1).values
        if point_priors is not None and point_priors.numel() > 0:
            anchor_scores = anchor_scores + torch.max(point_priors, dim=-1).values
        if proposal_priors is not None and proposal_priors.numel() > 0:
            anchor_scores = anchor_scores + torch.max(proposal_priors, dim=-1).values
        if active_proposals is not None and active_proposals.active_prob.numel() > 0:
            proposal_active = torch.clamp(active_proposals.active_prob.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
            active_width = min(int(proposal_active.numel()), physical_count)
            if active_width > 0:
                proposal_score_delta = torch.zeros_like(anchor_scores)
                proposal_score_delta = proposal_score_delta + torch.nn.functional.pad(
                    proposal_active[:active_width],
                    (0, max(int(anchor_scores.numel()) - active_width, 0)),
                )[: anchor_scores.numel()]
                anchor_scores = anchor_scores + proposal_score_delta
        task_owner_anchor_score = self._task_owner_anchor_score(
            proposal_priors=proposal_priors,
            task_owner_proposal_score=task_owner_proposal_score,
            visual_priors=visual_priors,
            task_owner_visual_prior=task_owner_visual_prior,
            roles=roles,
            query_types=query_types,
        )
        if task_owner_anchor_score is not None:
            anchor_scores = anchor_scores + torch.clamp(task_owner_anchor_score, min=0.0)
        if proposal_anchor_seed_strength is not None and proposal_anchor_seed_strength.numel() == anchor_scores.numel():
            anchor_scores = anchor_scores + torch.clamp(proposal_anchor_seed_strength, min=0.0)
        if object_candidate_row_strength is not None and object_candidate_row_strength.numel() == anchor_scores.numel():
            anchor_scores = anchor_scores + (
                float(getattr(self.config, "object_candidate_anchor_score_weight", 1.0))
                * torch.clamp(object_candidate_row_strength, min=0.0)
            )
        anchor_conf = torch.clamp(modality_conf.max(dim=-1).values, min=0.0, max=1.0)
        if proposal_anchor_seed_strength is not None and proposal_anchor_seed_strength.numel() == anchor_conf.numel():
            anchor_conf = torch.maximum(anchor_conf, torch.clamp(proposal_anchor_seed_strength, min=0.0, max=1.0))
        if object_candidate_row_strength is not None and object_candidate_row_strength.numel() == anchor_conf.numel():
            anchor_conf = torch.maximum(anchor_conf, torch.clamp(object_candidate_row_strength, min=0.0, max=1.0))
        if active_proposals is not None and active_proposals.active_prob.numel() > 0:
            proposal_active = torch.clamp(active_proposals.active_prob.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
            active_width = min(int(proposal_active.numel()), physical_count)
            if active_width > 0:
                proposal_conf = torch.nn.functional.pad(
                    proposal_active[:active_width],
                    (0, max(int(anchor_conf.numel()) - active_width, 0)),
                )[: anchor_conf.numel()]
                proposal_mask = torch.arange(int(anchor_conf.numel()), device=self.device) < int(active_width)
                anchor_conf = torch.where(proposal_mask, torch.maximum(anchor_conf, proposal_conf), anchor_conf)
        anchor_x = None
        anchor_S = None
        geometry_valid = torch.zeros((anchor_count,), device=self.device, dtype=torch.bool)
        if point_priors is not None and point_count > 0:
            point_positions = self._world_point_positions(token_field)
            anchor_x = point_priors @ point_positions
            anchor_S = _weighted_cov(point_positions, point_priors, anchor_x, self.config)
            geometry_valid = _row_has_mass(point_priors, eps=self.config.epsilon_a)
        elif posterior_priors is not None and previous is not None:
            anchor_x = posterior_priors @ previous.posterior.x.to(device=self.device, dtype=self.dtype)
            anchor_S = torch.eye(3, device=self.device, dtype=self.dtype)[None, :, :].expand(anchor_count, -1, -1).clone()
            geometry_valid = _row_has_mass(posterior_priors, eps=self.config.epsilon_a)
        anchor_active = self._aqr_active_slot_mask(
            roles=roles,
            visual_priors=visual_priors,
            point_priors=point_priors,
            temporal_priors=vjepa_temporal_priors,
            pg_priors=pg_priors,
            proposal_priors=proposal_priors,
            anchor_x=anchor_x,
            geometry_valid=geometry_valid,
            anchor_scores=anchor_scores,
            anchor_confidence=anchor_conf,
        )
        anchor_downstream_weight = self._aqr_downstream_slot_weights(
            roles=roles,
            visual_priors=visual_priors,
            active=anchor_active,
            point_priors=point_priors,
            temporal_priors=vjepa_temporal_priors,
            pg_priors=pg_priors,
            proposal_priors=proposal_priors,
            anchor_x=anchor_x,
            geometry_valid=geometry_valid,
            anchor_scores=anchor_scores,
            anchor_confidence=anchor_conf,
        )
        active_proposals = self._finalize_vcap_proposal_state(
            active_proposals,
            visual_priors=visual_priors,
            point_priors=point_priors,
            temporal_priors=vjepa_temporal_priors,
            pg_priors=pg_priors,
            proposal_priors=proposal_priors,
            physical_count=physical_count,
        )
        return PicfAnchorPriorGraphState(
            pg_priors=pg_priors,
            visual_priors=visual_priors,
            point_priors=point_priors,
            tactile_priors=tactile_priors,
            posterior_priors=posterior_priors,
            anchor_tokens=anchor_tokens,
            anchor_roles=roles,
            anchor_scores=anchor_scores,
            anchor_confidence=anchor_conf,
            anchor_x=anchor_x,
            anchor_S=anchor_S,
            geometry_valid=geometry_valid,
            obs_slot_assignment=None,
            task_assignment=None,
            modality_confidence=modality_conf,
            valid=torch.tensor(True, device=self.device),
            anchor_active=anchor_active,
            anchor_downstream_weight=anchor_downstream_weight,
            vjepa_temporal_priors=vjepa_temporal_priors,
            cache_priors=cache_priors,
            tracklet_priors=tracklet_priors,
            proposal_priors=proposal_priors,
            proposal_point_priors=proposal_point_priors,
            task_owner_point_priors=task_owner_point_priors,
            proposal_anchor_seed_priors=proposal_anchor_seed_priors,
            proposal_anchor_seed_assignment=proposal_anchor_seed_assignment,
            task_owner_visual_prior=task_owner_visual_prior,
            task_owner_proposal_score=task_owner_proposal_score,
            task_owner_anchor_score=task_owner_anchor_score,
            local_priors=local_priors,
            local_token_indices=local_token_indices,
            local_source_ids=local_source_ids,
            slot_address=self._physical_query_addresses(previous, physical_count) if self.aqr_physical_query_tokens is not None else None,
            slot_content=anchor_tokens,
            support_uncertainty=1.0 - anchor_conf,
            support_signature=modality_conf,
            active_proposals=active_proposals,
            proposal_to_graph_assignment=proposal_to_graph_assignment,
            proposal_unexplained_evidence=None if active_proposals is None else active_proposals.unexplained_evidence,
            proposal_duplicate_cost=None if active_proposals is None else active_proposals.duplicate_score,
            proposal_count=None if active_proposals is None else active_proposals.active_prob.sum(),
            object_candidate_assignment=object_candidate_assignment,
            object_candidate_owner_assignment=object_candidate_owner_assignment,
            object_candidate_owner_point_priors=object_candidate_owner_point_priors,
            object_candidate_coverage=object_candidate_coverage,
            object_candidate_background=object_candidate_background,
            object_candidate_duplicate_overlap=object_candidate_duplicate_overlap,
        )

    def _object_explanation_masks(
        self,
        priors: torch.Tensor | None,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            priors is None
            or priors.numel() == 0
            or quality.numel() == 0
            or priors.shape[0] != quality.shape[0]
        ):
            return None, None
        prob = _normalize_rows(
            torch.clamp(priors.to(device=self.device, dtype=self.dtype), min=0.0),
            eps=self.config.epsilon_a,
        )
        q = torch.clamp(quality.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0, max=1.0)
        weighted = prob * q[:, None]
        bg_prior = max(float(self.config.object_explanation_background_prior), self.config.epsilon_a)
        background_mass = torch.full((prob.shape[-1],), bg_prior, device=self.device, dtype=self.dtype)
        denom = torch.clamp(weighted.sum(dim=0) + background_mass, min=self.config.epsilon_a)
        return weighted / denom[None, :], background_mass / denom

    def _object_explanation_feature_variance(
        self,
        mask: torch.Tensor | None,
        tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if mask is None or tokens is None or mask.numel() == 0 or tokens.numel() == 0:
            return None
        if mask.shape[-1] != tokens.shape[0]:
            return None
        m = torch.clamp(mask.to(device=self.device, dtype=self.dtype), min=0.0)
        z = _normalize_tensor(tokens.to(device=self.device, dtype=self.dtype), eps=float(self.config.object_explanation_feature_eps))
        mass = torch.clamp(m.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        proto = _normalize_tensor((m @ z) / mass, eps=float(self.config.object_explanation_feature_eps))
        cos = torch.clamp(proto @ z.T, min=-1.0, max=1.0)
        variance = (m * (1.0 - cos)).sum(dim=-1) / mass.squeeze(-1)
        return torch.clamp(variance, min=0.0)

    def _object_explanation_point_variance(
        self,
        mask: torch.Tensor | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor | None:
        if mask is None or mask.numel() == 0:
            return None
        points = self._world_point_positions(token_field)
        if points.numel() == 0 or mask.shape[-1] != points.shape[0]:
            return None
        m = torch.clamp(mask.to(device=self.device, dtype=self.dtype), min=0.0)
        x = points.to(device=self.device, dtype=self.dtype)
        mass = torch.clamp(m.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        center = (m @ x) / mass
        dist2 = torch.sum((x[None, :, :] - center[:, None, :]) ** 2, dim=-1)
        sigma2 = max(float(self.config.object_explanation_point_sigma_m) ** 2, self.config.epsilon_s)
        variance = (m * dist2).sum(dim=-1) / mass.squeeze(-1)
        return torch.clamp(variance / sigma2, min=0.0)

    def _object_explanation_contact_score(
        self,
        mask: torch.Tensor | None,
        token_field: PicfTokenFieldState,
    ) -> torch.Tensor:
        if (
            mask is None
            or mask.numel() == 0
            or token_field.tactile_contact_prob is None
            or token_field.tactile_contact_prob.numel() == 0
        ):
            return torch.zeros((), device=self.device, dtype=self.dtype)
        contact = token_field.tactile_contact_prob.to(device=self.device, dtype=self.dtype).reshape(-1)
        if contact.numel() != mask.shape[-1] and token_field.tactile_group_ids is not None:
            group_ids = token_field.tactile_group_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            if group_ids.numel() == mask.shape[-1] and contact.numel() > 0:
                group_ids = torch.clamp(group_ids, min=0, max=contact.numel() - 1)
                contact = contact.index_select(0, group_ids)
        if contact.numel() != mask.shape[-1]:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        contact = torch.clamp(contact, min=0.0, max=1.0)
        if not bool((contact.sum() > self.config.epsilon_a).item()):
            return torch.zeros((), device=self.device, dtype=self.dtype)
        object_mass = torch.clamp(mask.to(device=self.device, dtype=self.dtype).sum(dim=0), min=0.0, max=1.0)
        return (object_mass * contact).sum() / torch.clamp(contact.sum(), min=self.config.epsilon_a)

    def _object_explanation_duplicate_overlap(
        self,
        masks: Sequence[torch.Tensor | None],
        *,
        anchor_count: int,
    ) -> torch.Tensor:
        overlap = torch.zeros((anchor_count, anchor_count), device=self.device, dtype=self.dtype)
        used = 0
        for mask in masks:
            if mask is None or mask.numel() == 0 or mask.shape[0] != anchor_count:
                continue
            m = torch.clamp(mask.to(device=self.device, dtype=self.dtype), min=0.0)
            denom = torch.sqrt(torch.clamp((m * m).sum(dim=-1, keepdim=True), min=self.config.epsilon_a))
            norm = m / denom
            overlap = torch.maximum(overlap, norm @ norm.T)
            used += 1
        if used == 0:
            return overlap
        eye = torch.eye(anchor_count, device=self.device, dtype=torch.bool)
        return overlap.masked_fill(eye, 0.0)

    def _build_object_explanation_measurements(
        self,
        token_field: PicfTokenFieldState,
        graph: PicfAnchorPriorGraphState | None,
    ) -> PicfObjectExplanationState | None:
        if (
            not bool(getattr(self.config, "object_explanation_enabled", True))
            or graph is None
            or not bool(graph.valid.item())
            or graph.anchor_tokens.shape[0] == 0
        ):
            return None
        anchor_count = int(graph.anchor_tokens.shape[0])
        quality = torch.clamp(graph.anchor_confidence.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0, max=1.0)
        if quality.numel() != anchor_count:
            quality = torch.ones((anchor_count,), device=self.device, dtype=self.dtype)
        if graph.anchor_downstream_weight is not None and graph.anchor_downstream_weight.numel() == anchor_count:
            quality = quality * torch.clamp(graph.anchor_downstream_weight.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        if not bool((quality.sum() > self.config.epsilon_a).item()):
            score = torch.clamp(graph.anchor_scores.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0)
            if score.numel() == anchor_count and bool((score.max() > self.config.epsilon_a).item()):
                quality = score / torch.clamp(score.max(), min=self.config.epsilon_a)
        visual_mask, visual_bg = self._object_explanation_masks(graph.visual_priors, quality)
        temporal_mask, temporal_bg = self._object_explanation_masks(graph.vjepa_temporal_priors, quality)
        point_mask, point_bg = self._object_explanation_masks(graph.point_priors, quality)
        tactile_mask, tactile_bg = self._object_explanation_masks(graph.tactile_priors, quality)
        tracklet_mask, tracklet_bg = self._object_explanation_masks(graph.tracklet_priors, quality)
        proposal_mask, proposal_bg = self._object_explanation_masks(graph.proposal_priors, quality)

        feature_terms: list[torch.Tensor] = []
        for mask, tokens in (
            (visual_mask, token_field.visual_tokens),
            (temporal_mask, None if token_field.temporal_visual is None else token_field.temporal_visual.tokens),
            (point_mask, token_field.point_tokens),
            (tactile_mask, token_field.tactile_tokens),
            (tracklet_mask, None if token_field.tracklet is None else token_field.tracklet.tokens),
            (proposal_mask, None if token_field.proposal is None else token_field.proposal.tokens),
        ):
            variance = self._object_explanation_feature_variance(mask, tokens)
            if variance is not None and variance.numel() == anchor_count:
                feature_terms.append(variance)
        if feature_terms:
            anchor_feature_variance = torch.stack(feature_terms, dim=0).mean(dim=0)
        else:
            anchor_feature_variance = torch.zeros((anchor_count,), device=self.device, dtype=self.dtype)

        point_variance = self._object_explanation_point_variance(point_mask, token_field)
        if point_variance is None or point_variance.numel() != anchor_count:
            point_variance = torch.zeros((anchor_count,), device=self.device, dtype=self.dtype)

        contact_score = self._object_explanation_contact_score(tactile_mask, token_field)
        duplicate_overlap = self._object_explanation_duplicate_overlap(
            (visual_mask, temporal_mask, point_mask, tactile_mask, tracklet_mask, proposal_mask),
            anchor_count=anchor_count,
        )
        evidence_quality = torch.exp(-0.5 * torch.clamp(anchor_feature_variance, min=0.0, max=8.0))
        point_quality = torch.exp(-0.5 * torch.clamp(point_variance, min=0.0, max=8.0))
        explanation_quality = torch.clamp(quality * torch.sqrt(torch.clamp(evidence_quality * point_quality, min=0.0)), min=0.0, max=1.0)
        graph.object_explanation_quality = explanation_quality
        graph.object_explanation_duplicate_overlap = duplicate_overlap
        has_mask = any(mask is not None for mask in (visual_mask, temporal_mask, point_mask, tactile_mask, tracklet_mask, proposal_mask))
        return PicfObjectExplanationState(
            object_mask_visual=visual_mask,
            background_mask_visual=visual_bg,
            object_mask_temporal=temporal_mask,
            background_mask_temporal=temporal_bg,
            object_mask_point=point_mask,
            background_mask_point=point_bg,
            object_mask_tactile=tactile_mask,
            background_mask_tactile=tactile_bg,
            object_mask_tracklet=tracklet_mask,
            background_mask_tracklet=tracklet_bg,
            object_mask_proposal=proposal_mask,
            background_mask_proposal=proposal_bg,
            anchor_quality=explanation_quality,
            anchor_duplicate_overlap=duplicate_overlap,
            anchor_feature_variance=anchor_feature_variance,
            point_spatial_variance=point_variance,
            contact_explanation_score=contact_score,
            valid=torch.tensor(bool(has_mask), device=self.device),
            candidate_coverage=graph.object_candidate_coverage,
            candidate_background=graph.object_candidate_background,
            candidate_duplicate_overlap=graph.object_candidate_duplicate_overlap,
        )

    def _mapg_slot_assignment(
        self,
        graph: PicfAnchorPriorGraphState | None,
        slot_role_ids: torch.Tensor,
        *,
        slot_tokens: torch.Tensor | None = None,
        slot_point_priors: torch.Tensor | None = None,
        slot_visual_priors: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if graph is None or not bool(graph.valid.item()) or graph.anchor_tokens.shape[0] == 0 or slot_role_ids.numel() == 0:
            return None
        k = int(graph.anchor_tokens.shape[0])
        slot_count = int(slot_role_ids.numel())
        roles = graph.anchor_roles.to(device=self.device, dtype=torch.long)
        slot_roles = slot_role_ids.to(device=self.device, dtype=torch.long)
        allowed = torch.zeros((slot_count, k), device=self.device, dtype=torch.bool)
        for slot_index, role in enumerate(slot_roles.tolist()):
            role_int = int(role)
            if role_int == 0:
                mask = roles == 0
            elif role_int == 1:
                mask = (roles == 1) | (roles == 2) | (roles == 3)
            else:
                mask = roles == role_int
            if not bool(mask.any().item()):
                mask = torch.ones((k,), device=self.device, dtype=torch.bool)
            allowed[slot_index] = mask
        scores = torch.clamp(graph.anchor_scores.to(device=self.device, dtype=self.dtype), min=self.config.epsilon_a)
        confidence = torch.clamp(graph.anchor_confidence.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
        scores = torch.clamp(scores * torch.clamp(confidence, min=float(self.config.mapg_confidence_floor)), min=self.config.epsilon_a)
        if (
            bool(getattr(self.config, "object_explanation_feed_quality_to_assignment", True))
            and graph.object_explanation_quality is not None
            and graph.object_explanation_quality.numel() == k
        ):
            explanation_quality = torch.clamp(
                graph.object_explanation_quality.to(device=self.device, dtype=self.dtype).reshape(-1),
                min=0.0,
                max=1.0,
            )
            floor = min(max(float(self.config.object_explanation_min_slot_quality), 0.0), 1.0)
            explanation_quality = torch.clamp(explanation_quality, min=floor, max=1.0)
            scores = torch.clamp(scores * explanation_quality, min=self.config.epsilon_a)
        active_mask = None
        if graph.anchor_active is not None and graph.anchor_active.numel() == k:
            active_mask = graph.anchor_active.to(device=self.device, dtype=self.dtype) > 0.5
        temperature = max(float(self.config.mapg_assignment_temperature), self.config.epsilon_a)
        mix = min(max(float(self.config.mapg_assignment_quality_uniform_mix), 0.0), 1.0)
        assignment = torch.zeros((slot_count, k), device=self.device, dtype=self.dtype)
        for role in torch.unique(slot_roles).tolist():
            rows = torch.nonzero(slot_roles == int(role), as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            candidate_mask = allowed.index_select(0, rows).any(dim=0)
            if active_mask is not None and bool((candidate_mask & active_mask).any().item()):
                candidate_mask = candidate_mask & active_mask
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() == 0:
                continue
            local_allowed = allowed.index_select(0, rows).index_select(1, candidate_indices)
            local_scores = scores.index_select(0, candidate_indices)
            logits = torch.log(local_scores)[None, :].expand(int(rows.numel()), -1) / temperature
            if slot_tokens is not None and slot_tokens.shape[0] == slot_count:
                slot_h = fn.normalize(slot_tokens.to(device=self.device, dtype=self.dtype).index_select(0, rows), dim=-1)
                anchor_h = fn.normalize(graph.anchor_tokens.to(device=self.device, dtype=self.dtype).index_select(0, candidate_indices), dim=-1)
                logits = logits + ((slot_h @ anchor_h.T) / temperature)
            if (
                slot_point_priors is not None
                and graph.point_priors is not None
                and slot_point_priors.shape[0] == slot_count
                and slot_point_priors.shape[-1] == graph.point_priors.shape[-1]
            ):
                slot_p = _normalize_rows(slot_point_priors.to(device=self.device, dtype=self.dtype).index_select(0, rows), eps=self.config.epsilon_a)
                anchor_p = _normalize_rows(graph.point_priors.to(device=self.device, dtype=self.dtype).index_select(0, candidate_indices), eps=self.config.epsilon_a)
                overlap = slot_p @ anchor_p.T
                logits = logits + torch.log(torch.clamp(overlap, min=self.config.epsilon_a))
            if (
                slot_visual_priors is not None
                and slot_visual_priors.shape[0] == slot_count
                and slot_visual_priors.shape[-1] == graph.visual_priors.shape[-1]
            ):
                slot_v = _normalize_rows(slot_visual_priors.to(device=self.device, dtype=self.dtype).index_select(0, rows), eps=self.config.epsilon_a)
                anchor_v = _normalize_rows(graph.visual_priors.to(device=self.device, dtype=self.dtype).index_select(0, candidate_indices), eps=self.config.epsilon_a)
                overlap = slot_v @ anchor_v.T
                logits = logits + torch.log(torch.clamp(overlap, min=self.config.epsilon_a))
            logits = logits.masked_fill(~local_allowed, -1.0e4)
            local = torch.softmax(logits, dim=-1) * local_allowed.to(dtype=self.dtype)
            local = local / torch.clamp(local.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            valid_cols = local_allowed.any(dim=0)
            uniform = valid_cols.to(dtype=self.dtype) / torch.clamp(valid_cols.to(dtype=self.dtype).sum(), min=1.0)
            quality = torch.where(valid_cols, local_scores, torch.zeros_like(local_scores))
            if bool((quality.sum() > self.config.epsilon_a).item()):
                quality = quality / torch.clamp(quality.sum(), min=self.config.epsilon_a)
            else:
                quality = uniform
            target_prob = ((1.0 - mix) * quality) + (mix * uniform)
            target_prob = torch.where(valid_cols, target_prob, torch.zeros_like(target_prob))
            target_prob = target_prob / torch.clamp(target_prob.sum(), min=self.config.epsilon_a)
            target_col = target_prob * float(rows.numel())
            for _ in range(max(int(self.config.mapg_assignment_sinkhorn_iters), 0)):
                col_mass = torch.clamp(local.sum(dim=0), min=self.config.epsilon_a)
                local = local * torch.where(valid_cols, target_col / col_mass, torch.zeros_like(col_mass))[None, :]
                local = local * local_allowed.to(dtype=self.dtype)
                local = local / torch.clamp(local.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            assignment[rows[:, None], candidate_indices[None, :]] = local
        return assignment

    def _observation_owner_active_from_graph(
        self,
        graph: PicfAnchorPriorGraphState | None,
        graph_assignment: torch.Tensor | None,
        role_ids: torch.Tensor,
        *,
        obs_x: torch.Tensor | None = None,
        obs_binding_signature: torch.Tensor | None = None,
        obs_point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if (
            graph is None
            or graph_assignment is None
            or graph.anchor_active is None
            or graph.anchor_active.numel() != graph.anchor_tokens.shape[0]
            or graph_assignment.ndim != 2
            or graph_assignment.shape[1] != graph.anchor_tokens.shape[0]
            or graph_assignment.shape[0] != role_ids.numel()
        ):
            return None
        active = graph.anchor_active.to(device=self.device, dtype=self.dtype).reshape(-1) > 0.5
        active_cols = torch.nonzero(active, as_tuple=False).squeeze(-1)
        if active_cols.numel() == 0:
            return None
        row_count = int(graph_assignment.shape[0])
        owner = torch.zeros((row_count,), device=self.device, dtype=self.dtype)
        used = torch.zeros_like(owner, dtype=torch.bool)
        confidence = torch.clamp(graph.anchor_confidence.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0, max=1.0)
        score = torch.clamp(graph.anchor_scores.to(device=self.device, dtype=self.dtype).reshape(-1), min=self.config.epsilon_a)
        priority = score * torch.clamp(confidence, min=float(self.config.mapg_confidence_floor))
        # `graph_assignment` is row-stochastic. Summing over active columns is
        # therefore a tautology when the assignment has already been restricted
        # to active anchors, and it makes every observation row look like a valid
        # object owner. Owner reliability must instead come from unique owner
        # peaks plus row-local winner margin and duplicate novelty.
        active_assignment = torch.clamp(
            graph_assignment.to(device=self.device, dtype=self.dtype).index_select(1, active_cols),
            min=0.0,
            max=1.0,
        )
        if active_assignment.shape[1] >= 2:
            top2 = torch.topk(active_assignment, k=2, dim=-1).values
            top_mass = top2[:, 0]
            second_mass = top2[:, 1]
        else:
            top_mass = active_assignment.reshape(row_count)
            second_mass = torch.zeros_like(top_mass)
        winner_margin = torch.clamp(
            (top_mass - second_mass) / torch.clamp(top_mass, min=self.config.epsilon_a),
            min=0.0,
            max=1.0,
        )
        soft_owner = torch.clamp(top_mass * winner_margin, min=0.0, max=1.0)
        order = torch.argsort(priority.index_select(0, active_cols), descending=True)
        peak_rows: list[int] = []
        for local_col in order.tolist():
            col = int(active_cols[int(local_col)].item())
            row_score = graph_assignment[:, col].to(device=self.device, dtype=self.dtype).clone()
            row_score = row_score.masked_fill(used, -1.0)
            row = int(torch.argmax(row_score).item())
            if float(row_score[row].item()) <= float(self.config.epsilon_a):
                continue
            owner[row] = 1.0
            used[row] = True
            peak_rows.append(row)
        roles = role_ids.to(device=self.device, dtype=torch.long)
        if peak_rows:
            peak_t = torch.as_tensor(peak_rows, device=self.device, dtype=torch.long)
            peak_roles = roles.index_select(0, peak_t)
            same_role = roles[:, None] == peak_roles[None, :]
            duplicate_terms: list[torch.Tensor] = []
            if obs_x is not None and obs_x.numel() > 0 and obs_x.shape[0] == row_count:
                x = obs_x.to(device=self.device, dtype=self.dtype)
                dist2 = torch.sum((x[:, None, :3] - x.index_select(0, peak_t)[None, :, :3]) ** 2, dim=-1)
                sigma = max(
                    float(getattr(self.config, "aqr_active_slot_geometry_duplicate_sigma_m", 0.04)),
                    self.config.epsilon_a,
                )
                duplicate_terms.append(torch.exp(-dist2 / (2.0 * sigma * sigma)))
            if (
                obs_binding_signature is not None
                and obs_binding_signature.numel() > 0
                and obs_binding_signature.shape[0] == row_count
            ):
                sig = _normalize_tensor(obs_binding_signature.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_residual)
                duplicate_terms.append(torch.clamp(sig @ sig.index_select(0, peak_t).T, min=0.0, max=1.0))
            if obs_point_weights is not None and obs_point_weights.numel() > 0 and obs_point_weights.shape[0] == row_count:
                point = torch.clamp(obs_point_weights.to(device=self.device, dtype=self.dtype), min=0.0)
                point_norm = torch.sqrt(torch.clamp((point * point).sum(dim=-1), min=self.config.epsilon_a))
                peak_point = point.index_select(0, peak_t)
                peak_norm = torch.sqrt(torch.clamp((peak_point * peak_point).sum(dim=-1), min=self.config.epsilon_a))
                duplicate_terms.append(torch.clamp((point @ peak_point.T) / (point_norm[:, None] * peak_norm[None, :]), min=0.0, max=1.0))
            if duplicate_terms:
                duplicate = torch.stack(duplicate_terms, dim=0).amax(dim=0)
                duplicate = torch.where(same_role, duplicate, torch.zeros_like(duplicate))
                duplicate_score = duplicate.amax(dim=-1)
                duplicate_score = duplicate_score.index_fill(0, peak_t, 0.0)
                soft_owner = soft_owner * (1.0 - duplicate_score.clamp(0.0, 1.0))
        owner = torch.maximum(owner, soft_owner)
        if peak_rows:
            owner[torch.as_tensor(peak_rows, device=self.device, dtype=torch.long)] = 1.0
        for role_value in torch.unique(roles).tolist():
            rows = torch.nonzero(roles == int(role_value), as_tuple=False).squeeze(-1)
            if rows.numel() == 0 or bool((owner.index_select(0, rows) > 0.5).any().item()):
                continue
            local_mass = soft_owner.index_select(0, rows)
            row = int(rows[int(torch.argmax(local_mass).item())].item())
            if float(soft_owner[row].item()) > float(self.config.epsilon_a):
                owner[row] = 1.0
                used[row] = True
        if not bool((owner > 0.5).any().item()):
            return None
        return owner

    def _build_anchor_prior_graph(
        self,
        *,
        semantic: _SemanticContext,
        token_field: PicfTokenFieldState,
        dense_memory: _StepDenseMemory,
        previous: PicfPreviousState | None,
        vl_grounding: PicfVLGroundingState | None,
    ) -> PicfAnchorPriorGraphState | None:
        if not bool(self.config.mapg_enabled):
            return None
        visual_count = int(token_field.visual_tokens.shape[0])
        point_count = int(token_field.point_tokens.shape[0])
        tactile_count = int(token_field.tactile_tokens.shape[0])
        post_count = 0 if previous is None else int(previous.posterior.tokens.shape[0])
        anchor_count = max(int(self.config.mapg_anchor_count), 0)
        roles = self._mapg_anchor_roles(anchor_count)
        if anchor_count == 0 or visual_count == 0:
            return PicfAnchorPriorGraphState(
                pg_priors=None,
                visual_priors=torch.zeros((anchor_count, visual_count), device=self.device, dtype=self.dtype),
                point_priors=None,
                tactile_priors=None,
                posterior_priors=None,
                anchor_tokens=torch.zeros((anchor_count, self.config.hidden_dim), device=self.device, dtype=self.dtype),
                anchor_roles=roles,
                anchor_scores=torch.zeros((anchor_count,), device=self.device, dtype=self.dtype),
                anchor_confidence=torch.zeros((anchor_count,), device=self.device, dtype=self.dtype),
                anchor_x=None,
                anchor_S=None,
                geometry_valid=torch.zeros((anchor_count,), device=self.device, dtype=torch.bool),
                obs_slot_assignment=None,
                task_assignment=None,
                modality_confidence=torch.zeros((anchor_count, 5), device=self.device, dtype=self.dtype),
                valid=torch.tensor(False, device=self.device),
            )
        geometry = token_field.projective_geometry
        visual_grid = (
            geometry.visual_grid_index.to(device=self.device, dtype=self.dtype)
            if geometry is not None and geometry.visual_grid_index.shape[0] == visual_count
            else torch.arange(visual_count, device=self.device, dtype=self.dtype)[:, None].expand(-1, 2)
        )
        q_v = self._mapg_visual_seed_priors(
            vl_grounding,
            roles=roles,
            visual_count=visual_count,
            visual_grid_index=visual_grid,
        )
        pg_tokens = None
        pg_priors = None
        if semantic.image_tokens is not None and len(semantic.image_token_ranges) > 0 and len(semantic.image_grid_shapes) > 0:
            start, end = semantic.image_token_ranges[0]
            pg_tokens = semantic.image_tokens[start:end].to(device=self.device, dtype=self.dtype)
            pg_priors = self._mapg_pg_seed_priors(
                vl_grounding,
                roles=roles,
                pg_count=int(pg_tokens.shape[0]),
                pg_hw=semantic.image_grid_shapes[0],
            )
        q_p = self._mapg_visual_to_point(q_v, token_field) if point_count > 0 else None
        q_t = self._mapg_tactile_seed_priors(token_field, roles) if tactile_count > 0 else None
        q_post = self._mapg_posterior_seed_priors(previous, roles) if post_count > 0 else None
        t_to_p = self._mapg_tactile_to_point(q_t, token_field) if q_t is not None else None
        post_to_p = self._mapg_posterior_to_point(q_post, previous, token_field) if q_post is not None else None
        t_to_v_direct = self._mapg_tactile_to_visual(q_t, token_field) if q_t is not None else None
        post_to_v_direct = self._mapg_posterior_to_visual(q_post, previous, token_field) if q_post is not None else None
        if q_p is None and (t_to_p is not None or post_to_p is not None):
            q_p = torch.zeros((anchor_count, point_count), device=self.device, dtype=self.dtype)
        if q_p is not None:
            if t_to_p is not None:
                q_p = q_p + t_to_p
            if post_to_p is not None:
                q_p = q_p + post_to_p
            q_p = _normalize_rows(q_p, eps=self.config.epsilon_a)

        p_v = q_v
        p_p = q_p
        p_t = q_t
        p_post = q_post
        floor = max(float(self.config.mapg_confidence_floor), 0.0)
        q_v_conf = _distribution_confidence(q_v, eps=self.config.epsilon_a, floor=floor)
        q_p_conf = _distribution_confidence(q_p, eps=self.config.epsilon_a, floor=floor)
        q_t_conf = _distribution_confidence(q_t, eps=self.config.epsilon_a, floor=floor)
        q_post_conf = _distribution_confidence(q_post, eps=self.config.epsilon_a, floor=floor)

        def _weighted(prior: torch.Tensor | None, confidence: torch.Tensor | None) -> torch.Tensor | None:
            if prior is None:
                return None
            if confidence is None or confidence.shape[0] != prior.shape[0]:
                confidence = torch.ones((prior.shape[0],), device=self.device, dtype=self.dtype)
            return prior.to(device=self.device, dtype=self.dtype) * confidence.to(device=self.device, dtype=self.dtype)[:, None]

        rounds = max(int(self.config.mapg_message_rounds), 1)
        for _ in range(rounds):
            next_v = _weighted(q_v, q_v_conf)
            if next_v is None:
                next_v = torch.zeros_like(q_v)
            p_to_v = self._mapg_point_to_visual(p_p, token_field)
            if p_to_v is not None:
                weighted = _weighted(p_to_v, q_p_conf)
                next_v = next_v + (p_to_v if weighted is None else weighted)
            if p_t is not None:
                t_visual = t_to_v_direct
                if t_visual is None:
                    t_point = self._mapg_tactile_to_point(p_t, token_field)
                    t_visual = self._mapg_point_to_visual(t_point, token_field)
                if t_visual is not None:
                    weighted = _weighted(t_visual, q_t_conf)
                    next_v = next_v + (t_visual if weighted is None else weighted)
            if p_post is not None:
                post_visual = post_to_v_direct
                if post_visual is None:
                    post_point = self._mapg_posterior_to_point(p_post, previous, token_field)
                    post_visual = self._mapg_point_to_visual(post_point, token_field)
                if post_visual is not None:
                    weighted = _weighted(post_visual, q_post_conf)
                    next_v = next_v + (post_visual if weighted is None else weighted)
            p_v = _normalize_rows(next_v, eps=self.config.epsilon_a)
            next_p = _weighted(q_p, q_p_conf) if q_p is not None else self._mapg_visual_to_point(p_v, token_field)
            v_to_p = self._mapg_visual_to_point(p_v, token_field)
            if next_p is not None and v_to_p is not None:
                weighted = _weighted(v_to_p, q_v_conf)
                next_p = next_p + (v_to_p if weighted is None else weighted)
            if next_p is not None and p_t is not None:
                t_point = self._mapg_tactile_to_point(p_t, token_field)
                if t_point is not None:
                    weighted = _weighted(t_point, q_t_conf)
                    next_p = next_p + (t_point if weighted is None else weighted)
            if next_p is not None and p_post is not None:
                post_point = self._mapg_posterior_to_point(p_post, previous, token_field)
                if post_point is not None:
                    weighted = _weighted(post_point, q_post_conf)
                    next_p = next_p + (post_point if weighted is None else weighted)
            p_p = _normalize_rows(next_p, eps=self.config.epsilon_a) if next_p is not None else None
            p_t = _normalize_rows(p_t, eps=self.config.epsilon_a) if p_t is not None else None
            p_post = _normalize_rows(p_post, eps=self.config.epsilon_a) if p_post is not None else None

        pg_h = torch.zeros((anchor_count, self.config.hidden_dim), device=self.device, dtype=self.dtype)
        if pg_priors is not None and pg_tokens is not None and self.mapg_pg_proj is not None:
            pg_h = self.mapg_pg_proj(pg_priors @ pg_tokens)
        visual_h = self.mapg_visual_proj(p_v @ token_field.visual_tokens) if self.mapg_visual_proj is not None else torch.zeros_like(pg_h)
        point_h = torch.zeros_like(visual_h)
        point_valid = torch.zeros((anchor_count,), device=self.device, dtype=torch.bool)
        if p_p is not None and point_count > 0 and self.mapg_point_proj is not None:
            point_h = self.mapg_point_proj(p_p @ token_field.point_tokens)
            point_valid = _row_has_mass(p_p, eps=self.config.epsilon_a)
        tactile_h = torch.zeros_like(visual_h)
        tactile_valid = torch.zeros((anchor_count,), device=self.device, dtype=torch.bool)
        if p_t is not None and tactile_count > 0 and self.mapg_tactile_proj is not None:
            tactile_h = self.mapg_tactile_proj(p_t @ token_field.tactile_tokens)
            tactile_valid = _row_has_mass(p_t, eps=self.config.epsilon_a)
        post_h = torch.zeros_like(visual_h)
        post_valid = torch.zeros((anchor_count,), device=self.device, dtype=torch.bool)
        if p_post is not None and previous is not None and self.mapg_posterior_proj is not None:
            post_h = self.mapg_posterior_proj(p_post @ previous.posterior.tokens.to(device=self.device, dtype=self.dtype))
            post_valid = _row_has_mass(p_post, eps=self.config.epsilon_a)
        pg_conf = _distribution_confidence(pg_priors, eps=self.config.epsilon_a, floor=floor)
        visual_conf = _distribution_confidence(p_v, eps=self.config.epsilon_a, floor=floor)
        point_conf = _distribution_confidence(p_p, eps=self.config.epsilon_a, floor=floor)
        tactile_conf = _distribution_confidence(p_t, eps=self.config.epsilon_a, floor=floor)
        post_conf = _distribution_confidence(p_post, eps=self.config.epsilon_a, floor=floor)
        zero_conf = torch.zeros((anchor_count,), device=self.device, dtype=self.dtype)
        modality_conf = torch.stack(
            [
                zero_conf if pg_conf is None else pg_conf.to(device=self.device, dtype=self.dtype),
                zero_conf if visual_conf is None else visual_conf.to(device=self.device, dtype=self.dtype),
                zero_conf if point_conf is None else point_conf.to(device=self.device, dtype=self.dtype),
                zero_conf if tactile_conf is None else tactile_conf.to(device=self.device, dtype=self.dtype),
                zero_conf if post_conf is None else post_conf.to(device=self.device, dtype=self.dtype),
            ],
            dim=-1,
        )
        modality_h = torch.stack([pg_h, visual_h, point_h, tactile_h, post_h], dim=1)
        pooled = torch.sum(modality_h * modality_conf[:, :, None], dim=1) / torch.clamp(
            modality_conf.sum(dim=-1, keepdim=True),
            min=1.0,
        )
        fusion_in = torch.cat([pg_h, visual_h, point_h, tactile_h, post_h, modality_conf], dim=-1)
        delta = self.mapg_anchor_fusion(fusion_in) if self.mapg_anchor_fusion is not None else torch.zeros_like(pooled)
        role_emb = self.mapg_role_embedding(roles) if self.mapg_role_embedding is not None else torch.zeros_like(pooled)
        anchor_tokens = pooled + delta + role_emb
        anchor_scores = torch.max(p_v, dim=-1).values
        if p_p is not None:
            anchor_scores = anchor_scores + torch.max(p_p, dim=-1).values
        anchor_conf = torch.clamp(modality_conf.max(dim=-1).values, min=0.0, max=1.0)
        anchor_x = None
        anchor_S = None
        geometry_valid = torch.zeros((anchor_count,), device=self.device, dtype=torch.bool)
        if p_p is not None and point_count > 0:
            point_positions = self._world_point_positions(token_field)
            anchor_x = p_p @ point_positions
            anchor_S = _weighted_cov(point_positions, p_p, anchor_x, self.config)
            geometry_valid = _row_has_mass(p_p, eps=self.config.epsilon_a)
        elif p_post is not None and previous is not None:
            anchor_x = p_post @ previous.posterior.x.to(device=self.device, dtype=self.dtype)
            anchor_S = torch.eye(3, device=self.device, dtype=self.dtype)[None, :, :].expand(anchor_count, -1, -1).clone()
            geometry_valid = _row_has_mass(p_post, eps=self.config.epsilon_a)
        return PicfAnchorPriorGraphState(
            pg_priors=pg_priors,
            visual_priors=p_v,
            point_priors=p_p,
            tactile_priors=p_t,
            posterior_priors=p_post,
            anchor_tokens=anchor_tokens,
            anchor_roles=roles,
            anchor_scores=anchor_scores,
            anchor_confidence=anchor_conf,
            anchor_x=anchor_x,
            anchor_S=anchor_S,
            geometry_valid=geometry_valid,
            obs_slot_assignment=None,
            task_assignment=None,
            modality_confidence=modality_conf,
            valid=torch.tensor(True, device=self.device),
        )

    def _previous_action(self, previous: PicfPreviousState | None) -> torch.Tensor:
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

    def _point_pool_ids(self, token_field: PicfTokenFieldState) -> torch.Tensor:
        point_count = int(token_field.point_tokens.shape[0])
        ids = token_field.point_pool_ids
        if ids is None or ids.numel() != point_count:
            return torch.zeros((point_count,), device=self.device, dtype=torch.long)
        return ids.to(device=self.device, dtype=torch.long)

    def _world_point_positions(self, token_field: PicfTokenFieldState) -> torch.Tensor:
        point_count = int(token_field.point_tokens.shape[0])
        world = token_field.point_positions_world
        if world is not None and world.shape == (point_count, 3):
            return world.to(device=self.device, dtype=self.dtype)
        return token_field.point_positions.to(device=self.device, dtype=self.dtype)

    def _scene_point_candidate_mask(
        self,
        token_field: PicfTokenFieldState,
        *,
        fallback_to_global: bool = True,
    ) -> torch.Tensor:
        point_count = int(token_field.point_tokens.shape[0])
        if point_count == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.bool)
        pool_ids = self._point_pool_ids(token_field)
        global_mask = pool_ids == 1
        if not bool(global_mask.any().item()):
            return global_mask
        mask = global_mask.clone()
        geometry = token_field.projective_geometry
        if geometry is not None and geometry.point_visibility.shape == (point_count,):
            mask = mask & (geometry.point_visibility.to(device=self.device, dtype=self.dtype) > 0.0)
            if geometry.point_depth_valid.shape == (point_count,):
                mask = mask & geometry.point_depth_valid.to(device=self.device, dtype=torch.bool)
            if geometry.point_proj_grid_index.shape == (point_count, 2) and geometry.visual_grid_index.numel() > 0:
                grid = geometry.point_proj_grid_index.to(device=self.device, dtype=self.dtype)
                visual_grid = geometry.visual_grid_index.to(device=self.device, dtype=self.dtype)
                max_x = torch.max(visual_grid[:, 0])
                max_y = torch.max(visual_grid[:, 1])
                border = max(float(self.config.scene_anchor_border_patches), 0.0)
                if border > 0.0:
                    interior = (
                        (grid[:, 0] >= border)
                        & (grid[:, 0] <= (max_x - border))
                        & (grid[:, 1] >= border)
                        & (grid[:, 1] <= (max_y - border))
                    )
                    mask = mask & interior
        if bool(mask.any().item()):
            return mask
        return global_mask if bool(fallback_to_global) else mask

    def _effector_observation_count(self) -> int:
        return min(max(int(self.config.effector_observation_anchors), 0), int(self.config.observation_anchors))

    def _effector_persistent_count(self) -> int:
        return min(max(int(self.config.effector_persistent_anchors), 0), int(self.config.persistent_anchors))

    def _task_effector_count(self) -> int:
        return min(max(int(self.config.task_effector_queries), 0), int(self.config.task_local_queries))

    def _fused_read_role_bias(self, query_role_ids: torch.Tensor, token_field: PicfTokenFieldState) -> torch.Tensor | None:
        fused_count = int(token_field.fused_tokens.shape[0])
        if fused_count == 0 or query_role_ids.numel() == 0:
            return None
        point_count = int(token_field.point_tokens.shape[0])
        tactile_count = 0 if token_field.tactile_tokens_active is None else int(token_field.tactile_tokens_active.shape[0])
        point_pool_ids = self._point_pool_ids(token_field)
        has_global = bool(point_count > 0 and torch.any(point_pool_ids == 1).item())
        bias = torch.zeros((int(query_role_ids.numel()), fused_count), device=self.device, dtype=self.dtype)
        blocked = torch.zeros_like(bias, dtype=torch.bool)
        if point_count > 0:
            local_point = point_pool_ids == 0
            global_point = self._scene_point_candidate_mask(token_field)
            if not bool(global_point.any().item()):
                global_point = point_pool_ids == 1
            for row, role in enumerate(query_role_ids.tolist()):
                if int(role) == 0:
                    blocked[row, :point_count] = ~local_point
                elif int(role) == 1 and has_global:
                    blocked[row, :point_count] = ~global_point
        if tactile_count > 0:
            tactile_start = point_count
            tactile_end = tactile_start + tactile_count
            attach_to_object = bool(getattr(self.config, "tactile_attach_to_object_owner", True))
            for row, role in enumerate(query_role_ids.tolist()):
                role_int = int(role)
                if attach_to_object:
                    if role_int != 1:
                        blocked[row, tactile_start:tactile_end] = True
                elif role_int == 1:
                    blocked[row, tactile_start:tactile_end] = True
        if bool(blocked.any().item()):
            bias = bias.masked_fill(blocked, -1.0e4)
        return bias

    def _task_public_role_bias(self, query_role_ids: torch.Tensor, token_field: PicfTokenFieldState) -> torch.Tensor | None:
        fused_bias = self._fused_read_role_bias(query_role_ids, token_field)
        if fused_bias is None:
            visual_public_count = token_field.visual_tokens.shape[0]
            if token_field.temporal_visual is not None:
                visual_public_count += token_field.temporal_visual.tokens.shape[0]
            if visual_public_count == 0:
                return None
            return torch.zeros((int(query_role_ids.numel()), int(visual_public_count)), device=self.device, dtype=self.dtype)
        visual_public_count = token_field.visual_tokens.shape[0]
        if token_field.temporal_visual is not None:
            visual_public_count += token_field.temporal_visual.tokens.shape[0]
        if visual_public_count == 0:
            return fused_bias
        visual_bias = torch.zeros((fused_bias.shape[0], int(visual_public_count)), device=self.device, dtype=self.dtype)
        return torch.cat([fused_bias, visual_bias], dim=1)

    def _build_public_read_memory(self, token_field: PicfTokenFieldState) -> torch.Tensor:
        pieces = []
        if token_field.fused_tokens.shape[0] > 0:
            pieces.append(token_field.fused_tokens)
        if token_field.visual_tokens.shape[0] > 0:
            pieces.append(token_field.visual_tokens)
        if token_field.temporal_visual is not None and token_field.temporal_visual.tokens.shape[0] > 0:
            pieces.append(token_field.temporal_visual.tokens)
        if pieces:
            return torch.cat(pieces, dim=0)
        return torch.zeros((0, self.config.hidden_dim), device=self.device, dtype=self.dtype)

    def _ordinal_relation_state(
        self,
        prompt: str | None,
        *,
        x: torch.Tensor,
        geometry_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(x.shape[0])
        active = torch.tensor(False, device=self.device)
        scores = torch.zeros((count,), device=self.device, dtype=self.dtype)
        ranks = torch.zeros((count,), device=self.device, dtype=self.dtype)
        axis = torch.zeros((3,), device=self.device, dtype=self.dtype)
        target_rank = torch.zeros((), device=self.device, dtype=self.dtype)
        selected_slot = torch.full((), -1, device=self.device, dtype=torch.long)
        confidence = torch.zeros((), device=self.device, dtype=self.dtype)
        if not bool(self.config.ordinal_relation_enabled) or prompt is None or count == 0:
            return active, scores, ranks, axis, target_rank, selected_slot, confidence
        text = str(prompt).lower()
        relation_words = (
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "left",
            "right",
            "front",
            "back",
            "nearest",
            "farthest",
            "第",
            "左",
            "右",
            "前",
            "后",
            "最近",
            "最远",
        )
        if not any(word in text for word in relation_words):
            return active, scores, ranks, axis, target_rank, selected_slot, confidence
        active = torch.tensor(True, device=self.device)
        if "right" in text or "右" in text:
            axis = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        elif "front" in text or "前" in text:
            axis = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
        elif "back" in text or "后" in text:
            axis = torch.tensor([0.0, -1.0, 0.0], device=self.device, dtype=self.dtype)
        else:
            axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        scores = x.to(device=self.device, dtype=self.dtype) @ axis
        scores = torch.where(geometry_valid.to(device=self.device, dtype=torch.bool), scores, torch.zeros_like(scores))
        tau = max(float(self.config.bind_sigma_m), self.config.epsilon_a)
        ranks = 1.0 + torch.sum(torch.sigmoid((scores[None, :] - scores[:, None]) / tau), dim=-1) - 0.5
        ranks = torch.where(geometry_valid.to(device=self.device, dtype=torch.bool), ranks, torch.zeros_like(ranks))
        if bool(self.config.ordinal_weak_target_enabled):
            rank_map = {
                "first": 1,
                "second": 2,
                "third": 3,
                "fourth": 4,
                "fifth": 5,
                "第一": 1,
                "第二": 2,
                "第三": 3,
                "第四": 4,
                "第五": 5,
            }
            for key, value in rank_map.items():
                if key in text:
                    target_rank = torch.as_tensor(float(value), device=self.device, dtype=self.dtype)
                    break
            if bool(geometry_valid.any().item()) and target_rank.item() > 0:
                selected_slot = torch.argmin(torch.abs(ranks - target_rank)).to(device=self.device, dtype=torch.long)
                sorted_scores = torch.sort(scores[geometry_valid.to(device=self.device, dtype=torch.bool)]).values
                if sorted_scores.numel() > 1:
                    separation = torch.mean(torch.abs(sorted_scores[1:] - sorted_scores[:-1]))
                    confidence = torch.sigmoid(separation / max(float(self.config.bind_sigma_m), self.config.epsilon_a))
                else:
                    confidence = torch.ones((), device=self.device, dtype=self.dtype)
        return active, scores, ranks, axis, target_rank, selected_slot, confidence

    def _build_task_readout(
        self,
        token_field: PicfTokenFieldState,
        dense_memory: _StepDenseMemory,
        semantic: _SemanticContext,
        proprio_token: torch.Tensor,
        prompt: str | None,
        vl_grounding: PicfVLGroundingState | None = None,
        anchor_graph: PicfAnchorPriorGraphState | None = None,
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
            local_count = int(self.config.task_local_queries)
            effector_count = self._task_effector_count()
            query_role_ids = torch.cat(
                [
                    torch.zeros((effector_count,), device=self.device, dtype=torch.long),
                    torch.ones((max(local_count - effector_count, 0),), device=self.device, dtype=torch.long),
                    torch.full(
                        (int(self.config.task_global_queries) + int(self.config.task_instruction_queries),),
                        2,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            queries, public_attention = self.task_public_reader(
                queries,
                public_read_memory[None, :],
                attn_bias=self._task_public_role_bias(query_role_ids, token_field),
            )
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
        task_effector_count = self._task_effector_count()
        local_role_ids = torch.cat(
            [
                torch.zeros((task_effector_count,), device=self.device, dtype=torch.long),
                torch.ones((max(local_count - task_effector_count, 0),), device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        graph_assignment = self._mapg_slot_assignment(
            anchor_graph,
            local_role_ids,
            slot_tokens=local_tokens,
            slot_point_priors=point_public_attention[:local_count] if point_public_attention.numel() > 0 else None,
            slot_visual_priors=visual_public_attention[:local_count] if visual_public_attention.numel() > 0 else None,
        )
        if anchor_graph is not None:
            anchor_graph.task_assignment = graph_assignment
        graph_visual_weights = None
        graph_tactile_weights = None
        mapg_task_gate = self._mapg_gate(self.mapg_task_gate_logit, anchor_graph)
        if graph_assignment is not None and bool((mapg_task_gate > 0.0).item()):
            graph_tokens = graph_assignment @ anchor_graph.anchor_tokens.to(device=self.device, dtype=self.dtype)
            local_tokens = local_tokens + (mapg_task_gate * graph_tokens)
            graph_visual_weights = graph_assignment @ anchor_graph.visual_priors.to(device=self.device, dtype=self.dtype)
            if anchor_graph.tactile_priors is not None:
                graph_tactile_weights = graph_assignment @ anchor_graph.tactile_priors.to(device=self.device, dtype=self.dtype)

        visual_weights = None
        tactile_weights = graph_tactile_weights
        if token_field.visual_tokens.shape[0] > 0 and local_count > 0:
            direct_visual = visual_public_attention[:local_count]
            direct_visual = direct_visual / torch.clamp(direct_visual.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            visual_weights = direct_visual
            if graph_visual_weights is not None and bool((mapg_task_gate > 0.0).item()):
                graph_visual = _normalize_rows(graph_visual_weights, eps=self.config.epsilon_a)
                graph_valid = _row_has_mass(graph_visual_weights, eps=self.config.epsilon_a)
                graph_mix = torch.where(graph_valid[:, None], graph_visual, direct_visual)
                visual_weights = ((1.0 - mapg_task_gate) * direct_visual) + (mapg_task_gate * graph_mix)
                visual_weights = _normalize_rows(visual_weights, eps=self.config.epsilon_a)
                if self.mapg_task_visual_proj is not None:
                    local_tokens = local_tokens + (mapg_task_gate * self.mapg_task_visual_proj(visual_weights @ token_field.visual_tokens))

        if token_field.point_tokens.shape[0] > 0 and local_count > 0:
            geometry_positions = self._world_point_positions(token_field)
            direct_weights = point_public_attention[:local_count]
            denom = torch.clamp(direct_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            direct_weights = direct_weights / denom
            point_weights = direct_weights
            if graph_assignment is not None and anchor_graph.point_priors is not None and bool((mapg_task_gate > 0.0).item()):
                graph_point = graph_assignment @ anchor_graph.point_priors.to(device=self.device, dtype=self.dtype)
                graph_valid = _row_has_mass(graph_point, eps=self.config.epsilon_a)
                graph_mix = torch.where(graph_valid[:, None], _normalize_rows(graph_point, eps=self.config.epsilon_a), point_weights)
                point_weights = ((1.0 - mapg_task_gate) * point_weights) + (mapg_task_gate * graph_mix)
                point_weights = torch.clamp(point_weights, min=0.0)
                point_weights = point_weights / torch.clamp(point_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            vl_weights, vl_valid = self._vl_slot_point_priors(
                vl_grounding,
                local_role_ids,
                point_count=int(token_field.point_tokens.shape[0]),
            )
            gate = self._vl_gate(self.vl_task_point_gate_logit, vl_grounding)
            if bool(vl_valid.any().item()) and bool((gate > 0.0).item()):
                vl_mix = torch.where(vl_valid[:, None], vl_weights, point_weights)
                point_weights = ((1.0 - gate) * point_weights) + (gate * vl_mix)
                point_weights = torch.clamp(point_weights, min=0.0)
                point_weights = point_weights / torch.clamp(point_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            x = point_weights @ geometry_positions
            S = _weighted_cov(geometry_positions, point_weights, x, self.config)
            a = _extent_from_cov(S, self.config)
            geometry_valid = _row_has_mass(point_weights, eps=self.config.epsilon_a)
            local_tokens = local_tokens + self.task_geom_proj(_geometry_pe(x, a, S, self.config))
        else:
            point_weights = torch.zeros((local_count, 0), device=self.device, dtype=self.dtype)
            x = torch.zeros((local_count, 3), device=self.device, dtype=self.dtype)
            S = _diag_embed(torch.full((local_count, 3), self.config.epsilon_s, device=self.device, dtype=self.dtype))
            a = _to_tensor(self.config.a_min_m, device=self.device, dtype=self.dtype)[None, :].expand(local_count, -1)
            geometry_valid = torch.zeros((local_count,), device=self.device, dtype=torch.bool)
        (
            ordinal_active,
            ordinal_scores,
            ordinal_ranks,
            ordinal_axis,
            ordinal_target_rank,
            ordinal_selected_slot,
            ordinal_confidence,
        ) = self._ordinal_relation_state(
            prompt,
            x=x,
            geometry_valid=geometry_valid,
        )

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
            local_role_ids=local_role_ids,
            graph_assignment=graph_assignment,
            visual_weights=visual_weights,
            tactile_weights=tactile_weights,
            geometry_valid=geometry_valid,
            graph_visual_weights=graph_visual_weights,
            graph_tactile_weights=graph_tactile_weights,
            ordinal_active=ordinal_active,
            ordinal_scores=ordinal_scores,
            ordinal_ranks=ordinal_ranks,
            ordinal_axis=ordinal_axis,
            ordinal_target_rank=ordinal_target_rank,
            ordinal_selected_slot=ordinal_selected_slot,
            ordinal_confidence=ordinal_confidence,
        )

    def _build_conditioned_control_state(
        self,
        posterior: PicfPosteriorAnchorState,
        innovation_token: torch.Tensor,
        proprio_token: torch.Tensor,
        task_readout: PicfTaskReadoutState,
        anchor_graph: PicfAnchorPriorGraphState | None = None,
    ) -> PicfConditionedControlState:
        posterior_gate = self._posterior_file_active_gate(
            posterior,
            count=int(posterior.tokens.shape[0]),
            dtype=self.dtype,
        )[:, None]
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
                posterior_gate * _add_role_embedding(control_posterior_tokens, self.control_role_embedding, 0),
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
        graph_tokens = None
        mapg_control_gate = self._mapg_gate(self.mapg_control_gate_logit, anchor_graph)
        if (
            anchor_graph is not None
            and bool((mapg_control_gate > 0.0).item())
            and anchor_graph.anchor_tokens.shape[0] > 0
            and self.mapg_to_control_proj is not None
            and self.mapg_control_role_embedding is not None
        ):
            graph_tokens = self.mapg_to_control_proj(anchor_graph.anchor_tokens.to(device=self.device, dtype=self.dtype))
            graph_tokens = graph_tokens + self.mapg_control_role_embedding.weight[0][None, :]
            graph_weight = None
            if (
                anchor_graph.anchor_downstream_weight is not None
                and anchor_graph.anchor_downstream_weight.numel() == graph_tokens.shape[0]
            ):
                graph_weight = torch.clamp(
                    anchor_graph.anchor_downstream_weight.to(device=self.device, dtype=self.dtype).reshape(-1, 1),
                    min=0.0,
                    max=1.0,
                )
            elif anchor_graph.anchor_active is not None and anchor_graph.anchor_active.numel() == graph_tokens.shape[0]:
                graph_weight = torch.clamp(
                    anchor_graph.anchor_active.to(device=self.device, dtype=self.dtype).reshape(-1, 1),
                    min=0.0,
                    max=1.0,
                )
            if graph_weight is not None:
                # Active object anchors are full action evidence, context
                # anchors are low-weight scene evidence, and reserve/dustbin
                # anchors remain diagnostics/capacity rather than object
                # prefix tokens. This gates the graph prefix only; typed
                # visual/semantic/temporal memories remain readable upstream.
                graph_tokens = graph_tokens * graph_weight
            graph_tokens = mapg_control_gate * graph_tokens
        conditioned_control_queries = self.control_query_tokens.to(device=self.device, dtype=self.dtype)
        prefix_parts = [base_tokens, task_tokens]
        if graph_tokens is not None:
            prefix_parts.append(graph_tokens)
        prefix_parts.append(_add_role_embedding(conditioned_control_queries, self.control_role_embedding, 7))
        control_prefix = torch.cat(prefix_parts, dim=0)
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
            graph_tokens=graph_tokens,
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
        posterior_gate = self._posterior_file_active_gate(
            posterior,
            count=int(posterior.tokens.shape[0]),
            dtype=self.dtype,
        )[:, None]
        pred_world_tokens = torch.cat(
            [
                posterior_gate * _add_role_embedding(posterior.tokens, self.predictive_physical_role_embedding, 0),
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

    def _empty_evidence_cache(self, *, slot_count: int | None = None) -> PicfEvidenceCacheState:
        h = max(int(self.config.evidence_cache_len), 1)
        k = int(slot_count if slot_count is not None else self.config.persistent_anchors)
        hidden = int(self.config.hidden_dim)
        return PicfEvidenceCacheState(
            tokens=torch.zeros((h, k, hidden), device=self.device, dtype=self.dtype),
            slot_address=torch.zeros((h, k, hidden), device=self.device, dtype=self.dtype),
            role_ids=torch.zeros((h, k), device=self.device, dtype=torch.long),
            source_ids=torch.zeros((h, k), device=self.device, dtype=torch.long),
            age=torch.zeros((h, k), device=self.device, dtype=self.dtype),
            uncertainty=torch.ones((h, k), device=self.device, dtype=self.dtype),
            innovation_at_write=torch.zeros((h, k), device=self.device, dtype=self.dtype),
            modality_validity=torch.zeros((h, k, 4), device=self.device, dtype=self.dtype),
            valid=torch.zeros((h, k), device=self.device, dtype=torch.bool),
        )

    def _write_evidence_cache(
        self,
        previous: PicfPreviousState | None,
        posterior: PicfPosteriorAnchorState,
        *,
        innovation_norm: torch.Tensor,
        availability: torch.Tensor,
        reset: bool,
    ) -> PicfEvidenceCacheState | None:
        if not bool(self.config.evidence_cache_enabled):
            return None
        base = None if previous is None or reset else getattr(previous.predictive, "evidence_cache", None)
        if base is None or base.tokens.shape[1] != posterior.tokens.shape[0] or base.tokens.shape[2] != posterior.tokens.shape[1]:
            cache = self._empty_evidence_cache(slot_count=int(posterior.tokens.shape[0]))
        else:
            cache = PicfEvidenceCacheState(
                tokens=base.tokens.to(device=self.device, dtype=self.dtype).clone(),
                slot_address=base.slot_address.to(device=self.device, dtype=self.dtype).clone(),
                role_ids=base.role_ids.to(device=self.device, dtype=torch.long).clone(),
                source_ids=base.source_ids.to(device=self.device, dtype=torch.long).clone(),
                age=base.age.to(device=self.device, dtype=self.dtype).clone(),
                uncertainty=base.uncertainty.to(device=self.device, dtype=self.dtype).clone(),
                innovation_at_write=base.innovation_at_write.to(device=self.device, dtype=self.dtype).clone(),
                modality_validity=base.modality_validity.to(device=self.device, dtype=self.dtype).clone(),
                valid=base.valid.to(device=self.device, dtype=torch.bool).clone(),
            )
            cache.tokens = torch.roll(cache.tokens, shifts=1, dims=0)
            cache.slot_address = torch.roll(cache.slot_address, shifts=1, dims=0)
            cache.role_ids = torch.roll(cache.role_ids, shifts=1, dims=0)
            cache.source_ids = torch.roll(cache.source_ids, shifts=1, dims=0)
            cache.age = torch.roll(cache.age + 1.0, shifts=1, dims=0)
            cache.uncertainty = torch.roll(cache.uncertainty, shifts=1, dims=0)
            cache.innovation_at_write = torch.roll(cache.innovation_at_write, shifts=1, dims=0)
            cache.modality_validity = torch.roll(cache.modality_validity, shifts=1, dims=0)
            cache.valid = torch.roll(cache.valid, shifts=1, dims=0)
        slot_address = posterior.slot_address if posterior.slot_address is not None else self.posterior_slot_token.to(device=self.device, dtype=self.dtype)
        uncertainty = torch.diagonal(posterior.Sigma, dim1=-2, dim2=-1).mean(dim=-1)
        role_ids = posterior.role_ids if posterior.role_ids is not None else self._posterior_role_ids()
        innovation_scalar = torch.linalg.norm(innovation_norm.reshape(-1)).to(device=self.device, dtype=self.dtype)
        cache.tokens[0] = posterior.tokens.detach()
        cache.slot_address[0] = slot_address.detach()
        cache.role_ids[0] = role_ids.to(device=self.device, dtype=torch.long)
        cache.source_ids[0] = torch.ones_like(cache.source_ids[0])
        cache.age[0] = torch.zeros_like(cache.age[0])
        cache.uncertainty[0] = uncertainty.detach()
        cache.innovation_at_write[0] = innovation_scalar.expand_as(cache.innovation_at_write[0])
        cache.modality_validity[0] = availability.to(device=self.device, dtype=self.dtype)[None, :].expand(cache.modality_validity.shape[1], -1)
        cache.valid[0] = self._posterior_file_active_gate(
            posterior,
            count=int(cache.valid.shape[1]),
            dtype=self.dtype,
        ).to(dtype=torch.bool)
        return cache

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

    def _encode_context_tokens(self, observation: PicfObservation, meta: RuntimeMeta, previous: PicfPreviousState | None) -> torch.Tensor:
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
        temporal_visual_maps: torch.Tensor | None,
        tactile_bundle: AnyTouchFeatureBundle | None,
        meta: RuntimeMeta,
        previous: PicfPreviousState | None,
    ) -> tuple[PicfTokenFieldState, _StepDenseMemory]:
        hidden_dim = self.config.hidden_dim
        point_tokens = torch.zeros((0, hidden_dim), device=self.device, dtype=self.dtype)
        point_positions = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        point_pool_ids = torch.zeros((0,), device=self.device, dtype=torch.long)
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
        tactile_evidence_mask = torch.zeros((0,), device=self.device, dtype=torch.bool)
        tactile_evidence_weight = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_contact_score = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_contact_score_ema = torch.zeros((0,), device=self.device, dtype=self.dtype)
        tactile_group_ids = torch.zeros((0,), device=self.device, dtype=torch.long)
        visual_payload = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        point_payload = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        tactile_group_tokens: tuple[torch.Tensor, ...] = ()
        visual_hw: tuple[int, int] | None = None
        if visual_map is not None and visual_map.numel() > 0:
            visual_hw = (int(visual_map.shape[0]), int(visual_map.shape[1]))
        point_positions_world = (
            _to_tensor(_frame_context_points_world(frame_context), device=self.device, dtype=self.dtype)
            if frame_context is not None
            else point_positions
        )
        projective_geometry = self._build_projective_geometry(
            observation=observation,
            point_positions=point_positions_world,
            visual_hw=visual_hw,
        )
        if frame_context is not None:
            point_positions = _to_tensor(frame_context.points_local, device=self.device, dtype=self.dtype)
            if frame_context.pool_ids is None:
                point_pool_ids = torch.zeros((int(point_positions.shape[0]),), device=self.device, dtype=torch.long)
            else:
                point_pool_ids = torch.as_tensor(frame_context.pool_ids, device=self.device, dtype=torch.long)
                if point_pool_ids.numel() != point_positions.shape[0]:
                    raise RuntimeError(
                        "PICF point-pool contract violated: "
                        f"pool_ids={int(point_pool_ids.numel())} point_positions={int(point_positions.shape[0])}"
                    )
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
        temporal_visual = None
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
            if (
                temporal_visual_maps is not None
                and temporal_visual_maps.numel() > 0
                and str(self.config.aqr_vjepa_temporal_mode) != "disabled"
            ):
                recent = temporal_visual_maps.to(device=self.device, dtype=self.dtype)
                if recent.ndim == 3:
                    recent = recent[None, :]
                if recent.ndim == 4:
                    recent = recent[None, :]
                if recent.ndim == 5 and int(recent.shape[2]) == h and int(recent.shape[3]) == w:
                    maps = [recent]
                    temporal_mode = str(self.config.aqr_vjepa_temporal_mode)
                    include_delta = bool(self.config.aqr_vjepa_temporal_include_delta) or temporal_mode == "last_mean_delta"
                    if include_delta and recent.shape[1] >= 2:
                        maps.append((recent[:, -1:] - recent[:, -2:-1]))
                    temporal_stack = torch.cat(maps, dim=1)
                    view_count = int(temporal_stack.shape[0])
                    t_count = int(temporal_stack.shape[1])
                    temporal_flat = temporal_stack.reshape(-1, temporal_stack.shape[-1])
                    temporal_grid = grid[None, None, :, :].expand(view_count, t_count, -1, -1).reshape(-1, 2)
                    temporal_ray = ray_features[None, None, :, :].expand(view_count, t_count, -1, -1, -1).reshape(-1, ray_features.shape[-1])
                    temporal_cam = cam_pose[None, None, :, :].expand(view_count, t_count, h * w, -1).reshape(-1, cam_pose.shape[-1])
                    view_ids = torch.arange(view_count, device=self.device, dtype=torch.long).repeat_interleave(t_count * h * w)
                    # Wrist/gripper temporal tokens are view evidence, not static-camera
                    # geometry truth. Without wrist extrinsics, only their own grid
                    # coordinate and view embedding are trusted.
                    non_static = view_ids != 0
                    temporal_ray = torch.where(non_static[:, None], torch.zeros_like(temporal_ray), temporal_ray)
                    temporal_cam = torch.where(non_static[:, None], torch.zeros_like(temporal_cam), temporal_cam)
                    temporal_in = torch.cat([temporal_flat, temporal_grid, temporal_cam, temporal_ray], dim=-1)
                    time_ids = torch.arange(t_count, device=self.device, dtype=self.dtype).repeat_interleave(h * w).repeat(view_count)
                    if t_count > 1:
                        time_norm = (2.0 * time_ids / float(t_count - 1)) - 1.0
                    else:
                        time_norm = torch.zeros_like(time_ids)
                    temporal_tokens = (
                        self.visual_token_proj(temporal_in)
                        + self.modality_embedding.weight[1][None, :]
                        + self.temporal_visual_time_proj(time_norm[:, None])
                        + self.temporal_visual_view_embedding(
                            torch.clamp(view_ids, min=0, max=self.temporal_visual_view_embedding.num_embeddings - 1)
                        )
                    )
                    view_names = self._configured_vjepa_views()[:view_count]
                    temporal_visual = PicfTemporalVisualSupportState(
                        tokens=temporal_tokens,
                        time_ids=time_ids.to(dtype=torch.long),
                        view_ids=view_ids,
                        grid_index=temporal_grid,
                        grid_hw=torch.as_tensor([h, w], device=self.device, dtype=torch.long),
                        current_token_count=torch.as_tensor(h * w, device=self.device, dtype=torch.long),
                        valid=torch.tensor(True, device=self.device),
                        view_names=view_names,
                        grid_hw_by_view=torch.as_tensor([[h, w]] * view_count, device=self.device, dtype=torch.long),
                        source_hw_by_view=torch.as_tensor(
                            [[int(observation.rgb_static.shape[0]), int(observation.rgb_static.shape[1])]] * view_count,
                            device=self.device,
                            dtype=torch.long,
                        ),
                    )

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
                # Frozen AnyTouch runs under torch.inference_mode(); clone turns
                # the encoder output into a normal detached tensor before it
                # enters trainable PICF projection layers.
                dense_tokens = sensor.tokens.to(device=self.device, dtype=self.dtype).clone()
                dense_tokens_all.append(self.tactile_patch_token_proj(dense_tokens) + self.modality_embedding.weight[2][None, :])
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
                tactile_evidence_mask = tactile_contact_prob >= float(self.config.tactile_evidence_prob_floor)
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
                tactile_evidence_mask = tactile_contact_prob >= float(self.config.tactile_evidence_prob_floor)
            floor = float(self.config.tactile_evidence_prob_floor)
            on = max(float(self.config.tactile_anchor_prob_on), floor + self.config.epsilon_a)
            tactile_evidence_weight = torch.clamp(
                (tactile_contact_prob - floor) / max(on - floor, self.config.epsilon_a),
                min=0.0,
                max=1.0,
            )
            selected_indices = torch.nonzero(tactile_evidence_mask | tactile_anchor_mask, as_tuple=False).squeeze(-1)
            if selected_indices.numel() > 0:
                # Keep a group for each physical tactile sensor so group ids can
                # remain original sensor ids. Confident contacts expose dense
                # AnyTouch patch tokens; weaker calibrated contacts expose a
                # single sensor-level token. This preserves the belief-filter
                # semantics: soft contact is evidence, not a hard object label.
                tactile_group_tokens = tuple(
                    dense_tokens_all[index] if bool(tactile_anchor_mask[index].item()) else tactile_tokens_all[index][None, :]
                    for index in range(tactile_tokens_all.shape[0])
                )
                proposal_tokens = []
                proposal_align = []
                proposal_positions = []
                proposal_normals = []
                proposal_group_ids = []
                for sensor_index in selected_indices.tolist():
                    sensor_index = int(sensor_index)
                    base_token = tactile_tokens_all[sensor_index]
                    if bool(tactile_anchor_mask[sensor_index].item()):
                        dense_group = dense_tokens_all[sensor_index]
                        route_queries = (
                            self.tactile_group_route_queries.to(device=self.device, dtype=self.dtype)[None, :]
                            + base_token[None, None, :]
                        )
                        route_tokens, _ = self.tactile_route_reread(route_queries, dense_group[None, :])
                        route_tokens = route_tokens[0]
                    else:
                        route_tokens = base_token[None, :] * tactile_evidence_weight[sensor_index].clamp(0.0, 1.0)
                    proposal_tokens.append(route_tokens)
                    proposal_align.append(_normalize_tensor(self.tactile_align_proj(route_tokens), eps=self.config.epsilon_residual))
                    proposal_positions.append(tactile_positions_world[sensor_index][None, :].expand(route_tokens.shape[0], -1))
                    proposal_normals.append(tactile_normals_world[sensor_index][None, :].expand(route_tokens.shape[0], -1))
                    proposal_group_ids.append(
                        torch.full(
                            (route_tokens.shape[0],),
                            sensor_index,
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

        tracklet_state = None
        if bool(self.config.tracklet_memory_enabled) and observation.tracklet_xy is not None:
            xy = _to_tensor(observation.tracklet_xy, device=self.device, dtype=self.dtype).reshape(-1, 2)
            max_tokens = max(int(self.config.tracklet_max_tokens), 0)
            if max_tokens > 0 and xy.shape[0] > max_tokens:
                xy = xy[:max_tokens]
            n_track = int(xy.shape[0])
            if n_track > 0:
                velocity = (
                    _to_tensor(observation.tracklet_velocity, device=self.device, dtype=self.dtype).reshape(-1, 2)
                    if observation.tracklet_velocity is not None
                    else torch.zeros_like(xy)
                )
                if velocity.shape[0] < n_track:
                    velocity = fn.pad(velocity, (0, 0, 0, n_track - velocity.shape[0]))
                velocity = velocity[:n_track]
                visibility = (
                    _to_tensor(observation.tracklet_visibility, device=self.device, dtype=self.dtype).reshape(-1)
                    if observation.tracklet_visibility is not None
                    else torch.ones((n_track,), device=self.device, dtype=self.dtype)
                )
                confidence = (
                    _to_tensor(observation.tracklet_confidence, device=self.device, dtype=self.dtype).reshape(-1)
                    if observation.tracklet_confidence is not None
                    else torch.ones((n_track,), device=self.device, dtype=self.dtype)
                )
                if visibility.numel() < n_track:
                    visibility = fn.pad(visibility, (0, n_track - visibility.numel()), value=0.0)
                if confidence.numel() < n_track:
                    confidence = fn.pad(confidence, (0, n_track - confidence.numel()), value=0.0)
                visibility = visibility[:n_track].clamp(0.0, 1.0)
                confidence = confidence[:n_track].clamp(0.0, 1.0)
                valid_track = (visibility * confidence) >= float(self.config.tracklet_confidence_floor)
                track_ids = (
                    torch.as_tensor(observation.tracklet_ids, device=self.device, dtype=torch.long).reshape(-1)
                    if observation.tracklet_ids is not None
                    else torch.arange(n_track, device=self.device, dtype=torch.long)
                )
                view_ids = (
                    torch.as_tensor(observation.tracklet_view_ids, device=self.device, dtype=torch.long).reshape(-1)
                    if observation.tracklet_view_ids is not None
                    else torch.zeros((n_track,), device=self.device, dtype=torch.long)
                )
                age = (
                    _to_tensor(observation.tracklet_age, device=self.device, dtype=self.dtype).reshape(-1)
                    if observation.tracklet_age is not None
                    else torch.zeros((n_track,), device=self.device, dtype=self.dtype)
                )
                if track_ids.numel() < n_track:
                    track_ids = fn.pad(track_ids, (0, n_track - track_ids.numel()), value=-1)
                if view_ids.numel() < n_track:
                    view_ids = fn.pad(view_ids, (0, n_track - view_ids.numel()), value=0)
                if age.numel() < n_track:
                    age = fn.pad(age, (0, n_track - age.numel()), value=0.0)
                track_features = torch.cat(
                    [
                        xy,
                        velocity,
                        visibility[:n_track, None],
                        confidence[:n_track, None],
                        torch.clamp(age[:n_track, None], min=0.0),
                        _point_proj_fourier(xy, bands=4),
                    ],
                    dim=-1,
                )
                track_tokens = (
                    self.tracklet_token_proj(track_features)
                    + self.modality_embedding.weight[3][None, :]
                    + self.temporal_visual_view_embedding(
                        torch.clamp(view_ids[:n_track], min=0, max=self.temporal_visual_view_embedding.num_embeddings - 1)
                    )
                )
                tracklet_state = PicfTrackletSupportState(
                    tokens=track_tokens,
                    xy_norm=xy,
                    velocity_norm=velocity,
                    visibility=visibility[:n_track],
                    confidence=confidence[:n_track],
                    track_ids=track_ids[:n_track],
                    view_ids=view_ids[:n_track],
                    age=age[:n_track],
                    valid=valid_track[:n_track],
                )

        proposal_state = None
        if bool(self.config.proposal_memory_enabled) and (
            observation.proposal_centers_xy is not None or observation.proposal_boxes_xyxy is not None
        ):
            centers = (
                _to_tensor(observation.proposal_centers_xy, device=self.device, dtype=self.dtype).reshape(-1, 2)
                if observation.proposal_centers_xy is not None
                else None
            )
            boxes = (
                _to_tensor(observation.proposal_boxes_xyxy, device=self.device, dtype=self.dtype).reshape(-1, 4)
                if observation.proposal_boxes_xyxy is not None
                else None
            )
            if boxes is not None and centers is None:
                centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
            if centers is not None:
                max_props = max(int(self.config.proposal_max_tokens), 0)
                n_prop = int(centers.shape[0])
                if boxes is not None:
                    n_prop = min(n_prop, int(boxes.shape[0]))
                if max_props > 0:
                    n_prop = min(n_prop, max_props)
                if n_prop > 0:
                    centers = centers[:n_prop].clamp(0.0, 1.0)
                    if boxes is None:
                        boxes = torch.cat([centers, centers], dim=-1)
                    boxes = boxes[:n_prop].clamp(0.0, 1.0)
                    objectness = (
                        _to_tensor(observation.proposal_objectness, device=self.device, dtype=self.dtype).reshape(-1)
                        if observation.proposal_objectness is not None
                        else torch.ones((n_prop,), device=self.device, dtype=self.dtype)
                    )
                    if objectness.numel() < n_prop:
                        objectness = fn.pad(objectness, (0, n_prop - objectness.numel()), value=0.0)
                    objectness = objectness[:n_prop].clamp(0.0, 1.0)
                    view_ids = (
                        torch.as_tensor(observation.proposal_view_ids, device=self.device, dtype=torch.long).reshape(-1)
                        if observation.proposal_view_ids is not None
                        else torch.zeros((n_prop,), device=self.device, dtype=torch.long)
                    )
                    source_ids = (
                        torch.as_tensor(observation.proposal_source_ids, device=self.device, dtype=torch.long).reshape(-1)
                        if observation.proposal_source_ids is not None
                        else torch.zeros((n_prop,), device=self.device, dtype=torch.long)
                    )
                    if view_ids.numel() < n_prop:
                        view_ids = fn.pad(view_ids, (0, n_prop - view_ids.numel()), value=0)
                    if source_ids.numel() < n_prop:
                        source_ids = fn.pad(source_ids, (0, n_prop - source_ids.numel()), value=0)
                    proposal_age = (
                        _to_tensor(observation.proposal_age, device=self.device, dtype=self.dtype).reshape(-1)
                        if getattr(observation, "proposal_age", None) is not None
                        else torch.zeros((n_prop,), device=self.device, dtype=self.dtype)
                    )
                    if proposal_age.numel() < n_prop:
                        proposal_age = fn.pad(proposal_age, (0, n_prop - proposal_age.numel()), value=0.0)
                    proposal_age = proposal_age[:n_prop].clamp(min=0.0)
                    proposal_mask_xy = None
                    proposal_mask_weights = None
                    proposal_mask_offsets = None
                    if (
                        getattr(observation, "proposal_mask_xy", None) is not None
                        and getattr(observation, "proposal_mask_weights", None) is not None
                        and getattr(observation, "proposal_mask_offsets", None) is not None
                    ):
                        raw_xy = _to_tensor(observation.proposal_mask_xy, device=self.device, dtype=self.dtype).reshape(-1, 2)
                        raw_weights = _to_tensor(observation.proposal_mask_weights, device=self.device, dtype=self.dtype).reshape(-1)
                        raw_offsets = torch.as_tensor(observation.proposal_mask_offsets, device=self.device, dtype=torch.long).reshape(-1)
                        if raw_xy.numel() > 0 and raw_weights.numel() == raw_xy.shape[0] and raw_offsets.numel() >= n_prop + 1:
                            start0 = int(torch.clamp(raw_offsets[0], min=0).item())
                            endn = int(torch.clamp(raw_offsets[n_prop], min=start0).item())
                            start0 = min(start0, int(raw_xy.shape[0]))
                            endn = min(endn, int(raw_xy.shape[0]))
                            if endn > start0:
                                proposal_mask_xy = torch.clamp(raw_xy[start0:endn], min=0.0, max=1.0)
                                proposal_mask_weights = torch.clamp(raw_weights[start0:endn], min=0.0)
                                proposal_mask_offsets = raw_offsets[: n_prop + 1] - int(raw_offsets[0].item())
                                proposal_mask_offsets = torch.clamp(proposal_mask_offsets, min=0, max=endn - start0)
                    age_decay = max(float(getattr(self.config, "proposal_age_decay_steps", 8.0)), self.config.epsilon_a)
                    # Sparse proposal sidecars may reuse a nearby frame. Keep the
                    # objectness evidence soft and age-aware instead of treating
                    # a temporally borrowed box as current-frame truth.
                    objectness = objectness * torch.exp(-proposal_age / age_decay)
                    wh = torch.clamp(boxes[:, 2:] - boxes[:, :2], min=0.0)
                    area = (wh[:, :1] * wh[:, 1:2]).clamp(0.0, 1.0)
                    proposal_features = torch.cat(
                        [
                            centers,
                            boxes,
                            objectness[:, None],
                            area,
                            view_ids[:n_prop, None].to(dtype=self.dtype) / max(float(self.config.vjepa_max_views), 1.0),
                            source_ids[:n_prop, None].to(dtype=self.dtype) / 16.0,
                            _point_proj_fourier(centers, bands=4),
                        ],
                        dim=-1,
                    )
                    proposal_tokens = (
                        self.proposal_token_proj(proposal_features)
                        + self.modality_embedding.weight[3][None, :]
                        + self.temporal_visual_view_embedding(
                            torch.clamp(view_ids[:n_prop], min=0, max=self.temporal_visual_view_embedding.num_embeddings - 1)
                        )
                    )
                    proposal_state = PicfPseudoProposalState(
                        tokens=proposal_tokens,
                        centers_xy=centers,
                        boxes_xyxy=boxes,
                        objectness=objectness,
                        view_ids=view_ids[:n_prop],
                        source_ids=source_ids[:n_prop],
                        age=proposal_age,
                        valid=objectness >= float(self.config.proposal_confidence_floor),
                        mask_xy=proposal_mask_xy,
                        mask_weights=proposal_mask_weights,
                        mask_offsets=proposal_mask_offsets,
                    )

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
            tactile_evidence_mask=tactile_evidence_mask,
            tactile_evidence_weight=tactile_evidence_weight,
            tactile_normals_world=tactile_normals_world,
            tactile_contact_score=tactile_contact_score,
            tactile_contact_score_ema=tactile_contact_score_ema,
            fusion_attention_mean=fusion_attention_mean,
            projective_geometry=projective_geometry,
            point_pool_ids=point_pool_ids,
            point_positions_world=point_positions_world,
            point_projectable_mask=None,
            temporal_visual=temporal_visual,
            tracklet=tracklet_state,
            proposal=proposal_state,
        )
        token_field.point_projectable_mask = self._scene_point_candidate_mask(token_field, fallback_to_global=False)
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
        vl_grounding: PicfVLGroundingState | None = None,
        anchor_graph: PicfAnchorPriorGraphState | None = None,
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
        effector_count = self._effector_observation_count()
        role_ids = torch.cat(
            [
                torch.zeros((effector_count,), device=self.device, dtype=torch.long),
                torch.ones((max(n_obs - effector_count, 0),), device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        seed_indices = torch.full((n_obs,), -1, device=self.device, dtype=torch.long)
        queries = torch.zeros((1, n_obs, hidden_dim), device=self.device, dtype=self.dtype)
        vl_slot_priors, vl_slot_valid = self._vl_slot_point_priors(
            vl_grounding,
            role_ids,
            point_count=point_count,
        )
        graph_assignment = None
        mapg_obs_gate = self._mapg_gate(self.mapg_obs_gate_logit, anchor_graph)
        obs_point_floor = min(max(float(self.config.mapg_obs_point_mix_floor), 0.0), 1.0)
        mapg_obs_point_mix_gate = torch.clamp(
            mapg_obs_gate,
            min=obs_point_floor if anchor_graph is not None else 0.0,
            max=1.0,
        )
        graph_point_weights = None
        graph_visual_weights = None
        graph_pg_weights = None
        graph_temporal_weights = None
        graph_tactile_weights = None
        graph_tracklet_weights = None
        graph_proposal_weights = None
        anchor_address = None
        owner_active = None
        seed_point_priors = None
        if point_count > 0:
            pool_ids = self._point_pool_ids(token_field)
            seed_parts: list[tuple[slice, torch.Tensor]] = []
            local_indices = torch.nonzero(pool_ids == 0, as_tuple=False).squeeze(-1)
            # Coverage seeding may fall back to all global rows so every scene
            # slot has a tensor seed. This is intentionally separate from the
            # strict no-fallback mask used by VL heatmap lifting above.
            scene_mask = self._scene_point_candidate_mask(token_field, fallback_to_global=True)
            global_indices = torch.nonzero(scene_mask, as_tuple=False).squeeze(-1)
            all_global_indices = torch.nonzero(pool_ids == 1, as_tuple=False).squeeze(-1)
            if global_indices.numel() > 0 and global_indices.numel() < max(n_obs - effector_count, 0):
                extra_global = all_global_indices[~torch.isin(all_global_indices, global_indices)]
                if extra_global.numel() > 0:
                    global_indices = torch.cat([global_indices, extra_global], dim=0)
            if global_indices.numel() == 0:
                global_indices = all_global_indices
            if effector_count > 0:
                if local_indices.numel() > 0:
                    chosen_local = local_indices[
                        _fps_indices(token_field.point_positions[local_indices], min(effector_count, int(local_indices.numel())))
                    ]
                else:
                    chosen_local = _fps_indices(token_field.point_positions, min(effector_count, point_count))
                seed_parts.append((slice(0, effector_count), chosen_local))
            scene_count = max(n_obs - effector_count, 0)
            if scene_count > 0:
                if global_indices.numel() > 0:
                    chosen_scene = global_indices[
                        _fps_indices(token_field.point_positions[global_indices], min(scene_count, int(global_indices.numel())))
                    ]
                else:
                    chosen_scene = _fps_indices(token_field.point_positions, min(scene_count, point_count))
                seed_parts.append((slice(effector_count, n_obs), chosen_scene))
            for target_slice, chosen in seed_parts:
                if chosen.numel() > 0:
                    min_idx = int(chosen.min().item())
                    max_idx = int(chosen.max().item())
                    if min_idx < 0 or max_idx >= point_count:
                        raise RuntimeError(
                            "PICF observation-anchor seed index out of bounds: "
                            f"valid=[0,{point_count - 1}] got min={min_idx} max={max_idx}"
                        )
                start = int(target_slice.start or 0)
                stop = int(target_slice.stop or n_obs)
                take = min(int(chosen.shape[0]), max(stop - start, 0))
                if take > 0:
                    seed_indices[start : start + take] = chosen[:take]
                    queries[0, start : start + take] = token_field.point_tokens[chosen[:take]]
            obs_vl_gate = self._vl_gate(self.vl_obs_anchor_gate_logit, vl_grounding)
            # Static-camera VL priors are global/scene evidence. Effector
            # observation anchors keep their local/proprio/tactile seed
            # contract and never get replaced by a static-camera VL seed.
            scene_vl_slot_valid = vl_slot_valid & (role_ids != 0)
            if bool(scene_vl_slot_valid.any().item()) and bool((obs_vl_gate > 0.0).item()):
                vl_seed_indices = torch.argmax(vl_slot_priors, dim=-1)
                vl_seed_tokens = torch.zeros_like(queries)
                vl_seed_mask = torch.zeros((1, n_obs, 1), device=self.device, dtype=torch.bool)
                for row in torch.nonzero(scene_vl_slot_valid, as_tuple=False).squeeze(-1).tolist():
                    idx = int(vl_seed_indices[int(row)].item())
                    if 0 <= idx < point_count:
                        seed_indices[int(row)] = idx
                        vl_seed_tokens[0, int(row)] = token_field.point_tokens[idx]
                        vl_seed_mask[0, int(row), 0] = True
                if bool(vl_seed_mask.any().item()):
                    queries = torch.where(
                        vl_seed_mask,
                        ((1.0 - obs_vl_gate) * queries) + (obs_vl_gate * vl_seed_tokens),
                        queries,
                    )
        if anchor_graph is not None:
            slot_point_priors = vl_slot_priors
            if point_count > 0:
                seed_priors = torch.zeros((n_obs, point_count), device=self.device, dtype=self.dtype)
                seed_valid = (seed_indices >= 0) & (seed_indices < point_count)
                if bool(seed_valid.any().item()):
                    rows = torch.nonzero(seed_valid, as_tuple=False).squeeze(-1)
                    seed_priors[rows, seed_indices[rows]] = 1.0
                seed_point_priors = seed_priors
                slot_point_priors = _normalize_rows(slot_point_priors + seed_priors, eps=self.config.epsilon_a)
            graph_assignment = self._mapg_slot_assignment(
                anchor_graph,
                role_ids,
                slot_tokens=queries[0],
                slot_point_priors=slot_point_priors if point_count > 0 else None,
            )
            anchor_graph.obs_slot_assignment = graph_assignment
        if graph_assignment is not None and bool((mapg_obs_gate > 0.0).item()):
            graph_tokens = graph_assignment @ anchor_graph.anchor_tokens.to(device=self.device, dtype=self.dtype)
            queries = ((1.0 - mapg_obs_gate) * queries) + (mapg_obs_gate * graph_tokens[None, :, :])
        attn_public = torch.zeros((n_obs, token_field.fused_tokens.shape[0]), device=self.device, dtype=self.dtype)
        attn_visual = torch.zeros((n_obs, visual_count), device=self.device, dtype=self.dtype)
        public_role_bias = self._fused_read_role_bias(role_ids, token_field)
        # Same role boundary for attention bias: role-0 observation anchors may
        # read local/contact evidence, but they must not be point-biased by a
        # static-camera global VL prior.
        scene_vl_slot_valid = vl_slot_valid & (role_ids != 0)
        if point_count > 0 and token_field.fused_tokens.shape[0] > 0 and bool(scene_vl_slot_valid.any().item()):
            gate = self._vl_gate(self.vl_obs_anchor_gate_logit, vl_grounding)
            if bool((gate > 0.0).item()):
                if public_role_bias is None:
                    public_role_bias = torch.zeros((n_obs, token_field.fused_tokens.shape[0]), device=self.device, dtype=self.dtype)
                vl_bias = torch.zeros_like(public_role_bias)
                vl_bias[:, :point_count] = self._vl_centered_log_prior_bias(vl_slot_priors)
                vl_bias = torch.where(scene_vl_slot_valid[:, None], vl_bias, torch.zeros_like(vl_bias))
                public_role_bias = public_role_bias + (gate * vl_bias)
        if graph_assignment is not None and anchor_graph.point_priors is not None and point_count > 0:
            gate = self._mapg_gate(self.mapg_obs_gate_logit, anchor_graph)
            if bool((gate > 0.0).item()):
                if public_role_bias is None:
                    public_role_bias = torch.zeros((n_obs, token_field.fused_tokens.shape[0]), device=self.device, dtype=self.dtype)
                graph_point_priors = graph_assignment @ anchor_graph.point_priors.to(device=self.device, dtype=self.dtype)
                graph_point_weights = _normalize_rows(graph_point_priors, eps=self.config.epsilon_a)
                graph_bias = torch.zeros_like(public_role_bias)
                graph_bias[:, :point_count] = self._vl_centered_log_prior_bias(graph_point_weights)
                public_role_bias = public_role_bias + (gate * graph_bias)
        if graph_assignment is not None:
            if anchor_graph.pg_priors is not None and anchor_graph.pg_priors.numel() > 0:
                graph_pg_weights = _normalize_rows(
                    graph_assignment @ anchor_graph.pg_priors.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_a,
                )
            if anchor_graph.vjepa_temporal_priors is not None and anchor_graph.vjepa_temporal_priors.numel() > 0:
                graph_temporal_weights = _normalize_rows(
                    graph_assignment @ anchor_graph.vjepa_temporal_priors.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_a,
                )
            if anchor_graph.tactile_priors is not None and anchor_graph.tactile_priors.numel() > 0:
                graph_tactile_weights = _normalize_rows(
                    graph_assignment @ anchor_graph.tactile_priors.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_a,
                )
            if anchor_graph.tracklet_priors is not None and anchor_graph.tracklet_priors.numel() > 0:
                graph_tracklet_weights = _normalize_rows(
                    graph_assignment @ anchor_graph.tracklet_priors.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_a,
                )
            if anchor_graph.proposal_priors is not None and anchor_graph.proposal_priors.numel() > 0:
                graph_proposal_weights = _normalize_rows(
                    graph_assignment @ anchor_graph.proposal_priors.to(device=self.device, dtype=self.dtype),
                    eps=self.config.epsilon_a,
                )
            if anchor_graph.slot_address is not None and anchor_graph.slot_address.numel() > 0:
                address = anchor_graph.slot_address.to(device=self.device, dtype=self.dtype)
                if address.shape[0] < anchor_graph.anchor_tokens.shape[0]:
                    pad = anchor_graph.anchor_tokens.to(device=self.device, dtype=self.dtype)[address.shape[0] :]
                    address = torch.cat([address, pad], dim=0)
                anchor_address = graph_assignment @ address[: anchor_graph.anchor_tokens.shape[0]]
        visual_attn_bias = None
        if graph_assignment is not None and visual_count > 0:
            gate = self._mapg_gate(self.mapg_obs_gate_logit, anchor_graph)
            if bool((gate > 0.0).item()):
                graph_visual_priors = graph_assignment @ anchor_graph.visual_priors.to(device=self.device, dtype=self.dtype)
                graph_visual_weights = _normalize_rows(graph_visual_priors, eps=self.config.epsilon_a)
                visual_attn_bias = gate * self._vl_centered_log_prior_bias(graph_visual_weights)
        for _ in range(max(self.config.query_rounds, 1)):
            if visual_count > 0:
                queries, visual_weights = self.visual_native_reread(
                    queries,
                    dense_memory.visual_payload[None, :],
                    attn_bias=visual_attn_bias,
                )
                attn_visual = visual_weights[0]
            queries, attn_public = self.obs_reader(queries, token_field.fused_tokens[None, :], attn_bias=public_role_bias)
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
            geometry_positions = self._world_point_positions(token_field)
            denom = torch.clamp(routing_mass_point.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            point_weights = routing_mass_point / denom
            if graph_point_weights is not None and bool((mapg_obs_point_mix_gate > 0.0).item()):
                graph_valid = _row_has_mass(graph_point_weights, eps=self.config.epsilon_a)
                graph_mix = torch.where(graph_valid[:, None], graph_point_weights, point_weights)
                point_weights = ((1.0 - mapg_obs_point_mix_gate) * point_weights) + (mapg_obs_point_mix_gate * graph_mix)
                point_weights = torch.clamp(point_weights, min=0.0)
                point_weights = point_weights / torch.clamp(point_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            seed_mix = min(max(float(getattr(self.config, "observation_anchor_seed_point_mix", 0.0)), 0.0), 1.0)
            if seed_mix > 0.0 and seed_point_priors is not None and seed_point_priors.numel() == point_weights.numel():
                seed_valid_rows = _row_has_mass(seed_point_priors, eps=self.config.epsilon_a)
                seed_mix_weights = torch.where(seed_valid_rows[:, None], seed_point_priors, point_weights)
                point_weights = ((1.0 - seed_mix) * point_weights) + (seed_mix * seed_mix_weights)
                point_weights = torch.clamp(point_weights, min=0.0)
                point_weights = point_weights / torch.clamp(point_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
            x = point_weights @ geometry_positions
            S = _weighted_cov(geometry_positions, point_weights, x, self.config)
            a = _extent_from_cov(S, self.config)
        else:
            point_weights = torch.zeros((n_obs, 0), device=self.device, dtype=self.dtype)
            x = torch.zeros((n_obs, 3), device=self.device, dtype=self.dtype)
            S = _diag_embed(torch.full((n_obs, 3), self.config.epsilon_s, device=self.device, dtype=self.dtype))
            a = _to_tensor(self.config.a_min_m, device=self.device, dtype=self.dtype)[None, :].expand(n_obs, -1)
        binding_parts: list[torch.Tensor] = []

        def _add_binding_part(weights: torch.Tensor | None, tokens: torch.Tensor | None) -> None:
            signature = self._support_binding_signature(weights, tokens)
            if signature is not None and signature.numel() > 0:
                binding_parts.append(signature)

        _add_binding_part(point_weights, token_field.point_tokens)
        _add_binding_part(
            graph_visual_weights if graph_visual_weights is not None else routing_mass_visual,
            token_field.visual_tokens,
        )
        _add_binding_part(
            graph_temporal_weights,
            None if token_field.temporal_visual is None else token_field.temporal_visual.tokens,
        )
        _add_binding_part(
            graph_tactile_weights if graph_tactile_weights is not None else routing_mass_tactile,
            token_field.tactile_tokens,
        )
        if token_field.tracklet is not None:
            _add_binding_part(graph_tracklet_weights, token_field.tracklet.tokens)
        if token_field.proposal is not None:
            _add_binding_part(graph_proposal_weights, token_field.proposal.tokens)
        obs_binding_signature = (
            _normalize_tensor(torch.stack(binding_parts, dim=0).mean(dim=0), eps=self.config.epsilon_residual)
            if binding_parts
            else self._binding_keys(obs_tokens)
        )
        if graph_assignment is not None:
            owner_active = self._observation_owner_active_from_graph(
                anchor_graph,
                graph_assignment,
                role_ids,
                obs_x=x,
                obs_binding_signature=obs_binding_signature,
                obs_point_weights=point_weights,
            )
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
            role_ids=role_ids,
            graph_assignment=graph_assignment,
            graph_point_weights=graph_point_weights,
            graph_visual_weights=graph_visual_weights,
            graph_pg_weights=graph_pg_weights,
            graph_temporal_weights=graph_temporal_weights,
            graph_tactile_weights=graph_tactile_weights,
            graph_tracklet_weights=graph_tracklet_weights,
            graph_proposal_weights=graph_proposal_weights,
            anchor_address=anchor_address if anchor_address is not None else obs_tokens,
            owner_active=owner_active,
            support_signature=torch.cat(
                [
                    routing_mass_point.mean(dim=-1, keepdim=True) if routing_mass_point.numel() > 0 else torch.zeros((n_obs, 1), device=self.device, dtype=self.dtype),
                    routing_mass_visual.mean(dim=-1, keepdim=True) if routing_mass_visual.numel() > 0 else torch.zeros((n_obs, 1), device=self.device, dtype=self.dtype),
                    routing_mass_tactile.mean(dim=-1, keepdim=True) if routing_mass_tactile.numel() > 0 else torch.zeros((n_obs, 1), device=self.device, dtype=self.dtype),
                ],
                dim=-1,
            ),
            binding_signature=obs_binding_signature,
        )

    def _initial_persistent(self) -> tuple[torch.Tensor, ...]:
        k = self.config.persistent_anchors
        mu = torch.zeros((k, self.config.latent_dim), device=self.device, dtype=self.dtype)
        var = torch.full((k, self.config.latent_dim), self.config.sigma_reset**2, device=self.device, dtype=self.dtype)
        h = self.posterior_slot_hidden.to(device=self.device, dtype=self.dtype)
        c = torch.zeros((k, self.config.posterior_hidden_dim), device=self.device, dtype=self.dtype)
        a = _to_tensor(self.config.a_min_m, device=self.device, dtype=self.dtype)[None, :].expand(k, -1).clone()
        S = _diag_embed(torch.clamp((a / 2.0) ** 2, min=self.config.epsilon_s))
        x = torch.zeros((k, 3), device=self.device, dtype=self.dtype)
        alpha = torch.full((k,), self.config.alpha_init, device=self.device, dtype=self.dtype)
        return h, c, mu, var, x, S, a, alpha

    def _bootstrap_prior_geometry_from_observation(
        self,
        x_prior: torch.Tensor,
        S_prior: torch.Tensor,
        a_prior: torch.Tensor,
        obs: PicfObservationAnchorState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(self.config.posterior_bootstrap_from_observation):
            return x_prior, S_prior, a_prior
        if obs.x.shape[0] == 0 or obs.role_ids is None or obs.role_ids.numel() != obs.x.shape[0]:
            return x_prior, S_prior, a_prior
        posterior_roles = self._posterior_role_ids().to(device=self.device, dtype=torch.long)
        obs_roles = obs.role_ids.to(device=self.device, dtype=torch.long)
        x = x_prior.clone()
        S = S_prior.clone()
        a = a_prior.clone()
        for role_value in torch.unique(posterior_roles).tolist():
            slot_indices = torch.nonzero(posterior_roles == int(role_value), as_tuple=False).squeeze(-1)
            obs_indices = torch.nonzero(obs_roles == int(role_value), as_tuple=False).squeeze(-1)
            if slot_indices.numel() == 0 or obs_indices.numel() == 0:
                continue
            take = min(int(slot_indices.numel()), int(obs_indices.numel()))
            if int(obs_indices.numel()) > take:
                selected = obs_indices[_fps_indices(obs.x[obs_indices], take)]
            else:
                selected = obs_indices[:take]
            slots = slot_indices[:take]
            x[slots] = obs.x[selected]
            S[slots] = obs.S[selected]
            a[slots] = obs.a[selected]
        return x, S, a

    def _current_prior(self, previous: PicfPreviousState | None, observation: PicfObservation) -> tuple[torch.Tensor, ...]:
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

    def _posterior_lifecycle_calibration(
        self,
        support_raw: torch.Tensor,
        dustbin_raw_all: torch.Tensor,
        owner_active: torch.Tensor | None,
        alpha_prior: torch.Tensor,
        identity_innovation_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Factor object-file lifecycle from raw binding and dustbin evidence.

        This is not an auxiliary loss. It is a posterior-calibration layer:
        stable high-support slots keep their object-file identity, while
        inactive/duplicate measurement rows are separated from genuinely
        unexplained object evidence.
        """
        slot_count = int(support_raw.shape[0])
        obs_count = int(support_raw.shape[1]) if support_raw.ndim == 2 else 0
        device = support_raw.device
        dtype = support_raw.dtype
        zeros_slot = torch.zeros((slot_count,), device=device, dtype=dtype)
        ones_slot = torch.ones((slot_count,), device=device, dtype=dtype)
        support_mass = support_raw.sum(dim=1) if obs_count > 0 else zeros_slot
        if obs_count > 0:
            support_cond = support_raw / torch.clamp(support_mass[:, None], min=self.config.epsilon_a)
            entropy = -torch.sum(
                support_cond * torch.log(torch.clamp(support_cond, min=self.config.epsilon_a)),
                dim=1,
            )
            if obs_count > 1:
                entropy = entropy / math.log(float(obs_count))
                top2 = torch.topk(support_cond, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
            else:
                entropy = zeros_slot
                margin = torch.where(support_mass > self.config.epsilon_a, ones_slot, zeros_slot)
        else:
            support_cond = torch.zeros((slot_count, 0), device=device, dtype=dtype)
            entropy = zeros_slot
            margin = zeros_slot
        if owner_active is not None and owner_active.numel() == obs_count and obs_count > 0:
            obs_owner = torch.clamp(
                torch.nan_to_num(owner_active.to(device=device, dtype=dtype).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0),
                min=0.0,
                max=1.0,
            )
            slot_owner = support_cond @ obs_owner
            inactive_dustbin = torch.sum(dustbin_raw_all.to(device=device, dtype=dtype).reshape(-1) * (1.0 - obs_owner))
            unexplained_dustbin = torch.sum(dustbin_raw_all.to(device=device, dtype=dtype).reshape(-1) * obs_owner)
            dustbin_for_recycle = dustbin_raw_all.to(device=device, dtype=dtype).reshape(-1) * obs_owner
        else:
            slot_owner = ones_slot if obs_count > 0 else zeros_slot
            inactive_dustbin = torch.zeros((), device=device, dtype=dtype)
            unexplained_dustbin = dustbin_raw_all.to(device=device, dtype=dtype).reshape(-1).sum()
            dustbin_for_recycle = dustbin_raw_all.to(device=device, dtype=dtype).reshape(-1)
        support_temp = max(float(getattr(self.config, "posterior_lifecycle_support_temperature", 0.05)), float(self.config.epsilon_a))
        margin_temp = max(float(getattr(self.config, "posterior_lifecycle_margin_temperature", 0.05)), float(self.config.epsilon_a))
        support_min = float(getattr(self.config, "posterior_lifecycle_support_min", 0.05))
        margin_min = float(getattr(self.config, "posterior_lifecycle_margin_min", 0.02))
        support_conf = torch.sigmoid((support_mass - support_min) / support_temp)
        margin_conf = torch.sigmoid((margin - margin_min) / margin_temp)
        entropy_conf = 1.0 - torch.clamp(entropy, min=0.0, max=1.0)
        owner_conf = torch.clamp(slot_owner, min=0.0, max=1.0)
        entropy_weight = min(max(float(getattr(self.config, "posterior_lifecycle_entropy_weight", 0.50)), 0.0), 1.0)
        owner_weight = min(max(float(getattr(self.config, "posterior_lifecycle_owner_weight", 0.50)), 0.0), 1.0)
        concentration_conf = ((1.0 - entropy_weight) + (entropy_weight * entropy_conf)).clamp(0.0, 1.0)
        owner_factor = ((1.0 - owner_weight) + (owner_weight * owner_conf)).clamp(0.0, 1.0)
        assignment_conf = (support_conf * margin_conf * concentration_conf * owner_factor).clamp(0.0, 1.0)
        innovation_downweight = max(float(getattr(self.config, "posterior_lifecycle_innovation_downweight", 1.0)), 0.0)
        innovation_values = identity_innovation_norm.to(device=device, dtype=dtype).reshape(-1)
        if innovation_values.numel() >= slot_count:
            slot_innovation = innovation_values[:slot_count]
        else:
            slot_innovation = zeros_slot
        innovation_stability = torch.exp(-innovation_downweight * slot_innovation)
        innovation_stability = torch.clamp(innovation_stability, min=0.0, max=1.0)
        alpha_conf = torch.clamp(alpha_prior.to(device=device, dtype=dtype).reshape(-1)[:slot_count], min=0.0, max=1.0)
        survival_prob = torch.maximum(assignment_conf, alpha_conf * innovation_stability)
        survival_prob = torch.clamp(survival_prob, min=0.0, max=1.0)
        reset_allowance = torch.clamp(1.0 - survival_prob, min=0.0, max=1.0)
        if not bool(getattr(self.config, "posterior_lifecycle_calibration_enabled", True)):
            reset_allowance = ones_slot
            survival_prob = torch.clamp(alpha_conf, min=0.0, max=1.0)
        return {
            "assignment_confidence": assignment_conf,
            "support_entropy": entropy,
            "support_margin": margin,
            "owner_reliability": slot_owner,
            "survival_prob": survival_prob,
            "reset_allowance": reset_allowance,
            "inactive_dustbin_mass": inactive_dustbin.reshape(()),
            "unexplained_dustbin_mass": unexplained_dustbin.reshape(()),
            "dustbin_for_recycle": dustbin_for_recycle,
        }

    def _posterior_file_competition(
        self,
        support_raw: torch.Tensor,
        dustbin_raw_all: torch.Tensor,
        *,
        x_prior: torch.Tensor,
        role_ids: torch.Tensor,
        alpha_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Move duplicate persistent file assignments into no-object dustbin.

        `_sinkhorn_dustbin` makes observation columns compete, but without a
        no-object column each persistent file still receives measurement mass.
        This layer adds the missing object-file explaining-away step: same-role
        files may all exist as capacity, but only distinct support owners are
        allowed to update from the same observation evidence.
        """

        if support_raw.ndim != 2:
            zeros = torch.zeros((0,), device=self.device, dtype=self.dtype)
            return {
                "support_raw": support_raw,
                "dustbin_raw_all": dustbin_raw_all,
                "active": zeros,
                "demoted_mass": zeros,
                "duplicate_overlap_max": torch.zeros((), device=self.device, dtype=self.dtype),
                "active_duplicate_overlap_max": torch.zeros((), device=self.device, dtype=self.dtype),
            }
        slot_count = int(support_raw.shape[0])
        if slot_count == 0:
            zeros = torch.zeros((0,), device=self.device, dtype=self.dtype)
            return {
                "support_raw": support_raw,
                "dustbin_raw_all": dustbin_raw_all,
                "active": zeros,
                "demoted_mass": zeros,
                "duplicate_overlap_max": torch.zeros((), device=self.device, dtype=self.dtype),
                "active_duplicate_overlap_max": torch.zeros((), device=self.device, dtype=self.dtype),
            }
        if not bool(getattr(self.config, "posterior_file_competition_enabled", True)):
            active = torch.ones((slot_count,), device=self.device, dtype=self.dtype)
            return {
                "support_raw": support_raw,
                "dustbin_raw_all": dustbin_raw_all,
                "active": active,
                "demoted_mass": torch.zeros_like(active),
                "duplicate_overlap_max": torch.zeros((), device=self.device, dtype=self.dtype),
                "active_duplicate_overlap_max": torch.zeros((), device=self.device, dtype=self.dtype),
            }

        support = torch.clamp(
            torch.nan_to_num(support_raw.to(device=self.device, dtype=self.dtype), nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        dustbin = torch.clamp(
            torch.nan_to_num(dustbin_raw_all.to(device=self.device, dtype=self.dtype), nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
        support_mass = support.sum(dim=1)
        support_cond = support / torch.clamp(support_mass[:, None], min=self.config.epsilon_a)
        support_norm = torch.sqrt(torch.clamp((support_cond * support_cond).sum(dim=1), min=self.config.epsilon_a))
        support_overlap = (support_cond @ support_cond.T) / torch.clamp(support_norm[:, None] * support_norm[None, :], min=self.config.epsilon_a)
        support_overlap = torch.clamp(torch.nan_to_num(support_overlap, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)

        duplicate_overlap = torch.where(
            support_overlap >= float(getattr(self.config, "posterior_file_competition_support_overlap_threshold", 0.80)),
            support_overlap,
            torch.zeros_like(support_overlap),
        )
        if (
            bool(getattr(self.config, "posterior_file_competition_geometry_duplicate_enabled", True))
            and x_prior.numel() > 0
            and x_prior.shape[0] >= slot_count
        ):
            centers = x_prior.to(device=self.device, dtype=self.dtype)[:slot_count, :3]
            dist2 = torch.cdist(centers, centers).pow(2)
            sigma = max(
                float(getattr(self.config, "posterior_file_competition_geometry_sigma_m", 0.04)),
                self.config.epsilon_a,
            )
            geom_overlap = torch.exp(-dist2 / (2.0 * sigma * sigma))
            geom_threshold = min(
                max(float(getattr(self.config, "posterior_file_competition_geometry_threshold", 0.70)), 0.0),
                1.0,
            )
            geom_duplicate = torch.where(geom_overlap >= geom_threshold, geom_overlap, torch.zeros_like(geom_overlap))
            duplicate_overlap = torch.maximum(duplicate_overlap, geom_duplicate)
        duplicate_overlap = duplicate_overlap - torch.diag(torch.diag(duplicate_overlap))

        roles = role_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if roles.numel() < slot_count:
            roles = torch.cat(
                [
                    roles,
                    torch.ones((slot_count - int(roles.numel()),), device=self.device, dtype=torch.long),
                ],
                dim=0,
            )
        roles = roles[:slot_count]
        alpha = torch.clamp(alpha_prior.to(device=self.device, dtype=self.dtype).reshape(-1)[:slot_count], min=0.0, max=1.0)
        if alpha.numel() < slot_count:
            alpha = torch.cat([alpha, torch.zeros((slot_count - int(alpha.numel()),), device=self.device, dtype=self.dtype)], dim=0)
        if support_cond.shape[1] > 1:
            top2 = torch.topk(support_cond, k=2, dim=1).values
            margin = torch.clamp(top2[:, 0] - top2[:, 1], min=0.0, max=1.0)
        else:
            margin = torch.where(support_mass > self.config.epsilon_a, torch.ones_like(support_mass), torch.zeros_like(support_mass))
        tie_break = torch.arange(slot_count, device=self.device, dtype=self.dtype) * self.config.epsilon_a
        score = support_mass * (0.25 + 0.75 * alpha) * (0.25 + 0.75 * margin) - tie_break
        active = torch.zeros((slot_count,), device=self.device, dtype=self.dtype)
        min_per_role = max(int(getattr(self.config, "posterior_file_competition_min_per_role", 1)), 0)
        max_per_role = max(int(getattr(self.config, "posterior_file_competition_max_per_role", 0)), 0)
        min_support = max(float(getattr(self.config, "posterior_file_competition_min_support", 0.02)), 0.0)
        relative_threshold = min(
            max(float(getattr(self.config, "posterior_file_competition_relative_score_threshold", 0.0)), 0.0),
            1.0,
        )
        for role_value in torch.unique(roles, sorted=True).tolist():
            role_indices = torch.nonzero(roles == int(role_value), as_tuple=False).squeeze(-1)
            if role_indices.numel() == 0:
                continue
            local_score = score.index_select(0, role_indices)
            order = torch.argsort(local_score, descending=True)
            kept: list[int] = []
            best_score = torch.clamp(local_score.max(), min=self.config.epsilon_a)
            for local_rank in order.tolist():
                idx = int(role_indices[int(local_rank)].item())
                if max_per_role > 0 and len(kept) >= max_per_role:
                    continue
                if len(kept) >= min_per_role and float(support_mass[idx].item()) < min_support:
                    continue
                if relative_threshold > 0.0 and len(kept) >= min_per_role:
                    relative_score = float((score[idx] / best_score).item())
                    if relative_score < relative_threshold:
                        continue
                if kept:
                    kept_t = torch.as_tensor(kept, device=self.device, dtype=torch.long)
                    max_dup = float(duplicate_overlap[idx, kept_t].max().item())
                    if max_dup > 0.0 and len(kept) >= min_per_role:
                        continue
                kept.append(idx)
            if not kept and role_indices.numel() > 0 and min_per_role > 0:
                kept.append(int(role_indices[int(order[0].item())].item()))
            if kept:
                active[torch.as_tensor(kept, device=self.device, dtype=torch.long)] = 1.0

        demoted = support * (1.0 - active[:, None])
        support_active = support * active[:, None]
        dustbin_active = dustbin + demoted.sum(dim=0)
        duplicate_values = duplicate_overlap[roles[:, None] == roles[None, :]]
        duplicate_max = (
            duplicate_values.max().reshape(())
            if duplicate_values.numel() > 0
            else torch.zeros((), device=self.device, dtype=self.dtype)
        )
        eye = torch.eye(slot_count, device=self.device, dtype=torch.bool)
        active_bool = active >= 0.5
        active_duplicate_mask = (
            (roles[:, None] == roles[None, :])
            & active_bool[:, None]
            & active_bool[None, :]
            & ~eye
        )
        active_duplicate_values = duplicate_overlap[active_duplicate_mask]
        active_duplicate_max = (
            active_duplicate_values.max().reshape(())
            if active_duplicate_values.numel() > 0
            else torch.zeros((), device=self.device, dtype=self.dtype)
        )
        return {
            "support_raw": support_active,
            "dustbin_raw_all": dustbin_active,
            "active": active,
            "demoted_mass": demoted.sum(dim=1),
            "duplicate_overlap_max": duplicate_max,
            "active_duplicate_overlap_max": active_duplicate_max,
        }

    def _posterior_birth_competition(
        self,
        recycle: torch.Tensor,
        *,
        file_active: torch.Tensor,
        role_ids: torch.Tensor,
        alpha_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Select reserve files allowed to consume dustbin residual evidence.

        Existing object updates and new-object births are different transport
        decisions. `_posterior_file_competition` demotes duplicate owners into
        the no-object/dustbin row. This method prevents that demoted dustbin
        evidence from being broadcast back into every inactive same-role file.
        It is the fixed-capacity analogue of DETR/Slot-style no-object
        competition: most reserve files stay null; only a small number of
        high-reset, low-alpha files can become birth candidates.
        """

        count = int(recycle.numel())
        if count == 0:
            return torch.zeros((0,), device=self.device, dtype=self.dtype)
        if not bool(getattr(self.config, "posterior_birth_competition_enabled", True)):
            return torch.ones((count,), device=self.device, dtype=self.dtype)

        recycle_score = torch.clamp(
            torch.nan_to_num(recycle.to(device=self.device, dtype=self.dtype).reshape(-1)[:count], nan=0.0),
            min=0.0,
            max=1.0,
        )
        active = torch.zeros((count,), device=self.device, dtype=self.dtype)
        if file_active is not None and file_active.numel() > 0:
            n = min(count, int(file_active.numel()))
            active[:n] = torch.clamp(
                file_active.to(device=self.device, dtype=self.dtype).reshape(-1)[:n],
                min=0.0,
                max=1.0,
            )
        alpha = torch.zeros((count,), device=self.device, dtype=self.dtype)
        if alpha_prior is not None and alpha_prior.numel() > 0:
            n = min(count, int(alpha_prior.numel()))
            alpha[:n] = torch.clamp(
                alpha_prior.to(device=self.device, dtype=self.dtype).reshape(-1)[:n],
                min=0.0,
                max=1.0,
            )
        roles = torch.ones((count,), device=self.device, dtype=torch.long)
        if role_ids is not None and role_ids.numel() > 0:
            n = min(count, int(role_ids.numel()))
            roles[:n] = role_ids.to(device=self.device, dtype=torch.long).reshape(-1)[:n]

        score = recycle_score.clone()
        if bool(getattr(self.config, "posterior_birth_competition_inactive_only", True)):
            score = score * (1.0 - active)
        alpha_power = max(float(getattr(self.config, "posterior_birth_alpha_suppression_power", 0.0)), 0.0)
        if alpha_power > 0.0:
            score = score * torch.pow(torch.clamp(1.0 - alpha, min=0.0, max=1.0), alpha_power)
        min_score = max(float(getattr(self.config, "posterior_birth_competition_min_score", 0.0)), 0.0)
        max_per_role = max(int(getattr(self.config, "posterior_birth_competition_max_per_role", 1)), 0)
        birth = torch.zeros((count,), device=self.device, dtype=self.dtype)
        if max_per_role == 0:
            return birth
        tie_break = torch.arange(count, device=self.device, dtype=self.dtype) * self.config.epsilon_a
        score = score - tie_break
        for role_value in torch.unique(roles, sorted=True).tolist():
            role_indices = torch.nonzero(roles == int(role_value), as_tuple=False).squeeze(-1)
            if role_indices.numel() == 0:
                continue
            local_score = score.index_select(0, role_indices)
            order = torch.argsort(local_score, descending=True)
            kept: list[int] = []
            for local_rank in order.tolist():
                idx = int(role_indices[int(local_rank)].item())
                if len(kept) >= max_per_role:
                    break
                if float(score[idx].item()) < min_score:
                    continue
                kept.append(idx)
            if kept:
                birth[torch.as_tensor(kept, device=self.device, dtype=torch.long)] = 1.0
        return birth

    def _binding_logits(
        self,
        h_prior: torch.Tensor,
        x_prior: torch.Tensor,
        S_prior: torch.Tensor,
        obs: PicfObservationAnchorState,
        previous: PicfPreviousState | None = None,
        innovation_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if obs.tokens.shape[0] == 0:
            return torch.zeros((self.config.persistent_anchors, 0), device=self.device, dtype=self.dtype), {}
        binding_debug: dict[str, torch.Tensor] = {}
        h_norm = _normalize_tensor(h_prior, eps=self.config.epsilon_residual)
        o_norm = _normalize_tensor(obs.tokens, eps=self.config.epsilon_residual)
        hidden_score = h_norm @ o_norm.T
        delta = obs.x[None, :, :] - x_prior[:, None, :]
        S_diag = torch.diagonal(S_prior, dim1=-2, dim2=-1)
        maha = torch.sum((delta**2) / torch.clamp(S_diag[:, None, :] + (self.config.bind_sigma_m**2), min=self.config.epsilon_s), dim=-1)
        logits = (self.config.lambda_bind_hidden * hidden_score) - (self.config.lambda_bind_geom * maha)
        if previous is not None:
            innovation_decay = torch.exp(
                -float(self.config.bind_address_innovation_downweight) * self._innovation_risk_scalar(innovation_norm)
            )
            support_terms: list[torch.Tensor] = []
            prev = previous.posterior
            for prev_name, obs_value in (
                ("point_signature", obs.graph_point_weights),
                ("visual_signature", obs.graph_visual_weights),
                ("temporal_signature", obs.graph_temporal_weights),
                ("pg_signature", obs.graph_pg_weights),
                ("tactile_signature", obs.graph_tactile_weights),
                ("tracklet_signature", obs.graph_tracklet_weights),
                ("proposal_signature", obs.graph_proposal_weights),
            ):
                prev_sig = getattr(prev, prev_name, None)
                if prev_sig is None or obs_value is None or prev_sig.numel() == 0 or obs_value.numel() == 0:
                    continue
                if prev_sig.shape[-1] != obs_value.shape[-1]:
                    continue
                support_terms.append(
                    _normalize_rows(prev_sig.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a)
                    @ _normalize_rows(obs_value.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_a).T
                )
            if support_terms:
                support_score = torch.stack(support_terms, dim=0).mean(dim=0)
                inertia = torch.ones((logits.shape[0],), device=self.device, dtype=self.dtype)
                if prev.alpha is not None:
                    inertia = inertia * prev.alpha.to(device=self.device, dtype=self.dtype).reshape(-1)[: logits.shape[0]].clamp(0.0, 1.0)
                if prev.recycle_gate is not None:
                    inertia = inertia * (1.0 - prev.recycle_gate.to(device=self.device, dtype=self.dtype).reshape(-1)[: logits.shape[0]].clamp(0.0, 1.0))
                inertia = inertia * innovation_decay
                logits = logits + (float(self.config.bind_support_signature_weight) * inertia[:, None] * support_score)
            if prev.binding_signature is not None and obs.binding_signature is not None and prev.binding_signature.numel() > 0 and obs.binding_signature.numel() > 0:
                prev_binding = prev.binding_signature.to(device=self.device, dtype=self.dtype)[: logits.shape[0]]
                obs_binding = obs.binding_signature.to(device=self.device, dtype=self.dtype)
                if prev_binding.shape[-1] == obs_binding.shape[-1]:
                    binding_score = _normalize_tensor(prev_binding, eps=self.config.epsilon_residual) @ _normalize_tensor(
                        obs_binding,
                        eps=self.config.epsilon_residual,
                    ).T
                    quadratic_score, low_rank_score = self._binding_signature_quadratic_scores(prev_binding, obs_binding)
                    bind_gate = torch.ones((logits.shape[0],), device=self.device, dtype=self.dtype)
                    if prev.alpha is not None:
                        bind_gate = bind_gate * prev.alpha.to(device=self.device, dtype=self.dtype).reshape(-1)[: logits.shape[0]].clamp(0.0, 1.0)
                    if prev.recycle_gate is not None:
                        bind_gate = bind_gate * (
                            1.0 - prev.recycle_gate.to(device=self.device, dtype=self.dtype).reshape(-1)[: logits.shape[0]].clamp(0.0, 1.0)
                        )
                    bind_gate = bind_gate * innovation_decay
                    binding_debug["binding_signature_linear_score_mean"] = binding_score.mean().reshape(())
                    binding_debug["binding_signature_linear_score_abs_mean"] = binding_score.abs().mean().reshape(())
                    binding_debug["binding_signature_gate_mean"] = bind_gate.mean().reshape(())
                    combined_score = float(self.config.bind_embedding_signature_weight) * binding_score
                    if quadratic_score is not None:
                        binding_debug["binding_signature_quadratic_score_mean"] = quadratic_score.mean().reshape(())
                        binding_debug["binding_signature_quadratic_score_abs_mean"] = quadratic_score.abs().mean().reshape(())
                        combined_score = combined_score + (
                            float(getattr(self.config, "bind_quadratic_signature_weight", 0.0)) * quadratic_score
                        )
                    if low_rank_score is not None:
                        binding_debug["binding_signature_low_rank_score_mean"] = low_rank_score.mean().reshape(())
                        binding_debug["binding_signature_low_rank_score_abs_mean"] = low_rank_score.abs().mean().reshape(())
                        combined_score = combined_score + (
                            float(getattr(self.config, "bind_low_rank_signature_weight", 0.0)) * low_rank_score
                        )
                    calibrated_score = self._calibrate_pairwise_binding_score(combined_score)
                    binding_debug["binding_signature_combined_score_mean"] = combined_score.mean().reshape(())
                    binding_debug["binding_signature_combined_score_abs_mean"] = combined_score.abs().mean().reshape(())
                    binding_debug["binding_signature_calibrated_score_mean"] = calibrated_score.mean().reshape(())
                    binding_debug["binding_signature_calibrated_score_abs_mean"] = calibrated_score.abs().mean().reshape(())
                    binding_debug["binding_signature_calibrated_score_std"] = torch.std(calibrated_score, unbiased=False).reshape(())
                    if calibrated_score.shape[1] > 1:
                        top2 = torch.topk(calibrated_score, k=2, dim=1).values
                        binding_debug["binding_signature_calibrated_top1_margin_mean"] = (top2[:, 0] - top2[:, 1]).mean().reshape(())
                    else:
                        binding_debug["binding_signature_calibrated_top1_margin_mean"] = torch.zeros((), device=self.device, dtype=self.dtype)
                    logits = logits + (bind_gate[:, None] * calibrated_score)
            if prev.slot_address is not None and obs.anchor_address is not None and prev.slot_address.numel() > 0 and obs.anchor_address.numel() > 0:
                prev_addr = prev.slot_address.to(device=self.device, dtype=self.dtype)[: logits.shape[0]]
                obs_addr = obs.anchor_address.to(device=self.device, dtype=self.dtype)
                if prev_addr.shape[-1] == obs_addr.shape[-1]:
                    address_score = _normalize_tensor(prev_addr, eps=self.config.epsilon_residual) @ _normalize_tensor(obs_addr, eps=self.config.epsilon_residual).T
                    addr_gate = torch.ones((logits.shape[0],), device=self.device, dtype=self.dtype)
                    if prev.alpha is not None:
                        addr_gate = addr_gate * prev.alpha.to(device=self.device, dtype=self.dtype).reshape(-1)[: logits.shape[0]].clamp(0.0, 1.0)
                    if prev.recycle_gate is not None:
                        recycle_prev = prev.recycle_gate.to(device=self.device, dtype=self.dtype).reshape(-1)[: logits.shape[0]].clamp(0.0, 1.0)
                        addr_gate = addr_gate * torch.exp(-float(self.config.bind_address_innovation_downweight) * recycle_prev)
                    addr_gate = addr_gate * innovation_decay
                    logits = logits + (float(self.config.bind_address_weight) * addr_gate[:, None] * address_score)
        return logits, binding_debug

    def _posterior_binding_role_bias(self, obs: PicfObservationAnchorState) -> torch.Tensor | None:
        if obs.role_ids is None or obs.role_ids.numel() == 0:
            return None
        obs_roles = obs.role_ids.to(device=self.device, dtype=torch.long)
        posterior_roles = self._posterior_role_ids()
        k = int(posterior_roles.numel())
        incompatible = posterior_roles[:, None] != obs_roles[None, :]
        if not bool(incompatible.any().item()):
            return None
        return torch.zeros((k, int(obs_roles.numel())), device=self.device, dtype=self.dtype).masked_fill(incompatible, -1.0e4)

    def _posterior_owner_active_binding_bias(self, obs: PicfObservationAnchorState) -> torch.Tensor | None:
        if (
            not bool(getattr(self.config, "posterior_owner_active_gate_enabled", True))
            or obs.owner_active is None
            or obs.owner_active.numel() != obs.tokens.shape[0]
            or obs.tokens.shape[0] == 0
        ):
            return None
        owner_score = torch.nan_to_num(
            obs.owner_active.to(device=self.device, dtype=self.dtype).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        owner_score = torch.clamp(owner_score, min=0.0, max=1.0)
        threshold = min(max(float(getattr(self.config, "posterior_owner_active_min", 0.25)), 0.0), 1.0)
        eligible = owner_score >= threshold
        if obs.role_ids is not None and obs.role_ids.numel() == obs.tokens.shape[0]:
            roles = obs.role_ids.to(device=self.device, dtype=torch.long)
            for role_value in torch.unique(roles).tolist():
                rows = torch.nonzero(roles == int(role_value), as_tuple=False).squeeze(-1)
                if rows.numel() == 0 or bool(eligible.index_select(0, rows).any().item()):
                    continue
                row = int(rows[int(torch.argmax(owner_score.index_select(0, rows)).item())].item())
                eligible[row] = True
        if not bool(eligible.any().item()):
            return None
        penalty = float(getattr(self.config, "posterior_owner_active_bias", -1.0e4))
        bias = torch.zeros_like(owner_score).masked_fill(~eligible, penalty)
        return bias[None, :].expand(int(self._posterior_role_ids().numel()), -1)

    def _posterior_vl_binding_bias(
        self,
        obs: PicfObservationAnchorState,
        vl_grounding: PicfVLGroundingState | None,
    ) -> torch.Tensor | None:
        if (
            vl_grounding is None
            or not bool(vl_grounding.valid.item())
            or obs.point_weights.numel() == 0
            or obs.tokens.shape[0] == 0
        ):
            return None
        posterior_roles = self._posterior_role_ids().to(device=self.device, dtype=torch.long)
        point_count = int(obs.point_weights.shape[1])
        slot_priors, slot_valid = self._vl_slot_point_priors(
            vl_grounding,
            posterior_roles,
            point_count=point_count,
        )
        if not bool(slot_valid.any().item()):
            return None
        obs_weights = torch.clamp(obs.point_weights.to(device=self.device, dtype=self.dtype), min=0.0)
        obs_weights = obs_weights / torch.clamp(obs_weights.sum(dim=-1, keepdim=True), min=self.config.epsilon_a)
        overlap = slot_priors @ obs_weights.T
        overlap = torch.where(slot_valid[:, None], overlap, torch.zeros_like(overlap))
        return self._vl_centered_log_prior_bias(overlap)

    def _posterior_mapg_binding_bias(
        self,
        obs: PicfObservationAnchorState,
        anchor_graph: PicfAnchorPriorGraphState | None,
    ) -> torch.Tensor | None:
        if (
            anchor_graph is None
            or not bool(anchor_graph.valid.item())
            or anchor_graph.posterior_priors is None
            or obs.tokens.shape[0] == 0
        ):
            return None
        obs_roles = obs.role_ids
        if obs_roles is None or obs_roles.numel() != obs.tokens.shape[0]:
            return None
        obs_assignment = self._mapg_slot_assignment(
            anchor_graph,
            obs_roles.to(device=self.device, dtype=torch.long),
            slot_tokens=obs.tokens,
            slot_point_priors=obs.point_weights if obs.point_weights.numel() > 0 else None,
            slot_visual_priors=obs.routing_mass_visual if obs.routing_mass_visual.numel() > 0 else None,
        )
        if obs_assignment is None:
            return None
        anchor_graph.obs_slot_assignment = obs_assignment
        post_priors = anchor_graph.posterior_priors.to(device=self.device, dtype=self.dtype)
        if post_priors.numel() == 0:
            return None
        overlap = post_priors.T @ obs_assignment.T
        return self._vl_centered_log_prior_bias(overlap)

    def _posterior_occupancy_binding_bias(self, obs: PicfObservationAnchorState) -> torch.Tensor | None:
        if (
            not bool(getattr(self.config, "posterior_occupancy_prior_enabled", True))
            or obs.tokens.shape[0] == 0
            or obs.role_ids is None
            or obs.role_ids.numel() != obs.tokens.shape[0]
            or obs.x.numel() == 0
        ):
            return None
        posterior_roles = self._posterior_role_ids().to(device=self.device, dtype=torch.long)
        obs_roles = obs.role_ids.to(device=self.device, dtype=torch.long)
        bias = torch.zeros((int(posterior_roles.numel()), int(obs_roles.numel())), device=self.device, dtype=self.dtype)
        sigma = max(float(getattr(self.config, "posterior_occupancy_prior_sigma_m", 0.04)), float(self.config.epsilon_s))
        sigma2 = max(sigma * sigma, float(self.config.epsilon_s))
        for role_value in torch.unique(posterior_roles).tolist():
            slot_indices = torch.nonzero(posterior_roles == int(role_value), as_tuple=False).squeeze(-1)
            obs_indices = torch.nonzero(obs_roles == int(role_value), as_tuple=False).squeeze(-1)
            if slot_indices.numel() <= 1 or obs_indices.numel() <= 1:
                continue
            take = min(int(slot_indices.numel()), int(obs_indices.numel()))
            selected_local = _fps_indices(obs.x[obs_indices], take)
            if selected_local.numel() == 0:
                continue
            selected = obs_indices[selected_local]
            slots = slot_indices[: int(selected.numel())]
            centers = obs.x[selected].to(device=self.device, dtype=self.dtype)
            candidates = obs.x[obs_indices].to(device=self.device, dtype=self.dtype)
            dist2 = torch.sum((candidates[None, :, :] - centers[:, None, :]) ** 2, dim=-1)
            role_bias = -0.5 * dist2 / sigma2
            role_bias = role_bias - role_bias.mean(dim=-1, keepdim=True)
            clip = float(getattr(self.config, "posterior_occupancy_prior_clip", 4.0))
            if clip > 0.0:
                role_bias = torch.clamp(role_bias, min=-clip, max=clip)
            bias[slots[:, None], obs_indices[None, :]] = role_bias
        if not bool((bias != 0.0).any().item()):
            return None
        return float(getattr(self.config, "posterior_occupancy_prior_weight", 1.0)) * bias

    def _posterior_role_ids(self) -> torch.Tensor:
        k = int(self.config.persistent_anchors)
        effector_count = self._effector_persistent_count()
        return torch.cat(
            [
                torch.zeros((effector_count,), device=self.device, dtype=torch.long),
                torch.ones((max(k - effector_count, 0),), device=self.device, dtype=torch.long),
            ],
            dim=0,
        )

    def _posterior_file_active_gate(
        self,
        posterior: PicfPosteriorAnchorState,
        *,
        count: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return downstream objectness for persistent posterior files.

        File competition is the no-object decision for fixed posterior
        capacity. Inactive files remain available as reserve state, but they
        should not compete with active object files in AQR posterior reads,
        action prefixes, predictive prefixes, or evidence cache writes.
        """

        width = int(count if count is not None else posterior.tokens.shape[0])
        target_dtype = dtype or self.dtype
        active = posterior.file_competition_active
        if active is not None and active.numel() >= width:
            return torch.clamp(
                active.to(device=self.device, dtype=target_dtype).reshape(-1)[:width],
                min=0.0,
                max=1.0,
            )
        return torch.ones((width,), device=self.device, dtype=target_dtype)

    def _posterior_owner_transport_measurement(
        self,
        *,
        obs_anchors: PicfObservationAnchorState,
        anchor_graph: PicfAnchorPriorGraphState | None,
        binding_support: torch.Tensor,
        file_gate: torch.Tensor,
        lifecycle: dict[str, torch.Tensor],
        file_competition: dict[str, torch.Tensor],
        posterior_roles: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        """Transport accepted object-owner graph responsibility into posterior files.

        This is the belief-state closure missing from the earlier object-pull
        probe: graph-side object responsibility is first assigned to observation
        anchors, then to persistent posterior files, and finally exposed as a
        soft high-precision geometry measurement.  It is deliberately gated and
        role-scoped; it does not turn sidecar/proposal masks into hard labels.
        """

        slot_count = int(binding_support.shape[0])
        if (
            not bool(getattr(self.config, "posterior_owner_transport_enabled", True))
            or slot_count == 0
            or anchor_graph is None
            or not bool(anchor_graph.valid.item())
            or anchor_graph.anchor_x is None
            or anchor_graph.anchor_x.numel() == 0
            or obs_anchors.tokens.shape[0] == 0
        ):
            return None
        graph_assignment = obs_anchors.graph_assignment
        if (
            graph_assignment is None
            or graph_assignment.numel() == 0
            or graph_assignment.ndim != 2
            or int(graph_assignment.shape[0]) != int(obs_anchors.tokens.shape[0])
            or int(graph_assignment.shape[1]) != int(anchor_graph.anchor_tokens.shape[0])
        ):
            graph_assignment = anchor_graph.obs_slot_assignment
        if (
            graph_assignment is None
            or graph_assignment.numel() == 0
            or graph_assignment.ndim != 2
            or int(graph_assignment.shape[0]) != int(obs_anchors.tokens.shape[0])
            or int(graph_assignment.shape[1]) != int(anchor_graph.anchor_tokens.shape[0])
        ):
            return None

        graph_count = int(anchor_graph.anchor_tokens.shape[0])
        graph_x = anchor_graph.anchor_x.to(device=self.device, dtype=self.dtype)
        if graph_x.shape[0] < graph_count:
            return None
        graph_x = graph_x[:graph_count, :3]
        if anchor_graph.anchor_S is not None and anchor_graph.anchor_S.numel() > 0 and anchor_graph.anchor_S.shape[0] >= graph_count:
            graph_S = anchor_graph.anchor_S.to(device=self.device, dtype=self.dtype)[:graph_count]
        else:
            graph_S = torch.eye(3, device=self.device, dtype=self.dtype)[None, :, :].expand(graph_count, -1, -1).clone()

        row_strength = torch.zeros((graph_count,), device=self.device, dtype=self.dtype)

        def _add_row_strength(value: torch.Tensor | None, *, reduce: str, weight: float = 1.0) -> None:
            nonlocal row_strength
            if value is None or value.numel() == 0 or float(weight) == 0.0:
                return
            tensor = torch.clamp(value.to(device=self.device, dtype=self.dtype), min=0.0)
            if tensor.shape[0] != graph_count:
                return
            if tensor.ndim == 1:
                mass = tensor
            elif reduce == "max":
                mass = tensor.max(dim=-1).values
            else:
                mass = tensor.sum(dim=-1)
            row_strength = torch.maximum(row_strength, torch.clamp(float(weight) * mass, min=0.0, max=1.0))

        _add_row_strength(anchor_graph.object_candidate_owner_assignment, reduce="sum", weight=1.0)
        _add_row_strength(anchor_graph.object_candidate_owner_point_priors, reduce="sum", weight=1.0)
        _add_row_strength(anchor_graph.proposal_anchor_seed_assignment, reduce="sum", weight=0.5)
        _add_row_strength(anchor_graph.task_owner_point_priors, reduce="sum", weight=0.5)
        if not bool((row_strength > self.config.epsilon_a).any().item()):
            return None

        graph_roles = anchor_graph.anchor_roles.to(device=self.device, dtype=torch.long).reshape(-1)[:graph_count]
        owner_roles = tuple(int(role) for role in getattr(self.config, "posterior_owner_transport_roles", (1,)))
        graph_role_mask = torch.zeros((graph_count,), device=self.device, dtype=torch.bool)
        for role in owner_roles:
            graph_role_mask = graph_role_mask | (graph_roles == int(role))
        if anchor_graph.anchor_active is not None and anchor_graph.anchor_active.numel() >= graph_count:
            graph_role_mask = graph_role_mask & (
                anchor_graph.anchor_active.to(device=self.device, dtype=self.dtype).reshape(-1)[:graph_count] > 0.0
            )
        row_strength = torch.where(graph_role_mask, row_strength, torch.zeros_like(row_strength))
        if not bool((row_strength > self.config.epsilon_a).any().item()):
            return None

        obs_graph = torch.clamp(graph_assignment.to(device=self.device, dtype=self.dtype), min=0.0)
        obs_owner_weight = obs_graph * row_strength[None, :]
        obs_owner_mass = obs_owner_weight.sum(dim=-1)
        obs_denom = torch.clamp(obs_owner_mass[:, None], min=self.config.epsilon_a)
        obs_owner_x = (obs_owner_weight @ graph_x) / obs_denom
        graph_centered = graph_x[None, :, :] - obs_owner_x[:, None, :]
        graph_second = graph_S[None, :, :, :] + (graph_centered[..., :, None] * graph_centered[..., None, :])
        obs_owner_S = torch.einsum("og,ogab->oab", obs_owner_weight, graph_second) / torch.clamp(
            obs_owner_mass[:, None, None],
            min=self.config.epsilon_a,
        )

        post_owner_weight = torch.clamp(binding_support.to(device=self.device, dtype=self.dtype), min=0.0) * obs_owner_mass[None, :]
        post_owner_mass = post_owner_weight.sum(dim=-1)
        post_denom = torch.clamp(post_owner_mass[:, None], min=self.config.epsilon_a)
        post_owner_x = (post_owner_weight @ obs_owner_x) / post_denom
        post_centered = obs_owner_x[None, :, :] - post_owner_x[:, None, :]
        post_second = obs_owner_S[None, :, :, :] + (post_centered[..., :, None] * post_centered[..., None, :])
        post_owner_S = torch.einsum("so,soij->sij", post_owner_weight, post_second) / torch.clamp(
            post_owner_mass[:, None, None],
            min=self.config.epsilon_a,
        )

        roles = posterior_roles.to(device=self.device, dtype=torch.long).reshape(-1)
        if roles.numel() < slot_count:
            roles = fn.pad(roles, (0, slot_count - int(roles.numel())), value=1)
        roles = roles[:slot_count]
        role_gate = torch.zeros((slot_count,), device=self.device, dtype=self.dtype)
        for role in owner_roles:
            role_gate = torch.maximum(role_gate, (roles == int(role)).to(dtype=self.dtype))

        active_gate = torch.zeros((slot_count,), device=self.device, dtype=self.dtype)
        if file_gate is not None and file_gate.numel() > 0:
            n = min(slot_count, int(file_gate.numel()))
            active_gate[:n] = torch.clamp(file_gate.to(device=self.device, dtype=self.dtype).reshape(-1)[:n], min=0.0, max=1.0)

        assignment_conf = torch.zeros((slot_count,), device=self.device, dtype=self.dtype)
        if lifecycle.get("assignment_confidence") is not None:
            n = min(slot_count, int(lifecycle["assignment_confidence"].numel()))
            assignment_conf[:n] = torch.clamp(
                lifecycle["assignment_confidence"].to(device=self.device, dtype=self.dtype).reshape(-1)[:n],
                min=0.0,
                max=1.0,
            )
        owner_rel = torch.zeros((slot_count,), device=self.device, dtype=self.dtype)
        if lifecycle.get("owner_reliability") is not None:
            n = min(slot_count, int(lifecycle["owner_reliability"].numel()))
            owner_rel[:n] = torch.clamp(
                lifecycle["owner_reliability"].to(device=self.device, dtype=self.dtype).reshape(-1)[:n],
                min=0.0,
                max=1.0,
            )
        demoted = torch.zeros((slot_count,), device=self.device, dtype=self.dtype)
        demoted_value = file_competition.get("demoted_mass")
        if demoted_value is not None and demoted_value.numel() > 0:
            n = min(slot_count, int(demoted_value.numel()))
            demoted[:n] = torch.clamp(demoted_value.to(device=self.device, dtype=self.dtype).reshape(-1)[:n], min=0.0, max=1.0)

        min_mass = max(float(getattr(self.config, "posterior_owner_transport_min_mass", 0.01)), 0.0)
        assignment_floor = min(max(float(getattr(self.config, "posterior_owner_transport_assignment_floor", 0.50)), 0.0), 1.0)
        reliability_floor = min(max(float(getattr(self.config, "posterior_owner_transport_reliability_floor", 0.50)), 0.0), 1.0)
        max_rate = min(max(float(getattr(self.config, "posterior_owner_transport_max_rate", 0.85)), 0.0), 1.0)
        confidence = torch.clamp(post_owner_mass, min=0.0, max=1.0)
        inactive_prior = min(
            max(float(getattr(self.config, "posterior_owner_transport_inactive_prior", 0.35)), 0.0),
            1.0,
        )
        activity_prior = inactive_prior + ((1.0 - inactive_prior) * torch.clamp(active_gate, min=0.0, max=1.0))
        confidence = confidence * role_gate * activity_prior * (1.0 - demoted)
        confidence = confidence * (assignment_floor + ((1.0 - assignment_floor) * assignment_conf))
        confidence = confidence * (reliability_floor + ((1.0 - reliability_floor) * owner_rel))
        confidence = torch.where(post_owner_mass >= min_mass, confidence, torch.zeros_like(confidence))

        max_per_role = max(int(getattr(self.config, "posterior_owner_transport_max_per_role", 1)), 0)
        if max_per_role > 0 and confidence.numel() > 0:
            keep = torch.zeros_like(confidence, dtype=torch.bool)
            for role_value in torch.unique(roles, sorted=True).tolist():
                if int(role_value) not in owner_roles:
                    continue
                rows = torch.nonzero((roles == int(role_value)) & (confidence > 0.0), as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                take = min(max_per_role, int(rows.numel()))
                local = confidence.index_select(0, rows)
                top = torch.topk(local, k=take, dim=0).indices
                keep.index_fill_(0, rows.index_select(0, top), True)
            confidence = torch.where(keep, confidence, torch.zeros_like(confidence))
        confidence = torch.clamp(max_rate * confidence, min=0.0, max=max_rate)
        if not bool((confidence > self.config.epsilon_a).any().item()):
            return None
        cov_scale = max(float(getattr(self.config, "posterior_owner_transport_covariance_scale", 0.50)), 0.0)
        post_owner_S = (cov_scale * post_owner_S) + (
            torch.eye(3, device=self.device, dtype=self.dtype)[None, :, :] * self.config.epsilon_s
        )
        return {
            "x": post_owner_x,
            "S": post_owner_S,
            "confidence": confidence,
            "mass": post_owner_mass,
            "obs_mass": obs_owner_mass,
        }

    def _cap_file_gate_by_role(
        self,
        gate: torch.Tensor,
        *,
        role_ids: torch.Tensor,
        score: torch.Tensor | None = None,
        max_per_role: int | None = None,
    ) -> torch.Tensor:
        if gate.numel() == 0:
            return gate
        cap = int(getattr(self.config, "posterior_file_competition_max_per_role", 0)) if max_per_role is None else int(max_per_role)
        if cap <= 0:
            return torch.clamp(gate, min=0.0, max=1.0)
        gate_t = torch.clamp(gate.to(device=self.device, dtype=self.dtype).reshape(-1), min=0.0, max=1.0)
        roles = role_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if roles.numel() < gate_t.numel():
            roles = fn.pad(roles, (0, gate_t.numel() - int(roles.numel())), value=1)
        roles = roles[: gate_t.numel()]
        if score is None or score.numel() == 0:
            score_t = gate_t
        else:
            score_t = score.to(device=self.device, dtype=self.dtype).reshape(-1)
            if score_t.numel() < gate_t.numel():
                score_t = fn.pad(score_t, (0, gate_t.numel() - int(score_t.numel())), value=0.0)
            score_t = score_t[: gate_t.numel()]
        keep = torch.zeros_like(gate_t, dtype=torch.bool)
        for role_value in torch.unique(roles, sorted=True).tolist():
            rows = torch.nonzero((roles == int(role_value)) & (gate_t > self.config.epsilon_a), as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            take = min(cap, int(rows.numel()))
            local_score = score_t.index_select(0, rows)
            top = torch.topk(local_score, k=take, dim=0).indices
            keep.index_fill_(0, rows.index_select(0, top), True)
        return torch.where(keep, gate_t, torch.zeros_like(gate_t))

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
        previous: PicfPreviousState | None,
        observation: PicfObservation,
        obs_anchors: PicfObservationAnchorState,
        dense_memory: _StepDenseMemory | None = None,
        vl_grounding: PicfVLGroundingState | None = None,
        anchor_graph: PicfAnchorPriorGraphState | None = None,
        innovation_norm: torch.Tensor | None = None,
    ) -> PicfPosteriorAnchorState:
        if dense_memory is None:
            dense_memory = _StepDenseMemory(
                point_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                visual_payload=torch.zeros((0, 0), device=self.device, dtype=self.dtype),
                tactile_group_tokens=(),
            )
        h_prior, c_prior, mu_prior, var_prior, x_prior, S_prior, a_prior, alpha_prior = self._current_prior(previous, observation)
        if previous is None:
            x_prior, S_prior, a_prior = self._bootstrap_prior_geometry_from_observation(x_prior, S_prior, a_prior, obs_anchors)
        identity_innovation_norm = self._measurement_innovation_norm(x_prior, S_prior, obs_anchors)
        bind_logits, binding_debug = self._binding_logits(
            h_prior,
            x_prior,
            S_prior,
            obs_anchors,
            previous=previous,
            innovation_norm=identity_innovation_norm,
        )
        role_bias = self._posterior_binding_role_bias(obs_anchors)
        if role_bias is not None:
            bind_logits = bind_logits + role_bias
        owner_bias = self._posterior_owner_active_binding_bias(obs_anchors)
        if owner_bias is not None:
            bind_logits = bind_logits + owner_bias
        vl_bias = self._posterior_vl_binding_bias(obs_anchors, vl_grounding)
        if vl_bias is not None:
            bind_logits = bind_logits + (self._vl_gate(self.vl_posterior_bind_gate_logit, vl_grounding) * vl_bias)
        graph_bias = self._posterior_mapg_binding_bias(obs_anchors, anchor_graph)
        if graph_bias is not None:
            bind_logits = bind_logits + (self._mapg_gate(self.mapg_posterior_gate_logit, anchor_graph) * graph_bias)
        occupancy_bias = self._posterior_occupancy_binding_bias(obs_anchors)
        if occupancy_bias is not None:
            bind_logits = bind_logits + occupancy_bias
        binding_raw = self._sinkhorn_dustbin(bind_logits)
        support_raw = binding_raw[:-1]
        dustbin_raw_all = binding_raw[-1]
        posterior_roles = self._posterior_role_ids()
        file_competition = self._posterior_file_competition(
            support_raw,
            dustbin_raw_all,
            x_prior=x_prior,
            role_ids=posterior_roles,
            alpha_prior=alpha_prior,
        )
        support_raw = file_competition["support_raw"]
        dustbin_raw_all = file_competition["dustbin_raw_all"]
        lifecycle = self._posterior_lifecycle_calibration(
            support_raw,
            dustbin_raw_all,
            obs_anchors.owner_active
            if (
                bool(getattr(self.config, "posterior_owner_active_gate_enabled", True))
                and obs_anchors.owner_active is not None
                and obs_anchors.owner_active.numel() == obs_anchors.tokens.shape[0]
            )
            else None,
            alpha_prior,
            identity_innovation_norm,
        )
        dustbin_raw = lifecycle["dustbin_for_recycle"]
        support_mass_raw = support_raw.sum(dim=1)
        residual_summary = (
            torch.sum(dustbin_raw[:, None] * obs_anchors.tokens, dim=0)
            if obs_anchors.tokens.shape[0] > 0
            else torch.zeros((self.config.hidden_dim,), device=self.device, dtype=self.dtype)
        )
        if bool(getattr(self.config, "posterior_slotwise_recycle_residual", True)) and obs_anchors.tokens.shape[0] > 0:
            raw_denom = torch.clamp(support_mass_raw[:, None], min=self.config.epsilon_a)
            raw_cond = support_raw / raw_denom
            slot_residual_summary = raw_cond @ obs_anchors.tokens
            has_slot_support = support_mass_raw > self.config.epsilon_a
            slot_residual_summary = torch.where(
                has_slot_support[:, None],
                slot_residual_summary,
                residual_summary[None, :].expand_as(slot_residual_summary),
            )
        else:
            slot_residual_summary = residual_summary[None, :].expand(h_prior.shape[0], -1)
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
        recycle_residual_summary = slot_residual_summary
        if bool(getattr(self.config, "recycle_normalize_residual_summary", True)):
            # Recycle is a trust/reset probability; it should depend on the
            # residual evidence direction and context, not the unbounded norm of
            # the aggregated residual. Use the slot-local measurement summary so
            # multiple same-role object files cannot all reset from one global
            # dustbin vector.
            recycle_norm_mode = str(getattr(self.config, "recycle_residual_norm_mode", "layernorm")).lower()
            if recycle_norm_mode in ("layernorm", "layer_norm"):
                recycle_residual_summary = fn.layer_norm(
                    slot_residual_summary,
                    normalized_shape=(int(slot_residual_summary.shape[-1]),),
                )
            elif recycle_norm_mode in ("rmsnorm", "rms_norm"):
                rms = torch.sqrt(
                    torch.mean(slot_residual_summary * slot_residual_summary, dim=-1, keepdim=True)
                    + self.config.epsilon_residual
                )
                recycle_residual_summary = slot_residual_summary / torch.clamp(rms, min=self.config.epsilon_residual)
            elif recycle_norm_mode in ("none", "off", "identity"):
                recycle_residual_summary = slot_residual_summary
            else:
                raise ValueError(f"Unsupported recycle_residual_norm_mode={recycle_norm_mode!r}")
        recycle_in = torch.cat(
            [
                h_prior,
                support_mass_raw[:, None],
                var_prior.mean(dim=-1, keepdim=True),
                recycle_residual_summary,
                alpha_prior[:, None],
            ],
            dim=-1,
        )
        recycle_logits = self.recycle_head(recycle_in).squeeze(-1)
        recycle_logit_clamp = float(getattr(self.config, "recycle_logit_clamp", 0.0))
        if recycle_logit_clamp > 0.0:
            recycle_logits = torch.clamp(recycle_logits, min=-recycle_logit_clamp, max=recycle_logit_clamp)
        recycle_raw = torch.sigmoid(recycle_logits)
        recycle = recycle_raw * lifecycle["reset_allowance"].to(device=self.device, dtype=self.dtype)
        file_active_initial = torch.clamp(
            file_competition["active"].to(device=self.device, dtype=self.dtype).reshape(-1),
            min=0.0,
            max=1.0,
        )
        if file_active_initial.numel() < recycle.numel():
            file_active_initial = fn.pad(file_active_initial, (0, recycle.numel() - int(file_active_initial.numel())), value=0.0)
        file_active_initial = file_active_initial[: recycle.numel()]
        birth_active = self._posterior_birth_competition(
            recycle,
            file_active=file_active_initial,
            role_ids=posterior_roles,
            alpha_prior=alpha_prior,
        )
        owner_file_gate = torch.clamp(torch.maximum(file_active_initial, birth_active), min=0.0, max=1.0)
        birth_recycle = recycle * birth_active
        birth_share = birth_recycle / torch.clamp(1.0 + birth_recycle.sum(), min=self.config.epsilon_a)
        recycle_update_mask = torch.clamp((support_mass_raw > self.config.epsilon_a).to(dtype=self.dtype) + birth_active, min=0.0, max=1.0)
        recycle_update = recycle * recycle_update_mask
        binding_support = support_raw + (birth_share[:, None] * dustbin_raw[None, :])
        dustbin_final = dustbin_raw / torch.clamp(1.0 + birth_recycle.sum(), min=self.config.epsilon_a)
        binding = torch.cat([binding_support, dustbin_final[None, :]], dim=0)
        support_mass = binding_support.sum(dim=1)
        if obs_anchors.tokens.shape[0] > 0:
            binding_cond = binding_support / torch.clamp(support_mass[:, None], min=self.config.epsilon_a)
            measurement_summary = binding_cond @ obs_anchors.tokens
            measurement_summary = torch.where(
                (support_mass > self.config.epsilon_a)[:, None],
                measurement_summary,
                torch.zeros_like(measurement_summary),
            )
        else:
            binding_cond = torch.zeros_like(binding_support)
            measurement_summary = torch.zeros_like(slot_residual_summary)
        res_mu = self.residual_mu_head(measurement_summary)
        res_var = _variance_from_logvar(
            self.residual_logvar_head(measurement_summary),
            min_var=self.config.sigma_min2,
            max_var=self.config.sigma_max2,
        )
        res_h = self.residual_h_head(measurement_summary)
        res_c = self.residual_c_head(measurement_summary)
        bar_h = (1.0 - recycle_update[:, None]) * h_prior + recycle_update[:, None] * res_h
        bar_c = (1.0 - recycle_update[:, None]) * c_prior + recycle_update[:, None] * res_c
        bar_mu = (1.0 - recycle_update[:, None]) * mu_prior + recycle_update[:, None] * res_mu
        bar_var = (1.0 - recycle_update[:, None]) * var_prior + recycle_update[:, None] * res_var
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
        slot_token = self.posterior_slot_token.to(device=self.device, dtype=self.dtype)
        query = (self.anchor_seed_proj(anchor_seed) + slot_token)[None, :]
        bias = None
        if obs_anchors.tokens.shape[0] > 0:
            bias = self.config.lambda_bind_prior * torch.log(torch.clamp(binding[:-1], min=self.config.epsilon_a))
        evidence_tokens, _ = self.anchor_reader(query, obs_anchors.tokens[None, :], attn_bias=bias)
        evidence_tokens = evidence_tokens[0]
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
            tactile_active_rows = tactile_weights.sum(dim=-1) > self.config.epsilon_a
            if bool(tactile_active_rows.any().item()):
                tactile_candidates, tactile_bias = self._gather_tactile_group_candidates(
                    dense_memory.tactile_group_tokens,
                    tactile_weights[tactile_active_rows],
                    top_groups=self.config.tactile_reread_groups,
                )
                tactile_read, _ = self.tactile_native_reread(
                    evidence_tokens[tactile_active_rows, None, :],
                    tactile_candidates,
                    attn_bias=tactile_bias,
                )
                tactile_evidence[tactile_active_rows] = tactile_read[:, 0, :]
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
        owner_transport_mass = torch.zeros((x.shape[0],), device=self.device, dtype=self.dtype)
        owner_transport_confidence = torch.zeros_like(owner_transport_mass)
        owner_transport_applied_fraction = torch.zeros((), device=self.device, dtype=self.dtype)
        owner_transport_dist_to_standard = torch.zeros_like(owner_transport_mass)
        owner_transport = self._posterior_owner_transport_measurement(
            obs_anchors=obs_anchors,
            anchor_graph=anchor_graph,
            binding_support=binding[:-1],
            file_gate=owner_file_gate,
            lifecycle=lifecycle,
            file_competition=file_competition,
            posterior_roles=posterior_roles,
        )
        if owner_transport is not None:
            owner_x = owner_transport["x"].to(device=self.device, dtype=self.dtype)
            owner_S = owner_transport["S"].to(device=self.device, dtype=self.dtype)
            owner_conf = torch.clamp(
                owner_transport["confidence"].to(device=self.device, dtype=self.dtype).reshape(-1)[: x.shape[0]],
                min=0.0,
                max=1.0,
            )
            if owner_conf.numel() < x.shape[0]:
                owner_conf = fn.pad(owner_conf, (0, x.shape[0] - int(owner_conf.numel())), value=0.0)
            owner_transport_mass = torch.clamp(
                owner_transport["mass"].to(device=self.device, dtype=self.dtype).reshape(-1)[: x.shape[0]],
                min=0.0,
            )
            if owner_transport_mass.numel() < x.shape[0]:
                owner_transport_mass = fn.pad(owner_transport_mass, (0, x.shape[0] - int(owner_transport_mass.numel())), value=0.0)
            owner_x_aligned = x.clone()
            owner_S_aligned = S.clone()
            n_owner = min(int(owner_x.shape[0]), int(x.shape[0]))
            if n_owner > 0:
                owner_x_aligned[:n_owner] = owner_x[:n_owner, :3]
            n_owner_S = min(int(owner_S.shape[0]), int(S.shape[0]))
            if n_owner_S > 0:
                owner_S_aligned[:n_owner_S] = owner_S[:n_owner_S]
            owner_transport_dist_to_standard = torch.linalg.norm(owner_x_aligned[: x.shape[0], :3] - x[:, :3], dim=-1)

            owner_active = owner_conf > self.config.epsilon_a
            if bool(owner_active.any().item()):
                eye3 = torch.eye(3, device=self.device, dtype=self.dtype)[None, :, :]
                jitter = eye3 * max(float(self.config.epsilon_s), float(self.config.sigma_min2))
                precision_gain = max(float(getattr(self.config, "posterior_owner_transport_precision_gain", 8.0)), 0.0)
                standard_precision = torch.linalg.pinv(S + jitter)
                owner_precision = torch.linalg.pinv(owner_S_aligned + jitter)
                measurement_weight = torch.clamp(precision_gain * owner_conf, min=0.0)[:, None, None]
                fused_precision = standard_precision + (measurement_weight * owner_precision)
                standard_eta = torch.matmul(standard_precision, x[:, :3, None]).squeeze(-1)
                owner_eta = torch.matmul(owner_precision, owner_x_aligned[:, :3, None]).squeeze(-1)
                fused_eta = standard_eta + (measurement_weight.squeeze(-1) * owner_eta)
                fused_S = torch.linalg.pinv(fused_precision + jitter)
                fused_x = torch.matmul(fused_S, fused_eta[:, :, None]).squeeze(-1)
                x = torch.where(owner_active[:, None], fused_x, x)
                S = torch.where(owner_active[:, None, None], fused_S, S)
                a = torch.where(owner_active[:, None], _extent_from_cov(S, self.config), a)
            owner_transport_confidence = owner_conf
            owner_transport_applied_fraction = (owner_conf > self.config.epsilon_a).to(dtype=self.dtype).mean().reshape(())
            if bool(getattr(self.config, "posterior_owner_transport_activates_file", True)):
                active_threshold = max(
                    float(getattr(self.config, "posterior_owner_transport_active_threshold", 0.05)),
                    self.config.epsilon_a,
                )
                owner_activation = torch.clamp(owner_conf / active_threshold, min=0.0, max=1.0)
                if owner_activation.numel() < owner_file_gate.numel():
                    owner_activation = fn.pad(
                        owner_activation,
                        (0, owner_file_gate.numel() - int(owner_activation.numel())),
                        value=0.0,
                    )
                owner_activation = owner_activation[: owner_file_gate.numel()]
                owner_file_gate = torch.clamp(torch.maximum(owner_file_gate, owner_activation), min=0.0, max=1.0)
        gate_width = int(owner_file_gate.numel())

        def _gate_aligned(value: torch.Tensor) -> torch.Tensor:
            aligned = value.to(device=self.device, dtype=self.dtype).reshape(-1)
            if aligned.numel() < gate_width:
                aligned = fn.pad(aligned, (0, gate_width - int(aligned.numel())), value=0.0)
            return aligned[:gate_width]

        owner_conf_for_gate = torch.clamp(_gate_aligned(owner_transport_confidence), min=0.0, max=1.0)
        if bool(getattr(self.config, "posterior_owner_transport_activates_file", True)) and gate_width > 0:
            active_threshold = max(
                float(getattr(self.config, "posterior_owner_transport_active_threshold", 0.05)),
                self.config.epsilon_a,
            )
            owner_candidate = owner_conf_for_gate >= active_threshold
            if bool(owner_candidate.any().item()):
                gate_roles = posterior_roles.to(device=self.device, dtype=torch.long).reshape(-1)
                if gate_roles.numel() < gate_width:
                    gate_roles = fn.pad(gate_roles, (0, gate_width - int(gate_roles.numel())), value=1)
                gate_roles = gate_roles[:gate_width]
                owner_roles = tuple(int(role) for role in getattr(self.config, "posterior_owner_transport_roles", (1,)))
                for role in owner_roles:
                    role_rows = gate_roles == int(role)
                    if bool((role_rows & owner_candidate).any().item()):
                        owner_file_gate = torch.where(
                            role_rows & (~owner_candidate),
                            torch.zeros_like(owner_file_gate),
                            owner_file_gate,
                        )
        final_gate_score = (
            (100.0 * owner_conf_for_gate)
            + torch.clamp(_gate_aligned(file_active_initial), min=0.0, max=1.0)
            + torch.clamp(_gate_aligned(birth_active), min=0.0, max=1.0)
            + torch.clamp(_gate_aligned(support_mass), min=0.0)
        )
        owner_file_gate = self._cap_file_gate_by_role(
            owner_file_gate,
            role_ids=posterior_roles,
            score=final_gate_score,
            max_per_role=int(getattr(self.config, "posterior_file_competition_max_per_role", 0)),
        )
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
        downstream_gate = owner_file_gate
        if downstream_gate.numel() < token_in.shape[0]:
            downstream_gate = fn.pad(downstream_gate, (0, token_in.shape[0] - int(downstream_gate.numel())), value=1.0)
        downstream_gate = downstream_gate[: token_in.shape[0]]
        token_gate = downstream_gate[:, None]
        tokens = (self.posterior_token_proj(token_in) + slot_token) * token_gate
        posterior_self_bias = None
        active_key = downstream_gate >= 0.5
        if bool(active_key.any().item()) and bool((~active_key).any().item()):
            posterior_self_bias = torch.zeros((tokens.shape[0], tokens.shape[0]), device=self.device, dtype=self.dtype)
            posterior_self_bias[:, ~active_key] = -1.0e4
        tokens = self.posterior_self(tokens[None, :], attn_bias=posterior_self_bias)[0] * token_gate
        if bool(active_key.any().item()):
            query = self.posterior_pool.query.to(device=self.device, dtype=self.dtype)[None, :]
            pool_score = self.posterior_pool.score(tokens + query).squeeze(-1)
            pool_score = torch.where(active_key, pool_score, torch.full_like(pool_score, -1.0e4))
            pool_weight = torch.softmax(pool_score, dim=-1)
            global_post = torch.sum(pool_weight[:, None] * tokens, dim=0)
        else:
            global_post = self.posterior_pool(tokens[None, :])[0]
        def _posterior_signature(obs_weights: torch.Tensor | None) -> torch.Tensor | None:
            if obs_weights is None or obs_weights.numel() == 0 or obs_weights.shape[0] != binding_cond.shape[1]:
                return None
            sig = binding_cond @ obs_weights.to(device=self.device, dtype=self.dtype)
            return _normalize_rows(sig, eps=self.config.epsilon_a)

        point_signature = _posterior_signature(obs_anchors.graph_point_weights if obs_anchors.graph_point_weights is not None else obs_anchors.point_weights)
        visual_signature = _posterior_signature(obs_anchors.graph_visual_weights if obs_anchors.graph_visual_weights is not None else obs_anchors.routing_mass_visual)
        temporal_signature = _posterior_signature(obs_anchors.graph_temporal_weights)
        pg_signature = _posterior_signature(obs_anchors.graph_pg_weights)
        tactile_signature = _posterior_signature(obs_anchors.graph_tactile_weights if obs_anchors.graph_tactile_weights is not None else obs_anchors.routing_mass_tactile)
        tracklet_signature = _posterior_signature(obs_anchors.graph_tracklet_weights)
        proposal_signature = _posterior_signature(obs_anchors.graph_proposal_weights)
        signature_parts = [
            sig.mean(dim=-1, keepdim=True)
            for sig in (
                point_signature,
                visual_signature,
                temporal_signature,
                pg_signature,
                tactile_signature,
                tracklet_signature,
                proposal_signature,
            )
            if sig is not None and sig.numel() > 0
        ]
        support_signature = torch.cat(signature_parts, dim=-1) if signature_parts else support_mass[:, None]
        instant_binding_signature = (
            _normalize_tensor(binding_cond @ obs_anchors.binding_signature.to(device=self.device, dtype=self.dtype), eps=self.config.epsilon_residual)
            if obs_anchors.binding_signature is not None and obs_anchors.binding_signature.numel() > 0
            else self._binding_keys(tokens)
        )
        binding_signature = instant_binding_signature
        binding_signature_update_rate = torch.ones_like(support_mass)
        assignment_trust = torch.clamp(
            lifecycle["assignment_confidence"].to(device=self.device, dtype=self.dtype).reshape(-1)[: support_mass.shape[0]],
            min=0.0,
            max=1.0,
        )
        owner_reliability = torch.clamp(
            lifecycle["owner_reliability"].to(device=self.device, dtype=self.dtype).reshape(-1)[: support_mass.shape[0]],
            min=0.0,
            max=1.0,
        )
        owner_weight = min(max(float(getattr(self.config, "posterior_binding_signature_owner_weight", 0.50)), 0.0), 1.0)
        binding_signature_measurement_trust = torch.clamp(
            assignment_trust * ((1.0 - owner_weight) + (owner_weight * owner_reliability)),
            min=0.0,
            max=1.0,
        )
        binding_signature_measurement_score_std = torch.zeros((), device=self.device, dtype=self.dtype)
        binding_signature_measurement_margin = torch.zeros_like(support_mass)
        binding_signature_measurement_dispersion_gate = torch.ones_like(support_mass)
        if bool(getattr(self.config, "posterior_binding_signature_dispersion_gate_enabled", True)):
            binding_signature_measurement_dispersion_gate = torch.zeros_like(support_mass)
            if instant_binding_signature.ndim == 2 and int(instant_binding_signature.shape[0]) > 1:
                instant_norm = _normalize_tensor(instant_binding_signature, eps=self.config.epsilon_residual)
                instant_score = instant_norm @ instant_norm.T
                calibrated_instant_score = self._calibrate_pairwise_binding_score(instant_score)
                binding_signature_measurement_score_std = (
                    torch.std(calibrated_instant_score, unbiased=False).reshape(())
                    if calibrated_instant_score.numel() > 1
                    else torch.zeros((), device=self.device, dtype=self.dtype)
                )
                role_vec = posterior_roles.to(device=self.device, dtype=torch.long).reshape(-1)[: instant_norm.shape[0]]
                same_role = role_vec[:, None] == role_vec[None, :]
                same_role = same_role & ~torch.eye(int(instant_norm.shape[0]), device=self.device, dtype=torch.bool)
                self_score = torch.diagonal(calibrated_instant_score, 0)
                if bool(same_role.any().item()):
                    best_other = calibrated_instant_score.masked_fill(~same_role, -1.0).max(dim=-1).values
                else:
                    best_other = torch.full_like(self_score, -1.0)
                binding_signature_measurement_margin = self_score - best_other
                min_margin = float(getattr(self.config, "posterior_binding_signature_measurement_margin_min", 0.25))
                margin_temp = max(
                    float(getattr(self.config, "posterior_binding_signature_measurement_margin_temperature", 0.10)),
                    float(self.config.epsilon_a),
                )
                margin_gate = torch.sigmoid((binding_signature_measurement_margin - min_margin) / margin_temp)
                min_std = max(
                    float(getattr(self.config, "posterior_binding_signature_measurement_min_std", 0.05)),
                    float(self.config.epsilon_a),
                )
                std_gate = (binding_signature_measurement_score_std >= min_std).to(dtype=self.dtype)
                binding_signature_measurement_dispersion_gate = torch.clamp(margin_gate * std_gate, min=0.0, max=1.0)
            if binding_signature_measurement_dispersion_gate.numel() < binding_signature_measurement_trust.numel():
                binding_signature_measurement_dispersion_gate = fn.pad(
                    binding_signature_measurement_dispersion_gate,
                    (0, binding_signature_measurement_trust.numel() - int(binding_signature_measurement_dispersion_gate.numel())),
                    value=0.0,
                )
            binding_signature_measurement_dispersion_gate = binding_signature_measurement_dispersion_gate[
                : binding_signature_measurement_trust.numel()
            ]
            binding_signature_measurement_trust = torch.clamp(
                binding_signature_measurement_trust * binding_signature_measurement_dispersion_gate,
                min=0.0,
                max=1.0,
            )
        if (
            bool(getattr(self.config, "posterior_binding_signature_memory_enabled", True))
            and previous is not None
            and previous.posterior.binding_signature is not None
            and previous.posterior.binding_signature.numel() > 0
            and previous.posterior.binding_signature.shape == instant_binding_signature.shape
        ):
            previous_binding_signature = _normalize_tensor(
                previous.posterior.binding_signature.to(device=self.device, dtype=self.dtype),
                eps=self.config.epsilon_residual,
            )
            min_support = max(
                float(getattr(self.config, "posterior_binding_signature_min_support", 0.02)),
                float(self.config.epsilon_a),
            )
            support_gate = (support_mass >= min_support).to(dtype=self.dtype)
            stable_file_gate = torch.clamp(
                downstream_gate * (1.0 - recycle_update.clamp(0.0, 1.0)) * support_gate,
                min=0.0,
                max=1.0,
            )
            base_rate = max(float(getattr(self.config, "posterior_binding_signature_update_rate", 0.20)), 0.0)
            max_rate = max(float(getattr(self.config, "posterior_binding_signature_update_max_rate", 0.50)), 0.0)
            measured_rate = torch.clamp(
                base_rate * binding_signature_measurement_trust * stable_file_gate,
                min=0.0,
                max=max_rate,
            )
            reset_rate = torch.clamp(torch.maximum(birth_active, recycle_update), min=0.0, max=1.0)
            binding_signature_update_rate = torch.maximum(measured_rate, reset_rate)
            binding_signature_update_rate = torch.where(
                (support_gate > 0.0) | (reset_rate > self.config.epsilon_a),
                binding_signature_update_rate,
                torch.zeros_like(binding_signature_update_rate),
            )
            binding_signature = _normalize_tensor(
                ((1.0 - binding_signature_update_rate[:, None]) * previous_binding_signature)
                + (binding_signature_update_rate[:, None] * instant_binding_signature),
                eps=self.config.epsilon_residual,
            )
        binding_signature_memory_keep_rate = 1.0 - torch.clamp(binding_signature_update_rate, min=0.0, max=1.0)
        base_address = (
            previous.posterior.slot_address.to(device=self.device, dtype=self.dtype)
            if previous is not None and previous.posterior.slot_address is not None
            else slot_token
        )
        address_update_rate = torch.zeros_like(support_mass)
        identity_innovation_risk = self._innovation_risk_scalar(identity_innovation_norm)
        if obs_anchors.anchor_address is not None and obs_anchors.anchor_address.numel() > 0:
            obs_address = binding_cond @ obs_anchors.anchor_address.to(device=self.device, dtype=self.dtype)
            rate = float(self.config.address_update_rate) * support_mass.clamp(0.0, 1.0) * (1.0 - recycle_update.clamp(0.0, 1.0))
            rate = rate * torch.exp(
                -float(self.config.bind_address_innovation_downweight) * identity_innovation_risk
            )
            rate = torch.clamp(rate, min=0.0, max=float(self.config.address_update_max_rate))
            address_update_rate = rate
            slot_address = _normalize_tensor(((1.0 - rate[:, None]) * base_address) + (rate[:, None] * obs_address), eps=self.config.epsilon_residual)
        else:
            slot_address = base_address
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
            recycle_gate=recycle_update,
            binding=binding,
            evidence_tokens=evidence_tokens,
            tokens=tokens,
            global_post=global_post,
            role_ids=self._posterior_role_ids(),
            slot_address=slot_address,
            slot_content=tokens,
            visual_signature=visual_signature,
            temporal_signature=temporal_signature,
            point_signature=point_signature,
            pg_signature=pg_signature,
            tactile_signature=tactile_signature,
            tracklet_signature=tracklet_signature,
            proposal_signature=proposal_signature,
            support_signature=support_signature,
            binding_signature=binding_signature,
            binding_signature_linear_score_mean=binding_debug.get("binding_signature_linear_score_mean"),
            binding_signature_linear_score_abs_mean=binding_debug.get("binding_signature_linear_score_abs_mean"),
            binding_signature_quadratic_score_mean=binding_debug.get("binding_signature_quadratic_score_mean"),
            binding_signature_quadratic_score_abs_mean=binding_debug.get("binding_signature_quadratic_score_abs_mean"),
            binding_signature_low_rank_score_mean=binding_debug.get("binding_signature_low_rank_score_mean"),
            binding_signature_low_rank_score_abs_mean=binding_debug.get("binding_signature_low_rank_score_abs_mean"),
            binding_signature_combined_score_mean=binding_debug.get("binding_signature_combined_score_mean"),
            binding_signature_combined_score_abs_mean=binding_debug.get("binding_signature_combined_score_abs_mean"),
            binding_signature_calibrated_score_mean=binding_debug.get("binding_signature_calibrated_score_mean"),
            binding_signature_calibrated_score_abs_mean=binding_debug.get("binding_signature_calibrated_score_abs_mean"),
            binding_signature_calibrated_score_std=binding_debug.get("binding_signature_calibrated_score_std"),
            binding_signature_calibrated_top1_margin_mean=binding_debug.get(
                "binding_signature_calibrated_top1_margin_mean"
            ),
            binding_signature_gate_mean=binding_debug.get("binding_signature_gate_mean"),
            binding_signature_update_rate=binding_signature_update_rate,
            binding_signature_measurement_trust=binding_signature_measurement_trust,
            binding_signature_memory_keep_rate=binding_signature_memory_keep_rate,
            binding_signature_measurement_score_std=binding_signature_measurement_score_std,
            binding_signature_measurement_margin=binding_signature_measurement_margin,
            binding_signature_measurement_dispersion_gate=binding_signature_measurement_dispersion_gate,
            recycle_logits=recycle_logits,
            recycle_support_mass_raw=support_mass_raw,
            recycle_prior_var_mean=var_prior.mean(dim=-1),
            recycle_prior_alpha=alpha_prior,
            recycle_residual_summary_norm=torch.linalg.norm(residual_summary).reshape(()),
            recycle_dustbin_raw_mass=dustbin_raw.sum().reshape(()),
            recycle_dustbin_final_mass=dustbin_final.sum().reshape(()),
            lifecycle_assignment_confidence=lifecycle["assignment_confidence"],
            lifecycle_support_entropy=lifecycle["support_entropy"],
            lifecycle_support_margin=lifecycle["support_margin"],
            lifecycle_owner_reliability=lifecycle["owner_reliability"],
            lifecycle_survival_prob=lifecycle["survival_prob"],
            lifecycle_reset_allowance=lifecycle["reset_allowance"],
            lifecycle_recycle_raw=recycle_raw,
            lifecycle_inactive_dustbin_mass=lifecycle["inactive_dustbin_mass"],
            lifecycle_unexplained_dustbin_mass=lifecycle["unexplained_dustbin_mass"],
            file_competition_active=downstream_gate,
            file_competition_demoted_mass=file_competition["demoted_mass"],
            file_competition_duplicate_overlap_max=file_competition["duplicate_overlap_max"],
            file_competition_active_duplicate_overlap_max=file_competition["active_duplicate_overlap_max"],
            file_competition_birth_active=birth_active,
            file_competition_birth_share=birth_share,
            identity_innovation_risk=identity_innovation_risk.reshape(()),
            address_update_rate=address_update_rate,
            owner_transport_mass=owner_transport_mass,
            owner_transport_confidence=owner_transport_confidence,
            owner_transport_applied_fraction=owner_transport_applied_fraction,
            owner_transport_dist_to_standard=owner_transport_dist_to_standard,
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
            points = _to_tensor(_frame_context_points_world(frame_context), device=self.device, dtype=self.dtype)
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
        if visual_map_override is None and self.clip_buffers:
            clip_snapshot = {name: buffer.snapshot() for name, buffer in self.clip_buffers.items()}
        try:
            visual_map, temporal_visual_maps = self._visual_maps(observation, visual_map_override, meta)
        finally:
            if clip_snapshot is not None:
                for name, snapshot in clip_snapshot.items():
                    buffer = self.clip_buffers.get(name)
                    if snapshot is not None and buffer is not None:
                        buffer.restore(snapshot)
        tactile_bundle = self._tactile_features(observation, meta)
        point_payload = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        visual_payload = (
            visual_map.reshape(-1, visual_map.shape[-1])
            if visual_map is not None and visual_map.numel() > 0
            else torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        )
        if frame_context is not None:
            point_positions_world = _to_tensor(_frame_context_points_world(frame_context), device=self.device, dtype=self.dtype)
            projective_geometry = self._build_projective_geometry(
                observation=observation,
                point_positions=point_positions_world,
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
                self.tactile_patch_token_proj(sensor.tokens.to(device=self.device, dtype=self.dtype).clone())
                + self.modality_embedding.weight[2][None, :]
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
        previous: PicfPreviousState | None,
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
        previous: PicfPreviousState | None = None,
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
        graph_can_run_without_points = bool(self.config.mapg_enabled) or bool(self.config.aqr_mapg_enabled)
        if previous is None and not meta.point_contract_ok and not graph_can_run_without_points:
            raise RuntimeError("PICF core requires a valid xyzrgb point cloud on the first control step.")
        local_frame_context = self._point_subset(observation) if meta.point_contract_ok else None
        if (
            previous is None
            and local_frame_context is not None
            and local_frame_context.points_local.shape[0] == 0
            and not graph_can_run_without_points
        ):
            raise RuntimeError("PICF core requires non-empty local xyzrgb support on the first control step.")
        point_context = (
            self._point_context_with_global_scene(observation, local_frame_context)
            if local_frame_context is not None and point_features_override is None
            else local_frame_context
        )
        point_features = self._extract_point_features(point_context, point_features_override) if point_context is not None else torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        visual_map, temporal_visual_maps = self._visual_maps(observation, visual_map_override, meta)
        tactile_bundle = self._tactile_features(observation, meta)
        semantic = self._semantic_context(observation, previous, semantic_override)
        token_field, dense_memory = self._build_token_field(
            observation,
            point_context,
            point_features,
            visual_map,
            temporal_visual_maps,
            tactile_bundle,
            meta,
            previous,
        )
        proprio = _to_tensor(
            np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1),
            device=self.device,
            dtype=self.dtype,
        )
        proprio_token = self.proprio_proj(proprio[None, :])[0]
        vl_grounding = self._build_vl_grounding(semantic=semantic, token_field=token_field)
        if bool(self.config.aqr_mapg_enabled):
            anchor_prior_graph = self._build_aqr_anchor_graph(
                semantic=semantic,
                token_field=token_field,
                previous=previous,
                vl_grounding=vl_grounding,
                proprio_token=proprio_token,
            )
        else:
            anchor_prior_graph = self._build_anchor_prior_graph(
                semantic=semantic,
                token_field=token_field,
                dense_memory=dense_memory,
                previous=previous,
                vl_grounding=vl_grounding,
            )
        object_explanation = self._build_object_explanation_measurements(token_field, anchor_prior_graph)
        observation_anchors = self._build_observation_anchors(
            token_field,
            dense_memory,
            vl_grounding=vl_grounding,
            anchor_graph=anchor_prior_graph,
        )
        current_targets, availability = self._current_targets(observation, local_frame_context, visual_map, dense_memory)
        innovation_token, innovation_norm = self._innovation(previous, current_targets, availability)
        posterior = self._posterior_update(
            previous,
            observation,
            observation_anchors,
            dense_memory,
            vl_grounding=vl_grounding,
            anchor_graph=anchor_prior_graph,
            innovation_norm=innovation_norm,
        )
        task_readout = self._build_task_readout(
            token_field,
            dense_memory,
            semantic,
            proprio_token,
            prompt=observation.prompt,
            vl_grounding=vl_grounding,
            anchor_graph=anchor_prior_graph,
        )
        conditioned_control = self._build_conditioned_control_state(
            posterior,
            innovation_token,
            proprio_token,
            task_readout,
            anchor_graph=anchor_prior_graph,
        )
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
            vl_grounding=vl_grounding,
            anchor_prior_graph=anchor_prior_graph,
            object_explanation=object_explanation,
            proprio_token=proprio_token,
            task_readout=task_readout,
            conditioned_control=conditioned_control,
            control=PicfControlState(hold_reason=hold_reason),
            last_prompt=observation.prompt,
            previous=previous,
        )

    def make_recurrent_carry(self, state: PicfCoreState) -> PicfRecurrentCarryState:
        return PicfRecurrentCarryState(
            runtime_meta=state.runtime_meta,
            token_field=PicfRecurrentTokenFieldState(
                tactile_contact_gate=state.token_field.tactile_contact_gate,
                tactile_anchor_mask=state.token_field.tactile_anchor_mask,
                tactile_contact_score_ema=state.token_field.tactile_contact_score_ema,
            ),
            posterior=state.posterior,
            predictive=PicfRecurrentPredictiveState(
                executed_action=state.predictive.executed_action,
                physical_prediction_cache=state.predictive.physical_prediction_cache,
                evidence_cache=getattr(state.predictive, "evidence_cache", None),
            ),
        )

    def recurrent_burnin_step(
        self,
        observation: PicfObservation,
        previous: PicfPreviousState | None = None,
        *,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        action_future: torch.Tensor | np.ndarray | None = None,
    ) -> PicfRecurrentCarryState:
        """Advance only the canonical recurrent carry for suffix-gradient burn-in.

        This deliberately skips semantic task readout, conditioned control,
        PI0.5 action flow loss, and conditioned future cache construction. Those
        objects are current-step control views and are not retained by
        `make_recurrent_carry(...)`; the recurrent carry is posterior +
        tactile-contact carry + world-only physical predictive cache.
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
        if observation.reset_scaffold:
            previous = None
        meta = self._build_runtime_meta(observation, previous.runtime_meta if previous is not None else None)
        graph_can_run_without_points = bool(self.config.mapg_enabled) or bool(self.config.aqr_mapg_enabled)
        if previous is None and not meta.point_contract_ok and not graph_can_run_without_points:
            raise RuntimeError("PICF core requires a valid xyzrgb point cloud on the first control step.")
        local_frame_context = self._point_subset(observation) if meta.point_contract_ok else None
        if (
            previous is None
            and local_frame_context is not None
            and local_frame_context.points_local.shape[0] == 0
            and not graph_can_run_without_points
        ):
            raise RuntimeError("PICF core requires non-empty local xyzrgb support on the first control step.")
        point_context = (
            self._point_context_with_global_scene(observation, local_frame_context)
            if local_frame_context is not None and point_features_override is None
            else local_frame_context
        )
        point_features = (
            self._extract_point_features(point_context, point_features_override)
            if point_context is not None
            else torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        )
        visual_map, temporal_visual_maps = self._visual_maps(observation, visual_map_override, meta)
        tactile_bundle = self._tactile_features(observation, meta)
        token_field, dense_memory = self._build_token_field(
            observation,
            point_context,
            point_features,
            visual_map,
            temporal_visual_maps,
            tactile_bundle,
            meta,
            previous,
        )
        empty_semantic = self._project_semantic_context(tokens_raw=torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype))
        proprio = _to_tensor(
            np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1),
            device=self.device,
            dtype=self.dtype,
        )
        proprio_token = self.proprio_proj(proprio[None, :])[0]
        # Keep state-only burn-in on the same AQR measurement model as the trainable suffix.
        if bool(self.config.aqr_mapg_enabled):
            anchor_prior_graph = self._build_aqr_anchor_graph(
                semantic=empty_semantic,
                token_field=token_field,
                previous=previous,
                vl_grounding=None,
                proprio_token=proprio_token,
            )
        else:
            anchor_prior_graph = self._build_anchor_prior_graph(
                semantic=empty_semantic,
                token_field=token_field,
                dense_memory=dense_memory,
                previous=previous,
                vl_grounding=None,
            )
        observation_anchors = self._build_observation_anchors(token_field, dense_memory, anchor_graph=anchor_prior_graph)
        current_targets, availability = self._current_targets(observation, local_frame_context, visual_map, dense_memory)
        innovation_token, innovation_norm = self._innovation(previous, current_targets, availability)
        posterior = self._posterior_update(
            previous,
            observation,
            observation_anchors,
            dense_memory,
            anchor_graph=anchor_prior_graph,
            innovation_norm=innovation_norm,
        )
        action_source = (
            action_future
            if action_future is not None
            else (observation.action_chunk if observation.action_chunk is not None else observation.action)
        )
        default_action, _ = self._default_predictive_action(action_source)
        executed_action = self._executed_action(observation, default_action)
        _, _, physical_prediction_cache = self._build_physical_predictive_basis(
            posterior,
            proprio_token=proprio_token,
            executed_action=executed_action,
        )
        evidence_cache = self._write_evidence_cache(
            previous,
            posterior,
            innovation_norm=innovation_norm,
            availability=availability,
            reset=bool(observation.reset_scaffold),
        )
        return PicfRecurrentCarryState(
            runtime_meta=meta,
            token_field=PicfRecurrentTokenFieldState(
                tactile_contact_gate=token_field.tactile_contact_gate,
                tactile_anchor_mask=token_field.tactile_anchor_mask,
                tactile_contact_score_ema=token_field.tactile_contact_score_ema,
            ),
            posterior=posterior,
            predictive=PicfRecurrentPredictiveState(
                executed_action=executed_action,
                physical_prediction_cache=physical_prediction_cache,
                evidence_cache=evidence_cache,
            ),
        )

    def _predictive_state(
        self,
        observation: PicfObservation,
        previous: PicfPreviousState | None,
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
        slot_count = int(posterior.tokens.shape[0])
        slot_prediction_tokens = physical_pred_tokens[:slot_count] if bool(self.config.slot_jepa_enabled) else None
        slot_prediction_supports = (
            torch.sigmoid(self.slot_support_pred_head(slot_prediction_tokens))
            if bool(self.config.support_prediction_enabled) and slot_prediction_tokens is not None
            else None
        )
        evidence_cache = self._write_evidence_cache(
            previous,
            posterior,
            innovation_norm=innovation_norm,
            availability=targets_availability,
            reset=bool(observation.reset_scaffold),
        )
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
            slot_prediction_tokens=slot_prediction_tokens,
            slot_prediction_supports=slot_prediction_supports,
            evidence_cache=evidence_cache,
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
            previous=observed.previous,
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
            vl_grounding=observed.vl_grounding,
            anchor_prior_graph=observed.anchor_prior_graph,
            object_explanation=observed.object_explanation,
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
        if observed.innovation_norm.numel() > 0:
            innov = observed.innovation_norm.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
            debug["innovation_norm_visual"] = float(innov[0].item()) if innov.numel() > 0 else 0.0
            debug["innovation_norm_tactile"] = float(innov[2].item()) if innov.numel() > 2 else 0.0
            debug["innovation_norm_point"] = float(innov[3].item()) if innov.numel() > 3 else 0.0
        if observed.token_field.tactile_contact_prob is not None and observed.token_field.tactile_contact_prob.numel() > 0:
            prob = observed.token_field.tactile_contact_prob.to(device=self.device, dtype=self.dtype).reshape(-1)
            debug["tactile_contact_prob_mean"] = float(prob.mean().item())
            debug["tactile_contact_prob_max"] = float(prob.max().item())
        if observed.token_field.tactile_anchor_mask is not None and observed.token_field.tactile_anchor_mask.numel() > 0:
            debug["tactile_active_rate"] = float(observed.token_field.tactile_anchor_mask.to(dtype=self.dtype).mean().item())
        if observed.token_field.tactile_evidence_mask is not None and observed.token_field.tactile_evidence_mask.numel() > 0:
            debug["tactile_evidence_rate"] = float(
                observed.token_field.tactile_evidence_mask.to(device=self.device, dtype=self.dtype).mean().item()
            )
        if observed.token_field.tactile_evidence_weight is not None and observed.token_field.tactile_evidence_weight.numel() > 0:
            weight = observed.token_field.tactile_evidence_weight.to(device=self.device, dtype=self.dtype).reshape(-1)
            debug["tactile_evidence_weight_mean"] = float(weight.mean().item())
            debug["tactile_evidence_weight_max"] = float(weight.max().item())
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
        if observed.vl_grounding is not None:
            debug["vl_grounding_valid"] = 1.0 if bool(observed.vl_grounding.valid.item()) else 0.0
            debug["vl_grounding_confidence"] = float(observed.vl_grounding.confidence.item())
            debug["vl_grounding_anchor_count"] = float(observed.vl_grounding.anchor_point_priors.shape[0])
        if observed.anchor_prior_graph is not None:
            graph = observed.anchor_prior_graph
            debug["mapg_valid"] = 1.0 if bool(graph.valid.item()) else 0.0
            debug["mapg_anchor_count"] = float(graph.anchor_tokens.shape[0])
            debug["mapg_visual_support_mean"] = float(graph.visual_priors.sum(dim=-1).mean().item()) if graph.visual_priors.numel() > 0 else 0.0
            debug["owm_pg_priors_available"] = 1.0 if graph.pg_priors is not None else 0.0
            debug["owm_temporal_priors_available"] = 1.0 if graph.vjepa_temporal_priors is not None else 0.0
            debug["owm_cache_priors_available"] = 1.0 if graph.cache_priors is not None else 0.0
            debug["owm_tracklet_priors_available"] = 1.0 if graph.tracklet_priors is not None else 0.0
            debug["owm_proposal_priors_available"] = 1.0 if graph.proposal_priors is not None else 0.0
            debug["owm_local_priors_available"] = 1.0 if graph.local_priors is not None else 0.0
            if graph.vjepa_temporal_priors is not None and graph.vjepa_temporal_priors.numel() > 0:
                temporal = _normalize_rows(
                    torch.clamp(graph.vjepa_temporal_priors.to(device=self.device, dtype=self.dtype), min=0.0),
                    eps=self.config.epsilon_a,
                )
                entropy = -(temporal * torch.log(torch.clamp(temporal, min=self.config.epsilon_a))).sum(dim=-1)
                entropy = entropy / math.log(max(int(temporal.shape[-1]), 2))
                debug["owm_temporal_support_mean"] = float(graph.vjepa_temporal_priors.sum(dim=-1).mean().item())
                debug["aqr_temporal_support_entropy_mean"] = float(entropy.mean().item())
                if observed.token_field.temporal_visual is not None:
                    time_ids = observed.token_field.temporal_visual.time_ids.to(device=self.device).reshape(-1)
                    for out_index, time_value in enumerate(tuple(torch.unique(time_ids, sorted=True).tolist())[:2]):
                        mask = time_ids == int(time_value)
                        if bool(mask.any().item()):
                            debug[f"aqr_temporal_support_time_mass_t{out_index}"] = float(temporal[:, mask].sum(dim=-1).mean().item())
                    view_ids = observed.token_field.temporal_visual.view_ids.to(device=self.device).reshape(-1)
                    for view_index in tuple(torch.unique(view_ids, sorted=True).tolist())[: int(self.config.vjepa_max_views)]:
                        mask = view_ids == int(view_index)
                        if bool(mask.any().item()):
                            debug[f"aqr_temporal_view_mass_{int(view_index)}"] = float(temporal[:, mask].sum(dim=-1).mean().item())
            if graph.pg_priors is not None and graph.pg_priors.numel() > 0:
                pg = _normalize_rows(
                    torch.clamp(graph.pg_priors.to(device=self.device, dtype=self.dtype), min=0.0),
                    eps=self.config.epsilon_a,
                )
                pg_entropy = -(pg * torch.log(torch.clamp(pg, min=self.config.epsilon_a))).sum(dim=-1)
                pg_entropy = pg_entropy / math.log(max(int(pg.shape[-1]), 2))
                pg_peak = pg.max(dim=-1).values
                debug["aqr_pg_support_entropy_mean"] = float(pg_entropy.mean().item())
                debug["aqr_pg_support_max"] = float(pg_peak.max().item())
                debug["aqr_pg_support_peak_mean"] = float(pg_peak.mean().item())
            if graph.support_uncertainty is not None and graph.support_uncertainty.numel() > 0:
                debug["owm_support_uncertainty_mean"] = float(graph.support_uncertainty.mean().item())
            if observed.object_explanation is not None and bool(observed.object_explanation.valid.item()):
                oeml = observed.object_explanation
                debug["oeml_valid"] = 1.0
                debug["oeml_anchor_quality_mean"] = float(oeml.anchor_quality.mean().item()) if oeml.anchor_quality.numel() > 0 else 0.0
                debug["oeml_anchor_quality_max"] = float(oeml.anchor_quality.max().item()) if oeml.anchor_quality.numel() > 0 else 0.0
                debug["oeml_duplicate_overlap_max"] = float(oeml.anchor_duplicate_overlap.max().item()) if oeml.anchor_duplicate_overlap.numel() > 0 else 0.0
                debug["oeml_duplicate_overlap_mean"] = float(oeml.anchor_duplicate_overlap.mean().item()) if oeml.anchor_duplicate_overlap.numel() > 0 else 0.0
                debug["oeml_feature_variance_mean"] = float(oeml.anchor_feature_variance.mean().item()) if oeml.anchor_feature_variance.numel() > 0 else 0.0
                debug["oeml_point_spatial_variance_mean"] = float(oeml.point_spatial_variance.mean().item()) if oeml.point_spatial_variance.numel() > 0 else 0.0
                debug["oeml_contact_explanation_score"] = float(oeml.contact_explanation_score.item())
                if oeml.background_mask_visual is not None and oeml.background_mask_visual.numel() > 0:
                    debug["oeml_visual_background_mean"] = float(oeml.background_mask_visual.mean().item())
                if oeml.background_mask_point is not None and oeml.background_mask_point.numel() > 0:
                    debug["oeml_point_background_mean"] = float(oeml.background_mask_point.mean().item())
                if oeml.candidate_coverage is not None and oeml.candidate_coverage.numel() > 0:
                    coverage = torch.clamp(oeml.candidate_coverage.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
                    debug["oeml_candidate_coverage_mean"] = float(coverage.mean().item())
                    debug["oeml_candidate_coverage_max"] = float(coverage.max().item())
                if oeml.candidate_background is not None and oeml.candidate_background.numel() > 0:
                    background = torch.clamp(oeml.candidate_background.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
                    debug["oeml_candidate_background_mean"] = float(background.mean().item())
                if oeml.candidate_duplicate_overlap is not None and oeml.candidate_duplicate_overlap.numel() > 0:
                    debug["oeml_candidate_duplicate_overlap_max"] = float(
                        oeml.candidate_duplicate_overlap.to(device=self.device, dtype=self.dtype).max().item()
                    )
            else:
                debug["oeml_valid"] = 0.0
            if graph.tracklet_priors is not None and graph.tracklet_priors.numel() > 0:
                tr = _normalize_rows(torch.clamp(graph.tracklet_priors.to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
                tr_entropy = -(tr * torch.log(torch.clamp(tr, min=self.config.epsilon_a))).sum(dim=-1)
                tr_entropy = tr_entropy / math.log(max(int(tr.shape[-1]), 2))
                debug["aqr_tracklet_support_entropy_mean"] = float(tr_entropy.mean().item())
                debug["aqr_tracklet_support_max"] = float(tr.max(dim=-1).values.max().item())
            if graph.proposal_priors is not None and graph.proposal_priors.numel() > 0:
                prop = _normalize_rows(
                    torch.clamp(graph.proposal_priors.to(device=self.device, dtype=self.dtype), min=0.0),
                    eps=self.config.epsilon_a,
                )
                prop_entropy = -(prop * torch.log(torch.clamp(prop, min=self.config.epsilon_a))).sum(dim=-1)
                prop_entropy = prop_entropy / math.log(max(int(prop.shape[-1]), 2))
                debug["aqr_proposal_support_entropy_mean"] = float(prop_entropy.mean().item())
                debug["aqr_proposal_support_max"] = float(prop.max(dim=-1).values.max().item())
                proposal_quality = self._proposal_shape_quality(observed.token_field.proposal)
                if proposal_quality is not None and proposal_quality.numel() > 0:
                    debug["aqr_proposal_shape_quality_mean"] = float(proposal_quality.mean().item())
                    debug["aqr_proposal_shape_quality_max"] = float(proposal_quality.max().item())
                    debug["aqr_proposal_shape_quality_nonzero_fraction"] = float(
                        (proposal_quality > self.config.epsilon_a).to(dtype=self.dtype).mean().item()
                    )
            if graph.proposal_point_priors is not None and graph.proposal_point_priors.numel() > 0:
                prop_point = _normalize_rows(
                    torch.clamp(graph.proposal_point_priors.to(device=self.device, dtype=self.dtype), min=0.0),
                    eps=self.config.epsilon_a,
                )
                prop_point_entropy = -(prop_point * torch.log(torch.clamp(prop_point, min=self.config.epsilon_a))).sum(dim=-1)
                prop_point_entropy = prop_point_entropy / math.log(max(int(prop_point.shape[-1]), 2))
                debug["aqr_proposal_point_bridge_entropy_mean"] = float(prop_point_entropy.mean().item())
                debug["aqr_proposal_point_bridge_max"] = float(prop_point.max(dim=-1).values.max().item())
            if graph.task_owner_point_priors is not None and graph.task_owner_point_priors.numel() > 0:
                owner_point = torch.clamp(graph.task_owner_point_priors.to(device=self.device, dtype=self.dtype), min=0.0)
                nonzero = owner_point.sum(dim=-1) > self.config.epsilon_a
                if bool(nonzero.any().item()):
                    owner_point_norm = _normalize_rows(owner_point.index_select(0, torch.nonzero(nonzero, as_tuple=False).squeeze(-1)), eps=self.config.epsilon_a)
                    owner_point_entropy = -(owner_point_norm * torch.log(torch.clamp(owner_point_norm, min=self.config.epsilon_a))).sum(dim=-1)
                    owner_point_entropy = owner_point_entropy / math.log(max(int(owner_point_norm.shape[-1]), 2))
                    debug["aqr_task_owner_point_bridge_entropy_mean"] = float(owner_point_entropy.mean().item())
                    debug["aqr_task_owner_point_bridge_max"] = float(owner_point_norm.max(dim=-1).values.max().item())
                    debug["aqr_task_owner_point_bridge_nonzero_fraction"] = float(nonzero.to(dtype=self.dtype).mean().item())
            if graph.proposal_anchor_seed_priors is not None and graph.proposal_anchor_seed_priors.numel() > 0:
                seed_point = torch.clamp(graph.proposal_anchor_seed_priors.to(device=self.device, dtype=self.dtype), min=0.0)
                seed_rows = seed_point.sum(dim=-1) > self.config.epsilon_a
                debug["aqr_proposal_anchor_seed_row_count"] = float(seed_rows.to(dtype=self.dtype).sum().item())
                debug["aqr_proposal_anchor_seed_nonzero_fraction"] = float(seed_rows.to(dtype=self.dtype).mean().item())
                if bool(seed_rows.any().item()):
                    seed_norm = _normalize_rows(seed_point.index_select(0, torch.nonzero(seed_rows, as_tuple=False).squeeze(-1)), eps=self.config.epsilon_a)
                    seed_entropy = -(seed_norm * torch.log(torch.clamp(seed_norm, min=self.config.epsilon_a))).sum(dim=-1)
                    seed_entropy = seed_entropy / math.log(max(int(seed_norm.shape[-1]), 2))
                    debug["aqr_proposal_anchor_seed_point_max"] = float(seed_norm.max(dim=-1).values.max().item())
                    debug["aqr_proposal_anchor_seed_entropy_mean"] = float(seed_entropy.mean().item())
            if graph.proposal_anchor_seed_assignment is not None and graph.proposal_anchor_seed_assignment.numel() > 0:
                seed_assign = torch.clamp(graph.proposal_anchor_seed_assignment.to(device=self.device, dtype=self.dtype), min=0.0)
                debug["aqr_proposal_anchor_seed_assignment_max"] = float(seed_assign.max().item())
            if graph.object_candidate_assignment is not None and graph.object_candidate_assignment.numel() > 0:
                candidate_assign = torch.clamp(graph.object_candidate_assignment.to(device=self.device, dtype=self.dtype), min=0.0)
                candidate_rows = candidate_assign.sum(dim=-1) > self.config.epsilon_a
                candidate_cols = candidate_assign.sum(dim=0) > self.config.epsilon_a
                debug["aqr_object_candidate_assigned_row_count"] = float(candidate_rows.to(dtype=self.dtype).sum().item())
                debug["aqr_object_candidate_assigned_candidate_count"] = float(candidate_cols.to(dtype=self.dtype).sum().item())
                debug["aqr_object_candidate_assignment_max"] = float(candidate_assign.max().item())
                if graph.object_candidate_owner_assignment is not None and graph.object_candidate_owner_assignment.numel() > 0:
                    owner_assign = torch.clamp(
                        graph.object_candidate_owner_assignment.to(device=self.device, dtype=self.dtype),
                        min=0.0,
                    )
                    owner_rows = owner_assign.sum(dim=-1) > self.config.epsilon_a
                    owner_cols = owner_assign.sum(dim=0) > self.config.epsilon_a
                    debug["aqr_object_candidate_owner_row_count"] = float(owner_rows.to(dtype=self.dtype).sum().item())
                    debug["aqr_object_candidate_owner_candidate_count"] = float(owner_cols.to(dtype=self.dtype).sum().item())
                    debug["aqr_object_candidate_owner_assignment_max"] = float(owner_assign.max().item())
                if graph.object_candidate_owner_point_priors is not None and graph.object_candidate_owner_point_priors.numel() > 0:
                    owner_point = torch.clamp(
                        graph.object_candidate_owner_point_priors.to(device=self.device, dtype=self.dtype),
                        min=0.0,
                    )
                    owner_point_rows = owner_point.sum(dim=-1) > self.config.epsilon_a
                    debug["aqr_object_candidate_owner_point_row_count"] = float(
                        owner_point_rows.to(dtype=self.dtype).sum().item()
                    )
                    if bool(owner_point_rows.any().item()):
                        owner_point_norm = _normalize_rows(
                            owner_point.index_select(0, torch.nonzero(owner_point_rows, as_tuple=False).squeeze(-1)),
                            eps=self.config.epsilon_a,
                        )
                        debug["aqr_object_candidate_owner_point_max"] = float(owner_point_norm.max(dim=-1).values.max().item())
                if graph.object_candidate_coverage is not None and graph.object_candidate_coverage.numel() > 0:
                    coverage = torch.clamp(graph.object_candidate_coverage.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
                    debug["aqr_object_candidate_coverage_mean"] = float(coverage.mean().item())
                    debug["aqr_object_candidate_coverage_max"] = float(coverage.max().item())
                if graph.object_candidate_background is not None and graph.object_candidate_background.numel() > 0:
                    background = torch.clamp(graph.object_candidate_background.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
                    debug["aqr_object_candidate_background_mean"] = float(background.mean().item())
                if graph.object_candidate_duplicate_overlap is not None and graph.object_candidate_duplicate_overlap.numel() > 0:
                    debug["aqr_object_candidate_duplicate_overlap_max"] = float(
                        graph.object_candidate_duplicate_overlap.to(device=self.device, dtype=self.dtype).max().item()
                    )
            if graph.task_owner_visual_prior is not None and graph.task_owner_visual_prior.numel() > 0:
                owner_visual = _normalize_rows(
                    torch.clamp(graph.task_owner_visual_prior.to(device=self.device, dtype=self.dtype).reshape(1, -1), min=0.0),
                    eps=self.config.epsilon_a,
                )[0]
                owner_visual_entropy = -(owner_visual * torch.log(torch.clamp(owner_visual, min=self.config.epsilon_a))).sum()
                owner_visual_entropy = owner_visual_entropy / math.log(max(int(owner_visual.numel()), 2))
                debug["aqr_task_owner_visual_prior_entropy"] = float(owner_visual_entropy.item())
                debug["aqr_task_owner_visual_prior_max"] = float(owner_visual.max().item())
            if graph.task_owner_proposal_score is not None and graph.task_owner_proposal_score.numel() > 0:
                owner_prop = torch.clamp(graph.task_owner_proposal_score.to(device=self.device, dtype=self.dtype), min=0.0)
                debug["aqr_task_owner_proposal_score_max"] = float(owner_prop.max().item())
                debug["aqr_task_owner_proposal_score_mean"] = float(owner_prop.mean().item())
                debug["aqr_task_owner_proposal_score_nonzero_fraction"] = float(
                    (owner_prop > self.config.epsilon_a).to(dtype=self.dtype).mean().item()
                )
                if bool((owner_prop.sum() > self.config.epsilon_a).item()):
                    owner_norm = owner_prop / torch.clamp(owner_prop.sum(), min=self.config.epsilon_a)
                    owner_entropy = -(owner_norm * torch.log(torch.clamp(owner_norm, min=self.config.epsilon_a))).sum()
                    owner_entropy = owner_entropy / math.log(max(int(owner_norm.numel()), 2))
                    debug["aqr_task_owner_proposal_score_entropy"] = float(owner_entropy.item())
                    debug["aqr_task_owner_proposal_selected_count"] = float(
                        (owner_prop > self.config.epsilon_a).to(dtype=self.dtype).sum().item()
                    )
                    proposal_quality = self._proposal_shape_quality(observed.token_field.proposal)
                    if proposal_quality is not None and proposal_quality.numel() == owner_prop.numel():
                        selected = owner_prop > self.config.epsilon_a
                        if bool(selected.any().item()):
                            debug["aqr_task_owner_proposal_shape_quality_mean"] = float(
                                proposal_quality[selected].mean().item()
                            )
            if graph.task_owner_anchor_score is not None and graph.task_owner_anchor_score.numel() > 0:
                owner_anchor = torch.clamp(graph.task_owner_anchor_score.to(device=self.device, dtype=self.dtype), min=0.0)
                debug["aqr_task_owner_anchor_score_max"] = float(owner_anchor.max().item())
                debug["aqr_task_owner_anchor_score_mean"] = float(owner_anchor.mean().item())
                debug["aqr_task_owner_anchor_score_nonzero_fraction"] = float(
                    (owner_anchor > self.config.epsilon_a).to(dtype=self.dtype).mean().item()
                )
            if graph.active_proposals is not None:
                proposals = graph.active_proposals
                active_prob = torch.clamp(proposals.active_prob.to(device=self.device, dtype=self.dtype), min=0.0, max=1.0)
                debug["vcap_enabled"] = 1.0
                debug["vcap_proposal_count"] = float(active_prob.sum().item())
                debug["vcap_active_prob_mean"] = float(active_prob.mean().item()) if active_prob.numel() > 0 else 0.0
                stop_prob = torch.sigmoid(proposals.stop_logits.to(device=self.device, dtype=self.dtype))
                stop_entropy = -(
                    stop_prob * torch.log(torch.clamp(stop_prob, min=self.config.epsilon_a))
                    + (1.0 - stop_prob) * torch.log(torch.clamp(1.0 - stop_prob, min=self.config.epsilon_a))
                )
                debug["vcap_stop_entropy"] = float(stop_entropy.mean().item()) if stop_entropy.numel() > 0 else 0.0
                if proposals.unexplained_evidence is not None:
                    debug["vcap_unexplained_evidence"] = float(proposals.unexplained_evidence.to(device=self.device, dtype=self.dtype).mean().item())
                debug["vcap_duplicate_cost"] = float(proposals.duplicate_score.to(device=self.device, dtype=self.dtype).mean().item())
                if proposals.count_cost is not None:
                    debug["vcap_count_cost"] = float(proposals.count_cost.to(device=self.device, dtype=self.dtype).mean().item())
                if proposals.continuity_cost is not None:
                    debug["vcap_continuity_cost"] = float(proposals.continuity_cost.to(device=self.device, dtype=self.dtype).mean().item())
                posterior = observed.posterior
                width = int(active_prob.numel())
                denom = torch.clamp(active_prob.sum(), min=self.config.epsilon_a)
                matched = None
                if posterior.file_competition_active is not None and posterior.file_competition_active.numel() >= width:
                    matched = torch.clamp(
                        posterior.file_competition_active.to(device=self.device, dtype=self.dtype).reshape(-1)[:width],
                        min=0.0,
                        max=1.0,
                    )
                    debug["vcap_matched_old_file_fraction"] = float(((active_prob * matched).sum() / denom).item())
                birth = None
                if posterior.file_competition_birth_active is not None and posterior.file_competition_birth_active.numel() >= width:
                    birth = torch.clamp(
                        posterior.file_competition_birth_active.to(device=self.device, dtype=self.dtype).reshape(-1)[:width],
                        min=0.0,
                        max=1.0,
                    )
                    debug["vcap_birth_fraction"] = float(((active_prob * birth).sum() / denom).item())
                if matched is not None or birth is not None:
                    matched_v = torch.zeros_like(active_prob) if matched is None else matched
                    birth_v = torch.zeros_like(active_prob) if birth is None else birth
                    noobject = 1.0 - torch.clamp(torch.maximum(matched_v, birth_v), min=0.0, max=1.0)
                    debug["vcap_noobject_fraction"] = float(((active_prob * noobject).sum() / denom).item())
                debug["vcap_action_grad_scale"] = float(getattr(self.config, "vcap_action_grad_scale", 0.0))
            else:
                debug["vcap_enabled"] = 0.0
            if graph.local_priors is not None and graph.local_priors.numel() > 0:
                lp = _normalize_rows(torch.clamp(graph.local_priors.to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
                lp_entropy = -(lp * torch.log(torch.clamp(lp, min=self.config.epsilon_a))).sum(dim=-1)
                lp_entropy = lp_entropy / math.log(max(int(lp.shape[-1]), 2))
                debug["aqr_local_support_entropy_mean"] = float(lp_entropy.mean().item())
                local_source_ids = graph.local_source_ids
                if local_source_ids is not None and local_source_ids.shape == lp.shape:
                    for source_id, source_name in (
                        (1, "visual"),
                        (2, "temporal"),
                        (3, "point"),
                        (4, "tracklet"),
                        (5, "proposal"),
                    ):
                        source_mask = (local_source_ids.to(device=self.device) == int(source_id)).to(device=self.device, dtype=self.dtype)
                        debug[f"aqr_local_source_mass_{source_name}"] = float((lp * source_mask).sum(dim=-1).mean().item())
                if lp.shape[0] > 1:
                    lp_overlap = lp @ lp.T
                    lp_diag = torch.clamp(torch.diag(lp_overlap), min=self.config.epsilon_a)
                    lp_overlap = lp_overlap / torch.sqrt(torch.clamp(lp_diag[:, None] * lp_diag[None, :], min=self.config.epsilon_a))
                    lp_same_role = graph.anchor_roles[:, None] == graph.anchor_roles[None, :]
                    lp_pair_mask = torch.triu(lp_same_role, diagonal=1)
                    if bool(lp_pair_mask.any().item()):
                        debug["aqr_same_role_local_overlap_max"] = float(lp_overlap[lp_pair_mask].max().item())
                    local_indices = graph.local_token_indices
                    if (
                        local_indices is not None
                        and local_indices.shape == lp.shape
                        and bool(lp_pair_mask.any().item())
                    ):
                        idx = local_indices.to(device=self.device, dtype=torch.long)
                        true_overlap_values: list[torch.Tensor] = []
                        jaccard_values: list[torch.Tensor] = []
                        anchor_count = int(lp.shape[0])
                        for i in range(anchor_count):
                            for j in range(i + 1, anchor_count):
                                if not bool(lp_same_role[i, j].item()):
                                    continue
                                same = idx[i, :, None] == idx[j, None, :]
                                true_dot = (lp[i, :, None] * lp[j, None, :] * same.to(dtype=self.dtype)).sum()
                                true_overlap_values.append(true_dot)
                                uniq_i = torch.unique(idx[i])
                                uniq_j = torch.unique(idx[j])
                                inter = (uniq_i[:, None] == uniq_j[None, :]).any(dim=1).sum().to(dtype=self.dtype)
                                union = torch.unique(torch.cat([uniq_i, uniq_j], dim=0)).numel()
                                jaccard_values.append(inter / max(float(union), 1.0))
                        if true_overlap_values:
                            true_overlap = torch.stack(true_overlap_values)
                            true_jaccard = torch.stack(jaccard_values)
                            debug["aqr_same_role_local_true_overlap_max"] = float(true_overlap.max().item())
                            debug["aqr_same_role_local_true_overlap_mean"] = float(true_overlap.mean().item())
                            debug["aqr_same_role_local_jaccard_max"] = float(true_jaccard.max().item())
                            debug["aqr_same_role_local_jaccard_mean"] = float(true_jaccard.mean().item())
                if graph.anchor_tokens.numel() > 0 and graph.anchor_roles.numel() == graph.anchor_tokens.shape[0]:
                    anchor_binding = self._binding_keys(graph.anchor_tokens.to(device=self.device, dtype=self.dtype))
                    if anchor_binding.shape[0] > 1:
                        bind_sim = anchor_binding @ anchor_binding.T
                        bind_same_role = graph.anchor_roles[:, None] == graph.anchor_roles[None, :]
                        bind_pair_mask = torch.triu(bind_same_role, diagonal=1)
                        if bool(bind_pair_mask.any().item()):
                            debug["aqr_same_role_anchor_binding_signature_overlap_max"] = float(
                                bind_sim[bind_pair_mask].max().item()
                            )
                            debug["aqr_same_role_anchor_binding_signature_overlap_mean"] = float(
                                bind_sim[bind_pair_mask].mean().item()
                            )
            debug["mapg_point_available"] = 1.0 if graph.point_priors is not None else 0.0
            debug["mapg_tactile_available"] = 1.0 if graph.tactile_priors is not None else 0.0
            debug["mapg_posterior_available"] = 1.0 if graph.posterior_priors is not None else 0.0
            debug["aqr_ownership_prior_enabled"] = 1.0 if bool(getattr(self.config, "aqr_ownership_prior_enabled", True)) else 0.0
            debug["aqr_ownership_prior_weight"] = float(getattr(self.config, "aqr_ownership_prior_weight", 0.0))
            debug["aqr_ownership_point_prior_weight"] = float(
                getattr(self.config, "aqr_ownership_point_prior_weight", 0.0)
            )
            debug["aqr_ownership_point_prior_sigma_m"] = float(
                getattr(self.config, "aqr_ownership_point_prior_sigma_m", 0.0)
            )
            debug["aqr_ownership_temporal_prior_weight"] = float(
                getattr(self.config, "aqr_ownership_temporal_prior_weight", 0.0)
            )
            debug["aqr_active_slot_relative_score_threshold"] = float(
                getattr(self.config, "aqr_active_slot_relative_score_threshold", 0.0)
            )
            debug["aqr_active_slot_geometry_duplicate_enabled"] = (
                1.0 if bool(getattr(self.config, "aqr_active_slot_geometry_duplicate_enabled", True)) else 0.0
            )
            debug["aqr_active_slot_geometry_duplicate_threshold"] = float(
                getattr(self.config, "aqr_active_slot_geometry_duplicate_threshold", 0.0)
            )
            debug["aqr_same_role_support_competition_enabled"] = (
                1.0 if bool(getattr(self.config, "aqr_same_role_support_competition_enabled", False)) else 0.0
            )
            debug["aqr_same_role_support_competition_weight"] = float(
                getattr(self.config, "aqr_same_role_support_competition_weight", 0.0)
            )
            usage = torch.zeros((graph.anchor_tokens.shape[0],), device=self.device, dtype=self.dtype)
            for assignment in (graph.obs_slot_assignment, graph.task_assignment):
                if assignment is not None and assignment.numel() > 0 and assignment.shape[-1] == usage.shape[0]:
                    usage = usage + assignment.to(device=self.device, dtype=self.dtype).sum(dim=0)
            if usage.numel() > 0 and bool((usage.sum() > self.config.epsilon_a).item()):
                usage_prob = usage / torch.clamp(usage.sum(), min=self.config.epsilon_a)
                usage_entropy = -torch.sum(usage_prob * torch.log(torch.clamp(usage_prob, min=self.config.epsilon_a)))
                debug["mapg_assignment_effective_anchors"] = float(torch.exp(usage_entropy).item())
                debug["mapg_assignment_max_column_mass"] = float(usage.max().item())
            if graph.visual_priors.shape[0] > 1 and graph.visual_priors.numel() > 0:
                visual_priors = _normalize_rows(torch.clamp(graph.visual_priors.to(device=self.device, dtype=self.dtype), min=0.0), eps=self.config.epsilon_a)
                overlap = visual_priors @ visual_priors.T
                diag = torch.clamp(torch.diag(overlap), min=self.config.epsilon_a)
                overlap = overlap / torch.sqrt(torch.clamp(diag[:, None] * diag[None, :], min=self.config.epsilon_a))
                object_core_overlap = self._object_core_overlap_matrix(
                    graph.visual_priors,
                    point_priors=graph.point_priors,
                    temporal_priors=graph.vjepa_temporal_priors,
                    pg_priors=graph.pg_priors,
                    proposal_priors=graph.proposal_priors,
                )
                same_role = graph.anchor_roles[:, None] == graph.anchor_roles[None, :]
                pair_mask = torch.triu(same_role, diagonal=1)
                if bool(pair_mask.any().item()):
                    debug["mapg_same_role_visual_overlap_max"] = float(overlap[pair_mask].max().item())
                    debug["aqr_same_role_support_overlap_max"] = debug["mapg_same_role_visual_overlap_max"]
                    if object_core_overlap is not None:
                        debug["aqr_same_role_object_core_overlap_max"] = float(object_core_overlap[pair_mask].max().item())
                        debug["aqr_same_role_object_core_overlap_mean"] = float(object_core_overlap[pair_mask].mean().item())
                if graph.anchor_active is not None and graph.anchor_active.numel() == graph.visual_priors.shape[0]:
                    active = graph.anchor_active.to(device=self.device, dtype=self.dtype).reshape(-1)
                    active_bool = active > 0.5
                    debug["aqr_active_anchor_count"] = float(active.sum().item())
                    debug["aqr_inactive_anchor_fraction"] = float((1.0 - active).mean().item())
                    if (
                        graph.anchor_downstream_weight is not None
                        and graph.anchor_downstream_weight.numel() == active.numel()
                    ):
                        downstream = torch.clamp(
                            graph.anchor_downstream_weight.to(device=self.device, dtype=self.dtype).reshape(-1),
                            min=0.0,
                            max=1.0,
                        )
                        context_bool = (downstream > self.config.epsilon_a) & (~active_bool)
                        reserve_bool = downstream <= self.config.epsilon_a
                        debug["aqr_context_anchor_count"] = float(context_bool.to(dtype=self.dtype).sum().item())
                        debug["aqr_reserve_anchor_fraction"] = float(reserve_bool.to(dtype=self.dtype).mean().item())
                        if bool(context_bool.any().item()):
                            debug["aqr_context_downstream_weight_mean"] = float(downstream[context_bool].mean().item())
                        else:
                            debug["aqr_context_downstream_weight_mean"] = 0.0
                    active_pair_mask = pair_mask & active_bool[:, None] & active_bool[None, :]
                    if bool(active_pair_mask.any().item()):
                        debug["aqr_active_same_role_support_overlap_max"] = float(overlap[active_pair_mask].max().item())
                        debug["aqr_active_same_role_support_overlap_mean"] = float(overlap[active_pair_mask].mean().item())
                        if object_core_overlap is not None:
                            debug["aqr_active_same_role_object_core_overlap_max"] = float(object_core_overlap[active_pair_mask].max().item())
                            debug["aqr_active_same_role_object_core_overlap_mean"] = float(object_core_overlap[active_pair_mask].mean().item())
                    else:
                        debug["aqr_active_same_role_support_overlap_max"] = 0.0
                        debug["aqr_active_same_role_support_overlap_mean"] = 0.0
                        debug["aqr_active_same_role_object_core_overlap_max"] = 0.0
                        debug["aqr_active_same_role_object_core_overlap_mean"] = 0.0
                    if graph.anchor_roles.numel() == active.shape[0]:
                        roles = graph.anchor_roles.to(device=self.device, dtype=torch.long)
                        for role_value in tuple(torch.unique(roles, sorted=True).tolist()):
                            role_mask = roles == int(role_value)
                            if bool(role_mask.any().item()):
                                debug[f"aqr_active_anchor_count_role_{int(role_value)}"] = float(active[role_mask].sum().item())
            if "mapg_assignment_effective_anchors" in debug:
                debug["aqr_effective_anchor_count"] = debug["mapg_assignment_effective_anchors"]
        if observed.token_field.temporal_visual is not None:
            debug["owm_temporal_visual_tokens"] = float(observed.token_field.temporal_visual.tokens.shape[0])
        if observed.token_field.tracklet is not None:
            debug["owm_tracklet_tokens"] = float(observed.token_field.tracklet.tokens.shape[0])
            debug["owm_tracklet_valid_fraction"] = float(observed.token_field.tracklet.valid.to(dtype=self.dtype).mean().item())
        if observed.token_field.proposal is not None:
            debug["owm_proposal_tokens"] = float(observed.token_field.proposal.tokens.shape[0])
            debug["owm_proposal_valid_fraction"] = float(observed.token_field.proposal.valid.to(dtype=self.dtype).mean().item())
        if observed.posterior.slot_address is not None:
            debug["owm_slot_address_norm_mean"] = float(torch.linalg.norm(observed.posterior.slot_address, dim=-1).mean().item())
        if observed.posterior.support_signature is not None:
            debug["owm_posterior_support_signature_mean"] = float(observed.posterior.support_signature.mean().item())
        if observed.posterior.binding_signature is not None:
            debug["owm_posterior_binding_signature_norm_mean"] = float(
                torch.linalg.norm(observed.posterior.binding_signature, dim=-1).mean().item()
            )
        for attr_name, debug_name in (
            ("binding_signature_linear_score_mean", "posterior_binding_signature_linear_score_mean"),
            ("binding_signature_linear_score_abs_mean", "posterior_binding_signature_linear_score_abs_mean"),
            ("binding_signature_quadratic_score_mean", "posterior_binding_signature_quadratic_score_mean"),
            ("binding_signature_quadratic_score_abs_mean", "posterior_binding_signature_quadratic_score_abs_mean"),
            ("binding_signature_low_rank_score_mean", "posterior_binding_signature_low_rank_score_mean"),
            ("binding_signature_low_rank_score_abs_mean", "posterior_binding_signature_low_rank_score_abs_mean"),
            ("binding_signature_combined_score_mean", "posterior_binding_signature_combined_score_mean"),
            ("binding_signature_combined_score_abs_mean", "posterior_binding_signature_combined_score_abs_mean"),
            ("binding_signature_calibrated_score_mean", "posterior_binding_signature_calibrated_score_mean"),
            ("binding_signature_calibrated_score_abs_mean", "posterior_binding_signature_calibrated_score_abs_mean"),
            ("binding_signature_calibrated_score_std", "posterior_binding_signature_calibrated_score_std"),
            ("binding_signature_calibrated_top1_margin_mean", "posterior_binding_signature_calibrated_top1_margin_mean"),
            ("binding_signature_gate_mean", "posterior_binding_signature_gate_mean"),
            ("binding_signature_update_rate", "posterior_binding_signature_update_rate_mean"),
            ("binding_signature_measurement_trust", "posterior_binding_signature_measurement_trust_mean"),
            ("binding_signature_memory_keep_rate", "posterior_binding_signature_memory_keep_rate_mean"),
            ("binding_signature_measurement_score_std", "posterior_binding_signature_measurement_score_std"),
            ("binding_signature_measurement_margin", "posterior_binding_signature_measurement_margin_mean"),
            ("binding_signature_measurement_dispersion_gate", "posterior_binding_signature_measurement_dispersion_gate_mean"),
        ):
            value = getattr(observed.posterior, attr_name, None)
            if value is not None and value.numel() > 0:
                debug[debug_name] = float(value.to(device=self.device, dtype=self.dtype).mean().item())
        if observed.observation_anchors.owner_active is not None and observed.observation_anchors.owner_active.numel() > 0:
            owner_active = torch.clamp(
                observed.observation_anchors.owner_active.to(device=self.device, dtype=self.dtype).reshape(-1),
                min=0.0,
                max=1.0,
            )
            owner_threshold = min(max(float(getattr(self.config, "posterior_owner_active_min", 0.25)), 0.0), 1.0)
            debug["posterior_owner_active_score_mean"] = float(owner_active.mean().item())
            debug["posterior_owner_active_score_max"] = float(owner_active.max().item())
            debug["posterior_owner_active_eligible_fraction"] = float((owner_active >= owner_threshold).to(dtype=self.dtype).mean().item())
            debug["posterior_owner_active_gate_enabled"] = (
                1.0 if bool(getattr(self.config, "posterior_owner_active_gate_enabled", True)) else 0.0
            )
        if (
            observed.observation_anchors.binding_signature is not None
            and observed.observation_anchors.binding_signature.numel() > 0
            and observed.observation_anchors.role_ids is not None
            and observed.observation_anchors.role_ids.numel() == observed.observation_anchors.binding_signature.shape[0]
        ):
            obs_binding = _normalize_tensor(
                observed.observation_anchors.binding_signature.to(device=self.device, dtype=self.dtype),
                eps=self.config.epsilon_residual,
            )
            if obs_binding.shape[0] > 1:
                obs_bind_sim = obs_binding @ obs_binding.T
                obs_roles = observed.observation_anchors.role_ids.to(device=self.device, dtype=torch.long)
                obs_same_role = obs_roles[:, None] == obs_roles[None, :]
                obs_pair_mask = torch.triu(obs_same_role, diagonal=1)
                if bool(obs_pair_mask.any().item()):
                    debug["aqr_same_role_obs_binding_signature_overlap_max"] = float(
                        obs_bind_sim[obs_pair_mask].max().item()
                    )
                    debug["aqr_same_role_obs_binding_signature_overlap_mean"] = float(
                        obs_bind_sim[obs_pair_mask].mean().item()
                    )
        if observed.posterior.binding is not None and observed.posterior.binding.numel() > 0:
            previous_posterior = None if observed.previous is None else getattr(observed.previous, "posterior", None)
            previous_binding = None if previous_posterior is None else getattr(previous_posterior, "binding", None)
            if previous_binding is not None and previous_binding.numel() > 0:
                current_binding = observed.posterior.binding.to(device=self.device, dtype=self.dtype)
                previous_binding = previous_binding.to(device=self.device, dtype=self.dtype)
                slot_count = min(int(current_binding.shape[0]), int(previous_binding.shape[0]))
                if slot_count > 0:
                    current_flat = current_binding[:slot_count].reshape(slot_count, -1)
                    previous_flat = previous_binding[:slot_count].reshape(slot_count, -1)
                    current_ids = current_flat.argmax(dim=-1)
                    previous_ids = previous_flat.argmax(dim=-1)
                    switched = current_ids != previous_ids
                    debug["posterior_identity_switch_rate"] = float(switched.to(dtype=self.dtype).mean().item())

                    def _binding_margin(flat: torch.Tensor) -> torch.Tensor:
                        if flat.shape[-1] <= 1:
                            return torch.ones((flat.shape[0],), device=self.device, dtype=self.dtype)
                        top2 = torch.topk(flat, k=2, dim=-1).values
                        return top2[:, 0] - top2[:, 1]

                    current_margin = _binding_margin(current_flat)
                    previous_margin = _binding_margin(previous_flat)
                    debug["posterior_binding_top1_margin_mean"] = float(current_margin.mean().item())
                    debug["posterior_binding_top1_margin_min"] = float(current_margin.min().item())

                    nonrecycled = torch.ones((slot_count,), device=self.device, dtype=torch.bool)
                    current_recycle = observed.posterior.recycle_gate
                    previous_recycle = getattr(previous_posterior, "recycle_gate", None)
                    if current_recycle is not None and current_recycle.numel() >= slot_count:
                        nonrecycled = nonrecycled & (
                            current_recycle.to(device=self.device, dtype=self.dtype).reshape(-1)[:slot_count] <= 0.5
                        )
                    if previous_recycle is not None and previous_recycle.numel() >= slot_count:
                        nonrecycled = nonrecycled & (
                            previous_recycle.to(device=self.device, dtype=self.dtype).reshape(-1)[:slot_count] <= 0.5
                        )
                    if bool(nonrecycled.any().item()):
                        debug["posterior_identity_switch_rate_nonrecycled"] = float(switched[nonrecycled].to(dtype=self.dtype).mean().item())
                    recycled = ~nonrecycled
                    if bool(recycled.any().item()):
                        debug["posterior_identity_switch_rate_recycled"] = float(switched[recycled].to(dtype=self.dtype).mean().item())

                    # Stable-slot switch rate separates true persistent identity
                    # instability from raw argmax churn on low-confidence slots.
                    stable = nonrecycled.clone()
                    for value in (observed.posterior.file_competition_active, getattr(previous_posterior, "file_competition_active", None)):
                        if value is not None and value.numel() >= slot_count:
                            stable = stable & (value.to(device=self.device, dtype=self.dtype).reshape(-1)[:slot_count] >= 0.5)
                    for value in (observed.posterior.alpha, getattr(previous_posterior, "alpha", None)):
                        if value is not None and value.numel() >= slot_count:
                            stable = stable & (value.to(device=self.device, dtype=self.dtype).reshape(-1)[:slot_count] >= 0.25)
                    for value in (observed.posterior.support_mass, getattr(previous_posterior, "support_mass", None)):
                        if value is not None and value.numel() >= slot_count:
                            stable = stable & (value.to(device=self.device, dtype=self.dtype).reshape(-1)[:slot_count] >= 0.05)
                    stable = stable & (current_margin >= 0.05) & (previous_margin >= 0.05)
                    debug["posterior_stable_slot_fraction"] = float(stable.to(dtype=self.dtype).mean().item())
                    if bool(stable.any().item()):
                        debug["posterior_identity_switch_rate_stable"] = float(switched[stable].to(dtype=self.dtype).mean().item())
                        debug["posterior_binding_top1_margin_stable_mean"] = float(current_margin[stable].mean().item())

                    # The row-argmax identity switch metrics above compare
                    # observation-anchor row ids across steps. Those rows are
                    # not stable object ids. Track posterior object-file
                    # continuity directly with the file-local binding signature
                    # and only interpret the active, non-recycled files.
                    current_sig = observed.posterior.binding_signature
                    previous_sig = getattr(previous_posterior, "binding_signature", None)
                    if current_sig is not None and previous_sig is not None and current_sig.numel() > 0 and previous_sig.numel() > 0:
                        file_count = min(slot_count, int(current_sig.shape[0]), int(previous_sig.shape[0]))
                        if file_count > 0:
                            curr_file_sig = _normalize_tensor(
                                current_sig.to(device=self.device, dtype=self.dtype)[:file_count],
                                eps=self.config.epsilon_residual,
                            )
                            prev_file_sig = _normalize_tensor(
                                previous_sig.to(device=self.device, dtype=self.dtype)[:file_count],
                                eps=self.config.epsilon_residual,
                            )
                            file_sim = curr_file_sig @ prev_file_sig.T
                            calibrated_file_sim = self._calibrate_pairwise_binding_score(file_sim)
                            self_sim = torch.diagonal(file_sim, 0)
                            calibrated_self_sim = torch.diagonal(calibrated_file_sim, 0)
                            same_role_file = torch.ones_like(file_sim, dtype=torch.bool)
                            curr_roles = observed.posterior.role_ids
                            prev_roles = getattr(previous_posterior, "role_ids", None)
                            if (
                                curr_roles is not None
                                and prev_roles is not None
                                and curr_roles.numel() >= file_count
                                and prev_roles.numel() >= file_count
                            ):
                                curr_role = curr_roles.to(device=self.device, dtype=torch.long).reshape(-1)[:file_count]
                                prev_role = prev_roles.to(device=self.device, dtype=torch.long).reshape(-1)[:file_count]
                                same_role_file = curr_role[:, None] == prev_role[None, :]
                            same_role_file = same_role_file & ~torch.eye(file_count, device=self.device, dtype=torch.bool)
                            if bool(same_role_file.any().item()):
                                best_other = file_sim.masked_fill(~same_role_file, -1.0).max(dim=-1).values
                                calibrated_best_other = calibrated_file_sim.masked_fill(~same_role_file, -1.0).max(dim=-1).values
                            else:
                                best_other = torch.full_like(self_sim, -1.0)
                                calibrated_best_other = torch.full_like(calibrated_self_sim, -1.0)
                            file_margin = self_sim - best_other
                            file_swap = best_other > (self_sim + 0.05)
                            calibrated_file_margin = calibrated_self_sim - calibrated_best_other
                            calibrated_file_swap = calibrated_best_other > (calibrated_self_sim + 0.05)
                            debug["posterior_file_self_signature_sim_mean"] = float(self_sim.mean().item())
                            debug["posterior_file_best_other_signature_margin_mean"] = float(file_margin.mean().item())
                            debug["posterior_file_potential_swap_rate"] = float(file_swap.to(dtype=self.dtype).mean().item())
                            debug["posterior_file_calibrated_self_signature_sim_mean"] = float(calibrated_self_sim.mean().item())
                            debug["posterior_file_calibrated_best_other_signature_margin_mean"] = float(
                                calibrated_file_margin.mean().item()
                            )
                            debug["posterior_file_calibrated_potential_swap_rate"] = float(
                                calibrated_file_swap.to(dtype=self.dtype).mean().item()
                            )
                            debug["posterior_file_calibrated_signature_score_std"] = float(
                                calibrated_file_sim.std(unbiased=False).item()
                            ) if calibrated_file_sim.numel() > 1 else 0.0

                            active_file = torch.ones((file_count,), device=self.device, dtype=torch.bool)
                            for value in (
                                observed.posterior.file_competition_active,
                                getattr(previous_posterior, "file_competition_active", None),
                            ):
                                if value is not None and value.numel() >= file_count:
                                    active_file = active_file & (
                                        value.to(device=self.device, dtype=self.dtype).reshape(-1)[:file_count] >= 0.5
                                    )
                            current_owner = observed.posterior.lifecycle_owner_reliability
                            previous_owner = getattr(previous_posterior, "lifecycle_owner_reliability", None)
                            owner_threshold = min(max(float(getattr(self.config, "posterior_owner_active_min", 0.25)), 0.0), 1.0)
                            for value in (current_owner, previous_owner):
                                if value is not None and value.numel() >= file_count:
                                    active_file = active_file & (
                                        value.to(device=self.device, dtype=self.dtype).reshape(-1)[:file_count] >= owner_threshold
                                    )
                            for value in (observed.posterior.alpha, getattr(previous_posterior, "alpha", None)):
                                if value is not None and value.numel() >= file_count:
                                    active_file = active_file & (
                                        value.to(device=self.device, dtype=self.dtype).reshape(-1)[:file_count] >= 0.25
                                    )
                            for value in (observed.posterior.support_mass, getattr(previous_posterior, "support_mass", None)):
                                if value is not None and value.numel() >= file_count:
                                    active_file = active_file & (
                                        value.to(device=self.device, dtype=self.dtype).reshape(-1)[:file_count] >= 0.05
                                    )
                            for value in (observed.posterior.recycle_gate, getattr(previous_posterior, "recycle_gate", None)):
                                if value is not None and value.numel() >= file_count:
                                    active_file = active_file & (
                                        value.to(device=self.device, dtype=self.dtype).reshape(-1)[:file_count] <= 0.5
                                    )
                            debug["posterior_active_file_fraction"] = float(active_file.to(dtype=self.dtype).mean().item())
                            if bool(active_file.any().item()):
                                debug["posterior_active_file_self_signature_sim_mean"] = float(self_sim[active_file].mean().item())
                                debug["posterior_active_file_best_other_signature_margin_mean"] = float(
                                    file_margin[active_file].mean().item()
                                )
                                debug["posterior_active_file_potential_swap_rate"] = float(
                                    file_swap[active_file].to(dtype=self.dtype).mean().item()
                                )
                                debug["posterior_active_file_calibrated_self_signature_sim_mean"] = float(
                                    calibrated_self_sim[active_file].mean().item()
                                )
                                debug["posterior_active_file_calibrated_best_other_signature_margin_mean"] = float(
                                    calibrated_file_margin[active_file].mean().item()
                                )
                                debug["posterior_active_file_calibrated_potential_swap_rate"] = float(
                                    calibrated_file_swap[active_file].to(dtype=self.dtype).mean().item()
                                )
        if observed.posterior.recycle_gate is not None and observed.posterior.recycle_gate.numel() > 0:
            recycle = observed.posterior.recycle_gate.to(device=self.device, dtype=self.dtype).reshape(-1)
            debug["posterior_recycle_rate"] = float(recycle.mean().item())
            debug["posterior_recycle_gate_std"] = float(recycle.std(unbiased=False).item()) if recycle.numel() > 1 else 0.0
            debug["posterior_recycle_gate_min"] = float(recycle.min().item())
            debug["posterior_recycle_gate_max"] = float(recycle.max().item())
            role_ids = observed.posterior.role_ids
            if role_ids is not None and role_ids.numel() == recycle.numel():
                roles = role_ids.to(device=self.device, dtype=torch.long).reshape(-1)
                effector_mask = roles == 0
                scene_mask = roles != 0
                if bool(effector_mask.any().item()):
                    debug["posterior_recycle_rate_effector"] = float(recycle[effector_mask].mean().item())
                if bool(scene_mask.any().item()):
                    debug["posterior_recycle_rate_scene"] = float(recycle[scene_mask].mean().item())
            file_active = observed.posterior.file_competition_active
            if file_active is not None and file_active.numel() >= recycle.numel():
                active_mask = file_active.to(device=self.device, dtype=self.dtype).reshape(-1)[: recycle.numel()] >= 0.5
                if bool(active_mask.any().item()):
                    debug["posterior_active_file_recycle_rate"] = float(recycle[active_mask].mean().item())
                inactive_mask = ~active_mask
                if bool(inactive_mask.any().item()):
                    debug["posterior_inactive_file_recycle_rate"] = float(recycle[inactive_mask].mean().item())

            def _debug_tensor_stats(prefix: str, value: torch.Tensor | None) -> None:
                if value is None or value.numel() == 0:
                    return
                tensor = value.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
                debug[f"{prefix}_mean"] = float(tensor.mean().item())
                debug[f"{prefix}_std"] = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
                debug[f"{prefix}_min"] = float(tensor.min().item())
                debug[f"{prefix}_max"] = float(tensor.max().item())

            _debug_tensor_stats("posterior_recycle_logit", observed.posterior.recycle_logits)
            _debug_tensor_stats("posterior_support_mass_raw", observed.posterior.recycle_support_mass_raw)
            _debug_tensor_stats("posterior_support_mass_final", observed.posterior.support_mass)
            _debug_tensor_stats("posterior_prior_var", observed.posterior.recycle_prior_var_mean)
            _debug_tensor_stats("posterior_prior_alpha", observed.posterior.recycle_prior_alpha)
            _debug_tensor_stats("posterior_address_update_rate", observed.posterior.address_update_rate)
            _debug_tensor_stats("posterior_owner_transport_mass", observed.posterior.owner_transport_mass)
            _debug_tensor_stats("posterior_owner_transport_confidence", observed.posterior.owner_transport_confidence)
            _debug_tensor_stats("posterior_owner_transport_dist_to_standard", observed.posterior.owner_transport_dist_to_standard)
            if observed.posterior.owner_transport_applied_fraction is not None:
                debug["posterior_owner_transport_applied_fraction"] = float(
                    observed.posterior.owner_transport_applied_fraction.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            _debug_tensor_stats("posterior_lifecycle_assignment_confidence", observed.posterior.lifecycle_assignment_confidence)
            _debug_tensor_stats("posterior_lifecycle_support_entropy", observed.posterior.lifecycle_support_entropy)
            _debug_tensor_stats("posterior_lifecycle_support_margin", observed.posterior.lifecycle_support_margin)
            _debug_tensor_stats("posterior_lifecycle_owner_reliability", observed.posterior.lifecycle_owner_reliability)
            _debug_tensor_stats("posterior_lifecycle_survival_prob", observed.posterior.lifecycle_survival_prob)
            _debug_tensor_stats("posterior_lifecycle_reset_allowance", observed.posterior.lifecycle_reset_allowance)
            _debug_tensor_stats("posterior_lifecycle_recycle_raw", observed.posterior.lifecycle_recycle_raw)
            _debug_tensor_stats("posterior_file_competition_active", observed.posterior.file_competition_active)
            _debug_tensor_stats("posterior_file_competition_demoted_mass", observed.posterior.file_competition_demoted_mass)
            _debug_tensor_stats("posterior_file_competition_birth_active", observed.posterior.file_competition_birth_active)
            _debug_tensor_stats("posterior_file_competition_birth_share", observed.posterior.file_competition_birth_share)
            if observed.posterior.file_competition_active is not None and observed.posterior.file_competition_active.numel() > 0:
                active = torch.clamp(
                    observed.posterior.file_competition_active.to(device=self.device, dtype=self.dtype).reshape(-1),
                    min=0.0,
                    max=1.0,
                )
                debug["posterior_file_competition_active_count"] = float(active.sum().item())
            if observed.posterior.file_competition_birth_active is not None and observed.posterior.file_competition_birth_active.numel() > 0:
                birth_active = torch.clamp(
                    observed.posterior.file_competition_birth_active.to(device=self.device, dtype=self.dtype).reshape(-1),
                    min=0.0,
                    max=1.0,
                )
                debug["posterior_file_competition_birth_count"] = float(birth_active.sum().item())
            if observed.posterior.file_competition_duplicate_overlap_max is not None:
                debug["posterior_file_competition_duplicate_overlap_max"] = float(
                    observed.posterior.file_competition_duplicate_overlap_max.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.file_competition_active_duplicate_overlap_max is not None:
                debug["posterior_file_competition_active_duplicate_overlap_max"] = float(
                    observed.posterior.file_competition_active_duplicate_overlap_max.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.recycle_residual_summary_norm is not None:
                debug["posterior_residual_summary_norm"] = float(
                    observed.posterior.recycle_residual_summary_norm.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.recycle_dustbin_raw_mass is not None:
                debug["posterior_dustbin_mass_raw"] = float(
                    observed.posterior.recycle_dustbin_raw_mass.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.recycle_dustbin_final_mass is not None:
                debug["posterior_dustbin_mass_final"] = float(
                    observed.posterior.recycle_dustbin_final_mass.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.lifecycle_inactive_dustbin_mass is not None:
                debug["posterior_lifecycle_inactive_dustbin_mass"] = float(
                    observed.posterior.lifecycle_inactive_dustbin_mass.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.lifecycle_unexplained_dustbin_mass is not None:
                debug["posterior_lifecycle_unexplained_dustbin_mass"] = float(
                    observed.posterior.lifecycle_unexplained_dustbin_mass.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
            if observed.posterior.identity_innovation_risk is not None:
                debug["posterior_identity_innovation_risk"] = float(
                    observed.posterior.identity_innovation_risk.to(device=self.device, dtype=self.dtype).reshape(()).item()
                )
        if observed.task_readout.ordinal_active is not None:
            debug["owm_ordinal_active"] = 1.0 if bool(observed.task_readout.ordinal_active.item()) else 0.0
            if observed.task_readout.ordinal_target_rank is not None:
                debug["owm_ordinal_target_rank"] = float(observed.task_readout.ordinal_target_rank.item())
            if observed.task_readout.ordinal_confidence is not None:
                debug["owm_ordinal_confidence"] = float(observed.task_readout.ordinal_confidence.item())
        if predictive.evidence_cache is not None:
            cache = predictive.evidence_cache
            valid = cache.valid.to(dtype=self.dtype)
            debug["owm_evidence_cache_valid_entries"] = float(valid.sum().item())
            debug["owm_evidence_cache_read_weight"] = float(self.config.evidence_cache_read_weight)
            cache_read_active = (
                observed.anchor_prior_graph is not None
                and observed.anchor_prior_graph.cache_priors is not None
                and float(self.config.evidence_cache_read_weight) > 0.0
            )
            debug["owm_evidence_cache_read_active"] = 1.0 if cache_read_active else 0.0
            if bool(cache.valid.any().item()):
                debug["owm_evidence_cache_age_mean"] = float(cache.age[cache.valid].mean().item())
                debug["owm_evidence_cache_uncertainty_mean"] = float(cache.uncertainty[cache.valid].mean().item())
                debug["owm_evidence_cache_innovation_mean"] = float(cache.innovation_at_write[cache.valid].mean().item())
                debug["evidence_cache_age_mean"] = debug["owm_evidence_cache_age_mean"]
                valid_uncertainty = cache.uncertainty[cache.valid].to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
                valid_age = cache.age[cache.valid].to(device=self.device, dtype=self.dtype)
                valid_innovation = torch.clamp(cache.innovation_at_write[cache.valid].to(device=self.device, dtype=self.dtype), min=0.0)
                valid_source = cache.source_ids[cache.valid].to(device=self.device, dtype=torch.long)
                source_factor = torch.where(
                    valid_source == 1,
                    torch.ones_like(valid_uncertainty),
                    torch.full_like(valid_uncertainty, 0.5),
                )
                innovation_cost = float(self.config.evidence_cache_innovation_downweight) * valid_innovation
                trust = source_factor / torch.clamp(1.0 + valid_uncertainty + valid_age + innovation_cost, min=self.config.epsilon_a)
                debug["evidence_cache_trust_mean"] = float(trust.mean().item())
        return PicfCoreOutput(state=state, debug=debug)

    def refresh_predictive_state_for_action(
        self,
        observation: PicfObservation,
        state: PicfCoreState,
        *,
        action_future: torch.Tensor | np.ndarray,
    ) -> PicfPredictiveState:
        observed = _ObservedStepState(
            previous=state,
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
            vl_grounding=state.vl_grounding,
            anchor_prior_graph=state.anchor_prior_graph,
            object_explanation=state.object_explanation,
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
        graph_can_run_without_points = bool(self.config.mapg_enabled) or bool(self.config.aqr_mapg_enabled)
        if not meta.point_contract_ok and not graph_can_run_without_points:
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
        previous: PicfPreviousState | None = None,
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

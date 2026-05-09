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
from openpi.picf.core.contracts import PicfCoreOutput
from openpi.picf.core.contracts import PicfCoreState
from openpi.picf.core.contracts import PicfObservationAnchorState
from openpi.picf.core.contracts import PicfPosteriorAnchorState
from openpi.picf.core.contracts import PicfPredictionCache
from openpi.picf.core.contracts import PicfPreviousState
from openpi.picf.core.contracts import PicfPredictiveState
from openpi.picf.core.contracts import PicfProjectiveGeometryState
from openpi.picf.core.contracts import PicfRecurrentCarryState
from openpi.picf.core.contracts import PicfRecurrentPredictiveState
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
            self.aqr_visual_reader = CrossAttentionRead(
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
            self.aqr_query_self = TransformerStack(
                hidden_dim,
                heads,
                1,
                activation_checkpointing=True,
                ff_chunk_size=self.config.tokenwise_ff_chunk_size,
            )
        else:
            self.aqr_physical_query_tokens = None
            self.aqr_task_query_tokens = None
            self.aqr_role_embedding = None
            self.aqr_type_embedding = None
            self.aqr_coverage_proj = None
            self.aqr_proprio_proj = None
            self.aqr_posterior_summary_proj = None
            self.aqr_task_conditioner = None
            self.aqr_visual_reader = None
            self.aqr_point_reader = None
            self.aqr_tactile_reader = None
            self.aqr_posterior_reader = None
            self.aqr_query_self = None
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
        for row, role in enumerate(roles.tolist()):
            role_int = int(role)
            mask = (post_roles == 0) if role_int == 0 else (post_roles != 0)
            if not bool(mask.any().item()):
                mask = torch.ones((post_count,), device=self.device, dtype=torch.bool)
            bias[row] = torch.where(mask, bias[row], neg[row])
        return bias

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
        active_roles = (roles == 0) | (roles == 2)
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
        point_count = int(token_field.point_tokens.shape[0])
        tactile_count = int(token_field.tactile_tokens.shape[0])
        post_count = 0 if previous is None else int(previous.posterior.tokens.shape[0])
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
                modality_confidence=torch.zeros((anchor_count, 5), device=self.device, dtype=self.dtype),
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
        queries = torch.cat(
            [
                self.aqr_physical_query_tokens[:physical_count],
                self.aqr_task_query_tokens[:task_count],
            ],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
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
        point_bias = self._aqr_point_bias(token_field, roles)
        posterior_bias = self._aqr_posterior_bias(previous, roles)
        tactile_bias = None
        if tactile_count > 0:
            tactile_bias = torch.zeros((anchor_count, tactile_count), device=self.device, dtype=self.dtype)
            tactile_roles = (roles == 0) | (roles == 2)
            tactile_bias = torch.where(tactile_roles[:, None], tactile_bias, torch.full_like(tactile_bias, -2.0))

        visual_priors = torch.zeros((anchor_count, visual_count), device=self.device, dtype=self.dtype)
        point_priors = torch.zeros((anchor_count, point_count), device=self.device, dtype=self.dtype) if point_count > 0 else None
        tactile_priors = torch.zeros((anchor_count, tactile_count), device=self.device, dtype=self.dtype) if tactile_count > 0 else None
        posterior_priors = torch.zeros((anchor_count, post_count), device=self.device, dtype=self.dtype) if post_count > 0 else None
        rounds = max(int(self.config.aqr_query_rounds), 1)
        q = queries[None, :, :]
        for _ in range(rounds):
            if self.aqr_visual_reader is not None and visual_count > 0:
                q, visual_weights = self.aqr_visual_reader(
                    q,
                    token_field.visual_tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=visual_bias,
                )
                visual_priors = self._aqr_competitive_support(visual_weights, eps=self.config.epsilon_a)
            if self.aqr_point_reader is not None and point_count > 0:
                q, point_weights = self.aqr_point_reader(
                    q,
                    token_field.point_tokens.to(device=self.device, dtype=self.dtype)[None, :],
                    attn_bias=point_bias,
                )
                point_priors = self._aqr_competitive_support(point_weights, eps=self.config.epsilon_a)
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
            if self.aqr_query_self is not None:
                q = self.aqr_query_self(q)

        anchor_tokens = q[0]
        visual_conf = _distribution_confidence(visual_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        point_conf = _distribution_confidence(point_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        tactile_conf = _distribution_confidence(tactile_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        post_conf = _distribution_confidence(posterior_priors, eps=self.config.epsilon_a, floor=float(self.config.mapg_confidence_floor))
        zero_conf = torch.zeros((anchor_count,), device=self.device, dtype=self.dtype)
        modality_conf = torch.stack(
            [
                zero_conf,
                zero_conf if visual_conf is None else visual_conf,
                zero_conf if point_conf is None else point_conf,
                zero_conf if tactile_conf is None else tactile_conf,
                zero_conf if post_conf is None else post_conf,
            ],
            dim=-1,
        )
        anchor_scores = torch.max(visual_priors, dim=-1).values
        if point_priors is not None and point_priors.numel() > 0:
            anchor_scores = anchor_scores + torch.max(point_priors, dim=-1).values
        anchor_conf = torch.clamp(modality_conf.max(dim=-1).values, min=0.0, max=1.0)
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
        return PicfAnchorPriorGraphState(
            pg_priors=None,
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
        temperature = max(float(self.config.mapg_assignment_temperature), self.config.epsilon_a)
        mix = min(max(float(self.config.mapg_assignment_quality_uniform_mix), 0.0), 1.0)
        assignment = torch.zeros((slot_count, k), device=self.device, dtype=self.dtype)
        for role in torch.unique(slot_roles).tolist():
            rows = torch.nonzero(slot_roles == int(role), as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            candidate_mask = allowed.index_select(0, rows).any(dim=0)
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
            for row, role in enumerate(query_role_ids.tolist()):
                if int(role) == 1:
                    blocked[row, tactile_start:tactile_end] = True
        if bool(blocked.any().item()):
            bias = bias.masked_fill(blocked, -1.0e4)
        return bias

    def _task_public_role_bias(self, query_role_ids: torch.Tensor, token_field: PicfTokenFieldState) -> torch.Tensor | None:
        fused_bias = self._fused_read_role_bias(query_role_ids, token_field)
        if fused_bias is None:
            if token_field.visual_tokens.shape[0] == 0:
                return None
            return torch.zeros((int(query_role_ids.numel()), int(token_field.visual_tokens.shape[0])), device=self.device, dtype=self.dtype)
        if token_field.visual_tokens.shape[0] == 0:
            return fused_bias
        visual_bias = torch.zeros((fused_bias.shape[0], int(token_field.visual_tokens.shape[0])), device=self.device, dtype=self.dtype)
        return torch.cat([fused_bias, visual_bias], dim=1)

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
        )

    def _build_conditioned_control_state(
        self,
        posterior: PicfPosteriorAnchorState,
        innovation_token: torch.Tensor,
        proprio_token: torch.Tensor,
        task_readout: PicfTaskReadoutState,
        anchor_graph: PicfAnchorPriorGraphState | None = None,
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
            point_pool_ids=point_pool_ids,
            point_positions_world=point_positions_world,
            point_projectable_mask=None,
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
            x = point_weights @ geometry_positions
            S = _weighted_cov(geometry_positions, point_weights, x, self.config)
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
            role_ids=role_ids,
            graph_assignment=graph_assignment,
            graph_point_weights=graph_point_weights,
            graph_visual_weights=graph_visual_weights,
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
        bind_logits = self._binding_logits(h_prior, x_prior, S_prior, obs_anchors)
        role_bias = self._posterior_binding_role_bias(obs_anchors)
        if role_bias is not None:
            bind_logits = bind_logits + role_bias
        vl_bias = self._posterior_vl_binding_bias(obs_anchors, vl_grounding)
        if vl_bias is not None:
            bind_logits = bind_logits + (self._vl_gate(self.vl_posterior_bind_gate_logit, vl_grounding) * vl_bias)
        graph_bias = self._posterior_mapg_binding_bias(obs_anchors, anchor_graph)
        if graph_bias is not None:
            bind_logits = bind_logits + (self._mapg_gate(self.mapg_posterior_gate_logit, anchor_graph) * graph_bias)
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
        slot_token = self.posterior_slot_token.to(device=self.device, dtype=self.dtype)
        query = (self.anchor_seed_proj(anchor_seed) + slot_token)[None, :]
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
        tokens = self.posterior_token_proj(token_in) + slot_token
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
            role_ids=self._posterior_role_ids(),
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
        visual_map = self._visual_map(observation, visual_map_override, meta)
        tactile_bundle = self._tactile_features(observation, meta)
        semantic = self._semantic_context(observation, previous, semantic_override)
        token_field, dense_memory = self._build_token_field(observation, point_context, point_features, visual_map, tactile_bundle, meta, previous)
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
        observation_anchors = self._build_observation_anchors(
            token_field,
            dense_memory,
            vl_grounding=vl_grounding,
            anchor_graph=anchor_prior_graph,
        )
        posterior = self._posterior_update(
            previous,
            observation,
            observation_anchors,
            dense_memory,
            vl_grounding=vl_grounding,
            anchor_graph=anchor_prior_graph,
        )
        current_targets, availability = self._current_targets(observation, local_frame_context, visual_map, dense_memory)
        innovation_token, innovation_norm = self._innovation(previous, current_targets, availability)
        task_readout = self._build_task_readout(
            token_field,
            dense_memory,
            semantic,
            proprio_token,
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
            proprio_token=proprio_token,
            task_readout=task_readout,
            conditioned_control=conditioned_control,
            control=PicfControlState(hold_reason=hold_reason),
            last_prompt=observation.prompt,
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
        visual_map = self._visual_map(observation, visual_map_override, meta)
        tactile_bundle = self._tactile_features(observation, meta)
        token_field, dense_memory = self._build_token_field(
            observation,
            point_context,
            point_features,
            visual_map,
            tactile_bundle,
            meta,
            previous,
        )
        empty_semantic = self._project_semantic_context(tokens_raw=torch.zeros((0, self.config.semantic_dim), device=self.device, dtype=self.dtype))
        anchor_prior_graph = self._build_anchor_prior_graph(
            semantic=empty_semantic,
            token_field=token_field,
            dense_memory=dense_memory,
            previous=previous,
            vl_grounding=None,
        )
        observation_anchors = self._build_observation_anchors(token_field, dense_memory, anchor_graph=anchor_prior_graph)
        posterior = self._posterior_update(previous, observation, observation_anchors, dense_memory, anchor_graph=anchor_prior_graph)
        proprio = _to_tensor(
            np.asarray(observation.proprio if observation.proprio is not None else observation.robot_obs, dtype=np.float32).reshape(-1),
            device=self.device,
            dtype=self.dtype,
        )
        proprio_token = self.proprio_proj(proprio[None, :])[0]
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
            ),
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
            vl_grounding=observed.vl_grounding,
            anchor_prior_graph=observed.anchor_prior_graph,
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
        if observed.vl_grounding is not None:
            debug["vl_grounding_valid"] = 1.0 if bool(observed.vl_grounding.valid.item()) else 0.0
            debug["vl_grounding_confidence"] = float(observed.vl_grounding.confidence.item())
            debug["vl_grounding_anchor_count"] = float(observed.vl_grounding.anchor_point_priors.shape[0])
        if observed.anchor_prior_graph is not None:
            graph = observed.anchor_prior_graph
            debug["mapg_valid"] = 1.0 if bool(graph.valid.item()) else 0.0
            debug["mapg_anchor_count"] = float(graph.anchor_tokens.shape[0])
            debug["mapg_visual_support_mean"] = float(graph.visual_priors.sum(dim=-1).mean().item()) if graph.visual_priors.numel() > 0 else 0.0
            debug["mapg_point_available"] = 1.0 if graph.point_priors is not None else 0.0
            debug["mapg_tactile_available"] = 1.0 if graph.tactile_priors is not None else 0.0
            debug["mapg_posterior_available"] = 1.0 if graph.posterior_priors is not None else 0.0
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
                same_role = graph.anchor_roles[:, None] == graph.anchor_roles[None, :]
                pair_mask = torch.triu(same_role, diagonal=1)
                if bool(pair_mask.any().item()):
                    debug["mapg_same_role_visual_overlap_max"] = float(overlap[pair_mask].max().item())
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
            vl_grounding=state.vl_grounding,
            anchor_prior_graph=state.anchor_prior_graph,
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

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import socket
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "packages" / "openpi-client" / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np
import torch
from openpi_client import base_policy as _base_policy

import picf_core_train as _trainer
from openpi.picf.action_normalization import PicfActionNormalizer
from openpi.picf.contracts import PicfObservation
from openpi.picf.policy import PicfPi05Policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def _as_sensor_names_arg(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
            if isinstance(parsed, (list, tuple)):
                return ",".join(str(item) for item in parsed)
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    raise TypeError(f"Unsupported tactile_sensor_names payload: {type(value).__name__}")


def _as_sensor_offsets_arg(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
            value = parsed
        else:
            return value
    if isinstance(value, (list, tuple)):
        blocks: list[str] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise TypeError(f"Unsupported tactile_sensor_offsets_m item: {item!r}")
            blocks.append(",".join(str(float(component)) for component in item))
        return ";".join(blocks)
    raise TypeError(f"Unsupported tactile_sensor_offsets_m payload: {type(value).__name__}")


def _resolve_checkpoint_dir(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "latest.pt":
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        checkpoint_dir = Path(payload["checkpoint_dir"]).expanduser()
        return checkpoint_dir.parent, checkpoint_dir
    if candidate.is_dir() and (candidate / "model.pt").is_file() and (candidate / "metadata.pt").is_file():
        return candidate.parent, candidate
    if candidate.is_dir() and (candidate / "latest.pt").is_file():
        payload = torch.load(candidate / "latest.pt", map_location="cpu", weights_only=False)
        checkpoint_dir = Path(payload["checkpoint_dir"]).expanduser()
        return candidate, checkpoint_dir
    raise FileNotFoundError(
        f"Could not resolve PICF checkpoint from {candidate}. Expected a step dir with model.pt/metadata.pt "
        "or an output dir containing latest.pt."
    )


def _load_runtime_args(checkpoint_dir: Path) -> argparse.Namespace:
    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    args_dict = dict(metadata["args"])
    if "tactile_sensor_names" in args_dict:
        args_dict["tactile_sensor_names"] = _as_sensor_names_arg(args_dict["tactile_sensor_names"])
    if "tactile_sensor_offsets_m" in args_dict:
        args_dict["tactile_sensor_offsets_m"] = _as_sensor_offsets_arg(args_dict["tactile_sensor_offsets_m"])
    args = argparse.Namespace(**args_dict)
    _trainer._normalize_train_args(args)
    _trainer._validate_train_args(args)
    _trainer._validate_backbone_args(args)
    return args


def _load_model_state_only(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> int:
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_state = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=False)
    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    if bool(getattr(_trainer, "_is_ablated_semantic_only_model_state", lambda _state: False)(model_state)):
        _trainer._load_ablated_semantic_only_model_state(module=module, model_state=model_state)
        return int(metadata.get("step", 0))
    try:
        module.load_state_dict(model_state, strict=True)
    except RuntimeError:
        try:
            _trainer._load_state_dict_picf_compat(module, model_state)
        except RuntimeError:
            try:
                module.core.load_state_dict(model_state, strict=True)
            except RuntimeError:
                _trainer._load_state_dict_picf_compat(module.core, model_state)
    return int(metadata.get("step", 0))


def _visual_override_if_needed(
    trainer: _trainer._PicfWindowTrainer,
    observation: PicfObservation,
) -> torch.Tensor | np.ndarray | None:
    if not trainer.use_visual_override:
        return None
    return _trainer._rgb_visual_override(observation.rgb_static, grid=trainer.visual_grid)


def _tensor_to_list(value: torch.Tensor | None, *, max_rows: int | None = None) -> Any:
    if value is None:
        return None
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if max_rows is not None and tensor.ndim > 0:
        tensor = tensor[: int(max_rows)]
    array = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).numpy()
    return array.tolist()


def _tensor_to_int_list(value: torch.Tensor | None, *, max_rows: int | None = None) -> Any:
    if value is None:
        return None
    tensor = value.detach().to(device="cpu", dtype=torch.long)
    if max_rows is not None and tensor.ndim > 0:
        tensor = tensor[: int(max_rows)]
    return tensor.numpy().tolist()


def _visual_real_grid_payload(value: torch.Tensor | None) -> Any:
    if value is None:
        return None
    tensor = value.detach().to(device="cpu", dtype=torch.float32).flatten()
    if tensor.numel() == 0 or tensor.numel() % 3 != 0:
        return None
    grid = int(round((int(tensor.numel()) // 3) ** 0.5))
    if 3 * grid * grid != int(tensor.numel()):
        return None
    array = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    array = array.reshape(3, grid, grid).permute(1, 2, 0).numpy()
    return array.tolist()


def _prediction_debug_payload(output: Any | None) -> dict[str, Any] | None:
    if output is None or getattr(output, "state", None) is None:
        return None
    predictive = getattr(output.state, "predictive", None)
    if predictive is None:
        return None
    physical_cache = getattr(predictive, "physical_prediction_cache", None)
    conditioned_cache = getattr(predictive, "prediction_cache", None)
    physical_visual = _visual_real_grid_payload(getattr(physical_cache, "visual_real", None))
    conditioned_visual = _visual_real_grid_payload(getattr(conditioned_cache, "visual_real", None))
    if physical_visual is None and conditioned_visual is None:
        return None
    return {
        "visual_real_grid": len(physical_visual or conditioned_visual or []),
        "physical_visual_real": physical_visual,
        "conditioned_visual_real": conditioned_visual,
        "physical_availability": _tensor_to_list(getattr(physical_cache, "availability", None)),
        "conditioned_availability": _tensor_to_list(getattr(conditioned_cache, "availability", None)),
    }


def _weighted_pixels_and_mass(
    point_weights: torch.Tensor,
    point_pixels: torch.Tensor,
    point_visibility: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if point_weights.numel() == 0 or point_pixels.numel() == 0:
        rows = int(point_weights.shape[0]) if point_weights.ndim > 0 else 0
        return (
            torch.zeros((rows, 2), device=point_weights.device, dtype=point_weights.dtype),
            torch.zeros((rows,), device=point_weights.device, dtype=point_weights.dtype),
        )
    count = min(int(point_weights.shape[-1]), int(point_pixels.shape[0]))
    if count <= 0:
        rows = int(point_weights.shape[0]) if point_weights.ndim > 0 else 0
        return (
            torch.zeros((rows, 2), device=point_weights.device, dtype=point_weights.dtype),
            torch.zeros((rows,), device=point_weights.device, dtype=point_weights.dtype),
        )
    weights = torch.clamp(point_weights, min=0.0)
    weights = weights[..., :count]
    pixels = point_pixels[:count].to(device=weights.device, dtype=weights.dtype)
    if point_visibility is not None and point_visibility.numel() > 0:
        visibility = torch.clamp(point_visibility[:count].to(device=weights.device, dtype=weights.dtype), min=0.0, max=1.0)
        weights = weights * visibility[None, :]
    denom = torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
    return (weights @ pixels) / denom, weights.sum(dim=-1)


def _weighted_point_pixels(
    point_weights: torch.Tensor,
    point_pixels: torch.Tensor,
    point_visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    centers, _ = _weighted_pixels_and_mass(point_weights, point_pixels, point_visibility)
    return centers


def _tensor_rows_to_list(value: torch.Tensor | None, valid: torch.Tensor | None = None, *, max_rows: int | None = None) -> Any:
    if value is None:
        return None
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 0:
        return float(tensor.item())
    if max_rows is not None and tensor.ndim > 0:
        tensor = tensor[: int(max_rows)]
    if valid is None:
        return tensor.numpy().tolist()
    valid_cpu = valid.detach().to(device="cpu", dtype=torch.bool)
    if max_rows is not None and valid_cpu.ndim > 0:
        valid_cpu = valid_cpu[: int(max_rows)]
    rows = tensor.numpy().tolist()
    flags = valid_cpu.numpy().tolist()
    if not isinstance(rows, list) or not isinstance(flags, list):
        return rows
    return [row if idx < len(flags) and bool(flags[idx]) else None for idx, row in enumerate(rows)]


def _pixel_ellipse_payload(
    point_weights: torch.Tensor | None,
    point_pixels: torch.Tensor,
    point_visibility: torch.Tensor | None,
    *,
    max_queries: int = 16,
) -> list[dict[str, Any]]:
    if point_weights is None or point_weights.numel() == 0 or point_pixels.numel() == 0:
        return []
    weights = torch.clamp(point_weights.detach().to(dtype=torch.float32), min=0.0)
    pixels = point_pixels.detach().to(device=weights.device, dtype=torch.float32)
    count = min(int(weights.shape[-1]), int(pixels.shape[0]))
    if count <= 0:
        return []
    weights = weights[: int(max_queries), :count]
    pixels = pixels[:count]
    if point_visibility is not None and point_visibility.numel() > 0:
        visibility = torch.clamp(point_visibility.detach().to(device=weights.device, dtype=torch.float32)[:count], min=0.0, max=1.0)
        weights = weights * visibility[None, :]
    payload: list[dict[str, Any]] = []
    for row in range(int(weights.shape[0])):
        row_weights = weights[row]
        mass = torch.sum(row_weights)
        if not torch.isfinite(mass) or float(mass.item()) <= 1e-6:
            payload.append({"valid": False, "visible_mass": float(max(float(mass.item()) if torch.isfinite(mass) else 0.0, 0.0))})
            continue
        probs = row_weights / mass
        center = probs @ pixels
        diff = pixels - center[None, :]
        cov = (diff.T * probs[None, :]) @ diff
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=0.0)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle = torch.atan2(eigvecs[1, 0], eigvecs[0, 0]) * (180.0 / float(np.pi))
        effective_points = 1.0 / torch.clamp(torch.sum(probs**2), min=1e-9)
        payload.append(
            {
                "valid": True,
                "center": center.detach().cpu().tolist(),
                "covariance": cov.detach().cpu().tolist(),
                "axis_lengths_2sigma": (2.0 * torch.sqrt(eigvals)).detach().cpu().tolist(),
                "angle_degrees": float(angle.item()),
                "visible_mass": float(mass.item()),
                "effective_points": float(effective_points.item()),
            }
        )
    return payload


def _attention_summary(
    weights: torch.Tensor | None,
    *,
    topk: int = 5,
    max_queries: int = 8,
    include_dense: bool = False,
    dense_max_queries: int | None = None,
    dense_max_keys: int | None = None,
    dense_max_values: int = 131_072,
    points: torch.Tensor | None = None,
    pixels: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
    pool_ids: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    if weights is None:
        return None
    value = torch.nan_to_num(weights.detach().to(device="cpu", dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        return {"shape": list(value.shape)}
    query_count = min(int(value.shape[0]), int(max_queries))
    key_count = int(value.shape[1])
    if query_count <= 0 or key_count <= 0:
        return {"shape": [int(value.shape[0]), key_count], "entropy": [], "topk": []}
    clipped = torch.clamp(value[:query_count], min=0.0)
    denom = torch.clamp(clipped.sum(dim=-1, keepdim=True), min=1e-9)
    probs = clipped / denom
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-9))).sum(dim=-1)
    if key_count > 1:
        entropy = entropy / float(np.log(key_count))
    k = min(int(topk), key_count)
    top_values, top_indices = torch.topk(probs, k=k, dim=-1)

    point_cpu = None if points is None else points.detach().to(device="cpu", dtype=torch.float32)
    pixel_cpu = None if pixels is None else pixels.detach().to(device="cpu", dtype=torch.float32)
    visibility_cpu = None if visibility is None else visibility.detach().to(device="cpu", dtype=torch.float32)
    pool_cpu = None if pool_ids is None else pool_ids.detach().to(device="cpu", dtype=torch.long)
    top_payload: list[list[dict[str, Any]]] = []
    for query_idx in range(query_count):
        row: list[dict[str, Any]] = []
        for rank in range(k):
            index = int(top_indices[query_idx, rank].item())
            item: dict[str, Any] = {
                "index": index,
                "weight": float(top_values[query_idx, rank].item()),
            }
            if point_cpu is not None and 0 <= index < int(point_cpu.shape[0]):
                item["xyz"] = point_cpu[index].tolist()
            if pixel_cpu is not None and 0 <= index < int(pixel_cpu.shape[0]):
                item["pixel"] = pixel_cpu[index].tolist()
            if visibility_cpu is not None and 0 <= index < int(visibility_cpu.shape[0]):
                item["visibility"] = float(visibility_cpu[index].item())
            if pool_cpu is not None and 0 <= index < int(pool_cpu.shape[0]):
                item["pool_id"] = int(pool_cpu[index].item())
            row.append(item)
        top_payload.append(row)
    payload: dict[str, Any] = {
        "shape": [int(value.shape[0]), key_count],
        "entropy": entropy.tolist(),
        "topk": top_payload,
    }
    if include_dense:
        dense_queries = query_count if dense_max_queries is None else min(query_count, int(dense_max_queries))
        dense_keys = key_count if dense_max_keys is None else min(key_count, int(dense_max_keys))
        dense_values = dense_queries * dense_keys
        if dense_queries > 0 and dense_keys > 0 and dense_values <= int(dense_max_values):
            payload["dense"] = probs[:dense_queries, :dense_keys].tolist()
            payload["dense_shape"] = [int(dense_queries), int(dense_keys)]
        else:
            payload["dense_omitted"] = {
                "dense_queries": int(dense_queries),
                "dense_keys": int(dense_keys),
                "dense_values": int(dense_values),
                "dense_max_values": int(dense_max_values),
            }
    return payload


def _slot_diversity(points: torch.Tensor | None, pixels: torch.Tensor | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, value in (("xyz", points), ("pixel", pixels)):
        if value is None:
            continue
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[0] < 2:
            payload[name] = {"count": int(tensor.shape[0]) if tensor.ndim > 0 else 0}
            continue
        distances = torch.pdist(tensor)
        payload[name] = {
            "count": int(tensor.shape[0]),
            "mean_pairwise_distance": float(distances.mean().item()) if distances.numel() > 0 else 0.0,
            "min_pairwise_distance": float(distances.min().item()) if distances.numel() > 0 else 0.0,
            "max_pairwise_distance": float(distances.max().item()) if distances.numel() > 0 else 0.0,
        }
    return payload


def _near_proprio_point_mass(
    *,
    point_weights: torch.Tensor | None,
    point_positions: torch.Tensor | None,
    proprio: np.ndarray,
    radius_m: float,
) -> Any:
    if point_weights is None or point_positions is None:
        return None
    if point_weights.numel() == 0 or point_positions.numel() == 0 or proprio.shape[0] < 3:
        return None
    weights = torch.clamp(point_weights.detach().to(device="cpu", dtype=torch.float32), min=0.0)
    positions = point_positions.detach().to(device="cpu", dtype=torch.float32)
    count = min(int(weights.shape[-1]), int(positions.shape[0]))
    if count <= 0:
        return None
    weights = weights[..., :count]
    positions = positions[:count]
    denom = torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-9)
    weights = weights / denom
    tcp = torch.as_tensor(np.asarray(proprio[:3], dtype=np.float32), dtype=torch.float32)
    mask = torch.linalg.norm(positions - tcp[None, :], dim=-1) <= float(radius_m)
    return (weights * mask.to(dtype=weights.dtype)[None, :]).sum(dim=-1).tolist()


def _anchor_debug_payload(
    output: Any | None,
    observation: PicfObservation,
    *,
    include_dense_intermediates: bool = False,
) -> dict[str, Any] | None:
    if output is None or getattr(output, "state", None) is None:
        return None
    state = output.state
    token_field = getattr(state, "token_field", None)
    geom = None if token_field is None else getattr(token_field, "projective_geometry", None)
    if geom is None or getattr(geom, "point_proj_grid_index", None) is None:
        return None
    rgb = np.asarray(observation.rgb_static)
    image_h = int(rgb.shape[0]) if rgb.ndim >= 2 else 0
    image_w = int(rgb.shape[1]) if rgb.ndim >= 2 else 0
    visual_grid = geom.visual_grid_index.detach()
    point_grid = geom.point_proj_grid_index.detach()
    point_visibility = getattr(geom, "point_visibility", None)
    point_visibility_t = None if point_visibility is None else point_visibility.detach()
    grid_w = int(torch.max(visual_grid[:, 0]).item()) + 1 if visual_grid.numel() > 0 else 1
    grid_h = int(torch.max(visual_grid[:, 1]).item()) + 1 if visual_grid.numel() > 0 else 1
    scale_x = float(max(image_w - 1, 0)) / float(max(grid_w - 1, 1))
    scale_y = float(max(image_h - 1, 0)) / float(max(grid_h - 1, 1))
    visual_pixels = (
        torch.stack([visual_grid[:, 0] * scale_x, visual_grid[:, 1] * scale_y], dim=-1)
        if visual_grid.numel() > 0
        else torch.zeros((0, 2), device=point_grid.device, dtype=point_grid.dtype)
    )
    if point_grid.numel() == 0:
        point_pixels = torch.zeros((0, 2), device=point_grid.device, dtype=point_grid.dtype)
    else:
        point_pixels = torch.stack([point_grid[:, 0] * scale_x, point_grid[:, 1] * scale_y], dim=-1)
    visible_count = int(torch.count_nonzero(point_visibility_t > 0.5).item()) if point_visibility_t is not None and point_visibility_t.numel() > 0 else 0
    projection_available = bool(point_pixels.numel() > 0 and visible_count > 0)

    observation_anchors = getattr(state, "observation_anchors", None)
    posterior = getattr(state, "posterior", None)
    task_readout = getattr(state, "task_readout", None)
    vl_grounding = getattr(state, "vl_grounding", None)
    anchor_graph = getattr(state, "anchor_prior_graph", None)
    point_positions_world = getattr(token_field, "point_positions_world", None)
    if point_positions_world is None:
        point_positions_world = getattr(token_field, "point_positions", None)
    obs_pixel_raw = None
    obs_pixel = None
    obs_pixel_mass = None
    obs_pixel = (
        _weighted_point_pixels(observation_anchors.point_weights, point_pixels, point_visibility_t)
        if observation_anchors is not None and getattr(observation_anchors, "point_weights", None) is not None
        else None
    )
    if observation_anchors is not None and getattr(observation_anchors, "point_weights", None) is not None:
        obs_pixel_raw = _weighted_point_pixels(observation_anchors.point_weights, point_pixels)
        obs_pixel, obs_pixel_mass = _weighted_pixels_and_mass(observation_anchors.point_weights, point_pixels, point_visibility_t)
    posterior_pixel = None
    posterior_pixel_raw = None
    posterior_pixel_valid = None
    if posterior is not None and obs_pixel is not None and getattr(posterior, "binding", None) is not None:
        binding = torch.clamp(posterior.binding[..., : obs_pixel.shape[0]], min=0.0)
        denom = torch.clamp(binding.sum(dim=-1, keepdim=True), min=1e-6)
        posterior_pixel = (binding @ obs_pixel.to(device=binding.device, dtype=binding.dtype)) / denom
        if obs_pixel_raw is not None:
            posterior_pixel_raw = (binding @ obs_pixel_raw.to(device=binding.device, dtype=binding.dtype)) / denom
        if obs_pixel_mass is not None:
            posterior_mass = binding @ obs_pixel_mass.to(device=binding.device, dtype=binding.dtype)[:, None]
            posterior_pixel_valid = posterior_mass.squeeze(-1) > 1e-6
    task_pixel_raw = None
    task_pixel_mass = None
    task_pixel = (
        _weighted_point_pixels(task_readout.point_weights, point_pixels, point_visibility_t)
        if task_readout is not None and getattr(task_readout, "point_weights", None) is not None
        else None
    )
    if task_readout is not None and getattr(task_readout, "point_weights", None) is not None:
        task_pixel_raw = _weighted_point_pixels(task_readout.point_weights, point_pixels)
        task_pixel, task_pixel_mass = _weighted_pixels_and_mass(task_readout.point_weights, point_pixels, point_visibility_t)
    task_pixel_valid = None if task_pixel_mass is None else task_pixel_mass > 1e-6
    proprio_np = np.asarray(
        observation.proprio if observation.proprio is not None else observation.robot_obs,
        dtype=np.float32,
    )
    task_attention = None
    if task_readout is not None:
        task_attention = {
            "note": (
                "task.pixel is a point-public-attention centroid. It is not the same as semantic or "
                "visual attention; inspect these summaries before interpreting gripper-centric overlays."
            ),
            "local_role_ids": _tensor_to_int_list(getattr(task_readout, "local_role_ids", None), max_rows=64),
            "semantic": _attention_summary(getattr(task_readout, "semantic_attention", None), topk=8, max_queries=16),
            "public": _attention_summary(getattr(task_readout, "public_attention", None), topk=8, max_queries=16),
            "visual_public": _attention_summary(
                getattr(task_readout, "visual_public_attention", None),
                topk=8,
                max_queries=16,
                pixels=visual_pixels,
            ),
            "point_public": _attention_summary(
                getattr(task_readout, "point_public_attention", None),
                topk=8,
                max_queries=16,
                points=point_positions_world,
                pixels=point_pixels,
                visibility=point_visibility_t,
                pool_ids=getattr(token_field, "point_pool_ids", None),
            ),
            "tactile_public": _attention_summary(getattr(task_readout, "tactile_public_attention", None), topk=8, max_queries=16),
            "visual_private": _attention_summary(getattr(task_readout, "visual_private_attention", None), topk=8, max_queries=16),
            "point_private": _attention_summary(getattr(task_readout, "point_private_attention", None), topk=8, max_queries=16),
            "tactile_private": _attention_summary(getattr(task_readout, "tactile_private_attention", None), topk=8, max_queries=16),
            "slot_diversity": _slot_diversity(getattr(task_readout, "x", None), task_pixel),
            "point_pixel_ellipse": _pixel_ellipse_payload(
                getattr(task_readout, "point_weights", None),
                point_pixels,
                point_visibility_t,
                max_queries=16,
            ),
            "visible_point_mass": _tensor_to_list(task_pixel_mass),
            "near_proprio_point_mass_10cm": _near_proprio_point_mass(
                point_weights=getattr(task_readout, "point_weights", None),
                point_positions=point_positions_world,
                proprio=proprio_np,
                radius_m=0.10,
            ),
            "near_proprio_point_mass_20cm": _near_proprio_point_mass(
                point_weights=getattr(task_readout, "point_weights", None),
                point_positions=point_positions_world,
                proprio=proprio_np,
                radius_m=0.20,
            ),
        }

    def _same_role_overlap_max(priors: torch.Tensor | None, roles: torch.Tensor | None) -> float | None:
        if priors is None or roles is None or priors.numel() == 0 or roles.numel() == 0:
            return None
        value = torch.clamp(torch.nan_to_num(priors.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
        if value.ndim != 2 or value.shape[0] < 2:
            return None
        value = value / torch.clamp(value.sum(dim=-1, keepdim=True), min=1e-9)
        overlap = value @ value.T
        diag = torch.clamp(torch.diag(overlap), min=1e-9)
        overlap = overlap / torch.sqrt(torch.clamp(diag[:, None] * diag[None, :], min=1e-9))
        roles_t = roles.detach().to(device=overlap.device, dtype=torch.long)
        pair_mask = torch.triu(roles_t[:, None] == roles_t[None, :], diagonal=1)
        if not bool(pair_mask.any().item()):
            return None
        return float(overlap[pair_mask].max().item())

    mapg_payload = None
    if anchor_graph is not None:
        graph_visual = getattr(anchor_graph, "visual_priors", None)
        graph_point = getattr(anchor_graph, "point_priors", None)
        graph_roles = getattr(anchor_graph, "anchor_roles", None)
        mapg_payload = {
            "valid": bool(getattr(anchor_graph, "valid", torch.tensor(False)).detach().to(device="cpu").item()),
            "anchor_roles": _tensor_to_int_list(graph_roles, max_rows=64),
            "anchor_scores": _tensor_to_list(getattr(anchor_graph, "anchor_scores", None), max_rows=64),
            "anchor_confidence": _tensor_to_list(getattr(anchor_graph, "anchor_confidence", None), max_rows=64),
            "modality_confidence": _tensor_to_list(getattr(anchor_graph, "modality_confidence", None), max_rows=64),
            "obs_assignment": _attention_summary(getattr(anchor_graph, "obs_slot_assignment", None), topk=8, max_queries=32),
            "task_assignment": _attention_summary(getattr(anchor_graph, "task_assignment", None), topk=8, max_queries=32),
            "visual_priors": _attention_summary(
                graph_visual,
                topk=8,
                max_queries=16,
                include_dense=include_dense_intermediates,
                pixels=visual_pixels,
            ),
            "point_priors": _attention_summary(
                graph_point,
                topk=8,
                max_queries=16,
                include_dense=include_dense_intermediates,
                points=point_positions_world,
                pixels=point_pixels,
                visibility=point_visibility_t,
                pool_ids=getattr(token_field, "point_pool_ids", None),
            ),
            "same_role_visual_overlap_max": _same_role_overlap_max(graph_visual, graph_roles),
            "same_role_point_overlap_max": _same_role_overlap_max(graph_point, graph_roles),
        }

    def _vl_heatmap_row(name: str) -> torch.Tensor | None:
        if vl_grounding is None:
            return None
        value = getattr(vl_grounding, name, None)
        if value is None:
            return None
        return value[None, :]

    return {
        "image_hw": [image_h, image_w],
        "segment_id": int(observation.segment_id),
        "step_id": int(observation.step_id),
        "projection": {
            "available": bool(projection_available),
            "point_count": int(point_pixels.shape[0]) if point_pixels.ndim > 0 else 0,
            "visible_point_count": int(visible_count),
            "visual_grid_hw": [int(grid_h), int(grid_w)],
            "note": (
                "task.pixel uses visible point projection. If available=false, image-space overlays are intentionally "
                "invalid instead of falling back to the misleading upper-left zero projection."
            ),
        },
        "observation": {
            "xyz": _tensor_to_list(getattr(observation_anchors, "x", None)),
            "pixel": _tensor_rows_to_list(obs_pixel, None if obs_pixel_mass is None else obs_pixel_mass > 1e-6),
            "pixel_raw_unmasked": _tensor_to_list(obs_pixel_raw),
            "visible_point_mass": _tensor_to_list(obs_pixel_mass),
            "role_ids": _tensor_to_int_list(getattr(observation_anchors, "role_ids", None), max_rows=128),
            "support_point": _tensor_to_list(getattr(observation_anchors, "routing_support_point", None)),
            "support_visual": _tensor_to_list(getattr(observation_anchors, "routing_support_visual", None)),
            "gate_point": _tensor_to_list(getattr(observation_anchors, "routing_gate_point", None)),
            "gate_visual": _tensor_to_list(getattr(observation_anchors, "routing_gate_visual", None)),
            "graph_assignment": _attention_summary(getattr(observation_anchors, "graph_assignment", None), topk=8, max_queries=32),
            "graph_point": _attention_summary(
                getattr(observation_anchors, "graph_point_weights", None),
                topk=8,
                max_queries=32,
                include_dense=include_dense_intermediates,
                points=point_positions_world,
                pixels=point_pixels,
                visibility=point_visibility_t,
                pool_ids=getattr(token_field, "point_pool_ids", None),
            ),
            "graph_visual": _attention_summary(
                getattr(observation_anchors, "graph_visual_weights", None),
                topk=8,
                max_queries=32,
                include_dense=include_dense_intermediates,
                pixels=visual_pixels,
            ),
        },
        "posterior": {
            "xyz": _tensor_to_list(getattr(posterior, "x", None)),
            "pixel": _tensor_rows_to_list(posterior_pixel, posterior_pixel_valid),
            "pixel_raw_unmasked": _tensor_to_list(posterior_pixel_raw),
            "role_ids": _tensor_to_int_list(getattr(posterior, "role_ids", None), max_rows=128),
            "alpha": _tensor_to_list(getattr(posterior, "alpha", None)),
            "support_mass": _tensor_to_list(getattr(posterior, "support_mass", None)),
            "contact_prob": _tensor_to_list(getattr(posterior, "contact_prob", None)),
        },
        "task": {
            "xyz": _tensor_to_list(getattr(task_readout, "x", None)),
            "pixel": _tensor_rows_to_list(task_pixel, task_pixel_valid),
            "pixel_raw_unmasked": _tensor_to_list(task_pixel_raw),
            "visible_point_mass": _tensor_to_list(task_pixel_mass),
            "local_role_ids": _tensor_to_int_list(getattr(task_readout, "local_role_ids", None), max_rows=64),
            "attention": task_attention,
        },
        "vl_grounding": {
            "valid": bool(getattr(vl_grounding, "valid", torch.tensor(False)).detach().to(device="cpu").item())
            if vl_grounding is not None
            else False,
            "confidence": _tensor_to_list(getattr(vl_grounding, "confidence", None)),
            "anchor_roles": _tensor_to_int_list(getattr(vl_grounding, "anchor_roles", None), max_rows=64),
            "anchor_scores": _tensor_to_list(getattr(vl_grounding, "anchor_scores", None), max_rows=64),
            "task_heatmap": _attention_summary(
                _vl_heatmap_row("task_heatmap"),
                topk=12,
                max_queries=1,
                include_dense=include_dense_intermediates,
                pixels=visual_pixels,
            ),
            "effector_heatmap": _attention_summary(
                _vl_heatmap_row("effector_heatmap"),
                topk=12,
                max_queries=1,
                include_dense=include_dense_intermediates,
                pixels=visual_pixels,
            ),
            "interaction_heatmap": _attention_summary(
                _vl_heatmap_row("interaction_heatmap"),
                topk=12,
                max_queries=1,
                include_dense=include_dense_intermediates,
                pixels=visual_pixels,
            ),
            "anchor_point_prior": _attention_summary(
                getattr(vl_grounding, "anchor_point_priors", None),
                topk=8,
                max_queries=16,
                include_dense=include_dense_intermediates,
                points=point_positions_world,
                pixels=point_pixels,
                visibility=point_visibility_t,
                pool_ids=getattr(token_field, "point_pool_ids", None),
            ),
        },
        "mapg": mapg_payload,
        "point_cloud": {
            "xyz": _tensor_to_list(getattr(token_field, "point_positions", None), max_rows=4096),
            "xyz_world": _tensor_to_list(point_positions_world, max_rows=4096),
            "pool_ids": _tensor_to_int_list(getattr(token_field, "point_pool_ids", None), max_rows=4096),
            "projectable_mask": _tensor_to_int_list(getattr(token_field, "point_projectable_mask", None), max_rows=4096),
            "projected_pixel": _tensor_to_list(point_pixels, max_rows=4096),
            "visibility": _tensor_to_list(getattr(geom, "point_visibility", None), max_rows=4096),
            "visual_projected_pixel": _tensor_to_list(visual_pixels, max_rows=4096),
        },
    }


class _PicfCheckpointPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        trainer: _trainer._PicfWindowTrainer,
        *,
        checkpoint_dir: Path,
        checkpoint_step: int,
        action_normalizer: PicfActionNormalizer | None,
        frame_dt_s: float = 1.0 / 30.0,
        export_anchor_debug: bool = False,
        export_anchor_debug_dense: bool = False,
        export_prediction_debug: bool = False,
    ) -> None:
        self._trainer = trainer.eval()
        self._core = trainer.core
        self._semantic_encoder = trainer.semantic_encoder
        self._policy = getattr(
            trainer,
            "policy",
            PicfPi05Policy(
                core=trainer.core,
                semantic_encoder=trainer.semantic_encoder,
                picf_enabled=str(getattr(trainer, "picf_mode", "enabled")).lower().replace("-", "_") == "enabled",
            ),
        )
        self._action_normalizer = action_normalizer
        self._export_anchor_debug = bool(export_anchor_debug)
        self._export_anchor_debug_dense = bool(export_anchor_debug_dense)
        self._export_prediction_debug = bool(export_prediction_debug)
        self._frame_dt_s = float(frame_dt_s)
        self._segment_id = 0
        self._step_id = 0
        self._previous: Any | None = None
        self._last_prompt = ""
        self._metadata = {
            "checkpoint_format": "picf_trainer_v2",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": int(checkpoint_step),
            "action_dim": 7,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _reset_episode(self) -> None:
        self._segment_id += 1
        self._step_id = 0
        self._previous = None

    def _build_observation(self, obs: dict[str, Any], *, reset: bool) -> PicfObservation:
        prompt = str(obs.get("prompt", self._last_prompt))
        self._last_prompt = prompt
        return PicfObservation(
            rgb_static=np.asarray(obs["observation/image"], dtype=np.uint8),
            depth_static=np.asarray(obs["observation/depth"], dtype=np.float32),
            rgb_gripper=None
            if "observation/wrist_image" not in obs or obs["observation/wrist_image"] is None
            else np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
            depth_gripper=None
            if "observation/depth_gripper" not in obs or obs["observation/depth_gripper"] is None
            else np.asarray(obs["observation/depth_gripper"], dtype=np.float32),
            robot_obs=np.asarray(obs["observation/state"], dtype=np.float32),
            prompt=prompt,
            step_id=int(self._step_id),
            segment_id=int(self._segment_id),
            timestamp_s=float(self._step_id) * self._frame_dt_s,
            reset_scaffold=bool(reset),
            proprio=np.asarray(obs["observation/state"], dtype=np.float32),
        )

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        reset = bool(obs.get("openpi/reset", False))
        if reset:
            self._reset_episode()
        observation = self._build_observation(obs, reset=reset)
        visual_override = _visual_override_if_needed(self._trainer, observation)
        with torch.inference_mode():
            act_result = self._policy.act(
                observation,
                previous=self._previous,
                visual_map_override=visual_override,
            )
        output = act_result.output
        self._previous = None if output is None else output.state
        action = act_result.action.detach().to(device="cpu", dtype=torch.float32).numpy()
        if self._action_normalizer is not None:
            action = self._action_normalizer.unnormalize_np(action)
        self._step_id += 1
        response = {
            "actions": action[None, :],
            "debug": act_result.debug,
        }
        if self._export_anchor_debug:
            anchor_debug = _anchor_debug_payload(
                output,
                observation,
                include_dense_intermediates=self._export_anchor_debug_dense,
            )
            if anchor_debug is not None:
                response["anchor_debug"] = anchor_debug
        if self._export_prediction_debug:
            prediction_debug = _prediction_debug_payload(output)
            if prediction_debug is not None:
                response["prediction_debug"] = prediction_debug
        return response


def _build_policy(
    *,
    checkpoint_path: Path,
    device: torch.device,
    picf_mode_override: str | None = None,
    export_anchor_debug: bool = False,
    export_anchor_debug_dense: bool = False,
    export_prediction_debug: bool = False,
) -> _PicfCheckpointPolicy:
    output_dir, checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
    args = _load_runtime_args(checkpoint_dir)
    if picf_mode_override is not None:
        args.picf_mode = str(picf_mode_override).lower().replace("-", "_")
        _trainer._normalize_train_args(args)
        _trainer._validate_train_args(args)
        _trainer._validate_backbone_args(args)
    args.device = str(device)
    core, semantic_encoder, use_visual_override = _trainer._build_model(args, device=device)
    trainer = _trainer._PicfWindowTrainer(
        core,
        semantic_encoder=semantic_encoder,
        visual_grid=int(args.visual_grid),
        use_visual_override=use_visual_override,
        loss_config=_trainer._build_loss_config(args),
        picf_mode=getattr(args, "picf_mode", "enabled"),
    ).to(device)
    action_normalizer = _trainer._resolve_action_normalizer(args)
    backgrounds = _trainer._load_tactile_backgrounds_npz(getattr(args, "tactile_backgrounds_path", None))
    source = _trainer._CalvinTransitionSource(
        args.calvin_root,
        split=args.split,
        backend=args.backend,
        unroll_steps=args.unroll_steps,
        use_wrist_rgb=True,
        use_tactile=bool(args.tactile_mode == "encoder"),
        tactile_sensor_names=args.tactile_sensor_names,
        tactile_sensor_offsets_m=args.tactile_sensor_offsets_m,
        tactile_calibration=getattr(args, "tactile_calibration_path", None),
        tactile_backgrounds_by_sensor=backgrounds,
        use_scene_obs=True,
        action_normalizer=action_normalizer,
    )
    try:
        _trainer._materialize_model_parameters(trainer, source=source, rank=0)
    finally:
        source.close()
    checkpoint_step = _load_model_state_only(checkpoint_dir=checkpoint_dir, model=trainer, device=device)
    return _PicfCheckpointPolicy(
        trainer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=checkpoint_step,
        action_normalizer=action_normalizer,
        export_anchor_debug=export_anchor_debug,
        export_anchor_debug_dense=export_anchor_debug_dense,
        export_prediction_debug=export_prediction_debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a PICF trainer checkpoint over websocket.")
    parser.add_argument("--checkpoint", required=True, help="PICF checkpoint dir or output dir containing latest.pt.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--picf-mode",
        choices=["enabled", "ablated"],
        default=None,
        help=(
            "Optional runtime override. By default serving uses the checkpoint's saved "
            "picf_mode. Use 'ablated' to force PI0.5-only action serving without PICF "
            "recurrent/control/future branches."
        ),
    )
    parser.add_argument(
        "--export-anchor-debug",
        action="store_true",
        default=os.environ.get("OPENPI_PICF_EXPORT_ANCHORS", "0").lower() in {"1", "true", "yes", "on"},
        help="Return compact PICF anchor projection and point-cloud debug payloads with each websocket action.",
    )
    parser.add_argument(
        "--export-anchor-debug-dense",
        action="store_true",
        default=os.environ.get("OPENPI_PICF_EXPORT_ANCHOR_DENSE", "0").lower() in {"1", "true", "yes", "on"},
        help=(
            "Include dense MAPG/VL intermediate arrays in anchor_debug. This is intended for short diagnostic "
            "CALVIN runs because it substantially increases JSONL size."
        ),
    )
    parser.add_argument(
        "--export-prediction-debug",
        action="store_true",
        default=os.environ.get("OPENPI_PICF_EXPORT_PREDICTIONS", "0").lower() in {"1", "true", "yes", "on"},
        help="Return compact 4x4 visual-real predictive-cache payloads with each websocket action.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    policy = _build_policy(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        picf_mode_override=args.picf_mode,
        export_anchor_debug=bool(args.export_anchor_debug),
        export_anchor_debug_dense=bool(args.export_anchor_debug_dense),
        export_prediction_debug=bool(args.export_prediction_debug),
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating PICF server (host: %s, ip: %s)", hostname, local_ip)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

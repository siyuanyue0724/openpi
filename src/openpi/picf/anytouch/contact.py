from __future__ import annotations

import math

import torch


def rot6d(rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)


def so3_log_map(rotation: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, min=-1.0, max=1.0)
    theta = torch.arccos(cos_theta)
    vee = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(
        theta.abs() < eps,
        0.5 - (theta * theta) / 12.0,
        theta / (2.0 * torch.clamp(sin_theta, min=eps)),
    )
    return scale[..., None] * vee


def pose6d(transform: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = so3_log_map(rotation, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1)
    omega_hat = _hat(omega)
    batch_shape = rotation.shape[:-2]
    eye = torch.eye(3, dtype=transform.dtype, device=transform.device)
    if batch_shape:
        eye = eye.expand(*batch_shape, 3, 3)
    v_inv = (
        eye
        - 0.5 * omega_hat
        + _series_coeff(theta, eps=eps)[..., None, None] * (omega_hat @ omega_hat)
    )
    rho = torch.einsum("...ij,...j->...i", v_inv, translation)
    return torch.cat([rho, omega], dim=-1)


def contact_normal(
    contact_pose: torch.Tensor,
    force_vec: torch.Tensor | None,
    *,
    epsilon_force: float,
    pose_normal_available: bool = True,
) -> torch.Tensor:
    normal = contact_pose[:3, 2]
    if pose_normal_available and torch.linalg.norm(normal) > epsilon_force:
        return normal / torch.clamp(torch.linalg.norm(normal), min=epsilon_force)
    if force_vec is not None:
        force = force_vec[:3]
        norm = torch.linalg.norm(force)
        if norm > epsilon_force:
            return force / norm
    return torch.tensor([0.0, 0.0, 1.0], device=contact_pose.device, dtype=contact_pose.dtype)


def explicit_contact_observation(
    *,
    force_vec: torch.Tensor | None,
    indent_depth_m: float | None,
    tactile_pressure: float | None,
    tau_force_n: float,
    tau_indent_m: float,
    tau_tactile_pressure: float,
) -> bool | None:
    clauses: list[bool] = []
    if force_vec is not None:
        clauses.append(bool(torch.linalg.norm(force_vec[:3]).item() > tau_force_n))
    if indent_depth_m is not None:
        clauses.append(float(indent_depth_m) > tau_indent_m)
    if tactile_pressure is not None:
        clauses.append(float(tactile_pressure) > tau_tactile_pressure)
    if not clauses:
        return None
    return any(clauses)


def _hat(omega: torch.Tensor) -> torch.Tensor:
    ox, oy, oz = omega.unbind(dim=-1)
    zeros = torch.zeros_like(ox)
    return torch.stack(
        [
            torch.stack([zeros, -oz, oy], dim=-1),
            torch.stack([oz, zeros, -ox], dim=-1),
            torch.stack([-oy, ox, zeros], dim=-1),
        ],
        dim=-2,
    )


def _series_coeff(theta: torch.Tensor, *, eps: float) -> torch.Tensor:
    theta_sq = theta * theta
    denom = 2.0 * (1.0 - torch.cos(theta))
    numer = 1.0 - (theta * torch.sin(theta)) / torch.clamp(denom, min=eps)
    coeff = numer / torch.clamp(theta_sq, min=eps)
    series = (1.0 / 12.0) + theta_sq / 720.0
    return torch.where(theta < math.sqrt(eps), series, coeff)

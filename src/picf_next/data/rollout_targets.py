"""Dataset-neutral loss-side targets for bounded object-state overshooting.

Dataset adapters resolve future demonstrator controls and independently
calibrated physical geometry. This module only validates and batches those
records; it never reads model predictions or deploy-visible evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from picf_next.geometry import PhysicalGeometryContract


@dataclass(frozen=True, slots=True)
class ObjectGeometryRolloutTarget:
    """Loss-only future controls and physical geometry for one batch.

    The object axis is a padded target-side inventory. ``identity_keys`` has
    shape batch-by-horizon-by-target-object and uses ``None`` for padding.
    Geometry missingness is coordinate-wise. Unused or unsupervised values are
    exactly zero, making accidental consumption detectable. These tensors are
    never inputs to discovery, runtime association, the current action expert
    or the committed posterior.
    """

    executed_actions: torch.Tensor
    delta_t_s: torch.Tensor
    step_valid: torch.Tensor
    identity_keys: tuple[tuple[tuple[str | None, ...], ...], ...]
    geometry: torch.Tensor
    geometry_variance: torch.Tensor
    geometry_supervised: torch.Tensor
    geometry_contract: PhysicalGeometryContract


@dataclass(frozen=True, slots=True)
class PhysicalObjectGeometryFrame:
    """Selective physical geometry at one future observation time."""

    identity_keys: tuple[str, ...]
    geometry: torch.Tensor
    geometry_variance: torch.Tensor
    geometry_supervised: torch.Tensor
    geometry_contract: PhysicalGeometryContract


@dataclass(frozen=True, slots=True)
class ObjectGeometryRolloutSample:
    """One contiguous loss-side rollout beginning after the current frame.

    Action row ``h`` is the command executed from state ``t+h`` to
    ``t+h+1``. Geometry frame ``h`` describes the independently labelled state
    at ``t+h+1``. Frames may contain no selected geometry target, but actions
    may not skip an intermediate transition.
    """

    executed_actions: torch.Tensor
    delta_t_s: torch.Tensor
    geometry_frames: tuple[PhysicalObjectGeometryFrame, ...]


def _validate_frame(
    frame: PhysicalObjectGeometryFrame,
    *,
    geometry_contract: PhysicalGeometryContract,
) -> None:
    if not isinstance(frame, PhysicalObjectGeometryFrame):
        raise TypeError("rollout geometry frames must use PhysicalObjectGeometryFrame")
    keys = frame.identity_keys
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("future physical identity keys must be nonempty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("future physical identity keys must be unique within one frame")
    if frame.geometry_contract != geometry_contract:
        raise ValueError("future frame and rollout geometry contracts differ")
    expected = (len(keys), geometry_contract.dimension)
    geometry = frame.geometry
    variance = frame.geometry_variance
    supervised = frame.geometry_supervised
    if geometry.shape != expected or variance.shape != expected:
        raise ValueError("future geometry and variance must be object-by-geometry")
    if supervised.shape != expected or supervised.dtype != torch.bool:
        raise ValueError("future geometry supervision must be bool object-by-geometry")
    if supervised.requires_grad:
        raise ValueError("future geometry supervision must be detached")
    for name, value in (("geometry", geometry), ("geometry variance", variance)):
        if not torch.is_floating_point(value) or value.requires_grad:
            raise ValueError(f"future {name} must be floating and detached")
        if not torch.isfinite(value).all():
            raise ValueError(f"future {name} contains NaN or infinity")
    if (variance < 0.0).any():
        raise ValueError("future geometry variance cannot be negative")
    if (geometry[~supervised] != 0.0).any() or (variance[~supervised] != 0.0).any():
        raise ValueError("unknown future geometry coordinates must be exactly zero")
    if keys and not supervised.any(dim=-1).all():
        raise ValueError("every future physical key requires at least one geometry coordinate")


def _validate_sample(
    sample: ObjectGeometryRolloutSample,
    *,
    action_dim: int,
    geometry_contract: PhysicalGeometryContract,
) -> None:
    if not isinstance(sample, ObjectGeometryRolloutSample):
        raise TypeError("rollout samples must use ObjectGeometryRolloutSample")
    actions = sample.executed_actions
    delta_t = sample.delta_t_s
    if (
        actions.ndim != 2
        or actions.shape[0] <= 0
        or actions.shape[1] != action_dim
        or not torch.is_floating_point(actions)
        or actions.requires_grad
        or not torch.isfinite(actions).all()
    ):
        raise ValueError("rollout sample actions must be finite detached horizon-by-action")
    horizon = actions.shape[0]
    if (
        delta_t.shape != (horizon,)
        or not torch.is_floating_point(delta_t)
        or delta_t.requires_grad
        or not torch.isfinite(delta_t).all()
        or (delta_t <= 0.0).any()
    ):
        raise ValueError("rollout sample delta_t must be finite positive and detached")
    if len(sample.geometry_frames) != horizon:
        raise ValueError("each rollout action must have one aligned future geometry frame")
    for frame in sample.geometry_frames:
        _validate_frame(frame, geometry_contract=geometry_contract)


def build_object_geometry_rollout_target(
    samples: tuple[ObjectGeometryRolloutSample, ...],
    *,
    action_dim: int,
    geometry_contract: PhysicalGeometryContract,
    device: torch.device | str,
    input_dtype: torch.dtype,
    target_dtype: torch.dtype,
) -> ObjectGeometryRolloutTarget:
    """Validate and pad future trajectories without conflating numeric planes.

    Actions and delta time are transition inputs and therefore use
    ``input_dtype``. Physical geometry and variance are loss-only labels and
    use ``target_dtype``. This distinction is required under mixed precision:
    transition inputs must match the posterior, while labels retain float32
    fidelity until the explicit loss boundary.
    """

    if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
        raise ValueError("rollout action dimension must be positive")
    if not isinstance(geometry_contract, PhysicalGeometryContract):
        raise TypeError("rollout target requires a physical geometry contract")
    geometry_dim = geometry_contract.dimension
    if not samples:
        raise ValueError("geometry rollout target requires at least one sample")
    if not input_dtype.is_floating_point:
        raise ValueError("geometry rollout input dtype must be floating point")
    if not target_dtype.is_floating_point:
        raise ValueError("geometry rollout supervision dtype must be floating point")
    for sample in samples:
        _validate_sample(
            sample,
            action_dim=action_dim,
            geometry_contract=geometry_contract,
        )

    batch_size = len(samples)
    horizon = max(sample.executed_actions.shape[0] for sample in samples)
    target_capacity = max(
        (len(frame.identity_keys) for sample in samples for frame in sample.geometry_frames),
        default=0,
    )
    if target_capacity <= 0:
        raise ValueError("geometry rollout batch contains no supervised physical object")
    input_factory = {"device": device, "dtype": input_dtype}
    target_factory = {"device": device, "dtype": target_dtype}
    actions = torch.zeros(batch_size, horizon, action_dim, **input_factory)
    delta_t = torch.zeros(batch_size, horizon, **input_factory)
    step_valid = torch.zeros(batch_size, horizon, device=device, dtype=torch.bool)
    geometry = torch.zeros(
        batch_size,
        horizon,
        target_capacity,
        geometry_dim,
        **target_factory,
    )
    variance = torch.zeros_like(geometry)
    supervised = torch.zeros_like(geometry, dtype=torch.bool)
    keys: list[list[tuple[str | None, ...]]] = [
        [(None,) * target_capacity for _step in range(horizon)] for _sample in range(batch_size)
    ]

    for batch_index, sample in enumerate(samples):
        sample_horizon = sample.executed_actions.shape[0]
        actions[batch_index, :sample_horizon] = sample.executed_actions.to(
            device=device,
            dtype=input_dtype,
        )
        delta_t[batch_index, :sample_horizon] = sample.delta_t_s.to(
            device=device,
            dtype=input_dtype,
        )
        step_valid[batch_index, :sample_horizon] = True
        for horizon_index, frame in enumerate(sample.geometry_frames):
            count = len(frame.identity_keys)
            if count == 0:
                continue
            geometry[batch_index, horizon_index, :count] = frame.geometry.to(
                device=device,
                dtype=target_dtype,
            )
            variance[batch_index, horizon_index, :count] = frame.geometry_variance.to(
                device=device,
                dtype=target_dtype,
            )
            supervised[batch_index, horizon_index, :count] = frame.geometry_supervised.to(
                device=device,
            )
            keys[batch_index][horizon_index] = (
                *frame.identity_keys,
                *((None,) * (target_capacity - count)),
            )

    return ObjectGeometryRolloutTarget(
        executed_actions=actions,
        delta_t_s=delta_t,
        step_valid=step_valid,
        identity_keys=tuple(tuple(sample) for sample in keys),
        geometry=geometry,
        geometry_variance=variance,
        geometry_supervised=supervised,
        geometry_contract=geometry_contract,
    )

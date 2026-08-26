"""Loss-side CALVIN indexing for bounded object-geometry overshooting.

The geometry provider is deliberately external: it may decode simulator state
or a versioned sidecar, but none of that privileged payload enters the runtime
CALVIN record. This module only establishes the exact action/target chronology.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.rollout_targets import (
    ObjectGeometryRolloutSample,
    PhysicalObjectGeometryFrame,
)
from picf_next.geometry import PhysicalGeometryContract

CalvinPhysicalGeometryProvider = Callable[[int, int], PhysicalObjectGeometryFrame]
CalvinSourcePhysicalGeometryProvider = Callable[[int], PhysicalObjectGeometryFrame]


def _empty_geometry_frame(
    geometry_contract: PhysicalGeometryContract,
) -> PhysicalObjectGeometryFrame:
    geometry_dim = geometry_contract.dimension
    return PhysicalObjectGeometryFrame(
        identity_keys=(),
        geometry=torch.zeros(0, geometry_dim),
        geometry_variance=torch.zeros(0, geometry_dim),
        geometry_supervised=torch.zeros(0, geometry_dim, dtype=torch.bool),
        geometry_contract=geometry_contract,
    )


def build_calvin_geometry_rollout_sample(
    index: CalvinDatasetIndex,
    *,
    segment_index: int,
    global_index: int,
    maximum_horizon: int,
    supervised_horizons: Sequence[int],
    geometry_contract: PhysicalGeometryContract,
    geometry_provider: CalvinPhysicalGeometryProvider,
) -> ObjectGeometryRolloutSample:
    """Resolve one contiguous CALVIN rollout without crossing a task segment.

    For one-based horizon ``h``, command ``global_index + h - 1`` causes the
    transition and geometry frame ``global_index + h`` is its target. Every
    intermediate command is retained even when only sparse horizons carry a
    geometry label.
    """

    if not isinstance(index, CalvinDatasetIndex):
        raise TypeError("CALVIN rollout indexing requires a CalvinDatasetIndex")
    if (
        not isinstance(maximum_horizon, int)
        or isinstance(maximum_horizon, bool)
        or maximum_horizon <= 0
    ):
        raise ValueError("CALVIN rollout horizon must be positive")
    if not isinstance(geometry_contract, PhysicalGeometryContract):
        raise TypeError("CALVIN rollout requires a physical geometry contract")
    if not callable(geometry_provider):
        raise TypeError("CALVIN rollout geometry provider must be callable")
    frozen_horizons = tuple(supervised_horizons)
    if (
        not frozen_horizons
        or any(
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or not 1 <= horizon <= maximum_horizon
            for horizon in frozen_horizons
        )
        or len(set(frozen_horizons)) != len(frozen_horizons)
        or tuple(sorted(frozen_horizons)) != frozen_horizons
    ):
        raise ValueError("CALVIN supervised horizons must be unique sorted one-based indices")
    if frozen_horizons[0] != 1:
        raise ValueError("CALVIN rollout supervision must include horizon one")
    if (
        not isinstance(segment_index, int)
        or isinstance(segment_index, bool)
        or not 0 <= segment_index < len(index.segments)
    ):
        raise ValueError("CALVIN rollout segment index is invalid")
    segment = index.segments[segment_index]
    if (
        not isinstance(global_index, int)
        or isinstance(global_index, bool)
        or not segment.start <= global_index < segment.end
    ):
        raise ValueError("CALVIN rollout start must be a valid transition in the segment")

    available_horizon = min(maximum_horizon, segment.end - global_index)
    actions = torch.as_tensor(
        np.stack(
            [index.action(global_index + offset) for offset in range(available_horizon)]
        ).copy(),
        dtype=torch.float32,
    )
    delta_t = torch.full(
        (available_horizon,),
        1.0 / float(index.control_hz),
        dtype=torch.float32,
    )
    selected = set(frozen_horizons)
    geometry_frames = []
    for one_based_horizon in range(1, available_horizon + 1):
        if one_based_horizon in selected:
            frame = geometry_provider(segment_index, global_index + one_based_horizon)
            if not isinstance(frame, PhysicalObjectGeometryFrame):
                raise TypeError("CALVIN geometry provider returned an invalid frame")
            if frame.geometry_contract != geometry_contract:
                raise ValueError("CALVIN provider and rollout geometry contracts differ")
            geometry_frames.append(frame)
        else:
            geometry_frames.append(_empty_geometry_frame(geometry_contract))
    return ObjectGeometryRolloutSample(
        executed_actions=actions,
        delta_t_s=delta_t,
        geometry_frames=tuple(geometry_frames),
    )


def build_calvin_source_geometry_rollout_sample(
    index: CalvinDatasetIndex,
    *,
    global_index: int,
    maximum_horizon: int,
    supervised_horizons: Sequence[int],
    geometry_contract: PhysicalGeometryContract,
    geometry_provider: CalvinSourcePhysicalGeometryProvider,
) -> ObjectGeometryRolloutSample:
    """Resolve a task-independent rollout without crossing a source episode.

    Language segments are annotations and can overlap or stop inside one
    physical trajectory.  Stage-B posterior training therefore follows the
    canonical source episode instead of inventing a task-conditioned reset.
    Command ``t+h-1`` remains paired with physical geometry at frame ``t+h``.
    """

    if not isinstance(index, CalvinDatasetIndex):
        raise TypeError("CALVIN source rollout indexing requires a CalvinDatasetIndex")
    if (
        not isinstance(maximum_horizon, int)
        or isinstance(maximum_horizon, bool)
        or maximum_horizon <= 0
    ):
        raise ValueError("CALVIN source rollout horizon must be positive")
    if not isinstance(geometry_contract, PhysicalGeometryContract):
        raise TypeError("CALVIN source rollout requires a physical geometry contract")
    if not callable(geometry_provider):
        raise TypeError("CALVIN source rollout geometry provider must be callable")
    frozen_horizons = tuple(supervised_horizons)
    if (
        not frozen_horizons
        or any(
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or not 1 <= horizon <= maximum_horizon
            for horizon in frozen_horizons
        )
        or len(set(frozen_horizons)) != len(frozen_horizons)
        or tuple(sorted(frozen_horizons)) != frozen_horizons
        or frozen_horizons[0] != 1
    ):
        raise ValueError(
            "CALVIN source supervised horizons must be unique, sorted, include one, "
            "and lie within the maximum horizon"
        )
    episode = index.source_episode(global_index)
    if global_index >= episode.end:
        raise ValueError("CALVIN source rollout start must have a next frame in its episode")

    available_horizon = min(maximum_horizon, episode.end - global_index)
    actions = torch.as_tensor(
        np.stack(
            [index.action(global_index + offset) for offset in range(available_horizon)]
        ).copy(),
        dtype=torch.float32,
    )
    delta_t = torch.full(
        (available_horizon,),
        1.0 / float(index.control_hz),
        dtype=torch.float32,
    )
    selected = set(frozen_horizons)
    geometry_frames = []
    for one_based_horizon in range(1, available_horizon + 1):
        if one_based_horizon in selected:
            frame = geometry_provider(global_index + one_based_horizon)
            if not isinstance(frame, PhysicalObjectGeometryFrame):
                raise TypeError("CALVIN source geometry provider returned an invalid frame")
            if frame.geometry_contract != geometry_contract:
                raise ValueError("CALVIN source provider and rollout geometry contracts differ")
            geometry_frames.append(frame)
        else:
            geometry_frames.append(_empty_geometry_frame(geometry_contract))
    return ObjectGeometryRolloutSample(
        executed_actions=actions,
        delta_t_s=delta_t,
        geometry_frames=tuple(geometry_frames),
    )

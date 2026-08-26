"""Loss-only calibration primitives for CALVIN physical ownership."""

from __future__ import annotations

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION,
)


def calvin_depth_consistent_supervision(
    source_depth_m: np.ndarray,
    rendered_depth_m: np.ndarray,
) -> np.ndarray:
    """Return pixels whose restored z-buffer agrees with the archived sensor.

    A rendered owner is a valid label only when it is visible in the archived
    observation. Depth disagreement therefore means unknown supervision rather
    than context. The fixed tolerance is part of the versioned sidecar schema.
    """

    source = np.asarray(source_depth_m)
    rendered = np.asarray(rendered_depth_m)
    if (
        source.ndim != 2
        or rendered.shape != source.shape
        or not np.issubdtype(source.dtype, np.floating)
        or not np.issubdtype(rendered.dtype, np.floating)
        or not np.isfinite(source).all()
        or not np.isfinite(rendered).all()
        or (source <= 0.0).any()
        or (rendered <= 0.0).any()
    ):
        raise ContractError("CALVIN depth-consistency inputs are invalid")
    tolerance_m = float(CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION["maximum_absolute_depth_error_m"])
    supervised = np.abs(source.astype(np.float32) - rendered.astype(np.float32)) <= tolerance_m
    supervised.setflags(write=False)
    return supervised


def calvin_depth_consistent_fraction(supervised: np.ndarray) -> float:
    mask = np.asarray(supervised)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size:
        raise ContractError("CALVIN depth-consistency mask is invalid")
    return float(mask.mean())

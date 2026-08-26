"""Shared explicit physical-coordinate contracts for synthetic tests."""

from __future__ import annotations

from picf_next.geometry import PhysicalGeometryContract


def synthetic_geometry_contract(
    dimension: int,
    *,
    name: str = "picf.synthetic-position.v1",
) -> PhysicalGeometryContract:
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("synthetic geometry dimension must be positive")
    axes = tuple(f"axis_{index}" for index in range(dimension))
    return PhysicalGeometryContract(
        name=name,
        quantity="synthetic_cartesian_position",
        reference_frame="synthetic_world",
        axes=axes,
        units=("m",) * dimension,
        normalization_offset=(0.0,) * dimension,
        normalization_scale=(1.0,) * dimension,
    )

from __future__ import annotations

import json

import pytest

from picf_next.geometry import PhysicalGeometryContract


def _contract(**overrides: object) -> PhysicalGeometryContract:
    values: dict[str, object] = {
        "name": "calvin.object-aabb-centre.robot-base.v1",
        "quantity": "object_aabb_centre",
        "reference_frame": "robot_base",
        "axes": ("x", "y", "z"),
        "units": ("m", "m", "m"),
        "normalization_offset": (0.0, 0.0, 0.0),
        "normalization_scale": (0.5, 0.5, 0.5),
    }
    values.update(overrides)
    return PhysicalGeometryContract(**values)  # type: ignore[arg-type]


def test_physical_geometry_contract_normalizes_and_denormalizes_values_and_variance() -> None:
    contract = _contract(
        normalization_offset=(0.1, -0.2, 0.3),
        normalization_scale=(0.5, 0.25, 2.0),
    )

    assert contract.dimension == 3
    assert contract.normalize_values((0.6, 0.05, 2.3)) == pytest.approx((1.0, 1.0, 1.0))
    assert contract.normalize_variance((0.25, 0.0625, 4.0)) == pytest.approx((1.0, 1.0, 1.0))
    assert contract.denormalize_values((1.0, 1.0, 1.0)) == pytest.approx((0.6, 0.05, 2.3))
    assert contract.denormalize_variance((1.0, 1.0, 1.0)) == pytest.approx((0.25, 0.0625, 4.0))


def test_physical_geometry_contract_fingerprint_covers_semantics_and_normalization() -> None:
    base = _contract()

    assert base.fingerprint == _contract().fingerprint
    assert base.fingerprint != _contract(quantity="object_centre_of_mass").fingerprint
    assert base.fingerprint != _contract(reference_frame="camera_front").fingerprint
    assert base.fingerprint != _contract(normalization_scale=(1.0, 1.0, 1.0)).fingerprint


def test_physical_geometry_contract_from_dict_rejects_scalar_type_confusion() -> None:
    contract = _contract()
    assert PhysicalGeometryContract.from_dict(contract.to_dict()) == contract
    round_trip = json.loads(json.dumps(contract.to_dict()))
    assert PhysicalGeometryContract.from_dict(round_trip) == contract

    payload = contract.to_dict()
    payload["normalization_scale"] = [True, 0.5, 0.5]
    with pytest.raises(ValueError, match=r"normalization_scale\[0\] must be a finite number"):
        PhysicalGeometryContract.from_dict(payload)

    payload = contract.to_dict()
    payload["axes"] = "xyz"
    with pytest.raises(ValueError, match="geometry.axes must be a sequence"):
        PhysicalGeometryContract.from_dict(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": ""},
        {"axes": ("x", "x", "z")},
        {"units": ("m", "m")},
        {"normalization_scale": (1.0, 0.0, 1.0)},
        {"normalization_offset": (0.0, float("nan"), 0.0)},
        {"schema": "picf.unknown.v9"},
    ],
)
def test_physical_geometry_contract_rejects_ambiguous_charts(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _contract(**overrides)

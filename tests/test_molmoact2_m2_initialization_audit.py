from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.geometry import PhysicalGeometryContract  # noqa: E402
from picf_next.models.discovery import (  # noqa: E402
    ObjectDiscoveryConfig,
    TaskIndependentObjectDiscovery,
)
from tools.audit_molmoact2_m2_initialization import (  # noqa: E402
    _geometry_mean_initialization,
)

GEOMETRY = PhysicalGeometryContract(
    name="test-xy",
    quantity="object-centre",
    reference_frame="test",
    axes=("x", "y"),
    units=("m", "m"),
    normalization_offset=(0.0, 0.0),
    normalization_scale=(1.0, 1.0),
)


def _config(initial_variance: float = 0.1) -> ObjectDiscoveryConfig:
    return ObjectDiscoveryConfig(
        input_dim=8,
        hidden_dim=8,
        num_queries=3,
        num_layers=1,
        num_heads=2,
        address_dim=3,
        content_dim=4,
        geometry_dim=2,
        geometry_contract=GEOMETRY,
        initial_variance=initial_variance,
    )


def test_initialization_audit_reproduces_both_geometry_mean_contracts() -> None:
    torch.manual_seed(7)
    chart_origin = TaskIndependentObjectDiscovery(_config())
    assert torch.count_nonzero(chart_origin.geometry_head.weight) == 0
    assert torch.count_nonzero(chart_origin.geometry_head.bias) == 0

    torch.manual_seed(7)
    with _geometry_mean_initialization("linear_default"):
        linear_default = TaskIndependentObjectDiscovery(_config())
    assert torch.count_nonzero(linear_default.geometry_head.weight) > 0
    assert torch.count_nonzero(linear_default.geometry_head.bias) > 0

    torch.manual_seed(7)
    restored = TaskIndependentObjectDiscovery(_config())
    assert torch.count_nonzero(restored.geometry_head.weight) == 0
    assert torch.count_nonzero(restored.geometry_head.bias) == 0


def test_initialization_audit_preserves_declared_variance() -> None:
    model = TaskIndependentObjectDiscovery(_config(initial_variance=0.16))

    raw = model.variance_head.bias
    expected = torch.full_like(raw, 0.16)
    actual = torch.nn.functional.softplus(raw) + model.config.minimum_variance
    torch.testing.assert_close(actual, expected)

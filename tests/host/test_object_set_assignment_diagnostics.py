from __future__ import annotations

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")

from picf_next.eval.object_set_assignment import (  # noqa: E402
    object_set_assignment_diagnostics,
)
from picf_next.models.discovery import (  # noqa: E402
    ObjectDiscoveryOutput,
    ObjectExistenceCalibration,
)
from picf_next.models.set_loss import ObjectSetTarget, SetMatch  # noqa: E402


def test_object_set_assignment_diagnostics_names_missing_object_and_extra_query() -> None:
    ownership_logits = torch.tensor(
        [
            [8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0],
            [-8.0, 8.0, -8.0],
        ]
    ).unsqueeze(0)
    output = ObjectDiscoveryOutput(
        query_features=torch.zeros(1, 2, 3),
        address_mean=torch.zeros(1, 2, 2),
        content_mean=torch.zeros(1, 2, 2),
        geometry_mean=torch.zeros(1, 2, 2),
        geometry_variance=torch.ones(1, 2, 2),
        geometry_contract=synthetic_geometry_contract(2),
        existence_logits=torch.tensor([[8.0, -8.0]]),
        localization_confidence_logits=torch.tensor([[4.0, 2.0]]),
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
        token_valid=torch.ones(1, 4, dtype=torch.bool),
        token_group_id=torch.full((1, 4), -1, dtype=torch.long),
        evidence_available=torch.tensor([True]),
        existence_calibration=ObjectExistenceCalibration(),
    )
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(4, dtype=torch.bool),
        object_inventory_complete=True,
        temporal_identity_keys=("object/red", "object/blue"),
    )
    diagnostics = object_set_assignment_diagnostics(
        output,
        target,
        SetMatch(
            prediction_indices=torch.tensor([0, 1]),
            target_indices=torch.tensor([0, 1]),
        ),
        batch_index=0,
    )

    assert diagnostics["objects"][0]["identity_key"] == "object/red"
    assert diagnostics["objects"][0]["active"] is True
    assert diagnostics["objects"][0]["soft_dice"] > 0.99
    assert diagnostics["objects"][0]["target_ownership_mass"] == 2.0
    assert diagnostics["objects"][1]["identity_key"] == "object/blue"
    assert diagnostics["objects"][1]["active"] is False
    assert diagnostics["supervised_token_count"] == 4
    assert diagnostics["unmatched_queries"] == []


def test_object_set_assignment_diagnostics_rejects_partial_match() -> None:
    output = ObjectDiscoveryOutput(
        query_features=torch.zeros(1, 1, 2),
        address_mean=torch.zeros(1, 1, 1),
        content_mean=torch.zeros(1, 1, 1),
        geometry_mean=torch.zeros(1, 1, 1),
        geometry_variance=torch.ones(1, 1, 1),
        geometry_contract=synthetic_geometry_contract(1),
        existence_logits=torch.zeros(1, 1),
        localization_confidence_logits=torch.zeros(1, 1),
        ownership_logits=torch.zeros(1, 1, 2),
        ownership=torch.full((1, 1, 2), 0.5),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        token_group_id=torch.full((1, 1), -1, dtype=torch.long),
        evidence_available=torch.tensor([True]),
        existence_calibration=ObjectExistenceCalibration(),
    )
    target = ObjectSetTarget(
        ownership=torch.tensor([[1.0, 0.0, 0.0]]),
        token_valid=torch.ones(1, dtype=torch.bool),
        object_inventory_complete=True,
        temporal_identity_keys=("object/a", "object/b"),
    )

    with pytest.raises(ValueError, match="every target object"):
        object_set_assignment_diagnostics(
            output,
            target,
            SetMatch(
                prediction_indices=torch.tensor([0]),
                target_indices=torch.tensor([0]),
            ),
            batch_index=0,
        )

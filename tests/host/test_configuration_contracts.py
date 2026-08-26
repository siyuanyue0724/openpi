from __future__ import annotations

# ruff: noqa: E402
import math

import pytest

torch = pytest.importorskip("torch")

from picf_next.association import associate_lifecycle
from picf_next.models.binding_loss import BindingLossConfig
from picf_next.models.core import PICFCoreConfig
from picf_next.models.discovery import ObjectDiscoveryConfig, ObjectExistenceCalibration
from picf_next.models.dynamics_loss import ObjectDynamicsLossConfig
from picf_next.models.evidence import (
    ModalityProjectionSpec,
    MultimodalBindingProjector,
    NativeTokenBank,
)
from picf_next.models.objective import PICFObjectiveConfig
from picf_next.models.set_loss import ObjectSetLossConfig, ObjectSetMatcherConfig
from picf_next.models.temporal import TemporalFilterConfig, empty_object_belief
from tests.geometry_contract import synthetic_geometry_contract

GEOMETRY = synthetic_geometry_contract(2)


def _discovery_config(**overrides) -> ObjectDiscoveryConfig:
    values = {
        "input_dim": 8,
        "hidden_dim": 8,
        "num_queries": 2,
        "num_layers": 1,
        "num_heads": 2,
        "address_dim": 2,
        "content_dim": 2,
        "geometry_dim": 2,
        "geometry_contract": GEOMETRY,
        "initial_variance": 0.1,
    }
    values.update(overrides)
    return ObjectDiscoveryConfig(**values)


def _temporal_config(**overrides) -> TemporalFilterConfig:
    values = {
        "address_dim": 2,
        "content_dim": 2,
        "geometry_dim": 2,
        "geometry_contract": GEOMETRY,
        "action_dim": 3,
        "reference_delta_t_s": 0.1,
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
    }
    values.update(overrides)
    return TemporalFilterConfig(**values)


def test_state_configs_reject_width_or_semantic_geometry_mismatch() -> None:
    with pytest.raises(ValueError, match="width differs"):
        _discovery_config(geometry_contract=synthetic_geometry_contract(3))
    with pytest.raises(ValueError, match="width differs"):
        _temporal_config(geometry_contract=synthetic_geometry_contract(3))

    with pytest.raises(ValueError, match="contracts must agree"):
        PICFCoreConfig(
            modality_specs=(ModalityProjectionSpec("vision", token_dim=4),),
            binding_dim=8,
            discovery=_discovery_config(),
            temporal=_temporal_config(
                geometry_contract=synthetic_geometry_contract(
                    2,
                    name="picf.same-width-other-semantics.v1",
                )
            ),
            posterior_capacity=2,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_loss_configs_reject_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValueError):
        BindingLossConfig(temperature=invalid)
    with pytest.raises(ValueError):
        BindingLossConfig(logit_bias=invalid)
    with pytest.raises(ValueError):
        ObjectSetMatcherConfig(existence_cost=invalid)
    with pytest.raises(ValueError):
        ObjectSetLossConfig(existence_weight=invalid)
    with pytest.raises(ValueError):
        ObjectSetLossConfig(localization_confidence_weight=invalid)
    with pytest.raises(ValueError):
        ObjectDynamicsLossConfig(content_cosine_weight=invalid)
    with pytest.raises(ValueError):
        PICFObjectiveConfig(
            action_weight=1.0,
            set_weight=invalid,
            dynamics_weight=1.0,
            binding_weight=1.0,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_probabilistic_configs_reject_nonfinite_variance(invalid: float) -> None:
    with pytest.raises(ValueError):
        ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=8,
            num_queries=2,
            num_layers=1,
            num_heads=2,
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
            minimum_variance=invalid,
        )
    with pytest.raises(ValueError):
        _discovery_config(initial_variance=invalid)
    with pytest.raises(ValueError):
        TemporalFilterConfig(
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            minimum_variance=invalid,
        )


@pytest.mark.parametrize("invalid", [True, -1.0, 0.0, 1e-4])
def test_discovery_rejects_initial_variance_outside_declared_bounds(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="initial_variance"):
        _discovery_config(initial_variance=invalid)


def test_discovery_variance_scale_has_no_arbitrary_unit_upper_bound() -> None:
    config = _discovery_config(minimum_variance=2.0, initial_variance=3.0)

    assert config.minimum_variance == 2.0
    assert config.initial_variance == 3.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0])
def test_association_metric_rejects_invalid_address_temperature(invalid: float) -> None:
    with pytest.raises(ValueError):
        TemporalFilterConfig(
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            association_address_temperature=invalid,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_association_metric_rejects_nonfinite_address_bias(invalid: float) -> None:
    with pytest.raises(ValueError):
        TemporalFilterConfig(
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            association_address_logit_bias=invalid,
        )


@pytest.mark.parametrize("field", ["initial_detection_probability"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0, 1.0])
def test_lifecycle_probabilities_must_be_finite_and_strictly_inside_unit_interval(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        TemporalFilterConfig(
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            **{field: invalid},
        )


@pytest.mark.parametrize(
    "field",
    [
        "empty_bank_birth_to_clutter_prior_odds",
        "recurrent_birth_to_clutter_prior_odds",
    ],
)
@pytest.mark.parametrize("invalid", [True, -1.0, 0.0, float("nan"), float("inf")])
def test_birth_intensity_odds_must_be_positive_finite(field: str, invalid: float) -> None:
    with pytest.raises(ValueError, match=field):
        _temporal_config(**{field: invalid})


@pytest.mark.parametrize("invalid", [True, -1.0, 0.0, float("nan"), float("inf")])
def test_existence_calibration_weight_must_be_positive_finite(invalid: float) -> None:
    with pytest.raises(ValueError, match="unmatched_query_weight"):
        ObjectExistenceCalibration(unmatched_query_weight=invalid)


@pytest.mark.parametrize("posterior", [0.01, 0.2, 0.5, 0.8, 0.99])
def test_existence_calibration_exactly_inverts_weighted_bce(posterior: float) -> None:
    calibration = ObjectExistenceCalibration(unmatched_query_weight=0.1)
    physical_logit = torch.logit(torch.tensor(posterior, dtype=torch.float64))
    weighted_bce_logit = physical_logit - math.log(calibration.unmatched_query_weight)

    actual = calibration.posterior_probability(weighted_bce_logit)
    torch.testing.assert_close(actual, torch.tensor(posterior, dtype=torch.float64))
    assert calibration.training_probability_at_half_posterior == pytest.approx(10.0 / 11.0)
    assert calibration.training_logit_at_half_posterior == pytest.approx(math.log(10.0))


def test_existence_calibration_promotes_reduced_precision_before_odds_shift() -> None:
    calibration = ObjectExistenceCalibration(unmatched_query_weight=0.1)
    reduced = torch.tensor([[6.0, -6.0]], dtype=torch.bfloat16)
    expected = calibration.posterior_probability(reduced.float())
    actual = calibration.posterior_probability(reduced)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ModalityProjectionSpec("vision", token_dim=True),
        lambda: ModalityProjectionSpec("vision", token_dim=4, geometry_dim=False),
        lambda: ModalityProjectionSpec("touch", token_dim=4, require_single_active_group=1),
        lambda: MultimodalBindingProjector(
            (ModalityProjectionSpec("vision", token_dim=4),), binding_dim=True
        ),
        lambda: _discovery_config(input_dim=True),
        lambda: _discovery_config(dropout=False),
        lambda: _discovery_config(minimum_variance=True),
        lambda: _temporal_config(action_dim=True),
        lambda: _temporal_config(reference_delta_t_s=True),
        lambda: _temporal_config(dropout=False),
        lambda: _temporal_config(minimum_variance=True),
        lambda: _temporal_config(association_address_temperature=True),
        lambda: _temporal_config(association_address_logit_bias=True),
        lambda: ObjectSetMatcherConfig(existence_cost=True),
        lambda: ObjectSetLossConfig(existence_weight=True),
        lambda: ObjectSetLossConfig(localization_confidence_weight=True),
        lambda: BindingLossConfig(temperature=True),
        lambda: ObjectDynamicsLossConfig(content_cosine_weight=True),
        lambda: PICFObjectiveConfig(True, 0.0, 0.0, 0.0),
        lambda: PICFObjectiveConfig(0.0, 0.0, 0.0, 1.0, 1),
    ],
)
def test_public_model_configs_reject_boolean_numbers(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_temporal_pair_requirement_needs_active_binding() -> None:
    with pytest.raises(ValueError, match="only when binding is active"):
        PICFObjectiveConfig(
            action_weight=1.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=0.0,
            require_temporal_positive_pairs=True,
        )


def test_core_capacity_and_empty_belief_reject_boolean_dimensions() -> None:
    discovery = _discovery_config()
    temporal = _temporal_config()
    spec = ModalityProjectionSpec("vision", token_dim=4)
    with pytest.raises(ValueError):
        PICFCoreConfig((spec,), True, discovery, temporal, 2)
    with pytest.raises(ValueError):
        PICFCoreConfig((spec,), 8, discovery, temporal, True)
    with pytest.raises(ValueError):
        empty_object_belief(temporal, batch_size=True, capacity=2)


def test_lifecycle_association_rejects_boolean_capacity() -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        associate_lifecycle(
            [[0.0]],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            capacity=True,
        )


@pytest.mark.parametrize("contract", [1, False, "", "   "])
def test_encoder_contract_is_a_nonempty_string(contract: object) -> None:
    torch = pytest.importorskip("torch")
    projector = MultimodalBindingProjector(
        (ModalityProjectionSpec("vision", token_dim=4),),
        binding_dim=4,
    )
    bank = NativeTokenBank(
        modality="vision",
        tokens=torch.zeros(1, 1, 4),
        valid=torch.ones(1, 1, dtype=torch.bool),
        encoder_contract=contract,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="encoder contract"):
        projector((bank,))

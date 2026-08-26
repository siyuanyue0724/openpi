from __future__ import annotations

from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
core_module = pytest.importorskip("picf_next.models.core")
discovery_module = pytest.importorskip("picf_next.models.discovery")
evidence_module = pytest.importorskip("picf_next.models.evidence")
filter_module = pytest.importorskip("picf_next.models.filter")
temporal_module = pytest.importorskip("picf_next.models.temporal")

PICFCore = core_module.PICFCore
PICFCoreConfig = core_module.PICFCoreConfig
ObjectDiscoveryConfig = discovery_module.ObjectDiscoveryConfig
TaskIndependentObjectDiscovery = discovery_module.TaskIndependentObjectDiscovery
ModalityProjectionSpec = evidence_module.ModalityProjectionSpec
MultimodalBindingProjector = evidence_module.MultimodalBindingProjector
NativeTokenBank = evidence_module.NativeTokenBank
PersistentObjectFilter = filter_module.PersistentObjectFilter
ObjectBeliefBatch = temporal_module.ObjectBeliefBatch
TemporalFilterConfig = temporal_module.TemporalFilterConfig
GEOMETRY = synthetic_geometry_contract(2)


def _config(*, runtime_validation: str = "full") -> PICFCoreConfig:
    return PICFCoreConfig(
        modality_specs=(ModalityProjectionSpec("vision", token_dim=6, geometry_dim=2),),
        binding_dim=8,
        discovery=ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=12,
            num_queries=3,
            num_layers=2,
            num_heads=3,
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
        ),
        temporal=TemporalFilterConfig(
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=12,
            num_layers=2,
            num_heads=3,
        ),
        posterior_capacity=3,
        runtime_validation=runtime_validation,
    )


def _bank() -> NativeTokenBank:
    generator = torch.Generator().manual_seed(1019)
    return NativeTokenBank(
        modality="vision",
        tokens=torch.randn(2, 4, 6, generator=generator),
        valid=torch.ones(2, 4, dtype=torch.bool),
        geometry=torch.randn(2, 4, 2, generator=generator),
    )


def _assert_core_outputs_exact(actual, expected) -> None:
    tensor_pairs = (
        (actual.projection.binding_features, expected.projection.binding_features),
        (actual.projection.token_valid, expected.projection.token_valid),
        (
            actual.projection.current_measurement_valid,
            expected.projection.current_measurement_valid,
        ),
        (actual.projection.token_group_id, expected.projection.token_group_id),
        (actual.discovery.query_features, expected.discovery.query_features),
        (actual.discovery.address_mean, expected.discovery.address_mean),
        (actual.discovery.content_mean, expected.discovery.content_mean),
        (actual.discovery.geometry_mean, expected.discovery.geometry_mean),
        (actual.discovery.geometry_variance, expected.discovery.geometry_variance),
        (actual.discovery.existence_logits, expected.discovery.existence_logits),
        (actual.discovery.ownership, expected.discovery.ownership),
        (actual.posterior.belief.state_mean, expected.posterior.belief.state_mean),
        (
            actual.posterior.belief.geometry_covariance_diag,
            expected.posterior.belief.geometry_covariance_diag,
        ),
        (actual.posterior.belief.existence_logits, expected.posterior.belief.existence_logits),
        (
            actual.posterior.belief.visibility_given_existence_logits,
            expected.posterior.belief.visibility_given_existence_logits,
        ),
        (
            actual.posterior.belief.measurement_age_s,
            expected.posterior.belief.measurement_age_s,
        ),
        (actual.posterior.belief.valid, expected.posterior.belief.valid),
        (actual.posterior.belief.age, expected.posterior.belief.age),
        (actual.posterior.ownership, expected.posterior.ownership),
        (actual.posterior.born, expected.posterior.born),
        (actual.posterior.event_type, expected.posterior.event_type),
        (actual.action_bank.address, expected.action_bank.address),
        (actual.action_bank.value, expected.action_bank.value),
        (actual.action_bank.valid, expected.action_bank.valid),
        (actual.action_bank.log_prior, expected.action_bank.log_prior),
    )
    for observed, reference in tensor_pairs:
        torch.testing.assert_close(observed, reference, atol=0.0, rtol=0.0)


def test_metadata_runtime_validation_is_math_identical_to_full_validation() -> None:
    torch.manual_seed(1021)
    full_config = _config()
    full = full_config.build().eval()
    metadata = replace(full_config, runtime_validation="metadata").build().eval()
    metadata.load_state_dict(full.state_dict())
    prior = full_config.empty_belief(batch_size=2)
    action = torch.zeros(2, full_config.temporal.action_dim)
    delta_t = torch.full((2,), full_config.temporal.reference_delta_t_s)

    full_output = full((_bank(),), prior, action, delta_t)
    metadata_output = metadata((_bank(),), prior, action, delta_t)

    _assert_core_outputs_exact(metadata_output, full_output)


@pytest.mark.parametrize("invalid", ["", "none", 1, None, ["full"]])
def test_core_config_rejects_unknown_runtime_validation(invalid: object) -> None:
    with pytest.raises(ValueError, match="runtime_validation"):
        replace(_config(), runtime_validation=invalid)  # type: ignore[arg-type]


def test_core_rejects_mixed_component_runtime_validation() -> None:
    config = _config()
    projector = MultimodalBindingProjector(
        config.modality_specs,
        binding_dim=config.binding_dim,
        validate_tensor_values=True,
    )
    discovery = TaskIndependentObjectDiscovery(
        config.discovery,
        validate_tensor_values=False,
    )
    posterior_filter = PersistentObjectFilter(
        config.temporal,
        validate_tensor_values=False,
    )

    with pytest.raises(ValueError, match="share one runtime validation policy"):
        PICFCore(projector, discovery, posterior_filter)


def test_metadata_mode_keeps_shape_dtype_and_device_contracts() -> None:
    config = _config(runtime_validation="metadata")
    core = config.build().eval()
    malformed_bank = replace(_bank(), tokens=torch.zeros(2, 4, 5))
    prior = config.empty_belief(batch_size=2)

    with pytest.raises(ValueError, match="token shape"):
        core(
            (malformed_bank,),
            prior,
            torch.zeros(2, config.temporal.action_dim),
            torch.full((2,), config.temporal.reference_delta_t_s),
        )

    mixed_dtype_prior = ObjectBeliefBatch(
        address_mean=prior.address_mean,
        content_mean=prior.content_mean,
        geometry_mean=prior.geometry_mean,
        geometry_covariance_diag=prior.geometry_covariance_diag.double(),
        existence_logits=prior.existence_logits,
        visibility_given_existence_logits=prior.visibility_given_existence_logits,
        measurement_age_s=prior.measurement_age_s,
        valid=prior.valid,
        age=prior.age,
    )
    with pytest.raises(ValueError, match="share one floating dtype"):
        core.posterior_filter.transition(
            mixed_dtype_prior,
            torch.zeros(2, config.temporal.action_dim),
            torch.full((2,), config.temporal.reference_delta_t_s),
        )


def test_full_mode_rejects_nonfinite_input_values() -> None:
    config = _config()
    core = config.build().eval()
    bank = _bank()
    corrupt_tokens = bank.tokens.clone()
    corrupt_tokens[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        core(
            (replace(bank, tokens=corrupt_tokens),),
            config.empty_belief(batch_size=2),
            torch.zeros(2, config.temporal.action_dim),
            torch.full((2,), config.temporal.reference_delta_t_s),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda config: MultimodalBindingProjector(
            config.modality_specs,
            binding_dim=config.binding_dim,
            validate_tensor_values=1,
        ),
        lambda config: TaskIndependentObjectDiscovery(
            config.discovery,
            validate_tensor_values=1,
        ),
        lambda config: PersistentObjectFilter(
            config.temporal,
            validate_tensor_values=1,
        ),
    ],
)
def test_runtime_validation_flags_must_be_boolean(factory) -> None:
    with pytest.raises(ValueError, match="boolean"):
        factory(_config())

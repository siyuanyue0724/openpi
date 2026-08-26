from __future__ import annotations

import copy

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")

from picf_next.models.core import PICFCoreConfig  # noqa: E402
from picf_next.models.discovery import ObjectDiscoveryConfig  # noqa: E402
from picf_next.models.dynamics_loss import (  # noqa: E402
    ObjectDynamicsCriterion,
    ObjectDynamicsLossConfig,
    ObjectLifecycleInventoryTarget,
)
from picf_next.models.evidence import ModalityProjectionSpec, NativeTokenBank  # noqa: E402
from picf_next.models.objective import PICFObjective, PICFObjectiveConfig  # noqa: E402
from picf_next.models.set_loss import ObjectSetTarget  # noqa: E402
from picf_next.models.temporal import TemporalFilterConfig  # noqa: E402
from picf_next.training.stationary_temporal import (  # noqa: E402
    STATIONARY_TEMPORAL_EXECUTION_CONTRACT,
    StationaryTemporalCoreTrainer,
    StationaryTemporalObservation,
    StationaryTemporalSupervision,
)

GEOMETRY = synthetic_geometry_contract(2)


def _config() -> PICFCoreConfig:
    return PICFCoreConfig(
        modality_specs=(ModalityProjectionSpec("vision", token_dim=6),),
        binding_dim=8,
        discovery=ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=12,
            num_queries=3,
            num_layers=1,
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
            num_layers=1,
            num_heads=3,
        ),
        posterior_capacity=5,
    )


def _clip() -> tuple[
    tuple[StationaryTemporalObservation, ...],
    tuple[StationaryTemporalSupervision, ...],
]:
    observations = []
    targets = []
    valid = torch.ones(1, 5, dtype=torch.bool)
    ownership = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    for step in range(3):
        torch.manual_seed(200 + step)
        observations.append(
            StationaryTemporalObservation(
                native_banks=(NativeTokenBank("vision", torch.randn(1, 5, 6), valid),),
                previous_executed_action=torch.full((1, 3), step * 0.01),
                delta_t_s=torch.full((1,), 0.1),
            )
        )
        targets.append(
            StationaryTemporalSupervision(
                set_targets=(
                    ObjectSetTarget(
                        ownership=ownership,
                        token_valid=valid[0],
                        object_inventory_complete=True,
                        temporal_identity_keys=("object:a",),
                    ),
                ),
                lifecycle_targets=(
                    ObjectLifecycleInventoryTarget(
                        alive_identity_keys=("object:a",),
                        inventory_complete=True,
                        visibility=torch.ones(1),
                        visibility_supervised=torch.ones(1, dtype=torch.bool),
                    ),
                ),
            )
        )
    return tuple(observations), tuple(targets)


def _trainer() -> StationaryTemporalCoreTrainer:
    config = _config()
    core = config.build().train()
    with torch.no_grad():
        core.discovery.existence_head.weight.zero_()
        core.discovery.existence_head.bias.fill_(6.0)
        core.discovery.localization_confidence_head.weight.zero_()
        core.discovery.localization_confidence_head.bias.fill_(6.0)
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.1,
            dynamics_weight=0.1,
            binding_weight=0.1,
            require_temporal_positive_pairs=True,
        ),
        dynamics_criterion=ObjectDynamicsCriterion(
            ObjectDynamicsLossConfig(
                content_cosine_weight=1.0,
                geometry_nll_weight=1.0,
                survival_weight=1.0,
                visibility_weight=1.0,
            )
        ),
    )
    return StationaryTemporalCoreTrainer(core, objective, capacity=config.posterior_capacity)


def test_stationary_temporal_training_replays_prefix_without_a_graph() -> None:
    trainer = _trainer()
    observations, supervision = _clip()
    grad_modes = []
    call_order = []

    def record_forward(_module, _inputs) -> None:
        grad_modes.append(torch.is_grad_enabled())
        call_order.append("forward")

    handle = trainer.core.projector.register_forward_pre_hook(record_forward)
    before = copy.deepcopy(trainer.state_dict())

    def build_supervision(frame_index: int) -> StationaryTemporalSupervision:
        call_order.append(f"target:{frame_index}")
        return supervision[frame_index]

    output = trainer(
        observations,
        prefix_length=1,
        supervision_builder=build_supervision,
    )
    handle.remove()

    assert grad_modes == [False, True, True]
    assert call_order == [
        "forward",
        "target:0",
        "forward",
        "target:1",
        "forward",
        "target:2",
    ]
    assert output.execution_contract == STATIONARY_TEMPORAL_EXECUTION_CONTRACT
    assert output.prefix_length == 1
    assert output.train_length == 2
    assert output.objective.diagnostics["transitions"] == 2
    assert output.objective.diagnostics["temporal_positive_pairs"] > 0
    assert torch.isfinite(output.objective.loss)
    assert all(
        not getattr(output.final_belief, name).requires_grad
        for name in (
            "address_mean",
            "content_mean",
            "geometry_mean",
            "geometry_covariance_diag",
            "existence_logits",
            "visibility_given_existence_logits",
            "measurement_age_s",
            "valid",
            "age",
        )
    )
    for name, value in trainer.state_dict().items():
        torch.testing.assert_close(value, before[name])

    output.objective.loss.backward()
    assert trainer.core.discovery.address_head.weight.grad is not None
    assert trainer.core.posterior_filter.transition.dynamic_head.weight.grad is not None
    transition = trainer.core.posterior_filter.transition
    assert transition.detectability_if_detected_head.weight.grad is not None
    assert transition.detectability_if_missed_head.weight.grad is not None
    assert transition.missed_duration_logit_slope.grad is not None


def test_stationary_temporal_training_replays_mature_detectability_transitions() -> None:
    trainer = _trainer()
    observations, supervision = _clip()

    output = trainer(
        observations,
        prefix_length=2,
        supervision_builder=lambda frame_index: supervision[frame_index],
    )

    assert output.objective.diagnostics["lifecycle_detection_prefix_replay_predictions"] > 0
    output.objective.loss.backward()
    transition = trainer.core.posterior_filter.transition
    assert transition.detectability_if_detected_head.weight.grad is not None
    assert transition.detectability_if_missed_head.weight.grad is not None


def test_stationary_temporal_training_isolates_no_grad_autocast_weight_cache() -> None:
    trainer = _trainer()
    observations, supervision = _clip()
    bf16_observations = tuple(
        StationaryTemporalObservation(
            native_banks=tuple(
                NativeTokenBank(
                    bank.modality,
                    bank.tokens.to(torch.bfloat16),
                    bank.valid,
                )
                for bank in observation.native_banks
            ),
            previous_executed_action=observation.previous_executed_action.to(torch.bfloat16),
            delta_t_s=observation.delta_t_s.to(torch.bfloat16),
        )
        for observation in observations
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = trainer(
            bf16_observations,
            prefix_length=1,
            supervision_builder=lambda frame_index: supervision[frame_index],
        )
    output.objective.loss.backward()

    cached_linear_weights = (
        trainer.core.projector.content_projection["vision"].weight,
        trainer.core.discovery.input_projection.weight,
        trainer.core.discovery.layers[0].self_attention.in_proj_weight,
        trainer.core.discovery.address_head.weight,
        trainer.core.posterior_filter.transition.state_projection.weight,
        trainer.core.posterior_filter.transition.layers[0].attention.in_proj_weight,
        trainer.core.posterior_filter.transition.dynamic_head.weight,
    )
    assert all(parameter.grad is not None for parameter in cached_linear_weights)


def test_stationary_temporal_training_resets_between_calls() -> None:
    trainer = _trainer().eval()
    observations, supervision = _clip()

    def builder(frame_index: int) -> StationaryTemporalSupervision:
        return supervision[frame_index]

    first = trainer(observations, prefix_length=1, supervision_builder=builder)
    second = trainer(observations, prefix_length=1, supervision_builder=builder)

    torch.testing.assert_close(first.final_belief.address_mean, second.final_belief.address_mean)
    assert first.objective.loss_track_keys_by_row == second.objective.loss_track_keys_by_row


def test_stationary_temporal_training_rejects_action_loss_and_bad_clip_boundary() -> None:
    config = _config()
    with pytest.raises(ValueError, match="cannot include an action objective"):
        StationaryTemporalCoreTrainer(
            config.build(),
            PICFObjective(PICFObjectiveConfig(1.0, 0.0, 0.0, 0.0)),
            capacity=config.posterior_capacity,
        )

    trainer = _trainer()
    observations, supervision = _clip()
    with pytest.raises(ValueError, match="leave at least one"):
        trainer(
            observations,
            prefix_length=len(observations),
            supervision_builder=lambda frame_index: supervision[frame_index],
        )

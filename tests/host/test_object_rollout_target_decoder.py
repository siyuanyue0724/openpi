from __future__ import annotations

from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
target_module = pytest.importorskip("picf_next.data.rollout_targets")

ObjectGeometryRolloutSample = target_module.ObjectGeometryRolloutSample
PhysicalObjectGeometryFrame = target_module.PhysicalObjectGeometryFrame
build_object_geometry_rollout_target = target_module.build_object_geometry_rollout_target
GEOMETRY = synthetic_geometry_contract(3)


def _frame(keys: tuple[str, ...], value: float) -> PhysicalObjectGeometryFrame:
    geometry = torch.full((len(keys), 3), value)
    supervised = torch.ones_like(geometry, dtype=torch.bool)
    if keys:
        supervised[-1, -1] = False
        geometry[-1, -1] = 0.0
    return PhysicalObjectGeometryFrame(
        identity_keys=keys,
        geometry=geometry,
        geometry_variance=torch.where(
            supervised,
            torch.full_like(geometry, 0.01),
            torch.zeros_like(geometry),
        ),
        geometry_supervised=supervised,
        geometry_contract=GEOMETRY,
    )


def test_rollout_target_builder_pads_horizon_objects_and_unknown_axes() -> None:
    samples = (
        ObjectGeometryRolloutSample(
            executed_actions=torch.arange(14, dtype=torch.float32).reshape(2, 7),
            delta_t_s=torch.tensor([0.1, 0.2]),
            geometry_frames=(
                _frame(("track:a", "track:b"), 1.0),
                _frame(("track:a",), 2.0),
            ),
        ),
        ObjectGeometryRolloutSample(
            executed_actions=torch.ones(1, 7),
            delta_t_s=torch.tensor([0.1]),
            geometry_frames=(_frame(("track:c",), 3.0),),
        ),
    )

    target = build_object_geometry_rollout_target(
        samples,
        action_dim=7,
        geometry_contract=GEOMETRY,
        device="cpu",
        input_dtype=torch.bfloat16,
        target_dtype=torch.float32,
    )

    assert target.executed_actions.shape == (2, 2, 7)
    assert target.executed_actions.dtype == torch.bfloat16
    assert target.delta_t_s.dtype == torch.bfloat16
    assert target.geometry.shape == (2, 2, 2, 3)
    assert target.geometry.dtype == torch.float32
    assert target.geometry_variance.dtype == torch.float32
    assert target.step_valid.tolist() == [[True, True], [True, False]]
    assert torch.count_nonzero(target.executed_actions[1, 1]) == 0
    assert target.delta_t_s[1, 1] == 0.0
    assert target.identity_keys == (
        (("track:a", "track:b"), ("track:a", None)),
        (("track:c", None), (None, None)),
    )
    assert target.geometry[0, 0, 1, -1] == 0.0
    assert not target.geometry_supervised[0, 0, 1, -1]
    assert not target.geometry_supervised[1, 1].any()


def test_rollout_target_builder_rejects_ambiguous_or_differentiable_geometry() -> None:
    frame = _frame(("track:a",), 1.0)
    sample = ObjectGeometryRolloutSample(
        executed_actions=torch.zeros(1, 7),
        delta_t_s=torch.tensor([0.1]),
        geometry_frames=(frame,),
    )
    bad_geometry = frame.geometry.clone()
    bad_geometry[~frame.geometry_supervised] = 1.0
    with pytest.raises(ValueError, match="unknown future geometry coordinates"):
        build_object_geometry_rollout_target(
            (replace(sample, geometry_frames=(replace(frame, geometry=bad_geometry),)),),
            action_dim=7,
            geometry_contract=GEOMETRY,
            device="cpu",
            input_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    differentiable = frame.geometry.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="floating and detached"):
        build_object_geometry_rollout_target(
            (replace(sample, geometry_frames=(replace(frame, geometry=differentiable),)),),
            action_dim=7,
            geometry_contract=GEOMETRY,
            device="cpu",
            input_dtype=torch.float32,
            target_dtype=torch.float32,
        )


def test_rollout_target_builder_rejects_a_batch_without_physical_geometry() -> None:
    empty = _frame((), 0.0)
    sample = ObjectGeometryRolloutSample(
        executed_actions=torch.zeros(1, 7),
        delta_t_s=torch.tensor([0.1]),
        geometry_frames=(empty,),
    )

    with pytest.raises(ValueError, match="no supervised physical object"):
        build_object_geometry_rollout_target(
            (sample,),
            action_dim=7,
            geometry_contract=GEOMETRY,
            device="cpu",
            input_dtype=torch.float32,
            target_dtype=torch.float32,
        )

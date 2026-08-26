from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.unified.state import (  # noqa: E402
    GeometrySchema,
    UnifiedBeliefState,
    deterministic_birth_noise,
    empty_belief_state,
)


def _state(*, batch: int = 2, capacity: int = 4) -> UnifiedBeliefState:
    torch.manual_seed(7)
    lifecycle = torch.log_softmax(torch.randn(batch, capacity, 3), dim=-1)
    geometry_valid = torch.tensor([True, True, False]).expand(batch, capacity, 3).clone()
    diagonal = torch.tensor([2.0, 3.0, 0.0]).expand(batch, capacity, 3)
    return UnifiedBeliefState(
        content=torch.randn(batch, capacity, 5),
        lifecycle_log_probs=lifecycle,
        geometry_mean=torch.randn(batch, capacity, 3) * geometry_valid,
        geometry_information=torch.diag_embed(diagonal),
        geometry_valid=geometry_valid,
        content_log_variance=torch.randn(batch, capacity, 2),
        expected_age=torch.rand(batch, capacity),
        evidence_age=torch.rand(batch, capacity),
    )


def _assert_state_close(left: UnifiedBeliefState, right: UnifiedBeliefState) -> None:
    assert left.batch_size == right.batch_size
    assert left.capacity == right.capacity
    for name in left.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value.dtype == torch.bool:
            assert torch.equal(left_value, right_value)
        else:
            torch.testing.assert_close(left_value, right_value, atol=2e-6, rtol=2e-6)


def test_geometry_schema_is_typed_and_static() -> None:
    schema = GeometrySchema(
        names=("x", "y", "z"),
        units=("metre", "metre", "metre"),
        frame="world",
    )
    assert schema.width == 3
    assert schema.canonical_dict() == {
        "names": ["x", "y", "z"],
        "units": ["metre", "metre", "metre"],
        "frame": "world",
    }
    with pytest.raises(ValueError, match="unique"):
        GeometrySchema(names=("x", "x"), units=("m", "m"), frame="world")
    with pytest.raises(TypeError, match="immutable tuples"):
        GeometrySchema(names=["x"], units=["m"], frame="world")  # type: ignore[arg-type]


def test_canonical_codec_round_trip_preserves_all_sufficient_statistics() -> None:
    state = _state()
    packed = state.canonical()
    assert packed.shape == (
        state.batch_size,
        state.capacity,
        UnifiedBeliefState.canonical_width(
            content_dim=state.content_dim,
            geometry_dim=state.geometry_dim,
            uncertainty_dim=state.uncertainty_dim,
        ),
    )
    decoded = UnifiedBeliefState.from_canonical(
        packed,
        content_dim=state.content_dim,
        geometry_dim=state.geometry_dim,
        uncertainty_dim=state.uncertainty_dim,
    )
    _assert_state_close(state, decoded)


def test_serialization_is_deterministic_and_fixed_size() -> None:
    state = _state()
    payload = state.serialize()
    assert payload == state.serialize()
    restored = UnifiedBeliefState.deserialize(payload)
    _assert_state_close(state, restored)

    changed = _state()
    changed = UnifiedBeliefState(
        **{
            **{name: getattr(changed, name) for name in changed.__dataclass_fields__},
            "content": changed.content + 1,
        }
    )
    assert len(changed.serialize()) == len(payload)
    with pytest.raises(ValueError, match="truncated|expected"):
        UnifiedBeliefState.deserialize(payload[:-4])


def test_row_permutation_moves_every_field_without_fixed_row_identity() -> None:
    state = _state()
    permutation = torch.tensor([2, 0, 3, 1])
    moved = state.permute_rows(permutation)
    for name in state.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(moved, name),
            getattr(state, name).index_select(1, permutation),
        )
    with pytest.raises(ValueError, match="exactly once"):
        state.permute_rows(torch.tensor([0, 0, 1, 2]))


def test_empty_state_is_soft_categorical_not_an_active_bit() -> None:
    state = empty_belief_state(
        batch_size=1,
        capacity=3,
        content_dim=4,
        geometry_dim=3,
        uncertainty_dim=2,
        birth_hazard=0.05,
    )
    torch.testing.assert_close(state.lifecycle_probs[..., 1], torch.full((1, 3), 0.05))
    torch.testing.assert_close(state.lifecycle_probs.sum(-1), torch.ones(1, 3))
    assert not hasattr(state, "active")
    assert not hasattr(state, "confidence")


def test_bfloat16_host_request_keeps_persistent_state_in_float32() -> None:
    state = empty_belief_state(
        batch_size=1,
        capacity=2,
        content_dim=3,
        geometry_dim=2,
        uncertainty_dim=1,
        dtype=torch.bfloat16,
    )
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        if value.dtype != torch.bool:
            assert value.dtype == torch.float32
    torch.testing.assert_close(state.lifecycle_probs.sum(-1), torch.ones(1, 2))


def test_birth_noise_is_replayable_without_a_persistent_row_embedding() -> None:
    arguments = {
        "episode_keys": ("episode-a", "episode-b"),
        "frame_indices": (7, 9),
        "capacity": 4,
        "content_dim": 5,
        "base_seed": 13,
    }
    first = deterministic_birth_noise(**arguments)
    second = deterministic_birth_noise(**arguments)
    assert torch.equal(first, second)
    changed = deterministic_birth_noise(**{**arguments, "frame_indices": (8, 9)})
    assert not torch.equal(first[0], changed[0])
    assert torch.equal(first[1], changed[1])
    with pytest.raises(TypeError, match="must be integers"):
        deterministic_birth_noise(**{**arguments, "base_seed": False})


def test_state_rejects_non_psd_or_unnormalized_statistics() -> None:
    state = _state(batch=1, capacity=1)
    fields = {name: getattr(state, name) for name in state.__dataclass_fields__}
    with pytest.raises(ValueError, match="normalized"):
        UnifiedBeliefState(**{**fields, "lifecycle_log_probs": torch.zeros(1, 1, 3)})
    bad_information = state.geometry_information.clone()
    bad_information[..., 0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        UnifiedBeliefState(**{**fields, "geometry_information": bad_information})
    with pytest.raises(TypeError, match="torch.float32"):
        UnifiedBeliefState(**{**fields, "content": state.content.bfloat16()})
    with pytest.raises(ValueError, match="rank"):
        UnifiedBeliefState(**{**fields, "content_log_variance": torch.zeros(1, 2)})

from __future__ import annotations

import json

import pytest
import torch

import picf_next.lingbot_native.session as session_module
from picf_next.lingbot_native.addresses import (
    EpisodeAddressState,
    deterministic_episode_permutation,
)
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.session import (
    NativeObservationBatch,
    NativeSessionConfig,
    NativeSessionManager,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    NativeLayerwisePosteriorState,
    NativePosteriorState,
)


def _observation(
    *,
    keys: tuple[str, ...] = ("env-a",),
    epochs: tuple[int, ...] = (0,),
    sequences: tuple[int, ...] = (0,),
    times: tuple[float, ...] = (0.0,),
    reset: tuple[bool, ...] = (True,),
) -> NativeObservationBatch:
    batch = len(keys)
    values = torch.zeros(batch, 1, 2)
    reset_tensor = torch.tensor(reset).reshape(batch, 1)
    return NativeObservationBatch(
        environment_keys=keys,
        reset_epochs=epochs,
        observation_sequences=sequences,
        observation_times=torch.tensor(times),
        reset=reset,
        controls=ExecutedControlBatch(
            values=values,
            field_valid=torch.zeros_like(values, dtype=torch.bool),
            token_valid=torch.ones(batch, 1, dtype=torch.bool),
            delta_time=torch.zeros(batch, 1),
            reset=reset_tensor,
            acknowledged=torch.ones(batch, 1, dtype=torch.bool),
        ),
    )


def _manager() -> NativeSessionManager:
    return NativeSessionManager(
        NativeSessionConfig(model_digest="lingbot-test", capacity=2, host_width=4)
    )


def _state(batch: int = 1, value: float = 1.0) -> NativePosteriorState:
    return NativePosteriorState(torch.full((batch, 2, 4), value, dtype=torch.float32))


def test_session_requires_atomic_reset_and_commits_detached_rows() -> None:
    manager = _manager()
    with pytest.raises(ValueError, match="begin with an atomic reset"):
        manager.prepare(_observation(sequences=(1,), times=(1.0,), reset=(False,)))
    first = manager.prepare(_observation())
    assert first.previous_state is None
    assert not first.previous_state_valid.any()
    committed = _state()
    manager.commit(first, committed)
    committed.rows.add_(5)
    second = manager.prepare(_observation(sequences=(1,), times=(0.1,), reset=(False,)))
    assert second.previous_state is not None
    assert second.previous_state_valid.all()
    assert torch.equal(second.previous_state.rows, torch.ones(1, 2, 4))


def test_session_rejects_duplicate_out_of_order_and_cross_epoch_inputs() -> None:
    manager = _manager()
    first = manager.prepare(_observation())
    manager.commit(first, _state())
    with pytest.raises(ValueError, match="sequence"):
        manager.prepare(_observation(reset=(False,)))
    with pytest.raises(ValueError, match="cross reset epochs"):
        manager.prepare(_observation(epochs=(1,), sequences=(1,), times=(1.0,), reset=(False,)))
    with pytest.raises(ValueError, match="strictly increase"):
        manager.prepare(_observation(reset=(True,)))


def test_session_isolates_environments_and_reset_drops_only_one_lane() -> None:
    manager = _manager()
    first = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(0, 0),
            sequences=(0, 0),
            times=(0.0, 0.0),
            reset=(True, True),
        )
    )
    manager.commit(
        first, NativePosteriorState(torch.cat((_state(value=1).rows, _state(value=2).rows)))
    )
    mixed = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(1, 0),
            sequences=(0, 1),
            times=(1.0, 1.0),
            reset=(True, False),
        )
    )
    assert mixed.previous_state is not None
    assert torch.equal(mixed.previous_state_valid, torch.tensor([False, True]))
    assert not mixed.previous_state.rows[0].any()
    assert torch.equal(mixed.previous_state.rows[1], torch.full((2, 4), 2.0))


def test_session_transactions_are_single_use_abortable_and_exactly_resumable() -> None:
    manager = _manager()
    transaction = manager.prepare(_observation())
    manager.commit(transaction, _state())
    with pytest.raises(RuntimeError, match="already committed"):
        manager.commit(transaction, _state())
    snapshot = manager.serialize()
    assert snapshot == manager.serialize()
    restored = NativeSessionManager.deserialize(manager.config, snapshot)
    assert restored.serialize() == snapshot
    pending = restored.prepare(_observation(sequences=(1,), times=(1.0,), reset=(False,)))
    with pytest.raises(RuntimeError, match="pending"):
        restored.serialize()
    restored.abort(pending)
    assert restored.serialize() == snapshot


def test_session_commit_is_atomic_when_record_staging_fails(monkeypatch) -> None:
    manager = _manager()
    first = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(0, 0),
            sequences=(0, 0),
            times=(0.0, 0.0),
            reset=(True, True),
        )
    )
    manager.commit(
        first,
        NativePosteriorState(torch.cat((_state(value=1).rows, _state(value=2).rows))),
    )
    baseline = manager.serialize()
    pending = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(0, 0),
            sequences=(1, 1),
            times=(1.0, 1.0),
            reset=(False, False),
        )
    )
    original_record = session_module._SessionRecord
    calls = 0

    def fail_second_record(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic staging failure")
        return original_record(*args, **kwargs)

    monkeypatch.setattr(session_module, "_SessionRecord", fail_second_record)
    with pytest.raises(RuntimeError, match="synthetic staging failure"):
        manager.commit(pending, _state(batch=2, value=3))
    manager.abort(pending)
    assert manager.serialize() == baseline


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.pop("records"), "top-level schema"),
        (lambda payload: payload.update({"undeclared": True}), "top-level schema"),
        (lambda payload: payload.update({"records": {}}), "records must be a list"),
    ),
)
def test_session_snapshot_rejects_non_exact_top_level_schema(mutation, message: str) -> None:
    manager = _manager()
    transaction = manager.prepare(_observation())
    manager.commit(transaction, _state())
    payload = json.loads(manager.serialize())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        NativeSessionManager.deserialize(manager.config, json.dumps(payload).encode())


def test_session_snapshot_rejects_non_exact_records_and_boolean_counters() -> None:
    manager = _manager()
    transaction = manager.prepare(_observation())
    manager.commit(transaction, _state())
    payload = json.loads(manager.serialize())
    payload["records"][0]["undeclared"] = 1
    with pytest.raises(ValueError, match="record has an incompatible schema"):
        NativeSessionManager.deserialize(manager.config, json.dumps(payload).encode())
    payload = json.loads(manager.serialize())
    payload["records"][0]["reset_epoch"] = False
    with pytest.raises(ValueError, match="counters must be non-negative integers"):
        NativeSessionManager.deserialize(manager.config, json.dumps(payload).encode())


def test_layerwise_session_mixed_reset_resume_and_schema_rejection() -> None:
    config = NativeSessionConfig(
        model_digest="layerwise-test",
        capacity=2,
        host_width=4,
        num_layers=3,
    )
    manager = NativeSessionManager(config)
    first = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(0, 0),
            sequences=(0, 0),
            times=(0.0, 0.0),
            reset=(True, True),
        )
    )
    rows = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    manager.commit(first, NativeLayerwisePosteriorState(rows))
    mixed = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(1, 0),
            sequences=(0, 1),
            times=(1.0, 1.0),
            reset=(True, False),
        )
    )
    assert isinstance(mixed.previous_state, NativeLayerwisePosteriorState)
    assert torch.equal(mixed.previous_state_valid, torch.tensor([False, True]))
    assert not mixed.previous_state.layer_rows[0].any()
    assert torch.equal(mixed.previous_state.layer_rows[1], rows[1])
    manager.abort(mixed)
    snapshot = manager.serialize()
    restored = NativeSessionManager.deserialize(config, snapshot)
    assert restored.serialize() == snapshot
    legacy = NativeSessionConfig(
        model_digest="layerwise-test",
        capacity=2,
        host_width=4,
    )
    with pytest.raises(ValueError, match="runtime contract"):
        NativeSessionManager.deserialize(legacy, snapshot)


def test_addressed_session_carries_reset_gauge_through_mixed_batch_and_snapshot() -> None:
    identity = "lingbot_task_query_object_value_read_v1"
    receipt = "a" * 64
    config = NativeSessionConfig(
        model_digest="addressed-layerwise-test",
        capacity=2,
        host_width=4,
        num_layers=3,
        addressed_architecture_identity=identity,
        address_codebook_sha256=receipt,
    )
    manager = NativeSessionManager(config)
    first = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(0, 0),
            sequences=(0, 0),
            times=(0.0, 0.0),
            reset=(True, True),
        )
    )
    first_state = AddressedLayerwisePosteriorState(
        layer_rows=torch.randn(2, 3, 2, 4),
        episode_address_state=EpisodeAddressState(
            permutation=deterministic_episode_permutation(first.episode_ids, 2),
            codebook_sha256=receipt,
        ),
        architecture_identity=identity,
    )
    manager.commit(first, first_state)

    mixed = manager.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(1, 0),
            sequences=(0, 1),
            times=(1.0, 1.0),
            reset=(True, False),
        )
    )
    assert isinstance(mixed.previous_state, AddressedLayerwisePosteriorState)
    assert torch.equal(mixed.previous_state_valid, torch.tensor([False, True]))
    assert not mixed.previous_state.layer_rows[0].any()
    assert torch.equal(mixed.previous_state.layer_rows[1], first_state.layer_rows[1])
    assert torch.equal(
        mixed.previous_state.episode_address_state.permutation[0],
        deterministic_episode_permutation(mixed.episode_ids[0:1], 2)[0],
    )
    assert torch.equal(
        mixed.previous_state.episode_address_state.permutation[1],
        first_state.episode_address_state.permutation[1],
    )
    manager.abort(mixed)

    snapshot = manager.serialize()
    payload = json.loads(snapshot)
    assert payload["version"] == 3
    restored = NativeSessionManager.deserialize(config, snapshot)
    assert restored.serialize() == snapshot

    unaddressed = NativeLayerwisePosteriorState(torch.randn(2, 3, 2, 4))
    pending = restored.prepare(
        _observation(
            keys=("env-a", "env-b"),
            epochs=(0, 0),
            sequences=(1, 1),
            times=(1.0, 1.0),
            reset=(False, False),
        )
    )
    with pytest.raises(ValueError, match="session state contract"):
        restored.commit(pending, unaddressed)
    restored.abort(pending)

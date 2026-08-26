from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.addresses import EpisodeAddressState, address_codebook_sha256
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    AddressedLayerwisePriorTrace,
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePosteriorState,
    NativeVidEoMTPairedPosteriorState,
    clone_persistent_state,
    layerwise_prior_trace_with_tensor,
    persistent_state_tensor,
    persistent_state_with_tensor,
    stack_layerwise_states,
    stack_native_states,
    unbind_layerwise_state,
    unbind_native_state,
)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_native_state_round_trip_is_exact_and_preserves_dtype(dtype: torch.dtype) -> None:
    rows = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(dtype)
    state = NativePosteriorState(rows)
    encoded = state.serialize()
    assert encoded == state.serialize()
    restored = NativePosteriorState.deserialize(encoded)
    assert restored.rows.dtype == dtype
    assert torch.equal(restored.rows, rows)


def test_native_state_is_only_host_width_rows_and_detects_corruption() -> None:
    state = NativePosteriorState(torch.randn(1, 4, 8))
    assert tuple(state.__dataclass_fields__) == ("rows",)
    encoded = bytearray(state.serialize())
    encoded[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        NativePosteriorState.deserialize(bytes(encoded))
    with pytest.raises(ValueError, match="truncated"):
        NativePosteriorState.deserialize(state.serialize()[:10])


def test_native_state_stack_unbind_detach_and_permutation_are_closed() -> None:
    rows = torch.randn(2, 4, 8, requires_grad=True)
    state = NativePosteriorState(rows)
    lanes = unbind_native_state(state)
    restored = stack_native_states(lanes)
    assert torch.equal(restored.rows, rows)
    detached = restored.detached()
    assert not detached.rows.requires_grad
    permutation = torch.tensor([2, 0, 3, 1])
    moved = state.permute_rows(permutation)
    assert torch.equal(moved.rows, rows[:, permutation])
    with pytest.raises(ValueError, match="exactly once"):
        state.permute_rows(torch.tensor([0, 0, 1, 2]))


def test_native_state_to_has_an_explicit_tensor_migration_contract() -> None:
    rows = torch.randn(1, 2, 3, dtype=torch.float32)
    state = NativePosteriorState(rows)
    moved = state.to(device="cpu", dtype=torch.bfloat16, copy=True)
    assert moved.rows.device.type == "cpu"
    assert moved.rows.dtype == torch.bfloat16
    assert moved.rows.data_ptr() != rows.data_ptr()
    assert state.rows.dtype == torch.float32


def test_native_state_rejects_side_channel_shapes_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        NativePosteriorState(torch.zeros(2, 3))
    with pytest.raises(TypeError, match="float16"):
        NativePosteriorState(torch.zeros(1, 2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="NaN"):
        NativePosteriorState(torch.tensor([[[float("nan")]]]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_layerwise_state_round_trip_is_exact_and_rejects_v1(dtype: torch.dtype) -> None:
    rows = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    state = NativeLayerwisePosteriorState(rows.to(dtype))
    encoded = state.serialize()
    restored = NativeLayerwisePosteriorState.deserialize(encoded)
    assert restored.layer_rows.dtype == dtype
    assert torch.equal(restored.layer_rows, state.layer_rows)
    with pytest.raises(ValueError, match="incompatible schema"):
        NativeLayerwisePosteriorState.deserialize(
            NativePosteriorState(torch.zeros(1, 4, 5)).serialize()
        )


def test_layerwise_state_stack_unbind_and_row_permutation_preserve_layer_axis() -> None:
    rows = torch.randn(2, 3, 4, 5, requires_grad=True)
    state = NativeLayerwisePosteriorState(rows)
    restored = stack_layerwise_states(unbind_layerwise_state(state))
    assert torch.equal(restored.layer_rows, rows)
    permutation = torch.tensor([2, 0, 3, 1])
    moved = state.permute_rows(permutation)
    assert torch.equal(moved.layer_rows, rows[:, :, permutation])
    assert not moved.detached().layer_rows.requires_grad


def test_layerwise_state_detects_corruption_and_invalid_shapes() -> None:
    state = NativeLayerwisePosteriorState(torch.randn(1, 2, 3, 4))
    encoded = bytearray(state.serialize())
    encoded[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        NativeLayerwisePosteriorState.deserialize(bytes(encoded))
    with pytest.raises(ValueError, match="shape"):
        NativeLayerwisePosteriorState(torch.zeros(1, 2, 3))


def test_native_videomt_paired_state_is_atomic_exact_and_permutation_closed() -> None:
    host = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    source = torch.randn(2, 4, 16, dtype=torch.float32, requires_grad=True)
    state = NativeVidEoMTPairedPosteriorState(
        layer_rows=host,
        source_queries=source,
        architecture_identity="native_videomt_query_posterior_v1",
    )

    restored = NativeVidEoMTPairedPosteriorState.deserialize(state.serialize())
    assert torch.equal(restored.layer_rows, host.detach())
    assert torch.equal(restored.source_queries, source.detach())
    assert restored.layer_rows.dtype == torch.bfloat16
    assert restored.source_queries.dtype == torch.float32
    assert restored.architecture_identity == state.architecture_identity

    stacked = stack_layerwise_states(unbind_layerwise_state(state))
    assert isinstance(stacked, NativeVidEoMTPairedPosteriorState)
    assert torch.equal(stacked.layer_rows, host)
    assert torch.equal(stacked.source_queries, source)

    permutation = torch.tensor([2, 0, 3, 1])
    moved = state.permute_rows(permutation)
    assert torch.equal(moved.layer_rows, host[:, :, permutation])
    assert torch.equal(moved.source_queries, source[:, permutation])
    detached = clone_persistent_state(state)
    assert isinstance(detached, NativeVidEoMTPairedPosteriorState)
    assert not detached.layer_rows.requires_grad
    assert not detached.source_queries.requires_grad


def test_native_videomt_paired_state_keeps_source_precision_on_host_cast() -> None:
    state = NativeVidEoMTPairedPosteriorState(
        layer_rows=torch.randn(1, 2, 3, 4),
        source_queries=torch.randn(1, 3, 5),
        architecture_identity="native_videomt_query_posterior_v1",
    )
    moved = state.to(dtype=torch.bfloat16, copy=True)
    assert moved.layer_rows.dtype == torch.bfloat16
    assert moved.source_queries.dtype == torch.float32
    assert moved.source_queries.data_ptr() != state.source_queries.data_ptr()


def test_addressed_layerwise_state_round_trip_and_helpers_preserve_one_receipt() -> None:
    rows = torch.randn(2, 3, 4, 8, requires_grad=True)
    codebook = torch.eye(4, 8)
    address_state = EpisodeAddressState(
        permutation=torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]]),
        codebook_sha256=address_codebook_sha256(codebook),
    )
    state = AddressedLayerwisePosteriorState(
        layer_rows=rows,
        episode_address_state=address_state,
        architecture_identity="lingbot_task_query_object_value_read_v1",
    )

    encoded = state.serialize()
    restored = AddressedLayerwisePosteriorState.deserialize(encoded)
    assert torch.equal(restored.layer_rows, rows.detach())
    assert restored.episode_address_state.same_assignment(address_state)
    assert restored.address_receipt == state.address_receipt
    assert restored.architecture_identity == state.architecture_identity

    lanes = unbind_layerwise_state(state)
    assert all(isinstance(lane, AddressedLayerwisePosteriorState) for lane in lanes)
    stacked = stack_layerwise_states(lanes)
    assert isinstance(stacked, AddressedLayerwisePosteriorState)
    assert torch.equal(stacked.layer_rows, rows)
    assert stacked.episode_address_state.same_assignment(address_state)

    cloned = clone_persistent_state(state)
    assert isinstance(cloned, AddressedLayerwisePosteriorState)
    assert cloned.episode_address_state.same_assignment(address_state)
    assert cloned.layer_rows.data_ptr() != rows.data_ptr()
    assert not cloned.layer_rows.requires_grad

    permutation = torch.tensor([3, 1, 0, 2])
    moved = state.permute_rows(permutation)
    assert torch.equal(moved.layer_rows, rows[:, :, permutation])
    assert torch.equal(
        moved.episode_address_state.permutation,
        address_state.permutation[:, permutation],
    )


def test_addressed_layerwise_state_rejects_corruption_and_mixed_stack() -> None:
    address_state = EpisodeAddressState(
        permutation=torch.tensor([[0, 1]]),
        codebook_sha256="0" * 64,
    )
    state = AddressedLayerwisePosteriorState(
        layer_rows=torch.randn(1, 2, 2, 4),
        episode_address_state=address_state,
        architecture_identity="candidate",
    )
    encoded = bytearray(state.serialize())
    encoded[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        AddressedLayerwisePosteriorState.deserialize(bytes(encoded))
    with pytest.raises(TypeError, match="cannot be stacked"):
        stack_layerwise_states(
            (
                state,
                NativeLayerwisePosteriorState(torch.randn(1, 2, 2, 4)),
            )
        )


def test_value_interventions_preserve_or_explicitly_replace_address_receipts() -> None:
    codebook_sha256 = "0" * 64
    local_address = EpisodeAddressState(
        permutation=torch.tensor([[0, 1, 2]]),
        codebook_sha256=codebook_sha256,
    )
    peer_address = EpisodeAddressState(
        permutation=torch.tensor([[2, 0, 1]]),
        codebook_sha256=codebook_sha256,
    )
    state = AddressedLayerwisePosteriorState(
        layer_rows=torch.randn(1, 2, 3, 4),
        episode_address_state=local_address,
        architecture_identity="task-addressed",
    )
    trace = AddressedLayerwisePriorTrace(
        layer_rows=torch.randn(1, 2, 3, 4),
        episode_address_state=local_address,
        architecture_identity="task-addressed",
    )

    zero_state = persistent_state_with_tensor(state, torch.zeros_like(state.layer_rows))
    peer_state = persistent_state_with_tensor(
        state,
        torch.ones_like(state.layer_rows),
        episode_address_state=peer_address,
    )
    zero_trace = layerwise_prior_trace_with_tensor(
        trace,
        torch.zeros_like(trace.layer_rows),
    )
    peer_trace = layerwise_prior_trace_with_tensor(
        trace,
        torch.ones_like(trace.layer_rows),
        episode_address_state=peer_address,
    )

    assert isinstance(zero_state, AddressedLayerwisePosteriorState)
    assert isinstance(peer_state, AddressedLayerwisePosteriorState)
    assert zero_state.episode_address_state.same_assignment(local_address)
    assert peer_state.episode_address_state.same_assignment(peer_address)
    assert isinstance(zero_trace, AddressedLayerwisePriorTrace)
    assert isinstance(peer_trace, AddressedLayerwisePriorTrace)
    assert zero_trace.episode_address_state.same_assignment(local_address)
    assert peer_trace.episode_address_state.same_assignment(peer_address)
    assert torch.count_nonzero(zero_state.layer_rows) == 0
    assert torch.count_nonzero(zero_trace.layer_rows) == 0


def test_unaddressed_value_interventions_reject_address_injection() -> None:
    address = EpisodeAddressState(
        permutation=torch.tensor([[0, 1]]),
        codebook_sha256="0" * 64,
    )
    state = NativeLayerwisePosteriorState(torch.randn(1, 2, 2, 4))
    trace = NativeLayerwisePriorTrace(torch.randn(1, 2, 2, 4))
    with pytest.raises(ValueError, match="unaddressed posterior"):
        persistent_state_with_tensor(
            state,
            torch.zeros_like(state.layer_rows),
            episode_address_state=address,
        )
    with pytest.raises(ValueError, match="unaddressed prior"):
        layerwise_prior_trace_with_tensor(
            trace,
            torch.zeros_like(trace.layer_rows),
            episode_address_state=address,
        )


def test_layerwise_prior_trace_is_attached_transient_and_not_persistent_state() -> None:
    rows = torch.randn(2, 3, 4, 5, requires_grad=True)
    trace = NativeLayerwisePriorTrace(rows)

    torch.testing.assert_close(trace.layer(1), rows[:, 1])
    assert trace.layer_rows.requires_grad
    assert not hasattr(trace, "serialize")
    assert not hasattr(trace, "detached")
    with pytest.raises(TypeError, match="unknown schema"):
        persistent_state_tensor(trace)  # type: ignore[arg-type]

    permutation = torch.tensor([2, 0, 3, 1])
    moved = trace.permute_rows(permutation)
    torch.testing.assert_close(moved.layer_rows, rows[:, :, permutation])
    moved.layer_rows.square().sum().backward()
    assert rows.grad is not None and rows.grad.abs().sum() > 0

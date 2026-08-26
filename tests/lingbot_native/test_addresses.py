from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.addresses import (
    EpisodeAddressState,
    address_codebook_sha256,
    deterministic_episode_permutation,
    episode_address_codes,
    fixed_orthogonal_address_codebook,
    validate_episode_permutation,
)


def test_fixed_address_codebook_is_deterministic_orthogonal_and_not_trainable() -> None:
    first = fixed_orthogonal_address_codebook(5, 24)
    second = fixed_orthogonal_address_codebook(5, 24)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first @ first.T, torch.eye(5), rtol=1e-6, atol=1e-6)
    assert not first.requires_grad
    assert len(address_codebook_sha256(first)) == 64


def test_episode_permutation_is_reproducible_and_episode_dependent() -> None:
    episode_ids = torch.tensor([17, 23, 17], dtype=torch.long)
    first = deterministic_episode_permutation(episode_ids, 8)
    second = deterministic_episode_permutation(episode_ids, 8)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[0], first[2], rtol=0, atol=0)
    assert not torch.equal(first[0], first[1])
    validate_episode_permutation(first, 8)


def test_episode_address_state_is_receipted_and_materializes_the_row_gauge() -> None:
    codebook = fixed_orthogonal_address_codebook(4, 16)
    permutation = torch.tensor([[2, 0, 3, 1]], dtype=torch.long)
    state = EpisodeAddressState(
        permutation=permutation,
        codebook_sha256=address_codebook_sha256(codebook),
    )
    torch.testing.assert_close(state.materialize(codebook), codebook[permutation])
    altered = codebook.clone()
    altered[0, 0] += 0.01
    with pytest.raises(ValueError, match="another immutable codebook"):
        state.materialize(altered)


def test_episode_address_state_helpers_preserve_or_change_receipts_explicitly() -> None:
    codebook = fixed_orthogonal_address_codebook(4, 16)
    episode_ids = torch.tensor([17, 23], dtype=torch.long)
    state = EpisodeAddressState.from_episode_ids(
        codebook=codebook,
        episode_ids=episode_ids,
    )
    repeated = EpisodeAddressState.from_episode_ids(
        codebook=codebook,
        episode_ids=episode_ids,
    )
    assert state.same_assignment(repeated)
    assert state.receipt == repeated.receipt
    assert state.to("cpu").same_assignment(state)

    selected = state.index_select(torch.tensor([1], dtype=torch.long))
    assert selected.batch_size == 1
    torch.testing.assert_close(selected.permutation, state.permutation[1:2])
    swapped = state.permute_rows(torch.tensor([1, 0, 3, 2], dtype=torch.long))
    torch.testing.assert_close(
        swapped.materialize(codebook),
        state.materialize(codebook)[:, [1, 0, 3, 2]],
    )
    assert swapped.receipt != state.receipt


def test_joint_row_and_address_permutation_preserves_pairwise_geometry() -> None:
    codebook = fixed_orthogonal_address_codebook(4, 16)
    episode = torch.tensor([[3, 1, 0, 2]], dtype=torch.long)
    addresses = episode_address_codes(codebook, episode)
    row_permutation = torch.tensor([2, 0, 3, 1])
    permuted_addresses = episode_address_codes(codebook, episode[:, row_permutation])
    torch.testing.assert_close(permuted_addresses, addresses[:, row_permutation])
    torch.testing.assert_close(
        permuted_addresses @ permuted_addresses.transpose(1, 2),
        (addresses @ addresses.transpose(1, 2))[:, row_permutation][:, :, row_permutation],
    )


@pytest.mark.parametrize("bad", [torch.tensor([[0, 0]]), torch.tensor([[0, 2]])])
def test_invalid_episode_address_assignment_is_rejected(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="permutation"):
        validate_episode_permutation(bad.to(torch.long), 2)


def test_address_codebook_rejects_insufficient_host_width() -> None:
    with pytest.raises(ValueError, match="too small"):
        fixed_orthogonal_address_codebook(9, 8)

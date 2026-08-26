from __future__ import annotations

import torch


def _embedding_backward_nodes(tensor: torch.Tensor) -> set[int]:
    pending = [tensor.grad_fn]
    seen: set[object] = set()
    embedding_nodes: set[int] = set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__.startswith("EmbeddingBackward"):
            embedding_nodes.add(id(node))
        pending.extend(next_node for next_node, _ in node.next_functions)
    return embedding_nodes


def test_fused_shared_embedding_lookup_preserves_outputs_and_weight_gradient() -> None:
    torch.manual_seed(7)
    vocabulary_size = 31
    hidden_size = 11
    batch_size = 3
    image_count = 2
    language_tokens = torch.tensor(
        [[2, 5, 7, 5], [11, 13, 17, 2], [19, 23, 29, 7]],
        dtype=torch.long,
    )
    special_token_ids = (3, 9)
    upstream_language = torch.randn(batch_size, 4, hidden_size)
    upstream_start = torch.randn(batch_size, image_count, 1, hidden_size)
    upstream_end = torch.randn(batch_size, image_count, 1, hidden_size)

    old_embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
    fused_embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
    fused_embedding.weight.data.copy_(old_embedding.weight.data)

    old_language = old_embedding(language_tokens)
    old_start = old_embedding(torch.tensor([special_token_ids[0]])).view(1, 1, 1, hidden_size)
    old_end = old_embedding(torch.tensor([special_token_ids[1]])).view(1, 1, 1, hidden_size)
    old_loss = (
        (old_language * upstream_language).sum()
        + (old_start.expand_as(upstream_start) * upstream_start).sum()
        + (old_end.expand_as(upstream_end) * upstream_end).sum()
    )

    fused_ids = torch.cat(
        (
            language_tokens.reshape(-1),
            torch.tensor(special_token_ids, dtype=language_tokens.dtype),
        )
    )
    fused_outputs = fused_embedding(fused_ids)
    language_count = language_tokens.numel()
    fused_language = fused_outputs[:language_count].reshape(
        *language_tokens.shape,
        hidden_size,
    )
    fused_boundaries = fused_outputs[language_count:]
    fused_start = fused_boundaries[0].view(1, 1, 1, hidden_size)
    fused_end = fused_boundaries[1].view(1, 1, 1, hidden_size)
    fused_loss = (
        (fused_language * upstream_language).sum()
        + (fused_start.expand_as(upstream_start) * upstream_start).sum()
        + (fused_end.expand_as(upstream_end) * upstream_end).sum()
    )

    assert torch.equal(old_language, fused_language)
    assert torch.equal(old_start, fused_start)
    assert torch.equal(old_end, fused_end)
    assert len(_embedding_backward_nodes(old_loss)) == 3
    assert len(_embedding_backward_nodes(fused_loss)) == 1

    old_loss.backward()
    fused_loss.backward()
    torch.testing.assert_close(
        old_embedding.weight.grad,
        fused_embedding.weight.grad,
        rtol=0,
        atol=0,
    )

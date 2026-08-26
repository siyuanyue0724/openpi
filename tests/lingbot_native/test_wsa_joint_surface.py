from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.lingbot_native.wsa_joint_surface import (  # noqa: E402
    WSAJointTokenLayout,
    block_wsa_future_to_action_information_edge,
    build_wsa_joint_attention_mask_with_layout,
    concatenate_wsa_joint_qkv,
    insert_future_history_queries,
    isolate_wsa_future_from_all_action_queries,
    split_wsa_joint_attention,
)


def _native_mask() -> torch.Tensor:
    mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    mask[:, :3, :3] = True
    mask[:, 3:, :3] = True
    mask[:, 3:, 3:] = torch.ones(2, 2, dtype=torch.bool).tril()
    return mask


def test_joint_mask_preserves_native_edges_and_adds_only_registered_future_edges() -> None:
    native = _native_mask()
    joint, layout = build_wsa_joint_attention_mask_with_layout(
        native,
        host_count=3,
        future_count=4,
    )
    assert layout == WSAJointTokenLayout(host_count=3, future_count=4, action_count=2)
    torch.testing.assert_close(joint[:, layout.host, layout.host], native[:, :3, :3])
    torch.testing.assert_close(joint[:, layout.host, layout.action], native[:, :3, 3:])
    torch.testing.assert_close(joint[:, layout.action, layout.host], native[:, 3:, :3])
    torch.testing.assert_close(joint[:, layout.action, layout.action], native[:, 3:, 3:])
    assert not joint[:, layout.host, layout.future].any()
    assert joint[:, layout.future, layout.host].all()
    assert joint[:, layout.future, layout.future].all()
    assert joint[:, layout.future, layout.action].all()
    assert joint[:, layout.action, layout.future].all()


def test_future_queries_cannot_read_external_persistent_memory() -> None:
    layout = WSAJointTokenLayout(host_count=3, future_count=4, action_count=2)
    native = torch.ones(1, 5, 7, dtype=torch.bool)
    expanded = insert_future_history_queries(native, layout=layout)
    assert expanded.shape == (1, 9, 7)
    assert expanded[:, layout.host].all()
    assert not expanded[:, layout.future].any()
    assert expanded[:, layout.action].all()


def test_future_to_action_intervention_changes_only_action_output_query_edges() -> None:
    native = _native_mask()
    joint, layout = build_wsa_joint_attention_mask_with_layout(
        native,
        host_count=3,
        future_count=4,
    )
    factual = joint.clone()
    intervened = block_wsa_future_to_action_information_edge(joint, layout=layout)
    torch.testing.assert_close(joint, factual)
    expected = factual.clone()
    expected[:, layout.action.start + 1 : layout.action.stop, layout.future] = False
    torch.testing.assert_close(intervened, expected)
    assert intervened[:, layout.action.start, layout.future].all()


def test_auxiliary_world_decoder_blocks_every_future_to_action_path() -> None:
    native = _native_mask()
    joint, layout = build_wsa_joint_attention_mask_with_layout(
        native,
        host_count=3,
        future_count=4,
    )
    factual = joint.clone()
    isolated = isolate_wsa_future_from_all_action_queries(joint, layout=layout)
    torch.testing.assert_close(joint, factual)
    expected = factual.clone()
    expected[:, layout.action, layout.future] = False
    torch.testing.assert_close(isolated, expected)

    # Convert [query, key] attention visibility to [source, sink] reachability
    # and close it across the complete repeated-layer graph.
    closure = isolated[0].T.clone()
    for intermediate in range(layout.total_count):
        closure |= (
            closure[:, intermediate].unsqueeze(1)
            & closure[intermediate].unsqueeze(0)
        )
    assert not closure[layout.future, layout.action].any()
    assert closure[layout.action, layout.future].all()


def test_qkv_concatenation_expands_only_lingbot_kv_heads() -> None:
    generator = torch.Generator().manual_seed(218)

    def qkv(tokens: int, kv_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn(1, tokens, 32, 128, generator=generator),
            torch.randn(1, tokens, kv_heads, 128, generator=generator),
            torch.randn(1, tokens, kv_heads, 128, generator=generator),
        )

    query, key, value = concatenate_wsa_joint_qkv(
        host_qkv=qkv(3, 8),
        future_qkv=qkv(4, 32),
        action_qkv=qkv(2, 8),
    )
    assert query.shape == key.shape == value.shape == (1, 9, 32, 128)


def test_attention_output_split_is_exact_and_ordered() -> None:
    layout = WSAJointTokenLayout(host_count=3, future_count=4, action_count=2)
    output = torch.arange(9 * 4096, dtype=torch.float32).reshape(1, 9, 4096)
    host, future, action = split_wsa_joint_attention(output, layout=layout)
    torch.testing.assert_close(host, output[:, :3])
    torch.testing.assert_close(future, output[:, 3:7])
    torch.testing.assert_close(action, output[:, 7:])

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from picf_next.lingbot_native.wsa_full_depth_adaptation import (  # noqa: E402
    WSA_COMMIT,
    WSA_PREPROCESS_SHA256,
    WSA_SOURCE_ARCHIVE_SHA256,
    adapt_wsa_future_state_dict,
    build_nearest_depth_assignments,
    official_resize_tensor_to_shape,
    official_resize_with_alpha,
    percent_align_source_layers,
    repeat_lingbot_kv_heads,
    source_key_for_target_key,
)


def test_wsa_source_provenance_is_frozen() -> None:
    assert WSA_COMMIT == "bfee742c585d5ee85722e658978111934c926ca3"
    assert WSA_PREPROCESS_SHA256 == (
        "8cec126303435e4049d31ee04291517a4d23b518de70f96d50678fd7afda29a8"
    )
    assert WSA_SOURCE_ARCHIVE_SHA256 == (
        "8d9cadb6f6c1abff8c8fd8354226c076aa0d33d5410f984bdfb03069e0520221"
    )


def test_official_resize_is_identity_when_shape_matches() -> None:
    source = torch.randn(3, 5, dtype=torch.bfloat16)
    resized = official_resize_tensor_to_shape(source, (3, 5))
    assert resized is source


def test_official_resize_matches_sequential_align_corners_formula() -> None:
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    transposed = source.permute(1, 0).contiguous()
    resized_first_axis = F.interpolate(
        transposed.reshape(-1, 1, 2),
        size=4,
        mode="linear",
        align_corners=True,
    ).reshape(3, 4).permute(1, 0).contiguous()
    expected = F.interpolate(
        resized_first_axis.reshape(-1, 1, 3),
        size=5,
        mode="linear",
        align_corners=True,
    ).reshape(4, 5)
    actual = official_resize_tensor_to_shape(source, (4, 5))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_official_alpha_scaling_matches_released_rule() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    resized = official_resize_tensor_to_shape(source, (5, 8)).to(torch.float32)
    actual = official_resize_with_alpha(
        source,
        (5, 8),
        apply_alpha_scaling=True,
    )
    torch.testing.assert_close(actual, resized * (4.0 / 8.0) ** 0.5)


def test_explicit_kv_repeat_is_equivalent_to_native_gqa_attention() -> None:
    generator = torch.Generator().manual_seed(218)
    query = torch.randn(2, 32, 7, 128, generator=generator)
    key = torch.randn(2, 8, 11, 128, generator=generator)
    value = torch.randn(2, 8, 11, 128, generator=generator)

    native = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    explicit_key = repeat_lingbot_kv_heads(key.transpose(1, 2), target_heads=32).transpose(1, 2)
    explicit_value = repeat_lingbot_kv_heads(value.transpose(1, 2), target_heads=32).transpose(1, 2)
    explicit = F.scaled_dot_product_attention(query, explicit_key, explicit_value)
    torch.testing.assert_close(native, explicit, rtol=1e-5, atol=1e-6)


def test_30_to_36_depth_mapping_is_complete_monotonic_and_endpoint_aligned() -> None:
    assignments = build_nearest_depth_assignments(source_depth=30, target_depth=36)
    source_layers = tuple(item.nearest_source_layer for item in assignments)
    assert len(assignments) == 36
    assert source_layers[0] == 0
    assert source_layers[-1] == 29
    assert source_layers == tuple(sorted(source_layers))
    assert set(source_layers) == set(range(30))
    assert max(source_layers.count(index) for index in range(30)) == 2


def test_wsa_intermediate_layers_are_percent_aligned_without_removal() -> None:
    assert percent_align_source_layers(
        (14, 19, 24, 29),
        source_depth=30,
        target_depth=36,
    ) == (17, 23, 29, 35)


def test_strict_state_adaptation_consumes_every_source_and_covers_every_target() -> None:
    source = {
        "blocks.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "blocks.1.weight": torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
        "query_tokens": torch.ones(1, 4, 2),
    }
    target_shapes = {
        "blocks.0.weight": (4, 3),
        "blocks.1.weight": (4, 3),
        "blocks.2.weight": (4, 3),
        "query_tokens": (1, 4, 2),
    }
    adapted, receipt = adapt_wsa_future_state_dict(
        source,
        target_shapes,
        source_depth=2,
        target_depth=3,
    )
    assert set(adapted) == set(target_shapes)
    assert receipt.source_tensor_count == 3
    assert receipt.target_tensor_count == 4
    assert receipt.copied_tensor_count == 1
    assert receipt.resized_tensor_count == 3
    assert receipt.duplicated_source_tensor_count == 1
    assert receipt.unused_source_keys == ()
    torch.testing.assert_close(adapted["blocks.0.weight"], adapted["blocks.1.weight"])
    assert adapted["blocks.0.weight"].data_ptr() != adapted["blocks.1.weight"].data_ptr()


def test_state_adaptation_rejects_an_unused_source_tensor() -> None:
    source = {
        "blocks.0.weight": torch.ones(2, 2),
        "unused": torch.ones(1),
    }
    with pytest.raises(ValueError, match="Unconsumed WSA tensors"):
        adapt_wsa_future_state_dict(
            source,
            {"blocks.0.weight": (2, 2)},
            source_depth=1,
            target_depth=1,
        )


def test_target_block_key_resolves_through_registered_depth_assignment() -> None:
    assignments = build_nearest_depth_assignments(source_depth=30, target_depth=36)
    assert source_key_for_target_key(
        "blocks.35.self_attn.q.weight",
        depth_assignments=assignments,
    ) == "blocks.29.self_attn.q.weight"


@pytest.mark.parametrize("shape", ((2, 3, 4), (2, 8, 4, 16)))
def test_kv_repeat_rejects_incompatible_surfaces(shape: tuple[int, ...]) -> None:
    tensor = torch.zeros(shape)
    with pytest.raises(ValueError):
        repeat_lingbot_kv_heads(tensor, target_heads=30)

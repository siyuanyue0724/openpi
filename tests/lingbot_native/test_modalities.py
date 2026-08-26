from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.modalities import (
    CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
    TOKEN_IDENTITY,
    NativeModalityBatch,
    NativeModalityOmissionPlan,
    NativeModalitySpec,
    NativeModalityStream,
    NativeRelationSurfaceSpec,
    initialize_column_isometry,
    merge_native_modality_batches,
    modality_bridge_input,
    normalized_modality_tokens,
    sample_native_modality_omission,
    validate_modality_specs,
    validate_relation_surface_specs,
)


def _stream(name: str, *, tokens: int, width: int, valid: bool = True) -> NativeModalityStream:
    values = torch.arange(tokens * width, dtype=torch.float32).reshape(1, tokens, width)
    return NativeModalityStream(
        name=name,
        tokens=values,
        valid=torch.full((1, tokens), valid, dtype=torch.bool),
    )


def test_modality_contract_is_sorted_typed_and_shape_bounded() -> None:
    specs = (
        NativeModalitySpec("geometry", 4, 3),
        NativeModalitySpec("touch", 3, 2),
    )
    validate_modality_specs(specs)
    batch = NativeModalityBatch(
        (
            _stream("geometry", tokens=3, width=4),
            _stream("touch", tokens=2, width=3),
        )
    )
    batch.validate_against(specs)
    assert batch.batch_size == 1
    assert batch.token_count == 5
    assert batch.device == torch.device("cpu")
    assert batch.dtype == torch.float32

    with pytest.raises(ValueError, match="sorted"):
        NativeModalityBatch(tuple(reversed(batch.streams)))
    with pytest.raises(ValueError, match="input width"):
        batch.validate_against(
            (NativeModalitySpec("geometry", 5, 3), NativeModalitySpec("touch", 3, 2))
        )
    with pytest.raises(ValueError, match="token budget"):
        batch.validate_against(
            (NativeModalitySpec("geometry", 4, 2), NativeModalitySpec("touch", 3, 2))
        )


def test_whole_modality_omission_removes_every_source_token() -> None:
    batch = NativeModalityBatch(
        (
            _stream("geometry", tokens=3, width=4),
            _stream("touch", tokens=2, width=3),
        )
    )
    omitted = batch.omit(("touch",))
    assert omitted.streams[0] is batch.streams[0]
    assert omitted.streams[1].token_count == 0
    assert omitted.streams[1].valid.shape == (1, 0)
    omitted.validate_against(
        (NativeModalitySpec("geometry", 4, 3), NativeModalitySpec("touch", 3, 2))
    )
    with pytest.raises(ValueError, match="absent"):
        batch.omit(("audio",))


def test_omission_sampling_depends_only_on_declared_availability_and_rng() -> None:
    batch = NativeModalityBatch(
        (
            _stream("geometry", tokens=3, width=4),
            _stream("touch", tokens=2, width=3),
        )
    )
    changed_content = NativeModalityBatch(
        tuple(
            NativeModalityStream(
                name=stream.name,
                tokens=torch.randn_like(stream.tokens),
                valid=stream.valid.clone(),
            )
            for stream in batch.streams
        )
    )

    first = sample_native_modality_omission(batch, seed=19)
    second = sample_native_modality_omission(changed_content, seed=19)

    assert first.omitted_name == second.omitted_name
    assert first.digest == second.digest
    assert torch.equal(first.source_valid, second.source_valid)


def test_omission_plan_rejects_source_availability_mutation() -> None:
    batch = NativeModalityBatch((_stream("touch", tokens=2, width=3),))
    plan = NativeModalityOmissionPlan(
        omitted_name="touch",
        source_valid=torch.ones(1, dtype=torch.bool),
        seed=5,
    )
    unavailable = NativeModalityBatch((_stream("touch", tokens=2, width=3, valid=False),))

    with pytest.raises(ValueError, match="availability differs"):
        plan.apply(unavailable)
    assert plan.apply(batch).streams[0].token_count == 0


def test_modality_boundary_rejects_nonfinite_or_untyped_semantics() -> None:
    values = torch.zeros(1, 2, 4)
    values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        NativeModalityStream(
            name="geometry",
            tokens=values,
            valid=torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="module-safe"):
        NativeModalitySpec("Object-ID", 4, 2)
    with pytest.raises(ValueError, match="at least two"):
        NativeModalitySpec("touch", 1, 2)


def test_modality_normalization_and_device_move_preserve_only_dense_tokens() -> None:
    base = _stream("touch", tokens=2, width=3)
    stream = NativeModalityStream(
        name=base.name,
        tokens=base.tokens,
        valid=base.valid,
        metadata=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        canonical_token_ids=torch.tensor([[0, 1]]),
    )
    normalized = normalized_modality_tokens(stream)
    torch.testing.assert_close(normalized.mean(dim=-1), torch.zeros(1, 2), atol=1e-6, rtol=0)
    moved = NativeModalityBatch((stream,)).to(device="cpu", dtype=torch.bfloat16)
    assert moved.streams[0].tokens.dtype == torch.bfloat16
    assert moved.streams[0].metadata is not None
    assert moved.streams[0].metadata.dtype == torch.bfloat16
    assert moved.streams[0].valid.dtype == torch.bool
    assert moved.streams[0].canonical_token_ids is not None
    assert moved.streams[0].canonical_token_ids.dtype == torch.long


def test_identity_bridge_policy_preserves_values_that_layer_norm_collapses() -> None:
    base = torch.tensor([[[1.0, 2.0, 4.0], [2.0, 5.0, 9.0]]])
    shifted_scaled = 3.0 * base + 7.0
    first = NativeModalityStream(
        name="queries",
        tokens=base,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    second = NativeModalityStream(
        name="queries",
        tokens=shifted_scaled,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    spec = NativeModalitySpec(
        "queries",
        3,
        2,
        token_normalization=TOKEN_IDENTITY,
    )

    torch.testing.assert_close(modality_bridge_input(first, spec), base)
    torch.testing.assert_close(modality_bridge_input(second, spec), shifted_scaled)
    assert not torch.equal(
        modality_bridge_input(first, spec),
        modality_bridge_input(second, spec),
    )
    torch.testing.assert_close(
        normalized_modality_tokens(first),
        normalized_modality_tokens(second),
        atol=2e-6,
        rtol=2e-6,
    )


def test_canonical_token_ids_restore_exact_pair_order_without_entering_features() -> None:
    source = NativeModalityStream(
        name="touch",
        tokens=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        valid=torch.ones(1, 3, dtype=torch.bool),
        metadata=torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]),
        canonical_token_ids=torch.tensor([[0, 1, 2]]),
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = NativeModalityStream(
        name=source.name,
        tokens=source.tokens[:, permutation],
        valid=source.valid[:, permutation],
        metadata=source.metadata[:, permutation],
        canonical_token_ids=source.canonical_token_ids[:, permutation],
    )

    restored = permuted.canonicalized()

    assert torch.equal(restored.tokens, source.tokens)
    assert torch.equal(restored.valid, source.valid)
    assert torch.equal(restored.metadata, source.metadata)
    assert torch.equal(restored.canonical_token_ids, source.canonical_token_ids)


def test_canonical_token_ids_reject_missing_duplicate_or_padded_identity() -> None:
    tokens = torch.zeros(1, 3, 2)
    valid = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="contiguous unique permutation"):
        NativeModalityStream(
            "touch",
            tokens,
            valid,
            canonical_token_ids=torch.tensor([[0, 0, -1]]),
        )
    with pytest.raises(ValueError, match="token id -1"):
        NativeModalityStream(
            "touch",
            tokens,
            valid,
            canonical_token_ids=torch.tensor([[0, 1, 2]]),
        )


def test_metadata_contract_and_omission_are_fail_closed() -> None:
    stream = NativeModalityStream(
        name="touch",
        tokens=torch.randn(1, 2, 3),
        valid=torch.ones(1, 2, dtype=torch.bool),
        metadata=torch.randn(1, 2, 4),
    )
    batch = NativeModalityBatch((stream,))
    batch.validate_against((NativeModalitySpec("touch", 3, 2, metadata_width=4),))
    with pytest.raises(ValueError, match="metadata width"):
        batch.validate_against((NativeModalitySpec("touch", 3, 2),))
    omitted = batch.omit(("touch",)).streams[0]
    assert omitted.metadata is not None
    assert omitted.metadata.shape == (1, 0, 4)


def test_modality_merge_is_lossless_sorted_and_duplicate_closed() -> None:
    touch = NativeModalityBatch((_stream("touch", tokens=2, width=3),))
    geometry = NativeModalityBatch((_stream("geometry", tokens=3, width=4),))
    merged = merge_native_modality_batches((touch, geometry))

    assert tuple(stream.name for stream in merged.streams) == ("geometry", "touch")
    assert merged.streams[0] is geometry.streams[0]
    assert merged.streams[1] is touch.streams[0]
    with pytest.raises(ValueError, match="duplicate"):
        merge_native_modality_batches((touch, touch))


def test_modality_coordinate_bridge_is_full_rank_and_norm_preserving() -> None:
    projection = torch.nn.Linear(5, 8, bias=False)

    initialize_column_isometry(projection)

    gram = projection.weight.T @ projection.weight
    torch.testing.assert_close(gram, torch.eye(5), atol=0.0, rtol=0.0)
    assert torch.linalg.matrix_rank(projection.weight) == 5
    values = torch.randn(7, 5)
    torch.testing.assert_close(
        torch.linalg.vector_norm(projection(values), dim=-1),
        torch.linalg.vector_norm(values, dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="output width"):
        initialize_column_isometry(torch.nn.Linear(8, 5, bias=False))


def test_relation_surfaces_are_geometry_only_and_fail_closed() -> None:
    modalities = (NativeModalitySpec("vjepa", 8, 1152, metadata_width=4),)
    surface = NativeRelationSurfaceSpec(
        name="vjepa",
        geometry_kind="image_grid",
        layout="vjepa21.calvin.static-gripper.24x24.v1",
        target_kind=CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
    )
    validate_relation_surface_specs((surface,), modality_specs=modalities)

    with pytest.raises(ValueError, match="declared modalities"):
        validate_relation_surface_specs(
            (surface,),
            modality_specs=(NativeModalitySpec("sonata", 8, 64, metadata_width=3),),
        )
    with pytest.raises(ValueError, match="explicit source geometry"):
        validate_relation_surface_specs(
            (surface,),
            modality_specs=(NativeModalitySpec("vjepa", 8, 1152),),
        )
    with pytest.raises(ValueError, match="requires the V-JEPA image-grid"):
        NativeRelationSurfaceSpec(
            name="sonata",
            geometry_kind="world_points",
            layout="sonata.calvin.world-points.v1",
            target_kind=CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
        )

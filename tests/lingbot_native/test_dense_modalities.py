from __future__ import annotations

import numpy as np
import pytest
import torch

from picf_next.contracts import DenseEvidence
from picf_next.lingbot_native.dense_modalities import (
    NativeDenseModalityBinding,
    dense_modality_bindings_sha256,
    native_modalities_from_dense_evidence,
)


def _evidence(
    name: str,
    *,
    values: list[list[float]],
    geometry: list[list[float]] | None,
    timestamps: list[float],
    current: list[bool],
    contract: str = "encoder@0123456789abcdef/v1",
) -> DenseEvidence:
    tokens = np.asarray(values, dtype=np.float32)
    return DenseEvidence(
        modality=name,
        encoder_contract=contract,
        tokens=tokens,
        available=True,
        timestamps=np.asarray(timestamps, dtype=np.float32),
        confidence=np.linspace(0.7, 0.9, len(values), dtype=np.float32),
        geometry=(None if geometry is None else np.asarray(geometry, dtype=np.float32)),
        current_measurement_valid=np.asarray(current, dtype=np.bool_),
    )


def _missing(name: str, *, width: int, geometry_width: int) -> DenseEvidence:
    return DenseEvidence(
        modality=name,
        encoder_contract="encoder@0123456789abcdef/v1",
        tokens=np.empty((0, width), dtype=np.float32),
        available=False,
        timestamps=np.empty(0, dtype=np.float32),
        confidence=np.empty(0, dtype=np.float32),
        geometry=(
            None
            if geometry_width == 0
            else np.empty((0, geometry_width), dtype=np.float32)
        ),
        current_measurement_valid=np.empty(0, dtype=np.bool_),
    )


def test_dense_bridge_preserves_tokens_and_exposes_only_typed_metadata() -> None:
    bindings = (
        NativeDenseModalityBinding(
            name="touch",
            encoder_contract="encoder@0123456789abcdef/v1",
            token_width=2,
            maximum_tokens=3,
            geometry_width=2,
        ),
    )
    first = _evidence(
        "touch",
        values=[[1, 2], [3, 4], [5, 6]],
        geometry=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        timestamps=[1.0, 2.0, 3.0],
        current=[False, False, True],
    )
    batch = native_modalities_from_dense_evidence(
        ((first,), (_missing("touch", width=2, geometry_width=2),)),
        bindings,
    )
    stream = batch.streams[0]
    torch.testing.assert_close(stream.tokens[0], torch.tensor(first.tokens))
    assert stream.valid.tolist() == [[True, True, True], [False, False, False]]
    assert stream.canonical_token_ids is not None
    assert stream.canonical_token_ids.tolist() == [[0, 1, 2], [-1, -1, -1]]
    assert stream.metadata is not None
    assert stream.metadata.shape == (2, 3, 5)
    torch.testing.assert_close(stream.metadata[0, :, :2], torch.tensor(first.geometry))
    torch.testing.assert_close(
        stream.metadata[0, :, 2],
        torch.log1p(torch.tensor([2.0, 1.0, 0.0])),
    )
    torch.testing.assert_close(stream.metadata[0, :, -1], torch.tensor([0.0, 0.0, 1.0]))
    assert not stream.metadata[1].any()
    batch.validate_against(tuple(binding.native_spec for binding in bindings))


def test_dense_bridge_rejects_over_budget_or_stale_encoder_contract() -> None:
    evidence = _evidence(
        "touch",
        values=[[1, 2], [3, 4]],
        geometry=None,
        timestamps=[1.0, 1.0],
        current=[True, True],
    )
    too_small = (
        NativeDenseModalityBinding(
            "touch",
            "encoder@0123456789abcdef/v1",
            2,
            1,
        ),
    )
    with pytest.raises(ValueError, match="token budget"):
        native_modalities_from_dense_evidence(((evidence,),), too_small)

    stale = (
        NativeDenseModalityBinding("touch", "different@contract/v2", 2, 2),
    )
    with pytest.raises(ValueError, match="encoder contract"):
        native_modalities_from_dense_evidence(((evidence,),), stale)


def test_dense_binding_digest_changes_with_upstream_or_budget() -> None:
    first = (
        NativeDenseModalityBinding("touch", "encoder@a/v1", 3, 8),
    )
    second = (
        NativeDenseModalityBinding("touch", "encoder@b/v1", 3, 8),
    )
    third = (
        NativeDenseModalityBinding("touch", "encoder@a/v1", 3, 9),
    )
    assert dense_modality_bindings_sha256(first) != dense_modality_bindings_sha256(second)
    assert dense_modality_bindings_sha256(first) != dense_modality_bindings_sha256(third)

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    NativeModalityBatch,
    NativeModalityStream,
    NativeObjectQuerySpatialRelation,
    NativeObjectQuerySpatialSpec,
)
from picf_next.lingbot_native.physical_relations import (
    ContextualObjectQuerySpatialInput,
    PhysicalEntityReadout,
)


def _spatial_relation(
    *,
    object_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    pixel_valid: torch.Tensor | None = None,
    dense_mask_features: torch.Tensor | None = None,
    segmenter_input_tokens: torch.Tensor | None = None,
    position_cos: torch.Tensor | None = None,
    position_sin: torch.Tensor | None = None,
    patch_grid_shape: tuple[int, int] | None = None,
) -> NativeObjectQuerySpatialRelation:
    batch, queries = object_logits.shape
    pixels = mask_logits.shape[-1]
    query_valid = torch.ones(batch, queries, dtype=torch.bool, device=object_logits.device)
    if pixel_valid is None:
        pixel_valid = torch.ones(batch, pixels, dtype=torch.bool, device=object_logits.device)
    canonical = torch.arange(queries, device=object_logits.device).expand(batch, -1)
    return NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
        object_logits=object_logits,
        mask_logits=mask_logits,
        query_valid=query_valid,
        pixel_valid=pixel_valid,
        canonical_query_ids=canonical,
        grid_shape=(2, 2),
        dense_mask_features=dense_mask_features,
        segmenter_input_tokens=segmenter_input_tokens,
        position_cos=position_cos,
        position_sin=position_sin,
        patch_grid_shape=patch_grid_shape,
    )


def _batch(relation: NativeObjectQuerySpatialRelation) -> NativeModalityBatch:
    batch, queries = relation.object_logits.shape
    tokens = torch.randn(batch, queries, 4, device=relation.object_logits.device)
    return NativeModalityBatch(
        (
            NativeModalityStream(
                name="videomt_queries",
                tokens=tokens,
                valid=relation.query_valid,
                canonical_token_ids=relation.canonical_query_ids,
            ),
        ),
        (relation,),
    )


def _frozen_identity_mask_head(width: int) -> nn.Module:
    head = nn.Linear(width, width, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(width))
    return head.requires_grad_(False).eval()


class _FrozenRefinerProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(
        self,
        *,
        posterior_rows,
        semantic_projection_weight,
        patch_grid_height,
        patch_grid_width,
        **_unused,
    ):
        decoded = posterior_rows @ semantic_projection_weight
        pixel_count = patch_grid_height * patch_grid_width
        support_logits = (decoded[..., :pixel_count] * self.scale).transpose(1, 2)
        return SimpleNamespace(support_logits=support_logits)


def _read(
    relation: NativeObjectQuerySpatialRelation,
    *,
    rows: torch.Tensor,
    query_hidden: torch.Tensor,
):
    readout = PhysicalEntityReadout(rows.shape[-1])
    sensor = torch.ones(rows.shape[0], 1, rows.shape[-1])
    return readout(
        posterior_rows=rows,
        sensor_hidden=sensor,
        sensor_valid=torch.ones(rows.shape[0], 1, dtype=torch.bool),
        object_query_spatial_inputs=(
            ContextualObjectQuerySpatialInput(
                relation=relation,
                query_hidden=query_hidden,
            ),
        ),
    ).surface("videomt_masks"), readout


def test_object_query_spatial_batch_preserves_complete_canonical_relation() -> None:
    relation = _spatial_relation(
        object_logits=torch.tensor([[0.2, -0.7]]),
        mask_logits=torch.tensor([[[2.0, 1.0, -1.0, -2.0], [-2.0, -1.0, 1.0, 2.0]]]),
    )
    batch = _batch(relation)
    batch.validate_object_query_spatial_relations(
        (
            NativeObjectQuerySpatialSpec(
                name="videomt_masks",
                query_modality="videomt_queries",
                geometry_kind="image_grid",
                target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
                layout="videomt.calvin.static.2x2.v1",
            ),
        )
    )
    moved = batch.to(device="cpu", dtype=torch.float64)
    assert moved.object_query_spatial_relations[0].mask_logits.dtype == torch.float64
    assert moved.object_query_spatial_relations[0].grid_shape == (2, 2)
    torch.testing.assert_close(
        moved.object_query_spatial_relations[0].mask_logits.float(),
        relation.mask_logits,
    )


def test_object_query_image_grid_rejects_a_mismatched_pixel_axis() -> None:
    with pytest.raises(ValueError, match="grid shape"):
        _spatial_relation(
            object_logits=torch.ones(1, 2),
            mask_logits=torch.ones(1, 2, 3),
        )


def test_object_query_spatial_marginal_is_a_simplex_and_row_equivariant() -> None:
    torch.manual_seed(701)
    relation = _spatial_relation(
        object_logits=torch.tensor([[1.0, 0.5, -0.25]]),
        mask_logits=torch.randn(1, 3, 4),
    )
    rows = torch.randn(1, 2, 4)
    queries = torch.randn(1, 3, 4)
    surface, readout = _read(relation, rows=rows, query_hidden=queries)
    assert surface.grid_shape == (2, 2)
    permutation = torch.tensor([1, 0])
    sensor = torch.ones(1, 1, 4)
    permuted = readout(
        posterior_rows=rows[:, permutation],
        sensor_hidden=sensor,
        sensor_valid=torch.ones(1, 1, dtype=torch.bool),
        object_query_spatial_inputs=(
            ContextualObjectQuerySpatialInput(relation=relation, query_hidden=queries),
        ),
    ).surface("videomt_masks")

    torch.testing.assert_close(
        surface.ownership.sum(dim=-1),
        torch.ones(1, 4),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        permuted.object_probability,
        surface.object_probability.index_select(-1, permutation),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        permuted.context_probability,
        surface.context_probability,
        rtol=0,
        atol=1e-6,
    )


def test_empty_donor_evidence_routes_pixels_to_context_without_a_threshold() -> None:
    relation = _spatial_relation(
        object_logits=torch.full((1, 3), -50.0),
        mask_logits=torch.full((1, 3, 4), -50.0),
    )
    surface, _readout = _read(
        relation,
        rows=torch.randn(1, 2, 4),
        query_hidden=torch.randn(1, 3, 4),
    )
    assert torch.all(surface.context_probability > 1 - 1e-6)


def test_spatial_loss_reaches_shared_rows_queries_and_complete_donor_evidence() -> None:
    torch.manual_seed(702)
    object_logits = torch.randn(1, 3, requires_grad=True)
    mask_logits = torch.randn(1, 3, 4, requires_grad=True)
    relation = _spatial_relation(object_logits=object_logits, mask_logits=mask_logits)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    queries = torch.randn(1, 3, 4, requires_grad=True)
    surface, readout = _read(relation, rows=rows, query_hidden=queries)
    target = torch.tensor([[0, 0, 1, 2]])
    loss = -surface.ownership_log_probability.gather(-1, target.unsqueeze(-1)).mean()
    loss.backward()

    for gradient in (
        rows.grad,
        queries.grad,
        object_logits.grad,
        mask_logits.grad,
        readout.projection.weight.grad,
        readout.no_object.grad,
        readout.temperature_parameter.grad,
    ):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_invalid_spatial_pixels_are_exactly_absent() -> None:
    valid = torch.tensor([[True, False, True, False]])
    relation = _spatial_relation(
        object_logits=torch.randn(1, 2),
        mask_logits=torch.randn(1, 2, 4),
        pixel_valid=valid,
    )
    surface, _readout = _read(
        relation,
        rows=torch.randn(1, 2, 4),
        query_hidden=torch.randn(1, 2, 4),
    )
    assert not surface.support_logits[:, ~valid[0]].any()
    assert not surface.ownership[:, ~valid[0]].any()


def test_direct_row_mask_surface_is_the_source_dot_product_and_row_equivariant() -> None:
    rows = torch.tensor([[[1.0, 2.0, 0.0, -1.0], [-1.0, 0.0, 2.0, 1.0]]])
    features = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    relation = _spatial_relation(
        object_logits=torch.randn(1, 3),
        mask_logits=torch.randn(1, 3, 4),
        dense_mask_features=features,
    )
    queries = torch.randn(1, 3, 4)
    tied_weight = torch.eye(4)
    readout = PhysicalEntityReadout(4, source_mask_head=_frozen_identity_mask_head(4))
    sensor = torch.ones(1, 1, 4)

    def run(value: torch.Tensor):
        return readout(
            posterior_rows=value,
            sensor_hidden=sensor,
            sensor_valid=torch.ones(1, 1, dtype=torch.bool),
            object_query_spatial_inputs=(
                ContextualObjectQuerySpatialInput(
                    relation=relation,
                    query_hidden=queries,
                    query_projection_weight=tied_weight,
                ),
            ),
        ).surface("videomt_masks")

    surface = run(rows)
    expected = torch.einsum("bkd,bpd->bpk", rows, features)
    torch.testing.assert_close(surface.support_logits, expected, rtol=0, atol=0)
    assert surface.donor_query_probability is None
    torch.testing.assert_close(
        surface.ownership.sum(dim=-1),
        torch.ones(1, 4),
        rtol=0,
        atol=1e-6,
    )
    permutation = torch.tensor([1, 0])
    permuted = run(rows[:, permutation])
    torch.testing.assert_close(
        permuted.support_logits,
        expected.index_select(-1, permutation),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        permuted.context_probability,
        surface.context_probability,
        rtol=0,
        atol=1e-6,
    )


def test_direct_row_mask_loss_reaches_rows_and_the_tied_projection() -> None:
    rows = torch.randn(1, 2, 4, requires_grad=True)
    tied_weight = torch.eye(4, requires_grad=True)
    frozen_features = torch.randn(1, 4, 4)
    relation = _spatial_relation(
        object_logits=torch.randn(1, 3),
        mask_logits=torch.randn(1, 3, 4),
        dense_mask_features=frozen_features,
    )
    readout = PhysicalEntityReadout(4, source_mask_head=_frozen_identity_mask_head(4))
    surface = readout(
        posterior_rows=rows,
        sensor_hidden=torch.ones(1, 1, 4),
        sensor_valid=torch.ones(1, 1, dtype=torch.bool),
        object_query_spatial_inputs=(
            ContextualObjectQuerySpatialInput(
                relation=relation,
                query_hidden=torch.randn(1, 3, 4),
                query_projection_weight=tied_weight,
            ),
        ),
    ).surface("videomt_masks")
    target = torch.tensor([[0, 0, 1, 2]])
    loss = -surface.ownership_log_probability.gather(-1, target.unsqueeze(-1)).mean()
    loss.backward()

    for gradient in (rows.grad, tied_weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert frozen_features.grad is None
    assert readout.projection.weight.grad is None


def test_full_refinement_surface_is_row_equivariant_and_keeps_source_frozen() -> None:
    rows = torch.randn(1, 2, 4, requires_grad=True)
    tied_weight = torch.randn(4, 1024, requires_grad=True)
    relation = _spatial_relation(
        object_logits=torch.randn(1, 3),
        mask_logits=torch.randn(1, 3, 4),
        dense_mask_features=torch.randn(1, 4, 1024),
        segmenter_input_tokens=torch.randn(1, 3 + 5 + 4, 1024),
        position_cos=torch.randn(4, 64),
        position_sin=torch.randn(4, 64),
        patch_grid_shape=(2, 2),
    )
    refiner = _FrozenRefinerProbe()
    readout = PhysicalEntityReadout(4, source_mask_refiner=refiner)

    def run(value: torch.Tensor):
        return readout(
            posterior_rows=value,
            sensor_hidden=torch.ones(1, 1, 4),
            sensor_valid=torch.ones(1, 1, dtype=torch.bool),
            object_query_spatial_inputs=(
                ContextualObjectQuerySpatialInput(
                    relation=relation,
                    query_hidden=torch.randn(1, 3, 4),
                    query_projection_weight=tied_weight,
                ),
            ),
        ).surface("videomt_masks")

    surface = run(rows)
    permutation = torch.tensor([1, 0])
    permuted = run(rows[:, permutation])
    torch.testing.assert_close(
        permuted.support_logits,
        surface.support_logits.index_select(-1, permutation),
    )
    loss = -surface.ownership_log_probability[..., 0].mean()
    loss.backward()
    for gradient in (rows.grad, tied_weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert refiner.scale.grad is None

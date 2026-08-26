"""Typed Stage-PQ/PQM boundaries from exact VidEoMT into shared LingBot.

The historical Stage-PQ arm exposes only the complete query bank. Stage-PQM
additionally retains the released class/mask relation as a dense sidecar while
the query bank enters the full shared host. Neither boundary selects objects or
adds a local semantic decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_MASK_LAYOUT,
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    TOKEN_IDENTITY,
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
    NativeObjectQuerySpatialRelation,
    NativeObjectQuerySpatialSpec,
)
from picf_next.videomt_exact.runtime import (
    VIDEOMT_DINOV3_L_CLASSES,
    VIDEOMT_DINOV3_L_QUERIES,
    VIDEOMT_DINOV3_L_WIDTH,
    ExactVidEoMTOutput,
)

VIDEOMT_QUERY_MODALITY = "videomt_queries"
VIDEOMT_MASK_RELATION = "videomt_masks"
VIDEOMT_STAGE_PQ_INTERFACE = "videomt-stage-pq-query-value-preserving/v1"
VIDEOMT_STAGE_PQM_INTERFACE = "videomt-stage-pqm-complete-query-mask/v1"
VIDEOMT_STAGE_PQMR_INTERFACE = "videomt-stage-pqmr-source-faithful-row-mask/v1"
VIDEOMT_CALVIN_MASK_LAYOUT = CALVIN_VIDEOMT_MASK_LAYOUT


def videomt_query_modality_spec() -> NativeModalitySpec:
    """Declare the exact 200x1024 upstream query bank without resampling."""

    return NativeModalitySpec(
        name=VIDEOMT_QUERY_MODALITY,
        input_width=VIDEOMT_DINOV3_L_WIDTH,
        maximum_tokens=VIDEOMT_DINOV3_L_QUERIES,
        token_normalization=TOKEN_IDENTITY,
    )


def videomt_row_mask_query_modality_spec() -> NativeModalitySpec:
    """Declare semantic queries plus the released mask-embedding metadata."""

    return NativeModalitySpec(
        name=VIDEOMT_QUERY_MODALITY,
        input_width=VIDEOMT_DINOV3_L_WIDTH,
        maximum_tokens=VIDEOMT_DINOV3_L_QUERIES,
        metadata_width=VIDEOMT_DINOV3_L_WIDTH,
        token_normalization=TOKEN_IDENTITY,
        metadata_normalization=TOKEN_IDENTITY,
    )


def videomt_calvin_object_query_spatial_spec() -> NativeObjectQuerySpatialSpec:
    """Declare the complete released CALVIN query-mask coordinate frame."""

    return NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout=VIDEOMT_CALVIN_MASK_LAYOUT,
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    )


@dataclass(frozen=True, slots=True)
class VidEoMTQueryObservation:
    """Latest-frame upstream evidence; no local object or task decision is made."""

    query_tokens: torch.Tensor
    class_logits: torch.Tensor
    mask_logits: torch.Tensor
    propagated_state: torch.Tensor
    query_valid: torch.Tensor
    mask_embeddings: torch.Tensor | None = None
    dense_mask_features: torch.Tensor | None = None
    segmenter_input_tokens: torch.Tensor | None = None
    position_cos: torch.Tensor | None = None
    position_sin: torch.Tensor | None = None
    patch_grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.query_tokens.ndim != 3:
            raise ValueError("query tokens must have shape [batch, query, 1024]")
        batch, queries, width = self.query_tokens.shape
        if (queries, width) != (VIDEOMT_DINOV3_L_QUERIES, VIDEOMT_DINOV3_L_WIDTH):
            raise ValueError("query tokens differ from the released VidEoMT bank")
        if self.class_logits.shape != (
            batch,
            queries,
            VIDEOMT_DINOV3_L_CLASSES + 1,
        ):
            raise ValueError("class logits differ from the released VidEoMT head")
        if self.mask_logits.ndim != 4 or self.mask_logits.shape[:2] != (batch, queries):
            raise ValueError("mask logits must have shape [batch, query, height, width]")
        if self.propagated_state.shape != (batch, queries, width):
            raise ValueError("propagated query state differs from the released VidEoMT state")
        if self.query_valid.shape != (batch, queries) or self.query_valid.dtype != torch.bool:
            raise ValueError("query validity must be boolean [batch, query]")
        floating = (
            self.query_tokens,
            self.class_logits,
            self.mask_logits,
            self.propagated_state,
        )
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("VidEoMT query observations must be floating")
        if (
            any(
                value.device != self.query_tokens.device or value.dtype != self.query_tokens.dtype
                for value in floating[1:]
            )
            or self.query_valid.device != self.query_tokens.device
        ):
            raise ValueError("VidEoMT query observations must share device and dtype")
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("VidEoMT query observations contain NaN or infinity")
        if not self.query_valid.all():
            raise ValueError(
                "the exact Stage-P boundary retains every structural VidEoMT query slot"
            )
        if (self.mask_embeddings is None) != (self.dense_mask_features is None):
            raise ValueError("VidEoMT row-mask decoder outputs must be present together")
        if self.mask_embeddings is not None:
            pixels = self.mask_logits.shape[-2] * self.mask_logits.shape[-1]
            if self.mask_embeddings.shape != (batch, queries, width):
                raise ValueError("VidEoMT mask embeddings have invalid axes")
            if self.dense_mask_features.shape != (batch, pixels, width):
                raise ValueError("VidEoMT dense mask features have invalid axes")
            if any(
                value.device != self.query_tokens.device
                or value.dtype != self.query_tokens.dtype
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
                for value in (self.mask_embeddings, self.dense_mask_features)
            ):
                raise ValueError("VidEoMT row-mask decoder outputs are invalid")
        refinement = (
            self.segmenter_input_tokens,
            self.position_cos,
            self.position_sin,
            self.patch_grid_shape,
        )
        if any(value is not None for value in refinement):
            if any(value is None for value in refinement):
                raise ValueError("VidEoMT refinement evidence must be present atomically")
            segmenter_input = self.segmenter_input_tokens
            position_cos = self.position_cos
            position_sin = self.position_sin
            patch_grid = self.patch_grid_shape
            if (
                segmenter_input is None
                or position_cos is None
                or position_sin is None
                or patch_grid is None
            ):
                raise RuntimeError("validated VidEoMT refinement evidence disappeared")
            patch_count = patch_grid[0] * patch_grid[1]
            if (
                patch_grid[0] <= 0
                or patch_grid[1] <= 0
                or segmenter_input.shape != (batch, queries + 5 + patch_count, width)
                or position_cos.shape[-2:] != (patch_count, 64)
                or position_sin.shape != position_cos.shape
            ):
                raise ValueError("VidEoMT refinement evidence has invalid axes")
            if any(
                value.device != self.query_tokens.device
                or value.dtype != self.query_tokens.dtype
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
                for value in (segmenter_input, position_cos, position_sin)
            ):
                raise ValueError("VidEoMT refinement evidence is invalid")

    @classmethod
    def from_exact_output(cls, output: ExactVidEoMTOutput) -> VidEoMTQueryObservation:
        """Select the latest frame, which is aligned with the propagated state."""

        if not isinstance(output, ExactVidEoMTOutput):
            raise TypeError("VidEoMT observation requires one exact typed output")
        batch, _time, queries, _width = output.query_embeddings.shape
        return cls(
            query_tokens=output.query_embeddings[:, -1],
            class_logits=output.class_logits[:, -1],
            mask_logits=output.mask_logits[:, :, -1],
            propagated_state=output.propagated_queries,
            query_valid=torch.ones(
                batch,
                queries,
                dtype=torch.bool,
                device=output.query_embeddings.device,
            ),
            mask_embeddings=output.latest_mask_embeddings,
            dense_mask_features=(
                None
                if output.latest_mask_features is None
                else output.latest_mask_features.flatten(2).transpose(1, 2)
            ),
            segmenter_input_tokens=output.latest_segmenter_input_tokens,
            position_cos=output.latest_position_cos,
            position_sin=output.latest_position_sin,
            patch_grid_shape=output.latest_patch_grid_shape,
        )

    @property
    def object_probability(self) -> torch.Tensor:
        """Released foreground mass, exposed for diagnostics but never thresholded."""

        return 1.0 - self.class_logits.softmax(dim=-1)[..., -1]

    @property
    def foreground_log_odds(self) -> torch.Tensor:
        """Exact foreground-vs-empty log odds under the released class head."""

        return torch.logsumexp(self.class_logits[..., :-1], dim=-1) - self.class_logits[..., -1]

    def as_native_modality_batch(
        self,
        *,
        dtype: torch.dtype | None = None,
    ) -> NativeModalityBatch:
        """Expose every query without selection, pooling, or resampling.

        A requested dtype is an explicit host-precision adapter. It changes only
        numeric precision; query count, order, width, and validity are retained.
        """

        batch, queries, _width = self.query_tokens.shape
        tokens = self.query_tokens if dtype is None else self.query_tokens.to(dtype=dtype)
        canonical_ids = torch.arange(
            queries,
            dtype=torch.long,
            device=tokens.device,
        ).expand(batch, -1)
        return NativeModalityBatch(
            (
                NativeModalityStream(
                    name=VIDEOMT_QUERY_MODALITY,
                    tokens=tokens,
                    valid=self.query_valid.to(device=tokens.device),
                    canonical_token_ids=canonical_ids,
                ),
            )
        )

    def as_native_pqm_batch(
        self,
        *,
        relation_spec: NativeObjectQuerySpatialSpec,
        dtype: torch.dtype | None = None,
    ) -> NativeModalityBatch:
        """Expose every query and the complete released dense mask relation."""

        if not isinstance(relation_spec, NativeObjectQuerySpatialSpec):
            raise TypeError("Stage-PQM requires a typed object-query spatial spec")
        if (
            relation_spec.name != VIDEOMT_MASK_RELATION
            or relation_spec.query_modality != VIDEOMT_QUERY_MODALITY
            or relation_spec.geometry_kind != "image_grid"
        ):
            raise ValueError("Stage-PQM relation spec differs from the exact VidEoMT boundary")
        height, width = self.mask_logits.shape[-2:]
        query_batch = self.as_native_modality_batch(dtype=dtype)
        tokens = query_batch.streams[0].tokens
        target_dtype = tokens.dtype
        batch, queries, _height, _width = self.mask_logits.shape
        canonical_ids = torch.arange(
            queries,
            dtype=torch.long,
            device=tokens.device,
        ).expand(batch, -1)
        relation = NativeObjectQuerySpatialRelation(
            name=relation_spec.name,
            query_modality=relation_spec.query_modality,
            geometry_kind=relation_spec.geometry_kind,
            target_kind=relation_spec.target_kind,
            layout=relation_spec.layout,
            object_logits=self.foreground_log_odds.to(dtype=target_dtype),
            mask_logits=self.mask_logits.flatten(2).to(dtype=target_dtype),
            query_valid=self.query_valid.to(device=tokens.device),
            pixel_valid=torch.ones(
                batch,
                height * width,
                dtype=torch.bool,
                device=tokens.device,
            ),
            canonical_query_ids=canonical_ids,
            grid_shape=(height, width),
            class_logits=self.class_logits.to(dtype=target_dtype),
        )
        return NativeModalityBatch(query_batch.streams, (relation,))

    def as_native_row_mask_batch(
        self,
        *,
        relation_spec: NativeObjectQuerySpatialSpec,
        dtype: torch.dtype | None = None,
    ) -> NativeModalityBatch:
        """Expose the complete donor and its source-faithful mask-decoder basis."""

        if self.mask_embeddings is None or self.dense_mask_features is None:
            raise ValueError("source-faithful row-mask mode requires decoder outputs")
        if not isinstance(relation_spec, NativeObjectQuerySpatialSpec):
            raise TypeError("row-mask mode requires a typed object-query spatial spec")
        if (
            relation_spec.name != VIDEOMT_MASK_RELATION
            or relation_spec.query_modality != VIDEOMT_QUERY_MODALITY
            or relation_spec.geometry_kind != "image_grid"
        ):
            raise ValueError("row-mask relation spec differs from the exact VidEoMT boundary")
        height, width = self.mask_logits.shape[-2:]
        target_dtype = self.query_tokens.dtype if dtype is None else dtype
        tokens = self.query_tokens.to(dtype=target_dtype)
        metadata = self.mask_embeddings.to(dtype=target_dtype)
        dense_features = self.dense_mask_features.to(dtype=target_dtype)
        batch, queries, _width = tokens.shape
        canonical_ids = torch.arange(
            queries,
            dtype=torch.long,
            device=tokens.device,
        ).expand(batch, -1)
        stream = NativeModalityStream(
            name=VIDEOMT_QUERY_MODALITY,
            tokens=tokens,
            metadata=metadata,
            valid=self.query_valid.to(device=tokens.device),
            canonical_token_ids=canonical_ids,
        )
        relation = NativeObjectQuerySpatialRelation(
            name=relation_spec.name,
            query_modality=relation_spec.query_modality,
            geometry_kind=relation_spec.geometry_kind,
            target_kind=relation_spec.target_kind,
            layout=relation_spec.layout,
            object_logits=self.foreground_log_odds.to(dtype=target_dtype),
            mask_logits=self.mask_logits.flatten(2).to(dtype=target_dtype),
            query_valid=self.query_valid.to(device=tokens.device),
            pixel_valid=torch.ones(
                batch,
                height * width,
                dtype=torch.bool,
                device=tokens.device,
            ),
            canonical_query_ids=canonical_ids,
            grid_shape=(height, width),
            class_logits=self.class_logits.to(dtype=target_dtype),
            dense_mask_features=dense_features,
            segmenter_input_tokens=(
                None
                if self.segmenter_input_tokens is None
                else self.segmenter_input_tokens.to(dtype=target_dtype)
            ),
            position_cos=(
                None if self.position_cos is None else self.position_cos.to(dtype=target_dtype)
            ),
            position_sin=(
                None if self.position_sin is None else self.position_sin.to(dtype=target_dtype)
            ),
            patch_grid_shape=self.patch_grid_shape,
        )
        return NativeModalityBatch((stream,), (relation,))

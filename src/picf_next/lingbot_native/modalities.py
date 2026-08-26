"""Typed dense-token boundary for optional physical modalities.

Tokenizers remain modality-specific because raw signals have incompatible
geometry.  They may only expose local dense features here; object identity,
relevance, persistence, and lifecycle remain responsibilities of the shared
LingBot host and its posterior rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn

NATIVE_RELATION_GEOMETRY_KINDS = frozenset({"image_grid", "world_points", "contact_sites"})
NO_RELATION_TARGET = "none"
CALVIN_VJEPA21_VISIBLE_OWNER_TARGET = "calvin_vjepa21_visible_owner_v1"
CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET = "calvin_videomt_visible_owner_v1"
CALVIN_VIDEOMT_MASK_LAYOUT = "videomt.calvin.static.native-mask-grid.v1"
NATIVE_RELATION_TARGET_KINDS = frozenset(
    {
        NO_RELATION_TARGET,
        CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
        CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    }
)
TOKEN_LAYER_NORM = "layer_norm"
TOKEN_IDENTITY = "identity"
NATIVE_TOKEN_NORMALIZATIONS = frozenset({TOKEN_IDENTITY, TOKEN_LAYER_NORM})


def _modality_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.lower()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
    ):
        raise ValueError("modality names must be nonempty lowercase module-safe identifiers")
    return value


@dataclass(frozen=True, slots=True)
class NativeModalitySpec:
    """Static adapter shape; no semantic or object-specific configuration."""

    name: str
    input_width: int
    maximum_tokens: int
    metadata_width: int = 0
    token_normalization: str = TOKEN_LAYER_NORM
    metadata_normalization: str = TOKEN_LAYER_NORM

    def __post_init__(self) -> None:
        _modality_name(self.name)
        for value in (self.input_width, self.maximum_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("modality widths and token limits must be positive integers")
        if self.input_width < 2:
            raise ValueError("modality input width must be at least two")
        if (
            isinstance(self.metadata_width, bool)
            or not isinstance(self.metadata_width, int)
            or self.metadata_width < 0
            or self.metadata_width == 1
        ):
            raise ValueError("modality metadata width must be zero or at least two")
        if self.token_normalization not in NATIVE_TOKEN_NORMALIZATIONS:
            raise ValueError("modality token normalization is unsupported")
        if self.metadata_normalization not in NATIVE_TOKEN_NORMALIZATIONS:
            raise ValueError("modality metadata normalization is unsupported")
        if not self.metadata_width and self.metadata_normalization != TOKEN_LAYER_NORM:
            raise ValueError("metadata normalization is meaningless without metadata")


@dataclass(frozen=True, slots=True)
class NativeRelationSurfaceSpec:
    """A geometry-only native surface exposed to shared physical rows.

    This declaration cannot contain classes, object IDs, salience, lifecycle or
    task relevance. ``target_kind`` only names a detached dataset projection.
    """

    name: str
    geometry_kind: str
    layout: str
    target_kind: str = NO_RELATION_TARGET

    def __post_init__(self) -> None:
        _modality_name(self.name)
        if self.geometry_kind not in NATIVE_RELATION_GEOMETRY_KINDS:
            raise ValueError("native relation surface geometry kind is unsupported")
        if not isinstance(self.layout, str) or not self.layout.strip():
            raise ValueError("native relation surface layout must be nonempty")
        if self.target_kind not in NATIVE_RELATION_TARGET_KINDS:
            raise ValueError("native relation surface target kind is unsupported")
        if self.target_kind == CALVIN_VJEPA21_VISIBLE_OWNER_TARGET and (
            self.name != "vjepa" or self.geometry_kind != "image_grid"
        ):
            raise ValueError("the CALVIN V-JEPA target requires the V-JEPA image-grid surface")


def validate_relation_surface_specs(
    specs: tuple[NativeRelationSurfaceSpec, ...],
    *,
    modality_specs: tuple[NativeModalitySpec, ...],
) -> None:
    if not isinstance(specs, tuple) or any(
        not isinstance(value, NativeRelationSurfaceSpec) for value in specs
    ):
        raise TypeError("native relation surface specs must be an immutable typed tuple")
    names = tuple(value.name for value in specs)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("native relation surface specs must be sorted with unique names")
    declared = {value.name: value for value in modality_specs}
    if not set(names) <= set(declared):
        raise ValueError("native relation surfaces must name declared modalities")
    for spec in specs:
        if declared[spec.name].metadata_width < 2:
            raise ValueError("native relation surfaces require explicit source geometry")


@dataclass(frozen=True, slots=True)
class NativeObjectQuerySpatialSpec:
    """Static contract for one mature object-query/dense-mask relation.

    The source modality supplies object queries to the shared host. The dense
    spatial evidence remains a sidecar so image resolution does not inflate the
    transformer sequence. This spec declares geometry only; it cannot declare
    object identity, task relevance, selection or lifecycle policy.
    """

    name: str
    query_modality: str
    geometry_kind: str
    layout: str
    target_kind: str = NO_RELATION_TARGET

    def __post_init__(self) -> None:
        _modality_name(self.name)
        _modality_name(self.query_modality)
        if self.geometry_kind not in NATIVE_RELATION_GEOMETRY_KINDS:
            raise ValueError("object-query spatial geometry kind is unsupported")
        if not isinstance(self.layout, str) or not self.layout.strip():
            raise ValueError("object-query spatial layout must be nonempty")
        if self.target_kind not in NATIVE_RELATION_TARGET_KINDS:
            raise ValueError("object-query spatial target kind is unsupported")
        if self.target_kind == CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET and (
            self.name != "videomt_masks"
            or self.query_modality != "videomt_queries"
            or self.geometry_kind != "image_grid"
        ):
            raise ValueError("the CALVIN VidEoMT target requires its frozen query-mask surface")


def validate_object_query_spatial_specs(
    specs: tuple[NativeObjectQuerySpatialSpec, ...],
    *,
    modality_specs: tuple[NativeModalitySpec, ...],
    resampled_modality_names: tuple[str, ...] = (),
) -> None:
    if not isinstance(specs, tuple) or any(
        not isinstance(value, NativeObjectQuerySpatialSpec) for value in specs
    ):
        raise TypeError("object-query spatial specs must be an immutable typed tuple")
    names = tuple(value.name for value in specs)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("object-query spatial specs must be sorted with unique names")
    modalities = {value.name for value in modality_specs}
    if any(value.query_modality not in modalities for value in specs):
        raise ValueError("object-query spatial specs must name a declared query modality")
    resampled = set(resampled_modality_names)
    if any(value.query_modality in resampled for value in specs):
        raise ValueError("object-query spatial sources cannot pass through a query resampler")


@dataclass(frozen=True, slots=True)
class NativeObjectQuerySpatialRelation:
    """Runtime object-query evidence from a complete pretrained spatial codec.

    ``object_logits`` are foreground-vs-empty log odds. ``mask_logits`` are the
    codec's native dense mask logits in canonical query order. No local object
    choice or threshold is represented by this type.
    """

    name: str
    query_modality: str
    geometry_kind: str
    target_kind: str
    layout: str
    object_logits: torch.Tensor
    mask_logits: torch.Tensor
    query_valid: torch.Tensor
    pixel_valid: torch.Tensor
    canonical_query_ids: torch.Tensor
    grid_shape: tuple[int, int] | None = None
    class_logits: torch.Tensor | None = None
    dense_mask_features: torch.Tensor | None = None
    segmenter_input_tokens: torch.Tensor | None = None
    position_cos: torch.Tensor | None = None
    position_sin: torch.Tensor | None = None
    patch_grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _modality_name(self.name)
        _modality_name(self.query_modality)
        if self.geometry_kind not in NATIVE_RELATION_GEOMETRY_KINDS:
            raise ValueError("object-query relation geometry kind is unsupported")
        if self.target_kind not in NATIVE_RELATION_TARGET_KINDS:
            raise ValueError("object-query relation target kind is unsupported")
        if not isinstance(self.layout, str) or not self.layout.strip():
            raise ValueError("object-query relation layout must be nonempty")
        if self.object_logits.ndim != 2 or not self.object_logits.is_floating_point():
            raise ValueError("object-query foreground logits must be floating [batch,query]")
        batch, queries = self.object_logits.shape
        if (
            self.mask_logits.ndim != 3
            or self.mask_logits.shape[:2] != (batch, queries)
            or not self.mask_logits.is_floating_point()
        ):
            raise ValueError("object-query masks must be floating [batch,query,pixel]")
        pixels = self.mask_logits.shape[2]
        if self.geometry_kind == "image_grid":
            if (
                self.grid_shape is None
                or not isinstance(self.grid_shape, tuple)
                or len(self.grid_shape) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in self.grid_shape
                )
                or self.grid_shape[0] * self.grid_shape[1] != pixels
            ):
                raise ValueError(
                    "object-query image-grid shape must be positive and match its pixel axis"
                )
        elif self.grid_shape is not None:
            raise ValueError("only image-grid object-query relations may declare a grid shape")
        if self.query_valid.shape != (batch, queries) or self.query_valid.dtype != torch.bool:
            raise ValueError("object-query validity must be boolean [batch,query]")
        if self.pixel_valid.shape != (batch, pixels) or self.pixel_valid.dtype != torch.bool:
            raise ValueError("object-query pixel validity must be boolean [batch,pixel]")
        if (
            self.canonical_query_ids.shape != (batch, queries)
            or self.canonical_query_ids.dtype != torch.long
        ):
            raise ValueError("object-query canonical ids must be long [batch,query]")
        tensors = (
            self.mask_logits,
            self.query_valid,
            self.pixel_valid,
            self.canonical_query_ids,
        )
        if any(value.device != self.object_logits.device for value in tensors):
            raise ValueError("object-query relation tensors must share one device")
        if self.mask_logits.dtype != self.object_logits.dtype:
            raise ValueError("object-query relation floating tensors must share one dtype")
        if not torch.isfinite(self.object_logits).all() or not torch.isfinite(
            self.mask_logits
        ).all():
            raise ValueError("object-query relation contains NaN or infinity")
        if self.class_logits is not None and (
            self.class_logits.ndim != 3
            or self.class_logits.shape[:2] != (batch, queries)
            or self.class_logits.shape[-1] < 2
            or not self.class_logits.is_floating_point()
            or self.class_logits.device != self.object_logits.device
            or self.class_logits.dtype != self.object_logits.dtype
            or not torch.isfinite(self.class_logits).all()
        ):
            raise ValueError("object-query class logits have invalid axes or values")
        if self.dense_mask_features is not None and (
            self.dense_mask_features.ndim != 3
            or self.dense_mask_features.shape[:2] != (batch, pixels)
            or self.dense_mask_features.shape[2] < 2
            or not self.dense_mask_features.is_floating_point()
            or self.dense_mask_features.device != self.object_logits.device
            or self.dense_mask_features.dtype != self.object_logits.dtype
            or not torch.isfinite(self.dense_mask_features).all()
        ):
            raise ValueError(
                "direct row-mask features must be finite floating [batch,pixel,width]"
            )
        refinement = (
            self.segmenter_input_tokens,
            self.position_cos,
            self.position_sin,
            self.patch_grid_shape,
        )
        if any(value is not None for value in refinement):
            if any(value is None for value in refinement):
                raise ValueError("object-query refinement evidence must be present atomically")
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
                raise RuntimeError(
                    "validated object-query refinement evidence disappeared"
                )
            patch_count = patch_grid[0] * patch_grid[1]
            if (
                patch_grid[0] <= 0
                or patch_grid[1] <= 0
                or segmenter_input.shape != (batch, queries + 5 + patch_count, 1024)
                or position_cos.shape[-2:] != (patch_count, 64)
                or position_sin.shape != position_cos.shape
            ):
                raise ValueError("object-query refinement evidence has invalid axes")
            if any(
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or value.device != self.object_logits.device
                or value.dtype != self.object_logits.dtype
                for value in (segmenter_input, position_cos, position_sin)
            ):
                raise ValueError("object-query refinement evidence is invalid")
        if pixels <= 0 or not self.pixel_valid.any(dim=1).all():
            raise ValueError("every object-query relation sample requires valid spatial evidence")
        if (self.canonical_query_ids.masked_select(~self.query_valid) != -1).any():
            raise ValueError("invalid object queries must use canonical id -1")
        for batch_index in range(batch):
            observed = self.canonical_query_ids[batch_index].masked_select(
                self.query_valid[batch_index]
            )
            expected = torch.arange(observed.numel(), dtype=torch.long, device=observed.device)
            if not torch.equal(observed, expected):
                raise ValueError("object-query relations must use canonical query order")

    @property
    def batch_size(self) -> int:
        return self.object_logits.shape[0]

    @property
    def query_count(self) -> int:
        return self.object_logits.shape[1]

    @property
    def pixel_count(self) -> int:
        return self.mask_logits.shape[2]

    def to(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> NativeObjectQuerySpatialRelation:
        target = torch.device(device)
        return NativeObjectQuerySpatialRelation(
            name=self.name,
            query_modality=self.query_modality,
            geometry_kind=self.geometry_kind,
            target_kind=self.target_kind,
            layout=self.layout,
            object_logits=self.object_logits.to(device=target, dtype=dtype),
            mask_logits=self.mask_logits.to(device=target, dtype=dtype),
            query_valid=self.query_valid.to(device=target),
            pixel_valid=self.pixel_valid.to(device=target),
            canonical_query_ids=self.canonical_query_ids.to(device=target),
            grid_shape=self.grid_shape,
            class_logits=(
                None
                if self.class_logits is None
                else self.class_logits.to(device=target, dtype=dtype)
            ),
            dense_mask_features=(
                None
                if self.dense_mask_features is None
                else self.dense_mask_features.to(device=target, dtype=dtype)
            ),
            segmenter_input_tokens=(
                None
                if self.segmenter_input_tokens is None
                else self.segmenter_input_tokens.to(device=target, dtype=dtype)
            ),
            position_cos=(
                None
                if self.position_cos is None
                else self.position_cos.to(device=target, dtype=dtype)
            ),
            position_sin=(
                None
                if self.position_sin is None
                else self.position_sin.to(device=target, dtype=dtype)
            ),
            patch_grid_shape=self.patch_grid_shape,
        )


@dataclass(frozen=True, slots=True)
class NativeModalityStream:
    """One tokenizer output with padding represented only by ``valid=False``."""

    name: str
    tokens: torch.Tensor
    valid: torch.Tensor
    metadata: torch.Tensor | None = None
    canonical_token_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _modality_name(self.name)
        if self.tokens.ndim != 3 or not self.tokens.is_floating_point():
            raise ValueError("modality tokens must be floating [batch,tokens,width]")
        if self.valid.shape != self.tokens.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError("modality validity must be boolean [batch,tokens]")
        if self.valid.device != self.tokens.device:
            raise ValueError("modality tokens and validity must share one device")
        if not torch.isfinite(self.tokens).all():
            raise ValueError("modality tokens contain NaN or infinity")
        if self.metadata is not None:
            if (
                self.metadata.ndim != 3
                or not self.metadata.is_floating_point()
                or self.metadata.shape[:2] != self.tokens.shape[:2]
                or self.metadata.shape[2] < 2
            ):
                raise ValueError("modality metadata must be floating [batch,tokens,width>=2]")
            if (
                self.metadata.device != self.tokens.device
                or self.metadata.dtype != self.tokens.dtype
            ):
                raise ValueError("modality metadata must share token device and dtype")
            if not torch.isfinite(self.metadata).all():
                raise ValueError("modality metadata contains NaN or infinity")
        if self.canonical_token_ids is not None:
            if (
                self.canonical_token_ids.shape != self.tokens.shape[:2]
                or self.canonical_token_ids.dtype != torch.long
                or self.canonical_token_ids.device != self.tokens.device
            ):
                raise ValueError(
                    "canonical modality token ids must be long [batch,tokens] on the token device"
                )
            if (self.canonical_token_ids.masked_select(~self.valid) != -1).any():
                raise ValueError("invalid modality rows must use canonical token id -1")
            for batch_index in range(self.batch_size):
                observed = self.canonical_token_ids[batch_index].masked_select(
                    self.valid[batch_index]
                )
                expected = torch.arange(
                    observed.numel(),
                    dtype=torch.long,
                    device=observed.device,
                )
                if not torch.equal(observed.sort().values, expected):
                    raise ValueError(
                        "valid canonical modality token ids must be a contiguous unique permutation"
                    )

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]

    @property
    def token_count(self) -> int:
        return self.tokens.shape[1]

    @property
    def input_width(self) -> int:
        return self.tokens.shape[2]

    @property
    def metadata_width(self) -> int:
        return 0 if self.metadata is None else self.metadata.shape[2]

    def canonicalized(self) -> NativeModalityStream:
        """Restore encoder-native row order without exposing ids to the model."""

        if self.canonical_token_ids is None:
            raise ValueError("modality stream has no canonical token ids")
        maximum = torch.iinfo(torch.long).max
        keys = torch.where(
            self.valid,
            self.canonical_token_ids,
            torch.full_like(self.canonical_token_ids, maximum),
        )
        order = torch.argsort(keys, dim=1, stable=True)

        def gather_rows(value: torch.Tensor) -> torch.Tensor:
            if value.ndim == 2:
                return value.gather(1, order)
            return value.gather(1, order.unsqueeze(-1).expand(-1, -1, value.shape[-1]))

        return NativeModalityStream(
            name=self.name,
            tokens=gather_rows(self.tokens),
            valid=gather_rows(self.valid),
            metadata=None if self.metadata is None else gather_rows(self.metadata),
            canonical_token_ids=gather_rows(self.canonical_token_ids),
        )


@dataclass(frozen=True, slots=True)
class NativeModalityBatch:
    """Sorted optional streams consumed by one shared-host observation event."""

    streams: tuple[NativeModalityStream, ...]
    object_query_spatial_relations: tuple[NativeObjectQuerySpatialRelation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.streams, tuple) or not self.streams:
            raise ValueError("a modality batch requires at least one typed stream")
        if any(not isinstance(value, NativeModalityStream) for value in self.streams):
            raise TypeError("a modality batch contains an untyped stream")
        names = tuple(value.name for value in self.streams)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("modality streams must be sorted with unique names")
        first = self.streams[0]
        if any(value.batch_size != first.batch_size for value in self.streams):
            raise ValueError("modality streams must share one batch size")
        if any(value.tokens.device != first.tokens.device for value in self.streams):
            raise ValueError("modality streams must share one device")
        if any(value.tokens.dtype != first.tokens.dtype for value in self.streams):
            raise ValueError("modality streams must share one floating dtype")
        if not isinstance(self.object_query_spatial_relations, tuple) or any(
            not isinstance(value, NativeObjectQuerySpatialRelation)
            for value in self.object_query_spatial_relations
        ):
            raise TypeError("object-query spatial relations must be one typed tuple")
        relation_names = tuple(value.name for value in self.object_query_spatial_relations)
        if relation_names != tuple(sorted(relation_names)) or len(set(relation_names)) != len(
            relation_names
        ):
            raise ValueError("object-query spatial relations must be sorted and unique")
        stream_by_name = {value.name: value for value in self.streams}
        for relation in self.object_query_spatial_relations:
            source = stream_by_name.get(relation.query_modality)
            if source is None:
                raise ValueError("object-query relation source modality is absent")
            if (
                relation.batch_size != first.batch_size
                or relation.object_logits.device != first.tokens.device
                or relation.object_logits.dtype != first.tokens.dtype
            ):
                raise ValueError("object-query relation differs from its modality batch")
            canonical = source.canonicalized()
            if (
                canonical.token_count != relation.query_count
                or not torch.equal(canonical.valid, relation.query_valid)
                or canonical.canonical_token_ids is None
                or not torch.equal(canonical.canonical_token_ids, relation.canonical_query_ids)
            ):
                raise ValueError("object-query relation axes differ from canonical source queries")

    @property
    def batch_size(self) -> int:
        return self.streams[0].batch_size

    @property
    def device(self) -> torch.device:
        return self.streams[0].tokens.device

    @property
    def dtype(self) -> torch.dtype:
        return self.streams[0].tokens.dtype

    @property
    def token_count(self) -> int:
        return sum(value.token_count for value in self.streams)

    def validate_against(self, specs: tuple[NativeModalitySpec, ...]) -> None:
        """Require exact declared modalities and bounded dense-token geometry."""

        if not isinstance(specs, tuple) or any(
            not isinstance(value, NativeModalitySpec) for value in specs
        ):
            raise TypeError("modality validation requires typed immutable specs")
        if tuple(value.name for value in specs) != tuple(value.name for value in self.streams):
            raise ValueError("runtime modality streams differ from graph declarations")
        for spec, stream in zip(specs, self.streams, strict=True):
            if stream.input_width != spec.input_width:
                raise ValueError(f"modality {spec.name!r} input width differs from its declaration")
            if stream.metadata_width != spec.metadata_width:
                raise ValueError(
                    f"modality {spec.name!r} metadata width differs from its declaration"
                )
            if stream.token_count > spec.maximum_tokens:
                raise ValueError(f"modality {spec.name!r} exceeds its frozen token budget")

    def validate_object_query_spatial_relations(
        self,
        specs: tuple[NativeObjectQuerySpatialSpec, ...],
    ) -> None:
        stream_by_name = {value.name: value for value in self.streams}
        active_specs = tuple(
            value for value in specs if stream_by_name[value.query_modality].token_count > 0
        )
        expected = tuple(
            (value.name, value.query_modality, value.geometry_kind, value.target_kind, value.layout)
            for value in active_specs
        )
        observed = tuple(
            (value.name, value.query_modality, value.geometry_kind, value.target_kind, value.layout)
            for value in self.object_query_spatial_relations
        )
        if observed != expected:
            raise ValueError("runtime object-query relations differ from graph declarations")

    def to(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> NativeModalityBatch:
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("native modality destination dtype must be floating")
        target = torch.device(device)
        return NativeModalityBatch(
            tuple(
                NativeModalityStream(
                    name=stream.name,
                    tokens=stream.tokens.to(device=target, dtype=dtype),
                    valid=stream.valid.to(device=target),
                    metadata=(
                        None
                        if stream.metadata is None
                        else stream.metadata.to(device=target, dtype=dtype)
                    ),
                    canonical_token_ids=(
                        None
                        if stream.canonical_token_ids is None
                        else stream.canonical_token_ids.to(device=target)
                    ),
                )
                for stream in self.streams
            ),
            tuple(
                relation.to(device=target, dtype=dtype)
                for relation in self.object_query_spatial_relations
            ),
        )

    def omit(self, names: tuple[str, ...]) -> NativeModalityBatch:
        """Remove whole source streams before the first shared contextual layer."""

        if not isinstance(names, tuple) or not names or tuple(sorted(set(names))) != names:
            raise ValueError("omitted modality names must be a sorted unique nonempty tuple")
        available = {stream.name for stream in self.streams}
        if not set(names) <= available:
            raise ValueError("an omitted modality is absent from the source batch contract")
        return NativeModalityBatch(
            tuple(
                (
                    NativeModalityStream(
                        name=stream.name,
                        tokens=stream.tokens[:, :0],
                        valid=stream.valid[:, :0],
                        metadata=(None if stream.metadata is None else stream.metadata[:, :0]),
                        canonical_token_ids=(
                            None
                            if stream.canonical_token_ids is None
                            else stream.canonical_token_ids[:, :0]
                        ),
                    )
                    if stream.name in names
                    else stream
                )
                for stream in self.streams
            ),
            tuple(
                relation
                for relation in self.object_query_spatial_relations
                if relation.query_modality not in names
            ),
        )


def merge_native_modality_batches(
    batches: tuple[NativeModalityBatch, ...],
) -> NativeModalityBatch:
    """Merge typed streams without changing content, validity, or ownership."""

    if not isinstance(batches, tuple) or not batches:
        raise ValueError("native modality merge requires a nonempty typed tuple")
    if any(not isinstance(batch, NativeModalityBatch) for batch in batches):
        raise TypeError("native modality merge contains an untyped batch")
    first = batches[0]
    if any(
        batch.batch_size != first.batch_size
        or batch.device != first.device
        or batch.dtype != first.dtype
        for batch in batches[1:]
    ):
        raise ValueError("merged modality batches must share batch, device, and dtype")
    streams = tuple(stream for batch in batches for stream in batch.streams)
    names = tuple(stream.name for stream in streams)
    if len(set(names)) != len(names):
        raise ValueError("merged modality batches contain a duplicate stream")
    relations = tuple(
        relation for batch in batches for relation in batch.object_query_spatial_relations
    )
    relation_names = tuple(relation.name for relation in relations)
    if len(set(relation_names)) != len(relation_names):
        raise ValueError("merged modality batches contain a duplicate object-query relation")
    return NativeModalityBatch(
        tuple(sorted(streams, key=lambda stream: stream.name)),
        tuple(sorted(relations, key=lambda relation: relation.name)),
    )


@dataclass(frozen=True, slots=True)
class NativeModalityOmissionPlan:
    """One label-independent whole-stream omission for a shared-host branch."""

    omitted_name: str
    source_valid: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        _modality_name(self.omitted_name)
        if self.source_valid.ndim != 1 or self.source_valid.dtype != torch.bool:
            raise ValueError("omitted-modality source validity must be boolean [batch]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("omitted-modality seed must be a non-negative integer")
        if not self.source_valid.any():
            raise ValueError("an omitted modality must be available for at least one sample")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "name": self.omitted_name,
                "seed": self.seed,
                "shape": list(self.source_valid.shape),
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        valid = self.source_valid.detach().cpu().contiguous().to(torch.uint8)
        return hashlib.sha256(payload + valid.numpy().tobytes()).hexdigest()

    def apply(self, batch: NativeModalityBatch) -> NativeModalityBatch:
        if not isinstance(batch, NativeModalityBatch):
            raise TypeError("modality omission requires one typed source batch")
        try:
            stream = next(value for value in batch.streams if value.name == self.omitted_name)
        except StopIteration as error:
            raise ValueError("omission plan references an absent modality stream") from error
        observed = stream.valid.any(dim=1)
        if not torch.equal(observed, self.source_valid):
            raise ValueError("omission plan availability differs from the source stream")
        return batch.omit((self.omitted_name,))


def sample_native_modality_omission(
    batch: NativeModalityBatch,
    *,
    seed: int,
    eligible_names: tuple[str, ...] | None = None,
) -> NativeModalityOmissionPlan:
    """Choose one available stream using only names, validity, and CPU RNG."""

    if not isinstance(batch, NativeModalityBatch):
        raise TypeError("modality omission sampling requires one typed batch")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("omitted-modality seed must be a non-negative integer")
    available = tuple(stream.name for stream in batch.streams if stream.valid.any())
    if eligible_names is not None:
        if (
            not isinstance(eligible_names, tuple)
            or tuple(sorted(set(eligible_names))) != eligible_names
            or any(name not in {stream.name for stream in batch.streams} for name in eligible_names)
        ):
            raise ValueError("eligible omission names must be sorted declared modalities")
        allowed = set(eligible_names)
        available = tuple(name for name in available if name in allowed)
    if not available:
        raise ValueError("no eligible modality is available for whole-stream omission")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index = int(torch.randint(len(available), (), generator=generator).item())
    name = available[index]
    stream = next(value for value in batch.streams if value.name == name)
    return NativeModalityOmissionPlan(
        omitted_name=name,
        source_valid=stream.valid.any(dim=1).detach().clone(),
        seed=seed,
    )


def validate_modality_specs(specs: tuple[NativeModalitySpec, ...]) -> None:
    if not isinstance(specs, tuple) or any(
        not isinstance(value, NativeModalitySpec) for value in specs
    ):
        raise TypeError("modality specs must be an immutable typed tuple")
    names = tuple(value.name for value in specs)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("modality specs must be sorted with unique names")


def normalized_modality_tokens(stream: NativeModalityStream) -> torch.Tensor:
    """Apply parameter-free per-token normalization before a linear bridge."""

    if not isinstance(stream, NativeModalityStream):
        raise TypeError("modality normalization requires one typed stream")
    if stream.input_width <= 1:
        raise ValueError("modality normalization requires at least two feature channels")
    result = torch.nn.functional.layer_norm(stream.tokens, (stream.input_width,))
    if not torch.isfinite(result).all():
        raise RuntimeError("normalized modality tokens are not finite")
    return result


def modality_bridge_input(
    stream: NativeModalityStream,
    spec: NativeModalitySpec,
) -> torch.Tensor:
    """Apply the frozen coordinate policy declared by one modality ABI."""

    if not isinstance(stream, NativeModalityStream) or not isinstance(spec, NativeModalitySpec):
        raise TypeError("modality bridge input requires a typed stream and specification")
    if stream.name != spec.name or stream.input_width != spec.input_width:
        raise ValueError("modality stream differs from its bridge specification")
    if spec.token_normalization == TOKEN_IDENTITY:
        return stream.tokens
    return normalized_modality_tokens(stream)


def initialize_column_isometry(linear: nn.Linear) -> None:
    """Initialize a cheap full-column-rank coordinate bridge.

    A signed row injection gives ``W.T @ W == I`` without the cubic QR cost of
    a tall Haar matrix.  The following pretrained host attention immediately
    mixes host coordinates; this boundary only guarantees that it does not
    compress or classify source features before that shared computation.
    """

    if not isinstance(linear, nn.Linear):
        raise TypeError("column-isometric initialization requires one linear layer")
    rows, columns = linear.weight.shape
    if rows < columns:
        raise ValueError("a lossless coordinate bridge requires output width >= input width")
    with torch.no_grad():
        linear.weight.zero_()
        row_indices = torch.randperm(rows, device=linear.weight.device)[:columns]
        column_indices = torch.arange(columns, device=linear.weight.device)
        signs = (
            torch.randint(
                0,
                2,
                (columns,),
                device=linear.weight.device,
                dtype=torch.int64,
            )
            .mul_(2)
            .sub_(1)
            .to(linear.weight.dtype)
        )
        linear.weight[row_indices, column_indices] = signs
        if linear.bias is not None:
            linear.bias.zero_()

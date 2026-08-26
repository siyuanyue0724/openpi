"""Task-independent physical entity relations read from the shared host."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.modalities import NativeObjectQuerySpatialRelation

TASK_INDEPENDENT_PHYSICAL_INTERFACE = "task_independent_physical_entities_v1"
NATIVE_OBJECT_QUERY_POSTERIOR_INTERFACE = "native_object_query_posterior_v1"


@dataclass(frozen=True, slots=True)
class NativeObjectQueryPosteriorOutput:
    """Pair a mature source-query bank with same-index shared-host rows.

    This zero-parameter boundary preserves independent source masks and the
    complete source class/no-object distribution. It performs no query
    selection, context competition, lifecycle decision, or host-to-source
    decoding.
    """

    posterior_rows: torch.Tensor
    relation: NativeObjectQuerySpatialRelation
    interface: str = NATIVE_OBJECT_QUERY_POSTERIOR_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.relation, NativeObjectQuerySpatialRelation):
            raise TypeError("native query posterior requires a typed source relation")
        if (
            self.posterior_rows.ndim != 3
            or not self.posterior_rows.is_floating_point()
            or self.posterior_rows.shape[:2]
            != (self.relation.batch_size, self.relation.query_count)
            or self.posterior_rows.device != self.relation.object_logits.device
            or self.posterior_rows.dtype != self.relation.object_logits.dtype
            or not torch.isfinite(self.posterior_rows).all()
        ):
            raise ValueError("native posterior rows do not match the source query bank")
        if self.relation.class_logits is None:
            raise ValueError("native query posterior requires complete source class logits")
        expected_ids = torch.arange(
            self.relation.query_count,
            dtype=torch.long,
            device=self.relation.canonical_query_ids.device,
        ).expand(self.relation.batch_size, -1)
        if (
            not self.relation.query_valid.all()
            or not torch.equal(self.relation.canonical_query_ids, expected_ids)
        ):
            raise ValueError("native query posterior requires every canonical source query")

    @property
    def support_logits(self) -> torch.Tensor:
        return self.relation.mask_logits.transpose(1, 2)

    @property
    def support_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.support_logits)

    @property
    def object_logits(self) -> torch.Tensor:
        return self.relation.object_logits

    @property
    def object_probability(self) -> torch.Tensor:
        class_logits = self.relation.class_logits
        if class_logits is None:
            raise RuntimeError("validated native query posterior lost source class logits")
        return 1.0 - class_logits.softmax(dim=-1)[..., -1]


@dataclass(frozen=True, slots=True)
class PhysicalRelationSurfaceInput:
    """One native-resolution evidence surface scored by shared host rows.

    The tensor has already passed through the same lossless modality ingress
    projection used by LingBot.  This object carries geometry provenance only;
    it contains no object label, mask, proposal or semantic prediction.
    """

    name: str
    geometry_kind: str
    target_kind: str
    layout: str
    sensor_hidden: torch.Tensor
    sensor_valid: torch.Tensor
    canonical_token_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("physical relation surface name must be nonempty")
        if not isinstance(self.geometry_kind, str) or not self.geometry_kind:
            raise ValueError("physical relation surface geometry kind must be nonempty")
        if not isinstance(self.target_kind, str) or not self.target_kind:
            raise ValueError("physical relation surface target kind must be nonempty")
        if not isinstance(self.layout, str) or not self.layout:
            raise ValueError("physical relation surface layout must be nonempty")
        if self.sensor_hidden.ndim != 3 or not self.sensor_hidden.is_floating_point():
            raise ValueError("physical relation surface hidden states must be floating rank three")
        if (
            self.sensor_valid.shape != self.sensor_hidden.shape[:2]
            or self.sensor_valid.dtype != torch.bool
            or self.sensor_valid.device != self.sensor_hidden.device
        ):
            raise ValueError("physical relation surface validity must match hidden states")
        if not torch.isfinite(self.sensor_hidden).all():
            raise ValueError("physical relation surface hidden states contain NaN or infinity")
        if self.canonical_token_ids is not None and (
            self.canonical_token_ids.shape != self.sensor_valid.shape
            or self.canonical_token_ids.dtype != torch.long
            or self.canonical_token_ids.device != self.sensor_valid.device
            or (self.canonical_token_ids.masked_select(~self.sensor_valid) != -1).any()
        ):
            raise ValueError("physical relation canonical token ids are invalid")


@dataclass(frozen=True, slots=True)
class ContextualObjectQuerySpatialInput:
    """One mature dense codec bound to its full-host contextual query states."""

    relation: NativeObjectQuerySpatialRelation
    query_hidden: torch.Tensor
    query_projection_weight: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation, NativeObjectQuerySpatialRelation):
            raise TypeError("contextual object-query input requires a typed spatial relation")
        if (
            self.query_hidden.ndim != 3
            or not self.query_hidden.is_floating_point()
            or self.query_hidden.shape[:2]
            != (self.relation.batch_size, self.relation.query_count)
        ):
            raise ValueError("contextual object-query hidden states have invalid axes")
        if (
            self.query_hidden.device != self.relation.object_logits.device
            or self.query_hidden.dtype != self.relation.object_logits.dtype
        ):
            raise ValueError("contextual queries and dense relation must share device and dtype")
        if not torch.isfinite(self.query_hidden).all():
            raise ValueError("contextual object-query hidden states contain NaN or infinity")
        direct = self.relation.dense_mask_features is not None
        if direct != (self.query_projection_weight is not None):
            raise ValueError(
                "direct row-mask features and their tied projection must be present together"
            )
        if self.query_projection_weight is not None:
            feature_width = self.relation.dense_mask_features.shape[-1]
            if (
                self.query_projection_weight.ndim != 2
                or self.query_projection_weight.shape
                != (self.query_hidden.shape[-1], feature_width)
                or not self.query_projection_weight.is_floating_point()
                or self.query_projection_weight.device != self.query_hidden.device
                or self.query_projection_weight.dtype != self.query_hidden.dtype
                or not torch.isfinite(self.query_projection_weight).all()
            ):
                raise ValueError("tied semantic-query projection has invalid axes or values")


@dataclass(frozen=True, slots=True)
class PhysicalRelationSurfaceOutput:
    """Native-resolution ownership read by the one physical relation map."""

    name: str
    geometry_kind: str
    target_kind: str
    layout: str
    support_logits: torch.Tensor
    ownership: torch.Tensor
    ownership_log_probability: torch.Tensor
    sensor_valid: torch.Tensor
    grid_shape: tuple[int, int] | None = None
    canonical_token_ids: torch.Tensor | None = None
    donor_query_probability: torch.Tensor | None = None
    donor_context_probability: torch.Tensor | None = None
    contextual_query_ownership: torch.Tensor | None = None
    query_valid: torch.Tensor | None = None
    canonical_query_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.name, "name"),
            (self.geometry_kind, "geometry kind"),
            (self.target_kind, "target kind"),
            (self.layout, "layout"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"physical relation surface {label} must be nonempty")
        if self.support_logits.ndim != 3:
            raise ValueError("physical relation surface support must be rank three")
        batch, tokens, rows = self.support_logits.shape
        if self.grid_shape is not None and (
            self.geometry_kind != "image_grid"
            or not isinstance(self.grid_shape, tuple)
            or len(self.grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.grid_shape
            )
            or self.grid_shape[0] * self.grid_shape[1] != tokens
        ):
            raise ValueError(
                "physical relation grid shape must be positive and match one image-grid surface"
            )
        if (
            self.ownership.shape != (batch, tokens, rows + 1)
            or self.ownership_log_probability.shape != (batch, tokens, rows + 1)
            or self.sensor_valid.shape != (batch, tokens)
            or self.sensor_valid.dtype != torch.bool
        ):
            raise ValueError("physical relation surface output axes are inconsistent")
        tensors = (
            self.support_logits,
            self.ownership,
            self.ownership_log_probability,
        )
        if any(
            not value.is_floating_point()
            or not torch.isfinite(value).all()
            or value.device != self.support_logits.device
            for value in tensors
        ):
            raise ValueError("physical relation surface outputs must be finite and colocated")
        if self.sensor_valid.device != self.support_logits.device:
            raise ValueError("physical relation surface validity must share the output device")
        if ((self.ownership < 0) | (self.ownership > 1)).any():
            raise ValueError("physical relation surface ownership must lie in [0,1]")
        active = self.sensor_valid
        tolerance = max(1e-5, 2 * torch.finfo(self.ownership.dtype).eps)
        if (
            not torch.allclose(
                self.ownership.float().sum(dim=-1)[active],
                torch.ones_like(self.ownership[..., 0].float()[active]),
                rtol=0,
                atol=tolerance,
            )
            or not torch.allclose(
                self.ownership_log_probability.float().exp()[active],
                self.ownership.float()[active],
                rtol=0,
                atol=tolerance,
            )
            or self.support_logits.masked_select(~active.unsqueeze(-1)).any()
            or self.ownership.masked_select(~active.unsqueeze(-1)).any()
        ):
            raise ValueError("physical relation surface categorical output is invalid")
        if self.canonical_token_ids is not None and (
            self.canonical_token_ids.shape != self.sensor_valid.shape
            or self.canonical_token_ids.dtype != torch.long
            or self.canonical_token_ids.device != self.sensor_valid.device
            or (self.canonical_token_ids.masked_select(~self.sensor_valid) != -1).any()
        ):
            raise ValueError("physical relation surface canonical ids are invalid")
        decomposition = (
            self.donor_query_probability,
            self.donor_context_probability,
            self.contextual_query_ownership,
            self.query_valid,
            self.canonical_query_ids,
        )
        if any(value is not None for value in decomposition):
            if (
                self.donor_query_probability is None
                or self.donor_context_probability is None
                or self.contextual_query_ownership is None
                or self.query_valid is None
                or self.canonical_query_ids is None
            ):
                raise ValueError("object-query decomposition must be present as one complete ABI")
            donor_query = self.donor_query_probability
            donor_context = self.donor_context_probability
            query_ownership = self.contextual_query_ownership
            query_valid = self.query_valid
            canonical_query_ids = self.canonical_query_ids
            queries = donor_query.shape[-1]
            if (
                donor_query.shape != (batch, tokens, queries)
                or donor_context.shape != (batch, tokens)
                or query_ownership.shape != (batch, queries, rows + 1)
                or query_valid.shape != (batch, queries)
                or query_valid.dtype != torch.bool
                or canonical_query_ids.shape != (batch, queries)
                or canonical_query_ids.dtype != torch.long
            ):
                raise ValueError("object-query decomposition axes are inconsistent")
            if any(
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or value.device != self.support_logits.device
                for value in (donor_query, donor_context, query_ownership)
            ):
                raise ValueError("object-query decomposition must be finite and colocated")
            if (
                query_valid.device != self.support_logits.device
                or canonical_query_ids.device != self.support_logits.device
                or (canonical_query_ids.masked_select(~query_valid) != -1).any()
                or donor_query.masked_select(~query_valid.unsqueeze(1)).any()
                or query_ownership.masked_select(~query_valid.unsqueeze(-1)).any()
            ):
                raise ValueError("object-query decomposition validity is inconsistent")
            active_pixel_simplex = donor_query.sum(dim=-1) + donor_context
            active_query_simplex = query_ownership.sum(dim=-1)
            if (
                not torch.allclose(
                    active_pixel_simplex.float()[active],
                    torch.ones_like(active_pixel_simplex.float()[active]),
                    rtol=0,
                    atol=tolerance,
                )
                or not torch.allclose(
                    active_query_simplex.float()[query_valid],
                    torch.ones_like(active_query_simplex.float()[query_valid]),
                    rtol=0,
                    atol=tolerance,
                )
            ):
                raise ValueError("object-query decomposition is not categorical")

    @property
    def context_probability(self) -> torch.Tensor:
        return self.ownership[..., -1]

    @property
    def object_probability(self) -> torch.Tensor:
        return self.ownership[..., :-1]


@dataclass(frozen=True, slots=True)
class PhysicalRelationOutput:
    """Prompt-free physical ownership and existence over one entity set."""

    support_logits: torch.Tensor
    visible_support: torch.Tensor
    ownership: torch.Tensor
    ownership_log_probability: torch.Tensor
    existence: torch.Tensor
    existence_logits: torch.Tensor
    row_embeddings: torch.Tensor
    relation_temperature: torch.Tensor
    sensor_valid: torch.Tensor
    structural_sensor_valid: torch.Tensor | None = None
    relation_surfaces: tuple[PhysicalRelationSurfaceOutput, ...] = ()
    interface: str = TASK_INDEPENDENT_PHYSICAL_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.relation_surfaces, tuple) or any(
            not isinstance(surface, PhysicalRelationSurfaceOutput)
            for surface in self.relation_surfaces
        ):
            raise TypeError("physical relation surfaces must be one typed tuple")
        names = tuple(surface.name for surface in self.relation_surfaces)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("physical relation surfaces must be sorted and unique")
        batch, _tokens, rows = self.support_logits.shape
        for surface in self.relation_surfaces:
            if (
                surface.support_logits.shape[0] != batch
                or surface.support_logits.shape[2] != rows
                or surface.support_logits.device != self.support_logits.device
                or surface.support_logits.dtype != self.support_logits.dtype
            ):
                raise ValueError("physical relation surface differs from shared posterior rows")

    @property
    def structural_valid(self) -> torch.Tensor:
        return (
            self.sensor_valid
            if self.structural_sensor_valid is None
            else self.structural_sensor_valid
        )

    @property
    def context_probability(self) -> torch.Tensor:
        return self.ownership[..., -1]

    @property
    def object_probability(self) -> torch.Tensor:
        return self.ownership[..., :-1]

    def surface(self, name: str) -> PhysicalRelationSurfaceOutput:
        """Return one typed native surface, rejecting ambiguous lookup."""

        selected = tuple(surface for surface in self.relation_surfaces if surface.name == name)
        if len(selected) != 1:
            raise KeyError(
                f"expected one physical relation surface {name!r}, found {len(selected)}"
            )
        return selected[0]


class PhysicalEntityReadout(nn.Module):
    """Expose physical row/token relations without task-conditioned parameters."""

    def __init__(
        self,
        host_width: int,
        *,
        source_mask_head: nn.Module | None = None,
        source_mask_refiner: nn.Module | None = None,
        temperature_init: float = 0.07,
        temperature_floor: float = 1e-3,
        norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(host_width, bool) or not isinstance(host_width, int) or host_width <= 0:
            raise ValueError("host_width must be a positive integer")
        controls = (temperature_init, temperature_floor, norm_epsilon)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) for value in controls
        ):
            raise TypeError("physical relation controls must be real-valued")
        if not all(math.isfinite(value) and value > 0 for value in controls):
            raise ValueError("physical relation controls must be finite and positive")
        if temperature_init <= temperature_floor:
            raise ValueError("initial temperature must exceed its numerical floor")
        self.host_width = host_width
        self.temperature_floor = float(temperature_floor)
        self.norm_epsilon = float(norm_epsilon)
        if source_mask_head is not None and source_mask_refiner is not None:
            raise ValueError("source mask head and full refiner are mutually exclusive")
        for source_module in (source_mask_head, source_mask_refiner):
            if source_module is None:
                continue
            parameters = tuple(source_module.parameters())
            if not parameters or any(parameter.requires_grad for parameter in parameters):
                raise ValueError("source spatial module must be frozen and parameterized")
            source_module.eval()
        self.source_mask_head = source_mask_head
        self.source_mask_refiner = source_mask_refiner
        self.projection = nn.Linear(host_width, host_width, bias=False)
        self.existence_projection = nn.Linear(host_width, 1)
        self.no_object = nn.Parameter(torch.empty(host_width))
        inverse_softplus = math.log(math.expm1(temperature_init - temperature_floor))
        self.temperature_parameter = nn.Parameter(torch.tensor([inverse_softplus]))
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(host_width))
            self.existence_projection.weight.zero_()
            self.existence_projection.bias.zero_()
            self.no_object.normal_(mean=0.0, std=host_width**-0.5)

    def train(self, mode: bool = True) -> PhysicalEntityReadout:
        super().train(mode)
        if self.source_mask_head is not None:
            self.source_mask_head.eval()
        if self.source_mask_refiner is not None:
            self.source_mask_refiner.eval()
        return self

    @property
    def temperature(self) -> torch.Tensor:
        return self.temperature_parameter.new_tensor(self.temperature_floor) + F.softplus(
            self.temperature_parameter
        )

    def _project(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(value), dim=-1, eps=self.norm_epsilon)

    def _relation(
        self,
        *,
        rows: torch.Tensor,
        sensor_hidden: torch.Tensor,
        sensor_valid: torch.Tensor,
        no_object: torch.Tensor,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensors = self._project(sensor_hidden)
        support_logits = torch.einsum("bnd,bkd->bnk", sensors, rows) / temperature
        no_object_logits = torch.einsum("bnd,d->bn", sensors, no_object) / temperature
        with torch.autocast(device_type=support_logits.device.type, enabled=False):
            ownership_log_probability = F.log_softmax(
                torch.cat(
                    (support_logits.float(), no_object_logits.float().unsqueeze(-1)),
                    dim=-1,
                ),
                dim=-1,
            )
            ownership = ownership_log_probability.exp()
        minimum = torch.finfo(rows.dtype).min
        ownership_log_probability = ownership_log_probability.masked_fill(
            ~sensor_valid.unsqueeze(-1),
            minimum,
        ).to(rows.dtype)
        ownership = ownership.masked_fill(~sensor_valid.unsqueeze(-1), 0).to(rows.dtype)
        visible_support = torch.sigmoid(support_logits) * sensor_valid.unsqueeze(-1).to(rows.dtype)
        support_logits = support_logits.masked_fill(~sensor_valid.unsqueeze(-1), 0)
        return support_logits, visible_support, ownership, ownership_log_probability

    def _object_query_spatial_surface(
        self,
        *,
        rows: torch.Tensor,
        no_object: torch.Tensor,
        temperature: torch.Tensor,
        value: ContextualObjectQuerySpatialInput,
    ) -> PhysicalRelationSurfaceOutput:
        """Read one complete donor surface through its declared source primitive."""

        relation = value.relation
        if value.query_hidden.shape[-1] != self.host_width:
            raise ValueError("contextual object queries differ from the shared host width")
        if self.source_mask_refiner is not None:
            if value.query_projection_weight is None:
                raise RuntimeError("full row refinement omitted its tied semantic projection")
            if (
                relation.segmenter_input_tokens is None
                or relation.position_cos is None
                or relation.position_sin is None
                or relation.patch_grid_shape is None
            ):
                raise RuntimeError("full row refinement omitted its released token boundary")
            refined = self.source_mask_refiner(
                posterior_rows=rows,
                semantic_projection_weight=value.query_projection_weight,
                segmenter_input_tokens=relation.segmenter_input_tokens,
                position_cos=relation.position_cos,
                position_sin=relation.position_sin,
                patch_grid_height=relation.patch_grid_shape[0],
                patch_grid_width=relation.patch_grid_shape[1],
            )
            support_logits = refined.support_logits.float()
            with torch.autocast(device_type=rows.device.type, enabled=False):
                ownership_log_probability = F.log_softmax(
                    torch.cat(
                        (
                            support_logits,
                            torch.zeros(
                                relation.batch_size,
                                relation.pixel_count,
                                1,
                                dtype=support_logits.dtype,
                                device=support_logits.device,
                            ),
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
                ownership = ownership_log_probability.exp()
            valid = relation.pixel_valid
            support_logits = support_logits.masked_fill(~valid.unsqueeze(-1), 0).to(rows.dtype)
            ownership = ownership.masked_fill(~valid.unsqueeze(-1), 0).to(rows.dtype)
            ownership_log_probability = ownership_log_probability.masked_fill(
                ~valid.unsqueeze(-1),
                torch.finfo(rows.dtype).min,
            ).to(rows.dtype)
            return PhysicalRelationSurfaceOutput(
                name=relation.name,
                geometry_kind=relation.geometry_kind,
                target_kind=relation.target_kind,
                layout=relation.layout,
                support_logits=support_logits,
                ownership=ownership,
                ownership_log_probability=ownership_log_probability,
                sensor_valid=valid,
                grid_shape=relation.grid_shape,
                canonical_token_ids=None,
            )
        if relation.dense_mask_features is not None:
            if value.query_projection_weight is None:
                raise RuntimeError("direct row-mask relation omitted its tied projection")
            if self.source_mask_head is None:
                raise RuntimeError("direct row-mask relation omitted the released mask head")
            with torch.autocast(device_type=rows.device.type, enabled=False):
                decoded_queries = torch.matmul(
                    rows.float(),
                    value.query_projection_weight.float(),
                )
                decoder_parameter = next(self.source_mask_head.parameters())
                decoded_rows = self.source_mask_head(
                    decoded_queries.to(dtype=decoder_parameter.dtype)
                ).float()
                support_logits = torch.einsum(
                    "bkd,bpd->bpk",
                    decoded_rows,
                    relation.dense_mask_features.float(),
                )
                ownership_log_probability = F.log_softmax(
                    torch.cat(
                        (
                            support_logits,
                            torch.zeros(
                                relation.batch_size,
                                relation.pixel_count,
                                1,
                                dtype=support_logits.dtype,
                                device=support_logits.device,
                            ),
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
                ownership = ownership_log_probability.exp()
            valid = relation.pixel_valid
            support_logits = support_logits.masked_fill(~valid.unsqueeze(-1), 0).to(rows.dtype)
            ownership = ownership.masked_fill(~valid.unsqueeze(-1), 0).to(rows.dtype)
            ownership_log_probability = ownership_log_probability.masked_fill(
                ~valid.unsqueeze(-1),
                torch.finfo(rows.dtype).min,
            ).to(rows.dtype)
            return PhysicalRelationSurfaceOutput(
                name=relation.name,
                geometry_kind=relation.geometry_kind,
                target_kind=relation.target_kind,
                layout=relation.layout,
                support_logits=support_logits,
                ownership=ownership,
                ownership_log_probability=ownership_log_probability,
                sensor_valid=valid,
                grid_shape=relation.grid_shape,
                canonical_token_ids=None,
            )
        _support, _visible, query_ownership, _query_log_probability = self._relation(
            rows=rows,
            sensor_hidden=value.query_hidden,
            sensor_valid=relation.query_valid,
            no_object=no_object,
            temperature=temperature,
        )
        with torch.autocast(device_type=rows.device.type, enabled=False):
            query_pixel_energy = (
                relation.object_logits.float().unsqueeze(-1) + relation.mask_logits.float()
            ).transpose(1, 2)
            minimum = torch.finfo(query_pixel_energy.dtype).min
            query_pixel_energy = query_pixel_energy.masked_fill(
                ~relation.query_valid.unsqueeze(1),
                minimum,
            )
            donor_log_probability = F.log_softmax(
                torch.cat(
                    (
                        query_pixel_energy,
                        torch.zeros(
                            relation.batch_size,
                            relation.pixel_count,
                            1,
                            dtype=query_pixel_energy.dtype,
                            device=query_pixel_energy.device,
                        ),
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            donor_probability = donor_log_probability.exp()
            query_probability = donor_probability[..., :-1]
            row_probability = torch.einsum(
                "bpq,bqk->bpk",
                query_probability,
                query_ownership[..., :-1].float(),
            )
            context_probability = donor_probability[..., -1] + torch.einsum(
                "bpq,bq->bp",
                query_probability,
                query_ownership[..., -1].float(),
            )
            ownership = torch.cat((row_probability, context_probability.unsqueeze(-1)), dim=-1)
            ownership = ownership / ownership.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(ownership.dtype).tiny
            )
            ownership_log_probability = ownership.clamp_min(
                torch.finfo(ownership.dtype).tiny
            ).log()
            support_logits = ownership_log_probability[..., :-1] - ownership_log_probability[
                ..., -1:
            ]
        valid = relation.pixel_valid
        support_logits = support_logits.masked_fill(~valid.unsqueeze(-1), 0).to(rows.dtype)
        ownership = ownership.masked_fill(~valid.unsqueeze(-1), 0).to(rows.dtype)
        ownership_log_probability = ownership_log_probability.masked_fill(
            ~valid.unsqueeze(-1),
            torch.finfo(rows.dtype).min,
        ).to(rows.dtype)
        return PhysicalRelationSurfaceOutput(
            name=relation.name,
            geometry_kind=relation.geometry_kind,
            target_kind=relation.target_kind,
            layout=relation.layout,
            support_logits=support_logits,
            ownership=ownership,
            ownership_log_probability=ownership_log_probability,
            sensor_valid=valid,
            grid_shape=relation.grid_shape,
            canonical_token_ids=None,
            donor_query_probability=query_probability.to(rows.dtype),
            donor_context_probability=donor_probability[..., -1].to(rows.dtype),
            contextual_query_ownership=query_ownership,
            query_valid=relation.query_valid,
            canonical_query_ids=relation.canonical_query_ids,
        )

    def forward(
        self,
        *,
        posterior_rows: torch.Tensor,
        sensor_hidden: torch.Tensor,
        sensor_valid: torch.Tensor,
        structural_sensor_valid: torch.Tensor | None = None,
        relation_surfaces: tuple[PhysicalRelationSurfaceInput, ...] = (),
        object_query_spatial_inputs: tuple[ContextualObjectQuerySpatialInput, ...] = (),
    ) -> PhysicalRelationOutput:
        if posterior_rows.ndim != 3 or sensor_hidden.ndim != 3:
            raise ValueError("physical rows and sensor hidden states must be rank three")
        if posterior_rows.shape[0] != sensor_hidden.shape[0]:
            raise ValueError("physical rows and sensors must share a batch")
        if (
            posterior_rows.shape[-1] != self.host_width
            or sensor_hidden.shape[-1] != self.host_width
        ):
            raise ValueError("physical relation inputs must use the configured host width")
        if sensor_valid.shape != sensor_hidden.shape[:2] or sensor_valid.dtype != torch.bool:
            raise ValueError("sensor_valid must be boolean and match sensor tokens")
        if structural_sensor_valid is None:
            structural_sensor_valid = sensor_valid
        if (
            structural_sensor_valid.shape != sensor_valid.shape
            or structural_sensor_valid.dtype != torch.bool
            or structural_sensor_valid.device != sensor_valid.device
            or (structural_sensor_valid & ~sensor_valid).any()
        ):
            raise ValueError("structural sensor validity must be a boolean subset of sensors")
        tensors = (posterior_rows, sensor_hidden)
        parameter = self.projection.weight
        if any(value.device != parameter.device for value in (*tensors, sensor_valid)):
            raise ValueError("physical relation inputs and parameters must share one device")
        if any(value.dtype != parameter.dtype for value in tensors):
            raise ValueError("physical floating inputs and parameters must share one dtype")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("physical relation inputs contain NaN or infinity")
        if not sensor_valid.any(dim=1).all():
            raise ValueError("every sample requires at least one valid sensor token")
        if not isinstance(relation_surfaces, tuple) or any(
            not isinstance(surface, PhysicalRelationSurfaceInput) for surface in relation_surfaces
        ):
            raise TypeError("native physical relation surfaces must be one typed tuple")
        surface_names = tuple(surface.name for surface in relation_surfaces)
        if surface_names != tuple(sorted(surface_names)) or len(set(surface_names)) != len(
            surface_names
        ):
            raise ValueError("native physical relation surfaces must be sorted and unique")
        for surface in relation_surfaces:
            if (
                surface.sensor_hidden.shape[0] != posterior_rows.shape[0]
                or surface.sensor_hidden.shape[-1] != self.host_width
                or surface.sensor_hidden.device != parameter.device
                or surface.sensor_hidden.dtype != parameter.dtype
            ):
                raise ValueError("native physical relation surface differs from the host rows")
        if not isinstance(object_query_spatial_inputs, tuple) or any(
            not isinstance(value, ContextualObjectQuerySpatialInput)
            for value in object_query_spatial_inputs
        ):
            raise TypeError("contextual object-query spatial inputs must be one typed tuple")
        object_surface_names = tuple(value.relation.name for value in object_query_spatial_inputs)
        if object_surface_names != tuple(sorted(object_surface_names)) or len(
            set(object_surface_names)
        ) != len(object_surface_names):
            raise ValueError("contextual object-query spatial inputs must be sorted and unique")
        if set(object_surface_names) & set(surface_names):
            raise ValueError("physical relation surface names must be globally unique")
        for value in object_query_spatial_inputs:
            if (
                value.query_hidden.shape[0] != posterior_rows.shape[0]
                or value.query_hidden.shape[-1] != self.host_width
                or value.query_hidden.device != parameter.device
                or value.query_hidden.dtype != parameter.dtype
            ):
                raise ValueError("contextual object-query input differs from shared host rows")

        rows = self._project(posterior_rows)
        no_object = F.normalize(self.no_object, dim=0, eps=self.norm_epsilon)
        temperature = self.temperature.to(dtype=rows.dtype)
        support_logits, visible_support, ownership, ownership_log_probability = self._relation(
            rows=rows,
            sensor_hidden=sensor_hidden,
            sensor_valid=sensor_valid,
            no_object=no_object,
            temperature=temperature,
        )
        surface_outputs: list[PhysicalRelationSurfaceOutput] = []
        for surface in relation_surfaces:
            surface_support, _visible, surface_ownership, surface_log_probability = self._relation(
                rows=rows,
                sensor_hidden=surface.sensor_hidden,
                sensor_valid=surface.sensor_valid,
                no_object=no_object,
                temperature=temperature,
            )
            surface_outputs.append(
                PhysicalRelationSurfaceOutput(
                    name=surface.name,
                    geometry_kind=surface.geometry_kind,
                    target_kind=surface.target_kind,
                    layout=surface.layout,
                    support_logits=surface_support,
                    ownership=surface_ownership,
                    ownership_log_probability=surface_log_probability,
                    sensor_valid=surface.sensor_valid,
                    canonical_token_ids=surface.canonical_token_ids,
                )
            )
        for value in object_query_spatial_inputs:
            surface_outputs.append(
                self._object_query_spatial_surface(
                    rows=(
                        posterior_rows
                        if value.relation.dense_mask_features is not None
                        else rows
                    ),
                    no_object=no_object,
                    temperature=temperature,
                    value=value,
                )
            )
        existence_logits = self.existence_projection(posterior_rows).squeeze(-1)
        return PhysicalRelationOutput(
            support_logits=support_logits,
            visible_support=visible_support,
            ownership=ownership,
            ownership_log_probability=ownership_log_probability,
            existence=torch.sigmoid(existence_logits),
            existence_logits=existence_logits,
            row_embeddings=rows,
            relation_temperature=temperature,
            sensor_valid=sensor_valid,
            structural_sensor_valid=structural_sensor_valid,
            relation_surfaces=tuple(sorted(surface_outputs, key=lambda value: value.name)),
        )

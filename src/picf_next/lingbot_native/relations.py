"""Read-only physical geometry and host-native task matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

HOST_NATIVE_MATCH_INTERFACE = "host_native_complete_prompt_match"
LEGACY_SHARED_COSINE_INTERFACE = "legacy_shared_cosine"


@dataclass(frozen=True, slots=True)
class RelationOutput:
    support_logits: torch.Tensor
    visible_support: torch.Tensor
    ownership: torch.Tensor
    task_relevance: torch.Tensor
    task_relevance_logits: torch.Tensor
    task_embedding: torch.Tensor | None
    row_embeddings: torch.Tensor
    relation_temperature: torch.Tensor
    dense_task_grounding: torch.Tensor
    dense_task_grounding_logits: torch.Tensor
    existence: torch.Tensor
    existence_logits: torch.Tensor
    sensor_valid: torch.Tensor
    structural_sensor_valid: torch.Tensor | None = None
    match_embeddings: torch.Tensor | None = None
    task_interface: str = LEGACY_SHARED_COSINE_INTERFACE
    task_relevance_logits_fp32: torch.Tensor | None = None
    task_object_log_probability: torch.Tensor | None = None
    task_object_probability: torch.Tensor | None = None
    task_event_distribution: torch.Tensor | None = None
    task_row_probability: torch.Tensor | None = None
    ownership_log_probability: torch.Tensor | None = None

    @property
    def structural_valid(self) -> torch.Tensor:
        return (
            self.sensor_valid
            if self.structural_sensor_valid is None
            else self.structural_sensor_valid
        )

    @property
    def persistent_anchor(self) -> torch.Tensor:
        if self.task_object_probability is not None:
            return self.task_object_probability
        return self.ownership[..., :-1] * self.task_relevance.unsqueeze(1)

    @property
    def display_union(self) -> torch.Tensor:
        return self.dense_task_grounding


class SharedRelationReadout(nn.Module):
    """Physical relation output plus one scalar read from each host match token."""

    def __init__(
        self,
        host_width: int,
        *,
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
            raise TypeError("relation numerical controls must be real-valued")
        if not all(math.isfinite(value) and value > 0 for value in controls):
            raise ValueError("relation numerical controls must be finite and positive")
        if temperature_init <= temperature_floor:
            raise ValueError("initial temperature must exceed its numerical floor")
        self.host_width = host_width
        self.temperature_floor = float(temperature_floor)
        self.norm_epsilon = float(norm_epsilon)
        self.projection = nn.Linear(host_width, host_width, bias=False)
        self.match_projection = nn.Linear(host_width, 1)
        self.existence_projection = nn.Linear(host_width, 1)
        self.no_object = nn.Parameter(torch.empty(host_width))
        inverse_softplus = math.log(math.expm1(temperature_init - temperature_floor))
        self.temperature_parameter = nn.Parameter(torch.tensor([inverse_softplus]))
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(host_width))
            self.existence_projection.weight.zero_()
            self.existence_projection.bias.zero_()
            self.no_object.normal_(mean=0.0, std=host_width**-0.5)

    @property
    def temperature(self) -> torch.Tensor:
        return self.temperature_parameter.new_tensor(self.temperature_floor) + F.softplus(
            self.temperature_parameter
        )

    def _project(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(value), dim=-1, eps=self.norm_epsilon)

    def forward(
        self,
        *,
        posterior_rows: torch.Tensor,
        sensor_hidden: torch.Tensor,
        sensor_valid: torch.Tensor,
        match_hidden: torch.Tensor,
        structural_sensor_valid: torch.Tensor | None = None,
    ) -> RelationOutput:
        if posterior_rows.ndim != 3 or sensor_hidden.ndim != 3:
            raise ValueError("relation rows and sensor hidden states must be rank three")
        if posterior_rows.shape[0] != sensor_hidden.shape[0]:
            raise ValueError("relation rows and sensors must share a batch")
        if (
            posterior_rows.shape[-1] != self.host_width
            or sensor_hidden.shape[-1] != self.host_width
        ):
            raise ValueError("relation inputs must use the configured host width")
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
        if match_hidden.shape != posterior_rows.shape:
            raise ValueError("match_hidden must match posterior row shape")
        tensors = (posterior_rows, sensor_hidden, match_hidden)
        parameter = self.projection.weight
        if any(value.device != parameter.device for value in (*tensors, sensor_valid)):
            raise ValueError("relation inputs and parameters must share one device")
        if any(value.dtype != parameter.dtype for value in tensors):
            raise ValueError("relation floating inputs and parameters must share one dtype")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("relation inputs contain NaN or infinity")
        if not sensor_valid.any(dim=1).all():
            raise ValueError("every relation sample requires at least one valid sensor token")

        rows = self._project(posterior_rows)
        sensors = self._project(sensor_hidden)
        no_object = F.normalize(self.no_object, dim=0, eps=self.norm_epsilon)
        temperature = self.temperature.to(dtype=rows.dtype)
        support_logits = torch.einsum("bnd,bkd->bnk", sensors, rows) / temperature
        no_object_logits = torch.einsum("bnd,d->bn", sensors, no_object) / temperature
        valid_float = sensor_valid.to(support_logits.dtype)
        task_relevance_logits = self.match_projection(match_hidden).squeeze(-1)
        with torch.autocast(device_type=match_hidden.device.type, enabled=False):
            task_relevance_logits_fp32 = F.linear(
                match_hidden.detach().float(),
                self.match_projection.weight.detach().float(),
                (
                    None
                    if self.match_projection.bias is None
                    else self.match_projection.bias.detach().float()
                ),
            ).squeeze(-1)
        invalid_sensor = ~sensor_valid.unsqueeze(-1)
        with torch.autocast(device_type=support_logits.device.type, enabled=False):
            support_logits_fp32 = support_logits.float()
            task_logits_fp32 = task_relevance_logits.float()
            # Each row answers an independent semantic question: is this
            # physical entity relevant to the prompt?  Bernoulli marginals are
            # required here because one instruction may name several entities
            # and because an unknown row must not steal probability mass from a
            # known row.  The MATCH states themselves are still produced by the
            # shared LingBot host; this readout introduces no side model.
            task_log_probability = F.logsigmoid(task_logits_fp32)
            task_row_probability = task_logits_fp32.sigmoid()
            physical_log_ownership = F.log_softmax(
                torch.cat(
                    (
                        support_logits_fp32,
                        no_object_logits.float().unsqueeze(-1),
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            ownership = physical_log_ownership.exp()

            # At each observation token n, Y_r marks whether row r is relevant
            # to the prompt and Z_n is its task-independent physical owner.  We
            # define the factorized model score
            #   A_nr := P(Y_r=1 | Q,O) P(Z_n=r | O).
            # The product is an explicit conditional-independence assumption,
            # not a claim that arbitrary learned marginals identify the true
            # joint.  Both factors receive separate proper losses and the
            # resulting anchor score must pass held-out calibration/visual
            # probes.  It needs no competition across semantically relevant
            # rows, so unknown entities do not steal task probability mass.
            task_object_log_probability = (
                task_log_probability.unsqueeze(1) + physical_log_ownership[..., :-1]
            )
            task_object_probability = task_object_log_probability.exp().masked_fill(
                invalid_sensor,
                0,
            )
            dense_task = task_object_probability.sum(dim=-1)
            task_not_object_probability = (1 - dense_task).clamp(min=0, max=1)
            task_event_distribution = torch.cat(
                (
                    task_object_probability,
                    task_not_object_probability.unsqueeze(-1),
                ),
                dim=-1,
            )
            task_event_distribution = task_event_distribution.masked_fill(
                invalid_sensor,
                0,
            )
            dense_task_logits = torch.logit(dense_task.clamp(min=1e-6, max=1 - 1e-6))

        invalid_log_probability = torch.finfo(rows.dtype).min
        task_object_log_probability = task_object_log_probability.masked_fill(
            invalid_sensor,
            invalid_log_probability,
        ).to(rows.dtype)
        task_object_probability = task_object_probability.to(rows.dtype)
        task_event_distribution = task_event_distribution.to(rows.dtype)
        task_row_probability = task_row_probability.to(rows.dtype)
        task_relevance = task_row_probability
        dense_task = dense_task.to(rows.dtype) * valid_float
        dense_task_logits = dense_task_logits.to(dense_task.dtype)
        dense_task_logits = dense_task_logits.masked_fill(~sensor_valid, 0)
        existence_logits = self.existence_projection(posterior_rows).squeeze(-1)
        return RelationOutput(
            support_logits=support_logits,
            visible_support=torch.sigmoid(support_logits) * valid_float.unsqueeze(-1),
            ownership=ownership.to(rows.dtype) * valid_float.unsqueeze(-1),
            task_relevance=task_relevance,
            task_relevance_logits=task_relevance_logits,
            task_embedding=None,
            row_embeddings=rows,
            relation_temperature=temperature,
            dense_task_grounding=dense_task,
            dense_task_grounding_logits=dense_task_logits,
            existence=torch.sigmoid(existence_logits),
            existence_logits=existence_logits,
            sensor_valid=sensor_valid,
            structural_sensor_valid=structural_sensor_valid,
            match_embeddings=match_hidden,
            task_interface=HOST_NATIVE_MATCH_INTERFACE,
            task_relevance_logits_fp32=task_relevance_logits_fp32,
            task_object_log_probability=task_object_log_probability,
            task_object_probability=task_object_probability,
            task_event_distribution=task_event_distribution,
            task_row_probability=task_row_probability,
            ownership_log_probability=physical_log_ownership,
        )

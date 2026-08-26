"""Equal-capacity bilinear relation probes over frozen LingBot states.

This module is diagnostic-only. It does not replace the production relation
readout, write posterior state, or participate in policy inference.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.relation_depth_probe import (
    RELATION_DEPTH_PROBE_GLOBAL_REFERENCES,
    RELATION_DEPTH_PROBE_LEARNING_RATES,
    RELATION_EXTERNAL_PROBE_PROVENANCE_FIELDS,
    relation_depth_recovery_summary,
    relation_depth_surfaces,
    validate_relation_probe_policy_parameter_boundary,
    validate_relation_probe_provenance,
    validate_relation_probe_visual_artifact,
)
from picf_next.lingbot_native.relations import RelationOutput, SharedRelationReadout

RELATION_BILINEAR_PROBE_ARM = "final_bilinear_recoverability"
RELATION_BILINEAR_PROBE_SCHEMA = "picf-next.relation-bilinear-probe/v1"
RELATION_BILINEAR_PROBE_MODES = (
    "symmetric_indefinite",
    "unconstrained",
)
RELATION_BILINEAR_PROBE_LEARNING_RATES = RELATION_DEPTH_PROBE_LEARNING_RATES
RELATION_BILINEAR_PROBE_WEIGHT_DECAY = 0.01
RELATION_BILINEAR_PROBE_UPDATE_COUNT = 40
RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT = RELATION_BILINEAR_PROBE_UPDATE_COUNT + 1
RELATION_BILINEAR_PROBE_VISUAL_POINTS = (0, 20, 40)
RELATION_BILINEAR_PROBE_CURVE_NAMES = (
    "ownership",
    "ownership_nll",
    "macro_soft_iou",
    "task_soft_iou",
)
RELATION_BILINEAR_PROBE_TRAINABLE_PATHS = (
    "projection.weight",
    "no_object",
    "temperature_parameter",
)
RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES = RELATION_DEPTH_PROBE_GLOBAL_REFERENCES
RELATION_BILINEAR_CONTROL_IDENTITY_FIELDS = (
    "source_commit",
    "checkpoint_revision",
    "patch_sha256",
    "plan_sha256",
    "dataset_manifest_sha256",
    "physical_sidecar_manifest_sha256",
    "seed",
    "fixed_sample_global_step",
    "sample_selection",
    "forward_seed_by_rank",
    "probe_seed_by_rank",
    "frame_sample_keys_by_rank",
    "frame_source_digests_by_rank",
    "row_bindings_by_rank",
    "official_action_by_rank",
    "candidate_initialization_sha256",
    "host_width",
    "host_layer_count",
)
RELATION_BILINEAR_C_CONTROL_IDENTITY_SHA256 = (
    "3e4e1491420024ea8d4db697405bba7a7a1d07a5f8783b4f60094ccabfb08c85"
)
_REPORT_FIELDS = {
    "schema",
    "status",
    "arm",
    "subject_sha256",
    "provenance",
    "policy_parameter_boundary",
    "candidate_initialization_sha256",
    "curve_point_count",
    "optimizer_update_count",
    "candidates",
    "mode_decisions",
    "maximum_peak_reserved_bytes",
    "total_time_s",
}
_CANDIDATE_REPORT_FIELDS = {
    "candidate",
    "trainable_numel",
    "global_curves",
    "rank_reports",
    "recovery",
}
_RANK_REPORT_FIELDS = {
    "rank",
    "curves",
    "gradient_norm_at_first_update",
    "visual_artifacts_by_point",
    "evaluation_times_s",
}


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer no smaller than {minimum}")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return result


def _curve(value: object, *, name: str, points: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numerical sequence")
    if len(value) != points:
        raise ValueError(f"{name} differs from the preregistered curve length")
    maximum = 1.0 if name.endswith(("macro_soft_iou", "task_soft_iou")) else None
    return tuple(
        _finite_float(item, name=f"{name}[{index}]", minimum=0.0, maximum=maximum)
        for index, item in enumerate(value)
    )


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FullRankBilinearRelationReadout(SharedRelationReadout):
    """One equal-capacity full-rank metric used only by frozen-state probes."""

    def __init__(
        self,
        host_width: int,
        *,
        mode: str,
        temperature_init: float = 0.07,
        temperature_floor: float = 1e-3,
        norm_epsilon: float = 1e-6,
    ) -> None:
        if mode not in RELATION_BILINEAR_PROBE_MODES:
            raise ValueError("unknown relation-bilinear probe mode")
        super().__init__(
            host_width,
            temperature_init=temperature_init,
            temperature_floor=temperature_floor,
            norm_epsilon=norm_epsilon,
        )
        self.mode = mode
        self.task_temperature = float(temperature_init)

    def effective_projection(self) -> torch.Tensor:
        """Return the exact matrix used by the cross-role bilinear score."""

        weight = self.projection.weight
        if self.mode == "unconstrained":
            return weight
        return (weight + weight.transpose(0, 1)) * 0.5

    # This sealed pre-ADR-117 diagnostic intentionally preserves the legacy
    # instruction-vector call surface and is never substituted for production.
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        posterior_rows: torch.Tensor,
        sensor_hidden: torch.Tensor,
        sensor_valid: torch.Tensor,
        instruction_hidden: torch.Tensor,
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
            or bool((structural_sensor_valid & ~sensor_valid).any().item())
        ):
            raise ValueError("structural sensor validity must be a boolean subset of sensors")
        if instruction_hidden.shape != (posterior_rows.shape[0], self.host_width):
            raise ValueError("instruction_hidden must have shape [batch, host_width]")
        tensors = (posterior_rows, sensor_hidden, instruction_hidden)
        parameter = self.projection.weight
        if any(value.device != parameter.device for value in (*tensors, sensor_valid)):
            raise ValueError("relation inputs and parameters must share one device")
        if any(value.dtype != parameter.dtype for value in tensors):
            raise ValueError("relation floating inputs and parameters must share one dtype")
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("relation inputs contain NaN or infinity")

        rows = F.normalize(posterior_rows, dim=-1, eps=self.norm_epsilon)
        sensors = F.normalize(sensor_hidden, dim=-1, eps=self.norm_epsilon)
        task = F.normalize(instruction_hidden, dim=-1, eps=self.norm_epsilon)
        projection = self.effective_projection()
        projected_rows = F.linear(rows, projection)
        support_logits = torch.einsum("bnd,bkd->bnk", sensors, projected_rows)
        support_logits = support_logits / self.temperature.to(dtype=support_logits.dtype)

        no_object = F.normalize(self.no_object, dim=0, eps=self.norm_epsilon)
        projected_no_object = F.linear(no_object, projection)
        no_object_logits = torch.einsum("bnd,d->bn", sensors, projected_no_object)
        no_object_logits = no_object_logits / self.temperature.to(dtype=no_object_logits.dtype)
        with torch.autocast(device_type=support_logits.device.type, enabled=False):
            ownership_log_probability = F.log_softmax(
                torch.cat(
                    (support_logits.float(), no_object_logits.float().unsqueeze(-1)),
                    dim=-1,
                ),
                dim=-1,
            )
            ownership = ownership_log_probability.exp()

        task_temperature = task.new_tensor(self.task_temperature)
        task_relevance_logits = torch.einsum("bd,bkd->bk", task, rows) / task_temperature
        dense_task_logits = torch.einsum("bd,bnd->bn", task, sensors) / task_temperature
        valid_float = sensor_valid.to(support_logits.dtype)
        existence_logits = self.existence_projection(posterior_rows).squeeze(-1)
        return RelationOutput(
            support_logits=support_logits,
            visible_support=torch.sigmoid(support_logits) * valid_float.unsqueeze(-1),
            ownership=ownership.to(support_logits.dtype) * valid_float.unsqueeze(-1),
            task_relevance=torch.sigmoid(task_relevance_logits),
            task_relevance_logits=task_relevance_logits,
            task_embedding=task,
            row_embeddings=rows,
            relation_temperature=task_temperature,
            dense_task_grounding=torch.sigmoid(dense_task_logits) * valid_float,
            dense_task_grounding_logits=dense_task_logits,
            existence=torch.sigmoid(existence_logits),
            existence_logits=existence_logits,
            sensor_valid=sensor_valid,
            structural_sensor_valid=structural_sensor_valid,
            ownership_log_probability=ownership_log_probability,
        )


@dataclass(frozen=True, slots=True)
class RelationBilinearCandidate:
    """One preregistered metric family and optimizer setting."""

    candidate_id: str
    mode: str
    learning_rate: float
    learning_rate_index: int

    def __post_init__(self) -> None:
        if self.mode not in RELATION_BILINEAR_PROBE_MODES:
            raise ValueError("unknown relation-bilinear candidate mode")
        expected_id = f"{self.mode}_lr{self.learning_rate_index}"
        if self.candidate_id != expected_id:
            raise ValueError("relation-bilinear candidate identifier is noncanonical")
        if (
            isinstance(self.learning_rate_index, bool)
            or not isinstance(self.learning_rate_index, int)
            or not 0 <= self.learning_rate_index < len(RELATION_BILINEAR_PROBE_LEARNING_RATES)
            or self.learning_rate
            != RELATION_BILINEAR_PROBE_LEARNING_RATES[self.learning_rate_index]
        ):
            raise ValueError("relation-bilinear learning rate differs from the grid")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "learning_rate_hex": self.learning_rate.hex(),
            "learning_rate_index": self.learning_rate_index,
            "surface": {
                "name": "final",
                "post_final_norm": True,
            },
        }


def relation_bilinear_candidates() -> tuple[RelationBilinearCandidate, ...]:
    return tuple(
        RelationBilinearCandidate(
            candidate_id=f"{mode}_lr{index}",
            mode=mode,
            learning_rate=learning_rate,
            learning_rate_index=index,
        )
        for mode in RELATION_BILINEAR_PROBE_MODES
        for index, learning_rate in enumerate(RELATION_BILINEAR_PROBE_LEARNING_RATES)
    )


def relation_bilinear_trainable_parameters(
    readout: FullRankBilinearRelationReadout,
) -> tuple[nn.Parameter, ...]:
    if not isinstance(readout, FullRankBilinearRelationReadout):
        raise TypeError("relation-bilinear probe requires its diagnostic readout")
    named_parameters = dict(readout.named_parameters())
    if not set(RELATION_BILINEAR_PROBE_TRAINABLE_PATHS).issubset(named_parameters):
        raise RuntimeError("relation-bilinear diagnostic parameter surface changed")
    return tuple(named_parameters[name] for name in RELATION_BILINEAR_PROBE_TRAINABLE_PATHS)


def build_relation_bilinear_probe_bank(
    *,
    host_width: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.ModuleDict, tuple[RelationBilinearCandidate, ...], str]:
    """Build bit-identical, full-rank and parameter-disjoint candidates."""

    width = _positive_integer(host_width, name="relation-bilinear host width")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("relation-bilinear seed must be an integer")
    target_device = torch.device(device)
    if not dtype.is_floating_point:
        raise TypeError("relation-bilinear probe dtype must be floating point")
    cuda_devices = (
        []
        if target_device.type != "cuda"
        else [
            target_device.index if target_device.index is not None else torch.cuda.current_device()
        ]
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        prototype = FullRankBilinearRelationReadout(
            width,
            mode=RELATION_BILINEAR_PROBE_MODES[0],
        ).to(device=target_device, dtype=dtype)
    prototype_state = {
        name: value.detach().clone() for name, value in prototype.state_dict().items()
    }
    initialization_sha256 = _state_digest(prototype_state)
    candidates = relation_bilinear_candidates()
    bank = nn.ModuleDict()
    parameter_ids: set[int] = set()
    for candidate in candidates:
        readout = copy.deepcopy(prototype)
        readout.mode = candidate.mode
        readout.load_state_dict(prototype_state, strict=True)
        for parameter in readout.parameters():
            parameter.requires_grad_(False)
        for parameter in relation_bilinear_trainable_parameters(readout):
            parameter.requires_grad_(True)
            if id(parameter) in parameter_ids:
                raise RuntimeError("relation-bilinear candidates share trainable parameters")
            parameter_ids.add(id(parameter))
        if (
            _state_digest({name: value.detach() for name, value in readout.state_dict().items()})
            != initialization_sha256
        ):
            raise RuntimeError("relation-bilinear candidates have different initialization")
        bank[candidate.candidate_id] = readout
    expected = len(candidates) * (width * width + width + 1)
    observed = sum(parameter.numel() for parameter in bank.parameters() if parameter.requires_grad)
    if observed != expected:
        raise RuntimeError("relation-bilinear probe bank changed trainable capacity")
    return bank, candidates, initialization_sha256


def relation_bilinear_decisions(
    candidate_reports: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    """Require adjacent optimizer settings for each bilinear family."""

    expected = relation_bilinear_candidates()
    if len(candidate_reports) != len(expected):
        raise ValueError("relation-bilinear decision omitted a candidate")
    passing_by_mode = {mode: set() for mode in RELATION_BILINEAR_PROBE_MODES}
    observed: set[str] = set()
    for raw, candidate in zip(candidate_reports, expected, strict=True):
        if raw.get("candidate") != candidate.as_dict():
            raise ValueError("relation-bilinear candidate order or descriptor changed")
        if candidate.candidate_id in observed:
            raise ValueError("relation-bilinear decision contains a duplicate")
        observed.add(candidate.candidate_id)
        recovery = raw.get("recovery")
        if not isinstance(recovery, dict) or not isinstance(
            recovery.get("passes_half_recovery"),
            bool,
        ):
            raise ValueError("relation-bilinear candidate lacks a recovery decision")
        if recovery["passes_half_recovery"]:
            passing_by_mode[candidate.mode].add(candidate.learning_rate_index)

    decisions = []
    for mode in RELATION_BILINEAR_PROBE_MODES:
        passing = sorted(passing_by_mode[mode])
        adjacent = [[left, left + 1] for left in passing if left + 1 in passing_by_mode[mode]]
        decisions.append(
            {
                "mode": mode,
                "passing_learning_rate_indices": passing,
                "adjacent_passing_pairs": adjacent,
                "recoverable": bool(adjacent),
            }
        )
    return decisions


def relation_bilinear_probe_subject(
    provenance: dict[str, object],
    *,
    curve_point_count: int,
) -> str:
    """Bind the exact B execution contract into one immutable subject."""

    if set(provenance) != set(RELATION_EXTERNAL_PROBE_PROVENANCE_FIELDS):
        raise ValueError("relation-bilinear provenance fields differ from schema")
    points = _positive_integer(
        curve_point_count,
        name="relation-bilinear curve points",
    )
    if points < 2:
        raise ValueError("relation-bilinear probe requires at least two curve points")
    return _canonical_digest(
        {
            "schema": RELATION_BILINEAR_PROBE_SCHEMA,
            "provenance": provenance,
            "curve_point_count": points,
            "optimizer_update_count": points - 1,
        }
    )


def relation_bilinear_control_identity_sha256(
    provenance: Mapping[str, object],
) -> str:
    """Hash only evidence that B must reproduce exactly from frozen C."""

    missing = set(RELATION_BILINEAR_CONTROL_IDENTITY_FIELDS).difference(provenance)
    if missing:
        raise ValueError(f"relation-bilinear control identity omits fields: {sorted(missing)}")
    return _canonical_digest(
        {name: provenance[name] for name in RELATION_BILINEAR_CONTROL_IDENTITY_FIELDS}
    )


def validate_relation_bilinear_probe_report(value: object) -> dict[str, Any]:
    """Validate B and recompute every aggregate and causal decision."""

    if not isinstance(value, Mapping) or set(value) != _REPORT_FIELDS:
        raise ValueError("relation-bilinear report fields differ from schema")
    if (
        value["schema"] != RELATION_BILINEAR_PROBE_SCHEMA
        or value["status"] != "PASS"
        or value["arm"] != RELATION_BILINEAR_PROBE_ARM
    ):
        raise ValueError("relation-bilinear probe did not complete")
    points = _integer(
        value["curve_point_count"],
        name="relation-bilinear curve points",
        minimum=2,
    )
    updates = _integer(
        value["optimizer_update_count"],
        name="relation-bilinear optimizer updates",
        minimum=1,
    )
    if (
        points != RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT
        or updates != RELATION_BILINEAR_PROBE_UPDATE_COUNT
        or updates != points - 1
    ):
        raise ValueError("relation-bilinear curve length differs from preregistration")

    provenance, capacity, _fixed_step, sample_keys = validate_relation_probe_provenance(
        value["provenance"]
    )
    if (
        relation_bilinear_control_identity_sha256(provenance)
        != RELATION_BILINEAR_C_CONTROL_IDENTITY_SHA256
    ):
        raise ValueError("relation-bilinear execution differs from the frozen C control identity")
    if _sha256(value["subject_sha256"], name="relation-bilinear subject") != (
        relation_bilinear_probe_subject(provenance, curve_point_count=points)
    ):
        raise ValueError("relation-bilinear subject differs from its provenance")
    host_width = _integer(
        provenance["host_width"],
        name="relation-bilinear host width",
        minimum=1,
    )
    num_layers = _integer(
        provenance["host_layer_count"],
        name="relation-bilinear host layers",
        minimum=4,
    )
    expected_surfaces = [surface.as_dict() for surface in relation_depth_surfaces(num_layers)]
    if provenance.get("surfaces") != expected_surfaces:
        raise ValueError("relation-bilinear provenance surfaces changed")
    if provenance.get("capture") != {
        "intermediate_hook": "next_layer_compute_kqv_input_after_block_and_deepstack",
        "final_hook": "post_final_norm",
        "forward_count": 2,
        "feature_dtype": "float32",
        "policy_grad_enabled": False,
        "prediction_queries": "absent",
    }:
        raise ValueError("relation-bilinear capture contract changed")
    if provenance.get("optimizer") != {
        "algorithm": "torch.optim.AdamW",
        "learning_rate_hex_grid": [
            learning_rate.hex() for learning_rate in RELATION_BILINEAR_PROBE_LEARNING_RATES
        ],
        "weight_decay_hex": RELATION_BILINEAR_PROBE_WEIGHT_DECAY.hex(),
        "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "warmup_updates": 0,
        "update_count": RELATION_BILINEAR_PROBE_UPDATE_COUNT,
        "distributed_gradient": "rank_sum_div_world_size",
    }:
        raise ValueError("relation-bilinear optimizer contract changed")
    references = RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES
    if provenance.get("global_references") != {
        "point_zero": dict(cast(Mapping[str, float], references["point_zero"])),
        "structural_full_host_point_40": dict(
            cast(Mapping[str, float], references["structural_full_host_point_40"])
        ),
        "rank_task_soft_iou": [
            dict(item)
            for item in cast(
                Sequence[Mapping[str, float]],
                references["rank_task_soft_iou"],
            )
        ],
    }:
        raise ValueError("relation-bilinear A/D references changed")
    validate_relation_probe_policy_parameter_boundary(
        value["policy_parameter_boundary"],
        host_width=host_width,
    )
    initialization_sha256 = _sha256(
        value["candidate_initialization_sha256"],
        name="relation-bilinear candidate initialization",
    )
    if provenance.get("candidate_initialization_sha256") != initialization_sha256:
        raise ValueError("relation-bilinear provenance omitted candidate initialization")

    raw_candidates = value["candidates"]
    expected_candidates = relation_bilinear_candidates()
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(expected_candidates):
        raise ValueError("relation-bilinear report lacks the full candidate grid")
    normalized_candidates: list[dict[str, object]] = []
    for raw, expected in zip(raw_candidates, expected_candidates, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _CANDIDATE_REPORT_FIELDS:
            raise ValueError("relation-bilinear candidate report fields differ")
        if raw["candidate"] != expected.as_dict():
            raise ValueError("relation-bilinear candidate descriptor or order changed")
        if (
            _integer(
                raw["trainable_numel"],
                name="relation-bilinear trainable elements",
                minimum=1,
            )
            != host_width * host_width + host_width + 1
        ):
            raise ValueError("relation-bilinear candidate capacity changed")
        global_raw = raw["global_curves"]
        if not isinstance(global_raw, Mapping) or set(global_raw) != set(
            RELATION_BILINEAR_PROBE_CURVE_NAMES
        ):
            raise ValueError("relation-bilinear global curves differ")
        global_curves = {
            name: _curve(
                global_raw[name],
                name=f"{expected.candidate_id} global {name}",
                points=points,
            )
            for name in RELATION_BILINEAR_PROBE_CURVE_NAMES
        }
        raw_rank_reports = raw["rank_reports"]
        if not isinstance(raw_rank_reports, list) or len(raw_rank_reports) != 2:
            raise ValueError("relation-bilinear candidate requires two rank reports")
        indexed_rank_curves: list[tuple[int, dict[str, tuple[float, ...]]]] = []
        for raw_rank in raw_rank_reports:
            if not isinstance(raw_rank, Mapping) or set(raw_rank) != _RANK_REPORT_FIELDS:
                raise ValueError("relation-bilinear rank report fields differ")
            rank = _integer(raw_rank["rank"], name="relation-bilinear rank")
            if rank not in (0, 1):
                raise ValueError("relation-bilinear rank must be zero or one")
            curves_raw = raw_rank["curves"]
            if not isinstance(curves_raw, Mapping) or set(curves_raw) != set(
                RELATION_BILINEAR_PROBE_CURVE_NAMES
            ):
                raise ValueError("relation-bilinear rank curves differ")
            indexed_rank_curves.append(
                (
                    rank,
                    {
                        name: _curve(
                            curves_raw[name],
                            name=f"{expected.candidate_id} rank {rank} {name}",
                            points=points,
                        )
                        for name in RELATION_BILINEAR_PROBE_CURVE_NAMES
                    },
                )
            )
            _finite_float(
                raw_rank["gradient_norm_at_first_update"],
                name="relation-bilinear first gradient norm",
                minimum=0.0,
            )
            times = _curve(
                raw_rank["evaluation_times_s"],
                name=(f"{expected.candidate_id} rank {rank} evaluation_times_s"),
                points=points,
            )
            if any(item <= 0 for item in times):
                raise ValueError("relation-bilinear evaluation time must be positive")
            visuals = raw_rank["visual_artifacts_by_point"]
            if not isinstance(visuals, list) or [
                item.get("curve_point") if isinstance(item, Mapping) else None for item in visuals
            ] != list(RELATION_BILINEAR_PROBE_VISUAL_POINTS):
                raise ValueError("relation-bilinear visuals differ from the frozen points")
            for item in visuals:
                if not isinstance(item, Mapping) or set(item) != {
                    "curve_point",
                    "artifacts",
                }:
                    raise ValueError("relation-bilinear visual record fields differ")
                artifacts = item["artifacts"]
                if not isinstance(artifacts, list) or len(artifacts) != 1:
                    raise ValueError("relation-bilinear visual point requires one local sample")
                validate_relation_probe_visual_artifact(
                    artifacts[0],
                    candidate_id=expected.candidate_id,
                    curve_point=item["curve_point"],
                    rank=rank,
                    expected_sample_key=sample_keys[rank][0],
                    capacity=capacity,
                )
        if sorted(rank for rank, _curves in indexed_rank_curves) != [0, 1]:
            raise ValueError("relation-bilinear rank reports are duplicated")
        rank_curves = [
            curves
            for _rank, curves in sorted(
                indexed_rank_curves,
                key=lambda item: item[0],
            )
        ]
        for name in RELATION_BILINEAR_PROBE_CURVE_NAMES:
            recomputed = tuple(
                sum(rank_curve[name][point] for rank_curve in rank_curves) / 2
                for point in range(points)
            )
            if any(
                not math.isclose(
                    observed,
                    expected_mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for observed, expected_mean in zip(
                    global_curves[name],
                    recomputed,
                    strict=True,
                )
            ):
                raise ValueError("relation-bilinear global curve is not the rank mean")
        recovery = relation_depth_recovery_summary(
            global_curves=global_curves,
            rank_task_curves=[rank_curve["task_soft_iou"] for rank_curve in rank_curves],
        )
        if raw["recovery"] != recovery:
            raise ValueError("relation-bilinear recovery decision was not recomputed")
        normalized_candidates.append(dict(raw))
    decisions = relation_bilinear_decisions(normalized_candidates)
    if value["mode_decisions"] != decisions:
        raise ValueError("relation-bilinear mode decision was not recomputed")
    _integer(
        value["maximum_peak_reserved_bytes"],
        name="relation-bilinear peak reserved bytes",
        minimum=1,
    )
    _finite_float(
        value["total_time_s"],
        name="relation-bilinear total time",
        minimum=0.0,
    )
    return dict(value)

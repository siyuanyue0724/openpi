"""Read-only native-depth probes for LingBot relation recoverability.

The probe observes detached residual streams and trains external copies of the
production relation readout. It never writes a host state or a checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import nn

from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import LingBotNativeContext
from picf_next.lingbot_native.relation_geometry_probe import (
    RelationGeometryTrainableScope,
    validate_relation_probe_sample_selection,
)
from picf_next.lingbot_native.relations import SharedRelationReadout
from picf_next.lingbot_native.visual_audit import NATIVE_VISUAL_AUDIT_SCHEMA

RELATION_DEPTH_PROBE_ARM = "host_depth_recoverability"
RELATION_DEPTH_PROBE_SCHEMA = "picf-next.relation-host-depth-probe/v1"
RELATION_DEPTH_PROBE_LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
RELATION_DEPTH_PROBE_WEIGHT_DECAY = 0.01
RELATION_DEPTH_PROBE_SURFACE_NAMES = ("q1", "q2", "q3", "final")
RELATION_DEPTH_PROBE_VISUAL_POINTS = (0, 20, 40)
RELATION_DEPTH_PROBE_CURVE_NAMES = (
    "ownership",
    "ownership_nll",
    "macro_soft_iou",
    "task_soft_iou",
)
RELATION_DEPTH_PROBE_UPDATE_COUNT = 40
RELATION_DEPTH_PROBE_CURVE_POINT_COUNT = RELATION_DEPTH_PROBE_UPDATE_COUNT + 1
RELATION_DEPTH_PROBE_GLOBAL_REFERENCES = MappingProxyType(
    {
        "point_zero": {
            "ownership": 1.4339685440063477,
            "macro_soft_iou": 0.013953149444444445,
            "task_soft_iou": 0.0072143500000000004,
        },
        "structural_full_host_point_40": {
            "ownership": 0.324454590678215,
            "macro_soft_iou": 0.34708294055555555,
            "task_soft_iou": 0.233897,
        },
        "rank_task_soft_iou": (
            {
                "rank": 0,
                "point_zero": 0.0091467,
                "structural_full_host_point_40": 0.261777,
            },
            {
                "rank": 1,
                "point_zero": 0.005282,
                "structural_full_host_point_40": 0.206017,
            },
        ),
    }
)

_TRAINABLE_READOUT_PATHS = (
    "projection.weight",
    "no_object",
    "temperature_parameter",
)
_DEPTH_REPORT_FIELDS = {
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
    "depth_decisions",
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
_DEPTH_PROVENANCE_FIELDS = {
    "source_commit",
    "checkpoint_revision",
    "patch_sha256",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
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
    "surfaces",
    "capture",
    "objective",
    "optimizer",
    "global_references",
}
RELATION_EXTERNAL_PROBE_PROVENANCE_FIELDS = frozenset(_DEPTH_PROVENANCE_FIELDS)
_DEPTH_PROVENANCE_DIGEST_FIELDS = {
    "patch_sha256",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
    "plan_sha256",
    "dataset_manifest_sha256",
    "physical_sidecar_manifest_sha256",
}
_VISUAL_ARTIFACT_FIELDS = {
    "schema",
    "path",
    "sha256",
    "bytes",
    "global_step",
    "input_weight_global_step",
    "weight_boundary",
    "rank",
    "batch_index",
    "sample_key",
    "task",
    "identity_keys",
    "source_time",
    "source_side",
    "source_phase",
    "binding_start_phase",
    "source_binding_valid",
    "row_to_track",
    "sequence_row_to_track",
    "row_existence",
    "row_task_relevance",
    "row_matched_soft_iou",
    "anchor_surface",
    "views",
    "loss_only_labels_visible_to_model",
}


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


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


@dataclass(frozen=True, slots=True)
class RelationDepthSurface:
    """One preregistered native residual surface."""

    name: str
    layer_index: int
    post_final_norm: bool

    def __post_init__(self) -> None:
        if self.name not in RELATION_DEPTH_PROBE_SURFACE_NAMES:
            raise ValueError("unknown relation-depth surface")
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("relation-depth layer index must be non-negative")
        if not isinstance(self.post_final_norm, bool):
            raise TypeError("relation-depth normalization marker must be boolean")
        if self.post_final_norm != (self.name == "final"):
            raise ValueError("only the final relation-depth surface uses the host final norm")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "layer_index": self.layer_index,
            "post_final_norm": self.post_final_norm,
        }


def relation_depth_surfaces(num_layers: int) -> tuple[RelationDepthSurface, ...]:
    """Return the ends of four equal host-depth quartiles."""

    layers = _positive_integer(num_layers, name="relation-depth host layers")
    if layers < 4:
        raise ValueError("relation-depth probing requires at least four host layers")
    indices = tuple(math.ceil(quartile * layers / 4) - 1 for quartile in range(1, 5))
    if len(set(indices)) != 4 or indices[-1] != layers - 1:
        raise ValueError("host depth cannot produce four distinct quartile surfaces")
    return tuple(
        RelationDepthSurface(
            name=name,
            layer_index=index,
            post_final_norm=name == "final",
        )
        for name, index in zip(RELATION_DEPTH_PROBE_SURFACE_NAMES, indices, strict=True)
    )


def _required_module(parent: object, name: str) -> nn.Module:
    value = getattr(parent, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"LingBot module surface {name!r} is absent or malformed")
    return value


def _lingbot_language_model_surfaces(
    policy: nn.Module,
) -> tuple[nn.Module, nn.ModuleList, nn.Module]:
    model = _required_module(policy, "model")
    qwenvl_with_expert = _required_module(model, "qwenvl_with_expert")
    qwenvl = _required_module(qwenvl_with_expert, "qwenvl")
    qwenvl_model = _required_module(qwenvl, "model")
    language_model = _required_module(qwenvl_model, "language_model")
    layers = getattr(language_model, "layers", None)
    norm = getattr(language_model, "norm", None)
    if not isinstance(layers, nn.ModuleList) or not isinstance(norm, nn.Module):
        raise TypeError("LingBot language-model layers or final norm are malformed")
    return language_model, layers, norm


class LingBotRelationDepthCapture:
    """Capture detached native prefix streams without mutating the host."""

    def __init__(self, policy: nn.Module) -> None:
        if not isinstance(policy, nn.Module):
            raise TypeError("relation-depth capture requires a torch module")
        language_model, layers, norm = _lingbot_language_model_surfaces(policy)
        self.surfaces = relation_depth_surfaces(len(layers))
        self._language_model = language_model
        self._layers = layers
        self._norm = norm
        self._captures: dict[str, list[torch.Tensor]] = {
            surface.name: [] for surface in self.surfaces
        }
        self._handles: list[Any] = []
        self._active = False

    @staticmethod
    def _validated_capture(value: object, *, name: str) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[-1] <= 0
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError(f"relation-depth surface {name!r} is malformed")
        captured = value.detach().clone()
        if captured.requires_grad or captured.grad_fn is not None:
            raise RuntimeError("relation-depth capture retained a training graph")
        return captured

    def _next_layer_pre_hook(
        self,
        name: str,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("compute_kqv") is not True:
            return
        if kwargs.get("output_atten") is True:
            raise RuntimeError("relation-depth hook observed conflicting LingBot phases")
        if len(args) != 1:
            raise RuntimeError("LingBot compute_kqv phase changed its hidden-state contract")
        self._captures[name].append(self._validated_capture(args[0], name=name))

    def _norm_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: object,
    ) -> None:
        self._captures["final"].append(self._validated_capture(output, name="final"))

    def __enter__(self) -> LingBotRelationDepthCapture:
        if self._active or self._handles:
            raise RuntimeError("relation-depth capture cannot be entered twice")
        for surface in self.surfaces[:-1]:
            next_layer_index = surface.layer_index + 1
            if next_layer_index >= len(self._layers):
                raise RuntimeError("non-final relation-depth surface lacks a following layer")
            layer = self._layers[next_layer_index]
            self._handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, kwargs, name=surface.name: self._next_layer_pre_hook(
                        name,
                        module,
                        args,
                        kwargs,
                    ),
                    with_kwargs=True,
                )
            )
        self._handles.append(self._norm.register_forward_hook(self._norm_hook))
        self._active = True
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._active = False

    def snapshot(
        self,
        *,
        expected_forward_count: int,
    ) -> MappingProxyType[str, tuple[torch.Tensor, ...]]:
        count = _positive_integer(
            expected_forward_count,
            name="relation-depth expected forward count",
        )
        if self._active:
            raise RuntimeError("relation-depth capture must be closed before publication")
        if set(self._captures) != set(RELATION_DEPTH_PROBE_SURFACE_NAMES):
            raise RuntimeError("relation-depth capture omitted a preregistered surface")
        shapes: set[tuple[tuple[int, ...], ...]] = set()
        frozen: dict[str, tuple[torch.Tensor, ...]] = {}
        for surface in self.surfaces:
            values = tuple(self._captures[surface.name])
            if len(values) != count:
                raise RuntimeError(
                    f"relation-depth surface {surface.name!r} captured "
                    f"{len(values)} forwards instead of {count}"
                )
            shapes.add(tuple(tuple(value.shape) for value in values))
            frozen[surface.name] = values
        if len(shapes) != 1:
            raise RuntimeError("relation-depth surfaces changed the native prefix shape")
        return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class RelationDepthInputs:
    """Production-equivalent relation inputs sliced from one host surface."""

    posterior_rows: torch.Tensor
    sensor_hidden: torch.Tensor
    sensor_valid: torch.Tensor
    match_hidden: torch.Tensor
    legacy_instruction_hidden: torch.Tensor
    structural_sensor_valid: torch.Tensor

    def __post_init__(self) -> None:
        rows = self.posterior_rows
        sensors = self.sensor_hidden
        if rows.ndim != 3 or sensors.ndim != 3:
            raise ValueError("relation-depth rows and sensors must be rank three")
        if rows.shape[0] != sensors.shape[0] or rows.shape[-1] != sensors.shape[-1]:
            raise ValueError("relation-depth rows and sensors have incompatible shapes")
        if self.match_hidden.shape != rows.shape:
            raise ValueError("relation-depth match-token shape is invalid")
        if self.legacy_instruction_hidden.shape != (rows.shape[0], rows.shape[-1]):
            raise ValueError("legacy relation-depth instruction shape is invalid")
        if (
            self.sensor_valid.shape != sensors.shape[:2]
            or self.structural_sensor_valid.shape != sensors.shape[:2]
            or self.sensor_valid.dtype != torch.bool
            or self.structural_sensor_valid.dtype != torch.bool
        ):
            raise ValueError("relation-depth sensor validity is malformed")
        if (self.structural_sensor_valid & ~self.sensor_valid).any():
            raise ValueError("relation-depth structural validity exceeds sensor validity")
        tensors = (rows, sensors, self.match_hidden, self.legacy_instruction_hidden)
        if any(value.device != rows.device or value.dtype != rows.dtype for value in tensors):
            raise ValueError("relation-depth floating inputs must share dtype and device")
        if any(value.requires_grad or value.grad_fn is not None for value in tensors):
            raise ValueError("relation-depth host inputs must be detached")

    def read(self, readout: nn.Module) -> Any:
        common = {
            "posterior_rows": self.posterior_rows,
            "sensor_hidden": self.sensor_hidden,
            "sensor_valid": self.sensor_valid,
            "structural_sensor_valid": self.structural_sensor_valid,
        }
        # Kept only so immutable pre-ADR-117 bilinear reports remain reproducible.
        from picf_next.lingbot_native.relation_bilinear_probe import (
            FullRankBilinearRelationReadout,
        )

        if isinstance(readout, FullRankBilinearRelationReadout):
            return readout(
                **common,
                instruction_hidden=self.legacy_instruction_hidden,
            )
        if isinstance(readout, SharedRelationReadout):
            return readout(**common, match_hidden=self.match_hidden)
        raise TypeError("relation-depth candidate uses an unknown readout interface")


def relation_depth_inputs(
    hidden: torch.Tensor,
    *,
    context: LingBotNativeContext,
    capacity: int,
) -> RelationDepthInputs:
    """Reproduce the exact production role slices at one detached host depth."""

    resolved_capacity = _positive_integer(capacity, name="relation-depth capacity")
    if not isinstance(context, LingBotNativeContext):
        raise TypeError("relation-depth slicing requires a native LingBot context")
    if context.prediction_request is not None:
        raise ValueError("relation-depth diagnostic forbids predictive query tokens")
    native_valid = context.native_valid
    native_roles = context.native_roles
    instruction_index = context.instruction_last_index
    if native_valid is None or native_roles is None or instruction_index is None:
        raise ValueError("relation-depth context lacks bound native prefix metadata")
    if (
        hidden.ndim != 3
        or hidden.shape[0] != native_valid.shape[0]
        or not hidden.is_floating_point()
        or hidden.requires_grad
        or hidden.grad_fn is not None
    ):
        raise ValueError("relation-depth hidden stream must be detached [batch,tokens,width]")
    if (
        native_valid.shape != native_roles.shape
        or native_valid.dtype != torch.bool
        or native_roles.dtype != torch.long
        or native_valid.device != hidden.device
        or native_roles.device != hidden.device
        or instruction_index.device != hidden.device
    ):
        raise ValueError("relation-depth native metadata is malformed")

    batch, original_count = native_valid.shape
    modality_valid_parts = (
        ()
        if context.modalities is None
        else tuple(stream.valid for stream in context.modalities.streams)
    )
    modality_valid = (
        torch.empty(batch, 0, dtype=torch.bool, device=hidden.device)
        if not modality_valid_parts
        else torch.cat(modality_valid_parts, dim=1)
    )
    modality_count = modality_valid.shape[1]
    control_count = context.controls.token_count
    modality_start = original_count
    control_start = modality_start + modality_count
    prior_start = control_start + control_count
    posterior_start = prior_start + resolved_capacity
    posterior_stop = posterior_start + resolved_capacity
    match_start = posterior_stop
    match_stop = match_start + resolved_capacity
    if hidden.shape[1] != match_stop:
        raise ValueError("relation-depth hidden stream differs from the no-prediction layout")

    original = hidden[:, :original_count]
    modality_hidden = hidden[:, modality_start:control_start]
    posterior = hidden[:, posterior_start:posterior_stop]
    match = hidden[:, match_start:match_stop]
    batch_index = torch.arange(batch, device=hidden.device)
    legacy_instruction = original[batch_index, instruction_index]
    native_sensor_valid = native_valid & (native_roles == int(NativeRole.SENSOR))
    sensor_hidden = torch.cat((original, modality_hidden), dim=1)
    sensor_valid = torch.cat((native_sensor_valid, modality_valid), dim=1)
    structural_valid = torch.cat(
        (native_sensor_valid, torch.zeros_like(modality_valid)),
        dim=1,
    )
    return RelationDepthInputs(
        posterior_rows=posterior,
        sensor_hidden=sensor_hidden,
        sensor_valid=sensor_valid,
        match_hidden=match,
        legacy_instruction_hidden=legacy_instruction,
        structural_sensor_valid=structural_valid,
    )


@dataclass(frozen=True, slots=True)
class RelationDepthCandidate:
    """One fixed depth and optimizer setting."""

    candidate_id: str
    surface: RelationDepthSurface
    learning_rate: float
    learning_rate_index: int

    def __post_init__(self) -> None:
        expected_id = f"{self.surface.name}_lr{self.learning_rate_index}"
        if self.candidate_id != expected_id:
            raise ValueError("relation-depth candidate identifier is noncanonical")
        if (
            isinstance(self.learning_rate_index, bool)
            or not isinstance(self.learning_rate_index, int)
            or not 0 <= self.learning_rate_index < len(RELATION_DEPTH_PROBE_LEARNING_RATES)
            or self.learning_rate != RELATION_DEPTH_PROBE_LEARNING_RATES[self.learning_rate_index]
        ):
            raise ValueError("relation-depth candidate learning rate differs from the grid")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "surface": self.surface.as_dict(),
            "learning_rate_hex": self.learning_rate.hex(),
            "learning_rate_index": self.learning_rate_index,
        }


def relation_depth_candidates(num_layers: int) -> tuple[RelationDepthCandidate, ...]:
    candidates = []
    for surface in relation_depth_surfaces(num_layers):
        for index, learning_rate in enumerate(RELATION_DEPTH_PROBE_LEARNING_RATES):
            candidates.append(
                RelationDepthCandidate(
                    candidate_id=f"{surface.name}_lr{index}",
                    surface=surface,
                    learning_rate=learning_rate,
                    learning_rate_index=index,
                )
            )
    return tuple(candidates)


def relation_depth_trainable_parameters(
    readout: SharedRelationReadout,
) -> tuple[nn.Parameter, ...]:
    if not isinstance(readout, SharedRelationReadout):
        raise TypeError("relation-depth probe requires SharedRelationReadout")
    named_parameters = dict(readout.named_parameters())
    if not set(_TRAINABLE_READOUT_PATHS).issubset(named_parameters):
        raise RuntimeError("production relation readout changed its diagnostic parameter surface")
    return tuple(named_parameters[name] for name in _TRAINABLE_READOUT_PATHS)


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_relation_depth_probe_bank(
    *,
    host_width: int,
    num_layers: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.ModuleDict, tuple[RelationDepthCandidate, ...], str]:
    """Build bit-identical, parameter-disjoint external readout candidates."""

    width = _positive_integer(host_width, name="relation-depth host width")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("relation-depth seed must be an integer")
    target_device = torch.device(device)
    if not dtype.is_floating_point:
        raise TypeError("relation-depth probe dtype must be floating point")
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
        prototype = SharedRelationReadout(width).to(device=target_device, dtype=dtype)
    prototype_state = {
        name: value.detach().clone() for name, value in prototype.state_dict().items()
    }
    initialization_sha256 = _state_digest(prototype_state)
    candidates = relation_depth_candidates(num_layers)
    bank = nn.ModuleDict()
    parameter_ids: set[int] = set()
    for candidate in candidates:
        readout = copy.deepcopy(prototype)
        readout.load_state_dict(prototype_state, strict=True)
        for parameter in readout.parameters():
            parameter.requires_grad_(False)
        for parameter in relation_depth_trainable_parameters(readout):
            parameter.requires_grad_(True)
            if id(parameter) in parameter_ids:
                raise RuntimeError("relation-depth candidates share trainable parameters")
            parameter_ids.add(id(parameter))
        if (
            _state_digest({name: value.detach() for name, value in readout.state_dict().items()})
            != initialization_sha256
        ):
            raise RuntimeError("relation-depth candidates have different initialization")
        bank[candidate.candidate_id] = readout
    expected_trainables = len(candidates) * (width * width + width + 1)
    observed_trainables = sum(
        parameter.numel() for parameter in bank.parameters() if parameter.requires_grad
    )
    if observed_trainables != expected_trainables:
        raise RuntimeError("relation-depth probe bank has an unexpected trainable capacity")
    return bank, candidates, initialization_sha256


def relation_depth_probe_subject(
    provenance: dict[str, object],
    *,
    curve_point_count: int,
) -> str:
    if set(provenance) != _DEPTH_PROVENANCE_FIELDS:
        raise ValueError("relation-depth provenance fields differ from schema")
    points = _positive_integer(curve_point_count, name="relation-depth curve points")
    if points < 2:
        raise ValueError("relation-depth probe requires at least two curve points")
    return _canonical_digest(
        {
            "schema": RELATION_DEPTH_PROBE_SCHEMA,
            "provenance": provenance,
            "curve_point_count": points,
            "optimizer_update_count": points - 1,
        }
    )


def relation_depth_recovery_summary(
    *,
    global_curves: Mapping[str, Sequence[float]],
    rank_task_curves: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Compute the preregistered D-normalized final-point recovery decision."""

    if set(global_curves) != set(RELATION_DEPTH_PROBE_CURVE_NAMES):
        raise ValueError("relation-depth recovery curves differ from schema")
    curves = {
        name: _curve(
            global_curves[name],
            name=f"recovery {name}",
            points=RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
        )
        for name in RELATION_DEPTH_PROBE_CURVE_NAMES
    }
    if len(rank_task_curves) != 2:
        raise ValueError("relation-depth recovery requires two rank task curves")
    rank_curves = tuple(
        _curve(
            value,
            name=f"recovery rank {rank} task_soft_iou",
            points=RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
        )
        for rank, value in enumerate(rank_task_curves)
    )
    point_zero = cast(
        Mapping[str, float],
        RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["point_zero"],
    )
    full_host = cast(
        Mapping[str, float],
        RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["structural_full_host_point_40"],
    )
    ratios = {
        "ownership": (point_zero["ownership"] - curves["ownership"][-1])
        / (point_zero["ownership"] - full_host["ownership"]),
        "macro_soft_iou": (curves["macro_soft_iou"][-1] - point_zero["macro_soft_iou"])
        / (full_host["macro_soft_iou"] - point_zero["macro_soft_iou"]),
        "task_soft_iou": (curves["task_soft_iou"][-1] - point_zero["task_soft_iou"])
        / (full_host["task_soft_iou"] - point_zero["task_soft_iou"]),
    }
    rank_references = cast(
        Sequence[Mapping[str, float]],
        RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["rank_task_soft_iou"],
    )
    rank_ratios = [
        (rank_curves[rank][-1] - reference["point_zero"])
        / (reference["structural_full_host_point_40"] - reference["point_zero"])
        for rank, reference in enumerate(rank_references)
    ]
    return {
        "global_recovery_ratios": ratios,
        "rank_task_recovery_ratios": rank_ratios,
        "passes_half_recovery": all(value >= 0.5 for value in (*ratios.values(), *rank_ratios)),
    }


def relation_depth_decisions(
    candidate_reports: Sequence[Mapping[str, object]],
    *,
    num_layers: int,
) -> list[dict[str, object]]:
    """Require adjacent optimizer settings before declaring one depth recoverable."""

    expected_candidates = relation_depth_candidates(num_layers)
    if len(candidate_reports) != len(expected_candidates):
        raise ValueError("relation-depth decision omitted a candidate")
    passed_by_surface: dict[str, set[int]] = {
        surface.name: set() for surface in relation_depth_surfaces(num_layers)
    }
    observed_ids: set[str] = set()
    for raw, expected in zip(candidate_reports, expected_candidates, strict=True):
        if not isinstance(raw, Mapping) or raw.get("candidate") != expected.as_dict():
            raise ValueError("relation-depth candidate order or descriptor changed")
        if expected.candidate_id in observed_ids:
            raise ValueError("relation-depth decision contains a duplicate candidate")
        observed_ids.add(expected.candidate_id)
        recovery = raw.get("recovery")
        if not isinstance(recovery, Mapping) or not isinstance(
            recovery.get("passes_half_recovery"),
            bool,
        ):
            raise ValueError("relation-depth candidate lacks a recovery decision")
        if recovery["passes_half_recovery"]:
            passed_by_surface[expected.surface.name].add(expected.learning_rate_index)

    decisions: list[dict[str, object]] = []
    for surface in relation_depth_surfaces(num_layers):
        passing = sorted(passed_by_surface[surface.name])
        adjacent = [
            [left, left + 1] for left in passing if left + 1 in passed_by_surface[surface.name]
        ]
        decisions.append(
            {
                "surface": surface.as_dict(),
                "passing_learning_rate_indices": passing,
                "adjacent_passing_pairs": adjacent,
                "recoverable": bool(adjacent),
            }
        )
    return decisions


def _validate_rank_seed_pair(
    value: object,
    *,
    name: str,
    require_identical: bool,
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(f"relation-depth {name} must bind two non-negative rank seeds")
    result = (value[0], value[1])
    if require_identical and result[0] != result[1]:
        raise ValueError(f"relation-depth {name} must be rank invariant")
    return result


def validate_relation_probe_policy_parameter_boundary(
    value: object,
    *,
    host_width: int,
) -> RelationGeometryTrainableScope:
    if not isinstance(value, Mapping) or set(value) != {
        "arm",
        "parameter_count",
        "trainable_numel",
        "schema_sha256",
        "parameters",
    }:
        raise ValueError("relation-depth policy parameter boundary fields differ")
    raw_parameters = value["parameters"]
    if not isinstance(raw_parameters, list):
        raise ValueError("relation-depth policy boundary parameters must be a list")
    descriptors: list[tuple[str, tuple[int, ...], str, int]] = []
    for raw in raw_parameters:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "shape",
            "dtype",
            "numel",
        }:
            raise ValueError("relation-depth policy parameter descriptor fields differ")
        shape = raw["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
        ):
            raise ValueError("relation-depth policy parameter shape is invalid")
        descriptors.append(
            (
                raw["name"] if isinstance(raw["name"], str) else "",
                tuple(shape),
                raw["dtype"] if isinstance(raw["dtype"], str) else "",
                _positive_integer(
                    raw["numel"],
                    name="relation-depth policy parameter elements",
                ),
            )
        )
    scope = RelationGeometryTrainableScope(
        arm=value["arm"] if isinstance(value["arm"], str) else "",
        parameter_count=_positive_integer(
            value["parameter_count"],
            name="relation-depth policy parameter count",
        ),
        trainable_numel=_positive_integer(
            value["trainable_numel"],
            name="relation-depth policy trainable elements",
        ),
        schema_sha256=_sha256(
            value["schema_sha256"],
            name="relation-depth policy parameter schema",
        ),
        parameter_descriptors=tuple(descriptors),
    )
    expected_shapes = {
        "picf_native_graph.relation_readout.no_object": (host_width,),
        "picf_native_graph.relation_readout.projection.weight": (
            host_width,
            host_width,
        ),
        "picf_native_graph.relation_readout.temperature_parameter": (1,),
    }
    if (
        scope.arm != "existing_readout_frozen_host"
        or scope.parameter_count != len(expected_shapes)
        or scope.trainable_numel != host_width * host_width + host_width + 1
        or {name: shape for name, shape, _dtype, _numel in scope.parameter_descriptors}
        != expected_shapes
    ):
        raise ValueError("relation-depth host parameter boundary is not the frozen readout")
    return scope


def validate_relation_probe_visual_artifact(
    value: object,
    *,
    candidate_id: str,
    curve_point: int,
    rank: int,
    expected_sample_key: str,
    capacity: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _VISUAL_ARTIFACT_FIELDS:
        raise ValueError("relation-depth visual artifact fields differ from schema")
    relative_value = value["path"]
    relative = PurePosixPath(relative_value) if isinstance(relative_value, str) else None
    if (
        value["schema"] != NATIVE_VISUAL_AUDIT_SCHEMA
        or value["global_step"] != curve_point + 1
        or value["input_weight_global_step"] != curve_point
        or value["weight_boundary"] != "pre_update_forward"
        or value["rank"] != rank
        or value["loss_only_labels_visible_to_model"] is not False
        or value["anchor_surface"]
        not in {
            "task_object_probability.max(row)",
            "ownership_or_support_times_task_relevance.max(row)",
        }
        or relative is None
        or relative.is_absolute()
        or len(relative.parts) < 2
        or relative.parts[0] != candidate_id
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("relation-depth visual artifact provenance is inconsistent")
    _sha256(value["sha256"], name="relation-depth visual artifact")
    _positive_integer(value["bytes"], name="relation-depth visual artifact bytes")
    if (
        _integer(
            value["batch_index"],
            name="relation-depth visual batch index",
        )
        != 0
        or value["sample_key"] != expected_sample_key
        or not isinstance(value["task"], str)
        or not value["task"].strip()
    ):
        raise ValueError("relation-depth visual sample provenance is inconsistent")
    identity_keys = value["identity_keys"]
    row_to_track = value["row_to_track"]
    sequence_row_to_track = value["sequence_row_to_track"]
    binding_start_phase = value["binding_start_phase"]
    source_binding_valid = value["source_binding_valid"]
    existence = value["row_existence"]
    relevance = value["row_task_relevance"]
    soft_iou = value["row_matched_soft_iou"]
    if (
        not isinstance(identity_keys, list)
        or any(not isinstance(item, str) or not item for item in identity_keys)
        or len(set(identity_keys)) != len(identity_keys)
        or not isinstance(row_to_track, list)
        or not isinstance(sequence_row_to_track, list)
        or not isinstance(binding_start_phase, list)
        or not isinstance(source_binding_valid, list)
        or not isinstance(existence, list)
        or not isinstance(relevance, list)
        or not isinstance(soft_iou, list)
        or any(
            len(item) != capacity
            for item in (
                row_to_track,
                sequence_row_to_track,
                binding_start_phase,
                source_binding_valid,
                existence,
                relevance,
                soft_iou,
            )
        )
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < -1
            or item >= len(identity_keys)
            for item in row_to_track
        )
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < -1
            or item >= len(identity_keys)
            for item in sequence_row_to_track
        )
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in binding_start_phase
        )
        or any(not isinstance(item, bool) for item in source_binding_valid)
    ):
        raise ValueError("relation-depth visual row metadata is malformed")
    if (
        value["source_time"] != 0
        or value["source_side"] != "posterior"
        or value["source_phase"] != 1
        or source_binding_valid
        != [
            track >= 0 and phase <= 1
            for track, phase in zip(sequence_row_to_track, binding_start_phase, strict=True)
        ]
        or row_to_track
        != [
            track if valid else -1
            for track, valid in zip(sequence_row_to_track, source_binding_valid, strict=True)
        ]
    ):
        raise ValueError("relation-depth visual source-cut assignment is inconsistent")
    for name, values in (("existence", existence), ("relevance", relevance)):
        for index, item in enumerate(values):
            _finite_float(
                item,
                name=f"relation-depth visual {name}[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
    for index, item in enumerate(soft_iou):
        if item is not None:
            _finite_float(
                item,
                name=f"relation-depth visual soft-IoU[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
    if not isinstance(value["views"], list) or not value["views"]:
        raise ValueError("relation-depth visual artifact has no rendered view")


def validate_relation_probe_provenance(
    value: object,
) -> tuple[dict[str, Any], int, int, list[list[str]]]:
    if not isinstance(value, Mapping) or set(value) != _DEPTH_PROVENANCE_FIELDS:
        raise ValueError("relation-depth provenance fields differ from schema")
    provenance = dict(value)
    for name in ("source_commit", "checkpoint_revision"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(f"relation-depth provenance {name} is absent")
    for name in _DEPTH_PROVENANCE_DIGEST_FIELDS:
        _sha256(provenance[name], name=f"relation-depth provenance {name}")
    _integer(provenance["seed"], name="relation-depth seed")
    fixed_step = _integer(
        provenance["fixed_sample_global_step"],
        name="relation-depth fixed sample step",
    )
    selection = validate_relation_probe_sample_selection(provenance["sample_selection"])
    if selection.selected_global_step != fixed_step:
        raise ValueError("relation-depth selected and executed sample steps differ")
    _validate_rank_seed_pair(
        provenance["forward_seed_by_rank"],
        name="forward seeds",
        require_identical=False,
    )
    _validate_rank_seed_pair(
        provenance["probe_seed_by_rank"],
        name="probe seeds",
        require_identical=True,
    )
    sample_keys = provenance["frame_sample_keys_by_rank"]
    source_digests = provenance["frame_source_digests_by_rank"]
    if (
        not isinstance(sample_keys, list)
        or len(sample_keys) != 2
        or any(
            not isinstance(rank_value, list)
            or len(rank_value) != 2
            or any(not isinstance(item, str) or not item for item in rank_value)
            for rank_value in sample_keys
        )
    ):
        raise ValueError("relation-depth sample keys must bind two frames on both ranks")
    if tuple(sample.sample_key for sample in selection.samples_by_rank) != tuple(
        rank_keys[0] for rank_keys in sample_keys
    ):
        raise ValueError("relation-depth selected samples differ from executed current frames")
    if (
        not isinstance(source_digests, list)
        or len(source_digests) != 2
        or any(
            not isinstance(rank_value, list)
            or len(rank_value) != 2
            or any(
                _sha256(item, name="relation-depth source digest") != item for item in rank_value
            )
            for rank_value in source_digests
        )
    ):
        raise ValueError("relation-depth source digests must bind two frames on both ranks")
    row_bindings = provenance["row_bindings_by_rank"]
    if not isinstance(row_bindings, list) or len(row_bindings) != 2:
        raise ValueError("relation-depth row gauge must bind both ranks")
    for rank, (raw_bindings, sample) in enumerate(
        zip(row_bindings, selection.samples_by_rank, strict=True)
    ):
        if (
            not isinstance(raw_bindings, list)
            or not raw_bindings
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 0
                or item[1] >= selection.capacity
                for item in raw_bindings
            )
        ):
            raise ValueError(f"relation-depth rank {rank} row gauge is malformed")
        identities = [item[0] for item in raw_bindings]
        rows = [item[1] for item in raw_bindings]
        if len(set(identities)) != len(identities) or len(set(rows)) != len(rows):
            raise ValueError("relation-depth row gauge is not one-to-one")
        target_keys = sample.target_identity_keys
        if target_keys is None or not set(target_keys).issubset(identities):
            raise ValueError("relation-depth row gauge omits a selected task target")
    actions = provenance["official_action_by_rank"]
    if not isinstance(actions, list) or len(actions) != 2:
        raise ValueError("relation-depth official action must bind both ranks")
    for rank, action in enumerate(actions):
        _finite_float(
            action,
            name=f"relation-depth rank {rank} official action",
            minimum=0.0,
        )
    _sha256(
        provenance["candidate_initialization_sha256"],
        name="relation-depth candidate initialization",
    )
    expected_objective = {
        "optimized_term": "set/ownership",
        "observed_terms": list(RELATION_DEPTH_PROBE_CURVE_NAMES),
        "window": "fixed_two_frame_detached_host",
        "labels_are_loss_side_only": True,
        "row_gauge": "production_point_zero_then_frozen",
        "official_policy_loss": "observed_not_optimized",
    }
    if provenance["objective"] != expected_objective:
        raise ValueError("relation-depth objective contract changed")
    return provenance, selection.capacity, fixed_step, sample_keys


def validate_relation_depth_probe_report(value: object) -> dict[str, Any]:
    """Validate C, recompute rank means, recoveries, decisions and subject."""

    if not isinstance(value, Mapping) or set(value) != _DEPTH_REPORT_FIELDS:
        raise ValueError("relation-depth report fields differ from schema")
    if (
        value["schema"] != RELATION_DEPTH_PROBE_SCHEMA
        or value["status"] != "PASS"
        or value["arm"] != RELATION_DEPTH_PROBE_ARM
    ):
        raise ValueError("relation-depth probe did not complete")
    points = _integer(
        value["curve_point_count"],
        name="relation-depth curve points",
        minimum=2,
    )
    updates = _integer(
        value["optimizer_update_count"],
        name="relation-depth optimizer updates",
        minimum=1,
    )
    if (
        points != RELATION_DEPTH_PROBE_CURVE_POINT_COUNT
        or updates != RELATION_DEPTH_PROBE_UPDATE_COUNT
        or updates != points - 1
    ):
        raise ValueError("relation-depth curve length differs from preregistration")
    provenance, capacity, _fixed_step, sample_keys = validate_relation_probe_provenance(
        value["provenance"]
    )
    if _sha256(value["subject_sha256"], name="relation-depth subject") != (
        relation_depth_probe_subject(provenance, curve_point_count=points)
    ):
        raise ValueError("relation-depth subject differs from its provenance")
    host_width = _integer(
        provenance["host_width"],
        name="relation-depth host width",
        minimum=1,
    )
    num_layers = _integer(
        provenance["host_layer_count"],
        name="relation-depth host layers",
        minimum=4,
    )
    expected_surfaces = [surface.as_dict() for surface in relation_depth_surfaces(num_layers)]
    if provenance.get("surfaces") != expected_surfaces:
        raise ValueError("relation-depth provenance surfaces changed")
    capture = provenance.get("capture")
    if capture != {
        "intermediate_hook": "next_layer_compute_kqv_input_after_block_and_deepstack",
        "final_hook": "post_final_norm",
        "forward_count": 2,
        "feature_dtype": "float32",
        "policy_grad_enabled": False,
        "prediction_queries": "absent",
    }:
        raise ValueError("relation-depth capture contract changed")
    optimizer = provenance.get("optimizer")
    if optimizer != {
        "algorithm": "torch.optim.AdamW",
        "learning_rate_hex_grid": [value.hex() for value in RELATION_DEPTH_PROBE_LEARNING_RATES],
        "weight_decay_hex": RELATION_DEPTH_PROBE_WEIGHT_DECAY.hex(),
        "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "warmup_updates": 0,
        "update_count": RELATION_DEPTH_PROBE_UPDATE_COUNT,
        "distributed_gradient": "rank_sum_div_world_size",
    }:
        raise ValueError("relation-depth optimizer contract changed")
    if provenance.get("global_references") != {
        "point_zero": dict(
            cast(
                Mapping[str, float],
                RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["point_zero"],
            )
        ),
        "structural_full_host_point_40": dict(
            cast(
                Mapping[str, float],
                RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["structural_full_host_point_40"],
            )
        ),
        "rank_task_soft_iou": [
            dict(value)
            for value in cast(
                Sequence[Mapping[str, float]],
                RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["rank_task_soft_iou"],
            )
        ],
    }:
        raise ValueError("relation-depth A/D references changed")
    validate_relation_probe_policy_parameter_boundary(
        value["policy_parameter_boundary"],
        host_width=host_width,
    )
    candidate_initialization_sha256 = _sha256(
        value["candidate_initialization_sha256"],
        name="relation-depth candidate initialization",
    )
    if provenance.get("candidate_initialization_sha256") != (candidate_initialization_sha256):
        raise ValueError("relation-depth provenance omitted candidate initialization")

    candidate_reports = value["candidates"]
    expected_candidates = relation_depth_candidates(num_layers)
    if not isinstance(candidate_reports, list) or len(candidate_reports) != len(
        expected_candidates
    ):
        raise ValueError("relation-depth report lacks the full candidate grid")
    normalized_candidates: list[dict[str, object]] = []
    for raw, expected in zip(candidate_reports, expected_candidates, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _CANDIDATE_REPORT_FIELDS:
            raise ValueError("relation-depth candidate report fields differ")
        if raw["candidate"] != expected.as_dict():
            raise ValueError("relation-depth candidate descriptor or order changed")
        if (
            _integer(
                raw["trainable_numel"],
                name="relation-depth trainable elements",
                minimum=1,
            )
            != host_width * host_width + host_width + 1
        ):
            raise ValueError("relation-depth candidate capacity changed")
        global_raw = raw["global_curves"]
        if not isinstance(global_raw, Mapping) or set(global_raw) != set(
            RELATION_DEPTH_PROBE_CURVE_NAMES
        ):
            raise ValueError("relation-depth global curves differ")
        global_curves = {
            name: _curve(
                global_raw[name],
                name=f"{expected.candidate_id} global {name}",
                points=points,
            )
            for name in RELATION_DEPTH_PROBE_CURVE_NAMES
        }
        rank_reports = raw["rank_reports"]
        if not isinstance(rank_reports, list) or len(rank_reports) != 2:
            raise ValueError("relation-depth candidate requires two rank reports")
        indexed_rank_curves: list[tuple[int, dict[str, tuple[float, ...]]]] = []
        for rank_raw in rank_reports:
            if not isinstance(rank_raw, Mapping) or set(rank_raw) != _RANK_REPORT_FIELDS:
                raise ValueError("relation-depth rank report fields differ")
            rank = _integer(rank_raw["rank"], name="relation-depth rank")
            if rank not in (0, 1):
                raise ValueError("relation-depth rank must be zero or one")
            curves_raw = rank_raw["curves"]
            if not isinstance(curves_raw, Mapping) or set(curves_raw) != set(
                RELATION_DEPTH_PROBE_CURVE_NAMES
            ):
                raise ValueError("relation-depth rank curves differ")
            indexed_rank_curves.append(
                (
                    rank,
                    {
                        name: _curve(
                            curves_raw[name],
                            name=f"{expected.candidate_id} rank {rank} {name}",
                            points=points,
                        )
                        for name in RELATION_DEPTH_PROBE_CURVE_NAMES
                    },
                )
            )
            _finite_float(
                rank_raw["gradient_norm_at_first_update"],
                name="relation-depth first gradient norm",
                minimum=0.0,
            )
            times = _curve(
                rank_raw["evaluation_times_s"],
                name=f"{expected.candidate_id} rank {rank} evaluation_times_s",
                points=points,
            )
            if any(value <= 0 for value in times):
                raise ValueError("relation-depth evaluation time must be positive")
            visual = rank_raw["visual_artifacts_by_point"]
            if not isinstance(visual, list) or [
                item.get("curve_point") if isinstance(item, Mapping) else None for item in visual
            ] != list(RELATION_DEPTH_PROBE_VISUAL_POINTS):
                raise ValueError("relation-depth visuals differ from the frozen points")
            for item in visual:
                if not isinstance(item, Mapping) or set(item) != {
                    "curve_point",
                    "artifacts",
                }:
                    raise ValueError("relation-depth visual record fields differ")
                artifacts = item["artifacts"]
                if not isinstance(artifacts, list) or len(artifacts) != 1:
                    raise ValueError("relation-depth visual point requires one local sample")
                validate_relation_probe_visual_artifact(
                    artifacts[0],
                    candidate_id=expected.candidate_id,
                    curve_point=item["curve_point"],
                    rank=rank,
                    expected_sample_key=sample_keys[rank][0],
                    capacity=capacity,
                )
        if sorted(rank for rank, _curves in indexed_rank_curves) != [0, 1]:
            raise ValueError("relation-depth rank reports are duplicated")
        rank_curves = [
            curves for _rank, curves in sorted(indexed_rank_curves, key=lambda item: item[0])
        ]
        for name in RELATION_DEPTH_PROBE_CURVE_NAMES:
            recomputed = tuple(
                sum(rank_curve[name][point] for rank_curve in rank_curves) / 2
                for point in range(points)
            )
            if any(
                not math.isclose(observed, expected_mean, rel_tol=0.0, abs_tol=1e-12)
                for observed, expected_mean in zip(
                    global_curves[name],
                    recomputed,
                    strict=True,
                )
            ):
                raise ValueError("relation-depth global curve is not the rank mean")
        recovery = relation_depth_recovery_summary(
            global_curves=global_curves,
            rank_task_curves=[rank_curve["task_soft_iou"] for rank_curve in rank_curves],
        )
        if raw["recovery"] != recovery:
            raise ValueError("relation-depth recovery decision was not recomputed")
        normalized_candidates.append(dict(raw))
    decisions = relation_depth_decisions(normalized_candidates, num_layers=num_layers)
    if value["depth_decisions"] != decisions:
        raise ValueError("relation-depth depth decision was not recomputed")
    _integer(
        value["maximum_peak_reserved_bytes"],
        name="relation-depth peak reserved bytes",
        minimum=1,
    )
    _finite_float(
        value["total_time_s"],
        name="relation-depth total time",
        minimum=0.0,
    )
    return dict(value)

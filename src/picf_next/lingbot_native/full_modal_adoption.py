"""Diagnostic-only contracts for full-modal action adoption.

These helpers never participate in a deployed forward.  They construct
counterfactual copies of typed modality streams and resolve exact production
parameter paths so a four-rank probe can establish that the released action
loss consumes each optional source through the shared LingBot host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from picf_next.artifact_io import directory_tree_sha256 as _directory_tree_sha256
from picf_next.lingbot_native.modalities import NativeModalityBatch, NativeModalityStream

DENSE_MODALITIES = ("anytouch", "sonata", "vjepa")
DENSE_PRESENCE_SUBSETS = (
    (),
    ("anytouch",),
    ("sonata",),
    ("vjepa",),
    ("anytouch", "sonata"),
    ("anytouch", "vjepa"),
    ("sonata", "vjepa"),
    DENSE_MODALITIES,
)
DENSE_PRESENCE_CODES = ("none", "A", "S", "V", "AS", "AV", "SV", "ASV")
MODALITY_INTERVENTIONS = (
    "value_zero",
    "metadata_zero",
    "value_permutation",
    "joint_permutation",
)
ACTION_ADOPTION_CORE_SCHEMA = "picf-next.adr150-full-modal-action-adoption-core.v1"
ACTION_ADOPTION_PRESENCE_SCHEMA = "picf-next.adr150-full-modal-action-adoption-presence.v1"
ACTION_ADOPTION_INTERVENTIONS_SCHEMA = (
    "picf-next.adr150-full-modal-action-adoption-interventions.v1"
)
ACTION_DCP_PHASE_SCHEMA = "picf-next.adr150-action-dcp-phase.v1"
FULL_MODAL_ACTION_ADOPTION_SCHEMA = "picf-next.adr150-full-modal-action-adoption.v1"
ACTION_ADOPTION_NONZERO_GRADIENT_MIN_NORM = 1e-12
ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT = 1e-6
ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT = 1e-5
ACTION_DCP_BOUNDARY_FIELDS = (
    "model_sha256",
    "optimizer_sha256",
    "lane_sha256",
    "rng_sha256",
)
ACTION_DCP_CHECKPOINT_BOUNDARY_FIELDS = {
    "model_sha256": "model_local_state_sha256",
    "optimizer_sha256": "optimizer_local_state_sha256",
    "lane_sha256": "lane_snapshot_sha256",
    "rng_sha256": "rank_rng_state_sha256",
}


@dataclass(frozen=True, slots=True)
class ActionAdoptionParameterGroup:
    """One exact, disjoint parameter group measured by the action-only probe."""

    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.parameter_names or not self.parameters:
            raise ValueError("action-adoption parameter groups must be nonempty")
        if len(self.parameter_names) != len(self.parameters):
            raise ValueError("action-adoption parameter names and tensors differ")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("action-adoption parameter names must be unique")


@dataclass(frozen=True, slots=True)
class ModalityInterventionBatch:
    """One counterfactual batch plus its exact within-sample token permutation."""

    batch: NativeModalityBatch
    token_permutations: tuple[tuple[int, ...], ...]
    valid_before: tuple[tuple[bool, ...], ...]
    valid_after: tuple[tuple[bool, ...], ...]
    changed_elements: int

    def __post_init__(self) -> None:
        batch_size = self.batch.batch_size
        if not (
            len(self.token_permutations)
            == len(self.valid_before)
            == len(self.valid_after)
            == batch_size
        ):
            raise ValueError("modality intervention evidence differs from the batch size")
        if isinstance(self.changed_elements, bool) or self.changed_elements < 0:
            raise ValueError("modality intervention changed-elements count is invalid")


def dense_presence_name(present: tuple[str, ...]) -> str:
    """Return the canonical report key for one optional-modality subset."""

    _validate_presence(present)
    return "none" if not present else "+".join(present)


def dense_presence_code(present: tuple[str, ...]) -> str:
    """Return the compact canonical code used by the frozen LBOT schema."""

    _validate_presence(present)
    return DENSE_PRESENCE_CODES[DENSE_PRESENCE_SUBSETS.index(present)]


def with_dense_presence(
    batch: NativeModalityBatch,
    present: tuple[str, ...],
) -> NativeModalityBatch:
    """Remove exactly the absent optional streams while retaining other inputs."""

    if not isinstance(batch, NativeModalityBatch):
        raise TypeError("dense-presence intervention requires one typed modality batch")
    _validate_presence(present)
    names = {stream.name for stream in batch.streams}
    if not set(DENSE_MODALITIES) <= names:
        raise ValueError("dense-presence intervention requires all three declared streams")
    omitted = tuple(name for name in DENSE_MODALITIES if name not in present)
    return batch if not omitted else batch.omit(omitted)


def intervene_modality(
    batch: NativeModalityBatch,
    *,
    modality: str,
    intervention: str,
    require_change: bool = True,
) -> ModalityInterventionBatch:
    """Create one deterministic value/metadata counterfactual.

    Permutations rotate only valid token rows within each sample. Therefore the
    joint arm is a pure set permutation, while the value-only arm breaks the
    value-to-metadata correspondence without moving evidence between samples.
    """

    if not isinstance(batch, NativeModalityBatch):
        raise TypeError("modality intervention requires one typed modality batch")
    if modality not in DENSE_MODALITIES:
        raise ValueError("modality intervention requires one declared dense modality")
    if intervention not in MODALITY_INTERVENTIONS:
        raise ValueError("unknown full-modal intervention")
    if not isinstance(require_change, bool):
        raise TypeError("modality intervention require_change must be boolean")
    try:
        source = next(stream for stream in batch.streams if stream.name == modality)
    except StopIteration as error:
        raise ValueError("intervened modality is absent from the source batch") from error
    if source.metadata is None:
        raise ValueError("full-modal interventions require typed source metadata")
    if source.token_count == 0:
        if require_change:
            raise ValueError("full-modal intervention did not change any valid source value")
        empty_rows = tuple(() for _ in range(source.batch_size))
        return ModalityInterventionBatch(
            batch=batch,
            token_permutations=empty_rows,
            valid_before=empty_rows,
            valid_after=empty_rows,
            changed_elements=0,
        )
    tokens = source.tokens.clone()
    metadata = source.metadata.clone()
    canonical_token_ids = (
        None if source.canonical_token_ids is None else source.canonical_token_ids.clone()
    )
    permutations: list[tuple[int, ...]] = []
    changed = 0
    for batch_index in range(source.batch_size):
        permutation = torch.arange(source.token_count, device=source.valid.device)
        indices = torch.nonzero(source.valid[batch_index], as_tuple=False).flatten()
        if indices.numel() >= 2:
            permutation[indices] = torch.roll(indices, shifts=1)
        permutations.append(tuple(int(value) for value in permutation.tolist()))
    if intervention == "value_zero":
        valid = source.valid.unsqueeze(-1).expand_as(tokens)
        changed = int(((tokens != 0) & valid).sum().item())
        tokens.masked_fill_(valid, 0)
    elif intervention == "metadata_zero":
        valid = source.valid.unsqueeze(-1).expand_as(metadata)
        changed = int(((metadata != 0) & valid).sum().item())
        metadata.masked_fill_(valid, 0)
    else:
        for batch_index, permutation_values in enumerate(permutations):
            permutation = torch.tensor(
                permutation_values,
                dtype=torch.long,
                device=source.valid.device,
            )
            original_tokens = source.tokens[batch_index]
            tokens[batch_index] = source.tokens[batch_index, permutation]
            changed += int((tokens[batch_index] != original_tokens).sum().item())
            if intervention == "joint_permutation":
                original_metadata = source.metadata[batch_index]
                metadata[batch_index] = source.metadata[batch_index, permutation]
                changed += int((metadata[batch_index] != original_metadata).sum().item())
                if canonical_token_ids is not None:
                    canonical_token_ids[batch_index] = source.canonical_token_ids[
                        batch_index, permutation
                    ]
    if require_change and changed <= 0:
        raise ValueError("full-modal intervention did not change any valid source value")

    replacement = NativeModalityStream(
        name=source.name,
        tokens=tokens,
        valid=source.valid.clone(),
        metadata=metadata,
        canonical_token_ids=canonical_token_ids,
    )
    streams = tuple(replacement if stream.name == modality else stream for stream in batch.streams)
    valid_before = tuple(tuple(bool(value) for value in row) for row in source.valid.tolist())
    valid_after = tuple(tuple(bool(value) for value in row) for row in replacement.valid.tolist())
    return ModalityInterventionBatch(
        batch=NativeModalityBatch(streams),
        token_permutations=tuple(permutations),
        valid_before=valid_before,
        valid_after=valid_after,
        changed_elements=changed,
    )


def resolve_action_adoption_parameter_groups(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    host_layers: tuple[int, ...] = (0, 18, 35),
) -> tuple[ActionAdoptionParameterGroup, ...]:
    """Resolve the exact shared-host/action paths used by the formal probe."""

    values = tuple(named_parameters)
    names = tuple(name for name, _parameter in values)
    if not values or len(set(names)) != len(names):
        raise ValueError("action-adoption parameter inventory must be nonempty and unique")
    if (
        not isinstance(host_layers, tuple)
        or not host_layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in host_layers
        )
        or tuple(sorted(set(host_layers))) != host_layers
    ):
        raise ValueError("action-adoption host layers must be sorted unique nonnegative integers")

    groups: list[ActionAdoptionParameterGroup] = []

    def exact_suffix(group_name: str, suffix: str) -> None:
        matched = tuple((name, parameter) for name, parameter in values if name.endswith(suffix))
        if len(matched) != 1:
            raise ValueError(
                f"action-adoption group {group_name!r} resolved {len(matched)} parameters"
            )
        groups.append(
            ActionAdoptionParameterGroup(
                name=group_name,
                parameter_names=(matched[0][0],),
                parameters=(matched[0][1],),
            )
        )

    for modality in DENSE_MODALITIES:
        exact_suffix(
            f"{modality}_value_adapter",
            f"picf_native_graph.modality_projections.{modality}.weight",
        )
        exact_suffix(
            f"{modality}_metadata_adapter",
            f"picf_native_graph.modality_metadata_projections.{modality}.weight",
        )
    for layer in host_layers:
        exact_suffix(
            f"host_layer_{layer}",
            f"qwenvl.model.language_model.layers.{layer}.input_layernorm.weight",
        )
    action_expert = tuple(
        (name, parameter) for name, parameter in values if ".qwen_expert." in name
    )
    if not action_expert:
        raise ValueError("action-adoption probe resolved no official Qwen action expert")
    groups.append(
        ActionAdoptionParameterGroup(
            name="action_expert",
            parameter_names=tuple(name for name, _parameter in action_expert),
            parameters=tuple(parameter for _name, parameter in action_expert),
        )
    )
    exact_suffix("action_output", "action_out_proj.weight")

    assigned = tuple(name for group in groups for name in group.parameter_names)
    if len(assigned) != len(set(assigned)):
        raise ValueError("action-adoption parameter groups overlap")
    return tuple(groups)


def action_adoption_metric_fragments(
    groups: tuple[ActionAdoptionParameterGroup, ...],
) -> tuple[tuple[str, str], ...]:
    """Return exact-name fragments only for singleton probe groups."""

    if not isinstance(groups, tuple) or not groups:
        raise ValueError("action-adoption metric groups must be a nonempty tuple")
    fragments = []
    for group in groups:
        if group.name == "action_expert":
            fragments.append((group.name, ".qwen_expert."))
        elif len(group.parameter_names) == 1:
            fragments.append((group.name, group.parameter_names[0]))
        else:
            raise ValueError("only the official action expert may aggregate parameters")
    return tuple(fragments)


def _validate_presence(present: tuple[str, ...]) -> None:
    if (
        not isinstance(present, tuple)
        or any(name not in DENSE_MODALITIES for name in present)
        or tuple(sorted(set(present))) != present
        or present not in DENSE_PRESENCE_SUBSETS
    ):
        raise ValueError("dense modality presence must be one canonical subset")


def parameter_group_names(
    groups: tuple[ActionAdoptionParameterGroup, ...],
) -> Mapping[str, tuple[str, ...]]:
    """Serialize the resolved parameter inventory without exposing tensors."""

    if not groups or len({group.name for group in groups}) != len(groups):
        raise ValueError("action-adoption parameter groups must be unique and nonempty")
    return {group.name: group.parameter_names for group in groups}


def distributed_action_adoption_gradients(
    groups: tuple[ActionAdoptionParameterGroup, ...],
    *,
    device: Any,
    dist: Any,
) -> Mapping[str, Mapping[str, float | int | None]]:
    """Measure exact action-only gradients across the FSDP data-parallel group."""

    if not groups or len({group.name for group in groups}) != len(groups):
        raise ValueError("distributed gradient groups must be unique and nonempty")
    packed: list[torch.Tensor] = []
    local_finite = torch.ones((), dtype=torch.int32, device=device)
    for group in groups:
        square = torch.zeros((), dtype=torch.float64, device=device)
        elements = 0
        for parameter in group.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            local = (
                gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
            )
            local_finite.mul_(torch.isfinite(local).all().to(device=device, dtype=torch.int32))
            square.add_(
                local.detach().float().square().sum().to(device=device, dtype=torch.float64)
            )
            elements += int(local.numel())
        packed.extend(
            (
                square,
                torch.tensor(float(elements), dtype=torch.float64, device=device),
            )
        )
    reduced = torch.stack(packed)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
    if int(local_finite.item()) != 1:
        raise FloatingPointError("action-only adoption gradients contain NaN or infinity")
    result: dict[str, Mapping[str, float | int | None]] = {}
    for index, group in enumerate(groups):
        square = float(reduced[2 * index].item())
        elements = int(reduced[2 * index + 1].item())
        result[group.name] = {
            "norm": None if elements == 0 else square**0.5,
            "elements": elements,
        }
    return result


@contextmanager
def capture_action_projection_output(model: Any) -> Iterator[list[torch.Tensor]]:
    """Capture the exact released action projection through an eager diagnostic call.

    PyTorch's default compile contract does not guard module hook dictionaries, so a
    hook installed after the first compiled call is intentionally invisible to the
    cached graph. Force-eager applies only while this sparse diagnostic is active;
    training remains on the released compiled path and the underlying module,
    parameters, inputs, and numerical forward are unchanged.
    """

    matches = tuple(
        (name, module) for name, module in model.named_modules() if name.endswith("action_out_proj")
    )
    if len(matches) != 1:
        raise ValueError(f"action projection capture resolved {len(matches)} modules")
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("released action projection emitted a non-tensor")
        local = output.to_local() if callable(getattr(output, "to_local", None)) else output
        if not isinstance(local, torch.Tensor):
            raise TypeError("released action projection emitted a non-tensor local shard")
        if not local.is_floating_point():
            raise TypeError("released action projection emitted a non-floating local shard")
        captured.append(local.detach().to(device="cpu", dtype=torch.float32).contiguous().clone())

    handle = matches[0][1].register_forward_hook(capture)
    try:
        with torch.compiler.set_stance("force_eager"):
            yield captured
    finally:
        handle.remove()


def single_captured_action_output(captured: list[torch.Tensor]) -> torch.Tensor:
    """Require exactly one action suffix invocation in a diagnostic arm."""

    if len(captured) != 1:
        raise RuntimeError(f"action diagnostic captured {len(captured)} projection outputs")
    output = captured[0]
    if not output.is_floating_point() or not torch.isfinite(output).all():
        raise RuntimeError("captured action projection contains NaN or infinity")
    return output


def action_projection_drift_report(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    """Compare captured action projections after a lossless CPU/FP32 analysis copy."""

    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise TypeError("action projection drift requires tensors")
    if reference.shape != candidate.shape:
        raise ValueError("action projection drift tensors have different shapes")
    if not reference.is_floating_point() or not candidate.is_floating_point():
        raise TypeError("action projection drift requires floating tensors")
    reference_fp32 = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    candidate_fp32 = candidate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(reference_fp32).all() or not torch.isfinite(candidate_fp32).all():
        raise FloatingPointError("action projection drift contains NaN or infinity")
    difference = candidate_fp32 - reference_fp32
    absolute = difference.abs()
    element_count = difference.numel()
    if element_count == 0:
        raise ValueError("action projection drift tensors must be nonempty")
    nonzero_count = int(torch.count_nonzero(difference).item())
    return {
        "shape": list(reference_fp32.shape),
        "element_count": element_count,
        "nonzero_count": nonzero_count,
        "nonzero_fraction": nonzero_count / element_count,
        "max_abs": float(absolute.max().item()),
        "rms": float(difference.square().mean().sqrt().item()),
        "reference_sha256": captured_action_outputs_sha256((reference_fp32,)),
        "candidate_sha256": captured_action_outputs_sha256((candidate_fp32,)),
    }


def distributed_maximum_action_drift(
    left: Any,
    right: Any,
    *,
    dist: Any,
) -> float:
    """Measure one global maximum through the runner's proven object collective."""

    left_local = left.to_local() if callable(getattr(left, "to_local", None)) else left
    right_local = right.to_local() if callable(getattr(right, "to_local", None)) else right
    if not isinstance(left_local, torch.Tensor) or not isinstance(right_local, torch.Tensor):
        raise TypeError("action drift requires local tensors")
    if left_local.shape != right_local.shape:
        raise ValueError("action drift tensors have different local shapes")
    local_value = float((left_local.float() - right_local.float()).abs().max().item())
    if not math.isfinite(local_value) or local_value < 0:
        raise FloatingPointError("distributed action drift is invalid")
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_value)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in gathered
    ):
        raise FloatingPointError("gathered action drift is invalid")
    return max(float(value) for value in gathered)


def captured_action_outputs_sha256(captured: Sequence[torch.Tensor]) -> str:
    """Digest every released action projection call in exact invocation order."""

    if not isinstance(captured, Sequence) or not captured:
        raise ValueError("action continuation must capture at least one projection output")
    digest = hashlib.sha256()
    digest.update(b"picf-next.adr150-action-projection-sequence/v1\0")
    for index, output in enumerate(captured):
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            raise TypeError("captured action continuation contains a non-floating tensor")
        local = output.to_local() if callable(getattr(output, "to_local", None)) else output
        if not torch.isfinite(local).all():
            raise FloatingPointError("captured action continuation contains NaN or infinity")
        contiguous = local.detach().contiguous()
        header = {
            "dtype": str(contiguous.dtype),
            "index": index,
            "shape": list(contiguous.shape),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(contiguous.view(torch.uint8).cpu().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def aggregate_rank_state_digests(
    rank_states: Sequence[Mapping[str, Any]],
    *,
    fields: tuple[str, ...] = ACTION_DCP_BOUNDARY_FIELDS,
) -> dict[str, str]:
    """Normalize and address ordered native checkpoint shards for DCP evidence."""

    if not isinstance(rank_states, Sequence) or not rank_states:
        raise ValueError("distributed state evidence must contain at least one rank")
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("distributed state digest fields must be unique and nonempty")
    if any(field not in ACTION_DCP_CHECKPOINT_BOUNDARY_FIELDS for field in fields):
        raise ValueError("distributed state digest fields are not DCP report fields")
    checkpoint_fields = tuple(ACTION_DCP_CHECKPOINT_BOUNDARY_FIELDS[field] for field in fields)
    ordered: list[Mapping[str, Any]] = []
    for expected_rank, item in enumerate(rank_states):
        if not isinstance(item, Mapping) or set(item) != {"rank", "boundary"}:
            raise ValueError("distributed state evidence fields differ")
        if item["rank"] != expected_rank:
            raise ValueError("distributed state ranks must be ordered and contiguous")
        boundary = item["boundary"]
        if not isinstance(boundary, Mapping) or set(boundary) != set(checkpoint_fields):
            raise ValueError("distributed state boundary fields differ")
        for checkpoint_field in checkpoint_fields:
            _require_sha256(
                boundary[checkpoint_field],
                f"rank {expected_rank} {checkpoint_field}",
            )
        ordered.append(boundary)
    return {
        field: _canonical_sha256(
            {
                "schema": "picf-next.adr150-distributed-state-field/v1",
                "field": field,
                "rank_sha256": [
                    boundary[ACTION_DCP_CHECKPOINT_BOUNDARY_FIELDS[field]] for boundary in ordered
                ],
            }
        )
        for field in fields
    }


def aggregate_rank_action_outputs(rank_outputs: Sequence[Mapping[str, Any]]) -> str:
    """Content-address ordered rank-local action projection sequences."""

    if not isinstance(rank_outputs, Sequence) or not rank_outputs:
        raise ValueError("distributed action evidence must contain at least one rank")
    values: list[str] = []
    for expected_rank, item in enumerate(rank_outputs):
        if not isinstance(item, Mapping) or set(item) != {"rank", "action_output_sha256"}:
            raise ValueError("distributed action evidence fields differ")
        if item["rank"] != expected_rank:
            raise ValueError("distributed action ranks must be ordered and contiguous")
        values.append(
            _require_sha256(
                item["action_output_sha256"],
                f"rank {expected_rank} action output",
            )
        )
    return _canonical_sha256(
        {
            "schema": "picf-next.adr150-distributed-action-output/v1",
            "rank_sha256": values,
        }
    )


def directory_tree_sha256(root: str | Path) -> str:
    """Digest one direct, symlink-free checkpoint tree by relative path and bytes."""

    return _directory_tree_sha256(
        root,
        schema="picf-next.adr150-checkpoint-tree/v1",
    )


def process_set_sha256(
    *,
    phase: str,
    rank_processes: Sequence[Mapping[str, Any]],
) -> str:
    """Bind acceptance evidence to the actual OS processes that produced it."""

    if phase not in {"uninterrupted", "restored"}:
        raise ValueError("DCP process phase is unknown")
    if not isinstance(rank_processes, Sequence) or not rank_processes:
        raise ValueError("DCP process identity requires at least one rank")
    canonical = []
    for expected_rank, item in enumerate(rank_processes):
        if not isinstance(item, Mapping) or set(item) != {"rank", "pid", "start_ticks"}:
            raise ValueError("DCP process identity fields differ")
        rank = item["rank"]
        pid = item["pid"]
        start_ticks = item["start_ticks"]
        if rank != expected_rank:
            raise ValueError("DCP process ranks must be ordered and contiguous")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (pid, start_ticks)
        ):
            raise ValueError("DCP process identity values must be positive integers")
        canonical.append({"rank": rank, "pid": pid, "start_ticks": start_ticks})
    return _canonical_sha256(
        {
            "schema": "picf-next.adr150-process-set/v1",
            "boot_id": _linux_boot_id(),
            "rank_processes": canonical,
        }
    )


def make_action_dcp_phase_report(
    *,
    phase: str,
    process_sha256: str,
    checkpoint_artifact_sha256: str,
    boundary: Mapping[str, Any],
    next_step: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one independently content-addressed side of the cold-restore experiment."""

    if phase not in {"uninterrupted", "restored"}:
        raise ValueError("DCP report phase is unknown")
    normalized_boundary = _boundary_mapping(boundary)
    normalized_next = _continuation_mapping(next_step)
    if normalized_next["global_step"] != 2:
        raise ValueError("DCP report must observe exact optimizer step 2")
    report = {
        "schema": ACTION_DCP_PHASE_SCHEMA,
        "status": "PASS",
        "phase": phase,
        "checkpoint_global_step": 1,
        "optimizer_step": 1,
        "process_sha256": _require_sha256(process_sha256, "DCP process"),
        "checkpoint_artifact_sha256": _require_sha256(
            checkpoint_artifact_sha256,
            "DCP checkpoint artifact",
        ),
        "boundary": normalized_boundary,
        "next_step": normalized_next,
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return report


def make_action_adoption_presence_report(
    *,
    probe_optimizer_step: int,
    nonzero_gradient_min_norm: float,
    presence_subsets: Sequence[Mapping[str, Any]],
    active_anytouch_sample_keys: Sequence[str],
    parameter_groups: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Create the independently replayable gradient-presence receipt."""

    report = {
        "schema": ACTION_ADOPTION_PRESENCE_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "probe_optimizer_step": probe_optimizer_step,
        "nonzero_gradient_min_norm": nonzero_gradient_min_norm,
        "presence_subsets": list(presence_subsets),
        "active_anytouch_sample_keys": list(active_anytouch_sample_keys),
        "parameter_groups": {
            name: list(parameter_names) for name, parameter_names in parameter_groups.items()
        },
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return dict(_validated_action_adoption_presence(report))


def make_action_adoption_interventions_report(
    *,
    probe_optimizer_step: int,
    modality_interventions: Sequence[Mapping[str, Any]],
    active_anytouch_sample_keys: Sequence[str],
) -> dict[str, Any]:
    """Create the independently replayable action-intervention receipt."""

    report = {
        "schema": ACTION_ADOPTION_INTERVENTIONS_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "probe_optimizer_step": probe_optimizer_step,
        "modality_interventions": list(modality_interventions),
        "active_anytouch_sample_keys": list(active_anytouch_sample_keys),
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return dict(_validated_action_adoption_interventions(report))


def compose_action_adoption_core(
    *,
    presence: Mapping[str, Any],
    interventions: Mapping[str, Any],
) -> dict[str, Any]:
    """Join fresh-process gradient and intervention evidence into the stable core ABI."""

    presence_payload = _validated_action_adoption_presence(presence)
    intervention_payload = _validated_action_adoption_interventions(interventions)
    if presence_payload["probe_optimizer_step"] != intervention_payload["probe_optimizer_step"]:
        raise ValueError("action-adoption phases used different optimizer steps")
    presence_keys = _action_adoption_sample_keys(
        presence_payload["presence_subsets"],
        name="presence subsets",
    )
    intervention_keys = _action_adoption_sample_keys(
        intervention_payload["modality_interventions"],
        name="modality interventions",
    )
    if presence_keys != intervention_keys:
        raise ValueError("action-adoption phases used different global samples")
    if (
        presence_payload["active_anytouch_sample_keys"]
        != intervention_payload["active_anytouch_sample_keys"]
    ):
        raise ValueError("action-adoption phases used different active AnyTouch samples")
    core = {
        "schema": ACTION_ADOPTION_CORE_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "probe_optimizer_step": presence_payload["probe_optimizer_step"],
        "nonzero_gradient_min_norm": presence_payload["nonzero_gradient_min_norm"],
        "presence_subsets": presence_payload["presence_subsets"],
        "modality_interventions": intervention_payload["modality_interventions"],
        "active_anytouch_sample_keys": presence_payload["active_anytouch_sample_keys"],
        "parameter_groups": presence_payload["parameter_groups"],
    }
    core["artifact_sha256"] = _canonical_sha256(core)
    return dict(_validated_core(core))


def compose_full_modal_action_adoption(
    *,
    core: Mapping[str, Any],
    uninterrupted: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> dict[str, Any]:
    """Join independent action-adoption and DCP evidence without weakening either gate."""

    core_payload = _validated_core(core)
    uninterrupted_payload = _validated_dcp_phase(uninterrupted, expected_phase="uninterrupted")
    restored_payload = _validated_dcp_phase(restored, expected_phase="restored")
    if uninterrupted_payload["process_sha256"] == restored_payload["process_sha256"]:
        raise ValueError("DCP restore evidence did not come from a distinct process set")
    if (
        uninterrupted_payload["checkpoint_artifact_sha256"]
        != restored_payload["checkpoint_artifact_sha256"]
    ):
        raise ValueError("DCP checkpoint artifact changed before cold restoration")
    if uninterrupted_payload["boundary"] != restored_payload["boundary"]:
        raise ValueError("DCP cold-restored distributed boundary differs")
    if uninterrupted_payload["next_step"] != restored_payload["next_step"]:
        raise ValueError("DCP restored next-step continuation differs")
    for field in ("model_sha256", "optimizer_sha256"):
        if uninterrupted_payload["next_step"][field] == uninterrupted_payload["boundary"][field]:
            raise ValueError(f"DCP step 2 did not update {field.removesuffix('_sha256')}")
    return {
        "schema": FULL_MODAL_ACTION_ADOPTION_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "nonzero_gradient_min_norm": core_payload["nonzero_gradient_min_norm"],
        "presence_subsets": core_payload["presence_subsets"],
        "modality_interventions": core_payload["modality_interventions"],
        "active_anytouch_sample_keys": core_payload["active_anytouch_sample_keys"],
        "dcp_cold_restore": {
            "checkpoint_global_step": 1,
            "optimizer_step": 1,
            "save_process_sha256": uninterrupted_payload["process_sha256"],
            "restore_process_sha256": restored_payload["process_sha256"],
            "checkpoint_artifact_sha256": uninterrupted_payload["checkpoint_artifact_sha256"],
            "saved_boundary": uninterrupted_payload["boundary"],
            "restored_boundary": restored_payload["boundary"],
            "uninterrupted_next_step": uninterrupted_payload["next_step"],
            "restored_next_step": restored_payload["next_step"],
        },
    }


def validate_action_dcp_phase_report(
    value: Mapping[str, Any],
    *,
    expected_phase: str,
) -> dict[str, Any]:
    """Validate and detach one standalone cold-restore phase report."""

    validated = _validated_dcp_phase(value, expected_phase=expected_phase)
    return json.loads(json.dumps(validated, allow_nan=False, sort_keys=True))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _linux_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        raise RuntimeError("Linux boot identity is unavailable")
    value = path.read_text(encoding="ascii").strip()
    if not value:
        raise RuntimeError("Linux boot identity is empty")
    return value


def current_process_start_ticks() -> int:
    """Read this process' kernel start tick without consuming a training RNG."""

    fields = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii").split()
    if len(fields) <= 21:
        raise RuntimeError("Linux process stat omits its start tick")
    value = int(fields[21])
    if value <= 0:
        raise RuntimeError("Linux process start tick is invalid")
    return value


def _boundary_mapping(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(ACTION_DCP_BOUNDARY_FIELDS):
        raise ValueError("DCP boundary fields differ")
    result = {
        field: _require_sha256(value[field], f"DCP boundary {field}")
        for field in ACTION_DCP_BOUNDARY_FIELDS
    }
    if len(set(result.values())) != len(result):
        raise ValueError("DCP boundary contains duplicate typed digests")
    return result


def _continuation_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"global_step", *ACTION_DCP_BOUNDARY_FIELDS, "action_output_sha256", "action_loss"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DCP continuation fields differ")
    global_step = value["global_step"]
    action_loss = value["action_loss"]
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("DCP continuation global step is invalid")
    if isinstance(action_loss, bool) or not isinstance(action_loss, (int, float)):
        raise ValueError("DCP continuation action loss is not numeric")
    action_loss = float(action_loss)
    if not math.isfinite(action_loss) or action_loss < 0:
        raise ValueError("DCP continuation action loss must be finite and nonnegative")
    result: dict[str, Any] = {"global_step": global_step, "action_loss": action_loss}
    for field in (*ACTION_DCP_BOUNDARY_FIELDS, "action_output_sha256"):
        result[field] = _require_sha256(value[field], f"DCP continuation {field}")
    if len({result[field] for field in (*ACTION_DCP_BOUNDARY_FIELDS, "action_output_sha256")}) != 5:
        raise ValueError("DCP continuation contains duplicate typed digests")
    return result


def _validated_probe_optimizer_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("action-adoption probe optimizer step is invalid")
    return value


def _validated_sample_key_list(value: object, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(key, str) or not key for key in value)
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must be one nonempty sorted unique string list")
    return list(value)


def _action_adoption_sample_keys(rows: object, *, name: str) -> list[str]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must be one nonempty list")
    resolved: list[str] | None = None
    for row in rows:
        if not isinstance(row, Mapping) or "sample_keys" not in row:
            raise ValueError(f"{name} omitted sample keys")
        keys = _validated_sample_key_list(row["sample_keys"], name=f"{name} sample keys")
        if resolved is None:
            resolved = keys
        elif keys != resolved:
            raise ValueError(f"{name} used inconsistent global samples")
    if resolved is None:
        raise RuntimeError(f"{name} sample-key validation vanished")
    return resolved


def _validated_action_adoption_presence(value: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = {
        "schema",
        "status",
        "action_loss_only",
        "probe_optimizer_step",
        "nonzero_gradient_min_norm",
        "presence_subsets",
        "active_anytouch_sample_keys",
        "parameter_groups",
        "artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action-adoption presence fields differ")
    if value["schema"] != ACTION_ADOPTION_PRESENCE_SCHEMA or value["status"] != "PASS":
        raise ValueError("action-adoption presence schema or status differs")
    if value["action_loss_only"] is not True:
        raise ValueError("action-adoption presence must use action loss only")
    _validated_probe_optimizer_step(value["probe_optimizer_step"])
    minimum = value["nonzero_gradient_min_norm"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or float(minimum) <= 0
    ):
        raise ValueError("action-adoption presence gradient minimum is invalid")
    subsets = value["presence_subsets"]
    if (
        not isinstance(subsets, list)
        or len(subsets) != len(DENSE_PRESENCE_CODES)
        or any(not isinstance(row, Mapping) for row in subsets)
        or tuple(row.get("name") for row in subsets) != DENSE_PRESENCE_CODES
    ):
        raise ValueError("action-adoption presence subsets differ")
    sample_keys = _action_adoption_sample_keys(subsets, name="presence subsets")
    active_keys = _validated_sample_key_list(
        value["active_anytouch_sample_keys"],
        name="active AnyTouch sample keys",
    )
    if not set(active_keys) <= set(sample_keys):
        raise ValueError("active AnyTouch samples are outside the probe batch")
    groups = value["parameter_groups"]
    if (
        not isinstance(groups, Mapping)
        or not groups
        or any(not isinstance(name, str) or not name for name in groups)
        or any(
            not isinstance(parameters, list)
            or not parameters
            or any(not isinstance(name, str) or not name for name in parameters)
            or len(parameters) != len(set(parameters))
            for parameters in groups.values()
        )
    ):
        raise ValueError("action-adoption presence parameter groups are invalid")
    artifact = _require_sha256(value["artifact_sha256"], "action-adoption presence artifact")
    unsigned = {key: child for key, child in value.items() if key != "artifact_sha256"}
    if _canonical_sha256(unsigned) != artifact:
        raise ValueError("action-adoption presence artifact digest differs")
    return value


def _validated_action_adoption_interventions(value: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = {
        "schema",
        "status",
        "action_loss_only",
        "probe_optimizer_step",
        "modality_interventions",
        "active_anytouch_sample_keys",
        "artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action-adoption intervention fields differ")
    if value["schema"] != ACTION_ADOPTION_INTERVENTIONS_SCHEMA or value["status"] != "PASS":
        raise ValueError("action-adoption intervention schema or status differs")
    if value["action_loss_only"] is not True:
        raise ValueError("action-adoption interventions must use action loss only")
    _validated_probe_optimizer_step(value["probe_optimizer_step"])
    interventions = value["modality_interventions"]
    if (
        not isinstance(interventions, list)
        or len(interventions) != len(DENSE_MODALITIES)
        or any(not isinstance(row, Mapping) for row in interventions)
        or tuple(row.get("modality") for row in interventions) != DENSE_MODALITIES
    ):
        raise ValueError("action-adoption modality interventions differ")
    sample_keys = _action_adoption_sample_keys(
        interventions,
        name="modality interventions",
    )
    active_keys = _validated_sample_key_list(
        value["active_anytouch_sample_keys"],
        name="active AnyTouch sample keys",
    )
    if not set(active_keys) <= set(sample_keys):
        raise ValueError("active AnyTouch samples are outside the intervention batch")
    artifact = _require_sha256(
        value["artifact_sha256"],
        "action-adoption intervention artifact",
    )
    unsigned = {key: child for key, child in value.items() if key != "artifact_sha256"}
    if _canonical_sha256(unsigned) != artifact:
        raise ValueError("action-adoption intervention artifact digest differs")
    return value


def _validated_core(value: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = {
        "schema",
        "status",
        "action_loss_only",
        "probe_optimizer_step",
        "nonzero_gradient_min_norm",
        "presence_subsets",
        "modality_interventions",
        "active_anytouch_sample_keys",
        "parameter_groups",
        "artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("action-adoption core fields differ")
    if value["schema"] != ACTION_ADOPTION_CORE_SCHEMA or value["status"] != "PASS":
        raise ValueError("action-adoption core schema or status differs")
    if value["action_loss_only"] is not True:
        raise ValueError("action-adoption core must use action loss only")
    artifact = _require_sha256(value["artifact_sha256"], "action-adoption core artifact")
    unsigned = {key: child for key, child in value.items() if key != "artifact_sha256"}
    if _canonical_sha256(unsigned) != artifact:
        raise ValueError("action-adoption core artifact digest differs")
    minimum = value["nonzero_gradient_min_norm"]
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or float(minimum) <= 0:
        raise ValueError("action-adoption core gradient minimum is invalid")
    return value


def _validated_dcp_phase(
    value: Mapping[str, Any],
    *,
    expected_phase: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "status",
        "phase",
        "checkpoint_global_step",
        "optimizer_step",
        "process_sha256",
        "checkpoint_artifact_sha256",
        "boundary",
        "next_step",
        "artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DCP phase report fields differ")
    if (
        value["schema"] != ACTION_DCP_PHASE_SCHEMA
        or value["status"] != "PASS"
        or value["phase"] != expected_phase
        or value["checkpoint_global_step"] != 1
        or value["optimizer_step"] != 1
    ):
        raise ValueError("DCP phase report contract differs")
    artifact = _require_sha256(value["artifact_sha256"], "DCP phase artifact")
    unsigned = {key: child for key, child in value.items() if key != "artifact_sha256"}
    if _canonical_sha256(unsigned) != artifact:
        raise ValueError("DCP phase artifact digest differs")
    _require_sha256(value["process_sha256"], "DCP phase process")
    _require_sha256(value["checkpoint_artifact_sha256"], "DCP phase checkpoint")
    _boundary_mapping(value["boundary"])
    _continuation_mapping(value["next_step"])
    return value

"""Disposable fixed-batch probes for LingBot relation-geometry recoverability.

This module changes no production forward or objective. It only freezes an
auditable parameter boundary and validates evidence produced by repeated
loss-only ownership supervision on one immutable two-frame observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from torch import nn

from picf_next.lingbot_native.host import LingBotNativeGraph
from picf_next.lingbot_native.relations import SharedRelationReadout

RELATION_GEOMETRY_FIXED_BATCH_ARMS = (
    "existing_readout_frozen_host",
    "structural_full_host",
)
RELATION_GEOMETRY_ARM_REPORT_SCHEMA = "picf-next.relation-geometry-fixed-batch-arm/v3"
RELATION_GEOMETRY_SAMPLE_SELECTION_RULE = (
    "earliest-source-only-exact-task-pair-with-future-visible-target-and-uncensored-inventory/v1"
)

_READOUT_COMPONENTS = (
    "picf_native_graph.relation_readout.projection.weight",
    "picf_native_graph.relation_readout.no_object",
    "picf_native_graph.relation_readout.temperature_parameter",
)
_CURVE_NAMES = (
    "ownership",
    "ownership_nll",
    "macro_soft_iou",
    "task_soft_iou",
    "action",
)
_PROVENANCE_DIGEST_FIELDS = (
    "patch_sha256",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
    "plan_sha256",
    "dataset_manifest_sha256",
    "physical_sidecar_manifest_sha256",
)
_PROVENANCE_FIELDS = {
    "source_commit",
    "checkpoint_revision",
    *_PROVENANCE_DIGEST_FIELDS,
    "seed",
    "fixed_sample_global_step",
    "sample_selection",
    "forward_seed_by_rank",
    "frame_sample_keys_by_rank",
    "frame_source_digests_by_rank",
    "objective",
    "optimizer",
}
_REPORT_FIELDS = {
    "schema",
    "status",
    "arm",
    "subject_sha256",
    "provenance",
    "trainable_scope",
    "curve_point_count",
    "optimizer_update_count",
    "global_curves",
    "rank_reports",
    "moe_routing_bias_unchanged",
    "maximum_peak_reserved_bytes",
    "total_time_s",
}


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    measured = float(value)
    if not math.isfinite(measured) or measured < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return measured


@dataclass(frozen=True, slots=True)
class RelationProbeSampleMetadata:
    """Loss-side source metadata used to decide probe eligibility."""

    sample_key: str
    task_key: str
    available_future_transitions: int
    target_identity_keys: tuple[str, ...] | None
    inventory_identity_keys: tuple[str, ...]
    target_supervised_pixel_counts: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_key, str) or not self.sample_key:
            raise ValueError("relation sample key must be non-empty")
        if not isinstance(self.task_key, str) or not self.task_key:
            raise ValueError("relation task key must be non-empty")
        _integer(
            self.available_future_transitions,
            name="relation available future transitions",
        )
        if self.target_identity_keys is not None and (
            not isinstance(self.target_identity_keys, tuple)
            or not self.target_identity_keys
            or len(set(self.target_identity_keys)) != len(self.target_identity_keys)
            or any(not isinstance(key, str) or not key for key in self.target_identity_keys)
        ):
            raise ValueError("relation exact task identities must be unique non-empty strings")
        if (
            not isinstance(self.inventory_identity_keys, tuple)
            or len(set(self.inventory_identity_keys)) != len(self.inventory_identity_keys)
            or any(not isinstance(key, str) or not key for key in self.inventory_identity_keys)
        ):
            raise ValueError("relation inventory identities must be unique non-empty strings")
        if self.target_identity_keys is None:
            if self.target_supervised_pixel_counts is not None:
                raise ValueError("inexact relation task cannot have target pixel counts")
        elif (
            not isinstance(self.target_supervised_pixel_counts, tuple)
            or len(self.target_supervised_pixel_counts) != len(self.target_identity_keys)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in self.target_supervised_pixel_counts
            )
        ):
            raise ValueError("relation target pixel counts must align with exact task identities")

    def eligible(self, *, capacity: int) -> bool:
        _integer(capacity, name="relation capacity", minimum=1)
        return (
            self.available_future_transitions >= 1
            and self.target_identity_keys is not None
            and bool(self.inventory_identity_keys)
            and set(self.target_identity_keys).issubset(self.inventory_identity_keys)
            and len(self.inventory_identity_keys) <= capacity
            and self.target_supervised_pixel_counts is not None
            and all(count > 0 for count in self.target_supervised_pixel_counts)
        )

    def as_dict(self, *, rank: int) -> dict[str, object]:
        _integer(rank, name="relation sample rank")
        return {
            "rank": rank,
            "sample_key": self.sample_key,
            "task_key": self.task_key,
            "available_future_transitions": self.available_future_transitions,
            "target_identity_keys": (
                None if self.target_identity_keys is None else list(self.target_identity_keys)
            ),
            "inventory_identity_keys": list(self.inventory_identity_keys),
            "target_supervised_pixel_counts": (
                None
                if self.target_supervised_pixel_counts is None
                else list(self.target_supervised_pixel_counts)
            ),
        }


@dataclass(frozen=True, slots=True)
class RelationProbeSampleSelection:
    """Earliest globally eligible fixed observation selected without model output."""

    selection_start_global_step: int
    selected_global_step: int
    inspected_step_count: int
    capacity: int
    samples_by_rank: tuple[RelationProbeSampleMetadata, ...]

    def __post_init__(self) -> None:
        start = _integer(
            self.selection_start_global_step,
            name="relation selection start step",
        )
        selected = _integer(
            self.selected_global_step,
            name="relation selected step",
        )
        inspected = _integer(
            self.inspected_step_count,
            name="relation inspected step count",
            minimum=1,
        )
        capacity = _integer(self.capacity, name="relation capacity", minimum=1)
        if selected < start or inspected != selected - start + 1:
            raise ValueError("relation selection interval is inconsistent")
        if (
            not isinstance(self.samples_by_rank, tuple)
            or len(self.samples_by_rank) != 2
            or any(
                not isinstance(sample, RelationProbeSampleMetadata)
                or not sample.eligible(capacity=capacity)
                for sample in self.samples_by_rank
            )
            or len({sample.sample_key for sample in self.samples_by_rank}) != 2
        ):
            raise ValueError("relation selection requires two distinct eligible rank samples")

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": RELATION_GEOMETRY_SAMPLE_SELECTION_RULE,
            "selection_start_global_step": self.selection_start_global_step,
            "selected_global_step": self.selected_global_step,
            "inspected_step_count": self.inspected_step_count,
            "capacity": self.capacity,
            "samples_by_rank": [
                sample.as_dict(rank=rank) for rank, sample in enumerate(self.samples_by_rank)
            ],
        }


def select_relation_geometry_probe_sample(
    *,
    selection_start_global_step: int,
    total_planned_steps: int,
    capacity: int,
    sample_keys_for_global_step: Callable[[int], Sequence[str]],
    metadata_for_sample_key: Callable[[str], RelationProbeSampleMetadata],
) -> RelationProbeSampleSelection:
    """Select the earliest eligible two-rank sample from source metadata only."""

    start = _integer(
        selection_start_global_step,
        name="relation selection start step",
    )
    total = _integer(total_planned_steps, name="relation total planned steps", minimum=1)
    resolved_capacity = _integer(capacity, name="relation capacity", minimum=1)
    if start >= total:
        raise ValueError("relation selection start lies outside the frozen plan")
    if not callable(sample_keys_for_global_step) or not callable(metadata_for_sample_key):
        raise TypeError("relation sample selection requires callable source resolvers")
    for candidate in range(start, total):
        sample_keys = tuple(sample_keys_for_global_step(candidate))
        if (
            len(sample_keys) != 2
            or len(set(sample_keys)) != 2
            or any(not isinstance(key, str) or not key for key in sample_keys)
        ):
            raise RuntimeError("relation stream plan must expose two distinct rank sample keys")
        metadata = tuple(metadata_for_sample_key(key) for key in sample_keys)
        if any(item.sample_key != key for item, key in zip(metadata, sample_keys, strict=True)):
            raise RuntimeError("relation source metadata changed sample identity")
        if all(item.eligible(capacity=resolved_capacity) for item in metadata):
            return RelationProbeSampleSelection(
                selection_start_global_step=start,
                selected_global_step=candidate,
                inspected_step_count=candidate - start + 1,
                capacity=resolved_capacity,
                samples_by_rank=metadata,
            )
    raise RuntimeError("frozen stream plan contains no two-rank exact-task relation probe sample")


def _curve(
    value: object,
    *,
    name: str,
    points: int,
    upper: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != points:
        raise ValueError(f"{name} must contain exactly {points} values")
    measured = tuple(_finite(item, name=name) for item in value)
    if upper is not None and any(item > upper for item in measured):
        raise ValueError(f"{name} exceeds {upper}")
    return measured


def _relation_readout_parameter_items(
    graph: LingBotNativeGraph,
) -> tuple[tuple[str, nn.Parameter], ...]:
    readout = graph.relation_readout
    if not isinstance(readout, SharedRelationReadout):
        raise TypeError("relation probe requires the production shared relation readout")
    raw_values = (
        (_READOUT_COMPONENTS[0], readout.projection.weight),
        (_READOUT_COMPONENTS[1], readout.no_object),
        (_READOUT_COMPONENTS[2], readout.temperature_parameter),
    )
    values: list[tuple[str, nn.Parameter]] = []
    for name, value in raw_values:
        if not isinstance(value, nn.Parameter):
            raise TypeError(f"relation readout component {name!r} is not a parameter")
        values.append((name, value))
    if len({id(value) for _, value in values}) != len(values):
        raise RuntimeError("relation probe readout parameters are aliased")
    return tuple(values)


def _relation_readout_parameters(graph: LingBotNativeGraph) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for _, parameter in _relation_readout_parameter_items(graph))


@dataclass(frozen=True, slots=True)
class RelationGeometryTrainableScope:
    """Exact trainable parameter schema for one disposable probe arm."""

    arm: str
    parameter_count: int
    trainable_numel: int
    schema_sha256: str
    parameter_descriptors: tuple[tuple[str, tuple[int, ...], str, int], ...]

    def __post_init__(self) -> None:
        if self.arm not in RELATION_GEOMETRY_FIXED_BATCH_ARMS:
            raise ValueError("unknown relation-geometry probe arm")
        _integer(self.parameter_count, name="relation parameter count", minimum=1)
        _integer(self.trainable_numel, name="relation trainable elements", minimum=1)
        _sha256(self.schema_sha256, name="relation trainable schema")
        names: list[str] = []
        total_numel = 0
        serialized: list[dict[str, object]] = []
        for name, shape, dtype, numel in self.parameter_descriptors:
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(shape, tuple)
                or any(
                    isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                    for dimension in shape
                )
                or not isinstance(dtype, str)
                or not dtype
            ):
                raise ValueError("relation trainable parameter descriptor is malformed")
            _integer(numel, name="relation trainable parameter numel", minimum=1)
            if math.prod(shape) != numel:
                raise ValueError("relation trainable shape and numel differ")
            names.append(name)
            total_numel += numel
            serialized.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype,
                    "numel": numel,
                }
            )
        if (
            len(names) != self.parameter_count
            or names != sorted(names)
            or len(set(names)) != len(names)
            or total_numel != self.trainable_numel
            or _canonical_digest(serialized) != self.schema_sha256
        ):
            raise ValueError("relation trainable parameter schema is inconsistent")
        if self.arm == "existing_readout_frozen_host" and tuple(names) != tuple(
            sorted(_READOUT_COMPONENTS)
        ):
            raise ValueError("frozen-host arm is not limited to the ownership readout")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(value[0] for value in self.parameter_descriptors)

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "parameter_count": self.parameter_count,
            "trainable_numel": self.trainable_numel,
            "schema_sha256": self.schema_sha256,
            "parameters": [
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype,
                    "numel": numel,
                }
                for name, shape, dtype, numel in self.parameter_descriptors
            ],
        }


def _describe_scope(
    policy: nn.Module,
    graph: LingBotNativeGraph,
    *,
    arm: str,
) -> RelationGeometryTrainableScope:
    if arm == "existing_readout_frozen_host":
        descriptors = sorted(
            (
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                parameter.numel(),
            )
            for name, parameter in _relation_readout_parameter_items(graph)
        )
    else:
        descriptors = sorted(
            (
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                parameter.numel(),
            )
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        )
    if not descriptors:
        raise RuntimeError("relation-geometry arm has no trainable parameters")
    serialized = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "numel": numel,
        }
        for name, shape, dtype, numel in descriptors
    ]
    return RelationGeometryTrainableScope(
        arm=arm,
        parameter_count=len(descriptors),
        trainable_numel=sum(value[3] for value in descriptors),
        schema_sha256=_canonical_digest(serialized),
        parameter_descriptors=tuple(descriptors),
    )


def configure_relation_geometry_trainable_scope(
    policy: nn.Module,
    graph: LingBotNativeGraph,
    *,
    arm: str,
) -> RelationGeometryTrainableScope:
    """Freeze the host only for arm A; arm D preserves production trainability."""

    if arm not in RELATION_GEOMETRY_FIXED_BATCH_ARMS:
        raise ValueError("unknown relation-geometry probe arm")
    if arm == "existing_readout_frozen_host":
        selected_ids = {id(value) for value in _relation_readout_parameters(graph)}
        for parameter in policy.parameters():
            parameter.requires_grad_(id(parameter) in selected_ids)
        if {id(value) for value in policy.parameters() if value.requires_grad} != selected_ids:
            raise RuntimeError("relation readout is not uniquely installed in the policy")
    return _describe_scope(policy, graph, arm=arm)


def verify_relation_geometry_trainable_scope(
    policy: nn.Module,
    graph: LingBotNativeGraph,
    *,
    expected: RelationGeometryTrainableScope,
) -> RelationGeometryTrainableScope:
    """Verify that FSDP2 wrapping preserved the disposable probe boundary."""

    if expected.arm == "existing_readout_frozen_host":
        selected_ids = {id(value) for value in _relation_readout_parameters(graph)}
        observed_ids = {id(value) for value in policy.parameters() if value.requires_grad}
        if observed_ids != selected_ids:
            raise RuntimeError("FSDP2 changed the frozen-host relation parameter boundary")
    observed = _describe_scope(policy, graph, arm=expected.arm)
    if observed != expected:
        raise RuntimeError("FSDP2 changed the relation-geometry trainable schema")
    return observed


def relation_geometry_probe_subject(
    provenance: Mapping[str, object],
    *,
    curve_point_count: int,
) -> str:
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("relation-geometry provenance fields differ from schema")
    points = _integer(curve_point_count, name="relation curve points", minimum=2)
    return _canonical_digest(
        {
            "schema": RELATION_GEOMETRY_ARM_REPORT_SCHEMA,
            "provenance": dict(provenance),
            "curve_point_count": points,
            "optimizer_update_count": points - 1,
        }
    )


def _parse_scope(value: object, *, arm: str) -> RelationGeometryTrainableScope:
    if not isinstance(value, Mapping) or set(value) != {
        "arm",
        "parameter_count",
        "trainable_numel",
        "schema_sha256",
        "parameters",
    }:
        raise ValueError("relation trainable-scope fields differ from schema")
    raw_parameters = value["parameters"]
    if not isinstance(raw_parameters, list):
        raise ValueError("relation trainable parameters must be one list")
    descriptors: list[tuple[str, tuple[int, ...], str, int]] = []
    for raw in raw_parameters:
        if not isinstance(raw, Mapping) or set(raw) != {"name", "shape", "dtype", "numel"}:
            raise ValueError("relation trainable parameter fields differ from schema")
        shape = raw["shape"]
        if not isinstance(shape, list):
            raise ValueError("relation trainable shape must be one list")
        descriptors.append(
            (
                raw["name"] if isinstance(raw["name"], str) else "",
                tuple(shape),
                raw["dtype"] if isinstance(raw["dtype"], str) else "",
                _integer(raw["numel"], name="relation trainable parameter numel", minimum=1),
            )
        )
    parsed = RelationGeometryTrainableScope(
        arm=value["arm"] if isinstance(value["arm"], str) else "",
        parameter_count=_integer(
            value["parameter_count"],
            name="relation trainable parameter count",
            minimum=1,
        ),
        trainable_numel=_integer(
            value["trainable_numel"],
            name="relation trainable elements",
            minimum=1,
        ),
        schema_sha256=_sha256(value["schema_sha256"], name="relation trainable schema"),
        parameter_descriptors=tuple(descriptors),
    )
    if parsed.arm != arm:
        raise ValueError("relation trainable scope belongs to another arm")
    return parsed


def validate_relation_probe_sample_selection(
    value: object,
) -> RelationProbeSampleSelection:
    """Validate and reconstruct one source-only fixed-sample selection."""

    expected_fields = {
        "rule",
        "selection_start_global_step",
        "selected_global_step",
        "inspected_step_count",
        "capacity",
        "samples_by_rank",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("relation sample-selection fields differ from schema")
    if value["rule"] != RELATION_GEOMETRY_SAMPLE_SELECTION_RULE:
        raise ValueError("relation sample-selection rule differs from preregistration")
    raw_samples = value["samples_by_rank"]
    if not isinstance(raw_samples, list) or len(raw_samples) != 2:
        raise ValueError("relation sample selection must bind two ranks")
    samples: list[RelationProbeSampleMetadata] = []
    for expected_rank, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping) or set(raw) != {
            "rank",
            "sample_key",
            "task_key",
            "available_future_transitions",
            "target_identity_keys",
            "inventory_identity_keys",
            "target_supervised_pixel_counts",
        }:
            raise ValueError("relation selected-sample fields differ from schema")
        if raw["rank"] != expected_rank:
            raise ValueError("relation selected samples must use frozen rank order")
        target = raw["target_identity_keys"]
        inventory = raw["inventory_identity_keys"]
        pixel_counts = raw["target_supervised_pixel_counts"]
        if (
            not isinstance(target, list)
            or not isinstance(inventory, list)
            or not isinstance(pixel_counts, list)
        ):
            raise ValueError("selected relation sample must have exact target and inventory lists")
        samples.append(
            RelationProbeSampleMetadata(
                sample_key=raw["sample_key"] if isinstance(raw["sample_key"], str) else "",
                task_key=raw["task_key"] if isinstance(raw["task_key"], str) else "",
                available_future_transitions=_integer(
                    raw["available_future_transitions"],
                    name="relation available future transitions",
                ),
                target_identity_keys=tuple(target),
                inventory_identity_keys=tuple(inventory),
                target_supervised_pixel_counts=tuple(pixel_counts),
            )
        )
    parsed = RelationProbeSampleSelection(
        selection_start_global_step=_integer(
            value["selection_start_global_step"],
            name="relation selection start step",
        ),
        selected_global_step=_integer(
            value["selected_global_step"],
            name="relation selected step",
        ),
        inspected_step_count=_integer(
            value["inspected_step_count"],
            name="relation inspected step count",
            minimum=1,
        ),
        capacity=_integer(value["capacity"], name="relation capacity", minimum=1),
        samples_by_rank=tuple(samples),
    )
    if parsed.as_dict() != dict(value):
        raise ValueError("relation sample selection is not canonical")
    return parsed


def _validate_provenance(value: object, *, points: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise ValueError("relation-geometry provenance fields differ from schema")
    parsed = dict(value)
    for name in ("source_commit", "checkpoint_revision"):
        if not isinstance(parsed[name], str) or not parsed[name]:
            raise ValueError(f"relation {name} must be non-empty")
    for name in _PROVENANCE_DIGEST_FIELDS:
        _sha256(parsed[name], name=f"relation {name}")
    _integer(parsed["seed"], name="relation seed")
    fixed_sample_step = _integer(
        parsed["fixed_sample_global_step"],
        name="relation sample step",
    )
    selection = validate_relation_probe_sample_selection(parsed["sample_selection"])
    if selection.selected_global_step != fixed_sample_step:
        raise ValueError("relation selected step differs from fixed sample step")
    forward_seeds = parsed["forward_seed_by_rank"]
    if (
        not isinstance(forward_seeds, list)
        or len(forward_seeds) != 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in forward_seeds
        )
    ):
        raise ValueError("relation forward seeds must bind both ranks")
    sample_keys = parsed["frame_sample_keys_by_rank"]
    source_digests = parsed["frame_source_digests_by_rank"]
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
        raise ValueError("relation sample keys must bind two frames on both ranks")
    if tuple(sample.sample_key for sample in selection.samples_by_rank) != tuple(
        rank_keys[0] for rank_keys in sample_keys
    ):
        raise ValueError("relation selected samples differ from executed current frames")
    if (
        not isinstance(source_digests, list)
        or len(source_digests) != 2
        or any(
            not isinstance(rank_value, list)
            or len(rank_value) != 2
            or any(_sha256(item, name="relation source digest") != item for item in rank_value)
            for rank_value in source_digests
        )
    ):
        raise ValueError("relation source digests must bind two frames on both ranks")
    expected_objective = {
        "optimized_term": "set/ownership",
        "observed_terms": list(_CURVE_NAMES),
        "window": "fixed_two_frame_local_bptt",
        "labels_are_loss_side_only": True,
        "row_gauge": "initial_assignment_then_frozen",
        "forward_randomness": "fixed_per_rank_torch_seed",
        "official_policy_loss": "observed_not_optimized",
        "predictive_queries": "absent",
    }
    if (
        not isinstance(parsed["objective"], Mapping)
        or dict(parsed["objective"]) != expected_objective
    ):
        raise ValueError("relation probe objective differs from its preregistration")
    optimizer = parsed["optimizer"]
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer)
        != {
            "algorithm",
            "learning_rate_hex",
            "weight_decay_hex",
            "scheduler",
            "moe_load_balance_hook_enabled",
            "update_count",
        }
        or optimizer["scheduler"] != "constant"
        or optimizer["moe_load_balance_hook_enabled"] is not False
        or optimizer["update_count"] != points - 1
    ):
        raise ValueError("relation probe optimizer differs from its preregistration")
    for name in ("learning_rate_hex", "weight_decay_hex"):
        raw = optimizer[name]
        if not isinstance(raw, str) or float.fromhex(raw) < 0:
            raise ValueError(f"relation optimizer {name} is invalid")
    if not isinstance(optimizer["algorithm"], str) or not optimizer["algorithm"]:
        raise ValueError("relation optimizer algorithm must be named")
    return parsed


def validate_relation_geometry_arm_report(value: object) -> dict[str, Any]:
    """Validate one arm and recompute rank means and immutable subject identity."""

    if not isinstance(value, Mapping) or set(value) != _REPORT_FIELDS:
        raise ValueError("relation-geometry report fields differ from schema")
    if value["schema"] != RELATION_GEOMETRY_ARM_REPORT_SCHEMA or value["status"] != "PASS":
        raise ValueError("relation-geometry arm did not complete")
    arm = value["arm"]
    if not isinstance(arm, str) or arm not in RELATION_GEOMETRY_FIXED_BATCH_ARMS:
        raise ValueError("relation-geometry arm is unsupported")
    points = _integer(value["curve_point_count"], name="relation curve points", minimum=2)
    updates = _integer(value["optimizer_update_count"], name="relation updates", minimum=1)
    if updates != points - 1:
        raise ValueError("relation updates must equal curve points minus one")
    provenance = _validate_provenance(value["provenance"], points=points)
    if _sha256(value["subject_sha256"], name="relation subject") != relation_geometry_probe_subject(
        provenance,
        curve_point_count=points,
    ):
        raise ValueError("relation subject differs from its provenance")
    scope = _parse_scope(value["trainable_scope"], arm=arm)
    global_raw = value["global_curves"]
    if not isinstance(global_raw, Mapping) or set(global_raw) != set(_CURVE_NAMES):
        raise ValueError("relation global curves differ from schema")
    global_curves = {
        name: _curve(
            global_raw[name],
            name=f"global {name}",
            points=points,
            upper=1.0 if name in {"macro_soft_iou", "task_soft_iou"} else None,
        )
        for name in _CURVE_NAMES
    }
    rank_reports = value["rank_reports"]
    if not isinstance(rank_reports, list) or len(rank_reports) != 2:
        raise ValueError("relation report requires exactly two rank reports")
    rank_curves: list[dict[str, tuple[float, ...]]] = []
    observed_ranks: list[int] = []
    sample_keys_by_rank = cast(list[list[str]], provenance["frame_sample_keys_by_rank"])
    source_digests_by_rank = cast(list[list[str]], provenance["frame_source_digests_by_rank"])
    forward_seeds = cast(list[int], provenance["forward_seed_by_rank"])
    for raw in rank_reports:
        if not isinstance(raw, Mapping) or set(raw) != {
            "rank",
            "frame_sample_keys",
            "frame_source_digests",
            "forward_seed",
            "row_bindings",
            "curves",
            "task_diagnostics_by_point",
            "visual_artifacts_by_point",
            "gradient_probe",
            "step_times_s",
            "peak_reserved_bytes",
        }:
            raise ValueError("relation rank-report fields differ from schema")
        rank = _integer(raw["rank"], name="relation rank")
        if rank not in (0, 1):
            raise ValueError("relation rank must be zero or one")
        observed_ranks.append(rank)
        if (
            raw["frame_sample_keys"] != sample_keys_by_rank[rank]
            or raw["frame_source_digests"] != source_digests_by_rank[rank]
            or raw["forward_seed"] != forward_seeds[rank]
        ):
            raise ValueError("relation rank data differs from common provenance")
        row_bindings = raw["row_bindings"]
        if (
            not isinstance(row_bindings, list)
            or not row_bindings
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 0
                for item in row_bindings
            )
        ):
            raise ValueError("relation rank report has malformed frozen row bindings")
        curves = raw["curves"]
        if not isinstance(curves, Mapping) or set(curves) != set(_CURVE_NAMES):
            raise ValueError("relation rank curves differ from schema")
        rank_curves.append(
            {
                name: _curve(
                    curves[name],
                    name=f"rank {rank} {name}",
                    points=points,
                    upper=1.0 if name in {"macro_soft_iou", "task_soft_iou"} else None,
                )
                for name in _CURVE_NAMES
            }
        )
        for name in ("task_diagnostics_by_point", "visual_artifacts_by_point"):
            sequence = raw[name]
            if (
                not isinstance(sequence, list)
                or len(sequence) != points
                or any(not isinstance(item, list) or not item for item in sequence)
            ):
                raise ValueError(f"relation {name} must contain evidence at every point")
        if not isinstance(raw["gradient_probe"], Mapping) or not raw["gradient_probe"]:
            raise ValueError("relation rank report omitted its first-update gradient probe")
        times = _curve(raw["step_times_s"], name="relation step time", points=points)
        if any(item <= 0 for item in times):
            raise ValueError("relation step times must be positive")
        _integer(raw["peak_reserved_bytes"], name="relation peak reserved bytes")
    if observed_ranks != [0, 1]:
        raise ValueError("relation rank reports must be in frozen rank order")
    for name in _CURVE_NAMES:
        for point in range(points):
            expected = sum(curves[name][point] for curves in rank_curves) / 2
            if not math.isclose(
                global_curves[name][point],
                expected,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError(f"global relation {name} differs from rank mean")
    if value["moe_routing_bias_unchanged"] is not True:
        raise ValueError("relation probe changed the LingBot MoE routing bias")
    maximum_peak = _integer(
        value["maximum_peak_reserved_bytes"],
        name="relation maximum peak bytes",
    )
    if maximum_peak != max(int(raw["peak_reserved_bytes"]) for raw in rank_reports):
        raise ValueError("relation maximum peak bytes differ from rank reports")
    if _finite(value["total_time_s"], name="relation total time") <= 0:
        raise ValueError("relation total time must be positive")
    return {
        **dict(value),
        "provenance": provenance,
        "trainable_scope": scope.as_dict(),
    }

"""Evidence-only fixed-batch capacity probes for the LingBot-native host.

This module does not add a model component or a training loss.  It freezes the
four-arm diagnostic contract used to distinguish shared-host learning from
native-graph or small prediction-interface fitting:

* ``full_host`` trains every normally trainable released-host/native-graph
  parameter in the production configuration;
* ``native_graph_only`` freezes the host and trains the complete native graph;
* ``readout_only`` trains only the prediction query/readout interface;
* ``shuffled_target`` uses the full trainable host but changes the object-to-
  target association over repeated presentations of the same observation.

The shuffled control preserves target validity and importance.  Therefore a
different curve cannot be explained by changing which rows carry supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import nn

from picf_next.lingbot_native.host import LingBotNativeGraph
from picf_next.lingbot_native.predictive_objective import NativePredictiveTarget
from picf_next.lingbot_native.predictive_probes import (
    PREDICTIVE_FIXED_BATCH_ARMS,
    predictive_fixed_batch_fit_diagnostics,
    predictive_fixed_batch_fit_from_mapping,
)

PREDICTIVE_FIXED_BATCH_ARM_REPORT_SCHEMA = "picf-next.lingbot-predictive-fixed-batch-arm/v3"
PREDICTIVE_FIXED_BATCH_EXPERIMENT_REPORT_SCHEMA = (
    "picf-next.lingbot-predictive-fixed-batch-experiment/v3"
)

_READOUT_ONLY_COMPONENTS = (
    "prediction_role",
    "prediction_route_embeddings",
    "prediction_horizon_projection",
    "prediction_address_projection",
    "predictive_readouts",
)
_NATIVE_GRAPH_ONLY_COMPONENTS = ("picf_native_graph",)
_PROVENANCE_DIGEST_FIELDS = (
    "patch_sha256",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
    "plan_sha256",
    "dataset_manifest_sha256",
    "physical_sidecar_manifest_sha256",
    "predictive_cache_manifest_sha256",
    "current_grid_cache_manifest_sha256",
)
_PROVENANCE_FIELDS = {
    "source_commit",
    "checkpoint_revision",
    *_PROVENANCE_DIGEST_FIELDS,
    "seed",
    "fixed_sample_global_step",
    "frame_sample_keys_by_rank",
    "frame_source_digests_by_rank",
    "objective",
    "optimizer",
}
_ARM_REPORT_FIELDS = {
    "schema",
    "status",
    "arm",
    "subject_sha256",
    "provenance",
    "trainable_scope",
    "curve_point_count",
    "optimizer_update_count",
    "global_loss_curve",
    "global_shuffle_distance_curve",
    "rank_reports",
    "shared_host_gradient_probe",
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


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    measured = float(value)
    if not math.isfinite(measured) or (non_negative and measured < 0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if non_negative else ""))
    return measured


def _curve(value: object, *, name: str, steps: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != steps:
        raise ValueError(f"{name} must contain exactly {steps} values")
    return tuple(_finite(item, name=name, non_negative=True) for item in value)


def _prediction_interface_parameters(graph: LingBotNativeGraph) -> tuple[nn.Parameter, ...]:
    values: list[nn.Parameter] = [
        graph.prediction_role,
        graph.prediction_route_embeddings,
        *graph.prediction_horizon_projection.parameters(),
        *graph.predictive_readouts.parameters(),
    ]
    if graph.prediction_address_projection is not None:
        values.extend(graph.prediction_address_projection.parameters())
    if not values or len({id(value) for value in values}) != len(values):
        raise RuntimeError("native prediction interface parameters are absent or aliased")
    return tuple(values)


def _native_graph_parameters(graph: LingBotNativeGraph) -> tuple[nn.Parameter, ...]:
    values = tuple(graph.parameters())
    if not values or len({id(value) for value in values}) != len(values):
        raise RuntimeError("native graph parameters are absent or aliased")
    return values


@dataclass(frozen=True, slots=True)
class FixedBatchTrainableScope:
    """Immutable description of the parameters owned by one diagnostic arm."""

    arm: str
    parameter_count: int
    trainable_numel: int
    schema_sha256: str
    component_names: tuple[str, ...]
    parameter_descriptors: tuple[tuple[str, tuple[int, ...], str, int], ...]

    def __post_init__(self) -> None:
        if self.arm not in PREDICTIVE_FIXED_BATCH_ARMS:
            raise ValueError("unknown fixed-batch arm")
        _positive_integer(self.parameter_count, name="trainable parameter count")
        _positive_integer(self.trainable_numel, name="trainable parameter elements")
        _sha256(self.schema_sha256, name="trainable parameter schema")
        if not isinstance(self.parameter_descriptors, tuple):
            raise ValueError("trainable parameter descriptors must be one tuple")
        normalized: list[dict[str, object]] = []
        names: list[str] = []
        total_numel = 0
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
                raise ValueError("trainable parameter descriptor is malformed")
            _positive_integer(numel, name="trainable parameter descriptor numel")
            if math.prod(shape) != numel:
                raise ValueError("trainable parameter shape and numel differ")
            names.append(name)
            total_numel += numel
            normalized.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype,
                    "numel": numel,
                }
            )
        if (
            len(normalized) != self.parameter_count
            or names != sorted(names)
            or len(set(names)) != len(names)
            or total_numel != self.trainable_numel
        ):
            raise ValueError("trainable parameter descriptors are inconsistent")
        if _canonical_digest(normalized) != self.schema_sha256:
            raise ValueError("trainable parameter schema digest is inconsistent")
        if self.arm == "readout_only":
            expected_components = _READOUT_ONLY_COMPONENTS
        elif self.arm == "native_graph_only":
            expected_components = _NATIVE_GRAPH_ONLY_COMPONENTS
        else:
            expected_components = ()
        if self.component_names != expected_components:
            raise ValueError("fixed-batch component scope differs from its arm")
        if self.arm == "readout_only":
            observed_components = {
                component
                for name in names
                for component in _READOUT_ONLY_COMPONENTS
                if component in name.split(".")
            }
            if observed_components != set(_READOUT_ONLY_COMPONENTS) or any(
                not any(component in name.split(".") for component in _READOUT_ONLY_COMPONENTS)
                for name in names
            ):
                raise ValueError(
                    "readout-only parameters are not exactly prediction-interface components"
                )
        if self.arm == "native_graph_only" and any(
            "picf_native_graph" not in name.split(".") for name in names
        ):
            raise ValueError("native-graph-only parameters escape the installed native graph")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(value[0] for value in self.parameter_descriptors)

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "parameter_count": self.parameter_count,
            "trainable_numel": self.trainable_numel,
            "schema_sha256": self.schema_sha256,
            "component_names": list(self.component_names),
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


def configure_fixed_batch_trainable_scope(
    policy: nn.Module,
    graph: LingBotNativeGraph,
    *,
    arm: str,
) -> FixedBatchTrainableScope:
    """Apply and hash the frozen trainability boundary for one fresh arm."""

    if not isinstance(policy, nn.Module) or not isinstance(graph, LingBotNativeGraph):
        raise TypeError("fixed-batch trainability requires the installed native policy graph")
    if arm not in PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("unknown fixed-batch arm")
    if arm in {"native_graph_only", "readout_only"}:
        selected = (
            _native_graph_parameters(graph)
            if arm == "native_graph_only"
            else _prediction_interface_parameters(graph)
        )
        selected_ids = {id(value) for value in selected}
        for parameter in policy.parameters():
            parameter.requires_grad_(id(parameter) in selected_ids)
        observed_ids = {id(value) for value in policy.parameters() if value.requires_grad}
        if observed_ids != selected_ids:
            raise RuntimeError(
                f"{arm.replace('_', '-')} parameters are not exactly installed in the policy"
            )

    return _describe_fixed_batch_trainable_scope(policy, arm=arm)


def _describe_fixed_batch_trainable_scope(
    policy: nn.Module,
    *,
    arm: str,
) -> FixedBatchTrainableScope:
    descriptors: list[tuple[str, tuple[int, ...], str, int]] = []
    trainable_numel = 0
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_numel += parameter.numel()
        descriptors.append(
            (
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                parameter.numel(),
            )
        )
    descriptors.sort(key=lambda value: value[0])
    if not descriptors:
        raise RuntimeError("fixed-batch arm has no trainable parameters")
    serialized = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "numel": numel,
        }
        for name, shape, dtype, numel in descriptors
    ]
    return FixedBatchTrainableScope(
        arm=arm,
        parameter_count=len(descriptors),
        trainable_numel=trainable_numel,
        schema_sha256=_canonical_digest(serialized),
        component_names=(
            _READOUT_ONLY_COMPONENTS
            if arm == "readout_only"
            else _NATIVE_GRAPH_ONLY_COMPONENTS
            if arm == "native_graph_only"
            else ()
        ),
        parameter_descriptors=tuple(descriptors),
    )


def verify_fixed_batch_trainable_scope(
    policy: nn.Module,
    graph: LingBotNativeGraph,
    *,
    expected: FixedBatchTrainableScope,
) -> FixedBatchTrainableScope:
    """Verify that distributed wrapping preserved the exact frozen boundary."""

    if not isinstance(policy, nn.Module) or not isinstance(graph, LingBotNativeGraph):
        raise TypeError("fixed-batch scope verification requires the installed native graph")
    if not isinstance(expected, FixedBatchTrainableScope):
        raise TypeError("fixed-batch scope verification requires one frozen expected scope")
    if expected.arm in {"native_graph_only", "readout_only"}:
        selected_ids = {
            id(value)
            for value in (
                _native_graph_parameters(graph)
                if expected.arm == "native_graph_only"
                else _prediction_interface_parameters(graph)
            )
        }
        observed_ids = {id(value) for value in policy.parameters() if value.requires_grad}
        if observed_ids != selected_ids:
            raise RuntimeError(
                f"distributed wrapping changed the {expected.arm.replace('_', '-')} "
                "parameter boundary"
            )
    observed = _describe_fixed_batch_trainable_scope(policy, arm=expected.arm)
    if observed != expected:
        raise RuntimeError("distributed wrapping changed the fixed-batch trainable schema")
    return observed


def shuffled_predictive_target(
    target: NativePredictiveTarget,
    *,
    curve_index: int,
) -> tuple[NativePredictiveTarget, float]:
    """Vary object features while preserving the factual support measure.

    Curve point zero is intentionally factual so all four arms must begin
    from the same loss. Later points cycle features only among currently
    supervised tracks. Validity, importance, route, horizon and track
    identities remain unchanged, isolating the semantic target association.
    """

    if not isinstance(target, NativePredictiveTarget):
        raise TypeError("target shuffling requires one native predictive target")
    step = _positive_integer(
        curve_index,
        name="fixed-batch curve index",
        minimum=0,
    )
    features = target.features.clone()
    permutations: list[dict[str, object]] = []
    for batch_index, identity_keys in enumerate(target.track_identity_keys):
        track_limit = len(identity_keys)
        for query_index in range(target.features.shape[2]):
            supervised = torch.nonzero(
                target.valid[batch_index, :track_limit, query_index],
                as_tuple=False,
            ).flatten()
            count = int(supervised.numel())
            shift = 0 if step == 0 or count < 2 else 1 + ((step - 1) % (count - 1))
            if shift:
                donor = supervised.roll(shifts=-shift)
                features[batch_index, supervised, query_index] = target.features[
                    batch_index,
                    donor,
                    query_index,
                ]
            permutations.append(
                {
                    "batch_index": batch_index,
                    "query_index": query_index,
                    "supervised_track_indices": [
                        int(value) for value in supervised.detach().cpu().tolist()
                    ],
                    "shift": shift,
                }
            )
    weighted_distance = ((features - target.features).abs().mean(dim=-1) * target.importance).sum()
    weight = target.importance.sum()
    distance = (
        0.0
        if float(weight.detach().float().item()) == 0.0
        else float((weighted_distance / weight).detach().float().item())
    )
    target_data_digest = _canonical_digest(
        {
            "base_target_data_digest": target.target_data_digest,
            "control": "cycle_features_within_currently_supervised_tracks",
            "curve_index": step,
            "permutations": permutations,
        }
    )
    return (
        replace(
            target,
            features=features.detach(),
            target_data_digest=target_data_digest,
        ),
        distance,
    )


class ShuffledCurrentGridTargetCache:
    """Probe-only facade that changes no input, state, validity or loss weight."""

    def __init__(self, base_cache: object) -> None:
        contract = getattr(base_cache, "contract", None)
        resolver = getattr(base_cache, "current_correction_summary_target_for", None)
        if contract is None or not callable(resolver):
            raise TypeError("shuffled target control requires a current-grid target cache")
        self._base_cache: Any = base_cache
        self.contract = contract
        self._curve_index = 0
        self._distances: list[float] = []

    def begin_curve_point(self, curve_index: int) -> None:
        self._curve_index = _positive_integer(
            curve_index,
            name="fixed-batch curve index",
            minimum=0,
        )
        self._distances = []

    @property
    def maximum_distance(self) -> float:
        return max(self._distances, default=0.0)

    def current_correction_summary_target_for(self, **kwargs: Any) -> NativePredictiveTarget:
        resolver = self._base_cache.current_correction_summary_target_for
        target = resolver(**kwargs)
        shuffled, distance = shuffled_predictive_target(
            target,
            curve_index=self._curve_index,
        )
        self._distances.append(distance)
        return shuffled


def fixed_batch_probe_subject(
    provenance: Mapping[str, object],
    *,
    curve_point_count: int,
) -> str:
    """Hash the common experiment subject without including an arm outcome."""

    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("fixed-batch provenance fields differ from schema")
    curve_points = _positive_integer(
        curve_point_count,
        name="fixed-batch curve-point count",
        minimum=2,
    )
    return _canonical_digest(
        {
            "schema": PREDICTIVE_FIXED_BATCH_EXPERIMENT_REPORT_SCHEMA,
            "provenance": dict(provenance),
            "curve_point_count": curve_points,
            "optimizer_update_count": curve_points - 1,
        }
    )


def _parse_trainable_scope(value: object, *, expected_arm: str) -> FixedBatchTrainableScope:
    required = {
        "arm",
        "parameter_count",
        "trainable_numel",
        "schema_sha256",
        "component_names",
        "parameters",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("fixed-batch trainable scope fields differ from schema")
    components = value["component_names"]
    parameters = value["parameters"]
    if not isinstance(components, list) or not isinstance(parameters, list):
        raise ValueError("fixed-batch trainable components/parameters must be lists")
    descriptors: list[tuple[str, tuple[int, ...], str, int]] = []
    for raw in parameters:
        if not isinstance(raw, Mapping) or set(raw) != {"name", "shape", "dtype", "numel"}:
            raise ValueError("fixed-batch parameter descriptor fields differ from schema")
        shape = raw["shape"]
        if not isinstance(shape, list):
            raise ValueError("fixed-batch parameter shape must be one list")
        descriptors.append(
            (
                raw["name"] if isinstance(raw["name"], str) else "",
                tuple(shape),
                raw["dtype"] if isinstance(raw["dtype"], str) else "",
                _positive_integer(
                    raw["numel"],
                    name="trainable parameter descriptor numel",
                ),
            )
        )
    scope = FixedBatchTrainableScope(
        arm=value["arm"] if isinstance(value["arm"], str) else "",
        parameter_count=_positive_integer(
            value["parameter_count"],
            name="trainable parameter count",
        ),
        trainable_numel=_positive_integer(
            value["trainable_numel"],
            name="trainable parameter elements",
        ),
        schema_sha256=_sha256(
            value["schema_sha256"],
            name="trainable parameter schema",
        ),
        component_names=tuple(components),
        parameter_descriptors=tuple(descriptors),
    )
    if scope.arm != expected_arm:
        raise ValueError("fixed-batch trainable scope belongs to another arm")
    return scope


def _validate_provenance(value: object, *, curve_point_count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise ValueError("fixed-batch provenance fields differ from schema")
    result = dict(value)
    for name in ("source_commit", "checkpoint_revision"):
        raw = result[name]
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"fixed-batch {name} must be non-empty")
    for name in _PROVENANCE_DIGEST_FIELDS:
        _sha256(result[name], name=f"fixed-batch {name}")
    _positive_integer(result["seed"], name="fixed-batch seed", minimum=0)
    _positive_integer(
        result["fixed_sample_global_step"],
        name="fixed sample global step",
        minimum=0,
    )
    sample_keys = result["frame_sample_keys_by_rank"]
    source_digests = result["frame_source_digests_by_rank"]
    if (
        not isinstance(sample_keys, list)
        or len(sample_keys) != 2
        or any(
            not isinstance(rank_frames, list)
            or len(rank_frames) != 2
            or any(not isinstance(key, str) or not key for key in rank_frames)
            for rank_frames in sample_keys
        )
    ):
        raise ValueError("fixed-batch sample keys must bind two frames on both ranks")
    if (
        not isinstance(source_digests, list)
        or len(source_digests) != 2
        or any(
            not isinstance(rank_frames, list)
            or len(rank_frames) != 2
            or any(
                _sha256(digest, name="fixed-batch source digest") != digest
                for digest in rank_frames
            )
            for rank_frames in source_digests
        )
    ):
        raise ValueError("fixed-batch source digests must bind two frames on both ranks")
    objective = result["objective"]
    expected_objective = {
        "optimized_family": "predictive",
        "target": "prior_to_current_object_summary",
        "window": "fixed_two_frame_local_bptt",
        "labels_are_loss_side_only": True,
    }
    if not isinstance(objective, Mapping) or dict(objective) != expected_objective:
        raise ValueError("fixed-batch probe must optimize only the predictive family")
    optimizer = result["optimizer"]
    expected_optimizer_fields = {
        "algorithm",
        "learning_rate_hex",
        "weight_decay_hex",
        "scheduler",
        "moe_load_balance_hook_enabled",
        "update_count",
    }
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != expected_optimizer_fields
        or optimizer.get("algorithm") != "lingbot_distributed_muon_with_adamw_fallback"
        or optimizer.get("scheduler") != "constant"
        or optimizer.get("moe_load_balance_hook_enabled") is not False
        or optimizer.get("update_count") != curve_point_count - 1
    ):
        raise ValueError("fixed-batch optimizer contract is not isolated")
    learning_rate_hex = optimizer["learning_rate_hex"]
    weight_decay_hex = optimizer["weight_decay_hex"]
    if not isinstance(learning_rate_hex, str) or not isinstance(weight_decay_hex, str):
        raise ValueError("fixed-batch optimizer scalars are malformed")
    try:
        learning_rate = float.fromhex(learning_rate_hex)
        weight_decay = float.fromhex(weight_decay_hex)
    except ValueError as error:
        raise ValueError("fixed-batch optimizer scalars are malformed") from error
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("fixed-batch optimizer scalars are outside their valid range")
    fixed_batch_probe_subject(result, curve_point_count=curve_point_count)
    return result


def _validate_shared_host_gradient_probe(value: object, *, arm: str) -> dict[str, Any] | None:
    host_trainable = arm in {"full_host", "shuffled_target"}
    if not host_trainable:
        if value is not None:
            raise ValueError("a frozen-host arm unexpectedly reports shared-host gradients")
        return None
    required = {
        "all_finite",
        "gradient_elements",
        "gradient_norms",
        "parameter_paths",
        "probe",
        "world_size",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("trainable-host arm omitted its exact shared-host gradient probe")
    if (
        value["all_finite"] is not True
        or value["probe"] != "lingbot.language_model.input_layernorm"
        or value["world_size"] != 2
    ):
        raise ValueError("shared-host gradient probe contract differs")
    depths = ("early", "middle", "late")
    elements = value["gradient_elements"]
    norms = value["gradient_norms"]
    paths = value["parameter_paths"]
    if not all(
        isinstance(mapping, Mapping) and set(mapping) == set(depths)
        for mapping in (elements, norms, paths)
    ):
        raise ValueError("shared-host gradient depths differ from the released contract")
    expected_layers = {"early": 0, "middle": 18, "late": 35}
    for depth in depths:
        if elements[depth] != 2560:
            raise ValueError("shared-host gradient width differs from the released contract")
        if (
            _finite(
                norms[depth],
                name=f"{depth} shared-host gradient norm",
                non_negative=True,
            )
            <= 0
        ):
            raise ValueError("predictive loss did not reach every shared-host depth")
        path = paths[depth]
        if not isinstance(path, str) or not path.endswith(
            f"layers.{expected_layers[depth]}.input_layernorm.weight"
        ):
            raise ValueError("shared-host gradient parameter path differs")
    return dict(value)


def validate_predictive_fixed_batch_arm_report(value: object) -> dict[str, Any]:
    """Validate one arm and recompute every within-report invariant."""

    if not isinstance(value, Mapping) or set(value) != _ARM_REPORT_FIELDS:
        raise ValueError("fixed-batch arm report fields differ from schema")
    if value["schema"] != PREDICTIVE_FIXED_BATCH_ARM_REPORT_SCHEMA or value["status"] != "PASS":
        raise ValueError("fixed-batch arm report did not complete")
    arm = value["arm"]
    if not isinstance(arm, str) or arm not in PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("fixed-batch report arm is unsupported")
    curve_points = _positive_integer(
        value["curve_point_count"],
        name="fixed-batch curve-point count",
        minimum=2,
    )
    optimizer_updates = _positive_integer(
        value["optimizer_update_count"],
        name="fixed-batch optimizer-update count",
    )
    if optimizer_updates != curve_points - 1:
        raise ValueError("fixed-batch optimizer updates must equal curve points minus one")
    provenance = _validate_provenance(
        value["provenance"],
        curve_point_count=curve_points,
    )
    subject = _sha256(value["subject_sha256"], name="fixed-batch subject")
    if subject != fixed_batch_probe_subject(
        provenance,
        curve_point_count=curve_points,
    ):
        raise ValueError("fixed-batch subject differs from its provenance")
    scope = _parse_trainable_scope(value["trainable_scope"], expected_arm=arm)
    global_loss = _curve(
        value["global_loss_curve"],
        name="global loss curve",
        steps=curve_points,
    )
    global_shuffle = _curve(
        value["global_shuffle_distance_curve"],
        name="global shuffle-distance curve",
        steps=curve_points,
    )
    rank_reports = value["rank_reports"]
    if not isinstance(rank_reports, list) or len(rank_reports) != 2:
        raise ValueError("fixed-batch report must contain exactly two rank reports")
    local_losses: list[tuple[float, ...]] = []
    local_shuffles: list[tuple[float, ...]] = []
    observed_ranks: list[int] = []
    sample_keys_by_rank = cast(list[list[str]], provenance["frame_sample_keys_by_rank"])
    source_digests_by_rank = cast(list[list[str]], provenance["frame_source_digests_by_rank"])
    for raw in rank_reports:
        required = {
            "rank",
            "frame_sample_keys",
            "frame_source_digests",
            "loss_curve",
            "shuffle_distance_curve",
            "step_times_s",
            "peak_reserved_bytes",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("fixed-batch rank report fields differ from schema")
        rank = raw["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (0, 1):
            raise ValueError("fixed-batch rank is invalid")
        observed_ranks.append(rank)
        if (
            raw["frame_sample_keys"] != sample_keys_by_rank[rank]
            or raw["frame_source_digests"] != source_digests_by_rank[rank]
        ):
            raise ValueError("fixed-batch rank data differs from common provenance")
        local_losses.append(_curve(raw["loss_curve"], name="rank loss curve", steps=curve_points))
        local_shuffles.append(
            _curve(
                raw["shuffle_distance_curve"],
                name="rank shuffle-distance curve",
                steps=curve_points,
            )
        )
        times = _curve(
            raw["step_times_s"],
            name="rank step times",
            steps=curve_points,
        )
        if any(value <= 0 for value in times):
            raise ValueError("fixed-batch step times must be positive")
        _positive_integer(
            raw["peak_reserved_bytes"],
            name="rank peak reserved bytes",
            minimum=0,
        )
    if observed_ranks != [0, 1]:
        raise ValueError("fixed-batch rank reports must use frozen rank order")
    for step in range(curve_points):
        expected_loss = sum(curve[step] for curve in local_losses) / 2
        expected_shuffle = sum(curve[step] for curve in local_shuffles) / 2
        if not math.isclose(global_loss[step], expected_loss, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("global fixed-batch loss differs from rank mean")
        if not math.isclose(
            global_shuffle[step],
            expected_shuffle,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError("global shuffle distance differs from rank mean")
    if arm == "shuffled_target":
        if (
            global_shuffle[0] != 0
            or not all(value > 0 for value in global_shuffle[1:])
            or any(
                curve[0] != 0 or not all(value > 0 for value in curve[1:])
                for curve in local_shuffles
            )
        ):
            raise ValueError(
                "shuffled-target arm did not execute a nontrivial negative control at every "
                "post-initial step on both ranks"
            )
    elif any(value != 0 for value in global_shuffle):
        raise ValueError("factual fixed-batch arm unexpectedly changed its target")
    shared_host_gradient_probe = _validate_shared_host_gradient_probe(
        value["shared_host_gradient_probe"],
        arm=arm,
    )
    if value["moe_routing_bias_unchanged"] is not True:
        raise ValueError("fixed-batch probe changed MoE routing bias")
    maximum_peak = _positive_integer(
        value["maximum_peak_reserved_bytes"],
        name="fixed-batch maximum peak bytes",
        minimum=0,
    )
    if maximum_peak != max(int(raw["peak_reserved_bytes"]) for raw in rank_reports):
        raise ValueError("fixed-batch maximum peak bytes differ from rank reports")
    if _finite(value["total_time_s"], name="fixed-batch total time", non_negative=True) <= 0:
        raise ValueError("fixed-batch total time must be positive")
    return {
        **dict(value),
        "provenance": provenance,
        "trainable_scope": scope.as_dict(),
        "shared_host_gradient_probe": shared_host_gradient_probe,
    }


def assemble_predictive_fixed_batch_experiment(
    reports: Mapping[str, Mapping[str, object]],
    *,
    report_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Bind four immutable arm reports without declaring scientific success."""

    if tuple(reports) != PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("fixed-batch reports must use frozen arm order")
    if tuple(report_sha256) != PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("fixed-batch report digests must use frozen arm order")
    validated: dict[str, dict[str, Any]] = {
        arm: validate_predictive_fixed_batch_arm_report(reports[arm])
        for arm in PREDICTIVE_FIXED_BATCH_ARMS
    }
    for arm in PREDICTIVE_FIXED_BATCH_ARMS:
        _sha256(report_sha256[arm], name=f"{arm} report digest")
        if validated[arm]["arm"] != arm:
            raise ValueError("fixed-batch arm report order differs")
    subjects = {str(value["subject_sha256"]) for value in validated.values()}
    if len(subjects) != 1:
        raise ValueError("fixed-batch arms do not share one experiment subject")
    curve_point_counts = {int(value["curve_point_count"]) for value in validated.values()}
    optimizer_update_counts = {int(value["optimizer_update_count"]) for value in validated.values()}
    if len(curve_point_counts) != 1 or len(optimizer_update_counts) != 1:
        raise ValueError("fixed-batch arms use different curve or optimization budgets")
    full_scope = cast(Mapping[str, Any], validated["full_host"]["trainable_scope"])
    shuffled_scope = cast(Mapping[str, Any], validated["shuffled_target"]["trainable_scope"])
    native_graph_scope = cast(
        Mapping[str, Any],
        validated["native_graph_only"]["trainable_scope"],
    )
    readout_scope = cast(Mapping[str, Any], validated["readout_only"]["trainable_scope"])
    if not all(
        isinstance(value, Mapping)
        for value in (
            full_scope,
            shuffled_scope,
            native_graph_scope,
            readout_scope,
        )
    ):
        raise TypeError("fixed-batch trainable scopes are malformed")
    if (
        full_scope["schema_sha256"] != shuffled_scope["schema_sha256"]
        or full_scope["trainable_numel"] != shuffled_scope["trainable_numel"]
    ):
        raise ValueError("full-host and shuffled-target trainability differ")
    scope_parameters: dict[str, dict[str, tuple[object, object, object]]] = {}
    for name, scope in (
        ("full_host", full_scope),
        ("native_graph_only", native_graph_scope),
        ("readout_only", readout_scope),
    ):
        scope_parameters[name] = {
            str(parameter["name"]): (
                parameter["shape"],
                parameter["dtype"],
                parameter["numel"],
            )
            for parameter in cast(list[Mapping[str, object]], scope["parameters"])
        }
    for smaller, larger in (
        ("native_graph_only", "full_host"),
        ("readout_only", "native_graph_only"),
    ):
        smaller_parameters = scope_parameters[smaller]
        larger_parameters = scope_parameters[larger]
        if not set(smaller_parameters) < set(larger_parameters) or any(
            larger_parameters[name] != descriptor for name, descriptor in smaller_parameters.items()
        ):
            raise ValueError(
                f"{smaller.replace('_', '-')} parameters are not an exact strict "
                f"subset of {larger.replace('_', '-')}"
            )
    curves = {
        arm: tuple(float(value) for value in validated[arm]["global_loss_curve"])
        for arm in PREDICTIVE_FIXED_BATCH_ARMS
    }
    initial = tuple(curves[arm][0] for arm in PREDICTIVE_FIXED_BATCH_ARMS)
    if any(not math.isclose(value, initial[0], rel_tol=1e-5, abs_tol=1e-6) for value in initial):
        raise ValueError("fixed-batch arms do not share one initial model/loss")
    diagnostics = predictive_fixed_batch_fit_diagnostics(curves)
    result = {
        "schema": PREDICTIVE_FIXED_BATCH_EXPERIMENT_REPORT_SCHEMA,
        "status": "PASS",
        "scientific_acceptance": "UNDECIDED_REQUIRES_OWNER_REVIEW",
        "subject_sha256": next(iter(subjects)),
        "arm_reports": [
            {
                "arm": arm,
                "sha256": report_sha256[arm],
            }
            for arm in PREDICTIVE_FIXED_BATCH_ARMS
        ],
        "diagnostics": diagnostics.as_dict(),
    }
    validate_predictive_fixed_batch_experiment_report(result)
    return result


def validate_predictive_fixed_batch_experiment_report(
    value: object,
) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "scientific_acceptance",
        "subject_sha256",
        "arm_reports",
        "diagnostics",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("fixed-batch experiment fields differ from schema")
    if (
        value["schema"] != PREDICTIVE_FIXED_BATCH_EXPERIMENT_REPORT_SCHEMA
        or value["status"] != "PASS"
        or value["scientific_acceptance"] != "UNDECIDED_REQUIRES_OWNER_REVIEW"
    ):
        raise ValueError("fixed-batch experiment status is invalid")
    _sha256(value["subject_sha256"], name="fixed-batch experiment subject")
    references = value["arm_reports"]
    if not isinstance(references, list) or len(references) != len(PREDICTIVE_FIXED_BATCH_ARMS):
        raise ValueError("fixed-batch experiment arm references are incomplete")
    for expected_arm, reference in zip(PREDICTIVE_FIXED_BATCH_ARMS, references, strict=True):
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"arm", "sha256"}
            or reference["arm"] != expected_arm
        ):
            raise ValueError("fixed-batch experiment arm reference is malformed")
        _sha256(reference["sha256"], name="fixed-batch arm report digest")
    diagnostics = predictive_fixed_batch_fit_from_mapping(value["diagnostics"])
    return {
        **dict(value),
        "diagnostics": diagnostics.as_dict(),
    }

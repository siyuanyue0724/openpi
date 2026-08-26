"""One production objective transaction for native CALVIN sequences."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_target_request import NativeCALVINStructuralTargetRequest
from picf_next.lingbot_native.calvin_supervision import (
    NativeCALVINSequenceTargetBundle,
    TaskIdentityResolver,
    build_native_calvin_sequence_target_bundle,
    stack_native_sequence_predictions,
)
from picf_next.lingbot_native.objective import (
    NativeObjectiveConfig,
    combine_native_objective,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveLossInput,
    materialize_native_predictive_terms,
)
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.row_binding import RowBindings
from picf_next.lingbot_native.supervision import (
    OWNERSHIP_ESTIMATORS,
    TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
    TOKEN_MICRO_OWNERSHIP,
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
    extend_sequence_row_bindings,
    match_sequence_rows,
    match_sequence_rows_with_bindings,
    sequence_set_terms,
)
from picf_next.lingbot_native.task_relation import (
    GLOBAL_MULTIPOSITIVE_TASK_RELATION,
    HOST_NATIVE_FACTORIZED_TASK_RELATION,
    HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION,
    LOCAL_BALANCED_TASK_RELATION,
    TASK_RELATION_ESTIMATORS,
    global_task_relation_term,
    host_native_factorized_task_relation_term,
    host_native_multi_positive_task_relation_term,
)
from picf_next.objective import ObjectiveTerm, UnifiedObjective

NativePredictiveInputFactory = Callable[
    [NativeCALVINSequenceTargetBundle],
    Sequence[NativePredictiveLossInput],
]


@dataclass(frozen=True, slots=True)
class NativeStructuralLossConfig:
    """Manifest-frozen component weights inside the structural family."""

    support_weight: float
    existence_weight: float
    task_weight: float
    dense_task_weight: float
    ownership_weight: float
    task_relation_estimator: str = LOCAL_BALANCED_TASK_RELATION
    ownership_estimator: str = TOKEN_MICRO_OWNERSHIP

    def __post_init__(self) -> None:
        for name, value in (
            ("support_weight", self.support_weight),
            ("existence_weight", self.existence_weight),
            ("task_weight", self.task_weight),
            ("dense_task_weight", self.dense_task_weight),
            ("ownership_weight", self.ownership_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.task_relation_estimator not in TASK_RELATION_ESTIMATORS:
            raise ValueError("unknown task relation estimator")
        if self.ownership_estimator not in OWNERSHIP_ESTIMATORS:
            raise ValueError("unknown ownership estimator")
        if self.task_relation_estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION and (
            self.dense_task_weight != 0 or self.ownership_weight <= 0 or self.task_weight <= 0
        ):
            raise ValueError(
                "factorized task relation derives grounding from task and ownership and "
                "requires zero independent dense-task weight plus positive task and ownership "
                "weights"
            )


@dataclass(frozen=True, slots=True)
class NativeCALVINObjectiveResult:
    """Auditable outputs of one label-only matching/objective transaction."""

    objective: UnifiedObjective
    predictions: NativeSequencePredictions
    targets: NativeSequenceTargets
    assignment: SequenceAssignment
    track_identity_keys_by_batch: tuple[tuple[str, ...], ...]
    row_bindings_by_batch: tuple[RowBindings, ...]
    predictive_terms: tuple[ObjectiveTerm, ...]
    structural_terms: tuple[ObjectiveTerm, ...]


def _validate_structural_term_schema(
    terms: Sequence[ObjectiveTerm],
    *,
    estimator: str,
    ownership_estimator: str,
    ownership_depth_count: int,
) -> None:
    """Fail before backward if an estimator is paired with another loss schema."""

    task_term = "set/task_row" if estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION else "set/task"
    expected = {"set/support", "set/existence", task_term, "set/task_dense"}
    for depth_index in range(ownership_depth_count):
        suffix = "" if depth_index == 0 else f"_q{depth_index}"
        expected.update({f"set/ownership{suffix}", f"set/ownership_nll{suffix}"})
        if ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP:
            expected.add(f"set/ownership_entity{suffix}")
    names = [term.name for term in terms]
    if len(set(names)) != len(names) or set(names) != expected:
        raise RuntimeError(
            f"{estimator} structural objective schema differs: "
            f"missing={sorted(expected - set(names))}, extra={sorted(set(names) - expected)}"
        )


def _validate_policy_loss(
    loss: torch.Tensor | None,
    *,
    action_weight: float,
    require_grad: bool,
) -> torch.Tensor | None:
    if not isinstance(require_grad, bool):
        raise TypeError("official policy-loss gradient requirement must be boolean")
    if action_weight == 0:
        if loss is not None:
            raise ValueError("an inactive action family requires an absent policy loss")
        return None
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not loss.is_floating_point()
        or not torch.isfinite(loss)
        or (require_grad and not loss.requires_grad)
    ):
        qualifier = "attached " if require_grad else ""
        raise ValueError(f"official policy loss must be one finite {qualifier}scalar")
    return loss


def compose_native_calvin_objective(
    *,
    official_policy_loss: torch.Tensor | None,
    requests_by_time: Sequence[Sequence[NativeCALVINStructuralTargetRequest]],
    model_inputs_by_time: Sequence[Mapping[str, Any]],
    relations: Sequence[RelationOutput],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    objective_config: NativeObjectiveConfig,
    structural_config: NativeStructuralLossConfig,
    require_policy_loss_grad: bool = True,
    predictive_inputs: Sequence[NativePredictiveLossInput] = (),
    predictive_input_factory: NativePredictiveInputFactory | None = None,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...] | None = None,
    intermediate_relations_by_layer: Mapping[int, Sequence[RelationOutput]] | None = None,
) -> NativeCALVINObjectiveResult:
    """Compose action, predictive and set losses after all model forwards."""

    time_count = len(relations)
    if not (len(requests_by_time) == len(model_inputs_by_time) == time_count):
        raise ValueError("native CALVIN objective inputs must share one time axis")
    if not isinstance(objective_config, NativeObjectiveConfig) or not isinstance(
        structural_config, NativeStructuralLossConfig
    ):
        raise TypeError("native CALVIN objective configs use frozen typed contracts")
    _validate_policy_loss(
        official_policy_loss,
        action_weight=objective_config.action_weight,
        require_grad=require_policy_loss_grad,
    )
    if predictive_inputs and predictive_input_factory is not None:
        raise ValueError("provide predictive inputs directly or through one factory, not both")
    if predictive_input_factory is not None and not callable(predictive_input_factory):
        raise TypeError("native predictive input factory must be callable")
    intermediate_relations = (
        {} if intermediate_relations_by_layer is None else intermediate_relations_by_layer
    )
    if not isinstance(intermediate_relations, Mapping):
        raise TypeError("intermediate relations must be keyed by host layer")
    intermediate_layers = tuple(intermediate_relations)
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in intermediate_layers
    ):
        raise ValueError("intermediate relation layers must be non-negative integers")
    if intermediate_layers != tuple(sorted(set(intermediate_layers))):
        raise ValueError("intermediate relation layers must be sorted and unique")
    for layer, depth_relations in intermediate_relations.items():
        if len(depth_relations) != time_count or any(
            not isinstance(relation, RelationOutput) for relation in depth_relations
        ):
            raise ValueError(
                f"intermediate relations at layer {layer} differ from the final time axis"
            )
    if objective_config.structural_weight > 0 and not any(
        value > 0
        for value in (
            structural_config.support_weight,
            structural_config.existence_weight,
            structural_config.task_weight,
            structural_config.dense_task_weight,
            structural_config.ownership_weight,
        )
    ):
        raise ValueError("a positive structural family weight requires a structural component")

    predictions = stack_native_sequence_predictions(relations)
    target_bundle: NativeCALVINSequenceTargetBundle = build_native_calvin_sequence_target_bundle(
        requests_by_time=requests_by_time,
        model_inputs_by_time=model_inputs_by_time,
        relations=relations,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=task_identity_resolver,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )
    pending_predictive = tuple(
        predictive_inputs
        if predictive_input_factory is None
        else predictive_input_factory(target_bundle)
    )
    if any(not isinstance(value, NativePredictiveLossInput) for value in pending_predictive):
        raise TypeError("native predictive losses must use unmatched typed inputs")
    if objective_config.predictive_weight > 0 and not pending_predictive:
        raise ValueError("a positive predictive family weight requires predictive inputs")
    targets = target_bundle.targets
    prior_bindings = (
        tuple(() for _ in target_bundle.identity_keys_by_batch)
        if prior_row_bindings_by_batch is None
        else prior_row_bindings_by_batch
    )
    assignment = (
        match_sequence_rows(predictions, targets)
        if prior_row_bindings_by_batch is None
        else match_sequence_rows_with_bindings(
            predictions,
            targets,
            identity_keys_by_batch=target_bundle.identity_keys_by_batch,
            prior_bindings_by_batch=prior_bindings,
        )
    )
    row_bindings = extend_sequence_row_bindings(
        assignment,
        targets,
        identity_keys_by_batch=target_bundle.identity_keys_by_batch,
        prior_bindings_by_batch=prior_bindings,
    )
    predictive_terms = materialize_native_predictive_terms(
        pending_predictive,
        assignment=assignment,
        expected_track_identity_keys=target_bundle.identity_keys_by_batch,
        sequence_time_count=time_count,
    )
    ownership_depth_count = len(intermediate_relations) + 1
    ownership_weight_per_depth = structural_config.ownership_weight / ownership_depth_count
    global_task_relation = (
        structural_config.task_relation_estimator == GLOBAL_MULTIPOSITIVE_TASK_RELATION
    )
    host_native_task_relation = (
        structural_config.task_relation_estimator == HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION
    )
    factorized_task_relation = (
        structural_config.task_relation_estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION
    )
    external_task_relation = (
        global_task_relation or host_native_task_relation or factorized_task_relation
    )
    structural_terms = sequence_set_terms(
        predictions,
        targets,
        assignment,
        support_weight=structural_config.support_weight,
        existence_weight=structural_config.existence_weight,
        task_weight=structural_config.task_weight,
        dense_task_weight=structural_config.dense_task_weight,
        ownership_weight=ownership_weight_per_depth,
        ownership_estimator=structural_config.ownership_estimator,
        include_task_term=not external_task_relation,
    )
    if global_task_relation:
        structural_terms = (
            *structural_terms[:2],
            global_task_relation_term(
                relations[-1],
                targets,
                assignment,
                identity_keys_by_batch=target_bundle.identity_keys_by_batch,
                weight=structural_config.task_weight,
            ),
            *structural_terms[2:],
        )
    elif host_native_task_relation:
        structural_terms = (
            *structural_terms[:2],
            host_native_multi_positive_task_relation_term(
                relations[-1],
                targets,
                assignment,
                weight=structural_config.task_weight,
            ),
            *structural_terms[2:],
        )
    elif factorized_task_relation:
        structural_terms = (
            *structural_terms[:2],
            host_native_factorized_task_relation_term(
                relations,
                targets,
                assignment,
                weight=structural_config.task_weight,
            ),
            *structural_terms[2:],
        )
    intermediate_ownership_terms: list[ObjectiveTerm] = []
    for depth_index, (_, depth_relations) in enumerate(
        intermediate_relations.items(),
        start=1,
    ):
        depth_predictions = stack_native_sequence_predictions(depth_relations)
        depth_terms = sequence_set_terms(
            depth_predictions,
            targets,
            assignment,
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=ownership_weight_per_depth,
            ownership_estimator=structural_config.ownership_estimator,
        )
        ownership_names = {"set/ownership", "set/ownership_nll"}
        if structural_config.ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP:
            ownership_names.add("set/ownership_entity")
        selected = {term.name: term for term in depth_terms if term.name in ownership_names}
        if set(selected) != ownership_names:
            raise RuntimeError("intermediate relation objective omitted ownership terms")
        for base_name in sorted(ownership_names):
            term = selected[base_name]
            intermediate_ownership_terms.append(
                ObjectiveTerm(
                    name=f"{base_name}_q{depth_index}",
                    values=term.values,
                    valid=term.valid,
                    weight=term.weight,
                    sample_weight=term.sample_weight,
                )
            )
    structural_terms = (*structural_terms, *intermediate_ownership_terms)
    _validate_structural_term_schema(
        structural_terms,
        estimator=structural_config.task_relation_estimator,
        ownership_estimator=structural_config.ownership_estimator,
        ownership_depth_count=ownership_depth_count,
    )
    objective = combine_native_objective(
        official_policy_loss=official_policy_loss,
        predictive_terms=predictive_terms,
        structural_terms=structural_terms,
        config=objective_config,
    )
    return NativeCALVINObjectiveResult(
        objective=objective,
        predictions=predictions,
        targets=targets,
        assignment=assignment,
        track_identity_keys_by_batch=target_bundle.identity_keys_by_batch,
        row_bindings_by_batch=row_bindings,
        predictive_terms=predictive_terms,
        structural_terms=structural_terms,
    )

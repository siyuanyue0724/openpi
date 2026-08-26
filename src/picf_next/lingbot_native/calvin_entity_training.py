"""One leak-closed CALVIN transaction for task-independent entity training."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.autograd.graph import save_on_cpu
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.lingbot_native.calvin import (
    CollatedNativeCALVINBatch,
    build_native_calvin_context,
)
from picf_next.lingbot_native.calvin_entity_set import (
    PhysicalCALVINFrameTargetBundle,
    build_task_independent_calvin_targets,
)
from picf_next.lingbot_native.current_grid_cache import LingBotCurrentGridTargetCache
from picf_next.lingbot_native.entity_training import (
    TaskIndependentEntityObjectiveConfig,
    TaskIndependentEntityObjectiveResult,
    TaskIndependentPersistentEntityObjectiveResult,
    compose_task_independent_entity_objective,
    compose_task_independent_persistent_entity_objective,
)
from picf_next.lingbot_native.full_training import (
    NativeCorrectionBranch,
    NativeCurrentGridBranch,
    NativeFilterPhaseBranch,
    NativeOmittedStaticBranch,
    _correction_loss_inputs,
    _correction_valid_by_time,
    _current_grid_loss_input,
    _filter_phase_loss_inputs,
    _native_primary_prediction_requests,
    _omitted_static_loss_input,
    _validate_prior_bindings_have_valid_source,
    make_native_current_filter_request,
    make_native_current_grid_request,
    make_native_omitted_static_request,
    omitted_static_target,
)
from picf_next.lingbot_native.future_latent_alignment import (
    FutureLatentTargetBatch,
    future_latent_objective_contribution,
)
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    native_context_from_persistent_state,
)
from picf_next.lingbot_native.objective import (
    NativeObjectiveConfig,
    NativePredictiveNormalizationLedger,
    build_native_predictive_normalization_ledger,
    combine_native_objective,
    combine_native_sequential_branch,
    merge_repeated_objective_terms,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.physical_sequence import PhysicalSequenceAssignment
from picf_next.lingbot_native.prediction import NativePredictionRequest, PredictionSource
from picf_next.lingbot_native.predictive_cache import (
    LINGBOT_PREDICTIVE_TARGET_SPACE,
    LingBotPredictiveTargetCache,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveLossInput,
    NativePredictiveTarget,
    materialize_native_predictive_support,
    materialize_native_predictive_terms,
)
from picf_next.lingbot_native.row_binding import RowBindings
from picf_next.lingbot_native.source_mask import QwenPackedPatchMask, QwenWholeViewOmission
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePersistentState,
    NativePosteriorState,
)
from picf_next.lingbot_native.temporal import NativePriorPredictiveRollout
from picf_next.lingbot_native.training import (
    NativeLocalBPTTAuxiliary,
    NativeLocalBPTTStep,
    NativePolicyForwardResult,
    NativeV3AttachedEgressResult,
    NativeV3FilterPredictionSpec,
    NativeV3OmittedStaticPolicyResult,
    NativeV3TwoPassStep,
    native_persistent_output,
    native_policy_uses_v3_two_pass,
    run_native_local_bptt,
    run_native_omitted_image_view_training_forward,
    run_native_policy_diagnostic_forward,
    run_native_policy_representation_training_forward,
    run_native_policy_training_forward,
    run_native_representation_window,
    run_native_source_masked_training_forward,
    run_native_v3_attached_egress,
    run_native_v3_omitted_static_view_policy_training_forward,
    run_native_v3_two_pass_sequence,
)
from picf_next.objective import ObjectiveTerm, UnifiedObjective

TaskIndependentPredictiveRolloutFactory = Callable[
    [NativePosteriorState], NativePriorPredictiveRollout
]
TaskIndependentPredictiveInputFactory = Callable[
    [tuple[tuple[str, ...], ...]], Sequence[NativePredictiveLossInput]
]
OmittedStaticCheckpointContextFactory = Callable[
    [],
    tuple[AbstractContextManager[None], AbstractContextManager[None]],
]
OmittedStaticForwardContextFactory = Callable[[], AbstractContextManager[None]]

OMITTED_STATIC_REMATERIALIZATION_NONE = "none"
OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT = "complete-checkpoint"
OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU = "save-on-cpu"
OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD = "sequential-backward"
OMITTED_STATIC_REMATERIALIZATION_MODES = (
    OMITTED_STATIC_REMATERIALIZATION_NONE,
    OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT,
    OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU,
    OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD,
)


def _noop_checkpoint_contexts() -> tuple[
    AbstractContextManager[None],
    AbstractContextManager[None],
]:
    return nullcontext(), nullcontext()


@dataclass(frozen=True, slots=True)
class TaskIndependentCALVINFrameStepResult:
    """Outputs from one official-host, loss-side-supervised frame transaction."""

    context: LingBotNativeContext
    relation: PhysicalRelationOutput
    targets: PhysicalCALVINFrameTargetBundle
    objective: TaskIndependentEntityObjectiveResult
    policy_forward: NativePolicyForwardResult | None


@dataclass(frozen=True, slots=True)
class TaskIndependentCALVINSequenceStepResult:
    """One recurrent physical window whose first posterior may advance a lane."""

    contexts: tuple[LingBotNativeContext, ...]
    relations: tuple[PhysicalRelationOutput, ...]
    targets: tuple[PhysicalCALVINFrameTargetBundle, ...]
    objective: TaskIndependentPersistentEntityObjectiveResult
    diagnostic_action_loss: torch.Tensor | None = None

    @property
    def committable_context(self) -> LingBotNativeContext:
        return self.contexts[0]


@dataclass(frozen=True, slots=True)
class TaskIndependentCALVINJointSequenceStepResult:
    """One action-bearing primary frame plus zero to three loss-only frames."""

    primary: NativePolicyForwardResult
    auxiliary: tuple[NativeLocalBPTTAuxiliary, ...]
    relations: tuple[PhysicalRelationOutput, ...]
    targets: tuple[PhysicalCALVINFrameTargetBundle, ...]
    objective: TaskIndependentPersistentEntityObjectiveResult
    correction_branches: tuple[NativeCorrectionBranch, ...] = ()
    filter_phase_branches: tuple[NativeFilterPhaseBranch, ...] = ()
    current_grid_branch: NativeCurrentGridBranch | None = None
    omitted_static_branch: NativeOmittedStaticBranch | None = None
    omitted_static_policy: NativeV3OmittedStaticPolicyResult | None = None
    attached_egress: NativeV3AttachedEgressResult | None = None
    v3_filter_specs: tuple[NativeV3FilterPredictionSpec | None, ...] = ()
    v3_prior_traces: tuple[NativeLayerwisePriorTrace, ...] = ()
    omitted_static_rematerialization: str = OMITTED_STATIC_REMATERIALIZATION_NONE
    sequential_omitted_static: TaskIndependentCALVINSequentialOmittedPlan | None = None

    @property
    def committable_context(self) -> LingBotNativeContext:
        return self.primary.context


@dataclass(frozen=True, slots=True)
class TaskIndependentCALVINSequentialOmittedPlan:
    """Frozen loss/target contract for one deferred omitted-static branch."""

    batch: CollatedNativeCALVINBatch
    request: NativePredictionRequest
    omission: QwenWholeViewOmission
    target: NativePredictiveTarget
    assignment: PhysicalSequenceAssignment
    track_identity_keys: tuple[tuple[str, ...], ...]
    sequence_time_count: int
    identity_source_phase: int
    predictive_ledger: NativePredictiveNormalizationLedger
    factual_backward_loss: torch.Tensor
    factual_predictive_terms: tuple[ObjectiveTerm, ...]
    structural_terms: tuple[ObjectiveTerm, ...]
    objective_config: TaskIndependentEntityObjectiveConfig
    omitted_static_weight: float
    predictive_loss_power: float
    posterior_adoption_route: torch.Tensor | None
    supervise_intermediate_relations: bool


@dataclass(frozen=True, slots=True)
class TaskIndependentCALVINSequentialOmittedResult:
    """Second forward and additive loss for one sequential omitted branch."""

    policy: NativeV3OmittedStaticPolicyResult
    branch: NativeOmittedStaticBranch
    predictive_terms: tuple[ObjectiveTerm, ...]
    backward_loss: torch.Tensor


def _detached_unified_objective(objective: UnifiedObjective) -> UnifiedObjective:
    """Keep metrics without retaining a second reduction graph for one update."""

    return UnifiedObjective(
        total=objective.total.detach(),
        normalized_terms={
            name: value.detach() for name, value in objective.normalized_terms.items()
        },
        valid_counts=dict(objective.valid_counts),
        family_terms={name: value.detach() for name, value in objective.family_terms.items()},
    )


def _validate_task_independent_sequence_inputs(
    *,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
) -> None:
    if not 1 <= len(batches) <= 4:
        raise ValueError("task-independent physical sequence requires 1..4 frames")
    if any(not isinstance(batch, CollatedNativeCALVINBatch) for batch in batches):
        raise TypeError("task-independent physical sequence requires collated batches")
    batch_size = batches[0].routing.batch_size
    if len(prior_row_bindings_by_batch) != batch_size:
        raise ValueError("physical prior row bindings differ from the sequence batch")
    first_routing = batches[0].routing
    for time_index, batch in enumerate(batches):
        if (
            batch.routing.batch_size != batch_size
            or batch.routing.lane_ids != first_routing.lane_ids
            or batch.routing.episode_keys != first_routing.episode_keys
            or batch.routing.optimizer_step != first_routing.optimizer_step
        ):
            raise ValueError("one physical sequence must retain lane and episode identity")
        expected_frames = tuple(value + time_index for value in first_routing.frame_indices)
        if batch.routing.frame_indices != expected_frames:
            raise ValueError("physical sequence frames must be contiguous")
        if time_index and any(batch.routing.reset):
            raise ValueError("a physical sequence cannot cross an episode reset")

    device = batches[0].controls.values.device
    source_valid = (
        torch.full(
            (batch_size,),
            previous_state is not None,
            dtype=torch.bool,
            device=device,
        )
        if previous_state_valid is None
        else previous_state_valid
    )
    if source_valid.shape != (batch_size,) or source_valid.dtype != torch.bool:
        raise ValueError("physical previous-state validity must be boolean [batch]")
    if source_valid.device != device:
        raise ValueError("physical previous-state validity and batches must share one device")
    if previous_state is None and source_valid.any():
        raise ValueError("an absent previous physical state cannot be valid")
    for batch_index, bindings in enumerate(prior_row_bindings_by_batch):
        if bindings and first_routing.reset[batch_index]:
            raise ValueError("episode reset must clear the physical row gauge")
        if bindings and not bool(source_valid[batch_index].item()):
            raise ValueError("physical prior bindings require a valid recurrent source")


def _current_static_observation_valid(
    batch: CollatedNativeCALVINBatch,
) -> torch.Tensor:
    """Return current static-view availability independently of recurrent validity."""

    image_valid = batch.model_inputs.get("img_masks")
    if not isinstance(image_valid, torch.Tensor):
        raise TypeError("v3 posterior filtering requires tensor img_masks")
    if (
        image_valid.ndim != 2
        or image_valid.shape[0] != batch.routing.batch_size
        or image_valid.shape[1] < 1
        or image_valid.dtype != torch.bool
    ):
        raise ValueError("v3 posterior img_masks must be boolean [batch, views>=1]")
    if image_valid.device != batch.controls.values.device:
        raise ValueError("v3 posterior image validity and controls must share one device")
    return image_valid[:, 0]


def _compose_task_independent_sequence_loss(
    *,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    relations: tuple[PhysicalRelationOutput, ...],
    committable_state: NativePersistentState,
    official_policy_loss: torch.Tensor | None,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
    predictive_rollout_factory: TaskIndependentPredictiveRolloutFactory | None,
    predictive_cache: LingBotPredictiveTargetCache | None,
    predictive_loss_power: float,
    minimum_supervised_fraction: float,
    capacity_seeds: Sequence[int | None] | None,
    predictive_input_factory: TaskIndependentPredictiveInputFactory | None = None,
) -> tuple[
    tuple[PhysicalCALVINFrameTargetBundle, ...],
    TaskIndependentPersistentEntityObjectiveResult,
]:
    predictive_active = objective_config.predictive_weight > 0
    rollout_active = predictive_cache is not None or predictive_rollout_factory is not None
    if (predictive_cache is None) != (predictive_rollout_factory is None):
        raise ValueError("physical rollout and target cache must be active together")
    if predictive_input_factory is not None and not callable(predictive_input_factory):
        raise TypeError("physical predictive input factory must be callable")
    if predictive_active != bool(rollout_active or predictive_input_factory is not None):
        raise ValueError("physical predictive family and its attached branches differ")
    if (
        isinstance(predictive_loss_power, bool)
        or not isinstance(predictive_loss_power, (int, float))
        or not math.isfinite(predictive_loss_power)
        or predictive_loss_power < 1
    ):
        raise ValueError("physical predictive loss power must be finite and at least one")

    capacity = relations[0].support_logits.shape[-1]
    target_bundles = build_task_independent_calvin_targets(
        requests_by_time=tuple(batch.structural_target_requests for batch in batches),
        model_inputs_by_time=tuple(batch.model_inputs for batch in batches),
        relations=relations,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )
    identity_keys = target_bundles[0].identity_keys_by_batch
    if any(bundle.identity_keys_by_batch != identity_keys for bundle in target_bundles[1:]):
        raise RuntimeError("physical target identity axis changed inside one sequence")

    predictive_inputs = (
        [] if predictive_input_factory is None else list(predictive_input_factory(identity_keys))
    )
    if any(not isinstance(value, NativePredictiveLossInput) for value in predictive_inputs):
        raise TypeError("additional physical predictions require typed loss inputs")
    if rollout_active:
        if predictive_cache is None or predictive_rollout_factory is None:
            raise RuntimeError("active predictive rollout lost its cache or rollout factory")
        if not isinstance(committable_state, NativePosteriorState):
            raise ValueError("row-only predictive rollout is unavailable for layerwise state")
        predictive_rollout = predictive_rollout_factory(committable_state)
        if not isinstance(predictive_rollout, NativePriorPredictiveRollout):
            raise TypeError("physical predictive rollout factory returned the wrong type")
        if predictive_rollout.target_name != LINGBOT_PREDICTIVE_TARGET_SPACE:
            raise ValueError("physical predictive rollout uses the wrong target space")
        source_batch = batches[0]
        target = predictive_cache.target_for(
            source_global_indices=tuple(
                value.source_global_index for value in source_batch.structural_target_requests
            ),
            source_rgb_sha256=tuple(
                value.source_sensor_hash_by_field["rgb_static"]
                for value in source_batch.structural_target_requests
            ),
            track_identity_keys=identity_keys,
            request=predictive_rollout.request,
            device=predictive_rollout.prediction.device,
        )
        predictive_inputs.append(
            NativePredictiveLossInput(
                prediction=predictive_rollout.prediction,
                request=predictive_rollout.request,
                target=target,
                weight=1.0,
                identity_source_phase=1,
                loss_power=float(predictive_loss_power),
            )
        )
    objective = compose_task_independent_persistent_entity_objective(
        official_policy_loss=official_policy_loss,
        relations=relations,
        targets=tuple(bundle.targets for bundle in target_bundles),
        identity_keys_by_batch=identity_keys,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        config=objective_config,
        predictive_inputs=tuple(predictive_inputs),
    )
    return target_bundles, objective


def _run_task_independent_calvin_context_objective(
    policy: nn.Module,
    *,
    batch: CollatedNativeCALVINBatch,
    context: LingBotNativeContext,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    action_attention_callback: Callable[..., object] | None = None,
    diagnostic: bool = False,
) -> TaskIndependentCALVINFrameStepResult:
    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("task-independent CALVIN training requires a collated batch")
    if not isinstance(context, LingBotNativeContext):
        raise TypeError("task-independent CALVIN training requires one native context")
    if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
        raise TypeError("task-independent CALVIN training requires the audited sidecar")
    if not isinstance(objective_config, TaskIndependentEntityObjectiveConfig):
        raise TypeError("task-independent CALVIN training requires its typed objective config")
    if not isinstance(diagnostic, bool):
        raise TypeError("task-independent CALVIN diagnostic flag must be boolean")

    policy_forward: NativePolicyForwardResult | None
    if diagnostic:
        if objective_config.action_weight > 0:
            raise ValueError("task-independent diagnostic cannot score the action suffix")
        policy_forward = run_native_policy_diagnostic_forward(
            policy,
            model_inputs=batch.model_inputs,
            context=context,
            action_attention_callback=action_attention_callback,
        )
        context = policy_forward.context
        official_policy_loss = None
    elif objective_config.action_weight > 0:
        policy_forward = run_native_policy_training_forward(
            policy,
            model_inputs=batch.model_inputs,
            context=context,
            action_attention_callback=action_attention_callback,
        )
        context = policy_forward.context
        official_policy_loss = policy_forward.official_total_loss
    else:
        policy_forward = None
        context = run_native_policy_representation_training_forward(
            policy,
            model_inputs=batch.model_inputs,
            context=context,
        )
        official_policy_loss = None

    relation = context.relation_output
    if not isinstance(relation, PhysicalRelationOutput):
        raise TypeError("CALVIN entity training requires the task-independent graph ABI")
    capacity = relation.support_logits.shape[-1]
    target_bundle = build_task_independent_calvin_targets(
        requests_by_time=(batch.structural_target_requests,),
        model_inputs_by_time=(batch.model_inputs,),
        relations=(relation,),
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )[0]
    objective = compose_task_independent_entity_objective(
        official_policy_loss=official_policy_loss,
        relations=(relation,),
        targets=(target_bundle.targets,),
        config=objective_config,
    )
    return TaskIndependentCALVINFrameStepResult(
        context=context,
        relation=relation,
        targets=target_bundle,
        objective=objective,
        policy_forward=policy_forward,
    )


def _task_independent_current_frame_context(
    batch: CollatedNativeCALVINBatch,
) -> LingBotNativeContext:
    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("task-independent CALVIN training requires a collated batch")
    return LingBotNativeContext(
        controls=batch.controls,
        previous_state=None,
        previous_state_valid=torch.zeros(
            batch.routing.batch_size,
            dtype=torch.bool,
            device=batch.controls.values.device,
        ),
        modalities=batch.modalities,
    )


def run_task_independent_calvin_current_frame_objective(
    policy: nn.Module,
    *,
    batch: CollatedNativeCALVINBatch,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    action_attention_callback: Callable[..., object] | None = None,
) -> TaskIndependentCALVINFrameStepResult:
    """Run P1 from discovery queries with no previous posterior input.

    The exact current observation and acknowledged executed control remain
    available. Only recurrent posterior content is withheld, so P1 measures
    current-frame entity discovery rather than temporal carry-over.
    """

    context = _task_independent_current_frame_context(batch)
    return _run_task_independent_calvin_context_objective(
        policy,
        batch=batch,
        context=context,
        physical_sidecar=physical_sidecar,
        objective_config=objective_config,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        action_attention_callback=action_attention_callback,
    )


@torch.no_grad()
def run_task_independent_calvin_current_frame_diagnostic(
    policy: nn.Module,
    *,
    batch: CollatedNativeCALVINBatch,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    action_attention_callback: Callable[..., object] | None = None,
) -> TaskIndependentCALVINFrameStepResult:
    """Evaluate current-frame entities through the official no-grad host root."""

    context = _task_independent_current_frame_context(batch)
    return _run_task_independent_calvin_context_objective(
        policy,
        batch=batch,
        context=context,
        physical_sidecar=physical_sidecar,
        objective_config=objective_config,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        action_attention_callback=action_attention_callback,
        diagnostic=True,
    )


@torch.no_grad()
def run_task_independent_calvin_recurrent_frame_diagnostic(
    policy: nn.Module,
    *,
    batch: CollatedNativeCALVINBatch,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    previous_state: NativePersistentState,
    previous_state_valid: torch.Tensor,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
) -> TaskIndependentCALVINSequenceStepResult:
    """Evaluate one recurrent frame through the exact no-grad observation host."""

    if objective_config.action_weight != 0:
        raise ValueError("recurrent entity diagnostics cannot execute the action suffix")
    if objective_config.predictive_weight != 0:
        raise ValueError("recurrent entity diagnostics cannot execute predictive branches")
    _validate_task_independent_sequence_inputs(
        batches=(batch,),
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
    )
    policy_forward = run_native_policy_diagnostic_forward(
        policy,
        model_inputs=batch.model_inputs,
        context=native_context_from_persistent_state(
            controls=batch.controls,
            persistent_state=previous_state,
            persistent_state_valid=previous_state_valid,
            modalities=batch.modalities,
        ),
    )
    context = policy_forward.context
    relation = context.relation_output
    if not isinstance(relation, PhysicalRelationOutput):
        raise TypeError("recurrent entity diagnostic requires the task-independent graph ABI")
    target_bundle = build_task_independent_calvin_targets(
        requests_by_time=(batch.structural_target_requests,),
        model_inputs_by_time=(batch.model_inputs,),
        relations=(relation,),
        physical_sidecar=physical_sidecar,
        capacity=relation.support_logits.shape[-1],
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )[0]
    objective = compose_task_independent_persistent_entity_objective(
        official_policy_loss=None,
        relations=(relation,),
        targets=(target_bundle.targets,),
        identity_keys_by_batch=target_bundle.identity_keys_by_batch,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        config=objective_config,
    )
    return TaskIndependentCALVINSequenceStepResult(
        contexts=(context,),
        relations=(relation,),
        targets=(target_bundle,),
        objective=objective,
        diagnostic_action_loss=policy_forward.official_action_loss,
    )


def run_task_independent_calvin_frame_objective(
    policy: nn.Module,
    *,
    batch: CollatedNativeCALVINBatch,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    previous_state: NativePersistentState | None = None,
    previous_state_valid: torch.Tensor | None = None,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
) -> TaskIndependentCALVINFrameStepResult:
    """Run a recurrent frame before constructing any loss-side target.

    P2/P3 use the released observation or policy root with exact causal state.
    The physical sidecar remains inaccessible until the deploy-visible host
    output is finalized.
    """

    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("task-independent CALVIN training requires a collated batch")
    context = build_native_calvin_context(
        batch,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
    )
    return _run_task_independent_calvin_context_objective(
        policy,
        batch=batch,
        context=context,
        physical_sidecar=physical_sidecar,
        objective_config=objective_config,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )


def run_task_independent_calvin_sequence_objective(
    policy: nn.Module,
    *,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
    predictive_rollout_factory: TaskIndependentPredictiveRolloutFactory | None = None,
    predictive_cache: LingBotPredictiveTargetCache | None = None,
    predictive_loss_power: float = 1.0,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
) -> TaskIndependentCALVINSequenceStepResult:
    """Run a 1..4-frame recurrent physical window before reading sidecar labels."""

    if objective_config.action_weight != 0:
        raise ValueError("the P2 physical sequence root is action-free; P3 owns joint action")
    _validate_task_independent_sequence_inputs(
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
    )

    window = run_native_representation_window(
        policy,
        steps=tuple(
            NativeLocalBPTTStep(
                model_inputs=batch.model_inputs,
                controls=batch.controls,
                modalities=batch.modalities,
            )
            for batch in batches
        ),
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
    )
    relations: list[PhysicalRelationOutput] = []
    for context in window.contexts:
        relation = context.relation_output
        if not isinstance(relation, PhysicalRelationOutput):
            raise TypeError("physical sequence requires the task-independent graph ABI")
        relations.append(relation)
    relation_axis = tuple(relations)
    committable_state = native_persistent_output(window.contexts[0])
    target_bundles, objective = _compose_task_independent_sequence_loss(
        batches=batches,
        relations=relation_axis,
        committable_state=committable_state,
        official_policy_loss=None,
        physical_sidecar=physical_sidecar,
        objective_config=objective_config,
        patch_size=patch_size,
        merge_size=merge_size,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        predictive_rollout_factory=predictive_rollout_factory,
        predictive_cache=predictive_cache,
        predictive_loss_power=predictive_loss_power,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )
    return TaskIndependentCALVINSequenceStepResult(
        contexts=window.contexts,
        relations=relation_axis,
        targets=target_bundles,
        objective=objective,
    )


def run_task_independent_calvin_joint_sequence_objective(
    policy: nn.Module,
    *,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    objective_config: TaskIndependentEntityObjectiveConfig,
    patch_size: int,
    merge_size: int,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
    graph: LingBotNativeGraph | None = None,
    current_grid_cache: LingBotCurrentGridTargetCache | None = None,
    source_mask: QwenPackedPatchMask | None = None,
    omitted_static_view: QwenWholeViewOmission | None = None,
    posterior_adoption_route: torch.Tensor | None = None,
    factual_posterior_adoption_route: torch.Tensor | None = None,
    factual_action_attention_callback: Callable[..., object] | None = None,
    future_latent_target: FutureLatentTargetBatch | None = None,
    future_latent_objective_scale: float = 1.0,
    egress_batch: CollatedNativeCALVINBatch | None = None,
    predictive_rollout_factory: TaskIndependentPredictiveRolloutFactory | None = None,
    predictive_cache: LingBotPredictiveTargetCache | None = None,
    correction_weight: float = 1.0,
    current_grid_weight: float = 1.0,
    omitted_static_weight: float = 1.0,
    predictive_minimum_visible_fraction: float = 0.0,
    predictive_loss_power: float = 1.0,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    prior_host_steps_by_batch: Sequence[int] | None = None,
    prior_gradient_suffix_steps_by_batch: Sequence[int] | None = None,
    egress_prior_host_steps: int | None = None,
    omitted_static_rematerialization: str = OMITTED_STATIC_REMATERIALIZATION_NONE,
    omitted_static_checkpoint_context_fn: OmittedStaticCheckpointContextFactory | None = None,
    omitted_static_forward_context_fn: OmittedStaticForwardContextFactory | None = None,
) -> TaskIndependentCALVINJointSequenceStepResult:
    """Train action, physical entities and sparse temporal credit in one host graph.

    The first frame executes the complete official LingBot policy suffix. Later
    frames, when sampled, reuse the same LingBot host and posterior transition
    but remain action-free. The sidecar and predictive cache are read only after
    every deploy-visible host forward has completed.
    """

    if objective_config.action_weight <= 0:
        raise ValueError("the joint physical sequence requires an active action family")
    if omitted_static_rematerialization not in OMITTED_STATIC_REMATERIALIZATION_MODES:
        raise ValueError("omitted-static rematerialization mode is unsupported")
    if (
        omitted_static_checkpoint_context_fn is not None
        and omitted_static_rematerialization
        != OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT
    ):
        raise ValueError(
            "an omitted-static checkpoint context requires complete-checkpoint rematerialization"
        )
    if (
        omitted_static_forward_context_fn is not None
        and omitted_static_rematerialization != OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU
    ):
        raise ValueError(
            "an omitted-static forward context requires save-on-cpu activation offload"
        )
    _validate_task_independent_sequence_inputs(
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
    )
    v3_active = native_policy_uses_v3_two_pass(policy)
    if prior_host_steps_by_batch is not None:
        if not v3_active:
            raise ValueError("a prior host schedule requires the v3 two-pass graph")
        if len(prior_host_steps_by_batch) != len(batches):
            raise ValueError("prior host schedule and v3 sequence have different lengths")
    if prior_gradient_suffix_steps_by_batch is not None:
        if not v3_active:
            raise ValueError("a prior gradient suffix requires the v3 two-pass graph")
        if len(prior_gradient_suffix_steps_by_batch) != len(batches):
            raise ValueError(
                "prior gradient suffix schedule and v3 sequence have different lengths"
            )
    if egress_prior_host_steps is not None and not v3_active:
        raise ValueError("an egress prior host schedule requires the v3 two-pass graph")
    if egress_prior_host_steps is not None and egress_batch is None:
        raise ValueError("an egress prior host schedule requires an egress batch")
    correction_active = current_grid_cache is not None
    if correction_active != (graph is not None):
        raise ValueError("current correction requires both the graph and frozen grid cache")
    if source_mask is not None and omitted_static_view is not None:
        raise ValueError("current-grid and omitted-static branches are mutually exclusive")
    if (source_mask is not None or omitted_static_view is not None) and not correction_active:
        raise ValueError("a physical source mask requires the frozen current-grid branch")
    if posterior_adoption_route is not None:
        if omitted_static_view is None or not v3_active:
            raise ValueError(
                "posterior-adoption routing requires the v3 omitted-static action branch"
            )
        if (
            posterior_adoption_route.shape != (batches[0].routing.batch_size,)
            or posterior_adoption_route.dtype != torch.bool
            or posterior_adoption_route.device != batches[0].controls.values.device
        ):
            raise ValueError("posterior-adoption routing must be boolean [batch]")
    if factual_posterior_adoption_route is not None:
        if not v3_active:
            raise ValueError("factual posterior-adoption routing requires the v3 graph")
        if (
            factual_posterior_adoption_route.shape
            != (batches[0].routing.batch_size,)
            or factual_posterior_adoption_route.dtype != torch.bool
            or factual_posterior_adoption_route.device
            != batches[0].controls.values.device
        ):
            raise ValueError(
                "factual posterior-adoption routing must be boolean [batch]"
            )
    if factual_action_attention_callback is not None and not v3_active:
        raise ValueError("factual action-attention collection requires the v3 graph")
    if v3_active and source_mask is not None:
        raise ValueError("v3 source masking requires a prior-trace action branch")
    if egress_batch is not None:
        if not v3_active:
            raise ValueError("attached egress is only available to the v3 two-pass graph")
        if not correction_active:
            raise ValueError("attached egress requires the frozen current-filter cache")
        if len(batches) != 1:
            raise ValueError(
                "production attached egress cannot accompany a full-frame probe window"
            )
        if not isinstance(egress_batch, CollatedNativeCALVINBatch):
            raise TypeError("attached egress requires one typed next-frame CALVIN batch")
        current_routing = batches[0].routing
        egress_routing = egress_batch.routing
        if (
            egress_routing.batch_size != current_routing.batch_size
            or egress_routing.lane_ids != current_routing.lane_ids
            or egress_routing.episode_keys != current_routing.episode_keys
            or egress_routing.optimizer_step != current_routing.optimizer_step
            or egress_routing.frame_indices
            != tuple(value + 1 for value in current_routing.frame_indices)
        ):
            raise ValueError("attached egress batch is not the contiguous next physical frame")
        if any(egress_routing.reset) or bool(egress_batch.prior_control_reset.any().item()):
            raise ValueError("attached egress cannot cross an episode reset")

    requests = tuple(None for _batch in batches)
    filter_specs: tuple[NativeV3FilterPredictionSpec | None, ...] = tuple(
        None for _batch in batches
    )
    if correction_active:
        if graph is None or current_grid_cache is None:
            raise RuntimeError("active current filter lost its graph or frozen target cache")
        if v3_active != graph.unified_predict_correct:
            raise ValueError("current-filter graph differs from the installed policy architecture")
        correction_valid = _correction_valid_by_time(
            batches=batches,
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
        )
        _validate_prior_bindings_have_valid_source(
            prior_row_bindings_by_batch=prior_row_bindings_by_batch,
            first_prior_valid=correction_valid[0],
        )
        if v3_active:
            filter_specs = tuple(
                NativeV3FilterPredictionSpec(
                    prior_request=make_native_current_filter_request(
                        source=PredictionSource.PRIOR,
                        batch_size=batch.routing.batch_size,
                        valid=valid,
                        device=batch.controls.values.device,
                        dtype=batch.controls.values.dtype,
                        route_id=current_grid_cache.contract.route_id,
                        address_width=graph.config.prediction_address_width,
                    ),
                    posterior_request=make_native_current_filter_request(
                        source=PredictionSource.POSTERIOR,
                        batch_size=batch.routing.batch_size,
                        valid=_current_static_observation_valid(batch),
                        device=batch.controls.values.device,
                        dtype=batch.controls.values.dtype,
                        route_id=current_grid_cache.contract.route_id,
                        address_width=graph.config.prediction_address_width,
                    ),
                    target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
                )
                for batch, valid in zip(batches, correction_valid, strict=True)
            )
        else:
            requests = _native_primary_prediction_requests(
                graph=graph,
                batches=batches,
                correction_valid=correction_valid,
                route_id=current_grid_cache.contract.route_id,
                behavior_conditioned=False,
            )

    v3_prior_traces: tuple[NativeLayerwisePriorTrace, ...] = ()
    v3_filter_predictions = ()
    if v3_active:
        if previous_state_valid is None:
            raise TypeError("v3 two-pass CALVIN training requires explicit previous-state validity")
        if previous_state is not None and not isinstance(
            previous_state,
            NativeLayerwisePosteriorState,
        ):
            raise TypeError("v3 two-pass CALVIN training requires layerwise posterior memory")
        v3_window = run_native_v3_two_pass_sequence(
            policy,
            steps=tuple(
                NativeV3TwoPassStep(
                    model_inputs=batch.model_inputs,
                    controls=batch.controls,
                    filter_prediction=spec,
                    modalities=batch.modalities,
                    prior_control_chunks=batch.prior_control_chunks,
                    prior_host_steps=(
                        None
                        if prior_host_steps_by_batch is None
                        else prior_host_steps_by_batch[index]
                    ),
                    prior_gradient_suffix_steps=(
                        None
                        if prior_gradient_suffix_steps_by_batch is None
                        else prior_gradient_suffix_steps_by_batch[index]
                    ),
                    future_latent_target=(future_latent_target if index == 0 else None),
                    wla_world_target=(batch.wla_world_target if index == 0 else None),
                )
                for index, (batch, spec) in enumerate(zip(batches, filter_specs, strict=True))
            ),
            previous_memory=previous_state,
            previous_memory_valid=previous_state_valid,
            posterior_adoption_route=factual_posterior_adoption_route,
            action_attention_callback=factual_action_attention_callback,
        )
        primary = v3_window.primary
        auxiliary = v3_window.auxiliary
        v3_prior_traces = v3_window.prior_traces
        v3_filter_predictions = v3_window.filter_predictions
    else:
        steps = tuple(
            NativeLocalBPTTStep(
                model_inputs=batch.model_inputs,
                controls=batch.controls,
                prediction_request=request,
                modalities=batch.modalities,
                future_latent_target=(future_latent_target if index == 0 else None),
                wla_world_target=(batch.wla_world_target if index == 0 else None),
            )
            for index, (batch, request) in enumerate(zip(batches, requests, strict=True))
        )
        if len(steps) == 1:
            primary = run_native_policy_training_forward(
                policy,
                model_inputs=batches[0].model_inputs,
                context=native_context_from_persistent_state(
                    controls=batches[0].controls,
                    persistent_state=previous_state,
                    persistent_state_valid=previous_state_valid,
                    prediction_request=requests[0],
                    modalities=batches[0].modalities,
                    supervise_intermediate_relations=(
                        False if graph is None else bool(graph.config.relation_supervision_layers)
                    ),
                ),
                future_latent_target=future_latent_target,
                wla_world_target=batches[0].wla_world_target,
            )
            auxiliary = ()
        else:
            local = run_native_local_bptt(
                policy,
                steps=steps,
                previous_state=previous_state,
                previous_state_valid=previous_state_valid,
            )
            primary = local.primary
            auxiliary = local.auxiliary

    primary_relation = primary.context.relation_output
    if not isinstance(primary_relation, PhysicalRelationOutput):
        raise TypeError("joint physical sequence requires the task-independent graph ABI")
    relation_values = [primary_relation]
    for value in auxiliary:
        relation = value.relation_output
        if not isinstance(relation, PhysicalRelationOutput):
            raise TypeError("joint physical sequence exposed a legacy relation interface")
        relation_values.append(relation)
    relation_axis = tuple(relation_values)
    prediction_outputs = (
        primary.context.prediction_outputs,
        *(value.prediction_outputs for value in auxiliary),
    )
    correction_branches: tuple[NativeCorrectionBranch, ...] = ()
    if correction_active and not v3_active:
        correction_values: list[NativeCorrectionBranch] = []
        for time_index, (batch, request, outputs) in enumerate(
            zip(batches, requests, prediction_outputs, strict=True)
        ):
            if request is None:
                raise RuntimeError("active current correction omitted its request")
            try:
                prediction = outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
            except KeyError as error:
                raise RuntimeError("current correction omitted its projected output") from error
            correction_values.append(
                NativeCorrectionBranch(
                    batch=batch,
                    request=request,
                    prediction=prediction,
                    identity_source_phase=2 * time_index,
                )
            )
        correction_branches = tuple(correction_values)
    filter_phase_branches: tuple[NativeFilterPhaseBranch, ...] = ()
    if correction_active and v3_active:
        filter_values: list[NativeFilterPhaseBranch] = []
        for time_index, (batch, spec, predictions) in enumerate(
            zip(batches, filter_specs, v3_filter_predictions, strict=True)
        ):
            if spec is None or predictions is None:
                raise RuntimeError("active v3 current filter omitted its paired predictions")
            filter_values.extend(
                (
                    NativeFilterPhaseBranch(
                        batch=batch,
                        request=spec.prior_request,
                        prediction=predictions.prior,
                        identity_source_phase=0 if time_index == 0 else 2 * time_index - 1,
                    ),
                    NativeFilterPhaseBranch(
                        batch=batch,
                        request=spec.posterior_request,
                        prediction=predictions.posterior,
                        identity_source_phase=2 * time_index + 1,
                    ),
                )
            )
        filter_phase_branches = tuple(filter_values)

    current_grid_branch = None
    if source_mask is not None:
        if graph is None or current_grid_cache is None:
            raise RuntimeError("active current-grid branch lost its graph or target cache")
        if graph.config.prediction_address_width != 2:
            raise ValueError("current-grid training requires a 2D prediction address")
        current_request = make_native_current_grid_request(
            source_mask=source_mask,
            route_id=current_grid_cache.contract.route_id,
            dtype=batches[0].controls.values.dtype,
        )
        masked = run_native_source_masked_training_forward(
            policy,
            model_inputs=batches[0].model_inputs,
            controls=batches[0].controls,
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
            prediction_request=current_request,
            source_mask=source_mask,
            modalities=batches[0].modalities,
        )
        try:
            current_prediction = masked.prediction_outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
        except KeyError as error:
            raise RuntimeError("current-grid branch omitted its projected output") from error
        current_grid_branch = NativeCurrentGridBranch(
            batch=batches[0],
            request=current_request,
            source_mask=source_mask,
            prediction=current_prediction,
            identity_source_phase=0,
        )
    omitted_static_branch = None
    omitted_static_policy = None
    sequential_omitted_request = None
    if omitted_static_view is not None:
        if graph is None or current_grid_cache is None:
            raise RuntimeError("active omitted-view branch lost its graph or target cache")
        omitted_request = make_native_omitted_static_request(
            omission=omitted_static_view,
            route_id=current_grid_cache.contract.route_id,
            address_width=graph.config.prediction_address_width,
            dtype=batches[0].controls.values.dtype,
        )
        if (
            omitted_static_rematerialization
            == OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD
        ):
            if not v3_active:
                raise ValueError("sequential omitted backward requires the v3 two-pass graph")
            sequential_omitted_request = omitted_request
        elif v3_active:
            if not v3_prior_traces:
                raise RuntimeError("v3 omitted-static branch lost the factual prior trace")
            omitted_arguments = {
                "model_inputs": batches[0].model_inputs,
                "controls": batches[0].controls,
                "prior_trace": v3_prior_traces[0],
                "prediction_request": omitted_request,
                "omission": omitted_static_view,
                "modalities": batches[0].modalities,
                "posterior_adoption_route": posterior_adoption_route,
                "supervise_intermediate_relations": bool(
                    graph.config.relation_supervision_layers
                ),
            }
            if (
                omitted_static_rematerialization
                == OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT
            ):
                context_fn = (
                    _noop_checkpoint_contexts
                    if omitted_static_checkpoint_context_fn is None
                    else omitted_static_checkpoint_context_fn
                )
                # The outer checkpoint drops the complete omitted-view host graph while
                # preserving the exact forward and RNG trajectory. Disabling checkpoint
                # early-stop keeps nested FSDP2/host collectives replayed in full order.
                with set_checkpoint_early_stop(False):
                    omitted_static_policy = checkpoint(
                        run_native_v3_omitted_static_view_policy_training_forward,
                        policy,
                        **omitted_arguments,
                        use_reentrant=False,
                        context_fn=context_fn,
                        determinism_check="default",
                    )
            elif (
                omitted_static_rematerialization
                == OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU
            ):
                # Offload only tensors saved by the omitted branch. The policy,
                # objective assembly, RNG trajectory, and optimizer transaction
                # remain unchanged; ordinary factual-only steps pay no transfer cost.
                forward_context = (
                    nullcontext()
                    if omitted_static_forward_context_fn is None
                    else omitted_static_forward_context_fn()
                )
                with forward_context, save_on_cpu(pin_memory=True, device_type="cuda"):
                    omitted_static_policy = (
                        run_native_v3_omitted_static_view_policy_training_forward(
                            policy,
                            **omitted_arguments,
                        )
                    )
            else:
                omitted_static_policy = (
                    run_native_v3_omitted_static_view_policy_training_forward(
                        policy,
                        **omitted_arguments,
                    )
                )
            omitted_outputs = omitted_static_policy.prediction_outputs
        else:
            omitted = run_native_omitted_image_view_training_forward(
                policy,
                model_inputs=batches[0].model_inputs,
                controls=batches[0].controls,
                previous_state=previous_state,
                previous_state_valid=previous_state_valid,
                prediction_request=omitted_request,
                omission=omitted_static_view,
                modalities=batches[0].modalities,
            )
            omitted_outputs = omitted.prediction_outputs
        if sequential_omitted_request is None:
            try:
                omitted_prediction = omitted_outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
            except KeyError as error:
                raise RuntimeError("omitted-static branch omitted its projected output") from error
            omitted_static_branch = NativeOmittedStaticBranch(
                batch=batches[0],
                request=omitted_request,
                omission=omitted_static_view,
                prediction=omitted_prediction,
                identity_source_phase=0,
            )
    committable_state = native_persistent_output(primary.context)
    attached_egress = None
    if egress_batch is not None:
        if graph is None or current_grid_cache is None:
            raise RuntimeError("active attached egress lost its graph or target cache")
        if not isinstance(committable_state, NativeLayerwisePosteriorState):
            raise RuntimeError("v3 attached egress requires factual layerwise posterior memory")
        egress_request = make_native_current_filter_request(
            source=PredictionSource.PRIOR,
            batch_size=egress_batch.routing.batch_size,
            valid=torch.ones(
                egress_batch.routing.batch_size,
                dtype=torch.bool,
                device=egress_batch.controls.values.device,
            ),
            device=egress_batch.controls.values.device,
            dtype=egress_batch.controls.values.dtype,
            route_id=current_grid_cache.contract.route_id,
            address_width=graph.config.prediction_address_width,
        )
        attached_egress = run_native_v3_attached_egress(
            policy,
            posterior_memory=committable_state,
            posterior_memory_valid=torch.ones(
                egress_batch.routing.batch_size,
                dtype=torch.bool,
                device=egress_batch.controls.values.device,
            ),
            controls=egress_batch.controls,
            prediction_request=egress_request,
            target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
            prior_control_chunks=egress_batch.prior_control_chunks,
            prior_host_steps=egress_prior_host_steps,
        )
        filter_phase_branches = (
            *filter_phase_branches,
            NativeFilterPhaseBranch(
                batch=egress_batch,
                request=egress_request,
                prediction=attached_egress.prediction,
                identity_source_phase=1,
            ),
        )

    predictive_input_factory = None
    if (
        correction_branches
        or filter_phase_branches
        or current_grid_branch is not None
        or omitted_static_branch is not None
    ):
        if current_grid_cache is None:
            raise RuntimeError("active predictive branches lost their frozen target cache")

        def build_predictive_inputs(
            identity_keys: tuple[tuple[str, ...], ...],
        ) -> Sequence[NativePredictiveLossInput]:
            values = list(
                _correction_loss_inputs(
                    branches=correction_branches,
                    cache=current_grid_cache,
                    physical_sidecar=physical_sidecar,
                    track_identity_keys=identity_keys,
                    minimum_visible_fraction=predictive_minimum_visible_fraction,
                    weight=correction_weight,
                    loss_power=predictive_loss_power,
                )
            )
            values.extend(
                _filter_phase_loss_inputs(
                    branches=filter_phase_branches,
                    cache=current_grid_cache,
                    physical_sidecar=physical_sidecar,
                    track_identity_keys=identity_keys,
                    minimum_visible_fraction=predictive_minimum_visible_fraction,
                    weight=correction_weight,
                    loss_power=predictive_loss_power,
                )
            )
            if current_grid_branch is not None:
                values.append(
                    _current_grid_loss_input(
                        branch=current_grid_branch,
                        cache=current_grid_cache,
                        physical_sidecar=physical_sidecar,
                        track_identity_keys=identity_keys,
                        merge_size=merge_size,
                        weight=current_grid_weight,
                        loss_power=predictive_loss_power,
                    )
                )
            if omitted_static_branch is not None:
                values.append(
                    _omitted_static_loss_input(
                        branch=omitted_static_branch,
                        cache=current_grid_cache,
                        physical_sidecar=physical_sidecar,
                        track_identity_keys=identity_keys,
                        weight=omitted_static_weight,
                        loss_power=predictive_loss_power,
                    )
                )
            return values

        predictive_input_factory = build_predictive_inputs
    future_weighted_loss = (
        None
        if primary.future_latent_alignment is None
        else primary.future_latent_alignment.weighted_loss
    )
    official_policy_loss = primary.official_total_loss
    if omitted_static_policy is not None:
        official_policy_loss = torch.stack(
            (
                primary.official_total_loss,
                omitted_static_policy.official_total_loss,
            )
        ).mean()
    if future_weighted_loss is not None:
        if primary.future_latent_alignment is None:
            raise AssertionError("future-latent result disappeared before objective composition")
        official_policy_loss = official_policy_loss + future_latent_objective_contribution(
            primary.future_latent_alignment,
            scale=future_latent_objective_scale,
        )
    target_bundles, objective = _compose_task_independent_sequence_loss(
        batches=batches,
        relations=relation_axis,
        committable_state=committable_state,
        official_policy_loss=official_policy_loss,
        physical_sidecar=physical_sidecar,
        objective_config=objective_config,
        patch_size=patch_size,
        merge_size=merge_size,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        predictive_rollout_factory=predictive_rollout_factory,
        predictive_cache=predictive_cache,
        predictive_loss_power=predictive_loss_power,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        predictive_input_factory=predictive_input_factory,
    )
    sequential_omitted_static = None
    if sequential_omitted_request is not None:
        if current_grid_cache is None:
            raise RuntimeError("sequential omitted branch lost its frozen target cache")
        track_identity_keys = target_bundles[0].identity_keys_by_batch
        target = omitted_static_target(
            batch=batches[0],
            request=sequential_omitted_request,
            omission=omitted_static_view,
            cache=current_grid_cache,
            physical_sidecar=physical_sidecar,
            track_identity_keys=track_identity_keys,
            device=batches[0].controls.values.device,
        )
        omitted_support = materialize_native_predictive_support(
            request=sequential_omitted_request,
            target=target,
            weight=omitted_static_weight,
            identity_source_phase=0,
            assignment=objective.assignment,
            expected_track_identity_keys=track_identity_keys,
            sequence_time_count=len(relation_axis),
        )
        predictive_ledger = build_native_predictive_normalization_ledger(
            (*tuple(term.support() for term in objective.predictive_terms), omitted_support)
        )
        native_config = NativeObjectiveConfig(
            action_weight=objective_config.action_weight,
            predictive_weight=objective_config.predictive_weight,
            structural_weight=objective_config.entity_weight,
        )
        factual_branch = combine_native_sequential_branch(
            official_policy_loss=primary.official_total_loss,
            action_scale=0.5,
            predictive_terms=objective.predictive_terms,
            structural_terms=objective.structural_terms,
            predictive_ledger=predictive_ledger,
            config=native_config,
        )
        factual_backward_loss = factual_branch.total
        if future_weighted_loss is not None:
            factual_backward_loss = factual_backward_loss + future_weighted_loss
        # Sequential execution has exactly one differentiable factual root. The
        # ordinary combined objective contains the same action, predictive and
        # structural leaves, but retaining both reduction graphs adds no
        # supervision and consumes the narrow steady-state FSDP2 margin. Keep
        # its values only as detached diagnostics; the final joint report is
        # rebuilt after the omitted branch has executed.
        objective = replace(
            objective,
            objective=_detached_unified_objective(objective.objective),
        )
        sequential_omitted_static = TaskIndependentCALVINSequentialOmittedPlan(
            batch=batches[0],
            request=sequential_omitted_request,
            omission=omitted_static_view,
            target=target,
            assignment=objective.assignment,
            track_identity_keys=track_identity_keys,
            sequence_time_count=len(relation_axis),
            identity_source_phase=0,
            predictive_ledger=predictive_ledger,
            factual_backward_loss=factual_backward_loss,
            factual_predictive_terms=objective.predictive_terms,
            structural_terms=objective.structural_terms,
            objective_config=objective_config,
            omitted_static_weight=omitted_static_weight,
            predictive_loss_power=predictive_loss_power,
            posterior_adoption_route=posterior_adoption_route,
            supervise_intermediate_relations=bool(graph.config.relation_supervision_layers),
        )
    return TaskIndependentCALVINJointSequenceStepResult(
        primary=primary,
        auxiliary=auxiliary,
        relations=relation_axis,
        targets=target_bundles,
        objective=objective,
        correction_branches=correction_branches,
        filter_phase_branches=filter_phase_branches,
        current_grid_branch=current_grid_branch,
        omitted_static_branch=omitted_static_branch,
        omitted_static_policy=omitted_static_policy,
        attached_egress=attached_egress,
        v3_filter_specs=filter_specs,
        v3_prior_traces=v3_prior_traces,
        omitted_static_rematerialization=(
            omitted_static_rematerialization
            if omitted_static_policy is not None or sequential_omitted_static is not None
            else OMITTED_STATIC_REMATERIALIZATION_NONE
        ),
        sequential_omitted_static=sequential_omitted_static,
    )


def run_task_independent_calvin_sequential_omitted_static_objective(
    policy: nn.Module,
    *,
    plan: TaskIndependentCALVINSequentialOmittedPlan,
    prior_trace: NativeLayerwisePriorTrace,
) -> TaskIndependentCALVINSequentialOmittedResult:
    """Run the deferred branch after factual backward released its activations."""

    if not isinstance(plan, TaskIndependentCALVINSequentialOmittedPlan):
        raise TypeError("sequential omitted execution requires its frozen plan")
    omitted_policy = run_native_v3_omitted_static_view_policy_training_forward(
        policy,
        model_inputs=plan.batch.model_inputs,
        controls=plan.batch.controls,
        prior_trace=prior_trace,
        prediction_request=plan.request,
        omission=plan.omission,
        modalities=plan.batch.modalities,
        posterior_adoption_route=plan.posterior_adoption_route,
        supervise_intermediate_relations=plan.supervise_intermediate_relations,
    )
    try:
        prediction = omitted_policy.prediction_outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
    except KeyError as error:
        raise RuntimeError("sequential omitted branch omitted its projected output") from error
    branch = NativeOmittedStaticBranch(
        batch=plan.batch,
        request=plan.request,
        omission=plan.omission,
        prediction=prediction,
        identity_source_phase=plan.identity_source_phase,
    )
    predictive_terms = materialize_native_predictive_terms(
        (
            NativePredictiveLossInput(
                prediction=prediction,
                request=plan.request,
                target=plan.target,
                weight=plan.omitted_static_weight,
                identity_source_phase=plan.identity_source_phase,
                loss_power=plan.predictive_loss_power,
            ),
        ),
        assignment=plan.assignment,
        expected_track_identity_keys=plan.track_identity_keys,
        sequence_time_count=plan.sequence_time_count,
    )
    branch_objective = combine_native_sequential_branch(
        official_policy_loss=omitted_policy.official_total_loss,
        action_scale=0.5,
        predictive_terms=predictive_terms,
        structural_terms=(),
        predictive_ledger=plan.predictive_ledger,
        config=NativeObjectiveConfig(
            action_weight=plan.objective_config.action_weight,
            predictive_weight=plan.objective_config.predictive_weight,
            structural_weight=plan.objective_config.entity_weight,
        ),
    )
    return TaskIndependentCALVINSequentialOmittedResult(
        policy=omitted_policy,
        branch=branch,
        predictive_terms=predictive_terms,
        backward_loss=branch_objective.total,
    )


def finalize_task_independent_calvin_sequential_omitted_result(
    factual: TaskIndependentCALVINJointSequenceStepResult,
    omitted: TaskIndependentCALVINSequentialOmittedResult,
) -> TaskIndependentCALVINJointSequenceStepResult:
    """Attach a detached-report-equivalent joint objective after both forwards."""

    plan = factual.sequential_omitted_static
    if plan is None:
        raise ValueError("factual result has no sequential omitted plan")
    predictive_terms = merge_repeated_objective_terms(
        (*plan.factual_predictive_terms, *omitted.predictive_terms)
    )
    policy_loss = torch.stack(
        (factual.primary.official_total_loss, omitted.policy.official_total_loss)
    ).mean()
    objective = combine_native_objective(
        official_policy_loss=policy_loss,
        predictive_terms=predictive_terms,
        structural_terms=plan.structural_terms,
        config=NativeObjectiveConfig(
            action_weight=plan.objective_config.action_weight,
            predictive_weight=plan.objective_config.predictive_weight,
            structural_weight=plan.objective_config.entity_weight,
        ),
    )
    merged_objective = replace(
        factual.objective,
        objective=objective,
        predictive_terms=predictive_terms,
    )
    return replace(
        factual,
        objective=merged_objective,
        omitted_static_branch=omitted.branch,
        omitted_static_policy=omitted.policy,
        sequential_omitted_static=None,
        omitted_static_rematerialization=(
            OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD
        ),
    )

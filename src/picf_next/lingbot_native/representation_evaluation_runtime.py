"""Runtime evidence extraction for the immutable representation evaluator.

Every target-facing operation in this module consumes tensors produced by an
already completed LingBot forward. The only full policy diagnostic is wrapped
in a transaction that audits and restores the released action MoE's mutable
monitoring buffers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.calvin import CalvinStatefulTransitionDataset
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.lingbot_native.calvin import (
    CollatedNativeCALVINBatch,
    NativeCALVINTrainingBatch,
    PlannedNativeCALVINReplayBatch,
    build_native_calvin_context,
    build_native_calvin_replay_batch,
)
from picf_next.lingbot_native.calvin_objective import (
    NativeCALVINObjectiveResult,
    NativeStructuralLossConfig,
    compose_native_calvin_objective,
)
from picf_next.lingbot_native.calvin_supervision import TaskIdentityResolver
from picf_next.lingbot_native.deepstack_integrity import tensor_sha256
from picf_next.lingbot_native.fixed_observation import (
    FixedObservationVariant,
    validate_fixed_observation_group_source,
)
from picf_next.lingbot_native.fixed_observation_evaluation import (
    FixedObservationEvaluationItem,
    FixedObservationEvaluationPlan,
    build_fixed_observation_evaluation_sample,
    build_fixed_observation_evaluation_snapshot,
    build_fixed_observation_forward_equivalence_probe,
    fixed_observation_evaluation_mass_strata,
    validate_fixed_observation_evaluation_visual_files,
)
from picf_next.lingbot_native.fixed_observation_training_contract import (
    FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_SCHEMA,
)
from picf_next.lingbot_native.host import LingBotNativeContext
from picf_next.lingbot_native.objective import NativeObjectiveConfig
from picf_next.lingbot_native.relation_precision_audit import (
    build_relation_score_precision_audit,
    build_relation_score_precision_evidence,
    build_relation_score_precision_sample,
)
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationItem,
    RepresentationEvaluationPlan,
    build_representation_evaluation_sample,
    build_representation_evaluation_snapshot,
    build_representation_ownership_row,
    build_representation_token_evidence,
    representation_target_mass_sha256,
    summarize_representation_ownership_rows,
    validate_representation_evaluation_visual_files,
)
from picf_next.lingbot_native.representation_stage import (
    NativeActionStateTensorDigest,
    NativeRepresentationParameterScope,
    native_representation_action_state_changes,
    native_representation_action_state_manifest_sha256,
    native_representation_action_state_tensor_digest,
    native_representation_frozen_action_state_manifest,
)
from picf_next.lingbot_native.row_binding import RowBindings
from picf_next.lingbot_native.state import NativePosteriorState
from picf_next.lingbot_native.supervision import (
    NativeSequenceTargets,
    assignment_binding_start_phase,
)
from picf_next.lingbot_native.task_diagnostics import build_task_row_diagnostics
from picf_next.lingbot_native.training import (
    NativePolicyForwardResult,
    run_native_policy_diagnostic_forward,
    run_native_policy_observation_diagnostic_forward,
)
from picf_next.lingbot_native.visual_audit import render_native_relation_visuals

_ACTION_DIAGNOSTIC_MUTABLE_BUFFER_SUFFIXES = (
    ".avg_topk_sigmoid_score",
    ".tokens_per_expert",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _tensor_set_sha256(values: Mapping[str, torch.Tensor]) -> str:
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError("representation tensor set must be a nonempty named mapping")
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise TypeError("representation tensor set contains a non-tensor")
    return _canonical_sha256(
        [
            {
                "dtype": str(value.dtype),
                "name": name,
                "sha256": tensor_sha256(value),
                "shape": list(value.shape),
            }
            for name, value in sorted(values.items())
        ]
    )


def fixed_observation_training_pair_fingerprint(
    batch: CollatedNativeCALVINBatch,
) -> dict[str, object]:
    """Hash one rank's realized fixed-X batch before any model forward."""

    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("fixed-X training fingerprint requires a collated CALVIN batch")
    if batch.routing.batch_size != 1:
        raise ValueError("fixed-X training fingerprint requires one sample per rank")
    language_fields = {"lang_tokens", "lang_masks"}
    if not language_fields <= set(batch.model_inputs):
        raise ValueError("fixed-X training batch omitted released-host language tensors")
    if any(not isinstance(value, torch.Tensor) for value in batch.model_inputs.values()):
        raise TypeError("fixed-X training model inputs must all be tensors")
    model_inputs = cast(Mapping[str, torch.Tensor], batch.model_inputs)
    non_language = {
        name: value for name, value in model_inputs.items() if name not in language_fields
    }
    controls = {
        "acknowledged": batch.controls.acknowledged,
        "delta_time": batch.controls.delta_time,
        "field_valid": batch.controls.field_valid,
        "reset": batch.controls.reset,
        "token_valid": batch.controls.token_valid,
        "values": batch.controls.values,
    }
    modality_sha256: str | None = None
    if batch.modalities is not None:
        modality_sha256 = _tensor_set_sha256(
            {f"{stream.name}.tokens": stream.tokens for stream in batch.modalities.streams}
            | {f"{stream.name}.valid": stream.valid for stream in batch.modalities.streams}
        )
    routing = batch.routing
    structural_requests = batch.structural_target_requests
    return {
        "batch_size": routing.batch_size,
        "controls_sha256": _tensor_set_sha256(controls),
        "language_masks_sha256": _tensor_set_sha256({"lang_masks": model_inputs["lang_masks"]}),
        "language_tokens_sha256": _tensor_set_sha256({"lang_tokens": model_inputs["lang_tokens"]}),
        "modalities_sha256": modality_sha256,
        "non_language_model_inputs_sha256": _tensor_set_sha256(non_language),
        "routing_source_sha256": _canonical_sha256(
            {
                "episode_keys": routing.episode_keys,
                "frame_indices": routing.frame_indices,
                "optimizer_step": routing.optimizer_step,
                "reset": routing.reset,
                "sample_keys": routing.sample_keys,
            }
        ),
        "schema": FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_SCHEMA,
        "structural_source_sha256": _canonical_sha256(
            [
                {
                    "episode_key": request.episode_key,
                    "sample_key": request.sample_key,
                    "segment_index": request.segment_index,
                    "source_global_index": request.source_global_index,
                    "source_sensor_sha256": request.source_sensor_sha256,
                }
                for request in structural_requests
            ]
        ),
        "task_keys": [request.task_key for request in structural_requests],
    }


def native_relation_output_sha256(relation: RelationOutput) -> str:
    """Hash every tensor that determines one relation diagnostic."""

    if not isinstance(relation, RelationOutput):
        raise TypeError("representation relation hash requires RelationOutput")
    values = {
        "dense_task_grounding_logits": relation.dense_task_grounding_logits,
        "existence_logits": relation.existence_logits,
        "ownership": relation.ownership,
        "sensor_valid": relation.sensor_valid,
        "support_logits": relation.support_logits,
        "task_relevance_logits": relation.task_relevance_logits,
        "row_embeddings": relation.row_embeddings,
        "relation_temperature": relation.relation_temperature,
    }
    if relation.task_embedding is not None:
        values["task_embedding"] = relation.task_embedding
    if relation.match_embeddings is not None:
        values["match_embeddings"] = relation.match_embeddings
    if relation.task_relevance_logits_fp32 is not None:
        values["task_relevance_logits_fp32"] = relation.task_relevance_logits_fp32
    if relation.ownership_log_probability is not None:
        values["ownership_log_probability"] = relation.ownership_log_probability
    if relation.task_object_log_probability is not None:
        values["task_object_log_probability"] = relation.task_object_log_probability
    if relation.task_object_probability is not None:
        values["task_object_probability"] = relation.task_object_probability
    if relation.task_event_distribution is not None:
        values["task_event_distribution"] = relation.task_event_distribution
    if relation.task_row_probability is not None:
        values["task_row_probability"] = relation.task_row_probability
    if relation.structural_sensor_valid is not None:
        values["structural_sensor_valid"] = relation.structural_sensor_valid
    return hashlib.sha256(
        json.dumps(
            {
                "task_interface": relation.task_interface,
                "tensor_sha256": _tensor_set_sha256(values),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def native_sequence_targets_sha256(targets: NativeSequenceTargets) -> str:
    """Hash every primitive in one post-forward physical target bundle."""

    if not isinstance(targets, NativeSequenceTargets):
        raise TypeError("representation target hash requires NativeSequenceTargets")
    tensor_digest = _tensor_set_sha256(
        {
            "capacity_censored": targets.capacity_censored,
            "existence": targets.existence,
            "existence_valid": targets.existence_valid,
            "inventory_exhaustive": targets.inventory_exhaustive,
            "mask_valid": targets.mask_valid,
            "masks": targets.masks,
            "task_relevance": targets.task_relevance,
            "task_valid": targets.task_valid,
            "token_observed_fraction": targets.token_observed_fraction,
            "track_valid": targets.track_valid,
        }
    )
    return _canonical_sha256(
        {
            "exclusive_ownership": targets.exclusive_ownership,
            "tensor_set_sha256": tensor_digest,
        }
    )


@dataclass(frozen=True, slots=True)
class RepresentationRuntimeEvidence:
    """Recomputable loss-side evidence extracted after one reset-frame forward."""

    token_evidence: dict[str, object]
    task_row_diagnostic: dict[str, object]
    ownership_rows: tuple[dict[str, object], ...]
    ownership_summary: dict[str, object]
    target_sha256: str
    target_mass_by_identity: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class _PendingRepresentationEvaluationSample:
    item: RepresentationEvaluationItem
    factual_instruction_sha256: str
    shuffled_task_instruction_sha256: str
    factual: RepresentationRuntimeEvidence
    shuffled_task: RepresentationRuntimeEvidence
    official_action_loss: float
    factual_forward_seconds: float
    shuffled_task_forward_seconds: float
    peak_cuda_reserved_bytes: int
    factual_relation_sha256: str
    shuffled_task_relation_sha256: str
    factual_relation_precision: dict[str, object]
    shuffled_task_relation_precision: dict[str, object]
    visual_artifact: dict[str, object]


def build_representation_runtime_evidence(
    objective: NativeCALVINObjectiveResult,
    *,
    structural_sensor_valid: torch.Tensor,
    batch_index: int,
) -> RepresentationRuntimeEvidence:
    """Extract task and ownership evidence without changing the forward graph."""

    if not isinstance(objective, NativeCALVINObjectiveResult):
        raise TypeError("representation runtime evidence requires a native CALVIN objective")
    predictions = objective.predictions
    targets = objective.targets
    batch, time, tokens, rows = predictions.support_logits.shape
    if time != 1 or targets.masks.shape[:2] != (batch, 1):
        raise ValueError("representation evaluation requires one reset-frame time slice")
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or not 0 <= batch_index < batch
    ):
        raise IndexError("representation evaluation batch index is outside the objective")
    if (
        structural_sensor_valid.shape != (batch, tokens)
        or structural_sensor_valid.dtype != torch.bool
        or structural_sensor_valid.device != predictions.support_logits.device
    ):
        raise ValueError("representation structural validity differs from relation tokens")

    observed = targets.token_observed_fraction[batch_index, 0] > 0
    evaluation_valid = structural_sensor_valid[batch_index] & observed
    if not bool(evaluation_valid.any()):
        raise ValueError("representation evaluation has no physically observed visual token")
    task_tracks = (
        targets.track_valid[batch_index]
        & targets.task_valid[batch_index]
        & targets.task_relevance[batch_index].gt(0)
    )
    target_mass = targets.masks[batch_index, 0, task_tracks].sum(dim=0)
    identity_keys = objective.track_identity_keys_by_batch[batch_index]
    target_mass_by_identity = {
        identity_key: tuple(
            targets.masks[batch_index, 0, track_index, evaluation_valid]
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        for track_index, identity_key in enumerate(identity_keys)
    }
    token_evidence = build_representation_token_evidence(
        predictions.dense_task_grounding_logits[batch_index, 0, evaluation_valid]
        .detach()
        .float()
        .cpu()
        .tolist(),
        target_mass[evaluation_valid].detach().float().cpu().tolist(),
    )

    row_to_track = objective.assignment.row_to_track[batch_index]
    binding_start_phase = assignment_binding_start_phase(
        objective.assignment,
        targets,
    )[batch_index]
    if row_to_track.shape != (rows,):
        raise ValueError("representation row assignment differs from prediction capacity")
    ownership_rows: list[dict[str, object]] = []
    for row_index, track_index in enumerate(row_to_track.detach().cpu().tolist()):
        if track_index < 0 or int(binding_start_phase[row_index].item()) > 1:
            continue
        if track_index >= len(identity_keys):
            raise ValueError("representation row assignment references an absent identity")
        valid = (
            targets.mask_valid[batch_index, 0, track_index]
            & structural_sensor_valid[batch_index]
            & observed
        )
        target = targets.masks[batch_index, 0, track_index]
        if not bool(valid.any()) or not bool(target[valid].sum() > 0):
            continue
        prediction = (
            predictions.ownership[batch_index, 0, :, row_index]
            if targets.exclusive_ownership
            else predictions.support_logits[batch_index, 0, :, row_index].sigmoid()
        )
        weight = (
            targets.token_observed_fraction[batch_index, 0]
            if targets.exclusive_ownership
            else torch.ones_like(target)
        )
        ownership_rows.append(
            build_representation_ownership_row(
                row_index=row_index,
                track_index=track_index,
                identity_key=identity_keys[track_index],
                is_task_target=bool(targets.task_relevance[batch_index, track_index] > 0),
                prediction=prediction[valid].detach().float().cpu().tolist(),
                target=target[valid].detach().float().cpu().tolist(),
                weight=weight[valid].detach().float().cpu().tolist(),
            )
        )
    if not ownership_rows:
        raise ValueError("representation evaluation has no visible matched physical row")
    task_row_diagnostics = build_task_row_diagnostics(objective)
    return RepresentationRuntimeEvidence(
        token_evidence=token_evidence,
        task_row_diagnostic=task_row_diagnostics[batch_index],
        ownership_rows=tuple(ownership_rows),
        ownership_summary=summarize_representation_ownership_rows(ownership_rows),
        target_sha256=native_sequence_targets_sha256(targets),
        target_mass_by_identity=target_mass_by_identity,
    )


def _target_mass_for_identities(
    evidence: RepresentationRuntimeEvidence,
    identity_keys: tuple[str, ...],
) -> tuple[float, ...]:
    logits = _runtime_token_logits(evidence)
    missing = sorted(set(identity_keys) - set(evidence.target_mass_by_identity))
    if missing:
        raise ValueError(
            f"representation counterfactual identities are absent from the scene: {missing}"
        )
    if not identity_keys:
        return (0.0,) * len(logits)
    columns = tuple(evidence.target_mass_by_identity[key] for key in identity_keys)
    if any(len(column) != len(logits) for column in columns):
        raise RuntimeError("representation identity target differs from the factual token domain")
    return tuple(sum(values) for values in zip(*columns, strict=True))


def _runtime_token_logits(evidence: RepresentationRuntimeEvidence) -> tuple[float, ...]:
    raw_logits = evidence.token_evidence.get("logits")
    if not isinstance(raw_logits, list) or not raw_logits:
        raise RuntimeError("representation runtime evidence lost token logits")
    logits: list[float] = []
    for value in raw_logits:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise RuntimeError("representation runtime evidence contains an invalid token logit")
        logits.append(float(value))
    return tuple(logits)


def _instruction_for_sample(
    dataset: CalvinStatefulTransitionDataset,
    sample_key: str,
    *,
    expected_transition_index: int,
) -> str:
    if (
        isinstance(expected_transition_index, bool)
        or not isinstance(expected_transition_index, int)
        or expected_transition_index < 0
    ):
        raise ValueError("representation evaluation donor transition is invalid")
    locator = dataset.locator_by_key(sample_key)
    source = dataset.index.segments[locator.segment_index]
    if locator.global_index != source.start + expected_transition_index or not source.instruction:
        raise ValueError("representation evaluation donor differs from its planned age")
    return source.instruction


def _instruction_sha256(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("representation evaluation instruction must be nonempty")
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def _evaluation_replay_seed(
    plan: RepresentationEvaluationPlan,
    item: RepresentationEvaluationItem,
) -> int:
    payload = (
        f"{plan.replay_seed_sha256}\0{item.partition}\0{item.ordinal}\0{item.sample_key}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _evaluation_forward_seed(
    plan: RepresentationEvaluationPlan,
    item: RepresentationEvaluationItem,
) -> int:
    payload = (
        f"{plan.replay_seed_sha256}\0{item.partition}\0{item.ordinal}\0"
        f"{item.sample_key}\0model-forward"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _evaluation_history_seed(forward_seed: int) -> int:
    if (
        isinstance(forward_seed, bool)
        or not isinstance(forward_seed, int)
        or not 0 <= forward_seed < 2**63
    ):
        raise ValueError("representation evaluation source forward seed is invalid")
    payload = f"{forward_seed}\0warm-history".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _seed_evaluation_forward(seed: int, *, device: torch.device) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("representation evaluation forward seed is invalid")
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state())
    with torch.cuda.device(device):
        torch.cuda.manual_seed(seed)


def _shuffled_task_replay(
    factual: PlannedNativeCALVINReplayBatch,
    *,
    instruction: str,
) -> PlannedNativeCALVINReplayBatch:
    if factual.training.routing.batch_size != 1 or len(factual.training.host_items) != 1:
        raise ValueError("representation evaluation requires one sample per rank")
    host_item = copy.deepcopy(factual.training.host_items[0])
    if "task" not in host_item or not isinstance(host_item["task"], str):
        raise ValueError("representation evaluation factual host item has no task")
    host_item["task"] = instruction
    training = NativeCALVINTrainingBatch(
        host_items=(host_item,),
        controls=factual.training.controls,
        routing=factual.training.routing,
        structural_target_requests=factual.training.structural_target_requests,
    )
    return replace(factual, training=training)


def _validate_matched_task_control_inputs(
    factual: CollatedNativeCALVINBatch,
    shuffled: CollatedNativeCALVINBatch,
) -> None:
    def equal(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and torch.equal(left, right)
            )
        if type(left) is not type(right):
            return False
        if is_dataclass(left) and not isinstance(left, type):
            return all(
                equal(getattr(left, field.name), getattr(right, field.name))
                for field in fields(left)
            )
        if isinstance(left, Mapping):
            return set(left) == set(right) and all(equal(left[name], right[name]) for name in left)
        if isinstance(left, tuple | list):
            return len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right, strict=True)
            )
        return bool(left == right)

    if (
        not equal(factual.controls, shuffled.controls)
        or factual.routing != shuffled.routing
        or factual.structural_target_requests != shuffled.structural_target_requests
        or factual.source_digest != shuffled.source_digest
        or not equal(factual.modalities, shuffled.modalities)
        or set(factual.model_inputs) != set(shuffled.model_inputs)
    ):
        raise ValueError("representation shuffled-task control changed non-language contracts")
    language_fields = {"lang_tokens", "lang_masks"}
    for name in factual.model_inputs:
        factual_value = factual.model_inputs[name]
        shuffled_value = shuffled.model_inputs[name]
        if not isinstance(factual_value, torch.Tensor) or not isinstance(
            shuffled_value, torch.Tensor
        ):
            raise TypeError("representation evaluation model input is not a tensor")
        if name not in language_fields and not torch.equal(factual_value, shuffled_value):
            raise ValueError(f"representation shuffled-task control changed model input {name!r}")
    if all(
        torch.equal(factual.model_inputs[name], shuffled.model_inputs[name])
        for name in language_fields
    ):
        raise ValueError("representation shuffled-task control retained tokenized language")


def _evaluation_objective(
    context: Any,
    batch: CollatedNativeCALVINBatch,
    *,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    structural_config: NativeStructuralLossConfig,
    minimum_supervised_fraction: float,
    capacity_seed: int,
    prior_row_bindings: RowBindings,
) -> NativeCALVINObjectiveResult:
    relation = getattr(context, "relation_output", None)
    if not isinstance(relation, RelationOutput):
        raise RuntimeError("representation evaluation forward emitted no relation output")
    return compose_native_calvin_objective(
        official_policy_loss=None,
        requests_by_time=(batch.structural_target_requests,),
        model_inputs_by_time=(batch.model_inputs,),
        relations=(relation,),
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=task_identity_resolver,
        patch_size=patch_size,
        merge_size=merge_size,
        objective_config=NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
            action_weight=0.0,
        ),
        structural_config=structural_config,
        require_policy_loss_grad=False,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=(capacity_seed,),
        prior_row_bindings_by_batch=(prior_row_bindings,),
    )


def _require_shared_assignment_gauge(
    left: NativeCALVINObjectiveResult,
    right: NativeCALVINObjectiveResult,
    *,
    comparison: str,
) -> None:
    """Reject row-wise evidence when two controls use different physical gauges."""

    left_phase = assignment_binding_start_phase(left.assignment, left.targets)
    right_phase = assignment_binding_start_phase(right.assignment, right.targets)
    if (
        not torch.equal(left.assignment.row_to_track, right.assignment.row_to_track)
        or not torch.equal(left_phase, right_phase)
        or left.track_identity_keys_by_batch != right.track_identity_keys_by_batch
        or left.row_bindings_by_batch != right.row_bindings_by_batch
    ):
        raise RuntimeError(f"{comparison} changed the loss-side physical row gauge")


def _distributed_phase_error(
    *,
    error: BaseException | None,
    phase: str,
    rank: int,
    world_size: int,
    dist_module: Any,
) -> None:
    local = (
        None
        if error is None
        else {
            "rank": rank,
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
        }
    )
    gathered: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(gathered, local)
    failures = tuple(item for item in gathered if item is not None)
    if failures:
        rendered = "; ".join(
            f"rank {item['rank']} {item['phase']} {item['type']}: {item['message']}"
            for item in failures
        )
        raise RuntimeError(f"distributed representation evaluation failed: {rendered}")


@contextmanager
def _distributed_action_diagnostic_transaction(
    policy: nn.Module,
    *,
    expected_scope: NativeRepresentationParameterScope,
    rank: int,
    world_size: int,
    dist_module: Any,
) -> Iterator[RepresentationActionDiagnosticGuard]:
    """Open and close the read-only action transaction on every rank together."""

    guard: RepresentationActionDiagnosticGuard | None = None
    enter_error: BaseException | None = None
    try:
        guard = RepresentationActionDiagnosticGuard(
            policy,
            expected_scope=expected_scope,
        )
        guard.__enter__()
    except BaseException as error:
        enter_error = error
    _distributed_phase_error(
        error=enter_error,
        phase="action-transaction-enter",
        rank=rank,
        world_size=world_size,
        dist_module=dist_module,
    )
    if guard is None:
        raise RuntimeError("representation action diagnostic transaction vanished")

    body_error: BaseException | None = None
    try:
        yield guard
    except BaseException as error:
        body_error = error
        raise
    finally:
        close_error: BaseException | None = None
        try:
            guard.close()
        except BaseException as error:
            close_error = error
        # A peer may still be inside FSDP; another collective would mismatch it.
        if body_error is not None:
            if close_error is not None:
                raise close_error from body_error
        else:
            _distributed_phase_error(
                error=close_error,
                phase="action-transaction-close",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )


def _prepare_evaluation_batches(
    *,
    plan: RepresentationEvaluationPlan,
    item: RepresentationEvaluationItem,
    checkpoint_global_step: int,
    rank: int,
    dataset: CalvinStatefulTransitionDataset,
    device: torch.device,
    collate_planned: Callable[[PlannedNativeCALVINReplayBatch], CollatedNativeCALVINBatch],
) -> tuple[
    PlannedNativeCALVINReplayBatch,
    tuple[CollatedNativeCALVINBatch, ...],
    CollatedNativeCALVINBatch,
    CollatedNativeCALVINBatch,
    str,
    str,
    int,
]:
    replay_seed = _evaluation_replay_seed(plan, item)
    factual = build_native_calvin_replay_batch(
        dataset,
        sample_key=item.sample_key,
        lane_id=rank,
        episode_instance_id=f"representation-evaluation/{item.partition}/{item.ordinal}",
        optimizer_step=checkpoint_global_step,
        replay_seed=replay_seed,
        device=device,
        dtype=torch.bfloat16,
    )
    factual_instruction = factual.training.host_items[0]["task"]
    shuffled_instruction = _instruction_for_sample(
        dataset,
        item.shuffled_task_sample_key,
        expected_transition_index=plan.history_transitions,
    )
    shuffled_target_instruction = _instruction_for_sample(
        dataset,
        item.shuffled_target_sample_key,
        expected_transition_index=plan.history_transitions,
    )
    factual_instruction_sha256 = _instruction_sha256(factual_instruction)
    shuffled_instruction_sha256 = _instruction_sha256(shuffled_instruction)
    shuffled_target_instruction_sha256 = _instruction_sha256(shuffled_target_instruction)
    if (
        factual_instruction_sha256 != item.factual_task_instruction_sha256
        or shuffled_instruction_sha256 != item.shuffled_task_instruction_sha256
        or shuffled_target_instruction_sha256 != item.shuffled_target_instruction_sha256
    ):
        raise ValueError("representation evaluation instruction differs from its frozen plan")
    factual_batch = collate_planned(factual)
    locator = dataset.locator_by_key(item.sample_key)
    episode = dataset.episode_manifest[locator.segment_index]
    if (
        plan.history_transitions >= len(episode.sample_keys)
        or episode.sample_keys[plan.history_transitions] != item.sample_key
    ):
        raise ValueError("representation evaluation sample age differs from its plan")
    history_planned = tuple(
        build_native_calvin_replay_batch(
            dataset,
            sample_key=sample_key,
            lane_id=rank,
            episode_instance_id=f"representation-evaluation/{item.partition}/{item.ordinal}",
            optimizer_step=checkpoint_global_step,
            replay_seed=replay_seed,
            device=device,
            dtype=torch.bfloat16,
        )
        for sample_key in episode.sample_keys[: plan.history_transitions]
    )
    if any(
        batch.training.host_items[0]["task"] != factual_instruction for batch in history_planned
    ):
        raise ValueError("warm representation history changed its natural instruction")
    history_batches = tuple(collate_planned(batch) for batch in history_planned)
    shuffled_batch = collate_planned(
        _shuffled_task_replay(factual, instruction=shuffled_instruction)
    )
    _validate_matched_task_control_inputs(factual_batch, shuffled_batch)
    capacity_seed = int.from_bytes(
        hashlib.sha256(f"{replay_seed}\0capacity".encode("ascii")).digest()[:8],
        "big",
    )
    return (
        factual,
        history_batches,
        factual_batch,
        shuffled_batch,
        factual_instruction_sha256,
        shuffled_instruction_sha256,
        capacity_seed,
    )


def _run_action_evaluation_forward(
    guard: RepresentationActionDiagnosticGuard,
    *,
    batch: CollatedNativeCALVINBatch,
    previous_state: NativePosteriorState | None,
    device: torch.device,
) -> tuple[NativePolicyForwardResult, float]:
    context = build_native_calvin_context(batch, previous_state=previous_state)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    diagnostic = guard.run(model_inputs=batch.model_inputs, context=context)
    torch.cuda.synchronize(device)
    if not isinstance(diagnostic.result, NativePolicyForwardResult):
        raise TypeError("representation action diagnostic returned another result type")
    return diagnostic.result, time.perf_counter() - started


def _run_observation_evaluation_forward(
    policy: nn.Module,
    *,
    batch: CollatedNativeCALVINBatch,
    previous_state: NativePosteriorState | None,
    device: torch.device,
) -> tuple[LingBotNativeContext, float]:
    context = build_native_calvin_context(batch, previous_state=previous_state)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = run_native_policy_observation_diagnostic_forward(
        policy,
        model_inputs=batch.model_inputs,
        context=context,
    )
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def _reconstruct_evaluation_prior(
    policy: nn.Module,
    *,
    history_batches: tuple[CollatedNativeCALVINBatch, ...],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    structural_config: NativeStructuralLossConfig,
    minimum_supervised_fraction: float,
    capacity_seed: int,
) -> tuple[NativePosteriorState | None, RowBindings]:
    """Replay posterior state and its loss-side physical row gauge together."""

    state: NativePosteriorState | None = None
    row_bindings: RowBindings = ()
    with torch.no_grad():
        for batch in history_batches:
            if batch.routing.batch_size != 1:
                raise ValueError("warm representation replay requires one sample")
            reset = bool((batch.controls.reset & batch.controls.token_valid).any().item())
            if reset:
                row_bindings = ()
            context = run_native_policy_observation_diagnostic_forward(
                policy,
                model_inputs=batch.model_inputs,
                context=build_native_calvin_context(batch, previous_state=state),
            )
            posterior = context.posterior_state
            if posterior is None:
                raise RuntimeError("warm representation replay omitted posterior state")
            objective = _evaluation_objective(
                context,
                batch,
                physical_sidecar=physical_sidecar,
                capacity=capacity,
                task_identity_resolver=task_identity_resolver,
                patch_size=patch_size,
                merge_size=merge_size,
                structural_config=structural_config,
                minimum_supervised_fraction=minimum_supervised_fraction,
                capacity_seed=capacity_seed,
                prior_row_bindings=row_bindings,
            )
            if len(objective.row_bindings_by_batch) != 1:
                raise RuntimeError("warm representation replay changed its batch gauge")
            row_bindings = objective.row_bindings_by_batch[0]
            state = NativePosteriorState(posterior.rows.detach().clone())
    return state, row_bindings


def _write_snapshot(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    write_bytes_durable_exclusive(path, payload.encode("ascii"))


def _fixed_observation_variant_replay(
    source: PlannedNativeCALVINReplayBatch,
    variant: FixedObservationVariant,
) -> PlannedNativeCALVINReplayBatch:
    if source.training.routing.batch_size != 1 or len(source.training.host_items) != 1:
        raise ValueError("fixed-X evaluation requires one source sample per rank")
    host_item = copy.deepcopy(source.training.host_items[0])
    if not isinstance(host_item.get("task"), str):
        raise ValueError("fixed-X evaluation source host item has no natural task")
    host_item["task"] = variant.instruction
    request = source.training.structural_target_requests[0]
    training = NativeCALVINTrainingBatch(
        host_items=(host_item,),
        controls=source.training.controls,
        routing=source.training.routing,
        structural_target_requests=(replace(request, task_key=variant.task_key),),
    )
    return replace(source, training=training)


def _same_runtime_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _same_runtime_value(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _same_runtime_value(left[name], right[name]) for name in left
        )
    if isinstance(left, tuple | list):
        return len(left) == len(right) and all(
            _same_runtime_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_fixed_observation_pair_inputs(
    first: CollatedNativeCALVINBatch,
    second: CollatedNativeCALVINBatch,
    *,
    variants: tuple[FixedObservationVariant, FixedObservationVariant],
) -> None:
    first_request = first.structural_target_requests[0]
    second_request = second.structural_target_requests[0]
    if (
        first.routing.batch_size != 1
        or second.routing.batch_size != 1
        or first.source_digest != second.source_digest
        or not _same_runtime_value(first.controls, second.controls)
        or first.routing != second.routing
        or not _same_runtime_value(first.modalities, second.modalities)
        or first_request != replace(second_request, task_key=first_request.task_key)
        or first_request.task_key != variants[0].task_key
        or second_request.task_key != variants[1].task_key
        or set(first.model_inputs) != set(second.model_inputs)
    ):
        raise ValueError("fixed-X evaluation pair changed a non-language source contract")
    language_fields = {"lang_tokens", "lang_masks"}
    for name in first.model_inputs:
        first_value = first.model_inputs[name]
        second_value = second.model_inputs[name]
        if not isinstance(first_value, torch.Tensor) or not isinstance(
            second_value,
            torch.Tensor,
        ):
            raise TypeError("fixed-X evaluation model input is not a tensor")
        if name not in language_fields and not torch.equal(first_value, second_value):
            raise ValueError(f"fixed-X evaluation pair changed model input {name!r}")
    if all(
        torch.equal(first.model_inputs[name], second.model_inputs[name]) for name in language_fields
    ):
        raise ValueError("fixed-X evaluation pair retained tokenized language")


def _fixed_observation_model_input_hashes(
    first: CollatedNativeCALVINBatch,
    second: CollatedNativeCALVINBatch,
) -> tuple[str, tuple[str, str]]:
    language_fields = {"lang_tokens", "lang_masks"}
    non_language = {
        name: value for name, value in first.model_inputs.items() if name not in language_fields
    }
    first_language = {
        name: value for name, value in first.model_inputs.items() if name in language_fields
    }
    second_language = {
        name: value for name, value in second.model_inputs.items() if name in language_fields
    }
    return (
        _tensor_set_sha256(non_language),
        (
            _tensor_set_sha256(first_language),
            _tensor_set_sha256(second_language),
        ),
    )


def _prepare_fixed_observation_evaluation_pair(
    *,
    item: FixedObservationEvaluationItem,
    checkpoint_global_step: int,
    rank: int,
    dataset: CalvinStatefulTransitionDataset,
    device: torch.device,
    collate_planned: Callable[[PlannedNativeCALVINReplayBatch], CollatedNativeCALVINBatch],
) -> tuple[
    tuple[PlannedNativeCALVINReplayBatch, PlannedNativeCALVINReplayBatch],
    tuple[CollatedNativeCALVINBatch, CollatedNativeCALVINBatch],
    int,
    str,
    tuple[str, str],
]:
    validate_fixed_observation_group_source(dataset, item.group)
    source = build_native_calvin_replay_batch(
        dataset,
        sample_key=item.group.stateful_sample_key,
        lane_id=rank,
        episode_instance_id=f"fixed-X-evaluation/{item.partition}/{item.ordinal}",
        optimizer_step=checkpoint_global_step,
        replay_seed=item.replay_seed,
        device=device,
        dtype=torch.bfloat16,
    )
    planned = tuple(_fixed_observation_variant_replay(source, variant) for variant in item.variants)
    batches = tuple(collate_planned(value) for value in planned)
    _validate_fixed_observation_pair_inputs(
        batches[0],
        batches[1],
        variants=item.variants,
    )
    non_language_sha256, language_sha256 = _fixed_observation_model_input_hashes(
        batches[0],
        batches[1],
    )
    capacity_seed = int.from_bytes(
        hashlib.sha256(f"{item.replay_seed}\0capacity".encode("ascii")).digest()[:8],
        "big",
    )
    return (
        (planned[0], planned[1]),
        (batches[0], batches[1]),
        capacity_seed,
        non_language_sha256,
        language_sha256,
    )


def run_fixed_observation_checkpoint_evaluation(
    policy: nn.Module,
    *,
    expected_scope: NativeRepresentationParameterScope,
    plan: FixedObservationEvaluationPlan,
    checkpoint_global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    representation_split_sha256: str,
    dataset: CalvinStatefulTransitionDataset,
    collate_planned: Callable[[PlannedNativeCALVINReplayBatch], CollatedNativeCALVINBatch],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    structural_config: NativeStructuralLossConfig,
    minimum_supervised_fraction: float,
    output_root: Path,
    rank: int,
    world_size: int,
    dist_module: Any,
    device: torch.device,
) -> dict[str, object] | None:
    """Evaluate two true prompts over each immutable source observation."""

    if not isinstance(plan, FixedObservationEvaluationPlan) or world_size != plan.world_size:
        raise ValueError("fixed-X evaluation plan differs from distributed topology")
    if rank < 0 or rank >= world_size or checkpoint_global_step < 0:
        raise ValueError("fixed-X evaluation rank or checkpoint step is invalid")
    if not callable(collate_planned):
        raise TypeError("fixed-X evaluation collator must be callable")
    for item in plan.items:
        for variant in item.variants:
            if tuple(task_identity_resolver(variant.task_key) or ()) != (
                variant.target_identity_key,
            ):
                raise ValueError("fixed-X evaluation target identity differs from its plan")
    local_items = plan.items_for("validation", rank) + plan.items_for("heldout", rank)
    if not local_items:
        raise ValueError("fixed-X evaluation rank has no planned samples")
    counts: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(counts, len(local_items))
    if len(set(counts)) != 1:
        raise ValueError("fixed-X evaluation ranks have unequal sample counts")

    setup_error: list[str | None] = [None]
    if rank == 0:
        try:
            if output_root.exists() or output_root.is_symlink():
                raise FileExistsError(f"fixed-X evaluation output root exists: {output_root}")
            output_root.mkdir(parents=True)
        except BaseException as error:
            setup_error[0] = f"{type(error).__name__}: {error}"
    dist_module.broadcast_object_list(setup_error, src=0)
    if setup_error[0] is not None:
        raise RuntimeError(f"fixed-X evaluation setup failed: {setup_error[0]}")
    dist_module.barrier()

    strata = fixed_observation_evaluation_mass_strata(plan)
    local_samples: list[dict[str, object]] = []
    local_forward_equivalence_probe: dict[str, object] | None = None
    action_state_sha256 = ""
    with _distributed_action_diagnostic_transaction(
        policy,
        expected_scope=expected_scope,
        rank=rank,
        world_size=world_size,
        dist_module=dist_module,
    ) as action_guard:
        action_state_sha256 = action_guard.action_state_sha256
        for local_index, item in enumerate(local_items):
            prepared = None
            prepare_error: BaseException | None = None
            try:
                prepared = _prepare_fixed_observation_evaluation_pair(
                    item=item,
                    checkpoint_global_step=checkpoint_global_step,
                    rank=rank,
                    dataset=dataset,
                    device=device,
                    collate_planned=collate_planned,
                )
            except BaseException as error:
                prepare_error = error
            _distributed_phase_error(
                error=prepare_error,
                phase=f"fixed-X-sample-{local_index}-prepare",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if prepared is None:
                raise RuntimeError("fixed-X evaluation preparation vanished")
            planned, batches, capacity_seed, non_language_sha256, language_sha256 = prepared
            torch.cuda.reset_peak_memory_stats(device)
            contexts: list[LingBotNativeContext] = []
            forward_seconds: list[float] = []
            forward_error: BaseException | None = None
            try:
                for batch in batches:
                    _seed_evaluation_forward(item.replay_seed, device=device)
                    context, seconds = _run_observation_evaluation_forward(
                        policy,
                        batch=batch,
                        previous_state=None,
                        device=device,
                    )
                    contexts.append(context)
                    forward_seconds.append(seconds)
                if local_index == 0:
                    _seed_evaluation_forward(item.replay_seed, device=device)
                    repeated_context, repeat_seconds = _run_observation_evaluation_forward(
                        policy,
                        batch=batches[0],
                        previous_state=None,
                        device=device,
                    )
                    first_relation = contexts[0].relation_output
                    repeated_relation = repeated_context.relation_output
                    if not isinstance(first_relation, RelationOutput) or not isinstance(
                        repeated_relation,
                        RelationOutput,
                    ):
                        raise RuntimeError(
                            "fixed-X forward-equivalence probe omitted relation output"
                        )
                    local_forward_equivalence_probe = (
                        build_fixed_observation_forward_equivalence_probe(
                            plan=plan,
                            item=item,
                            checkpoint_global_step=checkpoint_global_step,
                            model_inputs_sha256=_tensor_set_sha256(batches[0].model_inputs),
                            relation_sha256=native_relation_output_sha256(first_relation),
                            repeated_relation_sha256=native_relation_output_sha256(
                                repeated_relation
                            ),
                            repeat_forward_seconds=repeat_seconds,
                        )
                    )
            except BaseException as error:
                forward_error = error
            _distributed_phase_error(
                error=forward_error,
                phase=f"fixed-X-sample-{local_index}-forward",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if len(contexts) != 2 or len(forward_seconds) != 2:
                raise RuntimeError("fixed-X evaluation forward pair vanished")

            evidence_error: BaseException | None = None
            sample: dict[str, object] | None = None
            try:
                relations = tuple(context.relation_output for context in contexts)
                if any(not isinstance(relation, RelationOutput) for relation in relations):
                    raise RuntimeError("fixed-X evaluation forward omitted relation output")
                typed_relations = cast(tuple[RelationOutput, RelationOutput], relations)
                objectives = tuple(
                    _evaluation_objective(
                        context,
                        batch,
                        physical_sidecar=physical_sidecar,
                        capacity=capacity,
                        task_identity_resolver=task_identity_resolver,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        structural_config=structural_config,
                        minimum_supervised_fraction=minimum_supervised_fraction,
                        capacity_seed=capacity_seed,
                        prior_row_bindings=(),
                    )
                    for context, batch in zip(contexts, batches, strict=True)
                )
                _require_shared_assignment_gauge(
                    objectives[0],
                    objectives[1],
                    comparison="fixed-X prompt control",
                )
                evidence = tuple(
                    build_representation_runtime_evidence(
                        objective,
                        structural_sensor_valid=relation.structural_valid,
                        batch_index=0,
                    )
                    for objective, relation in zip(objectives, typed_relations, strict=True)
                )
                variant_results = []
                for variant_index, (
                    variant,
                    alternate,
                    relation,
                    objective,
                    runtime_evidence,
                ) in enumerate(
                    (
                        (
                            item.variants[0],
                            item.variants[1],
                            typed_relations[0],
                            objectives[0],
                            evidence[0],
                        ),
                        (
                            item.variants[1],
                            item.variants[0],
                            typed_relations[1],
                            objectives[1],
                            evidence[1],
                        ),
                    )
                ):
                    alternate_mass = _target_mass_for_identities(
                        runtime_evidence,
                        (alternate.target_identity_key,),
                    )
                    alternate_token = build_representation_token_evidence(
                        _runtime_token_logits(runtime_evidence),
                        alternate_mass,
                    )
                    visuals = render_native_relation_visuals(
                        output_root=output_root,
                        global_step=checkpoint_global_step,
                        input_weight_global_step=checkpoint_global_step,
                        weight_boundary="fixed_observation_checkpoint_evaluation",
                        rank=rank,
                        host_items=planned[variant_index].training.host_items,
                        model_inputs=batches[variant_index].model_inputs,
                        objective=objective,
                        structural_sensor_valid=relation.structural_valid,
                        sample_keys=(item.group.stateful_sample_key,),
                        merge_size=merge_size,
                    )
                    if len(visuals) != 1:
                        raise RuntimeError("fixed-X evaluation rendered another sample count")
                    variant_results.append(
                        {
                            "variant": variant.as_dict(),
                            "instruction_sha256": variant.instruction_sha256,
                            "own_target_token_evidence": runtime_evidence.token_evidence,
                            "alternate_target_token_evidence": alternate_token,
                            "task_row_diagnostic": runtime_evidence.task_row_diagnostic,
                            "ownership_rows": [
                                dict(row) for row in runtime_evidence.ownership_rows
                            ],
                            "ownership_summary": runtime_evidence.ownership_summary,
                            "relation_sha256": native_relation_output_sha256(relation),
                            "target_sha256": runtime_evidence.target_sha256,
                            "forward_seconds": forward_seconds[variant_index],
                            "visual_artifact": visuals[0],
                        }
                    )
                sample = build_fixed_observation_evaluation_sample(
                    checkpoint_global_step=checkpoint_global_step,
                    item=item,
                    mass_stratum=strata[(item.partition, item.ordinal)],
                    variant_results=variant_results,
                    source_digest=batches[0].source_digest,
                    non_language_model_inputs_sha256=non_language_sha256,
                    language_model_inputs_sha256=language_sha256,
                    peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                )
            except BaseException as error:
                evidence_error = error
            _distributed_phase_error(
                error=evidence_error,
                phase=f"fixed-X-sample-{local_index}-evidence",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if sample is None:
                raise RuntimeError("fixed-X evaluation evidence vanished")
            local_samples.append(sample)
            del batches, contexts, evidence_error, forward_seconds, planned, prepared, sample

    if local_forward_equivalence_probe is None:
        raise RuntimeError("fixed-X forward-equivalence probe vanished")
    rank_action_state_sha256: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(rank_action_state_sha256, action_state_sha256)
    distributed_action_state_sha256 = _canonical_sha256(
        {
            "rank_local_action_state_sha256": rank_action_state_sha256,
            "world_size": world_size,
        }
    )
    gathered: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(gathered, local_samples)
    gathered_forward_equivalence: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(
        gathered_forward_equivalence,
        local_forward_equivalence_probe,
    )

    snapshot: dict[str, object] | None = None
    publish_error: list[str | None] = [None]
    if rank == 0:
        try:
            by_key: dict[tuple[str, int], dict[str, object]] = {}
            for shard in gathered:
                if not isinstance(shard, list):
                    raise RuntimeError("fixed-X evaluation sample shard is malformed")
                for sample in shard:
                    if not isinstance(sample, dict):
                        raise RuntimeError("fixed-X evaluation sample is malformed")
                    raw_item = sample.get("item")
                    parsed = FixedObservationEvaluationItem.from_dict(raw_item)
                    key = (parsed.partition, parsed.ordinal)
                    if key in by_key:
                        raise RuntimeError("fixed-X evaluation gathered duplicate sample evidence")
                    by_key[key] = cast(dict[str, object], sample)
            ordered = [by_key[(item.partition, item.ordinal)] for item in plan.items]
            snapshot = build_fixed_observation_evaluation_snapshot(
                checkpoint_global_step=checkpoint_global_step,
                implementation_sha256=implementation_sha256,
                model_family_sha256=model_family_sha256,
                representation_split_sha256=representation_split_sha256,
                plan=plan,
                representation_frozen_action_state_sha256=(distributed_action_state_sha256),
                samples=ordered,
                forward_equivalence_probes=gathered_forward_equivalence,
            )
            validate_fixed_observation_evaluation_visual_files(
                snapshot,
                plan=plan,
                output_root=output_root,
            )
            _write_snapshot(
                output_root / "fixed_observation_evaluation_snapshot.json",
                snapshot,
            )
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist_module.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"fixed-X evaluation publication failed: {publish_error[0]}")
    dist_module.barrier()
    return snapshot


def run_representation_checkpoint_evaluation(
    policy: nn.Module,
    *,
    expected_scope: NativeRepresentationParameterScope,
    plan: RepresentationEvaluationPlan,
    checkpoint_global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    representation_split_sha256: str,
    dataset: CalvinStatefulTransitionDataset,
    collate_planned: Callable[[PlannedNativeCALVINReplayBatch], CollatedNativeCALVINBatch],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    structural_config: NativeStructuralLossConfig,
    minimum_supervised_fraction: float,
    output_root: Path,
    rank: int,
    world_size: int,
    dist_module: Any,
    device: torch.device,
) -> dict[str, object] | None:
    """Evaluate one immutable checkpoint on factual and counterfactual reset frames."""

    if not isinstance(plan, RepresentationEvaluationPlan) or world_size != plan.world_size:
        raise ValueError("representation evaluation plan differs from distributed topology")
    if rank < 0 or rank >= world_size or checkpoint_global_step < 0:
        raise ValueError("representation evaluation rank or checkpoint step is invalid")
    if not callable(collate_planned):
        raise TypeError("representation evaluation collator must be callable")
    item_by_sample = {item.sample_key: item for item in plan.items}
    for item in plan.items:
        factual = tuple(task_identity_resolver(item.task_key) or ())
        shuffled_task = tuple(
            task_identity_resolver(item_by_sample[item.shuffled_task_sample_key].task_key) or ()
        )
        shuffled_target = tuple(
            task_identity_resolver(item_by_sample[item.shuffled_target_sample_key].task_key) or ()
        )
        if (
            factual != item.factual_target_identity_keys
            or shuffled_task != item.shuffled_task_target_identity_keys
            or shuffled_target != item.shuffled_target_target_identity_keys
        ):
            raise ValueError(
                "representation evaluation target identities differ from the frozen plan"
            )
    local_items = tuple(item for item in plan.items if item.rank == rank)
    if not local_items:
        raise ValueError("representation evaluation rank has no planned samples")
    counts: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(counts, len(local_items))
    if len(set(counts)) != 1:
        raise ValueError("representation evaluation ranks have unequal sample counts")

    setup_error: list[str | None] = [None]
    if rank == 0:
        try:
            if output_root.exists() or output_root.is_symlink():
                raise FileExistsError(
                    f"representation evaluation output root exists: {output_root}"
                )
            output_root.mkdir(parents=True)
        except BaseException as error:
            setup_error[0] = f"{type(error).__name__}: {error}"
    dist_module.broadcast_object_list(setup_error, src=0)
    if setup_error[0] is not None:
        raise RuntimeError(f"representation evaluation setup failed: {setup_error[0]}")
    dist_module.barrier()

    pending: list[_PendingRepresentationEvaluationSample] = []
    action_state_sha256 = ""
    with _distributed_action_diagnostic_transaction(
        policy,
        expected_scope=expected_scope,
        rank=rank,
        world_size=world_size,
        dist_module=dist_module,
    ) as action_guard:
        action_state_sha256 = action_guard.action_state_sha256
        for local_index, item in enumerate(local_items):
            prepared = None
            prepare_error: BaseException | None = None
            try:
                prepared = _prepare_evaluation_batches(
                    plan=plan,
                    item=item,
                    checkpoint_global_step=checkpoint_global_step,
                    rank=rank,
                    dataset=dataset,
                    device=device,
                    collate_planned=collate_planned,
                )
            except BaseException as error:
                prepare_error = error
            _distributed_phase_error(
                error=prepare_error,
                phase=f"sample-{local_index}-prepare",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if prepared is None:
                raise RuntimeError("representation evaluation preparation vanished")
            (
                factual_planned,
                history_batches,
                factual_batch,
                shuffled_batch,
                factual_instruction_sha256,
                shuffled_instruction_sha256,
                capacity_seed,
            ) = prepared

            torch.cuda.reset_peak_memory_stats(device)
            factual_forward = None
            factual_seconds = 0.0
            forward_seed: int | None = None
            history_seed: int | None = None
            seed_error: BaseException | None = None
            try:
                forward_seed = _evaluation_forward_seed(plan, item)
                history_seed = (
                    None if not history_batches else _evaluation_history_seed(forward_seed)
                )
            except BaseException as error:
                seed_error = error
            _distributed_phase_error(
                error=seed_error,
                phase=f"sample-{local_index}-forward-seed",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if forward_seed is None:
                raise RuntimeError("representation evaluation forward seed vanished")

            previous_state: NativePosteriorState | None = None
            prior_row_bindings: RowBindings = ()
            if history_seed is not None:
                _seed_evaluation_forward(history_seed, device=device)
                previous_state, prior_row_bindings = _reconstruct_evaluation_prior(
                    policy,
                    history_batches=history_batches,
                    physical_sidecar=physical_sidecar,
                    capacity=capacity,
                    task_identity_resolver=task_identity_resolver,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    structural_config=structural_config,
                    minimum_supervised_fraction=minimum_supervised_fraction,
                    capacity_seed=capacity_seed,
                )
            previous_state_sha256 = (
                None if previous_state is None else tensor_sha256(previous_state.rows)
            )
            _seed_evaluation_forward(forward_seed, device=device)
            factual_forward, factual_seconds = _run_action_evaluation_forward(
                action_guard,
                batch=factual_batch,
                previous_state=previous_state,
                device=device,
            )
            factual_postcheck_error: BaseException | None = None
            try:
                if previous_state is not None and tensor_sha256(previous_state.rows) != (
                    previous_state_sha256
                ):
                    raise RuntimeError("factual evaluation mutated its warm prior")
            except BaseException as error:
                factual_postcheck_error = error
            _distributed_phase_error(
                error=factual_postcheck_error,
                phase=f"sample-{local_index}-factual-postcheck",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if factual_forward is None:
                raise RuntimeError("representation factual forward vanished")

            _seed_evaluation_forward(forward_seed, device=device)
            shuffled_forward, shuffled_seconds = _run_action_evaluation_forward(
                action_guard,
                batch=shuffled_batch,
                previous_state=previous_state,
                device=device,
            )
            shuffled_postcheck_error: BaseException | None = None
            try:
                if previous_state is not None and tensor_sha256(previous_state.rows) != (
                    previous_state_sha256
                ):
                    raise RuntimeError("shuffled evaluation mutated its shared warm prior")
            except BaseException as error:
                shuffled_postcheck_error = error
            _distributed_phase_error(
                error=shuffled_postcheck_error,
                phase=f"sample-{local_index}-shuffled-postcheck",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if shuffled_forward is None:
                raise RuntimeError("representation shuffled-task forward vanished")

            evidence_error: BaseException | None = None
            pending_sample: _PendingRepresentationEvaluationSample | None = None
            try:
                factual_relation = factual_forward.context.relation_output
                shuffled_relation = shuffled_forward.context.relation_output
                if not isinstance(factual_relation, RelationOutput) or not isinstance(
                    shuffled_relation,
                    RelationOutput,
                ):
                    raise RuntimeError("representation evaluation forward omitted relation output")
                factual_objective = _evaluation_objective(
                    factual_forward.context,
                    factual_batch,
                    physical_sidecar=physical_sidecar,
                    capacity=capacity,
                    task_identity_resolver=task_identity_resolver,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    structural_config=structural_config,
                    minimum_supervised_fraction=minimum_supervised_fraction,
                    capacity_seed=capacity_seed,
                    prior_row_bindings=prior_row_bindings,
                )
                shuffled_objective = _evaluation_objective(
                    shuffled_forward.context,
                    shuffled_batch,
                    physical_sidecar=physical_sidecar,
                    capacity=capacity,
                    task_identity_resolver=task_identity_resolver,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    structural_config=structural_config,
                    minimum_supervised_fraction=minimum_supervised_fraction,
                    capacity_seed=capacity_seed,
                    prior_row_bindings=prior_row_bindings,
                )
                _require_shared_assignment_gauge(
                    factual_objective,
                    shuffled_objective,
                    comparison="representation shuffled-task control",
                )
                factual_evidence = build_representation_runtime_evidence(
                    factual_objective,
                    structural_sensor_valid=factual_relation.structural_valid,
                    batch_index=0,
                )
                shuffled_evidence = build_representation_runtime_evidence(
                    shuffled_objective,
                    structural_sensor_valid=shuffled_relation.structural_valid,
                    batch_index=0,
                )
                if factual_evidence.target_sha256 != shuffled_evidence.target_sha256:
                    raise RuntimeError("shuffled-task evaluation changed the loss-side target")
                factual_relation_precision = build_relation_score_precision_evidence(
                    factual_relation,
                    factual_objective,
                    batch_index=0,
                )
                shuffled_task_relation_precision = build_relation_score_precision_evidence(
                    shuffled_relation,
                    shuffled_objective,
                    batch_index=0,
                )
                visuals = render_native_relation_visuals(
                    output_root=output_root,
                    global_step=checkpoint_global_step,
                    input_weight_global_step=checkpoint_global_step,
                    weight_boundary="checkpoint_evaluation",
                    rank=rank,
                    host_items=factual_planned.training.host_items,
                    model_inputs=factual_batch.model_inputs,
                    objective=factual_objective,
                    structural_sensor_valid=factual_relation.structural_valid,
                    sample_keys=(item.sample_key,),
                    merge_size=merge_size,
                )
                if len(visuals) != 1:
                    raise RuntimeError("representation evaluation rendered another sample count")
                pending_sample = _PendingRepresentationEvaluationSample(
                    item=item,
                    factual_instruction_sha256=factual_instruction_sha256,
                    shuffled_task_instruction_sha256=shuffled_instruction_sha256,
                    factual=factual_evidence,
                    shuffled_task=shuffled_evidence,
                    official_action_loss=float(
                        factual_forward.official_action_loss.detach().float().cpu()
                    ),
                    factual_forward_seconds=factual_seconds,
                    shuffled_task_forward_seconds=shuffled_seconds,
                    peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    factual_relation_sha256=native_relation_output_sha256(factual_relation),
                    shuffled_task_relation_sha256=native_relation_output_sha256(shuffled_relation),
                    factual_relation_precision=factual_relation_precision,
                    shuffled_task_relation_precision=shuffled_task_relation_precision,
                    visual_artifact=visuals[0],
                )
            except BaseException as error:
                evidence_error = error
            _distributed_phase_error(
                error=evidence_error,
                phase=f"sample-{local_index}-evidence",
                rank=rank,
                world_size=world_size,
                dist_module=dist_module,
            )
            if pending_sample is None:
                raise RuntimeError("representation evaluation evidence vanished")
            pending.append(pending_sample)
            del (
                factual_batch,
                factual_forward,
                factual_planned,
                history_batches,
                pending_sample,
                prepared,
                shuffled_batch,
                shuffled_forward,
            )

    rank_action_state_sha256: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(rank_action_state_sha256, action_state_sha256)
    distributed_action_state_sha256 = _canonical_sha256(
        {
            "rank_local_action_state_sha256": rank_action_state_sha256,
            "world_size": world_size,
        }
    )

    local_samples: list[dict[str, object]] = []
    local_precision_samples: list[dict[str, object]] = []
    control_assembly_error: BaseException | None = None
    try:
        for sample in pending:
            local_precision_samples.append(
                build_relation_score_precision_sample(
                    sample_key=sample.item.sample_key,
                    partition=sample.item.partition,
                    task_key=sample.item.task_key,
                    factual_relation_sha256=sample.factual_relation_sha256,
                    shuffled_task_relation_sha256=sample.shuffled_task_relation_sha256,
                    factual=sample.factual_relation_precision,
                    shuffled_task=sample.shuffled_task_relation_precision,
                )
            )
            shuffled_target_mass = _target_mass_for_identities(
                sample.factual,
                sample.item.shuffled_target_target_identity_keys,
            )
            shuffled_target_token = build_representation_token_evidence(
                _runtime_token_logits(sample.factual),
                shuffled_target_mass,
            )
            local_samples.append(
                build_representation_evaluation_sample(
                    checkpoint_global_step=checkpoint_global_step,
                    item=sample.item,
                    factual_task_instruction_sha256=sample.factual_instruction_sha256,
                    shuffled_task_instruction_sha256=sample.shuffled_task_instruction_sha256,
                    shuffled_target_instruction_sha256=(
                        sample.item.shuffled_target_instruction_sha256
                    ),
                    factual_token_evidence=sample.factual.token_evidence,
                    shuffled_task_token_evidence=sample.shuffled_task.token_evidence,
                    shuffled_target_token_evidence=shuffled_target_token,
                    factual_task_row_diagnostic=sample.factual.task_row_diagnostic,
                    shuffled_task_row_diagnostic=sample.shuffled_task.task_row_diagnostic,
                    factual_ownership_rows=sample.factual.ownership_rows,
                    factual_ownership_summary=sample.factual.ownership_summary,
                    shuffled_task_ownership_rows=sample.shuffled_task.ownership_rows,
                    shuffled_task_ownership_summary=sample.shuffled_task.ownership_summary,
                    official_action_loss=sample.official_action_loss,
                    factual_forward_seconds=sample.factual_forward_seconds,
                    shuffled_task_forward_seconds=sample.shuffled_task_forward_seconds,
                    peak_cuda_reserved_bytes=sample.peak_cuda_reserved_bytes,
                    factual_relation_sha256=sample.factual_relation_sha256,
                    factual_target_sha256=sample.factual.target_sha256,
                    shuffled_task_relation_sha256=sample.shuffled_task_relation_sha256,
                    shuffled_task_target_sha256=sample.shuffled_task.target_sha256,
                    shuffled_target_target_sha256=representation_target_mass_sha256(
                        sample.item.shuffled_target_target_identity_keys,
                        shuffled_target_mass,
                    ),
                    visual_artifact=sample.visual_artifact,
                )
            )
    except BaseException as error:
        control_assembly_error = error
    _distributed_phase_error(
        error=control_assembly_error,
        phase="control-assembly",
        rank=rank,
        world_size=world_size,
        dist_module=dist_module,
    )
    gathered_samples: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(gathered_samples, local_samples)
    gathered_precision_samples: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(gathered_precision_samples, local_precision_samples)

    snapshot: dict[str, object] | None = None
    publish_error: list[str | None] = [None]
    if rank == 0:
        try:
            by_key: dict[str, dict[str, object]] = {}
            for shard in gathered_samples:
                if not isinstance(shard, list):
                    raise RuntimeError("representation evaluation sample shard is malformed")
                for sample in shard:
                    sample_key = sample.get("sample_key") if isinstance(sample, dict) else None
                    if not isinstance(sample_key, str) or sample_key in by_key:
                        raise RuntimeError(
                            "representation evaluation gathered duplicate sample evidence"
                        )
                    by_key[sample_key] = sample
            ordered = [by_key[item.sample_key] for item in plan.items]
            precision_by_key: dict[str, dict[str, object]] = {}
            for shard in gathered_precision_samples:
                if not isinstance(shard, list):
                    raise RuntimeError("relation precision sample shard is malformed")
                for sample in shard:
                    sample_key = sample.get("sample_key") if isinstance(sample, dict) else None
                    if not isinstance(sample_key, str) or sample_key in precision_by_key:
                        raise RuntimeError(
                            "relation precision audit gathered duplicate sample evidence"
                        )
                    precision_by_key[sample_key] = sample
            ordered_precision = [precision_by_key[item.sample_key] for item in plan.items]
            snapshot = build_representation_evaluation_snapshot(
                checkpoint_global_step=checkpoint_global_step,
                implementation_sha256=implementation_sha256,
                model_family_sha256=model_family_sha256,
                representation_split_sha256=representation_split_sha256,
                representation_evaluation_plan=plan,
                representation_frozen_action_state_sha256=(distributed_action_state_sha256),
                samples=ordered,
            )
            validate_representation_evaluation_visual_files(
                snapshot,
                plan=plan,
                output_root=output_root,
            )
            precision_audit = build_relation_score_precision_audit(
                checkpoint_global_step=checkpoint_global_step,
                implementation_sha256=implementation_sha256,
                model_family_sha256=model_family_sha256,
                representation_split_sha256=representation_split_sha256,
                representation_evaluation_plan_sha256=plan.artifact_sha256,
                expected_sample_keys=[item.sample_key for item in plan.items],
                samples=ordered_precision,
            )
            _write_snapshot(
                output_root / "relation_score_precision_audit.json",
                precision_audit,
            )
            _write_snapshot(output_root / "representation_evaluation_snapshot.json", snapshot)
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist_module.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"representation evaluation publication failed: {publish_error[0]}")
    dist_module.barrier()
    return snapshot


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    local = to_local() if callable(to_local) else value
    if not isinstance(local, torch.Tensor):
        raise TypeError("representation action buffer has no tensor-local value")
    return local


def _action_buffer_snapshots(
    policy: nn.Module,
    manifest: tuple[NativeActionStateTensorDigest, ...],
) -> dict[str, torch.Tensor]:
    named_buffers = dict(policy.named_buffers())
    names = tuple(item.name for item in manifest if item.kind == "buffer")
    if not names or any(name not in named_buffers for name in names):
        raise RuntimeError("representation action-state manifest lost a buffer")
    return {name: _local_tensor(named_buffers[name]).detach().clone() for name in names}


def _restore_action_buffers(
    policy: nn.Module,
    snapshots: Mapping[str, torch.Tensor],
) -> None:
    named_buffers = dict(policy.named_buffers())
    if not set(snapshots) <= set(named_buffers):
        raise RuntimeError("representation action buffer disappeared during a diagnostic")
    with torch.no_grad():
        for name, snapshot in snapshots.items():
            local = _local_tensor(named_buffers[name])
            if (
                local.shape != snapshot.shape
                or local.dtype != snapshot.dtype
                or local.device != snapshot.device
            ):
                raise RuntimeError("representation action buffer metadata changed")
            local.copy_(snapshot)


@dataclass(frozen=True, slots=True)
class ReadOnlyActionDiagnostic:
    """A full official action forward with restored mutable runtime statistics."""

    result: Any
    changed_buffer_names: tuple[str, ...]
    content_unchanged_parameter_version_names: tuple[str, ...]
    action_state_sha256: str


class RepresentationActionDiagnosticGuard:
    """Amortize full action-state hashing across one evaluation transaction."""

    def __init__(
        self,
        policy: nn.Module,
        *,
        expected_scope: NativeRepresentationParameterScope,
        forward: Callable[..., Any] = run_native_policy_diagnostic_forward,
    ) -> None:
        if not isinstance(policy, nn.Module) or not isinstance(
            expected_scope, NativeRepresentationParameterScope
        ):
            raise TypeError("action diagnostic guard requires typed policy scope")
        if not callable(forward):
            raise TypeError("action diagnostic guard forward must be callable")
        self._policy = policy
        self._expected_scope = expected_scope
        self._forward = forward
        self._before: tuple[NativeActionStateTensorDigest, ...] | None = None
        self._parameter_baseline: dict[str, NativeActionStateTensorDigest] = {}
        self._buffer_snapshots: dict[str, torch.Tensor] = {}
        self._parameter_names = tuple(item.name for item in expected_scope.action_frozen)
        self._active = False

    def __enter__(self) -> RepresentationActionDiagnosticGuard:
        if self._active or self._before is not None:
            raise RuntimeError("action diagnostic guard cannot be reused")
        before = native_representation_frozen_action_state_manifest(
            self._policy,
            expected=self._expected_scope,
        )
        self._before = before
        self._parameter_baseline = {item.name: item for item in before if item.kind == "parameter"}
        if set(self._parameter_baseline) != set(self._parameter_names):
            raise RuntimeError("action diagnostic parameter baseline is incomplete")
        self._buffer_snapshots = _action_buffer_snapshots(self._policy, before)
        self._active = True
        return self

    def _parameter_versions(self) -> dict[str, int]:
        named_parameters = dict(self._policy.named_parameters())
        if any(name not in named_parameters for name in self._parameter_names):
            raise RuntimeError("action parameter disappeared during a diagnostic")
        return {
            name: int(_local_tensor(named_parameters[name])._version)
            for name in self._parameter_names
        }

    def _changed_parameter_values(
        self,
        version_changed_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        named_parameters = dict(self._policy.named_parameters())
        if any(name not in named_parameters for name in version_changed_names):
            raise RuntimeError("action parameter disappeared during a diagnostic")
        return tuple(
            name
            for name in version_changed_names
            if native_representation_action_state_tensor_digest(
                name,
                "parameter",
                named_parameters[name],
            )
            != self._parameter_baseline[name]
        )

    def _changed_buffers(self) -> tuple[str, ...]:
        named_buffers = dict(self._policy.named_buffers())
        if set(named_buffers) & set(self._buffer_snapshots) != set(self._buffer_snapshots):
            raise RuntimeError("action buffer disappeared during a diagnostic")
        changed: list[str] = []
        for name, snapshot in self._buffer_snapshots.items():
            local = _local_tensor(named_buffers[name])
            if (
                local.shape != snapshot.shape
                or local.dtype != snapshot.dtype
                or local.device != snapshot.device
            ):
                raise RuntimeError("action buffer metadata changed during a diagnostic")
            if not torch.equal(local, snapshot):
                changed.append(name)
        return tuple(sorted(changed))

    @property
    def action_state_sha256(self) -> str:
        if self._before is None:
            raise RuntimeError("action diagnostic guard has not entered")
        return native_representation_action_state_manifest_sha256(self._before)

    def run(
        self,
        *,
        model_inputs: Mapping[str, Any],
        context: Any,
    ) -> ReadOnlyActionDiagnostic:
        """Run one official action forward and restore its ephemeral buffers."""

        if not self._active:
            raise RuntimeError("action diagnostic guard is not active")
        versions_before = self._parameter_versions()
        result: Any = None
        forward_error: BaseException | None = None
        changed_buffers: tuple[str, ...] = ()
        changed_parameters: tuple[str, ...] = ()
        state_error: BaseException | None = None
        try:
            result = self._forward(
                self._policy,
                model_inputs=model_inputs,
                context=context,
            )
        except BaseException as error:
            forward_error = error
        finally:
            try:
                versions_after = self._parameter_versions()
                changed_parameters = tuple(
                    name
                    for name in self._parameter_names
                    if versions_after[name] != versions_before[name]
                )
                changed_parameter_values = self._changed_parameter_values(changed_parameters)
                changed_buffers = self._changed_buffers()
                unexpected_buffers = tuple(
                    name
                    for name in changed_buffers
                    if not name.endswith(_ACTION_DIAGNOSTIC_MUTABLE_BUFFER_SUFFIXES)
                )
                unexpected = (*changed_parameter_values, *unexpected_buffers)
                if unexpected:
                    raise RuntimeError(
                        "read-only action diagnostic changed non-ephemeral action state: "
                        + ",".join(unexpected)
                    )
            except BaseException as error:
                state_error = error
            finally:
                _restore_action_buffers(self._policy, self._buffer_snapshots)
        if state_error is not None:
            if forward_error is not None:
                raise state_error from forward_error
            raise state_error
        if forward_error is not None:
            raise forward_error
        return ReadOnlyActionDiagnostic(
            result=result,
            changed_buffer_names=changed_buffers,
            content_unchanged_parameter_version_names=changed_parameters,
            action_state_sha256=self.action_state_sha256,
        )

    def close(self) -> str:
        """Verify that the complete frozen action state matches the entry boundary."""

        if not self._active or self._before is None:
            raise RuntimeError("action diagnostic guard is not active")
        self._active = False
        restored = native_representation_frozen_action_state_manifest(
            self._policy,
            expected=self._expected_scope,
        )
        if restored != self._before:
            changed = native_representation_action_state_changes(self._before, restored)
            raise RuntimeError(
                "action diagnostic transaction changed frozen action state: " + ",".join(changed)
            )
        return native_representation_action_state_manifest_sha256(restored)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> Literal[False]:
        del exception_type, exception, traceback
        self.close()
        return False


def run_read_only_representation_action_diagnostic(
    policy: nn.Module,
    *,
    expected_scope: NativeRepresentationParameterScope,
    model_inputs: Mapping[str, Any],
    context: Any,
    forward: Callable[..., Any] = run_native_policy_diagnostic_forward,
) -> ReadOnlyActionDiagnostic:
    """Run the official action path without committing its MoE counters."""

    with RepresentationActionDiagnosticGuard(
        policy,
        expected_scope=expected_scope,
        forward=forward,
    ) as guard:
        diagnostic = guard.run(model_inputs=model_inputs, context=context)
    return diagnostic

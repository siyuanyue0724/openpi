#!/usr/bin/env python3
"""Run label-free ADR-157 G2 action receipts on the real four-rank host."""

from __future__ import annotations

# ruff: noqa: E402
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from tools.cuda_allocator_bootstrap import bootstrap_cuda_allocator

_BOOTSTRAPPED_CUDA_ALLOCATOR = bootstrap_cuda_allocator(sys.argv[1:])

import torch
import torch.distributed as dist

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native import calvin_entity_training
from picf_next.lingbot_native import training as native_training
from picf_next.lingbot_native.calvin import (
    NativeCALVINTrainingBatch,
    build_native_calvin_replay_batch,
)
from picf_next.lingbot_native.fixed_observation import calvin_source_state_sha256
from picf_next.lingbot_native.frozen_posterior_diagnostic import (
    FrozenPosteriorArm,
    FrozenPosteriorInterventionKind,
    FrozenPosteriorShapeContract,
    FrozenPosteriorVisibility,
    LabelFreePromptVariant,
    LanguagePromptBatch,
    capture_factual_posterior_snapshot,
    factual_frozen_posterior_arm,
    label_blind_visibility_removal_arms,
    run_frozen_posterior_action_diagnostic,
)
from picf_next.lingbot_native.frozen_posterior_lingbot_adapter import (
    run_native_frozen_posterior_action_forward,
)
from picf_next.lingbot_native.host import native_context_from_prior_trace
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.representation_evaluation_runtime import _same_runtime_value
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
)
from tools import run_lingbot_vla2_task_independent_full as runner

runner._BOOTSTRAPPED_CUDA_ALLOCATOR = _BOOTSTRAPPED_CUDA_ALLOCATOR

WORLD_SIZE = 4
RANK = int(os.environ["RANK"])
OUTPUT_ROOT = Path(os.environ["PICF_ADR157_G2_CAPTURE_ROOT"]).resolve()
PROGRESS_ROOT = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}.progress"
CONTRACT_PATH = Path(os.environ["PICF_ADR157_G2_EXECUTION_CONTRACT"]).resolve()
MODEL_CHECKPOINT = Path(os.environ["PICF_ADR157_G2_MODEL_CHECKPOINT"]).resolve()
EXPECTED_CONTRACT_SHA256 = os.environ["PICF_ADR157_G2_EXECUTION_CONTRACT_SHA256"]
CHECKPOINT_GLOBAL_STEP = int(os.environ.get("PICF_ADR157_G2_CHECKPOINT_STEP", "2000"))

if Path("/mnt") != OUTPUT_ROOT and Path("/mnt") not in OUTPUT_ROOT.parents:
    raise RuntimeError("ADR-157 G2 capture root must be persistent under /mnt")
if dist.is_available() and os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
    raise RuntimeError("ADR-157 G2 requires exactly four distributed ranks")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _emit_progress(event: str, **values: object) -> None:
    payload = {
        "event": event,
        "rank": RANK,
        "schema": "picf-next.adr157-g2-progress/v1",
        "time_ns": time.time_ns(),
        **values,
    }
    encoded = _canonical_bytes(payload)
    print(encoded.decode("ascii"), flush=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    write_bytes_durable_exclusive(PROGRESS_ROOT / f"rank-{RANK:02d}-{event}.json", encoded + b"\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    local = value.to_local() if callable(getattr(value, "to_local", None)) else value
    if not isinstance(local, torch.Tensor):
        raise TypeError("ADR-157 G2 tensor digest received a non-tensor")
    materialized = local.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(materialized.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(materialized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(materialized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _batch_shape_summary(batch: NativeCALVINTrainingBatch) -> dict[str, object]:
    model_inputs = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for name, value in sorted(batch.model_inputs.items())
        if isinstance(value, torch.Tensor)
    }
    modality_streams: list[dict[str, object]] = []
    if batch.modalities is not None:
        modality_streams = [
            {
                "name": stream.name,
                "token_shape": list(stream.tokens.shape),
                "valid_shape": list(stream.valid.shape),
            }
            for stream in batch.modalities.streams
        ]
    return {
        "model_inputs": model_inputs,
        "modality_streams": modality_streams,
    }


def _tensor_versions(policy: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    records: list[tuple[Any, ...]] = []
    for kind, values in (
        ("parameter", policy.named_parameters()),
        ("buffer", policy.named_buffers()),
    ):
        for name, value in values:
            records.append(
                (
                    kind,
                    name,
                    id(value),
                    int(value._version),
                    tuple(value.shape),
                    str(value.dtype),
                    str(value.device),
                )
            )
    return tuple(records)


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_recursive_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_recursive_keys(item) for item in value))
    return set()


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(CONTRACT_PATH) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("ADR-157 G2 execution contract file digest changed")
    value = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if not isinstance(value, dict) or set(value) != {
        "item_count",
        "items",
        "name",
        "provenance",
        "schema",
        "world_size",
    }:
        raise ValueError("ADR-157 G2 execution contract fields differ")
    if (
        value["schema"] != "picf-next.adr157-g2-label-free-execution/v1"
        or value["world_size"] != WORLD_SIZE
        or value["item_count"] != len(value["items"])
    ):
        raise ValueError("ADR-157 G2 execution contract header differs")
    forbidden = {"target_identity_key", "target_mass", "target_row", "sidecar", "labels"}
    retained = sorted(forbidden & _recursive_keys(value))
    if retained:
        raise ValueError(f"ADR-157 G2 execution contract retained target fields: {retained}")
    local = [item for item in value["items"] if item["execution_rank"] == RANK]
    if len(local) != 1:
        raise ValueError("ADR-157 G2 smoke requires exactly one pair per rank")
    item = local[0]
    if set(item) != {
        "execution_rank",
        "item_id",
        "ordinal",
        "partition",
        "prompts",
        "replay_seed",
        "sample_key",
        "source_global_index",
        "source_sensor_sha256",
        "source_state_sha256",
    }:
        raise ValueError("ADR-157 G2 execution item fields differ")
    if len(item["prompts"]) != 2:
        raise ValueError("ADR-157 G2 requires exactly two prompts")
    for prompt in item["prompts"]:
        if set(prompt) != {
            "instruction",
            "instruction_sha256",
            "name",
            "task_key",
        }:
            raise ValueError("ADR-157 G2 prompt fields differ")
        if (
            hashlib.sha256(prompt["instruction"].encode("utf-8")).hexdigest()
            != prompt["instruction_sha256"]
        ):
            raise ValueError("ADR-157 G2 prompt instruction digest changed")
    return value, item


CONTRACT, CONTRACT_ITEM = _load_contract()
RANK_ROOT = OUTPUT_ROOT / f"rank_{RANK}"
RANK_ROOT.mkdir(parents=True, exist_ok=False)


def _repository_commit() -> str:
    root = Path(runner.__file__).resolve().parents[1]
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime_locals_from_stack() -> dict[str, Any]:
    required = {
        "collate_planned",
        "canonical_dense_key_by_source_index",
        "dense_evidence_bank",
        "device",
        "evaluation_dataset",
        "merge_size",
        "patch_size",
        "restore_official_runtime_buffers",
        "snapshot_official_runtime_buffers",
    }
    frame = inspect.currentframe()
    while frame is not None:
        if required <= set(frame.f_locals):
            return {name: frame.f_locals[name] for name in required}
        frame = frame.f_back
    raise RuntimeError("ADR-157 G2 could not bind the production runtime resources")


def _label_free_variant_replay(source: Any, prompt: dict[str, Any]) -> Any:
    if source.training.routing.batch_size != 1 or len(source.training.host_items) != 1:
        raise ValueError("ADR-157 G2 requires one source sample per rank")
    host_item = copy.deepcopy(source.training.host_items[0])
    if not isinstance(host_item.get("task"), str):
        raise ValueError("ADR-157 G2 source host item has no natural task")
    host_item["task"] = prompt["instruction"]
    request = source.training.structural_target_requests[0]
    training = NativeCALVINTrainingBatch(
        host_items=(host_item,),
        controls=source.training.controls,
        routing=source.training.routing,
        structural_target_requests=(replace(request, task_key=prompt["task_key"]),),
        prior_control_chunks=source.training.prior_control_chunks,
        physical_control_span_sha256=source.training.physical_control_span_sha256,
        selected_segment_indices=source.training.selected_segment_indices,
    )
    return replace(source, training=training)


def _validate_source(dataset: Any, source: Any) -> None:
    request = source.training.structural_target_requests[0]
    if request.sample_key != CONTRACT_ITEM["sample_key"]:
        raise ValueError("ADR-157 G2 runtime sample key differs")
    if request.source_global_index != CONTRACT_ITEM["source_global_index"]:
        raise ValueError("ADR-157 G2 runtime source index differs")
    if request.source_sensor_hash_by_field != CONTRACT_ITEM["source_sensor_sha256"]:
        raise ValueError("ADR-157 G2 runtime sensor hashes differ")
    arrays = dataset.index.validated_source_frame_arrays(
        request.source_global_index,
        fields=("robot_obs", "scene_obs"),
    )
    state_sha256 = calvin_source_state_sha256(arrays["scene_obs"], arrays["robot_obs"])
    if state_sha256 != CONTRACT_ITEM["source_state_sha256"]:
        raise ValueError("ADR-157 G2 runtime source state differs")


def _validate_pair(first: Any, second: Any) -> tuple[str, tuple[str, str]]:
    first_request = first.structural_target_requests[0]
    second_request = second.structural_target_requests[0]
    prompts = CONTRACT_ITEM["prompts"]
    if (
        first.routing.batch_size != 1
        or second.routing.batch_size != 1
        or first.source_digest != second.source_digest
        or not _same_runtime_value(first.controls, second.controls)
        or not _same_runtime_value(
            first.effective_prior_control_chunks,
            second.effective_prior_control_chunks,
        )
        or first.routing != second.routing
        or not _same_runtime_value(first.modalities, second.modalities)
        or first_request != replace(second_request, task_key=first_request.task_key)
        or first_request.task_key != prompts[0]["task_key"]
        or second_request.task_key != prompts[1]["task_key"]
        or set(first.model_inputs) != set(second.model_inputs)
    ):
        raise ValueError("ADR-157 G2 fixed observation changed a non-language source")
    language_fields = {"lang_tokens", "lang_masks"}
    non_language_digest = hashlib.sha256()
    language_digests = []
    for batch in (first, second):
        language_digest = hashlib.sha256()
        for name in sorted(batch.model_inputs):
            value = batch.model_inputs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("ADR-157 G2 model input is not a tensor")
            digest = _tensor_sha256(value)
            if name in language_fields:
                language_digest.update(name.encode("ascii") + b"\0" + digest.encode("ascii"))
            elif batch is first:
                non_language_digest.update(name.encode("ascii") + b"\0" + digest.encode("ascii"))
            elif not torch.equal(value, first.model_inputs[name]):
                raise ValueError(f"ADR-157 G2 changed non-language field {name!r}")
        language_digests.append(language_digest.hexdigest())
    if language_digests[0] == language_digests[1]:
        raise ValueError("ADR-157 G2 tokenized prompts did not change")
    return non_language_digest.hexdigest(), (language_digests[0], language_digests[1])


def _relation_payload(relation: PhysicalRelationOutput) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields(relation):
        value = getattr(relation, field.name)
        if isinstance(value, torch.Tensor):
            result[field.name] = value.detach().to(device="cpu")
        else:
            result[field.name] = value
    return result


def _durable_torch_save(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    write_bytes_durable_exclusive(path, _canonical_bytes(value) + b"\n")


_original_diagnostic = native_training.run_native_policy_diagnostic_forward
_original_observation = native_training.run_native_policy_observation_diagnostic_forward
_original_training = native_training.run_native_policy_training_forward
_original_optimizer_builder = runner.build_lingbot_official_optimizer
_executed = False
_checkpoint_restored = False


def _restore_then_build_official_optimizer(
    policy: torch.nn.Module,
    optimizer_contract: object,
    **kwargs: object,
) -> object:
    """Restore the measured model before any prior, observation or action forward."""

    global _checkpoint_restored
    optimizer = _original_optimizer_builder(policy, optimizer_contract, **kwargs)
    if _checkpoint_restored:
        raise RuntimeError("ADR-157 G2 attempted to restore its checkpoint more than once")

    from lingbotvla.checkpoint import build_checkpointer

    _emit_progress("model_checkpoint_load_started", checkpoint_step=CHECKPOINT_GLOBAL_STEP)
    build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp").load(
        str(MODEL_CHECKPOINT),
        {"model": policy},
        allow_partial_load=False,
    )
    _emit_progress("model_checkpoint_dcp_returned", checkpoint_step=CHECKPOINT_GLOBAL_STEP)
    torch.cuda.synchronize(device=torch.cuda.current_device())
    _emit_progress("model_checkpoint_cuda_synchronized", checkpoint_step=CHECKPOINT_GLOBAL_STEP)
    _checkpoint_restored = True
    runner.build_lingbot_official_optimizer = _original_optimizer_builder
    return optimizer


def _restore_forward_bindings() -> None:
    native_training.run_native_policy_diagnostic_forward = _original_diagnostic
    native_training.run_native_policy_training_forward = _original_training
    calvin_entity_training.run_native_policy_diagnostic_forward = _original_diagnostic
    calvin_entity_training.run_native_policy_training_forward = _original_training


def _instrumented_diagnostic(policy: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    global _executed
    if _executed:
        return _original_diagnostic(policy, *args, **kwargs)
    _executed = True
    _restore_forward_bindings()
    if not policy.training:
        raise RuntimeError("ADR-157 G2 must enter through the production training root")
    if not _checkpoint_restored:
        raise RuntimeError("ADR-157 G2 reached a forward before restoring its model checkpoint")
    runtime = _runtime_locals_from_stack()
    device = runtime["device"]
    dataset = runtime["evaluation_dataset"]
    collate_planned = runtime["collate_planned"]
    snapshot_buffers = runtime["snapshot_official_runtime_buffers"]
    restore_buffers = runtime["restore_official_runtime_buffers"]
    started = time.perf_counter()

    if runtime["dense_evidence_bank"] is None:
        raise RuntimeError("ADR-157 G2 requires the complete frozen modality cache bank")
    covered = runtime["canonical_dense_key_by_source_index"]
    missing = sorted(
        item["source_global_index"]
        for item in CONTRACT["items"]
        if item["source_global_index"] not in covered
    )
    if missing:
        raise RuntimeError(f"ADR-157 G2 contract lacks dense evidence for source indices {missing}")
    _emit_progress("dense_evidence_preflight_finished")

    _emit_progress("fixed_observation_started")
    _emit_progress("fixed_source_build_started")
    source = build_native_calvin_replay_batch(
        dataset,
        sample_key=CONTRACT_ITEM["sample_key"],
        lane_id=RANK,
        episode_instance_id=f"adr157-g2/{CONTRACT_ITEM['item_id']}",
        optimizer_step=CHECKPOINT_GLOBAL_STEP,
        replay_seed=CONTRACT_ITEM["replay_seed"],
        device=device,
        dtype=torch.bfloat16,
    )
    _emit_progress("fixed_source_build_finished")
    _validate_source(dataset, source)
    planned = tuple(
        _label_free_variant_replay(source, prompt) for prompt in CONTRACT_ITEM["prompts"]
    )
    _emit_progress("fixed_collation_started")
    local_collation_error: BaseException | None = None
    collated_batches = []
    try:
        for prompt_index, item in enumerate(planned):
            _emit_progress(f"prompt_{prompt_index}_collation_started")
            collated_batches.append(collate_planned(item))
            _emit_progress(f"prompt_{prompt_index}_collation_finished")
    except BaseException as error:
        local_collation_error = error
    _emit_progress("fixed_collation_consensus_started")
    runner._distributed_raise_if_local_probe_error(
        dist=dist,
        rank=RANK,
        world_size=WORLD_SIZE,
        stage="ADR-157 G2 fixed-observation collation",
        local_error=local_collation_error,
    )
    _emit_progress("fixed_collation_consensus_finished")
    batches = tuple(collated_batches)
    _emit_progress(
        "fixed_collation_finished",
        prompt_batches=[_batch_shape_summary(batch) for batch in batches],
    )
    non_language_sha256, language_sha256 = _validate_pair(batches[0], batches[1])
    _emit_progress("fixed_pair_validation_finished")
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    if not graph.unified_predict_correct:
        raise RuntimeError("ADR-157 G2 requires the production two-pass PICF graph")
    local_prior_steps = len(batches[0].effective_prior_control_chunks)
    _emit_progress("prior_schedule_started", local_prior_steps=local_prior_steps)
    prior_host_steps = runner._distributed_prior_host_step_schedule(
        (local_prior_steps,),
        device=device,
        dist=dist,
        torch_module=torch,
    )[0]
    _emit_progress(
        "prior_schedule_finished",
        local_prior_steps=local_prior_steps,
        prior_host_steps=prior_host_steps,
    )
    _emit_progress("prior_buffer_snapshot_started")
    saved_prior_buffers = snapshot_buffers()
    _emit_progress("prior_buffer_snapshot_finished")
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(CONTRACT_ITEM["replay_seed"])
            torch.cuda.manual_seed(CONTRACT_ITEM["replay_seed"])
            _emit_progress("prior_forward_started")
            prior_trace, _prediction = native_training.run_native_v3_prior_chain(
                policy,
                graph=graph,
                previous_memory=None,
                previous_memory_valid=torch.zeros(
                    batches[0].routing.batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
                control_chunks=batches[0].effective_prior_control_chunks,
                filter_prediction=None,
                require_attached_memory=False,
                host_step_count=prior_host_steps,
                require_grad=False,
            )
            torch.cuda.synchronize(device=device)
            _emit_progress("prior_forward_finished")
    finally:
        _emit_progress("prior_buffer_restore_started")
        restore_buffers(saved_prior_buffers)
        _emit_progress("prior_buffer_restore_finished")
    if not isinstance(prior_trace, NativeLayerwisePriorTrace):
        raise RuntimeError("ADR-157 G2 prior pass omitted its layerwise trace")
    capacity_seed = int.from_bytes(
        hashlib.sha256(f"{CONTRACT_ITEM['replay_seed']}\0capacity".encode("ascii")).digest()[:8],
        "big",
    )

    contexts = []
    persistent_states = []
    for prompt_index, batch in enumerate(batches):
        event_prefix = f"prompt_{prompt_index}"
        _emit_progress(f"{event_prefix}_buffer_snapshot_started")
        saved_buffers = snapshot_buffers()
        _emit_progress(f"{event_prefix}_buffer_snapshot_finished")
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(CONTRACT_ITEM["replay_seed"])
                torch.cuda.manual_seed(CONTRACT_ITEM["replay_seed"])
                _emit_progress(f"{event_prefix}_observation_forward_started")
                context = _original_observation(
                    policy,
                    model_inputs=batch.model_inputs,
                    context=native_context_from_prior_trace(
                        controls=batch.controls,
                        prior_trace=prior_trace,
                        modalities=batch.modalities,
                    ),
                )
                torch.cuda.synchronize(device=device)
                _emit_progress(f"{event_prefix}_observation_forward_finished")
        finally:
            _emit_progress(f"{event_prefix}_buffer_restore_started")
            restore_buffers(saved_buffers)
            _emit_progress(f"{event_prefix}_buffer_restore_finished")
        _emit_progress(f"{event_prefix}_persistent_extract_started")
        persistent = native_training.native_persistent_output(context)
        if not isinstance(persistent, NativeLayerwisePosteriorState):
            raise RuntimeError(
                f"ADR-157 G2 prompt {prompt_index} did not return layerwise posterior memory"
            )
        if not isinstance(context.relation_output, PhysicalRelationOutput):
            raise RuntimeError("ADR-157 G2 observation omitted physical relation output")
        contexts.append(context)
        persistent_states.append(persistent.detached())
        _emit_progress(f"{event_prefix}_persistent_extract_finished")
    _emit_progress("fixed_observation_finished")

    shape_contract = FrozenPosteriorShapeContract(
        num_layers=graph.config.num_layers,
        capacity=graph.config.capacity,
        host_width=graph.config.host_width,
    )
    snapshot = capture_factual_posterior_snapshot(
        lambda: persistent_states[0],
        shape_contract=shape_contract,
        provenance_id=(f"{CONTRACT_ITEM['item_id']}/prompt-0/checkpoint-{CHECKPOINT_GLOBAL_STEP}"),
    )
    prompts = tuple(
        LabelFreePromptVariant(
            name=prompt["name"],
            language=LanguagePromptBatch(
                token_ids=batch.model_inputs["lang_tokens"].detach().clone(),
                token_valid=batch.model_inputs["lang_masks"].detach().bool().clone(),
            ),
        )
        for prompt, batch in zip(CONTRACT_ITEM["prompts"], batches, strict=True)
    )
    row_arms = label_blind_visibility_removal_arms(snapshot)
    factual_arm = factual_frozen_posterior_arm(snapshot)
    factual_repeat_arm = FrozenPosteriorArm(
        name="factual-repeat",
        kind=FrozenPosteriorInterventionKind.FACTUAL,
        state=NativeLayerwisePosteriorState(snapshot.state.layer_rows.detach().clone()),
        row_visible=factual_arm.row_visible.clone(),
    )
    arms = (factual_arm, factual_repeat_arm, *row_arms)
    flow = policy.model
    action_projection = flow.action_out_proj
    action_projection_calls = 0

    def count_action_projection(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal action_projection_calls
        action_projection_calls += 1

    policy.eval()
    versions_before = _tensor_versions(policy)
    hook = action_projection.register_forward_hook(count_action_projection)
    try:
        result = run_frozen_posterior_action_diagnostic(
            lambda request: run_native_frozen_posterior_action_forward(policy, request=request),
            snapshot=snapshot,
            prompts=prompts,
            controls=batches[0].controls,
            proprioception=batches[0].model_inputs["state"],
            inference_noise=batches[0].model_inputs["noise"],
            arms=arms,
            visibility_contracts=(FrozenPosteriorVisibility.BOTH,),
        )
    finally:
        hook.remove()
    if versions_before != _tensor_versions(policy):
        raise RuntimeError("ADR-157 G2 action-only reads mutated a parameter or buffer")
    expected_action_projection_calls = len(prompts) * len(arms) * int(flow.config.num_steps)
    if action_projection_calls != expected_action_projection_calls:
        raise RuntimeError(
            "ADR-157 G2 did not use the released action projection for every denoising step: "
            f"{action_projection_calls} != {expected_action_projection_calls}"
        )

    actions: dict[str, torch.Tensor] = {}
    receipts = []
    for index, receipt in enumerate(result.receipts):
        action_key = f"action-{index:03d}"
        actions[action_key] = receipt.action.detach().to(device="cpu")
        receipts.append(
            {
                "action_key": action_key,
                "action_sha256": _tensor_sha256(receipt.action),
                "arm_name": receipt.arm_name,
                "inference_noise_sha256": receipt.inference_noise_sha256,
                "posterior_sha256": receipt.posterior_sha256,
                "prompt_name": receipt.prompt_name,
                "request_sha256": receipt.request_sha256,
                "row_visibility_sha256": receipt.row_visibility_sha256,
                "visibility": receipt.visibility.value,
            }
        )
    receipt_actions = {
        (receipt.prompt_name, receipt.arm_name): receipt.action.float()
        for receipt in result.receipts
    }
    repeat_floor_by_prompt = {}
    for prompt in prompts:
        factual = receipt_actions[(prompt.name, factual_arm.name)]
        repeated = receipt_actions[(prompt.name, factual_repeat_arm.name)]
        difference = repeated - factual
        repeat_floor_by_prompt[prompt.name] = {
            "exact_equal": bool(torch.equal(factual, repeated)),
            "max_abs": float(difference.abs().max().item()),
            "relative_l2": float((difference.norm() / factual.norm().clamp_min(1e-8)).item()),
            "rms": float(difference.square().mean().sqrt().item()),
        }
    action_path = RANK_ROOT / "actions.pt"
    _durable_torch_save(action_path, actions)
    first_rows = persistent_states[0].layer_rows.float()
    second_rows = persistent_states[1].layer_rows.float()
    posterior_prompt_relative_l2 = float(
        ((second_rows - first_rows).norm() / first_rows.norm().clamp_min(1e-8)).item()
    )
    execution_report = {
        "actions_file": action_path.name,
        "actions_file_sha256": _sha256(action_path),
        "capacity": snapshot.state.capacity,
        "capacity_seed": capacity_seed,
        "checkpoint_global_step": CHECKPOINT_GLOBAL_STEP,
        "execution_contract_file_sha256": EXPECTED_CONTRACT_SHA256,
        "factual_snapshot_sha256": result.factual_snapshot_sha256,
        "action_projection_calls": action_projection_calls,
        "expected_action_projection_calls": expected_action_projection_calls,
        "implementation_sha256": runner._implementation_digest(Path(runner.__file__).parents[1]),
        "inference_noise_sha256": result.inference_noise_sha256,
        "item_id": CONTRACT_ITEM["item_id"],
        "language_sha256": list(language_sha256),
        "model_checkpoint": str(MODEL_CHECKPOINT),
        "non_language_sha256": non_language_sha256,
        "partition": CONTRACT_ITEM["partition"],
        "posterior_prompt_relative_l2": posterior_prompt_relative_l2,
        "posterior_sha256_by_prompt": [
            _tensor_sha256(item.layer_rows) for item in persistent_states
        ],
        "prompt_names": [item.name for item in prompts],
        "rank": RANK,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "repeat_floor_by_prompt": repeat_floor_by_prompt,
        "repository_commit": _repository_commit(),
        "sample_key": CONTRACT_ITEM["sample_key"],
        "schema": "picf-next.adr157-g2-label-free-action-receipt/v1",
        "status": "ACTION_RECEIPTS_SEALED",
        "parameter_buffer_versions_unchanged": True,
    }
    _write_json(RANK_ROOT / "execution.json", execution_report)

    offline_payload = {
        "capacity_seed": capacity_seed,
        "item_id": CONTRACT_ITEM["item_id"],
        "model_inputs": {
            name: batches[0].model_inputs[name].detach().to(device="cpu")
            for name in ("images", "img_masks", "image_grid_thw")
        },
        "relation_by_prompt": [_relation_payload(context.relation_output) for context in contexts],
        "structural_target_request": {
            field.name: getattr(batches[0].structural_target_requests[0], field.name)
            for field in fields(batches[0].structural_target_requests[0])
        },
    }
    offline_path = RANK_ROOT / "offline_inputs.pt"
    _durable_torch_save(offline_path, offline_payload)
    offline_report = {
        "execution_file_sha256": _sha256(RANK_ROOT / "execution.json"),
        "item_id": CONTRACT_ITEM["item_id"],
        "offline_inputs_file": offline_path.name,
        "offline_inputs_file_sha256": _sha256(offline_path),
        "schema": "picf-next.adr157-g2-offline-input-receipt/v1",
        "status": "SEALED_AFTER_ACTIONS",
    }
    _write_json(RANK_ROOT / "offline_receipt.json", offline_report)

    rank_summary = {
        "action_receipts_sealed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "item_id": CONTRACT_ITEM["item_id"],
        "offline_inputs_sealed_after_actions": True,
        "rank": RANK,
        "receipt_count": len(receipts),
    }
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rank_summary)
    if RANK == 0:
        aggregate = {
            "execution_contract_file_sha256": EXPECTED_CONTRACT_SHA256,
            "rank_reports": gathered,
            "schema": "picf-next.adr157-g2-label-free-action-aggregate/v1",
            "status": "ACTION_RECEIPTS_SEALED",
            "world_size": dist.get_world_size(),
        }
        _write_json(OUTPUT_ROOT / "aggregate.json", aggregate)
    dist.barrier()
    os._exit(0)


native_training.run_native_policy_diagnostic_forward = _instrumented_diagnostic
native_training.run_native_policy_training_forward = _instrumented_diagnostic
calvin_entity_training.run_native_policy_diagnostic_forward = _instrumented_diagnostic
calvin_entity_training.run_native_policy_training_forward = _instrumented_diagnostic
runner.build_lingbot_official_optimizer = _restore_then_build_official_optimizer
runner.TWO_PASS_FILTER_DIAGNOSTIC_STEPS = (1,)

if __name__ == "__main__":
    runner.main()

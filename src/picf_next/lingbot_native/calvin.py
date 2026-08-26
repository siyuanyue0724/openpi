"""Fail-closed CALVIN bridge for the ADR-74 LingBot-native graph."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinPhysicalSample,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
    CalvinStatefulTransitionSample,
)
from picf_next.data.calvin_target_request import (
    NativeCALVINStructuralTargetRequest,
    native_calvin_structural_target_request,
)
from picf_next.data.lingbot_calvin import (
    map_calvin_action_to_lingbot,
    map_calvin_transition_to_lingbot,
)
from picf_next.lingbot_native.controls import (
    ExecutedControlBatch,
    executed_control_chain_reset,
)
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    native_context_from_persistent_state,
)
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalityStream,
    merge_native_modality_batches,
)
from picf_next.lingbot_native.prediction import NativePredictionRequest
from picf_next.lingbot_native.state import NativePersistentState
from picf_next.lingbot_wla_calvin import WLACalvinTargetBatch
from picf_next.training.control import (
    EpisodeSampleSequence,
    EpisodeStreamPlan,
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
    FrozenSamplePlan,
    PlannedMicrobatch,
    PlannedSample,
    PlannedStreamMicrobatch,
    TrainingPlan,
    derive_subseed,
)

_CONTINUATION_SEED_STREAM = "picf-next.lingbot-native-continuation/v1"
_REPLAY_SEED_STREAM = "picf-next.lingbot-native-replay/v1"
_PHYSICAL_PROMPT_SELECTION = "picf-next.calvin-physical-prompt-overlay/v1"

_RAW_HOST_FIELDS = frozenset(
    {
        "action.lingbot",
        "action.lingbot_is_pad",
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
        "observation.state.lingbot",
        "task",
    }
)
_TRANSFORM_FIELDS = frozenset(
    {
        "action_is_pad",
        "action_joint_mask",
        "actions",
        "image_grid_thw",
        "images",
        "img_masks",
        "joint_mask",
        "lang_masks",
        "lang_tokens",
        "state",
        "state_joint_mask",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "action_is_pad",
        "actions",
        "image_grid_thw",
        "images",
        "img_masks",
        "joint_mask",
        "lang_masks",
        "lang_tokens",
        "state",
    }
)
_RANDOM_FIELDS = frozenset({"noise", "time"})
_FORBIDDEN_KEY_PARTS = frozenset(
    {
        "bbox",
        "bounding_box",
        "episode_key",
        "frame_index",
        "object_id",
        "sample_key",
        "scene_obs",
        "scene_state",
        "segmentation_target",
        "simulator_id",
        "success",
        "track_id",
        "transition_index",
    }
)


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _audit_names(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{path} contains a non-string or empty field name")
        normalized = key.lower().replace(".", "_").replace("-", "_")
        if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
            raise ValueError(f"forbidden privileged or identity field at {path}.{key}")
        _audit_names(child, path=f"{path}.{key}")


@dataclass(frozen=True, slots=True)
class NativeCALVINRouting:
    """Scheduler identity retained outside every learned model input."""

    lane_ids: tuple[int, ...]
    episode_keys: tuple[str, ...]
    frame_indices: tuple[int, ...]
    reset: tuple[bool, ...]
    sample_keys: tuple[str, ...]
    optimizer_step: int

    def __post_init__(self) -> None:
        lengths = {
            len(self.lane_ids),
            len(self.episode_keys),
            len(self.frame_indices),
            len(self.reset),
            len(self.sample_keys),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("CALVIN routing fields must be equal non-empty sequences")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.lane_ids
        ):
            raise ValueError("CALVIN lane IDs must be non-negative integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.frame_indices
        ):
            raise ValueError("CALVIN frame indices must be non-negative integers")
        if any(not isinstance(value, bool) for value in self.reset):
            raise TypeError("CALVIN reset flags must be boolean")
        if any(not value for value in (*self.episode_keys, *self.sample_keys)):
            raise ValueError("CALVIN routing identities must be non-empty")
        if any(
            flag != (frame == 0) for flag, frame in zip(self.reset, self.frame_indices, strict=True)
        ):
            raise ValueError("CALVIN reset flags must match segment boundaries")
        if (
            isinstance(self.optimizer_step, bool)
            or not isinstance(self.optimizer_step, int)
            or self.optimizer_step < 0
        ):
            raise ValueError("optimizer step must be a non-negative integer")

    @property
    def batch_size(self) -> int:
        return len(self.lane_ids)


@dataclass(frozen=True, slots=True)
class NativeCALVINTrainingBatch:
    """Official raw host items plus the only legal recurrent control payload."""

    host_items: tuple[dict[str, Any], ...]
    controls: ExecutedControlBatch
    routing: NativeCALVINRouting
    structural_target_requests: tuple[NativeCALVINStructuralTargetRequest, ...]
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = ()
    physical_control_span_sha256: tuple[str, ...] = ()
    selected_segment_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            len(self.host_items) != self.controls.batch_size
            or len(self.host_items) != self.routing.batch_size
            or len(self.host_items) != len(self.structural_target_requests)
        ):
            raise ValueError("CALVIN host, control, routing and target batches differ")
        for item in self.host_items:
            if set(item) != _RAW_HOST_FIELDS:
                raise ValueError("CALVIN raw host item differs from the frozen LingBot schema")
            _audit_names(item, path="host_item")
        if tuple(request.sample_key for request in self.structural_target_requests) != (
            self.routing.sample_keys
        ):
            raise ValueError("CALVIN structural targets and routing sample identities differ")
        if self.prior_control_chunks:
            if self.prior_control_chunks[-1] is not self.controls:
                raise ValueError("the final prior-control chunk must be the correction control")
            for chunk in self.prior_control_chunks:
                if not isinstance(chunk, ExecutedControlBatch):
                    raise TypeError("prior controls must use ExecutedControlBatch")
                if (
                    chunk.batch_size != self.routing.batch_size
                    or chunk.action_dim != self.controls.action_dim
                    or chunk.values.device != self.controls.values.device
                    or chunk.values.dtype != self.controls.values.dtype
                ):
                    raise ValueError("prior-control chunks differ from the CALVIN batch contract")
        physical_lengths = {
            len(self.physical_control_span_sha256),
            len(self.selected_segment_indices),
        }
        if physical_lengths != {0} and physical_lengths != {self.routing.batch_size}:
            raise ValueError("physical CALVIN receipts must be absent or complete per batch")
        if self.physical_control_span_sha256:
            for digest in self.physical_control_span_sha256:
                _require_sha256("physical control span", digest)
            if not self.prior_control_chunks:
                raise ValueError("physical CALVIN receipts require exact prior-control chunks")
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in self.selected_segment_indices
            ):
                raise ValueError("selected CALVIN segment indices must be non-negative integers")

    @property
    def effective_prior_control_chunks(self) -> tuple[ExecutedControlBatch, ...]:
        return self.prior_control_chunks or (self.controls,)

    @property
    def prior_control_reset(self) -> torch.Tensor:
        return executed_control_chain_reset(self.effective_prior_control_chunks)


@dataclass(frozen=True, slots=True)
class PlannedNativeCALVINBatch:
    """Topology-neutral stream shard and all replayable stochastic sources."""

    training: NativeCALVINTrainingBatch
    plan_microbatch: PlannedStreamMicrobatch | PlannedMicrobatch
    plan_sha256: str
    augmentation_seeds: tuple[int, ...]
    flow_noise_seeds: tuple[int, ...]
    flow_timestep_seeds: tuple[int, ...]
    task_intervention_sha256: str | None = None
    fixed_observation_pair_sha256: str | None = None
    physical_prompt_selection_sha256: str | None = None
    physical_prompt_selection_receipts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256("plan_sha256", self.plan_sha256)
        if self.task_intervention_sha256 is not None:
            _require_sha256(
                "task_intervention_sha256",
                self.task_intervention_sha256,
            )
        if self.fixed_observation_pair_sha256 is not None:
            _require_sha256(
                "fixed_observation_pair_sha256",
                self.fixed_observation_pair_sha256,
            )
        if self.physical_prompt_selection_sha256 is not None:
            _require_sha256(
                "physical_prompt_selection_sha256",
                self.physical_prompt_selection_sha256,
            )
        for receipt in self.physical_prompt_selection_receipts:
            _require_sha256("physical_prompt_selection_receipt", receipt)
        if (
            self.task_intervention_sha256 is not None
            and self.fixed_observation_pair_sha256 is not None
        ):
            raise ValueError("planned CALVIN batch cannot combine prompt interventions")
        physical = bool(self.training.physical_control_span_sha256)
        if physical != (self.physical_prompt_selection_sha256 is not None) or physical != bool(
            self.physical_prompt_selection_receipts
        ):
            raise ValueError("physical CALVIN batches require one prompt-selection receipt")
        batch = self.training.routing.batch_size
        if not (
            len(self.augmentation_seeds)
            == len(self.flow_noise_seeds)
            == len(self.flow_timestep_seeds)
            == batch
        ):
            raise ValueError("planned stochastic seeds must align with the CALVIN batch")
        if physical and len(self.physical_prompt_selection_receipts) != batch:
            raise ValueError("physical prompt receipts must align with the CALVIN batch")

    @property
    def source_digest(self) -> str:
        payload = {
            "accumulation_index": self.plan_microbatch.accumulation_index,
            "augmentation_seeds": self.augmentation_seeds,
            "flow_noise_seeds": self.flow_noise_seeds,
            "flow_timestep_seeds": self.flow_timestep_seeds,
            "optimizer_step": self.plan_microbatch.optimizer_step,
            "plan_sha256": self.plan_sha256,
            "rank": self.plan_microbatch.rank,
            "sample_keys": self.training.routing.sample_keys,
            "world_size": self.plan_microbatch.world_size,
        }
        if self.training.physical_control_span_sha256:
            payload["physical_control_span_sha256"] = self.training.physical_control_span_sha256
            payload["selected_segment_indices"] = self.training.selected_segment_indices
            payload["physical_prompt_selection_sha256"] = self.physical_prompt_selection_sha256
            payload["physical_prompt_selection_receipts"] = (
                self.physical_prompt_selection_receipts
            )
        if self.task_intervention_sha256 is not None:
            payload["task_intervention_sha256"] = self.task_intervention_sha256
        if self.fixed_observation_pair_sha256 is not None:
            payload["fixed_observation_pair_sha256"] = self.fixed_observation_pair_sha256
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PlannedNativeCALVINContinuationBatch:
    """One target-free contiguous auxiliary step derived from a frozen batch."""

    training: NativeCALVINTrainingBatch
    parent_source_digest: str
    offset: int
    augmentation_seeds: tuple[int, ...]
    flow_noise_seeds: tuple[int, ...]
    flow_timestep_seeds: tuple[int, ...]
    task_intervention_sha256: str | None = None
    physical_prompt_selection_sha256: str | None = None
    physical_prompt_selection_receipts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256("parent_source_digest", self.parent_source_digest)
        if self.task_intervention_sha256 is not None:
            _require_sha256(
                "task_intervention_sha256",
                self.task_intervention_sha256,
            )
        if self.physical_prompt_selection_sha256 is not None:
            _require_sha256(
                "physical_prompt_selection_sha256",
                self.physical_prompt_selection_sha256,
            )
        for receipt in self.physical_prompt_selection_receipts:
            _require_sha256("physical_prompt_selection_receipt", receipt)
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset <= 0:
            raise ValueError("native continuation offset must be a positive integer")
        physical = bool(self.training.physical_control_span_sha256)
        if physical != (self.physical_prompt_selection_sha256 is not None) or physical != bool(
            self.physical_prompt_selection_receipts
        ):
            raise ValueError("physical CALVIN continuation lacks its prompt-selection receipt")
        batch = self.training.routing.batch_size
        if not (
            len(self.augmentation_seeds)
            == len(self.flow_noise_seeds)
            == len(self.flow_timestep_seeds)
            == batch
        ):
            raise ValueError("continuation stochastic seeds must align with the CALVIN batch")
        if physical and len(self.physical_prompt_selection_receipts) != batch:
            raise ValueError("continuation prompt receipts must align with the CALVIN batch")

    @property
    def source_digest(self) -> str:
        payload = {
            "augmentation_seeds": self.augmentation_seeds,
            "flow_noise_seeds": self.flow_noise_seeds,
            "flow_timestep_seeds": self.flow_timestep_seeds,
            "offset": self.offset,
            "optimizer_step": self.training.routing.optimizer_step,
            "parent_source_digest": self.parent_source_digest,
            "sample_keys": self.training.routing.sample_keys,
            "schema": _CONTINUATION_SEED_STREAM,
        }
        if self.training.physical_control_span_sha256:
            payload["physical_control_span_sha256"] = self.training.physical_control_span_sha256
            payload["selected_segment_indices"] = self.training.selected_segment_indices
            payload["physical_prompt_selection_sha256"] = self.physical_prompt_selection_sha256
            payload["physical_prompt_selection_receipts"] = (
                self.physical_prompt_selection_receipts
            )
        if self.task_intervention_sha256 is not None:
            payload["task_intervention_sha256"] = self.task_intervention_sha256
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PlannedNativeCALVINReplayBatch:
    """One deterministic no-grad step for an offline fixed-weight replay audit."""

    training: NativeCALVINTrainingBatch
    replay_seed: int
    augmentation_seeds: tuple[int, ...]
    flow_noise_seeds: tuple[int, ...]
    flow_timestep_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.replay_seed, bool)
            or not isinstance(self.replay_seed, int)
            or self.replay_seed < 0
        ):
            raise ValueError("native replay seed must be a non-negative integer")
        batch = self.training.routing.batch_size
        if not (
            len(self.augmentation_seeds)
            == len(self.flow_noise_seeds)
            == len(self.flow_timestep_seeds)
            == batch
        ):
            raise ValueError("replay stochastic seeds must align with the CALVIN batch")

    @property
    def source_digest(self) -> str:
        payload = {
            "augmentation_seeds": self.augmentation_seeds,
            "flow_noise_seeds": self.flow_noise_seeds,
            "flow_timestep_seeds": self.flow_timestep_seeds,
            "optimizer_step": self.training.routing.optimizer_step,
            "replay_seed": self.replay_seed,
            "sample_keys": self.training.routing.sample_keys,
            "schema": _REPLAY_SEED_STREAM,
        }
        if self.training.physical_control_span_sha256:
            payload["physical_control_span_sha256"] = self.training.physical_control_span_sha256
            payload["selected_segment_indices"] = self.training.selected_segment_indices
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CollatedNativeCALVINBatch:
    """Only official forward tensors; scheduler identity remains out of band."""

    model_inputs: Mapping[str, Any]
    controls: ExecutedControlBatch
    routing: NativeCALVINRouting
    source_digest: str
    structural_target_requests: tuple[NativeCALVINStructuralTargetRequest, ...]
    modalities: NativeModalityBatch | None = None
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = ()
    wla_world_target: WLACalvinTargetBatch | None = None

    def __post_init__(self) -> None:
        _require_sha256("source_digest", self.source_digest)
        if self.controls.batch_size != self.routing.batch_size:
            raise ValueError("collated CALVIN controls and routing differ")
        if len(self.structural_target_requests) != self.routing.batch_size:
            raise ValueError("collated CALVIN structural targets and routing differ")
        if tuple(request.sample_key for request in self.structural_target_requests) != (
            self.routing.sample_keys
        ):
            raise ValueError("collated CALVIN target and routing sample identities differ")
        if self.prior_control_chunks:
            if self.prior_control_chunks[-1] is not self.controls:
                raise ValueError("collated final prior-control chunk must be correction controls")
            if any(
                chunk.batch_size != self.routing.batch_size
                or chunk.action_dim != self.controls.action_dim
                or chunk.values.device != self.controls.values.device
                or chunk.values.dtype != self.controls.values.dtype
                for chunk in self.prior_control_chunks
            ):
                raise ValueError("collated prior-control chunks differ from the batch")
        if self.modalities is not None:
            if not isinstance(self.modalities, NativeModalityBatch):
                raise TypeError("collated modalities must use the native typed contract")
            if self.modalities.batch_size != self.routing.batch_size:
                raise ValueError("collated modalities and routing differ")
        if self.wla_world_target is not None:
            if not isinstance(self.wla_world_target, WLACalvinTargetBatch):
                raise TypeError("collated WLA world target must use its typed contract")
            if self.wla_world_target.images.shape[0] != self.routing.batch_size:
                raise ValueError("collated WLA target and routing batch axes differ")
        audit_native_calvin_model_inputs(self.model_inputs)

    @property
    def effective_prior_control_chunks(self) -> tuple[ExecutedControlBatch, ...]:
        return self.prior_control_chunks or (self.controls,)

    @property
    def prior_control_reset(self) -> torch.Tensor:
        return executed_control_chain_reset(self.effective_prior_control_chunks)


def with_native_modalities(
    batch: CollatedNativeCALVINBatch,
    modalities: NativeModalityBatch,
) -> CollatedNativeCALVINBatch:
    """Attach one complete typed observation set without changing model inputs."""

    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("native modality binding requires a collated CALVIN batch")
    if not isinstance(modalities, NativeModalityBatch):
        raise TypeError("native modality binding requires one typed modality batch")
    if modalities.batch_size != batch.routing.batch_size:
        raise ValueError("native modalities and collated CALVIN routing differ")
    merged = (
        modalities
        if batch.modalities is None
        else merge_native_modality_batches((batch.modalities, modalities))
    )
    return replace(batch, modalities=merged)


def with_wla_world_target(
    batch: CollatedNativeCALVINBatch,
    target: WLACalvinTargetBatch,
) -> CollatedNativeCALVINBatch:
    """Attach a future-image loss target without exposing it as model evidence."""

    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("WLA target binding requires a collated CALVIN batch")
    if not isinstance(target, WLACalvinTargetBatch):
        raise TypeError("WLA target binding requires its typed target batch")
    if batch.wla_world_target is not None:
        raise ValueError("collated CALVIN batch already contains a WLA world target")
    if target.images.shape[0] != batch.routing.batch_size:
        raise ValueError("WLA target batch axis differs from CALVIN routing")
    return replace(batch, wla_world_target=target)


def with_official_proprioception_modality(
    batch: CollatedNativeCALVINBatch,
) -> CollatedNativeCALVINBatch:
    """Expose the released 55D state to the shared observation host.

    The official action suffix remains unchanged. This adds one typed dense
    observation token so the recurrent posterior can condition on the same
    current proprioception that the released action expert already consumes.
    """

    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("proprioception binding requires a collated CALVIN batch")
    state = batch.model_inputs.get("state")
    if (
        not isinstance(state, torch.Tensor)
        or state.ndim != 2
        or state.shape[0] != batch.routing.batch_size
        or not state.is_floating_point()
    ):
        raise ValueError("official LingBot state must be floating [batch,width]")
    if not torch.isfinite(state).all():
        raise ValueError("official LingBot state contains NaN or infinity")
    proprioception = NativeModalityStream(
        name="proprioception",
        tokens=state.unsqueeze(1),
        valid=torch.ones(
            state.shape[0],
            1,
            dtype=torch.bool,
            device=state.device,
        ),
    )
    proprioception_batch = NativeModalityBatch((proprioception,))
    if batch.modalities is not None and any(
        stream.name == proprioception.name for stream in batch.modalities.streams
    ):
        raise ValueError("collated CALVIN batch already contains proprioception")
    return with_native_modalities(batch, proprioception_batch)


@contextmanager
def _isolated_augmentation_rng(seed: int):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("augmentation seeds must be uint64 integers")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=()):
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def audit_native_calvin_model_inputs(
    model_inputs: Mapping[str, Any],
    *,
    require_randomness: bool = False,
) -> None:
    """Reject every field outside the frozen official forward boundary."""

    if not isinstance(model_inputs, Mapping):
        raise TypeError("LingBot model inputs must be a mapping")
    _audit_names(model_inputs, path="model_inputs")
    keys = set(model_inputs)
    required = set(_MODEL_FIELDS)
    if require_randomness:
        required.update(_RANDOM_FIELDS)
    allowed = set(_MODEL_FIELDS | _RANDOM_FIELDS)
    missing = sorted(required - keys)
    unexpected = sorted(keys - allowed)
    if missing:
        raise ValueError(f"LingBot model inputs omit required fields: {missing}")
    if unexpected:
        raise ValueError(f"LingBot model inputs contain undeclared fields: {unexpected}")
    if keys & _RANDOM_FIELDS and keys & _RANDOM_FIELDS != _RANDOM_FIELDS:
        raise ValueError("flow noise and time must be supplied together")

    actions = model_inputs.get("actions")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise TypeError("official LingBot actions must have shape [batch,horizon,width]")
    batch_size = actions.shape[0]
    for name in keys:
        value = model_inputs[name]
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"official LingBot field {name} must be a batched tensor")
        if value.shape[0] != batch_size:
            raise ValueError(f"official LingBot field {name} has the wrong batch axis")


def select_native_calvin_physical_prompt_segment(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    sample_key: str,
    plan_sha256: str,
    episode_instance_id: str,
) -> tuple[int, str]:
    """Select one exact annotation without creating a canonical event label."""

    _require_sha256("physical prompt plan", plan_sha256)
    if not isinstance(episode_instance_id, str) or not episode_instance_id:
        raise ValueError("physical prompt selection requires an episode occurrence")
    candidates = dataset.candidate_segment_indices_by_key(sample_key)
    payload = {
        "candidate_segment_indices": candidates,
        "episode_instance_id": episode_instance_id,
        "plan_sha256": plan_sha256,
        "sample_key": sample_key,
        "schema": _PHYSICAL_PROMPT_SELECTION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    selected = candidates[int(digest[:16], 16) % len(candidates)]
    return selected, digest


def native_calvin_sample_plan_instance_id(
    *,
    optimizer_step: int,
    sample: PlannedSample,
) -> str:
    """Name one reset-only sample occurrence for deterministic prompt selection."""

    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("sample-plan optimizer step must be non-negative")
    if not isinstance(sample, PlannedSample):
        raise TypeError("sample-plan occurrence requires one planned sample")
    return (
        f"sample-plan/step-{optimizer_step:08d}/"
        f"sample-index-{sample.sample_index:08d}"
    )


def _physical_prompt_batch_sha256(receipts: Sequence[str]) -> str:
    if not receipts:
        raise ValueError("physical prompt receipt batch cannot be empty")
    for digest in receipts:
        _require_sha256("physical prompt receipt", digest)
    payload = {
        "receipts": tuple(receipts),
        "schema": _PHYSICAL_PROMPT_SELECTION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_native_calvin_physical_control_chunks(
    samples: Sequence[CalvinPhysicalSample],
    *,
    maximum_control_tokens: int,
    gradient_suffix_control_tokens: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ExecutedControlBatch, ...]:
    """Map one exact raw action receipt to bounded, ordered LingBot controls.

    Formal two/four-rank runs use one sample per rank. Rejecting larger local
    batches avoids silently advancing short receipts through padded prior passes.
    """

    if len(samples) != 1 or not isinstance(samples[0], CalvinPhysicalSample):
        raise ValueError("physical control chunking requires exactly one sample per rank")
    if (
        isinstance(maximum_control_tokens, bool)
        or not isinstance(maximum_control_tokens, int)
        or maximum_control_tokens <= 0
    ):
        raise ValueError("maximum_control_tokens must be positive")
    if gradient_suffix_control_tokens is not None and (
        isinstance(gradient_suffix_control_tokens, bool)
        or not isinstance(gradient_suffix_control_tokens, int)
        or gradient_suffix_control_tokens <= 0
        or gradient_suffix_control_tokens > maximum_control_tokens
    ):
        raise ValueError(
            "gradient_suffix_control_tokens must be positive and no larger than "
            "maximum_control_tokens"
        )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("physical CALVIN controls require a supported floating dtype")

    sample = samples[0]
    span = sample.incoming_control_span
    mapped_actions = map_calvin_action_to_lingbot(span.raw_actions)
    mapped_valid = map_calvin_action_to_lingbot(
        sample.record.action_valid.astype(np.float32, copy=False)
    ).astype(np.bool_, copy=False)
    reset_count = int(span.left_censored_start)
    token_count = reset_count + mapped_actions.shape[0]
    if token_count <= 0:
        raise RuntimeError("a physical event has neither a reset nor executed controls")

    values = np.zeros((token_count, mapped_actions.shape[-1]), dtype=np.float32)
    field_valid = np.zeros_like(values, dtype=np.bool_)
    delta_time = np.zeros(token_count, dtype=np.float32)
    reset = np.zeros(token_count, dtype=np.bool_)
    if reset_count:
        reset[0] = True
    if mapped_actions.shape[0]:
        values[reset_count:] = mapped_actions
        field_valid[reset_count:] = mapped_valid
        delta_time[reset_count:] = np.float32(sample.record.delta_t_s)

    target_device = torch.device(device)
    chunk_ranges: list[tuple[int, int]] = []
    if (
        gradient_suffix_control_tokens is not None
        and token_count > gradient_suffix_control_tokens
    ):
        suffix_start = token_count - gradient_suffix_control_tokens
        chunk_ranges.extend(
            (start, min(start + maximum_control_tokens, suffix_start))
            for start in range(0, suffix_start, maximum_control_tokens)
        )
        chunk_ranges.append((suffix_start, token_count))
    else:
        chunk_ranges.extend(
            (start, min(start + maximum_control_tokens, token_count))
            for start in range(0, token_count, maximum_control_tokens)
        )

    chunks: list[ExecutedControlBatch] = []
    for start, stop in chunk_ranges:
        chunk_values = torch.from_numpy(values[start:stop]).to(
            device=target_device,
            dtype=dtype,
        )
        chunk_valid = torch.from_numpy(field_valid[start:stop]).to(device=target_device)
        chunk_delta = torch.from_numpy(delta_time[start:stop]).to(
            device=target_device,
            dtype=dtype,
        )
        chunk_reset = torch.from_numpy(reset[start:stop]).to(device=target_device)
        token_valid = torch.ones(stop - start, dtype=torch.bool, device=target_device)
        chunks.append(
            ExecutedControlBatch(
                values=chunk_values.unsqueeze(0),
                field_valid=chunk_valid.unsqueeze(0),
                token_valid=token_valid.unsqueeze(0),
                delta_time=chunk_delta.unsqueeze(0),
                reset=chunk_reset.unsqueeze(0),
                acknowledged=token_valid.unsqueeze(0),
            )
        )
    return tuple(chunks)


def build_native_calvin_training_batch(
    samples: Sequence[CalvinStatefulTransitionSample | CalvinPhysicalSample],
    *,
    lane_ids: Sequence[int],
    optimizer_step: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    episode_keys: Sequence[str] | None = None,
    frame_indices: Sequence[int] | None = None,
    reset: Sequence[bool] | None = None,
    allow_forced_routing_reset: bool = False,
) -> NativeCALVINTrainingBatch:
    """Map CALVIN targets and previous executed controls through disjoint paths."""

    if not samples or len(samples) != len(lane_ids):
        raise ValueError("samples and lane IDs must be equal non-empty sequences")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native CALVIN controls require a supported floating dtype")
    if not isinstance(allow_forced_routing_reset, bool):
        raise TypeError("forced CALVIN routing-reset control must be boolean")
    mapped = tuple(map_calvin_transition_to_lingbot(sample) for sample in samples)
    batch_size = len(mapped)
    resolved_episode_keys = (
        tuple(item.episode_key for item in mapped) if episode_keys is None else tuple(episode_keys)
    )
    resolved_frame_indices = (
        tuple(item.transition_index for item in mapped)
        if frame_indices is None
        else tuple(frame_indices)
    )
    resolved_reset = (
        tuple(item.transition_index == 0 for item in mapped) if reset is None else tuple(reset)
    )
    if not (
        len(resolved_episode_keys)
        == len(resolved_frame_indices)
        == len(resolved_reset)
        == batch_size
    ):
        raise ValueError("explicit CALVIN routing metadata must match the batch")
    natural_reset = tuple(item.transition_index == 0 for item in mapped)
    if allow_forced_routing_reset:
        if reset is None or episode_keys is None or frame_indices is None:
            raise ValueError("forced CALVIN routing resets require explicit occurrence metadata")
        if any(resolved_frame_indices) or not all(resolved_reset):
            raise ValueError("forced CALVIN routing occurrences must be one-frame episodes")
    else:
        if any(
            frame != item.transition_index
            for frame, item in zip(resolved_frame_indices, mapped, strict=True)
        ):
            raise ValueError("explicit frame indices differ from CALVIN transitions")
        if resolved_reset != natural_reset:
            raise ValueError("explicit reset flags differ from CALVIN segment boundaries")

    target_device = torch.device(device)
    previous_valid = torch.tensor(
        [item.previous_action_valid for item in mapped],
        dtype=torch.bool,
        device=target_device,
    )
    values = torch.from_numpy(np.stack([item.previous_executed_action for item in mapped])).to(
        device=target_device,
        dtype=dtype,
    )
    values = values * previous_valid.unsqueeze(-1).to(dtype)
    schema_valid = torch.from_numpy(np.stack([item.action_valid for item in mapped])).to(
        device=target_device,
        dtype=torch.bool,
    )
    field_valid = schema_valid & previous_valid.unsqueeze(-1)
    token_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=target_device)
    reset_tensor = (~previous_valid).unsqueeze(1)
    elapsed = torch.tensor(
        [item.elapsed_time_s for item in mapped],
        dtype=dtype,
        device=target_device,
    )
    elapsed = elapsed * previous_valid.to(dtype)
    controls = ExecutedControlBatch(
        values=values.unsqueeze(1),
        field_valid=field_valid.unsqueeze(1),
        token_valid=token_valid,
        delta_time=elapsed.unsqueeze(1),
        reset=reset_tensor,
        acknowledged=token_valid.clone(),
    )
    routing = NativeCALVINRouting(
        lane_ids=tuple(lane_ids),
        episode_keys=resolved_episode_keys,
        frame_indices=resolved_frame_indices,
        reset=resolved_reset,
        sample_keys=tuple(item.sample_key for item in mapped),
        optimizer_step=optimizer_step,
    )
    return NativeCALVINTrainingBatch(
        host_items=tuple(item.feature_transform_item() for item in mapped),
        controls=controls,
        routing=routing,
        structural_target_requests=tuple(
            native_calvin_structural_target_request(sample) for sample in samples
        ),
    )


def build_native_calvin_physical_training_batch(
    samples: Sequence[CalvinPhysicalSample],
    *,
    maximum_control_tokens: int,
    gradient_suffix_control_tokens: int | None = None,
    lane_ids: Sequence[int],
    optimizer_step: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    episode_keys: Sequence[str] | None = None,
    frame_indices: Sequence[int] | None = None,
    reset: Sequence[bool] | None = None,
    allow_forced_routing_reset: bool = False,
) -> NativeCALVINTrainingBatch:
    """Build host targets plus the exact chunked physical control receipt."""

    if len(samples) != 1:
        raise ValueError("physical CALVIN training requires one local sample per rank")
    base = build_native_calvin_training_batch(
        samples,
        lane_ids=lane_ids,
        optimizer_step=optimizer_step,
        device=device,
        dtype=dtype,
        episode_keys=episode_keys,
        frame_indices=frame_indices,
        reset=reset,
        allow_forced_routing_reset=allow_forced_routing_reset,
    )
    chunks = build_native_calvin_physical_control_chunks(
        samples,
        maximum_control_tokens=maximum_control_tokens,
        gradient_suffix_control_tokens=gradient_suffix_control_tokens,
        device=device,
        dtype=dtype,
    )
    return replace(
        base,
        controls=chunks[-1],
        prior_control_chunks=chunks,
        physical_control_span_sha256=tuple(
            sample.incoming_control_span.sha256 for sample in samples
        ),
        selected_segment_indices=tuple(sample.selected_segment.index for sample in samples),
    )


def collate_native_calvin_training_batch(
    batch: NativeCALVINTrainingBatch,
    *,
    feature_transform: Any,
    collator: Callable[[Sequence[dict[str, Any]]], Mapping[str, Any]],
    augmentation_seeds: Sequence[int],
    source_digest: str,
) -> CollatedNativeCALVINBatch:
    """Apply the released transform transactionally, then strip transform metadata."""

    _require_sha256("source_digest", source_digest)
    apply = getattr(feature_transform, "apply", None)
    if not callable(apply) or not callable(collator):
        raise TypeError("feature_transform.apply and collator must be callable")
    if len(augmentation_seeds) != len(batch.host_items):
        raise ValueError("augmentation seeds must provide one value per host item")
    transformed: list[dict[str, Any]] = []
    for item, seed in zip(batch.host_items, augmentation_seeds, strict=True):
        with _isolated_augmentation_rng(seed):
            result = apply(copy.deepcopy(item), policy_eval=False)
        if not isinstance(result, dict):
            raise TypeError("official LingBot feature transform must return a dictionary")
        transformed.append(result)
    collated = collator(tuple(transformed))
    if not isinstance(collated, Mapping):
        raise TypeError("official LingBot collator must return a mapping")
    _audit_names(collated, path="transformed_batch")
    missing = sorted(_TRANSFORM_FIELDS - set(collated))
    unexpected = sorted(set(collated) - _TRANSFORM_FIELDS)
    if missing:
        raise ValueError(f"official LingBot transform omits required fields: {missing}")
    if unexpected:
        raise ValueError(f"official LingBot transform emitted undeclared fields: {unexpected}")
    batch_size = batch.routing.batch_size
    for name in _TRANSFORM_FIELDS:
        value = collated[name]
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"official LingBot field {name} must be a batched tensor")
        if value.shape[0] != batch_size:
            raise ValueError(f"official LingBot field {name} has the wrong batch axis")
    model_inputs = {name: collated[name] for name in _MODEL_FIELDS}
    return CollatedNativeCALVINBatch(
        model_inputs=model_inputs,
        controls=batch.controls,
        routing=batch.routing,
        source_digest=source_digest,
        structural_target_requests=batch.structural_target_requests,
        prior_control_chunks=batch.prior_control_chunks,
    )


def materialize_native_flow_randomness(
    batch: CollatedNativeCALVINBatch,
    planned: (
        PlannedNativeCALVINBatch
        | PlannedNativeCALVINContinuationBatch
        | PlannedNativeCALVINReplayBatch
    ),
) -> CollatedNativeCALVINBatch:
    """Reproduce LingBot's released noise and ``sample_beta(1.5,1)`` per sample."""

    if batch.routing != planned.training.routing:
        raise ValueError("collated and planned CALVIN routing differs")
    if batch.source_digest != planned.source_digest:
        raise ValueError("collated CALVIN source digest differs from the frozen plan")
    actions = batch.model_inputs["actions"]
    batch_size = actions.shape[0]
    if not (len(planned.flow_noise_seeds) == len(planned.flow_timestep_seeds) == batch_size):
        raise ValueError("frozen flow seeds must provide one pair per batch item")
    noise_items: list[torch.Tensor] = []
    time_items: list[torch.Tensor] = []
    for noise_seed, timestep_seed in zip(
        planned.flow_noise_seeds,
        planned.flow_timestep_seeds,
        strict=True,
    ):
        noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        noise_items.append(
            torch.randn(actions.shape[1:], generator=noise_generator, dtype=torch.float32)
        )
        time_generator = torch.Generator(device="cpu").manual_seed(timestep_seed)
        uniforms = torch.rand(2, generator=time_generator, dtype=torch.float32)
        gamma_one = uniforms[0].pow(1.0 / 1.5)
        gamma_two = uniforms[1]
        time_items.append((gamma_one / (gamma_one + gamma_two)) * 0.999 + 0.001)
    model_inputs = dict(batch.model_inputs)
    model_inputs["noise"] = torch.stack(noise_items).to(actions)
    model_inputs["time"] = torch.stack(time_items).to(actions)
    audit_native_calvin_model_inputs(model_inputs, require_randomness=True)
    return CollatedNativeCALVINBatch(
        model_inputs=model_inputs,
        controls=batch.controls,
        routing=batch.routing,
        source_digest=batch.source_digest,
        structural_target_requests=batch.structural_target_requests,
        modalities=batch.modalities,
        prior_control_chunks=batch.prior_control_chunks,
    )


def build_native_calvin_context(
    batch: CollatedNativeCALVINBatch,
    *,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None = None,
    prediction_request: NativePredictionRequest | None = None,
) -> LingBotNativeContext:
    """Create an unbound context; the patched official host binds prefix roles."""

    reset = torch.tensor(batch.routing.reset, dtype=torch.bool, device=batch.controls.values.device)
    if previous_state_valid is None:
        previous_state_valid = ~reset if previous_state is not None else torch.zeros_like(reset)
    if previous_state_valid.shape != reset.shape or previous_state_valid.dtype != torch.bool:
        raise ValueError("previous state validity must be boolean [batch]")
    if previous_state_valid.device != reset.device:
        raise ValueError("previous state validity and controls must share one device")
    if (previous_state_valid & reset).any():
        raise ValueError("a reset sample cannot read previous posterior rows")
    if ((~reset) & ~previous_state_valid).any():
        raise ValueError("a non-reset CALVIN transition requires a previous posterior")
    if previous_state is not None and previous_state.batch_size != batch.routing.batch_size:
        raise ValueError("previous posterior and CALVIN batch sizes differ")
    return native_context_from_persistent_state(
        controls=batch.controls,
        persistent_state=previous_state,
        persistent_state_valid=previous_state_valid,
        prediction_request=prediction_request,
        modalities=batch.modalities,
    )


def build_native_calvin_episode_domain(
    dataset: CalvinStatefulTransitionDataset,
    *,
    excluded_source_episode_indices: Sequence[int] = (),
) -> tuple[EpisodeSampleSequence, ...]:
    """Reconstruct the exact source-disjoint episode domain used by a frozen plan."""

    excluded = tuple(excluded_source_episode_indices)
    if excluded != tuple(sorted(set(excluded))):
        raise ValueError("excluded CALVIN source episode indices must be unique and sorted")
    for value in excluded:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(dataset.index.episodes)
        ):
            raise ValueError("excluded CALVIN source episode index is outside the dataset")
    excluded_set = frozenset(excluded)
    episodes = tuple(
        EpisodeSampleSequence(
            episode_key=episode.episode_key,
            sample_keys=episode.sample_keys,
        )
        for episode, segment in zip(
            dataset.episode_manifest,
            dataset.index.segments,
            strict=True,
        )
        if int(segment.episode_index) not in excluded_set
    )
    if not episodes:
        raise ValueError("CALVIN stream planning excluded every language segment")
    return episodes


def build_native_calvin_physical_episode_domain(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    excluded_source_episode_indices: Sequence[int] = (),
    minimum_future_source_frames: int = 0,
) -> tuple[EpisodeSampleSequence, ...]:
    """Build unique events with an explicit same-episode source-future budget."""

    if not isinstance(dataset, CalvinPhysicalTransitionDataset):
        raise TypeError("physical episode planning requires a physical CALVIN dataset")
    if (
        isinstance(minimum_future_source_frames, bool)
        or not isinstance(minimum_future_source_frames, int)
        or minimum_future_source_frames < 0
    ):
        raise ValueError("minimum future source frames must be a non-negative integer")
    excluded = tuple(excluded_source_episode_indices)
    if excluded != tuple(sorted(set(excluded))):
        raise ValueError("excluded CALVIN source episode indices must be unique and sorted")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < len(dataset.index.episodes)
        for value in excluded
    ):
        raise ValueError("excluded CALVIN source episode index is outside the dataset")
    excluded_set = frozenset(excluded)
    episodes: list[EpisodeSampleSequence] = []
    for manifest in dataset.episode_manifest:
        if manifest.source_episode_index in excluded_set:
            continue
        eligible: list[str] = []
        for sample_key in manifest.sample_keys:
            try:
                dataset.future_source_global_indices_by_key(
                    sample_key,
                    count=minimum_future_source_frames,
                )
            except ContractError:
                continue
            eligible.append(sample_key)
        if eligible:
            episodes.append(
                EpisodeSampleSequence(
                    episode_key=manifest.episode_key,
                    sample_keys=tuple(eligible),
                )
            )
    if not episodes:
        raise ValueError(
            "CALVIN physical planning excluded every eligible labelled raw episode"
        )
    return tuple(episodes)


def build_native_calvin_physical_sample_domain(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    excluded_source_episode_indices: Sequence[int] = (),
    minimum_future_source_frames: int = 0,
) -> tuple[str, ...]:
    """Build the exact source-disjoint physical-event sample domain."""

    episodes = build_native_calvin_physical_episode_domain(
        dataset,
        excluded_source_episode_indices=excluded_source_episode_indices,
        minimum_future_source_frames=minimum_future_source_frames,
    )
    sample_keys = tuple(
        sample_key
        for episode in episodes
        for sample_key in episode.sample_keys
    )
    if len(sample_keys) != len(set(sample_keys)):
        raise RuntimeError("CALVIN physical sample domain contains duplicate event keys")
    return sample_keys


def build_native_calvin_physical_sample_plan(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
    excluded_source_episode_indices: Sequence[int] = (),
    minimum_future_source_frames: int = 0,
) -> FrozenSamplePlan:
    """Freeze reset-only physical events without inventing temporal continuity."""

    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("CALVIN physical sample planning requires a content-addressed manifest")
    return FrozenSamplePlan(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        sample_keys=build_native_calvin_physical_sample_domain(
            dataset,
            excluded_source_episode_indices=excluded_source_episode_indices,
            minimum_future_source_frames=minimum_future_source_frames,
        ),
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
    )


def build_native_calvin_physical_stream_plan(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
    lane_interleave_factor: int = 1,
    excluded_source_episode_indices: Sequence[int] = (),
    minimum_future_source_frames: int = 0,
) -> FrozenEpisodeStreamPlan:
    """Freeze one topology-neutral stream over unique physical events."""

    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("CALVIN physical stream planning requires a content-addressed manifest")
    return FrozenEpisodeStreamPlan(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        episodes=build_native_calvin_physical_episode_domain(
            dataset,
            excluded_source_episode_indices=excluded_source_episode_indices,
            minimum_future_source_frames=minimum_future_source_frames,
        ),
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
        lane_interleave_factor=lane_interleave_factor,
    )


def build_native_calvin_stream_plan(
    dataset: CalvinStatefulTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
    lane_interleave_factor: int = 1,
    excluded_source_episode_indices: Sequence[int] = (),
) -> FrozenEpisodeStreamPlan:
    """Build a frozen stream over a source-disjoint episode domain."""

    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("CALVIN stream planning requires a content-addressed manifest")
    episodes = build_native_calvin_episode_domain(
        dataset,
        excluded_source_episode_indices=excluded_source_episode_indices,
    )
    return FrozenEpisodeStreamPlan(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        episodes=episodes,
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
        lane_interleave_factor=lane_interleave_factor,
    )


def build_native_calvin_reset_mixture_stream_plan(
    dataset: CalvinStatefulTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
    lane_interleave_factor: int,
    reset_numerator: int,
    reset_denominator: int,
    excluded_source_episode_indices: Sequence[int] = (),
) -> FrozenResetMixtureStreamPlan:
    """Build a source-only real-reset/causal plan without repeated source frames."""

    if (
        isinstance(reset_numerator, bool)
        or not isinstance(reset_numerator, int)
        or reset_numerator <= 0
        or isinstance(reset_denominator, bool)
        or not isinstance(reset_denominator, int)
        or reset_denominator <= reset_numerator
    ):
        raise ValueError("CALVIN reset mixture weight must lie strictly between zero and one")
    if (total_steps * reset_numerator) % reset_denominator:
        raise ValueError("CALVIN reset mixture budget must realize its weight exactly")
    reset_steps = total_steps * reset_numerator // reset_denominator
    causal_steps = total_steps - reset_steps
    causal = build_native_calvin_stream_plan(
        dataset,
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=causal_steps,
        lane_interleave_factor=lane_interleave_factor,
        excluded_source_episode_indices=excluded_source_episode_indices,
    )
    causal_source_indices = {
        dataset.source_global_index_by_key(transition.sample.sample_key)
        for optimizer_step in range(causal.total_steps)
        for transition in causal.global_batch(optimizer_step).transitions
    }
    excluded_sources = set(excluded_source_episode_indices)
    candidates: list[tuple[bytes, str, int]] = []
    prefix = (f"picf-next.calvin-real-reset-selection.v1\0{comparison_id}\0{seed}\0").encode()
    for episode in causal.episodes:
        sample_key = episode.sample_keys[0]
        locator = dataset.locator_by_key(sample_key)
        segment = dataset.index.segments[locator.segment_index]
        source_global_index = int(locator.global_index)
        if (
            int(segment.episode_index) in excluded_sources
            or source_global_index in causal_source_indices
        ):
            continue
        digest = hashlib.sha256(
            prefix + episode.episode_key.encode("utf-8") + b"\0" + sample_key.encode("utf-8")
        ).digest()
        candidates.append((digest, sample_key, source_global_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    required = reset_steps * global_batch_size
    selected: list[tuple[bytes, str, int]] = []
    selected_sources: set[int] = set()
    for candidate in candidates:
        if candidate[2] in selected_sources:
            continue
        selected.append(candidate)
        selected_sources.add(candidate[2])
        if len(selected) == required:
            break
    if len(selected) < required:
        raise ValueError(
            "CALVIN reset mixture has insufficient unique source-disjoint reset frames"
        )
    return FrozenResetMixtureStreamPlan(
        causal_plan=causal,
        reset_sample_keys=tuple(item[1] for item in selected),
        reset_source_global_indices=tuple(item[2] for item in selected),
        total_steps=total_steps,
        reset_numerator=reset_numerator,
        reset_denominator=reset_denominator,
    )


def build_native_calvin_training_stream_plan(
    dataset: CalvinStatefulTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
    lane_interleave_factor: int = 1,
    excluded_source_episode_indices: Sequence[int] = (),
    reset_numerator: int | None = None,
    reset_denominator: int | None = None,
) -> EpisodeStreamPlan:
    """Build the sole typed training stream selected by explicit mixture arguments."""

    if (reset_numerator is None) != (reset_denominator is None):
        raise ValueError("CALVIN reset mixture numerator and denominator must be provided together")
    if reset_numerator is None or reset_denominator is None:
        return build_native_calvin_stream_plan(
            dataset,
            comparison_id=comparison_id,
            seed=seed,
            global_batch_size=global_batch_size,
            total_steps=total_steps,
            lane_interleave_factor=lane_interleave_factor,
            excluded_source_episode_indices=excluded_source_episode_indices,
        )
    return build_native_calvin_reset_mixture_stream_plan(
        dataset,
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
        lane_interleave_factor=lane_interleave_factor,
        reset_numerator=reset_numerator,
        reset_denominator=reset_denominator,
        excluded_source_episode_indices=excluded_source_episode_indices,
    )


def build_planned_native_calvin_batch(
    plan: TrainingPlan,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
    *,
    optimizer_step: int,
    rank: int,
    world_size: int,
    gradient_accumulation_steps: int,
    accumulation_index: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    maximum_control_tokens: int | None = None,
    gradient_suffix_control_tokens: int | None = None,
) -> PlannedNativeCALVINBatch:
    microbatch = plan.microbatch_for_rank(
        optimizer_step,
        rank=rank,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        accumulation_index=accumulation_index,
    )
    sample_plan = isinstance(plan, FrozenSamplePlan)
    if sample_plan != isinstance(microbatch, PlannedMicrobatch):
        raise RuntimeError("CALVIN training plan returned the wrong microbatch type")

    if isinstance(microbatch, PlannedMicrobatch):
        planned_samples = microbatch.samples
        episode_instance_ids = tuple(
            native_calvin_sample_plan_instance_id(
                optimizer_step=optimizer_step,
                sample=item,
            )
            for item in planned_samples
        )
    else:
        planned_samples = tuple(item.sample for item in microbatch.transitions)
        episode_instance_ids = tuple(
            item.episode_instance_id for item in microbatch.transitions
        )

    prompt_receipts: tuple[str, ...] = ()
    if isinstance(dataset, CalvinPhysicalTransitionDataset):
        if maximum_control_tokens is None:
            raise ValueError("physical CALVIN planning requires maximum_control_tokens")
        selected_and_receipts = tuple(
            select_native_calvin_physical_prompt_segment(
                dataset,
                sample_key=item.sample_key,
                plan_sha256=plan.plan_sha256,
                episode_instance_id=episode_instance_id,
            )
            for item, episode_instance_id in zip(
                planned_samples,
                episode_instance_ids,
                strict=True,
            )
        )
        samples = tuple(
            dataset.by_key(
                item.sample_key,
                selected_segment_index=selected,
            )
            for item, (selected, _receipt) in zip(
                planned_samples,
                selected_and_receipts,
                strict=True,
            )
        )
        prompt_receipts = tuple(receipt for _selected, receipt in selected_and_receipts)
    else:
        if maximum_control_tokens is not None or gradient_suffix_control_tokens is not None:
            raise ValueError("legacy CALVIN planning cannot receive physical control bounds")
        samples = tuple(dataset.by_key(item.sample_key) for item in planned_samples)

    if isinstance(microbatch, PlannedMicrobatch):
        partitions = world_size * gradient_accumulation_steps
        local_size = plan.global_batch_size // partitions
        partition = accumulation_index * world_size + rank
        lane_ids = tuple(partition * local_size + index for index in range(local_size))
        reset = tuple(True for _ in samples)
        frame_indices = tuple(0 for _ in samples)
    else:
        for planned, sample in zip(microbatch.transitions, samples, strict=True):
            if (
                sample.episode_key != planned.episode_key
                or sample.transition_index != planned.transition_index
            ):
                raise RuntimeError("frozen episode plan and CALVIN transition metadata disagree")
        lane_lookup = {lane_id: index for index, lane_id in enumerate(plan.lane_ids)}
        lane_ids = tuple(lane_lookup[item.lane_id] for item in microbatch.transitions)
        reset = tuple(item.transition_index == 0 for item in microbatch.transitions)
        frame_indices = tuple(item.transition_index for item in microbatch.transitions)

    batch_kwargs = {
        "lane_ids": lane_ids,
        "optimizer_step": optimizer_step,
        "device": device,
        "dtype": dtype,
        "episode_keys": episode_instance_ids,
        "frame_indices": frame_indices,
        "reset": reset,
    }
    if isinstance(dataset, CalvinPhysicalTransitionDataset):
        if maximum_control_tokens is None:
            raise RuntimeError("physical CALVIN planning lost its control-token bound")
        training = build_native_calvin_physical_training_batch(
            samples,
            maximum_control_tokens=maximum_control_tokens,
            gradient_suffix_control_tokens=gradient_suffix_control_tokens,
            allow_forced_routing_reset=sample_plan,
            **batch_kwargs,
        )
    else:
        training = build_native_calvin_training_batch(
            samples,
            allow_forced_routing_reset=sample_plan,
            **batch_kwargs,
        )
    return PlannedNativeCALVINBatch(
        training=training,
        plan_microbatch=microbatch,
        plan_sha256=plan.plan_sha256,
        augmentation_seeds=tuple(item.augmentation_seed for item in planned_samples),
        flow_noise_seeds=tuple(item.flow_noise_seed for item in planned_samples),
        flow_timestep_seeds=tuple(item.flow_timestep_seed for item in planned_samples),
        physical_prompt_selection_sha256=(
            None if not prompt_receipts else _physical_prompt_batch_sha256(prompt_receipts)
        ),
        physical_prompt_selection_receipts=tuple(prompt_receipts),
    )


def build_native_calvin_continuation_batch(
    primary: PlannedNativeCALVINBatch,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
    *,
    offset: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    maximum_control_tokens: int | None = None,
    gradient_suffix_control_tokens: int | None = None,
) -> PlannedNativeCALVINContinuationBatch:
    """Build one exact later full-graph step from source-known stream metadata."""

    if not isinstance(primary, PlannedNativeCALVINBatch):
        raise TypeError("native continuation requires a planned primary batch")
    if not isinstance(dataset, CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset):
        raise TypeError("native continuation requires a typed CALVIN dataset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0:
        raise ValueError("native continuation offset must be a positive integer")
    if primary.fixed_observation_pair_sha256 is not None:
        raise ValueError(
            "fixed-X source truth is audited only at the primary reset frame; "
            "continuation is forbidden"
        )

    future_keys = tuple(
        dataset.future_sample_keys(transition.sample.sample_key, count=offset)[-1]
        for transition in primary.plan_microbatch.transitions
    )
    prompt_receipts: tuple[str, ...] = ()
    if isinstance(dataset, CalvinPhysicalTransitionDataset):
        if maximum_control_tokens is None:
            raise ValueError("physical continuation requires maximum_control_tokens")
        selected_and_receipts = tuple(
            select_native_calvin_physical_prompt_segment(
                dataset,
                sample_key=sample_key,
                plan_sha256=primary.plan_sha256,
                episode_instance_id=transition.episode_instance_id,
            )
            for sample_key, transition in zip(
                future_keys,
                primary.plan_microbatch.transitions,
                strict=True,
            )
        )
        samples = tuple(
            dataset.by_key(sample_key, selected_segment_index=selected)
            for sample_key, (selected, _receipt) in zip(
                future_keys,
                selected_and_receipts,
                strict=True,
            )
        )
        prompt_receipts = tuple(receipt for _selected, receipt in selected_and_receipts)
    else:
        if maximum_control_tokens is not None or gradient_suffix_control_tokens is not None:
            raise ValueError("legacy continuation cannot receive physical control bounds")
        samples = tuple(dataset.by_key(sample_key) for sample_key in future_keys)
    routing = primary.training.routing
    batch_kwargs = {
        "lane_ids": routing.lane_ids,
        "optimizer_step": routing.optimizer_step,
        "device": device,
        "dtype": dtype,
        "episode_keys": routing.episode_keys,
        "frame_indices": tuple(value + offset for value in routing.frame_indices),
        "reset": (False,) * routing.batch_size,
    }
    if isinstance(dataset, CalvinPhysicalTransitionDataset):
        if maximum_control_tokens is None:
            raise RuntimeError("physical continuation lost its control-token bound")
        training = build_native_calvin_physical_training_batch(
            samples,
            maximum_control_tokens=maximum_control_tokens,
            gradient_suffix_control_tokens=gradient_suffix_control_tokens,
            **batch_kwargs,
        )
    else:
        training = build_native_calvin_training_batch(samples, **batch_kwargs)
    if primary.task_intervention_sha256 is not None:
        host_items = []
        target_requests = []
        for future_item, primary_item, future_request, primary_request in zip(
            training.host_items,
            primary.training.host_items,
            training.structural_target_requests,
            primary.training.structural_target_requests,
            strict=True,
        ):
            donor_instruction = primary_item["task"]
            if not isinstance(donor_instruction, str) or not donor_instruction:
                raise RuntimeError("intervened primary batch lost its donor instruction")
            replaced_item = dict(future_item)
            replaced_item["task"] = donor_instruction
            host_items.append(replaced_item)
            target_requests.append(
                replace(
                    future_request,
                    task_key=primary_request.task_key,
                )
            )
        training = replace(
            training,
            host_items=tuple(host_items),
            structural_target_requests=tuple(target_requests),
        )

    def seeds(parent: Sequence[int], kind: str) -> tuple[int, ...]:
        return tuple(
            derive_subseed(
                parent_seed,
                _CONTINUATION_SEED_STREAM,
                kind,
                str(offset),
                sample_key,
            )
            for parent_seed, sample_key in zip(parent, future_keys, strict=True)
        )

    return PlannedNativeCALVINContinuationBatch(
        training=training,
        parent_source_digest=primary.source_digest,
        offset=offset,
        augmentation_seeds=seeds(primary.augmentation_seeds, "augmentation"),
        flow_noise_seeds=seeds(primary.flow_noise_seeds, "flow-noise"),
        flow_timestep_seeds=seeds(primary.flow_timestep_seeds, "flow-timestep"),
        task_intervention_sha256=primary.task_intervention_sha256,
        physical_prompt_selection_sha256=(
            None if not prompt_receipts else _physical_prompt_batch_sha256(prompt_receipts)
        ),
        physical_prompt_selection_receipts=tuple(prompt_receipts),
    )


def build_native_calvin_replay_batch(
    dataset: CalvinStatefulTransitionDataset,
    *,
    sample_key: str,
    lane_id: int,
    episode_instance_id: str,
    optimizer_step: int,
    replay_seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PlannedNativeCALVINReplayBatch:
    """Build one label-independent fixed-weight audit frame."""

    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("native replay requires a stateful CALVIN dataset")
    values = (lane_id, optimizer_step, replay_seed)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("native replay counters must be non-negative integers")
    if not isinstance(episode_instance_id, str) or not episode_instance_id:
        raise ValueError("native replay episode instance must be non-empty")
    sample = dataset.by_key(sample_key)
    training = build_native_calvin_training_batch(
        (sample,),
        lane_ids=(lane_id,),
        optimizer_step=optimizer_step,
        device=device,
        dtype=dtype,
        episode_keys=(episode_instance_id,),
        frame_indices=(sample.transition_index,),
        reset=(sample.transition_index == 0,),
    )

    def seed(kind: str) -> int:
        return derive_subseed(replay_seed, _REPLAY_SEED_STREAM, kind, sample_key)

    return PlannedNativeCALVINReplayBatch(
        training=training,
        replay_seed=replay_seed,
        augmentation_seeds=(seed("augmentation"),),
        flow_noise_seeds=(seed("flow-noise"),),
        flow_timestep_seeds=(seed("flow-timestep"),),
    )

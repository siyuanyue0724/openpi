"""Ordered CALVIN batch bridge for LingBot's unified PICF training boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from picf_next.data.calvin import (
    CalvinStatefulTransitionDataset,
    CalvinStatefulTransitionSample,
)
from picf_next.data.lingbot_calvin import map_calvin_transition_to_lingbot
from picf_next.hosts.lingbot_unified import LingBotUnifiedBeliefGraph
from picf_next.hosts.lingbot_unified_training import LingBotUnifiedStepBatch
from picf_next.training.control import (
    EpisodeSampleSequence,
    FrozenEpisodeStreamPlan,
    PlannedStreamMicrobatch,
)


@dataclass(frozen=True, slots=True)
class LingBotCALVINTrainingBatch:
    """Target-bearing host items plus a separately causal posterior transaction."""

    host_items: tuple[dict[str, Any], ...]
    temporal: LingBotUnifiedStepBatch
    sample_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannedLingBotCALVINBatch:
    """One topology-invariant stream shard plus all replayable random seeds."""

    training: LingBotCALVINTrainingBatch
    plan_microbatch: PlannedStreamMicrobatch
    plan_sha256: str
    augmentation_seeds: tuple[int, ...]
    flow_noise_seeds: tuple[int, ...]
    flow_timestep_seeds: tuple[int, ...]

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
            "sample_keys": self.training.sample_keys,
            "world_size": self.plan_microbatch.world_size,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CollatedLingBotCALVINBatch:
    """Official host tensors paired with the untouched causal transaction."""

    model_inputs: Mapping[str, Any]
    temporal: LingBotUnifiedStepBatch
    sample_keys: tuple[str, ...]
    source_digest: str


_REQUIRED_LINGBOT_TRAINING_INPUTS = frozenset(
    {
        "images",
        "image_grid_thw",
        "img_masks",
        "state",
        "lang_tokens",
        "lang_masks",
        "actions",
        "action_is_pad",
        "joint_mask",
        "state_joint_mask",
    }
)


@contextmanager
def _isolated_augmentation_rng(seed: int):
    """Drive released transforms from one sample seed without consuming model RNG."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("augmentation seeds must be uint64 integers")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=()):
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(seed)
            torch.set_rng_state(cpu_generator.get_state())
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def collate_lingbot_calvin_training_batch(
    batch: LingBotCALVINTrainingBatch,
    *,
    feature_transform: Any,
    collator: Callable[[Sequence[dict[str, Any]]], Mapping[str, Any]],
    augmentation_seeds: Sequence[int],
    source_digest: str,
) -> CollatedLingBotCALVINBatch:
    """Run the released LingBot transform/collator without mutating replay input.

    ``FeatureTransform.apply`` mutates its argument while constructing relative
    actions and normalization. A deep copy is therefore part of the optimizer-
    retry contract, not defensive convenience.
    """

    apply = getattr(feature_transform, "apply", None)
    if not callable(apply) or not callable(collator):
        raise TypeError("feature_transform.apply and collator must be callable")
    if len(augmentation_seeds) != len(batch.host_items):
        raise ValueError("augmentation seeds must provide one value per host item")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise ValueError("source_digest must be a lowercase SHA-256 digest")
    transformed: list[dict[str, Any]] = []
    for host_item, augmentation_seed in zip(
        batch.host_items,
        augmentation_seeds,
        strict=True,
    ):
        with _isolated_augmentation_rng(augmentation_seed):
            result = apply(copy.deepcopy(host_item), policy_eval=False)
        if not isinstance(result, dict):
            raise TypeError("official LingBot feature transform must return a dictionary")
        transformed.append(result)
    collated = collator(tuple(transformed))
    if not isinstance(collated, Mapping):
        raise TypeError("official LingBot collator must return a mapping")
    missing = sorted(_REQUIRED_LINGBOT_TRAINING_INPUTS - set(collated))
    if missing:
        raise ValueError(f"official LingBot training batch omits required fields: {missing}")
    batch_size = len(batch.host_items)
    for name in _REQUIRED_LINGBOT_TRAINING_INPUTS:
        value = collated[name]
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"official LingBot field {name} must be a batched tensor")
        if value.shape[0] != batch_size:
            raise ValueError(f"official LingBot field {name} has the wrong batch axis")
    return CollatedLingBotCALVINBatch(
        model_inputs=dict(collated),
        temporal=batch.temporal,
        sample_keys=batch.sample_keys,
        source_digest=source_digest,
    )


def materialize_lingbot_flow_randomness(
    batch: CollatedLingBotCALVINBatch,
    planned: PlannedLingBotCALVINBatch,
) -> CollatedLingBotCALVINBatch:
    """Attach topology-invariant flow noise and timesteps to a host batch.

    LingBot's released flow objective samples both values from process-global
    device RNG state.  That is unsuitable for matched retries and distributed
    comparisons: data-loader timing or world-size changes can then alter the
    objective while preserving the sample IDs.  Sampling each item on CPU from
    its frozen-plan seeds preserves the released distributions while making the
    exact objective source-addressed.
    """

    if batch.sample_keys != planned.training.sample_keys:
        raise ValueError("collated and planned CALVIN sample identities differ")
    if batch.source_digest != planned.source_digest:
        raise ValueError("collated CALVIN source digest differs from the frozen plan")
    actions = batch.model_inputs.get("actions")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise TypeError("official LingBot actions must have shape [batch, horizon, width]")
    if not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("official LingBot actions must be finite floating point")
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
            torch.randn(
                actions.shape[1:],
                generator=noise_generator,
                dtype=torch.float32,
                device="cpu",
            )
        )
        time_generator = torch.Generator(device="cpu").manual_seed(timestep_seed)
        uniforms = torch.rand(
            2,
            generator=time_generator,
            dtype=torch.float32,
            device="cpu",
        )
        gamma_one = uniforms[0].pow(1.0 / 1.5)
        gamma_two = uniforms[1]
        time_items.append((gamma_one / (gamma_one + gamma_two)) * 0.999 + 0.001)

    model_inputs = dict(batch.model_inputs)
    model_inputs["noise"] = torch.stack(noise_items).to(actions)
    model_inputs["time"] = torch.stack(time_items).to(actions)
    return CollatedLingBotCALVINBatch(
        model_inputs=model_inputs,
        temporal=batch.temporal,
        sample_keys=batch.sample_keys,
        source_digest=batch.source_digest,
    )


def build_lingbot_calvin_training_batch(
    samples: Sequence[CalvinStatefulTransitionSample],
    *,
    lane_ids: Sequence[int],
    optimizer_step: int,
    graph: LingBotUnifiedBeliefGraph,
    capacity: int,
    device: torch.device | str = "cpu",
    episode_keys: Sequence[str] | None = None,
    frame_indices: Sequence[int] | None = None,
    reset: Sequence[bool] | None = None,
) -> LingBotCALVINTrainingBatch:
    """Build one ordered batch without exposing current targets to the state update."""

    if not samples or len(samples) != len(lane_ids):
        raise ValueError("samples and lane IDs must be equal non-empty sequences")
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
        raise ValueError("explicit stream metadata must match the CALVIN batch")
    if any(
        frame != item.transition_index
        for frame, item in zip(resolved_frame_indices, mapped, strict=True)
    ):
        raise ValueError("explicit frame indices differ from CALVIN transitions")
    if any(
        is_reset != (item.transition_index == 0)
        for is_reset, item in zip(resolved_reset, mapped, strict=True)
    ):
        raise ValueError("explicit reset flags differ from CALVIN segment boundaries")
    target_device = torch.device(device)
    graph_config = graph.config
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("belief capacity must be an integer")
    if capacity <= 0:
        raise ValueError("belief capacity must be positive")
    previous = torch.from_numpy(np.stack([item.previous_executed_action for item in mapped])).to(
        device=target_device, dtype=torch.float32
    )
    temporal = LingBotUnifiedStepBatch(
        lane_ids=tuple(lane_ids),
        episode_keys=resolved_episode_keys,
        frame_indices=resolved_frame_indices,
        reset=resolved_reset,
        optimizer_step=optimizer_step,
        elapsed_time=torch.tensor(
            [item.elapsed_time_s for item in mapped],
            device=target_device,
            dtype=torch.float32,
        ),
        previous_executed_action=previous,
        previous_action_valid=torch.tensor(
            [item.previous_action_valid for item in mapped],
            device=target_device,
            dtype=torch.bool,
        ),
        modality_geometry_valid=torch.zeros(
            len(mapped),
            graph_config.modality_count,
            capacity,
            graph_config.codec.geometry_dim,
            device=target_device,
            dtype=torch.bool,
        ),
    )
    return LingBotCALVINTrainingBatch(
        host_items=tuple(item.feature_transform_item() for item in mapped),
        temporal=temporal,
        sample_keys=tuple(item.sample_key for item in mapped),
    )


def build_lingbot_calvin_stream_plan(
    dataset: CalvinStatefulTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
) -> FrozenEpisodeStreamPlan:
    """Bind CALVIN's immutable segment manifest to the shared episode planner."""

    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("CALVIN stream planning requires a content-addressed dataset manifest")
    return FrozenEpisodeStreamPlan(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        episodes=tuple(
            EpisodeSampleSequence(
                episode_key=episode.episode_key,
                sample_keys=episode.sample_keys,
            )
            for episode in dataset.episode_manifest
        ),
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
    )


def build_planned_lingbot_calvin_batch(
    plan: FrozenEpisodeStreamPlan,
    dataset: CalvinStatefulTransitionDataset,
    *,
    optimizer_step: int,
    rank: int,
    world_size: int,
    gradient_accumulation_steps: int,
    accumulation_index: int,
    graph: LingBotUnifiedBeliefGraph,
    capacity: int,
    device: torch.device | str = "cpu",
) -> PlannedLingBotCALVINBatch:
    """Resolve one deterministic stream shard and validate every plan/data edge."""

    microbatch = plan.microbatch_for_rank(
        optimizer_step,
        rank=rank,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        accumulation_index=accumulation_index,
    )
    samples = tuple(dataset.by_key(item.sample.sample_key) for item in microbatch.transitions)
    for planned, sample in zip(microbatch.transitions, samples, strict=True):
        if (
            sample.episode_key != planned.episode_key
            or sample.transition_index != planned.transition_index
        ):
            raise RuntimeError("frozen episode plan and CALVIN transition metadata disagree")
    lane_lookup = {lane_id: index for index, lane_id in enumerate(plan.lane_ids)}
    lane_ids = tuple(lane_lookup[item.lane_id] for item in microbatch.transitions)
    training = build_lingbot_calvin_training_batch(
        samples,
        lane_ids=lane_ids,
        optimizer_step=optimizer_step,
        graph=graph,
        capacity=capacity,
        device=device,
        episode_keys=tuple(item.episode_instance_id for item in microbatch.transitions),
        frame_indices=tuple(item.transition_index for item in microbatch.transitions),
        reset=tuple(item.transition_index == 0 for item in microbatch.transitions),
    )
    return PlannedLingBotCALVINBatch(
        training=training,
        plan_microbatch=microbatch,
        plan_sha256=plan.plan_sha256,
        augmentation_seeds=tuple(item.sample.augmentation_seed for item in microbatch.transitions),
        flow_noise_seeds=tuple(item.sample.flow_noise_seed for item in microbatch.transitions),
        flow_timestep_seeds=tuple(
            item.sample.flow_timestep_seed for item in microbatch.transitions
        ),
    )

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.lingbot_calvin import map_calvin_action_to_lingbot
from picf_next.data.lingbot_libero import LINGBOT_VLA2_FEATURE_SLICES
from picf_next.lingbot_native.calvin import (
    CollatedNativeCALVINBatch,
    PlannedNativeCALVINBatch,
    audit_native_calvin_model_inputs,
    build_native_calvin_context,
    build_native_calvin_continuation_batch,
    build_native_calvin_physical_control_chunks,
    build_native_calvin_physical_episode_domain,
    build_native_calvin_physical_sample_plan,
    build_native_calvin_physical_stream_plan,
    build_native_calvin_physical_training_batch,
    build_native_calvin_replay_batch,
    build_native_calvin_stream_plan,
    build_native_calvin_training_batch,
    build_native_calvin_training_stream_plan,
    build_planned_native_calvin_batch,
    collate_native_calvin_training_batch,
    materialize_native_flow_randomness,
    select_native_calvin_physical_prompt_segment,
    with_native_modalities,
    with_official_proprioception_modality,
)
from picf_next.lingbot_native.modalities import NativeModalityBatch, NativeModalityStream
from picf_next.lingbot_native.state import NativePosteriorState
from picf_next.training.control import (
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
)
from tests.test_calvin_data import _rewrite_language_intervals, _split_manifest, _write_split
from tests.test_lingbot_calvin import _dataset

_TRANSFORM_FIELDS = (
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
)


class _MutatingOfficialTransform:
    def __init__(self, *, extra_field: str | None = None) -> None:
        self.extra_field = extra_field

    def apply(self, item, policy_eval=False):
        assert not policy_eval
        item["action.lingbot"].add_(1000)
        random_value = random.random() + float(np.random.random()) + float(torch.rand(()))
        actions = item["action.lingbot"] + random_value
        result = {
            "action_is_pad": item["action.lingbot_is_pad"],
            "action_joint_mask": torch.ones(55, dtype=torch.bool),
            "actions": actions,
            "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            "images": torch.ones(2, 3, 4, 4),
            "img_masks": torch.ones(2, dtype=torch.bool),
            "joint_mask": torch.ones(actions.shape, dtype=torch.bool),
            "lang_masks": torch.ones(3, dtype=torch.bool),
            "lang_tokens": torch.ones(3, dtype=torch.long),
            "state": item["observation.state.lingbot"],
            "state_joint_mask": torch.ones(55, dtype=torch.bool),
        }
        if self.extra_field is not None:
            result[self.extra_field] = torch.tensor([1])
        return result


def _collator(items):
    return {name: torch.stack([item[name] for item in items]) for name in items[0]}


def _collated(
    tmp_path: Path,
) -> tuple[CollatedNativeCALVINBatch, PlannedNativeCALVINBatch]:
    dataset = _dataset(tmp_path)
    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="native-calvin-test",
        seed=17,
        global_batch_size=1,
        total_steps=2,
    )
    planned = build_planned_native_calvin_batch(
        plan,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    collated = collate_native_calvin_training_batch(
        planned.training,
        feature_transform=_MutatingOfficialTransform(),
        collator=_collator,
        augmentation_seeds=planned.augmentation_seeds,
        source_digest=planned.source_digest,
    )
    return collated, planned


def test_native_calvin_bridge_separates_targets_from_executed_controls(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    batch = build_native_calvin_training_batch(
        (dataset[0], dataset[1]),
        lane_ids=(2, 7),
        optimizer_step=11,
    )
    end = LINGBOT_VLA2_FEATURE_SLICES["end.position"]
    assert batch.routing.reset == (True, False)
    assert batch.controls.reset[:, 0].tolist() == [True, False]
    assert batch.controls.token_valid.all()
    assert batch.controls.acknowledged.all()
    assert not batch.controls.field_valid[0].any()
    assert batch.controls.field_valid[1].sum().item() == 7
    assert batch.controls.delta_time[0, 0] == 0
    torch.testing.assert_close(
        batch.controls.values[1, 0, end.start : end.start + 6],
        torch.from_numpy(dataset[0].record.action[:6].copy()),
    )
    assert batch.host_items[1]["action.lingbot"].data_ptr() != batch.controls.values.data_ptr()
    assert tuple(request.sample_key for request in batch.structural_target_requests) == (
        batch.routing.sample_keys
    )
    assert tuple(request.task_key for request in batch.structural_target_requests) == (
        dataset[0].host_sample.task_key,
        dataset[1].host_sample.task_key,
    )
    for item in batch.host_items:
        assert not any(
            name in key
            for key in item
            for name in ("sample_key", "episode_key", "scene_obs", "frame_index")
        )


def test_physical_prompt_overlay_is_deterministic_and_never_changes_event_identity(
    tmp_path: Path,
) -> None:
    legacy = _dataset(tmp_path)
    dataset = CalvinPhysicalTransitionDataset(legacy.index, action_horizon=4)
    sample_key = legacy.index.physical_event(13).event_key
    kwargs = {
        "sample_key": sample_key,
        "plan_sha256": "a" * 64,
        "episode_instance_id": "occurrence-0001",
    }
    first = select_native_calvin_physical_prompt_segment(dataset, **kwargs)
    replay = select_native_calvin_physical_prompt_segment(dataset, **kwargs)

    assert first == replay
    selected_segment_index, receipt = first
    assert selected_segment_index in dataset.candidate_segment_indices_by_key(sample_key)
    assert len(receipt) == 64
    selected = dataset.by_key(sample_key, selected_segment_index=selected_segment_index)
    assert selected.sample_key == sample_key
    assert selected.episode_key == "calvin-source-episode-00000000"


def test_physical_sample_plan_is_reset_only_and_replay_exact(tmp_path: Path) -> None:
    legacy = _dataset(tmp_path)
    dataset = CalvinPhysicalTransitionDataset(legacy.index, action_horizon=1)
    plan = build_native_calvin_physical_sample_plan(
        dataset,
        comparison_id="physical-sample-plan-test",
        seed=71,
        global_batch_size=1,
        total_steps=2,
    )

    first = build_planned_native_calvin_batch(
        plan,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        accumulation_index=0,
        maximum_control_tokens=1,
    )
    replay = build_planned_native_calvin_batch(
        plan,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        accumulation_index=0,
        maximum_control_tokens=1,
    )

    assert first.plan_microbatch == replay.plan_microbatch
    assert first.training.routing == replay.training.routing
    assert first.training.selected_segment_indices == replay.training.selected_segment_indices
    assert first.training.routing.reset == (True,)
    assert first.training.routing.episode_keys[0].startswith("sample-plan/step-00000000/")
    assert first.physical_prompt_selection_sha256 is not None
    assert len(first.physical_prompt_selection_receipts) == 1
    assert first.physical_prompt_selection_receipts == replay.physical_prompt_selection_receipts
    assert first.source_digest == replay.source_digest


def test_physical_plan_filters_events_without_required_raw_future_frames(
    tmp_path: Path,
) -> None:
    legacy = _dataset(tmp_path)
    dataset = CalvinPhysicalTransitionDataset(legacy.index, action_horizon=1)
    minimum_future = 4
    episodes = build_native_calvin_physical_episode_domain(
        dataset,
        minimum_future_source_frames=minimum_future,
    )
    eligible_keys = tuple(
        sample_key for episode in episodes for sample_key in episode.sample_keys
    )

    assert eligible_keys
    assert len(eligible_keys) < len(dataset.sample_keys)
    assert all(
        len(
            dataset.future_source_global_indices_by_key(
                sample_key,
                count=minimum_future,
            )
        )
        == minimum_future
        for sample_key in eligible_keys
    )
    plan = build_native_calvin_physical_stream_plan(
        dataset,
        comparison_id="physical-future-eligible-stream-test",
        seed=73,
        global_batch_size=1,
        total_steps=8,
        minimum_future_source_frames=minimum_future,
    )
    assert tuple(
        sample_key
        for episode in plan.episodes
        for sample_key in episode.sample_keys
    ) == eligible_keys

    with pytest.raises(ValueError, match="non-negative integer"):
        build_native_calvin_physical_episode_domain(
            dataset,
            minimum_future_source_frames=True,  # type: ignore[arg-type]
        )


def test_physical_control_chunks_preserve_every_raw_action_and_only_one_reset(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    _rewrite_language_intervals(split, [(12, 14), (16, 17)])
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=4)

    first_key = index.physical_event(12).event_key
    first = dataset.by_key(first_key, selected_segment_index=0)
    first_chunks = build_native_calvin_physical_control_chunks((first,), maximum_control_tokens=2)
    assert tuple(chunk.token_count for chunk in first_chunks) == (2, 1)
    assert torch.cat([chunk.reset[0] for chunk in first_chunks]).tolist() == [True, False, False]
    first_values = torch.cat([chunk.values[0] for chunk in first_chunks])
    torch.testing.assert_close(first_values[0], torch.zeros(55))
    torch.testing.assert_close(
        first_values[1:],
        torch.from_numpy(map_calvin_action_to_lingbot(first.incoming_control_span.raw_actions)),
    )

    gap_key = index.physical_event(16).event_key
    after_gap = dataset.by_key(gap_key, selected_segment_index=1)
    gap_chunks = build_native_calvin_physical_control_chunks((after_gap,), maximum_control_tokens=2)
    assert tuple(chunk.token_count for chunk in gap_chunks) == (2, 1)
    assert not torch.cat([chunk.reset[0] for chunk in gap_chunks]).any()
    reconstructed = torch.cat([chunk.values[0] for chunk in gap_chunks])
    torch.testing.assert_close(
        reconstructed,
        torch.from_numpy(map_calvin_action_to_lingbot(after_gap.incoming_control_span.raw_actions)),
    )
    batch = build_native_calvin_physical_training_batch(
        (after_gap,),
        maximum_control_tokens=2,
        lane_ids=(0,),
        optimizer_step=3,
    )
    assert batch.prior_control_chunks[-1] is batch.controls
    assert batch.physical_control_span_sha256 == (after_gap.incoming_control_span.sha256,)
    assert batch.selected_segment_indices == (1,)
    torch.testing.assert_close(
        torch.cat([chunk.values[0] for chunk in batch.prior_control_chunks]),
        reconstructed,
    )

    burnin_chunks = build_native_calvin_physical_control_chunks(
        (first,),
        maximum_control_tokens=4,
        gradient_suffix_control_tokens=1,
    )
    assert tuple(chunk.token_count for chunk in burnin_chunks) == (2, 1)
    torch.testing.assert_close(
        torch.cat([chunk.values[0] for chunk in burnin_chunks]),
        first_values,
    )
    assert torch.cat([chunk.reset[0] for chunk in burnin_chunks]).tolist() == [
        True,
        False,
        False,
    ]

    with pytest.raises(ValueError, match="no larger than maximum_control_tokens"):
        build_native_calvin_physical_control_chunks(
            (first,),
            maximum_control_tokens=2,
            gradient_suffix_control_tokens=3,
        )


def test_official_proprioception_is_one_typed_shared_host_token(tmp_path: Path) -> None:
    collated, _planned = _collated(tmp_path)
    enriched = with_official_proprioception_modality(collated)

    assert collated.modalities is None
    assert enriched.model_inputs is collated.model_inputs
    assert enriched.modalities is not None
    (stream,) = enriched.modalities.streams
    assert stream.name == "proprioception"
    torch.testing.assert_close(stream.tokens[:, 0], collated.model_inputs["state"])
    assert stream.valid.all()

    with pytest.raises(ValueError, match="already contains"):
        with_official_proprioception_modality(enriched)


def test_native_modality_attachment_is_sorted_complete_and_batch_bound(tmp_path: Path) -> None:
    collated, _planned = _collated(tmp_path)
    touch = NativeModalityBatch(
        (
            NativeModalityStream(
                name="touch",
                tokens=torch.randn(collated.routing.batch_size, 2, 4),
                valid=torch.ones(collated.routing.batch_size, 2, dtype=torch.bool),
            ),
        )
    )
    enriched = with_native_modalities(collated, touch)
    enriched = with_official_proprioception_modality(enriched)

    assert enriched.model_inputs is collated.model_inputs
    assert enriched.modalities is not None
    assert tuple(stream.name for stream in enriched.modalities.streams) == (
        "proprioception",
        "touch",
    )
    wrong_batch = NativeModalityBatch(
        (
            NativeModalityStream(
                name="geometry",
                tokens=torch.randn(collated.routing.batch_size + 1, 1, 4),
                valid=torch.ones(collated.routing.batch_size + 1, 1, dtype=torch.bool),
            ),
        )
    )
    with pytest.raises(ValueError, match="routing differ"):
        with_native_modalities(collated, wrong_batch)


def test_native_calvin_transform_is_retry_exact_and_strips_metadata(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    training = build_native_calvin_training_batch(
        (dataset[0], dataset[1]),
        lane_ids=(0, 1),
        optimizer_step=0,
    )
    original_actions = tuple(item["action.lingbot"].clone() for item in training.host_items)
    kwargs = {
        "feature_transform": _MutatingOfficialTransform(),
        "collator": _collator,
        "augmentation_seeds": (11, 22),
        "source_digest": "a" * 64,
    }
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    expected_process_rng = (random.random(), np.random.random(), torch.rand(()))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    first = collate_native_calvin_training_batch(training, **kwargs)
    actual_process_rng = (random.random(), np.random.random(), torch.rand(()))
    replay = collate_native_calvin_training_batch(training, **kwargs)

    assert set(first.model_inputs) == set(_TRANSFORM_FIELDS) - {
        "action_joint_mask",
        "state_joint_mask",
    }
    for item, original in zip(training.host_items, original_actions, strict=True):
        torch.testing.assert_close(item["action.lingbot"], original)
    for name in first.model_inputs:
        torch.testing.assert_close(first.model_inputs[name], replay.model_inputs[name])
    assert first.structural_target_requests == training.structural_target_requests
    assert replay.structural_target_requests == training.structural_target_requests
    assert not set(first.model_inputs) & {
        "structural_target_requests",
        "sample_key",
        "task_key",
    }
    assert actual_process_rng[0] == expected_process_rng[0]
    assert actual_process_rng[1] == expected_process_rng[1]
    torch.testing.assert_close(actual_process_rng[2], expected_process_rng[2])


def test_native_calvin_bridge_rejects_privileged_or_undeclared_transform_fields(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    training = build_native_calvin_training_batch(
        (dataset[0],),
        lane_ids=(0,),
        optimizer_step=0,
    )
    with pytest.raises(ValueError, match="privileged"):
        collate_native_calvin_training_batch(
            training,
            feature_transform=_MutatingOfficialTransform(extra_field="scene_obs"),
            collator=_collator,
            augmentation_seeds=(1,),
            source_digest="b" * 64,
        )
    model_inputs = {
        name: torch.ones(1, 1, 1)
        for name in (
            "action_is_pad",
            "actions",
            "image_grid_thw",
            "images",
            "img_masks",
            "joint_mask",
            "lang_masks",
            "lang_tokens",
            "state",
        )
    }
    model_inputs["scene_obs"] = torch.ones(1, 1)
    with pytest.raises(ValueError, match="privileged"):
        audit_native_calvin_model_inputs(model_inputs)


def test_native_calvin_context_requires_causal_lane_state_and_stays_unbound(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    training = build_native_calvin_training_batch(
        (dataset[0], dataset[1]),
        lane_ids=(0, 1),
        optimizer_step=0,
    )
    transformed = _MutatingOfficialTransform()
    collated = collate_native_calvin_training_batch(
        training,
        feature_transform=transformed,
        collator=_collator,
        augmentation_seeds=(1, 2),
        source_digest="c" * 64,
    )
    with pytest.raises(ValueError, match="non-reset"):
        build_native_calvin_context(collated, previous_state=None)
    previous = NativePosteriorState(torch.randn(2, 4, 8))
    context = build_native_calvin_context(collated, previous_state=previous)
    assert context.previous_state_valid is not None
    assert context.previous_state_valid.tolist() == [False, True]
    assert context.native_roles is None
    assert context.native_valid is None
    assert context.instruction_last_index is None


def test_native_calvin_plan_flow_randomness_is_replayable(
    tmp_path: Path,
) -> None:
    collated, planned = _collated(tmp_path)
    first = materialize_native_flow_randomness(collated, planned)
    replay = materialize_native_flow_randomness(collated, planned)
    assert first.structural_target_requests == collated.structural_target_requests
    assert replay.structural_target_requests == collated.structural_target_requests
    torch.testing.assert_close(first.model_inputs["noise"], replay.model_inputs["noise"])
    torch.testing.assert_close(first.model_inputs["time"], replay.model_inputs["time"])
    assert first.model_inputs["noise"].shape == first.model_inputs["actions"].shape
    assert first.model_inputs["time"].shape == (1,)
    assert 0.001 <= float(first.model_inputs["time"][0]) <= 1.0

    timestep_generator = torch.Generator(device="cpu").manual_seed(planned.flow_timestep_seeds[0])
    uniforms = torch.rand(2, generator=timestep_generator, dtype=torch.float32)
    expected_time = (
        uniforms[0].pow(1.0 / 1.5) / (uniforms[0].pow(1.0 / 1.5) + uniforms[1])
    ) * 0.999 + 0.001
    torch.testing.assert_close(first.model_inputs["time"][0].cpu(), expected_time)


def test_native_calvin_training_stream_factory_preserves_baseline_and_requires_complete_mixture(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    baseline = build_native_calvin_training_stream_plan(
        dataset,
        comparison_id="native-calvin-central-baseline",
        seed=37,
        global_batch_size=1,
        total_steps=2,
    )
    assert isinstance(baseline, FrozenEpisodeStreamPlan)

    with pytest.raises(ValueError, match="provided together"):
        build_native_calvin_training_stream_plan(
            dataset,
            comparison_id="native-calvin-partial-reset-mixture",
            seed=37,
            global_batch_size=1,
            total_steps=2,
            reset_numerator=1,
        )


def test_native_calvin_reset_mixture_uses_unique_real_source_disjoint_resets(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    plan = build_native_calvin_training_stream_plan(
        dataset,
        comparison_id="native-calvin-real-reset-mixture",
        seed=41,
        global_batch_size=1,
        total_steps=2,
        reset_numerator=1,
        reset_denominator=2,
    )
    assert isinstance(plan, FrozenResetMixtureStreamPlan)
    assert tuple(plan.component_for_step(step) for step in range(plan.total_steps)) == (
        "reset",
        "causal",
    )
    assert len(set(plan.reset_sample_keys)) == plan.reset_sample_count
    assert len(set(plan.reset_source_global_indices)) == plan.reset_sample_count
    assert all(dataset.by_key(key).transition_index == 0 for key in plan.reset_sample_keys)

    causal_source_indices = {
        dataset.source_global_index_by_key(transition.sample.sample_key)
        for step in range(plan.causal_plan.total_steps)
        for transition in plan.causal_plan.global_batch(step).transitions
    }
    assert causal_source_indices.isdisjoint(plan.reset_source_global_indices)


def test_native_calvin_structural_request_identity_is_fail_closed(tmp_path: Path) -> None:
    collated, _planned = _collated(tmp_path)
    wrong = replace(collated.structural_target_requests[0], sample_key="wrong/sample")
    with pytest.raises(ValueError, match="target and routing sample identities differ"):
        replace(collated, structural_target_requests=(wrong,))


def test_native_calvin_continuation_is_segment_bounded_and_replay_exact(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    first_key = dataset.sample_keys[0]
    locator = dataset.locator_by_key(first_key)
    assert locator.global_index == dataset.source_global_index_by_key(first_key)
    assert (
        dataset.task_key_by_key(first_key) == dataset.index.segments[locator.segment_index].task_key
    )
    assert dataset.available_future_transitions_by_key(first_key) == 3
    assert dataset.future_sample_keys(first_key, count=2) == dataset.sample_keys[1:3]
    with pytest.raises(ContractError, match="crosses a language reset"):
        dataset.future_sample_keys(first_key, count=4)

    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="native-calvin-continuation-test",
        seed=31,
        global_batch_size=1,
        total_steps=16,
    )
    primary = next(
        build_planned_native_calvin_batch(
            plan,
            dataset,
            optimizer_step=step,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        )
        for step in range(plan.total_steps)
        if dataset.available_future_transitions_by_key(
            plan.global_batch(step).transitions[0].sample.sample_key
        )
        >= 2
    )
    first = build_native_calvin_continuation_batch(primary, dataset, offset=1)
    replay = build_native_calvin_continuation_batch(primary, dataset, offset=1)
    second = build_native_calvin_continuation_batch(primary, dataset, offset=2)

    assert first.training.routing == replay.training.routing
    assert first.augmentation_seeds == replay.augmentation_seeds
    assert first.flow_noise_seeds == replay.flow_noise_seeds
    assert first.flow_timestep_seeds == replay.flow_timestep_seeds
    for first_item, replay_item in zip(
        first.training.host_items,
        replay.training.host_items,
        strict=True,
    ):
        for name in first_item:
            first_value = first_item[name]
            replay_value = replay_item[name]
            if isinstance(first_value, torch.Tensor):
                torch.testing.assert_close(first_value, replay_value)
            else:
                assert first_value == replay_value
    assert first.source_digest == replay.source_digest
    assert first.source_digest != second.source_digest
    assert first.training.routing.lane_ids == primary.training.routing.lane_ids
    assert first.training.routing.episode_keys == primary.training.routing.episode_keys
    assert first.training.routing.reset == (False,)
    assert first.training.routing.frame_indices == tuple(
        value + 1 for value in primary.training.routing.frame_indices
    )
    assert second.training.routing.frame_indices == tuple(
        value + 2 for value in primary.training.routing.frame_indices
    )

    collated = collate_native_calvin_training_batch(
        first.training,
        feature_transform=_MutatingOfficialTransform(),
        collator=_collator,
        augmentation_seeds=first.augmentation_seeds,
        source_digest=first.source_digest,
    )
    materialized = materialize_native_flow_randomness(collated, first)
    assert materialized.model_inputs["noise"].shape == materialized.model_inputs["actions"].shape


def test_native_calvin_reconstruction_prefix_and_randomness_are_exact(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    current = dataset.sample_keys[2]
    assert dataset.history_sample_keys(current) == dataset.sample_keys[:2]
    assert dataset.history_sample_keys(dataset.sample_keys[0]) == ()

    first = build_native_calvin_replay_batch(
        dataset,
        sample_key=dataset.sample_keys[1],
        lane_id=7,
        episode_instance_id="episode-instance/3",
        optimizer_step=11,
        replay_seed=29,
    )
    replay = build_native_calvin_replay_batch(
        dataset,
        sample_key=dataset.sample_keys[1],
        lane_id=7,
        episode_instance_id="episode-instance/3",
        optimizer_step=11,
        replay_seed=29,
    )
    changed = build_native_calvin_replay_batch(
        dataset,
        sample_key=dataset.sample_keys[1],
        lane_id=7,
        episode_instance_id="episode-instance/3",
        optimizer_step=11,
        replay_seed=30,
    )
    assert first.training.routing == replay.training.routing
    assert first.augmentation_seeds == replay.augmentation_seeds
    assert first.flow_noise_seeds == replay.flow_noise_seeds
    assert first.flow_timestep_seeds == replay.flow_timestep_seeds
    assert first.source_digest == replay.source_digest
    assert first.source_digest != changed.source_digest
    assert first.training.routing.reset == (False,)
    assert first.training.routing.frame_indices == (1,)

    collated = collate_native_calvin_training_batch(
        first.training,
        feature_transform=_MutatingOfficialTransform(),
        collator=_collator,
        augmentation_seeds=first.augmentation_seeds,
        source_digest=first.source_digest,
    )
    materialized = materialize_native_flow_randomness(collated, first)
    assert materialized.model_inputs["noise"].shape == materialized.model_inputs["actions"].shape

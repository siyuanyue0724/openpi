from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset  # noqa: E402
from picf_next.data.lingbot_calvin import map_calvin_transition_to_lingbot  # noqa: E402
from picf_next.data.lingbot_libero import LINGBOT_VLA2_FEATURE_SLICES  # noqa: E402
from picf_next.hosts.lingbot_calvin_training import (  # noqa: E402
    CollatedLingBotCALVINBatch,
    build_lingbot_calvin_stream_plan,
    build_lingbot_calvin_training_batch,
    build_planned_lingbot_calvin_batch,
    collate_lingbot_calvin_training_batch,
    materialize_lingbot_flow_randomness,
)
from picf_next.hosts.lingbot_unified import (  # noqa: E402
    LingBotUnifiedBeliefGraph,
    LingBotUnifiedGraphConfig,
)
from picf_next.hosts.lingbot_unified_training import (  # noqa: E402
    LingBotUnifiedLaneSession,
    LingBotUnifiedSessionConfig,
)
from picf_next.unified.codec import BeliefCodecConfig  # noqa: E402
from picf_next.unified.state import GeometrySchema  # noqa: E402
from picf_next.unified.temporal import assert_deploy_payload_is_causal  # noqa: E402
from tests.test_calvin_data import _split_manifest, _write_split  # noqa: E402


def _dataset(tmp_path: Path) -> CalvinStatefulTransitionDataset:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    return CalvinStatefulTransitionDataset(index, action_horizon=4)


def test_calvin_lingbot_mapping_separates_target_and_previous_action(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = map_calvin_transition_to_lingbot(dataset[0])
    second = map_calvin_transition_to_lingbot(dataset[1])
    end = LINGBOT_VLA2_FEATURE_SLICES["end.position"]
    effector = LINGBOT_VLA2_FEATURE_SLICES["effector.position"]

    assert not first.previous_action_valid
    assert np.count_nonzero(first.previous_executed_action) == 0
    assert second.previous_action_valid
    np.testing.assert_array_equal(
        second.previous_executed_action[end.start : end.start + 6],
        dataset[0].record.action[:6],
    )
    np.testing.assert_array_equal(
        second.previous_executed_action[effector.start : effector.start + 1],
        dataset[0].record.action[6:7],
    )
    assert not np.array_equal(second.previous_executed_action, second.actions[0])
    assert np.count_nonzero(second.action_valid) == 7
    assert np.count_nonzero(second.state_valid) == 14
    assert all(
        not value.flags.writeable
        for value in (
            second.state,
            second.actions,
            second.action_is_pad,
            second.previous_executed_action,
        )
    )


def test_calvin_lingbot_feature_item_matches_the_pinned_robot_mapping(tmp_path: Path) -> None:
    mapped = map_calvin_transition_to_lingbot(_dataset(tmp_path)[0])
    item = mapped.feature_transform_item()
    assert set(item) == {
        "action.lingbot",
        "action.lingbot_is_pad",
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
        "observation.state.lingbot",
        "task",
    }
    assert item["observation.state.lingbot"].shape == (55,)
    assert item["action.lingbot"].shape == (4, 55)
    assert item["action.lingbot_is_pad"].dtype == torch.bool
    assert item["observation.images.camera_top"].shape == (3, 200, 200)
    assert item["observation.images.camera_wrist_left"].shape == (3, 84, 84)


def test_calvin_training_bridge_keeps_targets_out_of_the_temporal_payload(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    graph = LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(8, 3, 2, 64),
            geometry_schema=GeometrySchema(
                names=("x", "y", "z"),
                units=("metre", "metre", "metre"),
                frame="camera",
            ),
            attention_value_width=64,
            num_layers=3,
            executed_action_dim=55,
        )
    )
    batch = build_lingbot_calvin_training_batch(
        (dataset[0], dataset[1]),
        lane_ids=(3, 7),
        optimizer_step=11,
        graph=graph,
        capacity=16,
    )
    assert batch.temporal.reset == (True, False)
    assert batch.temporal.frame_indices == (0, 1)
    assert batch.temporal.modality_geometry_valid.shape == (2, 1, 16, 3)
    assert batch.temporal.previous_executed_action.shape == (2, 55)
    assert batch.host_items[1]["action.lingbot"].data_ptr() != (
        batch.temporal.previous_executed_action.data_ptr()
    )
    assert_deploy_payload_is_causal({"temporal": batch.temporal})
    with pytest.raises(TypeError, match="capacity"):
        build_lingbot_calvin_training_batch(
            (dataset[0],),
            lane_ids=(0,),
            optimizer_step=0,
            graph=graph,
            capacity=True,
        )


def test_calvin_training_bridge_uses_official_transform_without_mutating_retry_input(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    graph = LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(8, 3, 2, 64),
            geometry_schema=GeometrySchema(
                names=("x", "y", "z"),
                units=("metre", "metre", "metre"),
                frame="camera",
            ),
            attention_value_width=64,
            num_layers=3,
            executed_action_dim=55,
        )
    )
    batch = build_lingbot_calvin_training_batch(
        (dataset[0], dataset[1]),
        lane_ids=(0, 1),
        optimizer_step=0,
        graph=graph,
        capacity=4,
    )
    original_actions = tuple(item["action.lingbot"].clone() for item in batch.host_items)

    class MutatingTransform:
        def apply(self, item, policy_eval=False):
            assert not policy_eval
            item["action.lingbot"].add_(1000)
            random_offset = random.random() + float(np.random.random()) + float(torch.rand(()))
            sample = torch.tensor([float(item["action.lingbot"][0, 0]) + random_offset])
            return {
                name: sample.clone()
                for name in (
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
                )
            }

    def collator(items):
        return {name: torch.stack([item[name] for item in items]) for name in items[0]}

    with pytest.raises(ValueError, match="source_digest"):
        collate_lingbot_calvin_training_batch(
            batch,
            feature_transform=MutatingTransform(),
            collator=collator,
            augmentation_seeds=(11, 22),
            source_digest=7,  # type: ignore[arg-type]
        )

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    expected_process_rng = (random.random(), np.random.random(), torch.rand(()))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    first = collate_lingbot_calvin_training_batch(
        batch,
        feature_transform=MutatingTransform(),
        collator=collator,
        augmentation_seeds=(11, 22),
        source_digest="a" * 64,
    )
    actual_process_rng = (random.random(), np.random.random(), torch.rand(()))
    replay = collate_lingbot_calvin_training_batch(
        batch,
        feature_transform=MutatingTransform(),
        collator=collator,
        augmentation_seeds=(11, 22),
        source_digest="a" * 64,
    )
    for item, original in zip(batch.host_items, original_actions, strict=True):
        torch.testing.assert_close(item["action.lingbot"], original)
    torch.testing.assert_close(first.model_inputs["actions"], replay.model_inputs["actions"])
    assert actual_process_rng[0] == expected_process_rng[0]
    assert actual_process_rng[1] == expected_process_rng[1]
    torch.testing.assert_close(actual_process_rng[2], expected_process_rng[2])
    assert first.temporal is batch.temporal
    assert first.sample_keys == batch.sample_keys
    assert first.source_digest == "a" * 64

    def incomplete_collator(items):
        complete = collator(items)
        complete.pop("image_grid_thw")
        return complete

    with pytest.raises(ValueError, match="image_grid_thw"):
        collate_lingbot_calvin_training_batch(
            batch,
            feature_transform=MutatingTransform(),
            collator=incomplete_collator,
            augmentation_seeds=(11, 22),
            source_digest="a" * 64,
        )

    with pytest.raises(ValueError, match="one value per host item"):
        collate_lingbot_calvin_training_batch(
            batch,
            feature_transform=MutatingTransform(),
            collator=collator,
            augmentation_seeds=(11,),
            source_digest="a" * 64,
        )


def test_calvin_stream_plan_is_content_addressed_ordered_and_replayable(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    graph = LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(8, 3, 2, 64),
            geometry_schema=GeometrySchema(
                names=("x", "y", "z"),
                units=("metre", "metre", "metre"),
                frame="camera",
            ),
            attention_value_width=64,
            num_layers=3,
            executed_action_dim=55,
        )
    )
    plan = build_lingbot_calvin_stream_plan(
        dataset,
        comparison_id="lingbot-calvin-test",
        seed=17,
        global_batch_size=1,
        total_steps=6,
    )
    first = build_planned_lingbot_calvin_batch(
        plan,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        accumulation_index=0,
        graph=graph,
        capacity=16,
    )
    replay = build_planned_lingbot_calvin_batch(
        plan,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        accumulation_index=0,
        graph=graph,
        capacity=16,
    )
    transition = first.plan_microbatch.transitions[0]
    assert first.training.sample_keys == (transition.sample.sample_key,)
    assert first.training.temporal.episode_keys == (transition.episode_instance_id,)
    assert first.training.temporal.frame_indices == (transition.transition_index,)
    assert first.training.temporal.lane_ids == (0,)
    assert first.augmentation_seeds == replay.augmentation_seeds
    assert first.flow_noise_seeds == replay.flow_noise_seeds
    assert first.flow_timestep_seeds == replay.flow_timestep_seeds
    assert first.plan_sha256 == plan.plan_sha256
    assert first.source_digest == replay.source_digest
    assert plan.dataset_manifest_sha256 == dataset.index.dataset_manifest.tree_sha256


def test_calvin_flow_randomness_is_source_addressed_and_retry_exact(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    graph = LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(8, 3, 2, 64),
            geometry_schema=GeometrySchema(
                names=("x", "y", "z"),
                units=("metre", "metre", "metre"),
                frame="camera",
            ),
            attention_value_width=64,
            num_layers=3,
            executed_action_dim=55,
        )
    )
    plan = build_lingbot_calvin_stream_plan(
        dataset,
        comparison_id="lingbot-calvin-flow-rng",
        seed=41,
        global_batch_size=1,
        total_steps=2,
    )
    planned = build_planned_lingbot_calvin_batch(
        plan,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        accumulation_index=0,
        graph=graph,
        capacity=4,
    )
    actions = torch.zeros(1, 4, 55, dtype=torch.bfloat16)
    model_inputs = {"actions": actions}
    collated = CollatedLingBotCALVINBatch(
        model_inputs=model_inputs,
        temporal=planned.training.temporal,
        sample_keys=planned.training.sample_keys,
        source_digest=planned.source_digest,
    )

    first = materialize_lingbot_flow_randomness(collated, planned)
    replay = materialize_lingbot_flow_randomness(collated, planned)

    assert set(model_inputs) == {"actions"}
    assert first.model_inputs["noise"].dtype == actions.dtype
    assert first.model_inputs["time"].dtype == actions.dtype
    assert first.model_inputs["noise"].shape == actions.shape
    assert first.model_inputs["time"].shape == (1,)
    assert 0.001 <= float(first.model_inputs["time"][0]) <= 1.0
    assert torch.equal(first.model_inputs["noise"], replay.model_inputs["noise"])
    assert torch.equal(first.model_inputs["time"], replay.model_inputs["time"])
    assert first.source_digest == planned.source_digest


def test_planned_calvin_stream_closes_the_transactional_posterior_loop(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    graph = LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(8, 3, 2, 64),
            geometry_schema=GeometrySchema(
                names=("x", "y", "z"),
                units=("metre", "metre", "metre"),
                frame="camera",
            ),
            attention_value_width=64,
            num_layers=3,
            executed_action_dim=55,
        )
    )
    plan = build_lingbot_calvin_stream_plan(
        dataset,
        comparison_id="lingbot-calvin-transaction",
        seed=29,
        global_batch_size=1,
        total_steps=4,
    )
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(
            model_family_digest="fixed-lingbot-test",
            capacity=4,
            birth_noise_seed=29,
        ),
    )

    previous_serialized: bytes | None = None
    for optimizer_step in range(4):
        planned = build_planned_lingbot_calvin_batch(
            plan,
            dataset,
            optimizer_step=optimizer_step,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            graph=graph,
            capacity=4,
        )
        prepared = session.prepare(planned.training.temporal)
        if previous_serialized is not None and not planned.training.temporal.reset[0]:
            assert prepared.context.previous_posterior.serialize() == previous_serialized

        torch.manual_seed(planned.flow_noise_seeds[0])
        prefix = torch.randn(1, 3, 64)
        action = torch.randn(1, 2, 32)
        inputs, _, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=[prefix, action],
            attention_mask=torch.ones(1, 5, 5, dtype=torch.bool),
            position_ids=torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone(),
            visual_pos_masks=torch.tensor([[True, True, False]]),
            context=prepared.context,
        )
        assert runtime is not None
        total_tokens = inputs[0].shape[1] + inputs[1].shape[1]
        runtime = graph.observe_joint_qkv(
            layer_index=graph.config.penultimate_layer,
            query_states=torch.randn(1, total_tokens, 4, 16),
            key_states=torch.randn(1, total_tokens, 2, 16),
            value_states=torch.randn(1, total_tokens, 2, 16),
            runtime=runtime,
        )
        graph.after_layer(
            layer_index=graph.config.penultimate_layer,
            outputs_embeds=inputs,
            runtime=runtime,
        )
        assert prepared.context.posterior is not None
        previous_serialized = prepared.context.posterior.detached().serialize()
        session.commit_many((prepared,))

    assert len(session.lane_bank) == 1

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
configuration = pytest.importorskip("olmo.hf_model.configuration_molmoact2")
modeling = pytest.importorskip("olmo.hf_model.modeling_molmoact2")
lerobot_modeling = pytest.importorskip("lerobot.policies.molmoact2.modeling_molmoact2")
adapter_module = pytest.importorskip("picf_next.hosts.molmoact2")
training_module = pytest.importorskip("picf_next.hosts.molmoact2_training")
core_module = pytest.importorskip("picf_next.models.core")
discovery_module = pytest.importorskip("picf_next.models.discovery")
evidence_module = pytest.importorskip("picf_next.models.evidence")
filter_module = pytest.importorskip("picf_next.models.filter")
temporal_module = pytest.importorskip("picf_next.models.temporal")
objective_module = pytest.importorskip("picf_next.models.objective")
dynamics_module = pytest.importorskip("picf_next.models.dynamics_loss")
set_loss_module = pytest.importorskip("picf_next.models.set_loss")
calvin_module = pytest.importorskip("picf_next.data.calvin")
calvin_rollout_module = pytest.importorskip("picf_next.data.calvin_rollout_targets")
rollout_target_module = pytest.importorskip("picf_next.data.rollout_targets")
control_module = pytest.importorskip("picf_next.training.control")
stateful_runner_module = pytest.importorskip("picf_next.training.stateful_runner")
stream_state_module = pytest.importorskip("picf_next.training.stream_state")

MolmoAct2ActionExpertConfig = configuration.MolmoAct2ActionExpertConfig
MolmoAct2AdapterConfig = configuration.MolmoAct2AdapterConfig
MolmoAct2Config = configuration.MolmoAct2Config
MolmoAct2TextConfig = configuration.MolmoAct2TextConfig
MolmoAct2VitConfig = configuration.MolmoAct2VitConfig
MolmoAct2ForConditionalGeneration = modeling.MolmoAct2ForConditionalGeneration
MolmoAct2Policy = lerobot_modeling.MolmoAct2Policy
MolmoAct2PICFActionExpert = adapter_module.MolmoAct2PICFActionExpert
NativeTokenBank = adapter_module.NativeTokenBank
PICFActionEvidence = adapter_module.PICFActionEvidence
install_adapter = adapter_module.install_molmoact2_lerobot_picf_adapter
prepare_lerobot_observation = adapter_module.prepare_molmoact2_lerobot_observation
MolmoAct2PICFTrainingBridge = training_module.MolmoAct2PICFTrainingBridge
MolmoAct2PICFTrainingConfig = training_module.MolmoAct2PICFTrainingConfig
MolmoAct2PICFJointTrainingBridge = training_module.MolmoAct2PICFJointTrainingBridge
MolmoAct2PICFTransition = training_module.MolmoAct2PICFTransition
CalvinStatefulLossTargets = training_module.CalvinStatefulLossTargets
CalvinStatefulLossTargetLayout = training_module.CalvinStatefulLossTargetLayout
CalvinStatefulLossTargetRequest = training_module.CalvinStatefulLossTargetRequest
CalvinGeometryOvershootingTargetBuilder = training_module.CalvinGeometryOvershootingTargetBuilder
CalvinStatefulMolmoAct2TrainingModule = training_module.CalvinStatefulMolmoAct2TrainingModule
compose_calvin_loss_target_builders = training_module.compose_calvin_loss_target_builders
assemble_calvin_molmoact2_transitions = training_module.assemble_calvin_molmoact2_transitions
assemble_calvin_stateful_molmoact2_transition = (
    training_module.assemble_calvin_stateful_molmoact2_transition
)
materialize_flow_randomness = training_module.materialize_molmoact2_flow_randomness
PICFCore = core_module.PICFCore
ObjectDiscoveryConfig = discovery_module.ObjectDiscoveryConfig
TaskIndependentObjectDiscovery = discovery_module.TaskIndependentObjectDiscovery
ModalityProjectionSpec = evidence_module.ModalityProjectionSpec
ModalityTokenSpan = evidence_module.ModalityTokenSpan
MultimodalBindingProjector = evidence_module.MultimodalBindingProjector
PersistentObjectFilter = filter_module.PersistentObjectFilter
ObjectBeliefBatch = temporal_module.ObjectBeliefBatch
TemporalFilterConfig = temporal_module.TemporalFilterConfig
GEOMETRY = synthetic_geometry_contract(3)
PICFObjective = objective_module.PICFObjective
PICFObjectiveConfig = objective_module.PICFObjectiveConfig
ObjectDynamicsCriterion = dynamics_module.ObjectDynamicsCriterion
ObjectDynamicsLossConfig = dynamics_module.ObjectDynamicsLossConfig
ObjectGeometryOvershootingConfig = dynamics_module.ObjectGeometryOvershootingConfig
ObjectGeometryOvershootingCriterion = dynamics_module.ObjectGeometryOvershootingCriterion
ObjectGeometryRolloutTarget = dynamics_module.ObjectGeometryRolloutTarget
ObjectLifecycleInventoryTarget = dynamics_module.ObjectLifecycleInventoryTarget
ObjectSetTarget = set_loss_module.ObjectSetTarget
CALVIN_OBSERVATION_SPECS = calvin_module.CALVIN_OBSERVATION_SPECS
CALVIN_HOST_IMAGE_KEYS = calvin_module.CALVIN_HOST_IMAGE_KEYS
CalvinEpisode = calvin_module.CalvinEpisode
CalvinDatasetIndex = calvin_module.CalvinDatasetIndex
CalvinLanguageSegment = calvin_module.CalvinLanguageSegment
CalvinMolmoAct2Sample = calvin_module.CalvinMolmoAct2Sample
CalvinStatefulTransitionSample = calvin_module.CalvinStatefulTransitionSample
CalvinStatefulTransitionDataset = calvin_module.CalvinStatefulTransitionDataset
CalvinTrainingWindow = calvin_module.CalvinTrainingWindow
decode_calvin_frame = calvin_module.decode_calvin_frame
FrozenSamplePlan = control_module.FrozenSamplePlan
EpisodeSampleSequence = control_module.EpisodeSampleSequence
FrozenEpisodeStreamPlan = control_module.FrozenEpisodeStreamPlan
RunProgress = control_module.RunProgress
StatefulEpisodeTrainingRunner = stateful_runner_module.StatefulEpisodeTrainingRunner
StatefulForwardOutput = stateful_runner_module.StatefulForwardOutput
PosteriorStreamStateGroup = stream_state_module.PosteriorStreamStateGroup
build_calvin_geometry_rollout_sample = calvin_rollout_module.build_calvin_geometry_rollout_sample
PhysicalObjectGeometryFrame = rollout_target_module.PhysicalObjectGeometryFrame


def _tiny_host() -> MolmoAct2ForConditionalGeneration:
    config = MolmoAct2Config(
        vit_config=MolmoAct2VitConfig(
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            image_default_input_size=(14, 14),
            image_patch_size=14,
            image_num_pos=1,
        ),
        adapter_config=MolmoAct2AdapterConfig(
            vit_layers=(0,),
            pooling_attention_mask=True,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=8,
            text_hidden_size=16,
        ),
        text_config=MolmoAct2TextConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            additional_vocab_size=None,
            num_hidden_layers=1,
            intermediate_size=16,
        ),
        action_expert_config=MolmoAct2ActionExpertConfig(
            max_action_horizon=3,
            max_action_dim=4,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            ffn_multiple_of=8,
            timestep_embed_dim=8,
            dropout=0.0,
            attn_dropout=0.0,
            qk_norm=True,
        ),
        add_action_expert=True,
        image_end_token_id=2,
        image_patch_id=3,
        max_action_dim=4,
        max_action_horizon=3,
        n_obs_steps=1,
        action_mode="both",
        state_format="discrete",
        state_token_start_id=10,
        num_state_tokens=8,
        action_start_token_id=20,
        action_end_token_id=21,
        action_token_start_id=22,
        num_action_tokens=16,
        enable_depth_reasoning=False,
    )
    host = MolmoAct2ForConditionalGeneration(config)
    host.config.eos_token_id = 2
    host.model.config.eos_token_id = 2
    torch.manual_seed(41)
    with torch.no_grad():
        expert = host.model.action_expert
        for block in expert.blocks:
            torch.nn.init.xavier_uniform_(block.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.linear.weight)
    return host


def _policy(*, gradient_checkpointing: bool) -> MolmoAct2Policy:
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_mode="continuous",
        inference_action_mode="continuous",
        num_flow_timesteps=2,
        num_inference_steps=2,
        flow_matching_cutoff=1.0,
        flow_matching_time_offset=0.0,
        flow_matching_time_scale=1.0,
        flow_matching_beta_alpha=1.5,
        flow_matching_beta_beta=1.0,
        mask_action_dim_padding=False,
        enable_inference_cuda_graph=False,
        n_action_steps=3,
        chunk_size=3,
        output_features={"action": SimpleNamespace(shape=(4,))},
        rtc_config=None,
        per_episode_seed=False,
        eval_seed=None,
        gradient_checkpointing=gradient_checkpointing,
        model_dtype="float32",
        enable_knowledge_insulation=False,
        enable_lora_vlm=False,
        optimizer_lr=1e-5,
        optimizer_vit_lr=1e-5,
        optimizer_connector_lr=1e-5,
        optimizer_action_expert_lr=3e-4,
    )
    policy.model = _tiny_host()
    policy.action_layer_adapter = None
    policy.rtc_processor = None
    lerobot_modeling._patch_training_kv_collection(policy._backbone())
    policy.train()
    return policy


def _fixed_inputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(43)
    batch = {"action": torch.randn(2, 3, 4)}
    model_inputs = {
        "input_ids": torch.tensor([[7, 8, 9], [9, 8, 7]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    return batch, model_inputs


def _visual_observation_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(45)
    return {
        "input_ids": torch.tensor([[3, 2, 7, 8]], dtype=torch.long).expand(batch_size, -1),
        "pixel_values": torch.randint(
            0,
            256,
            (batch_size, 1, 14 * 14 * 3),
            dtype=torch.uint8,
            generator=generator,
        ),
        "image_token_pooling": torch.tensor([[0, -1, -1, -1]], dtype=torch.long).expand(
            batch_size, -1
        ),
        "image_grids": torch.tensor([[1, 1, 0, 0]], dtype=torch.long).expand(batch_size, -1),
        "image_num_crops": torch.ones(batch_size, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
    }


def _two_camera_visual_observation_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(46)
    return {
        "input_ids": torch.tensor([[3, 2, 3, 2]], dtype=torch.long).expand(batch_size, -1),
        "pixel_values": torch.randint(
            0,
            256,
            (batch_size * 2, 1, 14 * 14 * 3),
            dtype=torch.uint8,
            generator=generator,
        ),
        "image_token_pooling": torch.tensor([[0, -1, -1, -1]], dtype=torch.long).expand(
            batch_size * 2, -1
        ),
        "image_grids": torch.tensor([[1, 1, 0, 0]], dtype=torch.long).expand(batch_size * 2, -1),
        "image_num_crops": torch.ones(batch_size * 2, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
    }


def _evidence() -> PICFActionEvidence:
    torch.manual_seed(47)
    object_valid = torch.tensor([[True, True, False], [True, False, False]])
    return PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "vision",
                torch.randn(2, 5, 6),
                torch.ones(2, 5, dtype=torch.bool),
            ),
        ),
        object_address=(
            torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1) * object_valid.unsqueeze(-1)
        ),
        object_value=torch.randn(2, 3, 9) * object_valid.unsqueeze(-1),
        object_valid=object_valid,
        object_log_prior=torch.tensor([[-0.1, -0.4, 0.0], [-0.2, 0.0, 0.0]]),
    )


def _picf_core(*, action_dim: int = 4) -> PICFCore:
    projector = MultimodalBindingProjector(
        (ModalityProjectionSpec("vision", token_dim=6),),
        binding_dim=8,
    )
    discovery = TaskIndependentObjectDiscovery(
        ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=12,
            num_queries=3,
            num_layers=2,
            num_heads=3,
            address_dim=4,
            content_dim=2,
            geometry_dim=3,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
        )
    )
    with torch.no_grad():
        discovery.existence_head.weight.zero_()
        discovery.existence_head.bias.fill_(6.0)
    temporal_config = TemporalFilterConfig(
        address_dim=4,
        content_dim=2,
        geometry_dim=3,
        geometry_contract=GEOMETRY,
        action_dim=action_dim,
        reference_delta_t_s=0.1,
        hidden_dim=12,
        num_layers=2,
        num_heads=3,
    )
    return PICFCore(projector, discovery, PersistentObjectFilter(temporal_config))


def _empty_belief() -> ObjectBeliefBatch:
    valid = torch.zeros(2, 3, dtype=torch.bool)
    return ObjectBeliefBatch(
        address_mean=torch.zeros(2, 3, 4),
        content_mean=torch.zeros(2, 3, 2),
        geometry_mean=torch.zeros(2, 3, 3),
        geometry_covariance_diag=torch.zeros(2, 3, 3),
        existence_logits=torch.zeros(2, 3),
        visibility_given_existence_logits=torch.zeros(2, 3),
        measurement_age_s=torch.zeros(2, 3),
        valid=valid,
        age=torch.zeros(2, 3, dtype=torch.long),
    )


def _training_bridge(
    *,
    require_explicit_flow_randomness: bool = False,
) -> MolmoAct2PICFTrainingBridge:
    policy = _policy(gradient_checkpointing=False)
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=22,
    )
    install_adapter(policy, adapter)
    for branch in adapter.dense_branches:
        branch.gate.data.fill_(0.2)
    for branch in adapter.object_branches:
        branch.gate.data.fill_(0.2)
    return MolmoAct2PICFTrainingBridge(
        policy,
        _picf_core(),
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=2,
            gradient_transitions=2,
            picf_core_lr=2e-4,
            require_explicit_flow_randomness=require_explicit_flow_randomness,
        ),
    )


def _planned_samples():
    plan = FrozenSamplePlan(
        dataset_id="molmo-bridge-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        sample_keys=("episode-0/window-0", "episode-1/window-0"),
        comparison_id="molmo-explicit-flow-fixture",
        seed=83,
        global_batch_size=2,
        total_steps=2,
    )
    return plan.global_batch(0).samples


def _training_transitions() -> tuple[MolmoAct2PICFTransition, ...]:
    transitions = []
    generator = torch.Generator().manual_seed(61)
    for index in range(4):
        tokens = torch.randn(2, 5, 6, generator=generator, requires_grad=True)
        bank = NativeTokenBank(
            "vision",
            tokens,
            torch.ones(2, 5, dtype=torch.bool),
        )
        host_batch = None
        if index >= 2:
            host_batch = {
                "action": torch.randn(2, 3, 4, generator=generator),
                "input_ids": torch.tensor([[7, 8, 9], [9, 8, 7]], dtype=torch.long),
                "attention_mask": torch.ones(2, 3, dtype=torch.long),
            }
        transitions.append(
            MolmoAct2PICFTransition(
                native_banks=(bank,),
                previous_executed_action=torch.randn(2, 4, generator=generator),
                delta_t_s=torch.full((2,), 0.1),
                host_batch=host_batch,
            )
        )
    return tuple(transitions)


def _calvin_window(
    start: int,
    *,
    segment_index: int = 0,
    instruction: str = "move the block",
) -> CalvinTrainingWindow:
    episode = CalvinEpisode(0, start, start + 4)
    segment = CalvinLanguageSegment(
        segment_index,
        start,
        start + 4,
        "move_block",
        instruction,
        0,
    )
    records = []
    for offset in range(4):
        frame = {
            "robot_obs": np.zeros(15, dtype=np.float64),
            "actions": np.zeros(7, dtype=np.float64),
            "rel_actions": np.full(7, 0.01 * (offset + 1), dtype=np.float64),
        }
        for source_key, _contract_key, shape, dtype, _units in CALVIN_OBSERVATION_SPECS:
            frame[source_key] = np.zeros(shape, dtype=dtype)
        records.append(
            decode_calvin_frame(
                frame,
                source_path=Path(f"episode_{start + offset:07d}.npz"),
                dataset_id="calvin-test",
                dataset_revision="sha256:test",
                episode=episode,
                segment=segment,
                global_index=start + offset,
                verify_relative_action=False,
            )
        )
    return CalvinTrainingWindow(segment=segment, records=tuple(records))


def _calvin_stateful_sample(
    window: CalvinTrainingWindow,
    transition_index: int,
) -> CalvinStatefulTransitionSample:
    record = window.records[transition_index]
    previous = (
        np.zeros(7, dtype=np.float32)
        if transition_index == 0
        else np.asarray(window.records[transition_index - 1].action, dtype=np.float32).copy()
    )
    previous.setflags(write=False)
    action = np.asarray(record.action, dtype=np.float32)[None].copy()
    action.setflags(write=False)
    action_is_pad = np.zeros(1, dtype=np.bool_)
    action_is_pad.setflags(write=False)
    arrays = {item.key: item.value for item in record.array_observations}
    host_sample = CalvinMolmoAct2Sample(
        observation={
            CALVIN_HOST_IMAGE_KEYS[0]: arrays["observation.images.rgb_static"],
            CALVIN_HOST_IMAGE_KEYS[1]: arrays["observation.images.rgb_gripper"],
            "observation.state": record.state,
            "task": record.task,
        },
        action=action,
        action_is_pad=action_is_pad,
        source_global_index=record.global_index,
        task_key=window.segment.task_key,
    )
    return CalvinStatefulTransitionSample(
        sample_key=(
            f"calvin-language-segment-{window.segment.index:08d}/"
            f"transition-{transition_index:08d}-frame-{record.global_index:08d}"
        ),
        episode_key=f"calvin-language-segment-{window.segment.index:08d}",
        transition_index=transition_index,
        record=record,
        previous_executed_action=previous,
        host_sample=host_sample,
    )


def _stateful_calvin_dataset(root: Path) -> CalvinStatefulTransitionDataset:
    root.mkdir(parents=True)
    episode = CalvinEpisode(0, 10, 23)
    segments = (
        CalvinLanguageSegment(0, 10, 13, "move_block", "move the block", 0),
        CalvinLanguageSegment(1, 20, 23, "turn_on_led", "turn on the led", 0),
    )
    for global_index in (*range(10, 14), *range(20, 24)):
        relative = np.array(
            [
                (global_index - 9) * 0.01,
                -(global_index - 9) * 0.005,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
        absolute = np.zeros(7, dtype=np.float64)
        absolute[:3] = relative[:3] * 0.02
        absolute[3:6] = relative[3:6] * 0.05
        absolute[-1] = relative[-1]
        frame = {
            "robot_obs": np.zeros(15, dtype=np.float64),
            "actions": absolute,
            "rel_actions": relative,
        }
        for source_key, _contract_key, shape, dtype, _units in CALVIN_OBSERVATION_SPECS:
            frame[source_key] = np.zeros(shape, dtype=dtype)
        np.savez(root / f"episode_{global_index:07d}.npz", **frame)
    index = CalvinDatasetIndex(
        split_root=root,
        dataset_id="calvin-stateful-fixture",
        dataset_revision="sha256:fixture",
        control_hz=30,
        episodes=(episode,),
        segments=segments,
    )
    return CalvinStatefulTransitionDataset(index, action_horizon=3)


def test_calvin_geometry_rollout_uses_outgoing_actions_and_stops_at_segment_boundary(
    tmp_path: Path,
) -> None:
    dataset = _stateful_calvin_dataset(tmp_path / "calvin-rollout")
    geometry_calls = []

    def geometry_provider(segment_index: int, global_index: int):
        geometry_calls.append((segment_index, global_index))
        geometry = torch.tensor([[float(global_index), 0.0, 0.0]])
        return PhysicalObjectGeometryFrame(
            identity_keys=(f"segment:{segment_index}/object:0",),
            geometry=geometry,
            geometry_variance=torch.full_like(geometry, 0.01),
            geometry_supervised=torch.ones_like(geometry, dtype=torch.bool),
            geometry_contract=GEOMETRY,
        )

    rollout = build_calvin_geometry_rollout_sample(
        dataset.index,
        segment_index=0,
        global_index=11,
        maximum_horizon=4,
        supervised_horizons=(1, 2),
        geometry_contract=GEOMETRY,
        geometry_provider=geometry_provider,
    )

    assert rollout.executed_actions.shape == (2, 7)
    np.testing.assert_allclose(
        rollout.executed_actions.numpy(),
        np.stack((dataset.index.action(11), dataset.index.action(12))),
    )
    assert rollout.delta_t_s.tolist() == pytest.approx([1.0 / 30.0, 1.0 / 30.0])
    assert geometry_calls == [(0, 12), (0, 13)]
    assert rollout.geometry_frames[0].geometry[0, 0] == 12.0
    assert rollout.geometry_frames[1].geometry[0, 0] == 13.0

    geometry_calls.clear()
    boundary = build_calvin_geometry_rollout_sample(
        dataset.index,
        segment_index=0,
        global_index=12,
        maximum_horizon=4,
        supervised_horizons=(1, 2),
        geometry_contract=GEOMETRY,
        geometry_provider=geometry_provider,
    )
    assert boundary.executed_actions.shape == (1, 7)
    assert geometry_calls == [(0, 13)]


def test_calvin_geometry_target_builder_batches_post_forward_physical_rollouts(
    tmp_path: Path,
) -> None:
    dataset = _stateful_calvin_dataset(tmp_path / "calvin-geometry-builder")
    calls = []

    def geometry_provider(segment_index: int, global_index: int):
        calls.append((segment_index, global_index))
        geometry = torch.tensor([[float(global_index), 1.0, 2.0]])
        return PhysicalObjectGeometryFrame(
            identity_keys=(f"segment:{segment_index}/object:0",),
            geometry=geometry,
            geometry_variance=torch.zeros_like(geometry),
            geometry_supervised=torch.ones_like(geometry, dtype=torch.bool),
            geometry_contract=GEOMETRY,
        )

    builder = CalvinGeometryOvershootingTargetBuilder(
        dataset.index,
        geometry_contract=GEOMETRY,
        geometry_provider=geometry_provider,
        maximum_horizon=3,
        supervised_horizons=(1, 3),
    )
    requests = (
        CalvinStatefulLossTargetRequest("sample-a", 0, 10, 101),
        CalvinStatefulLossTargetRequest("sample-b", 1, 20, 202),
    )
    layout = CalvinStatefulLossTargetLayout(
        token_valid=torch.ones(2, 4, dtype=torch.bool),
        spans=(ModalityTokenSpan("vision", 0, 4),),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.float32,
    )

    result = builder(requests, layout)

    target = result.geometry_rollout_target
    assert target is not None
    assert target.executed_actions.shape == (2, 3, 7)
    np.testing.assert_allclose(target.executed_actions[0, 0].numpy(), dataset.index.action(10))
    np.testing.assert_allclose(target.executed_actions[1, 2].numpy(), dataset.index.action(22))
    assert target.identity_keys[0] == (
        ("segment:0/object:0",),
        (None,),
        ("segment:0/object:0",),
    )
    assert target.identity_keys[1] == (
        ("segment:1/object:0",),
        (None,),
        ("segment:1/object:0",),
    )
    assert calls == [(0, 11), (0, 13), (1, 21), (1, 23)]


def test_calvin_loss_target_layout_separates_supervision_and_rollout_dtypes() -> None:
    layout = CalvinStatefulLossTargetLayout(
        token_valid=torch.ones(1, 2, dtype=torch.bool),
        spans=(ModalityTokenSpan("vision", 0, 2),),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.bfloat16,
    )

    assert layout.target_dtype == torch.float32
    assert layout.rollout_input_dtype == torch.bfloat16
    with pytest.raises(ValueError, match="canonical float32"):
        replace(layout, target_dtype=torch.bfloat16)


def test_calvin_loss_target_composition_rejects_implicit_precedence() -> None:
    request = CalvinStatefulLossTargetRequest("sample", 0, 10, 0)
    layout = CalvinStatefulLossTargetLayout(
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        spans=(ModalityTokenSpan("vision", 0, 1),),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.float32,
    )

    def first(_requests, _layout):
        return CalvinStatefulLossTargets(set_targets=())

    def second(_requests, _layout):
        return CalvinStatefulLossTargets(lifecycle_targets=())

    merged = compose_calvin_loss_target_builders(first, second)((request,), layout)
    assert merged.set_targets == ()
    assert merged.lifecycle_targets == ()

    duplicate = compose_calvin_loss_target_builders(first, first)
    with pytest.raises(ValueError, match="multiple.*set_targets"):
        duplicate((request,), layout)


def test_calvin_loss_target_request_requires_unambiguous_segment_identity() -> None:
    with pytest.raises(ValueError, match="segment"):
        CalvinStatefulLossTargetRequest("sample", -1, 10, 0)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_official_joint_flow_bridge_is_exact_then_backpropagates(
    gradient_checkpointing: bool,
) -> None:
    policy = _policy(gradient_checkpointing=gradient_checkpointing)
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    )
    install_adapter(policy, adapter)
    context = adapter.prepare_picf_context(_evidence())
    batch, model_inputs = _fixed_inputs()
    timesteps = torch.tensor([[0.2, 0.6], [0.3, 0.8]])
    noise = torch.randn(2, 2, 3, 4, generator=torch.Generator().manual_seed(53))

    baseline, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=batch,
        model_inputs=model_inputs,
        timesteps=timesteps,
        noise=noise,
    )
    zero_gate, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=batch,
        model_inputs=model_inputs,
        timesteps=timesteps,
        noise=noise,
        action_layer_context=context,
    )
    assert torch.equal(zero_gate, baseline)

    for branch in adapter.dense_branches:
        branch.gate.data.fill_(0.2)
    for branch in adapter.object_branches:
        branch.gate.data.fill_(0.3)
    structured, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=batch,
        model_inputs=model_inputs,
        timesteps=timesteps,
        noise=noise,
        action_layer_context=context,
    )
    assert not torch.equal(structured, baseline)
    structured.backward()
    assert adapter.dense_k_proj["vision"].weight.grad is not None
    assert adapter.object_k_proj.weight.grad is not None

    adapter_parameter_ids = {id(parameter) for parameter in adapter.parameters()}
    optimizer_groups = policy.get_optim_params()
    matching_groups = [
        group
        for group in optimizer_groups
        if adapter_parameter_ids & {id(parameter) for parameter in group["params"]}
    ]
    assert len(matching_groups) == 1
    assert {id(parameter) for parameter in matching_groups[0]["params"]} == adapter_parameter_ids
    assert matching_groups[0]["lr"] == policy.config.optimizer_action_expert_lr


def test_same_forward_visual_preparation_matches_official_loss_and_calls_vit_once(
    monkeypatch,
) -> None:
    policy = _policy(gradient_checkpointing=False).eval()
    observation = _visual_observation_inputs()
    action_batch = {"action": torch.randn(2, 3, 4)}
    timesteps = torch.tensor([[0.2, 0.6], [0.3, 0.8]])
    noise = torch.randn(2, 2, 3, 4, generator=torch.Generator().manual_seed(57))

    raw_loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=action_batch,
        model_inputs=policy._model_inputs(observation),
        timesteps=timesteps,
        noise=noise,
    )

    original = policy._backbone().vision_backbone.encode_image
    calls = 0

    def counted(images):
        nonlocal calls
        calls += 1
        return original(images)

    monkeypatch.setattr(policy._backbone().vision_backbone, "encode_image", counted)
    prepared = prepare_lerobot_observation(policy, observation)
    assert prepared.vision_patch_bank is not None
    prepared_loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=action_batch,
        model_inputs=dict(prepared.model_inputs),
        timesteps=timesteps,
        noise=noise,
        action_condition_input_ids=prepared.action_condition_input_ids,
    )

    assert calls == 1
    assert torch.equal(prepared_loss, raw_loss)
    assert prepared.vision_patch_bank.tokens.shape == (2, 1, 8)
    assert torch.equal(
        prepared.vision_patch_bank.valid,
        torch.ones(2, 1, dtype=torch.bool),
    )
    assert "pixel_values" not in prepared.model_inputs
    assert "input_ids" not in prepared.model_inputs
    assert "inputs_embeds" in prepared.model_inputs
    assert torch.equal(prepared.action_condition_input_ids, observation["input_ids"])


def test_two_camera_dense_layout_matches_official_flattening_order() -> None:
    policy = _policy(gradient_checkpointing=False).eval()
    policy.config.image_keys = list(CALVIN_HOST_IMAGE_KEYS)
    observation = _two_camera_visual_observation_inputs()

    model_inputs = policy._model_inputs(observation)
    official_images, official_pooling = policy._backbone().merge_visual_inputs(
        input_ids=model_inputs["input_ids"],
        pixel_values=model_inputs["pixel_values"],
        image_token_pooling=model_inputs["image_token_pooling"],
        image_grids=model_inputs["image_grids"],
        image_num_crops=model_inputs["image_num_crops"],
        pixel_values_videos=None,
        video_token_pooling=None,
        video_grids=None,
    )
    prepared = prepare_lerobot_observation(policy, observation)

    assert official_images is not None and official_pooling is not None
    assert prepared.vision_patch_bank is not None
    assert prepared.vision_patch_layout is not None
    layout = prepared.vision_patch_layout
    assert layout.semantic_image_keys
    assert layout.tokens_per_row == 2
    assert tuple(span.image_key for span in layout.rows[0]) == CALVIN_HOST_IMAGE_KEYS
    assert tuple((span.start, span.stop) for span in layout.rows[0]) == ((0, 1), (1, 2))
    assert tuple((span.start, span.stop) for span in layout.rows[1]) == ((0, 1), (1, 2))
    assert prepared.vision_patch_bank.tokens.shape[:2] == (2, 2)
    assert prepared.vision_patch_bank.valid.all()
    assert torch.equal(official_pooling[:, :, 0], torch.tensor([[0, 1], [0, 1]]))


def test_dense_layout_marks_missing_camera_names_as_nonsemantic() -> None:
    policy = _policy(gradient_checkpointing=False).eval()
    prepared = prepare_lerobot_observation(policy, _visual_observation_inputs())

    assert prepared.vision_patch_layout is not None
    assert not prepared.vision_patch_layout.semantic_image_keys
    assert prepared.vision_patch_layout.rows[0][0].image_key == "__processor_image_0"


def test_same_forward_visual_preparation_rejects_target_and_video_fields() -> None:
    policy = _policy(gradient_checkpointing=False)
    observation = _visual_observation_inputs()
    with pytest.raises(ValueError, match="unsupported MolmoAct2 observation fields"):
        prepare_lerobot_observation(
            policy,
            {**observation, "object_mask_target": torch.ones(2, 1)},
        )
    with pytest.raises(NotImplementedError, match="video dense-token"):
        prepare_lerobot_observation(
            policy,
            {**observation, "pixel_values_videos": torch.zeros(2, 1)},
        )


def test_explicit_flow_randomness_is_reproducible_transition_local_and_rng_neutral() -> None:
    policy = _policy(gradient_checkpointing=False)
    actions = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(89))
    planned_samples = _planned_samples()

    torch.manual_seed(97)
    rng_before = torch.random.get_rng_state().clone()
    first_timesteps, first_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        actions,
        transition_index=2,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    repeated_timesteps, repeated_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        actions,
        transition_index=2,
    )
    value_changed_timesteps, value_changed_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        torch.full_like(actions, 1234.5),
        transition_index=2,
    )
    next_timesteps, next_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        actions,
        transition_index=3,
    )
    mixed_timesteps, mixed_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        actions,
        transition_index=(2, 3),
    )

    assert torch.equal(first_timesteps, repeated_timesteps)
    assert torch.equal(first_noise, repeated_noise)
    assert torch.equal(first_timesteps, value_changed_timesteps)
    assert torch.equal(first_noise, value_changed_noise)
    assert not torch.equal(first_timesteps[0], first_timesteps[1])
    assert not torch.equal(first_noise[0], first_noise[1])
    assert not torch.equal(first_timesteps, next_timesteps)
    assert not torch.equal(first_noise, next_noise)
    assert torch.equal(mixed_timesteps[0], first_timesteps[0])
    assert torch.equal(mixed_noise[0], first_noise[0])
    assert torch.equal(mixed_timesteps[1], next_timesteps[1])
    assert torch.equal(mixed_noise[1], next_noise[1])
    assert first_timesteps.shape == (2, 2)
    assert first_noise.shape == (2, 2, 3, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_explicit_flow_randomness_cuda_restores_cpu_and_device_rng() -> None:
    policy = _policy(gradient_checkpointing=False)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(103)
    actions = torch.randn(2, 3, 4, device=device, generator=generator)
    planned_samples = _planned_samples()

    torch.manual_seed(107)
    torch.cuda.manual_seed(109)
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone()
    first_timesteps, first_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        actions,
        transition_index=4,
    )

    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    repeated_timesteps, repeated_noise = materialize_flow_randomness(
        policy,
        planned_samples,
        actions,
        transition_index=4,
    )
    assert first_timesteps.device == device
    assert first_noise.device == device
    assert torch.equal(first_timesteps, repeated_timesteps)
    assert torch.equal(first_noise, repeated_noise)
    assert not torch.equal(first_timesteps[0], first_timesteps[1])
    assert not torch.equal(first_noise[0], first_noise[1])


def test_official_forward_consumes_explicit_flow_randomness_independent_of_global_rng() -> None:
    policy = _policy(gradient_checkpointing=False)
    action_batch, model_inputs = _fixed_inputs()
    batch = {**action_batch, **model_inputs}
    timesteps, noise = materialize_flow_randomness(
        policy,
        _planned_samples(),
        action_batch["action"],
        transition_index=1,
    )

    torch.manual_seed(101)
    first, _ = policy(
        batch,
        flow_timesteps=timesteps,
        flow_noise=noise,
    )
    torch.manual_seed(999)
    second, _ = policy(
        batch,
        flow_timesteps=timesteps,
        flow_noise=noise,
    )

    assert torch.equal(first, second)


def test_context_is_explicit_and_never_read_from_target_batch_fields() -> None:
    policy = _policy(gradient_checkpointing=False)
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    )
    install_adapter(policy, adapter)
    batch, model_inputs = _fixed_inputs()
    contaminated = {
        **batch,
        "object_mask_target": torch.rand(2, 3),
        "simulator_instance_id": torch.tensor([17, 29]),
        "task_owner_target": torch.tensor([0, 1]),
    }
    timesteps = torch.tensor([[0.2, 0.6], [0.3, 0.8]])
    noise = torch.randn(2, 2, 3, 4, generator=torch.Generator().manual_seed(59))

    clean_loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=batch,
        model_inputs=model_inputs,
        timesteps=timesteps,
        noise=noise,
    )
    contaminated_loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=contaminated,
        model_inputs=model_inputs,
        timesteps=timesteps,
        noise=noise,
    )
    assert torch.equal(clean_loss, contaminated_loss)


def test_official_inference_loop_has_bitwise_zero_gate_parity() -> None:
    policy = _policy(gradient_checkpointing=False).eval()
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    )
    install_adapter(policy, adapter)
    policy.eval()
    context = adapter.prepare_picf_context(_evidence())
    _, model_inputs = _fixed_inputs()

    baseline = policy.predict_action_chunk(
        model_inputs,
        generator=torch.Generator().manual_seed(67),
        num_steps=2,
    )
    zero_gate = policy.predict_action_chunk(
        model_inputs,
        generator=torch.Generator().manual_seed(67),
        num_steps=2,
        action_layer_context=context,
    )

    assert torch.equal(zero_gate, baseline)


@pytest.mark.parametrize(
    ("model_dtype", "torch_dtype"),
    [("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_prepared_embedding_inference_has_bitwise_raw_input_parity(
    model_dtype: str,
    torch_dtype: torch.dtype,
) -> None:
    policy = _policy(gradient_checkpointing=False)
    policy.config.model_dtype = model_dtype
    policy = policy.to(dtype=torch_dtype).eval()
    model_inputs = _visual_observation_inputs()
    prepared = prepare_lerobot_observation(policy, model_inputs)

    raw_action = policy.predict_action_chunk(
        model_inputs,
        generator=torch.Generator().manual_seed(71),
        num_steps=2,
    )
    prepared_action = policy.predict_action_chunk(
        prepared.model_inputs,
        generator=torch.Generator().manual_seed(71),
        num_steps=2,
        action_condition_input_ids=prepared.action_condition_input_ids,
    )

    assert "input_ids" not in prepared.model_inputs
    assert "inputs_embeds" in prepared.model_inputs
    assert torch.equal(prepared.action_condition_input_ids, model_inputs["input_ids"])
    assert prepared.vision_patch_bank is not None
    assert policy._action_expert().action_embed.weight.dtype == torch_dtype
    assert torch.equal(prepared_action, raw_action)


def test_prepared_embedding_inference_fails_closed_without_condition_tokens() -> None:
    policy = _policy(gradient_checkpointing=False).eval()
    prepared = prepare_lerobot_observation(policy, _visual_observation_inputs())

    with pytest.raises(ValueError, match="require action_condition_input_ids"):
        policy.predict_action_chunk(
            prepared.model_inputs,
            generator=torch.Generator().manual_seed(71),
            num_steps=2,
        )


def test_sequence_bridge_truncates_warmup_and_backpropagates_two_transitions() -> None:
    torch.manual_seed(67)
    bridge = _training_bridge()
    transitions = _training_transitions()
    output = bridge(transitions, _empty_belief())

    assert len(output.action_losses) == 2
    assert len(output.core_outputs) == 2
    assert len(output.evidences) == 2
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    output.loss.backward()

    assert transitions[0].native_banks[0].tokens.grad is None
    assert transitions[1].native_banks[0].tokens.grad is None
    assert transitions[2].native_banks[0].tokens.grad is not None
    assert transitions[3].native_banks[0].tokens.grad is not None
    gradient = bridge.core.discovery.ownership_query.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0.0
    transition_gradient = bridge.core.posterior_filter.transition.dynamic_head.weight.grad
    assert transition_gradient is not None and transition_gradient.abs().sum() > 0.0


def test_sequence_bridge_can_require_and_consume_explicit_flow_randomness() -> None:
    bridge = _training_bridge(require_explicit_flow_randomness=True)
    transitions = _training_transitions()
    with pytest.raises(ValueError, match="requires explicit flow randomness"):
        bridge(transitions, _empty_belief())

    samples = _planned_samples()
    explicit = []
    for index, transition in enumerate(transitions):
        if transition.host_batch is None:
            explicit.append(transition)
            continue
        timesteps, noise = materialize_flow_randomness(
            bridge.policy,
            samples,
            transition.host_batch["action"],
            transition_index=index,
        )
        explicit.append(
            replace(
                transition,
                flow_timesteps=timesteps,
                flow_noise=noise,
            )
        )

    output = bridge(tuple(explicit), _empty_belief())
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)

    mismatched = list(explicit)
    mismatched[2] = replace(mismatched[2], flow_noise=None)
    with pytest.raises(ValueError, match="must be supplied together"):
        bridge(tuple(mismatched), _empty_belief())


@pytest.mark.parametrize("invalid", [True, 2.5, 0, -1])
def test_flow_randomness_rejects_nonpositive_or_noninteger_timestep_count(invalid) -> None:
    policy = _policy(gradient_checkpointing=False)
    policy.config.num_flow_timesteps = invalid
    actions = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="num_flow_timesteps"):
        materialize_flow_randomness(
            policy,
            _planned_samples(),
            actions,
            transition_index=0,
        )


@pytest.mark.parametrize("invalid", [(0,), (0, -1), (0, True), "01"])
def test_flow_randomness_rejects_malformed_per_sample_transition_coordinates(invalid) -> None:
    policy = _policy(gradient_checkpointing=False)
    with pytest.raises(ValueError, match="one per planned sample"):
        materialize_flow_randomness(
            policy,
            _planned_samples(),
            torch.zeros(2, 3, 4),
            transition_index=invalid,
        )


def test_sequence_bridge_rejects_ignored_target_batch_on_detached_context() -> None:
    bridge = _training_bridge()
    transitions = list(_training_transitions())
    transitions[0] = replace(
        transitions[0],
        host_batch={"action": torch.zeros(2, 3, 4)},
    )
    with pytest.raises(ValueError, match="ignored host_batch"):
        bridge(tuple(transitions), _empty_belief())


def test_sequence_bridge_consumes_same_forward_molmo_patches_without_second_vit(
    monkeypatch,
) -> None:
    policy = _policy(gradient_checkpointing=False)
    core = PICFCore(
        MultimodalBindingProjector(
            (
                ModalityProjectionSpec("vision", token_dim=6),
                ModalityProjectionSpec("molmo_vision_patch", token_dim=8),
            ),
            binding_dim=8,
        ),
        TaskIndependentObjectDiscovery(
            ObjectDiscoveryConfig(
                input_dim=8,
                hidden_dim=12,
                num_queries=3,
                num_layers=2,
                num_heads=3,
                address_dim=4,
                content_dim=2,
                geometry_dim=3,
                geometry_contract=GEOMETRY,
                initial_variance=0.1,
            )
        ),
        PersistentObjectFilter(
            TemporalFilterConfig(
                address_dim=4,
                content_dim=2,
                geometry_dim=3,
                geometry_contract=GEOMETRY,
                action_dim=4,
                reference_delta_t_s=0.1,
                hidden_dim=12,
                num_layers=2,
                num_heads=3,
            )
        ),
    )
    with torch.no_grad():
        core.discovery.existence_head.weight.zero_()
        core.discovery.existence_head.bias.fill_(6.0)
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6, "molmo_vision_patch": 8},
        object_address_dim=4,
        object_value_dim=22,
    )
    install_adapter(policy, adapter)
    for branch in (*adapter.dense_branches, *adapter.object_branches):
        branch.gate.data.fill_(0.2)
    bridge = MolmoAct2PICFTrainingBridge(
        policy,
        core,
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=0,
            gradient_transitions=1,
            picf_core_lr=2e-4,
        ),
    )
    external_tokens = torch.randn(2, 2, 6, requires_grad=True)
    transition = MolmoAct2PICFTransition(
        native_banks=(
            NativeTokenBank(
                "vision",
                external_tokens,
                torch.ones(2, 2, dtype=torch.bool),
            ),
        ),
        previous_executed_action=torch.zeros(2, 4),
        delta_t_s=torch.full((2,), 0.1),
        host_observation_inputs=_visual_observation_inputs(),
        host_batch={"action": torch.randn(2, 3, 4)},
    )

    original = policy._backbone().vision_backbone.encode_image
    calls = 0

    def counted(images):
        nonlocal calls
        calls += 1
        return original(images)

    monkeypatch.setattr(policy._backbone().vision_backbone, "encode_image", counted)
    output = bridge((transition,), _empty_belief())
    assert calls == 1
    assert tuple(bank.modality for bank in output.evidences[0].dense_banks) == (
        "vision",
        "molmo_vision_patch",
    )
    output.loss.backward()
    assert external_tokens.grad is not None
    assert policy._backbone().vision_backbone.image_vit.patch_embedding.weight.grad is not None
    molmo_projection = bridge.core.projector.content_projection["molmo_vision_patch"]
    assert molmo_projection.weight.grad is not None


def test_sequence_bridge_accepts_same_forward_molmo_as_the_only_native_bank() -> None:
    policy = _policy(gradient_checkpointing=False)
    core = PICFCore(
        MultimodalBindingProjector(
            (ModalityProjectionSpec("molmo_vision_patch", token_dim=8),),
            binding_dim=8,
        ),
        TaskIndependentObjectDiscovery(
            ObjectDiscoveryConfig(
                input_dim=8,
                hidden_dim=12,
                num_queries=3,
                num_layers=1,
                num_heads=3,
                address_dim=4,
                content_dim=2,
                geometry_dim=3,
                geometry_contract=GEOMETRY,
                initial_variance=0.1,
            )
        ),
        PersistentObjectFilter(
            TemporalFilterConfig(
                address_dim=4,
                content_dim=2,
                geometry_dim=3,
                geometry_contract=GEOMETRY,
                action_dim=4,
                reference_delta_t_s=0.1,
                hidden_dim=12,
                num_layers=1,
                num_heads=3,
            )
        ),
    )
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"molmo_vision_patch": 8},
        object_address_dim=4,
        object_value_dim=22,
    )
    install_adapter(policy, adapter)
    bridge = MolmoAct2PICFTrainingBridge(
        policy,
        core,
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=0,
            gradient_transitions=1,
            picf_core_lr=2e-4,
        ),
    )
    transition = MolmoAct2PICFTransition(
        native_banks=(),
        previous_executed_action=torch.zeros(2, 4),
        delta_t_s=torch.full((2,), 0.1),
        host_observation_inputs=_visual_observation_inputs(),
        host_batch={"action": torch.randn(2, 3, 4)},
    )

    output = bridge((transition,), _empty_belief())

    assert tuple(bank.modality for bank in output.evidences[0].dense_banks) == (
        "molmo_vision_patch",
    )
    assert output.evidences[0].dense_banks[0].tokens.shape == (2, 1, 8)
    assert output.evidences[0].dense_banks[0].valid.all()


def test_stateful_runner_drives_official_molmo_bridge_with_carried_posterior() -> None:
    accelerate = pytest.importorskip("accelerate")
    policy = _policy(gradient_checkpointing=False)
    core = _picf_core()
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=22,
    )
    install_adapter(policy, adapter)
    for branch in (*adapter.dense_branches, *adapter.object_branches):
        branch.gate.data.fill_(0.2)
    bridge = MolmoAct2PICFTrainingBridge(
        policy,
        core,
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=0,
            gradient_transitions=1,
            picf_core_lr=2e-4,
        ),
    )
    core.requires_grad_(False)
    plan = FrozenEpisodeStreamPlan(
        dataset_id="molmo-stateful-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        episodes=(
            EpisodeSampleSequence(
                episode_key="episode-a",
                sample_keys=("a/0", "a/1", "a/2"),
            ),
            EpisodeSampleSequence(
                episode_key="episode-b",
                sample_keys=("b/0", "b/1", "b/2"),
            ),
        ),
        comparison_id="molmo-stateful-seed-97",
        seed=97,
        global_batch_size=2,
        total_steps=2,
    )
    progress = RunProgress(
        contract_sha256="c" * 64,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    streams = PosteriorStreamStateGroup.for_rank_partition(
        core.posterior_filter.config,
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        capacity=3,
        dtype=torch.float32,
        max_parameter_lag=0,
    )
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=1)
    optimizer = torch.optim.AdamW(bridge.get_optim_params(), lr=1e-4)
    bridge, optimizer = accelerator.prepare(bridge, optimizer)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=bridge,
        state_producer=core,
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=streams,
        max_grad_norm=1.0,
    )
    transition = _training_transitions()[2]
    initial_validity = []

    def forward_step(_microbatch, initial_belief, _loss_track_keys_by_row):
        initial_validity.append(initial_belief.valid.detach().clone())
        output = bridge((transition,), initial_belief)
        return StatefulForwardOutput(
            loss=output.loss,
            final_belief=output.final_belief,
            metrics=output.metrics,
        )

    first = runner.run_optimizer_step(forward_step)
    second = runner.run_optimizer_step(forward_step)
    assert first.parameter_version_after == 1
    assert second.parameter_version_before == 1
    assert not initial_validity[0].any()
    assert initial_validity[1].any()
    assert streams["accumulation-00000"].next_transition_indices == (2, 2)
    accelerator.end_training()


def test_calvin_stateful_production_module_closes_plan_target_and_runner_loop(
    tmp_path: Path,
) -> None:
    accelerate = pytest.importorskip("accelerate")
    dataset = _stateful_calvin_dataset(tmp_path / "calvin")
    policy = _policy(gradient_checkpointing=False)
    core = _picf_core(action_dim=7)
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=22,
    )
    install_adapter(policy, adapter)
    for branch in (*adapter.dense_branches, *adapter.object_branches):
        branch.gate.data.fill_(0.2)
    sequence_bridge = MolmoAct2PICFTrainingBridge(
        policy,
        core,
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=0,
            gradient_transitions=1,
            picf_core_lr=2e-4,
            require_explicit_flow_randomness=True,
        ),
    )
    joint_bridge = MolmoAct2PICFJointTrainingBridge(
        sequence_bridge,
        PICFObjective(
            PICFObjectiveConfig(
                action_weight=1.0,
                set_weight=0.2,
                dynamics_weight=0.2,
                binding_weight=0.0,
            ),
            dynamics_criterion=ObjectDynamicsCriterion(
                ObjectDynamicsLossConfig(
                    content_cosine_weight=1.0,
                    geometry_nll_weight=1.0,
                    survival_weight=1.0,
                    visibility_weight=1.0,
                )
            ),
            geometry_overshooting_criterion=ObjectGeometryOvershootingCriterion(
                ObjectGeometryOvershootingConfig(weight=0.25)
            ),
        ),
    )
    evidence_calls = []
    evidence_prefix_lengths = []
    target_calls = []
    host_calls = []
    core_forward_count = 0

    def record_core_forward(_module, _inputs, _output):
        nonlocal core_forward_count
        core_forward_count += 1

    core.register_forward_hook(record_core_forward)

    def build_native_banks(requests):
        evidence_calls.append(tuple(request.sample_key for request in requests))
        evidence_prefix_lengths.append(tuple(len(request.evidence_prefix) for request in requests))
        assert all(not hasattr(request.evidence_frame, "action") for request in requests)
        assert all(not hasattr(request.evidence_frame, "task") for request in requests)
        assert all(request.evidence_prefix[-1] is request.evidence_frame for request in requests)
        assert all(not hasattr(request, "flow_noise_seed") for request in requests)
        tokens = torch.stack(
            [
                torch.full(
                    (5, 6),
                    request.evidence_frame.timestamp_s
                    + float(request.augmentation_seed % 17) * 1e-4,
                )
                for request in requests
            ]
        )
        return (NativeTokenBank("vision", tokens, torch.ones(len(requests), 5, dtype=torch.bool)),)

    def build_host_batch(stateful_samples):
        host_calls.append(tuple(sample.transition_index for sample in stateful_samples))
        return {
            "action": torch.as_tensor(
                np.stack([sample.host_sample.action[:, :4] for sample in stateful_samples])
            ),
        }

    def build_host_observation_inputs(_evidence, _views):
        return {
            "input_ids": torch.tensor([[7, 8, 9], [9, 8, 7]], dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }

    def build_loss_targets(target_requests, layout):
        assert core_forward_count == len(target_calls) + 1
        assert not hasattr(layout, "binding_features")
        assert not hasattr(layout, "ownership")
        assert not hasattr(layout, "posterior")
        assert all(not hasattr(request, "sample") for request in target_requests)
        assert all(not hasattr(request, "action") for request in target_requests)
        assert all(not hasattr(request, "task") for request in target_requests)
        target_calls.append(tuple(request.source_global_index for request in target_requests))
        targets = []
        lifecycle_targets = []
        for batch_index in range(len(target_requests)):
            valid = layout.token_valid[batch_index]
            physical_key = (
                target_requests[batch_index].sample_key.split("/transition-", maxsplit=1)[0]
                + "/object-0"
            )
            ownership = torch.zeros(
                valid.shape[0],
                2,
                device=valid.device,
                dtype=layout.target_dtype,
            )
            ownership[valid, 0] = 1.0
            targets.append(
                ObjectSetTarget(
                    ownership=ownership,
                    token_valid=valid,
                    token_supervised=valid,
                    object_inventory_complete=False,
                    temporal_identity_keys=(physical_key,),
                )
            )
            lifecycle_targets.append(
                ObjectLifecycleInventoryTarget(
                    alive_identity_keys=(physical_key,),
                    inventory_complete=True,
                    visibility=torch.ones(1, device=valid.device, dtype=layout.target_dtype),
                    visibility_supervised=torch.ones(1, device=valid.device, dtype=torch.bool),
                )
            )
        physical_keys = tuple(
            request.sample_key.split("/transition-", maxsplit=1)[0] + "/object-0"
            for request in target_requests
        )
        batch_size = len(target_requests)
        rollout_horizon = 2
        rollout_geometry = torch.stack(
            [
                torch.full(
                    (rollout_horizon, 1, 3),
                    float(request.source_global_index) * 1e-3,
                    device=layout.token_valid.device,
                    dtype=layout.target_dtype,
                )
                for request in target_requests
            ]
        )
        return CalvinStatefulLossTargets(
            set_targets=tuple(targets),
            lifecycle_targets=tuple(lifecycle_targets),
            geometry_rollout_target=ObjectGeometryRolloutTarget(
                executed_actions=torch.zeros(
                    batch_size,
                    rollout_horizon,
                    7,
                    device=layout.token_valid.device,
                    dtype=layout.rollout_input_dtype,
                ),
                delta_t_s=torch.full(
                    (batch_size, rollout_horizon),
                    0.1,
                    device=layout.token_valid.device,
                    dtype=layout.rollout_input_dtype,
                ),
                step_valid=torch.ones(
                    batch_size,
                    rollout_horizon,
                    device=layout.token_valid.device,
                    dtype=torch.bool,
                ),
                identity_keys=tuple(
                    tuple((key,) for _step in range(rollout_horizon)) for key in physical_keys
                ),
                geometry=rollout_geometry,
                geometry_variance=torch.full_like(rollout_geometry, 0.01),
                geometry_supervised=torch.ones_like(rollout_geometry, dtype=torch.bool),
                geometry_contract=GEOMETRY,
            ),
        )

    module = CalvinStatefulMolmoAct2TrainingModule(
        dataset,
        joint_bridge,
        build_native_banks=build_native_banks,
        build_host_batch=build_host_batch,
        build_host_observation_inputs=build_host_observation_inputs,
        build_loss_targets=build_loss_targets,
        native_evidence_history_frames=4,
    )
    core.requires_grad_(False)
    plan = FrozenEpisodeStreamPlan(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256="a" * 64,
        episodes=tuple(
            EpisodeSampleSequence(episode.episode_key, episode.sample_keys)
            for episode in dataset.episode_manifest
        ),
        comparison_id="calvin-stateful-production-fixture",
        seed=109,
        global_batch_size=2,
        total_steps=2,
    )
    progress = RunProgress(
        contract_sha256="c" * 64,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    streams = PosteriorStreamStateGroup.for_rank_partition(
        core.posterior_filter.config,
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        capacity=3,
        dtype=torch.float32,
        max_parameter_lag=0,
    )
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=1)
    optimizer = torch.optim.AdamW(module.get_optim_params(), lr=1e-4)
    module, optimizer = accelerator.prepare(module, optimizer)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=module,
        state_producer=core,
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=streams,
        max_grad_norm=1.0,
    )

    first = runner.run_optimizer_step(module)
    second = runner.run_optimizer_step(module)

    assert first.parameter_version_after == 1
    assert second.parameter_version_after == 2
    assert host_calls == [(0, 0), (1, 1)]
    assert tuple(tuple(sorted(indices)) for indices in target_calls) == ((10, 20), (11, 21))
    assert len(evidence_calls) == 2
    assert evidence_prefix_lengths == [(1, 1), (2, 2)]
    assert all("transition-00000001" in key for key in evidence_calls[1])
    assert first.metrics[0]["picf_transitions"] == 1.0
    assert first.metrics[0]["picf_geometry_overshooting_active_horizons"] == 2.0
    assert first.metrics[0]["picf_geometry_overshooting_matched_predictions"] == 4.0
    assert first.metrics[0]["picf_lifecycle_predictions"] == 0.0
    for name in (
        "picf_loss_action",
        "picf_loss_set",
        "picf_loss_set_localization_confidence",
        "picf_loss_dynamics",
        "picf_loss_dynamics_geometry_overshooting",
        "picf_loss_binding",
        "picf_loss_total",
    ):
        assert name in first.metrics[0]
        assert math.isfinite(first.metrics[0][name])
    for name in (
        "picf_query_existence_mean",
        "picf_query_localization_confidence_mean",
        "picf_query_measurement_probability_mean",
        "picf_query_mask_quality_mean",
        "picf_query_mask_coherence_score_mean",
        "picf_conditional_detection_probability_mean",
    ):
        assert 0.0 <= first.metrics[0][name] <= 1.0
    assert first.metrics[0]["picf_detection_probability_rows"] == 0.0
    assert first.metrics[0]["picf_dense_gate_abs_mean"] == pytest.approx(0.2)
    assert first.metrics[0]["picf_object_gate_abs_mean"] == pytest.approx(0.2)
    assert (
        first.metrics[0]["picf_posterior_stored_rows"]
        >= first.metrics[0]["picf_posterior_map_rows"]
    )
    assert first.metrics[0]["picf_posterior_tentative_rows"] == pytest.approx(
        first.metrics[0]["picf_posterior_stored_rows"] - first.metrics[0]["picf_posterior_map_rows"]
    )
    assert (
        first.metrics[0]["picf_posterior_support_births"]
        >= first.metrics[0]["picf_posterior_map_births"]
    )
    assert first.metrics[0]["picf_posterior_tentative_ownership_leak_max"] == 0.0
    assert math.isfinite(first.metrics[0]["picf_address_relation_logit_scale"])
    assert math.isfinite(first.metrics[0]["picf_address_relation_logit_bias"])
    assert math.isfinite(second.metrics[0]["picf_dense_gate_abs_max"])
    assert math.isfinite(second.metrics[0]["picf_object_gate_abs_max"])
    assert second.metrics[0]["picf_lifecycle_predictions"] > 0.0
    assert second.metrics[0]["picf_detection_probability_rows"] > 0.0
    assert streams["accumulation-00000"].next_transition_indices == (2, 2)
    assert streams["accumulation-00000"].belief.valid.any()
    assert any(
        key is not None
        for sample in streams["accumulation-00000"].loss_track_keys_by_row
        for key in sample
    )
    accelerator.end_training()


def test_calvin_stateful_production_module_requires_post_forward_lifecycle_builder(
    tmp_path: Path,
) -> None:
    dataset = _stateful_calvin_dataset(tmp_path / "calvin")
    policy = _policy(gradient_checkpointing=False)
    core = _picf_core(action_dim=7)
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=22,
    )
    install_adapter(policy, adapter)
    sequence_bridge = MolmoAct2PICFTrainingBridge(
        policy,
        core,
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=0,
            gradient_transitions=1,
            picf_core_lr=2e-4,
        ),
    )
    joint_bridge = MolmoAct2PICFJointTrainingBridge(
        sequence_bridge,
        PICFObjective(
            PICFObjectiveConfig(
                action_weight=1.0,
                set_weight=0.0,
                dynamics_weight=1.0,
                binding_weight=0.0,
            ),
            dynamics_criterion=ObjectDynamicsCriterion(
                ObjectDynamicsLossConfig(
                    content_cosine_weight=1.0,
                    geometry_nll_weight=1.0,
                    survival_weight=1.0,
                )
            ),
        ),
    )

    with pytest.raises(TypeError, match="separate target-free host observation builder"):
        CalvinStatefulMolmoAct2TrainingModule(
            dataset,
            joint_bridge,
            build_native_banks=lambda _requests: (),
            build_host_batch=lambda _samples: {},
            build_loss_targets=lambda _requests, _layout: CalvinStatefulLossTargets(),
        )

    with pytest.raises(ValueError, match="active structural objectives"):
        CalvinStatefulMolmoAct2TrainingModule(
            dataset,
            joint_bridge,
            build_native_banks=lambda _requests: (),
            build_host_batch=lambda _samples: {},
            build_host_observation_inputs=lambda _evidence, _views: {},
        )


def test_calvin_sequence_assembly_separates_sensor_evidence_and_action_targets() -> None:
    config = MolmoAct2PICFTrainingConfig(
        detached_context_frames=2,
        gradient_transitions=2,
        picf_core_lr=2e-4,
    )
    windows = (
        _calvin_window(10),
        _calvin_window(30, segment_index=1, instruction="turn on the led"),
    )
    evidence_calls = []
    observation_calls = []
    host_calls = []

    def build_native_banks(causal_prefixes):
        evidence_calls.append(tuple(len(prefix) for prefix in causal_prefixes))
        assert all(not hasattr(prefix[-1], "action") for prefix in causal_prefixes)
        assert all(not hasattr(prefix[-1], "task") for prefix in causal_prefixes)
        assert all(not hasattr(prefix[-1], "global_index") for prefix in causal_prefixes)
        assert all(
            not hasattr(prefix[-1].sensor_observations[0], "source_path")
            for prefix in causal_prefixes
        )
        tokens = torch.tensor([[[float(prefix[-1].timestamp_s)] * 6] for prefix in causal_prefixes])
        return (NativeTokenBank("vision", tokens, torch.ones(2, 1, dtype=torch.bool)),)

    def build_host_batch(current_records):
        host_calls.append(tuple(record.global_index for record in current_records))
        return {"action": torch.as_tensor(np.stack([record.action for record in current_records]))}

    def build_host_observation_inputs(causal_prefixes, host_observations):
        assert all(not hasattr(observation, "action") for observation in host_observations)
        assert all(not hasattr(observation, "global_index") for observation in host_observations)
        assert all(not hasattr(observation, "source_path") for observation in host_observations)
        observation_calls.append(
            (
                tuple(len(prefix) for prefix in causal_prefixes),
                tuple(observation.task for observation in host_observations),
                tuple(observation.timestamp_s for observation in host_observations),
            )
        )
        return {
            "input_ids": torch.tensor(
                [[len(observation.task), 7] for observation in host_observations],
                dtype=torch.long,
            )
        }

    transitions = assemble_calvin_molmoact2_transitions(
        windows,
        config,
        build_native_banks=build_native_banks,
        build_host_batch=build_host_batch,
        build_host_observation_inputs=build_host_observation_inputs,
    )

    assert evidence_calls == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert [call[0] for call in observation_calls] == evidence_calls
    assert [call[2] for call in observation_calls] == [
        (0.0, 0.0),
        (1 / 30, 1 / 30),
        (2 / 30, 2 / 30),
        (3 / 30, 3 / 30),
    ]
    assert all(call[1] == ("move the block", "turn on the led") for call in observation_calls)
    assert host_calls == [(12, 32), (13, 33)]
    assert transitions[0].host_batch is None
    assert transitions[1].host_batch is None
    assert transitions[0].host_observation_inputs is not None
    assert transitions[0].host_observation_inputs["input_ids"].shape == (2, 2)
    torch.testing.assert_close(transitions[0].previous_executed_action, torch.zeros(2, 7))
    torch.testing.assert_close(
        transitions[1].previous_executed_action[0],
        torch.as_tensor(windows[0].records[0].action.copy()),
    )
    assert transitions[2].host_batch is not None
    assert not torch.equal(
        transitions[2].previous_executed_action[0],
        transitions[2].host_batch["action"][0],
    )


def test_calvin_stateful_assembly_uses_carried_action_without_history_replay() -> None:
    windows = (
        _calvin_window(10),
        _calvin_window(30, segment_index=1, instruction="turn on the led"),
    )
    samples = (
        _calvin_stateful_sample(windows[0], 2),
        _calvin_stateful_sample(windows[1], 3),
    )
    bank = NativeTokenBank(
        "vision",
        torch.randn(2, 3, 6),
        torch.ones(2, 3, dtype=torch.bool),
    )
    observation_calls = []
    host_calls = []

    def build_host_observation_inputs(current_evidence, host_observations):
        observation_calls.append(tuple(len(item) for item in current_evidence))
        assert all(not hasattr(item[0], "action") for item in current_evidence)
        assert all(not hasattr(item[0], "global_index") for item in current_evidence)
        return {
            "input_ids": torch.tensor(
                [[len(observation.task), 7] for observation in host_observations],
                dtype=torch.long,
            )
        }

    def build_host_batch(stateful_samples):
        host_calls.append(tuple(sample.record.global_index for sample in stateful_samples))
        return {
            "action": torch.as_tensor(
                np.stack([sample.host_sample.action[0] for sample in stateful_samples])
            )
        }

    transition = assemble_calvin_stateful_molmoact2_transition(
        samples,
        native_banks=(bank,),
        build_host_batch=build_host_batch,
        build_host_observation_inputs=build_host_observation_inputs,
    )

    assert observation_calls == [(1, 1)]
    assert host_calls == [(12, 33)]
    assert transition.native_banks == (bank,)
    assert transition.host_batch is not None
    assert transition.host_observation_inputs is not None
    torch.testing.assert_close(
        transition.previous_executed_action[0],
        torch.as_tensor(windows[0].records[1].action.copy()),
    )
    torch.testing.assert_close(
        transition.previous_executed_action[1],
        torch.as_tensor(windows[1].records[2].action.copy()),
    )
    assert not torch.equal(
        transition.previous_executed_action,
        transition.host_batch["action"],
    )


def test_calvin_stateful_assembly_rejects_future_external_evidence() -> None:
    windows = (
        _calvin_window(10),
        _calvin_window(30, segment_index=1, instruction="turn on the led"),
    )
    samples = (
        _calvin_stateful_sample(windows[0], 2),
        _calvin_stateful_sample(windows[1], 3),
    )
    future = NativeTokenBank(
        "vision",
        torch.randn(2, 1, 6),
        torch.ones(2, 1, dtype=torch.bool),
        timestamps=torch.full((2, 1), 0.5, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="causal cutoff"):
        assemble_calvin_stateful_molmoact2_transition(
            samples,
            native_banks=(future,),
            build_host_batch=lambda stateful_samples: {
                "action": torch.as_tensor(
                    np.stack([sample.host_sample.action[0] for sample in stateful_samples])
                )
            },
        )


def test_calvin_stateful_assembly_allows_same_forward_only_evidence() -> None:
    windows = (
        _calvin_window(10),
        _calvin_window(30, segment_index=1, instruction="turn on the led"),
    )
    samples = (
        _calvin_stateful_sample(windows[0], 2),
        _calvin_stateful_sample(windows[1], 3),
    )

    transition = assemble_calvin_stateful_molmoact2_transition(
        samples,
        native_banks=(),
        build_host_batch=lambda stateful_samples: {
            "action": torch.as_tensor(
                np.stack([sample.host_sample.action[0] for sample in stateful_samples])
            )
        },
        build_host_observation_inputs=lambda _evidence, _views: {
            "input_ids": torch.ones(2, 2, dtype=torch.long)
        },
        tensor_device="cpu",
        tensor_dtype=torch.float32,
    )

    assert transition.native_banks == ()
    assert transition.previous_executed_action.shape == (2, 7)
    assert transition.previous_executed_action.dtype == torch.float32


def test_same_forward_only_assembly_requires_an_explicit_posterior_tensor_contract() -> None:
    window = _calvin_window(10)
    sample = _calvin_stateful_sample(window, 2)

    with pytest.raises(ValueError, match="explicit floating posterior"):
        assemble_calvin_stateful_molmoact2_transition(
            (sample,),
            native_banks=(),
            build_host_batch=lambda stateful_samples: {
                "action": torch.as_tensor(
                    np.stack([item.host_sample.action[0] for item in stateful_samples])
                )
            },
            build_host_observation_inputs=lambda _evidence, _views: {
                "input_ids": torch.ones(1, 2, dtype=torch.long)
            },
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_training_config_rejects_nonfinite_core_learning_rate(invalid: float) -> None:
    with pytest.raises(ValueError, match="picf_core_lr"):
        MolmoAct2PICFTrainingConfig(
            detached_context_frames=1,
            gradient_transitions=1,
            picf_core_lr=invalid,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("detached_context_frames", "detached_context_frames"),
        ("gradient_transitions", "gradient_transitions"),
        ("picf_core_lr", "picf_core_lr"),
    ],
)
def test_training_config_rejects_boolean_numeric_fields(field: str, message: str) -> None:
    values = {
        "detached_context_frames": 1,
        "gradient_transitions": 1,
        "picf_core_lr": 2e-4,
    }
    values[field] = True
    with pytest.raises(ValueError, match=message):
        MolmoAct2PICFTrainingConfig(**values)


def test_sequence_bridge_context_is_independent_of_target_only_batch_fields() -> None:
    torch.manual_seed(71)
    bridge = _training_bridge().eval()
    clean = _training_transitions()
    contaminated = tuple(
        MolmoAct2PICFTransition(
            native_banks=transition.native_banks,
            previous_executed_action=transition.previous_executed_action,
            delta_t_s=transition.delta_t_s,
            host_batch=(
                None
                if transition.host_batch is None
                else {
                    **transition.host_batch,
                    "object_mask_target": torch.rand(2, 5),
                    "simulator_instance_id": torch.tensor([100, 200]),
                    "task_owner_target": torch.tensor([1, 0]),
                }
            ),
        )
        for transition in clean
    )

    torch.manual_seed(73)
    clean_output = bridge(clean, _empty_belief())
    torch.manual_seed(73)
    contaminated_output = bridge(contaminated, _empty_belief())
    assert torch.equal(clean_output.loss, contaminated_output.loss)
    for clean_evidence, contaminated_evidence in zip(
        clean_output.evidences, contaminated_output.evidences, strict=True
    ):
        torch.testing.assert_close(
            clean_evidence.object_address,
            contaminated_evidence.object_address,
        )
        torch.testing.assert_close(
            clean_evidence.dense_ownership[0],
            contaminated_evidence.dense_ownership[0],
        )


def test_action_target_intervention_changes_loss_not_posterior() -> None:
    bridge = _training_bridge(require_explicit_flow_randomness=True).eval()
    transitions = _training_transitions()
    samples = _planned_samples()
    explicit = []
    intervened = []
    for index, transition in enumerate(transitions):
        if transition.host_batch is None:
            explicit.append(transition)
            intervened.append(transition)
            continue
        timesteps, noise = materialize_flow_randomness(
            bridge.policy,
            samples,
            transition.host_batch["action"],
            transition_index=index,
        )
        fixed = replace(
            transition,
            flow_timesteps=timesteps,
            flow_noise=noise,
        )
        changed_host_batch = dict(transition.host_batch)
        changed_host_batch["action"] = transition.host_batch["action"] + 0.75
        explicit.append(fixed)
        intervened.append(replace(fixed, host_batch=changed_host_batch))

    clean = bridge(tuple(explicit), _empty_belief())
    changed = bridge(tuple(intervened), _empty_belief())

    assert not torch.equal(clean.loss, changed.loss)
    for clean_evidence, changed_evidence in zip(
        clean.evidences,
        changed.evidences,
        strict=True,
    ):
        assert torch.equal(clean_evidence.object_address, changed_evidence.object_address)
        assert torch.equal(clean_evidence.object_value, changed_evidence.object_value)
        assert torch.equal(clean_evidence.object_valid, changed_evidence.object_valid)
        for clean_ownership, changed_ownership in zip(
            clean_evidence.dense_ownership,
            changed_evidence.dense_ownership,
            strict=True,
        ):
            assert torch.equal(clean_ownership, changed_ownership)
    for field in (
        "address_mean",
        "content_mean",
        "geometry_mean",
        "geometry_covariance_diag",
        "existence_logits",
        "visibility_given_existence_logits",
        "measurement_age_s",
        "valid",
        "age",
    ):
        assert torch.equal(getattr(clean.final_belief, field), getattr(changed.final_belief, field))


def test_sequence_bridge_optimizer_groups_are_complete_and_disjoint() -> None:
    bridge = _training_bridge()
    groups = bridge.get_optim_params()
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in bridge.parameters() if parameter.requires_grad]

    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    core_ids = {id(parameter) for parameter in bridge.core.parameters()}
    core_groups = [
        group for group in groups if core_ids & {id(parameter) for parameter in group["params"]}
    ]
    assert len(core_groups) == 1
    assert core_groups[0]["lr"] == 2e-4


def test_joint_training_bridge_consumes_one_explicit_objective() -> None:
    torch.manual_seed(79)
    sequence_bridge = _training_bridge()
    joint_bridge = MolmoAct2PICFJointTrainingBridge(
        sequence_bridge,
        PICFObjective(
            PICFObjectiveConfig(
                action_weight=1.0,
                set_weight=0.2,
                dynamics_weight=0.1,
                binding_weight=0.3,
            )
        ),
    )
    ownership = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    target = ObjectSetTarget(
        ownership=ownership,
        token_valid=torch.ones(5, dtype=torch.bool),
        geometry=torch.zeros(1, 3),
        geometry_contract=GEOMETRY,
        object_inventory_complete=True,
        temporal_identity_keys=("sequence-79/object-0",),
    )
    targets = tuple((target, target) for _ in range(2))

    output = joint_bridge(
        _training_transitions(),
        _empty_belief(),
        set_targets=targets,
    )

    expected = (
        output.objective.losses["loss_action"]
        + 0.2 * output.objective.losses["loss_set"]
        + 0.1 * output.objective.losses["loss_dynamics"]
        + 0.3 * output.objective.losses["loss_binding"]
    )
    torch.testing.assert_close(output.loss, expected)
    assert torch.equal(output.sequence.loss, output.objective.losses["loss_action"])
    assert output.objective.losses["loss_set_localization_confidence"] > 0.0
    assert output.objective.losses["loss_binding_temporal_address"] > 0.0
    output.loss.backward()
    assert sequence_bridge.core.discovery.ownership_query.weight.grad is not None
    assert sequence_bridge.core.posterior_filter.address_relation.logit_bias.grad is not None
    groups = joint_bridge.get_optim_params()
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in joint_bridge.parameters() if parameter.requires_grad]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}

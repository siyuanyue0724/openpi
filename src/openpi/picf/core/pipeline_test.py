import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from openpi.picf.anytouch.contracts import AnyTouchFeatureBundle
from openpi.picf.anytouch.contracts import AnyTouchSensorFeatures
from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import PicfPointCloudFrame
from openpi.picf.contracts import PicfTactilePacket
from openpi.picf.contracts import TactileSensorFrame
from openpi.picf.core.config import PicfCoreConfig
from openpi.picf.core.contracts import PicfObservationAnchorState
from openpi.picf.core.pipeline import PicfFullCore
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.visual_expert import _project_world_points
from openpi.picf.posterior.visual_expert import _scale_to_grid
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticFeatures
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.test_utils import build_mini_calvin_dataset
from openpi.picf.vjepa.config import VjepaVisualConfig


class _UnusedVisualEncoder:
    def encode_clip(self, _clip):
        raise AssertionError("visual_map_override should bypass encoder use in this test")


class _StubTactileEncoder:
    def encode_sensor_clips(self, *, clips_by_sensor, backgrounds_by_sensor, poses_by_sensor):
        del backgrounds_by_sensor
        sensors = {}
        pooled = []
        for index, sensor_name in enumerate(sorted(clips_by_sensor)):
            clip = np.asarray(clips_by_sensor[sensor_name], dtype=np.float32)
            value = float(clip.mean()) / 255.0 if clip.size > 0 else 0.0
            pseudo_contact = float(np.abs(clip[-1] - clip[0]).mean() / 255.0) if clip.shape[0] > 1 else 0.0
            tokens = torch.full((32, 64), value + index, dtype=torch.float32)
            pooled_feature = torch.full((128,), value + index, dtype=torch.float32)
            pose = torch.as_tensor(poses_by_sensor[sensor_name], dtype=torch.float32)
            sensors[sensor_name] = AnyTouchSensorFeatures(
                sensor_name=sensor_name,
                sensor_id=index,
                tokens=tokens,
                pooled_feature=pooled_feature,
                T_sens_to_wrist=pose,
                pseudo_contact_score=pseudo_contact,
            )
            pooled.append(pooled_feature)
        if not pooled:
            return None
        global_feature = torch.stack(pooled, dim=0).mean(dim=0)
        return AnyTouchFeatureBundle(
            global_feature=global_feature,
            sensors=sensors,
            checkpoint_loaded=False,
            hidden_dim=64,
            pooled_dim=128,
        )


class _FeatureMapFromClip:
    def __init__(self, clip: np.ndarray):
        value = float(np.asarray(clip)[-1].mean()) / 255.0
        self._map = np.full((4, 4, 8), value, dtype=np.float32)

    def current_map(self, *, use_last_two_mean: bool = False) -> np.ndarray:
        del use_last_two_mean
        return self._map


class _ClipAwareVisualEncoder:
    def __init__(self) -> None:
        self.clips: list[np.ndarray] = []

    def encode_clip(self, clip: np.ndarray):
        clip_np = np.asarray(clip).copy()
        self.clips.append(clip_np)
        return _FeatureMapFromClip(clip_np)


def _make_core(tmp_path: Path, **overrides) -> tuple[PicfFullCore, CalvinSequentialReplay]:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    config_kwargs = dict(
        persistent_anchors=8,
        observation_anchors=10,
        hidden_dim=64,
        posterior_hidden_dim=64,
        latent_dim=24,
        innovation_dim=64,
        control_dim=64,
        semantic_dim=32,
        semantic_cross_dim=64,
        future_hidden_dim=64,
        future_vote_heads=3,
        fusion_layers=2,
        posterior_layers=1,
        predictive_layers=1,
        control_layers=1,
        predictive_semantic_reads=1,
        control_semantic_reads=1,
        predictive_semantic_dropout_prob=0.0,
        attention_heads=4,
        query_rounds=2,
    )
    config_kwargs.update(overrides)
    config = PicfCoreConfig(**config_kwargs)
    core = PicfFullCore(
        builder,
        config=config,
        visual_config=VjepaVisualConfig(camera_json_path=calvin_root, arch_name_override="vit_tiny", img_size=64, num_frames=4, device="cpu", dtype="float32"),
        visual_encoder=_UnusedVisualEncoder(),
        tactile_encoder=_StubTactileEncoder(),
    )
    return core, replay


def _point_override(core: PicfFullCore, observation: PicfObservation) -> np.ndarray:
    if observation.G_t is None:
        observation.G_t = core.local_frame.make_transform(observation.robot_obs)
    if observation.point_set is None:
        observation.point_set = core.pointcloud_builder(
            {
                "rgb_static": observation.rgb_static,
                "depth_static": observation.depth_static,
                "focus_center_world": np.asarray(observation.G_t[:3, 3], dtype=np.float32),
                "focus_radius_m": core.config.crop_radius_m,
            }
        )
    xyz = np.asarray(observation.point_set.xyz_world, dtype=np.float32)
    center = np.asarray(observation.G_t[:3, 3], dtype=np.float32)
    keep = np.linalg.norm(xyz - center[None, :], axis=1) <= core.config.crop_radius_m
    n_points = int(keep.sum())
    return np.linspace(0.0, 1.0, max(n_points * 8, 1), dtype=np.float32).reshape(n_points, 8) if n_points > 0 else np.zeros((0, 8), dtype=np.float32)


def _make_tactile_packet(step_id: int, *, pose_shift: float = 0.0) -> PicfTactilePacket:
    left = np.full((32, 32, 3), 40 + step_id, dtype=np.uint8)
    right = np.full((32, 32, 3), 80 + step_id, dtype=np.uint8)
    bg = np.zeros((32, 32, 3), dtype=np.uint8)
    left_pose = np.eye(4, dtype=np.float32)
    left_pose[:3, 3] = np.array([0.01 + pose_shift, 0.0, 0.0], dtype=np.float32)
    right_pose = np.eye(4, dtype=np.float32)
    right_pose[:3, 3] = np.array([-0.01 + pose_shift, 0.0, 0.0], dtype=np.float32)
    return PicfTactilePacket(
        sensors=(
            TactileSensorFrame(rgb=left, sensor_name="digit", T_sens_to_wrist=left_pose, timestamp_s=float(step_id) / 30.0),
            TactileSensorFrame(rgb=right, sensor_name="gelsight_mini", T_sens_to_wrist=right_pose, timestamp_s=float(step_id) / 30.0),
        ),
        background_rgb_by_sensor={"digit": bg, "gelsight_mini": bg},
    )


def _visual_override(value: float) -> np.ndarray:
    return np.full((4, 4, 8), value, dtype=np.float32)


def _semantic_features(value: float, *, num_tokens: int = 3, width: int = 32) -> PaliGemmaSemanticFeatures:
    tokens = torch.full((num_tokens, width), value, dtype=torch.float32)
    summary = torch.full((1, width), value, dtype=torch.float32)
    return PaliGemmaSemanticFeatures(tokens=tokens, summary=summary)


def test_full_core_emits_unified_field_observation_posterior_and_predictions(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    assert output.state.token_field.point_tokens.shape[1] == core.config.hidden_dim
    assert output.state.token_field.visual_tokens.shape[0] == 16
    assert output.state.token_field.context_tokens.shape[0] == 3
    assert output.state.observation_anchors.tokens.shape == (core.config.observation_anchors, core.config.hidden_dim)
    assert output.state.posterior.mu.shape == (core.config.persistent_anchors, core.config.latent_dim)
    assert output.state.posterior.Sigma.shape == (core.config.persistent_anchors, core.config.latent_dim, core.config.latent_dim)
    assert output.state.posterior.binding.shape == (core.config.persistent_anchors + 1, core.config.observation_anchors)
    assert float(output.state.posterior.support_mass.sum().item()) > 0.0
    assert output.state.predictive.action.shape == (7,)
    assert output.state.predictive.physical_global_pred.shape == (core.config.hidden_dim,)
    assert output.state.predictive.physical_prediction_cache.visual_latent is not None
    assert output.state.predictive.physical_prediction_cache.visual_real is not None
    assert output.state.predictive.physical_prediction_cache.tactile_real is not None
    assert output.state.predictive.physical_prediction_cache.point_real is not None
    assert output.state.predictive.prediction_cache.visual_latent is not None
    assert output.state.predictive.prediction_cache.visual_real is not None
    assert output.state.predictive.prediction_cache.tactile_real is not None
    assert output.state.predictive.prediction_cache.point_real is not None
    assert output.state.predictive.semantic_summary.shape == (1, core.config.hidden_dim)
    assert output.state.predictive.semantic_tokens.shape == (3, core.config.semantic_dim)
    assert output.state.token_field.fusion_attention_mean is not None


def test_full_core_preserves_2048_wide_semantic_tokens_and_backpropagates(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        semantic_dim=2048,
        semantic_cross_dim=256,
        predictive_semantic_reads=1,
        control_semantic_reads=1,
    )
    frames = list(replay)[:2]
    semantic0 = torch.randn((6, core.config.semantic_dim), dtype=torch.float32)
    semantic1 = torch.randn((6, core.config.semantic_dim), dtype=torch.float32)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override={"tokens": semantic0, "summary": semantic0.mean(dim=0, keepdim=True)},
        action_future=frames[0].action,
    )
    second = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override={"tokens": semantic1, "summary": semantic1.mean(dim=0, keepdim=True)},
        action_future=frames[1].action,
    )
    assert second.state.predictive.semantic_tokens.shape == (6, 2048)
    assert second.state.predictive.semantic_summary.shape == (1, core.config.hidden_dim)
    loss = (
        first.state.predictive.action.pow(2).mean()
        + second.state.predictive.physical_global_pred.pow(2).mean()
        + second.state.predictive.global_pred.pow(2).mean()
    )
    core.zero_grad(set_to_none=True)
    loss.backward()
    assert core.action_head.weight.grad is not None
    assert core.predictive_pool.score.weight.grad is not None


def test_language_is_late_and_does_not_change_current_posterior(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    common_kwargs = dict(
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    first = core.step(frame, semantic_override=_semantic_features(1.0), **common_kwargs)
    second = core.step(frame, semantic_override=_semantic_features(3.0, num_tokens=5), **common_kwargs)
    assert torch.allclose(first.state.posterior.mu, second.state.posterior.mu)
    assert torch.allclose(first.state.posterior.Sigma, second.state.posterior.Sigma)
    assert torch.allclose(first.state.posterior.binding, second.state.posterior.binding)
    assert first.state.predictive.semantic_tokens.shape[0] == 3
    assert second.state.predictive.semantic_tokens.shape[0] == 5


def test_missing_semantic_override_falls_back_to_zero_semantic_tokens(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(2.0),
    )
    second = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(1.0),
    )
    assert second.state.predictive.semantic_tokens.shape[0] == 0
    assert torch.allclose(
        second.state.predictive.semantic_summary,
        torch.zeros_like(second.state.predictive.semantic_summary),
    )


def test_previous_prediction_becomes_current_innovation_signal(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    second = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
    )
    assert torch.linalg.norm(first.state.predictive.innovation_token).item() == pytest.approx(0.0)
    assert torch.linalg.norm(second.state.predictive.innovation_token).item() > 0.0
    assert second.state.predictive.innovation_norm[0].item() > 0.0


def test_semantic_changes_do_not_pollute_physical_prediction_cache_or_next_innovation(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01)
    for layer in core.predictive_semantic_reads:
        layer.cross_gate.data.fill_(3.0)
    semantic_a = torch.linspace(-1.0, 1.0, steps=3 * core.config.semantic_dim, dtype=torch.float32).reshape(3, core.config.semantic_dim)
    semantic_b = torch.linspace(1.0, -1.0, steps=5 * core.config.semantic_dim, dtype=torch.float32).reshape(5, core.config.semantic_dim)
    first_a = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override={"tokens": semantic_a, "summary": semantic_a.mean(dim=0, keepdim=True)},
        action_future=frames[0].action,
    )
    first_b = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override={"tokens": semantic_b, "summary": semantic_b.mean(dim=0, keepdim=True)},
        action_future=frames[0].action,
    )
    torch.testing.assert_close(first_a.state.predictive.physical_global_pred, first_b.state.predictive.physical_global_pred)
    torch.testing.assert_close(
        first_a.state.predictive.physical_prediction_cache.visual_latent,
        first_b.state.predictive.physical_prediction_cache.visual_latent,
    )
    assert not torch.allclose(first_a.state.predictive.global_pred, first_b.state.predictive.global_pred)
    second_a = core.step(
        frames[1],
        previous=first_a.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features(2.0),
        action_future=frames[1].action,
    )
    second_b = core.step(
        frames[1],
        previous=first_b.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features(2.0),
        action_future=frames[1].action,
    )
    torch.testing.assert_close(second_a.state.predictive.innovation_token, second_b.state.predictive.innovation_token)
    torch.testing.assert_close(second_a.state.predictive.innovation_norm, second_b.state.predictive.innovation_norm)


def test_semantic_summary_is_bookkeeping_only_and_does_not_change_downstream_readout(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    for layer in core.predictive_semantic_reads:
        layer.cross_gate.data.fill_(3.0)
    for layer in core.control_semantic_reads:
        layer.cross_gate.data.fill_(3.0)
    semantic_tokens = torch.linspace(
        -1.0,
        1.0,
        steps=4 * core.config.semantic_dim,
        dtype=torch.float32,
    ).reshape(4, core.config.semantic_dim)
    summary_a = torch.full((1, core.config.semantic_dim), -2.0, dtype=torch.float32)
    summary_b = torch.full((1, core.config.semantic_dim), 3.0, dtype=torch.float32)
    common_kwargs = dict(
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        action_future=frame.action,
    )
    first = core.step(
        frame,
        semantic_override={"tokens": semantic_tokens, "summary": summary_a},
        **common_kwargs,
    )
    second = core.step(
        frame,
        semantic_override={"tokens": semantic_tokens, "summary": summary_b},
        **common_kwargs,
    )
    assert not torch.allclose(first.state.predictive.semantic_summary, second.state.predictive.semantic_summary)
    torch.testing.assert_close(first.state.posterior.mu, second.state.posterior.mu)
    torch.testing.assert_close(first.state.predictive.action, second.state.predictive.action)
    torch.testing.assert_close(first.state.predictive.physical_global_pred, second.state.predictive.physical_global_pred)
    torch.testing.assert_close(first.state.predictive.global_pred, second.state.predictive.global_pred)


def test_previous_semantic_conditioned_predictive_state_does_not_feed_next_prior_or_innovation(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
        action_future=frames[0].action,
    )
    mutated_future_cache = dataclasses.replace(
        first.state.predictive.prediction_cache,
        visual_latent=torch.full_like(first.state.predictive.prediction_cache.visual_latent, -4.0),
        visual_real=torch.full_like(first.state.predictive.prediction_cache.visual_real, 2.5),
        tactile_real=torch.full_like(first.state.predictive.prediction_cache.tactile_real, 1.5),
        point_real=torch.full_like(first.state.predictive.prediction_cache.point_real, -1.0),
        availability=torch.zeros_like(first.state.predictive.prediction_cache.availability),
    )
    mutated_predictive = dataclasses.replace(
        first.state.predictive,
        semantic_tokens=torch.full_like(first.state.predictive.semantic_tokens, -3.0),
        semantic_summary=torch.full_like(first.state.predictive.semantic_summary, 7.0),
        global_pred=torch.full_like(first.state.predictive.global_pred, 0.25),
        prediction_cache=mutated_future_cache,
    )
    mutated_previous = dataclasses.replace(first.state, predictive=mutated_predictive)
    second_base = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features(2.0),
        action_future=frames[1].action,
    )
    second_mutated = core.step(
        frames[1],
        previous=mutated_previous,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features(2.0),
        action_future=frames[1].action,
    )
    torch.testing.assert_close(second_base.state.posterior.mu, second_mutated.state.posterior.mu)
    torch.testing.assert_close(second_base.state.posterior.Sigma, second_mutated.state.posterior.Sigma)
    torch.testing.assert_close(second_base.state.token_field.context_tokens, second_mutated.state.token_field.context_tokens)
    torch.testing.assert_close(second_base.state.predictive.innovation_token, second_mutated.state.predictive.innovation_token)
    torch.testing.assert_close(second_base.state.predictive.innovation_norm, second_mutated.state.predictive.innovation_norm)
    torch.testing.assert_close(second_base.state.predictive.physical_global_pred, second_mutated.state.predictive.physical_global_pred)


def test_previous_physical_prediction_cache_is_the_only_predictive_cache_allowed_to_change_next_innovation(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
        action_future=frames[0].action,
    )
    disabled_physical_cache = dataclasses.replace(
        first.state.predictive.physical_prediction_cache,
        visual_latent=torch.zeros_like(first.state.predictive.physical_prediction_cache.visual_latent),
        visual_real=torch.zeros_like(first.state.predictive.physical_prediction_cache.visual_real),
        tactile_real=torch.zeros_like(first.state.predictive.physical_prediction_cache.tactile_real),
        point_real=torch.zeros_like(first.state.predictive.physical_prediction_cache.point_real),
        availability=torch.zeros_like(first.state.predictive.physical_prediction_cache.availability),
    )
    changed_previous = dataclasses.replace(
        first.state,
        predictive=dataclasses.replace(
            first.state.predictive,
            physical_prediction_cache=disabled_physical_cache,
        ),
    )
    second_base = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features(2.0),
        action_future=frames[1].action,
    )
    second_changed = core.step(
        frames[1],
        previous=changed_previous,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features(2.0),
        action_future=frames[1].action,
    )
    torch.testing.assert_close(second_base.state.posterior.mu, second_changed.state.posterior.mu)
    assert not torch.allclose(second_base.state.predictive.innovation_token, second_changed.state.predictive.innovation_token)


def test_extract_targets_does_not_mutate_visual_history_when_using_real_visual_path(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    encoder = _ClipAwareVisualEncoder()
    core = PicfFullCore(
        builder,
        config=PicfCoreConfig(
            persistent_anchors=8,
            observation_anchors=10,
            hidden_dim=64,
            posterior_hidden_dim=64,
            latent_dim=24,
            innovation_dim=64,
            control_dim=64,
            semantic_dim=32,
            semantic_cross_dim=64,
            future_hidden_dim=64,
            future_vote_heads=3,
            fusion_layers=2,
            posterior_layers=1,
            predictive_layers=1,
            control_layers=1,
            predictive_semantic_reads=1,
            control_semantic_reads=1,
            predictive_semantic_dropout_prob=0.0,
            attention_heads=4,
            query_rounds=2,
            device="cpu",
        ),
        visual_config=VjepaVisualConfig(camera_json_path=calvin_root, arch_name_override="vit_tiny", img_size=64, num_frames=4, device="cpu", dtype="float32"),
        visual_encoder=encoder,
        tactile_encoder=_StubTactileEncoder(),
    )
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        semantic_override=_semantic_features(1.0),
    )
    assert first.state.token_field.visual_tokens.shape[0] > 0
    history_before = core.clip_buffer.get_clip().copy()
    _ = core.extract_targets(frames[1])
    history_after = core.clip_buffer.get_clip().copy()
    np.testing.assert_allclose(history_after, history_before)


def test_prior_and_context_use_previous_executed_action_not_previous_policy_output(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
    )
    mutated_policy = dataclasses.replace(
        first.state.predictive,
        action=torch.full_like(first.state.predictive.action, 0.75),
    )
    same_executed = dataclasses.replace(first.state, predictive=mutated_policy)
    second_base = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(1.0),
    )
    second_same_executed = core.step(
        frames[1],
        previous=same_executed,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(1.0),
    )
    torch.testing.assert_close(second_base.state.posterior.mu, second_same_executed.state.posterior.mu)
    torch.testing.assert_close(second_base.state.token_field.context_tokens, second_same_executed.state.token_field.context_tokens)

    mutated_executed = dataclasses.replace(
        first.state.predictive,
        executed_action=torch.full_like(first.state.predictive.executed_action, -0.75),
    )
    changed_executed = dataclasses.replace(first.state, predictive=mutated_executed)
    second_changed_executed = core.step(
        frames[1],
        previous=changed_executed,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(1.0),
    )
    assert not torch.allclose(second_base.state.posterior.mu, second_changed_executed.state.posterior.mu)
    assert not torch.allclose(second_base.state.token_field.context_tokens, second_changed_executed.state.token_field.context_tokens)


def test_first_step_requires_valid_xyzrgb_pointcloud(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.point_set = PicfPointCloudFrame(
        grid_coord=np.zeros((0, 3), dtype=np.int32),
        xyz_world=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.float32),
        normal_world=np.zeros((0, 3), dtype=np.float32),
        valid_point_mask=np.zeros((0,), dtype=bool),
        frame_valid=False,
    )
    with pytest.raises(RuntimeError, match="valid xyzrgb point cloud"):
        core.step(frame, visual_map_override=_visual_override(1.0))


def test_missing_point_contract_after_first_step_raises_hold_reason_and_uses_zero_point_tokens(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
    )
    bad_point = PicfPointCloudFrame(
        grid_coord=np.zeros((0, 3), dtype=np.int32),
        xyz_world=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.float32),
        normal_world=np.zeros((0, 3), dtype=np.float32),
        valid_point_mask=np.zeros((0,), dtype=bool),
        frame_valid=False,
    )
    second = core.step(
        dataclasses.replace(frames[1], point_set=bad_point),
        previous=first.state,
        visual_map_override=_visual_override(1.0),
    )
    assert second.state.control.hold_reason == "point_contract_violation"
    assert second.state.token_field.point_tokens.shape[0] == 0
    assert not second.state.runtime_meta.point_contract_ok


def test_projective_geometry_matches_legacy_projection_helper(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    world_points = output.state.token_field.point_positions.detach().cpu().numpy()
    uv, _, valid = _project_world_points(
        world_points,
        camera_model=core.camera_model,
        image_height=int(frame.rgb_static.shape[0]),
        image_width=int(frame.rgb_static.shape[1]),
    )
    expected = _scale_to_grid(
        uv,
        source_hw=(int(frame.rgb_static.shape[0]), int(frame.rgb_static.shape[1])),
        grid_hw=(4, 4),
    )
    got = geom.point_proj_grid_index.detach().cpu().numpy()
    visible = geom.point_visibility.detach().cpu().numpy() > 0.5
    assert np.array_equal(visible, valid)
    np.testing.assert_allclose(got[valid], expected[valid], atol=1e-4, rtol=1e-4)


def test_invisible_points_use_null_projection_branch_without_nan(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.G_t = np.eye(4, dtype=np.float32)
    frame.point_set = PicfPointCloudFrame(
        grid_coord=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
        xyz_world=np.asarray([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]], dtype=np.float32),
        rgb=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normal_world=np.zeros((2, 3), dtype=np.float32),
        valid_point_mask=np.asarray([True, True]),
        frame_valid=True,
    )
    output = core.step(
        frame,
        point_features_override=np.ones((2, 8), dtype=np.float32),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    visibility = geom.point_visibility.detach().cpu().numpy()
    assert visibility.tolist() == pytest.approx([1.0, 0.0])
    assert torch.isfinite(output.state.token_field.point_tokens).all()
    compat = geom.projective_compatibility.detach().cpu().numpy()
    assert np.allclose(compat[1], 0.0)


def test_visual_override_without_camera_model_uses_stable_null_projective_branch(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    core = PicfFullCore(
        builder,
        config=PicfCoreConfig(
            persistent_anchors=8,
            observation_anchors=10,
            hidden_dim=64,
            posterior_hidden_dim=64,
            latent_dim=24,
            innovation_dim=64,
            control_dim=64,
            semantic_dim=32,
            semantic_cross_dim=64,
            future_hidden_dim=64,
            future_vote_heads=3,
            fusion_layers=2,
            posterior_layers=1,
            predictive_layers=1,
            control_layers=1,
            predictive_semantic_reads=1,
            control_semantic_reads=1,
            predictive_semantic_dropout_prob=0.0,
            attention_heads=4,
            query_rounds=2,
        ),
        visual_config=None,
        visual_encoder=_UnusedVisualEncoder(),
        tactile_encoder=_StubTactileEncoder(),
    )
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    assert torch.isfinite(output.state.token_field.point_tokens).all()
    assert geom.projective_candidate_mask.shape == geom.projective_compatibility.shape
    assert not bool(geom.projective_candidate_mask.any())
    assert torch.allclose(geom.point_visibility, torch.zeros_like(geom.point_visibility))
    assert torch.allclose(geom.projective_attention_bias, torch.zeros_like(geom.projective_attention_bias))


def test_posterior_geometry_update_includes_observation_center_scatter(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path, persistent_anchors=1, observation_anchors=2)
    frame = next(iter(replay))
    hidden_dim = core.config.hidden_dim
    device = core.device
    dtype = core.dtype
    obs = PicfObservationAnchorState(
        seed_indices=torch.tensor([0, 1], dtype=torch.long, device=device),
        tokens=torch.zeros((2, hidden_dim), dtype=dtype, device=device),
        point_weights=torch.zeros((2, 0), dtype=dtype, device=device),
        routing_mass_point=torch.zeros((2, 0), dtype=dtype, device=device),
        routing_mass_visual=torch.zeros((2, 0), dtype=dtype, device=device),
        routing_support_point=torch.zeros((0,), dtype=dtype, device=device),
        routing_support_visual=torch.zeros((0,), dtype=dtype, device=device),
        routing_gate_point=torch.zeros((0,), dtype=dtype, device=device),
        routing_gate_visual=torch.zeros((0,), dtype=dtype, device=device),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=dtype, device=device),
        S=torch.stack(
            [
                torch.diag(torch.tensor([0.01, 0.02, 0.03], dtype=dtype, device=device)),
                torch.diag(torch.tensor([0.01, 0.02, 0.03], dtype=dtype, device=device)),
            ],
            dim=0,
        ),
        a=torch.full((2, 3), 0.05, dtype=dtype, device=device),
    )
    posterior = core._posterior_update(None, frame, obs)
    binding = posterior.binding[:-1, :]
    weights = binding[0]
    denom = torch.clamp(weights.sum(), min=core.config.epsilon_a)
    expected_x = (weights[:, None] * obs.x).sum(dim=0) / denom
    centered = obs.x - expected_x[None, :]
    expected_S = (
        weights[:, None, None] * (obs.S + centered[:, :, None] * centered[:, None, :])
    ).sum(dim=0) / denom
    torch.testing.assert_close(posterior.x[0], expected_x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(posterior.S[0], expected_S, atol=1e-5, rtol=1e-5)


def test_invalid_depth_sample_does_not_zero_projective_compatibility(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.G_t = np.eye(4, dtype=np.float32)
    frame.depth_static = np.full_like(frame.depth_static, np.nan, dtype=np.float32)
    frame.point_set = PicfPointCloudFrame(
        grid_coord=np.asarray([[0, 0, 0]], dtype=np.int32),
        xyz_world=np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32),
        rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        normal_world=np.zeros((1, 3), dtype=np.float32),
        valid_point_mask=np.asarray([True]),
        frame_valid=True,
    )
    output = core.step(
        frame,
        point_features_override=np.ones((1, 8), dtype=np.float32),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    assert bool(geom.point_visibility[0].item())
    assert not bool(geom.point_depth_valid[0].item())
    assert float(geom.projective_compatibility[0].max().item()) > 0.0


def test_projective_compatibility_uses_patch_grid_units(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    visible = torch.nonzero(geom.point_visibility > 0.5, as_tuple=False).flatten()
    assert visible.numel() > 0
    point_index = int(visible[0].item())
    delta = geom.point_proj_grid_index[point_index][None, :] - geom.visual_grid_index
    expected = torch.exp(-torch.sum(delta**2, dim=-1) / (2.0 * (core.config.sigma_proj_patches**2)))
    if bool(geom.point_depth_valid[point_index].item()):
        depth_residual = geom.point_depth[point_index] - geom.point_depth_sample[point_index]
        expected = expected * torch.exp(-(depth_residual**2) / (2.0 * (core.config.tau_proj_depth_m**2)))
    torch.testing.assert_close(
        geom.projective_compatibility[point_index],
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_projective_compatibility_stays_in_probability_range(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    assert torch.isfinite(geom.projective_compatibility).all()
    assert bool((geom.projective_compatibility >= 0.0).all().item())
    assert bool((geom.projective_compatibility <= 1.0).all().item())


def test_projective_attention_bias_is_sparse_on_candidate_edges(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    assert geom.projective_attention_bias.shape == geom.projective_candidate_mask.shape
    assert torch.isfinite(geom.projective_attention_bias).all()
    assert torch.allclose(
        geom.projective_attention_bias[~geom.projective_candidate_mask],
        torch.zeros_like(geom.projective_attention_bias[~geom.projective_candidate_mask]),
    )
    if bool(geom.projective_candidate_mask.any()):
        assert torch.count_nonzero(geom.projective_attention_bias[geom.projective_candidate_mask]).item() > 0


def test_projective_attention_bias_backward_is_finite(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    core.zero_grad(set_to_none=True)
    loss = geom.projective_attention_bias.square().sum()
    loss.backward()
    grads = [param.grad for param in core.projective_bias_head.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_projective_candidate_mask_respects_sparse_patch_neighborhood(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    radius = np.sqrt(-2.0 * (core.config.sigma_proj_patches**2) * np.log(core.config.tau_proj))
    delta = geom.point_proj_grid_index[:, None, :] - geom.visual_grid_index[None, :, :]
    distance = torch.sqrt(torch.sum(delta**2, dim=-1))
    outside = distance > (radius + 1e-5)
    assert not bool((geom.projective_candidate_mask & outside).any())


def test_sinkhorn_dustbin_stays_finite_and_backward_stable(tmp_path: Path) -> None:
    core, _ = _make_core(tmp_path)
    logits = torch.tensor(
        [
            [120.0, -120.0, 40.0],
            [-80.0, 90.0, -60.0],
        ],
        device=core.device,
        dtype=core.dtype,
        requires_grad=True,
    )
    transport = core._sinkhorn_dustbin(logits)
    assert torch.isfinite(transport).all()
    weights = torch.arange(1, transport.numel() + 1, device=core.device, dtype=core.dtype).reshape_as(transport)
    loss = torch.sum(transport * weights)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

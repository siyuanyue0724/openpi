import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.picf.contracts import PicfPointCloudFrame
from openpi.picf.anytouch.contracts import AnyTouchFeatureBundle
from openpi.picf.anytouch.contracts import AnyTouchSensorFeatures
from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import PicfTactilePacket
from openpi.picf.contracts import TactileSensorFrame
from openpi.picf.core.config import PicfCoreConfig
from openpi.picf.core.contracts import PicfObservationAnchorState
from openpi.picf.core.contracts import PicfProjectiveGeometryState
from openpi.picf.core.contracts import PicfTokenFieldState
from openpi.picf.core.pipeline import PicfFullCore
from openpi.picf.core.training import PicfAlignmentLossConfig
from openpi.picf.core.training import compute_alignment_loss
from openpi.picf.core.training import compute_transition_loss
from openpi.picf.core.training import PicfTransitionLossConfig
from openpi.picf.core.training import PicfFutureTargets
from openpi.picf.core.training import extract_future_targets
from openpi.picf.core.training import future_targets_from_current_targets
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticFeatures
from openpi.picf.paligemma.wrapper import PaliGemmaViewTransform
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.test_utils import build_mini_calvin_dataset
from openpi.picf.vjepa.config import VjepaVisualConfig


class _UnusedVisualEncoder:
    def encode_clip(self, _clip):
        raise AssertionError("visual_map_override should bypass encoder use in this test")


class _StubTactileEncoder:
    def encode_sensor_clips(self, *, clips_by_sensor, backgrounds_by_sensor, poses_by_sensor):
        sensors = {}
        pooled = []
        for index, sensor_name in enumerate(sorted(clips_by_sensor)):
            clip = np.asarray(clips_by_sensor[sensor_name], dtype=np.float32)
            value = float(clip.mean()) / 255.0 if clip.size > 0 else 0.0
            background = backgrounds_by_sensor.get(sensor_name)
            if background is not None:
                background = np.asarray(background, dtype=np.float32)
                pseudo_contact = float(np.abs(clip[-1] - background).mean() / 255.0)
            else:
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
                rgb_residual_score=pseudo_contact,
                contact_score=pseudo_contact,
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


def _make_core(
    tmp_path: Path,
    *,
    vl_anchor_router_enabled: bool = False,
    mapg_enabled: bool = False,
) -> tuple[PicfFullCore, CalvinSequentialReplay]:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    config = PicfCoreConfig(
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
        vl_anchor_router_enabled=vl_anchor_router_enabled,
        vl_anchor_modes=3,
        mapg_enabled=mapg_enabled,
        mapg_anchor_count=6,
        mapg_message_rounds=2,
        device="cpu",
    )
    core = PicfFullCore(
        builder,
        config=config,
        visual_config=VjepaVisualConfig(camera_json_path=calvin_root, arch_name_override="vit_tiny", img_size=64, num_frames=4, device="cpu", dtype="float32"),
        visual_encoder=_UnusedVisualEncoder(),
        tactile_encoder=_StubTactileEncoder(),
    )
    return core, replay


def _semantic_features_for_frame(core: PicfFullCore, observation: PicfObservation) -> PaliGemmaSemanticFeatures:
    height, width = (int(v) for v in observation.rgb_static.shape[:2])
    image_tokens = torch.linspace(0.0, 1.0, 4 * core.config.semantic_dim, dtype=torch.float32).reshape(4, core.config.semantic_dim)
    text_tokens = torch.ones((2, core.config.semantic_dim), dtype=torch.float32)
    return PaliGemmaSemanticFeatures(
        tokens=torch.cat([image_tokens, text_tokens], dim=0),
        summary=torch.cat([text_tokens.mean(dim=0), image_tokens.mean(dim=0)], dim=0),
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        image_token_ranges=((0, 4),),
        image_grid_shapes=((2, 2),),
        image_view_names=("static",),
        image_view_transforms=(
            PaliGemmaViewTransform(
                original_hw=(height, width),
                target_hw=(224, 224),
                resized_hw=(224, 224),
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                scale_y=224.0 / max(height, 1),
                scale_x=224.0 / max(width, 1),
            ),
        ),
    )


def _point_override(core: PicfFullCore, observation: PicfObservation) -> np.ndarray:
    if observation.G_t is None:
        observation.G_t = core.local_frame.make_transform(observation.robot_obs)
    focus_centers = [np.asarray(observation.G_t[:3, 3], dtype=np.float32)]
    if observation.tactile is not None:
        for sensor in observation.tactile.sensors:
            focus_centers.append((np.asarray(observation.G_t, dtype=np.float32) @ np.asarray(sensor.T_sens_to_wrist, dtype=np.float32))[:3, 3])
    if observation.point_set is None:
        observation.point_set = core.pointcloud_builder(
            {
                "rgb_static": observation.rgb_static,
                "depth_static": observation.depth_static,
                "rgb_gripper": observation.rgb_gripper,
                "depth_gripper": observation.depth_gripper,
                "focus_centers_world": np.stack(focus_centers, axis=0),
                "focus_radius_m": core.config.crop_radius_m,
            }
        )
    xyz = np.asarray(observation.point_set.xyz_world, dtype=np.float32)
    centers = np.stack(focus_centers, axis=0)
    keep = np.linalg.norm(xyz[:, None, :] - centers[None, :, :], axis=-1).min(axis=1) <= core.config.crop_radius_m
    n_points = int(keep.sum())
    return np.linspace(0.0, 1.0, max(n_points * 8, 1), dtype=np.float32).reshape(n_points, 8) if n_points > 0 else np.zeros((0, 8), dtype=np.float32)


def _make_tactile_packet(step_id: int, *, pose_shift: float = 0.0, contact_shift: int = 0) -> PicfTactilePacket:
    left_bg = np.full((32, 32, 3), 40 + step_id, dtype=np.uint8)
    right_bg = np.full((32, 32, 3), 80 + step_id, dtype=np.uint8)
    left = np.full((32, 32, 3), 40 + step_id + contact_shift, dtype=np.uint8)
    right = np.full((32, 32, 3), 80 + step_id + contact_shift, dtype=np.uint8)
    left_pose = np.eye(4, dtype=np.float32)
    left_pose[:3, 3] = np.array([0.01 + pose_shift, 0.0, 0.0], dtype=np.float32)
    right_pose = np.eye(4, dtype=np.float32)
    right_pose[:3, 3] = np.array([-0.01 + pose_shift, 0.0, 0.0], dtype=np.float32)
    return PicfTactilePacket(
        sensors=(
            TactileSensorFrame(rgb=left, sensor_name="digit", T_sens_to_wrist=left_pose, timestamp_s=float(step_id) / 30.0),
            TactileSensorFrame(rgb=right, sensor_name="gelsight_mini", T_sens_to_wrist=right_pose, timestamp_s=float(step_id) / 30.0),
        ),
        background_rgb_by_sensor={"digit": left_bg, "gelsight_mini": right_bg},
    )


def _visual_override(value: float) -> np.ndarray:
    return np.full((4, 4, 8), value, dtype=np.float32)


def _empty_point_set() -> PicfPointCloudFrame:
    return PicfPointCloudFrame(
        grid_coord=np.zeros((0, 3), dtype=np.int32),
        xyz_world=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.float32),
        normal_world=np.zeros((0, 3), dtype=np.float32),
        valid_point_mask=np.zeros((0,), dtype=bool),
        frame_valid=True,
    )


def test_extract_future_targets_is_stop_grad_teacher_path() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def extract_targets(self, observation, *, visual_map_override=None):
            del observation, visual_map_override
            value = (self.weight * 3.0).reshape(1)
            targets = {
                "visual_latent": value,
                "visual_real": value,
                "tactile_real": value,
                "point_real": value,
            }
            availability = torch.ones((4,), dtype=value.dtype)
            return targets, availability

    core = _DummyCore()
    future = extract_future_targets(
        core,
        PicfObservation(
            rgb_static=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_static=np.zeros((2, 2), dtype=np.float32),
            robot_obs=np.zeros((7,), dtype=np.float32),
            prompt="test",
            step_id=0,
            segment_id=0,
            timestamp_s=0.0,
            reset_scaffold=True,
        ),
    )
    assert future.visual_latent is not None
    assert future.visual_latent.requires_grad is False
    assert future.visual_real is not None
    assert future.visual_real.requires_grad is False
    assert future.tactile_real is not None
    assert future.tactile_real.requires_grad is False
    assert future.point_real is not None
    assert future.point_real.requires_grad is False
    assert future.availability.requires_grad is False


def test_future_targets_from_current_targets_detaches_teacher_values() -> None:
    weight = torch.nn.Parameter(torch.tensor(2.0))
    value = (weight * 3.0).reshape(1)
    future = future_targets_from_current_targets(
        {
            "visual_latent": value,
            "visual_real": value,
            "tactile_real": value,
            "point_real": value,
        },
        torch.ones((4,), dtype=value.dtype, requires_grad=True),
    )
    assert future.visual_latent is not None and future.visual_latent.requires_grad is False
    assert future.visual_real is not None and future.visual_real.requires_grad is False
    assert future.tactile_real is not None and future.tactile_real.requires_grad is False
    assert future.point_real is not None and future.point_real.requires_grad is False
    assert future.availability.requires_grad is False


def test_future_targets_from_current_targets_carries_detached_posterior_teacher() -> None:
    token_weight = torch.nn.Parameter(torch.ones((2, 3)))
    tokens = token_weight * 2.0
    posterior = SimpleNamespace(
        tokens=tokens,
        alpha=torch.tensor([0.9, 0.2], requires_grad=True),
        support_mass=torch.tensor([0.7, 0.3], requires_grad=True),
        contact_prob=torch.tensor([0.1, 0.8], requires_grad=True),
        binding=torch.tensor([[0.2, 0.8], [0.6, 0.4]], requires_grad=True),
    )
    future = future_targets_from_current_targets(
        {
            "visual_latent": None,
            "visual_real": None,
            "tactile_real": None,
            "point_real": None,
        },
        torch.ones((4,), requires_grad=True),
        posterior=posterior,
    )

    assert future.posterior_tokens is not None
    assert future.posterior_tokens.requires_grad is False
    assert torch.allclose(future.posterior_tokens, tokens.detach())
    assert future.posterior_support_summary is not None
    assert future.posterior_support_summary.requires_grad is False
    assert future.posterior_support_summary.shape == (2, 4)
    assert torch.allclose(future.posterior_support_summary[:, -1], torch.tensor([0.8, 0.6]))


def test_transition_loss_closes_one_step_future_supervision_and_backward(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=frames[0].action,
    )
    core.zero_grad(set_to_none=True)
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
    )
    assert losses.total.requires_grad
    assert torch.isfinite(losses.total)
    assert losses.total.item() > 0.0
    assert losses.availability.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert torch.isfinite(losses.semantic_future_aux)
    losses.total.backward()
    assert core.point_real_head.weight.grad is not None
    assert core.visual_latent_head.weight.grad is not None


def test_transition_loss_reports_effective_budgeted_terms_consistently(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=frames[0].action,
    )
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
    )
    torch.testing.assert_close(
        losses.action_active7,
        ((3.0 * losses.action_pos) + (3.0 * losses.action_rot) + losses.action_gripper) / 7.0,
    )
    torch.testing.assert_close(
        losses.total,
        losses.action + losses.physical_aux_capped + losses.semantic_group_capped + losses.alignment,
    )
    torch.testing.assert_close(
        losses.total_minus_action,
        losses.physical_aux_capped + losses.semantic_group_capped + losses.alignment,
    )
    assert torch.isfinite(losses.physical_aux_capped)
    assert torch.isfinite(losses.semantic_group_capped)
    assert torch.isfinite(losses.alignment_raw)


def test_transition_loss_computes_guarded_owm_objectives_when_weighted(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=frames[0].action,
    )
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
        config=PicfTransitionLossConfig(
            lambda_slot_jepa=0.1,
            lambda_support_pred=0.1,
            lambda_binding_consistency=0.1,
            lambda_innovation_calib=0.1,
        ),
    )

    assert torch.isfinite(losses.slot_jepa)
    assert torch.isfinite(losses.support_pred)
    assert torch.isfinite(losses.binding_consistency)
    assert torch.isfinite(losses.innovation_calib)
    assert losses.slot_jepa.item() >= 0.0
    assert losses.support_pred.item() >= 0.0
    assert losses.binding_consistency.item() >= 0.0
    assert losses.innovation_calib.item() >= 0.0


def test_slot_jepa_prefers_next_posterior_teacher_over_visual_fallback(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=frames[0].action,
    )
    slot_tokens = first.state.predictive.slot_prediction_tokens
    assert slot_tokens is not None
    future = PicfFutureTargets(
        visual_latent=torch.zeros_like(first.state.predictive.physical_prediction_cache.visual_latent),
        visual_real=None,
        tactile_real=None,
        point_real=None,
        availability=torch.ones((4,), dtype=slot_tokens.dtype),
        posterior_tokens=(slot_tokens.detach() + 1.0),
        posterior_support_summary=None,
    )
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        config=PicfTransitionLossConfig(lambda_slot_jepa=1.0),
        future_targets_override=future,
    )

    assert torch.allclose(losses.slot_jepa, torch.ones_like(losses.slot_jepa), atol=1e-5)


def test_transition_loss_can_enable_vl_router_supervision_terms(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path, vl_anchor_router_enabled=True)
    frames = list(replay)[:2]
    frames[0].G_t = core.local_frame.make_transform(frames[0].robot_obs)
    frames[1].G_t = core.local_frame.make_transform(frames[1].robot_obs)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_for_frame(core, frames[0]),
        action_future=frames[0].action,
    )
    assert first.state.vl_grounding is not None
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
        config=PicfTransitionLossConfig(
            lambda_vl_heatmap_effector=0.1,
            lambda_vl_heatmap_interaction=0.1,
            lambda_vl_point_consistency=0.05,
            lambda_vl_anchor_diversity=0.01,
        ),
    )
    assert torch.isfinite(losses.vl_router)
    assert torch.isfinite(losses.vl_heatmap_effector)
    assert torch.isfinite(losses.vl_heatmap_interaction)
    assert torch.isfinite(losses.vl_point_consistency)
    assert torch.isfinite(losses.vl_anchor_diversity)
    assert losses.vl_router.item() >= 0.0


def test_transition_loss_can_enable_mapg_graph_terms(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path, vl_anchor_router_enabled=True, mapg_enabled=True)
    frames = list(replay)[:2]
    frames[0].G_t = core.local_frame.make_transform(frames[0].robot_obs)
    frames[1].G_t = core.local_frame.make_transform(frames[1].robot_obs)
    frames[0].tactile = _make_tactile_packet(frames[0].step_id, contact_shift=10)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, contact_shift=14)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_for_frame(core, frames[0]),
        action_future=frames[0].action,
    )
    assert first.state.anchor_prior_graph is not None
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
        config=PicfTransitionLossConfig(
            lambda_mapg_siglip=0.01,
            lambda_mapg_vicreg=0.01,
            lambda_mapg_cycle=0.01,
            lambda_mapg_masked_modality=0.01,
            lambda_mapg_routing=0.01,
            lambda_mapg_support_diversity=0.01,
            lambda_mapg_geometry_diversity=0.01,
        ),
    )

    assert torch.isfinite(losses.mapg_graph)
    assert torch.isfinite(losses.mapg_siglip)
    assert torch.isfinite(losses.mapg_vicreg)
    assert torch.isfinite(losses.mapg_cycle)
    assert torch.isfinite(losses.mapg_masked_modality)
    assert torch.isfinite(losses.mapg_routing)
    assert torch.isfinite(losses.mapg_support_diversity)
    assert torch.isfinite(losses.mapg_geometry_diversity)
    assert losses.mapg_graph.item() >= 0.0


def test_transition_loss_keeps_point_head_in_graph_when_future_point_target_is_unavailable(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=frames[0].action,
    )
    pointless_future = dataclasses.replace(
        frames[1],
        point_set=_empty_point_set(),
        G_t=core.local_frame.make_transform(frames[1].robot_obs),
    )
    core.zero_grad(set_to_none=True)
    losses = compute_transition_loss(
        core,
        first,
        pointless_future,
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
    )
    losses.total.backward()
    assert core.point_real_head.weight.grad is not None
    assert torch.allclose(core.point_real_head.weight.grad, torch.zeros_like(core.point_real_head.weight.grad))


def test_innovation_keeps_point_error_encoder_in_graph_when_current_point_target_is_unavailable(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=frames[0].action,
    )
    pointless_current = dataclasses.replace(
        frames[1],
        point_set=_empty_point_set(),
        G_t=core.local_frame.make_transform(frames[1].robot_obs),
    )
    core.zero_grad(set_to_none=True)
    second = core.step(
        pointless_current,
        previous=first.state,
        point_features_override=np.zeros((0, 8), dtype=np.float32),
        visual_map_override=_visual_override(2.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
        action_future=pointless_current.action,
    )
    second.state.predictive.innovation_token.sum().backward()
    assert core.point_error_encoder.weight.grad is not None
    assert torch.allclose(core.point_error_encoder.weight.grad, torch.zeros_like(core.point_error_encoder.weight.grad))


def test_semantic_future_aux_keeps_predictive_semantic_trunk_in_graph(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((4, core.config.semantic_dim), dtype=np.float32),
        action_future=frames[0].action,
    )
    core.zero_grad(set_to_none=True)
    losses = compute_transition_loss(
        core,
        first,
        frames[1],
        action_target=frames[0].action,
        next_visual_map_override=_visual_override(2.0),
    )
    losses.total.backward()
    assert torch.isfinite(losses.semantic_future_aux)
    assert core.predictive_semantic_world.layers[0].attn.out_proj.weight.grad is not None


def test_alignment_loss_uses_projective_candidates_and_is_finite(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    alignment = compute_alignment_loss(output.state)
    assert torch.isfinite(alignment.total)
    assert torch.isfinite(alignment.anchor_pv)
    assert torch.isfinite(alignment.pv_weak)
    assert torch.isfinite(alignment.focus_pv)
    assert alignment.candidate_edges > 0
    assert 0.0 < alignment.candidate_density < 1.0


def test_alignment_loss_sanitizes_probability_contract_before_bce() -> None:
    dtype = torch.float32
    geometry = PicfProjectiveGeometryState(
        point_proj_grid_norm=torch.zeros((1, 2), dtype=dtype),
        point_proj_grid_index=torch.zeros((1, 2), dtype=dtype),
        point_visibility=torch.ones((1,), dtype=dtype),
        point_depth=torch.ones((1,), dtype=dtype),
        point_depth_sample=torch.ones((1,), dtype=dtype),
        point_depth_valid=torch.ones((1,), dtype=torch.bool),
        visual_grid_norm=torch.zeros((1, 2), dtype=dtype),
        visual_grid_index=torch.zeros((1, 2), dtype=dtype),
        visual_pixel_centers=torch.zeros((1, 2), dtype=dtype),
        visual_ray_world=torch.zeros((1, 3), dtype=dtype),
        camera_origin_world=torch.zeros((3,), dtype=dtype),
        projective_compatibility=torch.tensor([[float("nan")]], dtype=dtype),
        projective_candidate_mask=torch.ones((1, 1), dtype=torch.bool),
        projective_attention_bias=torch.zeros((1, 1), dtype=dtype),
    )
    token_field = PicfTokenFieldState(
        point_tokens=torch.zeros((1, 4), dtype=dtype),
        visual_tokens=torch.zeros((1, 4), dtype=dtype),
        tactile_tokens=torch.zeros((0, 4), dtype=dtype),
        context_tokens=torch.zeros((0, 4), dtype=dtype),
        fused_tokens=torch.zeros((2, 4), dtype=dtype),
        point_positions=torch.zeros((1, 3), dtype=dtype),
        modality_ids=torch.tensor([0, 1], dtype=torch.long),
        point_align_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        visual_align_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        tactile_align_embeddings=torch.zeros((0, 4), dtype=dtype),
        tactile_positions_world=torch.zeros((0, 3), dtype=dtype),
        tactile_contact_gate=torch.zeros((0,), dtype=dtype),
        fusion_attention_mean=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype),
        projective_geometry=geometry,
    )
    obs = PicfObservationAnchorState(
        seed_indices=torch.tensor([0], dtype=torch.long),
        tokens=torch.zeros((1, 4), dtype=dtype),
        point_weights=torch.ones((1, 1), dtype=dtype),
        routing_mass_point=torch.tensor([[float("nan")]], dtype=dtype),
        routing_mass_visual=torch.tensor([[2.0]], dtype=dtype),
        routing_support_point=torch.zeros((1,), dtype=dtype),
        routing_support_visual=torch.zeros((1,), dtype=dtype),
        routing_gate_point=torch.zeros((1,), dtype=dtype),
        routing_gate_visual=torch.zeros((1,), dtype=dtype),
        x=torch.zeros((1, 3), dtype=dtype),
        S=torch.eye(3, dtype=dtype)[None, :, :],
        a=torch.ones((1, 3), dtype=dtype),
    )
    state = SimpleNamespace(token_field=token_field, observation_anchors=obs)
    alignment = compute_alignment_loss(state)
    assert torch.isfinite(alignment.total)
    assert torch.isfinite(alignment.anchor_pv)


def test_alignment_loss_raises_on_projective_shape_contract_mismatch() -> None:
    dtype = torch.float32
    geometry = PicfProjectiveGeometryState(
        point_proj_grid_norm=torch.zeros((1, 2), dtype=dtype),
        point_proj_grid_index=torch.zeros((1, 2), dtype=dtype),
        point_visibility=torch.ones((1,), dtype=dtype),
        point_depth=torch.ones((1,), dtype=dtype),
        point_depth_sample=torch.ones((1,), dtype=dtype),
        point_depth_valid=torch.ones((1,), dtype=torch.bool),
        visual_grid_norm=torch.zeros((1, 2), dtype=dtype),
        visual_grid_index=torch.zeros((1, 2), dtype=dtype),
        visual_pixel_centers=torch.zeros((1, 2), dtype=dtype),
        visual_ray_world=torch.zeros((1, 3), dtype=dtype),
        camera_origin_world=torch.zeros((3,), dtype=dtype),
        projective_compatibility=torch.ones((1, 2), dtype=dtype),
        projective_candidate_mask=torch.ones((1, 2), dtype=torch.bool),
        projective_attention_bias=torch.zeros((1, 2), dtype=dtype),
    )
    token_field = PicfTokenFieldState(
        point_tokens=torch.zeros((1, 4), dtype=dtype),
        visual_tokens=torch.zeros((1, 4), dtype=dtype),
        tactile_tokens=torch.zeros((0, 4), dtype=dtype),
        context_tokens=torch.zeros((0, 4), dtype=dtype),
        fused_tokens=torch.zeros((2, 4), dtype=dtype),
        point_positions=torch.zeros((1, 3), dtype=dtype),
        modality_ids=torch.tensor([0, 1], dtype=torch.long),
        point_align_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        visual_align_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        tactile_align_embeddings=torch.zeros((0, 4), dtype=dtype),
        tactile_positions_world=torch.zeros((0, 3), dtype=dtype),
        tactile_contact_gate=torch.zeros((0,), dtype=dtype),
        fusion_attention_mean=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype),
        projective_geometry=geometry,
    )
    obs = PicfObservationAnchorState(
        seed_indices=torch.tensor([0], dtype=torch.long),
        tokens=torch.zeros((1, 4), dtype=dtype),
        point_weights=torch.ones((1, 1), dtype=dtype),
        routing_mass_point=torch.ones((1, 1), dtype=dtype),
        routing_mass_visual=torch.ones((1, 1), dtype=dtype),
        routing_support_point=torch.ones((1,), dtype=dtype),
        routing_support_visual=torch.ones((1,), dtype=dtype),
        routing_gate_point=torch.ones((1,), dtype=dtype),
        routing_gate_visual=torch.ones((1,), dtype=dtype),
        x=torch.zeros((1, 3), dtype=dtype),
        S=torch.eye(3, dtype=dtype)[None, :, :],
        a=torch.ones((1, 3), dtype=dtype),
    )
    state = SimpleNamespace(token_field=token_field, observation_anchors=obs)
    with pytest.raises(RuntimeError, match="candidate mask shape mismatch"):
        compute_alignment_loss(state)


def test_alignment_loss_keeps_align_heads_in_graph_when_no_projective_candidates(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    geom = output.state.token_field.projective_geometry
    assert geom is not None
    empty_geom = dataclasses.replace(
        geom,
        projective_compatibility=torch.zeros_like(geom.projective_compatibility),
        projective_candidate_mask=torch.zeros_like(geom.projective_candidate_mask),
    )
    empty_token_field = dataclasses.replace(output.state.token_field, projective_geometry=empty_geom)
    empty_state = dataclasses.replace(output.state, token_field=empty_token_field)
    core.zero_grad(set_to_none=True)
    alignment = compute_alignment_loss(empty_state)
    alignment.total.backward()
    assert core.point_align_proj.weight.grad is not None
    assert core.visual_align_proj.weight.grad is not None
    assert torch.allclose(core.point_align_proj.weight.grad, torch.zeros_like(core.point_align_proj.weight.grad))
    assert torch.allclose(core.visual_align_proj.weight.grad, torch.zeros_like(core.visual_align_proj.weight.grad))


def test_alignment_loss_tau_pv_changes_bag_contrastive_temperature(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    warm = compute_alignment_loss(output.state, config=PicfAlignmentLossConfig(tau_pv=1.0))
    cold = compute_alignment_loss(output.state, config=PicfAlignmentLossConfig(tau_pv=0.01))
    assert torch.isfinite(warm.pv_weak)
    assert torch.isfinite(cold.pv_weak)
    assert not torch.allclose(warm.pv_weak, cold.pv_weak)


def test_alignment_loss_emits_focus_loss_from_fusion_attention(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    alignment = compute_alignment_loss(output.state, config=PicfAlignmentLossConfig(lambda_focus_pv=1.0))
    assert torch.isfinite(alignment.focus_pv)
    assert alignment.focus_pv.item() >= 0.0


def test_alignment_loss_suppresses_low_support_false_positive_routing() -> None:
    dtype = torch.float32
    geometry = PicfProjectiveGeometryState(
        point_proj_grid_norm=torch.zeros((1, 2), dtype=dtype),
        point_proj_grid_index=torch.zeros((1, 2), dtype=dtype),
        point_visibility=torch.ones((1,), dtype=dtype),
        point_depth=torch.ones((1,), dtype=dtype),
        point_depth_sample=torch.ones((1,), dtype=dtype),
        point_depth_valid=torch.ones((1,), dtype=torch.bool),
        visual_grid_norm=torch.zeros((1, 2), dtype=dtype),
        visual_grid_index=torch.zeros((1, 2), dtype=dtype),
        visual_pixel_centers=torch.zeros((1, 2), dtype=dtype),
        visual_ray_world=torch.zeros((1, 3), dtype=dtype),
        camera_origin_world=torch.zeros((3,), dtype=dtype),
        projective_compatibility=torch.ones((1, 1), dtype=dtype),
        projective_candidate_mask=torch.ones((1, 1), dtype=torch.bool),
        projective_attention_bias=torch.zeros((1, 1), dtype=dtype),
    )
    token_field = PicfTokenFieldState(
        point_tokens=torch.zeros((1, 4), dtype=dtype),
        visual_tokens=torch.zeros((1, 4), dtype=dtype),
        tactile_tokens=torch.zeros((0, 4), dtype=dtype),
        context_tokens=torch.zeros((0, 4), dtype=dtype),
        fused_tokens=torch.zeros((2, 4), dtype=dtype),
        point_positions=torch.zeros((1, 3), dtype=dtype),
        modality_ids=torch.tensor([0, 1], dtype=torch.long),
        point_align_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        visual_align_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        tactile_align_embeddings=torch.zeros((0, 4), dtype=dtype),
        tactile_positions_world=torch.zeros((0, 3), dtype=dtype),
        tactile_contact_gate=torch.zeros((0,), dtype=dtype),
        fusion_attention_mean=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype),
        projective_geometry=geometry,
    )

    def _state(scale: float) -> SimpleNamespace:
        routing_mass_point = torch.full((2, 1), scale, dtype=dtype)
        routing_mass_visual = torch.full((2, 1), scale, dtype=dtype)
        support = torch.full((1,), 2.0 * scale, dtype=dtype)
        gate = support / (support + 0.1)
        obs = PicfObservationAnchorState(
            seed_indices=torch.tensor([0, -1], dtype=torch.long),
            tokens=torch.zeros((2, 4), dtype=dtype),
            point_weights=torch.full((2, 1), 0.5, dtype=dtype),
            routing_mass_point=routing_mass_point,
            routing_mass_visual=routing_mass_visual,
            routing_support_point=support,
            routing_support_visual=support,
            routing_gate_point=gate,
            routing_gate_visual=gate,
            x=torch.zeros((2, 3), dtype=dtype),
            S=torch.eye(3, dtype=dtype)[None, :, :].expand(2, -1, -1).clone(),
            a=torch.ones((2, 3), dtype=dtype),
        )
        return SimpleNamespace(token_field=token_field, observation_anchors=obs)

    high_support = compute_alignment_loss(_state(0.5), config=PicfAlignmentLossConfig(tau_route_p=0.1, tau_route_v=0.1))
    low_support = compute_alignment_loss(_state(1e-3), config=PicfAlignmentLossConfig(tau_route_p=0.1, tau_route_v=0.1))
    assert torch.isfinite(high_support.anchor_pv)
    assert torch.isfinite(low_support.anchor_pv)
    assert low_support.anchor_pv > high_support.anchor_pv


def test_alignment_loss_point_tactile_branch_uses_tau_pt(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    frame.force_vec = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    warm = compute_alignment_loss(output.state, config=PicfAlignmentLossConfig(lambda_pt=1.0, tau_pt=1.0))
    cold = compute_alignment_loss(output.state, config=PicfAlignmentLossConfig(lambda_pt=1.0, tau_pt=0.01))
    assert torch.isfinite(warm.pt)
    assert torch.isfinite(cold.pt)
    assert not torch.allclose(warm.pt, cold.pt)


def test_alignment_loss_point_tactile_branch_defaults_off_without_explicit_contact(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=np.full((4, 4, 8), 1.0, dtype=np.float32),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    gate = output.state.token_field.tactile_contact_gate
    assert gate.shape[0] == 2
    assert torch.allclose(gate, torch.zeros_like(gate))
    assert output.state.token_field.tactile_tokens_all is not None
    assert output.state.token_field.tactile_tokens_all.shape[0] == 2
    assert output.state.token_field.tactile_tokens.shape[0] == 0
    alignment = compute_alignment_loss(output.state, config=PicfAlignmentLossConfig(lambda_pt=1.0))
    assert torch.allclose(alignment.pt, torch.zeros_like(alignment.pt))


def test_alignment_loss_point_tactile_branch_uses_pseudo_contact_from_tactile_history(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01, contact_shift=25)

    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )
    second = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(1.0),
        semantic_override=np.ones((core.config.semantic_dim,), dtype=np.float32),
    )

    gate = second.state.token_field.tactile_contact_gate
    assert gate.shape[0] == 2
    assert torch.all(gate > 0.0)
    assert second.state.token_field.tactile_tokens.shape[0] == 2 * core.config.tactile_group_proposals
    alignment = compute_alignment_loss(second.state, config=PicfAlignmentLossConfig(lambda_pt=1.0))
    assert torch.isfinite(alignment.pt)
    assert alignment.pt > 0.0


def test_alignment_loss_uses_fingertip_local_bag_and_front_halfspace() -> None:
    dtype = torch.float32
    token_field = PicfTokenFieldState(
        point_tokens=torch.zeros((2, 2), dtype=dtype),
        visual_tokens=torch.zeros((0, 2), dtype=dtype),
        tactile_tokens=torch.zeros((0, 2), dtype=dtype),
        context_tokens=torch.zeros((0, 2), dtype=dtype),
        fused_tokens=torch.zeros((0, 2), dtype=dtype),
        point_positions=torch.tensor([[-0.001, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=dtype),
        modality_ids=torch.zeros((0,), dtype=torch.long),
        point_align_embeddings=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype),
        visual_align_embeddings=torch.zeros((0, 2), dtype=dtype),
        tactile_align_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype),
        tactile_positions_world=torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=dtype),
        tactile_contact_gate=torch.tensor([1.0, 0.0], dtype=dtype),
        tactile_contact_prob=torch.tensor([1.0, 0.2], dtype=dtype),
        tactile_normals_world=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
    )
    obs = PicfObservationAnchorState(
        seed_indices=torch.zeros((0,), dtype=torch.long),
        tokens=torch.zeros((0, 2), dtype=dtype),
        point_weights=torch.zeros((0, 2), dtype=dtype),
        routing_mass_point=torch.zeros((0, 2), dtype=dtype),
        routing_mass_visual=torch.zeros((0, 0), dtype=dtype),
        routing_support_point=torch.zeros((2,), dtype=dtype),
        routing_support_visual=torch.zeros((0,), dtype=dtype),
        routing_gate_point=torch.zeros((2,), dtype=dtype),
        routing_gate_visual=torch.zeros((0,), dtype=dtype),
        x=torch.zeros((0, 3), dtype=dtype),
        S=torch.zeros((0, 3, 3), dtype=dtype),
        a=torch.zeros((0, 3), dtype=dtype),
    )
    state = SimpleNamespace(token_field=token_field, observation_anchors=obs)
    alignment = compute_alignment_loss(
        state,
        config=PicfAlignmentLossConfig(
            lambda_pt=1.0,
            tau_pt=0.1,
            pt_bag_radius_m=0.03,
            pt_bag_sigma_m=0.01,
            pt_bag_kmin=1,
            pt_back_slack_m=0.0005,
            p_align_on=0.55,
            p_align_off=0.35,
        ),
    )
    assert torch.isfinite(alignment.pt)
    assert float(alignment.pt.item()) < 1e-3

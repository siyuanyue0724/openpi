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
from openpi.picf.core.contracts import PicfVLGroundingState
from openpi.picf.core import pipeline as pipeline_module
from openpi.picf.core.pipeline import PicfFullCore
from openpi.picf.core.pipeline import _variance_from_logvar
from openpi.picf.core.training import compute_alignment_loss
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.visual_expert import _project_world_points
from openpi.picf.posterior.visual_expert import _scale_to_grid
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticFeatures
from openpi.picf.paligemma.wrapper import PaliGemmaViewTransform
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


class _FeatureMapFromClip:
    def __init__(self, clip: np.ndarray):
        value = float(np.asarray(clip)[-1].mean()) / 255.0
        self._map = np.full((4, 4, 8), value, dtype=np.float32)

    def current_map(self, *, use_last_two_mean: bool = False) -> np.ndarray:
        del use_last_two_mean
        return self._map


class _TemporalModeFeatureMap:
    def __init__(self) -> None:
        self.requested: list[int] = []

    def current_map(self, *, use_last_two_mean: bool = False) -> np.ndarray:
        del use_last_two_mean
        return np.zeros((4, 4, 8), dtype=np.float32)

    def recent_maps(self, n: int = 2) -> np.ndarray:
        self.requested.append(int(n))
        return np.zeros((int(n), 4, 4, 8), dtype=np.float32)


class _TemporalModeVisualEncoder:
    def __init__(self) -> None:
        self.feature_map = _TemporalModeFeatureMap()

    def encode_clip(self, _clip: np.ndarray) -> _TemporalModeFeatureMap:
        return self.feature_map


def test_transformer_stack_uses_activation_checkpointing_during_training(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def _checkpoint(fn, *args, **kwargs):
        calls.append(1)
        return fn(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _checkpoint)
    stack = pipeline_module.TransformerStack(16, 4, 2, activation_checkpointing=True).train()
    x = torch.randn(1, 5, 16, requires_grad=True)
    y = stack(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert len(calls) == 2


def test_transformer_stack_skips_checkpointing_for_attention_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("attention-return path should not use activation checkpointing")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _boom)
    stack = pipeline_module.TransformerStack(16, 4, 2, activation_checkpointing=True).train()
    x = torch.randn(1, 5, 16, requires_grad=True)
    y, attn = stack(x, return_attention=True)
    assert y.shape == x.shape
    assert attn is not None


def test_transformer_stack_clones_view_inputs_before_attention() -> None:
    stack = pipeline_module.TransformerStack(16, 4, 2, activation_checkpointing=False).train()
    base = torch.randn(5, 16, requires_grad=True)
    view = base[None, :]
    y, attn = stack(view, return_attention=True)
    assert y.shape == view.shape
    assert attn is not None
    y.sum().backward()
    assert base.grad is not None


def test_transformer_stack_tokenwise_ff_chunking_matches_unchunked() -> None:
    torch.manual_seed(0)
    base = pipeline_module.TransformerStack(
        16,
        4,
        2,
        activation_checkpointing=False,
        ff_chunk_size=0,
    ).eval()
    chunked = pipeline_module.TransformerStack(
        16,
        4,
        2,
        activation_checkpointing=False,
        ff_chunk_size=2,
    ).eval()
    chunked.load_state_dict(base.state_dict())
    x = torch.randn(1, 5, 16)
    y_base = base(x)
    y_chunked = chunked(x)
    assert isinstance(y_base, torch.Tensor)
    assert isinstance(y_chunked, torch.Tensor)
    torch.testing.assert_close(y_base, y_chunked, atol=1e-5, rtol=1e-5)


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
        visual_real_grid=4,
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


def _semantic_features(value: float, *, num_tokens: int = 3, width: int = 32) -> PaliGemmaSemanticFeatures:
    tokens = torch.full((num_tokens, width), value, dtype=torch.float32)
    summary = torch.full((1, width), value, dtype=torch.float32)
    return PaliGemmaSemanticFeatures(tokens=tokens, summary=summary)


def _semantic_features_with_spatial(value: float, *, width: int = 32) -> PaliGemmaSemanticFeatures:
    tokens = torch.full((3, width), value, dtype=torch.float32)
    summary = torch.full((1, width), value, dtype=torch.float32)
    image_tokens = torch.arange(4 * width, dtype=torch.float32).reshape(4, width) / 100.0
    text_tokens = torch.full((2, width), value + 0.5, dtype=torch.float32)
    return PaliGemmaSemanticFeatures(
        tokens=tokens,
        summary=summary,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        image_token_ranges=((0, 4),),
        image_grid_shapes=((2, 2),),
        image_view_names=("static",),
    )


def test_vl_point_prior_uses_column_normalized_projective_mass() -> None:
    compatibility = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    heatmap = torch.tensor([0.5, 0.5], dtype=torch.float32)

    prior, valid, visible_mass = pipeline_module._point_prior_from_heatmap(
        compatibility,
        heatmap,
        point_projectable_mask=torch.ones((3,), dtype=torch.bool),
        min_visible_mass=1e-4,
        eps=1e-6,
    )

    assert bool(valid.item())
    torch.testing.assert_close(visible_mass, torch.tensor(1.0))
    torch.testing.assert_close(prior, torch.tensor([0.25, 0.25, 0.5]), atol=1e-6, rtol=1e-6)


def test_vl_point_prior_invalid_projection_is_zero_not_top_left_fallback() -> None:
    compatibility = torch.zeros((3, 2), dtype=torch.float32)
    heatmap = torch.tensor([1.0, 0.0], dtype=torch.float32)

    prior, valid, visible_mass = pipeline_module._point_prior_from_heatmap(
        compatibility,
        heatmap,
        point_projectable_mask=torch.ones((3,), dtype=torch.bool),
        min_visible_mass=1e-4,
        eps=1e-6,
    )

    assert not bool(valid.item())
    torch.testing.assert_close(visible_mass, torch.tensor(0.0))
    torch.testing.assert_close(prior, torch.zeros((3,)))


def test_vl_heatmap_resize_preserves_probability_mass() -> None:
    heatmap = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    resized = pipeline_module._resize_flat_heatmap(
        heatmap,
        src_hw=(2, 2),
        dst_hw=(4, 4),
        eps=1e-6,
    )

    assert resized.shape == (16,)
    assert bool(torch.all(resized >= 0.0).item())
    torch.testing.assert_close(resized.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_vl_heatmap_mapping_uses_resize_with_pad_transform() -> None:
    transform = PaliGemmaViewTransform(
        original_hw=(100, 200),
        target_hw=(224, 224),
        resized_hw=(112, 224),
        pad_top=56,
        pad_bottom=56,
        pad_left=0,
        pad_right=0,
        scale_y=112.0 / 100.0,
        scale_x=224.0 / 200.0,
    )
    heatmap = torch.arange(1, 17, dtype=torch.float32)

    naive = pipeline_module._resize_flat_heatmap(
        heatmap,
        src_hw=(4, 4),
        dst_hw=(4, 4),
        eps=1e-6,
    )
    mapped = pipeline_module._map_pg_heatmap_to_visual_grid(
        heatmap,
        src_hw=(4, 4),
        dst_hw=(4, 4),
        view_transform=transform,
        eps=1e-6,
    )

    assert mapped.shape == (16,)
    assert bool(torch.all(mapped >= 0.0).item())
    torch.testing.assert_close(mapped.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(mapped, naive)


def test_semantic_context_carries_paligemma_view_transforms(tmp_path: Path) -> None:
    core, _replay = _make_core(tmp_path)
    transform = PaliGemmaViewTransform(
        original_hw=(100, 200),
        target_hw=(224, 224),
        resized_hw=(112, 224),
        pad_top=56,
        pad_bottom=56,
        pad_left=0,
        pad_right=0,
        scale_y=112.0 / 100.0,
        scale_x=224.0 / 200.0,
    )
    features = PaliGemmaSemanticFeatures(
        tokens=torch.ones((2, core.config.semantic_dim), dtype=torch.float32),
        summary=torch.ones((1, core.config.semantic_dim), dtype=torch.float32),
        image_tokens=torch.ones((4, core.config.semantic_dim), dtype=torch.float32),
        text_tokens=torch.ones((2, core.config.semantic_dim), dtype=torch.float32),
        image_token_ranges=((0, 4),),
        image_grid_shapes=((2, 2),),
        image_view_names=("static",),
        image_view_transforms=(transform,),
    )

    context = core._project_semantic_context(tokens_raw=features.tokens, features=features)

    assert context.image_view_transforms == (transform,)


def test_vl_point_prior_projectable_mask_excludes_local_frame_rows() -> None:
    compatibility = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    heatmap = torch.tensor([0.5, 0.5], dtype=torch.float32)
    # Row 0 represents a local-frame point. The VL 2D-to-3D lift must not assign
    # heatmap mass to it unless the code has explicitly converted it to world
    # coordinates.
    projectable = torch.tensor([False, True, True], dtype=torch.bool)

    prior, valid, visible_mass = pipeline_module._point_prior_from_heatmap(
        compatibility,
        heatmap,
        point_projectable_mask=projectable,
        min_visible_mass=1e-4,
        eps=1e-6,
    )

    assert bool(valid.item())
    torch.testing.assert_close(visible_mass, torch.tensor(1.0))
    torch.testing.assert_close(prior, torch.tensor([0.0, 0.5, 0.5]), atol=1e-6, rtol=1e-6)


def test_scene_point_candidate_mask_rejects_projective_border_points(tmp_path: Path) -> None:
    core, _replay = _make_core(tmp_path, scene_anchor_border_patches=1.0)
    visual_grid = torch.tensor(
        [[x, y] for y in range(4) for x in range(4)],
        dtype=torch.float32,
    )
    geom = pipeline_module.PicfProjectiveGeometryState(
        point_proj_grid_norm=torch.zeros((3, 2)),
        point_proj_grid_index=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        point_visibility=torch.ones((3,)),
        point_depth=torch.ones((3,)),
        point_depth_sample=torch.ones((3,)),
        point_depth_valid=torch.ones((3,), dtype=torch.bool),
        visual_grid_norm=torch.zeros((16, 2)),
        visual_grid_index=visual_grid,
        visual_pixel_centers=torch.zeros((16, 2)),
        visual_ray_world=torch.zeros((16, 3)),
        camera_origin_world=torch.zeros((3,)),
        projective_compatibility=torch.ones((3, 16)),
        projective_candidate_mask=torch.ones((3, 16), dtype=torch.bool),
        projective_attention_bias=torch.zeros((3, 16)),
    )
    zeros_h = torch.zeros((0, core.config.hidden_dim), dtype=torch.float32)
    token_field = pipeline_module.PicfTokenFieldState(
        point_tokens=torch.zeros((3, core.config.hidden_dim), dtype=torch.float32),
        visual_tokens=zeros_h,
        tactile_tokens=zeros_h,
        context_tokens=zeros_h,
        fused_tokens=torch.zeros((3, core.config.hidden_dim), dtype=torch.float32),
        point_positions=torch.zeros((3, 3), dtype=torch.float32),
        modality_ids=torch.zeros((3,), dtype=torch.long),
        point_align_embeddings=torch.zeros((3, core.config.hidden_dim), dtype=torch.float32),
        visual_align_embeddings=zeros_h,
        tactile_align_embeddings=zeros_h,
        tactile_positions_world=torch.zeros((0, 3), dtype=torch.float32),
        tactile_contact_gate=torch.zeros((0,), dtype=torch.float32),
        projective_geometry=geom,
        point_pool_ids=torch.ones((3,), dtype=torch.long),
    )

    mask = core._scene_point_candidate_mask(token_field)

    torch.testing.assert_close(mask.cpu(), torch.tensor([False, True, True]))
    strict_mask = core._scene_point_candidate_mask(token_field, fallback_to_global=False)
    torch.testing.assert_close(strict_mask.cpu(), torch.tensor([False, True, True]))

    border_only_geom = dataclasses.replace(
        geom,
        point_proj_grid_index=torch.tensor([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0]], dtype=torch.float32),
    )
    border_only_field = dataclasses.replace(token_field, projective_geometry=border_only_geom)

    # Coverage seeding may fall back to all global scene points, but VL 2D-to-3D
    # lift must stay invalid/no-op if every scene point is a projected border
    # artifact.
    fallback_mask = core._scene_point_candidate_mask(border_only_field)
    strict_border_mask = core._scene_point_candidate_mask(border_only_field, fallback_to_global=False)

    torch.testing.assert_close(fallback_mask.cpu(), torch.tensor([True, True, True]))
    torch.testing.assert_close(strict_border_mask.cpu(), torch.tensor([False, False, False]))


def test_weighted_anchor_modes_preserve_separated_high_weight_modes() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

    modes = pipeline_module._weighted_anchor_modes(positions, weights, count=2, radius_m=0.1)

    assert modes.tolist() == [0, 2]


def test_vl_grounding_disabled_does_not_instantiate_router_modules(tmp_path: Path) -> None:
    core, _replay = _make_core(tmp_path)

    assert core.config.vl_anchor_router_enabled is False
    assert core.vl_heatmap_head is None
    assert core.vl_anchor_token_proj is None
    assert core.vl_task_point_gate_logit is None
    assert core.vl_obs_anchor_gate_logit is None
    assert core.vl_posterior_bind_gate_logit is None


def _manual_vl_grounding(point_count: int) -> PicfVLGroundingState:
    heat = torch.full((point_count,), 1.0 / max(point_count, 1), dtype=torch.float32)
    anchor_priors = torch.eye(point_count, dtype=torch.float32)[:3]
    return PicfVLGroundingState(
        task_heatmap_logits=heat,
        effector_heatmap_logits=heat,
        interaction_heatmap_logits=heat,
        task_heatmap=heat,
        effector_heatmap=heat,
        interaction_heatmap=heat,
        task_point_prior=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)[:point_count],
        effector_point_prior=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)[:point_count],
        interaction_point_prior=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)[:point_count],
        anchor_point_priors=anchor_priors[:, :point_count],
        anchor_x=torch.zeros((anchor_priors.shape[0], 3), dtype=torch.float32),
        anchor_S=torch.eye(3, dtype=torch.float32)[None, :, :].expand(anchor_priors.shape[0], -1, -1).clone(),
        anchor_tokens=torch.zeros((anchor_priors.shape[0], 64), dtype=torch.float32),
        anchor_roles=torch.tensor([0, 1, 2], dtype=torch.long)[: anchor_priors.shape[0]],
        anchor_scores=torch.ones((anchor_priors.shape[0],), dtype=torch.float32),
        visual_pixel_centers=None,
        valid=torch.tensor(True),
        confidence=torch.tensor(1.0),
    )


def test_vl_slot_point_priors_are_role_aware(tmp_path: Path) -> None:
    core, _replay = _make_core(tmp_path, vl_anchor_router_enabled=True)
    grounding = _manual_vl_grounding(point_count=3)
    slot_roles = torch.tensor([0, 1, 1], dtype=torch.long)

    priors, valid = core._vl_slot_point_priors(grounding, slot_roles, point_count=3)

    assert valid.tolist() == [True, True, True]
    torch.testing.assert_close(priors[0], torch.tensor([1.0, 0.0, 0.0], device=priors.device))
    torch.testing.assert_close(priors[1], torch.tensor([0.0, 1.0, 0.0], device=priors.device))
    torch.testing.assert_close(priors[2], torch.tensor([0.0, 0.0, 1.0], device=priors.device))


def test_vl_observation_seed_does_not_override_effector_role_slots(tmp_path: Path) -> None:
    core, _replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=True,
        vl_obs_anchor_gate_init=20.0,
        observation_anchors=4,
        effector_observation_anchors=2,
    )
    hidden = core.config.hidden_dim
    device = core.device
    point_tokens = torch.arange(4 * hidden, device=device, dtype=torch.float32).reshape(4, hidden)
    zeros_h = torch.zeros((0, hidden), device=device, dtype=torch.float32)
    token_field = pipeline_module.PicfTokenFieldState(
        point_tokens=point_tokens,
        visual_tokens=zeros_h,
        tactile_tokens=zeros_h,
        context_tokens=zeros_h,
        fused_tokens=point_tokens.clone(),
        point_positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        ),
        modality_ids=torch.zeros((4,), device=device, dtype=torch.long),
        point_align_embeddings=torch.zeros((4, hidden), device=device, dtype=torch.float32),
        visual_align_embeddings=zeros_h,
        tactile_align_embeddings=zeros_h,
        tactile_positions_world=torch.zeros((0, 3), device=device, dtype=torch.float32),
        tactile_contact_gate=torch.zeros((0,), device=device, dtype=torch.float32),
        point_pool_ids=torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long),
    )
    grounding = PicfVLGroundingState(
        task_heatmap_logits=torch.full((4,), 0.25, device=device, dtype=torch.float32),
        effector_heatmap_logits=torch.full((4,), 0.25, device=device, dtype=torch.float32),
        interaction_heatmap_logits=torch.full((4,), 0.25, device=device, dtype=torch.float32),
        task_heatmap=torch.full((4,), 0.25, device=device, dtype=torch.float32),
        effector_heatmap=torch.full((4,), 0.25, device=device, dtype=torch.float32),
        interaction_heatmap=torch.full((4,), 0.25, device=device, dtype=torch.float32),
        task_point_prior=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32),
        effector_point_prior=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32),
        interaction_point_prior=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32),
        anchor_point_priors=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        ),
        anchor_x=torch.zeros((3, 3), device=device, dtype=torch.float32),
        anchor_S=torch.eye(3, device=device, dtype=torch.float32)[None, :, :].expand(3, -1, -1).clone(),
        anchor_tokens=torch.zeros((3, hidden), device=device, dtype=torch.float32),
        anchor_roles=torch.tensor([0, 1, 2], device=device, dtype=torch.long),
        anchor_scores=torch.ones((3,), device=device, dtype=torch.float32),
        visual_pixel_centers=None,
        valid=torch.tensor(True, device=device),
        confidence=torch.tensor(1.0, device=device),
    )

    obs = core._build_observation_anchors(token_field, vl_grounding=grounding)

    assert obs.role_ids is not None
    torch.testing.assert_close(obs.role_ids, torch.tensor([0, 0, 1, 1], dtype=torch.long, device=obs.role_ids.device))
    assert bool(torch.all(obs.seed_indices[:2] < 2).item())
    assert bool(torch.all(obs.seed_indices[2:] >= 2).item())


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
    assert output.state.token_field.context_tokens.shape[0] == 4
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
    assert output.state.predictive.semantic_tokens.shape == (3, core.config.semantic_dim)
    assert output.state.predictive.control_query_state.shape == (core.config.semantic_dim,)
    assert output.state.predictive.predictive_query_state.shape == (core.config.semantic_dim,)
    assert output.state.predictive.global_pred.shape == (core.config.hidden_dim,)
    assert output.state.token_field.fusion_attention_mean is not None
    assert output.state.vl_grounding is None


def test_vl_grounding_enabled_builds_state_without_changing_default_anchor_contract(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=True,
        vl_anchor_modes=3,
        vl_obs_anchor_gate_init=20.0,
        vl_task_point_gate_init=20.0,
        vl_posterior_bind_gate_init=20.0,
    )
    frame = next(iter(replay))
    output = core.step(
        frame,
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )

    vl = output.state.vl_grounding
    assert vl is not None
    assert vl.task_heatmap.shape == (output.state.token_field.visual_tokens.shape[0],)
    assert vl.task_point_prior.shape == (output.state.token_field.point_tokens.shape[0],)
    assert vl.anchor_point_priors.shape[1] == output.state.token_field.point_tokens.shape[0]
    assert output.state.observation_anchors.tokens.shape == (core.config.observation_anchors, core.config.hidden_dim)
    assert output.state.task_readout.point_weights.shape[0] == core.config.task_local_queries
    assert "vl_grounding_valid" in output.debug
    assert "vl_grounding_confidence" in output.debug
    assert "vl_grounding_anchor_count" in output.debug
    if bool(vl.valid.item()) and output.state.task_readout.point_weights.shape[1] > 0:
        expected = torch.ones(
            (output.state.task_readout.point_weights.shape[0],),
            device=output.state.task_readout.point_weights.device,
            dtype=output.state.task_readout.point_weights.dtype,
        )
        torch.testing.assert_close(output.state.task_readout.point_weights.sum(dim=-1), expected, atol=1e-5, rtol=1e-5)


def test_mapg_enabled_builds_full_anchor_graph_and_live_consumers(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=True,
        mapg_enabled=True,
        mapg_anchor_count=6,
        mapg_message_rounds=2,
        mapg_obs_gate_init=20.0,
        mapg_task_gate_init=20.0,
        mapg_posterior_gate_init=20.0,
        mapg_control_gate_init=20.0,
        tactile_contact_tau_on=0.01,
        tactile_contact_tau_off=0.005,
        tactile_anchor_prob_on=0.01,
    )
    # MAPG modality projections must not depend on a modality being present in
    # the first warmup window; FSDP materialization runs before every modality is
    # guaranteed to produce tokens.
    for module in (
        core.mapg_visual_proj,
        core.mapg_point_proj,
        core.mapg_tactile_proj,
        core.mapg_posterior_proj,
        core.mapg_task_visual_proj,
        core.mapg_to_control_proj,
    ):
        assert module is not None
        assert not isinstance(module.weight, torch.nn.parameter.UninitializedParameter)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id, contact_shift=12)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, contact_shift=16)
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )
    second = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(2.0),
        semantic_override=_semantic_features_with_spatial(2.0),
    )

    graph = second.state.anchor_prior_graph
    assert graph is not None
    assert bool(graph.valid.item())
    assert graph.anchor_tokens.shape == (core.config.mapg_anchor_count, core.config.hidden_dim)
    assert graph.visual_priors.shape == (core.config.mapg_anchor_count, second.state.token_field.visual_tokens.shape[0])
    assert graph.point_priors is not None
    assert graph.point_priors.shape == (core.config.mapg_anchor_count, second.state.token_field.point_tokens.shape[0])
    assert graph.tactile_priors is not None
    assert graph.tactile_priors.shape[0] == core.config.mapg_anchor_count
    assert graph.posterior_priors is not None
    assert graph.posterior_priors.shape == (core.config.mapg_anchor_count, core.config.persistent_anchors)
    assert graph.obs_slot_assignment is not None
    assert graph.obs_slot_assignment.shape == (core.config.observation_anchors, core.config.mapg_anchor_count)
    torch.testing.assert_close(
        graph.obs_slot_assignment.sum(dim=-1),
        torch.ones((core.config.observation_anchors,), device=graph.obs_slot_assignment.device, dtype=graph.obs_slot_assignment.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert graph.task_assignment is not None
    assert graph.task_assignment.shape == (core.config.task_local_queries, core.config.mapg_anchor_count)
    torch.testing.assert_close(
        graph.task_assignment.sum(dim=-1),
        torch.ones((core.config.task_local_queries,), device=graph.task_assignment.device, dtype=graph.task_assignment.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert second.state.observation_anchors.graph_assignment is not None
    assert second.state.observation_anchors.graph_point_weights is not None
    assert second.state.observation_anchors.graph_visual_weights is not None
    assert second.state.task_readout.graph_assignment is not None
    assert second.state.task_readout.visual_weights is not None
    assert second.state.task_readout.geometry_valid is not None
    assert second.state.task_readout.graph_visual_weights is not None
    assert second.state.task_readout.graph_tactile_weights is not None
    assert second.state.conditioned_control.graph_tokens is not None
    assert second.state.conditioned_control.graph_tokens.shape == (
        core.config.mapg_anchor_count,
        core.config.semantic_dim,
    )
    assert second.state.conditioned_control.tokens.shape[0] >= core.config.mapg_anchor_count
    assert second.debug["mapg_anchor_count"] == float(core.config.mapg_anchor_count)
    assert second.debug["mapg_point_available"] == 1.0
    assert second.debug["mapg_tactile_available"] == 1.0
    assert second.debug["mapg_posterior_available"] == 1.0


def test_mapg_builds_paligemma_grounding_without_point_router(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=False,
        mapg_enabled=True,
        mapg_anchor_count=4,
        mapg_message_rounds=1,
        mapg_task_gate_init=20.0,
    )
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )

    assert output.state.vl_grounding is not None
    assert output.state.anchor_prior_graph is not None
    assert output.state.anchor_prior_graph.pg_priors is not None
    assert output.state.anchor_prior_graph.visual_priors.shape[0] == core.config.mapg_anchor_count
    assert output.state.task_readout.visual_weights is not None
    assert output.state.task_readout.graph_visual_weights is not None


def test_mapg_observation_point_mix_floor_reaches_final_point_weights(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=True,
        mapg_enabled=True,
        mapg_anchor_count=6,
        mapg_message_rounds=2,
        mapg_obs_gate_init=-20.0,
        mapg_obs_point_mix_floor=0.5,
    )
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )

    obs = output.state.observation_anchors
    assert obs.graph_point_weights is not None
    assert obs.graph_point_weights.shape == obs.point_weights.shape
    direct = obs.routing_mass_point / torch.clamp(obs.routing_mass_point.sum(dim=-1, keepdim=True), min=core.config.epsilon_a)
    graph_valid = obs.graph_point_weights.sum(dim=-1) > core.config.epsilon_a
    graph_mix = torch.where(graph_valid[:, None], obs.graph_point_weights, direct)
    expected = (0.5 * direct) + (0.5 * graph_mix)
    expected = expected / torch.clamp(expected.sum(dim=-1, keepdim=True), min=core.config.epsilon_a)
    torch.testing.assert_close(obs.point_weights, expected, atol=1e-5, rtol=1e-5)


def test_mapg_visual_grounding_survives_missing_pointcloud(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=False,
        mapg_enabled=True,
        mapg_anchor_count=4,
        mapg_message_rounds=1,
        mapg_task_gate_init=20.0,
    )
    frame = next(iter(replay))
    missing_point = PicfPointCloudFrame(
        grid_coord=np.zeros((0, 3), dtype=np.int32),
        xyz_world=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.float32),
        normal_world=np.zeros((0, 3), dtype=np.float32),
        valid_point_mask=np.zeros((0,), dtype=bool),
        frame_valid=False,
    )
    output = core.step(
        dataclasses.replace(frame, point_set=missing_point),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )

    assert not output.state.runtime_meta.point_contract_ok
    assert output.state.vl_grounding is not None
    # Point-centric consumers stay disabled when point lift is invalid, but MAPG
    # must still receive language-conditioned visual heatmaps.
    assert not bool(output.state.vl_grounding.valid.item())
    assert output.state.vl_grounding.task_heatmap.numel() == output.state.token_field.visual_tokens.shape[0]
    torch.testing.assert_close(
        output.state.vl_grounding.task_heatmap.sum(),
        torch.ones((), dtype=output.state.vl_grounding.task_heatmap.dtype, device=output.state.vl_grounding.task_heatmap.device),
        atol=1e-5,
        rtol=1e-5,
    )
    graph = output.state.anchor_prior_graph
    assert graph is not None
    assert bool(graph.valid.item())
    assert graph.pg_priors is not None
    assert graph.visual_priors.shape == (core.config.mapg_anchor_count, output.state.token_field.visual_tokens.shape[0])
    assert graph.point_priors is None
    assert output.state.task_readout.visual_weights is not None
    assert output.state.task_readout.geometry_valid is not None
    assert not bool(output.state.task_readout.geometry_valid.any().item())


def test_aqr_owm_populates_temporal_pg_and_address_contracts(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        aqr_mapg_enabled=True,
        aqr_query_count_physical=4,
        aqr_query_count_task=3,
        aqr_query_rounds=1,
        aqr_pg_image_support_weight=0.0,
    )
    frame = next(iter(replay))
    temporal_visual = np.stack([_visual_override(1.0), _visual_override(2.0)], axis=0)

    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=temporal_visual,
        semantic_override=_semantic_features_with_spatial(1.0),
    )

    graph = output.state.anchor_prior_graph
    assert graph is not None
    assert graph.pg_priors is not None
    assert graph.pg_priors.shape[0] == 7
    assert graph.pg_priors.shape[1] == 4
    assert graph.vjepa_temporal_priors is not None
    # Two recent maps plus the optional delta map, each with 4x4 visual cells.
    assert graph.vjepa_temporal_priors.shape == (7, 3 * 16)
    assert graph.slot_address is not None
    assert graph.slot_address.shape == (4, core.config.hidden_dim)
    assert graph.slot_content is not None
    assert output.state.token_field.temporal_visual is not None
    assert output.state.posterior.slot_address is not None
    assert output.state.posterior.slot_content is not None
    assert output.state.predictive.slot_prediction_tokens is not None
    assert output.state.predictive.slot_prediction_supports is not None
    assert output.state.predictive.slot_prediction_supports.shape == (core.config.persistent_anchors, 4)
    assert output.debug["owm_pg_priors_available"] == 1.0
    assert output.debug["owm_temporal_priors_available"] == 1.0
    assert "aqr_temporal_support_entropy_mean" in output.debug
    assert "aqr_temporal_support_time_mass_t0" in output.debug
    assert "aqr_temporal_support_time_mass_t1" in output.debug
    assert "aqr_pg_support_entropy_mean" in output.debug
    assert "aqr_pg_support_max" in output.debug
    assert "aqr_effective_anchor_count" in output.debug
    assert "innovation_norm_visual" in output.debug


def test_vjepa_temporal_mode_controls_recent_map_count(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        aqr_vjepa_temporal_mode="last4_tokens",
        aqr_vjepa_temporal_tokens=2,
    )
    encoder = _TemporalModeVisualEncoder()
    core.visual_encoder = encoder
    frame = next(iter(replay))
    meta = core._build_runtime_meta(frame, previous=None)

    _, temporal = core._visual_maps(frame, override=None, meta=meta)

    assert temporal is not None
    assert temporal.shape == (4, 4, 4, 8)
    assert encoder.feature_map.requested == [4]


def test_evidence_cache_is_written_after_correction_and_read_next_step_only(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        aqr_mapg_enabled=True,
        aqr_query_count_physical=4,
        aqr_query_count_task=3,
        aqr_query_rounds=1,
        evidence_cache_read_weight=0.25,
    )
    frames = list(replay)[:2]
    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=np.stack([_visual_override(1.0), _visual_override(1.5)], axis=0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )
    assert first.state.anchor_prior_graph is not None
    assert first.state.anchor_prior_graph.cache_priors is None
    assert first.state.predictive.evidence_cache is not None
    assert bool(first.state.predictive.evidence_cache.valid[0].all().item())

    second = core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=np.stack([_visual_override(2.0), _visual_override(2.5)], axis=0),
        semantic_override=_semantic_features_with_spatial(2.0),
    )

    graph = second.state.anchor_prior_graph
    assert graph is not None
    assert graph.cache_priors is not None
    assert graph.cache_priors.shape[1] == core.config.persistent_anchors
    assert second.state.predictive.evidence_cache is not None
    assert bool(second.state.predictive.evidence_cache.valid[0].all().item())
    assert bool(second.state.predictive.evidence_cache.valid[1].all().item())
    assert second.debug["owm_evidence_cache_valid_entries"] >= float(core.config.persistent_anchors)
    assert "evidence_cache_trust_mean" in second.debug
    assert "evidence_cache_age_mean" in second.debug
    assert "posterior_address_drift_mean" in second.debug
    assert "posterior_identity_switch_rate" in second.debug
    assert "posterior_recycle_rate" in second.debug


def test_ordinal_relation_state_is_prompt_gated_and_does_not_rewrite_posterior(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path, ordinal_relation_enabled=True)
    frame = next(iter(replay))
    base = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    ordinal_frame = dataclasses.replace(frame, prompt="pick the fourth object from the left")
    ordinal = core.step(
        ordinal_frame,
        point_features_override=_point_override(core, ordinal_frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )

    assert base.state.task_readout.ordinal_active is not None
    assert not bool(base.state.task_readout.ordinal_active.item())
    assert ordinal.state.task_readout.ordinal_active is not None
    assert bool(ordinal.state.task_readout.ordinal_active.item())
    assert ordinal.debug["ordinal_loss_active"] == 1.0
    torch.testing.assert_close(base.state.posterior.mu, ordinal.state.posterior.mu)
    torch.testing.assert_close(base.state.posterior.Sigma, ordinal.state.posterior.Sigma)


def test_vl_grounding_enabled_backward_does_not_mutate_query_views(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        vl_anchor_router_enabled=True,
        vl_anchor_modes=3,
        vl_obs_anchor_gate_init=20.0,
        vl_task_point_gate_init=20.0,
        vl_posterior_bind_gate_init=20.0,
    )
    frame = next(iter(replay))
    output = core.step(
        frame,
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features_with_spatial(1.0),
    )

    loss = (
        output.state.observation_anchors.tokens.square().mean()
        + output.state.task_readout.conditioned_queries.square().mean()
        + output.state.posterior.tokens.square().mean()
    )
    loss.backward()

    assert core.vl_obs_anchor_gate_logit is not None
    assert core.vl_obs_anchor_gate_logit.ndim == 1
    assert core.vl_obs_anchor_gate_logit.numel() == 1
    assert core.vl_obs_anchor_gate_logit.grad is not None
    assert core.vl_task_point_gate_logit is not None
    assert core.vl_task_point_gate_logit.ndim == 1
    assert core.vl_task_point_gate_logit.numel() == 1
    assert core.vl_posterior_bind_gate_logit is not None
    assert core.vl_posterior_bind_gate_logit.ndim == 1
    assert core.vl_posterior_bind_gate_logit.numel() == 1


def test_effector_and_scene_anchor_roles_use_separate_point_pools(tmp_path: Path) -> None:
    core, replay = _make_core(
        tmp_path,
        persistent_anchors=6,
        observation_anchors=8,
        effector_persistent_anchors=2,
        effector_observation_anchors=2,
        task_local_queries=6,
        task_effector_queries=2,
        global_scene_point_cap=16,
    )
    frame = next(iter(replay))
    frame.G_t = core.local_frame.make_transform(frame.robot_obs)
    tcp = np.asarray(frame.G_t[:3, 3], dtype=np.float32)
    local_offsets = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.00, 0.00, 0.02],
        ],
        dtype=np.float32,
    )
    global_offsets = np.asarray(
        [
            [0.35, 0.00, 0.00],
            [0.00, 0.35, 0.00],
            [0.00, 0.00, 0.35],
            [0.35, 0.35, 0.00],
            [0.20, -0.25, 0.10],
            [-0.25, 0.20, 0.15],
        ],
        dtype=np.float32,
    )
    xyz = np.concatenate([tcp[None, :] + local_offsets, tcp[None, :] + global_offsets], axis=0)
    frame.point_set = PicfPointCloudFrame(
        grid_coord=np.arange(xyz.shape[0] * 3, dtype=np.int32).reshape(xyz.shape[0], 3),
        xyz_world=xyz,
        rgb=np.linspace(0.0, 1.0, num=xyz.shape[0] * 3, dtype=np.float32).reshape(xyz.shape[0], 3),
        normal_world=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (xyz.shape[0], 1)),
        valid_point_mask=np.ones((xyz.shape[0],), dtype=bool),
        frame_valid=True,
    )
    output = core.step(
        frame,
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    point_pool_ids = output.state.token_field.point_pool_ids
    assert point_pool_ids is not None
    assert int((point_pool_ids == 0).sum().item()) == local_offsets.shape[0]
    assert int((point_pool_ids == 1).sum().item()) == xyz.shape[0]
    assert output.state.token_field.point_positions_world is not None
    np.testing.assert_allclose(
        output.state.token_field.point_positions_world.detach().cpu().numpy()[: local_offsets.shape[0]],
        xyz[: local_offsets.shape[0]],
        atol=1e-6,
    )

    obs = output.state.observation_anchors
    assert obs.role_ids is not None
    torch.testing.assert_close(obs.role_ids[:2], torch.zeros((2,), dtype=torch.long, device=obs.role_ids.device))
    torch.testing.assert_close(obs.role_ids[2:], torch.ones((6,), dtype=torch.long, device=obs.role_ids.device))
    assert bool(torch.all(obs.seed_indices[:2] < local_offsets.shape[0]).item())
    assert bool(torch.all(obs.seed_indices[2:] >= local_offsets.shape[0]).item())

    posterior_roles = output.state.posterior.role_ids
    assert posterior_roles is not None
    torch.testing.assert_close(posterior_roles[:2], torch.zeros((2,), dtype=torch.long, device=posterior_roles.device))
    torch.testing.assert_close(posterior_roles[2:], torch.ones((4,), dtype=torch.long, device=posterior_roles.device))

    task_roles = output.state.task_readout.local_role_ids
    assert task_roles is not None
    torch.testing.assert_close(task_roles[:2], torch.zeros((2,), dtype=torch.long, device=task_roles.device))
    torch.testing.assert_close(task_roles[2:], torch.ones((4,), dtype=torch.long, device=task_roles.device))


def test_refresh_predictive_state_for_action_rebuilds_cache_from_supplied_action(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    action_chunk = torch.full((1, 32), 0.25, dtype=torch.float32)
    refreshed = core.refresh_predictive_state_for_action(frame, output.state, action_future=action_chunk)
    assert refreshed.action_chunk is not None
    assert refreshed.action_chunk.shape == (1, 32)
    assert torch.allclose(
        refreshed.action,
        torch.full((7,), 0.25, dtype=torch.float32, device=refreshed.action.device),
    )
    assert refreshed.executed_action.shape == (7,)
    assert refreshed.physical_prediction_cache.visual_latent is not None
    assert refreshed.physical_prediction_cache.tactile_real is not None


def test_tactile_tokens_only_enter_fusion_when_pseudo_contact_is_active(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, pose_shift=0.01, contact_shift=25)

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
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )

    assert first.state.token_field.tactile_tokens_all is not None
    assert first.state.token_field.tactile_tokens_all.shape[0] == 2
    assert first.state.token_field.tactile_tokens.shape[0] == 0
    assert first.state.token_field.tactile_contact_prob is not None
    assert torch.all(first.state.token_field.tactile_contact_prob < core.config.tactile_anchor_prob_on)

    assert second.state.token_field.tactile_tokens_all is not None
    assert second.state.token_field.tactile_tokens_all.shape[0] == 2
    assert second.state.token_field.tactile_tokens.shape[0] == 2 * core.config.tactile_group_proposals
    assert second.state.token_field.tactile_anchor_mask is not None
    assert bool(torch.all(second.state.token_field.tactile_anchor_mask).item())


def test_hysteresis_uses_previous_contact_gate_not_anchor_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core, replay = _make_core(tmp_path)
    frames = list(replay)[:2]
    frames[0].tactile = _make_tactile_packet(frames[0].step_id, contact_shift=10)
    frames[1].tactile = _make_tactile_packet(frames[1].step_id, contact_shift=10)

    first = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    first.state.token_field.tactile_contact_gate = torch.tensor([1.0, 1.0], dtype=core.dtype)
    first.state.token_field.tactile_anchor_mask = torch.tensor([False, False], dtype=torch.bool)

    captured: dict[str, torch.Tensor | None] = {}

    def _fake_hysteresis(
        scores: torch.Tensor,
        *,
        tau_on: float,
        tau_off: float,
        temperature: float,
        ema_beta: float,
        previous_score_ema: torch.Tensor | None = None,
        previous_active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del tau_on, tau_off, temperature, ema_beta, previous_score_ema
        captured["previous_active"] = None if previous_active is None else previous_active.clone()
        scores_t = torch.as_tensor(scores, device=core.device, dtype=core.dtype)
        probs = torch.full_like(scores_t, 0.85)
        active = torch.ones_like(scores_t, dtype=torch.bool)
        return scores_t, probs, active

    monkeypatch.setattr(pipeline_module, "contact_prob_with_hysteresis", _fake_hysteresis)

    core.step(
        frames[1],
        previous=first.state,
        point_features_override=_point_override(core, frames[1]),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )

    assert captured["previous_active"] is not None
    assert bool(torch.all(captured["previous_active"]).item())


def test_gated_cross_attention_read_is_identity_when_gate_is_closed_even_if_ff_learns() -> None:
    layer = pipeline_module.GatedCrossAttentionRead(query_dim=8, kv_dim=8, heads=2, inner_dim=16)
    assert layer.cross_gate.ndim == 1
    assert layer.cross_gate.numel() == 1
    queries = torch.randn((3, 8), dtype=torch.float32)
    keys = torch.randn((4, 8), dtype=torch.float32)
    with torch.no_grad():
        layer.ff[0].weight.fill_(0.25)
        layer.ff[0].bias.fill_(0.1)
        layer.ff[-1].weight.fill_(0.2)
        layer.ff[-1].bias.fill_(0.05)
        layer.cross_gate.zero_()
    output_closed, _ = layer(queries, keys)
    torch.testing.assert_close(output_closed, queries)

    with torch.no_grad():
        layer.cross_gate.fill_(3.0)
    output_open, _ = layer(queries, keys)
    assert not torch.allclose(output_open, queries)


def test_lazy_cross_attention_read_gate_is_fsdp_compatible_vector() -> None:
    layer = pipeline_module.LazyCrossAttentionRead(query_dim=8, inner_dim=16)
    assert layer.cross_gate.ndim == 1
    assert layer.cross_gate.numel() == 1


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
    assert second.state.predictive.control_query_state.shape == (core.config.semantic_dim,)
    assert second.state.predictive.predictive_query_state.shape == (core.config.semantic_dim,)
    assert second.state.predictive.global_pred.shape == (core.config.hidden_dim,)
    loss = (
        first.state.predictive.pooled_state.pow(2).mean()
        + second.state.predictive.physical_global_pred.pow(2).mean()
        + second.state.predictive.global_pred.pow(2).mean()
    )
    core.zero_grad(set_to_none=True)
    loss.backward()
    assert core.control_state_proj.weight.grad is not None
    assert core.predictive_pool.score.weight.grad is not None


def test_variance_from_logvar_avoids_exp_backward_nan_on_saturated_entries() -> None:
    logvar = torch.tensor([1000.0, -1000.0, 0.0], dtype=torch.float32, requires_grad=True)
    weights = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    var = _variance_from_logvar(logvar, min_var=1e-4, max_var=10.0)
    loss = torch.sum(var * weights)
    loss.backward()
    assert torch.allclose(var.detach(), torch.tensor([10.0, 1e-4, 1.0], dtype=torch.float32))
    assert torch.isfinite(logvar.grad).all()
    assert float(logvar.grad[0].item()) == 0.0
    assert float(logvar.grad[1].item()) == 0.0
    assert float(logvar.grad[2].item()) > 0.0


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


def test_semantic_prefix_prompt_changes_do_not_change_physical_branch_but_do_change_control_and_future(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    semantic_a = torch.linspace(-1.0, 1.0, steps=4 * core.config.semantic_dim, dtype=torch.float32).reshape(4, core.config.semantic_dim)
    semantic_b = torch.linspace(1.0, -1.0, steps=4 * core.config.semantic_dim, dtype=torch.float32).reshape(4, core.config.semantic_dim)
    common_kwargs = dict(
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        action_future=frame.action,
    )
    first = core.step(frame, semantic_override={"tokens": semantic_a}, **common_kwargs)
    second = core.step(frame, semantic_override={"tokens": semantic_b}, **common_kwargs)
    torch.testing.assert_close(first.state.posterior.mu, second.state.posterior.mu)
    torch.testing.assert_close(first.state.posterior.Sigma, second.state.posterior.Sigma)
    torch.testing.assert_close(first.state.posterior.binding, second.state.posterior.binding)
    torch.testing.assert_close(first.state.predictive.physical_global_pred, second.state.predictive.physical_global_pred)
    torch.testing.assert_close(
        first.state.predictive.physical_prediction_cache.visual_latent,
        second.state.predictive.physical_prediction_cache.visual_latent,
    )
    torch.testing.assert_close(
        first.state.predictive.physical_prediction_cache.visual_real,
        second.state.predictive.physical_prediction_cache.visual_real,
    )
    torch.testing.assert_close(
        first.state.predictive.physical_prediction_cache.tactile_real,
        second.state.predictive.physical_prediction_cache.tactile_real,
    )
    torch.testing.assert_close(
        first.state.predictive.physical_prediction_cache.point_real,
        second.state.predictive.physical_prediction_cache.point_real,
    )
    assert not torch.allclose(first.state.predictive.control_query_state, second.state.predictive.control_query_state)
    assert not torch.allclose(first.state.predictive.global_pred, second.state.predictive.global_pred)


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
    assert second.state.predictive.control_query_state.shape == (core.config.semantic_dim,)
    assert second.state.predictive.predictive_query_state.shape == (core.config.semantic_dim,)
    assert second.state.predictive.global_pred.shape == (core.config.hidden_dim,)


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
    semantic_a = torch.linspace(-1.0, 1.0, steps=3 * core.config.semantic_dim, dtype=torch.float32).reshape(3, core.config.semantic_dim)
    semantic_b = torch.linspace(1.0, -1.0, steps=5 * core.config.semantic_dim, dtype=torch.float32).reshape(5, core.config.semantic_dim)
    first_a = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override={"tokens": semantic_a},
        action_future=frames[0].action,
    )
    first_b = core.step(
        frames[0],
        point_features_override=_point_override(core, frames[0]),
        visual_map_override=_visual_override(1.0),
        semantic_override={"tokens": semantic_b},
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

def test_semantic_tokens_directly_condition_control_and_semantic_future_readout(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    semantic_tokens_a = torch.linspace(
        -1.0,
        1.0,
        steps=4 * core.config.semantic_dim,
        dtype=torch.float32,
    ).reshape(4, core.config.semantic_dim)
    semantic_tokens_b = torch.linspace(
        1.0,
        -1.0,
        steps=4 * core.config.semantic_dim,
        dtype=torch.float32,
    ).reshape(4, core.config.semantic_dim)
    common_kwargs = dict(
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        action_future=frame.action,
    )
    first = core.step(
        frame,
        semantic_override={"tokens": semantic_tokens_a},
        **common_kwargs,
    )
    second = core.step(
        frame,
        semantic_override={"tokens": semantic_tokens_b},
        **common_kwargs,
    )
    torch.testing.assert_close(first.state.posterior.mu, second.state.posterior.mu)
    torch.testing.assert_close(first.state.predictive.physical_global_pred, second.state.predictive.physical_global_pred)
    assert not torch.allclose(first.state.predictive.control_tokens, second.state.predictive.control_tokens)
    assert not torch.allclose(first.state.predictive.control_query_state, second.state.predictive.control_query_state)
    assert not torch.allclose(first.state.predictive.predictive_query_state, second.state.predictive.predictive_query_state)
    assert not torch.allclose(first.state.predictive.pooled_state, second.state.predictive.pooled_state)
    assert not torch.allclose(first.state.predictive.global_pred, second.state.predictive.global_pred)


def test_control_prefix_explicitly_depends_on_global_post(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    observed = core.observe_step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override=_semantic_features(1.0),
    )
    shifted_posterior = dataclasses.replace(
        observed.posterior,
        global_post=observed.posterior.global_post + 0.5,
    )
    base_control = core._build_conditioned_control_state(
        observed.posterior,
        observed.innovation_token,
        observed.proprio_token,
        observed.task_readout,
    )
    shifted_control = core._build_conditioned_control_state(
        shifted_posterior,
        observed.innovation_token,
        observed.proprio_token,
        observed.task_readout,
    )
    assert not torch.allclose(base_control.tokens, shifted_control.tokens)


def test_control_and_future_trunks_consume_task_readout_and_not_raw_semantic_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    semantic_tokens = torch.linspace(-1.0, 1.0, steps=6 * core.config.semantic_dim, dtype=torch.float32).reshape(6, core.config.semantic_dim)
    captured: dict[str, torch.Tensor] = {}

    original_control_forward = core.control_world.forward
    original_predictive_forward = core.predictive_semantic_world.forward

    def _capture_control(tokens: torch.Tensor) -> torch.Tensor:
        captured["control_prefix"] = tokens.detach().clone()
        return original_control_forward(tokens)

    def _capture_predictive(tokens: torch.Tensor) -> torch.Tensor:
        captured["predictive_prefix"] = tokens.detach().clone()
        return original_predictive_forward(tokens)

    monkeypatch.setattr(core.control_world, "forward", _capture_control)
    monkeypatch.setattr(core.predictive_semantic_world, "forward", _capture_predictive)

    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        semantic_override={"tokens": semantic_tokens},
        action_future=frame.action,
    )
    control_prefix = captured["control_prefix"][0]
    predictive_prefix = captured["predictive_prefix"][0]
    expected_control_tokens = (
        core.config.persistent_anchors
        + 1
        + 1
        + 1
        + core.config.task_local_queries
        + core.config.task_global_queries
        + core.config.task_instruction_queries
        + core.config.conditioned_control_queries
    )
    expected_predictive_tokens = (
        output.state.predictive.physical_pred_tokens.shape[0]
        + core.config.conditioned_future_queries
    )
    assert control_prefix.shape[0] == expected_control_tokens
    assert predictive_prefix.shape[0] == expected_predictive_tokens
    assert control_prefix.shape[1] == core.config.semantic_dim
    assert predictive_prefix.shape[1] == core.config.semantic_dim
    torch.testing.assert_close(
        output.state.predictive.semantic_tokens,
        semantic_tokens.to(device=output.state.predictive.semantic_tokens.device),
    )


def test_semantic_prefix_projection_is_identity_in_semantic_primary_mainline(tmp_path: Path) -> None:
    core, _ = _make_core(
        tmp_path,
        hidden_dim=64,
        posterior_hidden_dim=64,
        innovation_dim=64,
        control_dim=64,
        semantic_dim=64,
        semantic_cross_dim=64,
        future_hidden_dim=64,
        attention_heads=4,
    )
    assert isinstance(core.semantic_prefix_proj, torch.nn.Identity)
    semantic_tokens = torch.randn((5, core.config.semantic_dim), dtype=torch.float32)
    context = core._project_semantic_context(tokens_raw=semantic_tokens)
    torch.testing.assert_close(context.tokens, semantic_tokens.to(device=context.tokens.device))
    torch.testing.assert_close(context.prefix_tokens, semantic_tokens.to(device=context.prefix_tokens.device))


def test_semantic_tokens_alone_can_condition_action_without_cross_reads(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.tactile = _make_tactile_packet(frame.step_id)
    semantic_tokens_a = torch.linspace(-1.5, 1.5, steps=3 * core.config.semantic_dim, dtype=torch.float32).reshape(3, core.config.semantic_dim)
    semantic_tokens_b = torch.linspace(1.5, -1.5, steps=3 * core.config.semantic_dim, dtype=torch.float32).reshape(3, core.config.semantic_dim)
    common_kwargs = dict(
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
        action_future=frame.action,
    )
    first = core.step(
        frame,
        semantic_override={"tokens": semantic_tokens_a},
        **common_kwargs,
    )
    second = core.step(
        frame,
        semantic_override={"tokens": semantic_tokens_b},
        **common_kwargs,
    )
    torch.testing.assert_close(first.state.posterior.mu, second.state.posterior.mu)
    torch.testing.assert_close(first.state.predictive.physical_global_pred, second.state.predictive.physical_global_pred)
    assert first.state.predictive.semantic_tokens.shape[0] == 3
    assert second.state.predictive.semantic_tokens.shape[0] == 3
    assert not torch.allclose(first.state.task_readout.local_tokens, second.state.task_readout.local_tokens)
    assert not torch.allclose(first.state.predictive.control_tokens, second.state.predictive.control_tokens)


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
        global_pred=torch.full_like(first.state.predictive.global_pred, 0.25),
        predictive_query_state=torch.full_like(first.state.predictive.predictive_query_state, 0.25),
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


def test_projective_compatibility_backward_is_finite(tmp_path: Path) -> None:
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
    loss = compute_alignment_loss(output.state).total
    loss.backward()
    grad_tensors = [param.grad for param in core.obs_reader.parameters() if param.grad is not None]
    assert grad_tensors
    assert all(torch.isfinite(grad).all() for grad in grad_tensors)


def test_semantic_memory_dropout_preserves_shape_and_backward(tmp_path: Path) -> None:
    core, _ = _make_core(tmp_path)
    core.train()
    semantic_tokens = torch.randn((11, core.config.semantic_dim), device=core.device, dtype=core.dtype, requires_grad=True)
    dropped = core._semantic_memory(semantic_tokens, dropout_prob=0.25)
    assert dropped.shape == semantic_tokens.shape
    loss = dropped.square().mean()
    loss.backward()
    assert semantic_tokens.grad is not None
    assert torch.isfinite(semantic_tokens.grad).all()


def test_observation_anchor_seed_indices_stay_in_range(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    output = core.step(
        frame,
        point_features_override=_point_override(core, frame),
        visual_map_override=_visual_override(1.0),
    )
    obs = output.state.observation_anchors
    valid = obs.seed_indices >= 0
    if bool(valid.any()):
        assert int(obs.seed_indices[valid].min().item()) >= 0
        assert int(obs.seed_indices[valid].max().item()) < int(output.state.token_field.point_tokens.shape[0])


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


def test_extract_targets_tactile_real_is_summary_head_not_per_sensor_reconstruction(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    frame = next(iter(replay))
    frame.G_t = core.local_frame.make_transform(frame.robot_obs)
    left_pose = np.eye(4, dtype=np.float32)
    right_pose = np.eye(4, dtype=np.float32)
    frame.tactile = PicfTactilePacket(
        sensors=(
            TactileSensorFrame(
                rgb=np.zeros((32, 32, 3), dtype=np.uint8),
                sensor_name="digit",
                T_sens_to_wrist=left_pose,
                timestamp_s=float(frame.step_id) / 30.0,
            ),
            TactileSensorFrame(
                rgb=np.full((32, 32, 3), 255, dtype=np.uint8),
                sensor_name="gelsight_mini",
                T_sens_to_wrist=right_pose,
                timestamp_s=float(frame.step_id) / 30.0,
            ),
        ),
        background_rgb_by_sensor={
            "digit": np.zeros((32, 32, 3), dtype=np.uint8),
            "gelsight_mini": np.zeros((32, 32, 3), dtype=np.uint8),
        },
    )
    targets, availability = core.extract_targets(frame, visual_map_override=_visual_override(1.0))
    tactile_real = targets["tactile_real"]
    assert tactile_real is not None
    latent_dim = core.config.tactile_latent_dim
    base_dim = core.config.tactile_map_dim
    tactile_latent = tactile_real[:latent_dim]
    tactile_base = tactile_real[latent_dim : latent_dim + base_dim]
    tactile_aux = tactile_real[latent_dim + base_dim :]
    assert tactile_latent.shape[0] == latent_dim
    assert tactile_aux.shape[0] == core.config.tactile_aux_dim
    torch.testing.assert_close(tactile_base, torch.full_like(tactile_base, 0.5), atol=1e-6, rtol=0.0)
    assert float(tactile_aux[4].item()) == pytest.approx(0.5, abs=1e-6)
    assert bool(availability[2].item()) is True


def test_recurrent_carry_matches_full_state_next_step_outputs(tmp_path: Path) -> None:
    core, replay = _make_core(tmp_path)
    core.eval()
    iterator = iter(replay)
    first = next(iterator)
    second = next(iterator)
    first.tactile = _make_tactile_packet(first.step_id, contact_shift=8)
    second.tactile = _make_tactile_packet(second.step_id, contact_shift=12)
    with torch.no_grad():
        first_output = core.step(
            first,
            point_features_override=_point_override(core, first),
            visual_map_override=_visual_override(1.0),
            semantic_override=_semantic_features(1.0),
        )
        recurrent_carry = core.make_recurrent_carry(first_output.state)
        second_full = core.step(
            second,
            previous=first_output.state,
            point_features_override=_point_override(core, second),
            visual_map_override=_visual_override(1.5),
            semantic_override=_semantic_features(1.5),
        )
        second_carry = core.step(
            second,
            previous=recurrent_carry,
            point_features_override=_point_override(core, second),
            visual_map_override=_visual_override(1.5),
            semantic_override=_semantic_features(1.5),
        )

    torch.testing.assert_close(second_carry.state.posterior.h, second_full.state.posterior.h)
    torch.testing.assert_close(second_carry.state.posterior.mu, second_full.state.posterior.mu)
    torch.testing.assert_close(second_carry.state.posterior.global_post, second_full.state.posterior.global_post)
    torch.testing.assert_close(second_carry.state.conditioned_control.pi_prefix_tokens, second_full.state.conditioned_control.pi_prefix_tokens)
    torch.testing.assert_close(second_carry.state.predictive.physical_global_pred, second_full.state.predictive.physical_global_pred)
    torch.testing.assert_close(second_carry.state.predictive.global_pred, second_full.state.predictive.global_pred)

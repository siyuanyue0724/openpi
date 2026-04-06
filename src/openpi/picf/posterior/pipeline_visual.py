from __future__ import annotations

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import build_point_frame_context
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PosteriorDebugMetrics
from openpi.picf.posterior.contracts import PosteriorState
from openpi.picf.posterior.fusion_visual import fuse_point_visual
from openpi.picf.posterior.point_expert import build_point_expert
from openpi.picf.posterior.point_expert import empty_point_expert
from openpi.picf.posterior.prior import build_current_prior
from openpi.picf.posterior.visual_expert import build_visual_expert
from openpi.picf.posterior.visual_expert import empty_visual_expert
from openpi.picf.posterior.visual_expert import load_camera_model
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.history import VisualClipBuffer
from openpi.picf.vjepa.wrapper import Vjepa2VisualEncoder


class PointVisualPosteriorPipeline:
    def __init__(
        self,
        *,
        visual_config: VjepaVisualConfig,
        posterior_config: PosteriorConfig | None = None,
        scaffold_config: DeterministicScaffoldConfig | None = None,
        local_frame: EndEffectorLocalFrame | None = None,
        visual_encoder: Vjepa2VisualEncoder | None = None,
        enable_point_expert: bool = True,
        enable_visual_expert: bool = True,
    ):
        if not enable_point_expert and not enable_visual_expert:
            raise ValueError("At least one expert must be enabled.")
        self.visual_config = visual_config
        self.posterior_config = posterior_config or PosteriorConfig()
        self.scaffold_config = scaffold_config or DeterministicScaffoldConfig()
        self.local_frame = local_frame or EndEffectorLocalFrame()
        self.enable_point_expert = bool(enable_point_expert)
        self.enable_visual_expert = bool(enable_visual_expert)
        if self.enable_visual_expert:
            if visual_config.camera_json_path is None:
                raise ValueError("visual_config.camera_json_path must be set when the visual expert is enabled.")
            self.visual_encoder = visual_encoder or Vjepa2VisualEncoder(visual_config)
            self.camera_model = load_camera_model(
                visual_config.camera_json_path,
                camera_name=visual_config.camera_name,
            )
            self.clip_buffer = VisualClipBuffer(num_frames=visual_config.num_frames)
        else:
            self.visual_encoder = None
            self.camera_model = None
            self.clip_buffer = None

    def step(
        self,
        observation: PicfObservation,
        scaffold_state: SupportScaffoldState,
        previous: PosteriorState | None = None,
    ) -> PosteriorState:
        if observation.G_t is None:
            observation.G_t = scaffold_state.G_t

        frame_context = None
        if scaffold_state.debug.fresh_scaffold and (self.enable_point_expert or self.enable_visual_expert):
            frame_context = build_point_frame_context(
                observation,
                crop_radius_m=self.scaffold_config.crop_radius_m,
                local_frame=self.local_frame,
            )

        mu_prop, var_prop_block, matched_prior_count, reset_prior_count = build_current_prior(
            config=self.posterior_config,
            matched_mask=scaffold_state.matched_mask,
            pred_idx=scaffold_state.pred_idx,
            previous=previous,
        )

        if self.enable_point_expert and frame_context is not None:
            point = build_point_expert(
                posterior_config=self.posterior_config,
                scaffold_config=self.scaffold_config,
                scaffold_state=scaffold_state,
                frame_context=frame_context,
            )
        else:
            point = empty_point_expert(posterior_config=self.posterior_config, k_support=scaffold_state.x.shape[0])

        if self.enable_visual_expert:
            assert self.clip_buffer is not None
            assert self.camera_model is not None
            assert self.visual_encoder is not None
            self.clip_buffer.push(
                observation.rgb_static,
                segment_id=int(observation.segment_id),
                reset=bool(observation.reset_scaffold),
            )
            clip = self.clip_buffer.get_clip()
            visual_features = self.visual_encoder.encode_clip(clip)
            visual = build_visual_expert(
                posterior_config=self.posterior_config,
                visual_config=self.visual_config,
                observation=observation,
                scaffold_state=scaffold_state,
                visual_features=visual_features,
                camera_model=self.camera_model,
                frame_context=frame_context,
            )
        else:
            visual = empty_visual_expert(posterior_config=self.posterior_config, k_support=scaffold_state.x.shape[0])

        mu, var_block, precision_gain_count, point_gain_count, visual_gain_count = fuse_point_visual(
            config=self.posterior_config,
            mu_prop=mu_prop,
            var_prop_block=var_prop_block,
            point=point,
            visual=visual,
        )
        stale_error = 0.0
        posterior_equals_prior = True
        if not scaffold_state.debug.fresh_scaffold:
            stale_error = float(np.max(np.abs(mu - mu_prop))) if mu.size > 0 else 0.0
            posterior_equals_prior = bool(np.allclose(mu, mu_prop) and np.allclose(var_block, var_prop_block))
        nan_count = int(np.isnan(mu).sum() + np.isnan(var_block).sum())
        debug = PosteriorDebugMetrics(
            point_gate_ratio=float(point.gate.mean()) if point.gate.size > 0 else 0.0,
            stale_prior_match_error=stale_error,
            posterior_prior_equal_on_stale=posterior_equals_prior,
            matched_prior_count=matched_prior_count,
            reset_prior_count=reset_prior_count,
            precision_gain_count=precision_gain_count,
            nan_count=nan_count,
            max_abs_mu=float(np.max(np.abs(mu))) if mu.size > 0 else 0.0,
            min_var_block=float(np.min(var_block)) if var_block.size > 0 else 0.0,
            max_var_block=float(np.max(var_block)) if var_block.size > 0 else 0.0,
            visual_gate_ratio=float(visual.gate.mean()) if visual.gate.size > 0 else 0.0,
            point_precision_gain_count=point_gain_count,
            visual_precision_gain_count=visual_gain_count,
        )
        return PosteriorState(
            mu=mu,
            var_block=var_block,
            mu_prop=mu_prop,
            var_prop_block=var_prop_block,
            point=point,
            step_id=int(scaffold_state.step_id),
            segment_id=int(scaffold_state.segment_id),
            debug=debug,
            visual=visual,
        )

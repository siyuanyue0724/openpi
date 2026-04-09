from __future__ import annotations

import dataclasses

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import build_point_frame_context
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PosteriorDebugMetrics
from openpi.picf.posterior.contracts import PosteriorState
from openpi.picf.posterior.fusion import fuse_point_only
from openpi.picf.posterior.point_expert import build_point_expert
from openpi.picf.posterior.point_expert import empty_point_expert
from openpi.picf.posterior.prior import build_current_prior
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor


class PointOnlyPosteriorPipeline:
    def __init__(
        self,
        *,
        posterior_config: PosteriorConfig | None = None,
        scaffold_config: DeterministicScaffoldConfig | None = None,
        local_frame: EndEffectorLocalFrame | None = None,
        point_feature_extractor: SonataPointFeatureExtractor | None = None,
    ):
        self.posterior_config = posterior_config or PosteriorConfig()
        self.scaffold_config = scaffold_config or DeterministicScaffoldConfig()
        self.local_frame = local_frame or EndEffectorLocalFrame()
        self.point_feature_extractor = point_feature_extractor

    def step(
        self,
        observation: PicfObservation,
        scaffold_state: SupportScaffoldState,
        previous: PosteriorState | None = None,
    ) -> PosteriorState:
        if observation.G_t is None:
            observation.G_t = scaffold_state.G_t
        frame_context = None
        if scaffold_state.debug.fresh_scaffold:
            frame_context = build_point_frame_context(
                observation,
                crop_radius_m=self.scaffold_config.crop_radius_m,
                local_frame=self.local_frame,
            )
        point_features = None
        if frame_context is not None and self.point_feature_extractor is not None:
            feature_context = frame_context
            if not scaffold_state.runtime_meta.v_rgb_p:
                feature_context = dataclasses.replace(frame_context, colors=np.zeros_like(frame_context.colors, dtype=np.float32))
            point_features = self.point_feature_extractor.encode_local_context(feature_context).features
        mu_prop, var_prop_block, matched_prior_count, reset_prior_count = build_current_prior(
            config=self.posterior_config,
            matched_mask=scaffold_state.matched_mask,
            pred_idx=scaffold_state.pred_idx,
            previous=previous,
        )
        if frame_context is not None:
            point = build_point_expert(
                posterior_config=self.posterior_config,
                scaffold_state=scaffold_state,
                frame_context=frame_context,
                point_features=point_features,
            )
        else:
            point = empty_point_expert(posterior_config=self.posterior_config, k_support=scaffold_state.x.shape[0])
        mu, var_block, precision_gain_count = fuse_point_only(
            config=self.posterior_config,
            mu_prop=mu_prop,
            var_prop_block=var_prop_block,
            point=point,
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
            fresh_scaffold=bool(scaffold_state.debug.fresh_scaffold),
            matched_prior_count=matched_prior_count,
            reset_prior_count=reset_prior_count,
            precision_gain_count=precision_gain_count,
            nan_count=nan_count,
            max_abs_mu=float(np.max(np.abs(mu))) if mu.size > 0 else 0.0,
            min_var_block=float(np.min(var_block)) if var_block.size > 0 else 0.0,
            max_var_block=float(np.max(var_block)) if var_block.size > 0 else 0.0,
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
        )

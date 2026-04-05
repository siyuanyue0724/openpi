import dataclasses
from pathlib import Path

import numpy as np

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline import PointOnlyPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.test_utils import build_mini_calvin_dataset


def test_posterior_pipeline_fresh_and_stale(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frames = list(replay)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    scaffold = DeterministicScaffoldPipeline(builder)
    posterior = PointOnlyPosteriorPipeline()

    scaffold_state_0 = scaffold.step(frames[0])
    posterior_state_0 = posterior.step(frames[0], scaffold_state_0)
    assert posterior_state_0.debug.point_gate_ratio > 0
    assert posterior_state_0.debug.nan_count == 0

    scaffold_state_1 = scaffold.step(frames[1], scaffold_state_0)
    posterior_state_1 = posterior.step(frames[1], scaffold_state_1, posterior_state_0)
    assert posterior_state_1.debug.matched_prior_count >= 0

    stale_frame = dataclasses.replace(frames[2], point_set=None, runtime_meta=None)
    stale_frame.depth_static = np.zeros_like(stale_frame.depth_static, dtype=np.float32)
    scaffold_state_stale = scaffold.step(stale_frame, scaffold_state_1)
    posterior_state_stale = posterior.step(stale_frame, scaffold_state_stale, posterior_state_1)
    assert posterior_state_stale.debug.posterior_prior_equal_on_stale
    assert np.allclose(posterior_state_stale.mu, posterior_state_stale.mu_prop)

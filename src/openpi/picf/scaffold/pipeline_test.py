import dataclasses
from pathlib import Path

import numpy as np

from openpi.picf.contracts import RuntimeMeta
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.test_utils import build_mini_calvin_dataset


class _CapturingExtractor:
    def __init__(self) -> None:
        self.captured_colors: np.ndarray | None = None

    def encode_local_context(self, frame_context):
        self.captured_colors = np.asarray(frame_context.colors, dtype=np.float32).copy()
        return type("Features", (), {"features": self.captured_colors})()


def test_scaffold_pipeline_runs_fresh_and_stale(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frames = list(replay)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=2, max_points=256)
    pipeline = DeterministicScaffoldPipeline(builder)

    state0 = pipeline.step(frames[0])
    assert state0.debug.fresh_scaffold
    assert state0.debug.num_active > 0
    assert np.all(np.isfinite(state0.x))
    assert np.all((state0.r[state0.active_mask] >= 0.003) & (state0.r[state0.active_mask] <= 0.02))

    state1 = pipeline.step(frames[1], state0)
    assert state1.debug.fresh_scaffold
    assert np.all(state1.pred_idx[state1.matched_mask] >= 0)

    stale_frame = dataclasses.replace(frames[2], point_set=None, runtime_meta=None)
    stale_frame.depth_static = np.zeros_like(stale_frame.depth_static, dtype=np.float32)
    stale_state = pipeline.step(stale_frame, state1)
    assert not stale_state.debug.fresh_scaffold
    assert stale_state.debug.num_birth == 0
    assert stale_state.matched_mask.sum() == stale_state.active_mask.sum()


def test_scaffold_pi_rows_and_stale_timeout(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frames = list(replay)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=2, max_points=256)
    pipeline = DeterministicScaffoldPipeline(builder)

    state = pipeline.step(frames[0])
    populated = np.sum(state.pi_geom, axis=1) > 0
    assert np.allclose(state.pi_geom[populated].sum(axis=1), 1.0, atol=1e-5)

    stale_frame_1 = dataclasses.replace(frames[1], point_set=None, runtime_meta=None)
    stale_frame_1.depth_static = np.zeros_like(stale_frame_1.depth_static, dtype=np.float32)
    stale_state_1 = pipeline.step(stale_frame_1, state)
    assert not stale_state_1.debug.hold_triggered

    stale_frame_2 = dataclasses.replace(frames[2], point_set=None, runtime_meta=None)
    stale_frame_2.depth_static = np.zeros_like(stale_frame_2.depth_static, dtype=np.float32)
    stale_state_2 = pipeline.step(stale_frame_2, stale_state_1)
    assert stale_state_2.debug.hold_triggered
    assert stale_state_2.debug.hold_reason == "scaffold_stale_timeout"


def test_scaffold_respects_observation_runtime_meta(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frame = next(iter(replay))
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=128)
    pipeline = DeterministicScaffoldPipeline(builder)
    runtime_meta = RuntimeMeta(
        t_v_last=12.0,
        t_p_last=float(frame.timestamp_s),
        t_t_last=0.0,
        t_rgb_last=float(frame.timestamp_s),
        b_rgb_avail=False,
        rgb_proj_residual=0.25,
        n_vis_upd=7,
        v_rgb_p=False,
        v_pc_scaf=True,
        stale_scaffold_steps=0,
    )
    frame.runtime_meta = runtime_meta
    state = pipeline.step(frame)
    assert state.runtime_meta.t_v_last == 12.0
    assert state.runtime_meta.n_vis_upd == 7
    assert not state.runtime_meta.v_rgb_p


def test_scaffold_zeroes_point_rgb_when_runtime_gate_is_off(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frame = next(iter(replay))
    extractor = _CapturingExtractor()
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=128)
    pipeline = DeterministicScaffoldPipeline(builder, point_feature_extractor=extractor)
    frame.runtime_meta = RuntimeMeta(
        t_v_last=float(frame.timestamp_s),
        t_p_last=float(frame.timestamp_s),
        t_t_last=0.0,
        t_rgb_last=float(frame.timestamp_s),
        b_rgb_avail=False,
        rgb_proj_residual=1.0,
        n_vis_upd=1,
        v_rgb_p=False,
        v_pc_scaf=True,
        stale_scaffold_steps=0,
    )

    pipeline.step(frame)

    assert extractor.captured_colors is not None
    assert np.allclose(extractor.captured_colors, 0.0)

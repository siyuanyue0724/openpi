import dataclasses
from pathlib import Path

import numpy as np

from openpi.picf.contracts import PicfPointCloudFrame
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline_visual import PointVisualPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.test_utils import build_mini_calvin_dataset
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.wrapper import VjepaFeatureMap


class _StubVisualEncoder:
    def __init__(self) -> None:
        self.checkpoint_loaded = False

    def encode_clip(self, clip: np.ndarray) -> VjepaFeatureMap:
        return VjepaFeatureMap(
            tokens_thwc=np.ones((2, 4, 4, 8), dtype=np.float32),
            source_hw=(clip.shape[1], clip.shape[2]),
            resized_hw=(64, 64),
            checkpoint_loaded=False,
            model_name="stub",
        )


def _invalid_point_frame() -> PicfPointCloudFrame:
    return PicfPointCloudFrame(
        grid_coord=np.zeros((0, 3), dtype=np.int32),
        xyz_world=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.float32),
        normal_world=np.zeros((0, 3), dtype=np.float32),
        valid_point_mask=np.zeros((0,), dtype=bool),
        frame_valid=False,
    )


def test_point_visual_pipeline_supports_stale_visual_fallback(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frames = list(replay)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    scaffold = DeterministicScaffoldPipeline(builder)
    posterior = PointVisualPosteriorPipeline(
        visual_config=VjepaVisualConfig(
            camera_json_path=calvin_root,
            arch_name_override="vit_tiny",
            img_size=64,
            num_frames=4,
            device="cpu",
            dtype="float32",
        ),
        visual_encoder=_StubVisualEncoder(),
    )

    scaffold_state_0 = scaffold.step(frames[0])
    posterior_state_0 = posterior.step(frames[0], scaffold_state_0)
    assert posterior_state_0.debug.point_gate_ratio > 0.0
    assert posterior_state_0.debug.visual_gate_ratio > 0.0

    scaffold_state_1 = scaffold.step(frames[1], scaffold_state_0)
    posterior_state_1 = posterior.step(frames[1], scaffold_state_1, posterior_state_0)
    assert posterior_state_1.debug.visual_precision_gain_count > 0

    stale_frame = dataclasses.replace(frames[2], point_set=_invalid_point_frame(), runtime_meta=None)
    scaffold_state_stale = scaffold.step(stale_frame, scaffold_state_1)
    posterior_state_stale = posterior.step(stale_frame, scaffold_state_stale, posterior_state_1)
    assert not scaffold_state_stale.debug.fresh_scaffold
    assert posterior_state_stale.debug.visual_precision_gain_count > 0
    assert not posterior_state_stale.debug.posterior_prior_equal_on_stale

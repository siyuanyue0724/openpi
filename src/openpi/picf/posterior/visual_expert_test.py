from pathlib import Path

import numpy as np

from openpi.picf.frame_context import build_point_frame_context
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.visual_expert import build_visual_expert
from openpi.picf.posterior.visual_expert import load_camera_model
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.test_utils import build_mini_calvin_dataset
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.wrapper import VjepaFeatureMap


def test_visual_expert_gates_visible_supports_on_fresh_scaffold(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=256)
    scaffold = DeterministicScaffoldPipeline(builder)
    replay = CalvinSequentialReplay(calvin_root, backend="dir", segment_indices=[0])
    frame = next(iter(replay))
    scaffold_state = scaffold.step(frame)
    frame_context = build_point_frame_context(frame, crop_radius_m=scaffold.config.crop_radius_m)
    visual_output = VjepaFeatureMap(
        tokens_thwc=np.ones((2, 4, 4, 8), dtype=np.float32),
        source_hw=(frame.rgb_static.shape[0], frame.rgb_static.shape[1]),
        resized_hw=(64, 64),
        checkpoint_loaded=False,
        model_name="stub",
    )
    visual_state = build_visual_expert(
        posterior_config=PosteriorConfig(),
        visual_config=VjepaVisualConfig(camera_json_path=calvin_root, img_size=64, num_frames=4),
        observation=frame,
        scaffold_state=scaffold_state,
        visual_features=visual_output,
        camera_model=load_camera_model(calvin_root),
        frame_context=frame_context,
    )

    assert int(np.sum(visual_state.gate)) > 0
    assert np.all(np.isfinite(visual_state.mu))
    assert np.all(visual_state.visibility >= 0.0)
    replay.close()

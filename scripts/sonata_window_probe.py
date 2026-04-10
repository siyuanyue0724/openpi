from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _coerce_bool
from scripts.picf_core_train import _default_sonata_checkpoint
from scripts.picf_core_train import _normalize_train_args
from scripts.picf_core_train import _seed_everything
from scripts.picf_core_train import _validate_train_args
from openpi.picf.core.config import PicfCoreConfig
from openpi.picf.core.pipeline import _build_identity_frame_context
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame
from openpi.picf.sonata.config import SonataPointConfig
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor
from openpi.picf.sonata.wrapper import _normalize_colors
from openpi.picf.sonata.wrapper import _normalize_local_grid_coords


def _load_args(args_json: Path, *, device: str | None) -> argparse.Namespace:
    payload = json.loads(args_json.read_text(encoding="utf-8"))
    args = argparse.Namespace(**payload)
    if device is not None:
        args.device = str(device)
    tactile_names = getattr(args, "tactile_sensor_names", ("digit", "gelsight_mini"))
    if isinstance(tactile_names, str):
        tactile_names = tuple(part.strip() for part in tactile_names.split(",") if part.strip())
    else:
        tactile_names = tuple(str(name) for name in tactile_names)
    tactile_offsets = getattr(args, "tactile_sensor_offsets_m", ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)))
    if isinstance(tactile_offsets, str):
        blocks = [block.strip() for block in tactile_offsets.split(";") if block.strip()]
        tactile_offsets = tuple(tuple(float(value.strip()) for value in block.split(",") if value.strip()) for block in blocks)
    else:
        tactile_offsets = tuple(tuple(float(value) for value in offset) for offset in tactile_offsets)
    args.tactile_sensor_names = tactile_names
    args.tactile_sensor_offsets_m = tactile_offsets
    args.sonata_checkpoint_path = getattr(args, "sonata_checkpoint_path", None) or _default_sonata_checkpoint()
    _normalize_train_args(args)
    _validate_train_args(args)
    return args


def _parse_flat_indices(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one flat index.")
    return [int(value) for value in values]


def _identity_grid_build_sample(self: SonataPointFeatureExtractor, frame_context):
    coord = np.asarray(frame_context.points_local, dtype=np.float32)
    grid_coord = np.asarray(frame_context.grid_coord, dtype=np.int32)
    color = _normalize_colors(frame_context.colors)
    normal = np.asarray(frame_context.normals_local, dtype=np.float32)
    if coord.shape != grid_coord.shape:
        raise RuntimeError(
            "PICF Sonata sample contract violated: "
            f"coord.shape={coord.shape} != grid_coord.shape={grid_coord.shape}"
        )
    if normal.shape != coord.shape or color.shape != coord.shape:
        raise RuntimeError(
            "PICF Sonata sample contract violated: "
            f"coord.shape={coord.shape}, normal.shape={normal.shape}, color.shape={color.shape}"
        )
    feat = np.concatenate([coord, color], axis=1).astype(np.float32)
    in_channels = int(self.model.embedding.in_channels)
    if in_channels != 6:
        raise RuntimeError(f"Expected model.embedding.in_channels == 6, got {in_channels}.")
    n_points = int(coord.shape[0])
    return {
        "coord": torch.from_numpy(coord).to(device=self.device, dtype=torch.float32),
        "grid_coord": torch.from_numpy(grid_coord).to(device=self.device, dtype=torch.int32),
        "color": torch.from_numpy(color).to(device=self.device, dtype=torch.float32),
        "normal": torch.from_numpy(normal).to(device=self.device, dtype=torch.float32),
        "feat": torch.from_numpy(feat).to(device=self.device, dtype=torch.float32),
        "grid_size": float(self.config.voxel_size_m),
        "batch": torch.zeros((n_points,), device=self.device, dtype=torch.int64),
        "offset": torch.tensor([n_points], device=self.device, dtype=torch.int64),
    }


def _rebased_grid_build_sample(self: SonataPointFeatureExtractor, frame_context):
    coord = np.asarray(frame_context.points_local, dtype=np.float32)
    grid_coord = _normalize_local_grid_coords(frame_context.grid_coord)
    color = _normalize_colors(frame_context.colors)
    normal = np.asarray(frame_context.normals_local, dtype=np.float32)
    if coord.shape != grid_coord.shape:
        raise RuntimeError(
            "PICF Sonata sample contract violated: "
            f"coord.shape={coord.shape} != grid_coord.shape={grid_coord.shape}"
        )
    if normal.shape != coord.shape or color.shape != coord.shape:
        raise RuntimeError(
            "PICF Sonata sample contract violated: "
            f"coord.shape={coord.shape}, normal.shape={normal.shape}, color.shape={color.shape}"
        )
    feat = np.concatenate([coord, color], axis=1).astype(np.float32)
    in_channels = int(self.model.embedding.in_channels)
    if in_channels != 6:
        raise RuntimeError(f"Expected model.embedding.in_channels == 6, got {in_channels}.")
    n_points = int(coord.shape[0])
    return {
        "coord": torch.from_numpy(coord).to(device=self.device, dtype=torch.float32),
        "grid_coord": torch.from_numpy(grid_coord).to(device=self.device, dtype=torch.int32),
        "color": torch.from_numpy(color).to(device=self.device, dtype=torch.float32),
        "normal": torch.from_numpy(normal).to(device=self.device, dtype=torch.float32),
        "feat": torch.from_numpy(feat).to(device=self.device, dtype=torch.float32),
        "grid_size": float(self.config.voxel_size_m),
        "batch": torch.zeros((n_points,), device=self.device, dtype=torch.int64),
        "offset": torch.tensor([n_points], device=self.device, dtype=torch.int64),
    }


@contextlib.contextmanager
def _override_build_sample(mode: str) -> Callable[[], None]:
    original = SonataPointFeatureExtractor._build_sample
    try:
        if mode == "original":
            SonataPointFeatureExtractor._build_sample = _identity_grid_build_sample
        elif mode == "rebased":
            SonataPointFeatureExtractor._build_sample = _rebased_grid_build_sample
        else:
            raise ValueError(f"Unsupported grid mode: {mode}")
        yield
    finally:
        SonataPointFeatureExtractor._build_sample = original


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay real CALVIN windows through the standalone Sonata point backbone.")
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--flat-indices", required=True, help="Comma-separated flat indices from PICF training logs.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-mode", choices=("original", "rebased"), default="rebased")
    parser.add_argument("--optimizer-step", type=_coerce_bool, default=True)
    parser.add_argument("--rank-seed", type=int, default=1)
    args = parser.parse_args()

    train_args = _load_args(Path(args.args_json), device=str(args.device))
    flat_indices = _parse_flat_indices(args.flat_indices)
    device = torch.device(str(train_args.device))
    if device.type != "cuda":
        raise ValueError("sonata_window_probe requires CUDA.")

    _seed_everything(int(train_args.seed), int(args.rank_seed))

    source = _CalvinTransitionSource(
        train_args.calvin_root,
        split=train_args.split,
        backend=train_args.backend,
        unroll_steps=train_args.unroll_steps,
        use_tactile=bool(train_args.use_tactile),
        tactile_sensor_names=train_args.tactile_sensor_names,
        tactile_sensor_offsets_m=train_args.tactile_sensor_offsets_m,
    )
    builder = CalvinDepthToPicfPointCloud(train_args.calvin_root, stride=train_args.stride, max_points=train_args.max_points)
    crop_radius_m = float(PicfCoreConfig().crop_radius_m)
    local_frame = EndEffectorLocalFrame()

    with _override_build_sample(str(args.grid_mode)):
        extractor = SonataPointFeatureExtractor(
            SonataPointConfig(
                checkpoint_path=train_args.sonata_checkpoint_path,
                stage_name=train_args.sonata_stage_name,
                device=str(device),
                dtype=train_args.sonata_dtype,
                trainable=bool(train_args.point_backbone_trainable),
                allow_random_init=False,
            )
        )
        extractor.train()
        optimizer = torch.optim.AdamW(extractor.parameters(), lr=5e-5)

        try:
            step = 0
            for repeat_idx in range(int(args.repeat)):
                for flat_index in flat_indices:
                    window = source.window(flat_index)
                    for frame_offset, observation in enumerate(window.frames[:-1]):
                        step += 1
                        if observation.G_t is None:
                            observation.G_t = local_frame.make_transform(observation.robot_obs)
                        if observation.point_set is None:
                            observation.point_set = builder(
                                {
                                    "rgb_static": observation.rgb_static,
                                    "depth_static": observation.depth_static,
                                    "focus_center_world": np.asarray(observation.G_t[:3, 3], dtype=np.float32),
                                    "focus_radius_m": crop_radius_m,
                                }
                            )
                        frame_context = _build_identity_frame_context(
                            observation,
                            crop_radius_m=crop_radius_m,
                            focus_center_world=np.asarray(observation.G_t[:3, 3], dtype=np.float32),
                        )
                        raw_grid = np.asarray(frame_context.grid_coord, dtype=np.int32)
                        optimizer.zero_grad(set_to_none=True)
                        encoded = extractor.encode_local_context(frame_context)
                        feat = encoded.features if isinstance(encoded.features, torch.Tensor) else torch.from_numpy(encoded.features).to(device=device)
                        loss = feat.float().square().mean()
                        loss.backward()
                        if bool(args.optimizer_step):
                            optimizer.step()
                        torch.cuda.synchronize(device=device)
                        print(
                            json.dumps(
                                {
                                    "mode": str(args.grid_mode),
                                    "repeat": int(repeat_idx),
                                    "step": int(step),
                                    "flat_index": int(flat_index),
                                    "segment": int(window.segment_id),
                                    "start_step": int(window.start_step_id),
                                    "prompt": str(window.prompt),
                                    "frame_offset": int(frame_offset),
                                    "n_points": int(raw_grid.shape[0]),
                                    "grid_min": raw_grid.min(axis=0).tolist() if raw_grid.size else [0, 0, 0],
                                    "grid_max": raw_grid.max(axis=0).tolist() if raw_grid.size else [0, 0, 0],
                                    "grid_span": (raw_grid.max(axis=0) - raw_grid.min(axis=0)).tolist() if raw_grid.size else [0, 0, 0],
                                    "loss": float(loss.detach().item()),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        finally:
            source.close()


if __name__ == "__main__":
    main()

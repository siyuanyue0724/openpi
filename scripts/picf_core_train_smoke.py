from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from openpi.picf.core import PicfCoreConfig
from openpi.picf.core import PicfFullCore
from openpi.picf.core import compute_transition_loss
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.vjepa.config import VjepaVisualConfig

CUDA_RUNTIME_AT_IMPORT = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


class _NullTactileEncoder:
    def encode_sensor_clips(self, *, clips_by_sensor, backgrounds_by_sensor, poses_by_sensor):
        del clips_by_sensor, backgrounds_by_sensor, poses_by_sensor
        return None


class _NullVisualEncoder:
    def encode_clip(self, _clip):
        raise AssertionError("visual_map_override should bypass encoder use in this smoke")


def _rgb_visual_override(rgb: np.ndarray, grid: int = 8) -> torch.Tensor:
    rgb_t = torch.as_tensor(np.asarray(rgb, dtype=np.float32) / 255.0, dtype=torch.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(rgb_t.permute(2, 0, 1)[None, :], (grid, grid))[0]
    return pooled.permute(1, 2, 0).contiguous()


def run_smoke(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    segment_index: int,
    stride: int,
    max_points: int,
    device: str,
    lr: float,
) -> dict[str, float | int | str | bool]:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    replay = CalvinSequentialReplay(calvin_root, split=split, backend=backend, segment_indices=[segment_index])
    frames = list(replay)
    replay.close()
    if len(frames) < 2:
        raise RuntimeError("Need at least two frames in the selected CALVIN segment for transition training smoke.")

    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    config = PicfCoreConfig(
        device=device,
        hidden_dim=64,
        posterior_hidden_dim=64,
        latent_dim=24,
        innovation_dim=64,
        control_dim=64,
        semantic_dim=32,
        future_hidden_dim=64,
        persistent_anchors=8,
        observation_anchors=10,
        fusion_layers=2,
        posterior_layers=1,
        predictive_layers=1,
        control_layers=1,
        attention_heads=4,
        future_vote_heads=3,
    )
    core = PicfFullCore(
        builder,
        config=config,
        visual_config=VjepaVisualConfig(
            camera_json_path=calvin_root,
            arch_name_override="vit_tiny",
            img_size=64,
            num_frames=4,
            device=device,
            dtype="float32",
        ),
        visual_encoder=_NullVisualEncoder(),
        tactile_encoder=_NullTactileEncoder(),
    )
    optimizer = torch.optim.AdamW(core.parameters(), lr=lr)

    current = frames[0]
    nxt = frames[1]
    current_visual = _rgb_visual_override(current.rgb_static)
    next_visual = _rgb_visual_override(nxt.rgb_static)
    semantic = np.zeros((16,), dtype=np.float32)

    output = core.step(
        current,
        visual_map_override=current_visual,
        semantic_override=semantic,
        action_future=current.action,
    )
    optimizer.zero_grad(set_to_none=True)
    losses = compute_transition_loss(
        core,
        output,
        nxt,
        action_target=current.action,
        next_visual_map_override=next_visual,
    )
    losses.total.backward()
    action_grad_norm = float(core.action_head.weight.grad.norm().item()) if core.action_head.weight.grad is not None else 0.0
    point_grad_norm = float(core.point_real_head.weight.grad.norm().item()) if core.point_real_head.weight.grad is not None else 0.0
    optimizer.step()

    runtime_cuda_available = CUDA_RUNTIME_AT_IMPORT
    using_cuda = bool(runtime_cuda_available and str(core.device).startswith("cuda"))
    device_name = None
    if using_cuda:
        device_name = torch.cuda.get_device_name(core.device)
    return {
        "device": str(core.device),
        "device_name": device_name,
        # Keep the legacy field name, but make the value reflect the runtime
        # execution device for this smoke instead of ambient system capability.
        "cuda_available": using_cuda,
        "cuda_runtime_available": runtime_cuda_available,
        "segment_index": int(segment_index),
        "current_step_id": int(current.step_id),
        "next_step_id": int(nxt.step_id),
        "loss_total": float(losses.total.item()),
        "loss_action": float(losses.action.item()),
        "loss_visual_latent": float(losses.visual_latent.item()),
        "loss_visual_real": float(losses.visual_real.item()),
        "loss_tactile_real": float(losses.tactile_real.item()),
        "loss_point_real": float(losses.point_real.item()),
        "loss_alignment": float(losses.alignment.item()),
        "loss_anchor_pv": float(losses.anchor_pv.item()),
        "loss_pv_weak": float(losses.pv_weak.item()),
        "loss_focus_pv": float(losses.focus_pv.item()),
        "loss_pt": float(losses.pt.item()),
        "availability_visual_latent": float(losses.availability[0].item()),
        "availability_visual_real": float(losses.availability[1].item()),
        "availability_tactile_real": float(losses.availability[2].item()),
        "availability_point_real": float(losses.availability[3].item()),
        "action_grad_norm": action_grad_norm,
        "point_grad_norm": point_grad_norm,
        "projective_candidate_edges": float(output.debug.get("projective_candidate_edges", 0.0)),
        "projective_candidate_density": float(output.debug.get("projective_candidate_density", 0.0)),
        "mean_point_visibility": float(output.debug.get("mean_point_visibility", 0.0)),
        "mean_point_route_gate": float(output.debug.get("mean_point_route_gate", 0.0)),
        "mean_visual_route_gate": float(output.debug.get("mean_visual_route_gate", 0.0)),
        "mean_point_route_support": float(output.debug.get("mean_point_route_support", 0.0)),
        "mean_visual_route_support": float(output.debug.get("mean_visual_route_support", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step PICF core training smoke on sequential CALVIN frames.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="dir", choices=["dir", "zip"])
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    summary = run_smoke(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        segment_index=args.segment_index,
        stride=args.stride,
        max_points=args.max_points,
        device=args.device,
        lr=args.lr,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

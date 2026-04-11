from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.core import PicfCoreConfig
from openpi.picf.core import PicfFullCore
from openpi.picf.core import compute_transition_loss
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.sonata.config import SonataPointConfig
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor
from openpi.picf.vjepa.config import VjepaVisualConfig
from picf_core_train import _default_tactile_backgrounds_path
from picf_core_train import _default_tactile_calibration_path
from picf_core_train import _default_tactile_contact_stats_path
from picf_core_train import _load_tactile_backgrounds_npz
from picf_core_train import _load_tactile_contact_stats_json

CUDA_RUNTIME_AT_IMPORT = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
_DEFAULT_TACTILE_SENSOR_NAMES = ("digit", "gelsight_mini")
_DEFAULT_TACTILE_SENSOR_OFFSETS_M = ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0))


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


def _default_vjepa_checkpoint(model_name: str) -> str | None:
    filename_by_model = {
        "vjepa2_1_vit_base_384": "vjepa2_1_vitb_dist_vitG_384.pt",
        "vjepa2_1_vit_large_384": "vjepa2_1_vitl_dist_vitG_384.pt",
        "vjepa2_1_vit_giant_384": "vjepa2_1_vitg_384.pt",
        "vjepa2_1_vit_gigantic_384": "vjepa2_1_vitG_384.pt",
    }
    filename = filename_by_model.get(str(model_name))
    if filename is None:
        return None
    candidate = Path("checkpoints") / "foundation" / "vjepa2_1" / str(model_name) / filename
    return str(candidate) if candidate.is_file() else None


def _default_anytouch_checkpoint() -> str | None:
    candidate = Path("checkpoints") / "foundation" / "anytouch2" / "checkpoint-4frames.pth"
    return str(candidate) if candidate.is_file() else None


def _default_sonata_checkpoint() -> str | None:
    candidate = Path("src") / "pretrain" / "SpatialLM_Sonata_encoder.pth"
    return str(candidate) if candidate.is_file() else None


def _parse_tactile_sensor_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("Expected at least one tactile sensor name.")
    return names


def _parse_tactile_sensor_offsets(raw: str) -> tuple[tuple[float, float, float], ...]:
    offsets = []
    for block in str(raw).split(";"):
        block = block.strip()
        if not block:
            continue
        values = [float(piece.strip()) for piece in block.split(",") if piece.strip()]
        if len(values) != 3:
            raise ValueError(f"Expected 3 tactile offset values per sensor, got {block!r}.")
        offsets.append((values[0], values[1], values[2]))
    if not offsets:
        raise ValueError("Expected at least one tactile sensor offset triplet.")
    return tuple(offsets)


def _apply_foundation_profile(args: argparse.Namespace) -> None:
    if not bool(args.use_foundation_backbones):
        return
    args.point_backbone = "sonata"
    args.visual_mode = "encoder"
    args.tactile_mode = "encoder"
    args.use_tactile = True


def _validate_backbone_args(args: argparse.Namespace) -> None:
    args.tactile_sensor_names = _parse_tactile_sensor_names(args.tactile_sensor_names)
    args.tactile_sensor_offsets_m = _parse_tactile_sensor_offsets(args.tactile_sensor_offsets_m)
    if len(args.tactile_sensor_names) != len(args.tactile_sensor_offsets_m):
        raise ValueError("tactile_sensor_names and tactile_sensor_offsets_m must have the same length.")
    if args.visual_mode == "encoder":
        args.visual_checkpoint_path = args.visual_checkpoint_path or _default_vjepa_checkpoint(args.visual_model_name)
        if args.visual_checkpoint_path is None:
            raise FileNotFoundError("visual_mode=encoder requires --visual-checkpoint-path or a default V-JEPA checkpoint.")
    if args.tactile_mode == "encoder":
        args.use_tactile = True
        args.tactile_checkpoint_path = args.tactile_checkpoint_path or _default_anytouch_checkpoint()
        args.tactile_backgrounds_path = args.tactile_backgrounds_path or _default_tactile_backgrounds_path()
        args.tactile_calibration_path = args.tactile_calibration_path or _default_tactile_calibration_path()
        args.tactile_contact_stats_path = args.tactile_contact_stats_path or _default_tactile_contact_stats_path()
        if args.tactile_checkpoint_path is None:
            raise FileNotFoundError("tactile_mode=encoder requires --tactile-checkpoint-path or a default AnyTouch checkpoint.")
        if args.tactile_backgrounds_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder on CALVIN requires calibrated tactile backgrounds."
            )
        if args.tactile_calibration_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder on CALVIN requires fingertip geometry calibration."
            )
        if args.tactile_contact_stats_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder on CALVIN requires calibrated tactile contact thresholds."
            )
        stats = _load_tactile_contact_stats_json(args.tactile_contact_stats_path)
        if stats is None:
            raise FileNotFoundError(
                f"Failed to load tactile contact stats from {args.tactile_contact_stats_path!r}."
            )
    if args.point_backbone == "sonata":
        args.sonata_checkpoint_path = args.sonata_checkpoint_path or _default_sonata_checkpoint()
        if args.sonata_checkpoint_path is None:
            raise FileNotFoundError("point_backbone=sonata requires --sonata-checkpoint-path or a default Sonata checkpoint.")


def _build_core(args: argparse.Namespace) -> tuple[PicfFullCore, bool]:
    builder = CalvinDepthToPicfPointCloud(args.calvin_root, stride=args.stride, max_points=args.max_points)
    config = PicfCoreConfig(
        device=args.device,
        hidden_dim=64,
        posterior_hidden_dim=64,
        latent_dim=24,
        innovation_dim=64,
        control_dim=64,
        semantic_dim=32,
        semantic_cross_dim=64,
        future_hidden_dim=64,
        persistent_anchors=8,
        observation_anchors=10,
        fusion_layers=2,
        posterior_layers=1,
        predictive_layers=1,
        control_layers=1,
        predictive_semantic_reads=1,
        control_semantic_reads=1,
        attention_heads=4,
        future_vote_heads=3,
    )
    point_feature_extractor = None
    if args.point_backbone == "sonata":
        point_feature_extractor = SonataPointFeatureExtractor(
            SonataPointConfig(
                checkpoint_path=args.sonata_checkpoint_path,
                stage_name=args.sonata_stage_name,
                device=args.device,
                dtype=args.sonata_dtype,
                allow_random_init=False,
            )
        )

    if args.visual_mode == "encoder":
        visual_config = VjepaVisualConfig(
            model_name=args.visual_model_name,
            checkpoint_path=args.visual_checkpoint_path,
            checkpoint_key=args.visual_checkpoint_key,
            camera_json_path=args.calvin_root,
            device=args.device,
            dtype=args.visual_dtype,
            img_size=args.visual_img_size,
            num_frames=args.visual_num_frames,
            patch_size=args.visual_patch_size,
            tubelet_size=args.visual_tubelet_size,
            use_last_two_mean=bool(args.visual_use_last_two_mean),
        )
        visual_encoder = None
        use_visual_override = False
    else:
        visual_config = VjepaVisualConfig(
            camera_json_path=args.calvin_root,
            arch_name_override="vit_tiny",
            img_size=64,
            num_frames=4,
            device=args.device,
            dtype="float32",
        )
        visual_encoder = _NullVisualEncoder()
        use_visual_override = True

    tactile_config = None
    tactile_encoder = None
    if args.tactile_mode == "encoder":
        tactile_contact_stats = _load_tactile_contact_stats_json(args.tactile_contact_stats_path)
        if tactile_contact_stats is None:
            raise FileNotFoundError(
                f"Failed to load tactile contact stats from {args.tactile_contact_stats_path!r}."
            )
        tactile_config = AnyTouchConfig(
            checkpoint_path=args.tactile_checkpoint_path,
            device=args.device,
            dtype=args.tactile_dtype,
            num_frames=args.tactile_num_frames,
            stride=args.tactile_stride,
            allow_random_init=False,
            contact_tau_on=float(tactile_contact_stats["tau_on"]),
            contact_tau_off=float(tactile_contact_stats["tau_off"]),
            contact_temperature=float(tactile_contact_stats.get("temperature", max(0.5 * (float(tactile_contact_stats["tau_on"]) - float(tactile_contact_stats["tau_off"])), 1e-3))),
            contact_stats_payload=tactile_contact_stats,
        )
    else:
        tactile_encoder = _NullTactileEncoder()

    core = PicfFullCore(
        builder,
        config=config,
        point_feature_extractor=point_feature_extractor,
        visual_config=visual_config,
        visual_encoder=visual_encoder,
        tactile_config=tactile_config,
        tactile_encoder=tactile_encoder,
    )
    return core, use_visual_override


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
    use_tactile: bool,
    tactile_sensor_names: tuple[str, ...],
    tactile_sensor_offsets_m: tuple[tuple[float, float, float], ...],
    visual_mode: str,
    tactile_mode: str,
    point_backbone: str,
    visual_checkpoint_path: str | None,
    visual_checkpoint_key: str | None,
    visual_model_name: str,
    visual_dtype: str,
    visual_img_size: int,
    visual_num_frames: int,
    visual_patch_size: int,
    visual_tubelet_size: int,
    visual_use_last_two_mean: bool,
    tactile_checkpoint_path: str | None,
    tactile_dtype: str,
    tactile_num_frames: int,
    tactile_stride: int,
    sonata_checkpoint_path: str | None,
    sonata_stage_name: str,
    sonata_dtype: str,
    tactile_backgrounds_path: str | None = None,
    tactile_calibration_path: str | None = None,
    tactile_contact_stats_path: str | None = None,
) -> dict[str, float | int | str | bool]:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=[segment_index],
        use_tactile=bool(use_tactile),
        tactile_sensor_names=tactile_sensor_names,
        tactile_sensor_offsets_m=tactile_sensor_offsets_m,
        tactile_calibration=tactile_calibration_path,
        tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(tactile_backgrounds_path),
    )
    frames = list(replay)
    replay.close()
    if len(frames) < 2:
        raise RuntimeError("Need at least two frames in the selected CALVIN segment for transition training smoke.")

    args = argparse.Namespace(
        calvin_root=calvin_root,
        stride=stride,
        max_points=max_points,
        device=device,
        point_backbone=point_backbone,
        sonata_checkpoint_path=sonata_checkpoint_path,
        sonata_stage_name=sonata_stage_name,
        sonata_dtype=sonata_dtype,
        visual_mode=visual_mode,
        visual_checkpoint_path=visual_checkpoint_path,
        visual_checkpoint_key=visual_checkpoint_key,
        visual_model_name=visual_model_name,
        visual_dtype=visual_dtype,
        visual_img_size=visual_img_size,
        visual_num_frames=visual_num_frames,
        visual_patch_size=visual_patch_size,
        visual_tubelet_size=visual_tubelet_size,
        visual_use_last_two_mean=visual_use_last_two_mean,
        tactile_mode=tactile_mode,
        tactile_checkpoint_path=tactile_checkpoint_path,
        tactile_backgrounds_path=tactile_backgrounds_path,
        tactile_calibration_path=tactile_calibration_path,
        tactile_contact_stats_path=tactile_contact_stats_path,
        tactile_dtype=tactile_dtype,
        tactile_num_frames=tactile_num_frames,
        tactile_stride=tactile_stride,
    )
    core, use_visual_override = _build_core(args)
    optimizer = torch.optim.AdamW(core.parameters(), lr=lr)

    current = frames[0]
    nxt = frames[1]
    current_visual = _rgb_visual_override(current.rgb_static) if use_visual_override else None
    next_visual = _rgb_visual_override(nxt.rgb_static) if use_visual_override else None
    semantic = np.zeros((core.config.semantic_dim,), dtype=np.float32)

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
        "visual_mode": visual_mode,
        "tactile_mode": tactile_mode,
        "point_backbone": point_backbone,
        "tactile_enabled": bool(use_tactile),
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
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-foundation-backbones", action="store_true")
    parser.add_argument("--point-backbone", choices=["rgb", "sonata"], default="rgb")
    parser.add_argument("--sonata-checkpoint-path", default=None)
    parser.add_argument("--sonata-stage-name", default="enc4")
    parser.add_argument("--sonata-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--visual-mode", choices=["stub", "encoder"], default="stub")
    parser.add_argument("--visual-model-name", default="vjepa2_1_vit_base_384")
    parser.add_argument("--visual-checkpoint-path", default=None)
    parser.add_argument("--visual-checkpoint-key", default=None)
    parser.add_argument("--visual-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--visual-img-size", type=int, default=384)
    parser.add_argument("--visual-num-frames", type=int, default=64)
    parser.add_argument("--visual-patch-size", type=int, default=16)
    parser.add_argument("--visual-tubelet-size", type=int, default=2)
    parser.add_argument("--visual-use-last-two-mean", action="store_true")
    parser.add_argument("--tactile-mode", choices=["stub", "encoder"], default="stub")
    parser.add_argument("--tactile-checkpoint-path", default=None)
    parser.add_argument("--tactile-backgrounds-path", default=None)
    parser.add_argument("--tactile-calibration-path", default=None)
    parser.add_argument("--tactile-contact-stats-path", default=None)
    parser.add_argument("--tactile-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tactile-num-frames", type=int, default=4)
    parser.add_argument("--tactile-stride", type=int, default=2)
    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-sensor-names", default="digit,gelsight_mini")
    parser.add_argument("--tactile-sensor-offsets-m", default="0.01,0,0;-0.01,0,0")
    args = parser.parse_args()
    _apply_foundation_profile(args)
    _validate_backbone_args(args)

    summary = run_smoke(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        segment_index=args.segment_index,
        stride=args.stride,
        max_points=args.max_points,
        device=args.device,
        lr=args.lr,
        use_tactile=bool(args.use_tactile),
        tactile_sensor_names=args.tactile_sensor_names,
        tactile_sensor_offsets_m=args.tactile_sensor_offsets_m,
        visual_mode=args.visual_mode,
        tactile_mode=args.tactile_mode,
        point_backbone=args.point_backbone,
        visual_checkpoint_path=args.visual_checkpoint_path,
        visual_checkpoint_key=args.visual_checkpoint_key,
        visual_model_name=args.visual_model_name,
        visual_dtype=args.visual_dtype,
        visual_img_size=args.visual_img_size,
        visual_num_frames=args.visual_num_frames,
        visual_patch_size=args.visual_patch_size,
        visual_tubelet_size=args.visual_tubelet_size,
        visual_use_last_two_mean=bool(args.visual_use_last_two_mean),
        tactile_checkpoint_path=args.tactile_checkpoint_path,
        tactile_backgrounds_path=args.tactile_backgrounds_path,
        tactile_calibration_path=args.tactile_calibration_path,
        tactile_contact_stats_path=args.tactile_contact_stats_path,
        tactile_dtype=args.tactile_dtype,
        tactile_num_frames=args.tactile_num_frames,
        tactile_stride=args.tactile_stride,
        sonata_checkpoint_path=args.sonata_checkpoint_path,
        sonata_stage_name=args.sonata_stage_name,
        sonata_dtype=args.sonata_dtype,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from openpi.picf.contracts import PicfObservation
from openpi.picf.core import PicfCoreConfig
from openpi.picf.core import PicfFullCore
from openpi.picf.core import PicfTransitionLossBreakdown
from openpi.picf.core import compute_transition_loss
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.training.calvin_dataset import CalvinLangSegmentDataset


class _NullTactileEncoder:
    def encode_sensor_clips(self, *, clips_by_sensor, backgrounds_by_sensor, poses_by_sensor):
        del clips_by_sensor, backgrounds_by_sensor, poses_by_sensor
        return None


class _NullVisualEncoder:
    def encode_clip(self, _clip):
        raise AssertionError("visual_map_override should bypass encoder use in picf_core_train")


def _rgb_visual_override(rgb: np.ndarray, grid: int = 8) -> torch.Tensor:
    rgb_t = torch.as_tensor(np.asarray(rgb, dtype=np.float32) / 255.0, dtype=torch.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(rgb_t.permute(2, 0, 1)[None, :], (grid, grid))[0]
    return pooled.permute(1, 2, 0).contiguous()


def _setup_distributed(requested_device: str) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if str(requested_device).startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)
    if str(requested_device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device={requested_device!r}, but CUDA is not available.")
        if use_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            if ":" in requested_device:
                device = torch.device(requested_device)
            else:
                device = torch.device("cuda:0")
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return use_ddp, rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _seed_everything(seed: int, rank: int) -> None:
    mixed = int(seed) + (1009 * int(rank))
    random.seed(mixed)
    np.random.seed(mixed)
    torch.manual_seed(mixed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mixed)


def _reduce_mean(value: float, *, device: torch.device, world_size: int) -> float:
    if world_size <= 1:
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(world_size)
    return float(tensor.item())


def _grad_norm(parameters: Iterator[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        sq_sum += float(torch.sum(grad * grad).item())
        found = True
    return float(math.sqrt(sq_sum)) if found else 0.0


@dataclasses.dataclass(frozen=True)
class _TransitionWindow:
    segment_id: int
    start_step_id: int
    prompt: str
    frames: tuple[PicfObservation, ...]


class _CalvinTransitionSource:
    def __init__(
        self,
        root: str,
        *,
        split: str,
        backend: str,
        unroll_steps: int,
        use_wrist_rgb: bool = True,
        frame_dt_s: float = 1.0 / 30.0,
    ) -> None:
        if int(unroll_steps) < 1:
            raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")
        self.dataset = CalvinLangSegmentDataset(
            root=root,
            split=split,
            action_horizon=1,
            backend=backend,
            use_wrist_rgb=use_wrist_rgb,
            sample_within_segment=False,
        )
        self.reader = self.dataset.reader
        self.segments = self.dataset.segments
        self.split = split
        self.backend = backend
        self.unroll_steps = int(unroll_steps)
        self.use_wrist_rgb = bool(use_wrist_rgb)
        self.frame_dt_s = float(frame_dt_s)
        self.window_index: list[tuple[int, int]] = []
        for segment_id, segment in enumerate(self.segments):
            for step_id in range(segment.start, segment.end - self.unroll_steps):
                self.window_index.append((segment_id, step_id))
        if not self.window_index:
            raise RuntimeError(
                f"No valid CALVIN transition windows found for split={split}, backend={backend}, unroll_steps={unroll_steps}."
            )

    def __len__(self) -> int:
        return len(self.window_index)

    def close(self) -> None:
        self.reader.close()

    def _load_frame(self, segment_id: int, step_id: int, *, reset_scaffold: bool) -> PicfObservation:
        segment = self.segments[segment_id]
        keys = ["rgb_static", "depth_static", "robot_obs", "rel_actions"]
        if self.use_wrist_rgb:
            keys.append("rgb_gripper")
        frame = self.reader.read_npz(step_id, keys=keys)
        timestamp_s = float(step_id) * self.frame_dt_s
        return PicfObservation(
            rgb_static=frame["rgb_static"],
            depth_static=frame["depth_static"],
            robot_obs=frame["robot_obs"],
            prompt=segment.lang,
            step_id=int(step_id),
            segment_id=int(segment_id),
            timestamp_s=timestamp_s,
            reset_scaffold=bool(reset_scaffold),
            rgb_gripper=frame.get("rgb_gripper"),
            proprio=frame["robot_obs"],
            action=frame.get("rel_actions"),
        )

    def window(self, flat_index: int) -> _TransitionWindow:
        segment_id, start_step_id = self.window_index[int(flat_index)]
        frames = tuple(
            self._load_frame(
                segment_id,
                start_step_id + offset,
                reset_scaffold=(offset == 0),
            )
            for offset in range(self.unroll_steps + 1)
        )
        return _TransitionWindow(
            segment_id=int(segment_id),
            start_step_id=int(start_step_id),
            prompt=self.segments[segment_id].lang,
            frames=frames,
        )


@dataclasses.dataclass
class _MetricAccumulator:
    loss_total: float = 0.0
    loss_action: float = 0.0
    loss_visual_latent: float = 0.0
    loss_visual_real: float = 0.0
    loss_tactile_real: float = 0.0
    loss_point_real: float = 0.0
    loss_alignment: float = 0.0
    loss_anchor_pv: float = 0.0
    loss_pv_weak: float = 0.0
    loss_focus_pv: float = 0.0
    loss_pt: float = 0.0
    candidate_density: float = 0.0
    num_windows: int = 0

    def update(self, losses: PicfTransitionLossBreakdown, *, candidate_density: float) -> None:
        self.loss_total += float(losses.total.item())
        self.loss_action += float(losses.action.item())
        self.loss_visual_latent += float(losses.visual_latent.item())
        self.loss_visual_real += float(losses.visual_real.item())
        self.loss_tactile_real += float(losses.tactile_real.item())
        self.loss_point_real += float(losses.point_real.item())
        self.loss_alignment += float(losses.alignment.item())
        self.loss_anchor_pv += float(losses.anchor_pv.item())
        self.loss_pv_weak += float(losses.pv_weak.item())
        self.loss_focus_pv += float(losses.focus_pv.item())
        self.loss_pt += float(losses.pt.item())
        self.candidate_density += float(candidate_density)
        self.num_windows += 1

    def averages(self) -> dict[str, float]:
        denom = max(self.num_windows, 1)
        return {
            "loss_total": self.loss_total / denom,
            "loss_action": self.loss_action / denom,
            "loss_visual_latent": self.loss_visual_latent / denom,
            "loss_visual_real": self.loss_visual_real / denom,
            "loss_tactile_real": self.loss_tactile_real / denom,
            "loss_point_real": self.loss_point_real / denom,
            "loss_alignment": self.loss_alignment / denom,
            "loss_anchor_pv": self.loss_anchor_pv / denom,
            "loss_pv_weak": self.loss_pv_weak / denom,
            "loss_focus_pv": self.loss_focus_pv / denom,
            "loss_pt": self.loss_pt / denom,
            "projective_candidate_density": self.candidate_density / denom,
        }


class _PicfWindowTrainer(torch.nn.Module):
    def __init__(self, core: PicfFullCore, *, visual_grid: int) -> None:
        super().__init__()
        self.core = core
        self.visual_grid = int(visual_grid)

    def forward(self, window: _TransitionWindow) -> dict[str, torch.Tensor]:
        previous = None
        metrics: dict[str, torch.Tensor] | None = None
        totals: list[torch.Tensor] = []
        for index in range(len(window.frames) - 1):
            current = dataclasses.replace(window.frames[index], reset_scaffold=(index == 0))
            nxt = dataclasses.replace(window.frames[index + 1], reset_scaffold=False)
            current_visual = _rgb_visual_override(current.rgb_static, grid=self.visual_grid)
            next_visual = _rgb_visual_override(nxt.rgb_static, grid=self.visual_grid)
            output = self.core.step(
                current,
                previous=previous,
                visual_map_override=current_visual,
                action_future=current.action,
            )
            losses = compute_transition_loss(
                self.core,
                output,
                nxt,
                action_target=current.action,
                next_visual_map_override=next_visual,
            )
            totals.append(losses.total)
            candidate_density = torch.as_tensor(
                float(output.debug.get("projective_candidate_density", 0.0)),
                device=self.core.device,
                dtype=self.core.dtype,
            )
            if metrics is None:
                metrics = {
                    "loss_action": losses.action,
                    "loss_visual_latent": losses.visual_latent,
                    "loss_visual_real": losses.visual_real,
                    "loss_tactile_real": losses.tactile_real,
                    "loss_point_real": losses.point_real,
                    "loss_alignment": losses.alignment,
                    "loss_anchor_pv": losses.anchor_pv,
                    "loss_pv_weak": losses.pv_weak,
                    "loss_focus_pv": losses.focus_pv,
                    "loss_pt": losses.pt,
                    "projective_candidate_density": candidate_density,
                }
            else:
                metrics["loss_action"] = metrics["loss_action"] + losses.action
                metrics["loss_visual_latent"] = metrics["loss_visual_latent"] + losses.visual_latent
                metrics["loss_visual_real"] = metrics["loss_visual_real"] + losses.visual_real
                metrics["loss_tactile_real"] = metrics["loss_tactile_real"] + losses.tactile_real
                metrics["loss_point_real"] = metrics["loss_point_real"] + losses.point_real
                metrics["loss_alignment"] = metrics["loss_alignment"] + losses.alignment
                metrics["loss_anchor_pv"] = metrics["loss_anchor_pv"] + losses.anchor_pv
                metrics["loss_pv_weak"] = metrics["loss_pv_weak"] + losses.pv_weak
                metrics["loss_focus_pv"] = metrics["loss_focus_pv"] + losses.focus_pv
                metrics["loss_pt"] = metrics["loss_pt"] + losses.pt
                metrics["projective_candidate_density"] = metrics["projective_candidate_density"] + candidate_density
            previous = output.state

        assert metrics is not None
        denom = float(len(window.frames) - 1)
        mean_total = torch.stack(totals).mean()
        return {
            "loss_total": mean_total,
            "loss_action": metrics["loss_action"] / denom,
            "loss_visual_latent": metrics["loss_visual_latent"] / denom,
            "loss_visual_real": metrics["loss_visual_real"] / denom,
            "loss_tactile_real": metrics["loss_tactile_real"] / denom,
            "loss_point_real": metrics["loss_point_real"] / denom,
            "loss_alignment": metrics["loss_alignment"] / denom,
            "loss_anchor_pv": metrics["loss_anchor_pv"] / denom,
            "loss_pv_weak": metrics["loss_pv_weak"] / denom,
            "loss_focus_pv": metrics["loss_focus_pv"] / denom,
            "loss_pt": metrics["loss_pt"] / denom,
            "projective_candidate_density": metrics["projective_candidate_density"] / denom,
        }


def _lr_for_step(step: int, *, base_lr: float, warmup_steps: int, min_lr: float, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(max((step - warmup_steps) / float(total_steps - warmup_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _save_checkpoint(
    *,
    path: Path,
    model: _PicfWindowTrainer | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "model": module.core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
    }
    torch.save(payload, path)


def _load_checkpoint(
    *,
    path: Path,
    model: _PicfWindowTrainer | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.core.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


def _build_model(args: argparse.Namespace, *, device: torch.device) -> PicfFullCore:
    builder = CalvinDepthToPicfPointCloud(args.calvin_root, stride=args.stride, max_points=args.max_points)
    config = PicfCoreConfig(
        device=str(device),
        hidden_dim=args.hidden_dim,
        posterior_hidden_dim=args.posterior_hidden_dim,
        latent_dim=args.latent_dim,
        innovation_dim=args.innovation_dim,
        control_dim=args.control_dim,
        semantic_dim=args.semantic_dim,
        future_hidden_dim=args.future_hidden_dim,
        persistent_anchors=args.persistent_anchors,
        observation_anchors=args.observation_anchors,
        fusion_layers=args.fusion_layers,
        posterior_layers=args.posterior_layers,
        predictive_layers=args.predictive_layers,
        control_layers=args.control_layers,
        attention_heads=args.attention_heads,
        future_vote_heads=args.future_vote_heads,
    )
    return PicfFullCore(
        builder,
        config=config,
        visual_config=VjepaVisualConfig(
            camera_json_path=args.calvin_root,
            arch_name_override="vit_tiny",
            img_size=64,
            num_frames=4,
            device=str(device),
            dtype="float32",
        ),
        visual_encoder=_NullVisualEncoder(),
        tactile_encoder=_NullTactileEncoder(),
    )
def train(args: argparse.Namespace) -> None:
    use_ddp, rank, world_size, device = _setup_distributed(args.device)
    try:
        _seed_everything(args.seed, rank)
        is_main = _is_main(rank)
        source = _CalvinTransitionSource(
            args.calvin_root,
            split=args.split,
            backend=args.backend,
            unroll_steps=args.unroll_steps,
        )
        output_dir = Path(args.checkpoint_base_dir) / "picf_core" / args.exp_name
        latest_path = output_dir / "latest.pt"
        metrics_path = output_dir / "metrics.jsonl"
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

        core = _build_model(args, device=device).to(device)
        model = _PicfWindowTrainer(core, visual_grid=args.visual_grid).to(device)
        if use_ddp:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

        start_step = 0
        resume_path: Path | None = None
        if args.resume_checkpoint is not None:
            resume_path = Path(args.resume_checkpoint)
        elif args.resume:
            resume_path = latest_path
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            start_step = _load_checkpoint(path=resume_path, model=model, optimizer=optimizer, device=device)
            if is_main:
                print(f"[picf_core_train] resumed from {resume_path} at step={start_step}", flush=True)

        rng = np.random.default_rng(args.seed + 17 * rank)
        metric_accum = _MetricAccumulator()
        interval_start = time.time()

        for step in range(start_step, args.num_train_steps):
            lr = _lr_for_step(
                step,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                min_lr=args.min_lr,
                total_steps=args.num_train_steps,
            )
            _set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(args.accum_steps):
                flat_index = int(rng.integers(0, len(source)))
                window = source.window(flat_index)
                sync_context: Any
                if use_ddp and micro_step < args.accum_steps - 1:
                    sync_context = model.no_sync()
                else:
                    sync_context = contextlib.nullcontext()
                with sync_context:
                    outputs = model(window)
                    (outputs["loss_total"] / float(args.accum_steps)).backward()
                metric_accum.loss_total += float(outputs["loss_total"].detach().item())
                metric_accum.loss_action += float(outputs["loss_action"].detach().item())
                metric_accum.loss_visual_latent += float(outputs["loss_visual_latent"].detach().item())
                metric_accum.loss_visual_real += float(outputs["loss_visual_real"].detach().item())
                metric_accum.loss_tactile_real += float(outputs["loss_tactile_real"].detach().item())
                metric_accum.loss_point_real += float(outputs["loss_point_real"].detach().item())
                metric_accum.loss_alignment += float(outputs["loss_alignment"].detach().item())
                metric_accum.loss_anchor_pv += float(outputs["loss_anchor_pv"].detach().item())
                metric_accum.loss_pv_weak += float(outputs["loss_pv_weak"].detach().item())
                metric_accum.loss_focus_pv += float(outputs["loss_focus_pv"].detach().item())
                metric_accum.loss_pt += float(outputs["loss_pt"].detach().item())
                metric_accum.candidate_density += float(outputs["projective_candidate_density"].detach().item())
                metric_accum.num_windows += 1

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()

            if (step + 1) % args.log_interval == 0:
                elapsed = max(time.time() - interval_start, 1e-6)
                local_grad = _grad_norm(model.parameters())
                averages = metric_accum.averages()
                if world_size > 1:
                    averages = {k: _reduce_mean(v, device=device, world_size=world_size) for k, v in averages.items()}
                    local_grad = _reduce_mean(local_grad, device=device, world_size=world_size)
                if is_main:
                    record = {
                        "step": int(step + 1),
                        "lr": float(lr),
                        "grad_norm": float(local_grad),
                        "steps_per_sec": float(metric_accum.num_windows / elapsed),
                        **averages,
                    }
                    print(json.dumps(record, sort_keys=True), flush=True)
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, sort_keys=True) + "\n")
                metric_accum = _MetricAccumulator()
                interval_start = time.time()

            if is_main and ((step + 1) % args.save_interval == 0 or (step + 1) == args.num_train_steps):
                step_path = output_dir / f"step_{step + 1}.pt"
                _save_checkpoint(path=step_path, model=model, optimizer=optimizer, step=step + 1, args=args)
                _save_checkpoint(path=latest_path, model=model, optimizer=optimizer, step=step + 1, args=args)
                print(f"[picf_core_train] saved checkpoint step={step + 1} -> {step_path}", flush=True)
            if use_ddp:
                dist.barrier()

        source.close()
    finally:
        _cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-run PICF core training on CALVIN transition windows.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="dir", choices=["dir", "zip"])
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--num-train-steps", type=int, default=30000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--unroll-steps", type=int, default=2)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visual-grid", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--posterior-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--innovation-dim", type=int, default=64)
    parser.add_argument("--control-dim", type=int, default=64)
    parser.add_argument("--semantic-dim", type=int, default=32)
    parser.add_argument("--future-hidden-dim", type=int, default=64)
    parser.add_argument("--persistent-anchors", type=int, default=8)
    parser.add_argument("--observation-anchors", type=int, default=10)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--posterior-layers", type=int, default=1)
    parser.add_argument("--predictive-layers", type=int, default=1)
    parser.add_argument("--control-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--future-vote-heads", type=int, default=3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

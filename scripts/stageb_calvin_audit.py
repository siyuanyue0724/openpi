# scripts/stageb_calvin_audit.py
from __future__ import annotations

import argparse
import dataclasses
import os
import time
from typing import Any

import numpy as np
import torch

from openpi.training.config import get_config
from openpi.training.calvin_dataset import CalvinLangSegmentDataset
from openpi.training.data_loader import create_data_loader


def _torch_stats(x: torch.Tensor) -> str:
    if x.numel() == 0:
        return f"shape={tuple(x.shape)} dtype={x.dtype} EMPTY"
    return f"shape={tuple(x.shape)} dtype={x.dtype} min={x.min().item():.4g} max={x.max().item():.4g}"


def _safe_minmax_np(x: Any) -> str:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return f"shape={arr.shape} dtype={arr.dtype} EMPTY"
        # 字符串/对象数组不做 min/max
        if arr.dtype.kind in ("U", "S", "O"):
            return f"shape={arr.shape} dtype={arr.dtype}"
        return f"shape={arr.shape} dtype={arr.dtype} min={arr.min():.4g} max={arr.max():.4g}"
    except Exception as e:
        return f"<unprintable: {type(x)} err={e!r}>"


def _get_calvin_root(cli_root: str | None) -> str:
    if cli_root:
        return cli_root
    env = os.environ.get("CALVIN_ZIP") or os.environ.get("CALVIN_ROOT")
    if env:
        return env
    raise RuntimeError("CALVIN root not provided. Use --calvin-root or export CALVIN_ZIP=...")


def _apply_data_overrides(cfg, split: str | None = None, cameras_json_path: str | None = None):
    data = cfg.data
    changed = False
    if split is not None:
        data = dataclasses.replace(data, split=str(split))
        changed = True
    if cameras_json_path is not None:
        data = dataclasses.replace(data, cameras_json_path=str(cameras_json_path))
        changed = True
    if changed:
        cfg = dataclasses.replace(cfg, data=data)
    return cfg


@torch.no_grad()
def mode_dataset(cfg_name: str, calvin_root: str, num_workers: int, iters: int, split: str | None) -> None:
    cfg = get_config(cfg_name)
    cfg = _apply_data_overrides(cfg, split=split, cameras_json_path=None)
    dcfg = cfg.data  # CalvinDataConfig (factory)

    ds = CalvinLangSegmentDataset(
        root=calvin_root,
        split=str(getattr(dcfg, "split", "training")),
        backend=str(getattr(dcfg, "backend", "zip")),
        action_horizon=int(cfg.model.action_horizon),
        action_key=str(getattr(dcfg, "action_key", "rel_actions")),
        use_wrist_rgb=bool(getattr(dcfg, "use_wrist_rgb", True)),
        rng_seed=0,
    )

    mp_ctx = torch.multiprocessing.get_context("spawn") if num_workers > 0 else None
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_ctx,
        persistent_workers=(num_workers > 0),
    )

    print(f"[dataset] cfg={cfg_name} root={calvin_root} split={getattr(dcfg, 'split', 'training')}")
    print(f"[dataset] num_workers={num_workers} iters={iters} segments={len(ds)}")
    t0 = time.time()

    for i, sample in enumerate(dl):
        if i >= iters:
            break
        if i == 0:
            print("[dataset] keys:", list(sample.keys()))

        # 这些 key 是 calvin_dataset.py 明确产出的
        for k in ["rgb_static", "depth_static", "robot_obs", "rgb_gripper", "actions", "prompt"]:
            if k not in sample:
                continue
            v = sample[k]
            if isinstance(v, torch.Tensor):
                # depth/rgb/actions/robot_obs
                msg = _torch_stats(v)
            else:
                # prompt -> list[str]
                msg = _safe_minmax_np(v)
            print(f"[dataset] {k}: {msg}")

        if "prompt" in sample:
            p = sample["prompt"][0] if isinstance(sample["prompt"], list) else sample["prompt"]
            p = str(p)
            print(f"[dataset] prompt_preview(len={len(p)}): {p[:120]!r}")
        if i == 0:
            print("-" * 90)

    dt = time.time() - t0
    print(f"[dataset] done: {iters} iters in {dt:.2f}s")


@torch.no_grad()
def mode_loader(
    cfg_name: str,
    calvin_root: str,
    num_workers: int,
    num_batches: int,
    batch_size: int | None,
    max_token_len: int | None,
    split: str | None,
    cameras_json_path: str | None,
) -> None:
    os.environ["CALVIN_ZIP"] = calvin_root

    cfg = get_config(cfg_name)
    if num_workers is not None:
        cfg = dataclasses.replace(cfg, num_workers=int(num_workers))
    if batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=int(batch_size))
    if max_token_len is not None:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, max_token_len=int(max_token_len)))
    cfg = _apply_data_overrides(cfg, split=split, cameras_json_path=cameras_json_path)
    
    print(f"[loader] split={getattr(cfg.data,'split',None)} cameras_json_path={getattr(cfg.data,'cameras_json_path',None)}")
    print(f"[loader] cfg={cfg_name} root={calvin_root}")
    print(f"[loader] batch_size={cfg.batch_size} num_workers={cfg.num_workers} num_batches={num_batches}")
    print(f"[loader] model.max_token_len={cfg.model.max_token_len} action_horizon={cfg.model.action_horizon}")
    print(f"[loader] point_token_cap={getattr(cfg.model,'point_token_cap',None)} point_feat_dim={getattr(cfg.model,'point_feat_dim',None)}")

    dl = create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=num_batches, skip_norm_stats=False)

    tok_lens: list[int] = []
    trunc_cnt = 0
    win_bad = 0

    pc_valid: list[int] = []
    pc_mask_true = 0
    pc_total = 0
    z_mins: list[float] = []
    z_maxs: list[float] = []

    for bi, (obs, act) in enumerate(dl):
        if bi == 0:
            print("[loader] obs.state:", _torch_stats(obs.state))
            print("[loader] act:", _torch_stats(act))

        # token length / window token sanity
        if obs.tokenized_prompt_mask is not None:
            lens = obs.tokenized_prompt_mask.sum(dim=-1).to(torch.int64)  # [B]
            tok_lens.extend([int(x) for x in lens.cpu().tolist()])
            trunc_cnt += int((lens >= int(cfg.model.max_token_len)).sum().item())

            # 检查 point window token 是否“恰好一对”
            # 用 “masked tokens 的最大 id” 近似 end_id（通常是 vocab_size-1），start_id=end_id-1
            tokens = obs.tokenized_prompt.to(torch.long)
            masks = obs.tokenized_prompt_mask.to(torch.bool)
            for b in range(tokens.shape[0]):
                t = tokens[b][masks[b]]
                if t.numel() == 0:
                    win_bad += 1
                    continue
                end_id = int(t.max().item())
                start_id = end_id - 1
                c_end = int((t == end_id).sum().item())
                c_start = int((t == start_id).sum().item())
                if not (c_end == 1 and c_start == 1):
                    win_bad += 1

        # point cloud sanity
        if getattr(obs, "point_clouds", None) and ("pointcloud" in obs.point_clouds):
            pcs = obs.point_clouds["pointcloud"]  # [B,M,3+feat]
            pm = obs.point_cloud_masks.get("pointcloud", None)  # [B]
            if pm is None:
                pm = torch.ones((pcs.shape[0],), dtype=torch.bool, device=pcs.device)

            pc_total += int(pm.numel())
            pc_mask_true += int(pm.sum().item())

            grid = pcs[..., :3]
            feat = pcs[..., 3:]
            xyz = feat[..., :3]

            # 模型侧的 padding 判定：grid=0 & xyz=0 & feat=0
            pad = (grid == 0).all(dim=-1) & (xyz == 0).all(dim=-1) & (feat == 0).all(dim=-1)
            valid = (~pad).sum(dim=-1)  # [B]
            pc_valid.extend([int(x) for x in valid.cpu().tolist()])

            # z range（忽略 pad）
            for b in range(pcs.shape[0]):
                if not bool(pm[b].item()):
                    continue
                vmask = ~pad[b]
                if int(vmask.sum().item()) == 0:
                    continue
                z = xyz[b, vmask, 2]
                z_mins.append(float(z.min().item()))
                z_maxs.append(float(z.max().item()))

        if bi == 0:
            # 打印点云基本形状
            if getattr(obs, "point_clouds", None) and ("pointcloud" in obs.point_clouds):
                print("[loader] pointcloud:", _torch_stats(obs.point_clouds["pointcloud"]))
                print("[loader] pointcloud_mask:", obs.point_cloud_masks.get("pointcloud", None))
            print("-" * 90)

    # summary
    if tok_lens:
        p50 = int(np.median(tok_lens))
        p95 = int(np.percentile(tok_lens, 95))
        print(f"[loader] token_len: min={min(tok_lens)} p50={p50} p95={p95} max={max(tok_lens)}")
        print(f"[loader] token_len>=max_token_len: {trunc_cnt}/{len(tok_lens)}")
        print(f"[loader] window_token_bad_count: {win_bad}/{len(tok_lens)} (should be 0)")
    else:
        print("[loader] token_len: NA (tokenized_prompt_mask is None?)")

    if pc_total > 0:
        ratio = pc_mask_true / max(pc_total, 1)
        print(f"[loader] pointcloud_mask_true_ratio: {ratio:.3f} ({pc_mask_true}/{pc_total})")
    if pc_valid:
        p50 = int(np.median(pc_valid))
        p95 = int(np.percentile(pc_valid, 95))
        print(f"[loader] pc_valid_points(after_pad_filter): min={min(pc_valid)} p50={p50} p95={p95} max={max(pc_valid)}")
    if z_mins and z_maxs:
        print(f"[loader] z_range(valid): min={min(z_mins):.4g} max={max(z_maxs):.4g}")


@torch.no_grad()
def mode_sonata(cfg_name: str, calvin_root: str, num_batches: int, device: str, split: str | None, cameras_json_path: str | None) -> None:
    os.environ["CALVIN_ZIP"] = calvin_root

    cfg = get_config(cfg_name)
    cfg = dataclasses.replace(cfg, batch_size=1, num_workers=0)  # 先单进程，避免把问题混到 spawn
    cfg = _apply_data_overrides(cfg, split=split, cameras_json_path=cameras_json_path)
    dl = create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=num_batches, skip_norm_stats=False)

    try:
        from openpi.models.sonata_encoder import Sonata
    except Exception as e:
        print("[sonata] import failed:", repr(e))
        print("[sonata] 先跳过本模式；确认 spconv/torch-scatter/flash-attn 等依赖后再测。")
        return

    dev = torch.device(device)
    cap = int(getattr(cfg.model, "point_token_cap", 1024))
    feat_dim = int(getattr(cfg.model, "point_feat_dim", 6))
    sonata = Sonata(in_channels=feat_dim).to(device=dev, dtype=torch.float32)
    sonata.eval()

    print(f"[sonata] device={device} cap={cap} point_feat_dim={feat_dim}")

    near_cap = 0
    max_seen = 0

    for bi, (obs, _) in enumerate(dl):
        pm = obs.point_cloud_masks.get("pointcloud", None)
        if pm is not None and (not bool(pm[0].item())):
            print(f"[sonata] batch {bi}: pointcloud_mask=False (skip)")
            continue

        pcs = obs.point_clouds["pointcloud"][0].to(dev, dtype=torch.float32)  # [M, 3+feat]
        g = pcs[:, :3].to(torch.int64)
        f = pcs[:, 3:].to(torch.float32)
        c = f[:, :3].to(torch.float32)

        pad = (g == 0).all(dim=1) & (c == 0).all(dim=1) & (f == 0).all(dim=1)
        g = g[~pad]; c = c[~pad]; f = f[~pad]
        n = int(g.shape[0])
        if n == 0:
            print(f"[sonata] batch {bi}: all padded (skip)")
            continue

        sample = {
            "coord": c,
            "feat": f,
            "grid_coord": g,
            "batch": torch.zeros((n,), dtype=torch.int64, device=dev),
            "offset": torch.tensor([n], dtype=torch.int64, device=dev),
        }
        out = sonata(sample)
        enc = out if isinstance(out, torch.Tensor) else getattr(out, "feat", None)
        if enc is None:
            raise RuntimeError("Sonata.forward must return Tensor or object with .feat")
        token_len = int(enc.size(0))
        max_seen = max(max_seen, token_len)
        if token_len >= int(0.95 * cap):
            near_cap += 1

        print(f"[sonata] batch {bi}: input_points={n} token_len={token_len}")
        if token_len > cap:
            print(f"[sonata][ERROR] token_len({token_len}) > cap({cap}) !!!")
            break

    print(f"[sonata] max_token_len_seen={max_seen}, near_cap={near_cap}/{num_batches}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dataset", "loader", "sonata"], required=True)
    ap.add_argument("--config", default="pi05_calvin_sonata")
    ap.add_argument("--calvin-root", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--num-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-token-len", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=["training", "validation"], default=None)
    ap.add_argument("--cameras-json-path", default=None)
    args = ap.parse_args()

    root = _get_calvin_root(args.calvin_root)

    if args.mode == "dataset":
        mode_dataset(args.config, root, num_workers=args.num_workers, iters=args.iters, split=args.split)
    elif args.mode == "loader":
        mode_loader(
            args.config,
            root,
            num_workers=args.num_workers,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            max_token_len=args.max_token_len,
            split=args.split,
            cameras_json_path=args.cameras_json_path,
        )
    else:
        mode_sonata(
            args.config,
            root,
            num_batches=args.num_batches,
            device=args.device,
            split=args.split,
            cameras_json_path=args.cameras_json_path,
        )


if __name__ == "__main__":
    main()

"""
Sonata integration probe runner (PyTorch) - GPU-memory safe.

Key design:
  - This runner DOES NOT instantiate PI0Pytorch / PaliGemma weights.
  - It only:
      1) loads TrainConfig via openpi.training.config.cli()
      2) builds one batch from openpi.training.data_loader
      3) builds a tiny Sonata-only module
      4) runs scripts/sonata_probe.probe_sonata_integration twice:
           - run_encode=False (static checks + point window token checks)
           - run_encode=True  (actually runs Sonata encode once, reports token_len)

When to delete:
  rm scripts/run_sonata_probe_pytorch.py scripts/sonata_probe.py
"""

from __future__ import annotations

import os
# Must be set BEFORE importing torch to affect allocator behavior.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")

import sys
from pathlib import Path
import logging
import dataclasses as _dc

import torch
from torch import nn

# Make imports robust even if PYTHONPATH is not set (probe-only convenience)
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = _REPO / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models.sonata_encoder import Sonata

from scripts.sonata_probe import probe_sonata_integration


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def pick_device(require_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
        return dev
    if require_cuda:
        raise RuntimeError("CUDA is required for this probe but torch.cuda.is_available() is False.")
    return torch.device("cpu")


class SonataProbeModel(nn.Module):
    """
    Minimal Sonata-only model that mimics PI0Pytorch's encode contract:
      - attributes used by scripts/sonata_probe.py
      - method torch_sonata_encode_batch(observation, train=False)

    IMPORTANT:
      - No PI0/Paligemma weights here -> avoids 8GB GPU OOM.
    """

    def __init__(
        self,
        *,
        point_feat_dim: int,
        point_token_cap: int,
        enable_sonata: bool,
        sonata_mode: str,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        self.enable_sonata = bool(enable_sonata)
        self.sonata_mode = str(sonata_mode)
        if self.sonata_mode not in ("off", "projector", "all"):
            raise ValueError(f"Invalid sonata_mode: {self.sonata_mode!r}. Expected off|projector|all")

        self.require_cuda = bool(require_cuda)

        self.sonata_ckpt = None
        self.sonata_projector_ckpt = None

        self.point_feat_dim = int(point_feat_dim)
        self.point_token_cap = int(point_token_cap)

        self.sonata_in_channels = int(self.point_feat_dim)

        # match main model toggles (env-controlled)
        self.sonata_auto_pad_feat = str(os.environ.get("OPENPI_SONATA_AUTO_PAD_FEAT", "0")).strip().lower() in (
            "1", "true", "yes", "on"
        )
        self.sonata_validate = str(os.environ.get("OPENPI_SONATA_VALIDATE", "0")).strip().lower() in (
            "1", "true", "yes", "on"
        )

        # point window ids are usually set by tokenizer (not available here); probe will infer from prompt
        self.point_start_id = None
        self.point_end_id = None

        # projector is not used in this probe-only runner
        self.pc_projector = None
        self._pc_projector_loaded = False

        # cache populated by encode
        self._pt_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        # Build Sonata (random init is OK for shape/token_len validation)
        if self.sonata_in_channels < 3:
            raise RuntimeError(f"Invalid point_feat_dim={self.sonata_in_channels} (<3). Expect xyz in first 3 dims.")
        self.sonata = Sonata(in_channels=self.sonata_in_channels)
        # keep float32 by design
        self.sonata.to(dtype=torch.float32)

    @torch.no_grad()
    def torch_sonata_encode_batch(self, observation, train: bool = False):
        if not self.enable_sonata or self.sonata_mode == "off":
            raise RuntimeError("Sonata is disabled (enable_sonata=False or sonata_mode=off).")

        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Sonata, but CUDA is not available.")

        if not hasattr(observation, "point_clouds") or not observation.point_clouds:
            raise RuntimeError("Observation.point_clouds is missing or empty.")
        if "pointcloud" not in observation.point_clouds:
            raise RuntimeError("Observation.point_clouds['pointcloud'] is missing.")

        device = next(self.parameters()).device

        pcs = observation.point_clouds["pointcloud"]
        pmask = None
        if hasattr(observation, "point_cloud_masks") and observation.point_cloud_masks:
            pmask = observation.point_cloud_masks.get("pointcloud", None)

        if not isinstance(pcs, torch.Tensor):
            pcs = torch.as_tensor(pcs)
        pcs = pcs.to(device=device, dtype=torch.float32)

        if pcs.ndim != 3:
            raise RuntimeError(f"pointcloud must be [B,M,D], got {tuple(pcs.shape)}")
        B, M, D = pcs.shape
        if D < 6:
            raise RuntimeError(f"pointcloud last dim {D} < 6; expect [3 grid | 3 xyz | extras].")

        # pmask: [B] or [B,M]
        per_point_mask: torch.Tensor | None = None
        if pmask is None:
            pmask_b = torch.ones((B,), dtype=torch.bool, device=device)
        else:
            if not isinstance(pmask, torch.Tensor):
                pmask = torch.as_tensor(pmask)
            pmask = pmask.to(device=device)
            if pmask.ndim == 2:
                per_point_mask = pmask.to(torch.bool)
                pmask_b = per_point_mask.any(dim=1)
            elif pmask.ndim == 1:
                pmask_b = pmask.to(torch.bool)
            else:
                raise RuntimeError(
                    "point_cloud_masks['pointcloud'] must be shape [B] or [B,M], "
                    f"but got {tuple(pmask.shape)}"
                )

        grid = pcs[..., :3]
        if self.sonata_validate:
            if torch.isnan(pcs).any() or torch.isinf(pcs).any():
                raise RuntimeError("Point cloud contains NaN/Inf.")
            if (grid < 0).any():
                raise RuntimeError("Point grid (first 3 dims) must be non-negative.")

        obs_fd = int(D - 3)
        exp_fd = int(self.sonata_in_channels)
        if obs_fd < 3:
            raise RuntimeError(f"pointcloud feature dims (after first 3 grid dims) must be >=3, got {obs_fd}")
        if exp_fd < 3:
            raise RuntimeError(f"sonata_in_channels must be >=3, got {exp_fd}")

        if obs_fd != exp_fd:
            if not self.sonata_auto_pad_feat:
                raise RuntimeError(
                    f"pointcloud feature dim mismatch: observation has {obs_fd} dims (pcs[...,3:]) "
                    f"but Sonata expects {exp_fd}. "
                    "Fix the data pipeline / config.point_feat_dim, or set OPENPI_SONATA_AUTO_PAD_FEAT=1."
                )

        do_trunc = obs_fd > exp_fd
        do_pad = obs_fd < exp_fd

        cap = int(self.point_token_cap)
        if cap <= 0:
            raise RuntimeError("point_token_cap must be > 0.")

        pt_list: list[torch.Tensor | None] = []
        mask_list: list[torch.Tensor] = []
        enc_dim: int | None = None

        # Sonata should be eval in probe; no grads
        self.sonata.eval()

        for b in range(B):
            if not pmask_b[b]:
                pt_list.append(None)
                mask_list.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                continue

            arr = pcs[b]
            g = arr[:, :3].to(torch.int64)
            f = arr[:, 3:].to(torch.float32)

            # apply per-point mask first (if exists)
            if per_point_mask is not None:
                valid = per_point_mask[b]
                if valid.ndim != 1 or valid.shape[0] != g.shape[0]:
                    raise RuntimeError(
                        "per_point_mask[b] must be shape [M] matching pcs[b]. "
                        f"Got {tuple(valid.shape)} vs M={g.shape[0]}."
                    )
                g = g[valid]
                f = f[valid]

            if do_trunc:
                f = f[:, :exp_fd]

            c = f[:, :3].to(torch.float32)

            # If no explicit per-point mask, remove padded all-zero points
            if per_point_mask is None:
                pad = (g == 0).all(dim=1) & (c == 0).all(dim=1) & (f == 0).all(dim=1)
                g = g[~pad]
                c = c[~pad]
                f = f[~pad]

            n = int(g.shape[0])
            if n == 0:
                pt_list.append(None)
                mask_list.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                continue

            if do_pad:
                pad_ch = exp_fd - int(f.shape[1])
                if pad_ch > 0:
                    f = torch.cat([f, torch.zeros((n, pad_ch), dtype=f.dtype, device=f.device)], dim=1)

            sample = {
                "coord": c,
                "feat": f,
                "grid_coord": g,
                "batch": torch.zeros((n,), dtype=torch.int64, device=device),
                "offset": torch.tensor([n], dtype=torch.int64, device=device),
            }

            out_any = self.sonata(sample)
            enc = out_any if isinstance(out_any, torch.Tensor) else getattr(out_any, "feat", None)
            if enc is None:
                raise RuntimeError("Sonata.forward must return Tensor or object with .feat")

            real = int(enc.size(0))
            if real > cap:
                raise RuntimeError(f"[Sonata] token_len={real} exceeds point_token_cap={cap}.")

            if enc_dim is None:
                enc_dim = int(enc.size(1))
            elif int(enc.size(1)) != enc_dim:
                raise RuntimeError(f"[Sonata] inconsistent token dim: {int(enc.size(1))} vs {enc_dim}")

            if real < cap:
                enc = torch.cat([enc, torch.zeros((cap - real, enc_dim), dtype=enc.dtype, device=enc.device)], dim=0)

            m = torch.zeros((cap,), dtype=torch.bool, device=device)
            m[:real] = True
            pt_list.append(enc.to(dtype=torch.float32))
            mask_list.append(m)

        if enc_dim is None:
            enc_dim = 512  # reasonable default for Sonata enc dim

        # Fill empty samples with zeros
        for i, v in enumerate(pt_list):
            if v is None:
                pt_list[i] = torch.zeros((cap, enc_dim), dtype=torch.float32, device=device)

        pt_feat_raw = torch.stack([v for v in pt_list if v is not None], dim=0)
        pt_mask = torch.stack(mask_list, dim=0)
        self._pt_cache = (pt_feat_raw, pt_mask)
        return self._pt_cache


def main() -> None:
    init_logging()
    logger = logging.getLogger("openpi")

    # Parse config via OpenPI tyro CLI (same as training scripts)
    config = _config.cli()

    # determine sonata mode: env overrides config
    sonata_mode = str(os.environ.get("OPENPI_SONATA_MODE", getattr(config.model, "sonata_mode", "projector"))).lower()
    if sonata_mode not in ("off", "projector", "all"):
        raise ValueError(f"Invalid OPENPI_SONATA_MODE={sonata_mode!r}. Expected off|projector|all")

    enable_sonata = bool(getattr(config.model, "enable_sonata", True))
    point_feat_dim = int(getattr(config.model, "point_feat_dim", 6))
    point_token_cap = int(getattr(config.model, "point_token_cap", 1024))

    device = pick_device(require_cuda=True)
    logger.info(
        "[ProbeRunner] device=%s enable_sonata=%s sonata_mode=%s point_feat_dim=%d point_token_cap=%d",
        device, enable_sonata, sonata_mode, point_feat_dim, point_token_cap,
    )

    # Build one batch (no model weights involved)
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    batch = next(iter(loader))
    if not isinstance(batch, (list, tuple)) or len(batch) != 2:
        raise RuntimeError("Data loader must yield (observation, actions).")
    observation, _actions = batch

    # Build tiny Sonata-only probe model
    probe_model = SonataProbeModel(
        point_feat_dim=point_feat_dim,
        point_token_cap=point_token_cap,
        enable_sonata=enable_sonata,
        sonata_mode=sonata_mode,
        require_cuda=True,
    ).to(device)

    # Probe 1: static checks + point window token checks
    info0 = probe_sonata_integration(probe_model, observation, run_encode=False)

    # Hard fail if point window token check didn't pass (this is the remaining “must-fix” item)
    if enable_sonata and sonata_mode in ("projector", "all"):
        if info0.get("point_window_ids_missing", False):
            raise SystemExit(
                "Point window token IDs are missing and could NOT be inferred from tokenized_prompt. "
                "This likely means _EnsurePointWindow didn't run or prompt got truncated."
            )
        ok_frac = float(info0.get("ok_pair_frac", 0.0) or 0.0)
        if ok_frac < 1.0:
            raise SystemExit(
                f"Point window token check FAILED: ok_pair_frac={ok_frac}. "
                "Expected 1.0 (each sample must have exactly one start and one end token, start_pos < end_pos)."
            )

    # Probe 2: run encode once and report token_len distribution
    info1 = probe_sonata_integration(probe_model, observation, run_encode=True)
    if "encode_error" in info1:
        raise SystemExit(f"Sonata encode FAILED: {info1['encode_error']}")

    logger.info("[ProbeRunner] SUCCESS: point window tokens OK; Sonata encode OK.")


if __name__ == "__main__":
    main()

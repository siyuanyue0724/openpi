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

_SONATA_IGNORED_KEYS: set[str] = {"embedding.mask_token"}
_SONATA_VALID_ORDERS: set[str] = {"z", "z-trans", "hilbert", "hilbert-trans"}


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


def _get_openpi_data_home() -> Path:
    """OpenPI data/cache root.

    Matches OpenPI README:
      - default: ~/.cache/openpi
      - override: OPENPI_DATA_HOME=/abs/path
    """
    env = os.environ.get("OPENPI_DATA_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "openpi"


def _resolve_ckpt_path(path: str | None, *, kind: str) -> str | None:
    """Resolve a checkpoint path in a way consistent with OpenPI's ecosystem.

    Accepts:
      - absolute file path
      - relative file path: resolved under OPENPI_DATA_HOME
      - user home (~) expansion
    """
    if path is None:
        return None
    p = str(path).strip()
    if not p:
        return None

    p_exp = os.path.expanduser(p)
    cand = Path(p_exp)
    if cand.is_file():
        return str(cand)

    if not cand.is_absolute():
        dh = _get_openpi_data_home()
        cand2 = dh / cand
        if cand2.is_file():
            return str(cand2)

    raise FileNotFoundError(
        f"{kind} ckpt not found: '{p}'. "
        f"Tried: '{cand}' and '{(_get_openpi_data_home() / cand) if not cand.is_absolute() else cand}'."
    )


def _torch_load_cpu(path: str):
    """Load a checkpoint onto CPU (PyTorch>=2.6 safe-default compatible).

    PyTorch>=2.6 changed torch.load default to weights_only=True, which can fail for many
    training checkpoints (.pth/.pt) that contain numpy scalars and other pickled objects.

    SECURITY NOTE:
      weights_only=False can execute arbitrary code during unpickling.
      Only use it with checkpoints from a trusted source (your own cache / official models).
    """
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file  # noqa: WPS433

        return load_file(p, device="cpu")
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def _infer_sonata_in_channels(sd: dict[str, torch.Tensor]) -> int | None:
    """Infer Sonata in_channels from a state_dict.

    Most reliable signal: embedding.stem.linear.weight has shape [out_dim, in_dim],
    where in_dim == in_channels.
    """
    for k, v in sd.items():
        if (
            isinstance(k, str)
            and k.endswith("embedding.stem.linear.weight")
            and isinstance(v, torch.Tensor)
            and v.ndim == 2
        ):
            return int(v.shape[1])
    return None


def _infer_sonata_enable_fourier(sd: dict[str, torch.Tensor]) -> bool:
    """Detect whether ckpt expects enable_fourier_encode=True (input_proj.* present)."""
    return ("input_proj.weight" in sd) or ("input_proj.bias" in sd)


def _parse_bool_env(name: str) -> bool | None:
    v = os.environ.get(name, None)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid {name}={v!r} (expected 1/0 true/false yes/no on/off)")


def _parse_sonata_order_env(*, default: tuple[str, ...] = ("z", "z-trans")) -> tuple[str, ...]:
    """Parse OPENPI_SONATA_ORDER as comma-separated list."""
    raw = os.environ.get("OPENPI_SONATA_ORDER", "").strip()
    if not raw:
        return default
    parts = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return default
    bad = [p for p in parts if p not in _SONATA_VALID_ORDERS]
    if bad:
        raise ValueError(
            f"Invalid OPENPI_SONATA_ORDER entries: {bad}. "
            f"Valid: {sorted(_SONATA_VALID_ORDERS)}. Example: 'z,z-trans,hilbert,hilbert-trans'"
        )
    return tuple(parts)

def _resolve_sonata_mode_from_config_and_env(model_cfg) -> str:
    """Mirror PI0Pytorch Sonata-mode resolution exactly."""
    env_new = os.environ.get("OPENPI_SONATA_MODE")
    env_old = os.environ.get("OPENPI_SONATA_TRAIN_MODE")
    if env_new and env_old and env_new.lower() != env_old.lower():
        raise RuntimeError("Conflicting SONATA mode env vars: OPENPI_SONATA_MODE vs OPENPI_SONATA_TRAIN_MODE")

    cfg_mode = getattr(model_cfg, "sonata_mode", None)
    legacy_cfg_mode = getattr(model_cfg, "sonata_train_mode", None)
    if cfg_mode is not None:
        cfg_mode = str(cfg_mode).strip().lower()
    if legacy_cfg_mode is not None:
        legacy_cfg_mode = str(legacy_cfg_mode).strip().lower()
    if (cfg_mode is not None) and (legacy_cfg_mode is not None) and (cfg_mode != legacy_cfg_mode):
        raise RuntimeError(
            f"Conflicting config Sonata modes: sonata_mode={cfg_mode!r} vs sonata_train_mode={legacy_cfg_mode!r}"
        )

    mode = str(cfg_mode or legacy_cfg_mode or env_new or env_old or "all").strip().lower()
    if mode not in ("off", "projector", "all"):
        raise ValueError(f"Invalid Sonata mode: {mode!r}. Expected off|projector|all")
    return mode

def _unwrap_state_dict(
    obj: object,
    *,
    kind: str,
    sonata_pretrain: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Unwrap common checkpoint containers into a plain state_dict."""
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "encoder"):
            v = obj.get(k, None)
            if isinstance(v, dict):
                obj = v
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"{kind} checkpoint must be a dict-like state_dict, got {type(obj)}")

    sd: dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            sd[str(k)] = v

    if not sd:
        raise RuntimeError(
            f"{kind} checkpoint does not look like a state_dict of tensors. "
            f"Top-level keys preview: {list(obj.keys())[:40] if isinstance(obj, dict) else 'N/A'}"
        )

    # Infer sonata_pretrain if not provided.
    if sonata_pretrain is None:
        sonata_pretrain = "sonata" in str(kind).lower()

    # Special-case: facebook/sonata pretrain checkpoints
    if sonata_pretrain:
        candidates = (
            "teacher.backbone.",
            "module.teacher.backbone.",
            "student.backbone.",
            "module.student.backbone.",
            "backbone.",
            "module.backbone.",
        )
        selected = None
        for pre in candidates:
            if any(k.startswith(pre) for k in sd.keys()):
                mapped = {k[len(pre) :]: v for k, v in sd.items() if k.startswith(pre)}
                if mapped:
                    selected = mapped
                    break
        if selected is not None:
            sd = selected
        for k in _SONATA_IGNORED_KEYS:
            sd.pop(k, None)

    # Special-case: SpatialLM / VLM checkpoints store Sonata under `point_backbone.*`
    pb_candidates = (
        "point_backbone.",
        "module.point_backbone.",
        "model.point_backbone.",
        "model.module.point_backbone.",
        "module.model.point_backbone.",
    )
    for pre in pb_candidates:
        if any(k.startswith(pre) for k in sd.keys()):
            mapped = {k[len(pre) :]: v for k, v in sd.items() if k.startswith(pre)}
            if mapped:
                sd = mapped
            break

    # strip common prefixes (often introduced by DDP or wrapper modules)
    prefixes = ("module.", "sonata.", "encoder.", "model.")
    changed = True
    while changed:
        changed = False
        for pre in prefixes:
            if sd and all(k.startswith(pre) for k in sd.keys()):
                sd = {k[len(pre) :]: v for k, v in sd.items()}
                changed = True
    return sd


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
        sonata_validate: bool | None = None,
        sonata_auto_pad_feat: bool | None = None,
        sonata_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.enable_sonata = bool(enable_sonata)
        self.sonata_mode = str(sonata_mode)
        if self.sonata_mode not in ("off", "projector", "all"):
            raise ValueError(f"Invalid sonata_mode: {self.sonata_mode!r}. Expected off|projector|all")

        self.require_cuda = bool(require_cuda)

        self.sonata_ckpt = _resolve_ckpt_path(sonata_ckpt, kind="Sonata encoder") if sonata_ckpt else None
        self.sonata_projector_ckpt = None

        self.point_feat_dim = int(point_feat_dim)
        self.point_token_cap = int(point_token_cap)

        # match main model toggles (config first, env fallback)
        if sonata_auto_pad_feat is None:
            self.sonata_auto_pad_feat = str(os.environ.get("OPENPI_SONATA_AUTO_PAD_FEAT", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.sonata_auto_pad_feat = bool(sonata_auto_pad_feat)

        if sonata_validate is None:
            self.sonata_validate = str(os.environ.get("OPENPI_SONATA_VALIDATE", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.sonata_validate = bool(sonata_validate)


        # point window ids are usually set by tokenizer (not available here); probe will infer from prompt
        self.point_start_id = None
        self.point_end_id = None

        # projector is not used in this probe-only runner
        self.pc_projector = None
        self._pc_projector_loaded = False

        # cache populated by encode
        self._pt_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        # ---- Build Sonata (optionally from ckpt) ----
        sd = None
        in_ch = int(self.point_feat_dim)
        enable_fourier = False
        mask_token = False

        if self.sonata_ckpt:
            raw = _torch_load_cpu(self.sonata_ckpt)
            sd = _unwrap_state_dict(raw, kind="Sonata encoder", sonata_pretrain=True)
            in_ch_ckpt = _infer_sonata_in_channels(sd)
            if in_ch_ckpt is not None:
                in_ch = int(in_ch_ckpt)
            enable_fourier = _infer_sonata_enable_fourier(sd)
            mask_token = "embedding.mask_token" in sd

        override_fourier = _parse_bool_env("OPENPI_SONATA_ENABLE_FOURIER")
        if override_fourier is not None:
            enable_fourier = bool(override_fourier)

        self.sonata_in_channels = int(in_ch)
        if self.sonata_in_channels < 3:
            raise RuntimeError(f"Invalid point_feat_dim={self.sonata_in_channels} (<3). Expect xyz in first 3 dims.")

        order = _parse_sonata_order_env(default=("z", "z-trans"))

        self.sonata = Sonata(
            in_channels=self.sonata_in_channels,
            order=order,
            mask_token=mask_token,
            enable_fourier_encode=enable_fourier,
        )
        self.sonata.to(dtype=torch.float32)

        if sd is not None:
            info = self.sonata.load_state_dict(sd, strict=False)
            missing = list(getattr(info, "missing_keys", []) or [])
            unexpected_all = list(getattr(info, "unexpected_keys", []) or [])
            unexpected = [k for k in unexpected_all if k not in _SONATA_IGNORED_KEYS]
            if missing:
                raise RuntimeError(
                    "Sonata encoder ckpt is incompatible (missing keys). "
                    f"ckpt={self.sonata_ckpt} missing[:40]={missing[:40]}"
                )
            if unexpected:
                logging.getLogger("openpi").warning(
                    "Sonata encoder ckpt has unexpected keys (ignored). ckpt=%s unexpected[:40]=%s",
                    self.sonata_ckpt,
                    unexpected[:40],
                )

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

        # point_cloud_masks is frame-level throughout the main OpenPI pipeline:
        # shape [B] (or a scalar for a single-sample probe call).
        if pmask is None:
            pmask_b = torch.ones((B,), dtype=torch.bool, device=device)
        else:
            if not isinstance(pmask, torch.Tensor):
                pmask = torch.as_tensor(pmask)
            pmask = pmask.to(device=device)
            if pmask.ndim == 0:
                pmask_b = pmask.to(torch.bool).expand(B)
            elif pmask.ndim == 1:
                if int(pmask.shape[0]) != B:
                    raise RuntimeError(
                        "point_cloud_masks['pointcloud'] must be frame-level shape [B]. "
                        f"Got 1D mask shape={tuple(pmask.shape)} with B={B}."
                    )
                pmask_b = pmask.to(torch.bool)
            else:
                raise RuntimeError(
                    "point_cloud_masks['pointcloud'] must be frame-level shape [B] (or scalar for one sample), "
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

        self.sonata.eval()

        for b in range(B):
            if not pmask_b[b]:
                pt_list.append(None)
                mask_list.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                continue

            arr = pcs[b]
            g = arr[:, :3].to(torch.int64)
            f = arr[:, 3:].to(torch.float32)

            if do_trunc:
                f = f[:, :exp_fd]

            c = f[:, :3].to(torch.float32)

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
            enc_dim = 512

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

    config = _config.cli()

    sonata_mode = _resolve_sonata_mode_from_config_and_env(config.model)
    _enable = getattr(config.model, "enable_sonata", None)
    enable_sonata = True if _enable is None else bool(_enable)
    _require_cuda = getattr(config.model, "require_cuda", None)
    require_cuda = True if _require_cuda is None else bool(_require_cuda)
    point_feat_dim = int(getattr(config.model, "point_feat_dim", 6))
    point_token_cap = int(getattr(config.model, "point_token_cap", 1024))

    # Prefer explicit env override; fallback to config if present.
    sonata_ckpt = os.environ.get("OPENPI_SONATA_CKPT", None) or getattr(config.model, "sonata_ckpt_path", None)

    device = pick_device(require_cuda=require_cuda)
    logger.info(
        "[ProbeRunner] device=%s enable_sonata=%s sonata_mode=%s point_feat_dim=%d point_token_cap=%d sonata_ckpt=%s",
        device,
        enable_sonata,
        sonata_mode,
        point_feat_dim,
        point_token_cap,
        sonata_ckpt,
    )

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    batch = next(iter(loader))
    if not isinstance(batch, (list, tuple)) or len(batch) != 2:
        raise RuntimeError("Data loader must yield (observation, actions).")
    observation, _actions = batch

    probe_model = SonataProbeModel(
        point_feat_dim=point_feat_dim,
        point_token_cap=point_token_cap,
        enable_sonata=enable_sonata,
        sonata_mode=sonata_mode,
        require_cuda=require_cuda,
        sonata_ckpt=sonata_ckpt,
        sonata_validate=getattr(config.model, "sonata_validate", None),
        sonata_auto_pad_feat=getattr(config.model, "sonata_auto_pad_feat", None),
    ).to(device)

    info0 = probe_sonata_integration(probe_model, observation, run_encode=False)

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
        empty_frac = info0.get("empty_window_frac", None)
        if empty_frac is None:
            raise SystemExit(
                "Point window emptiness check could not run because tokenized_prompt_mask is missing "
                "or has an unexpected shape."
            )
        empty_frac = float(empty_frac)
        if empty_frac < 1.0:
            raise SystemExit(
                f"Point window emptiness check FAILED: empty_window_frac={empty_frac}. "
                "Expected 1.0 (no visible tokens may appear between point_start and point_end)."
            )

    info1 = probe_sonata_integration(probe_model, observation, run_encode=True)
    if "encode_error" in info1:
        raise SystemExit(f"Sonata encode FAILED: {info1['encode_error']}")

    logger.info("[ProbeRunner] SUCCESS: point window tokens OK; Sonata encode OK.")


if __name__ == "__main__":
    main()

import logging
import math
import os
from typing import Any
import sys
import importlib
import importlib.util
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models.sonata_encoder import Sonata

_SONATA_IGNORED_KEYS: set[str] = {"embedding.mask_token"}


_SONATA_VALID_ORDERS: set[str] = {"z", "z-trans", "hilbert", "hilbert-trans"}


def _parse_bool_env(name: str) -> bool | None:
    """Parse an optional boolean env var.

    Returns:
      - True/False if env var is set to a recognized value
      - None if env var is not set
    """
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


def _infer_sonata_enable_fourier(sd: dict[str, torch.Tensor]) -> bool:
    """Detect whether ckpt expects enable_fourier_encode=True (input_proj.* present)."""
    return ("input_proj.weight" in sd) or ("input_proj.bias" in sd)


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
        # Older PyTorch (<2.6) does not accept weights_only.
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


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device: torch.device = torch.device("cpu")
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # att_masks 需要是数值型才能 cumsum；统一转 int32
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    # pad_2d 使用逻辑与，避免 bool * bool 的不确定行为
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


_TRANSFORMERS_REPLACE_READY = False


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

    # expand "~"
    p_exp = os.path.expanduser(p)
    cand = Path(p_exp)
    if cand.is_file():
        return str(cand)

    # If relative, try under OPENPI_DATA_HOME
    if not cand.is_absolute():
        dh = _get_openpi_data_home()
        cand2 = dh / cand
        if cand2.is_file():
            return str(cand2)

    raise FileNotFoundError(
        f"{kind} ckpt not found: '{p}'. "
        f"Tried: '{cand}' and '{(_get_openpi_data_home() / cand) if not cand.is_absolute() else cand}'."
    )


def _unwrap_state_dict(
    obj: object,
    *,
    kind: str,
    sonata_pretrain: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Unwrap common checkpoint containers into a plain state_dict."""
    # common wrappers
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "encoder"):
            v = obj.get(k, None)
            if isinstance(v, dict):
                obj = v
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"{kind} checkpoint must be a dict-like state_dict, got {type(obj)}")

    # keep only tensor entries (drop optimizer states etc)
    sd: dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            sd[str(k)] = v

    if not sd:
        # If we filtered everything out, it is likely not a pure state_dict.
        raise RuntimeError(
            f"{kind} checkpoint does not look like a state_dict of tensors. "
            f"Top-level keys preview: {list(obj.keys())[:40] if isinstance(obj, dict) else 'N/A'}"
        )


    # ---- Special-case: facebook/sonata pretrain checkpoints (HF cache) ----
    # Those often store weights under:
    #   - teacher.backbone.*  (EMA weights, preferred for inference) and/or
    #   - student.backbone.*  (training weights)
    # OpenPI Sonata expects keys like `embedding.*` / `enc.*` without those prefixes.
    #
    # For backward-compat / debugging convenience:
    #   If sonata_pretrain is not provided, infer it from `kind`.
    if sonata_pretrain is None:
        sonata_pretrain = "sonata" in str(kind).lower()
    if sonata_pretrain:
        # Handle both plain and DDP-prefixed variants.
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
                mapped = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
                if mapped:
                    selected = mapped
                    break
        if selected is not None:
            sd = selected
        # Benign keys that exist in some pretrain ckpts but are not part of OpenPI's Sonata module.
        for k in _SONATA_IGNORED_KEYS:
            sd.pop(k, None)


    # ---- Special-case: SpatialLM / VLM checkpoints store Sonata under `point_backbone.*` ----
    # Example: manycore-research/SpatialLM1.1-Qwen-0.5B model.safetensors
    # We only need the point backbone weights; drop everything else.
    pb_candidates = (
        "point_backbone.",
        "module.point_backbone.",
        "model.point_backbone.",
        "model.module.point_backbone.",
        "module.model.point_backbone.",
    )
    for pre in pb_candidates:
        if any(k.startswith(pre) for k in sd.keys()):
            mapped = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
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
                sd = {k[len(pre):]: v for k, v in sd.items()}
                changed = True
    return sd


def _ensure_transformers_replace_is_ready() -> None:
    """Ensure OpenPI's patched transformers (transformers_replace) is active.

    We avoid touching site-packages (and uv cache hardlinks) by loading patched
    modules from the repo into sys.modules under the `transformers.*` namespace.

    This approach matches README's intention ("cp overlay") but is runtime-only.
    """
    global _TRANSFORMERS_REPLACE_READY  # noqa: PLW0603
    if _TRANSFORMERS_REPLACE_READY:
        return

    # 1) Hard pin version (the replace files are written against this exact version)
    try:
        import transformers  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        raise ValueError("transformers is required but cannot be imported.") from e

    expected_ver = "4.53.2"
    if getattr(transformers, "__version__", None) != expected_ver:
        raise ValueError(
            f"transformers_replace requires transformers=={expected_ver}, but got transformers=={transformers.__version__}. "
            "Please ensure your environment resolves the pinned version "
            f"(e.g. `uv sync`, or `uv pip install transformers=={expected_ver}`)."
        )

    # 2) Locate replacement sources in this repo
    patch_root = Path(__file__).resolve().parent / "transformers_replace"
    if not patch_root.is_dir():
        raise ValueError(
            f"transformers_replace source directory not found: {patch_root}. "
            "Make sure your repo has src/openpi/models_pytorch/transformers_replace/..."
        )

    def _load_override(fullname: str, file_path: Path) -> None:
        """Load file_path as module fullname and override sys.modules."""
        if not file_path.is_file():
            raise ValueError(f"transformers_replace file missing: {file_path}")

        # If already loaded from our patch_root, skip.
        mod = sys.modules.get(fullname)
        if mod is not None:
            mfile = getattr(mod, "__file__", "") or ""
            if str(patch_root) in str(mfile):
                return

        spec = importlib.util.spec_from_file_location(fullname, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {fullname} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)

        # Attach to parent module for `from transformers.xxx import yyy`.
        try:
            parent_name, leaf = fullname.rsplit(".", 1)
            parent = importlib.import_module(parent_name)
            setattr(parent, leaf, module)
        except Exception:
            # Parent may not exist (rare). Import machinery can still find sys.modules entry.
            pass

    # Load all patched python modules under patch_root into transformers.* namespace.
    # IMPORTANT: skip __init__.py (we do not want to replace package-level lazy import machinery).
    patch_files = [p for p in patch_root.rglob("*.py") if p.is_file() and p.name != "__init__.py"]
    if len(patch_files) == 0:
        raise ValueError(f"No python files found under transformers_replace: {patch_root}")


    def _patch_order(p: Path) -> tuple[int, int, str]:
        """Deterministic order to avoid importing unpatched deps.

        paligemma depends on gemma + siglip, so we must load siglip/gemma first.
        Also load configuration_* before modeling_*.
        """
        rel = p.relative_to(patch_root).as_posix()

        # Group priority (smaller loads earlier)
        if rel.startswith("models/siglip/"):
            grp = 20
        elif rel.startswith("models/gemma/"):
            grp = 30
        elif rel.startswith("models/paligemma/"):
            grp = 40
        elif rel.startswith("models/"):
            grp = 50
        else:
            grp = 10  # core / others first

        name = p.name
        if name.startswith("configuration_"):
            sub = 0
        elif name.startswith("modeling_"):
            sub = 1
        elif name == "check.py":
            sub = 2
        else:
            sub = 1

        return (grp, sub, rel)

    patch_files = sorted(patch_files, key=_patch_order)


    for file_path in patch_files:
        rel = file_path.relative_to(patch_root)
        fullname = "transformers." + rel.with_suffix("").as_posix().replace("/", ".")
        _load_override(fullname, file_path)

    # Minimal sanity check: `check` is a patched-only submodule; import should succeed now.
    try:
        from transformers.models.siglip import check as _check  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers_replace injection failed: cannot import transformers.models.siglip.check. "
            "This usually means the patch directory layout is unexpected."
        ) from e
    

    try:
        ok = bool(_check.check_whether_transformers_replace_is_installed_correctly())
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers_replace sanity check raised an exception. "
            "This usually indicates an import/layout mismatch inside transformers_replace."
        ) from e
    if not ok:
        raise RuntimeError(
            "transformers_replace sanity check returned False. "
            "Most likely some patched modules were not loaded (wrong order) "
            "or the directory structure under transformers_replace does not match transformers==4.53.2."
        )


    _TRANSFORMERS_REPLACE_READY = True


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = bool(getattr(config, "pi05", False))
        
        
        # ---- transformers_replace must be active BEFORE importing gemma_pytorch / transformers models ----
        _ensure_transformers_replace_is_ready()
        from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # noqa: WPS433


        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        _raw_sample_actions = self.sample_actions  # 原始 sample_actions

        def _no_grad_sample_actions(*args, **kwargs):
            with torch.no_grad():
                return _raw_sample_actions(*args, **kwargs)

        disable_compile = (
            os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1"
            or os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "0") == "1"
        )

        if disable_compile:
            self.sample_actions = _no_grad_sample_actions
        else:
            self.sample_actions = torch.compile(_no_grad_sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        # ---- Sonata configuration (default: enabled; mode 'all') ----
        # Treat None as "use default(True)" to avoid bool(None)==False disabling Sonata by accident
        _enable = getattr(config, "enable_sonata", None)
        self.enable_sonata = True if _enable is None else bool(_enable)
        env_new = os.environ.get("OPENPI_SONATA_MODE")
        env_old = os.environ.get("OPENPI_SONATA_TRAIN_MODE")
        if env_new and env_old and env_new.lower() != env_old.lower():
            raise RuntimeError("Conflicting SONATA mode env vars: OPENPI_SONATA_MODE vs OPENPI_SONATA_TRAIN_MODE")
        cfg_mode = getattr(config, "sonata_mode", None)
        legacy_cfg_mode = getattr(config, "sonata_train_mode", None)
        if cfg_mode is not None:
            cfg_mode = str(cfg_mode).strip().lower()
        if legacy_cfg_mode is not None:
            legacy_cfg_mode = str(legacy_cfg_mode).strip().lower()
        if (cfg_mode is not None) and (legacy_cfg_mode is not None) and (cfg_mode != legacy_cfg_mode):
            raise RuntimeError(
                f"Conflicting config Sonata modes: sonata_mode={cfg_mode!r} vs sonata_train_mode={legacy_cfg_mode!r}"
            )
        default_mode = env_new or env_old or "all"
        self.sonata_mode = str(cfg_mode or legacy_cfg_mode or default_mode).strip().lower()
        if self.sonata_mode not in ("off", "projector", "all"):
            raise RuntimeError(f"Invalid sonata_mode: {self.sonata_mode}")
        # Default to True unless explicitly set to False
        _req = getattr(config, "require_cuda", None)
        self.require_cuda = True if _req is None else bool(_req)
        self.sonata_ckpt   = getattr(config, "sonata_ckpt_path", None)
        self.sonata_projector_ckpt = getattr(config, "sonata_projector_ckpt_path", None)
        self.point_start_id = getattr(config, "point_start_id", None)
        self.point_end_id   = getattr(config, "point_end_id", None)
        self.point_token_cap = int(getattr(config, "point_token_cap", 0) or 0)
        self.point_feat_dim  = int(getattr(config, "point_feat_dim", 6) or 6)


        # Sonata真正吃进去的 feature 维度（in_channels）。默认等于 point_feat_dim，
        # 但如果加载了 sonata_ckpt，会从 ckpt 自动推断并覆盖（避免 shape mismatch）。
        self.sonata_in_channels: int = int(self.point_feat_dim)

        # 点云 feature 维度与 sonata_in_channels 不一致时的策略：
        #   - 默认严格：直接报错（推荐，避免吞吐下降/静默质量损失）
        #   - 显式开启 auto-pad：允许 runtime pad/truncate（调试/兼容用）
        _ap = getattr(config, "sonata_auto_pad_feat", None)
        if _ap is None:
            self.sonata_auto_pad_feat = str(os.environ.get("OPENPI_SONATA_AUTO_PAD_FEAT", "0")).strip().lower() in (
                "1", "true", "yes", "on"
            )
        else:
            self.sonata_auto_pad_feat = bool(_ap)
        self._warned_point_feat_mismatch = False


        # Expensive point-cloud validation checks can hurt throughput on large point clouds.
        # Keep OFF by default (OpenPI-style). Enable only for debugging data issues:
        #   - config.sonata_validate = True
        #   - or env OPENPI_SONATA_VALIDATE=1
        _val = getattr(config, "sonata_validate", None)
        if _val is None:
            self.sonata_validate = str(os.environ.get("OPENPI_SONATA_VALIDATE", "0")).strip().lower() in ("1", "true", "yes", "on")
        else:
            self.sonata_validate = bool(_val)


        # 允许通过环境变量覆盖 ckpt 路径（显式 config 优先）
        if (self.sonata_ckpt is None) and (os.environ.get("OPENPI_SONATA_CKPT")):
            self.sonata_ckpt = os.environ.get("OPENPI_SONATA_CKPT")
        if (self.sonata_projector_ckpt is None) and (os.environ.get("OPENPI_SONATA_PROJECTOR_CKPT")):
            self.sonata_projector_ckpt = os.environ.get("OPENPI_SONATA_PROJECTOR_CKPT")
        self._pc_projector_loaded = False

        # Resolve ckpt paths (OpenPI-style: allow relative paths under OPENPI_DATA_HOME).
        # Fail-fast here so errors are clear before training starts.
        self.sonata_ckpt = _resolve_ckpt_path(self.sonata_ckpt, kind="Sonata encoder")
        self.sonata_projector_ckpt = _resolve_ckpt_path(self.sonata_projector_ckpt, kind="Sonata projector")


        # 早期 fail-fast：显式设置了窗口 ID 但与 tokenizer 末两位不一致 → 立刻报错
        try:
            vsz = self.paligemma_with_expert.paligemma.language_model.get_input_embeddings().num_embeddings
            exp_start, exp_end = int(vsz - 2), int(vsz - 1)
            # If not provided, default to the last two vocab IDs (matches PaligemmaTokenizer behavior).
            if (self.point_start_id is None) and (self.point_end_id is None):
                self.point_start_id, self.point_end_id = exp_start, exp_end
            elif (self.point_start_id is None) != (self.point_end_id is None):
                raise RuntimeError(
                    "point_start_id and point_end_id must be set together (or both left as None)."
                )
            else:
                if (int(self.point_start_id) != exp_start) or (int(self.point_end_id) != exp_end):
                    raise RuntimeError(
                        f"point_start_id/point_end_id mismatch tokenizer: got ({self.point_start_id},{self.point_end_id}), "
                        f"expected ({exp_start},{exp_end}). 请将其设置为 vocab_size-2 / vocab_size-1。"
                    )
        except Exception:
            # 少数情况下 language_model 可能无 get_input_embeddings；跳过早检，后续 embed_prefix 仍会严格校验
            pass
        # Lazily constructed encoder and projector
        self.sonata: Sonata | None = None
        self.pc_projector: nn.Linear | None = None
        # Cache for current batch (raw features, before projection)
        self._pt_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        # Log once: whether Sonata encoder was loaded from ckpt or randomly initialized.
        self._sonata_init_logged: bool = False
        # Track whether pc_projector was materialized explicitly (useful for DDP safety diagnostics)
        self._pc_projector_materialized: bool = False

    def materialize_sonata_projector(self, *, enc_dim: int | None = None) -> None:
        """Create pc_projector as a real parameter BEFORE any DDP/FSDP wrapping.

        Why:
          - pc_projector is otherwise created lazily in embed_prefix(), which can silently break DDP/FSDP
            if the parameter is created after the model is wrapped.

        Args:
          enc_dim: input dim of point tokens (Sonata output dim). If None:
            - use cached pt_feat_raw dim if available
            - else fallback to 512 (matches existing fallback in torch_sonata_encode_batch)
        """
        if not (self.enable_sonata and self.sonata_mode in ("projector", "all")):
            return
        if self.pc_projector is not None:
            self._pc_projector_materialized = True
            return

        # Infer language embedding dim from PaliGemma embeddings.
        try:
            emb = self.paligemma_with_expert.paligemma.language_model.get_input_embeddings()
            lang_emb_dim = int(getattr(emb, "embedding_dim", emb.weight.shape[1]))
        except Exception as e:
            raise RuntimeError("Failed to infer language embedding dim for pc_projector materialization.") from e

        if enc_dim is None:
            if self._pt_cache is not None:
                enc_dim = int(self._pt_cache[0].shape[-1])
            else:
                # Sonata 默认 out dim 为 512（enc_channels[-1]）。
                enc_dim = 512

        # Create on the same device/dtype as the language embedding table.
        # Why not `next(self.parameters())`? Because some models keep certain params in fp32
        # (e.g. LayerNorm / embeddings policy or vision tower), and `next(self.parameters())`
        # may pick a fp32 param even when the LM embeddings are bf16, causing
        # bf16(lang_emb) x fp32(weight) dtype mismatches in F.linear.
        try:
            ref_w = emb.weight
            ref_device = ref_w.device
            ref_dtype = ref_w.dtype
        except Exception:
            p = next(self.parameters())
            ref_device = p.device
            ref_dtype = p.dtype
        self.pc_projector = nn.Linear(int(enc_dim), int(lang_emb_dim), bias=False).to(
            device=ref_device, dtype=ref_dtype
        )

        # If a projector ckpt is provided, load weights now (shape-checked inside).
        self._maybe_load_pc_projector()
        self._pc_projector_materialized = True

    def load_state_dict(self, state_dict: Any, strict: bool = True):
        """Make lazy Sonata/projector modules checkpoint-friendly.

        OpenPI's training/inference code may load a full-model state_dict before the first forward.
        Since pc_projector/sonata are created lazily in forward paths, we pre-create them if the
        incoming state_dict contains their keys; otherwise those weights would be dropped (strict=False)
        or raise (strict=True).
        """
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        # Common wrapper: {"state_dict": {...}}
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        # Strip common global prefixes (DDP/DataParallel and common wrapper "model.")
        # Use a loop so it also handles multi-layer prefixes like "model.module."
        prefixes = ("module.", "model.")
        changed = True
        while changed and state_dict:
            changed = False
            for pre in prefixes:
                if all(isinstance(k, str) and k.startswith(pre) for k in state_dict.keys()):
                    state_dict = {k[len(pre):]: v for k, v in state_dict.items()}
                    changed = True

        # --- Pre-create pc_projector if present in ckpt ---
        w_key = None
        for k in state_dict.keys():
            if isinstance(k, str) and k.endswith("pc_projector.weight"):
                w_key = k
                break
        if (w_key is not None) and (self.pc_projector is None):
            w = state_dict[w_key]
            if (not isinstance(w, torch.Tensor)) or (w.ndim != 2):
                raise RuntimeError(f"{w_key} must be a 2D tensor, got type={type(w)} shape={getattr(w, 'shape', None)}")
            in_dim = int(w.shape[1])
            out_dim = int(w.shape[0])
            self.pc_projector = nn.Linear(in_dim, out_dim, bias=False)
            p = next(self.parameters())
            self.pc_projector.to(device=p.device, dtype=p.dtype)

        # --- Pre-create Sonata if present in ckpt ---
        has_sonata = any(isinstance(k, str) and k.startswith("sonata.") for k in state_dict.keys())
        if has_sonata and (self.sonata is None):
            # Infer in_channels from ckpt if possible (avoids shape mismatch like [48,9] vs [48,6])
            sonata_sd = {k[len("sonata."):]: v for k, v in state_dict.items()
                         if isinstance(k, str) and k.startswith("sonata.") and isinstance(v, torch.Tensor)}
            in_ch = _infer_sonata_in_channels(sonata_sd)
            if in_ch is None:
                in_ch = int(self.point_feat_dim)
            enable_fourier = _infer_sonata_enable_fourier(sonata_sd)
            override_fourier = _parse_bool_env("OPENPI_SONATA_ENABLE_FOURIER")
            if override_fourier is not None:
                enable_fourier = bool(override_fourier)
            mask_token = "embedding.mask_token" in sonata_sd
            self.sonata_in_channels = int(in_ch)
            self.sonata = Sonata(
                in_channels=self.sonata_in_channels,
                order=_parse_sonata_order_env(default=("z", "z-trans")),
                mask_token=mask_token,
                enable_fourier_encode=enable_fourier,
            )
            p = next(self.parameters())
            # Keep Sonata in fp32 by design
            self.sonata.to(device=p.device, dtype=torch.float32)

        out = super().load_state_dict(state_dict, strict=strict)
        # If weights came from the main state_dict, don't override later via external projector ckpt.
        # Also covers the case where pc_projector was materialized BEFORE load_state_dict().
        #
        # IMPORTANT: decide based on actual load result to avoid false positives when ckpt keys
        # use unexpected prefixes (e.g. "model.pc_projector.weight") and strict=False.
        if self.pc_projector is not None:
            missing = set(getattr(out, "missing_keys", []) or [])
            if "pc_projector.weight" not in missing:
                self._pc_projector_loaded = True
        return out
        

    def _ensure_sonata_ready(self, device):
        if not self.enable_sonata or self.sonata_mode == "off":
            raise RuntimeError("Sonata is disabled but encoder was requested.")
        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Sonata, but CUDA is not available.")
        if self.sonata is None:
            sd = None
            in_ch = int(self.point_feat_dim)
            if self.sonata_ckpt:
                ckpt_path = str(self.sonata_ckpt)
                if not self._sonata_init_logged:
                    logging.getLogger("openpi").info("[Sonata] loading encoder ckpt: %s", ckpt_path)
                try:
                    raw = _torch_load_cpu(ckpt_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load Sonata encoder ckpt: {ckpt_path} ({e})") from e
                sd = _unwrap_state_dict(raw, kind="Sonata encoder", sonata_pretrain=True)
                in_ch_ckpt = _infer_sonata_in_channels(sd)
                if in_ch_ckpt is not None:
                    in_ch = int(in_ch_ckpt)
                else:
                    logging.getLogger("openpi").warning(
                        "Could not infer Sonata in_channels from ckpt; falling back to config.point_feat_dim=%d. ckpt=%s",
                        in_ch, ckpt_path,
                    )
            else:
                if not self._sonata_init_logged:
                    logging.getLogger("openpi").warning(
                        "[Sonata] sonata_ckpt_path/OPENPI_SONATA_CKPT not set -> encoder will be randomly initialized."
                    )

            if in_ch < 3:
                raise RuntimeError(f"Invalid Sonata in_channels={in_ch} (must be >=3; expects xyz in first 3 dims).")

            self.sonata_in_channels = int(in_ch)
            if self.sonata_in_channels != int(self.point_feat_dim):
                logging.getLogger("openpi").warning(
                    "Sonata ckpt expects in_channels=%d but config.point_feat_dim=%d. "
                    "Model will build Sonata(in_channels=%d). Ensure your pointcloud feature dims match, "
                    "or set OPENPI_SONATA_AUTO_PAD_FEAT=1 to pad/truncate at runtime (slower).",
                    self.sonata_in_channels, int(self.point_feat_dim), self.sonata_in_channels,
                )

            enable_fourier = _infer_sonata_enable_fourier(sd) if sd is not None else False
            override_fourier = _parse_bool_env("OPENPI_SONATA_ENABLE_FOURIER")
            if override_fourier is not None:
                enable_fourier = bool(override_fourier)
            mask_token = bool(sd is not None and ("embedding.mask_token" in sd))
            self.sonata = Sonata(
                in_channels=self.sonata_in_channels,
                order=_parse_sonata_order_env(default=("z", "z-trans")),
                mask_token=mask_token,
                enable_fourier_encode=enable_fourier,
            )

            if sd is not None:
                info = self.sonata.load_state_dict(sd, strict=False)  # returns _IncompatibleKeys
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
            # 始终使用 float32 运行点云编码器，避免与 bf16/fp16 主干产生 dtype 冲突
            self.sonata.to(device=device, dtype=torch.float32)  # 训练/评估态在 encode 时按 train 切换

            # Projector mode semantics: freeze the encoder parameters as soon as the encoder exists.
            # This avoids accidentally training the encoder (and accidentally including it in optimizers)
            # when only the projector should be trained.
            if self.sonata_mode == "projector":
                for p in self.sonata.parameters():
                    p.requires_grad_(False)

            if not self._sonata_init_logged:
                self._sonata_init_logged = True



    def _maybe_load_pc_projector(self) -> None:
        """Optionally load a pretrained projector (adapter) once it exists.

        Supports:
          - torch load (.pt/.pth) dicts
          - safetensors (.safetensors) dicts
        Accepts keys:
          - "pc_projector.weight" (full model state_dict)
          - "sonata_projector.weight"
          - "weight" (projector-only checkpoint)
          - any key that endswith("pc_projector.weight") / endswith("sonata_projector.weight")
        """
        if self._pc_projector_loaded:
            return
        if self.pc_projector is None:
            return
        if not self.sonata_projector_ckpt:
            return

        ckpt_path = str(self.sonata_projector_ckpt)
        try:
            sd = _torch_load_cpu(ckpt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Sonata projector ckpt: {ckpt_path} ({e})") from e

        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise RuntimeError(f"Projector ckpt must be a dict/state_dict, but got type={type(sd)} from {ckpt_path}")

        w = None
        for k in ("pc_projector.weight", "sonata_projector.weight", "weight"):
            if k in sd:
                w = sd[k]
                break
        if w is None:
            for k, v in sd.items():
                if k.endswith("pc_projector.weight") or k.endswith("sonata_projector.weight"):
                    w = v
                    break
        if w is None:
            keys_preview = list(sd.keys())[:40]
            raise RuntimeError(
                f"Projector ckpt {ckpt_path} does not contain projector weights. "
                f"Expected one of: pc_projector.weight / sonata_projector.weight / weight. keys[:40]={keys_preview}"
            )

        if tuple(w.shape) != tuple(self.pc_projector.weight.shape):
            raise RuntimeError(
                f"Projector weight shape mismatch: ckpt={tuple(w.shape)} vs expected={tuple(self.pc_projector.weight.shape)}. "
                "This usually means you changed the language hidden size (paligemma_variant) or Sonata enc dim."
            )
        with torch.no_grad():
            self.pc_projector.weight.copy_(
                w.to(device=self.pc_projector.weight.device, dtype=self.pc_projector.weight.dtype)
            )
        self._pc_projector_loaded = True


    def set_point_cache(self, pt_feat_raw: torch.Tensor, pt_mask: torch.Tensor) -> None:
        if pt_feat_raw.ndim != 3 or pt_mask.ndim != 2 or pt_feat_raw.shape[:2] != pt_mask.shape[:2]:
            raise RuntimeError(f"Invalid pt shapes: {pt_feat_raw.shape=}, {pt_mask.shape=}")
        self._pt_cache = (pt_feat_raw, pt_mask)

    def torch_sonata_encode_batch(self, observation, train: bool = False):
        if not self.enable_sonata or self.sonata_mode == "off":
            raise RuntimeError("Sonata is disabled (enable_sonata=False or sonata_mode=off).")
        if "point_clouds" not in observation.__dict__ or not observation.point_clouds:
            raise RuntimeError("Sonata enabled but observation.point_clouds is missing or empty.")
        if "pointcloud" not in observation.point_clouds:
            raise RuntimeError("Sonata enabled but observation.point_clouds['pointcloud'] is missing.")

        p0 = next(self.parameters())
        device = p0.device
        dtype_model = p0.dtype
        self._ensure_sonata_ready(device)
        # 与 backup 一致：按需开关训练态 + 梯度
        self.sonata.train(bool(train))

        pcs   = observation.point_clouds["pointcloud"]
        pmask = observation.point_cloud_masks.get("pointcloud", None)
        if not isinstance(pcs, torch.Tensor):
            pcs = torch.as_tensor(pcs)
        # IMPORTANT: keep point cloud in float32 to avoid bf16/fp16 losing integer precision
        # for grid coordinates in the first 3 dims.
        pcs = pcs.to(device=device, dtype=torch.float32)
        B, M, D = pcs.shape
        if D < 6:
            raise RuntimeError(f"pointcloud last dim {D} < 6; expect [3 grid | 3 xyz | extras].")
        # point_cloud_masks is a frame-level availability mask throughout the
        # OpenPI pipeline: shape [B] (or scalar for a single-sample call).
        if pmask is None:
            pmask = torch.ones((B,), dtype=torch.bool, device=device)
        else:
            if not isinstance(pmask, torch.Tensor):
                pmask = torch.as_tensor(pmask, device=device)
            else:
                pmask = pmask.to(device=device)
            if pmask.ndim == 0:
                pmask = pmask.to(dtype=torch.bool).expand(B)
            elif pmask.ndim == 1:
                if int(pmask.shape[0]) != B:
                    raise RuntimeError(
                        "point_cloud_masks['pointcloud'] must be frame-level shape [B]. "
                        f"Got 1D mask shape={tuple(pmask.shape)} with B={B}."
                    )
                pmask = pmask.to(dtype=torch.bool)
            else:
                raise RuntimeError(
                    "point_cloud_masks['pointcloud'] must be frame-level shape [B] (or scalar for one sample), "
                    f"but got {tuple(pmask.shape)}."
                )

        grid = pcs[..., :3]
        if self.sonata_validate:
            # NOTE: full-tensor reductions; expensive for large point clouds.
            if torch.isnan(pcs).any() or torch.isinf(pcs).any():
                raise RuntimeError("Point cloud contains NaN/Inf.")
            if (grid < 0).any():
                raise RuntimeError("Point grid (first 3 dims) must be non-negative.")
        obs_fd = int(D - 3)
        exp_fd = int(getattr(self, "sonata_in_channels", self.point_feat_dim))
        if obs_fd < 3:
            raise RuntimeError(
                f"pointcloud feature dims (after first 3 grid dims) must be >=3 (xyz), got {obs_fd}."
            )
        if exp_fd < 3:
            raise RuntimeError(f"sonata_in_channels must be >=3, got {exp_fd}.")
        if obs_fd != exp_fd:
            if not self.sonata_auto_pad_feat:
                raise RuntimeError(
                    f"pointcloud feature dim mismatch: observation has {obs_fd} dims (pcs[...,3:]) "
                    f"but Sonata expects {exp_fd} dims (from ckpt embedding.stem.linear.weight). "
                    "Fix by providing exactly exp_fd dims (and set config.point_feat_dim accordingly), "
                    "OR set OPENPI_SONATA_AUTO_PAD_FEAT=1 to allow runtime pad/truncate (slower, may hurt quality)."
                )
            if not self._warned_point_feat_mismatch:
                logging.getLogger("openpi").warning(
                    "pointcloud feature dim mismatch: obs=%d vs sonata_in_channels=%d; "
                    "will %s at runtime (OPENPI_SONATA_AUTO_PAD_FEAT=1).",
                    obs_fd, exp_fd, "pad zeros" if obs_fd < exp_fd else "truncate extras",
                )
                self._warned_point_feat_mismatch = True
        do_trunc = obs_fd > exp_fd
        do_pad   = obs_fd < exp_fd

        cap = int(self.point_token_cap)
        if cap <= 0:
            raise RuntimeError("point_token_cap must be > 0 when Sonata is enabled.")

        pt_list: list[torch.Tensor | None] = []
        mask_list: list[torch.Tensor] = []
        enc_dim: int | None = None
        # 仅对前向调用开启/关闭 grad；其余逻辑保持不变
        with torch.set_grad_enabled(bool(train)):
            for b in range(B):
                # 允许该样本没有点云（mask=False）：不编码，后续插入时将插入 0 个点 token。
                if not pmask[b]:
                    pt_list.append(None)
                    mask_list.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                    continue

                arr = pcs[b]
                g = arr[:, :3].to(torch.int64)
                f = arr[:, 3:].to(dtype=torch.float32)


                if do_trunc:
                    f = f[:, :exp_fd]
                c = f[:, :3].to(dtype=torch.float32)
                # Zero-padded rows are treated as padding and removed before Sonata encoding.
                pad = (g == 0).all(dim=1) & (c == 0).all(dim=1) & (f == 0).all(dim=1)
                g = g[~pad]
                c = c[~pad]
                f = f[~pad]
                n = g.shape[0]
                # 允许 padding 后为空：当作“无点云”处理。
                if n == 0:
                    pt_list.append(None)
                    mask_list.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                    continue

                # Fail-fast for a common silent error: degenerate/placeholder grid_coord.
                # If xyz spans a real volume but grid_coord collapses to a single value, Sonata voxelization degenerates.
                if n >= 256:
                    aabb = c.max(dim=0).values - c.min(dim=0).values
                    if (aabb > 0.1).any():
                        g_min = g.min(dim=0).values
                        g_max = g.max(dim=0).values
                        if torch.equal(g_min, g_max):
                            raise RuntimeError(
                                "[torch_sonata_encode_batch] grid_coord appears degenerate/constant while xyz spans a volume. "
                                "This is usually caused by DepthToPointCloud emitting placeholder zeros for grid_coord. "
                                "Fix the transform to compute real grid_coord (voxel quantization), or omit grid_coord and let Sonata derive it."
                            )

                if do_pad and f.numel() != 0:
                    # Pad missing channels with zeros AFTER removing padded points (cheaper than padding full M).
                    pad_ch = exp_fd - f.shape[1]
                    if pad_ch > 0:
                        f = torch.cat(
                            [f, torch.zeros((f.shape[0], pad_ch), dtype=f.dtype, device=f.device)],
                            dim=1,
                        )
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
                    raise RuntimeError(f"[Sonata] inconsistent token dim: {int(enc.size(1))} vs {enc_dim}.")

                if real < cap:
                    pad_zeros = torch.zeros(cap - real, enc_dim, dtype=enc.dtype, device=enc.device)
                    enc = torch.cat([enc, pad_zeros], dim=0)
                elif real >= int(0.95 * cap):
                    logging.getLogger("openpi").warning("[Sonata] token_len=%d is close to cap (%d).", real, cap)
                m = torch.zeros((cap,), dtype=torch.bool, device=device)
                m[:real] = True
                pt_list.append(enc.to(dtype=dtype_model))
                mask_list.append(m)

        # 若整 batch 都没有点云（全部 mask=False 或全部为空），我们仍需要返回一个固定形状。
        if enc_dim is None:
            # 若整个 batch 都没有点云（全 mask=False / 空），尽量复用上一次缓存维度；否则回退到 Sonata 默认 512。
            if self._pt_cache is not None:
                enc_dim = int(self._pt_cache[0].shape[-1])
            else:
                # Sonata 默认 out dim 为 512（enc_channels[-1]）。
                enc_dim = 512

        # 用零填充所有“无点云”样本的 pt features；mask 保持全 False。
        for i, v in enumerate(pt_list):
            if v is None:
                pt_list[i] = torch.zeros((cap, enc_dim), dtype=dtype_model, device=device)

        # All entries should be tensors after the fill above; keep batch dimension aligned with pt_mask.
        pt_list_t: list[torch.Tensor] = [v for v in pt_list if v is not None]
        if len(pt_list_t) != B:
            raise RuntimeError("Internal error: pt_list contains None after fill; batch alignment would break.")
        pt_feat_raw = torch.stack(pt_list_t, dim=0)
        pt_mask     = torch.stack(mask_list, dim=0)
        self._pt_cache = (pt_feat_raw, pt_mask)
        return self._pt_cache

    @staticmethod
    def _insert_points_torch(
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        token_ids: torch.Tensor,
        pt_emb: torch.Tensor,
        pt_mask: torch.Tensor,
        start_id: int,
        end_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, E = text_emb.shape
        if token_ids.dtype != torch.long:
            raise RuntimeError("token_ids must be torch.long for embedding/indexing.")
        starts = (token_ids == start_id).nonzero(as_tuple=False)
        ends   = (token_ids == end_id).nonzero(as_tuple=False)
        s_pos = torch.full((B,), -1, device=token_ids.device, dtype=torch.long)
        e_pos = torch.full((B,), -1, device=token_ids.device, dtype=torch.long)
        for b in range(B):
            s_idx = starts[starts[:,0]==b][:,1]
            e_idx = ends[ends[:,0]==b][:,1]
            if s_idx.numel() != 1 or e_idx.numel() != 1:
                # Better diagnostics: tells you WHICH sample is bad and how.
                raise RuntimeError(
                    f"Point window token error in sample b={b}: "
                    f"expected exactly one start_id={start_id} and one end_id={end_id}, "
                    f"but got start_count={int(s_idx.numel())}, end_count={int(e_idx.numel())}. "
                    "Check your prompt template / tokenizer special tokens insertion."
                )
            s_pos[b] = s_idx.item()
            e_pos[b] = e_idx.item()
            if not (0 <= s_pos[b] < e_pos[b] < S):
                raise RuntimeError(f"Invalid window positions: start={s_pos[b]}, end={e_pos[b]}, S={S}")
            mid_visible = text_mask[b, s_pos[b].item() + 1 : e_pos[b].item()]
            if bool(mid_visible.any().item()):
                raise RuntimeError(
                    f"Visible tokens found between point_start_id={start_id} and point_end_id={end_id} "
                    f"in sample b={b}. The point window must be empty."
                )
        fused_embs, fused_masks = [], []
        for b in range(B):
            s = s_pos[b].item(); e = e_pos[b].item()
            k = int(pt_mask[b].sum().item())
            # 允许 k==0：该样本没有点云 token（例如 dropout 或 depth 全无效）。
            left_emb  = text_emb[b, :s+1, :]
            left_m    = text_mask[b, :s+1]
            mid_emb   = pt_emb[b, :k, :]
            mid_m     = pt_mask[b, :k]
            right_emb = text_emb[b, e:, :]
            right_m   = text_mask[b, e:]
            fused_embs.append(torch.cat([left_emb, mid_emb, right_emb], dim=0))
            fused_masks.append(torch.cat([left_m,   mid_m,   right_m  ], dim=0))
        L = max(x.shape[0] for x in fused_embs)
        out_emb = text_emb.new_zeros((B, L, E))
        out_msk = text_mask.new_zeros((B, L), dtype=torch.bool)
        for b in range(B):
            n = fused_embs[b].shape[0]
            out_emb[b, :n, :] = fused_embs[b]
            out_msk[b, :n]    = fused_masks[b]
        return out_emb, out_msk


    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        # --- Sonata point cloud insertion (projector/all) ---
        if self.enable_sonata and self.sonata_mode in ("projector", "all"):
            if self.point_start_id is None or self.point_end_id is None:
                raise RuntimeError("point_start_id/point_end_id must be set when Sonata insertion is enabled.")
            if self._pt_cache is None:
                raise RuntimeError("Sonata mode in {'projector','all'} requires torch_sonata_encode_batch() before embed_prefix.")
            pt_feat_raw, pt_mask = self._pt_cache
            lang_emb_dim = lang_emb.shape[-1]
            if self.pc_projector is None:
                # DDP/FSDP safety: creating parameters lazily after wrapping will silently break training.
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    raise RuntimeError(
                        "pc_projector is None inside embed_prefix() under distributed training. "
                        "This indicates the projector would be created lazily AFTER DDP/FSDP wrapping, which is unsafe. "
                        "Call model.materialize_sonata_projector(enc_dim=...) BEFORE wrapping the model with DDP/FSDP."
                    )
                self.pc_projector = nn.Linear(pt_feat_raw.shape[-1], lang_emb_dim, bias=False).to(
                    device=lang_emb.device, dtype=lang_emb.dtype
                )
            # load once (if ckpt provided) as soon as projector exists
            self._maybe_load_pc_projector()
            # NOTE: torch.nn.Linear requires mat1/mat2 to have the same dtype.
            # pt_feat_raw may be float32 (from Sonata encoder) while lang_emb may be bf16/fp16.
            # Project in the projector's weight dtype, then cast back to lang_emb.dtype for fusion.
            pt_in = pt_feat_raw.to(device=lang_emb.device, dtype=self.pc_projector.weight.dtype)
            pt_emb = self.pc_projector(pt_in).to(dtype=lang_emb.dtype)
            lang_emb, lang_masks = self._insert_points_torch(
                lang_emb, lang_masks, lang_tokens, pt_emb, pt_mask, int(self.point_start_id), int(self.point_end_id)
            )

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.int32, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.int32, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        # --- Sonata encode: projector=freeze encoder grads; all=train encoder+projector
        if self.enable_sonata and self.sonata_mode in ("projector", "all"):
            self.torch_sonata_encode_batch(
                observation,
                train=(self.sonata_mode == "all"),
            )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        # 统一将输入 embedding 的 dtype 对齐到底模权重 dtype（支持 bf16/fp16；fp32 情况不触发）
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype in (torch.bfloat16, torch.float16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
            suffix_embs = suffix_embs.to(dtype=model_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks.to(torch.int64), dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        if self.enable_sonata and self.sonata_mode in ("projector","all"):
            self.torch_sonata_encode_batch(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        # 对齐 prefix_embs 的 dtype 到底模权重 dtype（支持 bf16/fp16；fp32 情况不触发）
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype in (torch.bfloat16, torch.float16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=prefix_embs.dtype)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # 位置编码：均使用 int64（HF 期望 long）
        prefix_offsets = prefix_pad_masks.to(torch.int64).sum(dim=-1, keepdim=True)
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int64), dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks).to(dtype=suffix_embs.dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # 与训练路径一致：对齐 suffix_embs dtype 到底模权重 dtype（bf16/fp16；fp32 情况不触发）
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype in (torch.bfloat16, torch.float16):
            suffix_embs = suffix_embs.to(dtype=model_dtype)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


    # --- Accessors for training-time utilities (no behavior change) ---
    def get_torch_sonata(self):
        """Return Sonata encoder module if constructed (may be None)."""
        return self.sonata

    def get_torch_sonata_projector(self):
        """Return point projector (Linear) if constructed (may be None)."""
        return self.pc_projector

    @property
    def sonata_projector(self):
        return self.pc_projector

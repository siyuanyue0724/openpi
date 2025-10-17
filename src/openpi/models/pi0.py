import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
import numpy as np
import os
import inspect

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

# 可选依赖（启用后缺失即 fail fast）
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore
try:
    import openpi.models.sonata_encoder as sonata_encoder  # type: ignore
except Exception:
    sonata_encoder = None  # type: ignore
# 回调：优先 pure_callback；其次 io_callback；若均缺失且启用点云→fail fast
try:
    pure_callback = jax.pure_callback  # type: ignore[attr-defined]
except Exception:
    try:
        from jax.experimental import io_callback as pure_callback  # type: ignore
    except Exception:
        pure_callback = None  # type: ignore
try:
    from openpi.models import PointBackboneType as _PBT  # type: ignore
except Exception:
    _PBT = None  # type: ignore

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

class _TorchSonataRunner:
    """Host 侧运行 Sonata（PyTorch），通过 JAX 回调整 batch 取特征（fail fast）。"""
    def __init__(
        self,
        *,
        point_feat_dim: int,
        point_token_cap: int,
        point_cfg: dict | None,
        use_pretrained: bool,
        ckpt_path: str | None,
        require_cuda: bool,
    ):
        if sonata_encoder is None or torch is None:
            raise ImportError("Sonata/Torch 不可用（启用了点云分支，fail fast）。")
        # 默认配置（可被 point_cfg 覆盖）
        sp_cfg = dict(
            in_channels=point_feat_dim,
            order=("z", "z-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 12, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            mlp_ratio=4.0,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=True,
            enc_mode=True,  # 有些实现是 bool，有些是 'voxel'/'point'
            enable_fourier_encode=True,  # 与 backup 对齐：默认开启 Fourier 编码
            upcast_attention=False,
            upcast_softmax=False,
            mask_token=True,  # 与 backup 对齐
            num_bins=1280,
        )
        if point_cfg:
            sp_cfg.update(point_cfg)
        self._cap = int(point_token_cap)
        self._enc_out_dim = int(sp_cfg["enc_channels"][-1])
        self._in_channels = int(sp_cfg.get("in_channels", point_feat_dim))

        # 设备选择：require_cuda=True 且无 CUDA → 直接报错（fail fast）
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if require_cuda and dev != "cuda":
            raise RuntimeError("require_cuda=True，但未检测到 CUDA（fail fast）。")
        self._device = torch.device(dev)

        # 使用 inspect.signature 更稳健地判断参数可用性
        try:
            sig = inspect.signature(sonata_encoder.Sonata)  # type: ignore[attr-defined]
            params = sig.parameters
        except Exception:
            params = {}
        if isinstance(sp_cfg.get("enc_mode"), bool):
            if ("enc_mode" in params) and (
                (getattr(params["enc_mode"], "annotation", None) is str)
                or isinstance(getattr(params["enc_mode"], "default", None), str)
            ):
                sp_cfg["enc_mode"] = "voxel" if sp_cfg["enc_mode"] else "point"
        if "enable_flash" in params:
            # 环境自适配：仅在 GPU + 实现支持时开启 flash；否则显式关闭，避免 CPU 场景下的构造/前向报错
            sp_cfg["enable_flash"] = bool(torch.cuda.is_available()) and hasattr(sonata_encoder, "flash_attn")
        else:
            sp_cfg.pop("enable_flash", None)
        # 兼容不同 Sonata 版本：若构造签名里没有这些参数，则移除
        if "enable_fourier_encode" not in params:
            sp_cfg.pop("enable_fourier_encode", None)
        if "mask_token" not in params:
            sp_cfg.pop("mask_token", None)

        # 构造 & 可选加载权重（权重缺失仅告警，不属于数据校验的严格性）
        self._inner = sonata_encoder.Sonata(**sp_cfg).to(self._device).eval()  # type: ignore
        if use_pretrained:
            self._maybe_load_pretrained(ckpt_path)

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def enc_out_dim(self) -> int:
        return self._enc_out_dim

    @property
    def in_channels(self) -> int:
        return self._in_channels

    def _maybe_load_pretrained(self, ckpt_path: str | None) -> None:
        cands: list[str] = []
        if ckpt_path:
            cands.append(ckpt_path)
        envp = os.getenv("OPENPI_SONATA_CKPT", "").strip()
        if envp:
            cands.append(envp)
        try:
            here = os.path.abspath(__file__)
            p = os.path.dirname(here)
            for _ in range(4):
                p = os.path.dirname(p)
                cands.append(os.path.join(p, "pretrain", "SpatialLM_Sonata_encoder.pth"))
        except Exception:
            pass
        path = None
        for c in cands:
            if isinstance(c, str) and c and os.path.isfile(c):
                path = c
                break
        if path is None:
            logger.warning("Sonata 预训练权重未找到，使用随机初始化（仅警告，不阻断）。")
            return
        try:
            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.0
            except TypeError:
                sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = self._inner.load_state_dict(sd, strict=False)  # type: ignore
            if missing or unexpected:
                logger.warning("加载 Sonata 权重时存在 missing=%d / unexpected=%d 键。", len(missing), len(unexpected))
            logger.info("已加载 Sonata 权重：%s", str(path))
        except Exception as e:
            logger.warning("Sonata 权重加载失败（忽略并继续）：%s", e)

    # —— 严格清洗（fail fast）：发现异常直接抛错；允许剔除全零 padding 行 ——
    def _sanitize_sample_arrays(
        self, coord: np.ndarray, feat: np.ndarray, grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1) NaN/Inf：严格报错
        if not (np.isfinite(coord).all() and np.isfinite(feat).all()):
            raise ValueError("[Sonata strict] 检测到 NaN/Inf。")
        # 2) 网格坐标必须非负
        if (grid < 0).any():
            raise ValueError("[Sonata strict] grid_coord 含负值（体素索引必须非负）。")
        # 3) 剔除显著 padding（grid/coord/feat 全零行）
        pad = (grid == 0).all(axis=1) & (coord == 0).all(axis=1) & (feat == 0).all(axis=1)
        if pad.any():
            keep = ~pad
            coord, feat, grid = coord[keep], feat[keep], grid[keep]
        # 4) feats[:,:3] 与 coord 一致性（L∞ < 1e-4）
        if feat.shape[1] >= 3 and coord.shape[0] > 0:
            err = float(np.max(np.abs(feat[:, :3] - coord)))
            if not np.isfinite(err) or err > 1e-4:
                raise ValueError("[Sonata strict] 契约违反：feat[:,:3] 必须等于 coord (xyz)。")
        return coord.astype(np.float32, copy=False), feat.astype(np.float32, copy=False), grid.astype(np.int64, copy=False)

    def _run_single(self, sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        tch = {k: torch.from_numpy(v).to(self._device) for k, v in sample.items()}
        # 统一 dtype
        tch["coord"] = tch["coord"].to(torch.float32)
        tch["feat"] = tch["feat"].to(torch.float32)
        tch["grid_coord"] = tch["grid_coord"].to(torch.int64)
        tch["batch"] = tch["batch"].to(torch.int64)
        tch["offset"] = tch["offset"].to(torch.int64)
        with torch.inference_mode():
            out_any = self._inner(sonata_encoder.Point(**tch))  # type: ignore
            enc = getattr(out_any, "feat", out_any)
            if not isinstance(enc, torch.Tensor):
                raise TypeError(f"Sonata.forward 必须返回 Tensor 或 Point.feat，实际 {type(out_any)}")
            real = int(enc.size(0))
            # 超出上限：与 backup 一致——直接 fail，而非静默截断
            if real > self._cap:
                raise RuntimeError(
                    f"[Sonata] token_len={real} exceeds point_token_cap={self._cap}. "
                    "Increase point_token_cap or downsample points."
                )
            if real < self._cap:
                pad = torch.zeros(self._cap - real, enc.size(1), dtype=enc.dtype, device=enc.device)
                enc = torch.cat([enc, pad], dim=0)
            elif real >= int(0.95 * self._cap):
                logger.warning(
                    "[Sonata] token_len=%d is close to cap (%d). Consider increasing it.",
                    real, self._cap
                )
        valid = np.zeros((self._cap,), dtype=bool)
        valid[:real] = True
        return enc.float().cpu().numpy(), valid

    def forward_batched(
        self,
        coord: np.ndarray,
        feat: np.ndarray,
        grid_coord: np.ndarray,
        batch: np.ndarray,
        offset: np.ndarray,
        frame_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        批处理编码一个 batch 的点云样本。
        Note: `batch` 参数在本实现未使用，保留仅为与早期接口/调用点兼容。
        """
        B = int(offset.shape[0])
        out_feat = np.zeros((B, self._cap, self._enc_out_dim), dtype=np.float32)
        out_mask = np.zeros((B, self._cap), dtype=bool)
        start = 0
        for b in range(B):
            end = int(offset[b])
            present = bool(frame_mask[b])
            if (not present) or end <= start:
                start = end
                continue
            c = coord[start:end, :]
            f = feat[start:end, :]
            g = grid_coord[start:end, :]
            c, f, g = self._sanitize_sample_arrays(c, f, g)
            n = int(c.shape[0])
            sample = {
                "coord": c,
                "feat":  f,
                "grid_coord": g,
                "batch": np.zeros((n,), dtype=np.int64),
                "offset": np.array([n], dtype=np.int64),
            }
            enc, m = self._run_single(sample)
            out_feat[b] = enc
            out_mask[b] = m
            start = end
        return out_feat, out_mask

# === Host-side helper: 将点云 token 原位插入到 <point_start>/<point_end> 窗口 ===
def _host_insert_points(
    text_emb: np.ndarray,          # (B, S, E)
    text_mask: np.ndarray,         # (B, S) bool
    token_ids: np.ndarray,         # (B, S) int
    pt_emb: np.ndarray,            # (B, P, E)
    pt_mask: np.ndarray,           # (B, P) bool
    start_id: int,
    end_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    B, S, E = text_emb.shape
    P = pt_emb.shape[1]
    fused_emb = np.zeros((B, S + P, E), dtype=text_emb.dtype)
    fused_mask = np.zeros((B, S + P), dtype=np.bool_)
    for b in range(B):
        ids = token_ids[b]
        starts = np.where(ids == start_id)[0]
        ends = np.where(ids == end_id)[0]
        if len(starts) == 0 or len(ends) == 0:
            raise RuntimeError(f"[Sonata insert] point_start_id/point_end_id not found in sample {b}.")
        # 窗口唯一性：与 backup 一致，要求各出现恰好一次
        if len(starts) != 1 or len(ends) != 1:
            raise RuntimeError(
                f"[Sonata insert] sample {b} expects exactly one <start> and one <end>, "
                f"got {len(starts)} and {len(ends)}."
            )
        s_idx = int(starts[0])
        e_idx = int(ends[0])
        if not (0 <= s_idx < e_idx < S):
            raise RuntimeError(f"[Sonata insert] Invalid window [{s_idx}, {e_idx}] in sample {b}.")
        left_len = s_idx + 1  # 包含 <point_start>
        # 左段
        fused_emb[b, 0:left_len, :] = text_emb[b, 0:left_len, :]
        fused_mask[b, 0:left_len] = text_mask[b, 0:left_len]
        # 点云段
        fused_emb[b, left_len:left_len + P, :] = pt_emb[b]
        fused_mask[b, left_len:left_len + P] = pt_mask[b]
        # 右段（含 <point_end> 以及其后的 token）——兼容 <start>/<end> 非相邻；保持总长 S+P
        right_len = S - e_idx
        fused_emb[b, left_len + P : left_len + P + right_len, :] = text_emb[b, e_idx:, :]
        fused_mask[b, left_len + P : left_len + P + right_len]   = text_mask[b, e_idx:]
    return fused_emb, fused_mask


def _pure_insert_points(
    text_emb: jnp.ndarray,
    text_mask: jnp.ndarray,
    token_ids: jnp.ndarray,
    pt_emb: jnp.ndarray,
    pt_mask: jnp.ndarray,
    start_id: int,
    end_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    B, S, E = text_emb.shape
    P = pt_emb.shape[1]
    out_emb = jax.ShapeDtypeStruct((B, S + P, E), text_emb.dtype)
    out_msk = jax.ShapeDtypeStruct((B, S + P), text_mask.dtype)
    return pure_callback(  # type: ignore
        lambda te, tm, ti, pe, pm: _host_insert_points(te, tm, ti, pe, pm, start_id, end_id),
        (out_emb, out_msk), text_emb, text_mask, token_ids, pt_emb, pt_mask,
    )

class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config  # <-- 必须：embed_prefix() 里读取 point_start_id/point_end_id 要用
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # ----- Sonata（严格 fail fast）-----
        enable_pc = False
        if getattr(config, "enable_sonata", None) is True:
            enable_pc = True
        if _PBT is not None and getattr(config, "point_backbone_type", None) == getattr(_PBT, "SONATA", None):
            enable_pc = True
        self._sonata_runner = None
        self._pt_projector = None
        if enable_pc:
            if (torch is None) or (sonata_encoder is None) or (pure_callback is None):
                raise RuntimeError("启用了点云分支但缺少依赖（torch/sonata_encoder/pure_callback）——fail fast。")
            self._sonata_runner = _TorchSonataRunner(
                point_feat_dim=int(getattr(config, "point_feat_dim", 6)),
                point_token_cap=int(getattr(config, "point_token_cap", 1024)),
                point_cfg=getattr(config, "point_config", None),
                use_pretrained=bool(getattr(config, "use_pretrained_point", True)),
                ckpt_path=getattr(config, "sonata_ckpt_path", None),
                require_cuda=bool(getattr(config, "require_cuda", False)),
            )
            # 契约前置：feats[:,:3] 将与 coord 对齐校验，要求 in_channels >= 3
            if self._sonata_runner.in_channels < 3:
                raise ValueError("point_feat_dim 必须 >= 3（feats[:,:3] 将与 coord 对齐校验）。")

            self._pt_projector = nnx.Linear(self._sonata_runner.enc_out_dim, paligemma_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        ref_dtype = None  # 用于对齐点云 token 的 dtype
        pt_tokens = None
        pt_mask = None
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
            if ref_dtype is None:
                ref_dtype = image_tokens.dtype

        # embed point clouds（Sonata，严格 fail fast；先得到 pt_tokens/pt_mask，稍后原位插入到文本窗口）
        if self._sonata_runner is not None:
            # 前置字段/形状检查
            if not (hasattr(obs, "point_clouds") and hasattr(obs, "point_cloud_masks") and "pointcloud" in obs.point_clouds):
                raise ValueError("Observation 缺少 pointcloud/point_cloud_masks 字段（fail fast）。")
            pc_arr  = obs.point_clouds["pointcloud"]  # [B, M, 3+C]
            pc_mask = obs.point_cloud_masks.get("pointcloud", None)
            if pc_mask is None:
                raise ValueError("缺少 point_cloud_masks['pointcloud']（fail fast）。")
            # 确保为 bool，避免上游 int/float 触发 & 类型错误
            pc_mask = (pc_mask != 0)
            B, M, Ctot = pc_arr.shape
            expected_last = 3 + int(self._sonata_runner.in_channels)
            if int(Ctot) != expected_last:
                raise ValueError(f"pointcloud 最后一维应为 3+in_channels ({expected_last})，实际 {int(Ctot)}。")
            # 拆分并展平
            # grid_coord 推荐上游直接提供 int32/int64；此处会在 JAX 侧统一 cast 到 int32，
            # Torch 侧 forward 时再提升到 int64，避免隐式 float→int 带来的歧义。
            grid_int = pc_arr[..., :3].astype(jnp.int32)
            coords   = pc_arr[..., 3:6].astype(jnp.float32)
            feats    = pc_arr[..., 3:].astype(jnp.float32)
            coord_f = coords.reshape(B * M, 3)
            feat_f  = feats.reshape(B * M, feats.shape[-1])
            grid_f  = grid_int.reshape(B * M, 3)
            batch_f = jnp.repeat(jnp.arange(B, dtype=jnp.int32), M)
            offset  = jnp.cumsum(jnp.full((B,), M, dtype=jnp.int32))
            # 回调输出规格（静态）
            out_struct = (
                jax.ShapeDtypeStruct((B, self._sonata_runner.cap, self._sonata_runner.enc_out_dim), jnp.float32),
                jax.ShapeDtypeStruct((B, self._sonata_runner.cap), jnp.bool_),
            )
            def _host_call(coord, feat, grid_coord, batch, offset, frame_mask):
                # `batch` 参数本实现未使用（保留占位以兼容旧接口）
                coord = np.asarray(coord); feat = np.asarray(feat); grid_coord = np.asarray(grid_coord)
                batch = np.asarray(batch); offset = np.asarray(offset); frame_mask = np.asarray(frame_mask)
                return self._sonata_runner.forward_batched(coord, feat, grid_coord, batch, offset, frame_mask)  # type: ignore
            pt_feat, pt_valid = pure_callback(  # type: ignore
                _host_call, out_struct, coord_f, feat_f, grid_f, batch_f, offset, pc_mask
            )
            pt_tokens = self._pt_projector(pt_feat) if (self._pt_projector is not None) else pt_feat
            # 统一 dtype，避免把整个前缀提升为 float32
            if ref_dtype is not None and pt_tokens.dtype != ref_dtype:
                pt_tokens = pt_tokens.astype(ref_dtype)
            # 失效 token 数值置零，更稳健
            pt_mask = jnp.broadcast_to(pc_mask[:, None], pt_tokens.shape[:2]) & pt_valid
            pt_tokens = pt_tokens * pt_mask[..., None]
            # 注意：此处不直接 append；稍后插入到文本窗口

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            text_emb  = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            text_mask = obs.tokenized_prompt_mask
            if self._sonata_runner is not None:
                # 原位插入到 <point_start>/<point_end> 窗口（backup 语义）
                start_id = getattr(self.config, "point_start_id", None)
                end_id   = getattr(self.config, "point_end_id", None)
                if (start_id is None) or (end_id is None):
                    raise RuntimeError("enable_sonata=True 但未设置 Pi0Config.point_start_id/point_end_id（fail fast）。")
                # （可选强约束）窗口中不得出现可见文本，保持与 backup 一致
                def _host_check_no_visible_between(prompt_np, mask_np, s_id, e_id):
                    P = np.asarray(prompt_np, dtype=np.int32)
                    M = np.asarray(mask_np, dtype=bool)
                    bad = []
                    for bb in range(P.shape[0]):
                        s = np.where(P[bb] == s_id)[0]
                        e = np.where(P[bb] == e_id)[0]
                        if len(s) == 0 or len(e) == 0:
                            continue
                        s0, e0 = int(s[0]), int(e[0])
                        if s0 < e0:
                            mid = M[bb, s0 + 1 : e0]  # True 表示可见
                            if mid.any():
                                bad.append(bb)
                    if bad:
                        raise RuntimeError(f"Visible tokens between <point_start> and <point_end> in samples: {bad[:8]}")
                    return np.int32(0)
                _ = pure_callback(  # type: ignore
                    _host_check_no_visible_between,
                    jax.ShapeDtypeStruct((), jnp.int32),
                    obs.tokenized_prompt, text_mask, int(start_id), int(end_id),
                )
                fused_text, fused_mask = _pure_insert_points(
                    text_emb, text_mask, obs.tokenized_prompt,
                    pt_tokens, pt_mask,
                    int(start_id), int(end_id),
                )
                tokens.append(fused_text)
                input_mask.append(fused_mask)
                # 前缀内部完全互看
                ar_mask += [False] * int(fused_text.shape[1])
            else:
                tokens.append(text_emb)
                input_mask.append(text_mask)
                ar_mask += [False] * int(text_emb.shape[1])
        else:
            if self._sonata_runner is not None:
                # 启用点云但无文本窗口 → 直接失败（与 backup 行为一致）
                raise RuntimeError("启用 Sonata 但缺少文本（无法进行 <point_start>/<point_end> 插入）。")

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, (batch_size,))
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

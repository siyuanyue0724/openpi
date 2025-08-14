"""
Pi0FAST-Sonata
--------------
将 SpatialLM 的 Sonata 点云编码器 **原样** 融合到 Pi0‑FAST 中，并收敛到与 SpatialLM 预训练**一致的接口与写法**。

唯一支持的输入接口（严格对齐 SpatialLM 数据形态）：
  • Observation.point_clouds["pointcloud"] : [B, M, 3 + C] (float32)
      - [:, :, 0:3] = grid_coord 体素网格（必须非负；模型内强制 cast→int32，强烈建议上游直接 int32）
      - [:, :, 3:6] = coord 连续 xyz (float32)
      - [:, :, 6:]  = 其它特征 (float32)
      - 严格契约：feats = [xyz, extras...]，且 feats[:,:3] 与 coord 完全一致（模型内**严格断言**）。
  • Observation.point_cloud_masks["pointcloud"] : [B] bool
      - 指示该样本是否提供点云；若配置 require_pointcloud=True，则本 batch **至少一个样本**必须为 True；
        为空的样本按 SpatialLM 逻辑走全零特征 + 全 False 掩码（不报错）。

与 SpatialLM 的一致性：
  - Sonata 超参、Fourier 编码、input_proj 完全一致；
  - **严格 6 通道**（xyz+rgb），禁止动态改 in_channels；
  - 必须提供 <|point_start|>/<|point_end|> 的 id，并**原位插入**点 token（删除降级前缀拼接路径）。
  - 若 grid/xyz/feat 不满足契约：直接报错，不做任何“自动修复/体素化”。
  - 若可获取 reduced_grid_size，若 grid_coord 超界则 **warning**（与 SpatialLM 行为一致），建议上游修正。
"""

# ========================= 训练模式（默认全量微调；无任何环境变量兜底） =========================
#   "all"       : 方案A，全量微调（Sonata + projector + 其他 JAX 参数）。训练循环需传入
#                 pt_feat_override / pt_mask_override（推荐 DLPack 零拷贝）。
#   "projector" : 冻结 Sonata，仅训练 projector（可走 pure_callback，无需 override）。
#   "frozen"    : Sonata + projector 均冻结（只训练 LLM/SigLIP 等）。
# ============================================================================================

# 该版本已知问题（这些问题目前暂时不用立刻解决）：
# 训练时梯度：pure_callback 分支不可微；override（方案A）分支可端到端训练 Sonata。
# 端到端训练无需把点云分支迁到 JAX：按本实现的 override+DLPack 梯度交接即可；若要 XLA 单体编译/进一步融合，
# 再考虑 JAX 化（可选优化，而非必要条件）。
# 性能注意：
#   - 纯 callback 路径：存在 host↔device 往返与 per-sample 调用开销，batch>1/梯度积累时更明显；
#   - override 路径：Torch→JAX 采用 DLPack 同卡零拷贝，无 CPU copy 瓶颈（需保证 JAX 与 Torch 在同一 GPU）。

# ------------------------------------------------------------------------------------------------------

# 仅提醒：
# 1024 的“点 token 容量”先与现有预算保持；显存允许时再调大 point_token_cap（与 enc_patch_size[-1] 无关）。
# grid 必须非负（已在 pure_callback/override 侧硬断言）。
# grid → coord 偏移：当前不会在模型内改动 grid_coord（只做“非负 + 形状 + dtype”校验）；是否归零或对齐，请在数据侧统一处理。
# 为避免 X64 警告：JAX 端 batch/offset 用 int32，host(PyTorch) 端统一 .long()。
# 关于“首个生成 token +1”：本实现已采用正确的 **+0**（首个新 token 的 position = prefill_len），与 one‑shot 对齐。

import dataclasses
import inspect
import logging
import typing
import jax
import jax.numpy as jnp
import numpy as np
import torch
import os
import warnings
from pathlib import Path
from functools import partial
from typing import Any, Dict, Tuple, Optional

# ---- JAX <-> openpi 兼容：KeyArray 在 JAX 0.4.14+ 被移除 ----
import jax.random as _jr
if not hasattr(_jr, "KeyArray"): _jr.KeyArray = jax.Array  # type: ignore[attr-defined]

from flax import nnx
# nnx_bridge allows bridging PyTorch modules to NNX (Flax) modules
import flax.nnx.bridge as nnx_bridge

from jax import ShapeDtypeStruct
from jax import pure_callback

from openpi.models import sonata_encoder  # This module should define the Sonata point cloud encoder
import openpi.models.gemma_fast as _gemma 
from openpi.models import pi0_fast as _pi0_fast
from openpi.models import siglip as _siglip
from openpi.models import PointBackboneType, ProjectorType
import openpi.models.model as _model  # for BaseModel and Observation
from openpi.shared import array_typing as at  # for inputs_spec override
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

# ───────────────────────────────────────────────────────────────────────────────
# 与 SpatialLM 严格一致的 Sonata 超参（单一真源，避免循环导入）
# 如果需要切换权重族，只需在实例化 config 时覆盖 point_config 即可。
# enc_mode 保持 True（你们实现中 True==voxel），如需字符串枚举可在外部覆盖为 "voxel"。
POINT_CONFIG_SPATIALLM: dict[str, typing.Any] = {
    "in_channels": 6,
    "order": ("z", "z-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (3, 3, 3, 12, 3),
    "enc_channels": (48, 96, 192, 384, 512),
    "enc_num_head": (3, 6, 12, 24, 32),
    "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    "mlp_ratio": 4.0,
    "mask_token": True,
    "enc_mode": True,               # 若你们 Sonata 用字符串枚举，可改为 "voxel"
    "enable_fourier_encode": True,
    "num_bins": 1280,
    # "enable_flash" 由运行期覆盖（跟随环境与驱动）
}
# ───────────────────────────────────────────────────────────────────────────────

# 用于生成测试用数据的工具函数
def _canonicalize_point_dict(pd):
    # 用 jnp；不做任何“数据修复”，只做形状规范化与 dtype 约束
    pd = {k: jnp.asarray(v) for k, v in pd.items()}

    if pd["coord"].ndim == 3:      # (B,N,3) → (B*N,3)；其它字段同步展平，保证 per‑sample 选择后维度一致
        B, N, _ = pd["coord"].shape
        pd["coord"] = jnp.reshape(pd["coord"], (B * N, 3))
        pd["feat"]  = jnp.reshape(pd["feat"],  (B * N, -1))
        # ★ 关键：grid_coord 也要一起展平，否则 selected_batch 后会与 coord 不对齐
        if "grid_coord" in pd and pd["grid_coord"].ndim == 3:
            pd["grid_coord"] = jnp.reshape(pd["grid_coord"], (B * N, 3))
        # JAX 端统一 int32；host(PyTorch) 端再升为 int64 以支持 (batch << 48)
        pd["batch"]  = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
        pd["offset"] = jnp.cumsum(jnp.full((B,), N, dtype=jnp.int32))

    if "grid_size" in pd and pd["grid_size"].ndim == 3:
        pd["grid_size"] = jnp.reshape(pd["grid_size"], (-1, 3))

    # JAX 端保持 int32（避免 x64 警告）；host 端再升为 int64
    for key in ("batch", "offset"):
         if key in pd:
            pd[key] = pd[key].astype(jnp.int32, copy=False)

    # SpatialLM 约定：显式提供 int32 体素坐标（不做零点平移/重建）
    if "grid_coord" in pd:
        # 按 SpatialLM 约定仅做 dtype/形状校验，不做数值修复
        if pd["grid_coord"].ndim != 2 or pd["grid_coord"].shape[-1] != 3:
            raise ValueError("pointcloud.grid_coord 的最后一维必须为 3（xyz）。")
        pd["grid_coord"] = pd["grid_coord"].astype(jnp.int32, copy=False)

    return pd

# ---------- host-side helper: find <point_start>/<point_end> (no adjacency check) ----------
def _host_find_window_only(
    prompt_np: np.ndarray,
    start_id: int,
    end_id: int,
) -> np.ndarray:
    """
    返回 np.int32[2] = [s_idx, e_idx]，只定位起止位置，
    不对二者间是否有文本做任何约束（对齐 SpatialLM 的宽松假设）。
    """
    arr = np.asarray(prompt_np).tolist()
    L = len(arr)
    s_pos = [i for i, t in enumerate(arr) if t == start_id]
    e_pos = [i for i, t in enumerate(arr) if t == end_id]
    if len(s_pos) != 1 or len(e_pos) != 1:
        raise ValueError(f"[point-window] expect exactly one <start>/<end>, got start={s_pos}, end={e_pos}")
    s, e = s_pos[0], e_pos[0]
    if not (0 <= s < e < L): raise ValueError(f"[point-window] invalid order: start={s}, end={e}, L={L}")
    # D) 软提示：start 与 end 相邻（或几乎无夹层文本）
    if e - s <= 1:
        warnings.warn(
            f"[Pi0FAST-Sonata] <point_start> and <point_end> are adjacent at positions ({s},{e}). "
            "No textual context between them.",
            RuntimeWarning
        )
    return np.array([s, e], dtype=np.int32)

# -------- 仅支持“新接口”抽取点云 ----------
def _extract_point_batch(obs, *, expected_feat_dim: int) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """
    返回 (pc_dict, frame_mask)：
      • obs.point_clouds["pointcloud"] : [B, M, 3 + C]，其中 C == expected_feat_dim
          - [:,:,0:3]  = grid_coord (int32, 非负；此处会 cast)
          - [:,:,3:6]  = coord xyz (float32)
          - [:,:,6:6+C-3] = 其它特征 (float32)
      • obs.point_cloud_masks["pointcloud"] : [B] bool
    """
    if not (hasattr(obs, "point_clouds") and "pointcloud" in getattr(obs, "point_clouds")):
        raise ValueError("未提供 Observation.point_clouds['pointcloud']（新接口）。")
    if not (hasattr(obs, "point_cloud_masks") and "pointcloud" in getattr(obs, "point_cloud_masks")):
        raise ValueError("未提供 Observation.point_cloud_masks['pointcloud']（新接口）。")
    pc_arr  = obs.point_clouds["pointcloud"]
    pc_mask = obs.point_cloud_masks["pointcloud"]
    if pc_arr.ndim != 3:
        raise ValueError(f"pointcloud 数组应为 [B, M, 3+C]，实际 shape={pc_arr.shape}.")
    B, M, Ctot = pc_arr.shape
    if pc_mask.shape != (B,):
        raise ValueError(f"point_cloud_masks['pointcloud'] 应为 [B]，实际 shape={pc_mask.shape}.")
    # 布尔 dtype：允许 jnp.bool_ / np.bool_ / 内建 bool，统一为 jnp.bool_，避免不同管线名字差异导致的假阳性
    if not (jnp.issubdtype(pc_mask.dtype, jnp.bool_) or np.issubdtype(pc_mask.dtype, np.bool_)):
        raise ValueError(
            f"point_cloud_masks['pointcloud'] 必须是布尔类型，实际为 {pc_mask.dtype}."
        )
    # 归一到 JAX 布尔，后续广播/逻辑运算更稳
    pc_mask = pc_mask.astype(jnp.bool_, copy=False)
    if Ctot != 3 + expected_feat_dim:
        raise ValueError(f"最后一维应为 3 + C（C={expected_feat_dim}），实际 {Ctot}。请裁剪/重排到 feats=[xyz, rgb] 共 6 维。")
    grid_int = pc_arr[..., :3].astype(jnp.int32)      # (B,M,3)
    coords   = pc_arr[..., 3:6].astype(jnp.float32)   # 连续 xyz
    feats    = pc_arr[..., 3:].astype(jnp.float32)    # xyz + 语义
    pc_dict = _canonicalize_point_dict(dict(coord=coords, grid_coord=grid_int, feat=feats))
    return pc_dict, pc_mask

# Alias the Sonata class from the sonata_encoder module for convenience
Sonata = sonata_encoder.Sonata

@dataclasses.dataclass(frozen=True)
class Pi0FASTSonataConfig(_pi0_fast.Pi0FASTConfig):
    """Configuration for the Pi0FASTSonata model (Pi0FAST with Sonata point cloud encoder)."""

    # === SpatialLM 契约 ===
    # point_feat_dim 定义为传入 Sonata 的 feats 维度（不含 grid）：
    # feats = [xyz(3), extra...]，例如 xyzrgb ⇒ point_feat_dim = 6
    # Observation.pointcloud 的最后一维 = 3(grid) + point_feat_dim
    point_feat_dim: int = 6  # 强制 6 通道（xyz+rgb）
    # 每帧点数的静态上界（仅用于 inputs_spec 的形状声明；不会截断实际输入）
    max_points: int = 32768
    # 让父类 Pi0FASTConfig.inputs_spec() 按原逻辑自动注入点云字段
    # （Pi0FASTSonata 本身不会用这个枚举去构建模块，只用于 spec）
    point_backbone_type: PointBackboneType = PointBackboneType.SONATA
    # projector_type 对 spec 无影响，这里保持 None 或 LINEAR 均可
    projector_type: ProjectorType | None = None
    # 与 SpatialLM 严格一致的点云编码器超参；默认使用本文件中的“单一真源”
    point_config: dict = dataclasses.field(default_factory=lambda: dict(POINT_CONFIG_SPATIALLM))

    # —— 解耦 “点 token 总容量” 与 Sonata 的注意力 patch 大小 ——
    # 仅用于 JAX 侧的静态形状（pure_callback 返回定长），与 enc_patch_size[-1] 无直接关系。
    # 默认仍为 1024（与常见 ckpt/显存预算匹配）；显存允许可在外部配置为 4096/8192。
    point_token_cap: int = 1024

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Inherits default action_dim, action_horizon, max_token_len from Pi0FASTConfig (e.g., 32, 32, 250)
    use_pretrained_point: bool = True      # 调试阶段可设 False 跳过下
    require_pointcloud: bool = True
    # 若无图像则直接不训练（训练时报错终止；推理不受影响，这里主要用于进行调试确保数据集正确）
    require_image: bool = True
    point_start_id: Optional[int] = None
    point_end_id:   Optional[int] = None
    # 可选：允许 <start>/<end> 之间出现的“可见”占位 token（例如 <|point_pad|>）的 id；
    # 若为 None，则该区间内不允许任何可见文本（mask=True 的 token）。
    point_pad_id: Optional[int] = None
    # 可选：强制要求 CUDA；默认 False 以兼容旧流程/CI
    require_cuda: bool = False
    # 训练模式：默认 "all"（全量微调）
    sonata_train_mode: str = "all"  # "all" | "projector" | "frozen"

    @property
    def model_type(self) -> _model.ModelType:
        # Reuse the PI0_FAST model type (since this is an extension of Pi0-FAST architecture)
        return _model.ModelType.PI0_FAST

    def create(self, rng: jax.Array) -> "Pi0FASTSonata":
        """Instantiate a Pi0FASTSonata model with random initialization."""
        return Pi0FASTSonata(self, rngs=nnx.Rngs(rng))

    # ---- 覆写 inputs_spec：复用父类规范，仅补充点云字段，避免 state 维度不一致 ----
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        parent_obs, act_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=parent_obs.images,
                image_masks=parent_obs.image_masks,
                state=parent_obs.state,
                tokenized_prompt=parent_obs.tokenized_prompt,
                tokenized_prompt_mask=parent_obs.tokenized_prompt_mask,
                token_ar_mask=parent_obs.token_ar_mask,
                token_loss_mask=parent_obs.token_loss_mask,
                # 点云（新接口）：与 SpatialLM 的 Sonata 调用保持等价语义
                point_clouds={"pointcloud": jax.ShapeDtypeStruct(
                    [batch_size, self.max_points, 3 + self.point_feat_dim], jnp.float32)},
                point_cloud_masks={"pointcloud": jax.ShapeDtypeStruct([batch_size], jnp.bool_)},
            )
        return observation_spec, act_spec

    # ---- LoRA 兼容：确保 Sonata.projector 在默认冻结策略下可训练 ----
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """
        使用 PathRegex 以兼容不同 flax.nnx 版本（避免依赖 nnx.Name/nnx.None_）。
        语义：
          - all       : 从 base 冻结集中排除 Sonata.encoder / Sonata.projector（两者可训练）
          - projector : 冻结 Sonata.encoder，放开 Sonata.projector（其余遵循 base）
          - frozen    : Sonata.encoder 与 Sonata.projector 都冻结（并集到 base）
        """
        base = super().get_freeze_filter()
        mode = (getattr(self, "sonata_train_mode", None) or "").strip().lower()
        if mode not in ("all", "projector", "frozen"):
            raise ValueError("config.sonata_train_mode 必须显式设为 'all' | 'projector' | 'frozen'")

        # 以路径正则匹配模块；state 路径一般形如 "Sonata/encoder/..." 或 "Sonata/projector/..."
        enc  = nnx_utils.PathRegex(r"^Sonata/encoder($|/)")
        proj = nnx_utils.PathRegex(r"^Sonata/projector($|/)")

        if mode == "all":
            # base ∩ ¬enc ∩ ¬proj
            return nnx.All(base, nnx.Not(enc), nnx.Not(proj))
        elif mode == "projector":
            # (base ∩ ¬proj) ∪ enc
            return nnx.Any(nnx.All(base, nnx.Not(proj)), enc)
        else:  # "frozen"
            # base ∪ enc ∪ proj
            return nnx.Any(base, enc, proj)

# 批量化 <start>/<end> 查找（单次回调，减少 B 次 pure_callback）的工具函数
def _host_find_windows_batched(prompts_np: np.ndarray, start_id: int, end_id: int) -> np.ndarray:
    # prompts_np: [B, L]
    B = prompts_np.shape[0]
    out = np.zeros((B, 2), dtype=np.int32)
    for b in range(B):
        out[b] = _host_find_window_only(prompts_np[b], start_id, end_id)
    return out
class Pi0FASTSonata(_model.BaseModel):
    """
    Pi0FASTSonata model: Extends Pi0-FAST to incorporate a Sonata point cloud encoder.
    Combines vision (SigLIP image encoder), language (PaLI-Gemma LLM), and point cloud (Sonata) encoders.
    """
    def __init__(self, config: Pi0FASTSonataConfig, rngs: nnx.Rngs):
        # Initialize base model (setup action_dim, action_horizon, etc.)
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # --------------------------------------------------------------
        # 记录是否强制要求点云
        self._require_pointcloud = bool(getattr(config, "require_pointcloud", True))
        self._require_image = bool(getattr(config, "require_image", True))
        # 设定统一 device（默认允许 CPU；如需强制请在 config.require_cuda=True）
        # --------------------------------------------------------------
        if getattr(config, "require_cuda", False) and not torch.cuda.is_available():
            raise RuntimeError(
                "Pi0FAST‑Sonata requires CUDA for spconv/flash‑attn when require_cuda=True."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ------------------------------------------------------------------
        # 1) 语言模型  Gemma  ------------------------------------------------
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Gemma‑fast LLM ----------------------------------------------------
        # ------------------------------------------------------------------
        # 与 pi0_fast.py 完全一致的用法：get_config 返回 dict
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )

        # 初始化
        llm.lazy_init(rngs=rngs, method="init")

        # ------------------------------------------------------------------
        # 2) 图像编码器  SigLIP  --------------------------------------------
        # ------------------------------------------------------------------
        # 与 pi0_fast.py 同源：统一使用 LLM 的 width
        model_width = paligemma_config.get("width", paligemma_config.get("hidden_size", 1024))

        # 兼容不同版本的 siglip.Module：仅传入其签名里存在的参数
        _sig_params = set(inspect.signature(_siglip.Module).parameters.keys())
        raw_img_kwargs = dict(
            variant="So400m/14",
            pool_type="none",
            scan=True,
        )
        if "num_classes" in _sig_params:
            raw_img_kwargs["num_classes"] = model_width
        # dtype 参数名在不同版本里可能叫 dtype 或 dtype_mm
        if "dtype" in _sig_params:
            raw_img_kwargs["dtype"] = config.dtype
        elif "dtype_mm" in _sig_params:
            raw_img_kwargs["dtype_mm"] = config.dtype

        # ---------------------------------------------------------------------------------

        img = nnx_bridge.ToNNX(_siglip.Module(**raw_img_kwargs))

        # Initialize image encoder with a dummy image to set dimensions
        dummy_image = next(iter(config.fake_obs(batch_size=1).images.values()))
        img.lazy_init(dummy_image, train=False, rngs=rngs)

        # ------------------------------------------------------------------
        # 3) 创建并加载点云编码器 (Sonata) — 参数对齐 SpatialLM‑1.1, 后面可以改成config传输
        # ------------------------------------------------------------------
        
        # ---------- Sonata hyper‑params：单一真源，严格对齐 ckpt ----------
        if int(config.point_feat_dim) != 6:
            raise ValueError("本实现严格复刻 SpatialLM：point_feat_dim 必须为 6（xyz+rgb）。")
        _in_channels = 6
        # 若未安装 flash_attn，自动禁用以避免断言失败
        _enable_flash = (self.device.type == "cuda" and getattr(sonata_encoder, "flash_attn", None) is not None)
        if not _enable_flash and not getattr(Pi0FASTSonata, "_warned_no_flash", False):
            logger.warning("[Sonata] flash-attn not found; falling back to non-flash path (slower, it is highly recommended to use flash-attn).")
            Pi0FASTSonata._warned_no_flash = True
        # 以 config.point_config 为单一真源；仅覆盖运行期相关开关
        sp_cfg = dict(config.point_config)
        # 必要字段齐备性（fail‑fast）
        for k in ("in_channels","order","stride","enc_depths","enc_channels","enc_num_head",
                  "enc_patch_size","mlp_ratio","mask_token","enc_mode","enable_fourier_encode","num_bins"):
            if k not in sp_cfg:
                raise KeyError(f"[Pi0FASTSonata] point_config 缺少字段：{k}")
        # in_channels 必须与实现保持 6（xyz+rgb）
        if int(sp_cfg["in_channels"]) != _in_channels:
            raise ValueError(f"[Pi0FASTSonata] point_config.in_channels={sp_cfg['in_channels']} != 6")
        # —— 构造参数鲁棒化：按 Sonata 签名过滤未知参数；必要时转换 enc_mode；仅在支持时注入 enable_flash —— #
        try:
            _sig = inspect.signature(Sonata)
        except Exception:
            _sig = None
        # enc_mode: 若实现以字符串枚举表示，且目前值是布尔，则做布尔→字符串映射
        if isinstance(sp_cfg.get("enc_mode"), bool) and _sig is not None:
            _p = _sig.parameters.get("enc_mode")
            if _p is not None and (isinstance(_p.default, str) or _p.annotation is str):
                sp_cfg["enc_mode"] = "voxel" if sp_cfg["enc_mode"] else "point"
        # enable_flash: 仅在 Sonata.__init__ 接受该形参时注入
        if _sig is not None and "enable_flash" in _sig.parameters:
            sp_cfg["enable_flash"] = _enable_flash
        else:
            sp_cfg.pop("enable_flash", None)
        # 过滤未知形参，避免不同分支/版本直接 TypeError
        if _sig is not None:
            sp_cfg = {k: v for k, v in sp_cfg.items() if k in _sig.parameters}
        point_model = Sonata(**sp_cfg)
        # 记录供后续断言／Projector 使用
        self._point_in_channels = _in_channels
        # 与 SpatialLM 一致：proj 输入维 = 编码器末层通道
        self._enc_out_dim = sp_cfg["enc_channels"][-1]
        # 暴露原生 torch Sonata 句柄，供方案A训练时使用
        self._torch_sonata = point_model

        if config.use_pretrained_point:
            # ------------------------------------------------------------------
            # 仅使用本地精简后的 SpatialLM1.1 Sonata 权重
            # 文件放置: <repo_root>/openpi/pretrain/SpatialLM_Sonata_encoder.pth
            # 如果没有模型，请使用uv run scripts/sonata_weight_gen.py来获取sonata的checkpoint
            # ------------------------------------------------------------------
            # 兼容旧路径：ENV → <repo_root>/pretrain → <repo_root>/src/pretrain
            parents = list(Path(__file__).resolve().parents)
            candidates = [os.getenv("OPENPI_SONATA_CKPT", "").strip()]
            for idx in (1, 2, 3, 4):
                if len(parents) > idx:
                    candidates.append(parents[idx] / "pretrain" / "SpatialLM_Sonata_encoder.pth")
            ckpt_path = None
            for p in candidates:
                p = Path(p)
                if str(p) and p.is_file():
                    ckpt_path = p
                    break
            if ckpt_path is None:
                raise FileNotFoundError(
                    "Sonata 预训练权重未找到。请设置 OPENPI_SONATA_CKPT，"
                    "或将权重放到 <repo_root>/pretrain/ 或 <repo_root>/src/pretrain/。"
                )
            logger.info("Using Sonata weights: %s", ckpt_path)


            # 加载权重 —— PyTorch 2.6+ 默认 weights_only=True，会拦住包含
            # numpy 标量的老式 state_dict，这里显式关掉即可。
            load_kwargs = dict(map_location="cpu")
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False

            raw_obj = torch.load(ckpt_path, **load_kwargs)

            # ①若是训练 checkpoint，先取出真正权重
            if isinstance(raw_obj, dict) and "state_dict" in raw_obj:
                state_dict = raw_obj["state_dict"]
            else:
                state_dict = raw_obj

            # ②去掉多余前缀（如 "module." 或 "model.")
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith(("module.", "model.", "student.backbone.", "student.", "point_backbone.")):
                    cleaned[k.split(".", 1)[1]] = v
                else:
                    cleaned[k] = v

            # in_channels 严格一致性检查（禁止“动态重建”）
            stem_w = cleaned.get("embedding.stem.linear.weight", None)
            if stem_w is None:
                raise KeyError("Sonata 权重缺少 'embedding.stem.linear.weight'。")
            in_from_ckpt = int(stem_w.shape[1])
            if in_from_ckpt != self._point_in_channels:
                raise ValueError(f"Sonata ckpt in_channels={in_from_ckpt} 与期望 {self._point_in_channels} 不一致；"
                                 "请使用与 SpatialLM 完全一致的 6 通道权重。")

            # ③严格一致性加载：shape 不匹配会直接抛错；并且 missing/unexpected 立刻 fail‑fast
            # 过滤历史分支中可能存在的 embedding.mask_token（仅当当前模型确实不存在该键时）
            _model_keys = set(point_model.state_dict().keys())
            if "embedding.mask_token" in cleaned and "embedding.mask_token" not in _model_keys:
                cleaned.pop("embedding.mask_token")

            # F) 严格一致加载（更直观的严格语义）；若不一致，直接报错
            try:
                ik = point_model.load_state_dict(cleaned, strict=True)
            except RuntimeError as e:
                raise RuntimeError(f"[Sonata] strict load_state_dict failed: {e}") from e
            missing = list(getattr(ik, "missing_keys", []))
            unexpected = list(getattr(ik, "unexpected_keys", []))
            if missing or unexpected:
                raise RuntimeError(
                    "[Sonata] Weight/state mismatch.\n"
                    f"  missing={missing[:10]}\n"
                    f"  unexpected={unexpected[:10]}\n"
                    "请确认 point_config 与权重版本严格一致。"
                )

            logger.info("Loaded pretrained Sonata weights from %s", ckpt_path)

        else:
            logger.warning(
                "Pi0FAST‑Sonata: use_pretrained_point=False -> "
                "Sonata encoder starts with random weights."
            )

        # -------- 保证 Sonata 整个网络在 GPU 且 dtype=fp32（对齐 SpatialLM 前向） --------
        point_model.to(dtype=torch.float32, device=self.device)
        point_model.eval()  # inference mode

        # -------------------- TorchSonataWrapper（host 侧运行，JAX 侧 pure_callback） --------------------
        class _TorchSonataWrapper(torch.nn.Module):
            """
            - 追踪期（JIT / lazy_init）只返回 ShapeDtypeStruct，不跑真模型。
            - 运行期通过 pure_callback 把 numpy → torch.cuda → numpy。
            """
            def __init__(
                self,
                pt_model: torch.nn.Module,
                device: torch.device,
                max_tokens_cap: int,
                enc_out_dim: int,
            ):
                super().__init__()
                # 确保 pt_model 已在目标 device
                self.inner = pt_model.to(device).eval()
                self.device = device
                # 仅作“静态输出容量”（JAX 侧形状），与 enc_patch_size[-1] 无关
                self.max_tokens_cap = max_tokens_cap
                self.enc_out_dim = enc_out_dim

            # ---------- 内部 util ----------
            @staticmethod
            @torch.no_grad()
            def _torch_forward(
                inner: torch.nn.Module,
                host_dict: Dict[str, np.ndarray],
                device: torch.device,
                max_tokens_cap: int,
                enc_out_dim: int,
            ) -> Tuple[np.ndarray, np.ndarray]:  # 第 2 个 ndarray 是 bool 掩码
                """
                接收 host 上的 numpy 输入 → torch.Tensor.cuda → 运行 → numpy 输出
                统一返回 float32 numpy 数组
                """

                # ---------- fast‑path：该帧不存在，直接返回全零 ----------
                present = int(host_dict.pop("present", 1))
                if present == 0:
                    pad_feat   = np.zeros((max_tokens_cap, enc_out_dim), dtype=np.float32)
                    valid_mask = np.zeros((max_tokens_cap,),       dtype=bool)
                    return pad_feat, valid_mask
                
                # =========================================================
                # 若上游传入 selected_batch，只保留这一帧（按 batch 等于 sb 过滤）
                # 简化为唯一稳定路径：要求 host_dict 含 batch；去掉收益极小的兜底分支。
                # ---------------------------------------------------------
                if "selected_batch" in host_dict:  # 按样本切片
                    sb = int(host_dict.pop("selected_batch"))
                    if "batch" not in host_dict:
                        raise ValueError("[SpatialLM‑Sonata] selected_batch 需要提供 'batch' 字段。")
                    b_arr = np.asarray(host_dict["batch"]).reshape(-1)
                    sel = (b_arr == sb)                     # sel.shape == (B*M,)
                    for k, v in list(host_dict.items()):
                        if hasattr(v, "shape") and v.ndim and v.shape[0] == sel.shape[0]:
                            host_dict[k] = v[sel]
                    # 单样本语义：batch 全 0，offset = [点数]
                    n_pts = int(host_dict["coord"].shape[0])
                    host_dict["batch"]  = np.zeros(n_pts, dtype=np.int64)
                    host_dict["offset"] = np.array([n_pts], dtype=np.int64)
                # =========================================================

                # ---------- 先安全扁平化 ----------
                if host_dict["coord"].ndim != 2:         # (B,N,3) → (B*N,3)
                    B, N, _ = host_dict["coord"].shape
                    host_dict["coord"] = host_dict["coord"].reshape(B * N, 3)
                    host_dict["feat"]  = host_dict["feat"].reshape(B * N, -1)
                    host_dict["batch"] = np.repeat(np.arange(B, dtype=np.int64), N)
                    host_dict["offset"] = np.cumsum(np.full((B,), N, dtype=np.int64))

                # grid_coord 可能来自体素网格 → 保证是 (P,3)
                for _key in ("grid_coord",):
                    if _key in host_dict and host_dict[_key].ndim == 3:
                        host_dict[_key] = host_dict[_key].reshape(-1, 3)
                
                # ---------- grid 非负性检查（host 侧，JIT 安全） ----------
                if "grid_coord" not in host_dict:
                    raise ValueError(
                        "[SpatialLM‑Sonata] 缺少 grid_coord；"
                        "严格模式要求上游显式提供非负 int32 体素坐标 (N,3)。"
                        "请将 Observation.point_clouds['pointcloud'] 的前 3 列作为 grid 传入，或在 legacy dict 中提供 'grid_coord'。"
                    )
                gc = host_dict["grid_coord"]
                if gc.ndim == 3:
                    gc = gc.reshape(-1, 3)
                    host_dict["grid_coord"] = gc
                if gc.shape[1] != 3:
                    raise ValueError(f"[SpatialLM‑Sonata] grid_coord 维度错误：期待 (N,3)，实际 {gc.shape}。")
                if not np.issubdtype(gc.dtype, np.integer):
                    raise ValueError(f"[SpatialLM‑Sonata] grid_coord dtype 必须为整数（建议 int32），实际 {gc.dtype}。")
                if np.any(gc < 0):
                    raise ValueError("[SpatialLM‑Sonata] grid_coord 含负值；请在上游体素化时保证每维索引 ≥ 0。")
                # ---- coord / feat 形状&dtype 严格校验（纯报错，不做转换，这是为了确保能找到数据集的错误而不会错误地把错误地数据集给弄得可用造成silent error） ----
                c = host_dict["coord"]
                f = host_dict["feat"]
                if c.ndim != 2 or c.shape[1] != 3:
                    raise ValueError(f"[SpatialLM‑Sonata] coord 形状必须为 (N,3)，实际 {c.shape}。")
                if f.ndim != 2:
                    raise ValueError(f"[SpatialLM‑Sonata] feat 形状必须为 (N,C)，实际 {f.shape}。")
                if c.dtype != np.float32:
                    raise ValueError(f"[SpatialLM‑Sonata] coord dtype 必须为 float32，实际 {c.dtype}。")
                if f.dtype != np.float32:
                    raise ValueError(f"[SpatialLM‑Sonata] feat dtype 必须为 float32，实际 {f.dtype}。")
                # ---- batch/offset（如提供）一致性校验 ----
                if "batch" in host_dict:
                    b = host_dict["batch"]
                    if b.ndim != 1:
                        raise ValueError(f"[SpatialLM‑Sonata] batch 必须是一维向量 (N,)，实际 {b.shape}。")
                    if b.shape[0] != c.shape[0]:
                        raise ValueError(f"[SpatialLM‑Sonata] batch 长度 {b.shape[0]} 与点数 {c.shape[0]} 不一致。")
                    if not np.issubdtype(b.dtype, np.integer):
                        raise ValueError(f"[SpatialLM‑Sonata] batch dtype 必须为整数，实际 {b.dtype}。")
                    if np.any(b < 0):
                        raise ValueError("[SpatialLM‑Sonata] batch 含负值。")
                if "offset" in host_dict:
                    off = host_dict["offset"]
                    if off.ndim != 1:
                        raise ValueError(f"[SpatialLM‑Sonata] offset 必须是一维向量 (B,)，实际 {off.shape}。")
                    if not np.issubdtype(off.dtype, np.integer):
                        raise ValueError(f"[SpatialLM‑Sonata] offset dtype 必须为整数，实际 {off.dtype}。")
                    if np.any(off <= 0) or np.any(off[1:] < off[:-1]):
                        raise ValueError("[SpatialLM‑Sonata] offset 必须为严格递增的前缀和。")
                    if off[-1] != c.shape[0]:
                        raise ValueError(f"[SpatialLM‑Sonata] offset[-1]={off[-1]} 不等于点数 N={c.shape[0]}。")
                # SpatialLM 未对上界做硬断言；若实现暴露了 reduced_grid_size，仅警告一次
                if getattr(inner, "enable_fourier_encode", False) and hasattr(inner, "reduced_grid_size"):
                    reduced_gs = int(getattr(inner, "reduced_grid_size"))
                    if np.any(gc >= reduced_gs):
                        warnings.warn(
                            f"[Sonata] grid_coord 存在越界（≥ reduced_grid_size={reduced_gs}）；"
                            "这可能影响四ier归一化效果，请检查体素化配置（num_bins/stride 或坐标归一）。",
                            RuntimeWarning
                        )
                
                # grid_coord 与 coord 点数必须一致
                if host_dict["grid_coord"].shape[0] != host_dict["coord"].shape[0]:
                    raise ValueError(
                        f"[SpatialLM‑Sonata] 点数不一致：grid_coord N={host_dict['grid_coord'].shape[0]} "
                        f"≠ coord N={host_dict['coord'].shape[0]}。"
                    )
                
                # ---------- 若缺 offset，则根据 batch 生成 ----------
                if "offset" not in host_dict:
                    if "batch" in host_dict:
                        # --------------------------------------------------
                        # 1. 保证 batch 是 1‑D int64
                        # --------------------------------------------------
                        b_arr = np.asarray(host_dict["batch"]).astype(np.int64)
                        if b_arr.ndim > 1:
                            b_arr = b_arr.reshape(-1)
                        host_dict["batch"] = b_arr

                        # 2. 统计每个 batch 的点数 → 前缀和
                        counts = np.bincount(b_arr)
                        host_dict["offset"] = np.cumsum(counts, dtype=np.int64)
                    else:
                        # 单 batch 情形
                        host_dict["offset"] = np.array(
                            [host_dict["coord"].shape[0]], dtype=np.int64
                        )

                assert host_dict["coord"].shape[0] == host_dict["offset"][-1], (
                    f"coord N={host_dict['coord'].shape[0]}, "
                    f"but offset[-1]={host_dict['offset'][-1]}"
                )

                # ---------- NaN 过滤：与 SpatialLM‑Qwen 完全一致 ----------
                nan_mask = (
                    np.isnan(host_dict["coord"]).any(axis=1)
                    | np.isnan(host_dict["feat"]).any(axis=1)
                )
                if nan_mask.any():
                    keep = ~nan_mask
                    for k in ("coord", "feat", "grid_coord"):
                        if k in host_dict:
                            host_dict[k] = host_dict[k][keep]
                    host_dict["batch"] = host_dict["batch"][keep]
                    # 重新计算 offset  (prefix‐sum of per‑batch counts)
                    counts = np.bincount(host_dict["batch"])
                    host_dict["offset"] = np.cumsum(counts, dtype=np.int64)

                # —— 可选：过滤“全零 padding 行”（默认关闭；仅当数据侧确有零填充时建议开启） —— #
                if os.environ.get("OPENPI_SONATA_FILTER_ZERO_PADDING", "0") == "1":
                    zero_coord = (np.abs(host_dict["coord"]).max(axis=1) == 0)
                    zero_feat  = (np.abs(host_dict["feat"]).max(axis=1)  == 0)
                    zero_grid  = (np.abs(host_dict["grid_coord"]).max(axis=1) == 0)
                    is_zero_row = zero_coord & zero_feat & zero_grid
                    if np.any(is_zero_row):
                        keep = ~is_zero_row
                        if not np.any(keep):
                            pad_feat   = np.zeros((max_tokens_cap, enc_out_dim), dtype=np.float32)
                            valid_mask = np.zeros((max_tokens_cap,),       dtype=bool)
                            return pad_feat, valid_mask
                        for k in ("coord", "feat", "grid_coord"):
                            if k in host_dict:
                                host_dict[k] = host_dict[k][keep]
                        if "batch" in host_dict:
                            host_dict["batch"] = host_dict["batch"][keep]
                        # 重新计算 offset
                        if "batch" in host_dict:
                            counts = np.bincount(host_dict["batch"])
                            host_dict["offset"] = np.cumsum(counts, dtype=np.int64)
                        else:
                            host_dict["offset"] = np.array([host_dict["coord"].shape[0]], dtype=np.int64)
                
                # ===== SpatialLM‑style 空点云保护 =====
                if host_dict["coord"].shape[0] == 0:
                    # 返回全零特征 + 全 False 有效标记（维度与正常路径一致）
                    pad_feat   = np.zeros((max_tokens_cap, enc_out_dim), dtype=np.float32)
                    valid_mask = np.zeros((max_tokens_cap,),       dtype=bool)
                    return pad_feat, valid_mask

                # 严格模式：不再在模型内“重建/偏移” grid。缺失已在前面报错。
                
                # ---------- 契约校验：feat 前 3 维应与 coord（连续 xyz）一致 ----------
                if host_dict["feat"].shape[1] >= 3:
                    max_abs = float(np.max(np.abs(host_dict["feat"][:, :3] - host_dict["coord"])))
                    if not np.isfinite(max_abs) or max_abs > 1e-4:
                        raise ValueError(
                            f"[SpatialLM‑Sonata] 契约不满足：feat[:,:3]（应为 xyz）与 coord 不一致；"
                            f"max|diff|={max_abs:.3e}。请保证 feats = [xyz, extras] 且 coord=xyz。"
                        )

                # pure_callback 把 jax.Array 直接送过来；必须先转成真正的 numpy
                # 显式 copy：避免 "The given NumPy array is not writable" 警告
                tch_in = {
                    k: torch.from_numpy(np.array(v, copy=True)).to(device)
                    for k, v in host_dict.items()
                }
                # Sonata 里要做 (batch << 48)，必须是 int64
                for key in ("batch", "offset"):
                    if key in tch_in:
                        tch_in[key] = tch_in[key].long()
                out = inner(tch_in)
                # SpatialLM 约定：Sonata.forward() 返回 torch.Tensor（最终特征）
                if not isinstance(out, torch.Tensor):
                    raise TypeError(
                        f"Sonata.forward is expected to return a torch.Tensor "
                        f"(as in SpatialLM), but got: {type(out)}. "
                        f"Please ensure Sonata.forward returns the feature tensor."
                    )

                real_len = out.size(0)
                if real_len > max_tokens_cap:
                    raise RuntimeError(
                        f"[SpatialLM‑Sonata] token_len={real_len} 超过 cap={max_tokens_cap}。"
                        "请增大 point_token_cap 或在上游减少每样本点数（cap 与 enc_patch_size 无关）。"
                    )
                elif real_len >= int(0.95 * max_tokens_cap):
                    warnings.warn(
                        f"[Sonata] token_len={real_len} 接近 cap ({max_tokens_cap})；"
                        "建议增大 point_token_cap 或降低点密度，避免隐性截断风险。",
                        RuntimeWarning,
                    )
                # SpatialLM: 不截断；右 pad 到 max_tokens_cap 的定长（JAX 需要静态形状）
                MAX_TOKEN = max_tokens_cap
                pad_len = MAX_TOKEN - real_len
                if pad_len:
                    pad = out.new_zeros(pad_len, out.size(1))
                    out = torch.cat([out, pad], 0)
                valid_mask = np.arange(MAX_TOKEN) < real_len
                return out.float().cpu().numpy(), valid_mask

            # ---------- forward ----------
            def forward(self, pc_dict, *, train: bool = False):
                host_inputs = {k: jnp.asarray(v) for k, v in pc_dict.items()}
                # tree_flatten 返回 (flat_list, treedef)；后者负责反向展开
                flat, treedef = jax.tree_util.tree_flatten(host_inputs)

                # -----------------------------------------------------------
                # 静态输出上限（仅为 JAX 形状所需；与 enc_patch_size 无关）
                MAX_TOKEN = self.max_tokens_cap
                C = self.enc_out_dim
                out_struct = (ShapeDtypeStruct((MAX_TOKEN, C), jnp.float32),
                              ShapeDtypeStruct((MAX_TOKEN,),  jnp.bool_))

                def _host_call(*flat_np):
                    # flat_np 是回传的扁平列表/元组，需用 treedef.unflatten 还原
                    np_dict = treedef.unflatten(list(flat_np))
                    # 训练/评估切换（便于将来需要）
                    self.inner.train(bool(train))
                    with torch.inference_mode(not bool(train)):
                        return self._torch_forward(
                            self.inner, np_dict, self.device, self.max_tokens_cap, self.enc_out_dim
                        )
                feat, valid_mask = pure_callback(
                    _host_call, out_struct, *flat, vectorized=False
                )

                # ---- Runtime contract check: channel dim must match projector ----
                assert feat.shape[-1] == self.enc_out_dim, (
                    f"[SonataWrapper] Output dim {feat.shape[-1]} "
                    f"!= expected {self.enc_out_dim}. "
                    "Check Sonata ckpt / config."
                )
                # 直接返回 (feat, valid_mask)；由调用方自行构造 mask
                return feat, valid_mask

            # ---------- NNX 初始化 ----------
            def init_with_output(
                self,
                rngs,
                pc_dict,
                *,
                train: bool = False,
                method: typing.Any = None,
                **_,
            ):
                dummy_out = self.forward(pc_dict, train=train)
                return dummy_out, {}  # 本 wrapper 不含可训练参数

            # --------------------------------------------------------------
            # ★ 兼容 nnx‑bridge 调用：重载 apply，忽略 variables / rngs. 以后需要rng需要将其传入但目前传入会造成兼容性问题
            # --------------------------------------------------------------
            def apply(                       # type: ignore[override]
                self,
                _variables,                  # nnx‑bridge 传入的占位变量包（unused）
                *args,
                rngs=None,
                method: str | None = "forward",
                **kwargs,
            ):
                """
                • method 为 nnx‑bridge 指定的函数名，例如 "forward"、
                  "init_with_output"；默认 = "forward"  
                • rngs 仅在 lazy_init 时会传入，Sonata 不使用，直接丢弃
                """
                if method is None:
                    method = "forward"
    
                # 选定被调函数
                target_fn = getattr(self, method)
    
                # init_with_output 的签名为 (rngs, pc_dict, …)
                if method == "init_with_output":
                    return target_fn(rngs, *args, **kwargs)
    
                # 其余方法（forward 等）
                return target_fn(*args, **kwargs)

        # ---------- 4) 原生 NNX 线性投影：enc_out_dim → PaLI‑Gemma hidden_size ----------
        # C) 投影层 dtype 与 LLM 对齐，减少不必要的 f32<->bf16 转换
        _dtype_map = {
            "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16,
            "float16": jnp.float16,   "fp16": jnp.float16,
            "float32": jnp.float32,   "fp32": jnp.float32,
        }
        proj_dtype = _dtype_map.get(str(getattr(config, "dtype", "bfloat16")).lower(), jnp.bfloat16)
        _linear_params = set(inspect.signature(nnx.Linear).parameters.keys())
        if "dtype" in _linear_params:
            point_proj = nnx.Linear(self._enc_out_dim, model_width, dtype=proj_dtype, rngs=rngs)
            self._proj_dtype = proj_dtype
        else:
            # 回退：nnx.Linear 没有 dtype 参数时，使用默认 dtype（通常是 float32）
            point_proj = nnx.Linear(self._enc_out_dim, model_width, rngs=rngs)
            self._proj_dtype = jnp.float32
            logger.warning("nnx.Linear has no 'dtype' parameter; projector uses default dtype (likely float32).")

        # 让投影层跟随 PyTorch 的 device，不必手动迁移除非后续继续重写到jax

        N = 64               # 64 points (dummy)
        # ────────────────────────────────────────────────────────────────
        # Sonata 约定的 Point 结构（务必扁平化）：
        #   coord : (Total_N, 3)   float32 / int32
        #   feat  : (Total_N, C)
        #   batch : (Total_N,)     **int64**  ← 要能做 «batch << 48»
        #   offset: (B,)           累计点数前缀和  **int64**
        #   grid_size : (B, 3)
        # ────────────────────────────────────────────────────────────────
        # 与 SpatialLM 契约一致：feat[:,:3] 必须等于 coord（连续 xyz）
        coord_dummy = jnp.arange(N * 3, dtype=jnp.float32).reshape(N, 3)
        extra_dims = int(self._point_in_channels) - 3
        feat_dummy = (
            jnp.concatenate(
                [coord_dummy, jnp.zeros((N, extra_dims), jnp.float32)], axis=-1
            )
            if extra_dims > 0 else coord_dummy
        )
        grid_dummy = jnp.zeros((N, 3), dtype=jnp.int32)   # 严格模式：dummy 也提供非负 int32 grid
        raw_dummy_pc = {
            "coord":  coord_dummy,
            "feat":   feat_dummy,                         # ← 前 3 列 = coord
            # JAX 端 int32；host 端再升到 int64
            "batch":  jnp.zeros((N,),  dtype=jnp.int32),
            "offset": jnp.array([N],   dtype=jnp.int32),
            "grid_coord": grid_dummy,
        }
        dummy_pc = _canonicalize_point_dict(raw_dummy_pc)

        # 仅将 point_token_cap 用作纯回调的静态输出容量；与 enc_patch_size 无关
        _cap = int(getattr(config, "point_token_cap", sp_cfg["enc_patch_size"][-1]))
        point = nnx_bridge.ToNNX(
            _TorchSonataWrapper(
                point_model,
                self.device,
                _cap,
                self._enc_out_dim,   # ← 与 __init__ 对齐
            )
        )
        # 记录“点 token 容量”（仅用于 label 对齐等静态逻辑）
        self._pt_block_len = int(_cap)
        self._pt_token_cap = int(_cap)  # alias，供 torch 侧 batch 编码便捷使用
        
        # 明确使用 wrapper 的 init_with_output 以避免在不同 nnx-bridge 版本下的歧义
        point.lazy_init(dummy_pc, train=False, rngs=rngs, method="init_with_output")
        # ↑ 若你的 nnx 版本会自动识别，则这行与上一行等价；显式化更稳

        # ------------------------------------------------------------------
        # 6) 打包所有子模块（边界归位）
        #    - PaliGemma 只保留 llm/img
        #    - Sonata 子树负责 encoder/projector
        # ------------------------------------------------------------------
        self.PaliGemma = nnx.Dict(
            llm = llm,
            img = img,
        )
        self.Sonata = nnx.Dict(
            encoder   = point,
            projector = point_proj,
        )
        # 需要在 forward 中用到，保存成静态字段
        self._point_pad_id = getattr(config, "point_pad_id", None)
        # special ids（原位插入必需）
        self._point_start_id = getattr(config, "point_start_id", None)
        self._point_end_id   = getattr(config, "point_end_id", None)
        if (self._point_start_id is None) or (self._point_end_id is None):
            raise ValueError("必须提供 point_start_id / point_end_id 才能进行 SpatialLM‑exact 原位插入。")

        # 训练模式（必须显式配置；无环境变量兜底）
        cfg_mode = (getattr(config, "sonata_train_mode", None) or "").strip().lower()
        if cfg_mode not in ("all", "projector", "frozen"):
            raise ValueError(
                "config.sonata_train_mode 必须显式设为 'all' | 'projector' | 'frozen'；"
                "已禁用环境变量兜底（OPENPI_SONATA_TRAIN_MODE）。未显式配置将直接报错。"
            )
        self._sonata_train_mode = cfg_mode

    # ----------------------- Debug: 30s 自检 projector 是否在学 -----------------------
    def _proj_param_ref(self):
        """返回 projector 主权重 Param（兼容 kernel/weight 命名差异）。"""
        p = getattr(self.Sonata.projector, "kernel", None)
        if p is None: p = getattr(self.Sonata.projector, "weight", None)
        if p is None:
            raise AttributeError("未找到 Sonata.projector 的权重参数（kernel/weight）。")
        return p

    def debug_snapshot_projector(self) -> float:
        """记录 projector 权重的 L2 范数快照，并返回当前 L2 值（float）。"""
        w = self._proj_param_ref()
        cur = float(jnp.linalg.norm(w.astype(jnp.float32)))
        self._dbg_proj_l2_prev = cur
        logger.info("[SONATA/DEBUG] projector ||W||_2 snapshot = %.6e", cur)
        return cur

    def debug_log_projector_delta(self, tag: str = "") -> float:
        """打印自上次 snapshot 起的 L2 变化；返回 delta（float）。"""
        w = self._proj_param_ref()
        cur = float(jnp.linalg.norm(w.astype(jnp.float32)))
        prev = getattr(self, "_dbg_proj_l2_prev", None)
        delta = float("nan") if prev is None else cur - prev
        logger.info("[SONATA/DEBUG] projector Δ||W||_2 = %.6e %s", delta, tag)
        self._dbg_proj_l2_prev = cur
        return delta

    # ----------------------- 便捷：PyTorch 侧 Sonata 编码（方案A） -----------------------
    def torch_sonata_encode_batch(self, obs: _model.Observation, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        在 PyTorch 上逐样本跑 Sonata，pad 到 point_token_cap，返回：
          pt_feat_t: [B, P_cap, C_enc] torch.float32（是否 requires_grad 由调用侧设置）
          pt_mask_t: [B, P_cap] torch.bool
        说明：与 pure_callback 分支严格等价（含 NaN 过滤、cap 检查、右侧 pad）。
        """
        pc = obs.point_clouds["pointcloud"]                       # jax.Array [B, M, 3+C]
        mask_frame = obs.point_cloud_masks["pointcloud"]          # [B]
        pc_np = np.asarray(pc)
        mask_np = np.asarray(mask_frame)
        B, M, Ctot = pc_np.shape
        cap = int(self._pt_token_cap)
        Cenc = int(self._enc_out_dim)
        device = self.device
        self._torch_sonata.train(bool(train))
        feat_batch, mask_batch = [], []
        for b in range(B):
            if not bool(mask_np[b]):
                feat_batch.append(torch.zeros((cap, Cenc), dtype=torch.float32, device=device))
                mask_batch.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                continue
            arr = pc_np[b]  # [M, 3+C]
            # 显式 copy 以避免 "The given NumPy array is not writable" 警告与潜在 UB
            grid = torch.from_numpy(arr[:, :3].copy()).to(device=device, dtype=torch.int32)
            coord = torch.from_numpy(arr[:, 3:6].copy()).to(device=device, dtype=torch.float32)
            feat  = torch.from_numpy(arr[:, 3:].copy()).to(device=device, dtype=torch.float32)
            # 契约 1：grid 非负
            if (grid < 0).any().item():
                raise ValueError("[Sonata/override] grid_coord 含负值；请在上游体素化时保证每维索引 ≥ 0。")
            good = ~(torch.isnan(coord).any(-1) | torch.isnan(feat).any(-1))
            grid, coord, feat = grid[good], coord[good], feat[good]
            # 契约 2：feat[:,:3] 与 coord 一致（允许微小数值误差）
            if feat.shape[1] >= 3:
                max_abs = torch.max(torch.abs(feat[:, :3] - coord)).item()
                if not np.isfinite(max_abs) or max_abs > 1e-4:
                    raise ValueError(
                        f"[Sonata/override] 契约不满足：feat[:,:3]（应为 xyz）与 coord 不一致；"
                        f"max|diff|={max_abs:.3e}。请保证 feats = [xyz, extras] 且 coord=xyz。"
                    )
            if feat.numel() == 0:
                feat_batch.append(torch.zeros((cap, Cenc), dtype=torch.float32, device=device))
                mask_batch.append(torch.zeros((cap,), dtype=torch.bool, device=device))
                continue
            inp = {
                "coord": coord,
                "feat":  feat,
                "grid_coord": grid,
                "batch": torch.zeros(coord.shape[0], dtype=torch.long, device=device),
                "offset": torch.as_tensor([coord.shape[0]], dtype=torch.long, device=device),
            }
            with torch.set_grad_enabled(train):
                out = self._torch_sonata(inp)  # [K_b, C_enc]
            K = int(out.shape[0])
            if K > cap:
                raise RuntimeError(f"[Sonata] token_len={K} > cap={cap}; 请增大 point_token_cap")
            pad = torch.zeros((cap - K, Cenc), dtype=out.dtype, device=device)
            feat_pad = torch.cat([out, pad], dim=0)
            mask_pad = torch.zeros((cap,), dtype=torch.bool, device=device)
            mask_pad[:K] = True
            feat_batch.append(feat_pad)
            mask_batch.append(mask_pad)
        pt_feat_t = torch.stack(feat_batch, dim=0)
        pt_mask_t = torch.stack(mask_batch, dim=0)
        return pt_feat_t, pt_mask_t

    def embed_inputs(
        self,
        obs: _model.Observation,
        *,
        pt_feat_override: jax.Array | None = None,   # [B, P_cap, C_enc]（方案A）
        pt_mask_override: jax.Array | None = None,   # [B, P_cap] bool
        train: bool | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Safety：在非 all 模式禁止覆盖输入（避免误把 Sonata 训练起来）
        if self._sonata_train_mode != "all" and (
            (pt_feat_override is not None) or (pt_mask_override is not None)
        ):
            raise RuntimeError(
                "sonata_train_mode != 'all' 时不应传入 pt_feat_override/pt_mask_override；"
                "该模式下应走纯 callback（Sonata 冻结）。"
            )
        """
        Embed all input modalities (images, point cloud, text tokens) into sequences of token embeddings.
        Returns:
            token_embeddings (jax.Array): concatenated token features of shape [B, N, emb_dim]
            input_mask (jax.Array[bool]): mask indicating valid tokens [B, N]
            ar_mask (jax.Array[int]): autoregressive mask for tokens [B, N] (0 where tokens attend freely, 1 where causal dependence begins).
        """
        token_embeddings: list[jax.Array] = []
        input_mask: list[jax.Array] = []
        ar_mask: list[jax.Array] = []

        # ---------- 1) 聚合图像 tokens （先合并所有相机，便于后续逐样本插入逻辑） ----------
        img_tok_list = []
        img_msk_list = []
        for cam_name, image in obs.images.items():
            # 兼容返回 (tokens, pooled) 或仅 tokens 的实现
            _img_out = self.PaliGemma.img(image, train=bool(train))
            if isinstance(_img_out, tuple):
                img_t = _img_out[0]
            else:
                img_t = _img_out
            img_tok_list.append(img_t)
            m = jnp.broadcast_to(
                obs.image_masks[cam_name][:, None],
                (obs.image_masks[cam_name].shape[0], img_t.shape[1]),
            )
            img_msk_list.append(m)
        if len(img_tok_list):
            img_tokens = jnp.concatenate(img_tok_list, axis=1)      # [B, Nimg, D]
            img_mask   = jnp.concatenate(img_msk_list, axis=1)      # [B, Nimg]
        else:
            # 理论上不会发生（你的 config.require_image=True），兜底
            B = obs.tokenized_prompt.shape[0]
            D = int(getattr(self.PaliGemma.llm.module, "hidden_size", 1024))
            img_tokens = jnp.zeros((B, 0, D), dtype=jnp.float32)
            img_mask   = jnp.zeros((B, 0),   dtype=bool)
        img_ar = jnp.zeros_like(img_mask, dtype=jnp.int32)

        # ---------- 2) Point cloud tokens ----------
        pc_dict_all, pc_frame_mask = _extract_point_batch(obs, expected_feat_dim=self._point_in_channels)
        # require_pointcloud：本 batch 至少有一个样本存在点云(避免数据集错误)；空帧允许（走零特征）
        if self._require_pointcloud:
            present_any = jnp.any(pc_frame_mask.astype(bool))
            def _host_assert_any(x):
                if not bool(np.asarray(x)):
                    raise RuntimeError("require_pointcloud=True 但该 batch 所有样本均无点云。")
                return np.int32(0)
            _ = pure_callback(
                _host_assert_any,
                ShapeDtypeStruct((), jnp.int32),
                present_any,
                vectorized=False,
            )

        # —— 文本与点云 batch 维度一致性（防止 silent misalignment）
        B_txt = int(obs.tokenized_prompt.shape[0])
        B_pc  = int(pc_frame_mask.shape[0])
        if B_txt != B_pc:
            raise ValueError(f"B 维度不一致：text batch={B_txt} 与 pointcloud batch={B_pc}。请在数据侧保证一致。")
        B = B_txt
        # 运行时维度检查：feats（xyz+extras）列数应等于 in_channels（=6）
        feat_dim = int(pc_dict_all["feat"].shape[-1])
        if feat_dim != self._point_in_channels:
            raise ValueError(f"点云特征维不匹配：期望 feats= {self._point_in_channels}，实际 {feat_dim}。"
                             "请将 feats 严格设为 [xyz, rgb] 共 6 维。")

        per_sample_tokens  = []
        per_sample_masks   = []
        max_len            = 0
        use_override = (pt_feat_override is not None)
        if use_override:
            # ---------- 严格对齐 override 的形状 ----------
            if int(pt_feat_override.shape[0]) != B:
                raise ValueError(f"pt_feat_override.shape[0]={int(pt_feat_override.shape[0])} ≠ batch={B}。")
            if pt_mask_override is None:
                raise ValueError("pt_mask_override is required when pt_feat_override is provided.")
            if int(pt_mask_override.shape[0]) != B:
                raise ValueError(f"pt_mask_override.shape[0]={int(pt_mask_override.shape[0])} ≠ batch={B}。")
            # 1) 长度必须与 point_token_cap 一致（否则 compute_loss 内的 LM/Nimg 会错位）
            if int(pt_feat_override.shape[1]) != int(self._pt_block_len):
                raise ValueError(
                    f"pt_feat_override.shape[1]={int(pt_feat_override.shape[1])} "
                    f"≠ point_token_cap(self._pt_block_len)={int(self._pt_block_len)}；"
                    "embed_inputs/compute_loss 的文本/点/图像拼接假设将出现错位。"
                )
            if int(pt_mask_override.shape[1]) != int(self._pt_block_len):
                raise ValueError(
                    f"pt_mask_override.shape[1]={int(pt_mask_override.shape[1])} "
                    f"≠ point_token_cap(self._pt_block_len)={int(self._pt_block_len)}。"
                )
            # 2) 通道维应等于编码器末层维度（防止错误地把“原始点特征”传进 projector）
            if int(pt_feat_override.shape[-1]) != int(self._enc_out_dim):
                raise ValueError(
                    f"pt_feat_override.shape[-1]={int(pt_feat_override.shape[-1])} "
                    f"≠ enc_out_dim={int(self._enc_out_dim)}；请传入 Sonata 编码后的特征而非原始点特征。"
                )
            # 明确两者的前两维应一致（防止静默广播）
            if (int(pt_feat_override.shape[0]) != int(pt_mask_override.shape[0])) or \
               (int(pt_feat_override.shape[1]) != int(pt_mask_override.shape[1])):
                raise ValueError(
                    "pt_feat_override 与 pt_mask_override 的前两维（B 与 P_cap）必须一致。"
                )
            # 方案A：外部已在 PyTorch 侧计算好 Sonata 特征（enc_out_dim）
            pt_tokens = self.Sonata.projector(pt_feat_override.astype(self._proj_dtype))   # [B, P_cap, D]
            valid_m   = pt_mask_override.astype(bool)                                      # [B, P_cap]
            pt_tokens = pt_tokens * valid_m[:, :, None]                                    # projector 后再次遮蔽
            max_len   = pt_tokens.shape[1]
        else:
            # pure_callback（不可微）。若处于训练阶段且声明为 "all"，给出明确报错提示。
            if (train is True) and (self._sonata_train_mode == "all"):
                raise RuntimeError(
                    "sonata_train_mode='all' 但未提供 pt_feat_override/pt_mask_override；"
                    "请在训练循环中先用 PyTorch 跑 Sonata，并通过 DLPack 将特征/掩码传入 embed_inputs。"
                )
            for b in range(B):  # 逐样本调用 Sonata
                present_b = pc_frame_mask[b]
                single_dict = {
                    **pc_dict_all,
                    "selected_batch": jnp.array(b, jnp.int32),
                    "present": jnp.asarray(present_b, jnp.int32),
                }
                if "grid_coord" in single_dict and single_dict["grid_coord"].shape[-1] != 3:
                    raise ValueError(f"pointcloud grid_coord 的最后一维必须为 3，实际 shape={single_dict['grid_coord'].shape}")
                # projector/frozen 模式下：纯回调一律 eval，避免 Dropout/BN 训练态
                enc_train = False
                tok, vmask = self.Sonata.encoder(single_dict, train=enc_train)
                # -------- 长度一致性断言 --------
                assert tok.shape[0] == vmask.shape[0], (
                    f"Sonata 返回 token 长度 {tok.shape[0]} "
                    f"≠ valid_mask 长度 {vmask.shape[0]}"
                )
                # projector 前后各遮蔽一次，数值更稳
                proj_in = (tok * vmask[:, None]).astype(self._proj_dtype)
                tok     = self.Sonata.projector(proj_in)
                tok     = tok * vmask[:, None]
                per_sample_tokens.append(tok)
                per_sample_masks.append(vmask)
                max_len = max(max_len, tok.shape[0])

        # ----- pad 到 batch 内最大长度（一次性做）-----------
        def _pad_to(x, tgt):
            pad = [(0, tgt - x.shape[0])] + [(0, 0)]*(x.ndim-1)
            return jnp.pad(x, pad)
        if not use_override:
            # wrapper 恒定返回 cap 长度，这里防御性断言一致
            assert int(max_len) == int(self._pt_block_len), (
                f"Sonata wrapper token_len={int(max_len)} "
                f"!= point_token_cap={int(self._pt_block_len)}"
            )
            pt_tokens = jnp.stack([_pad_to(t, max_len) for t in per_sample_tokens], axis=0)
            valid_m   = jnp.stack([_pad_to(m, max_len) for m in per_sample_masks], axis=0)
        assert (pt_tokens.shape[1] == valid_m.shape[1] == max_len), "pad 后 token / mask 长度不一致"

        # --- 帧级掩码广播到 token 维 ---
        pc_frame_mask_b = jnp.broadcast_to(pc_frame_mask[:, None], (B, max_len))
        pt_final_mask   = pc_frame_mask_b & valid_m
        pt_tokens       = pt_tokens * pt_final_mask[:, :, None]

        # 3. Text tokens (from language model embedding)
        # Ensure textual inputs are present
        assert obs.tokenized_prompt is not None and obs.tokenized_prompt_mask is not None and obs.token_ar_mask is not None, \
            "Tokenized prompt and corresponding masks must be provided for text inputs."
        if not (obs.tokenized_prompt_mask.shape == obs.tokenized_prompt.shape[:2]):
            raise ValueError("tokenized_prompt_mask.shape 必须与 tokenized_prompt 对齐。")
        if not (obs.token_ar_mask.shape == obs.tokenized_prompt.shape[:2]):
            raise ValueError("token_ar_mask.shape 必须与 tokenized_prompt 对齐。")
        txt_tokens = self.PaliGemma.llm(tokens=obs.tokenized_prompt, embed_only=True)  # [B, L, emb_dim]
        # 避免如果错误地给了空prompt或只有1个token，在后面逻辑里产生“0长度但不报错”
        if obs.tokenized_prompt.shape[1] < 2:
           raise ValueError("tokenized_prompt 长度必须 ≥ 2，用于构造 next-token 监督。")
        # —— 统一三模态 embedding 的 dtype（尤其是原位插入时 dynamic_update_slice 必须一致）—— #
        target_dtype = txt_tokens.dtype
        pt_tokens = pt_tokens.astype(target_dtype)

        # ===== 4) SpatialLM‑exact 原位插入（无降级路径） =====
        B, L, D = txt_tokens.shape
        P = pt_tokens.shape[1]               # = point_token_cap（由 wrapper 定长）
        LM = L + P                           # “文本+点”段固定 buffer 长度
        img_tokens = img_tokens.astype(target_dtype)
        pt_tokens  = pt_tokens.astype(target_dtype)

        # --- 定位每个样本的 <start>/<end>（不限制二者之间是否有文本；与 SpatialLM 一致） ---
        out_struct = (ShapeDtypeStruct((2,), jnp.int32),)
        def _host_call_batch(arr):
            return _host_find_windows_batched(np.asarray(arr), int(self._point_start_id), int(self._point_end_id))
        (win_all,) = pure_callback(
            _host_call_batch,
            (ShapeDtypeStruct((B, 2), jnp.int32),),
            obs.tokenized_prompt,
            vectorized=False
        )
        # 可选防呆：<start>/<end> 不应被 mask 掉（若被屏蔽，直接 fail-fast）
        pm = obs.tokenized_prompt_mask.astype(jnp.bool_)
        s_ok = jnp.take_along_axis(pm, win_all[:, 0:1], axis=1).squeeze(1)
        e_ok = jnp.take_along_axis(pm, win_all[:, 1:1+1], axis=1).squeeze(1)
        def _host_check(vs, ve):
            vs, ve = bool(np.asarray(vs)), bool(np.asarray(ve))
            if not (vs and ve):
                raise RuntimeError("[Pi0FAST-Sonata] <point_start>/<point_end> 必须是有效 token（mask=True）。")
            return np.int32(0)
        _ = pure_callback(_host_check, ShapeDtypeStruct((), jnp.int32), s_ok, e_ok, vectorized=False)

        # ======== B1：中间文本硬断言（最小化修复）========
        # 仅允许 mask=False 的不可见文本，或（若提供 point_pad_id）允许该 id 的可见占位 token。
        pad_id = -1 if (self._point_pad_id is None) else int(self._point_pad_id)
        def _host_assert_mid(prompts_np, masks_np, win_np, pad):
            P = np.asarray(prompts_np)
            M = np.asarray(masks_np).astype(bool)
            W = np.asarray(win_np)
            bad = []
            for i in range(P.shape[0]):
                s, e = int(W[i,0]), int(W[i,1])
                if e - s <= 1:
                    continue
                vis = M[i, s+1:e]
                if vis.any():
                    seg = P[i, s+1:e]
                    if pad >= 0:
                        ok = (~vis) | (seg == pad)
                    else:
                        ok = ~vis
                    if not bool(ok.all()):
                        bad.append(i)
            if bad:
                raise RuntimeError(
                    "[Pi0FAST-Sonata] Visible text between <point_start> and <point_end> is not allowed "
                    f"(except optional point_pad_id={pad}). Offending samples: {bad[:10]}"
                )
            return np.int32(0)
        _ = pure_callback(
            _host_assert_mid,
            ShapeDtypeStruct((), jnp.int32),
            obs.tokenized_prompt, obs.tokenized_prompt_mask, win_all, jnp.int32(pad_id),
            vectorized=False
        )

        # --- 逐样本装配：通过“条件索引 + take”避免动态形状更新 ---
        seq_list, msk_list, ar_list = [], [], []
        for b in range(B):
            s_idx = win_all[b, 0]                     # <start> 位置
            e_idx = win_all[b, 1]                     # <end>   位置
            K_b = jnp.sum(pt_final_mask[b].astype(jnp.int32))     # 有效点 token 数
            L_right = L - e_idx

            # 目标序列（仅文本+点段）的全长索引：0..LM-1
            t = jnp.arange(LM, dtype=jnp.int32)
            pt_start = s_idx + 1
            pt_end   = pt_start + K_b

            left_cond   = (t <  pt_start)
            points_cond = (t >= pt_start) & (t < pt_end)
            right_cond  = (t >= pt_end)   & (t < pt_end + L_right)

            txt_idx = jnp.where(left_cond, t, e_idx + (t - pt_end))
            txt_idx = jnp.clip(txt_idx, 0, L - 1)
            pt_idx  = jnp.clip(t - pt_start, 0, P - 1)

            txt_part = jnp.take(txt_tokens[b], txt_idx, axis=0)
            txt_part = txt_part * (left_cond | right_cond)[:, None].astype(txt_part.dtype)
            pt_part  = jnp.take(pt_tokens[b],  pt_idx,  axis=0)
            pt_part  = pt_part * points_cond[:, None].astype(pt_part.dtype)
            txtpts_scatter = txt_part + pt_part                      # [LM, D]

            # mask / ar：文本沿用源；点 token 严格因果（ar=1，label=-100）；padding 位置全 False
            txt_mask_src = jnp.take(obs.tokenized_prompt_mask[b].astype(bool), txt_idx, axis=0)
            m_txt = txt_mask_src & (left_cond | right_cond)
            m_pt  = points_cond
            m_txtpt = m_txt | m_pt                                    # [LM]

            txt_ar_src = jnp.take(obs.token_ar_mask[b].astype(jnp.int32), txt_idx, axis=0)
            ar_txt = txt_ar_src * (left_cond | right_cond).astype(jnp.int32)
            # B2：点 token 作为“前缀条件”（ar=0），但其 labels 仍为 -100（不预测）
            ar_pt  = jnp.zeros_like(t, dtype=jnp.int32)
            ar_txtpt = jnp.where(points_cond, ar_pt, ar_txt)

            # 与图像拼接： [img | (text+points)]
            seq_b = jnp.concatenate([img_tokens[b], txtpts_scatter], axis=0)
            m_b   = jnp.concatenate([img_mask[b],   m_txtpt], axis=0)
            ar_b  = jnp.concatenate([img_ar[b],     ar_txtpt], axis=0).astype(jnp.int32)
            seq_list.append(seq_b); msk_list.append(m_b); ar_list.append(ar_b)

        tokens = jnp.stack(seq_list, axis=0)   # [B, Nimg+LM, D]  (LM = L + P)
        mask   = jnp.stack(msk_list,  axis=0)  # [B, Nimg+LM]
        ar     = jnp.stack(ar_list,   axis=0)  # [B, Nimg+LM]
        return tokens, mask, ar

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        # 允许全量微调模式把 Torch 侧算好的点云特征/掩码透传进来
        pt_feat_override: jax.Array | None = None,
        pt_mask_override: jax.Array | None = None,
    ) -> jax.Array:
        """Compute sequence loss for the model (predict next token loss for language prompt)."""
        # Preprocess observation (normalize/augment images, ensure masks)
        # 向图像预处理函数传递 RNG（仅在训练阶段需要）
        prep_rng = rng if train else None
        observation = _model.preprocess_observation(prep_rng, observation, train=train, image_keys=list(observation.images.keys()))
        # 若要求样本必须包含图像，而当前 observation 无图像，则直接中止训练
        if self._require_image and len(observation.images) == 0:
            raise RuntimeError(
                "Pi0FAST‑Sonata: require_image=True but observation contains no images; "
                "abort this training step or filter such samples upstream."
            )
        # Embed all inputs to tokens and get masks
        tokens, mask, ar = self.embed_inputs(
            observation, train=train,
            pt_feat_override=pt_feat_override, pt_mask_override=pt_mask_override)
        # Compute attention mask for the sequence (prefix + causal masking as needed)
        attn_mask = _pi0_fast.make_attn_mask(mask, ar)
        # === SpatialLM‑exact：原位插入后文本位置改变，按新位置对齐标签 ===
        llm_mod = getattr(self.PaliGemma.llm, "module", self.PaliGemma.llm)
        vocab_size = getattr(llm_mod, "vocab_size")
        B, L = observation.tokenized_prompt.shape
        # 1) 先得到整段 pre_logits → logits_all
        # 显式给出 positions，确保训练前向与解码路径使用完全一致的绝对位置编号
        seq_positions = (jnp.cumsum(mask.astype(jnp.int32), axis=-1) - 1).astype(jnp.int32)
        # 防御性夹紧：极端情况下若序列左端存在整段无效位，避免出现 -1 位置
        tf_positions  = jnp.maximum(seq_positions[:, :-1], 0)
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=tokens[:, :-1],
            positions=tf_positions,
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True
        )
        logits_all, *_ = self.PaliGemma.llm(pre_logits=pre_logits)   # [B, T_total-1, V]

        # 2) 直接重建“插入后”的 labels（包含 <start>/<end>），点位置 IGNORE
        T_total = tokens.shape[1]
        P = int(getattr(self, "_pt_block_len", 1024))
        LM = observation.tokenized_prompt.shape[1] + P           # = L + P
        Nimg = T_total - LM
        IGNORE = jnp.int32(-100)

        # --- (a) <start>/<end> 位置（批量化，减少 host 往返） ---
        def _host_call_batch2(arr):
            return _host_find_windows_batched(np.asarray(arr), int(self._point_start_id), int(self._point_end_id))
        (win_all,) = pure_callback(
            _host_call_batch2,
            (ShapeDtypeStruct((B, 2), jnp.int32),),
            observation.tokenized_prompt,
            vectorized=False
        )
        s_all = win_all[:, 0]
        e_all = win_all[:, 1]

        # --- (b) 逐样本求有效点数 K —— 直接在“点区间”上对插入后 mask 求和（更直观）
        t_rel   = jnp.arange(LM, dtype=jnp.int32)[None, :]                 # [1, LM]
        pt_zone = (t_rel >= (s_all[:, None] + 1)) & (t_rel < (s_all[:, None] + 1 + P))
        # 插入后（含图像前缀）的 mask 在文本+点段的切片：
        m_txtpts = mask[:, Nimg:Nimg+LM]
        K_all = jnp.sum(m_txtpts & pt_zone, axis=1).astype(jnp.int32)      # [B]

        # 可选：对原“pt_zone 计数”做一次对照（仅调试；默认不启用）
        if int(os.environ.get("OPENPI_SONATA_DEBUG_K", "0")):
            t_rel   = jnp.arange(LM, dtype=jnp.int32)[None, :]
            pt_zone = (t_rel >= (s_all[:, None] + 1)) & (t_rel < (s_all[:, None] + 1 + P))
            K_bad   = jnp.sum(mask[:, Nimg:Nimg+LM] & pt_zone, axis=1).astype(jnp.int32)
            # 打印或报警（这里用 logger；如需强制报错可改成 assert jnp.all(K_all==K_bad)）
            try:
                diff = (K_bad - K_all).tolist()
                logger.info("[SONATA-KDBG] K_good(TOT-left-right) vs K_bad(pt_zone): %s", diff)
            except Exception:
                pass

        # --- (c) 构造“插入后”的 labels_text（长度 LM=L+P），再与图像前缀拼接 ---
        labels_full_list = []
        L = observation.tokenized_prompt.shape[1]
        for b in range(B):
            s_idx = s_all[b]; e_idx = e_all[b]; K_b = K_all[b]
            right_len = L - e_idx
            t = jnp.arange(LM, dtype=jnp.int32)
            pt_start = s_idx + 1
            pt_end   = pt_start + K_b
            left_cond   = (t <  pt_start)
            points_cond = (t >= pt_start) & (t < pt_end)
            right_cond  = (t >= pt_end)   & (t < pt_end + right_len)

            txt_idx = jnp.where(left_cond, t, e_idx + (t - pt_end))
            txt_idx = jnp.clip(txt_idx, 0, L - 1)

            # 源 labels：依据 token_loss_mask 过滤；忽略位 = -100，与 HF 一致
            lbl_src = jnp.where(
                observation.token_loss_mask[b].astype(bool),
                observation.tokenized_prompt[b],
                IGNORE,
            )
            lbl_txt = jnp.take(lbl_src, txt_idx, axis=0)
            lbl_txt = jnp.where((left_cond | right_cond), lbl_txt, IGNORE)
            lbl_txt = jnp.where(points_cond, IGNORE, lbl_txt)     # 点位忽略

            lbl_img = jnp.full((Nimg,), IGNORE, dtype=jnp.int32)
            lbl_all = jnp.concatenate([lbl_img, lbl_txt], axis=0)  # [T_total]
            labels_full_list.append(lbl_all)

        labels_full = jnp.stack(labels_full_list, axis=0)  # [B, T_total]

        # --- (d) 计算 NLL：index 方式（避免大 one-hot） ---
        log_probs = jax.nn.log_softmax(logits_all, axis=-1)         # [B, T_total-1, V]
        tgt = labels_full[:, 1:]                                    # 预测的是下一个 token
        valid = (tgt != IGNORE)
        tgt_safe = jnp.clip(tgt, 0, vocab_size - 1)
        picked = jnp.take_along_axis(log_probs, tgt_safe[..., None], axis=-1).squeeze(-1)  # [B, T_total-1]
        nll = -jnp.where(valid, picked, 0.0)
        denom = jnp.maximum(jnp.sum(valid, axis=-1), 1)
        seq_loss = jnp.sum(nll, axis=-1) / denom
        return seq_loss

    # --------- B3：稳健获取 Torch Sonata 句柄（供训练脚本 all 模式使用） ---------
    def get_torch_sonata(self) -> torch.nn.Module:
        """
        优先从模块树（ToNNX wrapper）路径获取 Torch Sonata 实例；
        若该路径不可用，则回退至 _torch_sonata（初始化时保存的冗余句柄）。
        """
        try:
            # nnx_bridge.ToNNX(module= _TorchSonataWrapper(...)) 暴露底层 torch 模块
            return self.Sonata.encoder.module.inner
        except Exception:
            handle = getattr(self, "_torch_sonata", None)
            if handle is None:
                raise AttributeError("Torch Sonata handle not found (encoder.module.inner / _torch_sonata).")
            return handle

    def sample_actions(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0
    ) -> _model.Actions:
        """
        Autoregressively sample a sequence of actions (or action tokens) from the model given an observation.
        This uses the model as a prefix model (prefix = images + prompt tokens + optional point tokens),
        then generates additional tokens up to max_decoding_steps or until an EOS token is produced.
        """
        # Preprocess observation (no augmentation, just ensure correct shapes/masks)
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))
        # Embed inputs to get prefix token embeddings and masks
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_inputs(observation, train=False)
        prefix_attn_mask = _pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)
        # Align all prefix sequences to the right (required for caching in prefix)
        prefix_tokens, prefix_mask, prefix_attn_mask = _pi0_fast.left_to_right_align(prefix_tokens, prefix_mask, prefix_attn_mask)
        # ①  ——  与 SpatialLM 一致：prefill_len 明确 cast 为 int32（后续与 lax.iota 等保持类型匹配）
        prefill_size = prefix_tokens.shape[1]   # total sequence length after alignment (prefix padded to some length)
        prefill_len = jnp.sum(prefix_mask, axis=-1).astype(jnp.int32)   # actual prefix length per batch (number of valid tokens)
        prefix_start = prefill_size - prefill_len   # start index of actual prefix tokens after right-align (for each batch)
        # positions of prefix tokens (0-indexed)；与 TF 路径一致：非负夹紧
        prefix_positions = (jnp.cumsum(prefix_mask.astype(jnp.int32), axis=-1) - 1).astype(jnp.int32)
        prefix_positions = jnp.maximum(prefix_positions, 0)
        # Run the LLM in decoding mode to fill KV cache with prefix
        # 与原版 pi0_fast.py 一致：预填充阶段就把注意力掩码右侧 pad 到预期解码长度，避免掩码宽度不一致
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_logits, kv_cache, *_ = self.PaliGemma.llm(
            embedded_prefix=prefix_tokens,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True
        )
        # Start from the last logit of the prefix as the beginning for new generation
        last_logit = prefix_logits[:, -1:]   # [B, 1, V]
        # Placeholder for generated token outputs (initialize with zeros)
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        # 维护 per‑batch done 掩码：已结束样本后续步固定为 EOS（对齐 HF generate 语义）
        def cond_fn(state):
            _rng, _last, _cache, _step, _out, _done = state
            return jnp.logical_and(_step < max_decoding_steps, jnp.any(~_done))

        def body_fn(state):
            rng_key, last_logits, cache, step, out_tokens, done = state
            rng_key, subkey = jax.random.split(rng_key)
            logits_step = last_logits.squeeze(1)
            token = jax.lax.cond(
                temperature > 1e-6,
                lambda key: jax.random.categorical(key, logits_step / temperature, axis=-1),
                lambda key: jnp.argmax(logits_step, axis=-1),
                operand=subkey,
            ).astype(jnp.int32)
            # ★ 已结束样本固定输出 EOS，避免“从 EOS 又解回别的 token”
            eos_id = jnp.int32(_pi0_fast.PALIGEMMA_EOS_TOKEN)
            token = jnp.where(done, eos_id, token)
            out_tokens = out_tokens.at[:, step].set(token)
            done = jnp.logical_or(done, token == eos_id)

            # Gemma‑fast & SpatialLM：位置 = 已填 prefix token 数 + 当前 step
            # ②  ——  positions 统一 int32，可避免 multi‑host sharding “mixed signedness” 报警
            # 与 SpatialLM 完全一致：首个新 token 位置 = prefill_len（下一步依次 +1）
            positions = (prefill_len[:, None] + step).astype(jnp.int32)
            # Gemma‑fast 无 token=kwarg：先嵌入，再 decode 一步
            token_emb = self.PaliGemma.llm(tokens=token[:, None], embed_only=True)
            # 与原 Pi0‑FAST 保持一致：显式传入 causal mask，
            # 屏蔽右对齐前缀左侧 padding 的 KV。
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps, dtype=jnp.int32)[None, None, :]
                >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps, dtype=jnp.int32)[None, None, :]
                < (
                    jnp.broadcast_to(
                        prefill_size + step + 1,
                        (prefix_start.shape[0], 1, 1),
                    )
                ),
            )
            logits, cache, *_ = self.PaliGemma.llm(
                embedded_prefix=token_emb,
                kv_cache=cache,
                positions=positions,
                decode=True,
                mask=mask,  # ★ 新增
            )
            return rng_key, logits, cache, step+1, out_tokens, done

        init_done = jnp.zeros((output_tokens.shape[0],), dtype=bool)
        init_state = (rng, last_logit, kv_cache, jnp.array(0, jnp.int32), output_tokens, init_done)
        _, _, _, final_step, output_seq, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

        # Return the output sequence of tokens as the model's predicted "actions"
        # (In practice, these tokens might represent discretized actions or a planned sequence encoded as text tokens)
        # 避免动态切片：返回定长序列和有效步数，调用侧按 final_step 截取/显示
        return output_seq                   # [B, max_decoding_steps]

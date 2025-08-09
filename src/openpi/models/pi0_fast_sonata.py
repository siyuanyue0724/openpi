"""
+Pi0FAST-Sonata
+--------------
+本文件将 SpatialLM 的 Sonata 点云编码器原样融合到 Pi0-FAST backbone 中。
+
+本补丁切换到“新接口”以对齐 SpatialLM 的实际数据形态：
+  • Observation.point_clouds["pointcloud"] : [B, M, 3 + C] (float32)
+      - [:, :, 0:3]     = grid_coord 体素网格（源数据中可以是 float，但会在模型侧强制 cast->int32；强烈建议上游直接 int32）
+      - [:, :, 3:6]     = coord 连续 xyz (float32)
+      - [:, :, 6:]      = 其它特征 (float32)
+      - 严格契约：feat = [xyz, extras...]，且 feat[:,:3] 必须与 coord 一致（wrapper 严格校验）。
+  • Observation.point_cloud_masks["pointcloud"] : [B] bool
+      - 指示该样本是否提供点云；空帧将跳过 Sonata 前向（直接返回零向量和全 False 掩码）。
+
+与 SpatialLM 的差异：
+  - 核心功能完全等价（同一份 Point dict 喂 Sonata，Fourier + input_proj 一致）。
+  - 我们显式化了“是否存在点云”的帧级掩码；SpatialLM 在数据侧等价处理，这里做成显式契约。
+  - 我们更严格：若 grid/xyz 不满足契约，直接抛错，不做任何“自动修复”或体素化，以避免把错误数据“悄悄修正”。
+数据要求与建议：
+  - grid_coord 必须是非负整数索引；推荐与 num_bins=1280、4 次 stride=2 对应的最末级范围 [0, 80)（内部有 warn），以利 Fourier 归一化。
+"""

# 该版本已知问题（这些问题目前暂时不用立刻解决）：
# 训练时梯度：不会回传到 **Sonata**（pure_callback 非可微）；**Projector 可训练**。
# 若后续需要端到端训练 Sonata，需要把点云分支迁到 JAX（或自定义可微 callback），再讨论 flash‑attn / spconv 的可微替代。
# 性能潜在瓶颈	CPU ↔ GPU copy / 多编译	后续迭代可能会影响这个，所以首先解决问题1
# 1024 token size，这个暂时不能设置太大因为会炸显存，首先使用提示方式来确定是否会有过多的情况，没有就继续训练，有的话后续再继续处理
# 注意，grid似乎不能是负数！
# per-sample pure_callback batch>1 时 CPU↔GPU 来回和 XLA → host 交互会拖慢，梯度积累场景尤甚
# grid → coord 偏移：当前不会在模型内改动 grid_coord（只做“非负 + 形状 + dtype”校验）；是否归零或对齐，请在数据侧统一处理。
# 为了避免警告，实施了JAX 端 batch/offset 用 int32，host(PyTorch) 端统一 .long()；避免 JAX_ENABLE_X64 相关警告。
# 【这个似乎修复了？】“插入位置”严格一致性问题存在：我们是前缀拼接；SpatialLM 是 <point_start>..点token.. <point_end> 插回到文本序列。语义等价（文本依旧能看到点 token），但不是完全同一位置。如果你要逐字节一致，需要让 tokenizer/prompt 中真的包含 <point_start>/<point_end>，并在拼接时找到这两个位置再做插入（成本较高，且对你当前 Pi0‑FAST 的多模态拼接接口不自然）。


import dataclasses
import inspect
import logging
import typing
import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import torch
import warnings
from pathlib import Path
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
# from openpi.shared import download     # utility for downloading resources like weights, not used for now

# optional – hug‑hub 优先
try:
    from huggingface_hub import hf_hub_download
    _HF_OK = True
except ImportError:          # 环境里没装 huggingface_hub 会走旧逻辑
    _HF_OK = False
logger = logging.getLogger("openpi")

# 用于生成测试用数据的工具函数
def _canonicalize_point_dict(pd):
    pd = {k: jnp.asarray(v) for k, v in pd.items()}  # ← 用 jnp；不做任何“数据修复”，只做形状规范化与 dtype 约束
 

    if pd["coord"].ndim == 3:      # (B,N,3) → (B*N,3)
        B, N, _ = pd["coord"].shape
        pd["coord"] = jnp.reshape(pd["coord"], (B * N, 3))
        pd["feat"]  = jnp.reshape(pd["feat"],  (B * N, -1))
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
        # 这里有个潜在的问题，若用户自己组装的grid_coord已经超过65535，则可能再次溢出，但这个实际上不太可能，因此只写注释，等以后如果训练有问题再回来看。
        if pd["grid_coord"].shape[-1] != 3:
            raise ValueError("pointcloud.grid_coord 的最后一维必须为 3（xyz）。")
        pd["grid_coord"] = pd["grid_coord"].astype(jnp.int32, copy=False)

    return pd

# ---------- host-side helper: find <point_start>/<point_end> & keep indices ----------
def _host_find_window_and_keep_idx(
    prompt_np: np.ndarray,
    start_id: int,
    end_id: int,
    *,
    allow_text_between: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 host 上解析一个样本的 tokenized_prompt，返回:
      • window: np.int32[2] = [s_idx, e_idx]
      • keep_idx: np.int32[L-2] —— 去掉 s/e 两个位置后，保留的 L-2 个文本位置索引
    若未找到或不合法，抛出异常。
    """
    arr = np.asarray(prompt_np).tolist()
    L = len(arr)
    s_pos = [i for i, t in enumerate(arr) if t == start_id]
    e_pos = [i for i, t in enumerate(arr) if t == end_id]
    if len(s_pos) != 1 or len(e_pos) != 1:
        raise ValueError(
            f"[point-window] expect exactly one <start>/<end>, got start={s_pos}, end={e_pos}"
        )
    s, e = s_pos[0], e_pos[0]
    if not (0 <= s < e < L):
        raise ValueError(f"[point-window] invalid order: start={s}, end={e}, L={L}")
    if not allow_text_between and e != s + 1:
        raise ValueError(
            f"[point-window] found text between <start> and <end> (start={s}, end={e}). "
            "Set allow_text_between_markers=True if this is intentional."
        )
    keep = np.array([i for i in range(L) if i != s and i != e], dtype=np.int32)
    win = np.array([s, e], dtype=np.int32)
    return win, keep

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
    return np.array([s, e], dtype=np.int32)

# -------- 在“新接口 / 旧接口(legacy)”两类 Observation 之间统一抽取点云 ----------
def _extract_point_batch(obs) -> tuple[dict[str, jnp.ndarray], jnp.ndarray] | None:
    """
    返回 (pc_dict, frame_mask) 或 None
    优先走“新接口”，与 SpatialLM 的 Sonata 调用语义等价：
      • obs.point_clouds["pointcloud"] : [B, M, 3 + C]
          - [:, :, :3]  = grid_coord (期望 int32，非负；这里会 cast)
          - [:, :, 3:6] = coord xyz   (float32)
          - [:, :, 6:]  = extras      (float32)
      • obs.point_cloud_masks["pointcloud"] : [B]  指示该样本是否有点云

    仍兼容 legacy（obs.pointcloud_data 字典），但不再在模型内“体素化/修复”，
    且会从 offset/batch 推导帧级掩码以与新接口保持一致语义。

    支持两种来源：
    1) 新接口 :  obs.point_clouds["pointcloud"]  +  obs.point_cloud_masks["pointcloud"]
    2) 旧接口 :  obs.pointcloud_data
       - a) 已是 Sonata 兼容 dict（必须含 grid_coord）→ 直接使用
       - b) 不再支持仅 [B,P,C] 原始数组（避免“越权修复”）
    """
    # ---------- 💡 新接口 ----------
    if hasattr(obs, "point_clouds") and "pointcloud" in getattr(obs, "point_clouds"):
        # 形状: [B, M, 3(grid) + point_feat_dim(feats)]
        # [:,:,0:3] grid (cast→int32), [:,:,3:6] 连续 xyz, [:,:,6:] 其它特征
        pc_arr  = obs.point_clouds["pointcloud"]
        pc_mask = obs.point_cloud_masks["pointcloud"]     # [B]

        B, M, _ = pc_arr.shape
        # SpatialLM‑Qwen 约定:
        #   0‑2 : 体素网格坐标 (int32)
        #   3‑5 : 连续 xyz (float32)
        #   6+ : 其他语义特征
        grid_int = pc_arr[..., :3].astype(jnp.int32)           # (B,M,3)
        coords   = pc_arr[..., 3:6].astype(jnp.float32)        # 连续 xyz
        feats    = pc_arr[..., 3:].astype(jnp.float32)         # xyz + 语义

        # 不在 JAX 侧做“修复”；展平/NaN 过滤等留到 host 侧（wrapper 内）做硬校验
        pc_dict = _canonicalize_point_dict(
            dict(coord=coords, grid_coord=grid_int, feat=feats)
        )

        # 直接使用静态维度 M 作为 pad 长度，避免 run‑time int()
        # 只返回帧级掩码 [B]；在 embed_inputs 中再按 token 维广播
        return pc_dict, pc_mask

    # ---------- 旧接口 ----------
    legacy = getattr(obs, "pointcloud_data", None)
    if legacy is None:
        return None

    # a) 已经是 dict
    if isinstance(legacy, dict) and "coord" in legacy:
        pc_dict = _canonicalize_point_dict(legacy)
        # 强契约：legacy-dict 必须已是 Sonata 兼容格式；不得在模型内“推断/修复”。
        # 1) 需要显式提供 grid_coord（N,3）并且为 int32 非负
        if "grid_coord" not in pc_dict:
            raise ValueError(
                "Legacy dict 缺少 grid_coord。严格对齐 SpatialLM：请在上游显式提供非负 int32 体素坐标 (N,3)。"
            )
        # 2) offset/batch 若缺省可由 wrapper 按 batch 重建；但不做任何值域修复
    else:
        # b) 旧接口若只是 [B,P,C] 原始数组：不再在模型内体素化/重建 grid（避免“越权修复”）。
        raise ValueError(
            "Legacy path 收到 [B,P,C] 数组，但严格模式下模型不再代替你体素化/构造 grid。"
            "请在上游把点云转换为 Sonata 兼容 dict，显式提供 grid_coord(int32,非负)、coord(float32,xyz)、feat([xyz,...])。"
        )

    # 旧接口：从 offset 或 batch 推导每帧是否存在点云（与新接口的帧级语义一致）
    if "offset" in pc_dict:
        off = pc_dict["offset"].astype(jnp.int32)
        B = off.shape[0]
        counts = jnp.diff(jnp.pad(off, (1, 0)))
        mask = counts > 0
        return pc_dict, mask
    if "batch" in pc_dict:
        b = pc_dict["batch"].astype(jnp.int32)
        B = int(jnp.max(b)) + 1 if b.size else 0
        counts = jnp.bincount(b, length=B)
        mask = counts > 0
        return pc_dict, mask
    raise ValueError("Legacy dict 需包含 'offset' 或 'batch' 以推断 batch 尺寸。")

# Alias the Sonata class from the sonata_encoder module for convenience
Sonata = sonata_encoder.Sonata

@dataclasses.dataclass(frozen=True)
class Pi0FASTSonataConfig(_pi0_fast.Pi0FASTConfig):
    """Configuration for the Pi0FASTSonata model (Pi0FAST with Sonata point cloud encoder)."""

    # === SpatialLM 契约 ===
    # point_feat_dim 定义为传入 Sonata 的 feats 维度（不含 grid）：
    # feats = [xyz(3), extra...]，例如 xyzrgb ⇒ point_feat_dim = 6
    # Observation.pointcloud 的最后一维 = 3(grid) + point_feat_dim
    point_feat_dim: int = 6
    # 每帧点数的静态上界（仅用于 inputs_spec 的形状声明；不会截断实际输入）
    max_points: int = 32768
    # 让父类 Pi0FASTConfig.inputs_spec() 按原逻辑自动注入点云字段
    # （Pi0FASTSonata 本身不会用这个枚举去构建模块，只用于 spec）
    point_backbone_type: PointBackboneType = PointBackboneType.SONATA
    # projector_type 对 spec 无影响，这里保持 None 或 LINEAR 均可
    projector_type: ProjectorType | None = None

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Inherits default action_dim, action_horizon, max_token_len from Pi0FASTConfig (e.g., 32, 32, 250)
    use_pretrained_point: bool = True      # 调试阶段可设 False 跳过下载
    # 若使用 Sonata，则默认**必须**提供点云；缺失时直接报错而不是静默降级
    # 若需要只用图像+文本做消融，可把该开关设为 False
    require_pointcloud: bool = True
    # 若无图像则直接不训练（训练时报错终止；推理不受影响，这里主要用于进行调试确保数据集正确）
    require_image: bool = True
    # === 原位插入（SpatialLM 对齐） ===
    insert_points_in_place: bool = True
    # 这两个 id 必须与你的 tokenizer 中的 special tokens 一致
    point_start_id: Optional[int] = None   # 例如 tokenizer("<|point_start|>") 的 id
    point_end_id:   Optional[int] = None   # 例如 tokenizer("<|point_end|>")   的 id
    # 默认不允许 start/end 之间仍有人类文本；若你的数据中确实存在，可设 True
    allow_text_between_markers: bool = False
    # 若为 True，则在原位插入路径严格复刻 SpatialLM：
    #   • 保留 <start>/<end>；• 删除中间文本；• 在 <start> 后插入 K 个点 token；
    spatiallm_exact: bool = True

    @property
    def model_type(self) -> _model.ModelType:
        # Reuse the PI0_FAST model type (since this is an extension of Pi0-FAST architecture)
        return _model.ModelType.PI0_FAST

    def create(self, rng: jax.Array) -> "Pi0FASTSonata":
        """Instantiate a Pi0FASTSonata model with random initialization."""
        return Pi0FASTSonata(self, rngs=nnx.Rngs(rng))

# ---- 覆写 inputs_spec：切换到“新接口”，与 SpatialLM 的 Sonata 输入契约等价 ----
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                # 点云（新接口）：与 SpatialLM 的 Sonata 调用保持等价语义
                #  - point_clouds["pointcloud"]: [B, M, 3 + C] (float32)
                #      [:,:,0:3] = grid_coord（期望 int32，非负；此处按 float32 规格声明，运行时会 cast）
                #      [:,:,3:6] = coord xyz（float32）
                #      [:,:,6:]  = 其它特征（float32）
                #  - point_cloud_masks["pointcloud"]: [B] bool  —— 指示该样本是否存在点云
                point_clouds={"pointcloud": jax.ShapeDtypeStruct([batch_size, self.max_points, 3 + self.point_feat_dim], jnp.float32)},
                point_cloud_masks={"pointcloud": jax.ShapeDtypeStruct([batch_size], jnp.bool_)},
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
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
        # 设定统一 device（GPU 如果可用，否则 CPU）
        # --------------------------------------------------------------
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # ------------------------------------------------------------------
        # 1) 语言模型  Gemma  ------------------------------------------------
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Gemma‑fast LLM ----------------------------------------------------
        # ------------------------------------------------------------------
        pal_cfg = _gemma.get_config(config.paligemma_variant)

        # 根据 gemma_fast.Module 的 __init__ 签名自动收集需要的参数
        gemma_sig = inspect.signature(_gemma.Module.__init__)
        pal_kwargs: dict[str, typing.Any] = {}
        for name in gemma_sig.parameters:
            if name == "self":          # 跳过 self
                continue
            if hasattr(pal_cfg, name):
                pal_kwargs[name] = getattr(pal_cfg, name)

        # 若 get_config 没返回 variant，则用传入的枚举字符串补齐
        pal_kwargs.setdefault("variant", config.paligemma_variant)

        # 其它 dtype 相关覆写
        pal_kwargs["embed_dtype"] = config.dtype
        pal_kwargs["cache_dtype"] = config.dtype

        llm = nnx_bridge.ToNNX(_gemma.Module(**pal_kwargs))

        # 初始化
        llm.lazy_init(rngs=rngs, method="init")

        # ------------------------------------------------------------------
        # 2) 图像编码器  SigLIP  --------------------------------------------
        # ------------------------------------------------------------------
        _model_width = getattr(pal_cfg, "width", getattr(pal_cfg, "hidden_size", 1024))
        
        raw_img_kwargs = dict(
            # _siglip.Module 可能不需要 num_classes；如果 signature 里没有会被自动丢弃
            num_classes=_model_width,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=config.dtype,
        )

        # ---------------------------------------------------------------------------------

        img = nnx_bridge.ToNNX(_siglip.Module(**raw_img_kwargs))

        # Initialize image encoder with a dummy image to set dimensions
        dummy_image = next(iter(config.fake_obs(batch_size=1).images.values()))
        img.lazy_init(dummy_image, train=False, rngs=rngs)

        # ------------------------------------------------------------------
        # 3) 创建并加载点云编码器 (Sonata) — 参数对齐 SpatialLM‑1.1, 后面可以改成config传输
        # ------------------------------------------------------------------
        
        # ---------- Sonata hyper‑params – 与 ckpt 保持 100 % 一致 ----------
        # in_channels = feats 维度（不含 grid 的3列）
        if not hasattr(config, "point_feat_dim"):
            raise ValueError(
                "Pi0FASTSonataConfig 需显式提供 point_feat_dim（= feats 维度，xyz+extras，不含 grid）"
            )
        # 与 SpatialLM 契约一致：in_channels = feats 列数（含 xyz，不含 grid）
        _in_channels = int(config.point_feat_dim)  # e.g., xyzrgb -> 6
        if _in_channels < 3:
            raise ValueError(
                f"配置 point_feat_dim={config.point_feat_dim} 无效，须 ≥ 3（至少包含连续 xyz）"
            )
        # 若后续输入点云特征维与配置不符，将在 embed_inputs 早期报错

        # 若未安装 flash_attn，自动禁用以避免断言失败
        _enable_flash = (getattr(sonata_encoder, "flash_attn", None) is not None)
        if not _enable_flash and not getattr(Pi0FASTSonata, "_warned_no_flash", False):
            logger.warning("[Sonata] flash-attn not found; falling back to non-flash path (slower, it is highly recommended to use flash-attn).")
            Pi0FASTSonata._warned_no_flash = True
        sp_cfg = dict(
            in_channels   = _in_channels,
            order         = ("z", "z-trans"),
            stride        = (2, 2, 2, 2),         # 5‑stage ⇒ 4 次下采样
            enc_depths    = (3, 3, 3, 12, 3),
            enc_channels  = (48, 96, 192, 384, 512),   # ★ 末端 512
            enc_num_head  = (3, 6, 12, 24, 32),
            enc_patch_size= (1024,)*5,            # ckpt 默认
            mlp_ratio     = 4.0,
            mask_token    = True,                   # Sonata.Embedding 中 mask_token 功能被禁用，此参数无效
            enc_mode=True,  # 即voxel
            enable_fourier_encode = True,         # ★ ckpt 含 fourier+input_proj
            num_bins      = 1280,
            enable_flash  = _enable_flash,
        )
        point_model = Sonata(**sp_cfg)
        # 记录供后续断言／Projector 使用
        self._point_in_channels = _in_channels  # feats 维度（xyz+extras）
        # 与 SpatialLM 一致：proj 输入维 = 编码器末层通道
        self._enc_out_dim = sp_cfg["enc_channels"][-1]

        if config.use_pretrained_point:
            # ------------------------------------------------------------------
            # 仅使用本地精简后的 SpatialLM1.1 Sonata 权重
            # 文件放置: <repo_root>/openpi/pretrain/SpatialLM_Sonata_encoder.pth
            # 如果没有模型，请使用uv run scripts/sonata_weight_gen.py来获取sonata的checkpoint
            # ------------------------------------------------------------------
            # 路径指向 <repo_root>/src/pretrain/
            ckpt_path = (
                Path(__file__).resolve().parents[2]  # -> …/openpi/src
                / "pretrain" / "SpatialLM_Sonata_encoder.pth"
            )
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Sonata 预训练权重未找到: {ckpt_path}\n"
                    "请先运行 scripts/sonata_weight_gen.py 生成文件，"
                    "并放置到 src/pretrain/ 目录。"
                )
            logger.info("Using local Sonata weights: %s", ckpt_path)


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

            # ②‑bis：按权重“推断” in_channels，一致性保护
            wkey = "embedding.stem.linear.weight"
            if wkey in cleaned:
                in_from_ckpt = cleaned[wkey].shape[1]  # [out, in]
                if in_from_ckpt != self._point_in_channels:
                    logger.warning(
                        "Sonata ckpt in_channels=%d != configured point_feat_dim=%d; "
                        "rebuilding Sonata to match checkpoint (prefer checkpoint to avoid shape mismatch).",
                        in_from_ckpt, self._point_in_channels
                    )
                    # 重新构建与权重一致的 Sonata，再载入
                    sp_cfg_ckpt = {**sp_cfg, "in_channels": int(in_from_ckpt)}
                    point_model = Sonata(**sp_cfg_ckpt)
                    point_model.to(self.device).eval()
                    self._point_in_channels = int(in_from_ckpt)
                    self._enc_out_dim = sp_cfg_ckpt["enc_channels"][-1]

            # ③部分匹配即可；strict=False 会跳过多余键，也会提示哪些没加载
            ik = point_model.load_state_dict(cleaned, strict=False)
            if getattr(ik, "missing_keys", None):
                mk = ik.missing_keys
                logger.warning("Sonata weights: %d missing params, e.g. %s", len(mk), mk[:5])
            if getattr(ik, "unexpected_keys", None):
                uk = ik.unexpected_keys
                logger.warning("Sonata weights: %d unexpected params, e.g. %s", len(uk), uk[:5])

            logger.info("Loaded pretrained Sonata weights from %s", ckpt_path)

        else:
            logger.warning(
                "Pi0FAST‑Sonata: use_pretrained_point=False -> "
                "Sonata encoder starts with random weights."
            )

        # -------- 保证 Sonata 整个网络在 GPU（含 spconv kernel） --------
        point_model.to(self.device)
        point_model.eval()  # inference mode

        # ------------------------------------------------------------------
        # 4) 定义 _TorchSonataWrapper —— 保持原 API，新增 GPU 支持
        # ------------------------------------------------------------------
        class _TorchSonataWrapper(torch.nn.Module):
            """
            - 追踪期（JIT / lazy_init）只返回 ShapeDtypeStruct，不跑真模型。
            - 运行期通过 pure_callback 把 numpy → torch.cuda → numpy。
            """
            def __init__(
                self,
                pt_model: torch.nn.Module,
                device: torch.device,
                patch_size: int,
                enc_out_dim: int,
            ):
                super().__init__()
                # 确保 pt_model 已在目标 device
                self.inner = pt_model.to(device).eval()
                self.device = device
                self.patch_size = patch_size
                self.enc_out_dim = enc_out_dim

            # ---------- 内部 util ----------
            @staticmethod
            @torch.no_grad()
            def _torch_forward(
                inner: torch.nn.Module,
                host_dict: Dict[str, np.ndarray],
                device: torch.device,
                patch_size: int,
                enc_out_dim: int,
            ) -> Tuple[np.ndarray, np.ndarray]:  # 第 2 个 ndarray 是 bool 掩码
                """
                接收 host 上的 numpy 输入 → torch.Tensor.cuda → 运行 → numpy 输出
                统一返回 float32 numpy 数组
                """

                # ---------- fast‑path：该帧不存在，直接返回全零 ----------
                present = int(host_dict.pop("present", 1))
                if present == 0:
                    pad_feat   = np.zeros((patch_size, enc_out_dim), dtype=np.float32)
                    valid_mask = np.zeros((patch_size,),       dtype=bool)
                    return pad_feat, valid_mask
                
                # =========================================================
                # ★ 新增：若上游传入 selected_batch，只保留这一帧的点
                #   (这样 embed_inputs 里不用 v[sel]，避免 JAX 布尔切片报错)
                # ---------------------------------------------------------
                if "selected_batch" in host_dict:
                    sb = int(host_dict.pop("selected_batch"))
                    if "batch" in host_dict:
                        b_arr = np.asarray(host_dict["batch"]).reshape(-1)
                        sel = (b_arr == sb)                     # sel.shape == (B*M,)
                        # 1) 先按点级别切所有“展平”的张量（coord/feat/batch 等）
                        for k, v in list(host_dict.items()):
                            if hasattr(v, "shape") and v.ndim and v.shape[0] == sel.shape[0]:
                                host_dict[k] = v[sel]
                        # 2) 再对“按帧打包”的张量切第 0 维（典型：grid_coord[B,M,3]、grid_size[B,3]）
                        #    推断 B：优先用 offset 长度，否则用 batch 中的最大值+1
                        if "offset" in host_dict:
                            B = int(np.asarray(host_dict["offset"]).reshape(-1).shape[0])
                        else:
                            B = int(b_arr.max()) + 1 if b_arr.size else 0
                        if B:
                            for k, v in list(host_dict.items()):
                                if hasattr(v, "shape") and v.ndim >= 2 and v.shape[0] == B:
                                    host_dict[k] = v[sb]
                    else:
                        # 无 batch 键：按第 0 维是 batch 轴处理（v 的形状形如 (B, M, ...) 或 (B, 3)）
                        for k, v in list(host_dict.items()):
                            if hasattr(v, "shape") and v.ndim >= 2 and v.shape[0] > sb:
                                host_dict[k] = v[sb]
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
                    if np.any(gc >= reduced_gs) and not getattr(_TorchSonataWrapper, "_warned_grid_upper", False):
                        warnings.warn(
                            f"[Sonata] grid_coord 超过 reduced_grid_size={reduced_gs}；如出现精度/性能异常，请检查体素化配置。",
                            RuntimeWarning
                        )
                        _TorchSonataWrapper._warned_grid_upper = True
                
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
                
                # ===== SpatialLM‑style 空点云保护 =====
                if host_dict["coord"].shape[0] == 0:
                    # 返回全零特征 + 全 False 有效标记（维度与正常路径一致）
                    pad_feat   = np.zeros((patch_size, enc_out_dim), dtype=np.float32)
                    valid_mask = np.zeros((patch_size,),       dtype=bool)
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
                if real_len > patch_size:
                    raise RuntimeError(
                        f"[SpatialLM‑Sonata] token_len={real_len} 超过上限 patch_size={patch_size}。"
                        "请增大 Sonata enc_patch_size[-1]（会增显存）或在上游减少每样本点数。"
                    )
                # SpatialLM: 不截断；右 pad 到 patch_size 的倍数
                # --- 保留固定 1024‑padding 以维持静态 shape (JAX 需求，改成不固定的代价很大，如果单纯加大则容易炸显存) ---
                # ------------------------------------------------------------------
                # 截断前做显式报警：这个是因为如果我们要改成非固定，会导致无法jax化，计算代价很大，所以我们用这种方式进行测试，来看看数据集能不能提供合理数据
                # ------------------------------------------------------------------
                MAX_TOKEN = patch_size                  # =1024 (enc_patch_size[-1])
                if real_len > MAX_TOKEN:
                    # 直接阻断：用户必须显式提高 enc_patch_size[-1] 或减少点数
                    raise RuntimeError(
                        f"[Sonata] token_len={real_len} exceeds configured MAX_TOKEN "
                        f"({MAX_TOKEN}). Increase `enc_patch_size[-1]` or reduce the "
                        "number of points per sample."
                    )
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
                # 例如静态上限, 这里设置1024，可以设置8192 (= 8 patch)；确保不会在真实输入中超出但是这样会炸显存,所以我们保留1024实现, 后续可以继续加
                # patch_size 由 Wrapper 构造函数传入；保持与 Sonata enc_patch_size 一致
                MAX_TOKEN = self.patch_size
                C = self.enc_out_dim
                out_struct = (ShapeDtypeStruct((MAX_TOKEN, C), jnp.float32),
                              ShapeDtypeStruct((MAX_TOKEN,),  jnp.bool_))

                def _host_call(*flat_np):
                    # flat_np 是回传的扁平列表/元组，需用 treedef.unflatten 还原
                    np_dict = treedef.unflatten(list(flat_np))
                    return self._torch_forward(
                        self.inner, np_dict, self.device, self.patch_size, self.enc_out_dim
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
                
        # ------------------------------------------------------------------
        # 4‑bis) 线性层包装器：让普通 Linear 支持 lazy_init
        # ------------------------------------------------------------------
        class _TorchLinearWrapper(torch.nn.Module):
            def __init__(self, in_dim: int, out_dim: int, device: torch.device, *, bias: bool = True):
                super().__init__()
                self.inner = torch.nn.Linear(in_dim, out_dim, bias=bias).to(device)
                self.device = device

            # -------- host‑side计算 --------
            @staticmethod
            @torch.no_grad()
            def _torch_forward(inner: torch.nn.Linear,
                               mat_np: np.ndarray,
                               device: torch.device) -> np.ndarray:
                """
               纯 host 调用：np -> torch(cuda) -> np
                """
                # 保证一定是 NumPy，再送入 PyTorch
                if not isinstance(mat_np, np.ndarray):
                    mat_np = np.asarray(mat_np)
                x = torch.from_numpy(mat_np).to(inner.weight.dtype).to(device)
                y = inner(x)
                return y.cpu().numpy()

            # -------- forward（JAX side）--------
            def forward(self, x_jax: jax.Array, *, train: bool = False):
                """
                • 在 jit 图里以 pure_callback 方式调用 _torch_forward  
                • 输出保持 float32，后续再 cast
                """
                # 计算输出形状（完全符号化，不触发 concrete）
                out_shape = (*x_jax.shape[:-1], self.inner.out_features)
                out_struct = ShapeDtypeStruct(out_shape, jnp.float32)

                def _host_call(mat):
                    # mat 是 numpy.ndarray （pure_callback 已转换）
                    return self._torch_forward(self.inner, mat, self.device)

                return pure_callback(_host_call, out_struct, x_jax, vectorized=False)

            # -------- lazy_init hook --------
            def init_with_output(self, rngs, x_np, *, method=None, **_):
                """
                lazy_init 阶段不会在 jit 内，
                直接走 host‑side 计算即可，加速初始化。
                """
                if isinstance(x_np, jax.Array):
                    x_np = np.asarray(jax.device_get(x_np))
                y = self._torch_forward(self.inner, x_np, self.device)
                return y, {}          # no trainable vars

            # -------- nnx‑bridge 兼容 --------
            def apply(self, _vars, *args, rngs=None, method: str | None = "forward", **kw):
                if method is None:
                    method = "forward"
                fn = getattr(self, method)
                if method == "init_with_output":
                    return fn(rngs, *args, **kw)
                return fn(*args, **kw)

        # ---------- 5) 原生 NNX 线性投影：支持端到端反向传播、避免 host 往返 ----------
        # enc_out_dim → PaLI‑Gemma hidden_size
        self.PointProjector = nnx.Linear(self._enc_out_dim, _model_width, rngs=rngs)

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

        _patch_sz = sp_cfg["enc_patch_size"][-1]   # 1024 (保持与 ckpt 一致)
        point = nnx_bridge.ToNNX(
            _TorchSonataWrapper(
                point_model,
                self.device,
                _patch_sz,
                self._enc_out_dim,   # ← 与 __init__ 对齐
            )
        )
        # 记录 point block 的固定长度（= Sonata enc_patch_size[-1]，默认 1024）
        self._pt_block_len = int(_patch_sz)
        
        # 明确使用 wrapper 的 init_with_output 以避免在不同 nnx-bridge 版本下的歧义
        point.lazy_init(dummy_pc, train=False, rngs=rngs, method="init_with_output")
        # ↑ 若你的 nnx 版本会自动识别，则这行与上一行等价；显式化更稳

        # ------------------------------------------------------------------
        # 6) 打包所有子模块
        # ------------------------------------------------------------------
        self.PaliGemma = nnx.Dict(
            llm   = llm,
            img   = img,
            point = point,
            point_proj = self.PointProjector,
        )
        # 插入策略与 special ids
        self._insert_points_in_place = bool(getattr(config, "insert_points_in_place", True))
        self._point_start_id = getattr(config, "point_start_id", None)
        self._point_end_id   = getattr(config, "point_end_id", None)
        self._warned_no_point_ids = False
        self._allow_text_between = bool(getattr(config, "allow_text_between_markers", False))
        self._spatiallm_exact = bool(getattr(config, "spatiallm_exact", True))

    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
            img_t, _ = self.PaliGemma.img(image, train=False)       # [B, n_img_tok, D]
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

        # 2. Point cloud tokens --------------------------------------------------
        pc_pack = _extract_point_batch(obs)
        # 若本模型启用了 Sonata（即本类本身）且配置要求点云，则缺失时直接报错
        if pc_pack is None and self._require_pointcloud:
            raise ValueError(
                "Pi0FAST‑Sonata: 未提供点云，但配置 require_pointcloud=True。"
                "请提供 Observation.point_clouds['pointcloud']（及 mask）或 Observation.pointcloud_data；"
                "若确需只用图像+文本，请将 Pi0FASTSonataConfig.require_pointcloud=False。"
            )
        if pc_pack is not None:
            # --------------------------------------------------------
            # 与 SpatialLM‑Qwen forward_point_cloud 完全一致的策略：
            # for‑loop 逐样本调用 Sonata → 每次得到 (K*1024, C)，
            # 再统一 pad 到同一长度。
            # --------------------------------------------------------
            pc_dict_all, pc_frame_mask = pc_pack        # pc_frame_mask : [B, M]
            # 如果点云也声明 require_pointcloud=True，但 pc_frame_mask 全 False，这一批数据应直接失败，而不是把点云当作不存在（防止悄悄训练在错误分布上）
            if self._require_pointcloud:
                # 【修复】禁止在 jit 中做 Python 布尔判断；改为 host 断言（运行期）：
                # present_any 为 0-d bool（traced），用 pure_callback 在 host 上判断并抛异常（必要时）。
                present_any = jnp.any(pc_frame_mask.astype(bool))

                def _host_assert_present(x):
                    # x: numpy 0-d bool
                    x = bool(np.asarray(x))
                    if not x:
                        raise RuntimeError(
                            "Pi0FAST‑Sonata: require_pointcloud=True 但该 batch 所有样本均无点云。"
                        )
                    # 返回一个虚占位（满足 pure_callback 的返回契约）
                    return np.int32(0)

                # 占位输出描述
                _ = pure_callback(
                    _host_assert_present,
                    ShapeDtypeStruct((), jnp.int32),
                    present_any,
                    vectorized=False,
                )
            # B 的鲁棒推断：优先 offset；否则退化到 mask / prompt 的 batch 维
            if "offset" in pc_dict_all:
                B = int(pc_dict_all["offset"].shape[0])
            elif pc_frame_mask.ndim in (1, 2):
                B = int(pc_frame_mask.shape[0])
            else:
                B = int(obs.tokenized_prompt.shape[0])

            # 运行时维度检查：feat = xyz(3) + extra(...)，其列数应等于 in_channels
            feat_dim = int(pc_dict_all["feat"].shape[-1])            # = feats 维度（xyz+extras）
            expected_feat_dim = int(self._point_in_channels)          # = config.point_feat_dim
            if feat_dim != expected_feat_dim:
                raise ValueError(
                    "点云特征维度不匹配：期望 feats 维度（xyz+extra）= "
                    f"{expected_feat_dim}，但收到 {feat_dim}。"
                    "请确保 Observation.pointcloud 的前3列为 grid，后续列为 feats=[xyz, extras]；"
                    "Sonata 的 in_channels=feats 列数（不含 grid）。"
                )

            per_sample_tokens  = []
            per_sample_masks   = []
            max_len            = 0

            for b in range(B):                                    # **逐 batch**
                # 把 sample id + present 传给 wrapper；真正切片在 PyTorch 侧完成
                if pc_frame_mask.ndim == 2:
                    present_b = pc_frame_mask[b].any()
                else:
                    present_b = pc_frame_mask[b]
                single_dict = {
                    **pc_dict_all,                                  # 全量点云
                    "selected_batch": jnp.array(b, jnp.int32),      # 指定样本
                    "present": present_b.astype(jnp.int32),         # 帧存在标志
                }
                # 早期形状检查：grid_coord 必须最后一维=3，这是为了测试数据集是不是真正的x,y,z,r,g,b,方便后续测试
                if "grid_coord" in single_dict:
                    if single_dict["grid_coord"].shape[-1] != 3:
                        raise ValueError(
                            f"pointcloud grid_coord 的最后一维必须为 3，"
                            f"实际 shape={single_dict['grid_coord'].shape}"
                        )
                # 严格模式：不再在模型内重建 grid；缺失立刻报错（保持与 SpatialLM 契约一致）
                if "grid_coord" not in single_dict:
                    raise ValueError(
                        "[SpatialLM‑Sonata] 当前样本缺少 grid_coord；请在数据管线中显式提供体素索引 (N,3,int32,非负)。"
                    )

                tok, vmask = self.PaliGemma.point(single_dict, train=False)

                # -------- 长度一致性断言 --------
                assert tok.shape[0] == vmask.shape[0], (
                    f"Sonata 返回 token 长度 {tok.shape[0]} "
                    f"≠ valid_mask 长度 {vmask.shape[0]}"
                )

                # 投影到 LLM hidden_size ，保持与图像 / 文本维度一致
                tok = self.PointProjector(tok.astype(jnp.float32))
                # 把 padding / 无效 token 特征强制归零，防止 Linear 偏置泄漏
                tok = tok * vmask[:, None]

                # SpatialLM 保留补零 token，靠 vmask 指示有效性
                per_sample_tokens.append(tok)       # ← 直接整块保存
                per_sample_masks.append(vmask)
                max_len = max(max_len, tok.shape[0])   # 一般就是 1024

            # ----- pad 到 batch 内最大长度 -----------
            def _pad_to(x, tgt):
                pad = [(0, tgt - x.shape[0])] + [(0, 0)]*(x.ndim-1)
                return jnp.pad(x, pad)

            pt_tokens = jnp.stack([_pad_to(t, max_len) for t in per_sample_tokens])  # (B,P,C) P=max_len(=patch_size)
            valid_m   = jnp.stack([_pad_to(m, max_len) for m in per_sample_masks])   # (B,P)

            # 二次验证：所有 batch 长度已经对齐
            assert (pt_tokens.shape[1] == valid_m.shape[1] == max_len), \
                "pad 后 token / mask 长度不一致"

            # --- 帧级掩码：先把 [B] 或 [B, M] 压成 [B,1]（是否存在该模态），再广播到 token 维 ---
            frame_present = (
                pc_frame_mask.any(axis=1, keepdims=True)  # [B,1] 适配 [B,M] 情况
                if pc_frame_mask.ndim == 2 else
                pc_frame_mask[:, None]                   # [B,1]
            )
            pc_frame_mask_b = jnp.broadcast_to(frame_present, (B, max_len))  # [B, max_len]
            pt_final_mask   = pc_frame_mask_b & valid_m

            # 数值更干净：把最终 mask 也乘进特征（仅用于可视化；下面原位插入时仍会再写入一次）
            pt_tokens = pt_tokens * pt_final_mask[:, :, None]
        else:
            # 没有点云：构造空 block（长度 = 0；但通常 require_pointcloud=True 不会走到这里）
            B = obs.tokenized_prompt.shape[0]
            D = int(getattr(self.PaliGemma.llm.module, "hidden_size", 1024))
            pt_tokens = jnp.zeros((B, 0, D), dtype=jnp.float32)
            pt_final_mask = jnp.zeros((B, 0), dtype=bool)

        # 3. Text tokens (from language model embedding)
        # Ensure textual inputs are present
        assert obs.tokenized_prompt is not None and obs.tokenized_prompt_mask is not None and obs.token_ar_mask is not None, \
            "Tokenized prompt and corresponding masks must be provided for text inputs."
        txt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)  # [B, L, emb_dim]
        # 避免如果错误地给了空prompt或只有1个token，在后面逻辑里产生“0长度但不报错”
        if obs.tokenized_prompt.shape[1] < 2:
           raise ValueError("tokenized_prompt 长度必须 ≥ 2，用于构造 next-token 监督。")
        # —— 统一三模态 embedding 的 dtype（尤其是原位插入时 dynamic_update_slice 必须一致）—— #
        target_dtype = txt_tokens.dtype
        pt_tokens = pt_tokens.astype(target_dtype)
        # ---- A) 原位插入（SpatialLM 一致）或安全回退 ----
        use_inplace = (
            self._spatiallm_exact
            and self._insert_points_in_place
            and (self._point_start_id is not None)
            and (self._point_end_id   is not None)
        )
        if self._insert_points_in_place and not use_inplace and not self._warned_no_point_ids:
            logger.warning(
                "insert_points_in_place=True 但未提供 point_start_id/point_end_id；"
                "将回退到前缀拼接路径。要获得与 SpatialLM 完全一致的插入行为，请在配置中提供这两个特殊 token 的 id。"
            )
            self._warned_no_point_ids = True
        if use_inplace:
            # ===== SpatialLM-exact 插入：保留 <start>/<end>，删除中间文本，<start> 后插入 K 个点 =====
            # 说明：
            #   • 为了避免 JAX 形状多态，本实现将“文本+点段”的 buffer 固定为 L + P（P=固定点块上限），
            #     然而右侧文本的“有效起点”基于真实 K（= sum(valid_mask)），并且 mask/labels 只覆盖有效范围。
            #   • 这样在语义与监督上完全等价于 SpatialLM 的“pad 到本 batch 的 max_num_tokens”，只是我们固定到 L+P。
            B, L, D = txt_tokens.shape
            P = pt_tokens.shape[1]                  # 固定块上限（通常=1024）
            LM = L + P                              # 固定的“文本+点”段 buffer 长度
            img_tokens = img_tokens.astype(target_dtype)
            pt_tokens  = pt_tokens.astype(target_dtype)

            # --- 定位每个样本的 <start>/<end>（不限制二者之间是否有文本；与 SpatialLM 一致） ---
            out_struct = (ShapeDtypeStruct((2,), jnp.int32),)
            win_list = []
            for b in range(B):
                def _host_call(arr):
                    return _host_find_window_only(
                        np.asarray(arr),
                        int(self._point_start_id),
                        int(self._point_end_id),
                    )
                (w_b,) = pure_callback(_host_call, out_struct, obs.tokenized_prompt[b], vectorized=False)
                win_list.append(w_b)
            win_all = jnp.stack(win_list, axis=0)         # [B,2]  -> (s,e)

            # --- 逐样本装配：通过“条件索引 + take”避免动态形状更新 ---
            seq_list, msk_list, ar_list = [], [], []
            for b in range(B):
                s_idx = win_all[b, 0]                     # <start> 位置
                e_idx = win_all[b, 1]                     # <end>   位置
                # 有效点数 K：只统计有效 mask（frame 存在 + Sonata 输出 valid）
                K_b = jnp.sum(pt_final_mask[b].astype(jnp.int32))     # scalar int32
                L_right = L - e_idx                                     # 右段文本长度（含 <end> 及其后的文本）

                # 目标序列（仅文本+点段）的全长索引：0..LM-1，其中：
                #   [0 .. s]         ← 文本左段（含 <start>）
                #   [s+1 .. s+K_b]   ← 点 token（真实 K_b 个）
                #   [s+K_b+1 .. ...] ← 文本右段（从 <end> 开始）
                t = jnp.arange(LM, dtype=jnp.int32)
                pt_start = s_idx + 1
                pt_end   = pt_start + K_b

                left_cond   = (t <  pt_start)
                points_cond = (t >= pt_start) & (t < pt_end)
                right_cond  = (t >= pt_end)   & (t < pt_end + L_right)

                # 文本（左+右）源位置：
                #   左：   idx = t
                #   右：   idx = e_idx + (t - pt_end)
                txt_idx = jnp.where(left_cond, t, e_idx + (t - pt_end))
                txt_idx = jnp.clip(txt_idx, 0, L - 1)
                # 点 源位置：
                pt_idx = jnp.clip(t - pt_start, 0, P - 1)

                # 取出并按条件拼装 embedding
                txt_part = jnp.take(txt_tokens[b], txt_idx, axis=0)
                txt_part = txt_part * (left_cond | right_cond)[:, None].astype(txt_part.dtype)
                pt_part  = jnp.take(pt_tokens[b],  pt_idx,  axis=0)
                pt_part  = pt_part * points_cond[:, None].astype(pt_part.dtype)
                txtpts_scatter = txt_part + pt_part                      # [LM, D]

                # mask / ar：文本沿用源，点一律 ar=0；padding 位置全 False
                txt_mask_src = jnp.take(obs.tokenized_prompt_mask[b].astype(bool), txt_idx, axis=0)
                m_txt = txt_mask_src & (left_cond | right_cond)
                m_pt  = points_cond
                m_txtpt = m_txt | m_pt                                    # [LM]

                txt_ar_src = jnp.take(obs.token_ar_mask[b].astype(jnp.int32), txt_idx, axis=0)
                ar_txt = txt_ar_src * (left_cond | right_cond).astype(jnp.int32)
                ar_pt  = jnp.zeros_like(t, dtype=jnp.int32)
                ar_txtpt = jnp.where(points_cond, ar_pt, ar_txt)          # 点 token ar=0

                # 与图像拼接： [img | (text+points)]
                seq_b = jnp.concatenate([img_tokens[b], txtpts_scatter], axis=0)
                m_b   = jnp.concatenate([img_mask[b],   m_txtpt], axis=0)
                ar_b  = jnp.concatenate([img_ar[b],     ar_txtpt], axis=0).astype(jnp.int32)

                seq_list.append(seq_b)
                msk_list.append(m_b)
                ar_list.append(ar_b)

            tokens = jnp.stack(seq_list, axis=0)   # [B, Nimg+LM, D]  (LM = L + P)
            mask   = jnp.stack(msk_list,  axis=0)  # [B, Nimg+LM]
            ar     = jnp.stack(ar_list,   axis=0)  # [B, Nimg+LM]
            return tokens, mask, ar

        # ---- B) 前缀拼接（默认回退路径） ----
        if len(token_embeddings):
            token_embeddings = [t.astype(target_dtype) for t in token_embeddings]
        token_embeddings.append(img_tokens.astype(target_dtype))
        input_mask.append(img_mask)
        ar_mask.append(img_ar)
        if pt_tokens.shape[1] > 0:
            token_embeddings.append(pt_tokens.astype(target_dtype))
            input_mask.append(pt_final_mask)
            ar_mask.append(jnp.zeros_like(pt_final_mask, dtype=jnp.int32))
        token_embeddings.append(txt_tokens)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)
        tokens = jnp.concatenate(token_embeddings, axis=1)
        mask   = jnp.concatenate(input_mask,       axis=1)
        ar     = jnp.concatenate(ar_mask,          axis=1)
        return tokens, mask, ar

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False
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
        tokens, mask, ar = self.embed_inputs(observation)
        # Compute attention mask for the sequence (prefix + causal masking as needed)
        attn_mask = _pi0_fast.make_attn_mask(mask, ar)
        # === SpatialLM 对齐的损失：原位插入后，文本 token 不再“整段位于末尾” ===
        # 方案：decode 全长度，然后按“文本 token 在插入后的新位置 - 1”去 gather 对应 logits，再做 CE。
        vocab_size = self.PaliGemma.llm.module.vocab_size
        B, L = observation.tokenized_prompt.shape
        # 1) 先得到整段 pre_logits → logits_all
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=tokens[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True
        )
        logits_all, _ = self.PaliGemma.llm(pre_logits=pre_logits)   # [B, T_total-1, V]

        # 静态 gating：原位插入需配置好两个特殊 token 的 id，否则回退尾部对齐损失
        use_inplace = (
            self._spatiallm_exact
            and self._insert_points_in_place
            and (self._point_start_id is not None)
            and (self._point_end_id   is not None)
        )
        if not use_inplace:
            targets = jax.nn.one_hot(observation.tokenized_prompt[:, 1:], vocab_size)
            logits_tail = logits_all[:, -targets.shape[1]:]
            log_probs = jax.nn.log_softmax(logits_tail, axis=-1)
            assert observation.token_loss_mask is not None
            loss_mask = observation.token_loss_mask[:, 1:]
            token_nll = -jnp.sum(targets * log_probs, axis=-1)
            denom = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1)
            seq_loss = jnp.sum(token_nll * loss_mask, axis=-1) / denom
            return seq_loss

        # 2) SpatialLM-exact：直接重建“插入后”的 labels（包含 <start>/<end>），点位置 IGNORE
        #    我们的 tokens 序列是 [img | text+points]，其中 text+points 固定为 L+P，真实有效长度依 K 决定。
        T_total = tokens.shape[1]
        P = int(getattr(self, "_pt_block_len", 1024))
        LM = observation.tokenized_prompt.shape[1] + P           # = L + P
        Nimg = T_total - LM
        IGNORE = jnp.int32(-100)

        # --- (a) <start>/<end> 位置 ---
        out_struct = (ShapeDtypeStruct((2,), jnp.int32),)
        wins = []
        for b in range(B):
            def _host_call(arr):
                return _host_find_window_only(
                    np.asarray(arr),
                    int(self._point_start_id),
                    int(self._point_end_id),
                )
            (w_b,) = pure_callback(_host_call, out_struct, observation.tokenized_prompt[b], vectorized=False)
            wins.append(w_b)
        win_all = jnp.stack(wins, axis=0)     # [B,2] -> (s,e)
        s_all = win_all[:, 0]
        e_all = win_all[:, 1]

        # --- (b) 逐样本求有效点数 K —— 与 embed_inputs 完全一致：再次调用 Sonata wrapper 只取 valid_mask
        #     注意：不能用 mask 在 [s+1 : s+1+P) 的和来近似 K，因为该窗口会包含从 <end> 起的右侧文本；
        #     这会把 K 算大，导致 label 错位。必须以 Sonata 的 valid_mask 为准。
        k_list = []
        pc_pack = _extract_point_batch(observation)
        if pc_pack is None:
            k_list = [jnp.array(0, jnp.int32) for _ in range(B)]
        else:
            pc_dict_all, pc_frame_mask = pc_pack
            for b in range(B):
                present_b = pc_frame_mask[b].any() if pc_frame_mask.ndim == 2 else pc_frame_mask[b]
                single_dict = {
                    **pc_dict_all,
                    "selected_batch": jnp.array(b, jnp.int32),
                    "present": present_b.astype(jnp.int32),
                }
                _, vmask = self.PaliGemma.point(single_dict, train=False)  # [P]
                K_b = jnp.sum(vmask.astype(jnp.int32))
                k_list.append(K_b)
        K_all = jnp.stack(k_list, axis=0)  # [B]

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
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = _pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)
        # Align all prefix sequences to the right (required for caching in prefix)
        prefix_tokens, prefix_mask, prefix_attn_mask = _pi0_fast.left_to_right_align(prefix_tokens, prefix_mask, prefix_attn_mask)
        # ①  ——  与 SpatialLM 一致：prefill_len 明确 cast 为 int32（后续与 lax.iota 等保持类型匹配）
        prefill_size = prefix_tokens.shape[1]   # total sequence length after alignment (prefix padded to some length)
        prefill_len = jnp.sum(prefix_mask, axis=-1).astype(jnp.int32)   # actual prefix length per batch (number of valid tokens)
        prefix_start = prefill_size - prefill_len   # start index of actual prefix tokens after right-align (for each batch)
        # Prepare attention mask for prefix + decoding steps
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = (jnp.cumsum(prefix_mask.astype(jnp.int32), axis=-1) - 1).astype(jnp.int32)  # positions of prefix tokens (0-indexed)
        # Run the LLM in decoding mode to fill KV cache with prefix
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_tokens,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True
        )
        # Start from the last logit of the prefix as the beginning for new generation
        last_logit = prefix_logits[:, -1:]   # [B, 1, V]
        # Placeholder for generated token outputs (initialize with zeros)
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        # Now run autoregressive decoding for at most max_decoding_steps
        # Prepare attention mask for single-step decoding (prefix length + current step)
        # We will update the attention mask dynamically in the loop if needed

        # Use a while loop to generate tokens until done or max steps
        def cond_fn(state):
            _rng, _last, _cache, _step, _out = state
            # while_loop 的条件必须是**标量 bool**。
            # 这里我们认为“所有样本都已生成 EOS”才提前停止，因此用 jnp.all(...) → 标量。
            def _false_scalar(_):
                return jnp.array(False, dtype=bool)
            def _all_eos_scalar(_):
                # _out[:, _step - 1] 形状为 [B]，jnp.all(...) 返回标量（所有样本均为 EOS）
                # 注意：step==0 时还没有有效 token，因此走 _false_scalar 分支。
                return jnp.all(_out[:, _step - 1] == _pi0_fast.PALIGEMMA_EOS_TOKEN)
            has_eos = jax.lax.cond(
                _step == 0,
                _false_scalar,
                _all_eos_scalar,
                operand=None,
            )
            return jnp.logical_and(_step < max_decoding_steps, jnp.logical_not(has_eos))

        def body_fn(state):
            rng_key, last_logits, cache, step, out_tokens = state
            rng_key, subkey = jax.random.split(rng_key)
            logits_step = last_logits.squeeze(1)
            token = jax.lax.cond(
                temperature > 1e-6,
                lambda key: jax.random.categorical(key, logits_step / temperature, axis=-1),
                lambda key: jnp.argmax(logits_step, axis=-1),
                operand=subkey,
            ).astype(jnp.int32)
            # 原 put_along_last_axis 构造 O(N²) one‑hot；改用 scatter 更新
            out_tokens = out_tokens.at[:, step].set(token)

            # Gemma‑fast & SpatialLM：位置 = 已填 prefix token 数 + 当前 step
            # ②  ——  positions 统一 int32，可避免 multi‑host sharding “mixed signedness” 报警
            # 与 SpatialLM/pi0_fast 对齐（0‑based）：首个新 token 位置 = prefill_len + 0
            positions = (prefill_len[:, None] + step).astype(jnp.int32)
            # Gemma‑fast 无 token=kwarg：先嵌入，再 decode 一步
            token_emb = self.PaliGemma.llm(token[:, None], embed_only=True)
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
            logits, cache = self.PaliGemma.llm(
                embedded_prefix=token_emb,
                kv_cache=cache,
                positions=positions,
                decode=True,
                mask=mask,  # ★ 新增
            )
            return rng_key, logits, cache, step+1, out_tokens

        init_state = (rng, last_logit, kv_cache, jnp.array(0, jnp.int32), output_tokens)
        _, _, _, final_step, output_seq = jax.lax.while_loop(cond_fn, body_fn, init_state)

        # Return the output sequence of tokens as the model's predicted "actions"
        # (In practice, these tokens might represent discretized actions or a planned sequence encoded as text tokens)
        return output_seq[:, :final_step]   # [B, <=max_dec_steps]
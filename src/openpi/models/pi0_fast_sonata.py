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
# 1024 token 块上限先与 ckpt 保持一致；显存允许时再调大 enc_patch_size[-1]。
# JAX 端 batch/offset 用 int32，host (PyTorch) 端统一 .long()，避免 X64 警告。

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
    # 严格 dtype：必须是 bool
    if pc_mask.dtype != jnp.bool_:
        raise ValueError(f"point_cloud_masks['pointcloud'] 必须是 bool dtype，实际为 {pc_mask.dtype}.")
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

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Inherits default action_dim, action_horizon, max_token_len from Pi0FASTConfig (e.g., 32, 32, 250)
    use_pretrained_point: bool = True      # 调试阶段可设 False 跳过下
    require_pointcloud: bool = True
    # 若无图像则直接不训练（训练时报错终止；推理不受影响，这里主要用于进行调试确保数据集正确）
    require_image: bool = True
    point_start_id: Optional[int] = None
    point_end_id:   Optional[int] = None

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
                #  - point_clouds["pointcloud"]: [B, M, 3 + 6] (float32)
                #      [:,:,0:3] = grid_coord（运行时 cast→int32）
                #      [:,:,3:6] = coord xyz（float32）；[:,:6] 后必须是 rgb 等额外 3 维
                #  - point_cloud_masks["pointcloud"]: [B] bool  —— 指示该样本是否存在点云
                point_clouds={"pointcloud": jax.ShapeDtypeStruct([batch_size, self.max_points, 3 + 6], jnp.float32)},
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

        # 按 gemma_fast.Module 的 __init__ 签名自动收集需要的参数
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
        
        # ---------- Sonata hyper‑params – 与 ckpt 完全一致 ----------
        # 强制 6 通道（xyz + rgb）
        if int(config.point_feat_dim) != 6:
            raise ValueError("本实现严格复刻 SpatialLM：point_feat_dim 必须为 6（xyz+rgb）。")
        _in_channels = 6

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
        self._point_in_channels = _in_channels
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

            # in_channels 严格一致性检查（禁止“动态重建”）
            stem_w = cleaned.get("embedding.stem.linear.weight", None)
            if stem_w is None:
                raise KeyError("Sonata 权重缺少 'embedding.stem.linear.weight'。")
            in_from_ckpt = int(stem_w.shape[1])
            if in_from_ckpt != self._point_in_channels:
                raise ValueError(f"Sonata ckpt in_channels={in_from_ckpt} 与期望 {self._point_in_channels} 不一致；"
                                 "请使用与 SpatialLM 完全一致的 6 通道权重。")

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
                if "selected_batch" in host_dict:  # 按样本切片
                    sb = int(host_dict.pop("selected_batch"))
                    if "batch" in host_dict:
                        b_arr = np.asarray(host_dict["batch"]).reshape(-1)
                        sel = (b_arr == sb)                     # sel.shape == (B*M,)
                        for k, v in list(host_dict.items()):
                            if hasattr(v, "shape") and v.ndim and v.shape[0] == sel.shape[0]:
                                host_dict[k] = v[sel]
                        # 还需要对“按帧打包”的张量做第 0 维切片（例如 grid_coord[B,M,3]、grid_size[B,3]）
                        # 这里的 B 优先用 offset 推断；否则用 batch 的最大值+1
                        if "offset" in host_dict:
                            B_est = int(np.asarray(host_dict["offset"]).reshape(-1).shape[0])
                        else:
                            B_est = int(b_arr.max()) + 1 if b_arr.size else 0
                        if B_est:
                            for k, v in list(host_dict.items()):
                                if hasattr(v, "shape") and v.ndim >= 2 and v.shape[0] == B_est:
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
                        "请增大 enc_patch_size[-1] 或在上游减少每样本点数。"
                    )
                # SpatialLM: 不截断；右 pad 到 patch_size 的倍数
                # --- 保留固定 1024‑padding 以维持静态 shape (JAX 需求，如果超过1024则容易炸显存) ---
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

        # ---------- 4) 原生 NNX 线性投影：支持端到端反向传播、避免 host 往返 ----------
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
        # special ids（原位插入必需）
        self._point_start_id = getattr(config, "point_start_id", None)
        self._point_end_id   = getattr(config, "point_end_id", None)
        if (self._point_start_id is None) or (self._point_end_id is None):
            raise ValueError("必须提供 point_start_id / point_end_id 才能进行 SpatialLM‑exact 原位插入。")

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

        # ---------- 2) Point cloud tokens（严格 SpatialLM 路径） ----------
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
        for b in range(B):  # 逐样本调用 Sonata
            present_b = pc_frame_mask[b]
            single_dict = {
                **pc_dict_all,
                "selected_batch": jnp.array(b, jnp.int32),
                "present": jnp.asarray(present_b, jnp.int32),
            }
            if "grid_coord" in single_dict and single_dict["grid_coord"].shape[-1] != 3:
                raise ValueError(f"pointcloud grid_coord 的最后一维必须为 3，实际 shape={single_dict['grid_coord'].shape}")
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
            per_sample_tokens.append(tok)
            per_sample_masks.append(vmask)
            max_len = max(max_len, tok.shape[0])

        # ----- pad 到 batch 内最大长度（一次性做）-----------
        def _pad_to(x, tgt):
            pad = [(0, tgt - x.shape[0])] + [(0, 0)]*(x.ndim-1)
            return jnp.pad(x, pad)
        pt_tokens = jnp.stack([_pad_to(t, max_len) for t in per_sample_tokens], axis=0)  # (B,P,C)
        valid_m   = jnp.stack([_pad_to(m, max_len) for m in per_sample_masks], axis=0)   # (B,P)
        assert (pt_tokens.shape[1] == valid_m.shape[1] == max_len), "pad 后 token / mask 长度不一致"

        # --- 帧级掩码广播到 token 维 ---
        pc_frame_mask_b = jnp.broadcast_to(pc_frame_mask[:, None], (B, max_len))
        pt_final_mask   = pc_frame_mask_b & valid_m
        pt_tokens       = pt_tokens * pt_final_mask[:, :, None]

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

        # ===== 4) SpatialLM‑exact 原位插入（无降级路径） =====
        B, L, D = txt_tokens.shape
        P = pt_tokens.shape[1]               # 固定块上限（通常=1024）
        LM = L + P                           # “文本+点”段固定 buffer 长度
        img_tokens = img_tokens.astype(target_dtype)
        pt_tokens  = pt_tokens.astype(target_dtype)

        # --- 定位每个样本的 <start>/<end>（不限制二者之间是否有文本；与 SpatialLM 一致） ---
        out_struct = (ShapeDtypeStruct((2,), jnp.int32),)
        win_list = []
        for b in range(B):
            def _host_call(arr):
                return _host_find_window_only(np.asarray(arr), int(self._point_start_id), int(self._point_end_id))
            (w_b,) = pure_callback(_host_call, out_struct, obs.tokenized_prompt[b], vectorized=False)
            win_list.append(w_b)
        win_all = jnp.stack(win_list, axis=0)         # [B,2]  -> (s,e)

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
        # === SpatialLM‑exact：原位插入后文本位置改变，按新位置对齐标签 ===
        vocab_size = self.PaliGemma.llm.module.vocab_size
        B, L = observation.tokenized_prompt.shape
        # 1) 先得到整段 pre_logits → logits_all
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=tokens[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True
        )
        logits_all, _ = self.PaliGemma.llm(pre_logits=pre_logits)   # [B, T_total-1, V]

        # 2) 直接重建“插入后”的 labels（包含 <start>/<end>），点位置 IGNORE
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

        # --- (b) 逐样本求有效点数 K —— 直接在“点占位区”内对 mask 求和（与 embed_inputs 完全一致）
        # 点占位区在拼接后的 text+point 段（长度 LM）的相对区间为 [s_idx+1, s_idx+1+P)
        t_rel = jnp.arange(LM, dtype=jnp.int32)[None, :]                    # [1, LM]
        pt_zone = (t_rel >= (s_all[:, None] + 1)) & (t_rel < (s_all[:, None] + 1 + P))  # [B, LM]
        # mask[:, Nimg:Nimg+LM] 是拼接后 text+point 段的有效性；其与 pt_zone 的按位与即为“点 token”有效位置
        K_all = jnp.sum(mask[:, Nimg:Nimg+LM] & pt_zone, axis=1).astype(jnp.int32)

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
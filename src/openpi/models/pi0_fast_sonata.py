# 该版本已知问题（这些问题目前暂时不用立刻解决）：
# 训练时梯度	不会尝试回传到 Sonata/Projector	freeze‑filter 非刚需, 这个我们后续再解决，因为我们实际上要允许训练，所以反而不能冻结，这个后面再说
# 性能潜在瓶颈	CPU ↔ GPU copy / 多编译	后续迭代可能会影响这个，所以首先解决问题1
# 1024 token size，这个暂时不能设置太大因为会炸显存，首先使用提示方式来确定是否会有过多的情况，没有就继续训练，有的话后续再继续处理
# 注意，grid似乎不能是负数！

import dataclasses
import inspect
import logging
import typing
import jax
import jax.numpy as jnp
import einops
import numpy as np
import torch
from pathlib import Path

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
import openpi.models.model as _model  # for BaseModel and Observation
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
    pd = {k: jnp.asarray(v) for k, v in pd.items()}  # ← 用 jnp

    if pd["coord"].ndim == 3:      # (B,N,3) → (B*N,3)
        B, N, _ = pd["coord"].shape
        pd["coord"] = jnp.reshape(pd["coord"], (B * N, 3))
        pd["feat"]  = jnp.reshape(pd["feat"],  (B * N, -1))
        # batch / offset 必须为 int64，Sonata 内部要做 (batch << 48)
        pd["batch"]  = jnp.repeat(jnp.arange(B, dtype=jnp.int64), N)
        pd["offset"] = jnp.cumsum(jnp.full((B,), N, dtype=jnp.int64))

    if "grid_size" in pd and pd["grid_size"].ndim == 3:
        pd["grid_size"] = jnp.reshape(pd["grid_size"], (-1, 3))

    # 统一强制 int64 以免后续再 cast
    for key in ("batch", "offset"):
         if key in pd:
            pd[key] = pd[key].astype(jnp.int64, copy=False)

    # SpatialLM 全程使用 int32 体素坐标；强制转换可消除潜在溢出
    if "grid_coord" in pd:
        # 这里有个潜在的问题，若用户自己组装的grid_coord已经超过65535，则可能再次溢出，但这个实际上不太可能，因此只写注释，等以后如果训练有问题再回来看。
        pd["grid_coord"] = pd["grid_coord"].astype(jnp.int32, copy=False)

    return pd

# -------- 在旧 / 新两类 Observation 之间统一抽取点云 ----------
def _extract_point_batch(obs) -> tuple[dict[str, jnp.ndarray], jnp.ndarray] | None:
    """
    返回 (pc_dict, batch_mask) 或 None

    支持两种来源：
    1. 旧版 :  obs.pointcloud_data
       - a) 已是 Sonata 兼容 dict      → 直接使用
       - b) [B, P, 6] ndarray          → 自动展开成 dict
    2. 新版 :  obs.point_clouds["pointcloud"]  +  obs.point_cloud_masks["pointcloud"]
    """
    # ---------- 💡 新接口 ----------
    if hasattr(obs, "point_clouds") and "pointcloud" in getattr(obs, "point_clouds"):
        pc_arr  = obs.point_clouds["pointcloud"]          # [B, M, 6]
        pc_mask = obs.point_cloud_masks["pointcloud"]     # [B]

        B, M, _ = pc_arr.shape
        # SpatialLM‑Qwen 约定:
        #   0‑2 : 体素网格坐标 (int32)
        #   3‑5 : 连续 xyz (float32)
        #   6+ : 其他语义特征
        grid_int_raw = pc_arr[..., :3].astype(jnp.int32)       # (B,M,3)
        # --- 关键修复：逐 batch 归零，防 Morton 位宽溢出 -----------------
        grid_base   = grid_int_raw.min(axis=1, keepdims=True)  # (B,1,3)
        grid_int    = grid_int_raw - grid_base                 # 保证每维 ≥0
        coords   = pc_arr[..., 3:6].astype(jnp.float32)        # 连续 xyz
        feats    = pc_arr[..., 3:].astype(jnp.float32)         # xyz + 语义

        # 不在 JAX 侧做可变长展平 / NaN 过滤；保持 (B,M,*) 静态形状，
        # 把展平、去 NaN、重新计算 offset 完全交由
        # _TorchSonataWrapper (host‑side) 处理，
        # 以避免 jnp.repeat(counts) 的编译期动态大小。
        pc_dict = _canonicalize_point_dict(
            dict(coord=coords, grid_coord=grid_int, feat=feats)
        )

        # 直接使用静态维度 M 作为 pad 长度，避免 run‑time int()
        max_len  = pc_arr.shape[1]                                 # == M
        mask     = einops.repeat(pc_mask, "b -> b l", l=max_len)
        return pc_dict, mask

    # ---------- 旧接口 ----------
    legacy = getattr(obs, "pointcloud_data", None)
    if legacy is None:
        return None

    # a) 已经是 dict
    if isinstance(legacy, dict) and "coord" in legacy:
        pc_dict = _canonicalize_point_dict(legacy)
    else:
        # b) assume legacy is a [B, P, 6] array  (xyz + feat)
        arr = jnp.asarray(legacy, dtype=jnp.float32)   # (B,P,6)
        B, P, C = arr.shape

        coord = arr[..., :3].reshape(-1, 3)            # (B*P,3)
        feat  = arr[..., 3:].reshape(-1, C - 3)        # (B*P,3)
        batch  = jnp.repeat(jnp.arange(B, dtype=jnp.int64), P)
        offset = jnp.cumsum(jnp.full((B,), P, dtype=jnp.int64))

        # ---------- grid_coord: 先转 int32，再逐维减最小值 ----------
        grid_raw   = coord.astype(jnp.int32)
        grid_coord = grid_raw - grid_raw.min(axis=0, keepdims=True)

        # 打包到 dict → canonicalize
        pc_dict = _canonicalize_point_dict(
            dict(
                coord      = coord,        # 连续 xyz
                grid_coord = grid_coord,   # 归零后的 int32
                feat       = feat,
                batch      = batch,
                offset     = offset,
            )
        )

    # legacy 路径假设每帧固定 P 个点
    B = pc_dict["offset"].shape[0]                 # 静态 batch_size
    P = pc_dict["coord"].shape[0] // B            # 每帧点数（静态）
    mask = jnp.ones((B, P), dtype=bool)
    return pc_dict, mask

# Alias the Sonata class from the sonata_encoder module for convenience
Sonata = sonata_encoder.Sonata

@dataclasses.dataclass(frozen=True)
class Pi0FASTSonataConfig(_pi0_fast.Pi0FASTConfig):
    """Configuration for the Pi0FASTSonata model (Pi0FAST with Sonata point cloud encoder)."""

    # 关键字段：每个点的总特征维度 = 3(xyz) + N(extra feat)
    # 对当前 DummyPointDataset：[coords/5,  rgb] ⇒ N=6 ⇒ 9
    point_feat_dim: int = 9

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Inherits default action_dim, action_horizon, max_token_len from Pi0FASTConfig (e.g., 32, 32, 250)
    use_pretrained_point: bool = True      # 调试阶段可设 False 跳过下载

    @property
    def model_type(self) -> _model.ModelType:
        # Reuse the PI0_FAST model type (since this is an extension of Pi0-FAST architecture)
        return _model.ModelType.PI0_FAST

    def create(self, rng: jax.Array) -> "Pi0FASTSonata":
        """Instantiate a Pi0FASTSonata model with random initialization."""
        return Pi0FASTSonata(self, rngs=nnx.Rngs(rng))

class Pi0FASTSonata(_model.BaseModel):
    """
    Pi0FASTSonata model: Extends Pi0-FAST to incorporate a Sonata point cloud encoder.
    Combines vision (SigLIP image encoder), language (PaLI-Gemma LLM), and point cloud (Sonata) encoders.
    """
    def __init__(self, config: Pi0FASTSonataConfig, rngs: nnx.Rngs):
        # Initialize base model (setup action_dim, action_horizon, etc.)
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # --------------------------------------------------------------
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
        # in_channels 按配置 point_feat_dim – 3 动态确定；若配置缺失立刻报错
        if not hasattr(config, "point_feat_dim"):
            raise ValueError(
                "Pi0FASTSonataConfig 需显式提供 point_feat_dim，用于推导 "
                "Sonata in_channels = point_feat_dim - 3"
            )
        # ---------------- 配置 → in_channels，并保留交叉校验 ---------------
        _in_channels = int(config.point_feat_dim) - 3
        if _in_channels <= 0:
            raise ValueError(
                f"配置 point_feat_dim={config.point_feat_dim} 无效，须 ≥ 4"
            )
        # 若后续输入点云特征维与配置不符，将在 embed_inputs 早期报错

        sp_cfg = dict(
            in_channels   = _in_channels,
            order         = ("z", "z-trans"),
            stride        = (2, 2, 2, 2),         # 5‑stage ⇒ 4 次下采样
            enc_depths    = (3, 3, 3, 12, 3),
            enc_channels  = (48, 96, 192, 384, 512),   # ★ 末端 512
            enc_num_head  = (3, 6, 12, 24, 32),
            enc_patch_size= (1024,)*5,            # ckpt 默认
            mlp_ratio     = 4.0,
            mask_token    = True,
            enc_mode      = "voxel",
            enable_fourier_encode = True,         # ★ ckpt 含 fourier+input_proj
            num_bins      = 1280,
        )
        point_model = Sonata(**sp_cfg)
        # 记录供后续断言／Projector 使用
        self._point_in_channels = _in_channels
        self._enc_out_dim       = sp_cfg["enc_channels"][-1]   # e.g. 512

        if config.use_pretrained_point:
            # ------------------------------------------------------------------
            # 仅使用本地精简后的 SpatialLM1.1 Sonata 权重
            # 文件放置: <repo_root>/openpi/pretrained/SpatialLM_Sonata_encoder.pth
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
                    "并放置到 openpi/pretrained/ 目录。"
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
                if k.startswith(("module.", "model.", "student.backbone.", "student.")):
                    cleaned[k.split(".", 1)[1]] = v
                else:
                    cleaned[k] = v

            # ③部分匹配即可；strict=False 会跳过多余键，也会提示哪些没加载
            missing, unexpected = point_model.load_state_dict(
                cleaned, strict=False
            )
            if missing:
                logger.warning("Sonata weights: %d missing params, e.g. %s",
                               len(missing), missing[:5])
            if unexpected:
                logger.warning("Sonata weights: %d unexpected params, e.g. %s",
                               len(unexpected), unexpected[:5])
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
                host_dict: dict[str, np.ndarray],
                device: torch.device,
                patch_size: int,
            ) -> tuple[np.ndarray, np.ndarray]:  # 第 2 个 ndarray 是 bool 掩码
                """
                接收 host 上的 numpy 输入 → torch.Tensor.cuda → 运行 → numpy 输出
                统一返回 float32 numpy 数组
                """

                # =========================================================
                # ★ 新增：若上游传入 selected_batch，只保留这一帧的点
                #   (这样 embed_inputs 里不用 v[sel]，避免 JAX 布尔切片报错)
                # ---------------------------------------------------------
                if "selected_batch" in host_dict:
                    sb = int(host_dict.pop("selected_batch"))
                    sel = host_dict["batch"] == sb
                    for k, v in list(host_dict.items()):
                        # 仅对第 0 维与点数对应的字段做筛选
                        if v.ndim and v.shape[0] == sel.shape[0]:
                            host_dict[k] = v[sel]
                    # 单样本语义：batch 全 0，offset = [点数]
                    n_pts = host_dict["coord"].shape[0]
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

                # grid_coord / grid_size 可能来自体素网格 → 保证都是 (P,3)
                for _key in ("grid_coord", "grid_size"):
                    if _key in host_dict and host_dict[_key].ndim == 3:
                        host_dict[_key] = host_dict[_key].reshape(-1, 3)

                # 无论上游如何，feat 必须为 2‑D；与 SpatialLM 对齐
                if host_dict["feat"].ndim != 2:
                    C = host_dict["feat"].shape[-1]
                    host_dict["feat"] = host_dict["feat"].reshape(-1, C)

                
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
                    # ① 计算输出维度 C_out（与 Sonata encoder 末层一致）
                    C_out = (
                        inner.input_proj.out_features
                        if getattr(inner, "enable_fourier_encode", False)
                        else inner.enc_channels[-1]
                    )
                    # ② 返回全零特征 + 全 False 有效标记
                    pad_feat   = np.zeros((patch_size, C_out), dtype=np.float32)
                    valid_mask = np.zeros((patch_size,),       dtype=bool)
                    return pad_feat, valid_mask

                # ---------- 若缺 grid_coord，则用 floor(coord) 并归零 ----------
                if "grid_coord" not in host_dict:
                    gc = np.floor(host_dict["coord"]).astype(np.int32, copy=False)
                    gc -= gc.min(axis=0, keepdims=True)        # 保证每维从 0 开始
                    host_dict["grid_coord"] = gc

                # pure_callback 把 jax.Array 直接送过来；必须先转成真正的 numpy
                tch_in = {
                    k: torch.from_numpy(np.asarray(v))  # ← 关键：np.asarray()
                    .to(device)
                    for k, v in host_dict.items()
                }
                # Sonata 里要做 (batch << 48)，必须是 int64
                for key in ("batch", "offset"):
                    if key in tch_in:
                        tch_in[key] = tch_in[key].long()
                out = inner(tch_in)
                if isinstance(out, dict):                # 取 "feat"
                    out = out.get("feat", list(out.values())[0])

                real_len = out.size(0)
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

                # ---------- 重新构造 dummy_np（避免 Tracer→NumPy） ----------
                dummy_np = {}
                for k, v in host_inputs.items():
                    shape, dtype = v.shape, v.dtype

                    if k == "grid_size":
                        # 用 128 填充，dtype 必须与原始一致 (int32)
                        dummy_np[k] = np.full(shape, 128, dtype=dtype)
                    elif k == "coord":
                        # —— 若原 coord 是 (B,N,3) → total = B*N ——
                        if v.ndim == 3:
                            B, N, _ = v.shape
                            total = B * N
                        else:
                            total = v.shape[0]
                        coords = np.arange(total * 3, dtype=dtype).reshape(total, 3)
                        dummy_np[k] = coords
                    elif k == "offset":
                        # 根据 **展平后的点数** 生成正确 offset
                        total = (
                            host_inputs["coord"].reshape(-1, 3).shape[0]
                            if host_inputs["coord"].ndim == 3
                            else host_inputs["coord"].shape[0]
                        )
                        dummy_np[k] = np.array([total], dtype=dtype)
                    elif k == "feat":
                        # 保证 dummy 特征也是 2‑D，与真实前向形状一致
                        if v.ndim == 3:
                            B, N, C = v.shape
                            dummy_np[k] = np.zeros((B * N, C), dtype=dtype)
                        else:
                            dummy_np[k] = np.zeros(shape, dtype=dtype)
                    elif k in ("grid_coord", "grid_size"):
                        # 始终展平成 (P,3)
                        if v.ndim == 3:
                            dummy_np[k] = np.zeros(
                                (v.shape[0] * v.shape[1], 3), dtype=dtype
                            )
                        else:
                            dummy_np[k] = np.zeros(shape, dtype=dtype)
                    else:
                        dummy_np[k] = np.zeros(shape, dtype=dtype)
                # -----------------------------------------------------------

                dummy_feat, _ = self._torch_forward(
                    self.inner, dummy_np, self.device, self.patch_size
                )
                # 例如静态上限, 这里设置1024，可以设置8192 (= 8 patch)；确保不会在真实输入中超出但是这样会炸显存,所以我们保留1024实现, 后续可以继续加
                # patch_size 由 Wrapper 构造函数传入；保持与 Sonata enc_patch_size 一致
                MAX_TOKEN = self.patch_size
                C = dummy_feat.shape[1]
                out_struct = (ShapeDtypeStruct((MAX_TOKEN, C), jnp.float32),
                              ShapeDtypeStruct((MAX_TOKEN,),  jnp.bool_))

                def _host_call(*flat_np):
                    # flat_np 是回传的扁平列表/元组，需用 treedef.unflatten 还原
                    np_dict = treedef.unflatten(list(flat_np))
                    return self._torch_forward(
                        self.inner, np_dict, self.device, self.patch_size
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


        # ---------- 5) wrap 为 NNX 模块并 lazy_init ----------
        # ⑤ 线性投影：enc_out_dim → PaLI‑Gemma hidden_size
        self.PointProjector = nnx_bridge.ToNNX(
            _TorchLinearWrapper(self._enc_out_dim, _model_width, self.device)
        )
        # small dummy → lazy_init
        self.PointProjector.lazy_init(
            jnp.zeros((1, self._enc_out_dim), jnp.float32),
            rngs=rngs,
        )
        # 让投影层跟随 PyTorch 的 device，不必手动迁移除非后续继续重写到jax

        N, B = 64, 1          # 64 points, 1 batch
        # ────────────────────────────────────────────────────────────────
        # Sonata 约定的 Point 结构（务必扁平化）：
        #   coord : (Total_N, 3)   float32 / int32
        #   feat  : (Total_N, C)
        #   batch : (Total_N,)     **int64**  ← 要能做 «batch << 48»
        #   offset: (B,)           累计点数前缀和  **int64**
        #   grid_size : (B, 3)
        # ────────────────────────────────────────────────────────────────
        raw_dummy_pc = {
            "coord": jnp.arange(N * 3, dtype=jnp.float32).reshape(N, 3),
            "feat":  jnp.zeros((N, self._point_in_channels), dtype=jnp.float32),
            "batch": jnp.zeros((N,),  dtype=jnp.int64),      # 1‑D  (Sonata 需 int64)
            "offset": jnp.array([N], dtype=jnp.int64),
            "grid_size": jnp.array([[32, 32, 32]], dtype=jnp.int32),   # <= num_bins / 2**4
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
        
        # 初始化 wrapper（一次性 shape 推断）
        point.lazy_init(dummy_pc, train=False, rngs=rngs)

        # ------------------------------------------------------------------
        # 6) 打包所有子模块
        # ------------------------------------------------------------------
        self.PaliGemma = nnx.Dict(
            llm   = llm,
            img   = img,
            point = point,
            point_proj = self.PointProjector,
        )

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
        token_embeddings = []
        input_mask = []
        ar_mask = []

        # 1. Image tokens (from image encoder)
        for cam_name, image in obs.images.items():
            img_tokens, _ = self.PaliGemma.img(image, train=False)  # shape [B, n_img_tokens, emb_dim]
            token_embeddings.append(img_tokens)
            # For each image token, the mask is valid if the image is present (broadcast image mask per token)
            mask = jnp.broadcast_to(obs.image_masks[cam_name][:, None], (obs.image_masks[cam_name].shape[0], img_tokens.shape[1]))
            input_mask.append(mask)
            # Image tokens have no autoregressive dependency amongst themselves (set AR mask = 0 for these tokens)
            ar_mask.append(jnp.zeros_like(mask, dtype=jnp.int32))

        # 2. Point cloud tokens --------------------------------------------------
        pc_pack = _extract_point_batch(obs)
        if pc_pack is not None:
            # --------------------------------------------------------
            # 与 SpatialLM‑Qwen forward_point_cloud 完全一致的策略：
            # for‑loop 逐样本调用 Sonata → 每次得到 (K*1024, C)，
            # 再统一 pad 到同一长度。
            # --------------------------------------------------------
            pc_dict_all, pc_frame_mask = pc_pack        # pc_frame_mask : [B, M]
            B = pc_dict_all["offset"].shape[0]

            # 运行时维度检查
            if pc_dict_all["feat"].shape[-1] != self._point_in_channels:
                raise ValueError(
                    f"Sonata in_channels={self._point_in_channels}, "
                    f"但输入特征维度={pc_dict_all['feat'].shape[-1]}"
                )

            per_sample_tokens  = []
            per_sample_masks   = []
            max_len            = 0

            for b in range(B):                                    # **逐 batch**
                # 仅把 sample id 传给 wrapper；真正的切片在 PyTorch 侧完成，
                # 因此这里不再产生动态形状。
                single_dict = {
                    **pc_dict_all,          # 全量点云
                    "selected_batch": jnp.array(b, jnp.int32),  # 新增键
                }
                # grid_coord 缺失则直接截断取 int —— 与 SpatialLM 相同
                if "grid_coord" not in single_dict:
                    single_dict["grid_coord"] = single_dict["coord"].astype(jnp.int32)

                tok, vmask = self.PaliGemma.point(single_dict, train=False)

                # -------- 长度一致性断言（问题 4）--------
                assert tok.shape[0] == vmask.shape[0], (
                    f"Sonata 返回 token 长度 {tok.shape[0]} "
                    f"≠ valid_mask 长度 {vmask.shape[0]}"
                )

                # 投影到 LLM hidden_size ，保持与图像 / 文本维度一致
                tok = self.PointProjector(tok.astype(jnp.float32))
                tgt_dtype = token_embeddings[0].dtype if token_embeddings else tok.dtype
                tok = tok.astype(tgt_dtype)

                # SpatialLM 保留补零 token，靠 vmask 指示有效性
                per_sample_tokens.append(tok)       # ← 直接整块保存
                per_sample_masks.append(vmask)
                max_len = max(max_len, tok.shape[0])   # 一般就是 1024

            # ----- pad 到 batch 内最大长度 -----------
            def _pad_to(x, tgt):
                pad = [(0, tgt - x.shape[0])] + [(0, 0)]*(x.ndim-1)
                return jnp.pad(x, pad)

            pt_tokens = jnp.stack([_pad_to(t, max_len) for t in per_sample_tokens])  # (B,max_len,C)
            valid_m   = jnp.stack([_pad_to(m, max_len) for m in per_sample_masks])

            # 二次验证：所有 batch 长度已经对齐
            assert (pt_tokens.shape[1] == valid_m.shape[1] == max_len), \
                "pad 后 token / mask 长度不一致"

            # --- mask 对齐到外层 pc_frame_mask ---  (与 SpatialLM 思路相同)
            if max_len > pc_frame_mask.shape[1]:
                pad_len = max_len - pc_frame_mask.shape[1]
                pc_frame_mask = jnp.pad(pc_frame_mask,
                                         ((0,0),(0,pad_len)),
                                         constant_values=False)
            else:                       # 罕见：1024 < M，裁剪外层掩码
                pc_frame_mask = pc_frame_mask[:, :max_len]
            pt_final_mask = pc_frame_mask & valid_m

            token_embeddings.append(pt_tokens)
            input_mask.append(pt_final_mask)
            ar_mask.append(jnp.zeros_like(pt_final_mask, dtype=jnp.int32))

        # 3. Text tokens (from language model embedding)
        # Ensure textual inputs are present
        assert obs.tokenized_prompt is not None and obs.tokenized_prompt_mask is not None and obs.token_ar_mask is not None, \
            "Tokenized prompt and corresponding masks must be provided for text inputs."
        txt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)  # shape [B, L, emb_dim]
        token_embeddings.append(txt_tokens)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # Concatenate all modality token sequences along the sequence dimension
        tokens = jnp.concatenate(token_embeddings, axis=1)      # [B, N_total, emb_dim]
        mask = jnp.concatenate(input_mask, axis=1)              # [B, N_total]
        ar = jnp.concatenate(ar_mask, axis=1)                   # [B, N_total]
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
        # Embed all inputs to tokens and get masks
        tokens, mask, ar = self.embed_inputs(observation)
        # Compute attention mask for the sequence (prefix + causal masking as needed)
        attn_mask = _pi0_fast.make_attn_mask(mask, ar)
        # Prepare next-token prediction targets: one-hot encoding of tokenized_prompt shifted by one
        vocab_size = self.PaliGemma.llm.module.vocab_size
        targets = jax.nn.one_hot(observation.tokenized_prompt[:, 1:], vocab_size)  # shape [B, L-1, vocab_size]
        # Run LLM for prefix (all tokens except the last one, which will be predicted)
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=tokens[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True
        )
        # Decode only the new suffix (the part to predict) to get logits for target tokens
        logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1]:])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Compute cross-entropy loss on the target token predictions
        assert observation.token_loss_mask is not None, "Token loss mask must be provided for computing loss."
        loss_mask = observation.token_loss_mask[:, 1:]  # align with targets (skip the first token which has no previous token to predict it)
        token_nll = -jnp.sum(targets * log_probs, axis=-1)  # negative log-likelihood per token
        # Compute mean loss per sequence (only averaging over actual target tokens)
        seq_loss = jnp.sum(token_nll * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), a_min=1)
        return seq_loss  # shape [B], per-sequence loss

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
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1  # positions of prefix tokens (0-indexed)
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
        tokens_to_decode = max_decoding_steps
        # Prepare attention mask for single-step decoding (prefix length + current step)
        # We will update the attention mask dynamically in the loop if needed

        # Use a while loop to generate tokens until done or max steps
        def cond_fn(state):
            _rng, _last, _cache, _step, _out = state
            has_eos = jax.lax.cond(
                _step == 0,
                lambda _: False,
                lambda _: jnp.all(_out[:, _step - 1] == _pi0_fast.PALIGEMMA_EOS_TOKEN),
                operand=None,
            )
            return jnp.logical_and(_step < max_decoding_steps, ~has_eos)

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
            positions = (prefill_len[:, None] + step).astype(jnp.int32)
            # Gemma‑fast 无 token=kwarg：先嵌入，再 decode 一步
            token_emb = self.PaliGemma.llm(token[:, None], embed_only=True)
            logits, cache = self.PaliGemma.llm(
                embedded_prefix=token_emb,
                kv_cache=cache,
                positions=positions,
                decode=True,
            )
            return rng_key, logits, cache, step+1, out_tokens

        init_state = (rng, last_logit, kv_cache, jnp.array(0, jnp.int32), output_tokens)
        _, _, _, final_step, output_seq = jax.lax.while_loop(cond_fn, body_fn, init_state)

        # Return the output sequence of tokens as the model's predicted "actions"
        # (In practice, these tokens might represent discretized actions or a planned sequence encoded as text tokens)
        return output_seq[:, :final_step]   # [B, <=max_dec_steps]
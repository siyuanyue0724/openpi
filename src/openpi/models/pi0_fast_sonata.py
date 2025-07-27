import dataclasses
import inspect
import logging
import typing
import jax
import jax.numpy as jnp
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
from openpi.shared import download     # utility for downloading resources like weights

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
        pd["batch"] = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
        pd["offset"] = jnp.cumsum(jnp.full((B,), N, dtype=jnp.int32))

    if "grid_size" in pd and pd["grid_size"].ndim == 3:
        pd["grid_size"] = jnp.reshape(pd["grid_size"], (-1, 3))

    for key in ("batch", "offset"):
        if key in pd:
            pd[key] = pd[key].astype(jnp.int32, copy=False)

    return pd

# Alias the Sonata class from the sonata_encoder module for convenience
Sonata = sonata_encoder.Sonata

# ---------------------------------------------------------------------------
#  helper: 过滤掉目标构造函数不支持的 kwargs
# ---------------------------------------------------------------------------
def _filter_kwargs_for_call(target, kw_dict, *, verbose: bool = False):
    """
    Parameters
    ----------
    target : class | callable
        要实例化 / 调用的对象（如 _gemma.Module）
    kw_dict : Mapping[str, Any]
        原始 kwargs
    verbose : bool
        是否打印被丢弃的字段
    """
    sig = inspect.signature(target)
    accepted = {}
    dropped = []
    for k, v in kw_dict.items():
        if k in sig.parameters:
            accepted[k] = v
        else:
            dropped.append(k)
    if verbose and dropped:
        logger.debug("%s: dropped unused kwargs %s", target.__name__, dropped)
    return accepted

@dataclasses.dataclass(frozen=True)
class Pi0FASTSonataConfig(_pi0_fast.Pi0FASTConfig):
    """Configuration for the Pi0FASTSonata model (Pi0FAST with Sonata point cloud encoder)."""
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
        raw_img_kwargs = dict(
            # _siglip.Module 可能不需要 num_classes；如果 signature 里没有会被自动丢弃
            num_classes=getattr(pal_cfg, "width", getattr(pal_cfg, "hidden_size", 1024)),
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=config.dtype,
        )
        img_kwargs = _filter_kwargs_for_call(_siglip.Module, raw_img_kwargs, verbose=True)
        img = nnx_bridge.ToNNX(_siglip.Module(**img_kwargs))
        # Initialize image encoder with a dummy image to set dimensions
        dummy_image = next(iter(config.fake_obs(batch_size=1).images.values()))
        img.lazy_init(dummy_image, train=False, rngs=rngs)


        # ------------------------------------------------------------------
        # 3) 创建并加载点云编码器 (Sonata) 权重
        # ------------------------------------------------------------------
        point_model = Sonata()

        if config.use_pretrained_point:
            # 先尝试用 huggingface_hub（断点续传 + etag）
            ckpt_path: Path | None = None
            if _HF_OK:
                try:
                    ckpt_path = Path(
                        hf_hub_download(
                            repo_id="facebook/sonata",
                            filename="pretrain-sonata-v1m1-0-base.pth",
                            resume_download=True,
                            # 与 openpi 统一缓存目录
                            cache_dir=download.get_cache_dir(),
                        )
                    )
                    logger.info(
                        "Sonata weights downloaded via huggingface_hub: %s", ckpt_path
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "huggingface_hub download failed (%s). "
                        "Falling back to openpi.download.maybe_download()", e
                    )

            if ckpt_path is None:
                # 回退到原 aiohttp + .partial 方案
                url = (
                    "https://huggingface.co/facebook/sonata/resolve/main/"
                    "pretrain-sonata-v1m1-0-base.pth"
                )
                ckpt_path = download.maybe_download(url)

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
                if k.startswith(("module.", "model.")):
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
            def __init__(self, pt_model: torch.nn.Module, device: torch.device):
                super().__init__()
                # 确保 pt_model 已在目标 device
                self.inner = pt_model.to(device).eval()
                self.device = device

            # ---------- 内部 util ----------
            @staticmethod
            @torch.no_grad()
            def _torch_forward(inner: torch.nn.Module,
                               host_dict: dict[str, np.ndarray],
                               device: torch.device) -> np.ndarray:
                """
                接收 host 上的 numpy 输入 → torch.Tensor.cuda → 运行 → numpy 输出
                统一返回 float32 numpy 数组
                """
                # ---------- 先安全扁平化 ----------
                if host_dict["coord"].ndim != 2:         # (B,N,3) → (B*N,3)
                    B, N, _ = host_dict["coord"].shape
                    host_dict["coord"] = host_dict["coord"].reshape(B * N, 3)
                    host_dict["feat"]  = host_dict["feat"].reshape(B * N, -1)
                    host_dict["batch"] = np.repeat(np.arange(B, dtype=np.int64), N)
                    host_dict["offset"] = np.cumsum(np.full((B,), N, dtype=np.int64))

                if "grid_size" in host_dict and host_dict["grid_size"].ndim == 3:
                    host_dict["grid_size"] = host_dict["grid_size"].reshape(-1, 3)
                
                assert host_dict["coord"].shape[0] == host_dict["offset"][-1], (
                    f"coord N={host_dict['coord'].shape[0]}, "
                    f"but offset[-1]={host_dict['offset'][-1]}"
                )

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
                # Sonata 返回 dict 时取 "feat"，否则取首 value
                if isinstance(out, dict):
                    out = out.get("feat", list(out.values())[0])
                return out.float().cpu().numpy()

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
                        # 生成分散坐标：0..N*3‑1 reshape 为 (N,3)
                        N = shape[0] or 1          # 静态维度；Tracer 上也可读取
                        coords = np.arange(N * 3, dtype=dtype).reshape(N, 3)
                        dummy_np[k] = coords
                    elif k == "offset":
                            # 正确的累计点数 (= coord 的行数)
                            n_pts = host_inputs["coord"].shape[0]          # 64
                            dummy_np[k] = np.array([n_pts], dtype=dtype)   # [64]
                    else:
                        # feat / batch 等 → 全 0
                        dummy_np[k] = np.zeros(shape, dtype=dtype)
                # -----------------------------------------------------------

                dummy_out = self._torch_forward(self.inner, dummy_np, self.device)
                out_struct = ShapeDtypeStruct(dummy_out.shape, jnp.float32)

                def _host_call(*flat_np):
                    # flat_np 是回传的扁平列表/元组，需用 treedef.unflatten 还原
                    np_dict = treedef.unflatten(list(flat_np))
                    return self._torch_forward(self.inner, np_dict, self.device)

                return pure_callback(_host_call, out_struct, *flat, vectorized=False)

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


        # ---------- 5) wrap 为 NNX 模块并 lazy_init ----------
        point = nnx_bridge.ToNNX(_TorchSonataWrapper(point_model, self.device))

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
            "feat":  jnp.zeros((N, 6), dtype=jnp.float32),
            "batch": jnp.zeros((N,),  dtype=jnp.int32),      # 1‑D
            "offset": jnp.array([N], dtype=jnp.int32),
            "grid_size": jnp.array([[128, 128, 128]], dtype=jnp.int32),
        }
        dummy_pc = _canonicalize_point_dict(raw_dummy_pc)
        point.lazy_init(dummy_pc, train=False, rngs=rngs)

        # ------------------------------------------------------------------
        # 6) 统一挂到 self.PaliGemma
        # ------------------------------------------------------------------
        self.PaliGemma = nnx.Dict(
            llm=llm,
            img=img,
            point=point,
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

        # 2. Point cloud tokens (from Sonata encoder), if point cloud data is provided
        if obs.pointcloud_data is not None:
            # Run Sonata encoder on the point cloud data to get token embeddings for points
            point_tokens = self.PaliGemma.point(obs.pointcloud_data, train=False)  # shape [B, n_point_tokens, emb_dim]
            token_embeddings.append(point_tokens)
            # All point tokens are valid (assuming pointcloud_data is not padded)
            pt_mask = jnp.ones((point_tokens.shape[0], point_tokens.shape[1]), dtype=bool)
            input_mask.append(pt_mask)
            # Treat point tokens similar to image tokens (no causal dependency among themselves)
            ar_mask.append(jnp.zeros_like(pt_mask, dtype=jnp.int32))

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
        then generates additional tokens up to `max_decoding_steps` or until an EOS token is produced.
        """
        # Preprocess observation (no augmentation, just ensure correct shapes/masks)
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))
        # Embed inputs to get prefix token embeddings and masks
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = _pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)
        # Align all prefix sequences to the right (required for caching in prefix)
        prefix_tokens, prefix_mask, prefix_attn_mask = _pi0_fast.left_to_right_align(prefix_tokens, prefix_mask, prefix_attn_mask)
        prefill_size = prefix_tokens.shape[1]  # total sequence length after alignment (prefix padded to some length)
        prefill_len = jnp.sum(prefix_mask, axis=-1)  # actual prefix length per batch (number of valid tokens)
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
        last_logit = prefix_logits[:, -1:]  # shape [B, 1, vocab_size]
        # Placeholder for generated token outputs (initialize with zeros)
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        # Define one decoding step function
        def step_fn(carry):
            rng_key, last_logit, cache, step = carry
            # Sample next token from last_logit (either greedy argmax or temperature-controlled random sample)
            rng_key, subkey = jax.random.split(rng_key)
            next_token = jax.lax.cond(
                temperature > 1e-6,
                lambda key: jax.random.categorical(key, last_logit[0] / temperature, axis=-1),    # sample with temperature
                lambda key: jnp.argmax(last_logit[0], axis=-1),                                   # greedy
                operand=subkey
            )
            next_token = next_token.astype(jnp.int32)
            # Place the sampled token into the output_tokens at current step position
            output = _pi0_fast.put_along_last_axis(output_tokens, jnp.broadcast_to(step, (output_tokens.shape[0], 1)), next_token[None, None])
            # Check if EOS token was generated (PALIGEMMA_EOS_TOKEN denotes EOS in Gemma vocabulary)
            eos_token = _pi0_fast.PALIGEMMA_EOS_TOKEN
            done = jnp.all(next_token == eos_token)
            # If EOS for all batch elements, we can stop early
            new_rng_key = rng_key
            return (new_rng_key, last_logit, cache, step), output, done

        # Now run autoregressive decoding for at most max_decoding_steps
        tokens_to_decode = max_decoding_steps
        # Prepare attention mask for single-step decoding (prefix length + current step)
        # We will update the attention mask dynamically in the loop if needed
        def decoding_body(carry):
            rng_key, last_logit, cache, step, output_tokens = carry
            # Get next token (sample or argmax)
            rng_key, subkey = jax.random.split(rng_key)
            if temperature > 1e-6:
                token = jax.random.categorical(subkey, last_logit[0] / temperature, axis=-1)
            else:
                token = jnp.argmax(last_logit[0], axis=-1)
            token = token.astype(jnp.int32)
            # Insert token into output_tokens at position `step`
            output_tokens = _pi0_fast.put_along_last_axis(output_tokens, jnp.broadcast_to(step, (output_tokens.shape[0], 1)), token[None, None])
            # If token is EOS for all batches, we can break
            eos_token = _pi0_fast.PALIGEMMA_EOS_TOKEN
            # Prepare attention mask for next step (prefix + generated tokens so far)
            attn_mask = jnp.concatenate(
                [prefix_attn_mask[:, :, :prefill_size + step], jnp.zeros((prefix_attn_mask.shape[0], prefix_attn_mask.shape[1], 1), dtype=bool)], axis=2
            )
            positions = prefix_start + step  # position index for the new token in each batch
            # Continue LLM decoding one step (with KV cache)
            logits, cache = self.PaliGemma.llm(token=token[:, None], kv_cache=cache, positions=positions, decode=True)
            last_logit = logits  # shape [B, 1, vocab_size]
            return (rng_key, last_logit, cache, step + 1, output_tokens)

        # Use a while loop to generate tokens until done or max steps
        rng_key = rng
        kv_cache_state = kv_cache
        output_seq = output_tokens
        last_logits = last_logit
        step = 0
        for _ in range(max_decoding_steps):
            # Sample or take argmax for next token
            rng_key, subkey = jax.random.split(rng_key)
            if temperature > 1e-6:
                token = jax.random.categorical(subkey, last_logits[0] / temperature, axis=-1)
            else:
                token = jnp.argmax(last_logits[0], axis=-1)
            token = token.astype(jnp.int32)
            output_seq = _pi0_fast.put_along_last_axis(output_seq, jnp.broadcast_to(step, (output_seq.shape[0], 1)), token[None, None])
            # Break if EOS token produced for all batch elements
            if jnp.all(token == _pi0_fast.PALIGEMMA_EOS_TOKEN):
                break
            # Compute attention mask for current prefix+output length
            curr_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, 1)))  # extend by 1
            positions = prefix_start + step  # next position index relative to prefix start
            logits, kv_cache_state = self.PaliGemma.llm(token=token[:, None], kv_cache=kv_cache_state, positions=positions, decode=True)
            last_logits = logits
            step += 1

        # Return the output sequence of tokens as the model's predicted "actions"
        # (In practice, these tokens might represent discretized actions or a planned sequence encoded as text tokens)
        return output_seq[:, :step]  # shape [B, step] of generated token IDs
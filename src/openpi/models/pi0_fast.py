import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
# 引入我们在 __init__.py 中定义的枚举类型
from openpi.models import PointBackboneType, ProjectorType

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision (attention mask生成逻辑，保持原状)…"""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned (批中每个序列右对齐)…"""
    # 保持原实现…
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask

def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)

@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # 模型特定默认参数
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250

    # 新增: 点云编码类型和参数（默认不使用点云）
    point_backbone_type: PointBackboneType | None = None  # 点云编码后端，可选"scenescript"/"sonata"
    projector_type: ProjectorType | None = None          # 点云投影器类型，可选"linear"/"mlp"
    max_points: int = 120000    # 每个样本点云最大点数（根据需要调整）
    point_feat_dim: int = 6     # 单个点特征维度（默认6，例如(x,y,z,+其他3维属性)）

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 定义图像输入规格，与原始保持一致
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
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                # 新增: 点云输入规格（如果启用point_backbone）
                **(
                    {
                        "point_clouds": {"pointcloud": jax.ShapeDtypeStruct([batch_size, self.max_points, self.point_feat_dim], jnp.float32)},
                        "point_cloud_masks": {"pointcloud": jax.ShapeDtypeStruct([batch_size], jnp.bool_)},
                    }
                    if self.point_backbone_type is not None
                    else {}
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        # 如果使用LoRA微调，则按原逻辑冻结LLM主干参数
        if "lora" in self.paligemma_variant:
            filters.append(nnx_utils.PathRegex(".*llm.*"))
            if "lora" not in getattr(self, "action_expert_variant", ""):
                # 若只在paligemma使用LoRA，排除action expert部分参数
                filters.append(nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
        elif hasattr(self, "action_expert_variant") and "lora" in self.action_expert_variant:
            filters.append(nnx_utils.PathRegex(".*llm.*_1.*"))
        # 如果任一使用了LoRA，则进一步排除所有LoRA层参数
        has_lora = any("lora" in str(getattr(self, attr, "")) for attr in ["paligemma_variant", "action_expert_variant"])
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        # 新增: 如果使用点云编码器，则默认冻结其参数（假定为预训练权重）
        if self.point_backbone_type is not None:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*PointBackbone.*")))
        # 若无任何过滤规则，则返回Nothing（训练全部参数）
        if not filters:
            return nnx.Nothing
        # 组合所有过滤规则
        return nnx.All(*filters)

class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        # 初始化基类（设定 action_dim, action_horizon, max_token_len 等）
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        # 建立语言模型 (LLM) 和图像编码模块，并延续原有 lazy_init 初始化
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config["width"],
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # 将 LLM 和图像模块打包进模型字典
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # 新增: 按配置初始化点云编码子模块
        if config.point_backbone_type is not None:
            if config.point_backbone_type == PointBackboneType.SONATA:
                # 加载 Sonata 点云编码网络（PyTorch实现），通过NNX桥接
                import openpi.models.sonata_encoder as _sonata
                # 实例化 Sonata 模块 (PyTorch nn.Module)，封装为NNX子模块
                point_model = nnx_bridge.ToNNX(_sonata.Sonata(
                    in_channels=config.point_feat_dim
                    # 注意: 可根据需要传入其他构造参数，如 enc_channels 等，
                    # 此处使用默认配置（输出通道为512）。稍后通过投影器映射到LLM宽度。
                ))
                # Lazy init：对于PyTorch子模块，一般在首次调用时初始化参数即可，无需显式 lazy_init
                # （Sonata中线性层权重已在构造函数创建）。
            elif config.point_backbone_type == PointBackboneType.SCENESCRIPT:
                # Scenescript 点云编码逻辑暂未实现
                raise NotImplementedError("PointBackboneType.SCENESCRIPT is not implemented in Pi0FAST.")
            else:
                raise ValueError(f"Unknown point_backbone_type: {config.point_backbone_type}")

            # 如果指定投影器，则初始化相应子模块，用于将点云特征投影到 LLM 嵌入维度
            if config.projector_type is not None and config.projector_type != ProjectorType.LINEAR:
                # （可扩展）如需MLP等更复杂的投影，可在此添加实现
                raise NotImplementedError("Only linear projector is implemented for point features.")
            projector_model = None
            if config.projector_type is None or config.projector_type == ProjectorType.LINEAR:
                # 默认使用线性投影：将Sonata输出的特征维度映射到paligemma宽度
                import torch
                linear_layer = torch.nn.Linear(
                    in_features=_sonata.Sonata().enc_channels[-1] if hasattr(_sonata.Sonata, "enc_channels") else 512,
                    out_features=paligemma_config["width"],
                    bias=True
                )
                projector_model = nnx_bridge.ToNNX(linear_layer)
                # 注意: 以上使用 PyTorch Linear 便于集成，其参数初始化已完成。
                # 如果想使用 Flax 实现线性层，也可通过 self.param() 定义权重然后 jnp.dot 应用。

            # 将点云编码模块及投影模块保存为模型属性，方便在 embed_inputs 中调用
            self.PointBackbone = point_model  # Sonata编码器作为PointBackbone子模块
            self.PointProjector = projector_model  # 线性投影子模块（若未配置projector_type则可能为None）
        else:
            self.PointBackbone = None
            self.PointProjector = None

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """将Observation中的各模态输入嵌入表示，并拼接成Transformer输入序列。"""
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # 1. 处理图像输入: 遍历每个摄像头视角，将图像嵌入为视觉令牌序列
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            token_embeddings.append(image_token_embeddings)
            # 图像mask: 将每张图像存在标志扩展为每个图像令牌的mask
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # 图像令牌彼此可相互注意（非生成性前缀），因此对应的AR mask全为0
            ar_mask.append(jnp.zeros_like(input_mask[-1]))

        # 2. 新增: 处理点云输入（若配置启用）
        if self.PointBackbone is not None:
            # 获取点云数据和mask（假设point_clouds以{"pointcloud": array}字典存储）
            point_data = obs.point_clouds["pointcloud"]  # 形状 [B, max_points, point_feat_dim]
            # **设计说明**: 这里假设每个batch样本都有等量的点云采样(max_points)。为了在PyTorch子模块中处理，我们将批内点展平。
            bsz, max_pts, feat_dim = point_data.shape
            # 将JAX中的点云拆分为PyTorch期望的格式: 
            # 平坦坐标与特征，以及对应的batch索引
            coords = point_data[..., :3]   # 假定前3个维度为xyz坐标
            feats = point_data[..., 3:] if feat_dim > 3 else jnp.zeros((bsz, max_pts, 0), dtype=point_data.dtype)
            # 展开批次和点维度
            coords_flat = coords.reshape(bsz * max_pts, 3)
            feats_flat = feats.reshape(bsz * max_pts, -1) if feats.size > 0 else None
            # 构造每个点对应的batch索引 (shape=[bsz*max_pts])
            batch_indices = jnp.repeat(jnp.arange(bsz), max_pts)
            # 为满足Sonata输入要求，提供grid_size参数（例如0.01米，用于量化坐标）
            grid_size = jnp.array(0.01, dtype=jnp.float32)
            # 准备输入字典
            data_dict = {
                "coord": coords_flat,
                "feat": feats_flat if feats_flat is not None else jnp.zeros((bsz * max_pts, 0), jnp.float32),
                "batch": batch_indices,
                "grid_size": grid_size
            }
            # 调用Sonata编码模块获得点云上下文特征 (PyTorch实现通过NNX桥接)
            point_context, = self.PointBackbone(data_dict)  # 输出 shape [bsz*max_pts, enc_out_dim]
            point_context = point_context.astype(jnp.float32)  # 转为JAX类型
            # 如果有投影器，则将点云特征映射到模型嵌入维度
            if self.PointProjector is not None:
                point_context, = self.PointProjector(point_context)
            # 将平坦输出重新整理为 [B, max_pts, emb_dim]
            point_token_embeddings = point_context.reshape(bsz, max_pts, -1)
            token_embeddings.append(point_token_embeddings)
            # 构造对应的mask: 简单起见，假设每个样本点云均存在且填充后长度相同
            # 因此直接将传感器存在标志repeat（若需要按实际点数mask，可将point_cloud_masks定义为每点mask）
            point_mask = einops.repeat(
                obs.point_cloud_masks["pointcloud"], "b -> b s", s=point_token_embeddings.shape[1]
            )
            input_mask.append(point_mask)
            # 点云令牌同样作为前缀输入，不相互依赖，AR mask置0
            ar_mask.append(jnp.zeros_like(point_mask))

        # 3. 处理文本 token 输入（提示符）
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # 拼接各模态嵌入序列和对应mask，并返回
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 预处理Observation（包括规范化图像等），此处会忽略未知字段（点云）保持不变
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        # 将所有输入嵌入
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        # 构造全序列注意力mask
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 计算下一个token的one-hot目标（预测任务）
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )
        # 前向通过语言模型（prefix + suffix一次性输入）
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        # 计算输出logits（仅针对目标部分，以节省显存）
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)
        # 计算交叉熵损失（仅在有标记的token位置）
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_logprob = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_logprob * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        # 将Observation进行预处理（类似compute_loss，但不需要随机性）
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        # 嵌入前缀输入（图像、点云、文本提示）
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # 将所有前缀令牌序列右对齐
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # 将前缀通过LLM，以填充KV缓存
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # 为进入解码准备起点
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry
            # 从上一步logit采样下一个token
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)
            # 判断是否提前结束（所有序列都生成EOS）
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)
            # 将新token嵌入，再通过LLM解码一步
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )
            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # 使用while_loop执行采样迭代（便于JIT编译整个生成过程）
        _, output_tokens, _, _, _ = jax.lax.while_loop(cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0))
        return output_tokens

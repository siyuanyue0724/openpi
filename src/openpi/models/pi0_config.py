from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Any

# Optional JAX imports (allow import in PyTorch-only envs)
try:  # pragma: no cover - best-effort fallback
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except Exception:  # fallback stubs
    jax = None  # type: ignore
    jnp = None  # type: ignore
    import numpy as _np

    class _ShapeDtypeStruct:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = _np.dtype(dtype)

    class _JaxStub:
        ShapeDtypeStruct = _ShapeDtypeStruct

    jax = _JaxStub()  # type: ignore
    jnp = _np  # type: ignore

from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
# NOTE: avoid importing array_typing & nnx_utils at module import time;
# they may depend on JAX/Flax. We import them lazily inside functions.

# 新枚举（软依赖，老分支不影响）
try:
    from openpi.models import PointBackboneType
except Exception:  # 软降级：若无该枚举，旧布尔开关仍可用
    PointBackboneType = object  # type: ignore

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False

    # ===== Point cloud (Sonata) 配置（严格 fail-fast） =====
    # 旧布尔开关（向后兼容）：是否启用点云。
    # `None` 表示“交由具体后端解释默认值”：
    # - PyTorch 路线默认开启
    # - JAX 路线默认关闭，除非显式要求或 point_backbone_type=SONATA
    enable_sonata: Optional[bool] = None
    # 新枚举（推荐）：选择点云后端（存在即用）
    point_backbone_type: Optional["PointBackboneType"] = None  # type: ignore[name-defined]
    # feats = [xyz(3) + extras]，常见 xyz+rgb => 6；Observation 的最后一维为 3 + point_feat_dim
    point_feat_dim: int = 6
    # 每帧原始点的静态上界（仅用于 inputs_spec 的静态形状；真实可小于等于此值）
    max_points: int = 32768
    # 编码后固定 token 上限（JAX 需要静态长度）
    point_token_cap: int = 1024
    # Sonata 编码器超参覆盖（None 用 SpatialLM 默认）
    point_config: dict | None = None
    # 权重路径（优先本字段，其次 OPENPI_SONATA_CKPT；找不到仅 warning）
    sonata_ckpt_path: str | None = None
    # 是否强制需要 CUDA（True=强制；False=允许 CPU 回退）。
    require_cuda: bool = True
    # 严格策略固定开启（fail-fast）；此字段保留仅为兼容，不影响行为
    strict_point_checks: bool = True
    # 是否尝试加载预训练（找不到仅警告，不阻断）
    use_pretrained_point: bool = True
    # 原位插入用的哨兵 token id（启用点云并做 <|point_start|>/<|point_end|> 插入时必须提供）。
    # 与 PaligemmaTokenizer 的约定保持一致：point_start_id = vocab_size - 2；point_end_id = vocab_size - 1。
    # 推荐在训练入口“显式”写入：
    #   from openpi.models.tokenizer import PaligemmaTokenizer
    #   vsz = int(PaligemmaTokenizer()._tokenizer.vocab_size())
    #   cfg.point_start_id, cfg.point_end_id = vsz - 2, vsz - 1
    point_start_id: Optional[int] = None
    point_end_id:   Optional[int] = None
    # Sonata runtime mode. None -> defer to environment / model default.
    sonata_mode: Optional[str] = None
    # Legacy alias kept for backward compatibility with older scripts.
    sonata_train_mode: Optional[str] = None
    # Optional projector checkpoint for point-token -> language-token projection.
    sonata_projector_ckpt_path: str | None = None
    # Optional runtime switches. None -> defer to environment / model defaults.
    sonata_validate: Optional[bool] = None
    sonata_auto_pad_feat: Optional[bool] = None

    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

        mode = self.sonata_mode
        legacy_mode = self.sonata_train_mode
        if mode is not None:
            mode = str(mode).strip().lower()
        if legacy_mode is not None:
            legacy_mode = str(legacy_mode).strip().lower()
        if (mode is not None) and (legacy_mode is not None) and (mode != legacy_mode):
            raise ValueError(
                f"sonata_mode={self.sonata_mode!r} conflicts with legacy sonata_train_mode={self.sonata_train_mode!r}."
            )
        if mode is None:
            mode = legacy_mode
        if mode is not None:
            if mode not in {"off", "projector", "all"}:
                raise ValueError(f"Invalid sonata_mode={mode!r}. Expected one of: off|projector|all")
            object.__setattr__(self, "sonata_mode", mode)
        if legacy_mode is not None:
            object.__setattr__(self, "sonata_train_mode", legacy_mode)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng) -> "Pi0":
        """Create the JAX model (only for JAX runs)."""
        try:
            import flax.nnx as nnx  # type: ignore
            from openpi.shared import array_typing as at  # type: ignore
            from openpi.models.pi0 import Pi0  # type: ignore
        except Exception as e:  # pragma: no cover - only raised on PyTorch-only envs
            raise RuntimeError("Pi0Config.create() requires JAX/Flax; not available in PyTorch-only environment.") from e
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # bool dtype fallback: numpy bool_ works with our FakeDataset logic
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], getattr(jnp, "bool_", bool))

        # 是否启用点云（双开关兼容）
        enable_pc = False
        if self.enable_sonata is True:
            enable_pc = True
        try:
            if self.point_backbone_type == PointBackboneType.SONATA:  # type: ignore[attr-defined]
                enable_pc = True
        except Exception:
            pass

        # 注意：此函数在 PyTorch-only 环境下只为 FakeDataset 提供 shape/dtype 信息；
        # 不会真正触发 JAX/Flax 执行。
        try:
            from openpi.shared import array_typing as at  # type: ignore
            _disable_ctx = at.disable_typechecking()
        except Exception:
            # Fallback context manager that does nothing
            class _NullCtx:
                def __enter__(self): return self
                def __exit__(self, *exc): return False
            _disable_ctx = _NullCtx()

        with _disable_ctx:
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                point_clouds=(
                    {
                        "pointcloud": jax.ShapeDtypeStruct(
                            [batch_size, self.max_points, 3 + self.point_feat_dim], jnp.float32
                        )
                        # 约定：最后一维 = 3 + point_feat_dim，其中前 3 为 grid_coord（推荐上游直接给 int32/int64），
                        # 后面包含 [xyz(3) + extras]；模型内部会校验 feats[:,:3] == coord(xyz)。
                    } if enable_pc else {}
                ),
                point_cloud_masks=(
                    {"pointcloud": jax.ShapeDtypeStruct([batch_size], getattr(jnp, "bool_", bool))} if enable_pc else {}
                ),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], getattr(jnp, "bool_", bool)),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> Any:
        """Returns the freeze filter based on the model config (JAX-only).
        On PyTorch-only environment, returns None.
        """
        try:
            import flax.nnx as nnx  # type: ignore
            import openpi.shared.nnx_utils as nnx_utils  # type: ignore
        except Exception:
            return None
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing  # type: ignore[attr-defined]
        return nnx.All(*filters)  # type: ignore[attr-defined]

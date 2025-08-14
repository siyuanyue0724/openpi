import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import jax.dlpack as jdlpack
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import torch
import warnings
from torch.utils import dlpack as tdlpack

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


# === FULL LOG PATCH ==========================================================
import logging, sys
from pathlib import Path

# 日志文件位置；按需修改
LOG_PATH = Path("train_full.log")      # 或 Path("/abs/path/to/train.log")

# ----------------------------------------------------------------------------- 
# 1) 先把 stdout/stderr 都“tee”到文件
def _setup_stdout_stderr_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)       # 行缓冲，实时写盘

    class Tee:
        """Write to several streams at once (保持 tqdm 检测 isatty 正常)."""
        def __init__(self, *streams): self._streams = streams
        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass
        def isatty(self):                 # 让 tqdm 仍能刷新进度条
            return any(getattr(s, "isatty", lambda: False)() for s in self._streams if s is not log_file)

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

# ----------------------------------------------------------------------------- 
# 2) 再配置 logging：终端 + 文件双写
#    若根 logger 还没有 handler，需要先 basicConfig 保证 handlers[0] 存在，
#    避免某些环境下 logger.handlers[0] 触发 IndexError。
def init_logging(log_file: str | Path | None = None, level: int = logging.INFO):
    """
    初始化 logging，且确保 *所有* 终端输出同步写入同一日志文件。
    - log_file: 自定义路径；默认使用 LOG_PATH。
    - level: 根 logger 等级；默认 INFO。
    """
    log_path = Path(log_file) if log_file is not None else LOG_PATH
    _setup_stdout_stderr_tee(log_path)        # 先完成 tee

    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

# 防止重复添加 handler，注意，如果第三方库等添加handler，这个可能带来问题
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.handlers:                   # 关键：保证至少有一个 handler
        logging.basicConfig(level=level)
    logger.handlers.clear()                   # 清空后再加我们自己的


    # ① 终端 handler
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ② 文件 handler（追加写入）
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# === END PATCH ===============================================================


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """
    允许“子集加载”：只要已加载的叶子是 expected 的子树，且每个叶子的 shape/dtype 一致即可；
    缺失的子树（例如 PaliGemma/point_proj）保留模型初始化值。
    同时丢弃任何 unexpected key。
    """
    loaded_params = loader.load(params_shape)
    if loaded_params is None:
        logging.warning("[weights] Loader returned None; proceeding with randomly initialized params.")
        return {}  # 空子集

    # 展平 expected / loaded，便于对齐比对
    exp_flat = traverse_util.flatten_dict(params_shape, keep_empty_nodes=True)
    got_flat = traverse_util.flatten_dict(loaded_params, keep_empty_nodes=True)

    # 1) 先丢弃 loaded 里是 ShapeDtypeStruct 的占位，以及 exp 中不存在的“意外键”
    pruned_got = {}
    dropped_unexpected = []
    dropped_shape_stub = 0
    for k, v in got_flat.items():
        if isinstance(v, jax.ShapeDtypeStruct):
            dropped_shape_stub += 1
            continue
        if k not in exp_flat:
            dropped_unexpected.append("/".join(k))
            continue
        pruned_got[k] = v

    if dropped_unexpected:
        logging.warning("[weights] Dropping %d unexpected keys from checkpoint (e.g. %s)",
                        len(dropped_unexpected), dropped_unexpected[:3])
    if dropped_shape_stub:
        logging.debug("[weights] Dropped %d ShapeDtypeStruct placeholders from loaded params.", dropped_shape_stub)

    # 2) 构造 expected 的“子集”用于严格校验 shape/dtype
    exp_subset = {k: exp_flat[k] for k in pruned_got.keys()}
    exp_subset_tree = traverse_util.unflatten_dict(exp_subset)
    got_subset_tree = traverse_util.unflatten_dict(pruned_got)

    # 3) 对子集进行严格 shape/dtype 校验
    at.check_pytree_equality(
        expected=exp_subset_tree,
        got=got_subset_tree,
        check_shapes=True,
        check_dtypes=True,
    )

    # 4) 统计缺失键（信息提示，不是错误）：这些会用模型初始化权重
    missing = sorted(set(exp_flat.keys()) - set(pruned_got.keys()))
    if missing:
        # 汇总到顶层模块名/一级路径，避免日志过长
        top_levels = sorted({m[0] for m in missing if len(m) > 0})
        logging.info("[weights] Loaded subset of params: %d/%d leaves. "
                     "Missing subtrees will keep model init values. Top-level missing: %s",
                     len(pruned_got), len(exp_flat), top_levels[:5])

    # 5) 返回“已加载子集”，由上层通过 state.replace_by_pure_dict 合并
    return got_subset_tree


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    # ------------------------------------------------------------------ #
    # ① 仅创建一次 full model → 抽出模板，后续 init() 全复用它           #
    # ------------------------------------------------------------------ #
    model_rng, init_rng = jax.random.split(init_rng)
    _full_model        = config.model.create(model_rng)
    _model_def_tmpl    = nnx.graphdef(_full_model)
    _params_tmpl       = nnx.state(_full_model)      # 只保存 shape/dtype

    def init(rng: at.KeyArrayLike,
             partial_params: at.Params | None = None
             ) -> training_utils.TrainState:
        # ② 复制模板，不再重新 model.create()
        model = nnx.merge(_model_def_tmpl, _params_tmpl)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 形状推断再也不会触发 model.create()，保证 NodeDef 元数据一致
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 真正初始化，同样复用模板
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # partial_params 可捐赠
        # ⚠️ 两个位置参数：rng 与 partial_params —— 必须给两棵树
        in_shardings=(replicated_sharding, replicated_sharding),
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

@at.typecheck
def train_step_all(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    pt_feat_override: at.Array,
    pt_mask_override: at.Array,
) -> tuple[training_utils.TrainState, dict[str, at.Array], at.Array]:
    """
    all 模式专用的训练步：
      • JAX 侧：更新除 Sonata.encoder 以外的训练参数（含 projector/LLM/SigLIP/LoRA 等）
      • 返回对输入 override（pt_feat_override）的梯度，供 host 侧 Torch 回传更新 Sonata.encoder
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        feat_override: at.Array,
        mask_override: at.Array,
    ):
        # 关键：把 override 显式传入，驱动 A 方案
        chunked_loss = model.compute_loss(
            rng, observation, actions, train=True,
            pt_feat_override=feat_override, pt_mask_override=mask_override,
        )
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # --------- (1) 对 NNX 可训练参数求梯度（保持原逻辑的 filter） ---------
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 优先尝试“一次反向”同时拿到 (参数梯度, 输入特征梯度)；不支持则回退到双路计算
    def _lfn_for_both(m, r, o, a, fov, mov):
        return loss_fn(m, r, o, a, fov, mov)
    try:
        loss, (grads, g_feat) = nnx.value_and_grad(_lfn_for_both, argnums=(diff_state, 4))(
            model, train_rng, observation, actions, pt_feat_override, pt_mask_override
        )
    except TypeError:
        # 某些 nnx 版本不支持 (DiffState, int) 复合 argnums：回退到原实现
        loss, grads = nnx.value_and_grad(_lfn_for_both, argnums=diff_state)(
            model, train_rng, observation, actions, pt_feat_override, pt_mask_override
        )
        def _lfn_for_feat(fov):
            return loss_fn(model, train_rng, observation, actions, fov, pt_mask_override)
        g_feat = jax.grad(_lfn_for_feat)(pt_feat_override)  # ← 回退：第二遍只对特征求梯度

    # --------- (2) 应用梯度到 JAX 侧参数 ---------
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")), lambda _, x: x.value.ndim > 1),
    )
    info = {"loss": loss, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(kernel_params)}
    return new_state, info, g_feat

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 定向过滤 spconv 内部的 AMP 自定义算子 FutureWarning（不影响训练正确性）
    try:
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.cuda\.amp\.custom_",
            category=FutureWarning,
            module=r"spconv\.pytorch\.functional",
        )
    except Exception:  # 安全兜底
        pass

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_val_dict = batch[0].images
    B0 = next(iter(images_val_dict.values())).shape[0]
    images_to_log = []
    for i in range(min(5, B0)):
        cat = np.concatenate([np.array(img[i]) for img in images_val_dict.values()], axis=1)
        if cat.dtype != np.uint8:
            if np.issubdtype(cat.dtype, np.floating):
                cat = np.clip(cat, 0, 1)
                cat = (cat * 255).astype(np.uint8)
            else:
                cat = cat.astype(np.uint8, copy=False)
        images_to_log.append(wandb.Image(cat))
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        # resume: 先恢复，再按需阻塞
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    else:
        # 非 resume：可选阻塞，确保初始化完成（只对真实数组调用）
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            train_state.params,
        )
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # ============== 根据训练模式选择 jitted 步骤（保持非 all 模式完全不变） ==============
    mode_all = str(getattr(config.model, "sonata_train_mode", "projector")).lower() == "all"
    if mode_all:
        # all：额外接受 override（按 data_sharding 分片），并额外返回对 override 的梯度（同分片）
        ptrain_step = jax.jit(
            functools.partial(train_step_all, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding, data_sharding),
            donate_argnums=(1,),
        )
    else:
        # projector / frozen：原样
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # ============== all 模式：构造 Torch 端 Sonata 优化器（独立于 JAX 优化器） ==============
    if mode_all:
        # 复用当前 GraphDef/Params 合并出的模型实例以拿到 Torch Sonata 句柄；
        # 不必担心与 JAX 参数不同步：all 模式里点云分支走 override，不使用模型内置的 Sonata。
        host_model_for_sonata = nnx.merge(train_state.model_def, train_state.params)
        # 通过模型方法稳健获取 Torch Sonata 句柄（避免 GraphDef/Params merge 后属性丢失）
        try:
            sonata_torch = host_model_for_sonata.get_torch_sonata()
        except AttributeError as e:
            raise RuntimeError("[SONATA/all] Failed to obtain Torch Sonata handle from model.") from e
        # 独立优化器（可用 config.sonata_lr / config.sonata_weight_decay 覆盖）
        sonata_lr = float(getattr(config, "sonata_lr", 1e-4))
        sonata_wd = float(getattr(config, "sonata_weight_decay", 0.0))
        fused_ok = False
        try:
            sonata_optim = torch.optim.AdamW(
                sonata_torch.parameters(), lr=sonata_lr, weight_decay=sonata_wd,
                betas=(0.9, 0.95), eps=1e-8, fused=True
            )
            fused_ok = True
        except TypeError:
            sonata_optim = torch.optim.AdamW(
                sonata_torch.parameters(), lr=sonata_lr, weight_decay=sonata_wd, betas=(0.9, 0.95), eps=1e-8
            )
        logging.info(f"[SONATA/all] Torch optimizer ready (lr={sonata_lr}, wd={sonata_wd}, fused={fused_ok}).")
        sonata_max_grad_norm = float(getattr(config, "sonata_max_grad_norm", 0.0) or 0.0)

    infos = []
    last_pt_usage_stats = None
    for step in pbar:
        with sharding.set_mesh(mesh):
            if mode_all:
                # 1) Torch 上先做 Sonata 前向（batch 级），产生 override
                #    注意：该函数内部会把 JAX / Sharded 数组转到 host（通过 np.asarray），
                #    但输出（pt_feat_t）→ JAX 我们优先用 DLPack（单卡零拷贝；多卡回退 host 拷贝）
                pt_feat_t, pt_mask_t = host_model_for_sonata.torch_sonata_encode_batch(batch[0], train=True)
                pt_feat_t.requires_grad_(True)  # 让 autograd 能把梯度传回 Sonata
                # Torch -> JAX：先做设备一致性检查；仅在“同一 GPU”时走 DLPack，否则回退 host-copy
                use_dlpack = False
                if jax.device_count() == 1:
                    try:
                        j_dev = jax.devices()[0]
                        t_dev = pt_feat_t.device
                        # 要求 Torch 在 CUDA，JAX 在 GPU，且设备索引一致（无索引则视为 0）
                        same_gpu = (t_dev.type == "cuda") and (j_dev.platform == "gpu") \
                                   and ((t_dev.index is None) or (int(getattr(j_dev, "id", 0)) == int(t_dev.index)))
                        use_dlpack = bool(same_gpu)
                        if not use_dlpack:
                            logging.warning(
                                "[SONATA/all] Torch/JAX devices mismatch for DLPack (torch=%s, jax=%s). "
                                "Falling back to host-copy.", str(t_dev), str(j_dev)
                            )
                    except Exception as _e:
                        logging.warning("[SONATA/all] Device check failed (%s); falling back to host-copy.", repr(_e))
                        use_dlpack = False

                if use_dlpack:
                    # ① 不能导出 requires_grad=True → detach；同时**强制默认内存布局**
                    export_feat = pt_feat_t.detach().contiguous(memory_format=torch.contiguous_format)
                    # 掩码一般不需要 grad；统一做一次 contiguous
                    export_mask = (
                        pt_mask_t.detach().contiguous() if pt_mask_t.requires_grad
                        else pt_mask_t.contiguous()
                    )
                    # ② 为避免 JAX 报 non-default layout，先把 (B,P,C) 展平成 2D 再导出
                    B_t, P_t, C_t = export_feat.shape
                    export_feat_2d = export_feat.view(B_t * P_t, C_t).contiguous()
                    try:
                        pt_feat_j = jdlpack.from_dlpack(export_feat_2d)
                    except Exception:
                        # 老版本栈：回退 capsule
                        pt_feat_j = jdlpack.from_dlpack(tdlpack.to_dlpack(export_feat_2d))
                    # 还原形状
                    pt_feat_j = jnp.reshape(pt_feat_j, (B_t, P_t, C_t))

                    # 掩码同理（2D/1D 导出，避免 strides 被记录成“非默认”）
                    MB, MP = export_mask.shape
                    export_mask_1d = export_mask.view(MB * MP).contiguous()
                    try:
                        pt_mask_j = jdlpack.from_dlpack(export_mask_1d)
                    except Exception:
                        pt_mask_j = jdlpack.from_dlpack(tdlpack.to_dlpack(export_mask_1d))
                    pt_mask_j = jnp.reshape(pt_mask_j, (MB, MP))
                    # 保底：转成布尔（部分栈可能给 uint8）
                    if pt_mask_j.dtype != jnp.bool_:
                        pt_mask_j = pt_mask_j.astype(jnp.bool_)
                else:
                    # 多设备沿用 host 拷贝回退
                    pt_feat_j = jnp.asarray(pt_feat_t.detach().cpu().numpy())
                    pt_mask_j = jnp.asarray(pt_mask_t.detach().cpu().numpy())

                # 2) JAX 步（更新 projector/LLM/SigLIP/…），并拿到 d(loss)/d(pt_feat_override)
                train_state, info, g_feat_j = ptrain_step(train_rng, train_state, batch, pt_feat_j, pt_mask_j)

                # 3) 反向把 JAX 的梯度喂回 Torch -> 更新 Sonata
                if jax.device_count() == 1:
                    # （可选）确保已经计算完成，减少不必要的同步隐患
                    jax.block_until_ready(g_feat_j)
                    g_feat_t = tdlpack.from_dlpack(jdlpack.to_dlpack(g_feat_j))
                    # 确保梯度 dtype 与前向张量一致（通常 pt_feat_t 是 float32）
                    if g_feat_t.dtype != pt_feat_t.dtype:
                        g_feat_t = g_feat_t.to(pt_feat_t.dtype)
                else:
                    g_feat_host = np.asarray(jax.device_get(g_feat_j))
                    # 明确 dtype 对齐到前向张量 dtype（通常 float32）
                    g_feat_t = torch.as_tensor(g_feat_host, device=pt_feat_t.device, dtype=pt_feat_t.dtype)
                pt_feat_t.backward(g_feat_t)
                if mode_all and sonata_max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(sonata_torch.parameters(), sonata_max_grad_norm)
                sonata_optim.step()
                sonata_optim.zero_grad(set_to_none=True)
                # 点 token 使用率统计（每样本 K；cap = P_cap）
                try:
                    K_host = np.asarray(jax.device_get(jnp.sum(pt_mask_j, axis=1)))
                    cap = int(pt_mask_j.shape[1])
                    last_pt_usage_stats = (K_host, cap)
                except Exception:
                    last_pt_usage_stats = None
            else:
                # projector/frozen：延续原有逻辑
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree_util.tree_map(jnp.mean, stacked_infos))
            def _fmt_metric(v):
                # 支持 Python/NumPy 标量；非数值安全转为字符串，避免 "Unknown format code 'f'"
                try:
                    # bool 也可以 float()，但通常希望打印 True/False；这里把 bool 当作字符串
                    if isinstance(v, (bool, np.bool_)):
                        return str(v)
                    return f"{float(v):.4f}"
                except Exception:
                    return str(v)
            info_str = ", ".join(f"{k}={_fmt_metric(v)}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            # 追加：all 模式下的 Torch 侧范数 & 点 token 使用率
            if mode_all:
                try:
                    with torch.no_grad():
                        sonata_param_norm = torch.sqrt(sum((p.detach().float()**2).sum() for p in sonata_torch.parameters())).item()
                        sonata_grad_norm = torch.sqrt(sum((p.grad.detach().float()**2).sum() for p in sonata_torch.parameters() if p.grad is not None)).item()
                    reduced_info = {**reduced_info,
                                    "sonata_param_norm": float(sonata_param_norm),
                                    "sonata_grad_norm": float(sonata_grad_norm)}
                except Exception:
                    pass
                if last_pt_usage_stats is not None:
                    K_host, cap = last_pt_usage_stats
                    try:
                        reduced_info = {**reduced_info,
                                        "pt_K_max": float(np.max(K_host)) if K_host.size else 0.0,
                                        "pt_K_p95": float(np.percentile(K_host, 95)) if K_host.size else 0.0,
                                        "pt_K_mean_frac": float(np.mean(K_host / cap)) if K_host.size else 0.0}
                        wandb.log({"pt_K_hist": wandb.Histogram(K_host)}, step=step)
                    except Exception:
                        pass
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

        # （可选）投影器 L2 监控：存在 config.proj_l2_log_interval 时启用
        try:
            l2_intv = int(getattr(config, "proj_l2_log_interval", 0))
        except Exception:
            l2_intv = 0
        if l2_intv and step > start_step and (step % l2_intv == 0):
            # 取一个 host 侧模型副本做快速统计
            host_model_dbg = nnx.merge(train_state.model_def, train_state.params)
            delta = host_model_dbg.debug_log_projector_delta(tag=f"(step={step})")
            try:
                wandb.log({"projector_l2_delta": float(delta)}, step=step)
            except Exception:
                pass

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
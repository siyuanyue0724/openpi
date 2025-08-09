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
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

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
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
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


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

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
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

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

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
"""

import os
# Ensure CUDA allocator config is set before torch (or any extension importing torch)
# initializes CUDA. This is a no-op if the user already set it in the shell.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

import dataclasses
import dataclasses as _dc
import gc
import logging
import platform
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
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


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    # Fail-fast: 在 DDP 模式下必须由 torchrun 设置 LOCAL_RANK；禁止把全局 RANK 当本地设备号用
    local_rank_env = os.environ.get("LOCAL_RANK")
    if use_ddp and local_rank_env is None:
        raise RuntimeError("LOCAL_RANK must be set when running under DDP (use torchrun or set LOCAL_RANK per process).")
    # 非 DDP 情况兼容单卡，默认 0
    local_rank = int(local_rank_env or "0")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def _to_device_tree(x, device):
    """dataclass-aware mover: Tensor/dict/list/tuple/any-with-.to()"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if _dc.is_dataclass(x) and not isinstance(x, type):
        return _dc.replace(x, **{f.name: _to_device_tree(getattr(x, f.name), device) for f in _dc.fields(x)})
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        xs = [_to_device_tree(v, device) for v in x]
        return tuple(xs) if isinstance(x, tuple) else xs
    if hasattr(x, "to"):
        try:
            return x.to(device)
        except Exception:
            return x
    return x


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, *, sonata_optimizer=None):
    """Save a checkpoint with model state, optimizer state(s), and metadata."""
    if not is_main:
        return

    # 以“完成的步数”为标签：原循环是先自增再保存，因此最后一步应为 == num_train_steps
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer(s)
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
        if sonata_optimizer is not None:
            torch.save(sonata_optimizer.state_dict(), tmp_ckpt_dir / "sonata_optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": int(global_step),
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "has_sonata_optimizer": sonata_optimizer is not None,
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats：契约加严（必须同时存在或同时缺失）
        norm_stats = getattr(data_config, "norm_stats", None)
        asset_id = getattr(data_config, "asset_id", None)
        if (norm_stats is None) ^ (asset_id is None):
            raise RuntimeError("Inconsistent norm_stats saving contract: norm_stats and asset_id must both exist or both be None.")
        if norm_stats is not None:
            assets_root = tmp_ckpt_dir / "assets" / asset_id
            assets_root.mkdir(parents=True, exist_ok=True)
            _normalize.save(assets_root, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, *, sonata_optimizer=None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        # Load Sonata optimizer state if present
        if sonata_optimizer is not None:
            s_path = ckpt_dir / "sonata_optimizer.pt"
            if s_path.exists():
                try:
                    s_state = torch.load(s_path, map_location=device)
                    sonata_optimizer.load_state_dict(s_state)
                    logging.info("Loaded Sonata optimizer state from pt format")
                    del s_state
                except Exception as e:
                    logging.warning(f"Failed to load Sonata optimizer state ({e!r}); continuing with fresh.")

        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch (fail-fast: require 'image' or 'images')
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_dict = observation.to_dict()
        img_dict = sample_dict.get("images")
        if img_dict is None:
            img_dict = sample_dict["image"]  # 若两者都无，立刻 KeyError（契约不满足就崩）

        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(img_dict.values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in img_dict.values()], dim=1)
            images_to_log.append(wandb.Image(img_concatenated.cpu().numpy()))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Allocator configuration was set at import-time via `setdefault` (see top of file).
        logging.info("Enabled memory optimizations for 8+ GPU training (allocator configured at import-time)")

    if use_ddp:
        # Keep prior-tested behavior: find_unused_parameters=True.
        # static_graph is incompatible with find_unused=True; leave it off unless you switch to find_unused=False.
        find_unused = True
        static_graph = False  # (world_size >= 8) and (not find_unused)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=True,
            static_graph=static_graph,
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
            model_path,
            device=str(device),
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # -------- SONATA 合流（off/projector/all） --------
    def _require(cond: bool, msg: str):
        if not cond:
            raise RuntimeError(msg)

    # Accept both new and legacy env var names; conflict ⇒ fail fast.
    env_new = os.environ.get("OPENPI_SONATA_MODE")
    env_old = os.environ.get("OPENPI_SONATA_TRAIN_MODE")
    if env_new and env_old and env_new.lower() != env_old.lower():
        raise RuntimeError("Conflicting SONATA mode env vars: OPENPI_SONATA_MODE vs OPENPI_SONATA_TRAIN_MODE")
    sonata_mode = (env_new or env_old or getattr(getattr(config, "model", object()), "sonata_train_mode", "off"))
    sonata_mode = str(sonata_mode or "off").lower()
    if sonata_mode not in {"off", "projector", "all"}:
        raise ValueError(f"Invalid SONATA mode: {sonata_mode!r}. Expected one of: off|projector|all")

    def _split_main_and_sonata_params(m, mode: str):
        base = m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m
        s_params = []
        # 1) encoder（旧契约）
        enc = None
        if hasattr(base, "get_torch_sonata"):
            try:
                enc = base.get_torch_sonata()
            except Exception:
                enc = None
        if enc is not None:
            s_params += list(enc.parameters())
        # 2) projector（仅在 all 模式并入 SONATA 优化器；否则留在主干优化器）
        if mode == "all":
            proj = None
            if hasattr(base, "sonata_projector"):
                proj = getattr(base, "sonata_projector")
            elif hasattr(base, "get_torch_sonata_projector"):
                try:
                    proj = base.get_torch_sonata_projector()
                except Exception:
                    proj = None
            if proj is not None:
                s_params += list(p for p in proj.parameters() if p.requires_grad)
        if not s_params:
            return list(base.parameters()), []
        s_ids = {id(p) for p in s_params}
        main = [p for p in base.parameters() if id(p) not in s_ids]
        return main, s_params

    if sonata_mode == "projector":
        base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if hasattr(base, "get_torch_sonata"):
            try:
                enc = base.get_torch_sonata()
                for p in enc.parameters():
                    p.requires_grad = False
                logging.info("[SONATA] projector mode: Sonata encoder parameters are frozen.")
            except Exception:
                logging.info("[SONATA] projector mode requested but encoder not found.")

    main_params, sonata_params = _split_main_and_sonata_params(model, sonata_mode)
    use_sonata_optim = (sonata_mode == "all")
    if use_sonata_optim:
        _require(len(sonata_params) > 0, "sonata_train_mode='all' requires SONATA parameters in the model.")

    # Create optimizer with config parameters（若 all 模式则只管主干，SONATA 单独优化器）
    optim = torch.optim.AdamW(
        main_params if use_sonata_optim else (model.module.parameters() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.parameters()),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    sonata_optim = (torch.optim.AdamW(
        sonata_params,
        lr=float(getattr(config, "sonata_lr", 1e-4)),
        betas=(getattr(config.optimizer, "b1", 0.9), getattr(config.optimizer, "b2", 0.999)),
        eps=getattr(config.optimizer, "eps", 1e-8),
        weight_decay=float(getattr(config, "sonata_weight_decay", 0.0)),
    ) if use_sonata_optim else None)

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, sonata_optimizer=sonata_optim)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training（强契约：要求 __len__ 存在且 >0）
        if use_ddp and hasattr(loader, "set_epoch"):
            if not hasattr(loader, "__len__"):
                raise RuntimeError("Data loader must define __len__() under DDP.")
            iters = len(loader)
            if iters <= 0:
                raise RuntimeError("Data loader length must be > 0 under DDP.")
            loader.set_epoch(global_step // iters)

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # Observation 是 dataclass：to_dict → 递归 .to(device) → from_dict
            if not hasattr(observation, "to_dict"):
                raise RuntimeError("Observation must implement to_dict().")
            obs_dict = observation.to_dict()
            obs_dict = _to_device_tree(obs_dict, device)
            if not hasattr(_model, "Observation") or not hasattr(_model.Observation, "from_dict"):
                raise RuntimeError("openpi.models.model.Observation.from_dict is required.")
            observation = _model.Observation.from_dict(obs_dict)
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR（SONATA 使用缩放后的曲线峰值为 sonata_lr）
            cur_lr = lr_schedule(global_step)
            for pg in optim.param_groups:
                pg["lr"] = cur_lr
            if sonata_optim is not None:
                base_lr = float(getattr(config, "sonata_lr", sonata_optim.defaults.get("lr", 1e-4)))
                scale = (cur_lr / peak_lr) if peak_lr > 0 else 1.0
                cur_lr_sonata = base_lr * scale
                for pg in sonata_optim.param_groups:
                    pg["lr"] = cur_lr_sonata

            # Forward（SONATA 开启时强制契约）
            if sonata_mode in {"projector", "all"}:
                base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                if not hasattr(base_model, "torch_sonata_encode_batch"):
                    raise RuntimeError("SONATA enabled but model.torch_sonata_encode_batch is missing.")
                pt_feat, pt_mask = base_model.torch_sonata_encode_batch(observation, train=True)
                if not (isinstance(pt_feat, torch.Tensor) and isinstance(pt_mask, torch.Tensor)):
                    raise RuntimeError("SONATA encode must return (Tensor, Tensor).")
                pt_feat = pt_feat.to(device)
                pt_mask = pt_mask.to(device)
                if sonata_mode == "all":
                    pt_feat.requires_grad_(True)
                try:
                    losses = model(observation, actions, pt_feat_override=pt_feat, pt_mask_override=pt_mask)
                except TypeError as e:
                    raise TypeError("SONATA enabled but model.forward(...) does not accept pt_feat_override/pt_mask_override.") from e
            else:
                losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping（仅阈值 >0 时启用）
            grad_norm = None
            max_gn = getattr(config.optimizer, "clip_gradient_norm", None)
            if isinstance(max_gn, (int, float)) and max_gn and max_gn > 0:
                params_for_clip = (main_params if sonata_optim is not None
                                   else (model.module.parameters() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.parameters()))
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(max_gn))
                if sonata_optim is not None:
                    torch.nn.utils.clip_grad_norm_(sonata_params, max_norm=float(max_gn))

            # Optimizer step（若 all 模式则同时 step 第二优化器）
            optim.step()
            if sonata_optim is not None:
                sonata_optim.step()
            optim.zero_grad(set_to_none=True)
            if sonata_optim is not None:
                sonata_optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism（以完成步数为名）
            save_checkpoint(model, optim, global_step, config, is_main, data_config, sonata_optimizer=sonata_optim)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()

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
import sys
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


def _torch_load_ckpt(path, *, map_location):
    """torch.load wrapper compatible with PyTorch>=2.6 default behavior changes.

    PyTorch 2.6+ tightened torch.load() defaults (weights_only-style restricted unpickling),
    which can break loading optimizer/metadata checkpoints that contain non-tensor objects
    (e.g. pathlib.Path inside dataclasses.asdict(config)).

    These checkpoints are produced by this codebase and are assumed trusted, so we load
    with weights_only=False when supported. For older PyTorch versions, fall back.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch (<2.6) does not accept weights_only.
        return torch.load(path, map_location=map_location)


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
    # Enable DDP debug logs *before* initialization so init-time messages are not missed.
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

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


def cleanup_ddp(*, barrier: bool = True):
    """Cleanup distributed process group.

    Args:
        barrier: If True, attempt a barrier before destroy. Use False during
            exception paths to avoid potential hangs.
    """
    if not dist.is_initialized():
        return

    if barrier:
        try:
            dist.barrier()
        except Exception:
            pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


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

        try:
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
        except Exception as e:
            # IMPORTANT: avoid leaving multi-GB tmp dirs around when disk is full / IO fails.
            logging.error(
                "Failed to save checkpoint at step %s (final=%s, tmp=%s): %r",
                global_step,
                final_ckpt_dir,
                tmp_ckpt_dir,
                e,
            )
            try:
                if tmp_ckpt_dir.exists():
                    shutil.rmtree(tmp_ckpt_dir)
            except Exception:
                pass
            raise

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
            optimizer_state_dict = _torch_load_ckpt(optimizer_path, map_location=device)
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
                    s_state = _torch_load_ckpt(s_path, map_location=device)
                    sonata_optimizer.load_state_dict(s_state)
                    logging.info("Loaded Sonata optimizer state from pt format")
                    del s_state
                except Exception as e:
                    logging.warning(f"Failed to load Sonata optimizer state ({e!r}); continuing with fresh.")

        logging.info("Loading metadata...")
        metadata = _torch_load_ckpt(ckpt_dir / "metadata.pt", map_location=device)
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
        # 兼容 'images' 或 'image'；两者都可能是“多视角 dict”
        img_dict = sample_dict.get("images")
        if img_dict is None and "image" in sample_dict:
            if isinstance(sample_dict["image"], dict):
                img_dict = sample_dict["image"]
            else:
                img = sample_dict["image"]
                if not isinstance(img, torch.Tensor):
                    img = torch.as_tensor(img)
                img_dict = {"image": img}
        if img_dict is None:
            raise KeyError("No 'images' or 'image' found in sample batch for logging.")

        images_to_log = []
        first_view = next(iter(img_dict.values()))
        batch_size = first_view.shape[0]
        def _as_hwc(x: torch.Tensor) -> torch.Tensor:
            """Convert either CHW or HWC to HWC for logging."""
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return x
            # CHW -> HWC
            if x.shape[0] in (1, 3, 4):
                return x.permute(1, 2, 0)
            return x  # already HWC

        for i in range(min(5, batch_size)):
            # multi-view horizontal concat in HWC
            nhwcs = [_as_hwc(img[i]) for img in img_dict.values()]
            img_concatenated = torch.cat(nhwcs, dim=1) if len(nhwcs) > 1 else nhwcs[0]
            images_to_log.append(wandb.Image(img_concatenated.detach().cpu().numpy()))

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
            enable_sonata=getattr(config.model, "enable_sonata", True),
            point_feat_dim=getattr(config.model, "point_feat_dim", 6),
            point_token_cap=getattr(config.model, "point_token_cap", 1024),
            sonata_ckpt_path=getattr(config.model, "sonata_ckpt_path", None),
            sonata_projector_ckpt_path=getattr(config.model, "sonata_projector_ckpt_path", None),
            sonata_mode=getattr(config.model, "sonata_mode", None),
            sonata_train_mode=getattr(config.model, "sonata_train_mode", None),
            sonata_validate=getattr(config.model, "sonata_validate", None),
            sonata_auto_pad_feat=getattr(config.model, "sonata_auto_pad_feat", None),
            require_cuda=getattr(config.model, "require_cuda", True),
            point_start_id=getattr(config.model, "point_start_id", None),
            point_end_id=getattr(config.model, "point_end_id", None),

        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # ==== Sonata: set point window token ids and enable fail-fast if mismatch ===
    try:
        if getattr(model, "enable_sonata", False) and getattr(model, "sonata_mode", "all") in ("projector", "all"):
            # Use language model embedding table size to infer special ids (agrees with PaligemmaTokenizer)
            vsz = int(model.paligemma_with_expert.paligemma.language_model.get_input_embeddings().num_embeddings)
            exp_start, exp_end = vsz - 2, vsz - 1
            # If not set, set them; if set but wrong, raise
            if getattr(model, "point_start_id", None) is None or getattr(model, "point_end_id", None) is None:
                model.point_start_id = exp_start
                model.point_end_id   = exp_end
            else:
                if int(model.point_start_id) != exp_start or int(model.point_end_id) != exp_end:
                    raise RuntimeError(
                        f"point_start_id/point_end_id mismatch tokenizer: got ({model.point_start_id},{model.point_end_id}), "
                        f"expected ({exp_start},{exp_end})."
                    )
            logging.info(f"[Sonata] point token ids set to start={model.point_start_id}, end={model.point_end_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to set point token ids for Sonata insertion: {e}") from e

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

    # --- Sonata / pointcloud integration: materialize lazy modules early ---
    # Required for DDP/FSDP correctness (avoid creating parameters after wrapping) and for safetensors.load_model().
    if getattr(model, "enable_sonata", False) and getattr(model, "sonata_mode", "off") in ("projector", "all"):
        # SpatialLM / Sonata encoder default output dim is 512.
        if hasattr(model, "materialize_sonata_projector"):
            model.materialize_sonata_projector(enc_dim=512)
        # If we will train the encoder itself ("all") OR we are about to load a full model checkpoint
        # into this instance, materialize the encoder before wrapping/loading so weights have a landing spot.
        if use_ddp or getattr(model, "sonata_mode", "off") == "all" or resuming or (config.pytorch_weight_path is not None):
            if hasattr(model, "_ensure_sonata_ready"):
                model._ensure_sonata_ready(device=device)

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
        from pathlib import Path
        import safetensors.torch
    
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
    
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
        ckpt_path = Path(config.pytorch_weight_path)
        model_path = ckpt_path / "model.safetensors" if ckpt_path.is_dir() else ckpt_path
        if not model_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint file not found: {model_path}")
    
        load_result = safetensors.torch.load_model(
            model_to_load,
            str(model_path),
            strict=False,
            device=str(device),
        )
    
        if load_result is None:
            missing_keys, unexpected_keys = [], []
        else:
            missing_keys, unexpected_keys = load_result
            missing_keys = list(missing_keys or [])
            unexpected_keys = list(unexpected_keys or [])
    
        allowed_missing_prefixes = (
            "sonata.",
            "pc_projector.",
        )
    
        bad_missing = [k for k in missing_keys if not k.startswith(allowed_missing_prefixes)]
    
        if unexpected_keys:
            raise RuntimeError(
                "Unexpected keys when loading PyTorch base checkpoint:\n"
                + "\n".join(unexpected_keys[:200])
            )
    
        if bad_missing:
            raise RuntimeError(
                "Unexpected missing keys when loading PyTorch base checkpoint.\n"
                "Only sonata.* and pc_projector.* are allowed to be missing here.\n"
                + "\n".join(bad_missing[:200])
            )
    
        logging.info(
            f"Loaded PyTorch base checkpoint from {config.pytorch_weight_path} "
            f"with strict=False. allowed_missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # -------- 优化器：单一 AdamW，惰性模块将动态注册到该优化器 --------
    # Read the Sonata mode from the instantiated model so the trainer and model
    # share one source of truth (config/env conflicts are already resolved there).
    base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    sonata_mode = str(getattr(base_model, "sonata_mode", "off") or "off").lower()
    if sonata_mode not in {"off", "projector", "all"}:
        raise ValueError(f"Invalid SONATA mode: {sonata_mode!r}. Expected one of: off|projector|all")

    # In projector mode, the Sonata encoder is frozen and MUST NOT be included in the optimizer.
    # This matters especially for resume runs where the encoder is materialized before building the optimizer.
    if sonata_mode == "projector":
        enc = getattr(base_model, "sonata", None)
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad_(False)

    trainable_params = [p for p in get_model_parameters(model) if getattr(p, "requires_grad", False)]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer; check configuration.")


    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    sonata_optim = None  # 已取消第二优化器；保持接口占位以兼容 load/save 调用

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, sonata_optimizer=None)
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

    # 惰性模块（Sonata encoder / projector）首次出现时，自动注册到现有 optimizer
    def _maybe_register_new_params(optimizer, wrapped_model, mode: str):
        base_model = wrapped_model.module if isinstance(wrapped_model, torch.nn.parallel.DistributedDataParallel) else wrapped_model
        existing = {id(p) for g in optimizer.param_groups for p in g["params"]}
        new_params = []
        # projector
        proj = getattr(base_model, "pc_projector", None)
        if proj is not None:
            for p in proj.parameters():
                if p.requires_grad and id(p) not in existing:
                    new_params.append(p)
        # encoder
        enc = getattr(base_model, "sonata", None)
        if enc is not None:
            if mode == "projector":
                for p in enc.parameters():
                    if p.requires_grad:
                        p.requires_grad_(False)
            else:
                for p in enc.parameters():
                    if p.requires_grad and id(p) not in existing:
                        new_params.append(p)
        if new_params:
            # IMPORTANT: inherit current hyperparams (especially lr) from existing param group.
            template = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
            template["params"] = new_params
            optimizer.add_param_group(template)
            logging.info(
                "[SONATA] registered %d new parameters into optimizer (lr=%s)",
                len(new_params),
                optimizer.param_groups[-1].get("lr", None),
            )
        return

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

    # DDP per-epoch shuffle:
    # Our DataLoaderImpl / TorchDataLoader is step-infinite (it restarts internally),
    # so we must call set_epoch() at epoch boundaries (every len(loader) steps).
    iters_per_epoch: int | None = None
    if use_ddp and hasattr(loader, "set_epoch"):
        if not hasattr(loader, "__len__"):
            raise RuntimeError("Data loader must define __len__() under DDP.")
        iters_per_epoch = len(loader)
        if iters_per_epoch <= 0:
            raise RuntimeError("Data loader length must be > 0 under DDP.")
        # Align epoch with the current global_step (important when resuming).
        loader.set_epoch(global_step // iters_per_epoch)

    while global_step < config.num_train_steps:

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # Observation 是 dataclass：to_dict → 递归 .to(device) → from_dict
            if not hasattr(observation, "to_dict"):
                raise RuntimeError("Observation must implement to_dict().")
            obs_dict = observation.to_dict()
            # --- numpy -> torch（最小必要，避免 preprocess_pytorch 在 numpy 上出错） ---
            def _to_torch_tree(x):
                if isinstance(x, torch.Tensor):
                    return x
                # numpy / jax array -> torch
                if hasattr(x, "dtype"):
                    s = str(x.dtype)
                    if s.startswith("int"):
                        # 语言 token等整型应为 torch.long（int64）供 embedding 使用
                        return torch.as_tensor(x, dtype=torch.long)
                    if s.startswith("bool"):
                        return torch.as_tensor(x, dtype=torch.bool)
                    return torch.as_tensor(x, dtype=torch.float32)
                return x
            if _dc.is_dataclass(obs_dict) and not isinstance(obs_dict, type):
                obs_dict = _dc.replace(
                    obs_dict, **{f.name: _to_torch_tree(getattr(obs_dict, f.name)) for f in _dc.fields(obs_dict)}
                )
            elif isinstance(obs_dict, dict):
                obs_dict = {k: _to_torch_tree(v) for k, v in obs_dict.items()}
            else:
                obs_dict = _to_torch_tree(obs_dict)


            obs_dict = _to_device_tree(obs_dict, device)
            # --- Sonata: 最小字段映射（fail-fast；只做键名对齐，不造假） ---
            try:
                base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                enable_sonata = bool(getattr(base_model, "enable_sonata", False))
                sonata_mode = str(getattr(base_model, "sonata_mode", "all"))
            except Exception:
                enable_sonata, sonata_mode = True, "all"
            need_points = enable_sonata and (sonata_mode in ("projector", "all"))
            pcs_ok = isinstance(obs_dict.get("point_clouds"), dict) and ("pointcloud" in obs_dict["point_clouds"])
            if not pcs_ok:
                # 兼容旧键：pointcloud / pointcloud_mask → Observation.point_clouds/point_cloud_masks
                pc = obs_dict.pop("pointcloud", None)
                pm = obs_dict.pop("pointcloud_mask", None)
                if pc is not None:
                    if not isinstance(pc, torch.Tensor):
                        pc = torch.as_tensor(pc, dtype=torch.float32, device=device)
                    obs_dict.setdefault("point_clouds", {})["pointcloud"] = pc
                    if pm is None:
                        B = int(pc.shape[0]); pm = torch.ones(B, dtype=torch.bool, device=device)
                    elif not isinstance(pm, torch.Tensor):
                        pm = torch.as_tensor(pm, dtype=torch.bool, device=device)
                    obs_dict.setdefault("point_cloud_masks", {})["pointcloud"] = pm
                elif need_points:
                    raise RuntimeError("Sonata enabled but Observation.point_clouds['pointcloud'] is missing (fail-fast).")
            if not hasattr(_model, "Observation") or not hasattr(_model.Observation, "from_dict"):
                raise RuntimeError("openpi.models.model.Observation.from_dict is required.")
            observation = _model.Observation.from_dict(obs_dict)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=device)  # robust to numpy/torch

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


            # Forward（由模型内部决定是否编码/插窗；训练脚本不越权）
            losses = model(observation, actions)


            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            # 动态注册惰性模块参数（首个前向后可能才创建）
            _maybe_register_new_params(optim, model, sonata_mode)

            loss = losses.mean()

            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping（仅阈值 >0 时启用）
            grad_norm = None
            max_gn = getattr(config.optimizer, "clip_gradient_norm", None)
            if isinstance(max_gn, (int, float)) and max_gn and max_gn > 0:
                params_for_clip = (get_model_parameters(model))
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(max_gn))

            # Optimizer step（若 all 模式则同时 step 第二优化器）
            optim.step()
            optim.zero_grad(set_to_none=True)

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

            if is_main and ((global_step + 1) % config.log_interval == 0):
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
                    f"step={global_step + 1} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step + 1} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step + 1,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step + 1)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism（以完成步数为名）
            save_checkpoint(model, optim, global_step, config, is_main, data_config, sonata_optimizer=None)

            # Update DDP sampler epoch at epoch boundaries so the next internal DataLoader iterator
            # uses a new deterministic shuffle order.
            if use_ddp and iters_per_epoch is not None and (global_step % iters_per_epoch == 0):
                loader.set_epoch(global_step // iters_per_epoch)

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
    try:
        train_loop(config)
    finally:
        # Always attempt to tear down the process group; avoid hanging barrier on exception.
        cleanup_ddp(barrier=(sys.exc_info()[0] is None))


if __name__ == "__main__":
    main()

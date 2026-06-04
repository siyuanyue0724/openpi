from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import types

import torch

try:  # Keep JSON logs on stdout; tqdm writes progress to stderr only.
    import tqdm.auto as _tqdm_auto
except Exception:  # pragma: no cover - optional runtime convenience.
    _tqdm_auto = None

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.picf_core_train import _apply_picf_trainable_scope
from scripts.picf_core_train import _build_loss_config
from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _freeze_recycle_path_parameters
from scripts.picf_core_train import _is_retryable_first_step_error
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _load_tactile_backgrounds_npz
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _resolve_action_normalizer
from scripts.picf_core_train import _seed_everything
from scripts.picf_fixed_window_action_probe import _aggregate
from scripts.picf_fixed_window_action_probe import DEFAULT_SUMMARY_KEYS
from scripts.picf_fixed_window_action_probe import FIXED_WINDOW_SCOPE_WARNING
from scripts.picf_fixed_window_action_probe import _install_action_context_probe_mode
from scripts.picf_fixed_window_action_probe import _load_window_records
from scripts.picf_fixed_window_action_probe import _numeric_snapshot
from scripts.picf_fixed_window_action_probe import _write_jsonl
from scripts.picf_replay_windows import _coerce_loaded_args
from scripts.picf_replay_windows import _resolve_rank_seed
from scripts.sonata_window_probe import _override_build_sample


def _set_override(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _trainable_summary(model: torch.nn.Module) -> dict[str, Any]:
    names: list[str] = []
    numel = 0
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        names.append(str(name))
        if not isinstance(param, torch.nn.parameter.UninitializedParameter):
            numel += int(param.numel())
    return {
        "trainable_param_tensors": len(names),
        "trainable_numel": int(numel),
        "matched_names_sample": names[:48],
    }


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        value = float(torch.sum(grad.detach().to(dtype=torch.float32).square()).item())
        if math.isfinite(value):
            total += value
    return float(math.sqrt(total))


def _mean_recent(rows: list[dict[str, Any]], key: str, count: int) -> float | None:
    values = [
        float(row[key])
        for row in rows[-int(count) :]
        if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, float]:
    keys: set[str] = set()
    for snapshot in snapshots:
        keys.update(str(key) for key in snapshot.keys())
    merged: dict[str, float] = {}
    for key in sorted(keys):
        values = [
            float(snapshot[key])
            for snapshot in snapshots
            if key in snapshot
            and isinstance(snapshot[key], (int, float))
            and math.isfinite(float(snapshot[key]))
        ]
        if values:
            merged[key] = float(sum(values) / len(values))
    return merged


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


class _DeterministicFlowRng:
    """Force deterministic action-flow noise/time sampling for eval passes.

    The PI0/PaliGemma action-flow loss samples both noise and diffusion time
    when they are not supplied.  Training should keep that stochasticity, but
    fixed-window before/after eval must compare the same mathematical object.
    This wrapper resets the RNG per action-flow call, preserving the original
    API while making every eval pass reproducible for a fixed call order.
    """

    def __init__(self, semantic_encoder: Any, *, seed: int, device: torch.device) -> None:
        self.semantic_encoder = semantic_encoder
        self.seed = int(seed)
        self.device = device
        self.call_index = 0
        self._original: Any | None = None

    def __enter__(self) -> "_DeterministicFlowRng":
        if self.semantic_encoder is None or not hasattr(self.semantic_encoder, "compute_action_flow_loss"):
            return self
        self._original = getattr(self.semantic_encoder, "compute_action_flow_loss")

        def _wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            seed = self.seed + int(self.call_index)
            self.call_index += 1
            kwargs = dict(kwargs)
            self._inject_explicit_flow_inputs(_self, kwargs, seed)
            devices: list[int] = []
            if self.device.type == "cuda":
                devices = [int(self.device.index or 0)]
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                return self._original(*args, **kwargs)

        setattr(self.semantic_encoder, "compute_action_flow_loss", types.MethodType(_wrapped, self.semantic_encoder))
        return self

    def _inject_explicit_flow_inputs(self, module: Any, kwargs: dict[str, Any], seed: int) -> None:
        if kwargs.get("noise") is not None and kwargs.get("time") is not None:
            return
        target_raw = kwargs.get("action_chunk_target")
        if target_raw is None:
            return
        base = getattr(module, "encoder", module)
        horizon = int(getattr(base, "action_horizon", 0) or 0)
        model_action_dim = int(getattr(base, "model_action_dim", 0) or 0)
        target = torch.as_tensor(target_raw, device=self.device, dtype=torch.float32)
        if target.ndim == 1:
            target = target[None, :]
        if target.ndim != 2:
            return
        if horizon <= 0:
            horizon = int(target.shape[0])
        if model_action_dim <= 0:
            model_action_dim = int(target.shape[1])
        if kwargs.get("noise") is None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))
            kwargs["noise"] = torch.randn(
                (1, horizon, model_action_dim),
                generator=gen,
                device=self.device,
                dtype=torch.float32,
            )
        if kwargs.get("time") is None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed) + 10_000_019)
            # Exact inverse-CDF sample for Beta(1.5, 1.0), matching the action
            # loss distribution while avoiding hidden RNG state in eval.
            u = torch.rand((1,), generator=gen, device=self.device, dtype=torch.float32)
            kwargs["time"] = (u.pow(1.0 / 1.5) * 0.999 + 0.001).to(device=self.device, dtype=torch.float32)

    def __exit__(self, *exc: object) -> bool:
        if self._original is not None:
            setattr(self.semantic_encoder, "compute_action_flow_loss", self._original)
        return False


class _PicfRuntimeStateGuard:
    """Keep fixed-window eval from mutating recurrent runtime caches.

    Fixed-window probes compare checkpoint parameters, not accumulated clip or
    tactile history from earlier eval windows.  The normal training forward
    intentionally updates these buffers, so eval must explicitly restore them.
    """

    def __init__(self, trainer: _PicfWindowTrainer) -> None:
        self.trainer = trainer
        self.clip_snapshots: dict[str, Any] = {}
        self.tactile_snapshot: Any | None = None
        self.teacher_tokens: torch.Tensor | None = None
        self.teacher_initialized: torch.Tensor | None = None
        self.cached_inference_observed: Any | None = None
        self.cached_inference_step: int | None = None
        self.last_finite_action_chunk: torch.Tensor | None = None

    def __enter__(self) -> "_PicfRuntimeStateGuard":
        core = getattr(self.trainer, "core", None)
        if core is not None:
            clip_buffers = getattr(core, "clip_buffers", None)
            if isinstance(clip_buffers, dict):
                self.clip_snapshots = {
                    str(name): buffer.snapshot()
                    for name, buffer in clip_buffers.items()
                    if hasattr(buffer, "snapshot")
                }
            tactile_buffer = getattr(core, "tactile_buffer", None)
            if tactile_buffer is not None and hasattr(tactile_buffer, "snapshot"):
                self.tactile_snapshot = tactile_buffer.snapshot()
            teacher_tokens = getattr(core, "action_prefix_teacher_tokens", None)
            if isinstance(teacher_tokens, torch.Tensor):
                self.teacher_tokens = teacher_tokens.detach().clone()
            teacher_initialized = getattr(core, "action_prefix_teacher_initialized", None)
            if isinstance(teacher_initialized, torch.Tensor):
                self.teacher_initialized = teacher_initialized.detach().clone()
        policy = getattr(self.trainer, "policy", None)
        if policy is not None:
            self.cached_inference_observed = getattr(policy, "_cached_inference_observed", None)
            self.cached_inference_step = int(getattr(policy, "_cached_inference_step", 0))
            last_chunk = getattr(policy, "_last_finite_action_chunk", None)
            if isinstance(last_chunk, torch.Tensor):
                self.last_finite_action_chunk = last_chunk.detach().clone()
        return self

    def __exit__(self, *exc: object) -> bool:
        core = getattr(self.trainer, "core", None)
        if core is not None:
            clip_buffers = getattr(core, "clip_buffers", None)
            if isinstance(clip_buffers, dict):
                for name, snapshot in self.clip_snapshots.items():
                    buffer = clip_buffers.get(name)
                    if buffer is not None and hasattr(buffer, "restore"):
                        buffer.restore(snapshot)
            tactile_buffer = getattr(core, "tactile_buffer", None)
            if tactile_buffer is not None and self.tactile_snapshot is not None and hasattr(tactile_buffer, "restore"):
                tactile_buffer.restore(self.tactile_snapshot)
            teacher_tokens = getattr(core, "action_prefix_teacher_tokens", None)
            if isinstance(teacher_tokens, torch.Tensor) and self.teacher_tokens is not None:
                teacher_tokens.copy_(self.teacher_tokens.to(device=teacher_tokens.device, dtype=teacher_tokens.dtype))
            teacher_initialized = getattr(core, "action_prefix_teacher_initialized", None)
            if isinstance(teacher_initialized, torch.Tensor) and self.teacher_initialized is not None:
                teacher_initialized.copy_(
                    self.teacher_initialized.to(device=teacher_initialized.device, dtype=teacher_initialized.dtype)
                )
        policy = getattr(self.trainer, "policy", None)
        if policy is not None:
            setattr(policy, "_cached_inference_observed", self.cached_inference_observed)
            if self.cached_inference_step is not None:
                setattr(policy, "_cached_inference_step", int(self.cached_inference_step))
            setattr(policy, "_last_finite_action_chunk", self.last_finite_action_chunk)
        return False


def _reset_picf_runtime_buffers(trainer: _PicfWindowTrainer) -> None:
    core = getattr(trainer, "core", None)
    if core is None:
        return
    clip_buffers = getattr(core, "clip_buffers", None)
    if isinstance(clip_buffers, dict):
        for buffer in clip_buffers.values():
            reset = getattr(buffer, "reset", None)
            if callable(reset):
                reset(segment_id=None)
    tactile_buffer = getattr(core, "tactile_buffer", None)
    reset = getattr(tactile_buffer, "reset", None)
    if callable(reset):
        reset(segment_id=None)


def _evaluate_prepared(
    trainer: _PicfWindowTrainer,
    prepared: list[tuple[int, dict[str, Any], Any, list[int]]],
    *,
    max_windows: int,
    deterministic_seed: int | None,
    device: torch.device,
    step: int,
) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    was_training = bool(trainer.training)
    trainer.eval()
    context = (
        _DeterministicFlowRng(trainer.semantic_encoder, seed=int(deterministic_seed), device=device)
        if deterministic_seed is not None
        else _NullContext()
    )
    with torch.no_grad(), context:
        for eval_idx, eval_record, eval_window, eval_counts in prepared[: int(max_windows)]:
            # Fixed-window eval is a checkpoint-level measurement.  Each window
            # must start from the same recurrent runtime state; otherwise the
            # eval loop itself can create cross-window teacher/cache carryover
            # and make repeated eval summaries disagree.
            with _PicfRuntimeStateGuard(trainer):
                _reset_picf_runtime_buffers(trainer)
                eval_outputs = trainer(eval_window, capture_visual_diagnostics=False)
            eval_rows.append(
                {
                    "stage": "eval_window",
                    "step": int(step),
                    "window_index": int(eval_idx),
                    "segment": int(eval_record["segment"]),
                    "start_step": int(eval_record["start_step"]),
                    "prompt": str(eval_record.get("prompt", "")),
                    "point_counts": eval_counts,
                    **_numeric_snapshot(eval_outputs),
                }
            )
    if was_training:
        trainer.train()
    else:
        trainer.eval()
    return eval_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the action-visible PICF bridge/readout on exact fixed windows. "
            "This is a capacity probe: it tests whether frozen PICF belief contains "
            "action-usable information under a chosen bridge, not global CALVIN quality."
        )
    )
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--rank-seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--windows-per-step",
        type=int,
        default=1,
        help=(
            "Number of exact windows to accumulate before one optimizer step. "
            "Use len(window_jsonl) to test balanced all-window gradients and "
            "separate capacity from sequential-window forgetting."
        ),
    )
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=12)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar on stderr while preserving JSON stdout logs.",
    )
    parser.add_argument(
        "--deterministic-eval-seed",
        type=int,
        default=None,
        help="When set, eval summaries use deterministic flow-loss RNG for same-window before/after comparison.",
    )
    parser.add_argument(
        "--eval-before-after",
        action="store_true",
        help="Emit deterministic eval summaries at step 0 and final step.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--action-loss-key",
        choices=("loss_action", "loss_action_default_equiv", "loss_action_active7"),
        default="loss_action",
        help="Differentiable scalar used for bridge-capacity optimization.",
    )
    parser.add_argument(
        "--action-context-integration",
        choices=("prefix_fusion", "suffix_cross_attention", "append"),
        default="suffix_cross_attention",
    )
    parser.add_argument("--action-context-tokens", type=int, default=24)
    parser.add_argument("--action-context-output-gate", type=float, default=0.25)
    parser.add_argument("--action-context-stopgrad", action="store_true")
    parser.add_argument("--no-action-context-stopgrad", dest="action_context_stopgrad", action="store_false")
    parser.set_defaults(action_context_stopgrad=True)
    parser.add_argument(
        "--action-context-probe-mode",
        choices=("none", "zero", "token_roll", "shuffle", "sign_flip", "rms_noise"),
        default="none",
        help=(
            "Causal perturbation of PICF action-context tokens before they reach "
            "the PI action path. 'shuffle' is accepted as an alias for token_roll. "
            "This is diagnostic-only and leaves the native PI input/action target unchanged."
        ),
    )
    parser.add_argument(
        "--disable-picf-action-condition",
        action="store_true",
        help=(
            "Keep PICF observe/finalize and auxiliary losses available, but do not pass "
            "PICF prefix/context tokens into the PI0.5 action loss. This is cleaner than "
            "gating prefixes to zero, because it preserves the native PI action layout."
        ),
    )
    parser.add_argument(
        "--picf-trainable-scope",
        choices=("policy_only", "all"),
        default="policy_only",
        help="policy_only freezes core.* PICF belief and isolates action bridge/readout capacity.",
    )
    parser.add_argument(
        "--semantic-trainable-scope",
        choices=("action_adapter_only", "action_head_only", "action_head_and_adapter", "backbone_only", "all"),
        default="action_head_and_adapter",
    )
    parser.add_argument("--semantic-lr-scale", type=float, default=1.0)
    parser.add_argument("--policy-head-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--burnin-steps-override",
        type=int,
        default=None,
        help=(
            "Override loaded burnin_steps for exact-window causal controls. "
            "Use this only to make PICF-enabled and ablated probes optimize the same transition indices."
        ),
    )
    parser.add_argument(
        "--picf-core-lr-scale",
        type=float,
        default=None,
        help=(
            "Optional override for PICF core LR scale. Leave unset for frozen-policy bridge probes. "
            "Set explicitly when testing action-gradient access to PICF core."
        ),
    )
    parser.add_argument("--point-grid-mode", choices=("default", "original", "rebased"), default="default")
    parser.add_argument(
        "--picf-mode",
        choices=("loaded", "enabled", "ablated"),
        default="loaded",
        help="Override picf_mode. 'ablated' gives a PI0.5 action-only capacity control on the same windows.",
    )
    args = parser.parse_args()

    if int(args.steps) < 0:
        raise ValueError("--steps must be non-negative.")
    if int(args.steps) == 0 and not bool(args.eval_before_after):
        raise ValueError("--steps=0 is only valid with --eval-before-after for eval-only determinism probes.")
    if int(args.windows_per_step) <= 0:
        raise ValueError("--windows-per-step must be positive.")
    if int(args.log_every) <= 0:
        raise ValueError("--log-every must be positive.")
    if int(args.max_eval_windows) <= 0:
        raise ValueError("--max-eval-windows must be positive.")

    started = time.time()
    args_json_path = Path(args.args_json)
    payload = json.loads(args_json_path.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
    if args.split is not None:
        train_args.split = str(args.split)

    _set_override(train_args, "training_strategy", "ddp")
    _set_override(train_args, "optimizer_sharding", "none")
    _set_override(train_args, "picf_trainable_scope", str(args.picf_trainable_scope))
    _set_override(train_args, "semantic_trainable", True)
    _set_override(train_args, "semantic_trainable_scope", str(args.semantic_trainable_scope))
    _set_override(train_args, "semantic_lr_scale", float(args.semantic_lr_scale))
    _set_override(train_args, "policy_head_lr_scale", float(args.policy_head_lr_scale))
    if args.burnin_steps_override is not None:
        _set_override(train_args, "burnin_steps", int(args.burnin_steps_override))
    if args.picf_core_lr_scale is not None:
        _set_override(train_args, "picf_core_lr_scale", float(args.picf_core_lr_scale))
    _set_override(train_args, "lr", float(args.lr))
    _set_override(train_args, "min_lr", float(args.lr))
    _set_override(train_args, "weight_decay", float(args.weight_decay))
    _set_override(train_args, "action_context_integration", str(args.action_context_integration))
    _set_override(train_args, "action_context_tokens", int(args.action_context_tokens))
    _set_override(train_args, "action_context_output_gate", float(args.action_context_output_gate))
    _set_override(train_args, "action_context_stopgrad", bool(args.action_context_stopgrad))
    _set_override(train_args, "picf_action_condition_enabled", not bool(args.disable_picf_action_condition))
    requested_picf_mode = str(args.picf_mode).strip().lower().replace("-", "_")
    # For the action-only capacity control, build the same full architecture as
    # the checkpoint, load it strictly, then disable PICF at the policy forward
    # boundary.  Overriding args.picf_mode before build can remove frozen
    # foundation submodules from the core and create meaningless checkpoint
    # "unexpected key" failures.
    if requested_picf_mode not in {"loaded", "ablated"}:
        _set_override(train_args, "picf_mode", str(args.picf_mode))
    device = torch.device(str(train_args.device))
    rank_seed = _resolve_rank_seed(rank_seed=args.rank_seed, rng_rank=None)
    _seed_everything(int(train_args.seed), int(rank_seed))

    calvin_segment_indices = None
    if getattr(train_args, "calvin_segment_indices", None):
        calvin_segment_indices = [
            int(part)
            for part in str(getattr(train_args, "calvin_segment_indices")).split(",")
            if part.strip()
        ]
    action_normalizer = _resolve_action_normalizer(train_args)
    override_context = (
        _override_build_sample(str(args.point_grid_mode))
        if str(args.point_grid_mode) != "default"
        else _NullContext()
    )

    with override_context:
        mvtrack_sidecar_root = getattr(train_args, "mvtrack_sidecar_root", None)
        source = _CalvinTransitionSource(
            train_args.calvin_root,
            split=train_args.split,
            backend=train_args.backend,
            unroll_steps=train_args.effective_unroll_steps,
            action_horizon=train_args.action_horizon,
            use_tactile=bool(train_args.use_tactile),
            tactile_sensor_names=train_args.tactile_sensor_names,
            tactile_sensor_offsets_m=train_args.tactile_sensor_offsets_m,
            tactile_calibration=train_args.tactile_calibration_path,
            tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(train_args.tactile_backgrounds_path),
            use_scene_obs=bool(train_args.use_scene_obs),
            load_tracklet_fields=bool(getattr(train_args, "tracklet_memory_enabled", False)),
            load_proposal_fields=bool(getattr(train_args, "proposal_memory_enabled", False)),
            mvtrack_sidecar_root=mvtrack_sidecar_root,
            mvtrack_sidecar_proposal_nearest_max_gap=int(
                getattr(train_args, "mvtrack_sidecar_proposal_nearest_max_gap", 0)
            ),
            action_normalizer=action_normalizer,
            augmentation_mode=train_args.picf_augmentation_mode,
            photometric_strength=train_args.picf_photometric_strength,
            segment_indices=calvin_segment_indices,
        )
        print(
            json.dumps(
                {
                    "stage": "source_ready",
                    "probe_scope_warning": FIXED_WINDOW_SCOPE_WARNING,
                    "dataset_size": int(len(source)),
                    "split": str(train_args.split),
                    "effective_unroll_steps": int(train_args.effective_unroll_steps),
                    "action_horizon": int(train_args.action_horizon),
                    "mvtrack_sidecar_root": None if mvtrack_sidecar_root is None else str(mvtrack_sidecar_root),
                    "elapsed_s": round(time.time() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        core, semantic_encoder, use_visual_override = _build_model(train_args, device=device)
        core = core.to(device)
        trainer = _PicfWindowTrainer(
            core,
            semantic_encoder=semantic_encoder,
            visual_grid=train_args.visual_grid,
            use_visual_override=use_visual_override,
            loss_config=_build_loss_config(train_args),
            picf_mode=str(getattr(train_args, "picf_mode", "enabled")),
            burnin_steps=int(getattr(train_args, "burnin_steps", 0)),
            burnin_mode=str(getattr(train_args, "burnin_mode", "full")),
        ).to(device)
        _materialize_model_parameters(trainer, source=source, rank=int(rank_seed))
        trainable_scope_info = _apply_picf_trainable_scope(trainer, args=train_args, logger=None)
        recycle_freeze_info = (
            _freeze_recycle_path_parameters(trainer)
            if bool(getattr(train_args, "freeze_recycle_path", False))
            else {}
        )
        dummy_optimizer, _ = _build_optimizer(trainer, args=train_args)
        loaded_step = _load_checkpoint(
            path=Path(args.checkpoint),
            model=trainer,
            optimizer=dummy_optimizer,
            device=device,
            optimizer_checkpoint_mode="model_only",
        )
        if requested_picf_mode == "ablated":
            trainer.picf_mode = "ablated"
            trainer.policy.picf_enabled = False
            _set_override(train_args, "picf_mode", "ablated")
        action_context_probe_mode = str(args.action_context_probe_mode).strip().lower().replace("-", "_")
        if action_context_probe_mode == "shuffle":
            action_context_probe_mode = "token_roll"
        _install_action_context_probe_mode(trainer, mode=action_context_probe_mode)
        # The probe deliberately starts with fresh Adam moments so it measures
        # local bridge capacity, not optimizer-state continuity.
        optimizer, optimizer_groups = _build_optimizer(trainer, args=train_args)
        trainer.train()
        window_records = _load_window_records(Path(args.window_jsonl))
        prepared: list[tuple[int, dict[str, Any], Any, list[int]]] = []
        for idx, record in enumerate(window_records):
            window = source.window_from_metadata(
                segment_id=int(record["segment"]),
                start_step_id=int(record["start_step"]),
            )
            try:
                point_counts = _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
            except RuntimeError as exc:
                if _is_retryable_first_step_error(exc):
                    print(
                        json.dumps(
                            {
                                "stage": "window_skipped",
                                "index": int(idx),
                                "segment": int(record["segment"]),
                                "start_step": int(record["start_step"]),
                                "error": str(exc),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                raise
            prepared.append((idx, record, window, [int(count) for count in point_counts]))
        if not prepared:
            raise RuntimeError("No valid windows prepared for bridge-capacity training.")

        print(
            json.dumps(
                {
                    "stage": "model_ready",
                    "checkpoint": str(args.checkpoint),
                    "loaded_step": int(loaded_step),
                    "action_loss_key": str(args.action_loss_key),
                    "action_context_integration": str(args.action_context_integration),
                    "action_context_tokens": int(args.action_context_tokens),
                    "action_context_stopgrad": bool(args.action_context_stopgrad),
                    "action_context_probe_mode": str(action_context_probe_mode),
                    "picf_mode": str(getattr(train_args, "picf_mode", "enabled")),
                    "picf_trainable_scope": str(args.picf_trainable_scope),
                    "semantic_trainable_scope": str(args.semantic_trainable_scope),
                    "windows_per_step": int(args.windows_per_step),
                    "trainable_scope": trainable_scope_info,
                    "trainable_after_load": _trainable_summary(trainer),
                    "optimizer_groups": optimizer_groups,
                    "recycle_freeze": recycle_freeze_info,
                    "valid_windows": int(len(prepared)),
                    "elapsed_s": round(time.time() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        eval_history: list[dict[str, Any]] = []
        if bool(args.eval_before_after):
            eval_rows = _evaluate_prepared(
                trainer,
                prepared,
                max_windows=int(args.max_eval_windows),
                deterministic_seed=args.deterministic_eval_seed,
                device=device,
                step=0,
            )
            eval_history.extend(eval_rows)
            eval_summary = {
                "stage": "eval_summary",
                "step": 0,
                "deterministic_eval_seed": args.deterministic_eval_seed,
                "metrics": _aggregate(eval_rows, DEFAULT_SUMMARY_KEYS),
                "elapsed_s": round(time.time() - started, 3),
            }
            print(json.dumps(eval_summary, sort_keys=True), flush=True)

        rows: list[dict[str, Any]] = []
        windows_per_step = int(args.windows_per_step)
        pbar = None
        if bool(args.progress):
            if _tqdm_auto is None:
                print(
                    json.dumps({"stage": "progress_unavailable", "reason": "tqdm import failed"}, sort_keys=True),
                    flush=True,
                )
            else:
                pbar = _tqdm_auto.tqdm(
                    total=int(args.steps),
                    desc=f"PICF exact probe K={windows_per_step}",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
        for step in range(1, int(args.steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            micro_snapshots: list[dict[str, Any]] = []
            micro_losses: list[float] = []
            micro_window_indices: list[int] = []
            micro_segments: list[int] = []
            micro_prompts: list[str] = []
            micro_point_counts: list[list[int]] = []
            for micro_idx in range(windows_per_step):
                prepared_index = ((step - 1) * windows_per_step + micro_idx) % len(prepared)
                window_index, record, window, point_counts = prepared[prepared_index]
                outputs = trainer(window, capture_visual_diagnostics=False)
                loss = outputs.get(str(args.action_loss_key))
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise RuntimeError(f"Missing scalar action loss key: {args.action_loss_key}.")
                if not bool(loss.requires_grad):
                    raise RuntimeError(
                        f"{args.action_loss_key} does not require grad; trainable scope likely disconnected."
                    )
                (loss / float(windows_per_step)).backward()
                micro_losses.append(float(loss.detach().item()))
                micro_snapshots.append(_numeric_snapshot(outputs))
                micro_window_indices.append(int(window_index))
                micro_segments.append(int(record["segment"]))
                micro_prompts.append(str(record.get("prompt", "")))
                micro_point_counts.append(point_counts)
            grad_norm_pre = _grad_norm(trainer)
            if float(args.clip_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in trainer.parameters() if bool(getattr(param, "requires_grad", False))],
                    max_norm=float(args.clip_grad_norm),
                )
            grad_norm_post = _grad_norm(trainer)
            optimizer.step()
            snapshot = _mean_snapshots(micro_snapshots)
            row = {
                "stage": "train_step",
                "step": int(step),
                "loaded_step": int(loaded_step),
                "window_count": int(windows_per_step),
                "window_indices": micro_window_indices,
                "segments": micro_segments,
                "prompts": micro_prompts,
                "point_counts": micro_point_counts,
                "optimized_loss": float(sum(micro_losses) / len(micro_losses)),
                "grad_norm_pre_clip": float(grad_norm_pre),
                "grad_norm_post_clip": float(grad_norm_post),
                "elapsed_s": round(time.time() - started, 3),
                **snapshot,
            }
            rows.append(row)
            if step == 1 or step % int(args.log_every) == 0:
                log_row = {
                    **row,
                    "recent_loss_action_default_equiv": _mean_recent(rows, "loss_action_default_equiv", int(args.log_every)),
                    "recent_optimized_loss": _mean_recent(rows, "optimized_loss", int(args.log_every)),
                }
                print(json.dumps(log_row, sort_keys=True), flush=True)

            if pbar is not None:
                pbar.set_postfix(
                    {
                        "step": int(step),
                        "loss": f"{float(row.get(str(args.action_loss_key), row['optimized_loss'])):.5f}",
                        "recent": f"{float(_mean_recent(rows, str(args.action_loss_key), min(10, len(rows))) or row['optimized_loss']):.5f}",
                    }
                )
                pbar.update(1)

            if int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
                eval_rows = _evaluate_prepared(
                    trainer,
                    prepared,
                    max_windows=int(args.max_eval_windows),
                    deterministic_seed=args.deterministic_eval_seed,
                    device=device,
                    step=int(step),
                )
                eval_history.extend(eval_rows)
                eval_summary = {
                    "stage": "eval_summary",
                    "step": int(step),
                    "deterministic_eval_seed": args.deterministic_eval_seed,
                    "metrics": _aggregate(eval_rows, DEFAULT_SUMMARY_KEYS),
                    "elapsed_s": round(time.time() - started, 3),
                }
                print(json.dumps(eval_summary, sort_keys=True), flush=True)

        if pbar is not None:
            pbar.close()

        if bool(args.eval_before_after):
            eval_rows = _evaluate_prepared(
                trainer,
                prepared,
                max_windows=int(args.max_eval_windows),
                deterministic_seed=args.deterministic_eval_seed,
                device=device,
                step=int(args.steps),
            )
            eval_history.extend(eval_rows)
            eval_summary = {
                "stage": "eval_summary",
                "step": int(args.steps),
                "deterministic_eval_seed": args.deterministic_eval_seed,
                "metrics": _aggregate(eval_rows, DEFAULT_SUMMARY_KEYS),
                "elapsed_s": round(time.time() - started, 3),
            }
            print(json.dumps(eval_summary, sort_keys=True), flush=True)

        summary = {
            "stage": "bridge_capacity_probe_complete",
            "probe_scope": "hard_diagnostic_not_global_metric",
            "probe_scope_warning": FIXED_WINDOW_SCOPE_WARNING,
            "args_json": str(args.args_json),
            "checkpoint": str(args.checkpoint),
            "loaded_step": int(loaded_step),
            "steps": int(args.steps),
            "valid_windows": int(len(prepared)),
            "action_loss_key": str(args.action_loss_key),
            "action_context_integration": str(args.action_context_integration),
            "action_context_tokens": int(args.action_context_tokens),
            "action_context_probe_mode": str(action_context_probe_mode),
            "picf_trainable_scope": str(args.picf_trainable_scope),
            "picf_mode": str(getattr(train_args, "picf_mode", "enabled")),
            "semantic_trainable_scope": str(args.semantic_trainable_scope),
            "windows_per_step": int(args.windows_per_step),
            "metrics": _aggregate(rows, DEFAULT_SUMMARY_KEYS + ("optimized_loss", "grad_norm_pre_clip", "grad_norm_post_clip")),
            "eval_metrics": _aggregate(eval_history, DEFAULT_SUMMARY_KEYS) if eval_history else {},
            "first10_action_default_equiv": _mean_recent(rows[:10], "loss_action_default_equiv", 10),
            "last10_action_default_equiv": _mean_recent(rows, "loss_action_default_equiv", 10),
            "elapsed_s": round(time.time() - started, 3),
        }
        output_jsonl = Path(args.output_jsonl)
        summary_json = Path(args.summary_json)
        _write_jsonl(output_jsonl, rows)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    # The probe must evaluate the checked-out worktree, not any `openpi`
    # package already installed in the training venv.  This matters for
    # same-window causal comparisons because stale config/code can silently
    # change the PICF/action interface under test.
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_loss_config
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _apply_picf_trainable_scope
from scripts.picf_core_train import _freeze_recycle_path_parameters
from scripts.picf_core_train import _is_retryable_first_step_error
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _load_tactile_backgrounds_npz
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _resolve_action_normalizer
from scripts.picf_core_train import _seed_everything
from scripts.picf_replay_windows import _coerce_loaded_args
from scripts.picf_replay_windows import _parse_flat_indices
from scripts.picf_replay_windows import _resolve_rank_seed
from scripts.sonata_window_probe import _override_build_sample


DEFAULT_SUMMARY_KEYS = (
    "loss_total",
    "loss_action",
    "loss_action_default_equiv",
    "loss_action_active7",
    "loss_action_pos",
    "loss_action_rot",
    "loss_action_gripper",
    "loss_total_minus_action",
    "loss_alignment",
    "loss_anchor_pv",
    "loss_anchor_object_pull",
    "loss_pv_weak",
    "loss_mapg_routing",
    "loss_mapg_support_diversity",
    "loss_mapg_geometry_diversity",
    "loss_aqr_denoising",
    "loss_slot_jepa",
    "loss_binding_consistency",
    "aqr_same_role_support_overlap_max",
    "aqr_active_same_role_support_overlap_max",
    "aqr_downstream_same_role_support_overlap_max",
    "aqr_context_same_role_support_overlap_max",
    "aqr_reserve_same_role_support_overlap_max",
    "posterior_active_file_fraction",
    "posterior_active_file_recycle_rate",
    "posterior_active_file_potential_swap_rate",
    "posterior_active_file_calibrated_potential_swap_rate",
    "posterior_recycle_rate",
    "posterior_address_update_rate_mean",
    "tactile_contact_prob_mean",
    "tactile_active_rate",
    "pi_prefix_gate_mean",
    "pi_prefix_rms_mean",
    "pi_prefix_pre_rms_mean",
    "pi_prefix_post_rms_mean",
    "pi_prefix_scale_mean",
    "pi_action_condition_token_count",
    "pi_context_token_count",
    "pi_context_gate",
    "pi_context_post_rms_mean",
    "pi_context_attention_entropy_mean",
    "pi_context_fused_prefix_token_count",
    "pi_context_fused_post_rms_mean",
    "pi_context_adapter_token_count",
    "pi_context_adapter_gate",
    "pi_context_adapter_attention_entropy_mean",
    "pi_context_adapter_residual_rms_mean",
    "pi_context_probe_mode_id",
    "pi_context_probe_delta_rms_mean",
    "pi_context_probe_post_rms_mean",
    "pi_prefix_probe_mode_id",
    "pi_prefix_probe_delta_rms_mean",
    "pi_prefix_probe_post_rms_mean",
)

REQUIRED_ACTION_PROBE_KEYS = (
    "loss_action_default_equiv",
    "loss_action_active7",
    "loss_total",
    "loss_total_minus_action",
)

FIXED_WINDOW_SCOPE_WARNING = (
    "hard_diagnostic_not_global_metric: this script is a stationary no-update "
    "checkpoint replay on a fixed window set. It is valid for same-window "
    "regression and causal probes only. It must not be reported as full CALVIN "
    "or full-training-set action quality without an explicit stratified sampling "
    "contract."
)


def _load_flat_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(value) for value in payload]
    if isinstance(payload, dict):
        for key in ("flat_indices", "accepted_flat_indices"):
            if key in payload:
                return [int(value) for value in payload[key]]
    raise ValueError(f"Could not find flat_indices in {path}.")


def _load_window_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Window record file is empty: {path}.")
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        rows = payload
    else:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Window record {index} in {path} is not an object.")
        segment = row.get("segment", row.get("segment_id"))
        start_step = row.get("start_step", row.get("start_step_id"))
        if segment is None or start_step is None:
            raise ValueError(
                f"Window record {index} in {path} must contain segment/segment_id and start_step/start_step_id."
            )
        records.append(
            {
                **row,
                "segment": int(segment),
                "start_step": int(start_step),
            }
        )
    if not records:
        raise ValueError(f"No window records loaded from {path}.")
    return records


def _generate_candidate_indices(
    *,
    dataset_size: int,
    seed: int,
    count: int,
    offset: int,
) -> list[int]:
    if int(count) < 1:
        raise ValueError(f"--num-windows must be >= 1, got {count}.")
    if int(offset) < 0:
        raise ValueError(f"--index-offset must be >= 0, got {offset}.")
    if int(dataset_size) < 1:
        raise ValueError(f"Dataset is empty: {dataset_size}.")
    rng = np.random.default_rng(int(seed))
    if int(offset) > 0:
        rng.integers(0, int(dataset_size), size=int(offset))
    # Oversample so retryable empty-support windows do not change the requested
    # accepted count. The loop below still extends if this conservative pool is
    # exhausted.
    return [int(value) for value in rng.integers(0, int(dataset_size), size=max(int(count) * 3, int(count) + 32))]


def _numeric_snapshot(outputs: dict[str, Any]) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalar = float(value.detach().item())
            if math.isfinite(scalar):
                snapshot[str(key)] = scalar
                snapshot[f"{key}_finite"] = 1.0
            else:
                # Do not silently drop non-finite scalar losses.  Action
                # platform probes are invalid if action/total becomes NaN, and
                # treating that as a missing metric creates false causal
                # conclusions.
                snapshot[f"{key}_finite"] = 0.0
                snapshot[f"{key}_nonfinite"] = 1.0
        elif isinstance(value, (int, float)):
            scalar = float(value)
            if math.isfinite(scalar):
                snapshot[str(key)] = scalar
                snapshot[f"{key}_finite"] = 1.0
            else:
                snapshot[f"{key}_finite"] = 0.0
                snapshot[f"{key}_nonfinite"] = 1.0
    return snapshot


def _aggregate(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(record[key])
            for record in records
            if key in record and isinstance(record[key], (int, float)) and math.isfinite(float(record[key]))
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": int(arr.size),
        }
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _install_action_context_probe_mode(
    trainer: _PicfWindowTrainer,
    *,
    mode: str,
) -> None:
    """Patch the action-visible PICF context only, leaving model weights intact.

    This makes the fixed-window probe a causal readout test:
    the checkpoint, sampled windows, diffusion target construction, and PI0.5
    action path stay fixed while the PICF context delivered to action is
    perturbed.  If action loss/prediction is insensitive to these perturbations,
    the bottleneck is action readout rather than another object-slot loss.
    """

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "normal", "none", "off"}:
        return
    valid = {"zero", "token_roll", "sign_flip", "rms_noise"}
    if normalized not in valid:
        raise ValueError(f"Unsupported --action-context-probe-mode={mode!r}; expected one of {sorted(valid)}.")

    policy = getattr(trainer, "policy", None)
    if policy is None or not hasattr(policy, "_action_context_tokens"):
        raise RuntimeError("Trainer policy does not expose _action_context_tokens; cannot install context probe.")

    original = policy._action_context_tokens
    mode_id = {
        "zero": 1.0,
        "token_roll": 2.0,
        "sign_flip": 3.0,
        "rms_noise": 4.0,
    }[normalized]
    call_index = 0

    def _patched_action_context_tokens(conditioned_control: Any, *, safety: dict[str, float] | None = None):
        nonlocal call_index
        context, metrics = original(conditioned_control, safety=safety)
        if context is None or not isinstance(context, torch.Tensor) or context.numel() == 0:
            return context, metrics
        before = context
        if normalized == "zero":
            after = torch.zeros_like(before)
        elif normalized == "token_roll":
            after = torch.roll(before, shifts=1, dims=0) if before.shape[0] > 1 else before.clone()
        elif normalized == "sign_flip":
            after = -before
        else:
            generator = torch.Generator(device=before.device)
            generator.manual_seed(20260601 + int(call_index))
            noise = torch.randn(before.shape, device=before.device, dtype=before.dtype, generator=generator)
            eps = 1.0e-6
            before_rms = torch.sqrt(torch.mean(before.detach().to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            noise_rms = torch.sqrt(torch.mean(noise.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            after = noise * (before_rms / torch.clamp(noise_rms, min=eps)).to(device=before.device, dtype=before.dtype)
        call_index += 1

        delta_rms = torch.sqrt(torch.mean((after.detach() - before.detach()).to(dtype=torch.float32).square()))
        post_rms = torch.sqrt(torch.mean(after.detach().to(dtype=torch.float32).square()))
        zero = after.reshape(-1)[0].detach() * 0.0
        metrics = dict(metrics)
        metrics["picf_action_context_probe_mode_id"] = zero + float(mode_id)
        metrics["picf_action_context_probe_delta_rms_mean"] = delta_rms.to(device=after.device, dtype=after.dtype)
        metrics["picf_action_context_probe_post_rms_mean"] = post_rms.to(device=after.device, dtype=after.dtype)
        if safety is not None:
            safety["action_context_probe_mode_id"] = float(mode_id)
            safety["action_context_probe_delta_rms_mean"] = float(delta_rms.item())
            safety["action_context_probe_post_rms_mean"] = float(post_rms.item())
        return after, metrics

    policy._action_context_tokens = _patched_action_context_tokens


def _install_action_prefix_probe_mode(
    trainer: _PicfWindowTrainer,
    *,
    mode: str,
) -> None:
    """Patch final PICF PI-prefix tokens for a causal bridge test.

    Context perturbation tests whether the optional dense context bridge is
    useful.  Prefix perturbation tests the stronger claim: whether the action
    generator is using the final four PICF PI-prefix tokens at all.
    """

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "normal", "none", "off"}:
        return
    valid = {"zero", "token_roll", "sign_flip", "rms_noise"}
    if normalized not in valid:
        raise ValueError(f"Unsupported --action-prefix-probe-mode={mode!r}; expected one of {sorted(valid)}.")

    policy = getattr(trainer, "policy", None)
    if policy is None or not hasattr(policy, "_training_action_prefix_tokens"):
        raise RuntimeError("Trainer policy does not expose _training_action_prefix_tokens; cannot install prefix probe.")

    original = policy._training_action_prefix_tokens
    mode_id = {
        "zero": 1.0,
        "token_roll": 2.0,
        "sign_flip": 3.0,
        "rms_noise": 4.0,
    }[normalized]
    call_index = 0

    def _patched_training_action_prefix_tokens(tokens: torch.Tensor):
        nonlocal call_index
        prefix, metrics = original(tokens)
        if not isinstance(prefix, torch.Tensor) or prefix.numel() == 0:
            return prefix, metrics
        before = prefix
        if normalized == "zero":
            after = torch.zeros_like(before)
        elif normalized == "token_roll":
            after = torch.roll(before, shifts=1, dims=0) if before.shape[0] > 1 else before.clone()
        elif normalized == "sign_flip":
            after = -before
        else:
            generator = torch.Generator(device=before.device)
            generator.manual_seed(20260611 + int(call_index))
            noise = torch.randn(before.shape, device=before.device, dtype=before.dtype, generator=generator)
            eps = 1.0e-6
            before_rms = torch.sqrt(torch.mean(before.detach().to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            noise_rms = torch.sqrt(torch.mean(noise.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            after = noise * (before_rms / torch.clamp(noise_rms, min=eps)).to(device=before.device, dtype=before.dtype)
        call_index += 1

        delta_rms = torch.sqrt(torch.mean((after.detach() - before.detach()).to(dtype=torch.float32).square()))
        post_rms = torch.sqrt(torch.mean(after.detach().to(dtype=torch.float32).square()))
        zero = after.reshape(-1)[0].detach() * 0.0
        metrics = dict(metrics)
        metrics["picf_action_prefix_probe_mode_id"] = zero + float(mode_id)
        metrics["picf_action_prefix_probe_delta_rms_mean"] = delta_rms.to(device=after.device, dtype=after.dtype)
        metrics["picf_action_prefix_probe_post_rms_mean"] = post_rms.to(device=after.device, dtype=after.dtype)
        return after, metrics

    policy._training_action_prefix_tokens = _patched_training_action_prefix_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PICF checkpoints on a fixed CALVIN window set without "
            "optimizer/backward updates. This is the stationary action probe "
            "for comparing old and step-indexed runs."
        )
    )
    parser.add_argument("--args-json", required=True, help="Path to args.json from the training run.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or latest.pt to evaluate.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--split", default=None, help="Override data split from args.json, e.g. validation.")
    parser.add_argument("--flat-indices", default=None, help="Comma-separated fixed flat indices.")
    parser.add_argument("--flat-index-json", default=None, help="JSON file containing flat_indices/accepted_flat_indices.")
    parser.add_argument(
        "--window-jsonl",
        default=None,
        help=(
            "JSONL/JSON list with explicit segment/start_step records. This is the exact-window mode; "
            "unlike flat-index mode it does not fall back to the segment first valid start."
        ),
    )
    parser.add_argument("--num-windows", type=int, default=None, help="Generate this many accepted fixed windows.")
    parser.add_argument("--index-seed", type=int, default=20260529, help="Seed for fixed-window index generation.")
    parser.add_argument("--index-offset", type=int, default=0, help="Skip this many generated candidate indices.")
    parser.add_argument("--rank-seed", type=int, default=None, help="RNG rank seed. Defaults to 1.")
    parser.add_argument(
        "--mode",
        choices=("eval", "train"),
        default="eval",
        help="Forward mode. eval is default for stationary validation; train matches dropout/training behavior.",
    )
    parser.add_argument(
        "--point-grid-mode",
        choices=("default", "original", "rebased"),
        default="default",
        help="Override Sonata local-grid preprocessing during the probe.",
    )
    parser.add_argument("--output-jsonl", default=None, help="Optional per-window metrics JSONL path.")
    parser.add_argument("--summary-json", default=None, help="Optional aggregate summary JSON path.")
    parser.add_argument("--dump-flat-indices", default=None, help="Optional path to write accepted flat indices.")
    parser.add_argument(
        "--allow-fixed-window-global-claim",
        action="store_true",
        help=(
            "Explicitly mark this probe as a global/representative claim. "
            "Do not use this for the known step8000 fixed64 hard diagnostic set. "
            "It exists only so future callers must make representativeness explicit."
        ),
    )
    parser.add_argument(
        "--allow-missing-action-metrics",
        action="store_true",
        help=(
            "Allow the probe to complete even if loss_action_default_equiv is absent. "
            "This should only be used for explicit structure-only diagnostics; action "
            "platform/root-cause probes must fail on missing action metrics."
        ),
    )
    parser.add_argument(
        "--action-context-probe-mode",
        choices=("normal", "zero", "token_roll", "sign_flip", "rms_noise"),
        default="normal",
        help=(
            "Causal action-readout perturbation applied only to PICF action context. "
            "normal preserves existing behavior; zero/token_roll/sign_flip/rms_noise "
            "test whether the action path actually uses PICF context."
        ),
    )
    parser.add_argument(
        "--action-prefix-probe-mode",
        choices=("normal", "zero", "token_roll", "sign_flip", "rms_noise"),
        default="normal",
        help=(
            "Causal action-readout perturbation applied to final PICF PI-prefix tokens. "
            "Use this after context probes: if prefix perturbations are also neutral, "
            "the action path is ignoring the full PICF bridge."
        ),
    )
    args = parser.parse_args()

    if sum(value is not None for value in (args.flat_indices, args.flat_index_json, args.window_jsonl, args.num_windows)) != 1:
        raise ValueError("Pass exactly one of --flat-indices, --flat-index-json, --window-jsonl, or --num-windows.")

    args_json_path = Path(args.args_json)
    payload = json.loads(args_json_path.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
    if args.split is not None:
        train_args.split = str(args.split)
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
        else torch.no_grad()  # placeholder context; replaced below by ExitStack-free simple branching
    )
    if str(args.point_grid_mode) == "default":
        class _NullContext:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *exc: object) -> bool:
                return False

        override_context = _NullContext()

    start_time = time.time()
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
        try:
            print(
                json.dumps(
                    {
                        "stage": "source_ready",
                        "probe_scope_warning": FIXED_WINDOW_SCOPE_WARNING,
                        "allow_fixed_window_global_claim": bool(args.allow_fixed_window_global_claim),
                        "split": str(train_args.split),
                        "dataset_size": int(len(source)),
                        "effective_unroll_steps": int(train_args.effective_unroll_steps),
                        "action_horizon": int(train_args.action_horizon),
                        "segment_indices_count": (
                            0 if calvin_segment_indices is None else int(len(calvin_segment_indices))
                        ),
                        "mvtrack_sidecar_root": (
                            None if mvtrack_sidecar_root is None else str(mvtrack_sidecar_root)
                        ),
                        "elapsed_s": round(time.time() - start_time, 3),
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
            _install_action_context_probe_mode(
                trainer,
                mode=str(args.action_context_probe_mode),
            )
            _install_action_prefix_probe_mode(
                trainer,
                mode=str(args.action_prefix_probe_mode),
            )
            _materialize_model_parameters(trainer, source=source, rank=int(rank_seed))
            trainable_scope_info = _apply_picf_trainable_scope(trainer, args=train_args, logger=None)
            recycle_freeze_info = (
                _freeze_recycle_path_parameters(trainer)
                if bool(getattr(train_args, "freeze_recycle_path", False))
                else {}
            )
            optimizer, _ = _build_optimizer(trainer, args=train_args)
            loaded_step = _load_checkpoint(path=Path(args.checkpoint), model=trainer, optimizer=optimizer, device=device)
            if str(args.mode) == "eval":
                trainer.eval()
            else:
                trainer.train()
            print(
                json.dumps(
                    {
                        "stage": "model_ready",
                        "probe_scope_warning": FIXED_WINDOW_SCOPE_WARNING,
                        "allow_fixed_window_global_claim": bool(args.allow_fixed_window_global_claim),
                        "checkpoint": str(args.checkpoint),
                        "loaded_step": int(loaded_step),
                        "mode": str(args.mode),
                        "action_context_probe_mode": str(args.action_context_probe_mode),
                        "action_prefix_probe_mode": str(args.action_prefix_probe_mode),
                        "trainable_scope": trainable_scope_info,
                        "recycle_freeze": recycle_freeze_info,
                        "elapsed_s": round(time.time() - start_time, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            window_records: list[dict[str, Any]] | None = None
            if args.window_jsonl is not None:
                window_records = _load_window_records(Path(args.window_jsonl))
                candidate_indices = list(range(len(window_records)))
                target_count = len(window_records)
                fixed_input = True
            elif args.flat_indices is not None:
                candidate_indices = _parse_flat_indices(str(args.flat_indices))
                target_count = len(candidate_indices)
                fixed_input = True
            elif args.flat_index_json is not None:
                candidate_indices = _load_flat_indices(Path(args.flat_index_json))
                target_count = len(candidate_indices)
                fixed_input = True
            else:
                target_count = int(args.num_windows)
                candidate_indices = _generate_candidate_indices(
                    dataset_size=len(source),
                    seed=int(args.index_seed),
                    count=target_count,
                    offset=int(args.index_offset),
                )
                fixed_input = False

            accepted_indices: list[int] = []
            records: list[dict[str, Any]] = []
            retryable_skips: list[dict[str, Any]] = []
            candidate_cursor = 0
            rng = np.random.default_rng(int(args.index_seed) + 1009)
            while len(accepted_indices) < target_count:
                if candidate_cursor >= len(candidate_indices):
                    if fixed_input:
                        break
                    candidate_indices.extend(
                        int(value)
                        for value in rng.integers(0, int(len(source)), size=max(64, target_count))
                    )
                flat_index = int(candidate_indices[candidate_cursor])
                candidate_cursor += 1
                explicit_record = None if window_records is None else window_records[flat_index]
                if explicit_record is None:
                    window = source.window(flat_index)
                else:
                    window = source.window_from_metadata(
                        segment_id=int(explicit_record["segment"]),
                        start_step_id=int(explicit_record["start_step"]),
                    )
                base_record: dict[str, Any] = {
                    "probe_step": int(len(accepted_indices) + 1),
                    "flat_index": int(flat_index),
                    "segment": int(window.segment_id),
                    "start_step": int(window.start_step_id),
                    "prompt": str(window.prompt),
                }
                if explicit_record is not None:
                    for key in (
                        "source_step",
                        "rank",
                        "micro_step",
                        "resume_step",
                        "local_step",
                        "task",
                        "bucket",
                        "bucket_request",
                    ):
                        if key in explicit_record:
                            base_record[key] = explicit_record[key]
                try:
                    point_counts = _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
                except RuntimeError as exc:
                    if not _is_retryable_first_step_error(exc):
                        raise
                    retry_record = {**base_record, "status": "retryable_window_skipped", "error": str(exc)}
                    retryable_skips.append(retry_record)
                    print(json.dumps(retry_record, sort_keys=True), flush=True)
                    continue
                with torch.no_grad():
                    outputs = trainer(window, capture_visual_diagnostics=False)
                record = {
                    **base_record,
                    "status": "accepted",
                    "point_counts": [int(count) for count in point_counts],
                    **_numeric_snapshot(outputs),
                }
                accepted_indices.append(int(flat_index))
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

            summary = {
                "stage": "fixed_window_probe_complete",
                "probe_scope": "hard_diagnostic_not_global_metric",
                "probe_scope_warning": FIXED_WINDOW_SCOPE_WARNING,
                "allow_fixed_window_global_claim": bool(args.allow_fixed_window_global_claim),
                "args_json": str(args.args_json),
                "checkpoint": str(args.checkpoint),
                "loaded_step": int(loaded_step),
                "mode": str(args.mode),
                "action_context_probe_mode": str(args.action_context_probe_mode),
                "action_prefix_probe_mode": str(args.action_prefix_probe_mode),
                "split": str(train_args.split),
                "dataset_size": int(len(source)),
                "effective_unroll_steps": int(train_args.effective_unroll_steps),
                "action_horizon": int(train_args.action_horizon),
                "segment_indices_count": (
                    0 if calvin_segment_indices is None else int(len(calvin_segment_indices))
                ),
                "mvtrack_sidecar_root": (
                    None if mvtrack_sidecar_root is None else str(mvtrack_sidecar_root)
                ),
                "accepted_windows": int(len(records)),
                "retryable_skip_count": int(len(retryable_skips)),
                "index_seed": int(args.index_seed),
                "index_offset": int(args.index_offset),
                "flat_indices": accepted_indices,
                "retryable_skips": retryable_skips[:128],
                "metrics": _aggregate(records, DEFAULT_SUMMARY_KEYS),
                "elapsed_s": round(time.time() - start_time, 3),
            }
            if len(records) != target_count:
                summary["status"] = "incomplete"
                summary["target_windows"] = int(target_count)
            else:
                summary["status"] = "ok"
            missing_required_metrics = [
                key for key in REQUIRED_ACTION_PROBE_KEYS if key not in summary["metrics"]
            ]
            nonfinite_required_metrics = [
                key
                for key in REQUIRED_ACTION_PROBE_KEYS
                if any(float(record.get(f"{key}_nonfinite", 0.0)) > 0.0 for record in records)
            ]
            if missing_required_metrics:
                summary["status"] = (
                    f"{summary['status']}_missing_action_metrics"
                    if summary["status"] != "ok"
                    else "missing_action_metrics"
                )
                summary["missing_required_metrics"] = missing_required_metrics
            if nonfinite_required_metrics:
                summary["status"] = (
                    f"{summary['status']}_nonfinite_action_metrics"
                    if summary["status"] != "ok"
                    else "nonfinite_action_metrics"
                )
                summary["nonfinite_required_metrics"] = nonfinite_required_metrics
                summary["nonfinite_required_metric_counts"] = {
                    key: int(
                        sum(
                            1
                            for record in records
                            if float(record.get(f"{key}_nonfinite", 0.0)) > 0.0
                        )
                    )
                    for key in nonfinite_required_metrics
                }
            if args.output_jsonl is not None:
                _write_jsonl(Path(args.output_jsonl), records)
            if args.summary_json is not None:
                summary_path = Path(args.summary_json)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            if args.dump_flat_indices is not None:
                dump_path = Path(args.dump_flat_indices)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                dump_path.write_text(
                    json.dumps({"flat_indices": accepted_indices}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            print(json.dumps(summary, sort_keys=True), flush=True)
            if (missing_required_metrics or nonfinite_required_metrics) and not bool(args.allow_missing_action_metrics):
                raise RuntimeError(
                    "Fixed-window action probe produced missing or non-finite required action metrics. "
                    "This run is invalid for action-platform conclusions. Pass "
                    "--allow-missing-action-metrics only for structure-only diagnostics."
                )
        finally:
            source.close()


if __name__ == "__main__":
    main()

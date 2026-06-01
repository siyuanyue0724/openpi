from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.picf.paligemma.wrapper import _masked_position_ids
from openpi.picf.paligemma.wrapper import _recover_flow_target
from openpi.picf.paligemma.wrapper import make_att_2d_masks
from scripts.picf_core_train import _apply_picf_trainable_scope
from scripts.picf_core_train import _build_loss_config
from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _load_tactile_backgrounds_npz
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _resolve_action_normalizer
from scripts.picf_core_train import _seed_everything
from scripts.picf_replay_windows import _coerce_loaded_args
from scripts.picf_replay_windows import _resolve_rank_seed


def _load_first_window_record(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty window record file: {path}")
    first = text.splitlines()[0]
    if first.startswith("["):
        payload = json.loads(text)
        if not payload:
            raise ValueError(f"Empty window record list: {path}")
        row = payload[0]
    else:
        row = json.loads(first)
    return {
        **row,
        "segment": int(row.get("segment", row.get("segment_id"))),
        "start_step": int(row.get("start_step", row.get("start_step_id"))),
    }


def _tensor_stats(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    if isinstance(value, np.ndarray):
        value = torch.as_tensor(value)
    if not isinstance(value, torch.Tensor):
        return {"present": True, "type": type(value).__name__, "repr": repr(value)[:256]}
    detached = value.detach()
    flat = detached.reshape(-1)
    stats: dict[str, Any] = {
        "present": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(flat.numel()),
    }
    if flat.numel() == 0:
        stats.update({"finite_all": True, "nan_count": 0, "inf_count": 0})
        return stats
    finite = torch.isfinite(flat)
    stats["finite_all"] = bool(finite.all().item())
    stats["finite_fraction"] = float(finite.to(dtype=torch.float32).mean().item())
    stats["nan_count"] = int(torch.isnan(flat).sum().item())
    stats["inf_count"] = int(torch.isinf(flat).sum().item())
    finite_flat = flat[finite]
    if finite_flat.numel() > 0:
        finite_float = finite_flat.to(dtype=torch.float32)
        stats["mean"] = float(finite_float.mean().item())
        stats["std"] = float(finite_float.std(unbiased=False).item()) if finite_float.numel() > 1 else 0.0
        stats["min"] = float(finite_float.min().item())
        stats["max"] = float(finite_float.max().item())
        stats["rms"] = float(torch.sqrt(torch.mean(finite_float.square())).item())
    return stats


def _scalar(value: torch.Tensor) -> float | str:
    scalar = float(value.detach().reshape(-1)[0].item())
    if math.isfinite(scalar):
        return scalar
    return "nan" if math.isnan(scalar) else ("inf" if scalar > 0 else "-inf")


def _parameter_stats(module: torch.nn.Module, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, torch.nn.Parameter):
            out[name] = _tensor_stats(value)
        elif isinstance(value, torch.nn.Module):
            params = list(value.parameters())
            if params:
                out[name] = _tensor_stats(torch.cat([param.detach().reshape(-1).to(dtype=torch.float32) for param in params]))
            else:
                out[name] = {"present": True, "module": type(value).__name__, "parameter_count": 0}
        else:
            out[name] = {"present": value is not None, "type": type(value).__name__ if value is not None else None}
    return out


def _install_action_debug_wrapper(
    semantic_encoder: Any,
    *,
    context_mode: str,
    attention_mask_dtype: str,
    embedding_dtype: str,
    mask_negative_value: float | None,
    records: list[dict[str, Any]],
) -> None:
    def debug_compute(self: Any, features: Any, *, extra_prefix_tokens: torch.Tensor | None, extra_action_context_tokens: torch.Tensor | None = None, action_chunk_target: torch.Tensor | np.ndarray, noise: torch.Tensor | None = None, time: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if context_mode == "no_context":
            extra_action_context_tokens = None
        elif context_mode == "no_extra":
            extra_action_context_tokens = None
            extra_prefix_tokens = None

        record: dict[str, Any] = {
            "stage": "compute_action_flow_loss_decompose",
            "context_mode": context_mode,
            "input_action_chunk_target": _tensor_stats(action_chunk_target),
            "input_extra_prefix_tokens": _tensor_stats(extra_prefix_tokens),
            "input_extra_action_context_tokens": _tensor_stats(extra_action_context_tokens),
            "adapter_params": _parameter_stats(
                self,
                [
                    "action_context_in_proj",
                    "action_context_q_proj",
                    "action_context_k_proj",
                    "action_context_v_proj",
                    "action_context_out_proj",
                    "action_context_gate_logit",
                    "action_in_proj",
                    "action_out_proj",
                ],
            ),
        }
        dtype = torch.float32
        target = self._prepare_action_chunk_target(action_chunk_target, device=self.device, dtype=dtype)[None, :]
        if noise is None:
            noise = torch.randn_like(target)
        if time is None:
            beta = torch.distributions.Beta(
                torch.tensor(1.5, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            time = (beta.sample((1,)) * 0.999 + 0.001).to(device=self.device, dtype=dtype)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * target
        u_t = noise - target
        record.update(
            {
                "target": _tensor_stats(target),
                "noise": _tensor_stats(noise),
                "time": _tensor_stats(time),
                "x_t": _tensor_stats(x_t),
                "u_t": _tensor_stats(u_t),
            }
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._combine_prefix(
            features,
            extra_prefix_tokens=extra_prefix_tokens,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(x_t, time)
        record.update(
            {
                "prefix_embs_pre_dtype": _tensor_stats(prefix_embs),
                "prefix_pad_masks": _tensor_stats(prefix_pad_masks),
                "prefix_att_masks": _tensor_stats(prefix_att_masks),
                "suffix_embs_pre_adapter": _tensor_stats(suffix_embs),
                "suffix_pad_masks": _tensor_stats(suffix_pad_masks),
                "suffix_att_masks": _tensor_stats(suffix_att_masks),
                "adarms_cond": _tensor_stats(adarms_cond),
            }
        )
        suffix_embs, adapter_metrics = self._apply_action_context_adapter(suffix_embs, extra_action_context_tokens)
        record["suffix_embs_post_adapter"] = _tensor_stats(suffix_embs)
        record["adapter_metrics"] = {key: _tensor_stats(value) for key, value in adapter_metrics.items()}
        model_dtype = self._model_runtime_dtype()
        record["model_runtime_dtype"] = str(model_dtype)
        if embedding_dtype == "float32":
            prefix_embs = prefix_embs.to(dtype=torch.float32)
            suffix_embs = suffix_embs.to(dtype=torch.float32)
        elif model_dtype in (torch.bfloat16, torch.float16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
            suffix_embs = suffix_embs.to(dtype=model_dtype)
        elif embedding_dtype != "runtime":
            raise ValueError(f"Unsupported embedding_dtype={embedding_dtype!r}")
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = _masked_position_ids(pad_masks)
        if mask_negative_value is None:
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        else:
            att_2d_masks_4d = torch.where(att_2d_masks[:, None, :, :], 0.0, float(mask_negative_value))
        if attention_mask_dtype == "runtime":
            att_2d_masks_4d = att_2d_masks_4d.to(dtype=prefix_embs.dtype)
        elif attention_mask_dtype == "float32":
            att_2d_masks_4d = att_2d_masks_4d.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported attention_mask_dtype={attention_mask_dtype!r}")
        record.update(
            {
                "prefix_embs_runtime": _tensor_stats(prefix_embs),
                "suffix_embs_runtime": _tensor_stats(suffix_embs),
                "pad_masks": _tensor_stats(pad_masks),
                "att_masks": _tensor_stats(att_masks),
                "att_2d_masks": _tensor_stats(att_2d_masks),
                "att_2d_masks_4d": _tensor_stats(att_2d_masks_4d),
                "position_ids": _tensor_stats(position_ids),
            }
        )

        def _forward(prefix: torch.Tensor, suffix: torch.Tensor, attn: torch.Tensor, pos: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, cond],
            )
            return suffix_out

        suffix_out_all = self._apply_checkpoint(_forward, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
        suffix_out = suffix_out_all[:, -self.action_horizon :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(lambda out: self.action_out_proj(out), suffix_out)
        total = torch.nn.functional.mse_loss(u_t, v_t)
        pos = torch.nn.functional.mse_loss(u_t[..., :3], v_t[..., :3])
        rot = torch.nn.functional.mse_loss(u_t[..., 3:6], v_t[..., 3:6])
        grip = torch.nn.functional.mse_loss(u_t[..., 6:7], v_t[..., 6:7])
        predicted_chunk = _recover_flow_target(x_t, v_t, time_expanded).detach()
        predicted = predicted_chunk[:, 0, :7]
        record.update(
            {
                "suffix_out_all": _tensor_stats(suffix_out_all),
                "suffix_out_action_horizon": _tensor_stats(suffix_out),
                "v_t": _tensor_stats(v_t),
                "predicted_chunk": _tensor_stats(predicted_chunk),
                "loss_total": _scalar(total),
                "loss_pos": _scalar(pos),
                "loss_rot": _scalar(rot),
                "loss_grip": _scalar(grip),
            }
        )
        records.append(record)
        return {
            "total": total,
            "action_pos": pos,
            "action_rot": rot,
            "action_gripper": grip,
            "predicted_action": predicted[0],
            "predicted_chunk": predicted_chunk[0],
            **adapter_metrics,
        }

    # PaliGemmaSemanticEncoder is a thin wrapper whose forward delegates to
    # `self.encoder(op, ...)`; FSDP can also wrap the callable module.  Patch the
    # inner implementation when present, otherwise the no-grad probe would still
    # report NaN outputs but produce an empty decomposition record.
    target = getattr(semantic_encoder, "encoder", semantic_encoder)
    target.compute_action_flow_loss = types.MethodType(debug_compute, target)
    if target is not semantic_encoder:
        semantic_encoder.compute_action_flow_loss = types.MethodType(debug_compute, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose a single PICF action-loss NaN into tensor stages.")
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank-seed", type=int, default=1)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--context-mode", choices=("as_called", "no_context", "no_extra"), default="as_called")
    parser.add_argument("--attention-mask-dtype", choices=("runtime", "float32"), default="runtime")
    parser.add_argument("--embedding-dtype", choices=("runtime", "float32"), default="runtime")
    parser.add_argument(
        "--mask-negative-value",
        type=float,
        default=None,
        help="Override additive attention mask negative value, e.g. -1e4 for bf16-safe SDPA diagnostics.",
    )
    parser.add_argument(
        "--grad-enabled",
        action="store_true",
        help="Run the single forward with gradient tracking enabled but without backward/optimizer update.",
    )
    args = parser.parse_args()

    args_json_path = Path(args.args_json)
    payload = json.loads(args_json_path.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
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
        mvtrack_sidecar_root=getattr(train_args, "mvtrack_sidecar_root", None),
        mvtrack_sidecar_proposal_nearest_max_gap=int(getattr(train_args, "mvtrack_sidecar_proposal_nearest_max_gap", 0)),
        action_normalizer=_resolve_action_normalizer(train_args),
        augmentation_mode=train_args.picf_augmentation_mode,
        photometric_strength=train_args.picf_photometric_strength,
        segment_indices=calvin_segment_indices,
    )
    records: list[dict[str, Any]] = []
    try:
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
        optimizer, _ = _build_optimizer(trainer, args=train_args)
        loaded_step = _load_checkpoint(path=Path(args.checkpoint), model=trainer, optimizer=optimizer, device=device)
        if args.mode == "eval":
            trainer.eval()
        else:
            trainer.train()
        _install_action_debug_wrapper(
            semantic_encoder,
            context_mode=str(args.context_mode),
            attention_mask_dtype=str(args.attention_mask_dtype),
            embedding_dtype=str(args.embedding_dtype),
            mask_negative_value=args.mask_negative_value,
            records=records,
        )

        window_record = _load_first_window_record(Path(args.window_jsonl))
        window = source.window_from_metadata(
            segment_id=int(window_record["segment"]),
            start_step_id=int(window_record["start_step"]),
        )
        point_counts = _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
        grad_context = torch.enable_grad() if bool(args.grad_enabled) else torch.no_grad()
        with grad_context:
            outputs = trainer(window, capture_visual_diagnostics=False)
        output_stats = {
            key: _tensor_stats(value)
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1 and (
                key.startswith("loss_") or key.startswith("pi_") or key.startswith("posterior_")
            )
        }
        payload_out = {
            "stage": "picf_action_nan_decompose_complete",
            "args_json": str(args.args_json),
            "checkpoint": str(args.checkpoint),
            "loaded_step": int(loaded_step),
            "mode": str(args.mode),
            "context_mode": str(args.context_mode),
            "attention_mask_dtype": str(args.attention_mask_dtype),
            "embedding_dtype": str(args.embedding_dtype),
            "mask_negative_value": args.mask_negative_value,
            "grad_enabled": bool(args.grad_enabled),
            "trainable_scope": trainable_scope_info,
            "window_record": window_record,
            "point_counts": [int(count) for count in point_counts],
            "debug_records": records,
            "output_stats": output_stats,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload_out, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload_out, sort_keys=True), flush=True)
    finally:
        source.close()


if __name__ == "__main__":
    main()

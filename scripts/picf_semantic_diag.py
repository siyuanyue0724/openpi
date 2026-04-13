from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as fn

from openpi.picf.core.training import _branch_is_usable
from openpi.picf.core.training import compute_transition_loss
from openpi.picf.core.training import extract_future_targets
from scripts.picf_core_train import _apply_foundation_profile
from scripts.picf_core_train import _build_loss_config
from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _load_state_dict_picf_compat
from scripts.picf_core_train import _normalize_train_args
from scripts.picf_core_train import _load_tactile_backgrounds_npz
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _PicfWindowTrainer


def _load_args(path: Path) -> argparse.Namespace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**payload)
    # Training artifacts already store the fully resolved runtime arguments. Re-running
    # the CLI parsers here is incorrect because fields such as tactile offsets are no
    # longer raw strings at this point.
    _normalize_train_args(args)
    if bool(getattr(args, "use_foundation_backbones", False)):
        _apply_foundation_profile(args)
    return args


def _load_model_only(*, path: Path, model: torch.nn.Module, device: torch.device) -> int:
    payload = torch.load(path / "metadata.pt", map_location=device, weights_only=False)
    state = torch.load(path / "model.pt", map_location=device, weights_only=False)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        try:
            _load_state_dict_picf_compat(model, state)
        except RuntimeError:
            try:
                model.core.load_state_dict(state, strict=True)
            except RuntimeError:
                _load_state_dict_picf_compat(model.core, state)
    return int(payload.get("step", 0))


def _semantic_components(output: Any, future: Any) -> dict[str, float]:
    predictive = output.state.predictive
    cache = predictive.prediction_cache
    pieces: dict[str, float] = {}

    def branch(name: str, pred: torch.Tensor | None, target: torch.Tensor | None, idx: int, kind: str) -> float:
        usable = _branch_is_usable(
            pred=pred,
            target=target,
            pred_available=cache.availability[idx],
            target_available=future.availability[idx],
        )
        pieces[f"{name}_usable"] = float(bool(usable))
        if not usable:
            return 0.0
        if kind == "mse":
            return float(fn.mse_loss(pred, target).item())
        if kind == "l1":
            return float(fn.l1_loss(pred, target).item())
        if kind == "bce":
            return float(fn.binary_cross_entropy_with_logits(pred, target).item())
        raise ValueError(kind)

    pieces["semantic_visual_latent"] = branch(
        "semantic_visual_latent", cache.visual_latent, future.visual_latent, 0, "mse"
    )
    pieces["semantic_visual_real"] = branch(
        "semantic_visual_real", cache.visual_real, future.visual_real, 1, "l1"
    )
    pieces["semantic_tactile_real"] = branch(
        "semantic_tactile_real", cache.tactile_real, future.tactile_real, 2, "l1"
    )
    pieces["semantic_point_real"] = branch(
        "semantic_point_real", cache.point_real, future.point_real, 3, "bce"
    )
    return pieces


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    return float(fn.cosine_similarity(a[None, :], b[None, :]).item())


def _describe_variant(
    *,
    trainer: _PicfWindowTrainer,
    window: Any,
    prompt: str,
) -> dict[str, Any]:
    current = dataclasses.replace(window.frames[0], prompt=prompt, reset_scaffold=True)
    nxt = dataclasses.replace(window.frames[1], reset_scaffold=False)
    semantic_override = None
    if trainer.semantic_encoder is not None:
        semantic_override = trainer.semantic_encoder.encode_observation(current)
    output = trainer.core.step(
        current,
        previous=None,
        semantic_override=semantic_override,
        action_future=current.action,
    )
    losses = compute_transition_loss(
        trainer.core,
        output,
        nxt,
        action_target=current.action,
        config=trainer.loss_config,
    )
    future = extract_future_targets(trainer.core, nxt)
    sem = semantic_override
    if sem is None:
        token_count = 0
        summary_norm = 0.0
        token_norm = 0.0
    else:
        token_count = int(sem.tokens.shape[0])
        summary_norm = float(torch.linalg.norm(sem.summary.float()).item())
        token_norm = float(torch.linalg.norm(sem.tokens.float()).item()) if sem.tokens.numel() > 0 else 0.0
    pieces = _semantic_components(output, future)
    return {
        "prompt": prompt,
        "token_count": token_count,
        "semantic_tokens_norm": token_norm,
        "control_tokens_norm": float(torch.linalg.norm(output.state.predictive.control_tokens.float()).item()),
        "control_query_state_norm": float(torch.linalg.norm(output.state.predictive.control_query_state.float()).item()),
        "predictive_query_state_norm": float(torch.linalg.norm(output.state.predictive.predictive_query_state.float()).item()),
        "pooled_state_norm": float(torch.linalg.norm(output.state.predictive.pooled_state.float()).item()),
        "posterior_global_norm": float(torch.linalg.norm(output.state.posterior.global_post.float()).item()),
        "action": output.state.predictive.action.detach().cpu().tolist(),
        "physical_global_pred_norm": float(torch.linalg.norm(output.state.predictive.physical_global_pred.float()).item()),
        "global_pred_norm": float(torch.linalg.norm(output.state.predictive.global_pred.float()).item()),
        "global_minus_physical_norm": float(
            torch.linalg.norm(
                (output.state.predictive.global_pred - output.state.predictive.physical_global_pred).float()
            ).item()
        ),
        "loss_total": float(losses.total.item()),
        "loss_action": float(losses.action.item()),
        "loss_semantic_future_aux": float(losses.semantic_future_aux.item()),
        "loss_visual_real": float(losses.visual_real.item()),
        "loss_tactile_real": float(losses.tactile_real.item()),
        "loss_point_real": float(losses.point_real.item()),
        **pieces,
        "semantic_summary_norm": summary_norm,
        "semantic_summary": None if sem is None else sem.summary.detach().cpu(),
        "posterior_global_post": output.state.posterior.global_post.detach().cpu(),
        "physical_global_pred": output.state.predictive.physical_global_pred.detach().cpu(),
    }


def _find_different_prompt_index(source: Any, *, base_prompt: str, start_index: int) -> int:
    for offset in range(max(1, len(source))):
        idx = (int(start_index) + offset) % len(source)
        prompt = source.window(idx).prompt
        if str(prompt) != str(base_prompt):
            return idx
    return int(start_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--wrong-window-index", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    runtime_args = _load_args(args.args_json)
    device = torch.device(args.device)
    source = _CalvinTransitionSource(
        runtime_args.calvin_root,
        split=runtime_args.split,
        backend=runtime_args.backend,
        unroll_steps=runtime_args.unroll_steps,
        use_tactile=bool(runtime_args.use_tactile),
        tactile_sensor_names=runtime_args.tactile_sensor_names,
        tactile_sensor_offsets_m=runtime_args.tactile_sensor_offsets_m,
        tactile_calibration=runtime_args.tactile_calibration_path,
        tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(runtime_args.tactile_backgrounds_path),
        use_scene_obs=bool(getattr(runtime_args, "use_scene_obs", False)),
    )
    core, semantic_encoder, use_visual_override = _build_model(runtime_args, device=device)
    trainer = _PicfWindowTrainer(
        core,
        semantic_encoder=semantic_encoder,
        visual_grid=runtime_args.visual_grid,
        use_visual_override=use_visual_override,
        loss_config=_build_loss_config(runtime_args),
    ).to(device)
    _materialize_model_parameters(trainer, source=source, rank=0)
    step = _load_model_only(path=args.checkpoint_dir, model=trainer, device=device)
    trainer.eval()

    window = source.window(args.window_index)
    wrong_index = _find_different_prompt_index(source, base_prompt=window.prompt, start_index=args.wrong_window_index)
    wrong_window = source.window(wrong_index)
    prompts = {
        "actual": window.prompt,
        "blank": "",
        "wrong": wrong_window.prompt,
    }
    results: dict[str, Any] = {}
    with torch.no_grad():
        for name, prompt in prompts.items():
            results[name] = _describe_variant(trainer=trainer, window=window, prompt=prompt)

    actual_summary = results["actual"].pop("semantic_summary")
    blank_summary = results["blank"].pop("semantic_summary")
    wrong_summary = results["wrong"].pop("semantic_summary")
    actual_post = results["actual"].pop("posterior_global_post")
    blank_post = results["blank"].pop("posterior_global_post")
    wrong_post = results["wrong"].pop("posterior_global_post")
    actual_phys = results["actual"].pop("physical_global_pred")
    blank_phys = results["blank"].pop("physical_global_pred")
    wrong_phys = results["wrong"].pop("physical_global_pred")
    summary = {
        "wrong_window_index": wrong_index,
        "prompt_cosine_actual_blank": _cosine(actual_summary, blank_summary),
        "prompt_cosine_actual_wrong": _cosine(actual_summary, wrong_summary),
        "posterior_l2_actual_blank": float(torch.linalg.norm((actual_post - blank_post).float()).item()),
        "posterior_l2_actual_wrong": float(torch.linalg.norm((actual_post - wrong_post).float()).item()),
        "physical_global_pred_l2_actual_blank": float(torch.linalg.norm((actual_phys - blank_phys).float()).item()),
        "physical_global_pred_l2_actual_wrong": float(torch.linalg.norm((actual_phys - wrong_phys).float()).item()),
        "action_l2_actual_blank": float(
            torch.linalg.norm(
                torch.tensor(results["actual"]["action"], dtype=torch.float32)
                - torch.tensor(results["blank"]["action"], dtype=torch.float32)
            ).item()
        ),
        "action_l2_actual_wrong": float(
            torch.linalg.norm(
                torch.tensor(results["actual"]["action"], dtype=torch.float32)
                - torch.tensor(results["wrong"]["action"], dtype=torch.float32)
            ).item()
        ),
    }
    print(json.dumps({"checkpoint_step": step, "window_index": args.window_index, "summary": summary, "variants": results}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
import sys
import torch

repo = Path("/root/openpi_probe_current_20260529")
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "src"))

from scripts.picf_core_train import _apply_picf_trainable_scope
from scripts.picf_core_train import _build_loss_config
from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _collect_nonfinite_parameter_diagnostics
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _load_tactile_backgrounds_npz
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _resolve_action_normalizer
from scripts.picf_core_train import _seed_everything
from scripts.picf_replay_windows import _coerce_loaded_args
from scripts.picf_replay_windows import _resolve_rank_seed

args_json = Path("/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531/args.json")
ckpt = Path("/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531/9200")
payload = json.loads(args_json.read_text())
args = _coerce_loaded_args(payload, device_override="cuda:0")
device = torch.device(str(args.device))
rank_seed = _resolve_rank_seed(rank_seed=1, rng_rank=None)
_seed_everything(int(args.seed), int(rank_seed))
segments = [int(p) for p in str(args.calvin_segment_indices).split(",") if p.strip()] if getattr(args, "calvin_segment_indices", None) else None
source = _CalvinTransitionSource(
    args.calvin_root,
    split=args.split,
    backend=args.backend,
    unroll_steps=args.effective_unroll_steps,
    action_horizon=args.action_horizon,
    use_tactile=bool(args.use_tactile),
    tactile_sensor_names=args.tactile_sensor_names,
    tactile_sensor_offsets_m=args.tactile_sensor_offsets_m,
    tactile_calibration=args.tactile_calibration_path,
    tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(args.tactile_backgrounds_path),
    use_scene_obs=bool(args.use_scene_obs),
    load_tracklet_fields=bool(getattr(args, "tracklet_memory_enabled", False)),
    load_proposal_fields=bool(getattr(args, "proposal_memory_enabled", False)),
    mvtrack_sidecar_root=getattr(args, "mvtrack_sidecar_root", None),
    mvtrack_sidecar_proposal_nearest_max_gap=int(getattr(args, "mvtrack_sidecar_proposal_nearest_max_gap", 0)),
    action_normalizer=_resolve_action_normalizer(args),
    augmentation_mode=args.picf_augmentation_mode,
    photometric_strength=args.picf_photometric_strength,
    segment_indices=segments,
)
try:
    core, semantic_encoder, use_visual_override = _build_model(args, device=device)
    core = core.to(device)
    trainer = _PicfWindowTrainer(
        core,
        semantic_encoder=semantic_encoder,
        visual_grid=args.visual_grid,
        use_visual_override=use_visual_override,
        loss_config=_build_loss_config(args),
        picf_mode=str(getattr(args, "picf_mode", "enabled")),
        burnin_steps=int(getattr(args, "burnin_steps", 0)),
        burnin_mode=str(getattr(args, "burnin_mode", "full")),
    ).to(device)
    _materialize_model_parameters(trainer, source=source, rank=int(rank_seed))
    _apply_picf_trainable_scope(trainer, args=args, logger=None)
    optimizer, _ = _build_optimizer(trainer, args=args)
    loaded = _load_checkpoint(path=ckpt, model=trainer, optimizer=optimizer, device=device)
    diag = _collect_nonfinite_parameter_diagnostics(trainer, optimizer=optimizer, max_items=200)
    print(json.dumps({"loaded_step": int(loaded), "diag": diag}, indent=2, sort_keys=True))
finally:
    source.close()

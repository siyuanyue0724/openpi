# CALVIN Validation README

This document is the current executable validation guide for CALVIN under the
present PICF codebase.

The canonical architecture and deployment handoff is:

- [`README.md`](/home/siyuanyue/Documents/openpi/README.md)
- [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
- [`src/openpi/picf/README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)

The formal contract is:

- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)

Important status note:

- `README_v2.2.md` is the current local architecture and deployment document
- `PICF_FORMAL_CONTRACT.md` records the current live-code contract that existing
  regression tests enforce
- `README_v2.1.md` is retained only as the archived pre-v2.2 deployment record
- the current active document set is:
  - `README.md`
  - `src/openpi/picf/README.md`
  - `src/openpi/picf/README_v2.2.md`
  - `PICF_FORMAL_CONTRACT.md`
  - `docs/CALVIN_VALIDATION_README.md`

## 1. Dataset / Loader Validation

Directory backend:

```bash
python scripts/stageb_calvin_audit.py \
  --mode dataset \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split validation
```

Loader validation:

```bash
python scripts/stageb_calvin_audit.py \
  --mode loader \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split validation \
  --batch-size 4 \
  --num-workers 0
```

Use the zip backend only if you intentionally want to validate zip loading.

## 2. Contract Validation

Run this before training changes or deployment:

```bash
python scripts/verify_picf_contract.py
```

Current live-code contract meaning:

- semantic wrapper restores the PI0.5 expert path and injects robot state back
  into the prompt path
- trainer primary action loss uses PI0.5 flow matching, and the training-time
  denoised chunk estimate is recovered as `x_t - t * v_t`
- serving primary action path uses the PI0.5 denoise sampler and refreshes the
  predictive cache with the sampled action chunk
- semantic does not affect physical observation anchors
- semantic does not affect physical posterior
- semantic does not affect `physical_prediction_cache`
- next innovation reads only `previous.predictive.physical_prediction_cache`
- semantic enters the core only through current-step task readout
- public task-readout memory is `fused_tokens + visual_tokens`
- task readout rereads private dense visual / tactile / point payloads
- the core now builds one canonical conditioned control state `C_t`
- PI0.5 action conditioning uses `C_t^{pi}` derived from `C_t`
- conditioned future now uses token-level physical predictive tokens plus
  future-condition tokens, not the raw semantic prefix
- live observation competition now uses native-first visual routing
- live tactile public routing now uses group-level proposal competition with
  winner-read over dense tactile group memory
- live visual innovation targets include denser native-V-JEPA latent probes
- live tactile innovation targets include dense tactile latent probes in
  addition to tactile map and auxiliaries
- live point innovation targets include native point latent probes in addition
  to occupancy
- the physical core remains width `512`, while the conditioned control/future
  trunks remain semantic-width `2048`
- physical posterior / innovation / physical predictive tokens are up-projected
  into the semantic-width conditioned-control / conditioned-future trunks
- `posterior.global_post` explicitly enters conditioned control
- core no longer uses a direct trainable `7D` action head

Runtime-mode note:

- `--picf-mode enabled` is the canonical v2.2 deployment path
- `--picf-mode ablated` disables PICF recurrent/control/future branches and
  runs PI0.5-only action training / serving with `extra_prefix_tokens=None`
- ablated mode is intended for parity checks against the main-branch PI0.5
  baseline, not as a replacement for the v2.2 contract
- ablated checkpoints intentionally serialize only the live PI0.5 semantic
  subtree plus optimizer state; they do not force-save the frozen lazy PICF
  core just to satisfy generic trainer checkpoint traversal
- with `optimizer_checkpoint_mode=auto`, ablated runs now default to model-only
  checkpoints; pass `--optimizer-checkpoint-mode full` only if optimizer-state
  resume is required
- `scripts/serve_picf_policy.py` now also accepts `--picf-mode {enabled,ablated}`;
  if omitted, serving uses the checkpoint's saved `picf_mode`
- when serving overrides checkpoint mode, runtime args are re-normalized before
  model/source construction so ablated serve does not retain enabled-mode
  tactile/visual branch assumptions

This section describes the deployed live baseline.

## 3. Trainer Smoke Validation

CPU smoke:

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cpu
```

PI0.5-only ablation smoke:

```bash
python scripts/picf_core_train.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --checkpoint-base-dir /tmp \
  --exp-name picf_ablation_smoke \
  --semantic-mode paligemma \
  --semantic-trainable \
  --picf-mode ablated \
  --num-train-steps 2 \
  --save-interval 2 \
  --log-interval 1 \
  --no-wandb
```

PI0.5-only ablation serve smoke:

```bash
python scripts/serve_picf_policy.py \
  --checkpoint /path/to/checkpoint_or_latest_dir \
  --device cpu \
  --picf-mode ablated
```

Current 2x40GB PI0.5-only ablation long-run profile:

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=/root/openpi_posterior_vla_clean/src
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name picf_v22_ablated_pi05_30000_ckpt2500_print100 \
  --overwrite \
  --device cuda \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --accum-steps 1 \
  --unroll-steps 2 \
  --num-train-steps 30000 \
  --save-interval 2500 \
  --log-interval 100 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --wandb-mode disabled \
  --no-wandb \
  --picf-mode ablated \
  --semantic-mode paligemma \
  --semantic-trainable \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --action-normalization quantile \
  --action-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json
```

Run-shape interpretation for this profile:

- `world_size=2`
- `accum_steps=1`
- `unroll_steps=2`
- each rank samples one window per optimizer step
- each window produces two action-only transitions
- therefore one optimizer step covers two windows and four action-only
  transition objectives globally

Monitoring:

```bash
tail -f /mnt/checkpoints/picf_core/debug/picf_v22_ablated_pi05_30000_ckpt2500_print100_*.log
```

```bash
watch -n 2 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

Parity note:

- this run preserves the current PICF trainer/window shell while disabling PICF
  semantics
- it is an operational ablation profile, not an exact reproduction of the
  official/main-branch PI0.5 training definition
- if exact PI0.5 training-definition parity is required, use `picf_mode=ablated`
  with `--unroll-steps 1`

Foundation-backbone CUDA smoke:

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile
```

## 4. Tactile Calibration

If calibration files are missing, regenerate them with the full supported
script instead of fabricating placeholders:

```bash
python scripts/calvin/precompute_tactile_contact_calibration.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --device cuda \
  --anytouch-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --output-dir /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8
```

Required outputs:

- `tactile_backgrounds.npz`
- `tactile_contact_stats.json`
- `tactile_fingertip_calibration.json`

## 5. Historical Baseline Training Command

The command below is preserved as a historical baseline from the pre-v2.2
semantic-prefix-primary mixed-width implementation.

It is **not** the recommended canonical launch command for current deployment.

Historical command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile \
  --accum-steps 1 \
  --save-interval 5000 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --exp-name picf_semantic_prefix_primary_mixedwidth_a4_acc1_pct75_clean_v1
```

This is a preserved historical launch path from before the fully restored
PI0.5-action-stack deployment.

Current v2.2 long-run training profile:

- `--num-train-steps 30000`
- `--save-interval 2500`
- `--grad-clip-mode percentile`
- `--grad-clip-percentile 75`
- `--grad-clip-window 100`
- `--visual-activation-checkpointing`
- `--semantic-gradient-checkpointing`
- `--window-activation-checkpointing` as an explicit fallback, not a silent
  foundation-profile default
- `--diagnostic-interval 0`
- `--training-strategy fsdp_full_shard` for the current 4x40GB A100 FSDP investigation profile
- `--optimizer-sharding none` on that FSDP path; `zero1` remains a DDP-only fallback and is not sufficient for all-backbone v2.2 finetuning
- `--visual-finetune-mode full|frozen` is the supported visual-backbone switch on this profile. `full` is the default all-backbone route; `frozen` keeps V-JEPA fixed while leaving the rest of the stack unchanged.
- the standard FSDP profile uses flat-parameter mode (`use_orig_params=False`) together with `backward_prefetch=BACKWARD_POST` and `limit_all_gathers=True`, so the 4x40GB job reduces parameter-view residency before changing anything about the optimization objective
- the semantic FSDP path now pre-wraps the directly called PI0/PaliGemma runtime hot leaves as nested exact shards (`embed_tokens`, per-layer `q/k/v/o`, per-layer `mlp`, and PI0 action/time projections), then applies the mixed-dtype semantic root wrapper only to the remaining parameters; the SigLIP vision tower and multimodal projector currently stay under the outer semantic root because their current image-path implementations are not yet nested-FSDP-safe under the present view-alias constraints
- the standard FSDP profile now recursively splits large uniform-dtype subtrees with a 512MiB parameter-storage budget per boundary, and it shards the safe core transformer stacks (`token_fusion`, `obs_self`, `posterior_self`, `task_self`, `predictive_world`, `predictive_semantic_world`, `control_world`) before the root wrapper so the remaining root shard stays light
- those safe core transformer stacks are now also explicitly reattached back onto `core` after wrapping; this is part of the 4x40GB contract so the root FSDP wrapper does not silently pull them back into one monolithic flat-parameter shard
- the root FSDP wrapper now ignores fully frozen backbone subtrees instead of flattening mixed `requires_grad` parameter sets. This is required for `visual_finetune_mode=frozen` and is part of the standard 40GB engineering contract whenever a backbone is frozen under full-shard training.
- transformer stacks on this profile now materialize every incoming activation once at stack entry before attention. PICF builds many `[1, T, C]` batches via `tokens[None, :]`, and FSDP can also surface storage-sharing tensors whose aliasing is not reliably visible through `_base`; the stack-entry clone is mathematically exact and avoids FSDP/autograd multi-view alias failures inside residual attention blocks
- the training stack still supports checkpointing the full `_PicfWindowTrainer.forward(...)` window body during training. That remains an exact fallback for extra peak-memory reduction, and the checkpoint input is still a standalone dummy leaf on the active CUDA device rather than a view into any FSDP flat parameter, so recompute keeps exact training math without feeding full-parameter gradients back into local shard metadata. It is now an explicit operator knob rather than something the foundation profile silently forces on every launch.
- the custom PI0/Gemma dual-branch semantic attention path on this profile uses SDPA instead of the eager attention workspace, which removes the large step-2 attention buffer without changing the training objective.
- tokenwise-only projections and FFNs on the current hottest paths now support exact sequence chunking. On the present profile this is enabled by default as `tokenwise_ff_chunk_size=64` for PICF core transformer/cross-attention FFNs and `semantic_tokenwise_chunk_size=64` as the semantic compatibility knob. The live trainer also resolves that semantic compatibility knob into `semantic_projection_chunk_size=128` and `semantic_mlp_chunk_size=64` on the standard 4x40GB full-shard profile, so future semantic backbones can tune projection-heavy and MLP-heavy regions independently without changing model math. This is an execution change, not a model-capacity change.
- the PI0/PaliGemma wrapper no longer adds an extra outer checkpoint around semantic forward blocks when the native language-model / vision-tower / expert-model checkpointing path is already active. This avoids redundant recompute while preserving the same gradients.
- the PI0/PICF semantic runtime now drops the unused outer causal-LM heads after checkpoint load. Those logits heads are not used by the live action-training path, so removing them from the runtime graph is mathematically exact and prevents dead semantic weights from bloating FSDP wrapping.
- the FSDP path uses explicit global-L2 grad norm / percentile clipping across local shards instead of `FSDP.clip_grad_norm_`, because the semantic stack deliberately mixes bf16 bulk weights with a small float32 stabilizer subset
- standard multi-rank FSDP startup now stages the PI0/PaliGemma semantic checkpoint into a node-local cache before rank-local `load_state_dict(...)`; this avoids four ranks faulting the same `/mnt/checkpoints/pi05_base_pytorch` file tree at once without changing model math. Default cache root: `~/.cache/openpi/pi0_checkpoints`. Override with `OPENPI_LOCAL_CHECKPOINT_CACHE_DIR` or disable/force via `OPENPI_STAGE_PI0_CHECKPOINT=off|on|auto`
- V-JEPA mixed precision on CUDA now uses the same safe autocast rule in both frozen and trainable modes. Keep `visual_dtype=float32` if you want the most conservative frozen path, or use `visual_dtype=bfloat16/float16` with the knowledge that the encoder remains in native fp32 and the forward path is autocast rather than hard-cast.
- window training on this profile now forwards only the canonical recurrent carry between transitions rather than the full `PicfCoreState`. This keeps the recurrence mathematically exact while removing non-recurrent semantic/control/task-readout state from the cross-step training graph.
- loss-side future supervision now treats next-observation targets as explicit
  stop-gradient teacher values, and the window trainer reuses the shared middle
  frame's already-computed `current_targets` as detached future targets for the
  preceding transition. This removes duplicated target-building work inside an
  unrolled window without changing the objective.

Current verification posture:

- this document records the supported v2.2 training / validation knobs and
  current engineering contract
- it should not be read as a blanket claim that a full long cloud run is
  currently active or completed
- current local verification is stronger than current run-completion evidence:
  - `training_test.py`, `wrapper_test.py`, and `picf_core_train_test.py` pass
  - `scripts/verify_picf_contract.py` passes static checks, documentation
    checks, targeted invariance regressions, the full core regression suite,
    and a smoke training check
- the latest exact-memory step-time evidence remains the short-run profiling
  numbers documented in `/tmp/picf_v22_speed_audit_20260420.md`

Important status note:

- this section records the implemented 4x40GB FSDP training contract and the
  currently supported launch knobs
- this section should be read as the current 4x40GB full-train contract and
  tuning surface, not as a claim that every optional memory valve is enabled by
  default
- it still does not overclaim beyond current evidence:
  - it does **not** claim that `step 2500` or full `30000` completion has
    already been observed unless a later audit explicitly records that fact

Distributed runtime note for current multi-rank bring-up:

- do not use `TORCH_DISTRIBUTED_DEBUG=DETAIL` as the default 4-GPU training
  setting
- current `scripts/picf_core_train.py` defaults DDP startup to
  `TORCH_DISTRIBUTED_DEBUG=INFO`
- if the environment injects `TORCH_DISTRIBUTED_DEBUG=DETAIL`, the trainer
  fails fast unless
  `OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL=1` is explicitly set
- reserve DETAIL for targeted distributed-runtime debugging only

## 6. Serving / Rollout

After training:

- keep the normalized action contract in trainer / serving aligned
- use the standard serving entrypoint
- then run CALVIN evaluator rollouts from the served checkpoint

The serving path must preserve:

- normalized action inside the core
- unnormalized action at environment execution

## 7. Historical Notes

Older CALVIN README fragments have been retired.
Use this file for the current validation workflow.

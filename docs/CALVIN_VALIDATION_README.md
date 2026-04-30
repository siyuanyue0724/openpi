# CALVIN Validation README

This document is the current executable validation guide for CALVIN under the
present PICF codebase.

The canonical architecture and deployment handoff is:

- [`README.md`](/home/siyuanyue/Documents/openpi/README.md)
- [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
- [`src/openpi/picf/README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)

The formal contract is:

- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)

Direct execution entry points in this document:

- current canonical full PICF long-run training:
  [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch)
- current cloud-tested 20-sequence CALVIN video evaluation, including the
  full PICF `step=7500` recipe and the maintained PI0.5-only ablation recipe:
  [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation)

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
- current `picf_mode=ablated` is not a `pi0.5_sonata` replica:
  - the Sonata point feature extractor is not built
  - the visual branch falls back to the null visual encoder
  - the tactile branch falls back to the null tactile encoder
  - the PICF core is frozen and only the PI0.5 semantic/action path remains live
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
  --action-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --prompt-state-normalization inherit \
  --prompt-state-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json
```

Run-shape interpretation for this profile:

- `world_size=2`
- `accum_steps=1`
- `unroll_steps=2`
- each rank samples one window per optimizer step
- each window produces two action-only transitions
- therefore one optimizer step covers two windows and four action-only
  transition objectives globally
- relative to a historical 2-GPU PI0.5 no-Sonata run with `unroll_steps=1`,
  one optimizer step on this ablated profile covers twice as many global
  action-training objectives
- therefore:
  - current ablated `2500` optimizer steps
  - should be compared against about `5000` optimizer steps from the historical
    2-GPU no-Sonata PI0.5 baseline
  - when the comparison is framed in terms of total global action objectives

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
- it also does not reproduce the preserved historical 2-GPU PI0.5 execution
  shell exactly:
  - the maintained ablation run uses `training_strategy=fsdp_full_shard`
  - the historical PI0.5 CALVIN baselines used the direct DDP trainer
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

## 5. Full PICF Training And Historical Baseline

### 5.1 Current Canonical Full PICF Long-Run Launch

This is the current canonical long-run launch template for the **full**
`picf_mode=enabled` v2.2 path on the maintained 4x40GB A100 profile.

It is the command to use when the goal is:

- full PICF recurrent/control/future training
- foundation backbones enabled
- tactile enabled
- PI0.5 action path active
- checkpoint cadence every `2500` optimizer steps
- progress/loss printed every `100` optimizer steps

Current canonical launch:

```bash
cd /root/openpi_run_latest
export PYTHONPATH=/root/openpi_run_latest/src
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/openpi/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name <exp_name> \
  --overwrite \
  --device cuda \
  --picf-mode enabled \
  --use-foundation-backbones \
  --use-tactile \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --accum-steps 1 \
  --num-train-steps 30000 \
  --save-interval 2500 \
  --log-interval 100 \
  --diagnostic-interval 0 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --visual-finetune-mode full \
  --visual-activation-checkpointing \
  --semantic-gradient-checkpointing \
  --wandb-mode disabled \
  --no-wandb \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --action-normalization quantile \
  --action-norm-stats-path /root/openpi_run_latest/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --prompt-state-normalization inherit \
  --prompt-state-norm-stats-path /root/openpi_run_latest/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --tokenwise-ff-chunk-size 64 \
  --semantic-tokenwise-chunk-size 64 \
  --semantic-projection-chunk-size 128 \
  --semantic-mlp-chunk-size 64
```

Interpretation notes:

- `--picf-mode enabled` is the explicit canonical full PICF mode
- `--save-interval 2500` is the maintained operational checkpoint cadence
- `--log-interval 100` is the maintained loss-print cadence
- `--diagnostic-interval 0` avoids extra diagnostic spam on the long-run path
- `--window-activation-checkpointing` is **not** part of this canonical launch;
  it remains an explicit fallback knob when the operator needs more peak-memory
  relief

Recommended live monitoring:

```bash
tail -f /mnt/checkpoints/picf_core/debug/<exp_name>.log
```

```bash
watch -n 2 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

If the operator wants early proof that the run crossed step 10 without changing
training math, keep the same command and temporarily add:

- `--log-interval 10`
- optionally `--no-progress`

6x40GB extension:

- use the same command and change `--nproc_per_node=6`
- set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- keep `--accum-steps 1`, `--save-interval 2500`, and `--log-interval 100`
- the run then has `effective_global_batch=6` instead of the 4-GPU profile's
  `effective_global_batch=4`
- this is a valid same-objective full-PICF run, but loss curves should not be
  interpreted as the exact same optimizer trajectory as the 4-GPU profile

Detached cloud launch rule:

- for rented cloud long runs, do not rely on a plain SSH-attached `torchrun`
  process, even with a trailing `&`
- the observed failure mode is
  `torch.distributed.elastic.multiprocessing.api.SignalException: ... signal: 1`,
  which is an external SIGHUP to the elastic launcher rather than a model,
  optimizer, OOM, or dataflow failure
- write the exact `torchrun` command into a launch script and start it with
  `nohup setsid "$RUN" </dev/null > "$LOG" 2>&1 &`
- after reconnecting, check that the launcher has no controlling TTY:

```bash
ps -o pid,ppid,sid,tty,etime,stat,cmd -C torchrun
```

Expected healthy state:

- `TTY=?`
- `PPID=1` or otherwise detached from the interactive SSH shell
- workers still visible in `ps -ef | grep picf_core_train.py`

### 5.2 Historical Baseline Training Command

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

### 6.1 Current Cloud CALVIN Video Evaluation

This section is the current cloud-side CALVIN video evaluation recipe for both:

- full PICF checkpoints, for example the current `step=7500` full-PICF
  checkpoint
- PI0.5-only ablation checkpoints

The operational pattern is the same:

- serve the checkpoint over websocket with
  `scripts/serve_picf_policy.py`
- run `scripts/calvin/evaluate_picf_policy.py` from the `calvin38`
  environment
- save videos to a local disk path such as `/tmp/...` during evaluation
- after the run finishes, copy videos and logs back into the desired `/mnt`
  checkpoint/eval directory

This is the validated operational recipe for the current cloud image. It is not
just a historical sketch.

#### 6.1.0 GPU Usage And Current Full-PICF 7500 Artifact Paths

CALVIN video evaluation is **not** the same parallel shape as FSDP training.

The current serving/evaluation topology is:

```text
CALVIN env step -> websocket request -> one policy server inference -> CALVIN env step
```

For one 20-sequence evaluation job, this is a single online rollout stream. It
does not automatically use all GPUs.

Current validated GPU allocation:

- `cuda:0`: full PICF policy server
- `cuda:1`: CALVIN EGL / evaluator process
- remaining GPUs: idle unless the operator intentionally starts additional
  independent server/evaluator shards on different ports and output folders

Do not interpret idle GPUs during a single CALVIN rollout as a training
configuration failure. It is an evaluation scheduling property. Multi-GPU speedup
requires multiple independent evaluator shards, not one server process.

Current full-PICF 7500 checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/picf_v22_full_picf_a6_30000_ckpt2500_print100_p192_20260424_r2/7500
```

Current full-PICF 7500 video-eval output directory:

```text
/mnt/checkpoints/picf_core/eval/picf_v22_full_picf_step7500_eval20_video_20260427_r1
```

Current full-PICF 7500 monitor commands:

```bash
tail -f /mnt/checkpoints/picf_core/eval/picf_v22_full_picf_step7500_eval20_video_20260427_r1/logs/eval_calvin38_full.log
```

```bash
watch -n 5 'find /mnt/checkpoints/picf_core/eval/picf_v22_full_picf_step7500_eval20_video_20260427_r1/videos -maxdepth 1 -type f -name "*.mp4" -printf "%f %s\n" | sort | tail -n 20'
```

```bash
grep -A20 'Results for Epoch' /mnt/checkpoints/picf_core/eval/picf_v22_full_picf_step7500_eval20_video_20260427_r1/logs/eval_calvin38_full.log
```

Operational note from the current cloud image:

- the server has been validated to answer a direct websocket inference request
  from the full-PICF `7500` checkpoint
- a single full-PICF inference call is much slower than ablated PI0.5-only
  serving, so the CALVIN progress bar can remain at `0/20` while the first
  sequence is still stepping
- videos are valid only if their size is non-trivial; `44B` files are broken
  placeholders, while the current first full-PICF video observed under `/tmp`
  was about `524KB`

#### 6.1.1 Serve The Full-PICF 7500 Checkpoint

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=/root/openpi_posterior_vla_clean/src:/root/openpi_posterior_vla_clean/packages/openpi-client/src
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
export OPENPI_DISABLE_TORCH_COMPILE=1

/root/openpi/.venv/bin/python scripts/serve_picf_policy.py \
  --checkpoint /mnt/checkpoints/picf_core/picf_core/picf_v22_full_picf_a6_30000_ckpt2500_print100_p192_20260424_r2/7500 \
  --device cuda:0 \
  --port 8000 \
  --picf-mode enabled
```

Useful checks:

```bash
ss -ltnp | grep :8000
ps -ef | grep -E 'serve_picf_policy.py' | grep -v grep
```

#### 6.1.2 Run The Full-PICF 20-Sequence Evaluator With Video

The most robust current recipe writes runtime videos to `/tmp` and mirrors them
back to `/mnt`. This avoids cloud-image issues where `cv2.VideoWriter` or remote
filesystem buffering can produce invalid tiny files when writing videos directly
into `/mnt`.

```bash
base=/mnt/checkpoints/picf_core/eval/picf_v22_full_picf_step7500_eval20_video_20260427_r1
tmp=/tmp/picf_v22_full_7500_eval20_video_20260427_r1
mkdir -p "$base/videos" "$base/logs" "$base/eval_logs" "$tmp/videos" "$tmp/eval_logs"

cd /mnt/calvin/calvin_models/calvin_agent
eval "$(/root/bin/micromamba shell hook -s bash)"
micromamba activate calvin38

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1
export EGL_VISIBLE_DEVICES=1
export OPENPI_SERVER_HOST=127.0.0.1
export OPENPI_SERVER_PORT=8000
export OPENPI_EVAL_TAG=picf_v22_full_7500
export PYTHONPATH=/root/calvin_patch:/mnt/calvin/calvin_env:/root/openpi_posterior_vla_clean:/root/openpi_posterior_vla_clean/src:/root/openpi_posterior_vla_clean/packages/openpi-client/src:/mnt/calvin/calvin_models/calvin_agent

python /root/openpi_posterior_vla_clean/scripts/calvin/evaluate_picf_policy.py \
  --dataset_path /mnt/calvin_data/task_ABC_D \
  --eval_log_dir "$tmp/eval_logs" \
  --num_sequences 20 \
  --save_video \
  --video_dir "$tmp/videos" \
  > "$tmp/eval_calvin38_full.log" 2>&1
```

After or during evaluation, mirror outputs to `/mnt`:

```bash
base=/mnt/checkpoints/picf_core/eval/picf_v22_full_picf_step7500_eval20_video_20260427_r1
tmp=/tmp/picf_v22_full_7500_eval20_video_20260427_r1
mkdir -p "$base/videos" "$base/logs" "$base/eval_logs"
cp -f "$tmp/eval_calvin38_full.log" "$base/logs/eval_calvin38_full.log"
cp -f "$tmp/eval_calvin38_full.log" "$base/eval_calvin38_full.log"
cp -f "$tmp"/eval_logs/* "$base/eval_logs"/ 2>/dev/null || true

for src in "$tmp"/videos/*.mp4; do
  [ -f "$src" ] || continue
  name=$(basename "$src")
  dst="$base/videos/$name"
  src_size=$(stat -c %s "$src")
  dst_size=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$src_size" -gt 4096 ] && [ "$src_size" -gt "$dst_size" ]; then
    cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
  fi
done
```

Do not use `cp -n` or `rsync --ignore-existing` for video mirroring. If an
earlier direct `/mnt` write created a tiny placeholder such as a `44B` `.mp4`,
ignore-existing copy modes will preserve the broken file forever instead of
replacing it with the valid `/tmp` output.

#### 6.1.3 Legacy Maintained Ablation Example

The current validated cloud-side evaluation recipe for the PI0.5-only ablation
path is preserved below.

Important interpretation note:

- the validated checkpoint below is the maintained **operational**
  `picf_mode=ablated` profile
- it is not an exact historical PI0.5 CALVIN parity run
- the older maintained ablation run referenced below was trained before two
  training-definition bugfixes landed globally:
  - `action_horizon` has now been restored to `16` by default
  - `_CalvinTransitionSource` now samples segment-first instead of uniformly
    over all valid window starts
- the maintained runtime now also normalizes prompt-state inputs from the shared
  CALVIN `norm_stats.json` before PI0.5 prompt discretization, matching the
  reference PI0.5 preprocessing contract without changing PICF physical-core
  inputs
- prompt-state tokenization keeps the live CALVIN state dimensionality and does
  not pad the text prompt state to `action_dim = 32`; padding remains a later
  state/action tensor contract, matching the reference transform order
- the maintained current ablation profile still differs from historical PI0.5
  in several other ways:
  - `semantic_max_length=256`
  - `lr=2e-4`, `min_lr=2e-5`, `warmup_steps=600`
  - `unroll_steps=2`
  - `training_strategy=fsdp_full_shard`
- the preserved cloud `pi05_calvin_nosonata/abc_train_nosonata_full_ddp2` run
  used:
  - `action_horizon=16`
  - `max_token_len=200`
  - `warmup=10000`, `peak_lr=5e-5`, `end_lr=5e-5`
- the generic codebase `CosineDecaySchedule` default is a different reference:
  - `peak_lr=2.5e-5`, `decay_lr=2.5e-6`, `warmup_steps=1000`

So this evaluation recipe is valid for the current operational ablation
checkpoint, but it should not be over-interpreted as an exact old-PI0.5 parity
measurement.

Validated example checkpoint:

- `/mnt/checkpoints/picf_core/picf_core/picf_v22_ablated_pi05_30000_ckpt2500_print100_20260421_r2/2500`

Validated example result directory:

- `/mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1`

Important current cloud notes:

- some cloud images currently carry a damaged
  `/mnt/calvin/calvin_env/calvin_env/utils/utils.py`; if that file is broken,
  the validated workaround is to prepend a shadow patch package such as
  `/root/calvin_patch` to `PYTHONPATH`
- on the same cloud image, `cv2.VideoWriter` may fail when writing directly into
  `/mnt/...`; the validated workaround is to write videos to `/tmp/...` first
  and copy them back into `/mnt` after the evaluator completes
- if `/mnt` quota is exhausted, video generation may appear to run but final
  artifact persistence will still fail; check `df -h /mnt` before assuming the
  evaluator path is wrong

#### 6.1.4 Serve The Ablated Checkpoint

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=/root/openpi_posterior_vla_clean/src:/root/openpi_posterior_vla_clean/packages/openpi-client/src

/root/openpi/.venv/bin/python scripts/serve_picf_policy.py \
  --checkpoint /mnt/checkpoints/picf_core/picf_core/picf_v22_ablated_pi05_30000_ckpt2500_print100_20260421_r2/2500 \
  --device cuda:0 \
  --port 8000 \
  --picf-mode ablated
```

Operational note:

- startup is not instantaneous; wait for the server to bind before launching the
  evaluator
- on the validated cloud run, the server process stayed resident as:
  `scripts/serve_picf_policy.py --checkpoint .../2500 --device cuda:0 --port 8000 --picf-mode ablated`

Useful checks:

```bash
ss -ltnp | grep :8000
ps -ef | grep -E 'serve_picf_policy.py' | grep -v grep
```

#### 6.1.5 Run The Ablated 20-Sequence Evaluator With Video

```bash
cd /root/openpi_posterior_vla_clean

eval "$(/root/bin/micromamba shell hook -s bash)"
micromamba activate calvin38

export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/calvin_patch:/mnt/calvin/calvin_env:/root/openpi_posterior_vla_clean/src:/root/openpi_posterior_vla_clean/packages/openpi-client/src

python scripts/calvin/evaluate_picf_policy.py \
  --dataset_path /mnt/calvin_data/task_ABC_D \
  --eval_log_dir /tmp/picf_eval_2500_dir_local \
  --num_sequences 20 \
  --save_video \
  --video_dir /tmp/picf_eval_2500_localvideo
```

Notes:

- this recipe uses the wrapper `scripts/calvin/evaluate_picf_policy.py`; do not
  edit upstream CALVIN evaluator code just to change `NUM_SEQUENCES`
- `--num_sequences 20` is passed directly through the wrapper to the upstream
  evaluator
- the validated path writes videos into `/tmp/picf_eval_2500_localvideo` during
  runtime because that path is known-good for `cv2.VideoWriter` on the current
  cloud image

#### 6.1.6 Monitor Progress And Final Success Rate

While the evaluator is running:

```bash
tail -f /tmp/picf_eval_2500_localvideo.log
```

Count videos as they are produced:

```bash
watch -n 5 'find /tmp/picf_eval_2500_localvideo -maxdepth 1 -type f -name "*.mp4" | wc -l'
```

After the run finishes, inspect the final success summary:

```bash
grep -A20 'Results for Epoch' /tmp/picf_eval_2500_localvideo.log
```

Validated example final summary for checkpoint `2500` in ablated mode:

- `Average successful sequence length: 0.0`
- `1: 0.0%`
- `2: 0.0%`
- `3: 0.0%`
- `4: 0.0%`
- `5: 0.0%`

That run completed all `20/20` sequences and produced `20` rollout videos.

#### 6.1.7 Copy Videos And Logs Back Into `/mnt`

After the evaluator finishes successfully:

```bash
base=/mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1
mkdir -p "$base/videos" "$base/logs"

cp -f /tmp/picf_eval_2500_localvideo/*.mp4 "$base/videos"/
cp -f /tmp/picf_eval_2500_localvideo.log "$base/logs/eval_calvin38_full.log"
cp -f /tmp/picf_eval_2500_localvideo.log "$base/eval_calvin38_full.log"
```

To also preserve a single-file archive:

```bash
mkdir -p /root/calvin_eval_videos/picf_v22_ablated_step2500_20260422
cp -f /tmp/picf_eval_2500_localvideo/*.mp4 /root/calvin_eval_videos/picf_v22_ablated_step2500_20260422/

cd /root/calvin_eval_videos
tar -czf picf_v22_ablated_step2500_20260422.tar.gz picf_v22_ablated_step2500_20260422
cp -f /root/calvin_eval_videos/picf_v22_ablated_step2500_20260422.tar.gz "$base"/
```

Useful final checks:

```bash
find /mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1/videos -maxdepth 1 -type f -name '*.mp4' | wc -l
```

```bash
tail -f /mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1/logs/eval_calvin38_full.log
```

```bash
grep -A20 'Results for Epoch' /mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1/logs/eval_calvin38_full.log
```

The validated completed artifact set under `/mnt` is:

- `/mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1/videos`
- `/mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1/logs/eval_calvin38_full.log`
- `/mnt/checkpoints/picf_core/eval/picf_v22_ablated_step2500_eval20_video_20260422_r1/picf_v22_ablated_step2500_20260422.tar.gz`

## 7. Historical Notes

Older CALVIN README fragments have been retired.
Use this file for the current validation workflow.

# CALVIN Validation README

Date: 2026-04-14

This document is the current executable validation guide for CALVIN under the
present PICF / sidecar codebase. It is intentionally narrower than the large
historical audit notes in `docs/calvin_readme.txt`.

It covers:
- dataset and loader validation
- core / trainer smoke validation
- sidecar contract validation
- checkpoint serving
- CALVIN evaluator rollout

It does not try to be a full research log.

## 1. Paths And Environments

Local workstation examples in this repo currently use:
- dataset dir: `/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D`
- dataset zip: `/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip`

Cloud evaluator examples used earlier in this project use:
- dataset dir: `/mnt/calvin_data/task_ABC_D`
- CALVIN repo: `/mnt/calvin/calvin_models/calvin_agent`
- eval env: `micromamba activate calvin38`

Use the path set that matches the machine you are on. Do not mix them.

## 2. Dataset / Loader Validation

### 2.1 Dataset structure check

Recommended:

```bash
python scripts/stageb_calvin_audit.py \
  --mode dataset \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split validation
```

Zip check:

```bash
python scripts/stageb_calvin_audit.py \
  --mode dataset \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip \
  --backend zip \
  --split validation
```

What this validates:
- split exists
- language annotations are readable
- episode indexing is sane

### 2.2 Loader check

Recommended:

```bash
python scripts/stageb_calvin_audit.py \
  --mode loader \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split validation \
  --batch-size 4 \
  --num-workers 0
```

Zip variant:

```bash
python scripts/stageb_calvin_audit.py \
  --mode loader \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip \
  --backend zip \
  --split validation \
  --batch-size 4 \
  --num-workers 0
```

What this validates:
- batch materialization
- prompt/state/image/action fields
- data transform path

## 3. Current PICF Contract Validation

Run the contract verifier before training changes or major deployment:

```bash
python scripts/verify_picf_contract.py
```

Current expected summary:
- `Targeted Invariance Regressions`: pass
- `Core Regression Suite`: pass
- `Smoke Training Check`: pass
- final summary: `PASS`

What this validates:
- semantic does not affect physical posterior
- semantic does not affect `physical_prediction_cache`
- next innovation reads only `previous.predictive.physical_prediction_cache`
- sidecar reads full `fused_tokens`
- control explicitly depends on `posterior.global_post`

## 4. Trainer Smoke Validation

### 4.1 Minimal local smoke

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cpu
```

CUDA smoke:

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cuda
```

### 4.2 Foundation-backbone smoke

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile
```

What this validates:
- one-step `forward + loss + backward + optimizer.step()`
- V-JEPA / Sonata / AnyTouch / semantic path wiring
- action normalization path

## 5. Sidecar Runtime Validation

Before a real sidecar training run, verify the core with:

```bash
python scripts/verify_picf_contract.py --skip-full-suite
```

And ensure the following conceptual invariants remain true:
- prompt changes may change:
  - sidecar outputs
  - action
  - conditioned future readout
- prompt changes must not change:
  - physical observation anchors
  - physical posterior
  - `physical_prediction_cache`
  - next-step innovation

## 6. Starting Training

### 6.1 Fresh training

Sidecar-only current recommended path:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile \
  --task-anchor-sidecar-enabled \
  --no-legacy-semantic-prefix-enabled
```

### 6.2 Resume training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_resume_train.py \
  --args-json /path/to/args.json \
  --resume-checkpoint /path/to/checkpoint_dir \
  --exp-name your_run_name \
  --device cuda \
  --task-anchor-sidecar-enabled \
  --no-legacy-semantic-prefix-enabled
```

### 6.3 Gradient clipping note

Current trainer default is:
- `grad_clip_mode=percentile`
- `grad_clip_percentile=75`
- `grad_clip_window=100`

So the trainer:
- does not clip at the beginning
- starts clipping only after the history window fills
- clips only the largest 25 percent of recent gradient norms

If you want to be explicit:

```bash
--grad-clip-mode percentile --grad-clip-percentile 75 --grad-clip-window 100
```

## 7. Serving A Checkpoint

Current serving path:

```bash
python scripts/serve_picf_policy.py \
  --checkpoint /path/to/checkpoint_dir \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000
```

What this does:
- rebuilds runtime args from checkpoint metadata
- rebuilds the trainer/core
- restores state with compatibility loading
- runs the core in normalized action space
- unnormalizes the action before returning it

## 8. CALVIN Evaluator Rollout

### 8.1 Server terminal

```bash
python scripts/serve_picf_policy.py \
  --checkpoint /path/to/checkpoint_dir \
  --device cuda:0 \
  --port 8000
```

### 8.2 Evaluator terminal

Cloud-style example:

```bash
eval "$(/root/bin/micromamba shell hook -s bash)"
micromamba activate calvin38

unset DISPLAY
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0
export OPENPI_SERVER_HOST=127.0.0.1
export OPENPI_SERVER_PORT=8000
export OPENPI_EVAL_TAG=custom
unset CALVIN_SAVE_VIDEO
unset CALVIN_VIDEO_DIR

cd /mnt/calvin/calvin_models/calvin_agent
python evaluation/evaluate_policy.py \
  --dataset_path /mnt/calvin_data/task_ABC_D \
  --custom_model \
  --eval_log_dir /mnt/calvin_eval_logs/custom \
  --device 0
```

### 8.3 Video rollout variant

```bash
export CALVIN_SAVE_VIDEO=1
export CALVIN_VIDEO_DIR=/mnt/calvin_eval_logs/custom/videos
mkdir -p /mnt/calvin_eval_logs/custom/videos
```

Then run the same evaluator command.

## 9. Acceptance Checklist

Before calling a training or evaluation setup valid, check all of:

1. Dataset audit passes.
2. Loader audit passes.
3. `python scripts/verify_picf_contract.py` passes.
4. `picf_core_train_smoke.py` passes on the target machine.
5. If using sidecar:
   - sidecar enabled
   - legacy semantic prefix disabled unless you intentionally want rollback
6. If serving:
   - checkpoint loads
   - actions are returned in environment scale, not normalized scale
7. If evaluating:
   - evaluator can connect to server
   - `results.json` is written
   - video path only appears when `CALVIN_SAVE_VIDEO=1`

## 10. Relationship To Other Docs

- Use this file for current validation and rollout.
- Use `src/openpi/picf/README_task_anchor_sidecar_followthrough.md` for the
  full mathematical and dataflow handoff.
- Use `docs/calvin_readme.txt` only as a long historical audit record.

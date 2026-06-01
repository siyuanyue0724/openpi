# PICF V-JEPA Cache And 200-Step Action-Weight Gate - 2026-05-23

Canonical parent documents:

```text
src/openpi/picf/README_v2.2.md
docs/picf_aqr_owm_202605/README.md
docs/PICF_AQR_OWM_SCALABILITY_AND_CALVIN_REPAIR_PLAN_20260523.md
```

## 1. Decision

Production token budget remains:

```text
semantic_max_length = 256
```

This gate deploys the first real frozen-feature cache:

```text
V-JEPA cache mode: off | read | read_or_encode
cache target: frozen V-JEPA recent temporal feature-map suffix used by PICF
default production test mode: read_or_encode
default diagnostic suffix: last 2 temporal maps for the current AQR profile
default storage dtype: bfloat16
```

It does **not** cache:

```text
posterior state
slot ownership
belief updates
action outputs
trainable PaliGemma embeddings
```

## 2. Mathematical Contract

For frozen V-JEPA parameters `phi_v` and fixed preprocessing `p`, the visual
feature used by PICF is deterministic:

```math
z_t = f_{\phi_v}(p(x_{\le t})).
```

PICF does not consume the full V-JEPA temporal volume.  It consumes:

```math
g(z_t) = (\operatorname{current\_map}(z_t), \operatorname{recent\_maps}_k(z_t)).
```

The cache substitutes:

```math
g(f_{\phi_v}(p(x_{\le t}))) \rightarrow
C[\operatorname{hash}(x_{\le t}, \phi_v, p, k)].
```

The objective is unchanged only when:

```math
C[k] = g(f_{\phi_v}(p(x_{\le t})))
```

and the key includes the frozen-backbone and preprocessing contract.  Therefore
the implementation uses a strict manifest with:

```text
model name
architecture name
checkpoint path / key / sha256
image size
frame count
patch / tubelet size
feature mode
stored temporal suffix length
storage dtype
normalization mean/std
clip hash
source image size
resized image size
```

Invalid or stale entries fail closed in `read` mode.  `read_or_encode` reads
valid entries and writes misses atomically.

## 3. Implementation

Code paths:

```text
src/openpi/picf/vjepa/config.py
src/openpi/picf/vjepa/wrapper.py
scripts/picf_core_train.py
```

CLI:

```text
--vjepa-feature-cache-root PATH
--vjepa-feature-cache-mode off|read|read_or_encode
```

Environment-backed A7 launch variables:

```text
VJEPA_FEATURE_CACHE_ROOT=/mnt/picf_frozen_feature_cache/vjepa2_1_base384_hierarchical
VJEPA_FEATURE_CACHE_MODE=read_or_encode
VJEPA_FEATURE_CACHE_TEMPORAL_SLICES=2
VJEPA_FEATURE_CACHE_STORAGE_DTYPE=bfloat16
```

The implementation rejects cache use when V-JEPA is trainable.  This is
intentional: once `phi_v` receives gradients, cached features are stale and
mathematically invalid.

Rejected cache variant:

```text
full dense-volume cache:
  rejected after cloud smoke exposed 216 MB per cache entry and about 110 GB
  written before step50.

reason:
  the full V-JEPA 32-slice hierarchical feature volume is not the PICF runtime
  sufficient statistic.  Storing it is mathematically conservative but
  operationally wrong: cold-cache I/O dominates step time and can be slower
  than direct encoding.

repair:
  suffix-cache only the recent temporal maps consumed by PICF and store them
  in bfloat16/float16.  For the maintained AQR profile this is last 2 maps.
```

## 4. 200-Step Comparison Plan

Both runs use the same slot/OEML/owner/context/sidecar contract and differ only
in action weight:

```text
Run A:
  picf_a7_vjepa_cache_action05_smoke200_20260523
  ACTION_LOSS_WEIGHT=0.50

Run B:
  picf_a7_vjepa_cache_action2_smoke200_20260523
  ACTION_LOSS_WEIGHT=2.00
```

Launch scripts:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_vjepa_cache_action05_smoke200_20260523.sh
scripts/experiments/picf_aqr_owm_202605_active/run_a7_vjepa_cache_action2_smoke200_20260523.sh
```

Metrics to compare:

```text
loss_action_default_equiv
loss_action_active7
loss_anchor_object_pull
loss_slot_quality
loss_object_explanation_point
loss_object_explanation_duplicate
aqr_same_role_support_overlap_max
aqr_active_same_role_support_overlap_max
aqr_downstream_same_role_support_overlap_max
posterior_owner_active_rate
posterior_recycle_rate
steps_per_second
```

Interpretation:

```text
ACTION=2.0 is better only if action_default_equiv drops faster without
materially worsening active/downstream overlap, object pull, or owner active
stability.

Raw same-role overlap is telemetry for reserve/context rows.  It is not a
failure if active/downstream overlap remains controlled.
```

Historical same-math comparison before enabling the V-JEPA cache path:

```text
metric at step200                         action=0.50     action=2.00
loss_action_default_equiv                 0.0616          0.0598
loss_action_active7                       0.2783          0.2712
loss_total_minus_action                   0.1503          0.2328
loss_anchor_object_pull                   0.2801          0.4773
loss_object_explanation_point             2.0865          2.7311
loss_object_explanation_duplicate         0.0933          0.1143
aqr_active_same_role_support_overlap_max  0.0146          0.1165
aqr_downstream_same_role_support_overlap  0.1537          0.2977
posterior_recycle_rate                    0.1094          0.0372
```

Reading:

```text
Starting at action=2.0 gives only a tiny step200 action advantage, but produces
higher non-action pressure and worse owner/overlap structure.  This is why the
maintained long-run profile uses either staged action pressure or scaffold
decay rather than blindly starting all runs at the legacy action scale.
```

## 5. Local Verification

Passed:

```bash
bash -n \
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh \
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_vjepa_cache_action05_smoke200_20260523.sh \
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_vjepa_cache_action2_smoke200_20260523.sh

PYTHONPATH=src uv run python -m py_compile \
  src/openpi/picf/vjepa/config.py \
  src/openpi/picf/vjepa/wrapper.py \
  scripts/picf_core_train.py

PYTHONPATH=src uv run python -m pytest src/openpi/picf/vjepa/wrapper_test.py -q
```

Result:

```text
12 passed
```

## 6. Cloud Results

Running on A7:

```text
tmux:
  picf_cache_seq_200_0523

sequence:
  1. picf_a7_vjepa_cache_action05_smoke200_20260523
  2. picf_a7_vjepa_cache_action2_smoke200_20260523

hardware contract:
  2xA100, FSDP full shard, effective_global_batch=2
```

Rejected launch attempt:

```text
single-GPU parallel comparison:
  rejected by CUDA OOM at optimizer.step

reason:
  trainable PaliGemma + PICF has about 3.8B trainable parameters, and Adam
  optimizer state does not fit in one A100-40GB under this profile.

conclusion:
  action-aware comparisons must use the 2xA100/FSDP contract.  Single-GPU
  parallel runs are not a valid shortcut for this gate.
```

Current status:

```text
2026-05-23 16:49:
  full dense-volume cache run stopped and cache root cleaned.

2026-05-23 16:55:
  suffix-cache action=0.50 run restarted.
  startup log confirms:
    vjepa_temporal_slices=2
    vjepa_storage_dtype=bfloat16

initial engineering check:
  full cache entry size: about 216 MB
  suffix cache entry size: about 6.75 MB
  cold-cache step time: about 25 sec/step after repair
```

Timing gate after suffix-cache repair:

```text
profile:
  2xA100, FSDP full shard, effective_global_batch=2
  semantic_trainable=True
  trainable_numel ~= 3.806B
  unroll_steps=2, burnin_steps=1, effective_window_steps=3
  semantic_max_length=256
  owm_temporal_visual_tokens=3456
  V-JEPA suffix cache stores last 2 temporal maps as bfloat16

cache-off:
  exp: picf_a7_timing_cacheoff_30_20260523
  stopped after step17 once step10 metrics existed
  step10 metrics sec/step: 24.65
  tqdm last/last5 around stop: 24.71 / 24.67 s/it

cache-read:
  exp: picf_a7_timing_cacheread_10_20260523
  completed 10 steps
  step5 metrics sec/step: 20.43
  step10 metrics sec/step: 20.48
  tqdm final: 20.44 s/it

observed gain:
  about 16.9% faster at step10:
    1 - 20.48 / 24.65 ~= 0.169
```

Interpretation:

```text
The corrected suffix cache is useful but not a 2x accelerator for this
profile.  It removes repeated frozen V-JEPA clip encoding, but the step is still
dominated by trainable PaliGemma, FSDP full-shard synchronization, unrolled
window cost, PICF typed routing/posterior computation, and action-head backward.

The earlier 10-13 s/step observation came from a lighter profile
(picf_a7_pi05tok200_frozen_policy_100_20260523, about 12.23 s/step last5)
that is not comparable: it used a smaller/frozen-policy diagnostic path rather
than this action-aware PaliGemma-trainable profile.

Therefore V-JEPA caching should be treated as a warm-cache optimization for
repeated sweeps and full-dataset precompute, not as the primary solution for
bringing normal training close to PI0.5 speed.
```

Tail commands:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/picf_run_logs/picf_a7_vjepa_cache_action05_smoke200_20260523.log
tail -f /mnt/picf_run_logs/picf_a7_vjepa_cache_action2_smoke200_20260523.log
```

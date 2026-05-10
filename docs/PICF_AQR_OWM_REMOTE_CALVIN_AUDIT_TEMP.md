# PICF-AQR-OWM Remote CALVIN Audit Temp

Purpose: independent audit of the previous CALVIN run artifacts copied from the test machine. This file separates old-run empirical failures from the current code-level deployment checks.

Remote source:

```text
host:
  px-cloud2.matpool.com:28060

run:
  /mnt/checkpoints/picf_core/picf_core/
  picf_v22_aqr_8fdb16f_pgimg_noheat_strict2x40_unroll2_30000_ckpt2500_progress_20260509_r1

eval:
  /mnt/checkpoints/picf_core/eval/
  picf_v22_aqr_8fdb16f_step2500_eval20_anchor_20260510_095404

local copy:
  /tmp/picf_remote_audit_8fdb16f
```

## 1. Run Identity

The copied `args.json` shows this was an older AQR validation run, not the current cleaned PICF-AQR-OWM checkout:

```text
aqr_mapg_enabled: true
mapg_enabled: false
aqr_pg_grounding_enabled: false
aqr_pg_image_support_enabled: true
aqr_pg_bias_weight: 0.0
burnin_mode: full
burnin_steps: 0
effective_unroll_steps: 2
lambda_focus_pv: 0.0
lambda_mapg_cycle: 0.004
aqr_temporal_memory_tokens: 32
```

Interpretation:

```text
This run predates the current cleanup:
  focus_pv still existed as a dead logged loss,
  aqr_temporal_memory_tokens still existed as a misleading knob,
  mapg_cycle was weaker than the current 0.02 default,
  current OWM debug keys were absent from metrics.
```

Therefore its failure is valid evidence of the old anchor problem, but it is not evidence that the current code remains broken.

## 2. Strict Diagnosis Result

Command:

```bash
python scripts/picf_owm_strict_diagnose.py \
  --metrics-jsonl /tmp/picf_remote_audit_8fdb16f/metrics.jsonl \
  --eval-dir /tmp/picf_remote_audit_8fdb16f \
  --markdown-out /tmp/picf_owm_strict_diagnosis_old_calvin.md \
  --json-out /tmp/picf_owm_strict_diagnosis_old_calvin.json
```

Result:

```text
3 FAIL, 2 WARN, 18 PASS/INFO
```

Failures:

```text
runtime_anchor_pv_not_worsening:
  first-quartile mean = 3.92823
  last-quartile mean  = 4.74358
  ratio               = 1.208

calvin_same_role_overlap_not_collapsed:
  same_role_visual_overlap_max mean = 1.00
  same_role_point_overlap_max  mean = 1.00

calvin_posterior_anchor_jump_reasonable:
  posterior pixel mean jump = 15.030 px
```

Warnings:

```text
runtime_current_run_has_owm_debug_keys:
  none found

runtime_same_role_support_pressure_low:
  last-quartile support-diversity loss = 0.586772
```

## 3. Loss Trend Audit

Metrics rows:

```text
count: 46
first step: 100
last step: 4600
```

Main trends:

```text
loss_action:
  first mean = 0.078408
  last mean  = 0.0532343
  verdict    = improved

loss_pv_weak:
  first mean = 4.48513
  last mean  = 2.89736
  verdict    = improved

loss_anchor_pv:
  first mean = 3.92823
  last mean  = 4.74358
  verdict    = worsened

loss_mapg_cycle:
  first mean = 0.501784
  last mean  = 0.607386
  verdict    = worsened

loss_mapg_support_diversity:
  first mean = 0.409554
  last mean  = 0.586772
  verdict    = worsened

loss_mapg_geometry_diversity:
  first mean = 0.479976
  last mean  = 0.762493
  verdict    = worsened
```

Interpretation:

```text
Action imitation improved, but anchor geometry/support quality did not improve.
The old run learned action-relevant behavior faster than it learned stable object-addressed anchors.
```

This exactly matches the observed videos: poor drawer success at 2500 steps is not surprising, but anchor collapse/jump is a structural diagnostic and cannot be dismissed as only low-step policy immaturity.

## 4. CALVIN Anchor Diagnostics

From `anchor_drift_diag_ep0_ep1.txt`:

```text
EP0 push the handle to close the drawer:
  task.pixel jump mean       = 7.86
  posterior.pixel jump mean  = 15.03
  obs.pixel jump mean        = 15.68
  same_role_visual_overlap   = 1.00
  same_role_point_overlap    = 1.00

EP1 go push the blue block right:
  task.pixel jump mean       = 6.58
  posterior.pixel jump mean  = 12.20
  obs.pixel jump mean        = 12.72
  same_role_visual_overlap   = 1.00
  same_role_point_overlap    = 1.00
```

From `anchor_trend_diag_ep0_ep1.txt`:

```text
EP0 posterior quartile jumps:
  12.85 -> 15.35 -> 15.81 -> 16.09

EP1 posterior quartile jumps:
  13.58 -> 10.84 -> 11.29 -> 13.08
```

Interpretation:

```text
EP0 gets worse through the episode.
EP1 improves briefly but remains high and rebounds.
Neither episode shows reliable posterior anchor convergence.
```

## 5. Collapse Evidence

From selected evidence bundle:

```text
role1_unique_centers_median = 1.0
role1_unique_centers_min    = 1
role1_unique_centers_max    = 1
role1_frac_unique1          = 1.0
```

This means every role-1 observation row collapsed to one unique center in the selected evidence window.

Field entropy summary:

```text
vl_task entropy mean        = 0.99669
vl_effector entropy mean    = 0.99644
vl_interaction entropy mean = 0.99677

mapg_visual entropy mean    = 0.67510
mapg_point entropy mean     = 0.62057
```

Interpretation:

```text
The direct PaliGemma/VL heatmaps were near-uniform and not a reliable where source.
The graph MAPG supports were more structured than VL heatmaps, but same-role rows still collapsed after assignment.
```

## 6. Relation To Current Code

Current code-level fixes directly address the old-run failure modes:

```text
old: focus_pv existed but was zero/dead
new: focus_pv removed; graph PV consistency uses bidirectional projected supports

old: aqr_temporal_memory_tokens existed but was not the active temporal V-JEPA control
new: misleading knob removed; recent temporal maps are consumed by AQR

old: mapg_cycle weight was 0.004
new: mapg_cycle default is 0.02

old: OWM debug keys absent
new: temporal/PG/cache/identity/support diagnostics are in trainer metrics

old: same-role overlap reached 1.00
new: support-diversity pressure remains active and strict diagnose checks overlap artifacts when supplied
```

## 7. Unresolved Empirical Item

The old CALVIN run proves the old anchor problem. It does not prove the current code is empirically fixed.

Required next evidence:

```text
Train/evaluate the current checkout with:
  explicit AQR enabled,
  current PV/support defaults,
  temporal V-JEPA active,
  first-class PG priors active,
  current OWM debug metrics logged.

Then run:
  python scripts/picf_owm_strict_diagnose.py \
    --metrics-jsonl <new_run>/metrics.jsonl \
    --eval-dir <new_eval_dir> \
    --markdown-out <new_run>/owm_strict_diagnosis.md \
    --json-out <new_run>/owm_strict_diagnosis.json \
    --fail-on-fail
```

Until that new-run evidence exists, the correct statement is:

```text
Code-level blockers found in the old review are fixed.
The previous 2500-step CALVIN result remains a failing baseline.
Empirical anchor quality for the current checkout is not yet 100% proven.
```

# PICF-AQR-OWM Action Platform Root-Cause Map - 2026-05-30

Canonical related notes:

```text
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
docs/PICF_AQR_OWM_CURRENT_PLATEAU_ROOT_CAUSE_PLAN_20260529_TEMP.md
docs/PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md
docs/PICF_AQR_OWM_ANCHOR_LR_SAMPLING_CAUSAL_PLAN_20260530_TEMP.md
docs/PICF_AQR_OWM_4_22_ABLATION_BASELINE_ARCHIVE_20260530.md
```

## Question

Why does the corrected PICF branch sit around `loss_action_default_equiv ~= 0.04`
instead of reproducing the old apparent `0.02` online train band?

## Non-Negotiable Metric Separation

There are three different objects:

```math
online row:
  \hat L_k = |B_k|^{-1}\sum_{w\in B_k} L(\theta_k;w)

same-checkpoint exact-window probe:
  \hat J_S(\theta) = |S|^{-1}\sum_{w\in S} L(\theta;w)

behavior:
  CALVIN / video success
```

The old `0.02` rows are online rows.  They are not fixed-window checkpoint
quality.  Current fixed-window results must not be compared directly with old
online rows.

## Evidence Ledger

### E0. 4-22 baseline is not a single scalar

From `PICF_AQR_OWM_4_22_ABLATION_BASELINE_ARCHIVE_20260530.md`:

```text
4-22 online train rows:
  late rows can hit 0.014..0.021, but nearby rows can be much higher.

4-22 fixed-window probe:
  step7500  = 0.058717
  step10000 = 0.053813
  step20000 = 0.051038
```

Interpretation:

```text
The old 0.02 band is real as online train loss, but it is not a stationary
fixed-window checkpoint-quality target.
```

### E1. Legacy resume/RNG replay explained the repeated 7550/7600 spike

From `PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md`:

```text
action-head-only, policy-semantic-only, and fixed-group runs matched almost
point-by-point around 7050..7600.

near-zero-LR replay reproduced:
  step7550 = 0.036840
  step7600 = 0.037180
```

Conclusion:

```text
The historical spike is not caused by PICF/cotrain/semantic drift in that
experiment family.  It is a train-window estimator spike under a resumed
sampled-window stream.
```

Code fix:

```text
scripts/picf_core_train.py now defaults to step-indexed window RNG keyed by
seed/rank/global_step/micro_step/retry.
```

### E2. Same-task-window explanation is not supported

From `PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md`:

```text
old resume-like reconstructed 19901..20000 records:
  rows = 200
  unique prompts = 151
  unique buckets = 25
  max prompt run = 1
  max bucket run = 3
```

Conclusion:

```text
The known old stream is mixed, not a long contiguous same-task curriculum.
```

### E3. Raw overlap is not the direct action-platform cause

From `PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md` and
`PICF_AQR_OWM_CURRENT_PLATEAU_ROOT_CAUSE_PLAN_20260529_TEMP.md`:

```text
raw same-role overlap can be 1.0, but active/downstream overlap stayed low.
policy-semantic-only had PICF frozen and non-action loss 0, yet reproduced the
late action band in the legacy replay family.
```

Conclusion:

```text
Raw reserve/context overlap is a watch metric, not the direct cure for the
action platform.
```

### E4. Non-action losses / sidecar scaffold are not sufficient explanations

Evidence:

```text
policy-semantic-only:
  loss_total_minus_action = 0
  still reproduced the historical bad band under the old replay stream.

current corrected runs:
  sidecar proposal/tracklet tokens are present;
  active/downstream overlap is controlled;
  object-pull can improve without action improving.
```

Conclusion:

```text
Sidecar or object-pull quality may affect final behavior, but the action
platform is not solved by adding/removing another object scalar.
```

### E5. FSDP optimizer grouping bug was real but not sufficient

Evidence:

```text
group logging now shows:
  lr_group_picf_core
  lr_group_policy_head
  lr_group_semantic_backbone
  grad_norm_group_*

plateau persists after group repair.
```

Conclusion:

```text
Correct grouping is required hygiene; it is not the whole root cause.
```

### E6. Action-visible prefix instability was real but not sufficient

Action-prefix EMA/RMS/gate reduced prefix drift:

```text
pi_prefix_teacher_cos_to_teacher ~= 0.99999
loss_action_prefix_trust tiny
active/downstream overlap healthy
```

Yet action still failed to improve decisively on corrected windows.

Conclusion:

```text
The gated/EMA prefix interface should be kept, but the remaining platform is
not only prefix drift.
```

### E7. Current corrected step8000 -> step9000 branch improved structure but not action

From `PICF_AQR_OWM_ANCHOR_LR_SAMPLING_CAUSAL_PLAN_20260530_TEMP.md`:

```text
same fixed64 windows:
  step8000 action = 0.057919
  step9000 action = 0.057213

structure:
  object_pull improved strongly;
  recycle improved;
  slot_jepa improved;
  active/downstream overlap mean worsened but median mostly unchanged.
```

Conclusion:

```text
The router/scaffold can improve while same-window action barely moves.  This is
the strongest direct evidence for action-transfer/interface or action-side
optimization bottleneck.
```

### E8. Current high-PICF-LR callback first rows

Run:

```text
picf_a7_stepindexed_from8000_picflr002_action2_30k_20260530
PICF_CORE_LR_SCALE = 0.02
```

First rows:

```text
step8050 action = 0.04334
step8100 action = 0.04408
object_pull: 0.318 -> 0.245
active overlap: 0.125 -> 0.100
downstream overlap: 0.119 -> 0.092
```

Interim conclusion:

```text
Higher PICF LR improves some structure signals but has not yet moved action.
This weakens "PICF LR too low" as the sole cause.
```

## Root-Cause Candidate Classification

### Strongly supported

1. **Online train-row estimator / window-mixture effect**

Evidence:

```text
near-zero-LR replay reproduced historical spike;
old 0.02 online rows differ from fixed-window checkpoint quality;
corrected step-indexed runs expose harder/more honest online windows.
```

This explains why old `0.02` should not be the immediate target for current
corrected live rows.

2. **Action-side optimizer / local-basin instability**

Evidence:

```text
fresh optimizer / phase reset can produce fast online action drops;
the drops repeatedly failed to stay stable;
policy/action branches can reproduce bad bands without PICF movement.
```

This remains the best explanation for "fast drop after reset, then plateau or
rise" once legacy RNG replay is accounted for.

3. **PICF-to-action transfer bottleneck**

Evidence:

```text
structure/object metrics can improve without same-window action improving;
prefix gating/EMA prevents gross prefix drift, but action still barely moves.
```

This is different from "anchor is bad."  It says the action model is not
extracting enough useful gradient from the improved belief state under the
current interface/training schedule.

### Plausible but not proven

4. **PICF core LR too low for new windows**

Evidence for:

```text
0.005 makes PICF nearly stationary;
current high-LR callback improves object_pull and overlap early.
```

Evidence against as sole cause:

```text
action did not improve at 8050/8100.
```

Status:

```text
Keep only as a controlled experiment.  Stop if action does not improve by
8200/8250 or fixed-window probe.
```

5. **Semantic/action representation adaptation bottleneck**

Evidence:

```text
PaliGemma backbone_only is trainable;
action rows remain high while structure is healthy.
```

But old policy-semantic/action-head-only records were contaminated by old
resume-window replay, so a corrected step-indexed split is still needed.

6. **Checkpoint basin issue**

Evidence:

```text
step7000/8000 branch may be in a basin where structure can improve but action
cannot transfer.
```

Needs:

```text
compare from earlier checkpoint or from scratch under step-indexed RNG and same
fixed-window probes.
```

### Weakened / not primary

```text
raw same-role support overlap;
missing sidecar data;
SAM/noisy proposals;
slot-JEPA raw explosion as primary cause;
FSDP LR grouping;
action metric display bug;
simple "same-task consecutive window" explanation.
```

## Required Next Diagnostic Matrix

Do not repeat old non-step-indexed experiments as decisive evidence.  Use the
corrected sampler and fixed-window probes.

### D1. Corrected action-head-only from step8000

```text
resume step8000;
PICF frozen;
PaliGemma frozen;
action head trainable only;
step-indexed RNG;
same log cadence.
```

Question:

```text
Can the current checkpoint's action head alone reduce the corrected online
windows?
```

If yes:

```text
platform is caused by cotrain/interface pressure.
```

If no:

```text
checkpoint/data/action target is hard even for direct action fitting.
```

### D2. Corrected policy-semantic-only from step8000

```text
PICF frozen;
PaliGemma backbone_only + action/policy trainable;
non-action losses 0;
step-indexed RNG.
```

Question:

```text
Does semantic/action adaptation break through where action-head-only cannot?
```

### D3. Current high-PICF-LR branch gate

```text
continue only to 8200/8250 unless action begins moving;
then fixed-window probe against step8000.
```

Question:

```text
Does faster PICF adaptation transfer to action?
```

### D4. Stratified fixed-window probes

Buckets:

```text
block
drawer
slider
button/switch/light
mixed balanced
```

Question:

```text
Is action stuck globally or only on specific task families?
```

This is the necessary diagnostic before another architecture/loss rewrite.

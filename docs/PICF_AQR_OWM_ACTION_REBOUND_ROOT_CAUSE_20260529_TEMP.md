# PICF-AQR-OWM Action Rebound Root-Cause Follow-Through - 2026-05-29

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_REBOUND_DEEP_AUDIT_20260528_TEMP.md
docs/PICF_AQR_OWM_ACTION_REBOUND_CAUSAL_PLAN_20260528.md
docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md
```

## 1. Current Verdict

The fixed FSDP optimizer-group repair is a real code fix, but it is not the
full action-rebound root cause.  The strict replay from the preserved step7000
checkpoint now reproduces the late action-specific rebound:

```text
run:
  picf_a7_reboundgate_ema_recipe_scopefix_lrgroupfix_from7000_30k_20260528

step7550:
  loss_action_default_equiv          0.035953
  loss_action_active7                0.163119
  loss_total_minus_action            0.011119
  active/downstream overlap          0.105000 / 0.109954
  lr_group_picf_core                 3.13073e-7
  lr_group_semantic_backbone         2.19151e-5
  lr_group_policy_head               6.26146e-5

step7600:
  loss_action_default_equiv          0.035442
  loss_action_active7                0.160621
  loss_total_minus_action            0.011023
  active/downstream overlap          0.060000 / 0.066035
  lr_group_picf_core                 3.12607e-7
  lr_group_semantic_backbone         2.18825e-5
  lr_group_policy_head               6.25214e-5

step7650:
  loss_action_default_equiv          0.038672
  loss_action_active7                0.175419
  loss_total_minus_action            0.010442
  active/downstream overlap          0.040000 / 0.047484
  loss_slot_jepa                     1.317115
  grad_norm                          0.268751
```

The strict two-row rejection rule is met:

```text
loss_action_default_equiv >= 0.035 at both step7550 and step7600.
```

The failure is still action-specific:

```text
loss_total_minus_action stays near 0.011;
active/downstream overlap stays far below the 0.25 structural reject gate;
PICF core LR is verified at 0.005x policy-head LR under FSDP.
```

Therefore the remaining root cause is not a missing raw-overlap patch, not
unverified FSDP grouping, and not broad structural-loss explosion.  Continuing
this exact run after step7650 is low-information: it repeatedly confirms the
same action-side rebound while lacking the window trace needed to distinguish
semantic/action optimizer drift from hard data-window exposure.

## 2. Locked Evidence Table

Comparable scalar: `loss_action_default_equiv`.

```text
step    old_ema   phase_p2  policy_head_only  policy_semantic_only  fixed_group
7000    0.02127   0.02130   n/a               n/a                   n/a
7050    0.02610   0.02602   0.02294           0.02248               0.02260
7100    0.02663   0.02644   0.02577           0.02559               0.02527
7150    0.02768   0.02776   0.02185           0.02273               0.02262
7200    0.02684   0.02679   0.01794           0.01845               0.01820
7250    0.02785   0.02792   0.02695           0.02636               0.02618
7300    0.03077   0.03056   0.02713           0.02703               0.02674
7350    0.03591   0.03589   0.02824           0.02749               0.02747
7400    0.03533   0.03491   n/a               0.02691               0.02674
7450    0.03795   0.03808   n/a               0.02796               0.02794
7500    0.03857   n/a       n/a               0.03076               0.03083
7550    0.03389   n/a       n/a               0.03628               0.03595
7600    0.03596   n/a       n/a               n/a                   0.03544
7650    n/a       n/a       n/a               n/a                   0.03867
8000    0.04371   n/a       n/a               n/a                   n/a
8300    0.04701   n/a       n/a               n/a                   n/a
```

Important separation:

```text
policy_head_only:
  fixed PICF + frozen semantic backbone + action head only.
  It has only been observed through step7350, where it remains below 0.029.

policy_semantic_only:
  fixed PICF + trainable semantic/action side + non-action loss 0.
  It reaches 0.03628 at step7550.

fixed_group cotrain:
  trainable semantic/action side + verified slow PICF core + weak structural
  terms.  It reaches 0.03595/0.03544 at step7550/7600.
```

This hierarchy is the strongest causal evidence so far:

```text
semantic/action-side training is sufficient to reproduce the late rebound even
when PICF core is frozen and non-action loss is zero.

PICF cotrain is not needed for the rebound to exist, although it can shift or
amplify the timing.
```

## 3. Falsified or Weakened Hypotheses

### 3.1 Raw Same-Role Overlap

Raw overlap remains saturated in many runs, but action rebound appears while:

```text
aqr_active_same_role_support_overlap_max <= about 0.105
aqr_downstream_same_role_support_overlap_max <= about 0.110
```

and while policy-semantic-only has:

```text
loss_total_minus_action = 0.0
PICF core frozen
```

Conclusion:

```text
Raw reserve/context overlap is not the direct late-action rebound cause.
It remains a watch metric, not the intervention target.
```

### 3.2 Weak Object/Sidecar Loss Budget

In the fixed-group run:

```text
loss_total_minus_action ~= 0.0095-0.0116
```

In the policy-semantic-only run:

```text
loss_total_minus_action = 0.0
```

Both can enter the rebound band.  Therefore adding or removing another object
loss is not a mathematically direct cure for this failure.

### 3.3 PICF Core LR Grouping Alone

The FSDP grouping bug is fixed and verified:

```text
lr_group_picf_core ~= 3.13e-7
lr_group_policy_head ~= 6.25e-5
grad_norm_group_picf_core is finite and tiny
```

The branch still rebounds at 7550/7600.  Slow PICF core is necessary hygiene,
not sufficient cure.

### 3.4 Action Metric Logging Bug

`loss_action_default_equiv` and `loss_action_active7` move together.  The
component-level action scalar confirms that this is not a display-only issue.

## 4. Mathematical Root-Cause Model

Let `s_t` be the action-visible PICF prefix/context and `u_t` the semantic/action
state.  The action objective is:

```math
L_a(\theta; x_t, s_t, a_t)
=
\ell(\pi_\theta(x_t, s_t), a_t).
```

Near the step7000 low-loss basin, one stochastic update changes expected action
loss approximately by:

```math
\Delta L_a
\approx
- \eta \|\nabla_\theta L_a\|^2
+ \frac{1}{2}\eta^2 \operatorname{Tr}(H_\theta \Sigma_\theta)
+ \nabla_s L_a^\top \Delta s
+ \Delta_{\text{data-window}}.
```

Evidence removes several terms as necessary causes:

```text
PICF core motion term:
  not necessary, because policy_semantic_only has fixed PICF and still rebounds.

non-action structural-loss term:
  not necessary, because policy_semantic_only has loss_total_minus_action=0.

raw-overlap term:
  not necessary, because action-visible overlap stays controlled.
```

The remaining terms are:

```text
1. semantic/action-side optimizer noise or curvature in an already-low basin;
2. semantic backbone representation drift under action loss;
3. a data-window / task-window shift around the 7500-7600 region;
4. action-head-only capacity or LR limits, still untested past 7350.
```

The current data cannot honestly collapse these into one final root cause yet.
The decisive missing counterfactual is action-head-only through 7600/8000 plus
explicit sample/window tracing.

## 5. Code Follow-Through

### 5.1 Action Target Path

The training window uses a pending-transition pattern:

```text
for transition i:
  forward current frame i with action_chunk_target=current.action_chunk
  compute previous pending loss using current observation as next target
  store current output as pending

after loop:
  compute final pending loss against its stored next_observation
```

For the PI0.5 flow action path:

```text
scripts/picf_core_train.py::_PicfWindowTrainer.forward
  action_chunk_target = current.action_chunk if available else current.action
  policy.forward_train_transition(... action_chunk_target=...)
  flow_override["total"] -> compute_transition_loss(action_loss_override=...)
```

Inside `compute_transition_loss`:

```text
action_default_equiv = action_loss_override
action = weighted action override, equal to default when lambdas are 2/2/2
```

No action-target off-by-one is currently evident in this path.

### 5.2 Optimizer Group Path

FSDP/DDP owner prefixes are canonicalized before assigning params:

```text
_fsdp_wrapped_module.core.* -> core.*
module.core.*               -> core.*
```

Runtime logs confirm:

```text
lr_group_picf_core is present;
grad_norm_group_picf_core is present;
PICF core LR is 0.005x policy-head LR.
```

So the current fixed-group run is valid for causal analysis.

## 6. Required Next Experiments

Do not add another slot/object module before these counterfactuals.  The current
failure is already action-side localized.

### E1. Extend Action-Head-Only to the Rebound Window

Purpose:

```text
Decide whether action head alone rebounds past step7550.
```

Protocol:

```text
resume step7000 checkpoint;
PICF_TRAINABLE_SCOPE=policy_only;
semantic_trainable_scope=action_head_only;
loss_total_minus_action=0;
run at least to step8000 with the same 30000-step LR horizon.
```

2026-05-29 execution decision:

```text
Stop:
  picf_a7_reboundgate_ema_recipe_scopefix_lrgroupfix_from7000_30k_20260528

Start:
  action-head-only trace run from the same preserved step7000 checkpoint.

Reason:
  The fixed-group run has already reproduced action rebound through step7650.
  It does not include window_trace metadata, so extra steps cannot resolve the
  remaining causal split.
```

Initial action-head-only trace validation:

```text
run:
  picf_a7_ema7000_policyonly_actionhead_trace_from7000_30k_20260529

configuration:
  resume checkpoint           step7000 EMA checkpoint
  NUM_TRAIN_STEPS             30000
  PICF_TRAINABLE_SCOPE        policy_only
  semantic_trainable_scope    action_head_only
  trainable_numel             2,165,792
  loss_total_minus_action     0 by construction
  window_trace                enabled

step7050:
  loss_action_default_equiv   0.022951
  loss_action_active7         0.102341
  loss_total_minus_action     0.000000
  grad_norm                   0.167682
  speed                       ~10.5 sec/step

files:
  metrics.jsonl               present
  window_trace_rank0.jsonl    present
  window_trace_rank1.jsonl    present
```

The first gate is valid and comparable.  This run must continue through the old
7550/7600 rebound band before drawing the semantic-drift vs action-head/data
conclusion.

Interpretation:

```text
If action_head_only stays below ~0.030 through 7600/8000:
  semantic backbone drift is required for the rebound.

If action_head_only also rebounds:
  root shifts to action-head optimizer/data-window/target distribution.
```

### E2. Data-Window Trace

Purpose:

```text
Test whether the 7500-7600 rows are a repeatable hard task/window cluster.
```

Required log fields:

```text
global step
dataset segment id / episode id / transition index
task language id or prompt hash
action target component norms
per-component action losses
sidecar proposal/tracklet counts
burnin/unroll frame indices
```

Implemented local instrumentation:

```text
scripts/picf_core_train.py now writes:
  window_trace_rank0.jsonl
  window_trace_rank1.jsonl

Each log interval records:
  global_step, rank, micro_step, flat_index, segment, start_step, prompt,
  retry_count, point_counts, action_norm, action_chunk_norm,
  action_chunk_first_norm, action_chunk_last_norm,
  proposal_count, proposal_mask_point_count, tracklet_count.

This is metadata-only.  It does not change sampling, model inputs, gradients,
or optimizer state.
```

Interpretation:

```text
If the same action spike aligns with the same task/window cluster across
policy_semantic_only and fixed_group:
  prioritize sampler/window/data diagnostics.

If it appears with different windows but same optimizer state:
  prioritize optimizer stability / semantic trust-region.
```

### E3. Same Windows, Frozen Semantic

Purpose:

```text
Separate data hardness from semantic representation drift.
```

Protocol:

```text
Replay or sample the same 7500-7600 window block with semantic frozen and only
the action head trainable.
```

Interpretation:

```text
If frozen semantic handles the block:
  trainable semantic drift is the proximate cause.

If frozen semantic also fails:
  the block itself is hard or target distribution differs from the 4-22 setup.
```

## 7. Immediate Engineering Actions

```text
1. Keep the fixed FSDP optimizer grouping patch.
2. Keep group LR/grad logging mandatory for rebound runs.
3. Add/enable a compact window-trace mode for action rebound diagnosis.
4. Run E1 before any new architecture change.
5. Do not treat raw overlap or object scaffold as the next direct repair target.
```

## 8. Current Scientific Conclusion

The most defensible current statement is:

```text
The recurrent late rebound is an action-side low-basin stability problem with
semantic/action representation involvement and possible data-window alignment.
It is not primarily caused by raw same-role overlap, non-action auxiliary loss,
or unfixed FSDP PICF-core LR grouping.
```

The root is narrowed but not fully closed until action-head-only is extended
past the 7550/7600 band and the data-window trace confirms or rejects a hard
window cluster.

## 9. 2026-05-29 A7 Action-Head-Only Trace Run Follow-Up

Active remote run:

```text
tmux:
  picf_a7_actionhead_trace_from7000_30k_20260529

checkpoint dir:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_ema7000_policyonly_actionhead_trace_from7000_30k_20260529

log:
  /mnt/picf_run_logs/
  picf_a7_ema7000_policyonly_actionhead_trace_from7000_30k_20260529.log
```

Configuration check:

```text
resume checkpoint:
  picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000

trainable scope:
  PICF_TRAINABLE_SCOPE=policy_only
  SEMANTIC_TRAINABLE_SCOPE=action_head_only

loss contract:
  loss_total == loss_action_default_equiv
  loss_total_minus_action == 0

trace contract:
  window_trace_rank0.jsonl and window_trace_rank1.jsonl are written.
```

Observed:

```text
step7050:
  loss_action_default_equiv          0.022951
  loss_action_active7                0.102341
  loss_total_minus_action            0.000000
  grad_norm                          0.167682
  windows_per_sec                    0.095153

step7100:
  loss_action_default_equiv          0.025771
  loss_action_active7                0.115666
  loss_total_minus_action            0.000000
  grad_norm                          0.084392
  windows_per_sec                    0.103976

step7150:
  loss_action_default_equiv          0.021833
  loss_action_active7                0.098441
  loss_total_minus_action            0.000000
  grad_norm                          0.105002
  windows_per_sec                    0.107019

step7200:
  loss_action_default_equiv          0.017932
  loss_action_active7                0.080687
  loss_total_minus_action            0.000000
  grad_norm                          0.114923
  windows_per_sec                    0.106987

step7250:
  loss_action_default_equiv          0.026885
  loss_action_active7                0.120204
  loss_total_minus_action            0.000000

step7300:
  loss_action_default_equiv          0.027154
  loss_action_active7                0.122115
  loss_total_minus_action            0.000000

step7350:
  loss_action_default_equiv          0.028195
  loss_action_active7                0.126068
  loss_total_minus_action            0.000000

step7400:
  loss_action_default_equiv          0.029102
  loss_action_active7                0.126918
  loss_total_minus_action            0.000000

step7450:
  loss_action_default_equiv          0.028440
  loss_action_active7                0.126261
  loss_total_minus_action            0.000000

step7500:
  loss_action_default_equiv          0.030503
  loss_action_active7                0.137044
  loss_total_minus_action            0.000000

step7550:
  loss_action_default_equiv          0.035659
  loss_action_active7                0.161547
  loss_total_minus_action            0.000000

step7600:
  loss_action_default_equiv          0.035610
  loss_action_active7                0.161139
  loss_total_minus_action            0.000000
```

Interpretation:

```text
The action-head-only branch is alive and still matches the earlier short
policy-head-only diagnostic through step7100, improves at step7150/7200, and
stays below 0.030 through step7350.  This already separates it from the old
EMA/phase-p2 7350 rebound, but it does not yet close the root cause: the
decisive policy_semantic/fixed_group failure band is still step7550/7600.
At step7550/7600, action-head-only also enters the strict reject band.
Therefore the two-row action rebound is reproduced without trainable PICF,
without trainable semantic backbone, and without non-action loss pressure.
```

Hard decision gate:

```text
If action-head-only also reaches >= 0.035 at step7550 and step7600:
  the rebound is inside the action head / optimizer / sampled-window action
  target path, not semantic backbone drift.

If action-head-only stays below about 0.030 through step7600 while
policy_semantic_only and fixed_group rebound:
  semantic/action backbone drift is the proximate cause.
```

Actual decision:

```text
action-head-only reaches:
  step7550 = 0.035659
  step7600 = 0.035610

This satisfies the hard bad gate.
```

## 11. Revised Root-Cause Interpretation After Action-Head-Only Failure

The strongest new evidence is not simply that action-head-only rebounded.  It
is that three causally different branches match almost point-by-point:

```text
step    action_head_trace  fixed_group  policy_semantic
7050    0.022951           0.022599     0.022478
7100    0.025771           0.025269     0.025595
7150    0.021833           0.022619     0.022732
7200    0.017932           0.018195     0.018454
7250    0.026885           0.026179     0.026358
7300    0.027154           0.026738     0.027032
7350    0.028195           0.027469     0.027494
7400    0.029102           0.026737     0.026914
7450    0.028440           0.027944     0.027958
7500    0.030503           0.030833     0.030759
7550    0.035659           0.035953     0.036278
7600    0.035610           0.035442     n/a
```

This weakens the previous optimizer/semantic-drift framing.  Because the
training sampler uses `np.random.default_rng(args.seed + 17 * rank)` after
resume and does not fast-forward RNG by `start_step`, every run resumed from
the same checkpoint replays the same sampled-window stream.  The near-identical
action curve across different trainable scopes is therefore more consistent
with a deterministic sampled-window difficulty block than with PICF structure
or semantic-backbone drift.

Mathematically, the observed log scalar is a training-window estimator:

```math
\hat L_{a,k}(\theta)
=
\frac{1}{|B_k|}
\sum_{(x,s,a)\in B_k}
\ell(\pi_\theta(x,s), a).
```

A rise in `\hat L_{a,k}` can come from:

```math
\hat L_{a,k+1}(\theta_{k+1})-\hat L_{a,k}(\theta_k)
=
\underbrace{\hat L_{a,k+1}(\theta_k)-\hat L_{a,k}(\theta_k)}_{\text{window distribution shift}}
+
\underbrace{\hat L_{a,k+1}(\theta_{k+1})-\hat L_{a,k+1}(\theta_k)}_{\text{model update effect}}.
```

The pointwise alignment across action-head-only, policy-semantic-only, and
fixed-group runs indicates the first term dominates in the 7500-7600 region.
This means the scalar should not be interpreted as a clean validation rebound
without a fixed held-out probe or a no-update same-window replay.

The trace files rule out a trivial single-task collapse:

```text
7500:
  100 windows, 90 unique prompts, action_norm_mean 5.375, proposal_mean 1.41,
  tracklet_mean 60.35.

7550:
  100 windows, 91 unique prompts, action_norm_mean 5.355, proposal_mean 1.34,
  tracklet_mean 61.02.

7600:
  100 windows, 89 unique prompts, action_norm_mean 5.415, proposal_mean 1.32,
  tracklet_mean 58.74.
```

So the hard block is not obvious from prompt count, proposal count, tracklet
count, or action-norm mean.  It is either a subtler composition of sampled
states/tasks or a property of the action target distribution not captured by
the coarse trace summary.

Immediate next diagnostic:

```text
Run a no-update or near-zero-LR same-window replay from the same step7000
checkpoint.  If it reproduces the same 7500-7600 curve, the late "rebound" is
primarily a training-window metric artifact / data-window hardness signal.

Then add a fixed validation-window action probe for future production 30K
runs.  Do not keep changing PICF/slot losses to fix a scalar that is not a
stationary validation metric.
```

## 12. Near-Zero-LR Replay Closure

Run:

```text
picf_a7_ema7000_policyonly_actionhead_nearzerolr_7600_20260529
```

Contract:

```text
resume checkpoint:
  picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000

trainable scope:
  PICF_TRAINABLE_SCOPE=policy_only
  SEMANTIC_TRAINABLE_SCOPE=action_head_only

learning rate:
  LR=1e-12, MIN_LR=0
  actual lr near the reject band is about 1e-16 to 1e-20

meaning:
  effectively no model update; it is a same-window forward/loss replay.
```

Observed:

```text
step7050  action=0.023108
step7100  action=0.025092
step7150  action=0.022025
step7200  action=0.017623
step7250  action=0.026254
step7300  action=0.027070
step7350  action=0.028855
step7400  action=0.028063
step7450  action=0.029194
step7500  action=0.030804
step7550  action=0.036840
step7600  action=0.037180
```

Conclusion:

```text
The rebound is reproduced when the model is not meaningfully updated.
Therefore the 7550/7600 spike is not caused by PICF/cotrain/semantic drift.
It is a train-window estimator spike under a resumed sampled-window stream.
```

The exact trainer-level fault is now clear:

```text
Before this repair:
  rng = np.random.default_rng(args.seed + 17 * rank)

After resume from step7000:
  the RNG started from the beginning again;
  every branch replayed the same early sampled-window stream;
  the same hard block appeared at the same logged steps.
```

Code repair:

```text
scripts/picf_core_train.py now defaults to --step-indexed-window-rng.

The sampler RNG is keyed by:
  seed, rank, global_step, micro_step, retry_count

This makes sampled windows deterministic and resume-safe:
  reaching step 7550 in one uninterrupted run and resuming to step 7550
  now see the same global-step-indexed stream rather than replaying step0
  windows after resume.
```

Scientific consequence:

```text
Do not classify the May-29 7550/7600 scalar as model failure.
Do classify it as a trainer resume/sampling diagnostic failure that made
train-window loss non-stationary and repeatedly comparable across branches.
```

## 10. Remote Checkpoint Cleanup Log

The cleanup policy used for this root-cause pass is deliberately conservative:

```text
Preserve:
  all pre-May checkpoints;
  all numeric 7000 and 8000 checkpoints;
  the current action-head trace run;
  the 20260527 actionprefix EMA 7000/8000 root-cause anchors.

Delete:
  post-May speed-profile / speed-ablation checkpoint weight directories only;
  keep their top-level logs, metrics, and args where present.
```

Remote audit files:

```text
/mnt/checkpoints/picf_core/picf_core/
  cleanup_manifest_20260529_action_rebound_dryrun.txt
  cleanup_deleted_20260529_action_rebound.txt
```

Executed deletion:

```text
picf_speed_prod_sem_backboneonly_nogc_ema7000_10_20260528/7010
picf_speed_ablate_sem_backboneonly_nogc_ema7000_3_20260528/7003
picf_speed_ablate_sem_backboneonly_ema7000_3_20260528/7003
picf_speed_ablate_teacher_off_ema7000_2_20260528/7002
picf_speed_ablate_sem_actionhead_ddp_ema7000_2_20260528/7002
picf_a7_speed_profile_sync_ema7000_fullcotrain_3_20260528/tmp_7003
picf_speed_prod_sem_backboneonly_gc_ema7000_10_20260528/tmp_7010
```

Result:

```text
/mnt available space:
  before cleanup: 191G
  after cleanup:  228G
```

## 11. Step-Indexed Confirmation And Relaunch

After the repair was synced to A7, a clean causal confirmation was run from the
preserved step7000 checkpoint:

```text
confirmation run:
  picf_a7_stepindexed_policyonly_actionhead_from7000_7600_20260529

scope:
  PICF core frozen
  PaliGemma/Gemma semantic backbone frozen
  PI0 action projection/time heads trainable only
  loss_total_minus_action = 0 by construction

result:
  step7050 loss_action_default_equiv = 0.049809
  step7100 loss_action_default_equiv = 0.051169
  active/downstream overlap at 7050 = 0.081 / 0.086
  active/downstream overlap at 7100 = 0.115 / 0.117
```

This intentionally does **not** match the old resumed action-head curve
(`7050 ~= 0.0229`, `7100 ~= 0.0258`).  That is the desired falsification:
the old curve was produced under a replayed sampled-window stream after resume,
so old step-matched scalar comparisons across resumed branches were invalid.

The corrected production relaunch is:

```text
run:
  picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000

contract:
  num_train_steps = 30000
  save_interval = 1000
  keep_last_checkpoints = 5
  log_interval = 50
  anchor_overlay_interval = 100
  step-indexed sampled-window RNG enabled by default
  Sonata / V-JEPA / AnyTouch frozen
  PaliGemma semantic backbone trainable
  policy head trainable
  PICF core trainable at low LR

first corrected production metric:
  step7050 loss_total                 = 0.060299
  step7050 loss_action_default_equiv  = 0.050193
  step7050 loss_total_minus_action    = 0.010105
  step7050 active/downstream overlap  = 0.044 / 0.057
  step7050 reserve raw overlap        = 1.000
  step7050 lr semantic/picf/policy    = 2.22e-5 / 3.18e-7 / 6.35e-5
```

Interpretation:

```text
The corrected run should not be judged against the old replayed 7000->7600
curve.  Its first corrected-window action loss is higher, but the structural
gates are healthy and the non-action budget remains bounded.  Continue unless
future fixed/step-indexed windows show sustained action degradation with
healthy sampling, or unless non-finite/structural reject gates trigger.
```

Local post-repair audit:

```text
python -m py_compile scripts/picf_core_train.py \
  scripts/verify_picf_owm_contract.py scripts/picf_core_train_test.py
  PASS

python scripts/verify_picf_owm_contract.py
  PASS, including trainer_window_rng_is_resume_safe

python -m pytest scripts/picf_core_train_test.py \
  -k "optimizer or picf_core_group or trainable_scope or window or rng" -q
  36 passed, 100 deselected
```

Operational decision:

```text
Do not stop the corrected run at step7050 solely because its action scalar is
higher than the old replayed-window step7050 value.  The first valid decision
points are the corrected step7100/7150/7200 trend and then the old suspect
7550/7600 region on the step-indexed stream.  If active/downstream overlap and
loss_total_minus_action remain bounded, continue until those gates are crossed.
```

7100 corrected production metric:

```text
step7050 -> step7100
  loss_total                0.060299 -> 0.061959
  loss_action_default_equiv 0.050193 -> 0.051401
  loss_total_minus_action   0.010105 -> 0.010558
  active/downstream overlap 0.044/0.057 -> 0.070/0.078
  grad_norm                 0.793 -> 0.704
  posterior recycle         0.133 -> 0.131
  posterior identity switch 0.178 -> 0.193
```

Interpretation: this is a mild first-window increase, not the old replayed-stream
rebound proof.  Structural gates remain healthy and the non-action budget is
still bounded.  Continue to 7150/7200 and then the legacy suspect 7550/7600
region before making an architecture decision.

7150 corrected production metric:

```text
step7050 -> step7100 -> step7150
  loss_total                0.060299 -> 0.061959 -> 0.053901
  loss_action_default_equiv 0.050193 -> 0.051401 -> 0.042441
  loss_action_active7       0.226786 -> 0.233111 -> 0.192911
  loss_total_minus_action   0.010105 -> 0.010558 -> 0.011460
  active/downstream overlap 0.044/0.057 -> 0.070/0.078 -> 0.110/0.110
  loss_anchor_object_pull   0.138 -> 0.175 -> 0.272
  loss_anchor_pv            0.522 -> 0.519 -> 0.516
  posterior recycle         0.133 -> 0.131 -> 0.131
  posterior identity switch 0.178 -> 0.193 -> 0.208
```

Interpretation: corrected-window action improves at 7150, so the run should
continue.  The structural warning to watch is not raw reserve overlap; it is
the mild upward drift in active/downstream overlap, object-pull difficulty, and
identity switch.  These are below reject bands but must be checked again at
7200 and the legacy suspect region 7550/7600.

7200 corrected production metric:

```text
step7050 -> step7100 -> step7150 -> step7200
  loss_total                0.060299 -> 0.061959 -> 0.053901 -> 0.048289
  loss_action_default_equiv 0.050193 -> 0.051401 -> 0.042441 -> 0.037938
  loss_action_active7       0.226786 -> 0.233111 -> 0.192911 -> 0.172060
  loss_total_minus_action   0.010105 -> 0.010558 -> 0.011460 -> 0.010352
  active/downstream overlap 0.044/0.057 -> 0.070/0.078 -> 0.110/0.110 -> 0.080/0.078
  loss_anchor_object_pull   0.138 -> 0.175 -> 0.272 -> 0.175
  loss_anchor_pv            0.522 -> 0.519 -> 0.516 -> 0.520
  posterior recycle         0.133 -> 0.131 -> 0.131 -> 0.128
  posterior identity switch 0.178 -> 0.193 -> 0.208 -> 0.184
```

Gate decision:

```text
CONTINUE.

7200 is not a replay of the old rebound pattern.  Action improves, active and
downstream overlap recover, non-action budget remains bounded, gradients remain
finite, and recycle/identity-switch do not saturate.  The next decisive gates
are 7550/7600 on the corrected step-indexed stream.
```

## 8. Current Scientific Root-Cause Plan

The current run is testing a sharper question than the previous optimizer,
two-timescale, and policy-only probes:

```text
Does the late action rise survive when the sampled-window stream is resume-safe?
```

The answer cannot be inferred from the old 7050/7550 scalar table because that
table was produced under the replayed-window bug.  The corrected run must be
judged internally:

```math
\Delta L_a(k)
=
L_a^{\text{step-index}}(k)
-
\min_{j \le k} L_a^{\text{step-index}}(j).
```

Stop/restart only if the corrected stream satisfies at least one hard condition:

```text
1. non-finite loss or gradient;
2. loss_total_minus_action > 0.02 for a logged gate;
3. active or downstream same-role support overlap > 0.25 for a logged gate;
4. action_default_equiv enters a sustained rebound on corrected windows:
     two consecutive logs above both 0.035 and 30% above the corrected local min;
5. action rebound is accompanied by posterior lifecycle degradation
     (identity switch/recycle rising together).
```

Otherwise, continue.  Raw reserve overlap alone is not a stop condition:

```text
reserve rows can duplicate no-object/background capacity;
active/downstream rows are the action-visible object files;
therefore aqr_same_role_support_overlap_max ~= 1.0 can coexist with a healthy
action-visible graph.
```

If the corrected 7550/7600 gates pass:

```text
old rebound root cause is closed as train-window replay / logging-regime error;
continue toward 8000 and preserve the current run.
```

If the corrected 7550/7600 gates fail with healthy structure:

```text
do not add object/slot losses first;
run a fixed held-out action probe or validation-window loss from the same ckpt
to distinguish true action regression from nonstationary training-window
difficulty.
```

If the corrected gates fail with unhealthy structure:

```text
then restart from the latest healthy checkpoint and inspect the corresponding
structural failure path: active/downstream overlap, non-action budget, posterior
identity switch/recycle, and object-owner transport.
```

2026-05-29 local audit follow-through:

```text
Remote state at audit start:
  tmux sessions:
    picf_a7_stepidx_ema7000_30k
    picf_a7_stepidx_gate_monitor
  latest observed progress:
    progress bar around step7175
    metrics last committed at step7150
  /mnt free:
    about 243G

Local executable checks:
  python -m compileall -q scripts src/openpi/picf
    PASS
  python scripts/verify_picf_owm_contract.py
    PASS, including trainer_window_rng_is_resume_safe
  python scripts/picf_owm_strict_diagnose.py --fail-on-fail
    PASS, WARN only for missing runtime artifact inputs
  python scripts/picf_owm_dataflow_trace.py --fail-on-fail
    PASS
  python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
    PASS
  python -m pytest scripts/picf_core_train_test.py \
    -k "optimizer or picf_core_group or trainable_scope or window or rng or checkpoint or retention or overlay" -q
    47 passed, 89 deselected
  python scripts/picf_latest_slot_deployment_audit.py
    PASS
  python scripts/picf_action_visible_reserve_gate_audit.py
    PASS
  python scripts/picf_binding_dataflow_math_audit.py
    PASS
  python scripts/picf_binding_logit_calibration_audit.py
    PASS
  python scripts/picf_binding_signature_common_mode_audit.py
    PASS
  python scripts/picf_posterior_file_competition_audit.py
    PASS
  python scripts/picf_posterior_birth_transport_audit.py
    PASS
  python scripts/picf_posterior_binding_signature_memory_audit.py
    PASS
  python scripts/picf_object_candidate_slot_binding_audit.py
    PASS
  python scripts/picf_oeml_dataflow_audit.py
    PASS
  python scripts/picf_owm_professor_grade_audit.py
    16/16 PASS
  python -m pytest src/openpi/picf/core/pipeline_test.py \
    -k "posterior or binding or active_slot or temporal_visual or pg_image or cache or tracklet or proposal or tactile" -q
    28 passed, 75 deselected
  python -m pytest src/openpi/picf/policy_test.py \
    src/openpi/picf/paligemma/wrapper_test.py \
    src/openpi/picf/vjepa/wrapper_test.py -q
    61 passed
  python -m pytest src/openpi/picf/core/training_test.py \
    scripts/picf_core_train_test.py \
    -k "loss or slot or support or action or optimizer or checkpoint or retention or window or rng or trainable_scope or sidecar or object" -q
    83 passed, 87 deselected
  python -m pytest src/openpi/picf/core/pipeline_test.py \
    src/openpi/picf/core/training_test.py \
    src/openpi/picf/policy_test.py \
    src/openpi/picf/paligemma/wrapper_test.py \
    src/openpi/picf/vjepa/wrapper_test.py \
    scripts/picf_core_train_test.py -q
    334 passed
  git diff --check
    PASS
  uv run ruff check scripts/picf_core_train.py scripts/verify_picf_owm_contract.py \
    scripts/picf_core_train_test.py src/openpi/picf/core/pipeline.py \
    src/openpi/picf/core/training.py src/openpi/picf/policy.py \
    src/openpi/picf/paligemma/wrapper.py src/openpi/picf/vjepa/wrapper.py --select F,E9
    PASS after removing unused exception binding, unused modality-valid locals,
    an unused simulation point-feature extraction, and an unused PaliGemma import.
  python -m pytest scripts/picf_core_train_test.py \
    -k "step_indexed_window_rng or normalize_train_args_enables_step_indexed_window_rng or checkpoint or optimizer" -q
    30 passed, 106 deselected
```

Audit interpretation:

```text
No active code-level bug was found in the high-risk paths that previously
caused false action rebound, duplicate posterior ownership, reserve/context
leakage, object-candidate cloning, or blind common-mode binding evidence.

This does not claim behavioral success before the corrected run crosses the
legacy suspect region.  It only means the current run is valid enough to keep
monitoring rather than restart on the basis of known historical bugs.
```

2026-05-29 corrected-run gate update:

```text
Current corrected run:
  picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529

Observed step 7200:
  loss_total: 0.0482893288
  loss_action_default_equiv: 0.0379376560
  loss_action_active7: 0.1720603704
  loss_total_minus_action: 0.0103516709
  loss_anchor_object_pull: 0.1753762811
  loss_anchor_pv: 0.5200435519
  loss_mapg_support_diversity: 0.1034859121
  loss_slot_jepa: 2.9811167717
  active/downstream overlap: 0.080000 / 0.077714
  reserve overlap: 1.0
  posterior_identity_switch_rate: 0.183889
  posterior_recycle_rate: 0.128231
  grad_norm: 0.582450
  verdict: CONTINUE

Interpretation:
  The corrected stream has passed the 7200 gate.  This is not enough to close
  the old rebound issue because the historical failure band was 7550/7600, but
  it is enough to reject an immediate-stop decision at 7200.  Raw reserve
  overlap remains high by design and is not treated as an action-visible
  failure unless active/downstream overlap also rises.

Next hard gates:
  7550 and 7600 determine whether the old spike survives step-indexed window
  sampling.  8000 is the first broader stability gate after the suspect band.
```

2026-05-29 step7250 update:

```text
step7250:
  loss_total: 0.0616875887
  loss_action_default_equiv: 0.0508641154
  loss_action_active7: 0.2303861529
  loss_total_minus_action: 0.0108234705
  active/downstream overlap: 0.084968 / 0.093113
  posterior_identity_switch_rate: 0.180000
  posterior_recycle_rate: 0.130084
  verdict: CONTINUE

Interpretation:
  This is the first logged action increase after the corrected step7200 local
  minimum.  It is not yet sustained rebound.  The structural gates are still
  healthy, so stopping here would repeat the old mistake of treating a scalar
  train-window sample as architecture failure before the suspect 7550/7600
  region is reached.
```

2026-05-29 step7300 update:

```text
step7300:
  loss_total: 0.0542847142
  loss_action_default_equiv: 0.0431232005
  loss_action_active7: 0.1955263913
  loss_total_minus_action: 0.0111615146
  active/downstream overlap: 0.080000 / 0.084150
  posterior_identity_switch_rate: 0.177778
  posterior_recycle_rate: 0.130421
  verdict: CONTINUE

Interpretation:
 The step7250 action increase did not continue at step7300.  It is consistent
  with corrected-window variance, not enough to diagnose structural rebound.
  The next decisive region remains 7550/7600.
```

2026-05-29 step7350 update:

```text
step7350:
  loss_total: 0.0505230948
  loss_action_default_equiv: 0.0397697911
  loss_action_active7: 0.1803223789
  loss_total_minus_action: 0.0107533075
  active/downstream overlap: 0.085000 / 0.083510
  posterior_identity_switch_rate: 0.183889
  posterior_recycle_rate: 0.127943
  verdict: CONTINUE

Interpretation:
  Step7350 is below both 7250 and 7300 while the non-action budget and
  active/downstream overlap remain bounded.  It further weakens the immediate
  structural-rebound hypothesis.  It still does not close the old issue because
  the historical failure band was 7550/7600.
```

2026-05-29 step7400 update:

```text
step7400:
  loss_total: 0.0563516207
  loss_action_default_equiv: 0.0457963534
  loss_action_active7: 0.2076118290
  loss_total_minus_action: 0.0105552664
  active/downstream overlap: 0.064948 / 0.068091
  posterior_identity_switch_rate: 0.177778
  posterior_recycle_rate: 0.130156
  verdict: CONTINUE

Interpretation:
  Action rose from 7350 but stayed below the sustained-rebound stop rule
  relative to the corrected local minimum, while active/downstream overlap
  improved and non-action budget stayed bounded.  This remains a continue
  decision until the old 7550/7600 band is observed.
```

Root-cause hypothesis matrix after re-reading experiment notes:

```text
H1: active slot/support collapse caused the action rebound.
  Status:
    not supported for the current corrected run.
  Evidence:
    historical raw overlap was often high, but active/downstream overlap stayed
    low in multiple diagnostics;
    current corrected step7200 active/downstream overlap is 0.080/0.078.
  Gate:
    react only if active/downstream overlap exceeds 0.25, not if reserve/raw
    overlap is high alone.

H2: non-action auxiliary losses dominate and push action away.
  Status:
    currently not supported.
  Evidence:
    step7200 loss_total_minus_action is 0.01035, below the 0.02 stop line.
  Gate:
    if loss_total_minus_action rises above 0.02 at a logged gate, inspect
    slot/object/prefix losses before continuing.

H3: optimizer restart or LR schedule alone caused the old improvement/rebound.
  Status:
    insufficient as root cause.
  Evidence:
    a near-zero-LR replay reproduced the historical action spike, which means
    the spike can be produced without meaningful model update.
  Consequence:
    do not keep resetting optimizer as a primary cure unless corrected-window
    validation shows a true optimization failure.

H4: resumed training replayed early sampled windows and made step numbers
    incomparable.
  Status:
    strongly supported and fixed in the trainer.
  Evidence:
    legacy trainer used stateful RNG seeded only by seed/rank; resumed step7000
    did not sample the same distribution as uninterrupted step7000.
  Fix:
    step-indexed RNG is keyed by seed/rank/global_step/micro_step/retry.
  Current test:
    corrected run from ckpt7000 is the necessary validation.

H5: PaliGemma/PICF interface prefix drift is the only cause.
  Status:
    not closed, but no longer first-order until corrected 7550/7600 evidence.
  Evidence:
    action prefix EMA and two-timescale changes reduced some instability, but
    the zero-LR replay shows the historical spike could be a sampling artifact.
  Gate:
    if corrected 7550/7600 still rebounds with healthy active structure, run a
    fixed held-out action-window probe before further architecture changes.

H6: current run is already useless because raw same-role overlap is 1.0.
  Status:
    rejected.
  Evidence:
    raw metric mixes active/context/reserve rows; reserve rows intentionally
    carry duplicated no-object/background capacity. The action-visible metrics
    split active/downstream from reserve and are healthy at step7200.
```

Operational decision:

```text
Continue the current corrected run until at least 7550/7600.
Do not restart before the old suspect band unless:
  loss becomes non-finite;
  loss_total_minus_action > 0.02;
  active/downstream overlap > 0.25;
  posterior recycle and identity switch rise together with action loss.
```

2026-05-29 step7450 update:

```text
step7450:
  loss_total: 0.0569891185
  loss_action_default_equiv: 0.0459649861
  loss_action_active7: not used for stopping; action-default is comparable metric
  loss_total_minus_action: 0.0110241296
  loss_anchor_object_pull: 0.2413659692
  loss_anchor_pv: 0.5216296315
  loss_mapg_support_diversity: 0.1721987128
  loss_slot_jepa: 1.1252224445
  active/downstream overlap: 0.134999 / 0.128150
  posterior_identity_switch_rate: 0.176111
  posterior_recycle_rate: 0.117888
  verdict: CONTINUE

Interpretation:
  Step7450 is a higher-action point, but it is not yet a stop signal.
  The non-action budget is still bounded near 0.011, active/downstream overlap
  is below the 0.25 stop line, and posterior recycle decreased rather than
  rising with identity-switch.  Continue to the historical 7550/7600 band.
```

2026-05-29 step7500 update:

```text
step7500:
  loss_total: 0.0542113073
  loss_action_default_equiv: 0.0423136577
  loss_total_minus_action: 0.0118976496
  loss_anchor_object_pull: 0.3272219598
  loss_anchor_pv: 0.4965823889
  loss_mapg_support_diversity: 0.2128035724
  loss_slot_jepa: 0.9850533009
  active/downstream overlap: 0.164993 / 0.174715
  posterior_identity_switch_rate: 0.199444
  posterior_recycle_rate: 0.105055
  verdict: CONTINUE

Interpretation:
  Action improved relative to 7450.  Active/downstream overlap rose but remains
  below the 0.25 stop gate; non-action pressure remains below 0.02 and recycle
  continues to decrease.  The run should not be stopped before observing
  7550/7600.
```

2026-05-29 step7550 update:

```text
step7550:
  loss_total: 0.0558114722
  loss_action_default_equiv: 0.0445233509
  loss_total_minus_action: 0.0112881213
  loss_anchor_object_pull: 0.2608813047
  loss_anchor_pv: 0.5086849928
  loss_mapg_support_diversity: 0.1088067591
  loss_slot_jepa: 0.9828047752
  active/downstream overlap: 0.084996 / 0.089842
  posterior_identity_switch_rate: 0.203889
  posterior_recycle_rate: 0.096984
  verdict: CONTINUE

Interpretation:
  The first historical suspect gate passed.  The scalar action value is not a
  new low, but it does not satisfy the sustained-rebound rule relative to the
  corrected stream, and the structural gates are healthy: active/downstream
  overlap dropped back below 0.10, non-action pressure stayed near 0.011, and
  recycle continued to decrease.  Continue to step7600 before declaring the old
  replay-window issue behaviorally closed.
```

2026-05-29 production-criterion correction:

```text
User challenge:
  loss_action_default_equiv around 0.04 is not acceptable if the reference
  2026-04-22 ablation and old EMA7000 continuation reached the 0.02 range.

Decision:
  Stop the corrected live run as a production candidate.

Reason:
  The run passed structural gates, but it did not pass the action-quality
  criterion.  More importantly, the live train-window scalar is not stationary:
  after the step-indexed RNG repair it is sampling a different deterministic
  window stream from the old 0.02 reference.  Therefore the current evidence
  can reject "production good enough", but it cannot by itself distinguish:

    A. the model is genuinely worse;
    B. the corrected stream contains harder windows;
    C. old 0.02 was partly a replay-window / window-distribution artifact.

Required repair to the diagnostic, not to the architecture:
  add a fixed-window no-update action probe and compare ckpts on exactly the
  same accepted window set.

Implemented local tool:
  scripts/picf_fixed_window_action_probe.py

Contract:
  no optimizer.step;
  no backward;
  fixed accepted flat indices;
  optional validation split;
  CALVIN source construction must match train() rather than a minimal replay
  source:
    effective_unroll_steps = burnin_steps + unroll_steps;
    action_horizon;
    tactile calibration/backgrounds;
    action normalizer;
    scene observation;
    tracklet/proposal/sidecar fields;
    augmentation mode;
    calvin_segment_indices;
  outputs per-window JSONL plus aggregate mean/std/min/max for:
    loss_action_default_equiv;
    loss_total_minus_action;
    structural overlap and posterior metrics.

Rule:
  Do not relaunch another blind 30K from action train-loss alone.  First run
  the fixed-window probe on the preserved 7000/old-best checkpoints.  Only if
  the fixed-window probe also shows degradation should the next experiment
  change optimizer/loss/architecture.

2026-05-29 probe deployment correction:
  A probe that only copied the script into A7's older
  `/root/openpi_posterior_vla_clean` tree failed with checkpoint/model schema
  mismatch because that tree lacked the current action-prefix teacher,
  control-state embedding, and slot-quality head fields.  The valid probe path
  is `/root/openpi_probe_current_20260529`, a separate non-training copy with
  current local `src/openpi/picf` and `scripts` overlaid.  A 2-window smoke
  probe on ckpt7000 completed with `status=ok`, zero retryable skips, and live
  tracklet/proposal sidecar fields.  The running 32-window comparison is:

    tmux: fixed_probe32_20260529
    log:  /mnt/picf_fixed_probe_20260529/fixed_probe32.log

  This probe is the next decision gate.  The old 0.02-vs-current-0.04 question
  must be answered on the fixed accepted window set, not by comparing nonstationary
  train streams.

2026-05-29 fixed-window result:

```text
Probe:
  /root/openpi_probe_current_20260529/scripts/picf_fixed_window_action_probe.py

Accepted windows:
  32 training windows, zero retryable skips, same flat_indices for both ckpts

Artifacts:
  /mnt/picf_fixed_probe_20260529/old_7000_train32_summary.json
  /mnt/picf_fixed_probe_20260529/old_8000_train32_summary.json
  /mnt/picf_fixed_probe_20260529/train32_indices.json

Mean metrics:
  old7000 loss_action_default_equiv: 0.069682
  old8000 loss_action_default_equiv: 0.067969
  old7000 loss_total_minus_action:   0.250987
  old8000 loss_total_minus_action:   0.259952
  old7000 active overlap:            0.032028
  old8000 active overlap:            0.031250
  old7000 downstream overlap:        0.035603
  old8000 downstream overlap:        0.040650
  raw overlap remains reserve-saturated near 1.0 in both.

Paired old8000 - old7000 deltas:
  action_default_equiv: -0.001713, improved on 20 / 32 windows
  total_minus_action:   +0.008965
  anchor_object_pull:   +0.038483
  anchor_pv:            -0.006176
  aqr_denoising:        -0.039121
  slot_jepa raw:        +3.367047, but lambda_slot_jepa is disabled

Decision:
  The historical train-log rebound from old7000 ~=0.021 to old8000 ~=0.043 is
  not reproduced on fixed windows. On the same accepted windows, old8000 is
  marginally better on action than old7000. Therefore the old 0.02 row is not a
  stationary production baseline; it was a sampled-window / train-stream scalar.

Consequence:
  Do not restart from architecture/loss changes just because a corrected live
  train stream logs ~0.04.  The correct production gate is now fixed-window
  validation/action probe plus CALVIN/video behavior, not raw train-window
  scalar equality with the old replayed-window branch.
```

2026-05-29 extended fixed-window comparison:

```text
Current corrected branch, 64 identical accepted windows:
  step7000 action_default_equiv = 0.061891
  step8000 action_default_equiv = 0.057919
  delta                         = -0.003971

Archived 2026-04-22 PI0.5-only ablation, same 64 windows:
  step7500 action = 0.058717
  step10000 action = 0.053813
  step20000 action = 0.051038
```

Interpretation:

```text
1. Current step8000 is not worse than current step7000 on stationary action.
2. Current step8000 is roughly old step7500 fixed-window action quality.
3. Current step8000 still trails old step20000 by about 0.0069 absolute.
4. The historical old train-log 0.02 row is not a fixed-window validation
   target; the archived 20k checkpoint itself is about 0.051 on the same
   windows.
5. The remaining gap is an action-capacity / behavior-eval question, not a
   raw-overlap collapse question.
```

2026-05-29 relaunch decision:

```text
Decision:
  Stop treating the old train-stream 0.02 row as a hard baseline.
  Relaunch the corrected 30K run from the preserved old step7000 checkpoint.

Launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_stepindexed_fixedwindow_from7000_30k_20260529.sh

Remote session:
  tmux:
    picf_a7_stepindexed_fixedwindow_from7000_30k_20260529
  log:
    /mnt/picf_run_logs/
      picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529.log
  output:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529

Contract:
  resume_checkpoint:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
  num_train_steps: 30000
  save_interval: 1000
  keep_last_checkpoints: 5
  log_interval: 50
  anchor_overlay_interval: 100
  unroll_steps: 2
  burnin_steps: 1
  semantic_trainable_scope: backbone_only
  frozen pretrained perception:
    Sonata, V-JEPA, AnyTouch
  trainable:
    PaliGemma backbone slice, policy/action interface, PICF non-frozen heads
  two-timescale core:
    picf_core_lr_scale = 0.005
  action pressure:
    action loss weight = 2.0
  action-prefix interface:
    rmsnorm + EMA teacher + trust region
  object scaffold:
    cosine decay to floor 0.03 by step1500

Acceptance:
  Do not declare success/failure from a single live loss row.
  Inspect:
    fixed-window probe at preserved checkpoints;
    50-step train trend;
    active/downstream overlap, not raw reserve overlap;
    CALVIN/video behavior after saved checkpoints.
```

2026-05-29 live relaunch first structured row:

```text
Run:
  picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529

step7050:
  loss_total:                    0.059594
  loss_action_default_equiv:     0.048841
  loss_total_minus_action:       0.010754
  active support overlap max:    0.055000
  downstream support overlap:    0.062652
  raw same-role overlap:         1.0 reserve-saturated
  posterior_identity_switch:     0.191667
  posterior_recycle_rate:        0.131581
  pi_prefix_teacher_cos:         0.999799
  pi_prefix_teacher_delta_rms:   0.011512
  lr_policy_head:                6.3519e-05
  lr_semantic_backbone:          2.2232e-05
  lr_picf_core:                  3.1760e-07
  grad_norm_policy_head:         7.1174e-04
  grad_norm_semantic_backbone:   1.3842
  grad_norm_picf_core:           1.6335e-05

Interpretation:
  This row is not production-good, but it is not a structural-collapse row.
  The old slot-collapse suspects are absent:
    non-action budget is under the 0.02 stop gate;
    active/downstream overlap is far under the 0.25 stop gate;
    PICF core is really on the intended slow LR group;
    action-prefix scale is bounded and EMA-trusted.

  The row is consistent with the fixed-window probe correction:
    old7000 fixed-window action mean was about 0.0697;
    old8000 fixed-window action mean was about 0.0680.
  Therefore a live train-stream row near 0.049 is not evidence that the
  architecture regressed relative to a stationary old 0.02 baseline.  The old
  0.02 row remains classified as a nonstationary sampled-window/train-stream
  scalar until a fixed-window probe contradicts that.

Decision:
  Continue this run at least to the next structured rows and the historical
  suspect band.  Do not restart from another architecture/loss patch at 7050.

Hard stop:
  If later rows show sustained action rebound together with either
  loss_total_minus_action > 0.02 or active/downstream overlap > 0.25, stop as a
  structural failure.  If action alone rises while structure stays healthy,
  run the fixed-window probe on the saved checkpoint before changing model
  architecture.
```

2026-05-29 follow-up rows:

```text
step7050:
  action_default_equiv: 0.048841
  non_action:           0.010754
  active/downstream:    0.055000 / 0.062652
  posterior switch/recycle: 0.191667 / 0.131581

step7100:
  action_default_equiv: 0.050966
  non_action:           0.010511
  active/downstream:    0.069994 / 0.075803
  posterior switch/recycle: 0.185556 / 0.128175

step7150:
  action_default_equiv: 0.042673
  non_action:           0.010629
  active/downstream:    0.095000 / 0.095714
  posterior switch/recycle: 0.199444 / 0.127542

Verdict:
  Continue.  The 7100 high row did not become a sustained rebound: by 7150
  action fell below both 7050 and 7100, while the non-action budget stayed near
  0.0106 and active/downstream overlap stayed far below the 0.25 structural
  stop line.

  Raw same-role overlap remains ~1.0, but this is the reserve/inactive capacity
  diagnostic already separated by the active/downstream metrics.  Do not stop
  or redesign on raw overlap alone.

Next hard gates:
  7200/7250: should not show two consecutive action rows above 1.3x the
  corrected local minimum.
  7350-7600: historical suspect band; if action rises there while structure
  stays healthy, run fixed-window probe on the saved checkpoint before changing
  architecture.
```
```

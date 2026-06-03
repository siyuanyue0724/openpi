# PICF-AQR-OWM G11 Full VLA Bridge/Data Plan

Date: 2026-06-03

This file is the execution contract after G10 rejected the static sampler and
simple action-boundary hypotheses.  It explicitly covers all requested
VLA-training-system points, marks what has already been tested, and defines the
remaining fast experiments.  The goal is not another "increase batch" run.  The
goal is to isolate why action does not descend when the production trainer uses
small physical batches.

## One-Line State

G10 shows that the current failure is not fixed by task-uniform K4 logical
batching, temperature sampling, trajectory sampling, per-bucket normalization,
cheap EMA action reweighting, adapter PCGrad, or policy-only PICF freezing.
The next root class is the action-context bridge, action data/window semantics,
or the effective action-head capacity under production sampling.

## Non-Negotiable Scaling Principle

Every accepted change must preserve large heterogeneous data scaling:

```text
many datasets
many tasks
missing modalities
different data quality
different embodiment / action spaces in future
```

Therefore rejected shortcuts:

```text
hand-picked CALVIN windows as the production sampler
per-task overfit heads for a single small benchmark
hard masks as mandatory labels
full-model gradient surgery without measured conflict evidence
fixed-window diagnostics as training recipes
```

Allowed changes:

```text
controlled task/dataset mixture
logical gradient accumulation
per-task/per-bucket objective normalization
explicit action/PICF gradient boundaries
modality/action adapters
optional future per-embodiment heads when multi-robot data exists
```

## Requested Points: Done / Rejected / Still To Test

| Requested point | Current state | Evidence | G11 action |
| --- | --- | --- | --- |
| probabilistic dataset/task mixing | deployed | task_uniform, temperature, trajectory, explicit weight spec exist | no rerun except controlled bridge cases |
| batch balancing ratios | deployed | `--calvin-bucket-weight-spec` and target `q_b` loss scaling exist | keep |
| gradient accumulation logical batch | deployed | K4 DDP path passed G10 dataflow | keep |
| per-dataset/per-task normalization | deployed for CALVIN task buckets | per-bucket target normalization and metrics exist | keep; dataset/embodiment hook is future-data only |
| task-uniform sampling | tested insufficient | G10a rejected | do not repeat alone |
| temperature sampling | tested insufficient | G10b rejected | do not repeat alone |
| trajectory-proportional sampling | tested insufficient | G10c rejected | do not repeat alone |
| dynamic PiKE-style mixing | not fully deployed | cheap EMA scalar F9b failed; true dynamic mixing needs positive static recipe first | defer unless G11 gradient probe proves it |
| PCGrad/CAGrad | adapter PCGrad tested insufficient | F9c rejected | do not repeat unless new gradient probe changes evidence |
| action/backbone insulation | partially tested | G10d policy-only rejected, but no-PICF-action and bridge modes not fully compared | G11a/b/c |
| VLM + action expert separation | partially native | PI0.5 already has separate policy/action head; PICF context bridge is the unknown | G11 bridge matrix |
| modality adapters | deployed | typed PICF projectors and action context adapter exist | keep |
| embodiment adapters | not applicable to CALVIN | one embodiment now | future multi-robot hook, no CALVIN test |
| continuous action chunking | native | PI0.5 action path predicts continuous action chunks | no rewrite in G11 |
| full action head rewrite | not justified | no evidence action path itself is incapable; E21 exact-window proves capacity | defer |
| action expert MoE | not justified | needs persistent task-family conflict after bridge/data audit | defer |
| System2/System1 planning split | architecture branch | useful for long-horizon future but not current 1-2h action plateau root | defer |

Strict interpretation:

```text
deployed != solved
tested dataflow != behavior accepted
OOM/resource blocked != model rejected
cheap proxy rejected != full method rejected
```

Therefore the current scientific state is:

```text
Confirmed not enough by itself:
  task_uniform
  temperature sampling
  trajectory sampling
  K4 logical accumulation
  per-bucket normalization
  cheap action-bucket EMA scalar
  adapter-scope PCGrad
  policy-only/stationary-PICF
  no-PICF-action K4 adapter-scope G11a

Completed in this gate:
  A2 gradient-cosine probe
  A3 action-window / target-scale audit

Recently tested:
  no-PICF-action with fuller semantic capacity under resource-safe FSDP G11b2
  Result: informative but not deployable; action only recovered to the G11a
  band and runtime became much slower.
  prefix-fusion bridge G11c
  Result: structural losses improved, but action worsened through 11040; not
  deployable as the final bridge fix.
  append bridge negative control G11d
  Result: failed immediately at step 11010 with action=0.054851; append is not
  a repair and validates that "more context tokens" is not the answer.

Still not justified until new evidence appears:
  full PiKE dynamic mixing
  full PCGrad/CAGrad
  action-expert MoE
  embodiment-specific heads on single-embodiment CALVIN
```

## 8-Hour Completion Contract

The requested VLA repair ideas are not all equivalent.  To avoid both
under-testing and unprincipled architecture churn, the remaining gate is split
into three levels.

### Level 1: Already Implemented And Behavior-Tested

These must stay in the baseline, but repeating them alone is not useful:

```text
task-uniform bucket sampler
temperature sampler
trajectory-proportional sampler
without-replacement logical K4 update
per-bucket logical loss normalization
per-bucket metrics
cheap per-bucket action EMA scaling
adapter-scope PCGrad
policy-only/stationary-PICF boundary
no-PICF-action K4 adapter bridge test
no-PICF-action K2 fuller-capacity FSDP test
```

Scientific conclusion:

```text
These cover the VLA Foundry / ABot-M0 / PiKE first-order data-mixing recipe,
but they did not by themselves restore production action descent.  Therefore
the problem is not "we forgot task-balanced mixing"; it is either action
conditioning geometry, action target/window semantics, or measured gradient
conflict that requires a stronger second-order method.
```

### Level 2: Must Finish In This Gate

These are the only remaining 1-2 hour tests that can change the next
deployment decision without weakening scaling:

```text
G11c prefix-fusion bridge:
  hypothesis: gated prefix injection gives the PI action path a cleaner,
  position-stable conditioning interface than suffix cross-attention.

G11d append bridge negative control:
  hypothesis: simply appending more context tokens is not enough; if append
  wins, we likely have a positional/layout bug rather than a true architecture
  improvement.

A2 gradient-cosine probe:
  hypothesis: if per-bucket action gradients are strongly negative in the
  policy/action adapter group, scoped PCGrad/CAGrad or action-expert MoE becomes
  justified; otherwise they should stay deferred.

A3 action-window / target-scale audit:
  hypothesis: if action statistics differ sharply by bucket/window, the
  action plateau is a data/window/normalization problem, not a PICF bridge
  problem.
```

Stop criteria:

```text
If G11c clearly descends and G11d does not:
  deploy prefix_fusion/hybrid gated prefix for the next longer run.

If G11c and G11d both fail:
  stop bridge tinkering and run A2/A3 before any new architecture.

If A2 shows strong negative action gradient cosine:
  test scoped gradient surgery / action-expert conflict handling.

If A3 shows bucket/window target-scale mismatch:
  repair action target/window normalization before any MoE.
```

### Level 3: Not Allowed Without Evidence

These are real VLA methods, but adding them now would be unscientific unless
Level 2 produces their trigger:

```text
full PiKE dynamic mixing:
  only after static mixture is positive but nonstationary, or A2/A3 shows
  measured bucket difficulty drift.

full PCGrad/CAGrad:
  only after A2 shows strong negative per-bucket gradient cosine.

action-expert MoE:
  only after persistent task-family conflict remains under good bridge and
  good action-window normalization.

per-embodiment heads:
  not a CALVIN gate; add when multi-robot data exists.

full action-head rewrite:
  not justified while PI0.5 native continuous chunk path has demonstrated
  capacity on controlled windows.
```

## Mathematical Root Test

The intended multi-task objective is:

```text
L = sum_b q_b L_b
G = sum_b q_b grad(L_b)
```

The production estimator after F8/G10 is:

```text
g_hat = sum_{b in B_step} q_b / n_b * grad(L_b)
```

where `B_step` is a K4 without-replacement bucket set and `n_b` is the number
of micro-windows from bucket `b` in the logical step.  This estimator and its
telemetry are now implemented.  G10 says this alone is not enough.

The next question is whether the action model receives a bad conditional
representation:

```text
action_pred = A(theta_a; pi0_context, PICF_context)
```

If `PICF_context` is noisy or the bridge uses it incorrectly, then improving
PICF structural losses can still hurt action:

```text
grad_theta_a L_action(pi0_context, PICF_context)
```

can point away from the action optimum even while:

```text
L_anchor_object_pull
L_anchor_pv
L_mapg_routing
overlap
```

improve.  This exact pattern appeared in G10a/b/c/d.

Therefore G11 tests:

```text
1. Does action descend when PICF is removed from the action condition?
2. If yes, which bridge mode is harmful?
3. If no, does a fuller semantic/action capacity setting descend?
4. If no, the remaining root is action data/window semantics or target scale,
   not PICF context.
```

## G11 Fast Experiment Matrix

All runs should use the same checkpoint, same sidecars, same step window, and
same logging.  Each case should run 100 optimizer steps first.  Stop early if
`loss_action_default_equiv` worsens monotonically past the known G10 control.

Common configuration:

```text
resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

sidecar root:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520

K4 logical update:
  world_size=2
  accum_steps=2
  logical_batch_task_count=4
  calvin_bucket_sampling_mode=task_uniform
  calvin_bucket_sample_without_replacement=1
  logical_batch_bucket_normalization=1
  logical_batch_log_bucket_metrics=1

action:
  action_loss_weight=4.0
  optimizer_checkpoint_mode=model-only
  checkpoint target=11100 for the first gate
```

### G11a: No-PICF-Action Condition

Purpose:

```text
Remove PICF context from action conditioning while keeping the same logical
batch and action trainable capacity.  This is the cleanest bridge-causality
test.
```

Config delta:

```text
PICF_ACTION_CONDITION_ENABLED=0
SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter
PICF_TRAINABLE_SCOPE=all
PICF_CORE_LR_SCALE=0.001
```

Decision:

```text
If action descends here but not G10a:
  PICF action-context bridge is causal.

If action still worsens:
  bridge noise is not sufficient explanation; move to action data/window or
  action-head capacity test.
```

### G11b: No-PICF-Action With Fuller Semantic Capacity

Purpose:

```text
Check whether G11a is falsely negative because `action_head_and_adapter` is too
weak under no-PICF-action.  This preserves the action data distribution but
allows the semantic/action backbone-facing trainable scope used by stronger
historical runs.
```

Config delta:

```text
PICF_ACTION_CONDITION_ENABLED=0
ACCUM_STEPS=1
LOGICAL_BATCH_TASK_COUNT=2
SEMANTIC_TRAINABLE_SCOPE=backbone_only
```

Reason for K2:

```text
K4 with fuller semantic scope has repeatedly hit memory limits on 2x40GB.
This case is not the final recipe; it tests whether action capacity exists
when PICF context is removed.
```

Decision:

```text
If G11b descends while G11a does not:
  action head/adapter capacity is the short-run bottleneck.

If G11b also fails:
  the root is not only PICF context nor adapter capacity.
```

Resource-safe implementation:

```text
G11b first launch used DDP and no window checkpointing.  It OOMed before
producing a training metric row on 2x40GB:

  torch.OutOfMemoryError during GELU forward

That is a resource block, not a scientific rejection.  The valid retry is
G11b2 with the same scientific variables but FSDP full-shard and window
activation checkpointing:

  PICF_ACTION_CONDITION_ENABLED=0
  ACCUM_STEPS=1
  LOGICAL_BATCH_TASK_COUNT=2
  SEMANTIC_TRAINABLE_SCOPE=backbone_only
  TRAINING_STRATEGY=fsdp_full_shard
  WINDOW_ACTIVATION_CHECKPOINTING=1
```

G11b2 first detailed metric:

```text
Status: rejected for production, informative for diagnosis.

Runtime confirmed:
  training_strategy=fsdp_full_shard
  window_activation_checkpointing=True
  PICF_ACTION_CONDITION_ENABLED=0
  LOGICAL_BATCH_TASK_COUNT=2
  SEMANTIC_TRAINABLE_SCOPE=backbone_only
  pi_action_condition_token_count=0
  pi_context_token_count=0
  pi_context_adapter_token_count=0

step 11010:
  loss_action_default_equiv = 0.053830
  loss_total_minus_action   = 0.009310
  loss_anchor_pv            = 0.508249
  loss_anchor_object_pull   = 0.070480
  loss_mapg_routing         = 0.437314
  loss_slot_jepa            = 0.727345

step 11020:
  loss_action_default_equiv = 0.043505
  loss_total_minus_action   = 0.009498
  loss_anchor_pv            = 0.469186
  loss_anchor_object_pull   = 0.091006
  loss_mapg_routing         = 0.418009
  loss_slot_jepa            = 0.658605

Interpretation:
  Stronger semantic/backbone trainability did not immediately restore action
  descent.  It improved from 11010 to 11020, but only to roughly the same
  action level as G11a while running much slower.  Therefore adapter capacity
  alone is not the root cause, and this path is not a viable production
  direction on 2x40GB.  Move to G11c bridge-form comparison.
```

### G11c: Prefix-Fusion Bridge

Purpose:

```text
Compare the current suffix cross-attention/action-side bridge with the older
prefix-fusion bridge under the same K4 task-balanced estimator.  This tests the
Flamingo/JEPA-VLA-style gated context injection choice.
```

Config delta:

```text
PICF_ACTION_CONDITION_ENABLED=1
ACTION_CONTEXT_INTEGRATION=prefix_fusion
ACTION_PREFIX_STOPGRAD=1
ACTION_PREFIX_OUTPUT_GATE=0.70
SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter
```

Decision:

```text
If prefix_fusion descends while suffix_cross_attention/G10a fails:
  deploy prefix_fusion or a hybrid gated-prefix bridge.

If it also fails:
  bridge mode alone is not the root.
```

Metrics:

```text
Status: running; early metrics are mixed, not final.

Runtime confirmed:
  PICF_ACTION_CONDITION_ENABLED=1
  ACTION_CONTEXT_INTEGRATION=prefix_fusion
  ACCUM_STEPS=2
  LOGICAL_BATCH_TASK_COUNT=4
  SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter
  pi_action_condition_token_count=4
  pi_context_token_count=24
  pi_context_fused_prefix_token_count=4
  pi_context_gate=0.25

step 11010:
  loss_action_default_equiv = 0.038835
  loss_total_minus_action   = 0.010705
  loss_anchor_pv            = 0.489133
  loss_anchor_object_pull   = 0.208098
  loss_mapg_routing         = 0.428918
  loss_slot_jepa            = 0.691647
  active_same_role_overlap  = 0.124981

step 11020:
  loss_action_default_equiv = 0.041611
  loss_total_minus_action   = 0.009738
  loss_anchor_pv            = 0.475054
  loss_anchor_object_pull   = 0.115423
  loss_mapg_routing         = 0.419787
  loss_slot_jepa            = 0.694658
  active_same_role_overlap  = 0.087500

step 11030:
  loss_action_default_equiv = 0.042580
  loss_total_minus_action   = 0.009229
  loss_anchor_pv            = 0.472663
  loss_anchor_object_pull   = 0.064368
  loss_mapg_routing         = 0.415389
  loss_slot_jepa            = 0.690158
  active_same_role_overlap  = 0.062500

step 11040:
  loss_action_default_equiv = 0.045863
  loss_total_minus_action   = 0.009969
  loss_anchor_pv            = 0.491523
  loss_anchor_object_pull   = 0.130486
  loss_mapg_routing         = 0.423539
  loss_slot_jepa            = 0.681311
  active_same_role_overlap  = 0.100000

Interpretation through 11040:
  Prefix fusion improves the structural side cleanly:
    total_minus_action 0.010705 -> 0.009229
    anchor_object_pull 0.208098 -> 0.064368
    active_same_role_overlap 0.124981 -> 0.062500

  Action, however, does not show descent and crosses the early-stop band:
    0.038835 -> 0.041611 -> 0.042580 -> 0.045863

  Therefore prefix_fusion is not accepted as the deployment fix.  It is
  diagnostically useful because it shows bridge form changes structural
  cleanliness, but it does not solve action descent.  G11c was stopped early
  and G11d append-mode negative control was launched.
```

### G11d: Append-Context Negative Control

Purpose:

```text
Append mode shifts prefix/suffix layout and is expected to be risky.  It is
included only as a sanity check that "more context tokens" is not the answer.
```

Config delta:

```text
ACTION_CONTEXT_INTEGRATION=append
ACTION_CONTEXT_TOKENS=24
```

Decision:

```text
Accept only if it clearly beats G11a/b/c, which is not expected.  Otherwise
keep append as legacy/diagnostic only.
```

Result:

```text
Status: rejected and early-stopped.

Runtime confirmed:
  PICF_ACTION_CONDITION_ENABLED=1
  ACTION_CONTEXT_INTEGRATION=append
  ACCUM_STEPS=2
  LOGICAL_BATCH_TASK_COUNT=4
  SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter
  pi_action_condition_token_count=28
  pi_context_token_count=24
  pi_context_fused_prefix_token_count=0

step 11010:
  loss_action_default_equiv = 0.054851
  loss_total_minus_action   = 0.010910
  loss_anchor_pv            = 0.489586
  loss_anchor_object_pull   = 0.227279
  loss_mapg_routing         = 0.431503
  loss_slot_jepa            = 0.692816
  active_same_role_overlap  = 0.137481

Interpretation:
  Append mode is worse than G11a/G11b2/G11c on action at the first aligned
  checkpoint.  This rejects the hypothesis that adding more PICF tokens to the
  action condition is sufficient.  It also warns against treating context-token
  count as a substitute for a clean action interface.
```

## Required Non-Training Audits During G11

These are not optional because they prevent repeating the same mistake.

### A1: Bucket Sampler Audit

Run the existing audit on the same segment/sidecar set:

```text
scripts/picf_calvin_bucket_sampler_audit.py
```

Required output:

```text
target q_b
observed q_b
max_abs_error
KL
bucket counts
```

Result:

```text
Status: pass.

Command:
  scripts/picf_calvin_bucket_sampler_audit.py
    --calvin-root /mnt/calvin_data/task_ABC_D
    --split training
    --world-size 2
    --accum-steps 2
    --steps 256
    --unroll-steps 2
    --calvin-bucket-sampling-mode task_uniform
    --calvin-bucket-sample-without-replacement

Observed:
  global_micro_count = 1024
  min_distinct_buckets_per_step = 4
  mean_distinct_buckets_per_step = 4.0
  max_distinct_buckets_per_step = 4
  KL(empirical || target) = 0.001131
  max_abs_deviation = 0.010463

Interpretation:
  The logical-batch sampler is functioning as designed.  The action plateau is
  not caused by a dataflow failure where optimizer steps secretly see one
  bucket only.
```

### A2: Gradient-Cosine Probe

Run:

```text
scripts/picf_bucket_gradient_cosine_probe.py
```

Required groups:

```text
semantic
policy_head
picf_core
```

Required losses:

```text
loss_action
loss_total_minus_action
```

Interpretation:

```text
if action/policy_head cosines are strongly negative:
  task-family conflict remains; action-expert MoE or adapter-level PCGrad may
  be justified, but only scoped to the policy/action expert.

if cosines are not strongly negative:
  do not add MoE/PCGrad; focus on bridge/data/window semantics.
```

### A3: Action Window / Target Scale Audit

This audit must compare the sampled action targets by bucket:

```text
action norm mean/std/p95
delta position norm
rotation norm
gripper state ratio
prompt bucket
window start/end distribution
sidecar coverage ratio
```

If G11a/b/c all fail, this is the next root candidate.  It is more likely than
another optimizer tweak.

## Final Deployment Decision Tree

```text
G11a positive:
  deploy no-PICF-action or heavily gated PICF-action bridge for the next 30k,
  while keeping PICF structural losses as auxiliary belief training.

G11b positive only:
  deploy fuller semantic/action capacity or a larger action adapter; keep PICF
  bridge gated until action proves stable.

G11c positive:
  deploy prefix_fusion/hybrid gated-prefix bridge.

G11d positive unexpectedly:
  inspect position layout carefully before accepting; append may hide a
  suffix-position bug rather than fix the model.

all G11 negative:
  stop sampler/bridge tinkering.  Run action-window/target-scale audit and
  consider action expert architecture or data normalization.
```

## Why This Is Not A Minimal Patch

This plan covers every requested method at the appropriate level:

```text
implemented and tested:
  task/dataset mixing
  K4 logical batch
  per-bucket normalization
  cheap dynamic action scaling
  adapter PCGrad
  policy-only action boundary
  typed modality adapters
  continuous action chunk path

now testing:
  no-PICF-action bridge causality
  stronger semantic/action capacity without PICF context
  prefix vs suffix gated context injection
  append negative control
  bucket/gradient/action-window audits

deferred with reason:
  full PiKE dynamic mixing: needs positive static recipe or gradient proof
  full CAGrad/PCGrad: too expensive without measured conflict
  action MoE: only after persistent task-family conflict is proven
  embodiment heads: CALVIN has one embodiment
  action head rewrite: E21 proved native action path has capacity
```

This is the full current scientific matrix.  It is not a claim that the model
is solved.  It is the fastest non-duplicative path to a falsifiable answer.

## Running Results

### G11a Result: No-PICF-Action K4 Adapter

Status: `[!]` rejected and early-stopped.

Runtime confirmation:

```text
PICF_ACTION_CONDITION_ENABLED=0
ACTION_CONTEXT_INTEGRATION=suffix_cross_attention
ACCUM_STEPS=2
LOGICAL_BATCH_TASK_COUNT=4
SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter
PICF_TRAINABLE_SCOPE=all
logical_batch_distinct_bucket_count=4
pi_action_condition_token_count=0
pi_context_token_count=0
pi_context_adapter_token_count=0
```

This confirms the case is a real no-PICF-action condition test, not a bridge
mode typo.

First rows:

```text
step 11010:
  loss_action_default_equiv = 0.040494
  loss_total_minus_action   = 0.010704
  loss_anchor_pv            = 0.489133
  loss_anchor_object_pull   = 0.208039
  loss_mapg_routing         = 0.428918

step 11020:
  loss_action_default_equiv = 0.042474
  loss_total_minus_action   = 0.009553
  loss_anchor_pv            = 0.474841
  loss_anchor_object_pull   = 0.098223
  loss_mapg_routing         = 0.417151

step 11030:
  loss_action_default_equiv = 0.042712
  loss_total_minus_action   = 0.009294
  loss_anchor_pv            = 0.473098
  loss_anchor_object_pull   = 0.070164
  loss_mapg_routing         = 0.417994

step 11040:
  loss_action_default_equiv = 0.046490
  loss_total_minus_action   = 0.009709
  loss_anchor_pv            = 0.492147
  loss_anchor_object_pull   = 0.105738
  loss_mapg_routing         = 0.423542
```

Interpretation:

```text
Removing PICF from the action condition does not restore action descent.
The same structural/action split remains: structural losses improve while
action worsens.  Therefore the single-cause hypothesis "PICF action context is
poisoning action" is rejected.  G11 moves to G11b, which tests whether the
action_head_and_adapter scope is too weak rather than whether PICF context is
bad.
```

## Complete Requested-Method Matrix

This section tracks every method requested in the VLA training-system repair
discussion.  The status words are strict:

```text
implemented:
  code/dataflow exists and local/remote script checks passed.

tested:
  a real CALVIN training/audit run reached enough steps or produced a diagnostic
  artifact to accept/reject the mechanism for the current failure.

rejected-as-single-fix:
  mechanism may remain useful as part of the final recipe, but evidence says it
  does not by itself fix the plateau/rebound.

evidence-triggered:
  do not deploy blindly; run only when the prerequisite diagnostic indicates
  that this mechanism addresses the measured failure.
```

| Requested method | Status | Evidence / next action |
| --- | --- | --- |
| VLA-Foundry-style probabilistic/task mixing | implemented + tested | `task_uniform`, `trajectory`, `temperature`, explicit bucket weights exist in `picf_core_train.py`; sampler audit A1 confirms K4 task-uniform covers 4 distinct buckets per optimizer step. |
| Batch balancing ratios | implemented + tested | `--calvin-bucket-weight-spec` implements ratio specs; not sufficient alone because task-uniform K4 still failed action descent in G11. |
| Gradient accumulation as logical batch | implemented + tested | `WORLD_SIZE * accum_steps` controls logical task count; K4 on 2xA100 through accum=2 was verified by A1 and G11 runs. |
| Per-task/bucket loss normalization | implemented + tested | `--logical-batch-bucket-normalization` scales each selected bucket before backward and compensates DDP averaging; G11 used it. |
| Per-bucket action EMA / cheap dynamic action scaling | implemented + rejected-as-single-fix | F9b/E-series showed this is not enough; it changes scalar magnitude, not gradient direction. |
| Task-uniform sampling | implemented + tested | Keep as default for CALVIN; it solves coverage, not semantic/action gradient conflict. |
| Temperature sampling | implemented + tested as sampler mode | Available for large datasets; not the current root until A3 shows bucket scale imbalance. |
| Dynamic PiKE-style mixing | evidence-triggered | Not deployed blindly. A2/A3 decide whether dynamic task weights are justified. If A2 shows non-adversarial but imbalanced grad norms, use PiKE-like weighting; if A2 shows strong negative cosine, use gradient surgery/action-expert separation first. |
| Gradient cosine diagnosis | implemented + tested | A2 result below: action gradients conflict strongly on semantic/action-adapter group. |
| PCGrad/CAGrad | partially implemented + evidence-triggered | `pcgrad` exists for logical-batch groups. Semantic-only PCGrad was previously runnable but not sufficient. A2 now justifies a narrower adapter/action-head conflict experiment, not whole-model PCGrad. |
| Whole-model gradient surgery | rejected for now | Too expensive and not aligned with measured conflict; action gradient into PICF core is zero under the current bridge. |
| VLM/action boundary / knowledge insulation | implemented in current PICF bridge, tested | Action does not directly gradient-update PICF core in A2. Prefix/append bridge variants did not fix action descent alone. |
| Modality adapters/projectors | implemented | PICF already has modality-specific adapters/projectors and typed memories. No separate CALVIN embodiment adapter needed yet because CALVIN is single embodiment. |
| Per-embodiment heads/adapters | evidence-triggered, not current CALVIN | Required for future multi-robot scaling; not a CALVIN plateau root because current run has one embodiment/action space. |
| Continuous action chunk | implemented | `action_chunk` path is native and audited by A3. This is not an AR action-token setup. |
| L1/Huber action objective / action scale normalization | implemented in current action path + A3 audit | A3 checks normalized target saturation and per-bucket target scale to determine whether action target normalization is a root cause. |
| Action expert MoE | evidence-triggered | Only if A2/A3 show persistent task-family conflict after balanced logical batch and adapter-level gradient handling. Do not put MoE into the whole VLM. |
| System-2/System-1 or subtask planning head | deferred | Architecturally compatible, but not a 1-2h diagnostic. Use after CALVIN action descent is restored. |
| Larger batch / 6-GPU replication | tested as resource/capacity probe | Large coverage helps, but relying on physical batch harms scalability. Final recipe must emulate coverage through logical batch + normalization. |

## A1 Sampler Audit Result

Command:

```bash
/root/openpi/.venv/bin/python scripts/picf_calvin_bucket_sampler_audit.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --split training \
  --world-size 2 \
  --accum-steps 2 \
  --steps 256 \
  --unroll-steps 2 \
  --calvin-bucket-sampling-mode task_uniform \
  --calvin-bucket-sample-without-replacement
```

Result:

```text
global_micro_count = 1024
min_distinct_buckets_per_step = 4
mean_distinct_buckets_per_step = 4.0
max_distinct_buckets_per_step = 4
KL(empirical || target) = 0.001131
max_abs_deviation = 0.010463
```

Conclusion:

```text
The logical-batch sampler is not secretly issuing single-task optimizer steps.
The action plateau is not caused by fake K4 coverage.
```

## A2 Gradient-Cosine Result

Command:

```bash
/root/openpi/.venv/bin/python scripts/picf_bucket_gradient_cosine_probe.py \
  --args-json /mnt/checkpoints/picf_core/picf_core/picf_f8r_k4_action_adapter_taskuniform_wor_11000_to11100_20260602/args.json \
  --checkpoint /mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000 \
  --output-json /mnt/picf_run_logs/g11_audits_20260603/gradient_cosine_suffix_baseline.json \
  --device cuda:0 \
  --split training \
  --buckets block_lift,block_other,block_push,drawer,other,slider,switch_button_light \
  --windows-per-bucket 1 \
  --loss-keys loss_action,loss_total_minus_action \
  --groups policy_head,semantic,picf_core \
  --max-elements-per-group 200000 \
  --picf-trainable-scope all \
  --semantic-trainable-scope action_head_and_adapter \
  --semantic-lr-scale 1.0 \
  --policy-head-lr-scale 1.0 \
  --picf-core-lr-scale 0.001 \
  --action-context-integration suffix_cross_attention \
  --action-context-tokens 24 \
  --enable-picf-action-condition
```

Result summary:

```text
loss_action -> semantic/action-adapter:
  finite_pairs = 42
  negative_pairs = 20
  negative_fraction = 0.476190
  min_cosine = -0.189485

loss_action -> picf_core:
  finite_pairs = 0
  gradient path effectively absent under current stop-gradient boundary

loss_total_minus_action -> picf_core:
  finite_pairs = 42
  negative_pairs = 10
  negative_fraction = 0.238095
  min_cosine = -0.174050

policy_head:
  no sampled trainable tensors in this checkpoint/scope probe
```

Conclusion:

```text
The measured root is not direct action gradient corrupting PICF core.  That
path is already insulated.  The measured root is cross-task action-gradient
conflict inside the semantic/action-adapter trainable group.

Therefore the next experiments must focus on adapter/action-head gradient
control or task-family action-expert separation.  More sampler-only runs,
blind action scalar changes, or more PICF anchor losses would not address the
measured negative cosine.
```

## A3 Action-Window / Target-Scale Audit

Status: `[x]` completed on A7.

Command:

```bash
/root/openpi/.venv/bin/python scripts/picf_action_window_target_audit.py \
  --args-json /mnt/checkpoints/picf_core/picf_core/picf_f8r_k4_action_adapter_taskuniform_wor_11000_to11100_20260602/args.json \
  --output-json /mnt/picf_run_logs/g11_audits_20260603/action_window_target_audit_taskuniform_k4_128.json \
  --split training \
  --world-size 2 \
  --accum-steps 2 \
  --steps 128 \
  --start-step 11000 \
  --calvin-bucket-sampling-mode task_uniform \
  --calvin-bucket-sample-without-replacement
```

Result:

```text
sample_count = 512 logical micro samples
logical_step_distinct_bucket_count = 4 / 4 / 4
unroll_steps = 3
action_horizon = 16

raw_action_chunk_l2:
  mean = 1.1490
  p95 = 1.4520
  max = 1.9310

norm_action_chunk_l2:
  mean = 1.3578
  p95 = 1.8959
  max = 3.1971

norm_action_chunk_outside_unit_fraction:
  mean = 0.0842
  p95 = 0.2054
  max = 0.4018

norm_action_chunk_outside_two_fraction:
  mean = 0.000279
  p95 = 0.0
  max = 0.0179

sidecar:
  proposal_count mean = 1.2715
  proposal_mask_point_count mean = 41.8145
  proposal_objectness mean = 0.4448
  tracklet_count mean = 53.8262
  tracklet_confidence mean = 0.3676

per-bucket norm_action_chunk_l2 mean:
  block_lift = 1.3460
  block_other = 1.3707
  block_push = 1.2579
  drawer = 1.4052
  other = 1.3973
  slider = 1.2923
  switch_button_light = 1.4327
```

Interpretation:

```text
The action target/window path is not corrupt:
  no non-finite targets
  raw actions are bounded
  normalized tails exist but are sparse
  per-bucket target scale differs, but the spread is not large enough to
  explain the production action plateau by itself

The sidecar path should stay weak:
  proposal/mask coverage is useful but not full
  tracklet coverage is dense but moderate-confidence
  therefore sidecar must not become a hard action authority or strong mask
  label in production training
```

Decision:

```text
A3 does not justify another action-target normalization rewrite.
A3 does not justify strengthening SAM/sidecar/mask labels.
A3 supports moving to scoped action-gradient conflict repair because A2
already measured negative cross-task action gradients in the semantic/action
adapter group.
```

Future acceptance rule:

```text
If normalized action targets have large bucket-specific saturation or scale
differences:
  fix action target normalization / per-bucket scale before adding PCGrad/MoE.

If sidecar/proposal coverage is bucket-skewed:
  separate sidecar-dependent structural losses from action loss weighting and
  do not let sparse sidecar coverage define action target difficulty.

If A3 is clean:
  proceed to adapter/action-head gradient conflict repair, because A2 already
  proved negative cross-task action gradients in that group.
```

## 8-Hour Execution Plan

This plan is explicitly designed for scalability.  It avoids solutions that
only work by increasing physical batch or overfitting CALVIN task order.

```text
Hour 0-1:
  finish A3 target/window audit.
  update this document with exact bucket target stats.

Hour 1-3:
  run B1 adapter/action-head PCGrad or CAGrad-lite on the semantic/action group
  only, using the measured A2 conflict.
  Do not touch PICF core unless a separate probe shows action gradients there.

Hour 3-5:
  run B2 dynamic task mixing only if A3 shows bucket scale imbalance or A2
  shows mostly non-negative but unequal grad norms.  Otherwise skip PiKE for
  this failure because the measured issue is negative cosine, not only weight.

Hour 5-7:
  run B3 action-expert boundary variant:
    keep semantic/action-adapter trainable,
    restrict continuous action gradient to the action adapter/head bridge,
    preserve representation/structural losses outside the action gradient path.
  This tests Knowledge-Insulation-style action boundary without replacing the
  whole model.

Hour 7-8:
  compare action trend, total_minus_action, anchor/object pull, routing,
  slot_jepa, active overlap, gradient diagnostics, and speed.
  choose final deployment:
    fixed task-balanced K4 + adapter gradient control,
    or fixed task-balanced K4 + target-scale repair,
    or action-expert boundary revision.
```

Hard rejection rules:

```text
Do not deploy full PiKE before A3/A2 justify it.
Do not deploy whole-model PCGrad.
Do not deploy action MoE unless adapter/action-head conflict remains after B1.
Do not add more anchor losses for an action-gradient conflict.
Do not call physical 6-GPU large batch a scalable fix.
```

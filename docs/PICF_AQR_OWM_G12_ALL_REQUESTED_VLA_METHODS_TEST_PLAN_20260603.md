# PICF-AQR-OWM G12 All Requested VLA Methods Test Plan

Date: 2026-06-03

This document is the execution contract for the user's full requested method
set: task-balanced logical batches, task/dataset mixing, gradient
accumulation, per-task/per-modality normalization, action/VLM gradient
boundary, modality/action adapters, continuous action chunks, gradient conflict
diagnosis, PCGrad/CAGrad, dynamic mixing, and action-expert/MoE-style
separation.

The purpose is not to prove that every advanced VLA method should be deployed.
The purpose is to test every relevant method class quickly, reject the ones
that do not address the measured failure, and converge to a scalable deployment
recipe.  Any accepted method must remain valid for large heterogeneous future
datasets with missing modalities and multiple embodiments.

## Primary Paper Grounding

The G12 matrix is grounded in current VLA / large multitask training practice,
but it does not blindly import modules whose assumptions do not match this
codebase.

```text
VLA Foundry, 2026:
  Supports unified VLA training with controlled data/task mixture as a first
  class framework concern.  G12 maps this to explicit bucket weights,
  temperature sampling, and per-bucket normalization.

ABot-M0, 2026:
  Emphasizes large heterogeneous robot data curation, standardization, and
  balanced sampling across tasks / embodiments.  G12 maps this to task-family
  bucket balance now and keeps embodiment hooks for future multi-robot data.

PiKE, 2025:
  Supports adaptive data mixing when gradients have low conflict and task
  progress differs.  G12 therefore treats dynamic mixing as evidence-triggered,
  not as a default fix under measured negative action-gradient cosine.

OpenVLA-OFT, 2025:
  Supports continuous action chunks and simple regression-style objectives as a
  stable and efficient VLA fine-tuning recipe.  PICF already uses continuous
  action chunks; G12 therefore audits scale/window first instead of rewriting
  the action head.

Knowledge Insulation, 2025:
  Shows that continuous action gradients can damage semantic VLM knowledge.
  G12 therefore keeps action/PICF stop-gradient and tests only the semantic /
  action-adapter bridge when needed.

AdaMoE-VLA / FedVLA-style expert routing, 2025:
  Supports action-specialized experts under task conflict.  G12 keeps MoE as a
  final action-expert-only branch, not a VLM-wide rewrite, because CALVIN is a
  single-embodiment gate and whole-backbone MoE would weaken scaling hygiene.
```

References:

```text
VLA Foundry: https://arxiv.org/abs/2604.19728
ABot-M0: https://arxiv.org/abs/2602.11236
PiKE: https://arxiv.org/abs/2502.06244
OpenVLA-OFT: https://arxiv.org/abs/2502.19645
Knowledge Insulation: https://arxiv.org/abs/2505.23705
AdaMoE-VLA: https://arxiv.org/abs/2510.14300
```

## Current Evidence Entering G12

```text
A1 sampler audit:
  K4 task_uniform without-replacement logical batch is real.
  Each optimizer step saw exactly 4 distinct buckets.

A2 gradient-cosine probe:
  action gradients conflict inside semantic/action-adapter trainable group.
  loss_action -> semantic/action-adapter:
    negative_fraction = 0.476190
    min_cosine = -0.189485
  loss_action -> PICF core:
    no direct finite action gradient under current stop-gradient boundary.

A3 action-window/target-scale audit:
  action target/window path is not corrupt.
  raw action chunks are bounded.
  normalized tail exists but is sparse.
  sidecar/proposal coverage is useful but not dense enough for hard labels.
```

Therefore the current root hypothesis is:

```text
The plateau is not primarily "batch was secretly single-task" and not an
obvious action-target bug.  The measured remaining issue is task-family action
gradient conflict in the semantic/action-adapter bridge, possibly amplified by
insufficient logical task coverage when physical batch is small.
```

## 2026-06-03 Mid-Gate Status

This section records what has actually been tested after the user's request to
cover every proposed VLA repair class.  It is deliberately strict: a method is
not counted as solved unless it changed the measured action trend, not merely
the structural auxiliary losses.

```text
G12-K8:
  K=8 logical task coverage through accum=4 was resource-rejected on the
  available 2x40G/A100-style gate.  It ran only through FSDP/checkpointing at
  roughly 130 sec/step, so it is not a practical 1-2 hour diagnostic.  This is
  not a model rejection, but it proves K8 is not the first-line dual-card gate.

G12-TEMP alpha=0.5:
  Correctly ran DDP/no-checkpoint K4 with q_b proportional to sqrt(N_b).
  Observed action trend:
    step 11010: loss_action_default_equiv = 0.036048
    step 11020: loss_action_default_equiv = 0.046093, preclip_grad_norm = 16.414
    step 11030: loss_action_default_equiv = 0.042926
    step 11040: loss_action_default_equiv = 0.043974
  Verdict:
    rejected as an action fix.  It can reduce pure task-uniform over-sampling,
    but it does not remove the plateau/rebound.

G12-RATIO explicit bucket weights:
  Correctly ran VLA-Foundry-style explicit bucket ratios:
    block_push/drawer/switch = 1.25
    block_lift/block_other/slider = 1.0
    other = 0.75
  Observed action trend:
    step 11010: loss_action_default_equiv = 0.045747
    step 11020: loss_action_default_equiv = 0.046368
  Structural terms improved, e.g. object_pull 0.300100 -> 0.094538.
  Verdict:
    rejected as an action fix; accepted only as a supported sampling knob.

G12-PCGrad semantic/action-adapter:
  Correctly ran scoped PCGrad over the semantic/action-adapter owner group:
    target tensors = 14
    target numel = 8,457,249
  Observed action trend:
    step 11010: loss_action_default_equiv = 0.040655
    step 11020: loss_action_default_equiv = 0.042560
    step 11040: loss_action_default_equiv = 0.046892
  Structural terms improved, e.g. anchor_pv 0.489586 -> 0.475054 and
  object_pull 0.227279 -> 0.115387.
  Verdict:
    rejected as an action fix.  It supports the conclusion that structure
    supervision can improve while the action expert remains under-optimized.

G12-DYN bounded PiKE-style dynamic task mixing:
  Implemented as sampler-side q_b(t), not as a metric-only rescale.  The same
  dynamic q_b(t) is used for bucket sampling and for the q_b / n_b logical loss
  estimator.  It is bounded by a nonzero floor and max-weight cap so no task
  family can be starved.
  Dataflow gate:
    dynamic_mixing=True at startup
    logical_batch_dynamic_mixing_active=True in metrics
    logical_batch_distinct_bucket_count=4
    min/max dynamic weight at step 11010: 0.087698 / 0.226114
    min/max dynamic weight at step 11020: 0.089687 / 0.196926
    min/max dynamic weight at step 11030: 0.087083 / 0.188602
  Observed action trend:
    step 11010: loss_action_default_equiv = 0.040642
    step 11020: loss_action_default_equiv = 0.040795
    step 11030: loss_action_default_equiv = 0.041484
    step 11040: loss_action_default_equiv = 0.045887
    step 11050: loss_action_default_equiv = 0.048912
  Structural terms improved:
    loss_anchor_object_pull: 0.213624 -> 0.177617 -> 0.092453
    loss_anchor_pv:          0.497226 -> 0.486837 -> 0.482442
    loss_mapg_routing:       0.434001 -> 0.425147 -> 0.415072
    loss_slot_jepa:          0.700523 -> 0.697731 -> 0.686433
  Later structure also reverted by step 11050:
    loss_anchor_object_pull = 0.255357
    loss_anchor_pv = 0.498692
    loss_mapg_routing = 0.438012
  Verdict at step 11050:
    rejected as an action fix.  The implementation is useful infrastructure
    for future heterogeneous data, but on the current CALVIN gate it behaves
    like another sampler/weighting perturbation: it can move bucket emphasis
    and transiently improve structure, but it does not solve action plateau.
    The run was stopped at 11057 progress.

G12-CAGrad semantic/action-adapter:
  Implemented locally as scoped CAGrad-lite over the same per-micro gradients
  used by PCGrad.  It solves a small simplex problem over logical-batch micro
  gradients and optionally rescales to the raw logical-batch gradient norm.
  Scope:
    semantic/action-adapter by default.
  Explicitly not allowed by default:
    whole-model CAGrad, because A2 measured no direct finite action gradient
    into PICF core under the current stop-gradient boundary.
  Local checks:
    py_compile passed.
    relevant pytest gate passed:
      13 passed, 136 deselected.
  Remote training gate:
    started on A7 after G12-DYN rejection:
      tmux session = picf_a7_g12_cagrad_20260603
      log = /mnt/picf_run_logs/g12_all_requested_vla_methods_20260603/g12_cagrad_semantic_k4.log
      checkpoint = step11000
      target = step11200
    Startup confirmed:
      gradient_surgery=cagrad
      gradient_surgery_groups=semantic
      target tensors = 14
      target numel = 8,457,249
      logical_batch_distinct_bucket_count = 4
    Observed action trend:
      step 11010: loss_action_default_equiv = 0.040583
      step 11020: loss_action_default_equiv = 0.042519
      step 11030: loss_action_default_equiv = 0.042945
    Structural terms improved while action did not:
      loss_anchor_pv:          0.489133 -> 0.474841 -> 0.472662
      loss_anchor_object_pull: 0.208098 -> 0.098223 -> 0.064368
      loss_mapg_routing:       0.428918 -> 0.417150 -> 0.415389
      loss_total_minus_action: 0.010705 -> 0.009553 -> 0.009229
    Verdict at step 11030:
      rejected as an action fix.  CAGrad behaves like PCGrad/DYN in this
      gate: it can improve structural/object supervision terms, but it does
      not make the action objective descend.  The session was stopped early
      to avoid spending the rest of the hour on a repeated failure mode.
```

Implication:

```text
The first-line scalable sampler fixes are now substantially tested:
  task_uniform K4, temperature K4, explicit-ratio K4, K8 resource probe,
  per-bucket normalization, scoped PCGrad, bounded dynamic mixing, and scoped
  CAGrad.

The remaining open classes are not "more batch tuning" and not another scalar
loss-weight run.  The next test must decompose the native PI0.5 action expert
boundary:
  1. action_head_only under the same K4 logical-batch contract;
  2. action_adapter_only under the same K4 logical-batch contract;
  3. action_head_and_adapter as the already-tested control;
  4. only if conflict persists after this decomposition, consider
     action-expert-only MoE or stronger expert routing.
```

## Method Matrix

| Method class | Must test? | G12 test | Deploy only if |
| --- | --- | --- | --- |
| Task-uniform logical batch | Already tested; keep | Included in every G12 run as baseline contract | Always, unless future data is not task-labeled |
| Temperature sampling | Yes, because it is scalable | G12-TEMP, alpha=0.5 | Better action slope than task_uniform without worse structure |
| Explicit ratio mixing | Yes | G12-RATIO, equal or inverse-frequency bucket weights | Better than task_uniform on action and no bucket starvation |
| Larger logical batch via accumulation | Yes | G12-K8, accum_steps=4 on 2 GPUs | Better action slope per optimizer step and acceptable wall time |
| Per-task/bucket normalization | Already implemented; keep | On in all G12 accepted runs | Always, because it makes estimator match `sum_b q_b L_b` |
| Per-modality normalization | Architecture-level; not a separate CALVIN quick test | Verify current action/PICF/sidecar losses are independently scaled | Keep as design rule |
| Dynamic PiKE-style mixing | tested and rejected as action fix | G12-DYN with bounded `q_b(t)` from bucket action-loss EMA/progress | Keep infrastructure for future heterogeneous data; do not use as current default |
| Gradient cosine diagnosis | Already done; repeat after candidate if needed | A2 result is trigger; repeat after G12-PCG or DYN | Use to justify surgery/MoE, not as a loss |
| PCGrad/CAGrad | Yes, but scoped | G12-PCG and G12-CAGrad done | Rejected as action fixes; keep code as diagnostic infrastructure |
| Whole-model PCGrad/CAGrad | No | Not run | Rejected: action gradients do not enter PICF core; cost/scaling mismatch |
| Action/VLM insulation | Yes | G12-BRIDGE: keep action context stopgrad; compare semantic scope/action adapter only | Better action slope and no structure collapse |
| Modality adapters | Already implemented | Verified by trainable scope and typed projectors | Keep |
| Embodiment adapters/heads | Not a CALVIN quick test | Document future hook only | Add when multi-robot data exists |
| Continuous action chunking | Already native | A3 audited action_horizon=16 | Keep |
| Huber/L1 action objective | Current PI0.5 path already continuous; no full action rewrite now | If action-target scale was bad, change objective; A3 did not trigger | Defer full rewrite |
| Action expert MoE | Not first-line | Only after PCGrad/bridge fail and gradient conflict remains | Must be inside action expert only, not VLM-wide |
| System2/System1 planning split | Not for this 8h gate | Document future long-horizon branch | Add only with subtask labels or reliable pseudo-subtasks |

## Final 8-Hour All-Point Plan

The remaining work is split into tests that can finish in 1-2 hours each and
deployment work that is allowed only after a positive test.  This avoids both
"only increase batch" and unbounded architecture churn.

### Experiment Part

| Order | Test | Runtime target | What it proves | Stop condition |
| --- | --- | --- | --- | --- |
| E1 | Finish/read G12-PCGrad to at least 100 optimizer steps if not already stopped | <= 1 hour | whether scoped gradient surgery changes action, not only structure | stop if action slope remains non-negative |
| E2 | Repeat bucket gradient-cosine probe on the current best G12 checkpoint | <= 1 hour | whether negative action conflict remains after sampler/PCGrad attempts | if negative_fraction remains high, dynamic q alone is insufficient |
| E3 | Dynamic mixing code/dataflow gate | <= 1 hour | whether `q_b(t)` is logged, bounded, and used in both sampling and `q_b/n_b` loss scale | reject if dynamic fields are absent or weights violate floor/cap |
| E4 | Dynamic mixing training gate, K4 | 1-2 hours | whether q_b(t) adaptation beats static q_b | reject if action slope non-negative by 100 steps |
| E5 | Stronger action-boundary/capacity gate | 1-2 hours | whether action expert capacity/boundary, not sampler, is the bottleneck | reject if action does not improve while structure stays stable |
| E6 | Action-expert-only MoE proxy or CAGrad gate | 1-2 hours | whether task-family conflict needs experts rather than scalar mixing | accept only if action improves across multiple buckets |

## G13 Action-Boundary Decomposition

G12 rejected the first-line scalable data/mixing/surgery family as action-loss
fixes.  The next controlled gate is not another sampler variant.  It decomposes
the native PI0.5 action boundary under the exact same data estimator:

```text
fixed across all G13 cases:
  checkpoint = step11000
  K4 task-uniform logical batch
  without-replacement task buckets
  per-bucket target normalization
  action weight = 4.0
  PICF action context = suffix_cross_attention
  context_stopgrad = true
  PICF core LR scale = 0.001
  optimizer checkpoint mode = model-only

changed variable:
  semantic_trainable_scope
```

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_g13_action_boundary_decomposition_20260603.sh
```

Cases:

```text
g13_action_head_only_k4:
  trains wrapper-local action_in/out projections and time MLPs only.
  Tests whether the action expert head can adapt without moving the
  PICF-to-action context adapter.

g13_action_adapter_only_k4:
  trains only the suffix cross-attention PICF-to-action context adapter.
  Tests whether action failure is caused by an underfit context bridge.

g13_action_head_and_adapter_control_k4:
  reproduces the current control boundary.
  Included so the matrix does not compare against stale G10/G12 logs after
  code changes.
```

Acceptance:

```text
accept a boundary if:
  loss_action_default_equiv decreases by steps 11020/11030,
  loss_total_minus_action does not explode,
  anchor_pv / object_pull / routing remain in the G12 healthy band,
  logical_batch_distinct_bucket_count remains 4.

reject if:
  action repeats the 0.040 -> 0.042/0.043 non-descent pattern.
```

Execution status:

```text
2026-06-03 local/remote verification:
  bash -n launchers: pass
  py_compile trainer/tests: pass
  pytest selected logical-batch / surgery / trainable-scope gate:
    13 passed, 136 deselected

2026-06-03 first G13 run:
  case = g13_action_head_only_k4
  tmux = picf_a7_g13_head_20260603
  log = /mnt/picf_run_logs/g13_action_boundary_decomposition_20260603/g13_action_head_only_k4.log
  startup confirmed:
    world_size = 2
    accum_steps = 2
    logical_batch_task_count = 4
    logical_batch_distinct_bucket_count target = 4
    bucket_sampling = task_uniform
    bucket_normalization = true
    semantic = paligemma(trainable=True scope=action_head_only)
    semantic trainable params = 2,165,792
    picf_core lr = 2e-08
    gradient_surgery = off
  first row:
    step 11010:
      loss_action_default_equiv = 0.040912
      loss_total_minus_action = 0.010910
      loss_anchor_pv = 0.489586
      loss_anchor_object_pull = 0.227279
      loss_mapg_routing = 0.431503
      loss_slot_jepa = 0.692816
      logical_batch_distinct_bucket_count = 4

  observed trend:
    step 11010:
      loss_action_default_equiv = 0.040912
      loss_total_minus_action = 0.010910
      loss_anchor_pv = 0.489586
      loss_anchor_object_pull = 0.227279
      loss_mapg_routing = 0.431503
      loss_slot_jepa = 0.692816
    step 11020:
      loss_action_default_equiv = 0.042755
      loss_total_minus_action = 0.009553
      loss_anchor_pv = 0.474841
      loss_anchor_object_pull = 0.098224
      loss_mapg_routing = 0.417150
      loss_slot_jepa = 0.695990
    step 11030:
      loss_action_default_equiv = 0.042882
      loss_total_minus_action = 0.009229
      loss_anchor_pv = 0.472665
      loss_anchor_object_pull = 0.064368
      loss_mapg_routing = 0.415389
      loss_slot_jepa = 0.690150

Verdict:
  g13_action_head_only_k4 is rejected as an action fix.  It reproduces the
  recurring pattern seen in PCGrad/CAGrad/DYN: structure/object terms improve
  but action does not descend.  This rejects the hypothesis that local
  wrapper action projections alone are the bottleneck.

Next case:
  g13_action_adapter_only_k4.  This isolates whether the PICF-to-action
  context bridge itself is underfit or mis-scaled.

2026-06-03 second G13 run:
  case = g13_action_adapter_only_k4
  tmux = picf_a7_g13_adapter_20260603
  log = /mnt/picf_run_logs/g13_action_boundary_decomposition_20260603/g13_action_adapter_only_k4.log
  startup confirmed:
    world_size = 2
    accum_steps = 2
    logical_batch_task_count = 4
    bucket_sampling = task_uniform
    bucket_normalization = true
    semantic = paligemma(trainable=True scope=action_adapter_only)
    semantic trainable params = 6,291,457
    picf_core lr = 2e-08
    gradient_surgery = off
  Decision pending:
    Need step 11010/11020/11030.  If this branch descends while head-only
    failed, the context bridge is the actionable boundary bottleneck.  If it
    fails too, the next accepted conclusion is that current head/adapter
    decomposition is insufficient and the remaining architecture branch must
    be action-expert capacity/routing, not another scalar sampler tweak.

  observed trend:
    step 11010:
      loss_action_default_equiv = 0.040835
      loss_total_minus_action = 0.010911
      loss_anchor_pv = 0.489586
      loss_anchor_object_pull = 0.227321
      loss_mapg_routing = 0.431503
      loss_slot_jepa = 0.690765
      grad_norm = 0.018260
    step 11020:
      loss_action_default_equiv = 0.042570
      loss_total_minus_action = 0.009732
      loss_anchor_pv = 0.475052
      loss_anchor_object_pull = 0.114870
      loss_mapg_routing = 0.419787
      loss_slot_jepa = 0.694764
      grad_norm = 0.017440
    step 11030:
      loss_action_default_equiv = 0.042687
      loss_total_minus_action = 0.009294
      loss_anchor_pv = 0.473094
      loss_anchor_object_pull = 0.070163
      loss_mapg_routing = 0.417994
      loss_slot_jepa = 0.691197
      grad_norm = 0.022507

Verdict:
  g13_action_adapter_only_k4 is rejected as an action fix.  It has much smaller
  semantic/action gradient norm than action_head_only, improves the same
  structural terms, and still repeats the action non-descent pattern.  This
  rejects the hypothesis that an underfit suffix-cross-attention context
  adapter alone is the missing production ingredient.

Current G13 implication:
  action_head_only failed and action_adapter_only failed under the same K4
  logical-batch contract.  The remaining issue is not isolated to one
  wrapper-local projection set.  If the control reproduces the same pattern,
  the next deployable architecture class is action-expert capacity/routing
  inside the action path, or a stronger semantic-action insulation recipe.

2026-06-03 third G13 run:
  case = g13_action_head_and_adapter_control_k4
  tmux = picf_a7_g13_control_20260603
  log = /mnt/picf_run_logs/g13_action_boundary_decomposition_20260603/g13_action_head_and_adapter_control_k4.log
  startup confirmed:
    world_size = 2
    accum_steps = 2
    logical_batch_task_count = 4
    bucket_sampling = task_uniform
    bucket_normalization = true
    semantic = paligemma(trainable=True scope=action_head_and_adapter)
    semantic trainable params = 8,457,249
    picf_core lr = 2e-08
    gradient_surgery = off

  observed trend:
    step 11010:
      loss_action_default_equiv = 0.040630
      loss_total_minus_action = 0.010911
      loss_anchor_pv = 0.489586
      loss_anchor_object_pull = 0.227331
      loss_mapg_routing = 0.431503
      loss_slot_jepa = 0.690762
    step 11020:
      loss_action_default_equiv = 0.042539
      loss_total_minus_action = 0.009553
      loss_anchor_pv = 0.474841
      loss_anchor_object_pull = 0.098222
      loss_mapg_routing = 0.417150
      loss_slot_jepa = 0.695991
    step 11030:
      loss_action_default_equiv = 0.042941
      loss_total_minus_action = 0.009229
      loss_anchor_pv = 0.472661
      loss_anchor_object_pull = 0.064367
      loss_mapg_routing = 0.415389
      loss_slot_jepa = 0.690146

Verdict:
  g13_action_head_and_adapter_control_k4 is rejected as an action fix and
  confirms the same pattern as G12 and the two G13 decompositions.

G13 final conclusion:
  The action objective does not descend under:
    1. action_head_only;
    2. action_adapter_only;
    3. action_head_and_adapter.

  In all three cases, structural/object terms improve while action increases
  from about 0.0406-0.0409 to about 0.0425-0.0429.  Therefore the current
  failure is not an isolated local projection or isolated context-adapter
  bottleneck.  It is a deeper action-path optimization issue under the current
  PI0.5/PICF conditioning interface.

Next architecture class:
  Do not repeat sampler/scalar/surgery/boundary-decomposition runs unless code
  changes in the causal path.  The next evidence-based test must target the
  action path itself:
    A. action-expert capacity/routing inside the action path only;
    B. stronger semantic-action insulation with an auxiliary representation
       target, not direct continuous-action gradient into PICF core;
    C. action objective form or action chunk head if capacity/routing remains
       negative.

  Whole-VLM MoE, whole-model gradient surgery, or more bucket scalar tuning are
  still rejected by evidence/cost/scaling fit.
```

Escalation:

```text
If action_head_only wins:
  the bottleneck is local action flow/time projection adaptation.  Keep
  context adapter stable and avoid action-expert MoE.

If action_adapter_only wins:
  the bottleneck is PICF context bridge calibration.  Keep action head stable
  and consider stronger gated cross-attention / context normalization.

If head+adapter wins but both single-scope branches fail:
  the update is coupled; keep both but consider scoped CAGrad/orthogonalization
  only inside this small action boundary.

If all three fail:
  action-expert capacity/routing, not scalar mixing, is the next root class.
  Only then consider action-expert-only MoE or a larger action expert branch.
```

E5/E6 must not alter the VLM-wide backbone or PICF core unless a probe proves
that gradients there are both present and conflicting.  The scaling-safe scope
is the action expert / action adapter / suffix bridge.

## G14: Action Expert Capacity Gate

Why this gate exists:

```text
G12 and G13 already rejected the scalable estimator-only and local-boundary
explanations:

  - K4 logical batch is real and necessary, but action did not descend.
  - temperature / explicit ratio / dynamic mixing did not fix action.
  - scoped PCGrad / CAGrad improved structure but not action.
  - action_head_only, action_adapter_only, and head+adapter all improved
    structural terms while action still rose.

Therefore the next causal class is not "another sampler scalar".  It is whether
the restored PI0.5 action expert / semantic stack itself must participate in
optimization under PICF conditioning.
```

Fixed mathematical contract:

```text
resume checkpoint = step 11000 E14h action4 branch
logical batch = K4 task_uniform without replacement
per-bucket normalization = on
action weights = 4 / 4 / 4 for pos / rot / gripper
PICF action condition = on
action context = suffix_cross_attention, stopgrad=true, RMS-normalized
PICF core LR scale = 0.001
optimizer checkpoint = model-only
```

Cases:

```text
g14_backbone_k4_capacity:
  semantic_trainable_scope = backbone_only

  Trains the restored PI0/PaliGemma semantic stack and action expert plus the
  PICF action-context adapter, while keeping wrapper-local flow/time/action
  projection heads fixed.  This is the production cotrain boundary.

g14_all_k4_capacity:
  semantic_trainable_scope = all

  Trains restored semantic/action expert, wrapper-local flow/time/action heads,
  and action-context adapter.  This is an upper-capacity diagnostic, not the
  default production recipe.

g14_backbone_k2_capacity / g14_all_k2_capacity:
  resource fallback cases if K4 OOMs on 2x40GB.  These reduce logical batch
  coverage to K2 to answer only whether the action expert path is locally
  trainable.  They must not be promoted as proof that the final K4
  task-balanced estimator is solved.
```

Decision:

```text
If backbone_only descends:
  promote production cotrain boundary and stop repeating local head/adapter
  gates.  The prior failures were under-capacity local-boundary probes.

If backbone_only fails but all descends:
  wrapper-local flow/time/action heads are required under PICF conditioning.
  This is a real architecture-boundary issue and must be documented before
  production because it changes the historical PI0.5 calibration contract.

If both fail:
  action path capacity alone is not the root.  Move to action objective/head
  form or action-expert-only routing/MoE, not more sampler scalars.

If either case OOMs or exceeds the 1-2 hour diagnostic budget:
  record it as a resource rejection.  Do not silently reduce K or disable the
  estimator contract, because that would invalidate the VLA-Foundry/ABot-M0
  task-balanced logical-batch hypothesis.
```

Launcher:

```bash
scripts/experiments/picf_aqr_owm_202605_active/run_a7_g14_action_expert_capacity_20260603.sh
```

Launch record:

```text
2026-06-03 15:11 Asia/Shanghai:
  remote = px-cloud1.matpool.com:26445
  tmux = picf_a7_g14_backbone_20260603
  case = g14_backbone_k4_capacity
  log = /mnt/picf_run_logs/g14_action_expert_capacity_20260603/g14_backbone_k4_capacity.log
  target = 11060

Startup contract confirmed:
  semantic = paligemma(trainable=True scope=backbone_only)
  training_strategy = fsdp_full_shard
  accum_steps = 2
  logical_batch_task_count = 4
  bucket_sampling = task_uniform
  bucket_sample_without_replacement = true
  logical_batch_bucket_normalization = true
  action_context = suffix_cross_attention stopgrad=true

Optimizer groups:
  semantic_backbone lr = 2e-5
  picf_core lr = 2e-8
  policy_head lr = 2e-5

Outcome:
  K4 backbone_only OOMed before the first optimizer step:
    GPU0 tried to allocate 14 MiB with about 17 MiB free;
    GPU1 tried to allocate 42 MiB with about 39 MiB free.

  This is a resource rejection, not a scientific rejection.  The K4 production
  estimator plus full restored semantic/action-expert stack does not fit on the
  current 2x40GB A7 contract.  The next valid fallback is
  g14_backbone_k2_capacity, which keeps the same action-expert production
  boundary but reduces task coverage to K2 strictly as a capacity diagnostic.

2026-06-03 15:19 Asia/Shanghai:
  fallback case launched:
    tmux = picf_a7_g14_backbone_k2_20260603
    case = g14_backbone_k2_capacity
    log = /mnt/picf_run_logs/g14_action_expert_capacity_20260603/g14_backbone_k2_capacity.log
    target = 11060

  Scientific interpretation:
    If K2 backbone descends, the action expert path has local trainability and
    the missing production piece is a memory-efficient way to restore K4/K8
    task coverage with that path.  If K2 backbone also fails, action-expert
    capacity alone is not enough; move to action objective/head/routing.

  2026-06-03 15:31 Asia/Shanghai first logged row:
    step = 11010
    logical_batch_distinct_bucket_count = 2
    logical_batch_global_micro_count = 2
    selected buckets = drawer, other
    loss_action_default_equiv = 0.054177
    loss_action_active7 = 0.246189
    loss_total_minus_action = 0.008979
    loss_anchor_pv = 0.507922
    loss_anchor_object_pull = 0.039600
    loss_mapg_routing = 0.432157
    loss_slot_jepa = 0.723397
    grad_norm = 0.553910

  Interim interpretation:
    this first K2 row is not healthy compared with the G13 K4 rows around
    0.040-0.043 action.  It is not yet a final rejection because K2 covers
    only two buckets and this row selected drawer/other.  The decision gate
    remains step 11020/11030.  If the action value stays high or rises, K2
    backbone capacity is rejected as an action fix and the next class is
    action-objective/head or action-expert routing, not another sampler scalar.

  2026-06-03 15:43 Asia/Shanghai second logged row:
    step = 11020
    logical_batch_distinct_bucket_count = 2
    logical_batch_global_micro_count = 2
    selected buckets = drawer, switch_button_light
    loss_action_default_equiv = 0.043810
    loss_action_active7 = 0.198880
    loss_total_minus_action = 0.009518
    loss_anchor_pv = 0.470441
    loss_anchor_object_pull = 0.093093
    loss_mapg_routing = 0.418227
    loss_slot_jepa = 0.655428
    active same-role support overlap = 0.075000
    grad_norm = 1.494491

  Interim interpretation:
    step11020 recovers from the bad step11010 row and improves multiple
    structure terms, but action is still only in the same 0.042-0.044 band
    where G12/G13 were rejected.  This remains non-decisive.  Continue to
    step11030: accept only if action continues toward/under the step11000
    baseline; reject if it stays at or above the G13 failure band.

  2026-06-03 15:43 Asia/Shanghai third logged row and decision:
    step = 11030
    logical_batch_distinct_bucket_count = 2
    logical_batch_global_micro_count = 2
    selected buckets = block_lift, slider
    loss_action_default_equiv = 0.046595
    loss_action_active7 = 0.211779
    loss_total_minus_action = 0.008950
    loss_anchor_pv = 0.486312
    loss_anchor_object_pull = 0.041387
    loss_mapg_routing = 0.429758
    loss_slot_jepa = 0.692661
    active same-role support overlap = 0.150000
    grad_norm = 1.206370

  Verdict:
    rejected as an action fix.  The K2 backbone_only fallback improves some
    structure rows but does not produce a descending action trend:
      0.054177 -> 0.043810 -> 0.046595
    This is not a proof against K4 backbone_only, because K4 was resource
    rejected, but it is enough to stop spending time on the reduced K2
    backbone-only proxy.  The next diagnostic must be an upper-bound action
    capacity test, not another sampler scalar.

2026-06-03 15:43 Asia/Shanghai:
  next upper-bound diagnostic launched:
    tmux = picf_a7_g14_all_k2_20260603
    case = g14_all_k2_capacity
    log = /mnt/picf_run_logs/g14_action_expert_capacity_20260603/g14_all_k2_capacity.log
    target = 11080

  Scientific interpretation:
    This is deliberately not the final scalable deployment recipe.  It unlocks
    the PaliGemma semantic/action stack more broadly under K2 to test whether
    action can descend when local action-expert capacity is no longer the
    bottleneck.  If all_k2 still fails, the remaining root class is action
    objective / action-head architecture / expert routing.  If all_k2 descends,
    the production problem is how to recover that capacity under K4/K8 task
    coverage without exceeding 2x40GB memory.

  Startup confirmed:
    semantic = paligemma(trainable=True scope=all)
    training_strategy = fsdp_full_shard
    accum_steps = 1
    logical_batch_task_count = 2
    bucket_sampling = task_uniform
    bucket_sample_without_replacement = true
    logical_batch_bucket_normalization = true
    action_context = suffix_cross_attention stopgrad=true
    optimizer_checkpoint_mode = model_only
    trainable_numel = 3,812,099,576
    approximate wall time = 36-37 sec/step

  This is a valid K2 upper-bound diagnostic, not a final production recipe:
    it trades task coverage for action-stack trainability and uses FSDP to fit
    all-scope semantic/action capacity on 2x40GB.  The decision gate remains
    action trend at step11010/11020/11030.

  2026-06-03 15:54 Asia/Shanghai first logged row:
    step = 11010
    logical_batch_distinct_bucket_count = 2
    logical_batch_global_micro_count = 2
    selected buckets = drawer, other
    loss_action_default_equiv = 0.054227
    loss_action_active7 = 0.246410
    loss_total_minus_action = 0.008961
    loss_anchor_pv = 0.507858
    loss_anchor_object_pull = 0.037844
    loss_mapg_routing = 0.432157
    loss_slot_jepa = 0.723491
    active same-role support overlap = 0.150000
    grad_norm = 0.599655

  Interim interpretation:
    non-decisive but not promising.  The first row is effectively identical
    to the rejected backbone_k2 first row because it selected the same
    drawer/other bucket pair.  Continue to 11020/11030.  Accept all_k2 only if
    action keeps descending past the G12/G13 failure band, not merely if
    structure terms improve.

  2026-06-03 16:05 Asia/Shanghai step11020/11030 and decision:
    step11020:
      selected buckets = drawer, switch_button_light
      loss_action_default_equiv = 0.044066
      loss_action_active7 = 0.199999
      loss_total_minus_action = 0.009528
      loss_anchor_pv = 0.469065
      loss_anchor_object_pull = 0.094162
      loss_mapg_routing = 0.418022
      loss_slot_jepa = 0.658079
      grad_norm = 1.646665
    step11030:
      selected buckets = block_lift, slider
      loss_action_default_equiv = 0.046744
      loss_action_active7 = 0.212351
      loss_total_minus_action = 0.009139
      loss_anchor_pv = 0.485921
      loss_anchor_object_pull = 0.048706
      loss_mapg_routing = 0.432340
      loss_slot_jepa = 0.680271
      grad_norm = 1.298305

  Verdict:
    rejected as an action fix.  The all-scope K2 upper-bound diagnostic closely
    reproduces the backbone_k2 failure trajectory:
      backbone_k2: 0.054177 -> 0.043810 -> 0.046595
      all_k2:      0.054227 -> 0.044066 -> 0.046744
    Therefore merely unlocking more of the PaliGemma/action stack is not the
    missing piece.  The next scientifically valid class is action objective /
    action-head architecture / action-expert routing.  Repeating sampler
    weights, K2/K4 scalars, PCGrad/CAGrad, or local adapter-only tests would
    duplicate rejected evidence.
```

### All Requested Method Status

| Method family | Status | Evidence / decision |
| --- | --- | --- |
| task-balanced logical batch | deployed, keep | sampler audit proved K4 without-replacement coverage; necessary baseline |
| gradient accumulation as logical batch | deployed, keep | K4 uses 2 ranks x accum=2; K8 resource-rejected on 2x40GB |
| per-bucket objective normalization | deployed, keep | part of all G12/G13/G14 gates |
| task-uniform sampling | deployed, keep | best fixed scalable default so far, but insufficient alone |
| temperature sampling | rejected as action fix | TEMP worsened action under controlled K4 |
| explicit bucket ratio | rejected as action fix | structure improved, action did not |
| bounded dynamic mixing / PiKE-style | rejected as action fix | transient structure improvement, action later worsened |
| PCGrad semantic/action group | rejected as action fix | structure improved, action worsened |
| CAGrad semantic/action group | rejected as action fix | structure improved, action worsened |
| action context stopgrad / gated bridge | deployed, keep | prevents direct continuous-action pressure into PICF core; insufficient alone |
| action_head_only | rejected | local projection heads alone cannot descend action |
| action_adapter_only | rejected | PICF-to-action adapter alone cannot descend action |
| action_head_and_adapter | rejected as local-boundary fix | reproduces same action increase |
| backbone_only action-expert capacity | rejected as reduced K2 action fix | K4 resource-rejected on 2x40GB; K2 fallback action 0.054177 -> 0.043810 -> 0.046595, no descent |
| all-scope action-expert upper bound | rejected as reduced K2 action fix | diagnostic only; all_k2 action 0.054227 -> 0.044066 -> 0.046744, no descent |
| action-expert-only conditional routing | deployed for G16 | G14 capacity and G15 robust-objective gates failed; G16 implements action-suffix-only routing, not whole-VLM MoE |
| whole-VLM MoE | rejected by design | harms scaling/semantics; not needed for current evidence |
| embodiment adapters | deferred | CALVIN is single embodiment; add when multi-embodiment data enters |
| System2/subtask head | deferred | needs reliable subtask labels/pseudo-labels; not a fast action-loss repair |
| robust action objective/head scalar redesign | rejected as G15 action fix | Huber/L1 worsen or fail to improve canonical action MSE; keep MSE objective for comparability |

### Deployment Part

| Component | Deploy now? | Required implementation |
| --- | --- | --- |
| K4 logical batch | yes | keep current DDP/no-checkpoint K4 contract |
| per-bucket normalization | yes | keep `q_b / n_b` scaling |
| explicit ratio / temperature knobs | config only | tested and rejected as default action fix |
| per-bucket metrics | yes | mandatory for all long runs |
| dynamic mixing | implemented; not default | bounded `q_b(t)` tested; structure transiently improved, action failed |
| CAGrad | implemented; not default | scoped semantic/action run failed action |
| action-expert capacity gate | rejected | G14 `backbone_only` K4 resource-rejected; K2 backbone/all both failed action |
| robust action-flow objective | deploy now | G15-A: train with Huber/L1-style objective while reporting canonical MSE for 4-22 comparison |
| action-expert MoE | after G15-A/B only | action-expert-only sparse experts; no VLM-wide MoE |
| embodiment adapters | future | add when multi-embodiment data exists; not CALVIN-only |
| System2 planning | future | add only with reliable subtask labels/pseudo-labels |

## G15: Action Objective / Head / Routing Gate

G14 closed the sampler/capacity branch:

```text
sampler/mixture/normalization: necessary but insufficient
PCGrad/CAGrad: structure improved, action did not
local action head/adapter scopes: rejected
backbone_only/all semantic capacity: rejected under reduced K2 proxy
```

Therefore G15 must not repeat scalar sampler tuning.  The next non-duplicate
root class is the action objective and action-expert architecture.

### G15-A: Robust Continuous-Flow Objective

Code contract:

```text
wrapper returns:
  total/action_pos/action_rot/action_gripper = canonical MSE report
  training_total/training_action_*           = actual objective

trainer uses:
  loss_action             <- training_total
  loss_action_default_equiv <- canonical MSE total
```

This preserves historical comparability with 4-22 while allowing robust
objectives:

```text
--semantic-action-flow-loss mse        # old default, parity
--semantic-action-flow-loss huber      # G15-A first test
--semantic-action-flow-loss smooth_l1  # alternate robust test
--semantic-action-flow-loss l1         # stronger OpenVLA-OFT-style diagnostic
```

Mathematical reason:

```text
PI0.5 flow predicts u_t = noise - target.
MSE minimizes E[||u_t - v_t||^2] and can overweight rare high-error action
windows/buckets.
Huber keeps quadratic curvature near zero but changes large residuals to
linear growth, so per-bucket gradient magnitude is less dominated by outliers.
```

Acceptance:

```text
Run from the same step11000 checkpoint and same K4 task-covered logical-batch
contract.
Accept only if canonical MSE loss_action_default_equiv descends below the
G12/G13/G14 failure band, not merely if the training Huber loss is smaller.
Reject if MSE action stays around 0.042-0.047 by 11030/11050.
```

Implementation status:

```text
deployed in code:
  src/openpi/picf/paligemma/config.py
    action_flow_loss
    action_flow_huber_delta
    action_flow_time_alpha / beta

  src/openpi/picf/paligemma/wrapper.py
    training_total/training_action_* use selected robust objective
    total/action_* remain canonical MSE report

  scripts/picf_core_train.py
    loss_action uses training_total
    loss_action_default_equiv uses canonical MSE
    pi_action_flow_objective_mode_id and pi_action_flow_time_mean are logged

  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
      forwards --semantic-action-flow-* through the launcher chain

    run_a7_g15_action_objective_matrix_20260603.sh
      fixed K4 task-uniform logical batch
      single-variable objective cases:
        g15_huber_k4_objective
        g15_smoothl1_k4_objective
        g15_l1_k4_objective
        g15_mse_k4_parity
```

Local validation:

```text
python -m pytest src/openpi/picf/paligemma/wrapper_test.py \
  scripts/picf_core_train_test.py -q \
  -k 'action_flow_objective or action_only_loss_preserves_default_equiv or semantic_action_flow_loss'

result:
  4 passed, 189 deselected

python -m py_compile \
  src/openpi/picf/paligemma/wrapper.py \
  src/openpi/picf/paligemma/config.py \
  src/openpi/picf/core/training.py \
  src/openpi/picf/policy.py \
  scripts/picf_core_train.py \
  scripts/picf_core_train_test.py

result:
  pass
```

Run command on A7:

```bash
cd /root/openpi_e21b2_1b07eab
ONLY_CASE=g15_huber_k4_objective \
CASE_TARGET_STEP=11120 \
bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_g15_action_objective_matrix_20260603.sh
```

Acceptance metric:

```text
Primary:
  loss_action_default_equiv, not loss_action.

Reason:
  loss_action is the selected robust training objective in G15.
  loss_action_default_equiv remains canonical MSE and is comparable to 4-22,
  G12, G13, and G14.
```

G15-Huber result, A7 2026-06-03:

```text
case:
  g15_huber_k4_objective

contract:
  resume = step11000
  logical batch = K4 task_uniform without replacement
  bucket normalization = true
  action context = suffix_cross_attention, stopgrad=true
  train objective = Huber(delta=1.0)
  acceptance metric = canonical MSE loss_action_default_equiv

observed:
  step 11010:
    loss_action = 0.037624955
    loss_action_default_equiv = 0.040642798
    loss_total_minus_action = 0.010910919
    loss_anchor_object_pull = 0.227323815
    loss_anchor_pv = 0.489586473
    loss_mapg_routing = 0.431502640
    loss_slot_jepa = 0.690742254

  step 11020:
    loss_action = 0.038946837
    loss_action_default_equiv = 0.042538650
    loss_total_minus_action = 0.009553451
    loss_anchor_object_pull = 0.098220713
    loss_anchor_pv = 0.474841177
    loss_mapg_routing = 0.417150617
    loss_slot_jepa = 0.695986152

  step 11030:
    loss_action = 0.039055854
    loss_action_default_equiv = 0.042828701
    loss_total_minus_action = 0.009228721
    loss_anchor_object_pull = 0.064367712
    loss_anchor_pv = 0.472665161
    loss_mapg_routing = 0.415389329
    loss_slot_jepa = 0.690147519

decision:
  reject Huber as the action repair.

reason:
  structure-side terms improved, but canonical MSE moved from 0.04064 to
  0.04283 and stayed in the G12/G13/G14 failure band.  Huber damped some
  non-action pressure but did not fix the action readout plateau.

next:
  run g15_l1_k4_objective under the identical contract.
```

G15-L1 result, A7 2026-06-03:

```text
case:
  g15_l1_k4_objective

contract:
  identical to G15-Huber except train objective = L1.

observed:
  step 11010:
    loss_action = 0.142052531
    loss_action_default_equiv = 0.040671028
    loss_total_minus_action = 0.010910917
    loss_anchor_object_pull = 0.227323666
    loss_anchor_pv = 0.489586502
    loss_mapg_routing = 0.431502640
    loss_slot_jepa = 0.690742135

  step 11020:
    loss_action = 0.141439885
    loss_action_default_equiv = 0.042575717
    loss_total_minus_action = 0.009737935
    loss_anchor_object_pull = 0.115420230
    loss_anchor_pv = 0.475053549
    loss_mapg_routing = 0.419787288
    loss_slot_jepa = 0.694663942

  step 11030:
    loss_action = 0.140226379
    loss_action_default_equiv = 0.042776577
    loss_total_minus_action = 0.009228718
    loss_anchor_object_pull = 0.064367421
    loss_anchor_pv = 0.472663760
    loss_mapg_routing = 0.415389299
    loss_slot_jepa = 0.690145493

decision:
  reject L1 as the action repair.

reason:
  L1 objective decreases in its own units, but canonical MSE worsens in the
  same pattern as Huber.  This is exactly why G15 reports canonical MSE
  separately: robust objective can look better while the historical action
  target remains unsolved.

smooth_l1 note:
  with beta/delta = 1.0, smooth_l1 is equivalent to the Huber shape already
  tested here.  Running it would be a duplicate single-factor test, not a new
  root-cause test.

G15 action-objective conclusion:
  The action plateau is not primarily caused by MSE outlier sensitivity.
  Structure terms can improve while action MSE worsens, so the next root class
  is action-head conditioning/capacity/routing, not another scalar objective.
```

### G15-B: Action Time-Noise Distribution

If G15-A Huber improves stability but not canonical MSE, test flow-time
sampling.  Current default is PI0.5 parity:

```text
t ~ Beta(1.5, 1.0), clipped to [0.001, 1.0]
```

The diagnostic knob is:

```text
--semantic-action-flow-time-alpha
--semantic-action-flow-time-beta
```

Do not change this before G15-A has a readout; otherwise objective and time
distribution are confounded.

### G15-C: Action-Expert-Only Routing / MoE

Only trigger if robust objective and time distribution fail.  The valid
architecture is:

```text
shared PaliGemma/VLM/PICF path remains dense
experts live only in the continuous action expert or action suffix adapter
router is conditioned on task bucket/prompt/embodiment metadata
```

Do not deploy whole-VLM MoE for CALVIN.  It is not supported by current
single-embodiment evidence and would weaken the intended scaling path.

G15-C trigger status, 2026-06-03:
  Triggered.  G15-Huber and G15-L1 both improved some structure terms while
  worsening or failing to improve canonical action MSE.  Therefore the next
  non-duplicate root class is action-expert conditioning/routing, not another
  robust scalar objective.

## G16: Action-Expert-Only Conditional Router

Mathematical position:

```text
semantic/PICF condition c = stopgrad(W_s * summary + mean(W_c * context))
router weights pi = softmax(W_r * LN(c) / tau)
expert residual r_h = sum_e pi_e * U_e * silu(D_e * LN(suffix_h))
suffix_h' = suffix_h + sigmoid(g) * cap_rms(r_h, suffix_h)
```

This is an action-expert-only routing adapter.  It does not:

```text
1. change token count;
2. change PI0.5 prefix/suffix masks;
3. MoE the full VLM/PaliGemma backbone;
4. make noisy sidecar masks a strong action label;
5. route by CALVIN-only hand-coded task ids.
```

Why this is the next coherent test:

```text
G12/F8 proved K4 task-covered logical batching is necessary but insufficient.
G13 proved action_head_only and action_adapter_only are insufficient.
G14 proved simply unfreezing more semantic/action stack capacity is insufficient.
G15 proved robust scalar action objectives are insufficient.

The remaining paper-supported class is action-specialized expert routing:
keep the semantic brain dense, but let the motor/action suffix path choose a
small residual expert conditioned on semantic/PICF context.
```

Implementation contract:

```text
config:
  action_expert_router_enabled
  action_expert_router_experts
  action_expert_router_rank
  action_expert_router_gate_init
  action_expert_router_temperature
  action_expert_router_rms_cap

runtime:
  _apply_action_expert_router() runs after PICF suffix cross-attention adapter
  and before PaliGemmaWithExpertModel.forward().

summary contract:
  _summary_from_outputs() returns cat([text_summary, image_summary]), so the
  production summary width is 2 * paligemma_width.  The router must support
  both single-summary and text+image pair-summary conditions.  It must not
  silently truncate a 4096-dim semantic summary into a 2048-dim projection.

initialization:
  expert up projections are zero-initialized;
  router logits are zero-initialized;
  gate starts at sigmoid(-2.5) ~= 0.076.

This makes the router an exact no-op at initialization, so any improvement or
damage is attributable to learned action-expert routing, not a changed restored
PI0.5 function.
```

Metrics:

```text
pi_action_expert_router_enabled
pi_action_expert_router_gate
pi_action_expert_router_entropy_mean
pi_action_expert_router_top_weight_mean
pi_action_expert_router_residual_rms_mean
```

Acceptance:

```text
primary:
  loss_action_default_equiv must descend relative to G13/G15 over the same
  11000->11120 gate.

secondary:
  router gate should remain bounded, not saturate immediately;
  router entropy/top_weight should show neither uniform-dead nor one-expert
  collapse from the first few steps;
  structure terms must not regress outside the G12 healthy band.
```

Launcher:

```bash
ONLY_CASE=g16_router4_rank64_k4 \
bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_g16_action_expert_router_matrix_20260603.sh
```

G16 launch gate, 2026-06-03:

```text
launch-0:
  failed before the first optimizer step.
  root cause:
    features.summary had width 4096 because the semantic summary is
    cat([text_summary, image_summary]), while the first router implementation
    projected only paligemma_width=2048.
  fix:
    added action_expert_router_summary_pair_proj for 2 * paligemma_width;
    retained action_expert_router_summary_proj for single-width summaries;
    added a same-width multi-summary average fallback for future concatenated
    summaries;
    added tests that exercise the pair-summary path and trainable scope.
  validation:
    local py_compile passed;
    local targeted pytest passed: 7 passed;
    remote targeted pytest passed: 7 passed.

launch-1:
  restarted on A7 under tmux `picf_a7_g16_router_20260603`.
  log:
    /mnt/picf_run_logs/g16_action_expert_router_20260603/g16_router4_rank64_k4.log
  status:
    interrupted by an operator-side diagnostic signal attempt before a valid
    loss readout.  This is not a training-signal rejection and should not be
    used in comparisons.

launch-2:
  restarted cleanly on A7 under tmux `picf_a7_g16_router_20260603`.
  log:
    /mnt/picf_run_logs/g16_action_expert_router_20260603/g16_router4_rank64_k4_launch2.log
  launch contract:
    LOG_INTERVAL=1 so the first optimizer step must emit the action and router
    metrics immediately.
  first required check:
    confirm first loss rows include pi_action_expert_router_* metrics and no
    shape error.
  operator tail:
    ssh -p 26445 root@px-cloud1.matpool.com
    tail -f /mnt/picf_run_logs/g16_action_expert_router_20260603/g16_router4_rank64_k4_launch2.log

  status:
    invalid as a training readout.  It reached checkpoint restore, then no
    optimizer-step loss was emitted for more than eight minutes.  Non-invasive
    process inspection showed rank0 sleeping and rank1 busy on one GPU.  This
    is treated as a launcher/runtime gate failure, not a model-quality result.

launcher audit:
  G16 originally reused the historical F8 -> E23 -> E9 -> fixed-window ->
  action-prefix -> slot-comprehensive launcher chain.  Although environment
  overrides usually propagate through that chain, the chain is too deep for a
  final decision gate because hidden historical defaults can survive if a
  variable is omitted.  Future G16/G17 runs should either:
    1. call the final slot-comprehensive launcher with all required variables
       explicitly set; or
    2. use a direct flat launcher that invokes `scripts/picf_core_train.py`.

sanity-k1:
  purpose:
    isolate whether the action-expert router path itself can run and emit
    metrics before reattempting K4 task-balanced DDP.
  contract:
    single GPU, NPROC_PER_NODE=1, ACCUM_STEPS=1,
    LOGICAL_BATCH_TASK_COUNT=1, LOG_INTERVAL=1, target step 11002.
  log:
    /mnt/picf_run_logs/g16_action_expert_router_20260603/g16_sanity_k1_launch.log
  first attempt:
    failed immediately because direct startup omitted REPO_ROOT and the final
    slot-comprehensive launcher fell back to historical
    `/root/openpi_slot_quality_ea2c5f2`.
  fix:
    explicitly pass REPO_ROOT=/root/openpi_e21b2_1b07eab,
    PYTHON_BIN=/root/openpi/.venv/bin/python, and
    CALVIN_ROOT=/mnt/calvin_data/task_ABC_D.

checkpoint compatibility gate:
  finding:
    the first valid K1 run failed at checkpoint restore because old step11000
    checkpoints do not contain the newly added
    `semantic_encoder.encoder.action_expert_router_*` parameters.
  root cause:
    `_load_state_dict_picf_compat()` correctly treats unexpected missing keys
    as fatal, but the new action-expert router was not yet listed as an
    explicitly allowed extension.
  fix:
    added narrow compatibility patterns for action-expert router parameters:
      semantic_encoder.encoder.action_expert_router_*
      policy.semantic_encoder.encoder.action_expert_router_*
      encoder.action_expert_router_*
      action_expert_router_*
    No unrelated missing keys are accepted.
  validation:
    local py_compile passed;
    local targeted pytest passed: 8 passed;
    remote targeted pytest passed: 8 passed.

sanity-k1 after compatibility fix:
  status:
    passed.  The router path restored from the old step11000 checkpoint,
    initialized only the new router parameters, emitted optimizer-step metrics,
    and exited at the requested target step 11002.
  log:
    /mnt/picf_run_logs/g16_action_expert_router_20260603/g16_sanity_k1_launch.log
  step11001:
    loss_total=0.227169
    loss_action_default_equiv=0.028377
    loss_anchor_pv=0.577899
    loss_anchor_object_pull=0.012036
    loss_mapg_routing=0.403165
    loss_slot_jepa=0.676733
    pi_action_expert_router_enabled=1
    pi_action_expert_router_gate=0.075858
    pi_action_expert_router_entropy_mean=1.386294
    pi_action_expert_router_top_weight_mean=0.250000
    pi_action_expert_router_residual_rms_mean=0.000000
  step11002:
    loss_total=0.288347
    loss_action_default_equiv=0.057425
    loss_anchor_pv=0.488269
    loss_anchor_object_pull=0.021407
    loss_mapg_routing=0.405573
    loss_slot_jepa=0.631208
    pi_action_expert_router_enabled=1
    pi_action_expert_router_gate=0.075858
    pi_action_expert_router_entropy_mean=1.386294
    pi_action_expert_router_top_weight_mean=0.250000
    pi_action_expert_router_residual_rms_mean=0.000177
  interpretation:
    this is a runtime and checkpoint-compatibility pass, not a quality pass.
    K1 intentionally covers only one task bucket per optimizer step, so its
    action values cannot validate the task-balanced G16 hypothesis.  It does
    prove the previous K4 blocker was not the router math itself but the
    checkpoint migration contract.  The next valid quality gate is a clean K4
    dual-GPU restart with the same compatibility fix.

launch-3:
  status:
    clean K4 dual-GPU restart after the checkpoint-compatibility fix.
  log:
    /mnt/picf_run_logs/g16_action_expert_router_20260603/g16_router4_rank64_k4_launch3.log
  runtime gate:
    passed through startup and emitted optimizer-step rows with
    `pi_action_expert_router_*` metrics.
  early readout, step11001-step11023:
    logical_batch_distinct_bucket_count=4 on every inspected row;
    loss_action_default_equiv ranged 0.014535-0.068273;
    loss_total_minus_action stayed about 0.0085-0.0159;
    loss_anchor_pv stayed about 0.4309-0.5259;
    loss_mapg_routing stayed about 0.4016-0.5056;
    loss_slot_jepa stayed about 0.6150-0.9118;
    router gate stayed about 0.07586;
    router entropy stayed near ln(4), about 1.384-1.386;
    router top_weight moved gently from 0.25 to about 0.275;
    router residual RMS stayed small, about 0.0000-0.0016.
  interpretation:
    this passes the implementation and bounded-router sanity criteria.  The
    router is active but initially conservative, which is intended by the
    gated no-op initialization.  It is not yet a final convergence result; the
    quality decision requires the completed K4 gate trend through the target
    step 11120.
  mid-gate readout, step11001-step11053:
    runtime remains healthy; tmux is alive and both A100 GPUs are active at
    about 28GB each.  All inspected rows keep
    logical_batch_distinct_bucket_count=4.  First10 action mean was 0.040652
    and last10 action mean was 0.047016, so the action trend is not yet an
    improvement signal.  loss_total_minus_action remains tightly bounded
    around 0.01; anchor_pv stays around 0.49; mapg_routing stays around 0.43;
    slot_jepa stays around 0.69.  Router gate remains near 0.07586, entropy
    remains high, and residual RMS remains small.  Current conclusion:
    implementation/routing stability is accepted, quality decision is still
    pending the full 11120-step gate.

  final readout, step11001-step11120:
    status=0; target step reached; checkpoint saved:
      /mnt/checkpoints/picf_core/picf_core/picf_g16_router4_rank64_k4_20260603_184234/11120
    runtime:
      about 49m48s for 120 optimizer steps, about 24.9s/step at the end.
      tmux exited normally and GPUs were released.
    action trend:
      first20 mean=0.041605
      second20 mean=0.044899
      third20 mean=0.048023
      fourth20 mean=0.044760
      last20 mean=0.042844
      all-step mean=0.045286
      min=0.013534, max=0.097796
    non-action/structure:
      loss_total_minus_action all-step mean=0.010343, last20 mean=0.010052.
      loss_anchor_pv all-step mean=0.492912, last20 mean=0.504652.
      loss_mapg_routing all-step mean=0.429362, last20 mean=0.448071.
      loss_slot_jepa all-step mean=0.691848, last20 mean=0.699770.
    router behavior:
      gate all-step mean=0.075844, last20 mean=0.075831.
      entropy all-step mean=1.379030, last20 mean=1.363006.
      top_weight all-step mean=0.291360, last20 mean=0.343388.
      residual_rms all-step mean=0.002278, last20 mean=0.003274.
    interpretation:
      implementation accepted, quality rejected as an action-convergence fix.
      The router is bounded and begins to specialize, but action does not show
      a reliable downward trend versus the first segment.  Structure losses
      remain bounded, so the rejection is not caused by structural collapse;
      the added action-expert-only routing is simply insufficient under the
      current K4 logical-batch recipe.
```

## Concrete Dynamic Mixing Candidate

The deployed bounded PiKE-style mixing is:

```text
base_q_b = task_uniform or temperature q_b
loss_ema_b(t) = beta * loss_ema_b(t-1) + (1-beta) * action_loss_b(t)
progress_b(t) = loss_ema_b(t-delta) - loss_ema_b(t)
lag_b = zscore(loss_ema_b) - gamma * zscore(progress_b)
raw_q_b(t) = base_q_b * exp(eta * clamp(lag_b, -c, c))
q_b(t) = normalize(clamp(raw_q_b(t), q_min, q_max))
```

Initial safe values:

```text
beta = 0.95
gamma = 0.5
eta = 0.25
c = 2.0
q_min = 0.05 / num_buckets
q_max = 0.35
warmup_updates = 50
```

Reasoning:

```text
This is not per-bucket overfitting.  It remains a dataset/task mixture policy,
keeps every bucket nonzero, and changes only sampling weights, not labels or
model semantics.  It is compatible with large heterogeneous data because the
same formula can operate over dataset/task/embodiment buckets.
```

Implementation contract:

```text
--logical-batch-dynamic-mixing
  enables sampler-side dynamic q_b(t).

q_b(t) is computed once at optimizer-step start from prior per-bucket
loss_action_default_equiv EMA/progress.

The same q_b(t) is used by:
  1. _bucket_sequence_for_logical_step(...) sampling;
  2. _logical_batch_loss_scales(...) estimator scaling.

This prevents the common false-positive bug where the sampler changes but the
loss estimator still assumes the old distribution.
```

Launcher hygiene fix:

```text
The final training launcher now executes `cd "${REPO_ROOT}"` and exports
`PYTHONPATH="${REPO_ROOT}/src:..."` before invoking Python.

Reason:
  A7 had an older `/root/openpi/src` on PYTHONPATH.  Without an absolute
  repo-local PYTHONPATH, tests could import stale core files even after the
  G12 scripts were synced.  This is a real follow-through bug, not a model
  change.
```

## Concrete Action-Boundary Candidate

If E2 shows conflict remains, the next scaling-safe structure is:

```text
semantic/PICF context -> stop-gradient bounded context tokens
context tokens -> action suffix cross-attention adapter
action suffix adapter -> trainable action expert/head only
optional expert routing -> action-expert FFN/adapter only
```

Do not:

```text
1. make PICF core chase action loss directly;
2. MoE the full VLM backbone;
3. hard-label sidecar masks as ground truth;
4. remove dense context tokens needed for V-JEPA/world-model continuity.
```

This matches the Knowledge-Insulation and action-expert literature while
preserving PICF's belief-router role.

## Full Accounting Of Requested Methods

This table is the strict checklist for the request.  "Not in this gate" means
the method is deliberately excluded from the fast CALVIN gate because it either
requires data the current run does not have or would weaken large-scale
scaling if deployed without evidence.

| Requested item | Current status | Reasoning |
| --- | --- | --- |
| Probabilistic dataset/task mixing | implemented | `task_uniform`, `temperature`, `trajectory`, and explicit `bucket_weight_spec` are present in `picf_core_train.py`. |
| Batch balancing ratios | implemented | `CALVIN_BUCKET_WEIGHT_SPEC` gives VLA-Foundry-style explicit ratios. G12-RATIO tests it. |
| Gradient accumulation as logical batch | implemented | `ACCUM_STEPS` and `LOGICAL_BATCH_TASK_COUNT` define the optimizer-step mixture. |
| Per-task objective normalization | implemented | `_logical_batch_loss_scales` realizes `q_b / n_b` per selected bucket. |
| Per-modality normalization | partially implemented | Action, PICF, sidecar/proposal, tactile/visual losses are separately scaled and logged. A full generic multi-dataset modality normalizer is future large-data infrastructure. |
| Task-uniform sampling | tested and kept | A1 proved K4 sees 4 distinct buckets. It is necessary but not sufficient by itself. |
| Temperature sampling | tested and rejected as action fix | G12-TEMP alpha=0.5 failed the fast action-slope gate. Keep the knob for future data, not as current default. |
| Explicit ratio mixing | tested and rejected as action fix | G12-RATIO improved structure terms but not action. Keep the config knob. |
| Dynamic PiKE-style mixing | implemented and rejected as current action fix | `--logical-batch-dynamic-mixing` computes bounded q_b(t) from per-bucket action-loss EMA/progress and logs dynamic weights/lags; G12-DYN improved structure transiently but action worsened. |
| Gradient cosine diagnosis | implemented and run | A2 measured action conflict in semantic/action-adapter group. |
| Scoped PCGrad/CAGrad | implemented and rejected as current action fix | G12-PCG and G12-CAGrad both confirm scoped surgery dataflow; both improved structure but failed action descent. |
| Whole-model PCGrad/CAGrad | rejected | Action gradients do not directly enter PICF core under current stop-gradient boundary; whole-model surgery is expensive and mathematically mismatched. |
| VLM/action expert boundary | implemented | PI0.5 action head is separate; PICF action context has stop-gradient and gated suffix cross-attention. |
| Modality projectors/adapters | implemented | Vision/point/tactile/proprio/PICF paths enter through typed projections/readers. |
| Embodiment-specific heads/adapters | not in CALVIN gate | CALVIN is single embodiment; add this only when multi-robot data is present. |
| Continuous action chunks | implemented | Action horizon path is audited by A3. |
| Huber/L1 action rewrite | not triggered | Current PI0.5 flow/chunk path is not proven corrupt by A3; do not rewrite action objective before sampler/conflict gates finish. |
| Action-expert MoE | evidence-triggered only | Add only if static mixing + bridge + scoped PCGrad fail and repeated gradient cosine still shows persistent task-family conflict. |
| System2/System1 split | future long-horizon branch | Requires reliable subtask labels/pseudo-subtasks; not a fast CALVIN action-loss repair. |

The immediate 8-hour goal is therefore not "try one thing".  It is:

```text
1. keep the already-correct logical-batch estimator;
2. compare scalable q_b choices: task_uniform vs temperature vs explicit ratio;
3. if q_b is insufficient, apply scoped gradient conflict handling;
4. only if conflict remains, justify action-expert MoE or a stronger action
   boundary rewrite.
```

## Mathematical Acceptance

The scalable target is:

```text
L = sum_b q_b L_b
G = sum_b q_b grad L_b
```

The estimator must approximate `G` without requiring giant physical batches:

```text
g_hat = sum_{b in B_step} q_b / n_b * grad L_b
```

Accepted tests must improve this estimator or reduce measured destructive
interference.  They must not overfit to fixed CALVIN windows.

Primary metrics:

```text
loss_action_default_equiv slope over 100-300 optimizer steps
loss_total_minus_action stability
bucket-wise loss_action trend
gradient_cosine(action, semantic/action adapter)
grad_norm and clip state
training seconds/step
```

Secondary metrics:

```text
anchor/object pull should not explode
same-role overlap is diagnostic only unless it corrupts action
sidecar/proposal losses must remain weak and bounded
slot_jepa is not a production acceptance metric if disabled/guarded
```

Hard rejection:

```text
action slope non-negative after 100-200 steps
action drops only on one bucket while others worsen
requires 6-GPU large physical batch as the only fix
requires hard sidecar/SAM labels
requires full-model gradient surgery
slows training >1.5x without a clear action gain
```

## 8-Hour Experiment Schedule

Each test is intended to run 100-300 optimizer steps.  Stop early at 100 if
action is clearly worsening and the failure matches prior rejected branches.

Concrete order:

```text
Hour 0:
  Sync launcher/docs.
  Run bash -n and py_compile.
  Confirm no FSDP/no checkpoint mistake for K4 gates.

Hour 0-1.5:
  G12-TEMP alpha=0.5, K4, accum=2, DDP/no-checkpoint.
  Purpose:
    test scalable frequency-aware q_b instead of pure task-uniform.
  Stop:
    if loss_action_default_equiv slope is non-negative by 50-100 optimizer
    steps and grad clipping repeats.

Hour 1.5-3:
  G12-RATIO, K4, accum=2.
  Purpose:
    test explicit VLA-Foundry-style batch balancing ratios for CALVIN buckets.
  Stop:
    if action only improves on one bucket or if long-tail buckets worsen.

Hour 3-4.5:
  G12-PCG, K4, accum=2, scoped semantic/action-adapter PCGrad.
  Purpose:
    directly test the measured A2 negative action cosine in the trainable bridge.
  Stop:
    if action slope does not beat no-PCGrad or if wall-time exceeds 1.5x.

Hour 4.5-6:
  Repeat gradient-cosine probe on the best candidate.
  Purpose:
    decide whether remaining failure is low-conflict imbalance or true
    destructive task conflict.

Hour 6-8:
  Branch:
    If static q_b works:
      launch a longer continuation with the accepted q_b and no extra modules.
    If bucket-specific lag remains with low conflict:
      implement/run bounded PiKE-style dynamic q_b.
    If strong negative conflict remains:
      prepare action-expert-only conflict handling; MoE only if PCGrad/bridge
      fail under repeated evidence.
```

All runs must write:

```text
runtime config
metrics.jsonl path
step/action slope
bucket-wise action loss
structure-loss stability
wall-time per optimizer step
accept/reject decision
```

## Execution Ledger

This section must be updated as each gate starts or is rejected.  Its purpose
is to prevent repeating old tests under a new name.

```text
2026-06-03 11:18 Asia/Shanghai:
  G12-K8 started on A7 2xA100-40GB.
  Config:
    ACCUM_STEPS=4
    LOGICAL_BATCH_TASK_COUNT=8
    task_uniform
    per-bucket normalization on
  Result:
    resource-rejected for the 1-2h fast gate.
    First optimizer step was about 130.8 s/step.
  Interpretation:
    K8 is not model-rejected.
    It is not acceptable as the fast dual-40GB deployment gate.
    Do not claim "larger logical batch solved it" until a cheaper K4/K6 or
    cached/precomputed path proves comparable action slope.

2026-06-03 11:25 Asia/Shanghai:
  G12-TEMP alpha=0.5 K4 launched on A7 2xA100-40GB.
  Confirmed runtime config:
    runtime_task_count=4
    bucket_normalization=True
    bucket_sampling_mode=temperature
    bucket_temperature_alpha=0.5
    sample_without_replacement=True
    action_context_stopgrad=True
    semantic_trainable_scope=action_head_and_adapter
  Status:
    running.
    Startup confirmed correct at step 11000.
    Early measured speed after resume: about 60-61 s/optimizer step.
  Interpretation:
    This is scientifically valid but slow for the full 8h matrix.
    Keep only until it produces enough early metrics to judge temperature
    sampling.  Do not use FSDP+checkpoint K4 as the final scalable recipe
    unless it shows a uniquely strong action improvement.

2026-06-03 correction:
  The G12 launcher default was fixed so K4 gates use the previously validated
  DDP/no-checkpoint path.  FSDP+checkpoint is now reserved for K8 resource
  probes or OOM fallback.  This matters because G10/F9c established that K4
  action-adapter tests are supposed to be fast DDP gates, while FSDP changes
  wall-time enough to invalidate the 1-2h comparison budget.

2026-06-03 11:40 Asia/Shanghai:
  Slow FSDP TEMP was stopped and G12-TEMP alpha=0.5 K4 was restarted through
  DDP/no-checkpoint.
  Confirmed runtime:
    training_strategy=ddp
    window_activation_checkpointing=False
    accum_steps=2
    runtime_task_count=4
    bucket_sampling_mode=temperature
    bucket_temperature_alpha=0.5
  First detailed metric at step 11010:
    loss_action_default_equiv = 0.036058988
    loss_total_minus_action   = 0.011101709
    loss_anchor_pv            = 0.482755780
    loss_anchor_object_pull   = 0.242818773
    loss_mapg_routing         = 0.438883394
    loss_slot_jepa            = 0.682871044
    logical_batch_distinct_bucket_count = 4
    speed ~= 21.2 s/optimizer step
  Interpretation:
    DDP TEMP is a valid fast gate.  Need 50-100 steps before accepting or
    rejecting temperature sampling.

2026-06-03 11:55 Asia/Shanghai:
  G12-TEMP DDP early metrics:
    step 11010:
      loss_action_default_equiv = 0.036048010
      loss_total_minus_action   = 0.010613246
      preclip_grad_norm         = 0.262030
    step 11020:
      loss_action_default_equiv = 0.046092674
      loss_total_minus_action   = 0.011245083
      preclip_grad_norm         = 16.414461
      grad_clip_applied         = true
  Interpretation:
    This is not enough for final rejection, but it is a concrete warning sign:
    alpha=0.5 frequency-aware mixing did not immediately reproduce the desired
    action descent and produced a clipped-gradient spike.  Continue only to the
    next 50-step trend.  If the trend stays non-negative, cut TEMP and run
    G12-RATIO immediately.

2026-06-03 local code gate:
  Commands:
    pytest -q scripts/picf_core_train_test.py -k "bucket or pcgrad or logical_batch or gradient_surgery"
    python -m py_compile scripts/picf_core_train.py scripts/picf_calvin_bucket_sampler_audit.py scripts/picf_bucket_gradient_cosine_probe.py scripts/picf_action_window_target_audit.py
  Result:
    8 selected tests passed.
    py_compile passed.
  Verified code contracts:
    _compute_bucket_sampling_weights covers task_uniform, trajectory,
    temperature, and explicit VLA-Foundry-style bucket ratios.
    _bucket_sequence_for_logical_step samples the optimizer-step global
    bucket sequence, not independent per-rank accidental buckets.
    _logical_batch_loss_scales realizes q_b / n_b and compensates DDP gradient
    averaging.
    PCGrad is scoped to selected trainable owner groups; whole-model surgery is
    not silently enabled.

2026-06-03 12:02 Asia/Shanghai:
  G12-TEMP DDP early trend:
    step 11010 action = 0.036048010
    step 11020 action = 0.046092674
    step 11030 action = 0.042926051
  Interpretation:
    Temperature alpha=0.5 has not shown the desired early action descent.
    It is still within the pre-decision window, but it must be cut if step
    11050 remains non-improving.

2026-06-03 12:01-12:05 Asia/Shanghai:
  Decision:
    Cut G12-TEMP before 11050 because step 11040 still failed the fast-gate
    trend:
      11010 action = 0.036048010
      11020 action = 0.046092674
      11030 action = 0.042926051
      11040 action = 0.043974437
    This is a non-negative action trend plus one clipped-gradient spike.  It
    is enough to reject alpha=0.5 temperature sampling for this fast gate.
  G12-RATIO launched:
    session = picf_a7_g12_ratio_20260603
    bucket_weight_spec =
      block_lift=1,block_other=1,block_push=1.25,drawer=1.25,
      other=0.75,slider=1,switch_button_light=1.25
    confirmed target weights =
      block_lift 0.1333
      block_other 0.1333
      block_push 0.1667
      drawer 0.1667
      other 0.1000
      slider 0.1333
      switch_button_light 0.1667
  Reason:
    Temperature q_b ∝ sqrt(N_b) still overweights large buckets relative to the
    hard action buckets that repeatedly show instability.  RATIO directly tests
    whether explicit action-critical coverage is the missing factor.

2026-06-03 12:09 Asia/Shanghai:
  G12-RATIO first metric:
    step 11010:
      loss_action_default_equiv = 0.045747153
      loss_total_minus_action   = 0.011745730
      loss_anchor_pv            = 0.496869326
      loss_anchor_object_pull   = 0.300100058
      loss_mapg_routing         = 0.441621661
      logical_batch_distinct_bucket_count = 4
      selected buckets =
        block_other, block_push, slider, switch_button_light
  Interpretation:
    First point is worse than G12-TEMP 11010 and not structurally cleaner.
    Continue only to step 11020 because progress-bar samples after 11010 showed
    transient low values.  If 11020 is not clearly better, reject explicit
    ratio mixing and run G12-PCG.

2026-06-03 12:13 Asia/Shanghai:
  G12-RATIO decision:
    step 11010 action = 0.045747153
    step 11020 action = 0.046368040
    structure improved but action did not:
      object_pull 0.300100058 -> 0.094537899
      routing     0.441621661 -> 0.418315291
  Decision:
    reject explicit ratio mixing as an action-descent fix.
  Interpretation:
    static q_b changes can help structure terms but do not remove the action
    plateau.  This supports the A2 diagnosis that the next root to test is
    semantic/action-adapter gradient conflict.
  G12-PCG launched:
    session = picf_a7_g12_pcgrad_20260603
    expected config =
      DDP
      K4 / accum=2
      task_uniform
      logical_batch_gradient_surgery=pcgrad
      logical_batch_gradient_surgery_groups=semantic

2026-06-03 12:18 Asia/Shanghai:
  G12-PCG first metric:
    step 11010:
      loss_action_default_equiv = 0.040655419
      loss_total_minus_action   = 0.010910450
      loss_anchor_pv            = 0.489586264
      loss_anchor_object_pull   = 0.227279156
      loss_mapg_routing         = 0.431502640
      logical_batch_gradient_surgery_enabled = true
      logical_batch_gradient_surgery_target_param_tensors = 14
  Interpretation:
    PCGrad dataflow is real and scoped to semantic/action-adapter tensors.  It
    is better than RATIO's first action point but still not a proven fix.  The
    decision point is step 11020: if action rises as in F9c, scoped PCGrad is
    rejected again under the G12 controlled order.

2026-06-03 12:39 Asia/Shanghai:
  G12-PCG decision:
    step 11010 action = 0.040655419
    step 11020 action = 0.042559538
    step 11040 action = 0.046892378
    structure terms still improved or stayed bounded:
      anchor_pv    0.489586 -> 0.475054 -> 0.491519
      object_pull  0.227279 -> 0.115387 -> 0.130488
      routing      0.431503 -> 0.419787 -> 0.423539
  Decision:
    reject scoped semantic/action-adapter PCGrad as the action-loss fix.
  Interpretation:
    This reproduces the earlier F9c conclusion under the all-requested-method
    order.  First-line task mixing and scoped gradient surgery are now ruled
    out as sufficient.  The next non-duplicative work is:
      1. repeat gradient-cosine after G12 candidates to quantify residual
         conflict;
      2. implement bounded dynamic mixing only if bucket-specific lag is real;
      3. otherwise move to action-expert-only boundary/capacity/expertization.
  Operational:
    The PCGrad tmux session was stopped after rejection to avoid wasting GPU
    time.
```

### G12-0: Script/Config Gate

Goal:
  ensure all experiments use the same resume checkpoint, same sidecar, same
  action horizon, same logging, and visible progress.

Pass:
  launcher `bash -n`, trainer argument print contains expected sampling,
  accumulation, normalization, action context, and trainable scopes.

### G12-K8: Larger Logical Batch Without Larger Physical Batch

Purpose:
  test whether the E21 behavior is mostly logical task coverage.

Config:

```text
WORLD_SIZE=2
ACCUM_STEPS=4
LOGICAL_BATCH_TASK_COUNT=8
calvin_bucket_sampling_mode=task_uniform
sample_without_replacement=true
logical_batch_bucket_normalization=true
```

Expected:
  lower gradient variance than K4.  If action descends here but not K4, root is
  coverage/variance, not architecture.

### G12-TEMP: Temperature Sampling

Purpose:
  test VLA Foundry / ABot-M0 style frequency-aware scalable mixing.

Config:

```text
ACCUM_STEPS=2
LOGICAL_BATCH_TASK_COUNT=4
calvin_bucket_sampling_mode=temperature
calvin_bucket_temperature_alpha=0.5
```

Expected:
  should not beat K8 if the issue is gradient conflict; may help if over-
  uniform sampling overexposes noisy rare buckets.

### G12-RATIO: Explicit Ratio Mixing

Purpose:
  test whether CALVIN's current bucket frequencies need manual q_b.

Config:

```text
ACCUM_STEPS=2
LOGICAL_BATCH_TASK_COUNT=4
calvin_bucket_sampling_mode=weighted
calvin_bucket_weight_spec=<equal or inverse-frequency ratios>
```

Expected:
  only accepted if it improves action without starving bucket diversity.

### G12-PCG: Scoped Gradient Surgery

Purpose:
  test the measured A2 conflict directly.

Config:

```text
logical_batch_gradient_surgery=pcgrad
logical_batch_gradient_surgery_groups=semantic
semantic_trainable_scope=action_head_and_adapter
training_strategy=ddp
window_activation_checkpointing=0
```

Expected:
  action slope improves if negative semantic/action bucket cosine is the root.
  If it fails again under this controlled setup, do not repeat PCGrad.

### G12-BRIDGE: Action Boundary / Adapter Bridge

Purpose:
  test Knowledge-Insulation-style boundary without discarding PICF belief
  learning.

Config:

```text
picf_action_context_stopgrad=true
action_context_integration=suffix_cross_attention or prefix_fusion
action_context_output_gate controlled
PICF structural losses remain trainable
continuous action gradient restricted away from PICF core
```

Expected:
  action improves while structural losses remain bounded.  If action still
  stalls, the action expert itself or task-mixing recipe is the likely root.

### G12-DYN: Dynamic Mixing

Run only if K8/TEMP/RATIO reveal bucket-specific slopes but not adversarial
negative cosine.

Purpose:
  PiKE-like adaptive q_b without weakening large-data scaling.

Config:

```text
start from static task_uniform or temperature
track recent per-bucket loss improvement
increase sampling/weight for lagging buckets within bounded range
do not use fixed-window or hand-picked tasks
```

Expected:
  improves lagging bucket action without destabilizing already-improving
  buckets.

### G12-MOE Decision

Do not implement MoE in this 8-hour gate unless all of these are true:

```text
K8/TEMP/RATIO fail
PCGrad or bridge fails
A2 repeated after best candidate still shows strong negative action gradient
cosine inside action-adapter group
```

If triggered, deploy action-expert-only MoE:

```text
shared VLM/PICF belief path stays dense
MoE only inside continuous action expert / adapter FFN
router conditioned on task bucket / prompt / embodiment token
```

This preserves scaling and avoids VLM-wide expert fragmentation.

## Final Deployment Candidate

The expected scalable final recipe, if G12 validates it, is:

```text
task-balanced logical batch
K4 or K8 depending on wall-time/action slope
per-bucket objective normalization
continuous action chunks
action context stop-gradient into PICF core
semantic/action adapter trainable under scoped gradient control if needed
sidecar/proposal/tracklet weak only
no SAM hard labels
no full-model gradient surgery
no fixed-window training recipe
```

This is not "just increasing batch".  It is a production estimator and
gradient-boundary design aligned with current VLA scaling practice.

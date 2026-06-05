# PICF-AQR-OWM Action Descent Checklist

Date: 2026-06-05

Entry point:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_G26_FULL_VLA_METHOD_AUDIT_AND_NEXT_GATE_20260604.md
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
docs/PICF_AQR_OWM_8H_FULL_VLA_REPAIR_MATRIX_20260603.md
```

Purpose:

```text
Prevent another action-loss experiment from repeating an already-rejected
sampler/LR/auxiliary branch.  Every new run must state which causal factor it
changes and which prior evidence it is not duplicating.
```

## 1. Core Mathematical Question

The target action objective is a mixture over CALVIN task buckets:

```text
L(theta) = sum_b q_b L_b(theta)
G(theta) = sum_b q_b grad L_b(theta)
```

A physical micro-batch that sees only one or two buckets per optimizer step has
high task-gradient variance:

```text
Var[g] = E_b ||grad L_b(theta) - G(theta)||^2
```

The maintained repair direction is therefore:

```text
1. make every optimizer update approximate G(theta) by task-balanced logical
   batches;
2. normalize per-bucket contributions so sampled-count imbalance does not
   dominate;
3. only if balanced updates still fail, inspect the action/semantic boundary
   and action path capacity.
```

This is consistent with the 2025-2026 VLA method family:

```text
VLA Foundry:
  probabilistic dataset/task mixing, batch balancing, gradient accumulation,
  and dataset-aware normalization.

ABot-M0:
  task/embodiment balance and action standardization for heterogeneous robot
  data.

PiKE:
  dynamic mixing only after static mixing is insufficient.

Knowledge Insulation / pi0.5:
  continuous action gradients need a controlled boundary to semantic/VLM
  representations.

OpenVLA-OFT:
  continuous action chunks and simple robust action losses are strong baselines,
  but objective changes must be judged by canonical action MSE.
```

Primary source refresh, 2026-06-05:

```text
VLA Foundry, arXiv:2604.19728:
  supports a unified LLM->VLM->VLA training stack and open framework for
  multi-task VLA training.  This supports keeping dataset/task mixing and
  training-stack consistency as first-class infrastructure.

ABot-M0, arXiv:2602.11236:
  emphasizes data cleaning, standardization, task/embodiment balance, and
  action-manifold-style continuous action stability for heterogeneous robot
  data.  This supports keeping CALVIN bucket balance and action normalization,
  not overfitting to one task family.

PiKE, arXiv:2502.06244:
  dynamic mixing is useful when task gradients have useful positive
  interactions and low conflict.  In this codebase it is therefore diagnostic
  or second-stage, not a default replacement for the already healthy K4 native
  action control.

Knowledge Insulation, arXiv:2505.23705:
  continuous action experts can harm VLM training dynamics unless the backbone
  and action expert are insulated.  This directly supports the current
  stop-grad PICF-context boundary and the fix that native action override must
  remain live.

OpenVLA-OFT, arXiv:2502.19645:
  optimized fine-tuning uses parallel decoding, action chunking, continuous
  action representation, and L1-style objectives.  The current PI0.5-style
  flow chunk path is already continuous/chunked, so the immediate root is not
  "switch to another objective"; compare by canonical `loss_action_default_equiv`.

GR00T N1, arXiv:2503.14734:
  separates vision-language interpretation and motor generation into a
  coupled dual-system VLA.  This supports the semantic/action boundary framing
  but does not justify a full architecture rewrite before the G44/G45 boundary
  gates finish.
```

## 2. Already Implemented And Tested

These should not be repeated as standalone fixes unless the code path changed.

| Method | Code path | History result | Current decision |
| --- | --- | --- | --- |
| task-uniform logical batch | `scripts/picf_core_train.py::_compute_bucket_sampling_weights` and `_bucket_sequence_for_logical_step` | K4 coverage verified; necessary but insufficient | keep default |
| gradient accumulation as logical batch | `WORLD_SIZE * accum_steps`, logged as `logical_batch_global_micro_count` | K4 feasible on 2x40GB; K8/K12 resource-limited | keep K4 where possible |
| per-bucket normalization | `_logical_batch_loss_scales` | deployed in G12/G26 | keep mandatory |
| temperature sampling | `calvin_bucket_sampling_mode=temperature` | tested; did not fix action | config only |
| trajectory/data-proportional sampling | `calvin_bucket_sampling_mode=trajectory` | G26 Contrast C worsened action by step600 | rejected as root fix |
| explicit ratio mixing | `calvin_bucket_weight_spec` | tested; no action fix | config only |
| dynamic PiKE-style mixing | `_dynamic_bucket_sampling_weights` | tested; structure transiently improved, action not fixed | not default |
| PCGrad | `_pcgrad_project_and_sum` | tested; action worsened or did not fix | not default |
| CAGrad | `_cagrad_project_and_sum` | tested; action worsened or did not fix | not default |
| Huber/L1 action-flow objective | `semantic_action_flow_loss` | tested; canonical MSE did not improve | not root fix |
| context-only action readout | `_compute_action_context_readout_aux` | proved PICF context is motor-readable but not native-flow useful alone | diagnostic |
| deployed flow residual | `_apply_action_context_flow_residual` | G25/G26 live; not enough to break platform | keep guarded |
| PICF-local FAST/action-token auxiliary | `_compute_action_context_token_aux` | G26 short gate passed, but long action still platformed | keep guarded |
| action expert router proxy | `_apply_action_expert_router` | tested; not sufficient | off by default |
| raw-overlap-only repairs | AQR/MAPG losses and overlap filters | repeatedly failed to explain action | do not reopen alone |
| SAM proposal branch | sidecar/SAM docs | noisy and archived | keep default off |

## 3. Live Control: PI0.5-Like K4 Ablation

Current run:

```text
tmux:
  picf_g26pi05like_k4_ablation_30k_20260605

log:
  /mnt/picf_run_logs/picf_g26pi05like_k4_ablation_30k_20260605.log

contract:
  picf_mode=ablated
  native PI0.5 semantic action path
  task_uniform K4 logical batch
  per-bucket normalization
  accum_steps=2
  lr=2e-4, warmup=600
```

Trainable-boundary follow-through:

```text
semantic_trainable_scope=backbone_only

This is not action-head-only.  In
src/openpi/picf/paligemma/wrapper.py::_apply_trainable_scope,
backbone_only/model_only trains `self.model.parameters()`, which includes the
PaliGemma/Gemma action expert stack, while freezing wrapper-local
`action_in_proj`, `action_out_proj`, and time MLP calibration heads.  The unit
test `test_pi0_trainable_scope_backbone_only_matches_historical_full_cotrain_boundary`
explicitly records this as the historical full-cotrain boundary.

Therefore this control is a valid current-shell PI0.5-like action-capacity
probe.  It is still not exact 4-22 parity because trainer shell, token budget,
sampling, FSDP, and optimizer schedule differ.
```

What it changes:

```text
It removes PICF recurrent/control/future semantics while preserving the current
trainer shell and K4 sampling.  This directly tests whether G26-B action
platform is caused by PICF/context integration or by the current semantic
action path / data mixture itself.
```

Step50:

```text
loss_action_default_equiv = 0.1291981190
logical_batch_distinct_bucket_count = 4
loss_total_minus_action = 0.0
lr = 1.6666666667e-05
```

Step50 reading:

```text
Not enough evidence.  K4 coverage and action-only loss are confirmed, but the
600-step warmup is still early and the action scalar is not yet better than the
G26-B fresh reference.
```

Step100:

```text
loss_action_default_equiv = 0.0770749152
loss_action_active7 = 0.3494936228
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 3.3333333333e-05
```

Step100 reading:

```text
This is the first positive control signal.  The PI0.5-like ablation is clearly
better than G26-B fresh step100 (0.1165) and already near G26-B fresh step200
(0.0766).  Therefore "PICF semantics/context integration is action-negative or
action-slowing in G26-B" is now supported.

Still unresolved: whether this branch keeps descending toward the old
0.02-0.03 band or merely reaches the same 0.06-0.08 platform earlier.  Continue
to step200 before stopping.
```

Step150:

```text
loss_action_default_equiv = 0.0646321550
loss_action_active7 = 0.2944722176
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 5.0e-05

bucket action_default_equiv:
  block_lift          0.0680387956
  block_other         0.0678645276
  block_push          0.0495955158
  drawer              0.0641414492
  other               0.0664097131
  slider              0.0504070998
  switch_button_light 0.0881502455
```

Step150 reading:

```text
The PI0.5-like ablation has already reached the G26-B fresh step300 region
(G26-B step300 action_default ~= 0.0654) by step150.  This strengthens the
causal interpretation that the enabled PICF/context path is action-negative or
action-slowing under the current integration.  The remaining question is
whether this clean action path keeps descending toward the old 0.02-0.03 band
or stalls in the 0.05-0.06 band.
```

Step200:

```text
loss_action_default_equiv = 0.0641728789
loss_action_active7 = 0.2913683653
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 6.6666666667e-05

action components:
  pos     0.2413374484
  rot     0.2888981402
  gripper 0.4488718212

bucket action_default_equiv:
  block_lift          0.0580355006
  block_other         0.0733913263
  block_push          0.0562251920
  drawer              0.0697186454
  other               0.0571673639
  slider              0.0510929807
  switch_button_light 0.0893955454
```

Step200 reading:

```text
The branch remains action-positive relative to G26-B step200 (0.0766) and still
slightly beats G26-B step300 (0.0654).  However, step150 -> step200 is nearly
flat: 0.06463 -> 0.06417.  The hard bucket is still switch/button/light, and
the gripper component rebounded.  This rules out immediate failure, but it also
rules out "PICF removal alone instantly restores old 0.02-0.03 convergence".
Continue to step300; use step300 to decide whether this is an early plateau or
a temporary warmup/bucket-composition pause.
```

Step250:

```text
loss_action_default_equiv = 0.0656914562
loss_action_active7 = 0.2976432741
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 8.3333333333e-05

action components:
  pos     0.2287622839
  rot     0.3194335699
  gripper 0.4389153123
```

Step300:

```text
loss_action_default_equiv = 0.0640842691
loss_action_active7 = 0.2884802520
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 1.0e-04

action components:
  pos     0.2238104343
  rot     0.3115257025
  gripper 0.4133533239
```

Step300 reading:

```text
This is not a clean failure, but it is not the requested action-collapse fix.
The current-shell PI0.5-like ablation improves sharply from step50 to step150,
then stays in the 0.064-0.066 band through step300.  It proves that enabled
PICF/context was action-negative relative to a clean ablated path, but it does
not prove that K4 task-balanced sampling plus current G26 action shell recovers
the old 4-22/PI0.5 0.02-0.03 action band.

Decision: stop this branch at step300 and run a stricter parity experiment:
old 4-22 args/action shell + current K4 task-uniform logical batch.  This is
the non-repeated missing test.  If old args + K4 descends faster, the root is
the G26 action shell / optimizer / trainable-boundary delta.  If it also
plateaus, then the old train-stream metric and current balanced objective are
not directly comparable, or current code has a deeper parity regression.
```

## 3.5 Parity Control E: Old 4-22 Args + Current K4 Sampler

Run:

```text
picf_g27_oldargs_k4_pi05ablated_300_20260605
```

Purpose:

```text
Hold the new task-uniform logical-batch sampler, but return the action shell to
the archived 4-22 ablated PI0.5 args as much as the current code allows.

This specifically checks the remaining uncontrolled deltas from the G26
PI0.5-like ablation:
  semantic_lr_scale: old 0.25 vs G26 current 1.0
  grad_clip_norm: old 1.0 vs G26 current 5.0-percentile shell
  semantic_trainable_scope: old action_head_and_adapter vs G26 backbone_only
  historical action/prompt/state defaults in the 4-22 args file
  explicit semantic_action_context_flow_residual_enabled=False

This is not a repeat of task-uniform, temperature sampling, ratio mixing,
dynamic PiKE-style mixing, PCGrad/CAGrad, Huber/L1, context-only readout, SAM,
or raw-overlap repair.  Those are already recorded as implemented/tested or
diagnostic-only in the method audit.  This is a trainer/action-shell parity
test.
```

Implementation note:

```text
The old 4-22 args file predates several current trainer fields.  To avoid
false failures from missing Namespace attributes, the remote run builds a
complete args file by taking the G26-B args as a current-code default template,
then overlaying the archived 4-22 args and finally applying the K4/sampler
control overrides.  This preserves old values where they exist while filling
new fields with current defaults.

Effective critical values:
  semantic_lr_scale = 0.25
  grad_clip_norm = 1.0
  grad_clip_mode = percentile
  semantic_trainable_scope = action_head_and_adapter
  lr = 2e-4
  warmup_steps = 600
  burnin_steps = 0
  effective_unroll_steps = 3
  accum_steps = 2
  picf_mode = ablated
  calvin_bucket_sampling_mode = task_uniform
  logical_batch_task_count = 4
  logical_batch_bucket_normalization = true
```

The archived 4-22 args contained a nonzero burn-in setting.  Current code
correctly rejects `burnin_steps > 0` with `picf_mode=ablated`, because the clean
PI0.5 ablation has no PICF recurrent carry to burn in.  For the parity control,
`burnin_steps=0` is therefore a contract fix, not a tuning choice.

Tail:

```bash
ssh -i /tmp/picf_g22_key -p 26120 root@px-cloud1.matpool.com 'tail -f /mnt/picf_run_logs/picf_g27_oldargs_k4_pi05ablated_300_20260605.log'
```

Startup verification:

```text
04:47:18 remote log entered PICF Training.

world_size = 2
training_strategy = fsdp_full_shard
picf_mode = ablated
trainable_numel = 13,751,362
total_numel = 3,744,957,843
accum_steps = 2
effective_global_batch = 4
num_steps = 300
lr = 2e-4
min_lr = 2e-5
warmup = 600
unroll_steps = 2
burnin_steps = 0
effective_window_steps = 2
calvin_balanced_bucket_sampler = true
calvin_bucket_sampling_mode = task_uniform
logical_batch_task_count = 4
logical_batch_bucket_normalization = true
calvin_buckets = block_lift, block_other, block_push, drawer, other, slider, switch_button_light
```

Step50:

```text
loss_action_default_equiv = 0.1529971361
loss_action_active7 = 0.5082107782
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 1.6666666667e-05

action components:
  pos     0.4047880769
  rot     0.5090084076
  gripper 0.8160861731
```

Step50 reading:

```text
This is worse than the current-shell G26 PI0.5-like ablation at step50
(0.129198).  The old action shell did not produce the expected immediate early
drop.  Because step50 is still inside low-LR warmup, continue only to step100
as the final parity gate.  If step100 remains above the current-shell step100
(0.077075), stop this branch: the plateau is not fixed by restoring old
semantic_lr_scale/trainable-scope/clip defaults under K4 balanced sampling.
```

Step100:

```text
loss_action_default_equiv = 0.1471613646
loss_action_active7 = 0.4896215200
loss_total_minus_action = 0.0
lr = 3.3333333333e-05

progress-bar recent single-window values near step100:
  step97  0.3729
  step98  0.1508
  step99  0.0731
  step100 0.1399
```

Step100 decision:

```text
Stop G27 immediately.  It is clearly worse than the current-shell G26
PI0.5-like ablation at the same step:

  G26 current-shell K4 step100: 0.077075
  G27 old-args K4 step100:     0.147161

Therefore the action plateau is not solved by restoring old semantic_lr_scale,
old grad clip, or old action_head_and_adapter trainable scope under the current
balanced K4 sampler.  The root is not simply "G26 action shell differs from
4-22".

Current causal status:
  1. Full G26-B PICF/context slows action relative to clean ablated action.
  2. Clean current-shell ablated action improves early but plateaus around
     0.064 by step300.
  3. Old-args action shell under K4 is worse than the clean current-shell
     ablation and is rejected at step100.

Next non-repeated direction:
  keep the current-shell ablated action path as the better control;
  do not restore old action_head_and_adapter as default;
  investigate whether balanced K4 objective is intrinsically harder than the
  old online train stream, and whether the action model needs a stronger native
  action-representation objective rather than more sampler/optimizer repair.
```

## 3.6 Parity Control F: Old 4-22 Online Stream Shape

Run:

```text
picf_g28_oldstream_pi05ablated_200_20260605
```

Purpose:

```text
G27 answered "old action shell + current K4 sampler" and failed.  The remaining
strict parity question is whether current code can still reproduce the old
4-22 online train-stream behavior when the sampler shape is also old-like.

This is diagnostic only.  It is not a scalable final recipe because it removes
task-balanced logical updates.  Its purpose is to separate two hypotheses:

H1. Balanced K4 objective is materially harder than the old online stream.
    If true, G28 should descend much faster than G27/G26 despite being less
    scalable.  Then the fix is not to keep old sampling, but to recover its
    action-descent signal inside a scalable balanced objective.

H2. Current code/action path cannot reproduce old 4-22 behavior even under an
    old-like stream.  If true, the root is a code/trainer/action-path regression
    or an old-metric comparability issue, not K4 mixing.
```

Effective critical values:

```text
accum_steps = 1
calvin_bucket_sampling_mode = round_robin
logical_batch_task_count = 0
logical_batch_bucket_normalization = false
logical_batch_dynamic_mixing = false
logical_batch_gradient_surgery = off
logical_batch_action_bucket_ema_normalization = false
semantic_trainable_scope = action_head_and_adapter
semantic_lr_scale = 0.25
grad_clip_norm = 1.0
picf_mode = ablated
burnin_steps = 0
semantic_action_context_flow_residual_enabled = false
```

Tail:

```bash
ssh -i /tmp/picf_g22_key -p 26120 root@px-cloud1.matpool.com 'tail -f /mnt/picf_run_logs/picf_g28_oldstream_pi05ablated_200_20260605.log'
```

Startup verification:

```text
05:20:59 remote log entered PICF Training.

world_size = 2
training_strategy = fsdp_full_shard
picf_mode = ablated
trainable_numel = 13,751,362
total_numel = 3,744,957,843
accum_steps = 1
effective_global_batch = 2
num_steps = 200
lr = 2e-4
min_lr = 2e-5
warmup = 600
unroll_steps = 2
burnin_steps = 0
effective_window_steps = 2
calvin_bucket_sampling_mode = round_robin
logical_batch_task_count = 0
logical_batch_bucket_normalization = false
```

Step50:

```text
loss_action_default_equiv = 0.1581168175

comparison:
  G26 current-shell K4 step50 = 0.1291981190
  G27 old-args K4 step50     = 0.1529971361
  G28 old-stream step50      = 0.1581168175
```

Step50 reading:

```text
The old online-stream shape does not immediately reproduce the historical
4-22 low-loss behavior in current code.  It is worse than both G26 and G27 at
step50.  Continue only to step100 as the final old-stream parity gate; if it
remains high, stop and treat "old sampling shape" as rejected under current
code.
```

Step100:

```text
loss_action_default_equiv = 0.1537972242

comparison:
  G26 current-shell K4 step100 = 0.0770749152
  G27 old-args K4 step100     = 0.1471613646
  G28 old-stream step100      = 0.1537972242
```

Step100 decision:

```text
Stop G28.  Old online-stream shape plus old action shell does not recover the
historical 4-22 low-loss behavior under current code.  It is worse than the
current-shell K4 control and slightly worse than old-args K4.

Rejected hypotheses:
  1. "Old stream shape alone fixes the action plateau."
  2. "Old action shell plus old stream shape is enough under current code."

Remaining strict parity gap:
  current action shell + old online-stream shape.

This is the only unclosed sampler/action-shell parity cell:
  G26 = current shell + K4
  G27 = old shell + K4
  G28 = old shell + old stream
  G29 = current shell + old stream
```

## 3.7 Parity Control G: Current Shell With Old Stream Shape

Planned run:

```text
picf_g29_currentshell_oldstream_pi05ablated_200_20260605
```

Purpose:

```text
G28 shows that restoring old shell and old stream together fails.  G29 closes
the final controlled cell by preserving the better current action shell while
removing K4 balancing.  This tests whether K4/task-balanced objective is the
only reason the current shell plateaus around 0.064.

If G29 drops below G26 step100/200 clearly, K4 is a major difficulty amplifier
and the next scalable fix should improve logical-batch coverage/curriculum
without reverting to old sampling.

If G29 is also high, current-code fresh PI0.5 action descent is not explained
by K4 or old-shell deltas.  Then the root is current trainer/action-path parity
or old metric comparability, and further PICF/sampler changes are not the right
next experiment.
```

Critical values:

```text
picf_mode = ablated
semantic_trainable_scope = backbone_only
semantic_lr_scale = 1.0
accum_steps = 1
calvin_bucket_sampling_mode = round_robin
logical_batch_task_count = 0
logical_batch_bucket_normalization = false
logical_batch_dynamic_mixing = false
logical_batch_gradient_surgery = off
logical_batch_action_bucket_ema_normalization = false
burnin_steps = 0
semantic_action_context_flow_residual_enabled = false
```

Tail:

```bash
ssh -i /tmp/picf_g22_key -p 26120 root@px-cloud1.matpool.com 'tail -f /mnt/picf_run_logs/picf_g29_currentshell_oldstream_pi05ablated_200_20260605.log'
```

Startup verification:

```text
05:41:55 remote log entered PICF Training.

world_size = 2
training_strategy = fsdp_full_shard
picf_mode = ablated
trainable_numel = 3,362,853,650
total_numel = 3,744,949,651
accum_steps = 1
effective_global_batch = 2
num_steps = 200
lr = 2e-5
warmup = 20
unroll_steps = 2
burnin_steps = 0
calvin_bucket_sampling_mode = round_robin
logical_batch_task_count = 0
logical_batch_bucket_normalization = false
semantic_trainable_scope = backbone_only
semantic_action_context_flow_residual_enabled = false
```

Important boundary:

```text
G29 is not the old light action-head shell.  It keeps the current action shell
and current semantic training boundary, which trains a large semantic-backbone
parameter set under an old-like online stream.  This closes the question:
can the current shell itself recover old-like descent if K4 balancing is
removed?
```

Step50:

```text
loss_action_default_equiv = 0.1035719067

comparison:
  G26 current-shell K4 step50 = 0.1291981190
  G27 old-args K4 step50     = 0.1529971361
  G28 old-stream step50      = 0.1581168175
  G29 current-shell oldstream step50 = 0.1035719067
```

Step50 reading:

```text
Current shell + old stream is the best of the old-stream/action-shell parity
controls so far.  It is better than G26 at step50, but this is not enough to
claim success because G26 reached 0.077075 by step100 and 0.064 by step200/300.
Continue to step100.  If G29 cannot beat G26 step100, "remove K4 balancing" is
not sufficient as a scalable repair.
```

Step100:

```text
loss_action_default_equiv = 0.0706284866
loss_action_active7       = 0.3209846914
loss_action_pos           = 0.2633489072
loss_action_rot           = 0.3439514339
loss_action_gripper       = 0.4249917865

bucket losses:
  block_lift          = 0.0818857601
  block_other         = 0.0609806129
  block_push          = 0.0519489014
  drawer              = 0.0707569626
  other               = 0.0784570306
  slider              = 0.0616720705
  switch_button_light = 0.0900231499

comparison:
  G26 current-shell K4 step100       = 0.0770749152
  G27 old-args K4 step100            = 0.1471613646
  G28 old-shell oldstream step100    = 0.1537972242
  G29 current-shell oldstream step100 = 0.0706284866
```

Step100 reading:

```text
Current shell + old stream beats the K4 current-shell control at step100 and
is much better than both old-action-shell controls.  This closes two failed
hypotheses:
  1. The old action shell is required for descent.
  2. Old shell + old stream recreates the old 4-22 behavior.

However, G29 is still far from the historical 4-22 late online band
(roughly 0.02-0.03).  The single-step progress rows also show intermittent
local lows around 0.02-0.03 mixed with higher windows, so the remaining
question is whether old-stream/current-shell gives stable descent or only
local contiguous-task dips.  Continue to step200 before making the next
deployment decision.
```

Step150:

```text
loss_action_default_equiv = 0.0693937242
loss_action_active7       = 0.3160532713

progress-row window means:
  steps 001-050 mean = 0.108502
  steps 051-100 mean = 0.069720
  steps 101-150 mean = 0.066614
```

Step150 reading:

```text
G29 continues to show local low-loss rows, but the 50-step mean has stabilized
near 0.066 rather than approaching the historical 0.02-0.03 online band.  This
is already strong evidence that old-stream/current-shell recovers local
contiguous-window fitting but not stable 4-22-style action descent.  Keep the
run only until step200 to close the planned gate.
```

Step200:

```text
loss_action_default_equiv = 0.0672983676
loss_action_active7       = 0.3066750765
loss_action_pos           = 0.2644939423
loss_action_rot           = 0.3411909938
loss_action_gripper       = 0.3296708465
grad_norm                 = 0.9264690876

bucket losses:
  block_lift          = 0.0597598653
  block_other         = 0.0653010977
  block_push          = 0.0545745368
  drawer              = 0.0742415699
  other               = 0.0818029729
  slider              = 0.0489311984
  switch_button_light = 0.0857013091

progress-row window means:
  steps 001-050 mean = 0.108502
  steps 051-100 mean = 0.069720
  steps 101-150 mean = 0.066614
  steps 151-200 mean = 0.069016
  steps 001-199 mean = 0.078291
```

Step200 final decision:

```text
Stop the old-stream/action-shell parity line.

G29 confirms that the current action shell with old online-stream shape is the
best of the parity controls, but it still plateaus in the same 0.06-0.07 band
as the current K4 ablation.  The intermittent rows near 0.02-0.03 are local
contiguous-window dips, not a stable old 4-22-style descent regime.

Rejected as root fixes:
  1. old action shell;
  2. old online stream shape;
  3. old shell + old stream;
  4. K4 removal alone.

The remaining root is not "we forgot VLA Foundry / ABot-M0 / PiKE style data
mixing."  Those mechanisms exist and have been tested.  The remaining high
value branch is action-path capacity/parity: why the current native PI0.5-style
action path cannot maintain historical low loss across task windows under the
current trainer and parameter boundary.
```

## 3.8 Parity Control H: Native PI0.5 PyTorch Trainer

Run:

```text
picf_g30_native_pi05_direct_300_20260605
```

Purpose:

```text
G26-G29 closed the PICF-core sampler/action-shell parity grid:

  G26 = current shell + K4
  G27 = old shell + K4
  G28 = old shell + old stream
  G29 = current shell + old stream

None restored the historical 4-22 / native PI0.5 low action band.  Therefore
the next non-repeated root test is not another sampler, optimizer, PCGrad,
dynamic mixing, or bridge variant.  It is a direct trainer-path parity test:

  Can the current repository still make the native PI0.5 PyTorch trainer reduce
  CALVIN action flow loss from the same base checkpoint and data path?
```

Old native PI0.5 metadata recovered from:

```text
/mnt/checkpoints/pi05_calvin_nosonata/abc_train_nosonata_full_ddp2/{20000,25000,30000}/metadata.pt
```

Old confirmed contract:

```text
config                 = pi05_calvin_nosonata
pytorch_weight_path    = /mnt/checkpoints/pi05_base_pytorch
calvin_root            = /mnt/calvin_data/task_ABC_D.zip
batch_size             = 2
max_token_len          = 200
action_horizon         = 16
action_dim             = 32
enable_sonata          = False
discrete_state_input   = True
peak_lr=end_lr         = 5e-5
warmup_steps           = 10000
clip_gradient_norm     = 1.0
```

Short-gate diagnostic contract:

```text
Use native `scripts/train_pytorch.py`, not `scripts/picf_core_train.py`.
Start from `/mnt/checkpoints/pi05_base_pytorch`.
Use CALVIN zip + norm stats from the old native run.
Use `batch_size=2`, `max_token_len=200`, `action_horizon=16`, no Sonata.
Use short-gate LR warmup (`warmup=20`, `lr=2e-5`) only to make the 1-2 hour
behavior falsifiable.  This is not exact old 30k schedule parity.
```

Startup caveat:

```text
The first attempt failed before training because `torch.compile` imported the
old remote `networkx` package through `sample_actions`.  The gate was restarted
with `OPENPI_DISABLE_TORCH_COMPILE=1`.

This does not alter the native training loss path: `scripts/train_pytorch.py`
calls `model(observation, actions)` and does not use `sample_actions`.
```

Observed results:

```text
step50  loss = 0.1141  lr = 1.60e-05  grad_norm = 2.00
step100 loss = 0.0669  lr = 2.00e-05  grad_norm = 0.82
step150 loss = 0.0685  lr = 2.00e-05  grad_norm = 1.04
step200 loss = 0.0689  lr = 2.00e-05  grad_norm = 0.88
step250 loss = 0.0572  lr = 2.00e-05  grad_norm = 0.84
step300 loss = 0.0550  lr = 2.00e-05  grad_norm = 0.83
```

Observed single-step behavior:

```text
Many individual rows enter the 0.01-0.03 range, e.g. around steps 33, 75,
83, 122, 178, 219, 239, 244-245, 257-258, 293-295.  The 50-step logged
window, however, remains far above the historical 0.02-0.03 stable band.
```

Decision:

```text
G30 descends below the PICF-core ablated band by step250-300, so enabled
PICF/context is not the only cause of the slow action descent.  The native
path is healthier and faster.

G30 still does not restore the old stable 0.02-0.03 low band.  Therefore the
remaining root is narrower:
  current-code native PI0.5 trainer/data/loss/schedule parity is not yet the
  same as the historical 4-22 native run, or the old low band was logged under
  a non-equivalent effective metric/window.

Do not repeat sampler, PCGrad/CAGrad, SAM, raw-overlap, or PICF auxiliary
repairs as standalone action fixes.  The next non-repeated gate is the exact
old native LR magnitude:
  native trainer + same base/data + lr=5e-5 + short warmup.

This is not blind LR tuning: `5e-5` is the recovered old native
`peak_lr=end_lr` from metadata.  If this lowers the 50-step windows below G30,
the G30 platform was partly LR-limited.  If it does not, the root is deeper
than LR and must be resolved by exact transform/loss/metric parity or a
motor-representation objective.
```

## 3.9 Parity Control I: Native PI0.5 Old LR Magnitude

Planned run:

```text
picf_g31_native_pi05_lr5e5_300_20260605
```

Purpose:

```text
G30 proved current native action flow can descend to ~0.055 by step300, but it
used `lr=2e-5`, below the recovered old native `peak_lr=end_lr=5e-5`.  G31
holds trainer, data, base checkpoint, token length, action horizon, no-Sonata
contract, and compile-disable fix constant while changing only LR magnitude to
the old native value.

This is a high-information one-variable gate:
  if G31 beats G30 windows, old-LR magnitude is a real contributor;
  if G31 matches or worsens G30, the action plateau is not an LR-magnitude
  artifact.
```

Contract:

```text
trainer              = scripts/train_pytorch.py
config               = pi05_calvin_nosonata
pytorch_weight_path  = /mnt/checkpoints/pi05_base_pytorch
calvin_root          = /mnt/calvin_data/task_ABC_D.zip
assets/norm_stats    = /root/openpi/assets/pi05_calvin_sonata
batch_size           = 2
max_token_len        = 200
action_horizon       = 16
enable_sonata        = false
lr                   = 5e-5
warmup_steps         = 20
clip_gradient_norm   = 1.0
num_train_steps      = 300
```

Observed results:

```text
step50  loss = 0.0972  lr = 4.00e-05  grad_norm = 1.55
step100 loss = 0.0651  lr = 5.00e-05  grad_norm = 0.60
step150 loss = 0.0669  lr = 5.00e-05  grad_norm = 0.71
step200 loss = 0.0670  lr = 5.00e-05  grad_norm = 0.57
step250 loss = 0.0603  lr = 5.00e-05  grad_norm = 0.67
step300 loss = 0.0551  lr = 5.00e-05  grad_norm = 0.62
```

Comparison against G30:

```text
G30 lr=2e-5:
  step50  0.1141
  step100 0.0669
  step150 0.0685
  step200 0.0689
  step250 0.0572
  step300 0.0550

G31 lr=5e-5:
  step50  0.0972
  step100 0.0651
  step150 0.0669
  step200 0.0670
  step250 0.0603
  step300 0.0551
```

Decision:

```text
G31 improves only the earliest window.  By step100-300 it is effectively the
same as G30 and does not enter the historical 0.02-0.03 low band.

Therefore the missing action descent is not explained by:
  - enabled PICF/context alone;
  - task-balanced sampler family alone;
  - old action-shell args alone;
  - old online-stream shape alone;
  - current native trainer using lr=2e-5 instead of the old metadata lr=5e-5.

The next non-repeated root-cause step is native parity, not another sampler or
LR branch:
  1. compare old 4-22/native checkpoint metadata and exact current native
     config after dataclass instantiation;
  2. verify CALVIN action target transform and normalization stats byte-for-byte
     or row-for-row against the old run assumptions;
  3. verify that the logged scalar is the same metric/window as the historical
     0.02-0.03 claim;
  4. if all parity checks pass, treat the old low band as a stream/window metric
     artifact and focus on validation/CALVIN behavior rather than chasing a
     non-equivalent scalar.
```

## 3.10 Parity Control J: Native DataLoader Worker Count

Finding from metadata/config audit:

```text
old 4-22/native metadata:
  num_workers = 0

current pi05_calvin_nosonata config default:
  num_workers = 4

G30/G31 command lines:
  did not override num_workers
```

Why this matters mathematically:

```text
CalvinLangSegmentDataset.__getitem__ samples a timestep inside each language
segment with:

  t = self.rng.integers(start, end - action_horizon + 1)

With num_workers=0, one dataset instance advances one RNG stream.  With multiple
workers, dataset instances and their RNG states are copied into workers.  That
changes the sampled-window process even if the checkpoint, action key, norm
stats, and LR are identical.  Because action loss is highly window-dependent,
this is a legitimate parity variable, not a cosmetic throughput setting.
```

Run:

```text
picf_g32_native_pi05_lr5e5_workers0_300_20260605
```

Purpose:

```text
Hold G31's native trainer / base / data / norm / LR contract constant, but set
num_workers=0 to match the old 4-22/native metadata.

Decision:
  if the 50-step windows enter the historical 0.02-0.03 band, the root is
  DataLoader/window sampling parity;
  if they remain near 0.055-0.067, the remaining root is not workers and must
  be metric/window comparability or a deeper current-code native regression.
```

Observed:

```text
step050 0.0977
step100 0.0655
step150 0.0706
step200 0.0584
step250 0.0694
step300 0.0624
```

Decision:

```text
Rejected as the main root cause.  num_workers=0 restores old metadata parity
but does not restore the historical low action-loss band.  It matches G31
closely, so the current plateau is not caused by DataLoader worker count.
```

## 3.11 Parity Control K: Old Checkpoint Same-Stream Forward Loss

Purpose:

```text
After G30-G32, the remaining ambiguity is whether the historical 4-22/native
loss is genuinely comparable to the current native metric.  This control loads
old checkpoints and evaluates them on the current native pi05_calvin_nosonata
data stream, same CALVIN zip, same old assets/norm stats, same model loss, and
num_workers=0.  No optimizer state is loaded and no training occurs.
```

Script:

```text
/mnt/picf_run_logs/eval_native_ckpt_same_stream_20260605.py
```

Important script fix:

```text
The first version incorrectly unpacked create_data_loader() as (loader, data).
That was fixed before recording numbers.  The final script uses the same
Observation.from_dict device-transfer path as scripts/train_pytorch.py.
```

Observed same-stream forward loss:

```text
base pi05 checkpoint, 100 batches:
  mean 0.1760198005
  p50  0.1746104807

old 20000 checkpoint, 100 batches:
  mean 0.0410367110
  p50  0.0348178130

old 25000 checkpoint, 200 batches:
  mean 0.0392374812
  p50  0.0350105427

old 30000 checkpoint, 100 batches:
  mean 0.0456605219
  p50  0.0408731773
```

Decision:

```text
The old checkpoint advantage is real under the current metric and current
loader.  The historical scalar should not be interpreted as a stable 0.02 mean
over the current stream, but the old trained model does reach a clear lower
same-stream band: roughly 0.039-0.046 mean and 0.035-0.041 median.

Therefore the current action plateau is not merely a logging artifact.  The
action target, norm stats, and current model loss can evaluate the old model as
better.  The next non-repeated diagnosis must explain why current training runs
do not move from the base region toward this old-checkpoint band quickly enough.
```

## 3.12 Duration Sufficiency Control L: Native 2000-Step Descent

Run:

```text
picf_g33_native_pi05_lr5e5_workers0_2000_20260605
```

Contract:

```text
native pi05_calvin_nosonata
base checkpoint: /mnt/checkpoints/pi05_base_pytorch
CALVIN zip: /mnt/calvin_data/task_ABC_D.zip
assets/norm: /root/openpi/assets/pi05_calvin_sonata
batch_size: 2
num_workers: 0
lr: 5e-5 after 20-step warmup, flat decay
log_interval: 50
save_interval: 1000
```

Purpose:

```text
G30-G32 only reached 300 steps.  Because old 20000/25000 checkpoints evaluate
around 0.04 mean on the same stream, this control asks whether current native
training naturally approaches that band by 1000-2000 steps, or whether it
stalls permanently around 0.055-0.07.

If it reaches the old same-stream band:
  the current native trainer is viable; PICF integration and complex auxiliary
  branches are the main action-slowing risk.

If it does not:
  the problem is deeper than PICF and workers.  The next control should compare
  current train_pytorch updates against old checkpoint metadata/code contract,
  not repeat sampler/PCGrad/action-aux experiments.
```

Critical old-log correction:

```text
Original log found:
  /mnt/checkpoints/pi05_calvin_nosonata/logs/abc_train_nosonata_full_ddp2.log

Logged old native curve:
  step01000 0.1168  lr=2.50e-06
  step02000 0.0637  lr=7.50e-06
  step03000 0.0556  lr=1.25e-05
  step04000 0.0501  lr=1.75e-05
  step05000 0.0481  lr=2.25e-05
  step10000 0.0461  lr=4.75e-05
  step20000 0.0406  lr=5.00e-05
  step25000 0.0408  lr=5.00e-05
  step30000 0.0406  lr=5.00e-05
```

Implication:

```text
The remembered claim that the old 4-22/native run reached a stable 0.02 action
loss around the early 500-1000 step region is not supported by the recovered
original log.  The verified old target is:
  early 1000-step region: ~0.1168
  2000-step region:       ~0.0637
  3000-5000 region:       ~0.048-0.056
  20000+ region:          ~0.040

Therefore current short controls must be compared against this verified curve,
not against a remembered 0.02 threshold.
```

Observed G33:

```text
step050 0.0971
step100 0.0656
step150 0.0730
step200 0.0578
step250 0.0683
step300 0.0620
step350 0.0552
step400 0.0686
step450 0.0677
step500 0.0558
step550 0.0657
step600 0.0502
step650 0.0613
step700 0.0503
step750 0.0621
step800 0.0495
step850 0.0561
step900 0.0457
step950 0.0484
step1000 0.0577
step1050 0.0579
step1100 0.0599
step1150 0.0429
step1200 0.0442
step1250 0.0486
step1300 0.0453
step1350 0.0440
step1400 0.0516
step1450 0.0393
step1500 0.0473
step1550 0.0525
step1600 0.0544
step1650 0.0501
step1700 0.0446
step1750 0.0588
step1800 0.0533
step1850 0.0529
step1900 0.0523
step1950 0.0393
step2000 0.0519
```

Step100 reading:

```text
G33 is not failing to descend.  It reaches roughly the old native step2000
level (0.0637) by step100 because this control uses a short warmup to reach
5e-5 quickly, while the old native run used a 10000-step warmup and logged
0.1168 at step1000.  Continue to step300/500/1000 to decide whether it also
approaches the old 20k same-stream band (~0.04).
```

Step300 reading:

```text
G33 remains in the 0.058-0.073 window band through step300.  This is not yet
negative relative to the recovered old curve: the old native run logged 0.0637
at step2000 and 0.0556 at step3000.  The correct next check is step500/1000,
not a new sampler or optimizer branch.
```

Step500 reading:

```text
G33 reaches 0.0558 at step500, essentially matching the old native step3000
loss of 0.0556.  This proves the current native action path can descend faster
than the recovered 4-22/native curve under the current code and metric.  The
remaining test is whether it keeps improving toward old step5000/20000 bands
(~0.048 and ~0.040) or stalls around 0.055.
```

Step1000 reading:

```text
G33 remains noisy at 50-step window scale, but it has multiple windows in the
old step4000-5000 band: step600 0.0502, step700 0.0503, step800 0.0495,
step900 0.0457, step950 0.0484.  The step1000 window rebounds to 0.0577, so
the current conclusion is not "fully converged", but it is also not "action
cannot descend".  It is at least as fast as the recovered old native curve in
the first 1000 steps.
```

Step2000 final reading:

```text
G33 decisively rejects "current native action cannot descend".

It reaches the old same-stream trained-checkpoint band twice:
  step1450 = 0.0393
  step1950 = 0.0393

It also reaches or beats the recovered old native log at much earlier logged
training progress:
  recovered old step2000 = 0.0637
  G33 step100            = 0.0656
  G33 step200            = 0.0578

  recovered old step3000 = 0.0556
  G33 step500            = 0.0558
  G33 step600            = 0.0502

  recovered old step20000+ ~= 0.0406
  G33 best windows          = 0.0393

The remaining issue is window/bucket variance, not inability to learn action:
the 50-step scalar still rebounds after good windows, e.g. step1450 0.0393 ->
step1600 0.0544 and step1950 0.0393 -> step2000 0.0519.  That variance does
not justify another sampler/PCGrad/LR branch unless a new code path is being
tested.
```

G33 decision:

```text
Use G33 as the current clean native action-descent control.

Do not diagnose the current repository as "action path cannot descend".
Do not chase a remembered early 0.02 scalar; the recovered original 4-22 log
does not support that threshold.

The causal separation is now:
  1. Native PI0.5 action training in current code is viable and reaches the
     old trained-checkpoint action band by ~1500-2000 short-warmup steps.
  2. PICF-core / slot / context integration is the action-slowing or
     action-noising branch, because the clean native control descends while
     the full PICF variants repeatedly platform or rebound.
  3. The next valid experiment is a gated reintroduction of PICF context into
     this clean native action path, with G33 as the baseline curve.  Any PICF
     variant must be judged against G33 at steps 500/1000/1500/2000, not
     against the incorrect remembered 0.02 early target.
```

## 3.13 Isolation Control M: PICF Enabled, No Action Condition

Run:

```text
picf_g34_picf_enabled_no_action_condition_100_20260605
```

Purpose:

```text
G33 proves the clean native action path descends.  The next non-repeated causal
question is where PICF hurts action:

H1. PICF hurts mainly through action conditioning/prefix/context injection.
    If true, PICF enabled with action condition disabled should stay close to
    the clean ablated/PICF-free action curve, modulo runtime overhead.

H2. PICF hurts through shared training pressure / trainer shell / enabled
    belief computation even when not injected into action.
    If true, action remains slow even with all PICF-to-action gates closed.

This is not a repeat of sampler, LR, old-stream, PCGrad/CAGrad, SAM, raw
overlap, or sidecar tests.  It isolates action-condition dataflow.
```

Contract:

```text
args source:
  /mnt/checkpoints/picf_core/picf_core/picf_g26b_tokenaux_fresh_k4_300_20260604/args.json

critical overrides:
  picf_mode = enabled
  picf_trainable_scope = policy_only
  semantic_trainable_scope = backbone_only
  picf_action_condition_enabled = false
  action_context_tokens = 0
  action_context_output_gate = 0.0
  action_prefix_output_gate = 0.0
  semantic_action_context_readout_aux_weight = 0.0
  semantic_action_context_token_aux_weight = 0.0
  semantic_action_context_flow_residual_enabled = false
  calvin_bucket_sampling_mode = task_uniform
  logical_batch_task_count = 4
  logical_batch_bucket_normalization = true
```

Startup correction:

```text
The first launcher used a missing args.json path and never started training.
The second launcher used the correct args source but accidentally left
picf_trainable_scope=all, causing CUDA OOM at step2.  Both are launcher
failures and must not be counted as experiment evidence.

The corrected launcher uses picf_trainable_scope=policy_only.  Startup config
confirms:
  trainable_scope = policy_only
  trainable_numel = 3,360,754,450
  picf_mode = enabled
  context_tokens = 0
  prefix_gate = 0.0
  context_gate = 0.0
  flow_residual_enabled = false
```

Execution result:

```text
The corrected `policy_only + backbone_only` launcher also OOMed at optimizer
step2:

  CUDA OOM while allocating 192 MiB
  trainable_numel = 3,360,754,450
  picf_mode = enabled
  context_tokens = 0
  prefix_gate = 0.0
  context_gate = 0.0

This is a resource rejection, not an action-loss result.  It proves that
full-backbone PaliGemma/Gemma cotrain plus enabled PICF belief forward does not
fit the 2x40GB diagnostic budget under the current FSDP/accumulation contract.
It must not be used as evidence that "PICF enabled no-action-condition" is bad.
```

## 3.14 Isolation Control N: PICF Enabled, No Action Condition, Lightweight Action Boundary

Planned run:

```text
picf_g35_picf_enabled_no_action_condition_headadapter_100_20260605
```

Purpose:

```text
G34 full-backbone no-action-condition is the scientifically clean control but
is resource-rejected on 2x40GB.  G35 keeps the same dataflow isolation:

  PICF enabled
  PICF-to-action condition disabled
  action context/prefix gates closed
  action context aux and flow residual disabled

but changes only the trainable semantic boundary to `action_head_and_adapter`
so it can run within memory.  This is not a final recipe and not a replacement
for G33.  It is a fast causality probe:

  if action remains close to known lightweight action-boundary controls, then
    enabled PICF forward without action injection is not the primary poison;
  if action is much worse even with no injection, then the enabled PICF trainer
    shell/belief computation itself is action-negative or too expensive.
```

Known comparability limits:

```text
G35 cannot be compared directly to G33 full native backbone cotrain, because
`action_head_and_adapter` trains only wrapper-local action/time projections and
PICF action adapters.  It must be compared to prior local-boundary probes
recorded in G13/G12, and only used to decide whether to spend resources on a
larger no-condition full-backbone isolation run.
```

Acceptance gate:

```text
Run only to step50/100.

If G35 OOMs:
  close the no-condition isolation as "requires larger GPU or further memory
  engineering"; do not repeat on 2x40GB.

If G35 runs but action is clearly worse than prior action_head_and_adapter K4
controls:
  enabled PICF forward/trainer shell is independently action-negative.

If G35 is similar to prior action_head_and_adapter K4 controls:
  the main action harm is likely not the mere presence of PICF forward, but the
  PICF-to-action condition/residual/aux path or full-backbone resource boundary.
```

Interpretation gate:

```text
Compare step50/100 against:
  G33 native clean: step50 0.0971, step100 0.0656
  G26 current-shell K4 ablated: step50 0.1292, step100 0.0771

If G34 is close to G26/G33:
  action conditioning/prefix/context injection is the likely culprit; next
  test a small bounded bridge with gate <= 0.05 and no aux pressure.

If G34 is much worse:
  the problem is not only condition injection.  The PICF enabled trainer shell,
  full belief forward, or shared semantic optimization pressure is already
  action-negative.  Then do not reintroduce PICF context before reducing
  trainable boundary or decoupling action expert more strictly.
```

## 4. Next Decision Tree

After G33:

```text
If a future full PICF action run cannot match G33:
  the root is not native action capacity.  Treat PICF context as an auxiliary
  condition source, not as a replacement for the native PI0.5 action path.
  Reintroduce PICF through a gated bridge with bounded residual scale and
  compare directly against G33.

If a future full PICF action run matches G33 through step2000:
  run longer and judge by CALVIN/video evidence.  Do not add more auxiliary
  losses only because raw overlap or slot diagnostics are imperfect.

If a future sampler/LR proposal repeats G26-G33:
  reject before running unless it changes a code path not covered here.  The
  sampler/LR/old-shell parity grid is already closed for the current action
  descent question.
```

## 5. Experiment Part Checklist

```text
[x] Verify current run uses task_uniform K4.
[x] Verify current run has no PICF structure losses in loss_total.
[x] Record step50.
[x] Record step100 and compare against G26-B step100 = 0.1165.
[x] Record step150 and compare against G26-B step300 ~= 0.0654.
[x] Record step200 and compare against G26-B step200 = 0.0766.
[x] Record step300 and compare against G26-B step300 = 0.0654.
[x] Stop decisively if the branch is clearly worse after step200/300.
[x] Start old 4-22 args + current K4 sampler parity control.
[x] Record G27 step50/100 and decide whether action-shell parity fixes the
    plateau: no, rejected at step100.
[x] Start old action shell + old online-stream parity control.
[x] Record G28 step50/100 and decide whether old stream shape fixes the
    plateau: no, rejected at step100.
[x] Start current action shell + old online-stream parity control.
[x] Record G29 step50/100 and decide whether K4 alone explains the current
    platform: partial.  G29 beats G26 at step100, but does not recover the
    historical 0.02-0.03 band.  Step200 is required to distinguish stable
    descent from local-window dips.
[x] Record G29 step200 and decide whether to keep old-stream evidence as a
    curriculum signal or reject it as non-scalable local-window fitting:
    rejected as a root fix.  It gives local-window dips but no stable descent.
[x] Run native train_pytorch G30/G31/G32 to isolate current native action path
    from PICF-core trainer and worker-count parity.
[x] Evaluate old 20k/25k/30k checkpoints on the current same-stream native
    metric to establish the true comparable target band.
[x] Recover original 4-22/native log and correct the historical target curve.
[x] Run G33 native 2000-step duration control and decide whether current native
    action can descend: yes.  It reaches 0.0393 windows at steps 1450 and 1950.
[x] Run G34 full-backbone no-action-condition isolation: resource-rejected by
    CUDA OOM at step2; not scientific evidence.
[x] Run G35 lightweight no-action-condition isolation to decide whether PICF
    enabled forward/trainer shell alone is action-negative under a runnable
    trainable boundary: it ran, but did not isolate forward-only because
    `loss_total_minus_action` stayed nonzero.
[ ] Run G36 PICF-enabled/no-action-condition/action-only objective.  This is
    the next non-repeated isolation: keep PICF forward enabled but set all
    non-action loss weights to zero so `loss_total_minus_action == 0`.

## 8. G35 Result: No Direct Action Condition Is Not Enough

Run:

```text
picf_g35_picf_enabled_no_action_condition_headadapter_100_20260605
```

Confirmed startup/dataflow:

```text
picf_mode = enabled
picf_trainable_scope = policy_only
semantic_trainable_scope = action_head_and_adapter
pi_action_condition_token_count = 0
pi_context_token_count = 0
pi_context_adapter_gate = 0
pi_context_flow_residual_enabled = 0
logical_batch_distinct_bucket_count = 4
```

Step50:

```text
loss_action_default_equiv = 0.1569157541
loss_total_minus_action    = 0.2365500778
loss_alignment             = 0.2365500778
loss_total                 = 0.5503816009
```

Step100:

```text
loss_action_default_equiv = 0.1383437216
loss_action_active7       = 0.4733563364
loss_total_minus_action   = 0.2510643303
loss_alignment            = 0.2510643303
loss_total                = 0.5277517438

loss_anchor_pv            = 0.6809128523
loss_anchor_object_pull   = 0.5972164869
loss_mapg_routing         = 0.3948031664
loss_slot_jepa            = 0.2466281950

aqr_same_role_support_overlap_max        = 0.4663918614
aqr_active_same_role_support_overlap_max = 0.0087323636
aqr_context_same_role_support_overlap_max= 0.0232953764
```

Bucket action at step100:

```text
block_lift          0.1448191361
block_other         0.1523317248
block_push          0.1152697237
drawer              0.1570764834
other               0.1316765038
slider              0.1181600572
switch_button_light 0.1567421088
```

Interpretation:

```text
G35 is worse than the clean native controls:
  G33 native clean step100 ~= 0.0656
  G26 current-shell K4 ablated step100 ~= 0.0771

Because all action-context/token gates are zero, this run rules out direct
PICF-to-action token injection as the only cause.  However it does not prove
that the PICF forward itself is poisonous, because the objective still contains
non-action PICF structure pressure:

  loss_total_minus_action = loss_alignment ~= 0.25

Therefore the correct next isolation is G36:

  PICF enabled
  no action condition
  action context disabled
  all non-action structure/future/alignment weights set to zero
  require loss_total_minus_action == 0

If G36 recovers the native action curve, the root is structural-loss gradient
contamination.  If G36 remains close to G35, the root is PICF enabled
forward/trainer-shell overhead or the lightweight action boundary itself.
```

## 9. G36 Plan: PICF Forward With Action-Only Objective

Run:

```text
picf_g36_picf_forward_actiononly_headadapter_100_20260605
```

Purpose:

```text
This is the exact follow-up required by G35.  It is not a new sampler, LR,
gradient-surgery, raw-overlap, SAM, or bridge experiment.

The only causal change from G35 is objective isolation:

  keep PICF enabled;
  keep PICF-to-action condition disabled;
  keep action context/prefix gates closed;
  keep the runnable action_head_and_adapter trainable boundary;
  set every non-action physical/semantic/alignment/object/OWM loss weight to
  zero;
  set aux budget ratios/floors to zero as a backstop.

Required runtime invariant:

  loss_total_minus_action == 0

If that invariant fails, G36 is invalid and must be stopped.
```

Launcher deltas from G35:

```text
lambda_visual_latent = 0
lambda_visual_real = 0
lambda_tactile_real = 0
lambda_point_real = 0
lambda_semantic_future_aux = 0
lambda_anchor_pv = 0
lambda_anchor_object_pull = 0
lambda_pv_weak = 0
lambda_pt = 0
lambda_vl_* = 0
lambda_mapg_* = 0
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
lambda_slot_quality = 0
lambda_vcap_* = 0
lambda_object_explanation_* = 0
aux_budget_physical_ratio = 0
aux_budget_semantic_ratio = 0
aux_budget_alignment_ratio = 0
aux_budget_floor = 0
aux_budget_alignment_floor = 0
```

Decision rule:

```text
Compare step50/100 directly against:

  G35 no-condition but non-action objective active:
    step50  action=0.1569, non-action=0.2366
    step100 action=0.1383, non-action=0.2511

  G33/G26 clean native controls:
    G33 native step100 ~= 0.0656
    G26 K4 ablated step100 ~= 0.0771

If G36 moves close to G33/G26:
  the root is structural-loss gradient contamination.  Future PICF integration
  must train action with a clean native path and schedule structure losses
  separately or through a bounded delayed bridge.

If G36 remains close to G35:
  the root is not the structure loss.  Enabled PICF forward/trainer shell or
  the lightweight action boundary itself is action-negative under this resource
  contract.  The next valid run is a paired current-shell `picf_mode=ablated`
  action_head_and_adapter control, not another sampler experiment.
```

G36 step50 result:

```text
loss_total              = 0.3138237596
loss_total_minus_action = 0.0
loss_action             = 0.3138237596
loss_action_default_equiv = 0.1569118798
loss_alignment          = 0.0
logical_batch_distinct_bucket_count = 4
```

Comparison:

```text
G35 step50 no-condition but non-action objective active:
  action_default = 0.1569157541
  non_action     = 0.2365500778

G36 step50 no-condition and action-only objective:
  action_default = 0.1569118798
  non_action     = 0.0
```

Decision:

```text
Stop G36 at step50.  It precisely satisfies the action-only invariant and still
matches G35 action almost exactly.  Therefore the G35 failure is not caused by
non-action structure-loss gradient contamination.

The remaining unisolated factor is:

  enabled PICF forward/trainer shell
  versus
  the lightweight `action_head_and_adapter` trainable boundary itself.

Next run G37:

  same current-shell K4 setup,
  same `semantic_trainable_scope=action_head_and_adapter`,
  but `picf_mode=ablated`.

If G37 is healthy, PICF enabled forward/trainer shell is action-negative.  If
G37 is also bad, the lightweight boundary is the cause and G35/G36 should not
be used to judge PICF forward.
```

G37 startup correction:

```text
The first G37 launcher inherited `burnin_steps>0` from the enabled-PICF args and
failed validation:

  burnin_steps > 0 requires picf_mode=enabled

This is a configuration rejection, not a training result.  The valid paired
control must set:

  picf_mode=ablated
  burnin_steps=0
```

G37 step50 result:

```text
picf_mode = ablated
semantic_trainable_scope = action_head_and_adapter
burnin_steps = 0
logical_batch_distinct_bucket_count = 4

loss_total = 0.1570495218
loss_total_minus_action = 0.0
loss_action_default_equiv = 0.1570495218
loss_action_active7 = 0.5505002141

bucket action_default:
  block_lift          0.1601140152
  block_other         0.1467074391
  block_push          0.1291863124
  drawer              0.1712336333
  other               0.1772762341
  slider              0.1390013826
  switch_button_light 0.1752149334
```

Comparison:

```text
G35 step50 PICF enabled + no action condition + structure objective:
  action_default = 0.1569157541

G36 step50 PICF enabled + no action condition + action-only objective:
  action_default = 0.1569118798

G37 step50 PICF ablated + action_head_and_adapter + action-only objective:
  action_default = 0.1570495218

G26 current-shell K4 ablated + backbone_only:
  step50 action_default = 0.1291981190
  step100 action_default = 0.0770749152
```

Decision:

```text
Stop G37 at step50.  The bad value reproduces even with PICF fully ablated.
Therefore this branch is not a PICF-forward issue and not an auxiliary-loss
issue.  It is the lightweight `action_head_and_adapter` trainable boundary
itself under this action shell.

Root now isolated:

  action_head_and_adapter is too narrow to reproduce fast action descent.
  The clean-descending boundary is the native/backbone/model action-expert path
  shown by G26/G33.

Do not use G35/G36/G37 to argue that PICF semantics are bad.  They only prove
that local action projection/time heads cannot carry the action descent alone.

Next valid repair:

  run action with the native/backbone action-expert boundary;
  keep K4 task-balanced logical batch and per-bucket normalization;
  if PICF is enabled, solve memory for `backbone_only` or use a strictly
  detached/bounded PICF condition that does not force the lightweight
  `action_head_and_adapter` boundary to carry action learning.
```
```

## 6. Deployment Part Checklist

```text
[x] Keep K4 task-balanced logical batch and per-bucket normalization.
[x] Keep temperature/ratio/dynamic mixing as explicit knobs, not defaults.
[x] Keep PCGrad/CAGrad as explicit diagnostics, not defaults.
[x] Keep SAM default off and archived.
[x] Keep action-context bridge guarded and measured by canonical action MSE.
[x] Decide whether PI0.5-like ablation proves PICF/context is action-negative:
    yes, relative to G26-B fresh, but not sufficient to recover old action
    descent.
[x] Audit current trainer parity against historical 4-22 PI0.5 before any new
    PICF architecture change.
[x] Set G33 as the required clean-action baseline curve for future PICF
    integration gates.
[ ] Build the next PICF integration test as a bounded/gated residual into the
    clean native action path, not as another sampler, LR, raw-overlap, or
    gradient-surgery repair.
[ ] Do not rerun G34 on 2x40GB unless the memory contract changes.  Use G35
    only as a lightweight causality probe, not as final architecture evidence.
```

## 7. Local Code Follow-Through Checks

Executed locally on 2026-06-05:

```bash
uv run pytest -q scripts/picf_core_train_test.py \
  -k 'calvin_bucket_sampling_weights or bucket_sequence_without_replacement or bucket_sequence_round_robin or logical_batch_loss_scales'

result:
  6 passed, 147 deselected

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'dynamic_bucket_sampling_weights or gradient_surgery_params or pcgrad or cagrad'

result:
  6 passed, 147 deselected
```

Code follow-through conclusion:

```text
The VLA Foundry / ABot-M0 / PiKE-style infrastructure is not just documented:
the local code paths for bucket weights, without-replacement logical bucket
sequence, DDP-aware logical-batch loss scaling, dynamic mixing, and scoped
gradient surgery are covered by passing tests.

Therefore the current action-descent failure should not be diagnosed as
"these methods were not wired."  They are wired and historically tested; they
are useful controls but not the remaining root after G33.  G33 shows the clean
native path descends.  Future work should use that clean path as the baseline
and add PICF context only through measured, bounded conditioning.
```

## 8. Current Root And Non-Repeated Next Gate

Current isolated root, after G35/G36/G37:

```text
The remaining action-descent failure is not caused by:
  - missing task-balanced logical batches;
  - missing per-bucket loss normalization;
  - temperature/ratio/dynamic sampler not being implemented;
  - PCGrad/CAGrad not being implemented;
  - non-action PICF structure losses;
  - enabled PICF forward by itself.

The supported root is:
  `semantic_trainable_scope=action_head_and_adapter` is too narrow.  It trains
  wrapper-local action/time/context adapters but not the native PI0.5/Gemma
  action expert stack.  G37 reproduces the bad action scalar even with PICF
  ablated and action-only objective.
```

Therefore the next experiment must not be another sampler or LR variant.  The
next non-repeated gate is:

```text
G38:
  Keep PICF enabled, but remove direct PICF action condition and all non-action
  auxiliary losses.

  Train the native/backbone action-expert path:
    semantic_trainable_scope = backbone_only

  Use FSDP + activation checkpointing because the previous no-checkpoint G34
  version OOMed on 2x40GB.
```

Mathematical purpose:

```text
Let A(theta) be the native PI0.5 action expert and C(phi) be the PICF belief
router/context path.

G35/G36 tested:
  update only small local adapters h, with A frozen:
    a_hat = A(frozen)(h_phi(context), x)
  result: no fast action descent.

G38 tests:
  update A(theta) while C(phi) is present but not directly conditioning action:
    a_hat = A_theta(x)
    C(phi) is measured as forward runtime/context scaffold, not the action
    carrier.

If G38 descends like G26/G33, the action solution is:
  train action through the native action expert boundary, then reintroduce
  PICF as a bounded residual/conditioning signal.

If G38 OOMs, the blocker is resource/memory, not a sampler/loss theory issue.

If G38 runs but remains near G35/G36/G37, the current PICF-enabled trainer shell
has an action-path side effect beyond direct condition and aux losses; inspect
forward dataflow before any 30K run.
```

G38 contract:

```text
picf_mode = enabled
picf_trainable_scope = policy_only
semantic_trainable = true
semantic_trainable_scope = backbone_only
picf_action_condition_enabled = false
action_context_tokens = 0
semantic_action_context_*_aux_weight = 0
semantic_action_context_flow_residual_enabled = false
all non-action lambdas = 0
calvin_bucket_sampling_mode = task_uniform
logical_batch_task_count = 4
logical_batch_bucket_normalization = true
training_strategy = fsdp_full_shard
semantic_gradient_checkpointing = true
window_activation_checkpointing = true
```

Success gate:

```text
step50 action_default should be materially below the G35/G36/G37 band:
  bad band:    ~0.157
  useful gate: <=0.130 by step50, matching G26 backbone_only behavior

step100 action_default should approach the G26/G33 region:
  useful gate: <=0.08 by step100

If neither condition holds, stop immediately and do not continue to 300/1000.
```

G38 launch audit, 2026-06-05 11:54 CST:

```text
Remote:
  host = px-cloud1:26120 / qe93X5
  repo = /root/openpi_g25_20260604
  tmux = picf_g38_picf_enabled_backbone_actiononly_150_20260605
  log  = /mnt/picf_run_logs/picf_g38_picf_enabled_backbone_actiononly_150_20260605.log

Verified runtime contract:
  requested_task_count = 4
  runtime_task_count   = 4
  accum_steps          = 2
  effective_global_batch = 4
  picf_mode            = enabled
  semantic_trainable_scope = backbone_only
  picf_action_condition_enabled = false
  action_context_tokens = 0
  non-action objective weights = 0

Resource observation:
  step1 reached.
  step1 wall time was about 64s.
  GPU memory was about 39.4-39.9GB on 2xA100-40GB.
  The trainer disabled PaliGemma gradient checkpointing under accum_steps=2
  to avoid DDP "mark ready twice" failures.  Therefore G38 is a valid causal
  gate but not a production-speed recipe on 2x40GB.

Log hygiene:
  The log contains a SignalException from the earlier killed K2 mislaunch.
  Current K4 ranks are still alive, so that SignalException is not the G38
  result.  Judge only after a real step50 metric.
```

Local contract tests, 2026-06-05:

```text
uv run pytest -q scripts/picf_core_train_test.py \
  -k 'pi0_trainable_scope_backbone_only or policy_only_trainable_scope or \
      logical_batch_loss_scales or bucket_sequence_without_replacement or \
      calvin_bucket_sampling_weights or dynamic_bucket_sampling_weights or \
      gradient_surgery_params or pcgrad or cagrad'

Result:
  13 passed, 140 deselected.

Coverage:
  - backbone_only/native action readout boundary
  - policy_only PICF trainability boundary
  - logical-batch loss scaling
  - without-replacement bucket sequence
  - bucket sampling weights
  - dynamic PiKE-style bucket weights
  - scoped PCGrad/CAGrad parameter selection and code paths
```

Requested VLA-method closure table:

```text
This table is intentionally repeated in the current action-descent checklist so
future runs do not restart already-closed branches.

Method family from 2025-2026 VLA literature:

1. Probabilistic/task-balanced dataset mixing
   Status: deployed and kept.
   Code: calvin_bucket_sampling_mode={task_uniform,temperature,trajectory},
         bucket_weight_spec, bucket_sequence_for_logical_step.
   Evidence: K4 without-replacement coverage and per-bucket metrics are logged.
   Decision: necessary baseline, not sufficient alone.

2. Gradient accumulation as logical batch
   Status: deployed and kept.
   Code: accum_steps plus logical_batch_task_count.
   Evidence: G38 uses world_size=2, accum_steps=2, logical_batch_task_count=4.
   Decision: keep K4 where feasible; K8/K12 are resource-limited on 2x40GB.

3. Per-task/per-bucket loss normalization
   Status: deployed and kept.
   Code: logical_batch_bucket_normalization.
   Evidence: all valid G12/G26/G35/G36/G37/G38 gates run with it.
   Decision: production default for CALVIN task-balanced updates.

4. Temperature / explicit ratio sampling
   Status: implemented, rejected as action-descent fix.
   Evidence: documented in G12; structure may change but action did not recover.
   Decision: keep as config knob only, not default.

5. Dynamic PiKE-style mixing
   Status: implemented, rejected as current action-descent fix.
   Code: _dynamic_bucket_sampling_weights and logged dynamic q_b(t).
   Evidence: G12-DYN improved some structure signals transiently but did not
             restore action descent.
   Decision: diagnostic knob only.

6. Scoped PCGrad/CAGrad
   Status: implemented, rejected as current action-descent fix.
   Code: _pcgrad_project_and_sum, _cagrad_project_and_sum,
         _logical_batch_gradient_surgery_params.
   Evidence: local tests cover code path; G12 runs did not fix action.
   Decision: do not default; only use after new gradient-cosine evidence.

7. Whole-model PCGrad/CAGrad
   Status: rejected by design for current scale.
   Reason: expensive per-task gradients over the large backbone and mismatched
           to the measured action-readout boundary issue.

8. Action expert / trainable boundary
   Status: still the active root branch.
   Evidence:
     G35/G36/G37 show action_head_and_adapter remains near the bad band
     (~0.157 at step50), even with PICF action condition and non-action losses
     removed.
   Current test:
     G38 tests PICF enabled + no direct action condition + backbone_only.

9. Action expert MoE / embodiment adapters / System2 planning
   Status: not the next default.
   Reason:
     CALVIN is single-embodiment and the current failure is already isolated
     before requiring expert routing or subtask labels.
   Decision:
     add only if backbone/native action path is healthy and later gradient
     diagnostics prove persistent task-family conflict.
```

G38 result:

```text
G38 is resource-rejected on 2xA100-40GB, not action-rejected.

Observed:
  step1 reached: loss=0.4628 at lr=3.33e-07.
  step2 micro_step=2 failed during backward on both ranks:
    torch.OutOfMemoryError: tried to allocate 192MiB
    free memory: about 185-190MiB
    process memory: about 39.30GiB / 39.49GiB

The failure happened before any step50 action metric, so G38 cannot answer
whether PICF-enabled + backbone_only restores action descent.
```

G39 resource-corrected gate:

```text
Keep the G38 mathematical contract:
  picf_mode = enabled
  semantic_trainable_scope = backbone_only
  picf_action_condition_enabled = false
  action_context_tokens = 0
  non-action objective weights = 0
  K4 task-uniform logical batch and per-bucket normalization

Change only the semantic token length:
  semantic_max_length = 200

Reason:
  The old PI0.5/4-22 native contract used max_token_len=200.  G38 inherited
  semantic_max_length=256 and OOMed by only 192MiB.  Reducing to 200 is not a
  capacity hack; it restores the historical PI0.5 token contract and should
  lower activation memory enough to make the causal test runnable.

If G39 runs:
  apply the same step50/step100 gates as G38.

If G39 still OOMs:
  the 2x40GB machine cannot run PICF-enabled + backbone_only + K4.  Do not
  reduce to K2 as a decisive action test; K2 is known to under-approximate the
  intended multi-task gradient.  Use larger memory or test action descent on
  the native/PICF-ablated path only.
```

G39 launch audit, 2026-06-05 12:05 CST:

```text
Startup fixes:
  The first remote G39 launch failed because torchrun child processes did not
  have the repo root on PYTHONPATH.
  The second launch found scripts but not src-layout openpi modules.
  Final launcher sets:
    PYTHONPATH=/root/openpi_g25_20260604:/root/openpi_g25_20260604/src

Verified runtime:
  semantic_max_length = 200
  requested_task_count = 4
  runtime_task_count = 4
  semantic scope = backbone_only
  PICF action condition = off
  non-action objectives = off

Resource result:
  step2 succeeded.
  GPU memory after step2: about 38.0-38.6GiB on 2xA100-40GB.
  Therefore G39 fixes the G38 OOM while preserving the causal contract.

Do not judge from step1/step2:
  step1 progress loss = 0.4628
  step2 progress loss = 0.0831
  These are too few windows and too early in warmup.  Keep the predeclared
  decision point at step50.
```

G39 result update, 2026-06-05 12:10 CST:

```text
G39 is also resource-rejected on 2xA100-40GB.

Observed:
  step1 reached: loss=0.4628
  step2 reached: loss=0.0831
  step3 micro_step=2 failed during backward:
    torch.OutOfMemoryError: tried to allocate 790MiB
    GPU0 free memory: about 776MiB
    process memory: about 38.73GiB

This confirms:
  semantic_max_length=200 fixes the initial G38 margin but does not leave
  enough activation headroom for PICF enabled + backbone_only + K4 on 2x40GB.

This does not confirm:
  action non-descent.
  The run did not reach the step50/step100 action gates.

Decision:
  Do not spend more 2x40GB time trying to force this exact contract with
  smaller K2/K1 and then claim it answers the multi-task-gradient question.
  K4 is the minimum meaningful logical-batch gate from the G12/G26 evidence.

Next non-repeated branch:
  Run the native action-descent positive-control contract on the same machine
  with the new task-balanced sampler, but without PICF runtime activation:
    picf_mode = ablated
    semantic_trainable_scope = backbone_only
    semantic_max_length = 200
    K4 logical batch
    non-action objectives = 0

Purpose:
  Separate two causes:
    A. native PI0.5/PaliGemma action expert + K4 still descends on this
       machine/config, meaning the remaining blocker is PICF runtime memory
       and trainable-boundary integration.
    B. native path also fails, meaning the current launcher/data/sampler stack
       diverged from the earlier G26 positive control and must be debugged
       before any PICF claim.
```

G40 live positive-control replay, 2026-06-05:

```text
tmux:
  picf_g40pi05like_k4_ablation_repro_300_20260605

log:
  /mnt/picf_run_logs/picf_g40pi05like_k4_ablation_repro_300_20260605.log

cmd:
  /mnt/picf_run_logs/picf_g40pi05like_k4_ablation_repro_300_20260605.cmd

local archived cmd:
  tmp/picf_g40pi05like_k4_ablation_repro_300_20260605.cmd
```

Contract:

```text
Source:
  copy the already successful G26pi05like command contract, changing only
  exp_name/save horizon.

Key settings:
  picf_mode = ablated
  burnin_steps = 0
  mapg_enabled = false
  aqr_mapg_enabled = false
  vl_anchor_router_enabled = false
  semantic_trainable = true
  semantic_trainable_scope = backbone_only
  training_strategy = fsdp_full_shard
  accum_steps = 2
  calvin_bucket_sampling_mode = task_uniform
  logical_batch_task_count = 4
  logical_batch_bucket_normalization = true
  calvin_bucket_sample_without_replacement = true
  lr = 2e-4
  warmup_steps = 600
```

Reason:

```text
G38/G39 failed by memory before step50, so they cannot answer action descent.
G40 does not test PICF-enabled runtime.  It tests whether the same remote
machine/current repo can still reproduce the known healthy native-action
positive control:

  G26pi05like step50  action_default = 0.1291981190
  G26pi05like step100 action_default = 0.0770749152

If G40 matches those bands, the launcher/data/sampler stack is healthy and the
remaining blocker is fitting PICF-enabled native-action cotrain into resource
or redesigning PICF/action integration.  If G40 misses those bands, the current
sync/runtime is not reproducing a known healthy control and must be debugged
before any new VLA-method conclusion.
```

G40 step50 result, 2026-06-05 12:29 CST:

```text
step = 50
loss_action_default_equiv = 0.1292344034
loss_total = 0.1292344034
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
logical_batch_global_micro_count = 4
lr = 1.6666666667e-05
grad_norm = 1.7212671041

bucket action losses:
  block_lift          0.1265668345
  block_other         0.1244853456
  block_push          0.1047459694
  drawer              0.1426645967
  other               0.1481361613
  slider              0.1036719375
  switch_button_light 0.1554354078
```

Comparison:

```text
G26pi05like step50 action_default = 0.1291981190
G40 step50 action_default         = 0.1292344034
absolute delta                    = 0.0000362844
relative delta                    = 0.0281%
```

Interpretation:

```text
G40 step50 reproduces the known healthy G26 native-action positive control.
Therefore the current remote launcher, data path, task-uniform K4 logical
batch, per-bucket normalization, quantile action normalization, and
backbone_only trainable boundary are not broken.

This result rejects the hypothesis that the current action plateau is caused
by forgetting to implement VLA Foundry / ABot-M0 / PiKE style data mixing.
Those mechanisms are present and can reproduce the historical healthy native
action curve when PICF runtime is ablated.

Continue G40 to step100.  If step100 also matches G26, stop repeating sampler,
PCGrad/CAGrad, dynamic-mixing, or optimizer-only branches.  The next valid root
branch is PICF-enabled resource/action-boundary integration.
```

G40 step100 result, 2026-06-05 12:45 CST:

```text
step = 100
loss_action_default_equiv = 0.0771813095
loss_total = 0.0771813095
loss_total_minus_action = 0.0
loss_action_active7 = 0.3500132561
logical_batch_distinct_bucket_count = 4
logical_batch_global_micro_count = 4
lr = 3.3333333333e-05
grad_norm = 0.8564581275

bucket action losses:
  block_lift          0.0935505359
  block_other         0.0837164198
  block_push          0.0599859872
  drawer              0.0976252031
  other               0.0705873210
  slider              0.0499551827
  switch_button_light 0.0915238922
```

Comparison:

```text
G26pi05like step100 action_default = 0.0770749152
G40 step100 action_default         = 0.0771813095
absolute delta                     = 0.0001063943
relative delta                     = 0.1380%
```

Interpretation:

```text
G40 step100 reproduces the known healthy G26 native-action positive control.
This closes the "missing VLA data mixing" hypothesis for the current repo:

  task-uniform logical batch is active;
  gradient accumulation is acting as a logical batch;
  per-bucket loss normalization is active;
  current trainable_scope=backbone_only restores the historical native
  PI0.5/PaliGemma action-capacity boundary;
  action-only scalar equals loss_total, so no auxiliary loss is hiding action.

Therefore the current action plateau/non-descent problem is not explained by:

  forgetting VLA Foundry/ABot-M0-style task-balanced mixing;
  missing gradient accumulation;
  missing per-bucket normalization;
  needing temperature/ratio sampling as a standalone fix;
  needing dynamic PiKE as a standalone fix;
  needing PCGrad/CAGrad as a standalone fix;
  needing another optimizer-reset-only or LR-only rerun.

The remaining valid root branch is narrower and harder:

  PICF-enabled runtime + native action-capacity boundary does not fit on
  2xA100-40GB under the K4 contract (G38/G39 OOM);
  earlier PICF-enabled action runs used action_head_and_adapter or direct
  PICF/context action conditions that were action-slowing;
  therefore the next repair must keep the native dense PI0.5 action path and
  introduce PICF only as bounded/controlled context, or reduce PICF memory
  without changing the K4/native-action mathematical contract.
```

Run status:

```text
Stopped after the step100 gate passed.  Continuing to step300 would only
remeasure the same native-action positive control and would not answer the
remaining PICF-enabled boundary/resource question.
```

## 12. Next Gate: G41 PICF Runtime With Native Action Boundary

G41 is the next non-duplicate experiment after G40.

Question:

```text
Can the current 2xA100-40GB machine run PICF runtime while preserving the
native PI0.5/PaliGemma action-capacity boundary that G40 proved healthy?
```

This is not another sampler, LR, optimizer, dynamic mixing, or PCGrad test.
Those branches have already been tested or locally verified and G40 proves the
native action path descends under the current K4 task-balanced sampler.

G41 contract:

```text
picf_mode = enabled
semantic_trainable = true
semantic_trainable_scope = backbone_only
training_strategy = fsdp_full_shard
accum_steps = 2
logical_batch_task_count = 4
calvin_bucket_sampling_mode = task_uniform
logical_batch_bucket_normalization = true
calvin_bucket_sample_without_replacement = true
picf_action_condition_enabled = false
action_context_tokens = 0
semantic_action_context_token_aux_weight = 0
semantic_action_context_flow_residual_enabled = false
semantic_action_context_readout_aux_weight = 0
semantic_max_length = 128
window_activation_checkpointing = true
```

Why this is mathematically clean:

```text
The optimized action estimator stays the same family as G40:

  grad L_action(theta_semantic_action)

PICF is computed as runtime belief state but is not allowed to perturb the
native PI0.5 action prefix/context in this gate:

  d L_action / d PICF_context = 0

So if action descends, PICF runtime can coexist with the native action boundary.
If action fails or OOMs, the failure is resource/boundary integration, not
task-balanced data mixing.
```

Memory changes relative to G39:

```text
semantic_max_length 200 -> 128:
  reduce semantic action-path activation length while preserving short CALVIN
  prompts and image/action path structure for a resource gate.

window_activation_checkpointing false -> true:
  trade compute for activation memory without changing the objective.
```

Decision:

```text
If G41 reaches step50/100 and action matches G40/G26 bands:
  PICF runtime itself can coexist with native action training; next branch may
  add bounded PICF context or belief losses one at a time.

If G41 OOMs:
  2xA100-40GB is insufficient for PICF runtime + native action boundary under
  K4; do not reduce K below K4 and claim success.  Use a larger machine or
  implement a structural memory reduction that preserves the action contract.

If G41 runs but action diverges:
  inspect whether PICF runtime changed sampling/window validation or semantic
  action inputs despite picf_action_condition_enabled=false.
```

Local code-path recheck, 2026-06-05:

```text
Command:
  uv run pytest -q scripts/picf_core_train_test.py \
    -k 'pi0_trainable_scope_backbone_only or policy_only_trainable_scope or \
        logical_batch_loss_scales or bucket_sequence_without_replacement or \
        calvin_bucket_sampling_weights or dynamic_bucket_sampling_weights or \
        gradient_surgery_params or pcgrad or cagrad'

Result:
  13 passed, 140 deselected, 1 warning
```

This verifies that the trainable-scope, task-balanced logical batch,
bucket-weighting, dynamic-mixing, and scoped gradient-surgery code paths still
compile and satisfy their local contracts while G40 is running remotely.

Allowed next branches after G40:

```text
If G40 step100 passes:
  Close:
    sampler-only reruns
    temperature-only reruns
    explicit-ratio-only reruns
    dynamic-mixing-only reruns
    PCGrad/CAGrad-only reruns
    action-LR-only reruns
    optimizer-reset-only reruns

  Continue only:
    1. PICF-enabled native-action resource gate
       - same backbone_only native action boundary
       - no direct PICF action condition
       - reduce memory cost without changing the mathematical contract
       - examples: shorter semantic length, checkpointing, fewer PICF runtime
         tokens, or larger GPU count

    2. PICF/action boundary redesign
       - keep dense semantic/action path native
       - inject PICF as controlled context, not as an always-on action
         optimizer target
       - maintain task-balanced logical batch and per-bucket normalization

If G40 step100 fails:
  Stop PICF work and debug current runtime parity first.
```

Reason:

```text
G40 step50 has already matched G26 within 0.03% relative error.  Repeating the
data-mixing/gradient-surgery family after that would not be a new scientific
test.  It would only remeasure a branch whose code path and positive control
are both already established.
```

G41 launch status, 2026-06-05 12:56 CST:

```text
remote:
  px-cloud1:26120

tmux:
  picf_g41_picf_runtime_native_action_boundary_120_20260605

log:
  /mnt/picf_run_logs/picf_g41_picf_runtime_native_action_boundary_120_20260605.log

tail:
  ssh -i /tmp/picf_g22_key -o StrictHostKeyChecking=no -p 26120 \
    root@px-cloud1.matpool.com \
    'tail -f /mnt/picf_run_logs/picf_g41_picf_runtime_native_action_boundary_120_20260605.log'
```

Initial contract observed in remote log:

```text
logical_batch_task_count = 4
calvin_bucket_sampling_mode = task_uniform
logical_batch_bucket_normalization = true
calvin_bucket_sample_without_replacement = true
semantic_trainable_scope = backbone_only
picf_action_condition_enabled = false
action_context_tokens = 0
semantic_max_length = 128
window_activation_checkpointing = true
semantic_gradient_checkpointing_disabled_for_accum = true
visual_mode = encoder
tactile_mode = encoder
```

Interpretation:

```text
This is a real PICF-runtime resource/boundary gate, not a G40 stub replay.
It keeps native action capacity and K4 task-balanced logical batching while
allowing V-JEPA/AnyTouch encoder runtime costs.  The first decision is whether
it passes the early step1-step3 OOM boundary that rejected G38/G39.
```

G41 result, 2026-06-05 12:59 CST:

```text
status:
  failed before first optimizer step

failure:
  RuntimeError: Non-finite gradients detected before optimizer step.

resource:
  not OOM

nonfinite gradient groups:
  policy_head
  picf_core
  semantic_backbone

recent rank0 windows:
  other: move the door all the way to the left and let go
  switch_button_light: push the button to turn on the led light

recent rank1 windows:
  block_push: push the red block towards the right
  slider: go slide the blue block to the left
```

Corrected interpretation:

```text
G41 is invalid as a pure action-boundary gate.  It disabled slot-JEPA,
support-pred, binding-consistency, denoising, and slot-quality, but it did not
zero every remaining structure loss inherited from the G26-B args.

Still active in the observed startup contract:
  lambda_anchor_object_pull = 0.35
  lambda_mapg_cycle = 0.01
  lambda_mapg_support_diversity = 0.005
  object_explanation losses > 0
  lambda_action_prefix_trust = 0.02

Therefore G41 cannot prove that PICF runtime itself makes native action
training numerically unstable.  It proves that the partially-disabled
PICF-runtime + structure-loss mix is not numerically safe at step1.
```

Next non-duplicate gate: G42.

```text
G42 keeps:
  picf_mode = enabled
  visual_mode = encoder
  tactile_mode = encoder
  semantic_trainable_scope = backbone_only
  task_uniform K4 logical batch
  per-bucket loss normalization
  semantic_max_length = 128
  window_activation_checkpointing = true

G42 changes:
  detach_action_loss_from_picf = true
  picf_action_condition_enabled = false
  action_context_tokens = 0
  every non-action lambda = 0
  lambda_action_prefix_trust = 0

Validity invariant:
  loss_total_minus_action == 0

Question:
  Does PICF runtime forward coexist with the native action-capacity boundary
  when it is not allowed to contribute any training gradient?
```

Decision:

```text
If G42 passes step50/100 near G40:
  the remaining problem is not sampler or action capacity; it is structure-loss
  gradient coupling / PICF loss numerics.  Reintroduce one bounded structure
  family at a time only after finite-gradient contracts are added.

If G42 still produces non-finite gradients:
  action path or semantic input differs from G40 despite disabled PICF action
  condition.  Inspect semantic_override and action flow numerics directly.

If G42 OOMs:
  resource boundary remains unresolved for PICF runtime + native action
  boundary on 2x40GB even without structure gradients.
```

G42 result, 2026-06-05:

```text
status:
  failed before first optimizer step

resource:
  not OOM

verified launch invariants:
  detach_action_loss_from_picf = true
  lambda_anchor_object_pull = 0
  lambda_action_prefix_trust = 0
  non-action auxiliary lambdas explicitly set to 0

failure:
  RuntimeError: Non-finite gradients detected before optimizer step.

nonfinite gradient groups:
  policy_head
  picf_core
  semantic_backbone

metrics:
  no step metrics written; failure happened before the first optimizer step
```

Strict interpretation:

```text
This is no longer a sampler / LR / optimizer-reset question.

G40 already proves:
  current launcher + task-uniform K4 logical batch + per-bucket normalization
  + backbone_only native action boundary can reproduce the known healthy
  PI0.5-like action descent at step50 and step100.

G42 proves:
  enabling PICF runtime with encoder paths creates a numerical/graph failure
  before step1 even when direct PICF action conditioning is off, action loss is
  detached from PICF, and non-action losses are zeroed.

Therefore the next valid branch is not another blind training run.  It is a
forward/backward non-finite source isolation gate.
```

Non-duplicate diagnostic patch added locally:

```text
scripts/picf_core_train.py
  _collect_nonfinite_output_diagnostics(outputs)
  OPENPI_DEBUG_NONFINITE_OUTPUTS=1
  finite(loss_total) check before backward

Purpose:
  distinguish three cases:
    A. forward outputs already contain NaN/Inf;
    B. loss_total is finite but backward derivative is NaN/Inf;
    C. outputs/loss/gradients are finite after disabling a specific runtime
       branch, identifying that branch as the source.
```

Local verification:

```text
uv run python -m py_compile scripts/picf_core_train.py

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'pi0_trainable_scope_backbone_only or policy_only_trainable_scope or \
      logical_batch_loss_scales or bucket_sequence_without_replacement or \
      calvin_bucket_sampling_weights or dynamic_bucket_sampling_weights or \
      gradient_surgery_params or pcgrad or cagrad'

result:
  13 passed, 140 deselected
```

Immediate gate order:

```text
G42d:
  rerun G42 with:
    OPENPI_DEBUG_NONFINITE_OUTPUTS=1
    OPENPI_DEBUG_AUTOGRAD_ANOMALY=1
    OPENPI_DEBUG_PHASE_LIMIT=1

If forward outputs are non-finite:
  inspect the named output and its originating branch.

If forward outputs are finite but anomaly names a backward op:
  isolate that op by disabling one runtime branch at a time.

If G42d is too slow with anomaly:
  cap to num_train_steps=2 and stop after first failure.
```

G42d result, 2026-06-05:

```text
status:
  failed during first backward

forward:
  completed; no forward-output finite check failure was raised

backward anomaly:
  RuntimeError: Function 'SqrtBackward0' returned nan values in its 0th output.

source:
  src/openpi/picf/core/pipeline.py
  PicfFullCore._measurement_innovation_norm

old expression:
  return torch.sqrt(torch.clamp(nearest, min=0.0))

call path:
  PicfPi05Policy.forward_train_transition
  -> PicfFullCore.observe_step
  -> _posterior_update
  -> _measurement_innovation_norm
```

Mathematical root:

```text
nearest is a non-negative squared Mahalanobis distance.

The old expression:
  d = sqrt(clamp(nearest, min=0))

is forward-valid but not backward-safe.  When nearest == 0, which is common
when a recycled posterior prior exactly matches a current observation anchor,
the derivative of sqrt(x) is:

  d/dx sqrt(x) = 1 / (2 sqrt(x))

At x = 0 this derivative is singular.  Clamp-to-zero does not remove that
singularity; it creates exactly the zero input to sqrt.  This explains why
G42/G42d failed before step1 even with action-only, detach-action-from-PICF,
and all non-action losses set to zero.
```

Fix:

```text
src/openpi/picf/core/pipeline.py

old:
  return torch.sqrt(torch.clamp(nearest, min=0.0))

new:
  return torch.sqrt(torch.clamp(nearest, min=self.config.epsilon_s))
```

Why this is not a cosmetic patch:

```text
This is the standard differentiable safe-norm form.  It preserves the semantic
meaning of "near-zero innovation" while preventing an infinite derivative at
exact matches.  The same design pattern already exists elsewhere in the code
for overlap denominators and Mahalanobis distances:

  sqrt(clamp(..., min=eps))
  sqrt(d2 + eps)
```

Local regression:

```text
uv run pytest -q src/openpi/picf/core/pipeline_test.py \
  -k 'measurement_innovation_norm_exact_match_has_finite_backward'

result:
  1 passed, 103 deselected

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'pi0_trainable_scope_backbone_only or policy_only_trainable_scope or \
      logical_batch_loss_scales or bucket_sequence_without_replacement or \
      calvin_bucket_sampling_weights or dynamic_bucket_sampling_weights or \
      gradient_surgery_params or pcgrad or cagrad'

result:
  13 passed, 140 deselected

uv run python -m py_compile \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/pipeline_test.py \
  scripts/picf_core_train.py

result:
  pass
```

Next gate: G42e.

```text
Rerun the exact G42 action-only-detached PICF-runtime gate after the safe-norm
fix.  Expected result:

  no SqrtBackward0 failure;
  if a new non-finite appears, anomaly should identify the next source;
  if step50/100 pass, action descent can be tested under the PICF runtime
  boundary instead of discussing sampler/LR again.
```

#### G42e result: safe innovation norm removes the original non-finite blocker

Run:

```text
picf_g42e_safe_innovation_norm_debug_2_20260605
```

Configuration:

```text
Same action-only-detached PICF-runtime gate as G42/G42d.
All non-action loss weights zero in total loss.
action_context_tokens=0.
detach_action_loss_from_picf=true.
OPENPI_DEBUG_AUTOGRAD_ANOMALY=1.
OPENPI_DEBUG_NONFINITE_OUTPUTS=1.
num_train_steps=2.
```

Observed result:

```text
step 1 completed successfully.
No SqrtBackward0 failure.
No non-finite output diagnostic fired.
loss_total_minus_action=0.0.
loss_total=0.2740724980831146.
loss_action_default_equiv=0.1370362490415573.
loss_action_weight_scale=2.0.
pi_action_condition_token_count=0.
```

Step 2 failed with CUDA OOM during backward inside Gemma MLP FSDP:

```text
torch.OutOfMemoryError:
  tried to allocate 64 MiB;
  39.46 GiB already in use on a 40 GiB GPU.
```

Interpretation:

```text
This is not the old numerical failure.  The original failure happened before
step 1 with:

  SqrtBackward0 returned nan values

after _measurement_innovation_norm.  G42e removes that blocker.

The G42e OOM is attributable to anomaly/debug overhead on an already tight
40GB two-rank run.  Therefore it is invalid to call the safe-norm fix
insufficient from G42e alone.  The next required gate is the same runtime
boundary without anomaly/debug.
```

Next gate: G42f.

```text
Run action-only-detached PICF runtime for 120 normal steps without anomaly
debug instrumentation.

Expected:
  no non-finite gradient failure;
  no FSDP OOM;
  step50 and step100 metrics available;
  if action still does not descend, the issue is real action/PICF coupling,
  not the already fixed sqrt singularity.
```

#### G42f result: normal mode removes NaN but 2x40GB cannot hold this trainable boundary

Run:

```text
picf_g42f_safe_innovation_norm_actiononly_120_20260605
```

Configuration:

```text
Same as G42e, but without anomaly/debug env.
num_train_steps=120.
log_interval=50.
PICF runtime enabled.
action condition disabled.
detach_action_loss_from_picf=true.
All non-action loss weights are zero in loss_total.
semantic_trainable=true, semantic_trainable_scope=backbone_only.
picf_trainable_scope=all.
training_strategy=fsdp_full_shard.
optimizer_sharding=none.
```

Observed:

```text
step 1 completed.
loss_total=0.3411.
No SqrtBackward0 / non-finite failure.

step 2 backward OOMed on both ranks:
  rank0 tried to allocate 20 MiB with only 7.56 MiB free;
  rank1 tried to allocate 64 MiB with only 57.56 MiB free.
```

Root interpretation:

```text
G42f confirms the numerical root found in G42d is fixed.

The remaining G42f failure is a memory boundary:
  PaliGemma backbone trainable
  + policy head trainable
  + PICF core trainable
  + PICF runtime activations
  + Adam/FSDP optimizer state after step 1
  > 2xA100-40GB capacity.

This is consistent with the timing of the failure.  The run survives the first
optimizer step and fails on step 2 after optimizer state has been materialized.

It is therefore invalid to use G42f to claim action optimization still fails.
G42f did not reach an action-descent window; it identified an infeasible
2x40GB trainable boundary.
```

Next memory-valid gate: G43.

```text
Run the same PICF-runtime/action-only-detached boundary with:

  picf_trainable_scope=policy_only

This freezes core.* parameters while keeping the native PaliGemma/action policy
path trainable.  It answers a narrower but necessary question:

  Can the PICF runtime exist in the forward graph without breaking native
  action optimization when its core parameters do not create optimizer/grad
  memory pressure?

If G43 runs to step50/100 and action descends, then the immediate production
direction is:
  train policy/action path with PICF runtime frozen on 2x40GB;
  train PICF core only on larger-memory hardware or with a redesigned lower
  memory boundary.

If G43 also fails numerically, the next blocker is still inside the runtime
forward/backward coupling, not optimizer state.
```

Live G43 status:

```text
picf_g43_policyonly_picf_runtime_actiononly_120_20260605

Initial evidence:
  step 1 passed;
  step 2 passed;
  step 3 passed;
  step 11 passed;
  step 22 passed;
  no non-finite gradient failure;
  no step2 optimizer-state OOM.

Memory:
  about 32.5-38.5 GiB per 40GB A100 after entering training.

Speed:
  about 51-55 sec/step on 2xA100-40GB.

Important interpretation:
  This is not yet a PICF-helping-action experiment because action_context_tokens=0
  and picf_action_condition_enabled=false.  It is the required numerical/memory
  gate proving that the safe-norm PICF runtime can coexist with policy training
  once core.* parameters are frozen.

Early loss note:
  progress-bar loss is high-variance across task-uniform windows before step50
  (observed approximate range: 0.1157 to 0.5519 by step22).  This is not yet a
  descent verdict; the first comparable metrics are the step50 JSON losses.
```

Do not repeat:

```text
Do not re-run full picf_trainable_scope=all on 2xA100-40GB with PaliGemma
backbone trainable unless the memory boundary is changed.  G42f already showed
that this boundary OOMs immediately after the first optimizer state is
materialized.
```

## 2026-06-05 G43 Gradient-Gate Correction

Run checked:

```text
picf_g43_policyonly_picf_runtime_actiononly_120_20260605
```

Configuration:

```text
PICF runtime enabled.
picf_trainable_scope=policy_only.
semantic_trainable=true, semantic_trainable_scope=backbone_only.
picf_action_condition_enabled=false.
action_context_tokens=0.
detach_action_loss_from_picf=true.
all non-action loss weights zero in loss_total.
```

Observed at step 50:

```text
loss_total=0.3174545765
loss_action_default_equiv=0.1587272882
loss_total_minus_action=0.0
pi_action_condition_token_count=0
pi_context_token_count=0
preclip_grad_norm=0.0
grad_norm=0.0
grad_norm_group_semantic_backbone=0.0
grad_absmax_group_semantic_backbone=0.0
```

Control comparison:

```text
G40 step50:
  loss_action_default_equiv=0.1292344034
  preclip_grad_norm=1.7212671041
  grad_norm_group_semantic_backbone=1.1792682649

G40 step100:
  loss_action_default_equiv=0.0771813095
  preclip_grad_norm=0.8564581275
  grad_norm_group_semantic_backbone=0.6095748374
```

Conclusion:

```text
G43 is not a valid learning run.
The zero-gradient signal is real, not a logging artifact.
```

Root cause:

```text
src/openpi/picf/core/training.py::compute_transition_loss detached
action_loss_override whenever detach_action_loss_from_picf=true.

That implementation cut gradients to the entire PI0.5/PaliGemma action loss,
not only to PICF.  The intended isolation is:

  detach PICF prefix/context tokens before they enter PI0.5 action generation;
  keep the native PI0.5 action scalar live so it can train policy parameters.
```

Fix:

```text
Keep action_loss_override live in compute_transition_loss.
Keep PICF isolation at action_prefix_stopgrad/action_context_stopgrad and at
the configured PICF trainable scope.
```

Validation:

```text
Local:
  uv run python -m py_compile
    src/openpi/picf/core/training.py
    src/openpi/picf/core/training_test.py
    scripts/picf_core_train.py
    src/openpi/picf/core/pipeline.py

  uv run pytest -q src/openpi/picf/core/training_test.py -k
    'action_override_detach_picf_keeps_override_gradient or
     action_override_preserves_default_parity or
     action_override_respects_zero_action_lambdas'
  result: 3 passed

  uv run pytest -q src/openpi/picf/core/pipeline_test.py -k
    measurement_innovation_norm_exact_match_has_finite_backward
  result: 1 passed

Remote:
  py_compile passed.
  training action-override tests: 3 passed.
  measurement innovation safe-norm test: 1 passed.
```

Repeated local verification after G44 step50, 2026-06-05:

```text
uv run python -m py_compile
  scripts/picf_core_train.py
  src/openpi/picf/core/training.py
  src/openpi/picf/core/training_test.py
  src/openpi/picf/core/pipeline.py
  src/openpi/picf/core/pipeline_test.py

result: pass

uv run pytest -q src/openpi/picf/core/training_test.py -k
  'action_override_detach_picf_keeps_override_gradient or
   action_override_preserves_default_parity or
   action_override_respects_zero_action_lambdas'

result: 3 passed

uv run pytest -q src/openpi/picf/core/pipeline_test.py -k
  measurement_innovation_norm_exact_match_has_finite_backward

result: 1 passed

uv run pytest -q scripts/picf_core_train_test.py -k
  'calvin_bucket_sampling_weights or bucket_sequence_without_replacement or
   bucket_sequence_round_robin or logical_batch_loss_scales or
   dynamic_bucket_sampling_weights or gradient_surgery_params or pcgrad or
   cagrad'

result: 12 passed
```

Interpretation:

```text
The current fixes and the VLA-method infrastructure remain locally testable.
The experiment should therefore continue along the action/PICF boundary branch,
not restart already-tested sampler or gradient-surgery branches.
```

Active follow-up:

```text
picf_g43b_fixed_action_override_grad_policyonly_120_20260605

Purpose:
  same memory-valid boundary as G43, but with action override gradient restored.

Required gate:
  step25/50 must show preclip_grad_norm > 0 and semantic_backbone grad > 0.
  If this fails, stop immediately and inspect the PI0.5 action graph.

Interpretation limit:
  This run still has picf_action_condition_enabled=false and action_context_tokens=0.
  It only tests whether PICF runtime can coexist with native action learning.
  It does not test whether PICF context helps action.
```

G43b step25 result:

```text
step=25
loss_total=0.2912192941
loss_action_default_equiv=0.1456096470
loss_total_minus_action=0.0
preclip_grad_norm=3.8307754993
grad_norm=3.8307754993
grad_norm_group_semantic_backbone=2.7212838634
grad_absmax_group_semantic_backbone=0.18359375
pi_action_condition_token_count=0
pi_context_token_count=0
lr=8.333333333e-06
```

Interpretation:

```text
The G43 zero-gradient failure is fixed at the first measurable gate.
The native PI0.5/PaliGemma action scalar now produces non-zero gradients under
detach_action_loss_from_picf=true.  This validates the action-override detach
fix in the real remote training graph, not only in unit tests.

Continue to step50 before judging action-loss descent.  Step25 is a gradient
restoration verdict, not a convergence verdict.
```

G43b step50 result:

```text
step=50
loss_total=0.2082970738
loss_action_default_equiv=0.1041485369
loss_total_minus_action=0.0
preclip_grad_norm=1.6823481321
grad_norm=1.6823481321
grad_norm_group_semantic_backbone=1.1781995386
grad_absmax_group_semantic_backbone=0.0869140625
pi_action_condition_token_count=0
pi_context_token_count=0
lr=1.666666667e-05
```

Comparison:

```text
G43 broken step50:
  action_default = 0.1587272882
  preclip_grad_norm = 0.0
  semantic_backbone_grad = 0.0

G43b fixed step50:
  action_default = 0.1041485369
  preclip_grad_norm = 1.6823481321
  semantic_backbone_grad = 1.1781995386

G40 native positive control step50:
  action_default = 0.1292344034
  preclip_grad_norm = 1.7212671041
  semantic_backbone_grad = 1.1792682649
```

Decision:

```text
G43b passes the memory-valid action-boundary gate.

The zero-gradient failure in G43 was caused by the action-override detach bug
and is fixed.  The safe innovation-norm fix also remains valid because G43b
reaches step50 without the earlier SqrtBackward0 failure.

Do not continue G43b as a long run.  It has picf_action_condition_enabled=false
and action_context_tokens=0, so it cannot answer whether PICF context helps
action.  Stop G43b after recording step50 and launch G44.
```

Immediate decision tree:

```text
If G43b step25/50 shows:
  preclip_grad_norm > 0
  grad_norm_group_semantic_backbone > 0
  no NaN / no OOM

then:
  the G43 zero-gradient failure is resolved by the action-override detach fix.
  Proceed to G44, the first valid PICF-context action gate.

If G43b still shows zero gradients:
  stop immediately.
  The remaining root is not sampler/LR/optimizer/PICF-core loss.
  Inspect the native PI0.5 action graph and optimizer trainable scope.

If G43b OOMs:
  do not interpret it as action-learning failure.
  It is a memory-boundary result under policy_only + PICF runtime.
```

Non-repetition rule:

```text
Do not re-run these as standalone "action descent fixes" unless new evidence
changes their premise:

  task-uniform logical batch
  K4/K8/K12 gradient accumulation alone
  per-bucket normalization alone
  temperature / ratio / dynamic mixing alone
  optimizer reset alone
  LR-only branches
  scoped PCGrad / CAGrad without a fresh gradient-cosine trigger
  raw-overlap / SAM / sidecar-only changes

Those families are already represented in G24/G26/G30-G40.  G40 proves the
current launcher/data/sampler/native-action path is capable of healthy action
descent.  The current root track is action/PICF boundary correctness.
```

Full method checklist against recent VLA practice:

| Remedy family | Current status | Evidence / decision |
| --- | --- | --- |
| probabilistic task/dataset mixing | implemented as CALVIN bucket/task sampler | keep mandatory; not sufficient alone |
| stratified logical batch | implemented as K-task logical batch with `logical_batch_task_count` | G40 proves K4 native action can descend; K8/K12/hardware branches alone did not solve PICF action |
| gradient accumulation as logical batch | implemented through `accum_steps` and logged effective batch | necessary for coverage; not a standalone cure |
| task-uniform sampling | implemented and default for current PICF gates | keep; matches the ABot-M0-style task-coverage argument |
| temperature / ratio sampling | implemented and tested in prior G-series branches | not default; no evidence it beats task-uniform on current CALVIN action gates |
| dynamic mixing | implemented and tested as a diagnostic | not default; no fresh gradient-cosine trigger justifying it now |
| per-bucket/per-task normalization | implemented as logical-batch bucket normalization | keep; prevents large buckets from dominating |
| per-modality adapters/projectors | structurally present in PICF/PaliGemma/Sonata/AnyTouch/V-JEPA connector stack | do not re-architect until action boundary is proven healthy |
| continuous/chunked action | PI0.5 path already uses flow action over action chunks; canonical metric is `loss_action_default_equiv` | keep for comparability; do not switch loss family before resolving boundary bugs |
| L1/Huber action head | considered from OpenVLA-OFT-style recipes | future ablation only after native/PICF boundary is stable |
| action expert / VLM boundary | active root track | G43 exposed detach bug; G43b validates the fix; G44 will test bounded context |
| stop-gradient / knowledge insulation | partially deployed through prefix/context stop-grad and trainable scope | must be validated by G43b/G44 before longer runs |
| PCGrad/CAGrad | code paths/tests exist | do not default; use only after a fresh gradient-cosine audit proves destructive conflict |
| action expert MoE/router | proxy exists and was tested as not sufficient | do not default; only revisit after context boundary gives useful action signal |

Non-repeat rule after G44 step75:

```text
Do not spend another 1-2h branch on these as standalone fixes unless a new code
change invalidates their previous evidence:
  - random vs task_uniform vs temperature vs explicit ratio sampling;
  - K4 logical-batch coverage and per-bucket loss normalization;
  - dynamic PiKE-style bucket mixing;
  - scoped PCGrad/CAGrad;
  - optimizer-reset-only or LR-only explanations;
  - Huber/L1/action-objective-only swaps;
  - SAM/sidecar-as-primary action fixes;
  - raw-overlap-only anchor penalties.

Reason:
  G40/G33 prove the current launcher/data/sampler/native action path descends.
  G35/G37 prove the narrow action_head_and_adapter boundary is bad even without
  PICF.  G42/G43 exposed concrete code blockers.  G43b/G44 now test the only
  remaining non-repeated causal boundary: bounded PICF context consumed by the
  native PI0.5/PaliGemma action path.
```

Allowed next experiments after G44:

```text
If G44 step100 passes:
  1. Run a 300/1000-step bounded-context gate from the same launcher.
  2. Optionally sweep only context gate/token count:
       gate=0.10 vs 0.25
       context_tokens=2 vs 4
     Keep sampler/LR/objective unchanged.
  3. Only after the 300/1000 gate passes, consider a 30k run.

If G44 step100 fails:
  1. Stop G44.
  2. Reduce PICF context bandwidth first:
       context_tokens=2
       context_gate=0.10
       include_query_tokens=false
       stopgrad=true
  3. Do not re-open structure losses, PCGrad, dynamic mixing, or SAM.

If G44 is numerically unstable:
  1. Treat it as a runtime bug, not an optimization result.
  2. Inspect finite diagnostics and the last stack trace before launching a new
     experiment.
```
| System-2/planning head | not deployed for CALVIN action descent | not first-order for current short-horizon CALVIN scalar; future large-data extension |

Current root-cause status:

```text
The old broad hypothesis "small batch lacks task coverage" is incomplete.
G40 proves current task-uniform K4 + native action can descend.

The sharper current hypothesis is:
  action/PICF integration had code-level blockers and boundary errors.

Confirmed blockers:
  1. _measurement_innovation_norm used sqrt(0), causing NaN backward.
  2. detach_action_loss_from_picf detached the entire native action scalar.

Active validation:
  G43b verifies whether blocker (2) is fixed in remote training.
```

Planned next gate if G43b passes:

```text
G44: PICF runtime + frozen PICF core + gated stop-grad context into native
PI0.5 action.

Initial conservative contract:
  picf_trainable_scope = policy_only
  semantic_trainable = true
  semantic_trainable_scope = backbone_only
  picf_action_condition_enabled = true
  action_context_tokens = 4 or 8
  action_context_stopgrad = true
  action_prefix_stopgrad = true
  semantic_action_context_readout_aux_weight = 0
  semantic_action_context_token_aux_weight = 0
  semantic_action_context_flow_residual_enabled = false
  all PICF structural loss weights = 0
  K4 task-uniform logical batch remains on

Question:
  can bounded PICF context be consumed by the native action path without
  breaking the gradient/loss behavior that G40 proved healthy?
```

G44 launch, 2026-06-05 15:47 CST:

```text
picf_g44_context_stopgrad_policyonly_120_20260605

Confirmed startup contract:
  world_size = 2
  accum_steps = 2
  effective_global_batch = 4
  picf_mode = enabled
  picf_trainable_scope = policy_only
  semantic_trainable = true
  semantic_trainable_scope = backbone_only
  prefix_stopgrad = true
  prefix_gate = 0.0
  context_tokens = 4
  context_integration = suffix_cross_attention
  context_stopgrad = true
  context_norm = rmsnorm
  context_gate = 0.25
  context_include_queries = false
  all non-action structure-loss weights = 0
  train_loss = mse
  canonical action metric = loss_action_default_equiv
```

Rationale:

```text
This is the first valid PICF-context action gate after fixing:
  1. the sqrt(0) innovation backward singularity;
  2. the action-override detach bug.

It deliberately does not train PICF core or structural losses.  The only new
dataflow relative to G43b is bounded stop-grad PICF context entering the native
PI0.5 action-side adapter.  This follows the knowledge-insulation principle:
semantic/action policy can consume context, but the noisy action scalar cannot
rewrite PICF core through this gate.
```

Early startup:

```text
Reached step5 without OOM or NaN.
First comparable JSON gate remains step25.
```

G44 step25 result:

```text
step=25
loss_action_default_equiv=0.1453970224
preclip_grad_norm=3.8745079041
grad_norm_group_semantic_backbone=2.7629773737
pi_action_condition_token_count=8
pi_context_token_count=4
pi_context_adapter_token_count=8
lr=8.333333333e-06
```

Comparison:

```text
G43b step25 without context:
  action_default = 0.1456096470
  preclip_grad_norm = 3.8307754993
  semantic_backbone_grad = 2.7212838634

G44 step25 with bounded stop-grad context:
  action_default = 0.1453970224
  preclip_grad_norm = 3.8745079041
  semantic_backbone_grad = 2.7629773737
```

Interpretation:

```text
The bounded stop-grad context path is live and does not break the first action
gradient/loss gate.  The numbers are almost identical to G43b at step25, which
is the desired first-order behavior: context can be introduced without
destroying native PI0.5 action optimization.

Continue to step50 before judging whether context helps, hurts, or is neutral.
```

G44 step50 result:

```text
step=50
loss_total=0.2073915601
loss_action_default_equiv=0.1036957800
loss_total_minus_action=0.0
loss_action_active7=0.4074040055
loss_action_pos=0.3203063011
loss_action_rot=0.3899260759
loss_action_gripper=0.7211307287
preclip_grad_norm=1.6651126146
grad_norm_group_semantic_backbone=1.1661199081
grad_absmax_group_semantic_backbone=0.0932617188
logical_batch_distinct_bucket_count=4
pi_action_condition_token_count=8
pi_context_token_count=4
pi_context_gate=0.25
pi_context_adapter_token_count=8
pi_context_adapter_gate=0.1192153022
lr=1.666666667e-05
steps_per_sec=0.0180840169
```

Comparison:

```text
G43b step50 without PICF context:
  action_default = 0.1041485369
  preclip_grad_norm = 1.6823481321
  semantic_backbone_grad = 1.1781995386

G44 step50 with bounded stop-grad PICF context:
  action_default = 0.1036957800
  preclip_grad_norm = 1.6651126146
  semantic_backbone_grad = 1.1661199081

G40 native positive control step50:
  action_default = 0.1292344034
```

Interpretation:

```text
G44 passes the first PICF-context action gate.

The context path is live, bounded, stop-grad, and action-only:
  loss_total_minus_action == 0
  pi_context_token_count == 4
  pi_context_gate == 0.25

It does not reproduce the G35/G37 bad band (~0.157) and does not show the
previous zero-gradient failure.  At step50 it is slightly better than G43b
without context and materially better than the G40 native step50 positive
control.  This is not yet proof that PICF context improves action in the long
run, but it does prove that the safe-norm and action-override fixes make the
bounded context path numerically/trainably valid.

Continue to step100.  If step100 remains close to or better than G43b/G40
healthy bands, the next branch can be a longer bounded-context run or a small
context-gate sweep.  If step100 regresses toward the G35/G37 bad band, stop and
reduce context gate/token count before any longer run.
```

G44 step75 result:

```text
step=75
loss_total=0.1588240713
loss_action_default_equiv=0.0794120356
loss_total_minus_action=0.0
loss_action_active7=0.3591422439
loss_action_pos=0.2891203165
loss_action_rot=0.3858025074
loss_action_gripper=0.4892269969
preclip_grad_norm=1.6092088223
grad_norm_group_semantic_backbone=1.1389422622
logical_batch_distinct_bucket_count=4
logical_batch_global_micro_count=4
pi_context_token_count=4
pi_context_gate=0.25
pi_context_adapter_token_count=8
pi_context_adapter_gate=0.1192293093
lr=2.5e-05
steps_per_sec=0.0177948682
```

Trend:

```text
G44 action_default:
  step25 = 0.1453970224
  step50 = 0.1036957800
  step75 = 0.0794120356

G44 total_minus_action:
  step25 = 0.0
  step50 = 0.0
  step75 = 0.0
```

Interpretation:

```text
G44 passes the mid-run PICF-context action gate.

The bounded context path remains action-only and numerically stable, and action
continues descending rather than returning to the G35/G37 bad band.  This
further rules out the immediate hypotheses:
  - bounded stop-grad PICF context destroys native action gradients;
  - the safe innovation-norm fix only survives the first logged window;
  - the restored action override gradient is a step25/50 artifact.

It is still not a 30k proof.  Continue to step100 as the hard gate.
```

Repeated local verification after G44 step75, 2026-06-05:

```bash
uv run python -m py_compile \
  scripts/picf_core_train.py \
  src/openpi/picf/core/training.py \
  src/openpi/picf/core/training_test.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/pipeline_test.py

uv run pytest -q src/openpi/picf/core/training_test.py \
  -k 'action_override_detach_picf_keeps_override_gradient or action_override_preserves_default_parity or action_override_respects_zero_action_lambdas'

uv run pytest -q src/openpi/picf/core/pipeline_test.py \
  -k measurement_innovation_norm_exact_match_has_finite_backward

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'calvin_bucket_sampling_weights or bucket_sequence_without_replacement or
      bucket_sequence_round_robin or logical_batch_loss_scales or
      dynamic_bucket_sampling_weights or gradient_surgery_params or pcgrad or
      cagrad'
```

Result:

```text
py_compile: passed
training override tests: 3 passed
safe innovation-norm backward test: 1 passed
logical-batch/dynamic/PCGrad/CAGrad tests: 12 passed
```

Interpretation:

```text
The current G44 decision is not blocked by the known local contract tests.
The tested code paths include:
  - action override keeps native PI0.5/PaliGemma action gradients live;
  - detach_action_loss_from_picf isolates PICF/predictive action without
    detaching the native override scalar;
  - exact-match innovation norm has finite backward;
  - task-balanced logical-batch sampling, per-bucket loss scaling, dynamic
    mixing, and scoped PCGrad/CAGrad remain contract-valid.
```

G44 step100 result:

```text
step=100
loss_total=0.1538133919
loss_action_default_equiv=0.0769066960
loss_total_minus_action=0.0
loss_action_active7=0.3494527340
loss_action_pos=0.2838588953
loss_action_rot=0.3927500844
loss_action_gripper=0.4163419604
preclip_grad_norm=1.0683760643
grad_norm_group_semantic_backbone=0.7584522445
grad_absmax_group_semantic_backbone=0.0432128906
logical_batch_distinct_bucket_count=4
logical_batch_global_micro_count=4
pi_action_condition_token_count=8
pi_context_token_count=4
pi_context_gate=0.25
pi_context_adapter_token_count=8
pi_context_adapter_gate=0.1192151532
lr=3.333333333e-05
steps_per_sec=0.0176371350
```

G44 step25/50/75/100 trend:

```text
loss_action_default_equiv:
  0.1453970224 -> 0.1036957800 -> 0.0794120356 -> 0.0769066960

loss_total:
  0.2907940447 -> 0.2073915601 -> 0.1588240713 -> 0.1538133919

preclip_grad_norm:
  3.8745079041 -> 1.6651126146 -> 1.6092088223 -> 1.0683760643

loss_total_minus_action:
  0.0 -> 0.0 -> 0.0 -> 0.0
```

Result:

```text
G44 PASSED the bounded PICF-context action gate.

It shows all required properties:
  - action decreases through step100;
  - total loss is action-only in this gate;
  - native PaliGemma/PI0.5 action gradients remain live;
  - PICF context is present but bounded and stop-grad;
  - no OOM, SqrtBackward, Traceback, or non-finite loss.
```

Interpretation:

```text
This resolves the immediate action-boundary blocker that remained after the
large sampler/mixing/loss-normalization experiment set.

The successful mechanism is not another optimizer reset or sampler change.  It
is:
  1. keep the native PI0.5/PaliGemma action expert as the trainable action path;
  2. keep PICF core/action auxiliary pressure isolated;
  3. inject only a small, normalized, stop-grad PICF context into the native
     action side;
  4. preserve task-uniform logical-batch K4 and per-bucket normalization.

G44 does not yet prove a 30k run.  It proves the 100-step validity of the first
safe PICF-context action dataflow after the two concrete code fixes.
```

Next experiment decision:

```text
Current speed is ~56.7s/step, so a 300-step gate would take roughly 4.7h and a
1000-step gate roughly 15.8h on the current 2x40GB setup.  For a 1-2h feedback
loop, use 120-step gates first.

Recommended next non-repeated gates:
  G45a: same as G44, context_gate=0.10, context_tokens=4.
  G45b: same as G44, context_gate=0.25, context_tokens=2.

Purpose:
  determine whether the context bandwidth can be reduced without losing the
  action descent seen in G44, before committing to any longer 300/1000/30k run.

Do not restart sampler/LR/PCGrad/dynamic-mixing/SAM/raw-overlap branches unless
G45 creates a new failure mode.
```

## 2026-06-05 G45a: Lower-Bandwidth PICF Context Gate

G45a launch:

```text
picf_g45a_context_gate010_policyonly_120_20260605
```

Change from G44:

```text
action_context_output_gate:
  G44  = 0.25
  G45a = 0.10
```

Held fixed:

```text
num_train_steps = 120
log_interval = 25
picf_mode = enabled
picf_trainable_scope = policy_only
semantic_trainable = true
semantic_trainable_scope = backbone_only
picf_action_condition_enabled = true
action_prefix_stopgrad = true
action_prefix_output_gate = 0.0
action_context_tokens = 4
action_context_integration = suffix_cross_attention
action_context_stopgrad = true
action_context_norm_mode = rmsnorm
action_context_include_query_tokens = false
all non-action/structure lambdas = 0
detach_action_loss_from_picf = true
semantic_max_length = 128
training_strategy = fsdp_full_shard
optimizer_sharding = none
window_activation_checkpointing = true
accum_steps = 2
calvin_bucket_sampling_mode = task_uniform
logical_batch_task_count = 4
logical_batch_bucket_normalization = true
calvin_bucket_sample_without_replacement = true
```

Question:

```text
Can a smaller stop-grad PICF context gate preserve the G44 action descent while
reducing risk that PICF context perturbs the native PI0.5 action expert?
```

Decision:

```text
If G45a step25/50/75/100 matches G44/G43b healthy bands:
  prefer gate=0.10 for the next 300/1000 gate because it is more conservative.

If G45a is materially worse than G44:
  keep G44's gate=0.25 and test context_tokens=2 separately.

If G45a is numerically unstable:
  inspect runtime; do not infer optimization failure.
```

Startup correction:

```text
First G45a launch failed before model construction:
  ModuleNotFoundError: No module named 'scripts'

Cause:
  the torchrun shell did not export the repository root as PYTHONPATH for child
  processes.

This is a launcher environment error, not an optimization or model result.
Relaunch with:
  PYTHONPATH=/root/openpi_g25_20260604:/root/openpi_g25_20260604/src
```

Startup verification:

```text
Remote:
  /root/openpi_g25_20260604

tmux:
  picf_g45a_context_gate010_policyonly_120_20260605

log:
  /mnt/picf_run_logs/picf_g45a_context_gate010_policyonly_120_20260605.log

Observed:
  reached progress step 4
  json_rows = 0 because log_interval=25
  GPU memory ~= 34-35GB per A100-40GB
  GPU utilization live

Bad signatures:
  CUDA out of memory: false
  SqrtBackward0: false
  Traceback: false
  Non-finite loss_total: false
```

Interpretation:

```text
G45a is running normally.  The two earlier failures were launcher PYTHONPATH
errors before training, not model/loss evidence.  The first comparable loss
gate remains step25.
```

Local contract verification during G45a startup:

```bash
uv run python -m py_compile \
  scripts/picf_core_train.py \
  src/openpi/picf/core/training.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/policy.py

uv run pytest -q src/openpi/picf/core/training_test.py \
  -k 'action_override_detach_picf_keeps_override_gradient or
      action_override_preserves_default_parity or
      action_override_respects_zero_action_lambdas'

uv run pytest -q src/openpi/picf/core/pipeline_test.py \
  -k measurement_innovation_norm_exact_match_has_finite_backward

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'calvin_bucket_sampling_weights or bucket_sequence_without_replacement or
      bucket_sequence_round_robin or logical_batch_loss_scales or
      dynamic_bucket_sampling_weights or gradient_surgery_params or pcgrad or
      cagrad'
```

Result:

```text
py_compile: passed
action override tests: 3 passed
safe innovation backward: 1 passed
logical-batch/dynamic/PCGrad/CAGrad: 12 passed
policy action-context tests: 2 passed
PaliGemma action-context adapter tests: 7 passed
```

### G45a Step25 Gate

Observed remote row:

```text
step = 25
loss_total = 0.2912646234
loss_action_default_equiv = 0.1456323117
loss_total_minus_action = 0.0
loss_action_active7 = 0.5059391260
loss_action_pos = 0.4224258065
loss_action_rot = 0.4548758864
loss_action_gripper = 0.9096685648
preclip_grad_norm = 3.8519954681
grad_norm = 3.8519954681
grad_norm_group_semantic_backbone = 2.7413436879
grad_absmax_group_semantic_backbone = 0.1777343750
logical_batch_distinct_bucket_count = 4
logical_batch_global_micro_count = 4
pi_action_condition_token_count = 8.0
pi_context_token_count = 4.0
pi_context_gate = 0.1000000015
pi_context_adapter_token_count = 8.0
pi_context_adapter_gate = 0.1192039698
lr = 8.3333333333e-06
steps_per_sec = 0.0189089798
```

Bucket action losses:

```text
block_lift = 0.1272698212
block_other = 0.1258605574
block_push = 0.1279053316
drawer = 0.1490252850
other = 0.1649751909
slider = 0.1365834195
switch_button_light = 0.1818723801
```

Structural diagnostics are recorded but do not train in this run:

```text
loss_anchor_pv = 0.6862529516
loss_anchor_object_pull = 0.7280785441
loss_slot_jepa = 0.2446186841
loss_mapg_routing = 0.3814765513
aqr_same_role_support_overlap_max = 0.4691928625
aqr_active_same_role_support_overlap_max = 0.0064460691
aqr_context_same_role_support_overlap_max = 0.0189009253
owm_tracklet_tokens = 59.7050018311
owm_proposal_tokens = 1.2349998951
posterior_recycle_rate = 0.0608770400
```

Comparison:

```text
G43b no PICF context step25:
  action_default = 0.1456096470
  preclip_grad_norm = 3.8307754993
  semantic_backbone_grad = 2.7212838634

G44 context gate=0.25 step25:
  action_default = 0.1453970224
  preclip_grad_norm = 3.8745079041
  semantic_backbone_grad = 2.7629773737

G45a context gate=0.10 step25:
  action_default = 0.1456323117
  preclip_grad_norm = 3.8519954681
  semantic_backbone_grad = 2.7413436879
```

Interpretation:

```text
G45a passes the first action-boundary gate.  Reducing PICF context gate from
0.25 to 0.10 does not break action loss, gradient scale, logical-batch coverage,
or structural logging at step25.

This is not enough to declare success.  The next hard decision point is step50:
if G45a remains near G44 step50 (0.1036957800) / G43b step50 (0.1041485369),
continue to step75/100.  If it is materially worse (>0.125 or trending back
toward the G35/G37 bad band), stop and switch to the next ablation.
```

### Current Non-Repeat Audit

Before launching any new action-descent experiment, check this list first.

```text
Do not repeat as root fixes:
  sampler-only task_uniform / temperature / explicit-ratio changes;
  optimizer-only reset or scalar LR changes;
  action-weight-only changes;
  raw-overlap-only penalties;
  SAM/proposal-only branches;
  scoped PCGrad/CAGrad-only branches;
  dynamic PiKE-style mixing-only branches.

Keep as infrastructure:
  K4 task-balanced logical batch;
  without-replacement bucket selection;
  per-bucket logical-batch loss scaling;
  per-bucket metric logging;
  optional dynamic mixing and gradient surgery code paths for future diagnostics;
  action/backbone insulation through stop-grad prefix/context boundaries.

Current open gate:
  Can bounded stop-grad PICF context enter the native PI0.5 action path without
  destroying the original action descent curve?

Current live discriminator:
  G45a context_gate=0.10 policy_only run.  Step25 passed.  Step50 must match
  G43b/G44 descent scale; otherwise the next branch should alter context
  capacity or revert to G44 gate=0.25, not repeat old sampler/LR branches.
```

### Condensed Prior-Evidence Audit

This section is the current short-form map of prior experiments.  It exists to
avoid repeating already-failed branches.

```text
E21:
  Exact 12-window balanced update proved the action path has capacity to descend
  fast when each optimizer update approximates a multi-task gradient.
  It is a diagnostic upper bound, not a production sampler.

E23:
  2-window/update production approximation did not reproduce E21-fast descent.
  It stayed in the weaker 0.041-0.048 band.

E24:
  K=4/K=6 logical coverage improved all-window eval.
  K=4 reached the 0.027-0.031 region, but slower than E21.
  Conclusion: coverage helps, but blindly increasing accumulation width is not
  the whole fix; production still needs readable native action conditioning.

G12/G23:
  task_uniform, temperature, explicit ratio, dynamic PiKE-style mixing,
  per-bucket normalization, scoped PCGrad, and scoped CAGrad were implemented
  and tested.
  Keep them as infrastructure; do not relaunch them alone as root fixes.

G25/G44/G45 family:
  remaining causal question is whether PICF belief context can affect the
  deployed PI0.5 action flow without corrupting the native action path.
```

Current mathematical discriminator:

```text
Let C be PICF context and A be the PI0.5 action suffix state.

Rejected old branch:
  optimize sampler/optimizer only while the action expert sees no readable C.

Current branch:
  A' = A + gate * CrossAttn(A, stopgrad(C))
  L = L_action(A')

This preserves action/backbone insulation and tests only whether bounded
context improves or preserves native PI0.5 action descent.
```

### G45a Step50 Gate

Observed remote row:

```text
step = 50
loss_total = 0.2082876563
loss_action_default_equiv = 0.1041438282
loss_total_minus_action = 0.0
loss_action_active7 = 0.4084271789
loss_action_pos = 0.3207883835
loss_action_rot = 0.3904267550
loss_action_gripper = 0.7253445387
preclip_grad_norm = 1.6872459650
grad_norm = 1.6872459650
grad_norm_group_semantic_backbone = 1.1848621246
grad_absmax_group_semantic_backbone = 0.0927734375
logical_batch_distinct_bucket_count = 4
logical_batch_global_micro_count = 4
pi_action_condition_token_count = 8.0
pi_context_token_count = 4.0
pi_context_gate = 0.1000000015
pi_context_adapter_token_count = 8.0
pi_context_adapter_gate = 0.1192178726
lr = 1.6666666667e-05
steps_per_sec = 0.0179612804
```

Structural diagnostics remain non-training:

```text
loss_anchor_pv = 0.6810881495
loss_anchor_object_pull = 0.3800145686
loss_slot_jepa = 0.2442054600
loss_mapg_routing = 0.3962164819
aqr_same_role_support_overlap_max = 0.4625840783
aqr_active_same_role_support_overlap_max = 0.0056569339
aqr_context_same_role_support_overlap_max = 0.0240908153
owm_tracklet_tokens = 60.4300003052
owm_proposal_tokens = 1.1050000191
posterior_recycle_rate = 0.0603432506
```

Comparison:

```text
G43b no PICF context step50:
  action_default = 0.1041485369

G44 context gate=0.25 step50:
  action_default = 0.1036957800

G45a context gate=0.10 step50:
  action_default = 0.1041438282
```

Interpretation:

```text
G45a passes the step50 discriminator.  The lower context gate does not degrade
the native action descent curve; it essentially matches the no-context control
and remains in the same healthy band as G44.

Continue to step75/100.  Do not stop at step50 and do not launch another
sampler/LR/PCGrad branch.
```

### G45a Step75 Gate

Observed remote row:

```text
step = 75
loss_total = 0.1587047577
loss_action_default_equiv = 0.0793523788
loss_total_minus_action = 0.0
loss_action_active7 = 0.3596418500
loss_action_pos = 0.2902964950
loss_action_rot = 0.3863212466
loss_action_gripper = 0.4876396954
preclip_grad_norm = 1.7881172895
grad_norm = 1.7881172895
grad_norm_group_semantic_backbone = 1.2749445929
grad_absmax_group_semantic_backbone = 0.0732421875
logical_batch_distinct_bucket_count = 4
logical_batch_global_micro_count = 4
pi_action_condition_token_count = 8.0
pi_context_token_count = 4.0
pi_context_gate = 0.1000000015
pi_context_adapter_token_count = 8.0
pi_context_adapter_gate = 0.1192366704
lr = 2.5e-05
steps_per_sec = 0.0175888249
```

Structural diagnostics remain non-training:

```text
loss_anchor_pv = 0.6804239750
loss_anchor_object_pull = 0.8684916496
loss_slot_jepa = 0.2449989021
loss_mapg_routing = 0.3960229456
aqr_same_role_support_overlap_max = 0.4622501135
aqr_active_same_role_support_overlap_max = 0.0033086878
aqr_context_same_role_support_overlap_max = 0.0223401114
owm_tracklet_tokens = 58.125
owm_proposal_tokens = 1.1849999428
posterior_recycle_rate = 0.0604016669
```

Comparison:

```text
G44 context gate=0.25 step75:
  action_default = 0.0794120356

G45a context gate=0.10 step75:
  action_default = 0.0793523788
```

Interpretation:

```text
G45a passes the step75 discriminator.  The curve is:
  0.1456323117 -> 0.1041438282 -> 0.0793523788

This matches the healthy G44 descent while using a stricter lower context gate.
Continue to step100.  The step100 decision should determine whether
context_gate=0.10 is acceptable as the safer default or whether G44's 0.25 gate
is preferable for slightly stronger PICF contribution.
```

### G45a Step100 Gate

Observed remote row:

```text
step = 100
loss_total = 0.1540389508
loss_action_default_equiv = 0.0770194754
loss_total_minus_action = 0.0
loss_action_active7 = 0.3501620293
loss_action_pos = 0.2850227654
loss_action_rot = 0.3933986723
loss_action_gripper = 0.4158699512
preclip_grad_norm = 1.0556385517
grad_norm = 1.0556385517
grad_norm_group_semantic_backbone = 0.7538653228
grad_absmax_group_semantic_backbone = 0.0510253906
logical_batch_distinct_bucket_count = 4
logical_batch_global_micro_count = 4
pi_action_condition_token_count = 8.0
pi_context_token_count = 4.0
pi_context_gate = 0.1000000015
pi_context_adapter_token_count = 8.0
pi_context_adapter_gate = 0.1192246825
lr = 3.3333333333e-05
steps_per_sec = 0.0177001554
```

Structural diagnostics remain non-training:

```text
loss_anchor_pv = 0.6786350012
loss_anchor_object_pull = 0.3918083906
loss_slot_jepa = 0.2435719669
loss_mapg_routing = 0.3887859881
aqr_same_role_support_overlap_max = 0.4645918012
aqr_active_same_role_support_overlap_max = 0.0045025651
aqr_context_same_role_support_overlap_max = 0.0225472338
owm_tracklet_tokens = 59.9300003052
owm_proposal_tokens = 1.2300000191
posterior_recycle_rate = 0.0607176237
```

Comparison:

```text
G44 context gate=0.25 step100:
  action_default = 0.0769066960

G45a context gate=0.10 step100:
  action_default = 0.0770194754
```

G45a curve:

```text
step25  = 0.1456323117
step50  = 0.1041438282
step75  = 0.0793523788
step100 = 0.0770194754
```

Bad signatures:

```text
CUDA out of memory: false
SqrtBackward0: false
Traceback: false
Non-finite loss_total: false
```

Decision:

```text
G45a passes the 100-step gate.  A lower PICF context output gate (0.10) keeps
the native PI0.5 action descent essentially identical to the G44 0.25 gate and
the no-context control, while maintaining stricter insulation.

This resolves the immediate action-boundary safety question for short-horizon
diagnostics.  It does not prove 30k success, CALVIN success, or long-run
anti-plateau behavior.  The next production candidate should keep:
  task-balanced K4 logical batch,
  per-bucket normalization,
  action context suffix cross-attention,
  stop-grad PICF context,
  context_gate in the 0.10-0.25 range.

For a conservative long run, start with context_gate=0.10.  For a stronger PICF
readout stress test, use context_gate=0.25.  Do not rerun sampler-only,
optimizer-only, or PCGrad/CAGrad-only branches.
```

### G45b 30k Launch

Run:

```text
picf_g45b_context_gate010_policyonly_long30k_20260605
```

Purpose:

```text
Production-scale action-boundary validation using the G45a-passed contract.
This is not a new method branch; it is the long run of the verified short-gate
configuration.
```

Key settings:

```text
num_train_steps = 30000
save_interval = 1000
keep_last_checkpoints = 5
progress_bar = true
log_interval = 50

picf_trainable_scope = policy_only
semantic_trainable = true
semantic_trainable_scope = backbone_only

picf_action_condition_enabled = true
action_prefix_stopgrad = true
action_prefix_output_gate = 0.0
action_context_tokens = 4
action_context_integration = suffix_cross_attention
action_context_stopgrad = true
action_context_norm_mode = rmsnorm
action_context_output_gate = 0.10
action_context_include_query_tokens = false

calvin_bucket_sampling_mode = task_uniform
logical_batch_task_count = 4
logical_batch_bucket_normalization = true
logical_batch_log_bucket_metrics = true
calvin_bucket_sample_without_replacement = true

lr = 0.0002
min_lr = 0.00002
warmup_steps = 600
accum_steps = 2
training_strategy = fsdp_full_shard
window_activation_checkpointing = true

all structure/auxiliary lambda weights = 0.0
```

Startup status:

```text
tmux = picf_g45b_context_gate010_policyonly_long30k_20260605
log = /mnt/picf_run_logs/picf_g45b_context_gate010_policyonly_long30k_20260605.log

Reached progress step7.
GPU memory ~= 39.3GB / A100-40GB.
GPU utilization active.

Bad signatures:
  CUDA out of memory: false
  Traceback: false
  ModuleNotFoundError: false
  Non-finite loss_total: false
```

Runtime re-audit after launch:

```text
tmux session still active.
GPUs are saturated and memory remains within A100-40GB capacity.
/mnt has sufficient headroom after cleanup:
  used ~= 1.5T
  available ~= 277G
  use% ~= 85%

Startup contract confirmed in log:
  save_interval = 1000
  keep_last_checkpoints = 5
  training_strategy = fsdp_full_shard
  accum_steps = 2
  effective_global_batch = 4
  calvin_bucket_sampling_mode = task_uniform
  logical_batch_task_count = 4
  logical_batch_bucket_normalization = true
  action_context_output_gate = 0.10
  action_context_stopgrad = true
  action_prefix_stopgrad = true

No runtime error signatures observed:
  Traceback = 0
  CUDA out of memory = 0
  Non-finite = 0
  RuntimeError = 0
  ModuleNotFoundError = 0
  NaN/nan = 0
```

Step50 gate:

```text
step = 50
loss_total = 0.2496266067
loss_action_default_equiv = 0.1248133034
loss_action_active7 = 0.4569933414
lr = 1.6666666667e-05
pi_context_gate = 0.1000000015
logical_batch_global_micro_count = 4

No runtime error signatures:
  Traceback = 0
  CUDA out of memory = 0
  Non-finite = 0
  RuntimeError = 0
  ModuleNotFoundError = 0
  NaN/nan = 0
```

Interpretation:

```text
G45b step50 is not directly equal to G45a step50 because the logging interval
changed:

  G45a log_interval = 25
  G45b log_interval = 50

Therefore:
  G45a step25 averages steps 1-25:
    loss_action_default_equiv = 0.1456323117

  G45a step50 averages steps 26-50:
    loss_action_default_equiv = 0.1041438282

  G45b step50 averages steps 1-50:
    loss_action_default_equiv = 0.1248133034

The G45b value lies between the two corresponding G45a windows.  This is
consistent with the wider averaging window and is not evidence of a regression.
The next hard discriminator is G45b step100, which will average steps 51-100.
That should be compared against the G45a step75/100 band.
```

Local verification before archiving:

```text
uv run python -m py_compile \
  scripts/picf_core_train.py \
  src/openpi/picf/core/training.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/policy.py
PASS

uv run pytest -q src/openpi/picf/core/training_test.py \
  -k 'action_override_detach_picf_keeps_override_gradient or action_override_preserves_default_parity or action_override_respects_zero_action_lambdas'
3 passed

uv run pytest -q src/openpi/picf/core/pipeline_test.py \
  -k measurement_innovation_norm_exact_match_has_finite_backward
1 passed

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'calvin_bucket_sampling_weights or bucket_sequence_without_replacement or bucket_sequence_round_robin or logical_batch_loss_scales or dynamic_bucket_sampling_weights or gradient_surgery_params or pcgrad or cagrad'
12 passed

uv run pytest -q src/openpi/picf/policy_test.py -k 'action_context or action_condition'
2 passed

uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py -k action_context
7 passed
```

Checkpoint cleanup after launch:

```text
Freed about 72GB on /mnt.
/mnt moved from 93% used to 89% used.

Removed:
  June-05 native PI0.5 probe checkpoints G30-G33;
  short G26b tokenaux diagnostic branches;
  G35 bad no-action-condition headadapter branch.

Kept:
  all pre-May checkpoints;
  4-22 ablation and full PICF baselines;
  G26b long30k base used as args source;
  G43b/G44/G45a/G45b action-boundary evidence.
```

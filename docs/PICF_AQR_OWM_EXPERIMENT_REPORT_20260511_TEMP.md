# PICF-AQR-OWM 2026-05-11 experiment report

Status: live experiment report for the current recycle/action-gradient,
object-binding, and long-run launch diagnosis. This document is subordinate to
[`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
and records what was actually tested, what was only theoretically inferred, and
what remains unproven until the current long run produces checkpoints/videos.

## Current cloud state

### A7 long run

Machine:

```text
ssh -p 28060 root@36.139.225.68
```

Active tmux:

```text
a7_30k_prefixstopgrad_all_a1_semtrain_fast_95ea69b
```

Log:

```text
/mnt/checkpoints/picf_core/picf_core/picf_a7_30k_prefixstopgrad_all_a1_semtrain_fast_from450_20260511_95ea69b.train_tmux.log
```

Metrics:

```text
/mnt/checkpoints/picf_core/picf_core/picf_a7_30k_prefixstopgrad_all_a1_semtrain_fast_from450_20260511_95ea69b/metrics.jsonl
```

Launch contract:

```text
runtime: 95ea69b
resume: model-only from A5 prefix-stopgrad step 450
training_strategy: fsdp_full_shard
trainable_scope: all
semantic/PaliGemma: trainable=True
Sonata/V-JEPA/AnyTouch: trainable=False
picf_action_prefix_stopgrad: enabled
accum_steps: 1
unroll_steps: 2
effective_global_batch: 2
lr: 2e-4
save_interval: 2500
keep_last_checkpoints: 3
progress: enabled
```

Early speed and health at step 460:

```text
speed: ~21.3 s/step
loss_total: 1.0268
loss_alignment: 0.9882
loss_action: 0.0386
loss_action_default_equiv: 0.0772
loss_action_active7: 0.3167
posterior_recycle_rate: 0.00838
posterior_recycle_logit_mean: -47.61
posterior_address_update_rate_mean: 0.0222
aqr_same_role_support_overlap_max: 0.738
aqr_same_role_local_true_overlap_max: 0.0497
aqr_temporal_view_mass_0/static: 0.444
aqr_temporal_view_mass_1/gripper: 0.556
owm_tracklet_tokens: 0
owm_proposal_tokens: 0
```

Interpretation:

```text
1. The run is using the intended current runtime and trainable/frozen profile.
2. Speed is comparable to the recent MVTrack semantic-trainable profile
   (~20 s/step) and slower than older v22 AQR (~17 s/step), but it is not the
   36-37 s/step caused by the earlier incomparable accum_steps=4 launch.
3. The early recycle signal remains healthy under prefix-stopgrad even with
   PaliGemma trainable and the action loss active on the PI0.5/action side.
4. Tracklet/proposal metrics are still zero because the current CALVIN training
   dataflow does not feed tracklet/proposal tensors. Those branches are runtime
   no-op in this run.
```

Later live check:

```text
step 480:
  recycle_rate=3.9e-7, action_default_equiv=0.0797,
  same_role_overlap=0.8505

step 500:
  recycle_rate=7.6e-8, action_default_equiv=0.0606,
  same_role_overlap=0.7299

step 520:
  recycle_rate=0.4627, recycle_logit_mean=-1.6363,
  action_default_equiv=0.0682, same_role_overlap=0.8539,
  local_true_overlap=0.0495
```

Interpretation:

```text
The A7 long run is still active and not yet accepted. The step-520 recycle
spike shows that the prefix-stopgrad/binding-signature repair is not proven
stable under long-run all/PaliGemma-trainable cotrain. It may be a transient
batch spike, but it must be judged by the next checkpoints and recycle/overlap
trend, not by the earlier healthy step-460 or step-500 samples alone.
```

Tail command:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_30k_prefixstopgrad_all_a1_semtrain_fast_from450_20260511_95ea69b.train_tmux.log
```

### A5 diagnosis state

Machine:

```text
ssh -p 29776 root@36.139.225.68
```

Current state:

```text
no active tmux
GPU memory: 0 MiB on both GPUs
```

The latest completed A5 diagnostic is:

```text
/mnt/checkpoints/picf_core/picf_core/picf_a5_prefixstopgrad_a025_bindsub_diag300_20260511_95ea69b/metrics.jsonl
```

Final step 600 summary:

```text
loss_total: 0.7226
loss_alignment: 0.7020
loss_action: 0.0205
loss_action_default_equiv: 0.1642
loss_action_active7: 0.5794
loss_anchor_pv: 2.4117
loss_pv_weak: 1.4475
loss_aqr_denoising: 1.3147
loss_slot_jepa: 1.8211 diagnostic-only, lambda=0
posterior_recycle_rate: 0.000469
posterior_recycle_logit_mean: -11.6767
posterior_address_update_rate_mean: 0.00421
posterior_identity_switch_rate: 0.0
aqr_same_role_support_overlap_max: 0.3027
aqr_temporal_view_mass_0/static: 0.318
aqr_temporal_view_mass_1/gripper: 0.682
owm_tracklet_tokens: 0
owm_proposal_tokens: 0
```

Interpretation:

```text
This is the strongest short-run evidence so far that action-prefix stopgrad
plus binding-subspace support signatures prevents the recycle-saturation
failure while preserving useful action-side feedback as a monitored signal.
```

### A5 runtime smoke and local audit addendum

This addendum records the extra validation performed after the A7 long run was
launched. It is not behavior acceptance; it is a code-path and runtime-entry
check for the current branch.

Local checks on the working tree:

```text
python -m py_compile:
  contracts/config/pipeline/training/vjepa wrapper/train/serve/diagnosis/evidence
  scripts all passed.

python scripts/verify_picf_owm_contract.py:
  31/31 PASS.

python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS. Warnings only when no metrics/eval bundle is supplied.

python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS.

python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS.

pytest -q scripts/verify_picf_owm_contract_test.py scripts/picf_owm_evidence_bundle_test.py:
  4 passed.

pytest -q src/openpi/picf/core/pipeline_test.py -k "tracklet or proposal or ordinal or mvtrack or cache or temporal":
  10 passed, 60 deselected.

pytest -q src/openpi/picf/core/training_test.py -k "jepa or support or binding or action or loss":
  24 passed, 5 deselected.
```

A5 idle-card runtime smoke:

```text
machine: A5, ssh -p 29776 root@36.139.225.68
runtime: 95ea69b
tmux: a5_runtime_smoke_2step_95ea69b, completed
log:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_runtime_smoke_2step_20260511_95ea69b.train_tmux.log
metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_runtime_smoke_2step_20260511_95ea69b/metrics.jsonl
profile:
  full foundation-backbone trainer entry, FSDP, PaliGemma trainable,
  Sonata/V-JEPA/AnyTouch frozen, action-prefix stopgrad enabled.
```

Step 2 smoke metrics:

```text
loss_total: 1.3319
loss_alignment: 1.0745
loss_action_default_equiv: 0.5149
loss_action_active7: 0.8047
posterior_recycle_rate: 0.0918
posterior_recycle_logit_mean: -2.4118
posterior_address_update_rate_mean: 0.00270
aqr_same_role_support_overlap_max: 0.9480
aqr_same_role_local_true_overlap_max: 0.00966
owm_posterior_binding_signature_norm_mean: 1.0
owm_tracklet_tokens: 0
owm_proposal_tokens: 0
grad_clip_applied: true
preclip_grad_norm: 106.1
grad_norm: 5.0
```

Interpretation:

```text
1. The current trainer entry, FSDP loading, frozen/trainable profile,
   action-prefix stopgrad, binding-signature path, and metric logging execute
   on GPU.
2. This two-step smoke is intentionally too short to prove convergence.
3. The no-resume smoke starts from a cold model, so its overlap/recycle numbers
   are not comparable to the resumed A5 step-600 or A7 step-460 runs.
4. It independently confirms that current CALVIN dataflow still leaves
   tracklet/proposal evidence inactive unless those tensors are supplied.
```

## Historical diagnosis matrix

The key question in the previous diagnosis was whether the apparent action-loss
improvement came from healthy anchors or from a posterior recycle shortcut. The
answer is now clear enough for the current stage:

| Run | Main condition | Health result | Interpretation |
| --- | --- | --- | --- |
| R0 anchor-only warmup | No action pressure | overlap improved but recycle still nonzero | anchors can separate when action shortcut is absent |
| R1/R2 direct action cotrain | action gradients enter PICF prefix/posterior path | recycle_rate=1.0, large recycle logits | scalar action loss can improve while posterior identity collapses |
| A3 freeze recycle path | recycle parameters frozen | recycle still saturates | problem is upstream gradient shaping, not just recycle-head weights |
| A4 recycle-logit clamp=6 | clamp only | sigmoid(6) ~= 0.998 | mathematically too loose to close recycle |
| B1 pos-only | position loss only | recycle saturates | not only rotation/gripper |
| B2 rot-only | rotation loss only | recycle saturates | not only position |
| C1 local-debug | true-local overlap instrumentation | true local overlap low despite local proxy high | local-overlap proxy had false positives |
| A7 direct 95ea69b | binding subspace but direct action | recycle_rate=1.0, slot_jepa diagnostic explodes | binding subspace alone does not protect from direct action gradients |
| A5 prefix-stopgrad 95ea69b | binding subspace + action-prefix stopgrad | recycle_rate=0.00047, overlap=0.303 | current repair is structurally effective in short run |
| A7 30k current | PaliGemma trainable, frozen pretrains, prefix-stopgrad | early recycle=0.0084 at step 460 | long-run acceptance is pending |

Core conclusion:

```text
Direct early action gradients into the PICF posterior/control-prefix path create
a recycle shortcut. The current repair is not "turn off action"; it is a
bridge-level stop-gradient at conditioned_control.pi_prefix_tokens. PI0.5/action
side still trains, while PICF binding/recycle/address are protected from the
shortcut until anchor identity is stable enough to re-open selectively.
```

## Loss and metric interpretation

### Action loss

Use these fields for comparison:

```text
loss_action:
  actual optimized weighted contribution.

loss_action_default_equiv:
  mapped to default old action-weight scale and therefore the closest scalar
  for comparing against 2026-04-22 and v22 action curves.

loss_action_active7:
  active-dimension unweighted action loss; useful for checking whether a low
  scalar loss is just a weighting artifact.
```

The current A7 step 460 `loss_action_default_equiv=0.0772` should be interpreted
as early-regime progress, not as proof of 20k/40k-level convergence. It is still
far too early for behavior acceptance.

### Support overlap

Use both support-overlap and local true-overlap:

```text
aqr_same_role_support_overlap_max:
  global support overlap. Healthy short-run values are clearly below the
  previous collapse regime near 0.95-1.0.

aqr_same_role_local_true_overlap_max:
  token-aware local-overlap sanity metric. It is less prone to the false
  positive seen in earlier local-refinement diagnostics.
```

The current A7 long run is not yet as clean as A5 prefix-stopgrad:

```text
A5 step 600 support overlap: 0.303
A7 step 460 support overlap: 0.738
```

This does not invalidate A7 because it is a different profile with PaliGemma
trainable, full long-run optimizer settings, and early warmup. It does mean the
first 2500-step checkpoint must be judged by overlap trend, not action loss
alone.

### Recycle and address

Healthy short-run signature:

```text
posterior_recycle_rate << 1
posterior_recycle_logit_mean not large positive
posterior_address_update_rate_mean > 0
posterior_identity_switch_rate not exploding
```

Current evidence:

```text
A7 direct action: recycle_rate=1.0, recycle_logit_mean ~= 393, address_update=0
A5 prefix-stopgrad: recycle_rate=0.00047, recycle_logit_mean=-11.68
A7 current 30k early: recycle_rate=0.00838, recycle_logit_mean=-47.61
```

This is the strongest causal evidence that the prefix-stopgrad repair is
addressing the correct failure mode.

## Object-binding paper audit

Primary source:

```text
Yihao Li, Saeed Salehi, Lyle Ungar, Konrad P. Kording.
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
arXiv:2510.24709v2, accepted as NeurIPS 2025 Spotlight.
https://arxiv.org/abs/2510.24709
```

Relevant claims from the paper:

```text
1. ViT patch embeddings encode an IsSameObject relation: whether two tokens
   belong to the same object.
2. The strongest probe is quadratic / pairwise, not a pointwise object-ID
   classifier and not plain cosine similarity alone.
3. The relation is encoded in a low-dimensional subspace on top of object
   features and correlates with attention query-key similarity.
4. DINO/CLIP/supervised ViTs show stronger binding than MAE, so binding is
   acquired from pretraining objectives, not guaranteed by transformer
   architecture alone.
5. Mid/upper layers are important; early/local features alone are not enough.
```

Mapping to PICF-AQR-OWM:

```text
Implemented:
  token -> binding_key = normalize(W_b token)
  slot_binding_signature = weighted_mean(binding_key, support_weights)
  binding_logit += lambda_b * gate * dot(prev_signature, obs_signature)

Why this matches the paper:
  It treats object membership as a pairwise relation between token-derived
  signatures, not as a pointwise class label. The learned projection W_b is a
  low-dimensional binding subspace, and the dot product is the operational
  pairwise/quadratic term used by posterior binding.

Why this is not overclaiming:
  The paper analyzes pretrained ViTs under image segmentation labels/probes.
  It does not prove that our robotic typed memory automatically solves
  manipulation binding. Our code-level repair is a justified architectural
  adaptation, and A5/A7 short runs provide early causal support, but final
  acceptance still requires current-run CALVIN/video/anchor-health evidence.
```

What has been tested:

```text
1. Binding-subspace structural code path exists and is FSDP-compatible.
2. A7 direct action with binding subspace but without prefix-stopgrad still
   fails through recycle saturation. This means binding subspace is not a
   magic patch for harmful action gradients.
3. A5 prefix-stopgrad + binding subspace produces healthy recycle and low
   support overlap through step 600.
4. A7 current long run starts with healthy recycle under the full PaliGemma
   trainable profile.
```

What has not been tested:

```text
1. Offline IsSameObject probe/audit over V-JEPA/static/wrist/PaliGemma tokens
   using weak robot labels from tracklets, point neighborhoods, contact
   neighborhoods, PG support overlap, and posterior slot continuity.
2. Layer-selection sweep for V-JEPA/PaliGemma binding signatures.
3. Tracklet/proposal binding, because current CALVIN training input still has
   owm_tracklet_tokens=0 and owm_proposal_tokens=0.
4. Selective re-opening of PICF action gradients after prefix-stopgrad warmup.
```

Recommended next paper-derived experiment:

```text
Add an offline IsSameObject-style audit before changing more training losses.
Do not require dataset relabeling. Construct weak same/different pairs from:
  - temporal track continuity when available
  - point-cloud local neighborhoods
  - tactile/contact neighborhoods
  - high-confidence PG support overlap
  - high-confidence posterior slot continuity from healthy runs

Train or evaluate a lightweight pairwise score:
  score(i, j) = z_i^T W z_j

Report:
  - same/different AUC by modality
  - static vs wrist token separability
  - V-JEPA layer/time/view separability
  - whether binding_signature_proj improves over raw cosine
```

This is the direct experimental counterpart of the paper's probe and is the
cleanest way to know whether the current typed evidence actually contains the
object-binding signal the architecture expects.

## Remaining risks and acceptance gates

Do not claim full behavior-level completion until these pass:

```text
1. A7 2500-step checkpoint:
   recycle remains low, address update remains nonzero, same-role overlap does
   not return to 0.95-1.0, action_default_equiv trends down.

2. A7 5000-step checkpoint:
   CALVIN/video/anchor overlays show that support separation is visible, not
   only metric-friendly.

3. Offline IsSameObject audit:
   binding_signature subspace beats raw hidden cosine and geometry-only
   baselines on weak same-object pairs.

4. Tracklet/proposal dataflow:
   owm_tracklet_tokens and owm_proposal_tokens become nonzero in a dedicated
   dataflow test before any claim that MVTrack tracklets/proposals are active
   in training.
```

## A5 10-hour follow-up plan, revised after E1 early failure

Current judgment from live data:

```text
A5 E1 has already produced enough diagnostic evidence.
It does not need to run to 900 steps.
```

Observed E1 profile:

```text
run: picf_a5_e1_unroll2_semtrain_repro900_20260511_95ea69b
runtime: 95ea69b
profile: all PICF trainable, PaliGemma trainable, unroll_steps=2, accum_steps=1, guarded OWM losses 0
latest inspected: step 720
posterior_recycle_rate: 1.0
posterior_address_update_rate_mean: 0.0
aqr_same_role_support_overlap_max: 0.99995
loss_anchor_pv: 4.74
loss_mapg_routing: 1.17
loss_slot_jepa diagnostic: O(180), lambda still 0
grad_norm: clipped at 5.0
```

Mathematical interpretation:

```text
This is a true identity-filter failure, not just noisy scalar loss.
The posterior recycle gate has saturated, the address update path is effectively
shut off, and same-role supports have collapsed to almost identical evidence.
Since lambda_slot_jepa/lambda_support_pred/lambda_binding_consistency are zero,
the failure cannot be blamed on enabled predictive auxiliary losses.
```

This updates the causal hypotheses:

```text
H1: Direct all-scope unroll=2 with PaliGemma cotrain is too aggressive for
    the current anchor identity filter.
H2: PaliGemma cotrain is still likely necessary for final action adaptation,
    based on older action runs, so freezing PG is an isolation test only, not
    the preferred final recipe.
H3: State-only burn-in may provide the identity inertia missing from short
    unroll=2 while keeping PaliGemma cotrain.
H4: Tiny predictive/binding aux losses should only be tested after the base
    identity path is stable; they are not the cause of E1 because they were 0.
```

Why 500 steps is enough for this diagnostic layer:

```text
The observed failure appears by the first post-500 samples and remains locked
through step 720. For early identity-filter acceptance, 500 steps is enough to
catch the failure mode: recycle saturation, overlap collapse, address update
shutdown, and clipped gradients. Passing 500 steps is not final training proof;
it only permits moving to a longer 2500-step acceptance run.
```

Revised A5 tmux plan:

```text
Machine:
  ssh -p 29776 root@36.139.225.68

Session:
  tmux attach -t a5_owm_10h_matrix500final2_95ea69b

Common runtime:
  /root/openpi_runtimec_95ea69b
  runtime commit 95ea69b
  resume /mnt/checkpoints/picf_core/picf_core/model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b
  FSDP full shard, 2xA100
  Sonata/V-JEPA/AnyTouch frozen
  PI0.5 action path active
  picf_action_prefix_stopgrad enabled
  evidence_cache_read_weight=0.05
  bind_embedding_signature_weight=0.25
  progress enabled
  resume starts near trainer step 450
  num_train_steps=950, i.e. about 500 new optimizer steps
  log_interval=20
  save_interval=950
```

Runs:

```text
M1 picf_a5_m1c_unroll2_semlr1e6_aux0_500new_20260512_95ea69b
  PaliGemma present but semantic_lr_scale=1e-6, unroll_steps=2, accum_steps=1, OWM aux losses 0.
  Question: does full all-scope unroll=2 collapse even without PaliGemma weight updates?
  The code path has no --no-semantic-trainable flag under foundation profile, so lr_scale=1e-6 is
  the clean isolation of semantic cotrain update pressure. If M1 collapses, the problem is
  not PG parameter update; it is all-scope/action-window pressure on identity.

M2 picf_a5_m2b_burnin4_semtrain_aux0_500new_20260512_95ea69b
  PaliGemma trainable, burnin_steps=4, burnin_mode=state_only, unroll_steps=1,
  accum_steps=1, OWM aux losses 0.
  Question: can we keep the action-useful PG cotrain while restoring posterior
  identity inertia through state-only burn-in?
  This is the most important candidate for the next production recipe.

M3 picf_a5_m3b_burnin4_semtrain_tinyaux_500new_20260512_95ea69b
  Same as M2, but lambda_slot_jepa=lambda_support_pred=lambda_binding_consistency=1e-4.
  Question: after the identity path is protected by burn-in, do tiny predictive
  hooks immediately conflict, or can they be considered for later warmup?

M4 picf_a5_m4b_unroll2_semtrain_semlr1e61_aux0_500new_20260512_95ea69b
  PaliGemma trainable, unroll_steps=2, accum_steps=1, semantic_lr_scale=0.1,
  OWM aux losses 0.
  Question: if M2 succeeds but direct unroll=2 fails, can reduced semantic LR
  make PG cotrain compatible with short unroll? This is a fallback, not the
  preferred recipe unless it beats M2 on identity metrics.
```


Aborted launch note:

```text
The first revised M1 launch used num_train_steps=500. Since the checkpoint
resume carries trainer step ~=450, that would only run about 50 new steps. It
was stopped and replaced by the `500new` matrix with num_train_steps=950.
Do not use the non-`500new` M1 directory for convergence conclusions.
```

Primary acceptance criteria for each 500-step run:

```text
posterior_recycle_rate <= 0.05 after step 300, no sustained spikes > 0.2
posterior_address_update_rate_mean > 0.003 unless recycle is intentionally active
aqr_same_role_support_overlap_max < 0.85, and preferably < 0.70
aqr_same_role_local_true_overlap_max remains low (< 0.12)
grad_norm not permanently clipped at 5.0
loss_anchor_pv not stuck near 4.7
loss_mapg_routing not monotonically worsening above ~1.17
loss_action_default_equiv is tracked but is not sufficient acceptance by itself
```

Decision table:

```text
M1 stable, M2 stable:
  E1 failure is mainly PaliGemma update pressure + short-window interaction. Prefer M2
  for final run because PG cotrain remains action-useful and burn-in protects identity.

M1 stable, M2 fails:
  PaliGemma cotrain update pressure itself is too strong even with burn-in. Use staged recipe:
  anchor/identity warmup with PG frozen, then low-LR PG cotrain.

M1 fails:
  all-scope unroll=2/action-window pressure is enough to break identity. Avoid
  direct unroll=2 all-scope long runs; use state-only burn-in or staged anchor warmup.

M2 stable, M3 fails:
  predictive/binding aux losses still conflict and must stay 0 for production.

M2 stable, M3 stable:
  tiny aux hooks are not immediately toxic, but still require a longer 2500-step
  acceptance before default enablement.

M4 stable while E1 failed:
  semantic LR is a significant lever. It can be used as a speed fallback, but
  compare against M2 before choosing final production profile.
```

Tail commands:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m1d_unroll2_semlr1e6_aux0_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m2b_burnin4_semtrain_aux0_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m3b_burnin4_semtrain_tinyaux_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m4b_unroll2_semtrain_semlr01_aux0_500new_20260512_95ea69b.train_tmux.log
```

### A7 follow-up allocation

The previous A7 direct `unroll_steps=2` 30k run is no longer the best use of
2xA100 time. By step 860 it still had:

```text
tail_mean loss_action_default_equiv ~= 0.061
tail_mean posterior_recycle_rate ~= 0.0017
tail_mean posterior_address_update_rate_mean ~= 0.0207
tail_mean aqr_same_role_support_overlap_max ~= 0.957
tail_mean aqr_same_role_local_true_overlap_max ~= 0.052
```

Interpretation:

```text
Action can train, and recycle is mostly suppressed, but same-role global support
overlap remains too high. This profile is not a clean acceptance run.
```

Therefore A7 was repurposed to a complementary stress test of the strongest
theoretical candidate:

```text
run: picf_a7_burnin4_semtrain_aux0_1000new_20260512_95ea69b
session: tmux attach -t a7_burnin4_semtrain_1000new_95ea69b
resume: model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b
num_train_steps: 1450, about 1000 new optimizer steps from the resume point
profile:
  PaliGemma trainable
  Sonata/V-JEPA/AnyTouch frozen
  picf_action_prefix_stopgrad enabled
  burnin_steps=4
  burnin_mode=state_only
  unroll_steps=1
  lambda_slot_jepa=lambda_support_pred=lambda_binding_consistency=0
  evidence_cache_read_weight=0.05
  bind_embedding_signature_weight=0.25
```

Rationale:

```text
A5 is used for causal short-run isolation. A7 is used for longer stress on the
candidate that best matches the theory: preserve PaliGemma cotrain for action
adaptation, but provide posterior identity inertia through state-only burn-in.
If A7 burnin4 stays healthy while direct unroll=2 does not, the production
recipe should move to burnin4/state_only before any predictive aux warmup.
```

Tail command:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_burnin4_semtrain_aux0_1000new_20260512_95ea69b.train_tmux.log
```

Scope boundary:

```text
These four runs are designed to isolate training-pressure conflicts. They do
not solve tracklet/proposal inactivity, ordinal grounding, or offline
IsSameObject probing. Those require separate dataflow/probe tests and should
not be inferred from scalar training loss alone.
```

Guarded losses remain guarded:

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
```

Reason:

```text
The previous direct-action failure demonstrated that scalar losses can descend
while posterior identity collapses. Predictive and binding auxiliary losses
should only be enabled after recycle/address/support metrics stay healthy under
longer cotrain.
```

## Current conclusion

```text
1. The core diagnosed failure was real: direct action gradients into PICF
   posterior/control-prefix state caused recycle saturation.
2. The current repair is mathematically targeted: stop action gradients at the
   PICF-to-PI0.5 prefix bridge while preserving action-side learning.
3. The paper-derived binding-subspace term is conceptually aligned with
   IsSameObject/quadratic probe evidence and is not a random patch.
4. Short-run evidence supports the repair, especially A5 prefix-stopgrad.
5. The old A7 direct-unroll2 long run is not an acceptance run because global
   same-role support overlap remained high. The new A7 burnin4/state_only run is
   the current medium-horizon candidate stress test.
```

## Compact Handoff, 2026-05-12 02:00 CST

This section is the minimal state needed to continue after context compaction.

### Current Code State

```text
branch: Posterior_VLA
latest documented commit: 6ba5a21 Document A7 burnin candidate stress test
runtime used on A5/A7: /root/openpi_runtimec_95ea69b
runtime commit marker: 95ea69b
```

Local validation before handoff:

```text
python -m py_compile scripts/picf_core_train.py src/openpi/picf/core/{config,contracts,pipeline,training}.py: PASS
python scripts/verify_picf_owm_contract.py: PASS, including MVTrack README guard
```

### What Is Already Decided

```text
Direct all-scope unroll=2 is not a healthy production recipe.
```

Evidence:

```text
A5 E1, PaliGemma trainable, unroll=2, aux=0:
  posterior_recycle_rate -> 1.0
  posterior_address_update_rate_mean -> 0.0
  aqr_same_role_support_overlap_max -> ~0.99995

A5 M1d, semantic_lr_scale=1e-6, unroll=2, aux=0:
  still collapses by step 580-680
  recycle_rate ~= 1.0
  address_update ~= 0.0
  same_role_overlap ~= 0.99
  grad_norm clipped at 5.0
```

Interpretation:

```text
The failure is not merely PaliGemma parameter-update pressure. It appears when
all-scope action training is paired with short direct unroll=2. The identity
filter can collapse even when semantic LR is effectively frozen.
```

### A7 Current Role

Old A7 direct `unroll_steps=2` run was stopped and is not an acceptance run.
It showed action learning but unhealthy global same-role overlap:

```text
old A7 direct tail mean:
  loss_action_default_equiv ~= 0.061
  posterior_recycle_rate ~= 0.0017
  posterior_address_update_rate_mean ~= 0.021
  aqr_same_role_support_overlap_max ~= 0.962
  aqr_same_role_local_true_overlap_max ~= 0.051
```

Current A7 run:

```text
session: a7_burnin4_semtrain_1000new_95ea69b
run: picf_a7_burnin4_semtrain_aux0_1000new_20260512_95ea69b
profile:
  PaliGemma trainable
  Sonata/V-JEPA/AnyTouch frozen
  burnin_steps=4
  burnin_mode=state_only
  unroll_steps=1
  aux losses = 0
  action_prefix_stopgrad enabled
  num_train_steps=1450, about 1000 new steps from resume step ~450
```

Latest inspected A7 burnin4 state at step 660:

```text
posterior_recycle_rate ~ 0
posterior_address_update_rate_mean ~= 0.039
posterior_identity_switch_rate ~= 0.79-0.82
same_role_support_overlap tail mean ~= 0.959
loss_anchor_pv tail mean ~= 4.40
grad_norm clipped at 5.0; preclip norms often huge
```

Interpretation:

```text
Burnin4/state_only suppresses recycle saturation, but it has not yet solved
same-role support overlap or identity switching. Let it continue only if the goal
is to see whether this recovers by ~950/1450; do not treat early burnin4 as a
healthy solution yet.
```

Tail:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_burnin4_semtrain_aux0_1000new_20260512_95ea69b.train_tmux.log
```

### A5 Current Role

A5 is now running only the remaining useful short diagnostics after M1d already
failed:

```text
session: a5_owm_m2to4_500new_95ea69b
script: /tmp/a5_owm_m2to4_500new.sh
```

Runs in order:

```text
M2 picf_a5_m2b_burnin4_semtrain_aux0_500new_20260512_95ea69b
  PaliGemma trainable, burnin4/state_only, unroll1, aux=0.
  Checks whether A7 burnin4 behavior reproduces in short diagnostic form.

M3 picf_a5_m3b_burnin4_semtrain_tinyaux_500new_20260512_95ea69b
  Same as M2 plus slot/support/binding aux = 1e-4.
  Only meaningful if M2 is at least not collapsed.

M4 picf_a5_m4b_unroll2_semtrain_semlr01_aux0_500new_20260512_95ea69b
  Direct unroll2, PaliGemma trainable but semantic_lr_scale=0.1, aux=0.
  Tests whether reducing semantic LR helps direct unroll2. If M1d already failed,
  M4 is a fallback, not the preferred production recipe.
```

Tail:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m2b_burnin4_semtrain_aux0_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m3b_burnin4_semtrain_tinyaux_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_m4b_unroll2_semtrain_semlr01_aux0_500new_20260512_95ea69b.train_tmux.log
```

### Decision Rules For Next Operator

```text
If M2/A7 burnin4 still has same_role_overlap > 0.9 and identity_switch > 0.5:
  burnin4 alone is insufficient. Do not start 30k from this recipe.
  Next test should reduce all-scope action pressure or use staged anchor warmup.

If M2 stabilizes overlap < 0.85 and recycle < 0.05:
  burnin4/state_only becomes the leading production candidate.
  Then compare M3 to decide whether tiny predictive aux is safe.

If M3 worsens relative to M2:
  keep slot_jepa/support_pred/binding_consistency at 0 for production.

If M4 improves direct unroll2 substantially:
  semantic LR is a useful control lever, but still compare against M2.
```

### Uncovered Issues

These are not solved by A5/A7 scalar training diagnostics:

```text
tracklet/proposal dataflow remains inactive in CALVIN: owm_tracklet_tokens=0, owm_proposal_tokens=0.
ordinal/fourth-object grounding remains weak diagnostic, not rank-supervised.
offline IsSameObject probe has not been run.
behavior acceptance still requires fresh CALVIN/video/anchor overlays.
```

## 2026-05-12 10-hour diagnostic matrix: burn-in length vs support collapse

Status:

```text
launched after freeing /mnt checkpoint quota by deleting 2026-05-11/12 numeric
checkpoint directories while preserving logs, metrics, and the shared
model_only_resume checkpoint.
```

### Storage cleanup ledger

Cleanup policy used on the shared training `/mnt`:

```text
delete:
  May-2026 numeric checkpoint subdirectories and model-only freeze artifacts
  that are not active and are not needed for resume.

preserve:
  April 4-22 ablation baseline.
  April full-PICF baseline.
  Current active A5/A7 2026-05-12 diagnostic run directories.
  model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b.
  Logs, train_tmux logs, metrics JSONL, and launch records.
```

The cleanup does not alter any training conclusion. It only removes replayable
checkpoint payloads after the diagnostic metrics have already been written.
The live-run record below remains authoritative for the current A5/A7 matrix.

### Why same-role support overlap reappeared

The earlier "same-role support fixed" conclusion was only true for short
anchor-isolation or early prefix-stopgrad windows. It was not proven under
longer full action cotrain. The new A5/A7 evidence separates two failures:

```text
Failure A: posterior recycle saturation
  direct unroll=2, burnin=0 drives posterior_recycle_rate -> 1.0 and
  address_update -> 0.0.

Failure B: same-role support assignment collapse
  burnin4/state_only fixes recycle_rate -> 0 and restores address_update,
  but aqr_same_role_support_overlap_max still returns to ~0.999.
```

Therefore `burnin4` is necessary evidence for recurrent carry stability, but it
is not sufficient proof of healthy anchor specialization. The remaining issue is
not "all anchors became one token"; `aqr_effective_anchor_count` remains high.
The issue is that same-role anchors learn nearly identical global support
distributions, so they do not divide evidence into stable object-specific
assignments.

Mathematically, the current action-cotrain objective has a degenerate basin:

```math
q_1,\ldots,q_K \rightarrow \text{same high-action-saliency support}
```

can reduce action loss and local alignment while leaving object identity
ambiguous. The support-diversity loss is currently too weak to dominate this
basin. In the failed tails it reports nonzero loss, but its weighted
contribution is small compared with action/alignment/PV pressure. This explains
why support overlap can be low early and then re-collapse later.

### What the next matrix tests

The matrix is designed to answer four separate causal questions, not to produce
a production checkpoint:

```text
Q1. Is burnin=1 enough to avoid recycle saturation?
Q2. Is burnin=2 enough to avoid recycle saturation while preserving speed?
Q3. If burnin=2 is enough for recycle, can stronger support/geometry diversity
    prevent same-role support collapse during action cotrain?
Q4. If strong diversity works in anchor_only but not all-scope cotrain, is the
    remaining collapse caused by action cotrain rather than support loss form?
```

### Active A5 runs

Machine:

```text
ssh -p 29776 root@36.139.225.68
tmux attach -t a5_burnin_sweep_95ea69b
```

Runs:

```text
picf_a5_u2_b1_semtrain_aux0_500new_20260512_95ea69b
  unroll_steps=2
  burnin_steps=1
  burnin_mode=state_only
  PaliGemma trainable
  Sonata/V-JEPA/AnyTouch frozen
  action_prefix_stopgrad enabled
  OWM predictive aux losses = 0
  support_div=0.05, geom_div=0.02

picf_a5_u2_b2_semtrain_aux0_500new_20260512_95ea69b
  same as above, but burnin_steps=2
```

Tail:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_u2_b1_semtrain_aux0_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_u2_b2_semtrain_aux0_500new_20260512_95ea69b.train_tmux.log
```

### Active A7 runs

Machine:

```text
ssh -p 28060 root@36.139.225.68
tmux attach -t a7_support_collapse_95ea69b
```

Runs:

```text
picf_a7_u2_b2_divstrong_semtrain_aux0_500new_20260512_95ea69b
  unroll_steps=2
  burnin_steps=2
  burnin_mode=state_only
  PaliGemma trainable
  all-scope PICF cotrain
  action_prefix_stopgrad enabled
  OWM predictive aux losses = 0
  support_div=0.25, geom_div=0.05

picf_a7_anchoronly_u2_b2_divstrong_500new_20260512_95ea69b
  same window and diversity settings, but picf_trainable_scope=anchor_only
```

Tail:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_u2_b2_divstrong_semtrain_aux0_500new_20260512_95ea69b.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_anchoronly_u2_b2_divstrong_500new_20260512_95ea69b.train_tmux.log
```

### Acceptance rules

For a recipe to be considered a viable next production candidate, the tail
window must satisfy all of:

```text
posterior_recycle_rate < 0.05
posterior_address_update_rate_mean > 0.02
aqr_same_role_support_overlap_max < 0.85, preferably < 0.70
aqr_same_role_local_true_overlap_max < 0.12
posterior_identity_switch_rate < 0.50, preferably trending down
loss_anchor_pv not stuck near 4.7
loss_mapg_routing not rising monotonically
loss_action_default_equiv decreases without being the only healthy metric
```

Interpretation table:

```text
A5 burnin1 fails, burnin2 passes:
  use unroll=2 + burnin=2 as the speed/stability default candidate.

A5 burnin2 still recycle-collapses:
  burnin=2 is insufficient; burnin4/state_only remains the minimum stable
  recurrent carry candidate.

A7 strong-div all-scope fixes overlap:
  support-diversity weight was the missing stabilizer; follow with a 2k-5k
  candidate run at the same window.

A7 anchor-only fixes overlap but all-scope does not:
  support loss form can work, but action cotrain is still dominating assignment.
  Next step should be staged warmup or selective action-gradient scheduling.

Neither A7 run fixes overlap:
  current support-diversity formulation is not the right anti-collapse term.
  Do not spend 30k on this recipe; implement a direct same-role overlap-max
  penalty or stronger assignment-level competition before long training.
```

### Guarded losses

`slot_jepa`, `support_pred`, and `binding_consistency` stay at zero in this
matrix. The current failure is visible without predictive aux pressure. Opening
these losses now would confound the diagnosis and could reintroduce
permutation-index pressure before support assignment is healthy.

### Mid-run decision, 2026-05-12 11:10 CST

The first two matrix runs produced enough evidence before step 950, so they
should be stopped and archived rather than allowed to consume another 1-2
hours:

```text
A5 picf_a5_u2_b1_semtrain_aux0_500new_20260512_95ea69b
  latest inspected step: 740
  posterior_recycle_rate: ~0
  posterior_address_update_rate_mean: 0.0338
  aqr_same_role_support_overlap_max: 0.9987
  posterior_identity_switch_rate: 0.778
  loss_anchor_pv: 4.752
  loss_action_default_equiv: 0.0641

Interpretation:
  burnin=1 is enough to avoid recycle saturation in this sample, but it is not
  enough to prevent same-role support collapse. Action loss improves while the
  anchor assignment remains unhealthy.

A7 picf_a7_u2_b2_divstrong_semtrain_aux0_500new_20260512_95ea69b
  latest inspected step: 720
  posterior_recycle_rate: 1.0
  posterior_address_update_rate_mean: 0.0
  aqr_same_role_support_overlap_max: 0.9948
  posterior_identity_switch_rate: 0.806
  loss_anchor_pv: 4.607
  loss_action_default_equiv: 0.0850

Interpretation:
  burnin=2 plus stronger support/geometry diversity is not a production
  candidate under all-scope cotrain. It still finds the recycle shortcut and
  same-role support collapse.
```

This is now enough to answer two questions:

```text
Q1. burnin=1 can suppress recycle in one all-scope sample, but it does not solve
    assignment collapse.
Q3. stronger support/geometry diversity alone is not sufficient under all-scope
    cotrain.
```

The useful next runs are therefore the queued counterfactuals:

```text
1. A5 burnin=2 with the normal diversity setting:
   tests whether burnin=2 is more stable than burnin=1 on the same A5 path.

2. A7 anchor_only with burnin=2 and strong diversity:
   tests whether the same support-diversity formulation can work when action
   and all-scope cotrain pressure are removed.
```

If A7 anchor-only is healthy while A7 all-scope failed, the problem is
action/all-scope cotrain pressure dominating assignment. If A7 anchor-only also
collapses, the support-diversity formulation itself is insufficient and should
be replaced by a direct same-role overlap-max or role-wise assignment
competition term before any long run.

### Second-stage observation, 2026-05-12 12:05 CST

The queued counterfactuals also produced early diagnostic evidence:

```text
A5 picf_a5_u2_b2_semtrain_aux0_500new_20260512_95ea69b
  latest inspected step: 560
  posterior_recycle_rate: ~0
  posterior_address_update_rate_mean: 0.0379
  aqr_same_role_support_overlap_max: 0.9824
  aqr_same_role_local_true_overlap_max: 0.0564
  posterior_identity_switch_rate: 0.799
  loss_anchor_pv: 4.647
  loss_mapg_routing: 1.140
  loss_action_default_equiv: 0.0947

Interpretation:
  burnin=2, like burnin=1, can keep recycle off and address updates alive, but
  it still does not create stable same-role support specialization.

A7 picf_a7_anchoronly_u2_b2_divstrong_500new_20260512_95ea69b
  latest inspected step: 680
  posterior_recycle_rate: ~0
  posterior_address_update_rate_mean: 0.0362
  aqr_same_role_support_overlap_max: 0.9995
  aqr_same_role_local_true_overlap_max: 0.0829
  posterior_identity_switch_rate: 0.794
  loss_anchor_pv: 4.733
  loss_mapg_routing: 1.142
  loss_action_default_equiv: 0.1493

Interpretation:
  anchor_only plus strong support/geometry diversity still collapses globally.
  This rules out the simple explanation that all-scope or PaliGemma cotrain
  alone causes the collapse. It does not yet rule out action-loss pressure,
  because action loss still backpropagates into trainable anchor modules in
  `anchor_only`.
```

Metric reading:

```text
posterior_recycle_rate:
  low values prove the recycle shortcut can be suppressed. They do not prove
  identity binding is healthy.

posterior_address_update_rate_mean:
  nonzero ~0.036-0.038 means the address/content pathway is not dead. It also
  does not prove object identity, because all same-role anchors can update while
  still reading the same support.

aqr_effective_anchor_count:
  remaining around 22-23 means anchors are active. The failure is not "all
  anchors disappeared"; it is that active same-role anchors share support.

aqr_same_role_support_overlap_max:
  0.98-0.999 is the core failure. It means at least one same-role pair has
  almost identical global visual support distribution.

aqr_same_role_local_true_overlap_max:
  values around 0.05-0.08 are lower than global overlap, so local weighting is
  not literally identical everywhere. However, combined with local Jaccard near
  1.0, it indicates the same local candidate set is being reused by multiple
  same-role anchors.

loss_anchor_pv:
  staying near 4.6-4.75 indicates the anchor-to-PV structure did not become a
  strong positive signal in these windows.

loss_mapg_routing:
  around 1.1 and not trending cleanly down means assignment/routing health is
  not improving even when action loss improves.

loss_action_default_equiv:
  improving action-equivalent scalar loss is not acceptance evidence here. The
  action path can improve while the anchor supports remain collapsed.
```

The next experiment should therefore stop asking whether `burnin=1/2` or
stronger ordinary support-diversity is enough. It is not enough. The next
counterfactual must isolate the support-diversity objective itself:

```text
E-next-1: anchor_only, no action, support-only
  lambda_action_pos/rot/gripper = 0
  lambda_anchor_pv = 0
  lambda_pv_weak = 0
  lambda_mapg_cycle = 0
  lambda_mapg_support_diversity = 1.0
  lambda_mapg_geometry_diversity = 0.1

Purpose:
  If same-role overlap still stays near 1.0, the current support-diversity
  implementation does not generate enough useful gradient on the priors. The
  next code change must be a direct health-metric-aligned role-wise competition
  or overlap-max loss.

E-next-2: anchor_only, no action, anchor/PV retained
  lambda_action_pos/rot/gripper = 0
  lambda_anchor_pv = 0.25
  lambda_pv_weak = 0.05
  lambda_mapg_cycle = 0.05
  lambda_mapg_support_diversity = 1.0
  lambda_mapg_geometry_diversity = 0.1

Purpose:
  If E-next-1 improves but E-next-2 collapses, PV/alignment pressure is pulling
  same-role anchors back to the same salient support. If both improve, action
  pressure was the missing cause. If both collapse, ordinary support-diversity
  is insufficient regardless of action.
```

### Third-stage isolation launch, 2026-05-12 12:30 CST

The next-stage experiments were launched as no-action counterfactuals. These
runs are not intended to optimize policy quality. They isolate whether the
current anti-collapse objective can move same-role supports when action
gradients are removed.

```text
A5 running:
  exp:
    picf_a5_anchoronly_noaction_supportonly_sdiv1_300new_20260512_95ea69b
  purpose:
    pure objective test for the support-diversity implementation
  resume:
    model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b
  training window:
    step 450 -> 750
  core flags:
    picf_trainable_scope = anchor_only
    unroll_steps = 2
    burnin_steps = 2
    burnin_mode = state_only
    lambda_action_pos/rot/gripper = 0
    lambda_anchor_pv = 0
    lambda_pv_weak = 0
    lambda_mapg_cycle = 0
    lambda_mapg_support_diversity = 1.0
    lambda_mapg_geometry_diversity = 0.1
    lambda_slot_jepa/support_pred/binding_consistency = 0

A7 running:
  exp:
    picf_a7_anchoronly_noaction_anchorpv_sdiv1_300new_20260512_95ea69b
  purpose:
    tests whether PV/anchor alignment, without action, is enough to re-collapse
    same-role supports
  resume:
    model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b
  training window:
    step 450 -> 750
  core flags:
    picf_trainable_scope = anchor_only
    unroll_steps = 2
    burnin_steps = 2
    burnin_mode = state_only
    lambda_action_pos/rot/gripper = 0
    lambda_anchor_pv = 0.25
    lambda_pv_weak = 0.05
    lambda_mapg_cycle = 0.05
    lambda_mapg_support_diversity = 1.0
    lambda_mapg_geometry_diversity = 0.1
    lambda_slot_jepa/support_pred/binding_consistency = 0
```

Decision logic:

```text
Case 1: A5 overlap stays high, A7 overlap stays high
  Conclusion:
    the current support-diversity loss does not sufficiently optimize the
    observed health metric. This is a loss-formulation problem, not an action
    cotrain problem.
  Next action:
    implement a direct same-role overlap-max or role-wise assignment competition
    loss aligned to `aqr_same_role_support_overlap_max`, then rerun the same
    no-action isolation before reintroducing action.

Case 2: A5 overlap improves, A7 overlap stays high
  Conclusion:
    support-diversity can work in isolation, but PV/anchor alignment pulls
    same-role anchors back to the same salient evidence.
  Next action:
    keep the current diversity objective but add a staged schedule:
    support-only warmup -> PV/alignment reintroduction -> action reintroduction.
    Do not open full cotrain from step 0.

Case 3: A5 overlap improves, A7 overlap improves
  Conclusion:
    action gradients are the dominant cause of the support collapse seen in the
    earlier windows.
  Next action:
    use a staged action schedule. Reintroduce action with a small coefficient
    after support specialization is established, and monitor whether overlap
    rebounds.

Case 4: A5 overlap high, A7 overlap improves
  Conclusion:
    unlikely but possible. It would mean the pure support loss lacks a useful
    anchor reference and needs PV/anchor structure to define what should be
    separated.
  Next action:
    keep PV/anchor terms but do not add action until overlap remains healthy for
    a longer window.
```

Primary acceptance metrics for this isolation stage:

```text
aqr_same_role_support_overlap_max:
  must fall materially below the previous 0.98-0.999 band. A useful short-window
  target is <0.85, with <0.75 preferred.

aqr_same_role_local_jaccard_max:
  should also fall if anchors stop reusing the same local candidate set. If
  global overlap falls but Jaccard remains near 1, the model is only reweighting
  the same local supports.

aqr_effective_anchor_count:
  should remain above ~20. A drop in overlap caused by dead anchors is not a
  valid improvement.

posterior_recycle_rate:
  should remain low. If overlap improves only by increasing recycle, the identity
  mechanism is not actually healthier.

loss_mapg_support_diversity:
  should move consistently with the health metric. If the loss decreases while
  the health metric stays collapsed, the loss is misaligned with the metric.

loss_anchor_pv / loss_pv_weak in A7:
  should not dominate the no-action run. If they improve while overlap collapses,
  PV/alignment is competing with object separation.
```

The expected time to a useful readout is about 100-200 new steps after resume.
Full 300-step windows are kept only to verify whether early improvements persist
or rebound.

### Third-stage audit result, 2026-05-12 13:35 CST

Both no-action isolation runs completed. The branch outcome is closer to
`Case 1` with an extra PV-conflict warning:

```text
A5 picf_a5_anchoronly_noaction_supportonly_sdiv1_300new_20260512_95ea69b
  final step: 750
  loss_action: 0.0
  aqr_same_role_support_overlap_max:
    step 460: 0.5252
    step 520: 0.9911
    step 540: 0.9955
    step 640: 0.9996
    step 750: 0.9912
  aqr_same_role_local_jaccard_max:
    step 460: 0.8260
    step 520: 0.9992
    step 750: 1.0000
  aqr_effective_anchor_count:
    step 750: 22.97
  posterior_recycle_rate:
    step 750: ~0
  posterior_address_update_rate_mean:
    step 750: 0.0357
  posterior_identity_switch_rate:
    step 750: 0.8333
  loss_mapg_support_diversity:
    step 750: 0.5308
  preclip_grad_norm:
    step 750: 418.0, clipping active

A7 picf_a7_anchoronly_noaction_anchorpv_sdiv1_300new_20260512_95ea69b
  final step: 750
  loss_action: 0.0
  aqr_same_role_support_overlap_max:
    step 540: 0.9523
    step 560: 0.9081
    step 580: 0.9849
    step 600: 0.9971
    step 750: 0.9998
  aqr_same_role_local_jaccard_max:
    step 750: 1.0000
  aqr_effective_anchor_count:
    step 750: 23.02
  posterior_recycle_rate:
    step 750: ~0
  posterior_address_update_rate_mean:
    step 750: 0.0369
  posterior_identity_switch_rate:
    step 750: 0.7444
  loss_anchor_pv:
    step 750: 4.7299
  loss_pv_weak:
    step 750: 3.2636
  loss_mapg_support_diversity:
    step 750: 0.6490
  preclip_grad_norm:
    step 750: 89.6, clipping active
```

Important correction to the experiment label:

```text
`no-action/support-only` disables action, anchor-PV, PV-weak, and cycle losses,
but it does not disable every base alignment / physical auxiliary term in the
trainer. It is therefore a no-action anti-collapse isolation, not a mathematically
pure single-loss optimization.
```

Interpretation:

```text
1. Action gradients are not the sole cause.
   A5 had all action weights at 0 and still ended at overlap ~0.991.

2. PV/anchor alignment is a conflict amplifier.
   A7 briefly improved to 0.908 at step 560 but returned to ~0.999 after PV and
   cycle terms remained active. PV-weak decreased, yet same-role supports
   collapsed, so PV improvement is not equivalent to object separation.

3. Recycle is not the current failure mode.
   Both runs kept posterior_recycle_rate near 0. The anchors are active and the
   address pathway updates, but same-role anchors still reuse the same support.

4. The effective anchor count is not a sufficient health signal.
   Both runs kept ~23 effective anchors. The failure is not dead anchors; it is
   non-specialized active anchors.

5. Local candidate reuse is the smoking gun.
   `aqr_same_role_local_jaccard_max=1.0` means same-role anchors are using the
   same local candidate sets. The lower true local overlap only means the weights
   within that same candidate set differ; it does not mean the slots have distinct
   object evidence.

6. Simply increasing the existing diversity weight is unlikely to be the clean
   solution.
   The support-diversity loss remains nonzero and gradients are already clipped.
   Raising the scalar weight further risks larger unstable gradients without
   necessarily optimizing the actual health metric.
```

Loss-formulation audit:

```text
Current `_mapg_support_overlap_loss` is real and not a placeholder. It includes:
  same-role pair masking;
  usage weighting;
  a confidence floor so low confidence cannot opt out;
  modality-specific margins;
  active-pair mean pressure;
  top-k worst-pair tail pressure.

However, the observed failure shows that this objective is still not sufficiently
aligned with the runtime health metrics:
  health metric:
    normalized same-role visual prior overlap and local candidate-set Jaccard.
  current loss:
    averaged modality penalties using kernel overlaps and margins.

The code detects collapse but does not force durable specialization of same-role
visual/local supports under the current training distribution.
```

Next required implementation:

```text
Do not start another schedule-only run before changing the anti-collapse
objective. The next version should add a health-metric-aligned role-wise
competition term:

1. direct same-role visual overlap-tail penalty
   Use the same normalized `visual_priors @ visual_priors.T` overlap family that
   produces `aqr_same_role_support_overlap_max`, with top-k same-role tail
   pressure rather than only averaged kernel overlap.

2. soft local candidate-set competition
   Penalize same-role anchors reusing the same local candidate set. The debug
   Jaccard is top-k and non-differentiable, so the training term should use a
   differentiable soft proxy over local support mass or selected priors.

3. role-wise assignment competition
   Add a same-role competition term that encourages same-role anchors to claim
   different high-confidence evidence, while preserving all-role/shared scene
   evidence where appropriate.

4. keep action/PV off for the first validation of this new loss
   Rerun the A5-style no-action isolation first. Only if overlap stays healthy
   should PV and then action be reintroduced.
```

Updated decision:

```text
The previous hypothesis "burn-in plus stronger ordinary diversity may be enough"
is rejected.

The hypothesis "action is the only cause" is rejected.

The hypothesis "PV/anchor terms can amplify collapse" is supported.

The current supported root cause is:
  same-role specialization requires a health-metric-aligned role-wise competition
  objective; current support-diversity is a useful but insufficient proxy.
```

### Health-aligned support objective patch, 2026-05-12 13:55 CST

The next implementation keeps the existing `loss_mapg_support_diversity` family
and changes its internal math. This is intentionally not a new auxiliary module:
the failure was inside the same-role anti-collapse objective, so the correction
belongs in that objective.

Implemented changes:

```text
1. direct same-role visual overlap tail
   Adds a penalty on the same normalized visual-prior overlap used by the runtime
   health metric `aqr_same_role_support_overlap_max`.

2. differentiable local candidate reuse penalty
   Uses `graph.local_token_indices` and `graph.local_priors` to penalize same-role
   anchors placing mass on the same local candidate ids. The term uses a
   Bhattacharyya-style same-candidate overlap, so it remains high when two
   anchors reuse the same local set with different weights. This targets the
   observed failure where local Jaccard reached 1.0 while true local overlap was
   only modest.

3. configurable but conservative weights
   The new terms live under the existing support-diversity lambda:
     mapg_support_div_direct_visual_weight = 1.0
     mapg_support_div_local_candidate_weight = 0.5
     mapg_support_div_local_margin = 0.10
     mapg_support_div_tail_topk = 4

4. no action/PV reintroduction yet
   The first validation must repeat the no-action isolation. If the loss cannot
   keep overlap healthy without action, schedule changes cannot be trusted.
```

Mathematical form:

```text
For normalized visual priors p_i and p_j:

  O_visual(i,j) =
    <p_i, p_j> / sqrt(<p_i,p_i><p_j,p_j>)

  L_visual_tail =
    mean_topk_same_role relu(O_visual - margin_visual)^2

For local candidate rows l_i, l_j and candidate ids c_i, c_j:

  O_local(i,j) =
    sum_{a,b: c_i[a] = c_j[b]} sqrt(l_i[a] l_j[b])

  L_local =
    weighted_same_role_mean relu(O_local - margin_local)^2
    plus top-k same-role tail pressure

The final support-diversity loss is:

  L_support =
    L_existing_kernel_proxy
    + w_visual L_visual_tail
    + w_local L_local
```

Why this is not a patchwork fix:

```text
The change does not add a new head, modality, teacher, or loss family. It only
aligns the already-existing anti-collapse objective with the exact health
metrics that exposed the failure. This preserves the belief-state design:
evidence routing remains in AQR, posterior remains authoritative, and action
remains disabled for first validation.
```

Local validation:

```text
py_compile:
  training.py, picf_core_train.py, training_test.py, picf_core_train_test.py

pytest:
  PYTHONPATH=src pytest -q src/openpi/picf/core/training_test.py -k support_diversity
    3 passed
  PYTHONPATH=src pytest -q scripts/picf_core_train_test.py -k 'loss_config or mapg'
    2 passed

verifiers:
  PYTHONPATH=src python scripts/verify_picf_owm_contract.py
    PASS
  PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
    PASS
```

Next validation runs:

```text
E-fix-1:
  repeat A5 no-action/support-only from the same 450-step resume.
  Acceptance:
    overlap must not rebound to >0.95 by step 750.
    local_jaccard should fall materially below 1.0.
    effective_anchor_count should remain >20.
    recycle should remain near 0.

E-fix-2:
  repeat A7 no-action with anchor/PV retained.
  Acceptance:
    if E-fix-1 is healthy but E-fix-2 collapses, PV/cycle pressure remains a
    conflict and needs staged reintroduction.
    if both are healthy, proceed to small-action reintroduction.
```

### Finite-gradient hardening for local candidate overlap, 2026-05-12 14:10 CST

The first remote validation of the health-aligned support patch exposed a real
numerical issue before any health conclusion could be drawn. A5 failed at the
first optimizer step with a non-finite gradient in the FSDP flat parameter. The
failure happened in a no-action/support-only configuration, so it was not an
action-loss conflict.

Root cause:

```text
The local-candidate reuse term used:

  sqrt(max(l_i[a] l_j[b], 0))

The value is finite at zero, but the derivative of sqrt(x) is singular at
x=0. Sparse local priors commonly contain exact zeros, so a pair with zero
mass on one candidate can produce NaN/Inf gradients even though the forward
loss is finite.
```

Fix:

```text
Use the same epsilon convention as the other normalized overlap terms:

  sqrt(clamp(l_i[a] l_j[b], min=eps))

This is a numerical hardening only. It does not change the loss family, the
belief-state structure, or the intended same-candidate overlap pressure.
```

Why this is mathematically consistent:

```text
The local overlap is a Bhattacharyya-style coefficient over a sparse local
candidate support. The epsilon floor is a smooth finite-gradient extension at
the boundary of the simplex. It prevents undefined boundary derivatives while
preserving the ordering of nonzero overlaps in the operating region.
```

Additional local validation:

```text
PYTHONPATH=src pytest -q src/openpi/picf/core/training_test.py -k support_diversity
  4 passed

New test:
  test_mapg_support_diversity_local_candidate_reuse_has_finite_grad_at_zero_mass

The test constructs exact-zero local priors, backpropagates through the local
candidate reuse penalty, and asserts finite gradients.
```

Remote validation after the finite-gradient fix:

```text
A5:
  picf_a5_anchoronly_noaction_supportfix_sdiv1_300new_20260512_81811b4

A7:
  picf_a7_anchoronly_noaction_anchorpv_supportfix_sdiv1_300new_20260512_81811b4

Both machines passed the same targeted support-diversity pytest through uv
before restart.
```

Early restarted-run observation at step 460:

```text
A5 support-only:
  aqr_same_role_support_overlap_max = 0.4879
  aqr_same_role_local_jaccard_max = 0.7577
  aqr_same_role_local_overlap_max = 0.9892
  aqr_same_role_local_true_overlap_max = 0.0403
  aqr_effective_anchor_count = 23.17
  posterior_recycle_rate = 0.0100
  preclip_grad_norm = 21457.33, clipped to 5.0

A7 anchor/PV no-action:
  aqr_same_role_support_overlap_max = 0.6945
  aqr_same_role_local_jaccard_max = 0.8454
  aqr_same_role_local_overlap_max = 0.9871
  aqr_same_role_local_true_overlap_max = 0.0490
  aqr_effective_anchor_count = 23.09
  posterior_recycle_rate = 0.0100
  preclip_grad_norm = 23763.46, clipped to 5.0
```

Interpretation:

```text
The early step is not acceptance. It does show the finite-gradient issue is
removed and the direct same-role support overlap is initially below the prior
failure zone (>0.95). The remaining risk is whether local candidate reuse and
support overlap rebound by steps 520/600/750. The very large preclip gradient
is expected for the first step after adding a tail pressure term from a migrated
checkpoint, but it means this run must be judged by stability after several
logging intervals, not by step 460 alone.
```

Step-480 monitor update:

```text
A5 support-only:
  aqr_same_role_support_overlap_max = 0.6996
  aqr_same_role_local_jaccard_max = 0.8823
  aqr_same_role_local_overlap_max = 0.9868
  aqr_same_role_local_true_overlap_max = 0.0418
  aqr_effective_anchor_count = 23.21
  posterior_recycle_rate = 3.5e-17
  preclip_grad_norm = 26017.06, clipped to 5.0

A7 anchor/PV no-action:
  aqr_same_role_support_overlap_max = 0.8501
  aqr_same_role_local_jaccard_max = 0.9572
  aqr_same_role_local_overlap_max = 0.9889
  aqr_same_role_local_true_overlap_max = 0.0576
  aqr_effective_anchor_count = 22.86
  posterior_recycle_rate = 3.7e-19
  preclip_grad_norm = 1276.15, clipped to 5.0
```

Step-480 interpretation:

```text
The supportfix objective is not numerically broken and the direct support
overlap is lower than the previous 0.99 collapse in both runs. However, A7 is
already drifting back toward the high-overlap regime while A5 remains only
moderately separated. This is exactly the causal split the experiment was meant
to expose:

  A5 isolates the support objective.
  A7 keeps anchor/PV pressure.

The step-480 gap suggests the PV/anchor terms still pull anchors toward reused
local candidates unless the local-candidate health term remains strong enough.
The high local_overlap_max near 0.99 in both runs means the local candidate
sets are still almost identical; the lower local_true_overlap shows that the
new true-intersection term is not the only failure mode. The remaining issue is
candidate-set reuse/Jaccard pressure, not simple dense support dot-product
overlap.

Acceptance remains pending. The decisive checks are step 520/600/750:

  if A5 and A7 both stay below 0.90, the health-aligned support objective is
  probably sufficient for no-action anchor stabilization;
  if A5 stays moderate but A7 rebounds above 0.95, PV/anchor terms require
  staged reintroduction or stronger anti-reuse pressure;
  if both rebound above 0.95, the loss still does not control the local
  assignment degeneracy and AQR-side candidate construction must be changed.
```

Step-500 monitor update:

```text
A5 support-only:
  aqr_same_role_support_overlap_max = 0.9820
  aqr_same_role_local_jaccard_max = 0.9877
  aqr_same_role_local_overlap_max = 0.9895
  aqr_same_role_local_true_overlap_max = 0.0651
  posterior_recycle_rate = 7.2e-24
  preclip_grad_norm = 4009.14, clipped to 5.0
  loss_total = 4.4493

A7 anchor/PV no-action:
  aqr_same_role_support_overlap_max = 0.8966
  aqr_same_role_local_jaccard_max = 0.9921
  aqr_same_role_local_overlap_max = 0.9920
  aqr_same_role_local_true_overlap_max = 0.0637
  posterior_recycle_rate = 4.6e-26
  preclip_grad_norm = 983.65, clipped to 5.0
  loss_total = 4.8922
```

Step-500 conclusion:

```text
The finite-gradient fix is correct, but the health-aligned support objective is
not sufficient. A5 has no action, no anchor/PV, no cycle, and no predictive
losses, yet same-role support overlap still rebounds to 0.982 by step 500.
Therefore the remaining failure cannot be attributed primarily to action
gradients or PV/cycle pressure.

The decisive common signal is local candidate-set reuse:

  A5 local_jaccard_max = 0.9877
  A7 local_jaccard_max = 0.9921

The true-overlap term remains small, so the anchors are not merely sharing exact
intersecting geometric tokens. They are selecting nearly identical local
candidate sets and then assigning similar dense support inside those sets. The
current support loss penalizes the symptom after candidate construction, but it
does not create enough role-wise competition at the candidate-construction
stage itself.

Mathematically, the failed assumption is:

  penalizing pairwise support overlap after local top-k construction
  is enough to change the top-k candidate sets.

The observed counterexample is:

  top-k candidate sets remain almost identical, so the optimizer can satisfy
  other alignment terms while staying in a high-Jaccard candidate basin.

Next repair should not add another unrelated auxiliary. It should move the same
anti-collapse principle one step earlier into the AQR/local candidate
assignment contract:

  role-wise candidate competition or exclusion during local candidate selection;
  or a direct same-role Jaccard/top-k anti-reuse objective with stronger tail
  pressure and a warm-start-safe gradient;
  then rerun the no-action/support-only isolation before reintroducing PV/action.
```

Step-520 A5 decisive failure:

```text
A5 support-only:
  aqr_same_role_support_overlap_max = 0.9988
  aqr_same_role_local_jaccard_max = 1.0000
  posterior_recycle_rate = 1.1e-26
  loss_total = 4.2220
```

Interpretation:

```text
This is a decisive rejection of the current supportfix objective as a complete
solution. The cleanest no-action/support-only run collapses into identical
local candidate sets by step 520. The failure occurs with recycle off, action
off, PV off, cycle off, and predictive losses off. Therefore the unresolved
mechanism is inside AQR/local candidate construction and role-wise assignment
competition.

Do not continue treating this as a scalar-weight tuning problem. The next
correct repair is structural but still minimal:

  add role-aware or same-role competitive candidate allocation before local
  support aggregation;
  make same-role anchors compete for candidate ownership/top-k slots, not only
  penalize overlap after they have already selected the same candidates;
  keep the no-action/support-only isolation as the first acceptance gate.
```

Step-520 A7 confirmation:

```text
A7 anchor/PV no-action:
  aqr_same_role_support_overlap_max = 0.9541
  aqr_same_role_local_jaccard_max = 1.0000
  posterior_recycle_rate = 5.8e-27
  loss_total = 6.1488
```

Interpretation:

```text
A7 confirms the same failure mode. PV/anchor terms are not required for the
collapse because A5 already failed without them, but they are still unsafe to
reintroduce until candidate ownership is fixed. Both branches now agree on the
root issue:

  same-role anchors converge to the same local candidate set.

This should be treated as an AQR/local-refinement assignment-contract problem,
not a learning-rate, action-loss, or recycle-gate problem.
```

### Role-wise soft local candidate competition, 2026-05-12 17:20 CST

The next code change moves the anti-collapse mechanism before top-k local
candidate truncation, but deliberately avoids hard exclusion or an extra loss.

For local support weights `p_ji` from anchor `j` to candidate token `i`, define
the same-role ownership share:

```text
own_ji = p_ji / sum_{k: role_k = role_j} p_ki
```

and the selection-only score:

```text
score_ji = p_ji * max(floor, own_ji * |role_j group|)^gamma
```

Implementation details:

```text
1. top-k local candidate indices are selected from score_ji;
2. local read vectors and local_priors still use the original p_ji values;
3. cross-role anchors may still reuse the same evidence;
4. if same-role anchors have exactly identical p_ji for all tokens, the
   mechanism preserves symmetry rather than fabricating identity evidence;
5. no posterior state, action target, cache truth, or auxiliary loss is added.
```

This is the Slot-Attention-style competition principle applied to typed local
candidate construction: evidence tokens have limited same-role capacity before
top-k truncation. It is not a "亡羊补牢" scalar loss patch, because it changes the
assignment contract at the point where the failure was observed: local candidate
sets had already become identical before overlap penalties could reliably keep
them apart.

New acceptance runs:

```text
A5 ownerfix:
  support-only, no action/PV/cycle/predictive loss
  expected: local_jaccard no longer returns to 1.0 by step 520/600/750

A7 ownerfix:
  anchor/PV no-action pressure retained
  expected: PV may increase pressure, but same-role overlap should not rebound
  to the >0.95 failure zone if candidate competition is sufficient
```

### Ownerfix launch contract, 2026-05-12 15:40 CST

These runs test the smallest structural repair implied by the previous
supportfix failure. They do not introduce a new loss, do not alter the action
path, and do not write a hard owner assignment into posterior state. The only
changed contract is local candidate selection:

```text
selection_score_ji =
  p_ji * max(floor, |R_j| * p_ji / sum_{k: role_k = role_j} p_ki)^gamma

local_read_ji still uses the original support weight p_ji.
```

This follows the 2025 object-binding probe result that object membership is a
pairwise/subspace relation, not a pointwise class label. The competition is
therefore only allowed to amplify existing same-role preference differences
before top-k truncation. It must not fabricate identity when all evidence rows
are exactly identical.

Started runs:

```text
A5:
  host/session: px-cloud1 / tmux a5_ownerfix_9947b0e
  repo: /root/openpi_ownerfix_9947b0e
  commit: 9947b0e
  exp_name: picf_a5_anchoronly_noaction_ownerfix_sdiv1_300new_20260512_9947b0e
  base args: picf_a5_anchoronly_noaction_supportfix_sdiv1_300new_20260512_81811b4
  role competition: enabled, strength=2.0, floor=0.05

A7:
  host/session: px-cloud2 / tmux a7_ownerfix_9947b0e
  repo: /root/openpi_ownerfix_9947b0e
  commit: 9947b0e
  exp_name: picf_a7_anchoronly_noaction_anchorpv_ownerfix_sdiv1_300new_20260512_9947b0e
  base args: picf_a7_anchoronly_noaction_anchorpv_supportfix_sdiv1_300new_20260512_81811b4
  role competition: enabled, strength=2.0, floor=0.05
```

Acceptance gates:

```text
Primary:
  aqr_same_role_local_jaccard_max must not return to 1.0 by step 520/600/750.

Secondary:
  aqr_same_role_support_overlap_max should stay below the previous failure
  zone. A5 should remain <0.90; A7 may be noisier due to anchor/PV pressure,
  but repeated >0.95 means PV still overwhelms local specialization.

Safety:
  posterior_recycle_rate should remain near zero.
  grad_norm should stay finite.
  aqr_local_role_competition_enabled must be logged as 1.0.
```

Decision table:

```text
A5 passes, A7 passes:
  Candidate construction was the immediate blocker. Reintroduce action in a
  staged run with prefix-stopgrad and low action/PV pressure.

A5 passes, A7 fails:
  Candidate construction is fixed, but anchor/PV terms still pull same-role
  anchors back together. Keep ownerfix and stage PV/action more slowly.

A5 fails:
  The issue is not repairable by soft same-role selection alone. Do not increase
  scalar diversity loss again. Move to evidence/seed-level changes:
    anchor-specific coverage/proposal seeds,
    real tracklet/proposal dataflow,
    or a dedicated offline IsSameObject probe to verify token separability.

Both fail:
  The current typed memory rows are effectively indistinguishable for same-role
  anchors under this setup. Treat this as an evidence identifiability problem,
  not a loss-weight tuning problem.
```

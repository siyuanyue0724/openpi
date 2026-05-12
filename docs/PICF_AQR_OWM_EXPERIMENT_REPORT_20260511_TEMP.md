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

### Ownerfix step-520 result, 2026-05-12 15:59 CST

The ownerfix runs were stopped at step 520 because the primary acceptance gate
failed on both machines.

```text
A5 ownerfix, support-only:
  step 460:
    same_role_support_overlap = 0.6008
    same_role_local_jaccard   = 0.7529
    posterior_recycle_rate    = 0.9100
    preclip_grad_norm         = 1047.1

  step 480:
    same_role_support_overlap = 0.9830
    same_role_local_jaccard   = 0.9921
    posterior_recycle_rate    = 1.0000

  step 520:
    same_role_support_overlap = 0.9992
    same_role_local_jaccard   = 1.0000
    posterior_recycle_rate    = 1.0000
    preclip_grad_norm         = 512.9

A7 ownerfix, anchor/PV retained:
  step 460:
    same_role_support_overlap = 0.6566
    same_role_local_jaccard   = 0.7680
    posterior_recycle_rate    = 0.9100
    preclip_grad_norm         = 798.3

  step 480:
    same_role_support_overlap = 0.9668
    same_role_local_jaccard   = 0.9909
    posterior_recycle_rate    = 1.0000

  step 520:
    same_role_support_overlap = 0.9987
    same_role_local_jaccard   = 1.0000
    posterior_recycle_rate    = 1.0000
    preclip_grad_norm         = 1098.3
```

Conclusion:

```text
Rejected:
  soft role-wise candidate competition alone is sufficient.

Supported:
  the failure is deeper than PV/action pressure and deeper than scalar support
  diversity loss. Even in the support-only A5 isolation run, same-role local
  candidate sets return to exact reuse by step 500.

Important nuance:
  step 460 shows the mechanism can temporarily reduce candidate overlap. The
  later rebound means the evidence rows or anchor seeds do not provide stable
  enough asymmetry for the competition to preserve.
```

Mathematical diagnosis:

```text
The ownerfix score was:

  score_ji =
    p_ji * max(floor, |R_j| * p_ji / sum_{k in same-role} p_ki)^gamma

This only amplifies existing differences in p_ji. If same-role rows converge
to p_1i ~= p_2i ~= ... over the relevant local candidates, then:

  |R| * p_ji / sum_k p_ki ~= 1

and the competition becomes a row-wise rescaling that cannot change top-k
membership. Therefore, once upstream AQR support rows collapse, role-wise
competition has no independent source of identity evidence.
```

Next repair must not be "increase gamma" or "increase diversity lambda." Those
would be scalar pressure on an already-identifiability-limited signal. The next
repair should introduce a principled source of anchor-specific evidence before
local top-k truncation:

```text
1. coverage/proposal seeded local candidates:
   use the existing aqr_coverage_codes, geometry bins, view/time bins, and
   projective candidates to seed distinct candidate neighborhoods per same-role
   anchor before support rows collapse.

2. offline IsSameObject audit:
   verify whether V-JEPA/PaliGemma/static/wrist tokens contain separable
   same-object subspace on weak robot labels. If the probe fails, local
   candidate separation cannot be expected from AQR alone.

3. real tracklet/proposal dataflow:
   if weak track/proposal evidence is available, feed it into training rather
   than leaving owm_tracklet_tokens=0 and owm_proposal_tokens=0.
```

Immediate action:

```text
Both ownerfix runs were stopped at step 520 to save GPU time.
Do not continue this branch as a long run.
```

## Coverage-Seeded Local Proposal, 2026-05-12

### Why this is the next test

The ownerfix failure rules out a purely scalar or purely competitive repair.
The failed mechanism can only amplify differences that already exist in
`p_ji`. Once same-role support rows become nearly identical, the role-wise
capacity factor becomes constant and cannot change the local top-k set.

The next intervention therefore moves one level earlier in the belief-state
pipeline: candidate proposal before local top-k truncation. This is not a new
posterior truth, not a hard tiling rule, and not an auxiliary loss. It supplies
a weak, deterministic, anchor-specific reference prior using the existing
low-discrepancy `aqr_coverage_codes`.

This matches three independent principles:

```text
1. Pairwise object-binding papers show object membership is separable in a
   low-dimensional same-object subspace, but the grouping signal is fragile and
   should be audited/probed rather than assumed from raw cosine similarity.

2. Deformable/anchor-query detection designs use reference points or denoising
   anchors because query-based routing needs stable candidate domains before
   global attention or action gradients can shape the final representation.

3. V-JEPA/TraceVLA-style temporal evidence can improve state awareness only if
   the router preserves candidate diversity long enough for the posterior to
   compare evidence. If all same-role anchors reread the same local candidates,
   the downstream belief filter cannot recover object identity.
```

### Mathematical definition

For anchor `j` and typed-memory token `i`, let:

```text
p_ji:
  original AQR support probability for local refinement.

u_j in [-1, 1]^2:
  deterministic anchor coverage coordinate from `aqr_coverage_codes`.

x_i in [-1, 1]^2:
  token coordinate in its own local view/grid frame.
```

The proposal score is:

```text
seed_ji = exp(-||u_j - x_i||^2 / (2 sigma^2))
select_ji = normalize_i p_ji * (1 + w_seed * seed_ji)
```

The local read remains:

```text
local_j = sum_{i in TopK(select_j)} p_ji * token_i
```

The important invariant is:

```text
selection uses the seeded proposal score;
value aggregation uses the original support weights.
```

So the seed can break identical local candidate sets, but it cannot directly
overwrite posterior evidence. If the original support assigns no mass to a
candidate, local read still receives no useful value from that candidate.

### Why this is not a hard patch

This is structurally different from increasing diversity loss:

```text
diversity loss:
  tries to punish collapse after it has already happened.

role competition:
  amplifies existing row differences, but has no independent anchor signal.

coverage-seeded proposal:
  gives each anchor a weak reference domain before top-k truncation, while
  preserving evidence-weighted local read.
```

The test is falsifiable. If this fails, the diagnosis becomes:

```text
local candidate identity cannot be recovered from current AQR support rows plus
weak anchor coverage. The next step must be an offline IsSameObject probe or
real tracklet/proposal dataflow, not more scalar loss tuning.
```

### Code contract

```text
Default production:
  local_refinement_role_competition_enabled = False
  local_refinement_coverage_seed_enabled = False

Coverage-seed diagnostic:
  local_refinement_role_competition_enabled = False
  local_refinement_coverage_seed_enabled = True
  local_refinement_coverage_seed_strength = 0.75
  local_refinement_coverage_seed_sigma = 0.75
```

The default stays off because the ownerfix result proved this family still
requires acceptance evidence. The diagnostic run explicitly enables it.

### Two-machine diagnostic plan

Run both diagnostics from the shared step-450 prefix-stopgrad checkpoint:

```text
A5 / support-only:
  exp_name:
    picf_a5_anchoronly_noaction_covseed_sdiv1_300new_20260512_<sha>
  purpose:
    isolate whether seeded local candidates can preserve same-role candidate
    diversity without anchor/PV pressure.

A7 / anchor/PV retained:
  exp_name:
    picf_a7_anchoronly_noaction_anchorpv_covseed_sdiv1_300new_20260512_<sha>
  purpose:
    test whether the same candidate fix survives the PV/anchor terms that were
    present in the previous A7 diagnostic.
```

Both runs should continue to step 750 unless the early failure gate triggers.
The first actionable signal should appear by step 520, usually within 10-15
minutes from launch on the current A100 nodes.

### Acceptance gates

Primary:

```text
aqr_same_role_local_jaccard_max:
  must not return to 1.0 by step 520.
  target <= 0.90; strong pass <= 0.80.
```

Secondary:

```text
aqr_same_role_support_overlap_max:
  A5 target < 0.90 by step 520.
  A7 target < 0.95 by step 520.

posterior_recycle_rate:
  must not saturate near 1.0 after step 480.

preclip_grad_norm:
  must not repeatedly exceed 100 or show clip-dominated dynamics.
```

Instrumentation gates:

```text
aqr_local_coverage_seed_enabled must log as 1.0.
aqr_local_role_competition_enabled must log as 0.0.
aqr_local_coverage_seed_strength must log as 0.75.
aqr_local_coverage_seed_sigma must log as 0.75.
```

### Decision table after 520-750

```text
A5 pass, A7 pass:
  Candidate proposal was the missing mechanism. Proceed to staged cotrain with
  action/PV pressure reintroduced slowly.

A5 pass, A7 fail:
  Candidate proposal works in isolation, but PV/anchor terms still pull anchors
  together. Stage PV/action pressure or lower anchor/PV terms before cotrain.

A5 fail:
  Candidate proposal is not enough. Stop scalar tuning. Run offline
  IsSameObject probe and/or feed real tracklet/proposal evidence into CALVIN.

Both fail with recycle saturation:
  The posterior update/recycle gate is reacting to unstable local evidence.
  Do not open predictive losses or action cotrain until identity/recycle
  instrumentation is corrected.
```

### Coverage-seed result, 2026-05-12 17:55 CST

Both coverage-seeded diagnostics completed to step 750. The diagnostic was
enabled as intended (`local_refinement_coverage_seed_enabled=1.0`) and
role-wise local competition stayed disabled. This makes the result a valid test
of weak anchor-specific proposal seeding, not a mixed ownerfix/coverage run.

Final status:

```text
A5 support-only coverage seed:
  step = 750
  aqr_same_role_support_overlap_max = 0.99936
  aqr_same_role_local_jaccard_max = 1.00000
  posterior_recycle_rate = 1.00000
  posterior_identity_switch_rate = 0.83889
  posterior_address_update_rate_mean = 0.00000
  loss_slot_jepa = 88.98
  preclip_grad_norm = 107.98
  owm_tracklet_tokens = 0
  owm_proposal_tokens = 0

A7 anchor/PV coverage seed:
  step = 750
  aqr_same_role_support_overlap_max = 0.99933
  aqr_same_role_local_jaccard_max = 1.00000
  posterior_recycle_rate = 0.00000
  posterior_identity_switch_rate = 0.75556
  posterior_address_update_rate_mean = 0.03636
  loss_slot_jepa = 4.38
  preclip_grad_norm = 212.43
  owm_tracklet_tokens = 0
  owm_proposal_tokens = 0
```

Interpretation:

```text
1. Coverage seeding works as an early proposal asymmetry but does not prevent
   same-role local candidate-set collapse. Both A5 and A7 returned to
   local_jaccard=1.0 and support_overlap≈0.999.

2. The failure is not primarily action loss, PV pressure, role competition, or
   burn-in length. A5 had no action/PV pressure and still collapsed.

3. The failure is not only recycle saturation. A5 collapsed with recycle=1.0;
   A7 collapsed with recycle=0.0.

4. Tracklet/proposal evidence remains inactive in the current CALVIN dataflow
   (`owm_tracklet_tokens=0`, `owm_proposal_tokens=0`), so MVTrack cannot yet
   rely on those branches for object separation.
```

Mathematical conclusion:

```text
The tested mechanisms try to preserve candidate diversity using scalar losses,
role-wise competition, or deterministic reference proposals. All three fail
once same-role evidence rows become indistinguishable. The next falsifiable
question is whether current typed memory contains an exploitable pairwise
same-object subspace at all.
```

This follows the object-binding probe literature: object membership is better
tested as a pairwise/quadratic relation between token embeddings than as a raw
pointwise cosine or a hard slot label. If such a subspace is present, binding
should use it before local candidate selection and posterior correction. If it
is absent, further scalar tuning of AQR losses cannot create the missing object
information.

## Pairwise Binding-Subspace Diagnostic, 2026-05-12 18:05 CST

### Purpose

The next one-hour diagnostic isolates whether the deployed pairwise/support
binding-signature path is causally useful. It does not introduce a new loss, new
truth source, or hard slot assignment. It changes only how strongly the existing
same-object subspace contributes to posterior binding.

Run both experiments from the shared step-450 prefix-stopgrad checkpoint and use
the same A5 support-only base configuration to avoid A5/A7 base-config
confounds.

### A5 negative control: pairwise binding off

```text
exp_name:
  picf_a5_anchoronly_noaction_pairbind_off_200new_20260512_41cf317

override:
  bind_support_signature_weight = 0.0
  bind_embedding_signature_weight = 0.0
  bind_address_weight = 0.0
  local_refinement_role_competition_enabled = false
  local_refinement_coverage_seed_enabled = false
  num_train_steps = 650
```

This removes the object-binding paper inspired term. If this run behaves the
same as the positive run, then the current binding-signature path is not the
dominant missing mechanism.

### A7 positive test: pairwise binding emphasized, address kept weak

```text
exp_name:
  picf_a7_anchoronly_noaction_pairbind_strong_200new_20260512_41cf317

override:
  bind_support_signature_weight = 1.25
  bind_embedding_signature_weight = 0.75
  bind_address_weight = 0.10
  bind_address_innovation_downweight = 2.0
  local_refinement_role_competition_enabled = false
  local_refinement_coverage_seed_enabled = false
  num_train_steps = 650
```

The address term stays weak because address inertia can lock in false identity
when support evidence is unstable. The intended test is pairwise support
subspace, not hard address identity.

### Acceptance gates by step 650

```text
Strong evidence for pairwise binding:
  A7 support_overlap and local_jaccard stay materially below A5.
  Target: A7 aqr_same_role_support_overlap_max < 0.90 and
          aqr_same_role_local_jaccard_max < 0.90.

Weak or negative evidence:
  both runs return to local_jaccard≈1.0 and support_overlap≈0.999.
  In this case, stop more loss/weight tuning and run an offline IsSameObject
  token-probe or feed real tracklet/proposal evidence.

Safety gates:
  posterior_recycle_rate must not saturate near 1.0;
  preclip_grad_norm should not remain clip-dominated;
  owm_tracklet_tokens/proposal_tokens are expected to remain 0 in this dataflow.
```

### Pairwise binding-subspace early-stop result, 2026-05-12 18:25 CST

The pairwise binding-subspace diagnostic was stopped at step 500 because both
branches had already failed the primary local-candidate reuse gate. This is a
valid early stop: the target was to keep `aqr_same_role_local_jaccard_max` below
the high-reuse regime, not to optimize final policy quality.

Result:

```text
A5 negative control, binding signature/address off:
  step 460:
    aqr_same_role_support_overlap_max = 0.7611
    aqr_same_role_local_jaccard_max = 0.8174
    posterior_recycle_rate = 0.0100
    preclip_grad_norm = 4745.9
  step 480:
    aqr_same_role_support_overlap_max = 0.9571
    aqr_same_role_local_jaccard_max = 0.9808
    posterior_recycle_rate ~= 0
  step 500:
    aqr_same_role_support_overlap_max = 0.9944
    aqr_same_role_local_jaccard_max = 0.9997
    posterior_recycle_rate ~= 0

A7 positive test, support/binding signature emphasized and address weak:
  step 460:
    aqr_same_role_support_overlap_max = 0.7197
    aqr_same_role_local_jaccard_max = 0.8477
    posterior_recycle_rate = 0.0100
    preclip_grad_norm = 17698.1
  step 480:
    aqr_same_role_support_overlap_max = 0.9956
    aqr_same_role_local_jaccard_max = 0.9992
    posterior_recycle_rate ~= 0
  step 500:
    aqr_same_role_support_overlap_max = 0.9987
    aqr_same_role_local_jaccard_max = 1.0000
    posterior_recycle_rate ~= 0
```

Interpretation:

```text
1. The current binding-signature path is not sufficient to prevent same-role
   local candidate reuse. Strengthening it did not keep A7 healthy.

2. Turning it off also fails quickly, so the old short healthy run was not a
   proof that all later anchor configurations would remain healthy.

3. The result does not mean the object-binding paper direction is wrong. It
   means the currently deployed support-weighted signature is only a weak
   approximation of a real IsSameObject probe/subspace. It is not enough when
   local candidate rows have already become nearly identical.

4. The failure is again not recycle saturation: both branches collapsed while
   `posterior_recycle_rate` was near zero.
```

Decision:

```text
Stop this branch. Do not keep running pairwise-weight sweeps.

Production/default should keep:
  - the moderate binding-signature path as a low-cost structural prior;
  - local refinement as an instrumented evidence path;
  - support-diversity at conservative weight.

Production/default should not enable:
  - role-wise local candidate competition;
  - coverage-seeded local proposal;
  - strong binding-signature weights;
  - strong address inertia;
  - high-risk predictive losses.

Next useful experiment:
  either run an offline IsSameObject token probe, or feed real tracklet/proposal
  evidence into CALVIN. More scalar/local-candidate heuristics are not expected
  to be clean or decisive.
```

---

## 2026-05-12 Production Cleanup and Theory Reconciliation

Decision:

```text
Remove from active production code:
  - local_refinement_role_competition_*
  - local_refinement_coverage_seed_*

Keep in active production code:
  - evidence-weighted local refinement over existing typed memories
  - moderate binding_signature support/address prior
  - support-diversity and geometry-diversity losses at conservative weights
  - guarded OWM predictive hooks with default lambda=0

Do not add new scalar heuristics until one of these is true:
  - offline IsSameObject probe identifies a real object-binding subspace;
  - real tracklet/proposal data enters the train/replay/serve dataflow;
  - a failure case proves a missing term with a falsifiable metric.
```

Code cleanup performed:

```text
1. Removed role-competition and coverage-seed fields from PicfCoreConfig.
2. Removed their train CLI arguments and startup/debug metric keys.
3. Removed their helper functions and tests from pipeline/pipeline_test.
4. Removed verifier/strict/dataflow/deep-audit requirements that treated them
   as part of the current contract.
5. Updated README_v2.2 so these are historical rejected diagnostics only.
```

Why removal is mathematically cleaner:

```math
Local refinement should approximate

  \Omega_j =
    TopK(p^v_j)
    \cup TopK(p^{temp}_j)
    \cup TopK(p^{point}_j)
    \cup TopK(p^{track}_j)
    \cup TopK(p^{prop}_j)

and then reread typed evidence inside \Omega_j.
```

The rejected role/coverage heuristics changed the candidate set by hand:

```math
score_{j,i}
  =
  p_{j,i} \cdot h(j,i)
```

where `h(j,i)` was not learned from current evidence and not tied to posterior
belief quality. The experiments showed the failure mode directly: when same-role
AQR rows become nearly identical, deterministic role/coverage factors cannot
recover object identity. Keeping those knobs would preserve a false sense of
control while adding operator burden and semantic noise.

Paper reconciliation:

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
  - Supports probing pairwise IsSameObject information in pretrained ViT tokens.
  - It does not support hand-coded role/coverage candidate ownership after rows
    have already collapsed.
  - Our binding_signature path is the coherent structural approximation; the
    missing rigorous step is an offline IsSameObject probe/audit.

V-JEPA 2 / VLA-JEPA:
  - Support temporal latent evidence and leakage-free future targets.
  - They support typed temporal memory and guarded predictive hooks, not
    strong default JEPA losses before identity is stable.

OA-WAM:
  - Supports persistent address + time-varying content.
  - It argues for address-aware binding/cache, but address must remain gated by
    current evidence, recycle, and innovation; it must not become a hard lock.

WristWorld / MLA / RoboGround / CoTracker3:
  - Support wrist/multisensory/grounded/tracklet evidence when available.
  - They do not justify pretending absent tracklet/proposal data is active.
```

Current clean next steps:

```text
1. Run an offline IsSameObject probe over V-JEPA/static/wrist tokens using weak
   labels from posterior continuity, point neighborhoods, contact neighborhoods,
   PG support overlap, and later tracklets when available.

2. Feed real tracklet/proposal fields through CALVIN train/replay/serve only
   when those files exist; missing fields must stay no-op.

3. Continue staged cotrain diagnostics with:
   action_prefix_stopgrad warmup,
   conservative support/geometry diversity,
   moderate binding_signature,
   no strong address inertia,
   OWM predictive lambdas default 0.

4. Treat high same-role overlap and local Jaccard as primary failure metrics.
   Action loss alone is not sufficient because the model can reduce action loss
   while collapsing object support.
```

---

## 2026-05-12 Next Experiment Plan After Production Cleanup

Current verified code state:

```text
HEAD = 146bb86
Rejected role/coverage local candidate heuristics are removed from active code.
README_v2.2 records the cleaned two-branch diagnostic deployment.
verify_picf_owm_contract.py = PASS
picf_owm_strict_diagnose.py --fail-on-fail = PASS
picf_owm_dataflow_trace.py --fail-on-fail = PASS
picf_owm_mvtrack_deep_audit.py --fail-on-fail = PASS
selected pipeline/training pytest = PASS
```

Primary diagnosis:

```text
The current failure is not explained by action alone, PV alone, recycle
saturation alone, role competition absence, coverage priors, or stronger
binding-signature weights. Multiple branches collapse through the same late
failure mode:

  aqr_same_role_local_jaccard_max -> 1.0
  aqr_same_role_support_overlap_max -> 0.99+

This indicates that same-role AQR rows become nearly identical before local
refinement can preserve distinct evidence. Once this happens, candidate
selection heuristics cannot recover identity.
```

Mathematical model of the failure:

```math
p_{j,i}^{(m)}
  =
  softmax_i(q_j^T k_i^{(m)} + b_{j,i}^{(m)})

\Omega_j
  =
  TopK_i(p_{j,i}^{(m)})

Collapse condition:

  p_{j,\cdot}^{(m)} \approx p_{k,\cdot}^{(m)}
  for same-role j,k

then

  Jaccard(\Omega_j,\Omega_k) \to 1

and any post-hoc candidate multiplier h(j,i) gives

  TopK(p_{j,i} h(j,i))

without creating new object information if p_{j,\cdot} and p_{k,\cdot} are
already indistinguishable.
```

Therefore, the next experiments must test whether distinct object information
exists before/inside AQR, not add more post-hoc local candidate heuristics.

### Experiment 1: Clean staged cotrain baseline

Purpose:

```text
Test whether the cleaned production path can keep anchor supports distinct when
using the previously healthiest training shape:

  action_prefix_stopgrad warmup
  conservative support/geometry diversity
  moderate binding_signature
  no strong address inertia
  predictive losses still zero
```

Run shape:

```text
resume from the common 450-step anchor checkpoint
unroll_steps = 1
burnin_steps = 4
burnin_mode = state_only
foundation backbones on
freeze Sonata / V-JEPA / AnyTouch
train PaliGemma if memory allows
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
```

Acceptance gates:

```text
At 500-750 diagnostic steps:
  aqr_same_role_local_jaccard_max should not stay > 0.98 for 3 logs.
  aqr_same_role_support_overlap_max should not stay > 0.95 for 3 logs.
  posterior_recycle_rate should not saturate near 1.0.
  grad_norm must stay finite.
  loss_action_default_equiv may fall, but it cannot override anchor-health fail.
```

Interpretation:

```text
Pass:
  cleaned production path is viable for a longer 2500-step diagnostic.

Fail:
  local collapse is upstream of current local refinement and must be diagnosed
  with an IsSameObject probe or real tracklet/proposal evidence.
```

### Experiment 2: Direct cotrain control

Purpose:

```text
Test whether staged warmup is actually necessary or whether action gradients can
co-train without destroying the object-binding subspace.
```

Only difference from Experiment 1:

```text
action_prefix_stopgrad disabled from the beginning
same resume / same data / same batch regime
```

Critical comparison:

```text
If direct cotrain reduces loss_action_default_equiv faster but local_jaccard
goes to 1.0 earlier, action is not the sole cause but it is a destabilizer.

If both staged and direct collapse at similar step, the issue is upstream AQR
evidence separability.

If direct remains healthy, staged warmup can be shortened or removed.
```

### Experiment 3: Offline IsSameObject probe

Purpose:

```text
Follow the object-binding ViT result directly: test whether pretrained
V-JEPA/static/wrist tokens encode pairwise same-object information before we
ask AQR to bind them.
```

Weak labels, no dataset annotation change:

```text
positive pairs:
  same high-confidence posterior slot across adjacent frames
  close point-cloud neighborhood with consistent motion
  same contact neighborhood
  high PG support overlap

negative pairs:
  different high-confidence posterior slots
  far point neighborhoods
  incompatible contact/role support
```

Probe:

```math
score(i,k) = z_i^T W z_k

Metrics:
  AUC / accuracy / calibration by layer, view, and modality.
```

Decision:

```text
Good probe:
  train or freeze a small pairwise binding subspace and feed it into binding
  logits as evidence.

Bad probe:
  do not keep increasing binding weights. The tokens do not expose enough
  object signal under current data/backbone path.
```

### Experiment 4: Real tracklet/proposal dataflow activation

Purpose:

```text
Test the MVTrack branch honestly. Current code supports tracklet/proposal
fields, but if CALVIN episodes do not contain those fields, the branch is a
safe no-op and must not be claimed as active.
```

Minimal valid test:

```text
Prepare a small subset with tracklet_xy / visibility / confidence / ids /
view_ids / age, or proposal_centers_xy / boxes / objectness.

Run the same clean staged profile.
Require:
  owm_tracklet_tokens > 0 or owm_proposal_tokens > 0
  aqr_tracklet_support_entropy or aqr_proposal_support_entropy logged
  same-role local_jaccard improves without action loss regression
```

Decision:

```text
If tracklet/proposal data improves anchor health, this is the clean path forward.
If not, tracklet/proposal should remain optional evidence, not default claims.
```

### Experiment 5: Predictive auxiliary delayed activation

Purpose:

```text
Do not open slot-JEPA/support/binding losses during unstable identity formation.
Only test them after Experiment 1 passes anchor-health gates.
```

Schedule:

```text
steps 0-750:
  all predictive lambdas = 0

after anchor health passes:
  lambda_slot_jepa = 1e-4
  lambda_support_pred = 1e-4
  lambda_binding_consistency = 1e-4

hard stop if:
  local_jaccard rises above 0.98,
  recycle saturates,
  or loss_slot_jepa becomes monotonic divergent.
```

Rationale:

```text
Predictive losses are mathematically valid only when slot identity is stable.
Before that, even permutation-tolerant matching can amplify wrong latent
assignments if all same-role supports are already identical.
```

Final priority order:

```text
1. Clean staged cotrain baseline.
2. Direct cotrain control.
3. Offline IsSameObject probe.
4. Real tracklet/proposal activation.
5. Delayed predictive auxiliary cotrain only after anchor health is stable.
```

---

## 2026-05-12 Local Audit After Clean Diagnostic Deployment

Source index:

```text
README entry:
  src/openpi/picf/README_v2.2.md

Detailed experiment ledger:
  docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md

MVTrack architecture contract:
  docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md
```

Local repository state:

```text
branch = Posterior_VLA
audited deployment base HEAD = 146bb86
previous code-cleanup commit = 145654a
previous experiment-plan commit = 7bff430
working tree status before this audit note = clean
```

Local verification commands run from `/home/siyuanyue/Documents/openpi`:

```bash
PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail

python -m py_compile \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  src/openpi/picf/vjepa/wrapper.py \
  scripts/picf_core_train.py \
  scripts/serve_picf_policy.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_owm_strict_diagnose.py \
  scripts/picf_owm_dataflow_trace.py \
  scripts/picf_owm_mvtrack_deep_audit.py \
  scripts/picf_owm_evidence_bundle.py

PYTHONPATH=src pytest -q \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py \
  src/openpi/picf/vjepa/wrapper_test.py

PYTHONPATH=src pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  -k 'cache or burnin or mvtrack or temporal or pg or ordinal or local or binding'

PYTHONPATH=src pytest -q \
  src/openpi/picf/core/training_test.py \
  -k 'slot or support or binding or jepa or denois or matched or next_posterior'

PYTHONPATH=src pytest -q \
  scripts/picf_core_train_test.py \
  scripts/serve_picf_policy_test.py \
  scripts/picf_replay_windows_test.py \
  scripts/picf_loss_audit_test.py \
  scripts/picf_watch_metrics_test.py \
  scripts/picf_plot_metrics_test.py \
  scripts/picf_resume_train_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py

PYTHONPATH=src pytest -q \
  src/openpi/picf/action_normalization_test.py \
  src/openpi/picf/pointcloud_picf_test.py \
  src/openpi/picf/policy_test.py \
  src/openpi/picf/replay/calvin_replay_test.py \
  src/openpi/picf/scaffold/matching_test.py \
  src/openpi/picf/scaffold/pipeline_test.py \
  src/openpi/picf/posterior/fusion_test.py \
  src/openpi/picf/posterior/pipeline_test.py \
  src/openpi/picf/posterior/prior_test.py

PYTHONPATH=src pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py
```

Observed result:

```text
OWM verifier:
  PASS

strict diagnose:
  PASS with expected runtime-artifact WARN boundaries when no metrics/eval
  path is supplied.

recursive dataflow trace:
  PASS

MVTrack deep audit:
  PASS

py_compile:
  PASS

targeted script/V-JEPA tests:
  13 passed

targeted pipeline tests:
  16 passed

targeted training tests:
  8 passed

broader script regression:
  initially exposed 3 stale test-stub failures in
  scripts/picf_core_train_test.py because monkeypatched dummy losses did not
  carry the current action comparability fields:
    action_default_equiv
    action_weight_scale

  The test stubs were updated; rerun result:
    162 passed

broader non-core PICF regression:
  31 passed

full core pipeline/training regression:
  101 passed

final combined local audit regression:
  232 passed
```

Boundary conditions:

```text
This local audit proves code-level contracts, parser/compile health, guarded
dataflow invariants, and selected regression coverage.

It does not prove:
  A5/A7 live training acceptance,
  CALVIN behavior success,
  video/overlay quality,
  active tracklet/proposal evidence on datasets that do not provide those
  fields,
  or final ordinal/fine-instance grounding.
```

Strict interpretation:

```text
If local audit passes but live metrics fail, do not add scalar losses first.
Use the failure gate:

  local_jaccard -> upstream same-role evidence reuse.
  same_role_support_overlap -> row-level AQR support collapse.
  recycle_rate/logit -> posterior identity/reset instability.
  loss_action_default_equiv -> secondary action comparability only.

The next clean branches remain:
  clean staged cotrain,
  direct cotrain control,
  offline IsSameObject probe,
  real tracklet/proposal dataflow activation,
  delayed predictive auxiliary only after anchor-health passes.
```

## 2026-05-12 Stricter Multi-Layer Local Audit

This pass was run because the previous local audit was too narrow. The goal was
not to add another mechanism. The goal was to check whether the current
README-to-code contract, deleted/guarded paths, tests, and runtime diagnostics
are mutually consistent before interpreting cloud loss curves.

Local repository state:

```text
branch: Posterior_VLA
starting HEAD for this stricter audit: 98fe770
working tree at audit start: clean
```

Layer 1, README and stale-path audit:

```text
repo README:
  routes to src/openpi/picf/README_v2.2.md

PICF README:
  routes to README_v2.2.md

README_v2.2:
  routes to MVTrack deployment README and this experiment report.

Removed or rejected active-code paths:
  posterior_address_drift_mean: absent from active code
  aqr_temporal_memory_tokens: absent from active code
  ordinal_confidence_threshold: absent from active code
  lambda_cross_modal_align: absent from active loss/config code
  lambda_innovation_calib: absent from active loss/config code
  rejected role-competition / coverage-seed local-candidate heuristics:
    absent from active src/scripts code, retained only as historical experiment
    notes and absence assertions in audit scripts.

Intentional explicit no-op / unsupported paths:
  picf_augmentation_mode=multimodal_geometry:
    raises NotImplementedError instead of silently running an unimplemented
    geometry augmentation path.
  SAM/DINO/proposal generation:
    not generated inside this pass; proposal tensors are consumed only if an
    upstream source supplies them.
```

Layer 2, key invariants checked by code search and scripts:

```text
action comparability:
  loss_action_default_equiv and action_weight_scale exist.

action/PICF gradient isolation:
  picf_action_prefix_stopgrad exists.

binding subspace:
  binding_signature_proj and support-weighted binding signatures exist.

cache:
  PicfCacheReadState exists.
  latest immediate posterior cache row is skipped.
  cache read remains residual-gated through q_before_cache.

future target leakage:
  future.posterior_tokens.detach() is present in prediction losses.

optional dataflow:
  tracklet and proposal observation fields are present and optional.
  missing tracklet/proposal data remains a valid no-op.

multiview:
  rgb_gripper and view_ids are present in the temporal visual path.

ordinal:
  ordinal state remains prompt-gated diagnostic / weak target; it is not a
  posterior rewrite.
```

Layer 3, commands and results:

```bash
python -m py_compile $(rg --files src/openpi/picf scripts | rg '\.py$' | tr '\n' ' ')

PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail

PYTHONPATH=src pytest -q $(cat /tmp/picf_test_files_no_train_test.txt)
PYTHONPATH=src pytest -q scripts/train_test.py
```

Observed result:

```text
full PICF/script py_compile:
  PASS

OWM verifier:
  PASS

strict diagnose:
  PASS with expected WARN boundaries when no runtime metrics/eval path is
  supplied.

recursive dataflow trace:
  PASS

MVTrack deep audit:
  PASS

44-file PICF/script pytest excluding scripts/train_test.py:
  370 passed, 3 skipped

scripts/train_test.py:
  collection blocked by the local base-training dependency environment:
  ModuleNotFoundError: No module named 'wandb_watchdog.observers.polling'
```

Fix made during this audit:

```text
scripts/calvin/evaluate_picf_policy_test.py:
  The monkeypatched _DummyModel was stale. It did not accept the current
  save_anchor_debug / save_prediction_debug constructor arguments and did not
  implement close(), while the production eval wrapper now closes the policy in
  a finally block. The test stub was updated to accept extra keyword arguments,
  preserve them for assertion, implement close(), and assert that close() was
  called. This was a test-lifecycle drift fix, not a runtime behavior change.
```

Strict conclusion:

```text
The current local code/dataflow audit is stronger than the earlier targeted
pass. It verifies that the maintained README story, active code, guarded no-op
paths, MVTrack invariants, and local regression tests are mutually consistent
under the available environment.

It still does not prove:
  CALVIN behavior success,
  anchor-health stability on A5/A7,
  video/overlay quality,
  active real tracklet/proposal evidence on CALVIN,
  or final ordinal/fine-instance grounding.

Those remain live-run acceptance questions, not local static-audit questions.
```

## 2026-05-12 Live A5/A7 Status After Stricter Local Audit

Checked from the active cloud tmux sessions after the stricter local audit.

### A5 clean staged cotrain

Runtime:

```text
machine: A5 / ZWWQO6
tmux: a5_clean_staged_7bff430
run: /mnt/checkpoints/picf_core/picf_core/picf_a5_clean_staged_burnin4_750new_20260512_7bff430
log: /mnt/checkpoints/picf_core/picf_core/picf_a5_clean_staged_burnin4_750new_20260512_7bff430.train_tmux.log
progress at check: step 560 / 1200
speed: about 16.6 s/step
GPU: both A100s active
```

Recent metrics, last 10 logged rows:

```text
loss_total:
  last 0.7407, mean10 0.7550
loss_alignment:
  last 0.7321, mean10 0.7453
loss_action_default_equiv:
  last 0.0695, mean10 0.0780
aqr_same_role_support_overlap_max:
  last 0.7426, mean10 0.8335, range10 0.7426..0.9213
aqr_same_role_local_jaccard_max:
  last 0.9079, mean10 0.9222
aqr_same_role_local_true_overlap_max:
  last 0.0483, mean10 0.0530
posterior_recycle_rate:
  last 0.4275, mean10 0.3633, range10 0.1683..0.6057
posterior_identity_switch_rate:
  last 0.6889, mean10 0.7611
grad_norm:
  last 5.0, mean10 2.1370
```

Interpretation:

```text
A5 is running normally and action/alignment losses are decreasing, but the
anchor-state health is not accepted. same-role support overlap is no longer
hard-collapsed at 0.99, yet local_jaccard remains high and recycle/identity
switch are still too active. This means the staged path is useful for
diagnosis but not yet proof that anchor identity is stable.
```

Tail command:

```bash
ssh -p 29776 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_clean_staged_burnin4_750new_20260512_7bff430.train_tmux.log
```

### A7 direct cotrain control

Runtime:

```text
machine: A7 / qgE72e
tmux: a7_direct_cotrain_7bff430
run: /mnt/checkpoints/picf_core/picf_core/picf_a7_direct_cotrain_burnin4_750new_20260512_7bff430
log: /mnt/checkpoints/picf_core/picf_core/picf_a7_direct_cotrain_burnin4_750new_20260512_7bff430.train_tmux.log
progress at check: step 550 / 1200
GPU: both A100s active
```

Recent metrics, last 10 logged rows:

```text
loss_total:
  last 0.7387, mean10 0.7650
loss_alignment:
  last 0.7296, mean10 0.7541
loss_action_default_equiv:
  last 0.0733, mean10 0.0877
aqr_same_role_support_overlap_max:
  last 0.5787, mean10 0.8762, range10 0.5787..0.9973
aqr_same_role_local_jaccard_max:
  last 0.9776, mean10 0.9525
aqr_same_role_local_true_overlap_max:
  last 0.0521, mean10 0.0726
posterior_recycle_rate:
  last 3.64e-14, mean10 0.00114
posterior_identity_switch_rate:
  last 0.8333, mean10 0.7861
grad_norm:
  last 0.7371, mean10 1.0176
```

Interpretation:

```text
A7 is also running normally. Direct cotrain does not currently show recycle
saturation, and the last logged same-role support overlap is low. However,
local_jaccard remains very high and the 10-row same-role support mean is still
high because earlier rows reached near-collapse. This run is therefore not yet
accepted either. The useful contrast is that A5 has nontrivial recycle while A7
has near-zero recycle but both still show high local candidate reuse. That
points the next analysis toward local candidate/support reuse and posterior
identity switching, not only action-gradient recycle.
```

Tail command:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_direct_cotrain_burnin4_750new_20260512_7bff430.train_tmux.log
```

Strict live-run conclusion:

```text
Both runs are alive and producing comparable action metrics. Neither run is
accepted yet. The current evidence supports this decomposition:

1. The previous hard recycle-saturation failure is not universally present.
2. Action/alignment losses can decrease under the cleaned profile.
3. same-role row support can temporarily improve.
4. local candidate reuse remains high enough that anchor identity is not
   proven stable.

The next decision should wait for later rows or checkpoint-level diagnostics,
not just the current loss decrease.
```

Latest minute check after this report entry:

```text
A5 step 580:
  loss_total=0.7515
  loss_action_default_equiv=0.0726
  same_role_support_overlap=0.8803
  local_jaccard=0.8938
  recycle=0.2938
  identity_switch=0.8000

A7 step 560:
  loss_total=0.7308
  loss_action_default_equiv=0.0691
  same_role_support_overlap=0.6789
  local_jaccard=0.9656
  recycle=2.15e-14
  identity_switch=0.7778
```

Interpretation stays unchanged: action/alignment losses are reasonable, but
local candidate reuse and identity switching are not yet healthy enough for
acceptance.

## 2026-05-12 Move-On Gate at A5/A7 Step 600

The clean/direct pair has now produced enough evidence. It should be stopped
and replaced with a local-refinement diagnostic rather than extended unchanged.

Final aligned evidence:

```text
A5 clean staged, step 600:
  loss_total=0.7486
  loss_action_default_equiv=0.0808
  same_role_support_overlap=0.9098
  local_jaccard=0.8803
  local_true_overlap=0.0439
  recycle=0.3291
  identity_switch=0.6833

A5 later stop point, step 620:
  loss_total=0.7442
  loss_action_default_equiv=0.1043
  same_role_support_overlap=0.7243
  local_jaccard=0.8948
  local_true_overlap=0.0528
  recycle=0.4152
  identity_switch=0.7944

A7 direct cotrain, step 600:
  loss_total=0.7412
  loss_action_default_equiv=0.0760
  same_role_support_overlap=0.9267
  local_jaccard=0.9400
  local_true_overlap=0.0656
  recycle=1.88e-12
  identity_switch=0.8833
```

Mathematical reading:

```text
Let P_j be the row-level AQR support distribution and Ω_j be the local
refinement candidate set formed from top-k visual/temporal/point/tracklet/
proposal evidence.

The observed combination is:
  J(Ω_i, Ω_j) is high for same-role pairs,
  but weighted local true overlap remains low.

Therefore the failure is not simply that every anchor has the identical local
weights. It is that same-role anchors repeatedly reuse the same candidate pool,
then make slightly different reads from that pool. That is enough to destabilize
posterior identity assignment because the binding path sees ambiguous evidence
sets over time even when final row weights are not exactly identical.

A7 is especially diagnostic:
  recycle is essentially zero, but identity_switch and local_jaccard remain
  high. This rules out recycle saturation as the only cause. The next test
  should isolate local refinement before changing recycle or action loss again.
```

Decision:

```text
Move on from the clean/direct cotrain pair.
Do not move to production long-run acceptance.
Do not add new scalar losses yet.
Run a two-branch local-refinement diagnostic:
  A5: local refinement off.
  A7: local refinement constrained by smaller top-k and lower residual weight.
```

### Next Diagnostic: Local Refinement Isolation

Purpose:

```text
Determine whether high same-role local candidate reuse is caused by the local
refinement branch itself or by upstream AQR supports before local refinement.
```

Experiment L0, A5:

```text
name: picf_a5_localoff_burnin4_750new_20260512_7bff430
change from clean staged:
  --no-local-refinement-enabled

Interpretation:
  If identity_switch / support_overlap improve while action loss remains
  comparable, local refinement is currently destabilizing identity.
  If they do not improve, the problem is upstream AQR support/binding rather
  than the local refinement residual.
```

Experiment L1, A7:

```text
name: picf_a7_localk8w01_burnin4_750new_20260512_7bff430
change from clean staged:
  --local-refinement-topk 8
  --local-refinement-weight 0.10

Interpretation:
  If L1 improves while L0 hurts action/anchor alignment, local refinement is
  useful but too broad/strong at topk=32, weight=0.25.
  If L1 still has high local_jaccard and identity_switch, a code-level
  competitive local allocation or better tracklet/proposal evidence is required.
```

Acceptance gates by step 600 or 750:

```text
hard gates:
  posterior_identity_switch_rate < 0.50
  aqr_same_role_local_jaccard_max < 0.80
  aqr_same_role_support_overlap_max < 0.80
  finite grad_norm without frequent clipping

secondary gates:
  loss_action_default_equiv not worse than the clean/direct pair by >20%
  loss_total continues around 0.73..0.76
  recycle does not saturate to either 0 or 1 in a way that hides assignment
  failures.
```

Tail commands:

```bash
ssh -p 29776 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_localoff_burnin4_750new_20260512_7bff430.train_tmux.log

ssh -p 28060 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_localk8w01_burnin4_750new_20260512_7bff430.train_tmux.log
```

Initial sanity check:

```text
A5 local-off, step 470:
  loss_total=0.7573
  loss_action_default_equiv=0.1146
  same_role_support_overlap=0.5098
  local_jaccard=0.0
  local_true_overlap=0.0
  recycle=0.0804
  identity_switch=0.7278

A7 local topk=8 weight=0.10, step 470:
  loss_total=0.7625
  loss_action_default_equiv=0.1102
  same_role_support_overlap=0.7917
  local_jaccard=0.5482
  local_true_overlap=0.0502
  recycle=0.2472
  identity_switch=0.7278
```

Early interpretation:

```text
The local-refinement hypothesis is plausible but not yet proven. A5 local-off
removes local candidate reuse by construction and immediately lowers
same-role support overlap. A7 constrained-local substantially lowers local
jaccard versus the previous topk=32 / weight=0.25 run. However identity_switch
is still high in both early rows, so the step-470 sample cannot yet distinguish
startup noise from persistent posterior assignment instability. Judge at
step 520/600/750.
```

### Local-Refinement Isolation Checkpoint: Step 600+

Status:

```text
date: 2026-05-12
runtime commit on cloud: 7bff430
local audit commit documenting this checkpoint: pending after this note

A5:
  run: picf_a5_localoff_burnin4_750new_20260512_7bff430
  intervention: local refinement disabled
  latest inspected step: 750

A7:
  run: picf_a7_localk8w01_burnin4_750new_20260512_7bff430
  intervention: local refinement kept but constrained
  overrides:
    --local-refinement-topk 8
    --local-refinement-weight 0.10
  latest inspected step: 750
```

Latest metrics:

```text
A5 local-off, step 750:
  loss_total = 0.7314
  loss_action_default_equiv = 0.0905
  aqr_same_role_support_overlap_max = 0.7294
  aqr_same_role_local_jaccard_max = 0.0
  aqr_same_role_local_true_overlap_max = 0.0
  posterior_identity_switch_rate = 0.8278

A7 local topk=8, weight=0.10, step 750:
  loss_total = 0.7274
  loss_action_default_equiv = 0.0920
  aqr_same_role_support_overlap_max = 0.4103
  aqr_same_role_local_jaccard_max = 0.4853
  posterior_identity_switch_rate = 0.8778
```

Mathematical interpretation:

```text
The local-refinement branch is a real overlap amplifier, but it is not the only
identity failure source.

A5 proves that disabling local refinement can strongly reduce same-role support
overlap while preserving clean loss descent. Because local priors are absent,
local_jaccard and local_true_overlap are exactly zero by construction.

A7 proves that reducing local top-k and residual weight also lowers local
candidate reuse versus the old topk=32 / weight=0.25 regime. The remaining
local_jaccard ~=0.49 and true overlap ~=0.036 are no longer catastrophic.

However both branches still report posterior_identity_switch_rate ~=0.79..0.83.
Therefore the current identity_switch signal cannot be explained solely by
local candidate reuse, action gradients, or recycle saturation. A7 even has
healthy support overlap at the latest inspected point but still high switch.
```

Code audit of the identity metric:

```text
pipeline.py currently computes posterior_identity_switch_rate by comparing
argmax(binding_j) at time t and t-1 for each slot index j.

This is useful as a coarse alarm, but it is not a mathematically sufficient
identity-health metric because it does not mask:
  low-alpha slots
  high-recycle slots
  low-support-mass slots
  ambiguous binding rows with small top1-top2 margin
  legitimate permutation/reassignment events

Thus the next step should not add another scalar training loss. It should add
stable-slot identity diagnostics that distinguish real identity swaps from
metric over-counting on uncertain slots.
```

Decision:

```text
Move on after both runs finish their 750-step closure rows.
Do not treat local-off or local-k8 as production acceptance yet.
Do not keep tuning local_refinement_topk/weight as the next main action.
Do not add a new identity loss before verifying whether the current switch
metric is over-counting unstable or low-confidence slots.

Next diagnostic:
  add masked/stable identity-switch metrics and binding-margin diagnostics,
  then run a short confirmation job with the best local settings.
```

Next diagnostic design:

```text
Add debug-only metrics:
  posterior_stable_slot_fraction
  posterior_identity_switch_rate_stable
  posterior_identity_switch_rate_recycled
  posterior_binding_top1_margin_mean
  posterior_binding_top1_margin_stable_mean
  posterior_binding_argmax_agreement_rate

Stable slot mask:
  alpha >= alpha threshold if alpha exists
  recycle <= recycle threshold
  support mass raw above threshold if available
  binding top1-top2 margin above threshold

Interpretation:
  If stable switch is low while raw switch is high:
    the previous identity alarm was too pessimistic and the next training
    question is support/action tradeoff, not identity collapse.

  If stable switch remains high:
    binding assignment is truly unstable and the next structural fix should
    target assignment compatibility or support-signature binding, not local
    refinement.
```

Local implementation status:

```text
Implemented on local branch after this checkpoint:
  posterior_identity_switch_rate_stable
  posterior_identity_switch_rate_nonrecycled
  posterior_identity_switch_rate_recycled
  posterior_stable_slot_fraction
  posterior_binding_top1_margin_mean
  posterior_binding_top1_margin_min
  posterior_binding_top1_margin_stable_mean

Implementation scope:
  debug metrics only
  no change to posterior update
  no change to AQR routing
  no change to action path
  no new loss

Mathematical guard:
  raw identity_switch remains the alarm metric;
  stable identity_switch is the acceptance diagnostic.

Stable mask:
  nonrecycled current/previous slots
  alpha >= 0.25 when available
  support_mass >= 0.05 when available
  current and previous binding top1-top2 margin >= 0.05
```

Local validation after implementation:

```text
python -m py_compile:
  pipeline.py
  picf_core_train.py
  picf_owm_evidence_bundle.py
  verify_picf_owm_contract.py
  picf_owm_strict_diagnose.py
  PASS

PYTHONPATH=src python scripts/verify_picf_owm_contract.py:
  PASS

PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS

PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS

PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS

PYTHONPATH=src pytest -q \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py:
  4 passed
```

Next run after pushing this patch:

```text
short confirmation only, not a new full training phase:
  use the better local-refinement setting from the completed 750-step closure
  rows, then inspect stable identity-switch metrics.

Expected resolution:
  If raw switch high but stable switch low:
    move on to longer cotrain with stable-switch as the true identity gate.

  If stable switch high:
    do not move to long-run; inspect binding assignment compatibility and
    support-signature/address coefficients.
```

Launched confirmation run:

```text
machine: A5
tmux: a5_stableid_253c9be
run: picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be
code: 253c9be
resume: model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b
num_train_steps: 650
local_refinement_topk: 8
local_refinement_weight: 0.10
purpose: collect stable-slot identity metrics under the constrained-local
  setting that reduced overlap without adding any new loss.

Tail:
  tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be.train_tmux.log
```

### Stable-Identity Confirmation Result

Run:

```text
name: picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be
code: 253c9be
resume: model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b
completed: step 650
local_refinement_topk: 8
local_refinement_weight: 0.10
```

Final row:

```text
step = 650
loss_total = 0.7243
loss_alignment = 0.7123
loss_action_default_equiv = 0.0963
aqr_same_role_support_overlap_max = 0.4313
aqr_same_role_local_jaccard_max = 0.4651
aqr_same_role_local_true_overlap_max = 0.0280
posterior_identity_switch_rate = 0.6722
posterior_identity_switch_rate_stable = 0.0
posterior_identity_switch_rate_nonrecycled = 0.6722
posterior_identity_switch_rate_recycled = 0.0
posterior_stable_slot_fraction = 0.1167
posterior_binding_top1_margin_mean = 0.1121
posterior_binding_top1_margin_min = 0.0000033
posterior_binding_top1_margin_stable_mean = 0.9603
posterior_recycle_rate = 0.3014
posterior_address_update_rate_mean = 0.0299
grad_norm = 0.3828
```

Last-10 mean:

```text
loss_total = 0.7256
loss_alignment = 0.7156
loss_action_default_equiv = 0.0803
same_role_support_overlap = 0.4491
local_jaccard = 0.5054
local_true_overlap = 0.0406
raw identity_switch = 0.7722
stable identity_switch = 0.0025
stable_slot_fraction = 0.1167
binding_margin_mean = 0.1133
binding_margin_stable_mean = 0.9554
recycle_rate = 0.2955
grad_norm = 0.5102
```

Conclusion:

```text
The high raw posterior_identity_switch_rate is not reliable as a standalone
identity-collapse claim. Under the stable-slot mask, identity switches are
near zero. Stable slots have very high binding margins, while the global margin
mean remains low, meaning most raw argmax churn comes from ambiguous slots.

This changes the next experiment target:
  do not add a new identity loss;
  do not change posterior math yet;
  do not continue tuning local_refinement alone.

The actual bottleneck is now stable-slot coverage:
  stable_slot_fraction ~= 0.12 is too low for claiming mature identity tracking.
```

Next experiment matrix:

```text
E1. Stable coverage warmup
  Purpose:
    increase stable_slot_fraction without breaking stable identity.
  Change:
    keep local_refinement_topk=8 and weight=0.10;
    keep action-prefix stopgrad;
    keep OWM predictive losses at 0;
    run longer from the same resume or from this 650 checkpoint if available.
  Acceptance:
    stable_slot_fraction trends upward above 0.20..0.30;
    posterior_identity_switch_rate_stable stays below 0.10;
    same_role_support_overlap stays below 0.70;
    action_default_equiv does not drift upward.

E2. Controlled action pressure
  Purpose:
    test whether action cotrain destroys stable identity once stable slots exist.
  Change:
    same as E1, but sweep action weight scale or prefix-stopgrad boundary only
    if E1 shows stable-slot fraction increasing.
  Acceptance:
    action loss improves while stable switch remains low.
  Reject:
    stable switch rises or stable_slot_fraction collapses.

E3. Support-signature coefficient audit
  Purpose:
    determine whether more same-object binding pressure increases stable-slot
    coverage or merely locks wrong identities.
  Change:
    small sweep only after E1:
      bind_support_signature_weight: current vs 0.75
      bind_embedding_signature_weight: current vs 0.35
      bind_address_weight unchanged
  Reason:
    object-binding papers imply pairwise same-object subspace should help,
    but address should stay gated. Increase support/signature before address.
  Acceptance:
    stable_slot_fraction rises, stable switch remains low, overlap does not
    re-collapse.

E4. Longer cotrain candidate
  Purpose:
    only after E1/E2 gates pass.
  Contract:
    local_refinement_topk=8
    local_refinement_weight=0.10
    prefix-stopgrad retained initially
    PaliGemma trainable
    predictive/JEPA losses still 0
  Reason:
    current evidence supports the routing/binding contract, but not yet
    predictive auxiliary cotrain.
```

Do not run next:

```text
Do not open slot-JEPA/support-pred/binding-consistency yet:
  stable_slot_fraction is still too low; predictive targets would supervise
  too many ambiguous slots.

Do not hard-penalize raw identity_switch:
  stable-switch evidence shows raw switch is dominated by ambiguous slots.

Do not increase address weight first:
  address is an inertia term; increasing it before stable-slot coverage grows
  risks lock-in.
```

## 2026-05-12 Three-Line Follow-Up: Anchor Visualization, Stable Coverage, and Paper-Grounded Fix Plan

This section records the next diagnostic stage after the stable-identity
confirmation run. The goal is not to add another compensating loss. The goal is
to separate three questions that were previously entangled:

```text
1. What do the anchor positions and support overlays actually do in CALVIN
   rollouts?
2. Does the stable-slot subset expand with more warmup under the current clean
   local-refinement contract?
3. Do recent object-binding / VLA tracking papers suggest a structural fix that
   is consistent with the current belief-state design?
```

### Deployment

A5 is assigned to visualization:

```text
machine: A5 / ZWWQO6
policy server tmux: a5_anchor_serve_83cf6df
eval tmux: a5_anchor_eval_83cf6df
checkpoint:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be
server port: 8015
eval output:
  /mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/
purpose:
  generate CALVIN anchor overlays, prediction debug, and videos so that the
  numeric stable-slot diagnosis can be checked against real spatial behavior.
```

A7 is assigned to stable-coverage continuation:

```text
machine: A7 / qgE72e
tmux: a7_stabcov_83cf6df
run:
  picf_a7_stabcov_localk8w01_burnin4_1050new_20260512_83cf6df
resume:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_localk8w01_burnin4_750new_20260512_7bff430/750
target step: 1050
local_refinement_topk: 8
local_refinement_weight: 0.10
high-risk predictive losses: 0
purpose:
  test whether stable_slot_fraction increases with additional clean warmup while
  stable identity switches stay low.
```

Tail commands:

```bash
ssh -p 29776 root@36.139.225.68
tail -f /mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/eval.log

ssh -p 28060 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_stabcov_localk8w01_burnin4_1050new_20260512_83cf6df.train_tmux.log
```

### Paper-Grounded Interpretation

Recent object-binding evidence changes what should be fixed first.

`Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?`
reports that pretrained ViTs can encode whether two patches belong to the same
object as an `IsSameObject` property, that a quadratic similarity probe can
decode it with high accuracy, and that the signal appears as a low-dimensional
subspace on top of object features. This supports the current
`binding_signature_proj` direction: binding should include a pairwise
same-object support/signature subspace, not only hidden cosine and geometry.
It also argues against adding a raw identity-switch loss before we know whether
the token evidence contains separable same-object information.

`TrackVLA: Embodied Visual Tracking in the Wild` treats visual tracking as a
core VLA capability under occlusion and dynamics. This supports tracklet/proposal
evidence as the clean long-term way to increase stable-slot coverage, but it
does not justify pretending that absent tracklet data is already active in the
current CALVIN training flow.

`WristWorld` explicitly frames wrist views as important because they capture
fine-grained hand-object interaction. This supports keeping wrist as a typed
temporal evidence view and using anchor overlays to inspect whether gripper-view
evidence helps contact-phase binding. Without calibrated extrinsics, wrist
tokens must remain typed evidence rather than static-frame geometry truth.

`MAP-VLA` supports memory as auxiliary retrieved context for long-horizon
manipulation. This is aligned with the current small residual cache gate. It
does not justify letting cache become posterior truth, nor opening predictive
losses before stable-slot coverage improves.

Sources used for this interpretation:

```text
https://arxiv.org/abs/2510.24709
https://proceedings.mlr.press/v305/wang25f.html
https://arxiv.org/abs/2510.07313
https://arxiv.org/abs/2511.09516
```

### Mathematical Reading

The current belief update should remain:

```math
b_t = U_{\mathrm{post}}\left(P(b_{t-1}, a_{t-1}), R_{\mathrm{AQR}}(o_t)\right)
```

The stable-id run showed:

```text
raw identity_switch high
stable identity_switch near zero
stable_slot_fraction low
```

Therefore the correct diagnosis is:

```text
not proven global identity collapse;
stable slots are reliable when they exist;
coverage of reliable slots is insufficient.
```

The next fix should target stable-slot coverage, not raw switch suppression.
The most coherent structural lever is:

```math
\ell_{j,i}
=
\ell^{hidden}_{j,i}
+ \ell^{geom}_{j,i}
+ \lambda_s g_j \langle P_\sigma \sigma^-_j, P_\sigma \sigma_i \rangle
+ \lambda_a g_j \langle P_a a^-_j, P_a a_i \rangle
```

with the ordering:

```text
1. use support/signature same-object evidence first;
2. keep address as a gated inertia term;
3. only strengthen cache/address after stable-slot coverage rises;
4. keep future predictive losses closed until stable slots cover enough objects.
```

This is a structural binding route, not a new auxiliary loss. It is consistent
with the IsSameObject paper because it probes and uses a pairwise object-binding
subspace. It is consistent with the POMDP belief filter because it changes the
assignment compatibility term, not the posterior authority.

### Acceptance Gates

A5 visualization acceptance:

```text
Anchor overlays must show whether stable slots correspond to task-relevant
objects/contact regions or only to easy high-margin background regions.

If overlays show stable slots mostly on irrelevant regions, stable_slot_fraction
alone is not sufficient.

If overlays show stable slots on task-relevant objects but coverage is low,
the next fix should increase coverage through support-signature / tracklet
evidence rather than changing action loss.
```

A7 stable-coverage acceptance:

```text
posterior_stable_slot_fraction:
  should rise above 0.20..0.30 before longer cotrain is accepted.

posterior_identity_switch_rate_stable:
  should stay below 0.10.

aqr_same_role_support_overlap_max:
  should stay below 0.70, with local_true_overlap remaining small.

loss_action_default_equiv:
  should not improve at the cost of stable-slot collapse.
```

If A7 passes:

```text
move to controlled action-pressure continuation;
keep predictive losses closed;
use CALVIN overlay evidence to decide whether tracklet/proposal dataflow is the
next necessary investment.
```

If A7 fails:

```text
do not add raw identity loss;
do not increase address weight first;
run a support-signature coefficient audit and an offline IsSameObject probe.
```

### Preliminary A5 Anchor-Overlay Reading

The first A5 CALVIN visualization pass is now producing:

```text
/mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/anchor_debug/anchor_debug.jsonl
/mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/anchor_debug/anchor_overlay_ep0000.mp4
/mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/prediction_debug/prediction_compare_ep0000.mp4
/mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/videos/push_blue_block_right_1778600884964.mp4
```

Initial parsed frames for goal `go push the blue block right`:

```text
frame 0:
  slot 0 role=0 alpha=0.497 pixel=[57.6, 95.6] xyz=[0.001, -0.197, 0.572]
  slots 1..7 role=1 alpha=0.499 pixel=[75.7, 107.3]
    xyz=[-0.183, -0.045, 0.348]

frame 30:
  slot 0 role=0 alpha=0.490 pixel=[55.0, 94.1] xyz=[0.009, -0.207, 0.589]
  slots 1..7 role=1 alpha=0.489 pixel=[82.0, 96.3]
    xyz=[-0.101, -0.048, 0.415]
  same_role_visual_overlap=0.939
  same_role_point_overlap=0.941

frame 60:
  slot 0 role=0 alpha=0.489 pixel=[56.5, 95.8] xyz=[0.002, -0.201, 0.575]
  slots 1..7 role=1 alpha=0.489 pixel=[73.3, 95.9]
    xyz=[-0.134, -0.054, 0.422]
  same_role_visual_overlap=0.914
  same_role_point_overlap=0.297
```

Per-slot alpha>=0.25 pixel stability over the first 61 frames:

```text
slot 0:
  n=61 mean_xy=[53.1, 94.2] std_xy=[2.0, 1.7]

slots 1..7:
  n=61 mean_xy=[76.6, 97.4] std_xy=[3.4, 3.3]
  nearly identical across all seven same-role slots
```

Interpretation:

```text
The stable-id metric was useful, but it is not sufficient as an acceptance gate.
Stable-switch near zero can mean that several same-role slots are stably sharing
the same candidate, not that object identity coverage is healthy.

The immediate problem is therefore sharper:
  same-role slot differentiation / candidate ownership is still insufficient in
  rollout space.

The correct next action is not a raw identity-switch loss and not predictive
JEPA cotrain. The next action should be a principled differentiation source:
  1. real tracklet/proposal evidence when available;
  2. an offline IsSameObject probe to test whether V-JEPA/PG/wrist/point tokens
     contain separable same-object information;
  3. a support-signature coefficient audit only if the probe or overlay evidence
     indicates separability exists.
```

This preliminary overlay result also explains why earlier numeric runs could
look partially healthy while same-role overlap reappeared: the slots can be
stable and high-margin, but stable around the same candidate. That is a
coverage/differentiation failure, not a gradient explosion, recycle saturation,
or action-loss-only failure.

### Next Test Matrix After Overlay Evidence

The next tests must distinguish three mechanisms. They should not add losses
until the mechanism is identified.

#### Test A: Stable-Coverage Continuation

Already running on A7:

```text
run:
  picf_a7_stabcov_localk8w01_burnin4_1050new_20260512_83cf6df

what it tests:
  whether the current clean local-refinement setting can increase the number
  of reliable slots by simply training longer from step 750 to 1050.

why it is mathematically coherent:
  It keeps the same belief update and assignment objective. It tests convergence
  of the existing objective before adding a new force.

accept:
  stable_slot_fraction > 0.20..0.30
  stable identity_switch < 0.10
  same_role_support_overlap < 0.70
  overlay shows more than one same-role object/candidate is represented.

reject:
  stable_slot_fraction remains near 0.10..0.12
  same-role slots 1..7 remain spatially identical in overlay
  action loss improves while anchor coverage remains collapsed.
```

If Test A passes, the next run can be controlled action-pressure continuation.
If Test A fails, do not continue action cotrain; move to Test B.

#### Test B: Offline IsSameObject Probe

This is the most direct test implied by the object-binding paper.

```text
inputs:
  saved V-JEPA static/wrist tokens
  PG support tokens
  point/projective neighborhoods
  high-confidence posterior supports
  optional tracklet continuity when data is available

weak positive pairs:
  same tracklet ID across nearby frames
  same high-confidence posterior slot with low recycle and high binding margin
  same point-neighborhood / projection cluster
  same PG/V-JEPA support peak under low entropy

weak negative pairs:
  far apart in image/projected 3D
  different high-confidence posterior slots with low overlap
  different tracklet IDs

probe:
  score(i, j) = z_i^T W z_j
  or projected cosine in binding_signature_proj space.

metrics:
  AUC
  same-minus-different margin
  per-view and per-modality separability
  layer/modality where object binding is strongest.
```

Acceptance:

```text
If AUC/margin is high:
  the tokens contain object-binding information; the correct structural fix is
  to strengthen support-signature assignment or feed real tracklet/proposal
  evidence.

If AUC/margin is low:
  the evidence representation itself does not separate same-role objects enough;
  more AQR loss tuning is unlikely to solve fourth-object/fine-instance binding.
  The next useful investment is better evidence: tracklets, proposals, higher
  local resolution, or dataset-level signals.
```

This probe is not a new training loss. It is a measurement of the information
available in `Z_{\le t}`. It directly tests the information-theoretic condition:

```math
I(Y; A_t \mid \ell) \le I(Y; Z_{\le t} \mid \ell)
```

If the weak same-object relation is not decodable from the tokens, the action
policy cannot reliably choose the fourth fine-grained object through routing
alone.

#### Test C: Support-Signature Coefficient Audit

Only run this if Test B or overlay evidence indicates that same-object evidence
exists but AQR is not using it enough.

```text
sweep:
  bind_support_signature_weight: current -> 0.75
  bind_embedding_signature_weight: current -> 0.35
  bind_address_weight: unchanged
  local_refinement_topk: 8
  local_refinement_weight: 0.10
  predictive losses: 0

why support first:
  support/signature overlap is current evidence compatibility.
  address is inertia. Increasing address first can lock the wrong object.

accept:
  same-role slots occupy different candidates in overlay;
  stable_slot_fraction rises;
  stable identity_switch remains low;
  same_role_support_overlap does not rebound.

reject:
  stronger signature raises overlap or makes slots share the same candidate
  more confidently.
```

#### Test D: Tracklet/Proposal Dataflow Activation

This is the cleanest long-term fix if the overlay confirms insufficient
same-role differentiation.

```text
why:
  TrackVLA and visual-trace style work suggest temporal correspondence is a
  first-class object-binding signal. Current CALVIN dataflow still often has
  owm_tracklet_tokens=0 and owm_proposal_tokens=0, so the runtime branch exists
  but does not yet supply evidence.

implementation:
  preprocess static and gripper videos into tracklet arrays;
  feed tracklet_* episode fields through the existing optional adapters;
  keep missing-tracklet no-op behavior;
  do not let tracklets overwrite posterior truth.

accept:
  owm_tracklet_tokens > 0 in training metrics;
  aqr_tracklet_support_entropy finite/nonzero;
  same-role slots differentiate in overlay;
  action loss does not regress.
```

#### Tests Not To Run Yet

```text
Do not open slot-JEPA/support-pred/binding-consistency:
  they supervise future slots and should wait until stable slots cover enough
  distinct candidates.

Do not add raw identity-switch penalty:
  stable-switch and overlay evidence show raw switch is partly an ambiguous-slot
  metric, not a direct training target.

Do not raise address/cache weight first:
  address/cache are historical inertia. They can preserve correct identity only
  after current evidence separates identities.

Do not use action-loss improvement as proof:
  action_default_equiv can improve while same-role slots remain collapsed.
```

The rigorous ordering is:

```text
1. finish A7 stable-coverage continuation;
2. read A5 overlays/videos;
3. if coverage fails, run offline IsSameObject probe;
4. if same-object signal exists, run support-signature coefficient audit;
5. if same-object signal is weak or absent, activate real tracklet/proposal
   dataflow rather than tuning losses;
6. only after anchor health passes, test controlled action cotrain;
7. only after anchor/action coexistence is stable, consider predictive aux.
```

### A7 Stable-Coverage Midpoint Reading

At the first recorded continuation points after resuming from step 750:

```text
step 760:
  loss_total=0.7149
  loss_action_default_equiv=0.0555
  same_role_support_overlap=0.4208
  local_jaccard=0.4341
  local_true_overlap=0.0471
  raw_identity_switch=0.8222
  stable_identity_switch=0.0
  stable_slot_fraction=0.1111
  recycle_rate=0.3457

step 780:
  loss_total=0.7360
  loss_action_default_equiv=0.0692
  same_role_support_overlap=0.6298
  local_jaccard=0.5581
  local_true_overlap=0.0548
  stable_identity_switch=0.0
  stable_slot_fraction=0.1111

step 800:
  loss_total=0.7252
  loss_action_default_equiv=0.0470
  same_role_support_overlap=0.5234
  local_jaccard=0.5716
  local_true_overlap=0.0528
  raw_identity_switch=0.7944
  stable_identity_switch=0.0
  stable_slot_fraction=0.1111
  recycle_rate=0.3232
```

Tail mean over available rows:

```text
loss_total=0.7273
loss_action_default_equiv=0.0614
same_role_support_overlap=0.5243
local_jaccard=0.5313
local_true_overlap=0.0505
raw_identity_switch=0.8244
stable_identity_switch=0.0
stable_slot_fraction=0.1111
recycle_rate=0.3109
stable_binding_margin=0.9793
```

Interpretation:

```text
The current clean local-refinement setting controls exact overlap better than
the failed collapse runs, and stable slots have high binding margin. However,
stable_slot_fraction is not increasing at all through step 800.

This is a stronger version of the A5 overlay finding:
  reliable slots exist,
  but there are too few of them,
  and same-role slots can be reliable around the same candidate.

Therefore a longer continuation by itself is unlikely to be sufficient unless
the later 800->1050 segment changes stable_slot_fraction. The acceptance gate
for the remaining segment is now explicit:
  if stable_slot_fraction remains near 0.111 at step 1050, this experiment is
  rejected as a coverage-expansion method even if action loss improves.
```

Immediate next plan after A7 reaches step 1050:

```text
If stable_slot_fraction > 0.20 and overlays show differentiated slots:
  run controlled action-pressure continuation.

If stable_slot_fraction remains ~0.11:
  do not tune action, JEPA, or raw switch;
  run Offline IsSameObject Probe v0.

If A5 overlays show all same-role slots sharing candidates:
  prioritize tracklet/proposal evidence activation over another coefficient
  sweep.
```

### Offline IsSameObject Probe v0 Design

This is the next highest-value diagnostic if A7 fails to expand stable coverage.
It is a probe, not a production loss.

Data sources available without changing the CALVIN dataset:

```text
1. anchor_debug JSON:
   posterior pixel/xyz, role ids, support mass, alpha, graph priors.

2. prediction_debug JSON/video:
   whether the policy predicts coherent motion around the same spatial region.

3. existing runtime token dumps if enabled in a short debug pass:
   V-JEPA temporal view tokens, PG support weights, point/projective tokens.

4. weak pairing from geometry:
   positives = close projected xyz / close posterior slot over adjacent frames;
   negatives = far projected xyz / different high-confidence same-role candidates.
```

Probe objective:

```math
s(i,j)=z_i^T W z_j
```

where `z_i` and `z_j` are frozen token/signature features from one modality or
a typed concatenation. The probe should report:

```text
AUC(same-object vs different-object)
same-minus-different margin
per-modality separability:
  static V-JEPA
  wrist V-JEPA
  PG image support
  point/projective support
  posterior support signatures
```

Why this is not a patch:

```text
The object-binding paper shows that IsSameObject can be a decodable property of
pretrained ViT activations and that ablating it hurts downstream use. If this
property is present in our tokens, AQR should use it as assignment compatibility.
If it is absent, no amount of slot loss tuning can create fine-instance identity
without new evidence.
```

Decision after probe:

```text
Probe positive:
  implement/support a small coefficient audit around support-signature binding,
  or add a read-only pairwise binding audit metric first.

Probe negative:
  activate real tracklet/proposal dataflow. This follows TrackVLA-style temporal
  correspondence rather than adding more scalar regularizers.
```

### A5 Completed Anchor-Overlay Reading And A7 Step-880 Update

A5 CALVIN anchor/prediction visualization completed for checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be
```

The two-sequence CALVIN smoke result is behavior-negative:

```text
push_blue_block_right: 0 / 1
open_drawer: 0 / 1
average successful sequence length: 0.0
```

This run is not treated as behavior acceptance. It is used to inspect anchor
placement. The completed `anchor_debug.jsonl` has 720 frames and gives a
consistent diagnosis:

```text
frame 0, push blue block right:
  slot 0 role=0 pixel=[57.6,95.6], xyz=[0.001,-0.197,0.572]
  slots 1..7 role=1 all pixel=[75.7,107.3], xyz=[-0.183,-0.045,0.348]
  same_role_visual_overlap=0.5005
  same_role_point_overlap=0.7107

frame 360, open drawer:
  slot 0 role=0 pixel=[58.0,95.9], xyz=[-0.015,-0.186,0.558]
  slots 1..7 role=1 all pixel=[71.6,97.3], xyz=[-0.136,-0.063,0.424]
  same_role_visual_overlap=0.7615
  same_role_point_overlap=0.5483

frame 719, open drawer:
  slot 0 role=0 pixel=[110.0,87.6], xyz=[0.104,-0.063,0.487]
  slots 1..7 role=1 all pixel=[84.9,89.3], xyz=[-0.076,-0.039,0.440]
  same_role_visual_overlap=0.8216
  same_role_point_overlap=0.3936
```

Interpretation:

```text
The issue is not that every anchor is random. The effector-like role-0 slot
tracks a separate region. The issue is that same-role scene slots 1..7 can be
stable while sharing one candidate. Therefore stable identity switch is a
necessary but insufficient metric. The acceptance target must include stable
coverage and visual/3D differentiation among same-role scene slots.
```

A7 stable-coverage continuation is still running. The latest inspected metric
row is step 880:

```text
loss_total=0.7226
loss_action_default_equiv=0.0626
same_role_support_overlap=0.3506
local_jaccard=0.5106
local_true_overlap=0.0353
raw_identity_switch=0.7667
stable_identity_switch=0.0250
stable_slot_fraction=0.1167
recycle_rate=0.3474
stable_binding_margin=0.9596
```

Tail mean through the continuation segment:

```text
loss_total=0.7249
loss_action_default_equiv=0.0657
same_role_support_overlap=0.5034
local_jaccard=0.5299
local_true_overlap=0.0460
raw_identity_switch=0.8000
stable_identity_switch=0.0038
stable_slot_fraction=0.1128
recycle_rate=0.3355
stable_binding_margin=0.9708
```

Decision status:

```text
Positive:
  same_role_support_overlap is much lower than the 0.99 collapse runs;
  local_true_overlap remains low;
  stable slots have high binding margin.

Negative:
  stable_slot_fraction remains near 0.11 through step 880;
  A5 overlays show same-role scene slots are not differentiated;
  tracklet/proposal evidence remains inactive in the training metrics.

Current decision:
  continue A7 to step 1050 because it is close to completion;
  if stable_slot_fraction remains near 0.11, stop coverage-continuation tests
  and run Offline IsSameObject Probe v0 before any new loss or action-pressure
  sweep.
```

### A5 IsSameObject Probe Result And A7 Step-970 Decision Update

The offline same-object probe was run on the completed A5 anchor-debug bundle:

```text
input:
  /mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/anchor_debug/anchor_debug.jsonl

output:
  /mnt/calvin_eval_logs/picf_a5_stableid_anchor_253c9be/same_object_probe.json
```

Probe summary:

```text
frames: 720
positive weak pairs: 97792
negative weak pairs: 22334
ambiguous weak pairs: 42142

combined_auc: 0.8929
combined_pos_mean: 0.2724
combined_neg_mean: 0.0700

geometry_auc: 1.0000
geometry_pos_mean: 0.6797
geometry_neg_mean: 0.0399

visual_cos_auc: 0.5176
point_cos_auc: 0.5250
support_mean_auc: 0.5227

duplicate_candidate_fraction_within_frame: 0.8268
duplicate_candidate_pairs_within_frame: 62509 / 75600 same-role pairs

decision:
  same_object_signal_decodable_but_assignment_duplicates_candidates
```

Interpretation:

```text
The probe is not evidence that the current visual/point/support signatures
already separate objects well. The high combined AUC is mainly geometry-driven;
single-modality support/visual/point cosine signals are close to random. This
matches the anchor-overlay reading: role-0 can separate an effector-like region,
while same-role scene slots frequently share one spatial candidate.

Therefore the next repair should not be another raw identity inertia, address,
cache, JEPA, or action-pressure sweep. Those would preserve or optimize over
the duplicated assignment. The next scientific question is coverage/competition:
can the router be made to allocate distinct same-role scene slots before
identity inertia and predictive losses are trusted?
```

The A7 stable-coverage continuation is still not an acceptance run. Latest
checked row:

```text
step: 970
loss_total: 0.7159
loss_action_default_equiv: 0.0548
loss_action_active7: 0.2499
same_role_support_overlap: 0.3113
local_jaccard: 0.4892
local_true_overlap: 0.0354
raw_identity_switch: 0.8056
stable_identity_switch: 0.0
stable_slot_fraction: 0.1111
recycle_rate: 0.4038
stable_binding_margin: 0.9688
speed: ~18.9 s/step
```

Tail mean through the latest inspected window:

```text
loss_total: 0.7215
loss_action_default_equiv: 0.0665
same_role_support_overlap: 0.4797
local_jaccard: 0.5081
local_true_overlap: 0.0426
raw_identity_switch: 0.8003
stable_identity_switch: 0.0075
stable_slot_fraction: 0.1158
recycle_rate: 0.3596
stable_binding_margin: 0.9547
```

Updated decision:

```text
A7 is not a dead run in the sense of 0.99 support collapse. It has low local
true overlap and stable slots have high binding margins. However, it has not
expanded stable coverage: stable_slot_fraction remains near 0.11 from step
880 to step 970.

Continue only to the planned step-1050 endpoint because it is close. Unless
stable_slot_fraction rises materially above 0.20 and overlays show separated
same-role scene slots, do not extend A7. Move to the coverage/competition
diagnostic stage below.
```

### Next 12-Hour Experiment Design

The next stage should answer one causal question:

```text
Is the remaining failure caused by missing same-object evidence, or by AQR
assignment/competition failing to use available evidence?
```

The A5 probe says:

```text
geometry contains a strong weak same-object signal;
exported visual/point/support signatures alone do not separate enough;
same-role duplicate candidates are common.
```

So the next 12 hours should not be spent on larger action cotrain, cache
inertia, predictive auxiliary losses, or reintroducing the previously rejected
role-wise competition / deterministic coverage-seed heuristics. Those are
downstream of assignment or hand-coded candidate ownership. The disciplined plan
is:

```text
0. Finish A7 to step 1050.
   Acceptance only if stable_slot_fraction > 0.20 and same-role scene slots are
   separated in overlays. Otherwise archive as "partial non-collapse, low
   coverage".

1. Run a read-only coverage/competition audit.
   Keep action-prefix stopgrad, OWM predictive losses at 0, cache small, and
   do not increase address inertia. Measure whether differentiated same-role
   evidence exists before adding any new pressure. This is not permission to
   revive the rejected role-competition or coverage-seed local-candidate
   heuristics.

2. Run a token-level IsSameObject probe.
   The current probe used exported anchor-debug supports, not raw V-JEPA/PG
   token embeddings. If token-level ViT/V-JEPA features have pairwise object
   separability, the binding reader should use that learned/probed subspace. If
   they do not, tracklet/proposal evidence must be activated before more loss
   tuning.

3. Activate optional tracklet/proposal dataflow only if token-level probe or
   overlays confirm that current per-frame evidence is insufficient.
   This follows the TrackVLA-style temporal correspondence route and is a data
   evidence upgrade, not a scalar regularizer patch.
```

Guardrails:

```text
Do not open slot_jepa/support_pred/binding_consistency yet.
Do not add raw identity-switch loss.
Do not increase cache/address inertia before coverage improves.
Do not treat loss_action_default_equiv as anchor acceptance.
Do not continue A7 just because action loss is low.
```

Mathematical reason:

```math
I(Y; A_t \mid \ell) \le I(Y; Z_{\le t} \mid \ell)
```

The current evidence says the model has enough geometry to maintain one stable
candidate, but not enough assignment diversity to allocate several same-role
object slots. Losses that preserve identity (`address`, `cache`, JEPA) should
only be trusted after the assignment stage can produce multiple differentiated
same-role candidates.

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

### A5 Signature-Level Probe Deployment

The first next-stage deployment is A5 signature-level probing. This is a
diagnostic-only code path:

```text
commit:
  c4ac7b3 Export binding signatures for same-object probe

machine:
  A5 / ssh -p 29776 root@36.139.225.68

base:
  /mnt/calvin_eval_logs/picf_a5_signature_probe_c4ac7b3_20260513_011838

server tmux:
  a5_sig_serve_c4ac7b3
  port 8016
  device cuda:1

eval tmux:
  a5_sig_eval_c4ac7b3
  num_sequences=1

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be
```

The deployed code exports these extra read-only debug fields:

```text
anchor_debug.observation.support_signature
anchor_debug.observation.binding_signature
anchor_debug.posterior.{visual,temporal,point,pg,tactile,tracklet,proposal}_signature
anchor_debug.posterior.support_signature
anchor_debug.posterior.binding_signature
```

This is not a new loss and not a runtime action-path change. It only exposes the
already-computed support/binding signatures so the probe can test whether the
object-binding-inspired pairwise subspace is actually separable in the current
checkpoint.

Decision rule:

```text
binding_signature_cos_auc >= 0.70:
  the pairwise binding subspace contains usable same-object information. The
  next engineering step can be a coefficient/reader audit that uses this
  subspace more directly, still without opening JEPA/support-pred losses.

binding_signature_cos_auc < 0.70 while geometry_auc remains high:
  current exported binding signatures are not enough. Do not increase identity
  inertia. Move to token-level ViT/V-JEPA/PG probing or activate real
  tracklet/proposal evidence.
```

This follows the object-binding paper's claim that IsSameObject is best treated
as a pairwise/quadratic relation in representation space, not as a pointwise
anchor class label. It also avoids reviving the rejected role-competition or
coverage-seed heuristics.

### 2026-05-13 12-Hour Signature Probe Deployment

This section is the active 12-hour acceptance plan after the A7 step-1050
endpoint. The goal is deliberately narrow:

```text
Determine whether the remaining low stable-slot coverage is caused by missing
same-object evidence in the exported signatures, or by AQR assignment failing
to use evidence that is already present.
```

This is a read-only diagnosis stage. It must not introduce a new scalar loss,
raise address/cache inertia, reopen slot-JEPA/support-prediction, or revive the
rejected role-competition / deterministic coverage-seed proposals. Those would
be downstream pressure terms. The current unresolved question is upstream:
whether the object-binding evidence exists in the representation at all.

#### A7 Endpoint Record

A7 completed the planned 1050-step stable-coverage endpoint:

```text
run:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_stabcov_localk8w01_burnin4_1050new_20260512_83cf6df

step:
  1050

loss_total:
  0.720651

loss_action_default_equiv:
  0.082193

aqr_same_role_support_overlap_max:
  0.293733

aqr_same_role_local_true_overlap_max:
  0.026267

posterior_stable_slot_fraction:
  0.122222

posterior_recycle_rate:
  0.320914

posterior_identity_switch_rate_stable:
  0.0

posterior_binding_top1_margin_stable_mean:
  0.926945
```

Interpretation:

```text
A7 is not the earlier 0.99 same-role support-collapse failure. It maintains
low support overlap and low true local overlap at the endpoint. However, it
does not solve the coverage bottleneck: only about 12% of posterior slots are
stable. Stable slots are reliable when they exist, but too few slots become
stable.
```

This makes A7 useful as a partial non-collapse endpoint, not as a long-run
acceptance line. Extending this exact run would mostly test whether more steps
eventually improve coverage, but it would not identify why coverage is low. The
correct next action is signature/token evidence diagnosis.

#### Mathematical Gate

Let `Y` be object identity, `Z` be typed evidence, `S` be the support/binding
signature, and `A` be the selected action/anchor assignment.

```math
I(Y; A_t \mid \ell) \le I(Y; Z_{\le t} \mid \ell)
```

If the exported signatures have low same-object separability, stronger action
loss, stronger cache, stronger address inertia, or predictive JEPA pressure can
only preserve or amplify a weak assignment. They cannot create object identity
information. If the signatures are separable, the next problem is the reader /
coefficient / assignment path, not the data representation.

The active probe therefore tests:

```math
score(i,j) = z_i^T W z_j
```

for weak same-object pairs derived from geometry / posterior support in anchor
debug output. This follows the 2025 object-binding result that IsSameObject is
best treated as a pairwise or quadratic relation in pretrained ViT-like
representations. The probe is diagnostic: it does not train the policy.

#### A5 Deployment

The first A5 attempt used the correct debug-export server but the CALVIN eval
environment was missing `calvin_agent` in `PYTHONPATH`. It was restarted with
the same checkpoint and corrected CALVIN paths.

Active A5 run:

```text
machine:
  A5 / ssh -p 29776 root@36.139.225.68

server tmux:
  a5_sig_serve_c4ac7b3

eval tmux:
  a5_sig_eval_c4ac7b3

port:
  8016

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be

active base:
  /mnt/calvin_eval_logs/picf_a5_signature_probe_c4ac7b3_retry_20260513_012306
```

The retry is confirmed to emit `anchor_debug` rows containing:

```text
anchor_debug.observation.support_signature
anchor_debug.observation.binding_signature
anchor_debug.posterior.support_signature
anchor_debug.posterior.binding_signature
```

and `observation.binding_signature` has shape:

```text
16 x 128
```

This confirms that A5 is not an empty probe. It is actively collecting the
signature evidence needed for the pairwise IsSameObject audit.

#### A5 Signature Probe Result

The A5 retry completed CALVIN rollout and the Python-3.8 compatibility issue in
the probe was fixed by removing the `zip(..., strict=False)` dependency. The
probe was rerun on the already generated anchor-debug bundle without rerunning
CALVIN.

Result:

```text
binding_signature_cos_auc:
  0.976374

binding_signature_cos_pos_mean:
  0.988203

binding_signature_cos_neg_mean:
  0.878755

support_signature_cos_auc:
  0.542649

visual_cos_auc:
  0.530407

point_cos_auc:
  0.538722

geometry_auc:
  1.0

combined_auc:
  0.913601

duplicate_candidate_fraction_within_frame:
  0.848016

decision:
  binding_subspace_decodable_but_assignment_duplicates_candidates
```

Interpretation:

```text
The object-binding-inspired projected binding signature is real signal. It is
not a random add-on and not just geometry: it separates weak same-object pairs
far better than raw visual/point/support means. However, same-role anchor
assignment still duplicates candidates heavily within frame. The unresolved
failure is therefore assignment/coverage usage of the binding subspace, not
absence of a pairwise binding subspace.
```

This changes the next engineering question:

```text
Old question:
  Does a same-object subspace exist?

Answered by A5:
  Yes, in binding_signature.

Current question:
  Why does AQR assignment still reuse the same candidate despite that subspace?
```

Consequences:

```text
Do not add a new identity loss just because stable coverage is low.
Do not increase address/cache inertia yet.
Do not turn on slot-JEPA/support-prediction.
Do not resurrect role-competition or deterministic coverage seed.

Next coherent test:
  compare A7 endpoint binding_signature AUC against A5;
  if A7 also has high binding AUC, audit binding coefficient scale and local
  candidate selection;
  if A7 binding AUC collapses, the issue is training-profile damage to the
  binding subspace.
```

#### A7 Deployment

The first A7 post-run script had a wait-loop bug:

```bash
pgrep -f "${RUN_NAME}"
```

matched the tmux/script command line after the actual trainer had finished.
This caused the signature evaluation to wait even though both GPUs were idle.
The waiting script was stopped and replaced by a direct post-run signature
evaluation.

Active A7 run:

```text
machine:
  A7 / ssh -p 28060 root@36.139.225.68

eval tmux:
  a7_sig_now_c4ac7b3

port:
  8017

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_stabcov_localk8w01_burnin4_1050new_20260512_83cf6df

active base:
  /mnt/calvin_eval_logs/picf_a7_signature_probe_c4ac7b3_now_20260513_013143
```

The A7 script synchronized through the configured China GitHub mirror:

```text
https://gh.llkk.cc/https://github.com/siyuanyue0724/openpi
```

and checked out the current signature-export scripts at commit `7702c49`.

#### A7 Signature Probe Result

The A7 endpoint signature eval completed on the 1050-step stable-coverage
checkpoint.

Result:

```text
binding_signature_cos_auc:
  0.964458

binding_signature_cos_pos_mean:
  0.988212

binding_signature_cos_neg_mean:
  0.858590

support_signature_cos_auc:
  0.533023

visual_cos_auc:
  0.504134

point_cos_auc:
  0.528002

geometry_auc:
  1.0

combined_auc:
  0.923401

duplicate_candidate_fraction_within_frame:
  0.854286

decision:
  binding_subspace_decodable_but_assignment_duplicates_candidates
```

A7 agrees with A5:

```text
A5 binding_signature_cos_auc:
  0.976374

A7 binding_signature_cos_auc:
  0.964458

A5 duplicate_candidate_fraction:
  0.848016

A7 duplicate_candidate_fraction:
  0.854286
```

Conclusion:

```text
The current training profile did not destroy the pairwise binding subspace.
Both A5 and A7 expose a strong decodable same-object signal in
binding_signature. The remaining failure is repeated candidate assignment /
coverage, not absence of object-binding information and not action-gradient
destruction of the binding subspace in this endpoint.
```

This rules out several hypotheses:

```text
Rejected:
  The binding_signature_proj path is useless.
  A7 failed because it lost all pairwise object-binding signal.
  The next fix should be a stronger identity loss.
  The next fix should be stronger cache/address inertia.
  The next fix should be opening slot-JEPA/support-prediction.

Still live:
  AQR assignment/local candidate selection is not using the binding subspace to
  distribute same-role slots over distinct candidates.
  Raw support_signature is nearly constant and cannot be used as the primary
  object discriminator.
  Tracklet/proposal evidence may still be needed if per-frame candidate
  assignment cannot produce enough coverage.
```

Next coherent engineering move:

```text
Use the already-decoded binding_signature as a read/assignment signal, not as a
new loss target. Audit scale and insertion point:

1. measure binding_signature contribution inside `_binding_logits`;
2. compare candidate selection before and after local top-k using the projected
   binding signature score;
3. if the signal is present before top-k but lost after top-k, fix local
   candidate selection;
4. if the signal is weak at assignment time despite high offline AUC, fix the
   binding reader/coefficient schedule;
5. only if both fail, activate real temporal correspondence evidence
   (tracklet/proposal dataflow).
```

This remains a structure/readout repair, not a loss pile-up.

#### 12-Hour Acceptance Matrix

The next checkpoint should be judged by the following matrix:

```text
Case A:
  binding_signature_cos_auc >= 0.70
  support_signature_cos_auc improves materially over raw visual/point cosine

Decision:
  same-object information exists in the deployed signature subspace. The next
  engineering work should be a coefficient/reader audit: verify that binding
  logits and AQR support routing actually use this subspace with the intended
  scale. Do not add more dataflow yet.

Case B:
  geometry_auc high but binding_signature_cos_auc < 0.70

Decision:
  the model can localize a weak same-object candidate geometrically, but the
  exported learned signature is not object-discriminative enough. Do not
  increase identity inertia or predictive losses. Move to token-level
  V-JEPA/PG/wrist probing, and then either improve signature extraction or feed
  real tracklet/proposal evidence.

Case C:
  all same-object AUCs weak and duplicate_candidate_fraction remains high

Decision:
  current CALVIN evidence is insufficient for robust multi-instance binding
  under this debug target. The next coherent upgrade is real temporal
  correspondence evidence (tracklets/proposals), not scalar loss tuning.
```

#### Tail Commands

```bash
# A5
ssh -p 29776 root@36.139.225.68
BASE=$(cat /tmp/a5_signature_probe_base.txt)
tail -f "$BASE/eval.log"
tail -f "$BASE/probe.log"
cat "$BASE/same_object_probe_signature.json"

# A7
ssh -p 28060 root@36.139.225.68
BASE=$(cat /tmp/a7_signature_probe_base.txt)
tail -f "$BASE/driver.log"
tail -f "$BASE/eval.log"
tail -f "$BASE/probe.log"
cat "$BASE/same_object_probe_signature.json"
```

#### Self-Critique

This plan is intentionally not a full new training experiment. That is the
point. The last several runs already showed that loss-level interventions can
produce locally plausible action loss and even low same-role overlap while
leaving stable coverage low. A new cotrain sweep would not distinguish whether
the failure is representational or assignment-side.

The probe answers the more basic question first:

```text
Can current frozen/perception-trained features express "same object" in a
pairwise binding subspace under the current CALVIN observations?
```

Only after that question is answered should we choose between reader/scale
repair, token-level signature extraction, or real tracklet/proposal dataflow.

## 2026-05-13 10-Hour Plan: Signature-Guided Local Candidate Repair

### Diagnosis From A5/A7 Signature Probes

Both A5 and A7 answered the first binding question:

```text
A5 binding_signature_cos_auc: 0.976374
A7 binding_signature_cos_auc: 0.964458

A5 duplicate_candidate_fraction_within_frame: 0.848016
A7 duplicate_candidate_fraction_within_frame: 0.854286
```

This is not an "identity signal absent" failure. The pairwise same-object
subspace exists and is highly decodable. The remaining observed failure is that
candidate selection can still reuse the same local supports across same-role
anchors before posterior binding gets a clean set of observation anchors.

Mathematically, the current local refiner previously selected top-k candidates
from each typed prior:

```math
\Omega_j^{(m)} = TopK_i(p_{j,i}^{(m)})
```

but it did not use the learned pairwise object-binding subspace that was shown
to be decodable by the probe. That creates a mismatch:

```text
offline probe:
  binding_signature can separate same/different object pairs

online local refinement:
  top-k selection may ignore that subspace and duplicate candidates
```

The coherent repair is therefore not a new loss and not a hard coverage rule.
It is to use the existing `binding_signature_proj` as a local readout geometry:

```math
\tilde p_{j,i}^{(m)}
=
softmax_i(
  \log p_{j,i}^{(m)}
  + \lambda_{local-bind}
    \langle
      \phi(q_j), \phi(z_i^{(m)})
    \rangle
)
```

where:

```text
q_j:
  current AQR anchor query after typed readers

z_i:
  candidate typed-memory token

phi:
  existing normalized binding_signature projection

lambda_local-bind:
  guarded experimental coefficient
```

This is structurally consistent with the object-binding paper direction:
object information is often recoverable in a nonlinear/projected subspace, so
the correct integration point is a pairwise readout/reranking term, not a new
classification head or a scalar collapse penalty.

### Code Change

Implemented as a guarded field:

```text
PicfCoreConfig.local_refinement_binding_weight = 0.0
```

and exposed through:

```bash
--local-refinement-binding-weight
```

Default remains `0.0` until the run evidence is available. This avoids silently
changing the maintained baseline. The 10-hour experiment explicitly enables it.

Additional debug metrics:

```text
aqr_same_role_anchor_binding_signature_overlap_max
aqr_same_role_anchor_binding_signature_overlap_mean
aqr_same_role_obs_binding_signature_overlap_max
aqr_same_role_obs_binding_signature_overlap_mean
```

These distinguish:

```text
anchor-query collapse:
  anchor binding signatures overlap before observation anchors.

observation-anchor duplicate collapse:
  observation anchor binding signatures overlap after local readout.

support duplicate collapse:
  local true overlap / local jaccard stay high.
```

### Design Constraints

This plan deliberately does not:

```text
add slot-JEPA/support-prediction pressure;
increase address/cache inertia;
turn on ordinal losses;
restore role competition or coverage-seed heuristics;
force cross-anchor hard exclusion;
change PI0.5 action semantics.
```

Reason:

```text
The probes show the representational subspace is present. The next scientifically
minimal move is to test whether using that subspace at the candidate-selection
site reduces duplicate candidates. Anything stronger would obscure attribution.
```

### 10-Hour Experiment Matrix

Run two independent profiles on A5 and A7.

Both jobs resume from the same 750-step stable local-refinement baseline and
run to absolute step 1650. Keeping `unroll_steps=1` here is intentional: the
best recent endpoint used burn-in 4 + unroll 1, and this experiment isolates
the candidate-selection change rather than retesting unroll length.

#### A5: Conservative Signature Rerank

Purpose:

```text
Test whether a small binding-subspace local rerank improves duplicate support
without harming action-scale training.
```

Profile:

```text
local_refinement_topk: 8
local_refinement_weight: 0.10
local_refinement_binding_weight: 0.25
burnin_steps: 4
unroll_steps: 1
action_prefix_stopgrad: true
slot_jepa/support_pred/binding_consistency: 0
paligemma cotrain: enabled if the current launch profile already uses it
```

Duration:

```text
900 new train steps from the shared 750-step baseline, then CALVIN 1-sequence eval + same-object probe.
```

Expected runtime:

```text
~5-7 hours for train, ~20-40 minutes for eval/probe depending on checkpoint IO.
```

#### A7: Stronger Signature Rerank Stress Test

Purpose:

```text
Check whether stronger signature reranking improves assignment or over-constrains
anchor specialization.
```

Profile:

```text
local_refinement_topk: 8
local_refinement_weight: 0.10
local_refinement_binding_weight: 0.50
burnin_steps: 4
unroll_steps: 1
action_prefix_stopgrad: true
slot_jepa/support_pred/binding_consistency: 0
paligemma cotrain: enabled if the current launch profile already uses it
```

Duration:

```text
900 new train steps from the shared 750-step baseline, then CALVIN 1-sequence eval + same-object probe.
```

Expected runtime:

```text
~5-7 hours for train, ~20-40 minutes for eval/probe.
```

### Acceptance Criteria

Primary structural acceptance:

```text
aqr_same_role_local_true_overlap_max:
  should remain materially below the previous duplicate regime.

aqr_same_role_local_jaccard_max:
  should fall or stay low, not rebound toward broad candidate reuse.

posterior_stable_slot_fraction:
  should not collapse relative to the prior best endpoint.

posterior_recycle_rate:
  should not return to sustained saturation.

loss_action_default_equiv:
  should remain comparable to the current action-scale baseline.
```

Probe acceptance:

```text
binding_signature_cos_auc:
  should remain high (>0.90). If it collapses, the rerank is damaging the
  binding subspace.

duplicate_candidate_fraction_within_frame:
  should improve from ~0.85. If it stays unchanged while local jaccard improves,
  the probe candidate definition may be too broad and needs a token-level audit.
```

Decision tree:

```text
If A5 improves duplicate metrics and A7 over-constrains:
  keep a small coefficient, likely 0.25.

If both improve:
  promote signature-guided local ranking into the next maintained profile after
  a longer cotrain check.

If neither improves but binding AUC remains high:
  the problem is not local top-k score use; move to real tracklet/proposal
  episode dataflow or a deeper assignment audit.

If action loss degrades sharply:
  reduce local_refinement_weight before reducing binding_weight; the problem is
  local residual magnitude, not pairwise subspace ranking.
```

### Current Verification

Local structural checks after adding the guarded repair:

```text
python -m py_compile src/openpi/picf/core/config.py src/openpi/picf/core/pipeline.py scripts/picf_core_train.py scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/verify_picf_owm_contract.py

Result:
  31/31 PASS
```

Additional local checks on the same checkout:

```text
PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
pytest -q scripts/verify_picf_owm_contract_test.py scripts/picf_owm_evidence_bundle_test.py

Result:
  strict diagnose: PASS, with only runtime-artifact WARNs when no metrics/eval
                   path is supplied.
  dataflow trace: PASS.
  MVTrack deep audit: PASS.
  targeted pytest: 4 passed.
```

### Active Remote Runs

Both jobs were launched in tmux after syncing through the China GitHub mirror to
commit:

```text
2a09b12 Clarify signature local experiment schedule
```

A5:

```text
machine:
  ssh -p 29776 root@36.139.225.68

tmux:
  siglocal_a5

run:
  picf_a5_siglocal025_burnin4_from750_to1650_20260513_08fbf31

base:
  /mnt/calvin_eval_logs/picf_a5_siglocal025_burnin4_from750_to1650_20260513_08fbf31

tail:
  BASE=$(cat /tmp/a5_siglocal_base.txt)
  tail -f "$BASE/train.log"
  tail -f "$BASE/driver.log"
  tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_siglocal025_burnin4_from750_to1650_20260513_08fbf31/metrics.jsonl

early speed:
  about 17 s/step after resume.
```

A7:

```text
machine:
  ssh -p 28060 root@36.139.225.68

tmux:
  siglocal_a7

run:
  picf_a7_siglocal050_burnin4_from750_to1650_20260513_08fbf31

base:
  /mnt/calvin_eval_logs/picf_a7_siglocal050_burnin4_from750_to1650_20260513_08fbf31

tail:
  BASE=$(cat /tmp/a7_siglocal_base.txt)
  tail -f "$BASE/train.log"
  tail -f "$BASE/driver.log"
  tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_siglocal050_burnin4_from750_to1650_20260513_08fbf31/metrics.jsonl

early speed:
  about 18.5 s/step after resume.
```

Estimated completion window from launch:

```text
training:
  A5 around 4.5 hours after first training step;
  A7 around 4.8-5.0 hours after first training step.

eval + same-object probe:
  approximately 20-40 minutes after training, depending on checkpoint IO and
  CALVIN server startup.
```

Current rule:

```text
Do not stop or alter these jobs unless one of the following happens:
  - train.log reports a traceback;
  - metrics show NaN/Inf;
  - loss_action_default_equiv or loss_total diverges sharply for multiple logs;
  - both GPUs become idle while driver.log has no "done" marker.
```

### 2026-05-13 Local Re-Audit And Live Remote Check

This checkpoint was taken after documentation commit:

```text
8cce28a Record active signature local runs
```

Local checkout:

```text
branch:
  Posterior_VLA

working tree:
  clean
```

Local audit commands rerun:

```text
python -m py_compile \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/pipeline.py \
  scripts/picf_core_train.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_owm_same_object_probe.py \
  scripts/picf_owm_strict_diagnose.py \
  scripts/picf_owm_dataflow_trace.py \
  scripts/picf_owm_mvtrack_deep_audit.py

PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
pytest -q scripts/verify_picf_owm_contract_test.py scripts/picf_owm_evidence_bundle_test.py
git diff --check
```

Results:

```text
py_compile:
  PASS

verify_picf_owm_contract:
  31/31 PASS

strict diagnose:
  PASS, with only expected runtime-artifact WARNs when no metrics/eval paths
  are supplied.

dataflow trace:
  PASS

MVTrack deep audit:
  PASS

targeted pytest:
  4 passed

git diff --check:
  PASS
```

Live remote status at the same audit point:

```text
A5:
  tmux session: siglocal_a5
  latest logged step: 800
  loss_total: 0.740409
  loss_action_default_equiv: 0.059182
  loss_anchor_pv: 2.391390
  loss_pv_weak: 1.637834
  loss_mapg_routing: 1.141746
  aqr_same_role_support_overlap_max: 0.791688
  aqr_same_role_local_true_overlap_max: 0.082113
  aqr_same_role_local_jaccard_max: 0.636993
  posterior_recycle_rate: 0.228806
  posterior_stable_slot_fraction: 0.111111
  owm_tracklet_tokens: 0
  owm_proposal_tokens: 0

A7:
  tmux session: siglocal_a7
  latest logged step: 775
  loss_total: 0.735043
  loss_action_default_equiv: 0.064387
  loss_anchor_pv: 2.404265
  loss_pv_weak: 1.660162
  loss_mapg_routing: 1.128123
  aqr_same_role_support_overlap_max: 0.530347
  aqr_same_role_local_true_overlap_max: 0.041773
  aqr_same_role_local_jaccard_max: 0.382661
  posterior_recycle_rate: 0.342753
  posterior_stable_slot_fraction: 0.111111
  owm_tracklet_tokens: 0
  owm_proposal_tokens: 0
```

Interpretation:

```text
No local code blocker was found.
Both remote jobs are still running in tmux and should continue.
A7 has the cleaner early local-duplicate signal, but neither job is far enough
for final acceptance. Tracklet/proposal tokens remain zero as expected for this
CALVIN dataflow test, so this round evaluates signature-guided local candidate
use only, not full MVTrack tracklet/proposal behavior.
```

### 2026-05-13 Remote Stop Diagnosis

Later status check found that both remote jobs had stopped before the planned
1650-step target.

Observed remote state:

```text
A5:
  tmux: no server running
  GPU: idle, 0 MiB used on both GPUs
  latest metrics step: 1225
  train.log progress tail: step 1239
  checkpoint artifacts: 900, 1200, latest.pt
  traceback / NaN / Inf in train.log or metrics: not found

A7:
  tmux: no server running
  GPU: idle, 0 MiB used on both GPUs
  latest metrics step: 1200
  train.log progress tail: step 1200
  checkpoint artifacts: 900, latest.pt, tmp_1200
  traceback / NaN / Inf in train.log or metrics: not found
```

Machine-level diagnosis:

```text
df -h /mnt:
  Size 1.8T, Used 1.8T, Avail 0, Use% 100%
```

Interpretation:

```text
The signature-local runs did not finish and should not be treated as completed
1650-step experiments. The failure mode is storage exhaustion on /mnt, not a
clear model-side NaN/Inf/traceback.

A7 appears to have stopped during the 1200 checkpoint write, leaving tmp_1200.
A5 completed the 1200 checkpoint and then ran further, but metrics/checkpoint
persistence stopped before the 1650 target.
```

Usable evidence from this interrupted run:

```text
A5 up to latest metrics step 1225:
  loss_total: 0.725866
  loss_action_default_equiv: 0.063656
  loss_anchor_pv: 2.370466
  loss_pv_weak: 1.547877
  loss_mapg_routing: 1.134488
  aqr_same_role_support_overlap_max: 0.720813
  aqr_same_role_local_true_overlap_max: 0.049409
  aqr_same_role_local_jaccard_max: 0.495348
  posterior_recycle_rate: 0.282764
  posterior_stable_slot_fraction: 0.113333

A7 up to latest metrics step 1200:
  loss_total: 0.744512
  loss_action_default_equiv: 0.059805
  loss_anchor_pv: 2.360714
  loss_pv_weak: 1.555988
  loss_mapg_routing: 1.150512
  aqr_same_role_support_overlap_max: 0.949820
  aqr_same_role_local_true_overlap_max: 0.102317
  aqr_same_role_local_jaccard_max: 0.688900
  posterior_recycle_rate: 0.266900
  posterior_stable_slot_fraction: 0.111111
```

Conclusion:

```text
Do not resume or launch more /mnt-writing experiments until old checkpoints or
eval artifacts are explicitly pruned. The A5/A7 partial metrics are useful as
interrupted diagnostics only:
  - A5 is the better partial result for local duplicate suppression.
  - A7's stronger local_refinement_binding_weight did not clearly improve the
    late same-role overlap and may be too strong.
  - Neither run reached the planned endpoint or post-run probe/eval stage.
```

### 2026-05-13 Storage Cleanup

Cleanup policy:

```text
Keep:
  args.json
  metrics.jsonl
  train_tmux.log / launch logs
  lightweight run roots
  written experiment notes

Remove:
  May-or-later numeric checkpoint payload directories
  May-or-later tmp_* incomplete checkpoint directories
  May-or-later model-only resume checkpoint payloads
  May-or-later eval heavy artifacts: videos, anchor_debug, prediction_debug
```

Removed apparent payload:

```text
about 408 GiB
```

Disk state after cleanup:

```text
/mnt:
  1.8T total
  1.4T used
  409G available
  78% used
```

Important details:

```text
The interrupted A5/A7 signature-local checkpoint payloads were removed, but
their metrics/log records remain.

The cleanup included one checkpoint payload under:
  picf_v22_frozen2x40_photometric_unroll2_30000_ckpt5000_20260430_r1/10000

Rationale:
  although the run name contains 20260430, that checkpoint payload had May
  mtime and was a removable weight artifact under the current "remove May-era
  checkpoint payloads, preserve records" policy.
```

Post-cleanup check:

```text
No May-or-later numeric checkpoint or tmp_* checkpoint payloads remain under:
  /mnt/checkpoints/picf_core/picf_core
```

## 2026-05-13 2-3 Hour Causal Check: Local Signature Rerank

### Question

The interrupted A5/A7 run suggested:

```text
local_refinement_binding_weight=0.25:
  better late local duplicate metrics than 0.50.

local_refinement_binding_weight=0.50:
  stronger early duplicate suppression, but late same-role overlap rebounded.
```

Because those runs stopped early due to `/mnt` exhaustion, the next experiment
must answer a narrower causal question:

```text
Does signature-guided local candidate reranking improve local duplicate
suppression relative to the same profile with the rerank disabled?
```

This is not a new architecture change and not a new loss. It tests whether the
same-object subspace already decoded by `binding_signature_proj` is useful when
applied to the local refinement candidate ranking.

### Mathematical Rationale

The local refiner builds candidate sets:

```math
\Omega_j =
TopK(p_j^{visual})
\cup
TopK(p_j^{temporal})
\cup
TopK(p_j^{point})
\cup
TopK(p_j^{tracklet})
\cup
TopK(p_j^{proposal})
```

The observed failure is not that local evidence is absent. It is that different
same-role anchors can select overlapping local candidate sets:

```math
|\Omega_i \cap \Omega_j| / |\Omega_i \cup \Omega_j|
```

can remain high even when action loss decreases.

The repair under test modifies only the local candidate distribution:

```math
\tilde p_{j,k}
\propto
p_{j,k}
\lambda_{local-bind}
\langle
  W_b q_j,\ W_b z_k
\rangle
```

where `W_b` is the existing projected same-object subspace. This is coherent
with the object-binding probe result:

```text
binding_signature AUC is high, but duplicate local candidate fraction remains
high.
```

So the hypothesis is:

```text
the representation has separability, but the local readout is not using it
strongly enough.
```

A higher weight is not automatically better. If `lambda_local-bind` is too high,
anchors can chase the same dominant same-object signature and reduce local
diversity. The interrupted A7 run is consistent with that failure mode.

### Design

Use the preserved full PICF baseline checkpoint as the common start point:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_v22_full_picf_a6_30000_ckpt2500_print100_p192_20260424_r2/10000
```

Use the current May template args for the guarded MVTrack profile, with:

```text
action_prefix_stopgrad = true
burnin_steps = 4
unroll_steps = 1
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
local_refinement_weight = 0.10
save_interval = 100
keep_last_checkpoints = 1
num_train_steps = 10300
```

Two runs differ only in `local_refinement_binding_weight`:

```text
A5 control:
  local_refinement_binding_weight = 0.0
  run = picf_a5_siglocal000_full10000_to10300_20260513_2a09b12

A7 test:
  local_refinement_binding_weight = 0.25
  run = picf_a7_siglocal025_full10000_to10300_20260513_2a09b12
```

### Acceptance Criteria

The test run is better only if it satisfies all of:

```text
1. aqr_same_role_local_true_overlap_max lower than control.
2. aqr_same_role_local_jaccard_max lower than control.
3. aqr_same_role_support_overlap_max not worse by more than about 0.1.
4. loss_action_default_equiv not worse by more than about 10%.
5. posterior_recycle_rate does not rise toward saturation.
6. post-run same-object probe remains high-AUC.
```

If A7 improves local overlap without action/recycle degradation:

```text
promote local_refinement_binding_weight=0.25 as the next maintained local
candidate profile.
```

If A7 does not improve over A5:

```text
do not keep pushing local rerank weight. Move to dataflow-level evidence:
tracklet/proposal activation or local assignment visualization.
```

If A7 worsens same-role support overlap:

```text
the 0.25 structural prior is already too strong from this checkpoint; keep the
binding-signature term in posterior binding only and disable it inside local
refinement by default.
```

### 2026-05-13 10:50 Status: Full10000 Resume Invalidated

The first paired launch from the preserved April full-PICF checkpoint did not
reach training:

```text
A5:
  picf_a5_siglocal000_full10000_to10300_20260513_2a09b12

A7:
  picf_a7_siglocal025_full10000_to10300_20260513_2a09b12

status:
  both failed during FSDP checkpoint load before metrics.jsonl was created.
```

This is not evidence against local signature reranking. It is a checkpoint
architecture mismatch:

```text
old checkpoint:
  April full-PICF architecture.

current code:
  MVTrack/AQR architecture with many new AQR readers, query tokens,
  binding_signature_proj, tracklet/proposal projections, temporal view
  embeddings, and a widened visual prediction head.
```

The loader correctly refused a broad migration. The failure included many
missing AQR/MVTrack parameters and shape mismatches such as:

```text
core.binding_signature_proj.*
core.aqr_*_reader.*
core.aqr_physical_query_tokens
core.aqr_task_query_tokens
core.tracklet_token_proj.*
core.proposal_token_proj.*
core.temporal_visual_view_embedding.weight
core.visual_real_head: checkpoint [48, 512] vs model [12288, 512]
core.visual_real_error_encoder: checkpoint [128, 144] vs model [128, 36864]
```

Mathematically, allowing this with a loose compatibility rule would not be a
clean experiment: the action backbone would be restored, but the AQR/MVTrack
router under test would be mostly random-initialized. That would mix
architecture migration, initialization shock, and local reranking into one
confounded run.

### Replacement 2-3 Hour Causal Check

Because May checkpoint payloads were intentionally pruned to free `/mnt`, no
same-architecture resume checkpoint remains available. The clean replacement is
a fresh paired run from the same current MVTrack args and seed:

```text
A5 control:
  run = picf_a5_siglocal000_fresh300_20260513_2a09b12
  local_refinement_binding_weight = 0.0

A7 test:
  run = picf_a7_siglocal025_fresh300_20260513_2a09b12
  local_refinement_binding_weight = 0.25

shared:
  resume_checkpoint = null
  num_train_steps = 300
  burnin_steps = 4
  unroll_steps = 1
  action_prefix_stopgrad = true
  local_refinement_topk = 8
  local_refinement_weight = 0.10
  lambda_slot_jepa = 0
  lambda_support_pred = 0
  lambda_binding_consistency = 0
  lambda_aqr_denoising = 0
```

This replacement answers a narrower but still useful question:

```text
Does the local signature rerank term produce better early local candidate
separation than the same fresh MVTrack profile with the term disabled?
```

It does not answer:

```text
Can current MVTrack safely resume from the old April full-PICF 10000-step
checkpoint?
```

That would require either a deliberately engineered one-time migration path or
a new same-architecture warm checkpoint. The former is not appropriate for this
diagnostic unless we explicitly accept that most AQR/MVTrack parameters are
newly initialized.

### 2026-05-13 11:05 Fresh Local-Signature Causal Check: Early Result

The fresh paired check is already sufficient to reject stronger local signature
reranking as the next default.

Observed A5 control, `local_refinement_binding_weight=0.0`:

```text
step 100:
  loss_action_default_equiv = 0.08214
  aqr_same_role_support_overlap_max = 0.33274
  aqr_same_role_local_jaccard_max = 0.04121
  aqr_same_role_local_true_overlap_max = 0.00388
  aqr_effective_anchor_count = 23.16
  posterior_recycle_rate = 0.81983
  posterior_recycle_logit_mean = 1.5387
  posterior_residual_summary_norm = 230.37
  posterior_address_update_rate_mean = 0.00790

step 150:
  loss_action_default_equiv = 0.06677
  aqr_same_role_support_overlap_max = 0.28883
  aqr_same_role_local_jaccard_max = 0.12883
  aqr_same_role_local_true_overlap_max = 0.00825
  aqr_effective_anchor_count = 23.35
  posterior_recycle_rate = 0.92052
  posterior_recycle_logit_mean = 2.5168
  posterior_residual_summary_norm = 239.57
  posterior_address_update_rate_mean = 0.00341
```

Observed A7 test, `local_refinement_binding_weight=0.25`:

```text
step 100:
  loss_action_default_equiv = 0.08891
  aqr_same_role_support_overlap_max = 0.50247
  aqr_same_role_local_jaccard_max = 0.16337
  aqr_same_role_local_true_overlap_max = 0.04555
  aqr_effective_anchor_count = 19.97
  posterior_recycle_rate = 0.99037
  posterior_recycle_logit_mean = 4.6980
  posterior_residual_summary_norm = 261.89
  posterior_address_update_rate_mean = 0.00042

step 125:
  loss_action_default_equiv = 0.07492
  aqr_same_role_support_overlap_max = 0.68613
  aqr_same_role_local_jaccard_max = 0.29834
  aqr_same_role_local_true_overlap_max = 0.05239
  aqr_effective_anchor_count = 21.70
  posterior_recycle_rate = 0.99535
  posterior_recycle_logit_mean = 5.4812
  posterior_residual_summary_norm = 248.70
  posterior_address_update_rate_mean = 0.00020
```

Interpretation:

```text
1. A5 proves the no-rerank profile can already reduce same-role support overlap
   and local duplicate metrics early.
2. A7 proves moderate local signature reranking worsens recycle saturation,
   address-update starvation, effective anchor count, and local duplicate
   metrics under the same fresh profile.
3. Therefore the issue is not "the representation lacks same-object signal" and
   not "local rerank weight should be increased." The immediate root is that
   recycle probability can still saturate from the residual path.
```

This is mathematically consistent with the posterior update:

```math
bar h_j = (1-recycle_j)h_j^- + recycle_j h_{res}
```

When `recycle_j -> 1` for most slots, the prior identity path is erased, the
address update rate collapses because it is gated by `(1-recycle_j)`, and
multiple slots can be reset toward the same residual evidence. This is a reset
dominance failure, not a missing-loss problem.

### 2026-05-13 11:15 Root-Cause Repair And Two-Hour Deployment

The next deployed repair is:

```text
recycle_normalize_residual_summary = true
local_refinement_binding_weight = 0.0
```

Implementation:

```math
r = sum_i d_i o_i
hat r = LayerNorm(r)
recycle_j = sigma(f(h_j^-, support_j, var_j, hat r, alpha_j^-))
```

Only the recycle gate sees `hat r`. The residual heads still consume the
original `r`, so the model does not lose residual content. The repair is
therefore a trust-gate scale normalization, not a new supervision term and not
a hand-coded ownership rule.

Acceptance for the first A7 run:

```text
posterior_recycle_rate should stay well below the A7 0.99 saturation zone;
posterior_address_update_rate_mean should not collapse toward 0;
aqr_effective_anchor_count should stay close to A5 control;
aqr_same_role_support_overlap_max should not rebound above 0.7 early;
local_jaccard/local_true should not worsen versus A5 control;
loss_action_default_equiv should remain comparable to A5 control.
```

Deployment order:

```text
1. Stop A7 local-rerank run because it has already failed the causal gate.
2. Sync the recycle-normalization patch to A7.
3. Start fresh 300-step A7 normalized-recycle run.
4. Let A5 no-rerank control finish to 300 unless the GPU is needed.
5. If A7 normalized-recycle passes the early gates, repeat on A5 or extend to
   600/1200; if not, do not add losses. Inspect recycle feature scaling and
   support-mass/dustbin evidence next.
```

Expected readout window:

```text
first comparable metrics: ~25 steps after launch;
useful decision: ~125-200 steps;
full 300-step check: about 1.5-2.5 hours depending on current I/O.
```

Actual A7 deployment:

```text
machine:
  A7 / ssh -p 28060 root@36.139.225.68

commit:
  b286c3e Refresh dataflow audit line refs

tmux:
  recyclenorm_a7

run:
  picf_a7_recyclenorm000_fresh300_20260513_b286c3e

log base:
  /mnt/calvin_eval_logs/picf_a7_recyclenorm000_fresh300_20260513_b286c3e

checkpoint dir:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_recyclenorm000_fresh300_20260513_b286c3e
```

Live inspection:

```bash
ssh -p 28060 root@36.139.225.68
BASE=$(cat /tmp/a7_recyclenorm_base.txt)
tail -f "$BASE/train.log"
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_recyclenorm000_fresh300_20260513_b286c3e/metrics.jsonl
```

A5 control remains active for comparison:

```bash
ssh -p 29776 root@36.139.225.68
BASE=$(cat /tmp/a5_sigcausal_base.txt)
tail -f "$BASE/train.log"
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_siglocal000_fresh300_20260513_2a09b12/metrics.jsonl
```

#### A7 Normalized-Recycle Result

The A7 normalized-recycle diagnostic completed the planned 300-step run.

Final metrics at step 300:

```text
loss_total: 0.735033
loss_action_default_equiv: 0.064488
aqr_same_role_support_overlap_max: 0.387882
aqr_same_role_local_jaccard_max: 0.268997
aqr_same_role_local_true_overlap_max: 0.024479
aqr_effective_anchor_count: 23.247017
posterior_recycle_rate: 0.540068
posterior_recycle_logit_mean: 0.160628
posterior_address_update_rate_mean: 0.019599
posterior_stable_slot_fraction: 0.111111
```

Tail-5 metric averages:

```text
loss_action_default_equiv: 0.065913
aqr_same_role_support_overlap_max: 0.495418
aqr_same_role_local_jaccard_max: 0.309623
aqr_same_role_local_true_overlap_max: 0.028179
aqr_effective_anchor_count: 23.208909
posterior_recycle_rate: 0.537484
posterior_recycle_logit_mean: 0.150239
posterior_address_update_rate_mean: 0.019662
posterior_stable_slot_fraction: 0.111111
```

Interpretation:

```text
1. The recycle-gate scale repair is validated for the diagnosed failure mode.
   The previous failed A7 local-rerank trial saturated at recycle≈0.99 with
   near-zero address update. With residual-summary normalization, A7 stays near
   recycle≈0.54, recycle_logit≈0.15, and address_update≈0.02.

2. The repair does not rely on a new loss, action suppression, or heuristic
   ownership rule. It keeps the belief-filter semantics intact: recycle is a
   trust/reset gate over normalized evidence direction rather than a proxy for
   unbounded residual magnitude.

3. Same-role support overlap no longer shows the 0.99 collapse pattern in this
   run. The final row is 0.388 and the tail-5 average is 0.495. Local Jaccard
   remains moderate, so the next theoretical stage should focus on coverage
   and local-candidate distribution rather than recycle saturation.

4. This is a diagnostic run, not a behavior checkpoint. The 300-step checkpoint
   payload was deleted after metrics/log capture to free /mnt space. Preserved
   evidence files are args.json, metrics.jsonl, stackdump files, and the train
   log under /mnt/calvin_eval_logs.
```

Storage status after cleanup:

```text
deleted:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_recyclenorm000_fresh300_20260513_b286c3e/300
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_recyclenorm000_fresh300_20260513_b286c3e/latest.pt

retained:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_recyclenorm000_fresh300_20260513_b286c3e/args.json
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_recyclenorm000_fresh300_20260513_b286c3e/metrics.jsonl
  /mnt/calvin_eval_logs/
    picf_a7_recyclenorm000_fresh300_20260513_b286c3e/train.log
```

### 2026-05-13 Next-Stage Local-Refinement Attribution Plan

#### Question

After normalized recycle input fixed the recycle/address failure chain, the
remaining question is not "how do we force stronger object ownership?" The
remaining question is:

```text
Does the local refinement residual help coverage, or does it reintroduce
same-role local-candidate reuse after AQR has already produced usable support?
```

This must be answered before adding any new loss, stronger address/cache
inertia, or new proposal source.

#### Self-critique Against Rejected Repairs

The 2026-05-12 experiments already rejected hard role-wise local competition and
deterministic coverage-seeded local proposals. Those failures are not ignored.
Therefore the next experiment deliberately does not add a new coverage seed,
ownership rule, IsSameObject loss, SAM/DINO proposal, or tracklet dependency.

The only tested variables are existing residual strength and existence of local
refinement:

```text
local refinement off:
  tests whether the local refiner itself is currently necessary.

local refinement light:
  tests whether the existing typed-memory refiner is useful but too strong.
```

This is a module-attribution experiment, not a new architecture patch.

#### Mathematical Contract

Current local refinement applies:

```math
q'_j = q_j + \lambda_{local}(\bar m_j - q_j)
```

where `\bar m_j` is the top-k aggregation over existing typed evidence:

```text
visual, temporal, point, tracklet, proposal
```

The experiment varies only `\lambda_{local}` and the presence of this residual.
It preserves:

```text
posterior authority
normalized recycle gate
action path
cache residual gate
no future leakage
no additional loss pressure
no hard object ownership
```

Acceptance should be judged by the joint vector:

```text
aqr_same_role_support_overlap_max
aqr_same_role_local_jaccard_max
aqr_same_role_local_true_overlap_max
aqr_effective_anchor_count
posterior_recycle_rate
posterior_recycle_logit_mean
posterior_address_update_rate_mean
loss_action_default_equiv
loss_total
```

If local-off is as healthy as local-on, local refinement is not justified for
the current training profile and should remain disabled or delayed. If
local-light keeps action/alignment benefits while lowering local Jaccard, the
next maintained profile should reduce local residual strength rather than add a
new module.

#### Relation To 2025 Object-Binding Evidence

The 2025 object-binding ViT result argues that pretrained ViTs can encode
same-object pairwise relations in a low-dimensional subspace and that those
relations guide attention. It does not imply that downstream training should
add hard ownership or deterministic spatial seeds. For PICF, the consistent
interpretation is:

```text
use the binding subspace as diagnostic/support evidence;
do not let it override posterior correction;
first verify whether the existing local residual preserves or destroys that
subspace during cotrain.
```

#### Two-Hour Deployment Matrix

Deployment revision:

```text
local/remote commit:
  5241279

code change:
  none beyond the already committed normalized recycle-gate repair and docs.

reason:
  this is an attribution test over existing local-refinement residual strength,
  not a new architecture patch.
```

A7 runs the strongest ablation:

```text
name:
  picf_a7_recyclenorm_localoff_fresh300_20260513_5241279

tmux:
  localoff_a7

metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_recyclenorm_localoff_fresh300_20260513_5241279/metrics.jsonl

train log:
  /mnt/calvin_eval_logs/picf_a7_recyclenorm_localoff_fresh300_20260513_5241279/train.log

overrides:
  num_train_steps=300
  save_interval=100000
  burnin_steps=4
  unroll_steps=1
  action_prefix_stopgrad=true
  local_refinement_enabled=false
  local_refinement_weight=0.0
  local_refinement_binding_weight=0.0
  recycle_normalize_residual_summary=true
  lambda_slot_jepa=0.0
  lambda_support_pred=0.0
  lambda_binding_consistency=0.0
  lambda_aqr_denoising=0.0
```

A5 runs the light-residual variant:

```text
name:
  picf_a5_recyclenorm_locallight_fresh300_20260513_5241279

tmux:
  locallight_a5

metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_recyclenorm_locallight_fresh300_20260513_5241279/metrics.jsonl

train log:
  /mnt/calvin_eval_logs/picf_a5_recyclenorm_locallight_fresh300_20260513_5241279/train.log

overrides:
  num_train_steps=300
  save_interval=100000
  burnin_steps=4
  unroll_steps=1
  action_prefix_stopgrad=true
  local_refinement_enabled=true
  local_refinement_topk=8
  local_refinement_weight=0.05
  local_refinement_binding_weight=0.0
  recycle_normalize_residual_summary=true
  lambda_slot_jepa=0.0
  lambda_support_pred=0.0
  lambda_binding_consistency=0.0
  lambda_aqr_denoising=0.0
```

Expected first useful readout:

```text
step 25: first logged scalar sanity check
step 125-150: early causal direction
step 300: full two-hour attribution result
```

#### Step-25 Sanity Readout

The first scalar row was written by both machines. It is too early for an
acceptance conclusion, but it rules out one simplistic explanation.

```text
A7 local-off:
  step=25
  loss_total=1.1109
  loss_action_default_equiv=0.1995
  aqr_same_role_support_overlap_max=0.9082
  aqr_same_role_local_jaccard_max=0.0
  aqr_same_role_local_true_overlap_max=0.0
  aqr_effective_anchor_count=23.31
  posterior_recycle_rate=0.5580
  posterior_recycle_logit_mean=0.2333
  posterior_address_update_rate_mean=0.0202

A5 local-light:
  step=25
  loss_total=1.1164
  loss_action_default_equiv=0.2151
  aqr_same_role_support_overlap_max=0.8986
  aqr_same_role_local_jaccard_max=0.4867
  aqr_same_role_local_true_overlap_max=0.0253
  aqr_effective_anchor_count=23.30
  posterior_recycle_rate=0.5575
  posterior_recycle_logit_mean=0.2310
  posterior_address_update_rate_mean=0.0202
```

Interpretation:

```text
1. Early same-role support overlap is high in both local-off and local-light.
   Therefore the step-25 overlap cannot be attributed to local refinement alone.
2. A5 local-light has overlapping local top-k sets, but its weighted local true
   overlap remains low. This means local candidate reuse and local mass reuse
   must be separated; high Jaccard by itself is not collapse.
3. Recycle is around 0.56 on both machines, matching the previous normalized
   recycle diagnostic's early/tail level. This is not the old recycle=0.99
   saturation, but it is still high enough that the step-125/150 trend matters.
4. No gradient instability is visible at the first row: grad_norm is 0.87 on A7
   and 1.58 on A5, both below the fixed clip threshold.
```

Do not stop either run on the step-25 row. The next causal discriminator is
whether local-light separates from local-off by step 125-150 in action loss,
support overlap, recycle trend, or local true overlap.

#### Step-175 Midpoint Readout

Both machines are still running and have written seven scalar rows through
step 175. The attribution signal is now stronger than at step 25, but it is
not yet a final decision.

```text
A7 local-off:
  step=175
  loss_total=0.7710
  loss_action_default_equiv=0.0821
  loss_alignment=0.7607
  loss_anchor_pv=2.3783
  loss_pv_weak=2.5045
  loss_mapg_routing=1.0795
  loss_mapg_support_diversity=0.3748
  aqr_same_role_support_overlap_max=0.5633
  aqr_same_role_local_jaccard_max=0.0
  aqr_same_role_local_true_overlap_max=0.0
  aqr_effective_anchor_count=23.10
  posterior_recycle_rate=0.5152
  posterior_recycle_logit_mean=0.0607
  posterior_address_update_rate_mean=0.0215
  posterior_identity_switch_rate=0.7356
  preclip_grad_norm=0.4712
  grad_clip_applied=false

A5 local-light:
  step=175
  loss_total=0.7646
  loss_action_default_equiv=0.0840
  loss_alignment=0.7541
  loss_anchor_pv=2.3332
  loss_pv_weak=2.5735
  loss_mapg_routing=1.0535
  loss_mapg_support_diversity=0.4024
  aqr_same_role_support_overlap_max=0.5164
  aqr_same_role_local_jaccard_max=0.2106
  aqr_same_role_local_true_overlap_max=0.0212
  aqr_local_source_mass_visual=0.5015
  aqr_local_source_mass_point=0.2795
  aqr_local_source_mass_temporal=0.2190
  aqr_effective_anchor_count=23.24
  posterior_recycle_rate=0.5405
  posterior_recycle_logit_mean=0.1623
  posterior_address_update_rate_mean=0.0186
  posterior_identity_switch_rate=0.8200
  preclip_grad_norm=9.4832
  grad_clip_applied=true
```

Interpretation:

```text
1. Local-off remains viable. A7 has no local refinement residual and still
   reaches lower recycle rate and higher address-update rate than A5 by
   step 175. This means the local refiner is not necessary for early
   non-collapse after normalized recycle repair.

2. Local-light has slightly better loss_total, alignment, anchor_pv, and
   routing at the same step, but it pays for this with higher recycle,
   lower address update, higher identity switch, and one fixed-threshold
   gradient clipping event. This is not a catastrophic failure, but it is
   evidence that the residual is heavier than local-off.

3. A5 local true-overlap is still low even when local Jaccard is nonzero.
   The local refiner is not simply putting all same-role mass on exactly
   the same local evidence. The remaining issue is broader assignment /
   identity stability, not just local mass collapse.

4. Same-role support overlap is no longer in the old 0.99 collapse regime,
   but it has rebounded from the A7 step-100 low point. Do not finalize from
   step 175; step 300 is needed to see whether the rebound stabilizes or
   keeps increasing.
```

Current decision:

```text
continue both runs to step 300.

If A7 stays comparable in loss and better in recycle/address/overlap, the
maintained training profile should keep local refinement disabled or delayed.

If A5 keeps a clear action/alignment advantage without further overlap or
gradient deterioration, local-light can remain an optional evidence residual.
```

#### Local-Refinement Archive Decision

The maintained code path now treats local refinement as legacy/archived rather
than production routing.

```text
default config:
  legacy_local_refinement_opt_in=false
  local_refinement_enabled=false
  local_refinement_topk=0
  local_refinement_weight=0.0

to reproduce archived local-refinement ablations, an operator must set all:
  --legacy-local-refinement-opt-in
  --local-refinement-enabled
  --local-refinement-topk > 0
  --local-refinement-weight > 0
```

Reasoning:

```text
1. The root repair for the observed recycle/address failure chain is normalized
   recycle input, not a stronger local reread.

2. A7 local-off remains viable through step 250 with no gradient clipping and
   with lower recycle / higher address update than A5 local-light.

3. A5 local-light gives slightly lower alignment/anchor loss but adds recycle,
   identity-switch, and gradient pressure. This is a poor default tradeoff for
   a maintained belief-state router.

4. Removing it from the default profile preserves mathematical consistency:
   AQR typed supports remain the measurement router; posterior correction is
   the belief update; cache is residual historical context; local top-k reread
   is not a second unproven residual evidence path in production.
```

This is not a claim that local evidence can never be useful. It is a cleanup
decision: keep the archived path behind an explicit opt-in until a future
long-run ablation proves that its benefit exceeds the extra instability.

#### Step-300 Final Readout For Local-Refinement Attribution

Both attribution runs completed 300 steps.

```text
A7 local-off final:
  loss_total=0.7315
  loss_action_default_equiv=0.0657
  loss_anchor_pv=2.2986
  loss_mapg_routing=1.1015
  loss_mapg_support_diversity=0.3182
  aqr_same_role_support_overlap_max=0.4350
  aqr_effective_anchor_count=23.38
  posterior_recycle_rate=0.5034
  posterior_recycle_logit_mean=0.0138
  posterior_address_update_rate_mean=0.0218
  posterior_identity_switch_rate=0.7644
  preclip_grad_norm=0.4163
  grad_clip_applied=false

A5 local-light final:
  loss_total=0.7232
  loss_action_default_equiv=0.0670
  loss_anchor_pv=2.2480
  loss_mapg_routing=1.0575
  loss_mapg_support_diversity=0.3335
  aqr_same_role_support_overlap_max=0.3688
  aqr_same_role_local_jaccard_max=0.2158
  aqr_same_role_local_true_overlap_max=0.0190
  aqr_effective_anchor_count=23.31
  posterior_recycle_rate=0.5359
  posterior_recycle_logit_mean=0.1440
  posterior_address_update_rate_mean=0.0191
  posterior_identity_switch_rate=0.7244
  preclip_grad_norm=0.8227
  grad_clip_applied=false
```

Tail-3 averages:

```text
A7 local-off:
  loss_total=0.7352
  action_default_equiv=0.0666
  same_role_support_overlap=0.4606
  recycle_rate=0.5026
  recycle_logit_mean=0.0102
  address_update_rate=0.0220
  identity_switch_rate=0.7600
  preclip_grad_norm=0.5184

A5 local-light:
  loss_total=0.7259
  action_default_equiv=0.0668
  same_role_support_overlap=0.3538
  local_true_overlap=0.0181
  recycle_rate=0.5350
  recycle_logit_mean=0.1401
  address_update_rate=0.0192
  identity_switch_rate=0.7407
  preclip_grad_norm=0.9147
```

Final attribution:

```text
1. Local-light is not a simple failure. It improves loss_total, anchor_pv,
   routing, and final same-role support overlap in this 300-step window.

2. Local-light still carries a worse belief-filter trust profile: higher
   recycle rate, higher recycle logit, lower address-update rate, and a prior
   clipping event at step 175. These are exactly the variables that caused the
   previous recycle/address failure chain.

3. Local-off reaches nearly identical action_default_equiv, maintains healthy
   effective anchor count, avoids clipping, and preserves the cleaner
   recycle/address dynamics. Since the long-run objective is stable belief
   cotrain rather than a small 300-step anchor-PV advantage, local-off is the
   better maintained default.
```

Decision:

```text
Archive local refinement as a legacy opt-in ablation path.
Do not remove the implementation yet, because it produced a measurable
short-window alignment benefit and may be useful for future controlled
experiments. Do remove it from the production default and make activation
deliberately explicit.

Next production-style long run should use:
  legacy_local_refinement_opt_in=false
  local_refinement_enabled=false
  local_refinement_topk=0
  local_refinement_weight=0.0
  recycle_normalize_residual_summary=true
  action_prefix_stopgrad=true
  high-risk predictive/denoising losses still zero
```

### 2026-05-13 Recycle-Normalization Closure Plan

#### Question

The previous diagnosis identified normalized dustbin residual input as the
root repair for recycle/address saturation. The remaining closure question is
not whether to add another module. The remaining question is:

```text
Which causal, per-sample scale anchor should the recycle trust gate use?
```

The code now exposes:

```text
recycle_residual_norm_mode:
  layernorm
  rmsnorm
  none
```

while retaining the compatibility switch:

```text
recycle_normalize_residual_summary=true
```

#### Mathematical Contract

The recycle gate sees:

```math
z_j = [h^-_j,\ m_j,\ \bar S^-_j,\ N(r),\ \alpha^-_j]
```

where:

```math
r = \sum_i d_i o_i
```

is the dustbin-weighted residual evidence. The production failure was:

```text
||r|| became a magnitude shortcut for recycle/reset.
```

The maintained fix is to remove that radial shortcut:

```math
N_{LN}(r) = (r-\mu(r)) / \sqrt{\sigma^2(r)+\epsilon}
```

The ablation asks whether preserving the residual mean/DC component is useful:

```math
N_{RMS}(r) = r / \sqrt{mean(r^2)+\epsilon}
```

No quantile normalization is used in forward. Quantile transforms are
distribution-level, non-linear rank maps; they are useful for offline
diagnostics but inappropriate for an online per-step belief gate because they
would introduce batch/history dependence or collapse extreme evidence to
saturation boundaries.

#### Experimental Matrix

A7 is the production-candidate closure run:

```text
name:
  picf_a7_recyclenorm_layernorm_closure300_20260513

purpose:
  validate the current maintained default after local-refinement archival.

key settings:
  recycle_normalize_residual_summary=true
  recycle_residual_norm_mode=layernorm
  legacy_local_refinement_opt_in=false
  local_refinement_enabled=false
  local_refinement_weight=0.0
  action_prefix_stopgrad=true
  lambda_slot_jepa=0.0
  lambda_support_pred=0.0
  lambda_binding_consistency=0.0
  lambda_aqr_denoising=0.0
```

A5 is the conservative normalization ablation:

```text
name:
  picf_a5_recyclenorm_rmsnorm_closure300_20260513

purpose:
  test whether preserving residual mean/DC through RMSNorm improves identity
  stability without returning to norm-driven recycle saturation.

key settings:
  same as A7, except recycle_residual_norm_mode=rmsnorm.
```

#### Acceptance Criteria

LayerNorm remains the maintained default if it has:

```text
posterior_recycle_rate <= RMSNorm + 0.03
posterior_address_update_rate >= RMSNorm - 0.003
same_role_support_overlap comparable or better
no persistent grad clipping
action_default_equiv comparable
```

RMSNorm becomes the next candidate only if it improves at least two identity
stability metrics without increasing recycle saturation:

```text
lower identity_switch
lower same_role_support_overlap
higher address_update
same or lower posterior_recycle_rate
same or lower action_default_equiv
```

`none` is not part of this two-hour closure run because the old failure already
showed unnormalized residual magnitude can dominate the trust gate.

#### Self-Critique

This is not a patch-on-patch repair:

```text
1. The module topology is unchanged.
2. Local refinement stays archived/off.
3. No predictive/denoising loss is enabled.
4. No hard ownership rule, quantile map, or dataset-specific heuristic is added.
5. The only tested variable is the scale anchor for one known failure point:
   recycle trust-gate residual evidence.
```

If both LayerNorm and RMSNorm remain healthy, the next phase can move from
diagnostic closure to a longer production-style run. If RMSNorm is unstable,
the current LayerNorm repair is accepted as the clean default.

#### Live Observation Schedule

The two closure runs were launched on 2026-05-13 after commit `4ec25ae`.

```text
A7:
  run: picf_a7_recyclenorm_layernorm_closure300_20260513_4ec25ae
  machine: px-cloud2 / A7
  tmux: recyclenorm_layernorm_a7
  norm: layernorm

A5:
  run: picf_a5_recyclenorm_rmsnorm_closure300_20260513_4ec25ae
  machine: px-cloud1 / A5
  tmux: recyclenorm_rmsnorm_a5
  norm: rmsnorm
```

At the mid/late observation point, both runs were alive, used real
`visual_mode=encoder`, kept local refinement archived/off, and had no gradient
clipping. A7 had reached metrics row `step=225`; A5 had reached metrics row
`step=250`. The remaining wall-clock time was estimated from observed speed:

```text
A7:
  progress log around 246/300
  observed speed ~= 17.7-18.0 sec/step
  sleep window to final readout: 1100 sec

A5:
  progress log around 254/300
  observed speed ~= 17.2-18.2 sec/step
  sleep window to final readout: 1000 sec
```

Do not finalize from the midpoint rows. The accepted readout is the final row
plus tail-3/tail-5 means, because same-role overlap and recycle rate have shown
batch-level spikes in earlier diagnostics.

#### Theory Check Before Final Readout

The successful outcome of this run would close a specific failure chain:

```text
unbounded dustbin residual norm
  -> recycle/reset shortcut
  -> address-update suppression or unstable reset pressure
  -> misleading action loss improvement with unhealthy belief state
```

It would not close every open scientific question. In particular, action loss is
necessary but not sufficient. A valid next phase requires:

```text
action_default_equiv decreases
posterior_recycle_rate stays away from 0/1 saturation
posterior_address_update_rate_mean remains nonzero
aqr_same_role_support_overlap_max does not rebound to the 0.95-0.99 collapse zone
stable-slot identity switch remains near zero
effective anchor count remains high
temporal / PG supports remain non-empty and non-uniform
```

This interpretation is also aligned with the recent object-binding and video
world-model literature:

```text
Object-binding ViT result:
  Pretrained ViTs can encode IsSameObject in a low-dimensional quadratic
  pairwise subspace, but the paper's implication is not "attention alone solves
  binding for control." It supports our moderate binding-signature term and
  offline IsSameObject probe; it does not justify adding hard ownership losses
  or direct slot-JEPA pressure before identity is stable.

V-JEPA 2 / video world-model result:
  Predictive video representations are useful for physical planning and robot
  world models, but future latent evidence must remain a target or diagnostic,
  not a current-step shortcut. This supports the current choice to keep
  slot-JEPA/support-prediction lambdas at zero during this closure run.

Object-centric / geometry-grounded VLA result:
  Explicit object-centric and geometry-aware grounding improves robustness in
  clutter, but the evidence argues for clean typed evidence and belief-state
  correction, not for piling residual modules into the action path.
```

References checked for this reasoning:

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
  arXiv:2510.24709 / NeurIPS 2025 Spotlight

V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
  arXiv:2506.09985

Clutter-Resistant Vision-Language-Action Models through Object-Centric and Geometry Grounding
  arXiv:2512.22519
```

#### Decision After Final Rows

If LayerNorm and RMSNorm are both healthy:

```text
1. Keep LayerNorm as default unless RMSNorm has a clear identity-stability win.
2. Do not re-enable local refinement by default.
3. Do not enable slot-JEPA/support-pred/binding-consistency yet.
4. Move to a longer production-style run with the same guarded profile.
5. Use action loss as a major metric, but retain recycle/address/overlap/stable
   identity as hard health gates.
```

If either run shows recycle or overlap regression:

```text
1. Do not add modules or losses.
2. Inspect recycle feature scaling, support assignment tails, and stable-slot
   coverage first.
3. Treat action loss as non-diagnostic until belief-state health is restored.
```

#### Final Readout: LayerNorm vs RMSNorm Closure

Both runs completed the planned 300 steps and exited their tmux sessions.

```text
A7 LayerNorm final row:
  step=300
  loss_total=0.739813
  loss_action_default_equiv=0.065339
  loss_anchor_pv=2.323036
  loss_pv_weak=2.242894
  loss_mapg_cycle=0.458378
  loss_mapg_support_diversity=0.305271
  loss_mapg_routing=1.068413
  aqr_same_role_support_overlap_max=0.316708
  aqr_effective_anchor_count=23.2881
  posterior_recycle_rate=0.505580
  posterior_recycle_logit_mean=0.022324
  posterior_address_update_rate_mean=0.021289
  posterior_identity_switch_rate=0.797778
  posterior_identity_switch_rate_stable=0.0
  posterior_stable_slot_fraction=0.111111
  preclip_grad_norm=0.737325
  grad_clip_applied=false

A7 LayerNorm tail-5:
  loss_total=0.746271
  loss_action_default_equiv=0.066861
  aqr_same_role_support_overlap_max=0.473148
  aqr_effective_anchor_count=23.2192
  posterior_recycle_rate=0.510424
  posterior_recycle_logit_mean=0.041720
  posterior_address_update_rate_mean=0.021164
  posterior_identity_switch_rate=0.769333
  posterior_identity_switch_rate_stable=0.0
  posterior_stable_slot_fraction=0.111556
  preclip_grad_norm=0.645776
```

```text
A5 RMSNorm final row:
  step=300
  loss_total=0.737683
  loss_action_default_equiv=0.065142
  loss_anchor_pv=2.319191
  loss_pv_weak=2.249437
  loss_mapg_cycle=0.437790
  loss_mapg_support_diversity=0.302989
  loss_mapg_routing=1.082569
  aqr_same_role_support_overlap_max=0.286274
  aqr_effective_anchor_count=23.3169
  posterior_recycle_rate=0.518672
  posterior_recycle_logit_mean=0.074726
  posterior_address_update_rate_mean=0.020734
  posterior_identity_switch_rate=0.773333
  posterior_identity_switch_rate_stable=0.0
  posterior_stable_slot_fraction=0.111111
  preclip_grad_norm=0.386549
  grad_clip_applied=false

A5 RMSNorm tail-5:
  loss_total=0.744731
  loss_action_default_equiv=0.066739
  aqr_same_role_support_overlap_max=0.442238
  aqr_effective_anchor_count=23.1342
  posterior_recycle_rate=0.521212
  posterior_recycle_logit_mean=0.084918
  posterior_address_update_rate_mean=0.020584
  posterior_identity_switch_rate=0.783111
  posterior_identity_switch_rate_stable=0.0
  posterior_stable_slot_fraction=0.111111
  preclip_grad_norm=0.804348
```

Decision against the pre-registered criteria:

```text
posterior_recycle_rate:
  LayerNorm tail5 0.5104 <= RMSNorm tail5 0.5212 + 0.03
  pass for LayerNorm.

posterior_address_update_rate_mean:
  LayerNorm tail5 0.02116 >= RMSNorm tail5 0.02058 - 0.003
  pass for LayerNorm.

same-role support overlap:
  RMSNorm final and tail5 are slightly lower, while LayerNorm tail3 is lower.
  Difference is small and not enough to override recycle/address preference.

action_default_equiv:
  both are essentially tied at tail5 ~= 0.0667.

gradient health:
  no persistent clipping in either run.
```

Conclusion:

```text
1. The normalized-recycle repair is accepted for this failure chain.
2. RMSNorm is a healthy ablation but does not show a decisive identity-stability
   win over LayerNorm.
3. Keep `recycle_residual_norm_mode=layernorm` as the maintained default.
4. Keep legacy local refinement archived/off.
5. Do not enable slot-JEPA, support prediction, binding consistency, denoising,
   ordinal loss, tracklet/proposal assumptions, or extra ownership rules yet.
```

#### Next Experiment Recommendation

The next run should be a longer guarded production-style run, not another
module-addition pass:

```text
profile:
  recycle_normalize_residual_summary=true
  recycle_residual_norm_mode=layernorm
  legacy_local_refinement_opt_in=false
  local_refinement_enabled=false
  local_refinement_weight=0.0
  action_prefix_stopgrad=true
  use_foundation_backbones=true
  visual_mode=encoder
  perception_finetune_mode=frozen
  semantic_mode=paligemma
  PaliGemma trainable with semantic_lr_scale=0.25
  lambda_slot_jepa=0.0
  lambda_support_pred=0.0
  lambda_binding_consistency=0.0
  lambda_aqr_denoising=0.0
```

Recommended scale:

```text
short confirmation:
  1200 steps if the goal is to catch medium-horizon overlap/recycle rebound.

acceptance checkpoint:
  2500-5000 steps if the goal is to compare against old 4-22 ablation action
  loss and produce the first video/anchor-debug evidence.
```

Hard gates for the next run:

```text
1. action_default_equiv should continue decreasing.
2. posterior_recycle_rate should not drift toward 0.95+ or collapse to a
   degenerate all-no-recycle state that hides assignment errors.
3. posterior_address_update_rate_mean should remain nonzero.
4. aqr_same_role_support_overlap_max tail should stay far from the old 0.99
   collapse zone; transient spikes are acceptable only if the tail recovers.
5. posterior_identity_switch_rate_stable should remain near zero.
6. posterior_stable_slot_fraction should be watched as the next bottleneck:
   current runs stabilize only about 11% of slots, so improving stable coverage
   is the next scientific target after this closure.
```

Interpretation:

```text
The previously blocking recycle/address failure is now closed well enough to
move on. The remaining issue is no longer "which emergency patch prevents
recycle collapse"; it is "does the guarded belief router maintain support and
stable-slot coverage under longer action cotrain." That must be answered by a
longer run plus CALVIN/video/anchor overlays, not by adding another auxiliary
loss in this stage.
```

### 2026-05-13 30k Guarded Long-Run Launch Plan

This section records the planned long-run deployment after the recycle
normalization closure. It is subordinate to
[`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md).

#### Question

Can the maintained PICF-AQR-OWM/MVTrack belief router train for a real 30000
step horizon with action cotrain active, frozen pretrained perception
backbones, trainable PaliGemma/semantic-action stack, stable support
competition, and no recycle/address collapse?

This run is not designed to prove ordinal/fourth-object grounding, tracklet
benefit, or proposal benefit. Current CALVIN training does not feed
tracklet/proposal tensors. Those branches remain no-op unless upstream data
provides them.

#### Why `burnin_steps=4`, `burnin_mode=state_only`, `unroll_steps=1`

The short diagnostics exposed a consistent distinction:

```text
direct unroll=2:
  faster credit assignment in principle, but historically coupled to unstable
  recycle/overlap episodes when the posterior lacked enough no-grad context.

burnin4/state_only + unroll1:
  four no-grad AQR posterior updates build the recurrent physical belief;
  the suffix transition receives gradients and action/alignment losses.
```

Mathematically this is the cleaner production compromise for 2x40GB:

```math
b_{t-4:t-1} = U_{AQR}^{no\ grad}(o, P(b,a)),
\qquad
b_t = U_{AQR}^{grad}(o_t, P(b_{t-1},a_{t-1}))
```

The burn-in is not an auxiliary trick. It approximates the inference-time
belief filter state before applying trainable suffix losses. It also preserves
the faster profile that previously ran around the 17-18 sec/step range when
pretrained perception backbones were frozen.

If "warmup" refers to recurrent warmup, use `burnin_steps=4`, not `1`. If it
refers to optimizer LR warmup, keep the already-tested `warmup_steps=100`
rather than introducing a new schedule in the acceptance run.

#### A7 Production Candidate

```text
machine: A7
ssh: ssh -p 28060 root@36.139.225.68
purpose: main production-candidate 30000-step guarded run
start: fresh from base checkpoint/config, not resume from diagnostic checkpoint
num_train_steps: 30000
save_interval: 2500
keep_last_checkpoints: 3
progress: enabled
training_strategy: fsdp_full_shard
optimizer_sharding: none
perception_finetune_mode: frozen
frozen pretraining modules: Sonata, V-JEPA, AnyTouch
trainable: PICF, PI0.5 action/semantic stack, PaliGemma with semantic_lr_scale=0.25
unroll_steps: 1
burnin_steps: 4
burnin_mode: state_only
action_horizon: 16
action_normalization: quantile
picf_action_prefix_stopgrad: true
recycle_normalize_residual_summary: true
recycle_residual_norm_mode: layernorm
legacy_local_refinement_opt_in: false
local_refinement_enabled: false
local_refinement_topk: 0
local_refinement_weight: 0.0
local_refinement_binding_weight: 0.0
evidence_cache_read_weight: 0.05
lambda_slot_jepa: 0.0
lambda_support_pred: 0.0
lambda_binding_consistency: 0.0
lambda_aqr_denoising: 0.0
```

#### A5 Conservative Long-Test Control

```text
machine: A5
ssh: ssh -p 29776 root@36.139.225.68
purpose: conservative control for semantic cotrain pressure
same as A7 except:
  semantic_lr_scale: 0.1
```

This is not the preferred production profile. It answers whether reducing
PaliGemma/semantic update pressure materially improves recycle/support health.
If A7 and A5 are both healthy, prefer A7 because previous action-ready tests
showed PaliGemma cotrain is useful. If A7 degrades and A5 remains healthy,
semantic LR pressure becomes the next controlled variable.

#### Hard Gates

```text
1. Startup log must show visual_mode=encoder and use_foundation_backbones=True.
2. Startup log must show Sonata/V-JEPA/AnyTouch frozen and PaliGemma trainable.
3. loss_action_default_equiv should be used for comparison to old 4-22 style
   action curves, not raw active-horizon action loss.
4. posterior_recycle_rate must not saturate near 0.95+ or collapse to a
   degenerate always-off state hiding assignment errors.
5. posterior_address_update_rate_mean must remain nonzero.
6. aqr_same_role_support_overlap_max should stay far from the old 0.99 collapse
   zone in tail windows.
7. posterior_identity_switch_rate_stable should remain near zero.
8. stable_slot_fraction is expected to remain a bottleneck; improvement is a
   positive sign, but low stable coverage alone is not an immediate stop unless
   recycle/overlap/action also degrade.
9. First real behavior acceptance starts at the 2500-step checkpoint:
   metrics, CALVIN/video, anchor overlays, and support health must be checked.
```

#### Tail Commands After Launch

```bash
# A7
ssh -p 28060 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_30k_recyclenorm_layernorm_burnin4_sem025_20260513_6c58f46.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_30k_recyclenorm_layernorm_burnin4_sem025_20260513_6c58f46/metrics.jsonl

# A5
ssh -p 29776 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_30k_recyclenorm_layernorm_burnin4_sem010_20260513_6c58f46.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_30k_recyclenorm_layernorm_burnin4_sem010_20260513_6c58f46/metrics.jsonl
```

#### Expected Timing

At the observed frozen-perception/PaliGemma-cotrain speed of roughly
17-18 sec/step:

```text
first metrics row: usually after startup and the first log interval;
first checkpoint at 2500 steps: about 12-13 hours;
30000 steps: about 6 days.
```

The first 2500-step checkpoint is the next serious acceptance point. Earlier
tail rows are useful for failure detection, not for final behavior claims.

## 2026-05-13 A5 Overlay Warmup-To-Cotrain Diagnostic

This section supersedes the older purely scalar anchor diagnostics for the next
A5 run. The unresolved question is not just whether
`aqr_same_role_support_overlap_max` is high; it is whether that high visual
support overlap corresponds to actual physical/posterior anchor collapse in the
main camera.

### Code-Level Diagnostic

The trainer now exposes:

```bash
--anchor-overlay-interval 100
--anchor-overlay-max-anchors 64
```

Every enabled interval writes:

```text
anchor_overlays/step_XXXXXX.png
anchor_overlays/step_XXXXXX.json
```

The implementation reuses the real training forward output. It does not do an
extra no-grad forward because that would push the same frame through the
V-JEPA/tactile buffers a second time and could perturb the recurrent evidence
state. It snapshots detached CPU copies of graph and posterior anchor state from
the actual step, then projects their 3D positions through the static camera
model for visualization.

Mathematical purpose:

```math
support\ overlap(i,j) \approx \langle p_i^{visual}, p_j^{visual}\rangle
```

is a support-space statistic. It does not by itself prove:

```math
\mu_i \approx \mu_j
```

where `mu/x` is the posterior/anchor 3D location. The overlay tests whether the
support-space collapse is also a physical projected-location collapse.

### A5 Staged Run Design

The next A5 run is a two-stage diagnostic:

```text
Stage 1: quick anchor warmup
  action losses off
  predictive/high-risk aux losses off
  foundation perception frozen
  AQR/MVTrack anchor, support, binding, recycle/address path trainable
  anchor overlays every 100 steps

Stage 2: cotrain continuation
  resume from the warmup checkpoint
  tested burnin4/state_only recurrent warmup
  PaliGemma cotrain enabled
  prefix-stopgrad retained
  predictive/high-risk aux losses still zero
  anchor overlays every 100 steps
```

This is not a new architecture claim. It tests the old hypothesis that a brief
anchor separation phase can move the system away from the degenerate all-support
basin before scalar action pressure is reintroduced.

### Acceptance Criteria

```text
1. Overlay PNG/JSON exists at step 100, 200, 300 and cotrain step 400+.
2. posterior_recycle_rate stays below the old saturation zone.
3. posterior_address_update_rate_mean remains nonzero.
4. aqr_same_role_support_overlap_max is interpreted with overlay evidence:
   high overlap is worse if the projected posterior anchors also co-locate.
5. loss_action_default_equiv is used for old 4-22/ablation comparability.
6. If action improves but overlays show physical co-location, the recipe is not
   accepted as healthy even if scalar loss falls.
```

Tail commands and exact run ids must be updated after launch.

### 2026-05-13 A7 Unroll=2 Overlay Counterfactual

The earlier A7 30k layernorm run was archived and stopped because its scalar
action diagnostics were improving while anchor-health metrics remained in the
same-risk region:

```text
step 500..700:
  loss_action_default_equiv ~= 0.062..0.069
  aqr_same_role_support_overlap_max ~= 0.98..0.998
  posterior_recycle_rate ~= 0.53..0.55
```

Continuing that run would mostly repeat the old failure mode: action loss can
fall while same-role support reuse and recycle remain high. It also lacked the
new per-100-step training anchor overlays, so it could not answer whether the
support overlap corresponds to physical co-location in the main static view.

The replacement A7 run is the strict unroll counterfactual to the active A5
overlay run:

```text
machine: A7 / ssh -p 28060 root@36.139.225.68
commit: b9ad838
clean worktree: /root/openpi_a7_overlay_unroll2_b9ad838
tmux: picf_a7_overlay_unroll2_warmcotrain

warmup run:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warm300_20260513_b9ad838

cotrain run:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_cotrain_from300_to900_20260513_b9ad838

shared log:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warmcotrain_20260513_b9ad838.train_tmux.log
```

The controlled difference from A5 is:

```text
A5:
  burnin_steps=4
  burnin_mode=state_only
  unroll_steps=1
  effective_window_steps=5

A7:
  burnin_steps=4
  burnin_mode=state_only
  unroll_steps=2
  effective_window_steps=6
```

Important speed caveat:

```text
direct unroll2:
  burnin_steps=0
  unroll_steps=2
  effective_window_steps=2

burnin4/unroll1:
  burnin_steps=4
  unroll_steps=1
  effective_window_steps=5
```

The direct `burnin0/unroll2` profile can be faster in wall-clock time because it
processes fewer recurrent transitions per optimizer step. It is not the same
test as the current A7 counterfactual. The current A7 run intentionally keeps
`burnin_steps=4` fixed and only changes the trainable suffix length from 1 to
2, so it isolates recurrent suffix credit assignment while preserving the
state distribution that was healthier in previous tests. If the question is
"can direct unroll2 be the faster production profile?", run a separate
`burnin0/unroll2` overlay ablation and compare both metrics and physical anchor
overlays.

All other important guards match the A5 overlay run: frozen Sonata/V-JEPA/
AnyTouch, foundation visual mode, local refinement disabled, high-risk
predictive losses zero, prefix-stopgrad retained, and anchor overlays every
100 steps.

The mathematical question is whether adding one more trainable suffix
transition improves recurrent credit assignment:

```math
L_{unroll1}=L(b_t)
```

versus:

```math
L_{unroll2}=\frac{1}{2}\left[L(b_t)+L(b_{t+1})\right]
```

where both are preceded by the same no-grad state-only burn-in:

```math
b_{t-4:t-1}=U_{AQR}(o_{\le t-1}, b_{t-5})\quad\text{without gradient}
```

Possible outcomes:

```text
1. A7 unroll2 improves overlap/recycle and overlays show spatial separation:
   recurrent suffix credit was helpful; consider unroll2 for the next longer
   candidate despite slower step time.

2. A7 unroll2 has similar or worse overlap/recycle than A5:
   the bottleneck is not suffix length; keep burnin4/unroll1 for speed and
   diagnose support evidence / physical anchor geometry instead.

3. A7 action proxy improves faster but overlays co-locate:
   unroll2 helps scalar action fitting but not healthy binding; do not promote
   it as the production recipe.
```

Tail commands:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warmcotrain_20260513_b9ad838.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warm300_20260513_b9ad838/metrics.jsonl
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_cotrain_from300_to900_20260513_b9ad838/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warm300_20260513_b9ad838/anchor_overlays
```

### 2026-05-13 23:25 Local Audit And Live A5/A7 Status

Local verification was rerun after the A5/A7 overlay deployments:

```text
python -m py_compile:
  scripts/picf_core_train.py
  scripts/verify_picf_owm_contract.py
  scripts/picf_owm_strict_diagnose.py
  scripts/picf_owm_dataflow_trace.py
  scripts/picf_owm_mvtrack_deep_audit.py
  scripts/picf_owm_evidence_bundle.py

python scripts/verify_picf_owm_contract.py:
  31/31 PASS

python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS

python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS

python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS

uv run --no-sync pytest -q \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py:
  4 passed
```

`picf_owm_strict_diagnose.py` still reports WARN entries for missing runtime
metrics/eval files when no runtime paths are passed. This is expected for a
local code audit and is not a code-contract failure.

A5 live status:

```text
tmux:
  picf_a5_overlay_warmcotrain

warmup run:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_overlay_warm300_20260513_b9ad838

cotrain run:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_overlay_cotrain_from300_to900_20260513_b9ad838

step 50:
  loss_total = 0.0875
  loss_action_default_equiv = 0.1549
  loss_action_active7 = 0.4967
  loss_anchor_pv = 2.9131
  loss_mapg_support_diversity = 0.8107
  aqr_same_role_support_overlap_max = 0.9508
  aqr_effective_anchor_count = 23.22
  posterior_recycle_rate = 0.5527
  posterior_address_update_rate_mean = 0.0203
  steps_per_sec = 0.0847

step 100:
  loss_total = 0.0875
  loss_action_default_equiv = 0.1757
  loss_action_active7 = 0.5373
  loss_anchor_pv = 3.5987
  loss_mapg_support_diversity = 0.9444
  aqr_same_role_support_overlap_max = 0.9918
  aqr_effective_anchor_count = 22.83
  posterior_recycle_rate = 0.0992
  posterior_address_update_rate_mean = 0.0384
  steps_per_sec = 0.0834

anchor overlay:
  anchor_overlays/step_000100.png exists
```

Interpretation:

```text
The A5 warmup has generated the requested per-100-step physical anchor overlay
and the recurrent address/recycle machinery is active. The recycle rate improved
sharply by step 100, but same-role support overlap remains very high. Therefore
this run cannot yet be treated as evidence that the anchor supports are
physically separated. The step-100 overlay and the step-200/300 metrics are the
next acceptance gates.
```

A7 live status:

```text
tmux:
  picf_a7_overlay_unroll2_warmcotrain

warmup run:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warm300_20260513_b9ad838

cotrain run:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_cotrain_from300_to900_20260513_b9ad838

current status:
  no metrics.jsonl yet at the 23:25 check
  GPUs active with the expected memory footprint
```

Interpretation:

```text
A7 is still before its first scalar logging point. It remains the controlled
counterfactual for burnin4/state_only plus unroll2. It must not be used yet to
conclude whether recurrent suffix length fixes or fails the overlap problem.
```

### 2026-05-13 23:45 Task-Pressure Warmup Restart

The previous A5/A7 overlay warmups were intentionally conservative, but they
used `lambda_action_{pos,rot,gripper}=0.0` while also retaining
`picf_action_prefix_stopgrad`. That profile is now archived as an important
negative control rather than a production warmup:

```text
Pure action-off warmup result:
  recycle can recover, but same-role support overlap stays high.

A5 examples:
  step 50:  same_role_overlap=0.9508, recycle=0.5527
  step 100: same_role_overlap=0.9918, recycle=0.0992
  step 150: same_role_overlap=0.9987, recycle=0.0094
  step 200: same_role_overlap=0.9996, recycle=0.0097
```

This is not just a low-step issue. With action loss removed, same-role anchors
can minimize the remaining weak alignment/diversity objectives by repeatedly
reading the same high-confidence evidence. The physical task provides the
missing symmetry-breaking signal: two supports that look similarly plausible in
PG/V-JEPA space can still differ in action relevance.

The replacement warmup is therefore not full cotrain and not a patchy heuristic.
It is a task-pressured anchor warmup:

```math
L_{\text{taskwarm}}
=
0.25 L_{\text{anchor-pv}}
+0.05 L_{\text{cycle}}
+0.05 L_{\text{support-div}}
+0.02 L_{\text{geom-div}}
+0.25 L_{\text{action,pos}}
+0.25 L_{\text{action,rot}}
+0.25 L_{\text{action,grip}}
```

The important contract is:

```text
1. trainable scope remains anchor_only;
2. Sonata / V-JEPA / AnyTouch remain frozen;
3. high-risk predictive losses remain 0;
4. local refinement remains disabled / legacy opt-in only;
5. recycle residual summary remains normalized with LayerNorm;
6. action prefix stopgrad is disabled only for this task-pressure warmup.
```

Mathematically, the change is the smallest coherent way to test the missing
causal factor. The pure anchor objective sees only evidence consistency:

```math
q_j \rightarrow \operatorname{Read}(M_t)
```

but same-role query permutation is still nearly symmetric. The task-pressure
objective adds a low-weight downstream gradient:

```math
\nabla_{\theta_{anchor}}
L_{action}
=
\frac{\partial L_{action}}{\partial a_t}
\frac{\partial a_t}{\partial b_t}
\frac{\partial b_t}{\partial q_j}
\frac{\partial q_j}{\partial \theta_{anchor}}
```

This is deliberately weaker than full action cotrain. It tests whether the
action-relevant object identity can break the support symmetry without allowing
the action head to dominate the belief router.

#### Active A5 Task-Pressure Run

```text
machine:
  A5 / ssh -p 29776 root@36.139.225.68

worktree:
  /root/openpi_recyclenorm_4ec25ae

tmux:
  picf_a5_taskwarm_a025_unroll1

run:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838

tmux log:
  /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838.train_tmux.log

controlled settings:
  unroll_steps=1
  burnin_steps=4
  burnin_mode=state_only
  picf_trainable_scope=anchor_only
  lambda_action_pos/rot/gripper=0.25
  no_picf_action_prefix_stopgrad
```

#### Active A7 Task-Pressure Counterfactual

```text
machine:
  A7 / ssh -p 28060 root@36.139.225.68

worktree:
  /root/openpi_a7_overlay_unroll2_b9ad838

tmux:
  picf_a7_taskwarm_a025_unroll2

run:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838

tmux log:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838.train_tmux.log

controlled difference from A5:
  unroll_steps=2
```

#### Acceptance Criteria

At step 100 the task-pressure warmup must already separate itself from the
pure action-off negative control:

```text
primary:
  aqr_same_role_support_overlap_max should be clearly below the 0.95-0.99
  collapse band seen in pure action-off warmup.

secondary:
  posterior_recycle_rate should not saturate at 0 or 1;
  posterior_address_update_rate_mean should remain nonzero;
  aqr_effective_anchor_count should stay near the previous healthy range;
  loss_action_default_equiv should not spike while overlap improves.

visual:
  anchor_overlays/step_000100.png must show physical separation, not only a
  lower scalar overlap.
```

If both A5 and A7 still return to `same_role_overlap > 0.95`, the conclusion is
not "increase warmup length." It means task-pressure alone is insufficient and
the next root-cause work should move to object ownership in the assignment
mechanism: support competition, actual tracklet/proposal input plumbing, or a
cleaner per-object weak target. If A7 improves materially over A5, unroll2 is
worth carrying into the next short cotrain despite its slower step time. If A5
matches A7, keep unroll1 for speed.

Tail commands:

```bash
# A5
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838/anchor_overlays

# A7
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838/anchor_overlays
```

#### Remote Environment Note

The fresh remote worktrees initially failed before training because `uv sync`
attempted to build `av==14.4.0` through the transitive `lerobot` dependency.
That package requires system FFmpeg development libraries and is not on the
PICF CALVIN training path used by these runs. The remote environment was
therefore restored with:

```bash
uv sync --project . --frozen --no-install-package av
PYTHONPATH=src:. uv run --no-sync --project . python -m py_compile \
  scripts/picf_core_train.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py
```

This keeps the locked `uv` environment and installs the training dependencies,
including `transformers==4.53.2` and `numpydantic==1.6.9`, without pulling an
unneeded video I/O build dependency into the training container. The training
scripts still run through `uv run --no-sync --project . torchrun ...`.

The remote Sonata sparse-convolution runtime is provided by the pre-existing
server environment:

```bash
PYTHONPATH=src:/root/openpi/.venv/lib/python3.11/site-packages \
  uv run --no-sync --project . python - <<'PY'
from openpi.picf.sonata.wrapper import _load_sonata_runtime
print(_load_sonata_runtime()[0], _load_sonata_runtime()[1])
PY
```

#### Live Read: Step 50-75

The first reads do not yet validate task-pressure warmup. They are useful
because they separate two failure modes:

```text
A5 / unroll=1:
  step 25:
    same_role_overlap=0.9484
    recycle_rate=0.5233
    address_update_rate_mean=0.0217
    action_default_equiv=0.1410

  step 50:
    same_role_overlap=0.9534
    recycle_rate=0.5019
    address_update_rate_mean=0.0227
    action_default_equiv=0.1524

  step 75:
    same_role_overlap=0.9151
    recycle_rate=0.1511
    address_update_rate_mean=0.0379
    action_default_equiv=0.1386

A7 / unroll=2:
  step 25:
    same_role_overlap=0.9532
    recycle_rate=0.5349
    address_update_rate_mean=0.0211
    action_default_equiv=0.1499

  step 50:
    same_role_overlap=0.9769
    recycle_rate=0.5370
    address_update_rate_mean=0.0210
    action_default_equiv=0.1416
```

Interpretation:

```text
1. Task pressure is not causing the old immediate recycle saturation. A5
   recycle falls to 0.151 by step 75, so recycle normalization remains active.

2. Task pressure has not yet solved same-role ownership. A5 is better than the
   pure action-off collapse band but still above the desired <0.75 threshold;
   A7 is worse at step 50.

3. Unroll=2 is not currently the explanation for early support separation. It
   is slower and has worse overlap in the matched counterfactual.

4. If step 100 remains above 0.90 overlap, the root problem is likely object
   ownership/assignment evidence, not warmup length. The next design should
   target the binding assignment itself rather than only changing schedules.
```

Decision rule:

```text
Do not restart before step 100 unless the run crashes. If step 100 still shows
same_role_overlap > 0.90 on both A5 and A7, archive this as a negative result
for schedule-only task-pressure warmup and move to an assignment-level fix.
```

#### Final Read: Schedule-Only Task Pressure Fails Ownership

The step-100/125 reads crossed the failure threshold and the A5/A7 tmux runs
were stopped:

```text
A5 / unroll=1:
  step 100:
    same_role_overlap=0.9640
    recycle_rate=0.0779
    action_default_equiv=0.1338

  step 125:
    same_role_overlap=0.9960
    recycle_rate=0.5223
    action_default_equiv=0.1468

A7 / unroll=2:
  step 100:
    same_role_overlap=0.9374
    recycle_rate=0.6398
    action_default_equiv=0.1356
```

Conclusion:

```text
1. Low task pressure is not sufficient to create stable same-role ownership.
2. Unroll=2 is not the root fix; it is slower and still over the 0.90 overlap
   failure threshold.
3. Recycle normalization is no longer the only bottleneck. A5 can reach low
   recycle while support ownership still collapses.
4. The next fix must enter the AQR assignment model itself.
```

Mathematical root cause:

```math
W_{j,i}=\operatorname{softmax}_i(\ell_{j,i})
```

If two same-role query rows are already identical, then Sinkhorn or column
balancing cannot create identity:

```math
\ell_{j,:}=\ell_{k,:}
\Rightarrow
\operatorname{Sinkhorn}(W)_{j,:}
=
\operatorname{Sinkhorn}(W)_{k,:}
```

The previous fixes were valid but incomplete:

```text
support-diversity loss:
  penalizes overlap after the graph exists, but cannot reliably break a fully
  symmetric assignment basin.

task-pressure warmup:
  gives action relevance, but the gradient is downstream and weak relative to
  the identical support rows.

binding-signature subspace:
  reads same-object information once signatures differ, but identical support
  rows produce nearly identical signatures.
```

The coherent root repair is an assignment-level ownership prior. It is not a
new loss and not a local-refinement residual. It adds a low-amplitude role-local
coverage prior directly to AQR support logits:

```math
\ell'_{j,i}
=
\ell_{j,i}
+
\lambda_{own}
\left(
\log \pi^{own}_{j,i}
-
\frac{1}{N}\sum_i\log \pi^{own}_{j,i}
\right)
```

where `pi_own` is built by farthest-point / mode coverage within each role over
the typed visual or temporal support coordinates. This gives identical same-role
queries distinct initial ownership while preserving current evidence dominance:

```text
lambda_own_visual = 0.35
lambda_own_temporal = 0.20
uniform_mix = 0.05
```

The design follows the object-binding probe evidence: pretrained ViTs can encode
same-object structure, but a downstream policy should explicitly read and route
that subspace instead of assuming it will remain stable under action gradients.
The ownership prior provides the missing object-file / slot ownership seed; the
existing binding-signature term then has non-identical supports to stabilize.

Implementation:

```text
src/openpi/picf/core/config.py:
  aqr_ownership_prior_enabled
  aqr_ownership_prior_weight
  aqr_ownership_temporal_prior_weight
  aqr_ownership_prior_uniform_mix

src/openpi/picf/core/pipeline.py:
  _aqr_ownership_priors_from_coords
  _aqr_visual_ownership_bias
  _aqr_temporal_ownership_bias

scripts/picf_core_train.py:
  CLI flags and startup log entries for ownership prior settings.

tests:
  test_aqr_ownership_prior_breaks_same_role_visual_symmetry
  test_aqr_ownership_prior_breaks_temporal_multiview_symmetry
```

This is the next A5/A7 experiment family. Acceptance at 100-300 steps:

```text
same_role_overlap should stay below 0.75, not briefly dip and recover to >0.9;
recycle should not saturate at 0 or 1;
action_default_equiv should remain comparable to the failed task-pressure runs;
anchor overlays should show distinct physical ownership rather than only scalar
improvement.
```

The smoke test passed on both A5 and A7, loading `spconv` and `torch_scatter`
from `/root/openpi/.venv/lib/python3.11/site-packages` while keeping the active
worktree `src` first in `PYTHONPATH`. The task-pressure tmux launches therefore
set the inherited `PYTHONPATH` to that site-packages directory, and the run
script expands it to `src:/root/openpi/.venv/lib/python3.11/site-packages`.

#### First Scalar Check

Both task-pressure warmups reached the first metrics point:

```text
A5 step 25:
  same_role_support_overlap_max = 0.9484
  posterior_recycle_rate = 0.5233
  posterior_address_update_rate_mean = 0.0217
  loss_action_default_equiv = 0.1410
  loss_total = 0.1051
  steps_per_sec = 0.0876

A7 step 25:
  same_role_support_overlap_max = 0.9532
  posterior_recycle_rate = 0.5349
  posterior_address_update_rate_mean = 0.0211
  loss_action_default_equiv = 0.1499
  loss_total = 0.1062
  steps_per_sec = 0.0706
```

This is not yet a pass. It is a valid early signal: recycle is no longer
saturating and address updates are nonzero, but same-role overlap is still near
the previous collapse boundary. The decisive read remains step 100 plus the
anchor overlay image. If step 100 remains in the 0.95-0.99 band, the failure is
not "insufficient warmup length"; it indicates that weak task pressure alone
does not create object ownership and the next mechanism must address assignment
competition or actual object-correspondence evidence.

At step 50 the first trend became less favorable:

```text
A5 step 50:
  same_role_support_overlap_max = 0.9534
  posterior_recycle_rate = 0.5019
  loss_action_default_equiv = 0.1524
  steps_per_sec = 0.0877

A7 step 50:
  same_role_support_overlap_max = 0.9769
  posterior_recycle_rate = 0.5370
  loss_action_default_equiv = 0.1416
  steps_per_sec = 0.0713
```

The important interpretation is negative but useful: unroll2 did not improve
early same-role support separation; it made overlap worse at this point while
being slower. Recycle remains non-saturated, so the previous recycle-collapse
failure is not the active bottleneck. If step 100 confirms this trend, the next
root-cause item is not more recurrent context but a stronger object-ownership
assignment signal.

### 2026-05-14 Remote Ownership-Prior Launch

After the schedule-only task-pressure warmup failed on both A5 and A7, commit
`33dd330` deployed the assignment-level ownership prior as the maintained
root-cause repair. The local validation gate before remote launch was:

```text
python -m py_compile src/openpi/picf/core/pipeline.py
python scripts/verify_picf_owm_contract.py                  # 32/32 PASS
python scripts/picf_owm_strict_diagnose.py --fail-on-fail   # PASS, no metrics input
python scripts/picf_owm_dataflow_trace.py --fail-on-fail    # PASS
python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail# PASS
uv run --no-sync pytest -q src/openpi/picf/core/pipeline_test.py -k ownership_prior
uv run --no-sync pytest -q src/openpi/picf/core/training_test.py -k support_diversity
```

Remote sync used the China GitHub mirror and fast-forwarded both runtime
worktrees to `33dd330`.

Checkpoint cleanup was intentionally conservative: only May-2026 checkpoint
weight files were deleted, while run directories, metrics, args, overlays, and
logs were preserved.

```text
A5 cleanup log:
  /mnt/checkpoints/picf_core/cleanup_logs/removed_may_ckpt_weights_20260514_004224.txt
  removed weight files: 52

A7 cleanup log:
  /mnt/checkpoints/picf_core/cleanup_logs/removed_may_ckpt_weights_20260514_004225.txt
  removed weight files: 36
```

Launched diagnostics:

```text
A5 tmux:
  picf_a5_ownership_u1
  exp: picf_a5_ownership_prior_taskwarm_u1_300_20260514_33dd330
  unroll_steps=1, burnin_steps=4, anchor_only, action pressure enabled,
  local_refinement disabled, OWM predictive losses disabled,
  ownership prior enabled with visual=0.35 temporal=0.20 uniform_mix=0.05.

A7 tmux:
  picf_a7_ownership_u2
  exp: picf_a7_ownership_prior_taskwarm_u2_300_20260514_33dd330
  same as A5 except unroll_steps=2.
```

Tail commands:

```bash
# A5
ssh -p 29776 root@px-cloud2.matpool.com
tail -f /mnt/picf_run_logs/picf_a5_ownership_prior_taskwarm_u1_300_20260514_33dd330.log

# A7
ssh -p 28060 root@px-cloud2.matpool.com
tail -f /mnt/picf_run_logs/picf_a7_ownership_prior_taskwarm_u2_300_20260514_33dd330.log
```

Acceptance gate:

```text
step 25: early signal only, not decisive.
step 100: first decisive scalar + anchor overlay gate.
step 300: endpoint gate.

Pass if:
  aqr_same_role_support_overlap_max does not stay in/rebound to 0.95-0.99,
  posterior_recycle_rate does not saturate,
  action_default_equiv remains comparable to failed task-pressure runs,
  overlays show distinct physical ownership rather than scalar-only improvement.

Fail interpretation if both A5 and A7 remain >0.95 overlap:
  ownership priors are too weak or current typed evidence cannot support role-local
  ownership; the next repair must feed real object-correspondence evidence
  (tracklet/proposal/pseudo target), not another schedule-only change.
```

#### First Ownership-Prior Scalar Read

The first metrics point is already qualitatively different from the failed
schedule-only task-pressure runs:

```text
Failed schedule-only reference:
  A5 step25 same_role_overlap=0.9484
  A7 step25 same_role_overlap=0.9532

Ownership-prior run:
  A5 step25 same_role_overlap=0.2298
              recycle=0.4848
              action_default_equiv=0.1397
              temporal_view_mass_static=0.5721
              temporal_view_mass_gripper=0.4279

  A5 step50 same_role_overlap=0.2280
              recycle=0.4588
              action_default_equiv=0.1342

  A7 step25 same_role_overlap=0.2176
              recycle=0.4997
              action_default_equiv=0.1530
              temporal_view_mass_static=0.5930
              temporal_view_mass_gripper=0.4070
```

Interpretation:

```text
1. The ownership prior is not merely changing a scalar loss. It changes the
   support assignment geometry before Sinkhorn, exactly where the failure was
   mathematically located.
2. A5 held the low-overlap regime through step50, so this is not just a one-step
   initialization artifact.
3. A7 also starts in the low-overlap regime despite unroll=2, so the previous
   collapse was not caused by insufficient recurrent window alone.
4. The decisive gates remain step100 and step300 plus overlay inspection.
```

#### Ownership-Prior Endpoint: Early Fix, Late Collapse

The ownership prior fixed the initial same-role symmetry failure, but it did not
survive training pressure. The runs were stopped/archived because the failure
was already visible before the planned endpoint.

```text
A5 / unroll=1 endpoint:
  exp:
    /mnt/checkpoints/picf_core/picf_core/picf_a5_ownership_prior_taskwarm_u1_300_20260514_33dd330
  log:
    /mnt/picf_run_logs/picf_a5_ownership_prior_taskwarm_u1_300_20260514_33dd330.log
  overlays:
    anchor_overlays/step_000100.png
    anchor_overlays/step_000200.png
    anchor_overlays/step_000300.png

  same_role_overlap:
    25: 0.2298
    50: 0.2280
    75: 0.2530
    100: 0.3009
    125: 0.7871
    150: 0.9481
    175: 0.9890
    200: 0.9949
    225: 0.9992
    250: 0.9994
    275: 0.9998
    300: 0.9998

  posterior_recycle_rate:
    25: 0.4848
    100: 0.6624
    150: 0.7177
    225: 0.9169
    300: 0.9853

  slot_address_update_rate_mean:
    25: 0.0234
    100: 0.0142
    150: 0.0099
    225: 0.0033
    300: 0.0006
```

```text
A7 / unroll=2 endpoint:
  exp:
    /mnt/checkpoints/picf_core/picf_core/picf_a7_ownership_prior_taskwarm_u2_300_20260514_33dd330
  log:
    /mnt/picf_run_logs/picf_a7_ownership_prior_taskwarm_u2_300_20260514_33dd330.log
  overlays:
    anchor_overlays/step_000100.png
    anchor_overlays/step_000200.png

  same_role_overlap:
    25: 0.2176
    50: 0.2615
    75: 0.3265
    100: 0.3102
    125: 0.5065
    150: 0.7597
    175: 0.7301
    200: 0.8971
    225: 0.9827
    250: 0.9843

  posterior_recycle_rate:
    25: 0.4997
    75: 0.1261
    100: 0.4643
    150: 0.2120
    175: 0.0208
    200: 0.0156
    250: 0.0124

  slot_address_update_rate_mean:
    25: 0.0234
    100: 0.0299
    150: 0.0464
    200: 0.0396
    250: 0.0403
```

Critical interpretation:

```text
1. The ownership prior is a real improvement, not a false positive:
   both A5 and A7 start around 0.22 overlap instead of 0.95+.

2. The repair is not sufficient:
   after step100/125 the shared support attractor reappears and same-role
   overlap returns to the 0.98-1.00 failure regime.

3. Action loss is not the primary direct cause:
   loss_action_default_equiv stays in the same band while overlap worsens.
   The strongest common signals correlated with collapse are raw graph/PV
   structure terms, especially loss_anchor_pv, loss_mapg_cycle,
   loss_mapg_support_diversity, and loss_mapg_routing.

4. Recycle is not the sole root cause:
   A5 collapses with recycle rising to 0.98, while A7 collapses after recycle
   falls near zero. Recycle is a coupled symptom, not a universal explanation.

5. Unroll=2 helps delay the failure but does not solve it:
   A7 reaches the same high-overlap regime later than A5, so recurrence length
   matters, but the assignment geometry remains unstable.

6. The scalar effective_anchor_count is misleading:
   it stays near 22-23 even when same-role supports are almost identical.
   It measures nonzero mass, not distinct object ownership.
```

The consistent mathematical root cause is now sharper than before:

```text
The AQR rows can be made distinct initially, but the trainable readers and
task/PV/routing objectives still have a shared high-salience support attractor.
The static coordinate ownership prior is too weak to remain an ownership
constraint after the model starts optimizing toward common task-relevant
regions. The support-diversity term observes the overlap after it happens, but
does not impose an assignment-level capacity or dustbin/no-object structure.

In addition, the alignment objective is budget/cap scaled in these diagnostics:
  loss_alignment = 0.0125
  loss_total_minus_action = 0.0875
so raw graph losses can worsen substantially without dominating total loss.
This prevents total loss from being a reliable health metric for object
ownership.
```

This result changes the next repair target:

```text
Do not proceed to 30k from this ownership-prior run.
Do not treat a stronger static prior, longer warmup, or unroll-only change as
the root fix.

The next coherent repair must make same-role object ownership adaptive and
capacity-aware:
  1. distinguish active object slots from inactive/background/dustbin slots;
  2. apply same-role diversity only to active high-confidence slots;
  3. avoid forcing all physical slots onto a scene with fewer useful objects;
  4. move ownership pressure closer to assignment/logit construction instead
     of relying only on a downstream capped auxiliary loss;
  5. continue using posterior authority and avoid hard-locking identity by
     static address.
```

## 2026-05-14 Capacity-Aware Active/Dustbin Repair and 10-Hour Matrix

Status: code deployed locally, verification passed, remote A5/A7 rollout
pending after this commit is synchronized.

### Root-Cause Restatement

The ownership-prior runs proved two facts simultaneously:

```text
1. Same-role rows can be separated early:
   A5/A7 start around 0.22 same-role support overlap instead of 0.95+.

2. The separation is not durable:
   fixed same-role physical slots still converge toward a shared high-salience
   support attractor after task/PV/routing pressure starts shaping the reader.
```

The failure is not fully explained by action loss, recycle saturation, or
unroll length. A5 collapses with high recycle, while A7 collapses after recycle
falls near zero. Action-equivalent loss stays in the same band while overlap
worsens. Unroll=2 delays but does not remove collapse.

The sharper mathematical failure is a capacity mismatch. Let role `r` have
`K_r` physical queries and a scene have only `M_r` useful supports for that role,
with `K_r > M_r`. If every query must bind some support, the optimum can place
multiple same-role rows on the same high-value support even when rows were
initially distinct:

```math
\max_{P \in \Delta} \sum_{j=1}^{K_r} \sum_i P_{j,i} Q_i
\quad \text{with} \quad K_r > M_r
```

Without a no-object/inactive state, redundant rows are penalized for staying
uncommitted and are attracted to the best support. A downstream diversity loss
observes the duplicate after it forms; it does not provide a valid assignment
state for extra capacity.

### Maintained Repair

The new repair adds an active/inactive split before assignment:

```text
active anchors:
  high-confidence role-local support owners, capped per role and filtered by
  support overlap.

inactive/dustbin anchors:
  redundant same-role candidates kept as recurrent/query carriers but excluded
  from observation/task assignment when active candidates exist.
```

Defaults:

```text
aqr_active_slot_filter_enabled=true
aqr_active_slot_min_per_role=1
aqr_active_slot_max_per_role=4
aqr_active_slot_min_confidence=0.05
aqr_active_slot_overlap_threshold=0.75
```

This is intentionally not a new loss. It changes the feasible assignment set so
the belief router no longer has to hallucinate more objects than the evidence
supports. It is the same design class as no-object/dustbin assignment in
set-prediction detectors, adapted to recurrent PICF slots.

Connection to recent papers:

```text
Object Binding in pretrained ViTs, NeurIPS 2025 / arXiv:2510.24709:
  motivates pairwise same-object subspaces and IsSameObject probing. We already
  added binding_signature_proj and support-weighted binding signatures. The
  current repair is the complementary assignment-capacity step: if the same-
  object signal exists but multiple same-role rows reuse the same candidate,
  the issue is assignment/capacity rather than representation absence.

MetaSlot, NeurIPS 2025 / arXiv:2505.20772:
  highlights the fixed-slot-count limitation in object-centric learning. PICF
  keeps a fixed parameter budget for engineering stability but adds an effective
  active-slot count per role.

When Slots Compete, arXiv:2603.11246:
  studies slot competition/merging from overlap statistics. PICF does not merge
  query parameters during training; instead it uses overlap statistics to make
  redundant anchors inactive/dustbin candidates, preserving recurrent identity
  state while avoiding duplicate object pressure.
```

### Code Verification

Local verification completed before remote deployment:

```text
python -m py_compile:
  contracts/config/pipeline/training/train/diagnosis/evidence scripts PASS.

python scripts/verify_picf_owm_contract.py:
  PASS, including pipeline_active_slot_filter_adds_capacity_aware_dustbin_path.

python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS.

python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS.

python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS.

uv run --no-sync pytest -q src/openpi/picf/core/pipeline_test.py \
  -k "ownership_prior or active_slot_filter or slot_assignment_ignores":
  4 passed.

uv run --no-sync pytest -q src/openpi/picf/core/training_test.py \
  -k support_diversity:
  4 passed.
```

### Acceptance Metrics

Raw same-role overlap is no longer sufficient after active/dustbin filtering.
Inactive duplicate anchors may intentionally overlap. The active subset is the
real contract:

```text
Primary:
  aqr_active_same_role_support_overlap_max
  aqr_active_same_role_support_overlap_mean
  aqr_active_anchor_count
  aqr_inactive_anchor_fraction
  aqr_active_anchor_count_role_0..3

Secondary:
  raw aqr_same_role_support_overlap_max
  posterior_recycle_rate
  posterior_address_update_rate_mean
  posterior_identity_switch_rate
  loss_action_default_equiv
  loss_anchor_pv
  loss_mapg_cycle
  loss_mapg_routing
  anchor overlay images
```

Pass condition for the repair:

```text
1. active_same_role_overlap stays far below the old 0.95-0.99 collapse band;
2. active_anchor_count stays plausible, not one active slot and not all slots;
3. action_default_equiv and recycle health do not degrade relative to the
   recent prefix-stopgrad/all-scope diagnostics;
4. overlays show active anchors on distinct useful regions, while inactive
   anchors may duplicate or drift without controlling assignment.
```

### Ten-Hour A5/A7 Experiment Matrix

This matrix is designed to run without intervention for roughly ten hours.
Each machine runs bounded diagnostics sequentially under tmux. The goal is not
to maximize CALVIN success yet; it is to determine whether active/dustbin
capacity is the correct root repair under both anchor-isolated and cotrain
pressure.

```text
A5 primary isolation line:
  run 1: active/dustbin anchor_only, burnin=4, unroll=1, low task pressure
  run 2: same profile with unroll=2, burnin=1 or 2
  run 3: same unroll=2/burnin=1 with a wider active cap for capacity sensitivity

Purpose:
  isolate whether capacity-aware assignment stabilizes active object ownership
  without full policy gradients dominating the reader.

A7 cotrain line:
  run 1: active/dustbin all-scope cotrain, frozen Sonata/V-JEPA/AnyTouch,
         PaliGemma trainable, unroll=2, burnin=1, prefix-stopgrad action path.
  run 2: all-scope cotrain with low action weight and direct action-flow
         gradients into PICF, to test whether action pressure itself destroys
         the active subset.
  run 3: prefix-stopgrad cotrain with a wider active cap for capacity
         sensitivity.

Purpose:
  test whether the repair survives realistic action/semantic pressure and
  whether the raw overlap can be safely reinterpreted through active metrics.
```

Launch script:

```text
scripts/run_picf_active_slot_matrix.sh
```

Remote launch commands:

```bash
# A5
cd /root/openpi_recyclenorm_4ec25ae
tmux new-session -d -s picf_a5_activecap_matrix \
  'PICF_REPO_ROOT=/root/openpi_recyclenorm_4ec25ae scripts/run_picf_active_slot_matrix.sh a5'

# A7
cd /root/openpi_a7_overlay_unroll2_b9ad838
tmux new-session -d -s picf_a7_activecap_matrix \
  'PICF_REPO_ROOT=/root/openpi_a7_overlay_unroll2_b9ad838 scripts/run_picf_active_slot_matrix.sh a7'
```

Tail commands:

```bash
# A5 current matrix log directory
tail -f /mnt/picf_run_logs/picf_a5_activecap_anchor_u1b4_a025_600_ac273a2.log
tail -f /mnt/picf_run_logs/picf_a5_activecap_anchor_u2b1_a025_600_ac273a2.log
tail -f /mnt/picf_run_logs/picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2.log

# A7 current matrix log directory
tail -f /mnt/picf_run_logs/picf_a7_activecap_cotrain_prefix_u2b1_a1_600_ac273a2.log
tail -f /mnt/picf_run_logs/picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2.log
tail -f /mnt/picf_run_logs/picf_a7_activecap_cotrain_prefix_u2b1_max6_a1_600_ac273a2.log
```

Important negative criteria:

```text
If active_same_role_overlap still goes above 0.95 on both isolation and cotrain,
then the active/dustbin selector is not sufficient; the next root work must add
real object-correspondence evidence, e.g. tracklets/proposals/IsSameObject
probe supervision, not just more schedule tuning.

If active_same_role_overlap is low but action_default_equiv worsens sharply,
then the active cap is too restrictive or the active subset is not aligned with
task-relevant objects. Adjust max_per_role/threshold before changing losses.
```

### 2026-05-14 02:20 Live Deployment Check

Local repository state:

```text
commit:
  f09d18d Add active slot experiment matrix
  ac273a2 Add capacity-aware AQR active slot filter

local checks:
  python py_compile on PICF core/train/audit files: PASS
  bash -n scripts/run_picf_active_slot_matrix.sh: PASS
  git diff --check: PASS
  scripts/verify_picf_owm_contract.py: PASS
  scripts/picf_owm_strict_diagnose.py --fail-on-fail: PASS
  scripts/picf_owm_dataflow_trace.py --fail-on-fail: PASS
  scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
  pipeline active-slot/ownership tests: 4 passed
  training support-diversity tests: 4 passed
```

Remote launch state:

```text
A5:
  tmux: picf_a5_activecap_matrix
  first run: picf_a5_activecap_anchor_u1b4_a025_600_ac273a2
  config: anchor_only, unroll=1, burnin=4, action_weight=0.25,
          active_max_per_role=4, active_overlap=0.75
  observed speed at step 50: ~11.5 sec/step

A7:
  tmux: picf_a7_activecap_matrix
  first run: picf_a7_activecap_cotrain_prefix_u2b1_a1_600_ac273a2
  config: all-scope cotrain, unroll=2, burnin=1, action_weight=1.0,
          prefix-stopgrad enabled, PaliGemma trainable,
          active_max_per_role=4, active_overlap=0.75
  observed early speed: ~22.7 sec/step before first metrics point
```

First A5 step-50 result:

```text
loss_total: 0.1050
loss_action_default_equiv: 0.1404
loss_action_active7: 0.4624
loss_anchor_pv: 2.8511
loss_mapg_cycle: 0.3678
loss_mapg_routing: 0.8109
loss_mapg_support_diversity: 0.3670

raw aqr_same_role_support_overlap_max: 0.2402
aqr_active_same_role_support_overlap_max: 0.1621
aqr_active_same_role_support_overlap_mean: 0.0305
aqr_active_anchor_count: 14.0
aqr_inactive_anchor_fraction: 0.4167
active role counts: role0=2, role1=4, role2=4, role3=4

posterior_recycle_rate: 0.4978
posterior_stable_slot_fraction: 0.1111
grad_norm: 8.3377
```

Interpretation:

```text
This is the first run where the acceptance metric is active overlap, not raw
overlap. Step 50 is still warmup and cannot prove long-run stability, but it
does prove the new active/dustbin path is live:

  active slots are not all slots;
  inactive fraction is nonzero and plausible;
  active same-role overlap is far below the previous 0.95-0.99 collapse band;
  the active role counts retain multiple candidate objects per semantic role.

The posterior recycle rate is still high at step 50. That is not yet a failure,
because the run is in LR warmup and active-slot selection operates before the
posterior identity state has settled. It becomes a failure only if recycle
remains high after the active subset stays stable through the 100/200/300-step
checkpoints.
```

Second live check:

```text
A5 step 100:
  loss_total: 0.1059
  loss_action_default_equiv: 0.1473
  loss_action_active7: 0.5009
  loss_anchor_pv: 3.0886
  loss_mapg_cycle: 0.3937
  loss_mapg_routing: 0.8694
  loss_mapg_support_diversity: 0.4076
  raw aqr_same_role_support_overlap_max: 0.7943
  aqr_active_same_role_support_overlap_max: 0.3166
  aqr_active_same_role_support_overlap_mean: 0.0691
  aqr_active_anchor_count: 14.0
  aqr_inactive_anchor_fraction: 0.4167
  posterior_recycle_rate: 0.4551
  posterior_stable_slot_fraction: 0.1111
  grad_norm: 13.6992
  speed: 0.0867 steps/sec

A7 step 50:
  loss_total: 0.1740
  loss_action_default_equiv: 0.1707
  loss_action_active7: 0.5006
  loss_anchor_pv: 2.8643
  loss_mapg_cycle: 0.3799
  loss_mapg_routing: 0.8403
  loss_mapg_support_diversity: 0.3660
  raw aqr_same_role_support_overlap_max: 0.2748
  aqr_active_same_role_support_overlap_max: 0.1673
  aqr_active_same_role_support_overlap_mean: 0.0294
  aqr_active_anchor_count: 14.0
  aqr_inactive_anchor_fraction: 0.4167
  posterior_recycle_rate: 0.5737
  posterior_stable_slot_fraction: 0.1094
  grad_norm: 2.1651
  speed: 0.0438 steps/sec
```

Interpretation:

```text
The key early signal is positive: active overlap remains far below the old
collapse band in both anchor isolation and all-scope cotrain. A5 raw overlap
rose from 0.24 to 0.79 by step 100, but active overlap stayed at 0.32. This is
exactly the distinction the repair is meant to expose: inactive/dustbin slots
may overlap because they are not allowed to dominate assignment.

The remaining caution is posterior recycle. Recycle is still around 0.45-0.57
during warmup. This must improve or at least not corrupt active assignments
after step 200/300. If active overlap stays low but recycle remains high, the
next root issue is posterior identity carry, not support assignment capacity.
```

Third live check:

```text
A5 step 150:
  raw overlap: 0.9856
  active overlap max/mean: 0.5973 / 0.2592
  active_anchor_count: 11.27
  inactive_anchor_fraction: 0.5304
  posterior_recycle_rate: 0.6585
  loss_anchor_pv: 3.3331
  loss_mapg_routing: 0.8850
  loss_total: 0.1069

A5 step 200:
  raw overlap: 0.9999
  active overlap max/mean: 0.4020 / 0.2481
  active_anchor_count: 6.88
  inactive_anchor_fraction: 0.7133
  posterior_recycle_rate: 0.8594
  loss_anchor_pv: 4.7375
  loss_mapg_routing: 0.7569
  loss_total: 0.1049

A7 step 100:
  raw overlap: 0.3219
  active overlap max/mean: 0.2121 / 0.0439
  active_anchor_count: 14.0
  inactive_anchor_fraction: 0.4167
  posterior_recycle_rate: 0.4805
  loss_action_default_equiv: 0.0936
  loss_action_active7: 0.3758
  loss_anchor_pv: 4.2699
  loss_mapg_routing: 0.9096
  loss_total: 0.1343
```

Interpretation:

```text
The A5 isolation run is no longer the preferred proxy for final behavior. It
proves the active/dustbin path prevents all fixed queries from being treated as
valid objects, but it also shows anchor-only pressure can become too sparse:
active count fell from 14 to 6.88 while recycle rose to 0.86. This is not the
old failure mode, because active overlap stayed far below 0.95, but it is an
under-allocation / posterior carry warning.

The A7 cotrain run is the more important signal for production. Through step
100, it keeps active overlap low, keeps 14 active anchors, reduces action loss,
and does not reproduce the old raw/active collapse. That supports the current
hypothesis that task/semantic/action context helps select useful active slots,
whereas pure anchor-only isolation lacks enough task pressure to maintain a
stable useful active set.
```

Additional A5 step-250 observation:

```text
A5 step 250:
  raw overlap: 0.999996
  active overlap max/mean: 0.1023 / 0.0717
  active_anchor_count: 4.57
  inactive_anchor_fraction: 0.8096
  posterior_recycle_rate: 0.9474
  posterior_residual_summary_norm: 1375.03
  loss_anchor_pv: 4.7546
  loss_slot_jepa_raw_logged_only: 321.07
  loss_total: 0.1028
```

Interpretation:

```text
This confirms that the A5 anchor-only line is not reproducing the old active
collapse. Instead, it is over-pruning active capacity: active overlap is low
because the selector has demoted most slots to inactive/dustbin. That is useful
diagnostically, but it is not a production profile by itself. The remaining A5
u2/b1 and max6 runs are still needed to separate three causes:

  1. u1/b4 state-only burn-in makes isolated anchor training over-prune;
  2. max_per_role=4 is too restrictive for anchor-only isolation;
  3. anchor-only lacks task/action context and is therefore the wrong proxy.

The A7 cotrain line remains the primary production signal.

### 2026-05-14 03:40 Active-Slot Matrix Runtime Check And Theory Gate

The A5/A7 active-slot matrix is running normally on both machines. This is a
runtime check, not an acceptance claim.

Remote liveness:

```text
A5:
  tmux: picf_a5_activecap_matrix, picf_a5_activecap_monitor
  current first run:
    picf_a5_activecap_anchor_u1b4_a025_600_ac273a2
  GPU:
    both ranks resident, ~17.2GB each, active utilization
  progress:
    step 350 logged; train tail advancing through step 380+

A7:
  tmux: picf_a7_activecap_matrix, picf_a7_activecap_monitor
  current first run:
    picf_a7_activecap_cotrain_prefix_u2b1_a1_600_ac273a2
  GPU:
    both ranks resident, high memory residency, active utilization
  progress:
    step 200 logged; train tail advancing beyond step 200
```

Current A5 status:

```text
step: 350
loss_total: 0.1049
loss_action_default_equiv: 0.1389
loss_action_active7: 0.4841
loss_anchor_pv: 4.7557
loss_mapg_cycle: 0.4688
loss_mapg_routing: 0.6571
loss_mapg_support_diversity: 0.0410

raw same_role_support_overlap_max: 0.99998
active same_role_support_overlap_max: 0.0320
active same_role_support_overlap_mean: 0.0233
active_anchor_count: 4.23
inactive_anchor_fraction: 0.8238
posterior_recycle_rate: 0.9568
posterior_residual_summary_norm: 3420.17
```

A5 interpretation:

```text
This is not the old "all active anchors overlap" failure. Active overlap is
low because the capacity-aware selector demotes most duplicate candidates.

It is also not a production success. Anchor-only pressure is over-pruning:
active_anchor_count has fallen to about one active object per role and recycle
has returned to a high-reset regime. Therefore A5 first run is useful only as
an isolation result:

  active/dustbin routing works;
  anchor-only without enough task/semantic/action context is too sparse;
  the remaining A5 u2/b1 and max6 branches are required to test whether this
  is caused by unroll/burn-in shape or by an active-capacity threshold.
```

Current A7 status:

```text
step: 200
loss_total: 0.1246
loss_action_default_equiv: 0.0742
loss_action_active7: 0.3332
loss_anchor_pv: 4.2988
loss_mapg_cycle: 0.4225
loss_mapg_routing: 0.9625
loss_mapg_support_diversity: 0.4761

raw same_role_support_overlap_max: 0.8016
active same_role_support_overlap_max: 0.4958
active same_role_support_overlap_mean: 0.1487
active_anchor_count: 13.39
inactive_anchor_fraction: 0.4421
posterior_recycle_rate: 0.9286
posterior_residual_summary_norm: 758.80
```

A7 interpretation:

```text
A7 remains the production-relevant branch because it includes task/action/
semantic cotrain and keeps many active anchors. However, the step-200 row is a
real warning: recycle has returned to a high-reset regime while action loss is
falling. This means the first A7 configuration is not accepted yet.

The important point is that this is exactly what the matrix is meant to
separate. The first A7 run tests full action weight plus prefix-stopgrad with
max 4 active slots per role. The queued A7 runs test:

  1. lower action weight and no prefix-stopgrad,
     picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2;

  2. same full action weight and prefix-stopgrad, but wider active capacity,
     picf_a7_activecap_cotrain_prefix_u2b1_max6_a1_600_ac273a2.

If lower action weight reduces recycle without destroying active overlap, the
failure is cotrain pressure scale. If max6 reduces recycle/over-pruning while
keeping action loss competitive, the failure is capacity threshold. If both
fail, active-slot filtering alone is not the final root repair and we must move
to a differentiable merge/consolidation or better object-correspondence source.
```

Mathematical gate for this 10-hour matrix:

```text
Let A_j be the support distribution for slot j and M_j be its active mask.

Raw overlap:
  max_{same role i,j} overlap(A_i, A_j)

Active overlap:
  max_{same role i,j: M_i=M_j=1} overlap(A_i, A_j)

The repair is acceptable only if it keeps active overlap low while preserving
enough active capacity:

  active_overlap_max <= 0.60 tail average
  active_anchor_count in a plausible range, roughly 8-16 for current CALVIN
  posterior_recycle_rate not saturating near 0 or 1
  action_default_equiv decreases without forcing recycle back to saturation

Low active overlap with active_anchor_count near 4 is over-pruning, not object
binding. Low action loss with recycle near 1 is not belief-state success.
```

Paper-to-code consistency:

```text
Object Binding in pretrained ViTs / IsSameObject:
  supports the pairwise same-object subspace interpretation. Our
  binding_signature_proj and offline probes address representation
  separability; the current active-slot matrix addresses assignment/capacity,
  not representation absence.

MetaSlot:
  supports dynamic effective slot count rather than treating the fixed AQR
  query budget as the true object count. Our active_anchor mask is the current
  non-learned, guarded version of this principle.

Slot Merging:
  warns that pure pruning can discard useful duplicate evidence; when multiple
  slots represent the same object, differentiable consolidation can be better
  than deletion. If this matrix shows persistent over-pruning or recycle
  rebound, the next principled repair should be merge/consolidation, not a new
  unrelated loss.

References:
  https://arxiv.org/abs/2510.24709
  https://arxiv.org/abs/2505.20772
  https://arxiv.org/abs/2603.11246
```

Operational decision:

```text
Do not stop the matrix at the first warning row. The first A5/A7 rows have
already exposed the two intended edge cases: A5 over-pruning under anchor-only
pressure, and A7 recycle rebound under full cotrain pressure. The queued runs
are required to identify whether the controlling variable is unroll/burnin,
action pressure, prefix-stopgrad, or active capacity.

The matrix should continue unattended unless:
  - a process exits,
  - CUDA OOM occurs,
  - metrics stop advancing for more than one expected log interval,
  - losses become NaN/Inf.
```

03:40 second liveness poll:

```text
A5:
  hostname: ZWWQO6
  tmux alive: picf_a5_activecap_matrix, picf_a5_activecap_monitor
  GPU: both ranks resident, about 17.2GB each, active utilization
  latest metrics:
    run: picf_a5_activecap_anchor_u1b4_a025_600_ac273a2
    step: 400
    loss_total: 0.1043
    loss_action_default_equiv: 0.1346
    loss_anchor_pv: 4.7565
    loss_mapg_cycle: 0.4731
    loss_mapg_routing: 0.6585
    raw same_role_support_overlap_max: 0.99999
    active same_role_support_overlap_max: 0.0488
    active same_role_support_overlap_mean: 0.0344
    active_anchor_count: 4.28
    inactive_anchor_fraction: 0.8217
    posterior_recycle_rate: 0.9940
    grad_norm: 1.55

A7:
  hostname: qgE72e
  tmux alive: picf_a7_activecap_matrix, picf_a7_activecap_monitor
  GPU: both ranks resident with active utilization
  latest metrics:
    run: picf_a7_activecap_cotrain_prefix_u2b1_a1_600_ac273a2
    step: 200
    loss_total: 0.1246
    loss_action_default_equiv: 0.0742
    loss_anchor_pv: 4.2988
    loss_mapg_cycle: 0.4225
    loss_mapg_routing: 0.9625
    raw same_role_support_overlap_max: 0.8016
    active same_role_support_overlap_max: 0.4958
    active same_role_support_overlap_mean: 0.1487
    active_anchor_count: 13.39
    inactive_anchor_fraction: 0.4421
    posterior_recycle_rate: 0.9286
    grad_norm: 2.23
```

The second poll confirms normal runtime, but not success. A5 is a clear
over-pruning branch: active overlap is low only because most slots are demoted.
A7 is the scientifically useful first branch: it has enough active capacity and
low action loss, but recycle is too high. The matrix must therefore continue to
the lower-action/no-prefix and max6-capacity branches before any architecture
decision is made.

03:52 third liveness poll after the requested 10-minute wait:

```text
A5:
  tmux alive.
  GPU: both ranks resident, about 17.2GB each.
  latest metrics:
    run: picf_a5_activecap_anchor_u1b4_a025_600_ac273a2
    step: 450
    loss_total: 0.1042
    loss_action_default_equiv: 0.1335
    loss_anchor_pv: 4.7550
    loss_mapg_cycle: 0.4699
    loss_mapg_routing: 0.6711
    raw same_role_support_overlap_max: 0.99997
    active same_role_support_overlap_max: 0.0959
    active same_role_support_overlap_mean: 0.0619
    active_anchor_count: 4.60
    inactive_anchor_fraction: 0.8083
    posterior_recycle_rate: 0.9961
    grad_norm: 8.02
  train tail:
    advancing beyond step 460.

A7:
  tmux alive.
  GPU: both ranks resident and near full utilization.
  latest metrics:
    still step 200 because log_interval=50 and the run is about 22.5s/step.
  train tail:
    advancing beyond step 239, so the run is not stuck.
```

This confirms normal runtime after the 10-minute wait. It also strengthens the
scientific interpretation: A5 first branch is an over-pruning/dustbin-capacity
isolation result. A7 first branch is alive and production-relevant, but the
next accepted/rejected decision needs the next metrics rows and the queued
lower-action/max6 branches.

04:55 one-hour requested poll:

```text
A5:
  tmux alive.
  current run:
    picf_a5_activecap_anchor_u2b1_a025_600_ac273a2
  first run completed:
    picf_a5_activecap_anchor_u1b4_a025_600_ac273a2

  first run final step 600:
    loss_total: 0.1055
    loss_action_default_equiv: 0.1439
    loss_anchor_pv: 4.7555
    raw same_role_support_overlap_max: 0.99998
    active same_role_support_overlap_max: 0.1064
    active_anchor_count: 4.74
    inactive_anchor_fraction: 0.8025
    posterior_recycle_rate: 0.9366

  second run step 200:
    loss_total: 0.1070
    loss_action_default_equiv: 0.1561
    loss_anchor_pv: 4.7233
    raw same_role_support_overlap_max: 0.99974
    active same_role_support_overlap_max: 0.5313
    active same_role_support_overlap_mean: 0.3706
    active_anchor_count: 7.72
    inactive_anchor_fraction: 0.6781
    posterior_recycle_rate: 0.0030
    grad_norm: 37.28

A7:
  tmux alive.
  current run:
    picf_a7_activecap_cotrain_prefix_u2b1_a1_600_ac273a2
  latest step 350:
    loss_total: 0.1171
    loss_action_default_equiv: 0.0591
    loss_action_active7: 0.2678
    loss_anchor_pv: 4.7224
    loss_mapg_cycle: 0.4794
    loss_mapg_routing: 0.8595
    raw same_role_support_overlap_max: 0.9975
    active same_role_support_overlap_max: 0.4446
    active same_role_support_overlap_mean: 0.2478
    active_anchor_count: 8.66
    inactive_anchor_fraction: 0.6394
    posterior_recycle_rate: 0.9981
    posterior_residual_summary_norm: 2814.58
```

Interpretation:

```text
Runtime:
  normal. Both matrices are still alive and metrics are advancing.

A5 u1/b4:
  rejected as a production proxy. It achieves low active overlap by over-
  demoting slots and keeps recycle high.

A5 u2/b1:
  materially better capacity than u1/b4: active_anchor_count increases from
  about 4.7 to about 7.7 and active overlap is near the acceptance threshold.
  However recycle is now saturated in the opposite direction, near zero, and
  this remains anchor-only. It is useful for isolating unroll/burn-in capacity,
  not for production acceptance.

A7 first branch:
  action loss improves substantially, and active overlap stays under the
  provisional 0.60 gate. This is the best action signal in the matrix so far.
  But posterior_recycle_rate is saturated near 1.0, which means the belief
  state is still not accepted. The first A7 branch therefore shows that active
  capacity can preserve separated active supports under action pressure, but
  the posterior reset/recycle dynamics are still unstable under full action
  plus prefix-stopgrad.
```

Mathematical consequence:

```text
The old failure mode was:
  all fixed same-role slots are forced to explain the same support.

The current active-slot repair changes that to:
  an active subset can remain separated, while redundant slots are demoted.

The remaining failure is now posterior identity dynamics, not raw AQR support
assignment alone. A low active-overlap value is necessary but not sufficient.
The belief filter must also avoid recycle saturation:

  0 << posterior_recycle_rate << 1

with a non-trivial stable_slot_fraction. Therefore the queued A7 branches are
still required:

  lower action / no prefix-stopgrad:
    tests whether recycle saturation is caused by action-gradient scale or the
    prefix stop-gradient boundary.

  max6 active capacity:
    tests whether the max4 active cap makes posterior dynamics too brittle.
```

Do not stop the matrix yet. The one-hour poll confirms that the experiment is
working as a discriminator, but the production answer depends on the queued A7
lower-action/no-prefix and max6-capacity branches.

06:00 post-sleep requested poll and overlay audit:

```text
Local verification:
  python scripts/verify_picf_owm_contract.py:
    PASS
  python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
    PASS
  python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
    PASS
  python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
    PASS
  py_compile on Python sources:
    PASS

A5:
  tmux alive:
    picf_a5_activecap_matrix
    picf_a5_activecap_monitor
  current run:
    picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2
    advancing at about 8.9 sec/step, before first metrics row

  completed u1/b4 final step 600:
    loss_total                         = 0.1055
    loss_action_default_equiv           = 0.1439
    raw same_role_support_overlap_max   = 0.99998
    active same_role_support_overlap_max= 0.1064
    active_anchor_count                 = 4.74
    inactive_anchor_fraction            = 0.8025
    posterior_recycle_rate              = 0.9366
    posterior_recycle_logit_mean        = 2.73
    posterior_address_update_rate_mean  = 0.00218

  completed u2/b1 final step 600:
    loss_total                         = 0.1067
    loss_action_default_equiv           = 0.1533
    raw same_role_support_overlap_max   = 0.99996
    active same_role_support_overlap_max= 0.2304
    active same_role_support_overlap_mean=0.1811
    active_anchor_count                 = 5.21
    inactive_anchor_fraction            = 0.7829
    posterior_recycle_rate              = 0.0021
    posterior_recycle_logit_mean        = -6.30
    posterior_address_update_rate_mean  = 0.0336
    posterior_identity_switch_rate      = 0.8011
    posterior_binding_top1_margin_mean  = 0.1109

A7:
  tmux alive:
    picf_a7_activecap_matrix
    picf_a7_activecap_monitor
  current run:
    picf_a7_activecap_cotrain_prefix_u2b1_a1_600_ac273a2
    advancing beyond step 570 at about 22.7 sec/step

  latest scalar row, step 550:
    loss_total                         = 0.1184
    loss_action_default_equiv           = 0.0618
    loss_action_active7                 = 0.2809
    loss_anchor_pv                      = 4.6347
    loss_mapg_routing                   = 0.9767
    loss_mapg_cycle                     = 0.4713
    raw same_role_support_overlap_max   = 0.9837
    active same_role_support_overlap_max= 0.5950
    active same_role_support_overlap_mean=0.1512
    active_anchor_count                 = 13.01
    inactive_anchor_fraction            = 0.4577
    posterior_recycle_rate              = 0.9976
    posterior_recycle_logit_mean        = 6.09
    posterior_recycle_gate_std          = 6.6e-6
    posterior_stable_slot_fraction      = 0.1072
    posterior_address_update_rate_mean  = 8.9e-5
    posterior_identity_switch_rate      = 0.7994
    posterior_binding_top1_margin_mean  = 0.1094

  useful positive signals:
    action_default_equiv improved from about 0.1707 at step 50 to 0.0618.
    temporal view mass uses both views:
      temporal_view_mass_0 = 0.456
      temporal_view_mass_1 = 0.544
    active same-role overlap is near, but not safely below, the 0.60 gate.

  hard negative signals:
    recycle is saturated high and nearly constant across slots.
    address update is effectively starved.
    binding margin remains low.
```

Overlay audit from JSON diagnostics:

```text
A5 u2/b1:
  step 100 graph same-role min pixel distances:
    role0=2.34, role1=0.77, role2=1.82, role3=2.13
  step 400 graph same-role min pixel distances:
    role0=10.98, role1=0.002, role2=0.002, role3=0.001
  step 600 graph same-role min pixel distances:
    role0=3.01, role1=0.004, role2=0.237, role3=0.099
  posterior role1 min pixel distance:
    0.0 at every sampled overlay

A5 u1/b4:
  step 100 graph same-role min pixel distances:
    role0=4.03, role1=1.01, role2=0.95, role3=0.07
  step 300 graph same-role min pixel distances:
    role0=8.16, role1=0.018, role2=0.019, role3=0.014
  step 600 graph same-role min pixel distances:
    role0=6.84, role1=0.005, role2=0.002, role3=0.002
  posterior role1 min pixel distance:
    0.0 at every sampled overlay
```

Interpretation of the overlay audit:

```text
1. The raw graph anchors do physically duplicate same-role slots. This is not
   just an overlap metric artifact.
2. The active-slot filter is doing a real job: it can demote many duplicates,
   which is why active overlap can look healthy while raw overlap is 1.0.
3. A5 anchor-only is not a production proxy. It either over-prunes with high
   recycle or keeps recycle near zero with high identity switching.
4. A7 is the better production proxy because it includes task/action/semantic
   pressure, but it still fails the posterior identity criterion: recycle
   saturation starves address update and prior memory.
```

Theoretical implication:

```text
The current failure is not "all anchors are one object" in the active readout.
The more precise failure is:

  fixed query capacity creates redundant same-role proposals
  -> active filter can select a plausible subset
  -> action can still improve
  -> posterior recycle can saturate
  -> address update is multiplied by (1 - recycle)
  -> identity continuity is not learned
  -> overlays show repeated physical duplicate proposals

Therefore the remaining blocker is a joint object-count / identity-continuity
problem, not a single scalar loss problem.
```

Connection to 2025+ object-binding literature:

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
  shows that IsSameObject is a pairwise/quadratic, low-dimensional binding
  subspace in ViT activations. Our binding_signature path follows this idea:
  it adds a projected pairwise same-object term, not a new hard loss.

STORM:
  argues for multi-phase adaptation: stabilize object-centric slots with
  visual/semantic structure before joint manipulation cotrain. The A7 result
  supports this: action loss falls, but object identity/recycle is not yet
  accepted.

MetaSlot / adaptive-slot methods:
  warn that fixed slot counts create duplicate parts when object count varies.
  Our active-slot filter is a runtime analog of this idea, but the overlay
  audit shows that demotion alone is not enough; posterior identity must also
  remain stable.

SLOT-CONTRAST:
  emphasizes temporal consistency and active slot sparsity. This aligns with
  the acceptance gates here: recycle must be moderate, address updates nonzero,
  binding margins healthy, and active anchors must remain physically distinct.
```

Next decision:

```text
Do not stop the matrix before the queued branches. The first A5/A7 rows have
already exposed useful failure modes, but the queued branches are exactly the
causal tests we need:

  A7 lower-action/no-prefix:
    If recycle improves, action scale or prefix boundary is the trigger.
    If recycle remains saturated, recycle calibration/object-count dynamics are
    the trigger.

  A7 max6:
    If active count and recycle improve, max4 over-prunes production slots.
    If raw duplicate overlays remain and recycle stays bad, capacity alone is
    not enough.

  A5 max6:
    If anchor-only still collapses/duplicates, then the issue is not action
    alone; the anchor objective still lacks a clean object-count/merge prior.
```

Acceptance after queued branches:

```text
Required:
  active same-role overlap max < 0.60
  active_anchor_count plausible, not <= 4 by over-pruning
  posterior_recycle_rate not saturated high or low
  posterior_address_update_rate_mean nonzero
  posterior_binding_top1_margin_mean improves materially
  overlays show active anchors on distinct physical regions

Reject even if action loss is low:
  recycle > 0.95 or < 0.01 for long windows
  posterior role duplicates remain exactly co-located
  raw duplicates are only hidden by inactive demotion
```

06:16 stability re-check after the extra 12-minute window:

```text
A5:
  current max6 branch produced first metrics row at step100:
    run:
      picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2
    loss_total                         = 0.1053
    loss_action_default_equiv           = 0.1425
    loss_anchor_pv                      = 3.3292
    loss_mapg_routing                   = 0.9392
    loss_mapg_cycle                     = 0.3888
    raw same_role_support_overlap_max   = 0.2327
    active same_role_support_overlap_max= 0.2168
    active same_role_support_overlap_mean=0.0310
    active_anchor_count                 = 20.0
    inactive_anchor_fraction            = 0.1667
    posterior_recycle_rate              = 0.2623
    posterior_recycle_logit_mean        = -1.19
    posterior_address_update_rate_mean  = 0.0323
    posterior_identity_switch_rate      = 0.7950
    posterior_binding_top1_margin_mean  = 0.1100
    speed                               = 8.9 sec/step

A7:
  first full-action prefix branch completed at step600:
    loss_total                         = 0.1213
    loss_action_default_equiv           = 0.0676
    raw same_role_support_overlap_max   = 0.9901
    active same_role_support_overlap_max= 0.6168
    active_anchor_count                 = 12.91
    posterior_recycle_rate              = 0.9979
    posterior_recycle_logit_mean        = 6.20
    posterior_address_update_rate_mean  = 7.9e-5
    posterior_identity_switch_rate      = 0.8056

  queued lower-action/no-prefix branch started automatically:
    run:
      picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2
    step:
      advancing, before first metrics row
```

Interpretation:

```text
1. Runtime is confirmed stable after branch transition. No manual restart was
   needed.
2. A7 full-action prefix is rejected for posterior identity health despite good
   action loss. This is now a clear negative result, not an inconclusive row.
3. A5 max6 step100 is the first branch in this matrix that satisfies all early
   structural gates simultaneously:
     raw overlap low
     active overlap low
     active capacity high
     recycle moderate
     address update nonzero
   It is still anchor-only, so it cannot be accepted as production, but it
   strongly supports the hypothesis that max4 was too restrictive.
4. The remaining decisive checks are A5 max6 at step200/300/600 and the A7
   lower-action/no-prefix branch at step50/100/200. If A7 lower action still
   saturates recycle, action cotrain needs a staged or lower-gradient interface.
   If it stays moderate, the production recipe should combine max6 active
   capacity with lower action gradient pressure before full action weight.

06:22 unattended-watch gate:

```text
Both remote training matrices are live and using GPUs.

A5:
  tmux:
    picf_a5_activecap_matrix
    picf_a5_activecap_watch10h
  current run:
    picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2
  process:
    torchrun world_size=2 is alive
  latest structured metrics:
    step                               = 150
    loss_total                         = 0.105153
    loss_action_default_equiv           = 0.141223
    loss_anchor_pv                      = 4.408022
    loss_mapg_routing                   = 0.993615
    raw same_role_support_overlap_max   = 0.659477
    active same_role_support_overlap_max= 0.476170
    active same_role_support_overlap_mean=0.129296
    active_anchor_count                 = 18.69
    inactive_anchor_fraction            = 0.22125
    posterior_recycle_rate              = 0.182168
    posterior_recycle_logit_mean        = -1.613870
    posterior_recycle_gate_std          = 0.000185
    posterior_address_update_rate_mean  = 0.028754
    posterior_identity_switch_rate      = 0.796111
    posterior_binding_top1_margin_mean  = 0.110327
    speed                               = about 8.9 sec/step
  watch log:
    /mnt/picf_run_logs/picf_a5_activecap_watch10h.log

A7:
  tmux:
    picf_a7_activecap_matrix
    picf_a7_activecap_watch10h
  current run:
    picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2
  process:
    torchrun world_size=2 is alive
  current branch parameters:
    scope=all
    unroll=2
    burnin=1
    action scale=0.25
    active_max_per_role=4
    prefix_stopgrad=no
    PaliGemma trainable=true
    Sonata/V-JEPA/AnyTouch frozen=true
  latest structured metrics:
    not yet emitted for the flow branch at this timestamp.
    The completed full-action prefix branch remains a rejected negative row:
      step600 action_default_equiv=0.067610
      active overlap=0.616848
      recycle=0.997863
      address update=7.9e-5
  watch log:
    /mnt/picf_run_logs/picf_a7_activecap_watch10h.log
```

Interpretation:

```text
1. The matrix is running normally. No restart or code change is justified before
   A5 max6 reaches step200/300 and A7 flow reaches its first metrics gate.
2. A5 max6 remains the strongest structural candidate, but step150 is no longer
   as clean as step100: raw overlap increased from 0.2327 to 0.6595 and
   active overlap increased from 0.2168 to 0.4762. This is still within the
   active acceptance gate, but it warns that max6 capacity is necessary, not
   sufficient.
3. A7 full-action prefix proves that action loss alone can look good while
   posterior identity is broken. The current A7 flow branch is the causal test
   for whether lower action pressure and removing prefix-stopgrad improve
   recycle/address dynamics.
4. Do not introduce a new architecture patch before these two causal branches
   emit metrics. Otherwise we would lose the ability to separate active-cap
   capacity, action-pressure scale, and recycle dynamics.
```

Next hard gates:

```text
A5 max6 step200/300:
  accept as anchor-only structural candidate only if:
    active overlap max stays < 0.60
    active_anchor_count remains > 12
    recycle stays between roughly 0.05 and 0.70
    address update remains nonzero

A7 flow step50/100:
  if recycle drops materially below the rejected 0.997 row:
    action pressure/prefix boundary is a real cause.
  if recycle remains > 0.95:
    recycle calibration or object-count dynamics remains unresolved.

A7 max6:
  only becomes decisive after the A7 flow branch because it tests capacity under
  full action pressure.
```

06:29 tens-step and overlay gate:

```text
Remote port map was revalidated before reading metrics:
  qgE72e / port 28060:
    A7 matrix worktree /root/openpi_a7_overlay_unroll2_b9ad838
  ZWWQO6 / port 29776:
    A5 matrix worktree /root/openpi_recyclenorm_4ec25ae

A7 current branch:
  picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2
  state:
    running normally
    progress bar reached at least step 35
    no structured metrics row yet because the first metrics gate is step50
  confirmed startup contract:
    scope=all
    unroll=2
    burnin=1
    action scale=0.25
    active_max_per_role=4
    PaliGemma trainable=true
    Sonata/V-JEPA/AnyTouch frozen=true
    local_refinement_enabled=false
    recycle_residual_norm_mode=layernorm

A5 current branch:
  picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2
  latest structured metrics:
    step                                = 200
    loss_total                          = 0.105700
    loss_action_default_equiv            = 0.145600
    loss_action_active7                  = 0.512589
    loss_anchor_pv                       = 4.097566
    loss_mapg_routing                    = 0.920289
    loss_mapg_cycle                      = 0.453425
    loss_mapg_support_diversity          = 0.528119
    raw same_role_support_overlap_max    = 0.995910
    active same_role_support_overlap_max = 0.573348
    active same_role_support_overlap_mean= 0.208245
    active_anchor_count                  = 12.16
    inactive_anchor_fraction             = 0.493333
    posterior_recycle_rate               = 0.138333
    posterior_recycle_logit_mean         = -1.969415
    posterior_recycle_gate_std           = 0.000246
    posterior_address_update_rate_mean   = 0.031237
    posterior_identity_switch_rate       = 0.805000
    posterior_binding_top1_margin_mean   = 0.110352
    speed                                = about 9.0 sec/step
```

Overlay audit for A5 step200:

```text
Artifact:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2/
      anchor_overlays/step_000200.png
      anchor_overlays/step_000200.json

Image-level result:
  graph anchors still show local spatial spread around the scene and gripper
  region, but posterior role-1 slots are exactly co-located at the same pixel:
    posterior role 1 slots:
      pixel_xy = [105.2, 114.1] for all seven visible role-1 posterior slots

Pairwise geometry summary from the JSON:
  graph role 1:
    min pixel distance 0.65, mean 8.07, max 17.10
  graph role 2:
    min pixel distance 0.62, mean 4.16, max 8.04
  graph role 3:
    min pixel distance 0.65, mean 4.27, max 8.58
  posterior role 1:
    min pixel distance 0.00, mean 0.00, max 0.00
```

Interpretation:

```text
1. A7 has satisfied the "started and ran tens of steps" requirement, but the
   flow branch has not yet produced step50 metrics. It must not be judged from
   progress-bar loss alone.
2. A5 max6 no longer supports a clean "capacity alone fixes it" conclusion.
   The active filter keeps active overlap just below the provisional gate
   (0.573 < 0.60), but raw overlap has already returned to near one and active
   count has dropped from 20 at step100 to 12.16 at step200.
3. The overlay proves that the high overlap is not a pure metric artifact:
   posterior slots of the same role can become physically identical after the
   graph-level anchors have already produced local candidates.
4. Therefore the current unresolved mechanism is posterior identity/binding
   collapse after candidate generation. Active-slot demotion is helpful, but it
   is not a complete cure.
5. No code patch should be made before A7 flow reaches step50/100 and A5 max6
   reaches step300. Those two gates separate action-pressure effects from
   anchor-only posterior collapse.
```

06:40 causal gate:

```text
A7 flow branch:
  run:
    picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2
  state:
    running normally
    reached structured metrics gate at step50
  step50:
    loss_total                          = 0.103292
    loss_action_default_equiv            = 0.126334
    loss_action_active7                  = 0.465151
    loss_anchor_pv                       = 2.851656
    loss_mapg_routing                    = 0.837551
    loss_mapg_cycle                      = 0.371206
    loss_mapg_support_diversity          = 0.367256
    raw same_role_support_overlap_max    = 0.363132
    active same_role_support_overlap_max = 0.182056
    active same_role_support_overlap_mean= 0.030686
    active_anchor_count                  = 14.0
    inactive_anchor_fraction             = 0.416667
    posterior_recycle_rate               = 0.634925
    posterior_recycle_logit_mean         = 0.584528
    posterior_recycle_gate_std           = 0.000502
    posterior_address_update_rate_mean   = 0.016516
    posterior_identity_switch_rate       = 0.785556
    posterior_binding_top1_margin_mean   = 0.109429
    speed                                = about 22.6 sec/step

A5 max6 branch:
  run:
    picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2
  state:
    running normally
    reached step300
  step300:
    loss_total                          = 0.102821
    loss_action_default_equiv            = 0.122571
    loss_action_active7                  = 0.438201
    loss_anchor_pv                       = 4.704764
    loss_mapg_routing                    = 0.777597
    loss_mapg_cycle                      = 0.481435
    loss_mapg_support_diversity          = 0.321377
    raw same_role_support_overlap_max    = 0.999587
    active same_role_support_overlap_max = 0.349169
    active same_role_support_overlap_mean= 0.215113
    active_anchor_count                  = 6.46
    inactive_anchor_fraction             = 0.730833
    posterior_recycle_rate               = 0.003663
    posterior_recycle_logit_mean         = -5.920321
    posterior_recycle_gate_std           = 0.000035
    posterior_address_update_rate_mean   = 0.037308
    posterior_identity_switch_rate       = 0.786111
    posterior_binding_top1_margin_mean   = 0.111016
    speed                                = about 8.95 sec/step
```

A5 step300 overlay:

```text
Artifact:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2/
      anchor_overlays/step_000300.png
      anchor_overlays/step_000300.json

Image/geometry result:
  graph role 1:
    min pixel distance 0.62, mean 2.32, max 5.33
  graph role 2:
    min pixel distance 0.11, mean 0.81, max 1.58
  graph role 3:
    min pixel distance 0.19, mean 1.18, max 2.74
  posterior role 1:
    min pixel distance 0.00, mean 0.00, max 0.00
    all seven visible role-1 posterior slots are at pixel [107.2, 120.3]
```

Interpretation:

```text
1. A7 flow step50 is the first cotrain row in this matrix that looks structurally
   plausible at the first gate: raw overlap is low, active overlap is low,
   active count is exactly 14, recycle is high but not saturated, and address
   update is nonzero. It directly improves over the rejected A7 full-action
   prefix row at the same action family boundary.
2. A5 max6 is now rejected as a standalone anchor-only cure. It looked clean at
   step100, started degrading at step200, and at step300 raw graph/posterior
   collapse is again visible. The active filter keeps the reported active
   overlap low only by demoting most same-role duplicates; active count has
   collapsed to 6.46.
3. The current evidence argues against "capacity alone fixes anchor collapse".
   It supports the more specific hypothesis that controlled action pressure
   without prefix-stopgrad is needed to keep the active subset task-grounded
   while avoiding the full-action recycle saturation seen in the rejected A7
   branch.
4. The next decisive A7 checks are step100/200. If A7 flow keeps raw overlap low
   and recycle below saturation, use it as the base for the next long run. If it
   also collapses after step100, the root cause is posterior binding dynamics
   rather than only action scale or active-capacity selection.
```

07:45 one-hour audit:

```text
A7 flow branch:
  run:
    picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2
  state:
    still running normally
    progress log reached about step238
    structured metrics reached step200
  step50 -> step200:
    loss_action_default_equiv:
      0.126334 -> 0.102355 -> 0.160800 -> 0.118942
    raw same_role_support_overlap_max:
      0.363132 -> 0.420756 -> 0.641526 -> 0.991242
    active same_role_support_overlap_max:
      0.182056 -> 0.261390 -> 0.361490 -> 0.557258
    active_anchor_count:
      14.0 -> 14.0 -> 13.675 -> 10.085
    posterior_recycle_rate:
      0.634925 -> 0.829716 -> 0.789538 -> 0.614845
    posterior_address_update_rate_mean:
      0.016516 -> 0.007266 -> 0.007560 -> 0.015441

A5 matrix:
  state:
    completed its queued branches
    GPUs are idle after completion
  final max6 row at step600:
    raw same_role_support_overlap_max    = 0.999686
    active same_role_support_overlap_max = 0.470589
    active_anchor_count                  = 7.505
    posterior_recycle_rate               = 0.000400
    posterior_address_update_rate_mean   = 0.036996
```

A7 step200 overlay audit:

```text
Artifact:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2/
      anchor_overlays/step_000200.png
      anchor_overlays/step_000200.json

Image/geometry result:
  graph role 1:
    min pixel distance 0.67, mean 15.41, max 32.99
  graph role 2:
    min pixel distance 0.08, mean 11.36, max 24.04
  graph role 3:
    min pixel distance 0.54, mean 13.05, max 25.29
  posterior role 1:
    min pixel distance 0.00, mean 0.00, max 0.00
    all seven visible role-1 posterior slots are at pixel [90.2, 108.4]
```

Interpretation:

```text
1. The lower-action/no-prefix A7 branch is better than the rejected full-action
   prefix branch with respect to recycle saturation: recycle remains around
   0.61 at step200 instead of saturating near 0.998, and address update remains
   nonzero.
2. It still fails the structural acceptance test by step200. Raw overlap climbs
   back to 0.991, active count drops to about 10, and the overlay shows exact
   posterior co-location for seven same-role slots.
3. This rules out three simpler explanations:
     capacity-only fix:
       false; A5 max6 completed with raw collapse and low active count.
     full-action pressure only:
       incomplete; lowering action pressure helps recycle but not posterior
       co-location.
     metric-only false positive:
       false; overlay geometry shows exact same-pixel posterior duplicates.
4. The remaining root cause is now narrower:
     the graph stage can create multiple local candidates,
     but posterior binding/correction maps same-role candidates back onto one
     physical posterior state.
   That points to the posterior assignment/update interface, not just support
   diversity or active demotion.
5. Next experiment should target posterior binding/correction directly rather
   than adding another outer diversity penalty. Candidate causal knobs should
   be limited to mathematically consistent mechanisms:
     stronger per-slot posterior anti-coalescence during correction,
     role-aware object-count prior before posterior fusion,
     assignment entropy/temperature schedule at the posterior binding step,
     or a posterior matching penalty that operates on physical posterior
     geometry rather than graph supports alone.
```

08:51 requested one-hour follow-up:

```text
A7 runtime:
  machine:
    qgE72e, ssh port 28060
  tmux:
    picf_a7_activecap_matrix
    picf_a7_activecap_monitor
    picf_a7_activecap_watch10h
  current branch:
    picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2
  live state:
    still running normally
    progress bar around step407/450
    both GPUs busy
    latest structured metrics row: step400
  output directory:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2

A5 runtime:
  machine:
    ZWWQO6, ssh port 29776
  state:
    matrix branches completed
    GPUs idle
  important completed branches:
    picf_a5_activecap_anchor_u1b4_a025_600_ac273a2
    picf_a5_activecap_anchor_u2b1_a025_600_ac273a2
    picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2
```

A7 flow step400:

```text
loss_total                              = 0.096719
loss_action_default_equiv               = 0.073750
loss_action_active7                     = 0.313459
loss_anchor_pv                          = 4.641698
loss_mapg_routing                       = 0.958322
loss_mapg_cycle                         = 0.459890
loss_mapg_support_diversity             = 0.517221
aqr_same_role_support_overlap_max       = 0.986589
aqr_active_same_role_support_overlap_max= 0.598955
aqr_active_anchor_count                 = 12.23
aqr_inactive_anchor_fraction            = 0.490417
posterior_recycle_rate                  = 0.859305
posterior_recycle_logit_mean            = 1.930588
posterior_recycle_gate_std              = 0.000136
posterior_address_update_rate_mean      = 0.004486
posterior_identity_switch_rate          = 0.825556
posterior_binding_top1_margin_mean      = 0.109431
posterior_stable_slot_fraction          = 0.110000
speed                                   = about 22.7 sec/step
```

A7 step400 overlay audit:

```text
Artifact:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2/
      anchor_overlays/step_000400.png
      anchor_overlays/step_000400.json

Debug row in overlay JSON:
  aqr_effective_anchor_count      = 13.1840
  aqr_same_role_support_overlap   = 0.9927
  temporal view mass static       = 0.5759
  temporal view mass gripper      = 0.4241
  posterior_identity_switch_rate  = 0.8889
  posterior_recycle_rate          = 0.8665

Image/geometry result:
  graph role 1:
    min pixel distance 0.60, mean 9.76, max 21.33
  graph role 2:
    min pixel distance 0.76, mean 11.87, max 25.97
  graph role 3:
    min pixel distance 0.66, mean 7.43, max 16.44
  posterior role 1:
    min pixel distance 0.00, mean 0.00, max 0.00
    all seven visible role-1 posterior slots are at pixel [66.84, 116.62]
```

Interpretation:

```text
1. The A7 flow branch is running normally, so the matrix infrastructure is not
   the current failure source.
2. A7 flow improves over the rejected full-action prefix branch in one way:
   recycle is not saturated at 0.998, and address update remains nonzero.
3. It still fails the structural acceptance test by step400. The raw
   same-role overlap is again in the collapse band, active overlap is above
   the provisional gate, and posterior role-1 slots are exactly co-located.
4. The overlay separates graph-stage and posterior-stage failure:
     graph anchors are still spatially spread enough to contain multiple
     candidates;
     posterior anchors coalesce to a single physical state.
5. This further rules out "unroll=2 only", "action scale only",
   "active-capacity only", and "metric false positive" as sufficient
   explanations.
6. The remaining root cause is posterior binding/correction. The next code
   change should target the posterior assignment/update interface, not add
   another outer support-diversity loss.
```

Mathematical update:

```text
Current failure shape:
  Q_graph can still produce multiple same-role candidate locations, but
  posterior correction maps them into the same state:

    {x_i^obs}_{i in role r}  ->  x_post,j ~= x_post,k for j != k.

Therefore the missing pressure is not simply support entropy. It is a
state-space occupancy and assignment constraint at the posterior transition:

  B = assignment(previous_slots, observation_anchors)
  x_post = Correction(x_prior, B @ x_obs)

If several same-role rows of B select the same observation mode, the posterior
duplicates are mathematically stable even when graph supports had some spread.

The fix must preserve the belief-filter interpretation:
  - current evidence remains authoritative;
  - no hard random jitter;
  - no cache-as-truth;
  - no extra unrelated head;
  - posterior assignment should be discouraged from mapping multiple active
    same-role slots onto one physical observation mode unless object count is
    genuinely low and the extra slots are sent to dustbin/inactive.
```

Literature note:

```text
The 2025 object-binding ViT probe result supports using pairwise same-object
subspaces as evidence, but it does not imply that a downstream belief filter
will preserve separate object files automatically. Our current metrics match
that distinction: the graph/token stage retains enough pairwise signal to
spread candidates, while the posterior object-file update coalesces them.

Therefore the next principled repair is not another feature extractor or a
new auxiliary task. It is posterior-stage set assignment with same-role
occupancy control, using the existing binding signatures/geometry as evidence.
```

Root-cause code audit after the 08:51 poll:

```text
New code-level finding:
  The persistent posterior object files were still initialized as an exactly
  symmetric set:

    posterior_slot_identity_std = 0.0
    task_slot_identity_std      = 0.0
    posterior_bootstrap_from_observation = False

Why this matters:
  AQR graph queries already have role/type/coverage identity, so graph anchors
  can separate for a while. Posterior slots, however, are the recurrent object
  files. With zero identity seed and no first-step geometry bootstrap, all
  same-role scene posterior slots start with identical h/c/token/address and
  identical geometry. If the shared residual/recycle path sees the same input
  for those slots, the posterior update is equivariant and keeps them
  identical:

    S_j^0 = S_k^0  and  U(S_j, o_t) = U(S_k, o_t)
    => S_j^t = S_k^t

  This is exactly what the A7 step400 overlay shows: graph candidates are not
  all identical, but posterior role-1 object files become identical in pixel,
  support mass, recycle gate, confidence, and address update.

Why this is a root fix rather than another patch:
  The object-binding paper motivates using pairwise same-object evidence, but
  the model still needs separate object files to preserve that evidence through
  time. A nonzero posterior identity seed and first-step FPS geometry bootstrap
  are the minimal object-file birth prior. They do not add a new loss, do not
  touch the PI0.5 action target, and do not invent labels.
```

Implemented local code change for the next causal run:

```text
src/openpi/picf/core/config.py:
  posterior_slot_identity_std = 0.02
  task_slot_identity_std = 0.02
  posterior_bootstrap_from_observation = True

scripts/picf_core_train.py:
  added CLI/logging for:
    --posterior-slot-identity-std
    --task-slot-identity-std
    --posterior-bootstrap-from-observation / --no-posterior-bootstrap-from-observation

scripts/verify_picf_owm_contract.py:
  now checks that the production contract is not the legacy symmetric posterior
  initialization.
```

Next causal test:

```text
Use A5 and A7 after the current A7 flow branch finishes or is stopped:

A5:
  posterior_identity_std = 0.02
  posterior_bootstrap_from_observation = true
  anchor/cotrain smoke, unroll=2, burnin=1, action scale 0.25
  purpose:
    test whether symmetry breaking alone prevents same-pixel posterior
    co-location in the fast diagnostic window.

A7:
  same object-file birth prior, production-like all-scope cotrain
  purpose:
    test whether the fix survives action/PaliGemma cotrain pressure.

Acceptance after 100/200/400:
  posterior role-1 pairwise pixel min distance must be nonzero for active slots;
  posterior_recycle_gate_std must no longer be near 1e-4 globally;
  posterior_recycle_rate must avoid both near-0 and near-1 saturation;
  aqr_same_role_support_overlap_max must not rebound into 0.98+;
  action_default_equiv must remain comparable to the current A7 flow row.
```

Deployment script:

```text
scripts/run_picf_posterior_birth_matrix.sh

A5 sequence:
  picf_a5_birth_anchor_u2b1_a025_450_1e5af2c
    anchor_only, unroll=2, burnin=1, action scale=0.25
  picf_a5_birth_cotrain_u2b1_a025_450_1e5af2c
    all-scope cotrain, unroll=2, burnin=1, action scale=0.25
  picf_a5_birth_cotrain_u2b1_a05_450_1e5af2c
    all-scope cotrain, unroll=2, burnin=1, action scale=0.50

A7 sequence:
  picf_a7_birth_cotrain_u2b1_a025_600_1e5af2c
    all-scope cotrain, unroll=2, burnin=1, action scale=0.25
  picf_a7_birth_cotrain_u2b1_a05_600_1e5af2c
    all-scope cotrain, unroll=2, burnin=1, action scale=0.50

Why these rows:
  A5 first isolates whether posterior object-file birth alone fixes posterior
  co-location without full action pressure.
  A5 second and A7 first test production-like cotrain at the action scale that
  improved recycle but previously still collapsed.
  The 0.50 action-scale rows test whether action pressure can be increased
  after identity birth without immediately forcing same-state coalescence.
```

## 2026-05-14 Slot-Local Recycle/Reset Residual Fix

### One-Hour Posterior-Birth Matrix Audit

The requested one-hour audit was run after the posterior object-file birth
defaults were deployed. Both remote matrices were live and GPU-active, so the
result is a model/dataflow signal rather than a runtime failure.

```text
A5 current branch:
  picf_a5_birth_anchor_u2b1_a025_450_1e5af2c
  step 400:
    aqr_same_role_support_overlap_max        = 0.9994
    aqr_active_same_role_support_overlap_max = 0.4435
    aqr_active_anchor_count                  = 8.83
    posterior_recycle_rate                   = 0.0181
    posterior_recycle_gate_std               = 0.000209
    posterior_identity_switch_rate           = 0.7783
    loss_action_default_equiv                = 0.1374

  overlay result:
    seven role-1 posterior slots are exactly co-located at pixel
    [116.79, 95.52].

A7 current branch:
  picf_a7_birth_cotrain_u2b1_a025_600_1e5af2c
  step 150:
    aqr_same_role_support_overlap_max        = 0.2685
    aqr_active_same_role_support_overlap_max = 0.2401
    aqr_active_anchor_count                  = 20.0
    posterior_recycle_rate                   = 0.8385
    posterior_recycle_gate_std               = 0.000115
    posterior_address_update_rate_mean       = 0.00696
    loss_action_default_equiv                = 0.1764

  overlay result:
    graph role-1 anchors are still spatially spread, but seven role-1
    posterior slots are exactly co-located at pixel [67.65, 83.66].
```

### Interpretation

The posterior-birth prior is useful but insufficient. It breaks the initial
object-file symmetry, yet the posterior correction can still erase that
separation because the recycle/reset path used a single global dustbin residual
for all slots:

```math
r_t = \sum_i d_i o_i
```

and then reset every recycled object file from the same vector:

```math
(\bar h_j,\bar c_j,\bar \mu_j,\bar \Sigma_j)
=
(1-\rho_j)(h_j^-,c_j^-,\mu_j^-,\Sigma_j^-)
+\rho_j F(r_t).
```

When several same-role slots recycle in the same window, this update is
permutation-equivariant and can map distinct object files into the same latent
and physical state. This matches the overlays: graph-stage candidates still
exist, but posterior object files collapse to identical pixel coordinates,
support mass, recycle gate, confidence, and address update.

### Maintained Fix

The maintained repair is slot-local recycle/reset residuals. Each slot now
computes the raw measurement mixture implied by its own pre-recycle binding row:

```math
r_{t,j}
=
\frac{\sum_i b^{raw}_{j,i} o_i}
       {\max(\sum_i b^{raw}_{j,i}, \epsilon)}.
```

Only slots with no support fall back to the global dustbin residual. Recycle
trust still uses the normalized slot-local residual direction, and residual
state heads now produce per-slot reset states:

```math
\rho_j = \sigma(G(h_j^-, mass_j, var_j, norm(r_{t,j}), \alpha_j^-))
```

```math
(\bar h_j,\bar c_j,\bar \mu_j,\bar \Sigma_j)
=
(1-\rho_j)(h_j^-,c_j^-,\mu_j^-,\Sigma_j^-)
+\rho_j F(r_{t,j}).
```

This is the belief-filter-consistent form: recycle is an object-file trust
decision about that object's current measurement, not a shared scene-level
reset. It does not add a new loss, does not introduce random jitter, does not
make cache authoritative, and does not change the PI0.5 action objective.

### Code Contract

```text
src/openpi/picf/core/config.py:
  posterior_slotwise_recycle_residual = True

src/openpi/picf/core/pipeline.py:
  support_raw / support_mass_raw -> slot_residual_summary
  recycle_head receives normalized slot_residual_summary
  residual_mu/logvar/h/c heads receive slot_residual_summary

scripts/picf_core_train.py:
  --posterior-slotwise-recycle-residual
  startup log includes posterior_slotwise_recycle_residual

scripts/run_picf_posterior_birth_matrix.sh:
  run names now include the current git short SHA instead of the old 1e5af2c
  suffix, so repeated causal matrices do not collide.
```

### Next Causal Matrix

Stop the old posterior-birth matrix after this commit is synced, because it has
already served its purpose. Restart the same A5/A7 matrix with the new
slot-local recycle residual enabled. This is the clean causal comparison:

```text
A5:
  anchor-only, unroll=2, burnin=1, action scale=0.25
  then all-scope cotrain rows at action scale 0.25 and 0.50

A7:
  production-like all-scope cotrain rows at action scale 0.25 and 0.50
```

Acceptance gates at step 50/100/200:

```text
posterior role-1 pairwise pixel distance is not exactly zero for active slots;
posterior_recycle_gate_std is not near 1e-4;
posterior_recycle_rate avoids both near-0 and near-1 saturation;
aqr_same_role_support_overlap_max does not rebound into 0.98+;
loss_action_default_equiv remains comparable to the prior A7 flow baseline.
```

## 2026-05-14 Slot-Local Residual Runtime Gate

Remote worktrees were synced to `e4f0b91` and the old posterior-birth matrix was
stopped. The new matrix is running in:

```text
A5:
  /root/openpi_recyclenorm_4ec25ae
  tmux: picf_a5_slotlocal_recycle_matrix
  first run: picf_a5_birth_anchor_u2b1_a025_450_e4f0b91

A7:
  /root/openpi_a7_overlay_unroll2_b9ad838
  tmux: picf_a7_slotlocal_recycle_matrix
  first run: picf_a7_birth_cotrain_u2b1_a025_600_e4f0b91
```

Local and remote validation before launch:

```text
python -m py_compile config/pipeline/trainer/verifier: PASS
python scripts/verify_picf_owm_contract.py: PASS
python scripts/picf_owm_strict_diagnose.py --fail-on-fail: PASS
python scripts/picf_owm_dataflow_trace.py --fail-on-fail: PASS
python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
uv run --no-sync pytest -q src/openpi/picf/core/pipeline_test.py -k "...": 3 passed
uv run --no-sync pytest -q src/openpi/picf/core/training_test.py -k "...": 6 passed
```

Early metrics:

```text
A5 step50:
  aqr_same_role_support_overlap_max        = 0.2578
  aqr_active_same_role_support_overlap_max = 0.2392
  aqr_active_anchor_count                  = 20.0
  posterior_recycle_rate                   = 0.5338
  posterior_recycle_gate_std               = 0.00523
  posterior_address_update_rate_mean       = 0.0212
  grad_norm                                = 16.06

A5 step100:
  aqr_same_role_support_overlap_max        = 0.2861
  aqr_active_same_role_support_overlap_max = 0.2309
  aqr_active_anchor_count                  = 20.0
  posterior_recycle_rate                   = 0.3305
  posterior_recycle_gate_std               = 0.00984
  posterior_address_update_rate_mean       = 0.0296
  grad_norm                                = 6.27

A7 step50:
  aqr_same_role_support_overlap_max        = 0.2156
  aqr_active_same_role_support_overlap_max = 0.1964
  aqr_active_anchor_count                  = 20.0
  posterior_recycle_rate                   = 0.4812
  posterior_recycle_gate_std               = 0.0173
  posterior_address_update_rate_mean       = 0.0236
  loss_action_default_equiv                = 0.2621
  grad_norm                                = 2.47
```

These numbers confirm the slot-local residual repair changes the failed
recycle regime: recycle is neither pinned near 0 nor 1, and gate variance is no
longer at the old `1e-4` symmetry level. However, the A5 step100 overlay still
rejects this as a complete posterior identity fix:

```text
A5 step100 overlay:
  graph role-1 pairwise pixel distance:
    min=2.58 px, mean=17.54 px, max=37.37 px
  posterior role-1 pairwise pixel distance:
    min=0.0009 px, mean=0.0928 px, max=0.2828 px
```

Interpretation:

```text
1. Slot-local recycle/reset residuals fix the recycle-gate symmetry and are a
   correct maintained improvement.
2. They do not by themselves fix posterior physical co-location. The graph
   stage still contains separated candidates, while posterior correction maps
   seven same-role object files to effectively the same pixel.
3. The remaining root cause is not only reset-state generation. It is the
   posterior assignment/correction occupancy problem: multiple active same-role
   object files can still use nearly the same observation mode.
4. The next architecture change should be posterior-stage same-role occupancy
   control or set-assignment regularization inside the belief update, not an
   outer action-side or support-diversity patch.
```

Do not call `e4f0b91` accepted. Continue the A7 run to step100 overlay to check
whether the same posterior co-location appears under all-scope cotrain; the A5
anchor-only overlay already shows the current fix is incomplete.

2026-05-14 posterior occupancy prior update: the slot-local recycle run exposed
the next lower-level failure. A5 step50 and step100 had healthy AQR support
overlap (`0.2578 -> 0.2861`) and non-saturated recycle (`0.5338 -> 0.3305`),
but the step100 overlay showed graph role-1 candidates were separated while
posterior role-1 object files were almost exactly co-located:

```text
A5 step100:
  graph role-1 pairwise pixel distance:
    min=2.58 px, mean=17.54 px, max=37.37 px
  posterior role-1 pairwise pixel distance:
    min=0.0009 px, mean=0.0928 px, max=0.2828 px

A5 step200:
  graph role-1 pairwise pixel distance:
    min=0.38 px, mean=14.38 px, max=28.45 px
  posterior role-1 pairwise pixel distance:
    min=0.0004 px, mean=0.0052 px, max=0.0117 px
  aqr_same_role_support_overlap_max = 0.9382
  aqr_active_same_role_support_overlap_max = 0.6215
```

This falsifies the hypothesis that the remaining failure is only recycle/reset
symmetry. The posterior association itself lacks an object-file occupancy prior:
the current Sinkhorn has an observation dustbin row but no per-slot missed
detection/coverage prior, so each same-role posterior row is forced to take
measurement mass. When logits are not already identity-separated, each row uses
a similar broad observation mixture and the posterior position becomes the same
role-level centroid:

```math
B = Sinkhorn(L),\qquad
x_j^{obs} =
  { \sum_i B_{j,i} x_i \over \sum_i B_{j,i} }.
```

The maintained repair is now a label-free posterior occupancy binding prior:
for each role, current observation-anchor hypotheses are farthest-point sampled
and assigned to same-role object-file rows as a measurement coverage prior. The
prior is centered per row and clipped, so it breaks same-role symmetry without
becoming a hard label:

```math
\Delta L_{j,i}^{occ}
=
\lambda_{occ}
\operatorname{clip}
\left(
- { \|x_i - c_j^{fps}\|^2 \over 2\sigma_{occ}^2}
- mean_i(\cdot),
-c_{max}, c_{max}
\right).
```

This is not an extra loss and not an action-side patch. It is part of the
posterior measurement model, matching the object-file interpretation of the
belief filter: same-role files need separate measurement hypotheses before the
precision-form correction step. Defaults:

```text
posterior_occupancy_prior_enabled = True
posterior_occupancy_prior_weight = 1.0
posterior_occupancy_prior_sigma_m = 0.04
posterior_occupancy_prior_clip = 4.0
```

Local validation before remote redeploy:

```text
python -m py_compile config/pipeline/trainer/verifier/deep_audit: PASS
uv run --no-sync pytest -q pipeline_test.py -k "posterior_occupancy ...": 4 passed
python scripts/verify_picf_owm_contract.py: PASS
python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
```

Acceptance gate for the next A5/A7 run is stricter than loss decrease:

```text
1. posterior role-1 pairwise pixel mean should no longer be near zero at
   step100/200 overlays.
2. aqr_same_role_support_overlap_max should not rebound to 0.95+.
3. posterior_recycle_rate should not pin to 0 or 1.
4. action default-equivalent loss can be interpreted only after these structural
   gates pass.
```

2026-05-14 `1dceaef` first-run observation: A5 step100 did not pass the overlay
gate. It improved the old posterior collapse but still showed near-coincident
role-1 posterior files:

```text
1dceaef A5 step100:
  aqr_same_role_support_overlap_max = 0.4972
  aqr_active_same_role_support_overlap_max = 0.3589
  posterior_recycle_rate = 0.5678
  graph role-1 pairwise pixel distance:
    min=0.65 px, mean=6.54 px, max=12.76 px
  posterior role-1 pairwise pixel distance:
    min=0.07 px, mean=0.68 px, max=1.21 px
```

The posterior occupancy prior is therefore necessary but insufficient. The
failure is now localized one step earlier: observation-anchor construction can
still let every same-role candidate reread a broad, nearly identical point-cloud
mixture before posterior data association sees it. The corrected measurement
model keeps observation anchors as coverage hypotheses by retaining a seed-point
component:

```math
w_j^{obs}
\leftarrow
(1-\lambda_{seed})w_j^{reader}
+
\lambda_{seed} e_{seed(j)},\qquad
\lambda_{seed}=0.35.
```

This is not a new loss and does not rewrite posterior truth. It makes the
measurement hypotheses match the object-file model: AQR/support reading may
refine each observation anchor, but it should not erase the initial per-role FPS
coverage before posterior assignment. The new A5/A7 candidate must improve both
the graph and posterior role-1 pairwise distances at step100/200.

2026-05-14 `07bdf66` seed-coverage runtime gate: the A5/A7 matrix was relaunched
from the maintained `07bdf66` worktree. Startup contracts match the intended
profile on both hosts:

```text
unroll_steps = 2
burnin_steps = 1
local_refinement_enabled = False
posterior_occupancy_prior_enabled = True
observation_anchor_seed_point_mix = 0.35
lambda_slot_jepa/support_pred/binding_consistency/aqr_denoising = 0
```

A5 anchor-only early metrics:

```text
step50:
  raw same-role overlap     = 0.2532
  active same-role overlap  = 0.2293
  effective anchor count    = 19.76
  posterior recycle rate    = 0.2474
  action default-equiv loss = 0.1377

step100:
  raw same-role overlap     = 0.3832
  active same-role overlap  = 0.3012
  effective anchor count    = 19.70
  posterior recycle rate    = 0.0719
  action default-equiv loss = 0.1416

step150:
  raw same-role overlap     = 0.7340
  active same-role overlap  = 0.4731
  effective anchor count    = 19.42
  posterior recycle rate    = 0.0212
  action default-equiv loss = 0.1416

step200:
  raw same-role overlap     = 0.9523
  active same-role overlap  = 0.6029
  effective anchor count    = 13.32
  posterior recycle rate    = 0.1445
  action default-equiv loss = 0.1397
```

Overlay audit:

```text
A5 step100:
  graph role-1 pairwise pixel distance:
    min=0.51 px, mean=15.90 px, max=28.22 px
  posterior role-1 pairwise pixel distance:
    min=1.17 px, mean=25.59 px, max=41.08 px

A5 step200:
  graph role-1 pairwise pixel distance:
    min=0.22 px, mean=9.15 px, max=20.47 px
  posterior role-1 pairwise pixel distance:
    min=0.31 px, mean=18.38 px, max=35.19 px
```

Interpretation:

```text
1. The previous posterior exact co-location failure is materially fixed in this
   branch. The role-1 posterior files no longer collapse to near-zero pairwise
   pixel distance at step100/200.
2. This does not yet accept the anchor-only line. Raw support overlap rebounds
   to the old collapse band by step200 and active overlap is just above the
   provisional 0.60 gate, while effective anchor count drops.
3. The failure mode has shifted from physical posterior co-location to support
   reuse / active candidate demotion under anchor-only pressure. This is why A5
   is a diagnostic isolation line, not the production acceptance line.
4. The decisive production branch is A7 cotrain. A7 step50 is structurally
   healthy on overlap (`0.2305` raw, `0.2133` active) but has high recycle
   (`0.6163`) before the first overlay. Wait for A7 step100 and step200 before
   making the next code change.
```

Do not patch again before A7 reaches step100 unless the remote run crashes. The
current evidence says the measurement construction/correction repair is useful
but incomplete under anchor-only isolation; the all-scope cotrain branch is the
intended discriminator for whether task/semantic/action pressure stabilizes or
destroys that separation.

2026-05-14 A7 step100 discriminator:

```text
A7 step100:
  loss_total                 = 0.0979
  loss_action_default_equiv  = 0.0829
  raw same-role overlap      = 0.6559
  active same-role overlap   = 0.4281
  effective anchor count     = 19.42
  posterior recycle rate     = 0.3463
  posterior identity switch  = 0.5317
  posterior address update   = 0.0260
  grad norm                  = 1.43
```

Overlay audit:

```text
A7 step100:
  graph role-1 pairwise pixel distance:
    min=1.46 px, mean=5.95 px, max=12.01 px
  posterior role-1 pairwise pixel distance:
    min=0.69 px, mean=22.21 px, max=37.70 px
```

Manual overlay reading:

```text
A5 step200:
  The visible anchors are no longer mathematically co-located, but the graph
  candidates are crowded around the drawer/table-edge neighborhood. This matches
  the metric split: posterior role-1 distance is repaired, while same-role
  support reuse is still high.

A7 step100:
  The anchors are concentrated around the task-relevant object/handle region,
  but they are not the old exact same-pixel collapse. This is consistent with
  high effective anchor count and posterior role-1 mean distance > 20px.
```

Interpretation:

```text
1. A7 is the production-relevant branch and is healthier than A5 anchor-only.
   It keeps high effective anchor count, nonzero address updates, non-saturated
   recycle, and posterior physical separation while action loss improves.
2. A7 is not accepted yet. Raw overlap is above the preferred 0.60 early gate,
   and graph role-1 candidates are only moderately separated even though the
   posterior is not co-located.
3. This falsifies the strongest "action always destroys anchors" hypothesis:
   under the current lower action weight / cotrain profile, action and semantic
   pressure do not immediately recreate the old exact co-location failure.
4. This also falsifies "anchor-only warmup is enough": A5 anchor-only is worse
   by step250. The better path is production-like cotrain with guarded action
   pressure, not isolated anchor-only optimization.
```

Next gate:

```text
Continue A7 to step200.
If step200 keeps active overlap < 0.60, effective count near 19, posterior
role-1 mean pixel distance > 10px, and recycle in the non-saturated band, treat
the seed-coverage/occupancy candidate as the first viable production candidate.
If A7 rebounds into raw/active overlap collapse by step200, the remaining issue
is not unroll, action presence, or posterior exact co-location; it is same-role
measurement competition before posterior correction.
```

## 2026-05-14 Same-Role Support Competition Repair

The A5/A7 split localizes the current failure more precisely than the older
"all anchors collapse to one point" diagnosis.

Observed failure:

```text
A5 anchor-only:
  posterior exact co-location is fixed by seed coverage + occupancy prior.
  raw same-role support overlap still rebounds toward 0.95-0.999.
  effective anchor count falls as redundant same-role supports are demoted.

A7 cotrain:
  action/semantic pressure does not instantly recreate exact co-location.
  raw same-role support is still above the preferred early gate.
```

Mathematical interpretation:

```text
The posterior correction can now separate physical object files once it receives
distinct measurements. The remaining error is earlier: same-role AQR support
rows may keep the same evidence distribution, so the measurement model provides
duplicate evidence to multiple object files.
```

The maintained fix is role-local support competition:

```math
E_{j,n}^{0}=P_{j,n}
```

For each physical same-role group \(G_r\):

```math
E_{j,n}^{k+1}
=
\operatorname{Normalize}_{n}
\left(
  \frac{E_{j,n}^{k}}
       {\epsilon+\sum_{\ell\in G_r}E_{\ell,n}^{k}}
\right)
```

Then:

```math
P'_{j,n}
=
\operatorname{Normalize}_{n}
\left((1-\lambda)P_{j,n}+\lambda E_{j,n}^{K}\right)
```

Default:

```text
aqr_same_role_support_competition_enabled = true
aqr_same_role_support_competition_weight = 0.35
aqr_same_role_support_competition_iters = 2
aqr_same_role_support_competition_physical_only = true
```

Why this is not a late patch:

```text
1. It lives inside measurement routing, before posterior correction.
2. It is not an auxiliary loss, so it cannot fight action/cotrain gradients.
3. It is role-local and physical-only, so task queries may still intentionally
   share evidence.
4. It cannot create evidence. If two rows are exactly identical, the update is
   unchanged; it only amplifies weak object-specific differences already
   supplied by ownership priors, seed-point coverage, geometry, multiview
   temporal tokens, or the binding-signature subspace.
```

Paper alignment:

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
  The paper motivates pairwise IsSameObject relations and low-dimensional
  binding subspaces that guide attention. This supports putting the repair in
  support routing rather than in action loss.

SlotVLA / STORM / object-centric world-model papers:
  These support persistent object slots as a belief-state interface and staged
  object-centric adaptation. The repair preserves posterior authority and does
  not add a new supervised object label requirement.
```

Validation completed locally:

```text
py_compile: config/pipeline/pipeline_test/trainer/verifier/deep_audit passed
verify_picf_owm_contract.py: PASS
picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
picf_owm_dataflow_trace.py --fail-on-fail: PASS
picf_owm_strict_diagnose.py --fail-on-fail: PASS
pytest:
  pipeline_test.py::test_aqr_same_role_support_competition_amplifies_relative_evidence
  scripts verifier/evidence-bundle tests
  5 passed
```

Remote test plan:

```text
A5 is the right first test because it is the negative control that still
rebounds under anchor-only pressure. Restart A5 with the same anchor-only
configuration plus same-role support competition. Acceptance is not action
loss; acceptance is:

  raw same-role support overlap stays below the old 0.95 collapse band,
  active overlap stays below 0.60,
  effective anchor count does not fall toward 8-13,
  posterior pairwise pixel distance stays above 10px,
  anchor overlays show object-file spread rather than same-region reuse.

If A5 still collapses, the next root cause is that the weak row-specific
evidence is not present in the support rows and must be moved earlier into
token/probe extraction, not solved by stronger losses.
```

First A5 support-competition gate:

```text
Run:
  picf_a5_birth_anchor_u2b1_a025_450_24c6cf7_support_comp

step50:
  raw same-role support overlap    = 0.2179
  active same-role support overlap = 0.2035
  effective anchor count           = 19.49
  posterior recycle rate           = 0.2734
  stable slot fraction             = 0.3128

step100:
  raw same-role support overlap    = 0.6874
  active same-role support overlap = 0.5120
  effective anchor count           = 18.18
  posterior recycle rate           = 0.2811
  stable slot fraction             = 0.3594
```

Step100 overlay:

```text
/mnt/checkpoints/picf_core/picf_core/picf_a5_birth_anchor_u2b1_a025_450_24c6cf7_support_comp/anchor_overlays/step_000100.png

graph role-1 pairwise pixel distance:
  min=2.41 px, mean=23.24 px, max=48.44 px
graph role-2 pairwise pixel distance:
  min=1.36 px, mean=24.28 px, max=50.57 px
graph role-3 pairwise pixel distance:
  min=3.48 px, mean=23.62 px, max=44.59 px
posterior role-1 pairwise pixel distance:
  min=0.60 px, mean=21.70 px, max=36.41 px
```

Interim interpretation:

```text
This is the first A5 anchor-only branch that directly targets the remaining
support-reuse failure without adding a new loss. It has not solved the problem
yet at step100 because raw overlap is above the preferred 0.60 gate, but it has
not reproduced the old A5 rebound into 0.95+ overlap either. Active overlap is
below the gate, effective anchor count is still high, and graph/posterior
overlay distances are materially larger than the previous A5 step100/200
candidate. Continue to step200 before deciding whether the role-local
competition is sufficient or whether the support rows still lack enough
row-specific evidence upstream.
```

Step300 follow-up:

```text
Run:
  picf_a5_birth_anchor_u2b1_a025_450_24c6cf7_support_comp

step200:
  raw same-role support overlap    = 0.9963
  active same-role support overlap = 0.5776
  effective anchor count           = 9.05
  posterior recycle rate           = 0.0015
  stable slot fraction             = 0.3811

step300:
  raw same-role support overlap    = 0.9999
  active same-role support overlap = 0.1867
  effective anchor count           = 4.63
  posterior recycle rate           = 0.0537
  stable slot fraction             = 0.3506
```

Interpretation:

```text
Reject this as a standalone anchor-only fix. Same-role support competition is
not a wrong mathematical operation; it is a correct measurement-routing
competition and it clearly improves the first 50-100 steps. The failure is that
anchor-only pressure does not force all same-role object files to stay useful.
Once the active-slot filter demotes duplicates, the raw support rows may
collapse again while the active overlap metric looks superficially healthier.
The low recycle rate rules out the older recycle-saturation diagnosis.

The next iteration is therefore not another action-side penalty or stronger
support-diversity loss. It is a production-like cotrain test under the same
routing repair. Acceptance depends on whether task/action/semantic gradients
reward object files enough to keep effective anchor count healthy. If cotrain
also collapses, then the support rows lack sufficient object-specific evidence
upstream and the next fix must target token/probe extraction rather than the
loss surface.
```

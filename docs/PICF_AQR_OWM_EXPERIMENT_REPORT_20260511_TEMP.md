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
5. The A7 long run is the current acceptance run; only its future checkpoints
   can decide behavior-level readiness.
```

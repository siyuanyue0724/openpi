# PICF-AQR-OWM Open Issue Tracker - 2026-05-17

This file is the working issue ledger for the current PICF-AQR-OWM/MVTrack
cleanup.  It separates solved engineering bugs from remaining scientific or
data-limited questions.  A problem is not closed here unless code/dataflow,
math contract, local scripts, and at least one runtime metric agree.

## Status Legend

```text
closed:
  Code and runtime evidence agree; only normal long-run behavior validation
  remains.

open-code:
  A concrete engineering/dataflow change is still needed.

open-train:
  Code path is present, but only a long run / behavior eval can decide.

open-data:
  Cannot be fully solved without new labels, sidecars, calibration, or dataset
  fields.

watch:
  Not a blocker, but must stay in diagnostics because it can regress.
```

## 2026-05-18 Concentrated Current Issue Summary

2026-05-19 blind-SAM disposition:

```text
Blind automatic SAM is closed as rejected / archived.
Do not use it for current training or diagnostics.
The only retained interface is generic proposal_* sidecar evidence from
inspected contact/task/tracklet-aware sources.
Legacy reproduction requires archived scripts plus
--allow-legacy-blind-sam-sidecar.
```

2026-05-20 clean sidecar disposition:

```text
Old SAM-seeded, smoke-only, and partial diagnostic sidecar roots are rejected
for production validation.

The accepted next-run data contract is:
  /mnt/picf_sidecars/contact_motion_full_20260519
    proposal + sparse mask source root

  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
    clean proposal + sparse mask + KLT tracklet root

The clean root is generated from an empty output directory and must pass
scripts/picf_prepare_full_sidecar_root.py with:
  --require-proposal --require-mask --require-tracklet

Status at 2026-05-20 22:25 CST:
  generation in progress, 2800 / 7545 progress_segments, no GPU use,
  ETA roughly 3.0-4.0 hours.
```

This section centralizes the current A7-facing issue list.  Earlier versions of
this tracker did record the same symptoms, but they were spread across I5,
I11, I12, I13 and the long experiment report.  The operator-facing summary is:

```text
Keep A7 for model-side training/diagnostics.
Use the clean A7 full sidecar generation as the only production data gate.
Old A5 partial/samseed/diagnostic sidecar runs remain historical evidence only.
Do not judge the model by action loss alone.
The main remaining risk is long-run rebound of object-binding health metrics
after an initially healthy phase.
```

### Current P0 Watch Metrics

The most important live metrics are the ones that improve early and then
rebound:

```text
1. loss_anchor_pv
   Meaning:
     object-routed point/visual alignment pressure.
   Bad pattern:
     rises while loss_pv_weak falls.
   Interpretation:
     dense visual/point correspondence can improve while object-owner routing
     gets worse.

2. aqr_same_role_support_overlap_max
   Meaning:
     raw same-role support duplication, including reserve/context capacity.
   Bad pattern:
     returns toward 0.95-1.00.
   Interpretation:
     not an automatic production failure after active/context/reserve routing,
     but it shows reserve/candidate rows still share the same evidence.

3. aqr_active_same_role_support_overlap_max
   Meaning:
     action-visible same-role support duplication.
   Bad pattern:
     sustained rise above the low early range.
   Interpretation:
     this is a real production health signal; active object files are starting
     to duplicate owners.

4. aqr_same_role_object_core_overlap_max /
   aqr_active_same_role_object_core_overlap_max
   Meaning:
     object-core duplicate evidence, stricter than raw support overlap.
   Bad pattern:
     rises together with active support overlap.
   Interpretation:
     stronger evidence of true object-owner duplication.

5. posterior_identity_switch_rate,
   posterior_active_file_potential_swap_rate,
   posterior_active_file_calibrated_potential_swap_rate,
   posterior_recycle_rate
   Meaning:
     posterior lifecycle stability.
   Bad pattern:
     high calibrated swap or recycle rising after initial stability.
   Interpretation:
     action can keep improving while object-file identity is unstable.  Raw
     swap alone is no longer enough: a high raw swap with zero calibrated
     dispersion is a common-mode diagnostic artifact.

6. loss_aqr_denoising / loss_mapg_routing
   Meaning:
     anchor/readout structural health indicators.
   Bad pattern:
     rises when action falls.
   Interpretation:
     action loss is hiding support-routing degradation.
```

### Attempts Already Made And Current Verdicts

These records are important because several intuitive fixes were tested and are
not sufficient by themselves.

```text
1. More support diversity / same-role competition
   Tried:
     stronger support competition weights and iterations.
   Result:
     often lowers active overlap early, but does not reliably prevent later
     rebound once same-role rows become feature-similar.
   Verdict:
     keep as a guard, not a complete root fix.

2. Action-off / anchor-only diagnostics
   Tried:
     detached/no-action structural runs.
   Result:
     same-role overlap still rebounded in Phase A.
   Verdict:
     action gradients are not the only cause.

3. Action-on strict capacity diagnostics
   Tried:
     active_slot_max_per_role, active_slot_overlap_threshold,
     active_slot_relative_score_threshold, stronger same-role competition.
   Result:
     active overlap improved early, but lifecycle/recycle/swap remained a
     watch item and later diagnostics still required additional repairs.
   Verdict:
     capacity helps, but does not close the issue alone.

4. Posterior file competition
   Tried:
     same-role support/geometry duplicate demotion after Sinkhorn.
   Result:
     fixed the false "all orange files are active" failure; overlays now show
     gray reserve files separately from action-visible active files.
   Verdict:
     real repair; not enough to guarantee target-object selection.

5. Posterior birth transport
   Tried:
     no-object/birth competition so dustbin residual is not broadcast into
     every inactive file.
   Result:
     fixed the dustbin residual / inactive recycle broadcast issue in the
     tested diagnostic.
   Verdict:
     real repair; remaining overlap is upstream support assignment.

6. Downstream active/context/reserve routing
   Tried:
     active files get full downstream weight, context gets low weight, reserve
     gets zero.
   Result:
     fixed gray reserve leakage into action graph prefix.
   Verdict:
     real repair; inspect active_only overlays for remaining target errors.

7. Legacy local refinement / coverage seed / role-wise local candidate variants
   Tried:
     extra top-k local aggregation and deterministic coverage-style local
     scaffolds.
   Result:
     did not solve same-role local candidate reuse and added graph complexity.
   Verdict:
     archived/off by default.

8. Blind SAM proposal memory
   Tried:
     SAM proposal sidecars and proposal-to-point bridging.
   Result:
     dataflow works, but visual quality is noisy and behavior benefit is not
     proven; boxes can target wall/robot/drawer fragments.
   Verdict:
     production-default off; keep only prompted/contact/reranked ablations.

9. Tracklet sidecar
   Tried:
     generation and a full-sidecar anchor-only diagnostic.
   Result:
     the old SAM-seeded/diagnostic roots proved the dataflow but were not clean
     production evidence.  The maintained root is now regenerated from
     contact-motion proposal/mask evidence into a clean KLT tracklet root, with
     sparse mask keys preserved during merge and final proposal/mask/tracklet
     auditing required before long training.
   Verdict:
     open-train: data generation is in progress; if the final audit passes,
     tracklet/proposal/mask sidecars are accepted for the next 30K validation
     but behavior benefit still requires long-run evidence.

10. Frozen-PaliGemma action diagnostic
    Tried:
      freeze PG and train action-side/PICF only.
    Result:
      action loss improved but anchor structure degraded.
    Verdict:
      not production recipe; keep PaliGemma cotrain for semantic adaptation.

11. Slot-JEPA / support-pred / binding-consistency / AQR-denoising losses
    Tried:
      hooks and diagnostics exist; earlier slot-JEPA-like pressure showed
      unstable behavior when identity assignment was not stable.
    Result:
      not accepted as production losses.
    Verdict:
      keep default zero until identity assignment is stable over long runs.
```

### Current Working Interpretation

The repeated failure pattern is not one simple bug anymore.  Several concrete
bugs were fixed.  The remaining scientific/optimization issue is:

```text
The model can learn action quickly, and active duplicate capacity is much
better controlled than before, but AQR support assignment and posterior
lifecycle can still drift toward repeated same-role evidence unless the active
owner selected by task/semantic/contact evidence remains stable.
```

Therefore the next A7 decisions must use this acceptance bundle, not a single
scalar:

```text
loss_action_default_equiv:
  should keep improving relative to 4-22 baselines.

loss_anchor_pv:
  should not show sustained upward drift while action improves.

aqr_active_same_role_support_overlap_max:
  should remain low; this is the active-object duplicate metric.

aqr_active_same_role_object_core_overlap_max:
  should remain low; this is the stricter object-core duplicate metric.

posterior_active_file_calibrated_potential_swap_rate / identity_switch / recycle:
  should not worsen after early warmup. Raw
  posterior_active_file_potential_swap_rate is kept for compatibility, but is
  not decisive without calibrated file-signature dispersion.

active_only overlays:
  active orange files should cover the intended task/contact object region.

with_gray overlays:
  can show reserve capacity, but gray reserve files must not be interpreted as
  action-visible object files.
```

### 2026-05-19 V7 Calibrated Continuity Verdict

The calibrated metric run
`picf_a7_v7_calibrated_file_continuity_anchor200_20260519` completed 200 steps.

```text
Resolved as dominant causes:
  object-PV dense/background mismatch;
  raw same-role overlap as an active-owner failure;
  raw file-swap metric false-positive ambiguity.

Still open:
  posterior object-file continuity.
```

Key evidence:

```text
step 200:
  aqr_active_same_role_support_overlap_max        = 0.1936
  aqr_active_same_role_object_core_overlap_max    = 0.0678
  aqr_same_role_support_overlap_max               = 0.9859
  aqr_same_role_object_core_overlap_max           = 0.8531
  posterior_active_file_calibrated_potential_swap = 0.3157
  posterior_file_calibrated_signature_score_std   = 0.9944
  posterior_binding_signature_calibrated_score_std= 0.0000
```

Interpretation:

```text
Reserve/context overlap remains high, but active object-core separation is
mostly healthy. The remaining production target is that active posterior files
do not preserve stable calibrated identity strongly enough across updates.
```

Resolution plan now implemented locally:

```text
V8b posterior binding-signature memory:
  Treat binding_signature as a persistent posterior file descriptor.
  Low-support / low-owner-confidence measurements keep prior signature.
  Common-mode current measurements are rejected by calibrated pairwise
  dispersion gating.
  Trusted dispersed measurements update by bounded EMA.
  Birth/recycle resets to the instantaneous signature.

New decisive metrics:
  posterior_binding_signature_update_rate_mean
  posterior_binding_signature_measurement_trust_mean
  posterior_binding_signature_memory_keep_rate_mean
  posterior_binding_signature_measurement_score_std
  posterior_binding_signature_measurement_margin_mean
  posterior_binding_signature_measurement_dispersion_gate_mean
  posterior_active_file_calibrated_potential_swap_rate
  posterior_binding_signature_calibrated_score_std
```

2026-05-19 V8b validation update:

```text
Run:
  picf_a7_v8b_binding_signature_dispersion_memory_anchor200_20260519

Closed for this issue:
  Blind per-frame binding_signature overwrite is fixed.
  Active calibrated potential swap stays low:
    step 50  = 0.0261
    step 100 = 0.0245
    step 150 = 0.0056
    step 200 = 0.0085
  All-file calibrated swap improves:
    step 50  = 0.1863
    step 100 = 0.0894
    step 150 = 0.0569
    step 200 = 0.0538
  Measurement dispersion remains high, and update rate falls as memory keep rises.

Still open:
  loss_anchor_pv rises from 0.5777 to 0.6171.
  loss_mapg_routing rises from 0.6025 to 0.7760.
  active same-role support overlap rises from 0.0176 to 0.3777.
  raw same-role support overlap returns to ~0.992 in reserve/context/raw rows.
  posterior_binding_signature_calibrated_score_std becomes 0.0 by step 150.
```

Interpretation:

```text
The V8b fix solves the narrow posterior file-continuity overwrite bug, but it
does not prove complete object-anchor binding.  The remaining failure is now a
PV/routing/signature-discriminability problem rather than a blind memory
overwrite problem.
```

2026-05-19 object-pull-only diagnostic:

```text
Hypothesis:
  Orange/task-object anchors are not landing on the visible proposal object
  because either (A) the query/support trainable path cannot move anchors even
  under direct object supervision, or (B) unrelated objectives pull the anchors
  away after sidecar evidence arrives.

Implemented probe:
  loss_anchor_object_pull

Formula:
  x_target_j = stopgrad(sum_i p_sidecar_ji x_i / sum_i p_sidecar_ji)
  L_pull = sum_j w_j Huber((x_anchor_j - x_target_j) / sigma)

Scope:
  Diagnostic-only; default weight 0.
  Use only confirmed sidecar/proposal/task-owner point evidence.
  Freeze perception, semantic, action, predictive, mapg, pv_weak, tactile/point
  reconstruction and all unrelated losses.

Launcher:
  run_a7_anchor_object_pull_only_1000_20260519.sh

Acceptance:
  loss_anchor_object_pull must drop substantially.
  Step-100/200/1000 overlays must show active physical anchors entering the
  green proposal/mask on the red block task.
```

### vNext Capacity Hypothesis: AR Active-Anchor Proposal

Current repeated evidence suggests that several repaired bugs are closed, but
fixed overcomplete graph capacity can still create repeated same-role candidate
rows before active/context/reserve demotion. The audited vNext hypothesis is:

```text
Generate only the needed active measurement hypotheses, then pad them back to
the existing AQR/posterior tensor contract.
```

The full math/code/literature audit is:

```text
docs/PICF_AQR_OWM_AR_ANCHOR_PROPOSAL_AUDIT_20260518_TEMP.md
```

This is not a current blocker for the running default and not a reason to
remove posterior file competition. It is a candidate root fix if long-run A7
continues to show that fixed active candidate capacity, not dataflow bugs, is
the remaining source of active same-role rebound.

Acceptance requirements before implementation becomes default:

```text
proposal_count must be evidence-dependent;
coverage/unexplained-evidence must not regress;
active same-role support/object-core overlap must fall;
posterior active-file potential swap must not rise;
action loss must not regress;
active overlays must show task objects receiving active posterior files.
```

Immediate rejection conditions:

```text
proposal_count collapses to too few objects;
coverage/unexplained-evidence regresses;
proposal index becomes a persistent id;
dense typed memory is pruned;
posterior file/birth competition is bypassed;
action improves while active task-object overlays lose the named object.
```

Current implementation status:

```text
VCAP is implemented as a disabled-by-default runtime prototype:
  core runtime config + contracts + padded AQR query initialization;
  posterior-subordinate proposal rows, not persistent proposal identities;
  coverage/duplicate/count/continuity transition-loss hooks;
  action-gradient guard during bring-up;
  overlay/debug metrics and verifier coverage.

Still required before promotion:
  short fixed-vs-VCAP diagnostics with action_grad_scale=0;
  then small action_grad_scale only if coverage is preserved and active
  same-role duplication falls;
  no direct 30k production claim until those metrics pass.

Still rejected:
  no stop-token-only patch;
  no count-loss-only patch;
  no dense-memory pruning;
  no proposal-index identity;
  no bypass of posterior file/birth competition.
```

## Archived Completed Repairs

These items are no longer active blockers.  They stay in this file only as
audit history and regression guards.

```text
I1 dense_anytouch_patch_dim:
  status: archived-closed
  closure: tactile_patch_token_proj projects 768-D dense AnyTouch patches into
    hidden_dim before PICF tactile reread.
  guard: py_compile, verify_picf_owm_contract, remote startup.

I2 tactile_soft_evidence:
  status: archived-closed-watch
  closure: tactile evidence is now soft between p_floor and p_on; dense patch
    reread remains gated by p_on.
  guard: tactile_evidence_rate / tactile_evidence_weight / tactile_active_rate.

I3 tactile_zero_topk_false_evidence:
  status: archived-closed
  closure: posterior tactile reread is skipped for slots with zero bound tactile
    routing mass.
  guard: tactile evidence does not appear for every slot when tactile mass is 0.

I4 active_duplicate_metric:
  status: archived-closed-watch
  closure: active duplicate overlap is separated from raw reserve/context
    overlap.
  guard: posterior_file_competition_active_duplicate_overlap_max.

cache_residual_scaling:
  status: archived-closed
  closure: evidence_cache_read_weight scales residual output, not a softmax
    constant shift.
  guard: cache read code remains q + lambda * (Read(q)-q).

duplicate_latest_posterior_cache_row:
  status: archived-closed
  closure: latest previous-posterior row is skipped in cache read so posterior
    reader and cache reader do not mechanically double-count it.

burnin_suffix_graph_consistency:
  status: archived-closed
  closure: state_only burn-in uses the AQR measurement graph when AQR is
    enabled.
```

These repairs should not be reopened unless a metric or script regression shows
the exact failure mode again.

### 2026-05-20 Owner Measurement Transport Update

Run:

```text
picf_a7_context_support_dedup_300_20260520
```

Verdict:

```text
context/downstream duplicate rebound:
  closed for this diagnostic.

task-owner active/posterior localization:
  still open-code before this repair.
```

Evidence:

```text
step 300:
  aqr_active_same_role_support_overlap_max     = 0.049
  aqr_context_same_role_support_overlap_max    = 0.512
  aqr_downstream_same_role_support_overlap_max = 0.628
  loss_object_explanation_point                = 6.09
  grad clipping                                = active
```

Overlay interpretation:

```text
sidecar/mask core exists, but graph active owner is only near the core and
posterior active file can remain farther away. This means sidecar/contact
owner evidence was still too indirect: it entered support priors but was not
first-class owner geometry for active selection and posterior transport.
```

Implemented locally:

```text
object_candidate_owner_x / object_candidate_owner_S:
  accepted owner candidate point-mask geometry.

object_candidate_owner_geometry_mix:
  blends owner measurement geometry into graph anchor geometry before active
  selection.

posterior_owner_transport_candidate_geometry_mix:
  uses accepted owner measurement geometry during posterior precision fusion.

aqr_slot_quality_owner_active_floor:
  prevents QASA-style quality from suppressing an accepted owner candidate as
  early duplicate/no-object noise.
```

Validation doc:

```text
docs/PICF_AQR_OWM_OWNER_MEASUREMENT_TRANSPORT_20260520_TEMP.md
```

Next acceptance:

```text
Run picf_a7_owner_measurement_transport_300_20260520.

Pass condition:
  owner geometry active distance near zero;
  posterior_owner_transport_confidence nonzero;
  active and posterior overlays move to sidecar/mask core;
  active/downstream overlap stays below old failure band;
  loss_object_explanation_point remains bounded without late grad explosion.
```

## Open Work Plan

```text
P0 task_owner_behavior_acceptance:
  related issue: I5
  current state: code-live, open-train
  next evidence: finish 300-step task-owner diagnostic, inspect overlays, rerun
    IsSameObject probe, then decide whether to proceed to long cotrain.

P0 posterior_birth_transport:
  related issue: I11
  current state: patched-local / local audits passed / A7 step100 shows specific repair works
  observed evidence: A7 SAM proposal anchor-only diagnostic reached step300
    with full same-role overlap ~= 1.0, active overlap rising to 0.4216,
    anchor_pv worsening, recycle rising, and slot-JEPA diagnostic drifting.
  implementation plan: split posterior transport into existing-file support
    update and reserve-file birth/no-object competition. Demoted duplicate
    dustbin residual must not be broadcast back into every inactive same-role
    file.
  local verification: py_compile, verify_picf_owm_contract, and
    picf_posterior_birth_transport_audit all pass.

P0 aqr_support_assignment_pressure:
  related issue: I12
  current state: open-analysis
  observed evidence: after posterior birth transport, step100 still shows
    raw support overlap 0.923 and active support overlap 0.189 while birth
    count/recycle are controlled.
  interpretation: remaining degradation is upstream support assignment/routing
    pressure rather than dustbin birth broadcast.

P1 dataset_scale_contact_proposals:
  related issue: I10
  current state: blind SAM is archived/rejected. Contact/task-guided proposal
    sidecars are the maintained proposal source. 2026-05-18 update: proposal
    sidecars now have the complete object-support contract
    `proposal_mask_xy / proposal_mask_weights / proposal_mask_offsets` in
    addition to center/box/objectness, so boxes are only reference fallbacks.
    Dataset-scale coverage is still missing.
  implementation plan: offline contact-motion / task-guided proposal generation
    for available views, saved as proposal_* sidecars with sparse soft masks;
    no online blind-SAM dependency in training.

P1 dataset_scale_tracklets:
  related issue: I6/I10
  current state: code can consume tracklets, but current CALVIN run has
    owm_tracklet_tokens=0.
  implementation plan: offline CoTracker/TAPIR/flow-style tracklets for static
    and wrist videos, with confidence/visibility filtering and safe no-op when
    video continuity is absent.

P1 same_object_probe_repeat:
  related issue: I7
  current state: step150 probe shows decodable binding subspace but duplicate
    candidates.
  implementation plan: rerun at step300 and after sidecar-enabled runs; accept
    only if duplicate_candidate_fraction falls while AUC remains high.

P2 action_cotrain_acceptance:
  related issue: I9
  current state: current run is anchor/action-default diagnostic, not full
    behavior acceptance.
  implementation plan: long cotrain with the current post-blind-SAM-default-off
    training contract:
      freeze V-JEPA/Sonata/AnyTouch pretrained perception encoders;
      train PaliGemma/PI0.5 semantic-action stack plus PICF adapters/heads;
      use unroll_steps=2, burnin_steps=1, burnin_mode=state_only;
      keep action weights at 0.50 and use action-prefix stopgrad;
      keep slot-JEPA/support-pred/binding-consistency/AQR-denoising at 0;
      keep blind SAM memory off; enable proposal memory only for inspected
        contact/task-guided proposal sidecars;
      log and save anchor overlays every 50 steps;
      save checkpoints every 2500 steps and retain the latest 3.

P3 ordinal_rank_grounding:
  related issue: I8
  current state: diagnostic only.
  implementation plan: do not hard-enable until target-object selection and
    selected-slot weak targets are reliable; true closure requires rank labels
    or a validated weak target source.
```

## Current Runtime Evidence

Run:

```text
picf_a7_task_owner_bias_samseg0_diag300_20260517
```

Key step-50 / step-100 / step-150 / step-200 evidence:

```text
loss_total                              0.476377 -> 0.529843 -> 0.505773 -> 0.562951
loss_anchor_pv                          0.737828 -> 1.133378 -> 1.185894 -> 1.187573
loss_pv_weak                            6.056437 -> 4.277687 -> 3.873690 -> 3.306262
loss_aqr_denoising                      1.771745 -> 2.138244 -> 2.006886 -> 1.930972
loss_mapg_cycle                         0.367694 -> 0.410990 -> 0.436437
loss_mapg_routing                       0.677847 -> 0.722459 -> 0.708319 -> 0.733738
loss_mapg_support_diversity             0.306764 -> 0.249809 -> 0.136392
aqr_task_owner_visual_prior_entropy     0.665582 -> 0.415869 -> 0.336349
aqr_task_owner_visual_prior_max         0.034154 -> 0.189592 -> 0.208159
aqr_task_owner_proposal_score_nonzero   0.291520 -> 0.294486 -> 0.210545
aqr_active_same_role_support_max        0.012676 -> 0.046688 -> 0.050520 -> 0.347965
aqr_active_same_role_object_core_max    0.064381 -> 0.173627 -> 0.062917 -> 0.025162
aqr_raw_same_role_support_max           0.157804 -> 0.587146 -> 0.928805 -> 0.999196
aqr_raw_same_role_object_core_max       0.094022 -> 0.507334 -> 0.801448
posterior_active_duplicate_max          0.000000 -> 0.000000 -> 0.000000 -> 0.000000
posterior_identity_switch_rate          0.687778 -> 0.710556 -> 0.738889 -> 0.736667
posterior_recycle_rate                  0.106766 -> 0.122773 -> 0.051220 -> 0.015970
loss_action_default_equiv               0.182817 -> 0.204859 -> 0.177226 -> 0.180460

Offline IsSameObject probe on step-50/100/150 overlays:
  binding_signature_cos_auc             0.943820
  combined_auc                           1.000000
  duplicate_candidate_fraction           0.238095
  decision                               binding_subspace_decodable_but_assignment_duplicates_candidates
```

Interpretation:

```text
1. Task-owner proposal routing is alive: proposal score nonzero fraction is
   stable around 0.29 and task visual prior sharpens from 0.034 to 0.190.
2. Active duplicate posterior files remain controlled.
3. Active same-role overlap is still low relative to earlier collapse runs,
   but it rises from step 50 to step 100 and must be watched.
4. Raw/reserve overlap rises substantially.  This is not the acceptance metric
   by itself, but it indicates reserve/context capacity is still not fully
   decorrelated.
5. PV weak improves monotonically, and active object-core overlap remains low at
   step200.  However anchor-PV, raw/reserve overlap, active support overlap, and
   identity switch are still not accepted.  Therefore the new task-owner
   dataflow is code-live, but target-object behavior is not yet accepted.
6. The offline IsSameObject probe shows a decodable binding subspace, but it
   also reports duplicate candidates.  The issue is therefore no longer
   "no binding signal"; it is "assignment/lifecycle still leaves duplicate
   candidate owners under the current short diagnostic".
7. Identity switch remains high.  This remains an open training/behavior
   question, not a closed scientific result.
```

## Issue Ledger

### I1. Dense AnyTouch patch dimensional mismatch

Status:

```text
closed
```

Problem:

```text
Raw dense AnyTouch patch tokens were 768-D, while PICF tactile reread modules
expect hidden_dim=512.  The old high hard-contact gate hid the bug because dense
patches rarely activated.
```

Repair:

```text
PicfFullCore.tactile_patch_token_proj projects dense AnyTouch patch tokens to
hidden_dim before tactile_route_reread / tactile_native_reread.
```

Validation:

```text
py_compile passes.
verify_picf_owm_contract passes.
Remote startup no longer fails before first forward.
```

### I2. Tactile all-or-nothing evidence

Status:

```text
closed / watch
```

Math:

```math
e_t = 1[p_t \ge p_{floor}]
```

```math
w_t =
clip((p_t-p_{floor})/(p_{on}-p_{floor}+\epsilon), 0, 1)
```

Repair:

```text
Soft tactile evidence enters as weighted sensor-level evidence when
p_floor <= p_t < p_on.  Dense AnyTouch patch reread opens only when p_t >= p_on.
```

Runtime:

```text
tactile_contact_prob_mean ~= 0.3606
tactile_evidence_rate     = 0.5
tactile_evidence_weight   ~= 0.0953
tactile_active_rate       = 0.0
```

Conclusion:

```text
This is correct for the current calibration: tactile is evidence but not a hard
contact label.  The hard gate being zero is a warning, not a failure.
```

### I3. False tactile evidence through all-zero top-k

Status:

```text
closed
```

Problem:

```text
If a posterior slot had zero tactile routing mass, top-k over all-zero tactile
weights could still select arbitrary tactile groups and create fake tactile
evidence.
```

Repair:

```text
_posterior_update now runs tactile_native_reread only for slots with nonzero
bound tactile routing mass.  Other slots keep tactile_evidence=0.
```

### I4. Raw duplicate overlap confused with active object-file collapse

Status:

```text
closed / watch
```

Math:

```math
D_{raw} =
max_{i \ne j, r_i=r_j} overlap(i,j)
```

```math
D_{active} =
max_{i \ne j, r_i=r_j, a_i=1, a_j=1} overlap(i,j)
```

Runtime:

```text
posterior_file_competition_active_duplicate_overlap_max = 0.0
posterior_file_competition_duplicate_overlap_max        ~= 0.996-1.000
```

Conclusion:

```text
Raw duplicate overlap includes reserve/no-object capacity.  Active duplicate
overlap is the acceptance metric.
```

### I5. Task-conditioned target-object selection

Status:

```text
closed-code / open-train / watch
```

Observed user symptom:

```text
For prompts such as "pick up the red block lying in the drawer", many anchors
appear near the interaction region, but the active object anchor may not sit on
the intended red block.  Gray reserve files can dominate visual inspection.
```

Current evidence:

```text
Task-owner proposal route is live:
  aqr_task_owner_proposal_score_nonzero_fraction 0.291520 -> 0.294486
  aqr_task_owner_visual_prior_max                0.034154 -> 0.189592
  aqr_task_owner_visual_prior_entropy            0.665582 -> 0.415869

Active duplicate owner collapse is still controlled:
  posterior_file_competition_active_duplicate_overlap_max = 0.0

But target-object behavior is not yet accepted:
  loss_anchor_pv                  0.737828 -> 1.133378
  loss_aqr_denoising              1.771745 -> 2.138244
  loss_mapg_routing               0.677847 -> 0.722459
  aqr_active_same_role_object_core_overlap_max 0.064381 -> 0.173627
  posterior_identity_switch_rate  0.687778 -> 0.710556

Tracklet identity evidence remains absent in the current run:
  owm_tracklet_tokens = 0
```

Diagnosis:

```text
This is not explained by "only 300 steps are missing" alone.  The new code makes
the task-owner route nonzero and sharper, but step-50 to step-100 still worsens
anchor-PV, denoising, routing, active object-core overlap, and identity switch.
That means the missing edge has been repaired at the dataflow level, but
training behavior still has to prove that the edge is strong and stable enough
to select the intended object.
```

Likely causes:

```text
1. Current SAM proposals are class-agnostic objectness boxes.
2. Current PG image support is first-class but not a hard task mask.
3. PaliGemma is frozen in the current anchor-only diagnostic.
4. Tracklet sidecar evidence is absent, so there is no temporal same-object
   stabilizer.
5. Posterior file competition demotes duplicate owners but does not create a
   semantic "red block" owner by itself.
```

Implemented code repair:

```text
Task query rows now produce a task-owner visual prior from the same AQR visual
support that already reads PaliGemma text/image-conditioned task queries.

That prior is projected onto SAM/proposal boxes through static-view projective
visual grid geometry and proposal objectness:

  task_owner_visual_prior = mean_rows(visual_priors[task_scene_rows])

  proposal_task_score(p)
    = objectness(p)^alpha
      * sum_i task_owner_visual_prior(i) * 1[visual_cell_i in box_p]
        / sqrt(coverage_p + eps)

The centered log of this score is added as a low-amplitude measurement bias for
physical scene object rows and task scene rows during the proposal read.

The previous round's task-owner visual prior is also applied as a low-amplitude
visual bias for physical scene object rows on the next AQR query round.  This
keeps the fix causal within the AQR iterative measurement update: task rows
first read language/image support, then physical rows receive a soft ownership
bias.  It is not a hard label and it does not overwrite posterior.
```

New config/debug contract:

```text
task_owner_bias_enabled = True
task_owner_visual_bias_weight = 0.20
task_owner_proposal_bias_weight = 0.50
task_owner_proposal_objectness_power = 0.50
task_owner_proposal_static_only = True

debug:
  aqr_task_owner_visual_prior_entropy
  aqr_task_owner_visual_prior_max
  aqr_task_owner_proposal_score_max
  aqr_task_owner_proposal_score_mean
  aqr_task_owner_proposal_score_nonzero_fraction
```

Remaining acceptance:

```text
This is code-closed but not behavior-closed.  The next diagnostic must show
that task-owner proposal scores are nonzero and that active object anchors move
toward the intended task object without increasing active same-role duplicate
overlap.
```

### I6. Tracklet typed memory is wired but inactive

Status:

```text
open-data / scheduled
```

Reason:

```text
The contracts and readers exist, but current CALVIN run has no tracklet sidecar:
owm_tracklet_tokens=0.
```

Fix requirement:

```text
Generate tracklet sidecars or provide dataset tracklet fields.  Without that,
the model cannot use tracklet same-object continuity.
```

Dataset-scale agenda:

```text
This is now promoted to a planned offline preprocessing track.  It is feasible
for most large robot datasets that contain continuous RGB frames:

1. static-view tracklets from rgb_static videos;
2. wrist-view tracklets from rgb_gripper videos when available;
3. optional RGB-D lift to 3D when depth/intrinsics are reliable;
4. confidence/visibility/forward-backward-error filtering;
5. safe no-op when a dataset lacks video continuity.

Expected sidecar fields:
  tracklet_xy
  tracklet_velocity
  tracklet_visibility
  tracklet_confidence
  tracklet_ids
  tracklet_view_ids
  tracklet_age
```

### I7. Offline IsSameObject probe

Status:

```text
closed-code / open-train / watch
```

Reason:

```text
The audit/probe script exists and has now been run on the latest task-owner
overlay artifacts.  It is still a weak offline diagnostic, not an online loss
or a behavior proof.
```

Latest result:

```text
run:
  picf_a7_task_owner_bias_samseg0_diag300_20260517

artifact:
  same_object_probe_posterior_step150.json

binding_signature_cos_auc       = 0.943820
combined_auc                    = 1.000000
duplicate_candidate_fraction    = 0.238095
decision                        = binding_subspace_decodable_but_assignment_duplicates_candidates
```

Next:

```text
Repeat after the 300-step diagnostic and after any long cotrain run.  Acceptance
requires duplicate_candidate_fraction to fall while binding/combined AUC stays
high.  Do not turn this into an online pseudo-label loss without real masks,
tracklets, or a stronger weak-label source.
```

### I8. Ordinal / fourth-object grounding

Status:

```text
open-data
```

Reason:

```text
Current ordinal module is weak diagnostic.  Without ordinal/rank labels or
reliable weak selected-slot targets, this cannot be declared solved.
```

### I9. Action cotrain acceptance

Status:

```text
open-train
```

Reason:

```text
Current tactile/proposal diagnostic has loss_action=0.  It tests anchor dataflow
only.  Full action cotrain must be judged in a separate long run.
```

### I10. Dataset-scale proposal and tracklet sidecar generation

Status:

```text
paused / archived as unproven-benefit data upgrade
```

Reason:

```text
Proposal sidecars and tracklet sidecars solve different missing evidence
edges:

contact/task-guided proposal:
  weak task-object mask/box/point support per frame. The mask support is stored
  as normalized sparse samples plus weights, not point-row ids, so it remains
  valid after training-time point sampling/FPS.

  Tracklet:
    temporal same-object continuity across frames.

  The current code can consume both.  Blind SAM generation is archived because
  visual checks showed noisy non-task boxes.  The maintained proposal generator
  is contact-motion/task-guided.  The tracklet generator exists and can use
  inspected proposal roots as seeds.  The remaining missing evidence edge is
  dataset-scale contact proposal / tracklet production and training acceptance.
```

Required pipeline:

```text
1. Run contact-motion / task-guided offline proposal generation per available RGB view.
2. Save proposal_centers_xy / proposal_boxes_xyxy / objectness / view_ids /
   source_ids plus proposal_mask_xy / proposal_mask_weights /
   proposal_mask_offsets sidecars.
3. Run CoTracker/TAPIR/optical-flow-style offline tracklets per video view.
4. Save tracklet_* sidecars with visibility/confidence filtering.
5. Join sidecars by episode/frame without modifying the original dataset.
6. Run a small training smoke with:
     owm_proposal_tokens > 0
     owm_tracklet_tokens > 0
     aqr_task_owner_proposal_score_nonzero_fraction > 0
     aqr_tracklet_support_max > 0
7. Run IsSameObject probe before/after sidecars to test whether duplicate
   candidate fraction falls while binding AUC remains high.
```

2026-05-18 update:

```text
The sidecar line is paused, not accepted and not rejected.

What is proven:
  - Tracklet sidecar files can be generated and loaded.
  - Visualization samples show valid static/gripper track points.
  - AQR can receive nonzero `owm_tracklet_tokens` in a sidecar diagnostic.

What is not proven:
  - Tracklet-only sidecars improve task-object posterior binding versus a
    no-sidecar run under matched trainability/action/unroll settings.
  - Blind SAM proposal memory improves binding; visual checks rejected it as
    too noisy for production-default evidence.

Current decision:
  - Stop/pause dataset-scale sidecar generation to avoid spending compute on
    an unproven branch.
  - Keep the code path, partial artifacts, and restart notes.
  - Resume only for a controlled A/B/C:
      A: no sidecar;
      B: tracklet-only sidecar, proposal memory off;
      C: contact/task-guided proposal sidecar, if generated and inspected.
```

Legacy SAM/proposal operational state:

```text
2026-05-17:
  script: scripts/archive/picf_sam_proposal_precompute_legacy.py
  machine: A7, two A100-40GB GPUs
  output: /mnt/picf_sidecars/sam_proposals_vitb_full_20260517
  smoke output: /mnt/picf_sidecars/sam_proposals_vitb_smoke_20260517
  smoke preview: /mnt/picf_sidecars/sam_proposals_vitb_smoke_20260517_previews
  model: SAM ViT-B
  views: static + gripper
  shard mode: segment-level, 2 shards
  restart: --skip-existing

  Smoke acceptance:
    6 frames generated;
    282 total proposals;
    both static and gripper view ids present;
    first previews inspected and non-empty.

  Full generation:
    sessions:
      sam_sidecar_vitb_shard0_20260517
      sam_sidecar_vitb_shard1_20260517
    early output count exceeded 180 npz files;
    full training split size is 1,053,873 frames, so complete generation is an
    offline production job, not an interactive test.

  12-hour bounded generation:
    Full per-frame generation was projected at roughly 10 days.  The active
    production job was changed to:
      output=/mnt/picf_sidecars/sam_proposals_vitb_stride8_p8_12h_20260517
      frame_stride=8
      points_per_side=8
      workers=4
      two workers per A100
    This keeps the same proposal_* contract while producing broad dataset
    coverage inside the current time budget.

  Active phase-0 coverage:
    The stride=8 estimate still risked exceeding the 12-hour budget.  Active
    run was switched to hierarchical phase-0:
      output=/mnt/picf_sidecars/sam_proposals_vitb_stride16_p8_phase0_20260517
      frame_stride=16
      frame_offset=0
      points_per_side=8
      workers=4
    Later offsets can densify the same contract without changing code.

  2026-05-17 12:24 CST strict audit:
    status=running
    npz_count≈60,296
    active_workers=4
    sampled_files=159
    bad_fields=0
    sampled frames with both static and gripper proposals=159/159
    proposals per frame min/median/max=25/32/43
    objectness min/median/max≈0.891/0.961/1.0

  decision:
    phase0 generation may continue.  This closes the immediate sidecar-quality
    gate, not the downstream behavior gate.  Later offsets can densify coverage
    without changing the proposal schema.
```

Non-goals:

```text
Do not turn SAM masks or tracklets into hard posterior labels.  They are typed
evidence for measurement routing and posterior correction, not ground truth.
```

## Immediate Priority

```text
P0:
  Task-conditioned proposal/PG owner bias.  This is the most direct answer to
  "anchors do not choose the correct task object".

P1:
  Repeat IsSameObject probe at step 300 and after any long cotrain.  Current
  step-150 result shows decodable binding subspace but duplicate candidates.

P1:
  Monitor the running dataset-scale SAM/proposal sidecar generation and archive
  its manifest/acceptance once it completes or once a usable coverage slice is
  produced.

P2:
  Tracklet sidecar production for static and wrist RGB videos.  The generator
  is implemented as scripts/picf_tracklet_sidecar_precompute.py and should be
  run after enough SAM proposal coverage exists.

Tracklet operational state:

```text
2026-05-17:
  generator=scripts/picf_tracklet_sidecar_precompute.py
  smoke=A5, segments 32/33/47/63
  smoke result:
    saved_frames=239
    tracklet_count=14,315
    static+gripper tracklets present on every saved frame
    keyframes preserve proposal_* when proposal sidecar exists

  production=A5
  output=/mnt/picf_sidecars/tracklets_samseed_stride16_w15_phase0_20260517
  sessions=tracklet_samseed_phase0_shard0..7_20260517
  quality guard=--require-proposal-keyframe
  seed mix=sam_seed_fraction=0.5 plus low-confidence grid coverage

  2026-05-17 12:24 CST strict audit:
    status=running
    npz_count≈30,463
    active_workers=8
    sampled_files=139
    bad_fields=0
    sampled frames with both static and gripper tracklets=139/139
    per-frame tracklets min/median/max=45/59/64
    duplicate track-id frames=0
    confidence min/median/max≈0.152/0.437/1.0
    age min/median/max=0.0/0.467/1.0

  decision:
    generation may continue.  This closes the immediate script-quality gate,
    not the downstream behavior gate.  Because production started before SAM
    phase0 completed, a second pass with --skip-existing-tracklets is required
    after SAM completion to fill proposal-gated windows that were skipped.
```

2026-05-17 14:45 CST update:

```text
SAM phase0:
  resolved-for-diagnostic
  A7 output contains 66,928 proposal sidecars and passed strict sampled
  structural validation.  It is now acceptable input for proposal-memory
  diagnostic training.

Tracklet phase0:
  still open-data
  A5 output has hundreds of thousands of valid files, but two shards failed on
  corrupt/empty CALVIN source records.  The generated files remain usable for
  debugging; the root is not yet a complete training sidecar until failed
  source records are skipped and missing shards are backfilled.

Tactile dense reread materialization:
  resolved-code
  The lazy warmup previously initialized tactile dense reread projections with
  AnyTouch native width.  The real PICF dense tactile memory uses
  `tactile_patch_token_proj(...)`, so the correct warmup width is
  `core.config.hidden_dim`.  This has been fixed in scripts/picf_core_train.py.
```

2026-05-17 22:10 CST A5 tracklet validation update:

```text
A5 state:
  GPUs idle, no tmux, no live tracklet/cotracker/tapir/raft process.

Current root:
  /mnt/picf_sidecars/tracklets_samseed_stride16_w15_phase0_20260517

Observed files:
  npz_count≈543,959

Completed shards:
  shard2, shard3, shard5 wrote final manifests.

Interrupted shards:
  shard0, shard1, shard4, shard6, shard7 hit EOFError/BadZipFile while reading
  an existing npz during resume/merge.

Engineering root cause:
  scripts/picf_tracklet_sidecar_precompute.py read existing npz files without
  corrupt-file tolerance and wrote compressed npz files directly to the final
  path.  A preemption or partial write could leave an empty/bad zip that killed
  the next resume attempt.

Repair:
  _load_sidecar_payload now treats EOFError/BadZipFile/ValueError/OSError as
  missing evidence, unlinks the bad sidecar when possible, and resumes.
  _merge_and_save now writes through atomic temp-npz + os.replace.

Decision:
  Continue A5 only after deploying this script repair.  The existing files are
  useful partial evidence, but the root is not complete until the interrupted
  shards are resumed and final manifests exist for all shards.
```

P3:
  Ordinal weak supervision.  Do not hard-enable before target selection is
  reliable.
```

## Module Disposition Audit - 2026-05-18

This section is the current cleanup answer to the repeated
`loss_anchor_pv`/raw-overlap failures.  It separates the maintained production
profile from historical diagnostics so old failed branches do not keep
re-entering the launch recipe.

### Keep In Production

```text
AQR typed evidence router:
  keep.  It is the core measurement-routing model and is not the source of
  the historical false positives.

PaliGemma image/text support:
  keep and cotrain in production until sidecar evidence proves it can be
  replaced.  The frozen-PaliGemma diagnostic improved action loss but showed
  structural drift by step300.

V-JEPA temporal multiview support:
  keep.  It provides past/current visual evidence without future leakage and
  is not implicated as the collapse cause.

Point/Sonata and soft tactile evidence:
  keep.  Dense AnyTouch projection and soft tactile gating are repaired and
  mathematically necessary evidence channels, not extra objectives.

Active/context/reserve routing:
  keep.  It is the correct no-object/capacity model for fixed slots.  Raw
  overlap can remain high in reserve/context rows; the acceptance metric is
  active object overlap plus overlay correctness.

Posterior owner-active gate, posterior file competition, posterior birth
transport:
  keep.  These repair concrete dataflow bugs where reserve/inactive evidence
  reached persistent object files or dustbin residuals re-created duplicates.

Cache residual gate and skip-latest-posterior cache row:
  keep.  These are closed math bugs with explicit regression guards.
```

### Keep Guarded Or Data-Dependent

```text
Tracklet typed memory:
  keep code path.  It is data-dependent and currently absent when
  owm_tracklet_tokens=0.  Do not judge it from no-sidecar CALVIN runs.

Prompted/contact/tracklet-reranked proposal sidecars:
  keep as explicit ablation/data sidecars.  They must remain weak typed
  evidence, not hard labels.

Offline IsSameObject probe:
  keep as audit, not online loss.  It diagnoses whether the binding subspace
  is decodable and whether duplicate candidates remain.
```

### Archive Or Keep Off By Default

```text
Blind SAM proposal memory:
  production-default off.  Automatic masks are high-recall objectness
  candidates, but experiments and visual inspection showed wall/robot/drawer
  fragments.  Keep only for prompted/reranked sidecar experiments.

Legacy local refinement:
  archived/off by default.  It added an extra top-k residual evidence path and
  showed recycle/gradient pressure without being necessary for early
  non-collapse.

Role-wise local candidate competition and deterministic coverage-seeded local
proposal:
  rejected.  Prior A5/A7 evidence showed they did not prevent local candidate
  reuse once same-role rows had already become similar.

Slot-JEPA, support prediction, binding-consistency, AQR denoising:
  keep hooks but default zero.  They are not production losses until identity
  assignment is stable over a long run; otherwise they risk training on
  unstable pseudo-correspondences.
```

### Current Interpretation Of The Old Failure

```text
loss_anchor_pv rising while loss_pv_weak falls means dense visual/point
correspondence can improve while object-routed anchor ownership worsens.

raw same-role overlap near 1.0 is not by itself a production failure after
active/context/reserve routing, because raw includes reserve/context capacity.
It becomes a failure when it is coupled to:
  active same-role overlap rising,
  active object-core overlap rising,
  loss_anchor_pv rising,
  posterior identity switch rising,
  overlays showing active owners missing the task object.

The maintained simplification is therefore not "add another overlap loss".
It is:
  keep the belief-filter dataflow repairs,
  keep active/context/reserve separation,
  keep PaliGemma cotrain for semantic adaptation,
  keep blind SAM off,
  wait for sidecar tracklet/proposal evidence only when it is actually present,
  judge long-run behavior by active ownership, overlays, action loss, and
  identity stability together.
```

### I11. Dustbin residual broadcast re-duplicates inactive posterior files

Status:

```text
patched-local / local audits passed / remote diagnostic confirms target fix
```

Run exposing the issue:

```text
picf_a7_sam_phase0_anchoronly_diag500_20260517
stopped after step300/350; do not wait to 500.
```

Evidence:

```text
step50 -> step300:
  loss_anchor_pv                         0.7533 -> 1.1787
  loss_pv_weak                           6.2616 -> 4.0849
  loss_slot_jepa                         0.7818 -> 4.7354
  aqr_same_role_support_overlap_max      0.1733 -> 0.9996
  aqr_active_same_role_support_overlap   0.0184 -> 0.4216
  posterior_recycle_rate                 0.0556 -> 0.2743
  posterior_identity_switch_rate         0.7278 -> 0.7761
  posterior_active_duplicate_overlap     0.0000 -> 0.0000
  owm_proposal_tokens                    >0
  owm_tracklet_tokens                    0
```

Diagnosis:

```text
This is not caused by action cotrain because action loss is disabled.  It is
not caused by SAM proposal dataflow being absent because proposal tokens are
nonzero.  The remaining mechanism is posterior lifecycle transport:

1. Sinkhorn+dustbin and file competition demote duplicate support into the
   observation dustbin.
2. The recycle path then allowed multiple inactive/reserve files to reset from
   the same dustbin residual.
3. Those reserve files became duplicate candidates even though active duplicate
   filtering still protected the direct active path.

Therefore the missing component is a birth/no-object competition after file
competition, not another generic diversity loss.
```

Repair contract:

```math
support,dustbin
  \xrightarrow{file\ competition}
support_{active}, dustbin'
  \xrightarrow{birth\ competition}
support_{active}+birth\_share*dustbin', dustbin''
```

```text
Only a small number of high-reset, low-alpha reserve files may consume dustbin
residual as new-object birth evidence.  Other inactive files remain null
capacity.  This preserves all evidence mass without broadcasting the same
dustbin vector into every duplicate file.
```

Acceptance:

```text
Short diagnostic must show:
  posterior_file_competition_birth_count bounded;
  posterior_recycle_rate does not climb with active overlap;
  aqr_active_same_role_support_overlap_max does not steadily increase;
  loss_anchor_pv does not trend upward;
  full duplicate overlap may remain a reserve telemetry warning, but active
  duplicate and active same-role overlap must stay controlled.
```

Remote diagnostic update:

```text
picf_a7_birth_transport_sam_phase0_anchoronly_diag300_20260517:
  step50:
    posterior_birth_count                    0.7700
    posterior_inactive_file_recycle_rate     0.0000
    posterior_active_duplicate_overlap       0.0000
  step150:
    posterior_birth_count                    0.0000
    posterior_inactive_file_recycle_rate     0.0000
    posterior_active_duplicate_overlap       0.0000
  step200:
    posterior_birth_count                    0.0000
    posterior_inactive_file_recycle_rate     0.0000
    posterior_active_duplicate_overlap       0.0000
```

Conclusion:

```text
The dustbin-broadcast failure is fixed for the tested anchor-only diagnostic.
Remaining overlap is upstream AQR support assignment / sparse proposal
coverage, not posterior birth transport.
```

### I12. AQR support assignment pressure still re-concentrates same-role support

Status:

```text
open-analysis
```

Evidence:

```text
picf_a7_birth_transport_sam_phase0_anchoronly_diag300_20260517:
  step50:
    aqr_same_role_support_overlap_max        0.1999
    aqr_active_same_role_support_overlap_max 0.0213
    posterior_birth_count                    0.7700
    posterior_recycle_rate                   0.0499
  step100:
    aqr_same_role_support_overlap_max        0.9230
    aqr_active_same_role_support_overlap_max 0.1889
    posterior_birth_count                    0.0150
    posterior_recycle_rate                   0.0047
  step150:
    aqr_same_role_support_overlap_max        0.9995
    aqr_active_same_role_support_overlap_max 0.2858 average / 0.9407 overlay sample
    posterior_birth_count                    0.0000
    posterior_recycle_rate                   0.0023
  step200:
    aqr_same_role_support_overlap_max        0.9989
    aqr_active_same_role_support_overlap_max 0.3457 average
    posterior_birth_count                    0.0000
    posterior_recycle_rate                   0.0014
```

Diagnosis:

```text
Posterior birth transport is no longer the dominant failure.  The evidence
residual is not being broadcast into reserve births, and active duplicate
overlap stays zero.  However, AQR same-role supports still re-concentrate
upstream, which worsens anchor_pv.  This points to support assignment/routing
calibration: task/proposal/PV support can sharpen onto a common object region
faster than same-role competition can maintain distinct object supports.
```

Revised diagnosis:

```text
Posterior birth transport is no longer the dominant failure.  The evidence
residual is not broadcast into reserve births, inactive recycle is zero, and
active duplicate overlap stays zero.  The active overlay still fails to put a
clear posterior object file on the red block in the sampled frames.  This
points to task-conditioned proposal/visual support assignment rather than file
lifecycle.
```

Next action:

```text
Historical next action superseded on 2026-05-18: blind SAM sparse coverage is
no longer repaired for production. Use contact-motion sidecars and inspect
active_only + with_gray + sidecar_proposals overlays instead.
```

### I13. Sparse legacy SAM sidecar silently disappears from sampled overlays

Status:

```text
superseded by blind-SAM archival / retained as historical failure analysis
```

Evidence:

```text
Current sparse sidecar root:
  /mnt/picf_sidecars/sam_proposals_vitb_stride16_p8_phase0_20260517

Captured overlay frames:
  step50  step_id=1572920 exact sidecar exists -> proposals=32, boxes drawn
  step100 step_id=1572925 exact sidecar absent -> proposals=0, no boxes drawn
  step150 step_id=1572917 exact sidecar absent -> proposals=0, no boxes drawn
  step200 step_id=1572925 exact sidecar absent -> proposals=0, no boxes drawn

Nearest sidecars around missing frames:
  step_id=1572917 nearest +3 / -13
  step_id=1572925 nearest -5 / +11
```

Root cause:

```text
The training metric owm_proposal_tokens is averaged over the batch/window and
can be nonzero while the one saved overlay frame has no exact sidecar file.
This creates a false visual diagnostic: the operator sees no boxes even though
some windows in the same logging interval use proposal tokens.
```

Repair:

```text
Add an explicit nearest-proposal sidecar fallback:
  --mvtrack-sidecar-proposal-nearest-max-gap N

If an exact frame proposal is absent, use the nearest sidecar within N CALVIN
steps as weak evidence and record proposal_age.  Proposal objectness is
exponentially age-decayed:

  objectness_eff = objectness * exp(-proposal_age / proposal_age_decay_steps)

This is not hard label propagation.  It is age-aware frozen weak evidence,
matching the sparse sidecar generation contract.
```

Acceptance:

```text
1. Anchor overlays at non-sidecar frames still draw boxes with /a{gap} tags.
2. JSON records proposal_age for every borrowed proposal.
3. owm_proposal_tokens becomes less sparse without treating old boxes as truth.
4. Active posterior anchors must be checked against the task object after this
   repair; success is not implied by proposal coverage alone.

### I14. Proposal boxes exist but do not move active object geometry

Status:

```text
open / v2 bridge is live but not stable
```

Evidence:

```text
A7 sparse-proposal diagnostic, step100:
  instruction: pick up the red block lying in the drawer
  proposals: 32
  nearest proposal to red block center:
    center ~= (106, 156) px
    red block visual center ~= (107, 157) px
    age = 5

  nearest active anchors to red block:
    ~20 px, ~25 px, ~30 px

  closest graph row:
    ~6 px from red block but inactive/context
```

Root cause:

```text
SAM/proposal evidence entered AQR as proposal tokens and proposal priors, but
anchor_x was still computed from point_priors only:

  x_j = sum_i p_point[j,i] X_i

Therefore a red-block proposal could be visible and scored but still fail to
move physical active object geometry unless the independent point reader also
selected the same projected points.  This is a proposal-to-3D geometry dataflow
gap, not a pure training-step issue.
```

Repair v1:

```text
Bridge proposal priors to point priors through static projective geometry:

  P_prop_to_point[j,i] =
    sum_k P_prop[j,k] Normalize_i(soft_inside(point_i, box_k))

  P_point_new[j] =
    Normalize((1-lambda) P_point[j] + lambda P_prop_to_point[j])

Defaults:
  proposal_point_bridge_weight = 0.35
  proposal_point_bridge_edge_tau = 0.02
```

Follow-up evidence:

```text
A7 proposal-point bridge diagnostic, step50:
  proposals: 32
  nearest proposal to red block:
    center ~= (106,156) px
    red block visual center ~= (107,157) px
    age = 0

  bridge metrics:
    aqr_proposal_point_bridge_max ~= 0.058
    aqr_task_owner_anchor_score_max ~= 0.272
    aqr_task_owner_proposal_score_max = 1.0

  nearest active anchors to red block:
    graph/task/posterior still ~= 37-53 px away
```

This proves v1 repaired proposal coverage and proposal-to-point transport, but
it still depended on a physical row first reading the red-block proposal.  If no
physical row attends to the task-owner proposal, geometry still will not move.

Repair v2:

```text
Use the task-owner proposal score itself as a weak point measurement for
physical scene-object rows:

  P_owner_point[i] =
    sum_k P_owner_proposal[k] Normalize_i(soft_inside(point_i, box_k))

  P_point_new[j] =
    Normalize((1-lambda_owner) P_point[j] + lambda_owner P_owner_point)

only for task-owner-eligible physical scene rows.

Defaults:
  task_owner_proposal_point_bridge_weight = 0.50
```

This is still not a hard SAM label.  It is the missing transport leg:

```text
task/query semantic prior -> task-owner proposal -> projected point likelihood
-> physical object-file competition -> posterior correction
```

V2 initial diagnostic:

```text
A7 task-owner point bridge diagnostic, step50:
  owm_proposal_tokens ~= 31
  nearest proposal to red block:
    center ~= (106,156) px
    red block visual center ~= (107,157) px
    distance ~= 1.4 px

  bridge metrics:
    aqr_proposal_point_bridge_max ~= 0.041
    aqr_task_owner_point_bridge_max ~= 0.017
    aqr_task_owner_anchor_score_max ~= 0.193
    aqr_task_owner_proposal_score_max = 1.0

  nearest active anchors to red block:
    graph active ~= 11.2 px
    posterior active ~= 11.9 px
    graph active ~= 13.2 px

  structure:
    aqr_same_role_support_overlap_max ~= 0.174
    posterior_file_competition_active_duplicate_overlap_max = 0
```

This no longer matches the previous failure mode.  The active object files now
reach the task object neighborhood instead of remaining only near the
gripper/drawer edge.  Continue the diagnostic to step100/300 to confirm this is
stable and not only a step50 transient.

Step100 regression:

```text
step100:
  aqr_task_owner_point_bridge_max ~= 0.020
  aqr_task_owner_proposal_score_max = 1.0
  nearest active anchor to red block ~= 24 px
  nearest active posterior ~= 41 px
  loss_anchor_pv ~= 1.18
  loss_aqr_denoising ~= 2.03
  aqr_same_role_support_overlap_max ~= 0.69
```

Conclusion:

```text
V2 proves the bridge exists and can pull the task object at step50, but the
weak measurement is not retained under continued anchor/PV/support optimization.
The remaining root issue is a stability/calibration problem: task-owner
geometry evidence must remain a persistent observation likelihood for eligible
object files instead of a small prior that can be overwhelmed after warmup.
```

Repair v3:

```text
Move the same task-owner proposal-to-point evidence into point-reader logits:

  point attention logits =
    q_j k_i
    + base point bias
    + task-owner proposal-point log-likelihood bias

for eligible physical scene-object rows.

Default:
  task_owner_proposal_point_bias_weight = 0.75
```

This is a cleaner fix than another auxiliary loss because it repairs the
measurement model itself.  The SAM/proposal sidecar remains frozen evidence;
posterior competition still decides which object file survives.

V3 initial result:

```text
step50:
  nearest active graph ~= 10 px
  nearest active posterior ~= 13 px
  aqr_same_role_support_overlap_max ~= 0.169
  active duplicate overlap = 0

step100:
  nearest active graph ~= 14 px
  nearest active task ~= 15 px
  nearest active posterior ~= 24 px
  loss_anchor_pv ~= 1.06
  aqr_same_role_support_overlap_max ~= 0.688
  active duplicate overlap = 0
```

V3 is materially better than v2 step100, where the nearest active file was
about 24 px and nearest active posterior about 41 px, but it is not a full
closure yet.  The task object is retained near the proposal edge; posterior
ownership and same-role separation still need step300/long-run validation.

Files:

```text
docs/PICF_AQR_OWM_PROPOSAL_POINT_BRIDGE_FOLLOWTHROUGH_20260517_TEMP.md
src/openpi/picf/core/config.py
src/openpi/picf/core/contracts.py
src/openpi/picf/core/pipeline.py
scripts/picf_core_train.py
scripts/verify_picf_owm_contract.py
scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py
```

Acceptance:

```text
1. aqr_proposal_point_bridge_max > 0.
2. aqr_task_owner_anchor_score_max > 0.
3. The active-only overlay for red-block tasks shows at least one active
   physical/posterior object file on the red-block proposal region.
4. Active duplicate overlap stays controlled.
```
```

## I15 - Loss-Family Audit: PV Improves While Graph Ownership Re-Collapses

Status: open.

Reference:

```text
docs/PICF_AQR_OWM_LOSS_AUDIT_20260517_TEMP.md
```

Current A7 task-owner point-bias diagnostic:

```text
step50 -> step100 -> step150

loss_pv_weak:
  6.0704 -> 4.2056 -> 3.5800

loss_anchor_pv:
  0.7182 -> 1.0567 -> 1.1711

loss_mapg_support_diversity:
  0.3667 -> 0.2879 -> 0.5140

aqr_same_role_support_overlap_max:
  0.1695 -> 0.6881 -> 0.9979

posterior_file_competition_active_duplicate_overlap_max:
  0.0 -> 0.0 -> 0.0
```

Diagnosis:

```text
Dense point/visual representation learning is improving, but object-routed
anchor assignment is not stable.  The posterior active-file duplicate filter is
working; the remaining failure is upstream AQR graph support reuse by same-role
anchors.
```

Important negative finding:

```text
This is not caused by action loss in the current run.  Action weights are zero.
It is also not caused by slot-JEPA/support-pred/binding-consistency/denoising,
because all guarded OWM loss weights are zero in this diagnostic.
```

Next action:

```text
Fix the graph ownership layer rather than adding another unrelated auxiliary
loss.  Task/object evidence should create one owner plus bounded context/reserve
capacity, not be cloned across many same-role graph rows.
```

## I16 - Blind SAM Proposals Include Wall/Robot/Drawer Fragments

Status: closed as rejected / archived. Blind automatic SAM is no longer a
candidate for current task-object sidecar generation.

Reference:

```text
docs/archive/picf_aqr_owm_202605/sam_rejected_20260519/PICF_AQR_OWM_SAM_PROPOSAL_QUALITY_REPAIR_20260517_TEMP.md
```

Observed problem:

```text
SAM sidecars contain useful task-object boxes, but also wall patches, robot
protrusions, drawer sides, and other image fragments.  These are valid
high-recall automatic masks, but they are not manipulation object labels.
```

Root cause:

```text
The previous runtime proposal score trusted SAM objectness and task-visual box
mass too broadly.  SAM objectness measures mask quality, not task ownership.
Therefore false-positive fragments could enter proposal attention and
proposal-to-point bridge as if they were plausible object evidence.
```

Mathematical repair:

```text
proposal_quality =
  sigmoid(area - area_min)
  * sigmoid(area_max - area)
  * sigmoid(aspect - aspect_min)

task_owner_proposal_score =
  TopKFloor(task_visual_box_mass * objectness^gamma * proposal_quality)

proposal_context_score =
  objectness * proposal_quality^eta
```

Design constraint:

```text
This is not a hard SAM label.  Dense V-JEPA/PG/point tokens remain available.
SAM proposals only become calibrated typed evidence.
```

Implemented code:

```text
src/openpi/picf/core/config.py
  proposal_shape_quality_enabled
  proposal_shape_area_min / max
  proposal_shape_aspect_min
  proposal_context_quality_power
  task_owner_proposal_topk
  task_owner_proposal_score_floor

src/openpi/picf/core/pipeline.py
  _proposal_shape_quality
  _postprocess_task_owner_proposal_score
  proposal reader uses objectness * shape_quality^eta
  task-owner proposal score uses shape_quality + top-k/floor
  proposal-to-point matrix is shape-quality weighted
```

Acceptance:

```text
1. aqr_task_owner_proposal_selected_count <= task_owner_proposal_topk.
2. aqr_proposal_shape_quality_* metrics are present.
3. Task-owner selected proposals on overlays are object-like boxes, not wall or
   robot fragments.
4. Active task-object anchor/presenter reaches the correct task-object region.
```

Remaining data work:

```text
Prompted SAM/SAM2 from language/contact/action/tracklet seeds is a sidecar
generation upgrade.  It should be added, but runtime cannot pretend tactile
object masks exist before those sidecars or contact labels are generated.
```

Runtime diagnostic, A7 `picf_a7_sam_quality_anchoronly_diag300_20260517`:

```text
step50:
  owm_proposal_tokens=31.02
  aqr_proposal_shape_quality_mean=0.8704
  aqr_task_owner_proposal_selected_count=3.895
  aqr_task_owner_proposal_score_entropy=0.3483
  aqr_task_owner_point_bridge_max=0.0251
  aqr_active_same_role_support_overlap_max=0.0207
  aqr_same_role_support_overlap_max=0.2669

step100:
  owm_proposal_tokens=31.27
  aqr_proposal_shape_quality_mean=0.8745
  aqr_task_owner_proposal_selected_count=3.815
  aqr_task_owner_proposal_score_entropy=0.3390
  aqr_task_owner_point_bridge_max=0.0351
  aqr_active_same_role_support_overlap_max=0.0975
  aqr_same_role_support_overlap_max=0.6963
```

Interpretation:

```text
The new runtime gates are definitely active: proposal count is nonzero,
task-owner selection is sparse, and shape quality is recorded.  However, the
step100 overlay for "pick up the red block lying in the drawer" still shows
active anchors concentrated around the gripper/drawer region rather than a
clean task-object posterior on the red block.  This means the shape-quality
repair is a necessary safety calibration for blind SAM, not a complete
task-object grounding solution.
```

Next repair direction:

```text
Move task-object proposal generation upstream:

  language/PG task prior
  + action endpoint / gripper trajectory
  + tactile/contact point when calibrated
  + tracklet continuity when available
  -> prompted SAM/SAM2 masks or reranked automatic masks
  -> proposal_* sidecars with source_ids and quality

This is the mathematically coherent path because the missing variable is not
mask geometry quality alone; it is task ownership.  Runtime AQR should consume
that evidence as weak typed support, not hallucinate it from blind SAM
objectness.
```

2026-05-17 22:38 CST full-sidecar frozen-anchor diagnostic:

```text
Run:
  picf_a7_full_sidecar_anchoronly_diag300_20260517

Host:
  A7 /root/openpi_posterior_vla_clean

Purpose:
  Verify that the repaired SAM/proposal sidecar path and the resumed
  tracklet sidecar path can be consumed together by the AQR graph while
  PaliGemma, V-JEPA, AnyTouch, and Sonata are frozen.  This intentionally
  tests evidence routing and anchor dynamics without semantic/action
  gradients masking sidecar dataflow problems.

Key launch guards:
  trainable_scope=anchor_only
  perception_finetune_mode=frozen
  semantic=paligemma(trainable=False)
  visual=encoder(trainable=False)
  tactile=encoder(trainable=False)
  point=sonata(trainable=False)
  lambda_action_pos/rot/gripper=0
  tracklet_memory_enabled=True
  proposal_memory_enabled=True
  mvtrack_sidecar_root=/mnt/picf_sidecars/tracklets_samseed_stride16_w15_phase0_20260517
  mvtrack_sidecar_proposal_nearest_max_gap=8
  proposal_shape_quality_enabled=True
  task_owner_proposal_topk=4
  local_refinement_enabled=False
```

First-pass acceptance:

```text
1. owm_tracklet_tokens > 0.
2. owm_proposal_tokens > 0.
3. aqr_tracklet_support_* metrics are present.
4. aqr_proposal_shape_quality_* metrics are present.
5. aqr_task_owner_proposal_selected_count <= 4.
6. posterior_file_competition_active_duplicate_overlap_max == 0.
7. Active overlap stays separated from raw/reserve overlap.
8. Anchor overlays draw active anchors and SAM/proposal boxes for direct visual audit.
```

Step-50 check:

```text
step=50
  loss_total = 0.4759
  loss_anchor_pv = 0.7304
  loss_pv_weak = 6.0599
  loss_aqr_denoising = 1.8454
  loss_mapg_support_diversity = 0.3053
  loss_mapg_routing = 0.6789

  owm_tracklet_tokens = 58.24
  owm_proposal_tokens = 1.92
  aqr_tracklet_support_max = 0.1236
  aqr_tracklet_support_entropy_mean = 0.9514
  aqr_proposal_shape_quality_mean = 0.0533
  aqr_task_owner_proposal_selected_count = 0.235

  aqr_same_role_support_overlap_max = 0.2162
  aqr_active_same_role_support_overlap_max = 0.0562
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.0501
  posterior_identity_switch_rate = 0.7139
```

Interpretation:

```text
The combined sidecar path is connected: tracklet and proposal tokens are both
nonzero in the same frozen-perception anchor-only graph.  The active-owner
overlap and active duplicate overlap are healthy at the first checkpoint.
Identity switch is still high at step50, which is expected for an early
anchor-only warmup with newly introduced sidecar evidence; it must trend down
before this configuration is treated as behavior-ready.

The sampled overlay exists:
  anchor_overlays/step_000050__pick_up_the_red_block_lying_in_the_drawer__active_only.png
  anchor_overlays/step_000050__pick_up_the_red_block_lying_in_the_drawer__with_gray.png
  anchor_overlays/step_000050__pick_up_the_red_block_lying_in_the_drawer__sam_proposals.png
  anchor_overlays/step_000050__pick_up_the_red_block_lying_in_the_drawer.json
```

Step-100/150 check:

```text
metrics rows:

step=100
  loss_total = 0.4931
  loss_anchor_pv = 0.9341
  loss_pv_weak = 4.1835
  loss_aqr_denoising = 2.1765
  loss_mapg_support_diversity = 0.3120
  loss_mapg_routing = 0.7335
  owm_tracklet_tokens = 57.42
  owm_proposal_tokens = 1.44
  aqr_tracklet_support_max = 0.2755
  aqr_task_owner_proposal_selected_count = 0.170
  aqr_task_owner_point_bridge_max = 0.00069
  aqr_same_role_support_overlap_max = 0.4477
  aqr_active_same_role_support_overlap_max = 0.1225
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_identity_switch_rate = 0.6956
  posterior_recycle_rate = 0.0053

step=150
  loss_total = 0.4941
  loss_anchor_pv = 0.9781
  loss_pv_weak = 3.4528
  loss_aqr_denoising = 1.9193
  loss_mapg_support_diversity = 0.3093
  loss_mapg_routing = 0.7372
  owm_tracklet_tokens = 57.76
  owm_proposal_tokens = 1.44
  aqr_tracklet_support_max = 0.5213
  aqr_task_owner_proposal_selected_count = 0.130
  aqr_task_owner_point_bridge_max = 0.00091
  aqr_same_role_support_overlap_max = 0.9617
  aqr_active_same_role_support_overlap_max = 0.2727
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_identity_switch_rate = 0.6528
  posterior_recycle_rate = 0.0014
```

Overlay-side dataflow caveat:

```text
The saved step100/step150 overlay JSONs for
`pick_up_the_red_block_lying_in_the_drawer` recorded:

  anchors = 40
  proposals = 0
  tracklets = None

while the metrics rows still show nonzero averaged `owm_proposal_tokens` and
`owm_tracklet_tokens`.  Therefore the sidecar path is connected for the sampled
training windows, but the specific visual diagnostic frame did not receive
proposal/tracklet evidence.  Do not use that overlay as proof that SAM or
tracklets helped the red-block posterior.
```

Updated interpretation:

```text
SAM/proposal + tracklet sidecars solved an important dataflow problem: optional
evidence can enter the AQR graph and coexist with frozen PaliGemma/perception.
They did not solve task-object posterior binding in this diagnostic.

The evidence is weak for task ownership:
  aqr_task_owner_proposal_selected_count stays below 0.25 on average,
  task-owner point bridge max stays near 1e-3,
  and the saved task overlay has no proposals/tracklets.

Raw same-role overlap worsened from 0.216 at step50 to 0.962 at step150 even
though active duplicate overlap remains zero.  This means the active-file gate
and posterior file competition prevent duplicate active owners, but reserve/raw
support collapse remains present.  The current run is not sufficient evidence
that SAM fixed the red-block binding issue.
```

Production-default decision:

```text
Blind automatic SAM proposal influence is default-off from this point onward.
The proposal dataflow is retained, but it is no longer production-default
measurement evidence.

Default config:
  proposal_memory_enabled = False
  proposal_read_weight = 0.0
  proposal_point_bridge_weight = 0.0
  task_owner_proposal_bias_weight = 0.0
  task_owner_proposal_point_bias_weight = 0.0
  task_owner_proposal_point_bridge_weight = 0.0

Retained:
  sidecar loader
  proposal_* schema
  proposal overlay diagnostics
  SAM/proposal precompute scripts
  proposal AQR path behind explicit opt-in
  tracklet sidecar path

Rationale:
  Sidecar is an evidence-transport mechanism and remains useful.  Blind SAM is
  a noisy proposal source whose mask quality is not task ownership.  The
  negative A7 diagnostics show that blind SAM should not perturb production
  posterior training until prompted/reranked proposal sidecars are available.
```

Future proposal acceptance:

```text
Only re-enable proposal memory by default after a prompted/reranked proposal
sidecar passes these checks:

1. The saved task overlay contains proposal boxes/masks on the named task
   object for the same diagnostic frame, not only nonzero batch averages.
2. aqr_task_owner_proposal_selected_count is nonzero but sparse.
3. aqr_task_owner_point_bridge_max is large enough to move projected point
   likelihoods, not ~1e-3.
4. Active task-object posterior reaches the named object in overlay review.
5. raw/reserve same-role overlap does not rise after proposal evidence enters.
```

A5 tracklet resume status:

```text
The tracklet generator was patched to treat corrupt partial npz sidecars as
missing evidence and to save via atomic temp-file replacement.  Failed shards
0/1/4/6/7 were restarted in tmux after a one-segment smoke test and a sampled
zip-integrity check.

This is not a new model change.  It is a data-production correctness repair:
the sidecar generator must be restartable and must not leave half-written npz
files that break subsequent resume passes.
```

## 2026-05-18 Blind SAM Archival Decision

Issue state:

```text
Blind automatic SAM proposal memory:
  closed as rejected / archived.

Generic proposal_* sidecar schema:
  retained.

Maintained proposal source:
  contact/task/tracklet-aware sidecars, currently
  scripts/picf_contact_motion_sidecar_precompute.py with source_id=8.

Legacy implementation:
  scripts/archive/picf_sam_proposal_precompute_legacy.py
  scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py
```

Reason:

```text
Blind SAM boxes repeatedly introduced non-task wall panels, robot protrusions,
drawer sides, and visually salient but causally irrelevant fragments.  These
boxes are high-recall objectness candidates, not task ownership.  In the PICF
belief model, treating them as active proposal memory increases false transport
pressure into anchor geometry and makes overlay interpretation ambiguous.
```

Maintained invariant:

```text
proposal_* remains an optional typed observation interface.
Blind SAM is no longer an active producer for that interface.
Contact-motion proposals, task-owner sidecars, future prompted/reranked masks,
and tracklet-derived proposals may use the same proposal_* schema after visual
and metric acceptance.
```

Operator rule:

```text
Do not launch production or anchor diagnostics with blind SAM sidecar roots.
If proposal memory is enabled, the sidecar root must be explicitly documented
and inspected.  Future overlay names use sidecar_proposals; old
sam_proposals filenames are historical artifacts only.
```

## 2026-05-18 Slot/Object-Centric Reference Correction

Issue state:

```text
Resolved as documentation/process correction.
```

Problem:

```text
Several discussions over-emphasized SAM because proposal masks were being
debugged.  That is not the correct theoretical reference for PICF object
binding.  Blind SAM remains archived/off and is not the maintained slot model.
```

Maintained reference file:

```text
temp/audits_20260518/slot_object_centric_2025_2026_math_followthrough.md
```

Maintained slot principles copied into PICF:

```text
MetaSlot-style duplicate demotion and adaptive active capacity:
  implemented through active/context/reserve routing and posterior file
  competition rather than a pure image-OCL VQ codebook.

SlotAttention / QASA-style competition:
  implemented through same-role support competition and bounded active slots.

SlotContrast-style temporal same-object caution:
  retained as guarded evidence/audit.  Strong slot-JEPA/support-pred/binding
  losses remain default-zero until recycle and permutation stability are
  empirically accepted.
```

Current acceptance gate:

```text
The next dual-GPU 1000-step mask-sidecar anchor-only run must show:
  proposal_mask_* fields read;
  proposal and tracklet tokens nonzero;
  active same-role object-core overlap controlled;
  posterior recycle not saturated;
  anchor overlays covering task objects rather than only gripper/background.
```

## Issue: Proposal Mask Selected But Physical Anchor Misses It

Status:

```text
fixed in local code by proposal reference-anchor transport;
pending A7 diagnostic acceptance.
```

Observed evidence:

```text
The A7 mask-sidecar step-100 run selected task/contact proposals:
  aqr_proposal_support_max = 0.8833
  aqr_task_owner_proposal_score_max = 0.7000

But proposal-to-point transport was too weak:
  aqr_proposal_point_bridge_max = 0.0271
  aqr_task_owner_point_bridge_max = 0.0223
```

Why this matters:

```text
The green sidecar proposal/mask can be visually correct while no physical
anchor is actually born or transported into that geometry.  In that case the
failure is not lack of proposal evidence and not same-role active collapse;
it is missing reference transport from selected proposal to physical rows.
```

Mathematical repair:

```text
Use selected proposal masks as soft reference measurements:

  proposal score -> proposal mask/box -> point prior -> physical rows

The seeded rows still compete through AQR and are still corrected by posterior.
The sidecar remains evidence, not truth.
```

Code/dataflow keys:

```text
src/openpi/picf/core/config.py:
  proposal_anchor_seed_enabled
  proposal_anchor_seed_rows
  proposal_anchor_seed_weight
  proposal_anchor_seed_token_weight
  proposal_anchor_seed_point_topk
  proposal_anchor_seed_point_power

src/openpi/picf/core/pipeline.py:
  _proposal_anchor_seed_transport
  PicfAnchorPriorGraphState.proposal_anchor_seed_priors
  PicfAnchorPriorGraphState.proposal_anchor_seed_assignment

scripts/picf_core_train.py:
  --proposal-anchor-seed-enabled
  aqr_proposal_anchor_seed_* metrics
```

Acceptance:

```text
If proposal_anchor_seed_* metrics are nonzero and overlays still miss the task
proposal, the remaining fault is proposal-to-point calibration or sidecar
quality.  If proposal_anchor_seed_* metrics are zero, the fault is dataflow or
configuration, not learning.
```

## Issue: Current Router Still Lacks A Primary Object-Explanation Objective

Status:

```text
open vNext design issue;
not solved by current AQR/proposal/sidecar routing alone.
```

Maintained deployment plan:

```text
docs/PICF_AQR_OWM_OBJECT_EXPLANATION_DEPLOYMENT_PLAN_20260518_TEMP.md
```

Problem:

```text
Current PICF-AQR-OWM has a coherent posterior-centered belief router, but it
does not yet train each object file to explain dense evidence with
object/background masks.  Mature slot/OCL systems rely on this explanation loop
to make object decomposition stable:

  slot -> token/object mask -> feature/geometry/contact explanation

PICF currently has routing priors, competition, active/context/reserve gates,
posterior file competition, proposal seed transport, and binding signatures.
Those are necessary, but they are downstream guards; they do not force dense
visual/point/contact/proposal evidence to be explained by distinct object files.
```

Why this is not a small loss-weight bug:

```text
loss_anchor_pv can rise while action loss improves because action can exploit
task-relevant evidence without requiring stable object partitioning.

same-role overlap can be demoted downstream, but duplicate anchors are not
made intrinsically wasteful unless duplicate object explanations fail to reduce
an explanation likelihood.

gray/reserve anchors can be made action-invisible, but background/no-object
capacity is not trained as an explicit object/background partition.
```

Required repair:

```text
Add an Object-Explanation Measurement Layer (OEML):

  dense typed evidence
    -> per-slot object/background masks
    -> visual/temporal feature explanation
    -> point/geometry likelihood
    -> contact/tactile likelihood
    -> object-quality / duplicate / background terms
    -> AQR/posterior measurement rows

Posterior remains authoritative.  Proposals/contact/tracklets remain weak
measurement evidence, not labels or action targets.
```

Current boundary:

```text
Do not claim this issue is fixed until OEML contracts, runtime, losses, and
audits exist and a frozen 1000-step anchor/object validation passes.
Existing runs without OEML are valid router baselines only.
```

## Issue: Object Candidate Reaches Model But Is Cloned Across Raw Rows

Status:

```text
partially fixed by object-candidate top-k ownership and soft row capacity;
follow-up active-object loss-scope repair is now implemented locally and needs
the next 200-step frozen action/PaliGemma diagnostic.
```

Observed evidence from the A7 v2 object-candidate diagnostic:

```text
fixed part:
  aqr_object_candidate_coverage_mean rose to ~0.99
  aqr_object_candidate_background_mean fell to ~0.005

remaining failure:
  aqr_object_candidate_duplicate_overlap_max saturated near 1.0
  posterior_file_competition_duplicate_overlap_max saturated near 0.98
  aqr_same_role_support_overlap_max rose from ~0.16 to ~0.87
  loss_anchor_pv / denoising / routing / cycle all worsened after early improvement
```

Root cause:

```text
The v2 assignment conserved candidate-column mass but did not constrain row
capacity.  Several same-role raw anchors could all claim the same sidecar
object candidate.  Posterior file competition then demoted some files later,
but the raw AQR losses had already received duplicate object support.
```

Mathematical repair:

```text
For sidecar object candidates only:

  1. keep top-k physical row owners per candidate, default k=1;
  2. optionally apply Sinkhorn-style row-capacity scaling;
  3. keep explicit background/no-object residual;
  4. do not prune dense V-JEPA/point/tactile tokens.

This copies the SlotAttention/MetaSlot object-candidate invariant: a measurement
candidate is explained by one or a small mixture of object slots plus
background, not cloned across many same-role raw rows.
```

Code/dataflow keys:

```text
src/openpi/picf/core/config.py:
  object_candidate_max_rows_per_candidate
  object_candidate_row_capacity
  object_candidate_row_capacity_iters

src/openpi/picf/core/pipeline.py:
  _proposal_object_candidate_assignment(...)
  top-k candidate ownership
  soft row capacity before final slot/background normalization

scripts/picf_object_candidate_slot_binding_audit.py:
  candidate_top1_suppresses_raw_same_candidate_clones
  row_capacity_limits_one_slot_from_eating_all_candidates

docs/PICF_AQR_OWM_OBJECT_CANDIDATE_SLOT_BINDING_20260519_TEMP.md:
  current canonical math/dataflow record
```

Acceptance:

```text
A7 200-step diagnostic result:
  PASS:
    object_candidate coverage remains high;
    object_candidate/background does not regress to background collapse;
    object_candidate_duplicate_overlap stays 0.0;
    posterior_file_competition_active_duplicate_overlap stays 0.0;
    active same-role support/object-core overlap stays low.

  FAIL / STILL OPEN:
    raw same-role support/object-core overlap rises;
    raw posterior_file_competition_duplicate_overlap remains saturated;
    loss_anchor_pv / denoising / routing remain worse than the step-50 point.

Conclusion:
  The upstream object-candidate cloning bug is fixed.  The remaining issue is
  raw/reserve duplicate files and how they should participate in losses and
  diagnostics.  Do not use raw duplicate saturation alone to reject the active
  path, but do not claim raw object partitioning is solved.
```

Follow-up repair now in local code:

```text
src/openpi/picf/core/training.py:
  anchor_pv_active_object_gate_only=True
  aqr_denoising_active_object_only=True
  object_explanation_active_object_only=True
  _active_object_row_weight(...)

Meaning:
  object-level losses train active object files only;
  reserve/context/no-object rows remain visible as raw telemetry and context,
  but no longer act as failed object rows inside anchor-PV, support denoising,
  or object-explanation duplicate/feature terms.

Required validation:
  run a fresh 200-step frozen action/PaliGemma diagnostic and check whether
  loss_anchor_pv / loss_aqr_denoising / loss_mapg_routing stop worsening while
  active overlap remains low.
```

Validation result, `picf_a7_active_object_scope_anchor200_20260519`:

```text
PASS:
  object_candidate_coverage_mean remains high:
    0.8635 -> 0.9272 -> 0.9732 -> 0.9670
  object_candidate_background_mean remains low:
    0.1365 -> 0.0728 -> 0.0268 -> 0.0330
  object_candidate_duplicate_overlap_max stays 0.0
  posterior_file_competition_active_duplicate_overlap_max stays 0.0
  active_same_role_support_overlap_max remains far below the old collapse band:
    0.0145 -> 0.0215 -> 0.0508 -> 0.1115
  active_same_role_object_core_overlap_max remains low:
    0.1363 -> 0.0644 -> 0.0127 -> 0.0203

FAIL / STILL OPEN:
  raw same_role_support_overlap_max still saturates:
    0.1545 -> 0.5125 -> 0.9854 -> 0.9973
  raw same_role_object_core_overlap_max still saturates:
    0.1666 -> 0.6722 -> 0.9341 -> 0.9393
  raw posterior_file_competition_duplicate_overlap_max remains high:
    0.9460 -> 0.9782 -> 0.9830 -> 0.9770
  loss_anchor_pv rises and plateaus:
    0.7839 -> 1.1843 -> 1.1729 -> 1.1778
  loss_aqr_denoising rises and stays high:
    1.3174 -> 1.9060 -> 1.9695 -> 1.8828
  loss_mapg_routing drifts upward:
    0.6093 -> 0.6325 -> 0.6758 -> 0.7078
```

Updated conclusion:

```text
The active-object loss-scope repair is correct and should be kept, but it does
not fully solve the anchor-PV / denoising / routing drift.  The remaining
problem is now narrower:

  active object rows are correctly selected and do not duplicate candidates,
  but the active rows still receive inconsistent anchor-PV / denoising targets.

The next fix should not add another same-role overlap scalar.  It should split
object-level anchor-PV from global PV weak coverage, and it should either keep
AQR denoising disabled by default or restrict denoising to confirmed
object-candidate/proposal/point active rows.
```

V5 implementation status:

```text
implemented locally:
  anchor_pv_object_gate_floor=0.0
  anchor_pv_object_normalize_by_object_mass=True
  aqr_denoising_confirmed_object_only=True
  aqr_denoising_confirmation_threshold=0.05
  _confirmed_object_row_weight(...)

meaning:
  object-PV is now a true object-edge expectation;
  dense/global PV is isolated in pv_weak;
  denoising cannot turn an active-but-unconfirmed row into a teacher.

required validation:
  repeat the 200-step frozen action/PaliGemma diagnostic.
```

V5 validation snapshot, A7 step 50:

```text
run:
  picf_a7_v5_object_pv_split_anchor200_20260519

healthy:
  candidate coverage mean                  0.8477
  candidate duplicate overlap max          0.0000
  active same-role support overlap max     0.0121
  active object-core overlap max           0.1137
  proposal tokens                          1.94
  tracklet tokens                          75.16

not yet healthy:
  loss_anchor_pv                           5.3087
  loss_aqr_denoising                       1.4504
  posterior_identity_switch_rate           0.6400
  preclip_grad_norm                        18.2694
```

Interpretation:

```text
V5 has not reproduced the active-slot collapse.  It has also exposed the real
object-edge alignment difficulty because object-PV is no longer diluted by
background/dense edges.  Do not treat the step-50 high anchor-PV value as a
regression by itself; require the 100/150/200 trend before accepting or
rejecting the repair.
```

V5 first 100-step decision, invalidated by launch-script override:

```text
NOT A VALID PURE V5 TEST.

Active rows:
  still healthy enough for the diagnostic:
    active same-role support overlap 0.0121 -> 0.0569
    active object-core overlap       0.1137 -> 0.2149
    candidate duplicate overlap      0.0000 -> 0.0000

Object-edge losses:
  failing:
    loss_anchor_pv          5.3087 -> 8.5283
    loss_aqr_denoising      1.4504 -> 2.0182
    preclip_grad_norm       18.27  -> 183.21
```

Root-cause status:

```text
follow-through bug:
  run_a7_object_candidate_anchor1000_20260519.sh still passed
  --anchor-pv-object-gate-floor 0.25.

meaning:
  the stale script reintroduced the dense/background floor that V5 is designed
  to remove.
```

Required next action:

```text
Patch the launch script and repeat the 200-step diagnostic with an explicit
floor of 0.0 plus explicit object-mass normalization and confirmed-only
denoising flags.
```

Pure V5 step-50 evidence:

```text
run:
  picf_a7_v5_pure_object_pv_split_anchor200_20260519

launch flags verified:
  --anchor-pv-object-gate-floor 0.0
  --anchor-pv-object-normalize-by-object-mass
  --aqr-denoising-confirmed-object-only

result versus stale-floor run at step 50:
  loss_total             1.6550 -> 1.0692
  loss_alignment         1.5800 -> 0.9942
  loss_anchor_pv         5.3087 -> 2.9648
  loss_aqr_denoising     1.4504 -> 1.3502
  preclip_grad_norm      18.27  -> 9.21
  active support overlap 0.0121 -> 0.0164
  active object overlap  0.1137 -> 0.1054
```

Status:

```text
The launch-script follow-through fix is validated at step 50.  Continue to
100/150/200 before accepting this as the training profile.
```

Pure V5 step-100 status:

```text
PARTIAL FIX, NOT FINAL.

Compared with stale-floor run:
  much better loss_total / anchor_pv / gradient at step 50.

Within pure V5:
  loss_anchor_pv       2.9648 -> 3.7491
  loss_aqr_denoising   1.3502 -> 1.9225
  loss_mapg_routing    0.5908 -> 0.6852
  preclip_grad_norm    9.21   -> 61.48
  active overlap       remains much lower than raw overlap.
```

Open issue:

```text
The active object selector is no longer the main blocker.  The remaining
blocker is auxiliary teacher pressure: self-denoising / routing / raw support
terms can still amplify unstable support peaks after warmup.
```

2026-05-19 V6 update:

```text
implemented:
  distributional object-PV replaces dense-edge object-PV BCE.

math:
  for active/confirmed object row j:
    v_hat_j = normalize(p_j C)
    p_hat_j = normalize(v_j C^T)
    L_anchor_pv = mean_j 0.5 * [JS(v_j, v_hat_j) + JS(p_j, p_hat_j)]

why:
  the previous loss still compared object rows to a dense projective edge field;
  V6 compares each object row's point support and visual support to each other.

verification:
  py_compile PASS
  picf_object_candidate_slot_binding_audit PASS
  picf_anchor_pv_object_gate_audit PASS
  verify_picf_owm_contract PASS

status:
  ready for a new A7 200-step diagnostic; not yet accepted by runtime trend.
```

V6 200-step runtime verdict:

```text
run:
  picf_a7_v6_distribution_object_pv_anchor200_20260519

accepted:
  yes, for the specific object-PV root cause.

evidence:
  loss_total         0.4732 -> 0.3563
  loss_alignment     0.3982 -> 0.2813
  loss_alignment_raw 0.2622 -> 0.1952
  loss_anchor_pv     0.5684 -> 0.5516
  loss_pv_weak       6.0047 -> 2.8625
  active support ovl 0.0125 -> 0.0986
  active object ovl  0.1286 -> 0.0620
  candidate dup ovl  0.0000 -> 0.0000

remaining:
  raw support overlap rises to 0.8881 because raw metrics include reserve /
  context / background rows.  This remains a diagnostic, not the object-binding
  acceptance criterion.

next:
  use V6 for the next long/full run; keep aqr_denoising / mapg_routing as
  diagnostics unless separately accepted.
```

V6 post-diagnostic boundary, 2026-05-19:

```text
local report:
  /tmp/picf_a7_v6_distribution_object_pv_anchor200_20260519/report.json

paper/runtime note:
  temp/audits_20260519/v6_slot_paper_runtime_followthrough.md

closed for current issue:
  dense-edge object-PV mismatch.

not closed:
  posterior active-file continuity.

why:
  active support overlap remains below 0.10 and active object overlap falls to
  0.0620 by step 200, while raw overlap rises because reserve/context rows are
  included.  The diagnostic report therefore rejects another raw-overlap patch.

next if we modify code:
  build a posterior active-file continuity audit before changing losses or
  adding another module.

next if we train:
  use the V6 profile; keep denoising/routing/slot-JEPA/support-pred as logged
  diagnostics, not active losses.
```

2026-05-19 object-pull role-leakage follow-up:

```text
symptom:
  In the object-pull-only probe, blue role-0 effector anchors can move onto the
  task object while orange role-1 object anchors are less cleanly centered.

root cause:
  The production task-owner proposal/point routes already target role-1
  physical rows, but the diagnostic object-pull loss confirmed rows using
  generic proposal_priors.  If a role-0 effector row already attended to a
  proposal, the diagnostic loss could pull that effector row further onto the
  object, creating effector/object role leakage.

fix:
  Add anchor_object_pull_allowed_roles, default (1,2).  The diagnostic sidecar
  object/contact pull now applies to task-object and interaction/contact rows,
  but not effector rows, unless explicitly overridden.  This keeps tactile
  contact evidence able to pull the contacted object through role-2 support
  without letting role-0 gripper rows become object files.

status:
  code patched locally; requires a rerun of the object-pull diagnostic before
  accepting the probe.

next visualization:
  Produce role-separated GIFs: mask-only, mask+active anchors, mask+gray
  anchors, active-only anchors, with-gray anchors, and proposal-box view.
```

Visualization implementation:

```text
scripts/picf_core_train.py now writes mask_only, mask_active, and
mask_with_gray PNG variants in addition to with_gray, active_only, and
sidecar_proposals whenever proposal_mask_xy / proposal_mask_weights /
proposal_mask_offsets are present.

scripts/picf_anchor_overlay_make_gifs.py compiles each variant into a GIF and
also writes combined_6view.gif for one-file inspection of all six views.
This is diagnostic-only; it does not alter forward, losses, or checkpoint
semantics.
```

2026-05-19 object/contact dual-role candidate repair:

```text
symptom:
  The role-scoped object-pull diagnostic correctly excluded role-0 effector
  rows from the loss, but the runtime candidate assignment/seed path could
  still explain each sidecar object candidate with only role-1 task-object
  rows.  This left no clean role-2 contact bridge for tactile/contact evidence
  to attach to the same object.

repair:
  Add object_candidate_eligible_roles=(1,2) and make proposal/visual/point
  sidecar candidate routing use _object_candidate_physical_rows(...).  The
  default candidate capacity is now 2: one role-1 object owner plus one role-2
  contact bridge.  Role 0 is explicitly skipped.

why this is not a gripper-role deletion:
  Role 0 still carries effector/proprio/tactile context.  It is only forbidden
  from becoming an object-candidate owner.  Contact evidence should route via
  role 2 into the role-1 object file, not be owned by the gripper row.

verification:
  scripts/picf_object_candidate_slot_binding_audit.py now checks the dual-role
  candidate contract, role-0 exclusion, CLI configurability, and the top-2
  object/contact bridge assignment math.

next:
  rerun the 1000-step anchor-object-pull probe and inspect combined_6view.gif at
  step 100/200.  Acceptance requires the mask object candidate to be reached by
  role-1/role-2 rows without role-0 blue anchors becoming the object owner.
```

2026-05-19 dual-role object/contact probe, first evidence:

```text
run:
  picf_a7_dualrole_object_contact_pull_1000_20260519

launcher:
  run_a7_anchor_object_pull_only_1000_20260519.sh

active settings:
  trainable_scope=anchor_only
  action losses = 0
  semantic / V-JEPA / Sonata / AnyTouch backbones frozen
  lambda_anchor_object_pull=1
  anchor_object_pull_allowed_roles=(1,2)
  object_candidate_eligible_roles=(1,2)
  object_candidate_max_rows_per_candidate=2
  sidecar root = /mnt/picf_sidecars/contact_motion_mask_1000_20260518

step 50:
  loss_anchor_object_pull               0.5643
  active same-role support overlap      0.0212
  active same-role object-core overlap  0.0751
  raw same-role support overlap         0.2322
  object candidate assigned row count   2.4950
  object candidate assignment max       0.5335
  object candidate coverage mean        0.9518
  object candidate background mean      0.0248
  active file duplicate overlap         0.0000
  posterior recycle rate                0.0789

step 100:
  loss_anchor_object_pull               0.3054
  active same-role support overlap      0.0588
  active same-role object-core overlap  0.1513
  raw same-role support overlap         0.7229
  raw same-role object-core overlap     0.5668
  object candidate assigned row count   2.7150
  object candidate assignment max       0.5215
  object candidate coverage mean        0.9587
  object candidate background mean      0.0038
  active file duplicate overlap         0.0000
  posterior recycle rate                0.0874
  grad_norm                             5.0000
  grad_clip_applied                     true

visual artifacts:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_dualrole_object_contact_pull_1000_20260519/anchor_overlays/gifs/combined_6view.gif

interpretation:
  The repair is active.  The assigned-row count is near 2-3 rather than 1,
  which means a sidecar object/contact candidate is no longer forced through a
  single task-object row.  Coverage remains high and background mass is low, so
  the candidate is not being discarded.

  Active same-role overlap remains low through step 100, and active file
  duplicate overlap stays 0.0.  This is the health metric for object files.

  Raw same-role overlap rises by step 100.  This must not be conflated with the
  old failure without checking the role-aware views: raw overlap now includes
  reserve/context rows and also counts the intentional role-1 object owner plus
  role-2 contact bridge sharing one candidate.  It remains a warning metric,
  not an acceptance metric by itself.

remaining checks:
  Inspect step 200 overlays and confirm that the orange role-1 owner and role-2
  bridge are spatially on the sidecar mask rather than only numerically
  assigned.  If visual binding remains off-mask despite falling
  loss_anchor_object_pull, the issue is in anchor geometry transport / query
  parametrization rather than in candidate assignment.
```

Step-200 follow-up:

```text
step 150:
  loss_anchor_object_pull               0.4249
  active same-role support overlap      0.0548
  active same-role object-core overlap  0.0462
  raw same-role support overlap         0.9461
  object candidate coverage mean        0.9513
  object candidate background mean      0.0003
  active file duplicate overlap         0.0000

step 200:
  loss_anchor_object_pull               0.4752
  active same-role support overlap      0.1320
  active same-role object-core overlap  0.0126
  raw same-role support overlap         0.9883
  raw same-role object-core overlap     0.9239
  object candidate assigned row count   2.4900
  object candidate assignment max       0.5350
  object candidate coverage mean        0.9821
  object candidate background mean      0.0004
  posterior recycle rate                0.0009
  active file duplicate overlap         0.0000

visual:
  step_000200__slide_the_door_to_the_left__mask_active.png shows active anchors
  in the sidecar mask neighborhood, but role-1/orange owner rows are still not
  the sole stable object owner.  Several orange anchors remain near the gripper
  side while role-2/context rows explain the sidecar candidate.

updated diagnosis:
  The dual-role candidate repair fixed candidate discard and the role-0
  effector leak at the dataflow level.  It did not fully solve geometry owner
  selection.  The remaining problem is not that sidecar evidence is absent:
  coverage is >0.95 and background mass is near zero.  The remaining problem is
  that the query/update geometry path can explain the mask with a contact bridge
  or reserve rows without making a clean role-1 task-object file the stable
  owner.

next code direction:
  Do not add SAM back.  Do not delete role-0 gripper rows.  The clean next
  repair should target owner transport: when a high-quality object/contact
  candidate is confirmed, the role-1 owner row must receive an explicit,
  bounded geometry transport/update toward the candidate mask centroid or
  point/mask barycenter, while role-2 remains a contact bridge and role-0 stays
  effector context.  This is a geometry ownership update, not another
  proposal-source loss.
```

Implementation note after step-200 diagnosis:

```text
repair:
  bounded role-1 owner geometry transport

math:
  O = {j | role_j=1 and query_type_j=physical}
  o(p) = argmax_{j in O} E_{j,p}
  T_{o(p),p} = max(A_{o(p),p}, rho * coverage_p)
  P_owner[j] = normalize(sum_p T[j,p] * candidate_mask_to_point[p])

code:
  config:
    object_candidate_owner_transport_enabled=True
    object_candidate_owner_roles=(1,)
    object_candidate_owner_min_share=0.65
    object_candidate_owner_point_mix=0.85
  graph:
    object_candidate_owner_assignment
    object_candidate_owner_point_priors
  training:
    object-pull prioritizes owner point priors
    confirmed-object row gating includes owner assignment / owner point priors

why this is not a patch-on-patch:
  The previous run proved the candidate evidence exists and is not discarded.
  The missing edge is candidate -> object-owner geometry.  This repair adds
  exactly that edge while preserving the role split:
    role 0 = effector context
    role 1 = task object owner
    role 2 = contact bridge

next run:
  picf_a7_owner_transport_anchor1000_20260519
```

## Owner Transport v2 Runtime Evidence

Run:

```text
picf_a7_owner_transport_anchor1000_v2_20260519
```

Configuration:

```text
trainable_scope=anchor_only
action/semantic/perception losses frozen
object_candidate_eligible_roles=(1,2)
object_candidate_owner_transport_enabled=True
object_candidate_owner_roles=(1,)
object_candidate_owner_min_share=0.65
object_candidate_owner_point_mix=0.85
```

Step 50:

```text
loss_anchor_object_pull                         0.3982
active same-role support overlap               0.0204
active same-role object-core overlap           0.0763
raw same-role support overlap                  0.1914
raw same-role object-core overlap              0.2509
object candidate assigned row count            2.4950
object candidate coverage mean                 0.9512
object candidate background mean               0.0254
owner assignment max                           0.6601
owner row count                                1.2800
owner point row count                          1.2800
posterior active file duplicate overlap        0.0000
```

Step 100:

```text
loss_anchor_object_pull                         0.2204
active same-role support overlap               0.0244
active same-role object-core overlap           0.0578
raw same-role support overlap                  0.7116
raw same-role object-core overlap              0.7392
object candidate assigned row count            2.7350
object candidate coverage mean                 0.9621
object candidate background mean               0.0004
owner assignment max                           0.6627
owner row count                                1.3950
owner point row count                          1.3950
posterior active file duplicate overlap        0.0000
```

Overlay/JSON check at step 100 for `push_the_switch_downwards`:

```text
sidecar proposal center px: (151.8, 31.0)
role-1 active graph owner:  (152.6, 34.5)
distance:                   ~3.5 px
```

Interpretation:

```text
The v2 owner-transport repair is live and materially different from the prior
run.  The candidate mask is not discarded, the role-1 owner receives explicit
owner mass and transported point priors, and the active role-1 owner is visibly
inside the sidecar mask/box at step 100.

The remaining high raw overlap is mostly inactive/reserve/context accounting;
the acceptance metric for object files is active same-role overlap and active
file duplicate overlap, both of which remain low at step 100.  Therefore this
fix addresses the specific failure where the mask existed but role-1 did not
become the spatial owner.

Still not concluded from this probe alone:
  Long-horizon cotrain stability with action/PaliGemma unfrozen.
  Posterior identity-switch behavior after action gradients return.
  Whether every CALVIN task family gets equally clean owner transport.
```

## 2026-05-19 Issue Update: Blue/Effector Competition in Object-Pull Probe

Status: active fix deployed locally; remote A7 clean probe pending.

Observed issue:

```text
The v2 owner-transport run showed the sidecar mask and role-1 owner transport
were numerically live, but overlay inspection still showed blue/role-0 rows
occupying the manipulated object in some frames.  This made the previous probe
ambiguous: it did not cleanly answer whether role-1/orange object anchors can
be pulled to the sidecar mask because the structured layout still created an
effector/context owner row.
```

Root cause isolated in code:

```text
1. _mapg_anchor_roles() used the structured layout by default, which always
   creates at least one role-0 effector row.
2. The probe still had a per-role active minimum, so role-0 could remain
   visible/active even when object-pull was intended to test role-1 ownership.
3. Tactile routing used the old convention where role-1 rows were blocked from
   tactile tokens, which is inverted for object binding: contact/tactile should
   attach to the contacted object owner, not to a separate gripper owner.
```

Implemented local repair:

```text
config:
  tactile_attach_to_object_owner=True
  aqr_role_layout="structured" by default for compatibility

pipeline:
  aqr_role_layout=object_only returns all physical rows as role=1.
  aqr_role_layout=no_effector returns role 1/2/3 rows without role 0.
  tactile_attach_to_object_owner=True lets role-1 rows read tactile tokens and
  blocks tactile from non-object rows.  This applies to public/fused read,
  AQR graph tactile reader bias, and MAPG tactile seed priors.

training CLI:
  --aqr-role-layout {structured,no_effector,object_contact_context,object_only,object}
  --tactile-attach-to-object-owner / --no-tactile-attach-to-object-owner
  validator now permits zero effector/task-effector counts only for the
  no-effector layouts.  structured layout keeps the old >=1 checks.

probe:
  run_a7_object_owner_only_pull_probe_1000_20260519.sh
  uses object_only layout, zero effector counts, role-1-only object candidates,
  role-1-only object-pull, and tactile-to-object-owner routing.
```

Acceptance:

```text
No role-0/blue object owner in overlay JSON.
loss_anchor_object_pull decreases by step 50/100.
role-1/orange active owner moves into/near the sidecar mask.
If this fails, the remaining fault is anchor geometry/query capacity or
sidecar projection, not blue-role competition.
```

## 2026-05-20 Issue Update: Context-Core Dedup Was Necessary But Not Sufficient

Status: partially closed.  Support-aware context dedup is implemented and A7
`picf_a7_context_support_dedup_300_20260520` completed 300 steps.  It fixes
the old downstream duplicate-context rebound but exposes a separate remaining
owner-localization issue.

What the previous A7 run proved:

```text
picf_a7_context_dedup_300_20260520

step 50/100:
  context rows dropped from about 22 to about 6.
  downstream/context same-role support overlap stayed around 0.25-0.41.
  active same-role support overlap stayed below 0.02.

step 150:
  active same-role support overlap still stayed low.
  context object-core overlap stayed low (~0.26).
  downstream object-core overlap stayed low (~0.28).
  context/downstream full-support overlap rebounded to ~0.83-0.84.
  grad clipping triggered.
```

Conclusion:

```text
This is not active posterior collapse.
This is not a missing SAM/proposal module.
This is not an action-loss conflict, because the run has zero action loss.

The remaining engineering fault is narrower:
  the context selector suppressed duplicate object-core owners but did not
  suppress diffuse duplicate support rows before the control graph.
```

Implemented repair:

```text
config:
  aqr_context_slot_self_support_overlap_enabled=True
  aqr_context_slot_self_support_overlap_threshold=0.70

pipeline:
  context-context greedy selection now rejects a candidate if either:
    object-core/geometry overlap > aqr_context_slot_self_overlap_threshold
    full-support overlap > aqr_context_slot_self_support_overlap_threshold

tests:
  added a regression where two rows have distinct proposal/object-core evidence
  but identical diffuse visual support.  The old selector keeps both; the new
  support-aware selector suppresses the lower-scoring duplicate.
```

Acceptance for the active validation:

```text
By step 100:
  aqr_context_same_role_support_overlap_max should stay materially below the
  old 0.83+ rebound.
  aqr_downstream_same_role_support_overlap_max should stay below raw overlap.
  aqr_active_same_role_support_overlap_max should remain low.
  loss_object_explanation_point should stay bounded and not return to the old
  unbounded 10-30 range.

If this fails, the next issue is upstream support-map formation, not context
dedup visibility.
```

Observed active validation:

```text
picf_a7_context_support_dedup_300_20260520

step 50:
  active/context/downstream support overlap max = 0.0148 / 0.2982 / 0.3382
  raw same-role support overlap max = 0.4713
  loss_object_explanation_point = 0.7746
  grad_clip_applied = false

step 100:
  active/context/downstream support overlap max = 0.0239 / 0.2433 / 0.2692
  raw same-role support overlap max = 0.8454
  loss_object_explanation_point = 2.6901
  grad_clip_applied = false

step 150:
  active/context/downstream support overlap max = 0.0093 / 0.2120 / 0.2521
  raw same-role support overlap max = 0.9951
  loss_object_explanation_point = 4.4468
  grad_clip_applied = false

step 200:
  active/context/downstream support overlap max = 0.0390 / 0.3786 / 0.4614
  raw same-role support overlap max = 0.9995
  loss_object_explanation_point = 4.2610
  grad_clip_applied = false

step 250:
  active/context/downstream support overlap max = 0.0923 / 0.4327 / 0.6175
  raw same-role support overlap max = 0.9999
  loss_object_explanation_point = 5.4936
  grad_clip_applied = true

step 300:
  active/context/downstream support overlap max = 0.0490 / 0.5121 / 0.6277
  raw same-role support overlap max = 0.9999
  loss_object_explanation_point = 6.0854
  grad_clip_applied = true
```

Current conclusion:

```text
The old step-150 failure did not reproduce.  Raw reserve overlap remains high,
which is expected for an overcomplete fixed-capacity query bank, but active,
context, and downstream rows remain bounded below the old 0.83-0.84 band.

However, this is not a full binding closure.  The point/object-pull losses rise
late and grad clipping appears after step 250.  Overlay review shows active
graph anchors can remain near but not cleanly centered on the sidecar mask, and
posterior active files can stay farther away.  The next issue is owner
transport / active-quality calibration, not another generic context dedup rule.
```

## 2026-05-20 Issue Update: Owner Measurement Transport Fixed Active Localization, Not Downstream Reuse

Status: partially closed.  The accepted sidecar/contact owner candidate is now
promoted to a first-class geometry measurement and transported into graph and
posterior owner updates.

Run:

```text
picf_a7_owner_measurement_transport_300_20260520
```

Implemented repair:

```text
PicfAnchorPriorGraphState.object_candidate_owner_x/S
_object_candidate_owner_geometry()
object_candidate_owner_geometry_mix
posterior_owner_transport_candidate_geometry_mix
aqr_slot_quality_owner_active_floor
```

Observed:

```text
step 50:
  owner geometry active distance ~= 0.00020 m
  active/context/downstream support overlap max = 0.0018 / 0.2502 / 0.3642
  loss_anchor_object_pull = 0.3229
  loss_object_explanation_point = 2.1870

step 100:
  owner geometry active distance ~= 0.00022 m
  active/context/downstream support overlap max = 0.0024 / 0.3577 / 0.4063
  loss_anchor_object_pull = 0.2827
  loss_object_explanation_point = 2.1716

step 150:
  owner geometry active distance ~= 0.00028 m
  active/context/downstream support overlap max = 0.0422 / 0.4623 / 0.6048
  loss_anchor_object_pull = 0.2638
  loss_object_explanation_point = 2.8618

step 200:
  owner geometry active distance ~= 0.00047 m
  active/context/downstream support overlap max = 0.1131 / 0.4986 / 0.7659
  loss_anchor_object_pull = 0.5630
  loss_object_explanation_point = 3.1591
```

Overlay inspection:

```text
step 50 push_the_switch_downwards:
  active graph/posterior owner sits on the sidecar mask.

step 100 push_the_switch_downwards:
  active graph/posterior owner still sits on the sidecar mask.

step 150 open_the_drawer:
  active graph/posterior owner sits on the drawer sidecar mask.

step 200 slide_the_door_to_the_left:
  active graph/posterior owner sits on the sidecar mask.
```

Conclusion:

```text
Fixed:
  the active owner localization path.  The previous "mask exists but orange
  owner cannot be pulled to it" diagnosis is no longer true for the inspected
  50/100/150/200 overlays.

Still open:
  downstream/context rows continue to reuse the same object support after the
  active owner is selected.  This causes raw/context/downstream overlap and
  object-explanation point loss to rise even though the selected active owner
  is correct.

Next root repair:
  duplicate demotion must be applied to downstream/context weights for rows
  that explain an already-owned sidecar object.  It must preserve dense
  context tokens as low-weight background/context evidence, but prevent them
  from acting as additional object owners in the control graph.
```

## 2026-05-20 Issue Update: Active-Context Support Dedup Implemented Locally

Status: implemented locally; A7 revalidation pending.

Root cause:

```text
The owner-geometry repair made the selected active owner correct, but
downstream/context rows could still reuse the same full support distribution.
The earlier duplicate filter checked object-core/geometry overlap; it missed
diffuse full-support duplicates that should be reserve/background context, not
second object owners.
```

Repair:

```text
aqr_context_slot_active_support_overlap_enabled
aqr_context_slot_active_support_overlap_threshold
```

Rule:

```text
If a non-active context candidate's full support overlaps an already-active
owner above threshold, the row remains in the dense graph state but loses
downstream owner weight.
```

Why this is not an ad-hoc patch:

```text
It completes the same slot-competition invariant already used for active rows
and context-context deduplication.  It adds no new loss and deletes no dense
tokens.  It only prevents duplicate explanations of an already-owned object
from entering the action-visible owner set.
```

Local verifier:

```text
test_context_slot_downstream_weight_deduplicates_active_diffuse_support
```

The test deliberately creates a failure case where object-core overlap is low
but full visual support overlap is identical.  The duplicate context row must
be suppressed; an unrelated context row must remain available.

Remote validation started:

```text
picf_a7_active_support_dedup_300_20260520
```

Decision rule:

```text
At step 100/200/300, judge active/context/downstream overlap separately.  Do
not reject on raw reserve overlap alone.  Reject if downstream-visible overlap
rebounds like picf_a7_owner_measurement_transport_300_20260520 while
loss_object_explanation_point continues rising.
```

Interim validation through step 200:

```text
step 50:
  active/context/downstream/raw support overlap max = 0.0017 / 0.2398 / 0.3327 / 0.4620
  loss_anchor_object_pull = 0.3701
  loss_object_explanation_point = 2.2234

step 100:
  active/context/downstream/raw support overlap max = 0.0045 / 0.3399 / 0.3779 / 0.6498
  loss_anchor_object_pull = 0.2629
  loss_object_explanation_point = 2.4188

step 150:
  active/context/downstream/raw support overlap max = 0.0182 / 0.4342 / 0.4569 / 0.9017
  loss_anchor_object_pull = 0.2479
  loss_object_explanation_point = 2.5368

step 200:
  active/context/downstream/raw support overlap max = 0.0287 / 0.4842 / 0.5005 / 0.9669
  loss_anchor_object_pull = 0.3043
  loss_object_explanation_point = 2.1444

step 250:
  active/context/downstream/raw support overlap max = 0.0228 / 0.4786 / 0.5019 / 0.9855
  loss_anchor_object_pull = 0.2443
  loss_object_explanation_point = 2.2171

step 300:
  active/context/downstream/raw support overlap max = 0.0367 / 0.4819 / 0.5080 / 0.9928
  loss_anchor_object_pull = 0.2668
  loss_object_explanation_point = 1.7866
```

Step-200 comparison to the previous owner-measurement run:

```text
downstream overlap:       0.7659 -> 0.5005
loss_anchor_object_pull:  0.5630 -> 0.3043
loss_object_expl_point:   3.1591 -> 2.1444
```

Current issue status:

```text
Fixed for short validation:
  active owner localization and the old downstream/context clone rebound.

Still monitored in long runs:
  raw reserve overlap remains high.  This is acceptable only while downstream
  overlap stays bounded and overlays show the active owner on the sidecar mask.

Next gate:
  move to longer training/eval.  Do not add another patch to this failure mode
  unless downstream-visible overlap again rises into the old failure band or
  overlays show active owner localization leaving the sidecar mask.
```

## 2026-05-20 Issue Update: Direct Owner Write-Through and After-Fusion Metric

Status: implemented locally and deployed to A7; 300-step validation running.

Root cause:

```text
The graph-side owner candidate could already be close to the sidecar/contact
mask, but posterior owner transport was still judged through a pre-fusion
distance and could be diluted by observation-averaged owner geometry before
posterior-file write-through.
```

Repair:

```text
posterior_owner_transport_direct_candidate_assignment
posterior_owner_transport_direct_candidate_min_score
posterior_owner_transport_dist_after_fusion
posterior_owner_transport_active_dist_after_fusion_*
```

Rule:

```text
If a graph owner candidate wins bounded candidate/file responsibility, write
that candidate's owner geometry directly into the posterior owner measurement
for the selected file.  Use the old obs-averaged transport only as fallback for
roles without a direct candidate/file match.
```

Implementation guard:

```text
Selected (slot, graph, score) triples are collected first and written through
out-of-place index_copy.  This keeps the score differentiable and avoids
PyTorch autograd version-counter failures from in-place slice writes.
```

Why this is not an ad-hoc patch:

```text
It is the posterior-filter translation of Slot Attention / SAVi-style
responsibility-preserving write-back.  It adds no new proposal source, no SAM,
no reconstruction decoder, and no hard visual VQ truth.  It only preserves the
candidate identity that the existing slot-axis competition already selected.
```

Local verifiers:

```text
test_posterior_owner_transport_uses_direct_graph_candidate_write_through
picf_latest_slot_deployment_audit.py --fail-on-fail
picf_object_candidate_slot_binding_audit.py --json
verify_picf_owm_contract.py
picf_owm_strict_diagnose.py --fail-on-fail
picf_owm_dataflow_trace.py --fail-on-fail
picf_owm_mvtrack_deep_audit.py --fail-on-fail
```

Remote validation:

```text
picf_a7_owner_direct_autogradsafe_smoke300_20260520
```

Interim validation through step 100:

```text
step 50:
  loss_total = 0.1716
  loss_anchor_object_pull = 0.3409
  loss_object_explanation_point = 2.2158
  object_candidate_owner_geometry_active_dist_mean = 0.00021 m
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00426 m
  active/context/downstream/raw support overlap max = 0.0007 / 0.2626 / 0.3412 / 0.4752
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.0256

step 100:
  loss_total = 0.1662
  loss_anchor_object_pull = 0.3305
  loss_object_explanation_point = 2.1009
  object_candidate_owner_geometry_active_dist_mean = 0.00020 m
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00466 m
  active/context/downstream/raw support overlap max = 0.0029 / 0.3442 / 0.3674 / 0.5788
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.0366

step 150:
  loss_total = 0.1289
  loss_anchor_object_pull = 0.2205
  loss_object_explanation_point = 2.1893
  object_candidate_owner_geometry_active_dist_mean = 0.00017 m
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00312 m
  active/context/downstream/raw support overlap max = 0.0085 / 0.3515 / 0.3962 / 0.8407
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.1102

step 200:
  loss_total = 0.1412
  loss_anchor_object_pull = 0.2728
  loss_object_explanation_point = 1.8646
  object_candidate_owner_geometry_active_dist_mean = 0.00021 m
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00236 m
  active/context/downstream/raw support overlap max = 0.0567 / 0.1500 / 0.3002 / 0.9961
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.1207

step 250:
  loss_total = 0.1787
  loss_anchor_object_pull = 0.3537
  loss_object_explanation_point = 2.3259
  object_candidate_owner_geometry_active_dist_mean = 0.00028 m
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00273 m
  active/context/downstream/raw support overlap max = 0.0582 / 0.0929 / 0.2464 / 0.9990
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.1211

step 300:
  loss_total = 0.1930
  loss_anchor_object_pull = 0.4210
  loss_object_explanation_point = 1.8818
  object_candidate_owner_geometry_active_dist_mean = 0.00028 m
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00305 m
  active/context/downstream/raw support overlap max = 0.0427 / 0.1460 / 0.3116 / 0.9977
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  posterior_recycle_rate = 0.1114
```

Interim verdict:

```text
The old posterior-owner transport failure is fixed at the first 100-step gate:
the active posterior owner closes to the accepted owner candidate within ~5 mm,
while the candidate owner geometry itself is sub-millimeter relative to the
sidecar point mask.  Through step 300 the active closure remains ~2-5 mm,
downstream-visible overlap stays near 0.25-0.40 instead of the old 0.7-0.9
failure band, and active duplicate overlap stays zero.  Raw reserve overlap
still rises to ~1.0, but this run confirms it is currently contained in
reserve/non-owner context rather than becoming an action-visible duplicate owner.

Remaining caution:
  loss_anchor_object_pull is not monotonic and returns to 0.421 at step 300.
  This is not the previous posterior write-through failure, because the direct
  after-fusion residual stays low.  It should be watched in the next action-aware
  or longer co-training run together with overlays.
```

Decision rule:

```text
Judge posterior owner closure by posterior_owner_transport_active_dist_after_fusion_*,
not only by posterior_owner_transport_dist_to_standard.  The pre-fusion metric
is correction magnitude; the after-fusion metric is the actual posterior
closure residual.
```

2026-05-20 local strict re-audit closure:

```text
resolved-code:
  proposal debug logging was accidentally tracklet-gated.  It is now emitted
  whenever proposal tokens exist, preventing proposal-only sidecar false
  negatives.

resolved-test-contract:
  evidence cache writes active posterior files only.  The old all-row-valid
  assertion was inconsistent with active/context/reserve semantics and is now
  replaced by active-row and next-step-read assertions.

resolved-code:
  guarded binding-consistency temporal matching used current-slot weights for
  both current->future and future->current terms.  It now uses current weights
  for the forward term and future weights for the backward term, restoring
  permutation tolerance under detached future-slot reordering.
```

Validation:

```text
pipeline_test.py + training_test.py:
  133 passed

picf_latest_slot_deployment_audit.py --fail-on-fail:
  14/14 PASS

picf_object_candidate_slot_binding_audit.py --json:
  ok=true, 37 checks

verify_picf_owm_contract.py:
  PASS

picf_owm_dataflow_trace.py --fail-on-fail:
  ok=true

picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS

git diff --check:
  PASS
```

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
  --require-tracklet
  --proposal-nearest-max-gap 8
  --min-proposal-nonempty-fraction 0.80
  --min-proposal-reachable-fraction 0.85
  --min-mask-reachable-fraction 0.85

Status at 2026-05-21 00:24 CST:
  generation finished, 240,758 npz files, no GPU use.
  A strict physical-key proposal/mask gate fails because proposals are sparse:
  about 82% of sampled files have current non-empty proposal/mask. This is not
  a tracklet failure. The maintained contract is current tracklet plus nearest
  non-empty proposal/mask borrowing with proposal_age decay.

2026-05-21 reserve/raw-overlap audit disposition:
  the 300-step reserve-scope A7 gate completed.  The old active-owner collapse
  is not reproduced.  Raw same-role overlap still saturates, but it tracks
  reserve/no-object rows rather than active object owners.  Treat raw overlap
  as mixed telemetry; judge production health with active overlap, downstream
  overlap, posterior active duplicate overlap, and split graph/posterior
  object-pull.
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

2026-05-26 cotrain rebound update:

```text
PC1 policy-only causal probe:
  freezing `core.*` while keeping PICF forward enabled removed direct PICF
  prefix motion and action improved in the 6050-6300 window.

A7 same-timescale cotrain:
  action improved early but rebounded while structural/PICF state continued to
  move.

Root issue:
  not "cotrain is bad"; the issue is same-timescale cotrain where structural
  losses move the action-visible PICF prefix as fast as the action head learns.

Maintained repair:
  two-timescale cotrain with `--picf-core-lr-scale` below 1.0, action at the
  normal 2.0 scale, weak scaffold floor, and raw predictive losses disabled.

Reference:
  docs/PICF_AQR_OWM_COTRAIN_TWO_TIMESCALE_FINAL_20260526.md
```

2026-05-28 optimizer-group bug update:

```text
Issue:
  FSDP wraps root parameter names as `_fsdp_wrapped_module.core.*`.
  The optimizer split previously checked only `name.startswith("core.")`.

Consequence:
  In FSDP full-shard runs, wrapped PICF core parameters could be assigned to
  `policy_head`, so `PICF_CORE_LR_SCALE` did not necessarily slow the core.
  This invalidates prior causal claims that core0.02/core0.005 had truly
  tested slow-PICF two-timescale cotrain.

Fix:
  scripts/picf_core_train.py now canonicalizes leading `module.` and
  `_fsdp_wrapped_module.` before assigning `picf_core` vs `policy_head`.

Local validation:
  python -m pytest scripts/picf_core_train_test.py -k "optimizer or picf_core_group" -q
    16 passed
  python -m pytest scripts/picf_core_train_test.py -q
    134 passed
  verify_picf_owm_contract / strict_diagnose / dataflow_trace / latest slot
  audit all pass.

Runtime acceptance:
  the next FSDP run must log both `lr_group_picf_core` and
  `grad_norm_group_picf_core`.  If either is absent, the run is invalid for
  action-rebound root-cause analysis.

Runtime first gate:
  `picf_a7_reboundgate_ema_recipe_scopefix_lrgroupfix_from7000_30k_20260528`
  step7050 logs both required fields and shows the intended 0.005x PICF-core
  LR (`3.175974e-7` vs policy-head `6.351949e-5`).  Action at 7050 is
  `0.022599`, better than the prior EMA 7050 `~0.02610`.

Runtime 7300 update:
  step7200 action_default_equiv = 0.018195
  step7250 action_default_equiv = 0.026179
  step7300 action_default_equiv = 0.026738
  step7300 loss_total_minus_action = 0.009539
  step7300 active/downstream overlap = 0.015000 / 0.020547

Runtime 7450 update:
  step7350 action_default_equiv = 0.027469
  step7400 action_default_equiv = 0.026737
  step7450 action_default_equiv = 0.027944
  tail6 action_default_equiv mean = 0.025544
  tail6 action_default_equiv max = 0.027944
  step7450 loss_total_minus_action = 0.011623
  step7450 active/downstream overlap = 0.079741 / 0.079446

Runtime 7550 update:
  step7500 action_default_equiv = 0.030833
  step7550 action_default_equiv = 0.035953
  step7550 loss_total_minus_action = 0.011119
  step7550 active/downstream overlap = 0.105000 / 0.109954
  recent >=0.035 count = 1 row, not yet two-row rejection

Decision:
  yellow warning at 7550.  Do not switch until step7600 unless the run crashes.
  If step7600 is also >=0.035, mark the rebound reproduced under the fixed
  FSDP LR groups.  If step7600 falls below 0.035, keep watching 7650/8000.

Status:
  closed-code / open-late-window / yellow warning.  The code bug is fixed and
  the 7350 runtime hard gate passed, but 7550 shows the first action-specific
  warning row.

Runtime 7600 update:
  step7600 action_default_equiv = 0.035442
  step7600 loss_total_minus_action = 0.011023
  step7600 active/downstream overlap = 0.060000 / 0.066035
  step7600 lr_group_picf_core = 3.126069e-7
  step7600 lr_group_policy_head = 6.252138e-5

Decision:
  fixed-group rebound reproduced under the strict two-row rule.  The FSDP
  grouping fix is closed-code and retained, but it is not the full action
  rebound cure.  Because policyonly_actionsemantic also reaches
  `loss_action_default_equiv=0.036278` at step7550 with PICF frozen and
  `loss_total_minus_action=0`, the direct root is now narrowed to
  semantic/action-side low-basin stability and/or a repeated hard data-window
  cluster, not raw overlap or non-action structural losses.

Next required counterfactuals:
  1. extend action_head_only from the same step7000 checkpoint beyond 7550/8000;
  2. add action-window trace fields for dataset segment, episode/transition,
     prompt hash, target norms, and per-component action loss;
  3. replay or isolate the 7500-7600 window with semantic frozen vs trainable.

Reference:
  docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
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
     production evidence.  The maintained root was regenerated from
     contact-motion proposal/mask evidence into a clean KLT tracklet root, with
     sparse mask keys preserved during merge.  The 2026-05-21 reserve audit
     shows proposal and tracklet dataflow alive through the training path.
   Verdict:
     closed-dataflow / open-train: sidecar generation and runtime threading are
     accepted for the next 30K validation, but behavior benefit still requires
     long-run evidence.

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
lifecycle still require long-run validation.  The latest reserve audit shows
that high raw same-role overlap is mostly reserve/no-object telemetry, not
active object-owner collapse.  The open question is whether the active owner
path, downstream context capacity, and weak sidecar/object-pull stay stable
under full action-aware co-training.
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

## 2026-05-21 30K Full-Sidecar Run Interruption And Loss Triage

Run:

```text
remote:
  A7 / qgE72e

experiment:
  picf_a7_actionaware_ownerdirect_long30k_fullsidecar_20260521

log:
  /mnt/picf_run_logs/picf_a7_actionaware_ownerdirect_long30k_fullsidecar_20260521.log

intended contract:
  30000 steps
  save every 2500 steps
  keep last 3 checkpoints
  anchor overlay every 100 steps
  visual diagnostics every 500 steps
  PaliGemma trainable
  Sonata / V-JEPA / AnyTouch frozen
  guarded slot_jepa / support_pred / binding_consistency / denoising losses off
```

Hard interruption:

```text
status:
  stopped at step 500 before the first 2500-step checkpoint

root cause:
  scripts/picf_core_train.py::_save_visual_diagnostics built compare_grid rows
  with current/physical/semantic/target image heights 256/256/200/256.

classification:
  engineering diagnostics bug, not a numeric training divergence.

fix:
  resize current, physical, semantic, and target images to the same compare_size
  before concatenate; missing physical/semantic predictions now use zeros with
  the selected compare_size instead of raw target_next size.

regression test:
  test_save_visual_diagnostics_handles_missing_prediction_size_mismatch
```

Validated metric window before interruption:

```text
step:
  50 -> 100 -> 150 -> 200 -> 250 -> 300 -> 350 -> 400 -> 450 -> 500

loss_action_default_equiv:
  0.1003 -> 0.0712 -> 0.0679 -> 0.0601 -> 0.0620 ->
  0.0645 -> 0.0655 -> 0.0623 -> 0.0672 -> 0.0662

loss_anchor_pv:
  0.6784 -> 0.6481 -> 0.6408 -> 0.6098 -> 0.5773 ->
  0.6050 -> 0.6205 -> 0.6176 -> 0.6290 -> 0.6246

loss_anchor_object_pull:
  0.3563 -> 0.4056 -> 0.7177 -> 0.4286 -> 0.8464 ->
  0.7654 -> 0.6064 -> 0.8417 -> 1.0885 -> 1.1643

loss_object_explanation_point:
  1.4734 -> 2.4118 -> 2.2104 -> 2.5046 -> 3.3775 ->
  3.2065 -> 3.6612 -> 4.1334 -> 5.5552 -> 6.2926

active_same_role_support_overlap_max:
  0.0091 -> 0.0252 -> 0.0188 -> 0.0472 -> 0.1212 ->
  0.0491 -> 0.0813 -> 0.1841 -> 0.1690 -> 0.2122

downstream_same_role_support_overlap_max:
  0.2939 -> 0.3543 -> 0.3678 -> 0.4805 -> 0.4521 ->
  0.4260 -> 0.4799 -> 0.4626 -> 0.4548 -> 0.4335

raw_same_role_support_overlap_max:
  0.4713 -> 0.6874 -> 0.7205 -> 0.9436 -> 0.9974 ->
  0.9988 -> 0.9991 -> 0.9999 -> 1.0000 -> 1.0000

context_anchor_count:
  7.69 -> 7.63 -> 7.51 -> 6.85 -> 3.92 ->
  4.32 -> 4.22 -> 2.40 -> 2.68 -> 1.97

posterior_file_competition_active_duplicate_overlap_max:
  0.0 at every logged gate

posterior_recycle_rate:
  0.087 -> 0.063 -> 0.051 -> 0.099 -> 0.100 ->
  0.092 -> 0.097 -> 0.071 -> 0.063 -> 0.092

sidecar evidence:
  owm_proposal_valid_fraction remains about 0.81-0.93
  owm_tracklet_valid_fraction remains about 0.95-1.00
```

Interpretation:

```text
resolved relative to older catastrophic failures:
  active duplicate owner collapse is not present.
  active same-role overlap stays below 0.22 through step 500, not the old 0.95+
  action-visible collapse band.
  recycle does not saturate.
  tracklet/proposal sidecar dataflow is alive.
  action_default_equiv is in the historical early-good band.

not resolved:
  raw reserve/context same-role overlap still saturates to ~1.0.
  context anchor count collapses from ~7.7 to ~2.0.
  object pull and point explanation losses trend upward despite stable action loss.

theoretical diagnosis:
  The sidecar object evidence is a weak sparse scaffold, not a dense hard mask.
  The current active owner path can remain bounded while reserve/context files
  receive weaker independent objectives.  Those reserve/context rows then drift
  toward the same high-evidence sidecar object or background proxy, which raises
  raw overlap and reduces context count.  Because active duplicate overlap is
  zero, this is not yet the old action-visible duplicate-owner failure.  Because
  object_pull and point_explanation rise, it is still a structural risk: the
  current object explanation pressure is not consistently aligned with the
  posterior owner closure objective over long co-training.
```

Current action:

```text
must-fix-before-restart:
  visual diagnostics size normalization bug.

must-watch-after-restart:
  loss_anchor_object_pull
  loss_object_explanation_point
  raw/context support overlap
  context_anchor_count
  active/downstream overlap
  overlay: object owner should remain inside sidecar object mask, not just near
  the gripper/contact bridge.
```

### 2026-05-21 Loss Root-Cause Analysis

This section separates actual failures from noisy diagnostics.  The goal is to
avoid patching one scalar at a time while missing the shared mathematical cause.

#### Loss / metric definitions that matter

```text
loss_action_default_equiv:
  action loss rescaled back to the historical default-equivalent scale.  This is
  the only action number directly comparable with 4-22-style ablations.

loss_anchor_pv:
  object-row point/visual projective distribution consistency.  For confirmed
  object rows it compares point support p_j and visual support v_j through the
  projective compatibility C:

    v_hat_j = normalize(p_j C)
    p_hat_j = normalize(v_j C^T)
    L_pv_j = 0.5 JS(v_j, v_hat_j) + 0.5 JS(p_j, p_hat_j)

  Background/reserve rows are intentionally excluded.  A flat or mildly
  improving loss_anchor_pv means the projective point/visual bridge is not the
  primary failure.

loss_anchor_object_pull:
  diagnostic pull from confirmed graph/posterior object rows to sidecar point
  centers:

    target_x_j = normalize(m_j)^T X_point
    L_pull_j = SmoothL1(x_j / sigma, stopgrad(target_x_j) / sigma)

  It currently mixes graph-anchor and posterior-file terms.  Therefore it is a
  structural warning when it rises, but it must be interpreted together with
  posterior_owner_transport_active_dist_after_fusion.

loss_object_explanation_point:
  compactness/variance of point support under the object-explanation layer.  It
  measures whether the selected object file explains a compact point-mask core,
  not whether action is improving.

loss_mapg_support_diversity:
  same-role support overlap penalty.  It is meaningful only after splitting raw,
  context/downstream, and active scopes.

aqr_same_role_support_overlap_max:
  raw same-role overlap over all fixed-capacity rows, including reserve/context.

aqr_active_same_role_support_overlap_max:
  overlap only among active object owners.  This is the collapse signal that
  directly threatens action-visible duplicate owners.

aqr_context_anchor_count:
  number of low-weight context rows still retained as background/peripheral
  carriers.
```

#### Observed pattern from the interrupted 50-500 window

```text
healthy action:
  loss_action_default_equiv enters 0.060-0.067 by step 200-500.

healthy active owner competition:
  posterior_file_competition_active_duplicate_overlap_max stays 0.0.
  active_same_role_support_overlap_max stays <= 0.212.
  posterior_recycle_rate stays roughly 0.05-0.10.

still problematic reserve/context structure:
  raw_same_role_support_overlap_max goes to ~1.0.
  context_anchor_count falls from ~7.7 to ~2.0.

still problematic object explanation:
  loss_anchor_object_pull rises 0.356 -> 1.164.
  loss_object_explanation_point rises 1.47 -> 6.29.
```

#### Theoretical root cause

```text
The current design has fixed-capacity object files but most CALVIN steps expose
only one or a few task-relevant object candidates.  Active owner competition now
correctly selects one object file and demotes duplicate active files, which fixes
the old catastrophic same-object active collapse.

However, reserve/context rows do not receive an equally strong independent
objectness objective.  They are retained to preserve background/peripheral dense
context, but their support distributions are still produced by the same readers
and see the same high-evidence sidecar/proposal/contact region.  Under action
co-training, those low-weight rows can converge to the same easy evidence while
remaining mostly non-action-visible.  This explains:

  raw overlap saturates,
  active duplicate overlap stays zero,
  context count shrinks,
  action loss remains good.

Object-pull and object-explanation-point rise for a related but distinct reason:
sidecar masks are sparse, trajectory-derived weak evidence rather than exact
instance labels.  They are good enough as scaffold, but if interpreted as a
long-term compactness target on every active object step, they can disagree with
the current posterior/action objective.  In particular, contact-motion masks may
track the changing contact bridge or motion core rather than the whole semantic
object.  The graph/posterior object owner may then remain action-useful while the
sidecar compactness scalar worsens.
```

#### Paper alignment

```text
Object Binding in ViTs, NeurIPS 2025:
  https://arxiv.org/abs/2510.24709
  Supports our use of pairwise/projected same-object subspace rather than raw
  hidden cosine only.  It does not imply that fixed rows with weak labels will
  automatically become clean object files; it motivates probing and calibrating
  same-object signals.

SlotVLA, ICRA 2026:
  https://arxiv.org/abs/2511.06754
  Uses object-centric annotations, mask/box labels, temporal tracking, slot
  tokenizer, and relation decoder.  Compared with our setting, it has stronger
  object supervision.  Therefore our sidecar masks should stay weak evidence,
  not hard truth.

SlotMIM / data-centric PVM revisit, 2025:
  https://huggingface.co/papers/2503.06960
  Supports semantic bottleneck and cross-view consistency as object-centric
  pretraining pressure.  It supports bottlenecking object rows, but not deleting
  dense context tokens.

Embodied-SlotSSM / LIBERO-Mem, 2025:
  https://www.sciencestack.ai/paper/2511.11478
  Supports persistent slot identity plus temporal dynamics/memory.  This matches
  our posterior-file route, but it also highlights that identity should be judged
  through temporal consistency, not a single-frame raw overlap scalar.

OBEYED-VLA, 2026:
  https://arxiv.org/abs/2512.22519
  Supports explicit task-conditioned object-centric and geometry-aware grounding
  to avoid monolithic action training eroding grounding.  It argues for our
  sidecar/proposal/geometry scaffold, but also suggests those signals must be
  calibrated and not treated as perfect labels.
```

#### Immediate plan

```text
P0 fixed:
  visual diagnostics crash at step 500.

P1 do not patch yet:
  action loss is not the failing term.  Do not lower action pressure solely
  because total/alignment rises.

P2 next instrumentation:
  done locally on 2026-05-21:
    loss_anchor_object_pull_graph
    loss_anchor_object_pull_posterior
    loss_anchor_object_pull_graph_weight_sum
    loss_anchor_object_pull_posterior_weight_sum
    loss_anchor_object_pull_target_mass_mean
  This split does not change the objective; it makes the next 100/200/300-step
  gate identifiable.

P3 next training gate:
  restart only after P0 fix and inspect 50/100/200/300/500:
    loss_anchor_object_pull
    loss_object_explanation_point
    object_candidate_owner_geometry_active_dist_mean
    posterior_owner_transport_active_dist_after_fusion_mean
    active/downstream/raw overlap
    context_anchor_count
    overlays against sidecar mask

P4 likely mathematical repair if P2 confirms the same trend:
  keep active object files under owner transport,
  keep dense background/context tokens outside object-file truth,
  weaken or confidence-gate object_explanation_point on sparse/noisy masks,
  and add a context-retention objective only for non-object context rows rather
  than forcing reserve rows to behave as extra objects.

P5 avoid:
  do not re-enable blind SAM.
  do not make sidecar masks hard labels.
  do not add a full reconstruction decoder to the action path as a cure for
  object-pull; this would change the belief router into a pixel reconstruction
  model and conflict with the current PI0.5 path.
```

### 2026-05-21 Loss Issue Matrix And Root Cause Plan

This matrix is the current loss-level triage after the interrupted 30K
full-sidecar run.  It separates symptoms that are already fixed from symptoms
that still require a structural response.

```text
1. loss_action_default_equiv

   observation:
     0.1003 -> 0.060-0.067 by step 200-500.

   status:
     healthy in the early-run sense.

   physical meaning:
     action loss in the historical default-equivalent scale; this is the only
     action number directly comparable with the 4-22 baseline family.

   root-cause read:
     not the current failing term.  Action can improve while object-file
     structure still has reserve/context problems.

   action:
     do not reduce action pressure just because total/alignment rises.
     Keep action_default_equiv as the external usefulness gauge.

2. loss_anchor_pv

   observation:
     0.6784 -> 0.6246, with a best value 0.5773 at step 250.

   status:
     mildly improving; not the primary failure.

   physical meaning:
     confirmed object-row point/visual projective distribution consistency:
       v_hat = normalize(p C), p_hat = normalize(v C^T),
       L = 0.5 JS(v, v_hat) + 0.5 JS(p, p_hat).

   root-cause read:
     the point/visual projective bridge is not globally broken.

   action:
     keep it as a geometry sanity term.  Do not overfit fixes to this scalar.

3. active_same_role_support_overlap_max

   observation:
     stays <= 0.212 through step 500.

   status:
     old action-visible duplicate-owner collapse is largely fixed in this run.

   physical meaning:
     overlap among active object-owner rows only.

   root-cause read:
     posterior file competition and active owner gating are doing their main
     job.  This is why the high raw overlap does not immediately imply the
     action path has collapsed.

   action:
     keep watching; failure threshold remains the old 0.7-0.95+ collapse band.

4. raw_same_role_support_overlap_max

   observation:
     0.4713 -> ~1.0 while active overlap stays low.

   status:
     still bad, but scope-specific: reserve/context rows, not active owners.

   physical meaning:
     same-role support overlap over all fixed-capacity graph rows, including
     reserve/context rows that may not be action-visible.

   root-cause read:
     fixed object-file capacity is larger than the number of confident objects
     in many CALVIN frames.  Active competition selects one/few owners; leftover
     reserve/context rows still see the same high-evidence region and drift
     together.  This is capacity pressure plus weak context supervision, not
     necessarily duplicate active posterior identity.

   action:
     separate object files from dense context more explicitly.  Background and
     peripheral dense tokens should remain available as context carriers rather
     than being forced to become extra object rows.
     Added follow-through logging on 2026-05-21:
       aqr_reserve_same_role_support_overlap_max
       aqr_reserve_same_role_object_core_overlap_max
     The next gate can now prove whether raw saturation is reserve/no-object
     capacity rather than active or context overlap.

5. context_anchor_count

   observation:
     7.69 -> 1.97.

   status:
     bad companion symptom of raw overlap saturation.

   physical meaning:
     number of low-weight rows still acting as peripheral/background carriers.

   root-cause read:
     the model is losing "peripheral vision" capacity while keeping one/few
     action-useful owners alive.  That can preserve early action loss but risks
     longer-horizon generalization and layout sensitivity.

   action:
     add or preserve a context-retention path for non-object dense evidence.
     Do not solve this by making every background row a hard object slot.

6. loss_anchor_object_pull

   observation:
     0.356 -> 1.164.

   status:
     unresolved mixed diagnostic.

   physical meaning:
     SmoothL1 pull from object-confirmed graph/posterior centers to sidecar
     point-space targets.

   root-cause read:
     this scalar currently conflates graph-anchor center error and posterior
     file center error.  It can rise even when the active posterior transport
     distance after fusion is small.  Therefore it identifies a structural
     tension but not the exact failing submodule.

   action:
     next code/logging step should split it into graph_pull and posterior_pull,
     plus expose row weights and target prior mass.  Without that split, any
     direct fix is underidentified.

7. loss_object_explanation_point

   observation:
     1.47 -> 6.29.

   status:
     unresolved and important.

   physical meaning:
     compactness/point-mask explanation pressure for object candidates.

   root-cause read:
     sidecar masks are weak sparse contact/motion scaffolds, not exact full
     instance masks.  If the loss treats them as a persistent compact target,
     it can fight the posterior/action objective when contact points, motion
     cores, and full semantic objects differ.

   action:
     confidence-gate, quality-gate, or anneal this term on sparse/noisy masks.
     Do not convert sidecar masks into hard labels.

8. loss_mapg_support_diversity / loss_mapg_routing

   observation:
     diversity worsens moderately; routing mildly worsens.

   status:
     secondary symptom, not root cause.

   physical meaning:
     diversity discourages same-role support reuse; routing regularizes
     anchor-token routing structure.

   root-cause read:
     these rise because reserve/context rows collapse toward common evidence.
     Increasing diversity weight alone was historically insufficient and risks
     fighting active owner selection.

   action:
     fix context/object decomposition first, then tune diversity if active and
     downstream overlap still degrade.
```

Current root-cause conclusion:

```text
The central remaining problem is not "action training destroys everything" and
not "point/visual projection is broken."  The current evidence says:

  active object-owner competition is mostly repaired;
  action loss is in a healthy early band;
  sidecar/tracklet dataflow is alive;
  reserve/context rows still collapse;
  object explanation/pull targets are under-calibrated weak scaffolds.

Therefore the next scientifically clean step is not to add another unrelated
module.  It is to decompose object files versus dense context, split the mixed
pull metric, and calibrate sidecar compactness by confidence/quality.
```

Next verification gate:

```text
Before trusting another 30K run:
  1. log graph_pull and posterior_pull separately.
  2. log object target prior mass and mask quality.
  3. inspect 50/100/200/300/500 after restart.
  4. require active overlap to stay low and context count not to collapse.
  5. require action_default_equiv to stay in the early-good band.

If graph_pull rises but posterior_pull is stable:
  repair graph candidate/sidecar assignment.

If posterior_pull rises but graph_pull is stable:
  repair posterior binding/write-through.

If both are stable but object_explanation_point rises:
  the compactness target is too strict/noisy; gate or anneal it.

If action worsens while object metrics improve:
  object supervision is overpowering VLA co-training; lower object weights or
  stage the schedule.
```

### 2026-05-21 Split-Pull Frozen-Policy 300-Step Gate

Run:

```text
picf_a7_slot_splitpull_frozen_policy_300_20260521

launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_slot_splitpull_frozen_policy_300_20260521.sh

contract:
  freeze PaliGemma, PI0.5 action pressure, Sonata, V-JEPA, AnyTouch;
  train PICF/AQR/posterior/slot/OEML/sidecar routing stack;
  full sidecar root:
    /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

Interim metrics:

```text
step 50:
  loss_total = 0.15999
  loss_anchor_object_pull = 0.34821
  loss_anchor_object_pull_graph = 0.32296
  loss_anchor_object_pull_posterior = 0.39706
  loss_anchor_object_pull_target_mass_mean = 0.91040
  loss_object_explanation_point = 1.43001
  active_same_role_support_overlap_max = 0.00418
  raw_same_role_support_overlap_max = 0.45544
  downstream_same_role_support_overlap_max = 0.28494
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  context_anchor_count = 7.735

step 100:
  loss_total = 0.13953
  loss_anchor_object_pull = 0.26883
  loss_anchor_object_pull_graph = 0.25054
  loss_anchor_object_pull_posterior = 0.29998
  loss_anchor_object_pull_target_mass_mean = 0.95280
  loss_object_explanation_point = 1.81894
  active_same_role_support_overlap_max = 0.01363
  raw_same_role_support_overlap_max = 0.75059
  downstream_same_role_support_overlap_max = 0.41240
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  context_anchor_count = 7.365

step 200:
  loss_total = 0.14757
  loss_anchor_object_pull = 0.26552
  loss_anchor_object_pull_graph = 0.25601
  loss_anchor_object_pull_posterior = 0.28424
  loss_anchor_object_pull_target_mass_mean = 0.98444
  loss_object_explanation_point = 2.21727
  active_same_role_support_overlap_max = 0.02295
  raw_same_role_support_overlap_max = 0.99022
  downstream_same_role_support_overlap_max = 0.43976
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  context_anchor_count = 6.065
```

Interim read:

```text
fixed or strongly improved:
  sidecar-to-graph path is alive:
    graph pull is lower than the interrupted 30K window and improves by step 100.
  graph-to-posterior write-through is alive:
    posterior pull improves from 0.397 to 0.284 by step 200.
  active object-owner duplicate collapse is not recurring:
    active overlap stays below 0.023 through step 200.
  posterior active duplicate overlap remains exactly 0.

not fixed:
  raw same-role overlap rises to 0.99 because reserve/context rows still cluster.
  context count declines from 7.7 to 6.1.
  loss_object_explanation_point rises monotonically.

root-cause update:
  the remaining failure is no longer "anchors cannot reach sidecar objects".
  The split proves graph and posterior object-owner paths are functional.  The
  unresolved problem is the interaction between reserve/context rows and the
  OEML point-compactness target.  Sidecar masks are useful weak measurements,
  but the compactness/explanation term is stricter than the noisy sparse mask
  evidence can justify.

pending:
  step 300 final gate and overlay inspection.
```

Final 300-step metrics:

```text
step 250:
  loss_total = 0.16062
  loss_anchor_object_pull = 0.31666
  loss_anchor_object_pull_graph = 0.31341
  loss_anchor_object_pull_posterior = 0.32173
  loss_anchor_object_pull_target_mass_mean = 0.99963
  loss_anchor_pv = 0.59546
  loss_object_explanation_point = 1.97679
  loss_object_explanation_contact = 0.48737
  loss_object_explanation_duplicate = 0.09680
  active_same_role_support_overlap_max = 0.00899
  raw_same_role_support_overlap_max = 0.97303
  downstream_same_role_support_overlap_max = 0.43558
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  context_anchor_count = 7.005
  posterior_recycle_rate = 0.09287
  posterior_owner_transport_active_dist_mean = 0.00296
  object_candidate_owner_geometry_active_dist_mean = 0.000234
  owm_proposal_valid_fraction = 0.86917
  owm_tracklet_valid_fraction = 0.96884

step 300:
  loss_total = 0.09667
  loss_anchor_object_pull = 0.14183
  loss_anchor_object_pull_graph = 0.14536
  loss_anchor_object_pull_posterior = 0.13167
  loss_anchor_object_pull_target_mass_mean = 1.03759
  loss_anchor_pv = 0.54491
  loss_object_explanation_point = 1.90588
  loss_object_explanation_contact = 0.43113
  loss_object_explanation_duplicate = 0.04387
  active_same_role_support_overlap_max = 0.00684
  raw_same_role_support_overlap_max = 0.94493
  downstream_same_role_support_overlap_max = 0.43680
  posterior_file_competition_active_duplicate_overlap_max = 0.0
  context_anchor_count = 7.230
  posterior_recycle_rate = 0.08878
  posterior_owner_transport_active_dist_mean = 0.00193
  object_candidate_owner_geometry_active_dist_mean = 0.000133
  owm_proposal_valid_fraction = 0.92667
  owm_tracklet_valid_fraction = 0.99862
```

Final read:

```text
clearly fixed relative to the old failures:
  active object-owner support collapse:
    0.004 -> 0.023 -> 0.0068, never entering the historical 0.7-0.99 band.
  posterior active duplicate collapse:
    exactly 0.0 at every logged gate.
  graph-to-sidecar and posterior-to-sidecar pull:
    graph pull 0.323 -> 0.145,
    posterior pull 0.397 -> 0.132.
  point/visual bridge:
    loss_anchor_pv 0.683 -> 0.545.
  lifecycle instability:
    posterior_recycle_rate remains around 0.09 instead of saturating.
  sidecar dataflow:
    proposal valid fraction ends at 0.927,
    tracklet valid fraction ends at 0.999.

still not clean:
  raw same-role support overlap:
    still high at 0.945, because raw includes reserve/context rows.
  downstream same-role support overlap:
    stays around 0.43; lower than raw but not negligible.
  loss_object_explanation_point:
    improved from its 200-step peak but remains higher than step 50.
```

Updated root cause:

```text
The 300-step split falsifies the strongest previous failure hypothesis:
  "object sidecar evidence cannot pull graph/posterior anchors."

Both graph and posterior pull improve substantially.  The active object file is
not losing the object.  Therefore the remaining loss symptoms are not caused by
a missing owner transport path or broken object-candidate assignment.

The remaining issue is a scope mismatch:
  active object-owner rows behave correctly,
  reserve/context rows still share support because they are not independent
  object hypotheses,
  object_explanation_point still treats sparse/noisy sidecar point masks as a
  compactness signal that is stricter than the weak scaffold can justify.

This is why raw overlap can remain high while active overlap, posterior
duplicates, graph pull, posterior pull, recycle rate, and overlays are healthy.
```

Loss-by-loss issue list after the final gate:

```text
1. loss_anchor_object_pull
   status:
     fixed for the graph/posterior transport path in this diagnostic.
   evidence:
     total 0.348 -> 0.142,
     graph 0.323 -> 0.145,
     posterior 0.397 -> 0.132.
   remaining risk:
     may still oscillate under full action/PaliGemma co-training.
   next check:
     keep split metrics in every long run; do not collapse them back into one
     scalar.

2. loss_anchor_pv
   status:
     healthy.
   evidence:
     0.683 -> 0.545.
   interpretation:
     point/visual projective agreement is not the root failure.
   next check:
     watch under full action pressure only.

3. loss_object_explanation_point
   status:
     partially improved but unresolved.
   evidence:
     1.430 -> 2.217 peak -> 1.906.
   root cause:
     sidecar masks are contact/motion weak scaffolds.  They identify the object
     core/contact region, not a dense exact instance mask.  A strict compactness
     loss can punish an otherwise useful object file when the object support is
     elongated, partially visible, or mixed with contact/gripper points.
   clean repair if needed:
     confidence/quality-gate this term by sidecar quality, tracklet stability,
     and candidate mass.  Do not convert sidecar masks to hard segmentation
     labels.

4. raw_same_role_support_overlap_max
   status:
     still high, but no longer an active-collapse proof.
   evidence:
     0.455 -> 0.945, while active overlap ends at 0.0068.
   root cause:
     raw scope includes reserve/context rows.  Those rows are deliberately
     capacity/background carriers, not confirmed object files.  If only one or
     two true objects are exposed, the remaining fixed-capacity rows can share
     the same dense/background support without entering the posterior active
     owner set.
   clean repair if needed:
     report raw/context/active separately and keep dense context as context,
     not as forced object files.  Do not add a blind stronger diversity loss
     against all rows.
   2026-05-21 follow-through:
     add reserve-scope overlap metrics:
       aqr_reserve_same_role_support_overlap_max
       aqr_reserve_same_role_object_core_overlap_max
     This closes the last ambiguity in the raw-overlap scalar.  If reserve is
     high while active and downstream remain low, raw overlap is a no-object
     capacity artifact, not a slot-binding failure.

5. downstream_same_role_support_overlap_max
   status:
     acceptable but not ideal.
   evidence:
     0.285 -> 0.437.
   root cause:
     downstream still sees context/reserve carriers as auxiliary graph evidence.
   clean repair if needed:
     add a context-attention budget/regularizer only for non-object context
     rows, preserving dense evidence while preventing many identical context
     prefixes.

6. active_same_role_support_overlap_max
   status:
     fixed in this diagnostic.
   evidence:
     always below 0.023, final 0.0068.
   interpretation:
     posterior file competition and active owner gating are doing their job.

7. posterior_file_competition_active_duplicate_overlap_max
   status:
     fixed.
   evidence:
     exactly 0.0 at every gate.

8. posterior_recycle_rate
   status:
     healthy.
   evidence:
     around 0.09, no saturation.

9. loss_action_default_equiv
   status:
     diagnostic only in this frozen-policy run because action weight is zero.
   evidence:
     0.14-ish across the run.
   interpretation:
     not comparable as trained action progress here; use it only as a frozen
     readout sanity number.
```

Decision:

```text
Do not restart a 30K full train solely because raw overlap is high.
Do not claim all losses are solved.

The correct next step is one of:
  A. run a short full-policy gate with the same split metrics if we want to see
     whether action/PaliGemma pressure reintroduces active collapse;
  B. implement a principled object_explanation_point confidence gate if the
     user wants to reduce the remaining compactness risk before another long
     run.

The mathematically unjustified next steps are:
  add another global overlap penalty,
  force reserve/context rows to be objects,
  revive blind SAM,
  treat sidecar masks as hard truth,
  remove dense context from the action path.
```

### 2026-05-21 Reserve-Scope Raw-Overlap Audit Run

Why this run exists:

```text
The previous 300-step split-pull gate showed that graph and posterior object
pull improved, active overlap stayed near zero, and posterior active duplicate
overlap stayed exactly zero.  The only ambiguous bad scalar was raw
same_role_support_overlap_max.

Historically, we tried increasing same-role competition, diversity losses,
active-slot filtering, geometry duplicate gates, context gates, posterior file
competition, object-owner transport, and sidecar proposal/mask routing.  The
consistent outcome was:
  active-object collapse can be fixed,
  raw fixed-capacity overlap often remains high.

This matches the slot literature: MetaSlot/QASA-style methods suppress or mask
duplicate/no-object slots instead of forcing every unused fixed slot to become a
distinct object.  Therefore raw overlap must be decomposed before another model
change.
```

Code follow-through:

```text
src/openpi/picf/core/pipeline.py:
  now logs reserve/no-object overlap:
    aqr_reserve_same_role_support_overlap_max
    aqr_reserve_same_role_support_overlap_mean
    aqr_reserve_same_role_object_core_overlap_max
    aqr_reserve_same_role_object_core_overlap_mean

scripts/picf_core_train.py:
  OWM_DEBUG_METRIC_KEYS now exports these metrics to JSONL.

2026-05-21 local follow-through tightening:
  reserve_bool is now defined as:
    downstream <= epsilon AND NOT active
  context_bool remains:
    downstream > epsilon AND NOT active
  This makes active/context/reserve attribution mutually exclusive for future
  audits.  The running A7 gate was launched before this local tightening, but
  step-50 active rows already had downstream exposure, so the first read is not
  expected to change materially.
```

Run:

```text
experiment:
  picf_a7_slot_reserveaudit_frozen_policy_300_20260521

launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_slot_splitpull_frozen_policy_300_20260521.sh

remote log:
  /mnt/picf_run_logs/picf_a7_slot_reserveaudit_frozen_policy_300_20260521.log

contract:
  identical to the previous split-pull 300-step gate, except for the added
  reserve-scope observability.
```

Decision rule:

```text
If raw overlap is high but:
  active overlap is low,
  downstream overlap is bounded,
  reserve overlap is high,
  posterior active duplicate overlap is zero,
  graph/posterior pull improve,
then raw overlap is a reserve/no-object artifact and should not trigger another
global diversity/competition repair.

If context/downstream overlap rises with reserve overlap:
  fix context budget/attention, not active object ownership.

If active overlap rises:
  revisit active owner/file competition.
```

First live read:

```text
step 50:
  loss_total                                  0.1581
  loss_anchor_object_pull                    0.3429
  loss_anchor_object_pull_graph              0.3186
  loss_anchor_object_pull_posterior          0.3899
  loss_anchor_object_pull_target_mass_mean   0.9129
  loss_anchor_pv                             0.6793
  loss_object_explanation_point              1.4307
  raw_same_role_support_overlap_max          0.4581
  active_same_role_support_overlap_max       0.0044
  context_same_role_support_overlap_max      0.1865
  reserve_same_role_support_overlap_max      0.3105
  downstream_same_role_support_overlap_max   0.2825
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       7.725
  reserve_anchor_fraction                    0.633
  posterior_recycle_rate                     0.0898
  owm_proposal_valid_fraction                0.8250
  owm_tracklet_valid_fraction                0.9690

initial interpretation:
  This does not reproduce the old active-owner collapse.  The active object
  rows are nearly non-overlapping, posterior duplicate overlap is zero, and
  tracklet/proposal sidecar dataflow is alive.  Continue to step 100/200 before
  classifying raw overlap as harmless reserve capacity, because prior runs
  sometimes showed delayed context/downstream rebound.

step 100:
  loss_total                                  0.1545
  loss_anchor_object_pull                    0.3100
  loss_anchor_object_pull_graph              0.3024
  loss_anchor_object_pull_posterior          0.3211
  loss_anchor_object_pull_target_mass_mean   0.9730
  loss_anchor_pv                             0.6230
  loss_object_explanation_point              1.8431
  raw_same_role_support_overlap_max          0.6192
  active_same_role_support_overlap_max       0.0099
  context_same_role_support_overlap_max      0.3447
  reserve_same_role_support_overlap_max      0.5555
  downstream_same_role_support_overlap_max   0.3753
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       7.325
  reserve_anchor_fraction                    0.646
  posterior_recycle_rate                     0.1012
  owm_proposal_valid_fraction                0.8492
  owm_tracklet_valid_fraction                0.9890

step-100 decision:
  continue.  This does not justify another architectural patch yet.  Active
  object owners remain almost non-overlapping, posterior active duplicate
  overlap remains zero, graph and posterior pull improve, and point/visual
  consistency improves.  Raw overlap rises with reserve/context overlap, which
  is exactly the ambiguity this audit was designed to expose.  The main watch
  item is loss_object_explanation_point, which rises from 1.43 to 1.84 and must
  be checked at 200/300 before deciding whether compactness/sidecar quality
  gating needs another repair.

step 150:
  loss_total                                  0.1708
  loss_anchor_object_pull                    0.3195
  loss_anchor_object_pull_graph              0.3142
  loss_anchor_object_pull_posterior          0.3282
  loss_anchor_pv                             0.5670
  loss_object_explanation_point              2.4404
  raw_same_role_support_overlap_max          0.9983
  active_same_role_support_overlap_max       0.0482
  context_same_role_support_overlap_max      0.3074
  reserve_same_role_support_overlap_max      0.9981
  downstream_same_role_support_overlap_max   0.4144
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       3.705
  reserve_anchor_fraction                    0.796
  posterior_recycle_rate                     0.1141

step-150 interpretation:
  raw overlap saturation is now explained almost entirely by reserve/no-object
  rows.  This is not the old active duplicate-owner failure: active overlap is
  still low and posterior active duplicate overlap remains zero.  However,
  context capacity is shrinking quickly and downstream overlap is rising into a
  moderate-risk band.  The next decision point is step 200:
    if active remains low and downstream stabilizes, classify raw as reserve
    telemetry and keep the architecture;
    if downstream/context continue to deteriorate, repair context budget /
    peripheral attention, not active owner binding.

step 200:
  loss_total                                  0.1393
  loss_anchor_object_pull                    0.2548
  loss_anchor_object_pull_graph              0.2539
  loss_anchor_object_pull_posterior          0.2556
  loss_anchor_object_pull_target_mass_mean   0.9898
  loss_anchor_pv                             0.5335
  loss_object_explanation_point              2.0168
  raw_same_role_support_overlap_max          0.9993
  active_same_role_support_overlap_max       0.0383
  context_same_role_support_overlap_max      0.2002
  reserve_same_role_support_overlap_max      0.9992
  downstream_same_role_support_overlap_max   0.3642
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       2.325
  reserve_anchor_fraction                    0.858
  posterior_recycle_rate                     0.1127
  owm_proposal_valid_fraction                0.8600
  owm_tracklet_valid_fraction                0.9792

step-200 interpretation:
  Continue to 300.  The old failure is not present: active overlap remains low,
  posterior duplicate overlap remains zero, downstream overlap decreases from
  the step-150 bump, and graph/posterior pull plus PV both improve.  The high
  raw overlap is now almost entirely a reserve/no-object effect.  Context count
  continues to compress; this should be monitored as a dense-context capacity
  question, but it is not evidence that object-owner binding collapsed.

step 250:
  loss_total                                  0.2955
  loss_anchor_object_pull                    0.6748
  loss_anchor_object_pull_graph              0.6410
  loss_anchor_object_pull_posterior          0.7427
  loss_anchor_object_pull_target_mass_mean   1.0100
  loss_anchor_pv                             0.5372
  loss_object_explanation_point              2.4484
  raw_same_role_support_overlap_max          0.9993
  active_same_role_support_overlap_max       0.0732
  context_same_role_support_overlap_max      0.2772
  reserve_same_role_support_overlap_max      0.9993
  downstream_same_role_support_overlap_max   0.4130
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       3.045
  reserve_anchor_fraction                    0.822
  posterior_recycle_rate                     0.1000
  owm_proposal_valid_fraction                0.8692
  owm_tracklet_valid_fraction                0.9688

step 300:
  loss_total                                  0.2417
  loss_anchor_object_pull                    0.5113
  loss_anchor_object_pull_graph              0.4674
  loss_anchor_object_pull_posterior          0.5964
  loss_anchor_object_pull_target_mass_mean   1.0415
  loss_anchor_pv                             0.5398
  loss_object_explanation_point              2.6793
  raw_same_role_support_overlap_max          0.9952
  active_same_role_support_overlap_max       0.0253
  context_same_role_support_overlap_max      0.3658
  reserve_same_role_support_overlap_max      0.9933
  downstream_same_role_support_overlap_max   0.4236
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       4.700
  reserve_anchor_fraction                    0.756
  posterior_recycle_rate                     0.1013
  owm_proposal_valid_fraction                0.9267
  owm_tracklet_valid_fraction                0.9986

final 300-step reserveaudit conclusion:
  The old catastrophic same-role active-owner collapse is not reproduced.
  active_same_role_support_overlap_max stays low across the full gate:
    0.0044 -> 0.0099 -> 0.0482 -> 0.0383 -> 0.0732 -> 0.0253
  posterior_active_duplicate_overlap_max stays exactly 0.0 at every measured
  gate.  Therefore the active object owners are not all becoming the same
  posterior object file.

  raw_same_role_support_overlap_max still saturates:
    0.4581 -> 0.6192 -> 0.9983 -> 0.9993 -> 0.9993 -> 0.9952
  but reserve_same_role_support_overlap_max tracks it:
    0.3105 -> 0.5555 -> 0.9981 -> 0.9992 -> 0.9993 -> 0.9933
  so the raw scalar is primarily a fixed-capacity reserve/no-object telemetry
  artifact, not evidence that active object binding collapsed.

  The remaining real watch items are:
    context/downstream capacity:
      downstream overlap stays moderate at 0.28-0.42 and context count
      compresses before recovering.
    posterior-side weak pull:
      loss_anchor_object_pull improves through step 200 but rebounds at
      250/300, with posterior pull higher than graph pull at the endpoint.
    point explanation compactness:
      loss_object_explanation_point remains noisy/rising, so sidecar quality
      should stay weak/gated and must not be treated as hard object truth.

decision:
  Do not add another global raw same-role overlap penalty.
  Do not force reserve/context rows to become distinct objects.
  Keep active/context/reserve decomposition as the acceptance contract.
  For the next full-policy gate, judge active owner collapse by active overlap,
  downstream overlap, posterior duplicate overlap, and split graph/posterior
  object-pull, not by raw fixed-capacity overlap alone.

completion:
  remote A7 run completed at 300/300 and saved:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_slot_reserveaudit_frozen_policy_300_20260521/300
  tmux exited normally and both A7 GPUs were idle after completion.

local validation after documenting the result:
  git diff --check: PASS
  py_compile pipeline/training/train/test entrypoints: PASS
  verify_picf_owm_contract.py: PASS
  picf_owm_strict_diagnose.py --fail-on-fail: PASS
  picf_owm_dataflow_trace.py --fail-on-fail: PASS
  picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
  targeted pytest suite: 140 passed
```

### 2026-05-21 Strict-Scope Reserve Audit Relaunch

Why this relaunch exists:

```text
The completed reserveaudit run was enough to classify raw overlap as mostly
reserve/no-object telemetry.  After that run, the local code tightened the
diagnostic definitions so active/context/reserve are mutually exclusive.  This
strict-scope relaunch verifies the same conclusion with the exact code that is
now in the workspace and with the paper-aligned complete-deployment rule
documented in PICF_AQR_OWM_SLOT_PAPER_DATAFLOW_COMPARE_20260521_TEMP.md.
```

Run:

```text
experiment:
  picf_a7_slot_reserveaudit_strictscope_frozen_policy_300_20260521

launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_slot_reserveaudit_strictscope_frozen_policy_300_20260521.sh

remote log:
  /mnt/picf_run_logs/
    picf_a7_slot_reserveaudit_strictscope_frozen_policy_300_20260521.log

contract:
  freeze PaliGemma/action head/Sonata/V-JEPA/AnyTouch;
  train PICF/AQR/posterior/object-file routing and sidecar/contact-motion path;
  inspect 100/200/300 before any long-run decision.
```

Local and remote preflight:

```text
local:
  git diff --check: PASS
  py_compile: PASS
  verify_picf_owm_contract.py: PASS
  picf_owm_strict_diagnose.py --fail-on-fail: PASS
  picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
  targeted pytest suite: 131 passed

remote A7:
  py_compile: PASS
  verify_picf_owm_contract.py: PASS
  picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
  launch started in tmux session:
    picf_a7_slot_reserveaudit_strictscope_300_20260521
```

Live gate:

```text
step 50:
  loss_total                                  0.1619
  loss_anchor_object_pull                    0.3529
  loss_anchor_object_pull_graph              0.3345
  loss_anchor_object_pull_posterior          0.3871
  loss_anchor_object_pull_target_mass_mean   0.9061
  loss_anchor_pv                             0.6839
  loss_object_explanation_point              1.4474
  raw_same_role_support_overlap_max          0.4596
  active_same_role_support_overlap_max       0.0053
  active_same_role_object_core_overlap_max   0.0005
  context_same_role_support_overlap_max      0.1884
  reserve_same_role_support_overlap_max      0.3185
  downstream_same_role_support_overlap_max   0.2909
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       7.655
  reserve_anchor_fraction                    0.629
  posterior_recycle_rate                     0.0920
  owm_proposal_valid_fraction                0.8250
  owm_tracklet_valid_fraction                0.9690

step 100:
  loss_total                                  0.1283
  loss_anchor_object_pull                    0.2351
  loss_anchor_object_pull_graph              0.2283
  loss_anchor_object_pull_posterior          0.2445
  loss_anchor_object_pull_target_mass_mean   0.9628
  loss_anchor_pv                             0.6355
  loss_object_explanation_point              1.8327
  raw_same_role_support_overlap_max          0.5988
  active_same_role_support_overlap_max       0.0107
  active_same_role_object_core_overlap_max   0.0023
  context_same_role_support_overlap_max      0.2938
  reserve_same_role_support_overlap_max      0.5038
  downstream_same_role_support_overlap_max   0.3301
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       7.265
  reserve_anchor_fraction                    0.645
  posterior_recycle_rate                     0.0940
  owm_proposal_valid_fraction                0.8492
  owm_tracklet_valid_fraction                0.9890

step-100 strict-scope interpretation:
  The old active-owner collapse is not reproduced.  Raw overlap rises, but the
  active object rows remain separated, active object-core overlap is nearly
  zero, posterior active duplicate overlap remains exactly zero, and both graph
  and posterior object-pull improve.  This is consistent with the fixed-capacity
  no-object/reserve interpretation from MetaSlot/QASA-style slot diagnostics.

decision:
  Continue to step 200/300.  Do not add another raw all-row overlap penalty.
  Only revisit model math if active overlap, downstream overlap, posterior
  active duplicates, or split graph/posterior pull deteriorate together.

step 150:
  loss_total                                  0.1407
  loss_anchor_object_pull                    0.2496
  loss_anchor_object_pull_graph              0.2419
  loss_anchor_object_pull_posterior          0.2632
  loss_anchor_object_pull_target_mass_mean   0.9454
  loss_anchor_pv                             0.5975
  loss_object_explanation_point              2.1452
  raw_same_role_support_overlap_max          0.9336
  active_same_role_support_overlap_max       0.0157
  active_same_role_object_core_overlap_max   0.0083
  context_same_role_support_overlap_max      0.3751
  reserve_same_role_support_overlap_max      0.8806
  downstream_same_role_support_overlap_max   0.4021
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       5.995
  reserve_anchor_fraction                    0.696
  posterior_recycle_rate                     0.1127

step 200:
  loss_total                                  0.1886
  loss_anchor_object_pull                    0.3949
  loss_anchor_object_pull_graph              0.4043
  loss_anchor_object_pull_posterior          0.3754
  loss_anchor_object_pull_target_mass_mean   0.9850
  loss_anchor_pv                             0.5692
  loss_object_explanation_point              1.9902
  raw_same_role_support_overlap_max          0.9918
  active_same_role_support_overlap_max       0.0311
  active_same_role_object_core_overlap_max   0.0039
  context_same_role_support_overlap_max      0.4122
  reserve_same_role_support_overlap_max      0.9891
  downstream_same_role_support_overlap_max   0.4424
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       5.715
  reserve_anchor_fraction                    0.709
  posterior_recycle_rate                     0.1049
  owm_proposal_valid_fraction                0.8600
  owm_tracklet_valid_fraction                0.9792

step-200 strict-scope interpretation:
  Raw overlap now saturates, but reserve overlap saturates with it. Active
  support overlap remains low, active object-core overlap remains near zero,
  and posterior active duplicate overlap remains exactly zero. This confirms
  that the old many-active-owner collapse is still not present.

  The nontrivial watch item is downstream/context capacity: downstream overlap
  reaches 0.4424 and context rows are still visible. This is a context-budget /
  dense-peripheral-routing question, not evidence that every active object
  owner has collapsed to the same object. Do not repair it with a global raw
  diversity term, because that would again penalize legitimate reserve/no-object
  rows and conflicts with the MetaSlot/QASA variable-count interpretation.

decision:
  Continue to step 300. At 300, classify the remaining issue as one of:
    healthy reserve telemetry,
    context/downstream budget pressure,
    or true active-owner collapse.

step 250:
  loss_total                                  0.3149
  loss_anchor_object_pull                    0.7115
  loss_anchor_object_pull_graph              0.7325
  loss_anchor_object_pull_posterior          0.6564
  loss_anchor_object_pull_target_mass_mean   1.0061
  loss_anchor_pv                             0.5683
  loss_object_explanation_point              2.7385
  raw_same_role_support_overlap_max          0.9980
  active_same_role_support_overlap_max       0.1062
  active_same_role_object_core_overlap_max   0.0069
  context_same_role_support_overlap_max      0.4420
  reserve_same_role_support_overlap_max      0.9977
  downstream_same_role_support_overlap_max   0.5199
  posterior_active_duplicate_overlap_max     0.0

step 300:
  loss_total                                  0.4438
  loss_anchor_object_pull                    1.0477
  loss_anchor_object_pull_graph              1.0905
  loss_anchor_object_pull_posterior          0.9829
  loss_anchor_object_pull_target_mass_mean   1.0397
  loss_anchor_pv                             0.5665
  loss_object_explanation_point              3.3188
  raw_same_role_support_overlap_max          0.9984
  active_same_role_support_overlap_max       0.1452
  active_same_role_object_core_overlap_max   0.0091
  context_same_role_support_overlap_max      0.4185
  reserve_same_role_support_overlap_max      0.9982
  downstream_same_role_support_overlap_max   0.5217
  posterior_active_duplicate_overlap_max     0.0
  context_anchor_count                       5.270
  reserve_anchor_fraction                    0.724
  posterior_recycle_rate                     0.1059
  owm_proposal_valid_fraction                0.9267
  owm_tracklet_valid_fraction                0.9986

final strict-scope diagnosis:
  The old active-owner duplicate collapse is still not reproduced:
    posterior_active_duplicate_overlap_max stays exactly 0.0;
    active object-core overlap remains near zero;
    active support overlap ends at 0.145, far below the old collapse band.

  However, the run is not a clean pass. The real late-stage issues are:
    1. weak object-pull drift:
       loss_anchor_object_pull improves through step 100, then rises sharply
       after 200. Both graph and posterior terms drift, so this is not only a
       posterior update bug.
    2. context/downstream pressure:
       downstream_same_role_support_overlap_max rises to about 0.52 while raw
       and reserve saturate. This means some same-role context capacity remains
       action-visible enough to duplicate broad supports.
    3. point explanation compactness:
       loss_object_explanation_point rises to 3.32, so the sidecar/contact
       object explanation remains a weak cue and must not be treated as hard
       object truth.

root-cause boundary:
  Do not fix raw overlap. Raw is explained by reserve/no-object capacity.
  Do inspect object-pull weighting/normalization and context/downstream budget.
```
```

### 2026-05-21 follow-up repair: object-pull target quality gate

Status:

```text
implemented locally; A7 rerun pending.
```

Root cause:

```text
The strict-scope run proved raw overlap was mostly reserve/no-object telemetry,
but it also showed late object-pull drift.  The previous object-pull target
used max-union over weak owner/proposal/task/point priors.  That can turn broad
or tailed contact-motion evidence into a hard geometric center.
```

Repair:

```text
1. Compute anchor_object_pull target center from a high-confidence TopCore of
   the weak target prior, not from the entire weak max-union tail.
2. Compute compactness quality q_j from the target core in point space.
3. Weight object-pull rows by q_j and drop rows below the minimum target
   quality threshold.
4. Restore quality-guided context gating by default, so low-quality duplicate
   context rows do not remain action-visible.
```

New metric:

```text
loss_anchor_object_pull_target_quality_mean
```

Local validation:

```text
targeted pytest:
  scripts/picf_core_train_test.py
  training_test::test_anchor_object_pull_ignores_broad_noisy_target_tail

result:
  129 passed

strict scripts:
  verify_picf_owm_contract.py PASS
  picf_owm_strict_diagnose.py --fail-on-fail PASS
  picf_owm_dataflow_trace.py --fail-on-fail PASS
  picf_owm_mvtrack_deep_audit.py --fail-on-fail PASS

core regression:
  pipeline_test.py + training_test.py
  134 passed
```

Next required gate:

```text
Launch:
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_qualitytarget_frozen_policy_300_20260521.sh

This is a fresh 300-step frozen PaliGemma/action-head validation on A7.
Do not open a 30000-step run until object_pull, object_explanation_point and
downstream overlap pass the 100/200/300 checks.
```

### 2026-05-21 A7 quality-target live gate

Run:

```text
picf_a7_slot_qualitytarget_frozen_policy_300_20260521
```

Config:

```text
frozen:
  PaliGemma
  PI0.5 action loss/head pressure
  Sonata/V-JEPA/AnyTouch pretrained backbones

trainable:
  PICF/AQR/posterior/OEML object-file stack

sidecar:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

step 50:

```text
loss_total                                  0.1475
loss_anchor_object_pull                    0.3128
loss_anchor_object_pull_graph              0.2699
loss_anchor_object_pull_posterior          0.3381
loss_anchor_object_pull_target_quality     0.5183
loss_object_explanation_point              1.4126
raw_same_role_support_overlap_max          0.4583
active_same_role_support_overlap_max       0.0061
active_same_role_object_core_overlap_max   0.0007
context_same_role_support_overlap_max      0.0213
downstream_same_role_support_overlap_max   0.0298
reserve_same_role_support_overlap_max      0.4212
posterior_active_duplicate_overlap_max     0.0
```

step 100:

```text
loss_total                                  0.1669
loss_anchor_object_pull                    0.3109
loss_anchor_object_pull_graph              0.2385
loss_anchor_object_pull_posterior          0.3164
loss_anchor_object_pull_target_quality     0.5078
loss_object_explanation_point              2.4107
raw_same_role_support_overlap_max          0.6349
active_same_role_support_overlap_max       0.0155
active_same_role_object_core_overlap_max   0.0020
context_same_role_support_overlap_max      0.0810
downstream_same_role_support_overlap_max   0.1126
reserve_same_role_support_overlap_max      0.6015
posterior_active_duplicate_overlap_max     0.0
```

step 150:

```text
loss_total                                  0.1845
loss_anchor_object_pull                    0.3438
loss_anchor_object_pull_graph              0.2747
loss_anchor_object_pull_posterior          0.3985
loss_anchor_object_pull_target_quality     0.4753
loss_object_explanation_point              2.6571
raw_same_role_support_overlap_max          0.8048
active_same_role_support_overlap_max       0.0089
active_same_role_object_core_overlap_max   0.0020
context_same_role_support_overlap_max      0.1011
downstream_same_role_support_overlap_max   0.1428
reserve_same_role_support_overlap_max      0.7657
posterior_active_duplicate_overlap_max     0.0
```

step 200:

```text
loss_total                                  0.1869
loss_anchor_object_pull                    0.3631
loss_anchor_object_pull_graph              0.3233
loss_anchor_object_pull_posterior          0.3883
loss_anchor_object_pull_target_quality     0.5216
loss_object_explanation_point              2.4545
raw_same_role_support_overlap_max          0.9531
active_same_role_support_overlap_max       0.0294
active_same_role_object_core_overlap_max   0.0049
context_same_role_support_overlap_max      0.0952
downstream_same_role_support_overlap_max   0.1586
reserve_same_role_support_overlap_max      0.9488
posterior_active_duplicate_overlap_max     0.0
posterior_recycle_rate                     0.0017
```

step 250:

```text
loss_total                                  0.2015
loss_anchor_object_pull                    0.4142
loss_anchor_object_pull_graph              0.3721
loss_anchor_object_pull_posterior          0.4419
loss_anchor_object_pull_target_quality     0.5103
loss_object_explanation_point              2.2829
raw_same_role_support_overlap_max          0.9921
active_same_role_support_overlap_max       0.0216
active_same_role_object_core_overlap_max   0.0032
context_same_role_support_overlap_max      0.1340
downstream_same_role_support_overlap_max   0.2046
reserve_same_role_support_overlap_max      0.9912
posterior_active_duplicate_overlap_max     0.0
posterior_recycle_rate                     0.0022
```

step 300:

```text
loss_total                                  0.1872
loss_anchor_object_pull                    0.3417
loss_anchor_object_pull_graph              0.2734
loss_anchor_object_pull_posterior          0.3023
loss_anchor_object_pull_target_quality     0.5728
loss_object_explanation_point              2.8956
raw_same_role_support_overlap_max          0.9831
active_same_role_support_overlap_max       0.0349
active_same_role_object_core_overlap_max   0.0035
context_same_role_support_overlap_max      0.1343
downstream_same_role_support_overlap_max   0.2125
reserve_same_role_support_overlap_max      0.9801
posterior_active_duplicate_overlap_max     0.0
posterior_recycle_rate                     0.0019
```

Interpretation through step 100:

```text
fixed:
  The context/downstream pressure is materially lower than strict-scope:
    step 100 downstream overlap 0.1126 vs old 0.3301.
  Active object rows remain clean:
    active support overlap 0.0155;
    active object-core overlap 0.0020;
    active duplicate overlap 0.0.
  Step-100 overlay visually places active anchors on the sidecar mask for
  move_the_door_to_the_right.

not yet fixed:
  loss_anchor_object_pull no longer falls to the old step-100 minimum
  0.2351; it stays around 0.31 because low-quality weak targets are no longer
  allowed to act as strong teachers.

watch:
  loss_object_explanation_point rises to 2.4107.  This is still below the old
  worst runs, but worse than strict-scope step 100 (1.8327).  Step 200/300 must
  decide whether this is bounded sidecar noise or a remaining target-quality
  problem.
```

Interpretation through step 200:

```text
fixed:
  The old late object-pull explosion is not reproduced:
    quality-target run: 0.3128 -> 0.3109 -> 0.3438 -> 0.3631
    strict-scope run:   0.3529 -> 0.2351 -> 0.2496 -> 0.3949
  The new run is more conservative at step 100, but it remains bounded by
  step 200 while strict-scope had already begun drifting.

  Downstream/context pressure is much lower:
    step 200 downstream overlap 0.1586 vs old 0.4424.

still unresolved:
  loss_object_explanation_point remains high:
    new step 200 2.4545 vs old step 200 1.9902.
  This suggests sidecar/contact point compactness is the remaining weak
  target-quality bottleneck, not active-owner duplicate collapse.

overlay audit:
  step 100 and step 200 active anchors are visually on the sidecar masks for
  the sampled door/switch tasks.  The current failure mode is therefore no
  longer "active anchor cannot reach the mask" in these samples.
```

Final 300-step gate conclusion:

```text
passed:
  active-owner duplicate collapse is fixed in this run:
    posterior_active_duplicate_overlap_max remains 0.0.
  active object rows remain separated:
    active support overlap <= 0.035;
    active object-core overlap <= 0.005.
  downstream/context pressure is much lower than strict-scope:
    step 300 downstream overlap 0.2125 vs old 0.5217.
  object-pull late explosion is fixed relative to strict-scope:
    new step 300 object_pull 0.3417 vs old 1.0477.
  target quality is now visible and bounded:
    target_quality_mean stays about 0.48-0.57.
  overlays at steps 100/200/300 put active anchors on the sidecar masks for
    sampled door/switch/red-block tasks.

not passed:
  loss_object_explanation_point remains high and noisy:
    new step 300 2.8956 vs old strict-scope 3.3188.
  This is an improvement over the old worst endpoint, but not a clean bounded
  compactness result.  It should be treated as the remaining sidecar/OEML point
  compactness issue before 30K.

decision:
  Do not open 30K yet on this exact setting.  The next repair should target
  OEML point explanation target quality/weighting or decouple that diagnostic
  from the object-pull success gate.  Do not reintroduce global raw-overlap
  penalties; raw overlap is still reserve/no-object telemetry.
```

### 2026-05-21 OEML point-quality repair

Root cause from code follow-through:

```text
pipeline:
  point_mask_j = normalized object explanation mask over point evidence.
  p_j = normalized point spatial variance of point_mask_j.
  point_quality_j = exp(-0.5 * clamp(p_j)).
  explanation_quality_j includes point_quality_j.

old loss:
  L_point = sum_j explanation_quality_j * p_j / sum_j explanation_quality_j

problem:
  If all active rows in a batch have noisy/sparse point masks, the same quality
  appears in numerator and denominator, so the reliability factor cancels.
  The term still behaves like a hard compactness label even when sidecar/contact
  evidence is weak, broad, or partially contaminated by trajectory tails.
```

Deployed repair:

```text
state:
  PicfObjectExplanationState now also stores:
    anchor_base_quality
    anchor_point_quality

loss:
  point compactness is now a quality-gated robust weak likelihood:

    rho(p) = -2 log((1-eta) exp(-p/2) + eta)

    L_point =
      sum_j base_quality_j * stopgrad(point_quality_j)^alpha * rho(p_j)
      / (sum_j base_quality_j + eps)

  defaults:
    eta = 0.05
    alpha = 1.0
    min point_quality = 0.05

diagnostic retained:
  oeml_point_spatial_variance_mean remains the raw compactness telemetry.
  oeml_point_quality_mean/max are now logged separately.
```

Why this is not a cosmetic loss suppression:

```text
It matches the QASA/MetaSlot principle that low-quality or duplicate/no-object
slots should not be treated as hard object truth.  It also preserves the PICF
contract: dense point/V-JEPA/tactile memory is not pruned, object-pull still
uses high-confidence target cores, and raw point variance remains visible.
```

Required next gate:

```text
run another 300-step frozen PaliGemma/action-head validation.

accept if:
  active overlap remains low;
  posterior active duplicate stays zero;
  object_pull remains bounded;
  downstream overlap stays below the old 0.5 failure band;
  loss_object_explanation_point is no longer the dominant rising term;
  oeml_point_spatial_variance_mean remains reported so bad sidecar masks are
  not hidden.
```

300-step ablation result:

```text
run:
  picf_a7_slot_qualitytarget_pointrobust_frozen_policy_300_20260521

good:
  loss_object_explanation_point:
    0.4764 -> 0.4794 -> 0.5041 -> 0.5089
  This proves the robust point-quality gate works mechanically.

bad:
  loss_anchor_object_pull:
    0.3275 -> 0.3375 -> 0.5315 -> 0.5416
  downstream_same_role_support_overlap_max:
    0.0350 -> 0.0756 -> 0.3364 -> 0.2588
  active_same_role_support_overlap_max:
    0.0061 -> 0.0098 -> 0.1135 -> 0.0809

comparison to previous quality-target run:
  previous step 300 object_pull       0.3417
  point-robust step 300 object_pull   0.5416
  previous step 300 downstream        0.2125
  point-robust step 300 downstream    0.2588

decision:
  Reject point-quality-gated OEML point loss as production default.  It fixes
  the scalar point loss but weakens useful geometric compactness pressure.
  Keep the implementation as explicit guarded ablation only:
    object_explanation_point_quality_gate_enabled = false by default
    object_explanation_point_outlier_prior = 0 by default

production interpretation:
  High loss_object_explanation_point is not by itself a blocker if:
    active overlap remains low;
    downstream overlap remains below the old failure band;
    object_pull stays bounded;
    overlays show active owners on sidecar/contact masks.
  It should remain a raw weak-target compactness diagnostic, not a required
  scalar to suppress.
```

### 2026-05-21 default-sync frozen-policy gate

After rejecting the point-quality robust gate as a production default, the local
validated code was re-synced to A7 and relaunched as:

```text
picf_a7_slot_qualitytarget_defaultsync_frozen_policy_300_20260521
```

Run contract:

```text
trainable:
  PICF/AQR/posterior/OEML/router stack

frozen:
  PaliGemma
  PI0.5 action head/loss pressure
  V-JEPA/Sonata/AnyTouch pretrained modules

explicitly not enabled:
  object_explanation_point_quality_gate_enabled
  object_explanation_point_outlier_prior
```

step50:

```text
loss_total                                      0.1482
loss_anchor_object_pull                        0.3146
loss_anchor_object_pull_graph                  0.2711
loss_anchor_object_pull_posterior              0.3408
loss_anchor_object_pull_target_quality_mean    0.5144
loss_object_explanation_point                  1.4150
oeml_point_spatial_variance_mean               1.5135
aqr_active_same_role_support_overlap_max       0.0057
aqr_downstream_same_role_support_overlap_max   0.0329
aqr_same_role_support_overlap_max              0.4538
posterior_file_competition_active_duplicate_overlap_max 0.0
```

step100:

```text
loss_total                                      0.1496
loss_anchor_object_pull                        0.2918
loss_anchor_object_pull_graph                  0.1362
loss_anchor_object_pull_posterior              0.3138
loss_anchor_object_pull_target_quality_mean    0.5075
loss_object_explanation_point                  1.9014
oeml_point_spatial_variance_mean               1.1603
aqr_active_same_role_support_overlap_max       0.0039
aqr_downstream_same_role_support_overlap_max   0.0438
aqr_same_role_support_overlap_max              0.4514
posterior_file_competition_active_duplicate_overlap_max 0.0
```

Interpretation at step100:

```text
The production default recovered the important owner metrics that the
point-robust ablation damaged.  Object-pull improves from step50 to step100,
active/downstream overlap remains very low, and active duplicate overlap stays
zero.  The raw point compactness scalar is higher than the point-robust
ablation by design, but raw point spatial variance is lower than step50 and no
longer justifies another scalar-suppression repair.

Continue this gate to step200/300.  Do not stop it at step100.
```

completed step sweep:

```text
step  loss_total  object_pull  graph_pull  posterior_pull  point_loss  point_var  active_ov  downstream_ov  raw_ov  reserve_ov  active_dup
  50     0.1482       0.3146      0.2711         0.3408       1.4150     1.5135     0.0057        0.0329     0.4538    0.4148       0
 100     0.1496       0.2918      0.1362         0.3138       1.9014     1.1603     0.0039        0.0438     0.4514    0.4289       0
 150     0.1369       0.2357      0.2120         0.2493       2.1885     0.5581     0.0312        0.1241     0.8716    0.8501       0
 200     0.1649       0.3243      0.2796         0.3499       2.0545     0.3591     0.0478        0.1870     0.9838    0.9783       0
 250     0.1738       0.3540      0.3060         0.3875       1.9767     0.5740     0.0680        0.2204     0.9943    0.9927       0
 300     0.1231       0.2157      0.1528         0.2173       1.9348     0.3652     0.0314        0.1498     0.9962    0.9959       0
```

completed decision:

```text
This default-sync gate passes the frozen-policy structural check.

What is fixed in this gate:
  active-owner duplicate collapse is absent;
  active support overlap remains low;
  downstream overlap stays far below the old 0.5-0.8 failure band;
  object-pull is bounded and improves strongly by step300;
  point-robust suppression is not needed as a production default.

What is not fixed by this gate:
  raw overlap still saturates because reserve/no-object rows reuse capacity;
  this is expected and should not be treated as the same old active collapse;
  behavior acceptance still requires action-aware long training and CALVIN/video
  evidence.
```

2026-05-21 default-sync 30K action-aware follow-up:

```text
launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_actionaware_defaultsync_long30k_ckpt500_20260521.sh

purpose:
  Test whether the frozen-policy structural repair survives production-relevant
  action/PaliGemma pressure.

contract:
  action_weight=0.50
  semantic_trainable=true
  semantic_lr_scale=0.25
  frozen pretrained backbones: Sonata, V-JEPA, AnyTouch
  steps=30000
  save_interval=500
  keep_last_checkpoints=3
  log_interval=50
  anchor_overlay_interval=100

online smoke rule:
  Treat the first 100-200 steps as a live gate.  Stop early only if active
  owner overlap, posterior active duplicate overlap, recycle saturation, or
  overlays reproduce the historical active-collapse failure.  Do not stop only
  because raw same-role overlap saturates; the completed reserve audit shows
  raw overlap is dominated by reserve/no-object fixed-capacity rows.

action-weight rule:
  Do not start above 0.50.  If action_default_equiv plateaus while structure
  remains healthy, resume from one of the 500-step checkpoints with a staged
  action-weight intervention.  This keeps the plateau question separate from
  the structural acceptance question.
```

Planned action-dominant continuation after step500:

```text
script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_actionaware_defaultsync_action2_from500_long30k_20260521.sh

resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_actionaware_defaultsync_long30k_ckpt500_20260521/500

action_weight:
  2.0

gate before launching:
  step500 checkpoint exists;
  active duplicate overlap is 0 or near 0;
  active same-role support overlap remains below the historical fail band;
  recycle is not saturated;
  overlays do not show active-owner collapse.

gate after launching:
  watch to at least step700.  If action_weight=2.0 immediately drives active
  collapse or recycle saturation, stop this continuation and retain the 0.50
  run as the safer baseline.
```

Step500 handoff observation:

```text
source run:
  picf_a7_actionaware_defaultsync_long30k_ckpt500_20260521

step500 metrics:
  loss_total=0.6659
  loss_action_default_equiv=0.0644
  aqr_active_same_role_support_overlap_max=0.1394
  aqr_downstream_same_role_support_overlap_max=0.3168
  aqr_same_role_support_overlap_max=0.9997
  aqr_reserve_same_role_support_overlap_max=0.9997
  posterior_file_competition_active_duplicate_overlap_max=0
  posterior_recycle_rate=0.0004
  loss_anchor_object_pull=1.4949

decision:
  The hard active-owner gate passed, so the action-dominant continuation was
  allowed.  The elevated object-pull/downstream metrics make this a pressure
  test, not a clean final acceptance.  If action=2.0 worsens active overlap,
  recycle, or object-pull, revert to the safer action=0.50 baseline and treat
  the owner-target alignment term as the next repair target.
```

Action-dominant step700 observation:

```text
run:
  picf_a7_actionaware_defaultsync_action2_from500_long30k_20260521

resume:
  source step=500
  action_weight=2.0
  loss_action_weight_scale=1.0

step  total   align   action_eq  active_ov  downstream_ov  active_dup  recycle  object_pull
550   0.4960  0.4350  0.0609     0.0601     0.3159         0           0.0791   0.9662
600   0.4952  0.4384  0.0568     0.0880     0.3506         0           0.1172   0.9592
650   0.3626  0.3111  0.0515     0.1803     0.3514         0           0.1143   0.5487
700   0.3390  0.2930  0.0460     0.1285     0.2337         0           0.1169   0.4271

interpretation:
  The action-dominant continuation did not reproduce the old active-collapse or
  recycle-saturation failures through step700.  It improved both action and
  owner-target alignment after the elevated step500 handoff.  Continue the run,
  but keep watching downstream overlap and overlays at later 100-step intervals.
```

Action-dominant step1000 observation:

```text
run:
  picf_a7_actionaware_defaultsync_action2_from500_long30k_20260521

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_actionaware_defaultsync_action2_from500_long30k_20260521/1000

step  total   align   action_eq  active_ov  downstream_ov  active_dup  recycle  object_pull
550   0.4960  0.4350  0.0609     0.0601     0.3159         0           0.0791   0.9662
600   0.4952  0.4384  0.0568     0.0880     0.3506         0           0.1172   0.9592
650   0.3626  0.3111  0.0515     0.1803     0.3514         0           0.1143   0.5487
700   0.3390  0.2930  0.0460     0.1285     0.2337         0           0.1169   0.4271
750   0.3177  0.2658  0.0519     0.0889     0.1269         0           0.1199   0.3262
800   0.3065  0.2587  0.0478     0.0891     0.1200         0           0.1204   0.3184
850   0.2599  0.2073  0.0526     0.0602     0.1451         0           0.1222   0.1437
900   0.2442  0.1947  0.0496     0.0600     0.1845         0           0.1269   0.0905
950   0.2813  0.2315  0.0498     0.0649     0.1614         0           0.1279   0.2002
1000  0.3547  0.3025  0.0522     0.1336     0.1908         0           0.1199   0.4006

decision:
  The run is healthy enough to continue.  Action has moved down versus the
  first action-2.0 interval but is oscillating around 0.05, so do not raise
  action weight above legacy/default scale yet.  The old hard failures remain
  absent: active duplicate overlap is 0, active overlap stays low, recycle does
  not saturate, and downstream overlap is controlled.  Owner-target pull bounced
  at step1000 after improving sharply through step900; keep it as the next watch
  item rather than restarting.  Raw slot-JEPA telemetry is large, but this run
  has lambda_slot_jepa=0; it is not part of the optimized objective.
```

## 2026-05-22 Weak-Scaffold Decay Issue

Observed in `picf_a7_actionaware_defaultsync_action2_from0_long30k_20260522`:

```text
step  action_eq  alignment  object_pull  object_point  active_ov  downstream_ov
100   0.0720     0.1258     0.2260       1.8507        0.0228     0.1142
200   0.0599     0.2328     0.4773       2.7311        0.1165     0.2977
300   0.0628     0.3510     0.7561       3.7503        0.2277     0.3230
450   0.0639     0.5343     1.1703       5.6222        0.2379     0.3391
```

Root cause:

```text
The action scale is now normal, but sidecar/OEML scaffold remains at the
bootstrap strength forever.  By step450, object_pull alone contributes about
0.410 to total loss, while action contributes about 0.064.  The sidecar teacher
is therefore no longer weak measurement evidence; it dominates optimization.
This is a curriculum/weighting error, not a reason to add another slot module.
```

Accepted repair:

```text
code:
  scripts/picf_core_train.py now supports object-scaffold decay.

fields scaled:
  lambda_anchor_object_pull
  lambda_object_explanation_point
  lambda_object_explanation_contact
  lambda_object_explanation_duplicate
  lambda_object_explanation_background
  lambda_mapg_support_diversity

production continuation:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_actionaware_qgdecay_from500_long30k_20260522.sh

schedule:
  cosine, start=500, end=1500, floor=0.10

quality:
  anchor_object_pull_target_quality_power=2.0
  anchor_object_pull_target_quality_min=0.05
  object_explanation_point_quality_gate remains disabled by default because
  the previous ablation reduced the scalar point loss but worsened owner pull.
```

Expected effect:

```text
At the step450 scale, a 0.10 long-run scaffold floor would reduce the dominant
sidecar contribution from about 0.53 to about 0.053, comparable to action.
During 500-1500 cosine decay, compact masks still guide owner binding, while
diffuse/noisy masks stop dominating action and dense context learning.
```

Acceptance:

```text
must improve:
  loss_total_minus_action should fall after decay begins
  loss_action_default_equiv should leave the 0.06 plateau or at least not worsen
  loss_anchor_object_pull should no longer rise monotonically
  loss_object_explanation_point can remain noisy telemetry but its weighted
  contribution must no longer dominate total

must stay safe:
  posterior_file_competition_active_duplicate_overlap_max == 0
  active_same_role_support_overlap_max < 0.30
  downstream_same_role_support_overlap_max < 0.45
  posterior_recycle_rate not saturated
```

## 2026-05-24 Action-Phase / Optimizer-State Issue

Current runs:

```text
A7 continuous long run:
  picf_a7_actionprefix_rmsnorm_long30k_20260524

A5 model-only resume / fresh-optimizer probe:
  picf_a5_optreset_from1000_action2_probe1500_20260524
  resume ckpt: A7 step1000 model-only checkpoint
```

Latest checked metrics:

```text
A7 step1700:
  loss_total                         0.0628
  loss_action_default_equiv          0.0542
  loss_action_active7                0.2456
  loss_anchor_object_pull            0.1398
  loss_anchor_pv                     0.6312
  loss_slot_jepa_raw                 13598.9
  aqr_active_same_role_overlap_max   0.0461
  aqr_downstream_same_role_overlap   0.1122
  posterior_identity_switch_rate     0.1800
  posterior_recycle_rate             0.1357
  grad_norm                          0.278
  nonfinite action prefix            0

A5 step1350:
  loss_total                         0.0656
  loss_action_default_equiv          0.0466
  loss_action_active7                0.2112
  loss_anchor_object_pull            0.2512
  loss_anchor_pv                     0.5588
  loss_slot_jepa_raw                 5.08
  aqr_active_same_role_overlap_max   0.0943
  aqr_downstream_same_role_overlap   0.1002
  posterior_identity_switch_rate     0.1839
  posterior_recycle_rate             0.1308
  grad_norm                          0.282
```

Interpretation:

```text
fixed / healthy relative to the old failure modes:
  active overlap no longer collapses toward 1.0;
  downstream overlap is controlled;
  recycle is not saturating;
  action prefix nonfinite rate remains 0;
  gradients are small/stable;
  A5 fresh-optimizer continuation proves the representation can still lower
  action after the A7 step1000 plateau.

still open:
  A7 continuous optimizer state lets action hover around 0.05 instead of
  continuing the A5-style drop toward 0.04;
  raw loss_slot_jepa can explode on A7, but lambda_slot_jepa is zero and it is
  diagnostic only;
  loss_anchor_object_pull still oscillates by sample/owner quality and should
  be treated as a weak-teacher measurement, not as hard truth;
  raw same-role overlap remains near 1.0, but current action-visible active
  and downstream overlaps are the acceptance metrics.
```

Root cause hypothesis:

```text
This is no longer primarily an anchor-collapse problem.  It is an optimization
phase-boundary problem.  The same model checkpoint improves action much faster
when resumed with fresh optimizer/scheduler state on A5.  In Adam-style
training, stale first/second moments from the earlier mixed belief/object phase
can become miscalibrated after the objective shifts to action-dominant polish.

This does not usually appear as strongly in vanilla PI0.5 because its objective
is more homogeneous: action learning is the main target from the beginning.
PICF has a deliberate phase change: weak object scaffold -> active object
ownership -> action utilization.  That phase change should be represented in
the optimizer state, not only in scalar loss weights.
```

Accepted next experiment:

```text
Run a staged model-only resume at the action-polish boundary.

Candidate boundary:
  ckpt1000 or ckpt1500, whichever has healthier overlays and active/downstream
  overlap.

Stage-2 recipe:
  resume model weights only;
  reset Adam/optimizer state;
  use short-tail or lower-LR action-polish schedule;
  keep action weight at normal/default-dominant scale;
  keep sidecar/object scaffold at weak floor;
  keep lambda_slot_jepa/support_pred/binding_consistency = 0;
  keep anchor overlays every 100 steps and loss details every 50 steps.

Acceptance:
  loss_action_default_equiv should beat the continuous A7 run over the same
  step interval;
  active/downstream overlap should remain controlled;
  loss_anchor_pv and loss_anchor_object_pull should not monotonically worsen;
  raw slot_jepa explosion remains non-blocking only while lambda_slot_jepa=0.
```

## 2026-05-25 Action-Polish Continuity Status

Accepted continuation:

```text
picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524
```

Step-2500 status:

```text
loss_action_default_equiv          0.0363
loss_total                         0.0496
loss_anchor_pv                     0.5316
loss_anchor_object_pull            0.4165
active same-role support overlap   0.1300
downstream same-role support       0.1289
posterior_identity_switch_rate     0.2106
posterior_recycle_rate             0.1284
```

Closed by this run, relative to the prior May failure modes:

```text
1. The action objective is no longer trapped in the A7 continuous 0.05-0.056
   band.  It reached 0.0345 at step2350 and 0.0363 at step2500.

2. Active/downstream anchor overlap has not returned to the old 0.9+ collapse
   regime.

3. Posterior recycle is not saturated.

4. The step1500 model-only/fresh-optimizer transition is a useful phase-boundary
   repair and should be represented explicitly in future training recipes.
```

Still open:

```text
1. CALVIN/video behavior evidence is still pending.  The 2500 checkpoint is a
   valid small-eval candidate, but the active run should not be interrupted
   solely for eval while it is still improving.

2. loss_anchor_object_pull remains noisy because sidecar/contact proposal
   quality is weak-teacher evidence, not hard object truth.

3. raw same-role overlap remains 1.0 and must continue to be interpreted as
   reserve/inactive telemetry unless active/downstream overlap also fails.

4. raw slot_jepa remains telemetry only; lambda_slot_jepa must stay 0 until the
   normalized/matched version has its own acceptance run.
```

Next gate:

```text
Watch step3000 and step3500 before changing weights or resetting optimizer.
If action remains in the 0.03x band and active/downstream overlap stays below
the warning band, preserve continuity and continue the run.
```

Follow-up causal probe:

```text
run:
  picf_pc1_from_a7_2500_freshopt_lr5e5_30k_20260525

purpose:
  resume A7 step2500 with fresh optimizer to determine whether the later
  action rebound is optimizer-state/phase driven or checkpoint/data-state
  driven.

important:
  use num_train_steps=30000 even though the run may be stopped early.  A short
  total horizon changes the LR schedule and is not comparable to the maintained
  long-run recipe.
```

Low-LR control:

```text
run:
  picf_pc1_from_a7_2500_freshopt_lr2e5_30k_20260525

purpose:
  reuse the exact A7 step2500 checkpoint and fresh-optimizer setup, but lower
  LR from 5e-5 to 2e-5 while keeping the 30000-step schedule.  This separates
  fresh-optimizer benefit from possible high-LR structure noise.

acceptance:
  action should stay materially below the A7 continuous 0.045-0.053 band;
  active/downstream same-role overlap should not climb above the warning band;
  anchor-object-pull variance should be lower than the 5e-5 probe.
```

Execution correction:

```text
The strict A7 step2500 LR=2e-5 control is no longer runnable because the A7
checkpoint retention window removed the step2500 directory.  The active
replacement is:

  picf_pc1_from_a7_4500_freshopt_lr2e5_30k_20260525

This is a current-phase test, not a strict step2500 paired control.
```

2026-05-26 update:

```text
The 2e-5 PC1 current-phase control has been stopped after showing safe structure
but only partial action release.  The maintained A7 line remains running as the
continuous-optimizer reference.

New active issue probe:
  picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526

Question:
  Is the rebound mainly an underpowered optimizer phase at 2e-5, or does it
  persist under a middle LR fresh-optimizer continuation from the more mature
  A7 step5500 weights?

Checkpoint retention:
  save every 1000 steps, keep last 5.
```

PC1 mid-LR conclusion and frozen-PICF causal probe:

```text
Resolved evidence:
  picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526
  briefly reached loss_action_default_equiv ~= 0.0285 but rebounded to the
  0.04x band by steps 6550-6850.

Open root-cause question:
  Does action rebound happen because PICF core belief/prefix values keep moving
  under structural/object losses while the policy path is adapting?

New diagnostic:
  picf_pc1_freezepicf_policyonly_from_pc1_6000_action2_30k_20260526

Required invariants:
  picf_trainable_scope=policy_only
  all core.* parameters frozen
  semantic/action non-core path trainable
  all structural/object scaffold losses set to 0
  PICF forward still enabled; do not switch to picf_mode=ablated

Why this is not a production recipe:
  This is a causal probe.  It intentionally prevents PICF from improving so we
  can test whether a stationary prefix removes rebound.  If successful, the
  production answer is staged prefix stabilization or a gated context/prefix
  schedule, not permanently freezing PICF.
```

## 2026-05-29 Step-Indexed Resume Bug Closure

Status:

```text
Resolved as code-level bug, still under corrected-run behavior monitoring.
```

Root cause:

```text
The old resumed 7000->7600 action rebound evidence was contaminated by a
stateful sampled-window RNG.  Resume branches replayed early-window streams, so
step-matched action curves were not valid causal comparisons.
```

Fix:

```text
scripts/picf_core_train.py:
  _step_indexed_window_rng(seed, rank, step, micro_step, retry_count)
  source.window(flat_index, rng=sample_rng)
  --step-indexed-window-rng default true
  --no-step-indexed-window-rng legacy reproduction only

scripts/verify_picf_owm_contract.py:
  trainer_window_rng_is_resume_safe

scripts/picf_core_train_test.py:
  test_step_indexed_window_rng_is_resume_stable_and_step_specific
  test_normalize_train_args_enables_step_indexed_window_rng_by_default
```

Current monitored run:

```text
picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529

Observed corrected metrics:
  7050 action_default=0.050193 active/downstream=0.044/0.057
  7100 action_default=0.051401 active/downstream=0.070/0.078
  7150 action_default=0.042441 active/downstream=0.110/0.110
  7200 action_default=0.037938 active/downstream=0.080/0.078
  7250 action_default=0.050864 active/downstream=0.085/0.093
  7300 action_default=0.043123 active/downstream=0.080/0.084
  7350 action_default=0.039770 active/downstream=0.085/0.084
  7400 action_default=0.045796 active/downstream=0.065/0.068
  7450 action_default=0.045965 active/downstream=0.135/0.128
  7500 action_default=0.042314 active/downstream=0.165/0.175
  7550 action_default=0.044523 active/downstream=0.085/0.090

Decision:
  Structural gates passed through 7550, but the run was stopped as a production
  candidate because action remained around 0.04 instead of the expected 0.02
  reference range.

Next required issue:
  Add and run a fixed-window no-update action probe before any further
  architecture/loss rewrite.

Implemented tool:
  scripts/picf_fixed_window_action_probe.py

Why:
  The corrected step-indexed stream is not the same sampled-window stream as
  the old 0.02 reference, so live train-window scalar loss is not a stationary
  comparator.  Compare preserved checkpoints on identical accepted windows.
```

Local validation added to the root-cause document:

```text
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
```

Additional strict local audit rerun:

```text
docs/PICF_AQR_OWM_STEPINDEXED_RUN_LOCAL_AUDIT_20260529_TEMP.md

Result:
  compileall, contract verifier, strict diagnose, dataflow trace,
  MVTrack deep audit, professor-grade audit, binding/competition/action-visible
  audits, focused trainer tests, and full PICF regression all pass.

Current state:
  Step-indexed resume bug is code-closed.  Production action-quality acceptance
  remains open pending fixed-window probe results.
```

## 2026-05-31 Action-Visible Interface Bottleneck

Status:

```text
Open under E7 diagnostic; root cause narrowed substantially.
```

Evidence already established:

```text
Fixed-window no-update probe on the same 24 accepted windows:

gate0:
  loss_action_default_equiv=0.064012

gate07:
  loss_action_default_equiv=0.064106

fusion24:
  loss_action_default_equiv=0.064091

append24:
  loss_action_default_equiv=0.076344
```

Interpretation:

```text
PICF prefix/context is not exerting useful action leverage through passive
extra-prefix or fixed-prefix fusion. Direct append is harmful because it grows
the PI prefix and shifts action suffix positions. This rules out raw overlap as
the primary action plateau cause at this checkpoint.
```

Root-cause dataflow defects found while deploying E7:

```text
1. PICF context width != action suffix width.
   PICF context: 2048 PI prefix width.
   Action suffix: 1024 action expert width.
   Fix: explicit trainable action_context_in_proj before suffix cross-attn.

2. FSDP cannot manage a scalar gate parameter.
   Fix: action_context_gate_logit is a [1] tensor.

3. FSDP use_orig_params=False cannot flatten mixed frozen/trainable params
   inside adapter-only semantic root.
   Fix for this diagnostic: use DDP.
   Production full semantic cotrain can still use FSDP because the semantic root
   is uniformly trainable under backbone_only/model_only.
```

Implemented E7 design:

```text
action_context_integration=suffix_cross_attention
semantic_trainable_scope=action_adapter_only
action_context_stopgrad=1
picf_core_lr_scale=0.001
training_strategy=ddp
```

Acceptance:

```text
E7 must show pi_context_adapter_* metrics in JSONL and demonstrate whether
action-side cross-attention can lower action loss versus E6. If it does not,
the remaining bottleneck is downstream action-head/backbone adaptation rather
than slot routing or raw overlap.
```

E7 step9150 result:

```text
loss_action_default_equiv=0.047400
loss_total_minus_action=0.010265
active/downstream overlap=0.200000 / 0.188100
pi_context_adapter_token_count=28
pi_context_adapter_gate=0.119178
pi_context_adapter_attention_entropy_mean=3.075562
pi_context_adapter_residual_rms_mean=0.269857
```

Decision:

```text
adapter dataflow is alive, but adapter-only has not improved action loss over
the step9100 prefix-fusion baseline (`0.044162`).  This is not evidence for a
slot-overlap relapse; it narrows the live root cause to action-side readout
adaptation.

Next:
  run E8 from the same step9100 checkpoint with
  `semantic_trainable_scope=action_head_and_adapter` and the same
  `suffix_cross_attention` path.
```

E7 step9200 confirmation:

```text
loss_action_default_equiv=0.047333
loss_total_minus_action=0.010093
active/downstream overlap=0.100000 / 0.097959
loss_slot_jepa=0.862420
loss_mapg_routing=0.422907
```

Closure:

```text
E7 is stopped at the 9200 decision gate.  Structure got healthier while action
did not, so repeating overlap/slot-loss repairs here would be non-causal.  E8
is the correct next diagnostic: suffix_cross_attention with action head/time
MLP trainable.
```

E8 step9200 result:

```text
scope:
  action_head_and_adapter

loss_action_default_equiv:
  9150 = 0.047409
  9200 = 0.047135

grad_norm_group_semantic_backbone:
  9150 = 0.239949
  9200 = 0.177523

active/downstream overlap at 9200:
  0.100000 / 0.097957
```

Decision:

```text
E8 proves action-head/time-MLP gradients are live but insufficient.  The live
root is now the frozen semantic/action transformer interface: the action
expert/backbone must cotrain with the suffix-side PICF belief adapter.  Start
E9 with `semantic_trainable_scope=backbone_only` and keep the same suffix
cross-attention path.
```

E9 implementation blocker:

```text
status:
  first E9 launch crashed before training

error:
  Action context adapter dimension mismatch:
  context=(1, 28, 2048) suffix=(1, 16, 1024)

root cause:
  FSDP wraps `action_context_in_proj`, so runtime type is no longer bare
  `nn.Linear`.  The projection bridge rejected a valid wrapped module before
  calling it.  Therefore this is an implementation compatibility issue, not
  evidence that `backbone_only + suffix_cross_attention` failed.

fix:
  projection bridge now accepts generic `nn.Module` wrappers, optionally
  unwraps `.module` / `._fsdp_wrapped_module` for feature checks, calls the
  projection, then validates output width.

validation:
  local wrapper/policy tests and contract/dataflow audits pass.

next:
  rerun E9 as
  `picf_a7_stepindexed_from9100_suffixadapter_backbone_retry1_9400_20260531`.
```

E9 retry1 follow-up:

```text
status:
  retry1 passed the projection mismatch and entered the first FSDP training
  step, then rank0 terminated with SIGSEGV before metrics.

root cause:
  The adapter projections were incorrectly treated as nested FSDP runtime hot
  leaves.  They are tiny trainable interface projections, not large transformer
  leaves.  Nested full-sharding them adds no mathematical value and destabilizes
  the production backward path.

fix:
  keep `action_context_*` trainable but remove them from
  `fsdp_runtime_leaf_module_specs`; they now live under the semantic root FSDP
  boundary.  This preserves the same suffix-cross-attention map while removing
  the unstable nested-FSDP small-module boundary.

next:
  run retry2 and require at least the 9150 metrics row before treating E9 as a
  real loss diagnostic.
```

E9 retry2 closure:

```text
status:
  mechanically passed; scientifically invalid as final action diagnostic

observed:
  step9150 action_default_equiv=0.048359
  non_action=0.010331
  active/downstream overlap=0.170000/0.171056
  adapter gate/residual=0.119175/0.271960
  semantic_backbone grad=0.339463

why not final:
  This run used NUM_TRAIN_STEPS=9400 from a step9100 checkpoint, so the cosine
  LR schedule was already at the tail (`semantic_backbone lr ~= 2.0e-5`).  The
  same project documentation says resumed action diagnostics must preserve the
  30K horizon; otherwise the optimizer/LR condition is not comparable.

fix:
  `run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh` now
  defaults to NUM_TRAIN_STEPS=30000 and an h30k experiment name.  Rerun this
  production-horizon E9 before declaring the backbone/action-transformer path
  ineffective.

remaining open issue:
  action plateau under the correct production LR horizon.  Do not reopen raw
  overlap or sidecar-noise as primary causes unless active/downstream overlap
  or non-action budget fails at the corrected horizon.
```

E9-h30k result:

```text
run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_h30k_20260531

9150:
  action_default_equiv=0.055658
  non_action=0.010807
  active/downstream=0.134979/0.136243
  slot_jepa=0.671574
  semantic_backbone_lr=5.9407e-5

9200:
  action_default_equiv=0.055003
  non_action=0.010334
  active/downstream=0.089954/0.082378
  slot_jepa=0.584209
  semantic_backbone_lr=5.9300e-5
```

Root-cause update:

```text
The corrected h30k horizon does not rescue E9.  It makes the structure healthier
but action worse than E7/E8.  This strongly rejects raw overlap, slot_jepa, and
sidecar structure as the immediate cause of the action plateau.  The live cause
is the action-side cotrain boundary: fully trainable semantic/action backbone at
`SEMANTIC_LR_SCALE=1.0` drifts the PI action basis faster than the new suffix
PICF adapter can become useful.

Next run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_sem035_h30k_20260531

Only intended repair:
  keep suffix_cross_attention and h30k, but restore the large backbone LR scale
  to 0.35 while leaving policy_head LR high.
```

E10 result:

```text
run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_sem035_h30k_20260531

9150:
  action_default_equiv=0.048495
  non_action=0.011015
  active/downstream=0.184977/0.186723
  object_pull=0.225764
  slot_jepa=0.729864
  semantic_backbone_lr=2.079e-5

9200:
  action_default_equiv=0.047530
  non_action=0.010215
  active/downstream=0.089956/0.086722
  object_pull=0.150400
  slot_jepa=0.583056
  semantic_backbone_lr=2.075e-5
```

Root-cause update:

```text
E10 resolves the E9 high-LR degradation but does not exceed E7/E8 and remains
worse than the E6 prefix-fusion reference.  The immediate cause is no longer
raw overlap or slot loss; the live unknown is the action-interface topology.

Next matched control:
  picf_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531

Only intended change from E10:
  ACTION_CONTEXT_INTEGRATION=suffix_cross_attention -> prefix_fusion

E11 startup:
  tmux=e11_prefixfusion_sem035_h30k_20260531
  resumed_step=9100
  num_steps=30000
  action_context_integration=prefix_fusion
  semantic_trainable_scope=backbone_only
  semantic_backbone_lr=2.45e-5
  policy_head_lr=7.0e-5
  picf_core_lr=7.0e-8
```

E11 result:

```text
9150:
  action_default_equiv=0.047375
  non_action=0.010323
  active/downstream=0.184988/0.176964
  anchor_pv=0.501754
  object_pull=0.159862
  slot_jepa=0.803113
  semantic_backbone_lr=2.079e-5

9200:
  action_default_equiv=0.047304
  non_action=0.010585
  active/downstream=0.114996/0.103346
  anchor_pv=0.500755
  object_pull=0.187458
  slot_jepa=0.777948
  semantic_backbone_lr=2.075e-5
```

Issue update:

```text
E11 does not recover the E6 prefix-fusion reference band.  It is only marginally
better than E10 and essentially in the E7/E8 band.  This means the remaining
action plateau is not explained by suffix-vs-prefix topology alone.

Next decisive control:
  picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531

Only intended change:
  ACTION_CONTEXT_TOKENS=0

Purpose:
  isolate whether dense PICF action-context tokens are the remaining action-path
  noise source.  If prefix-only recovers action, dense context should be retired
  or heavily gated from the maintained action path.  If prefix-only is still
  flat, the remaining cause is the step9100 basin / train stream / optimizer
  boundary rather than the dense context channel.
```

E12 launch:

```text
tmux:
  e12_prefixonly_sem035_h30k_20260531

run:
  picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531

confirmed command-line boundary:
  resume checkpoint        = prefixfusion step9100
  num_steps                = 30000
  semantic_trainable_scope = backbone_only
  semantic_lr_scale        = 0.35
  action_context_tokens    = 0
  action_context_integration = prefix_fusion
  unroll/burnin            = 2 / 1
  frozen backbones         = Sonata / V-JEPA / AnyTouch

startup speed:
  first steps ~= 24-26 sec/step

readout:
  step9150 is the first decisive readout.
  step9200 is the confirmation readout if 9150 is ambiguous.
```

E12 result and root-cause refinement:

```text
9150:
  action_default_equiv=0.047398
  non_action=0.009776
  active/downstream=0.140000/0.140714
  object_pull=0.111165
  slot_jepa=0.813201
  policy_lr=5.941e-5
  semantic_lr=2.079e-5
  pi_context_token_count=0

Conclusion:
  Dense action-context tokens are not the action plateau root cause.  Removing
  them gives essentially the same action value as E11.

New confirmed mismatch:
  source E6 step9100 policy_lr ~= 2.006e-5
  source E6 step9100 semantic_lr ~= 7.020e-6
  resumed h30k step9150 policy_lr ~= 5.941e-5
  resumed h30k step9150 semantic_lr ~= 2.079e-5

Issue:
  h30k resume diagnostics were LR-discontinuous with the source checkpoint.
  They restarted an already low-LR checkpoint with roughly 3x action/semantic
  LR, causing action-path drift while structure metrics stayed healthy.
```

E13 repair:

```text
run:
  picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531

only intended repair:
  LR=2.0e-5
  MIN_LR=2.0e-5
  SEMANTIC_LR_SCALE=0.35
  ACTION_CONTEXT_INTEGRATION=prefix_fusion
  ACTION_CONTEXT_TOKENS=24

Expected:
  if LR discontinuity was the root cause, action at 9150 should return toward
  the E6 source band rather than stay in the E10/E11/E12 0.047 band.
```

E13 9150 result:

```text
9150:
  action_default_equiv=0.046477
  non_action=0.010350
  active/downstream=0.200000/0.191951
  object_pull=0.167834
  slot_jepa=0.821991
  policy_lr=2.000e-5
  semantic_lr=7.000e-6

Interpretation:
  LR continuity improves action versus the 0.047-0.048 band, so the LR
  discontinuity is a real cause.  It does not fully recover the source E6
  0.044 band at 9150, so this is not yet a complete fix.

Next read:
  wait to 9200.  If action continues down, keep LR-continuity as the maintained
  resume rule.  If action stays near 0.046+, the next issue is missing optimizer
  state or source/train-window mismatch, not a slot-loss problem.
```

E13 9200 final:

```text
9200:
  action_default_equiv=0.046615
  non_action=0.010613
  active/downstream=0.095000/0.090000
  anchor_pv=0.498752
  object_pull=0.192729
  slot_jepa=0.701867
  recycle=0.099770
  policy_lr=2.000e-5
  semantic_lr=7.000e-6
```

Issue status:

```text
Partially fixed:
  LR discontinuity.  E13 is better than the E10/E11/E12 0.047-0.048 band.

Not fixed by LR continuity alone:
  source E6 0.044 action band was not restored.

New hard facts:
  source /9100 checkpoint has no optimizer.pt;
  E13 /9200 checkpoint has no optimizer.pt;
  trainer supports --optimizer-checkpoint-mode full, but these runs used
  model-only and therefore reinitialized Adam on resume.

Next required test:
  exact-window replay on the source step9100 windows.  Do not use fixed64 as a
  global claim.  The launched probe compares source /9100 and E13 /9200 on the
  same 100 window_trace records:

    /mnt/picf_exact_window_probes/e6_step9100_rank01_windows.jsonl
    /mnt/picf_exact_window_probes/e6_vs_e13_9100_windows_20260531/
```

Maintained rule update:

```text
Any future long run that may be resumed for action-loss comparison must use
--optimizer-checkpoint-mode full at phase boundaries.  model-only checkpoints
remain acceptable for eval/export, but they must not be used as evidence that a
continued optimizer trajectory is healthy or unhealthy.
```

Implementation added:

```text
script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_lrcontinuity_fullopt_prefixfusion_h30k_20260531.sh

contract:
  LR=2e-5
  MIN_LR=2e-5
  SEMANTIC_LR_SCALE=0.35
  ACTION_CONTEXT_INTEGRATION=prefix_fusion
  ACTION_CONTEXT_TOKENS=24
  OPTIMIZER_CHECKPOINT_MODE=full
  SAVE_INTERVAL=1000
  KEEP_LAST_CHECKPOINTS=5

local checks:
  bash -n passed
  git diff --check passed
  targeted checkpoint tests: 7 passed
```

Exact-window source check:

```text
source /9100 evaluated on source E6 step9100 exact windows:
  accepted windows = 100
  action_default_equiv mean = 0.041924
  active/downstream overlap mean = 0.130000 / 0.122288
  slot_jepa mean = 0.883698
  recycle mean = 0.101549

Implication:
  source /9100 is not broken under the current probe.  If E13 /9200 is worse on
  these same windows, the degradation is from resumed training after /9100, not
  from an invalid probe or the old fixed64 comparison.
```

Exact-window E13 result:

```text
same 100 source E6 step9100 windows:
  source /9100 action_default_equiv = 0.041924
  E13 /9200 action_default_equiv    = 0.042149
  delta                             = +0.000225

structure:
  slot_jepa improves from 0.883698 to 0.710429
  active overlap improves from 0.130000 to 0.115000
  downstream overlap improves from 0.122288 to 0.118174
  recycle stays essentially equal, 0.101549 -> 0.100065
```

Root-cause update:

```text
The apparent E13 action gap is not a same-window weight regression.  The E13
checkpoint is effectively action-equivalent to source /9100 on the exact source
windows.  The 0.0466 training-log value at E13 step9200 is therefore dominated
by later-window/difficulty non-stationarity.

Issue class changes from:
  "PICF/action-context architecture is likely degrading action"

to:
  "rolling train-window action loss is not a stationary quality metric; resume
   studies require exact-window or stratified held-window controls."

Still keep:
  full optimizer checkpoints for future resume hygiene.

Do not do:
  another slot-structure rewrite solely because a later rolling window logs a
  higher action mean.
```

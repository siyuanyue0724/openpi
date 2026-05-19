# PICF-AQR-OWM Loss Audit - 2026-05-17

Status: current loss audit for `picf_a7_task_owner_point_bias_anchoronly_diag300_20260517`.

This note tracks the current loss families, whether they are active objective
terms or diagnostics, the mathematical role of each term, and the observed
failure modes.  It is linked from `src/openpi/picf/README_v2.2.md`.

## Current Run Contract

Run:

```text
picf_a7_task_owner_point_bias_anchoronly_diag300_20260517
```

Important launch settings:

```text
trainable_scope = anchor_only
unroll_steps = 2
burnin_steps = 1
lambda_action_pos = 0
lambda_action_rot = 0
lambda_action_gripper = 0
lambda_anchor_pv = 0.25
lambda_pv_weak = 0.05
lambda_pt = 1.0
lambda_mapg_cycle = 0.05
lambda_mapg_support_diversity = 0.25
lambda_mapg_geometry_diversity = 0.05
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
aux_budget_alignment_ratio = 1.0
aux_budget_alignment_floor = 2.0
```

Therefore the current run is an anchor/measurement diagnostic.  It is not an
action co-training run and it does not train the guarded OWM prediction losses.

## Step Trend Snapshot

```text
metric                                      step50      step100     step150
loss_total                                  0.4990      0.5235      0.6019
loss_alignment                              0.4240      0.4485      0.5269
loss_action                                 0.0000      0.0000      0.0000
loss_action_default_equiv                   0.1809      0.1855      0.1869
loss_anchor_pv                              0.7182      1.0567      1.1711
loss_pv_weak                                6.0704      4.2056      3.5800
loss_mapg_graph                             0.1231      0.1002      0.1626
loss_mapg_cycle                             0.3169      0.3575      0.3817
loss_mapg_support_diversity                 0.3667      0.2879      0.5140
loss_mapg_geometry_diversity                0.3111      0.2080      0.2995
loss_slot_jepa                              1.0866      1.3746      1.6703
loss_support_pred                           0.1584      0.1773      0.1866
loss_binding_consistency                    0.4599      0.4994      0.5087
loss_aqr_denoising                          1.3421      2.0003      2.0217
aqr_same_role_support_overlap_max           0.1695      0.6881      0.9979
active_duplicate_overlap_max                0.0000      0.0000      0.0000
posterior_identity_switch_rate              0.6617      0.6306      0.6067
posterior_recycle_rate                      0.0807      0.0069      0.0007
aqr_task_owner_anchor_score_max             0.1810      0.2743      0.5354
```

Main observation:

```text
Dense point/visual alignment improves:
  loss_pv_weak: 6.07 -> 3.58

Object-routed anchor alignment worsens:
  loss_anchor_pv: 0.72 -> 1.17

Same-role support separation collapses again:
  aqr_same_role_support_overlap_max: 0.17 -> 0.69 -> 0.998
```

The current bottleneck is therefore not dense feature learning.  It is
object-file ownership and same-role support separation under the active
anchor/PV graph objective.

## Loss Family Audit

### 1. Action Loss

Definition:

```text
L_action =
  lambda_pos * L1(action_pos, target_pos)
  + lambda_rot * L1(action_rot, target_rot)
  + lambda_gripper * L1(action_gripper, target_gripper)
```

Current status:

```text
Active objective: no
Current weight: 0
Diagnostic keys:
  loss_action_default_equiv
  loss_action_active7
```

Interpretation:

```text
loss_action_default_equiv ~= 0.181 -> 0.187 is telemetry only.
It is useful for comparability with old 4-22 style runs, but it does not drive
the current diagnostic.
```

Current issue:

```text
No direct action-loss problem in this run.
Do not attribute same-role support collapse to action pressure here.
```

### 2. Anchor PV Alignment

Implementation:

```text
src/openpi/picf/core/training.py:
  compute_alignment_loss(...)
```

Conceptual form:

```text
L_anchor_pv =
  BCE(routing(point -> visual), projective_compatibility)
  weighted by projective candidate edges
  optionally gated by object-routed point/visual support
```

Current status:

```text
Active objective: yes
Current run weight: lambda_anchor_pv = 0.25
Trend: 0.718 -> 1.057 -> 1.171
```

Interpretation:

```text
This is currently the main red flag.
Dense visual/point descriptors are improving, but the anchor-routed assignment
is becoming less consistent with projective correspondence.
```

Likely cause:

```text
The object gate makes this loss less global than the old dense PV loss, but it
still depends on the current graph's point/visual priors.  When multiple
same-role anchors move toward the same high-evidence task/proposal region, the
same dense correspondence can be explained redundantly by multiple anchors.

This can make loss_pv_weak improve while loss_anchor_pv worsens.
```

Current issue status:

```text
Open.  This is the highest-priority active loss issue.
```

### 3. PV Weak Alignment

Conceptual form:

```text
For each visual token u:
  build a point bag from projective-compatible points
  classify visual token u against all visual embeddings
```

Current status:

```text
Active objective: yes
Current run weight: lambda_pv_weak = 0.05
Trend: 6.070 -> 4.206 -> 3.580
```

Interpretation:

```text
This is healthy, but it is dense correspondence learning, not object-slot
health.  It proves the point/visual feature space is learning.  It does not
prove anchors bind to separate task objects.
```

Current issue status:

```text
No immediate issue.  Use it as a dense feature-health signal only.
```

### 4. Point/Tactile Alignment

Conceptual form:

```text
L_pt aligns contact/tactile evidence with nearby point evidence.
```

Current status:

```text
Active objective: yes by configuration, but current metric loss_pt = 0.
Tactile telemetry exists, but the current sampled windows do not produce a
nonzero point/tactile alignment term.
```

Interpretation:

```text
The current same-role support collapse is not being corrected by tactile
alignment in these windows.  Tactile is not the active source of the observed
loss regression.
```

Current issue status:

```text
Monitor.  Not the current primary failure.
```

### 5. MAPG/AQR Graph Loss

Active weighted graph terms:

```text
L_graph =
  0.05 * L_cycle
  + 0.25 * L_support_diversity
  + 0.05 * L_geometry_diversity
```

Disabled graph telemetry:

```text
lambda_mapg_siglip = 0
lambda_mapg_vicreg = 0
lambda_mapg_masked_modality = 0
lambda_mapg_routing = 0
```

Current trends:

```text
loss_mapg_graph:              0.123 -> 0.100 -> 0.163
loss_mapg_cycle:              0.317 -> 0.357 -> 0.382
loss_mapg_support_diversity:  0.367 -> 0.288 -> 0.514
loss_mapg_geometry_diversity: 0.311 -> 0.208 -> 0.300
```

Interpretation:

```text
The graph loss initially improves at step100, then regresses at step150.
The support diversity term tracks the same-role overlap regression: it is not
silent, but its pressure is not sufficient to prevent reconvergence.
```

Current issue status:

```text
Open.  The support-diversity objective detects the failure but does not fully
control it under the current anchor/PV/proposal measurement pressure.
```

### 6. Same-Role Support Overlap Metrics

Relevant metrics:

```text
aqr_same_role_support_overlap_max
aqr_active_same_role_support_overlap_max
posterior_file_competition_active_duplicate_overlap_max
```

Current status:

```text
raw same-role overlap:
  0.169 -> 0.688 -> 0.998

active duplicate overlap:
  0 -> 0 -> 0
```

Interpretation:

```text
The active file competition fix still prevents active posterior file duplicates.
That part is working.

The raw graph support still reconverges across same-role anchors.  This means
the AQR graph can reuse the same evidence region even though the posterior
active-file filter demotes duplicated object files.
```

Current issue status:

```text
Partially fixed.
Closed for active posterior duplicate files.
Open for raw same-role graph support reuse.
```

### 7. Slot-JEPA

Implementation:

```text
_matched_prediction_loss(...)
```

Current status:

```text
Active objective: no
Current weight: lambda_slot_jepa = 0
Raw trend: 1.087 -> 1.375 -> 1.670
```

Interpretation:

```text
The old index-aligned target has been replaced by soft matching, but this loss
is still guarded.  The raw trend is getting worse, so it should not be enabled
until object identity and support overlap are stable.
```

Current issue status:

```text
Guarded hook only.  Do not enable in current state.
```

### 8. Support Prediction

Conceptual form:

```text
Predict detached future posterior support summary:
  [alpha, support_mass, contact_prob, binding_confidence]
```

Current status:

```text
Active objective: no
Current weight: lambda_support_pred = 0
Raw trend: 0.158 -> 0.177 -> 0.187
```

Interpretation:

```text
This is not currently training.  The raw trend is mildly worse, which is
expected if posterior ownership is unstable.
```

Current issue status:

```text
Guarded hook only.  Do not enable until support identity is stable.
```

### 9. Binding Consistency

Current form:

```text
L_binding =
  mean normalized entropy of current posterior binding
  + soft matched current/future posterior token consistency
```

Current status:

```text
Active objective: no
Current weight: lambda_binding_consistency = 0
Raw trend: 0.460 -> 0.499 -> 0.509
```

Interpretation:

```text
The term is now more permutation-tolerant than the old index-only form, but it
still depends on posterior identity being meaningful.  Raw worsening is another
signal that the current posterior ownership is not stable enough for predictive
auxiliary pressure.
```

Current issue status:

```text
Guarded hook only.  Keep disabled.
```

### 10. AQR Denoising

Conceptual role:

```text
Training-only support denoising auxiliary.  It should help anchor queries
recover high-confidence support, but only after the support target is reliable.
```

Current status:

```text
Active objective: no
Current weight: lambda_aqr_denoising = 0
Raw trend: 1.342 -> 2.000 -> 2.022
```

Interpretation:

```text
This raw rise is useful diagnostics: current anchor recovery gets harder as
the graph support reconverges.  Because the weight is zero, it is not causing
the regression.
```

Current issue status:

```text
Guarded hook only.  Do not enable until task-owner/object evidence is stable.
```

### 11. Physical Future Aux

Current components:

```text
visual_latent, visual_real, tactile_real, point_real
```

Current status:

```text
Active objective: capped
Step150:
  loss_physical_aux ~= 1.60
  loss_physical_aux_capped = 0.05
```

Interpretation:

```text
The raw physical prediction terms are not allowed to dominate the current
diagnostic.  The cap is doing its job.
```

Current issue status:

```text
No active blocker for this anchor-only diagnostic.
```

### 12. Semantic Future Aux

Current status:

```text
Active objective: capped
Step150:
  loss_semantic_group_raw ~= 0.30
  loss_semantic_group_capped = 0.025
```

Interpretation:

```text
Semantic auxiliary pressure is bounded and is not the immediate cause of
support overlap regression.
```

Current issue status:

```text
No active blocker for this anchor-only diagnostic.
```

### 13. Tactile Aux Telemetry

Step150:

```text
loss_tactile_aux ~= 13.38
loss_tactile_real ~= 4.70
tactile_evidence_rate = 0.5
```

Interpretation:

```text
These raw values look large, but they are inside the capped physical auxiliary
group and do not dominate the current objective.  They still matter for future
full co-training, but they are not the driver of the current graph support
collapse.
```

Current issue status:

```text
Monitor before full co-training; not the current primary issue.
```

## Current Loss-Level Diagnosis

The current failure can be stated precisely:

```text
The dense point/visual descriptor space improves, but the object-routed graph
does not maintain separated same-role ownership.  The result is that multiple
same-role graph anchors can reuse the same high-evidence object/proposal region,
while active posterior-file competition prevents only the final active-file
duplicates.
```

Mathematically:

```text
loss_pv_weak decreases:
  good dense projective representation

loss_anchor_pv increases:
  graph-routed object assignment no longer matches projective evidence

loss_mapg_support_diversity increases:
  support separation term detects overlap

aqr_same_role_support_overlap_max -> 0.998:
  raw graph support has reconverged

active_duplicate_overlap = 0:
  final active posterior file filtering is still working
```

This separates two layers:

```text
Layer A: final posterior active-file duplicate filter
  Status: working.

Layer B: upstream AQR graph same-role support assignment
  Status: still open.
```

## Practical Acceptance Rules

Do not call a run healthy only because `loss_pv_weak` falls.  The required
acceptance conditions are:

```text
loss_anchor_pv should not trend upward after warmup.
loss_mapg_support_diversity should not rebound sharply after step100.
aqr_same_role_support_overlap_max should stay controlled, not return to ~1.0.
posterior_file_competition_active_duplicate_overlap_max should remain 0.
task-owner proposal/point metrics should remain nonzero.
active-only overlay should show an active object file on the task object.
```

## Current Open Loss Problems

High priority:

```text
1. loss_anchor_pv worsens despite loss_pv_weak improving.
2. raw same-role support overlap reconverges by step150.
3. support-diversity loss detects the overlap but does not prevent it.
```

Medium priority:

```text
4. raw slot_jepa/support_pred/binding_consistency/aqr_denoising worsen, so
   guarded OWM losses must remain disabled.
5. posterior identity switch remains high, although recycle has been suppressed.
```

Not current blockers:

```text
6. action loss is inactive in this run.
7. active posterior duplicate overlap is controlled.
8. dense PV representation is improving.
9. physical/semantic future auxiliaries are capped.
```

## Recommended Next Fix Direction

Do not add another unrelated loss.  The loss audit points to a measurement
ownership problem:

```text
task/object evidence should create one owner and bounded context/reserve files,
not let every same-role graph anchor reuse the same support.
```

The next fix should therefore target the graph ownership layer:

```text
1. Strengthen same-role graph support competition on object-owner candidates,
   not on context/reserve anchors.
2. Keep task-owner proposal-to-point evidence as measurement likelihood, but
   prevent it from being cloned across multiple same-role graph rows.
3. Preserve active posterior-file competition, since that part is already
   working and should not be removed.
4. Keep slot-JEPA/support-pred/binding-consistency/denoising disabled until
   anchor_pv and same-role overlap stabilize.
```

This is a coherent fix because it acts on the layer that failed: AQR graph
measurement ownership.  It is not a post-hoc action-loss patch.


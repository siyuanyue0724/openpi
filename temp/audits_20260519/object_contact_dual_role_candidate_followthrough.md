# Object/Contact Dual-Role Candidate Follow-Through

Date: 2026-05-19

## Question

The sidecar mask can identify the task/contact object, but A7 overlays still
showed role leakage: blue effector rows could occupy the object while orange
task-object rows did not reliably bind to the mask.  Removing the effector role
would be wrong because tactile/proprio context must still enter PICF.  The
required fix is to separate ownership from contact.

## Maintained State Model

For physical row `j`:

```math
S_j = (role_j, q_j, x_j, P_j, T_j, a_j)
```

where `role=0` is effector context, `role=1` is object/task-file, `role=2` is
contact/interaction bridge, and `role=3` is context/coverage.

Sidecar candidate `p` supplies a soft object/contact mask:

```math
M_{p,n} = p(point_n \mid candidate_p)
```

It is a measurement, not a label.  The posterior remains authoritative.

## Paper-Code Invariant Translation

SlotAttention / SlotContrast use slot-wise competition and mask explanation:

```text
tokens compete into slots
slots explain dense support with a soft background/no-object residual
```

PICF cannot copy the RGB decoder objective.  The correct translation is:

```text
candidate mask p competes for eligible physical rows
plus explicit background residual
plus posterior correction after measurement routing
```

Reference-query methods such as Deformable DETR / DINO support initializing or
biasing queries from candidate evidence, but still keep query competition.  PICF
therefore uses proposal/point/visual candidate evidence as bounded measurement
priors, not as hard coordinate teleportation.

## Repair

Eligible candidate rows:

```math
R_c = \{j : query\_type_j = physical,\ role_j \in \{1,2\}\}
```

Role 0 is excluded:

```math
role_j = 0 \Rightarrow j \notin R_c
```

Candidate score:

```math
E_{j,p}
= w_P \hat P_{j,p}
 w_Q \widehat{(Q_j M_p)}
 w_S \hat S_{j,p}
 w_T \hat T_p
```

with the row-specific guard:

```math
E_{j,p}=-\infty
```

unless at least one row-specific source connects row `j` to candidate `p`.

Capacity:

```math
K_p = TopK_j(E_{j,p}, k=2)
```

The two allowed explanations are object owner and contact bridge.  Extra raw
clones are still rejected, and background absorbs invalid candidates:

```math
A_{j,p}
= \frac{\exp(E_{j,p}/\tau)}
        {\sum_{k \in K_p}\exp(E_{k,p}/\tau)+\exp(\log b-\beta q_p)}
```

## Code Follow-Through

Implemented in:

```text
src/openpi/picf/core/config.py
  object_candidate_eligible_roles=(1,2)
  object_candidate_max_rows_per_candidate=2

src/openpi/picf/core/pipeline.py
  _object_candidate_physical_rows(...)
  _task_owner_visual_bias(...)
  _task_owner_proposal_bias(...)
  _task_owner_proposal_to_point_priors(...)
  _proposal_anchor_seed_transport(...)
  _proposal_object_candidate_assignment(...)
  _task_owner_proposal_point_bias(...)
  _task_owner_anchor_score(...)

scripts/picf_core_train.py
  --object-candidate-eligible-roles
  --object-candidate-max-rows-per-candidate

scripts/picf_object_candidate_slot_binding_audit.py
  dual-role candidate contract checks

scripts/picf_anchor_overlay_make_gifs.py
  combined_6view.gif diagnostic output
```

## Acceptance

The next 1000-step anchor-only test must show:

```text
role-1/role-2 rows receive object-candidate assignment
role-0 blue anchors do not become the object owner
mask_active and mask_with_gray overlays align with the object/contact mask
loss_anchor_object_pull decreases without needing action/semantic losses
```

If the probe still fails, the remaining hypothesis is not loss interaction.  It
would mean the current query/update parametrization cannot move physical rows
onto inspected candidate masks even under isolated object-pull supervision.

## First Runtime Evidence

Run:

```text
picf_a7_dualrole_object_contact_pull_1000_20260519
```

At step 50:

```text
loss_anchor_object_pull               0.5643
active_same_role_support_overlap      0.0212
active_same_role_object_core_overlap  0.0751
raw_same_role_support_overlap         0.2322
object_candidate_assigned_row_count   2.4950
object_candidate_assignment_max       0.5335
object_candidate_coverage_mean        0.9518
object_candidate_background_mean      0.0248
active_file_duplicate_overlap         0.0000
```

At step 100:

```text
loss_anchor_object_pull               0.3054
active_same_role_support_overlap      0.0588
active_same_role_object_core_overlap  0.1513
raw_same_role_support_overlap         0.7229
raw_same_role_object_core_overlap     0.5668
object_candidate_assigned_row_count   2.7150
object_candidate_assignment_max       0.5215
object_candidate_coverage_mean        0.9587
object_candidate_background_mean      0.0038
active_file_duplicate_overlap         0.0000
grad_norm                             5.0000
grad_clip_applied                     true
```

Interpretation:

```text
The dataflow is live.  The object candidate is assigned to more than one row,
which is the intended object-owner/contact-bridge behavior.  Candidate coverage
is high and background is near zero, so the sidecar evidence is not being
ignored.

The object-pull loss falls sharply, and active same-role overlap remains low.
This is a positive signal for the repaired assignment path.

The raw same-role overlap rises by step 100.  This is not automatically the old
collapse because raw overlap includes reserve/context rows and the intentional
role-1/role-2 sharing of a candidate.  It is still a warning requiring step-200
visual inspection.
```

Step-200 result:

```text
loss_anchor_object_pull               0.4752
active_same_role_support_overlap      0.1320
active_same_role_object_core_overlap  0.0126
raw_same_role_support_overlap         0.9883
raw_same_role_object_core_overlap     0.9239
object_candidate_assigned_row_count   2.4900
object_candidate_assignment_max       0.5350
object_candidate_coverage_mean        0.9821
object_candidate_background_mean      0.0004
posterior_recycle_rate                0.0009
active_file_duplicate_overlap         0.0000
```

Conclusion:

```text
The dataflow/assignment half of the repair is correct, but the geometric owner
selection half is still incomplete.  Evidence is not missing: candidate
coverage is high and background mass is near zero.  The failure mode is that a
confirmed mask can be represented by the role-2 contact bridge or reserve rows
without forcing the role-1 object owner to become the stable spatial owner.

Therefore the next repair should not be another source of proposals.  It should
be a bounded owner-transport update from candidate mask/point barycenter into
the role-1 object owner geometry, with role-2 retained as a contact bridge and
role-0 retained as effector context.
```

## Owner Geometry Transport Repair

The step-200 failure is not candidate absence:

```text
object_candidate_coverage_mean      > 0.95
object_candidate_background_mean    near 0
active_file_duplicate_overlap       0
```

The failure is ownership: the model can explain a mask with role-2 contact
bridge rows or reserve/context rows while the role-1 object file does not become
the stable spatial owner.  The repair therefore adds an explicit but bounded
owner-transport leg.

For each covered candidate `p`, let:

```math
O = \{j: role_j = 1,\ query\_type_j = physical\}
```

Use the pre-top-k row/candidate evidence `E_{j,p}` to select the best role-1
owner:

```math
o(p)=\arg\max_{j\in O} E_{j,p}
```

Then assign bounded owner mass:

```math
T_{o(p),p}
=
\max(A_{o(p),p},\rho \cdot coverage_p)
```

where `A` is the ordinary object/contact assignment and `rho =
object_candidate_owner_min_share`.  The transported owner point prior is:

```math
P^{owner}_{j,x}
=
normalize_x
\sum_p T_{j,p} M_{p,x}
```

where `M_{p,x}` maps candidate masks to point evidence.  Runtime point priors
are then mixed only for owner rows:

```math
P_j
\leftarrow
normalize((1-\lambda)P_j+\lambda P^{owner}_j)
```

This is not a new proposal source and not a hard label.  It is a missing
geometry ownership transport edge inside the existing belief update:

```text
candidate mask -> role-1 object owner point prior -> anchor geometry / pull loss
```

The contact bridge remains role 2; the gripper/effector remains role 0 and is
excluded from object ownership.

Implemented in:

```text
src/openpi/picf/core/config.py
  object_candidate_owner_transport_enabled
  object_candidate_owner_roles=(1,)
  object_candidate_owner_min_share
  object_candidate_owner_point_mix

src/openpi/picf/core/contracts.py
  PicfAnchorPriorGraphState.object_candidate_owner_assignment
  PicfAnchorPriorGraphState.object_candidate_owner_point_priors

src/openpi/picf/core/pipeline.py
  _proposal_object_candidate_assignment(...)
  graph debug:
    aqr_object_candidate_owner_row_count
    aqr_object_candidate_owner_candidate_count
    aqr_object_candidate_owner_assignment_max
    aqr_object_candidate_owner_point_row_count
    aqr_object_candidate_owner_point_max

src/openpi/picf/core/training.py
  object-pull target prioritizes object_candidate_owner_point_priors
  confirmed object rows include owner assignment / owner point priors

scripts/picf_core_train.py
  --object-candidate-owner-transport-enabled
  --object-candidate-owner-roles
  --object-candidate-owner-min-share
  --object-candidate-owner-point-mix
  OWM_DEBUG_METRIC_KEYS includes:
    aqr_object_candidate_owner_row_count
    aqr_object_candidate_owner_candidate_count
    aqr_object_candidate_owner_assignment_max
    aqr_object_candidate_owner_point_row_count
    aqr_object_candidate_owner_point_max
```

Acceptance for the next probe:

```text
aqr_object_candidate_owner_row_count >= 1 on sidecar frames
aqr_object_candidate_owner_point_row_count >= 1 on sidecar frames
loss_anchor_object_pull should decrease without relying on action/semantic loss
role-1/orange owner should visibly move onto the mask in mask_active overlays
role-2/contact may remain near gripper/contact, but must not be sole object owner
role-0/blue must not own object candidates
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

## 2026-05-19 Object-Owner-Only Probe

Motivation:

```text
The v2 owner-transport probe still allowed role-0/blue effector rows to exist
because the structured graph layout always created an effector row and
`aqr_active_slot_min_per_role=1` could keep it active.  This contaminates the
question "can the object owner be pulled to the sidecar mask?" because a blue
effector/context row can visually occupy the manipulated object even when the
role-1 owner row is the intended object file.

The tactile route also used the old assumption that tactile belongs to the
effector role.  For object binding this is the wrong direction: tactile/contact
is evidence about the contacted object and should be attached to the selected
object owner at assignment time.  The gripper/wrist remains an evidence source,
not a persistent object-owner file.
```

Implemented invariant:

```text
aqr_role_layout=object_only
  all physical graph rows have role=1.

effector_persistent_anchors=0
effector_observation_anchors=0
task_effector_queries=0
  no blue/role-0 graph owner is created in the clean probe.
  The training CLI validator accepts these zero counts only for no-effector
  layouts; the structured default still requires positive effector counts.

tactile_attach_to_object_owner=True
  tactile tokens are readable by role-1 object owner rows and blocked from
  non-object rows.  The legacy behavior is still available through
  --no-tactile-attach-to-object-owner for compatibility.
  This flag must affect all three tactile paths:
    public/fused read bias,
    AQR graph tactile reader bias,
    MAPG tactile seed priors.

anchor_object_pull_allowed_roles=(1,)
object_candidate_eligible_roles=(1,)
object_candidate_owner_roles=(1,)
  sidecar candidates and object-pull loss test only role-1 ownership.
```

Mathematical interpretation:

```text
Let M_p be a sidecar object/contact mask and let q_j be candidate physical
object slots.  This probe removes effector rows from the candidate set:

  J_owner = {j | role_j = 1}

Candidate assignment, owner transport, and object-pull loss are all restricted
to J_owner.  Contact/tactile evidence T is treated as measurement evidence for
the contacted object:

  score_tactile(j, T) is valid only if j in J_owner

This tests the model capacity to learn:

  M_p -> role-1 owner -> posterior object file

without blue/effector competition.  If this fails, the fault is not role
competition; it is anchor geometry/query capacity, sidecar projection, or the
object-pull objective itself.
```

Step-50 follow-through narrowed the failure mode further:

```text
graph anchors:
  role=1 only and located at the transported sidecar/mask center.

posterior active files:
  still offset from the sidecar/mask center in the overlay JSON.

loss_anchor_object_pull before the closure fix:
  supervised only anchor_prior_graph.anchor_x.
  It could therefore pass while posterior.x remained wrong.
```

This is a graph-to-belief diagnostic gap.  The corrected owner transport is:

```math
M_{obj}
  \rightarrow p_g(j,p)
  \rightarrow p_o(i,p) = A_{obs,graph} p_g
  \rightarrow p_b(k,p) = B_{post,obs} p_o
```

The object-pull objective now supervises both graph measurement geometry and
posterior belief-file geometry:

```math
L_{obj}
=
\lambda_g \sum_j w_j \rho(x^g_j, \bar x_j)
+
\lambda_b \sum_k a_k \rho(x^b_k, \bar x_k)
```

`a_k` is detached posterior file-competition activity.  This is deliberate:
active posterior files must move to the object evidence; they cannot reduce the
loss by lowering objectness/activity gates.

The clean object-only probe also sets:

```text
posterior_file_competition_max_per_role=1
```

One sidecar object owner should be explained by one active posterior file.
Reserve files may exist, but they should not appear as multiple active orange
owners for the same object in this probe.

Script:

```text
run_a7_object_owner_only_pull_probe_1000_20260519.sh
```

Acceptance:

```text
No role-0/blue active owner rows in overlay JSON.
loss_anchor_object_pull decreases by step 50/100.
aqr_object_candidate_owner_row_count > 0 on sidecar frames.
role-1/orange active owner is inside or very near the sidecar mask/box.
active same-role overlap remains low; raw inactive overlap is secondary.
```

Audit:

```text
scripts/picf_object_candidate_slot_binding_audit.py now checks:
  config exposes tactile_attach_to_object_owner and aqr_role_layout;
  pipeline has object_only role layout;
  tactile can attach to role-1 object owner;
  anchor_object_pull includes posterior belief-file geometry;
  training CLI exposes both switches;
  the object-owner-only probe disables role-0 effector competition.
```

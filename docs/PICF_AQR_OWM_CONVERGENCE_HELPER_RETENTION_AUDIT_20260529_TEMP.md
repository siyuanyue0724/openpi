# PICF-AQR-OWM Convergence Helper Retention Audit - 2026-05-29

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_REBOUND_DEEP_AUDIT_20260528_TEMP.md
docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md
```

## 1. Question

If the fixed FSDP optimizer-group run passes the old action-rebound window, do
we still keep the complex structures added to help anchor/slot convergence?

Short answer:

```text
Keep the belief-state invariants.
Decay or guard the training scaffolds.
Do not revive archived blind-SAM/local-refinement/predictive-loss branches.
Do not delete working runtime gates before a full 30k/CALVIN acceptance run.
```

The reason is mathematical: the current improvement is not evidence that all
previous convergence structures were unnecessary.  The current run is healthy
under the current contract, which includes object-owner routing, active/context
slot gating, posterior competition, sidecar/tracklet typed evidence, action
prefix stabilization, and the fixed two-timescale optimizer split.

Removing several of these at once would create a new architecture, not a cleanup.

## 2. Latest Runtime Evidence

Live run:

```text
picf_a7_stepindexed_fixedwindow_from7000_30k_20260529
```

The run resumes the preserved EMA step7000 model-only checkpoint with:

```text
NUM_TRAIN_STEPS=30000
ACTION_LOSS_WEIGHT=2.0
ACTION_PREFIX_TEACHER_MODE=ema
ACTION_PREFIX_NORM_MODE=rmsnorm
ACTION_PREFIX_OUTPUT_GATE=0.70
LAMBDA_ACTION_PREFIX_TRUST=0.02
PICF_CORE_LR_SCALE=0.005
SEMANTIC_TRAINABLE_SCOPE=backbone_only
Sonata/V-JEPA/AnyTouch frozen
PaliGemma semantic backbone trainable
contact-motion sidecar/tracklet/proposal typed evidence enabled
blind SAM rejected
step_indexed_window_rng=True
```

The corrected optimizer-group hard gate is real, but the old sampled-window
`step7000≈0.021` row is no longer treated as a stationary checkpoint value.
The fixed-window no-update probe showed:

```text
old7000 fixed-window action mean ~= 0.06968
old8000 fixed-window action mean ~= 0.06797
```

Current step-indexed rows through step7200:

```text
step7050:
  loss_action_default_equiv          0.04884
  loss_total_minus_action            0.01075
  active/downstream support overlap  0.05500 / 0.06265

step7100:
  loss_action_default_equiv          0.05097
  loss_total_minus_action            0.01051
  active/downstream support overlap  0.06999 / 0.07580

step7150:
  loss_action_default_equiv          0.04267
  loss_total_minus_action            0.01063
  active/downstream support overlap  0.09500 / 0.09571

step7200:
  loss_action_default_equiv          0.04319
  loss_total_minus_action            0.01051
  active/downstream support overlap  0.09500 / 0.10729
  raw same-role overlap              1.00000
  lr_group_picf_core                 3.162662e-7
  lr_group_policy_head               6.325324e-5
```

Interpretation:

```text
The current run is structurally valid and has not triggered a helper-removal
gate.  It has not yet proven final action acceptance.  The next decisive band
is the old 7350-7600 region under the step-indexed stream, followed by
fixed-window probe and CALVIN/video evidence.
```

## 3. Classification Principle

Each helper is classified by where it enters the computation:

```text
Belief-state invariant:
  changes routing, assignment, posterior correction, or action-visible prefix
  in a way required for a well-defined object-addressable belief filter.

Training scaffold:
  adds an auxiliary objective to shape early geometry/object evidence but should
  be weak, budgeted, or decayed once action training is stable.

Diagnostic/legacy:
  useful for ablation or monitoring but not a production path.
```

Mathematically, keep a helper if it reduces a structural identifiability defect
without adding a new label assumption:

```math
Z_t \rightarrow B_t \rightarrow A_t
```

where `Z_t` is typed evidence, `B_t` is object-addressable belief, and `A_t` is
the action.  A helper is justified when it improves the map `Z_t -> B_t` or
stabilizes the interface `B_t -> A_t` without replacing posterior inference
with a hard side label.

## 4. Keep As Production Invariants

### 4.1 Active / Context / Reserve Slot Partition

Code:

```text
src/openpi/picf/core/config.py:
  aqr_active_slot_filter_enabled=True
  aqr_context_slot_enabled=True
  aqr_context_slot_deduplicate_enabled=True
  aqr_control_graph_attention_bias_enabled=True
  aqr_control_graph_token_scaling_enabled=False
```

Keep.

Reason:

```text
Fixed-capacity slots are necessary for compatibility, but not every slot should
be an object owner at every step.  Active rows own current task/object evidence;
context rows carry lower-priority scene evidence; reserve rows remain available
capacity.
```

This is not information pruning.  The maintained path uses attention bias and
state embeddings, not token scaling, so dense V-JEPA/point/tactile memory stays
available.

### 4.2 Same-Role Support Competition

Code:

```text
aqr_same_role_support_competition_enabled=True
```

Keep.

Reason:

```math
E_{jn}^{k+1}
=
\operatorname{Normalize}_n
\left(
  \frac{E_{jn}^k}{\sum_{\ell \in same\_role(j)} E_{\ell n}^k + \epsilon}
\right)
```

This is the role-local responsibility step that prevents multiple same-role
object files from reading the same support when weak row-specific evidence
already exists.  It is not enough by itself, as the A5 anchor-only record
proved, but it is a correct measurement-routing invariant.

### 4.3 Slot Quality Gate

Code:

```text
aqr_slot_quality_enabled=True
aqr_slot_quality_learned_enabled=True
lambda_slot_quality=0 by default
```

Keep runtime gate; keep the optional loss guarded.

Reason:

```text
The learned quality head is zero-initialized around deterministic
sidecar/tracklet/contact evidence.  It determines active/context/reserve
reliability, but the supervised BCE loss is not production-required and remains
off unless explicitly tested.
```

### 4.4 Object Candidate Assignment / Owner Transport

Code:

```text
object_candidate_assignment_enabled=True
object_candidate_owner_transport_enabled=True
object_candidate_owner_geometry_mix=0.90
posterior_owner_transport_enabled=True
posterior_owner_transport_direct_candidate_assignment=True
```

Keep.

Reason:

```text
This is the missing object-file write-back leg.  Sidecar/tracklet/contact
evidence becomes a soft measurement candidate, competes for an eligible object
row, then writes bounded geometry into posterior correction.  Without this, a
sidecar can improve a transient support map without binding a persistent file.
```

This is not a hard mask label and not SAM revival.

### 4.5 Posterior File / Birth Competition

Code:

```text
posterior_file_competition_enabled=True
posterior_birth_competition_enabled=True
posterior_owner_active_gate_enabled=True
```

Keep.

Reason:

```text
The posterior is a persistent object file system.  Duplicate same-role files
that explain the same measurement support should be demoted to no-object/reserve
instead of all updating from the same residual.  Birth competition then controls
which reserve file may consume unexplained evidence.
```

This directly addresses the old duplicate-posterior and inactive-overlap
failures.

### 4.6 Tactile Attach-To-Object Owner

Code:

```text
tactile_attach_to_object_owner=True
tactile_anchor_prob_on=0.55
tactile_evidence_prob_floor=0.35
```

Keep.

Reason:

```text
Contact is evidence about the contacted object, not a separate blue gripper
object owner.  Tactile tokens should attach to the object owner under calibrated
contact gates; otherwise the gripper/effector row can steal the manipulated
object support.
```

### 4.7 Action Prefix RMS / Gate / EMA Teacher

Code:

```text
action_prefix_norm_mode=rmsnorm
action_prefix_output_gate=0.70
action_prefix_teacher_mode=ema
lambda_action_prefix_trust=0.02
```

Keep through long validation.

Reason:

```math
A_t = \pi_\theta(x_t, g(B_t))
```

where `g(B_t)` is the PICF prefix.  RMS/gating bounds prefix scale; EMA teacher
stabilizes the conditioning distribution seen by PI0.5 while online PICF
continues to train.  This is an interface contract, not an extra object label.

### 4.8 Two-Timescale Optimizer Split

Code:

```text
picf_core_lr_scale=0.005
semantic_trainable_scope=backbone_only
```

Keep.

Reason:

```math
\eta_{PICF} \ll \eta_{policy}
```

The old action rebound is now strongly linked to an optimizer grouping bug and
moving-prefix instability.  The corrected FSDP group shows the intended slow
PICF update is real.

## 5. Keep But Treat As Weak / Decayed Training Scaffolds

### 5.1 Object Scaffold Losses

Fields:

```text
lambda_anchor_object_pull
lambda_object_explanation_point
lambda_object_explanation_contact
lambda_object_explanation_duplicate
lambda_object_explanation_background
lambda_mapg_support_diversity
```

Current contract:

```text
OBJECT_SCAFFOLD_DECAY_MODE=cosine
OBJECT_SCAFFOLD_DECAY_END_STEP=1500
OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
```

Keep the decayed floor for this run; do not strengthen unless a later gate
proves object evidence is drifting.

Reason:

```text
These terms help early object-owner geometry and duplicate suppression, but
after the action basin is reached they must not dominate PI0.5 flow training.
At step7200 in the corrected step-indexed run, loss_total_minus_action is about
0.0105, which is bounded and not a structural-loss takeover.
```

### 5.2 MAPG Cycle / Support Diversity

Keep weak.

Reason:

```text
`lambda_mapg_cycle` enforces bidirectional point/visual consistency and remains
mathematically clean.  `lambda_mapg_support_diversity` is weaker: it helped
early but failed as a standalone cure in anchor-only tests.  It should stay
small and decay with the scaffold, not become a main objective.
```

## 6. Keep Off / Legacy Only

### 6.1 Slot-JEPA / Support Prediction / Binding Consistency / Denoising

Code defaults:

```text
lambda_slot_jepa=0
lambda_support_pred=0
lambda_binding_consistency=0
lambda_aqr_denoising=0
```

Keep off.

Reason:

```text
They are useful diagnostics and future hooks, but previous runs showed slot-JEPA
can spike or explode.  They should not be enabled until a separate normalized
predictive-target audit passes.
```

### 6.2 VCAP

Code:

```text
vcap_enabled=False
vcap_action_grad_scale=0
```

Keep off.

Reason:

```text
VCAP changes the active query allocator.  Current fixed-capacity + quality gate
is already passing the first old rebound gate.  Enabling VCAP now would confound
the causal test.
```

### 6.3 Local Refinement

Code:

```text
legacy_local_refinement_opt_in=False
local_refinement_enabled=False
```

Keep archived.

Reason:

```text
Earlier diagnostics found it adds recycle/gradient pressure and is not the root
repair.  Do not re-enable unless a dedicated ablation is requested.
```

### 6.4 Blind SAM

Keep rejected.

Reason:

```text
Blind automatic SAM produced noisy wall/robot/drawer fragments and is rejected
by argument validation unless explicitly allowed.  Current production sidecars
are contact-motion/task-aware typed evidence, not blind SAM labels.
```

## 7. Current Answer

If the current step-indexed run continues to pass through 7350-7600 and later
fixed-window/CALVIN gates:

```text
Do not remove the production invariants.
Do not strengthen the auxiliary scaffolds.
Do not enable dormant predictive losses.
Do consider simplifying documentation and launch recipes so the maintained
profile is clear:
  runtime invariants on;
  object scaffold weak/decayed;
  high-risk prediction losses off;
  blind SAM/local refinement/VCAP off.
```

The most likely cleanup after acceptance is not code deletion.  It is
configuration hardening:

```text
1. preserve the fixed optimizer-group regression test;
2. preserve the semantic_trainable_scope=backbone_only production contract;
3. add a "production profile" launch script locally matching the remote run;
4. leave legacy branches behind explicit opt-in flags;
5. keep runtime logs for lr_group_picf_core and grad_norm_group_picf_core as a
   mandatory acceptance gate.
```

## 8. Remaining Watch Items

```text
1. step7350-7600 action_default_equiv:
   must be judged on the corrected step-indexed stream, then verified with a
   fixed-window no-update probe if it looks bad.  Do not use the legacy
   sampled-window 0.021 row as a direct checkpoint scalar.

2. active/downstream overlap:
   raw overlap may stay 1.0 because reserve rows are saturated; the action graph
   should use active/downstream overlap, which is currently healthy.

3. loss_total_minus_action:
   if it expands beyond about 0.02, the scaffold is too strong or a recipe bug
   is present.  It is currently about 0.0105.

4. proposal/sidecar path:
   current run has proposal_memory_enabled=True for inspected contact-motion
   sidecars.  This is acceptable, but blind SAM roots remain forbidden.

5. optimizer group telemetry:
   every FSDP run that claims a slower PICF core must log lr_group_picf_core and
   grad_norm_group_picf_core.
```

# 2026-05-19 Tactile/Object Slot Binding Audit

Status: analysis and repair plan. This file records the current failure mode
after the object-owner-only pull probe and compares PICF-AQR-OWM against recent
object-centric/tactile robotics work. It is not a behavior acceptance report.

## 1. Observed Failure

Run:

```text
picf_a7_object_owner_only_pull_probe_1000_postclosure_20260519
```

The probe is intentionally narrow:

```text
trainable_scope = anchor_only
action loss = 0
semantic/action heads frozen
role layout = object_only
role-0 effector anchors = disabled
lambda_anchor_object_pull = 1
anchor_object_pull_graph_weight = 1
anchor_object_pull_posterior_weight = 1
posterior_file_competition_max_per_role = 1
```

At step 50/100:

```text
proposal center px ≈ [151.5, 31.4] / [151.8, 31.0]
active graph anchor distance to proposal ≈ 1 px
active posterior distances to proposal ≈ 32-52 px
aqr_active_anchor_count_role_0 = 0
posterior_file_competition_active_count = 2
loss_anchor_object_pull: 7.34 -> 4.27
posterior_identity_switch_rate: 0.148 -> 0.191
```

Interpretation:

```text
The measurement graph can bind the sidecar/mask object.
The posterior belief files do not inherit that binding.
The current failure is graph -> observation -> posterior closure, not sidecar
generation and not graph-query capacity.
```

## 2. Recent Paper Pattern

### Object Binding in Pretrained ViTs

`Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?`
argues that pretrained ViTs encode an IsSameObject relation in a low-dimensional,
pairwise/quadratic subspace, and that this relation helps guide attention.

Implication for PICF:

```math
s_{ij}^{same} =
z_i^\top W z_j
+ u^\top z_i
+ v^\top z_j
```

This supports PICF's projected binding-signature path, but it does not by
itself solve persistent object-file ownership. A same-object subspace can score
whether two measurements match; the belief update still needs a deterministic
write-back contract from accepted measurements into the persistent file.

### SlotVLA / STORM / Embodied-SlotSSM

Recent robotics slot work is converging on three invariants:

```text
1. slots must be stabilized before strong action co-training;
2. task-relevant object slots are few and filtered, not every background token;
3. persistent object identity is updated through the same responsibility/mask
   that selected the object.
```

This directly explains why a downstream-only pull loss is insufficient. If the
slot responsibility that found the object is not the same responsibility used to
write the posterior state, the model can satisfy graph-level metrics while the
belief file stays elsewhere.

### OCRA / SPOT / ViTaSCOPE / OmniVTA / OmniVTLA

The more tactile/contact-oriented 2025-2026 work does not treat tactile as a
free-floating gripper token:

```text
OCRA:
  builds object-centric 3D priors, separates manipulable/context objects, and
  fuses tactile priors after object-centric reconstruction.

SPOT:
  makes object pose trajectory the intermediate control representation.

ViTaSCOPE:
  registers tactile/contact fields onto object geometry.

OmniVTA:
  predicts visuo-tactile contact evolution instead of treating tactile as a
  passive observation.

OmniVTLA:
  aligns tactile representation semantically before using it in VLA policy.
```

Shared implication:

```text
contact/tactile evidence should attach to the contacted object state, not to an
independent gripper/effector owner that competes for the same object evidence.
```

PICF's recent role-0 removal in the object-owner probe is therefore conceptually
right. The remaining issue is not role-0; it is the missing hard/soft transport
from accepted object measurement to posterior owner file.

## 3. Current PICF Dataflow Compared To Paper Pattern

Current intended chain:

```math
M_{sidecar}, M_{point}, M_{vjepa}, M_{tactile}
\rightarrow
p^g_{j,i}
\rightarrow
A_{o,g}
\rightarrow
B_{b,o}
\rightarrow
b_t
```

Where:

```text
p^g_{j,i}: graph/object-owner support over evidence tokens
A_{o,g}: observation-anchor assignment to graph anchors
B_{b,o}: posterior-file binding to observation anchors
b_t: persistent posterior belief file
```

The observed probe shows:

```text
p^g is correct.
B * A * p^g is not strong enough to move b_t.x.
```

Current `_posterior_update` still forms posterior geometry mainly from:

```math
x_b =
\frac{\sum_o B_{b,o} x_o}{\sum_o B_{b,o}}
```

where `B` is produced by hidden/geometry/binding/role biases and Sinkhorn. The
accepted object candidate prior is only an additive bias path, not an
authoritative measurement write. If initial posterior files are far away, the
geometry and prior identity terms can dominate and the file does not move to
the object, even when graph anchors are correct.

## 4. Root Cause

The model is not failing because it cannot see the object. It is failing because
PICF currently lacks an explicit object-owner measurement write-back:

```math
accepted_object_measurement
\nRightarrow
posterior_owner_file_geometry
```

The current object pull loss tries to close this after the fact:

```math
L_{pull}
=
\lambda_g \rho(x^g_j, \bar{x}_j)
+
\lambda_b \rho(x^b_k, \bar{x}_k)
```

but the forward update itself still lets the posterior choose a different
measurement mixture. This is weaker than modern slot/OCL systems, where the
responsibility/mask used to select the object is also the responsibility that
updates the object slot.

## 5. Repair Principle

Do not add another generic overlap penalty. Add a belief-state closure operator:

```math
T_{b,p}
=
B_{b,o} A_{o,g} P_{g,p}
```

where `P` is the accepted object-owner graph prior over point/proposal evidence.

Then create a trusted owner measurement:

```math
\bar{x}^{obj}_b
=
\frac{\sum_p T_{b,p} x_p}{\sum_p T_{b,p}}
```

and a confidence:

```math
c^{obj}_b
=
\mathrm{clip}
\left(
  owner_b \cdot assignment\_margin_b \cdot proposal\_quality_b
  \cdot (1 - duplicate_b),
  0, 1
\right)
```

The posterior geometry update should use this as a high-precision measurement,
not as a plain convex interpolation coefficient:

```math
\Lambda_b^{+}
=
\Lambda_b^{std}
+
\kappa c^{obj}_b \Lambda_b^{obj},
\quad
\eta_b^{+}
=
\Lambda_b^{std} x_b^{std}
+
\kappa c^{obj}_b \Lambda_b^{obj}\bar{x}^{obj}_b
```

```math
x_b^{+}
=
(\Lambda_b^+)^{-1}\eta_b^+,
\quad
S_b^{+}
=
(\Lambda_b^+)^{-1}.
```

This matters because `c_b` is a responsibility/reliability gate, while
`S_b^{obj}` is the measurement uncertainty.  A reliable object/contact
measurement should dominate a stale posterior file through precision, not only
move the file by `c_b` percent of the distance.

This is not a hard sidecar label:

```text
- the source must pass object-candidate / task-owner / proposal quality gates;
- the confidence is soft;
- dense V-JEPA, point, tactile, tracklet tokens remain available;
- posterior remains the authoritative state;
- bad sidecars can be routed to background/no-object.
```

## 6. Concrete Code Target

The repair belongs inside:

```text
src/openpi/picf/core/pipeline.py::_posterior_update
```

not only inside:

```text
src/openpi/picf/core/training.py::_object_anchor_pull_loss
```

Implementation target:

```text
1. Build transported owner point prior:
   post_owner_point_prior = B_post,obs @ (A_obs,graph @ graph_owner_point_prior)

2. Build owner confidence:
   use posterior.file_competition_active, lifecycle assignment confidence,
   owner reliability, object candidate coverage, and duplicate demotion.

3. Fuse owner measurement into x/S/a before token_in/posterior token write:
   x_standard from current binding;
   x_owner from transported owner prior;
   x = (1-c)x_standard + c x_owner.

4. Store metrics:
   posterior_owner_transport_mass_mean
   posterior_owner_transport_conf_mean
   posterior_owner_transport_dist_to_graph_mean
   posterior_owner_transport_applied_fraction

5. Overlay:
   report distance from active posterior to proposal/mask center.
```

This is the mathematically direct fix for the observed gap.

## 7. Acceptance Test

The object-owner-only probe should pass these before any action co-training:

```text
role0 active count = 0
graph active object distance to proposal center <= 3 px
posterior active object distance to proposal center <= 10-15 px by step 100
posterior_file_competition_active_count ~= 1 for object-only max-per-role=1
posterior_identity_switch_rate should not rise from step 50 to 100
loss_anchor_object_pull should fall and overlay should confirm posterior motion
```

If graph passes but posterior fails again, the repair is still incomplete.

## 8. What This Says About Maturity

PICF is not conceptually wrong. Its typed-memory + posterior-belief design is
stronger than a pure slot tokenizer for control. But relative to the newest
object-centric robotics papers, the previous implementation was immature in one
specific place:

```text
object evidence selection and persistent object-file update were not the same
operator.
```

That is the key missing module. Adding it is not feature creep; it is the
belief-filter equivalent of slot responsibility write-back.

## 9. 2025+ Tactile/Object-Slot Design Pattern

This section records the design rule that should drive the next implementation
rather than ad-hoc loss additions.

Recent tactile/object-centric robotics systems differ in details, but their
binding pattern is consistent:

```text
candidate evidence -> responsibility/mask -> object slot update -> temporal
identity/memory -> action/control
```

The important part is not the exact proposal generator. It is that the same
responsibility that selects object evidence also writes the object state. A
system that computes a correct object graph anchor but then lets posterior files
update from a different hidden/geometry mixture is only half object-centric.

PICF currently has most of the surrounding machinery:

```text
typed evidence:
  V-JEPA, point/geometry, tactile/contact, sidecar proposal, tracklet schema

competition:
  graph support competition, posterior file competition, inactive/context files

identity:
  binding signatures, calibrated file-swap telemetry, posterior file memory

contact:
  tactile can attach to object-owner path instead of an effector-only path
```

The missing mature-slot operator is:

```text
accepted object responsibility must directly close the posterior object-file
state update.
```

This is the same structural principle used by mask-/responsibility-based slot
systems. It is also the only explanation consistent with the latest probe:
graph object center is correct to about one pixel, while posterior active files
remain tens of pixels away.

## 10. Why More Losses Are Not The Right Fix

A posterior-only pull loss is useful as a probe, but it is not the correct
production mechanism. It asks gradient descent to learn a closure that should
already exist in the probabilistic update:

```math
L_{pull}(x^b, x^{obj})
```

does not guarantee that the next forward pass uses the object responsibility to
form the posterior measurement. The mature design is:

```math
q(o \mid b) \quad\Rightarrow\quad x_b^+
=
(1-c_b)x_b^{std} + c_b x_b^{obj}
```

where `c_b` is a soft reliability gate from the same assignment chain. This is
not stronger supervision; it is the correct belief-filter factorization.

Additional generic penalties, stronger SAM proposals, or more anchor rows would
not fix the observed bug. They can improve candidate quality, but the candidate
is already good in the probe. The failure is downstream ownership closure.

## 11. Concrete Next Implementation

The next code change should implement owner-responsibility transport in
`_posterior_update`:

```text
graph owner rows:
  rows with object-candidate / task-owner / proposal support and allowed
  object/contact roles.

obs owner responsibility:
  A_obs,graph transports graph-owner responsibility into observation anchors.

posterior owner responsibility:
  B_post,obs transports observation-owner responsibility into posterior files.

posterior owner measurement:
  weighted object measurement x/S/signature built from the transported
  responsibility.

belief closure:
  x/S/signature are softly fused into the standard posterior update using a
  reliability gate, not a hard overwrite.
```

The gate must be high only when:

```text
object/proposal quality is nontrivial;
assignment confidence is nontrivial;
posterior file is active or birth-eligible;
duplicate/file competition did not demote the file;
owner reliability is nontrivial;
the transported mass is concentrated enough.
```

This keeps the design robust to noisy sidecars and sparse contact.

## 12. What Must Not Be Added Now

Do not reintroduce blind SAM as a production branch. The CALVIN trial showed
class-agnostic SAM creates many task-irrelevant object fragments. The generic
proposal schema remains valid, but accepted proposals must be contact/task/
tracklet-aware.

Do not restore an effector owner file as the default object owner. Tactile and
contact are evidence about the contacted object. The gripper can provide
proprioceptive/context evidence, but it should not compete as a persistent
object file for the manipulated object in the object-owner probe.

Do not rely on long training to fix a missing state-update operator. If graph
binding is correct and posterior ownership is wrong in an isolated pull probe,
the repair belongs in the posterior update.

## 2026-05-19 update: implemented posterior owner-responsibility closure

Implemented code-level closure in `src/openpi/picf/core/pipeline.py`:

```math
R^{graph}_{g,p} = \max(R^{cand}_{g,p}, R^{prop}_{g,p}, R^{task}_{g,p})
```

Only eligible object-owner graph rows are allowed to carry this responsibility:

```math
R^{graph}_{g,p} \leftarrow R^{graph}_{g,p}\,1[role(g)\in\mathcal R_{owner}]\,1[active(g)].
```

The accepted graph responsibility is transported through the existing measurement assignment chain:

```math
R^{obs}_{o,p}=\sum_g A^{obs\leftarrow graph}_{o,g}R^{graph}_{g,p},
\quad
R^{post}_{b,p}=\sum_o B^{post\leftarrow obs}_{b,o}R^{obs}_{o,p}.
```

The resulting proposal/point mean and covariance become a bounded
posterior-file measurement:

```math
\hat x_b=\frac{\sum_p R^{post}_{b,p}x_p}{\sum_p R^{post}_{b,p}},
\quad
\hat S_b=\frac{\sum_p R^{post}_{b,p}(S_p+(x_p-\hat x_b)(x_p-\hat x_b)^T)}
{\sum_p R^{post}_{b,p}}.
```

It is fused as a high-precision measurement:

```math
\Lambda_b^{+}
=
(S_b^{std})^{-1}
+
\kappa c_b(\hat S_b)^{-1},
\quad
x_b^{+}
=
(\Lambda_b^+)^{-1}
\left((S_b^{std})^{-1}x_b^{std}
+
\kappa c_b(\hat S_b)^{-1}\hat x_b\right).
```

The confidence `c_b` is gated by role eligibility, file/birth activity, lifecycle assignment confidence, lifecycle owner reliability, and a max-per-role object-file cap. This is intentionally not a new loss: it closes the belief-filter dataflow so the object mask/contact responsibility can become posterior geometry.

Runtime contract exposure was added to `scripts/picf_core_train.py` via `--posterior-owner-transport-*` flags, and the owner-only probe launcher now sets them explicitly. Static audit `picf_object_candidate_slot_binding_audit.py` checks that the closure helper, CLI flags, posterior-state fields, and log metrics are present.

## 2026-05-19 update: removed the remaining circular gate

The first closure probe showed a sharper diagnosis:

```text
graph/object candidate:
  active graph owner reached the sidecar/mask center by step 100.

posterior file:
  active posterior file still did not sit on the sidecar/mask center.
```

This means the failure is no longer object evidence or graph assignment.  The
remaining bug was circular:

```math
c_b \propto R^{post}_b \cdot file\_active_b
```

so an inactive file could not use strong owner responsibility to become the
active object file.  That violates the belief-filter contract: a newly explained
object measurement must be able to create or activate the corresponding
persistent file, subject to role capacity and reliability gates.

The repaired update is:

```math
c_b =
R^{post}_b\,
g_{role}(b)\,
g_{activity-prior}(b)\,
g_{life}(b)\,
(1-d_b)
```

where

```math
g_{activity-prior}(b)
= \rho_{inactive} + (1-\rho_{inactive})\,file\_active_b .
```

Thus prior active files are favored, but inactive files are not hard-blocked
when they receive strong owner responsibility.  The owner confidence then
participates in final action-visible file selection:

```math
file\_gate_b
\leftarrow
\max(file\_gate_b,\; clamp(c_b/\tau_{active},0,1)).
```

Finally the same per-role file-cap is applied after birth and owner activation:

```math
|\{b: file\_gate_b>0,\ role(b)=r\}|\le K_r.
```

For the object-only probe this enforces one active role-1 posterior file rather
than allowing an old active file plus a birth file to both remain visible.

This is not an extra auxiliary loss.  It is the missing posterior lifecycle
operator: accepted object responsibility is allowed to activate the file that it
updates, then the existing capacity constraint chooses the final visible file.
Overlay JSON now records `owner_transport_mass`,
`owner_transport_confidence`, and `owner_transport_dist_to_standard` per
posterior anchor so this follow-through can be checked directly.

## 2026-05-19 update: owner evidence is now dominant in same-role file selection

The first active-gate probe exposed one more ordering bug:

```text
slot 1:
  received transported owner confidence and was close to the mask center,
  but remained inactive.

slot 4:
  had larger generic support mass, no owner confidence, and became the sole
  active posterior file.
```

This is mathematically inconsistent with object-file ownership.  Generic
support mass is evidence that a file explains observations; transported owner
confidence is evidence that the file explains the selected task/contact object.
When both compete under a single same-role capacity cap, owner evidence must be
lexicographically dominant.  Otherwise the model can detect the object in the
transient graph and still expose a different belief file to the action path.

The final selection now applies:

```math
if\ \exists b: role(b)=r,\ c_b\ge\tau:
  file\_gate_{b'}=0\quad
  \forall b': role(b')=r,\ c_{b'}<\tau .
```

The remaining per-role cap then ranks files by:

```math
score_b = 100 c_b + file\_active_b + birth_b + support\_mass_b.
```

The large owner coefficient is not a new hyperparameterized loss.  It encodes
the ordering contract: accepted object-owner responsibility is the primary
criterion for which same-role file becomes the visible object file; generic
support is only a tie-breaker among files with comparable owner confidence.

## 2026-05-19 update: owner geometry must be precision-fused, not convex-blended

The owner-priority probe fixed file selection at step 50, but step 100 exposed
a final geometry bug:

```text
active graph owner:
  still near sidecar/mask center (~1-2 px).

active posterior file:
  selected by owner confidence, but offset from the same mask center (~20 px).
```

The reason was mathematical, not another evidence problem.  The implementation
used:

```math
x_b^+=(1-c_b)x_b^{std}+c_b\hat x_b.
```

With `c_b≈0.6`, a stale `x_b^{std}` can still pull the posterior 40 percent of
the way away from the accepted object measurement.  That violates the intended
belief-filter semantics: once a sidecar/contact object owner is accepted, it is
a measurement with covariance, not a weak residual suggestion.

The implementation now uses precision fusion:

```math
\Lambda_b^+=(S_b^{std})^{-1}+\kappa c_b(\hat S_b)^{-1},
\quad
x_b^+=(\Lambda_b^+)^{-1}
\left((S_b^{std})^{-1}x_b^{std}+\kappa c_b(\hat S_b)^{-1}\hat x_b\right).
```

`posterior_owner_transport_precision_gain` controls `\kappa`.  This is still
bounded and uncertainty-aware, but it lets a precise owner/contact measurement
override a stale posterior file without adding another training loss.

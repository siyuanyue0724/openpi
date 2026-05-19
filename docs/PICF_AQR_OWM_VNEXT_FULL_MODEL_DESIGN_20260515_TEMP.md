# PICF-AQR-OWM vNext Full Model Design

Status: design and verification plan for the post-MVTrack slot/object-file fixes.

This document answers the 2026-05-15 audit question: which remaining slot/object
issues should be fixed, whether they can be fixed without human object labels,
and how to do so without turning PICF into an incoherent collection of patches.

The conclusion is strict:

```text
Immediate structural fix:
  posterior lifecycle calibration.

Immediate no-label verification:
  IsSameObject-style binding subspace audit.

Guarded next training hook:
  temporal identity contrast / matched prediction, only after lifecycle metrics
  are stable.

Do not immediately rewrite:
  full generative object-mask slot model or full OA-WAM-style world/action model.
```

The design goal is not to add modules for their own sake. The goal is to make
the PICF object-file belief update mathematically explicit:

```math
b_t = U_\theta(P_\theta(b_{t-1}, a_{t-1}), E_\theta(o_t, \ell)).
```

Every new component below must be one of:

```text
1. evidence extraction E_theta,
2. measurement-to-object assignment,
3. posterior lifecycle calibration,
4. auxiliary offline/guarded verification,
5. prediction target with future detach.
```

Anything outside those categories is rejected unless it has a separate
behavioral ablation.

---

## 1. Current Facts From Code

The current implementation already has several important repairs:

```text
binding_signature_centering_enabled = True
posterior_slotwise_recycle_residual = True
cache residual scaling is output-level gated
latest posterior cache row is skipped
active same-role support overlap is substantially reduced in short diagnostics
```

The relevant code paths are:

```text
src/openpi/picf/core/config.py
src/openpi/picf/core/pipeline.py
src/openpi/picf/core/training.py
scripts/picf_binding_signature_common_mode_audit.py
```

The latest centered short diagnostic shows that the old same-role support
collapse is largely repaired, but posterior lifecycle remains undercalibrated:

```text
active same-role support overlap:
  healthy / low in the centered run.

binding-signature common-mode similarity:
  reduced from near-collinear to a usable subspace.

posterior recycle/dustbin/identity-switch:
  still high enough that the object-file lifecycle should not be called mature.
```

This is the key distinction:

```text
AQR support routing problem:
  mostly fixed.

Posterior object-file lifecycle problem:
  not yet independently calibrated.
```

---

## 2. Paper Code Review Boundary

We inspected external paper code as design evidence, not as drop-in code:

```text
/tmp/picf_paper_code_20260515/vit-object-binding
/tmp/picf_paper_code_20260515/slotcontrast
```

Important boundary:

```text
vit-object-binding:
  no LICENSE file found in the cloned repository, so code must not be copied
  directly into PICF. The IsSameObject probe protocol and pairwise/quadratic
  subspace idea can be reimplemented natively.

slotcontrast:
  MIT licensed. The object-level temporal contrast idea is directly relevant,
  but the PICF implementation should still be native because PICF uses
  posterior object files, typed memory, and action-conditioned recurrence.

OA-WAM:
  useful architectural direction for address/content separation, but a full
  rewrite is a major model change and should not be conflated with the immediate
  lifecycle fix.
```

Therefore the correct engineering policy is:

```text
reuse equations and verification protocols;
do not blindly import training loops or model classes.
```

---

## 3. Gap 1: Posterior Lifecycle Is Not Independently Calibrated

### 3.1 Why This Must Be Fixed

The current posterior update mixes three different concepts:

```text
assignment support:
  which observation belongs to which existing object file.

recycle/reset:
  whether an existing object file should be overwritten.

dustbin/unexplained evidence:
  which observation rows are background, duplicate reserve anchors, or new
  object evidence.
```

In code, the recycle head consumes:

```text
h_prior
support_mass_raw
prior variance
slot residual summary
alpha_prior
```

This is coherent but not independently calibrated. It lets the network infer
object-file lifecycle implicitly from a single vector. The current diagnostics
show that this is not enough: after fixing support overlap and binding subspace,
recycle/dustbin can still become high.

### 3.2 Mathematical Fix

Replace the implicit lifecycle decision with a factored lifecycle model:

```math
\begin{aligned}
c^{assign}_j &= f_c(m_j, H_j, \Delta_j, \alpha^-_j), \\
s_j &= \sigma(f_s(c^{assign}_j, \alpha^-_j, \nu_j, r_j)), \\
\rho_j &= \sigma(f_\rho(1-c^{assign}_j, \nu_j, u_j, \alpha^-_j)), \\
d_i &= f_d(owner_i, active_i, role_i, confidence_i, novelty_i).
\end{aligned}
```

Where:

```text
m_j:
  raw support mass for posterior slot j.

H_j:
  support entropy / concentration.

Delta_j:
  top1-top2 assignment margin.

nu_j:
  innovation risk.

u_j:
  unexplained evidence routed near slot j.

owner_i / active_i:
  active-slot reliability for observation row i.
```

The posterior update should use:

```math
binding_support_j = s_j \cdot support_j + \rho_j \cdot birth_residual_j.
```

With hard invariants:

```text
1. inactive reserve anchors cannot create recycle residual.
2. low-confidence background cannot reset object files.
3. a stable object with high support and low innovation should have low reset.
4. high innovation can reduce address inertia but should not blindly recycle.
5. birth/recycle must be role-compatible.
```

### 3.3 Implementation Plan

Add a native lifecycle calibration layer:

```python
class PicfPosteriorLifecycleState:
    assign_confidence: Tensor
    assign_margin: Tensor
    support_entropy: Tensor
    survival_prob: Tensor
    reset_prob: Tensor
    birth_prob: Tensor
    inactive_dustbin_mass: Tensor
    unexplained_object_mass: Tensor
```

Code location:

```text
src/openpi/picf/core/contracts.py
src/openpi/picf/core/pipeline.py::_posterior_update
```

Do not expose this as an auxiliary loss first. It should first be a deterministic
calibration and debug decomposition. Only after short-run metrics show stable
posterior lifecycle should a tiny learned calibration head be allowed.

### 3.4 Acceptance Criteria

```text
posterior_dustbin_mass_raw:
  should not rise solely because binding signatures were centered.

posterior_recycle_rate:
  should be low for stable high-support slots.

posterior_identity_switch_rate:
  should fall together with stable support overlap, not decouple from it.

debug decomposition:
  must show whether dustbin mass is inactive reserve, low confidence background,
  or unexplained object evidence.
```

---

## 4. Gap 2: Binding Subspace Can Be Learned/Audited Without Human Labels

### 4.1 User Constraint

The dataset has no precise object identity labels. The intended design relies on
large data and self-supervised emergence. This is acceptable, but it requires a
probe/audit because the object-binding paper shows that same-object information
may exist in a subspace that is not visible through raw cosine similarity.

### 4.2 No-Label IsSameObject Audit

Use weak positives:

```text
same stable posterior slot across adjacent frames;
same high-confidence tracklet id when available;
same point neighborhood under small motion;
same tactile/contact neighborhood;
same high-overlap PG/V-JEPA support peak;
same low-innovation object file across burn-in.
```

Use weak negatives:

```text
different active same-role anchors with low support overlap;
far geometry under the same frame;
different tracklet ids;
mutually exclusive owner-active rows;
different contact neighborhoods.
```

Train or evaluate an offline pairwise score:

```math
score(i,j) = z_i^\top W z_j,
```

with probe variants:

```text
dot product
cosine
diagonal quadratic
low-rank quadratic
full quadratic for small diagnostic batches only
```

This is not a new production loss. It is a scientific instrument:

```text
If the probe cannot separate weak same/different pairs, then AQR cannot be
expected to discover stable object binding from those embeddings alone.
```

### 4.3 Implementation Plan

Add:

```text
scripts/picf_owm_same_object_probe.py
```

Inputs:

```text
saved metrics/evidence bundles
anchor overlay metadata if available
optional cached token dumps from diagnostic runs
```

Outputs:

```text
same/different AUC
calibration curve
per-modality separability:
  V-JEPA static
  V-JEPA wrist
  PG image
  point
  tactile
  posterior
```

This can be done without modifying CALVIN labels.

### 4.4 Phase B Deployment: Training Overlay Signature Audit

The offline probe now supports both evaluation and training diagnostics:

```text
Evaluation path:
  --anchor-debug /path/to/anchor_debug.jsonl

Training path:
  --anchor-overlays /path/to/run/anchor_overlays
  --overlay-source posterior
```

Training overlays remain light by default. Dense signatures are exported only
when explicitly requested:

```bash
--anchor-overlay-interval 100 \
--anchor-overlay-dump-signatures
```

This is intentionally a diagnostic switch, not a model switch. It does not add
loss terms, does not perform an extra forward pass, and does not alter V-JEPA or
tactile buffers. It closes the dataflow gap that previously forced same-object
analysis to rely on evaluation artifacts only.

The mathematical probe uses weak positive/negative pairs from adjacent frames:

```math
y_{i,j}=1
\quad \text{if}\quad
\lVert x_i^t-x_j^{t+1}\rVert_2 \le \epsilon_x
\ \land\
\lVert p_i^t-p_j^{t+1}\rVert_2 \le \epsilon_p,
```

```math
y_{i,j}=0
\quad \text{if}\quad
\lVert x_i^t-x_j^{t+1}\rVert_2 \ge \delta_x
\ \lor\
\lVert p_i^t-p_j^{t+1}\rVert_2 \ge \delta_p.
```

It reports AUC for:

```text
geometry
visual/point sparse support
support_signature cosine
binding_signature cosine
combined score
within-frame duplicate candidate fraction
```

The deployed probe is not a cosine-only shortcut. It includes the same
quadratic family used by the object-binding analysis, reimplemented natively:

```text
--quadratic-probe diag_quadratic
--quadratic-probe low_rank_quadratic
--quadratic-probe full_quadratic
--quadratic-probe all
```

The trained probe scores are:

```math
s_{diag}(x,y)=\sum_k w_k x_k y_k + b,
```

```math
s_{lr}(x,y)=
\frac{1}{2\sqrt r}
\left(
  \langle A x, B y\rangle
  +
  \langle A y, B x\rangle
\right)
+ b,
```

```math
s_{full}(x,y)=
\frac{1}{\sqrt d}x^\top\frac{W+W^\top}{2}y+b.
```

This is the non-truncated version of the diagnostic: raw cosine tells whether
the current exported signature already separates objects linearly by angle;
diagonal/low-rank/full quadratic probes test whether the same-object relation
is present in a richer pairwise subspace, as reported for pretrained ViT
features. The result is still an audit, not an online loss.

The key decision logic is:

```text
binding_signature AUC high + duplicate fraction high:
  the same-object subspace is decodable, but assignment/lifecycle still creates
  duplicate candidates.

binding_signature AUC low:
  do not increase identity inertia; improve evidence/signature extraction or
  add real temporal correspondence.
```

---

## 5. Gap 3: Temporal Identity Needs a Guarded SlotContrast-Style Teacher

### 5.1 Current State

The code already has:

```text
lambda_slot_jepa = 0.0
lambda_support_pred = 0.0
lambda_binding_consistency = 0.0
```

This default is correct. These hooks should not be opened until lifecycle
stability is demonstrated.

### 5.2 Professional Fix

Use a temporal identity objective only for stable slots:

```math
L_{temp}
=
\operatorname{CE}
\left(
  \frac{q(c_{t,j})^\top k(\operatorname{sg}(c_{t+1,k}))}{\tau},
  \pi_{j,k}
\right).
```

The target assignment \(\pi\) is not fixed index. It is matched by:

```text
posterior binding continuity
geometry proximity
support signature overlap
role compatibility
low recycle
low innovation
```

Masks:

```text
alpha high
support entropy low
owner active high
recycle low
innovation low
role compatible
```

Hard invariant:

```text
future tokens are detached and never enter current action/AQR input.
```

### 5.3 Why Not Full Immediate JEPA

Opening slot-JEPA before lifecycle is calibrated can enforce a bad identity map:

```text
stable wrong address -> predictive loss rewards the wrong continuity.
```

Therefore temporal contrast is a second-stage training hook, not the immediate
repair for current dustbin/recycle behavior.

---

## 6. Gap 4: Objectness Without Human Masks Is Possible, But Not as Hard RGB Masks

### 6.1 What Is Not Feasible Immediately

Without object masks or precise instance labels, a full generative object-mask
slot model is not a safe immediate transplant. It would likely learn:

```text
table/background reconstruction
lighting/color shortcuts
large static surfaces
```

instead of task-relevant manipulation objects.

### 6.2 What Is Feasible

Use weak objectness as typed evidence coverage:

```math
L_{cover}
=
\sum_i w_i \min_j d(e_i, slot_j),
```

and exclusivity:

```math
L_{exclusive}
=
\sum_{j \ne k}
\langle p_j, p_k \rangle
```

but only over task-relevant typed evidence:

```text
point/contact neighborhoods
V-JEPA temporal motion evidence
PG image support peaks
gripper/wrist temporal view
action endpoint neighborhoods
stable posterior support
```

This is weaker than full mask reconstruction but much better aligned with PICF:

```text
explain manipulation evidence, not every pixel.
```

### 6.3 Recommendation

Do not implement full RGB reconstruction first. Implement:

```text
objectness coverage audit
objectness/exclusivity debug metrics
optional tiny low-weight evidence coverage regularizer
```

Only consider mask/proposal memory after the IsSameObject probe and lifecycle
metrics show the embeddings support object separation.

---

## 7. Gap 5: Address/Content Routing Should Be Strengthened Incrementally

### 7.1 Current State

PICF already has:

```text
slot_address
slot_content
address-aware cache terms
binding address terms
slow address update
```

But it is not yet a full OA-WAM-style architecture where address is the dominant
object key and content is the time-varying state.

### 7.2 Correct PICF-Native Upgrade

Use address as identity prior, not truth:

```math
score_{j,i}^{cache}
=
\lambda_a \cos(W_a a_j, W_a a_i)
+ \lambda_c \cos(W_c c_j, W_c c_i)
+ \lambda_r 1[r_j=r_i]
- \lambda_{age} age_i
- \lambda_\nu \nu_i.
```

Gate address by lifecycle confidence:

```math
g_j
=
survival_j
\cdot \alpha_j
\cdot (1-reset_j)
\cdot \exp(-\kappa \nu_j).
```

Update address slowly:

```math
a^+_j =
\operatorname{norm}
\left(
  (1-\rho_j)a^-_j + \rho_j \tilde a_j
\right),
```

where:

```math
\rho_j \propto survival_j \cdot support_conf_j \cdot (1-reset_j).
```

On birth/reset:

```text
address should be reinitialized from role prior + current evidence summary,
not dragged through stale content.
```

### 7.3 What Not To Do

Do not make address a hard nearest-neighbor lock. That would create:

```text
early wrong identity -> cache lock-in -> action reinforces wrong object file.
```

Address is a prior under posterior correction, not a ground-truth ID.

---

## 8. Proposed vNext Execution Order

```text
Phase A: Lifecycle calibration
  implement debug decomposition and deterministic survival/reset/birth gates.

Phase B: IsSameObject probe
  no-label audit over V-JEPA/PG/point/tactile/posterior subspaces.

Phase C: Address/content strengthening
  use lifecycle-gated address in binding/cache/update paths.

Phase D: Temporal identity teacher
  SlotContrast-style matched contrastive objective, default lambda 0.

Phase E: Weak objectness coverage
  typed-evidence coverage/exclusivity audit, not full RGB masks.

Phase F: Optional proposal/mask memory
  only after the above passes; do not block current lifecycle fix.
```

This order is important. Temporal/predictive losses and strong address cache
should not be opened before lifecycle calibration; otherwise the model can
learn to preserve a wrong object file more consistently.

---

## 9. Dataflow Follow-Through

The intended vNext dataflow is:

```text
observation
  -> typed evidence memories
  -> AQR support routing
  -> centered binding signatures
  -> observation anchors
  -> posterior binding logits
  -> lifecycle calibration
  -> posterior object files
  -> action context
  -> bounded cache write
```

With explicit diagnostics:

```text
support:
  same_role_support_overlap
  active_same_role_support_overlap
  object_core_overlap

binding subspace:
  binding_signature_overlap_mean/max
  IsSameObject AUC

lifecycle:
  survival_prob_mean
  reset_prob_mean
  birth_prob_mean
  inactive_dustbin_mass
  unexplained_object_mass
  posterior_recycle_rate
  posterior_identity_switch_rate

address/cache:
  cache_address_score_mean
  cache_content_score_mean
  address_update_rate
  address_reset_fraction
```

Acceptance requires the metrics to agree. A low action loss alone is not enough.

---

## 10. Final Recommendation

The immediate fix should not be another support-overlap penalty. It should be:

```text
1. factor posterior lifecycle calibration;
2. add IsSameObject-style no-label audit;
3. strengthen address/content only through lifecycle-gated priors;
4. keep predictive/temporal/objectness losses guarded until lifecycle stabilizes.
```

This is the coherent path because it preserves the existing PICF invariants:

```text
posterior remains authoritative;
cache remains auxiliary;
future remains detached;
action does not receive raw future or hard object labels;
typed evidence stays modular;
missing modalities remain no-op.
```

The design is not a paper-code transplant and not a patch stack. It is a
posterior-centered object-file calibration upgrade derived from the same
belief-state equation that already defines PICF.

---

## 11. Phase A Deployment Record

2026-05-15 local deployment status:

```text
Implemented:
  posterior_lifecycle_calibration_enabled
  posterior_lifecycle_support_min / temperature
  posterior_lifecycle_margin_min / temperature
  posterior_lifecycle_entropy_weight
  posterior_lifecycle_owner_weight
  posterior_lifecycle_innovation_downweight

Implemented posterior state/debug:
  lifecycle_assignment_confidence
  lifecycle_support_entropy
  lifecycle_support_margin
  lifecycle_owner_reliability
  lifecycle_survival_prob
  lifecycle_reset_allowance
  lifecycle_recycle_raw
  lifecycle_inactive_dustbin_mass
  lifecycle_unexplained_dustbin_mass
```

The deployed rule is:

```math
recycle_j =
\sigma(r_j^{raw})
\cdot
(1 - survival_j),
```

where:

```math
survival_j =
\max(
  assignment\_confidence_j,
  \alpha^-_j \exp(-\lambda_\nu \nu_j)
).
```

The assignment confidence is:

```math
assignment\_confidence_j
=
support\_conf_j
\cdot margin\_conf_j
\cdot concentration\_conf_j
\cdot owner\_conf_j.
```

This means:

```text
high support + high margin + concentrated support + active-owner reliability:
  protect object-file identity from recycle/reset.

inactive reserve dustbin rows:
  are reported as inactive_dustbin_mass and do not drive object reset.

active unexplained dustbin rows:
  remain available as unexplained_dustbin_mass for true birth/reset evidence.
```

This is deliberately structural, not an extra loss. It preserves action training
and typed-memory routing while making posterior lifecycle auditable.

Required verification:

```text
python -m py_compile src/openpi/picf/core/config.py src/openpi/picf/core/contracts.py src/openpi/picf/core/pipeline.py src/openpi/picf/core/training.py
python scripts/verify_picf_owm_contract.py
pytest -q src/openpi/picf/core/pipeline_test.py::test_posterior_lifecycle_calibration_protects_stable_supported_slots
```

Remote diagnostic launch script:

```text
run_a7_lifecycle_calibration_diagnostic_20260515.sh
```

Primary comparison against the pre-lifecycle centered diagnostic:

```text
picf_a7_diag_binding_centered_u2b1_200_20260515
```

Acceptance focus:

```text
posterior_lifecycle_survival_prob_mean:
  should increase for stable high-support slots.

posterior_lifecycle_reset_allowance_mean:
  should be below raw recycle when assignments are confident.

posterior_lifecycle_inactive_dustbin_mass:
  should explain reserve/duplicate rows instead of feeding reset.

posterior_lifecycle_unexplained_dustbin_mass:
  should remain as true birth/reset evidence.

posterior_recycle_rate / posterior_identity_switch_rate:
  should fall without reintroducing active same-role support overlap.
```

---

## 14. 2026-05-15 Non-Truncated Paper-Code Follow-Through Addendum

This addendum records the stricter follow-through requested after the initial
binding-subspace patch. The goal was to check whether PICF had only implemented
a weak cosine shortcut or whether it actually followed the object-binding
paper's pairwise/quadratic protocol closely enough for a serious audit.

### 14.1 External code inspected

Local snapshots inspected:

```text
/tmp/picf_paper_code_20260515/vit-object-binding/src/utils/models.py
/tmp/picf_paper_code_20260515/vit-object-binding/src/trainer.py
/tmp/picf_paper_code_20260515/slotcontrast/slotcontrast/losses.py
```

Observed protocol:

```text
vit-object-binding:
  - diagonal quadratic probe
  - full quadratic probe
  - fixed-rank / low-rank quadratic probe
  - pairwise BCE over same-object / different-object patch pairs

slotcontrast:
  - adjacent-frame slot-slot contrastive identity loss
  - normalized slots
  - CE over slot correspondence logits
```

Engineering boundary:

```text
vit-object-binding has no LICENSE file in the inspected snapshot.
Therefore PICF must not copy its code. It may reimplement the published
pairwise/quadratic probe equations and validation protocol natively.

slotcontrast is MIT licensed, but its standalone slot objective is not a direct
drop-in replacement for PICF's posterior object-file state. The correct use is
as design evidence for later guarded, matched temporal slot losses.
```

### 14.2 Native PICF implementation

PICF now has a no-label offline probe that can read both evaluation
`anchor_debug.jsonl` artifacts and training `anchor_overlays/*.json` artifacts:

```text
scripts/picf_owm_same_object_probe.py
```

It supports:

```text
--anchor-debug <anchor_debug.jsonl>
--anchor-overlays <anchor_overlays_dir_or_step_json>
--overlay-source posterior|graph
--quadratic-probe diag_quadratic
--quadratic-probe low_rank_quadratic
--quadratic-probe full_quadratic
--quadratic-probe all
```

The trained native probe family is:

```math
s_{diag}(x,y)=\sum_k w_k x_k y_k + b,
```

```math
s_{lr}(x,y)=
\frac{1}{2\sqrt r}
\left(
  \langle A x, B y\rangle
  +
  \langle A y, B x\rangle
\right)
+ b,
```

```math
s_{full}(x,y)=
\frac{1}{\sqrt d}x^\top\frac{W+W^\top}{2}y+b.
```

This is a diagnostic probe, not an online model loss. The weak labels come from
adjacent-frame geometry/pixel proximity, not from ground-truth instance masks.
Using these weak labels as a training loss would risk self-confirming identity
lock-in; using them as an offline probe is mathematically safer.

### 14.3 Training artifact dataflow

Training overlays can now explicitly export signatures:

```bash
--anchor-overlay-dump-signatures
```

This writes:

```text
support_signature
binding_signature
```

into `anchor_overlays/step_*.json` without running an extra forward pass and
without advancing V-JEPA/tactile buffers outside the real training step. This is
important because the probe must audit the exact dataflow used by training, not
a side-effecting diagnostic rerun.

### 14.4 Local verification results

Commands rerun after this addendum:

```bash
python -m py_compile \
  scripts/picf_core_train.py \
  scripts/picf_owm_same_object_probe.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_binding_signature_common_mode_audit.py \
  scripts/picf_owm_professor_grade_audit.py

PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_binding_signature_common_mode_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_owner_gate_followthrough_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
PYTHONPATH=src uv run pytest -q \
  scripts/picf_owm_same_object_probe_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py \
  src/openpi/picf/core/pipeline_test.py::test_posterior_lifecycle_calibration_protects_stable_supported_slots \
  src/openpi/picf/core/pipeline_test.py::test_binding_signature_centering_removes_common_mode

git diff --check
```

Observed result:

```text
py_compile: PASS
verify_picf_owm_contract.py: PASS
nontruncated_paper_audit.py: 8/8 PASS
binding_signature_common_mode_audit.py: PASS
professor_grade_audit.py: 16/16 PASS
owner_gate_followthrough_audit.py: 12/12 PASS
picf_owm_dataflow_trace.py: ok, 20 nodes
targeted pytest: 9 passed, 1 warning
git diff --check: PASS
```

### 14.5 Remote status when audited

Remote A7 diagnostic session:

```text
tmux: picf_a7_lifecycle_phasea_20260515
run:  picf_a7_diag_lifecycle_calibrated_u2b1_240_20260515
```

At the audited metric point:

```text
step = 50
loss_total = 1.4536
loss_action_default_equiv = 0.1219
aqr_active_same_role_support_overlap_max = 0.0154
aqr_same_role_support_overlap_max = 0.1519
posterior_recycle_rate = 0.0587
posterior_lifecycle_survival_prob_mean = 0.7923
posterior_lifecycle_reset_allowance_mean = 0.2077
posterior_identity_switch_rate = 0.7217
posterior_identity_switch_rate_stable = 0.6054
preclip_grad_norm = 33.71
grad_norm = 5.0
```

Interpretation:

```text
support-overlap collapse:
  not present at step 50 in this run.

posterior recycle:
  materially improved relative to the earlier reset-saturated failures.

identity switch:
  still high; this remains the next behavioral acceptance metric and must not
  be hidden behind better overlap/recycle numbers.
```

### 14.6 Remaining non-negotiable boundaries

The implementation is now less truncated than the earlier cosine-only audit, but
the following statements remain strict:

```text
1. The quadratic IsSameObject probe is an audit, not a training loss.
2. Full instance-mask object binding cannot be claimed without masks or reliable
   pseudo-mask/tracklet evidence.
3. Tracklet/proposal branches remain no-op on CALVIN unless the dataflow supplies
   those fields.
4. Ordinal "fourth object" grounding remains weak diagnostic/auxiliary only.
5. A clean 30k run still needs anchor overlays, probe output, and CALVIN/video
   evidence before scientific closure.
```

### 14.7 Prepared non-truncated diagnostic run

Prepared scripts:

```text
run_a7_nontruncated_signature_probe_diagnostic_20260515.sh
run_a7_nontruncated_signature_probe_post_20260515.sh
```

Purpose:

```text
1. rerun the lifecycle-calibrated AQR configuration;
2. explicitly enable --anchor-overlay-dump-signatures;
3. save anchor overlays every 50 steps;
4. run the full diagonal/low-rank/full quadratic IsSameObject probe on the
   exported training overlays.
```

This is the first diagnostic that directly answers:

```text
does the runtime posterior binding_signature contain a decodable same-object
subspace on actual training overlays, or are good overlap/recycle metrics hiding
a weak object-binding substrate?
```

Remote deployment state:

```text
A7 current run:
  picf_a7_lifecycle_phasea_20260515

Queued follow-up run:
  tmux session = picf_a7_nontruncated_signature_wait_20260515
  run script   = run_a7_nontruncated_signature_probe_diagnostic_20260515.sh
  post script  = run_a7_nontruncated_signature_probe_post_20260515.sh

The queued session waits for the current lifecycle run to exit, then runs the
signature-dump diagnostic and the quadratic overlay probe automatically.
```

Latest observed lifecycle run state before queuing:

```text
step = 100
loss_action_default_equiv = 0.0821
aqr_active_same_role_support_overlap_max = 0.1034
aqr_same_role_support_overlap_max = 0.5785
aqr_same_role_obs_binding_signature_overlap_mean = 0.8126
posterior_recycle_rate = 0.0192
posterior_identity_switch_rate = 0.7106
posterior_lifecycle_survival_prob_mean = 0.8004
posterior_lifecycle_reset_allowance_mean = 0.1996
```

Interpretation:

```text
recycle saturation:
  mostly fixed in this run.

support/object overlap:
  improved relative to collapse failures, but not solved enough for closure.

binding signature common mode:
  still material at step100, so the queued quadratic overlay probe is necessary.

identity switch:
  still high, so low recycle alone is not sufficient evidence of object-file
  stability.
```

## 15. Runtime quadratic same-object binding closure

### 15.1 Why this is the next repair

The previous implementation had two different levels:

```text
offline audit:
  diagonal / low-rank / full quadratic IsSameObject probes over exported
  binding_signature vectors.

runtime posterior binding:
  support overlap + normalized binding_signature dot + address gate.
```

That left a gap. If the object-binding signal is actually pairwise/quadratic,
then proving it with an offline quadratic probe while using only a cosine-like
runtime score is incomplete.

### 15.2 Native runtime formulation

PICF now keeps the existing posterior binding base:

```math
\ell_{j,i}
=
\lambda_h \cos(h^-_j, o_i)
-
\lambda_g \operatorname{Maha}(x_i,\mu^-_j)
+
\lambda_s g_j O^{support}_{j,i}
+
\lambda_a g_j \cos(a^-_j,\tilde a_i)
```

and adds a gated same-object subspace score over centered binding signatures:

```math
q^{diag}_{j,i}
=
\frac{(b^-_j \odot d)^T b_i}{\sqrt D}
```

```math
q^{lr}_{j,i}
=
\frac{1}{2\sqrt R}
\left[
  (A b^-_j)^T(B b_i)
  +
  (B b^-_j)^T(A b_i)
\right]
```

```math
\ell_{j,i}
\leftarrow
\ell_{j,i}
+
g_j
\left(
  \lambda_{diag} q^{diag}_{j,i}
  +
  \lambda_{lr} q^{lr}_{j,i}
\right)
```

where:

```text
b:
  centered, support-weighted binding_signature.
d:
  diagonal quadratic reliability vector, initialized so q_diag starts
  identity-equivalent to the existing cosine subspace.
A,B:
  small low-rank symmetric quadratic factors.
g_j:
  alpha/recycle/innovation gate, so unstable or recycled object files do not
  hard-lock stale identity evidence.
```

This is a structural binding term, not an auxiliary loss. It does not require
mask labels, does not consume weak same-object pairs online, and does not bypass
posterior correction.

### 15.3 Code/dataflow follow-through

```text
config:
  bind_quadratic_signature_weight = 0.10
  bind_low_rank_signature_weight = 0.05
  binding_low_rank_signature_rank = 16

pipeline:
  binding_quadratic_diag
  binding_low_rank_left/right
  _binding_signature_quadratic_scores(...)
  _binding_logits(...) adds gated diag/low-rank scores next to the existing
  binding_signature dot score.

trainer:
  CLI threads --bind-quadratic-signature-weight,
  --bind-low-rank-signature-weight, and --binding-low-rank-signature-rank.
  anchor_only trainability includes the new quadratic binding parameters.

audits/tests:
  verify_picf_owm_contract.py checks the config/trainer/pipeline dataflow.
  picf_owm_nontruncated_paper_audit.py checks native implementation and the
  no-copy/no-weak-loss boundary.
  pipeline_test.py checks that the quadratic score can express a pairwise
  relation not reducible to plain positive cosine.
```

### 15.4 Self-critique

This repair is not a broad new module. It is the minimal closure of the
object-binding paper insight into the actual posterior binding equation. It
does not solve missing masks, ordinal rank supervision, or absent tracklet
data. Those remain data/evidence limitations. It does fix the inconsistency
where PICF audited a quadratic same-object subspace but did not use a quadratic
same-object score in the runtime binder.

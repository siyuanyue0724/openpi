# PICF-AQR-OWM Advanced Slot Theory Audit

Status: theory and code-architecture audit for the 2026-05-15 posterior
lifecycle / object-file repair.

This document answers a narrow question:

```text
Does the current PICF-AQR-OWM design still have theoretical holes when compared
with modern slot/object-centric methods, and what must be fixed before claiming
the architecture is mature?
```

The short answer:

```text
No fatal contradiction:
  The PICF design is a coherent posterior-centered belief router.

Still incomplete as a mature slot system:
  It lacks independent object-lifecycle calibration, an offline IsSameObject
  audit, and a proven temporal correspondence teacher.

Immediate repair:
  posterior lifecycle calibration, now implemented as Phase A.

Do not do immediately:
  full RGB mask reconstruction, hard address locking, or strong predictive
  losses before lifecycle stabilizes.
```

---

## 1. Mathematical Model

Let:

```math
o_t = {RGB_static, RGB_wrist, point, tactile, proprio, language, a_{t-1}}
```

be the observation, and let:

```math
b_t^j = (a_t^j, c_t^j, x_t^j, S_t^j, alpha_t^j, r_j)
```

be posterior object file `j`, where:

```text
a_t^j:
  persistent address / identity prior

c_t^j:
  time-varying content

x_t^j, S_t^j:
  geometry belief and uncertainty

alpha_t^j:
  activity/existence

r_j:
  role
```

The intended belief update is:

```math
b_t = U_theta(P_theta(b_{t-1}, a_{t-1}), E_theta(o_t, language)).
```

PICF decomposes this into:

```text
typed evidence:
  V-JEPA static/wrist, PG image/text, point, tactile, posterior, cache

AQR:
  query-to-typed-memory routing and support selection

observation anchors:
  measurement hypotheses

posterior binding:
  association from persistent object files to observation anchors

lifecycle:
  survival / reset / birth / dustbin calibration

action:
  PI0.5-style generator over current belief/context
```

This is not identical to Slot Attention. It is closer to a POMDP belief filter
with object-file state.

---

## 2. Advanced Slot Methods: Required Ingredients

### 2.1 Slot Attention

Slot Attention provides:

```text
1. exchangeable slots,
2. iterative competitive attention,
3. normalization/competition over input tokens,
4. task-dependent object binding.
```

PICF equivalent:

```text
exchangeable object files:
  persistent anchors / physical slots

competition:
  AQR support competition and active-slot filtering

input tokens:
  typed memory rather than a single image feature map
```

Remaining gap:

```text
Slot Attention usually gets objectness pressure from reconstruction or property
prediction. PICF does not have dense object masks or RGB reconstruction. It
therefore needs lifecycle and evidence-coverage calibration instead.
```

### 2.2 SAVi / video slot attention

Video slot methods add:

```text
1. recurrent slot state,
2. temporal continuity,
3. hints or motion/temporal evidence,
4. identity preservation across frames.
```

PICF equivalent:

```text
posterior carry:
  recurrent object files

temporal evidence:
  V-JEPA last-two temporal tokens and wrist/static view ids

identity continuity:
  binding signatures, address/content, cache
```

Remaining gap:

```text
Temporal identity is not independently proven. Slot-JEPA/support-pred hooks are
guarded and off, which is correct, but mature temporal consistency still needs a
matched/contrastive teacher after lifecycle stabilization.
```

### 2.3 SlotContrast / SlotMatch

SlotContrast shows that recurrent processing alone may not provide stable
object identity; an object-level temporal contrastive loss improves temporal
consistency. SlotMatch pushes a simpler view: if a good teacher exists, matching
corresponding slots by cosine can transfer object-centric representations
without many extra losses.

PICF implication:

```text
Do not add many unrelated auxiliary losses.
Use one clear temporal identity objective only after the posterior lifecycle is
stable.
```

Correct future form:

```math
L_temp =
CE(sim(q(c_t^j), sg(k(c_{t+1}^k))) / tau, pi_{j,k}),
```

where `pi` is matched by:

```text
support overlap
geometry
role compatibility
low recycle
low innovation
```

Wrong form:

```text
index-aligned slot loss under active permutation/recycle.
```

### 2.4 Grounded Correspondence / prediction-to-correspondence

Recent work argues that learned slot predictors can be replaced or simplified
by deterministic bipartite correspondence over strong frozen features.

PICF implication:

```text
Before opening learned predictive losses, first test whether V-JEPA/PG/point
features already contain enough object identity for matching.
```

This supports the offline IsSameObject audit and discourages opening JEPA
losses as a first response to recycle instability.

### 2.5 Slot merging / SlotCurri

Recent slot methods highlight two opposite failures:

```text
over-fragmentation:
  multiple slots explain the same object or object part.

under-fragmentation / collapse:
  multiple objects merge into one slot/support.
```

PICF historically showed both symptoms:

```text
same-role overlap high:
  active anchors duplicate the same support.

dustbin/recycle high:
  object files fail lifecycle stability even after support is separated.
```

PICF response should not be a generic merge operator first. Because PICF has
fixed persistent object files and action coupling, the safer equivalent is:

```text
active-slot filter:
  reserve duplicate anchors instead of forcing every query to bind an object.

lifecycle calibration:
  separate inactive duplicates from unexplained object evidence.
```

This is the Phase A repair.

### 2.6 OA-WAM-style address/content

OA-WAM motivates:

```text
persistent address vector + time-varying content vector.
```

PICF already has:

```text
slot_address
slot_content
address-aware cache
slow address update
```

Remaining gap:

```text
Address must be lifecycle-gated. Hard address locks are dangerous because early
wrong identity can cause cache/action lock-in.
```

Correct form:

```math
score_{j,i}^{cache}
= lambda_a cos(W_a a_j, W_a a_i)
+ lambda_c cos(W_c c_j, W_c c_i)
+ lambda_r 1[r_j=r_i]
- lambda_{age} age_i
- lambda_{nu} nu_i.
```

With:

```math
g_j = survival_j * alpha_j * (1 - reset_j) * exp(-kappa nu_j).
```

---

## 3. Layer-by-Layer PICF Audit

### 3.1 Typed Evidence Layer

Current status:

```text
static+wrist V-JEPA temporal support:
  implemented.

PG image support:
  implemented as first-class branch.

point/tactile:
  implemented.

tracklet/proposal:
  contracts and runtime paths exist, but typical CALVIN dataflow may feed zero
  tokens unless episode fields are present.
```

Theoretical gap:

```text
No fatal issue, but the model cannot create object identity that is absent from
all typed evidence. This remains an information limit:

I(Y; A_t | instruction) <= I(Y; Z_{<=t} | instruction).
```

### 3.2 AQR Routing Layer

Current status:

```text
support competition:
  implemented.

active slot filter:
  implemented.

ownership prior:
  implemented.

binding-signature centering:
  implemented and verified by common-mode audit.
```

Remaining risk:

```text
Same-role support overlap can be low while posterior lifecycle is still poor.
Therefore support overlap alone is not sufficient acceptance.
```

### 3.3 Posterior Binding Layer

Current binding score roughly contains:

```math
logit_{j,i}
= hidden_{j,i}
- geometry_{j,i}
+ support_signature_{j,i}
+ binding_signature_{j,i}
+ address_{j,i}
+ role/owner/occupancy/graph priors.
```

This is mathematically coherent. It is closer to modern pairwise object-binding
than raw hidden cosine.

Remaining gap:

```text
The pairwise same-object subspace is not independently validated. The correct
next diagnostic is an offline IsSameObject probe.
```

### 3.4 Lifecycle Layer

Before Phase A:

```text
recycle/reset was learned implicitly from support mass, prior variance, alpha,
and residual summary.
```

Problem:

```text
The model could confuse inactive reserve anchors, low-confidence background,
and true unexplained object evidence.
```

Phase A repair:

```math
recycle_j = recycle_raw_j * reset_allowance_j
```

where:

```math
reset_allowance_j = 1 - survival_j
```

and:

```math
survival_j =
max(assignment_confidence_j,
    alpha^-_j exp(-lambda_nu innovation_j)).
```

This directly addresses the current theoretical hole.

### 3.5 Address/Cache Layer

Current status:

```text
cache read is residual-gated.
latest posterior duplicate is skipped.
address/content terms exist.
```

Remaining risk:

```text
Address routing should be strengthened only after lifecycle is stable. Otherwise
it can preserve wrong identities more consistently.
```

### 3.6 Loss Layer

Current status:

```text
action loss:
  active.

anchor/PV/graph support losses:
  guarded but active in diagnostics.

slot_jepa/support_pred/binding_consistency:
  default zero.
```

This is correct. Modern slot papers do not imply "more losses is better".
SlotMatch especially supports simple matching over extra loss stacks when a
good object subspace exists.

### 3.7 Action Coupling Layer

The key risk is:

```text
action loss can reward task completion shortcuts before object files stabilize.
```

Current mitigation:

```text
action_prefix_stopgrad diagnostics
staged tests
anchor overlays
low action weights in some runs
```

Remaining requirement:

```text
If lifecycle calibration works, action can remain in cotrain. If lifecycle
metrics fail, the problem is not solved by lowering action weight alone.
```

---

## 4. Are We Missing a Critical Advanced Slot Module?

### 4.1 Missing But Not Immediate: Generative Masks

Advanced OCL often uses reconstruction/masks. PICF does not currently have a
pixel/object mask decoder. This is a real difference.

However, adding full RGB reconstruction now is not the right fix:

```text
CALVIN manipulation success depends on task-relevant objects and contact, not
reconstructing every table/background pixel.
```

The proper PICF-native alternative is:

```text
typed-evidence objectness coverage:
  cover point/contact/motion/PG/V-JEPA peaks, not all pixels.
```

### 4.2 Missing But Diagnosable: Independent Object-Binding Probe

The object-binding paper motivates testing whether same-object information is
available in a quadratic/pairwise subspace. PICF has the production binding
signature, but still needs an offline probe.

This is not optional for scientific closure, but it is not required before the
Phase A lifecycle diagnostic.

### 4.3 Missing But Guarded: Temporal Correspondence Teacher

SlotContrast / Grounded Correspondence suggest temporal identity must be
explicitly tested and eventually trained. But the teacher should be:

```text
matched
detached
low-weight
stable-slot only
after lifecycle stability
```

Opening it now would risk preserving bad object files.

### 4.4 Not Missing: Raw Slot Count Exhaustion

The overlay showing many colored anchors is not itself proof of failure. Modern
slot systems often use more slots than objects. The failure is not "many slots
exist"; the failure is when:

```text
active same-role support overlap high
posterior recycle high
stable-slot identity switch high
```

The active-slot filter is the correct capacity-control mechanism for PICF.

---

## 5. Concrete Current Verdict

The design still had one important theoretical hole before Phase A:

```text
posterior lifecycle was not independently calibrated.
```

That hole is now directly targeted by the Phase A implementation.

The design still has scientific gaps, but they are not the same category:

```text
IsSameObject probe:
  needed for proof, not immediate runtime repair.

temporal correspondence teacher:
  needed for mature object identity training, but only after lifecycle works.

typed objectness coverage:
  useful future refinement, not the current dustbin/recycle root cause.
```

Therefore the immediate deployment decision is:

```text
Run Phase A lifecycle diagnostic.

Do not open slot-JEPA/support-pred/binding-consistency yet.

Do not add full mask reconstruction yet.

Do not hard-lock addresses yet.
```

---

## 6. Acceptance Metrics For Current Trial

Compare:

```text
baseline:
  picf_a7_diag_binding_centered_u2b1_200_20260515

new:
  picf_a7_diag_lifecycle_calibrated_u2b1_240_20260515
```

Required improvements:

```text
posterior_recycle_rate:
  should be substantially below the old ~0.75 at step 200.

posterior_identity_switch_rate:
  should fall or at least not increase.

posterior_lifecycle_reset_allowance_mean:
  should be lower than lifecycle_recycle_raw_mean on stable batches.

posterior_lifecycle_inactive_dustbin_mass:
  should explain duplicate/reserve evidence.

active_same_role_support_overlap:
  must remain low; lifecycle repair must not reintroduce support collapse.
```

If these pass, the current architecture has addressed the live theoretical
failure mode and can move to longer cotrain validation.

If these fail, the next clean fix is not another diversity penalty. It is:

```text
offline IsSameObject probe + deterministic correspondence audit,
then lifecycle thresholds/gates calibrated from those diagnostics.
```


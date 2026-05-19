# PICF Multimodal Prototype And Reconstruction-Aux Roadmap

Date: 2026-05-19
Status: design roadmap, not yet a runtime change
Canonical entry: `src/openpi/picf/README_v2.2.md`

## 0. Why This File Exists

This note fixes the next two architecture decisions that should not be
rediscovered from scattered experiment logs:

1. MetaSlot-style VQ / prototype logic is useful, but must be translated into a
   PICF-compatible multimodal birth/proposal/dedup prior.
2. Reconstruction-style supervision is useful as a guarded auxiliary, but must
   stay a masked frozen-feature/object-explanation objective, not an
   action-time full-image decoder or posterior truth.

Strict distinction:

```text
Allowed:
  prototype / decoder output as measurement evidence, proposal, birth prior,
  dedup prior, training-only auxiliary, or diagnostic.

Rejected:
  prototype / decoder output as posterior truth, hard identity overwrite,
  dense-image objective on the PI0.5 action path, or a mandatory visual-only
  codebook for heterogeneous datasets.
```

## 1. Non-Negotiable PICF Contract

PICF is a sequential multimodal belief router, not a pure image slot
reconstruction model. Its authoritative state is the posterior object file:

```math
F_{t,j} =
(x_{t,j}, S_{t,j}, a_{t,j}, c_{t,j}, \alpha_{t,j}, r_j,
 \sigma_{t,j}, \phi_{t,j})
```

where `x,S` are geometry belief and uncertainty, `a,c` are address/content,
`alpha` is existence/support confidence, `r` is role, `sigma` is typed support,
and `phi` is the binding signature.

Every future prototype or reconstruction module must enter through one of
these legal channels:

```text
measurement evidence -> AQR support competition -> posterior update
birth/proposal prior -> file/birth competition -> posterior update
training-only aux -> budgeted loss -> no posterior overwrite
diagnostic/probe -> no gradient or guarded gradient
```

It must not bypass:

```text
support competition
owner transport
posterior file gates
innovation/recycle/birth handling
PI0.5 action-path authority
```

## 2. Roadmap A: Multimodal Prototype Bank

### 2.1 What MetaSlot Gives Us

MetaSlot's relevant principle is not "copy a visual VQ codebook into every
robotic posterior file". The useful principle is:

```text
fixed slot counts create duplicate/split-object failures;
prototype / aggregate / deduplicate mechanisms can stabilize adaptive object
cardinality.
```

PICF already implements the compatible part:

```text
duplicate demotion
slot-quality object/no-object/duplicate gates
active/context/reserve file capacity
birth competition
sidecar/proposal-conditioned initialization
object/background residual
```

The missing optional upgrade is a multimodal prototype initializer, not a hard
posterior VQ.

### 2.2 Why Hard Visual VQ Posterior Is Wrong Here

Hard visual VQ posterior:

```math
F_{t,j}.a \leftarrow P_{\arg\min_k \|z^v_{t,j}-P_k\|}
```

is unsafe for PICF because identity becomes tied to one visual prototype space.
This conflicts with the dataset-scaling goal:

```text
some datasets have RGB only
some have RGB-D
some have tactile
some have wrist
some have tracklets / sidecars
some are missing any subset of the above
```

If posterior identity requires a visual codebook, then missing-modality samples
cannot contribute cleanly to the same object-addressable belief state.

### 2.3 PICF-Compatible Multimodal Prototype Form

Each modality token is projected into a shared object-candidate space:

```math
z_i^{(m)} = \operatorname{norm}(W_m e_i^{(m)}),\quad
m \in \{\text{vjepa}, \text{pg}, \text{point}, \text{tactile},
\text{tracklet}, \text{sidecar}, \text{language}\}.
```

The prototype bank is:

```math
P = \{p_k\}_{k=1}^{K_p},\quad \|p_k\|_2=1.
```

Soft assignment is modality-masked:

```math
\pi_{i,k}^{(m)}
=
\operatorname{softmax}_k
\left(
\frac{(z_i^{(m)})^\top p_k + b_m + b_{\text{quality},i}}{\tau_p}
\right)
\cdot \mathbf{1}[\text{modality }m\text{ present}].
```

Prototype candidate summaries are:

```math
u_k =
\frac{
\sum_{m,i} w_i^{(m)} \pi_{i,k}^{(m)} W_u^{(m)} e_i^{(m)}
}{
\epsilon + \sum_{m,i} w_i^{(m)} \pi_{i,k}^{(m)}
}.
```

They may influence only:

```text
birth query initializer
proposal geometry prior
duplicate/no-object prior
address seed prior
support-signature seed prior
```

For example:

```math
q_j^0
=
q_{j,\text{base}}
+
\lambda_{\text{proto}}
\sum_k \beta_{j,k} W_q u_k
```

where `beta` is matched through role, sidecar/contact confidence, previous
posterior compatibility, and duplicate suppression.

The posterior update remains:

```math
F_t^+ =
\operatorname{PosteriorUpdate}
\left(
F_t^-,
\operatorname{AQR}(q^0, M_t),
\text{innovation},
\text{recycle/birth gates}
\right)
```

not:

```math
F_t^+ = \operatorname{VQ}(M_t).
```

### 2.4 Training Signal

The prototype bank should be trained with weak, multimodal, missing-modality
tolerant terms:

```math
L_{\text{proto}}
=
\lambda_{\text{same}}
L_{\text{same-object-probe}}
+
\lambda_{\text{dedup}}
L_{\text{prototype-dedup}}
+
\lambda_{\text{birth}}
L_{\text{birth-candidate}}.
```

Allowed weak positives:

```text
same tracklet segment
same sidecar/contact object mask
same high-confidence owner-transport object
same point cluster under contact/motion evidence
same language/task object candidate when confidence is high
```

Allowed weak negatives:

```text
different sidecar object ids
far point clusters
high-confidence different active posterior owners
tracklets with incompatible trajectories
```

No online weak IsSameObject loss should be enabled until these labels have
artifact-level validation; otherwise the model can learn its own bad pseudo
labels.

### 2.5 Acceptance Gates

Before enabling this in long training:

```text
candidate assignment max remains high
sidecar/tracklet/prototype candidates are nonzero when configured
active posterior owner sits on the task/contact object in overlays
active same-role overlap stays low
posterior owner confidence increases rather than raw support collapse
prototype usage entropy is non-degenerate
missing-modality batches still run as no-op rather than low-quality positives
```

## 3. Roadmap B: Masked Frozen-Feature/Object Explanation Aux

### 3.1 What We Already Have

Current `compute_transition_loss` already contains future-prediction style
auxiliaries:

```text
physical prediction cache:
  predicts future visual_latent, visual_real, tactile_real, point_real

semantic future cache:
  predicts the same targets from semantic-conditioned future context

slot-JEPA / support-prediction hooks:
  predict detached future posterior/support targets when explicitly enabled

object_explanation hooks:
  guarded object explanation components, controlled by explicit lambdas and
  acceptance gates
```

Code anchors:

```text
src/openpi/picf/core/training.py:
  visual_latent / visual_real / tactile_real / point_real losses
  semantic_future_aux
  slot_jepa / support_pred / binding_consistency
  slot_quality / vcap / object_explanation
```

This is the mechanism described as "using two modules to predict another". It
is real and useful, but it is not the same as a full per-slot object
reconstruction decoder.

### 3.2 Why Existing Future Aux Is Not Enough

Existing future aux answers:

```text
Can the current belief / semantic-conditioned cache predict future modality
summaries?
```

It does not necessarily answer:

```text
Does slot j explain object token i?
Does one object slot own one coherent mask/support region?
Are duplicate slots demoted because they explain the same object?
Can background be explained without stealing active object slots?
```

Therefore it can improve dynamics while still failing to force the orange
object slot onto the red block if assignment responsibility is wrong.

### 3.3 Correct Reconstruction-Aux Form

The PICF-compatible version is not full RGB reconstruction. It is masked
frozen-feature object explanation:

```math
\hat z_i^{(m)}
=
\sum_j r_{i,j}^{(m)} D_m(F_{t,j}) + r_{i,\text{bg}}^{(m)} D_m(F_{t,\text{bg}})
```

where:

```text
z_i^(m): frozen token feature from V-JEPA / PG / point / tactile / tracklet
r_ij^(m): slot-to-token responsibility from AQR support, sidecar mask, point
          cluster, tracklet, or contact-owner evidence
D_m: small modality-specific feature decoder
background: explicit residual/context explanation, not an object slot
```

The loss is:

```math
L_{\text{explain}}
=
\sum_{m,i}
w_i^{(m)}
\rho
\left(
\left\|
\operatorname{stopgrad}(z_i^{(m)})
-
\hat z_i^{(m)}
\right\|_2
\right)
+
\lambda_{\text{mask}} L_{\text{resp}}
+
\lambda_{\text{dup}} L_{\text{duplicate}}.
```

This should be:

```text
training-only
budgeted
guarded by candidate/tracklet/contact quality
off by default or low-weight until anchor closure is verified
never posterior truth
never a PI0.5 action-path replacement
never full-resolution RGB generation
```

### 3.4 Why Full RGB Reconstruction Decoder Is Still Wrong

A full RGB decoder would optimize:

```text
appearance reconstruction of table, wall, robot, lighting, and background
```

PICF needs:

```text
action-relevant object belief, support ownership, contact ownership, and
posterior continuity.
```

The dense RGB objective is not inherently bad as a research auxiliary, but as a
production loss it can dominate the belief router and pull slots toward
appearance/background explanations instead of task object files. If used at
all, it should first be reduced to frozen-feature/object-mask explanation, then
introduced with an explicit budget and acceptance gates.

## 4. Recommended Next Work Order

Do not do both upgrades blindly before the current A7/A5 behavior acceptance
finishes.

Recommended order:

```text
1. Finish current anchor/posterior owner closure diagnostics.
2. If active object owner still fails despite good sidecar/contact candidates,
   add multimodal prototype bank as birth/proposal/dedup prior.
3. If anchors bind but duplicate/background ownership remains unstable, add
   masked frozen-feature object explanation aux.
4. Only after both are stable consider stronger future predictive auxiliaries.
```

Do not enable:

```text
hard VQ posterior truth
full image reconstruction decoder as action-time objective
online IsSameObject loss from unvalidated pseudo labels
predictive slot losses before posterior identity stabilizes
```

## 5. Resume Checklist

When returning to this work:

```text
Read first:
  src/openpi/picf/README_v2.2.md
  docs/PICF_AQR_OWM_MULTIMODAL_PROTOTYPE_RECON_ROADMAP_20260519.md
  temp/audits_20260519/latest_slot_code_gap_matrix_20260519.md
  temp/audits_20260519/latest_slot_full_deployment_closure_20260519.md

Check current run type:
  scripts/picf_run_contract_audit.py

Check current behavior gates:
  active posterior owner overlay
  owner_transport_confidence / distance
  active same-role overlap
  proposal/tracklet token counts
  prototype or sidecar coverage if enabled
```

This file is a roadmap. It does not claim the two upgrades are already
implemented.

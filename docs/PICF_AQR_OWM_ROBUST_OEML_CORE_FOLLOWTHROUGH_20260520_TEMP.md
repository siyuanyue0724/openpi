# PICF-AQR-OWM Robust OEML Core Follow-Through

Date: 2026-05-20

Status: implemented locally; A7 300-step validation pending.

## Trigger

The A7 comprehensive frozen-policy validation after the context attention-bias
repair showed that context rows were no longer starved:

```text
aqr_context_downstream_weight_mean ~= 0.12
```

However, the object-explanation point compactness term became the dominant
unstable term:

```text
step 50:  loss_object_explanation_point ~= 1.72
step 100: loss_object_explanation_point ~= 10.77
step 200: loss_object_explanation_point ~= 20.27
step 250: loss_object_explanation_point ~= 34.27
```

This is not evidence that the dense V-JEPA/point memory is useless.  It means
the auxiliary compactness measurement was mathematically too brittle for
contact-motion sidecar masks.

## Mathematical Diagnosis

Before this repair, OEML computed point compactness over the full soft object
mask row:

```math
\mu_j =
\frac{\sum_i m_{ji} x_i}{\sum_i m_{ji}}
```

```math
L^{point}_j =
\frac{1}{\sigma^2}
\frac{\sum_i m_{ji}\lVert x_i-\mu_j\rVert^2}{\sum_i m_{ji}}
```

with a fixed small `sigma = 0.06m`.  This is correct only if `m_j` is already a
clean compact object mask.  The sidecar masks are weak contact/task evidence:
they can contain projected tails, short motion spread, gripper fragments, or
background leakage.  Treating those tails as hard compactness labels creates an
unbounded auxiliary:

```text
weak tail far away -> large variance -> large loss -> row moves to satisfy tail
```

That conflicts with the PICF contract.  A sidecar is typed measurement
evidence, not posterior truth.

## Paper-Code Boundary

The checked local references are:

```text
temp/external_repos/MetaSlot/object_centric_bench/model/metaslot.py
temp/paper_code_20260518/slotcontrast/slotcontrast/losses.py
temp/paper_code_20260518/slot-attention-video/savi/modules/attention.py
temp/external_repos/slot_refs_20260520/vit-object-binding/src/utils/models.py
temp/external_repos/slot_refs_20260520/vit-object-binding/src/utils/score.py
```

The common invariant across recent slot/OCL/object-binding methods is:

```text
slot assignment may use masks/support/probes as soft measurement evidence;
bad or unexplained pixels/tokens must be absorbed by a background/residual path;
auxiliary measurements should not become unbounded hard labels.
```

The Object-Binding NeurIPS 2025 code supports the existing
`binding_signature_proj`/quadratic binding subspace path.  It does not justify
turning noisy mask tails into hard geometric truth.

## Repair

### 1. Robust Point Core

OEML point compactness now computes geometry on the high-confidence core of
each object-mask row:

```math
C_j =
\operatorname{TopCore}(m_j; \rho, k)
```

where:

```text
rho = object_explanation_point_core_mass = 0.90
k   = object_explanation_point_core_topk = 128
```

The compactness center is:

```math
\mu_j^{core} =
\frac{\sum_{i\in C_j} m_{ji}x_i}{\sum_{i\in C_j}m_{ji}}
```

and the diagnostic compactness is:

```math
\hat L^{point}_j =
\frac{1}{\sigma^2}
\frac{\sum_{i\in C_j} m_{ji}\lVert x_i-\mu_j^{core}\rVert^2}
{\sum_{i\in C_j}m_{ji}}
```

Dense memory is not pruned.  Only this auxiliary compactness measurement uses
the core.

### 2. Bounded Point Loss

Training now clips the point compactness auxiliary:

```math
L^{point}_j = \min(\hat L^{point}_j, c)
```

with:

```text
c = object_explanation_point_loss_clip = 8.0
```

This keeps one noisy sidecar row from dominating `loss_total` while preserving
the sign of the compactness signal.

## Why This Is Not A Patch-On-Patch

The repair restores the intended probabilistic role:

```text
sidecar mask:
  weak object evidence

OEML:
  measurement quality / explanation consistency

posterior:
  authoritative belief state
```

It is not a new module, not a second object detector, not a SAM revival, and
not an action-path rewrite.  It only makes the existing OEML measurement obey
the same soft-assignment/background-residual contract as the rest of MVTrack.

## Acceptance Criteria

For the next A7 validation:

```text
loss_object_explanation_point:
  bounded and no longer monotonic explosive.

loss_total:
  should not show the step-150 -> step-250 regression driven by point tails.

aqr_context_downstream_weight_mean:
  remains near configured context weight, confirming context is not re-starved.

aqr_active_same_role_support_overlap_max:
  remains low; active posterior rows should not collapse.

aqr_same_role_support_overlap_max:
  raw reserve/context overlap may remain high and is not alone a blocker.

anchor overlays:
  task/object rows should move toward inspected sidecar cores without blue
  effector rows stealing the object.
```

If `loss_object_explanation_point` still exceeds about `8` by step 100, the
problem is no longer only mask-tail compactness; the next suspect is the
proposal-to-point bridge or candidate assignment row ownership.

## A7 300-Step Validation

Run:

```text
picf_a7_robust_oeml_core_300_20260520
```

Artifacts:

```text
/mnt/picf_run_logs/picf_a7_robust_oeml_core_300_20260520.log
/mnt/checkpoints/picf_core/picf_core/picf_a7_robust_oeml_core_300_20260520/metrics.jsonl
/mnt/checkpoints/picf_core/picf_core/picf_a7_robust_oeml_core_300_20260520/anchor_overlays
```

Configuration:

```text
world_size=2
num_steps=300
unroll=2
burnin=1 state_only
trainable_scope=all
large pretrained perception frozen
PaliGemma semantic frozen
action loss weight = 0
object_only role layout
context slots enabled
control graph reliability as attention bias
sidecar root = /mnt/picf_sidecars/contact_motion_mask_1000_20260518
```

Core metrics:

```text
step   total   point   obj_exp  obj_pull  active_ov  raw_ov  core_ov  grad_pre  clipped
50     2.198   0.622   0.019    6.211     0.023      0.457   0.351    0.990     no
100    1.628   1.787   0.043    4.514     0.042      0.510   0.558    2.900     no
150    2.624   2.386   0.054    7.329     0.009      0.914   0.882    1.001     no
200    1.064   3.647   0.080    2.796     0.034      0.999   0.973    1.849     no
250    2.276   3.611   0.078    6.266     0.017      0.967   0.884    9.808     yes
300    2.526   5.037   0.106    6.899     0.018      0.931   0.858    12.572    yes
```

Comparison against the failed context-quality validation:

```text
old point loss:
  step 50  ~= 1.72
  step 100 ~= 10.77
  step 200 ~= 20.27
  step 250 ~= 34.27

new point loss:
  step 50  = 0.62
  step 100 = 1.79
  step 200 = 3.65
  step 250 = 3.61
  step 300 = 5.04
```

Verdict:

```text
Fixed:
  The unbounded point-tail compactness failure is repaired.  The auxiliary no
  longer dominates the run by step 100/200, and it remains below the configured
  clip threshold through step 300.

Still not fully closed:
  The raw same-role support overlap and object-core overlap are still high in
  context/reserve rows.  This is not the old active-posterior collapse because
  active_same_role_support_overlap_max stays below 0.05, but it means context
  duplicate rows are still redundant.

Residual risk:
  Step 250 and step 300 trigger fixed grad clipping.  This is a controlled
  batch spike rather than the previous monotonic explosion, but long-run use
  should track whether point loss keeps drifting upward beyond step 300.

Overlay read:
  Step 200 and step 300 overlays show active/object rows near the inspected
  sidecar mask core.  They are not perfectly centered, and dense gray/context
  rows still cluster around the same task region.  This supports the metric
  interpretation: active owner routing is usable, while context dedup remains
  the next cleanup target.
```

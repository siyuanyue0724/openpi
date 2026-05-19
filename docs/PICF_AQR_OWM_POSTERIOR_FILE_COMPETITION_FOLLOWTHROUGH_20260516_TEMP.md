# PICF-AQR-OWM Posterior File Competition Follow-Through

Status: temporary audit note for the 2026-05-16 repair.

For the broader paper-code/math/dataflow comparison, see
[`PICF_AQR_OWM_FULL_BINDING_MATH_DATAFLOW_AUDIT_20260516_TEMP.md`](./PICF_AQR_OWM_FULL_BINDING_MATH_DATAFLOW_AUDIT_20260516_TEMP.md).

## Observed Failure

The A7 long run improved action loss but the anchor overlays at steps
`1600-1700` showed multiple role-1 posterior object files projected to the same
static-camera pixels. The metrics matched the image evidence:

```text
aqr_active_same_role_support_overlap_max ~= 0.04-0.10
aqr_same_role_support_overlap_max        ~= 1.00
posterior_active_file_potential_swap_rate ~= 0.24-0.32
```

The important distinction is source and shape:

```text
graph squares:
  AQR candidate owners; inactive candidates are reserve capacity.

posterior circles:
  persistent object files. Duplicate role-1 posterior circles are real
  object-file duplication, not just reserve visualization.
```

At `step_001600`, seven role-1 posterior files were visible and the minimum
same-role posterior pixel distance was `0.4 px`. That is an object-file binding
failure.

## Dataflow Root Cause

The posterior update path before this repair was:

```text
bind_logits[persistent_file, observation_anchor]
  -> _sinkhorn_dustbin(bind_logits)
  -> support_raw = binding_raw[:-1]
  -> lifecycle calibration
  -> support_mass
  -> x_obs = support_raw @ obs_anchors.x / support_mass
  -> posterior file x/S/address/content update
```

The previous transport had an observation dustbin row but no file/no-object
column:

```math
P = Sinkhorn([L; d_obs])
```

with uniform row marginals. This makes observation anchors compete for columns,
but it also forces every persistent file row to receive nonzero measurement
mass. If a CALVIN scene contains fewer stable object files than capacity, or if
several files have similar logits for the same owner, the transport can still
update multiple files toward the same object.

This is the mathematical reason that action pressure could reduce action loss
while posterior object files duplicated around the same high-value contact
region.

## Paper/Code Cross-Check

The pulled object-binding probe code (`/tmp/vit-object-binding`) computes a
pairwise same-object score:

```math
s(i,j) = probe(z_i, z_j)
```

and uses that as affinity evidence. This supports PICF's
`binding_signature_proj`/quadratic binding subspace, but it does not by itself
solve object-file capacity. Pairwise affinity says which evidence is same
object; it does not decide how many persistent files are allowed to update.

The pulled SlotContrast/Slot Attention code performs slot competition over the
slot dimension:

```math
a_{s,f} = softmax_s(q_s^T k_f)
```

followed by normalization over features. Its decoder similarly uses a
slot-wise mask softmax. This is an explaining-away mechanism: a feature should
not be explained independently by every slot.

DETR-style set prediction and the object-binding instance loss include a
`no_object`/background class. That is the missing posterior side of the PICF
transport: unused persistent files must be allowed to become no-object instead
of being forced to explain some observation anchor.

## Repair

The repair is not an extra auxiliary loss. It is a posterior binding
calibration layer:

```text
support_raw, dustbin_raw
  -> posterior_file_competition
  -> support_raw_active, dustbin_raw + demoted_support
  -> lifecycle calibration
  -> posterior write
```

For same-role files `i,j`, compute duplicate evidence from support overlap and
optionally geometry:

```math
O_{ij}^{support}
=
\frac{p_i^T p_j}{\|p_i\|\|p_j\|}

O_{ij}^{geom}
=
\exp(-\|x_i-x_j\|^2 / 2\sigma^2)
```

where `p_i = support_raw_i / sum(support_raw_i)`.

Files are ranked by:

```math
score_i
=
support\_mass_i
(0.25 + 0.75\alpha_i)
(0.25 + 0.75 margin_i)
```

Within each role, high-score files are kept if they are not duplicates of an
already kept file. Duplicate or low-support files are demoted:

```math
support'_i = active_i support_i
```

and the removed mass is conserved as observation dustbin/no-object mass:

```math
dustbin' = dustbin + \sum_i (1-active_i)support_i
```

This preserves the posterior authority principle:

```text
current evidence still updates files;
duplicate files do not receive the same evidence twice;
demoted evidence is not discarded, it becomes no-object/dustbin evidence.
```

## Why This Is Not A Patchy Loss

The repair changes the assignment model, not the optimizer objective. It is
equivalent to adding the missing no-object decision for persistent files after
pairwise/object-core evidence has been computed.

It is aligned with:

```text
Slot Attention:
  feature explaining-away across slots.

object-binding probes:
  pairwise same-object affinity is evidence, not a scalar object label.

DETR/no-object set prediction:
  fixed capacity requires an explicit no-object route.
```

## Expected Acceptance Signals

The next diagnostic should show:

```text
posterior_file_competition_active_count materially below persistent slot count
posterior_file_competition_demoted_mass_mean nonzero on duplicate scenes
posterior_active_file_potential_swap_rate lower than the A7 long run
posterior role-1 overlay circles no longer duplicated at sub-pixel distance
action loss remains comparable to 4-22 / full-PICF baselines
```

This does not claim ordinal/fourth-object grounding is solved. It only fixes the
specific posterior capacity/binding bug exposed by the A7 overlays.

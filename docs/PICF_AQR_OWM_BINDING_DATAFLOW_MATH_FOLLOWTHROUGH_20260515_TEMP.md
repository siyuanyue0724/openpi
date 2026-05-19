# PICF-AQR-OWM Binding Dataflow And Math Follow-Through

Date: 2026-05-15

Status:

```text
Purpose:
  Re-audit the binding subsystem from external paper equations to PICF runtime
  dataflow and current A7 diagnostic evidence.

Boundary:
  This document validates code-level and math-level follow-through. It does not
  claim fresh CALVIN behavior acceptance.
```

## 1. External Paper Code Objects

Inspected code snapshot:

```text
/tmp/vit-object-binding/src/utils/models.py
/tmp/vit-object-binding/src/trainer.py
/tmp/vit-object-binding/src/utils/score.py
```

The paper code evaluates `IsSameObject(x, y)` with pairwise probe families:

```math
DiagonalQuadraticProbe:
  l(x,y)=w^T(x \odot y)+b

QuadraticProbe:
  l(x,y)=x^T W_{sym} y + b,
  W_{sym}=(W+W^T)/(2\sqrt{d})

QuadraticFixedRankProbe:
  l(x,y)=x^T (W_1^T W_2 + W_2^T W_1)y/(2\sqrt{d})+b
```

Training in the paper code uses true instance masks:

```math
y_{ij}=1[\text{instance}(i)=\text{instance}(j)]
```

and:

```math
L=\operatorname{BCEWithLogits}(l_{ij}, y_{ij})
```

Critical implication:

```text
The paper's pairwise score is a calibrated logit because it is trained with
mask labels. PICF does not have mask labels in CALVIN training. Therefore PICF
must not treat raw cosine/quadratic common-mode values as calibrated identity
logits.
```

## 2. PICF Runtime Dataflow

PICF binding dataflow is:

```text
observation
  -> typed memory
     PG image/text, V-JEPA visual/temporal, point, tactile,
     optional tracklet/proposal, previous posterior/cache

typed priors
  -> AQR graph priors
     pg_priors, visual_priors, temporal_priors, point_priors,
     tactile_priors, tracklet_priors, proposal_priors

support priors + typed tokens
  -> _support_binding_signature(weights, tokens)
     normalized support row times centered projected binding keys

observation anchors
  -> PicfObservationAnchorState.binding_signature

previous posterior
  -> PicfPosteriorAnchorState.binding_signature

_binding_logits(prev, obs)
  -> hidden similarity
  -> geometry Mahalanobis
  -> support-signature overlap
  -> calibrated pairwise binding-signature score
  -> gated address score
  -> role/owner-active/dustbin biases

posterior assignment
  -> new posterior binding matrix
  -> posterior.binding_signature = assignment @ obs.binding_signature
```

The runtime code paths are:

```text
src/openpi/picf/core/pipeline.py
  _binding_keys
  _support_binding_signature
  _binding_signature_quadratic_scores
  _calibrate_pairwise_binding_score
  _binding_logits
  _posterior_update

src/openpi/picf/core/contracts.py
  PicfPosteriorAnchorState.binding_signature
  PicfPosteriorAnchorState.binding_signature_calibrated_*

scripts/picf_core_train.py
  OWM_DEBUG_METRIC_KEYS
  loss_action_default_equiv
  anchor overlay signature dump
```

## 3. PICF Math Versus Paper Math

PICF computes:

```math
S_{raw}
=
w_e \cos(e^-_j,e_i)
+w_d(e^-_j)^TD e_i
+w_r(L e^-_j)^T(R e_i)
```

where `e^-_j` is a previous posterior object-file binding signature and `e_i`
is an observation-anchor binding signature.

Because PICF has no mask-label BCE calibration, raw `S_raw` is converted into a
relative assignment logit:

```math
S_c
=
S_{raw}
-\operatorname{rowmean}(S_{raw})
-\operatorname{colmean}(S_{raw})
+\operatorname{mean}(S_{raw})
```

```math
S_{cal}
=
\begin{cases}
0, & \operatorname{std}(S_c)<\sigma_{\min}\\
\operatorname{clip}(S_c/\operatorname{std}(S_c),-c,c), & \text{otherwise}
\end{cases}
```

Posterior binding receives:

```math
L_{bind}
\leftarrow
L_{hidden}
-L_{geom}
+g_{\alpha,recycle,innovation}S_{cal}
+g_{\alpha,recycle,innovation}S_{support}
+g_{\alpha,recycle,innovation}S_{address}
+B_{role}
+B_{owner}
```

This is mathematically different from adding a new loss. It changes the
assignment energy inside the belief filter.

## 4. Why Double-Centering Is Required

If:

```math
S_{raw}(j,i)=a_j+b_i+c
```

then `S_raw` contains row saliency and column saliency but no pair identity. A
softmax assignment can still be distorted by it. Double-centering removes this:

```math
S_c(j,i)=0
```

If:

```math
S_{raw}(j,i)=a_j+b_i+\delta 1[j=i]
```

then double-centering preserves the relative diagonal structure. Therefore the
calibration rejects "all anchors like the same salient thing" while preserving
"this object file matches this observation anchor".

## 5. Current A7 Diagnostic Evidence

Run:

```text
picf_a7_diag_binding_logit_calibrated_u2b1_180_20260515
```

Log:

```text
/mnt/picf_run_logs/picf_a7_diag_binding_logit_calibrated_u2b1_180_20260515.log
```

Step100:

```text
loss_action_default_equiv                         0.0809
aqr_same_role_support_overlap_max                 0.1831
aqr_active_same_role_support_overlap_max          0.0359
aqr_same_role_object_core_overlap_max             0.2400
aqr_active_same_role_object_core_overlap_max      0.1419
posterior_recycle_rate                            0.1087
posterior_identity_switch_rate                    0.6994
posterior_identity_switch_rate_stable             0.6119
posterior_binding_signature_low_rank_score_abs    0.0172
posterior_binding_signature_calibrated_score_std  0.3644
posterior_binding_signature_calibrated_margin     0.2423
```

Interpretation:

```text
1. The old common-mode failure is materially improved. The non-calibrated r2
   run had raw support overlap 0.6141 by step100; calibrated run is 0.1831.
2. Active same-role support overlap is healthy at step100.
3. Low-rank score is no longer numerically dead.
4. Posterior identity continuity is not yet proven: identity_switch remains
   high. The next audit must determine whether this is metric strictness under
   active-slot filtering or real object-file permutation.
```

## 6. Remaining Non-Negotiable Limits

The current repair does not solve:

```text
tracklet/proposal evidence absence:
  dataflow is wired but current CALVIN rows report owm_tracklet_tokens=0 and
  owm_proposal_tokens=0.

ordinal/fourth-object grounding:
  still weak diagnostic without rank-supervised labels.

online IsSameObject supervision:
  not added because weak pseudo-labels without mask/tracklet ground truth would
  be self-confirming.

posterior object-file continuity:
  still needs step150/180 and possibly an active-slot-aware identity-switch
  audit.
```

## 7. Executable Audit

New script:

```text
scripts/picf_binding_dataflow_math_audit.py
```

It checks:

```text
paper code exposes quadratic pairwise probe families;
paper trainer uses BCEWithLogits over instance-mask IsSameObject labels;
PICF exposes calibration config;
PICF builds support-weighted binding signatures;
PICF natively reimplements diagonal/low-rank quadratic scoring;
PICF calibrates raw scores before posterior binding;
PICF gates binding evidence by posterior trust;
observation signatures reach posterior object files;
train/replay/serve optional tracklet/proposal fields are threaded;
docs/metrics expose both repair and remaining limits;
common-mode, row/column bias, relative-pair, and low-dispersion math tests pass.
```

Result:

```text
PYTHONPATH=src python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
  PASS 16/16
```

## 8. Decision

Current step150 evidence:

```text
step50:
  active support overlap 0.0117, active object-core overlap 0.0373,
  calibrated std 0.0551, calibrated margin 0.0554

step100:
  active support overlap 0.0359, active object-core overlap 0.1419,
  calibrated std 0.3644, calibrated margin 0.2423

step150:
  raw support/object-core overlap 0.8343/0.7405,
  active support/object-core overlap 0.0663/0.0562,
  calibrated std 0.0000, calibrated margin 0.0000,
  identity_switch 0.7494

step180:
  raw support/object-core overlap 0.9784/0.8357,
  active support/object-core overlap 0.0984/0.0418,
  calibrated std 0.0000, calibrated margin 0.0000,
  identity_switch 0.7611
```

Mathematical reading:

```text
The active-owner layer is still separating selected object owners at step150.
The raw overlap rebound is mostly reserve/inactive overlap or candidate reuse.

The calibrated pairwise binding term correctly shuts off when the
double-centered score matrix has insufficient dispersion. This is not a
failure of the guard; it is evidence that this batch/window does not expose a
trustworthy relative IsSameObject logit.

The unsolved problem is posterior object-file continuity. Step180 strengthens
this conclusion: active owners remain separated while raw/reserve overlap and
identity-switch remain high. The next audit should measure identity switches
over active owners and binding-eligible object files instead of treating every
reserve/dustbin object file as an equally meaningful object.
```

Current decision:

```text
Do not launch a new architecture variant until active-owner identity continuity
is audited. The 180-step run is sufficient to reject "binding score was dead or
common-mode" and to expose the remaining object-file continuity ambiguity.
```

Next decision:

```text
If active-owner identity continuity is healthy:
  raw overlap is mainly reserve/dustbin bookkeeping, and the next long run can
  focus on action/acceptance metrics.

If active-owner identity continuity is unhealthy:
  the posterior object-file update, not AQR support separation, is the remaining
  target. Fix binding eligibility / file update before long training.

If calibrated score is often zero on windows with real active-owner switches:
  run an offline IsSameObject probe on those exact overlays to decide whether
  the representation lacks separable evidence or the calibration threshold is
  too conservative.
```

## 9. Posterior file continuity audit update

The previous identity-switch metric compared this-step and previous-step
argmax observation-anchor row ids:

```math
\operatorname{switch}_j =
1[\arg\max_i B_{t,j,i} \ne \arg\max_i B_{t-1,j,i}]
```

This is useful as a churn diagnostic, but it is not a stable object-identity
metric. Observation anchor rows are reconstructed every step, so row id `i` is
not an object id. A high value can therefore mean:

```text
true file/object swap,
or simply re-indexed observation anchors,
or inactive/reserve capacity choosing different dustbin-adjacent rows.
```

The runtime now also measures posterior object-file continuity directly. For
each persistent file `j`, let `s_tj` be its normalized posterior
`binding_signature`. The metric compares each file to its own previous content
and to same-role competing files:

```math
C_{j,k} = s_{t,j}^{\top}s_{t-1,k}
```

```math
m_j =
C_{j,j} - \max_{k \ne j,\ r_k=r_j} C_{j,k}
```

The active-file subset requires:

```math
\alpha_{t,j},\alpha_{t-1,j}\ge 0.25,\quad
support_{t,j},support_{t-1,j}\ge 0.05,\quad
recycle_{t,j},recycle_{t-1,j}\le 0.5,\quad
owner_{t,j},owner_{t-1,j}\ge owner\_min.
```

New metrics:

```text
posterior_file_self_signature_sim_mean
posterior_file_best_other_signature_margin_mean
posterior_file_potential_swap_rate
posterior_active_file_fraction
posterior_active_file_self_signature_sim_mean
posterior_active_file_best_other_signature_margin_mean
posterior_active_file_potential_swap_rate
```

Decision rule:

```text
If old row-id switch is high but active-file potential swap is low:
  do not add another loss; the old switch metric was over-reporting identity
  churn.

If active-file margin is low or active-file potential swap is high:
  the posterior file update is the real target. Then fix assignment/file
  continuity before opening predictive losses.
```

This keeps the repair mathematically local: it does not alter action, cache,
or AQR support routing. It only prevents a false-positive diagnostic from
driving architecture changes.

## 10. Step220 audit conclusion: what is missing and what is not

The 2026-05-15 A7 run
`picf_a7_posterior_file_continuity_u2b1_220_20260515` provides a useful
separation between three variables that were previously conflated:

```text
raw candidate-anchor overlap
active owner overlap
posterior object-file continuity
```

At step220:

```text
raw same-role support overlap max        = 0.9207
active same-role support overlap max     = 0.1694
posterior active-file self similarity    = 0.8404
posterior active-file same-role margin   = 0.2459
posterior active-file potential swap     = 0.1307
old row-id posterior switch              = 0.7319
```

Therefore the high raw overlap is real, but it is mainly a reserve/candidate
pool phenomenon. The active object files are not showing the same level of
collapse. The old row-id switch metric is over-reporting identity churn because
it compares regenerated observation-anchor row ids rather than persistent
posterior files.

The calibrated same-object score is also intentionally zero at the last rows:

```text
posterior_binding_signature_calibrated_score_std          = 0.0
posterior_binding_signature_calibrated_top1_margin_mean   = 0.0
```

This is not a failure of the guard. It means the current pairwise score matrix
does not contain enough relative dispersion after centering/z-scoring. The
object-binding probe literature uses supervised IsSameObject labels to train
and calibrate pairwise logits. PICF does not have mask/object labels in this
run, so the correct online behavior is:

```text
use the same-object subspace only when it has relative evidence;
otherwise emit zero binding evidence rather than hallucinating identity.
```

### Remaining gap inventory

```text
1. Online pairwise binding subspace:
   present, calibrated, guarded; not always informative.

2. Offline IsSameObject probe:
   still required for scientific validation. It needs weak labels from
   tracklets/point neighborhoods/contact/posterior overlays or true masks.
   It should not be converted into an online loss without reliable labels.

3. Tracklet/proposal memory:
   runtime dataflow-valid, but inactive for this CALVIN run
   (owm_tracklet_tokens=0, owm_proposal_tokens=0).

4. Raw reserve overlap:
   still high. Treat as reserve-pool redundancy/capacity management, not as
   proof that active object files collapsed.

5. Predictive losses:
   remain guarded. Do not open slot-JEPA/support-pred/binding-consistency
   because this run tests identity/dataflow, not future latent supervision.
```

### Next decision rule

```text
If active-file potential swap remains <= 0.15 and active support overlap
remains < 0.25 over a longer run:
  continue controlled long training with these diagnostics.

If active-file potential swap rises > 0.25:
  fix posterior file update / binding eligibility.

If active support overlap rises > 0.50:
  fix active-owner selection, not raw reserve diversity.

If raw reserve overlap remains high while active metrics stay healthy:
  archive it as reserve redundancy and avoid adding new losses.
```

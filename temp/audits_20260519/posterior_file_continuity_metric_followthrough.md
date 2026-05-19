# Posterior File Continuity Metric Follow-Through

Date: 2026-05-19

Scope: tighten the remaining V6 diagnostic boundary before another model
change.  The V6 run fixed the object-PV target mismatch, but the run diagnostic
still reported high `posterior_active_file_potential_swap_rate`.  This note
checks whether that number is a real posterior-file failure or a raw-metric
artifact.

## 1. Mathematical Issue

Runtime PICF binding already treats object-binding paper probes as relative
assignment evidence, not raw identity truth.  The score matrix

```math
S_{ij} = \phi(f^+_i)^T \phi(f^-_j)
```

is calibrated as:

```math
\tilde S_{ij}
=
S_{ij}
- \frac{1}{J}\sum_k S_{ik}
- \frac{1}{I}\sum_k S_{kj}
+ \frac{1}{IJ}\sum_{k,l} S_{kl}
```

then z-scored only if the matrix has enough dispersion.  If the whole matrix is
common-mode, `\tilde S = 0` and the pairwise subspace is not allowed to decide
identity.

The old file-continuity debug path did not follow this.  It used raw normalized
cosine:

```math
swap_i = 1[\max_{j\ne i} S_{ij} > S_{ii} + 0.05]
```

This can be false-positive when all active files share the same task/background
signature.  The model is not swapping files; the metric is measuring common
appearance.

## 2. Corrected Runtime Contract

Keep raw metrics for compatibility:

```text
posterior_file_potential_swap_rate
posterior_active_file_potential_swap_rate
```

Add calibrated metrics that match the runtime binding semantics:

```text
posterior_file_calibrated_self_signature_sim_mean
posterior_file_calibrated_best_other_signature_margin_mean
posterior_file_calibrated_potential_swap_rate
posterior_file_calibrated_signature_score_std
posterior_active_file_calibrated_self_signature_sim_mean
posterior_active_file_calibrated_best_other_signature_margin_mean
posterior_active_file_calibrated_potential_swap_rate
```

Decision rule:

```text
If raw swap is high but calibrated std is zero or calibrated swap is low:
  do not change architecture; inspect overlays and treat raw swap as common-mode.

If calibrated active-file swap is high:
  the remaining issue is real posterior object-file continuity.
```

## 3. Paper-Code Link

Object-binding probe papers use pairwise scores to test separability, but they
do not say raw cosine is a stable object id.  Runtime PICF therefore uses the
same principle as the calibrated IsSameObject probe: only relative pairwise
dispersion should contribute identity evidence.

This preserves the PICF belief-filter invariant:

```text
current evidence and posterior file competition remain authoritative;
raw common-mode embedding similarity is never treated as object identity.
```

## 4. Code Follow-Through

Updated files:

```text
src/openpi/picf/core/pipeline.py
scripts/picf_core_train.py
scripts/picf_anchor_run_diagnostic_report.py
scripts/picf_owm_evidence_bundle.py
scripts/picf_binding_dataflow_math_audit.py
scripts/verify_picf_owm_contract.py
scripts/picf_posterior_file_continuity_metric_audit.py
src/openpi/picf/README_v2.2.md
```

Validation commands:

```bash
PYTHONPATH=src:scripts uv run python scripts/picf_posterior_file_continuity_metric_audit.py
PYTHONPATH=src:scripts uv run python scripts/picf_binding_dataflow_math_audit.py
PYTHONPATH=src uv run python scripts/verify_picf_owm_contract.py
```

## 5. Remaining Runtime Question

This patch is deliberately diagnostic-first.  It does not add another loss and
does not change AQR/posterior update behavior.  If the calibrated metric is
healthy on the next 200-step run, the V6 model fix stands.  If the calibrated
metric is high, the next model-level fix should target posterior file update
continuity, not object-PV, support overlap, or sidecar proposal generation.

## 6. A7 V7 200-Step Result

Run:

```text
picf_a7_v7_calibrated_file_continuity_anchor200_20260519
```

Profile:

```text
anchor_only
world_size=2
unroll_steps=2
burnin_steps=1
action loss weight = 0
PaliGemma / Sonata / V-JEPA / AnyTouch frozen
sidecar proposals and tracklets enabled
legacy local refinement disabled
proposal reference-anchor seeding disabled
```

Trend:

```text
step                                      50       100      150      200
loss_total                               0.4761   0.4214   0.4217   0.3997
loss_alignment                           0.4011   0.3464   --       0.3247
loss_anchor_pv                           0.5836   0.6100   --       0.6304
loss_pv_weak                             6.0819   4.3985   --       3.3801
aqr_active_same_role_support_overlap_max 0.0266   0.0752   0.3084   0.1936
aqr_active_same_role_object_core_overlap 0.1508   0.2363   0.0425   0.0678
aqr_same_role_support_overlap_max        0.1413   0.5561   0.9972   0.9859
aqr_same_role_object_core_overlap_max    0.1620   0.5227   0.8912   0.8531
posterior_active_file_calibrated_swap    0.3082   0.3062   0.2482   0.3157
posterior_file_calibrated_score_std      0.9632   0.9992   0.9798   0.9944
posterior_binding_calibrated_score_std   0.0385   0.0050   0.0000   0.0000
posterior_recycle_rate                   0.0391   0.0005   0.0064   0.0031
```

Remote diagnostic report:

```text
raw object-core overlap is high while active-owner object-core overlap is low;
active owners are still separated in geometry/support space.

calibrated pairwise binding evidence is off for the last row; the score matrix
did not have enough relative dispersion to trust as identity evidence.

posterior active-file calibrated potential swap is high; this is a real
object-file continuity target, not only raw common-mode signature overlap.
```

Interpretation:

```text
1. V6 object-PV target repair remains valid:
   loss_total decreases and loss_pv_weak decreases; active object-core overlap
   is low by step 150/200.

2. Raw support/object overlap remains mostly reserve/context duplication:
   raw same-role overlap reaches ~0.99 while active object-core overlap remains
   <= ~0.07 at step 200.

3. The remaining issue is not a raw-metric false positive:
   calibrated active-file swap stays around 0.25-0.32 with nonzero calibrated
   file matrix dispersion.

4. The object-binding subspace itself is too weak in posterior update:
   posterior_binding_signature_calibrated_score_std falls to 0 after step 150,
   so the online binding signature stops providing useful relative assignment
   evidence, while file-level calibrated continuity still detects cross-file
   risk.
```

Conclusion:

```text
ACCEPTED:
  diagnostic correction is valid;
  object-PV / active support root cause is no longer the dominant failure.

NOT ACCEPTED:
  posterior object-file continuity is not fully solved.

NEXT MODEL TARGET:
  add an explicit posterior file-continuity update path that uses stable owner
  history / assignment continuity under current evidence gates, rather than
  relying only on instantaneous binding signatures.
```

## V8b repair: posterior file continuity via dispersion-gated binding signature memory

The V7 failure is not a reason to add another auxiliary loss. The failure is in
the belief-state update equation:

```math
s_t = \operatorname{normalize}(B_t \hat{s}_t)
```

where `s_t` is the posterior file binding signature and `B_t` is the current
posterior binding. This overwrites the persistent file descriptor with the
current measurement summary every step. If the current typed-memory signatures
have low calibrated pairwise dispersion, the file descriptor can lose relative
identity even when active owners are separated.

The corrected posterior update is:

```math
\hat{s}_{t,j}
=
\operatorname{normalize}\left(\sum_i B_{t,j,i} e_{t,i}^{bind}\right)
```

The first V8 idea was only an EMA. That is not sufficient if the current
measurement is common-mode: a slow update still pollutes the file descriptor.
V8b therefore adds a measurement-dispersion gate before the EMA:

```math
C_t =
\operatorname{Calibrate}\left(\hat{S}_t\hat{S}_t^\top\right)
```

```math
g^{disp}_{t,j}
=
\mathbf{1}[\operatorname{std}(C_t)\ge \sigma_{\min}]
\cdot
\sigma\left(
  \frac{C_{t,j,j}-\max_{k\ne j, r_k=r_j} C_{t,j,k}-\delta_{\min}}{\tau}
\right)
```

```math
\rho_{t,j}
=
\max\left(
  \rho_0\,
  \operatorname{assignConf}_{t,j}\,
  \operatorname{ownerRel}_{t,j}\,
  g^{disp}_{t,j}\,
  \mathbf{1}[m_{t,j}\ge m_{\min}]\,
  \operatorname{active}_{t,j}\,
  (1-r_{t,j}),
  \operatorname{birth/recycle}_{t,j}
\right)
```

```math
s_{t,j}
=
\operatorname{normalize}\left(
  (1-\rho_{t,j})s_{t-1,j}
  +
  \rho_{t,j}\hat{s}_{t,j}
\right)
```

This is a latent-state memory update:

```text
low support / low owner confidence:
  keep previous file identity descriptor.

trusted current measurement:
  move slowly toward current evidence only when pairwise identity evidence has
  relative dispersion.

common-mode current measurement:
  keep previous file identity descriptor.

birth or recycle:
  reset to the current instantaneous descriptor.
```

It is consistent with the object-binding paper code boundary: projected
pairwise signatures are relative identity evidence, but runtime PICF has no
mask labels, so identity should be a filtered state, not a hard per-frame
classification target. The implementation exposes:

```text
posterior_binding_signature_memory_enabled
posterior_binding_signature_update_rate
posterior_binding_signature_update_max_rate
posterior_binding_signature_min_support
posterior_binding_signature_owner_weight
posterior_binding_signature_dispersion_gate_enabled
posterior_binding_signature_measurement_min_std
posterior_binding_signature_measurement_margin_min
posterior_binding_signature_measurement_margin_temperature
posterior_binding_signature_update_rate_mean
posterior_binding_signature_measurement_trust_mean
posterior_binding_signature_memory_keep_rate_mean
posterior_binding_signature_measurement_score_std
posterior_binding_signature_measurement_margin_mean
posterior_binding_signature_measurement_dispersion_gate_mean
```

Acceptance target for the next 200-step validation:

```text
posterior_binding_signature_calibrated_score_std should not collapse to 0;
if current measurements are common-mode, posterior_binding_signature_update_rate_mean
should be low and memory_keep_rate should be high;
posterior_active_file_calibrated_potential_swap_rate should fall below V7;
active support/object-core overlap should remain low;
loss_total / loss_pv_weak should keep decreasing.
```
## 2026-05-19 V8b posterior file-continuity validation note

Run: `picf_a7_v8b_binding_signature_dispersion_memory_anchor200_20260519`

Purpose: validate the posterior binding-signature memory fix with a calibrated measurement-dispersion gate.

Mathematical target:

```math
\hat{s}_{t,j}=normalize(\sum_i B_{t,j,i}e^{bind}_{t,i})
```

```math
g^{disp}_{t,j}=1[std(C_t)>=\sigma_{min}]\,\sigma((C_{jj}-\max_{k\ne j,r_k=r_j}C_{jk}-\delta_{min})/\tau)
```

```math
s_{t,j}=normalize((1-\rho_{t,j})s_{t-1,j}+\rho_{t,j}\hat{s}_{t,j})
```

where `rho` includes assignment confidence, owner reliability, support mass, active gate, recycle/birth reset, and `g_disp`.

Step 50:
- loss_total = 0.4716
- loss_pv_weak = 5.9688
- loss_tactile_aux = 36.6283
- active same-role support overlap = 0.0176
- active calibrated potential swap = 0.0261
- binding signature measurement std = 0.9715
- binding signature dispersion gate = 0.6127
- binding signature update rate = 0.2279
- binding signature memory keep = 0.7721

Step 100:
- loss_total = 0.4271, improving from step 50.
- loss_pv_weak = 4.3072, improving from step 50.
- loss_tactile_aux = 14.1854, improving sharply.
- loss_physical_aux = 1.6761, improving sharply.
- active same-role support overlap = 0.1320, still low.
- active object-core overlap = 0.2023, acceptable but higher than step 50.
- raw same-role support overlap = 0.4099, higher; mostly reserve/context/raw rows.
- active calibrated potential swap = 0.0245, still low.
- all-file calibrated potential swap = 0.0894, improved vs step 50.
- posterior binding signature calibrated std = 0.0194, low for current posterior state.
- measurement std = 0.9836, current measurement is dispersed.
- dispersion gate = 0.4363, more conservative than step 50.
- update rate = 0.0737, memory keep = 0.9263; V8b is no longer blindly overwriting signatures.

Overlay step 100:
- Prompt: `pick up the red block lying in the drawer`.
- Sidecar proposal box covers the red-block region.
- Several active graph/physical anchors and one posterior active file land inside the proposal box.
- Several active posterior/task anchors remain outside; task anchors cluster above the drawer and should not be interpreted as physical files.

Interim interpretation:
- V8b fixes the specific blind-overwrite posterior file-signature failure.
- The old active-overlap collapse is not present at 100 steps.
- The remaining issue is incomplete physical posterior file cleanliness: only part of active posterior mass is on the task object. Need step 200 and overlays before declaring closure.

Step 150:
- loss_total = 0.4206, still below step 100.
- loss_pv_weak = 3.5097, continuing to improve.
- loss_anchor_pv = 0.6142, worse than step 50/100.
- loss_mapg_routing = 0.7515, worse than step 50/100.
- active same-role support overlap = 0.3001, rising but not catastrophic.
- active object-core overlap = 0.0307, excellent.
- raw same-role support overlap = 0.9913, high again in context/reserve/raw rows.
- active calibrated potential swap = 0.0056, improved.
- all-file calibrated potential swap = 0.0569, improved.
- measurement std = 0.9754, so current measurements are still pairwise-dispersed.
- dispersion gate = 0.3546, update rate = 0.0590, memory keep = 0.9410.
- posterior binding signature calibrated std = 0.0000, meaning the current posterior signature state has collapsed to common-mode under the calibrated pairwise audit even though the incoming measurement remains dispersed.

Step 200:
- loss_total = 0.4266, slightly worse than step 150 but still better than step 50.
- loss_pv_weak = 3.2546, continuing to improve.
- loss_tactile_aux = 8.5423 and loss_physical_aux = 1.1068, both strongly improved.
- loss_anchor_pv = 0.6171, monotonically worse than step 50.
- loss_mapg_routing = 0.7760, monotonically worse than step 50.
- active same-role support overlap = 0.3777, worse than step 100/150 but far below the old near-1.0 active collapse.
- active object-core overlap = 0.0338, still excellent.
- raw same-role support overlap = 0.9918 and raw object-core overlap = 0.8450, high in context/reserve/raw capacity.
- active calibrated potential swap = 0.0085, still very low.
- all-file calibrated potential swap = 0.0538, improved vs step 50/100.
- measurement std = 0.9715, measurement dispersion remains healthy.
- dispersion gate = 0.3276, update rate = 0.0538, memory keep = 0.9462.  V8b is conservatively preserving file state rather than blindly overwriting.
- posterior binding signature calibrated std = 0.0000, while posterior file calibrated signature std = 0.4649.  This means the V8b memory guard prevents destructive overwrites, but the persisted posterior signature subspace itself is not yet becoming strongly discriminative.
- posterior recycle rate = 0.0003, so the run is not failing through excessive recycling.
- preclip grad norm = 62.57, clipped.  This is a real stability warning and should be watched before any long run.

Step 200 overlay interpretation:
- The sidecar proposal box covers the red block region.
- The red block region contains several active anchors; this is no longer the earlier zero-anchor-on-object failure.
- Several orange same-role active anchors remain clustered near the gripper/drawer side, not cleanly split across object files.
- Gray context/reserve anchors remain numerous in the full view.  They explain why raw overlap metrics are high, but they should not be confused with active object-file success.
- The overlay and metrics agree: object evidence is reaching the model, active object-core separation is good, but physical posterior file assignment is still incomplete.

Final 200-step conclusion:

```text
V8b passes the narrow posterior-file continuity target:
  blind binding-signature overwrite is fixed;
  active calibrated potential swap is low;
  all-file calibrated swap improves;
  recycle is low;
  current measurements remain dispersed and are gated before memory update.

V8b does not close the whole anchor-binding problem:
  loss_anchor_pv rises;
  loss_mapg_routing rises;
  active support overlap rises after step 100;
  raw reserve/context overlap returns to near 1.0;
  posterior binding_signature calibrated std still collapses to 0.
```

Do not launch a 30000-step production run solely on this result.  The correct
next decision is either a narrow follow-up that fixes PV/routing pressure and
posterior signature discriminability, or an explicitly documented long-run
diagnostic whose acceptance criteria include these failure modes.

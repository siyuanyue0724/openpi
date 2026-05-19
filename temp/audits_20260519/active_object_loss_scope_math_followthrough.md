# Active-Object Loss Scope Math Follow-Through

Date: 2026-05-19

This note records the dataflow and mathematical reason for the active-object
loss-scope repair after the A7 200-step object-candidate diagnostic.

## 1. Observed Failure

The v3 candidate assignment repair passed the candidate ownership invariant:

```text
object_candidate_coverage_mean ~= 0.97-0.98
object_candidate_background_mean ~= 0.02-0.03
object_candidate_duplicate_overlap_max = 0.0
active_same_role_support_overlap_max ~= 0.02-0.06
posterior_file_competition_active_duplicate_overlap_max = 0.0
```

But raw/reserve metrics still rose:

```text
same_role_object_core_overlap_max ~= 0.75
posterior_file_competition_duplicate_overlap_max ~= 0.98
loss_anchor_pv and loss_aqr_denoising worse than step50
```

This means the upstream sidecar candidate assignment is fixed, but some losses
still see fixed-capacity reserve/context rows as object rows.

## 2. Slot-Theoretic Invariant

Fixed slot systems maintain more slots than current objects.  The correct
latent set is not simply `K` objects:

```math
Q = Q^{obj} \cup Q^{ctx} \cup Q^{null}
```

where:

```text
Q_obj:
  active object files that may explain object candidates.

Q_ctx:
  low-weight context/peripheral capacity.

Q_null:
  background/no-object reserve capacity.
```

SlotAttention-style competition, MetaSlot-style adaptive slot count, and
temporal slot-binding work all rely on this separation: extra slots must not be
judged as object failures merely because they carry background or duplicate
no-object evidence.

## 3. PICF Dataflow Before Repair

```text
sidecar/contact/proposal mask
  -> _proposal_object_candidate_assignment
  -> graph.object_candidate_assignment
  -> graph.point_priors / proposal_priors
  -> posterior file competition
  -> active files + demoted reserve files
```

Then training used:

```text
loss_anchor_pv:
  object gate from graph.anchor_downstream_weight
  problem: context rows have small nonzero weights.

loss_aqr_denoising:
  loops over all support priors
  problem: no active-row mask.

object_explanation duplicate/feature/point:
  uses anchor_quality > 0
  problem: context/reserve rows can still have nonzero quality.
```

Thus a row could be demoted correctly by posterior file competition but still
contribute to object-level losses and raw diagnostics.

## 4. Repair

Define active object weight:

```math
w_j^{obj} =
\begin{cases}
anchor\_active_j, & anchor\_active \text{ exists} \\
1[anchor\_downstream\_weight_j > 0.5], & fallback \\
1, & legacy states
\end{cases}
```

Apply it to object-level losses:

```math
L_{anchorPV}
\leftarrow
L_{anchorPV}(w^{obj} \odot p^{point}, w^{obj} \odot p^{visual})
```

```math
L_{denoise}
\leftarrow
\sum_j w_j^{obj} 1[\max_i p_{j,i} > u_i + \epsilon] CE(p_j,\arg\max p_j)
```

```math
L_{OEML}
\leftarrow
L_{OEML}(w^{obj} \odot quality)
```

Dense evidence is not pruned.  Background/PV coverage remains through:

```text
loss_pv_weak
raw telemetry
reserve/context graph prefix weights
```

The follow-up v5 repair removes the object-loss floor:

```text
anchor_pv_object_gate_floor = 0.0
anchor_pv_object_normalize_by_object_mass = True
aqr_denoising_confirmed_object_only = True
```

This makes the split explicit:

```math
L_{pv}
=
\lambda_{obj}
\mathbb{E}_{(p,v)\sim E_{obj}}
\operatorname{BCE}(r_{p,v}, y_{p,v})
+
\lambda_{weak}
\mathbb{E}_{(p,v)\sim E_{dense}}
\operatorname{CE}(v \mid \operatorname{bag}(p))
```

where `E_obj` is induced by active object rows and candidate/proposal/point
evidence, while `E_dense` remains the global projective correspondence set.
No dense token is dropped; only the object identity loss domain is narrowed.

## 5. Why This Is Not a Patch

The repair aligns the optimization domain with the latent semantics:

```text
object loss:
  active object files only.

background/context visibility:
  retained by dense weak losses and low-weight context path.

reserve/no-object capacity:
  retained for birth/dustbin transport, not trained as object.
```

This is the same invariant used by modern object-centric systems: extra slots

## 6. V5 Step-50 Runtime Check

Run:

```text
picf_a7_v5_object_pv_split_anchor200_20260519
```

Step 50 observed values:

```text
candidate_coverage_mean       0.8477
candidate_duplicate_max       0.0000
active_support_overlap_max    0.0121
active_object_core_max        0.1137
raw_support_overlap_max       0.1929
loss_anchor_pv                5.3087
loss_aqr_denoising            1.4504
loss_mapg_routing             0.5952
posterior_identity_switch     0.6400
preclip_grad_norm             18.2694
```

Mathematical interpretation:

```text
The selector part is correct at step 50:
  active rows are sparse and non-duplicated.

The teacher part is still under validation:
  object-PV is now measured only on confirmed object edges, so the loss is a
  sharper and harder objective than the previous floor-averaged mixed loss.

Therefore step 50 falsifies neither the repair nor the concern.  The decisive
question is the trend after warmup:
  if loss_anchor_pv and denoising fall while active overlap stays low, the
  repair is working;
if they rise while active overlap stays low, the object-edge teacher itself
is still mislocalized or too noisy.
```

## 7. V5 Step-100 Negative Control, Launch Override Found

Observed:

```text
step                         50        100
active_support_overlap_max   0.0121    0.0569
active_object_core_max       0.1137    0.2149
candidate_duplicate_max      0.0000    0.0000
loss_anchor_pv               5.3087    8.5283
loss_aqr_denoising           1.4504    2.0182
preclip_grad_norm            18.2694   183.2092
```

Follow-through decision:

```text
This was not a valid pure V5 run.  The launch script still overrode
anchor_pv_object_gate_floor to 0.25, so the mixed dense/background floor was
still active.
```

Why this matters:

```math
L_{pv}^{obj}
=
\frac{\sum_{e \in E_{obj}} w_e BCE(r_e, y_e)}
       {\sum_{e \in E_{obj}} w_e}
```

was supposed to use only object-confirmed edges.  A nonzero floor changes it
back into:

```math
L_{pv}^{mixed}
=
\frac{\sum_e (0.25 + 0.75 w_e^{obj}) w_e^{proj} BCE(r_e,y_e)}
       {\sum_e (0.25 + 0.75 w_e^{obj}) w_e^{proj}}
```

which is precisely the target-domain mixing V5 was designed to remove.

Corrected run contract:

```text
anchor_pv_object_gate_floor = 0.0
anchor_pv_object_normalize_by_object_mass = True
aqr_denoising_confirmed_object_only = True
```

## 8. Pure V5 Step-50 Check

Pure V5 run:

```text
picf_a7_v5_pure_object_pv_split_anchor200_20260519
```

Comparison against the stale-floor run:

```text
metric                          stale floor=0.25   pure V5 floor=0.0
loss_total                      1.6550             1.0692
loss_alignment                  1.5800             0.9942
loss_alignment_raw              1.4474             0.8606
loss_anchor_pv                  5.3087             2.9648
loss_aqr_denoising              1.4504             1.3502
preclip_grad_norm               18.2694            9.2125
active_support_overlap_max      0.0121             0.0164
active_object_core_max          0.1137             0.1054
```

Mathematical conclusion:

```text
The floor term was not harmless.  It changed the training domain from
object-confirmed edges back to mixed dense/projective edges and doubled the
effective object-PV difficulty at step 50.

The pure V5 result supports the object/dense split:
  active object rows stay sparse and non-duplicated;
  object-PV and gradient scale are much lower;
  dense background remains represented through pv_weak and context/reserve
  slots rather than through object-PV.
```

## 9. Pure V5 Step-100 Trend

Observed:

```text
step                         50        100
loss_anchor_pv               2.9648    3.7491
loss_aqr_denoising           1.3502    1.9225
loss_mapg_routing            0.5908    0.6852
preclip_grad_norm            9.2125    61.4807
active_support_overlap_max   0.0164    0.1335
raw_support_overlap_max      0.1513    0.5884
```

Interpretation:

```text
The object/dense split is necessary but not sufficient.  Once the LR reaches
full value, raw support and self-denoising signals still move faster than
stable object identity.  Because active overlap remains far below raw overlap,
the failure is not a reversion to "all active slots collapse".  It is residual
auxiliary-teacher instability.
```

Mathematical implication:

```text
Keep object-PV externally confirmed.
Disable or delay self-denoising/routing pressure unless it is also explicitly
object-confirmed at the edge level, not merely row-confirmed.
```
are capacity, not necessarily active objects.

## 6. Code Follow-Through

```text
src/openpi/picf/core/training.py:
  _active_object_row_weight(...)
  PicfTransitionLossConfig.anchor_pv_active_object_gate_only=True
  PicfTransitionLossConfig.aqr_denoising_active_object_only=True
  PicfTransitionLossConfig.object_explanation_active_object_only=True
  PicfAlignmentLossConfig.anchor_pv_active_object_gate_only=True

scripts/picf_core_train.py:
  CLI flags expose these gates with BooleanOptionalAction.

scripts/picf_object_candidate_slot_binding_audit.py:
  active_object_scope_ignores_reserve_duplicates
  downstream_weight_fallback_excludes_context_rows
  denoising_active_object_scope_excludes_no_object_peaks

scripts/verify_picf_owm_contract.py:
  static contract now requires these gates.
```

## 7. Acceptance Criteria

The next 200-step frozen action/PaliGemma diagnostic should show:

```text
must remain good:
  object_candidate_coverage_mean high
  object_candidate_duplicate_overlap_max low
  active_same_role_support_overlap_max low
  posterior_file_competition_active_duplicate_overlap_max low

should improve:
  loss_anchor_pv should not rise because of reserve rows
  loss_aqr_denoising should not train reserve/no-object peaks
  loss_mapg_routing should not regress from active object evidence

allowed:
  raw duplicate telemetry can remain high if reserve/context files carry
  similar no-object evidence.
```

## 8. 200-Step Validation Result

Remote run:

```text
picf_a7_active_object_scope_anchor200_20260519
```

Metrics:

```text
step  active_support  active_core  candidate_coverage  raw_support  raw_core  anchor_pv  denoise  routing
  50          0.0145       0.1363             0.8635       0.1545   0.1666     0.7839   1.3174   0.6093
 100          0.0215       0.0644             0.9272       0.5125   0.6722     1.1843   1.9060   0.6325
 150          0.0508       0.0127             0.9732       0.9854   0.9341     1.1729   1.9695   0.6758
 200          0.1115       0.0203             0.9670       0.9973   0.9393     1.1778   1.8828   0.7078
```

Decision:

```text
The hypothesis "reserve rows were the only source of anchor-PV / denoising
drift" is falsified.

The repair is still valid:
  active object rows remain non-duplicated and candidate coverage stays high.

The incomplete part is:
  active rows still receive mixed objectives.  Object-candidate/proposal
  ownership wants compact task-object support, while anchor-PV and denoising can
  still reward broader or unstable typed-support peaks.
```

Next mathematical split:

```math
L_{pv}
=
\lambda_{obj} L_{pv}^{obj}(Q_{obj}, E_{candidate/point})
+
\lambda_{weak} L_{pv}^{weak}(Q_{all}, E_{dense})
```

where:

```text
L_pv_obj:
  no background floor; active-object rows only; confirmed candidate/point
  evidence only.

L_pv_weak:
  dense background/projective coverage; does not define object identity.
```

For denoising:

```math
L_{dn}
\leftarrow
1[\text{confirmed object candidate}]
L_{dn}^{candidate}
```

or keep `lambda_aqr_denoising=0` in production until this confirmed-object
scope is implemented and validated.

## 9. V6 Distributional Object-PV Follow-Through

Pure V5 fixed the dense/background floor but left a deeper semantic mismatch:
`loss_anchor_pv` was still an edge BCE over observation co-routing and dense
projective compatibility.  That is not the same mathematical object as a slot
explaining its own evidence distribution.

Legacy object-gated edge BCE:

```math
L_{legacy}
=
\frac{\sum_{p,u} w^{obj}_{p,u} C_{p,u} BCE(R_{p,u}, C_{p,u})}
{\sum_{p,u} w^{obj}_{p,u} C_{p,u}}
```

V6 per-object distribution target:

```math
\hat v_j = normalize(p_j C)
```

```math
\hat p_j = normalize(v_j C^T)
```

```math
L^{V6}_{anchorPV}
=
\mathbb E_{j \sim active/confirmed}
\frac{
JS(v_j, \hat v_j)
+
JS(p_j, \hat p_j)
}{2}
```

This is the slot/object-centric invariant we actually need:

```text
slot j point support and slot j visual support must describe the same object;
dense/background PV remains in pv_weak;
reserve/context rows remain available but are not object identity labels.
```

Implementation:

```text
src/openpi/picf/core/training.py:
  _object_projective_distribution_loss(...)
  anchor_pv_object_distribution_loss=True
  anchor_pv_object_distribution_confirmed_only=True

scripts/picf_core_train.py:
  --anchor-pv-object-distribution-loss
  --anchor-pv-object-distribution-confirmed-only
  --anchor-pv-object-distribution-confirmation-threshold

scripts/picf_object_candidate_slot_binding_audit.py:
  distributional_object_pv_penalizes_slot_projective_mismatch

scripts/picf_anchor_pv_object_gate_audit.py:
  distributional_object_pv_replaces_dense_edge_bce
```

This is a replacement of the mismatched object-PV target, not an added
after-the-fact penalty.

## 10. V6 200-Step Post-Diagnostic Boundary

Run:

```text
picf_a7_v6_distribution_object_pv_anchor200_20260519
```

Result:

```text
loss_total             0.4732 -> 0.3563
loss_alignment         0.3982 -> 0.2813
loss_anchor_pv         0.5684 -> 0.5516
loss_pv_weak           6.0047 -> 2.8625
active_support_overlap 0.0125 -> 0.0986
active_object_overlap  0.1286 -> 0.0620
raw_support_overlap    0.1805 -> 0.8881
candidate_dup_overlap  0.0000 -> 0.0000
posterior_swap_rate    0.3061 -> 0.2798
```

Conclusion:

```text
The object-PV target-domain bug is fixed.

The next unresolved target is posterior active-file continuity, not raw
reserve/context overlap.  Raw overlap should stay logged, but it must not be
used as the primary object-binding acceptance metric after active-slot and
candidate assignment filters are enabled.
```

Do not regress:

```text
Do not restore dense-edge object-PV.
Do not add another raw overlap loss.
Do not enable denoising/slot-JEPA/support-pred merely because they are logged.
```

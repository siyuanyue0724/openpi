# V6 Slot-Paper / Runtime / Loss Follow-Through

Date: 2026-05-19

Scope: reconcile the latest A7 V6 200-step diagnostic with the pulled
slot/object-centric paper code, the PICF dataflow, and the current loss
semantics.  This note is intentionally strict: it separates a real active-object
binding failure from reserve/context overlap and from diagnostic-only losses.

## 1. Runtime Evidence

Run:

```text
picf_a7_v6_distribution_object_pv_anchor200_20260519
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

Local diagnostic artifact:

```text
/tmp/picf_a7_v6_distribution_object_pv_anchor200_20260519/run
/tmp/picf_a7_v6_distribution_object_pv_anchor200_20260519/report.json
```

Measured trend:

```text
step                   50        100       150       200
loss_total             0.4732    0.4154    0.3679    0.3563
loss_alignment         0.3982    0.3404    0.2929    0.2813
loss_alignment_raw     0.2622    0.2213    0.1973    0.1952
loss_anchor_pv         0.5684    0.5735    0.5419    0.5516
loss_pv_weak           6.0047    3.8959    3.0909    2.8625
loss_aqr_denoising     1.4723    1.8976    1.9533    1.9169
loss_mapg_routing      0.6080    0.6477    0.6706    0.6746
loss_slot_jepa         1.3086    1.5235    1.6078    1.7893
loss_support_pred      0.1717    0.2551    0.2859    0.2948
active support overlap 0.0125    0.0735    0.0965    0.0986
active object overlap  0.1286    0.2871    0.1001    0.0620
raw support overlap    0.1805    0.3253    0.7740    0.8881
raw object overlap     0.1720    0.4739    0.6552    0.6857
candidate coverage     0.8671    0.9751    0.9794    0.9826
candidate dup overlap  0.0000    0.0000    0.0000    0.0000
posterior swap rate    0.3061    0.3401    0.2440    0.2798
posterior recycle      0.0814    0.0234    0.0009    0.0006
```

Acceptance:

```text
ACCEPTED for the V5/V6 object-PV root cause.
NOT ACCEPTED as final behavior proof.
NOT evidence that slot-JEPA/support-pred/denoising can be enabled as losses.
```

Post-V6 diagnostic correction:

```text
The raw posterior_active_file_potential_swap_rate from this run is not decisive.
Runtime binding uses calibrated IsSameObject-style relative scores, but the old
file-continuity metric used raw cosine.  Use
posterior_active_file_calibrated_potential_swap_rate and
posterior_file_calibrated_signature_score_std on the next run before treating
file swap as a model failure.
```

Follow-through:

```text
temp/audits_20260519/posterior_file_continuity_metric_followthrough.md
```

## 2. Paper-Code Invariants Checked

Local paper-code snapshots:

```text
temp/paper_code_20260518/MetaSlot
temp/paper_code_20260518/slotcontrast
temp/paper_code_20260518/AdaSlot
temp/paper_code_20260518/object-centric-learning-framework
temp/paper_code_20260518/slot-attention-video
```

MetaSlot / SlotAttention invariant:

```math
a_{ij} = softmax_j(q_j^T k_i / sqrt(d))
```

Input evidence first competes over slots, then each slot aggregates its assigned
evidence.  Extra capacity is not the same as extra objects; duplicate active
slots must be demoted or masked.

SlotContrast invariant:

```math
L = CE(s_t s_{t+1}^T / tau, identity)
```

Temporal slot consistency is useful only when identity matching is stable.
Without stable matching, the loss becomes a self-confirming permutation error.

PICF retained invariants:

```text
1. typed evidence competes for object files;
2. active/context/reserve rows are distinct;
3. reserve/context rows are not forced to be object labels;
4. dense/background projective coverage is kept in pv_weak;
5. object-PV compares a slot's own point support to its own visual support;
6. temporal/predictive losses stay guarded until matching is accepted.
```

PICF does not copy a standalone MetaSlot codebook or a full SlotContrast model
because PICF is a recurrent POMDP belief filter with geometry, actions,
tactile/contact, cache, posterior files, and PI0.5 action prefixes.  Copying a
pure image/video slot learner would replace the state model instead of repairing
the failing likelihood term.

## 3. Mathematical Root Cause Fixed by V6

The failing target was not "slot count" or "unroll length".  The failing target
was the object-PV objective domain.

Old edge-style object-PV:

```math
L_{edge}
=
sum_{p,u} w_{j,p,u} BCE(R_{j,p,u}, C_{p,u})
```

This can punish a correct object slot because it compares the slot to a dense
projective edge field that also includes background/reserve/context evidence.

V6 object-distribution PV:

```math
\hat v_j = normalize(p_j C)
```

```math
\hat p_j = normalize(v_j C^T)
```

```math
L^{V6}_{anchorPV}
=
mean_j 0.5 [ JS(v_j, \hat v_j) + JS(p_j, \hat p_j) ]
```

This is the correct object-centric statement:

```text
if slot j claims point support p_j and visual support v_j, those two
distributions should project to the same object.
```

It does not force the slot to reconstruct every dense/background edge.

## 4. Loss Interpretation After V6

Healthy signals:

```text
loss_total, loss_alignment, loss_alignment_raw:
  monotonic decline; the optimized alignment objective is trainable.

loss_anchor_pv:
  low and roughly flat, unlike V5's upward drift; the root mismatch is fixed.

loss_pv_weak:
  strong decline; dense/background projective coverage is still learned.

active same-role support/object overlap:
  remains low; active object rows are not duplicating the same candidate.

candidate duplicate overlap:
  remains zero; candidate assignment is not cloning one proposal into many
  active rows.
```

Diagnostic-only warning signals:

```text
raw same-role overlap:
  high because reserve/context/background rows are included.
  It is a capacity/background diagnostic, not active-object failure.

loss_aqr_denoising / loss_mapg_routing / loss_slot_jepa / loss_support_pred:
  mildly rise in this run.  Their optimization weights are zero here, so they
  must not be read as proof that those losses are ready to enable.

posterior active-file potential swap:
  remains about 0.24-0.34.  This is the next real scientific/engineering target:
  object-file continuity after active owner selection.
```

## 5. Current Problem Boundary

Do not fix the wrong thing:

```text
Do not add another same-role raw-overlap penalty.
Do not re-enable local refinement.
Do not use blind SAM.
Do not enable denoising/slot-JEPA/support-pred as training losses.
Do not judge this run by raw gray/reserve overlap.
```

The remaining real target:

```text
posterior active-file continuity:
  active owners are separated, but persistent posterior files still have a
  nontrivial potential-swap rate.  This needs a file-continuity audit or a
  longer full training run with overlays before another architecture change.
```

## 6. Executable Verification

Commands run locally after the V6 implementation:

```bash
PYTHONPATH=src:scripts uv run python scripts/picf_object_candidate_slot_binding_audit.py
PYTHONPATH=src:scripts uv run python scripts/picf_anchor_pv_object_gate_audit.py
PYTHONPATH=src uv run python scripts/verify_picf_owm_contract.py
PYTHONPATH=src:scripts uv run python scripts/picf_owm_nontruncated_paper_audit.py
PYTHONPATH=src:scripts uv run python scripts/picf_oeml_dataflow_audit.py
PYTHONPATH=src:scripts uv run python scripts/picf_anchor_run_diagnostic_report.py \
  /tmp/picf_a7_v6_distribution_object_pv_anchor200_20260519/run --fail-on-missing
```

All completed successfully.  The diagnostic report explicitly found:

```text
raw overlap high while active-owner overlap low:
  reserve/inactive anchors are the likely raw-overlap source.

identity switch high despite healthy active-owner overlap:
  next audit should isolate active-owner object-file continuity.

posterior active-file potential swap high:
  remaining target is object-file update/continuity, not raw support separation.
```

## 7. Next Action

If the next goal is training:

```text
launch the V6 profile into a longer full run;
keep object-PV distributional;
keep denoising/slot-JEPA/support-pred/binding-consistency as diagnostics;
write active-only, with-gray, and sidecar-proposal overlays.
```

If the next goal is one more architecture fix:

```text
do not touch active owner selection.
build an active-file continuity diagnostic first:
  compare active owner identity, posterior file id, support signature, geometry,
  and task proposal assignment across windows.
Only if that audit shows real file swaps should we modify posterior continuity.
```

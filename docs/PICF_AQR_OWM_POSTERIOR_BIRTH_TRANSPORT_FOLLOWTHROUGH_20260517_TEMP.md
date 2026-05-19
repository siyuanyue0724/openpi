# PICF-AQR-OWM Posterior Birth Transport Follow-Through - 2026-05-17

Status: local patch prepared; A7 short diagnostic pending.

This note tracks the step300 failure of the A7 SAM-proposal anchor-only
diagnostic and the corresponding posterior transport repair.

## 1. Observed Failure

Run:

```text
picf_a7_sam_phase0_anchoronly_diag500_20260517
```

The run was stopped after step300/350 because the diagnostic already exposed
the failure.

```text
step50 -> step300:
  loss_anchor_pv                         0.7533 -> 1.1787
  loss_slot_jepa                         0.7818 -> 4.7354
  aqr_same_role_support_overlap_max      0.1733 -> 0.9996
  aqr_active_same_role_support_overlap   0.0184 -> 0.4216
  posterior_recycle_rate                 0.0556 -> 0.2743
  posterior_active_duplicate_overlap     0.0000 -> 0.0000
  owm_proposal_tokens                    3-5
  owm_tracklet_tokens                    0
```

Interpretation:

```text
The active duplicate guard works, but the underlying posterior candidate
lifecycle is not stable.  The model still creates duplicate reserve/context
files and active overlap starts rising.  Because action loss is disabled, this
is a posterior transport/lifecycle issue rather than action cotrain pressure.
```

## 2. Previous Transport

Let object files be `j=1..K` and observation anchors be `i=1..N`.

Posterior binding first computes:

```math
P = SinkhornDustbin(L),\quad
S_{j,i}=P_{j,i},\quad
D_i=P_{dustbin,i}
```

File competition demotes duplicate same-role files:

```math
S'_{j,i}=a_j S_{j,i}
```

```math
D'_i = D_i + \sum_j (1-a_j) S_{j,i}
```

where `a_j in {0,1}` is the file-active owner decision.

The previous recycle step then redistributed the dustbin residual with:

```math
r_j = sigmoid(f_j) \cdot reset\_allowance_j
```

```math
\rho_j = \frac{r_j}{1+\sum_l r_l}
```

```math
\tilde S_{j,i}=S'_{j,i}+\rho_j D'_i
```

This is mathematically inconsistent for fixed-capacity object files: if many
inactive files have similar recycle logits, they all receive the same dustbin
observation vector and become duplicate candidates.

## 3. Repaired Transport

The repaired model splits existing-object update from new-object birth.

Existing files:

```math
S'_{j,i}=a_j S_{j,i}
```

Reserve-file birth competition:

```math
b_j =
TopK_{role(j)}
\left[
  r_j
  (1-a_j)
  (1-\alpha_j)^\gamma
\right]
```

with at most `posterior_birth_competition_max_per_role` births per role and a
minimum score threshold.

Only selected birth files consume the demoted dustbin residual:

```math
\beta_j = \frac{b_j r_j}{1+\sum_l b_l r_l}
```

```math
\tilde S_{j,i}=S'_{j,i}+\beta_j D'_i
```

```math
D''_i = \frac{D'_i}{1+\sum_l b_l r_l}
```

Slots with neither existing support nor selected birth keep null/reserve state:

```math
update_j = 1[\sum_i S'_{j,i}>0] \lor b_j
```

```math
r^{update}_j = r_j \cdot update_j
```

Thus dustbin evidence is preserved, but it is not broadcast into every reserve
file.

## 4. Relation To Paper Code

The VIT object-binding repository uses pairwise IsSameObject probes rather
than raw token cosine:

```text
/tmp/vit-object-binding/src/utils/models.py
/tmp/vit-object-binding/src/utils/score.py
```

The relevant lesson is that binding is a projected pairwise subspace and must
be used in assignment, not treated as a scalar label.  Our current binding
signature already follows this principle.  The step300 failure is downstream:
assignment has a binding signal, but the lifecycle/birth transport reuses the
same dustbin residual for too many reserve files.

SlotContrast uses temporal slot consistency and explicit object-centric
matching/evaluation:

```text
/tmp/picf_paper_code_20260515/slotcontrast/slotcontrast/losses.py
/tmp/picf_paper_code_20260515/slotcontrast/slotcontrast/metrics.py
```

The relevant lesson is that duplicate slots are not solved by a generic
diversity scalar; they require object-level competition/matching and a stable
temporal owner decision.  The repaired birth competition is the posterior
transport analogue of that principle.

## 5. Code Changes

Config:

```text
posterior_birth_competition_enabled
posterior_birth_competition_max_per_role
posterior_birth_competition_min_score
posterior_birth_competition_inactive_only
posterior_birth_alpha_suppression_power
```

Contracts:

```text
PicfPosteriorAnchorState.file_competition_birth_active
PicfPosteriorAnchorState.file_competition_birth_share
```

Pipeline:

```text
_posterior_birth_competition(...)
_posterior_update(...):
  support_raw,dustbin_raw
    -> file competition
    -> lifecycle/recycle
    -> birth competition
    -> measurement summary from support + bounded birth residual
```

Trainer/verification:

```text
scripts/picf_core_train.py:
  exposes --posterior-birth-competition-* CLI switches;
  prints a dedicated Posterior birth transport startup contract;
  logs birth_active/birth_share debug metrics.

scripts/picf_posterior_birth_transport_audit.py:
  executable torch-free math audit for one-birth-per-role selection,
  no active-file birth, no dustbin broadcast, and mass conservation.
```

Short diagnostic launch:

```text
run_a7_birth_transport_sam_phase0_anchoronly_diag300_20260517.sh
```

## 6. Acceptance Criteria

Short diagnostic acceptance:

```text
posterior_file_competition_birth_count is bounded.
posterior_recycle_rate does not climb with active overlap.
aqr_active_same_role_support_overlap_max does not steadily increase.
loss_anchor_pv does not trend upward.
loss_slot_jepa diagnostic does not monotonically drift upward.
proposal tokens remain nonzero.
tracklet tokens are either nonzero or explicitly recorded as absent.
```

Long-run acceptance:

```text
30000-step action cotrain can be considered only after a short diagnostic
passes the lifecycle criteria above.
```

# PICF-AQR-OWM Proposal-To-Point Bridge Follow-Through

Status: bridge implementation retained behind explicit opt-in. Later A7
diagnostics showed blind SAM proposals are not reliable enough for production
defaults, so proposal memory and all proposal bridge/task-owner proposal weights
now default to off/zero.

## Problem

The A7 sparse-proposal diagnostic repaired SAM/proposal coverage: saved overlay
frames now contain proposal boxes and `proposal_age`.  At step 100 for
`pick up the red block lying in the drawer`, the red block proposal was present:

```text
nearest proposal center ~= (106, 156) px
red block visual center ~= (107, 157) px
proposal age = 5
```

However, active posterior object files still did not bind the red block.  The
nearest active anchors were around 20-30 px away, while the closest graph row at
about 6 px was demoted to context/inactive.

## Root Cause

The previous proposal path was live but incomplete:

```text
SAM proposal -> proposal token -> AQR proposal read -> proposal_priors
```

but physical anchor geometry was still computed as:

```math
x_j = \sum_i p^{point}_{j,i} X_i
```

where `p^{point}` came from point support only.  A 2D proposal could increase
proposal support and anchor scores, but it could not directly move `anchor_x`
toward the task object unless the point reader independently selected the same
3D points.

This is a dataflow gap, not a training-duration explanation.

## Repair V1: Row Proposal-To-Point Bridge

The repaired path treats proposal boxes as frozen weak measurement evidence and
bridges them to point geometry through the existing static projective geometry.

For proposal box `b_k=[x0,y0,x1,y1]` and projected point coordinate `u_i`, define
a soft membership:

```math
M_{i,k}
=
\sigma((u^x_i-x^0_k)/\tau)
\cdot
\sigma((x^1_k-u^x_i)/\tau)
\cdot
\sigma((u^y_i-y^0_k)/\tau)
\cdot
\sigma((y^1_k-u^y_i)/\tau)
```

implemented multiplicatively in code:

```math
M_{i,k}
=
\sigma((u^x_i-x^0_k)/\tau)
\cdot
\sigma((x^1_k-u^x_i)/\tau)
\cdot
\sigma((u^y_i-y^0_k)/\tau)
\cdot
\sigma((y^1_k-u^y_i)/\tau)
```

with visibility and valid-depth gates applied.  Then:

```math
P^{proposal\to point}_{j,i}
=
\sum_k P^{proposal}_{j,k}
\operatorname{Normalize}_i(M_{i,k})
```

and point support is mixed as bounded measurement fusion:

```math
P^{point,new}_j
=
\operatorname{Normalize}
\left(
  (1-\lambda_{bridge}) P^{point}_j
  + \lambda_{bridge} P^{proposal\to point}_j
\right)
```

Default:

```text
proposal_point_bridge_weight = 0.35
proposal_point_bridge_edge_tau = 0.02
```

This does not turn SAM into truth.  It only lets a task-relevant 2D proposal
create a weak 3D measurement so `anchor_x` can move to the object when matching
projected points exist.

## V1 Diagnostic Result

The first A7 diagnostic confirmed that this bridge is live but not sufficient:

```text
step50:
  owm_proposal_tokens ~= 31
  nearest proposal to red block ~= 1.4 px
  aqr_proposal_point_bridge_max ~= 0.058
  aqr_task_owner_proposal_score_max = 1.0
  nearest active anchor to red block ~= 37 px
```

The failure is more specific: the task owner can select the red-block proposal,
but physical rows still need to independently read that proposal before the v1
bridge can move their `anchor_x`.

## Repair V2: Task-Owner Proposal-To-Point Bridge

The v2 bridge uses the task-owner proposal score directly as a weak projected
point measurement for physical scene-object rows:

```math
P^{owner\to point}_{i}
=
\sum_k P^{owner\_proposal}_k
\operatorname{Normalize}_i(M_{i,k})
```

Then, only for task-owner-eligible physical scene rows:

```math
P^{point,new}_j
=
\operatorname{Normalize}
\left(
  (1-\lambda_{owner})P^{point}_j
  + \lambda_{owner}P^{owner\to point}
\right)
```

Default:

```text
task_owner_proposal_point_bridge_weight = 0.50
```

This keeps the posterior authoritative.  The new path only supplies the missing
measurement transport:

```text
PaliGemma/task visual prior
  -> task-owner proposal score
  -> soft projected point likelihood
  -> physical scene-object file competition
  -> posterior correction
```

## Why This Is Not A Patchy Hard Rule

The repair preserves the belief-state contract:

```text
proposal boxes:
  frozen observation evidence

projected points:
  3D measurement support

AQR:
  assignment/routing over typed evidence

posterior:
  authoritative corrected belief
```

No token is dropped.  Dense V-JEPA/visual tokens and point tokens still flow.
The bridge is a bounded measurement-fusion term, analogous to using a detector
box as a weak observation likelihood over projected points.

## Code Follow-Through

```text
src/openpi/picf/core/config.py
  proposal_point_bridge_weight
  proposal_point_bridge_edge_tau
  task_owner_proposal_point_bridge_weight

src/openpi/picf/core/contracts.py
  PicfAnchorPriorGraphState.proposal_point_priors
  PicfAnchorPriorGraphState.task_owner_point_priors
  PicfAnchorPriorGraphState.task_owner_anchor_score

src/openpi/picf/core/pipeline.py
  _proposal_to_point_matrix()
  _proposal_priors_to_point_priors()
  _task_owner_proposal_to_point_priors()
  _task_owner_anchor_score()
  point_priors <- bounded mix(point_priors, proposal_point_priors)
  point_priors <- bounded mix(point_priors, task_owner_point_priors)
  anchor_scores += task_owner_anchor_score
  debug:
    aqr_proposal_point_bridge_entropy_mean
    aqr_proposal_point_bridge_max
    aqr_task_owner_point_bridge_entropy_mean
    aqr_task_owner_point_bridge_max
    aqr_task_owner_point_bridge_nonzero_fraction
    aqr_task_owner_anchor_score_max
    aqr_task_owner_anchor_score_mean
    aqr_task_owner_anchor_score_nonzero_fraction

scripts/picf_core_train.py
  CLI:
    --proposal-point-bridge-weight
    --proposal-point-bridge-edge-tau
    --task-owner-proposal-point-bridge-weight
  compact metrics include bridge/debug scores.

scripts/verify_picf_owm_contract.py
  verifier checks task-owner proposal-to-point bridge.

scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py
  audit checks that proposal evidence can become 3D point support.
```

## Acceptance Criteria

1. `owm_proposal_tokens > 0`.
2. `aqr_proposal_point_bridge_max > 0`.
3. `aqr_task_owner_anchor_score_max > 0`.
4. `aqr_task_owner_point_bridge_max > 0`.
5. For `pick up the red block lying in the drawer`, the active-only overlay
   should place at least one active physical/posterior object file on or very
   near the red-block proposal, not only on the gripper/drawer edge.
6. Same-role active duplicate overlap must remain controlled.

## Remaining Boundary

If this bridge works but the active posterior still misses the red block, the
remaining issue is not proposal coverage or 2D-to-3D geometry.  It is then
semantic task-owner calibration: whether the PaliGemma/task-query visual prior
selects the red-block proposal strongly enough.

## A7 V2 Step50 Result

The first v2 checkpoint point changed the failure mode:

```text
step50:
  nearest proposal to red block ~= 1.4 px
  nearest active graph anchor to red block ~= 11.2 px
  nearest active posterior anchor to red block ~= 11.9 px
  aqr_task_owner_point_bridge_max ~= 0.017
  aqr_same_role_support_overlap_max ~= 0.174
  posterior_file_competition_active_duplicate_overlap_max = 0
```

Compared with v1 step50, where the nearest active anchors were still around
37-53 px from the red block, v2 confirms that task-owner proposal evidence can
now move active object geometry into the intended task-object neighborhood.
This is not final acceptance yet; the step100/300 trend must verify that the
binding remains stable instead of drifting back to the gripper/drawer edge.

## A7 V2 Step100 Regression

Step100 shows that v2 is live but not sufficient:

```text
step100:
  aqr_task_owner_point_bridge_max ~= 0.020
  aqr_task_owner_anchor_score_max ~= 0.214
  aqr_task_owner_proposal_score_max = 1.0
  nearest active anchor to red block ~= 24 px
  nearest active posterior to red block ~= 41 px
  loss_anchor_pv ~= 1.18
  loss_aqr_denoising ~= 2.03
  aqr_same_role_support_overlap_max ~= 0.69
```

The bridge is therefore not a fake field, but as currently weighted it behaves
like a transient weak prior.  After warmup, the anchor/PV/support objectives can
still pull active files away from the task-object proposal.

Next repair should not add another unrelated module.  It should strengthen the
same belief-model term by making task-owner proposal-to-point evidence a stable
observation likelihood for eligible object files, while keeping posterior
competition and duplicate demotion authoritative.

## Repair V3: Task-Owner Point-Reader Likelihood Bias

V2 mixed owner point priors after AQR point attention.  That moves `anchor_x`,
but it does not alter the query-to-point read itself.  V3 injects the same
task-owner proposal-to-point evidence into the point-reader logits:

```math
b^{owner\_point}_{j,i}
=
\lambda_{point\_bias}
\left(
  \log \operatorname{Normalize}_i(P^{owner\to point}_i)
  -
  \operatorname{mean}_i \log \operatorname{Normalize}_i(P^{owner\to point}_i)
\right)
```

for task-owner-eligible physical scene rows.  Then:

```math
p^{point}_{j,i}
=
\operatorname{softmax}_i(q_j^\top k_i + b^{base}_{j,i}
+ b^{owner\_point}_{j,i})
```

Default:

```text
task_owner_proposal_point_bias_weight = 0.75
```

This is the same observation-likelihood path, not a new loss.  It should make
the task-owner measurement persistent through hidden-state update, point priors,
and `anchor_x`, rather than only changing `anchor_x` after attention.

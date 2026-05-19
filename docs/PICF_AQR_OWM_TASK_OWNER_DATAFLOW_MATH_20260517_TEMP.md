# PICF-AQR-OWM Task-Owner Dataflow and Math Audit - 2026-05-17

Status: code repair deployed locally; behavior acceptance requires the next
remote diagnostic run.

## Problem

SAM/proposal boxes are class-agnostic objectness.  PaliGemma task semantics are
available, but before this repair they mainly shaped task rows and weak visual
bias.  Physical scene-object anchors could therefore attach to object-like
regions without a strong path that says which object is the prompt target.

This is not a slot-capacity problem and not a reason to hard-label a proposal as
the posterior truth.  It is a missing measurement-routing edge:

```text
task semantic support -> physical scene-object ownership evidence
```

## Paper-Code Comparison

The local paper-code audit used:

```text
/tmp/picf_sam_code/vit-object-binding
/tmp/picf_sam_code/segment-anything
/tmp/picf_sam_code/sam2
/tmp/picf_sam_code/openmask3d
/tmp/picf_sam_code/open3dis
```

The object-binding code treats same-object information as a pairwise/projected
subspace, not as raw attention.  SAM/SAM2 provide objectness masks/boxes and
object pointers, but do not provide task identity.  OpenMask3D/Open3DIS fuse
2D/3D proposals, but still require a text/task association step.

Therefore the correct PICF repair is:

```text
class-agnostic proposal objectness
+ task-conditioned visual support
-> soft task-owner proposal bias
-> AQR measurement routing
-> posterior correction
```

It is not:

```text
SAM mask -> hard posterior target
```

## Dataflow

```text
PaliGemma text/image support
  -> task AQR rows
  -> visual_priors[task rows]
  -> task_owner_visual_prior
  -> proposal box projection over static visual grid
  -> task_owner_proposal_score
  -> proposal attention bias for scene physical rows
  -> graph.proposal_priors
  -> observation anchor binding signature / active-slot gating / posterior
```

The visual prior is also recycled across AQR rounds:

```text
round k:
  task rows read visual/PG support and produce task_owner_visual_prior

round k+1:
  physical scene rows receive a low-amplitude centered-log visual bias from
  that prior
```

This keeps the graph causal inside the iterative measurement update.

## Math

Task visual prior:

```math
\pi^{task}_i =
{1 \over |Q_{task}|}
\sum_{j \in Q_{task}} p^{visual}_{j,i}
```

Proposal projection:

```math
s_p =
o_p^\alpha
{ \sum_i \pi^{task}_i \mathbf{1}[x_i \in box_p] \over \sqrt{c_p+\epsilon} }
```

where:

```text
o_p: proposal objectness
c_p: visual-grid coverage of proposal p
alpha: task_owner_proposal_objectness_power
```

The square-root coverage correction prevents large proposals from winning
only because they contain many visual cells.  It is deliberately softer than
dividing by coverage, because large objects can be valid targets.

Centered log bias:

```math
b_p =
\lambda
\left[
\log {s_p \over \sum_q s_q+\epsilon}
-
mean_q \log {s_q \over \sum_r s_r+\epsilon}
\right]
```

The centering is important.  A constant logit shift disappears inside softmax,
while centered logits preserve relative task ownership without changing the
proposal branch into a hard gate.

## Invariants

```text
1. No visual/V-JEPA tokens are discarded.
2. Gray/reserve anchors are action-gated, not memory-gated.
3. SAM/proposals remain weak evidence, never posterior truth.
4. Wrist/non-static proposals are not projected to static geometry without
   calibration; task_owner_proposal_static_only defaults true.
5. The repair affects measurement routing only; it adds no online weak-label
   loss and does not touch PI0.5 action generation.
```

## Acceptance Metrics

```text
aqr_task_owner_visual_prior_max
aqr_task_owner_visual_prior_entropy
aqr_task_owner_proposal_score_max
aqr_task_owner_proposal_score_nonzero_fraction
aqr_active_same_role_support_overlap_max
aqr_active_same_role_object_core_overlap_max
loss_anchor_pv
loss_aqr_denoising
loss_action_default_equiv
```

Behavior is accepted only if the next run shows nonzero task-owner proposal
scores and the active object anchors move toward the prompt target without a
new active duplicate collapse.

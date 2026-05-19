# PICF-AQR-OWM Object-Candidate Slot Binding Follow-Through

Status: code-level implementation and verifier contract landed 2026-05-19.
This file is the canonical temp record for the sidecar-mask-to-slot deployment.
It is linked from `src/openpi/picf/README_v2.2.md` and should be kept with
future experiment notes until the 1000-step and 30000-step behavior gates
settle.

## 1. Problem Statement

The earlier sidecar path did not fail because the sidecar masks were noisy.
Modern object-centric learning assumes noisy foreground/background, overlapping
or incomplete masks, and imperfect pseudo labels.  The actual gap was that PICF
used the sidecar only as a weak proposal seed:

```text
proposal sidecar
-> proposal token
-> proposal-to-point bridge
-> proposal anchor seed / task-owner bias
-> normal AQR
```

This tells an anchor where it may look, but it does not make object candidates
compete for slots.  Therefore a red-block proposal could be present and selected
while physical anchors still drifted to wrist/contact/background supports.

The missing invariant is:

```text
object candidate p must be explained by object slot j or by background/no-object
```

not:

```text
object candidate p merely adds a small attention bias.
```

## 2. Paper/Code Alignment

The local paper-code comparison root is:

```text
temp/paper_code/slotcontrast
```

Reference scope:

```text
SlotAttention / SlotContrast:
  Use slot-wise feature competition and mask-style explanation as the core
  invariant. PICF copies this invariant, not the RGB autoencoder objective.

Deformable DETR / DINO-style query systems:
  Use reference/candidate evidence to initialize or bias object queries, then
  keep query competition and downstream matching. PICF keeps sidecar proposals
  as measurement candidates, not as truth labels.

IsSameObject / object-binding probe literature:
  Treat object identity as a pairwise/projected relation that can be decoded
  from representations. PICF keeps binding-signature and support-overlap terms
  as evidence terms, but does not add a new online supervised probe loss without
  reliable same-object labels.

PICF belief-filter invariant:
  posterior remains authoritative; sidecar masks, point supports, tactile
  contact, V-JEPA temporal support, and PaliGemma image support are typed
  measurements entering AQR/posterior correction.
```

The relevant SlotContrast / Slot-Attention invariants are visible in the cloned
code:

```text
temp/paper_code/slotcontrast/slotcontrast/modules/groupers.py
  SlotAttention.step:
    pre_norm_attn = softmax(dots, dim=1)
    attn = pre_norm_attn + eps
    attn = attn / attn.sum(-1, keepdim=True)
    updates = einsum(attn, values)
```

This is slot-wise competition over features: slots compete to explain tokens,
then token assignments are renormalized before slot updates.

```text
temp/paper_code/slotcontrast/slotcontrast/modules/decoders.py
  recons, alpha = mlp(slots).split(...)
  masks = softmax(alpha, dim=1)
  recon = sum(recons * masks, dim=1)
```

This is object-mask competition over a dense output lattice.  Each token/pixel is
explained by one or a small mixture of slots.

```text
temp/paper_code/slotcontrast/slotcontrast/metrics.py
  true/pred masks are checked for overlap/background handling;
  ignore_background and ignore_overlaps are explicit.
```

The PICF translation cannot copy an image reconstruction decoder literally,
because PICF is a robotic belief-state router, not an RGB autoencoder.  The
correct invariant translation is:

```text
sidecar object candidates compete for physical scene slots
+ explicit background residual absorbs invalid/noisy candidates
+ posterior remains authoritative
+ dense V-JEPA/point/tactile tokens are not pruned
```

## 3. Mathematical Contract

Let:

```text
j = physical scene slot / anchor row
p = sidecar object candidate, from contact/motion mask or inspected proposal
n = 3D point token
M[p,n] = proposal-to-point soft mask transport
P[j,p] = AQR proposal prior
Q[j,n] = AQR point prior
S[j,p] = proposal anchor seed assignment
T[p]   = task-owner proposal score
B      = background residual
```

Candidate-slot score after the 2026-05-19 runtime-scale repair:

```math
\hat P_{j,p}
= P_{j,p} / max_k P_{k,p}
```

```math
\hat Q_{j,p}
= (sum_n Q_{j,n} M_{p,n}) / max_k (sum_n Q_{k,n} M_{p,n})
```

```math
E_{j,p}
= w_p \hat P_{j,p}
+ w_q \hat Q_{j,p}
+ w_s \hat S_{j,p}
+ w_t \hat T_p
```

Important guard remains:

```math
E_{j,p} = -inf
```

unless at least one row-specific support source connects slot `j` to candidate
`p`.  This prevents a task-level proposal score from being copied uniformly into
all same-role slots, which was one source of duplicate/overlap collapse.

The repair deliberately avoids multiplying weak sources as likelihood factors.
A sidecar mask projected to only a few visible points may produce a small
absolute point-overlap value; this is weak positive evidence, not evidence that
the candidate is background. This matches the SlotAttention/SlotContrast code
pattern where features compete over slots through positive logits and mask
softmaxes, while missing weak sources are neutral rather than multiplicative
vetoes.

Assignment with background:

```math
A_{j,p}
= exp(E_{j,p}/tau)
  / (sum_k exp(E_{k,p}/tau) + exp(log b - beta q_p))

B_p
= exp(log b - beta q_p)
  / (sum_k exp(E_{k,p}/tau) + exp(log b - beta q_p))
```

where `b = object_candidate_background_prior`, `q_p` is candidate quality from
task-owner score/objectness/proposal support, and
`beta = object_candidate_background_quality_weight`. High-quality task/contact
candidates therefore do not disappear into the background merely because one
auxiliary source is numerically small. If there is no row-specific source, the
candidate still goes to background.

Candidate-capacity repair:

```math
K_p = TopK_j(E_{j,p}, k)
```

```math
E_{j,p} = -\infty \quad \text{if } j \notin K_p
```

and, for the optional soft row-capacity refinement:

```math
\tilde A_{j,p}
=
\frac{\exp(E_{j,p}/\tau)}
     {\sum_k \exp(E_{k,p}/\tau) + \exp(b_p)}
```

```math
\exp(E_{j,p}/\tau)
\leftarrow
\exp(E_{j,p}/\tau)
\cdot
\min\left(1,\frac{c}{\sum_p \tilde A_{j,p}+\epsilon}\right)
```

for a small fixed number of iterations.

The 2026-05-19 contact-bridge repair changes the maintained default from
`k=1` to `k=2` and narrows the eligible physical rows to:

```math
R_{\text{candidate}}
= \{j: query\_type_j=\text{physical}, role_j \in \{1,2\}\}
```

Role 1 is the task/object file. Role 2 is the interaction/contact bridge that
lets tactile/contact evidence attach to the same object. Role 0 effector rows
are explicitly excluded: the gripper may provide context and contact evidence,
but it must not become the owner of the task object. This is the correct PICF
translation of SlotAttention-style explanation: a sidecar candidate may be
explained by one object owner plus one contact bridge, but not cloned across
arbitrary raw rows.

The resulting object-candidate measurement is used only as a measurement prior:

```math
Q'_{j,n}
= normalize((1-lambda_q) Q_{j,n}
             + lambda_q sum_p A_{j,p} M_{p,n})

P'_{j,p}
= normalize((1-lambda_p) P_{j,p}
             + lambda_p A_{j,p})
```

Anchor confidence/score also receives bounded candidate support:

```math
c_j = max(c_j, sum_p A_{j,p})
s_j = s_j + w_c sum_p A_{j,p}
```

This is not a hard label.  A noisy proposal can be absorbed by `B_p`; an anchor
cannot become posterior truth unless the normal AQR/posterior update accepts it.

## 4. Current Code Deployment

New/updated contracts:

```text
src/openpi/picf/core/contracts.py
  PicfAnchorPriorGraphState.object_candidate_assignment
  PicfAnchorPriorGraphState.object_candidate_coverage
  PicfAnchorPriorGraphState.object_candidate_background
  PicfAnchorPriorGraphState.object_candidate_duplicate_overlap

  PicfObjectExplanationState.candidate_coverage
  PicfObjectExplanationState.candidate_background
  PicfObjectExplanationState.candidate_duplicate_overlap
```

New/updated config:

```text
src/openpi/picf/core/config.py
  object_candidate_assignment_enabled
  object_candidate_assignment_temperature
  object_candidate_background_prior
  object_candidate_background_quality_weight
  object_candidate_row_support_floor
  object_candidate_eligible_roles
  object_candidate_max_rows_per_candidate
  object_candidate_row_capacity
  object_candidate_row_capacity_iters
  object_candidate_point_weight
  object_candidate_proposal_weight
  object_candidate_seed_weight
  object_candidate_task_owner_weight
  object_candidate_anchor_score_weight
  object_candidate_point_mix
  object_candidate_proposal_mix
  object_candidate_min_shape_quality
```

Current defaults:

```text
object_candidate_assignment_temperature = 0.35
object_candidate_background_prior = 0.25
object_candidate_background_quality_weight = 2.0
object_candidate_row_support_floor = 0.01
object_candidate_eligible_roles = (1, 2)
object_candidate_max_rows_per_candidate = 2
object_candidate_row_capacity = 1.25
object_candidate_row_capacity_iters = 10
```

The row-support floor is deliberately above numerical epsilon.  A7 exposed real
task/contact point support around `0.05`, so that support remains valid, while
near-zero random rows do not become candidate owners.

The 2026-05-19 v2 A7 diagnostic exposed a second-order failure after the
runtime-scale background repair:

```text
object candidates reached the model:
  coverage_mean rose to ~0.99
  background_mean fell to ~0.005

but raw same-role duplication returned:
  aqr_object_candidate_duplicate_overlap_max ~ 0.999
  posterior_file_competition_duplicate_overlap_max ~ 0.98
  aqr_same_role_support_overlap_max rose from ~0.16 to ~0.87
```

This means the sidecar proposal was no longer swallowed by the background, but
the assignment still had only candidate-column conservation.  Multiple raw
same-role rows could claim the same object candidate before posterior file
competition demoted them.  That contaminates routing, binding, and anchor-PV
even if active posterior duplicate overlap is controlled later.

The v3 repair therefore adds candidate-to-row capacity:

```text
top-k candidate ownership:
  by default, each candidate keeps only the strongest physical row owner.

soft row capacity:
  an optional Sinkhorn-style row-mass scaling prevents one row from eating too
  many candidates if top-k is relaxed for ablations.
```

This is not a hard semantic label.  It is the same object-centric assignment
invariant used by SlotAttention/MetaSlot-style systems: each measurement token
or mask is explained by one or a small mixture of object slots, with an explicit
background/no-object residual for invalid candidates.  Dense V-JEPA/point/tactile
tokens remain available to AQR; only the sidecar object-candidate measurement is
capacity-normalized.

New runtime method:

```text
src/openpi/picf/core/pipeline.py
  _proposal_object_candidate_assignment(...)
```

Dataflow:

```text
PicfObservation.proposal_mask_*
-> PicfPseudoProposalState.mask_*
-> _proposal_to_point_matrix(...)
-> _proposal_object_candidate_assignment(...)
-> object_candidate_assignment / coverage / background
-> mixed point_priors and proposal_priors
-> anchor_scores / anchor_conf
-> active/context/reserve selection
-> OEML object/background explanation debug
-> posterior update
```

New debug metrics:

```text
aqr_object_candidate_assigned_row_count
aqr_object_candidate_assigned_candidate_count
aqr_object_candidate_assignment_max
aqr_object_candidate_coverage_mean
aqr_object_candidate_coverage_max
aqr_object_candidate_background_mean
aqr_object_candidate_duplicate_overlap_max
oeml_candidate_coverage_mean
oeml_candidate_coverage_max
oeml_candidate_background_mean
oeml_candidate_duplicate_overlap_max
```

## 5. What This Fixes

This directly addresses the observed failure:

```text
green/task proposal selected
but active anchors do not bind to the object
```

because a selected proposal now has to be explained by a physical object row or
by background.  It no longer only nudges point priors.

It also addresses same-role duplication more directly than another scalar
overlap penalty: duplicate rows that explain the same proposal are stopped at
the object-candidate assignment itself through top-k ownership and row capacity.
This is upstream of posterior file competition, so routing/binding losses do not
first receive duplicate raw object supports and only later try to clean them up.

## 6. What This Still Does Not Claim

This is still not a full RGB generative slot decoder.  PICF intentionally keeps
V-JEPA/point/tactile dense memories intact and uses object candidates as
measurement evidence for posterior correction.  It does not:

```text
- replace V-JEPA dense tokens with proposal masks;
- hard-label a sidecar as object truth;
- assume SAM masks are valid;
- guarantee fourth-from-left ordinal grounding;
- guarantee behavior acceptance without 1000-step and long-run evidence.
```

## 7. Acceptance Gates

Short diagnostic must show:

```text
owm_proposal_tokens > 0 when sidecar root is enabled
aqr_object_candidate_assigned_candidate_count > 0
aqr_object_candidate_coverage_mean > 0
aqr_object_candidate_background_mean not near 1 for high-quality task proposals
aqr_object_candidate_duplicate_overlap_max not saturating
active same-role support/object-core overlap remains low
posterior_file_competition_active_duplicate_overlap_max remains low
raw/reserve duplicate metrics are explicitly tracked and must not be mistaken
  for active-path health
anchor overlays place active object rows inside/near task proposal masks
```

If these fail, the module is not accepted, regardless of action loss.

## 8. Verification Commands

```bash
python -m py_compile \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/pipeline.py \
  scripts/picf_core_train.py

PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
python scripts/picf_object_candidate_slot_binding_audit.py
```

Local verification on 2026-05-19:

```text
py_compile:
  PASS for contracts/config/pipeline/trainer/verifier

verify_picf_owm_contract.py:
  PASS, including sidecar_masks_route_through_object_candidate_slot_assignment

picf_object_candidate_slot_binding_audit.py:
  PASS for row-specific support, slot/background mass conservation, duplicate
  explanation visibility, mask-to-point support transport, runtime-scale
  background absorption regression, candidate top-1 de-duplication, soft row
  capacity, and the task-quality no-clone guard

picf_owm_strict_diagnose.py --fail-on-fail:
  PASS for static/code-level checks; runtime artifact checks are only evaluated
  when metrics/eval paths are provided.

uv run pytest -q scripts/picf_contact_motion_sidecar_precompute_test.py \
  scripts/picf_owm_same_object_probe_test.py:
  5 passed
```

Expected early-run metric additions:

```text
aqr_object_candidate_*
oeml_candidate_*
```

## 9. Current A7 1000-Step Diagnostic

Launched on 2026-05-19:

```text
host:
  A7, ssh -p 28060 root@36.139.225.68

tmux:
  picf_a7_object_candidate_anchor1000_20260519

script:
  run_a7_object_candidate_anchor1000_20260519.sh

output:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_object_candidate_anchor1000_20260519

log:
  /mnt/picf_run_logs/picf_a7_object_candidate_anchor1000_20260519.log
```

Run contract:

```text
1000 optimizer steps
2 GPUs, FSDP full shard
anchor_only trainable scope
PaliGemma / V-JEPA / AnyTouch / Sonata frozen
action loss disabled
unroll_steps=2
burnin_steps=1
contact-motion sidecar root:
  /mnt/picf_sidecars/contact_motion_mask_1000_20260518
object_candidate_assignment_enabled=True
proposal reference-anchor seed disabled
log_interval=50
anchor_overlay_interval=100
```

Operator commands:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t picf_a7_object_candidate_anchor1000_20260519

tail -f /mnt/picf_run_logs/picf_a7_object_candidate_anchor1000_20260519.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_object_candidate_anchor1000_20260519/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a7_object_candidate_anchor1000_20260519/anchor_overlays
```

Interpretation boundary:

```text
PASS:
  object-candidate metrics become nonzero and overlays show active rows near
  task/contact proposal masks without duplicate saturation.

FAIL:
  sidecar proposals are present but object_candidate coverage stays zero, or
  active same-role overlap and anchor-PV rise together while overlays miss the
  task object.

NOT CLAIMED:
  This 1000-step run is not action/CALVIN behavior acceptance and does not
  validate long-run co-training by itself.
```

Step-100 diagnosis from the first A7 run:

```text
owm_proposal_tokens:
  nonzero (~1.8-1.9), so sidecar proposals reached the model.

aqr_proposal_support_max:
  high (~0.91-0.99), so proposal tokens were readable.

aqr_proposal_point_bridge_max:
  weak (~0.05), which is expected for sparse projected 3D points.

aqr_object_candidate_coverage_mean:
  collapsed from ~0.0057 to ~0.00069.

aqr_object_candidate_background_mean:
  rose from ~0.994 to ~0.999.
```

Root cause:

```text
The old log-product score treated weak point-overlap as strong negative
evidence.  With tau=0.35, a positive but small point-overlap such as 0.05
contributed log(0.05)/0.35, making the fixed background prior dominate even for
task/contact proposals.  This was a math/modeling error, not a missing sidecar
dataflow.
```

Deployed repair:

```text
Use positive additive source scores normalized per candidate across physical
rows, keep the row-specific guard, and make background quality-adaptive:

  E_jp = weighted positive evidence
  bg_logit_p = log(background_prior) - beta * candidate_quality_p

This preserves invalid-candidate background absorption while preventing real
task/contact candidates from disappearing because one auxiliary source is
numerically small.
```

Next steps after the 1000-step gate:

```text
1. If object-candidate metrics and overlays pass, run a matched 30000-step
   long train with action enabled and frozen foundation backbones except the
   intended trainable connectors/PaliGemma policy stack.

2. If object-candidate metrics fail, do not add another scalar overlap penalty.
   Inspect whether the failure is proposal quality, proposal-to-point transport,
   row-specific support, or posterior active-file selection.

3. Keep blind SAM archived. Only contact/task/tracklet-aware sidecars are
   allowed to enter this object-candidate path.
```

## 10. Runtime-Scale Repair Rerun

The first A7 run is stopped after confirming the background-collapse root cause.
The repaired run reuses the same script with an override:

```bash
EXP=picf_a7_object_candidate_anchor1000_v2_20260519 \
  bash run_a7_object_candidate_anchor1000_20260519.sh
```

New v2 gate:

```text
aqr_object_candidate_coverage_mean:
  should be clearly above the first run's ~0.0007 background-collapse level.

aqr_object_candidate_background_mean:
  should not remain near 0.999 for high-quality task/contact proposals.

aqr_object_candidate_assignment_max:
  should become visible without duplicate saturation.

aqr_object_candidate_duplicate_overlap_max:
  may rise if two rows compete for one proposal; this is diagnostic and should
  not saturate into symmetric cloning.
```

## 11. V3 Row-Capacity 200-Step Diagnostic

Run:

```text
host:
  A7, ssh -p 28060 root@36.139.225.68

tmux:
  picf_a7_object_candidate_capacity_anchor200_20260519

output:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_object_candidate_capacity_anchor200_20260519

scope:
  anchor_only
  action loss disabled
  PaliGemma / V-JEPA / AnyTouch / Sonata frozen
  sidecar root = /mnt/picf_sidecars/contact_motion_mask_1000_20260518
```

Compact metrics:

```text
step 50:
  object_candidate_coverage_mean=0.8107
  object_candidate_background_mean=0.1893
  object_candidate_duplicate_overlap_max=0.0000
  same_role_support_overlap_max=0.1701
  active_same_role_support_overlap_max=0.0202
  same_role_object_core_overlap_max=0.1176
  active_same_role_object_core_overlap_max=0.0528
  posterior_file_competition_duplicate_overlap_max=0.9559
  posterior_file_competition_active_duplicate_overlap_max=0.0000
  loss_anchor_pv=0.7932
  loss_aqr_denoising=1.4233
  loss_mapg_routing=0.5873

step 100:
  object_candidate_coverage_mean=0.9838
  object_candidate_background_mean=0.0162
  object_candidate_duplicate_overlap_max=0.0000
  same_role_support_overlap_max=0.2893
  active_same_role_support_overlap_max=0.0450
  same_role_object_core_overlap_max=0.4596
  active_same_role_object_core_overlap_max=0.0740
  posterior_file_competition_duplicate_overlap_max=0.9542
  posterior_file_competition_active_duplicate_overlap_max=0.0000
  loss_anchor_pv=1.1670
  loss_aqr_denoising=2.0472
  loss_mapg_routing=0.6457

step 150:
  object_candidate_coverage_mean=0.9812
  object_candidate_background_mean=0.0188
  object_candidate_duplicate_overlap_max=0.0000
  same_role_support_overlap_max=0.4017
  active_same_role_support_overlap_max=0.0555
  same_role_object_core_overlap_max=0.7009
  active_same_role_object_core_overlap_max=0.0226
  posterior_file_competition_duplicate_overlap_max=0.9858
  posterior_file_competition_active_duplicate_overlap_max=0.0000
  loss_anchor_pv=1.1406
  loss_aqr_denoising=2.1036
  loss_mapg_routing=0.6561

step 200:
  object_candidate_coverage_mean=0.9700
  object_candidate_background_mean=0.0300
  object_candidate_duplicate_overlap_max=0.0000
  same_role_support_overlap_max=0.5799
  active_same_role_support_overlap_max=0.0515
  same_role_object_core_overlap_max=0.7510
  active_same_role_object_core_overlap_max=0.0186
  posterior_file_competition_duplicate_overlap_max=0.9865
  posterior_file_competition_active_duplicate_overlap_max=0.0000
  loss_anchor_pv=1.1811
  loss_aqr_denoising=2.0355
  loss_mapg_routing=0.6591
```

Judgment:

```text
FIXED:
  The object-candidate assignment bug is fixed.  Coverage remains high,
  background no longer swallows high-quality proposals, and direct candidate
  cloning is held at 0.0 for the full 200-step diagnostic.

FIXED FOR ACTIVE ACTION PATH:
  active_same_role_support_overlap_max stays around 0.02-0.06;
  active_same_role_object_core_overlap_max stays around 0.02-0.08;
  posterior_file_competition_active_duplicate_overlap_max stays 0.0.

NOT FIXED:
  raw same-role support/object-core overlap continues to rise;
  raw posterior_file_competition_duplicate_overlap_max remains saturated;
  loss_anchor_pv / denoising / routing are worse than at step 50.

INTERPRETATION:
  v3 repairs the sidecar-candidate ownership invariant, but it does not remove
  duplicate reserve/raw files.  The current active-file gates prevent those
  duplicates from entering the active path, but raw losses and diagnostics still
  see them.  This run is therefore a partial acceptance, not a final long-train
  acceptance.
```

Required follow-up:

```text
1. Do not start a 30000-step train based only on this diagnostic if the target
   is clean raw object partitioning.

2. Decide whether raw/reserve duplicate files should be:
   a. excluded from losses that are meant to evaluate active object binding; or
   b. given an explicit reserve/no-object objective so their duplication is not
      treated as object failure; or
   c. suppressed earlier by a posterior-file capacity mechanism.

3. Keep the v3 object-candidate repair.  It fixes a real upstream assignment
   bug and should not be reverted.
```

## 12. V4 Active-Object Loss-Scope Repair

The v3 diagnostic separated two facts that were previously conflated:

```text
active object path:
  healthy after candidate top-k ownership and row capacity.

raw/reserve telemetry:
  still high-overlap because fixed-capacity systems keep reserve/no-object rows.
```

The remaining failure is therefore not that dense evidence is invisible.  It is
that some object-level losses still treat reserve/context rows as if they were
active object files.

The mathematically correct fixed-capacity decomposition is:

```math
Q =
Q^{obj}
\cup
Q^{ctx}
\cup
Q^{null}
```

where:

```text
Q_obj:
  active object files allowed to claim object candidates and train object losses.

Q_ctx:
  low-weight context/reserve files allowed to carry peripheral evidence.

Q_null:
  no-object/background capacity.
```

Object-centric losses should be evaluated on `Q_obj`; dense PV/background
coverage is preserved by the global floor and `pv_weak`, not by forcing every
reserve row to become an object.

This matches the SlotAttention/MetaSlot invariant: extra slots may exist, but
they are not all valid object explanations.  Duplicate no-object/context rows
are not the same failure as duplicate active object rows.

Implementation:

```text
src/openpi/picf/core/training.py:
  _active_object_row_weight(...)
  anchor_pv_active_object_gate_only=True
  aqr_denoising_active_object_only=True
  object_explanation_active_object_only=True

scripts/picf_core_train.py:
  --anchor-pv-active-object-gate-only / --no-anchor-pv-active-object-gate-only
  --aqr-denoising-active-object-only / --no-aqr-denoising-active-object-only
  --object-explanation-active-object-only / --no-object-explanation-active-object-only

scripts/picf_object_candidate_slot_binding_audit.py:
  active_object_scope_ignores_reserve_duplicates
  downstream_weight_fallback_excludes_context_rows
  denoising_active_object_scope_excludes_no_object_peaks
```

The repair is intentionally not a deletion:

```text
kept:
  dense V-JEPA/PG/point/tactile evidence
  reserve/context rows
  background/no-object residual
  raw telemetry for debugging

changed:
  object PV gate
  AQR support denoising
  OEML duplicate/feature/point object terms

rule:
  these object-level losses now use active object files only.
```

Acceptance expectation:

```text
active same-role support overlap:
  should remain low.

active same-role object-core overlap:
  should remain low.

raw duplicate telemetry:
  may remain high if reserve/context rows carry similar no-object evidence.

loss_anchor_pv / loss_aqr_denoising:
  should no longer be pulled upward by reserve/no-object rows with sharp but
  non-object support.
```

## 13. V4 Active-Object Loss-Scope 200-Step Diagnostic

Run:

```text
picf_a7_active_object_scope_anchor200_20260519
remote metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_active_object_scope_anchor200_20260519/metrics.jsonl
remote overlays:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_active_object_scope_anchor200_20260519/anchor_overlays
scope:
  anchor-only, action/PaliGemma frozen, sidecar enabled, SAM disabled
window:
  unroll_steps=2, burnin_steps=1
```

Observed metrics:

```text
step  coverage  background  candidate_dup  active_support  active_core  raw_support  raw_core  anchor_pv  denoise  routing  grad_preclip
  50    0.8635      0.1365        0.0000          0.0145       0.1363       0.1545   0.1666     0.7839   1.3174   0.6093       15.93
 100    0.9272      0.0728        0.0000          0.0215       0.0644       0.5125   0.6722     1.1843   1.9060   0.6325       14.59
 150    0.9732      0.0268        0.0000          0.0508       0.0127       0.9854   0.9341     1.1729   1.9695   0.6758       25.05
 200    0.9670      0.0330        0.0000          0.1115       0.0203       0.9973   0.9393     1.1778   1.8828   0.7078       66.21
```

Posterior file competition:

```text
active_duplicate_overlap_max:
  0.0 throughout steps 50/100/150/200

raw_duplicate_overlap_max:
  0.9460 -> 0.9782 -> 0.9830 -> 0.9770

recycle_rate:
  0.0502 -> 0.00047 -> 0.00298 -> 0.00035

identity_switch_rate:
  0.657 -> 0.655 -> 0.691 -> 0.683
```

Verdict:

```text
PASS:
  object-candidate coverage remains high;
  candidate duplicate overlap stays zero;
  active same-role support/object-core overlap stays low;
  active posterior duplicate overlap stays zero;
  recycle no longer saturates.

FAIL / STILL OPEN:
  raw support/object-core duplicate telemetry still saturates through
  reserve/context rows;
  loss_anchor_pv rises from the step-50 value and plateaus around 1.18;
  loss_aqr_denoising rises from 1.32 to roughly 1.9;
  loss_mapg_routing drifts upward from 0.61 to 0.71;
  preclip gradient norm grows sharply by step 200.
```

Mathematical interpretation:

```text
V4 correctly scopes object-level losses to active object rows, but the residual
failure is not solely "reserve rows were included in the object loss".

The active rows themselves still see inconsistent pressure between:
  1. object-candidate/proposal ownership, which wants compact task-object mass;
  2. anchor-PV/projective matching, which can still reward broader visual/point
     correspondence through the current object gate and PV floor;
  3. AQR denoising, which can train sharp typed-support peaks even when the
     candidate-to-point/visual support is not yet a stable object explanation.

Therefore V4 is a necessary loss-domain repair, not the complete root fix for
anchor_pv/denoising/routing drift.
```

Relation to slot literature:

```text
SlotAttention-style systems use competition so an input element is explained by
one or a small mixture of slots, not cloned by every extra slot.

MetaSlot and 2026 slot-curriculum/slot-merging work highlight the same failure
mode in a different form: fixed overcomplete slot capacity tends to fragment or
duplicate object explanations unless active capacity and object evidence are
controlled.

The 2025 IsSameObject result supports using a pairwise binding subspace, but it
does not remove the need for correct candidate ownership and loss scope.
```

Required next repair:

```text
Do not revert V3/V4.

Next isolate anchor-PV and denoising target mismatch:
  1. split active-object anchor-PV from global PV weak coverage;
  2. remove or sharply lower the object-gate floor inside the active object
     anchor-PV term, while keeping dense background coverage in loss_pv_weak;
  3. keep AQR denoising disabled by default for production, or scope it to
     object-candidate/proposal/point-confirmed active rows only during the
     anchor-only diagnostic;
  4. continue to report raw duplicate telemetry, but do not use it as the
     active-object acceptance metric.
```

## 14. V5 Object-PV / Dense-PV Split

The V4 validation falsified the hypothesis that reserve-row loss contamination
was the only remaining cause.  V5 therefore moves the split one level deeper:

```text
object PV:
  active object rows only;
  no object-loss floor;
  normalized over object-confirmed edges.

dense PV:
  all dense projective coverage;
  remains in loss_pv_weak;
  does not define object identity.
```

Implementation:

```text
src/openpi/picf/core/training.py:
  anchor_pv_object_gate_floor = 0.0
  anchor_pv_object_normalize_by_object_mass = True
  aqr_denoising_confirmed_object_only = True
  aqr_denoising_confirmation_threshold = 0.05
  _confirmed_object_row_weight(...)

scripts/picf_core_train.py:
  --anchor-pv-object-normalize-by-object-mass
  --aqr-denoising-confirmed-object-only
  --aqr-denoising-confirmation-threshold

scripts/picf_object_candidate_slot_binding_audit.py:
  denoising_confirmed_object_scope_excludes_unconfirmed_active_rows
  object_pv_normalizes_by_confirmed_object_mass_not_dense_floor
```

Mathematically:

```math
L_{pv}^{obj}
=
\frac{
  \sum_{p,v} w_{p,v}^{obj}\operatorname{BCE}(r_{p,v}, y_{p,v})
}{
  \sum_{p,v} w_{p,v}^{obj}
}
```

```math
L_{pv}^{weak}
=
\mathbb{E}_{v}
\operatorname{CE}
\left(
  v
  \mid
  \operatorname{bag}_{p\sim y_{p,v}} f_p
\right)
```

The object floor is deliberately zero because a nonzero floor reintroduces the
same mixed target: active object rows again receive gradients from edges that
are not confirmed object evidence.  Global dense coverage is not lost; it is
carried by `loss_pv_weak`.

Denoising:

```math
L_{dn}
=
\sum_j
1[j \in Q_{obj}]
1[j \text{ has candidate/proposal/point confirmation}]
CE(p_j,\operatorname{argmax} stopgrad(p_j)).
```

This preserves the guarded-teacher semantics: a sharp support peak is not a
teacher unless the row is also confirmed by object evidence.

### 14.1 A7 V5 50-step validation snapshot

Run:

```text
picf_a7_v5_object_pv_split_anchor200_20260519
```

This is an in-progress 200-step frozen action/PaliGemma anchor diagnostic.  At
step 50, the V5 dataflow is live and the active-object selection invariants are
healthy:

```text
aqr_object_candidate_coverage_mean             = 0.8477
aqr_object_candidate_duplicate_overlap_max     = 0.0000
aqr_active_same_role_support_overlap_max       = 0.0121
aqr_active_same_role_object_core_overlap_max   = 0.1137
aqr_same_role_support_overlap_max              = 0.1929
owm_proposal_tokens                            = 1.94
owm_tracklet_tokens                            = 75.16
```

This means the previous active-slot failure is not reproduced at step 50:
confirmed object rows are selected, same-role active rows are separated, and
sidecar proposal/tracklet evidence is present.

However, step 50 does not validate object-PV convergence:

```text
loss_anchor_pv              = 5.3087
loss_aqr_denoising          = 1.4504
loss_mapg_routing           = 0.5952
posterior_identity_switch   = 0.6400
preclip_grad_norm           = 18.2694
```

This high `loss_anchor_pv` is not directly comparable to the V4 value because
V5 removed the background/dense floor from object-PV.  The metric is now a
true object-edge alignment loss rather than a background-averaged mixed loss.
The correct acceptance criterion is therefore the 50/100/150/200 trend:

```text
PASS:
  active overlap remains low;
  loss_anchor_pv trends down or stabilizes without gradient spikes;
  denoising/routing do not drift upward;
  posterior identity switch falls as object rows stabilize.

FAIL:
  active overlap stays low but loss_anchor_pv/denoising/routing keep rising;
  this would mean the object rows are selected correctly but the object-edge
  teacher is still geometrically or semantically misaligned.
```

### 14.2 A7 V5 100-step result, invalidated by launch-script override

The first A7 V5 run reached step 100, but the run is invalid as a pure V5
test.  Follow-through found that `run_a7_object_candidate_anchor1000_20260519.sh`
still passed:

```text
--anchor-pv-object-gate-floor 0.25
```

This overrode the V5 default `0.0` and reintroduced exactly the dense/background
floor that V5 is meant to remove.  The observed numbers are still useful as a
negative control:

```text
step                                50        100
candidate coverage mean             0.8477    0.9770
candidate duplicate overlap max      0.0000    0.0000
active same-role support overlap     0.0121    0.0569
active object-core overlap           0.1137    0.2149
raw same-role support overlap        0.1929    0.5921
raw object-core overlap              0.1391    0.4706
loss_anchor_pv                       5.3087    8.5283
loss_aqr_denoising                   1.4504    2.0182
loss_mapg_routing                    0.5952    0.6574
posterior_identity_switch_rate       0.6400    0.5933
preclip_grad_norm                    18.2694   183.2092
```

Conclusion:

```text
This run does not reject V5.  It rejects the old mixed-floor launch profile.

The selected active rows remain non-duplicated and mostly separated, while
object-PV and denoising become worse and the unclipped gradient explodes.  The
cause is consistent with the stale floor reintroducing dense/projective edges
into the object loss.
```

Corrective action:

```text
restart with:
  --anchor-pv-object-gate-floor 0.0
  --anchor-pv-object-normalize-by-object-mass
  --aqr-denoising-confirmed-object-only
  --aqr-denoising-confirmation-threshold 0.05
```

### 14.3 Pure V5 step-50 result

Run:

```text
picf_a7_v5_pure_object_pv_split_anchor200_20260519
```

This run explicitly verified the launch flags:

```text
--anchor-pv-object-gate-floor 0.0
--anchor-pv-object-normalize-by-object-mass
--aqr-denoising-confirmed-object-only
```

Step-50 comparison:

```text
metric                                stale floor=0.25   pure V5 floor=0.0
loss_total                            1.6550             1.0692
loss_alignment                        1.5800             0.9942
loss_alignment_raw                    1.4474             0.8606
loss_anchor_pv                        5.3087             2.9648
loss_aqr_denoising                    1.4504             1.3502
loss_mapg_routing                     0.5952             0.5908
loss_mapg_cycle                       0.3525             0.3601
preclip_grad_norm                     18.2694            9.2125
active same-role support overlap      0.0121             0.0164
active object-core overlap            0.1137             0.1054
raw same-role support overlap         0.1929             0.1513
candidate coverage mean               0.8477             0.7322
candidate duplicate overlap max       0.0000             0.0000
```

Interpretation:

```text
The floor override was the immediate cause of the step-100 deterioration in the
invalid run.  Pure V5 keeps the same healthy active-row separation while
substantially lowering object-PV, alignment, and preclip gradient at step 50.

The repair is not yet fully accepted: it still must hold through steps
100/150/200.  The required trend is stable or falling anchor-PV/denoising with
active overlap staying low.
```

### 14.4 Pure V5 step-100 result

Step-100 result:

```text
step                                  50        100
loss_total                            1.0692    1.2123
loss_alignment                        0.9942    1.1373
loss_alignment_raw                    0.8606    1.0179
loss_anchor_pv                        2.9648    3.7491
loss_pv_weak                          5.9687    4.0321
loss_aqr_denoising                    1.3502    1.9225
loss_mapg_routing                     0.5908    0.6852
loss_mapg_cycle                       0.3601    0.3973
loss_slot_jepa                        1.2275    1.1771
active same-role support overlap      0.0164    0.1335
active object-core overlap            0.1054    0.1614
raw same-role support overlap         0.1513    0.5884
raw object-core overlap               0.1656    0.6018
posterior_identity_switch_rate        0.6356    0.6444
posterior_recycle_rate                0.0646    0.0075
preclip_grad_norm                     9.2125    61.4807
```

Conclusion:

```text
Pure V5 is materially better than the stale-floor run, but still not accepted.
The active object rows remain much healthier than the raw support population,
yet anchor-PV/denoising/routing/cycle and preclip gradient rise after step 50.
```

## 15. V6 Distributional Object-PV, Deployed 2026-05-19

Pure V5 proved that the active object selector is no longer the main blocker:
confirmed active rows stay much healthier than raw/reserve rows.  The remaining
failure is the form of `loss_anchor_pv` itself.  Even with an object gate, the
old implementation was still a dense edge BCE:

```math
L_{legacy}
=
\frac{\sum_{p,u} w^{obj}_{p,u} C_{p,u}
  BCE(R_{p,u}, C_{p,u})}
{\sum_{p,u} w^{obj}_{p,u} C_{p,u}}
```

where `R` is observation-anchor point/visual co-routing and `C` is projective
compatibility.  This is not a clean object-slot loss: it asks object files to
match dense projective edges rather than asking each object file whether its own
point and visual supports describe the same object.

V6 replaces the object part with per-row distributional consistency:

```math
\hat v_j = normalize(p_j C)
```

```math
\hat p_j = normalize(v_j C^T)
```

```math
L^{obj}_{pv}
=
\frac{\sum_j w_j
\left[
  JS(v_j, \hat v_j) + JS(p_j, \hat p_j)
\right] / 2}
{\sum_j w_j}
```

where:

```text
p_j:
  object row j point support distribution.

v_j:
  object row j visual support distribution.

C:
  point-to-visual projective compatibility, masked by valid projective edges.

w_j:
  active object row weight, optionally requiring object-candidate/proposal/point
  confirmation.
```

This matches the invariant used by slot/object-centric systems: a slot explains
its own evidence distribution; dense background evidence remains in a separate
reconstruction/weak objective.  In PICF, dense/background PV remains in
`loss_pv_weak`; reserve/context rows remain visible to AQR and the action prefix
but are not treated as object identity labels.

Implementation:

```text
src/openpi/picf/core/training.py
  _object_projective_distribution_loss(...)
  PicfTransitionLossConfig.anchor_pv_object_distribution_loss = True
  PicfAlignmentLossConfig.anchor_pv_object_distribution_loss = True

scripts/picf_core_train.py
  --anchor-pv-object-distribution-loss
  --anchor-pv-object-distribution-confirmed-only
  --anchor-pv-object-distribution-confirmation-threshold

run_a7_object_candidate_anchor1000_20260519.sh
  explicitly enables V6 and keeps object gate floor = 0.0
```

Local verification:

```text
py_compile:
  PASS

scripts/picf_object_candidate_slot_binding_audit.py:
  adds distributional_object_pv_penalizes_slot_projective_mismatch
  PASS

scripts/picf_anchor_pv_object_gate_audit.py:
  adds distributional_object_pv_replaces_dense_edge_bce
  PASS

scripts/verify_picf_owm_contract.py:
  PASS, including V6 distributional object-PV contract
```

Acceptance criterion for the next 200-step diagnostic:

```text
loss_anchor_pv:
  should no longer rise because raw dense edges or reserve rows disagree with
  the active object row.

loss_aqr_denoising / loss_mapg_routing:
  may still be diagnostics unless explicitly weighted, but should not show the
  previous step50->step100 gradient spike.

active same-role support overlap:
  must remain low.

raw same-role support overlap:
  may be higher because reserve/context rows are allowed to carry background;
  it is not the acceptance metric for object identity.
```

## 16. V6 A7 200-Step Runtime Result, 2026-05-19

Run:

```text
picf_a7_v6_distribution_object_pv_anchor200_20260519
```

Scope:

```text
anchor_only
world_size=2
unroll_steps=2
burnin_steps=1
action loss weight = 0
PaliGemma / Sonata / V-JEPA / AnyTouch frozen
sidecar proposals and tracklets enabled
```

Result:

```text
step                   50        100       150       200
loss_total             0.4732    0.4154    0.3679    0.3563
loss_alignment         0.3982    0.3404    0.2929    0.2813
loss_alignment_raw     0.2622    0.2213    0.1973    0.1952
loss_anchor_pv         0.5684    0.5735    0.5419    0.5516
loss_pv_weak           6.0047    3.8959    3.0909    2.8625
loss_aqr_denoising     1.4723    1.8976    1.9533    1.9169
loss_mapg_routing      0.6080    0.6477    0.6706    0.6746
loss_mapg_cycle        0.3547    0.3826    0.3962    0.4016
active_support_overlap 0.0125    0.0735    0.0965    0.0986
active_object_overlap  0.1286    0.2871    0.1001    0.0620
raw_support_overlap    0.1805    0.3253    0.7740    0.8881
raw_object_overlap     0.1720    0.4739    0.6552    0.6857
candidate_coverage     0.8671    0.9751    0.9794    0.9826
candidate_dup_overlap  0.0000    0.0000    0.0000    0.0000
posterior_recycle      0.0814    0.0234    0.0009    0.0006
preclip_grad_norm      12.80     2.95      4.34      16.58
```

Interpretation:

```text
ACCEPTED for the object-PV root cause.

V5 failure:
  loss_anchor_pv rose from 2.9648 to 3.7491 by step 100 and gradients spiked.

V6 behavior:
  loss_anchor_pv stays low and roughly flat, while loss_total, alignment,
  alignment_raw, and pv_weak all decrease through step 200.

Therefore:
  the dense-edge object-PV objective was the correct root-cause target.
  The distributional object-PV replacement fixes the specific mathematical
  mismatch that made object rows chase dense/background projective edges.
```

Remaining non-acceptance signals:

```text
raw same-role support overlap rises to 0.8881.

This is not the same as active object collapse:
  active_support_overlap remains below 0.10;
  active_object_overlap falls to 0.0620 by step 200;
  candidate duplicate overlap remains 0.0.

The raw metric includes reserve/context/background carriers, so it should be
tracked as a background-capacity diagnostic, not as the primary object-binding
acceptance metric.

loss_aqr_denoising and loss_mapg_routing still rise mildly.  In this diagnostic
their configured optimization weights are zero, so they are health diagnostics
rather than current training pressure.  They should not be enabled as losses
until object-PV and active binding remain stable in a longer run.
```

Overlay check:

```text
step_000100 active_only:
  active anchors cluster around the red-block / drawer interaction region.

step_000200 active_only:
  active anchors still cover the task-relevant region, with a few task/context
  anchors on the drawer front/side.  This is acceptable for a 200-step frozen
  anchor-only diagnostic, but it is not proof of final semantic grounding.

with_gray:
  gray reserve/context rows remain numerous and overlap background/side regions.
  This matches the raw-overlap metric and must not be confused with active
  object slot collapse.
```

Decision:

```text
Use V6 as the next candidate profile.

Do not revert to V5.
Do not re-enable legacy dense-edge object-PV.
Do not optimize aqr_denoising / mapg_routing yet.

Next valid test:
  full training profile with action/PaliGemma unfrozen as planned, but keep the
  V6 object-PV objective and continue writing active-only / with-gray /
  sidecar-proposal anchor overlays.
```

Updated root cause:

```text
1. The launch-script floor bug was real and fixed.
2. The remaining failure is not primarily active-row selection.
3. The remaining failure is that the auxiliary alignment teachers still include
   unstable self-teacher or raw-support pressure after the active rows are
   selected.
```

Next design rule:

```text
Object identity losses should be externally confirmed object-edge objectives.
Self-denoising and raw graph/routing diagnostics must not become optimization
pressure until object-PV is stable.  This follows the same principle as modern
object-centric methods: slots compete for evidence, but unverified self-peaks
are not treated as labels.
```

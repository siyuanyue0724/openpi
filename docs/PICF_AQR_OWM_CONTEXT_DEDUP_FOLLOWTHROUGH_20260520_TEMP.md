# PICF-AQR-OWM Context Dedup Follow-Through

Date: 2026-05-20

Status: context-core dedup validated to step 100, failed at step 150 through
diffuse full-support duplication; support-aware context dedup implemented
locally; A7 validation `picf_a7_context_support_dedup_300_20260520` completed
300 steps.  The old duplicate-context failure is materially repaired, but the
run is not yet a 30000-step launch candidate because active/posterior object
ownership remains visually imperfect and the point/object-pull losses still
rise late in the run.

## Trigger

The robust OEML core validation fixed the previous unbounded point-tail failure:

```text
old loss_object_explanation_point:
  step 100 ~= 10.77
  step 200 ~= 20.27
  step 250 ~= 34.27

new loss_object_explanation_point:
  step 100 = 1.787
  step 200 = 3.647
  step 300 = 5.037
```

But the same run exposed a separate issue:

```text
aqr_active_same_role_support_overlap_max:
  stayed below 0.05

aqr_same_role_support_overlap_max:
  returned to about 0.93-0.99 by step 150-300

aqr_context_anchor_count:
  stayed around 22
```

This means the active posterior/object rows were not collapsing.  The high
raw overlap came from inactive reserve/context rows repeatedly exposing the
same object-like support to the control graph.

## Dataflow Boundary

PICF has three levels that must remain separate:

```text
dense typed memory:
  all V-JEPA / PaliGemma / point / tactile / sidecar evidence remains present.

AQR fixed query bank:
  overcomplete anchor capacity remains available for discovery and birth.

downstream control graph:
  should receive active object files and a small number of distinct context
  rows, not all duplicate reserve rows.
```

The previous code already removed context rows that duplicated active rows:

```math
C_0 =
\{i : a_i < 0.5,\ c_i \ge c_{min},\ s_i \ge s_{min},
\max_{j:a_j=1} O_{ij}<\tau_a\}
```

It did not remove context rows that duplicated each other.

2026-05-20 A7 context-core dedup result:

```text
run:
  picf_a7_context_dedup_300_20260520

step 50:
  aqr_context_anchor_count = 6.295
  aqr_context_same_role_support_overlap_max = 0.254
  aqr_downstream_same_role_support_overlap_max = 0.302
  loss_object_explanation_point = 0.915

step 100:
  aqr_context_anchor_count = 6.270
  aqr_context_same_role_support_overlap_max = 0.391
  aqr_downstream_same_role_support_overlap_max = 0.410
  loss_object_explanation_point = 3.350

step 150:
  aqr_context_same_role_object_core_overlap_max = 0.262
  aqr_downstream_same_role_object_core_overlap_max = 0.278
  aqr_context_same_role_support_overlap_max = 0.827
  aqr_downstream_same_role_support_overlap_max = 0.840
  loss_object_explanation_point = 6.077
  grad_clip_applied = true
```

Interpretation:

```text
object-core overlap was low, so the first dedup rule did its job;
full-support overlap was high, so diffuse context rows still duplicated each
other in the control graph.
```

Therefore the remaining repair is not another loss. It is a stricter
context-selector invariant:

```text
context rows must be distinct both as object-core owners and as downstream
support maps.
```

## Paper-Code Check

Local code references checked:

```text
temp/external_repos/MetaSlot/object_centric_bench/model/metaslot.py
temp/paper_code_20260518/AdaSlot/ocl/models/image_grouping_adaslot.py
temp/paper_code_20260518/slot-attention-video/savi/modules/attention.py
temp/external_repos/slot_refs_20260520/vit-object-binding/src/utils/models.py
temp/external_repos/slot_refs_20260520/vit-object-binding/src/utils/score.py
```

Relevant invariants:

```text
Slot Attention / SAVi:
  object slots are formed through competition over inputs, not by dumping every
  redundant slot downstream.

MetaSlot:
  fixed-slot OCL suffers duplicate slots; duplicate removal is a first-class
  mechanism when object count varies.

QASA:
  selection/quality should be decoupled from reconstruction/control pressure,
  avoiding a direct conflict between "use fewer slots" and "explain all tokens".

Object-Binding:
  same-object evidence is pairwise/quadratic and should support binding, not
  become a blind hard lock.
```

This repair adopts the compatible subset: quality-ordered duplicate suppression
of context rows.  It does not copy a visual VQ codebook into posterior truth,
does not add a full reconstruction decoder, and does not prune dense tokens.

## Implemented Math

Let:

```text
a_i     = active posterior/object indicator for row i
q_i     = deterministic object/context score
r_i     = role id
O_ij    = object-core overlap, maxed with geometry overlap when valid
S_ij    = full-support overlap over the current visual support map
tau_a   = active-duplicate threshold
tau_c   = context-context duplicate threshold
tau_s   = context-context full-support duplicate threshold
k_c     = max accepted context rows per role
lambda_c = context downstream weight
```

Candidate context rows:

```math
C_0 =
\{i : a_i < 0.5,\ c_i \ge c_{min},\ q_i \ge q_{min},
\max_{j:a_j=1} O_{ij}<\tau_a\}.
```

For each role:

```math
C_r=\{i\in C_0 : r_i=r\}.
```

Rows are sorted by `q_i`, then greedily selected:

```math
K_r =
\operatorname{GreedyNMS}(C_r, q_i, (O_{ij}, S_{ij}), (\tau_c, \tau_s), k_c).
```

A candidate is rejected if either condition holds:

```math
\max_{j\in K_r} O_{ij} > \tau_c
\quad\text{or}\quad
\max_{j\in K_r} S_{ij} > \tau_s.
```

The downstream weight is:

```math
w_i =
\begin{cases}
1, & a_i=1 \\
\lambda_c, & i\in \cup_r K_r \\
0, & \text{otherwise.}
\end{cases}
```

Important constraints:

```text
1. w_i only controls downstream graph visibility.
2. fixed AQR query capacity is not reduced.
3. dense typed evidence is not dropped.
4. posterior ownership remains authoritative.
5. sidecar masks remain weak measurement evidence.
```

## Code Changes

Configuration:

```text
PicfCoreConfig.aqr_context_slot_deduplicate_enabled = True
PicfCoreConfig.aqr_context_slot_max_per_role = 8
PicfCoreConfig.aqr_context_slot_self_overlap_threshold = 0.75
PicfCoreConfig.aqr_context_slot_self_support_overlap_enabled = True
PicfCoreConfig.aqr_context_slot_self_support_overlap_threshold = 0.70
```

Implementation:

```text
src/openpi/picf/core/pipeline.py::_aqr_downstream_slot_weights
```

Training CLI:

```text
scripts/picf_core_train.py
  --aqr-context-slot-deduplicate-enabled
  --aqr-context-slot-max-per-role
  --aqr-context-slot-self-overlap-threshold
  --aqr-context-slot-self-support-overlap-enabled
  --aqr-context-slot-self-support-overlap-threshold
```

New metrics:

```text
aqr_downstream_same_role_support_overlap_max
aqr_downstream_same_role_object_core_overlap_max
aqr_context_same_role_support_overlap_max
aqr_context_same_role_object_core_overlap_max
```

These are more diagnostic than raw `aqr_same_role_support_overlap_max`, because
raw overlap still includes reserve rows that are not visible downstream.

## Why This Is Not A Patch-On-Patch

The failure mode was not:

```text
active object files collapse.
```

It was:

```text
too many low-priority duplicate context rows stay visible.
```

Therefore the correct fix is not another object loss, not SAM, not a new
detector, not a different action loss, and not deleting dense context.  The
correct fix is a selector that preserves scene context while enforcing
object-distinctness before the control graph.

This is the same structural principle as recent adaptive-slot methods:

```text
overcomplete capacity internally;
quality/duplicate selection externally.
```

## Acceptance Criteria

For the next A7 300-step validation:

```text
aqr_active_same_role_support_overlap_max:
  remains low, preferably < 0.10.

aqr_downstream_same_role_support_overlap_max:
  must be lower than raw same-role overlap and should not sit at 0.95+.

aqr_context_same_role_support_overlap_max:
  should drop materially versus the robust OEML run.

aqr_context_anchor_count:
  should fall from about 22 to a smaller distinct set, but not to zero.

loss_object_explanation_point:
  should remain bounded; no return to 10+ by step 100.

loss_anchor_object_pull / loss_total:
  may still fluctuate because sidecar masks are weak evidence, but should not
  be driven by duplicate context exposure.
```

If downstream/context overlap remains high after this repair, the problem is no
longer simply "reserve rows are visible" or "object-core metric misses diffuse
duplicates"; the next suspects are role definition or the upstream support map
itself.

## Active Validation: 2026-05-20 A7 Support-Aware Dedup

Run:

```text
picf_a7_context_support_dedup_300_20260520
```

Mode:

```text
slot-comprehensive frozen-policy validation, not anchor_only.
picf-trainable-scope = all
action loss = 0
PaliGemma/action pressure frozen
PICF slot/router/posterior/OEML/context selector trainable
sidecar proposals and tracklet fields enabled
blind SAM disabled
```

Observed metrics:

```text
step 50:
  aqr_active_same_role_support_overlap_max = 0.0148
  aqr_context_same_role_support_overlap_max = 0.2982
  aqr_downstream_same_role_support_overlap_max = 0.3382
  raw aqr_same_role_support_overlap_max = 0.4713
  loss_object_explanation_point = 0.7746
  grad_clip_applied = false

step 100:
  aqr_active_same_role_support_overlap_max = 0.0239
  aqr_context_same_role_support_overlap_max = 0.2433
  aqr_downstream_same_role_support_overlap_max = 0.2692
  raw aqr_same_role_support_overlap_max = 0.8454
  loss_object_explanation_point = 2.6901
  grad_clip_applied = false

step 150:
  aqr_active_same_role_support_overlap_max = 0.0093
  aqr_context_same_role_support_overlap_max = 0.2120
  aqr_downstream_same_role_support_overlap_max = 0.2521
  raw aqr_same_role_support_overlap_max = 0.9951
  loss_object_explanation_point = 4.4468
  grad_clip_applied = false

step 200:
  aqr_active_same_role_support_overlap_max = 0.0390
  aqr_context_same_role_support_overlap_max = 0.3786
  aqr_downstream_same_role_support_overlap_max = 0.4614
  raw aqr_same_role_support_overlap_max = 0.9995
  loss_object_explanation_point = 4.2610
  grad_clip_applied = false

step 250:
  aqr_active_same_role_support_overlap_max = 0.0923
  aqr_context_same_role_support_overlap_max = 0.4327
  aqr_downstream_same_role_support_overlap_max = 0.6175
  raw aqr_same_role_support_overlap_max = 0.9999
  loss_object_explanation_point = 5.4936
  grad_clip_applied = true

step 300:
  aqr_active_same_role_support_overlap_max = 0.0490
  aqr_context_same_role_support_overlap_max = 0.5121
  aqr_downstream_same_role_support_overlap_max = 0.6277
  raw aqr_same_role_support_overlap_max = 0.9999
  loss_object_explanation_point = 6.0854
  grad_clip_applied = true
```

Interpretation:

```text
The old failure point did not recur through step 300.  Raw reserve-pool overlap
can still approach 1.0, but the action/control-visible context and downstream
rows remain below the previous 0.83-0.84 failure band.

This supports the intended mathematical boundary:
  keep overcomplete AQR capacity and dense typed memories;
  suppress only duplicate low-priority context rows before downstream control.

This does not prove full anchor binding closure:
  late point/object-pull losses still rise;
  grad clipping appears after step 250;
  overlays show that active graph anchors can be near but not consistently
  centered on the sidecar mask, and posterior active files can remain farther
  away from the task object.
```

Visual overlay check:

```text
step 150 open_the_drawer:
  active object anchor is on the sidecar/contact mask region.
  gray context rows remain visible only as low-weight scene context.

step 200 slide_the_door_to_the_left:
  one active object anchor is on the sidecar/mask region.
  some reserve/context graph rows remain around the nearby gripper/scene, but
  downstream exposure is reduced to three graph rows and does not recreate the
  old many-row duplicate field.

step 300 push_the_switch_downwards:
  sidecar/mask is on the switch region.
  active graph anchor is in the neighborhood but not cleanly centered on the
  mask; several low-weight downstream context rows remain visible around the
  same task area.
  posterior active file is still farther from the sidecar owner than desired.
```

Remaining watch item:

```text
The slot-quality gate can make raw-active and downstream weight differ.
This is acceptable only if active/downstream overlap and overlays remain
healthy.  If future runs show active rows selected but suppressed before
control, the next fix should calibrate active-quality gating rather than add a
new loss.
```

## Closure Verdict For This Repair

```text
Solved:
  previous context/downstream full-support duplicate rebound at step 150.

Not solved:
  complete active object localization and posterior owner convergence.

Next mathematically targeted issue:
  sidecar/contact owner evidence is strong enough to select a task neighborhood,
  but not yet strong/reliable enough to make the active graph row and posterior
  file consistently coincide with the sidecar mask center.  The next repair
  should audit owner transport and active-quality calibration, not add more
  generic diversity/context losses.
```

## External References

```text
MetaSlot:
  https://arxiv.org/abs/2505.20772

QASA:
  https://arxiv.org/abs/2601.12936

Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?:
  https://arxiv.org/abs/2510.24709

SlotVLA:
  https://arxiv.org/abs/2511.06754

STORM:
  https://arxiv.org/abs/2601.20381
```

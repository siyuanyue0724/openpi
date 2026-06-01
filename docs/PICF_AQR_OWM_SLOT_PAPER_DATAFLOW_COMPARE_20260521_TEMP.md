# PICF-AQR-OWM Slot Paper Dataflow Compare

Date: 2026-05-21

Purpose: map the current PICF object-file/belief-router implementation against
recent object-centric slot work, then define the next 300-step validation gate.
This document is intentionally about mathematical dataflow, not cosmetic
module inventory.

## External Code / Paper Snapshot

Checked locally under `/tmp/picf_slot_paper_code`:

```text
vit-object-binding:
  repo: https://github.com/liyihao0302/vit-object-binding
  relevant file:
    /tmp/picf_slot_paper_code/vit-object-binding/src/utils/models.py

MetaSlot:
  repo: https://github.com/lhj-lhj/MetaSlot
  relevant file:
    /tmp/picf_slot_paper_code/MetaSlot/object_centric_bench/model/metaslot.py

naive-e-slotssm:
  repo advertised for Embodied-SlotSSM / LIBERO-Mem
  current clone state:
    empty repository; no deployable code to reuse.

QASA:
  paper: https://arxiv.org/abs/2601.12936
  code snapshot:
    no official deployable repository found in the 2026-05-21 web/GitHub
    search pass; use paper-level mechanism only.
```

Papers used as design constraints:

```text
Object Binding in ViTs, NeurIPS 2025:
  https://arxiv.org/abs/2510.24709
  Pairwise IsSameObject is not raw attention/cosine; quadratic or low-rank
  bilinear probes expose a binding subspace.

MetaSlot, NeurIPS 2025:
  https://arxiv.org/abs/2505.20772
  Fixed slot count causes duplicate/background slots; useful mechanisms are
  prototype-guided slot identity, duplicate masking, and progressively weaker
  slot-aggregation noise.

STORM, 2026:
  https://arxiv.org/abs/2601.20381
  Robotic manipulation benefits from semantic-aware object slots stabilized
  before policy co-training; frozen foundation backbones plus lightweight slot
  adapters are sufficient.

QASA, 2026:
  https://arxiv.org/abs/2601.12936
  Quality-guided adaptive slot count should separate object, no-object, and
  duplicate slots before downstream losses interpret slot geometry.

SlotVLA, 2025/2026:
  https://arxiv.org/abs/2511.06754
  Robotic manipulation should preserve temporally consistent object/relation
  representations rather than flattening everything into one dense feature.

When Slots Compete, 2026:
  https://arxiv.org/abs/2603.11246
  Overlapping slots should be handled as duplicate/no-object competition,
  including merge/suppression-style policies.  This supports judging active
  object owners separately from raw fixed-capacity reserve rows.
```

## PICF Object-File Dataflow

PICF does not implement plain Slot Attention over RGB tokens.  Its unit is a
posterior object file in a belief filter:

```text
dense evidence:
  V-JEPA/static+wrist tokens
  PaliGemma support tokens
  point tokens
  tactile tokens
  sidecar proposal/mask/tracklet tokens

measurement graph:
  graph rows receive support distributions over each evidence type
  sidecar candidates are weak measurement priors, not truth labels
  object_candidate_owner_* transports accepted point/mask evidence to role-1
  object rows

posterior:
  persistent file state receives graph measurements through binding and
  correction
  duplicate owner competition suppresses multiple active files explaining the
  same object

action path:
  PI0.5 action head reads posterior/action prefix plus retained context
```

Mathematically, the object-file contract is:

```text
support row:
  s_j^m = soft assignment of slot/file j to modality m tokens

object candidate prior:
  c_j(i) = weak sidecar/contact/motion support over point token i

owner target center:
  x*_j = sum_i normalize(c_j(i)) x_i

graph pull:
  L_graph_j = SmoothL1((x_graph_j - stopgrad(x*_j)) / sigma)

posterior pull:
  L_post_j = SmoothL1((x_post_j - stopgrad(x*_j)) / sigma)

projective point/visual consistency:
  v_hat_j = normalize(p_j C)
  p_hat_j = normalize(v_j C^T)
  L_pv_j = 0.5 JS(v_j, v_hat_j) + 0.5 JS(p_j, p_hat_j)

object explanation:
  quality_j = confidence_j * exp(-feature_var_j / 2) * exp(-point_var_j / 2)
```

This intentionally differs from image-reconstruction OCL:

```text
not used as production loss:
  full RGB reconstruction decoder
  hard segmentation labels
  sidecar masks as posterior truth

reason:
  PICF is a control belief router.  Full pixel reconstruction would push the
  router toward explaining every background pixel equally, while the policy
  needs stable task-object belief plus dense context retention.
```

## Paper Mechanism Mapping

### Object Binding in ViTs

Code evidence:

```text
QuadraticProbe:
  IsSameObject(x,y) = x^T W y + b

QuadraticFixedRankProbe:
  IsSameObject(x,y) = x^T W_lowrank y + b
```

PICF mapping:

```text
implemented:
  binding_signature_proj
  support-weighted binding_signature
  bind_quadratic_signature_weight
  bind_low_rank_signature_weight
  double-center z-score calibration
  posterior binding-signature memory

not copied literally:
  supervised ADE20K IsSameObject probe training

reason:
  current CALVIN data lacks dense object masks.  We use the same mathematical
  class of pairwise/projected subspace but train/evaluate it through weak
  tracklet/contact/posterior evidence.
```

### MetaSlot

Code evidence:

```text
MetaSlot.forward:
  1. run initial slot attention
  2. quantize intermediate slots with VQ codebook
  3. mask duplicate prototype indices
  4. run masked slot attention with attenuated noise
```

PICF mapping:

```text
implemented equivalent:
  posterior file competition removes duplicate active object owners
  slot_quality/no-object/duplicate targets model variable effective count
  context/reserve rows are logged separately from active object owners
  active duplicate owner overlap is rejected by metric gates

not copied literally:
  visual prototype VQ codebook as posterior truth

reason:
  PICF must support missing modalities and non-visual object identity.  A visual
  VQ prototype can over-lock identity to image appearance and weaken tactile,
  point, and language-driven ownership.  The correct PICF analogue is posterior
  file competition plus weak prototype-like binding signatures, not a hard VQ
  visual codebook.
```

### STORM / SlotVLA

PICF mapping:

```text
implemented equivalent:
  staged/frozen-backbone validation
  PaliGemma/semantic support as task object evidence when enabled
  persistent posterior files and temporal tracklet evidence

current 300-step test contract:
  freeze PaliGemma and action head;
  freeze Sonata/V-JEPA/AnyTouch pretrained modules;
  train the object-file, routing, sidecar, binding, posterior, and connector
  components that determine whether slots form correctly.
```

### QASA

Paper-level evidence:

```text
quality-guided K-adaptive slot attention:
  estimate whether a slot explains an object
  suppress duplicate/no-object slots
  avoid forcing a fixed number of slots to explain every scene
```

PICF mapping:

```text
implemented equivalent:
  slot_quality graph state
  no-object / duplicate weak targets
  active-slot filter and context-slot separation
  posterior file competition for same-owner duplicates
  context/reserve visibility without treating reserve rows as object owners

not copied literally:
  a standalone QASA slot-attention block replacing the belief filter

reason:
  PICF's state variable is a persistent multimodal posterior file, not a
  single-frame image slot set.  Replacing posterior correction with a generic
  QASA block would drop temporal identity, tactile/point missing-modality
  handling, and PI0.5 action-prefix contracts.  The mathematically relevant
  mechanism is adaptive object/no-object/duplicate quality, already expressed
  in PICF-native state.
```

## Current Gap Found In 2026-05-21 30K Window

The 50-500 step interrupted long-run showed:

```text
fixed:
  active owner duplicate collapse did not occur.
  posterior active duplicate overlap stayed 0.
  action default-equivalent loss was healthy.

not fixed:
  raw reserve/context overlap saturated.
  context count collapsed.
  anchor_object_pull and object_explanation_point rose.
```

The old scalar `loss_anchor_object_pull` was underidentified:

```text
old:
  L_pull = weighted average of graph-anchor pull and posterior-file pull

problem:
  if L_pull rises, we cannot tell whether graph assignment, posterior
  write-through, or sidecar target quality is responsible.

deployed fix:
  log these separately:
    loss_anchor_object_pull_graph
    loss_anchor_object_pull_posterior
    loss_anchor_object_pull_graph_weight_sum
    loss_anchor_object_pull_posterior_weight_sum
    loss_anchor_object_pull_target_mass_mean
```

This is not a new auxiliary objective.  It is a required observability upgrade
before another 300/30k run can be interpreted.

## 300-Step Gate

Run contract:

```text
launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_slot_splitpull_frozen_policy_300_20260521.sh

steps:
  300

freeze:
  PaliGemma
  action head
  Sonata pretrained module
  V-JEPA pretrained module
  AnyTouch pretrained module

train:
  PICF/AQR/posterior object-file routing
  sidecar/contact-motion object assignment
  binding signatures
  adapter/connector parts not in the frozen foundation modules

checkpoints:
  not required for 300-step diagnosis

diagnostics:
  print detailed losses every 50 steps
  save anchor overlay every 100 steps
```

Kill/continue rules:

```text
at step 100:
  stop if active overlap is already in the old collapse band or graph/posterior
  pull split proves the active object cannot receive sidecar supervision.

at step 200:
  continue only if graph/posterior pull split identifies a stable or improving
  object-file path and context count is not rapidly collapsing.

at step 300:
  accept for next 30k only if:
    active_same_role_support_overlap_max remains low;
    posterior duplicate overlap remains zero;
    graph_pull and posterior_pull are both interpretable;
    object_explanation_point is bounded or explained by target-mass quality;
    overlays show active object rows on the sidecar/contact object core.
```

## Current Design Position

The deployment is not a minimal patch.  It keeps the mathematically necessary
mechanisms from the latest slot literature while rejecting mechanisms that
break the PICF control-router contract:

```text
keep:
  pairwise binding subspace
  variable effective object count
  duplicate active-owner suppression
  weak sidecar/contact/tracklet measurement evidence
  background/dense context retention
  staged frozen-backbone slot validation

reject for now:
  blind SAM
  hard visual VQ prototype truth
  full RGB reconstruction decoder in the action path
  hard sidecar mask labels
```

## Complete-Deployment Rule

This project should not add isolated losses just because one scalar looks bad.
Every imported slot mechanism must pass all of these tests:

```text
1. object-binding meaning:
   It must improve object ownership, duplicate/no-object handling, temporal
   identity, or missing-modality fusion.

2. belief-filter compatibility:
   It must enter as evidence, gating, diagnostics, or training-only weak
   auxiliary.  It must not overwrite posterior truth outside the precision
   posterior update and file-competition contract.

3. missing-modality scaling:
   It must still work when RGB, point cloud, tactile, tracklet, or sidecar
   evidence is absent.  A mechanism that requires dense masks or fixed visual
   prototypes for every dataset is not a production default.

4. action-path safety:
   Dense V-JEPA / point / tactile context must not be destructively pruned.
   Slots organize task-relevant object evidence; dense context remains available
   as peripheral scene evidence.

5. measurable acceptance:
   It must expose metrics that distinguish active object failure from
   reserve/no-object capacity.  A raw all-row scalar is not enough.
```

Consequences:

```text
MetaSlot-style duplicate suppression:
  accepted as active/context/reserve demotion and posterior file competition.
  rejected as hard visual VQ posterior truth because PICF must support missing
  visual masks and multimodal object files.

QASA-style slot quality:
  accepted as weak quality/no-object/duplicate gating.
  rejected as a standalone decoder replacement because the action belief
  router already has posterior authority and dense context must survive.

When-Slots-Compete-style overlap handling:
  accepted as active duplicate suppression / reserve classification.
  rejected as a requirement that all fixed-capacity rows become distinct
  objects.

SlotVLA-style object/relation control:
  accepted as the high-level target: object files plus relation/task context
  should condition action.
  not copied literally because SlotVLA assumes object-centric annotations and
  instance tracking that CALVIN does not provide natively.
```

## Raw Overlap Root-Cause Contract

The latest 300-step gate makes raw same-role overlap the only ambiguous scalar:

```text
raw same_role_support_overlap:
  all fixed-capacity rows, including active object rows, context rows, and
  reserve/no-object rows.

active same_role_support_overlap:
  confirmed action-visible object owners only.

context same_role_support_overlap:
  low-weight rows retained as peripheral/dense scene context.

reserve same_role_support_overlap:
  no-object / duplicate capacity rows with downstream weight zero.
```

Recent slot papers do not require every unused slot in a fixed-capacity bank to
become a distinct object.  MetaSlot masks duplicate prototype slots before the
second refinement pass; QASA separates object, no-object, and duplicate slots;
Slot-merging work treats overlapping maps as duplicate competition rather than
independent object truth; SlotVLA uses object/relation slots for manipulation
rather than forcing every token into an action-visible object.  Therefore
PICF's mathematically correct acceptance rule is:

```text
must be low:
  active overlap
  posterior active duplicate overlap
  downstream overlap if it becomes large enough to affect the action path

may be high if explained:
  reserve overlap, because reserve rows are explicitly no-object capacity
  and have downstream_weight = 0

must be reported separately:
  raw overlap = active/context/reserve mixture
```

The code now logs reserve-scope overlap explicitly:

```text
aqr_reserve_same_role_support_overlap_max
aqr_reserve_same_role_object_core_overlap_max
```

The scopes are defined as mutually exclusive diagnostics:

```text
active_i  = graph.anchor_active_i > 0.5
context_i = downstream_weight_i > eps AND NOT active_i
reserve_i = downstream_weight_i <= eps AND NOT active_i

raw_overlap      = max_{i<j, role_i=role_j} cos_support(s_i, s_j)
active_overlap   = max over active_i AND active_j
context_overlap  = max over context_i AND context_j
reserve_overlap  = max over reserve_i AND reserve_j
downstream_overlap = max over downstream_i>eps AND downstream_j>eps
```

This matters because a single scalar over all fixed rows cannot distinguish
"two active files bound to the same object" from "two no-object reserve files
holding unused capacity."  The first is a behavioral failure; the second is
expected in K-fixed or K-capped object-centric systems unless a full adaptive
slot-count replacement is used.

This is not a new objective.  It is the missing follow-through needed to decide
whether high raw overlap is a harmless reserve/no-object artifact or an actual
action-visible object-binding failure.

Next code-level requirement after this doc:

```text
run local tests and audits;
sync to A7;
launch the strict-scope 300-step frozen Paligemma/action-head test;
inspect 100/200/300 with active/context/reserve overlap and split pull metrics.
```

## 2026-05-21 Repair: Quality-Gated Weak Object Targets

The strict-scope gate after the mutually exclusive active/context/reserve audit
did not reproduce the old active-owner collapse.  The actual late failure was:

```text
step 50 -> 100:
  object_pull improved.

step 200 -> 300:
  object_pull rose in both graph and posterior branches;
  object_explanation_point rose;
  downstream/context overlap rose;
  posterior active duplicate overlap stayed exactly zero.
```

That points to weak-target drift, not to a need for another global all-row
overlap penalty.

### Old Target

The previous object-pull target trusted the maximum over all weak source priors:

```math
T_j(i)
=
\max_s P^{(s)}_j(i)
```

where `s` covers owner, proposal, point, and task-owner priors.  This preserved
all evidence, but it also let contact-motion tails or broad proposal fields
define the same geometric center as compact object cores.

### Repaired Target

Dense typed memory remains untouched.  Only the training-time pull target uses
a robust high-confidence core:

```math
C_j
=
\operatorname{TopCore}(T_j;\rho,k),
\qquad
\rho=0.90,\ k=128.
```

The detached center is:

```math
\mu_j
=
\frac{\sum_i C_j(i)x_i}{\sum_i C_j(i)+\epsilon}.
```

The weak target receives a compactness quality:

```math
q_j
=
\exp\left(
-\frac{1}{2\sigma_q^2}
\frac{\sum_i C_j(i)\lVert x_i-\mu_j\rVert^2}
{\sum_i C_j(i)+\epsilon}
\right),
\qquad
\sigma_q=0.08m.
```

Rows with `q_j < 0.05` are not allowed to behave as hard pull teachers.  Valid
rows are weighted by `q_j`:

```math
L^{pull}_j
=
q_j\,
\operatorname{SmoothL1}
\left(
\frac{\hat x_j-\mu_j}{\sigma_{pull}}
\right).
```

The downstream context path now uses the slot-quality context gate by default:

```text
active:
  selected by active-slot/posterior competition.

context:
  must pass confidence/support/duplicate checks and quality-guided context
  visibility.

reserve:
  keeps fixed capacity for future birth/no-object state and has downstream
  weight zero.
```

This is the PICF-compatible import of the recent slot literature:

```text
Slot Attention / SAVi:
  object tokens compete across slots, but masked/empty slots are not downstream
  object truth.

MetaSlot:
  duplicate fixed-K slots are demoted/masked.

QASA:
  quality-guided slot selection is decoupled from the main objective so active
  count, duplicate/no-object handling, and fidelity do not fight each other.
```

New diagnostic:

```text
loss_anchor_object_pull_target_quality_mean
```

Local verification before A7 rerun:

```text
py_compile:
  pipeline.py
  training.py
  config.py
  picf_core_train.py
  picf_core_train_test.py
  training_test.py

strict scripts:
  verify_picf_owm_contract.py
  picf_owm_strict_diagnose.py --fail-on-fail
  picf_owm_dataflow_trace.py --fail-on-fail
  picf_owm_mvtrack_deep_audit.py --fail-on-fail

pytest:
  scripts/picf_anchor_run_diagnostic_report_test.py
  scripts/picf_core_train_test.py
  scripts/picf_loss_audit_test.py
  pipeline_test.py
  training_test.py
```

Next 300-step acceptance:

```text
must stay good:
  posterior_active_duplicate_overlap_max == 0
  active_same_role_object_core_overlap_max remains near zero
  active_same_role_support_overlap_max remains far below the old collapse band

must improve versus strict-scope:
  loss_anchor_object_pull should not show the 0.235 -> 1.048 late drift
  loss_object_explanation_point should remain bounded
  downstream_same_role_support_overlap_max should not continue rising past 0.5
  target_quality_mean should expose whether object-pull targets are compact

still not a blocker alone:
  raw/reserve overlap when reserve downstream weight is zero
```

## OEML Point Compactness Follow-Through

The 300-step quality-target gate fixed active-owner duplicate collapse and the
late object-pull explosion, but left one real issue:

```text
loss_object_explanation_point remains high/noisy even when overlays show active
anchors on the sidecar/contact object masks.
```

The code-level cause is not insufficient overlap suppression.  It is a
normalization mismatch:

```math
q_j = q^{base}_j \sqrt{q^{feat}_j q^{point}_j},
\qquad
q^{point}_j = \exp(-p_j/2)
```

Old loss:

```math
L_{point}^{old}
=
\frac{\sum_j q_j p_j}{\sum_j q_j+\epsilon}.
```

When every active weak target in a batch is noisy, `q_j` cancels in the
normalization and the loss still acts like a hard compactness label.  This is
inconsistent with the imported slot principles:

```text
QASA:
  slot quality should decide whether a row is an object-quality row before
  downstream losses interpret it.

MetaSlot / fixed-K OCL:
  duplicate/no-object/noisy slots must be demoted or masked rather than forced
  to explain object truth.

PICF:
  sidecar/contact/motion masks are weak evidence, not dense mask labels.
```

Repaired weak likelihood:

```math
\rho(p_j)
=
-2\log\left((1-\eta)\exp(-p_j/2)+\eta\right),
\qquad \eta=0.05
```

```math
L_{point}^{new}
=
\frac{
\sum_j q^{base}_j\,
\operatorname{stopgrad}(q^{point}_j)^\alpha\,
\rho(p_j)
}{
\sum_j q^{base}_j+\epsilon
}.
```

Properties:

```text
compact high-quality rows:
  p_j small -> rho(p_j) approximately p_j, so the loss still pulls toward
  compact object explanations.

noisy/broad weak targets:
  q_point is detached and downweights the row, so the model cannot reduce the
  objective by deliberately making point variance worse.

diagnostics:
  raw oeml_point_spatial_variance_mean remains logged; the loss is robust, not
  blind.
```

This is the correct PICF-native form of quality-guided slot supervision.  It
does not import a full image reconstruction decoder, a visual VQ truth codebook,
or hard sidecar labels, because those would conflict with missing-modality
scaling and the PI0.5 action-router contract.

300-step result:

```text
point-quality/outlier robust loss is mechanically valid but not production
positive in this configuration.

It reduces:
  loss_object_explanation_point

but worsens:
  object_pull
  downstream overlap
  active support overlap

Therefore the production default keeps OEML point compactness as the original
bounded raw weak regularizer.  The robust formula remains an explicit ablation
knob only.  This is not a retreat to an incomplete model; it is the result of
the slot-literature contract: quality gates should demote weak/no-object slots,
but they must not remove useful compactness pressure from the only geometry
teacher available in CALVIN sidecars.
```

# PICF-AQR-OWM Binding Follow-Through Audit

Date: 2026-05-16

Status: active follow-through document. This file is intentionally more explicit
than the deployment README: it records the current math, the dataflow path, the
paper-code comparison, and what the current A7 run must prove or falsify.

## 1. Non-Negotiable Invariants

The current PICF-AQR-OWM line is only coherent if these invariants hold:

1. Posterior files are the authoritative current belief.
2. AQR routes typed evidence; it does not create evidence not present in the
   typed memory.
3. Reserve/no-object posterior files may exist as fixed capacity, but inactive
   files must not enter action, predictive, cache, or posterior reread paths.
4. Cache is historical evidence, residual-gated, and never current truth.
5. Future latent/posterior targets are detached and never fed into current
   action inputs.
6. Object-binding paper code is used as equation/protocol evidence only. It is
   not copied into PICF, and its BCE IsSameObject training is not used online
   without labels.
7. Overlay diagnostics must separate active files from inactive capacity.
   `with_gray` is a capacity/debug view; `active_only` is the binding-quality
   view.

## 2. Runtime Dataflow

The current forward path is:

```text
PicfObservation
  rgb_static / rgb_gripper / depth / point / tactile / proprio / prompt / action
  optional tracklet_* / proposal_*

-> _visual_maps
  static + optional gripper V-JEPA temporal maps

-> _build_token_field
  point_tokens
  visual_tokens
  temporal_visual tokens with time/view ids
  tactile tokens
  optional tracklet/proposal states

-> _build_aqr_anchor_graph
  typed readers:
    PG image, visual, temporal visual, point, tactile, posterior, cache,
    optional tracklet/proposal
  output:
    graph priors, anchor roles, support uncertainty, owner active state

-> _build_observation_anchors
  seed observation anchors from point coverage plus graph support
  compute support-weighted signatures:
    point / visual / temporal / PG / tactile / tracklet / proposal
  compute binding_signature

-> _posterior_update
  prior prediction
  binding logits:
    hidden + geometry + support overlap + projected pairwise binding subspace
    + gated address + role/owner/occupancy biases
  Sinkhorn dustbin
  posterior_file_competition
  posterior file active gate
  posterior self/update

-> downstream views
  action PI prefix
  predictive basis
  cache write
  overlay diagnostic
```

File map:

```text
src/openpi/picf/core/config.py
  binding, file-competition, lifecycle, cache, multiview, and loss knobs

src/openpi/picf/core/contracts.py
  posterior fields for active file state, signatures, and diagnostics

src/openpi/picf/core/pipeline.py
  all runtime dataflow and posterior math

src/openpi/picf/core/training.py
  losses and guarded auxiliary hooks

scripts/picf_core_train.py
  CALVIN frame loading, metrics, dual overlay dump, train launch

scripts/picf_owm_same_object_probe.py
  offline IsSameObject probe; not an online training loss
```

## 3. Mathematical Model

PICF is a belief filter:

```math
b_t(s) \propto p(o_t | s_t) \int p(s_t | s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds_{t-1}.
```

For slot/file `j`, typed evidence `i`, and modality `m`, AQR computes:

```math
p^{(m)}_{j,i}
=
\operatorname{softmax}_i
\left(
q_j^\top k^{(m)}_i / \sqrt d
+ b^{(m)}_{j,i}
\right).
```

Posterior binding has the form:

```math
\ell_{j,i}
=
\lambda_h \cos(h^-_j, o_i)
- \lambda_g (x_i-\mu^-_j)^T(\Sigma^-_j+\sigma^2I)^{-1}(x_i-\mu^-_j)
+ \lambda_s g_j O_{j,i}^{support}
+ g_j S_{j,i}^{bind}
+ \lambda_a g_j \cos(a^-_j,\tilde a_i)
+ b_{role}
+ b_{owner}
+ b_{occupancy}.
```

The support term is a mean over available typed signatures:

```math
O_{j,i}^{support}
=
\operatorname{mean}_m
\langle \sigma_{j}^{m,-}, \sigma_i^m \rangle.
```

The binding subspace term follows the object-binding paper's pairwise family:

```math
S_{j,i}^{bind}
=
\lambda_l \cos(P c^-_j, P o_i)
+ \lambda_d (P c^-_j \odot P o_i)^T w
+ \lambda_r (L P c^-_j)^T (R P o_i).
```

However, the paper's raw probe logits are trained with instance-mask labels.
PICF has no such labels in CALVIN, so raw pairwise scores are converted into
relative assignment evidence:

```math
\tilde S
=
zscore(
S - rowmean(S) - colmean(S) + mean(S)
).
```

If dispersion is below threshold, PICF emits zero binding evidence rather than
amplifying common-mode noise. This is why the current implementation uses
`double_center_zscore` calibration.

Posterior file competition is the no-object step missing from fixed-capacity
slots:

```math
a_j \in \{0,1\},\quad
\hat M_{j,i} = a_j M_{j,i},\quad
D'_i = D_i + \sum_j (1-a_j)M_{j,i}.
```

This is mass conserving: duplicate files are demoted to dustbin/no-object, not
deleted.

Downstream exposure must respect the same active decision:

```math
z^{down}_j = a_j z_j.
```

This same gate is used for posterior reread, posterior self attention, control
prefix, predictive basis, and evidence-cache validity. Without this gate, gray
reserve files can still leak into action or cache even if the overlay marks them
inactive.

The remaining information bound is unchanged:

```math
I(Y; A_t \mid \ell)
\le
I(Y; Z_{\le t} \mid \ell).
```

File competition and pairwise binding reduce the error in the map
`Z_{\le t} -> A_t`; they cannot create target identity if static/wrist/point/
tactile/PG evidence does not contain it.

## 4. Paper-Code Comparison

Local reference snapshots:

```text
/tmp/vit-object-binding
/tmp/picf_paper_code_20260515/vit-object-binding
/tmp/picf_paper_code_20260515/slotcontrast
```

Object-binding paper code:

```text
/tmp/vit-object-binding/src/utils/models.py
  DiagonalQuadraticProbe:        x * y -> linear logit
  QuadraticProbe:                x^T W_sym y + bias
  QuadraticFixedRankProbe:       x^T (W1^T W2 + W2^T W1) y + bias
  SelfAttentionProbe:            q(x)^T k(y) / sqrt(d) + bias

/tmp/vit-object-binding/src/trainer.py
  labels_pairwise = instance_masks[:, :, None] == instance_masks[:, None, :]
  BCEWithLogitsLoss(pairwise_similarity, labels_pairwise)

/tmp/vit-object-binding/src/utils/score.py
  compute_issameobject_all(probe, act)
  compute_batch_pairwise_similarity(probe, x, y)
```

Implication:

```text
Allowed:
  use quadratic/low-rank pairwise score family as structural binding evidence;
  use offline probes on exported overlay/signature data;
  use same-object signal as a diagnostic.

Not allowed:
  add online BCE IsSameObject training in CALVIN without masks, tracklets, or
  equivalent reliable labels;
  treat raw cosine/common-mode high similarity as object identity;
  copy unlicensed code into PICF.
```

Slot-related 2025 papers support duplicate handling and reserve-file semantics:

```text
DIAS:
  redundant slots can compete with informative slots; re-initialization and
  self-distillation reduce redundant aggregation.

MetaSlot:
  fixed slot count can split one object into multiple slots; duplicate slots
  should be removed/merged through a variable-count/prototype mechanism.
```

PICF mapping:

```text
posterior_file_competition:
  mass-conserving duplicate demotion to dustbin/no-object.

downstream active gate:
  inactive reserve files are not allowed to influence action/cache/prediction.

not yet imported:
  full prototype-codebook MetaSlot. That would be a larger architecture change
  and is not justified until active-only overlays show persistent duplicate
  active files after current fixes.
```

Reference links used for theory audit:

```text
Object Binding in ViTs:
  https://arxiv.org/abs/2510.24709

DIAS redundant-slot/re-initialization line:
  https://arxiv.org/abs/2507.23755

MetaSlot variable/dynamic slot line:
  https://arxiv.org/abs/2505.20772
```

## 5. Loss Interaction Audit

Current production/diagnostic stance:

```text
Action loss:
  must train the policy, but can collapse binding if inactive reserve files
  leak into the action prefix. This is why downstream active gating is structural.

PV / weak alignment / denoising:
  useful only if they align observation anchors to real typed evidence. They
  cannot make two identical support rows represent different objects.

Support diversity:
  helps only when there is weak per-anchor evidence. It is not sufficient as a
  hard fix for duplicate posterior files.

slot_jepa / support_pred / binding_consistency:
  remain guarded unless matched targets and identity stability are proven.
  They are not the current cause of gray reserve-file leakage.

same-object probe:
  offline diagnostic only unless reliable labels are available.
```

Main failure modes and current status:

```text
1. Active AQR support collapse:
   mostly falsified by prior overlayfix runs; same-role overlap can be high in
   raw capacity but active files may be distinct.

2. Gray reserve files visually interpreted as active files:
   fixed by dual overlay:
     with_gray     = full capacity view
     active_only   = active posterior binding view

3. Gray reserve files stealing downstream tokens:
   fixed by _posterior_file_active_gate in posterior reread, posterior self,
   control prefix, predictive basis, and cache validity.

4. Active files off target:
   not yet proven fixed. Requires active_only overlay inspection at step50+.
   If present, inspect owner/contact/point geometry rather than adding new
   same-object losses.

5. Wrist binding confusion:
   wrist is typed temporal/PG evidence; it is not static world geometry truth.
   If active files bind to wrist-like locations, the likely issue is owner/contact
   evidence or projection assumptions, not gray reserve capacity.
```

## 6. Current A7 Evidence Gate

Run:

```text
tmux: picf_a7_downstream_gate_diag300
run:  picf_a7_downstream_gate_diag300_20260516
log:  /mnt/picf_run_logs/picf_a7_downstream_gate_diag300_20260516.log
dir:  /mnt/checkpoints/picf_core/picf_core/picf_a7_downstream_gate_diag300_20260516
```

Startup verified:

```text
unroll_steps=2
burnin_steps=1
anchor_overlay_interval=50
visual=encoder frozen
sonata frozen
anytouch frozen
paligemma trainable
losses(slot_jepa=0, support_pred=0, bind=0, denoise=0)
local_refinement_enabled=False
posterior_file_competition_enabled=True
posterior active downstream gate patch deployed
dual overlay patch deployed
```

Acceptance logic:

```text
At step50:
  read metrics.jsonl
  inspect step_000050__task__with_gray.png
  inspect step_000050__task__active_only.png
  parse step_000050__task.json
  run scripts/picf_anchor_overlay_summary.py on the JSON sidecar

If with_gray is crowded but active_only is clean:
  reserve capacity is no longer the issue; continue.

If active_only still has multiple active same-role files on one object:
  posterior file competition thresholds or lifecycle need revision.

If active_only active files are off target:
  inspect observation-anchor owner evidence, contact/point centers, and prompt
  task grounding. Do not blame gray reserve files.

If metrics show active file recycle high but inactive recycle high only:
  active objects are stable enough; reserve churn is acceptable.

If both active and inactive recycle high:
  lifecycle survival/reset model still unstable.
```

Observed step50:

```text
prompt = grasp the red block and turn it left

aqr_active_same_role_support_overlap_max = 0.0182
aqr_same_role_support_overlap_max        = 0.1424
aqr_active_same_role_object_core_max     = 0.0266
posterior_active_file_recycle_rate       = 0.0621
posterior_inactive_file_recycle_rate     = 0.2911
posterior_identity_switch_rate           = 0.7372
posterior_active_file_potential_swap     = 0.3720

overlay summary:
  posterior_visible          = 8
  posterior_active_visible   = 5
  posterior_inactive_visible = 3
  posterior_demoted_visible  = 3
  role-1 active visible      = 4
  role-1 inactive visible    = 3
  role-1 min active distance = 10.38 px
```

Interpretation:

```text
The old failure "gray reserve capacity is being read as active object state" is
not supported by step50: active-only removes the gray files and active overlap
is low. The remaining visible issue is that task/object allocation may still be
too broad for a red-block command: active role-1 files cover a small cluster
around the blocks/gripper instead of a single task target.

This does not justify adding an online IsSameObject loss. It points to
task-conditioned target selection / contact-owner evidence if the pattern
persists at step100/150.
```

## 7. Scripts Already Run

Local:

```text
python -m py_compile scripts/picf_core_train.py scripts/verify_picf_owm_contract.py ...
python scripts/verify_picf_owm_contract.py
python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
python scripts/picf_binding_logit_calibration_audit.py --fail-on-fail
python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
uv run python scripts/picf_posterior_file_competition_audit.py --fail-on-fail
uv run pytest -q src/openpi/picf/core/pipeline_test.py -k "posterior_inactive_files or posterior_file_competition or posterior_lifecycle or binding_signature"
```

Remote A7:

```text
/root/openpi/.venv/bin/python -m py_compile scripts/picf_core_train.py scripts/verify_picf_owm_contract.py
/root/openpi/.venv/bin/python scripts/verify_picf_owm_contract.py
```

## 8. Current Conclusion

The current repair is not a patchwork loss change. It is a correction to the
belief-state semantics:

```text
duplicate fixed-capacity files -> no-object decision -> inactive reserve state
-> no downstream evidence/action/cache exposure.
```

This is mathematically required by the fixed-capacity posterior-file design and
is directly aligned with recent slot work on redundant-slot handling.

What is not yet justified:

```text
claiming all active binding is correct before active_only overlays are inspected;
adding online IsSameObject pseudo-loss without reliable labels;
adding full MetaSlot-style prototype codebooks before proving active duplicate
files remain after file competition and downstream gating.
```

## 9. Gray Reserve Versus Active Object Binding

This section addresses the concrete overlay concern from the A7 diagnostic:
many gray posterior files may appear near the task object, while the user needs
to know whether those files still steal evidence or action capacity.

The two claims are different:

```text
Claim A:
  gray reserve files are visible in the overlay.

Claim B:
  gray reserve files still participate in action, predictive state, cache write,
  or posterior reread.
```

Claim A is expected with a fixed-capacity posterior. A fixed-size posterior file
bank must keep reserve rows so that new objects can be born later. The correct
no-object operation is not deleting those rows; it is demoting their measurement
mass into the dustbin and masking them from downstream reads.

Claim B is the real bug to rule out. The current code path uses
`posterior.file_competition_active` as the active-file gate for downstream
consumers:

```text
posterior update:
  support_raw, dustbin_raw
  -> posterior_file_competition
  -> support_active, dustbin_active, file_competition_active

downstream:
  AQR posterior reread     uses active gate
  posterior self update    uses active gate
  global posterior pooling uses active gate
  PI prefix/action state   uses active gate
  predictive basis         uses active gate
  evidence cache write     uses active gate
```

Mathematically:

```math
g_j \in \{0,1\}
```

is the posterior no-object decision for file `j`. The action-visible posterior
summary is:

```math
\bar h_t
=
\frac{\sum_j g_j h_{t,j}}{\epsilon + \sum_j g_j}.
```

The cache write validity is:

```math
valid_{cache,j}=g_j.
```

Therefore gray files can remain visible in `with_gray` while being removed from
action/predictive/cache evidence. This is not a cosmetic distinction; it is the
belief-state semantics of a fixed-capacity object-file model.

### Step50 A7 Evidence

At step50 of `picf_a7_downstream_gate_diag300_20260516`:

```text
prompt:
  grasp the red block and turn it left

posterior_visible          = 8
posterior_active_visible   = 5
posterior_inactive_visible = 3
posterior_demoted_visible  = 3
role-1 active visible      = 4
role-1 inactive visible    = 3
role-1 min active distance = 10.38 px

aqr_active_same_role_support_overlap_max = 0.0182
aqr_same_role_support_overlap_max        = 0.1424
posterior_active_file_recycle_rate       = 0.0621
posterior_inactive_file_recycle_rate     = 0.2911
```

Interpretation:

```text
Old failure:
  multiple reserve files all flow into action/cache as if they were objects.

Step50 result:
  not supported. active-only hides the gray files, and active same-role overlap
  is low.

Remaining question:
  role-1 active files are still broad around the red/blue block and gripper
  neighborhood. If this persists at step100/150, the fault is task-conditioned
  target/contact-owner selection, not gray reserve leakage.
```

## 10. Wrist Pose Versus Contact Point

The overlay also raises a second concern: an active file may bind to the wrist
or gripper neighborhood rather than the manipulated object/contact point.

This cannot be decided from color alone. The correct dataflow distinction is:

```text
wrist view / gripper pose:
  proximal evidence. It says "where the robot hand sees or is".

contact / point owner:
  interaction evidence. It says "what object/contact patch is being acted on".

task language:
  target evidence. It says "which object among possible owners matters".
```

The current design already avoids the most dangerous shortcut: wrist temporal
tokens are typed view evidence, not static-world geometry truth unless an
explicit projection model exists. Therefore a bad active orange placement near
the wrist should not be "fixed" by treating wrist pose as object ground truth.

If step100/150 active-only overlays show active role-1 files consistently
near the wrist and not the object/contact region, the coherent repair is:

```text
1. keep wrist as typed proximal evidence;
2. strengthen contact-owner or point-owner evidence for object files;
3. use task-conditioned owner selection to choose among active object files;
4. do not turn wrist pose into hard object labels.
```

That repair is compatible with the POMDP belief view:

```math
b_t(s)
\propto
p(o_t^{static}, o_t^{wrist}, o_t^{point}, o_t^{tactile}, \ell \mid s_t)
\int p(s_t\mid s_{t-1},a_{t-1})b_{t-1}(s_{t-1})ds_{t-1}.
```

Wrist evidence contributes to the likelihood, but object identity must still be
resolved by current support, contact/point evidence, task language, and posterior
continuity.

## 11. DIAS Code Check And Why We Do Not Paste It

Additional paper-code snapshot:

```text
/tmp/picf_paper_code_20260516/DIAS
```

Relevant code facts:

```text
object_centric_bench/model/ocl.py:
  SlotAttention uses inverted attention:
    softmax is taken over slots/queries for each token, then renormalized.
  This makes tokens compete for slots and is explicitly designed to reduce
  redundant slot claims.

object_centric_bench/model/dias.py:
  DIAS = Slot Attention with Re-Initialization and Self-Distillation.
  It keeps all attentions, uses slot masks, and decodes/reconstructs feature
  tokens from slots.

config-dias/*.py:
  uses NormalShared slot initialization and distillation scheduling.
```

What transfers to PICF:

```text
1. token-to-slot competition is essential;
2. invalid/redundant slots must be represented as masked/no-object capacity;
3. reserve slot reinitialization/retirement is a structural object-file issue,
   not just a scalar diversity-loss issue;
4. self-distillation can be useful only when the teacher signal is trustworthy.
```

What does not transfer directly:

```text
DIAS is an object-centric reconstruction/segmentation model. PICF is a
posterior belief router inside a VLA action pipeline. Directly pasting DIAS
reconstruction losses would add a new objective that is not aligned with
CALVIN action/state evidence and could reintroduce the loss-conflict problem.
```

Therefore PICF adopts the compatible structural semantics:

```text
posterior file competition,
active/reserve no-object state,
downstream active gate,
optional offline IsSameObject/probe diagnostics,
and future reset/reinitialization only if active-only evidence proves reserve
capacity still harms action-visible binding.
```

## 12. Background Tokens And Gray Reserve Files

Question:

```text
If gray reserve files bind background/buttons, do they steal tokens from active
orange object files? If they do not bind background, does the policy lose the
background?
```

Current code-level answer has three layers.

Layer 1: typed memories are not consumed.

```text
V-JEPA / PG / point / tactile tokens are attention memories. A gray reserve
file reading a token does not erase that token for other active files, task
readout, or PI0.5. This is not a hard DETR one-token-one-slot assignment.
```

Layer 2: posterior reserve files are not action-visible object files.

```text
posterior.file_competition_active
  -> _posterior_file_active_gate
  -> posterior tokens multiplied by downstream_gate
  -> inactive files excluded from posterior AQR read, control posterior prefix,
     global posterior pooling, and evidence cache writes.
```

Therefore many gray posterior circles in `with_gray` are expected fixed
capacity. They represent reserve/no-object state and dustbin lifecycle, not
active object files.

Layer 3: AQR graph anchors use tri-state downstream routing before the control
prefix.

The binary version:

```python
graph_tokens = graph_tokens * graph_active
```

was rejected as too coarse. It protected the action path from duplicate
reserve files, but it made "inactive but real context object" indistinct from
"no-object/dustbin reserve". The corrected code applies:

```python
graph_tokens = graph_tokens * graph_downstream_weight
```

with:

```text
active object file:
  graph_downstream_weight = 1.0

context object file:
  graph_downstream_weight = aqr_context_slot_weight, default 0.15

reserve/dustbin file:
  graph_downstream_weight = 0.0
```

This closes the semantic leak where duplicate reserve anchors could contribute
as full graph-prefix tokens, while preserving low-weight context evidence for
real but non-target scene objects such as buttons, lamp, drawer, or table
features.

Background is still visible through the correct channels:

```text
1. task/local/global readout over visual/point/semantic memories;
2. raw PaliGemma/PI0.5 semantic pathway;
3. global posterior and innovation context;
4. future task-specific activation if a background button/lamp becomes the
   prompted object.
```

So the intended semantics are:

```text
active orange/green/etc. files:
  object/contact candidates for posterior/action.

context files:
  real scene objects with lower action-prefix weight; they are retained as
  context without being forced to compete as target-owner posterior files.

gray reserve files:
  no-object/background capacity and lifecycle evidence, visible in diagnostics
  but not action-prefix object evidence.

background tokens:
  retained as contextual memory, not forced into object files.
```

This is the coherent object-centric compromise: do not force every scene patch
to become an object slot, but do not delete background context from the policy.

# PICF-AQR-OWM Full Binding Math/Dataflow Audit

Status: temporary follow-through note for the 2026-05-16 binding/posterior
repair. Canonical entry remains
[`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md).

## Scope

This note audits the binding stack end-to-end:

```text
observation modalities
  -> typed token field
  -> AQR candidate owners
  -> observation anchors
  -> posterior binding logits
  -> Sinkhorn + no-object transport
  -> posterior file competition
  -> lifecycle/recycle calibration
  -> posterior object files
  -> PI0.5 action prefix
```

It addresses the failure exposed by A7 overlays: several orange role-1
posterior object files projected to the same physical/contact region even when
active graph owners looked separated.

## Paper-Code Cross-Check

The pulled reference implementation is:

```text
/tmp/vit-object-binding
git sha: 014c66b45ea262f9b6eec83ff388a1e1c10dfcaa
```

The paper "Does Object Binding Naturally Emerge in Large Pretrained Vision
Transformers?" (`https://arxiv.org/abs/2510.24709`) reports that ViTs can
encode a pairwise property `IsSameObject(i,j)` and that quadratic probes decode
it best. The local code implements these probe families:

```text
DiagonalQuadraticProbe
QuadraticProbe
QuadraticFixedRankProbe
```

```math
s_{diag}(i,j) = w^\top (z_i \odot z_j) + b
```

```math
s_{full}(i,j) = z_i^\top W_{sym} z_j + b,\quad
W_{sym}={W+W^\top \over 2\sqrt d}
```

```math
s_{rank}(i,j) =
z_i^\top {U^\top V + V^\top U \over 2\sqrt d} z_j + b
```

The reference trainer uses instance masks:

```math
y_{ij}=1[\operatorname{mask}(i)=\operatorname{mask}(j)]
```

and optimizes pairwise BCE. That is a supervised probe/evaluation setup, not a
safe online loss for CALVIN without instance labels.

PICF therefore uses the paper in two constrained ways:

```text
runtime:
  support-weighted binding_signature + projected/quadratic pairwise scores.

audit:
  offline IsSameObject-style probe on anchor debug/overlay artifacts.
```

It intentionally does not add online weak-label BCE from noisy pseudo labels.
That would be an uncalibrated loss and would violate the current no-mask data
contract.

## PICF Binding Model

For persistent file `j` and observation anchor `i`, the intended posterior
binding score is:

```math
\ell_{j,i}
=
\lambda_h \cos(h_j,o_i)
- \lambda_g D_M(x_i,\mu_j)
+ \lambda_s g_j O^{support}_{j,i}
+ g_j S^{sameobj}_{j,i}
+ \lambda_a g_j \cos(a_j,\tilde a_i)
+ b_{role}
+ b_{owner}
+ b_{occupancy}
```

where:

```math
g_j = \alpha_j (1-r_j)\exp(-\kappa\nu_j)
```

and `S^{sameobj}` is the calibrated projected/quadratic binding-signature
score. The calibration must reject common-mode row/column bias:

```math
\hat S =
\operatorname{zscore}
\left(S - \operatorname{rowmean}(S) - \operatorname{colmean}(S)
+ \operatorname{mean}(S)\right)
```

This is why `scripts/picf_binding_dataflow_math_audit.py` and
`scripts/picf_binding_logit_calibration_audit.py` are required. A large raw
quadratic score is not enough; it must encode relative pair structure.

## Dataflow Follow-Through

### File-by-file runtime map

```text
src/openpi/picf/core/config.py
  Defines the maintained structural switches:
    posterior_file_competition_enabled
    posterior_owner_active_gate_enabled
    aqr_same_role_support_competition_enabled
    aqr_active_slot_filter_enabled
    legacy_local_refinement_opt_in = False

src/openpi/picf/core/contracts.py
  Carries the state needed to separate object files from reserve capacity:
    PicfPosteriorAnchorState.file_competition_active
    PicfPosteriorAnchorState.file_competition_demoted_mass
    PicfPosteriorAnchorState.support_signature
    PicfPosteriorAnchorState.binding_signature

src/openpi/picf/core/pipeline.py
  Implements the belief-state math:
    _build_aqr_anchor_graph            typed evidence -> observation owners
    _posterior_file_competition        duplicate files -> no-object/dustbin
    _posterior_file_active_gate        active-file objectness for downstream
    _posterior_update                  prior + measurement -> posterior files
    _build_conditioned_control_state   active posterior -> PI prefix
    _build_physical_predictive_basis   active posterior -> predictive basis
    _write_evidence_cache              active posterior -> valid cache row

scripts/picf_core_train.py
  Implements diagnostics without changing training loss:
    _anchor_overlay_snapshot_from_output
    _save_anchor_overlay_diagnostic
    with_gray / active_only PNG variants
    metrics.jsonl logging for active/inactive lifecycle splits

scripts/verify_picf_owm_contract.py
  Fails the checkout if these invariants disappear.

src/openpi/picf/core/pipeline_test.py
  Contains targeted runtime tests for inactive-file downstream gating and file
  competition math.
```

The intended tensor contract is:

```text
file_competition_active is not a visualization-only field.
It is the objectness gate used by every path that can affect future action.
```

```math
z^{control}_j = active_j\,W_c z^{post}_j
```

```math
z^{pred}_j = active_j\,W_p z^{post}_j
```

```math
valid^{cache}_j = 1[active_j > 0.5]
```

```math
bias^{AQR-post}_{q,j} =
\begin{cases}
0, & role(q)\sim role(j)\ \land\ active_j=1\\
-10^4, & otherwise
\end{cases}
```

### 1. Visual and wrist evidence

```text
rgb_static/rgb_gripper
  -> V-JEPA typed temporal maps with view_ids
  -> PicfTemporalVisualSupportState
  -> AQR temporal reader
  -> graph.vjepa_temporal_priors
```

Wrist view is typed evidence. Without calibrated wrist extrinsics it must not
be projected as static-camera geometry truth. Posterior geometry still comes
from point/world observation anchors:

```math
x^{obs}_j =
{\sum_i B_{j,i} x^{anchor}_i \over \sum_i B_{j,i} + \epsilon}
```

So the observed orange-circle failure is not "wrist position writes object
files directly"; it is multiple persistent files receiving the same posterior
measurement mixture.

### 2. AQR graph owners

```text
typed supports
  -> AQR physical/task queries
  -> same-role support competition
  -> active slot filter
  -> observation anchors
```

Active graph owners can be healthy while raw reserve candidates remain highly
overlapping. Therefore diagnostics must separate:

```text
aqr_active_same_role_support_overlap_max
aqr_same_role_support_overlap_max
posterior file overlay distances
```

### 3. Posterior binding before repair

Before the 2026-05-16 repair:

```text
bind_logits[K_files, N_obs]
  -> _sinkhorn_dustbin
  -> support_raw[K_files, N_obs]
  -> posterior update
```

The transport included an observation dustbin row but no persistent-file
no-object column. This forced every persistent file row to receive nonzero
measurement mass.

When there are fewer real objects than persistent files:

```math
K_{files} > K_{objects}
```

the model can lower action loss while duplicating object files around the same
contact owner. This explains the A7 overlays.

### 4. Posterior file competition after repair

The maintained repair adds a file/no-object step:

```text
support_raw, dustbin_raw
  -> same-role duplicate support/geometry test
  -> duplicate files demoted to no-object
  -> dustbin mass receives demoted measurement mass
```

For files `i,j`:

```math
O^{support}_{ij}
=
{p_i^\top p_j \over \|p_i\|\|p_j\|},\quad
p_i={B_i \over \sum_n B_{i,n}+\epsilon}
```

```math
O^{geom}_{ij}=\exp\left(-{\|x_i-x_j\|^2 \over 2\sigma^2}\right)
```

Duplicate files are demoted if same-role overlap is high and they are not the
best available owner for that role. Demotion is mass-conserving:

```math
B'_i = a_i B_i,\quad
d' = d + \sum_i (1-a_i)B_i
```

This is a model-level assignment correction, not a new loss.

## Loss Interaction Audit

The observed loss problems map to distinct mechanisms:

```text
slot_jepa divergence:
  detached future target exists, but loss remains default 0 until matching and
  identity diagnostics are stable.

support_pred instability:
  same reason; default 0.

binding_consistency:
  permutation-tolerant soft matching is implemented, but default 0 because
  behavior evidence is still pending.

action pressure degrading binding:
  action still trains, but PI prefix stop-gradient can prevent action gradients
  from rewriting PICF binding features directly.

same-role support overlap:
  graph-level overlap is handled by support competition/active filtering.

posterior orange duplicates:
  handled by posterior file competition.

local refinement:
  archived opt-in only; diagnostics showed it can add gradient/recycle pressure.

cache stale/double read:
  fixed by residual scaling and skipping immediate previous posterior row.
```

The current principle is:

```text
use structural assignment fixes for assignment bugs;
keep high-risk prediction/identity losses guarded until overlays and metrics
show stable object files.
```

## What Is Not Claimed

This repair does not claim:

```text
ordinal/fourth-object grounding is solved;
tracklet/proposal evidence is active without upstream tensors;
online IsSameObject labels exist;
CALVIN behavior has passed;
all sub-token objects can be recovered without observable evidence.
```

The information bound remains:

```math
I(Y;A_t \mid \ell) \le I(Y;Z_{\le t}\mid \ell)
```

The repair reduces assignment error in `Z -> posterior files -> A`; it does not
create missing evidence in `Z`.

## Executable Checks

Required local checks:

```bash
python -m py_compile \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  scripts/picf_core_train.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_posterior_file_competition_audit.py

python scripts/verify_picf_owm_contract.py
python scripts/picf_owm_strict_diagnose.py --fail-on-fail
python scripts/picf_owm_dataflow_trace.py --fail-on-fail
python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
uv run python scripts/picf_posterior_file_competition_audit.py
```

The file-competition audit must pass:

```text
same_support_duplicate_demotes_one_file
measurement_mass_is_conserved
distinct_support_keeps_capacity
geometry_duplicate_demotes_even_with_distinct_support
```

## Runtime Acceptance For The Current Diagnostic

The current A7 diagnostic is:

```text
tmux: picf_a7_file_comp_diag600
run:  /mnt/checkpoints/picf_core/picf_core/picf_a7_posterior_file_competition_diag600_20260516
log:  /mnt/picf_run_logs/picf_a7_posterior_file_competition_diag600_20260516.log
```

Acceptance is not based on loss alone. At step 50/100/150, inspect:

```text
posterior_file_competition_active_count
posterior_file_competition_demoted_mass_mean/max
posterior_file_competition_duplicate_overlap_max
posterior_active_file_potential_swap_rate
aqr_active_same_role_support_overlap_max
aqr_same_role_support_overlap_max
loss_action_default_equiv
anchor_overlays/*.json and *.png
```

The key image-level criterion is:

```text
orange role-1 posterior files should not keep multiple sub-pixel duplicates on
the same contact/object location across steps.
```

The overlay must use the posterior file-competition state, not the older
alpha/support/recycle approximation. A demoted persistent file is reserve
capacity, so `anchor_overlays/*.json` records:

```text
active = posterior.file_competition_active
file_competition_demoted_mass = posterior.file_competition_demoted_mass
```

and the PNG draws demoted posterior capacity in gray with an `i` suffix. If a
future overlay draws every orange file active while
`posterior_file_competition_active_count < persistent_anchors`, the diagnostic
path is wrong even if the model path is right.

If active graph anchors are healthy but posterior role-1 circles duplicate
again, the remaining bug is downstream of AQR and inside posterior lifecycle or
file competition. If both active graph and posterior collapse, the bug is
upstream in AQR support ownership.

## Fixed-Overlay Step50 Result

The first fixed-overlay diagnostic separates the layers:

```text
run:
  picf_a7_posterior_file_competition_overlayfix_diag300_20260516

step50 metrics:
  aqr_active_same_role_support_overlap_max = 0.0167
  aqr_same_role_support_overlap_max        = 0.1435
  posterior_file_competition_active_count  = 4.47
  posterior_file_competition_demoted_mean  = 0.274
  posterior_file_competition_duplicate_max = 0.998
  posterior_identity_switch_rate           = 0.704

step50 overlay JSON:
  visible posterior role-1 files = 7
  active posterior role-1 files  = 3
  demoted posterior role-1 files = 4
  min active role-1 distance     ~= 15.6 px
  min all role-1 distance        = 0.0 px
```

This means:

```text
1. The old "many orange active files on one object" conclusion was partly a
   diagnostic-path false positive; demoted reserve files were drawn as active.
2. The model-side no-object/file-competition repair is active and mass-
   conserving.
3. The remaining problem is identity continuity, not active AQR support
   collapse at this checkpoint.
```

Therefore the next diagnostic should track whether:

```text
posterior_identity_switch_rate falls;
posterior_active_file_potential_swap_rate falls;
posterior_file_competition_duplicate_overlap_max falls or remains safely
  isolated to demoted reserve files;
active posterior role-1 files remain separated in overlay JSON/PNG.
```

## Fixed-Overlay Step100 Result

Step100 of the same diagnostic keeps the important layer separation:

```text
step100 metrics:
  aqr_active_same_role_support_overlap_max = 0.0115
  aqr_same_role_support_overlap_max        = 0.2462
  posterior_file_competition_active_count  = 4.37
  posterior_file_competition_demoted_mean  = 0.258
  posterior_file_competition_demoted_max   = 0.607
  posterior_file_competition_duplicate_max = 0.999
  posterior_identity_switch_rate           = 0.743
  posterior_identity_switch_rate_stable    = 0.516
  posterior_recycle_rate                   = 0.221
  posterior_active_file_potential_swap_rate= 0.320
  loss_action_default_equiv                = 0.0809
  loss_action_active7                      = 0.351
```

The overlay JSON also remains consistent with the model-side active-file state:

```text
prompt:
  turn on the yellow lamp

visible posterior role-1 files:
  active = 2
  demoted/inactive reserve = 5

min active role-1 pixel distance:
  ~= 40.0 px

min all visible role-1 pixel distance:
  0.0 px, but only among demoted reserve files
```

This confirms the previous visual failure mode has been split into two separate
facts:

```text
resolved:
  Active AQR supports are not same-role collapsed at step50/100.
  Duplicate persistent role-1 posterior capacity is demoted and no longer
  shown as active orange object files.

not resolved:
  Posterior identity continuity is still weak during warmup / early full-LR
  transition. The switch and recycle metrics remain too high to justify a
  30k-step run without at least step150/300 evidence.
```

The current diagnosis is therefore not "binding solved"; it is:

```text
The old AQR-collapse and overlay-false-positive hypotheses are falsified for
step50/100. The remaining bottleneck is posterior file lifecycle stability
under action cotrain.
```

## Fixed-Overlay Step150 Result

Step150 confirms the same diagnosis, but with a stronger lifecycle warning:

```text
aqr_active_same_role_support_overlap_max = 0.0838
aqr_same_role_support_overlap_max        = 0.5314
posterior_file_competition_active_count  = 4.11
posterior_identity_switch_rate           = 0.781
posterior_recycle_rate                   = 0.309
posterior_active_file_potential_swap_rate= 0.281
loss_action_default_equiv                = 0.0773
```

The image-level state remains separated for active role-1 files:

```text
active posterior role-1 files = 2
demoted posterior role-1 reserve files = 5
min active role-1 pixel distance ~= 24.6 px
min all role-1 pixel distance = 0.0 px, again only among demoted reserve files
```

Thus the failure decomposition is now:

```text
not active AQR collapse:
  active same-role support overlap remains low enough and active posterior
  files are visibly separated.

not overlay false positive:
  demoted reserve files are correctly inactive and carry demoted mass.

remaining real issue:
  lifecycle/recycle is too eager under early action cotrain. The model keeps
  enough active files separated, but their identity continuity is not stable.
```

If step300 does not reduce `posterior_identity_switch_rate` and
`posterior_recycle_rate`, the correct next repair is a posterior lifecycle
hysteresis/survival model. That repair is mathematically different from adding
another support-diversity term: it should slow file death/rebirth when current
evidence is merely ambiguous, while still allowing recycle under high
innovation or true no-object evidence.

One diagnostic correction was added after inspecting step150: all "active file"
continuity metrics must use `posterior.file_competition_active`, not just
owner/alpha/support/recycle heuristics. Otherwise reserve files can still leak
into lifecycle summaries even though the overlay correctly marks them inactive.
The runtime now also logs:

```text
posterior_active_file_recycle_rate
posterior_inactive_file_recycle_rate
```

This split is required before changing the lifecycle model. If high mean
recycle is mostly inactive reserve capacity, it is not a training blocker. If
active-file recycle is high, the lifecycle model itself is still too eager.

## Downstream No-Object Exposure Fix

The step150 overlay also exposed a second, more subtle follow-through issue.
File competition demoted duplicate posterior capacity correctly, but downstream
control/predictive/cache paths still consumed `posterior.tokens` as a fixed
list. That means gray reserve files could still enter:

```text
posterior -> control posterior tokens -> PI prefix reader -> action generator
posterior -> predictive physical tokens -> future state prediction
posterior -> evidence cache write -> future AQR cache read
previous posterior tokens -> AQR posterior reader
```

This is mathematically inconsistent with the no-object/file-competition model.
A demoted persistent file is capacity, not current object evidence. The fix is
therefore not a new loss and not a hand-tuned heuristic; it is the downstream
objectness gate implied by the file-competition posterior:

```math
g_j = active^{file}_j \in \{0,1\}
```

and for downstream exposure:

```math
\tilde z_j = g_j z_j.
```

The implementation now applies this gate to:

```text
1. posterior self-attention keys/outputs before global_post pooling;
2. AQR posterior-read attention bias for previous posterior files;
3. conditioned-control posterior tokens before role embeddings reach PI prefix;
4. predictive posterior tokens;
5. evidence-cache valid mask for the current posterior row.
```

This addresses the user's observed failure mode directly:

```text
Gray reserve files may still be drawn for debugging, but they no longer compete
as object tokens in the action or predictive path.
```

The targeted regression check is:

```text
test_posterior_inactive_files_are_gated_from_downstream_reads
```

This is the same structural idea as no-object query handling in DETR-style
set prediction: unused query capacity is allowed to exist, but it must not be
treated as an active object output.

## Dual Overlay Contract

The training overlay is now explicitly two-view:

```text
step_xxxxxx__<task>__with_gray.png
step_xxxxxx__<task>__active_only.png
step_xxxxxx__<task>.json
```

The two PNGs answer different questions:

```text
with_gray:
  Shows the full fixed-capacity posterior/graph set. Gray `i` markers are
  reserve/no-object files. This is the correct view for auditing whether
  posterior capacity is being demoted rather than deleted.

active_only:
  Hides inactive reserve files. This is the correct view for deciding whether
  action-visible active posterior files are actually on the target object or
  contact region.
```

The JSON remains the source of truth and records every projected and
non-projected file:

```text
active
support_mass
recycle_gate
file_competition_demoted_mass
support_signature
binding_signature
```

Mathematically, this separates the two sets:

```math
\mathcal F_{all} = \mathcal F_{active} \cup \mathcal F_{reserve}
```

```math
\mathcal F_{active} = \{j : active^{file}_j > 0.5\}
```

Only `\mathcal F_{active}` should be interpreted as current object files.
`\mathcal F_{reserve}` is capacity and should not be counted as object collapse
unless it leaks into downstream computation. The downstream gate above is the
runtime repair; the dual overlay is the diagnostic repair.

## Current Loss/Overlay Failure Hypotheses

The user's latest manual inspection was:

```text
push red block right:
  many posterior/anchor markers are near the red block;
  many are gray;
  some orange active posterior markers appear off the red block.
```

The hypotheses must be separated:

```text
H1: gray reserve files steal action/predictive attention.
  Status before downstream gate: true risk.
  Status after downstream gate: structurally repaired; verify at step50+.

H2: active files are bound to wrist instead of the contact/object.
  Status: not the first explanation. Wrist is typed temporal evidence; without
  extrinsics it is not written as static geometry truth. If active files stay
  off-object after gray gating, inspect owner/contact geometry and observation
  anchor selection.

H3: too many fixed slots hide the object evidence.
  Status: only if inactive slots leak downstream. Fixed capacity is acceptable
  when reserve/no-object files are gated.

H4: action loss overrides binding.
  Status: mitigated by action-prefix stop-gradient and guarded auxiliary losses,
  but still requires runtime evidence because action can favor a useful contact
  proxy over human-interpretable object centers.
```

The immediate acceptance metric for this repair is not `loss_total` alone:

```text
posterior_active_file_recycle_rate:
  should be materially lower than all-file recycle if reserve files dominate
  churn.

posterior_inactive_file_recycle_rate:
  may be high; reserve capacity is allowed to recycle.

aqr_active_same_role_support_overlap_max:
  should remain low.

active_only overlay:
  active orange posterior files should not duplicate on one pixel and should
  track target/contact evidence better than the with_gray view suggests.
```

## Reference Paper-Code vs PICF Runtime

The pulled object-binding reference code is supervised/probe-oriented:

```text
/tmp/vit-object-binding/src/utils/models.py:
  DiagonalQuadraticProbe
  QuadraticProbe
  QuadraticFixedRankProbe

/tmp/vit-object-binding/src/trainer.py:
  labels_pairwise = labels[:, :, None] == labels[:, None, :]
  BCEWithLogitsLoss(pairwise_similarity, labels_pairwise)
```

The reference training target requires instance masks or equivalent object
labels:

```math
y_{ij}=1[o_i=o_j].
```

PICF does not have those labels in CALVIN, so the correct import is structural,
not supervised:

```text
adopt:
  projected/quadratic same-object subspace as a binding logit term;
  offline IsSameObject-style probes on exported support/binding signatures;
  active/no-object capacity split.

do not adopt directly:
  online BCE from pseudo object labels;
  hard mask supervision;
  any loss that treats noisy wrist/contact heuristics as object ground truth.
```

This is why the maintained fix is a no-object downstream gate and diagnostic
overlay split, not a new pseudo-label loss. It preserves the current belief
filter:

```math
b_t = Correction(Prediction(b_{t-1},a_{t-1}), Evidence(o_t))
```

and only changes whether reserve capacity is exposed downstream after the
posterior has already made a no-object decision.

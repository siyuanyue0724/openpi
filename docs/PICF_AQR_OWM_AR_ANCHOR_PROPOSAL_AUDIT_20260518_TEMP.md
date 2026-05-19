# PICF-AQR-OWM AR Active-Anchor Proposal Audit

Date: 2026-05-18
Status: disabled-by-default runtime prototype is present; this is not enabled in
the current training default and is not accepted for a 30k run until short
fixed-vs-VCAP diagnostics pass.

This document audits the proposal to replace the current fixed overcomplete
active anchor bank with an autoregressive active-anchor proposal mechanism. The
goal is to decide whether the idea is mathematically coherent, whether it is
supported by established or recent literature, and where it can enter the
current PICF-AQR-OWM code without becoming an ad-hoc patch.

## Short Verdict

The proposal is coherent only under this definition:

```text
Autoregressive active-anchor proposal is a variable-cardinality measurement
hypothesis generator. It proposes active object-owner candidates before AQR
posterior correction. It does not replace dense typed memory, posterior belief,
cache policy, or the PI0.5 action generator.
```

Under that definition it is a natural extension of the current system:

```text
fixed query bank
  -> AQR typed evidence read
  -> active/context/reserve downstream routing
  -> posterior file competition and birth competition
```

Implementation status as of the 2026-05-18 deployment pass:

```text
implemented:
  disabled runtime config
  PicfActiveProposalState contract
  padded VCAP query initialization
  straight-through stop-threshold active gate
  action-gradient guard
  proposal coverage/duplicate/count/continuity diagnostics
  VCAP transition-loss hooks, default zero
  trainer CLI/metrics and verifier coverage
  tensor smoke for enabled-path shape/finalization

not promoted:
  production default
  30k training claim
  replacement for posterior file identity
```

becomes:

```text
variable active proposal sequence
  -> padded fixed-shape AQR typed evidence read
  -> posterior matching / birth / no-object reserve
  -> active/context/reserve downstream routing
```

This is not a loss patch. It is a capacity-allocation change. It directly
targets the repeatedly observed failure where too many fixed same-role anchors
compete for a small number of high-evidence object regions.

Second-pass adversarial conclusion:

```text
The idea is not automatically superior because it is autoregressive.
It is only superior if it changes the capacity-allocation model while preserving
the existing belief-filter invariants.  If it becomes a direct object decoder,
or if it prunes dense typed memory, or if it trains object count without a
coverage constraint, it is worse than the current fixed-query path.
```

Therefore the accepted design claim is deliberately narrow:

```text
AR active-anchor proposal is a plausible vNext root-fix candidate for fixed
overcomplete active-query pressure.  It is not a current proof of solved object
binding, and it is not a reason to remove the existing posterior competition
machinery.
```

## Current Code Reality

The current runtime still starts from a fixed learned query bank:

```text
config.py:
  aqr_query_count_physical = 16
  aqr_query_count_task = 8

pipeline.py:
  self.aqr_physical_query_tokens
  self.aqr_task_query_tokens
  _build_aqr_anchor_graph(...)
  _aqr_active_slot_mask(...)
  _aqr_downstream_slot_weights(...)
  _posterior_file_competition(...)
  _posterior_birth_competition(...)
```

This is already a guarded fixed-capacity design. The active/context/reserve and
birth/no-object logic prevents reserve capacity from entering action as an
active object. It does not eliminate the upstream fact that many graph rows are
constructed and allowed to read the same high-evidence support before they are
demoted.

The observed repeated pattern is therefore consistent with the code:

```text
fixed overcomplete graph rows
  -> several same-role rows read similar support
  -> active filters demote many rows
  -> raw overlap can remain high
  -> active overlap can still rebound when active selection is unstable
```

AR active proposal addresses the first line of that chain instead of repeatedly
patching the last line.

## Mathematical Definition

Let current typed memory be:

```math
M_t =
\{M^{pg}_t, M^{vjepa}_t, M^{point}_t, M^{tactile}_t,
  M^{post}_{t-1}, M^{cache}_{t-1}, M^{track}_t, M^{prop}_t\}
```

and previous posterior object-file belief be:

```math
b_{t-1} = \{a_{t-1,j}, c_{t-1,j}, \mu_{t-1,j}, \Sigma_{t-1,j},
             \alpha_{t-1,j}, r_j\}_{j=1}^{K_{file}}.
```

The AR proposal model should generate a variable set of active measurement
hypotheses:

```math
p(N_t, \tilde z_{1:N_t} \mid M_t, b_{t-1}, \ell)
=
\prod_{m=1}^{N_{max}}
p(\tilde z_m, stop_m \mid M_t, b_{t-1}, \ell, \tilde z_{<m})
```

where a proposal contains only measurement-hypothesis fields:

```math
\tilde z_m =
(\tilde q_m, \tilde a_m, \tilde \mu_m, \tilde r_m,
 \tilde \gamma_m, \tilde \sigma_m).
```

These are not posterior truth. They are query initializers and birth priors.
After generation, proposals are padded to a fixed maximum:

```math
\tilde Z_t^{pad} \in \mathbb{R}^{K_{max} \times d}
```

so the existing AQR and posterior tensor contracts remain stable.

## Posterior Matching Contract

Generated proposals must be matched to old object files before they can become
belief updates:

```math
C_{m,j}
=
\lambda_g d_{\Sigma}(\tilde \mu_m, \mu^-_j)
+ \lambda_s d(\tilde \sigma_m, \sigma^-_j)
+ \lambda_a (1-\cos(\tilde a_m, a^-_j))
+ \lambda_r 1[\tilde r_m \ne r_j]
- \lambda_o owner\_evidence_m.
```

Then:

```text
matched proposal + matched old file:
  posterior correction/update

unmatched high-confidence proposal:
  bounded birth candidate

unmatched old file:
  context/reserve/no-object, not action-visible active object
```

This preserves the current posterior-authority invariant:

```text
proposal is measurement hypothesis
posterior is current belief
action sees corrected belief, not raw proposal
```

## Loss Design

A naive "fewer anchors is better" loss is wrong. It would collapse the model to
under-generate anchors. The count cost must be weak and paired with coverage.

The acceptable objective is:

```math
L =
L_{action}
+ \lambda_{cov} L_{unexplained}
+ \lambda_{dup} L_{duplicate}
+ \lambda_N L_{count}
+ \lambda_{temp} L_{continuity}
+ \lambda_{birth} L_{birth/noobj}
```

where:

```math
L_{unexplained}
=
\sum_i w_i^{evidence}
\left(1 - \max_m p(i \mid \tilde z_m)\right)
```

penalizes missing important evidence, and:

```math
L_{duplicate}
=
\sum_{m<n, r_m=r_n}
\operatorname{overlap}(\sigma_m,\sigma_n)
\;+\;
\operatorname{geomdup}(\mu_m,\mu_n)
```

penalizes multiple active proposals explaining the same object evidence.

The count term should be:

```math
L_{count} = \sum_m p(active_m)
```

with a small weight. It is a capacity prior, not the main objective.

## Literature Alignment

The proposal is supported by existing patterns, but not as a blind copy.

DETR supports set prediction with learned object queries and bipartite matching.
Its key lesson for PICF is not "use fixed queries forever"; it is that object
ownership needs a set-level uniqueness constraint rather than independent row
classification.

Pix2Seq supports representing object detection as sequence generation. Its key
lesson for PICF is that variable-length object readout with an end token is a
standard way to express "generate only the needed objects".

The 2025 ViT object-binding work supports pairwise IsSameObject-style subspaces.
Its key lesson for PICF is that anchor proposal and binding should not be raw
hidden cosine only; they should consume projected same-object support signals.

MetaSlot and related variable-slot OCL work support the diagnosis that a fixed
slot count can split one object into multiple slots or waste slots when object
count varies. Its key lesson for PICF is that duplicate removal / variable
effective cardinality is not an ad-hoc robotic patch; it is a current
object-centric learning concern.

The important negative lesson is also clear. DETR's fixed object queries work
because supervised matching and a no-object class define which predictions are
real objects. Pix2Seq works because the object sequence is trained against
explicit object descriptions and end tokens. PICF currently does not have
ground-truth object masks, boxes, ranks, or counts. Therefore PICF cannot simply
copy those training losses. It can only borrow the structural ideas:

```text
set uniqueness
end/no-object capacity
variable effective cardinality
matching before belief write
```

and must obtain supervision from internal typed evidence, posterior continuity,
task/contact evidence, and action success rather than from unavailable object
labels.

The proposal is therefore literature-aligned as:

```text
DETR: set uniqueness / matching
Pix2Seq: variable-length object sequence
ViT object binding: pairwise same-object subspace
MetaSlot: fixed-slot-count failure and duplicate slot pressure
```

## Self-Critique

This section is intentionally adversarial. The proposal is rejected unless each
risk has a clean mathematical guard.

### Risk 1: Order bias

Autoregressive generation introduces an order. Object sets are unordered.

Mitigation:

```text
Use AR only to propose candidates. Use set matching before posterior update.
Do not attach persistent identity to proposal index.
```

### Risk 2: Count collapse

A count penalty can make the model generate too few anchors.

Mitigation:

```text
Coverage / unexplained-evidence loss must be stronger than count cost.
The count term is a weak prior, not an optimization target by itself.
```

### Risk 3: Exposure bias

Early wrong proposals can condition later proposals.

Mitigation:

```text
Use teacher-forced proposal order during warmup from current posterior active
files and high-confidence support peaks. Switch gradually to free-running.
```

### Risk 4: Breaking dense V-JEPA / PG / point memory

If proposal selection masks dense memory, it can destroy world-model evidence.

Mitigation:

```text
Never prune dense typed memory. Proposals choose which object files become
active; dense tokens remain available to AQR readers.
```

### Risk 5: Replacing posterior truth

If proposals are treated as object truth, the design becomes brittle.

Mitigation:

```text
Proposal -> AQR read -> posterior matching/correction remains mandatory.
Raw proposals do not enter action prefix as posterior files.
```

### Risk 6: Runtime cost

Autoregressive decoding is sequential.

Mitigation:

```text
K_active is small. The generator can run only for active proposals while
posterior capacity remains padded. This is likely cheaper than repeatedly
reading a large overcomplete active bank.
```

### Risk 7: Hidden new supervision requirement

Autoregressive object generation looks natural in Pix2Seq because the training
target is an object sequence. PICF does not have object-count labels.

Mitigation:

```text
Do not introduce a supervised object-count objective. Train proposal count only
through coverage, duplicate, posterior matching, and weak no-object/birth
signals. If those signals are insufficient, keep the fixed-query path.
```

### Risk 8: Breaking permutation symmetry incorrectly

A pure AR order can make the first proposal always become the task object,
second proposal always become background, etc. That would be a hidden
hand-coded slot ordering.

Mitigation:

```text
Use AR order as an internal construction order only. Before posterior update,
run set matching. Persistent identity belongs to posterior file address/content,
not proposal index.
```

### Risk 9: Collapsing background/context awareness

Generating only task objects can remove non-target scene context that action
still needs.

Mitigation:

```text
Separate active object proposals from low-weight context slots. The generator
may reduce active object owners, but dense typed memory and context exposure
remain available. Background is not deleted; it is not promoted to active
object truth unless supported.
```

### Risk 10: Becoming a second policy head

If the generator is trained primarily by action loss, it can become an action
shortcut instead of an object proposal system.

Mitigation:

```text
During bring-up, stop-gradient or downweight action-flow into proposal logits.
Let action train the posterior/action path while proposal health is judged by
coverage, duplicate, matching, and overlay metrics. Only relax this after the
proposal model is stable.
```

## Fit With Existing PICF Modules

| Existing module | Integration rule |
| --- | --- |
| V-JEPA temporal tokens | remain dense typed evidence; proposals do not mask them |
| PaliGemma image/text support | provides semantic/task priors for proposal generation and AQR read |
| Sonata point tokens | provide geometry/support evidence for coverage and matching |
| AnyTouch/tactile | soft contact evidence; can bias proposal confidence but not hard object truth |
| Pairwise binding signature | should be one input to proposal novelty and posterior matching |
| posterior file competition | remains the authority that demotes duplicates |
| posterior birth competition | consumes unmatched high-confidence proposals |
| active/context/reserve routing | remains the downstream action-exposure mechanism |
| PI0.5 action | unchanged; sees posterior-corrected active/context prefix |

The proposal is therefore not a new parallel model. It is a replacement for the
front-end fixed active candidate allocator.

## Code Integration Boundary

The current code has three relevant layers:

```text
Layer A: fixed AQR graph proposal rows
  _build_aqr_anchor_graph
  aqr_physical_query_tokens / aqr_task_query_tokens
  typed memory readers

Layer B: active/context/reserve action exposure
  _aqr_active_slot_mask
  _aqr_downstream_slot_weights
  anchor_downstream_weight

Layer C: posterior object-file authority
  _posterior_file_competition
  _posterior_birth_competition
  _posterior_update
```

AR proposal may only replace or initialize Layer A's active candidate rows. It
must not bypass Layer B or Layer C. In particular:

```text
Allowed:
  generated proposal -> query initializer
  generated proposal -> active prior
  generated proposal -> birth prior
  generated proposal -> padded no-object row

Forbidden:
  generated proposal -> direct posterior file write
  generated proposal -> direct action prefix truth
  generated proposal -> dense token pruning
  generated proposal index -> persistent identity
```

This is the criterion for whether the implementation is integrated or
protruding. If a code change does not fit this boundary, it should be rejected.

## Why This Is Not Just Another Patch

Previous repairs acted after fixed query construction:

```text
support diversity:
  discourages duplicate supports after rows already exist.

active/context/reserve:
  controls which rows reach action after rows already exist.

posterior file competition:
  controls which rows update persistent files after rows already exist.

birth/no-object competition:
  controls inactive file birth after rows already exist.
```

These are correct and should stay. But they do not change the upstream fact
that the graph builds too many same-role candidate rows for scenes with fewer
task-relevant objects.

AR proposal attacks the upstream capacity mismatch:

```text
infer active candidate count and location first,
then run the same AQR/posterior machinery on a smaller active set.
```

That makes it a coherent architectural extension rather than a local
post-failure penalty.

## What Would Falsify The Idea

The idea should be rejected or delayed if a prototype shows any of these:

```text
1. unexplained evidence rises while active overlap falls;
2. proposal_count collapses to 1-2 across all tasks;
3. action loss improves but active task-object overlays lose the named object;
4. posterior_active_file_potential_swap_rate rises;
5. dense V-JEPA/PG/point/tactile support entropy becomes uninformative;
6. generator output order becomes a hidden role-label convention;
7. the old fixed-query path outperforms it under matched trainability and
   compute.
```

These falsification tests are necessary because the proposal is plausible, not
proved.

## Required Implementation Boundary

The user-facing request is for a one-step final change plan, not another
sequence of ad-hoc repairs.  The correct interpretation is:

```text
Implement the full proposal system behind one disabled-by-default switch, with
all contracts, loss terms, metrics, overlays, and rejection tests present from
the first commit.  Then enable it only for controlled diagnostics.
```

Partial implementations are rejected.  In particular, adding only a count loss,
only an EOS head, or only a generated query initializer is considered a
half-measure because it cannot prove coverage, duplicate control, posterior
authority, and action compatibility at the same time.

## Final One-Step Change Specification

Name:

```text
VCAP: Variable-Cardinality Active Proposal
```

Definition:

```text
VCAP is a disabled-by-default replacement for the active graph-query allocator.
It generates active measurement proposals and no-object/reserve rows, pads them
to the existing AQR tensor contract, and lets the unchanged AQR/posterior stack
decide final belief.
```

### 1. Config

Runtime config must contain only knobs consumed by the proposal allocator:

```python
vcap_enabled: bool = False
vcap_max_active: int = 12
vcap_min_active: int = 1
vcap_stop_threshold: float = 0.5
vcap_action_grad_scale: float = 0.0
```

The default must be `vcap_enabled=False`.  The current fixed AQR path remains
the production default until VCAP passes controlled diagnostics.

Training-pressure weights live in `PicfTransitionLossConfig`, not in core
runtime config:

```python
lambda_vcap_unexplained: float = 0.0
lambda_vcap_duplicate: float = 0.0
lambda_vcap_count: float = 0.0
lambda_vcap_continuity: float = 0.0
```

This separation is intentional: runtime decides how candidate rows are
initialized; the transition loss decides whether proposal health is optimized.
Teacher/free-run schedules are launch recipes, not inactive core knobs, until a
step-aware VCAP curriculum is explicitly implemented.

### 2. Contracts

Add a proposal state without changing posterior truth:

```python
class PicfActiveProposalState:
    tokens: torch.Tensor
    stop_logits: torch.Tensor
    active_prob: torch.Tensor
    role_logits: torch.Tensor
    address_seed: torch.Tensor
    geometry_seed: torch.Tensor
    support_signature_seed: torch.Tensor
    coverage_score: torch.Tensor
    duplicate_score: torch.Tensor
    valid: torch.Tensor
```

Thread it through `PicfAnchorPriorGraphState` as optional diagnostics:

```text
active_proposals
proposal_to_graph_assignment
proposal_unexplained_evidence
proposal_duplicate_cost
proposal_count
```

Do not add it to `PicfPosteriorAnchorState` as truth.  Posterior files remain
the only persistent object state.

### 3. Generator Inputs

The generator may attend to:

```text
previous posterior object files
current typed evidence summaries
task/PG text-image support
V-JEPA temporal support summaries
point/tactile support summaries
binding_signature summaries
optional tracklet/proposal sidecar summaries
```

It may not directly consume future posterior targets or action targets.

### 4. Generator Form

Use an autoregressive decoder only for active proposals:

```math
h_m = Dec(h_{<m}, summary(M_t), summary(b_{t-1}), \ell)
```

```math
(\tilde q_m, \tilde a_m, \tilde\mu_m, \tilde r_m, stop_m)
= Head(h_m)
```

The final set is padded:

```math
\tilde Q^{pad}_t =
[\tilde q_{1:N_t}, q^{reserve}_{N_t+1:K_{max}}].
```

Reserve rows are no-object capacity, not active object files.

### 5. AQR Integration

Replace fixed physical query initialization only when enabled:

```text
if vcap_enabled:
  q_physical = padded_vcap_queries
else:
  q_physical = self.aqr_physical_query_tokens
```

All typed memory readers remain unchanged:

```text
PG image/text reader
V-JEPA static/wrist temporal reader
point reader
tactile reader
posterior reader
cache reader
tracklet/proposal reader
```

VCAP must not prune dense memory.  It only controls which graph rows are active
measurement hypotheses.

### 6. Posterior Integration

After AQR read, use existing machinery:

```text
_aqr_active_slot_mask
_aqr_downstream_slot_weights
_posterior_file_competition
_posterior_birth_competition
_posterior_update
```

VCAP can provide priors to active/birth decisions, but cannot bypass these
functions.

### 7. Training Objective

The VCAP objective is not a standalone object detector loss:

```math
L_{vcap}
=
\lambda_{cov}L_{unexplained}
+ \lambda_{dup}L_{duplicate}
+ \lambda_NL_{count}
+ \lambda_{cont}L_{continuity}
+ \lambda_{birth}L_{birth/noobj}.
```

Action-gradient handling during bring-up:

```math
\nabla_{\theta_{vcap}} L_{action}
\leftarrow
\gamma_{action}
\nabla_{\theta_{vcap}} L_{action},
\quad
\gamma_{action}=vcap\_action\_grad\_scale.
```

Default diagnostic value:

```text
vcap_action_grad_scale = 0.0
```

This prevents the generator from becoming a shortcut policy head before object
proposal health is proven.

### 8. Teacher Forcing Without Labels

There are no object-count labels.  Teacher forcing may use only weak internal
teachers:

```text
current active posterior files
high-confidence object-core support peaks
contact/task-owner evidence
tracklet continuity if available
```

All teacher targets must be detached.  They are curriculum scaffolds, not
ground truth.

### 9. Metrics

The first implementation must log:

```text
vcap_enabled
vcap_proposal_count
vcap_stop_entropy
vcap_active_prob_mean
vcap_unexplained_evidence
vcap_duplicate_cost
vcap_count_cost
vcap_continuity_cost
vcap_matched_old_file_fraction
vcap_birth_fraction
vcap_noobject_fraction
vcap_action_grad_scale
aqr_active_same_role_support_overlap_max
aqr_active_same_role_object_core_overlap_max
posterior_active_file_potential_swap_rate
posterior_file_competition_active_duplicate_overlap_max
```

Without these metrics, VCAP cannot be accepted.

### 10. Overlays

Anchor overlays must add:

```text
generated proposal order index
stop probability
active probability
matched posterior file id if any
birth/no-object status
```

The visual contract remains:

```text
with_gray: inspect reserve/no-object capacity
active_only: inspect action-visible object files
```

### 11. Tests / Audits

Add executable checks before any training claim:

```text
vcap_disabled_matches_fixed_path:
  disabling VCAP preserves current behavior.

vcap_padding_contract:
  padded proposal tensors match existing AQR shapes.

vcap_no_dense_memory_prune:
  typed memory token counts are unchanged by VCAP.

vcap_no_direct_posterior_write:
  proposals cannot write posterior without posterior_update.

vcap_count_requires_coverage:
  count loss cannot be nonzero unless unexplained-evidence loss is present.

vcap_proposal_index_not_identity:
  posterior identity metrics use posterior file ids, not proposal order.

vcap_action_grad_guard:
  action gradient into generator is scaled by vcap_action_grad_scale.

vcap_rejection_metrics_present:
  all acceptance/rejection metrics are logged.
```

### 12. Short Diagnostic Before Long Run

Before any 30k run:

```text
Run A: fixed-query current default.
Run B: VCAP enabled with action_grad_scale=0.
Run C: VCAP enabled with small action_grad_scale after warmup.
```

Matched controls:

```text
same checkpoint
same trainability
same PaliGemma policy
same unroll/burnin
same LR
same data
same overlay interval
```

Minimum pass:

```text
proposal_count is neither constant-max nor constant-1;
unexplained evidence does not rise;
active same-role support/object-core overlap falls;
posterior active-file swap does not rise;
task-object overlays improve or remain at least as good;
action loss is not worse under matched compute.
```

## Minimal Versus Complete

The complete first commit must include all interfaces, metrics, and audits
above.  It does not need to enable VCAP in production.  It does need to make
future experiments unambiguous.

Rejected "minimal" changes:

```text
only add stop token;
only add count regularization;
only reduce aqr_query_count_physical;
only add another support-diversity penalty;
only add AR-generated query tokens but keep no coverage/duplicate metrics;
only make overlays prettier without changing proposal semantics.
```

Those changes either repeat previous failed attempts or make the system harder
to audit.

Accepted one-step final change:

```text
Add VCAP as a full, disabled-by-default, posterior-subordinate,
coverage-preserving, duplicate-aware, padded variable-cardinality proposal
layer with explicit tests and metrics.
```

This is the only version that is mathematically consistent with the current
PICF-AQR-OWM belief-state architecture.

Acceptable first implementation:

```text
1. Add disabled-by-default AR proposal module.
2. Generate at most Kmax proposal query initializers plus stop logits.
3. Pad proposals to existing AQR shape.
4. Use proposals only as query initialization / active prior / birth prior.
5. Keep existing fixed query path as fallback.
6. Keep all dense memory readers unchanged.
7. Keep posterior file competition and birth competition active.
8. Log proposal_count, stop_entropy, unexplained_evidence, duplicate_cost,
   matched_old_file_fraction, birth_fraction, and action-visible active count.
```

Unacceptable implementation:

```text
1. Directly replacing posterior files with generated proposals.
2. Hard-pruning V-JEPA/PG/point/tactile dense memory.
3. Optimizing only a count penalty without coverage.
4. Treating proposal index as persistent object id.
5. Feeding raw proposals directly to action as truth.
```

## Decision

This is a valid vNext direction. It is more mathematically principled than
continuing to add local penalties to a fixed overcomplete query bank. It should
not be enabled in the current 30k run until a short disabled-by-default
prototype proves:

```text
active proposal count is evidence-dependent;
active overlap stays low;
coverage does not drop;
posterior files remain stable;
action loss does not regress;
overlays show named task objects receiving active posterior files.
```

The strictest wording is:

```text
AR active-anchor proposal is a coherent capacity-allocation upgrade candidate.
It is not a proof that object binding is solved, and it is not a license to
remove posterior competition. It becomes self-consistent only when embedded as
a proposal/birth prior under posterior authority.
```

## References Used For This Audit

```text
DETR:
  End-to-End Object Detection with Transformers.
  Set prediction, object queries, bipartite matching, no-object class.
  https://arxiv.org/abs/2005.12872

Pix2Seq:
  A Language Modeling Framework for Object Detection.
  Object descriptions as generated token sequences with variable length.
  https://arxiv.org/abs/2109.10852

Object Binding in ViTs:
  Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
  Pairwise IsSameObject-style probes and same-object subspace evidence.
  https://arxiv.org/abs/2510.24709

MetaSlot:
  Break Through the Fixed Number of Slots in Object-Centric Learning.
  Variable effective slot counts and duplicate-slot pruning.
  https://arxiv.org/abs/2505.20772

Slot merging:
  When Slots Compete: Slot Merging in Object-Centric Learning.
  Duplicate slot pressure under fixed slot sets.
  https://arxiv.org/abs/2603.11246
```

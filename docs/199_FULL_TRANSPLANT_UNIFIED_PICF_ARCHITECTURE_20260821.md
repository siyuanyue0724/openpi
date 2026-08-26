# ADR-199: Full-Transplant Unified PICF Architecture

Date: 2026-08-21

Status: `FIRST_FULL_OPTIMIZER_STEP_PASSED_STEP2_MEMORY_PLACEMENT_RETEST`; the
complete VidEoMT backbone has passed an adapted spatial ablation and a real
five-frame causal released-weight execution receipt. The combined strict cloud
audit now also passes against the persistent LingBot, processor, VidEoMT and
DINOv3 assets. The first full released LingBot host plus donor execution on the
target two-A100 topology completed one finite full forward/backward/optimizer
update. The following update exposed a post-optimizer-state memory-placement
failure, not a scientific pass or failure. Unified multimodal binding, causal
action benefit, closed-loop CALVIN and long training remain unapproved.

## 0. Executive decision

The target is not another custom anchor head. It is the conjunction of two
complete released systems plus one explicit PICF hypothesis:

1. keep the released LingBot-VLA 2.0 model, checkpoint-compatible source,
   Qwen3-VL host, action expert, DeepStack, MRoPE, MoE, flow matching and
   current/future query-distillation path intact;
2. keep the complete released DINOv3-L VidEoMT spatial-temporal object-query
   engine intact, including all 200 queries, final four ViT blocks, query
   propagation, mask/class readouts, matcher, losses, auxiliary outputs and
   schedule;
3. test one new PICF claim: full-width VidEoMT query observations and every
   available typed modality can update persistent object rows inside the same
   large LingBot host, and factual rows causally improve action over a matched
   LingBot baseline.

The third item is not present in either donor. It must never be described as an
upstream reproduction. It is the paper hypothesis and must survive ablation.

The frozen production LingBot source remains
`2838c1862bbec1ea47942fb61512130f635eb595` with released checkpoint revision
`11c703bf6a5c1f45b3b69168482da11fdbba53d7`. A fresh audit against public head
`951475ae1b1d87553e7dc47c97b53a3d695c0d13` found byte-identical model-core
files. Public-head differences are recipe, data and deployment changes, not a
new backbone. Replacing the frozen source would therefore add provenance risk
without adding model capacity.

## 1. Non-negotiable philosophy

### 1.1 Full transplant means full transplant

For each selected donor, source topology, released dimensions, query count,
layer count, attention graph, auxiliary losses, matcher, optimizer grouping and
schedule are one normative unit. Convenience, memory limits and fast smoke
tests do not authorize removing any part.

A local change belongs to exactly one class:

- `EXACT_UPSTREAM`: copied or imported source whose values and control flow are
  byte- or numerically equivalent to the pinned donor;
- `MECHANICAL_ADAPTER`: import relocation, typed wrapper, device placement,
  distributed wrapping or injective coordinate change with a proof that no
  represented value is discarded;
- `SCIENTIFIC_ADAPTER`: any change to information flow, supervision, query
  semantics, persistence, visibility or optimization. It receives a separate
  name, derivation, control arm and acceptance gate.

No scientific adapter may be hidden inside an upstream namespace or reported
as a reproduction.

For every donor component, implementation review must answer all four questions
with executable evidence before integration:

1. Is the selected upstream file and revision frozen by hash?
2. Is the local executable byte-identical, or is its complete diff limited to a
   named import/coordinate/distributed adapter?
3. Do value-order, gradient, optimizer-state and continuation tests cover the
   behavior affected by that adapter?
4. Is every remaining difference listed as a scientific arm rather than
   silently justified as a convenience?

Failure of any item blocks promotion. A smaller query bank, shorter official
block stack, substitute decoder, dropped auxiliary loss, altered matcher,
frozen-zero-LR gradient shortcut or reduced precision recipe may be useful as a
named ablation; none is an exact transplant.

### 1.2 One large semantic and action owner

The shared LingBot Qwen3-VL/action graph owns:

- task meaning;
- cross-modal interpretation;
- physical identity and persistent belief;
- uncertainty and context use;
- task relevance;
- action generation.

VidEoMT is a large spatial-temporal observation encoder. V-JEPA/DINO-Video,
AnyTouch and Sonata are large modality encoders or teachers. None may own the
PICF task winner, persistent row identity, lifecycle or final action.

Projection and mask heads are allowed only as coordinate/readout operators.
Their outputs cannot independently decide which physical entity matters.

### 1.3 Preserve evidence instead of forcing certainty

Raw LingBot RGB/language/state evidence remains action-visible. PICF augments
that information set; it does not replace it with anchors. Every structural
VidEoMT query enters the host. No objectness threshold, top-k selector or
hand-written winner discards evidence before shared large-model reasoning.

Uncertain evidence may remain context. Overlapping objects may have overlapping
soft support. The model is not forced to assign every pixel or token to exactly
one object.

### 1.4 Missing modality means absence

If touch, 3D or temporal evidence is unavailable, its token span and its loss
term are absent. A zero tensor is not an observation. Modality availability is
typed metadata, and the same contract is used during training, evaluation and
deployment.

### 1.5 Results, not prose, authorize training

A mask-quality pass proves only the spatial engine. A finite backward pass
proves only mechanics. PICF is accepted only after matched, same-seed curves
show:

- persistent rows retain and correct physical evidence;
- same-object multimodal evidence binds better than shuffled evidence;
- factual row/history interventions change the intended action path;
- action is no worse than LingBot across the complete registered curve and is
  materially better on at least one preregistered criterion;
- closed-loop CALVIN improves or gives a defensible robustness benefit.

## 2. Current evidence and current limit

### 2.1 Passed

The DINOv3-L VidEoMT backbone has passed a useful but incomplete local spatial
ablation on hash-bound CALVIN clips:

- 315,986,985 trainable parameters;
- complete 200-query path, four shared blocks and readouts retained;
- 1,500 optimizer steps completed without non-finite values;
- all 435 trainable parameter tensors received nonzero gradients after the
  local ViT unfreeze;
- strict checkpoint reload passed;
- full 96-window held-out evaluation reached oracle hard IoU `0.629951` and
  model-ranked top-25 hard IoU `0.601570` at 224 resolution;
- the 360-resolution evaluation improved movable-block oracle masks but exposed
  scale-sensitive confidence ranking;
- source parity, data provenance, visual inspection and focused tests passed.

The run did not preserve the selected online matcher, released augmentation,
effective batch, AMP/parameter precision, 160k schedule horizon or AdamW moment
evolution during zero-LR warmup. It therefore does not close exact online
training parity and its checkpoint cannot be promoted into Stage P as the
exact donor.

This is real progress over ADR-193's coarse cosine readout. It does not prove
multimodal binding, persistent identity, task selection, action gain or
open-world generalization.

The exact one-way integration has now also run locally on five official,
manifest-authenticated CALVIN source frames. It executes the frozen 315,986,985
parameter FP32 donor and emits class logits `[1,5,200,41]`, mask logits
`[1,200,5,120,120]`, query embeddings `[1,5,200,1024]` and propagated state
`[1,200,1024]`. The host adapter retains all 200 latest queries in order. This
closes the donor/C5/precision execution mechanics only; it does not execute the
full 6B LingBot host or establish an action advantage.

### 2.2 Not passed

The following claims remain false until measured:

- the 200 short-memory VidEoMT queries map consistently to a smaller persistent
  object set;
- V-JEPA/DINO-Video, touch and 3D evidence bind to the correct object rows;
- posterior rows survive long occlusion and then re-identify;
- action uses factual rows rather than bypassing or ignoring them;
- PICF beats matched LingBot on action or closed-loop CALVIN;
- CALVIN-only adaptation transfers to broad public robot datasets.

## 3. Frozen donor matrix

| Role | Exact donor | Frozen identity | Normative content |
| --- | --- | --- | --- |
| VLA host/action | LingBot-VLA 2.0 | source `2838c186...`, checkpoint `11c703bf...` | Qwen3-VL-4B, 36-layer joint VLM/action attention, DeepStack, MRoPE, MoE, flow matching, dual-query distillation |
| Spatial object observations | VidEoMT DINOv3-L | source `025b9384...`, released weight SHA-256 `2cfa7a2d...17c04f` | 200 queries, final four complete DINOv3-L blocks, mask/class heads, temporal propagation, matcher, criterion, schedule |
| Spatial backbone | DINOv3-L | source `6876159a...` | released ViT-L/16 representation and weights |
| Temporal teacher/evidence | official LingBot DINO-Video and existing V-JEPA 2.1 arm | source/checkpoint hashes already frozen by their own ledgers | causal temporal tokens; no identity or lifecycle ownership |
| Contact evidence | AnyTouch2 | existing source/checkpoint ledger | valid-contact tokens only |
| 3D evidence | Sonata/SpatialLM path | existing source/checkpoint ledger | calibrated geometry tokens only |

VidEoMT and LingBot remain separately callable. This is required for exact
controls and for proving which component supplies an observed effect.

One upstream subtlety is preserved rather than "completed" locally: the
selected VidEoMT online constructor declares re-identification loss weights,
but the released DINOv3-L backbone path does not emit `pred_reid_embed`.
Therefore those losses are inactive in the pinned donor. A locally invented
re-identification head would be a separate model hypothesis and is forbidden in
the exact observation arm.

## 4. LingBot upstream-drift audit

The following model files have identical SHA-256 at frozen commit `2838c186`
and public head `951475ae`:

```text
9b3c9a2ad6823f9d58da51b732477785a1e4db0d29db80e030e094c7b605e40c  modeling_lingbot_vla_v2.py
af6a365f05e4ccd7d2585368412be5d186a3afdabfc6e9eaf0c97e71a51037dc  modeling_lingbot_vla.py
78d3a617b8b84647ee3811c606f6554d40efadd5ff330d5efc91a64af919fd79  depth_head.py
21e8444e7af259746c7786ec0064e8325b446305677fffebcc9569d749549ab8  resampler.py
```

The current public recipe changes batch size, schedule, gradient
checkpointing, future-query action visibility and cosine supervision. Those
changes form one complete sensitivity arm. Copying selected fields into the
frozen experiment would create an unregistered hybrid and is forbidden.

## 5. Target information graph

At control step `t`, let:

- `R_t` be raw LingBot RGB/language/state tokens;
- `V_t = {(q_t^i, m_t^i, c_t^i)}_(i=1)^200` be exact VidEoMT query, mask and
  class observations;
- `E_t^r` be valid dense evidence from modality `r`;
- `P_(t-1) = {p_(t-1)^j}_(j=1)^K` be the persistent PICF rows;
- `U_(t-1)` be executed controls;
- `F_LB` be the complete LingBot joint host;
- `G_act` be LingBot's released flow-matching action path.

The target causal graph is:

```text
V_t = F_VidEoMT(RGB_t, V_(t-1))

(H_t, P_t) = F_LB(
    R_t,
    identity_isometry(V_t.query),
    {typed_adapter(E_t^r) for observed r},
    P_(t-1),
    U_(t-1)
)

A_t = G_act(H_t, P_t, raw current evidence)
```

The mask and class tensors remain exact VidEoMT readouts for supervision and
inspection. The complete 1024-dimensional query vectors, not hand-selected
masks, enter LingBot. The shared host decides which evidence updates which
persistent row and which rows matter for action.

### 5.1 Stage P-Q: a query-value-preserving scientific interface

The exact upstream final mask is not a function of the query alone. In the
pinned source, for query `i` and final patch field `X_t`,

```text
C_t^i = Linear_class(q_t^i)
M_t^i = MaskMLP(q_t^i)^T Upscale(X_t).
```

Consequently, retaining all `q_t^i` does not permit exact reconstruction of
`M_t^i` after `X_t` is discarded. Stage P-Q therefore means exactly:

- the complete VidEoMT graph, class head, mask head, auxiliary reads and
  propagated state execute unchanged;
- all complete donor outputs remain available for the donor objective,
  receipts, evaluation and visualization;
- all 200 final query vectors enter LingBot without query selection, pooling,
  resampling or a second normalization;
- mask logits, class logits and donor patch fields are not claimed to be
  losslessly encoded in that query stream.

The last two bullets are a named `SCIENTIFIC_ADAPTER`, not an upstream
reproduction. The reason to test it first is architectural, not computational:
the object queries have already co-reasoned with donor patches in all four
released segmenter blocks, while the large shared LingBot host retains its own
raw RGB tokens and owns task-conditioned cross-modal reasoning. This hypothesis
must be rejected if the host cannot causally use the spatially supervised query
bank. It must not be rescued silently with a local mask decoder.

Any future Stage P-QM arm that injects mask support, patch fields, region pools,
or attention bias changes the host information graph. It requires its own name,
derivation, memory/throughput receipt and matched ablation. In particular,
mask-weighted pooling is a compression and mask-derived attention bias is a new
scientific prior; neither is a mechanical adapter.

### 5.2 Real-arithmetic query bridge invariant

For an upstream width `d <= D_host`, a mechanical bridge `W in R^(D_host x d)`
must have full column rank. The preferred initialization is column-isometric:

```text
W^T W = I_d.
```

Therefore `Wx1 = Wx2` implies `x1 = x2` in exact real arithmetic, so the bridge
cannot structurally collapse two distinct upstream vectors at initialization. A
learned bridge remains rank- and singular-value-monitored.

For the exact VidEoMT stream, `x` is the official final `backbone.norm` output
captured directly at the released class/mask-head input. Stage P applies no
second LayerNorm, pooling, top-k or resampler before `W`. This matters because
LayerNorm is not injective: `LN(ax + b1) = LN(x)` for positive `a`, up to
epsilon. The local ABI therefore declares an explicit `identity` coordinate
policy for VidEoMT queries and tests the raw-value equation end to end.

The runtime is not exact real arithmetic. Casting FP32 donor queries into a BF16
LingBot host is quantization and is not injective over all FP32 values. It is
allowed only as a named precision adapter with a measured cosine, relative-L2,
maximum-absolute-error and collision receipt on real batches. Until that receipt
passes, the implementation may claim no structural/dimensional loss, not
bitwise or mathematical losslessness.

This injectivity statement is deliberately not generalized to every existing
modality adapter. Existing V-JEPA, AnyTouch and Sonata paths retain their frozen
normalization/resampling contracts until separately ablated; they preserve
typed evidence and validity, but are not claimed to reconstruct arbitrary raw
encoder vectors exactly.

This proof concerns represented vectors only. It does not prove that the
upstream vector is sufficient for every task.

### 5.3 Shared-host object update

Previous rows and current evidence participate in the same per-layer Q/K/V
graph. For row `j` at host layer `l`:

```text
p_t^(j,l+1) = p_t^(j,l) + Attn_l(
    query=p_t^(j,l),
    keys=[R_t, V_t, E_t, P_(t-1), U_(t-1)],
    values=[R_t, V_t, E_t, P_(t-1), U_(t-1)]
) + FFN_l(...).
```

This is schematic notation for the released LingBot block, not a new local
attention module. Object discovery, multimodal association and task relevance
must be represented in these large-host states. A small post-host scorer may
measure them but may not feed a semantic decision back into the causal graph.

### 5.4 Set symmetry

Persistent rows are exchangeable. For any row permutation `Pi`, the physical
update must satisfy:

```text
F(R_t, V_t, E_t, Pi P_(t-1)) = Pi F(R_t, V_t, E_t, P_(t-1)).
```

Losses that compare object sets use the released Hungarian/set machinery or an
explicitly registered permutation-invariant objective. Fixed row-to-category
semantics are forbidden.

### 5.5 Object, overlap and context

Let `s_pj` be soft support of visual element `p` for row `j` and `s_p0` be
context support. The primary representation is multi-label:

```text
s_pj in [0,1],
sum_j s_pj is not required to equal 1.
```

This permits overlap, transparent/occluded boundaries and object parts. An
exclusive diagnostic may normalize `[s_p0, s_p1, ..., s_pK]`, but it cannot
write posterior state. Uncertain or unsupported evidence can remain context;
the system does not force every patch into an object row.

### 5.6 Multimodal binding

For observed modalities referring to the same physical object, the desired row
is a shared sufficient statistic rather than a forced identical vector:

```text
maximize I(P_t^j; E_t^r, E_t^s, future evidence, action | same object j)
minimize I(P_t^j; sensor identity, augmentation nuisance, unrelated context).
```

Implementation uses mature predictive/distillation and set losses, not a
learned mutual-information estimator. Critically, modality embeddings are not
required to become numerically identical. Each may retain private information;
the shared row must make cross-modal correspondence and held-out prediction
recoverable.

The official LingBot dual-query route supplies the mature pattern:

```text
L_depth = ||Proj_depth(Q_t) - D_t||_1
        + ||Proj_depth(Q_(t+T)) - D_(t+T)||_1

L_video = ||Proj_video(Q_t) - Z_t||_F^2
        + ||Proj_video(Q_(t+T)) - Z_(t+T)||_F^2.
```

Its official `TaskTokenDepthHead` is preserved in full. It is a loss/readout
projection, not an identity owner. Applying an analogous object-indexed target
to PICF rows is a new scientific arm and requires same-object/shuffle/missing
modality controls.

### 5.7 Stage P-Q-C5 causal temporal adapter

The released training mapper uses five frames with range two. Because
`2 * range + 1 == frame_count`, its primary branch samples one contiguous
five-frame clip from a complete offline video. The selected donor therefore
learns from complete local clips, but the mapper does not define an online
control contract: a random reference frame may have observations after it.

PICF must not expose those future observations. The first action-facing arm is
therefore named `Stage P-Q-C5` and is a `SCIENTIFIC_ADAPTER` with the exact
domain map

```text
CALVIN sample at t -> raw-episode RGB[t-4:t+1] -> released eval geometry
                   -> complete VidEoMT forward -> latest 200 queries.
```

Its invariants are:

1. all five frames are real, contiguous, strictly ordered source frames from
   one physical episode; the final frame is exactly the LingBot action frame;
2. no future frame, language label, action target, owner mask, simulator state
   or task identity enters the donor forward input;
3. no frame is duplicated, interpolated or silently left-padded in the primary
   training arm; samples without four real predecessors are ineligible for that
   arm and the exclusion count is receipted;
4. action training uses the released deterministic evaluation resize,
   normalization and padding; the complete released stochastic augmentation is
   used only by the separately registered CALVIN donor-adaptation objective;
5. each independent training sample resets VidEoMT before its five-frame clip,
   preventing state leakage across random optimizer samples;
6. deployment starts with one real frame and then uses the released `resume`
   state one frame at a time. Exact multi-frame-versus-resume parity is a gate;
7. the VidEoMT propagated query is short visual tracking state. It is not the
   PICF long posterior and cannot replace long-history intervention tests.

The causal map is necessarily not the upstream YouTube-VIS sampling
distribution. Calling it exact upstream sampling is forbidden. Conversely,
changing donor layers, query propagation, prediction heads, preprocessing
operators or weights is not authorized by this temporal adaptation.

## 6. What "decoder-free" means here

The target has no standalone semantic Transformer decoder or tracker beside the
large backbones. It does not mean that masks or teacher-space predictions can
be produced without readout computation.

Allowed exact readouts are:

- VidEoMT's released three-layer mask MLP, learned upscaler and class linear;
- LingBot's released `TaskTokenDepthHead`, including its full
  `TaskTokenResampler` cross-attention block;
- LingBot's released action output projection.

These readouts do not own physical identity. Removing or shrinking them would
break the released methods and is prohibited.

## 7. Long posterior without full long-horizon backpropagation

### 7.1 Deployment

The complete layerwise posterior state crosses every control boundary and may
persist for tens or hundreds of steps. Current evidence corrects, preserves or
supersedes that state through the shared host. There is no hand-written age,
decay, delete or recycle law in the causal path.

### 7.2 Training

Backpropagating through hundreds of full 6B-model steps is unnecessary and
impractical. The accepted strategy separates state exposure from gradient
horizon:

1. sample real contiguous episode positions rather than independent frames;
2. construct the start posterior by a variable-length causal burn-in under the
   same model and information contract;
3. stop gradient at the burn-in boundary;
4. backpropagate through a short, randomly selected local horizon;
5. vary burn-in length over a broad distribution and include reset, occlusion,
   reappearance and long-gap strata;
6. refresh cached start states whenever model drift crosses a frozen threshold;
7. reject stale states whose producing checkpoint, data hash or source contract
   differs.

This is truncated recurrent training with state replay, not a claim that two
gradient steps equal two steps of memory. The model experiences long-lived
states in its forward distribution while memory remains bounded by the short
gradient horizon. A matched no-history arm and wrong-time/wrong-episode state
interventions are mandatory.

The training cost relative to a one-frame baseline is approximately:

```text
cost ~= C_grad * H_grad + C_nograd * H_burnin.
```

Cached, hash-bound burn-in states reduce the recurring term to approximately
`C_grad * H_grad`, plus cache refresh. Exact seconds/step must be measured on
the final two- or four-A100 topology; estimates are not acceptance evidence.

## 8. Adaptation boundary

### 8.1 Mechanical and allowed

- immutable checkout preparation and source-hash verification;
- import-path rewrites with byte-diff receipts;
- typed dataclasses around unchanged tensors;
- DDP/FSDP placement that preserves numerical semantics;
- full-column-rank width projection with rank/singular-value probes;
- caching detached upstream targets with source/checkpoint/data hashes;
- absence masks and layout metadata that do not alter observed values.

### 8.2 Scientific and separately named

- inserting all 200 VidEoMT query observations into LingBot;
- allowing PICF rows to attend to those observations;
- mapping short-memory query observations to persistent rows;
- object-indexed multimodal predictive losses;
- posterior recurrence across control steps;
- action visibility of posterior rows;
- feeding posterior rows back into VidEoMT;
- replacing category supervision with class-agnostic physical existence;
- changing query count, resolution or temporal sampling.

The first target is one-way Stage P: exact VidEoMT observations enter LingBot,
but LingBot rows do not alter the exact VidEoMT graph. Bidirectional feedback is
a later arm because it changes the donor's query distribution.

## 9. Explicitly forbidden implementations

1. A linear/cosine head that independently chooses the task object.
2. A hand-written confidence, age, decay, birth, death or recycle controller.
3. Top-k or threshold pruning before the shared host.
4. Compressing 200 queries to `K` outside the shared host.
5. Replacing VidEoMT with SAM proposals in the forward graph.
6. Reducing donor layers, widths, queries, losses, matcher or schedule for a
   smoke test and then treating the result as donor evidence.
7. Mixing frozen and public-head LingBot recipe fields.
8. Allowing future teacher targets into forward inputs.
9. Treating CALVIN owner masks, identities or task labels as deployment input.
10. Claiming a tiny/toy probe validates full-weight behavior.

## 10. Ordered implementation plan

### Phase A: exact source and checkpoint closure

- [x] Pin and hash the exact VidEoMT and DINOv3 sources and weights.
- [x] Pin the checkpoint-compatible LingBot source and released checkpoint.
- [x] Prove current public LingBot head does not change the selected model
  files.
- [x] Publish `references/full_transplant_sources.json`, covering all ten
  audited LingBot source files, both ordered overlays, all seven resulting
  patched files, all released checkpoint/processor assets, all normative
  VidEoMT files and the exact DINOv3 conversion contract.
- [x] Re-run the LingBot overlays from the immutable source object and prove
  all seven resulting files match the frozen hashes. The auditor also proves
  the four selected model-core files remain identical at public head.
- [x] Run the same auditor in `--strict-runtime` mode against the full cloud
  checkpoint, processor, VidEoMT checkpoint, DINOv3 bundle and prepared
  LingBot checkout. A repository-only pass is deliberately not a runtime pass.

### Phase B: exact spatial observations

- [x] Preserve all 200 query vectors, masks, class logits and propagated state.
- [x] Train the complete donor backbone in an explicitly incomplete CALVIN
  spatial ablation.
- [x] Evaluate all held-out windows, model-ranked top-k and multiple scales.
- [x] Inspect source-disjoint human-readable panels.
- [ ] Run the selected online matcher and complete released multi-scale/crop
  distribution with released optimizer-state and AMP semantics rather than
  adding a custom ranking head.

### Phase C: one-way shared-host integration

- [x] Implement typed Stage P-Q without task/lifecycle logic.
- [x] Pass all 200 query tokens through an exact-token, non-resampling bridge.
- [x] Prove column-isometric initialization and query storage/value preservation
  in the source dtype.
- [x] Measure the FP32-to-host-dtype precision adapter on a real official
  five-frame query batch.
- [x] Record explicitly that Stage P-Q does not encode the complete donor mask
  field; do not promote it as full-output injection.
- [x] Implement and receipt Stage P-Q-C5 with five real causal source frames,
  exact current-frame identity and reset isolation.
- [x] Prove Stage P-Q-C5 multi-frame versus online one-frame `resume` parity on
  the exact deployed preprocessing and state-reset boundaries.
- [x] Execute the full released LingBot host with real VidEoMT queries and
  verify every query reaches all intended shared layers.
- [x] Prove gradient reaches the shared host and posterior while no new
  selector owns semantics.
- [x] Prove raw LingBot evidence and baseline action path remain present.

### Phase D: full multimodal PICF

- [ ] Bind V-JEPA/DINO-Video, AnyTouch and Sonata source/checkpoint receipts.
- [ ] Use the exact official LingBot current/future query-distillation modules
  and losses; do not rewrite or shrink `TaskTokenDepthHead`.
- [ ] Add object-indexed binding only as a named scientific arm.
- [ ] Test same-object versus modality-shuffle, time-shuffle, row-shuffle,
  missing-modality and corrupt-modality controls.
- [ ] Verify absent modalities create neither tokens nor losses.

### Phase E: persistence and action

- [ ] Train variable-burn-in, short-gradient-horizon state replay with exact
  reset and source contracts.
- [ ] Test occlusion survival, correction, re-identification and long-gap
  calibration.
- [ ] Run factual, zero-history, wrong-time, cross-episode, row-swap and
  anchor-disabled interventions.
- [ ] Compare complete same-seed LingBot/PICF action curves at every registered
  milestone, not selected minima.
- [ ] Run closed-loop CALVIN before authorizing 30k.

## 11. Acceptance matrix

| Gate | Required evidence | Current state |
| --- | --- | --- |
| U0 source parity | exact source/hash/patch/checkpoint receipt | `PASS`: repository/source/ordered-patch replay and strict persistent cloud assets verified |
| U1 spatial quality | held-out masks, ranking, scale, human review | local gate pass with ranking/scale caveats |
| U2 host reachability | all 200 tokens, shared-layer and gradient probes | complete real C5 donor and typed 200-query boundary pass; full LingBot-weight probe pending |
| U3 multimodal binding | same-object advantage and shuffle controls | pending |
| U4 persistence | long-burn-in and occlusion interventions | pending |
| U5 action | complete matched LingBot/PICF curves | pending |
| U6 deployment | resume, memory, throughput, CALVIN rollout | pending |
| U7 scalability | second dataset/embodiment under same interface | pending |

No 30k run is authorized by U0-U2 alone.

## 12. Adversarial architecture questions

Before every implementation or experiment, answer all of the following:

1. Is this code copied from a pinned mature implementation? If not, why is a
   new hypothesis necessary?
2. Did any tensor lose dimensions, queries, layers, temporal context or loss
   terms for convenience?
3. Does a small module now decide identity, relevance, lifecycle or action?
4. Can the same effect be expressed inside the shared LingBot host?
5. Does the candidate preserve the baseline information set?
6. Can future labels or simulator identities affect deployment-visible
   forward values?
7. Is uncertainty represented softly, or forced into a false object decision?
8. Are object rows permutation-equivariant?
9. Is the change isolated and paired against a matched control?
10. What concrete result would falsify the change and trigger removal?

If any answer is missing, implementation stops before GPU training.

## 13. Honest maturity score

As of this document:

- architecture rationale: `8.8/10`;
- exact spatial donor implementation: `8.5/10` locally; source, tensor, objective,
  augmentation and optimizer contracts pass, while a selected-online
  full-recipe rerun and second-dataset evidence remain pending;
- unified full-weight implementation: `7.2/10`; the real C5 donor boundary is
  implemented and measured, while full LingBot plus donor execution remains a
  cloud gate;
- scientific evidence for PICF advantage: `3/10`;
- long-training authorization: `0/10`.

These scores cannot be raised to 10/10 by adding documentation or unit tests.
Only the registered full-weight, causal, matched and closed-loop evidence can
close the remaining gap.

## 14. Combined provenance gate

The no-simplification gate is executable:

```bash
python tools/audit_full_transplant_contract.py \
  --lingbot-repository /path/to/official/lingbot-vla-v2
```

The repository pass verifies exact source snapshots, byte-identical vendoring,
the sole approved criterion import rewrite, both LingBot patches in order, and
the final patched-file hashes. Before a full-weight run, cloud preflight must
add all runtime assets and `--strict-runtime`:

```bash
python tools/audit_full_transplant_contract.py \
  --lingbot-repository /path/to/official/lingbot-vla-v2 \
  --prepared-lingbot-checkout /path/to/patched/lingbot \
  --lingbot-checkpoint-dir /path/to/lingbot-vla-v2-6b \
  --processor-dir /path/to/qwen3-vl-processor \
  --videomt-checkpoint /path/to/yt_2019_dinov3_68.9.pth \
  --dinov3-bundle /path/to/derived/dinov3-bundle \
  --strict-runtime --json-out /persistent/path/full_transplant_audit.json
```

Strict mode hashes all declared LingBot and processor assets, deeply inspects
the released VidEoMT tensor inventory, and compares every DINOv3 bundle tensor
bit-for-bit with the deterministic conversion from that checkpoint. Any
missing asset, version drift, patch drift or extra source edit is fatal.

## 15. Stage-PQ implementation closure and no-simplification audit

### 15.1 Executable boundary classification

| Surface | Classification | Executed behavior | Forbidden alternative absent |
| --- | --- | --- | --- |
| VidEoMT backbone/segmenter | `EXACT_UPSTREAM` | released 200-query DINOv3-L graph and blocks 20-23 | no reduced layers, width or query count |
| Class/mask heads and temporal propagation | `EXACT_UPSTREAM` | all released final outputs execute in eval mode | no replacement decoder or tracker |
| Final-query capture | `MECHANICAL_ADAPTER` | a read-only pre-hook observes the existing final class-head input | no value mutation or extra forward branch |
| DINOv3 constructor bundle | `MECHANICAL_ADAPTER` | deterministic key reversal from the released checkpoint; every tensor is compared bit-for-bit | no substitute pretrained backbone |
| CALVIN causal clip | `SCIENTIFIC_ADAPTER: Stage-P-Q-C5` | exact real source frames `t-4...t`, current frame at `t`, no future or padding | no duplicated first frame or label input |
| FP32 to BF16 cast | `MECHANICAL_ADAPTER` with measured finite-precision loss | all 200 rows and 1024 coordinates retained | no pooling, resampling or second normalization |
| Query injection | `SCIENTIFIC_ADAPTER: Stage-P-Q` | latest all-200 query bank enters the shared LingBot host | no top-k, task winner or local lifecycle head |
| Short episode prefix | typed absence | zero tokens and zero valid entries | no fabricated zero-valued observation |
| Donor optimization | frozen released-eval arm | donor is outside the LingBot optimizer and checkpoint state | no accidental partial fine-tuning |
| Production launch provenance | `MECHANICAL_GATE` | ADR-199 strict-audits source replay, prepared source, all host assets and donor tensors before launch | no path-existence-only launch |

The released evaluation graph has zero auxiliary outputs because upstream
guards auxiliary readouts with `self.training`. The code has not deleted those
readouts: the same vendored graph emits them under the released training mode.
Calling evaluation with training-only auxiliary computation forced on would be
an upstream semantic change, not a more complete transplant.

Stage P-Q deliberately does not claim that query vectors losslessly encode
mask logits or the final donor patch field. Both heads execute and remain
available for receipts and visualization, but only the query bank is injected
in this named first scientific arm. Injecting masks or patch fields is Stage
P-QM and requires a separate matched experiment; doing it silently would
violate the frozen information graph.

### 15.2 Real official-CALVIN C5 receipt

`tools/probe_videomt_calvin_stage_pq.py` reuses the production
`CalvinDatasetIndex`, `prepare_calvin_stage_pq_c5`, `ExactVidEoMTRuntime` and
`VidEoMTStageP` code paths. It has no toy dimensions or alternate model. The
executed receipt is
three per-clip receipts under `evidence/videomt_exact_pq_runtime/`, summarized
by `streaming_parity_summary.json`; the source members were range-extracted
from the official archive and independently match the frozen dataset manifest,
with archive evidence recorded in
`evidence/videomt_exact_pq_runtime/official_c5_range_extraction_receipt.json`.

Measured on the local RTX 3070 Ti with source indices `359073...359077`:

| Fact | Measured value |
| --- | ---: |
| released checkpoint | 437 tensors, 315,987,030 values, exact SHA-256 `2cfa7a2d...17c04f` |
| executable donor | 435 tensors, 315,986,985 parameters, FP32, frozen, eval |
| load time | 2.384-5.049 s across three local processes |
| five-frame forward | 1.446-1.999 s after warm local loads |
| peak allocated/reserved | 2,028,878,336 / 2,422,210,560 bytes |
| class output | `[1,5,200,41]` finite FP32 |
| mask output | `[1,200,5,120,120]` finite FP32 |
| query output | `[1,5,200,1024]` finite FP32 |
| propagated state | `[1,200,1024]` finite FP32 |
| BF16 host stream | `[1,200,1024]`, 200 valid rows, IDs 0-199 |
| BF16 relative L2 error | 0.0016568854 |
| BF16 minimum row cosine | 0.9999974370 |
| BF16 maximum absolute error | 0.0310964584 |
| BF16-induced row collisions | 0 |

Across three real clips, one five-frame call and one reset plus four resumed
one-frame calls pass the frozen functional FP32 parity gate. The worst query
cosine is `0.9999997616`; worst mask probability delta is `0.0006260872`;
worst scale-normalized mask-logit error is `5.5638e-5`. Bitwise equality is not
claimed because batched and sliced CUDA matrix multiplications use different
FP32 reduction orders.

This is a real donor and adapter pass, not a full-host pass. The target A100
receipt must still measure the replicated frozen-donor cost beside sharded
LingBot, shared-layer reachability, host/posterior gradients, distributed
resume and end-to-end seconds per optimizer step.

### 15.3 Human visual audit

Visual inspection is not replaced by aggregate IoU:

- zero-shot oracle inspection over all 200 queries finds object-like support
  for the blue block (`0.726` IoU), pink block (`0.811`) and table button
  (`0.647`), proving that the released donor contains non-grid object proposals;
- the released YouTube-VIS class ranking is badly out of domain on CALVIN: its
  top unique queries are dominated by robot/background masks and do not expose
  the useful block proposals reliably;
- after the 1,500-step incomplete-recipe spatial ablation, held-out table
  button, drawer, light and switch masks are visibly object-shaped, while the
  occluded movable blocks remain weak in the inspected frame;
- model-ranked Top-10 at 360 pixels contains duplicate/noisy table and robot
  support. Top-25 recovers much of the oracle score, so ranking/duplicate
  allocation is unresolved rather than hidden by oracle matching.

Therefore the visual verdict is `SPATIAL_CAPACITY_PRESENT_BUT_RANKING_AND_SMALL_OCCLUDED_OBJECTS_UNRESOLVED`.
It justifies testing the exact large donor inside the shared host; it does not
authorize a 30k action run or support a claim that anchor quality is complete.

### 15.4 Local source and regression closure

The executable source/asset auditor passes in repository mode. It accounts for
all 16 normative VidEoMT files as nine byte-identical files, one import-only
rewrite and six reference-only files; replays all seven LingBot patched files
from ten pinned upstream files; verifies four public-head model-core files; and
checks the released VidEoMT inventory plus the deterministic 415-tensor,
1,212,559,808-byte DINOv3 bundle. `external_checks_complete=false` remains
intentional because the complete LingBot checkpoint and processor directories
are not present locally; only strict cloud mode may close that field.

Post-integration verification is:

- exact VidEoMT plus real-runner focused suite: `163 passed`;
- complete repository suite: `4116 passed, 53 skipped` in `213.49 s`;
- both launchers pass `bash -n`, the touched Python surface passes Ruff, and
  `git diff --check` passes.

The 53 skips are declared optional dependency, source-checkout, dataset or
released-asset gates. A green local suite proves interface and regression
closure only; it does not replace any cloud, causal, action or rollout gate.

The official `adr199/run_full_transplant_stage_pq_2gpu.sh` launcher now enforces
strict mode before constructing the training process. Its durable sibling
receipt verifies the same checkout as both immutable Git-object source and
prepared patched tree, every declared LingBot checkpoint/processor file, the
released VidEoMT checkpoint, and every converted DINOv3 tensor. Directly
invoking the underlying generic runner is not an approved ADR-199 launch.

### 15.5 Remaining promotion blockers

1. Run the complete released LingBot host plus exact C5 donor on two A100s and
   prove all 200 rows reach every intended shared layer without hidden
   resampling.
2. Prove host projection and persistent-row gradients while the frozen donor
   remains gradient-free and outside optimizer/DCP state.
3. Run matched same-seed LingBot/PICF curves and the multimodal shuffle,
   persistence and action interventions. Only those results can establish the
   PICF scientific claim.

### 15.6 First strict cloud launch ledger

The strict two-A100 launch on 2026-08-21 uses persistent assets exclusively.
Its audit reports `status=passed`, `strict_runtime=true` and
`external_checks_complete=true`. It verifies 14 LingBot checkpoint assets
(`28,233,257,088` bytes), eight processor assets, 437 VidEoMT tensors
(`315,987,030` values), and a 415-tensor DINOv3 runtime bundle
(`1,212,559,808` bytes). The launch uses all 200 donor queries and the frozen
FP32 released-evaluation donor; the bounded scientific budget is 250 updates,
not 30k.

The failed launch attempts preceding that run did not execute a model update.
They exposed four deployment-contract defects: conflating the immutable and
prepared LingBot trees, an incomplete partial-clone object store, a stale Muon
overlay identity, and a missing exact Detectron2/Timm runtime overlay. Each was
repaired by immutable object transfer, ordered patch replay, hotfix-aware source
validation, or a persistent hash-bound dependency overlay. They are deployment
closure, not positive scientific evidence.

The first admitted full-weight run is
`full-transplant-stage-pq-c5-250-20260821T195429+0800`. Its step-zero action
snapshot completed on all 34 validation and 68 held-out examples, with means
`0.4746668199` and `0.4861701517`. Both ranks then completed update one: rank-0
official action loss was `0.64453125`, rank-1 was `0.58984375`, all registered
gradient metrics were finite, and peak reserved memory was about `25.43 GiB`.
Both execution receipts prove exact five-frame C5 input, all 200 ordered
queries, complete class/mask/query/propagated outputs, FP32 frozen donor
parameters, no donor optimizer membership and no host-side query reduction.

Update two reached a finite objective but failed during FSDP backward. After the
first update lazily materialized optimizer state, each rank held about
`38.33 GiB`; the next 48-MiB all-gather had only 25--29 MiB free. This result
does not falsify Stage P-Q. It falsifies the original `cuda-resident` placement
of a replicated 315,987,030-parameter frozen FP32 donor while the trainable host
owns optimizer state.

### 15.7 Frozen-donor idle placement repair

ADR-199 therefore uses the named mechanical policy `cpu-between-forwards`.
Let the frozen deterministic donor be `q = C(S(F_theta(x)))`, where `F_theta`
is the complete released FP32 graph, `S` selects all latest-frame query rows and
`C` is the measured BF16 host cast. The host loss is `L(H_phi(q, z))` and
`theta` is outside autograd and the optimizer. Once `q` is materialized, the
host computation has no read edge to `theta`; relocating the storage of
`theta` from CUDA to CPU therefore cannot change `q`, `L`, or any gradient with
respect to `phi`. The donor is returned unchanged to CUDA before its next
forward. Its internal propagated-query cache is cleared before idling because
Stage-PQ-C5 explicitly executes every independent five-frame clip with
`resume=False`; no deployment-visible state is discarded.

This repair changes neither source, weights, precision, layers, query count,
outputs, host interface nor objective. It is a `MECHANICAL_ADAPTER`, not a new
learned module. The launcher records load/offload counts, placement time and
current device. Promotion requires a two-rank run to complete at least three
updates, thereby crossing the optimizer-state boundary that failed previously;
only then may the bounded 250-update gate restart.

### 15.8 Two-rank steady-state closure: selective frozen-vision FSDP offload

The direct three-update `cpu-between-forwards` gate subsequently proved that
the donor relocation itself works. Update one completed with finite objectives
and gradients on both ranks. It took about 50.5 seconds, started backward near
18.7 GiB allocated, and peaked near 31.9 GiB allocated / 38.5--38.7 GiB
reserved. After optimizer-state materialization, update two again reached a
finite objective but entered backward near 28.9 GiB allocated and failed when a
48-MiB FSDP all-gather encountered only 17--48 MiB of physical free memory.
Thus donor idling recovered roughly 1.2 GiB but did not provide a defensible
steady-state safety margin for the full 200-row host activation on two A100
40GB GPUs.

The next and final semantics-preserving two-rank repair is
`selective-embedding-frozen-vision-offload`. It extends the already approved
PyTorch `CPUOffloadPolicy(pin_memory=False)` placement from the shared frozen
embedding to the already frozen Qwen visual Transformer blocks. This is not a
new model arm:

- the complete VidEoMT donor, all 200 query rows, LingBot layers, losses,
  optimizer groups, trainable parameter set and update law remain identical;
- only FSDP parameter-shard storage for declared frozen visual block units is
  changed between CPU and CUDA;
- the launcher rejects any trainable parameter beneath an offloaded visual
  block, rejects any undeclared CPU parameter, and requires nonempty CUDA
  storage to prevent accidental full-model offload;
- the exact third overlay is replayed after the native and distributed-Muon
  overlays and is bound by source and patch SHA-256 values in
  `references/full_transplant_sources.json`.

For fixed parameters `theta_v`, FSDP materializes the same parameter tensor on
the compute device before evaluating each visual block. Consequently
`f_{theta_v}(x)`, the downstream loss, and every gradient with respect to the
trainable set are invariant to the idle storage location of `theta_v`, up to
the same device-copy identity already used by PyTorch FSDP. No learned routing,
query compression, mask substitution or extra objective is introduced.

Promotion remains empirical and fail closed. The new placement must pass a
strict three-update gate across the lazy optimizer-state boundary. If it still
exceeds two-card memory, ADR-199 requires four-way FSDP2; reducing the 200
queries, donor layers or output surfaces is forbidden. A successful three-step
gate authorizes the bounded 250-update scientific gate, not a 30k run.

### 15.9 Complete visual-root placement and bounded-gate evidence

Offloading only the 24 frozen Qwen visual Transformer blocks moved 151,154,688
rank-local parameter elements from CUDA to CPU. It completed updates one and
two, but rank 1 failed at update three during the same 48-MiB FSDP all-gather.
The failure therefore narrowed the remaining exact two-card option to the
112,660,224 frozen parameters outside those blocks. It did not measure anchor
or action quality.

The final two-card placement wraps the complete already-frozen Qwen visual root
as the parent FSDP unit after wrapping its 24 child blocks. It introduces a
fourth ordered source overlay and changes no trainable parameter, tensor value,
forward operator, loss, query row, data stream or optimizer rule. The runtime
receipt accounts for all 315 visual tensors plus the shared embedding on CPU:
316 CPU tensors / 402,151,936 rank-local elements, with the remaining 1,322
trainable tensors / 2,752,317,342 rank-local elements on CUDA.

The strict runtime audit passed with four ordered overlays and every external
asset hash verified. The focused contract suite passed `187` tests with one
declared optional CALVIN-source skip. A wider run reached `1244` passes before
being stopped for efficiency; its three failures were cloud-filesystem or
source-snapshot environment tests (unsupported `/mnt` symlinks and the
intentional absence of `.git`), not model regressions.

Run
`full-transplant-stage-pq-frozen-visual-root-gate250-20260821T220051+0800`
crossed the prior update-three OOM and remained stable through the first 20
updates, including a source-masked sequential factual-gradient step. Ordinary
backward starts were stable near 28.10/28.14 GiB allocated; graph release was
stable near 20.65 GiB. The first three objectives and action losses were
numerically identical to the preceding placements, establishing that the
repair changed execution placement rather than the mathematical experiment.

The frozen-input step-20 action evaluation is positive but not final:

| partition | step 0 | step 20 | matched LingBot step 20 | relative to LingBot |
|---|---:|---:|---:|---:|
| heldout | 0.486170 | 0.385125 | 0.441363 | 12.74% lower |
| validation | 0.474667 | 0.371668 | 0.421473 | 11.82% lower |

These values are within 0.56%/1.31% of the historical PICF step-20 heldout and
validation values. The complete transplant therefore reproduces the old
PICF's early action-convergence advantage; it does not yet establish a durable
endpoint advantage. Promotion remains blocked on the step-50/100/250 anchor
evidence, step-100/200 action curve, matched AUC, and visual inspection.

### 15.10 Step-100 scientific verdict: Stage P-Q is not the root repair

The uninterrupted run reached the registered step-100 anchor and fixed-action
gates without an engineering, numerical or provenance failure. It was stopped
at the start of update 104 after those artifacts were complete. Continuing to
250 would have repeated the historical arm after the registered discriminator
had already failed; the stop is scientific early termination, not a runtime
failure.

The anchor readout did learn a non-trivial object organization:

| measurement | step 50 | step 100 | historical ADR-193 step 100 |
|---|---:|---:|---:|
| matched soft-IoU mean | 0.021315 | 0.073253 | 0.085212 |
| matched soft-IoU median | 0.004999 | 0.017392 | 0.022292 |
| matched existence mean | 0.449219 | 0.683479 | 0.9006 |
| unmatched existence mean | 0.408343 | 0.274414 | 0.4245 |
| existence margin | 0.040876 | 0.409065 | 0.4761 |

Mean matched IoU improved by 3.44x from step 50 to 100, and visual inspection
shows distinct rows for the blue block, drawer/slide and several large table
parts. This falsifies the claim that Stage P-Q is merely random, collapsed or
an unlearned patch grid. It does **not** pass the spatial gate: mean IoU remains
14.0% below ADR-193, median IoU 22.0% below it, small switch support remains
poor, and boundaries remain broad.

The fixed-bank action result reproduces the historical crossover rather than
removing it:

| partition | current step 100 | matched LingBot step 100 | historical PICF step 100 | current vs LingBot |
|---|---:|---:|---:|---:|
| heldout | 0.345674 | 0.325641 | 0.342127 | 6.15% higher |
| validation | 0.333209 | 0.328958 | 0.338379 | 1.29% higher |

The same run was 12.74%/11.82% better than LingBot at step 20. Therefore the
early result was genuine, but it measured optimization acceleration rather
than a durable action advantage. The complete donor and larger host do not by
themselves resolve the old crossover.

The architecture-level diagnosis is narrower than "PICF failed" and stronger
than a learning-rate hypothesis. The released VidEoMT mask is a joint readout
of an object query and the spatial patch field:

```text
M_i = MaskMLP(q_i)^T Upscale(X)
```

Stage P-Q transmits all `q_i` values, but deliberately discards `X` and does
not transmit `M_i` as a host-visible support relation. Consequently, it
preserves candidate identity values but not the sufficient spatial statistic
used by the mature donor to bind those values to pixels. LingBot still sees raw
RGB, so recovery is possible in principle, but it must relearn cross-model
query-to-pixel alignment from the small CALVIN objective. The observed broad
supports and weak small-object masks are the expected failure mode of that
information bottleneck. More query count, a task winner, a lifecycle heuristic
or a larger projection does not restore the discarded spatial field.

Stage P-Q is therefore a **NO-GO for 30k**. This verdict does not reject the
PICF posterior hypothesis. It rejects query-only transplantation as its object
observation interface. The next admissible arm is Stage P-QM: preserve the
released donor's complete query--patch mask relation and expose that relation
inside the shared LingBot host, with no SAM labels, top-k selector, private
lifecycle owner or reduced decoder. Because the official zero-shot donor is
out of domain on CALVIN, any donor adaptation must use the complete released
training recipe and a strict episode-disjoint split before frozen integration.
The existing segment-disjoint 1500-step adaptation proves spatial capacity
only; it is not admissible final evidence.

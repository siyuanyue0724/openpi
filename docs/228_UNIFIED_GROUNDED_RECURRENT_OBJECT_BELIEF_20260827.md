# ADR-228: Unified Grounded Recurrent Object Belief

Date: 2026-08-27 Asia/Shanghai

Status: **TARGET ARCHITECTURE FROZEN FOR SOURCE-DIFF AND LOCAL CONTRACTS; NOT IMPLEMENTED OR SCIENTIFICALLY VALIDATED**

Supersedes ADR-226 as the complete next-system design. ADR-226 remains the
source receipt and derivation for its SWIM grounding subproblem. ADR-227's
source/host geometry repair remains mandatory.

## 0. Direct decision

PICF has not yet demonstrated an action advantage over exact LingBot. It has
demonstrated a useful but incomplete object-proposal substrate:

- the complete 200-query VidEoMT bank reaches about `0.79/0.82` soft/binary
  oracle IoU on the fixed bank, with some small-object proposals near `0.93`;
- the task-agnostic Top-10 selection remains about `0.10--0.11` soft IoU;
- the geometry-corrected step-20 action curve is `2.96%` worse than ADR-207 on
  heldout and only `0.31%` better on validation;
- the active profile has no large-host language-to-object objective and no
  authenticated optional-modality-to-object identity objective.

Therefore the next step is neither a 30k continuation nor another selector
head. It is one bounded architecture replacement with three coupled repairs:

1. actual instruction tokens learn object grounding inside the shared
   LingBot/Qwen3-VL layers;
2. a compact final-row posterior is carried through complete episodes using
   source-faithful recurrent training rather than a sampled `2+2` proxy or a
   persistent `36 x 200` layerwise state;
3. all available modalities meet at the same posterior rows through the same
   host attention operation, while uncertain or uncovered evidence retains a
   real context path instead of being forced into an arbitrary row.

This is one coherent latent-state model, not three independent prediction
heads. The LingBot host owns semantic selection, uncertainty, temporal
correction and action. VidEoMT remains a complete mature proposal operator; it
does not own task relevance or action. Small linear projections may translate
released feature widths, but no small network may decide object identity,
lifecycle, task relevance or action routing.

## 1. Non-negotiable design laws

1. **One semantic owner.** The complete 36-layer LingBot/Qwen3-VL host is the
   only component allowed to infer task relevance, cross-modal identity,
   posterior uncertainty and action-relevant state.
2. **No source simplification.** Retain the complete released LingBot,
   VidEoMT, V-JEPA 2.1, AnyTouch and Sonata paths. Query reduction, layer
   removal, source-loss removal, replacement decoders and reduced source
   schedules require a new ADR and owner approval.
3. **No hidden selector.** No MLP winner, lifecycle gate, confidence router,
   top-k training path, SAM mask, Hungarian task-row winner or threshold-driven
   object controller is part of the target architecture.
4. **Context is first-class.** Native RGB/language/context tokens are retained.
   Low-quality proposals and unassigned modality evidence remain context; the
   objective must not fabricate object certainty.
5. **Objects are scene instances, not dataset classes.** The 200 rows are a
   permutation-compatible per-scene capacity. Dataset expansion does not
   reserve early rows for frequent categories or collapse rare categories into
   a permanent context class.
6. **Uncertainty is soft.** Overlap, partial visibility and ambiguous contact
   produce distributions over rows plus context. A hard winner is forbidden
   unless an upstream source contract itself supplies one.
7. **Labels are loss-side only.** CALVIN masks, task identity, future frames and
   correspondence labels cannot enter deployment observations or state.
8. **Missing evidence is absent.** An unavailable modality contributes no
   token and no corresponding loss. It is never replaced by a learned fake
   token that could identify the dataset.
9. **State is causal.** Prior rows cannot read current RGB, current optional
   modalities, current language-grounding targets, posterior rows, future
   targets or action suffixes.
10. **No scientific credit by association.** Exact upstream operations inherit
    upstream evidence. Every changed interface is a PICF hypothesis and must
    pass its own ablation and causal intervention.
11. **Whole-curve superiority is required.** A transient or sub-percent gain,
    a prettier mask, or a lower auxiliary loss cannot authorize a long run.
12. **Fail closed.** If the shared host learns grounding but action ignores the
    rows, reject the composition rather than adding a private action head.

## 2. Root-cause model

Let the current observation be `X_t`, instruction tokens be `L`, optional
modalities be `E_t`, previous executed control be `U_(t-1)`, action be `A_t`,
source masks be `M_t=(M_t1,...,M_tQ)` and recurrent object belief be `B_t`.

ADR-227 approximately implements:

```text
M_t = VidEoMT(X_t, source_state_(t-1))
O_ti = Pool(QwenVision(X_t), M_ti)
A_t  = LingBot(L, X_t, E_t, O_t, old_state).
```

The first two lines can create useful object proposals. They do not imply the
three information relations required for PICF:

```text
I(L_entity ; target row) > 0,
I(E_t(object) ; same row) > 0,
I(A_t ; B_t | L, X_t, E_t) > 0.
```

The observed failure is therefore not evidence that all object-centric
posteriors are useless. It is evidence that an appended proposal bank without
large-host semantic grounding and causal adoption is insufficient. Existing
native context gives action an easier path, while the old exclusive-row route
damaged the released conditioning interface. ADR-228 keeps native context but
trains the relation in the same large host and demands intervention evidence
that the action actually adopts it.

## 3. Probabilistic object-belief abstraction

The intended state is a finite approximation to a filtering posterior:

```text
p(B_t | X_<=t, E_<=t, U_<t)
  proportional to
p(X_t, E_t | B_t) * integral p(B_t | B_(t-1), U_(t-1))
                                  p(B_(t-1) | history) dB_(t-1).
```

PICF does not claim exact Bayesian inference. It realizes the two terms as
typed token paths in one pretrained transformer:

```text
P_t = host_prior(B_(t-1), U_(t-1))
B_t = host_correct(P_t, X_t, E_t, M_t)
A_t = host_action(L, X_t, E_t, B_t, U_(t-1)).
```

`P_t` and `B_t` are not outputs of separate recurrent networks. They are token
sets transformed by the same LingBot layers and parameters used for language
and action. The distinction is enforced only by causal attention permissions.

This gives the residual interpretation required by the original PICF design:
`P_t` represents predicted persistent state, current evidence corrects it into
`B_t`, and disagreement can alter action immediately. Whether learned states
actually exhibit this interpretation is an empirical intervention target, not
an architectural assertion.

## 4. Complete target architecture

### 4.1 Preserved released components

The following remain complete and independently receipted:

- LingBot's released Qwen3-VL visual/language host, DeepStack multi-level
  visual injection, action expert, tokenizer and native action objective;
- the complete 200-query VidEoMT decoder, all source blocks, auxiliary outputs,
  source criterion and released online query propagation;
- V-JEPA 2.1's released frozen encoder features and multi-level feature
  convention; its predictor is neither claimed nor simplified in ADR-228;
- AnyTouch and Sonata released encoders and their existing typed projections;
- ADR-227's deterministic host-aligned online source view alongside the
  released augmented source-training view.

This architecture is not literally a single monolithic checkpoint because a
complete mature proposal model and frozen modality encoders remain present.
It is a single **semantic and decision model**: none of those encoders decides
which object matters, when an object dies, or what action to take.

### 4.2 Object observations without a task winner

For host-aligned visual tokens `V_t in R^(P x d)` and source soft masks
`M_t in [0,1]^(Q x P)`, create all object observations with the existing native
merger projection:

```text
w_tip = M_tip / (sum_p M_tip + epsilon)
O_ti  = merger(sum_p w_tip V_tp).
```

All `Q=200` rows remain. There is no task-conditioned top-k in training or
deployment. The source's propagated query index supplies temporal proposal
address continuity. Row-permutation tests must prove that content follows the
permutation; a fixed category meaning for row index is forbidden.

Native visual tokens remain in the sequence. The row bank is a structured
object basis, not a lossy replacement for the scene.

### 4.3 Compact episodic recurrent ABI

Remove the unproven persistent `36 x 200 x d` layerwise host state and the
separate full-host prior-only pass. Carry only the final corrected row bank:

```text
state_t = B_t^(36) in R^(Q x d).
```

At the next frame, initialize two typed views:

```text
P_t^0 = address + B_(t-1)^36                 # predicted/prior carrier
B_t^0 = address + O_t                        # current object observation
```

At LingBot layer `ell`, the legal graph is:

```text
P_t^(ell+1) <- P_t^ell, past control only
B_t^(ell+1) <- B_t^ell, P_t^ell, current sensor/modal tokens
L^(ell+1)   <- L^ell, native visual tokens, B_t^ell
A^(ell+1)   <- A^ell, L^ell, native context, B_t^ell, global control
```

The standard simultaneous transformer update gives correction a one-layer
lagged prior at every depth. Posterior rows do not read language: the physical
object belief remains task-independent. Language reads posterior rows and
therefore selects task-relevant objects inside the shared host. Neither prior
nor posterior may read action tokens.

The actual instruction tokens are recomputed in this joint sequence at every
host layer. Action may see only their final row-conditioned states; an
unchanged pre-row language cache is not retained as a parallel shortcut. This
is the integration boundary that distinguishes ADR-228 from the rejected
optional sidecar family. Native visual context is still retained because
removing it caused a different, measured pretrained-interface failure. The
remaining possibility that action ignores the row contribution is real and is
resolved only by the causal adoption gate, not by assertion.

This is one LingBot execution per frame. It removes the second full prior pass
and does not preserve a separate activation tensor for every host depth across
frames.

At `d=2560`, BF16 persistent values change from approximately:

```text
36 * 200 * 2560 * 2 bytes = 35.16 MiB per sample
     200 * 2560 * 2 bytes =  0.98 MiB per sample.
```

The value-state reduction is exactly `36x`; total training-memory reduction is
smaller because current-frame activations and optimizer state still dominate.
It must be measured, not inferred from this storage calculation.

### 4.4 Long-memory training without an unbounded graph

Adopt the exact semantic contract demonstrated by the official muVLA source:

- recurrent lanes traverse complete episodes in chronological order;
- `is_first` resets a lane and `is_last` closes it;
- the numerical final-token state is carried at every frame, including across
  truncated-backpropagation boundaries;
- gradients are detached every `K` frames, but values are never reset at a
  boundary;
- action tokens cannot write into recurrent memory.

The first registered CALVIN setting is `K=2`, the reported source setting for
MIKASA-Robo. The source's LIBERO result uses `K=8`, but eight complete LingBot
graphs cannot be assumed to fit after ADR-227 already measured about
`35.252 GiB` reserved per rank. `K=8` is the target credit-span upgrade only if
a released-weight memory/throughput receipt passes without reducing any source
model. `K=1` is a mechanics control, not the promoted recurrent setting.

A frame at age 100 therefore receives a state that has actually evolved through
100 causal observations even at `K=2`; gradients span two transitions rather
than 100. This is categorically different from sampling two frames plus two
burn-in frames. Longer credit assignment is a separately measured resource
choice, not a prerequisite for exposing the model to long-age state values.

One optimizer update consumes one `K`-frame lane chunk and averages losses over
valid frames. Its wall time is expected to scale approximately with `K`, while
per-frame throughput should remain in the same order if host overhead is
amortized. Activation memory also grows with the retained graph and must be
measured. Logs must separately report optimizer updates, frame-batches, frames,
episodes and effective samples. Comparisons use consumed frame-batches and
examples, never the ambiguous word `step` alone. Exact LingBot controls use the
same chronological batches, gradient accumulation and consumed-frame budget.

### 4.5 Large-host language grounding

Copy SWIM's executed-attention supervision, adapted only where architecture
indices and token surfaces differ. Map the source six normalized depths to
LingBot layers:

```text
[2, 7, 12, 17, 22, 27] / 28
                 -> [3, 9, 16, 22, 29, 35] / 36.
```

For authenticated instruction entity tokens `S_L` and native visual keys
`S_V`, replay actual post-MRoPE Q/K attention at those depths:

```text
A_ell = mean_heads softmax(Q_ell[S_L] K_ell^T / sqrt(d_h) + causal_mask)[:, S_V]
F_V   = row_normalize(exp(sum_ell log(A_ell + epsilon)))
L_vis = BCE(minmax(spatialize(F_V)), resize(G)).
```

The log-space expression is numerically stable and mathematically equivalent
to SWIM's product across layers. It is not a learned decoder and introduces no
trainable parameters. `G` is a loss-side current-frame mask.

Entity spans come from an immutable task annotation manifest with character
offsets and tokenizer-offset verification. Substring guesses and manually
chosen token IDs fail closed.

### 4.6 Language-to-object grounding with a context reservoir

The same actual entity-token Q/K relation is read against posterior-row keys.
For each proposal define a soft quality score that penalizes both missing the
object and covering excessive background:

```text
q_i = soft_IoU(M_i, G) in [0,1]
rho = stop_gradient(max_i q_i)
y_i = rho * q_i / (sum_j q_j + epsilon)
y_context = 1 - rho.
```

If all rows poorly explain the object, most target mass remains context. If
several overlapping rows explain it, target mass remains set-valued. The model
is never forced to invent a unique row. `rho` is a loss target, not an inference
threshold or lifecycle control.

At each selected depth, measure entity-token probability mass over the union
of native visual context and all posterior rows. Fuse depths with SWIM's
product rule and optimize cross entropy against `(y_context,y_1,...,y_Q)`.

This context-reservoir equation is an explicit PICF hypothesis. SWIM supplies
the multi-depth attention operation; it does not supply this object-basis
target. The hypothesis must be ablated against visual grounding alone.

The context term is mathematically necessary when proposal evidence is weak.
Without it, the conventional normalization

```text
y_i^forced = q_i / sum_j q_j
```

has total row mass one even when every `q_i` approaches zero. It therefore
trains an arbitrary confident identity from an uninformative proposal set. In
ADR-228, total object-row target mass is `sum_i y_i = rho`; as proposal quality
vanishes, row-identity supervision vanishes and context receives the mass. For
high-quality overlapping proposals, `rho` is high but the row distribution
remains soft. This avoids both false certainty and a hard winner.

Context is evaluated per entity and frame. It is not a permanent class for
rare objects: the same object moves from context mass to row mass whenever
current proposals explain it. This is why dataset growth must not allocate
early rows to frequent categories or permanently demote later categories.

### 4.7 How the host knows two modalities refer to one object

The posterior row is the shared rendezvous address. There is no separate
pairwise identity classifier and no hard object ID injected into raw tokens.
At selected LingBot depths, actual posterior-row queries attend each modality's
typed keys. Authenticated geometry creates only loss targets:

- RGB and V-JEPA dense grids use the same host-aligned soft object masks;
- Sonata points use released calibration and camera projection to construct
  soft point-to-mask support;
- AnyTouch contributes object correspondence only at a real contact event and
  only when calibrated contact position supports it;
- proprioception and robot state remain global/control evidence unless a
  physical contact relation is present.

For modality `m`, let `R_tmi` be the normalized target distribution over its
tokens for object row `i`, and let `H_tmi` be the selected-depth posterior-row
attention distribution from the executed host:

```text
L_bind(m) = sum_i valid_tmi * KL(stopgrad(R_tmi) || H_tmi).
```

No valid correspondence means no term. Ambiguous support remains soft. The
same LingBot parameters that form object belief, read language and produce
action receive these gradients; a projection may change feature width but may
not score identity.

This design is implicit in the important sense: the high-dimensional row
embedding, not a scalar class or hard ID, represents the relation. It is not
fully unsupervised: authenticated geometry supplies training evidence. Pure
implicit co-occurrence is not adopted because it permits modality shortcuts
and gives no falsifiable object-identity contract.

### 4.8 V-JEPA 2.1 evidence and quarantined world objective

V-JEPA 2.1 is not a task-object selector. Its complete released frozen encoder
may supply dense typed current-observation evidence to the shared LingBot host,
using the official multi-level feature convention. That is the only V-JEPA
role authorized in the first ADR-228 gate.

ADR-228 does **not** authorize a linear row predictor and does not claim to
transplant V-JEPA's predictive algorithm. Replacing V-JEPA's full predictor
with a linear transport would simplify a successful source and could not
inherit its evidence. A future object-conditioned world objective requires a
separate ADR that either:

1. transplants a complete released predictor and its optimizer/loss contract;
   or
2. names a different complete, code-available world-model donor whose native
   interface accepts the shared object belief.

Until that ADR passes source parity, `L_world` is absent rather than replaced
by an ad hoc small predictor. Historical predictive losses improved structure
without proving action value, so this quarantine also prevents an auxiliary
curve from masking another action-adoption failure.

### 4.9 Lifecycle and confidence

There is no TTL, hand-coded decay, learned scalar gate or small lifecycle
network. Visibility, persistence and uncertainty remain dimensions of the
large-host row embedding and its context probability. Continuous episode carry
allows an occluded row to persist; current evidence lets the same host correct
or overwrite it.

A detached linear diagnostic probe may measure visibility, age and identity
from `B_t`; it cannot feed back into the model, route action or change state.
Lifecycle is accepted only if causal occlusion/reappearance interventions show
that the host retains useful identity and releases stale belief without a hard
rule.

## 5. Objective and optimization contract

The first complete ADR-228 gate objective is:

```text
L_total = L_action
        + L_VidEoMT_source
        + lambda_vis   L_vis
        + lambda_row   L_row
        + sum_m lambda_m L_bind(m).
```

This sum does not authorize simultaneous blind activation. The architecture is
complete throughout; objective families are enabled by causal gates:

1. **Grounding gate:** exact SWIM visual loss plus PICF row/context loss.
2. **Adoption gate:** action improvement and row interventions with complete
   context retained.
3. **Multimodal gate:** authenticated V-JEPA/Sonata/AnyTouch binding and
   modality-shuffle interventions.
4. **Separate world ADR:** only after a complete predictor donor passes source
   parity; it is not part of the first ADR-228 promotion decision.

Loss weights are fixed from one training-only gradient-norm receipt, then
frozen before the comparison. Test curves cannot tune them. Each objective is
logged unweighted and weighted, with gradient norms into the shared host,
proposal model and action expert.

There is no claim that adding every positive auxiliary loss improves action.
An objective that learns its metric but does not improve action or causal use
is removed from the promoted configuration.

## 6. Data contract and scalability

### 6.1 CALVIN role

CALVIN is suitable for mechanism debugging, matched LingBot comparison and
closed-loop action evaluation. Its limited task language and object diversity
are not sufficient evidence for universal open-vocabulary grounding.

### 6.2 Mixed-data training

Use transaction-level alternation through the existing native co-training
surface:

- action episodes with language/RGB/action and whichever optional modalities
  truly exist;
- grounded image/video transactions with phrase spans and masks;
- video transactions whose current frames can produce frozen V-JEPA evidence;
- touch/3D transactions only when calibrated correspondence is available.

Missing labels skip only their loss. They do not exclude the sample from
action or representation training. Dataset ID is metadata, never a semantic
input.

### 6.3 Capacity under dataset growth

Rows are reused per scene. Adding one million object categories does not
require one million anchors. Required capacity scales with the maximum number
of simultaneously resolvable instances and proposal diversity in a frame.

No early rows are reserved for important categories. Importance is a
language-conditioned attention distribution over the current row set. If a
scene exceeds 200 resolvable instances, the source proposal capacity is a real
limit; increasing `Q` is then a measured source-scale change, not a category
allocation policy.

The current target assumes RGB is available to generate the proposal bank.
It is robust to missing optional modalities, not proven camera-free. Claiming
that touch or point clouds alone can instantiate the complete anchor bank would
require a separate cross-modal proposal source and is outside ADR-228.

## 7. Upstream source/adaptation ledger

### 7.1 Directly copied semantic contracts

| Primitive | Official source | Pinned commit | Preserved contract |
|---|---|---|---|
| Multi-depth executed-attention grounding | [SWIM](https://github.com/HumanMLLM/SWIM) | `1b97d22fbb84097a2117feb1f1f10353dfe252c2` | selected depths, head mean, product fusion, min-max spatial map, BCE |
| Multi-level native visual host | [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | DeepStack and interleaved MRoPE remain intact inside LingBot |
| Dense current video evidence | [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) | `204698b45b3712590f06245fbfba32d3be539812` | complete frozen encoder and official multi-level feature convention; no predictor claim |
| Episodic recurrent training | [muVLA](https://github.com/CognitiveAISystems/muVLA) | `13e1cf9a34d40726c9f4eeafff464d45c25181bc` | complete episode carry, `is_first/is_last`, value-preserving TBPTT detach, no action-to-memory write |

### 7.2 Explicit PICF hypotheses

- mapping SWIM's six normalized depths to LingBot `[3,9,16,22,29,35]`;
- projecting phrase support into a soft 200-row-plus-context target;
- using final posterior rows as muVLA-style recurrent state;
- using typed prior and posterior row roles inside one LingBot pass;
- supervising posterior-to-modality attention with authenticated geometry;
- binding frozen current V-JEPA evidence to rows by authenticated geometry;
- retaining full native context while expecting action to adopt rows.

None of these hypotheses inherits an upstream success claim.

### 7.3 Audited but rejected donors

- MemoryVLA's learned gate, ToMe consolidation and FIFO lifecycle are not
  copied because they assign state control to auxiliary machinery rather than
  the shared semantic host.
- UniPixel/SAM-style mask decoders are not copied because they introduce an
  external segmentation authority and do not test PICF's shared multimodal
  object belief.
- The failed PICF task-query/object-read, hard winner, exclusive posterior
  route, shallow readout and fixed task-object heads are not reintroduced.

## 8. Implementation plan

### 8.0 Audited repository surfaces

| Surface | Decision | Reason |
|---|---|---|
| `lingbot_native/state.py::NativePosteriorState` | reuse and version its checkpoint envelope | it already represents exactly final `[batch,Q,d]` rows without lifecycle side state |
| `lingbot_native/training.py::NativeTrainingLaneCoordinator` | reuse transaction and continuity checks; add a new contiguous `K`-frame execution path | atomic lane publication, frame stamps and reset checks are already stronger than a new loader |
| `lingbot_native/host.py::LingBotNativeGraph` | add a new ADR-228 architecture identity; do not alter ADR-227 identities | the current VidEoMT profiles hard-code layerwise recurrence and must remain reproducible controls |
| `lingbot_native/graph.py` | add and formally validate the ADR-228 role graph | current PRIOR/POSTERIOR causality is useful, but current SENSOR and language/cache behavior needs an explicit new contract |
| `lingbot_native/task_address_receipt.py` | preserve unchanged as historical code; factor its exact Q/K kernel into a new generic grounding receipt | the arithmetic is reusable, while OBJECT_READ token semantics are rejected |
| `lingbot_native/pretrained_object_memory.py` | reuse ADR-227 host-aligned mask pooling | this is the accepted geometry repair |
| `lingbot_native/vl_cotraining.py` | reuse transaction scheduling and trainable-scope validation | grounded language data must update the same Qwen host, not a side model |
| `lingbot_native/vl_cotraining.py` | do not reuse its generated-answer parser as the CALVIN entity-span authority | ADR-228 needs immutable tokenizer-offset spans, not generated text parsing |
| current relation/predictive readouts | disabled in the ADR-228 profile unless explicitly retained as detached diagnostics | private semantic/lifecycle ownership would violate the target graph |
| `tools/run_lingbot_vla2_task_independent_full.py` | leave as the ADR-227 control; create a dedicated ADR-228 runner/config | a conditional maze in the historical runner would make the comparison and rollback ambiguous |

The new profile is additive and isolated. Shared low-level utilities may be
factored only with exact before/after tests; historical launchers, configs and
checkpoints remain loadable.

### Phase A: source receipts and zero-change contracts

- [ ] Vendor or pin immutable source receipts for SWIM, Qwen3-VL, V-JEPA 2.1
      and muVLA with file hashes and line-level adaptation notes.
- [ ] Mark ADR-226 visual grounding as a retained subdesign and replace its raw
      intersection row target with ADR-228 soft-IoU-plus-context semantics.
- [ ] Freeze the LingBot depth mapping `[3,9,16,22,29,35]`.
- [ ] Prove exact LingBot outputs, action loss and gradients when every new loss
      scale is zero and recurrent state is disabled.
- [ ] Prove all labels/future frames are absent from deployment inputs.

### Phase B: recurrent ABI replacement

- [ ] Add a typed final-row state schema without deleting the old state schema.
- [ ] Add chronological episodic lanes with `is_first/is_last`, complete value
      carry and configurable TBPTT `K`.
- [ ] Implement the one-pass PRIOR/POSTERIOR role graph and forbid action keys
      from prior/posterior writes.
- [ ] Prove actual instruction states are recomputed with posterior visibility
      and that action cannot consume a stale, unconditioned language cache.
- [ ] Keep the old ADR-227 branch runnable as an exact control; do not mutate it
      in place.
- [ ] Prove age-100 state numerically descends from all preceding frames while
      autograd history is bounded by `K`.
- [ ] Probe separate-process checkpoint/resume in the middle of an episode.

### Phase C: shared-host grounding

- [ ] Generalize the existing post-MRoPE selected-Q/K receipt to actual
      instruction entity tokens; do not instantiate task-query/object-read.
- [ ] Implement exact SWIM visual fusion and source-equality vector tests.
- [ ] Implement the row-plus-context distribution with overlap, empty-support,
      giant-mask, row-permutation and numerical-stability tests.
- [ ] Verify gradients reach the intended shared LingBot layers and do not
      create an external semantic owner.
- [ ] Produce human-readable original-resolution panels showing phrase,
      ground-truth support, best oracle rows, language-ranked rows and context
      mass; fixed-grid color plots alone are insufficient.

### Phase D: authenticated multimodal binding

- [ ] Reuse the same selected-depth attention receipt for V-JEPA, Sonata and
      AnyTouch keys.
- [ ] Build calibration-derived soft targets and fail closed on missing or
      ambiguous metadata.
- [ ] Verify set permutation, modality omission, wrong-object swap and no-touch
      controls.
- [ ] Demonstrate that every enabled modality changes posterior values and
      action under matched interventions, rather than merely producing a loss.

### Phase E: separate world-model authority

- [ ] Receipt V-JEPA 2.1's exact multi-level layer groups for current-evidence
      ingestion only.
- [ ] Do not implement the proposed linear future transport.
- [ ] Select a complete code-available predictor donor and publish a separate
      source/adaptation ledger before adding any future objective.
- [ ] Require source parity, strict stop-gradient/no-future-leakage, future
      shuffle, wrong-time, occlusion and horizon interventions in that ADR.

## 9. Scientific experiment ladder

### 9.1 Local and released-weight mechanics

These establish only correctness:

- source equality and source-diff review;
- zero-scale equivalence;
- tokenizer-span and no-leakage checks;
- Q/K probability equality against full attention on small tensors;
- row/context normalization and permutation equivariance;
- episode carry, TBPTT boundary and restart equality;
- four-modality value/gradient/correspondence interventions;
- measured memory and frame throughput.

No unit-test count raises scientific maturity.

### 9.2 Bounded cloud gate

Use exact LingBot and frozen ADR-227 controls on the same CALVIN samples, seed,
noise, timestep, gradient accumulation and consumed-frame budget. Launch at
most 250 frame-batches, with evaluation at frame-batch 0, 20, 100 and 250.
With `K=2`, this is at most 125 optimizer updates. Checkpoint only at
scientifically useful boundaries.

Report the entire curve for:

- fixed heldout and validation action loss;
- expected row IoU under language attention;
- probability mass on rows with IoU at least `0.5`;
- top-1 language-selected row IoU;
- context calibration when no proposal reaches `0.5` IoU;
- row-shuffle, wrong-row, history-reset and modality-swap action deltas;
- frames/sec, optimizer updates/sec and peak reserved memory.

Candidate promotion thresholds, frozen before execution:

- expected language-selected soft IoU improves by at least `0.15` absolute;
- mass on IoU-at-least-`0.5` rows improves by at least `0.20` absolute;
- fixed action loss is at least `5%` better on the mean of heldout and
  validation than exact LingBot, with neither partition regressing;
- correct rows outperform row-shuffled and wrong-row interventions in action;
- long-state occlusion beats reset-state without future leakage;
- throughput remains within a predeclared budget measured after the recurrent
  ABI replacement.

If grounding remains near `0.05--0.10`, stop immediately: the transplant is
not reproduced. If grounding passes but action does not, reject the action
composition rather than waiting for 2k or adding a head.

### 9.3 Two-thousand-frame-batch gate

Only a passing bounded gate may run to 2000 consumed frame-batches. Compare
complete curves by consumed frames, not optimizer-update labels. Require:

- sustained action superiority rather than a step-20 crossing;
- source-disjoint grounding and object-size strata;
- complete episode state ages including tens and hundreds of frames;
- separate-process resume equality;
- closed-loop CALVIN improvement with confidence intervals;
- no degradation under missing optional modalities.

### 9.4 Thirty-thousand-frame-batch authority

A `30000` frame-batch configuration may exist for resumability, but the
launcher remains fail closed. At `K=2` this means 15,000 optimizer updates, and
both counters must be printed in every status report. Authorization requires
the 2k conjunction above. A stable process or falling training loss is not
authorization.

## 10. Compute plan

1. Keep cloud GPUs off while source receipts, schema tests and synthetic
   recurrence/grounding contracts are unfinished.
2. Use local CPU/GPU for source diffs, shape/causality tests, data manifests,
   tokenizer spans and deterministic probes.
3. Use two A100 40G GPUs only for frozen/cached mechanics or reduced-memory
   released-weight forward probes. Current four-GPU measurements leave too
   little evidence that the complete jointly trainable stack fits two cards.
4. Use four A100 40G GPUs for the first complete optimizer smoke and bounded
   scientific gate. ADR-227 measured about `35.252 GiB` peak reserved per rank
   and `59.219 s` median per update; ADR-228 must remeasure rather than promise
   a speedup from the state-size calculation.
5. Keep all code, manifests, logs, source receipts, fixed batches and accepted
   checkpoints under persistent `/mnt` before any cloud shutdown. Do not retain
   failed dense checkpoint series.

## 11. Falsification and rollback rules

- Good oracle masks plus poor language ranking falsify grounding, not geometry.
- Good grounding plus unchanged row interventions falsifies action adoption.
- Good short-state action plus reset-equivalent occlusion falsifies useful
  recurrence.
- Good RGB results plus modality-shuffle invariance falsifies multimodal fusion.
- Lower auxiliary losses without action/closed-loop benefit do not count as
  progress.
- A gain smaller than the preregistered material margin is not a reason to tune
  thresholds or continue blindly.
- Failure rolls back to exact LingBot plus the retained ADR-227 geometry/source
  evidence. It does not erase the source proposal result, and it does not
  authorize another ad hoc module.

## 12. Architecture review score

### Theoretical maturity: `8.0/10`

Strengths:

- each major primitive has a mature, official source contract;
- one large host owns semantics, uncertainty and action;
- recurrence now exposes training to real long-lived numerical states without
  an unbounded graph;
- overlap, missing modalities and low proposal confidence have explicit soft
  semantics;
- the design has causal failure gates rather than relying on auxiliary loss.

Unresolved deductions preventing `10/10`:

- SWIM grounding, muVLA recurrence and PICF object rows have not been proven to
  compose inside LingBot;
- the row-plus-context target is original and unvalidated;
- retaining native context may still let action ignore posterior rows;
- RGB remains the proposal authority, so camera-free anchoring is not solved;
- a 200-row capacity has not been stress-tested on denser open datasets.

### Deployment maturity: `0/10`

No ADR-228 code, released-weight smoke, bounded curve or closed-loop result
exists. Documentation cannot raise this score. The next allowed work is source
diff plus local implementation contracts, followed by one bounded cloud gate.

## 13. Final architectural verdict

ADR-228 is the strongest coherent next PICF test supported by the current
evidence. It is not yet a proven next-generation VLA and must not be marketed
as one. Its value is that success or failure will answer the central question:
whether a mature object proposal bank, grounded and recurrent inside the same
large VLA host, yields a material causal action advantage. The experiment is
worth doing once under the gates above. Repeating the old sidecar-row family,
adding a selector, or authorizing 30k before those gates is not.

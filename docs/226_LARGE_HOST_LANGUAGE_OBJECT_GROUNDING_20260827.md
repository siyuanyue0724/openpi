# ADR-226: Large-Host Language-to-Object Grounding

Date: 2026-08-27 Asia/Shanghai

Status: **RETAINED GROUNDING SUBDESIGN; SUPERSEDED AS THE COMPLETE TARGET BY ADR-228**

ADR-228 keeps the authenticated SWIM visual-grounding primitive from this ADR
but replaces the raw-intersection row target, temporal ABI and complete-system
plan. This document remains authoritative only for the SWIM source receipt and
the proof that ADR-225 lacked large-host language/object grounding.

## 0. Decision

If ADR-225 fails to produce a material fixed-bank action advantage by step 100,
do not extend it and do not add a selector head. Preserve its pretrained
mask-conditioned object memory, then train the shared LingBot/Qwen host itself
to ground instruction entity tokens in visual support and in the corresponding
soft set of object rows. The source operation is SWIM's six-depth attention
supervision; the object-row target is an explicit PICF basis projection and is
not falsely described as upstream SWIM.

## 1. Design laws

1. LingBot's 36-layer shared host owns language understanding, object relevance
   and action. No independent task-object classifier, winner head, lifecycle
   network or replacement decoder is allowed.
2. The released LingBot RGB/language prefix, action expert and all 200 ADR-225
   object rows remain present. Grounding supervises an existing attention
   computation; it does not replace context or hard-select one row.
3. Every operation copied from SWIM is byte/source receipted. Every Qwen2.5 to
   Qwen3, video to CALVIN, or visual-patch to object-row adaptation is named and
   independently ablated.
4. Target masks and task identity are training labels only. They are forbidden
   from observation inputs and deployment code.
5. Overlap and uncertainty are represented as soft target mass over a set of
   rows. A hard winner is mathematically wrong when two proposals overlap or a
   physical object occupies several proposals.
6. Context remains the full native LingBot prefix. The auxiliary objective
   cannot force every visual patch into an object row and cannot delete
   low-confidence evidence.
7. Failure at the registered gate rejects this composition. It does not permit
   an unregistered small module, threshold sweep or longer blind run.

## 2. Why ADR-225 alone is incomplete

For instruction `T`, native visual tokens `X`, object masks `M=(M_1,...,M_Q)`,
object rows `O=(O_1,...,O_Q)` and action `A`, ADR-225 supplies

```text
O_i = merger(E[X | M_i])
A   = LingBot(T, X, O, modalities, history).
```

This repairs `I(O_i; X under M_i)`, which was absent in ADR-207. It does not
directly optimize `I(T_entity; O_target)` or require `A` to depend on the
target row. With 200 redundant proposals, action loss can obtain a locally
good solution from the unchanged native prefix while largely ignoring `O`.
That explains the observed combination of finite action improvement, strong
all-query geometry and weak material advantage over the historical curve.

The missing factor is a learned large-host relation, not another object
feature projection.

## 3. Exact SWIM source primitive

The authenticated official SWIM implementation modifies the full Qwen2.5-VL
language model and freezes its vision encoder. It:

1. replays exact post-RoPE attention probabilities from the model's own `Q,K`;
2. reads layers `[2,7,12,17,22,27]`;
3. averages attention heads at each depth;
4. multiplies the six maps elementwise;
5. row-normalizes the fused map;
6. selects tagged instance-word tokens;
7. min-max normalizes each predicted spatial map;
8. resizes the labelled mask and applies binary cross entropy;
9. optimizes `0.05 * text_loss + mask_loss`.

These operations, including multiplication rather than an invented weighted
sum, are the source contract. SWIM demonstrates that grounding can be trained
inside the large VLM's own attention instead of assigning semantics to a small
external selector.

The official source is research code with hard-coded Qwen2.5 depths and data
format assumptions. Source fidelity therefore means preserving its semantic
operation while explicitly mapping architecture indices and typed spans; it
does not mean copying invalid tensor indices into a different model.

## 4. Target architecture

### 4.1 Exact language-to-native-visual arm

Let `L={l_1,...,l_6}` be six depths mapped by normalized depth fractions from
SWIM's 28-layer host to LingBot's 36-layer host. The mapping is frozen before
training and recorded as an adaptation. For an instruction entity-token span
`S_T` and native static-camera visual key span `S_X`, replay only the needed
submatrix from the executed post-MRoPE `Q,K` and mask:

```text
A_l = mean_heads softmax((Q_l[S_T] K_l^T) / sqrt(d) + C_l)[:, S_X]
F_X = normalize(sum_entity_tokens(product_l A_l))
L_visual = BCE(minmax(reshape(F_X)), resize(G)).
```

`G` is the labelled current-frame task-object mask. This is the closest
Qwen3/LingBot adaptation of SWIM and introduces zero trainable parameters.
Only selected query/key spans are replayed, so the receipt is much smaller
than materializing full `[sequence,sequence]` attention.

### 4.2 Soft object-basis grounding arm

Visual grounding alone does not prove that PICF rows carry the relation. Map
the same labelled support into the complete proposal basis without a winner:

```text
s_i = sum_p G_p sigmoid(M_ip)
y_i = s_i / (sum_j s_j + epsilon).
```

For each registered layer, replay the same entity-token query attention to the
200 posterior keys. Preserve probability mass on ordinary LingBot context for
diagnostics, but condition on posterior consultation for row identity:

```text
B_l(i) = mean_heads P_l(key=O_i | query=T_entity)
C_l(i) = B_l(i) / (sum_j B_l(j) + epsilon)
F_O    = normalize(product_l C_l)
L_row  = -sum_i y_i log(F_O(i) + epsilon).
```

This target is an explicit PICF adaptation, not an operation present in SWIM.
It is set-valued, differentiable, permutation-equivariant in row index and
well-defined under overlapping proposals. If one proposal exactly covers the
label and all others are disjoint, `y` converges to a one-hot target; otherwise
it preserves ambiguity rather than fabricating certainty.

### 4.3 Joint objective

The first registered causal arm is

```text
L = L_action + L_source + lambda_visual L_visual + lambda_row L_row.
```

No new trainable module is introduced. `L_visual` and `L_row` update the same
large LingBot host that computes action. VidEoMT retains its complete released
objective and ADR-225 retains the copied native merger projection.

Loss scales may not be tuned from the test curve. The source SWIM ratio is
preserved inside its visual-grounding branch; one declared gradient-norm
calibration on a training-only batch sets the branch-to-action scale before
the run, then freezes it. The row branch is tested separately before the joint
arm so the PICF-specific adaptation cannot borrow SWIM's credibility.

## 5. Multimodal scope

V-JEPA, Sonata, AnyTouch and proprioception remain typed LingBot evidence in
this ADR. They are not yet claimed to be object-bound merely because they are
visible to the same transformer. A later same-object multimodal objective must
use authenticated sensor correspondence and the same soft object basis; it
must not infer a physical binding from modality type alone.

This ordering is deliberate:

- ADR-225: object geometry receives native visual semantics;
- ADR-226A: language tokens learn spatial grounding in the large host;
- ADR-226B: the same large-host relation is projected into the object basis;
- only after both pass: bind authenticated 3D/touch/world evidence to the same
  posterior distribution and test causal action benefit.

Thus the design remains universal under missing modalities: absent evidence is
invalid and contributes no loss; no placeholder modality is hallucinated.

## 6. Information-theoretic claim and limit

ADR-226 increases a lower-bound surrogate for mutual information between the
instruction entity and labelled visual/object support. It does not prove that
the action uses the relation. The required causal chain is

```text
T_entity <-> X_target <-> O_target -> shared LingBot state -> A.
```

Each arrow needs an intervention:

- shuffle `G` across samples: grounding should fail;
- permute `M_i,O_i` together: predictions should be equivariant;
- permute rows without masks: row grounding and action should change;
- zero or shuffle object rows: action should degrade only after adoption;
- preserve rows but shuffle optional modalities: only the affected causal
  metric should degrade;
- remove history under occlusion: persistent-object and action metrics should
  degrade after temporal training, not before.

Without the action interventions, good grounding is representation evidence,
not a PICF performance result.

## 7. Historical non-repeat constraints

This ADR is not permission to reactivate failed mechanisms unchanged:

- ADR-177 task-query/object-read two-hop mediation failed;
- ADR-178 supervised only late action heads against hard Hungarian row
  bindings before rows had ADR-225 native semantics and failed to produce
  material action advantage;
- fixed object heads and shallow implicit readouts failed held-out transfer;
- direct WSA/Future3D additions regressed or remained causally unproved.

ADR-226 differs by grounding actual instruction entity tokens across six
shared-host depths, preserving soft set-valued targets and using ADR-225's
mask-conditioned native semantic rows. These differences are hypotheses and
must be factorially ablated; they are not asserted to guarantee success.

## 8. Implementation checklist

- [x] Authenticate official SWIM commit and critical file hashes.
- [x] Record exact source operations and source loss equation.
- [x] Establish that ADR-225 has no active large-host grounding objective.
- [ ] Freeze six LingBot depth indices by an architecture mapping receipt.
- [ ] Add zero-parameter selected-Q/K receipts for language-to-visual and
      language-to-posterior attention on the actual executed mask surface.
- [ ] Build instruction entity-token spans from authenticated CALVIN task text;
      reject substring guesses and ambiguous tokenizer mappings.
- [ ] Build current-frame labelled masks strictly on the loss side; prove they
      cannot reach observation preparation or deployment.
- [ ] Implement exact SWIM fusion/min-max/BCE with source-equality unit vectors.
- [ ] Implement the separately named soft object-basis target and permutation,
      overlap, empty-support and gradient probes.
- [ ] Prove unchanged LingBot outputs when both auxiliary scales are zero.
- [ ] Measure receipt memory/time against ADR-225 on released weights.
- [ ] Run matched training-only gradient calibration once and freeze scales.
- [ ] Execute A: visual grounding only; B: row grounding only; C: joint, all on
      the same fixed bank and stream.
- [ ] Require action, grounding and causal-intervention conjunction before 2k.
- [ ] Require closed-loop CALVIN and long-state tests before 30k.

## 9. Promotion rule

ADR-226 is eligible for a bounded cloud test only after source-equality,
no-leakage, zero-scale equivalence and released-weight memory probes pass.
It is eligible for 2k only if grounding improves materially and fixed action
does not regress. It is eligible for 30k only if the entire action curve beats
exact LingBot by the preregistered margin and row/multimodal/history
interventions demonstrate causal use.

Current theoretical maturity is `7.5/10`: the visual branch is source-backed,
the object-basis branch is mathematically coherent but new, and action benefit
is unproved. Deployment maturity is `0/10` because no ADR-226 code or run
exists. These scores must not be raised by documentation alone.

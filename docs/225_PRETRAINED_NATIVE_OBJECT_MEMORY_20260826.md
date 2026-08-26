# ADR-225: Pretrained Native Object Memory in LingBot

Date: 2026-08-26 Asia/Shanghai

Status: **APPROVED FOR ISOLATED IMPLEMENTATION; NO LONG TRAINING AUTHORIZED**

## 0. Decision in one sentence

Keep the released LingBot-VLA2 Qwen3-VL and action expert intact, keep the
complete VidEoMT object-query/mask source intact, and replace the failed
random-query-only posterior content with object memory computed from LingBot's
own pretrained native visual features by the source-faithful UniPixel pooling
and native-merger primitive.

This is not a claim that PICF is already useful. It is the narrowest mature
repair for the measured representation disconnect. Only a matched whole-curve
action and rollout result can promote it.

## 1. Non-negotiable law

1. Successful source topology is copied, not summarized into a smaller local
   substitute. Layer order, nonlinearities, initialization and source feature
   location remain exact.
2. Every unavoidable PICF/CALVIN/Qwen3 adaptation is named and ablated. It is
   never called an upstream reproduction.
3. LingBot remains the only semantic and action owner. No selector, lifecycle
   controller, task-object classifier or replacement action head is added.
4. Full image and language tokens remain in the original LingBot prefix.
   Object memory adds typed evidence; it never deletes context.
5. No SAM, target mask, task label or future frame is available at deployment
   time. Such information may be used only as a labelled training objective or
   an explicitly named oracle diagnostic.
6. The complete 200-query source is retained. No Top-K selection is allowed in
   the first causal arm because source objectness is known to be badly
   calibrated on CALVIN.
7. A finite update, attractive mask panel, or lower auxiliary loss is not a
   scientific pass. Promotion requires material action advantage over the
   exact LingBot baseline across the registered curve and closed-loop rollout.
8. A failed registered effect is stopped immediately. Scalar tuning and waiting
   for a round-number checkpoint are forbidden rescue operations.

## 2. Measured defect in ADR-207/221/224

The source mask bank is not the failed component:

- all-query oracle binary IoU is about `0.817`;
- all-query Recall@0.5 is about `0.910`;
- foreground-ranked Top-10 Recall@0.5 is only about `0.126`;
- exact fixed-set PICF action difference-in-differences at step 20 is
  `+1.147%` (positive means worse), 95% interval
  `[+0.479%, +1.619%]`;
- source/action gradient cosine is approximately `+0.001665`, so direct
  destructive gradient conflict is not the root cause;
- freezing the source changes neither conclusion.

The implementation explains the result. For source query `q_i`, the old row
content is

```text
z_i = P_random q_i + e_modality + e_posterior.
```

The complete source mask `M_i` and class distribution are attached only to a
diagnostic sidecar after the LingBot host. No operation pools LingBot/Qwen3
visual semantics under `M_i`. Consequently an excellent geometric mask does
not imply that the action-visible row contains the block, drawer or switch
semantics represented by that mask.

This is a representation-interface failure, not evidence that persistent
object posteriors are intrinsically useless.

## 3. Mature source primitive

UniPixel caches the exact input to Qwen2.5-VL's native vision merger MLP. For
each supplied object mask it selects the corresponding native spatial cells,
averages those cells, and applies a separately trainable MLP copied exactly
from the native merger. The resulting token already has language-model width
and replaces a memory token inside the large VLM.

For Qwen3-VL the exact corresponding merger is

```text
LayerNorm -> reshape merged 2x2 cells
          -> linear_fc1 -> GELU -> linear_fc2.
```

The hook point is the input of `linear_fc1`, exactly after native normalization
and 2x2 grouping. ADR-225 copies `linear_fc1`, `GELU`, and `linear_fc2` without
changing dimensions, bias, order or initial values.

The source receipt is
`references/adr225_pretrained_object_memory_sources.json`.

SWIM is retained as evidence for a later, separate stage: it supervises
language-token attention to object masks within the large VLM. It is not mixed
into the first object-memory causal test because doing so would confound the
representation repair with task grounding.

## 4. Target computation

Let:

- `H_t in R^(P x d_v)` be the exact grouped input to LingBot Qwen3-VL's native
  merger MLP for the current static camera;
- `L_t in R^(Q x h_s x w_s)` be complete VidEoMT mask logits in canonical
  query order, with `Q=200`;
- `R` be the source-consistent bilinear resize from the VidEoMT mask grid to
  the Qwen3 merged-token grid;
- `m(h) = W_2 GELU(W_1 h + b_1) + b_2` be an exact copy of the pretrained
  Qwen3 native merger MLP;
- `P_(t-1)` be the recurrent PICF prior supplied by the existing layerwise
  predict/correct path.

The differentiable posterior-mask adaptation is

```text
w_tiq = sigmoid(R(L_ti))_q
c_ti  = sum_q w_tiq H_tq / (sum_q w_tiq + epsilon)
o_ti  = m(c_ti)
```

and the correction pass is

```text
P_t = LingBot36(P_(t-1), RGB_t, language_t, typed modalities_t, O_t, controls_t)
```

where `O_t={o_t1,...,o_t200}` occupies the same canonical posterior rows used
by the source queries. The rejected random projection is not the sole row
content in this profile.

### 4.1 Why the soft mask is an explicit adaptation

UniPixel receives binary reference masks. PICF receives predicted mask logits.
Replacing a predicted mask by a hard threshold would break the action-to-mask
gradient and discard calibrated uncertainty. The normalized probability mean
above is the conditional-expectation relaxation of UniPixel's binary mean. If
all logits tend to plus or minus infinity, it converges to the exact binary
masked mean. This adaptation is therefore mathematically controlled but is
still a PICF hypothesis and must be ablated against a detached hard-mask arm.

### 4.2 Context and overlap

The original LingBot RGB tokens are unchanged, so unassigned background and
uncertain evidence remain context. Masks are not forced into a partition:
overlapping object hypotheses independently pool the same cell. The large
LingBot host can resolve the overlap from language, appearance, motion, 3D,
touch and history. No pixel-wise winner or hand-written context class is used.

### 4.3 No-object and query capacity

Every canonical query remains addressable in the first arm. This is deliberate:
CALVIN target masks often have donor foreground probability `0.001--0.013`, so
class-threshold or Top-K pruning would delete useful hypotheses. The native
image prefix remains available, and the 36-layer host is responsible for
ignoring redundant evidence. Support mass, row similarity and attention
concentration are logged so duplicate context rows can be diagnosed without a
new selector.

The architecture scales in query count as `O(B Q P d_v)` for pooling and does
not add transformer rows relative to the existing 200-row profile. A future
large-dataset capacity change must be learned or source-backed and is outside
this causal arm.

### 4.4 Missing modalities

V-JEPA2.1, Sonata and AnyTouch retain their typed evidence paths. Physically
absent evidence has exact-zero values and an invalid attention row. No missing
modality is hallucinated. ADR-225 does not claim that these global typed tokens
are already object-bound; same-object cross-modal binding requires a separate
supervised intervention after this visual-object bridge passes.

### 4.5 Registered hypothesis risks and falsifiers

The transplant closes a measured interface defect, but three properties are
PICF hypotheses rather than consequences of the UniPixel source:

1. **Normalized soft pooling loses support scale.** For any positive scalar
   `a`, replacing `w_i` by `a w_i` leaves `c_i` unchanged (up to `epsilon`). A
   low, nearly uniform posterior can therefore produce an apparently valid
   whole-image average token. `support_mass` is logged but is deliberately not
   consumed by a hand-written gate. The detached binary-mask control must show
   whether this relaxation, rather than the copied primitive, causes a failure.
2. **The complete 200-query bank can be redundant.** Retaining every row avoids
   deleting the poorly calibrated CALVIN foreground hypotheses, but many
   near-duplicate context rows may consume host attention. This is accepted in
   the first causal arm because a new confidence selector would move semantic
   ownership into an unproved small module. Object-memory versus shuffled-mask
   and row-permuted interventions must show that the host uses mask-specific
   content instead of row count or duplicated scene averages.
3. **ADR-225 replaces, rather than concatenates, the old projected query row.**
   This isolates whether native mask-conditioned semantics repair the failed
   interface without increasing sequence length, but it removes any useful
   detector-query statistic from the action-visible row. ADR-207 is therefore
   a mandatory exact control. A future combined representation is not
   authorized unless ADR-225 proves mask-specific necessity and a source-backed
   combination rule is identified.

These risks are not implementation defects that may be repaired post hoc. A
failed registered falsifier rejects the arm and triggers a new ADR; it does not
authorize a confidence head, threshold, Top-K rule or scalar sweep.

## 5. Information-flow argument

Let `X` denote LingBot's pretrained visual feature field, `M_i` a source object
hypothesis, `T` the instruction and `A` the action. The old interface supplies
the action host with a projected detector query but not with the selected
LingBot visual statistic:

```text
A <- LingBot(T, X, P_random q_i),     M_i is sidecar-only.
```

ADR-225 supplies

```text
A <- LingBot(T, X, m(E[X | M_i])).
```

The full `X` is still present, hence no original LingBot observation
information is removed. The added statistic is deterministic conditional on
`(X,M_i)`, and therefore can be ignored by a sufficiently expressive host if
irrelevant. It gives the host a direct same-index semantic statistic that the
old interface lacked. This establishes a strict representational repair, not
an action-performance theorem: optimization can still choose to ignore it.

## 6. Exact copy versus adaptation ledger

### Copied unchanged

- LingBot-VLA2 Qwen3-VL prefix construction, MRoPE, DeepStack and all 36 shared
  host layers;
- released LingBot Qwen2 action expert and flow-matching ABI;
- complete VidEoMT query/mask/class source and canonical 200-query order;
- UniPixel hook location, masked native-feature mean, copied merger MLP
  topology, activation, dimensions, biases and initialization;
- Qwen3-VL native merger layer definitions from Transformers 4.57.3.

### Explicit adaptations

- Qwen2.5 merger names are mapped to the exact Qwen3 merger names;
- VidEoMT logits are bilinearly resized to Qwen3's merged-token grid;
- binary reference-mask averaging is relaxed to normalized posterior-probability
  averaging in the production arm;
- the static CALVIN camera is selected from LingBot's fixed three-camera ABI;
- object memory seeds PICF posterior correction rows instead of textual
  `<mem>` placeholders;
- canonical query index remains the temporal address used by PICF recurrence.

### Forbidden in ADR-225

- SAM/SAM2 inference or labelled target masks as observation input;
- Top-K, task-object, class or confidence selector heads;
- manually coded lifecycle, decay, release or winner rules;
- replacement or reinitialization of the LingBot action expert;
- WLA, WSA or FLARE modules from rejected/confounded arms;
- silent source simplification.

## 7. Engineering contract

1. Installation finds `host.qwenvl.visual.merger` and fails closed unless it
   has the exact Qwen3 `linear_fc1 -> GELU -> linear_fc2` topology.
2. Copied parameters must be tensor-equal to source parameters before the first
   optimizer step.
3. A visual forward-pre-hook captures `grid_thw`; a pre-hook on the source
   `linear_fc1` captures its exact grouped input. One observation consumes one
   capture generation exactly once.
4. The capture validates total token count, batch/camera factorization,
   temporal grid size, static-camera slot and relation grid shape.
5. No tensor is detached in the probability-mask arm. Gradients must reach the
   copied merger projection, source mask logits and LingBot visual features.
6. The object rows reuse the existing 200 posterior positions, so action
   sequence length and compact-cache ABI do not grow.
7. FSDP state, optimizer coverage, DCP restart and source digests include the
   copied merger projection.
8. The old ADR-207 profile remains behaviorally unchanged for an exact ablation.
9. The ADR-225 launcher defaults to a 250-step bounded run. Up to 2000 steps
   may close the registered representation/action curve; a longer run fails
   closed unless `PICF_ADR225_LONG_RUN_AUTHORIZED=1` is supplied after Gate D.

## 8. Required probes before cloud training

- [x] Source topology and byte/tensor equality at installation.
- [x] Hook generation is single-use and rejects stale, missing or double use.
- [x] Multiview split selects CALVIN static slot 0 and rejects an unexpected
      camera count or temporal grid.
- [x] One-hot mask reproduces the exact corresponding native merger token.
- [x] Binary-mask mode reproduces UniPixel mean-plus-copied-MLP numerically.
- [x] Soft-mask limit converges to binary-mask output.
- [x] Overlapping masks remain independent; no winner is introduced.
- [x] Full native image/language prefix bytes and indices are unchanged.
- [x] Gradient reaches copied MLP, mask logits and native visual features.
- [x] Missing optional modalities remain exact-zero/invalid.
- [x] Long prior plus zero current evidence preserves a causal prior path.
- [x] No future frame, target mask or task label reaches observation inputs.
- [ ] Two-rank FSDP forward/backward/update and DCP restart pass.
- [ ] Added wall time and peak memory are reported against exact LingBot.

The checked items above are local executable contracts, not released-weight or
action evidence. The fixed CALVIN camera contract is independently grounded in
the official data path: `calvin_data.json` declares top/left/right, the robot
mapping supplies top/left, and LingBot `prepare_images` inserts the invalid
right-camera placeholder in the declared order before `embed_prefix` flattens
`[batch,camera]` to batch-major images.

## 9. Scientific gates

### Gate A: representation mechanics

All probes in section 8 pass. This authorizes only a bounded causal experiment.

### Gate B: object memory effect

On a fixed held-out bank, mask-conditioned rows must show materially higher
same-object consistency than shuffled-mask and row-permuted controls, without
using task labels. Geometry alone is insufficient.

### Gate C: action curve

Run exact LingBot, ADR-207 random-query posterior and ADR-225 object memory on
the same samples, seeds, optimizer contract and checkpoints. Compare the entire
curve, with registered windows rather than one endpoint. Promotion requires at
least a 5% held-out action-loss advantage over exact LingBot with a source-
episode bootstrap interval excluding zero, no material middle-window
regression, and a matching causal intervention.

### Gate D: rollout and persistence

Require closed-loop CALVIN success improvement and 32/64/128-step causal replay
under occlusion, reappearance, overlap, row permutation and missing modality.
Only Gate D can authorize 30k.

## 10. Claim boundary

If ADR-225 passes only Gate A, the code is mechanically credible. If it passes
Gate B, the pretrained visual-object interface is useful. If it passes Gate C,
PICF has action evidence. If it passes Gate D, long training is justified.

Before those results, the honest theoretical maturity is `8/10` and deployment
maturity is `6/10`: the repair is source-backed and mathematically coherent,
but no design document can substitute for causal action and rollout evidence.

## 11. Implementation closure on 2026-08-26

The isolated ADR-225 implementation now contains:

- one exact copied Qwen3 `linear_fc1 -> GELU -> linear_fc2` projection with
  source/copy tensor equality and SHA-256 receipts;
- one-shot hooks on the native Qwen3 visual call and source merger input;
- static-camera extraction under the official three-camera batch-major ABI;
- differentiable soft-mask pooling into the existing 200 same-index posterior
  rows while preserving the original image/language prefix;
- optimizer ownership, state-dict round-trip and per-step support diagnostics;
- a separate source-freeze and two/four-rank launcher that leaves ADR-207
  available as the exact random-query control.

The focused transplant/runner/source-boundary suite passes `167/167`; the
expanded host, training, runtime and VidEoMT boundary suite passes `323/323`.
The broader inherited sparse repository suite previously produced `2456`
passes, `31` failures and `13` skips. Sampling every failure class against the
untouched ADR-221 base reproduced those failures: they are missing historical
documents/assets and pre-existing source-contract debts, not regressions caused
by ADR-225. They are not silently relabelled as passes.

### Remaining blockers

1. A real released-weight two-rank FSDP forward/backward/update must prove that
   Python capture hooks survive sharding and BF16 execution.
2. A separate-process DCP restore must prove exact copied-projection recovery.
3. Runtime and peak-memory deltas must be measured against exact LingBot and
   ADR-207 on identical inputs.
4. The detached hard-mask arm must be run as the source-faithful control for the
   soft posterior adaptation.
5. Same-object/shuffled-mask/row-permutation probes, matched action curves and
   closed-loop CALVIN remain wholly unproved.

Accordingly, the local transplant is complete, but Gate A is still open and no
30k run is authorized.

## 12. 2026 source review boundary

The review also checked the official Qwen3-VL, V-JEPA2.1 and Isaac GR00T N1.7
releases. They strengthen the choice of a large shared Qwen3-VL semantic host,
dense predictive visual evidence and typed multimodal/action data contracts.
None publishes a source-complete replacement for the exact operation required
here: converting an external 200-query mask posterior into same-index native
LingBot object-memory tokens while preserving the released LingBot action ABI.
Therefore those systems remain architectural evidence, not copied code falsely
attributed to ADR-225. The only transplanted object-memory primitive in this ADR
is the authenticated UniPixel implementation listed in the source receipt.

## 13. Four-A100 execution ledger

### `adr225-a4b8b15-4gpu-30k-v2`

The first authorized four-A100 launch passed the exact native-source replay,
loaded all six LingBot shards, formed the 36-text/24-vision-block FSDP2 topology,
completed the fixed step-zero held-out evaluation, and committed one complete
forward/backward/AdamW update. Step 1 took `75.78 s`; rank-zero official action
loss was `0.63671875`. Step 2 failed in the unchanged FlexAttention backward:
rank 3 needed another `1.03 GiB` with only `755.56 MiB` free after AdamW state
materialization. This is an execution-placement failure, not action or anchor
evidence, and the run was stopped immediately.

The step-zero visual audit establishes a precise boundary. The complete 200-query
bank already contains oracle-matched masks with approximately `0.736--0.942` IoU
for most inspected table parts and `0.899` for the red block, while the blue-block
oracle is only `0.169`. The untrained model-ranked Top-10 does not select those
high-IoU rows. Thus pretrained geometric candidates exist, but learned ranking
and multimodal/action binding remain unproved.

### Accepted execution-only repair

ADR-225 now reuses ADR-224's exact eight-overlay PyTorch FSDP2 placement, including
the VLM explicit-dispatch patch with SHA-256
`7634367ee5dbfe08161c405a25a4e44014d2fdf3a9bc6ecb6cef5331840a93c9`.
ADR-224 `v17` already completed two full four-A100 updates after optimizer-state
materialization and reduced the step-two reservation from the rejected
`39.49 GiB` peak to `33.404 GiB`. For ADR-225 the same policy offloads the exact
Qwen3-VL text block class and trainable vision blocks. It does not change module
functions, tensors, query count, modalities, data order, losses, learning rates,
trainability or optimizer equations. This is a copied, previously demonstrated
execution mechanism; it carries no scientific credit by itself.

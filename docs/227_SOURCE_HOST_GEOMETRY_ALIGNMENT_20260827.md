# ADR-227: Source/Host Geometry-Aligned Object Memory

Date: 2026-08-27 Asia/Shanghai

Status: **GATE COMPLETE; RETAIN ALIGNMENT REPAIR; REJECT AS SUFFICIENT ACTION REPAIR**

## 0. Decision

ADR-225 results before this repair are not valid evidence for or against the
action value of mask-conditioned object memory. The training source masks and
the LingBot visual features were expressed in different random coordinate
systems. Stop that arm rather than extending it. Preserve the complete released
VidEoMT training path, but separate it from a deterministic, host-aligned online
view that alone owns recurrent source state and supplies object masks to
LingBot.

This repair adds no learned decoder, selector, scorer or lifecycle module. Both
views execute the same released VidEoMT parameters.

## 1. Measured root defect

For raw current RGB `R_t`, the source training adapter sampled a clip-consistent
random transform `g_s` containing resize, optional crop and horizontal flip:

```text
R_source = g_s(R_t:t+4)
M_source = g_s(M_t:t+4).
```

The source objective was internally correct because RGB and labels shared
`g_s`. LingBot was explicitly configured with `image_augment=False`, however,
so its native visual features were

```text
X_host = QwenVision(R_t).
```

ADR-225 then resized source mask logits only by tensor dimensions and pooled
`X_host` at the same numeric indices:

```text
O_i = merger(sum_p sigmoid(M_source_i[p]) X_host[p]
             / sum_p sigmoid(M_source_i[p])).
```

For a crop or flip, source index `p` and host index `p` do not denote the same
physical point. Bilinear interpolation changes resolution but does not invert
`g_s`. Therefore the pooled token is not an estimator of the object described
by the source mask.

A second violation affected time. The propagated source query from random
geometry `g_(s-1)` was reused under independently sampled `g_s`. The state was
therefore asked to preserve object identity while its coordinate frame changed
discontinuously for reasons absent at deployment.

The defect is deterministic from the code path:

- `prepare_calvin_videomt_training_clip` applies released random crop/resize/
  flip jointly to source RGB and source labels;
- LingBot `FeatureTransform` is instantiated with `image_augment=False`;
- `run_complete_native_videomt_lingbot_step` attached source masks to that
  independently prepared host batch;
- `NativePretrainedObjectMemory.consume` performed only a direct bilinear resize.

## 2. Corrected shared-weight transaction

Use two activation paths through one source model `F_theta`.

### 2.1 Released source-training view

```text
Q_source, Y_source = F_theta(g_s(R_t:t+4); q_learned)
L_source = L_VidEoMT(Y_source, g_s(M_t:t+4)).
```

This branch retains the released five-frame augmentation, architecture,
prediction heads, matching and complete objective. It starts from the learned
query bank for each sampled source clip, matching the independent-video source
training contract. It does not carry online state across unrelated random
coordinate frames.

### 2.2 Deterministic online host view

Let `h` be the released deterministic resize/pad path. It preserves normalized
image coordinates and applies neither crop nor flip:

```text
Q_online_t, Y_online_t = F_theta(h(R_t); Q_online_(t-1))
O_i = merger(E[X_host | sigmoid(Y_online_t.mask_i)]).
```

Only `Q_online_t` is committed as recurrent source state. Only `Y_online_t` is
attached to the LingBot host. The host sees current data only; future source
frames never enter its activation graph.

### 2.3 Joint optimization

```text
L = L_action(Y_online_t, R_t, history, modalities) + L_source(Y_source).
```

Both terms update the same `theta`; no teacher copy or extra source network is
introduced. The source branch preserves robust object learning under mature
augmentation, while the online branch makes object pooling and long-lived state
well-defined. The extra online forward is one source frame in addition to the
released five-frame source transaction; expected source-forward overhead is
about 20%, while end-to-end wall time and memory remain empirical gates.

## 3. Causality and information contract

The action activation at time `t` depends on raw current observations,
authenticated optional modalities and committed state from at most `t-1`.
Future frames `t+1:t+4` occur only in `L_source`. Their gradients train shared
parameters, as in ordinary auxiliary future prediction, but their activations
are not exposed to the action forward or committed state.

The repair removes a false cross-coordinate association; it does not by itself
prove task grounding, row relevance, optional-modality identity or action
advantage. ADR-226 large-host grounding remains conditional on the corrected
ADR-225 curve.

## 4. Implementation and evidence

- [x] Add a deterministic current-frame tensor to the typed source batch.
- [x] Preserve the complete augmented five-frame source tensor and targets.
- [x] Execute source-training and host-online views through shared source weights.
- [x] Reset the augmented training transaction to learned queries.
- [x] Carry previous source state only through the deterministic online view.
- [x] Require the pretrained-object-memory profile to provide the aligned view.
- [x] Add a regression proving arbitrary changes to the augmented source view
      cannot change the host-visible current output or committed online state.
- [x] Pass focused cloud contracts: `15 passed`, with only pre-existing upstream
      deprecation warnings.
- [x] Restore the registered PyTorch 2.8 compatibility environment. The first
      four-rank launch reached fixed evaluation and failed before any model
      result because its command omitted the existing
      `videomt-torch280-functorch-v1` overlay. Direct import with the complete
      overlay path passes; this was an environment-launch failure, not a model
      observation.
- [x] Pass released-weight four-rank forward/backward/update without OOM.
- [x] Measure wall time and peak memory relative to ADR-225.
- [x] Compare the fixed step-20 action and anchor curves against ADR-207 and
      the invalid pre-repair ADR-225 arm. Stop at the decisive step-20 gate
      instead of spending four-A100 time to confirm a non-material margin at
      step 100.

### 4.1 Four-A100 result

The corrected run is
`adr226-geometry-aligned-4gpu-step100-v3`. It completed the fixed step-0
evaluation and 22 finite updates; it was intentionally terminated immediately
after the step-20 fixed artifacts committed. Step 1 took `81.252 s`; the
step-2--20 rank-zero median was `59.219 s`, versus `58.963 s` before the
repair. Thus the additional one-frame shared-source execution costs only about
`0.43%` measured steady-state wall time in the complete workload, not the
source-only upper-bound estimate of 20%. Maximum recorded reserved memory over
all ranks was `35.252 GiB`, below the registered `39 GiB` gate.

Fixed action loss is:

| arm | heldout step 0 | heldout step 20 | validation step 0 | validation step 20 |
|---|---:|---:|---:|---:|
| ADR-227 aligned | `0.477826` | `0.396829` | `0.463436` | `0.361903` |
| invalid ADR-225 | `0.477826` | `0.408419` | `0.463436` | `0.386202` |
| historical ADR-207 | `0.483097` | `0.385412` | `0.454992` | `0.363023` |

Alignment therefore repairs real damage: relative to invalid ADR-225 it lowers
heldout loss by `2.84%` and validation loss by `6.29%`. It does not establish
the required advantage over history: it remains `2.96%` worse than ADR-207 on
heldout and is only `0.31%` better on validation. Averaging the two fixed
partitions, ADR-227 remains about `1.38%` worse than ADR-207.

The object bank remains geometrically strong but semantically unranked.
Heldout full-bank soft/binary IoU at step 20 is `0.791590/0.817855`; validation
is `0.796315/0.821867`. Heldout Top-10 soft IoU is only `0.106464` and
Recall@0.5 is `0.122975`; neither improves from step 0. Human review confirms
both sides of this aggregate: one heldout image has tight small-block oracle
masks around `0.93` IoU, while another blue block remains around `0.18`, and
the foreground-ranked Top-10 often omits the useful query.

## 5. Promotion and rejection rule

Engineering promotion passed: the repair is finite, causal, source-complete and
cheap enough to retain. Scientific promotion failed: the mixed fixed result is
not a material whole-curve advantage and Top-10 row semantics remain flat.
Retain the geometry/state correction as a necessary invariant, but reject it as
the sufficient action repair and proceed to ADR-226 large-host grounding. Do
not add a small selector, tune thresholds or extend this arm blindly.

The alignment repair itself has theoretical maturity `9/10` and deployment
maturity `9/10`; its remaining point is reserved for longer-state/rollout
evidence. The current PICF composition has scientific maturity `4/10`: it has
strong object candidates and now-correct coordinates, but not yet a learned
large-host language-to-row relation, optional-modality identity or causal
action adoption.

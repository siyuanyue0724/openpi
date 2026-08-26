# ADR-227: Source/Host Geometry-Aligned Object Memory

Date: 2026-08-27 Asia/Shanghai

Status: **IMPLEMENTED; FOCUSED CONTRACTS PASS; FOUR-GPU CURVE ACTIVE**

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
- [ ] Pass released-weight four-rank forward/backward/update without OOM.
- [ ] Measure wall time and peak memory relative to ADR-225.
- [ ] Compare the fixed step-20/100 action and anchor curves against exact
      LingBot, ADR-207 and the invalid pre-repair ADR-225 arm.

## 5. Promotion and rejection rule

Engineering promotion requires finite four-rank backward/update and a committed
online state receipt. Scientific promotion requires a material fixed-bank curve
advantage, not merely lower training loss. If geometry alignment does not
improve fixed action or causal row use, reject mask-conditioned object memory as
the sufficient repair and proceed to the already documented ADR-226 large-host
grounding arm. Do not add a small selector or tune thresholds to rescue it.

Current theoretical maturity is `8.5/10`: the coordinate and state correction
is mathematically necessary and preserves the mature source algorithm, but the
two-view composition is a PICF adaptation. Deployment maturity is `4/10` until
released-weight backward and fixed-curve evidence complete.

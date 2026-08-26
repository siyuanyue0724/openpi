# ADR-223: Shared World Interface Decision

Date: 2026-08-26 Asia/Shanghai

Status: **ADR-222 REJECTED; NO 30K RUN AUTHORIZED; NEXT ARCHITECTURE SOURCE
FIDELITY DECISION IN PROGRESS**

## 0. Hourly operating law

Until the 11:00 CST decision boundary, every architecture and experiment review
must restate and enforce these rules:

1. Do not simplify a successful upstream method. Prefer pinned upstream source,
   checkpoints, preprocessing, optimizer, and action interfaces over a local
   reimplementation. Any unavoidable adaptation is named as a PICF hypothesis,
   not as a reproduction.
2. A scientific failure is stopped immediately. Do not wait for a round-number
   step, tune a scalar, add a selector, or investigate an irrelevant local patch
   after the registered effect has already failed.
3. PICF promotion requires a material whole-curve advantage over the exact
   LingBot control. A roughly one-percent change or a regression is a rejection,
   not a near-success.
4. One large model must own semantics, object/world representation, uncertainty,
   temporal correction, and action. Projection and typed transport are allowed;
   a private scorer, detector, lifecycle controller, or action head is not.
5. All evidence, source identities, negative results, and recovery commands must
   remain durable under `/mnt`.

## 1. Exact ADR-222 verdict

ADR-222 retained the complete LingBot 36-layer host/action expert, the complete
200-query VidEoMT source, all authenticated AnyTouch/Sonata/V-JEPA evidence, and
the complete adapted 36-layer/432-slot WSA Future3D expert. It made two causal
changes: action could not read Future3D keys, and current visual/dense evidence
could reach action only through the 200 posterior rows. Task language and typed
proprioception remained direct.

The run was exactly paired to ADR-207 by rank, sample key, flow seed, timestep,
and temporal plan. It completed 16 updates and was stopped before update 17:

| Window | Baseline action | ADR-222 action | Relative gain |
|---|---:|---:|---:|
| 1--5 | 0.56552734 | 0.53945312 | +4.611% |
| 6--10 | 0.43984375 | 0.45761719 | -4.041% |
| 11--15 | 0.46650391 | 0.46152344 | +1.068% |
| 11--16 | 0.45369466 | 0.45003255 | +0.807% |
| 1--16 | 0.48431396 | 0.48034668 | +0.819% |

The aggregate confidence interval crosses zero and the middle window is
materially worse. Under rule 3 this is a scientific rejection. Waiting for step
20, generating another image, or tuning a route weight would have no authorized
interpretation.

Durable receipt:

```text
/mnt/picf-next/adr222/evidence/adr222_gate20_rejection.json
```

## 2. Why the historical early advantage is not a rollback target

ADR-149 is often remembered as a clearly faster PICF. Its complete matched
curve is more restrictive:

| Step | LingBot held-out / validation | ADR-149 held-out / validation |
|---:|---:|---:|
| 20 | 0.365522 / 0.366670 | 0.324449 / 0.343491 |
| 100 | 0.290585 / 0.294434 | 0.290412 / 0.308938 |

At step 20 the aggregate paired delta was `-0.032126 +/- 0.011369 SE`, a real
early signal. At step 100 held-out was tied and validation was significantly
worse. Candidate action improved less than LingBot from steps 20 to 100 while
its structural and predictive losses continued to improve. The arm was closed
at step 255; it was never authorized for 2K or 30K.

ADR-149's Pass B let action read native RGB, language, proprioception, prior,
and posterior simultaneously. It therefore preserved the released LingBot
interface but did not identify posterior mediation. Later every-step posterior
routing improved predictive loss by 28.95% at step 200 while fixed action moved
only about 0.18%, and wrong-row action was exactly invariant. The old early
gain is evidence that extra shared-host state can transiently help optimization;
it is not evidence that the policy learned persistent object identity.

## 3. Structural root cause

Let `H = f_phi(O,L)` be the complete current vision-language sequence, `P` a
task-independent posterior bank, `A` the action, and `Y` a future/world target.
The rejected families implement one of these graphs:

```text
H ------> A               # easy released bypass
 \-> P -> A               # optional aggregate side channel

H -> P -> A               # ADR-222 exclusive route, but P was not jointly
                          # pretrained with the released action interface
```

The first graph permits a risk minimizer that ignores row identity:

```text
I(A; row_identity(P) | H) = 0.
```

The second removes the shortcut but replaces the action expert's pretrained
conditioning distribution. High oracle mask quality cannot guarantee a useful
action representation because the sufficient statistic for segmentation is not
necessarily sufficient for task-conditioned control:

```text
I(P; object_masks) high  does not imply  I(P; A | L, proprio) high.
```

This matches the measurements: the full 200-query bank has high oracle spatial
capacity (`IoU ~= 0.82`, recall@0.5 `~= 0.91--0.93`), but autonomous ranking is
weak and replacing the direct LingBot context with those rows gives no material
action advantage.

The required invariant is stronger:

```text
Q = B_psi(f_phi(O,L))
A ~ g_theta(. | Q)
Y ~ w_omega(. | Q)
```

`Q` must be formed from the complete visual-language sequence, must be the
action expert's sole visual-language context from the beginning of joint
training, and must receive both action and world-model gradients. This is an
interface-pretraining requirement, not a route-mask detail.

## 4. Historical non-repeat matrix

The next experiment must not repeat any of these rejected families:

| Family | Decisive evidence | Decision |
|---|---|---|
| 16-row two-pass plus direct scene bypass | step-20 gain vanished by step 100 | reject unchanged continuation |
| 100% posterior-only route | predictive -28.95%, action approximately tied, wrong-row null | reject dose escalation |
| task-query/object-read two-hop route | action did not adopt target row | reject |
| direct action-to-posterior route | aggregate use without target-row selectivity | reject |
| GuidedVLA-style fixed object heads | in-distribution optimization, cold held-out failure | reject |
| shallow implicit relation/readout | useful structure but unusable fine-object masks | reject |
| resolution/deep-supervision-only changes | historical controlled negative results | reject |
| SAM proposal/mask input | violates implicit shared-host ontology; query control failed | diagnostic only |
| reduced/frozen final VidEoMT readout | approximately 0.05--0.08 IoU | terminally reject |
| direct Future3D-to-action WSA | matched action regression | reject |
| FLARE generic future target | constant-template shortcut; no action gain | reject |
| object-future token set / row transition variants | target/control identifiability or action gate failed | reject |
| ADR-222 exclusive 200-row route | +0.819% over steps 1--16 | reject |

## 5. Upstream source audit at the 2026-08-26 cutoff

### 5.1 World Tokens (arXiv 2608.09730)

This is the closest successful architecture in mathematical form:

- Qwen3-VL-2B jointly encodes all views and the full instruction;
- a World Adapter uses 256 learned queries, 12 blocks, width 2048, 8 heads,
  cross-attention to the complete VLM sequence, query self-attention, and an
  FFN whose inner width equals 2048 (approximately 0.5B parameters);
- those tokens are the action DiT's sole visual-language context;
- the same tokens condition a jointly fine-tuned Cosmos Predict2.5-2B denoiser;
- the denoiser receives a Canny edge first-frame anchor rather than RGB;
- action and adapter are trained from scratch; VLM and denoiser are jointly
  fine-tuned; the VAE target encoder is frozen;
- loss is `L_action + 0.5 L_video`;
- LIBERO uses 70K steps, effective batch 128 on 8 H100s.

Its controlled ablations are directly relevant. Full performance is 98.2
average / 97.0 Long. VLM bypass falls to 97.2 / 94.1, an FFN adapter falls to
97.0 / 93.4, and removing world modeling gives 97.4 / 95.0. On R1 Pro the
matched baseline improves from 59.4% to 76.0% success.

No official repository or training code is published. The arXiv source gives
precise mathematics and hyperparameters but not omitted implementation details.
Therefore a LingBot version would be a source-grounded adaptation, not an exact
source transplant.

Pinned arXiv source SHA-256:

```text
2f32904f48422687244b3422bc43eff072e1c70883932232a72b20079c012fa5
```

### 5.2 WLA (official source commit 155ac94e...)

WLA is the strongest complete code donor found. It appends 64 learned
metaqueries inside the full Qwen3-VL/RynnBrain sequence, extracts their states
from every backbone layer, and feeds the last 28 layers to a complete 28-layer,
width-1024 flow-matching action DiT by layerwise cross-attention. A separate
world expert consumes the shared states. The repository includes training code,
configs, checkpoints, and dataset adapters.

The action DiT, metaquery tokenization, backbone, and training recipe form one
jointly trained interface. Copying only its cross-attention into LingBot would
not reproduce WLA. Exact reuse therefore requires switching the complete model
family, which conflicts with the current LingBot-backbone constraint.

### 5.3 Other inspected 2026 sources

- W2-VLA has complete code and a Qwen3-VL-4B/V-JEPA2.1/action stack, but its
  wrist-world adapter is not a persistent object/world-token action interface.
- WSA-Large provides complete full-depth world/action mixed attention and was
  already transplanted without reduction. Its direct action coupling regressed,
  and ADR-222's acyclic repair remained approximately tied.
- POT-VLA reports a large persistent-object benefit, but no official code was
  found. It cannot be a code donor under rule 1.
- RepWAM advertises code/weights, but the inspected repository contains only
  documentation/assets and no trainable implementation.

## 6. Decision options

### Option A: exact WLA family

Use the complete WLA backbone, metaquery tokenizer, 28-layer action DiT,
optimizer, and checkpoints without reducing them. CALVIN is only a dataset and
action-schema adaptation. This satisfies source fidelity but changes the base
model and does not by itself add PICF persistence.

### Option B: LingBot-native World Tokens reconstruction

Retain LingBot, instantiate the full 256-query/12-block/2048-wide World Adapter,
route the complete visual-language sequence into it, make it the sole
visual-language context for the complete LingBot action expert, and train a
matched action-only control and world-supervised candidate from the same
conditioning interface. Use the complete jointly trainable video denoiser and
the exact edge-anchor recipe.

This preserves the desired base and mathematical graph, but no official World
Tokens code exists. The Perceiver algebra can be copied from LingBot's pinned
OpenFlamingo-derived resampler source, yet the LingBot action integration and
world-denoiser composition remain an adaptation hypothesis. It must never be
reported as an exact World Tokens reproduction.

### Option C: continue the current 200-row architecture

Rejected. Every plausible route/dose/decoder variant has already failed the
material action or identity-selectivity gate.

## 7. Current recommendation

Do not launch another current-PICF short run or a 30K job. Under the simultaneous
requirements of retaining LingBot and copying a complete successful source,
there is no source-faithful executable candidate today. The scientifically
strongest LingBot-compatible design is Option B, but it requires explicit
authorization as an adaptation and a fair training regime: World Tokens trains
its adapter and action interface from scratch for 70K steps at batch 128. A
20-step requirement for immediate superiority is not supported by that source.

If exact-source fidelity is strictly dominant, approve Option A and treat WLA
as the new policy family. If retaining LingBot is strictly dominant, approve
Option B and accept that the source-equivalent claim is mathematical rather
than code-identical.

## 8. Pre-implementation checklist

- [x] Persist ADR-222 rejection before any successor change.
- [x] Reconstruct ADR-149's complete curve and reject simple rollback.
- [x] Record all major historical negative controls.
- [x] Inspect exact World Tokens arXiv source and hyperparameters.
- [x] Inspect exact WLA tokenization, layerwise extraction, action DiT, config,
      checkpoint, and dataset paths.
- [x] Confirm that no official World Tokens or POT-VLA code is currently found.
- [ ] Select Option A or Option B without silently changing the base constraint.
- [ ] Freeze complete donor/source hashes and an immutable implementation map.
- [ ] Run shape/causality/gradient/mechanics gates without scientific credit.
- [ ] Run a matched whole-curve action gate against the exact same-interface
      control; stop immediately on a non-material effect.
- [ ] Authorize 30K only after material action advantage, source-disjoint
      world-token adoption, checkpoint/resume, and CALVIN rollout evidence.


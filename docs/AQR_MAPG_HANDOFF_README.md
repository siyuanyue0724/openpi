# AQR-MAPG Handoff Readme

Date: 2026-05-09

This is the handoff index for the current PICF/AQR-MAPG line. It is meant for
a new engineer or researcher who needs to understand the current design,
which README files are authoritative, what is deployed, what is historical,
and where the remaining risks are.

## 1. One-Sentence State

The maintained graph/anchor path is now:

```text
AQR-MAPG:
  learned physical/task anchor queries
  over typed support memory
  with competition, cross-modal support pooling, posterior memory,
  and PI0.5 action path unchanged.
```

The old MAPG-v0 candidate-prior assignment path is not the production design.
The PaliGemma heatmap/grounding head is disabled by default for AQR production
runs. PaliGemma still contributes language/semantic tokens and image-token
visual-semantic support for task queries.

## 2. Read These First

Read in this order.

1. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md`

   Current canonical PICF operator/developer README. It records the current
   deployment profile, training contract, freeze policy, and links to the
   current AQR documents. Treat this as the top-level PICF entry point.

2. `/home/siyuanyue/Documents/openpi/docs/AQR_MAPG_DIRECT_FINAL_DEPLOYMENT_README.md`

   Current AQR-MAPG final architecture contract. This is the most important
   design document for the anchor/router direction. It explains why MAPG-v0 was
   replaced, what AQR-MAPG is, how PaliGemma is used, what is trainable, what is
   disabled, and what evidence/acceptance tests matter.

3. `/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md`

   Final PICF-AQR-OWM deployment contract and concrete code-level blueprint.
   Read this after the AQR direct-final README if you are implementing the next
   architecture step. It audits the current code file-by-file and specifies the
   exact final interfaces for object-addressable posterior belief state,
   first-class recent V-JEPA temporal support, first-class PaliGemma image
   support, posterior-grounded evidence cache, slot-level JEPA prediction,
   support prediction, relation/ordinal grounding, diagnostics, tests,
   acceptance criteria, a file-by-file code audit/deployment map, a final
   Definition of Done, and a point-by-point resolution of the proposed method as
   adopted, guarded, or rejected. It is the direct-to-final OWM target, with
   hard guards and `scripts/verify_picf_owm_contract.py` preventing any
   unchecked branch from being mislabeled as final OWM before the
   implementation, diagnostics, and no-leakage checks pass.

4. `/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_DEPLOYMENT_STATUS_TEMP.md`

   Temporary live deployment ledger for the current OWM implementation branch.
   Read it when reviewing whether the final README contract has actually been
   wired into code and which guarded losses remain at zero/default weight.

5. `/home/siyuanyue/Documents/openpi/docs/AQR_MAPG_DEPLOYMENT_README.md`

   Evidence and reasoning document for the MAPG-v0 failure mode. Read this to
   understand why near-uniform PaliGemma heatmaps and same-role row symmetry
   made the old candidate-prior assignment path behave as control-path noise.

5. `/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md`

   CALVIN training/evaluation operations record. Use this for evaluation
   commands, video/debug generation, cloud validation, and checkpoint/eval
   conventions.

6. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`

   Frozen-perception profile and augmentation details. Useful for understanding
   the current "freeze V-JEPA/Sonata/AnyTouch pretrained parts, train the PICF
   adapters/router/semantic-action stack" profile.

7. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_PI05_PARITY_AUDIT.md`

   PI0.5 parity and integration audit. Read this before changing anything that
   touches the action generator path.

## 3. Historical Documents

These are useful for context, but they are not the current production route.

- `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_MAPG_PICF.md`

  Historical MAPG-PICF architecture contract. Useful for understanding the
  support-graph idea, but the old MAPG-v0 live path has been superseded by AQR.

- `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_VL_GUIDED_ANCHOR_ROUTER.md`

  Historical PaliGemma/VL-guided anchor router path. Do not treat PaliGemma
  heatmap-to-point routing as the production AQR where mechanism.

- `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md`

  Older v2.1 architecture/ops archive. Use only for ancestry and regression
  context.

## 4. Current Design

The current system should be understood as four layers.

### 4.1 Dense/Typed Support Memory

Inputs are encoded into typed evidence sources:

```text
PaliGemma semantic/text tokens:
  task semantics and language context

PaliGemma image tokens:
  visual-semantic support for task queries

V-JEPA visual tokens:
  main dense visual substrate

Sonata/point tokens:
  optional geometry and point support

AnyTouch/tactile tokens:
  optional contact/tactile support

Posterior tokens:
  recurrent anchor identity and temporal memory
```

This is not an LLM-style autoregressive KV cache. It is typed support memory
that AQR anchor queries read through cross-attention.

### 4.2 Anchor Queries

AQR uses learned queries, not weak PaliGemma heatmap points.

```text
physical queries:
  task-neutral-ish observation/posterior anchors
  conditioned by role, coverage, proprio, posterior summary

task queries:
  task/language-conditioned anchors
  conditioned by role, coverage, PaliGemma semantic tokens,
  PaliGemma image-token support, and dense visual/point/tactile/posterior reads
```

The query path is implemented in:

```text
/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py
```

Look for:

```text
aqr_physical_query_tokens
aqr_task_query_tokens
_aqr_pg_image_support_read
_aqr_competitive_support
_build_aqr_anchor_graph
```

### 4.3 Competition and Support Distributions

Each anchor query reads support distributions:

```text
p_visual^k
p_point^k
p_tactile^k
p_posterior^k
```

The competitive support normalization is a Sinkhorn-like row/column competition
step. The point is to reduce "all anchors look at the same place" behavior.

This is not meant to be the only anti-collapse mechanism. Training also uses
support diversity, validity-aware cross-modal matching, VICReg-style
anti-collapse, cycle consistency, and geometry diversity at low weight.

### 4.4 Consumers

AQR emits the existing `PicfAnchorPriorGraphState` interface so the rest of
PICF can consume it without replacing the full control stack.

Consumers include:

```text
observation anchors
task readout
posterior binding/update
conditioned control
PI0.5 action path
```

The action path is intentionally preserved. AQR is the structured spatial
interface, not a replacement for PI0.5.

## 5. PaliGemma Contract

Current production rule:

```text
PaliGemma heatmap/grounding:
  off by default for AQR production.

PaliGemma semantic tokens:
  active; condition task queries.

PaliGemma image tokens:
  active; task queries cross-attend to them and map the resulting support onto
  the V-JEPA visual grid as visual-semantic bias.
```

Important config defaults:

```text
aqr_pg_grounding_enabled = False
aqr_pg_image_support_enabled = True
aqr_pg_image_support_weight = 0.35
aqr_pg_bias_weight = 0.0
```

Do not re-enable PaliGemma heatmap influence unless you are running an explicit
ablation. The previous failure evidence showed near-uniform PaliGemma heatmaps
can inject spatial noise if treated as a strong where source.

## 6. Trainable/Frozen Contract

Current intended profile:

```text
Frozen:
  V-JEPA visual encoder
  Sonata pretrained point encoder
  AnyTouch pretrained tactile encoder

Trainable:
  PaliGemma semantic/action path under the normal semantic trainable profile
  AQR query/router modules
  PICF adapters/readout/posterior/control layers
  graph consumers and auxiliary heads
```

This preserves the previous frozen-perception cost profile while allowing the
new router and action-side PICF components to learn.

## 7. Scaling Judgment

The AQR direction is scaling-friendly because it avoids dense all-to-all
attention across every modality:

```text
dense all-to-all:
  O((V + P + T + M + L)^2)

AQR query-to-memory reads:
  O(K * (V + P + T + M + L)) + O(K^2)
```

where `K` is the number of anchor queries. This is the right direction for
large multimodal robot data, missing modalities, and temporal posterior memory.

However, AQR should not be the only information path. The global semantic path,
dense support memory, posterior memory, and PI0.5 action path must remain
available. Slots/anchors are a structured action-relevant interface, not a full
replacement for global reasoning.

## 8. Known Limitation

Current AQR is not a guaranteed fine-instance solver for cases like:

```text
select one chopstick among four or five very close chopsticks
```

Reason:

```text
V-JEPA default input is 384 with patch_size 16,
so native visual support is about 24 x 24.
```

If several thin objects fall inside the same effective visual patch/receptive
field and point/tactile/posterior evidence is not separable, no anchor mechanism
can reliably invent sub-token instance identity.

The structurally correct next extension is:

```text
coarse AQR anchor
-> local high-resolution crop / point-neighborhood refinement
-> local query-to-support attention
-> local Sinkhorn / support diversity
-> refined support or offset
```

Do not solve this by switching back to global dense all-to-all attention or by
trusting PaliGemma heatmaps. The issue is resolution and local instance
binding, not just model size.

## 9. Current Cloud Run Snapshot

The current live training line at handoff time was launched from the AQR
PaliGemma-image-support code path:

```text
branch: Posterior_VLA
recent commit: 8fdb16f Fix AQR PaliGemma support remap call
cloud repo: /root/openpi_aqr_dbd94d4
tmux session: aqr_8fdb16f_pgimg_train
experiment:
  picf_v22_aqr_8fdb16f_pgimg_noheat_strict2x40_unroll2_30000_ckpt2500_progress_20260509_r1
```

Primary logs:

```text
/mnt/checkpoints/picf_core/train_logs/picf_v22_aqr_8fdb16f_pgimg_noheat_strict2x40_unroll2_30000_ckpt2500_progress_20260509_r1.log

/mnt/checkpoints/picf_core/picf_core/picf_v22_aqr_8fdb16f_pgimg_noheat_strict2x40_unroll2_30000_ckpt2500_progress_20260509_r1/metrics.jsonl
```

Basic watch commands on the cloud machine:

```bash
tmux attach -t aqr_8fdb16f_pgimg_train

tail -f /mnt/checkpoints/picf_core/train_logs/picf_v22_aqr_8fdb16f_pgimg_noheat_strict2x40_unroll2_30000_ckpt2500_progress_20260509_r1.log

tail -f /mnt/checkpoints/picf_core/picf_core/picf_v22_aqr_8fdb16f_pgimg_noheat_strict2x40_unroll2_30000_ckpt2500_progress_20260509_r1/metrics.jsonl
```

Detach from tmux without stopping training:

```text
Ctrl-b d
```

## 10. What Not To Do

Do not treat these as production defaults:

```text
--mapg-enabled true
--vl-anchor-router-enabled true
--aqr-pg-grounding-enabled true
--aqr-pg-bias-weight > 0
```

Those are ablation/diagnostic settings unless the direct-final AQR contract is
intentionally being changed.

Do not turn AQR into a standalone keypoint extractor. The router should remain
inside the PICF path so the same support distributions feed observation, task
readout, posterior, control, losses, and debug.

Do not remove global semantic/dense memory paths and force all information
through a tiny number of anchors. That would turn anchors into an information
bottleneck and hurt global reasoning.

## 11. Main Code Entry Points

Current implementation:

```text
/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py
  AQR config defaults and production PaliGemma contract.

/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py
  AQR query construction, PaliGemma image-token support remap,
  visual/point/tactile/posterior support reads, consumers.

/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py
  MAPG/AQR graph losses: SigLIP-style matching, VICReg, cycle,
  masked modality, routing, support diversity, geometry diversity.

/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py
  CLI flags, startup contract logging, training launch profile.

/home/siyuanyue/Documents/openpi/scripts/serve_picf_policy.py
  Serving/eval entry path.
```

## 12. Acceptance Checks For The Next Engineer

Before claiming success, inspect:

```text
action loss trend
loss_mapg_graph and sub-loss trends
support entropy per modality
same-role support overlap
effective anchor count
posterior identity switches
geometry_valid fraction
raw heatmaps and heatmap-over-RGB overlays
anchor videos
CALVIN 20-trial eval JSON
```

For fine manipulation tasks, also inspect whether anchors bind to:

```text
task object
interaction point
effector/contact region
not just a repeated image corner or one generic saliency basin
```

## 13. Final Handoff Judgment

AQR-MAPG is the maintained direction because it is more coherent than the old
candidate-prior MAPG-v0 path:

```text
semantic/global reasoning remains global
where/instance binding is handled by anchor queries and competition
modality evidence remains typed and optional
posterior memory remains recurrent and explicit
PI0.5 action path remains intact
```

The next major architectural improvement should be local high-resolution
refinement for tiny nearby instances. That should extend AQR, not replace it.

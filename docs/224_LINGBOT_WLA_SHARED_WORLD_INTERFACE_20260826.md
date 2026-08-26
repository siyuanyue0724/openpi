# ADR-224: LingBot + Full WLA Shared World Interface

Date: 2026-08-26 Asia/Shanghai

Status: **CURRENT WLA HYBRID ACTION PATH REJECTED; ANCHOR GEOMETRY REPAIR RETAINED; NO 30K**

## Scientific distance ledger

This ledger is the only progress summary that may be used for model claims.
Environment repair, memory placement, launcher work, source hashing, unit tests,
and finite updates are engineering prerequisites; they are not scientific
progress.

Current accepted scientific results are:

1. The matched host-evidence intervention is causal-valid across training,
   cold evaluation, and causal-warm replay.  This repairs the interpretation
   of the full-versus-masked comparison; it does not improve action.
2. The 200-query bank has strong object-mask capacity on the fixed set
   (oracle binary IoU about `0.817`, oracle Recall@0.5 about `0.910`), while
   autonomous Top-10 recall is only about `0.126`.  The measured failure is
   therefore primarily proposal reading/ranking and action mediation, not the
   absence of fine object candidates.
3. The registered step-0-to-step-20 fixed-set difference-in-differences is
   `+1.147%` for PICF full relative to the matched evidence-masked arm, with
   source-episode bootstrap 95% interval `[+0.479%, +1.619%]` (positive means
   worse).  PICF has no accepted action advantage.
4. On the exact step-1 shared-query surface,
   `cos(g_action, g_source)=+0.001665`; direct negative source/action gradient
   conflict is rejected.  The source gradient norm is about `43.0` times the
   action gradient norm and nearly orthogonal, which registers source-coordinate
   motion as a causal hypothesis, not as a confirmed root cause.
5. The matched source-frozen control rejects that hypothesis as the primary
   action root cause.  Across steps 1--20, freezing changes training action
   loss by only `+0.015%`; the fixed-set learning difference-in-differences is
   `+0.084%`, with 95% interval `[+0.0047%, +0.1381%]` (positive is worse).
   Source freezing, gradient surgery, and source-LR tuning are closed.

Distance to promotion is currently `0/2`: no material whole-curve action lead
and no rollout lead.  Relative to the registered `-5%` fixed-set action gate,
the existing full-versus-masked result is still on the wrong side by about
`6.15` percentage points.  The current WLA hybrid is not "close to long
training" and receives no further scalar tuning.

## 0. Non-negotiable operating law

1. A successful upstream algorithm is copied in full. Topology, layer count,
   token count, attention order, objectives, optimizer groups, preprocessing,
   schedules, and checkpoint semantics are not simplified for convenience.
2. Every change required by LingBot, CALVIN, or PICF is named as an adaptation
   hypothesis. It is never reported as an upstream reproduction.
3. A scientific failure is stopped as soon as the registered effect fails. A
   round-number checkpoint, a selector, or a scalar adjustment cannot rescue a
   failed architecture claim.
4. Promotion requires a material whole-curve and rollout advantage over the
   exact LingBot control. A roughly one-percent action-loss change or a
   regression is a rejection.
5. Semantic interpretation, multimodal binding, temporal correction, world
   state, and action must be learned by a large shared model. Small modules may
   transport or normalize typed evidence; they may not privately decide object
   identity, task relevance, lifecycle, or action.

## 1. Decision

The current 200-row action-routing family is closed. ADR-222 was mechanically
healthy and gave only `+0.819%` aggregate action-loss improvement over updates
1--16, with a materially worse middle window. Earlier controls established
that high oracle mask quality and lower predictive loss do not make the
released LingBot action expert use row identity.

The next architecture uses the complete WLA action/world interface as the
source donor while retaining LingBot-VLA2 as the required semantic backbone.
The core donor is:

```text
World-Language-Action Model for Unified World Modeling,
Language Reasoning, and Action Synthesis
arXiv:2606.05979, source revision dated 2026-08-24
official repository: https://github.com/SJTU-DENG-Lab/WLA
commit: 155ac94eaca8b3d1ae0789ae298fc55e37936081
```

This choice is narrower and more reproducible than reconstructing World Tokens
(arXiv:2608.09730), whose official training source is not published. It also
fixes the precise defect measured in ADR-149/150/152/207/222: the action expert
is jointly trained around the shared latent interface rather than receiving an
alien posterior bank after pretraining.

## 2. What the successful WLA source actually does

The official LIBERO image-action configuration contains all of the following:

- a complete RynnBrain/Qwen3-VL autoregressive vision-language backbone;
- exactly 64 learned meta-query tokens appended inside the language sequence;
- hidden states of those tokens extracted from every backbone layer;
- a complete 28-layer, width-1024, 32-head flow-matching Action Expert;
- layerwise cross-attention from action block `j` to the corresponding late
  backbone meta-query state, with the source's alternating self-attention;
- a complete SANA-600M World Expert and its VAE/noise scheduler;
- current-image features and the same layerwise meta-query states passed
  through the source connector to the World Expert;
- current and historical observations (`use_history_obs: true`, history step 8);
- Beta(1.5, 1.0) flow time sampling, 1,000 timestep buckets, noise scale 0.999;
- action chunk 8 on LIBERO and 32 denoising inference steps;
- AdamW, weight decay `1e-8`, gradient clipping 1.0;
- cosine learning rate `5e-5 -> 5e-6`, with 1,000 warmup updates;
- full joint optimization of the VLM, meta-query embeddings, connector, World
  Expert, and Action Expert; only the VAE is frozen;
- objective `L_action + 0.1 L_world` for LIBERO image-action training;
- 100K official training updates, with the paper reporting strong performance
  by 30K updates.

The paper reports 98.6% average LIBERO success for the complete model and 97.9%
without world loss. The action head is not a probe or small readout: it is a
390M parameter policy trained jointly with a 2.1B backbone and a 600M World
Expert. This is the source-backed mechanism that replaces the failed practice
of feeding a new row bank into LingBot's already-conditioned action interface.

## 3. Source identity and immutable files

The complete source carrier is verified before import.  The active, read-only
tree is:

```text
/mnt/picf-next/adr224/upstreams/WLA-155ac94e-immutable

regular files excluding root .git: 72
total bytes: 617184
canonical full-tree SHA-256:
fa1c9a8857a2280b14eeb2a3864d55825ca7e2411548643ead89ebf8068abc3f
canonical receipt SHA-256:
71c56e3c24c628ae721d978f374a06bc0a10d3a1c08e651dd43ea5f0436e4d59
```

Critical file hashes:

| File | SHA-256 |
|---|---|
| `models/model.py` | `73f2c65f0b6450aeb5f5a4f5687fe0e0afefd0c0200f3fddeb29869658ac7c2f` |
| `models/action_model/action_model.py` | `89c620c9e09d18e09d958eb1dc4fe5877d8290531d1b5a4acaab87465e3ec9d5` |
| `models/action_model/cross_attention_dit.py` | `216a94f32172fe30d5bea07309b7402f9ef543fc115e73be82585ae475edbe74` |
| `configs/libero_all_image_action.yaml` | `af7623ea09aa3596da6c9c1a07e5fe7c4995830e8342eaf0aa731a614888584f` |
| `train.py` | `e5972641ae13dd4efd0faa5643b5afd0afe702eec09025775e90a5d83ab235ed` |
| `models/wla.py` | `17dd50632deee8bf6cd037b875b324d61fd0e49a09481090e75e8b36f51f18d8` |

The integration loads these exact files from the pinned tree. It does not copy
individual equations into a reduced local implementation.

## 4. Target execution graph

Let `X_t` be LingBot's complete current RGB/language token sequence, `E_t` the
set of available typed optional evidence, `P_{t-1}` the persistent PICF object
posterior, `V_t` the complete task-independent VidEoMT object evidence, and
`Q` the 64 WLA meta-query embeddings.

```text
E_t = {V-JEPA2.1, Sonata 3D, AnyTouch2 tactile} when observed

Z_t^0 = [X_t, E_t, P_{t-1}, V_t, Q]
Z_t^l = LingBotLayer_l(Z_t^(l-1)),                  l = 1..36
H_t^l = gather_Q(Z_t^l),                            64 query rows

A_t = WLAActionExpert({H_t^9, ..., H_t^36}, state_t)
Y_t = WLAWorldExpert({H_t^9, ..., H_t^36}, RGB_t)
P_t = posterior_update(P_{t-1}, {Z_t^l}, V_t, E_t)
```

The action branch has no key/value edge to raw RGB, task tokens, Future3D,
VidEoMT rows, or the posterior. It receives visual-language information only
through the 64 layerwise WLA meta-query states. This is the causal property
that ADR-222 lacked at interface pretraining time.

The world branch and action branch share those same states. Therefore, for
trainable parameters `phi` upstream of `H`, both objectives update the same
representation:

```text
grad_phi L = J_H(phi)^T [grad_H L_action + 0.1 grad_H L_world].
```

There is no independent action selector that can ignore the world interface.
There is also no decoder whose private embedding becomes the policy state.

## 5. Object anchors and world tokens are different roles

The old design overloaded one row bank with four incompatible obligations:
fine masks, persistent identity, task selection, and action conditioning. That
made a segmentation-successful statistic compete with an action-sufficient
statistic.

ADR-224 separates roles without separating semantics:

- `V_t`: 200 complete VidEoMT object proposals/masks. They provide fine spatial
  evidence and an auditable anchor visualization. They are not the action API.
- `P_t`: persistent object posterior rows. They carry identity and uncertainty
  across observations. They are evidence tokens inside LingBot, not a private
  lifecycle controller.
- `H_t`: 64 WLA world/action interface tokens. The complete LingBot backbone
  computes task-conditioned combinations of RGB, language, optional modalities,
  current object evidence, and posterior evidence. Action and world objectives
  jointly shape them.

This avoids forcing every uncertain patch into an object. Low-confidence or
unassigned evidence remains context inside the large backbone. Object rows are
used only when the data support a persistent entity hypothesis. Overlap is
represented by multiple object rows attending the same spatial tokens; the
shared host resolves identity using appearance, language, depth, motion,
touch, and history rather than a hard pixel partition.

## 6. Multimodal binding contract

Optional encoders remain complete, frozen pretrained evidence producers:

- V-JEPA2.1 contributes temporal/dynamics tokens;
- Sonata contributes 3D geometry tokens;
- AnyTouch2 contributes tactile tokens only when contact evidence exists;
- RGB/language remain native LingBot tokens.

Each modality uses only a typed width projection/resampler to enter LingBot.
Those transport modules do not score object identity. The complete LingBot
backbone sees modality type, validity, time, camera/sensor origin, object
evidence, and the 64 WLA queries in one attention graph. Binding is implicit in
the shared hidden space and is supervised by available paired observations,
future prediction, action, and object correspondence losses.

Missing modality `m` is marginalized by absence, not replaced by a learned
hallucination:

```text
p(A | observed) = integral p(A, E_missing | E_observed) dE_missing.
```

In code this means invalid optional rows are masked and contribute exact zero;
the shared model is trained with naturally missing modalities and controlled
modality dropout. Tactile tokens are never fabricated when contact is absent.

## 7. Temporal posterior contract

The posterior is an inference state, not an indefinitely backpropagated RNN
graph. At deployment:

```text
P_t = F_theta(stopgrad(P_{t-1}), O_t, E_t, A_{t-1}).
```

Training uses replayed state snapshots and randomized burn-in lengths rather
than differentiating through hundreds of environment steps. This is truncated
state-space training: gradients learn the one-step correction operator while
the state distribution is sampled from real model rollouts. Long-horizon
validity must be measured with 32/64/128-step replay interventions, not inferred
from a 2+2 unroll.

No hard-coded row lifetime is allowed in the production policy. Existence,
uncertainty, occlusion persistence, and release are represented in posterior
token state and learned by the shared LingBot graph. A deterministic validity
mask is permitted only for physically absent measurements and padding.

## 8. Exact donor versus explicit adaptations

### Copied unchanged

- WLA meta-query count and placement semantics (64);
- all 28 Action Expert blocks and their source ordering;
- action encoder, state encoder, timestep encoder, decoder, normalization;
- Beta flow-time distribution and velocity target;
- interleaved action self/cross-attention schedule;
- SANA World Expert, connector, VAE, and noise scheduler;
- full world/action losses and source loss weights;
- optimizer family, schedule, clipping, and warmup;
- history-observation use and the source action/world horizon of 8 for the
  initial CALVIN gate;
- checkpoint and inference denoising semantics.

### Explicit PICF/LingBot adaptations

1. RynnBrain/Qwen3-VL is replaced by the required LingBot-VLA2 VLM backbone.
2. WLA's exact suffix token order is retained:
   `newline, BOI, IMG0..IMG63, EOI, IM_END`. The 66 newly added rows use the
   same Hugging Face mean-resize initializer invoked by WLA. Newline and
   `IM_END` continue to share their existing LingBot vocabulary rows.
3. The last 28 of LingBot's 36 hidden layers replace WLA's last 28 backbone
   hidden layers; no layer is averaged or projected into one final feature.
   Because LingBot's hidden width is 2560 and WLA's donor RynnBrain width is
   2048, the published `cross_attention_dim` configuration field is set to
   2560. The upstream action source and 28-layer topology are unchanged; no
   local bottleneck adapter is inserted.
4. Complete optional modality and posterior evidence is inserted before the
   meta-query rows with typed masks.
5. CALVIN action/state dimensions and normalization replace LIBERO dimensions;
   the flow/action architecture and WLA's source `chunk_size=8` are unchanged.
6. WLA's explicit single history observation at `t-8` is replaced by PICF's
   causal posterior state. The donor does not consume eight consecutive images:
   its dataset selects `history_idx=max(0,t-8)` and its LIBERO evaluator reads
   the oldest observation from an eight-entry deque. This replacement is a
   major PICF hypothesis and requires a control that retains the donor's
   explicit `t-8` image.
7. Persistent object posterior state is an additional input/output contract.
8. The released LingBot action expert is removed, as required by replacing the
   action interface with WLA.  PICF prior-only propagation consequently runs
   through the same complete 36-layer LingBot host used by WLA factual calls;
   it does not retain a second action expert or add a recurrence head.

For the prior-only call, let `B_l` denote base PICF rows and `S_l` the appended
WLA suffix at layer `l`.  The exact WLA suffix mask satisfies
`M(B_l -> S_l)=0`.  Therefore the base recurrence is block triangular:

```text
B_(l+1) = F_l(B_l, Memory_l, Controls_l),
d B_(l+1) / d S_l = 0.
```

Running the complete WLA host path during prior propagation is thus equivalent
on the persisted PICF rows to a host-only pass, while avoiding an unreviewed
shortened implementation.  The suffix is retained for source consistency even
though it adds compute to this mechanical gate.

Each item above requires its own ablation. None is evidence that WLA itself was
reproduced.

## 9. Baselines and scientific gates

Three comparisons are required:

1. `LBOT`: exact released LingBot control on the frozen CALVIN stream.
2. `WLA-LBOT`: LingBot plus the exact WLA action/world interface and the donor's
   explicit `t-8` history image, with PICF rows absent. This measures the
   interface transplant without silently weakening the mature donor.
3. `PICF-WLA-LBOT`: the same model with posterior and all available modalities.
   This measures PICF's incremental value without changing the action head.

The donor baseline history surface is now source-traced rather than inferred.
WLA's LIBERO evaluator constructs exactly
`[agentview_(t-8), agentview_t, wrist_t]`; its dataset uses
`history_idx=max(0,t-8)`. The CALVIN counterpart is therefore
`[static_(t-8), static_t, gripper_t]`, clamped to the first frame of the same
episode. It is one lagged static image plus two current views, not eight input
frames. Any donor-baseline implementation must preserve this ordering and
must reject an episode-boundary or future-frame crossing.

All arms must share sample keys, ranks, seeds, flow times, action noise,
normalization, effective batch, and optimizer schedule. Report every registered
window and the complete curve; never select only a favorable checkpoint.

Mechanical gates grant no scientific credit:

- source hashes and exact class topology;
- 64 query rows and 28 layerwise states;
- no raw visual-language action bypass;
- nonzero action and world gradients into shared query states and LingBot;
- exact-zero gradients from invalid optional modalities;
- no future-target leakage into action-visible keys;
- checkpoint/resume bitwise metadata and optimizer continuity;
- four-rank collective participation and finite bf16 updates.

Scientific promotion gates:

- material action-loss lead over the complete matched curve, not about 1%;
- material CALVIN rollout success lead over LBOT;
- `PICF-WLA-LBOT` materially better than `WLA-LBOT`;
- action changes under source-disjoint posterior/object interventions;
- 32/64/128-step posterior replay remains calibrated under occlusion;
- anchor visualizations show fine objects rather than fixed patch ownership;
- missing-modality and wrong-modality interventions behave monotonically.

## 10. Compute and training policy

The target model is approximately LingBot 6B + WLA Action Expert 390M + WLA
World Expert/VAE about 900M. Four A100-40G GPUs require full parameter/gradient/
optimizer sharding, activation checkpointing, bf16, and microbatch 1 with
gradient accumulation to reproduce the effective batch as closely as possible.
Optimizer or CPU offload is an engineering adaptation and must be recorded; it
must not alter model topology or objectives.

The first valid budget is configured as 30K because the WLA paper explicitly
reports strong LIBERO performance by 30K. It remains early-stoppable. Source
configuration writes scalar curves every 100 steps, anchor/world diagnostics
every 250, and recoverable checkpoints every 2K. Checkpointing at step 1 is
forbidden unless required by a crash probe.

Fresh jointly trained interfaces cannot be judged by a 20-step superiority
requirement: WLA uses 1,000 warmup updates and 100K total updates. Early runs may
be stopped for mechanical failure, leakage, divergence, or a decisive matched
regression, but absence of a 20-step advantage alone is not a source-supported
scientific rejection.

## 11. Implementation checklist

- [x] Reject ADR-222 and persist the exact whole-window evidence.
- [x] Correct the ADR-149 historical narrative with its complete curve.
- [x] Pin and archive the complete WLA source tree.
- [x] Record immutable critical-file hashes and official LIBERO config.
- [x] Add a 72-file full-tree hash validator and enforce it before WLA import.
- [x] Add a LingBot host-only 36-layer execution path that records exactly 64
      query states at every layer.
- [x] Call the exact WLA 28-layer Action Expert on LingBot layers 9--36.
- [x] Call the exact WLA World Expert and source connector/objective.
- [x] Insert complete RGB/language/V-JEPA/Sonata/AnyTouch/VidEoMT/posterior
      evidence without an action bypass.
- [x] Retain the existing bounded-gradient posterior replay contract without
      long BPTT; long-horizon calibration remains a scientific gate.
- [x] Expose every differentiable WLA root result through a PyTorch-native
      pytree so FSDP2 can unshard root-owned parameters before backward.
- [ ] Complete shape, mask, gradient, causality, resume, and source-parity
      probes.
- [x] Complete a four-GPU one-update full-weight mechanics gate.
- [x] Complete a four-GPU two-update gate after optimizer-state materialization.
- [x] Implement a fail-closed, same-sample/same-randomness block-bootstrap
      comparator for the two WLA host-evidence arms.
- [ ] Run matched WLA-LBOT and PICF-WLA-LBOT curves.
- [ ] Authorize 30K only after material action and rollout evidence.

## 12. Current honest score

- theory/source specification: `8.5/10`;
- implementation maturity: `8/10` for ADR-224 (the complete action/world
  donor, full multimodal host, source identity, prior/correction host ownership,
  optimizer and FSDP topology are implemented and two real optimizer updates
  pass; inference/resume, controls, and scientific curves remain open);
- deployment maturity: `5/10` for ADR-224 because two complete four-rank updates
  now pass after optimizer-state materialization, but inference/resume and
  matched scientific controls remain open;
- authorization for 30K: `NO`.

The theory score is not 10/10 because retaining LingBot changes the WLA
backbone and persistent object state is a new PICF hypothesis. No amount of
documentation can turn those adaptations into an already-proven theorem. The
score can rise only after source-parity mechanics, matched ablations, and
rollout evidence.

## 12.1 Scientific-progress accounting

Engineering closure and scientific progress are reported separately.

- The ADR-224 `v17` run completed two full four-rank forward/backward/optimizer
  updates. This closes the post-AdamW-state OOM that prevented an experiment;
  it is not evidence of an action advantage.
- The newest architecture has produced zero matched scientific action curves.
  Therefore its measured distance to an action lead is still `0/2` required
  same-interface curves: `WLA-LBOT masked` and `PICF-WLA-LBOT full`.
- Historical ADR-149 and ADR-222 curves remain negative evidence and cannot be
  relabelled as support for the new WLA architecture.
- `tools/compare_adr224_wla_host_evidence_curves.py` validates identical model
  family, implementation, parameter manifest, stream, samples, augmentation,
  flow noise/timesteps, WLA future targets and optimizer contract. It reports
  complete and registered-window paired action-loss ratios with moving-block
  bootstrap intervals. Its output is explicitly training-curve evidence only
  and always sets `authorizes_long_train=false`; held-out action and CALVIN
  rollout gates remain mandatory.

The same-interface control is an input intervention, not another small model:

1. Both arms retain the complete LingBot host, all 200 rows, complete WLA
   action/world experts, complete trainable VidEoMT source objective, identical
   parameters, optimizer groups, schedule, stream, seeds and targets.
2. `wla_lbot_masked` zeroes the current VidEoMT query features before their
   host projection, invalidates AnyTouch/Sonata/V-JEPA/proprioception rows before
   the first shared-host layer, and zeroes only the previous LingBot posterior
   rows. The factual VidEoMT source recurrence and source objective continue.
3. `picf_full` restores only the current multimodal object evidence and previous
   host posterior. Hence a paired action-curve difference estimates the value
   of PICF information under one fixed action/world interface.
4. The object-query spatial relation is retained only for source accounting and
   visualization in the masked arm; it has no hidden-state path once query
   features are zero and relation losses are disabled.
5. The same masking function now owns training, cold held-out evaluation, and
   four-frame causal-warm evaluation. In the masked warm arm, factual VidEoMT
   source recurrence remains active while every previous host posterior input
   is replaced by an equal-shaped zero state. This removes a discovered
   train/eval confound in which both evaluated arms would otherwise receive
   full PICF evidence.

This same-interface masked control does not establish that WLA itself was
reproduced, because it omits WLA's explicit `t-8` history image. Nor may its
result be compared numerically to released LingBot loss when the loss scales
differ. Released LingBot comparison requires a common normalized action metric
and CALVIN rollout success; direct training-loss comparison is valid only
between the two WLA arms.

The held-out evaluator's legacy `official_action_loss` field name is not its
backend identity. Complete-WLA installation removes the released LingBot
action expert and `_run_native_policy_forward` dispatches through
`policy.picf_wla_calvin_forward`; the snapshot therefore measures WLA's action
objective. The evaluator now fails closed unless its recorded backend is
`wla_complete`. WLA's untouched 32-step `ActionHead.predict_action` sampler is
already installed as `policy.sample_actions`, but normalized sampled-action
metrics and closed-loop CALVIN rollout remain separate open gates.

## 13. Live implementation evidence

At 2026-08-26 05:44 CST, the untouched pinned WLA Action Expert passed a CUDA
forward/backward source probe:

- `389,574,663` trainable parameters;
- 28 source transformer blocks and 64 query rows at width 2048;
- source-faithful alternating conditioning at blocks
  `0,2,4,...,26`, with exact-zero query gradients at interleaved self-attention
  blocks;
- nonzero Action Expert and query-state gradients;
- peak allocated CUDA memory `1,650,834,432` bytes for the batch-1 primitive.

The LingBot adapter appends the complete source suffix role sequence
`[NEWLINE, BOI, Q_1..Q_64, EOI, IM_END]`. The existing LingBot prefix retains its
released mask. Prefix rows cannot read the appended suffix, while each suffix
row reads all valid prefix rows and its causal suffix prefix. Position IDs
continue from the maximum valid (not padded) prefix position. The adapter
captures all 36 post-layer query states, replaces the final unnormalized state
with the final Qwen3-VL normalized state exactly as Transformers' hidden-state
recorder does, then selects the final 28 states. Meta-token initialization
calls the same Hugging Face mean-resizing primitive invoked by WLA's
`resize_token_embeddings` for the 66 added vocabulary rows; existing newline
and `IM_END` embeddings stay tied to LingBot's shared embedding table.

At 2026-08-26 06:08 CST, the real released LingBot checkpoint passed the
integrated two-GPU source probe:

- all 36 LingBot host layers executed and exposed the final 28 query states;
- every state had shape `[1,64,2560]`;
- a one-token prompt intervention changed the final query state by mean
  absolute value `0.1168815`;
- the adapted source Action Expert had `404,254,727` parameters;
- query gradients were nonzero exactly at source conditioning blocks
  `0,2,...,26` and exactly zero at source self-attention blocks;
- host/action peak allocations were `13,620,946,944` and `1,938,147,328`
  bytes respectively;
- receipt: `/mnt/picf-next/adr224/evidence/lingbot_wla_real_host_probe.json`.

At 2026-08-26 09:28 CST, four-rank `v7` loaded the complete LingBot, WLA Action
Expert, SANA World Expert, connector, full modalities, optimizer, and FSDP2
topology, then failed before the first loss because the PICF prior root still
called the deliberately removed LingBot `qwen_expert`.  No optimizer step and
no scientific metric were produced.  This was an interface-ownership defect,
not evidence for or against PICF.  The repair moves prior propagation onto the
same complete 36-layer host and explicitly forbids reviving the old expert;
isolated source tests pass.

At 2026-08-26 09:51 CST, four-rank `v8` completed the full prior, factual WLA
action, WLA world, and joint-objective forward, but all ranks failed at the
first backward with a zero-storage access to the World Expert's `[64,1152]`
output projection.  It produced no update and no scientific metric.  PyTorch
2.8's composable-FSDP implementation establishes the exact cause:
`FSDPState._register_pre_backward_hook` discovers differentiable root outputs
with `torch.utils._pytree.tree_flatten`; the registered WLA root returned a
plain dataclass, which pytree treated as an opaque leaf.  The root therefore
resharded its parameters after forward without installing a pre-backward
unshard hook.  The repair changes only that typed return container to a
`NamedTuple`, preserving field names, tensor identities, losses, topology,
optimizer, and all donor computation.  A contract test verifies that total,
action, world, and PICF native-root tensors are all visible to pytree; the
focused suite passes `13/13`.

At 2026-08-26 10:12 CST, four-rank `v9` proved that the corrected root ABI
reaches the real joint backward, replacing the `v8` zero-storage failure.  It
then failed before any update when rank 3 needed another `1.08 GiB` during the
FlexAttention backward: the rank already used `39.05 GiB` of its `39.49 GiB`
device, with `36.219 GiB` reserved before backward.  This is a zero-update
memory-placement failure and carries no scientific credit.  `v10` never loaded
the model: the source validator correctly rejected the older five-overlay
checkout because it lacked the already-authored trainable-vision and
mixed-device Muon overlays.  It therefore produced no mechanical or scientific
evidence.  An isolated six-overlay source checkout was then constructed by
replaying the exact pre-existing patch chain and passed the complete source
validator and frozen-file hashes.  `v11` uses that checkout and changes only the
existing PyTorch FSDP2 placement to
`selective-embedding-trainable-vision-offload`.  The 24 trainable Qwen vision
blocks, their gradients, and optimizer shards move between CPU and CUDA under
`CPUOffloadPolicy`; model topology, trainability, tokens, all modalities,
action/world losses, and optimizer mathematics remain unchanged.  This
placement is an explicit engineering adaptation and must be identical across
matched scientific arms.  At 10:37 CST, `v11` completed the full action, world,
PICF and VidEoMT objective but again failed during the first FlexAttention
backward before any update: rank 3 needed `1.08 GiB` with `1.01 GiB` free and
`38.48/39.49 GiB` in use.  Trainable-vision offload lowered rank 3's
pre-backward reservation from `36.219` to `35.768 GiB`, but did not clear the
backward peak.  This is a rejected zero-update placement result, not a reason to
change model topology or loss weights.  `v12` therefore tested PyTorch's complete
FSDP2 `CPUOffloadPolicy` with the same graph, trainability, objective, stream and
four ranks.  At 10:55 CST it failed during the first objective forward, before
backward or any update: the complete root offload left a WLA connector
gradient-checkpoint recomputation with CUDA hidden states and a CPU RMSNorm
weight.  All ranks failed at the same upstream `TransformerEncoderLayer`
boundary.  This is another zero-update engineering rejection and has no
scientific value.  It specifically rejects full-root CPU offload for this
composition; it does not justify changing WLA/PICF mathematics or reporting a
learning result.  The next admissible memory-placement experiment is the
already implemented PyTorch FSDP2 selective-class `CPUOffloadPolicy`, limited
to exact WLA action/world transformer block classes while preserving the
connector and shared-host device boundary.  Every matched scientific arm must
use the identical accepted placement.

At 11:04 CST, focused source and placement-contract tests passed `173/173` for
that selective policy.  Mechanical run `v13` applies it to the donor's exact 28
`BasicTransformerBlock` action units and exact 28 `SanaTransformerBlock` world
units, plus the already accepted trainable-vision placement.  It deliberately
does not offload the Qwen2 connector layer that exposed the full-root
checkpoint boundary mismatch in `v12`.  This transformation changes only the
storage location of FSDP parameter, gradient and optimizer shards between
uses: forward functions, tensors, graph edges, horizons, objective weights,
trainability and optimizer equations are unchanged.  A mechanical pass must
still demonstrate a real finite nonzero-gradient optimizer update; it grants
no scientific credit by itself.

`v13` was rejected before its first training step.  The WLA runner requested
the selective-class policy, but the prepared LingBot checkout contained the
trainable-vision overlay without the already frozen combined selective-class
overlay.  The storage validator detected the missing topology after model
load; no objective, backward or update ran.  The validator dispatch had
required the combined overlay only for the older ADR-221 WSA profile, not for
the new complete-WLA backend.  That dispatch condition now covers both
profiles.  A new isolated checkout was produced by applying the existing
`lingbot_fsdp2_selective_class_after_trainable_vision_offload.patch` verbatim,
with SHA-256
`7035b239f6c94ae58ac7bd66969ce9b2a5b1676ce105ed95e36e8597f11734d4`.
The immutable-commit plus seven-ordered-patch verifier passed, as did the
focused `173/173` test suite.  Mechanical run `v14` uses that verified source;
`v13` contributes no scientific evidence.

`v14` completed the entire first forward, backward and optimizer transaction
on all four ranks in about `50.04 s`, proving that the root ABI and selective
WLA block offload can execute a real update.  Its pre-backward reservation was
`29.463--34.887 GiB`, safely below the earlier `v11` peak.  It nevertheless
failed during the second action forward: materializing AdamW first/second
moments after step one raised rank 3 to `39.35/39.49 GiB`, leaving only
`139.56 MiB` before a `276 MiB` FlexAttention allocation.  The run was stopped
immediately.  One completed step is useful mechanical localization but is not
a stable training result and receives no scientific credit.  The manifest
shows that WLA action/world blocks and trainable vision held CPU shards while
the 36-layer shared Qwen text host remained CUDA-resident.  `v15` therefore
extends the same existing selective-class `CPUOffloadPolicy` to the exact
`Qwen3VLTextDecoderLayer` units.  Connector and non-block root parameters stay
on CUDA; graph, losses, trainability, optimizer and all donor code remain
unchanged.  Its acceptance criterion is completion of step two after optimizer
state materialization, not merely another first update.

`v15` was rejected by the parameter-placement validator before its first
forward. Source inspection found a precise execution-layer cause: LingBot first
applied selective-class offload through its generic module traversal, then its
explicit Qwen-VL text/vision FSDP loop wrapped the same text blocks a second
time with `mp_fsdp_kwargs`, omitting the existing selective-class offload
policy. Thus the requested text placement was not present in the final FSDP
topology. `v16` was a diagnostic launch only; it was stopped immediately after
the source-level cause became decisive, so it produced no update and no
scientific evidence.

ADR-224 adds one execution-only eighth overlay at that explicit VLM dispatch.
It reuses the already approved PyTorch `CPUOffloadPolicy` and chooses between
the existing per-kind kwargs and a copied kwargs mapping containing that
policy based on the exact block class name. It does not alter layer functions,
weights, trainability, tensors, objectives, horizons, data, or optimizer
equations. The patch SHA-256 is
`7634367ee5dbfe08161c405a25a4e44014d2fdf3a9bc6ecb6cef5331840a93c9`;
the resulting `torch_parallelize.py` SHA-256 is
`a02db3cfa6fcad6a4704e3f6efdda29d530f8aa1c473cfc889831ee71e86e635`.
The validator rebuilds all eight ordered overlays from immutable LingBot commit
`2838c1862bbec1ea47942fb61512130f635eb595` and reverse-checks the final
checkout. The focused suite passes `176/176` with one optional CALVIN checkout
test skipped. Mechanical run `v17` completed two full
forward/backward/optimizer updates on all four ranks. Step one took `82.03 s`;
step two, after AdamW state materialization, took `55.44 s`. The maximum
step-two reservation was `33.404 GiB`, rather than the rejected `39.49 GiB`
peak. All complete source, WLA action, WLA world and shared-host gradients were
finite. This accepts the execution-only placement and closes the
post-optimizer-state OOM. It does not measure action learning and therefore
carries no scientific credit.

Mechanical run `v18` exercises the same complete graph with the registered
`wla_lbot_masked` evidence intervention. Its only purpose is to prove that the
matched control can execute after optimizer-state materialization before paid
curve acquisition begins. Unit and contract tests for this intervention and
the curve comparator pass; a `v18` mechanical result still carries no
scientific credit.

Before 13:00 CST, `v18` completed two full forward/backward/AdamW updates. Step one
took `83.41 s` and step two `55.57 s`; the maximum step-two reservation was
`33.424 GiB`. This accepts only the masked arm's execution mechanics and still
provides no evidence that PICF improves action learning.

The first registered 100-step masked acquisition was rejected before step zero.
Its fixed held-out plan contains language-transition keys, while two data
boundaries assumed physical-event keys. Timestamp validation was first rewritten
to resolve the runtime and authenticated cache key to the same immutable
`source_global_index`, reject an identity mismatch, and derive observation time
from the raw frame's source episode. This is an algebraically equivalent identity
normalization, not the direct root cause: a rerun failed with the same aggregated
message. A data-only reproduction then located the actual blocker in WLA
world-target construction. That boundary now accepts immutable source indices
and defines the target exactly as raw-episode frame `t+horizon`, rejecting
episode-reset crossings. Training physical keys and held-out language keys
therefore share one target definition rather than a string conversion. Both
attempts failed before evaluation or optimizer step zero and add no model
evidence.

A third masked acquisition passed all fixed held-out data preparation but was
rejected before its first model forward. The diagnostic state-preservation code
still required the released LingBot action-MoE counters, although complete WLA
deliberately removes that action expert. Source tracing found no forward-mutated
buffer in the pinned WLA action/world modules; WLA stochastic state is already
paired through explicit RNG capture and replay. The state contract is now
backend-tagged: `lingbot_released` must still expose and restore its MoE counters,
whereas `wla_complete` must expose none of those legacy counters and restore a
typed empty snapshot. Backend mismatch or unexpected counter presence fails
closed. The focused suite passes `159/159`. This is an evaluator-backend validity
repair only; it changes no model tensor, loss, optimizer, stream, or action
capability and therefore receives no scientific credit.

Scientific accounting after this repair is therefore unchanged: zero new
matched action curves and zero demonstrated action advantage. The acquisition
is adaptive but preregistered by information value: compare both arms first over
steps `1--20` and fixed snapshots `0/20`; only a material non-regression can
justify extending both to windows `21--50`, `51--100`, and `1--100`. Paired
moving-block and episode-bootstrap intervals are mandatory. A single endpoint,
a finite loss, or a successful update is not scientific progress. Even a
training-curve lead cannot authorize 30k: native 32-step sampled-action
evaluation and CALVIN closed-loop rollout remain mandatory.

The scientific repair ledger must not be merged with the action-result ledger:

| Question | Substantive repair or finding | Evidence status | Distance closed toward action superiority |
|---|---|---|---|
| Why did structural learning fail to improve action? | Historical interventions identified both a released-action bypass and a conditioning-interface mismatch: useful posterior structure was neither necessary nor interface-pretrained for action. ADR-224 replaces that boundary with the complete WLA action/world suffix conditioned by one shared LingBot host state. | Architecture-level causal hypothesis; not yet capability evidence. | `0` action gates. |
| Can the source represent small CALVIN objects at all? | The complete 200-query VidEoMT bank has high oracle geometric coverage, including small blocks and controls. | Accepted geometric-capacity evidence; autonomous query usefulness remains unproved. | `0` action gates; this rules out source mask capacity as the sole root cause. |
| Did the matched ablation measure the intended information? | One mask now owns training, cold heldout, and causal-warm heldout; the masked arm cannot regain multimodal evidence or host posterior during evaluation. | Accepted causal-validity repair. | Enables the first interpretable action gate but does not pass it. |
| Does PICF evidence improve action? | The paired `picf_full` versus `wla_lbot_masked` fixed-set DiD is `+1.147%` (worse), with 95% interval `[+0.479%, +1.619%]`. | Current interface rejected at step 20. | `0` action gates. |
| Does the complete PICF architecture beat a deployable baseline? | The current run replaces the initialized LingBot action expert with an uninitialized adapted WLA expert, so it is not a deployable absolute baseline. | Not established; current hybrid closed. | `0` action gates. |

Accordingly, ADR-224 has moved from an invalid or bypassed test to an
interpretable test, not from parity to demonstrated action superiority. Any
status report must state both facts separately.

Progress is counted in three separate ledgers. Engineering execution evidence
(loading, memory placement, finite steps, checkpointing) never counts as model
progress. Causal-validity repairs count only when they change whether a planned
comparison answers its stated intervention. Action progress begins only with a
strictly matched whole curve or held-out/rollout result. Under that accounting,
the train/evaluation-consistent host-evidence mask is a scientific validity
repair.  The later matched curve contributes a negative action-capability
result: the current interface does not improve action.  It still passes zero
action gates.

The first valid step-zero masked snapshot also resolves an anchor-capacity
question without claiming action advantage. Across 102 fixed samples, oracle
Hungarian matching over all 200 VidEoMT queries gives held-out mean binary IoU
`0.815913` and recall@0.5 `0.908805`. Foreground-ranked proposal recall is only
`0.124078` at 10 queries, then `0.660907`, `0.880270`, and `0.909600` at 50,
100, and 200 queries. Direct image inspection confirms precise masks for small
blocks, buttons, switches and links even when the released class head assigns
foreground probability around `0.001--0.013`. The source has strong geometric
capacity, while its closed-taxonomy objectness is poorly calibrated on CALVIN.

This ranking defect is not an action gate in ADR-224. The exact Stage-PQM
boundary marks every canonical query valid and sends all 200 query tokens into
the complete 36-layer host; no Top-k selection or threshold exists. Under
`native_videomt_query_posterior`, the class and mask logits are attached only as
a same-index sidecar after the final host layer. They are root-visible for
auditing and supervision but are not fed back into WLA action computation.
Consequently, poor donor objectness cannot by itself explain action loss and
does not justify adding a selector head. The open scientific question is whether
the shared large host and WLA suffix learn to use the complete query bank; only
the matched full-versus-masked action curves can answer it.

The first masked step-20 snapshot provides a negative single-arm result. On the
same fixed samples, heldout action loss changes from `1.370979` at step zero to
`1.557502` at step 20. The paired geometric change is `+12.027%` with a
source-episode bootstrap 95% interval of `[+9.079%, +20.727%]`; 45 of 68
heldout samples worsen. Across all 102 samples the corresponding change is
`+10.908%`, interval `[+7.227%, +17.244%]`. Meanwhile heldout anchor oracle IoU
remains stable (`0.815913 -> 0.817113`) and recall@0.5 remains stable
(`0.908805 -> 0.910377`). This is direct evidence that geometric source quality
and action generalization remain decoupled in the masked arm. It is not evidence
against PICF because the full arm has not yet been compared.

Continuing that single arm to step 100 could not answer the intervention, so it
was interrupted after step 25 while retaining the complete step `0/20`
snapshots and first 20 rank-paired records. The unchanged-source `picf_full`
arm is now acquired to step 20. If it does not materially improve the paired
curve and heldout degradation, the interface does not pass the early gate. If
it does, both arms may be extended; neither outcome substitutes for the
explicit-`t-8` donor baseline.

The adapter remains an explicit LingBot adaptation, not a claim of bitwise WLA
reproduction. Scientific authorization remains `NO` until matched controls and
whole-curve evidence pass.

## Completed step-20 scientific decision

The matched acquisition is complete through step 20 for both registered arms.
The result rejects the current shared-interface hypothesis at this gate.  It
does not authorize an extension to step 100, 2k, or 30k.

The strictly paired training stream initially favors `picf_full`, but the lead
is small and does not persist monotonically:

| Window | Full mean | Masked mean | Paired geometric PICF improvement |
|---|---:|---:|---:|
| `1--5` | `1.353516` | `1.360547` | `0.501%` |
| `6--10` | `1.294531` | `1.313281` | `1.468%` |
| `11--15` | `1.245703` | `1.287500` | `3.266%` |
| `16--20` | `1.233203` | `1.244922` | `0.789%` |
| `1--20` | `1.281738` | `1.301563` | `1.512%` |

This is below the preregistered materiality threshold and, more importantly,
does not transfer to the fixed evaluation set.  From step zero to step 20, the
fixed-set action loss worsens in both arms.  After subtracting the arms' small
step-zero difference, `picf_full` learns relatively worse than
`wla_lbot_masked` by `1.147%`; the source-episode bootstrap 95% interval is
`[0.479%, 1.619%]`.  The heldout and validation partitions agree in sign.
The endpoint comparison at step 20 is also worse for full evidence by `0.663%`
overall.  Therefore the lower online loss is sample-stream fitting, not an
action-generalization advantage.

Durable evidence:

- `matched-picf-full-vs-wla-lbot-masked-step1-20-training-curve-v1.json`,
  SHA-256 `7bb9c685d3912fdcfeab57ecb6b3075984d17d0f22844000de1e1b5be1a626a1`.
- `matched-picf-full-vs-wla-lbot-masked-step0-20-heldout-did-v1.json`,
  SHA-256 `6886c10a22adce8382916c930016528db2f54927d1d50ee6698de3c96e215c1e`.
- `matched-step20-anchor-action-coupling-diagnostic-v1.json`, SHA-256
  `ab2eb262bf2d0616fe06f41efde8e6856fdbe0600ac3070bfcbb68d64ace19cf`.

### What was scientifically repaired

1. The same host-evidence intervention now applies in training, cold fixed
   evaluation, and causal-warm fixed evaluation.  Before this repair, the
   masked arm recovered full PICF evidence during evaluation and could not
   answer the intended causal question.
2. A fixed step-zero/step-20 difference-in-differences estimator now removes
   the small initialization-level difference between arms.  Online loss can no
   longer be mistaken for heldout learning.
3. Direct mask inspection and all-query oracle matching separate source
   geometric capacity from autonomous action addressability.  The complete
   query bank has heldout oracle IoU about `0.816` and recall@0.5 about `0.91`,
   while foreground-ranked Top10 recall is only about `0.12`.
4. Source-query identity stability is now measured rather than inferred.  Only
   `60/102` fixed samples keep the entire oracle-matched query tuple from step
   zero to step 20; `42/102` change at least one matched address.

These repairs make the negative action result interpretable.  They do not
constitute movement toward an action lead.  Under the scientific ledger, the
distance closed toward action superiority in this acquisition is **zero action
gates**.

### Scientific progress ledger, not an engineering progress ledger

The acquisition changed scientific knowledge in exactly three ways:

1. **Resolved:** the complete VidEoMT bank has sufficient geometric capacity
   for the small CALVIN objects.  A lack of object-shaped masks is no longer an
   admissible primary explanation for the action result.
2. **Rejected:** host reachability is not object-level multimodal binding.
   AnyTouch, Sonata, V-JEPA and proprioception are inserted as global typed
   sensor-token streams.  They have no factual object-row assignment in this
   run.  Every posterior row and every WLA metaquery can read the global sensor
   surface, so the large host can use a modality without binding it to the
   corresponding VidEoMT object query.
3. **Rejected at the registered early gate:** the current jointly moving query
   coordinate does not improve fixed-set action learning.  The statistically
   significant `+1.147%` difference-in-differences is in the wrong direction.

Installing source code, making FSDP run, eliminating evaluation contamination,
and producing a finite loss are necessary mechanics.  They are not counted as
movement toward action superiority.  The only accepted movement toward that
goal is a material, matched, fixed-set whole-curve lead followed by sampled
action and closed-loop rollout evidence.  On that scale ADR-224 remains at
`0/2` action gates.

### Root-cause constraints established by the result

The data reject a pure mask-capacity explanation.  High-quality object masks
exist, remain stable in aggregate, and do not correlate materially with the
relative action-learning effect: oracle-IoU Pearson correlation is about
`0.10`, and Top10-recall Pearson correlation is about `0.003`.  Better source
geometry therefore does not imply a better action coordinate.

The current implementation sends all 200 source query tokens into the shared
36-layer host.  The class/mask tensors are an audit and supervision sidecar,
not an action input, and no Top-k selector is present.  Nevertheless the
registered run has no explicit relation-supervision depth, no entity loss, no
predictive loss, and empty factual row bindings.  Same-index source rows are
projected into host rows, but action generalization is expected to emerge only
from the complete action/world loss.  The experiment shows that this expectation
is insufficient at the tested interface and schedule.

The WLA action path does not directly read LingBot sensor tokens, but this does
not force posterior mediation.  Its 64 appended metaqueries read the complete
host prefix, including global sensor streams and posterior rows, at every host
layer.  The 28-layer action expert then reads those metaqueries.  Consequently
the effective graph is

```text
global sensors ------> WLA metaqueries ------> action
       \--------------> posterior -----------/
```

rather than the PICF-identifying graph `evidence -> object posterior -> action`.
This is a large-model shortcut, not a small-head bug, and explains why physical
modality presence and finite gradients cannot establish object-level fusion.

The comparison also starts from a newly instantiated WLA action expert.  The
pinned upstream constructs `LayerwiseFlowmatchingActionHead(...)` inside a fresh
model and trains it jointly for up to 100K updates; only an explicit resume
loads a trained WLA checkpoint.  ADR-224 likewise instantiates the complete
390M action topology from source but has no robot-policy WLA checkpoint.  Thus
the 20-step full-versus-masked comparison is a valid early causal test of PICF
information at one common random action interface.  It is not a calibrated
absolute comparison against LingBot's pretrained released action expert, and
it cannot by itself establish final-policy superiority.

There is also a concrete optimization-timescale mismatch.  At step 20 the
complete WLA/LingBot optimizer is still at `1e-6` under its 1000-step warmup,
whereas the non-backbone VidEoMT groups are at `4e-6` under the released
500-step warmup.  The measured source pre-clip gradient norm is about `45.2`,
versus about `6.38` for the complete host.  Thus the query coordinate can move
faster than the large action reader that must learn it.  This is a supported
hypothesis for the observed online-fit/fixed-set split, not yet a causal proof.
Any schedule repair must be sourced from a mature upstream staged-training
procedure and tested against a frozen-coordinate control; an invented selector
or lifecycle head is prohibited.

### Remaining valid scientific path

1. Add the missing donor-faithful control.  Official WLA uses exactly one
   lagged observation at `t-8`, not an eight-frame sequence: the training
   dataset selects `max(0, t-8)`, and LIBERO evaluation passes the oldest item
   in an eight-entry deque.  The corresponding CALVIN control is
   `[static_(t-8), static_t, gripper_t]`, clamped within an episode, with no
   PICF rows.  `wla_lbot_masked` is an information ablation with 200 generic
   placeholders and is not this deployable baseline.
2. Before another paid curve, decompose the source gradient from the source
   segmentation objective and the action/world objective at a fixed batch.
   Measure cosine conflict and per-group update-to-weight ratios.  This tests
   the nonstationary-coordinate hypothesis directly.
3. Compare a mature staged/frozen source schedule against the unchanged joint
   schedule while preserving the exact 200-query bank and complete large host.
   No custom selector, threshold, lifecycle rule, or reduced donor topology is
   admissible.
4. Require a material fixed-set whole-curve lead over both the same-interface
   ablation and the donor-faithful `t-8` baseline before sampled-action and
   closed-loop CALVIN gates.  Until then, action superiority remains
   unestablished and long training remains prohibited.

### Preregistered shared-query gradient diagnosis

The next measurement is diagnostic, not a training result.  It differentiates
the unchanged source, action, world, and host losses to the exact final
prediction-query tensor consumed by VidEoMT's released class/mask heads and,
through a view, by the LingBot posterior projection.  It adds no learned head,
loss, optimizer term, selector, or lifecycle rule.  Across all four ranks it
reduces only the sufficient Gram statistics

\[
  G_{ij}=\langle \nabla_z L_i,\nabla_z L_j\rangle,
  \qquad i,j\in\{\mathrm{source},\mathrm{action},\mathrm{world},
  \mathrm{host}\},
\]

where `host` is the unchanged WLA host objective and \(z\) is the shared
VidEoMT query surface.  For an infinitesimal joint update directly in this
coordinate, the first-order action change has sign

\[
  \Delta L_{action}\simeq
  -\eta\,\langle g_{action},g_{host}+g_{source}\rangle.
\]

The result is interpreted before any schedule is changed:

- a negative action-versus-joint dot product, together with a source gradient
  materially larger than the action gradient, supports direct coordinate-level
  interference and authorizes a matched frozen-source schedule control;
- a nonnegative dot product rejects direct conflict on this batch.  It does not
  prove parameter-space compatibility because the upstream Jacobian changes the
  metric, so it cannot by itself authorize either joint training or a freeze;
- either outcome closes zero action gates.  Only the subsequent matched
  frozen/joint curve and donor-faithful `t-8` control can establish action
  progress.

The staged-control hypothesis has upstream code support rather than an invented
PICF mechanism.  NVIDIA Isaac-GR00T commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1` defaults to
`tune_llm=false`, `tune_visual=false`, `tune_projector=true`, and
`tune_diffusion_model=true`, and explicitly keeps frozen modules in evaluation
mode.  APT commit `d875861d93071d3bb627ca85166fe2723e923960`
implements a stricter two-stage action-expert procedure: a frozen-Qwen3-VL
vision-action prior is trained first, then complete interleaved language layers
are inserted and loaded through the repository's `load_from_va` path.  APT is
useful independent 2026 evidence but is not treated as a commercial-grade donor
or copied into PICF merely because its explanation matches the observed
failure.  Any adoption still requires source-level parity review and a matched
causal control.

### Shared-query result and scientific decision

The registered four-GPU diagnosis completed at step 1 on the unchanged full
arm.  All ranks reduced the same Gram matrix over `819,200` local query-surface
elements.  The durable artifact is
`shared-query-gradient-step1-v1.json`, SHA-256
`ad6233bde8b08f99437609f44ddb29b8eee69fda464063b152eb829ee28162b8`.

| Quantity | Value |
|---|---:|
| `||g_action||` | `0.00760177` |
| `||g_host||` | `0.00751620` |
| `||g_source||` | `0.32684368` |
| `||g_world||` | `0.00375702` |
| `cos(action, host)` | `0.99874372` |
| `cos(action, source)` | `0.00166475` |
| `cos(action, world)` | `-0.22136491` |
| `||g_source|| / ||g_action||` | `42.9957` |
| `cos(action, host + source)` | `0.02462479` |

This result rejects direct source-versus-action sign conflict on the measured
batch: the source dot product is small but positive, and
`<g_action, g_host + g_source> = 6.12008e-5 > 0`.  Gradient surgery, PCGrad,
or clipping away a negatively aligned source component therefore has no
evidential basis.

The result instead supports a scale-separated moving-coordinate failure.  The
host gradient is almost exactly action-aligned, while the source gradient is
about 43 times larger and nearly orthogonal.  After adding source to host, the
joint surface direction has only `0.0246` cosine with action; its squared
action-aligned directional fraction is about `0.000606`.  The actual step also
reports a source-model pre-clip norm of `72.286` and a host pre-clip norm of
`8.918`; both optimizers clip at `1.0`.  Thus the source optimizer can spend
most of its bounded update moving the object coordinate for segmentation while
the randomly initialized action reader is still learning that coordinate.

This is not yet a parameter-space causal proof.  If the source Jacobian is
\(J\), the parameter-space dot product is governed by
\(g_a^T J J^T g_s\), not the Euclidean surface dot product measured above.
The result therefore narrows the next experiment rather than authorizing a
schedule by assertion:

1. reject gradient-conflict surgery;
2. compare the unchanged joint schedule with a source-frozen, projector/action
   training stage on the exact same stream and fixed evaluation set;
3. compare both against the donor-faithful explicit-`t-8` WLA control;
4. count progress only if a complete fixed-set curve materially improves.

Scientific accounting remains `0/2` action gates.  The substantive gain is a
root-cause discrimination: direct antagonism is rejected, while severe
near-orthogonal coordinate dominance is measured and now has one clear causal
control.

### Action-initialization audit and what the early curve can prove

The current arm is not initialized from a trained WLA robot-policy checkpoint.
The official repository does publish
`SJTU-DENG-Lab/wla_libero_all_image_action`; the statement here is about the
actual ADR-224 initialization, not checkpoint availability.
The installation path first records and removes LingBot's released action
expert, then `LingBotWLASharedInterface.from_pinned_source(...)` constructs a
new upstream `LayerwiseFlowmatchingActionHead` directly from the pinned YAML.
There is no action-head `load_state_dict` or `from_pretrained` call on this path.
The supplied `Sana_600M_512px_diffusers_64channels` root initializes the world
expert/VAE path, not the 28-layer action expert.  The released full WLA
checkpoint cannot be loaded unchanged after ADR-224 changes the cross-attention
width from RynnBrain's 2048 to LingBot's 2560 and expands the action/state ABI
to 55 dimensions; loading only shape-compatible tensors would itself be a new
partial-transplant hypothesis.  The resulting comparison is
therefore a pretrained LingBot host plus a freshly initialized approximately
390M-parameter WLA action topology.

This changes the permitted interpretation but not the validity of matched
causal controls:

1. `picf_full` versus `wla_lbot_masked`, and frozen versus joint source updates,
   start from the same random action interface and can test whether PICF evidence
   causes *relative* early action learning under that interface;
2. twenty global-batch-four updates expose only 80 samples and occur inside
   WLA's 1,000-step learning-rate warmup.  They cannot establish the final
   capability of a newly initialized action expert;
3. absolute superiority over released LingBot remains untested until a
   deployable matched baseline is retained.  The donor-faithful explicit-`t-8`
   arm is required, and a mature initialized action policy is preferable if an
   exact compatible checkpoint exists;
4. the historical ADR-149/Stage-P-Q early advantage is not evidence against
   this audit.  Those arms retained LingBot's mature action route, improved by
   about 12% at step 20, then lost the lead by step 100.  ADR-224 deliberately
   changes that initialization and interface, so losing the old fast curve is
   expected unless the new WLA action reader is pretrained or given its proper
   training timescale.

Accordingly, action progress remains `0/2`.  Discovering why the old early
advantage disappeared is scientific root-cause progress, but it is not itself
movement toward an action lead.

### Preregistered source-coordinate update intervention

The next acquisition changes no model, data, query, mask, modality, loss,
forward, backward, FSDP, learning-rate, or random-draw surface.  Both arms run
the complete train-mode VidEoMT graph and its unchanged five-frame source
objective, and both compute the same source gradients.  The sole intervention
occurs after the shared backward transaction:

- `joint`: apply the registered VidEoMT optimizer and scheduler update;
- `frozen-coordinate-control`: audit and clip the same source gradients, then
  discard them without stepping the VidEoMT optimizer or scheduler.  The host
  optimizer and WLA scheduler still step normally.

Both arms must be reacquired under the same implementation digest.  Reusing an
older joint run would confound the intervention with code identity.  Before
interpreting learning, rank-paired step-one source, host, action, and world
losses must be exactly equal; any mismatch invalidates the experiment.

The registered evidence is the complete rank-paired training curve through
step 20, windows `1--5`, `6--10`, `11--15`, and `16--20`, plus the fixed
step-zero/step-20 action difference-in-differences.  The moving-coordinate
hypothesis is supported as a *primary early root cause* only if freezing
improves the fixed-set learning ratio of ratios by at least `2%` and its
source-episode bootstrap 95% upper bound is below zero, without a material
training-curve regression.  A smaller effect rejects coordinate motion as the
primary explanation.  This diagnostic threshold does not pass an action gate:
promotion still requires an approximately `5%` stable whole-curve lead, the
donor-faithful explicit-`t-8` control, sampled actions, and CALVIN rollout.

## Final source-coordinate control and superseding decision

The source-frozen and joint arms completed through step 20 under the same
implementation (`c7f09d5d...36163`), model family, data stream, random draws,
host-evidence arm, complete source forward/backward graph, and fixed evaluation
set.  The only intervention was whether the already-computed VidEoMT source
optimizer and scheduler update was applied.

The immutable run manifests correctly remain `DECLARED`; completion is proven
by matching `EARLY_STOP` summaries at step 20 and passing fixed-set snapshots at
steps 0 and 20.  The initial bitwise comparator recorded six source CE scalars
that differed by at most `1.1920928955078125e-7`.  Ranks 0 and 1 were bitwise
equal, and action loss, total objective, posterior hash, inputs, and all
discrete source fields were bitwise equal on every rank.  This is one float32
ULP, so the final comparator retains `exact=false` while registering
float32-equivalence under a two-ULP IEEE-754 bound.  No mismatch is hidden.

Registered result:

| Surface | Frozen relative to joint |
|---|---:|
| training, steps 1--20 | `+0.0150%` |
| training, steps 1--5 | `0.0000%` |
| training, steps 6--10 | `-0.0032%` |
| training, steps 11--15 | `+0.0297%` |
| training, steps 16--20 | `+0.0336%` |
| fixed-set learning DiD, step 0 to 20 | `+0.0841%` |
| fixed-set DiD 95% interval | `[+0.0047%, +0.1381%]` |

Decision:
`REJECTS_COORDINATE_MOTION_AS_PRIMARY_EARLY_ROOT_CAUSE`.  The formal payload is
`/mnt/picf-next/adr224/evidence/matched-source-frozen-vs-joint-step0-20-v2.json`,
file SHA-256 `b69124f93a3f905fb754673c1a8531df83b7fd7d752ca653537fb2687095cc04`,
internal artifact SHA-256
`5dd1408a1994338c4e60e7d0fb10cc95e0a6fc5cebd52e8f311796c8ba838a9b`.
This result authorizes zero action gates.

### What was scientifically repaired

1. The previous query-only transplant discarded the donor's spatial field
   `X` from `MaskMLP(q_i)^T Upscale(X)`.  Restoring the complete query-patch
   relation raises fixed-set oracle IoU from about `0.073` to about `0.817` and
   oracle Recall@0.5 to about `0.910`.  Original-resolution review confirms
   fine masks for small blocks, buttons, switches, drawer/slide, and other
   objects.  This is a genuine object-candidate representation repair.
2. The failure has been separated into capacity and use.  Top-10 recall is only
   about `0.126`, while Top-100 recall is about `0.883`; candidate geometry is
   strong, autonomous ranking and binding are weak.
3. Direct negative source/action gradient conflict is rejected, and the matched
   freeze control now rejects source-coordinate motion as the primary action
   root.  PCGrad, source freezing, and source-LR tuning are therefore forbidden
   continuations.
4. The implementation audit identifies an independent action-interface break:
   `lingbot_wla_install.py` deletes LingBot's released action expert and creates
   a fresh approximately 390M-parameter WLA action topology without loading a
   trained robot-policy checkpoint.  Twenty updates occur inside a 1,000-step
   warmup.  This explains why ADR-224 cannot reproduce the historical early
   LingBot/PICF curve, but it does not count as an action gain.

### Distance to action superiority

Action progress is still `0/2`: no accepted whole-curve loss lead and no
rollout lead.  The registered full-versus-masked fixed-set DiD is `+1.147%`
(worse), so reaching the `-5%` gate requires a swing of about `6.15` percentage
points.  The historical ADR-149/Stage-P-Q path retained LingBot's mature action
route and led by about 12% at step 20, but tied or regressed by step 100.  It
proves transient optimization acceleration, not durable posterior mediation.

### Next admissible architecture boundary

The current WLA hybrid is closed.  The next arm must retain the released,
initialized LingBot action expert and its exact action ABI.  Object evidence may
enter only through the same large LingBot host that already conditions that
expert.  A small private selector, lifecycle controller, replacement action
head, or partial WLA checkpoint transplant is forbidden.

The target causal contract is:

```text
complete RGB/language/typed modalities/history
                    |
                    v
        one shared LingBot host
             |             |
             v             v
   addressable object state  released LingBot action expert
             |___________________________^
                    mandatory mediation test
```

The first gate is not another 30K run.  It is a same-checkpoint, same-stream
LingBot baseline/PICF pair that proves all of the following before scaling:

1. step-zero sampled action parity when PICF evidence is masked;
2. object-address swap changes the corresponding action-conditioned state and
   wrong-row swaps do not;
3. Top-K is replaced by addressable soft access or a large-host learned query,
   not a private scorer;
4. the complete fixed-set curve leads by at least 5% through the registered
   early and crossover windows;
5. rollout success leads under the same checkpoint-selection rule.

OA-WAM provides the closest 2026 mathematical evidence for persistent
address/content separation and a slot-swap causal test, but no official
executable repository is currently public and its published preprocessing uses
external SAM/DINO object construction.  AffordanceVLA publishes a progressive
Understanding-Affordance-Action implementation, but omits robot dataset loaders
and trained task weights and depends on external RexOmni/SAM labels.  MemoryVLA
and VLA-JEPA publish useful mature code for long memory and leakage-free latent
world prediction, respectively; neither supplies the missing object-to-action
addressing contract.  None is silently reimplemented or simplified here.

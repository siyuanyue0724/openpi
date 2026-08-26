# ADR-209: LingBot-Native FLARE Future-Latent Arm

Date: 2026-08-23 (Asia/Shanghai)

Status: architecture and implementation contracts frozen; typed integration,
source replay, focused regressions, the 250-step frozen-target cache and the
real-weight mechanics gate pass. Restoring LingBot's released
FSDP2-then-whole-model-compile order removed the uncompiled dense
FlexAttention fallback. The strict matched `lambda=1`/`lambda=0` 20-step curve,
fixed cold evaluation and target-identifiability audit are complete. The
generic-SigLIP FLARE arm is rejected for 100, 2k and 30k escalation: it learns
its auxiliary target but does not improve fixed action evaluation and admits a
strong action-independent constant-template shortcut on CALVIN.

## 0. Decision

Final experimental verdict (2026-08-24): **reject this arm for escalation**.
The implementation decision below is retained as the preregistered historical
contract; its exact execution succeeded, but its scientific promotion gates did
not. This rejects the generic global SigLIP target on the current CALVIN domain,
not all future-latent objectives and not the whole PICF hypothesis.

Implement one new arm in which LingBot's released 36-layer action expert owns
both action denoising and implicit future-state prediction. The arm reproduces
the complete publicly specified FLARE mechanism using the paper's successful
generic target variant:

1. learned future tokens inside the action-expert token sequence;
2. joint self-attention between noised-action and future tokens;
3. a two-linear-layer prediction MLP;
4. a frozen `google/siglip2-large-patch16-256` future-image encoder;
5. `2x2` average pooling from 256 to 64 tokens per image;
6. intermediate-depth cosine alignment at the paper's `6/8` relative depth;
7. future target offset `t+16` source frames;
8. alignment coefficient `lambda=0.2`;
9. target stop-gradient; and
10. future tokens retained during policy inference while the target encoder is
    absent.

No published mechanism above may be deleted, narrowed or replaced for memory,
speed or convenience. A resource problem must be solved by FSDP, activation
checkpointing, offline exact target caching, or additional hardware. It does
not authorize a smaller teacher, fewer tokens, one-camera substitution, final-
layer-only alignment, a shallower action expert, or an independent predictor.

The complete FLARE implementation repository is not public: its project-page
GitHub link is unavailable, and public GR00T N1.5 restores future tokens but
explicitly disables FLARE hidden-state training. Therefore this arm is a
paper-faithful implementation plus explicitly listed LingBot/CALVIN adapters,
not a source-identical transplant. Any report must preserve that distinction.

## 1. Normative evidence and exact provenance

### 1.1 FLARE paper contract

The normative paper is arXiv `2505.15659v1`, local PDF SHA-256:

```text
37245f9a586229ed477b1c5194b051973ba5b3f050d25cc76e8aefe6a2845aeb
```

The main text and Appendix D jointly specify:

```text
sa_tokens = concat([state_token, action_tokens, future_tokens], dim=1)
action_output = action_decoder(policy_output[:, 1:1+A])
future_prediction = embedding_decoder(policy_output[:, -M:])
L_flare = 1 - cosine(future_prediction, stopgrad(target(future_obs)))
L = L_flow + 0.2 * L_flare
```

The main experiment aligns after layer 6 of 8. The successful generic-target
ablation uses pretrained SigLIP2-Large vision tokens at `t+16`; its best generic
variant applies `2x2` average pooling, yielding 64 tokens per image and a
reported success rate of `50.9%` versus `43.9%` without FLARE. This is the
selected arm because its checkpoint and model implementation are public.

### 1.2 Public implementation boundary

GR00T N1.5 commit
`4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` supplies the mature insertion
scaffold:

- learned `nn.Embedding` future tokens;
- future tokens in both training and inference action sequences;
- action decoding restricted to action positions; and
- optional intermediate hidden-state return in the action DiT.

It is not a complete FLARE implementation. Its fine-tuning forward sets
`return_all_hidden_states=False` and contains no target encoder, two-layer
future decoder, cosine objective, or EMA implementation.

LingBot source commit
`2838c1862bbec1ea47942fb61512130f635eb595` supplies the unchanged large action
owner: a 36-layer, width-768 Qwen2 action expert jointly attending with the
Qwen3-VL observation stream. The current PICF source patch already preserves
all 36 action and language layers and the complete LingBot action loss.

The generic target is fixed to Hugging Face model revision:

```text
model: google/siglip2-large-patch16-256
revision: 787800c8990e6f058423089178e718139608408c
```

The revision, processor files and model tensors must be content-hashed in every
cache manifest and run receipt.

## 2. Non-negotiable architecture philosophy

### 2.1 One large semantic and control owner

LingBot remains the only semantic/action owner. Future-state reasoning occurs
inside its released large action expert. The future decoder is exactly the
paper-required projection from expert hidden width to teacher width; it cannot
select objects, determine lifecycle, route tasks or produce actions by itself.

VidEoMT, V-JEPA2.1, AnyTouch and Sonata remain evidence encoders feeding the
same LingBot-owned posterior. They are not independent task winners. The new
FLARE target is loss-only and cannot enter the deploy-visible current input.

### 2.2 Implicit future modeling, not image reconstruction

The new arm predicts a future representation, not RGB pixels. This retains the
information relevant to the teacher geometry and semantics without paying an
explicit image-generation likelihood. No image decoder, SAM mask, winner head,
hard lifecycle threshold or handcrafted object label is introduced.

### 2.3 What this arm may claim

A passing arm may establish that the action expert learns an action-mediated
future representation and that this improves or preserves the matched LingBot
action curve. It does not by itself establish object identity, persistent row
tracking, cross-modal correspondence or closed-loop CALVIN success.

Those PICF claims still require the retained complete VidEoMT object bank,
all-modal interventions, temporal controls and policy evaluation. Global FLARE
future tokens must never be renamed "object anchors" without a separate,
permutation-equivariant object-indexed bridge and its own ablation.

## 3. Exact mathematical contract

For sample `b`, current observation `o_t`, native CALVIN state `q_t`, clean
action chunk `a_t`, flow noise `epsilon`, and LingBot flow time `tau`, preserve
LingBot's released flow convention:

```text
x_tau = tau * epsilon + (1 - tau) * a_t
u_tau = epsilon - a_t
L_flow = mean ||v_theta(o_t, q_t, x_tau, tau) - u_tau||^2.
```

FLARE does not alter this parameterization. It adds learned future tokens
`F in R^(M x 768)` to the native action suffix:

```text
S_0 = [E_state(q_t), E_action(x_tau, tau), F]
H_1, ..., H_36 = LingBotActionExpert(S_0; current VLM/PICF prefix)
Z_hat = D(H_27[future positions])
Z_target = stopgrad(G(o_(t+16)))
L_flare = mean_(b,m) [1 - cos(Z_hat[b,m], Z_target[b,m])]
L_total = L_existing_PICF + 0.2 * L_flare.
```

`H_27` means output after the 27th one-based LingBot action layer, i.e. Python
index 26. This is the exact relative-depth map `27/36 = 6/8 = 0.75`. Because
FLARE does not publish a 36-layer LingBot checkpoint, this depth mapping is an
explicit adapter and must be ablated against nearby depths before 30k.

The suffix block mask remains LingBot-native:

```text
block 1: state token
block 2: all action tokens and all future tokens
```

Thus state cannot read future/noised-action tokens, while action and future
tokens interact bidirectionally exactly as FLARE requires. All suffix tokens
continue to read the current LingBot/PICF prefix through the released joint
attention path.

## 4. Complete generic SigLIP2 target

### 4.1 Two-camera CALVIN mapping

CALVIN supplies `rgb_static` and `rgb_gripper`; neither is deleted. For each
eligible current source frame `i`, read raw same-episode frames `i+16` in fixed
view order:

```text
[rgb_static, rgb_gripper]
```

Each view is processed independently by the exact frozen model revision and
its released image processor at `256x256`. The vision model must yield a
`16x16` grid of 256 patch tokens of width 1024. The width is taken directly
from the pinned official model config; it is not a PICF projection. Pool without
learned weights:

```text
reshape [16,16,1024]
mean over each non-overlapping [2,2] cell
reshape [64,1024].
```

Concatenate the two view grids in fixed order to obtain
`Z_target in R^(128 x 1024)`. Therefore `M=128` future tokens. Using only 64
tokens would silently remove one CALVIN camera and is forbidden for this arm.

### 4.2 Prediction decoder

Appendix D requires a two-layer MLP but does not publish hidden width or
activation. The adapter therefore reuses LingBot's released two-layer action-
time MLP convention rather than inventing a new decoder family:

```text
Linear(768, 768) -> SiLU -> Linear(768, 1024).
```

No LayerNorm, Q-former, selector, token pooling or extra attention block is
added. This dimensional map is an explicit implementation adapter and must be
reported as such.

### 4.3 Frozen target and EMA

The selected generic SigLIP2 target remains frozen and receives no gradient.
EMA `0.995` in FLARE applies to its action-aware target that is homologous to
the trainable policy encoder. There is no homologous SigLIP2 policy encoder in
this generic-target arm, so fabricating an EMA update would not reproduce that
ablation. Frozen target is the selected paper-proven generic variant, not a
deletion of its mechanism.

### 4.4 Exact cache equivalence

Offline caching is allowed because `G` is frozen and target computation has no
training-time randomness. A cache record is valid only if it binds:

- dataset ID, revision and immutable dataset tree hash;
- current and future source indices and same-episode proof;
- both future source-frame content hashes;
- view order;
- model ID and immutable revision;
- every model and processor file hash;
- preprocessing parameters;
- unpooled and pooled tensor shape contract;
- target dtype and byte hash; and
- the frozen stream/split coverage digest.

The reference cache stores pooled FP32 tokens. It may be cast only at the
cosine computation boundary; no lossy PCA, quantization, resampling or token
selection is authorized. Build a short-gate cache first, then extend the same
contract only after the gate passes.

## 5. LingBot integration contract

### 5.1 Baseline preservation

The primary FLARE causal control retains the complete candidate architecture:
the same 128 future tokens, prediction decoder, cache reads, parameter groups,
suffix length, forward graph and LingBot/PICF inputs. It changes only the
objective coefficient from `lambda=1` to `lambda=0` after the paper-fixed
internal weight `0.2`. This isolates the learning signal without confounding it
with sequence length or parameter count.

An architecture-off LingBot/PICF run remains necessary as a secondary control
for the cost and effect of inserting future tokens at all. It is not the
primary `lambda=0` ablation and must never be reported as one. In every arm the
released official 11-field LingBot output ABI remains unchanged.

The FLARE weighted loss is exposed separately from `official_total_loss` and
`official_action_loss`. This prevents the auxiliary term from making a matched
action comparison circular.

### 5.2 Training behavior

When active:

1. append all 128 learned future tokens after the native action tokens;
2. preserve native state/action embeddings and flow randomness;
3. decode velocity only from positions `[1:1+n_action_steps]`;
4. capture future positions only after action layer 27;
5. apply the exact two-layer prediction decoder;
6. compute FP32 tokenwise cosine against the detached cache target;
7. add exactly `0.2 * L_flare` once to the optimizer objective; and
8. report raw cosine, raw loss, weighted loss, token count and gradient norms.

No future source image, cache key, target tensor or future frame identity may
enter the current VLM/PICF prefix, attention keys/values, posterior state,
action decoder or inference API.

### 5.3 Inference behavior

Future tokens remain in the action suffix and jointly influence action tokens,
as in the FLARE policy architecture and public GR00T scaffold. The SigLIP2
target encoder/cache and alignment decoder output are not needed. Action
integration, state/action normalization and output shape remain unchanged.

### 5.4 FSDP and checkpoint behavior

The future embedding and two decoder linears must be registered before FSDP2
parallelization and included in optimizer/checkpoint manifests. The frozen
SigLIP2 encoder is never an optimizer member. A resume must restore future
embedding, decoder, optimizer and scheduler state exactly. Missing new-arm keys
are allowed only when initializing from the immutable LingBot baseline, never
when resuming an ADR-209 checkpoint.

The production write cadence remains `100` metric steps, `250` visual steps
and `2000` checkpoint steps. A mechanics or short scientific stop that is not
on a 2000-step boundary must close its metric journals without serializing the
complete 6.265B model and optimizer merely because the process is ending.
Explicit DCP, causal and acceptance checkpoints keep their separately frozen
rules.

## 6. CALVIN eligibility and no-leakage ABI

Only physical events with at least 16 raw same-episode future source frames are
eligible. The existing stream-domain builder must enforce this before freezing
the sample plan. No frame is padded, repeated or borrowed across a reset.

Current host input is unchanged. Future RGB may be read only after the complete
current model input has been constructed and only through the loss-target
loader. The target loader must expose token tensors, not RGB, simulator state,
task labels, object masks, row IDs or future actions.

## 7. Required controls and gates

### 7.1 Local/static gates

- `lambda=0` preserves the candidate module graph, suffix length and action slice;
- the architecture-off secondary control preserves the released action ABI;
- active path has sequence `[1 state, A actions, 128 future]`;
- action decoder cannot read future positions by slicing error;
- future tokens and decoder receive nonzero finite gradients from FLARE;
- action expert layers before and at layer 27 receive FLARE gradients;
- target tensor, model and cache remain gradient-free;
- zero weight removes only FLARE objective gradients while retaining gradients
  from the action objective through the shared future-token architecture;
- target permutation changes only FLARE loss, never current forward inputs;
- inference requires no target and retains future tokens;
- reset/end-of-episode samples cannot cross source boundaries;
- two-camera order and 16x16 to 8x8 pooling are exact;
- FSDP2 placement and DCP resume include every trainable new parameter; and
- implementation/source patch replay and file hashes are deterministic.

### 7.2 Short real-weight GPU gate

Use the same immutable CALVIN stream, seed, global batch, trainable scope,
optimizer, LingBot checkpoint, 128-token suffix and complete cache ABI for the
candidate and `lambda=0` control. The sole primary-control difference is the
objective coefficient. First run mechanics, then a same-seed 20-update curve
to reject broken execution or an immediate action regression, then the
preregistered 100--250-step gate. Do not infer scientific superiority from 20
updates and do not wait for a rounded endpoint after a hard failure is already
established.

At steps 0, 50, 100 and terminal, report:

- matched official action loss and its entire curve/AUC;
- raw and weighted FLARE loss and mean cosine;
- future-token/decoder/action-layer gradient norms;
- throughput and peak memory relative to matched LingBot;
- all existing anchor and modality metrics; and
- actual/zero/reversed/shuffled action and correct/current/reversed/shuffled
  future-target interventions under fixed noise.

The arm fails immediately for leakage, missing new gradients, target gradients,
wrong token count, action-slice drift, nonfinite values, FSDP/DCP mismatch,
cross-reset targets or a target-insensitive prediction. It cannot be promoted
because training FLARE loss decreases alone.

### 7.3 Promotion boundaries

`2k` requires all of:

1. mechanics and no-leakage gates pass;
2. correct future target is preferred to current/reversed/shuffled controls;
3. actual-action conditioning is measurably distinct from zero/reversed/
   shuffled controls;
4. the matched action curve is not materially worse and shows a credible
   improvement trend rather than one selected endpoint;
5. complete VidEoMT geometry and all present modality routes remain active;
6. throughput/memory are measured and operationally acceptable; and
7. the complete checkpoint resumes exactly.

`30k` additionally requires a 2k matched action advantage, autonomous anchor
and all-modal interventions, long-posterior controls and closed-loop CALVIN.
No auxiliary training loss can substitute for policy evidence.

## 8. Implementation checklist

- [x] Obtain explicit approval for the LingBot-owned FLARE arm.
- [x] Freeze the paper, source and public-code availability boundary.
- [x] Freeze the generic SigLIP2-Large model ID and immutable revision.
- [x] Freeze two-camera, `t+16`, 128-token target semantics.
- [x] Freeze token order, action slice, relative layer-depth mapping and loss.
- [x] Freeze no-leakage, intervention and promotion contracts.
- [x] Add a replayable LingBot source patch exposing complete future-token
      insertion, layer-27 capture and inference behavior.
- [x] Implement the typed future-latent module and result/gradient receipts.
- [x] Implement immutable SigLIP2 target generation and exact cache manifests.
- [x] Wire targets through both the task-independent and native-VidEoMT paths.
- [x] Add architecture-off, strict `lambda=0`, active, negative, gradient,
      FSDP2 and DCP tests.
- [x] Pass focused local/static regression and source-patch replay. The
      latest production Python run passed `182/182` focused FLARE, runner and
      placement tests and Ruff; this is not
      a substitute for the real-weight gate.
- [x] Freeze all code/assets to `/mnt/picf-next/adr209` with hashes.
- [x] Pass the real-weight three-update four-GPU mechanics gate.
- [x] Run the strict matched 20-step scientific rejection gate and fixed cold
      evaluation.
- [x] Decide whether a 100--250-step extension is justified: rejected because
      the candidate loses the fixed endpoint comparison and the target admits
      an action-independent lower-loss solution.
- [x] Decide `reject` or `authorize 2k`: reject. `30k` remains forbidden.

## 9. Current honest assessment

This is a strong, paper-supported repair for the specific missing mechanism
identified by ADR-207: the action network was not required to encode the
future consequences of its action. It is materially better founded than a new
winner head, calibrator or independent world-model adapter.

It is not yet proof that PICF beats LingBot, and it is not a complete released
source transplant. The dominant residual uncertainty is scientific rather
than architectural: whether a generic future-vision target on CALVIN produces
useful action-mediated features soon enough and whether those features improve
the matched action curve while coexisting with persistent object rows.

## 10. Live execution ledger

At 2026-08-23 19:49 CST, the immutable four-rank 250-step gate contract and the
complete frozen SigLIP2 target cache exist under `/mnt/picf-next/adr209`:

```text
training-prefix steps: 250
physical visits / unique future targets: 1000 / 1000
target tensor: FP32 [128, 1024]
cache canonical manifest SHA-256:
  b25c59c077ffac3716446f43b03a4aa9ab93dbbdb03c16ef8337f7fb9f8292c3
teacher file receipt SHA-256:
  e843c6db2d57637789470a90457a4533cc66d7bd88eb711c7dd5cbd9edeac235
```

The first AnyTouch/Sonata/V-JEPA publication attempt failed before reading data
because a previous host-specific Python path was absent. The launcher now pins
the available production Python 3.12 runtime, and all three complete frozen
evidence routes are rebuilding concurrently. This was an execution-path defect,
not evidence about any modality or the model. No encoder, token, view, layer or
objective was reduced in response.

All three evidence publications subsequently completed with the frozen
manifests used by the live gate:

```text
AnyTouch: f5dbe6eb84e90a492a90c664bf226b27ec8332537a328e7384fb9e1c40292bb7
Sonata:   2af1baca8fc285e3143fc8933e7078ab2f086aeae43ebb3f3d780e5f23794833
V-JEPA:   c949b1769029c16e72f11bc0f28acd69d1b1373b234b39949e3d6950ae17e57e
```

### 10.1 Four-GPU mechanics history

The execution attempts are cumulative evidence, not interchangeable model
ablations:

1. `v6`, selective shared-embedding offload, retained the complete `full-host`
   graph. Step 1 completed forward, backward and Muon update in `50.465 s` with
   all `435/435` trainable source tensors finite and present. Step 2 failed in
   `flex_attention_backward` on rank 3 when a sample-dependent peak requested
   another `1.03 GiB` with only `933 MiB` free. This proves one full update but
   rejects the placement for the immutable stream.
2. `v7`, public PyTorch full-model `CPUOffloadPolicy`, completed the same full
   forward and backward with substantially lower GPU residency. Its CPU Muon
   update still had not completed after more than seven minutes, so it was
   terminated as operationally unusable. No model-quality conclusion follows.
3. `v8`, complete frozen-visual-root offload, was rejected before weight loading:
   ADR-209 declares `full-host`, including `415,347,712` trainable visual
   parameters. Running a frozen-vision source would have changed the scientific
   model instead of solving execution capacity.
4. `v9` used a fifth execution-only source overlay on the exact v3 chain. It
   applies public FSDP2 CPU offload only to the complete *trainable* visual root
   and shared embedding. It changes no module, parameter trainability, forward,
   target, loss or optimizer hyperparameter. The ordered source replay passed
   with patch SHA-256
   `e0ca8b0587ebe6e38b16b4ff83a0298d59918904e5c8839ea33dd20015359fde`
   and parallel-source SHA-256
   `f0836993c3dcd1b43dfe03c8c78f2360b56f2d6fe22b5b3bf54355474d263def`.
   Step 1 completed forward and backward, but the unmodified Muon mega-batch
   keyed only by global shape and dtype. It therefore stacked CPU-offloaded
   visual shards with CUDA language shards of the same shape and failed in
   `torch.stack(local_updates)`. This is an optimizer execution-device defect,
   not action, anchor or FLARE evidence.
5. `v10` adds one execution-only Muon closure to the exact ordered chain. It
   groups equal-shape FSDP shards by device type, stages only bounded CPU
   mega-batches to the local CUDA rank for the existing NCCL/Newton-Schulz
   computation, and writes the result back to the parameter's original device.
   It changes no optimizer equation, hyperparameter, trainability or model
   module. Patch SHA-256 is
   `3b59e07a41617f627f67e013e11952b25a95f69bdb7bd4e8441880402fc32f56`;
   the resulting Muon source SHA-256 is
   `98a1de761c2f8f8e1c041d3c0e4b053b2aef2df3b84a7c34fc1c5973c10553b5`.
   All six overlays passed immutable ordered replay and the prepared FLARE
   source check. Four ranks then completed two consecutive full forward,
   backward and optimizer updates in `48.759 s` and `36.406 s`. Every source
   gradient was finite and present. Peak reserved memory across ranks was
   `37.191 GiB` at step 1 and `38.818 GiB` at step 2. The second-step full-model
   checkpoint was intentionally cancelled after both updates when inspection
   proved it was only writing an out-of-cadence terminal checkpoint; short
   gates now retain metrics but checkpoint only at the frozen 2000-step cadence.
   This passes the two-update execution gate but is not scientific evidence.
6. `v11` replayed the same immutable candidate for a 20-update curve. It
   completed seven full optimizer updates. Across the first six global means,
   FLARE cosine increased monotonically from `-0.014248` to `0.234754`, while
   raw FLARE loss decreased from `1.014248` to `0.765247`. Update eight then
   failed in rank-3 `flex_attention_backward`: the native allocator had
   `37.65 GiB` allocated and `948.04 MiB` reserved but unusable, and could not
   provide one contiguous `264 MiB` block with `255.56 MiB` physically free.
   This rejects the `expandable-segments` execution placement for this fixed
   stream. It is not an action, anchor or FLARE-quality failure.
7. `v12` kept the exact source, trainability, model, data, seed, optimizer and
   objective and selected PyTorch's public `cudaMallocAsync` allocator through
   a pre-import, manifest-recorded mode. It failed during update one after a
   `1.03 GiB` request with only `15.56 MiB` physically free. This is strictly
   worse than `v11`; allocator substitution is rejected.
8. The root cause was then traced to `lingbot_compile_mode=disabled`. This made
   FlexAttention backward enter PyTorch's `sdpa_dense_backward` fallback,
   whose temporary mask materialization is sample-shape dependent. ADR-207 had
   already demonstrated the released LingBot order `FSDP2 -> torch.compile`
   for 200 updates. `v13` attempted that restoration but was stopped before
   model execution because it accidentally read the 27 GiB checkpoint from
   slow `/mnt`; it provides no model evidence.
9. `v14` restored the released whole-model compile order and read immutable
   weights from `/opt`. Four ranks completed two full forward, backward and
   optimizer updates. Rank-0 peak reserved memory was `31.020 GiB` and
   `35.613 GiB`; step times were `139.269 s` including first compilation and
   `66.740 s` for the second new shape. FLARE raw loss moved from `1.009609` to
   `0.958964`, and all host/source gradients were finite. This accepts the
   compile repair as a mechanics fix, not as quality evidence.
10. The old control deleted the future-token branch and therefore confounded
    objective benefit with architecture and sequence length. The current
    candidate/control pair instantiates identical complete arms and differs
    only by `future_latent_objective_scale in {1, 0}`. Its focused cloud
    regressions pass `182/182`; the matched curves remain pending.

None of these mechanics attempts establishes action or anchor superiority.
Matched candidate/control curves and visual/intervention evidence remain the
only promotion evidence.

### 10.2 Strict candidate `v15`

The first complete 20-update candidate is:

```text
/mnt/picf-next/adr209/runs/flare-strict-candidate20-20260824T0055CST-v15
```

It completed all 20 forward, backward and optimizer updates with the complete
6B LingBot host, 200 native VidEoMT queries, both CALVIN cameras, all frozen
AnyTouch/Sonata/V-JEPA evidence routes, 128 `t+16` future tokens and the
released whole-model compile path. Maximum reserved memory was `38.824 GiB`;
steady-state median update time was approximately `30.79 s`.

The future target is learnable. Global-mean raw FLARE loss decreased at every
update from `1.014250` to `0.423861`, while cosine increased from `-0.014250`
to `0.576139`. The official action-loss training AUC was `0.470996`; first-five
and last-five means were `0.595898` and `0.396191`.

The fixed, cold-reset action evaluation improved from step 0 to step 20:

```text
partition    step 0       step 20      relative change
validation   0.472283     0.369026     -21.86%
heldout      0.450353     0.369126     -18.04%
```

This is evidence of short-horizon action learning, not evidence that FLARE
caused the improvement. The strict `lambda=0` result is required for that
claim.

Source geometry did not collapse. Oracle-assigned binary IoU at step 20 was
`0.826581` on validation and `0.820628` on heldout, with recalls `0.924051` and
`0.916006`. Autonomous top-10 proposal selection remained weak: binary IoU was
`0.122835` and `0.124118`, and recall was `0.124615` and `0.130130`. The source
bank therefore contains strong fine-grained object masks, but neither this
20-step arm nor FLARE has yet demonstrated reliable autonomous row ranking.
That unresolved distinction must not be hidden by oracle visualizations.

Against the secondary architecture-off ADR-207 `v18` reference, all 80
rank/step stream records match in sample key, augmentation seed, flow noise,
flow timestep and source digest. Its first-20 action AUC was `0.467065`, versus
`0.470996` for `v15`; last-five means were effectively equal (`0.396387`
versus `0.396191`). This does not show a short-horizon action advantage over
the architecture-off reference. Because its fixed evaluation split differs,
that reference cannot replace the strict same-graph control.

### 10.3 Strict-pair integrity

The primary control is:

```text
/mnt/picf-next/adr209/runs/flare-strict-control20-20260824T0123CST-v16
```

Before training, an automated leaf-level manifest comparison found exactly
four differences and no others: the objective scale, the candidate/control
objective label, the candidate/control architecture-profile label and the
execution-contract hash derived from those fields. Both runs use implementation
SHA-256
`91c0ae5bc6af673f57d838640b09eddb5b45a4bb8a7a1a55321902e218ee33e4`.
All 102 fixed step-0 action losses match exactly, sample by sample; validation
and heldout means are identical with zero paired delta.

`tools/compare_adr209_flare_strict_pair.py` freezes this comparison. It rejects
any extra manifest difference, any rank-wise mismatch in sample keys,
augmentation seeds, flow noise, flow timestep, source digest or temporal plan,
and any initial fixed-evaluation mismatch. Four focused regression tests and
Ruff pass.

The complete strict result is:

```text
metric                                  candidate       lambda=0 control
20-step training action AUC             0.470996        0.476587
last-five training action mean          0.396191        0.396777
step-20 heldout cold action              0.369126        0.367417
step-20 validation cold action           0.369026        0.361242
step-20 FLARE raw loss                   0.423861        1.019175
20-step source-loss AUC                  7.279049        7.278506
maximum reserved memory GiB             38.824          38.824
```

Lower action loss is better. Candidate-minus-control training AUC is
`-0.005591` (`-1.17%`), but the last-five delta is only `-0.000586`. The small
training-curve advantage does not survive fixed evaluation: candidate is
`0.47%` worse on heldout and `2.15%` worse on validation. Heldout has 31
candidate wins, 31 losses and 6 ties; validation has 11 wins, 20 losses and 3
ties. This is not a broad action improvement.

The immutable comparison artifact is:

```text
/mnt/picf-next/adr209/comparisons/flare-strict-v15-v16-step20.json
semantic SHA-256: 80230c13b50e492d15665ecc7108579b017659b62ef842f788066b5a399ecdd4
file SHA-256:     dff7c7440f645b1d69eb6df3175a9f5953980d1948ed25e4d2a27d2a05629ff6
```

Against the same-stream architecture-off ADR-207 `v18`, candidate action AUC
is also `0.84%` worse (`0.470996` versus `0.467065`). Its last-five mean is
effectively equal (`0.396191` versus `0.396387`), while median steady update
time rises from `29.87 s` to `30.81 s` (about `3.1%`). The arm therefore learns
the future target and recovers most of its early sequence-insertion cost, but
has not established a net policy advantage.

### 10.4 Target-identifiability audit

The strict result suggested that target learning and action learning were
decoupled. A hash-verified analysis of all 1000 frozen `[128,1024]` target
tensors confirms a strong shortcut.

For unit-normalized target token `y_j`, the best predictor that ignores sample,
observation and action and emits only one fixed vector per token position is

```text
c_j = E[y_j] / ||E[y_j]||.
```

It achieves expected cosine `||E[y_j]||`; this is the exact optimum among all
sample-independent unit predictors, not a fitted neural baseline. Averaged over
the 128 positions, the CALVIN cache gives:

```text
target                             cosine       raw cosine loss
fixed position template            0.847202     0.152798
cached current frame -> t+16        0.818167     0.181833
deterministic cross-sample offset   0.719452     0.280548
trained v15 endpoint                0.576139     0.423861
```

The static-camera fixed template is even more concentrated: cosine `0.880578`
and loss `0.119422`. Its paired current-to-`t+16` cosine is `0.905270`. The
gripper view is less static but still has fixed-template cosine `0.813825`.
Same-episode adjacent future targets have mean cosine `0.941266`.

Thus the observed candidate endpoint loss is `2.774` times the loss achievable
without reading the image, action, posterior or task. Long training can reduce
the FLARE objective by learning camera-position/background templates; the loss
does not identify action-mediated dynamics on this narrow fixed-scene domain.
This audit does not prove that v15 already uses that shortcut, because the
short run has no saved endpoint checkpoint for action/target intervention. It
does prove that the objective permits a substantially better non-causal
solution, so auxiliary convergence cannot authorize a longer run.

The immutable audit is:

```text
/mnt/picf-next/adr209/audits/flare-target-identifiability-v1.json
semantic SHA-256: 8f74e3e852dad209b3997d574aa3701a9613dc5500394c7066a0bbc06e5bc2e3
file SHA-256:     03b5e33ac725abf6dd01d6e930d374ea46302057b05d18142aaf363a4f564f23
```

`tools/audit_flare_target_identifiability.py` reproduces the calculation from
the authenticated cache. Four focused mathematical/indexing tests and Ruff
pass. The indexing test explicitly prevents shard-local row IDs from being
mistaken for global manifest rows.

### 10.5 Visual and architectural verdict

Original-resolution review of both step-20 control panels confirms the numeric
anchor result. Oracle matching finds crisp, small-object-shaped queries for the
colored blocks, button, drawer, slider and several articulated parts. The
model-ranked top-10 often selects blank, off-image or broad-background queries
and misses most small objects. Candidate and control metrics are nearly
identical: heldout oracle IoU is `0.820628` versus `0.820666`, while top-10 IoU
is `0.124118` versus `0.124505`.

This is expected from the graph. Global FLARE future tokens are not indexed by
the 200 VidEoMT rows; their target has no query assignment, objectness,
cross-modal row identity or row-to-action intervention. They can improve a
global future representation without teaching the host which object query to
retain or use. The next admissible arm must therefore satisfy both conditions:

1. its target is identifiable against constant/current/shuffled/action-zero
   controls on CALVIN; and
2. the predictive variable is permutation-equivariantly indexed by the same
   object hypotheses consumed by action, with context/null retained for
   uncertain evidence.

Inventing a new selector, lifecycle head or residual loss after this failure is
not authorized. A next arm requires a mature released donor or a separately
approved and explicitly ablated PICF composition. Until then, extending this
arm would optimize a known shortcut rather than test the claimed architecture.

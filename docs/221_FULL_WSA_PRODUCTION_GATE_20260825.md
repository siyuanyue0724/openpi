# ADR-221: Full WSA Source Transplant Production Gate

Status date: 2026-08-26 (Asia/Shanghai)

## Decision boundary

ADR-221 is not accepted because it imports or starts a model. It is accepted for
curve testing only after the exact full graph completes real distributed updates.
Scientific advantage over the matched LingBot baseline remains unproven until the
same frozen stream, seeds, batch semantics, and action metric are compared over a
meaningful curve.

## Hourly architecture audit

Every active hour before the 2026-08-26 11:00 Asia/Shanghai deadline must recheck
these non-negotiable rules:

1. Prefer a content-pinned transplant of successful upstream source code. Record
   every adaptation and do not simplify a proven mechanism without an explicit
   mathematical and empirical justification.
2. Stop a scientifically failed run immediately. Do not spend GPU time completing
   a round-number step boundary; use the failure to select a mature upstream
   replacement rather than adding an unreviewed local patch.
3. PICF promotion requires a material matched advantage across the curve. A roughly
   one-percent change, noise-level endpoint win, or regression is not success.

### 2026-08-26 03:00 Asia/Shanghai

Diagnostic v19 changes only the paired measurement construction. It removes no
upstream feature and changes no model, objective, optimizer, or training edge. The
failed gate20-v15 run remains stopped. Any later structural repair must first name
and pin a successful upstream code primitive, enumerate the exact transplant and
adaptation boundary, and then pass a short matched material-advantage gate.

### 2026-08-26 04:00 Asia/Shanghai

1. No successful donor was reduced: LingBot remains 36 layers, VidEoMT remains
   the complete 200-query source, WSA remains 36 layers/432 slots, and every
   dense modality remains active. ADR-222 adds no trainable module or scalar gate.
2. ADR-221 is stopped after decisive scientific failure. ADR-222's first launch
   was rejected before model load only because the fail-closed shell whitelist
   named ADR-221 but not ADR-222; the repair extends that asset-profile whitelist
   and changes no model computation.
3. Promotion still requires a material matched advantage over ADR-207 and official
   LingBot across the registered curve. A noise-level endpoint win is failure.
4. PICF's task-independent object inventory, same-row temporal posterior, missing-
   modality tolerance, task-conditioned action selection, and acyclic world-model
   auxiliary objective remain the governing architecture invariants.

### 2026-08-26 04:21 Asia/Shanghai

The ADR-222 two-update mechanics gate completed on four A100-40G devices. It did
not authorize 30k. All five declared host modalities were present, every rank had
finite gradients for all `435/435` trainable VidEoMT tensors, and each update used
exactly one factual WSA call and zero measurement calls. Step 1/2 took
`63.230/34.699 s`; the maximum recorded reservation was `37.23 GiB`. The execution
contract records `posterior_adoption_route=true`, `posterior_adoption_dose=false`,
and `wsa_action_coupling=auxiliary_world_decoder_no_future_action_keys`.

The frozen step-0 action loss was validation `0.48756319`, heldout `0.47875977`.
This is `5.03/4.57%` below official LingBot step 0, but versus the strongest
ADR-207 step-0 control it is `7.16%` worse on validation and only `0.90%` better on
heldout. It is therefore neither a material win nor a catastrophic cold-start
failure. The exact next experiment is the predeclared 20-step gate.

Anchor evidence separates representational capacity from ranking. Full-bank
oracle binary IoU was validation `0.82140`, heldout `0.81591`, with recall@0.50
`0.9258/0.9088`. Model-ranked Top-10 binary IoU was only `0.11214/0.11777`, with
recall@0.50 `0.1285/0.1241`. Machine-vision inspection found object-shaped oracle
masks for small blocks and articulated table parts (several sample IoUs
`0.79--0.94`), while Top-10 selection omitted some of those exact high-quality
queries. This is evidence for a query-ranking/adoption bottleneck, not permission
to add a local selector head.

### 2026-08-26 04:55 Asia/Shanghai

ADR-222 was stopped during the attempted step 17 after 16 complete updates. This
was a scientific early stop, not a runtime failure. Every completed rank-step was
finite, all declared modalities remained present, all `435/435` trainable VidEoMT
tensors retained gradients, and candidate/control sample keys and temporal plans
matched exactly. All GPU worker processes were terminated and all four devices
returned to zero allocation.

The matched action curve failed the material-advantage rule. Steps 1--5 appeared
4.61 percent better, but the paired bootstrap interval crossed zero. Steps 6--10
were 4.04 percent worse, with the interval excluding zero. The predeclared
post-warmup window had only a 1.07 percent advantage at steps 11--15 and 0.81
percent at steps 11--16; both intervals crossed zero. Across all 16 completed
updates the candidate improved by only 0.82 percent. This is precisely the
noise-level outcome that the registered rule defines as failure, so waiting for
step 20 solely to obtain a round-number endpoint was forbidden.

Decision: ADR-222 is **REJECTED**. No step-20 image evaluation, later short gate,
or 30k continuation is authorized. The result strengthens a specific diagnosis:
the complete 200-query bank contains strong object-shaped masks, but replacing
LingBot's visual action context with task-independent rows does not provide the
released action expert with a materially more learnable control representation.
It does not license a selector head, scalar gate, or learning-rate rescue. The next
candidate must transplant a successful shared-representation and action-adoption
interface together from released source, or be described honestly as a new
adaptation hypothesis and gated before expensive training.

## Architecture contract

ADR-221 keeps LingBot VLA v2 as the host/action backbone and the accepted ADR-207
native VidEoMT posterior path. It adds the released WSA Future3D expert as a third
stream in the same 36-layer attention computation. No reduced WSA decoder is used:

- 36 `Future3DBlock` layers;
- 32 attention heads;
- 432 future query slots;
- query readouts at layers 17, 23, 29, and 35;
- DA3 teacher targets from layers 11, 15, 19, and 23;
- two 504x504 future views from the exact same episode at `t+4`;
- released WSA layer weights `(1.0, 1.2, 1.4, 1.6)`;
- released objective coefficient `lambda_3d = 0.1`.

For query layer `q_i` and detached DA3 target layer `z_i`, the imported WSA loss is

```text
L_i = w_i * [1 - cos(q_i, z_i) + MSE(LN(q_i), LN(z_i))]
L_3d = (1 / 4) * sum_i L_i
L_total = L_LingBot/PICF + 0.1 * L_3d
```

Only valid camera tokens contribute. A missing wrist view is cloned for tensor
shape compatibility but remains false in the validity mask, so it contributes no
teacher loss.

The WSA parameters retain the released AdamW contract: learning rate `1e-4`,
betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-2`, gradient clip `1.0`,
5 percent warmup, cosine decay, and minimum learning-rate ratio `0.01`.

### Exact transplant boundary

"Full source" means that the complete Future3D expert computation and all 481
prepared Future3D tensors are present. It does **not** mean that this candidate is
an unmodified reproduction of the complete WSA policy. The released WSA-Large
Future3D donor has 30 layers and 24 attention heads, while the LingBot joint host
has 36 layers and 32 heads. ADR-221 uses WSA's released width-interpolation
primitive, but the 30-to-36 nearest-normalized-depth assignment in
`wsa_full_depth_adaptation.py` is a local compatibility hypothesis. It duplicates
some source-depth assignments; it is not an upstream WSA training recipe.

The more important boundary is action coupling. The successful released WSA
checkpoint contains an ActionDiT and Future3DExpert that were pretrained together
inside one MoT. ADR-221 intentionally retains LingBot's released action expert,
whose query projections were never pretrained against WSA Future3D keys and
values. Therefore the direct joint-attention edge tests a cross-checkpoint
composition hypothesis. A decreasing WSA teacher loss cannot by itself validate
that interface.

This distinction changes the repair rule. If the fixed-input intervention shows
that the direct Future3D edge harms action, a scalar gate or a reduced decoder is
not an authorized repair. The next candidate must transplant a released,
action-adoption mechanism with its complete pretrained interface, or explicitly
retrain the coupled interface under a preregistered objective.

### 2026 upstream action-adoption audit

The following source trees were inspected at their released commits rather than
reimplemented from paper prose:

- WSA, `https://github.com/zaleni/WSA`, commit
  `bfee742c585d5ee85722e658978111934c926ca3`. The released policy couples complete
  ActionDiT and Future3D experts during pretraining; released downstream recipes
  initialize from the complete coupled checkpoint. This is the closest exact
  evidence for WSA features, but replacing only LingBot action attention is not
  the same algorithm.
- LaWAM, `https://github.com/RLinf/LaWAM`, commit
  `4ea6fdadce6c9b8746028307a246b79ee2c4fd55`. Its LaWM decoder predicts the full
  future DINO feature grid. A 16-layer, width-1024 `AlternateVLDiT` action model
  alternates cross-attention between the complete current/future visual grids and
  VLM tokens. Released fine-tuning starts from a LaWAM policy-pretraining
  checkpoint and trains the flow model, LaWM decoder, and selected VLM layers.
  This is strong evidence for multi-layer future-to-action adoption, but its
  action model and pretraining are part of the method; copying only its attention
  mask would be another unvalidated simplification.
- WorldPilot, `https://github.com/ZefuLin/WorldPilot`, commit
  `a30e1fbebb208f2f6cb9b5f807e843419398d034`. It was designed to connect a frozen
  world advisor to a VLA through both latent steering and action steering.
  However, the released `CosmosImageFuser` flattens each 16x28x28 camera latent
  into one token, and the advisor action is projected to one DiT prefix token.
  That exact implementation is useful evidence that action adoption needs two
  explicit routes, but its spatial compression conflicts with PICF's 200-row
  fine-object contract. Retaining all rows would be an adaptation, not an exact
  WorldPilot reproduction.
- WLA, `https://github.com/SJTU-DENG-Lab/WLA`, commit
  `155ac94eaca8b3d1ae0789ae298fc55e37936081`. Its released implementation appends
  64 learned metaquery tokens inside the complete Qwen3-VL backbone. The complete
  action DiT layerwise cross-attends the final 28 backbone-layer metaquery states;
  the complete world expert is a separate auxiliary decoder conditioned by the
  shared backbone states and action. The world decoder never becomes an action
  key. This is the strongest released source evidence for a shared world-token
  bottleneck with an acyclic auxiliary world branch. Its 28-layer width-1024
  action expert and RynnBrain checkpoint are jointly trained components and cannot
  be replaced by a random or reduced connector inside LingBot.
- World Tokens, arXiv `2608.09730`, describes an even closer shared-world-token
  action interface, with the world decoder removed at inference. No official
  source repository was available during this audit. It is therefore theoretical
  corroboration only and is not an authorized code donor.

No one of these sources currently licenses an arbitrary hybrid. Selection waits
for the fixed-input intervention so that the replacement targets the measured
failure rather than accumulating modules.

### Historical non-repeat boundary

ADR-152 already tested a high-dose every-step posterior-only action route for 200
steps. It reduced predictive loss by 28.95 percent, but matched action was a
rough tie (about 0.18 percent) and wrong-row action was exactly invariant. Merely
increasing route probability, route duration, or predictive-loss dose is therefore
forbidden. That result used the legacy 16-row `two_pass_v3` representation rather
than the later complete 200-query VidEoMT source, so it does not directly test the
current representation, but it proves that routing alone is not a sufficient
mechanism.

### ADR-222 preregistered repair boundary

Diagnostic v19 establishes one causal repair target: the direct WSA Future3D key
surface harms LingBot action. ADR-222 may therefore change exactly two information
contracts while retaining every complete model and objective:

1. every LingBot action query, including the state query, is forbidden from
   reading Future3D keys at every layer;
2. current-scene visual and dense multimodal evidence reaches action through the
   complete 200-row posterior bank, while task language and typed proprioception
   remain direct action evidence.

Let `S`, `L`, `P`, `A`, and `F` denote current sensor tokens, task language, the
200 posterior rows, action tokens, and WSA Future3D tokens. The registered graph is

```text
S -> P -> A
L ------> A
proprio -> A
(S, P, A) -> F -> L_3d
F -X-> A
```

The decisive acyclicity condition is `reach(F, A) = false` under the transitive
closure of all repeated-layer edges. This mirrors WLA's released separation of
shared metaqueries from its auxiliary world decoder. The implementation does not
copy or shrink WLA's incompatible action head. It preserves the complete released
LingBot action expert and complete WSA Future3D expert, adds no trainable parameter,
and changes only typed attention visibility. This remains a PICF/LingBot adaptation
hypothesis, not an exact WLA reproduction, and must earn promotion at matched
20/50/100/250-step gates.

One donor-faithful risk remains explicit. Future3D queries retain WSA's released
ability to read action keys, so `L_3d` can backpropagate through LingBot action
hidden states even though Future3D cannot be read by action. WLA instead conditions
its world decoder on a separate raw-action encoder. Removing WSA's action condition,
detaching it, or inventing a reduced encoder would each modify the donor and is not
authorized without causal evidence. The matched early curve therefore tests both
forward isolation and this remaining gradient coupling; a material action regression
rejects ADR-222 rather than triggering a local scalar/gate patch.

#### ADR-222 material-advantage gate

The two-update run is a mechanics gate only. The fresh 20-step run is accepted for
step 50 only when all of these pre-result conditions hold:

1. step-20 validation and heldout action loss are each at least `5%` below the
   matched ADR-207 step-20 values (`0.36302275/0.38541188`), and the paired
   candidate-minus-control confidence interval excludes zero;
2. the all-rank steps 11--20 action mean is at least `5%` below ADR-207's matched
   `0.43144531` mean, not merely below ADR-222's own cold start;
3. heldout Top-10 soft IoU improves by at least `0.02` absolute over ADR-222 step
   0 (`0.10710190`), while full-bank oracle IoU does not regress materially;
4. all complete-graph gradient, WSA-call, finite-loss, and `<39 GiB` mechanics
   invariants continue to hold.

A one-percent win, a partition split, or a confidence interval spanning zero is a
scientific failure under the material-advantage rule. It does not license waiting
for step 50 or adding a selector, scalar gate, reduced decoder, or custom loss. The
20-step gate can reject a bad information contract; it cannot by itself prove a
next-generation final policy.

## Information-flow contract

The host, Future3D, and action streams execute synchronously in the released MoT
layer stack. AnyTouch, Sonata, and V-JEPA use the authenticated ADR-207 dense
evidence caches and enter through the accepted native posterior interface. The WSA
future stream can attend jointly with host/action tokens; factual training requires
exactly one typed WSA forward per optimizer transaction. Measurement-only causal
interventions cannot silently add another supervised loss.

This gate does not claim that standalone anchor quality implies better action. It
tests whether the high-quality anchor donor, posterior, future prediction, and
action objective remain jointly trainable.

## Source and storage identity

- LingBot source commit: `2838c1862bbec1ea47942fb61512130f635eb595`
- WSA source commit: `bfee742c585d5ee85722e658978111934c926ca3`
- full WSA checkpoint SHA-256:
  `29d789d9a97459e33ed95aa85fb6e0ec0661879789db090bb8cabc1edf6a9130`
- selective-class FSDP patch SHA-256:
  `3ba693ac4ac2158bf60756aaee067fbd368ae6e2770ab340838fa5b63bb226fa`
- final LingBot parallel source SHA-256:
  `d3233e5ec507a75c50778e10edb355e89d4d082b34ad989d9caccab51ce63c6c`
- DA3 runtime overlay manifest SHA-256:
  `54af863ae5d7c2bc0085fc38d9c09da669e6672b652a0be7dbe94b7d8a57afc2`
- full-modal runtime archive:
  `/mnt/picf-next/adr207/runtime-archives/picf-runtime-py312-cu128-fullmodal-v2-20260823.tar`;
- full-modal runtime archive SHA-256:
  `726e89b18d3b3755dc2a5c1958f5634da808b478c9901eba43db38b17a0cf7df`;
- persistent DA3-Large model:
  `/mnt/picf-next/adr218/assets/da3-large-1.1`;
- two/four-GPU launcher SHA-256 after the persistent DA3 path repair:
  `07be5555aa0f3c2bd23a073e055787fd76cd8ae73c3c28d59a84631d2b5a7559` /
  `570995f31377e646f99d7d84613f5b92911990a40fa9541d4a9bcbbe02d6858b`.

The persistent DA3 overlay contains the API's runtime-only dependencies
`moviepy==1.0.3`, `proglog==0.1.12`, `imageio-ffmpeg==0.6.0`,
`pycolmap==4.1.1`, and `evo==1.37.0`. The launcher verifies all 198 controlled
files before adding the overlay to `PYTHONPATH`. A real construction probe loaded
the local DA3-Large checkpoint into the released `DA3BackboneTeacher` with
304,373,760 frozen parameters, width 2048, and layers 11/15/19/23.

The selective CPU placement changes execution storage only. The CPU parameter set
must equal the shared embedding plus every parameter below
`model.qwenvl_with_expert.adr218_wsa_training_runtime.future.expert`, with declared
classes exactly `Future3DBlock` and `Future3DExpert`. Extra or missing CPU parameters
are rejected.

## Gate checklist

- [x] Exact upstream source/checkpoint identities are fail-closed.
- [x] Native patch, Muon hotfix, and selective-class offload replay in order.
- [x] Same-episode exact `t+4` observations feed the official DA3 teacher.
- [x] All four released WSA teacher/readout layers and loss terms are active.
- [x] WSA optimizer, scheduler, checkpoint, and resume state are explicit.
- [x] Dense coverage v1 artifacts reproduce under the current v2-capable loader.
- [x] Candidate-focused suite: 274 passed, 9 optional-path skips; selected
      official LingBot source suite: 34 passed, 1 optional CALVIN-source skip.
- [x] Real distributed forward completes with finite WSA, anchor, and action losses.
- [x] Backward reaches host, action, posterior/modal projections, and WSA.
- [x] Host, source, and WSA optimizers commit atomically; scheduler advances once.
- [x] Step 2 checkpoint is complete and cold-resumable.
- [x] The fresh 20-step curve is compared against the matched LingBot and
      ADR-207 controls.
- [ ] 50/100/250-step curves remain forbidden until ADR-222 passes its preceding
      matched material-advantage gate.
- [x] Step-0 and step-20 anchor visualizations are inspected by machine vision
      and a human.
- [x] The fixed-source WSA `future -> action` intervention closes the immediate
      regression diagnosis: blocking the edge improves validation/heldout action
      by 17.92/14.65 percent on identical posterior states and RNG.
- [ ] Same-object versus wrong-object interventions demonstrate correct
      non-visual evidence-to-row binding; causal reachability alone is not enough.

## Experiment ledger

### gate2-v1

Failed before model load. The checkout contained the approved WSA selective-class
offload patch but the old ADR-207 final-source digest was still required. This was
an identity-contract failure, not training evidence.

### gate2-v2

Failed before model load. A frozen v1 dense-coverage artifact was rebuilt as v2 and
then compared by dataclass identity. The replay now preserves the authenticated
artifact schema while still recomputing every source visit and digest.

### gate2-v3

Loaded all six LingBot checkpoint shards and built base FSDP2. It then failed because
the old placement validator allowed only the shared embedding on CPU. The validator
now requires the exact WSA parameter prefix and class topology in addition to the
embedding. Peak observed during host load was about 24.8 GiB per A100.

### gate2-v4

Loaded all six LingBot checkpoint shards, completed base FSDP2 construction, and
passed the exact selective-class CPU parameter-placement contract. It then stopped
before the first forward because the optimizer audit ran after `LinearLR`
construction. PyTorch scheduler construction had correctly changed the live
learning rate from the donor peak to the step-zero warmup value, while the static
optimizer audit still required the peak value. The audit now runs before scheduler
construction, followed by a separate scheduler-state audit. This was an audit
ordering defect, not training evidence; the corrected ordering passed 150 targeted
tests in the remote runtime.

### gate2-v5

Passed the corrected AdamW and scheduler audits after complete host/FSDP loading.
It then stopped before the first forward because the fixed Python runtime lacked
dependencies imported by DA3's public API (`moviepy`, then `pycolmap` and `evo`).
The missing official runtime requirements are now isolated in a persistent,
content-verified overlay. This was an environment-completeness failure, not
training evidence.

### gate2-v6

Rejected immediately by the inherited launcher because it treated a colon-separated
Python overlay list as one directory. The base launcher now validates every direct
directory in a standard `PYTHONPATH` list and rejects empty elements. This is
backward compatible with all single-overlay launchers and passed the same 151-test
targeted suite. No model or GPU work occurred.

### gate2-v7

Passed source/data verification, full host load, FSDP, WSA optimizer/scheduler,
both official DA3 model loads, and reached `step_started` on both ranks. The t+4
teacher input then failed because the typed CALVIN source observation exposes
`observation.images.image` and `observation.images.wrist_image`, while the wrapper
looked up WSA's default LeRobot top/wrist names. The wrapper now requires an
explicit two-key dataset camera contract; the CALVIN runner supplies its frozen
host keys. No fallback or image substitution was added. The adapter and ambiguity
guards passed 159 targeted tests.

### gate2-v8

Passed the complete rank-local forward, objective, backward, and optimizer
transaction at step 1. Both ranks consumed exactly one factual two-view DA3
teacher target, reported finite WSA losses (`0.84375` and `0.90625`), finite
host/action gradients, and finite gradients for all 435 trainable VidEoMT
tensors. The WSA scheduler and every model optimizer advanced exactly once.

The run failed during the step-2 forward when rank 1 needed a 194 MiB FSDP
all-gather with 39.37/39.49 GiB already occupied. Persistent posterior state is
detached and cloned on publication, so an attached recurrent graph is not the
cause. Step 1 also emitted many whole-model `torch.compile` graph breaks and
per-layer recompilations; after first-step optimizer-state materialization, that
execution cache left insufficient headroom for the next full forward. This is a
steady-state deployment failure, not evidence about the learning curve.

### gate2-v9

Rejected before model load because the original ADR-221 profile correctly froze
the released whole-model compile backend. A temporary separately named execution
profile was introduced for the diagnostic rather than weakening that contract.

### gate2-v10

Rejected before model load because the inherited shell launcher forwarded WSA/DA3
assets only for the original profile name. No GPU work occurred.

### gate2-v11

The exact model with whole-model compile disabled reached the first backward, but
FlexAttention used its dense backward fallback and rank 1 OOMed while requesting
another 1.37 GiB. This is a stronger failure than v8 and proves that the released
compile path is part of LingBot's memory-efficient attention execution. The
temporary uncompiled profile was removed from the production source.

### gate2-v12

Restore the original frozen ADR-221 profile and released compile path. Reuse the
existing ADR-209 `selective-embedding-trainable-vision-offload` FSDP2 placement,
which preserves trainability and all optimizer mathematics while moving only idle
vision parameter/gradient storage to CPU.

The run was rejected after FSDP construction and before the first data step. The
launcher declared trainable-vision offload, but ADR-221's source-verification branch
had prepared only the native, Muon-alignment, and selective-WSA-class overlays. The
prepared `torch_parallelize.py` therefore contained the WSA class dispatch but not
the trainable-vision dispatch. The parameter-storage validator correctly rejected
that source/placement mismatch; no loss or learning evidence was produced.

### gate2-v13

Compose the two existing execution mechanisms without changing model mathematics:
replay the complete six-overlay trainable-vision chain first, then apply an exact
context adaptation of the already approved selective-class offload. The seven-layer
replay is pinned to parallel-source SHA-256
`d7d3ba3ced4ff53d82f34a67c6541afa0f5b011a3acd33fb3d3e89cfee8b7f3f` and
Muon SHA-256
`98a1de761c2f8f8e1c041d3c0e4b053b2aef2df3b84a7c34fc1c5973c10553b5`.
Its isolated checkout is
`/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr221-combined-offload-v1`.
The storage validator now requires the exact union of shared embedding, complete
trainable visual root, and declared WSA-class parameter prefixes on CPU; all other
FSDP shards must remain on CUDA. This is the final dual-40-GiB placement gate.

The exact combined source passed one full update. Every 435 VidEoMT tensor had a
finite gradient, host/action gradients were finite, the factual WSA objective was
consumed once per rank, and all optimizers completed. Step 1 took 71.52 seconds;
after graph release each rank retained about 20.52 GiB allocated and 21.31 GiB
reserved. The same-seed action losses were 1.265625 and 0.87890625, and factual WSA
losses were 0.84375 and 0.90625.

Step 2 nevertheless failed at the same 194 MiB WSA-host FSDP all-gather after
optimizer-state materialization, with rank 1 at 39.36/39.49 GiB. The execution-only
vision overlay therefore recovered roughly 10 GiB around the first backward but did
not make steady-state training reliable on two 40-GiB devices. This rejects dual
A100-40G production for the exact model. It does not reject the model's learning
hypothesis, because no second update or matched curve exists. The next authorized
gate is the identical model and objective on four A100-40G devices using the already
frozen `native-query-posterior-4gpu-30k-v1` stream and dense-evidence contracts.

### gate2-v14

Restore the later frozen full-modal runtime archive on a fresh four-A100-40G host
and move the launcher's DA3 default from the ephemeral `/root` path to the
content-identical persistent model under `/mnt`. The DA3 model, DA3 config, WSA
checkpoint, prepared LingBot parallel source, and Muon source all matched their
declared SHA-256 identities. The candidate-focused suite passed `274` tests with
`9` optional-path skips; the selected official LingBot source passed `34` tests
with the optional CALVIN checkout test skipped.

The exact four-rank graph completed two consecutive real optimizer updates. Every
rank reported finite host gradients, finite source gradients, and gradients for all
`435/435` trainable VidEoMT tensors. AnyTouch, proprioception, Sonata, VidEoMT
queries, and V-JEPA were present in the host modality list. Each WSA transaction
contained exactly one factual call, no measurement call, and scheduler steps `1`
then `2`.

Step 1 was dominated by first-use compilation and took `179.49 s`. Step 2 took
`104.48 s`. The step-2 action losses by rank were
`0.85547/0.62500/0.86328/0.99609`; WSA losses were
`0.71875/0.78516/0.72266/0.69141`. The largest recorded step-2 CUDA reservation
was `38.584 GiB`, below the declared `39 GiB` gate, and graph release returned all
ranks to at most `11.039 GiB` reserved. This accepts four-A100-40G steady-state
mechanics and neither rejects nor confirms the learning hypothesis. A fresh
20-step run is the next numerical gate.

## Meaning of full-modal object binding

The ADR-221 graph must not equate three different claims:

1. VidEoMT emits the explicit 2D object-query masks. Its complete 200-query bank
   seeds the same-index recurrent posterior rows without a second decoder.
2. AnyTouch, Sonata, and V-JEPA evidence is present as complete typed sensor-token
   streams. Only the audited linear width/metadata projections precede the shared
   LingBot transformer. Posterior rows may attend every valid sensor token, so
   cross-modal object assignment is an implicit soft binding learned by the large
   shared host rather than a separate small-model segmentation decision.
3. Correct object-level binding is an empirical claim. Presence in
   `host_modality_names`, finite gradients, and an empty explicit `row_bindings`
   ledger prove neither same-object alignment nor causal utility. The native
   VidEoMT profile intentionally leaves that legacy ledger empty and stages the
   complete same-index query posterior directly.

Therefore non-visual modalities are not expected to produce independent 2D masks.
Sonata points, AnyTouch contact tokens, and V-JEPA grid tokens must instead alter
the appropriate visual-object posterior rows. Acceptance requires frozen-input
same-object versus wrong-object/shuffled evidence interventions, modality omission,
and posterior-row consistency measurements. Until those tests pass, the accurate
status is "all modalities are active and can bind implicitly", not "all modalities
are correctly segmented or bound". Task language remains outside posterior-row
updates by design so that the object inventory is task-independent; language and
posterior rows meet in the action path for task-conditioned selection.

### Historical source-versus-action separation

Cold Top-10 ranking around `0.11` must not be read as proof that the complete
VidEoMT source cannot represent CALVIN objects. The accepted ADR-210 warm replay
raised validation Top-10 binary/soft IoU from `0.108525/0.099078` to
`0.495634/0.476972`, and heldout from `0.119299/0.110459` to
`0.470446/0.453859`, while full-bank oracle IoU stayed near `0.82--0.83`. Its
object-shaped masks show that temporal recurrence can make the correct source
queries rank highly.

The same ADR-210 step-100 causal action audit did not show adoption. Heldout warm
versus cold action loss was `0.32742059` versus `0.32713243` (warm was 0.088
percent worse); validation was `0.31143466` versus `0.31128670` (warm was 0.048
percent worse), with confidence intervals spanning zero. Warm posterior state
fixed source ranking but had approximately no causal action benefit. Thus the
dominant historical bottleneck is not merely source geometry or lifecycle: it is
whether the released action path learns to consume object-persistent evidence.

## Frozen 20-step comparison gate

The contract files and the semantic artifacts inside them have separate hashes.
For the four-GPU contract, the stream/evaluation file SHA-256 values are
`4edd2fa504a8fb5f4c3b92cc612816be47d84dc40d9943bf71494008568b74fa`
and
`bb9eefc24d77a090ca3131921077918196818b32dc9b75d1c481bffa6afd57cc`.
Their embedded semantic artifact identities are
`0dfde7a9ae75308a5062269c161420d151f6e8d74d595a401b154cb85f962c7c`
and
`832e69cace670b20efd6449173cd2b913c03180c3632ebc01e29637213643f9b`.
The candidate, ADR-207 native-query PICF control, and official LingBot control
share the semantic identities. File hashes must not be compared to embedded
artifact hashes as if they were the same identity layer.

The pre-existing controls establish these frozen reference values:

- official LingBot `v19` step-0 action loss: validation `0.51338465`, heldout
  `0.50168026`; rank-0 training mean over steps 1--20: `0.49140625`;
- ADR-207 PICF `v18` step-0 action loss: validation `0.45499196`, heldout
  `0.48309685`;
- ADR-207 PICF `v18` step-20 action loss: validation `0.36302275`, heldout
  `0.38541188`;
- ADR-207 PICF `v18` all-rank training mean over steps 1--20: `0.46706543`;
- ADR-207 PICF `v18` heldout Top-10 soft IoU is `0.10710190` at step 0 and
  `0.10896427` at step 20. The near-zero change defines the historical
  no-improvement behavior, while the full-bank oracle soft IoU is about `0.79`.

The fresh ADR-221 20-step run is promotable only if all of the following hold:

1. every rank completes all updates with finite host and source gradients, all
   `435/435` trainable VidEoMT tensors present, one factual WSA call and no
   measurement call per transaction, and peak reservation below `39 GiB`;
2. WSA loss is finite and has a decreasing 5-step-window trend rather than a
   one-point endpoint improvement;
3. step-20 validation and heldout action losses improve over ADR-221 step 0 and
   are not more than 5 percent worse than the matched ADR-207 PICF `v18`
   step-20 values;
4. the all-rank action mean over steps 11--20 is not more than 5 percent worse
   than the identical-window ADR-207 PICF `v18` control;
5. heldout Top-10 soft IoU improves by at least `0.02` absolute over the exact
   ADR-221 step-0 snapshot, which exceeds the historical step-0--20 fluctuation;
6. post-compilation step time is no more than 2.5 times the official LingBot
   steady-state reference unless a measured component-level cost explains the
   excess.

Failure of items 3--5 rejects direct WSA-to-action promotion. The next permitted
operation is the already typed, measurement-only `future -> action` attention-edge
intervention on frozen inputs. It tests whether uncalibrated Future3D attention is
the action regression cause. It is not permission to add a new gate, auxiliary
head, reduced decoder, or unreviewed loss. A 250-step or 30k run is forbidden
until this decision is closed.

### gate20-v15 result

The fresh run
`/mnt/picf-next/adr221/runs/full-source-wsa-4gpu-gate20-20260826-v15`
completed all 20 updates and both fixed 102-sample snapshots. Contract identities
were unchanged between step 0 and step 20. All 80 rank-steps had finite host and
source gradients, all `435/435` VidEoMT gradient tensors, exactly one factual WSA
call and no measurement call. Peak CUDA reservation was `38.816 GiB`. Mean
post-first-step time was `33.663 s`, about 2.05 times the official LingBot steady
state and below the 2.5-times limit.

WSA learned monotonically. Its successive five-step means were
`0.681250`, `0.505664`, `0.438867`, and `0.382324`. This accepts WSA objective
execution but does not accept the combined policy.

The combined policy failed the matched action gate:

- ADR-221 all-rank action mean at steps 11--20 was `0.47216797`, versus
  ADR-207 v18 `0.43144531`; the candidate is 9.44 percent worse and exceeds the
  registered 5 percent tolerance.
- ADR-221 step-20 validation/heldout action losses were
  `0.38111788/0.41806928`, versus ADR-207 v18
  `0.36302275/0.38541188`. Validation is barely within 5 percent, while heldout
  is 8.47 percent worse.
- The candidate did learn relative to its own poor step-0 action snapshot
  (`0.58909697/0.60707721`), so this is not a dead optimizer or inert host. It is
  a matched relative regression after adding the direct WSA surface.

The combined policy also failed the anchor-ranking gate. Heldout Top-10 soft IoU
changed from `0.10710190` to `0.10695245`; validation changed from `0.10377232`
to `0.10016136`. Neither approaches the required `+0.02`. Full-bank heldout
oracle soft IoU changed only from `0.78958092` to `0.79084711`. Human inspection
of rank-0 ordinal 0 confirms the same result: geometry-aware masks for the button,
drawer, light, switch, and blocks remain available in the full bank, but the
model-ranked Top-10 contains essentially the same wrong queries at steps 0 and 20.

Decision: direct ADR-221 promotion is **REJECTED**. No 50/100/250/30k continuation
is authorized. The WSA loss improvement does not repair winner selection and the
direct Future3D/action composition degrades the matched action curve. The next
experiment remains the typed measurement-only `future -> action` edge intervention;
it must be evaluated on fixed inputs before any architecture repair is proposed.

### diagnostic-v16

Rejected by the launcher before model load because the inherited production guard
required at least two optimizer steps. The guard now retains that requirement for
all training runs and permits one step only when the explicit, measurement-only
ADR-221 edge-diagnostic mode is active. This run produced no model evidence.

### diagnostic-v17

Rejected fail-closed during step-0 evaluation. The first diagnostic implementation
ran the stateful VidEoMT source twice, so ranks 0 and 2 did not reproduce exact
source/prior/posterior state. That is an invalid paired-experiment construction,
not evidence for or against the architecture. The corrected diagnostic generates
the source once and reuses the identical typed `host_batch` object for the blocked
host replay, matching the established ADR-207 fixed-source intervention primitive.

### diagnostic-v18

Rejected fail-closed on sample 0. It reused the exact source `host_batch`, but it
compared the standard source-plus-host forward against a host-only blocked replay
after resetting both to the same seed. Because the train-mode source forward had
already consumed RNG in only the standard branch, ranks 0 and 2 produced unequal
prior/posterior tensors. This is a second paired-measurement construction failure,
not model evidence.

### diagnostic-v19

Completed on four A100-40G devices. Source generation was single-shot. The factual
and blocked-host replays started from the same runtime-buffer snapshot, identical
typed `host_batch`, and identical dedicated RNG seed. Posterior tensors were
bitwise equal for every sample and source/host batch identities were unchanged.

On validation (`n=34`), factual direct-edge action loss was `0.58909697`; blocking
the edge reduced it to `0.48354205`, an absolute `-0.10555492` and relative
`-17.92%`, with 91.18 percent of samples improved. On heldout (`n=68`), factual
loss was `0.60707721`; blocked loss was `0.51815257`, an absolute `-0.08892463`
and relative `-14.65%`, with 86.76 percent of samples improved.

This is a measured causal rejection of ADR-221's direct Future3D-to-action
composition, not an endpoint correlation. No 50/100/250/30k continuation of
ADR-221 is authorized. The missing step-20 checkpoint is not regenerated because
the fixed-input diagnosis is already decisive and rerunning the rejected graph
would add no scientific information.

## Early-stop policy

Stop immediately for non-finite loss/gradient, missing modality gradients, WSA loss
not consumed exactly once, optimizer/scheduler drift, OOM above the declared 39 GiB
limit, or a materially inferior matched curve without a predeclared causal reason.
Do not rescue a failed curve by adding an unreviewed auxiliary head or weakening an
upstream method.

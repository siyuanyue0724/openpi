# ADR-210: Causal Warm-Posterior Ranking Audit

Date: 2026-08-24 CST

Status: source-ranking root cause identified; causal-warm action gate passed a
two-step four-GPU runtime smoke.  A formally declared 30k-budget process is
running, but continuation beyond step 100 remains unauthorized until the gate
passes.

## 1. Decision summary

The adapted complete VidEoMT source is not a failed single-frame mask model.
It is a recurrent object-query model whose autonomous object ranking depends
strongly on causal query history.  ADR-207 evaluated its registered fixed bank
only in `cold_reset` mode.  That protocol correctly measured cold-start
behavior, but it did not measure the persistent-posterior operating regime
claimed by PICF.

On the same adapted checkpoint and the same current RGB/physical targets:

| Fixed-bank partition | Eligible samples | Cold top-10 binary IoU | Four-past-frame warm top-10 binary IoU | Ratio |
|---|---:|---:|---:|---:|
| validation | 33 | 0.108525 | 0.495634 | 4.57x |
| heldout | 61 | 0.119299 | 0.470446 | 3.94x |

The corresponding oracle-all-200 binary IoU rises only from `0.816833` to
`0.835826` on validation and from `0.818167` to `0.833349` on heldout.  Thus
the principal change is not the existence of object-shaped masks.  It is the
calibration and identity stabilization needed to rank useful queries without
an oracle.

This evidence rejects the earlier working diagnosis that the adapted source
had simply lost its object-ranking capability.  It does **not** prove that
PICF improves action, cross-modal binding, long-delay behavior, or CALVIN
rollouts.  A matched warm-state action evaluation is now the mandatory next
gate.

## 2. Immutable evidence

Checkpoint:

```text
/mnt/picf-next/adr202/assets/videomt-calvin-adapted-step250-v1.pt
SHA-256: 4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d
```

Historical-protocol factorization:

```text
/mnt/picf-next/adr209/audits/videomt-temporal-ranking-v1/report.json
local: evidence/adr209_videomt_temporal_ranking_v1/report.json
```

Complete fixed-bank causal audit:

```text
/mnt/picf-next/adr209/audits/videomt-fixed-bank-causal-ranking-v1/report.json
local: evidence/adr209_videomt_fixed_bank_causal_ranking_v1/report.json
```

Normative tools:

```text
tools/audit_videomt_temporal_ranking_protocol.py
tools/audit_videomt_fixed_bank_causal_ranking.py
```

Both tools are read-only.  They add no parameter, selector, decoder, loss,
threshold, padding frame, task input, or label-conditioned model input.  They
call the complete adapted source model and use physical masks only for metrics
and rendering.

## 3. Factorized experiment

Let the released learned cold queries be (Q_0), current RGB evidence be
(X_t), the complete source update be (F_\theta), and its released query
propagator be (P_\theta).  The source recurrence is represented abstractly as

\[
Q_t = P_\theta(F_\theta(Q_{t-1}, X_t)) + Q_0.
\]

For each query (i), the released class head supplies foreground logit
(s_{t,i}), and the complete mask path supplies (M_{t,i}).  The diagnostic
top-10 set is

\[
K_t = \operatorname{TopK}_{i,10}\;p_\theta(\mathrm{object}\mid s_{t,i}).
\]

No top-k operation is active in PICF training: all 200 rows enter LingBot.
Top-10 is used only to measure whether the source can autonomously expose its
useful object proposals.

The first audit separated six statistics on the four exact Stage-D heldout
windows:

| Mode | Causal at decision? | Current mask | Ranking statistic | Top-10 binary IoU |
|---|---|---|---|---:|
| cold first | yes | first frame | first-frame class | 0.095973 |
| future-ranked first | no | first frame | mean over current + 4 future | 0.396860 |
| historical future window | no | all five | mean over current + 4 future | 0.426993 |
| cold last | yes | last frame | last-frame cold class | 0.093726 |
| causal warm last | yes | last frame | final recurrent class | 0.441245 |
| causal history-ranked last | yes | last frame | mean over 4 past + current | 0.377545 |

The historical report value was `0.429273`; the direct source re-execution was
`0.426993`.  Oracle binary IoU reproduced exactly, while top-10 differed by
`-0.002280`, consistent with FP16/top-k boundary sensitivity.  The report keeps
the exact deltas rather than claiming bitwise identity.

Two conclusions follow:

1. Future-window averaging explains why the old Stage-D metric looked much
   stronger than the ADR-207 cold metric.  The old number must not be cited as
   cold causal evidence.
2. Future access is not necessary to recover the effect.  Four strictly past
   frames followed by the current frame produce an even higher `0.441245`.
   Averaging past class scores reduces it to `0.377545`; therefore the gain is
   carried by the learned recurrent state, not by an evaluator-side EMA.

## 4. Complete fixed-bank result

The second audit reused the exact 102-sample ADR-207 step-0 fixed anchor report.
Eight samples occur at transition indices zero through three and therefore
have fewer than four real predecessors.  They were listed explicitly and
excluded from the warm comparison.  No first-frame duplication or cross-reset
padding was permitted.  The remaining 94 samples were distributed over four
A100s.

| Metric | Validation cold | Validation warm | Heldout cold | Heldout warm |
|---|---:|---:|---:|---:|
| oracle soft IoU | 0.791366 | 0.813154 | 0.792297 | 0.811793 |
| oracle binary IoU | 0.816833 | 0.835826 | 0.818167 | 0.833349 |
| top-10 soft IoU | 0.099078 | 0.476972 | 0.110459 | 0.453859 |
| top-10 binary IoU | 0.108525 | 0.495634 | 0.119299 | 0.470446 |
| top-10 Recall@0.5 | 0.129568 | 0.564784 | 0.125000 | 0.512324 |
| oracle-query mean P(object) | 0.078745 | 0.528868 | 0.074392 | 0.499460 |

Cold re-execution agrees with the original FSDP/BF16 report within `0.002551`
absolute oracle IoU and `0.000614` top-10 IoU.  This verifies that the gain is
not caused by a new sample bank or target implementation.

Original-resolution visual review confirms the aggregate result:

- cold top-10 panels mostly select background, robot-edge, or broad table
  fragments despite strong useful masks elsewhere in the 200-query bank;
- warm top-10 panels recover coherent blue/red/pink blocks, button, drawer,
  slide, switch, and plank masks;
- tiny or heavily occluded led/plank regions still fail in some examples;
- warm masks remain object-shaped and are not fixed patch-grid tiles;
- the result is not perfect segmentation and must not be reported as such.

## 5. Why ADR-207 action evidence remains inconclusive

ADR-207 training itself did preserve state.  Across four ranks and 200 steps,
the rank journals contain four initial reset records and 796 continuation
records.  The first reset records have mean source object probability
`0.017783`; state ages one through four have mean `0.218618`.  This is
consistent with the causal ranking audit.

However, the registered heldout action snapshots are explicitly
`state_mode=cold_reset`.  They reset both source and host posterior before each
sample.  Their small heldout advantage and validation crossover therefore do
not identify the claimed persistent-memory benefit.  Conversely, the lower
training loss of old state ages cannot prove that benefit because state age is
confounded with optimizer step and sample order.

The primary causal action estimand is the same-checkpoint paired difference

\[
\Delta_{\mathrm{state}} =
\mathbb{E}_{e,t}\left[
L_{\mathrm{PICF}}(a_t\mid X_{t-k:t}, u)
-L_{\mathrm{PICF}}(a_t\mid X_t, u)
\right],
\]

under the same source-disjoint episodes, current model inputs, prompts,
actions, flow-noise seeds, timesteps, checkpoint step, and sample order.  Only
the causally available four-frame state differs.  This identifies whether the
claimed persistent posterior actually helps action rather than merely making
better masks.

A second system-level comparison is

\[
\Delta_{\mathrm{system}} =
\mathbb{E}_{e,t}\left[
L_{\mathrm{PICF}}(a_t\mid X_{t-k:t}, u)
-L_{\mathrm{LingBot}}(a_t\mid X_t, u)
\right].
\]

The released LingBot baseline has no PICF persistent posterior state.  The
evaluator must not invent one.  Consequently, this second comparison measures
the utility of the complete PICF system and its larger causal information set;
it is not an isolated state ablation.  Both estimands are required and must be
reported separately.

## 6. Required next implementation

The evaluator correction is implemented without adding a learned module:

1. Build a manifest-bound warm fixed bank from real predecessor frames only.
   The 94 eligible samples are published; the eight samples with fewer than
   four real predecessors are explicitly excluded.  Equal-rank padding is
   permitted only for collective alignment and is never published.
2. Replay the same past observations through the candidate's complete VidEoMT
   recurrence and LingBot posterior.  Commit only states available before the
   current action.
3. Compare against the same PICF checkpoint in registered `cold_reset` mode
   and the released current-frame LingBot baseline.  Do not fabricate a
   recurrent baseline network that does not exist upstream.
4. Join the warm current frame to the cold snapshot and fail closed unless its
   sample key, RGB digest, model-input digest, prompt, action target, flow-noise
   seed, and timestep are identical.
5. Keep the existing cold action curve at steps `0,20,100`.  Run the expensive
   full-bank warm gate once at step `100`, because step-0 source recurrence is
   already established by the immutable 94-sample audit.  Never average warm
   and cold state modes.
6. Add source warm-anchor summaries to each candidate snapshot, using the
   current-frame final recurrent score rather than a hand-built temporal EMA.
7. Launch the formal process with its intended 30k budget so a passing gate can
   continue without another expensive model load.  At step 100, write the
   external `STOP` marker unless candidate warm action beats both cold PICF and
   current-frame LingBot on validation and heldout by at least 2% relative loss
   reduction with a paired-bootstrap upper confidence bound below zero.  In the
   implemented contract, "matched LingBot" means the frozen released
   current-frame baseline, not a fictional warm baseline.

Implementation evidence:

```text
adr210/run_causal_warm_action_gate_4gpu.sh
src/picf_next/lingbot_native/entity_evaluation_plan.py
src/picf_next/videomt_exact/lingbot_joint.py
tools/run_lingbot_vla2_task_independent_full.py
```

The local and exact cloud runtime suites each pass `159` selected tests.  The
runtime source is frozen read-only at:

```text
/mnt/picf-next/adr207/source-freezes/native-query-posterior-v21
Git freeze commit: 9d3413e3694fba3f0cd421fae8ef6e71d4e0964d
```

The v21 four-GPU smoke completed two real optimizer steps.  Step 1 took
`99.58 s` including first-graph compilation; step 2 took `38.72 s`.  All four
ranks emitted finite official action losses, peak allocated memory remained
within the 39 GiB contract, and no rank, OOM, or collective failure occurred.

Formal process and log:

```text
/mnt/picf-next/adr210/runs/causal-warm-action-gate-v21-4gpu-30k-20260824-0345
/mnt/picf-next/adr210/logs/causal-warm-action-gate-v21-4gpu-30k-20260824-0345.log
```

The runtime manifest declares 30,000 steps, a global batch of four, metrics
every 100 steps, visuals every 250 steps, and checkpoints every 2,000 steps.
It registers all three dense evidence families (`anytouch`, `sonata`, and
`vjepa`) with 120,102 records each, the complete trainable 6.264B LingBot host,
and all 435 trainable VidEoMT tensors.  Step-zero evaluation reproduced the
historical ADR-207 v18 sample keys, partitions, tasks, action losses,
model-input digests, flow-noise seeds, and timesteps exactly.  Its means remain
`0.4830968520` heldout and `0.4549919577` validation.

A fail-closed operational guard is persisted at:

```text
/mnt/picf-next/adr210/tools/guard_causal_warm_action_gate.sh
```

After rank zero commits its step-100 training journal, the guard pre-creates
`RUN/STOP`.  It removes that marker only when the strict paired comparator
returns `AUTHORIZE_30K`; a missing artifact, schema mismatch, comparison error,
non-significant result, or regression leaves the stop in place.  The guard is
outside every model, data, loss, and optimizer path.

The original two-second shell polling loop had an operational race: the runner
could publish the final warm snapshot and observe the pre-armed stop marker
before the external Python comparator removed it.  It was replaced before
step 100 by `guard_causal_warm_action_gate_fast.py`.  The replacement preloads
the unchanged comparator, polls only the active publication window at low
latency, briefly sends `SIGSTOP` to the four identified torchrun workers after
both direct snapshots exist, computes the registered 10,000-replicate report,
durably resolves `STOP`, and resumes every worker.  A four-process pause/resume
probe passed, and one complete synthetic gate took `1.37 s`; the lower-level
94-sample three-comparison bootstrap took `0.80 s`.  This changes no model,
sample, loss, threshold, or optimizer state and prevents both false release and
false termination.

The declared 30k budget is an operational optimization, not scientific
authorization.  Continuation beyond step 100 requires the paired warm-action
gate; broader claims still require full-modal causal interventions, recovery,
and closed-loop CALVIN evaluation.

The first in-run cold snapshot at step 20 completed on all 102 registered
samples.  Its action means are `0.3905101103` heldout and `0.3631663603`
validation.  Relative to matched LingBot at the same step, these are reductions
of `1.08%` and `0.60%`; relative to ADR-207 v18 PICF, they are regressions of
`1.32%` and `0.04%`.  All four paired-bootstrap 95% confidence intervals cross
zero.  Step 20 therefore establishes runtime and curve continuity only.  It is
neither positive nor negative scientific evidence for the warm-posterior
hypothesis, and it does not modify the pre-registered step-100 gate.

## 7. Engineering failures encountered in this audit

The following zero-forward startup failures are retained rather than hidden:

1. repository root was used instead of `repo/src` in `PYTHONPATH`;
2. the persistent VidEoMT dependency overlay was initially omitted;
3. `torch.device("cuda")` was not normalized to an explicit index;
4. the obsolete v2 sidecar reader was incompatible with the all-source v5
   manifest and was replaced by the exact Stage-D loader stack;
5. the typed observation hooks expose a mixed-dtype FP16 autocast ABI issue.
   The audit did not weaken that shared ABI; it called the complete official
   source model directly and consumed only its official prediction tensors;
6. eight fixed samples lacked four real predecessors.  Padding was rejected
   and the samples were explicitly excluded from the warm estimand.
7. the first four-rank CLI smoke found a mismatched disabled-cache family name
   (`future_latent` versus the registered `future_latent_cache`).  The caller
   was corrected rather than weakening the fail-closed validator with an alias.
   The failure occurred before model loading or any optimizer step.
8. the initial external gate used a two-second polling interval and could lose
   the publication-to-stop-check race even after a passing result.  The live
   process was replaced at step 68 with the preloaded worker-pause guard above;
   training remained uninterrupted and no stop marker was active during the
   replacement.

None of these failed attempts executed an optimizer step or wrote a model
checkpoint.

## 8. Upstream evidence and hypothesis boundary

The implementation deliberately separates source-faithful learned primitives
from the PICF-specific composition claim:

- LingBot-VLA 2.0 supplies the released 6B VLA host, action expert, and its
  current/future dual-query distillation path.  Its official design already
  uses DINO-Video and depth teachers for predictive dynamics and geometry; PICF
  must therefore beat the released host rather than claim those capabilities as
  novel.
- The official PMT repository supplies the complete EoMT/VidEoMT family and its
  learned object-query/mask machinery.  ADR-207/210 uses the complete adapted
  source recurrence and final learned query score; it does not replace it with
  SAM, a connected-component heuristic, a grid selector, or a newly invented
  lifecycle head.
- The official V-JEPA 2 repository supplies the frozen predictive video
  representation used by the registered evidence cache.  The active cache is
  produced by the V-JEPA 2.1 dense-feature implementation, whose released
  recipe adds dense predictive loss and deep self-supervision for temporal
  consistency.  Its presence proves neither object identity nor action benefit
  in PICF; those effects remain intervention targets.

In the active graph, each native VidEoMT query index seeds the same LingBot
posterior-row index.  AnyTouch, Sonata, V-JEPA 2.1, proprioception, current RGB,
language and the previous posterior are then integrated by the shared LingBot
transformer.  The width/metadata projections are typed linear adapters, not
semantic selectors.  There is no auxiliary lifecycle model, learned top-k
selector, cross-modal ownership head, or reverse host-to-VidEoMT decoder.  This
is a deliberate implicit-binding hypothesis: it keeps semantic binding inside
the large shared host, but code-level gradient connectivity cannot establish
that the model learned the correct same-object relation.  Warm action and
full-modal interventions are therefore scientific requirements, not optional
debugging.

Official sources reviewed on 2026-08-24:

```text
https://github.com/robbyant/lingbot-vla-v2
https://github.com/tue-mps/pmt
https://github.com/facebookresearch/vjepa2
```

The donor repositories do **not** prove the PICF composition: carrying the
VidEoMT causal query posterior into the LingBot action path, binding optional
modalities to that state, and obtaining lower action loss.  That is the single
system hypothesis tested at the step-100 warm gate.  Source fidelity raises its
prior plausibility; only the matched experiment can validate it.

## 9. Current verdict

PICF should **not** be abandoned on the basis of the ADR-207 cold anchor plots.
They measured the wrong operating regime for the recurrent source.  The new
evidence materially restores the plausibility of the persistent-object
posterior hypothesis.

PICF is still not proved superior.  The strongest honest statement is:

- complete object geometry: strong;
- autonomous object ranking after short causal history: materially strong;
- cold-start ranking: weak;
- warm cross-modal binding: unproved;
- warm action advantage over matched LingBot: unproved;
- long-horizon memory and closed-loop deployment: unproved.

The architecture is now executing its one corrected matched warm-action test.
It is not authorized to continue beyond that gate yet.

## 10. Step-100 warm-evaluation cache-contract failure and v22 repair

The v21 process completed optimizer step 100 and the complete cold fixed-bank
evaluation without OOM, NaN, collective failure, or model exception.  It then
failed before the first causal-warm model forward with:

```text
RuntimeError: runtime CALVIN sample has no canonical dense-evidence identity
```

This is an engineering failure, not positive or negative model evidence.  The
v21 dense-evidence plan covered the 120,000 frozen training visits and the 102
current evaluation frames.  The new warm estimand additionally consumes four
real predecessor frames for each of 94 eligible items.  Exact enumeration on
the canonical physical-event axis found 376 history visits, 376 unique source
frames, zero overlaps with the old cache, and a valid canonical key for every
frame.  No warm forward or warm action sample executed in v21.

The following apparent fixes are forbidden because they change the estimand or
hide missing evidence:

- zero AnyTouch, Sonata, or V-JEPA rows on history frames;
- repeat the current frame as history;
- disable cache coverage checks;
- treat evaluation history as training evidence;
- rebuild or rewrite the already authenticated 120,102-record primary caches.

The v22 repair generalizes the coverage artifact instead.  Schema v2 records
evaluation target count, unique evaluation record count, history length,
history visit count, and the ordered history-visit digest separately.  It is
backward compatible with authenticated v1 plans.  The exact CALVIN artifact is:

```text
evaluation targets: 102
eligible warm targets: 94
evaluation history transitions: 4
evaluation history visits: 376
unique evaluation records: 478
training records/visits: 120000 / 120000
total unique records: 120478
coverage artifact SHA-256: d0ddc3964aae76d400926410e44fcd6823a4006dd6ca2352685b7b839a15daed
coverage file SHA-256: d683e81270a0b89d067d79917c17ec42a4f15f74605a8eb06aa5bba558cb0b57
```

Three authenticated supplement caches encode only the 376 identities absent
from the v1 donor, using the unchanged production AnyTouch, Sonata, and
V-JEPA 2.1 builders.  At runtime the existing zero-copy cache-view primitive
composes the v1 primary bank and v2 supplement into the exact ordered 120,478
record view.  A dedicated preflight reads every one of the 470 causal-warm
frame visits, authenticates shard data, validates observation timestamps and
materializes the unchanged typed shared-host bridge before another training
launch is allowed.

The v22 launcher also corrects the continuation boundary: the process has a
real 30,000-step limit.  The external step-100 gate temporarily writes `STOP`
and removes it only after authorization, allowing the same optimizer process
to continue rather than terminating unconditionally at step 100.

The final immutable runtime is v23 at
`/mnt/picf-next/adr207/source-freezes/native-query-posterior-v23`, freeze commit
`22701a2601a579ea38dea17c6c47684474307fc2`.  Relative to v22 it changes only
the operational guard import root: comparator and guard now resolve from the
same immutable tree, preventing a v22-training/v21-decision mixture.

The first v23 launch exited before model loading because the external guard
created `RUN/audits` before the runner's empty-root preflight.  This executed no
forward or optimizer step.  The guard now waits for the runner's immutable
`run_manifest.json` before writing inside the run root; the supervisor itself
can still be started immediately and no longer races preflight.

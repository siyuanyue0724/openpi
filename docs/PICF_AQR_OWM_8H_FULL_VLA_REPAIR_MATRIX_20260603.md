# PICF-AQR-OWM 8H Full VLA Repair Matrix

Status legend:

- `[ ]` not started
- `[~]` running / partially checked
- `[x]` done and accepted
- `[!]` tested and rejected / intentionally deferred

Date: 2026-06-03

This document is the execution contract for the requested 8-hour gate.  It
covers every previously proposed VLA training-system point, separates what has
already been tested from what still needs a controlled run, and prevents
repeating F8/F9 variants that have already been rejected.

## Execution Ledger

The table below is the source of truth for this gate.  A method is not counted
as "done" merely because it was discussed; it must either be wired into code and
tested, or explicitly rejected with a reason.

| Area | Concrete item | Status | Evidence / next check |
| --- | --- | --- | --- |
| Data mixing | task-uniform bucket sampler | `[x]` deployed | F8/F9 and G10a control |
| Data mixing | temperature sampler `q_b ∝ N_b^0.5` | `[!]` tested and rejected | G10b action rebounded |
| Data mixing | trajectory-proportional sampler | `[!]` tested and rejected | G10c action rebounded |
| Data mixing | explicit bucket-ratio spec | `[x]` deployed | parser/audit exists; no new run unless custom ratios are requested |
| Logical batch | K4 logical update through DDP + accumulation | `[x]` deployed | G10 all branches |
| Logical batch | without-replacement bucket choice per logical step | `[x]` deployed | startup telemetry and sampler audit |
| Loss scale | per-bucket logical loss normalization | `[x]` deployed | G10 all branches |
| Loss scale | per-bucket action EMA scalar | `[!]` rejected | F9b action did not improve |
| Telemetry | per-bucket metrics in `metrics.jsonl` | `[x]` deployed | required for G10 summaries |
| Gradient conflict | semantic/action-adapter PCGrad | `[!]` rejected | F9c DDP path ran; action did not improve |
| Gradient conflict | full-model PCGrad/CAGrad | `[!]` intentionally not deployed | cost/root evidence mismatch; defer until positive gradient-conflict proof |
| Action boundary | stop-gradient PICF prefix/context | `[x]` deployed | current production interface |
| Action boundary | policy-only stationary-PICF branch | `[!]` tested and rejected | G10d action rebounded like G10a |
| Action interface | chunked continuous action | `[x]` native | current PI0.5 action path already chunked/continuous-compatible |
| Action interface | rewrite action head to OpenVLA-OFT style | `[!]` not in this gate | would be a separate architecture branch, not a sampler diagnosis |
| Modality adapters | typed PICF modality adapters/projectors | `[x]` deployed | current MVTrack runtime |
| Embodiment adapter | per-embodiment heads/adapters | `[!]` deferred | CALVIN has one embodiment; add for multi-robot data |
| Dynamic mixing | PiKE-style adaptive weights | `[!]` deferred | only after static G10 rows prove static mixing is insufficient |
| MoE | action-expert MoE | `[!]` deferred | only after measured persistent task-family conflict |

Current G10 therefore does not claim to have tested every possible architecture.
It tests every immediately actionable, non-duplicated point from the requested
VLA recipe that can be evaluated within the 8-hour budget without harming
large-data scaling.

## Root Problem Being Tested

Observed action behavior:

```text
large / exact / balanced update windows:
  action can descend quickly on specific windows;

small random or insufficiently covered production updates:
  action plateaus, rebounds, or stays near 0.04-0.06 default-equivalent loss;

F8/F9 short gates:
  task coverage and telemetry were repaired, but action still did not descend.
```

Formal objective:

```text
L = Σ_b q_b L_b
G = Σ_b q_b ∇L_b
```

where `b` is a CALVIN task bucket.  A small physical batch can be an unbiased
estimator only in expectation, but one optimizer step may still have high task
variance:

```text
Var[g] = E_b ||∇L_b - G||^2
```

The repair must make each optimizer step closer to `G` without specializing to
a tiny CALVIN subset.  The production model must still scale to large
heterogeneous data where modalities and task families are missing or uneven.

## Root-Cause Hypothesis Tree

This gate distinguishes four different explanations that were previously mixed
together under "small batch does not converge".

```text
H1: task coverage variance
  Each optimizer update sees too few task families.  Expected fix:
    K4 logical batch + without-replacement task coverage + per-bucket loss
    normalization should improve action.
  Tested by:
    G10a.
  Current evidence:
    negative so far; structure improves but action worsens.

H2: task-uniform over-samples small/noisy buckets
  CALVIN small buckets may be noisier or less action-informative.  Expected fix:
    temperature sampling q_b ∝ N_b^0.5 should reduce noise while preserving
    long-tail exposure.
  Tested by:
    G10b.

H3: data-proportional action prior is needed
  Pure task balancing may fight the true empirical action distribution.
  Expected fix:
    trajectory-proportional sampling improves action, but may harm long-tail
    scaling.
  Tested by:
    G10c.

H4: moving PICF/action-boundary conflict
  Action head is learning against a changing PICF prefix/context distribution.
  Expected fix:
    stationary or policy-only PICF boundary descends when G10a/b/c do not.
  Tested by:
    G10d.
```

G10a/b/c/d are now negative.  The root cause is not merely static sampling, and
it is not fixed by a simple policy-only stationary-PICF boundary.  The next
root-cause class is action data/window construction, action loss/normalization
semantics, or action-expert bridge design, not another static sampler tweak.

## Literature-Derived Requirements

The following are the requested points and their PICF status.

| Point | Meaning in PICF | Current status | Gate decision |
| --- | --- | --- | --- |
| VLA Foundry-style dataset/task mixing | explicit bucket distributions, ratios, without-replacement logical batch | `[x]` implemented | Keep; test only combinations not already rejected |
| ABot-M0-style task/embodiment balance | task-uniform / temperature bucket objective now; dataset/embodiment hooks later | `[x]` implemented for CALVIN task family | Keep; no CALVIN embodiment ablation needed |
| π0.5-style heterogeneous cotrain | action remains PI0.5 path; PICF is prefix/context belief router | `[~]` partially implemented | G10 shows sampler/boundary controls are insufficient |
| Per-task/per-bucket normalization | logical loss scale uses target `q_b`, not raw sampled count | `[x]` implemented and audited | Keep |
| Gradient accumulation as logical batch | K=4 on 2xA100 through accum=2 memory-safe scopes | `[x]` implemented | Use K4 gates only |
| Per-bucket metrics | flattened bucket metrics in `metrics.jsonl` | `[x]` implemented | Mandatory in all gates |
| Dynamic PiKE-style mixing | gradient-informed future bucket weights | `[!]` not deployed | Do not implement before a positive static/boundary recipe; F9b scalar EMA was insufficient |
| PCGrad/CAGrad | per-bucket gradient surgery | `[!]` semantic PCGrad tested and rejected | Do not repeat; whole-model PCGrad rejected by cost/root evidence |
| Knowledge Insulation/action boundary | action gradients should not freely rewrite semantic/PICF belief path | `[!]` simple policy-only control rejected | Needs deeper bridge/data-window diagnosis |
| OpenVLA-OFT action chunking / continuous action | PI0.5 action path already predicts action chunks with comparable action metrics | `[x]` already native to current path | No action-head rewrite in this 8h gate |
| Modality-specific adapters | PICF uses typed modality projectors and action context adapter | `[~]` present but not a new root fix | Keep; do not add new modality tower now |
| Embodiment adapters/heads | relevant for multi-robot data, not current CALVIN one-embodiment gate | `[!]` future-data hook | Defer |
| MoE action expert | useful only after measured persistent task-family conflict under good boundary | `[!]` not justified yet | Defer |

References used for this gate:

```text
VLA Foundry (2604.19728, https://arxiv.org/abs/2604.19728):
  dataset/task mixing, batch balancing ratios, framework-level VLA training.

ABot-M0 (2602.11236, https://arxiv.org/abs/2602.11236):
  large heterogeneous robotics data, task/embodiment balance, action manifold
  stabilization.

PiKE (2502.06244, https://arxiv.org/abs/2502.06244):
  adaptive data mixing when gradient interactions are mostly non-conflicting.

OpenVLA-OFT (2502.19645, https://arxiv.org/abs/2502.19645):
  parallel decoding, action chunking, continuous action representation, L1
  action regression.

Knowledge Insulation (2505.23705, https://arxiv.org/abs/2505.23705):
  isolate continuous action expert gradients from the VLM backbone.

GR00T N1 (2503.14734, https://arxiv.org/abs/2503.14734) and
Gemini Robotics 1.5 (2510.03342, https://arxiv.org/abs/2510.03342):
  dual-system / multi-embodiment VLA design pressure.
```

## Already-Rejected Branches

Do not rerun these unless code changed in the exact causal area:

```text
F8r:
  K4 task-uniform without-replacement + per-bucket normalization.
  Dataflow passed; action 0.04069 -> 0.04256 -> 0.04293.
  Rejected as sufficient action fix.

F9a:
  Freeze semantic/action adapter.
  Action 0.04120 -> 0.04302, grad_norm collapsed.
  Rejected.

F9b:
  Per-bucket action EMA scalar normalization.
  Telemetry passed; action 0.040670 -> 0.042547.
  Rejected.

F9c:
  Semantic/action adapter PCGrad.
  FSDP path incompatible; DDP/no-checkpoint telemetry passed.
  Action 0.040648 -> 0.042562.
  Rejected.
```

## 8-Hour Experiment Matrix

All tests resume from the same step-11000 checkpoint unless explicitly stated.
Each gate uses:

```text
K4 logical optimizer update on 2 GPUs:
  world_size=2
  accum_steps=2
  logical_batch_task_count=4
  sample_without_replacement=true
  per-bucket logical normalization=true
  per-bucket metrics=true

Metrics:
  loss_action_default_equiv
  loss_total_minus_action
  loss_anchor_pv
  loss_anchor_object_pull
  loss_mapg_routing
  loss_slot_jepa
  grad_norm and owner-group grad norms
  logical_batch_distinct_bucket_count
  bucket_* action and non-action losses
  active/downstream same-role overlap
```

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_g10_8h_full_vla_matrix_20260603.sh
```

Remote tmux command:

```bash
cd /root/openpi_e21b2_1b07eab
tmux new-session -d -s picf_g10_8h_matrix_20260603 \
  'bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_g10_8h_full_vla_matrix_20260603.sh'
```

Monitor:

```bash
tail -f /mnt/picf_run_logs/g10_8h_full_vla_matrix_20260603/matrix_status.log
tail -f /mnt/picf_run_logs/g10_8h_full_vla_matrix_20260603/g10a_taskuniform_k4_adapter_norm.log
tail -f /mnt/picf_run_logs/g10_8h_full_vla_matrix_20260603/g10b_temperature05_k4_adapter_norm.log
tail -f /mnt/picf_run_logs/g10_8h_full_vla_matrix_20260603/g10c_trajectory_k4_adapter_norm.log
tail -f /mnt/picf_run_logs/g10_8h_full_vla_matrix_20260603/g10d_taskuniform_k4_policyonly_boundary.log
```

### G10a: Current Static-Mix Control

Purpose:

```text
Verify the current code still reproduces the known F8/F9 baseline under
task-uniform K4 coverage.  This is the control for all other gates.
```

Configuration:

```text
calvin_bucket_sampling_mode=task_uniform
semantic_trainable_scope=action_head_and_adapter
action_context_integration=suffix_cross_attention
action_context_stopgrad=1
action_prefix_stopgrad=1
gradient_surgery=off
action_bucket_ema=off
```

Decision:

```text
If action descends now, earlier F8/F9 negatives were stale-code artifacts.
If action remains F8/F9-like, use it as the current control row.
```

### G10b: Temperature-Mix Control

Purpose:

```text
Test whether pure task-uniform is over-sampling small/noisy buckets and hurting
action.  This corresponds to the VLA Foundry / ABot-M0 temperature-sampling
choice without weakening scaling: q_b ∝ N_b^0.5.
```

Configuration delta:

```text
calvin_bucket_sampling_mode=temperature
calvin_bucket_temperature_alpha=0.5
```

Decision:

```text
Positive if action improves over G10a and bucket metrics do not show one task
dominating.  If neutral/negative, keep task_uniform for CALVIN.
```

### G10c: Trajectory-Proportional Control

Purpose:

```text
Test the opposite limit: data-proportional sampling.  This tells us whether
task-uniform itself is the source of action variance.
```

Configuration delta:

```text
calvin_bucket_sampling_mode=trajectory
```

Decision:

```text
Positive only if action improves without structural collapse.  If trajectory
is better but rare buckets degrade, prefer temperature over trajectory for
large-data scaling.
```

### G10d: Strong Action-Boundary / Policy-Only Control

Purpose:

```text
Test the Knowledge-Insulation hypothesis directly: action should learn when
the belief/PICF prefix is stationary and action gradients cannot keep changing
the semantic/PICF interface.
```

Configuration:

```text
picf_trainable_scope=policy_only
semantic_trainable=1
semantic_trainable_scope=action_head_and_adapter
picf_core_lr_scale=1e-12  # nonzero parser placeholder; policy_only freezes PICF core
action_context_stopgrad=1
action_prefix_stopgrad=1
```

Decision:

```text
If this descends while G10a/b/c do not, the next deployment is a staged or
gated action-boundary recipe, not more sampler work.

If this also fails, then the problem is not only PICF moving-prefix conflict;
the next branch must inspect action data normalization/window distribution or
rewrite the action expert bridge.
```

## What This Gate Can and Cannot Prove

Can prove within 8 hours:

```text
whether task_uniform vs temperature vs trajectory matters under the current
  K4 logical estimator;
whether the remaining bottleneck is sampler mixture or action-boundary;
whether F8/F9 conclusions still hold after the latest code;
whether a production-safe next recipe exists for a longer run.
```

Cannot prove within 8 hours:

```text
final 30k convergence;
CALVIN success rate;
large heterogeneous dataset generalization;
MoE benefit;
full PiKE dynamic sampling benefit.
```

## Failure Interpretation Rules

The gate must not hide a negative result behind another optimizer tweak.

```text
If G10a is flat:
  task-uniform K4 + current action boundary is not sufficient.

If G10b improves:
  pure task-uniform is likely over-sampling small/noisy buckets; deploy
  temperature sampling.

If G10c improves but rare buckets degrade:
  trajectory weighting helps action but is risky for long-tail scaling; use
  temperature rather than pure trajectory for large-data runs.

If G10d improves:
  the root cause is moving-prefix / action-boundary conflict.  Deploy staged
  action-boundary cotrain.

If all four are flat:
  the remaining root is not sampler coverage alone.  Stop repeating batch or
  optimizer variants and move to action-data/window audit or action-expert
  bridge redesign.
```

## Final Deployment Rule

After G10:

```text
If G10a/b/c all stay flat and G10d descends:
  deploy action-boundary staged cotrain:
    stage 1 policy/action bridge learns against stationary PICF;
    stage 2 PICF core re-enters with low LR and stopgrad action context;
    K4 task-balanced logical batch remains mandatory.

If temperature beats task_uniform:
  deploy q_b ∝ N_b^0.5 for CALVIN and keep task/dataset temperature as the
  large-data default.

If trajectory beats both task_uniform and temperature:
  deploy trajectory-proportional only for CALVIN, but keep temperature as the
  large-data default because trajectory sampling can drown long-tail tasks.

If none descends:
  stop optimizer/sampler tinkering.  Move to structural action-expert bridge
  redesign or action-data/window audit before another 30k run.
```

## Required Archival

After each branch:

```text
append metrics summary here;
record exact command, checkpoint, and git diff hash;
mark accept/reject;
do not rerun rejected branches without a causal code change.
```

## Running Results

### G10a Interim: task-uniform K4 control

Status: `[!]` rejected and early-stopped at step `11050`.

Early rows from step `11010 -> 11030`:

```text
loss_action_default_equiv:
  0.040636 -> 0.042953  (worse)

loss_action_active7:
  0.184337 -> 0.195523  (worse)

loss_total_minus_action:
  0.010911 -> 0.009229  (better)

loss_anchor_pv:
  0.489586 -> 0.472663  (better)

loss_anchor_object_pull:
  0.227332 -> 0.064367  (better)

loss_mapg_routing:
  0.431503 -> 0.415389  (better)

loss_slot_jepa:
  0.690762 -> 0.690151  (flat)

aqr_active_same_role_support_overlap_max:
  0.137481 -> 0.062500  (better)

aqr_downstream_same_role_support_overlap_max:
  0.137481 -> 0.064548  (better)
```

Interpretation:

```text
The current task-uniform K4 control improves structural / alignment losses but
does not improve action in the early window.  This reproduces the F8/F9 pattern:
cleaner PICF structure alone is not sufficient.  The remaining G10 branches are
therefore necessary:

G10b checks whether task-uniform is over-sampling small/noisy buckets.
G10c checks whether trajectory-proportional data weighting is better for action.
G10d checks whether action-boundary / stationary PICF is the true bottleneck.
```

Early-stop update at step `11050`:

```text
loss_action_default_equiv:
  0.040636 -> 0.049548  (clear worsening)

loss_action_active7:
  0.184337 -> 0.225558  (clear worsening)

loss_total:
  0.092183 -> 0.110432  (worse)

loss_anchor_pv:
  0.489586 -> 0.499048  (worse vs first row)

loss_anchor_object_pull:
  0.227332 -> 0.268542  (worse vs first row)

aqr_active_same_role_support_overlap_max:
  0.137481 -> 0.150000  (not the main failure, but no longer improving)
```

Decision:

```text
G10a is not a usable final recipe.  It is retained as the control showing that
task-uniform K4 + current action adapter/boundary still reproduces the old
"structure may improve while action worsens" failure mode.
```

### G10b Result: temperature K4 control

Status: `[!]` rejected and early-stopped at step `11050`.

Rows from step `11010 -> 11050`:

```text
loss_action_default_equiv:
  0.036027 -> 0.049377  (clear worsening)

loss_action_active7:
  0.163377 -> 0.224814  (clear worsening)

loss_total:
  0.082667 -> 0.108247  (clear worsening)

loss_total_minus_action:
  0.010613 -> 0.009492  (better)

loss_anchor_pv:
  0.483178 -> 0.500207  (worse)

loss_anchor_object_pull:
  0.198369 -> 0.098907  (better)

loss_mapg_routing:
  0.436161 -> 0.435425  (effectively flat)

loss_slot_jepa:
  0.687386 -> 0.701100  (worse)

aqr_active_same_role_support_overlap_max:
  0.162481 -> 0.162500  (flat)

aqr_downstream_same_role_support_overlap_max:
  0.162481 -> 0.162500  (flat)
```

Interpretation:

```text
Temperature sampling is better than G10a on the first action row and briefly
improves object-pull/routing/overlap, but it does not solve the action
objective.  By the 11050 decision point action and total loss have clearly
rebounded.  This rejects H2 as a sufficient explanation: the failure is not
simply pure task-uniform over-sampling small/noisy buckets.

The matrix therefore moved to G10c at `2026-06-03T05:13:58+08:00`.
```

### G10c Result: trajectory-proportional K4 control

Status: `[!]` rejected and early-stopped at step `11050`.

Rows from step `11010 -> 11050`:

```text
loss_action_default_equiv:
  0.040106 -> 0.046618  (worse; step 11020 peaked at 0.048699)

loss_action_active7:
  0.182044 -> 0.212004  (worse)

loss_total:
  0.091802 -> 0.102647  (worse)

loss_total_minus_action:
  0.011589 -> 0.009411  (better)

loss_anchor_pv:
  0.488304 -> 0.497322  (worse)

loss_anchor_object_pull:
  0.288557 -> 0.088320  (better)

loss_mapg_routing:
  0.431709 -> 0.432804  (flat/slightly worse)

loss_slot_jepa:
  0.691216 -> 0.697618  (worse)

aqr_active_same_role_support_overlap_max:
  0.137481 -> 0.150000  (worse)
```

Interpretation:

```text
Trajectory-proportional sampling partially reduced the immediate action rebound
relative to G10a/G10b, but it still did not produce action descent.  By the
11050 decision point action, total, anchor_pv, slot_jepa, and overlap were all
above their first rows.

This rejects H3 as a sufficient explanation.  The failure is not fixed by
task-uniform, temperature, or trajectory-proportional static data mixing under
the current moving PICF/action boundary.  The matrix therefore moved to G10d at
`2026-06-03T05:35:49+08:00`.
```

### G10d First Launch: Configuration Failure, Not Scientific Failure

Status: `[x]` rerun completed after config fix.

The first `G10d` launch failed before producing training metrics:

```text
ValueError: picf_core_lr_scale must be > 0, got 0.0.
```

This did **not** test or reject the policy-only/action-boundary hypothesis.
The launcher was corrected to use:

```text
picf_trainable_scope=policy_only
picf_core_lr_scale=1e-12
```

The nonzero LR scale is only a parser/validator placeholder.  The actual
scientific control remains policy-only: PICF core is not trainable, while the
action head/adapter boundary is trainable.  The rerun was launched with:

```bash
ONLY_CASE=g10d_taskuniform_k4_policyonly_boundary \
bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_g10_8h_full_vla_matrix_20260603.sh
```

Rerun rows from step `11010 -> 11050`:

```text
loss_action_default_equiv:
  0.040636 -> 0.049548  (worse)

loss_action_active7:
  0.184333 -> 0.225551  (worse)

loss_total:
  0.091505 -> 0.110433  (worse)

loss_total_minus_action:
  0.010232 -> 0.011336  (worse)

loss_anchor_pv:
  0.487621 -> 0.499050  (worse)

loss_anchor_object_pull:
  0.163828 -> 0.268529  (worse by 11050 after a transient improvement)

loss_mapg_routing:
  0.423765 -> 0.433028  (worse)

loss_slot_jepa:
  0.693263 -> 0.704728  (worse)

aqr_active_same_role_support_overlap_max:
  0.099981 -> 0.150000  (worse)
```

Decision:

```text
G10d is rejected.  Freezing PICF core and training only the policy/action
boundary did not make action descend.  It temporarily improved several
structure metrics by step 11030, but by step 11050 action, total, anchor_pv,
object_pull, routing, slot_jepa, and active overlap all rebounded.

This means the current 8-hour matrix has not found a suitable repair.  The
negative result is specific: static sampler variants and a simple
policy-only/action-boundary control are insufficient.  The next branch should
not be another sampler tweak; it should inspect action window/data semantics
and the action-expert conditioning bridge.
```

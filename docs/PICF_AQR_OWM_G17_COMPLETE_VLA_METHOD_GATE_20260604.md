# PICF-AQR-OWM G17 Complete VLA Method Gate

Date: 2026-06-04

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

This document is the current execution contract for the requested "all methods"
gate after the June 3 G10-G16 experiments.  It exists to prevent repeating old
experiments and to make every new run pass three checks before launch:

```text
1. theory follow-through: the method optimizes a stated mathematical target;
2. code follow-through: the exact code path and flags are identified;
3. experiment follow-through: the run has a checklist, logs, and acceptance rule.
```

## Current Decision

The previous failure should not be described as "we never tried task-balanced
VLA training".  That branch has been implemented and tested.

Current evidence says:

```text
Already implemented and tested:
  task-balanced logical batches
  without-replacement bucket coverage
  explicit bucket-ratio sampling
  temperature sampling
  trajectory-proportional sampling
  per-bucket logical loss normalization
  dynamic PiKE-style bucket mixing
  scoped PCGrad/CAGrad
  action/PICF stop-gradient boundary
  suffix cross-attention PICF action context
  prefix/append bridge controls
  action-expert router residual
  continuous chunked flow action objective with MSE/L1/Huber/SmoothL1 variants

Still not fully closed:
  whether PICF action context has positive causal value under a fixed-window
  bridge-capacity probe;
  whether the missing pi0-FAST / Knowledge-Insulation-style action-token
  representation objective is needed for stable action descent;
  whether the new 2xA100 machine reproduces the maintained logical-batch
  dataflow exactly.
```

Therefore G17 does **not** spend time re-running every sampler variant.  It
runs only non-duplicate experiments that can change the next deployment
decision.

## Literature Position

The requested methods map to current VLA practice as follows.

```text
VLA Foundry / ABot-M0 / PiKE:
  optimize the multi-task objective by making each optimizer step approximate
  the target mixture gradient.

Knowledge Insulation / pi0.5:
  do not let continuous action-expert gradients freely rewrite the semantic
  VLM trunk; use controlled action interfaces and action-relevant representation
  objectives.

OpenVLA-OFT:
  prefer continuous action chunks and stable scalar objectives for action
  fine-tuning, rather than long autoregressive low-level action decoding.

AdaMoE/FedVLA-style action experts:
  use task-aware experts only when gradient-conflict evidence justifies them;
  do not MoE the whole VLM as a first response.
```

External references used for the theoretical checklist:

```text
VLA Foundry: https://arxiv.org/abs/2604.19728
ABot-M0: https://arxiv.org/abs/2602.11236
PiKE: https://arxiv.org/abs/2502.06244
OpenVLA-OFT: https://arxiv.org/abs/2502.19645
Knowledge Insulation: https://arxiv.org/abs/2505.23705
FAST action tokenization: https://www.physicalintelligence.company/research/fast
AdaMoE-VLA: https://charleshen1412.github.io/AdaMoE-VLA/
```

## Mathematical Contract

### 1. Task-balanced logical batch

The desired large-data objective is:

```text
L = sum_b q_b L_b
G = sum_b q_b grad L_b
```

where `b` is a task or dataset bucket.  A physical micro-batch is acceptable
only if the optimizer-step estimator is:

```text
g_hat = sum_{b in B_step} (q_b / n_b) grad L_b
```

where `n_b` is the number of selected micro-windows from bucket `b` in that
logical update.

Code follow-through:

```text
scripts/picf_core_train.py
  _compute_bucket_sampling_weights()
  _bucket_sequence_for_logical_step()
  _logical_batch_loss_scales()
  _CalvinTransitionSource.balanced_bucket_slot_index()
  training loop: source.balanced_bucket_slot_index()
  training loop: loss scaling before backward
```

Historical evidence:

```text
docs/PICF_AQR_OWM_8H_FULL_VLA_REPAIR_MATRIX_20260603.md
docs/PICF_AQR_OWM_G11_FULL_VLA_BRIDGE_AND_DATA_PLAN_20260603.md
docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
```

Status: `[x] implemented and tested; keep as baseline, do not repeat alone`.

### 2. Dynamic mixing and gradient surgery

Dynamic bucket weights are mathematically valid when they still represent a
bounded target mixture:

```text
q_b(t) = normalize_clip(q_b^base * exp(eta * lag_b), q_min, q_max)
```

Scoped gradient surgery is valid only on small trainable groups where per-task
gradients can be measured:

```text
if <g_i, g_j> < 0:
  g_i <- g_i - (<g_i, g_j> / ||g_j||^2) g_j
```

Code follow-through:

```text
scripts/picf_core_train.py
  _dynamic_bucket_sampling_weights()
  _pcgrad_project_and_sum()
  _cagrad_project_and_sum()
  _logical_batch_autograd_grads()
```

Historical evidence:

```text
G12-DYN, G12-PCG, G12-CAGrad:
  dataflow worked;
  structure sometimes improved;
  action descent did not pass.
```

Status: `[!] tested and rejected as current action fix; keep knobs, not default`.

### 3. Action/PICF bridge

The bridge question is causal:

```text
action_loss = L_a(A | S, C_picf)
```

We need compare:

```text
normal:          C_picf available through suffix cross-attention
no_picf_action:  C_picf removed from action path
prefix_fusion:   C_picf enters as prefix-fused condition
append:          C_picf only appended as raw extra tokens
```

The bridge is useful only if:

```text
normal or prefix_fusion improves action loss on identical windows
relative to no_picf_action/append, without causing non-action drift.
```

Code follow-through:

```text
src/openpi/picf/paligemma/wrapper.py
  _apply_action_context_adapter()
  _apply_action_expert_router()
  compute_action_flow_loss()

scripts/picf_action_bridge_capacity_probe.py
  exact fixed-window bridge-capacity probe
```

Status: `[~] needs G17 exact-window causal probe on the new 2xA100 machine`.

### 4. Action representation objective

Current PICF training uses continuous flow action chunks and logs the canonical
MSE-compatible metric:

```text
loss_action_default_equiv
```

OpenPI also contains pi0-FAST tokenization infrastructure, but PICF core
training currently has no deployed auxiliary action-token representation loss:

```text
FASTTokenizer exists;
TokenizeFASTInputs exists;
PI0_FAST exists;
PICF/PaliGemma training does not yet add:
  y = Q(a_{1:H})
  L_token = CE(p_theta(y | semantic tokens, state, PICF context))
```

Why this matters:

```text
Knowledge Insulation-style training can let the semantic VLM learn
motor-relevant representation through a representation/action-token objective
while keeping continuous action gradients controlled.
```

Status: `[ ] not deployed in PICF training; must not be claimed as tested`.

G17 action:

```text
1. audit exact FAST/token code compatibility;
2. if the required token head is absent, record as a true future branch;
3. do not bolt on a fake CE head in this gate without a dataflow test.
```

## G17 Experiment Checklist

## Complete Requested Method Matrix

This is the strict non-duplicate accounting for the methods requested on
2026-06-04.  The rule is:

```text
Run again only if it can change a deployment decision.
Do not burn GPU on a method already rejected under the same code/dataflow.
Keep scalable infrastructure even when it is not the current default.
```

| Method | Code follow-through | Math target | Historical/active experiment | Current decision |
| --- | --- | --- | --- | --- |
| Task-balanced logical batch | `_bucket_sequence_for_logical_step`, balanced source slot index, accum loop | `g_hat=sum_b(q_b/n_b) grad L_b` | G10-G12 plus local smoke; K4 distinct coverage proven | Keep as required baseline |
| Gradient accumulation as logical batch | `ACCUM_STEPS`, `logical_batch_task_count`, optimizer step after logical group | same estimator, larger `B_step` | K4/K8 resource probes, 2x and 6x attempts | Keep; do not rely on physical huge batch only |
| Explicit bucket ratio / VLA-Foundry-style ratios | `calvin_bucket_weight_spec`, `_compute_bucket_sampling_weights` | fixed target mixture `q_b` | G12-RATIO; structure improved, action not fixed | Keep knob, not default fix |
| Temperature sampling | `calvin_bucket_sampling_mode=temperature`, alpha | `q_b proportional N_b^alpha` | G12-TEMP alpha=0.5 failed action slope | Keep knob for future datasets, not current fix |
| Trajectory-proportional sampling | `calvin_bucket_sampling_mode=trajectory` | sample by empirical frequency | local smoke + prior dataflow | Rejected as default because it under-covers long-tail buckets |
| Per-bucket logical loss normalization | `_logical_batch_loss_scales` | each selected bucket contributes `q_b/n_b` | local smoke returned `[1/6,1/3,1/6,1/3]` for duplicate bucket example | Keep mandatory |
| Per-modality/action normalization | action/PICF/sidecar losses logged separately; action default-equivalent metric | avoid hidden scale domination | A3/G12 target-scale audits | Keep metrics; no generic multi-dataset normalizer until new datasets |
| Dynamic PiKE-style mixing | `_dynamic_bucket_sampling_weights` and dynamic logs | bounded adaptive `q_b(t)` from recent lag | G12-DYN dataflow real; action not fixed | Keep infrastructure; not default |
| Gradient cosine diagnosis | fixed-window/per-bucket gradient probes in G11/G12 | detect negative inter-task gradient cosine | A2/G12 used to decide PCGrad/CAGrad | Keep diagnostic |
| Scoped PCGrad | `_pcgrad_project_and_sum`, scoped groups | remove negative pairwise components on small trainable group | G12-PCG: dataflow real; action not fixed | Keep diagnostic/off by default |
| Scoped CAGrad | `_cagrad_project_and_sum`, scoped groups | conflict-averse simplex update | G12-CAGrad: dataflow real; action not fixed | Keep diagnostic/off by default |
| Whole-model PCGrad/CAGrad | intentionally not used | would require per-task full-model grads | rejected by math/cost/action stop-gradient boundary | Do not run unless architecture changes |
| Action/PICF stop-gradient boundary | wrapper action-context path uses detached context / trainable action adapters | continuous action gradients should not rewrite PICF core directly | active E17-B exact-window causal probe | Current active test |
| Suffix cross-attention action context | `_apply_action_context_adapter` | action loss conditioned on bounded PICF context tokens | active normal-vs-noPICF probe | Current active test |
| Action-expert residual/router | `_apply_action_expert_router` | task-conditioned residual inside action expert only | implemented; older ckpt lacks params; not enabled in current E17-B | Evidence-triggered only |
| Continuous chunked action flow | `compute_action_flow_loss` | stable chunk-level continuous objective | current PICF/PaliGemma action path | Keep current action metric |
| L1/Huber/SmoothL1 variants | wrapper/training flags and logs | robust action regression alternative | implemented in prior gates; not selected as default | Evidence-triggered after flow-specific failure |
| FAST/action-token auxiliary | OpenPI native FAST exists; PICF core lacks deployed CE branch | representation objective for motor tokens | E17-D audit: missing in PICF path | True future branch, not tested |
| Embodiment adapters/heads | not relevant in CALVIN single embodiment | avoid action-space conflict across robots | not a CALVIN gate | Future large-data branch |
| Action-expert MoE | wrapper router exists; off by default | expert split only if persistent task-family conflict remains | not justified by current completed evidence | Future evidence-triggered branch |
| System2/System1 split | no reliable CALVIN subtask pseudo-label in current gate | separate long-horizon plan from low-level control | not a fast action-loss repair | Future long-horizon branch |

The 2-3 hour G17 gate therefore has two real experiments:

```text
E17-B:
  exact fixed-window normal_suffix vs no_picf_action causal probe.
  Purpose: determine whether PICF context helps or hurts the action readout.

E17-D:
  FAST/action-token audit.
  Purpose: determine whether the Knowledge-Insulation-style representation
  objective is already present in PICF training or is a missing future branch.
```

If E17-B passes, the next deployable recipe is:

```text
task-balanced K4/K8 logical batch
per-bucket q_b/n_b normalization
continuous chunk action flow
PICF action context through bounded suffix cross-attention
PICF/action stop-gradient boundary
no dynamic mixing / no PCGrad / no CAGrad / no MoE by default
```

If E17-B fails, the next non-duplicate branch is not "more sampler tuning"; it is
either:

```text
1. implement the missing FAST/action-token representation auxiliary, or
2. test action-expert/router capacity with the same fixed-window contract.
```

### E17-A: Code/path audit

Goal:

```text
prove which requested VLA methods are actually wired into current PICF training.
```

Checklist:

```text
[x] task sampler code path identified
[x] logical loss-scaling code path identified
[x] dynamic mixing code path identified
[x] PCGrad/CAGrad code path identified
[x] action context bridge code path identified
[x] continuous action objective code path identified
[ ] action-token auxiliary code path identified or declared missing
```

Acceptance:

```text
all implemented methods have file/function references;
missing methods are explicitly marked missing, not silently counted as done.
```

### E17-B: Bridge causal probe

Goal:

```text
test whether PICF action context is a useful action conditioning signal under
identical windows, independent of live sampler noise.
```

Cases:

```text
normal_suffix:
  suffix_cross_attention, PICF action context on

no_picf_action:
  suffix_cross_attention, --disable-picf-action-condition

append_negative:
  append, PICF action context on
```

Run shape:

```text
steps: 50-100 per case
windows_per_step: 4 if memory allows, otherwise 2
scope: policy_only PICF + action_head_and_adapter semantic scope
metric: loss_action_default_equiv first10 -> last10
```

Acceptance:

```text
normal_suffix or prefix_fusion must beat no_picf_action by a clear margin
on identical windows;
append must not be the best case.
```

### E17-C: Production logical-batch smoke

Goal:

```text
confirm the new 2xA100 machine runs the maintained scalable baseline:
K4 logical batch, task_uniform, without replacement, per-bucket normalization,
action weight 4, action context on.
```

Acceptance:

```text
startup metrics show logical_batch_distinct_bucket_count >= 4;
loss_action_default_equiv prints every 10-50 steps;
no NaN/OOM/runtime path mismatch before 50-100 steps.
```

### E17-D: FAST/action-token branch audit

Goal:

```text
decide whether the missing representation objective is a code task or already
covered by current flow loss.
```

Checklist:

```text
[x] locate FAST tokenizer use in OpenPI native path
[x] verify PICF core training path does not currently use FAST token CE
[x] locate action-token prediction head compatibility gap
[x] locate loss wiring and logging gap
[x] if absent, document exact new module required:
    action_token_projector/head
    FASTTokenizer action target generation
    CE loss scaled as representation auxiliary
    stop-gradient boundary to continuous action expert
```

Acceptance:

```text
do not run a fake experiment.  Either the path exists and can be tested, or it
is recorded as the next implementation branch.
```

Code audit result:

```text
OpenPI native FAST path:
  src/openpi/transforms/_base.py
    TokenizeFASTInputs
    ExtractFASTActions
  src/openpi/training/config.py
    ModelType.PI0_FAST transform branch
  src/openpi/models/pi0_fast.py
    Pi0FAST.compute_loss(): autoregressive next-token CE using token_loss_mask

Current PICF/PaliGemma action path:
  src/openpi/picf/paligemma/wrapper.py
    compute_action_flow_loss(): continuous flow action chunk objective
    _apply_action_context_adapter(): gated suffix cross-attention from PICF context
    _apply_action_expert_router(): detached semantic/PICF-conditioned action residual experts

Conclusion:
  FAST token infrastructure exists in the repository, but PICF core training
  does not yet expose a FAST/action-token representation auxiliary.  It is not
  valid to count Knowledge-Insulation-style FAST CE as tested for PICF.
```

Future implementation contract if this branch is selected:

```text
y = FASTTokenizer(a_{1:H})
z = semantic/PICF condition tokens, with the continuous-action expert boundary
    kept controlled unless explicitly testing backbone adaptation.
L_fast = CE(p_theta(y_i | y_{<i}, z), token_loss_mask)
L_total = L_flow + lambda_fast * L_fast + non-action PICF terms

The FAST branch must be logged separately:
  loss_action_fast_ce
  loss_action_flow
  loss_action_default_equiv
and must not replace the current continuous chunk metric.
```

## Run Commands

The exact cloud commands are filled in after syncing the new 2xA100 machine.
Expected remote root:

```text
/root/openpi_g17
```

Expected log roots:

```text
/mnt/picf_run_logs/g17_complete_vla_method_gate_20260604
/mnt/picf_fixed_window_probes/g17_bridge_20260604
```

## Current Outcome Log

This section is updated during execution.

```text
2026-06-04 local:
  E17-A started.  Existing code paths confirm sampler/mixing/normalization/
  gradient-surgery/action-bridge functionality; action-token auxiliary remains
  the main unverified/missing path.
  Local py_compile passed:
    scripts/picf_core_train.py
    scripts/picf_action_bridge_capacity_probe.py
    src/openpi/picf/paligemma/wrapper.py
    src/openpi/picf/core/training.py
  Local direct-import caveat:
    importing scripts.picf_core_train in this workstation triggers a
    tqdm_loggable -> IPython -> wandb vendored pygments assertion on
    "ansibrightred".  This is an environment/import-side issue, not a syntax
    failure.  A temporary fake-IPython module was used only for pure-function
    sampler smoke.
  Local sampler smoke:
    task_uniform -> 0.25/0.25/0.25/0.25
    temperature alpha=0.5 -> size-smoothed weights
    trajectory -> proportional to bucket sizes
    explicit ratio block_push=1,drawer=2,button=1,slider=1 -> 0.2/0.4/0.2/0.2
    _logical_batch_loss_scales([block_push, drawer, block_push, button])
      -> [1/6, 1/3, 1/6, 1/3], sum=1.0
    This confirms the local estimator implements q_b / n_b for selected
    buckets.  It does not prove behavior convergence.

2026-06-04 remote 2xA100 setup:
  Synced local src/scripts/docs to:
    /root/openpi_g17_20260604
  Reused the existing venv through:
    /root/openpi_g17_20260604/.venv -> /root/openpi/.venv
  Remote py_compile passed for:
    scripts/picf_action_bridge_capacity_probe.py

2026-06-04 E17-B bug found and fixed:
  scripts/picf_action_bridge_capacity_probe.py was loading a fresh dummy
  optimizer from the resume checkpoint before discarding it and creating a new
  optimizer.  On the 11000 checkpoint this forced both probe processes to read
  optimizer.pt (~17 GB) from /mnt even though the fixed-window probe is meant
  to start from fresh Adam moments.  Both processes entered D-state
  wait_on_page_bit_common while reading shared-storage pages.

  Fix:
    _load_checkpoint(..., optimizer_checkpoint_mode="model_only")

  Mathematical reason:
    E17-B measures local bridge/readout capacity from identical model weights.
    It must load model.pt and metadata.pt, but optimizer moments are explicitly
    outside the target.  Reading optimizer.pt is not only unnecessary; it also
    changes the experiment budget by making startup depend on shared-storage
    I/O instead of model behavior.

  Status:
    old D-state sessions were stopped after I/O released;
    model-only E17-B normal/no-PICF sessions were relaunched.

2026-06-04 E17-B shared-storage mitigation:
  Even after optimizer loading was disabled, launching two probe processes from
  the same /mnt checkpoint caused concurrent reads of model.pt (~9.9 GB).  This
  again entered disk sleep.  The checkpoint was staged once to local overlay:

    /root/picf_local_ckpts/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

  The active E17-B probes now read model.pt/metadata.pt from that local path.
  This is an experiment hygiene fix only.  It does not change model weights,
  data windows, losses, optimizer, or trainable scopes.

2026-06-04 E17-B active probe evidence:
  Remote machine:
    px-cloud1.matpool.com:26120, hostname qe93X5, 2x A100-PCIE-40GB
  Active tmux sessions:
    g17_bridge_normal
    g17_bridge_nopicf
  Shared fixed window trace:
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/g17_flat_windows_from_f8_rank0_64.jsonl
    valid_windows=60
  Common settings:
    loaded_step=11000
    steps=50
    windows_per_step=2
    picf_trainable_scope=policy_only
    semantic_trainable_scope=action_head_and_adapter
    optimizer_checkpoint_mode=model_only
    local checkpoint path:
      /root/picf_local_ckpts/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

  Before-update eval:
    no_picf_action:
      loss_action_default_equiv mean = 0.042090146569535136
      pi_context_token_count = 0
      pi_context_adapter_gate = 0
    normal_suffix:
      loss_action_default_equiv mean = 0.04229679385510584
      pi_context_token_count = 24
      pi_context_adapter_gate ~= 0.1192

  Early training parse from log JSON lines:
    no_picf_action:
      step 1  action_default = 0.05209890753030777
      step 15 action_default = 0.03948785923421383
    normal_suffix:
      step 1  action_default = 0.05242122430354357
      step 20 action_default = 0.011278576217591763

  Interpretation:
    This is not yet a final E17-B pass because no_picf_action is behind in
    completed optimizer steps.  It is, however, already a useful sanity check:
    the normal_suffix branch has nonzero PICF action condition tokens and a
    real adapter gate, and it shows much faster same-window action descent than
    the no-PICF branch in the early window sequence.  Final judgment must use
    the completed 50-step summaries or a matched-step comparison.

2026-06-04 E17-B matched-step update at common steps through 40:
  Robust parsing note:
    probe train JSON is embedded in tqdm log lines, so parsing must extract the
    JSON substring rather than only accepting lines that start with `{`.

  Matched common steps:
    1, 5, 10, 15, 20, 25, 30, 35, 40

  Mean loss_action_default_equiv over these matched steps:
    normal_suffix  = 0.0410703658643696
    no_picf_action = 0.040487923918084964
    delta normal - no_picf = +0.000582441946284637

  Key matched points:
    step 20:
      normal_suffix  = 0.011278576217591763
      no_picf_action = 0.011115290690213442
    step 40:
      normal_suffix  = 0.04120447114109993
      no_picf_action = 0.041819446720182896

  Interpretation:
    The early step-20 low action value is not PICF-context-specific; no-PICF
    reaches the same low value on the same fixed-window sequence.  Through the
    matched step-40 comparison, suffix PICF action context has no measurable
    positive causal effect on fixed-window action fitting.  This does not prove
    PICF context is harmful globally, but it rejects "PICF bridge alone solves
    the convergence problem" as the next default deployment claim.

2026-06-04 E17-B final 50-step result:
  Both probes finished; GPU memory returned to 0 MiB and tmux exited.

  Output files:
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/normal_suffix_localckpt.log
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/normal_suffix_localckpt_train.jsonl
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/normal_suffix_localckpt_summary.json
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/no_picf_action_localckpt.log
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/no_picf_action_localckpt_train.jsonl
    /mnt/picf_fixed_window_probes/g17_bridge_20260604/no_picf_action_localckpt_summary.json

  Train-step loss_action_default_equiv:
    no_picf_action:
      first5 mean = 0.03379740314558148
      last5 mean  = 0.04329671245068312
      last        = 0.03944391943514347
    normal_suffix:
      first5 mean = 0.03482789658010006
      last5 mean  = 0.04342405591160059
      last        = 0.04012051410973072

  Post-eval loss_action_default_equiv mean:
    no_picf_action = 0.043928243355670325
    normal_suffix  = 0.04419125779531896

  Matched-step conclusion:
    normal_suffix is not better than no_picf_action on the identical fixed
    windows.  The difference is within noise and slightly worse for normal in
    post-eval.  Therefore E17-B rejects "bounded suffix PICF action context is
    the missing convergence fix" for the current checkpoint/action-head
    setting.

  Deployment implication:
    Keep the bounded suffix adapter as an optional diagnostic/integration path,
    but do not make it the claimed final fix.  The next non-duplicate branch is
    PICF-side action representation supervision (FAST/action-token auxiliary)
    or action-expert capacity/boundary redesign, not another sampler-only or
    raw-overlap experiment.
```

## G17 Gate Conclusion

```text
Status:
  Completed for this 2-3 hour gate.

What is closed:
  The requested sampler/mixing/normalization/gradient-surgery family has
  code follow-through and prior experimental follow-through.  It should remain
  as infrastructure but is not the missing action-convergence fix.

What G17 newly tested:
  Exact-window action bridge causality on the new 2xA100 machine.

Result:
  normal_suffix PICF action context did not outperform no_picf_action.

What this rules out:
  1. claiming the suffix PICF bridge is the final convergence solution;
  2. spending more GPU on sampler-only variants without a new causal reason;
  3. treating raw same-role overlap as the action bottleneck by itself.

What remains genuinely open:
  1. PICF-side FAST/action-token representation auxiliary;
  2. action-expert capacity/boundary redesign if FAST/action-token is not
     enough;
  3. multi-dataset/embodiment normalizers when the training data expands beyond
     CALVIN.

Next deployable experiment:
  Build a PICF-side action-token auxiliary that keeps the current continuous
  action chunk metric, logs FAST CE separately, and preserves the controlled
  action/PICF boundary.  This is the only remaining branch in the requested
  2025-2026 VLA method set that is both not already rejected and directly
  aligned with Knowledge-Insulation/OpenPI FAST practice.
```

# PICF-AQR-OWM G20 2xA100 Action Readout Experiment Gate

Date: 2026-06-04

Remote:

```text
ssh -p 26120 root@px-cloud1.matpool.com
```

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

This is the strict execution plan and result record for the 2-3 hour gate
requested after G19.
It prevents repeating old experiments and separates three categories:

```text
1. already implemented and historically tested;
2. immediately runnable non-duplicate tests;
3. mathematically justified but not yet deployable without code changes.
```

## 1. Current Root-Cause Hypothesis

The desired action-training path is:

```text
X_t: native PI0.5 visual-language-state input
Z_t: PICF multimodal evidence
B_t = F_phi(Z_t, X_t)
C_t = S_psi(B_t)
I_t = H_theta(X_t, C_t)
a_hat = A_theta(I_t)
L_action = loss(a_hat, a)
```

The action gradient through PICF is:

```text
dL_action/dphi =
  dL_action/da_hat
  * da_hat/dI_t
  * dI_t/dC_t
  * dC_t/dB_t
  * dB_t/dphi
```

G12-G18 evidence shows:

```text
task-balanced gradient estimator exists;
large-card/K12 alone did not reproduce stable 0.02;
PCGrad/CAGrad/dynamic mixing did not solve action descent;
normal_suffix action bridge did not beat no_picf_action on G17 exact windows.
```

Therefore the current primary hypothesis is:

```text
dI_t/dC_t is low-gain or action-irrelevant under the current continuous
action-only objective.
```

## 2. Method Checklist

| Method | Formula / target | Code path | Prior result | G20 status |
| --- | --- | --- | --- | --- |
| Task-uniform logical batch | `g_hat=sum_b q_b/n_b grad L_b` | `scripts/picf_core_train.py:_logical_batch_loss_scales` | tested G12 | do not rerun alone |
| Without-replacement coverage | reduce task-repeat variance inside logical step | `_bucket_sequence_for_logical_step` | tested G12 | do not rerun alone |
| Ratio/temperature sampling | alternate `q_b` definitions | `_compute_bucket_sampling_weights` | ratio/temp tested G12 | do not rerun alone |
| Dynamic PiKE-style mixing | `q_b(t)=clip(q_b exp(eta lag_b))` | `_dynamic_bucket_sampling_weights` | tested G12-DYN, rejected as action fix | keep off |
| Per-bucket action EMA scale | bounded backward scale equalization | `_logical_action_bucket_scale` | implemented; not decisive | keep as knob |
| Scoped PCGrad/CAGrad | remove negative bucket gradient components | `_pcgrad_project_and_sum`, `_cagrad_project_and_sum` | tested G12-PCG/CAGrad, rejected as action fix | keep diagnostic |
| Continuous action chunk | stable chunk-level control objective | `wrapper.compute_action_flow_loss` | active path | keep mandatory |
| Action bridge suffix cross-attn | gated action suffix read of `C_t` | `wrapper._apply_action_context_adapter` | G17 no gain vs no-PICF | rerun only as exact-window sanity on new machine |
| No-PICF action control | remove PICF action condition | `--disable-picf-action-condition` in bridge probe | G17 control | rerun as paired sanity if windows exist |
| Zero/shuffle/sign-flip context | causal sensitivity of `C_t` | not implemented in probe | not tested | code required before experiment |
| PICF-side FAST CE | `CE(FAST(a_chunk)|X,C)` | native FAST exists; PICF aux absent | not tested | true future branch |
| Action expert MoE/router | action-side capacity | router exists, MoE not default | router did not fix G17/G18 | only after FAST/probe evidence |

## 2.1 Code Follow-Through Status

This section is the hard boundary between deployed methods and ideas that still
need code.  It exists to prevent another round of falsely claiming that a method
was tested just because the paper-level idea was discussed.

Verified local code paths:

```text
scripts/picf_core_train.py:3386
  _compute_bucket_sampling_weights
  Covers task-uniform, ratio and temperature bucket probabilities.

scripts/picf_core_train.py:3507
  _dynamic_bucket_sampling_weights
  Covers PiKE-style dynamic bucket lag weighting.

scripts/picf_core_train.py:3606
  _bucket_sequence_for_logical_step
  Covers without-replacement coverage inside a logical update.

scripts/picf_core_train.py:3719
  _logical_batch_loss_scales
  Covers per-logical-bucket loss scaling.

scripts/picf_core_train.py:3798
  _logical_action_bucket_scale
  Covers per-bucket action EMA scaling.

scripts/picf_core_train.py:3936
  _pcgrad_project_and_sum

scripts/picf_core_train.py:4034
  _cagrad_project_and_sum

scripts/picf_action_bridge_capacity_probe.py:401
  --disable-picf-action-condition

scripts/picf_action_bridge_capacity_probe.py:481-484
  action_context_integration / tokens / output_gate / stopgrad overrides.

src/openpi/models/tokenizer.py:122
  FASTTokenizer exists.

src/openpi/transforms/_base.py:293,315
  TokenizeFASTInputs and ExtractFASTActions exist.

src/openpi/models/pi0_fast.py:134,198
  Pi0FAST and its native compute_loss exist.
```

Verified absent from the current PICF core path:

```text
loss_action_fast_ce:
  not present in scripts/picf_core_train.py or src/openpi/picf.

action_context_perturbation in {zero, shuffle, sign_flip}:
  not present in scripts/picf_action_bridge_capacity_probe.py.
```

Conclusion:

```text
The sampler/mixing/normalization/gradient-surgery family is already concrete
code and has prior experiment coverage.  The FAST/PICF representation branch
and context perturbation branch are not yet runnable claims; they require
implementation before GPU experiments can be called valid.
```

## 2.2 External VLA Code Cross-Check

Reference code was pulled into `/tmp` for local inspection only:

```text
/tmp/vla_foundry
/tmp/openvla_oft
```

Relevant VLA Foundry code:

```text
/tmp/vla_foundry/vla_foundry/data/dataloader.py
  get_datastring_input(...)

The implementation schedules samples per dataset from dataset_weighting, picks
enough shards per dataset, and then builds mixed WebDataset streams.  This
supports the paper-level point that dataset/task mixing is a dataloader-level
contract, not an optimizer trick.
```

Relevant OpenVLA-OFT code:

```text
/tmp/openvla_oft/prismatic/vla/datasets/rlds/traj_transforms.py
  chunk_act_obs(...)

/tmp/openvla_oft/experiments/robot/openvla_utils.py
  get_action_head(...)

/tmp/openvla_oft/vla-scripts/finetune.py
  action hidden states -> L1RegressionActionHead or DiffusionActionHead
  normalized_loss = loss / grad_accumulation_steps
```

External-code conclusion:

```text
The current PICF training stack already contains the corresponding recipe
families:
  - bucket/task weighting,
  - logical update coverage,
  - gradient accumulation semantics,
  - per-bucket scaling,
  - continuous action chunks.

The external code does not justify another sampler-only rerun.  It supports
the current G19/G20 conclusion: if exact-window PICF-vs-noPICF readout does
not separate, the missing piece is action-side representation supervision or
action-expert boundary/capacity, not more blind task mixing.
```

## 3. Runnable G20 Experiments

### G20-A: Remote environment and artifact smoke

Checklist:

```text
[x] ssh reachable
[x] repo exists: /root/openpi_g17_20260604
[x] CUDA visible on both GPUs: 2 x A100-PCIE-40GB
[x] previous fixed-window/probe artifacts available for current_trace32/old_exact32
[x] no stale tmux training occupying GPUs before launch
[x] correct Python env identified: /root/openpi/.venv/bin/python
```

Acceptance:

```text
No experiment starts until the machine state is known.
```

### G20-B: Exact-window bridge sanity

Run only if the same fixed-window JSONL and checkpoint are available.

Two paired modes:

```text
normal_suffix:
  suffix_cross_attention + PICF action condition

no_picf_action:
  suffix_cross_attention + --disable-picf-action-condition
```

Launch record:

```text
remote output root:
  /mnt/picf_g20_action_readout_20260604

window set:
  /mnt/picf_onehour_diag_20260604/current_k12_trace_first32.jsonl

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
    picf_6x40g_k12_acc2_from11000_11100_20260603/11100

normal tmux:
  g20_normal_suffix

no-PICF tmux:
  g20_no_picf_action

common settings:
  steps=120
  windows_per_step=4
  lr=1e-4
  trainable_scope=policy_only
  semantic_trainable_scope=action_head_and_adapter
  deterministic_eval_seed=20260604
```

Interim result at step 10:

```text
normal_suffix:
  step 1  loss_action_default_equiv = 0.027756
  step 10 loss_action_default_equiv = 0.039428
  step 10 recent_loss_action_default_equiv = 0.041189
  pi_action_condition_token_count = 28
  pi_context_adapter_gate ~= 0.119

no_picf_action:
  step 1  loss_action_default_equiv = 0.027431
  step 10 loss_action_default_equiv = 0.040823
  step 10 recent_loss_action_default_equiv = 0.041035
  pi_action_condition_token_count = 0
  pi_context_adapter_gate = 0

throughput:
  normal_suffix ~= 47-48 sec/step
  no_picf_action ~= 43-44 sec/step
```

Interim interpretation:

```text
The bounded suffix PICF action context is not yet showing measurable action
benefit over the no-PICF control on identical windows.  It also carries a
non-trivial throughput penalty in this probe.  This is not a final 120-step
result, but it is already consistent with G17: current action bridge capacity
is unlikely to be the missing fix by itself.
```

Run-health follow-up:

```text
step/progress poll:
  normal_suffix reached progress step 13
  no_picf_action reached progress step 15

The process is alive.  JSON train_step is emitted only at step 1 and log_every
boundaries; progress-bar lines carry intermediate step/loss values.
```

Step 20 paired poll:

```text
normal_suffix:
  step 20 loss_action_default_equiv = 0.037094
  step 20 recent_loss_action_default_equiv = 0.041142
  pi_action_condition_token_count = 28
  pi_context_adapter_gate ~= 0.119

no_picf_action:
  step 20 loss_action_default_equiv = 0.036154
  step 20 recent_loss_action_default_equiv = 0.040919
  pi_action_condition_token_count = 0
  pi_context_adapter_gate = 0

Interpretation:
  The two curves remain effectively tied, with no-PICF slightly better and
  faster.  This strengthens, but does not yet fully finalize, the G17/G20
  negative readout result.
```

Step 30 paired stop decision:

```text
normal_suffix:
  step 30 loss_action_default_equiv = 0.036963
  step 30 recent_loss_action_default_equiv = 0.043508

no_picf_action:
  step 30 loss_action_default_equiv = 0.036098
  step 30 recent_loss_action_default_equiv = 0.043498

Decision:
  Stop the 120-step bridge-only run early.  The paired curves are effectively
  identical through step 30, while normal_suffix is slower.  Continuing this
  run is lower-information than launching explicit context perturbation probes.
```

Mathematical acceptance:

```text
If normal_suffix <= no_picf_action by a meaningful margin on identical windows,
then PICF context has positive action readout value.

If normal_suffix ~= no_picf_action or worse, the current bridge does not solve
the plateau and the next branch must be FAST/action-token representation.
```

### G20-C: Zero/shuffle/sign-flip probe feasibility

Checklist:

```text
[x] inspect probe code for context perturbation hook
[x] found mature implementation in scripts/picf_fixed_window_action_probe.py:
      _install_action_context_probe_mode
[x] patched scripts/picf_action_bridge_capacity_probe.py to reuse it
[x] remote script backed up and updated:
      /root/openpi_g17_20260604/scripts/picf_action_bridge_capacity_probe.py.g20_pre_perturb_bak
[x] remote py_compile passed
```

Acceptance:

```text
Only code-present perturbations may be launched.  After this patch, valid modes
are:
  none
  zero
  token_roll
  shuffle  # alias for token_roll
  sign_flip
  rms_noise
```

### G20-C Result: Action-Context Causal Perturbation

Run record:

```text
remote output root:
  /mnt/picf_g20_action_context_perturb_20260604

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
    picf_6x40g_k12_acc2_from11000_11100_20260603/11100

window set:
  /mnt/picf_onehour_diag_20260604/current_k12_trace_first32.jsonl

common settings:
  steps=0
  eval_before_after=true
  deterministic_eval_seed=20260604
  max_eval_windows=16
  action_context_integration=suffix_cross_attention
  action_context_tokens=24
  action_context_output_gate=0.25
  action_context_stopgrad=true
```

The comparison is same checkpoint, same windows, same deterministic eval seed,
and same action target.  The only changed variable is the content of the PICF
action context immediately before it reaches the PI action path.

First deterministic eval pass:

| mode | `loss_action_default_equiv` | `loss_action` | context tokens | gate | perturb delta rms | post rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0.033852748 | 0.067705496 | 28 | 0.119203 | 0.000000 | 0.000000 |
| `zero` | 0.033643211 | 0.067286421 | 28 | 0.119203 | 0.250000 | 0.000000 |
| `token_roll` | 0.033852748 | 0.067705496 | 28 | 0.119203 | 0.128825 | 0.250000 |
| `sign_flip` | 0.033667957 | 0.067335915 | 28 | 0.119203 | 0.500000 | 0.250000 |

Second deterministic eval pass:

| mode | `loss_action_default_equiv` | `loss_action` | context tokens | gate | perturb delta rms | post rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0.043279073 | 0.086558147 | 28 | 0.119203 | 0.000000 | 0.000000 |
| `zero` | 0.043007695 | 0.086015390 | 28 | 0.119203 | 0.250000 | 0.000000 |
| `token_roll` | 0.043279073 | 0.086558147 | 28 | 0.119203 | 0.128825 | 0.250000 |
| `sign_flip` | 0.043128838 | 0.086257677 | 28 | 0.119203 | 0.500000 | 0.250000 |

Interpretation:

```text
Zeroing, rolling, or sign-flipping the PICF action context does not worsen
action loss on the fixed-window probe.  The bounded suffix action context is
therefore not causally useful to the current action head on this gate.
```

Mathematically, this rejects the hypothesis that the current plateau is fixed
by tuning sampler, optimizer, or bridge strength alone.  If:

```text
L_action(A_theta(X, C)) ~= L_action(A_theta(X, perturb(C)))
```

for zero, token-roll, and sign-flip perturbations, then:

```text
|| d A_theta / d C || is effectively too small or action-irrelevant
```

under the current continuous action objective.  The next non-duplicate branch is
not another task-mixing rerun; it is an action-representation/readout branch
that makes `C` predict motor-relevant symbols or actions explicitly.

### G20-D: FAST auxiliary feasibility

Checklist:

```text
[x] confirm FASTTokenizer exists
[x] confirm TokenizeFASTInputs exists
[x] confirm Pi0FAST CE objective exists
[x] confirm PICF core lacks loss_action_fast_ce
[x] document implementation design
```

Acceptance:

```text
No GPU run should claim FAST has been tested until loss_action_fast_ce exists
in the PICF PyTorch training path and logs its own scalar.
```

## 3.1 Local Validation

```text
git diff --check:
  PASS

python3 -m py_compile:
  scripts/picf_core_train.py
  scripts/picf_action_bridge_capacity_probe.py
  src/openpi/picf/paligemma/wrapper.py
  src/openpi/models/tokenizer.py
  src/openpi/transforms/_base.py
  src/openpi/models/pi0_fast.py
  PASS

python3 -m pytest -q src/openpi/picf/paligemma/wrapper_test.py -k action_context_adapter:
  3 passed
```

One broader pytest command accidentally collected `scripts/train_test.py` and
failed on a local missing optional wandb dependency
`wandb_watchdog.observers.polling`; this is not evidence against the PICF
action-context path and is intentionally not counted as a G20 failure.

## 4. Deployment Decision Rules

If G20-B repeats G17:

```text
normal_suffix ~= no_picf_action
```

then the immediate next code branch is:

```text
PICF-side FAST/action-token representation auxiliary.
```

G20-B and G20-C both match this branch condition.

```text
normal_suffix ~= no_picf_action
none ~= zero ~= token_roll ~= sign_flip
```

Decision:

```text
Do not spend the next GPU cycle on another sampler-only, optimizer-only, or
bridge-strength-only run.  Those method classes are either already implemented
and historically tested, or directly contradicted by the perturbation result.
The next deployable fix is PICF-side action-token/FAST representation
supervision or an equivalent action-readout objective.
```

If G20-B contradicts G17:

```text
normal_suffix clearly improves no_picf_action
```

then inspect:

```text
1. machine/env mismatch;
2. window selection;
3. action bridge LR/scope;
4. whether longer bridge-only fitting can improve frozen-window action.
```

If remote artifacts are missing:

```text
do not invent conclusions;
run only smoke and prepare exact-window artifact regeneration.
```

## 5. Current Expected Final Recommendation

Based on G12-G20, the result is:

```text
training infrastructure is not the missing piece;
action readout/motor representation is the missing piece.
```

The recommended final deployment direction is therefore:

```text
task-balanced logical batch + existing PICF object/belief contract
+ continuous action flow
+ PICF-side FAST/action-token CE auxiliary
+ exact action-context causal probes as the gate
```

This remains scaling-compatible because:

```text
FAST labels require only action chunks, not masks, point clouds, tactile, or SAM;
missing modalities can still feed PICF as typed optional evidence;
the objective does not specialize to CALVIN-only object labels.
```

# PICF-AQR-OWM G22 Action-Readout Auxiliary Gate

Date: 2026-06-04

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Status:

```text
implemented locally; remote 2xA100 G22-K4 validation running.
```

This document is the audit gate for the requested 2025-2026 VLA method list.
It prevents three invalid actions:

```text
1. repeating sampler-only or optimizer-only experiments already covered by G12-G21;
2. claiming FAST/action-token supervision exists in PICF when the generation head is dropped;
3. launching a new run without a theory/code/experiment checklist.
```

## 1. Problem Re-Statement

The current plateau is not explained by batch size alone.  G20 showed that
changing PICF action context did not materially change the continuous action
loss:

```text
L_action(A(X, C)) ~= L_action(A(X, perturb(C))).
```

Therefore the missing factor is:

```text
dA / dC ~= 0
```

where:

```text
X = native PI0.5 image/language/state input
C = bounded PICF belief/action context
A = continuous action expert
Y = target action chunk
```

Changing only task sampling, optimizer state, or gradient surgery can improve
the estimator for `dL/dA`, but it cannot repair a near-zero `dA/dC`.

## 2. Full Method Checklist

| Requested method | Code status | Historical status | G22 decision |
| --- | --- | --- | --- |
| Task-uniform logical batch | implemented in `scripts/picf_core_train.py` | covered by G12/G18 | keep mandatory, do not rerun alone |
| Temperature sampling | implemented | covered by G12-TEMP | keep knob, not current fix |
| Explicit bucket ratios | implemented | covered by G12-RATIO | keep knob, not current fix |
| Dynamic PiKE-style mixing | implemented | covered by G12-DYN | keep off by default |
| Per-bucket `q_b/n_b` loss scaling | implemented | covered by G12 unit/dataflow | keep mandatory |
| Per-bucket action EMA scaling | implemented | covered by G12/G18 | keep recipe knob |
| Scoped PCGrad | implemented | covered by G12-PCG | diagnostic only |
| Scoped CAGrad | implemented | covered by G12-CAGrad | diagnostic only |
| Whole-model PCGrad/CAGrad | not deployed | rejected by compute/mismatch | do not run now |
| Suffix gated action-context attention | implemented in wrapper | G17/G20 showed no action benefit alone | keep optional |
| Action expert router | implemented | not proven as default | evidence-triggered only |
| Continuous action chunks | already native PI0.5 path | current action path | keep mandatory |
| FAST/action-token CE | native OpenPI has FAST, PICF wrapper drops generation heads | missing in PICF | do not fake; use G22 proxy |
| Modality adapters/projectors | PICF has typed projectors/adapters | enough for CALVIN | keep |
| Embodiment adapters/heads | not needed for CALVIN single embodiment | future multi-robot branch | defer |
| System2/System1 split | lacks reliable subtask labels in CALVIN gate | future long-horizon branch | defer |
| Action expert MoE | router exists, full MoE absent | justified only after context is causal | defer |

## 3. G22 New Branch

### 3.1 Why not direct FAST CE now

PICF's restored PI0.5 wrapper explicitly removes generation heads:

```text
src/openpi/picf/paligemma/wrapper.py
  _drop_unused_generation_heads()
```

Native FAST code exists in OpenPI:

```text
src/openpi/models/tokenizer.py
src/openpi/transforms/_base.py
src/openpi/models/pi0_fast.py
```

But the PICF PaliGemma wrapper cannot honestly train FAST CE without restoring
or rebuilding a token generation head.  Doing that inside a 2-3 hour gate would
be a large architecture branch, not a controlled diagnosis.

### 3.2 What G22 implements instead

G22 adds a context-only action-readout auxiliary:

```text
R_eta(C_t) -> \hat{Y}_t
L_readout = Huber_or_MSE(R_eta(C_t), Y_t)
L_train = L_flow + lambda_readout * L_readout
```

Critical constraints:

```text
1. R_eta reads only PICF action-context tokens C_t.
2. R_eta does not read noisy action suffix tokens x_t.
3. loss_action_default_equiv remains canonical PI0.5 flow MSE.
4. Readout metrics are logged separately as pi_context_readout_*.
5. Default lambda_readout = 0, so legacy recipes are unchanged.
```

This is not a replacement for FAST.  It is the fast falsifiable test for the
same missing factor:

```text
Can PICF context carry motor-readable action information?
```

If this cannot be made to decrease under a frozen/controlled gate, direct FAST
CE is unlikely to fix the plateau by itself.

## 4. Code Follow-Through

Implemented files:

```text
src/openpi/picf/paligemma/config.py
  action_context_readout_aux_weight
  action_context_readout_aux_loss
  action_context_readout_aux_huber_delta

src/openpi/picf/paligemma/wrapper.py
  action_context_readout_query
  action_context_readout_q_proj/k_proj/v_proj/out_proj
  _compute_action_context_readout_aux()
  training_total += lambda * L_readout

src/openpi/picf/policy.py
  pi_context_readout_* debug metrics

scripts/picf_core_train.py
  CLI flags
  argument validation
  trainable allowlist
  OWM debug metric logging

src/openpi/picf/paligemma/wrapper_test.py
  readout loss/metric unit test
  trainable-scope coverage
```

## 5. Experiment Checklist

### G22-A: local static/unit gate

Required before remote launch:

```text
[x] py_compile: config/wrapper/policy/train script
[x] wrapper_test readout subset
[x] policy action debug subset
[x] git diff --check
[x] README points to G22/G23
```

### G22-B: remote 2xA100 short gate

Target:

```text
100-300 optimizer steps
task-uniform logical batch
suffix_cross_attention action context
readout auxiliary enabled
progress bar on
no SAM
sidecar weak evidence allowed
```

Actual remote command family:

```text
runner:
scripts/picf_resume_from_args_json.py

base args:
/mnt/checkpoints/picf_core/picf_core/picf_g10a_taskuniform_k4_adapter_norm_20260603_042659/args.json

run:
picf_g22_readout_aux_k4_from11000_to11300_20260604

log:
/mnt/picf_run_logs/picf_g22_readout_k4_300_20260604.log
```

Why K4, not K8:

```text
K8 FSDP run started correctly but measured ~127s/step on 2xA100, so it cannot
produce a 2-3 hour diagnostic.  It was stopped as a throughput mismatch, not as
a method failure.  K4/DDP keeps task-uniform logical-batch normalization and is
the fastest historically validated two-card gate.
```

Current first point:

```text
step=11010
loss_action_default_equiv=0.040747
pi_context_readout_loss=0.051960
pi_context_readout_mse=0.105277
pi_context_readout_token_count=28
pi_context_readout_weighted_total=0.005196
```

Interpretation:

```text
The auxiliary is active and nonzero.  Canonical action loss is comparable to
the G10A baseline at the same step; loss_total is higher because it now includes
the explicit readout term.  Pass/fail still requires 50/100-step trend.
```

50-step update:

```text
step 11010:
  loss_action_default_equiv = 0.040747
  pi_context_readout_loss   = 0.051960
  pi_context_readout_mse    = 0.105277

step 11050:
  loss_action_default_equiv = 0.049536
  pi_context_readout_loss   = 0.031932
  pi_context_readout_mse    = 0.065003
```

Interpretation:

```text
The context-only readout auxiliary is trainable and improves quickly.  This
supports the narrower statement that PICF context contains motor-readable
information once directly supervised.  It does not yet prove that the native
PI0.5 flow action expert uses that context, because loss_action_default_equiv
tracks the historical G10A/G12 class rather than dropping immediately.
Continue to 100 steps before changing the branch.
```

100-step stopped result:

```text
step 11010:
  loss_action_default_equiv = 0.040747
  pi_context_readout_loss   = 0.051960
  pi_context_readout_mse    = 0.105277
  loss_anchor_pv            = 0.489142
  loss_slot_jepa            = 0.691821

step 11100:
  loss_action_default_equiv = 0.052298
  pi_context_readout_loss   = 0.032896
  pi_context_readout_mse    = 0.067227
  loss_anchor_pv            = 0.500822
  loss_slot_jepa            = 0.676992
```

Decision:

```text
G22-K4 is stopped after 100 optimizer steps.
Pass: PICF context is directly motor-readable under an explicit readout
      objective.
Fail/Unsolved: the native PI0.5 flow action expert does not automatically use
               that readable context; loss_action_default_equiv follows the
               historical G10/G12 bucket-difficulty pattern.
```

Implication:

```text
The remaining action plateau is not solved by task-uniform mixing alone and is
not solved by merely appending/suffix-attending PICF context.  The next valid
branch must change the action-head/context training boundary: direct action
token/FAST CE restoration, a stronger train-time bridge that forces the native
action expert to consume the context, or a small action expert capacity branch.
Do not spend the next run on sampler-only, raw-overlap-only, or action-weight
only changes.
```

Primary metrics:

```text
loss_action_default_equiv
pi_context_readout_loss
pi_context_readout_mse
pi_context_readout_weighted_total
pi_context_adapter_gate
pi_context_adapter_attention_entropy_mean
loss_anchor_pv
aqr_same_role_support_overlap_max
```

Pass criteria:

```text
1. pi_context_readout_loss/mse decreases over the first 100-300 steps;
2. loss_action_default_equiv remains comparable, not silently redefined;
3. no NaN/inf or trainable-scope no-op;
4. context metrics are nonzero when action-context tokens are enabled.
```

Failure criteria:

```text
1. readout loss is flat while context token count is nonzero;
2. readout loss decreases but action flow remains context-invariant under perturbation;
3. auxiliary destabilizes canonical action loss immediately.
```

## 6. Deployment Decision Rule

If G22 passes:

```text
Keep current task-balanced infrastructure.
Keep G22 readout auxiliary as a staged motor-readability warmup.
Next branch can be direct FAST CE or action expert MoE only after context
perturbation shows nonzero action sensitivity.
```

If G22 fails:

```text
Do not run more sampler/optimizer-only gates.
The root problem is PICF context content/readability, not update coverage.
Move to direct FAST/generation-head restoration or a smaller action-readable
semantic bridge.
```

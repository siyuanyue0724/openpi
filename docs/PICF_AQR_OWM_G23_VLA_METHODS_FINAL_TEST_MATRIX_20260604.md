# PICF-AQR-OWM G23 VLA Method Matrix and Final Gate

Date: 2026-06-04

Entry point:

```text
src/openpi/picf/README_v2.2.md
```

Purpose:

```text
Convert the full requested 2025-2026 VLA method list into a tracked
experiment/deployment checklist.  Every item must be either implemented and
tested, explicitly rejected with a scaling reason, or deferred behind a missing
data/model prerequisite.
```

## 1. Root-Cause Hypothesis

The action plateau cannot be treated as one optimizer bug.  The current evidence
separates three effects:

```text
H1. Update coverage:
    small physical batches see too few task buckets per optimizer step.

H2. Loss scale:
    action/anchor/aux losses can have incompatible magnitudes per task bucket.

H3. Context causality:
    PI0.5 action may ignore PICF context even when context tokens are present.
```

Mathematically, the desired objective is:

```text
L = sum_b q_b L_b
G = sum_b q_b grad L_b
```

A naive small batch observes:

```text
g_step = grad L_b
Var(g_step) = E_b ||grad L_b - G||^2
```

The production estimator must therefore satisfy:

```text
1. each optimizer step covers multiple task buckets;
2. the backward loss is scaled as an unbiased/bounded logical-batch estimator;
3. the action head must actually depend on the multimodal PICF belief context.
```

## 2. Requested Methods Checklist

### 2.1 Data Mixing and Logical Batch

```text
[x] Task-uniform bucket sampler
    Code: scripts/picf_core_train.py
    Evidence: G10/G12/G18 logs.
    Status: default for current gates.

[x] Temperature sampling q_b ∝ N_b^alpha
    Code: scripts/picf_core_train.py
    Evidence: G12-TEMP.
    Status: implemented, not superior enough to replace task-uniform.

[x] Explicit VLA-Foundry-style bucket ratios
    Code: --calvin-bucket-weight-spec
    Evidence: G12-RATIO.
    Status: useful for future dataset mixture; not current root fix.

[x] Sample without replacement inside logical step
    Code: --calvin-bucket-sample-without-replacement.
    Evidence: current G12/G22 configs.
    Status: production default.

[x] Logical-batch bucket normalization
    Formula: loss_i *= q_b / (n_b / K)
    Code: _logical_batch_loss_scales().
    Evidence: G12 unit/dataflow and active logs.
    Status: production default.
```

### 2.2 Loss Scaling and Dynamic Mixing

```text
[x] Per-bucket action EMA normalization
    Code: --logical-batch-action-bucket-ema-normalization.
    Evidence: G12/G18.
    Status: available knob; off in G22 to isolate context causality.

[x] PiKE-style bounded dynamic mixing
    Code: --logical-batch-dynamic-mixing.
    Evidence: G12-DYN.
    Status: implemented; not first default because context causality was not solved.

[x] Per-bucket metric logging
    Code: --logical-batch-log-bucket-metrics.
    Evidence: active in current gates.
    Status: mandatory for diagnosis.
```

### 2.3 Gradient Conflict

```text
[x] Scoped PCGrad
    Code: --logical-batch-gradient-surgery=pcgrad.
    Evidence: G12-PCGrad.
    Status: diagnostic only; not enough by itself.

[x] Scoped CAGrad
    Code: --logical-batch-gradient-surgery=cagrad.
    Evidence: G12-CAGrad.
    Status: diagnostic only; not enough by itself.

[ ] Whole-model PCGrad/CAGrad
    Status: rejected for current scale.
    Reason: per-task full-model gradients are too expensive and not supported by
    the evidence after G20 showed near-zero context causality.
```

### 2.4 Action Architecture Boundary

```text
[x] Continuous action chunks
    Code: native PI0.5 flow/action path.
    Status: already used; action loss remains default_equiv for comparability.

[x] Action context suffix cross-attention
    Code: action_context_integration=suffix_cross_attention.
    Evidence: G17/G20.
    Status: enabled, but alone did not make action sensitive to context.

[x] Context perturbation causal audit
    Evidence: G20 pass1/pass2.
    Result: none/zero/token_roll/sign_flip were nearly equal.
    Conclusion: action path largely ignored PICF context.

[x] Context-only action readout auxiliary
    Code: G22, action_context_readout_* modules.
    Running: G22-K4 300-step gate on 2xA100.
    Decision: tests H3 directly before more sampler-only gates.
```

### 2.5 Methods Not Deployed As Defaults

```text
[ ] Direct FAST/action-token CE in PICF
    Native OpenPI FAST exists, but PICF wrapper drops generation heads.
    Status: deferred until context-readout gate passes or generation head is restored.

[ ] Full action-expert MoE
    Router hooks exist, full MoE not deployed.
    Status: defer until gradient-cosine/context-causality evidence justifies it.

[ ] Embodiment-specific heads/adapters
    CALVIN is single embodiment.
    Status: future large heterogeneous dataset branch.

[ ] System2/System1 subtask planner
    CALVIN lacks reliable subtask labels for this gate.
    Status: future branch, not current convergence fix.
```

## 3. Current Experiment Gate

### G22-K4 readout gate

Base:

```text
args.json:
/mnt/checkpoints/picf_core/picf_core/picf_g10a_taskuniform_k4_adapter_norm_20260603_042659/args.json
```

Overrides:

```text
exp_name=picf_g22_readout_aux_k4_from11000_to11300_20260604
num_train_steps=11300
resume_checkpoint=/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000
action_context_stopgrad=False
semantic_action_context_readout_aux_weight=0.1
semantic_action_context_readout_aux_loss=smooth_l1
```

Expected logs:

```text
/mnt/picf_run_logs/picf_g22_readout_k4_300_20260604.log
/mnt/checkpoints/picf_core/picf_core/picf_g22_readout_aux_k4_from11000_to11300_20260604/metrics.jsonl
```

Pass criteria:

```text
[x] pi_context_readout_loss decreases within first 50 steps.
[x] loss_action_default_equiv does not become worse than the G10/G12 class at step 11050.
[x] pi_context_readout_token_count > 0.
[x] No NaN/inf through 100 steps.
[x] Logical-batch fields show K=4 and distinct bucket coverage.
```

100-step result:

```text
readout_loss: 0.051960 -> 0.032896
readout_mse : 0.105277 -> 0.067227
action_mse  : 0.040747 -> 0.052298
anchor_pv   : 0.489142 -> 0.500822
slot_jepa   : 0.691821 -> 0.676992
```

Conclusion:

```text
G22 proves PICF context can be decoded into action under direct auxiliary
supervision, but it does not make the native action flow head use that context.
This closes the sampler-only hypothesis for the current plateau.  The next
deployment branch must modify the action/context boundary, not another
bucket-sampler or optimizer-only setting.
```

Fail criteria:

```text
[ ] readout loss flat or absent despite context token count > 0.
[ ] canonical action loss worsens immediately.
[ ] metrics show context still not causally used.
```

## 4. Next Deployment Rule

If G22-K4 passes:

```text
1. keep task-uniform logical batch + bucket normalization as baseline;
2. keep G22 readout as staged motor-readability warmup;
3. run 1000-step confirmation;
4. only then consider direct FAST CE or action-expert MoE.
```

If G22-K4 fails:

```text
1. stop repeating sampler/optimizer-only experiments;
2. restore/build a true action-token/generation auxiliary or a stronger
   context bridge;
3. treat action plateau as context-readability failure, not batch-size failure.
```

## 5. Scaling Constraints

The final scalable solution must preserve:

```text
1. task-balanced logical batches for large heterogeneous datasets;
2. per-task/per-modality loss normalization;
3. modular modality adapters;
4. action/backbone gradient boundary;
5. no dataset-specific shortcuts that only work for CALVIN.
```

Therefore, methods rejected for now are not rejected because they are
"unimportant", but because they either require missing labels, create excessive
per-task gradient cost, or solve a later multi-embodiment branch rather than the
current action plateau.

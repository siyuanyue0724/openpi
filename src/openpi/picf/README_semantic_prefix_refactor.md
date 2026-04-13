# PICF Semantic Prefix Refactor Plan

## Goal

Replace the current `semantic_summary`-driven action conditioning path with a
`pi0.5 / pi0.5-sonata`-style token-level semantic prefix, while preserving the
core PICF mathematical contract:

- current posterior remains language-free
- innovation remains prediction-vs-truth on the language-free physical cache
- `physical_prediction_cache` remains language-free
- language enters only after posterior is fixed

This document is the proposed source of truth for the semantic/action refactor.

## Executive Summary

The current `v29/10000` checkpoint does not meaningfully distinguish tasks or
environments at action time.

The failure is not primarily in the dataset, pointcloud path, or posterior
anchors. The failure is downstream:

- the dataset is sufficiently diverse
- posterior and control states do change across environments
- action remains nearly invariant under both prompt changes and environment
  changes

The strongest architectural reason is that current PICF still treats language as
an external semantic side path:

- `semantic.tokens` are only used by posterior-late semantic reads
- `semantic.summary` is injected as a single compressed token
- action is read from a pooled bottleneck state

This is weaker than `pi0.5-sonata`, where multimodal prefix tokens directly
condition the main action trunk.

## Scope

This refactor is intentionally narrow in one sense and broad in another.

It is narrow because it does **not** redesign:

- pointcloud construction
- tactile gating
- posterior anchor birth / update
- innovation construction
- loss weights

It is broad because it **does** redesign the full semantic-to-action interface:

- semantic token contract
- control-path token composition
- predictive semantic-conditioned path
- action readout structure
- serving and diagnostics

The target is not a patch over the current summary path. The target is a new
posterior-late semantic prefix path that is closer to `pi0.5 / pi0.5-sonata`
without violating PICF's posterior-centered world-model contract.

## Empirical Evidence

### Dataset Diversity

Cloud-side sampling on `task_ABC_D`:

- sampled windows: `800`
- unique prompts: `321`
- same-prompt random pair:
  - `state_l2_avg = 3.56`
  - `img_l2_avg = 1143.80`
  - `action_l2_avg = 1.336`
- largest between-prompt action-mean distance: `2.97`

Conclusion:

- the dataset is not degenerate
- same prompt across environments still requires substantially different actions

### Current Runtime Prompt Sensitivity

Current runtime: `d80872d`, checkpoint:

- `/mnt/checkpoints/picf_core/picf_core/tactile_final_train_30000_full_ddp2_resume5000_langfix_v29/10000`

Measured on `20` windows with current PICF runtime:

- same observation, `actual` vs `blank` prompt:
  - mean action `L2 = 3.22e-06`
  - max `L2 = 1.65e-05`
- same observation, `actual` vs `wrong` prompt:
  - mean action `L2 = 2.72e-06`
  - max `L2 = 1.60e-05`

Conclusion:

- task text currently has near-zero effect on action

### Current Runtime Environment Sensitivity

Measured on same-prompt different-environment pairs:

- predicted action difference:
  - mean `L2 = 3.49e-03`
  - max `L2 = 9.62e-03`
- real dataset action difference on comparable same-prompt pairs:
  - mean `L2 = 3.23e-01`

Conclusion:

- environment sensitivity in action is suppressed by roughly two orders of
  magnitude relative to the data

### Posterior vs Action Decoupling

Same prompt, two different windows:

- `posterior.global_post L2 = 10.78`
- `control_tokens L2 = 43.11`
- `pooled_state L2 = 14.68`
- `action L2 = 2.64e-03`

Conclusion:

- upstream world state is changing
- downstream action readout is collapsing those differences

### Current Semantic Gate State

Current semantic diagnostic on window `0`:

- `prompt_cosine_actual_blank = 0.948`
- `prompt_cosine_actual_wrong = 0.995`
- predictive semantic gates:
  - `[-0.0021, -0.0033]`
- control semantic gates:
  - `[-0.00044, -0.00045]`

Conclusion:

- semantic features are not exactly identical
- token-level semantic reads are effectively closed
- action does not use task-conditioned semantic memory

## Why Current PICF Is Weaker Than pi0.5-sonata

## pi0.5 / pi0.5-sonata

Relevant files:

- [pi0.py](/home/siyuanyue/Documents/openpi/src/openpi/models/pi0.py)
- [pi0_fast_sonata.py](/home/siyuanyue/Documents/openpi/src/openpi/models/pi0_fast_sonata.py)

Their main prefix contract is:

- image tokens
- text tokens
- point tokens when available

All of these enter the same main transformer prefix. Action loss directly
trains that multimodal trunk.

Important properties:

- mixed image+text prefix is correct
- no summary bottleneck is required for action conditioning
- language tokens directly participate in main action generation

## Current PICF

Relevant file:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)

Current control path:

```text
posterior.tokens
+ innovation_token
+ proprio_token
-> control_world
-> append semantic_summary
-> posterior-late semantic reads over semantic.tokens
-> pool
-> action_head
```

Current predictive path:

```text
posterior.tokens
+ global_post
+ proprio
+ action_cond
-> predictive_world
-> physical_prediction_cache
-> append semantic_summary
-> posterior-late semantic reads over semantic.tokens
-> prediction_cache
```

This is weaker than `pi0.5-sonata` because:

- semantic sequence is not part of the main action trunk
- `semantic_summary` is a single compressed token
- action is read from a pooled bottleneck, not a query token over the multimodal
  prefix

## Exact Comparison: pi0.5-sonata vs Current PICF vs Target PICF

| Aspect | `pi0.5 / pi0.5-sonata` | Current PICF | Target PICF |
| --- | --- | --- | --- |
| Image/text composition | mixed prefix | mixed prefix inside `PaliGemmaSemanticFeatures.tokens` | mixed prefix retained |
| Where language enters action path | directly in main prefix | posterior-late, mostly through summary + late reads | posterior-late, but direct token-level semantic prefix |
| Current posterior | not a separate PICF posterior concept | language-free | language-free |
| Physical predictive basis | N/A in same sense | language-free | language-free |
| Main semantic bottleneck | none | `semantic_summary` + pooled action state | no summary bottleneck |
| Action readout | token-level trunk output | pooled state -> linear head | learned action query token(s) |
| Semantic token role | first-class prefix tokens | external memory + summary token | first-class posterior-late prefix tokens |

The target design is not "make PICF equal to pi0.5". It is:

- keep the proven token-level conditioning strength of `pi0.5-sonata`
- keep PICF's world-state / posterior / innovation decomposition
- enforce the plan's language-late physical contract

## Mathematical Constraints That Must Stay True

These constraints come from:

- [plan_readme_ray_geometry.md](/home/siyuanyue/Documents/openpi/plan_readme_ray_geometry.md)
- [README.md](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)

They must not be violated.

### Constraint 1: Current Posterior Remains Language-Free

Language must not modify:

- `_posterior_update(...)`
- current posterior anchor belief
- current posterior global token

This remains the defining PICF belief-state boundary.

### Constraint 2: Innovation Uses the Physical Prediction Basis Only

Innovation must continue to compare:

- current truth
- previous `physical_prediction_cache`

Language must not rewrite the cache used as the innovation baseline.

### Constraint 3: Language Is Posterior-Late

Language can enter:

- control path after current posterior is fixed
- semantic-conditioned future readout after physical predictive basis is fixed

Language cannot enter:

- current posterior
- current observation-anchor evidence fusion
- physical innovation basis

## Target Architecture

## Core Principle

Move from:

- summary-driven posterior-late conditioning

to:

- token-level posterior-late direct conditioning

This keeps PICF mathematically language-late, but makes the action path as
direct as `pi0.5-sonata` at the token level.

## Semantic Summary

`semantic_summary` should no longer exist as a main-path modeling concept.

Target state:

- not used in control path
- not used in semantic-conditioned predictive future readout
- not stored as a required field in formal core state
- optional diagnostics may compute a summary outside the core path if needed

This removes the current compression bottleneck.

The practical implication is strict:

- no `semantic_summary_proj` in the action path
- no `_condition_world_tokens_with_semantic_summary(...)`
- no action dependence that exists only because a single pooled semantic vector
  was appended to world tokens
- if a summary is still kept for logging, it must be outside the formal control
  path and clearly marked diagnostics-only

## Semantic Token Path

Let `T_t^sem` be the valid multimodal semantic token sequence emitted by
PaliGemma. This sequence already contains mixed image+text context and should be
treated as the semantic prefix.

We do need width alignment:

- `T_t^sem` has width `semantic_dim`
- world/control trunk uses `hidden_dim`

So we keep token identity but align width only:

```math
\hat T_t^{sem} = W_{sem} \, T_t^{sem}
```

This is not summary compression. It is the same kind of token-width alignment
that `pi0.5` performs before entering the common trunk.

## Control Path

Current control should be replaced by:

```math
X_t^{ctrl} =
[
T_t^{post},
e_t^{innov},
p_t^{prop},
\hat T_t^{sem},
q_t^{act}
]
```

where:

- `T_t^{post}`: posterior anchor tokens
- `e_t^{innov}`: innovation token
- `p_t^{prop}`: proprio token
- `\hat T_t^{sem}`: projected semantic token sequence
- `q_t^{act}`: learned policy query token

Then:

```math
H_t^{ctrl} = Transformer_{ctrl}(X_t^{ctrl})
```

and action is read only from the policy query token:

```math
\hat a_t = Head_{act}(H_t^{ctrl}[q_t^{act}])
```

This removes the pooled-state bottleneck.

The learned query token is required because empirical diagnostics already show
that the current `control_pool -> control_state_proj -> action_head` path is
collapsing large control-state differences into nearly invariant actions.

## Predictive Path

Physical predictive basis remains unchanged:

```math
X_t^{pred,phys} =
[
T_t^{post},
g_t^{post},
p_t^{prop},
a_t^{cond}
]
```

```math
H_t^{pred,phys} = Transformer_{pred,phys}(X_t^{pred,phys})
```

```math
g_t^{pred,phys}, \, cache_t^{phys} = Readout_{phys}(H_t^{pred,phys})
```

Then semantic-conditioned future readout becomes:

```math
X_t^{pred,sem} =
[
H_t^{pred,phys},
\hat T_t^{sem},
q_t^{pred}
]
```

```math
H_t^{pred,sem} = Transformer_{pred,sem}(X_t^{pred,sem})
```

```math
g_t^{pred}, \, cache_t = Readout_{sem}(H_t^{pred,sem}[q_t^{pred}])
```

So:

- `physical_prediction_cache` remains language-free
- semantic future remains posterior-late
- semantic conditioning is token-level, not summary-level

This is the minimum change set that keeps the innovation contract intact while
making semantic future conditioning strong enough to matter.

## Modality Identity

Direct concatenation requires explicit modality identity tokens or embeddings.

Required token-type identities:

- posterior anchor token type
- innovation token type
- proprio token type
- semantic token type
- control query token type
- predictive query token type

Without these, the joint transformer has no explicit cue for heterogeneous
token roles.

## File-by-File Refactor Plan

The file plan below is exact in the sense that every modified file has a clear
contract change and a defined acceptance test. It is not a placeholder list.

## 0. Source-of-Truth Alignment

Before code changes, align documentation hierarchy:

- source of truth for physical semantics:
  - [plan_readme_ray_geometry.md](/home/siyuanyue/Documents/openpi/plan_readme_ray_geometry.md)
- current implemented behavior:
  - [README.md](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
- target semantic-prefix redesign:
  - this file

Required documentation rule after refactor:

- `README.md` may describe the current implementation
- this file describes the target semantic-prefix redesign and migration steps
- stale references in `PICF_FORMAL_CONTRACT.md` and `HANDOFF_2026-04-10_COMPACT.md`
  must be updated or explicitly marked historical

## 1. `src/openpi/picf/paligemma/wrapper.py`

Current state:

- returns `tokens`
- returns `summary`

Target:

- keep `tokens`
- demote `summary` to optional diagnostics only
- update API comments to state that token stream is the semantic contract

Preferred end state:

- `PaliGemmaSemanticFeatures.tokens` remains required
- `summary` becomes optional or removed from core-facing contract

Exact changes:

- preserve mixed image+text prefix token extraction
- preserve right-padded valid-token contract
- keep returning `tokens`
- stop treating `summary` as part of the downstream action contract
- if compatibility requires temporary retention, annotate `summary` as
  diagnostics-only and schedule removal

Why this file matters:

- this is where semantic information is already available in the correct form
- no extra semantic compression should be introduced here

## 2. `src/openpi/picf/core/contracts.py`

Current state:

- predictive state stores `semantic_summary`

Target:

- remove `semantic_summary` from required predictive state
- if temporary compatibility is needed, mark it diagnostic-only and not part of
  the formal contract
- add fields only if the new design truly needs them, for example:
  - `control_query_state`
  - `predictive_query_state`

Exact contract target:

- `PicfPredictiveState.semantic_tokens`: stays
- `PicfPredictiveState.semantic_summary`: removed from formal runtime contract
- add query-state tensors only if diagnostics or tests need them explicitly
- do not add new fields that reintroduce pooled semantic bottlenecks

## 3. `src/openpi/picf/core/pipeline.py`

This is the main refactor.

Required changes:

- remove `_condition_world_tokens_with_semantic_summary(...)` from the main path
- add `semantic_prefix_proj`
- add learned `control_query_token`
- add learned `predictive_query_token`
- add token-type embeddings for heterogeneous control/predictive tokens
- build joint control prefix:
  - posterior tokens
  - innovation
  - proprio
  - projected semantic tokens
  - control query
- run control transformer over the full joint prefix
- read action from the control query token only
- keep `physical_prediction_cache` production fully language-free
- build semantic-conditioned predictive prefix:
  - physical predictive tokens
  - projected semantic tokens
  - predictive query
- read semantic-conditioned future state from the predictive query token
- remove dependence on semantic late cross-reads as the primary conditioning
  route

Open implementation choice:

- either remove `control_semantic_reads / predictive_semantic_reads`
- or deprecate them and replace them with joint transformer layer counts

The preferred design is to stop treating semantic as an external memory read and
move to direct token-level prefix fusion in the joint trunk.

Exact code-level targets in this file:

- remove `self.semantic_summary_proj`
- remove `_condition_world_tokens_with_semantic_summary(...)`
- add:
  - `self.semantic_prefix_proj`
  - `self.control_query_tokens`
  - `self.predictive_query_tokens`
  - token-type embedding parameters or token-role embeddings
- replace `control_pool -> control_state_proj -> action_head` with:
  - joint control transformer over `[posterior, innovation, proprio, semantic, action-query]`
  - read action only from action-query token(s)
- keep `_posterior_update(...)` untouched w.r.t. semantic inputs
- keep `physical_prediction_cache` creation untouched w.r.t. semantic inputs
- make semantic-conditioned predictive readout token-level and query-based

Important non-change:

- raw observation anchors must **not** bypass posterior into the action path
- only posterior tokens, not raw observation-anchor evidence, are allowed into
  the action/control prefix

## 4. `src/openpi/picf/core/config.py`

Target config changes:

- add token-level semantic prefix controls:
  - `control_query_tokens`
  - `predictive_query_tokens`
  - optional semantic token dropout
- deprecate summary-specific path assumptions
- deprecate or replace:
  - `control_semantic_reads`
  - `predictive_semantic_reads`

Exact config migration plan:

- add new explicit controls:
  - `control_query_tokens: int`
  - `predictive_query_tokens: int`
  - `semantic_prefix_dropout_prob: float`
  - optional `semantic_prefix_max_tokens: int | None`
- keep old semantic-read knobs during migration only
- after refactor lands and tests pass, deprecate old read-count knobs in favor
  of direct joint-trunk layer counts

## 5. `src/openpi/picf/core/training.py`

Loss weights do not need immediate redesign before the structural fix lands.

What must be updated:

- any diagnostics that assume `semantic_summary` is part of the main path
- any semantic cache expectations tied to the old summary injection

Potential later addition, only after structural fix is verified:

- direct control-side semantic supervision if action conditioning is still too
  weak

Exact immediate rule for this file:

- do **not** change loss weights in the same CL as the structural semantic
  prefix refactor
- keep current `loss_action` and auxiliary weighting fixed
- only update diagnostics and caches if they reference removed summary fields

Reason:

- otherwise we cannot attribute behavior changes to the structural fix itself

## 6. `scripts/picf_semantic_diag.py`

Must be rewritten around the new token-level design.

Required diagnostics:

- same observation, `actual / blank / wrong prompt`
- same prompt, different environment
- posterior invariance to prompt
- `physical_prediction_cache` invariance to prompt
- control query state sensitivity to prompt
- predictive query state sensitivity to prompt
- action sensitivity to prompt and environment

Acceptance thresholds to encode in script output:

- prompt sensitivity mean action `L2` must be reported against the current
  pre-refactor baseline `~3e-06`
- same-prompt diff-env mean action `L2` must be reported against the current
  pre-refactor baseline `~3e-03`
- posterior prompt invariance must stay effectively zero
- physical cache prompt invariance must stay effectively zero

Operationally, this script becomes the primary semantic regression gate before
any CALVIN video is trusted.

## 7. `scripts/serve_picf_policy.py`

Serving must use the same token-level runtime path as training.

Required checks:

- checkpoint loading for the new control/predictive query modules
- action inference path uses the new joint control prefix
- no fallback to legacy summary-driven conditioning

Deployment rule:

- no CALVIN evaluation should run on a runtime that still contains legacy
  summary-driven action conditioning
- serving tests must prove the runtime path is the same semantic path as the
  training path

## 8. Documentation

Update:

- [README.md](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
- [PICF_FORMAL_CONTRACT.md](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
- [HANDOFF_2026-04-10_COMPACT.md](/home/siyuanyue/Documents/openpi/HANDOFF_2026-04-10_COMPACT.md)

Required contract correction:

- `semantic_summary` is not a main-path control token
- semantic conditioning is token-level and posterior-late
- action consumes:
  - posterior tokens
  - innovation
  - proprio
  - semantic tokens
  - action query

Also add a short pointer from `README.md` to this file so the current
implementation note and the target refactor note are discoverable from the same
entry point.

## Why This Does Not Weaken PICF Mathematics

This refactor is compatible with PICF because it does not violate the belief
state boundary.

Still true after refactor:

- current posterior is language-free
- innovation baseline is language-free
- `physical_prediction_cache` is language-free
- language enters only after posterior is fixed

What changes is only the downstream action/future conditioning mechanism:

- from summary bottleneck + late memory read
- to direct token-level posterior-late conditioning

This is mathematically consistent with:

- step 16 in `plan_readme_ray_geometry.md`
- `posterior tokens + innovation token + language token + proprio`

The only correction is interpretational:

- `language token` should be understood as a posterior-late language token
  sequence, not a single compressed summary token

## Test Plan

## A. Core Correctness Tests

These must pass before training resumes.

1. prompt changes do not alter current posterior
2. prompt changes do not alter `physical_prediction_cache`
3. prompt changes do alter control query state
4. prompt changes do alter predictive query state
5. prompt changes do alter action by a non-trivial amount
6. same prompt, different environments do alter action by a non-trivial amount

Additional exact unit/integration tests:

7. removing semantic tokens changes action materially more than current baseline
8. removing semantic tokens does not change posterior or physical cache
9. action-query token readout changes while pooled posterior tokens stay within
   expected bounds
10. serving runtime and trainer runtime produce the same action on the same
    checkpoint and same observation/prompt pair

## B. Regression Tests

Keep existing guarantees:

1. tactile gating and `L_pt` behavior unchanged
2. trainer preflight pointcloud contract unchanged
3. calibration-driven `pt_bag_radius_m / sigma_m` unchanged
4. semantic logging remains correct

## C. Quantitative Acceptance

Baseline before refactor on current runtime:

- prompt sensitivity mean action `L2`: `~3e-06`
- same-prompt diff-env mean action `L2`: `~3e-03`

Post-refactor acceptance should require:

- prompt sensitivity to rise by multiple orders of magnitude
- environment sensitivity to rise substantially toward dataset action diversity
- posterior and `physical_prediction_cache` prompt invariance to remain near zero

Concrete initial thresholds:

- prompt sensitivity mean action `L2 >= 1e-3`
- same-prompt diff-env mean action `L2 >= 5e-2`
- posterior prompt invariance mean `L2 <= 1e-6`
- physical-cache prompt invariance mean `L2 <= 1e-6`

These are deliberately not "final quality" thresholds. They are anti-collapse
thresholds proving the action path no longer behaves like a near-constant
policy.

## D. Training Checks

After refactor and resume:

1. smoke from checkpoint
2. `100 / 500 / 1000` step metrics review
3. semantic diagnostic re-run on new checkpoint
4. CALVIN small-batch behavior check

## Recommended Execution Order

1. implement token-level posterior-late semantic prefix fusion
2. remove summary-driven action conditioning
3. add query-token action readout
4. update diagnostics and serving
5. run local unit and integration tests
6. sync cloud runtime
7. resume from stable checkpoint
8. re-run prompt/env sensitivity tests before trusting videos

## Deployment Plan

1. land docs and tests first
2. land wrapper/contracts/pipeline/config/training refactor in one coherent CL
3. run local:
   - semantic diag tests
   - pipeline tests
   - training tests
   - serving tests
4. push and sync cloud runtime
5. run prompt/env sensitivity diagnostics on the same checkpoint before
   resuming training
6. only after those diagnostics pass:
   - restart training from the latest stable checkpoint
   - resume CALVIN behavior evaluation

## Out-of-Scope For This Refactor

These are explicitly not part of the semantic-prefix redesign CL:

- changing pointcloud crop or calibration
- changing tactile gating thresholds
- changing point-tactile alignment loss
- changing optimizer or learning-rate schedule
- changing gradient clip threshold

Those may be revisited later, but they must not be mixed into the semantic-path
refactor if we want a defensible causal readout.

## Final Recommendation

Do not continue building on the current summary-driven action path.

The correct refactor is:

- keep PaliGemma mixed prefix
- keep full semantic token stream
- keep posterior language-free
- keep innovation language-free
- move semantic conditioning into a token-level posterior-late joint control and
  future trunk
- remove `semantic_summary` from the main modeling path

This is the closest design to `pi0.5-sonata` that still preserves PICF's
posterior-centered world-model contract.

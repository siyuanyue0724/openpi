# PICF Task-Anchor Sidecar Deployment Plan

Status: converged Phase 0 / Phase 1 implemented locally. This file now
documents the implemented phase-1 shape, the preserved invariants, and the
remaining rollout work for later phases such as width-384 migration.

Date: 2026-04-14

## 1. Goal

Replace the current raw semantic-prefix conditioning path with a more structured
semantic-guided task readout path, while preserving the existing PICF physical
state and innovation contracts.

The target design is:

- keep one language-free physical posterior and one language-free physical
  predictive cache
- add one semantic-guided task-anchor sidecar for current control and
  conditioned future readout
- do not let semantic tokens write into the physical posterior, physical
  predictive basis, or next-step innovation base

This is not a "second world model". It is one physical core state plus one
task-conditioned readout sidecar.

## 2. Current Code Facts That This Plan Must Respect

These points were verified against the current implementation.

### 2.1 Current observation-anchor path is task-agnostic

Current observation anchors are built by:

- seeding from point FPS over `point_positions`
- reading from full `fused_tokens` using `obs_reader`
- summarizing point attention into `x / S / a`

Current implementation:

- `src/openpi/picf/core/pipeline.py`
  - `_build_observation_anchors(...)`

Important consequence:

- semantic does not currently affect observation-anchor construction
- semantic does not currently affect what gets written into the physical
  posterior

### 2.2 Current physical posterior update is language-free

Current posterior update only uses:

- carried prior
- current observation anchors

Current implementation:

- `src/openpi/picf/core/pipeline.py`
  - `_posterior_update(...)`

Important consequence:

- current posterior is prompt-invariant by design

This is also locked by tests:

- `src/openpi/picf/core/pipeline_test.py`
  - `test_language_is_late_and_does_not_change_current_posterior`

### 2.3 Current control path is posterior-late semantic conditioning

Current control prefix is:

- `posterior.tokens`
- `innovation_token`
- `proprio_token`
- `semantic_prefix_tokens`
- `control_query_tokens`

Current implementation:

- `src/openpi/picf/core/pipeline.py`
  - `_predictive_state(...)`

Important consequence:

- semantic does affect current action
- but only after the physical posterior has already been fixed

### 2.4 Current predictive path is explicitly split

Current predictive path has two levels:

1. physical-only predictive basis
2. semantic-conditioned predictive readout

Current implementation:

- `src/openpi/picf/core/pipeline.py`
  - `_predictive_state(...)`

Current physical predictive basis uses:

- `posterior.tokens`
- `posterior.global_post`
- `proprio_token`
- `action_cond_token`

and produces:

- `physical_prediction_cache`

Then the semantic-conditioned branch appends:

- `semantic_prefix_tokens`
- `predictive_query_tokens`

and produces:

- `prediction_cache`

Important consequence:

- current code already distinguishes:
  - a language-free physical predictive cache
  - a semantic-conditioned predictive cache

### 2.5 Next-step innovation is explicitly protected

Current innovation is built from:

- `previous.predictive.physical_prediction_cache`
- current real targets

Current implementation:

- `src/openpi/picf/core/pipeline.py`
  - `_innovation(...)`

Current tests already enforce:

- semantic changes do not alter `physical_prediction_cache`
- previous semantic-conditioned predictive state does not change next prior or
  next innovation
- previous physical prediction cache does change next innovation

Relevant tests:

- `src/openpi/picf/core/pipeline_test.py`
  - `test_semantic_changes_do_not_pollute_physical_prediction_cache_or_next_innovation`
  - `test_previous_semantic_conditioned_predictive_state_does_not_feed_next_prior_or_innovation`
  - `test_previous_physical_prediction_cache_is_the_only_predictive_cache_allowed_to_change_next_innovation`

### 2.6 Current control path does not explicitly consume `posterior.global_post`

Current control prefix does not include `posterior.global_post`.

Current predictive physical branch does explicitly consume `posterior.global_post`.

Important consequence:

- control currently sees global scene information only implicitly through
  `posterior.tokens`
- this is weaker than an explicit physical global token for control tasks such
  as obstacle avoidance or long-range scene disambiguation

### 2.7 Current code does not "predict next posterior"

This is an important wording correction.

Current code predicts:

- `physical_prediction_cache`
- `prediction_cache`

It does not predict `posterior_{t+1}` directly.

The next posterior is still formed only when the next real observation arrives
and `_posterior_update(...)` runs again.

So the correct phrasing is:

- language does not enter current physical posterior
- language enters current-step control and conditioned future readout

It is not correct to describe the current architecture as "language only enters
next posterior prediction", because that branch does not exist.

## 3. Final Architecture Decision

The final design decision is:

- reject semantic-conditioned physical observation-anchor selection
- reject semantic writeback into the physical posterior
- reject any second recurrent world-state machine
- adopt one language-free physical core state plus one semantic-guided
  task-anchor sidecar

In short:

- one physical core
- one task readout sidecar

## 4. Why Semantic Must Not Rewrite Physical Anchors or Physical Posterior

This refactor explicitly does **not** allow semantic to participate in physical
anchor selection or physical posterior writeback.

If semantic changed physical observation anchors, then:

- same scene + different prompt would produce different physical posterior
- `physical_prediction_cache` would stop being prompt-invariant
- next innovation would become indirectly prompt-conditioned

That would violate the current physical innovation contract and would directly
break existing mathematical guarantees and tests.

The physical branch must remain the answer to:

- what the world is

The task sidecar must answer:

- what the task wants us to focus on in the world

These are different problems and should remain different state pathways.

## 5. Final Target Dataflow

### 5.1 Physical core path

This path stays language-free.

```text
observation_t
-> token_field.fused_tokens
-> observation_anchors_phys
-> posterior_phys
-> physical_prediction_cache
-> next-step innovation
```

Invariants:

- prompt changes do not alter `observation_anchors_phys`
- prompt changes do not alter `posterior_phys`
- prompt changes do not alter `physical_prediction_cache`
- prompt changes do not alter next-step innovation

### 5.2 New semantic-guided task-anchor sidecar

This path does not write physical state. It only selects task-relevant evidence.

```text
semantic_tokens
-> task query conditioner
-> conditioned task queries

conditioned task queries + fused_tokens
-> task-anchor reader
-> task_anchor_tokens
-> task_global_token
-> instruction_token
```

Key rule:

- semantic tokens condition the queries
- the task anchors themselves still carry observation-derived content, not
  language token content

This keeps the sidecar prompt-conditioned but observation-grounded.

### 5.3 Control path

Current control path should be replaced with:

```text
[
  posterior.tokens,
  posterior.global_post,
  innovation_token,
  proprio_token,
  task_anchor_tokens,
  task_global_token,
  instruction_token,
  control_query_tokens,
]
-> control world transformer
-> control_query_state
-> action
```

This explicitly provides both:

- physical global scene context
- task-conditioned local and global context

### 5.4 Predictive conditioned path

Physical predictive basis remains unchanged:

```text
[
  posterior.tokens,
  posterior.global_post,
  proprio_token,
  action_cond_token,
]
-> predictive_world
-> physical_prediction_cache
```

Semantic-conditioned predictive readout becomes:

```text
[
  physical_pred_tokens,
  task_anchor_tokens,
  task_global_token,
  instruction_token,
  predictive_query_tokens,
]
-> predictive_task_world
-> prediction_cache
```

Important rule:

- `prediction_cache` may be task-conditioned
- `physical_prediction_cache` must remain language-free

## 6. Task-Anchor Sidecar Design

### 6.1 Task queries

Use learned queries:

- `K_task = 8` local task queries
- `K_global = 1` global task query

These should be conditioned by semantic tokens before reading fused observation
tokens.

Implemented phase-1 choice:

- use `CrossAttentionRead` to condition the learned task queries with semantic
  prefix tokens
- keep this conditioner outside the physical posterior path
- do not use `GatedCrossAttentionRead` in phase 1, because its zero-initialized
  gate would make an untrained sidecar appear inert in invariance tests

Current modules:

- `src/openpi/picf/core/pipeline.py`
  - `CrossAttentionRead`

### 6.2 Task anchors must read full `fused_tokens`

Task-anchor readout must operate on full `fused_tokens`, not on physical
observation anchors.

Reason:

- if task-anchor selection happens only after physical observation anchors,
  task-relevant but geometrically weak objects may already have been dropped

So task anchors must read:

- point tokens
- visual tokens
- tactile active tokens
- context tokens

through the full fused token field.

### 6.3 Task-anchor geometry

Task anchors must also carry geometry summaries similar to current physical
observation anchors:

- `point_weights`
- `x`
- `S`
- `a`

These should be derived from the task-anchor attention over point tokens.

Reason:

- the control branch should receive not only "task-relevant embeddings"
- it should receive "task-relevant object evidence with spatial support"

This is important for:

- placement
- spatial relation tasks
- obstacle avoidance
- path-level planning

### 6.4 Task-global token

The sidecar must include one dedicated global readout:

- `task_global_token`

This should be produced by a separate global task query reading full
`fused_tokens`, not by simply pooling local task anchors.

Reason:

- local task anchors are good for target objects
- they are not sufficient for free space, remote obstacles, or route-level
  scene structure

### 6.5 Instruction token

The final design should keep one small direct semantic-intent carrier:

- `instruction_token`

This should come from the semantic-conditioned query state, not from raw
semantic-prefix passthrough and not from `semantic_summary`.

Reason:

- some action differences are about intent and relation constraints rather than
  object selection alone
- a single structured instruction token is cleaner than keeping an entire raw
  semantic token stream in the downstream trunk

## 7. Width Plan: 256 First, 384 Second

### 7.1 Current widths

Current code defaults:

- `hidden_dim = 256`
- `posterior_hidden_dim = 256`
- `innovation_dim = 256`
- `control_dim = 256`
- `future_hidden_dim = 256`
- `attention_heads = 8`
- `semantic_cross_dim = 512`

### 7.2 Why 384 is desirable

Increasing per-anchor token width from 256 to 384 should improve:

- anchor content capacity
- task-anchor carrying capacity
- global token expressiveness
- downstream control bandwidth

### 7.3 Why 384 should not be phase 1

Width change is not a cosmetic toggle. It affects:

- nearly every core module projection
- control and predictive trunks
- posterior recurrence hidden width
- checkpoint compatibility

Also, the current trainer validates:

- `hidden_dim % attention_heads == 0`
- `semantic_cross_dim % attention_heads == 0`

With current defaults:

- `semantic_cross_dim = 512`
- `attention_heads = 8`

If attention heads are changed to 12 immediately, `semantic_cross_dim=512`
fails trainer validation.

Therefore:

- first implement the sidecar correctly at width 256
- then widen the system to 384

### 7.4 Width migration recommendation

Phase 1:

- keep:
  - `hidden_dim = 256`
  - `posterior_hidden_dim = 256`
  - `innovation_dim = 256`
  - `control_dim = 256`
  - `future_hidden_dim = 256`
  - `attention_heads = 8`

Phase 2:

- change together:
  - `hidden_dim = 384`
  - `posterior_hidden_dim = 384`
  - `innovation_dim = 384`
  - `control_dim = 384`
  - `future_hidden_dim = 384`
- keep `attention_heads = 8` for the first 384 rollout

Reason:

- `384 / 8 = 48` is valid
- trainer validation still passes with `semantic_cross_dim = 512`
- this avoids introducing a second independent variable at the same time

Later, if desired:

- move `attention_heads -> 12`
- and also change `semantic_cross_dim` to a compatible value

But that should be a separate, smaller follow-up change.

## 8. Checkpoint and Migration Strategy

### 8.1 Phase-1 sidecar at width 256

This phase can still use compatibility migration from current semantic-prefix
checkpoints, because width is unchanged and the refactor is mostly:

- new added parameters
- removed semantic-prefix main-path parameters

The current compatibility loader already handles added/removed keys with
`strict=False`.

### 8.2 Phase-2 widening to 384

This phase is not realistically checkpoint-compatible at the core level.

Changing width from 256 to 384 changes shapes of:

- posterior projections
- recurrence projections
- control trunk
- predictive trunk
- innovation projections
- task sidecar modules

The current compatibility loader can ignore missing or unexpected keys, but it
cannot load mismatched tensor shapes as a meaningful continuation of the same
core state.

Therefore width-384 deployment should be treated as:

- new core initialization
- keep pretrained backbones
- do not expect full core checkpoint resume from width-256 runs

## 9. File-by-File Implementation Plan

### 9.1 `src/openpi/picf/core/config.py`

Implemented sidecar config fields:

- `task_anchor_sidecar_enabled: bool = False`
- `legacy_semantic_prefix_enabled: bool = True`
- `task_anchor_queries: int = 8`
- `task_global_queries: int = 1`
- `task_query_layers: int = 1`
- `task_query_rounds: int = 2`
- `task_anchor_dropout_prob: float = 0.0`

Phase 2 width changes:

- `hidden_dim`
- `posterior_hidden_dim`
- `innovation_dim`
- `control_dim`
- `future_hidden_dim`

Do not change anchor counts in phase 1:

- keep `persistent_anchors = 16`
- keep `observation_anchors = 24`

### 9.2 `src/openpi/picf/core/contracts.py`

Implemented:

- `PicfTaskAnchorState`

Recommended fields:

- `conditioned_queries`
- `tokens`
- `global_token`
- `instruction_token`
- `point_weights`
- `routing_mass_point`
- `routing_mass_visual`
- `x`
- `S`
- `a`
- `semantic_attention`
- `fused_attention`

Implemented in `PicfCoreState`:

- `task_anchors: PicfTaskAnchorState`

Important rule:

- this state is stored for debugging/runtime inspection
- it must not be read by next-step prior or innovation construction

### 9.3 `src/openpi/picf/core/pipeline.py`

Implemented phase-1 modules:

- learned task query tokens
- learned task global query token
- semantic-to-query conditioner
- task-anchor reader over fused tokens
- task-anchor self stack
- instruction pool

Current reuse:

- `CrossAttentionRead` for semantic conditioning of task queries
- `CrossAttentionRead` for reading from full fused tokens

Implemented functions:

- `_build_task_anchors(token_field, semantic) -> PicfTaskAnchorState`

Update control path:

- append explicitly:
  - `posterior.global_post`
  - `task_anchor_tokens`
  - `task_global_token`
  - `instruction_token`
- keep raw semantic prefix behind `legacy_semantic_prefix_enabled` for phase-1
  rollback and A/B comparison

Update conditioned predictive path:

- append:
  - `task_anchor_tokens`
  - `task_global_token`
  - `instruction_token`
- keep raw semantic prefix behind `legacy_semantic_prefix_enabled` for phase-1
  rollback and A/B comparison

Keep unchanged:

- `_build_observation_anchors(...)`
- `_posterior_update(...)`
- `_innovation(...)`
- physical predictive basis construction

### 9.4 `src/openpi/picf/core/training.py`

Phase 1:

- keep main supervision structure unchanged
- do not add a new primary loss just for task anchors

Allow only optional light regularization:

- `task_anchor_diversity`

If added:

- it belongs to the semantic auxiliary group
- it must be budget-capped with semantic auxiliary losses
- it must not affect physical innovation contract

### 9.5 `scripts/picf_core_train.py`

Implemented:

- CLI args for new task-anchor config fields
- compatibility-loader whitelist entries for new sidecar parameters
- fallback-to-default handling in `_build_model(...)` for tests and older
  namespaces

Remaining optional work:

- task-anchor sidecar settings
- explicit note that physical posterior remains language-free

Phase 2:

- update width defaults only when moving to 384 rollout

### 9.6 `scripts/serve_picf_policy.py`

Phase 1 does not require semantic contract changes beyond model loading and
optional future debug export.

Add optional debug export for:

- `task_anchor_attention`
- `task_global_token`
- `instruction_token`

### 9.7 Documentation

Update after implementation:

- `src/openpi/picf/README.md`
- `PICF_FORMAL_CONTRACT.md`

Add explicit wording:

- physical posterior remains task-agnostic
- semantic guides only task-anchor sidecar
- `physical_prediction_cache` remains the only valid next-step innovation base
- raw semantic prefix is no longer the default primary semantic entry

## 10. Red-Line Tests Required Before Training

### 10.1 Physical invariance tests

Fix observation, change semantic:

- `token_field` unchanged
- `observation_anchors_phys` unchanged
- `posterior` unchanged
- `physical_prediction_cache` unchanged
- `innovation_token` unchanged

### 10.2 Sidecar sensitivity tests

Fix observation, change semantic:

- `task_anchors.tokens` change
- `task_global_token` changes
- `instruction_token` changes
- control state changes
- action changes
- semantic-conditioned predictive state changes

### 10.3 No sidecar recurrence leakage

Change only previous task-anchor sidecar state:

- next posterior unchanged
- next innovation unchanged
- next physical prediction cache unchanged

### 10.4 Full-fused-token read test

Task-anchor path must be shown to read directly from full `fused_tokens`, not
from `observation_anchors_phys`.

### 10.5 Global-token presence test

Control prefix must explicitly include:

- `posterior.global_post`
- `task_global_token`

This should be verified by shape and by mutation tests.

### 10.6 Prompt diagnostic tests

For same observation:

- correct prompt
- blank prompt
- wrong prompt

Expected:

- `posterior` nearly unchanged
- `physical_prediction_cache` nearly unchanged
- `task_global_token` changes
- task-anchor attention changes
- action changes materially

## 11. Quantitative Acceptance Targets

These targets are for the first phase after sidecar implementation at width 256.

Expected to remain near zero:

- posterior prompt sensitivity
- `physical_prediction_cache` prompt sensitivity
- innovation prompt sensitivity

Expected to increase substantially:

- prompt-conditioned action sensitivity
- same-prompt different-environment action sensitivity
- task-anchor attention separation between correct / wrong prompts

Suggested initial targets:

- prompt-conditioned mean action `L2 >= 1e-3`
- same-prompt different-environment mean action `L2 >= 5e-2`
- posterior prompt-invariance mean `L2 <= 1e-6`
- physical-cache prompt-invariance mean `L2 <= 1e-6`

## 12. Rollout Order

### Phase 0: freeze current contract

Implemented:

- kept current physical-core contract intact
- added missing red-line tests for sidecar invariance and explicit control
  global-token dependence
- verified trainer, serving, and contract diagnostics locally

### Phase 1: implement task-anchor sidecar at width 256

Implemented:

- `PicfTaskAnchorState`
- semantic-conditioned task queries
- task-anchor readout from full `fused_tokens`
- `task_global_token`
- `instruction_token`
- explicit `posterior.global_post` injection into control
- feature flags:
  - `task_anchor_sidecar_enabled`
  - `legacy_semantic_prefix_enabled`

Current acceptance status:

- local invariance tests pass
- physical posterior invariants remain intact
- trainer and serving regressions are green
- prompt-sensitivity behavior still needs to be re-measured in a fresh run

### Phase 2: widen to 384

- widen core hidden widths together
- keep heads at 8 for first width-384 rollout
- treat this as a new core training run, not a shape-compatible resume

Acceptance for phase 2:

- no regressions in invariance tests
- stable training initialization
- improved task-anchor capacity and control behavior

### Phase 3: optional head-count follow-up

Only after width-384 training is stable:

- consider `attention_heads -> 12`
- also adjust `semantic_cross_dim` to satisfy trainer validation

This is explicitly not part of the first sidecar rollout.

## 13. Final Recommendation

Adopt:

- one language-free physical core
- one semantic-guided task-anchor sidecar

Do not adopt:

- semantic-conditioned physical observation anchors
- semantic writeback into physical posterior
- a second recurrent world-state machine

The correct mental model is:

- one core state
- one task-conditioned readout sidecar

That is the cleanest design compatible with current PICF code, tests, and
innovation mathematics.

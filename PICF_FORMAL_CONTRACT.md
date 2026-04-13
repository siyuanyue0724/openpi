# PICF Formal Contract

This document records the intended state-transition contract for the current
PICF core implementation. It is a persistent design/specification document,
not a temporary handoff note.

This contract is stronger than an informal README description, but weaker than
machine-checked formal proof. It defines the invariants that the implementation
must preserve and the executable tests that currently support those claims.

## 1. Scope

This contract covers the current `src/openpi/picf/core` implementation:

- language-late posterior update
- posterior-late semantic fusion
- explicit innovation construction
- split between physical predictive cache and semantic-conditioned future cache

It does not claim:

- Coq / Lean proof
- TLA+ model checking
- optimality of training dynamics
- global generalization guarantees

The right interpretation is:

- this is a precise engineering contract
- the repository should stay consistent with it
- regression tests should fail if the contract is broken

## 2. State Decomposition

At control step `t`, define:

- observation:
  - `O_t`
- semantic context extracted from current observation and prompt:
  - `S_t = semantic_tokens_t`
  - optional diagnostic aggregate: `semantic_summary_t`
- persistent world/posterior state:
  - `P_t = (posterior_t, predictive_t, control_t, meta_t)`

The important internal subsets are:

- physical posterior/world stream:
  - `posterior_t`
- physical predictive basis:
  - `physical_prediction_cache_t`
- semantic-conditioned future readout:
  - `prediction_cache_t`
- internal action-space contract:
  - normalized `action_t`

## 3. Transition Order

The intended transition order is:

1. Build current runtime metadata and current token field from `O_t`.
2. Build observation anchors from the current token field.
3. Update the current posterior using only:
   - `P_{t-1}`
   - `O_t`
   - current observation anchors
4. Build current targets.
5. Build current innovation by comparing:
   - current true targets
   - `physical_prediction_cache_{t-1}`
6. Build current predictive/control readouts:
   - first world-only readout
   - then posterior-late semantic conditioning
   - semantic tokens may enter the control and semantic-conditioned predictive
     trunks as token-level prefix inputs, provided posterior and the physical
     predictive basis are already fixed

In symbols:

```text
posterior_t = PosteriorUpdate(P_{t-1}, O_t, anchors_t)
innovation_t = Innovation(physical_prediction_cache_{t-1}, targets_t)
physical_pred_t = WorldPredict(posterior_t, innovation_t, proprio_t, action_cond_t)
semantic_pred_t = SemanticRead(physical_pred_t, semantic_tokens_t)
action_t = ActionRead(posterior_t, innovation_t, proprio_t, semantic_tokens_t)
```

The action contract is:

- training-time replay actions are normalized before entering the core
- the core predicts actions in normalized space
- serving / evaluation unnormalize actions before environment execution
- any optional output clipping applies in normalized space, not as a fixed
  physical-unit bottleneck inside the core

## 4. Forbidden Edges

These edges are forbidden by contract.

### F1. No semantic writeback into current posterior

Current semantic inputs must not affect:

- `posterior_t.h`
- `posterior_t.c`
- `posterior_t.mu`
- `posterior_t.Sigma`
- `posterior_t.x`
- `posterior_t.S`
- `posterior_t.a`
- `posterior_t.alpha`

Equivalent statement:

```text
posterior_t ⟂ semantic_t | (P_{t-1}, O_t, anchors_t)
```

### F2. No semantic writeback into carried prior

The next-step prior constructor may depend on:

- previous posterior state
- previous executed action

It must not depend on:

- previous semantic tokens
- previous semantic summary
- previous semantic-conditioned future cache

### F3. No semantic-conditioned future cache into next innovation base

The next-step innovation constructor must read:

- `physical_prediction_cache_{t-1}`

It must not read:

- `prediction_cache_{t-1}`
- `global_pred_{t-1}`
- `semantic_tokens_{t-1}`
- `semantic_summary_{t-1}`

### F4. No fake dual-stream collapse

The architecture must not reduce to:

```text
raw current observation tokens || semantic || innovation -> shared pre-posterior writeback stream
```

Semantic may not participate in the current posterior update. After posterior is
fixed, semantic tokens are allowed to enter downstream control / semantic-
conditioned predictive trunks as same-rank prefix tokens or read-memory, so long
as they do not write back into:

- current posterior
- carried prior
- `physical_prediction_cache`
- next-step innovation base

## 5. Allowed Edges

These edges are intentionally allowed.

### A1. Current semantic may modulate current readout

Current semantic memory may affect:

- current action readout
- current semantic-conditioned future readout

provided that posterior is already fixed.

### A2. Previous physical prediction cache must affect current innovation

If `physical_prediction_cache_{t-1}` changes while current targets stay fixed,
the current innovation is allowed, and generally expected, to change.

### A3. Previous executed action may affect current prior/context

`executed_action_{t-1}` is a legitimate causal input to current prior/context.

## 6. Current Width Contract

Current intended widths are:

- `semantic_dim = 2048`
- `hidden_dim = 256`
- `semantic_cross_dim = 512`

Interpretation:

- semantic stream keeps the wide `PaliGemma` token representation
- world stream remains compact
- posterior-late semantic tokens are projected into `hidden_dim` and then enter
  control / semantic-conditioned predictive trunks as same-rank prefix tokens

Current compatibility note:

- `semantic_cross_dim`
- `predictive_semantic_reads`
- `control_semantic_reads`

are still accepted by configs/checkpoints for migration stability, but they do
not control the current semantic-prefix forward path.

This contract explicitly rejects the older design:

- full semantic tokens or summaries entering the pre-posterior/current-world
  stream

It allows the current design:

- keep full semantic tokens at width `semantic_dim`
- fix posterior and physical predictive basis first
- then project semantic tokens posterior-late into same-width semantic prefixes
  for control / semantic-conditioned predictive trunks

## 7. Predictive Split Contract

Current predictive state contains two distinct future objects.

### 7.1 Physical predictive basis

`physical_prediction_cache_t` is the world-only predictive basis.

Allowed uses:

- future supervision on the physical branch
- next-step innovation construction

Forbidden uses:

- semantic writeback into it after creation

### 7.2 Semantic-conditioned future readout

`prediction_cache_t` is the future readout after semantic conditioning.

Allowed uses:

- future-head supervision
- diagnostic comparison against the physical branch

Forbidden uses:

- next-step innovation construction
- posterior writeback
- carried-prior writeback

## 8. Loss Budget Contract

The current trainer groups non-action losses into:

- `physical_aux`
- `semantic_aux`
- `alignment`

and is allowed to budget-cap those groups against a detached reference derived
from `action_loss`.

This is allowed by contract because it changes optimizer budgeting, not the
causal state-transition graph.

The intended effect is:

- `action` remains the dominant optimization target
- world-model auxiliaries continue to shape the representation
- but auxiliary groups cannot dominate the total objective purely because of
  transient raw scale drift on a batch

## 9. Summary Object Contract

`semantic_summary` is a bookkeeping/diagnostic aggregate only.

It may be stored in state for:

- logging
- debugging
- lightweight summaries

It must not be the main downstream fusion input.

Changing only `semantic_summary` while keeping semantic tokens fixed must not
change:

- current posterior
- current action
- current physical predictive basis
- current semantic-conditioned predictive readout

## 9. Mapping to Current Code

Important implementation anchors:

- prior input / previous executed action:
  - `src/openpi/picf/core/pipeline.py`
- posterior update:
  - `src/openpi/picf/core/pipeline.py`
- innovation constructor:
  - `src/openpi/picf/core/pipeline.py`
- predictive split:
  - `src/openpi/picf/core/pipeline.py`
- state dataclasses:
  - `src/openpi/picf/core/contracts.py`

Most important exact line references at the time of writing:

- `src/openpi/picf/core/pipeline.py#L1404`
- `src/openpi/picf/core/pipeline.py#L1458`
- `src/openpi/picf/core/pipeline.py#L1742`
- `src/openpi/picf/core/pipeline.py#L1796`
- `src/openpi/picf/core/contracts.py#L83`

## 10. Executable Verification

The following tests are part of the practical support for this contract.

- semantic changes do not alter current posterior
- semantic changes do not alter `physical_prediction_cache`
- semantic changes do not alter next-step innovation base
- semantic summary is bookkeeping only
- previous executed action, not previous policy output, drives prior/context
- previous semantic-conditioned predictive state does not feed next prior or innovation
- previous physical prediction cache does feed next innovation

Primary test file:

- `src/openpi/picf/core/pipeline_test.py`

Recommended local command:

```bash
pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/picf_core_train_test.py \
  src/openpi/picf/paligemma/wrapper_test.py
```

Single-entry verification script:

```bash
python scripts/verify_picf_contract.py
```

This script combines:

- static AST-level contract checks
- targeted invariance regressions
- full core regression suite
- local CPU smoke training

## 11. Verification Level

Current verification level is:

- code-path audit
- explicit invariance regressions
- local smoke training
- cloud dual-GPU smoke training
- active long-run cloud training

This is sufficient to support:

- "the implementation currently satisfies the intended engineering contract"

It is not sufficient to support:

- "the implementation is formally proved absolutely correct"

## 12. Open Gaps

The following are still outside the current formalized guarantee.

- machine-checked proof artifact
- proof that current training converges to the intended causal use of state
- proof that semantic shortcutting is impossible under all optimization paths
- proof that `hidden_dim=256` is capacity-optimal

Those are future proof obligations, not current contract violations.

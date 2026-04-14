# PICF Formal Contract

This document records the intended transition contract for the current PICF
core implementation.

It is the durable specification for the present codebase, not a temporary
experiment note.

## 1. Scope

This contract covers:

- language-free physical observation anchors
- language-free physical posterior update
- language-free physical predictive basis
- explicit innovation construction from previous physical predictive cache
- full semantic-prefix-primary control path
- full semantic-prefix-primary conditioned future path
- normalized action-space training contract

It does not claim:

- formal proof
- convergence guarantees
- optimality of training dynamics

## 2. State Decomposition

At control step `t`:

- current observation:
  - `O_t`
- current semantic token sequence:
  - `S_t`
- persistent physical state:
  - `P_t`
- previous physical predictive cache:
  - `C_{t-1}^{phys}`

Important internal state:

- physical token field:
  - `token_field_t`
- physical observation anchors:
  - `anchors_t`
- physical posterior:
  - `posterior_t`
- physical predictive cache:
  - `physical_prediction_cache_t`
- conditioned future cache:
  - `prediction_cache_t`

No task-anchor sidecar state is part of the current contract.

## 3. Transition Order

The intended order is:

```text
token_field_t = Tokenize(O_t)
anchors_t = ObservationAnchors(token_field_t)
posterior_t = PosteriorUpdate(P_{t-1}, O_t, anchors_t)
targets_t = CurrentTargets(O_t)
innovation_t = Innovation(C_{t-1}^{phys}, targets_t)

control_t = ControlRead(
    semantic_prefix_t,
    posterior_t,
    global_post_t,
    innovation_t,
    proprio_t,
    control_query_t,
)

physical_pred_t = PhysicalPredict(
    posterior_t,
    global_post_t,
    proprio_t,
    action_cond_t,
)

semantic_pred_t = ConditionedFutureRead(
    semantic_prefix_t,
    physical_pred_t,
    predictive_query_t,
)
```

## 4. Control Prefix Contract

The control trunk must read the following sequence in this logical order:

```text
[
  semantic_prefix_tokens,
  posterior.tokens,
  posterior.global_post,
  innovation_token,
  proprio_token,
  control_query_tokens,
]
```

The semantic prefix is primary.
The posterior stream is appended as structured physical context.

## 5. Conditioned Future Prefix Contract

The conditioned future trunk must read the following sequence in this logical
order:

```text
[
  semantic_prefix_tokens,
  physical_pred_tokens,
  predictive_query_tokens,
]
```

The physical predictive basis remains the only legal source for next-step
innovation.

## 6. Forbidden Edges

### F1. Semantic must not affect physical observation anchors

Current semantic tokens must not affect:

- observation anchor selection
- observation anchor token values
- observation anchor geometry summaries

### F2. Semantic must not affect physical posterior update

Current semantic tokens must not affect:

- `posterior.h`
- `posterior.c`
- `posterior.mu`
- `posterior.Sigma`
- `posterior.binding`
- `posterior.alpha`
- `posterior.tokens`
- `posterior.global_post`

### F3. Next innovation must read only previous physical predictive cache

Allowed previous predictive inputs:

- `previous.predictive.physical_prediction_cache`

Forbidden previous predictive inputs:

- `previous.predictive.prediction_cache`
- `previous.predictive.global_pred`
- `previous.predictive.semantic_tokens`
- `previous.predictive.predictive_query_state`
- `previous.predictive.control_query_state`

### F4. Conditioned future must not back-write into physical core

The conditioned future branch may affect:

- current action readout
- current conditioned future readout

It must not affect:

- current physical posterior
- current physical predictive cache
- next-step innovation base

## 7. Action Contract

Replay actions are normalized before entering the core when normalization is
enabled.

The core predicts actions in normalized space.

Serving / evaluation must unnormalize actions before environment execution.

Optional action clipping, if enabled, applies in normalized space.

## 8. Expected Behavioral Invariants

When the prompt changes but the observation is fixed:

- physical posterior should remain unchanged
- physical predictive cache should remain unchanged
- innovation should remain unchanged on the next step
- control output may change
- conditioned future output may change

When the observation changes:

- physical posterior may change
- physical predictive cache may change
- innovation may change

## 9. Executable Checks

The current executable contract checks are maintained in:

- `scripts/verify_picf_contract.py`
- `src/openpi/picf/core/pipeline_test.py`

The contract is considered healthy only if:

- targeted invariance regressions pass
- the core regression suite passes
- the smoke training check passes

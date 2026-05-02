# PICF Formal Contract

This document records the concise executable contract for the current local
v2.2 PICF codebase.

Current document roles:

- `src/openpi/picf/README_v2.2.md`
  Broad current-live architecture record and implementation audit.
- `PICF_FORMAL_CONTRACT.md`
  Concise executable contract that `scripts/verify_picf_contract.py` and the
  regression suite are expected to enforce.
- `docs/CALVIN_VALIDATION_README.md`
  Runtime, training, rollout, and validation workflow.
- `src/openpi/picf/README_v2.1.md`
  Historical pre-v2.2 deployment record retained only for reference.

## 1. Scope

This contract covers:

- language-free physical observation anchors
- language-free physical posterior update
- language-free physical predictive basis
- explicit innovation from previous physical predictive cache only
- semantic-conditioned current-step task readout
- one canonical conditioned control state `C_t`
- one final action path through PI0.5 flow matching / denoise sampling
- one world-only predictive basis `K_t^{phys}`
- one conditioned future cache `K_t^{cond}`
- native-width semantic stream (`2048`) at the PI0.5 / conditioned-control side
- physical core width `512`

This contract does not claim:

- formal proof
- convergence guarantees
- optimality of training dynamics

## 2. Canonical State Decomposition

At control step `t`:

- current observation:
  - `O_t`
- semantic token stream:
  - `S_t`
- current token field:
  - `F_t`
- point-pool roles:
  - local effector/contact point pool `P_t^{eff}`
  - global scene/object point pool `P_t^{scene}`
- current-step private dense memory:
  - `M_t^{priv}`
- physical observation anchors:
  - `A_t^{obs}`
- physical recurrent world state:
  - `W_t`
- innovation token:
  - `I_t`
- current-step semantic-conditioned task readout:
  - `R_t`
- canonical conditioned control state:
  - `C_t`
- PI0.5 action-conditioning view of `C_t`:
  - `C_t^{pi}`
- world-only predictive basis:
  - `K_t^{phys}`
- conditioned future cache:
  - `K_t^{cond}`
- executed action:
  - `a_t^{exec}`

The only recurrent physical state family is `W_t`. `R_t` and `C_t` are
current-step conditioned objects, not recurrent world-state replacements.

The training/inference recurrence carrier must therefore remain compact. The
canonical recurrent carry contains only the next-step inputs actually consumed
by the live code:

- `runtime_meta`
- tactile contact hysteresis state
- `posterior`
- `predictive.executed_action`
- `predictive.physical_prediction_cache`

Non-recurrent state such as task readout, conditioned control, semantic token
streams, and conditioned future cache must not be forwarded across the training
window as the canonical `previous` object.

Exact within-step memory reductions that preserve the same v2.2 math are
allowed, including:

- activation recompute / checkpointing
- SDPA replacement for eager attention workspace
- tokenwise sequence chunking for purely token-local projections / FFNs

## 3. Transition Order

The intended order is:

```text
F_t, M_t^{priv} = Tokenize(O_t)
A_t^{obs} = ObservationAnchors(F_t, M_t^{priv})
W_t = PosteriorUpdate(W_{t-1}, O_t, A_t^{obs})
Y_t = CurrentTargets(O_t)
I_t = Innovation(K_{t-1}^{phys}, Y_t)
R_t = TaskReadout(S_t, F_t, M_t^{priv}, proprio_t)
C_t = ConditionedControl(W_t, I_t, R_t, proprio_t)
C_t^{pi} = PiPrefix(C_t)
a_t ~ PI0.5(S_t, C_t^{pi})
K_t^{phys} = PhysicalPredict(W_t, a_t^{exec}, proprio_t)
K_t^{cond} = ConditionedFuture(H_t^{phys_pred}, C_t^{future})
```

Where:

- `H_t^{phys_pred}` is the physical predictive token sequence
- `C_t^{future}` is the future-conditioning token view derived from `C_t`

## 4. Public / Private Readout Contract

The current-step task readout must obey:

- public read memory:
  - `public_read_memory = [fused_tokens, visual_tokens]`
- private dense reread memory:
  - `dense_memory.visual_payload`
  - `dense_memory.tactile_group_tokens`
  - `dense_memory.point_payload`

The public read must not regress to raw pre-fusion
`[point_tokens, tactile_tokens_active, context_tokens]` as the sole public
memory. Existing `fused_tokens` remain the public fused point/tactile/context
representation and must be preserved.

Task readout is current-step only. It may shape `C_t` and `K_t^{cond}`. It must
not become a second posterior or a recurrent task memory.

## 5. Conditioned Control Contract

There must be exactly one canonical conditioned control state:

```text
C_t = G_cond([P_t^{base}, R_t])
```

Where:

- `P_t^{base}` includes:
  - `posterior.tokens`
  - `posterior.global_post`
  - `innovation_token`
  - `proprio_token`
- `R_t` includes:
  - `task_local_tokens`
  - `task_global_token`
  - `instruction_tokens`

`C_t` must retain token-level richness. `C_t^{pi}` is only an interface view of
`C_t` for PI0.5 action conditioning; it is not a second independent control
semantics.

Raw semantic prefix tokens must not be injected directly into the core control
trunk as a parallel primary route.

## 6. Predictive Contract

### 6.1 Physical Predictive Basis

`K_t^{phys}` must be computed only after the executed action is known:

```text
K_t^{phys} = P_phys(W_t, a_t^{exec}, proprio_t)
```

Training uses teacher-forced current first action as `a_t^{exec}`.
Inference uses the sampled first action from the PI0.5 action chunk.

Training-side executed action resolution must be explicit:

- prefer `current.action` when it is present
- otherwise derive `a_t^{exec}` from the first row of `action_chunk_target`

### 6.2 Conditioned Future

`K_t^{cond}` must be computed from token-level physical predictive tokens plus
future-conditioning tokens:

```text
K_t^{cond} = P_cond(H_t^{phys_pred}, C_t^{future})
```

It must not regress to a pure global/cache-only bottleneck, and it must not use
raw semantic-prefix injection as an independent primary route.

## 7. Forbidden Edges

### F1. Semantic must not affect physical observation anchors

Current semantic tokens must not affect:

- observation-anchor selection
- observation-anchor token values
- observation-anchor geometry summaries

Observation-anchor selection may use language-free point-pool roles. The
maintained role split reserves a small effector/contact subset and assigns the
remaining observation anchors to the global scene/object point pool. This does
not relax the semantic isolation rule.

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

Allowed previous predictive input:

- `previous.predictive.physical_prediction_cache`

Forbidden previous predictive inputs:

- `previous.predictive.prediction_cache`
- `previous.predictive.global_pred`
- `previous.conditioned_control`
- `previous.task_readout`
- previous semantic-conditioned future state

### F4. Task readout must not become recurrent world state

Task readout may affect:

- current conditioned control state `C_t`
- current conditioned future cache `K_t^{cond}`

Task readout must not affect:

- physical posterior recursion
- next-step prior
- next-step innovation base

### F5. Final action path must be unique

Formal deployment must not use a direct trainable PICF-local 7D action head.
Final action generation must go through PI0.5:

- training: flow matching objective
- inference: PI0.5 denoise sampler

Serving must fail fast if the PI0.5 action generator is unavailable.

## 8. Action / Normalization Contract

Replay actions are normalized before entering training when normalization is
enabled.

PI0.5 flow training operates in normalized internal action space.

Serving / evaluation must unnormalize actions before environment execution.

Optional clipping, if enabled, applies in normalized space.

External CALVIN action semantics remain 7D-compatible even though internal
action chunk modeling uses the PI0.5 action-space machinery.

## 9. Expected Behavioral Invariants

When the prompt changes but the observation and executed action are fixed:

- physical posterior should remain unchanged
- physical predictive basis should remain unchanged
- next-step innovation should remain unchanged
- task readout may change
- conditioned control may change
- conditioned future may change
- PI0.5 action conditioning may change

When the observation changes:

- physical posterior may change
- physical predictive basis may change
- innovation may change
- task readout may change
- conditioned control may change
- conditioned future may change

## 10. Executable Checks

The current executable contract checks are maintained in:

- `scripts/verify_picf_contract.py`
- `src/openpi/picf/core/pipeline_test.py`
- `src/openpi/picf/core/training_test.py`
- `scripts/picf_core_train_test.py`
- `scripts/serve_picf_policy_test.py`
- `src/openpi/picf/paligemma/wrapper_test.py`

The contract is considered healthy only if:

- targeted invariance regressions pass
- the core regression suite passes
- the smoke training check passes

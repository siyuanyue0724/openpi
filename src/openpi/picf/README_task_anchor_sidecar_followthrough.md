# PICF Task-Anchor Sidecar Follow-Through

Date: 2026-04-14

Status: local implementation completed and strict cloud validation completed for
converged Phase 0 / Phase 1.

This document is a faithful follow-through of the current code, not a future
design sketch. It records:

- what is implemented now
- what was verified locally and on cloud
- the recursive dataflow from replay window to loss and serving
- the exact deployment switches and commands

The corresponding planning document remains:

- [README_task_anchor_sidecar_deployment_plan.md](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_task_anchor_sidecar_deployment_plan.md)


## 1. Scope

Implemented scope is the user-approved converged Phase 0 / Phase 1 only:

- keep one language-free physical core
- do not let semantic affect physical observation-anchor construction
- do not let semantic affect physical posterior update
- do not let semantic affect next-step innovation construction
- add one semantic-guided task-anchor sidecar
- keep width at `256`
- keep `attention_heads=8`
- keep a legacy semantic-prefix rollback switch

Not implemented in this phase:

- width `384`
- `attention_heads=12`
- semantic-conditioned physical anchor selection
- semantic writeback into physical posterior
- a second recurrent world-state machine


## 2. Files Changed For Phase 1

Core implementation:

- [config.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py)
- [contracts.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py)
- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)

Training integration:

- [picf_core_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)
- [picf_resume_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_resume_train.py)
- [verify_picf_contract.py](/home/siyuanyue/Documents/openpi/scripts/verify_picf_contract.py)

Tests:

- [pipeline_test.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline_test.py)
- [picf_resume_train_test.py](/home/siyuanyue/Documents/openpi/scripts/picf_resume_train_test.py)

Planning / audit docs:

- [README_task_anchor_sidecar_deployment_plan.md](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_task_anchor_sidecar_deployment_plan.md)
- [picf_full_audit_20260413.md](/tmp/picf_full_audit_20260413.md)


## 3. What Was Verified Locally

Local regression commands that passed:

```bash
pytest -q src/openpi/picf/core/pipeline_test.py
pytest -q scripts/picf_core_train_test.py
pytest -q scripts/picf_resume_train_test.py
pytest -q scripts/serve_picf_policy_test.py
python scripts/verify_picf_contract.py --skip-full-suite
python scripts/verify_picf_contract.py
```

Observed results:

- `pipeline_test.py`: `37 passed`
- `picf_core_train_test.py`: `37 passed`
- `picf_resume_train_test.py`: `2 passed`
- `serve_picf_policy_test.py`: `6 passed`
- combined regression: `82 passed`
- full contract verifier regression suite: `104 passed`
- contract verifier summary: `PASS`

Additional sidecar-on smoke run:

```python
{
  'task_anchor_available': True,
  'task_anchor_tokens_shape': (8, 64),
  'task_global_shape': (64,),
  'instruction_shape': (64,),
  'posterior_mu_shape': (8, 24),
  'physical_global_pred_shape': (64,),
  'action_shape': (7,),
  'loss_total': 1.0121902227401733,
  'loss_action': 0.7497705221176147,
  'loss_semantic_future_aux': 8.767021179199219
}
```

Interpretation:

- sidecar tokens are produced when enabled
- physical branch still emits the expected posterior / predictive shapes
- transition loss computes end-to-end with sidecar enabled


## 3.1 What Was Verified On Cloud

Cloud host used for validation:

- `px-cloud1.matpool.com:26593`
- isolated work tree:
  - `/root/openpi_sync_sidecar_phase1_20260414`
- base commit in that tree before overlay:
  - `0c2177e`

Only the approved Phase 0 / Phase 1 file set was overlaid onto the clean cloud
tree:

- `scripts/picf_core_train.py`
- `scripts/picf_resume_train.py`
- `src/openpi/picf/core/config.py`
- `src/openpi/picf/core/contracts.py`
- `src/openpi/picf/core/pipeline.py`
- `src/openpi/picf/core/pipeline_test.py`
- `src/openpi/picf/README_task_anchor_sidecar_deployment_plan.md`
- `src/openpi/picf/README_task_anchor_sidecar_followthrough.md`

Cloud regression commands that passed:

```bash
cd /root/openpi_sync_sidecar_phase1_20260414
export PYTHONPATH=$PWD/src
/root/openpi/.venv/bin/python -m pytest -q src/openpi/picf/core/pipeline_test.py
/root/openpi/.venv/bin/python -m pytest -q scripts/picf_core_train_test.py
/root/openpi/.venv/bin/python -m pytest -q scripts/picf_resume_train_test.py
/root/openpi/.venv/bin/python -m pytest -q scripts/serve_picf_policy_test.py
/root/openpi/.venv/bin/python scripts/verify_picf_contract.py --skip-full-suite
/root/openpi/.venv/bin/python scripts/verify_picf_contract.py --skip-smoke
```

Observed cloud results:

- `pipeline_test.py`: `37 passed`
- `picf_core_train_test.py`: `37 passed`
- `picf_resume_train_test.py`: `2 passed`
- `serve_picf_policy_test.py`: `6 passed`
- contract verifier full regression suite: `104 passed`
- contract verifier summary: `PASS`

Cloud sidecar-on smoke:

```python
{
  'task_available': True,
  'task_tokens_shape': (8, 64),
  'task_global_shape': (64,),
  'instruction_shape': (64,),
  'posterior_mu_shape': (8, 24),
  'physical_global_pred_shape': (64,),
  'action_shape': (7,),
  'legacy_semantic_prefix_enabled': False,
  'task_anchor_sidecar_enabled': True,
}
```

Cloud resume-script CLI verification:

```text
--task-anchor-sidecar-enabled / --no-task-anchor-sidecar-enabled
--legacy-semantic-prefix-enabled / --no-legacy-semantic-prefix-enabled
--task-anchor-queries
--task-global-queries
--task-query-layers
--task-query-rounds
--task-anchor-dropout-prob
```

Interpretation:

- the exact Phase 0 / Phase 1 file overlay runs cleanly on cloud
- the sidecar can be enabled in the cloud work tree without changing the
  physical-core invariants
- `picf_resume_train.py` exposes the required sidecar rollout flags for
  checkpoint-based A/B deployment


## 4. Feature Flags And Their Meaning

Current phase-1 flags live in [config.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py) and are surfaced by [picf_core_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py) and [picf_resume_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_resume_train.py).

Flags:

- `task_anchor_sidecar_enabled`
- `legacy_semantic_prefix_enabled`
- `task_anchor_queries`
- `task_global_queries`
- `task_query_layers`
- `task_query_rounds`
- `task_anchor_dropout_prob`

Recommended phase-1 settings:

- `task_anchor_sidecar_enabled=true`
- `legacy_semantic_prefix_enabled=true` for rollback/A-B
- `legacy_semantic_prefix_enabled=false` for sidecar-only experiments

Meaning by combination:

- `sidecar=false`, `legacy=true`
  - legacy semantic-prefix behavior
- `sidecar=true`, `legacy=true`
  - sidecar added, legacy path retained
- `sidecar=true`, `legacy=false`
  - sidecar-only semantic conditioning
- `sidecar=false`, `legacy=false`
  - semantic is effectively removed from downstream control/predictive trunks
  - useful only as a negative ablation


## 5. Recursive Dataflow

This section follows the actual code path recursively from replay data to
runtime state, loss, and serving.

### 5.1 Replay Window Construction

Entry:

- [picf_core_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)
  - `_CalvinTransitionSource`

Flow:

```text
CALVIN reader
-> per-frame numpy payload
-> PicfObservation
-> _TransitionWindow(frames[0], ..., frames[T])
```

Important details:

- `CalvinLangSegmentDataset` provides segment language and frame indices
- each frame includes:
  - `rgb_static`
  - `depth_static`
  - optional `rgb_gripper`
  - `depth_gripper`
  - `robot_obs`
  - `rel_actions`
  - optional tactile data
- action normalization happens here, before the core sees the action

Action path in replay:

```text
raw rel_actions
-> PicfActionNormalizer.normalize_np(...)
-> observation.action
```

Relevant file:

- [action_normalization.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/action_normalization.py)


### 5.2 Trainer Forward

Entry:

- [picf_core_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)
  - `_PicfWindowTrainer.forward(...)`

Per transition:

```text
current frame
+ next frame
-> optional semantic encoder on current frame
-> core.step(current, previous, semantic_override, action_future=current.action)
-> compute_transition_loss(output, next frame, action_target=current.action)
-> accumulate metrics
-> previous = output.state
```

Important detail:

- `action_future=current.action` means conditioned future uses teacher-forced
  current action during training
- the recurrent carry is `previous = output.state`


### 5.3 Semantic Input Construction

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_semantic_context(...)`

Semantic contract:

```text
PaliGemmaSemanticFeatures.tokens
-> _project_semantic_context(...)
-> _SemanticContext(
     tokens=semantic tokens in semantic_dim,
     prefix_tokens=projected hidden_dim tokens,
     available=bool
   )
```

Important:

- core now requires token-level semantic inputs
- summary-only semantic input is rejected
- sidecar uses `semantic.prefix_tokens`
- `semantic.tokens` are still recorded into `PicfPredictiveState.semantic_tokens`


### 5.4 Token Field Construction

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_build_token_field(...)`

Flow:

```text
point features
+ visual map
+ tactile features
+ proprio / previous action / timing / contact context
-> point_tokens
-> visual_tokens
-> tactile_tokens
-> context_tokens
-> fused_tokens
```

This produces [PicfTokenFieldState](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py).

Key invariant:

- task sidecar reads this full `fused_tokens`
- physical observation anchors also read this full `fused_tokens`
- sidecar does not reuse physical observation anchors as its input


### 5.5 Physical Observation Anchors

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_build_observation_anchors(...)`

Flow:

```text
point_positions
-> FPS seeds
-> obs_reader over fused_tokens
-> obs_self
-> observation anchors
-> point_weights / x / S / a
```

Produced state:

- `PicfObservationAnchorState`

Critical invariant:

- semantic does not enter this function
- observation-anchor construction remains task-agnostic


### 5.6 Physical Posterior Update

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_posterior_update(...)`

Flow:

```text
previous posterior
+ current observation anchors
+ current proprio / executed action via prior
-> current prior
-> binding logits
-> sinkhorn-style binding with dustbin/recycle
-> evidence tokens
-> posterior write
-> posterior tokens
-> global_post
```

Produced state:

- `PicfPosteriorAnchorState`

Critical invariant:

- semantic does not enter `_posterior_update(...)`
- current posterior remains language-free


### 5.7 Task-Anchor Sidecar

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_build_task_anchors(...)`

Flow:

```text
semantic.prefix_tokens
+ learned local task query tokens
+ learned global task query tokens
-> CrossAttentionRead semantic conditioner
-> conditioned queries

conditioned queries
+ token_field.fused_tokens
-> task_obs_reader
-> task_anchor_self
-> local task anchors
-> task_global_token
-> instruction_token
-> point attention summaries x / S / a
```

Produced state:

- `PicfTaskAnchorState`

Important facts:

- sidecar is enabled only when `task_anchor_sidecar_enabled=true`
- sidecar consumes `fused_tokens`, not `observation_anchors`
- sidecar is current-step only
- it is stored in `PicfCoreState`
- it does not participate in next-step prior or innovation


### 5.8 Current Targets

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_current_targets(...)`

Targets built from current observation:

- `visual_latent`
- `visual_real`
- `tactile_real`
- `point_real`

This yields:

- `targets`
- `availability`

These are used for:

- current-step innovation residual construction
- next-step training supervision through `extract_targets(...)`


### 5.9 Innovation

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_innovation(...)`

Flow:

```text
previous.predictive.physical_prediction_cache
+ current targets
+ availability
-> standardized residual per branch
-> branch encoders
-> innovation latent
-> innovation_token
-> innovation_norm
```

Critical invariant:

- innovation reads only `previous.predictive.physical_prediction_cache`
- it does not read:
  - `prediction_cache`
  - `global_pred`
  - `task_anchors`
  - semantic-conditioned future state


### 5.10 Control Path

Entry:

- [pipeline.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  - `_predictive_state(...)`

Current phase-1 control prefix:

```text
[
  posterior.tokens,
  posterior.global_post,
  innovation_token,
  proprio_token,
  sidecar tokens:
    - task_anchor_tokens
    - task_global_token
    - instruction_token
    - optional legacy semantic_prefix_tokens
  control_query_tokens,
]
-> control_world
-> control_query_state
-> pooled_state
-> action head
-> normalized action
```

Important facts:

- `posterior.global_post` is now explicitly present in control
- raw semantic prefix remains optional behind `legacy_semantic_prefix_enabled`
- sidecar does not replace physical posterior; it augments downstream reading


### 5.11 Predictive Paths

Physical predictive basis:

```text
[
  posterior.tokens,
  posterior.global_post,
  proprio_token,
  action_cond_token,
]
-> predictive_world
-> physical_pred_tokens
-> physical_global_pred
-> physical_prediction_cache
```

Semantic-conditioned predictive path:

```text
[
  physical_pred_tokens,
  conditioned tokens:
    - task_anchor_tokens
    - task_global_token
    - instruction_token
    - optional legacy semantic_prefix_tokens
  predictive_query_tokens,
]
-> predictive_semantic_world
-> predictive_query_state
-> global_pred
-> prediction_cache
```

Critical invariant:

- `physical_prediction_cache` remains language-free
- `prediction_cache` may be task-conditioned


### 5.12 Runtime State Assembly

Final runtime state is [PicfCoreState](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py):

- `runtime_meta`
- `G_t`
- `token_field`
- `observation_anchors`
- `posterior`
- `task_anchors`
- `predictive`
- `control`

This is the recurrent carry returned by `core.step(...)`.

Important:

- next-step prior and innovation use physical parts of this state
- `task_anchors` are stored for inspection and current-step downstream use
- they are not used as recurrent world memory


### 5.13 Loss

Entry:

- [training.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py)
  - `compute_transition_loss(...)`

Loss groups:

- action:
  - `loss_action_pos`
  - `loss_action_rot`
  - `loss_action_gripper`
- physical auxiliary:
  - `loss_visual_latent`
  - `loss_visual_real`
  - `loss_tactile_real`
  - `loss_point_real`
- semantic-conditioned auxiliary:
  - `loss_semantic_future_aux`
- alignment:
  - `loss_anchor_pv`
  - `loss_pv_weak`
  - `loss_focus_pv`
  - `loss_pt`

Budgeting:

```text
action is primary
physical / semantic / alignment groups are capped relative to detached action loss
```

Important phase-1 fact:

- no dedicated task-anchor primary loss was added
- sidecar learns through existing action and semantic-conditioned future losses


### 5.14 Serving

Entry:

- [serve_picf_policy.py](/home/siyuanyue/Documents/openpi/scripts/serve_picf_policy.py)

Serving flow:

```text
checkpoint dir
-> load runtime args from metadata
-> _normalize_train_args(...)
-> _build_model(...)
-> materialize params
-> load state dict with compat loader if needed
-> infer():
     PicfObservation
     -> optional semantic encoder
     -> core.step(...)
     -> normalized action
     -> action_normalizer.unnormalize_np(...)
     -> websocket response
```

Important:

- serving now shares the normalized-action contract
- `picf_resume_train.py` now supports sidecar flags, so sidecar deployments can
  be resumed cleanly without hand-editing `args.json`


## 6. Invariants That Must Hold

These are the load-bearing invariants for this phase.

Physical invariants:

- prompt changes must not change current posterior
- prompt changes must not change `physical_prediction_cache`
- previous semantic-conditioned future state must not change next innovation
- next innovation may change only if `previous.physical_prediction_cache` changes

Sidecar invariants:

- prompt changes may change:
  - `task_anchors.tokens`
  - `task_global_token`
  - `instruction_token`
  - control readout
  - action
  - semantic-conditioned predictive readout
- sidecar must read full `fused_tokens`
- control must explicitly depend on `posterior.global_post`


## 7. Exact Local Validation Procedure

### 7.1 Core regression

```bash
pytest -q src/openpi/picf/core/pipeline_test.py
```

### 7.2 Trainer regression

```bash
pytest -q scripts/picf_core_train_test.py
```

### 7.3 Serving regression

```bash
pytest -q scripts/serve_picf_policy_test.py
```

### 7.4 Contract smoke

```bash
python scripts/verify_picf_contract.py --skip-full-suite
```

### 7.5 Combined quick gate

```bash
pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  scripts/picf_core_train_test.py \
  scripts/serve_picf_policy_test.py
```


## 8. Concrete Deployment Commands

### 8.1 Fresh training from full `picf_core_train.py`

Sidecar + legacy fallback retained:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  ...existing args... \
  --task-anchor-sidecar-enabled \
  --legacy-semantic-prefix-enabled
```

Sidecar-only:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  ...existing args... \
  --task-anchor-sidecar-enabled \
  --no-legacy-semantic-prefix-enabled
```

### 8.2 Resume from `args.json` and checkpoint

Resume script now supports phase-1 sidecar overrides directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_resume_train.py \
  --args-json /path/to/args.json \
  --resume-checkpoint /path/to/checkpoint_dir \
  --exp-name sidecar_phase1_ab \
  --device cuda \
  --task-anchor-sidecar-enabled \
  --legacy-semantic-prefix-enabled
```

Sidecar-only resume:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_resume_train.py \
  --args-json /path/to/args.json \
  --resume-checkpoint /path/to/checkpoint_dir \
  --exp-name sidecar_phase1_only \
  --device cuda \
  --task-anchor-sidecar-enabled \
  --no-legacy-semantic-prefix-enabled
```

Optional sidecar hyperparameters are also exposed:

- `--task-anchor-queries`
- `--task-global-queries`
- `--task-query-layers`
- `--task-query-rounds`
- `--task-anchor-dropout-prob`


## 9. Recommended Deployment Order

1. Run the local validation commands in Section 7.
2. Start with `sidecar=true` and `legacy=true`.
3. Verify:
   - no physical invariant regressions
   - action prompt sensitivity improves
   - environment sensitivity remains non-trivial
4. Then run `sidecar=true` and `legacy=false`.
5. Compare:
   - action sensitivity
   - physical invariance
   - training stability
6. Only after this should width `384` be considered.


## 10. Current Conclusion

As of this document:

- Phase 0 / Phase 1 is implemented locally
- physical core invariants are preserved
- task-anchor sidecar is live behind feature flags
- explicit `global_post` control injection is implemented
- trainer / serving / contract verification are green
- resume deployment path supports sidecar flags

The remaining work is experimental rather than structural:

- run sidecar A/B training
- measure prompt sensitivity and behavior
- only then consider width `384`

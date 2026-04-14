# PICF Task-Anchor Sidecar Final Handoff

Date: 2026-04-14

Status: current code audited, local validation completed, and strict cloud
validation completed for the current single-core + task-sidecar implementation.

This document is the single final handoff document for the current
task-anchor-sidecar implementation. It is a faithful follow-through of the
current code, not a future design sketch. It records:

- what is implemented now
- what was verified locally and on cloud
- the recursive dataflow from replay window to loss and serving
- the exact deployment switches and commands
- the current rollout strategy and acceptance gates

For the current CALVIN dataset / evaluator validation workflow, use:

- [CALVIN_VALIDATION_README.md](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)

It supersedes the previously split sidecar notes:

- `README_task_anchor_sidecar_deployment_plan.md`
- `README_semantic_prefix_refactor.md`


## 1. Scope

Implemented scope in current code is:

- keep one language-free physical core
- do not let semantic affect physical observation-anchor construction
- do not let semantic affect physical posterior update
- do not let semantic affect next-step innovation construction
- add one semantic-guided task-anchor sidecar
- current default widths are `384`
- current `attention_heads=8`
- keep a legacy semantic-prefix rollback switch

What this means in practice:

- the architecture boundary is still the approved one:
  - one physical recurrent core
  - one non-recurrent semantic-guided sidecar
- but the current repo has already widened the core defaults to `384`
- so the code is internally self-consistent, but it no longer exactly matches
  the earlier "Phase 1 stays at 256" planning constraint

Not implemented in current code:

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

- [picf_full_audit_20260413.md](/tmp/picf_full_audit_20260413.md)


## 2.1 Code / plan divergence that must be stated explicitly

The current repo defaults in
[config.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py) are:

- `hidden_dim = 384`
- `posterior_hidden_dim = 384`
- `innovation_dim = 384`
- `control_dim = 384`
- `future_hidden_dim = 384`
- `attention_heads = 8`

This is load-bearing because:

- trainer CLI defaults inherit these values
- model materialization uses these values
- cloud and local tests that passed were run against this exact width

Therefore the truthful statement is:

- the current code is mathematically and programmatically consistent with
  `single physical core + semantic-guided sidecar + width 384 + heads 8`
- it is not literally the same as the earlier planning statement
  "`Phase 1 first stays at 256`"

If strict adherence to the older planning scope is required, the width defaults
must be changed back in code before deployment. This document does not hide that
difference.


## 3. What Was Verified Locally

Local regression commands that passed:

```bash
pytest -q src/openpi/picf/core/pipeline_test.py
pytest -q scripts/picf_core_train_test.py
pytest -q scripts/picf_resume_train_test.py
pytest -q scripts/serve_picf_policy_test.py
pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/picf_core_train_test.py \
  scripts/picf_resume_train_test.py \
  scripts/serve_picf_policy_test.py \
  src/openpi/picf/action_normalization_test.py \
  src/openpi/picf/pointcloud_picf_test.py \
  src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py
python scripts/verify_picf_contract.py --skip-full-suite
python scripts/verify_picf_contract.py
```

Observed results:

- `pipeline_test.py`: `37 passed`
- `picf_core_train_test.py`: `49 passed`
- `picf_resume_train_test.py`: `2 passed`
- `serve_picf_policy_test.py`: `6 passed`
- wider local regression: `135 passed`
- entrypoint regression (`resume + serve + action_normalization`): `11 passed`
- full contract verifier regression suite: `114 passed`
- contract verifier summary: `PASS`

Additional sidecar-on runtime smoke:

```python
{
  'hidden_dim': 384,
  'task_anchor_sidecar_enabled': True,
  'legacy_semantic_prefix_enabled': False,
  'task_available': True,
  'task_tokens_shape': (8, 384),
  'task_global_shape': (384,),
  'instruction_shape': (384,),
  'loss_total': 2.5598,
  'loss_action': 1.9441,
  'loss_semantic_future_aux': 0.5185,
  'posterior_mu_diff_prompt': 0.0,
  'physical_global_pred_diff_prompt': 0.0,
  'task_tokens_diff_prompt': 31.8046,
  'task_global_diff_prompt': 11.2446,
  'instruction_diff_prompt': 9.3210,
  'action_diff_prompt': 0.0556,
  'next_posterior_mu_diff_prev_prompt': 0.0,
  'innovation_token_diff_prev_prompt': 0.0,
  'next_physical_global_pred_diff_prev_prompt': 0.0
}
```

Interpretation:

- sidecar tokens are produced when enabled
- prompt changes affect task sidecar and action
- prompt changes do not affect physical posterior
- prompt changes do not affect physical predictive state
- previous semantic-conditioned state still does not affect next innovation

Additional percentile gradient-clip CLI smoke:

```python
[
  {
    'step': 1,
    'threshold_ready': False,
    'history_size': 1,
    'threshold': 0.0,
    'applied': False,
  },
  {
    'step': 2,
    'threshold_ready': False,
    'history_size': 2,
    'threshold': 0.0,
    'applied': False,
  },
  {
    'step': 3,
    'threshold_ready': False,
    'history_size': 3,
    'threshold': 0.0,
    'applied': False,
  },
  {
    'step': 4,
    'threshold_ready': True,
    'history_size': 3,
    'threshold': 75.0669,
    'applied': False,
  }
]
```

Checkpoint metadata after the same smoke:

```python
{
  'controller_mode': 'percentile',
  'controller_history_len': 3,
}
```

Interpretation:

- percentile mode starts with no clipping
- the threshold only becomes active after the configured history window fills
- clip history is persisted in checkpoint metadata and restored on resume when
  configuration matches


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

Cloud sidecar-on runtime smoke:

```python
{
  'hidden_dim': 384,
  'legacy_semantic_prefix_enabled': False,
  'task_anchor_sidecar_enabled': True,
  'task_available': True,
  'task_tokens_shape': (8, 384),
  'task_global_shape': (384,),
  'instruction_shape': (384,),
  'posterior_mu_diff_prompt': 0.0,
  'physical_global_pred_diff_prompt': 0.0,
  'task_tokens_diff_prompt': 31.8046,
  'task_global_diff_prompt': 11.2446,
  'instruction_diff_prompt': 9.3210,
  'action_diff_prompt': 0.0556,
  'next_posterior_mu_diff_prev_prompt': 0.0,
  'innovation_token_diff_prev_prompt': 0.0,
  'next_physical_global_pred_diff_prev_prompt': 0.0
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
- missing tactile calibration artifacts on cloud were rebuilt using the full
  repo script
  [precompute_tactile_contact_calibration.py](/home/siyuanyue/Documents/openpi/scripts/calvin/precompute_tactile_contact_calibration.py),
  not with placeholder payloads

Additional cloud training probes completed on `3 x A100 40GB`:

```python
{
  'probe': 'accum_steps=4',
  'effective_global_batch': 12,
  'save_path': '/mnt/checkpoints/picf_core/picf_core/sidecar_probe_acc4_save1/1',
  'loss_total': 4.3996,
  'preclip_grad_norm': 153.9567,
  'grad_clip_threshold_ready': False,
  'steps_per_sec': 0.04043,
  'windows_per_sec': 0.16171,
  'checkpoint_saved': True,
}
{
  'probe': 'accum_steps=8',
  'effective_global_batch': 24,
  'save_path': '/root/checkpoints_probe/picf_core/sidecar_probe_acc8_local/1',
  'loss_total': 4.4535,
  'preclip_grad_norm': 123.6511,
  'grad_clip_threshold_ready': False,
  'steps_per_sec': 0.02190,
  'windows_per_sec': 0.17523,
  'checkpoint_saved': True,
}
```

Interpretation:

- the current sidecar code runs correctly on `3 x A100 40GB`
- percentile clipping behaves as designed: no clipping before the history window
  is full
- `accum_steps=4` is the safer larger-batch production start
- `accum_steps=8` is viable and increases effective global batch further, but
  optimizer updates are materially slower

Cloud note for the new gradient-clip mode:

- the new percentile gradient-clip mode was fully validated locally, including
  real CLI training smoke and checkpoint persistence
- a cloud rerun for this exact change set was attempted on 2026-04-14
- SSH to `px-cloud1.matpool.com:26593` was reset during banner / key-exchange,
  so a cloud rerun for the clip-mode change could not be completed in this
  session
- therefore the cloud validation above is authoritative for the final sidecar
  architecture, and the new clip-mode validation is authoritative locally only


## 4. Feature Flags And Their Meaning

Current phase-1 flags live in [config.py](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py) and are surfaced by [picf_core_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py) and [picf_resume_train.py](/home/siyuanyue/Documents/openpi/scripts/picf_resume_train.py).

Flags:

- `grad_clip_mode`
- `grad_clip_norm`
- `grad_clip_percentile`
- `grad_clip_window`
- `task_anchor_sidecar_enabled`
- `legacy_semantic_prefix_enabled`
- `task_anchor_queries`
- `task_global_queries`
- `task_query_layers`
- `task_query_rounds`
- `task_anchor_dropout_prob`

Recommended phase-1 settings:

- current default trainer behavior:
  - `grad_clip_mode=percentile`
  - `grad_clip_percentile=75`
  - `grad_clip_window=100`
  - no clipping until the first 100-step history window has filled
- fixed clipping:
  - `grad_clip_mode=fixed`
  - `grad_clip_norm=1.0` or another explicit threshold
- percentile clipping:
  - `grad_clip_mode=percentile`
  - `grad_clip_percentile=75`
  - `grad_clip_window=100`
  - clipping stays disabled until the window has filled
- `task_anchor_sidecar_enabled=true`
- `legacy_semantic_prefix_enabled=true` for rollback/A-B
- `legacy_semantic_prefix_enabled=false` for sidecar-only experiments

Meaning by combination:

- `grad_clip_mode=fixed`
  - use the constant `grad_clip_norm` threshold
- `grad_clip_mode=percentile`
  - compute the threshold from the previous sliding window of
    `preclip_grad_norm`
  - with `75 / 100`, only the top 25% of recent gradient norms are clipped
  - no clipping is applied before the history window is full
- `sidecar=false`, `legacy=true`
  - legacy semantic-prefix behavior
- `sidecar=true`, `legacy=true`
  - sidecar added, legacy path retained
- `sidecar=true`, `legacy=false`
  - sidecar-only semantic conditioning
- `sidecar=false`, `legacy=false`
  - semantic is effectively removed from downstream control/predictive trunks
  - useful only as a negative ablation

Current implementation detail:

- percentile clip state is persisted in checkpoint metadata as
  `metadata['grad_clip_controller']`
- the history is restored only when
  `mode / fixed_norm / percentile / window` all match current runtime config
- hot-switching the clip mode during a live process is not implemented
- changing clip mode still requires a restart or resume


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

Recurrent interpretation:

- `previous.posterior` is the only persistent world-state carrier
- `previous.predictive.physical_prediction_cache` is the only legal predictive
  input to next-step innovation
- `previous.task_anchors` may be stored in the state for inspection, but they
  are not part of the recurrent world-model contract


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

`PicfTokenFieldState` fields and meanings:

- `point_tokens`
  - point / geometry tokens after projection
- `visual_tokens`
  - CLIP/V-JEPA-derived visual tokens projected to core width
- `tactile_tokens`
  - active tactile tokens that survived contact gating
- `context_tokens`
  - proprio, previous action, timing, and contact-summary context tokens
- `fused_tokens`
  - multimodal field after token fusion; this is the full field both physical
    anchors and the task sidecar read
- `point_positions`
  - point coordinates used for physical anchor geometry summaries
- `modality_ids`
  - token modality labels
- `point_align_embeddings`, `visual_align_embeddings`, `tactile_align_embeddings`
  - alignment-space projections for geometric/routing losses
- `tactile_positions_world`, `tactile_normals_world`
  - world-frame tactile geometry
- `tactile_contact_gate`, `tactile_contact_prob`, `tactile_anchor_mask`
  - tactile gating state
- `fusion_attention_mean`
  - optional diagnostic mean attention from token fusion
- `projective_geometry`
  - point-to-image projective compatibility state used by alignment losses

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

`PicfObservationAnchorState` fields:

- `seed_indices`
  - FPS-selected point seeds
- `tokens`
  - observation anchor tokens after task-agnostic readout
- `point_weights`
  - normalized point attention weights per anchor
- `routing_mass_point`, `routing_mass_visual`
  - raw routing attention mass into point/visual subsets
- `routing_support_point`, `routing_support_visual`
  - aggregate support mass over tokens
- `routing_gate_point`, `routing_gate_visual`
  - gated support statistics used by alignment objectives
- `x`, `S`, `a`
  - position, covariance, and extent summaries inferred from point attention

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

`PicfPosteriorAnchorState` fields:

- `h`, `c`
  - recurrent posterior hidden/cell state
- `mu`, `Sigma`
  - latent posterior mean and covariance
- `x`, `S`, `a`
  - anchor geometry summary after write
- `alpha`
  - anchor activity
- `contact_prob`
  - posterior-side contact belief
- `support_mass`
  - evidence mass assigned to each persistent anchor
- `recycle_gate`
  - how much of each anchor is reset/recycled
- `binding`
  - anchor-to-observation assignment matrix
- `evidence_tokens`
  - bound observation evidence per persistent anchor
- `tokens`
  - persistent anchor tokens used downstream
- `global_post`
  - pooled physical global scene token

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

`PicfTaskAnchorState` fields:

- `conditioned_queries`
  - learned task queries after semantic conditioning
- `tokens`
  - local task-anchor tokens
- `global_token`
  - task-global token read directly from full `fused_tokens`
- `instruction_token`
  - pooled query-level instruction token
- `point_weights`
  - task-anchor attention over point tokens
- `routing_mass_point`, `routing_mass_visual`
  - raw task-anchor routing mass
- `x`, `S`, `a`
  - task-anchor geometry summaries
- `semantic_attention`
  - attention of learned queries into semantic prefix tokens
- `fused_attention`
  - attention of conditioned queries into the full fused token field
- `available`
  - whether the sidecar produced usable anchors for this step

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

`PicfPredictiveState` is the main current-step downstream carrier:

- `semantic_tokens`
  - raw semantic tokens at semantic width, retained for diagnostics
- `innovation_token`, `innovation_norm`
  - current-step explicit innovation signal and per-branch norms
- `availability`
  - branch availability mask for current targets
- `control_tokens`
  - full control trunk token sequence after control transformer
- `control_query_state`
  - query-token readout used for control
- `pooled_state`
  - projected control query readout passed to action head
- `action`
  - normalized current-step action prediction
- `executed_action`
  - teacher-forced or observed executed action carried for next-step prior
- `physical_global_pred`
  - language-free predictive global token
- `physical_prediction_cache`
  - language-free cache used by next-step innovation
- `predictive_query_state`
  - semantic-conditioned predictive query readout
- `global_pred`
  - alias of `predictive_query_state` in current code; not a separate pooled
    scene-global variable
- `prediction_cache`
  - semantic-conditioned future cache


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

Exact loss formulas from current code:

- action:
  - `loss_action_pos = L1(action[:3], target[:3])`
  - `loss_action_rot = L1(action[3:6], target[3:6])`
  - `loss_action_gripper = L1(action[6:], target[6:])`
  - `loss_action = lambda_pos * pos + lambda_rot * rot + lambda_gripper * gripper`
- physical future:
  - `loss_visual_latent = MSE(physical_cache.visual_latent, future.visual_latent)`
  - `loss_visual_real = L1(physical_cache.visual_real, future.visual_real)`
  - `loss_tactile_real = tactile_map + tactile_aux`
  - `loss_point_real = BCEWithLogits(physical_cache.point_real, future.point_real)`
- semantic future:
  - same branch-wise form, but evaluated on `prediction_cache`
  - aggregated as `loss_semantic_future_aux`
- alignment:
  - `loss_anchor_pv`
  - `loss_pv_weak`
  - `loss_focus_pv`
  - `loss_pt`

Budget capping in current code:

- `physical_aux_capped = budget(physical_aux, action_loss, ratio=aux_budget_physical_ratio)`
- `semantic_group_capped = budget(lambda_semantic_future_aux * semantic_future_aux, action_loss, ratio=aux_budget_semantic_ratio)`
- `alignment_group_capped = budget(alignment.total, action_loss, ratio=aux_budget_alignment_ratio)`
- `loss_total = action_loss + physical_aux_capped + semantic_group_capped + alignment_group_capped`

Interpretation:

- action remains the primary optimization target
- auxiliary branches are allowed to learn
- but they are not allowed to dominate total gradient budget when action loss is
  much smaller


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
- serving loads model weights with strict load first, then compatibility loader
  fallback; this matters for sidecar rollouts resumed from older checkpoints


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

### 7.6 What the verifier proves

`python scripts/verify_picf_contract.py` currently checks both static and
runtime regression conditions, including:

- step order:
  - `posterior -> task_anchors -> innovation -> predictive`
- sidecar reads full `fused_tokens`
- control explicitly includes `posterior.global_post`
- prompt changes do not change physical branch state
- previous semantic/task-conditioned branch does not change next innovation

This script is the closest thing in-repo to an executable contract checker for
the present architecture.

### 7.7 Most recent validation results

The commands above were not just proposed; they were rerun on the current
single-core + sidecar code.

Local workstation results:

```bash
pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/picf_core_train_test.py \
  scripts/picf_resume_train_test.py \
  scripts/serve_picf_policy_test.py \
  src/openpi/picf/action_normalization_test.py \
  src/openpi/picf/pointcloud_picf_test.py \
  src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py
```

Result:

- `127 passed`

Local executable contract gate:

```bash
python scripts/verify_picf_contract.py
```

Result:

- targeted invariance regressions: `11 passed`
- core regression suite: `106 passed`
- smoke training check: `PASS`
- overall summary: `PASS`

Cloud isolated worktree:

- path: `/root/openpi_sync_sidecar_phase1_20260414`
- note: the cloud worktree initially lagged behind the final local code; the
  final file set was resynced before the results below were recorded

Cloud broad pytest:

```bash
cd /root/openpi_sync_sidecar_phase1_20260414
export PYTHONPATH=$PWD/src
/root/openpi/.venv/bin/python -m pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/picf_core_train_test.py \
  scripts/picf_resume_train_test.py \
  scripts/serve_picf_policy_test.py \
  src/openpi/picf/action_normalization_test.py \
  src/openpi/picf/pointcloud_picf_test.py \
  src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py
```

Result:

- `125 passed`

Cloud executable contract gate:

```bash
cd /root/openpi_sync_sidecar_phase1_20260414
export PYTHONPATH=$PWD/src
/root/openpi/.venv/bin/python scripts/verify_picf_contract.py
```

Result:

- targeted invariance regressions: `11 passed`
- core regression suite: `104 passed`
- smoke training check: `PASS`
- overall summary: `PASS`

### 7.8 Runtime smoke with sidecar enabled

In addition to pytest/verifier coverage, a direct sidecar-on forward/loss smoke
was rerun on the current default-width code path.

Configuration:

- `task_anchor_sidecar_enabled=True`
- `legacy_semantic_prefix_enabled=False`
- width forced to the current spec defaults (`hidden_dim=384`, heads `8`)
- current-step prompt changed by swapping synthetic semantic token streams
- second step fed with `previous=...` from those two different prompt branches

Observed local results on the current code:

- `hidden_dim = 384`
- `task_available = True`
- `task_tokens_shape = (8, 384)`
- `task_global_shape = (384,)`
- `instruction_shape = (384,)`
- `loss_total = 1.5750`
- `loss_action = 1.1667`
- `loss_semantic_future_aux = 0.5082`
- `posterior_mu_diff_prompt = 0.0`
- `physical_global_pred_diff_prompt = 0.0`
- `task_tokens_diff_prompt = 37.5657`
- `task_global_diff_prompt = 13.2815`
- `instruction_diff_prompt = 10.2362`
- `action_diff_prompt = 0.0872`
- `next_posterior_mu_diff_prev_prompt = 0.0`
- `innovation_token_diff_prev_prompt = 0.0`
- `next_physical_global_pred_diff_prev_prompt = 0.0`

Observed cloud results after resyncing the final file set:

- `hidden_dim = 384`
- `task_available = True`
- `task_tokens_shape = (8, 384)`
- `task_global_shape = (384,)`
- `instruction_shape = (384,)`
- `loss_total = 2.5598`
- `loss_action = 1.9441`
- `loss_semantic_future_aux = 0.5185`
- `posterior_mu_diff_prompt = 0.0`
- `physical_global_pred_diff_prompt = 0.0`
- `task_tokens_diff_prompt = 31.8046`
- `task_global_diff_prompt = 11.2446`
- `instruction_diff_prompt = 9.3210`
- `action_diff_prompt = 0.0556`
- `next_posterior_mu_diff_prev_prompt = 0.0`
- `innovation_token_diff_prev_prompt = 0.0`
- `next_physical_global_pred_diff_prev_prompt = 0.0`

Interpretation:

- prompt changes do change the sidecar readout and the action readout
- prompt changes do not change:
  - physical posterior
  - physical predictive global state
  - next-step innovation
- this is exactly the intended mathematical boundary for the current design


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
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
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
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --task-anchor-sidecar-enabled \
  --no-legacy-semantic-prefix-enabled
```

Optional sidecar hyperparameters are also exposed:

- `--grad-clip-mode`
- `--grad-clip-percentile`
- `--grad-clip-window`
- `--task-anchor-queries`
- `--task-global-queries`
- `--task-query-layers`
- `--task-query-rounds`
- `--task-anchor-dropout-prob`


## 9. Recommended Deployment Order

1. Run the local validation commands in Section 7.
2. Decide whether to deploy the current code exactly as-is, or first revert the
   width defaults back to `256`.
3. If deploying current code as-is, note that you are deploying:
   - sidecar architecture
   - width `384`
   - heads `8`
4. Start with `sidecar=true` and `legacy=true`.
5. Verify:
   - no physical invariant regressions
   - action prompt sensitivity improves
   - environment sensitivity remains non-trivial
6. Then run `sidecar=true` and `legacy=false`.
7. Compare:
   - action sensitivity
   - physical invariance
   - training stability

Why A/B was recommended:

- this is not a multi-stage training algorithm
- each run is still a single end-to-end training graph
- the A/B split is a rollout strategy:
  - `sidecar=true, legacy=true` checks whether adding the sidecar is safe while
    retaining the previous semantic path
  - `sidecar=true, legacy=false` checks whether the sidecar can replace the
    legacy semantic prefix cleanly

So A/B here means:

- two separate end-to-end training runs for engineering comparison
- not a mathematically required two-phase optimization procedure


## 10. Current Conclusion

As of this document:

- the single-core + task-sidecar architecture is implemented locally
- physical core invariants are preserved
- task-anchor sidecar is live behind feature flags
- explicit `global_post` control injection is implemented
- trainer / serving / contract verification are green
- resume deployment path supports sidecar flags
- current code defaults are width `384`, not `256`

The remaining work is experimental rather than structural:

- run sidecar A/B training
- measure prompt sensitivity and behavior
- decide whether the current width-384 code should be kept, or whether strict
  Phase-1 scope requires reverting widths to 256 before longer training

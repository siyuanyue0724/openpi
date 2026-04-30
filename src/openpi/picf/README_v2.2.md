# PICF v2.2

Date: 2026-04-20
Repo: `/home/siyuanyue/Documents/openpi`
Status: current local v2.2 architecture record after the one-shot
action/control contract rewrite and the current exhaustive local audit

## Quick Navigation

- [`README.md`](/home/siyuanyue/Documents/openpi/README.md)
  Repo-level entry point.
- [`src/openpi/picf/README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
  PICF entry pointer that routes readers to the current and archived PICF docs.
- [`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
  Current live PICF v2.2 architecture and deployment record.
- [`src/openpi/picf/README_PI05_PARITY_AUDIT.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_PI05_PARITY_AUDIT.md)
  Local code-level audit comparing reference PI0.5 / PI0.5+Sonata dataflow
  against current PICF enabled and PI0.5-only ablated modes.
- [`src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md)
  Design record for the 2x40GB frozen-perception profile and geometry-safe
  augmentation policy. Use this before freezing V-JEPA/Sonata/AnyTouch or
  enabling any train-time augmentation in full PICF.
  The live CLI now exposes `--perception-finetune-mode auto|full|frozen`,
  `--visual-feature-mode auto|hierarchical|final`,
  `--picf-augmentation-mode off|photometric|multimodal_geometry`, and
  `--picf-photometric-strength conservative|reference`; the default remains
  `auto/off`.
- [`src/openpi/picf/README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
  Historical pre-v2.2 record.
- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
  Compact executable contract for the live code.
- [`docs/CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
  Training, serving, and CALVIN validation workflow.
  The current canonical full PICF long-run training command is recorded in
  [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch).
  The current cloud-tested 20-sequence video evaluation recipes, including the
  full PICF `step=7500` run and the maintained PI0.5-only ablation run, are
  recorded in
  [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation).

## 0. Current Audit Snapshot

This file should currently be read as a **local correctness / architecture
record first**, and only secondarily as a historical record of earlier cloud
bring-up attempts.

As of the latest local audit pass:

- the v2.2 physical / task-readout / conditioned-control / PI0 action contract
  is internally consistent
- the runtime surface now has two explicit modes:
  - `picf_mode=enabled`: full v2.2 PICF path
  - `picf_mode=ablated`: PI0.5-only ablation path with PICF branches disabled
- future-target supervision is now explicit stop-gradient teacher supervision
- shared middle frames inside a training window are now reused as detached
  future targets instead of being redundantly rebuilt
- the live training policy remains:
  - one canonical recurrent physical carry
  - one canonical conditioned control state `C_t`
  - one final PI0.5 action path
- no current local regression evidence shows semantic leakage into posterior or
  innovation
- no current local regression evidence shows dual control semantics reappearing

Latest fully local verification evidence:

- `pytest -q src/openpi/picf/core/training_test.py` -> `18 passed`
- `pytest -q src/openpi/picf/paligemma/wrapper_test.py` -> `23 passed`
- `pytest -q scripts/picf_core_train_test.py` -> `88 passed`
- `pytest -q src/openpi/picf/policy_test.py scripts/serve_picf_policy_test.py` ->
  policy/serve ablation coverage passes
- `pytest -q scripts/serve_picf_policy_test.py src/openpi/picf/policy_test.py scripts/picf_core_train_test.py src/openpi/picf/core/training_test.py src/openpi/picf/core/pipeline_test.py src/openpi/picf/paligemma/wrapper_test.py scripts/picf_resume_train_test.py`
  -> `203 passed`
- `python scripts/verify_picf_contract.py` -> static checks, documentation
  checks, targeted invariance regressions, core regression suite, and smoke
  training check all pass

Performance note:

- the current main open issue is still throughput, not mathematical contract
  correctness
- the latest measured short-run exact-memory speed evidence remains the
  historical 5-step probe recorded in
  [`/tmp/picf_v22_speed_audit_20260420.md`](/tmp/picf_v22_speed_audit_20260420.md)
- do not read this file as claiming that a full local or cloud `30000`-step run
  has already completed on the current code unless that evidence is recorded
  explicitly
## 1. Document Map

This file is the **current local v2.2 architecture record and implementation
audit**. The one-shot action/control refactor described below has been deployed
locally; the detailed planning sections are retained because they explain the
implemented contract rewrite and the reasoning behind it.

Relevant documents:

1. `/home/siyuanyue/Documents/openpi/README.md`
   Repo-level entry point. Use this if you are opening the repository cold and
   want the broad project context before diving into PICF-specific docs.

2. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md`
   This file. Current v2.2 architecture record. It records:
   - what the current live code implements
   - what changed in the v2.2 contract rewrite
   - the canonical object/state decomposition
   - file-by-file implementation scope and rationale
   - migration, testing, and rollout gates used to validate the patch

3. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md`
   Historical v2.1 deployment record from before the v2.2 refactor landed.

4. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_PI05_PARITY_AUDIT.md`
   Detailed local audit of reference PI0.5 / PI0.5+Sonata dataflow versus the
   current PICF enabled and PI0.5-only ablated paths. Use it when interpreting
   ablation quality, CALVIN loss comparability, normalization, preprocessing,
   and parity claims.

5. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`
   Design record for the planned `picf_frozen_perception_2x40` profile. It
   documents which perception modules may be frozen without changing the PICF
   architecture, why full PI0.5 RGB geometry augmentation cannot be blindly
   copied into full PICF, and which augmentation modes are safe to make
   configurable.

6. `/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md`
   Current concise executable contract enforced by regression tests. It is the
   compact version of the live v2.2 semantics described in this file.

7. `/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md`
   Current training / validation / rollout / serving workflow document.

8. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md`
   Directory entry / pointer file. It points at both the current live
   record (`README_v2.2.md`) and the archived v2.1 record so neither is lost.

Maintained/current docs:

- `src/openpi/picf/README_v2.2.md`
- `src/openpi/picf/README_PI05_PARITY_AUDIT.md`
- `src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`
- `PICF_FORMAL_CONTRACT.md`
- `docs/CALVIN_VALIDATION_README.md`
- `src/openpi/picf/README.md` as entry pointer

Historical/archive docs:

- `src/openpi/picf/README_v2.1.md`

### 1.1 Read Order

If you are opening the repo cold, use this order:

1. [`README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
   PICF entry pointer only.
2. [`/home/siyuanyue/Documents/openpi/README.md`](/home/siyuanyue/Documents/openpi/README.md)
   Repo-level entry point.
3. [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
   Current live architecture record and implementation audit.
4. [`README_FROZEN_PERCEPTION_AUGMENTATION.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md)
   Read this before using frozen V-JEPA/Sonata/AnyTouch profiles or enabling
   train-time augmentation in full PICF.
5. [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
   Concise executable contract for the current code.
6. [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
   Runtime / training / rollout workflow.
   For the current canonical full PICF long-run launch, jump directly to
   [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch).
   For the current validated cloud rollout/video test recipes, including
   full PICF `step=7500` serving/evaluation, single-rollout GPU usage, and
   `/tmp` to `/mnt` artifact mirroring, jump directly to
   [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation).
   For the current 2x40GB ablated training definition and the explicit
   `2500 current optimizer steps ~= 5000 historical no-Sonata PI0.5 steps`
   comparison rule, use the ablated long-run profile in
   [`Section 3`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#3-trainer-smoke-validation).
6. [`README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
   Historical pre-v2.2 record only.

### 1.2 Temporary Audit Companions

For this local v2.2 rollout, the primary temporary audit documents are:

1. [`/tmp/picf_v22_temp_current_dataflow.md`](/tmp/picf_v22_temp_current_dataflow.md)
   Recursive current-code dataflow audit for the live trainer/policy/core/loss path, including the one-step lookahead loss wiring.
2. [`/tmp/picf_v22_temp_theory_reconciliation.md`](/tmp/picf_v22_temp_theory_reconciliation.md)
   Theory-side reconciliation of the current implementation, including detached future-target reuse on internal window frames.
3. [`/tmp/picf_v22_bug_optimization_register_20260421.md`](/tmp/picf_v22_bug_optimization_register_20260421.md)
   Explicit bug / optimization register for the current audit pass.

Supporting temporary audit documents for the same rollout:

4. [`/tmp/picf_v22_current_code_dataflow_20260420.md`](/tmp/picf_v22_current_code_dataflow_20260420.md)
   Prior recursive current-code dataflow audit for the initial 2026-04-20 pass.
5. [`/tmp/picf_v22_audit_report_20260420.md`](/tmp/picf_v22_audit_report_20260420.md)
   Prior audit conclusion file: cloud/runtime evidence, real findings, non-findings, and next exact-optimization targets.
6. [`/tmp/picf_v22_mathematical_spec_20260420.md`](/tmp/picf_v22_mathematical_spec_20260420.md)
   Refreshed v2.2 mathematical specification with the canonical recurrent-carry contract.
7. [`/tmp/picf_v22_design_reconciliation_20260420.md`](/tmp/picf_v22_design_reconciliation_20260420.md)
   Explicit list of design mismatches / unnecessary glue, plus the fixes landed during the audit.
8. [`/tmp/picf_v22_memory_audit_20260420.md`](/tmp/picf_v22_memory_audit_20260420.md)
   Quantitative 4x40GB A100 memory audit and backbone-contribution ranking.
9. [`/tmp/picf_v22_speed_audit_20260420.md`](/tmp/picf_v22_speed_audit_20260420.md)
   Performance-specific historical speed audit for the earlier exact-memory bring-up passes.
10. [`/tmp/picf_v22_readme_sync_20260420.md`](/tmp/picf_v22_readme_sync_20260420.md)
   README synchronization audit for the current live deployment profile, observability modes, and GitHub handoff scope.

Historical temp audits from the earlier 2026-04-18 pass are retained only as
archive context:

11. [`/tmp/picf_v22_current_code_dataflow_20260418.md`](/tmp/picf_v22_current_code_dataflow_20260418.md)
12. [`/tmp/picf_v22_mathematical_spec_20260418.md`](/tmp/picf_v22_mathematical_spec_20260418.md)
13. [`/tmp/picf_v22_final_reconciliation_20260418.md`](/tmp/picf_v22_final_reconciliation_20260418.md)

These `/tmp` documents are audit artifacts, not persistent maintained docs.

### 1.3 Section Guide

Use the sections in this file as follows:

- Section 2: scope
- Section 3: recursive audit of the current live implementation
- Section 4: canonical v2.2 object/state shape
- Section 5: non-negotiable invariants
- Section 6: corrections to the earlier external proposal
- Section 7: detailed end-to-end v2.2 dataflow
- Section 8: file-by-file implementation map
- Section 9: checkpoint and compatibility migration
- Section 10: validation matrix
- Section 11: rollout gate record
- Section 12: forbidden regressions
- Section 13: definition of done
- Section 14: final recommendation

### 1.4 Navigation Summary

When verifying the local v2.2 codebase, use this navigation split:

- architecture and rationale:
  - [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
- concise executable rules:
  - [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
- runtime / training / rollout workflow:
  - [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
  - current canonical full PICF long-run launch:
    [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch)
  - current cloud 20-sequence CALVIN video evaluation:
    [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation)
    This section records that one CALVIN rollout uses one policy-server GPU plus
    one EGL/evaluator GPU; it does not automatically consume all GPUs the way
    FSDP training does.
- historical pre-v2.2 reference only:
  - [`README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
- temporary deep audits for this local rollout:
  - [`/tmp/picf_v22_temp_current_dataflow.md`](/tmp/picf_v22_temp_current_dataflow.md)
  - [`/tmp/picf_v22_temp_theory_reconciliation.md`](/tmp/picf_v22_temp_theory_reconciliation.md)
  - [`/tmp/picf_v22_bug_optimization_register_20260421.md`](/tmp/picf_v22_bug_optimization_register_20260421.md)
  - supporting:
    [`/tmp/picf_v22_current_code_dataflow_20260420.md`](/tmp/picf_v22_current_code_dataflow_20260420.md),
    [`/tmp/picf_v22_audit_report_20260420.md`](/tmp/picf_v22_audit_report_20260420.md),
    [`/tmp/picf_v22_mathematical_spec_20260420.md`](/tmp/picf_v22_mathematical_spec_20260420.md),
    [`/tmp/picf_v22_design_reconciliation_20260420.md`](/tmp/picf_v22_design_reconciliation_20260420.md),
    [`/tmp/picf_v22_memory_audit_20260420.md`](/tmp/picf_v22_memory_audit_20260420.md),
    [`/tmp/picf_v22_speed_audit_20260420.md`](/tmp/picf_v22_speed_audit_20260420.md),
    [`/tmp/picf_v22_readme_sync_20260420.md`](/tmp/picf_v22_readme_sync_20260420.md)
  - archived 2026-04-18 temp audits only when historical comparison matters

## 2. Scope

The goal of v2.2 is **not** to redesign the physical core. The goal is to
preserve the already-correct physical world-state machinery and perform a
single, coherent refactor of the action/control integration layer.

The intended one-shot result is:

- one exported policy object
- one canonical conditioned control state `C_t`
- one final action path, always through PI0.5 flow matching / sampling
- semantic used for task-relevant current-step readout, not for physical world
  state update
- world-only predictive basis kept clean for next-step innovation

### 2.1 Runtime Modes

The live trainer / serve stack now has two explicit runtime modes.

`picf_mode=enabled`

- canonical v2.2 path
- builds and trains the PICF recurrent/task-readout/conditioned-control/future
  branches
- PI0.5 action generation consumes `conditioned_control.pi_prefix_tokens`

`picf_mode=ablated`

- PI0.5-only ablation path for parity checks against the main-branch PI0.5
  baseline
- PICF recurrent/control/future losses are disabled
- trainer uses only PI0.5 flow loss and serve uses only the PI0.5 sampler
- PICF core parameters are frozen and excluded from the optimizer
- PICF-only point/visual/tactile backbone branches are normalized back to the
  stub/off path so the ablation does not pay for unused foundation modules
- PI0.5 is called with `extra_prefix_tokens=None`
- checkpoint save/load in this mode is semantic-only by construction:
  `model.pt` stores only the PI0.5 semantic subtree, while the frozen lazy
  PICF core is deliberately omitted instead of being force-materialized for
  serialization
- `optimizer_checkpoint_mode=auto` now resolves to model-only checkpoints in
  this mode; use `--optimizer-checkpoint-mode full` only when an ablated run
  truly needs optimizer-state resume
- if serving overrides an enabled checkpoint with `--picf-mode ablated`, the
  runtime args are re-normalized and re-validated before model/source
  construction so enabled-mode tactile/visual branch assumptions do not leak
  into the ablation path

The architectural contract described in the rest of this document remains the
canonical contract for `picf_mode=enabled`. The ablated mode is a deliberate
control experiment, not a second canonical PICF semantics.

### 2.2 Shared Trainer-Shell Contract

The runtime mode switch does **not** create a second trainer shell.

Inside `scripts/picf_core_train.py`, the following operational knobs are shared
by both `picf_mode=enabled` and `picf_mode=ablated`:

- `--save-interval`
- `--log-interval`
- `--accum-steps`
- `--unroll-steps`
- `--action-horizon`
- progress-bar cadence
- `metrics.jsonl` write cadence
- `_CalvinTransitionSource` window sampling contract

What changes with runtime mode is narrower:

- the per-window semantic/PICF execution path inside `PicfPi05Policy` and
  `_PicfWindowTrainer`
- the default checkpoint payload policy under
  `--optimizer-checkpoint-mode auto`

Operational rule:

- if you change `--save-interval 2500` or `--log-interval 100` for an ablation
  run, the same flags work the same way for `picf_mode=enabled`
- if you change `--action-horizon` or `--unroll-steps`, the same parser and
  window source own those flags for both modes
- do **not** infer from this that `picf_mode=ablated` is operationally
  identical to the preserved `pi0.5_sonata` trainer; it only means both PICF
  runtime modes share one maintained training shell

v2.2 is therefore treated as:

- contract rewrite
- interface unification
- task-readout refactor

and **not** as a cosmetic cleanup.

## 3. Current Live Code Audit

This section records what was already correct in current code and what the
v2.2 rewrite changed.

### 3.1 Physical Core: Keep

The following pieces were already the right mathematical object and were kept
structurally unchanged:

- `PicfFullCore._build_observation_anchors(...)`
- `PicfFullCore._posterior_update(...)`
- `PicfFullCore._innovation(...)`
- the split between `physical_prediction_cache` and `prediction_cache`

Observed current properties from code:

- observation-anchor construction is task-agnostic
- posterior update is language-free
- innovation reads only `previous.predictive.physical_prediction_cache`
- the only recurrent physical state family remains the posterior anchor state

Relevant code:

- `src/openpi/picf/core/pipeline.py`
- `src/openpi/picf/core/contracts.py`
- `PICF_FORMAL_CONTRACT.md`

### 3.2 Current-Step Private Dense Memory: Already Present

Current code already has:

```python
@dataclasses.dataclass(frozen=True)
class _StepDenseMemory:
    point_payload: torch.Tensor
    visual_payload: torch.Tensor
    tactile_group_tokens: tuple[torch.Tensor, ...]
```

This matters because v2.2 needs semantic-conditioned readout over:

- current public multimodal memory
- current private dense payloads

That substrate already exists. v2.2 reuses it rather than replacing it.

### 3.3 Attention Primitives: Already Sufficient

Current core already includes:

- `CrossAttentionRead`
- `GatedCrossAttentionRead`
- `LazyCrossAttentionRead`

This is enough to implement:

- semantic-conditioned task queries
- public token read
- private visual reread
- private tactile reread
- private point reread

without introducing a fourth attention subsystem.

### 3.4 Historical Seam 1: Action Path Used To Be Glue

The pre-v2.2 trainer/serve path used to be split across:

- `core.step(...)`
- PI0.5 wrapper calls in trainer / serve
- `refresh_predictive_state_for_action(...)`

That seam is now resolved. Current live action integration is:

- trainer:
  - `PicfPi05Policy.forward_train_transition(...)`
- serving:
  - `PicfPi05Policy.act(...)`

The unified policy now owns:

- semantic encoding
- `observe_step(...)`
- PI0.5 flow loss / sampler call
- executed-action finalization through `finalize_with_action(...)`

### 3.5 Historical Seam 2: Dual Control Semantics Used To Exist

Pre-v2.2 `_predictive_state(...)` used to construct two parallel control routes:

1. `action_condition_prefix -> action_condition_tokens`
2. `control_prefix -> control_tokens -> control_query_state -> pooled_state`

That seam is now resolved. Current live code builds one canonical conditioned
control state `C_t` and derives:

- `conditioned_control.tokens`
- `conditioned_control.query_state`
- `conditioned_control.pi_prefix_tokens`
- `conditioned_control.future_condition_tokens`

Compatibility fields such as `predictive.action_condition_tokens` and
`predictive.control_query_state` survive only as views/debug aliases of the
canonical conditioned-control outputs.

### 3.6 Historical Seam 3: Raw Semantic Prefix Used To Enter Core Control/Future

Pre-v2.2 `_predictive_state(...)` directly injected the raw semantic prefix
into:

- `control_world`
- `predictive_semantic_world`

That route has now been removed from the core contract. Current live behavior
is:

- raw semantic prefix stays native in PI0.5 semantic/action generation
- semantic enters the core only through current-step task readout
- task readout contributes to the unique conditioned state `C_t`
- conditioned future is built from token-level physical predictive tokens plus
  `C_t^{future}`, not from raw semantic-prefix injection

### 3.7 Important Current Code Fact: `fused_tokens` Is Not Full Multimodal Public Memory

Current `_build_token_field(...)` constructs:

```python
all_tokens = torch.cat([point_tokens, tactile_tokens_active, context_tokens], dim=0)
```

and then runs `token_fusion` over `all_tokens` to produce `fused_tokens`.

Current `visual_tokens` exist in `PicfTokenFieldState`, but they are **not**
part of `fused_tokens`.

Therefore:

- `token_field.fused_tokens` is not a full public multimodal memory
- it is a fused point / tactile-active / context memory
- visual public routing is currently handled separately by native-first visual
  reread before public fused read

This matters because the external proposal said "read full fused tokens". That
was corrected in v2.2: task-readout public memory is not defined as
`fused_tokens` alone.

### 3.8 Current State Dataclasses: Exact Audit

Current `src/openpi/picf/core/contracts.py` exposes these load-bearing state
containers.

`PicfTokenFieldState`

- `point_tokens`
- `visual_tokens`
- `tactile_tokens`
- `context_tokens`
- `fused_tokens`
- `point_positions`
- `modality_ids`
- alignment embeddings for point / visual / tactile
- tactile per-step routing metadata:
  - `tactile_tokens_all`
  - `tactile_tokens_active`
  - `tactile_group_ids`
  - `tactile_contact_prob`
  - `tactile_anchor_mask`
  - `tactile_normals_world`
  - `tactile_contact_score`
  - `tactile_contact_score_ema`
- `fusion_attention_mean`
- `projective_geometry`

`PicfObservationAnchorState`

- `seed_indices`
- `tokens`
- `point_weights`
- `routing_mass_point`
- `routing_mass_visual`
- `routing_support_point`
- `routing_support_visual`
- `routing_gate_point`
- `routing_gate_visual`
- geometry summaries `x, S, a`
- tactile routing extensions already exist:
  - `routing_mass_tactile`
  - `routing_support_tactile`
  - `routing_gate_tactile`

`PicfPosteriorAnchorState`

- recurrent core: `h, c`
- belief state: `mu, Sigma`
- geometry state: `x, S, a`
- activity / contact / support:
  - `alpha`
  - `contact_prob`
  - `support_mass`
  - `recycle_gate`
- binding / evidence:
  - `binding`
  - `evidence_tokens`
- exported posterior tokens:
  - `tokens`
  - `global_post`

`PicfPredictiveState`

- semantic side:
  - `semantic_tokens`
  - `innovation_token`
  - `innovation_norm`
  - `availability`
- current dual control path residues:
  - `control_tokens`
  - `action_condition_tokens`
  - `control_query_state`
  - `pooled_state`
- action fields:
  - `action`
  - `action_chunk`
  - `executed_action`
- physical predictive side:
  - `physical_global_pred`
  - `physical_prediction_cache`
- conditioned predictive side:
  - `predictive_query_state`
  - `global_pred`
  - `prediction_cache`

`PicfCoreState`

- `runtime_meta`
- `G_t`
- `token_field`
- `observation_anchors`
- `posterior`
- `predictive`
- `control`
- `last_prompt`

This audit matters because v2.2 is not just adding new objects. It also removes
the independent semantic meaning of the current control-related fields in
`PicfPredictiveState`.

### 3.9 Current `PicfFullCore.__init__`: Exact Module Inventory

Current `PicfFullCore` already contains most of the building blocks required by
the v2.2 patch. The important ones are:

Physical token construction:

- `point_token_proj`
- `visual_token_proj`
- `tactile_token_proj`
- `point_align_proj`
- `visual_align_proj`
- `tactile_align_proj`
- `token_fusion`

Observation-anchor and posterior update:

- `obs_reader`
- `obs_self`
- `anchor_seed_proj`
- `anchor_reader`
- `vote_heads`
- `post_write_proj`
- `post_lstm`
- `posterior_token_proj`
- `posterior_self`
- `posterior_pool`

Native reread paths already live in current code:

- `visual_native_reread`
- `tactile_native_reread`
- `point_native_reread`
- `tactile_group_route_queries`
- `tactile_route_reread`

Innovation and target heads:

- latent query banks:
  - `visual_latent_queries`
  - `tactile_latent_queries`
  - `point_latent_queries`
- prediction heads:
  - `visual_latent_head`
  - `visual_real_head`
  - `tactile_real_head`
  - `point_real_head`
- error encoders:
  - `visual_error_encoder`
  - `visual_real_error_encoder`
  - `tactile_error_encoder`
  - `point_error_encoder`
- `innovation_proj`
- `innovation_token_proj`

Current control / future modules:

- `posterior_to_control_proj`
- `global_post_to_control_proj`
- `innovation_to_control_proj`
- `proprio_to_control_proj`
- `control_role_embedding`
- `control_query_tokens`
- `control_world`
- `control_state_proj`
- `physical_pred_to_conditioned_proj`
- `predictive_conditioned_role_embedding`
- `predictive_query_tokens`
- `predictive_world`
- `predictive_semantic_world`
- `predictive_state_proj`

This is why v2.2 is a refactor, not a rewrite. Most required primitives
already existed and were reused.

### 3.10 Current `_build_token_field(...)`: Recursive Flow

Current `_build_token_field(...)` already does more than a simple concatenation.
Its real stages are:

1. initialize empty tensors for all modalities and all tactile routing fields
2. build projective geometry from:
   - point positions
   - current visual grid shape
3. construct point tokens:
   - point backbone features
   - RGB colors
   - point positional encoding
   - projection features
4. construct visual tokens:
   - flattened current V-JEPA map
   - grid features
   - camera pose features
   - ray features
5. construct tactile per-sensor base tokens:
   - pooled tactile feature
   - bundle global feature
   - world position
   - world rotation
6. run tactile contact/hysteresis logic
7. choose active tactile groups
8. expand each active tactile group into multiple public routing proposals using:
   - `tactile_group_route_queries`
   - `tactile_route_reread`
9. build context tokens:
   - proprio context
   - previous action context
   - timing context
   - contact context
10. run `token_fusion` **only** over:
    - point tokens
    - tactile active proposal tokens
    - context tokens
11. export `PicfTokenFieldState`
12. export `_StepDenseMemory`

Consequences for v2.2:

- current tactile group proposal routing already exists and is reused
- current visual public tokens exist separately from fused tokens
- current `fused_tokens` must not be mistaken for full multimodal public
  memory

### 3.11 Current `_build_observation_anchors(...)`: Recursive Flow

Current observation-anchor construction is already two-stage:

1. initialize learned / point-seeded queries
2. do native visual reread first:
   - `queries, visual_weights = self.visual_native_reread(...)`
3. then do public fused read:
   - `queries, attn_public = self.obs_reader(...)`
4. run `obs_self`
5. derive routing masses:
   - point routing from `attn_public`
   - visual routing from native reread
   - tactile routing by aggregating proposal-token masses back to tactile group
     ids
6. derive anchor geometry summaries

This is a strong base for v2.2. It confirms that:

- physical observation anchors already do native-first visual competition
- tactile ownership already exists at group level in the physical path
- semantic does not belong here

### 3.12 Current `_posterior_update(...)`: Recursive Flow

Current posterior update is already a multi-source evidence fusion block:

1. construct current prior from previous state
2. compute observation-anchor binding logits
3. apply dustbin sinkhorn
4. compute recycle / residual summary path
5. build anchor reader query from prior hidden + latent + geometry
6. read observation-anchor evidence via `anchor_reader`
7. native visual reread:
   - `binding_cond @ routing_mass_visual`
   - gather top-k visual payload candidates
   - `visual_native_reread`
8. native tactile reread:
   - `binding_cond @ routing_mass_tactile`
   - gather winning tactile groups
   - `tactile_native_reread`
9. fuse measurement evidence
10. update geometry summaries
11. run vote heads
12. do precision fusion
13. update posterior recurrent state
14. emit posterior tokens and global summary

This means v2.2 does not re-invent dense reread. It already exists in
the physical posterior path and remains untouched.

### 3.13 Current `_current_targets(...)` and `_innovation(...)`

Current targets are already denser than the original coarse version:

- visual latent target from native V-JEPA payload probes
- visual real target from RGB downsample
- tactile latent target from dense tactile group tokens
- tactile real target from:
  - tactile latent
  - tactile map
  - tactile auxiliaries
- point latent target from point payload probes
- point real target from:
  - point latent
  - occupancy

Current `_innovation(...)` then:

1. reads only `previous.predictive.physical_prediction_cache`
2. compares branchwise against current targets
3. standardizes residuals
4. encodes per-branch residual features
5. concatenates branch features + availability
6. produces one innovation token and branch norms

This confirms:

- innovation is already correctly world-only
- v2.2 preserves the current physical innovation semantics exactly

### 3.14 Current `_predictive_state(...)`: Post-v2.2 Role

Current `_predictive_state(...)` is no longer the place where control semantics
are invented. It now has a narrower role:

1. accepts already-built `conditioned_control`
2. resolves executed action / action chunk
3. builds the physical predictive basis
4. builds the conditioned future cache from:
   - physical predictive token sequence
   - future-condition tokens
5. emits predictive state fields plus compatibility/debug aliases

The control-semantics split was moved out of `_predictive_state(...)` into:

- `observe_step(...)`
- `_build_task_readout(...)`
- `_build_conditioned_control_state(...)`
- `finalize_with_action(...)`

### 3.15 Current `refresh_predictive_state_for_action(...)` and `step(...)`

`refresh_predictive_state_for_action(...)` now acts as a compatibility bridge
around the new observe/finalize split. It reconstructs the minimal observed
state needed to re-run predictive finalization with the externally supplied
action chunk.

Current `step(...)` is also a compatibility wrapper only. The canonical live
path is:

- `observe_step(...)`
- PI0.5 action generation / teacher-forced action resolution
- `finalize_with_action(...)`

Trainer and serve no longer depend on `step(...)` as the primary action API.

### 3.16 Current Wrapper / PI0.5 Audit

Current `src/openpi/picf/paligemma/wrapper.py` already restored the PI0.5 stack:

- `PaliGemmaWithExpertModel`
- `gemma_expert`
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`
- state injection back into prompt tokenization
- flow loss:
  - `x_t`
  - `u_t`
  - `v_t`
  - denoised target recovery as `x_t - t * v_t`
- denoise sampler using cached PI0.5 prefix state

Current wrapper contract already supports:

- `encode_observation(...)`
- `supports_pi0_action_generation()`
- `compute_action_flow_loss(...)`
- `sample_action_chunk(...)`

Therefore v2.2 does not need to redesign PI0.5 integration. It only needs to
replace the source of `extra_prefix_tokens`.

### 3.17 Current Trainer Audit

Current `scripts/picf_core_train.py` still sequences training manually:

1. semantic encode current observation
2. `core.step(...)`
3. if PI0.5 available:
   - `compute_action_flow_loss(...)`
   - pass `output.state.predictive.action_condition_tokens` as extra prefix
   - override action fields in state
4. call `compute_transition_loss(...)`

Current trainer also already contains important runtime logic that v2.2
preserves:

- compat checkpoint loader
- shape mismatch filtering
- allowlists for missing/unexpected keys
- DDP guards
- invalid first-step rejection sampling
- gradient clipping / logging

So the trainer is not simplified blindly. Only the action/control glue moved
behind `PicfPi05Policy`.

### 3.18 Current Serve Audit

Current `scripts/serve_picf_policy.py` sequences inference manually:

1. encode semantic observation
2. `core.step(..., action_future=None)`
3. sample PI0.5 action chunk using
   `output.state.predictive.action_condition_tokens`
4. call `core.refresh_predictive_state_for_action(...)`
5. export first action

This confirms exactly what the v2.2 exported policy now unifies.

### 3.19 Current Verifier and Tests Audit

Current `scripts/verify_picf_contract.py` is still aligned to the current live
semantic-prefix-primary core semantics. It currently asserts things such as:

- control prefix uses full semantic prefix directly
- conditioned future uses full semantic prefix directly
- task-sidecar path is absent

So verifier changes are not optional in v2.2; they are part of the contract
rewrite.

Current tests already cover:

- physical posterior / innovation invariants
- native visual reread
- tactile group routing / winner read
- serve-time predictive refresh
- wrapper PI0.5 action generation
- trainer loss override behavior

This is useful because v2.2 can extend an existing test base rather than
starting from zero.

### 3.20 Current Default Configuration Snapshot

The current live defaults that v2.2 treats as baseline, not as part of
the refactor, are:

`PicfCoreConfig`

- `persistent_anchors = 16`
- `observation_anchors = 24`
- `hidden_dim = 512`
- `posterior_hidden_dim = 512`
- `latent_dim = 112`
- `innovation_dim = 512`
- `control_dim = 512`
- `semantic_dim = 2048`
- `semantic_cross_dim = 2048`
- `future_hidden_dim = 512`
- `attention_heads = 8`
- `tactile_group_proposals = 2`
- `visual_reread_topk = 32`
- `tactile_reread_groups = 2`

`PaliGemmaSemanticConfig`

- `pi05 = True`
- `action_dim = 32`
- `action_horizon = 16`
- `denoise_steps = 10`
- `inject_state_into_prompt = True`
- `prompt_state_normalization = quantile` for CALVIN-aligned training
- `prompt_state_norm_stats_path = same CALVIN norm_stats.json used by action normalization`

Important boundary:

- prompt-state normalization is applied only on the semantic prompt-tokenization
  path before state discretization
- prompt-state tokenization uses the live CALVIN `robot_obs` / `proprio`
  dimensionality, matching the reference transform order where
  `TokenizePrompt(...)` runs before `PadStatesAndActions(...)`
- the zero padding to `action_dim = 32` is only for the model state/action tensor
  contract, not for the text prompt's discretized state string
- raw `robot_obs` / `proprio` stay untouched for the PICF physical core
- this preserves the PICF physical boundary while matching the reference
  `pi0.5_sonata` preprocessing contract, where `Normalize(norm_stats)` happens
  before `TokenizePrompt(...)`

Current training/runtime assumptions relevant to v2.2:

- trainer already expects compat loading and non-strict migration paths
- trainer already has DDP-specific gradient-checkpointing guards
- serving already assumes normalized-core / unnormalized-environment action
  contract

These are operating assumptions to preserve during the refactor, not knobs to
revisit inside the same patch.

## 4. Canonical v2.2 Shape

The current v2.2 system has exactly these canonical objects:

- physical recurrent state: `W_t`
- conditioned current-step control state: `C_t`
- world-only predictive basis: `K_t^{phys}`
- conditioned predictive cache: `K_t^{cond}`
- final action path: `a_t ~ PI0.5(S_t, C_t^{pi})`

High-level flow:

```text
observation_t
-> token_field_t + dense_memory_t
-> observation_anchors_t
-> posterior_t = W_t
-> targets_t
-> innovation_t from previous K_{t-1}^{phys}
-> semantic tokens S_t
-> task readout R_t over [public_read_memory_t, dense_memory_t]
-> conditioned control state C_t
-> PI0.5 flow loss / sampler using C_t^{pi}
-> executed_action a_t
-> K_t^{phys} = physical predictive basis from [W_t, a_t, proprio]
-> K_t^{cond} = conditioned future from [K_t^{phys}, C_t]
```

## 5. Non-Negotiable v2.2 Invariants

These are hard constraints, not suggestions.

### 5.1 Physical Core Remains Language-Free

Semantic must not enter:

- `_build_observation_anchors(...)`
- `_posterior_update(...)`
- `_innovation(...)`

### 5.2 Innovation Base Remains World-Only

Next-step innovation may read only:

- `previous.predictive.physical_prediction_cache`

It must not read:

- `previous.predictive.prediction_cache`
- `previous.conditioned_control`
- `previous.task_readout`
- semantic-conditioned state of any kind

### 5.3 `C_t` Is the Only Canonical Conditioned Control State

After the refactor, these must no longer exist as independent control
semantics:

- `action_condition_tokens`
- `control_query_state`
- `pooled_state`

If any compatibility alias survives, it may only be a view/debug snapshot of
the canonical `conditioned_control`.

### 5.4 `C_t^{pi}` Is an Interface View, Not a Second Control State

`C_t^{pi}` is only the prefix representation exported to PI0.5.

It is derived from `C_t` and is not itself a second control-path object.

### 5.5 `task_readout` Is Current-Step Only

`task_readout` must not become:

- recurrent memory
- task posterior
- next-step prior input
- next-step innovation base

It exists only to build `C_t` and conditioned future context.

### 5.6 `K_t^{phys}` Must Be Computed After Executed Action Is Known

Training:

- use teacher-forced first action

Serving:

- use sampled first action from PI0.5 chunk

Then compute:

```text
K_t^{phys} = P_phys(W_t, a_t^{exec}, proprio_t)
```

Never predict `K_t^{phys}` from a placeholder action and then overwrite the
executed action later.

### 5.7 Serving Must Fail Fast Without PI0.5 Action Generation

Formal deployed serving must not silently fall back to placeholder action.

The only valid deployed action path is:

- `PicfPi05Policy.act(...)`

## 6. Implemented Corrections to the External Proposal

The external proposal was directionally correct. These corrections were applied
in the same patch.

### 6.1 Public Read Memory Must Preserve Existing Public Fusion

Because current `fused_tokens` excludes `visual_tokens`, v2.2 defines a new
explicit public read memory. That public memory does not bypass the existing
`token_fusion` result.

The corrected v2.2 definition is:

```text
public_read_memory =
[
  fused_tokens,
  visual_tokens,
]
```

Single-line contract form:

```text
public_read_memory = [fused_tokens, visual_tokens]
```

where:

- `fused_tokens` preserves the current fused point/tactile/context public field
- `visual_tokens` preserves the separate public visual branch

`task readout` reads this corrected public memory and then rereads private
dense memory. It does not regress to raw pre-fusion
`point_tokens/tactile_tokens_active/context_tokens` as its only public memory.

### 6.2 Contract Rewrite Must Be Explicit

Current formal contract is semantic-prefix-primary in core control/future.

v2.2 replaces that with:

- PI0.5 keeps raw semantic stream
- core gets semantic influence only through `task_readout`
- `C_t` becomes the only conditioned-control object

This is a contract rewrite, not a minor cleanup.

### 6.3 `C_t` Must Keep Token-Level Richness

`C_t` must not collapse too early to one pooled vector.

It preserves at minimum:

- `base_tokens`
- `task_tokens`
- `tokens` (post-control-trunk sequence)

and only then derive:

- `pi_prefix_tokens`
- `future_condition_tokens`

### 6.4 Instruction Query Count Should Not Stay at 1

If raw semantic prefix is removed as a direct core control/future input, one
instruction token is too aggressive a bottleneck.

The current live default is:

- `task_instruction_queries = 2`

This is the conservative default. It keeps instruction semantics richer without
exploding complexity.

### 6.5 Compat Loader Migration Must Ship in the Same Patch

Current trainer uses explicit compat filters for:

- allowed missing keys
- allowed unexpected keys
- shape-mismatch filtering before relaxed load

Any v2.2 patch that changes:

- control role embeddings
- control/predictive query tokens
- new task-readout modules
- removal of old semantic-prefix-primary modules

updates compat loading in the same patch. Otherwise warm-start from current
checkpoints will fail abruptly.

### 6.6 DDP Safety Guards Stay in Trainer/Runtime Layer

`PicfPi05Policy` unifies action/state interfaces. It does **not** absorb
runtime/DDP-specific guards that are already working in the training script.

Those include:

- gradient-checkpointing restrictions under DDP
- pre-DDP rejection sampling for invalid first-step windows
- compact startup logging
- current compat checkpoint loader behavior
- post-sampler predictive refresh timing

## 7. Detailed v2.2 Dataflow

### 7.1 Observation and Physical World Update

Unchanged:

```text
observation_t
-> point / visual / tactile feature extraction
-> token_field_t + dense_memory_t
-> observation_anchors_t
-> posterior_t = W_t
-> targets_t
-> innovation_t from previous.K_{t-1}^{phys}
```

### 7.2 Semantic-Conditioned Task Readout

New:

```text
semantic tokens S_t
-> condition task queries
-> read public_read_memory_t
-> private reread over:
   - dense_memory.visual_payload
   - dense_memory.tactile_group_tokens
   - dense_memory.point_payload
-> task_local_tokens
-> task_global_token
-> instruction_tokens
-> geometry summaries (x, S, a)
```

This stage is current-step only.

### 7.3 Unique Conditioned Control State

New:

```text
base_tokens =
[
  posterior.tokens,
  posterior.global_post,
  innovation_token,
  proprio_token,
]

task_tokens =
[
  task_local_tokens,
  task_global_token,
  instruction_tokens,
]

C_t = control_world([base_tokens, task_tokens, conditioned_control_queries])
```

Then derive:

```text
C_t^{pi} = pi_prefix_reader(C_t.tokens)
C_t^{future} = future_condition_reader(C_t.tokens)
```

### 7.4 Unique Final Action Path

Training:

```text
flow_loss = semantic_encoder.compute_action_flow_loss(
    semantic_features,
    extra_prefix_tokens=C_t^{pi},
    action_chunk_target=teacher_chunk,
)
```

Serving:

```text
sampled_chunk = semantic_encoder.sample_action_chunk(
    semantic_features,
    extra_prefix_tokens=C_t^{pi},
)
```

No second action path may remain.

### 7.5 Physical Predictive Basis and Conditioned Future

```text
K_t^{phys} = P_phys(W_t, a_t^{exec}, proprio_t)
K_t^{cond} = P_cond(K_t^{phys}, C_t^{future})
```

This preserves:

- world-only predictive basis for innovation
- conditioned future for semantic/task-aware forecasting

## 8. File-by-File v2.2 Implementation Record

### 8.1 `src/openpi/picf/core/config.py`

Keep unchanged:

- `hidden_dim = 512`
- `posterior_hidden_dim = 512`
- `innovation_dim = 512`
- `control_dim = 512`
- `future_hidden_dim = 512`
- `semantic_dim = 2048`
- `attention_heads = 8`

Rationale: width churn is not part of this refactor.

Current file defines:

- `task_local_queries: int = 8`
- `task_global_queries: int = 1`
- `task_instruction_queries: int = 2`
- `task_self_layers: int = 1`
- `conditioned_control_queries: int = 4`
- `pi_prefix_queries: int = 4`
- `conditioned_future_queries: int = 2`
- `task_visual_reread_topk: int = 32`
- `task_tactile_reread_groups: int = 2`
- `task_point_reread_topk: int = 32`
- `require_pi0_action_generator: bool = True`

Reserved / compatibility-only field:

- `task_query_rounds: int = 2`
  - retained in `PicfCoreConfig` for a not-yet-implemented iterative task-readout variant
  - not currently consumed by the live v2.2 core/trainer path

Mark as deprecated compatibility-only:

- `predictive_semantic_reads`
- `control_semantic_reads`
- `predictive_semantic_dropout_prob`
- `semantic_prefix_dropout_prob`

### 8.2 `src/openpi/picf/core/contracts.py`

Add:

`PicfTaskReadoutState`

- conditioned semantic queries
- local tokens
- global token
- instruction tokens
- point weights
- geometry summaries `x, S, a`
- public/private attention diagnostics

`PicfConditionedControlState`

- base tokens
- task tokens
- unified control tokens
- query state
- pi prefix tokens
- future condition tokens

Current `PicfPredictiveState` semantics were reworked as follows:

- remove independent control-semantics status from:
  - `control_tokens`
  - `action_condition_tokens`
  - `control_query_state`
  - `pooled_state`
- retain only real predictive/action outputs:
  - semantic tokens
  - innovation token / norm
  - availability
  - action
  - action_chunk
  - executed_action
  - physical_global_pred
  - physical_prediction_cache
  - global_pred
  - prediction_cache

Current `PicfCoreState` was extended to:

- add `task_readout`
- add `conditioned_control`

Current-to-target field mapping must be written explicitly in code comments and
compat migration notes:

- current `predictive.action_condition_tokens`
  -> target `conditioned_control.pi_prefix_tokens`
- current `predictive.control_tokens`
  -> target `conditioned_control.tokens`
- current `predictive.control_query_state`
  -> target `conditioned_control.query_state`
- current `predictive.pooled_state`
  -> optional derived debug summary only; no longer canonical state

### 8.3 `src/openpi/picf/core/pipeline.py`

Do not structurally change:

- `_build_token_field`
- `_build_observation_anchors`
- `_posterior_update`
- `_current_targets`
- `_innovation`

Current helper: `_build_public_read_memory(...)`

Construct:

```text
public_read_memory =
[
  fused_tokens,
  visual_tokens,
]
```

Do not collapse this back to `fused_tokens` alone, but also do not bypass the
existing fused public field by reverting to raw pre-fusion point/tactile/context
tokens as the only public read source.

Important correction:

- `fused_tokens` stays the current fused point/tactile/context field
- `public_read_memory` becomes the new explicit public task-readout
  memory
- v2.2 does not overload one tensor to mean both

Current helper: `_build_task_readout(...)`

Inputs:

- token_field
- public_read_memory
- dense_memory
- semantic
- proprio token

Hard rule:

- `_build_task_readout(...)` must not take `posterior` as a direct input

Flow:

1. condition learned task queries from semantic tokens
2. read public read memory
3. reread visual private payload
4. reread tactile private groups
5. reread point private payload
6. run lightweight task self stack
7. derive geometry summaries from point attention
8. output `PicfTaskReadoutState`

This helper must explicitly read:

- `token_field.fused_tokens`
- `token_field.visual_tokens`
- `dense_memory.visual_payload`
- `dense_memory.tactile_group_tokens`
- `dense_memory.point_payload`

It must not directly consume `posterior`. Posterior only merges with task
readout later inside `_build_conditioned_control_state(...)`.

It must not silently fall back to reading only observation-anchor tokens.

Implementation hard requirement:

- if `task_query_conditioner` is built from `GatedCrossAttentionRead`, it must
  not inherit the current dormant `cross_gate=0` startup behavior unchanged
- either add a nonzero `gate_init` path, or use a dedicated ungated semantic
  conditioner for this block

Current helper: `_build_conditioned_control_state(...)`

Inputs:

- posterior
- innovation token
- proprio token
- task readout

Flow:

1. project physical base tokens to semantic width
2. project task-readout tokens to semantic width
3. concatenate base + task + conditioned control query tokens
4. run a single `control_world`
5. export:
   - unified control tokens
   - query state
   - `pi_prefix_tokens`
   - `future_condition_tokens`

This helper replaces both current control routes inside `_predictive_state(...)`:

- current PI0.5 action-conditioning route
- current semantic-prefix-primary internal control route

After v2.2 there is exactly one conditioned-control route through
`control_world(...)`.

Current helper: `_build_physical_predictive_basis(...)`

Inputs:

- posterior
- executed action
- proprio token

Outputs:

- physical predictive tokens
- physical global pred
- physical prediction cache

This helper owns the world-only predictive-basis logic and must only run after
executed action is known.

Current helper: `_build_conditioned_predictive_cache(...)`

Inputs:

- physical predictive token sequence
- conditioned control future tokens

Outputs:

- conditioned `global_pred`
- conditioned `prediction_cache`

This helper becomes the only conditioned future constructor. Raw semantic prefix
must not be concatenated directly here after the refactor.

The intended token-level contract is:

```text
K_t^{cond} = P_cond(H_t^{phys_pred}, C_t^{future})
```

Do not collapse this helper to a summary-only global/cache path.

The API is restructured as:

- add `observe_step(...)`
- add `finalize_with_action(...)`
- keep `step(...)` as compatibility wrapper only

`step(...)` no longer remains the official serving/export entrypoint.

Intended split:

- `observe_step(...)`: canonical pre-action stage
- `finalize_with_action(...)`: canonical post-action stage
- `step(...)`: compatibility wrapper only

### 8.4 `src/openpi/picf/paligemma/wrapper.py`

Keep action generation logic intact:

- `compute_action_flow_loss(...)`
- `sample_action_chunk(...)`

Rename interface semantics in code/docs:

- `extra_prefix_tokens` -> conceptually `pi_action_condition_tokens`

The implementation keeps signature compatibility where needed, and internal
naming/documentation reflect PI0.5 action-conditioning semantics.

Require:

- `supports_pi0_action_generation()` fail-fast enforcement when config requires
  deployed action generation

Wrapper non-goals for v2.2:

- do not redesign PI0.5 flow equations
- do not redesign denoise scheduling
- do not redesign prompt-side state injection
- do not redesign checkpoint topology

v2.2 changes the source and semantics of action-conditioning tokens, not the
PI0.5 generator itself.

### 8.5 `src/openpi/picf/policy.py`

Current exported policy class:

`PicfPi05Policy`

Fields:

- `core: PicfFullCore`
- `semantic_encoder: PaliGemmaSemanticEncoder`

Methods:

- `forward_train_transition(...)`
- `forward_window(...)` if batching windows at policy level is useful
- `act(...)`

Engineering rule:

- the policy surface returns typed result objects, not bare dictionaries
- the training-facing result exposes:
  - `output`
  - `observed`
  - `semantic_override`
  - `flow_override`
  - `next_state` as the compact recurrent carry, not the full `PicfCoreState`
- the serving-facing result exposes:
  - `action`
  - `action_chunk`
  - `state`
  - `debug`
  - `output`

This keeps the exported interface stable and prevents trainer/serve from
silently depending on ad-hoc string keys.

Training flow:

```text
encode semantic
-> core.observe_step(...)
-> compute PI0.5 flow loss with C_t^{pi}
-> finalize core with teacher-forced executed action
-> compute transition loss using action overrides
```

Teacher-forced executed-action rule:

- prefer `current.action` when present
- otherwise use the first action from `action_chunk_target`
- never silently depend on dataset-specific implicit assumptions that both must
  always be present

Serving flow:

```text
encode semantic
-> core.observe_step(...)
-> sample PI0.5 chunk with C_t^{pi}
-> take first action
-> finalize core with sampled executed action
-> return action + next state
```

`PicfPi05Policy` is the only exported semantic+core action surface used by:

- trainer
- serving
- rollout/eval entrypoints

### 8.6 `scripts/picf_core_train.py`

The trainer no longer treats this manual distributed glue sequence as the
canonical action path:

- `core.step(...)`
- `semantic_encoder.compute_action_flow_loss(...)`
- state action mutation
- `compute_transition_loss(...)`

It now uses:

- construct `PicfPi05Policy`
- call `policy.forward_train_transition(...)`

Loss-side supervision rule:

- `compute_transition_loss(...)` must build future targets from the next
  observation as stop-gradient teacher signals
- `extract_future_targets(...)` therefore wraps `core.extract_targets(...)` in
  `torch.no_grad()`
- next-observation targets are supervision values, not a second trainable branch
  of the same transition graph
- when `unroll_steps > 1`, the window trainer now uses a one-step lookahead:
  the already-computed `observed.current_targets` from transition `t+1` are
  detached and reused as the future supervision targets for transition `t`
- therefore shared middle frames inside one training window are not rebuilt
  twice on the loss side; only the final frame in the window still needs an
  explicit `extract_future_targets(...)` pass

Historical trainer glue removed as first-class action integration:

- direct canonical dependence on `core.step(...)`
- direct canonical dependence on `semantic_encoder.compute_action_flow_loss(...)`
- direct use of `output.state.predictive.action_condition_tokens`
- post-hoc mutation of predictive action fields as the normal training path

Retain:

- current DDP safety guards
- current compat checkpoint loader
- current invalid-window rejection behavior
- current grad clipping / logging infrastructure

These are already validated runtime guards and remain in trainer/runtime
scope after policy unification.

Additional trainer runtime rule:

- in multi-rank runs, `scripts/picf_core_train.py` must not silently inherit
  `TORCH_DISTRIBUTED_DEBUG=DETAIL` as the default runtime mode
- DDP startup now defaults `TORCH_DISTRIBUTED_DEBUG` to `INFO`
- if `TORCH_DISTRIBUTED_DEBUG=DETAIL` is present under DDP, the trainer
  fails fast at startup unless
  `OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL=1` is explicitly set
- this guard exists because DETAIL-level TCPStore/NCCL trace traffic can
  destabilize standalone multi-rank bring-up and produce misleading
  `Broken pipe` heartbeat noise even while training continues
- DDP launch also fails fast if `LOCAL_RANK` is missing

Current standard long-run launch profile:

- `--num-train-steps 30000`
- `--save-interval 2500`
- `--grad-clip-mode percentile`
- `--grad-clip-percentile 75`
- `--grad-clip-window 100`
- `--visual-activation-checkpointing`
- `--semantic-gradient-checkpointing`
- `--window-activation-checkpointing`
- `--diagnostic-interval 0`
- `--training-strategy fsdp_full_shard` for the current 4x40GB A100 FSDP investigation profile
- `--optimizer-sharding none` on that FSDP path; `zero1` remains a DDP-only fallback and is not sufficient for all-backbone v2.2 finetuning
- `--visual-finetune-mode full|frozen` is now the canonical visual-backbone contract. `full` preserves the default all-backbone profile; `frozen` keeps the V-JEPA encoder weights fixed without changing the rest of the training graph.
- semantic FSDP wrapping now uses a two-level exact contract: directly called PI0/PaliGemma runtime hot leaves (`embed_tokens`, per-layer `q/k/v/o` projections, per-layer `mlp`, and PI0 action/time projections) are wrapped first as explicit nested leaves, and the remaining semantic root still uses `ignored_states` for the minority float32 stabilizer parameters. The SigLIP vision tower and multimodal projector currently remain under the outer semantic root because their current image-path implementations are not yet nested-FSDP-safe under the present view-alias constraints.
- FSDP full-shard on this profile should use flat-parameter mode (`use_orig_params=False`) together with `backward_prefetch=BACKWARD_POST` and `limit_all_gathers=True`; the goal is to reduce parameter-view residency and backward all-gather overlap peaks without changing model math
- standard 4x40GB FSDP sharding is now recursive for large uniform-dtype subtrees with a 512MiB parameter-storage budget per boundary; this lets point/visual/tactile backbone wrappers and safe internal stacks split into smaller shards instead of wrapping an entire uniform subtree as one flat unit
- safe core stacks are now explicit FSDP child boundaries on this profile: `token_fusion`, `obs_self`, `posterior_self`, `task_self`, `predictive_world`, `predictive_semantic_world`, and `control_world`; the trainer now reattaches those wrapped children back onto `core` before the root wrap, so root FSDP only carries the remaining lighter core/projection parameters instead of one monolithic core shard
- the root FSDP boundary now explicitly ignores fully frozen backbone subtrees instead of flattening mixed `requires_grad` parameter sets. This is required for `visual_finetune_mode=frozen` and is the mature contract for any future frozen backbone mode, because `use_orig_params=False` root flattening is only valid over uniform trainability.
- transformer-stack entry now materializes every incoming activation once (`x = x.clone()`) before attention. This is mathematically exact and is now part of the 4x40GB contract because many PICF call sites batch tokens via `tokens[None, :]`, while FSDP can also hand stacks storage-sharing tensors whose aliasing is not reliably visible through `_base`; a single stack-entry clone is the clean boundary that prevents autograd multi-view alias failures inside residual attention blocks
- FSDP grad-norm measurement and percentile clipping on this profile must use an explicit global L2 reduction over local gradient shards instead of `FullyShardedDataParallel.clip_grad_norm_`; the semantic stack intentionally carries both bf16 bulk weights and minority float32 stabilizer parameters, so the stock helper's uniform-dtype assumption is not a valid contract here
- semantic gradient checkpointing remains enabled on that FSDP path; after routing PI0 flow-loss calls through module `forward(op, ...)` and collapsing the semantic stack to one FSDP boundary, non-reentrant checkpoint recomputation is again the correct memory-saving path rather than a forbidden custom-method re-entry
- the training stack still supports checkpointing the full `_PicfWindowTrainer.forward(...)` window body during training. That remains an exact fallback for extra peak-memory reduction, and the checkpoint input is still a standalone dummy leaf on the active CUDA device rather than a view into any FSDP flat parameter, so recompute keeps exact training math without feeding full-parameter gradients back into local shard metadata. It is now an explicit operator knob rather than something the foundation profile silently forces on every launch.
- the custom PI0/Gemma dual-branch semantic attention path now uses SDPA instead of the eager attention workspace. This is part of the standard 4x40GB profile because the eager path materializes a large attention buffer that fits at step 1 but blows up once optimizer state becomes resident; SDPA preserves the same training objective while removing that workspace peak.
- core transformer stacks (`token_fusion`, `obs_self`, `posterior_self`, `task_self`, `predictive_world`, `predictive_semantic_world`, `control_world`) now also use train-time non-reentrant activation recompute; this is part of the standard all-backbone v2.2 training path and does not alter the underlying objective
- the trainable Sonata point backbone and AnyTouch tactile encoder now also use train-time non-reentrant recompute on their main backbone forwards; this keeps all-backbone finetuning mathematically identical while shifting more of the per-rank memory burden from saved activations into recompute
- tokenwise-only projections and FFNs on the current hottest paths now support exact sequence chunking instead of monolithic execution. On the current profile this is enabled by default as `tokenwise_ff_chunk_size=64` for PICF core transformer/cross-attention FFNs and `semantic_tokenwise_chunk_size=64` as the legacy semantic compatibility knob. The live trainer now also exposes `semantic_projection_chunk_size` and `semantic_mlp_chunk_size` as finer-grained execution controls; under the standard 4x40GB full-shard profile, the balanced default is `semantic_projection_chunk_size=128` and `semantic_mlp_chunk_size=64`. This preserves the old semantic compatibility surface while giving the heavier semantic MLP path and the lighter projection path different exact-memory execution policies.
- the PI0/PaliGemma wrapper no longer adds an extra outer checkpoint around semantic forward blocks when the native language-model / vision-tower / expert-model checkpointing path is already active. This avoids redundant recompute while preserving the same gradients.
- the PI0/PICF semantic runtime now drops the unused outer causal-LM heads (`paligemma.lm_head`, `gemma_expert.lm_head`) immediately after checkpoint load. The live training path never routes through those logits heads, so removing them from the runtime graph is mathematically exact and prevents dead generation weights from inflating FSDP wrapping.
- standard multi-rank `FSDP full_shard` training now also standardizes `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; after the backbone recompute cut, the dominant remaining failure mode was allocator fragmentation (`reserved but unallocated` growing much larger than free memory), so v2.2 now treats expandable segments as part of the clean startup contract rather than a post-hoc workaround
- standard FSDP full-finetune startup no longer serializes checkpoint construction rank-by-rank; each rank builds in parallel, because the serialized path turned one large checkpoint load into a multi-stage startup stall without changing training semantics
- standard FSDP full-finetune startup now also stages the local PI0/PaliGemma checkpoint from shared `/mnt` storage into a node-local cache before rank-local `safe_open(...)/load_state_dict(...)`; this preserves training math while removing the shared-filesystem page-wait stall that appeared when four ranks loaded the same multi-GB semantic checkpoint concurrently. The default cache root is `~/.cache/openpi/pi0_checkpoints`, `OPENPI_STAGE_PI0_CHECKPOINT=auto` stages only `/mnt/...` sources, and `OPENPI_LOCAL_CHECKPOINT_CACHE_DIR` overrides the cache location when needed
- V-JEPA mixed precision on CUDA now uses one safe autocast contract for both frozen and trainable modes. The encoder stays in native fp32 weights and the forward path enters autocast when `visual_dtype` is `float16`/`bfloat16`, avoiding frozen-path conv bias dtype mismatches without changing the feature contract.
- window training now carries a canonical recurrent-carry object instead of forwarding the full `PicfCoreState` into `previous`. The carry contains only the fields that the next step actually consumes: `runtime_meta`, tactile contact hysteresis state, `posterior`, `predictive.executed_action`, and `predictive.physical_prediction_cache`. This is mathematically exact for the current v2.2 recurrence contract and removes non-recurrent semantic/control/task-readout state from the cross-step training graph.

These values are the current operational training defaults for v2.2 runs even
if historical baseline commands in older docs still show `--save-interval 5000`.

Important status note:

- this section records the implemented 4x40GB FSDP training contract and the
  code paths that now exist in `scripts/picf_core_train.py`
- the maintained v2.2 README should be read as the current operator/developer
  contract for the 4x40GB full-train path, not as a promise that every listed
  runtime knob is part of the default launch profile
- this file records:
  - what is implemented in code
  - which runtime measures are standard defaults
  - which runtime measures are explicit operator fallbacks
- it still must not overclaim beyond current evidence:
  - this file does **not** claim that `step 2500` or full `30000` completion has
    already been observed unless a later audit explicitly records that fact

### 8.6.1 Current Standard 4x40GB Deployment Profile

The current standard 4x40GB training profile is:

- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `accum_steps=1`
- `num_train_steps=30000`
- `save_interval=2500`
- `grad_clip_mode=percentile`
- `grad_clip_percentile=75`
- `grad_clip_window=100`
- `use_foundation_backbones=True`
- `use_tactile=True`
- `visual_finetune_mode=full`
- `visual_trainable=True`
- `tactile_trainable=True`
- `point_backbone_trainable=True`
- `semantic_trainable=True`
- `window_activation_checkpointing=False` by default
- `semantic_gradient_checkpointing=True`
- `tokenwise_ff_chunk_size=64`
- `semantic_tokenwise_chunk_size=64`
- `semantic_projection_chunk_size=128`
- `semantic_mlp_chunk_size=64`

This is a **full-train** profile.

It does **not** freeze:

- V-JEPA
- AnyTouch
- Sonata
- PaliGemma

It also does **not** rely on:

- LoRA
- CPU offload
- watchdog restart logic

### 8.6.1Z Current 6x40GB Full-PICF Extension

The same full-PICF contract can be launched on a 6x40GB A100 node by changing
only the distributed launch width:

- `--nproc_per_node=6`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`

The current 6-GPU long-run profile keeps the same v2.2 semantics:

- `picf_mode=enabled`
- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `accum_steps=1`
- `unroll_steps=2`
- `action_horizon=16`
- `num_train_steps=30000`
- `save_interval=2500`
- `log_interval=100`
- `window_activation_checkpointing=False`
- all foundation backbones trainable
- action normalization uses CALVIN `norm_stats.json`
- prompt-state normalization inherits the same CALVIN `norm_stats.json`

Important comparison note:

- the 4x40GB profile has `effective_global_batch=4`
- the 6x40GB extension has `effective_global_batch=6`
- this is intentional when all 6 GPUs are used with `accum_steps=1`
- loss curves from 4-GPU and 6-GPU runs should therefore be compared as
  same-objective but different-global-batch runs, not as bitwise-identical
  optimizer trajectories

The 6-GPU extension does not change:

- PI0.5 flow-matching action objective
- PICF observe/finalize dataflow
- physical posterior / innovation boundary
- task-readout / conditioned-control contract
- checkpoint cadence

### 8.6.1A Current 2x40GB PI0.5-Only Ablation Profile

The current operator-validated PI0.5-only ablation launch profile is:

- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `world_size=2`
- `accum_steps=1`
- `unroll_steps=2`
- `action_horizon=16`
- `num_train_steps=30000`
- `save_interval=2500`
- `log_interval=100`
- `lr=2e-4`
- `min_lr=2e-5`
- `warmup_steps=600`
- `grad_clip_mode=percentile`
- `grad_clip_percentile=75`
- `grad_clip_window=100`
- `picf_mode=ablated`
- `semantic_mode=paligemma`
- `semantic_trainable=True`
- `semantic_max_length=256`
- `action_normalization=quantile`
- `prompt_state_normalization=inherit`
- `window_activation_checkpointing=False`
- `semantic_gradient_checkpointing=True`
- `semantic_tokenwise_chunk_size=64`
- `semantic_projection_chunk_size=128`
- `semantic_mlp_chunk_size=64`

This profile is **not** the canonical full PICF training profile above.

It is an operational ablation profile used to answer a narrower question:

- if PICF recurrent/control/future branches are disabled, can the current
  repository train the native PI0.5 semantic action path cleanly under the
  PICF trainer/runtime shell?

The current semantics of this run shape are:

- each rank samples `1` training window per optimizer step
- with `unroll_steps=2`, each sampled window contains `3` frames and produces
  `2` action-only transitions
- with `world_size=2` and `accum_steps=1`, one optimizer step therefore covers:
  - `2` sampled windows globally
  - `4` action-only transition objectives globally

Current training-definition bugfixes that now apply globally to both
`picf_mode=enabled` and `picf_mode=ablated`:

- `action_horizon` has been restored to `16` by default
  - this re-aligns the maintained training contract with the historical CALVIN
    PI0.5 chunk contract
- `_CalvinTransitionSource` no longer samples uniformly over all valid window
  starts
  - it now samples segments uniformly first, then samples a valid start within
    the chosen segment
  - this matches the historical CALVIN dataset semantics more closely and is the
    maintained sampling contract for both full PICF and PI0.5-only ablations
- prompt-state tokenization now reuses the shared CALVIN `norm_stats.json`
  contract instead of clipping raw `robot_obs`
  - `state` is normalized before prompt discretization, matching the reference
    PI0.5 preprocessing path
  - state is **not** padded to `action_dim = 32` before prompt tokenization;
    reference PI0.5 tokenizes the normalized live state first and pads only
    after tokenization
  - raw physical-core `robot_obs` remains unnormalized

Interpretation rule:

- this is the current **operational** ablation profile because it preserves the
  present PICF window trainer shape while disabling PICF semantics
- it is **not** identical to the main-branch PI0.5 training definition
- it is also **not** identical to the historical `pi0.5_sonata` model path,
  because `picf_mode=ablated` does not instantiate the live PICF point / visual /
  tactile backbones:
  - Sonata point feature extractor is not built
  - V-JEPA visual encoder is replaced by the null visual encoder
  - AnyTouch tactile encoder is replaced by the null tactile encoder
  - the PICF core is frozen and only the PI0.5 semantic/action path remains live
- it is also **not** identical to the preserved historical 2-GPU PI0.5 runtime
  shell:
  - the maintained ablation launch uses `training_strategy=fsdp_full_shard`
  - the preserved historical PI0.5 CALVIN baselines were run through the direct
    DDP trainer in `scripts/train_pytorch.py`
- it is also **not** identical to the preserved PI0.5 CALVIN prompt budget:
  - the maintained ablation launch uses `semantic_max_length=256`
  - the historical CALVIN PI0.5 configs use `max_token_len=200`
- it is also **not** identical to the preserved PI0.5 CALVIN optimizer regime:
  - the maintained ablation launch uses `lr=2e-4`, `min_lr=2e-5`,
    `warmup_steps=600`
  - the preserved cloud `pi05_calvin_nosonata/abc_train_nosonata_full_ddp2`
    run used `warmup=10000`, `peak_lr=5e-5`, `end_lr=5e-5`
  - the generic codebase `CosineDecaySchedule` default is
    `peak_lr=2.5e-5`, `decay_lr=2.5e-6`, `warmup_steps=1000`
  - these are not the same reference, so optimizer-parity claims should cite
    which historical baseline they mean
- if the goal is exact training-definition parity with the official/main-branch
  PI0.5 stack, the cleaner baseline is `picf_mode=ablated` with
  `unroll_steps=1`, `semantic_max_length=200`, and the exact optimizer regime of
  the PI0.5 reference being compared
- if the goal is exact `pi0.5_sonata` parity, the current `picf_mode=ablated`
  profile is not sufficient by itself because it does not preserve the old
  Sonata prefix-injection path
- if the goal is "same trainer shell, same loader shape, same optimizer loop,
  but no PICF semantics", then the current `unroll_steps=2` ablation profile is
  a legitimate control experiment

Current cloud launch command for this profile:

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=/root/openpi_posterior_vla_clean/src
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name picf_v22_ablated_pi05_30000_ckpt2500_print100 \
  --overwrite \
  --device cuda \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --accum-steps 1 \
  --unroll-steps 2 \
  --action-horizon 16 \
  --num-train-steps 30000 \
  --save-interval 2500 \
  --log-interval 100 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 600 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --wandb-mode disabled \
  --no-wandb \
  --picf-mode ablated \
  --semantic-mode paligemma \
  --semantic-trainable \
  --semantic-max-length 256 \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --action-normalization quantile \
  --action-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --prompt-state-normalization inherit \
  --prompt-state-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json
```

Recommended live monitoring commands:

```bash
tail -f /mnt/checkpoints/picf_core/debug/picf_v22_ablated_pi05_30000_ckpt2500_print100_*.log
```

```bash
watch -n 2 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

Operational checkpoint rule:

- on this profile, `save_interval=2500` is the maintained default because it is
  frequent enough for early checkpoint inspection without changing the training
  objective or runtime mode

Shared-shell reminder:

- the launch block above is an ablation launch, but the shell-level controls it
  demonstrates are still owned by the shared `scripts/picf_core_train.py`
  parser/runtime
- in practice, `--save-interval`, `--log-interval`, `--accum-steps`,
  `--unroll-steps`, and `--action-horizon` can be applied to
  `picf_mode=enabled` as well
- what changes between modes is the inner policy/core path and the default
  checkpoint payload under `--optimizer-checkpoint-mode auto`, not the outer
  training-loop cadence semantics

### 8.6.2 Current Exact-Memory Runtime Measures

The current 4x40GB full-train path relies on the following mathematically exact
runtime measures:

1. tokenwise exact chunking on the hot PICF core FFN/cross-attention FFN paths
2. tokenwise exact chunking on the hot PI0/Gemma tokenwise projection/MLP paths
3. nested semantic hot-leaf FSDP wrapping
4. dead outer semantic generation heads dropped after checkpoint load
5. recursive FSDP subtree splitting on large uniform-dtype subtrees
6. explicit safe core-stack FSDP child boundaries
7. global-L2 shard-aware grad norm / percentile clipping
8. optional full-window activation checkpointing during training when the
   operator explicitly enables it
9. semantic gradient checkpointing
10. train-time recompute on the core transformer stacks
11. train-time recompute on the Sonata / AnyTouch backbone forwards
12. SDPA on the custom PI0/Gemma dual-branch attention path
13. allocator contract `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
14. node-local staging of the shared PI0/PaliGemma checkpoint
15. compact recurrent-carry instead of forwarding the full `PicfCoreState`
16. suppression of redundant outer wrapper checkpointing when the native
    PI0/PaliGemma layer checkpointing path is already active

The current safe nested semantic hot-leaf set is:

- `embed_tokens`
- per-layer `self_attn.q_proj`
- per-layer `self_attn.k_proj`
- per-layer `self_attn.v_proj`
- per-layer `self_attn.o_proj`
- per-layer `mlp`
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`

The following remain under the outer semantic root because they are not yet
nested-FSDP-safe under the current image-path alias constraints:

- `vision_tower`
- `multi_modal_projector`

### 8.6.3 Current Throughput Diagnosis

The current exact-memory contract above is mathematically correct, but the
latest audits now point to a specific throughput problem:

- the dominant slowdown is in semantic execution fragmentation
- it is not primarily in `task_readout`, conditioned control, or the physical
  finalize path

Two concrete execution facts matter:

1. nested semantic FSDP is currently very fine-grained
   - the live runtime-hot semantic set expands to `185` nested leaves:
     `1 + 18*5 + 18*5 + 4`
   - this preserves exact training math, but it also means the custom
     dual-branch semantic path can trigger many small FSDP gather / reshard
     events
2. semantic tokenwise chunking is currently one blunt knob
   - under the standard 4x40GB profile, `semantic_tokenwise_chunk_size=64`
     remains the compatibility default
   - the live trainer now resolves that compatibility knob into:
     - `semantic_projection_chunk_size`
     - `semantic_mlp_chunk_size`
   - under the current balanced full-shard default, those resolve to:
     - `semantic_projection_chunk_size=128`
     - `semantic_mlp_chunk_size=64`
   - if an operator explicitly sets the split knobs, those explicit values win
   - for rough live sequence scales near `784` tokens, this implies about:
     - `7` chunks per projection-family tokenwise op
     - `13` chunks per MLP-family tokenwise op
   - the old single-knob arithmetic therefore overestimates projection-side
     launch count once the split controls are active

Current engineering conclusion:

- the exact-memory profile is currently clean but throughput-expensive
- direct training-side prefix-KV reuse must **not** be assumed to be exact under
  the current contract, because the PI0 semantic path still appends
  `extra_prefix_tokens` into a bidirectional prefix-LM block; any future prefix
  runtime reuse therefore needs an explicit capability contract from the
  semantic backbone rather than a hard-coded PaliGemma-specific shortcut
- the next optimization pass should stay mathematically exact and target:
  - coarser semantic FSDP execution blocks
  - separated chunk controls for semantic projections vs semantic MLPs

This section is intentionally a diagnosis, not a claim that the throughput
problem has already been solved.

### 8.6.4 Operator Display / Observability Modes

Two display modes are currently useful and both preserve the same optimization
math:

1. Standard long run
   - keep the progress bar enabled
   - use `--log-interval 100`
   - this is the normal operator-facing mode for 30000-step training
2. Early observability verification
   - optionally use `--no-progress`
   - use `--log-interval 10`
   - this exists only to prove early training progress quickly, for example
     when the operator explicitly wants direct evidence that the job crossed
     `step 10`

Important clarification:

- `--no-progress` changes only what gets rendered to the terminal
- `--log-interval` changes only how often the metrics JSON line is printed
- neither changes training math

### 8.6.5 Current Cloud Launch Templates

The current standard long-run launch template is:

```bash
cd /root/openpi_run_latest
export PYTHONPATH=/root/openpi_run_latest/src
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

/root/openpi/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name <exp_name> \
  --overwrite \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --accum-steps 1 \
  --num-train-steps 30000 \
  --save-interval 2500 \
  --log-interval 100 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --wandb-mode disabled \
  --no-wandb \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --tokenwise-ff-chunk-size 64 \
  --semantic-tokenwise-chunk-size 64
```

The current early-observability verification template is the same command with:

- `--log-interval 10`
- optionally `--no-progress`

Cloud detach rule:

- long runs on rented cloud machines should be launched in a detached session,
  not as a plain SSH-attached foreground/background command
- this is an operational rule only; it does not change the training graph,
  losses, optimizer, checkpoint cadence, or model math
- the observed failure mode from a plain SSH-attached `torchrun` launch is
  `torch.distributed.elastic.multiprocessing.api.SignalException: ... signal: 1`,
  i.e. an external SIGHUP delivered to the elastic launcher after the shell or
  SSH session exits
- the maintained clean launch pattern is to write the exact `torchrun` command
  into a script and start it with `setsid + nohup + stdin redirected from
  /dev/null`

Example detached launch skeleton:

```bash
RUN=/mnt/checkpoints/picf_core/debug/run_<exp_name>.sh
LOG=/mnt/checkpoints/picf_core/debug/<exp_name>.log

chmod +x "$RUN"
nohup setsid "$RUN" </dev/null > "$LOG" 2>&1 &
```

After reconnecting, verify the launcher has no controlling TTY:

```bash
ps -o pid,ppid,sid,tty,etime,stat,cmd -C torchrun
```

The expected healthy state is `PPID=1` or otherwise independent from the
interactive SSH shell, and `TTY=?`.

### 8.6.6 Operationally Important Knobs

The current operator-facing knobs that matter most are:

Display / logging:

- `--log-interval`
- `--save-interval`
- `--no-progress`
- `--diagnostic-interval`

Training envelope:

- `--training-strategy`
- `--optimizer-sharding`
- `--accum-steps`
- `--num-train-steps`
- `--grad-clip-mode`
- `--grad-clip-percentile`
- `--grad-clip-window`

Backbone trainability:

- `--use-foundation-backbones`
- `--use-tactile`
- `--visual-finetune-mode`

Exact-memory controls:

- `--semantic-gradient-checkpointing`
- `--window-activation-checkpointing`
- `--tokenwise-ff-chunk-size`
- `--semantic-tokenwise-chunk-size`

Interpretation rule:

- if a run must preserve the current standard 4x40GB full-train contract, do not
  change the exact-memory controls casually; those are part of the current fit
  proof, not decorative micro-optimizations

### 8.6.7 GitHub Handoff Scope

For this v2.2 rollout, the GitHub commit scope should include:

- code implementing the exact-memory training contract
- `README_v2.2.md`
- `PICF_FORMAL_CONTRACT.md`
- `docs/CALVIN_VALIDATION_README.md`
- test and verifier updates

It should **not** include:

- `/tmp/...` audit documents
- `/tmp` cloud launch helper scripts
- transient cloud logs
- transient checkpoints

The `/tmp` audits are intentionally local operator artifacts. They are used to
derive the maintained README and contract docs, not to replace them in version
control.

### 8.7 `scripts/serve_picf_policy.py`

Serving no longer treats this manual sequence as the deployed action path:

- `core.step(...)`
- `sample_action_chunk(...)`
- `refresh_predictive_state_for_action(...)`

Serving now uses:

- `policy.act(...)`

Historical serving glue removed from the deployed path:

- `core.step(..., action_future=None)` as the public action API
- direct `sample_action_chunk(...)` call from the serve script
- direct `refresh_predictive_state_for_action(...)` call from the serve script

Serving must fail fast if:

- semantic encoder missing
- PI0.5 action generation unsupported

Serving also accepts an explicit runtime override:

- `python scripts/serve_picf_policy.py --checkpoint ... --picf-mode enabled`
- `python scripts/serve_picf_policy.py --checkpoint ... --picf-mode ablated`

If `--picf-mode` is omitted, serving uses the checkpoint's saved runtime mode.

### 8.8 `scripts/verify_picf_contract.py`

The verifier was migrated away from semantic-prefix-primary inside core and now
checks the v2.2 semantics directly.

Removed checks that asserted:

- raw semantic prefix remains primary input to core control path
- raw semantic prefix remains primary input to conditioned future path
- dual control semantics are expected

Added checks that assert:

1. semantic does not enter observation anchors
2. semantic does not enter posterior update
3. innovation reads only previous physical prediction cache
4. `_build_task_readout(...)` exists
5. task readout consumes public read memory and `_StepDenseMemory`
6. exactly one conditioned control-state builder exists
7. PI0.5 action generation consumes only `conditioned_control.pi_prefix_tokens`
8. conditioned future depends on `K_phys` and `C_t`
9. raw semantic prefix is no longer a direct core control/future trunk input
10. serving/export requires PI0.5 action generator

Also add negative checks:

11. `action_condition_tokens` is no longer a canonical independent control
    semantics
12. `control_query_state` is no longer a second control semantics
13. `task_readout` is not stored as recurrent world state
14. public task-readout memory includes `visual_tokens`
15. only one conditioned-control route through `control_world(...)` remains

## 9. Checkpoint and Compatibility Record

This is a breaking structural patch. Compatibility is explicit.

### 9.1 Reuse Existing Weights Where Possible

Warm-start:

- point / visual / tactile backbones
- token builders
- observation anchor reader
- posterior update stack
- innovation stack
- predictive world stack
- prediction heads
- PI0.5 semantic/action generator
- `control_world`
- `predictive_semantic_world`

### 9.2 Reinitialize New v2.2 Modules

New modules to initialize:

- task query tokens
- task query conditioner
- task public reader
- task private reread readers
- task self stack
- task geometry projection
- `pi_prefix_query_tokens`
- `pi_prefix_reader`
- new conditioned-control projections
- new role embeddings
- new dataclass-carried interface heads

### 9.3 Compat Loader Migration Is Part of v2.2

The patch updates:

- `_COMPAT_ALLOWED_MISSING_KEYS`
- `_COMPAT_ALLOWED_UNEXPECTED_KEYS`
- any shape-mismatch whitelist assumptions

This is required for safe warm-start from current checkpoints.

Because current trainer already uses:

- `_COMPAT_ALLOWED_MISSING_KEYS`
- `_COMPAT_ALLOWED_UNEXPECTED_KEYS`
- shape mismatch filtering before relaxed load

the patch updates those explicitly. Do not assume generic `strict=False`
loading semantics.

## 10. Validation Matrix

### 10.1 Mathematical Boundary Tests

1. `test_semantic_does_not_change_physical_posterior`
2. `test_semantic_does_not_change_physical_prediction_basis_when_action_fixed`
3. `test_previous_conditioned_state_does_not_change_next_innovation`

### 10.2 Task-Readout Structure Tests

4. `test_task_readout_reads_public_read_memory_and_private_dense_memory`
5. `test_task_readout_changes_with_prompt_but_physical_core_does_not`
6. `test_only_one_conditioned_control_state_exists`
7. `test_pi_prefix_tokens_are_the_only_action_conditioning_tokens`

These tests explicitly cover:

- `visual_tokens` being part of public read memory
- `_StepDenseMemory.visual_payload`
- `_StepDenseMemory.tactile_group_tokens`
- `_StepDenseMemory.point_payload`
- no writes from task-readout outputs into recurrent posterior state

### 10.3 Exported Policy Tests

8. `test_policy_act_matches_manual_observe_sample_finalize_sequence`
9. `test_policy_fails_fast_without_pi05_action_generator`
10. `test_conditioned_future_depends_only_on_kphys_and_Ct`

Add parity tests before deleting old glue:

- trainer parity: old manual glue vs `PicfPi05Policy.forward_train_transition(...)`
- serve parity: old manual glue vs `PicfPi05Policy.act(...)`

### 10.4 Loader / Compat Tests

11. shape-changed role embedding migration test
12. task-readout missing keys allowed during compat warm-start
13. removed semantic-prefix-primary control/future keys allowed as unexpected

### 10.5 Existing Test Files To Extend

Primary extension targets:

- `src/openpi/picf/core/pipeline_test.py`
- `src/openpi/picf/paligemma/wrapper_test.py`
- `scripts/picf_core_train_test.py`
- `scripts/serve_picf_policy_test.py`

Add new test files only if these become unreadable.

## 11. Rollout Gate Record

Completed local gates:

- `py_compile` on modified core / wrapper / policy / scripts
- unit tests
- `python scripts/verify_picf_contract.py`
- local smoke through the contract verifier

Remaining runtime-stage gates:

- single-GPU minimal train on target hardware
- 2-GPU / multi-GPU DDP minimal train
- warm-start and partial-reinit short-run A/B on cloud hardware
- cloud long run

## 12. Explicit "Do Not Do" List

Do not:

1. inject semantic into observation-anchor construction
2. inject semantic into posterior update
3. let next innovation read conditioned future or `C_t`
4. keep a direct trainable 7D core action head as deployed action path
5. keep raw semantic prefix as a separate direct core control/future input
6. leave Route A / Route B dual control semantics alive
7. turn task readout into recurrent state
8. silently fall back to placeholder action in serving/export

## 13. Local Completion Criteria

Current local v2.2 satisfies the structural completion criteria when all of the
following are true:

- physical core math remains unchanged in the protected world-only portions
- `PicfPi05Policy` is the only exported action API
- there is one canonical `conditioned_control` state `C_t`
- PI0.5 is the only final action path
- raw semantic prefix no longer directly enters core control/future trunks
- task readout exists and reads both explicit public memory and private dense
  memory
- compat loader migration lands with the structural patch
- contract verifier is rewritten to the new semantics
- train / serve both use the unified policy path
- new tests for boundaries, structure, compat, and export behavior pass

## 14. Final Local Verdict

This refactor was worth doing in one patch because current code already had the
hard substrate in place:

- physical core math is already right
- private dense memory already exists
- native V-JEPA and tactile dense reread are already active
- PI0.5 flow/sampler path is already restored
- current trainer/serve already prove the pre-action / action / post-action
  timing is workable

Therefore the v2.2 patch does **not** redesign the physical core. It performs
one clean integration-layer rewrite:

1. add `task readout`
2. collapse dual control semantics into one canonical `C_t`
3. introduce `PicfPi05Policy`
4. remove raw semantic prefix as a direct core control/future mainline

That is the correct one-shot target.

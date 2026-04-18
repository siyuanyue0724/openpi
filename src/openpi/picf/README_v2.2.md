# PICF v2.2

Date: 2026-04-17
Repo: `/home/siyuanyue/Documents/openpi`
Status: current local v2.2 architecture record after the one-shot
action/control contract rewrite

## 0. Document Map

This file is the **current local v2.2 architecture record and implementation
audit**. The one-shot action/control refactor described below has been deployed
locally; the detailed planning sections are retained because they explain the
implemented contract rewrite and the reasoning behind it.

Relevant documents:

1. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md`
   This file. Current v2.2 architecture record. It records:
   - what the current live code implements
   - what changed in the v2.2 contract rewrite
   - the canonical object/state decomposition
   - file-by-file implementation scope and rationale
   - migration, testing, and rollout gates used to validate the patch

2. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md`
   Historical v2.1 deployment record from before the v2.2 refactor landed.

3. `/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md`
   Current concise executable contract enforced by regression tests. It is the
   compact version of the live v2.2 semantics described in this file.

4. `/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md`
   Current training / validation / rollout / serving workflow document.

5. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md`
   Directory entry / pointer file. It points at both the current live
   record (`README_v2.2.md`) and the archived v2.1 record so neither is lost.

Maintained/current docs:

- `src/openpi/picf/README_v2.2.md`
- `PICF_FORMAL_CONTRACT.md`
- `docs/CALVIN_VALIDATION_README.md`
- `src/openpi/picf/README.md` as entry pointer

Historical/archive docs:

- `src/openpi/picf/README_v2.1.md`

### 0.1 Read Order

If you are opening the repo cold, use this order:

1. [`README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
   Entry pointer only.
2. [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
   Current live architecture record and implementation audit.
3. [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
   Concise executable contract for the current code.
4. [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
   Runtime / training / rollout workflow.
5. [`README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
   Historical pre-v2.2 record only.

### 0.2 Temporary Audit Companions

For this local v2.2 rollout, the following temporary audit documents were
generated and are read alongside this file when deep verification is
needed:

1. [`/tmp/picf_v22_current_code_dataflow_20260418.md`](/tmp/picf_v22_current_code_dataflow_20260418.md)
   Recursive current-code dataflow audit.
2. [`/tmp/picf_v22_mathematical_spec_20260418.md`](/tmp/picf_v22_mathematical_spec_20260418.md)
   Mathematical object/transition specification derived from current code and
   the v2.2 contract.
3. [`/tmp/picf_v22_final_reconciliation_20260418.md`](/tmp/picf_v22_final_reconciliation_20260418.md)
   Final reconciliation of current code vs v2.2 contract, including pass/fail
   items and residual compatibility notes.

These `/tmp` documents are audit artifacts, not persistent maintained docs.

### 0.3 Section Guide

Use the sections in this file as follows:

- Section 2: recursive audit of the current live implementation
- Section 3: canonical v2.2 object/state shape
- Section 4: non-negotiable invariants
- Section 5: corrections to the earlier external proposal
- Section 6: detailed end-to-end v2.2 dataflow
- Section 7: file-by-file implementation map
- Section 8: checkpoint and compatibility migration
- Section 9: validation matrix
- Section 10: rollout gate record
- Section 11: forbidden regressions
- Section 12: definition of done
- Section 13: final recommendation

### 0.4 Navigation Summary

When verifying the local v2.2 codebase, use this navigation split:

- architecture and rationale:
  - [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
- concise executable rules:
  - [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
- runtime / training / rollout workflow:
  - [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
- historical pre-v2.2 reference only:
  - [`README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
- temporary deep audits for this local rollout:
  - [`/tmp/picf_v22_current_code_dataflow_20260418.md`](/tmp/picf_v22_current_code_dataflow_20260418.md)
  - [`/tmp/picf_v22_mathematical_spec_20260418.md`](/tmp/picf_v22_mathematical_spec_20260418.md)
  - [`/tmp/picf_v22_final_reconciliation_20260418.md`](/tmp/picf_v22_final_reconciliation_20260418.md)

## 1. Scope

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

v2.2 is therefore treated as:

- contract rewrite
- interface unification
- task-readout refactor

and **not** as a cosmetic cleanup.

## 2. Current Live Code Audit

This section records what was already correct in current code and what the
v2.2 rewrite changed.

### 2.1 Physical Core: Keep

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

### 2.2 Current-Step Private Dense Memory: Already Present

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

### 2.3 Attention Primitives: Already Sufficient

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

### 2.4 Historical Seam 1: Action Path Used To Be Glue

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

### 2.5 Historical Seam 2: Dual Control Semantics Used To Exist

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

### 2.6 Historical Seam 3: Raw Semantic Prefix Used To Enter Core Control/Future

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

### 2.7 Important Current Code Fact: `fused_tokens` Is Not Full Multimodal Public Memory

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

### 2.8 Current State Dataclasses: Exact Audit

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

### 2.9 Current `PicfFullCore.__init__`: Exact Module Inventory

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

### 2.10 Current `_build_token_field(...)`: Recursive Flow

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

### 2.11 Current `_build_observation_anchors(...)`: Recursive Flow

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

### 2.12 Current `_posterior_update(...)`: Recursive Flow

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

### 2.13 Current `_current_targets(...)` and `_innovation(...)`

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

### 2.14 Current `_predictive_state(...)`: Post-v2.2 Role

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

### 2.15 Current `refresh_predictive_state_for_action(...)` and `step(...)`

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

### 2.16 Current Wrapper / PI0.5 Audit

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

### 2.17 Current Trainer Audit

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

### 2.18 Current Serve Audit

Current `scripts/serve_picf_policy.py` sequences inference manually:

1. encode semantic observation
2. `core.step(..., action_future=None)`
3. sample PI0.5 action chunk using
   `output.state.predictive.action_condition_tokens`
4. call `core.refresh_predictive_state_for_action(...)`
5. export first action

This confirms exactly what the v2.2 exported policy now unifies.

### 2.19 Current Verifier and Tests Audit

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

### 2.20 Current Default Configuration Snapshot

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

Current training/runtime assumptions relevant to v2.2:

- trainer already expects compat loading and non-strict migration paths
- trainer already has DDP-specific gradient-checkpointing guards
- serving already assumes normalized-core / unnormalized-environment action
  contract

These are operating assumptions to preserve during the refactor, not knobs to
revisit inside the same patch.

## 3. Canonical v2.2 Shape

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

## 4. Non-Negotiable v2.2 Invariants

These are hard constraints, not suggestions.

### 4.1 Physical Core Remains Language-Free

Semantic must not enter:

- `_build_observation_anchors(...)`
- `_posterior_update(...)`
- `_innovation(...)`

### 4.2 Innovation Base Remains World-Only

Next-step innovation may read only:

- `previous.predictive.physical_prediction_cache`

It must not read:

- `previous.predictive.prediction_cache`
- `previous.conditioned_control`
- `previous.task_readout`
- semantic-conditioned state of any kind

### 4.3 `C_t` Is the Only Canonical Conditioned Control State

After the refactor, these must no longer exist as independent control
semantics:

- `action_condition_tokens`
- `control_query_state`
- `pooled_state`

If any compatibility alias survives, it may only be a view/debug snapshot of
the canonical `conditioned_control`.

### 4.4 `C_t^{pi}` Is an Interface View, Not a Second Control State

`C_t^{pi}` is only the prefix representation exported to PI0.5.

It is derived from `C_t` and is not itself a second control-path object.

### 4.5 `task_readout` Is Current-Step Only

`task_readout` must not become:

- recurrent memory
- task posterior
- next-step prior input
- next-step innovation base

It exists only to build `C_t` and conditioned future context.

### 4.6 `K_t^{phys}` Must Be Computed After Executed Action Is Known

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

### 4.7 Serving Must Fail Fast Without PI0.5 Action Generation

Formal deployed serving must not silently fall back to placeholder action.

The only valid deployed action path is:

- `PicfPi05Policy.act(...)`

## 5. Implemented Corrections to the External Proposal

The external proposal was directionally correct. These corrections were applied
in the same patch.

### 5.1 Public Read Memory Must Preserve Existing Public Fusion

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

### 5.2 Contract Rewrite Must Be Explicit

Current formal contract is semantic-prefix-primary in core control/future.

v2.2 replaces that with:

- PI0.5 keeps raw semantic stream
- core gets semantic influence only through `task_readout`
- `C_t` becomes the only conditioned-control object

This is a contract rewrite, not a minor cleanup.

### 5.3 `C_t` Must Keep Token-Level Richness

`C_t` must not collapse too early to one pooled vector.

It preserves at minimum:

- `base_tokens`
- `task_tokens`
- `tokens` (post-control-trunk sequence)

and only then derive:

- `pi_prefix_tokens`
- `future_condition_tokens`

### 5.4 Instruction Query Count Should Not Stay at 1

If raw semantic prefix is removed as a direct core control/future input, one
instruction token is too aggressive a bottleneck.

The current live default is:

- `task_instruction_queries = 2`

This is the conservative default. It keeps instruction semantics richer without
exploding complexity.

### 5.5 Compat Loader Migration Must Ship in the Same Patch

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

### 5.6 DDP Safety Guards Stay in Trainer/Runtime Layer

`PicfPi05Policy` unifies action/state interfaces. It does **not** absorb
runtime/DDP-specific guards that are already working in the training script.

Those include:

- gradient-checkpointing restrictions under DDP
- pre-DDP rejection sampling for invalid first-step windows
- compact startup logging
- current compat checkpoint loader behavior
- post-sampler predictive refresh timing

## 6. Detailed v2.2 Dataflow

### 6.1 Observation and Physical World Update

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

### 6.2 Semantic-Conditioned Task Readout

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

### 6.3 Unique Conditioned Control State

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

### 6.4 Unique Final Action Path

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

### 6.5 Physical Predictive Basis and Conditioned Future

```text
K_t^{phys} = P_phys(W_t, a_t^{exec}, proprio_t)
K_t^{cond} = P_cond(K_t^{phys}, C_t^{future})
```

This preserves:

- world-only predictive basis for innovation
- conditioned future for semantic/task-aware forecasting

## 7. File-by-File v2.2 Implementation Record

### 7.1 `src/openpi/picf/core/config.py`

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
- `task_query_rounds: int = 2`
- `task_self_layers: int = 1`
- `conditioned_control_queries: int = 4`
- `pi_prefix_queries: int = 4`
- `conditioned_future_queries: int = 2`
- `task_visual_reread_topk: int = 32`
- `task_tactile_reread_groups: int = 2`
- `task_point_reread_topk: int = 32`
- `require_pi0_action_generator: bool = True`

Mark as deprecated compatibility-only:

- `predictive_semantic_reads`
- `control_semantic_reads`
- `predictive_semantic_dropout_prob`
- `semantic_prefix_dropout_prob`

### 7.2 `src/openpi/picf/core/contracts.py`

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

### 7.3 `src/openpi/picf/core/pipeline.py`

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

### 7.4 `src/openpi/picf/paligemma/wrapper.py`

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

### 7.5 `src/openpi/picf/policy.py`

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
  - `next_state`
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

### 7.6 `scripts/picf_core_train.py`

The trainer no longer treats this manual distributed glue sequence as the
canonical action path:

- `core.step(...)`
- `semantic_encoder.compute_action_flow_loss(...)`
- state action mutation
- `compute_transition_loss(...)`

It now uses:

- construct `PicfPi05Policy`
- call `policy.forward_train_transition(...)`

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
- `--training-strategy fsdp_full_shard` for the standard 4x40GB A100 full-finetune profile
- `--optimizer-sharding none` on that FSDP path; `zero1` remains a DDP-only fallback and is not sufficient for all-backbone v2.2 finetuning

These values are the current operational training defaults for v2.2 runs even
if historical baseline commands in older docs still show `--save-interval 5000`.

### 7.7 `scripts/serve_picf_policy.py`

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

### 7.8 `scripts/verify_picf_contract.py`

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

## 8. Checkpoint and Compatibility Record

This is a breaking structural patch. Compatibility is explicit.

### 8.1 Reuse Existing Weights Where Possible

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

### 8.2 Reinitialize New v2.2 Modules

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

### 8.3 Compat Loader Migration Is Part of v2.2

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

## 9. Validation Matrix

### 9.1 Mathematical Boundary Tests

1. `test_semantic_does_not_change_physical_posterior`
2. `test_semantic_does_not_change_physical_prediction_basis_when_action_fixed`
3. `test_previous_conditioned_state_does_not_change_next_innovation`

### 9.2 Task-Readout Structure Tests

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

### 9.3 Exported Policy Tests

8. `test_policy_act_matches_manual_observe_sample_finalize_sequence`
9. `test_policy_fails_fast_without_pi05_action_generator`
10. `test_conditioned_future_depends_only_on_kphys_and_Ct`

Add parity tests before deleting old glue:

- trainer parity: old manual glue vs `PicfPi05Policy.forward_train_transition(...)`
- serve parity: old manual glue vs `PicfPi05Policy.act(...)`

### 9.4 Loader / Compat Tests

11. shape-changed role embedding migration test
12. task-readout missing keys allowed during compat warm-start
13. removed semantic-prefix-primary control/future keys allowed as unexpected

### 9.5 Existing Test Files To Extend

Primary extension targets:

- `src/openpi/picf/core/pipeline_test.py`
- `src/openpi/picf/paligemma/wrapper_test.py`
- `scripts/picf_core_train_test.py`
- `scripts/serve_picf_policy_test.py`

Add new test files only if these become unreadable.

## 10. Rollout Gate Record

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

## 11. Explicit "Do Not Do" List

Do not:

1. inject semantic into observation-anchor construction
2. inject semantic into posterior update
3. let next innovation read conditioned future or `C_t`
4. keep a direct trainable 7D core action head as deployed action path
5. keep raw semantic prefix as a separate direct core control/future input
6. leave Route A / Route B dual control semantics alive
7. turn task readout into recurrent state
8. silently fall back to placeholder action in serving/export

## 12. Local Completion Criteria

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

## 13. Final Local Verdict

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

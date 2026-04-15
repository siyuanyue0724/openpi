# PICF Semantic Prefix Primary Mixed-Width Handoff

Date: 2026-04-15

This document is the canonical handoff for the current PICF refactor.

It replaces:

- the earlier sidecar-primary lineage
- the earlier full-core-2048 handoff

The durable specification this file must match is:

- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)

## 1. Executive Summary

The current intended architecture is:

- one language-free physical core
- one full semantic-prefix-primary control path
- one full semantic-prefix-primary conditioned future path
- no task-anchor sidecar in the mainline
- no semantic width compression in the mainline
- mixed widths:
  - physical core width `512`
  - semantic trunk width `2048`

The semantic sequence is the primary control / conditioned-future input.
The physical posterior remains separate and language-free.
Physical world-state is appended after semantic tokens through explicit
up-projections into semantic-width trunks.

## 2. Design Goal

This refactor is meant to satisfy all of the following simultaneously:

- preserve the full PaliGemma image+text token sequence
- preserve the native PaliGemma token width in the semantic main path
- keep physical observation anchors language-free
- keep physical posterior language-free
- keep `physical_prediction_cache` language-free
- keep next-step innovation dependent only on
  `previous.predictive.physical_prediction_cache`
- avoid the memory blowup of making the entire physical core 2048-wide

## 3. Width Contract

Current defaults in [`config.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py):

- `hidden_dim = 512`
- `posterior_hidden_dim = 512`
- `innovation_dim = 512`
- `control_dim = 512`
- `future_hidden_dim = 512`
- `semantic_dim = 2048`
- `semantic_cross_dim = 2048`
- `attention_heads = 8`

Interpretation:

- physical token field, observation anchors, posterior, innovation latent,
  physical predictive basis, cache heads, and action head all remain in the
  512-wide physical state family
- semantic tokens remain in native 2048 width
- control and conditioned-future trunks run at semantic width `2048`
- physical-world signals are up-projected into those semantic-width trunks

## 4. Actual Code Path

### 4.1 PaliGemma semantic path

Files:

- [`wrapper.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/paligemma/wrapper.py)
- [`pi0.py`](/home/siyuanyue/Documents/openpi/src/openpi/models/pi0.py)
- [`pi0_fast_sonata.py`](/home/siyuanyue/Documents/openpi/src/openpi/models/pi0_fast_sonata.py)

Current behavior:

- `rgb_static` contributes image patch tokens
- `rgb_gripper` contributes image patch tokens when enabled
- prompt text contributes language tokens
- the full mixed image+text prefix is embedded and encoded by the PaliGemma /
  Gemma trunk
- the resulting semantic token sequence is exposed to PICF as
  `PaliGemmaSemanticFeatures.tokens`

PICF consumes those semantic tokens through:

- `_semantic_context(...)`
- `_project_semantic_context(...)`

Current mainline contract:

- `semantic_prefix_proj = Identity()`
- `semantic.tokens` and `semantic.prefix_tokens` are both 2048-wide
- there is no semantic down-projection in the mainline

### 4.2 Physical token field

File:

- [`pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)

Current physical token field lives at width `512`.

It is built from:

- point features
- visual map features
- tactile features
- proprio / previous-action / timing / contact context

Important invariant:

- semantic does not participate in token-field construction

### 4.3 Observation anchors

Current observation anchors:

- are seeded from point FPS
- read from the fused physical token field
- remain task-agnostic

Important invariant:

- semantic does not participate in observation-anchor selection or values

### 4.4 Physical posterior

Current posterior update:

- reads only carried physical prior state
- reads only current physical observation anchors
- produces:
  - `posterior.tokens`
  - `posterior.global_post`
  - recurrent physical anchor state

Important invariant:

- semantic does not participate in posterior update

### 4.5 Innovation

Current innovation path:

- reads only `previous.predictive.physical_prediction_cache`
- compares it against current real targets
- produces:
  - `innovation_token`
  - `innovation_norm`

Important invariant:

- semantic-conditioned future state is not allowed to feed next innovation

### 4.6 Control trunk

Current control trunk lives at semantic width `2048`.

The actual logical prefix is:

```text
[
  semantic_prefix_tokens,
  posterior_to_control(posterior.tokens),
  global_post_to_control(posterior.global_post),
  innovation_to_control(innovation_token),
  proprio_to_control(proprio_token),
  control_query_tokens,
]
```

Where:

- `semantic_prefix_tokens` are the full 2048-wide semantic sequence
- `posterior.tokens`, `posterior.global_post`, `innovation_token`, and
  `proprio_token` originate in the 512-wide physical state family
- those physical tensors are explicitly up-projected before entering the
  semantic-width control transformer

Control outputs:

- `control_tokens`: 2048-wide
- `control_query_state`: 2048-wide
- `pooled_state = control_state_proj(control_query_state)`: 512-wide
- `action = action_head(pooled_state)`: 7-wide normalized action

### 4.7 Physical predictive basis

The physical predictive basis remains purely physical:

```text
[
  posterior.tokens,
  posterior.global_post,
  proprio_token,
  action_cond_token,
]
-> predictive_world (512)
-> physical_pred_tokens
-> physical_global_pred
-> physical_prediction_cache
```

Everything above stays in the physical width family.

### 4.8 Conditioned future trunk

Current conditioned-future trunk lives at semantic width `2048`.

The actual logical prefix is:

```text
[
  semantic_prefix_tokens,
  physical_pred_to_conditioned(physical_pred_tokens),
  predictive_query_tokens,
]
```

Outputs:

- `predictive_query_state`: 2048-wide
- `global_pred = predictive_state_proj(predictive_query_state)`: 512-wide
- `prediction_cache = _prediction_cache_from_global(global_pred)`: physical
  cache heads at 512 width

So:

- semantic reasoning stays full-width
- physical future supervision heads remain in physical width

## 5. Why This Is Different From The Failed Full-Core-2048 Attempt

The failed full-core-2048 version widened:

- token fusion
- observation anchors
- posterior recurrent state
- physical predictive basis
- control trunk
- conditioned future trunk

all at once.

That was mathematically clean but too memory-heavy on `4 x A100 40GB`.

The current design keeps only the semantic main path at 2048 and widens the
physical core modestly from `384 -> 512`.

This preserves:

- full semantic sequence length
- full semantic width
- full semantic global context

without paying the cost of a fully 2048-wide physical world model.

## 6. PaliGemma Bandwidth

With current wrapper defaults:

- image size `224`
- patch size `14`
- `256` image tokens per image
- two images:
  - `rgb_static`
  - `rgb_gripper`

So before text tokens are counted:

- semantic prefix length already includes `512` image tokens

If prompt length is `L`, semantic prefix length is approximately:

- `512 + L`

Current mainline preserves:

- token count: `512 + L`
- token width: `2048`

What is appended after the semantic sequence:

- `16` persistent posterior tokens
- `1` `global_post`
- `1` innovation token
- `1` proprio token
- `1` control query token

So control sees:

- the full semantic prefix
- plus structured physical world-state context

## 7. Module Map

Main files and responsibilities:

- [`config.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py)
  physical-vs-semantic width defaults
- [`contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py)
  runtime state dataclasses
- [`pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  tokenization, posterior, innovation, control, physical predictive, conditioned future
- [`training.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py)
  action loss, physical auxiliary, semantic future auxiliary, alignment, budgeting
- [`picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)
  trainer, gradient clipping, logging, checkpointing, DDP
- [`picf_resume_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_resume_train.py)
  runtime-args restore and override path
- [`verify_picf_contract.py`](/home/siyuanyue/Documents/openpi/scripts/verify_picf_contract.py)
  static contract checks and targeted invariance regressions

## 8. Validation Ledger

Current local validation that must stay green:

- `src/openpi/picf/core/pipeline_test.py`
- `scripts/picf_core_train_test.py`
- `src/openpi/picf/core/training_test.py`
- `scripts/picf_resume_train_test.py`
- `scripts/serve_picf_policy_test.py`
- `src/openpi/picf/action_normalization_test.py`
- `src/openpi/picf/paligemma/wrapper_test.py`
- `src/openpi/picf/vjepa/wrapper_test.py`
- `scripts/verify_picf_contract.py`

For the mixed-width refactor, the required contract results are:

- semantic does not change physical posterior
- semantic does not change `physical_prediction_cache`
- previous semantic-conditioned future state does not change next innovation
- full semantic prefix directly enters control and conditioned future
- physical world-state is explicitly up-projected into semantic-width trunks
- `posterior.global_post` explicitly enters control

Latest local results:

- `pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py scripts/picf_core_train_test.py scripts/picf_resume_train_test.py scripts/serve_picf_policy_test.py src/openpi/picf/action_normalization_test.py src/openpi/picf/paligemma/wrapper_test.py src/openpi/picf/vjepa/wrapper_test.py`
  - `131 passed`
- `python scripts/verify_picf_contract.py`
  - static contract checks: `PASS`
  - targeted invariance regressions: `10 passed`
  - core regression suite: `115 passed`
  - smoke training check: `PASS`

## 9. Deployment Notes

When launching training:

- semantic must stay enabled
- full semantic prefix must stay primary
- no sidecar flags are used
- percentile grad clipping remains:
  - mode `percentile`
  - percentile `75`
  - window `100`

Cloud deployment should use a clean worktree and a clean training directory.

## 10. Current Status

This document is the active handoff for the mixed-width refactor.

Implementation status:

- local code path updated
- local regression complete
- cloud sync and clean training trial pending

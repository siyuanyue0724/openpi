# PICF v2.1

Date: 2026-04-17
Repo: `/home/siyuanyue/Documents/openpi`
Status: canonical deployed architecture record for the current live code

## 0. Document Map

The maintained PICF document set is intentionally limited to 3 persistent
documents:

1. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md`
2. `/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md`
3. `/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md`

`src/openpi/picf/README.md` is only a directory entry that points at these 3
files and is not treated as a fourth maintained design document.

This file and `PICF_FORMAL_CONTRACT.md` are expected to stay synchronized.
`README_v2.1.md` is the broader architecture and handoff document;
`PICF_FORMAL_CONTRACT.md` is the concise executable contract checked by tests.

## 0.1 Current Live Deployment

The current repository now deploys the primary `PI0.5 / PI0.5_Sonata + PICF`
hybrid described here.

Current live code restores these load-bearing pieces:

- semantic wrapper now restores `PaliGemmaWithExpertModel`
- semantic prompt tokenization now injects robot state into the PI0.5 prompt
  path
- trainer primary action loss now uses PI0.5-style flow matching
- serving primary action path now uses the PI0.5-style denoise sampler and
  immediately refreshes the predictive cache with the sampled action chunk
- replay / trainer now synthesize multi-step `action_chunk` windows from CALVIN
  `rel_actions`
- training-time PI0.5 flow supervision now recovers the denoised chunk estimate
  as `x_t - t * v_t` rather than misreading the predicted velocity as an
  action chunk
- posterior now re-reads native V-JEPA dense payload after anchor competition
- posterior now uses tactile group routing plus winner-read over full dense
  tactile tokens for active groups
- innovation now compares denser:
  - visual latent targets built from native V-JEPA payload probes
  - tactile targets built from dense tactile latent probes plus tactile map and
    tactile auxiliaries
  - point targets built from native point latent probes plus occupancy
- V-JEPA observation competition is now native-first
- AnyTouch public routing is now group-level multi-proposal routing with
  winner-read over full dense tactile group memory
- core no longer uses a direct trainable `7D` action head

There is still room for further training-driven refinement, but the primary
deployment task recorded in this document is now implemented:

- PI0.5 / PI0.5_Sonata action stack is restored as the primary action path
- external CALVIN `7D` action contract remains compatible
- PICF posterior / innovation semantics are preserved
- dense visual / tactile / point evidence now reaches posterior or innovation
  without the earlier harmful bottlenecks
- all of the above is done without adding a second recurrent physical state
  family

## 0.2 Deployment Tracker

Document / validation closure:

- maintained PICF documents reduced to the 3-file set above: done
- README entry reduced to a thin pointer only: done
- CALVIN validation README updated to point at `README_v2.1.md`: done
- formal contract explicitly scoped as live-code baseline: done
- `scripts/verify_picf_contract.py` updated to use the maintained 3-file set:
  done
- local baseline regression suite re-run after the document migration: done
- local replay / trainer / contract regression suite re-run after
  `action_chunk` replay synthesis fix: done
- PI0.5 flow-target recovery in training-time `predicted_action` /
  `predicted_chunk` fixed to use `x_t - t * v_t`: done
- `scripts/verify_picf_contract.py` now checks key live-code wiring facts for:
  - PaliGemma wrapper mode
  - semantic prompt-side state injection
  - native-first V-JEPA competition
  - native V-JEPA reread inside posterior
  - tactile group proposal routing
  - tactile group winner-read inside posterior
  - dense visual / tactile / point innovation targets
  - absence of the old direct `7D` action-head path inside core
  - serve-time predictive refresh after PI0.5 sampling

Follow-on work after deployment is limited to refinement and training
iteration, not unresolved architecture restoration.

## 1. Executive Summary

PICF v2.1 keeps the same outer control chain:

```text
observation
-> token_field
-> token_fusion
-> observation_anchors
-> posterior_update
-> innovation
-> predictive
-> control
```

but changes what is allowed to be compressed, and where.

The key rule is:

```text
posterior may stay compact
innovation input comparison does not stay overly compact
innovation output may stay compact
```

In other words:

- the recurrent physical state remains small and structured
- the residual comparison that constructs innovation stays close to
  backbone-native dense latents
- the final innovation interface to control can still be one innovation token

The action-side rule is:

```text
use the pi0.5 action generator as the primary action path
```

In other words:

- restore `PaliGemmaWithExpertModel`
- restore the Gemma action expert branch
- restore pi0.5-style flow matching training
- restore pi0.5-style denoising / sampling at inference
- treat PICF as the state / innovation / auxiliary-prediction extension around
  that action generator, not as a replacement for it

This document replaces the older split notes and is the only maintained PICF
README in this directory.

## 2. Non-Negotiable Invariants

### 2.1 Language-free physical posterior

The following remain language-free:

- current physical token field construction
- current observation-anchor construction
- current physical posterior update

Semantic tokens do not participate in current physical posterior update.

### 2.2 World-only innovation base

Next-step innovation must continue to read only:

- `previous.predictive.physical_prediction_cache`

It must not read:

- `previous.predictive.prediction_cache`
- semantic-conditioned future state
- any step-local private dense memory

### 2.3 One recurrent physical state family

The only recurrent physical state remains:

- `PicfPosteriorAnchorState`

No visual recurrent sidecar.
No tactile recurrent sidecar.
No second posterior family.

### 2.4 Step-local private dense memory

Private dense memory is allowed only inside the current step.

That includes:

- native V-JEPA dense visual payload
- tactile group private dense memory
- any future point-field private dense payload

These must not be stored in:

- `PicfTokenFieldState`
- `PicfCoreState`
- `previous`

## 3. Live Code Facts

### 3.1 PaliGemma

Live PICF preserves:

- full semantic prefix token sequence
- native semantic width `2048`

The semantic/action stack is now restored end-to-end:

- wrapper restores `PaliGemmaWithExpertModel`
- trainer uses PI0.5 flow matching
- serving uses the PI0.5 denoise sampler
- core no longer uses a direct trainable `7D` action head

### 3.2 V-JEPA

The V-JEPA wrapper exposes a native dense latent:

```text
tokens_thwc ~ [T, H, W, D_vjepa]
```

Projected visual tokens still exist as compatibility / auxiliary features:

```text
current_map
-> flatten
-> concat geometry features
-> visual_token_proj(hidden_dim)
```

But they are no longer the primary observation-competition path. The deployed
primary visual path is native-first:

- native visual payload is read before public fused competition
- posterior re-reads native V-JEPA payload after competition
- visual innovation targets are built from native V-JEPA latent probes

### 3.3 AnyTouch

The AnyTouch encoder returns full `sensor.tokens`.

The deployed public routing path is intentionally compressed and group-based:

```text
sensor.tokens
-> pooled_feature
-> base group token
-> multi-proposal group routing tokens
-> contact gate / hysteresis
-> only active group proposals enter fused competition
```

The deployed tactile path therefore has two levels:

- each active tactile group keeps full dense private memory
- each active tactile group emits multiple public routing proposals
- posterior winner-reads the full dense tactile tokens of the selected groups
- tactile innovation targets include dense latent probes in addition to tactile
  map and auxiliaries

### 3.4 Point / Sonata

The point path uses the point backbone output plus geometry /
projection features, and then projects that into the shared public physical
width.

The point innovation target now also includes latent probes, so the point
branch is no longer occupancy-only.

### 3.5 Innovation is world-only and branch-dense

Innovation still compares only against:

- `previous.predictive.physical_prediction_cache`

This preserves the world-only innovation contract.

The deployed target branches are no longer coarse summaries:

- `visual_latent`: dense visual latent probes read from native current-step
  V-JEPA payload
- `visual_real`: `4x4x3` RGB target
- `tactile_real`: dense tactile latent probes + `4x4` tactile map + tactile
  auxiliaries
- `point_real`: point latent probes + `4^3` occupancy target

So the live code implements the intended compact-posterior / dense-innovation
split across all enabled physical branches.

### 3.6 PI0.5 + PICF hybrid status

The deployed hybrid is:

```text
keep PI0.5 / PI0.5_Sonata as the primary action generator
+ add PICF posterior / innovation / world-only prediction cache / auxiliary heads
```

The current live PICF code is now there on the primary path.

The remaining distinctions versus historical PI0.5 / PI0.5_Sonata are
intentional implementation choices, not missing deployment:

- current primary trainer / serve path uses:
  - PI0.5 prefix path
  - Gemma action-expert suffix path
  - flow-matching objective
  - denoising sampler
- core now contains:
  - semantic prefix tokens
  - projected posterior/global_post/innovation/proprio
  - `control_world`
  - `control_query_state`
  - non-action physical conditioning tokens for the restored PI0.5 path

The codebase is no longer in a dual-action transitional state.

### 3.7 PI0.5 modules still missing from the current PICF primary path

The remaining repository-level distinctions versus historical PI0.5_Sonata are
now intentional, not accidental omissions:

- PICF injects richer physical conditioning tokens around the restored PI0.5
  action stack instead of reproducing PI0.5_Sonata's exact raw point-prefix
  insertion pattern
- native-first V-JEPA and tactile winner-read replace the older simpler
  multimodal prefix assumptions

### 3.8 Action-contract compatibility is now resolved for CALVIN deployment

Current PICF replay / training paths are built around:

- `action_horizon = 1`
- current-step `rel_actions`
- current-step `7D` action semantics

The PI0.5 family is built around:

- multi-step action chunks
- `action_dim = 32`
- much longer `action_horizon` (for example `50` in `Pi0Config`, and `32` in
  the fast family)
- flow-matching denoising over those action chunks

This must be interpreted carefully for CALVIN:

- the dataset-level action stream is still `rel_actions`
- raw CALVIN actions are still `7D`
- the existing OpenPI training stack already supports padding lower-dimensional
  state/action vectors up to the model action dimension through
  `PadStatesAndActions`

So restoring the PI0.5 action generator does **not** automatically mean the
external CALVIN action contract must stop being `7D`.

The mature recovery path is:

- preserve CALVIN raw `7D` action semantics externally
- restore PI0.5 flow-matching / denoising action generation internally
- use the same kind of action-dimension alignment strategy already present in
  the repo when needed

The current deployment resolves this by:

- preserving external CALVIN `7D` action semantics
- restoring PI0.5 flow-matching / denoising internally
- using internal action-dimension alignment and chunk synthesis where needed

So the architecture is now a practical drop-in `PI0.5 + PICF` system for the
CALVIN route.

For current CALVIN deployment planning, treat the following as fixed:

- external CALVIN action semantics remain `7D`
- the mature PI0.5 generation mechanism is restored around that contract
- action-space adaptation is an internal interface problem, not an excuse to
  change the external task contract

One useful current-code fact:

- `PicfObservation` already has `action_chunk`
- `_CalvinTransitionSource` has already been extended to support
  `action_horizon > 1`
- `scripts/picf_core_train.py` now exposes `--action-horizon`
- `CalvinSequentialReplay` now accepts `action_horizon` and emits
  `PicfObservation.action_chunk`
- both `_CalvinTransitionSource` and `CalvinSequentialReplay` synthesize
  `action_chunk` from future `rel_actions` windows; they do **not** require a
  raw `actions` key to exist in CALVIN episode NPZ files

So the dataset / observation contract is now aligned with the restored PI0.5
action generator and its training / inference path.

## 3.9 Latest local verification status

The following local checks were re-run on 2026-04-17 after synchronizing README
references, fixing replay-side `action_chunk` construction, and restoring the
PI0.5 expert/state-in-prompt path:

- `pytest -q src/openpi/picf/replay/calvin_replay_test.py -q`
- `pytest -q scripts/picf_core_train_test.py -q`
- `pytest -q src/openpi/picf/paligemma/wrapper_test.py -q src/openpi/picf/core/pipeline_test.py -q`
- `python scripts/verify_picf_contract.py --skip-smoke`

These are deployment-contract checks. They prove that:

- the current live code remains internally consistent after the PI0.5 expert
  restoration
- replay / trainer / contract docs are synchronized again
- the current live-code wiring facts for PaliGemma / V-JEPA / AnyTouch / action
  path are executable assertions inside `scripts/verify_picf_contract.py`
- the deployed architecture matches the documented primary action and evidence
  paths

## 4. Final v2.1 Principles

### 4.1 Posterior remains compact and structured

Posterior is still the compact recurrent world belief:

- finite anchor count
- finite latent width
- explicit geometry
- explicit uncertainty
- object/world-centric recurrence

Posterior is not a dense memory dump.

### 4.2 Innovation compare is much less compressed

Innovation compares:

- previous world-only predicted dense latent state
against
- current dense latent state

as close to backbone-native representations as practical.

This does **not** mean:

- sending every dense token directly into the control trunk
- making the recurrent posterior fully dense
- making all public trunks fully heterogeneous at once

It means:

- compare on dense/native latent space
- encode residuals branch-wise
- then compress those residuals into one innovation token

### 4.3 Innovation output may still be compact

The final action/control path may still consume:

- one `innovation_token`
- one `innovation_norm`

This is not the problem.
The problem is the current innovation comparison stage being too coarse.

### 4.4 Action generation aligns back to pi0.5

The deployed PICF action path is:

- pi0.5-style PaliGemma prefix path
- pi0.5-style Gemma action expert suffix path
- pi0.5-style flow-matching objective
- pi0.5-style iterative denoising sampler

PICF plugs into that stack by providing:

- richer posterior state
- richer innovation signal
- richer world-only prediction cache
- auxiliary prediction heads

It no longer replaces the stronger pi0.5 action generator with a simpler direct
linear head.

## 5. Final Per-Modality Policy

### 5.1 PaliGemma

Preserve:

- full prefix token sequence
- full semantic width `2048`

Usage:

- semantic-only trunks
- never current physical posterior update

Clarification:

- this is close to pi0.5 at the level of preserving full semantic tokens and
  semantic width
- it does **not** mean raw PaliGemma tokens are merged into the public physical
  token field
- it does **not** mean the whole action stack is byte-for-byte pi0.5

Actual intentional differences vs historical full pi0.5:

- PICF appends PICF-derived physical conditioning tokens around the restored
  PI0.5 action path
- PICF does not try to reproduce PI0.5_Sonata's exact raw point-prefix
  insertion pattern; it uses PICF physical conditioning and native-first
  evidence paths instead

Interpretation:

- semantic-token count and width were already preserved
- the expert/state/action stack around those tokens is restored at the wrapper
  + trainer/serve level
- further changes are refinements rather than missing action-stack restoration

Restored action-side items already live:

- `PaliGemmaWithExpertModel`
- `gemma_expert`
- `action_in_proj`, `action_out_proj`, and time-conditioning MLPs
- flow-matching target construction `x_t`, `u_t`
- denoising inference loop
- robot-state injection into the semantic/action stack

### 5.2 V-JEPA

#### Posterior-side policy

Preserve as natively as practical:

- native current-step dense visual map
- native V-JEPA payload width
- full current spatial token count

Keep:

- a step-local native visual payload path as the primary visual evidence path
- projected visual compatibility tensors only where auxiliary losses or
  diagnostics still need them

Clarification:

- native V-JEPA dense tokens are the primary visual evidence memory for
  observation-anchor competition and posterior reread
- `visual_token_proj(hidden_dim)` remains only as a compatibility /
  auxiliary-feature path and is not the main visual evidence path
- grid / camera / ray geometry is attached as side information and attention
  bias rather than replacing the native V-JEPA payload as the primary path
- the native V-JEPA dense payload is preserved privately for current-step
  anchor competition, posterior reread, and dense innovation comparison
- native V-JEPA dense tokens do **not** directly enter the same raw-token
  self-attention block as PaliGemma tokens

#### Innovation-side policy

Innovation no longer compares only:

- pooled latent
- `4x4` RGB summaries

It compares:

- previous world-only predicted dense visual latent field
against
- current native visual latent field

The residual of that dense comparison is then encoded into the innovation
branch.

### 5.3 AnyTouch

#### Posterior-side policy

The deployed tactile design is:

```text
group-level competition
-> winner-only dense reread
```

not:

```text
pooled tactile token only
-> generic anchors maybe notice it
```

The deployed tactile posterior path is:

- keep tactile group proposal tokens for competition
- keep full tactile group dense memory privately in the current step
- let tactile group proposals participate in the same observation-anchor
  competition workspace as the other physical evidence branches
- then run group-level ownership routing into posterior anchors
- allow only winner or top-2 owners to reread the full tactile dense group
  memory

Clarification:

- AnyTouch token length is preserved only in the private dense group memory, not
  in a generic dense fused token stream
- tactile groups still need a proposal representation for competition, but that
  proposal must carry group feature, position, rotation, normal/contact context,
  and ownership statistics rather than collapsing the whole modality to one
  generic token family
- full tactile dense tokens are not broadcast into generic token fusion
- instead, tactile group proposals compete in the physical anchor workspace so
  they can align with visual and point evidence, and only the winner or top-2
  posterior owners reread the full dense group memory
- tactile groups carry world-frame position and rotation information derived
  from `T_sens_to_wrist` and `G_t`

#### Innovation-side policy

Innovation no longer compares only:

- coarse tactile summaries alone

and instead compare:

- previous world-only predicted tactile group state
against
- current tactile group dense latent state

Current live code now does this:

- `tactile_real` now includes dense tactile latent probes
- plus coarse tactile map and tactile auxiliaries

The tactile innovation branch remains:

- world-only
- step-local on the current observation side
- compact only after branch-wise residual encoding

### 5.4 Sonata / Point

#### Posterior-side policy

v2.1 keeps the current public compressed point path:

- point backbone output
- geometry / color / projection features
- `point_token_proj(hidden_dim)`
- public token fusion / observation anchors / posterior

Clarification:

- point-token count is preserved at the point-backbone output level used by PICF
- point-token width is compressed into the shared public width `hidden_dim`
- this does not mean raw unordered point clouds are carried through the public
  trunk verbatim

#### Innovation-side policy

Innovation no longer compares only:

- low-resolution occupancy summaries

It compares:

- previous world-only predicted point-field latent
against
- current point backbone token field / point latent field

This does **not** mean raw unordered point clouds must enter innovation
verbatim.
In other words, innovation is now kept closer to the backbone-native point
latent representation rather than staying at a very coarse occupancy head.

### 5.5 Approximate Token Counts and Why They Matter

These counts matter because the v2.1 target is intentionally **not**:

- one giant raw-token self-attention block over every modality

Instead, native dense payloads remain accessible where they matter most:

- posterior evidence access
- innovation comparison

Approximate current modality scales from the audited code:

- V-JEPA 2.1:
  - full temporal grid:
    - `num_frames = 64`
    - `tubelet_size = 2`
    - `img_size = 384`
    - `patch_size = 16`
    - temporal tokens: `64 / 2 = 32`
    - spatial tokens per side: `384 / 16 = 24`
    - full dense token count: `32 * 24 * 24 = 18432`
  - current-step map only:
    - `24 * 24 = 576`
- AnyTouch:
  - CLIP-B/16 tactile sequence per sensor:
    - `1 CLS + 5 sensor tokens + 392 patch tokens = 398`
- Sonata / PI0.5_Sonata route:
  - configured point-token cap in the PI0.5_Sonata CALVIN config:
    - `1024`
- PaliGemma:
  - variable prefix token count
  - preserved at native semantic width

Interpretation:

- V-JEPA dense access is the largest current-step native payload
- AnyTouch dense access is moderate per sensor but high-value for local contact
  structure
- Sonata point tokens are already bounded strongly enough that count
  preservation is usually less dangerous than for V-JEPA

This is why v2.1 prioritizes:

- native-first V-JEPA access
- full-length private AnyTouch group memory
- compact recurrent posterior
- compact final innovation/control interfaces

## 6. Final Dataflow

### 6.1 Current-step observation path

```text
rgb / depth / tactile / proprio / prompt
-> point backbone / V-JEPA / AnyTouch / PaliGemma
-> physical competition memories + semantic prefix trunk
-> observation-anchor competition workspace
-> posterior_update
```

Additional v2.1 step-local private memory:

- native visual dense latent
- tactile group dense latent memory

These are consumed only inside current-step anchor competition, current-step
posterior update, and current-step innovation target construction.

Concrete split:

```text
PaliGemma image+text
-> full semantic prefix tokens [L_pg, 2048]
-> semantic-only trunks

Sonata point backbone output
-> point competition memory [N_pt, hidden_dim]

V-JEPA current-step dense map
-> native visual competition memory [N_v, D_vjepa_native]

AnyTouch active tactile groups
-> group proposal memory [N_tg, D_tactile_prop]
+ private dense tactile group memory [N_tg, L_group, D_tactile_native]

context / proprio-derived context
-> context competition memory [N_ctx, hidden_dim]

observation-anchor queries [K_obs, hidden_dim]
-> read point competition memory
-> read native visual competition memory
-> read tactile group proposal memory
-> read context competition memory
-> fuse branch reads
-> observation_anchors [K_obs, hidden_dim]
-> posterior_update
```

So v2.1 does **not** do:

- one giant raw-token self-attention block over PaliGemma tokens plus native
  V-JEPA tokens plus full dense tactile tokens

It does:

- separate semantic raw-token trunks
- a physical anchor-competition workspace over multimodal physical evidence
- separate step-local private reread paths for native dense payloads

### 6.2 Posterior path

Posterior keeps its current mathematical role:

```text
prior + current physical evidence -> posterior
```

with improvements:

- visual native reread
- tactile group ownership and winner-read

Anchor competition semantics:

- observation anchors compete over the physical competition memories
- they do not literally partition all raw native tokens in the system
- PaliGemma raw tokens do not participate in observation-anchor competition
- V-JEPA native visual tokens do participate as a visual evidence branch
- tactile group proposals do participate as a tactile evidence branch
- tactile full dense group memory is then accessed by winner-only reread after
  competition / ownership is established

Posterior still outputs:

- compact anchor state
- compact posterior tokens
- compact global posterior summary

### 6.3 Physical predictive path

The physical predictive basis remains world-only:

```text
posterior.tokens
+ posterior.global_post
+ proprio
+ action_cond
-> predictive_world
-> physical_pred_tokens
-> physical_global_pred
-> physical_prediction_cache
```

This is the only prediction cache allowed to feed next-step innovation.

### 6.4 Innovation path

The final intended innovation construction is:

```text
previous physical_prediction_cache
vs
current dense/native latent targets
-> branch-wise dense residual encoders
-> fused innovation latent
-> innovation_token
```

Innovation remains:

- explicit
- world-only
- action-consumable

and is no longer restricted to very coarse summary targets.

### 6.5 Action path

The deployed action path is:

```text
PaliGemma image tokens
+ PaliGemma language tokens
+ semantic/state injection
+ PICF physical conditioning tokens
-> pi0.5 prefix embeddings

noisy action suffix + timestep conditioning + state-conditioned modulation
-> Gemma action expert suffix embeddings

prefix + suffix
-> PaliGemmaWithExpertModel
-> suffix outputs
-> action_out_proj
-> flow velocity prediction v_t
```

Training:

```text
actions + noise + sampled t
-> x_t = t * noise + (1 - t) * actions
-> u_t = noise - actions
-> model predicts v_t
-> flow-matching loss on v_t vs u_t
```

Inference:

```text
noise initialization
-> repeated denoise_step(...)
-> Euler updates
-> final action chunk
```

PICF is inserted around this by providing:

- posterior state
- innovation token
- world-only prediction cache
- auxiliary prediction heads
- multimodal physical evidence shaping

Robot state is injected in both places:

- continuous physical/control/predictive paths
- semantic/action path, either as pi0.5-style serialized state in the prompt or
  as dedicated state tokens in the semantic trunk

## 7. Module-by-Module Live Wiring

### 7.1 `_visual_map()`

- returns the current-step visual map used to construct both:
  - projected compatibility visual tokens
  - step-local native visual dense payload

### 7.2 `_build_token_field()`

- builds public physical competition tokens
- also builds step-local dense memory carriers for:
  - native visual payload
  - dense tactile group memory
  - point payload

### 7.3 `_build_observation_anchors()`

- remains physically task-agnostic
- remains language-free
- performs native-first visual reread before public fused-token competition
- sees tactile through public group proposal tokens, not direct full tactile
  dense memory

### 7.4 `_posterior_update()`

- takes prior + observation anchors as the main recurrent interface
- adds visual native reread
- adds tactile group competition and winner-only tactile reread

### 7.5 `_innovation()`

- compares previous prediction against denser backbone-native branch targets
  for all enabled modalities:
  - visual latent probes + RGB target
  - tactile latent probes + tactile map + auxiliaries
  - point latent probes + occupancy
- still outputs one compact innovation token

### 7.6 `_predictive_state()`

- builds the world-only physical predictive basis
- builds the semantic-conditioned future branch separately
- expands the physical prediction cache so innovation can compare richer latent
  state on the next step

### 7.7 Action-generation module

- keeps `control_world`, `control_query_state`, and `control_state_proj` as
  PICF physical-conditioning modules
- uses the PI0.5 action generator as the primary path
- wires PICF state into the PI0.5 action stack
- does not use a direct trainable PICF action head in core

## 8. Implemented Structural Layout

### `src/openpi/picf/core/config.py`

Config now exposes fields for:

- native visual reread
- dense visual innovation branch
- tactile group routing
- tactile local encoding
- dense tactile innovation branch
- denser point innovation branch

### `src/openpi/picf/core/contracts.py`

Returned public state still excludes private dense memory.

Allow richer world-only physical prediction cache definitions so innovation can
compare denser latent state on the next step.

### `src/openpi/picf/core/pipeline.py`

The pipeline now includes:

- step-local visual dense memory carrier
- step-local tactile group carrier
- visual dense reread
- tactile group competition
- tactile winner-only dense reread
- dense branch innovation comparison

It still does not add:

- second recurrent physical state
- semantic writeback into current posterior

### `src/openpi/picf/paligemma/wrapper.py`

The semantic/action path now restores:

- restore `PaliGemmaWithExpertModel`
- restore Gemma action expert loading
- restore state-aware semantic/action input path

It does not rely on:

- prefix-only semantic extraction as the final action design
- silent omission of the expert branch

### `src/openpi/picf/core/training.py`

The primary action objective is:

- pi0.5-style flow matching

It keeps:

- innovation world-only contract
- auxiliary prediction losses around the physical world model

### `scripts/picf_core_train.py`

The trainer CLI exposes the needed config fields and now uses:

- action objective semantics back to pi0.5-style flow matching
- action-side model construction to restore the expert branch

It does not change:

- innovation world-only contract
- semantic isolation of current physical posterior

## 9. Deployed Regression Requirements

### 9.1 Posterior invariants

Same observation, different semantic:

- current posterior unchanged
- world-only physical prediction cache unchanged
- next innovation unchanged

### 9.2 Visual native reread liveness

If native visual payload changes while public visual summary is effectively
held fixed:

- posterior must change

### 9.3 Tactile ownership concentration

For one tactile group:

- winner or top-2 ownership must dominate
- dense tactile evidence must not diffuse uniformly

### 9.4 Tactile winner-read liveness

If private tactile dense memory changes while the public tactile proposal stays
approximately fixed:

- posterior must change

### 9.5 Dense innovation liveness

If dense latent state changes while coarse summary targets are held nearly
fixed:

- innovation must still change

### 9.6 Returned-state compaction

Returned `PicfCoreState` must not contain:

- native visual payload
- tactile private dense memory
- any other private dense cache

### 9.7 Flow-matching action regression

The restored primary action path must verify:

- `x_t = t * noise + (1 - t) * actions`
- `u_t = noise - actions`
- model predicts `v_t`
- training loss is applied on `v_t` vs `u_t`

### 9.8 Denoising sampler regression

Inference must verify:

- prefix cache creation from the PaliGemma branch
- suffix denoising through the Gemma expert branch
- iterative Euler update loop
- final action chunk shape and dtype contracts

## 10. Final Answers

### Q1. Should innovation continue using heavily compressed targets?

No.

The current coarse innovation path is mathematically valid but too weak if the
goal is to preserve fine world detail that a compact posterior cannot hold.

### Q2. Should innovation therefore use raw sensor data directly inside the
control trunk?

No.

The correct move is:

- compare on dense/native latent space
- encode branch-wise residuals
- then compress to one innovation token

### Q3. Does this break the current posterior / innovation contract?

No.

It preserves:

- language-free current posterior
- world-only next-step innovation base
- one recurrent physical state family

### Q3b. Was it more reasonable to start from pi0.5 and attach PICF onto it than
to keep a PICF-local direct action head?

Yes. That is now the deployed direction.

That is the preferred direction because:

- pi0.5 already contains the stronger action-generation design
- a PICF-local direct action head would be architecturally simpler and therefore
  a regression in action modeling capacity
- PICF's real value is world-state tracking, innovation, and auxiliary
  prediction, not replacing the proven pi0.5 action generator with a weaker one

### Q4. Does tactile dense memory become recurrent?

No.

It stays step-local only.

### Q5. Do raw PaliGemma tokens self-attend together with raw V-JEPA and raw
AnyTouch dense tokens?

No.

v2.1 keeps these interfaces separate:

- PaliGemma raw tokens stay in semantic-only trunks
- physical competition memories feed observation-anchor competition
- native V-JEPA dense payload and tactile dense groups stay step-local and are
  accessed through branch reads and reread paths

The meeting point is later and narrower:

- full semantic tokens
- plus up-projected compact physical state such as posterior tokens,
  global posterior summary, innovation token, and proprio token

### Q6. Does AnyTouch keep token length unchanged?

Publicly, no.

Privately, yes.

More precisely:

- the public tactile path is still width-compressed and proposal-like
- the full tactile token length is preserved in private step-local group memory
- ownership routing ensures each active tactile group is assigned to winner or
  top-2 posterior anchors before full dense reread

### Q7. Where does robot state enter now, and is omitting it from PaliGemma
lossless?

Current PICF injects robot state as continuous proprio in several non-text
places:

- local-frame construction / world transforms
- context tokens
- recurrent prior update
- control trunk
- physical predictive trunk

This is strong for metric geometry and dynamics, but it is **not** lossless with
respect to semantic conditioning if the semantic prefix path omitted state.

The live code now injects state into both:

- continuous physical/control/predictive paths
- the semantic PI0.5 prompt path

So the current regime should be understood as:

- stronger continuous physical-state injection
- restored direct semantic-state injection

## 11. Deployed Implementation Summary

PICF v2.1 means:

```text
same PICF outer chain
+ full PaliGemma semantic trunk
+ restored pi0.5 action expert and denoising stack
+ native V-JEPA dense access for posterior and denser visual innovation
+ tactile group competition + winner-only dense reread for posterior
+ denser tactile innovation comparison
+ Sonata remains on compressed public posterior path
+ denser point innovation comparison
+ one recurrent physical state family
+ one world-only innovation token output
```

This is the cleanest version because it keeps the recurrent world model
structured, restores the stronger pi0.5 action generator, and removes the most
harmful early compression from both posterior ingestion and innovation
construction.

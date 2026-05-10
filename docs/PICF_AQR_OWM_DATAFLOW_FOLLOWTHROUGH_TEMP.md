# PICF-AQR-OWM Dataflow Follow-Through Temp

Source contract: `docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md`

Purpose: recursive dataflow audit for the deployed PICF-AQR-OWM graph. This file tracks every major tensor path from observation to action, including the mathematical contract, current code owner, causality rule, and verification status.

Status: temporary deployment/audit document.

## 0. Global Invariant

The system is a belief-state controller, not a cache-augmented feed-forward router:

```text
o_t, a_{t-1}, b_{t-1}
  -> prior b_t^-
  -> typed measurement routing
  -> posterior correction b_t^+
  -> world prediction \hat b_{t+1}
  -> innovation-aware PI0.5 action conditioning
```

The authoritative state is always the corrected posterior:

```math
b_t^+ = \operatorname{PosteriorCorrection}(b_t^-, M_t, I_t)
```

Cache and slot-JEPA predictions are auxiliary evidence/targets. They cannot replace `b_t^+`.

## 1. Observation To Typed Evidence

### 1.1 Input Observation

Code:

```text
src/openpi/picf/core/pipeline.py::observe_step
src/openpi/picf/core/pipeline.py::_observe_impl
```

Inputs:

```text
RGB/static image
depth/point features
tactile packet
robot proprio
prompt / PaliGemma semantic override
previous recurrent state
previous action
```

Mathematical object:

```math
o_t = (I_t, D_t, T_t, q_t, \ell, a_{t-1})
```

Status: deployed.

### 1.2 V-JEPA Current And Temporal Maps

Code:

```text
src/openpi/picf/vjepa/wrapper.py::VjepaFeatureMap.current_map
src/openpi/picf/vjepa/wrapper.py::VjepaFeatureMap.recent_maps
src/openpi/picf/core/pipeline.py::_visual_maps
```

Dataflow:

```text
V-JEPA clip latent
  -> current 2D map for legacy visual path
  -> recent temporal maps for OWM typed support
```

Formula:

```math
Z_t^{vjepa} = \{z_{\tau,h,w}\}_{\tau=t-T+1}^{t}
```

Production path:

```math
M_{vjepa,t} = \{z_{\tau,h,w}: \tau \in \mathcal T_{recent}\}
```

Mean-only path is ablation:

```math
\bar z_t = \frac{1}{2}(z_t + z_{t-1})
```

The production path preserves time before AQR:

```math
p_{j}^{vjepa}(\tau,h,w)
=
\operatorname{softmax}_{\tau,h,w}(q_j^\top k_{\tau,h,w})
```

Status: deployed. `recent_maps(n)` preserves temporal slices; `last_two_mean` is not the default production support.

### 1.3 PaliGemma Text/Image Typed Evidence

Code:

```text
src/openpi/picf/core/pipeline.py::_semantic_context_from_override
src/openpi/picf/core/pipeline.py::_aqr_pg_image_support_read
```

Dataflow:

```text
PaliGemma text tokens -> task semantic conditioning
PaliGemma image token ranges/views -> first-class PG image support branch
```

Formula:

```math
M_{pg,img,t} = \{e_{v,u}^{pg}\}
```

For each view/range:

```math
\ell_{j,v,u}^{pg}
=
\frac{(W_q q_j)^\top (W_k e_{v,u}^{pg})}{\sqrt d}
```

```math
p_{j,v,u}^{pg}
=
\operatorname{softmax}_{v,u}(\ell_{j,v,u}^{pg})
```

Status: deployed. AQR returns `graph.pg_priors`; PG image support may bias visual support but is no longer destroyed into visual bias only.

### 1.4 Point/Tactile/Posterior/Cache Typed Evidence

Code:

```text
src/openpi/picf/core/pipeline.py::_build_token_field
src/openpi/picf/core/pipeline.py::_build_aqr_anchor_graph
src/openpi/picf/core/pipeline.py::_previous_evidence_cache_tokens
```

Memory set:

```math
M_t =
\{M_{text}, M_{pg}, M_{vjepa}, M_{point}, M_{tactile}, M_{post}, M_{cache}\}
```

Missing modalities are represented by empty tokens or masks:

```math
M_t^{(m)} = \emptyset \Rightarrow \gamma_{j,m}=0
```

Status: deployed.

## 2. Typed Token Field

Code:

```text
src/openpi/picf/core/contracts.py::PicfTokenFieldState
src/openpi/picf/core/contracts.py::PicfTemporalVisualSupportState
src/openpi/picf/core/pipeline.py::_build_token_field
```

OWM token field:

```text
visual_tokens
temporal_visual.tokens
temporal_visual.time_ids
temporal_visual.grid_index
point_tokens
tactile_group_tokens
semantic tokens
posterior tokens
cache tokens
```

Formula:

```math
e_{i}^{(m)}
=
W_m z_i^{(m)}
+ E_m
+ E_{view(i)}
+ E_{valid(i)}
+ PE_m(\xi_i)
```

Constraint:

```text
RoPE/coordinate encodings describe time/space/view.
Slot identity is address/binding, not coordinate distance.
```

Status: deployed for temporal support and typed AQR consumers.

## 3. AQR Measurement Routing

Code:

```text
src/openpi/picf/core/pipeline.py::_build_aqr_anchor_graph
src/openpi/picf/core/contracts.py::PicfAnchorPriorGraphState
```

Inputs:

```text
physical queries
task queries
typed memories
posterior summary
previous evidence cache
prompt
proprio
```

Query split:

```math
S_t^{phys} = \{s_{t,j}\}_{j=1}^{K_p}
```

```math
Q_t^{task} = \{q_{t,k}\}_{k=1}^{K_t}
```

Routing formula:

```math
\ell_{j,i}^{(m)}
=
\frac{
  (W_q [a_j, c^-_{t,j}, r_j])^\top
  W_k e_{t,i}^{(m)}
}{\sqrt d}
+ b_{geom}
+ b_{role}
+ b_{valid}
```

```math
p_{j,i}^{(m)}
=
\operatorname{softmax}_i(\ell_{j,i}^{(m)}/\tau_m)
```

Competition:

```math
P^{(m)} = \operatorname{SinkhornLike}(\ell^{(m)})
```

Deployed graph fields:

```text
visual_priors
vjepa_temporal_priors
pg_priors
point_priors
tactile_priors
posterior_priors
cache_priors
slot_address
slot_content
support_uncertainty
modality_confidence
```

Status: deployed.

## 4. Prior And Posterior Correction

### 4.1 Prior

Code:

```text
src/openpi/picf/core/pipeline.py::_current_prior
```

Formula:

```math
b_t^- = f_\theta(b_{t-1}^+, q_t, a_{t-1})
```

Slot state:

```math
S_{t,j}^- = (a_{j}, c_{t,j}^-, \mu_{t,j}^-, \Sigma_{t,j}^-, \alpha_{t,j}^-, r_j)
```

Status: existing PICF posterior loop preserved.

### 4.2 Assignment / Binding

Code:

```text
src/openpi/picf/core/pipeline.py::_posterior_update
```

Binding:

```math
B_{j,k}
=
\operatorname{SinkhornDustbin}
(
score(S_{t,j}^-, A_{t,k}^{obs})
)
```

Compatibility terms:

```text
address / slot compatibility
geometry compatibility
role compatibility
current evidence similarity
innovation/recycle gate
```

Status: deployed with posterior binding preserved; address/content fields are exposed but binding remains correction-aware.

### 4.3 Precision-Like Correction

Code:

```text
src/openpi/picf/core/pipeline.py::_posterior_update
```

Natural parameter form:

```math
\Lambda_t^+ = \Lambda_t^- + \Lambda_t^{meas}
```

```math
\eta_t^+ = \Lambda_t^- \mu_t^- + \Lambda_t^{meas}\mu_t^{meas}
```

```math
\mu_t^+ = (\Lambda_t^+)^{-1}\eta_t^+
```

Content gate:

```math
c_t^+ = c_t^- + K_t \odot (\tilde c_t^{meas} - c_t^-)
```

Gate inputs:

```text
measurement confidence
support entropy
innovation norm
modality validity
posterior uncertainty
```

Status: existing PICF correction preserved and extended with OWM fields.

## 5. Evidence Cache

Code:

```text
src/openpi/picf/core/contracts.py::PicfEvidenceCacheState
src/openpi/picf/core/pipeline.py::_empty_evidence_cache
src/openpi/picf/core/pipeline.py::_previous_evidence_cache_tokens
src/openpi/picf/core/pipeline.py::_write_evidence_cache
```

State:

```text
tokens: [H, K, D]
address: [H, K, D]
age: [H, K]
source: [H, K]
valid: [H, K]
uncertainty: [H, K]
innovation: [H, K]
write_index
```

Causal order:

```text
read previous cache
current evidence -> posterior correction
corrected posterior -> write cache for next step
```

Formula:

```math
\tilde M_{cache,t}
=
g_{cache}(q_j, age, uncertainty, source, \nu_t)
\operatorname{Attn}(q_j, C_{t-H:t-1})
```

Innovation downweight:

```math
g_{cache} \downarrow \quad \text{as} \quad \|\nu_t\| \uparrow
```

Status: deployed. Same-step cache write is not read by current action.

## 6. World Prediction And Innovation

### 6.1 World-Only Predictive Basis

Code:

```text
src/openpi/picf/core/pipeline.py::_build_physical_predictive_basis
```

Formula:

```math
\hat P_{t+1}
=
F_\theta(b_t^+, q_t, a_t)
```

The world-only cache predicts:

```text
visual_latent
visual_real
tactile_real
point_real
```

Status: existing PICF prediction path preserved.

### 6.2 Innovation

Code:

```text
src/openpi/picf/core/pipeline.py::_innovation
```

Formula:

```math
\nu_t
=
\Sigma_{pred}^{-1/2}(y_t - \hat y_t)
```

Rule:

```text
innovation compares current real targets only against previous world-only prediction.
It must not compare against semantic-conditioned future readout.
```

Status: deployed.

### 6.3 Slot-Level JEPA

Code:

```text
src/openpi/picf/core/pipeline.py::_predictive_state
src/openpi/picf/core/training.py::future_targets_from_current_targets
src/openpi/picf/core/training.py::compute_transition_loss
scripts/picf_core_train.py::_future_targets_override_from_observed
```

Prediction:

```math
\hat S_{t+1,j}
=
F_\theta(S_{t,j}^+, a_t, q_t, \ell)
```

Primary target:

```math
S_{t+1,j}^{target}
=
\operatorname{stopgrad}(S_{t+1,j}^{posterior})
```

Loss:

```math
L_{slot-jepa}
=
\frac{1}{K}\sum_j
\|\hat c_{t+1,j} - stopgrad(c_{t+1,j}^{posterior})\|_2^2
```

No leakage rule:

```text
next posterior is used only by transition loss, not by current AQR/posterior/action input.
```

Fallback:

```text
If a standalone transition loss has no next posterior override, detached future visual latent summary is used only as compatibility target.
```

Status: deployed.

### 6.4 Support Prediction

Code:

```text
src/openpi/picf/core/pipeline.py::slot_support_pred_head
src/openpi/picf/core/training.py::_posterior_support_summary
```

Target:

```math
y_{t+1,j}^{support}
=
stopgrad([\alpha, support\_mass, contact\_prob, binding\_confidence]_{t+1,j})
```

Loss:

```math
L_{support}
=
\frac{1}{K}\sum_j
\|\hat y_{t+1,j}^{support} - y_{t+1,j}^{support}\|_2^2
```

Status: deployed.

## 7. Task Readout And Ordinal Relation

Code:

```text
src/openpi/picf/core/pipeline.py::_ordinal_relation_state
src/openpi/picf/core/pipeline.py::_build_task_readout
```

Relation prompt detector:

```text
first / second / third / fourth
left / right / front / back
nearest / farthest
```

Axis score:

```math
s_j = u_\ell^\top \mu_{t,j}
```

Soft rank:

```math
rank_j
=
1 + \sum_{l \ne j}
\sigma((s_l - s_j)/\tau_{rank})
```

Loss is gated:

```math
L_{ordinal}=0
\quad \text{if prompt has no high-confidence ordinal/relation signal}
```

Constraint:

```text
ordinal head can select/read physical slots but cannot overwrite posterior address/content identity.
```

Status: deployed.

## 8. PI0.5 Action Path

Code:

```text
src/openpi/picf/core/pipeline.py::_build_conditioned_control_state
src/openpi/picf/core/pipeline.py::finalize_with_action
src/openpi/picf/policy.py::forward_train_transition
```

Action conditioning:

```math
a_t \sim \pi_{\theta}^{PI0.5}
(
b_t^+,
Q_t^{task},
\nu_t,
M_t^{typed}
)
```

Rule:

```text
PI0.5 remains final action generator.
OWM provides structured belief/action prefix, not a replacement action head.
```

Status: deployed.

## 9. Loss Family

Code:

```text
src/openpi/picf/core/training.py::PicfTransitionLossConfig
src/openpi/picf/core/training.py::PicfTransitionLossBreakdown
src/openpi/picf/core/training.py::compute_transition_loss
scripts/picf_core_train.py::_build_loss_config
```

Total:

```math
L =
L_{action}
+ \lambda_{slot}L_{slot-jepa}
+ \lambda_{support}L_{support}
+ \lambda_{bind}L_{binding}
+ \lambda_{xmod}L_{xmod}
+ \lambda_{ordinal}L_{ordinal}
+ \lambda_{innov}L_{innovation}
+ L_{existing}
```

Binding consistency:

```math
L_{binding}
=
\frac{1}{K}\sum_j
\frac{H(B_j)}{\log N}
```

Cross-modal pooled alignment:

```math
L_{xmod}
=
\frac{1}{|\mathcal M|}
\sum_{m \in \mathcal M}
\|\gamma_m-\gamma_{visual}\|_1
```

Innovation calibration:

```math
L_{innovation}
=
\frac{\sum_m mask_m(\nu_m-1)^2}{\sum_m mask_m}
```

Status: deployed with guarded weights. All loss terms are finite/zero when branches are missing.

## 10. Recursive Causality Check

Valid recursion:

```text
previous posterior/cache
  -> current AQR read
  -> current posterior correction
  -> current action conditioning
  -> prediction/cache write for next step
  -> next-step loss target only
```

Invalid recursion:

```text
future target -> current AQR
future target -> current posterior
future target -> current action
same-step cache write -> current action truth
raw cache -> action truth bypassing posterior
```

Current code status: valid recursion is implemented.

## 11. Tests Mapped To Dataflow

```text
V-JEPA recent_maps temporal preservation:
  src/openpi/picf/vjepa/wrapper_test.py

AQR temporal/PG/address fields:
  src/openpi/picf/core/pipeline_test.py

Evidence cache previous-read/post-correction-write:
  src/openpi/picf/core/pipeline_test.py

Ordinal relation prompt gate and no posterior rewrite:
  src/openpi/picf/core/pipeline_test.py

OWM guarded finite losses:
  src/openpi/picf/core/training_test.py

CLI/config/loss metric exposure:
  scripts/picf_core_train_test.py

Strict README-to-code verifier:
  scripts/verify_picf_owm_contract_test.py

OWM evidence bundle and verifier snapshot:
  scripts/picf_owm_evidence_bundle_test.py
```

## 12. Follow-Through Verdict

The dataflow now follows the README architecture:

```text
typed evidence
  -> AQR support routing
  -> posterior prior/binding/correction
  -> world prediction / slot-JEPA target loss
  -> innovation-gated cache trust
  -> PI0.5 action conditioning
```

Remaining requirement before calling any run scientifically validated:

```text
run training diagnostics on real CALVIN/robot rollouts and inspect temporal support,
PG support, identity switch, cache trust, and innovation plots.
```

This is an empirical validation requirement, not a missing code path in the deployed graph.

## 13. Scripted Audit Closure

The stricter scripted audit is now part of this dataflow document:

```text
python scripts/verify_picf_owm_contract.py
```

The verifier follows the same recursive chain:

```text
README invariant
  -> dataclass/state contract
  -> V-JEPA/PG typed evidence path
  -> AQR graph fields
  -> posterior/cache causality
  -> future-target-only training losses
  -> trainer metric propagation
  -> evidence-bundle handoff
```

Required failure interpretation:

```text
If the verifier fails, the branch is not final OWM even if unit tests pass.
If the verifier passes but rollout diagnostics fail, the code path exists but
the trained system is not empirically accepted.
```

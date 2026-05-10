# PICF-AQR-OWM Theory Method Temp

Source contract: `docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md`

Purpose: theoretical method statement for the deployed PICF-AQR-OWM architecture. This document explains why the modules belong to one mathematical system instead of being loosely attached patches.

Status: temporary deployment/audit document.

## 1. Problem Statement

Robot manipulation is a partially observable control problem:

```math
s_t =
(objects_t,\ robot_t,\ contact_t,\ task_t)
```

```math
o_t =
(RGB_t,\ video_{\le t},\ depth_t,\ tactile_t,\ proprio_t,\ language)
```

The policy should not act on a single holistic frame embedding. It should act on a corrected belief:

```math
b_t(s)
=
p(s_t=s \mid o_{\le t}, a_{<t}, \ell)
```

The canonical update is:

```math
b_t(s)
\propto
p(o_t \mid s)
\int p(s \mid s', a_{t-1})b_{t-1}(s')ds'
```

PICF-AQR-OWM implements this as a neural belief-state system.

## 2. State Factorization

Each physical slot is:

```math
S_{t,j}
=
(a_j,\ c_{t,j},\ \mu_{t,j},\ \Sigma_{t,j},\ \alpha_{t,j},\ r_j)
```

Where:

```text
a_j:
  persistent address / identity carrier

c_{t,j}:
  time-varying content

mu/Sigma:
  metric geometric belief

alpha:
  visibility/existence confidence

r_j:
  role/type/task class
```

The address/content split prevents identity from being overwritten by transient evidence:

```text
address says which persistent slot this is
content says what evidence currently says about it
```

Address alone is not identity. Identity is:

```math
identity_j =
(a_j,\ binding\ history,\ role_j,\ posterior\ carry,\ recycle/existence)
```

## 3. Typed Evidence Principle

Evidence is not fused into one opaque latent before routing. It remains typed:

```math
M_t =
\{M_{text},M_{pg},M_{vjepa},M_{point},M_{tactile},M_{proprio},M_{post},M_{cache}\}
```

For each modality:

```math
e_{i}^{(m)}
=
W_mz_i^{(m)}
+E_m
+E_{view(i)}
+E_{valid(i)}
+PE_m(\xi_i)
```

Coordinate encodings describe coordinates and time. They are not slot identity.

## 4. AQR As Measurement Router

AQR is the measurement model:

```math
p(o_t \mid S_{t,j})
```

For each slot and modality:

```math
\ell_{j,i}^{(m)}
=
\frac{(W_q[a_j,c^-_{t,j},r_j])^\top W_ke_{t,i}^{(m)}}{\sqrt d}
+ b_{geom}
+ b_{role}
+ b_{valid}
```

```math
p_{j,i}^{(m)}
=
\operatorname{softmax}_i(\ell_{j,i}^{(m)}/\tau_m)
```

AQR does not replace posterior. It provides typed measurements to posterior.

## 5. Posterior Is The Belief Authority

The prior is:

```math
S_{t,j}^{-}
=
F_\theta(S_{t-1,j}^{+}, a_{t-1}, proprio_t)
```

Measurement produces:

```math
\tilde S_{t,j}^{meas}
=
R_\theta(S_{t,j}^{-}, M_t)
```

Correction uses a precision-like update:

```math
\Lambda_t^+ = \Lambda_t^- + \Lambda_t^{meas}
```

```math
\eta_t^+ = \Lambda_t^-\mu_t^- + \Lambda_t^{meas}\mu_t^{meas}
```

```math
\mu_t^+ = (\Lambda_t^+)^{-1}\eta_t^+
```

Content update:

```math
c_t^+
=
c_t^-
+K_t\odot(\tilde c_t^{meas}-c_t^-)
```

This is the core PICF invariant:

```text
the current physical truth used by action is posterior after correction
```

## 6. V-JEPA Temporal Evidence

The final temporal design is:

```math
M_{vjepa,t}
=
\{z_{\tau,h,w}\mid \tau\in\mathcal T_{recent}\}
```

Not:

```math
\bar z_t = \frac{1}{2}(z_t + z_{t-1})
```

Mean can reduce static noise, but it can smear motion/contact timing:

```math
E[\bar z_t-h(s_t)]
\approx
\frac{1}{2}(h(s_{t-1})-h(s_t))
```

Therefore production AQR reads:

```math
p_j^{vjepa}(\tau,h,w)
```

and can decide whether older, newer, delta, or static evidence is most relevant.

## 7. PaliGemma Image Evidence

PaliGemma text gives semantic referring:

```math
M_{text} = \{e_l^{text}\}
```

PaliGemma image tokens give open-world image-language support:

```math
M_{pg,img} = \{e_{v,u}^{pg}\}
```

The final design preserves:

```math
p_{j,v,u}^{pg}
```

It may bias V-JEPA or task anchors, but it is not erased into a V-JEPA heatmap only.

## 8. Evidence Cache Theory

Cache is historical evidence:

```math
C_{t-H:t-1}
=
\{(token,address,age,source,uncertainty,innovation,valid)\}
```

Cache read:

```math
r_{cache,j}
=
g_{cache}
\operatorname{Attn}(q_j,C_{t-H:t-1})
```

Cache gate:

```math
g_{cache}
=
\sigma(W[q_j,age,uncertainty,source,\nu_t])
```

High innovation reduces stale cache trust:

```math
\|\nu_t\|\uparrow \Rightarrow g_{cache}\downarrow
```

Cache cannot be action truth:

```text
cache -> AQR evidence -> posterior correction -> action
```

not:

```text
cache -> action
```

## 9. Slot-Level JEPA Theory

The world model predicts future belief latents:

```math
\hat S_{t+1,j}
=
F_\theta(S_{t,j}^{+}, a_t, proprio_t,\ell)
```

The teacher target is detached next posterior:

```math
S_{t+1,j}^{target}
=
stopgrad(S_{t+1,j}^{posterior})
```

Loss:

```math
L_{slot-jepa}
=
\sum_j
d(\hat c_{t+1,j}, stopgrad(c_{t+1,j}^{posterior}))
```

Future target is never current input:

```text
future posterior target affects gradients only through loss
future posterior target does not enter current AQR/posterior/action
```

Support prediction uses:

```math
y_{t+1,j}^{support}
=
[\alpha, support\_mass, contact\_prob, binding\_confidence]_{t+1,j}
```

## 10. Ordinal/Relation Theory

Ordinal references require more than object saliency:

```text
candidate set + axis + rank
```

Axis:

```math
u_\ell = g_\theta(language, frame, task)
```

Score:

```math
s_j = u_\ell^\top \mu_{t,j}
```

Soft rank:

```math
rank_j
=
1+\sum_{l\ne j}\sigma((s_l-s_j)/\tau)
```

This head is a task selector over posterior slots. It cannot rewrite physical identity.

## 11. Information-Theoretic Limit

The architecture lowers architecture error:

```math
E_{total}
=
E_{observation}
+E_{architecture}
+E_{optimization}
```

But it cannot recover information absent from all inputs:

```math
I(Y;A_t) \le I(Y;Z_t)
```

Where:

```text
Y:
  true referred object identity

Z_t:
  all current typed evidence and admissible history

A_t:
  action/anchor decision
```

Thus the system can exploit temporal, point, tactile, posterior, and language evidence, but cannot invent sub-token information that no modality contains.

## 12. Complete Loss Method

The final loss family is:

```math
L =
L_{action}
+ \lambda_{slot} L_{slot-jepa}
+ \lambda_{support} L_{support-pred}
+ \lambda_{bind} L_{binding}
+ \lambda_{xmod} L_{cross-modal}
+ \lambda_{ordinal} L_{ordinal}
+ \lambda_{innov} L_{innovation}
+ L_{existing}
```

Each term must satisfy:

```text
valid-masked
finite when branch missing
no future leakage
does not bypass posterior
budgeted relative to action loss
```

This is not module stacking. Each term supervises one part of the belief-state loop:

```text
slot-JEPA:
  future belief predictability

support prediction:
  future where/contact support predictability

binding consistency:
  identity stability

cross-modal pooled alignment:
  same-slot evidence agreement without raw support shape forcing

ordinal:
  language-conditioned slot selection

innovation:
  surprise/uncertainty calibration
```

## 13. Engineering Method

The direct final deployment should include all graph objects at once:

```text
temporal V-JEPA support
PG image support
address/content slots
evidence cache
slot-JEPA prediction
support prediction
ordinal relation state
diagnostics and loss knobs
```

But the optimization pressure remains guarded:

```text
complete graph present
loss weights explicit
missing modalities produce finite zeros
future target never leaks into current action
cache read/write order preserves causality
```

This is complete deployment with safe mathematical guards, not a reduced architecture.

## 14. Scripted Consistency Criterion

The theory is accepted in code only if the scripted contract verifier passes:

```text
python scripts/verify_picf_owm_contract.py
```

This maps the method to implementation:

```text
belief-state posterior authority
  -> README invariant and posterior/cache checks

typed temporal/semantic evidence
  -> V-JEPA recent_maps, temporal visual support, PG priors

object-addressable slots
  -> slot_address, slot_content, posterior identity metrics

closed-loop correction
  -> previous-cache read, post-correction cache write, innovation metrics

leakage-free prediction
  -> detached next posterior targets for slot-JEPA/support prediction

reviewable deployment evidence
  -> OWM debug metrics and evidence-bundle verifier snapshot
```

The verifier is not a replacement for rollouts. It proves that the mathematical
objects and causal wiring required by the method exist in code. Empirical
acceptance still requires CALVIN/robot evidence that posterior identity,
temporal support, innovation correction, and cache trust behave correctly.

## 15. Method Verdict

PICF-AQR-OWM is coherent because every component has a role in one equation:

```text
belief prior
  + typed measurement
  + posterior correction
  + future prediction
  + innovation
  + action conditioning
```

The architecture remains PICF-native because posterior is never demoted. AQR, V-JEPA, PaliGemma, cache, slot-JEPA, and ordinal relation all serve posterior-centered control.

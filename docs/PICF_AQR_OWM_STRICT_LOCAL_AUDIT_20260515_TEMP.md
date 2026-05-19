# PICF-AQR-OWM Strict Local Audit Temp - 2026-05-15

Status: local code-level audit complete; behavior acceptance remains pending the
active A7 30k long run.

Canonical entry point:
[`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md).

This audit was added because a normal verifier pass is not sufficient. It
records the paper-to-math assumptions, the current data follow-through, the
script checks that were actually executed, and the exact remaining limits. It is
not a CALVIN success claim.

This document now uses three fixed audit phrases so it can be checked by the
professor-grade interaction script:

```text
Data follow-through:
  Observation fields -> typed token field -> AQR support -> posterior binding
  -> predictive/cache state -> training metrics/artifacts.

Script evidence:
  compileall, py_compile, verifier, strict diagnose, recursive dataflow trace,
  MVTrack deep audit, professor-grade interaction audit, and pytest sweeps.

Mathematical contract:
  posterior is the authoritative belief state; cache, tracklets, proposals, and
  V-JEPA/PG branches are typed evidence, not truth; future targets are detached.
```

## 1. Audit Scope

The audit covers the local checkout:

```text
branch: Posterior_VLA
commit: d163d18
working tree before audit: clean except one later test-only fix
```

Primary files followed:

```text
src/openpi/picf/core/config.py
src/openpi/picf/core/contracts.py
src/openpi/picf/core/pipeline.py
src/openpi/picf/core/training.py
src/openpi/picf/contracts.py
src/openpi/picf/vjepa/wrapper.py
scripts/picf_core_train.py
scripts/run_picf_posterior_birth_matrix.sh
scripts/verify_picf_owm_contract.py
scripts/picf_owm_strict_diagnose.py
scripts/picf_owm_dataflow_trace.py
scripts/picf_owm_mvtrack_deep_audit.py
src/openpi/picf/core/pipeline_test.py
```

The active cloud run is intentionally treated as external behavior evidence:
this document only validates that the local code contract and dataflow are
coherent enough to let that run continue.

## 2. Paper Cross-Check

The following papers were rechecked during this audit.

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
arXiv:2510.24709, v2 2026-01-21
Key point: ViT patch embeddings encode IsSameObject with a quadratic similarity
probe, and the signal lies in a low-dimensional subspace above object features.
PICF implication: support-weighted binding_signature is a structural binding
term, not a new scalar loss. It is mathematically consistent to add a projected
same-object subspace to binding logits.
```

```text
OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation
arXiv:2605.06481, 2026-05-07
Key point: object state is decomposed into persistent address and time-varying
content; addressability improves object-specific instruction/action decoding.
PICF implication: slot_address/slot_content/cache metadata are directionally
correct, but address must be gated by current evidence and posterior innovation.
Hard address lock-in would be mathematically wrong.
```

```text
V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
Planning
arXiv:2506.09985, 2025-06-11
Key point: video predictive latents support motion understanding and robot
planning with a latent action-conditioned world model.
PICF implication: recent V-JEPA temporal maps must remain typed temporal
evidence; last-two mean is insufficient as the production representation.
```

```text
TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for
Generalist Robotic Policies
arXiv:2412.10345, v3 2025-06-05
Key point: explicit trajectory/trace evidence improves spatial-temporal
awareness for VLA action prediction.
PICF implication: optional tracklet evidence is justified as typed temporal
support, but if CALVIN does not feed tracklet tensors then the branch is a no-op
and must not be counted as active evidence.
```

```text
DINO: DETR with Improved DeNoising Anchor Boxes
arXiv:2203.03605
Key point: denoising queries and mixed query initialization improve convergence
for query-based detection.
PICF implication: AQR denoising is reasonable only as a guarded training-only
auxiliary. It must not become an inference-time posterior/action source.
```

The paper check supports the current architecture class:

```text
typed evidence -> query/object binding -> posterior belief correction ->
guarded cache/prediction -> PI0.5 action
```

It does not prove current behavior. Papers justify why these mechanisms are
plausible and what invariants they must obey.

## 3. Mathematical Contract

PICF is audited as a POMDP-style belief filter:

```text
b_t(s) proportional p(o_t | s_t)
                 * integral p(s_t | s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds
```

Code roles:

```text
typed memory:
  observation evidence o_t.

AQR:
  measurement routing / assignment p(z_i^(m) | slot_j, task).

posterior:
  authoritative corrected belief b_t^+.

cache:
  bounded historical evidence from t-H:t-1, never current truth.

prediction:
  transition/world-model hook; future targets must be detached.

PI0.5:
  final action generator over corrected belief/context.
```

The core posterior correction still matches precision-style fusion:

```text
Lambda^+ = Lambda^- + Lambda_meas
eta^+    = Lambda^- mu^- + Lambda_meas mu_meas
mu^+     = (Lambda^+)^{-1} eta^+
```

Code evidence:

```text
src/openpi/picf/core/pipeline.py:6467 lambda_prior
src/openpi/picf/core/pipeline.py:6468 eta_prior
src/openpi/picf/core/pipeline.py:6469 lambda_meas
src/openpi/picf/core/pipeline.py:6470 eta_meas
src/openpi/picf/core/pipeline.py:6472 mu_post
```

The information limit is still explicit:

```text
I(Y; A_t | instruction) <= I(Y; Z_{<=t} | instruction)
```

Thus this code can improve use of available evidence, but it cannot guarantee a
sub-token object like "the fourth chopstick" when the visual/point/tactile/trace
evidence does not contain separable instance information.

## 4. Data Follow-Through

### 4.1 Observation to typed token field

```text
PicfObservation
  rgb_static, rgb_gripper, point_set, tactile, proprio, task text,
  optional tracklet/proposal tensors
```

Code evidence:

```text
src/openpi/picf/contracts.py: tracklet_xy / proposal_centers_xy fields
scripts/picf_core_train.py:2978 tracklet_view_ids adapter
scripts/picf_core_train.py:2983 proposal_view_ids adapter
```

Follow-through:

```text
If optional tracklet/proposal arrays exist in an episode frame, the trainer
passes them into PicfObservation. If absent, they are None and the runtime branch
is a valid no-op. This avoids false claims that tracklet/proposal are active in
current CALVIN when metrics show owm_tracklet_tokens=0 and owm_proposal_tokens=0.
```

### 4.2 V-JEPA temporal support

Formula:

```text
M_vjepa = { z_{view,tau,h,w} }
p_j(view,tau,h,w) = softmax(q_j^T k_{view,tau,h,w} + bias)
```

Code evidence:

```text
src/openpi/picf/vjepa/wrapper.py:62 recent_maps()
src/openpi/picf/core/pipeline.py:1599 _visual_maps()
src/openpi/picf/core/pipeline.py:1649 fmap.recent_maps(...)
src/openpi/picf/core/pipeline.py:5126 temporal view_ids
src/openpi/picf/core/pipeline.py:5144 temporal_visual_view_embedding
```

Follow-through:

```text
static and gripper/wrist can enter typed temporal V-JEPA memory with view ids.
Wrist tokens do not inherit static projective rays without extrinsics. This is
the correct conservative design: wrist is evidence, not static-geometry truth.
```

### 4.3 PaliGemma image support

Formula:

```text
p_pg(j,u) = softmax((Wq q_j)^T (Wk e_pg,u) / sqrt(d))
```

Code evidence:

```text
src/openpi/picf/core/pipeline.py:2576 iterates semantic.image_token_ranges
src/openpi/picf/core/pipeline.py:2603 writes pg_priors rows
src/openpi/picf/core/pipeline.py:2328 graph returns pg_priors
```

Follow-through:

```text
PG image evidence is first-class graph support, not only a visual-grid bias.
This matches the typed-memory design and avoids the previous false positive
where PG image evidence could disappear from the graph.
```

### 4.4 Point/projective evidence and observation anchors

Formula:

```text
L_pv = D(p_point, p_visual P_{v->p}) + D(p_visual, p_point P_{p->v})
```

Code evidence:

```text
src/openpi/picf/core/pipeline.py:463 projective_compatibility
src/openpi/picf/core/training.py:1106 point_from_visual
src/openpi/picf/core/training.py:1107 visual_from_point
src/openpi/picf/core/pipeline.py:5781 observation_anchor_seed_point_mix
```

Follow-through:

```text
The new observation-anchor seed-point coverage mix is a real runtime mechanism
used after MAPG graph point mix. The local test had to isolate it explicitly;
otherwise the legacy point-mix unit test compares against an incomplete formula.
```

### 4.5 AQR support and active/dustbin selection

Formula:

```text
support_j = AQR(q_j, typed_memory)
active_j = capacity(relative_score_j, support_overlap_j, geometry_duplicate_j)
inactive anchors become dustbin/recurrent carriers, not object owners.
```

Code evidence:

```text
src/openpi/picf/core/pipeline.py:2293 _aqr_active_slot_mask()
src/openpi/picf/core/pipeline.py:2334 geometry duplicate branch
src/openpi/picf/core/pipeline.py:2361 relative score threshold
src/openpi/picf/core/pipeline.py:3631 anchor_active call
```

Follow-through:

```text
Raw same-role overlap is no longer the acceptance metric by itself. Active
same-role support/core overlap and overlay ownership are the metrics that matter.
Inactive/dustbin anchors may overlap by design.
```

### 4.6 Binding signature and posterior identity

Formula:

```text
binding_logit =
  hidden_similarity
  - geometry_cost
  + support_signature_similarity
  + gated_address_bias
```

Code evidence:

```text
src/openpi/picf/core/pipeline.py:1058 binding_signature_proj
src/openpi/picf/core/pipeline.py:2746 _support_binding_signature()
src/openpi/picf/core/pipeline.py:5951 _binding_logits()
src/openpi/picf/core/pipeline.py:6002 prev/obs binding_signature term
```

Follow-through:

```text
This implements the IsSameObject-inspired subspace structurally inside binding.
It is not an extra loss and does not require object labels. It remains gated by
geometry/recycle/posterior dynamics and therefore avoids hard object-ID lock-in.
```

### 4.7 Cache read/write

Formula:

```text
q' = q + lambda_cache * (ReadCache(q, C_{t-H:t-1}) - q)
```

Code evidence:

```text
src/openpi/picf/core/pipeline.py:2765 reads previous.predictive.evidence_cache
src/openpi/picf/core/pipeline.py:2775 skips immediate posterior duplicate
src/openpi/picf/core/pipeline.py:3438 q_before_cache
src/openpi/picf/core/pipeline.py:3445 residual scaling
```

Follow-through:

```text
The previous softmax-constant cache-weight bug is fixed. The current cache is
causal, skips the newest posterior duplicate, and uses residual scale. It is
still auxiliary evidence, not a posterior replacement.
```

### 4.8 Loss and action cotrain

Code evidence:

```text
src/openpi/picf/core/training.py:77 lambda_slot_jepa = 0.0
src/openpi/picf/core/training.py:78 lambda_support_pred = 0.0
src/openpi/picf/core/training.py:79 lambda_binding_consistency = 0.0
src/openpi/picf/core/training.py:80 lambda_aqr_denoising = 0.0
scripts/picf_core_train.py:7191 --picf-action-prefix-stopgrad
scripts/picf_core_train.py:3145 loss_action_default_equiv
```

Follow-through:

```text
High-risk OWM auxiliary losses remain hooks with zero default. The 30k long run
uses action cotrain with prefix-stopgrad so the action loss still trains the
action side while protecting PICF pi-prefix tokens from directly destroying the
binding subspace.
```

### 4.9 Long-run launch artifacts

Code evidence:

```text
scripts/run_picf_posterior_birth_matrix.sh:248 a7_dustbin_long30k
scripts/picf_core_train.py:6212 startup training config log
scripts/picf_core_train.py:6630 anchor_overlay_interval path
scripts/picf_core_train.py:6949 save interval path
scripts/picf_core_train.py:6973 keep_last_checkpoints
```

Follow-through:

```text
The selected run saves checkpoints every 2500 steps, keeps last 3, and writes
anchor overlays every 100 steps. This is the correct artifact set for behavior
acceptance: metrics alone are not enough.
```

## 5. Script Checks Executed

The following commands were run locally in `/home/siyuanyue/Documents/openpi`.

```text
git status --short:
  initially clean except the later test-only isolation fix.

PYTHONPATH=src python -m compileall -q src/openpi/picf scripts:
  PASS

find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n:
  PASS

PYTHONPATH=src python -m py_compile scripts/picf_core_train.py scripts/serve_picf_policy.py scripts/verify_picf_owm_contract.py scripts/picf_owm_strict_diagnose.py scripts/picf_owm_dataflow_trace.py scripts/picf_owm_mvtrack_deep_audit.py scripts/picf_owm_same_object_probe.py:
  PASS

PYTHONPATH=src python scripts/verify_picf_owm_contract.py:
  34/34 PASS

PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS, with expected WARN only because no runtime metrics/CALVIN artifact was supplied.

PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS, regenerated docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md.

PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  28/28 PASS

PYTHONPATH=src python scripts/picf_owm_professor_grade_audit.py --fail-on-fail --markdown docs/PICF_AQR_OWM_PROFESSOR_GRADE_INTERACTION_AUDIT_TEMP.md:
  15/15 PASS
  This is the strictest interaction-level script in this checkout. It checks
  multiview temporal evidence, cache residual/address gating, active/dustbin
  capacity control, support/address binding trust, posterior recycle math,
  detached future teachers, action-loss comparability, and diagnostic coverage.

PYTHONPATH=src uv run pytest -q targeted lightweight group:
  50 passed, 3 warnings

PYTHONPATH=src uv run pytest -q core runtime group:
  first run: 289 passed, 1 failed
  root cause: stale legacy MAPG test ignored observation_anchor_seed_point_mix
  fix: set observation_anchor_seed_point_mix=0.0 inside that test only
  rerun: 290 passed, 26 warnings

PYTHONPATH=src uv run pytest -q src/openpi/picf scripts --ignore=scripts/train_test.py:
  380 passed, 3 skipped, 28 warnings
```

Blocked checks:

```text
PYTHONPATH=src uv run pytest -q src/openpi/picf scripts:
  blocked at collection by scripts/train_test.py importing wandb, which then
  fails because local wandb lacks wandb_watchdog.observers.polling.
  This is an environment/dependency issue in the generic train test, not a
  PICF/MVTrack runtime failure.

uv run ruff check src/openpi/picf/core/pipeline_test.py:
  fails on existing test-file lint baseline: import order, private-method test
  access, old dict() style, boolean positional literals, etc. These existed
  before this audit and do not indicate a runtime contract failure.
```

## 6. Test Fix Made During Audit

File:

```text
src/openpi/picf/core/pipeline_test.py
```

Change:

```text
test_mapg_observation_point_mix_floor_reaches_final_point_weights now sets
observation_anchor_seed_point_mix=0.0.
```

Reason:

```text
The test is specifically about legacy MAPG observation point mix floor. The
runtime now also applies observation-anchor seed coverage after graph mix. If
the test does not disable seed mix, it compares against the wrong formula and
creates a false failure. Runtime behavior was not changed.
```

## 7. Strict Interpretation

## 2026-05-15 Owner/Reserve Follow-Through Re-Audit

Additional audit added after the rejected A7 long-run exposed that active graph
owners were not sufficient: reserve observation rows could still update
posterior object files.

New strict TEMP file:

```text
docs/PICF_AQR_OWM_OWNER_GATE_FOLLOWTHROUGH_20260515_TEMP.md
```

New script:

```bash
PYTHONPATH=src python scripts/picf_owm_owner_gate_followthrough_audit.py --fail-on-fail
```

Result:

```text
SUMMARY pass=12 warn=0 fail=0 total=12
```

The script verifies the full cross-layer chain:

```text
PicfAnchorPriorGraphState.anchor_active
  -> _observation_owner_active_from_graph(...)
  -> PicfObservationAnchorState.owner_active
  -> _posterior_owner_active_binding_bias(...)
  -> _posterior_update(... bind_logits += owner_bias ...)
  -> posterior_owner_active_* metrics
  -> trainer/evidence bundle/README audit keys
```

The mathematical constraint is:

```math
\ell^{post}_{j,i}
\leftarrow
\ell^{post}_{j,i}
+
\begin{cases}
0, & owner_i \ge \tau_{owner}\\
-10^4, & owner_i < \tau_{owner}
\end{cases}
```

This is a posterior measurement-eligibility constraint, not a new loss. It
does not create missing object evidence or solve ordinal sub-token grounding.
It prevents inactive fixed-capacity reserve rows from becoming persistent
object-file measurements.

What is proven locally:

```text
1. Code parses and targeted/local runtime tests pass after one stale test fix.
2. AQR/MVTrack contract, strict diagnose, dataflow trace, and deep audit pass.
3. The active/dustbin 30k profile is present in the launch script.
4. The math/dataflow path is internally consistent at code level:
   typed evidence -> AQR routing -> posterior correction -> cache/prediction ->
   PI0.5 action.
```

What is not proven locally:

```text
1. 30k behavior success.
2. CALVIN/video improvement.
3. Long-run recycle stability.
4. Fine-grained ordinal instance resolution.
5. Tracklet/proposal branch usefulness when current CALVIN run feeds zero tokens.
```

The current conclusion is therefore:

```text
Local code-level audit: PASS with one test-only fix.
Mathematical dataflow: coherent.
Paper alignment: plausible and specifically mapped to mechanisms.
Behavior acceptance: pending A7 30k metrics/checkpoints/anchor overlays.
```

## 2026-05-16 Confidence Semantics Re-Audit

The reviewer criticism that the confidence machinery looked over-complex is
valid if every field named "confidence" is treated as the same probability. The
local code path is coherent only under the following separation:

```text
graph.anchor_confidence:
  measurement_quality = max typed-support concentration over visual, temporal,
  PG, point, tactile, posterior, cache, tracklet, and proposal priors.

posterior.alpha:
  belief_activity = posterior object-file survival/activity after support mass,
  assignment margin, entropy, owner-active eligibility, innovation stability,
  previous alpha, and recycle/lifecycle calibration.

graph.anchor_downstream_weight:
  action_exposure = active/context/reserve routing weight applied to graph
  prefix tokens before PI0.5 control consumption.
```

The dataflow remains:

```text
typed evidence
  -> measurement quality for AQR active/context/reserve selection
  -> posterior correction and lifecycle calibration
  -> belief activity / file competition
  -> action exposure through graph/posterior prefix gates
```

This is a reliability-gated belief filter, not a fully calibrated Bayesian
probability model. The accepted local cleanup is to document the semantics and
audit the dataflow, not to delete the gates. Deleting them would reintroduce
the observed failure modes: duplicate reserve rows entering posterior updates,
inactive files entering action prefix, and support concentration being confused
with object-file survival.

Validation rerun after the cleanup:

```bash
python -m py_compile \
  scripts/picf_core_train.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_owm_evidence_bundle.py \
  scripts/picf_owm_same_object_probe.py \
  scripts/picf_action_visible_reserve_gate_audit.py \
  scripts/picf_binding_dataflow_math_audit.py \
  scripts/picf_binding_logit_calibration_audit.py \
  scripts/picf_binding_signature_common_mode_audit.py \
  scripts/picf_posterior_file_competition_audit.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/paligemma/wrapper.py

python scripts/verify_picf_owm_contract.py
python scripts/picf_action_visible_reserve_gate_audit.py --fail-on-fail
python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
uv run python scripts/picf_posterior_file_competition_audit.py --fail-on-fail
uv run python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
uv run python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
uv run pytest -q scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py \
  scripts/picf_anchor_run_diagnostic_report_test.py \
  scripts/picf_owm_same_object_probe_test.py
uv run pytest -q src/openpi/picf/core/pipeline_test.py \
  -k "posterior_inactive_files or posterior_file_competition or posterior_lifecycle or binding_signature or active_slot or temporal_visual or pg_image"
uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py
```

Result:

```text
All commands above passed after one audit-only false positive fix in
scripts/picf_action_visible_reserve_gate_audit.py. The fix changed the audit to
look for the actual overlay dataflow markers (`variant_name="with_gray"`,
`variant_name="active_only"`, `include_inactive=True/False`) instead of stale
older note text.
```

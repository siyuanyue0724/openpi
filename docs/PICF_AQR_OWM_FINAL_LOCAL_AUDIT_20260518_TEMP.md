# PICF-AQR-OWM Final Local Audit: Paper Math, Dataflow, And 30k Readiness

Date: 2026-05-18
Scope: local-only code/math/dataflow audit. Remote A7 training was not modified
by this audit except for the earlier already-recorded VCAP in-place fix and the
30k launch recorded in `PICF_AQR_OWM_CURRENT_STATE_20260518.md`.

## Executive Conclusion

The current local code passes the strictest available static, mathematical, and
scripted dataflow checks. I do not see a local code/dataflow blocker that
requires stopping the current A7 30k run.

This is not a claim that behavior is solved. The current 30k must still pass
runtime acceptance:

```text
step 50: detailed JSON losses are sane
step 100: anchor overlays show task-object posterior ownership
step 2500+: checkpoint trend does not regress action or posterior health
CALVIN/video eval: behavior evidence, not just graph evidence
```

## Paper-Code Boundary

### Object Binding In ViTs

Local code inspected:

```text
/tmp/vit-object-binding
/tmp/picf_paper_code_20260515/vit-object-binding
```

Relevant paper-code equations:

```text
Diagonal quadratic:
  s(x,y) = w^T (x * y) + b

Full quadratic:
  s(x,y) = x^T ((W + W^T) / 2 sqrt(d)) y + b

Fixed-rank quadratic:
  s(x,y) = x^T ((W1^T W2 + W2^T W1) / 2 sqrt(d)) y + b

Training in paper code:
  labels_pairwise = instance_mask_i == instance_mask_j
  loss = BCEWithLogitsLoss(s(x_i, x_j), labels_pairwise)
```

PICF implementation decision:

```text
Do use:
  the pairwise quadratic same-object score family as runtime binding evidence.

Do not use:
  online BCE IsSameObject loss, because current CALVIN/PICF does not provide
  reliable instance-mask labels for every visual/point/tactile token.
```

PICF runtime math:

```text
raw_pair_score =
  lambda_linear * cos(project(prev_signature), project(obs_signature))
  + lambda_diag * diag_quadratic(prev_signature, obs_signature)
  + lambda_low_rank * low_rank_quadratic(prev_signature, obs_signature)

calibrated_score =
  double_center_zscore(raw_pair_score)

binding_logit =
  hidden_similarity
  - geometry_mahalanobis
  + support_overlap_gate * support_overlap
  + posterior_trust_gate * calibrated_score
  + address_gate * address_similarity
```

The key correctness point is calibration. A raw positive common-mode matrix is
not object identity. The double-center/zscore path removes row saliency, column
saliency, and global bias before adding pairwise evidence to posterior binding.

### SlotContrast / Slot Attention Family

Local code inspected:

```text
/tmp/picf_paper_code_20260515/slotcontrast
```

Relevant paper-code components:

```text
SlotAttention:
  softmax over slots, normalized over features, iterative GRU/MLP update

Slot-Slot contrastive loss:
  normalize slots
  compare adjacent-frame slot rows
  CrossEntropyLoss with identity matrix target
```

PICF implementation decision:

```text
Do use:
  competitive support routing, posterior file competition, birth competition,
  and active/context/reserve routing as native belief-filter equivalents.

Do not use by default:
  same-index slot contrastive/predictive pressure, because PICF slots can
  recycle, birth, and swap; index identity is not guaranteed.
```

Current guarded losses:

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
```

This is intentional. The runtime exposes hooks and diagnostics, but production
training does not force a potentially wrong slot-index target.

## PICF Dataflow Follow-Through

Current training dataflow:

```text
PicfObservation
  rgb_static / rgb_gripper / depth / point_set / tactile / language
  optional tracklet_* and proposal_* sidecars if present

TokenField
  PaliGemma text/image typed support
  V-JEPA static+wrist temporal typed support
  Sonata point typed support
  AnyTouch tactile typed support
  optional tracklet/proposal typed support

VCAP active proposal initializer
  reads compact evidence summary
  emits padded physical query initializers and active probabilities
  does not prune dense memory
  does not replace posterior files
  action_grad_scale=0 for current 30k

AQR graph
  dense typed readers over visual/temporal/PG/point/tactile/posterior/cache
  ownership priors and same-role competition
  active/context/reserve slot routing

Observation anchors
  support signatures, geometry, role ids, binding signatures

Posterior update
  hidden + geometry + support overlap + calibrated binding subspace + gated
  address
  posterior file competition
  posterior birth transport
  posterior remains authoritative belief state

Action path
  PI0.5 semantic/action generator
  PaliGemma trainable in the current A7 30k
  Sonata/V-JEPA/AnyTouch frozen
```

No dense evidence token is dropped by VCAP. Inactive proposals are reserve
initializers, not memory pruning. Background/context remains available through
dense visual/temporal/fused public reads and the context/reserve route.

## Loss Interaction Audit

Current 30k loss design:

```text
Primary:
  action loss over PI0.5 action generator

Budgeted support:
  alignment graph loss
  physical auxiliary group
  semantic auxiliary group

Active object allocator:
  low-weight VCAP coverage / duplicate / count / continuity health terms

Guarded but zero:
  slot-JEPA
  support prediction
  binding consistency
  denoising
```

Why this is mathematically consistent:

```text
1. action loss trains control and semantic/action coupling.
2. AQR/posterior losses constrain measurement routing and belief health.
3. VCAP only reallocates active proposal capacity under posterior authority.
4. high-risk future/identity losses stay zero until permutation and identity
   diagnostics justify them.
5. no weak mask/tracklet pseudo-label is used as online ground-truth identity.
```

The current design is not "loss sprawl" in the dangerous sense because each
non-action term maps to one belief-filter role:

```text
evidence routing
belief correction
capacity allocation
diagnostics / guarded teachers
```

## Scripted Verification Results

Executed locally from `/home/siyuanyue/Documents/openpi` on 2026-05-18.

```text
python -m py_compile selected core/scripts: PASS
python scripts/verify_picf_owm_contract.py: PASS
python scripts/picf_owm_strict_diagnose.py --fail-on-fail: PASS
python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail: PASS
python scripts/picf_owm_dataflow_trace.py --fail-on-fail: PASS
python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail: PASS
uv run python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail: PASS
uv run python scripts/picf_posterior_file_competition_audit.py --fail-on-fail: PASS
python scripts/picf_posterior_birth_transport_audit.py --fail-on-fail: PASS
python scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py --fail-on-fail: PASS
uv run pytest -q targeted script tests: PASS, 10 passed
git diff --check: PASS
```

Full command output was captured at:

```text
/tmp/picf_final_local_audit_20260518.log
```

## Self-Critique / Remaining Boundaries

### Not Solved By Local Audit Alone

```text
1. Behavior acceptance still requires current 30k metrics, overlays, and eval.
2. "Fourth object from left" remains impossible to guarantee without sufficient
   observable instance/rank evidence.
3. Optional tracklet/proposal dataflow is valid, but current A7 run has no
   sidecar root; tracklet/proposal tokens can remain zero.
4. Blind SAM proposal memory remains default-off because previous visual checks
   showed noisy fragments.
5. VCAP is running as guarded 30k diagnostic-long training, not yet proven as
   final production default.
```

### Failure Conditions That Would Require Stopping The 30k

```text
1. step-50 JSON shows VCAP count collapse plus unexplained evidence rising.
2. step-100 overlays show active posterior files consistently missing task
   objects.
3. posterior_file_competition_duplicate_overlap_max stays high while
   posterior_active_file_fraction collapses.
4. action loss stops improving while anchor health metrics regress together.
5. grad clipping is constantly active with exploding preclip gradients.
```

## Final Local Audit Decision

The current code is locally coherent enough to continue the running A7 30k.
No new run is needed from this local audit. The next hard evidence should come
from step 50 losses and step 100 overlays.

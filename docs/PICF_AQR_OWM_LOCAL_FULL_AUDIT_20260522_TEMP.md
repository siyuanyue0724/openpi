# PICF-AQR-OWM Local Full Audit, 2026-05-22

Status: local code/dataflow/math audit passed after one stale-audit correction.
This is not a CALVIN success claim and not a 30k behavior acceptance result.

Navigation: use
[`docs/PICF_AQR_OWM_MATH_CONSISTENCY_AND_DOC_INDEX_20260522_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_MATH_CONSISTENCY_AND_DOC_INDEX_20260522_TEMP.md)
as the current mathematical/documentation index before reopening older
SAM/raw-overlap/slot-paper threads.

## 1. Purpose

This audit answers whether the current local codebase still matches the
maintained v2.2 belief-router contract before the active remote long run is
judged or modified.

The scope is deliberately stricter than a compile check:

```text
code syntax
shell launch syntax
contract verifier
strict diagnose
MVTrack/OEML/static dataflow
latest slot-paper deployment audit
Object-Binding/SlotContrast paper-code provenance
binding-signature math
owner/reserve/context action visibility
posterior birth/file competition/file continuity
tactile acceptance
core training/pipeline regression tests
documented known limits
```

## 2. Paper Grounding Checked

The audit uses paper code as external design provenance only. Runtime PICF does
not import these repositories.

```text
/tmp/vit-object-binding
  remote: https://github.com/liyihao0302/vit-object-binding.git
  head:   014c66b45ea262f9b6eec83ff388a1e1c10dfcaa
  checked:
    DiagonalQuadraticProbe
    QuadraticProbe
    QuadraticFixedRankProbe
    BCEWithLogitsLoss over labels_pairwise

/tmp/picf_paper_code_20260515/slotcontrast
  remote: https://github.com/martius-lab/slotcontrast.git
  head:   55ec66dc02eeade630805789ef4a6c5df06f21ff
  checked:
    MIT License
    Slot_Slot_Contrastive_Loss
    batch_contrast
    CrossEntropyLoss
```

Design interpretation:

```text
Accept:
  pairwise/quadratic same-object binding subspace;
  calibrated relative pair scores, not raw cosine;
  temporal/object consistency as design evidence;
  slot competition, background residual, duplicate suppression.

Reject for current runtime:
  direct copy of paper source into runtime;
  online IsSameObject BCE without reliable labels;
  hard VQ visual prototype truth as posterior identity;
  full RGB reconstruction decoder inside PI0.5 action path;
  blind SAM as maintained object evidence.
```

## 3. Mathematical Follow-Through

### 3.1 Binding Subspace

PICF follows the Object-Binding result at the equation-family level:

```math
s_{ij}^{lin} = q_i^\top k_j
```

```math
s_{ij}^{diag} = (q_i \odot d)^\top k_j
```

```math
s_{ij}^{lr} = (L q_i)^\top (R k_j)
```

The combined pair score is not injected directly. It is converted into a
relative assignment signal:

```math
\tilde{s}_{ij}
=
s_{ij}
- \frac{1}{J}\sum_j s_{ij}
- \frac{1}{I}\sum_i s_{ij}
+ \frac{1}{IJ}\sum_{ij}s_{ij}
```

Then it is z-scored only if its dispersion exceeds a minimum standard
deviation. This is the critical anti-false-positive step:

```text
common-mode scene saliency -> zero identity evidence
row/column saliency bias   -> zero identity evidence
relative same-object pair  -> survives
low-dispersion noise       -> rejected
```

The audit explicitly checked these cases in
`scripts/picf_binding_dataflow_math_audit.py`,
`scripts/picf_binding_logit_calibration_audit.py`, and
`scripts/picf_binding_signature_common_mode_audit.py`.

### 3.2 Posterior File Memory

Posterior file identity is not overwritten by a single frame:

```math
b_t
=
(1-\eta_t) b_{t-1}
+ \eta_t \hat{b}_t
```

where the update rate is gated by assignment trust, owner reliability,
support mass, recycle/birth state, and calibrated measurement dispersion.

This prevents two historical failure modes:

```text
false lock:
  a noisy observation overwrites identity;

false inertia:
  an old address/cache locks stale identity despite innovation.
```

### 3.3 Active / Context / Reserve Semantics

The maintained control graph is tri-state:

```text
active object:
  full action evidence;

context object:
  low-priority action-visible scene evidence;

reserve/no-object:
  fixed capacity, duplicate/null row, not action evidence.
```

Important correction in this audit:

Older audit text expected:

```text
graph_tokens = graph_tokens * downstream_weight
```

That is no longer the maintained path. Current code keeps token content intact
and uses downstream reliability as an attention prior:

```math
\mathrm{bias}_i = \log(\max(w_i, \epsilon))
```

This bias is applied to:

```text
control_world self-attention
PI prefix reader
future-condition reader
```

This is more mathematically consistent than token scaling because it does not
destroy dense slot content or V-JEPA-style embedding geometry. Reserve rows are
hard downweighted by attention bias; context rows remain readable.

## 4. Commands And Results

### 4.1 Static And Contract Checks

```text
uv run python -m py_compile \
  scripts/picf_core_train.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  scripts/picf_owm_dataflow_trace.py \
  scripts/picf_owm_strict_diagnose.py \
  scripts/picf_owm_mvtrack_deep_audit.py \
  scripts/verify_picf_owm_contract.py

result:
  pass
```

```text
bash -n scripts/experiments/picf_aqr_owm_202605_active/*.sh

result:
  selected active launch scripts pass syntax
```

```text
git diff --check

result:
  pass
```

```text
uv run python scripts/verify_picf_owm_contract.py

result:
  all verifier checks pass
```

```text
uv run python scripts/picf_owm_strict_diagnose.py --fail-on-fail

result:
  ok=true
  expected WARN only when no runtime metrics/eval artifact path is supplied
```

```text
uv run python scripts/picf_owm_dataflow_trace.py --fail-on-fail

result:
  ok=true
  nodes=20
  refreshed docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md
```

### 4.2 Paper/Dataflow Audits

```text
uv run python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_oeml_dataflow_audit.py --fail-on-fail
result: 6/6 PASS

uv run python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
result: PASS after restoring /tmp/vit-object-binding paper-code snapshot

uv run python scripts/picf_latest_slot_deployment_audit.py --fail-on-fail
result: 16/16 PASS

uv run python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
result: 16/16 PASS

uv run python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
result: 9/9 PASS after restoring /tmp/picf_paper_code_20260515 paper-code paths
```

### 4.3 Binding/Owner/Posterior Audits

```text
uv run python scripts/picf_binding_logit_calibration_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_binding_signature_common_mode_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_action_visible_reserve_gate_audit.py --fail-on-fail
result: PASS after updating stale audit expectation from token scaling to attention bias

uv run python scripts/picf_object_candidate_slot_binding_audit.py --json
result: ok=true

uv run python scripts/picf_posterior_binding_signature_memory_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_posterior_birth_transport_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_posterior_file_competition_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_posterior_file_continuity_metric_audit.py --fail-on-fail
result: PASS

uv run python scripts/picf_tactile_acceptance_audit.py
result: overall_status=pass
```

### 4.4 Regression Tests

```text
uv run pytest -q src/openpi/picf/core/training_test.py
result: 34 passed

uv run pytest -q scripts/picf_core_train_test.py
result: 128 passed

uv run pytest -q src/openpi/picf/core/pipeline_test.py
result: 101 passed

uv run pytest -q \
  scripts/picf_owm_same_object_probe_test.py \
  scripts/picf_owm_evidence_bundle_test.py \
  scripts/verify_picf_owm_contract_test.py
result: 7 passed

uv run pytest -q \
  scripts/picf_loss_audit_test.py \
  scripts/picf_anchor_run_diagnostic_report_test.py \
  scripts/picf_tactile_acceptance_audit_test.py
result: 7 passed
```

## 5. Findings

### 5.1 What Is Code-Level Closed

```text
MVTrack contract:
  static+wrist temporal view typing, optional tracklet/proposal fields,
  address-aware cache, denoising guards, matched predictive hooks.

Binding:
  support-weighted signatures, native linear/diagonal/low-rank pairwise scores,
  double-centered/z-scored calibration, posterior trust gates.

Posterior:
  active owner gate, file competition, bounded birth/dustbin transport,
  trust-gated binding-signature memory, calibrated continuity metrics.

Context:
  active/context/reserve split is explicit;
  context survives as low-priority attention-readable evidence;
  reserve/no-object rows are strongly downweighted;
  dense typed memory is not pruned.

Tactile:
  object-owner tactile routing is contract-valid;
  role-0 effector ownership can be excluded in object-owner probes.

SAM:
  blind SAM is archived/rejected as maintained object source.
```

### 5.2 What Is Not Proven By This Audit

```text
30k behavior:
  not proven; requires long-run action loss, overlays, CALVIN/video evidence.

Fine ordinal / "fourth chopstick":
  not solved without additional evidence/labels; remains weak diagnostic.

Full sidecar coverage:
  code/dataflow are valid, but complete dataset-side sidecar coverage is an
  artifact-generation and quality-control issue, not a local code proof.

Tracklet/proposal usefulness:
  typed memory contract is valid; actual gain depends on generated sidecars.

Action plateau:
  current remote action-dominant run is structurally healthy through the latest
  checked interval, but action default-equivalent loss has plateaued around
  the old 1500-2500 step band. This needs longer evidence before changing
  architecture or action weight again.
```

## 6. Current Remote Run Interpretation Snapshot

The active remote run is:

```text
tmux:
  picf_a7_action2_from500_20260521

experiment:
  picf_a7_actionaware_defaultsync_action2_from500_long30k_20260521

resume:
  step500 checkpoint from picf_a7_actionaware_defaultsync_long30k_ckpt500_20260521

action pressure:
  ACTION_LOSS_WEIGHT=2.0
  loss_action_weight_scale=1.0
```

Local notes from the latest available metrics during this audit:

```text
latest checked step:
  1600

hard structural failures:
  absent in the checked window:
    posterior active duplicate overlap = 0
    active same-role support overlap = 0.070
    downstream same-role support overlap = 0.095
    recycle rate ~= 0

raw same-role overlap:
  still near 1.0, but dominated by reserve/context fixed-capacity rows.
  It is not a stop signal by itself.

action:
  latest loss_action_default_equiv = 0.0504.
  Compared with 2026-04-22 ablation, this is roughly old 2000-3000 step level,
  not yet old 5k/10k/20k level.

owner target:
  loss_anchor_object_pull = 0.216
  loss_anchor_pv = 0.611
  loss_aqr_denoising = 1.193

slot-JEPA telemetry:
  latest loss_slot_jepa = 0.279.  Earlier spikes remain telemetry because
  lambda_slot_jepa is zero in this run; it is not optimized pressure.
```

## 7. Next Gate

Do not add another module only because raw overlap is high. The next legitimate
decision gate is:

```text
wait for step2500 checkpoint/metrics unless active collapse appears earlier;
compare action_default_equiv against 2026-04-22 ablation;
inspect active_only and with_gray overlays;
check:
  active duplicate overlap
  active/downstream same-role overlap
  recycle rate
  object_pull
  denoising
  binding/file-continuity diagnostics
```

If action stays flat while structural metrics remain healthy, the next
principled architecture candidate is not another slot-loss patch. It is a
separate dense-context injection path:

```math
H' = H + \tanh(g_\ell)\,\mathrm{CrossAttn}
  \left(
    Q=\mathrm{LN}(H),
    K,V=\mathrm{LN}(R(C_{dense}))
  \right)
```

where `C_dense` is low-confidence/inactive/raw dense context and `R` is a small
resampler. This is conceptually aligned with Flamingo-style gated
cross-attention and JEPA-VLA/VLA-JEPA-style latent video conditioning, but it is
not part of the active 30k run and should not be mixed into the current
acceptance test.

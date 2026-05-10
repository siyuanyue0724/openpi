# PICF-AQR-OWM Deployment Status Temp

Source contract: `docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md`

Purpose: live deployment ledger for the direct-to-final PICF-AQR-OWM implementation. This file records every README contract item as code status, test evidence, and remaining guard. It is intentionally temporary and should be kept in sync while code is being deployed.

## Verdict

Current status: code deployed for the direct final graph, with training/runtime guards still controlling high-risk losses.

Deployment interpretation:

```text
Target:
  implement the complete final PICF-AQR-OWM graph directly.

Guard:
  do not call a changed checkout final until every contract object is present,
  wired into forward paths, covered by tests, and visible in debug/status.
  This branch satisfies that code-level guard for PICF-AQR-OWM v1.2 and must
  be re-verified after any follow-up change.

Non-negotiable:
  posterior remains authoritative; cache and future prediction stay auxiliary.
```

Version:

```text
PICF-AQR-OWM v1.3
Status: complete for the code-level deployment contract after the point/visual cleanup and default-profile cleanup pass.
```

Default-profile result:

```text
aqr_mapg_enabled=True
semantic_mode=paligemma
mapg_enabled=False
vl_anchor_router_enabled=False
aqr_pg_grounding_enabled=False
aqr_pg_image_support_enabled=True
aqr_vjepa_temporal_mode=last_two_tokens
evidence_cache_read_weight=0.05
lambda_mapg_cycle=0.02
lambda_mapg_support_diversity=0.01
lambda_slot_jepa=lambda_support_pred=lambda_binding_consistency=0.0
```

This means the latest OWM wiring is the default training profile. Explicit CLI
overrides still exist for ablations, but are not required to avoid a legacy
MAPG-v0 path.

Serve-time compatibility:

```text
If old checkpoint metadata has semantic_mode=zero and lacks an explicit
aqr_mapg_enabled field, serve_picf_policy keeps aqr_mapg_enabled=False.
```

This prevents new production defaults from silently promoting legacy
zero-semantic checkpoints into a PaliGemma-required AQR path.

## Contract Ledger

| README contract | Code target | Status | Evidence |
|---|---|---:|---|
| Temporal V-JEPA support | `PicfTemporalVisualSupportState`, `VjepaFeatureMap.recent_maps`, token field temporal branch, AQR temporal priors | deployed | `recent_maps(n)` preserves time; AQR emits `vjepa_temporal_priors`; tests cover non-mean path |
| First-class PG image support | AQR fills `graph.pg_priors` from all PG image ranges/views | deployed | AQR returns `pg_priors` independently from visual bias; tests assert non-None priors |
| Posterior address/content split | posterior and graph expose `slot_address` / `slot_content`; binding remains correction-aware | deployed | posterior exposes persistent address and current content; graph exposes physical query addresses and content |
| Posterior-grounded evidence cache | fixed-size `PicfEvidenceCacheState`; previous-cache read, post-correction write | deployed | recurrent predictive state carries cache; first step cannot read same-step cache; second step reads previous cache |
| Slot-level JEPA prediction | slot prediction tensors and next-posterior target loss | deployed, guarded | `slot_prediction_tokens` exist; window training passes detached next posterior slots as teacher; `lambda_slot_jepa` defaults to 0 |
| Support prediction | support-prediction fields and next-posterior support summary target | deployed, guarded | `slot_prediction_supports` predicts detached `[alpha, support_mass, contact_prob, binding_confidence]`; `lambda_support_pred` defaults to 0 |
| Ordinal/relation grounding | gated relation state/debug, no posterior identity rewrite | deployed, guarded | prompt-gated ordinal state exists; test confirms posterior is unchanged |
| Graph point/visual projection consistency | bidirectional `graph.visual_priors` / `graph.point_priors` consistency through `projective_compatibility`; legacy `focus_pv` removed | deployed | `_mapg_cycle_loss` directly compares point-from-visual and visual-from-point supports; defaults `lambda_mapg_cycle=0.02`, `lambda_mapg_support_diversity=0.01` |
| Training CLI/config entry | `scripts/picf_core_train.py` maps OWM config and guarded losses | deployed | CLI, defaults, `PicfCoreConfig`, `PicfTransitionLossConfig`, and log contract include OWM fields |
| Debug/diagnostics | temporal, PG, cache trust, identity switch/recycle, support overlap, guarded prediction, relation diagnostics | deployed | debug emits OWM temporal/PG/cache/identity/ordinal metrics; training metrics expose deployed loss terms without false address-drift or ordinal-loss keys |
| Strict scripted diagnosis | README-to-code verifier and evidence bundle verifier snapshot | deployed | `scripts/verify_picf_owm_contract.py` checks contracts, forward wiring, losses, metrics, and bundle coverage |
| Tests | shape, no-leakage, cache causality, PG priors, temporal priors, script entry stability | deployed | Targeted and broad PICF/script suites pass locally; see verification block |

## Running Audit Notes

1. `PicfPosteriorAnchorState` already implements the PICF belief-state core through recurrent `h/c`, `mu/Sigma`, geometry, support mass, binding, evidence tokens, and `global_post`.
2. `_current_prior(...)`, `_posterior_update(...)`, `_build_physical_predictive_basis(...)`, and `_innovation(...)` already form the posterior prediction/correction/innovation loop.
3. Final deployment must strengthen that loop, not bypass it.

## Code Changes Deployed

1. `src/openpi/picf/core/contracts.py`
   - Added `PicfTemporalVisualSupportState`.
   - Added fixed-size `PicfEvidenceCacheState`.
   - Extended anchor graph with `vjepa_temporal_priors`, `cache_priors`, `slot_address`, `slot_content`, and `support_uncertainty`.
   - Extended posterior, predictive, recurrent predictive, token field, and task readout contracts with OWM fields.

2. `src/openpi/picf/vjepa/wrapper.py`
   - Added `VjepaFeatureMap.recent_maps(n)` so temporal evidence is preserved instead of averaged.

3. `src/openpi/picf/core/pipeline.py`
   - Added `_visual_maps(...)` to produce both current 2D map and recent temporal maps from one V-JEPA encode.
   - Projects recent temporal maps into typed temporal visual support tokens.
   - AQR reads temporal visual support and exposes `graph.vjepa_temporal_priors`.
   - AQR reads all PaliGemma image-token ranges and exposes `graph.pg_priors`.
   - Added previous-cache read as weak AQR evidence and post-correction cache write for the next step.
   - Added posterior address/content fields.
   - Added slot prediction tokens and support summary prediction.
   - Added prompt-gated ordinal relation state without posterior identity rewrite.

4. `src/openpi/picf/core/training.py`
   - Added guarded OWM losses: `slot_jepa`, `support_pred`, and `binding_consistency`.
   - `slot_jepa` uses detached next posterior slot tokens when available; detached future visual latent summary is only a compatibility fallback.
   - `support_pred` uses detached next posterior support summary `[alpha, support_mass, contact_prob, binding_confidence]` when available; availability summary is only a compatibility fallback.
   - Removed placeholder cross-modal/ordinal/innovation losses because they lacked real support/rank/calibration targets.

5. `scripts/picf_core_train.py`
   - Added OWM CLI/default/config mapping for temporal V-JEPA tokens, evidence cache, slot-JEPA, support prediction, ordinal relation, and guarded OWM loss weights.
   - Updated training log contract so launch logs expose the active OWM graph and loss configuration.
   - Window training forwards the next observed posterior into `future_targets_from_current_targets(...)` as the no-leakage JEPA teacher.
   - OWM loss metrics are backward-compatible with older/dummy loss objects while fully reporting the new loss fields on the deployed path.

6. Tests
   - Added V-JEPA `recent_maps` test.
   - Added AQR OWM temporal/PG/address test.
   - Added evidence-cache causality test.
   - Added ordinal prompt-gating/no-posterior-rewrite test.
   - Extended training-script tests to assert OWM CLI/config/loss propagation.
   - Added temporal-mode test so `last4_tokens` requests four V-JEPA temporal maps even if the legacy token count remains lower.
   - Added guarded OWM objective test so weighted OWM losses are finite and participate in the total loss.

7. `scripts/picf_owm_evidence_bundle.py`
   - Added an OWM evidence bundle exporter for run directories.
   - Reads `args.json`, `metrics.jsonl`, and `diagnostics/*`.
   - Writes `owm_evidence_bundle.json` with OWM args, latest OWM metrics, metrics tail, diagnostic artifact paths, README-to-code verifier status, and audit invariants.

8. `scripts/verify_picf_owm_contract.py`
   - Added a strict static verifier for the final README-to-code contract.
   - Checks posterior authority, temporal V-JEPA token preservation, fixed evidence cache, graph fields, AQR temporal/PG routing, cache causality, OWM debug metrics, detached next-posterior teacher targets, final loss knobs, trainer metric propagation, and evidence-bundle coverage.

## Additional Temporary Audit Documents

```text
docs/PICF_AQR_OWM_DATAFLOW_FOLLOWTHROUGH_TEMP.md
  Recursive observation -> typed evidence -> AQR -> posterior -> prediction -> action dataflow,
  including formulas and code owners.

docs/PICF_AQR_OWM_THEORY_METHOD_TEMP.md
  Theory/method document explaining why temporal V-JEPA, PG image support, posterior,
  cache, slot-JEPA, support prediction, and ordinal relation form one belief-state system.

docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md
  Script-generated recursive dataflow audit. Each node records formula, invariant,
  and source evidence from observation/carry through AQR, posterior correction,
  prediction/cache, and action.

docs/PICF_AQR_OWM_REMOTE_CALVIN_AUDIT_TEMP.md
  Remote CALVIN audit of the older 8fdb16f run. This is an old failing baseline,
  not proof that the current checkout fails.
```

## Verification

```text
python -m py_compile \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/vjepa/wrapper.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  scripts/picf_core_train.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_owm_strict_diagnose.py \
  scripts/picf_owm_evidence_bundle.py
  passed

python scripts/verify_picf_owm_contract.py --json
  passed 16/16

git diff --check
  passed

python scripts/picf_owm_strict_diagnose.py \
  --markdown-out docs/PICF_AQR_OWM_STRICT_DIAGNOSIS_TEMP.md \
  --json-out /tmp/picf_owm_strict_diagnosis.json \
  --fail-on-fail
  0 FAIL, 2 WARN, 14 PASS/INFO

python scripts/picf_owm_dataflow_trace.py \
  --markdown-out docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md \
  --json-out /tmp/picf_owm_dataflow_trace.json \
  --fail-on-fail
  passed 16/16 recursive dataflow nodes

python -m pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py \
  scripts/picf_core_train_test.py
  219 passed, 26 warnings

python -m pytest -q \
  src/openpi/picf \
  scripts/picf_owm_evidence_bundle_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_core_train_test.py \
  scripts/picf_loss_audit_test.py \
  scripts/picf_replay_windows_test.py \
  scripts/picf_resume_train_test.py \
  scripts/picf_plot_metrics_test.py \
  scripts/serve_picf_policy_test.py
  343 passed, 1 skipped, 28 warnings
```

## Remaining Guarded Items

The final graph objects and forward paths are present. The following remain guarded by zero/default-small training weights and should only be increased after diagnostics are stable:

1. `lambda_slot_jepa`
2. `lambda_support_pred`
3. `lambda_binding_consistency`

This is not a reduced deployment. It is the full graph with guarded optimization pressure.

## 2026-05-10 Strict Audit Update

This update records the post-deployment strict diagnosis pass.

### Code Fixes Applied

```text
evidence cache read:
  before:
    score = 1 / (1 + uncertainty + age)

  after:
    score = source_factor / (1 + uncertainty + age + lambda_innov * innovation)
```

The prior implementation wrote `innovation_at_write` into the cache but did not
use it when reading previous cache entries. That violated the README rule that
high-innovation cache must be downweighted. The read path and debug trust metric
now use source, age, uncertainty, and innovation consistently.

```text
default evidence_cache_read_weight:
  before: 0.0
  after:  0.05
```

The final graph now reads the previous evidence cache by default with a small
weight. This keeps the deployment complete while preserving posterior authority.

### New Strict Diagnosis Script

Added:

```text
scripts/picf_owm_strict_diagnose.py
docs/PICF_AQR_OWM_STRICT_DIAGNOSIS_TEMP.md
```

The script checks:

```text
temporal V-JEPA path
first-class PG image support
projective point/visual path
posterior precision update
previous-cache-only read and innovation gating
cache read weight
detached next-posterior slot-JEPA target
binding-consistency limitations
address-metric limitations
trainer OWM diagnostics
runtime metrics when supplied
CALVIN drift artifacts when supplied
state-only burn-in graph consistency
placeholder-loss cleanup
```

### Local Verification

```text
python scripts/verify_picf_owm_contract.py --json
  passed 16/16

python scripts/picf_owm_strict_diagnose.py \
  --markdown-out docs/PICF_AQR_OWM_STRICT_DIAGNOSIS_TEMP.md \
  --json-out /tmp/picf_owm_strict_diagnosis.json \
  --fail-on-fail
  0 FAIL, 2 WARN, 14 PASS/INFO

python scripts/picf_owm_dataflow_trace.py \
  --markdown-out docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md \
  --json-out /tmp/picf_owm_dataflow_trace.json \
  --fail-on-fail
  16/16 recursive dataflow nodes passed

python scripts/picf_owm_strict_diagnose.py \
  --metrics-jsonl /tmp/picf_remote_audit_8fdb16f/metrics.jsonl \
  --eval-dir /tmp/picf_remote_audit_8fdb16f \
  --markdown-out /tmp/picf_owm_strict_diagnosis_old_calvin.md \
  --json-out /tmp/picf_owm_strict_diagnosis_old_calvin.json
  old 8fdb16f CALVIN baseline: 3 FAIL, 2 WARN, 18 PASS/INFO

python -m pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py \
  scripts/picf_core_train_test.py
  219 passed, 26 warnings

python -m pytest -q \
  src/openpi/picf \
  scripts/picf_owm_evidence_bundle_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_core_train_test.py \
  scripts/picf_loss_audit_test.py \
  scripts/picf_replay_windows_test.py \
  scripts/picf_resume_train_test.py \
  scripts/picf_plot_metrics_test.py \
  scripts/serve_picf_policy_test.py
  343 passed, 1 skipped, 28 warnings
```

The strict audit originally found that `binding_consistency` was only a
current-step binding entropy term. It has been upgraded to a detached temporal
identity-contrast loss and is now covered by test:

```text
test_binding_consistency_uses_detached_temporal_identity_target
```

The address-drift false-positive metric has been removed. Runtime acceptance
must use `posterior_identity_switch_rate`, `posterior_recycle_rate`,
same-role support overlap, effective anchor count, and task visualizations.

The state-only burn-in path now calls `_build_aqr_anchor_graph(...)` whenever
`aqr_mapg_enabled=True`, so burn-in and suffix posterior updates share the same
measurement model. This closes the main recurrent distribution-mismatch risk
identified in the external review.

The stale `aqr_temporal_memory_tokens` knob was removed. The only active V-JEPA
temporal controls are `aqr_vjepa_temporal_mode`,
`aqr_vjepa_temporal_tokens`, and `aqr_vjepa_temporal_include_delta`.

The unused `ordinal_confidence_threshold` knob was removed. Ordinal/relation is
kept as prompt-gated diagnostic state only; it must not be treated as a trained
rank selector until a real ordinal target and confidence definition are added.

Placeholder losses were removed rather than kept at zero:

```text
removed:
  lambda_cross_modal_align
  lambda_ordinal_relation
  lambda_innovation_calib

reason:
  the implemented versions were confidence-balancing / score-spread /
  innovation-to-one surrogates, not real support/rank/calibration objectives.
```

### Remote Old-Run Diagnosis

The inspected remote training/CALVIN artifacts were produced by an older
checkpoint/code revision and cannot validate the final OWM graph.

Observed training state through step 4300:

```text
loss_total:
  decreased from about 0.167 to 0.142.

loss_action:
  decreased from about 0.079 to 0.054.

loss_pv_weak:
  decreased from about 4.58 to 2.96.

loss_anchor_pv:
  worsened from about 3.85 to 4.75.

legacy focus_pv:
  was not a valid maintained PV repair because its attention matrix lacked real
  visual-token rows. It is removed from the current code path.

loss_mapg_cycle:
  must be inspected on new runs because it now carries direct graph
  point/visual projection consistency.

loss_mapg_support_diversity:
  worsened from about 0.40 to 0.60.

loss_mapg_geometry_diversity:
  worsened from about 0.48 to 0.76.

OWM debug keys:
  absent.
```

CALVIN checkpoint-2500 debug artifacts showed:

```text
posterior pixel jump:
  mean about 12-15 px on sampled episodes.

same-role visual/point overlap:
  max overlap mean about 1.00.

anchor jump trend:
  no clear convergence across episode quartiles.
```

Strict interpretation:

```text
The old run shows action learning but poor anchor convergence.
PV weak alignment alone is insufficient.
Anchor-level PV, diversity, same-role overlap, and identity metrics must improve
before claiming anchor quality.
```

### Current Deployment Status

```text
static graph deployment:
  pass after the cache innovation-gating fix.

mathematical causality:
  pass for posterior-centered cache read/write and detached future targets.

old-run empirical anchor acceptance:
  fail / not accepted.

final OWM empirical acceptance:
  pending a fresh run from the current code with OWM debug metrics enabled.
```

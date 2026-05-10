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
  do not call the current code final until every contract object is present,
  wired into forward paths, covered by tests, and visible in debug/status.

Non-negotiable:
  posterior remains authoritative; cache and future prediction stay auxiliary.
```

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
| Training CLI/config entry | `scripts/picf_core_train.py` maps OWM config and guarded losses | deployed | CLI, defaults, `PicfCoreConfig`, `PicfTransitionLossConfig`, and log contract include OWM fields |
| Debug/diagnostics | temporal, PG, address drift, cache trust, slot-JEPA, relation diagnostics | deployed | debug emits OWM temporal/PG/cache/address/ordinal metrics; training metrics expose all OWM loss terms |
| Strict scripted diagnosis | README-to-code verifier and evidence bundle verifier snapshot | deployed | `scripts/verify_picf_owm_contract.py` checks contracts, forward wiring, losses, metrics, and bundle coverage |
| Tests | shape, no-leakage, cache causality, PG priors, temporal priors, script entry stability | deployed | PICF and script test suites pass; full repo collection blocked by external dependency imports |

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
   - Added guarded OWM losses: `slot_jepa`, `support_pred`, `binding_consistency`, `cross_modal_align`, `ordinal_relation`, and `innovation_calib`.
   - `slot_jepa` uses detached next posterior slot tokens when available; detached future visual latent summary is only a compatibility fallback.
   - `support_pred` uses detached next posterior support summary `[alpha, support_mass, contact_prob, binding_confidence]` when available; availability summary is only a compatibility fallback.
   - All high-risk OWM loss weights default to zero, matching the README guard policy while keeping the full graph deployed.

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
  scripts/picf_owm_evidence_bundle.py

python scripts/verify_picf_owm_contract.py
  15/15 contract checks passed:
    README definition/posterior authority
    temporal V-JEPA support contracts
    fixed evidence cache contracts
    graph OWM fields
    recent_maps time preservation
    OWM config defaults
    AQR temporal tokens/priors
    first-class PG priors
    cache causal order
    required OWM debug keys
    next-posterior teacher targets
    OWM loss family
    trainer next-posterior teacher threading
    trainer OWM metrics
    evidence-bundle OWM coverage

git diff --check
  passed

pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/vjepa/wrapper_test.py
  98 passed, 3 warnings

pytest -q src/openpi/picf
  184 passed, 1 skipped, 3 warnings

pytest -q scripts/picf_core_train_test.py scripts/picf_loss_audit_test.py scripts/picf_replay_windows_test.py scripts/picf_resume_train_test.py
  140 passed, 26 warnings

pytest -q scripts/picf_owm_evidence_bundle_test.py scripts/picf_core_train_test.py scripts/picf_loss_audit_test.py scripts/picf_replay_windows_test.py scripts/picf_resume_train_test.py
  142 passed, 26 warnings

pytest -q src/openpi/picf/core/training_test.py scripts/picf_core_train_test.py
  145 passed, 26 warnings

pytest -q src/openpi/picf/core/training_test.py
  23 passed, 1 warning

pytest -q src/openpi/picf scripts/picf_owm_evidence_bundle_test.py scripts/picf_core_train_test.py scripts/picf_loss_audit_test.py scripts/picf_replay_windows_test.py scripts/picf_resume_train_test.py
  326 passed, 1 skipped, 28 warnings

pytest -q scripts/verify_picf_owm_contract_test.py scripts/picf_owm_evidence_bundle_test.py
  4 passed

pytest -q scripts/verify_picf_owm_contract_test.py src/openpi/picf/core/pipeline_test.py scripts/picf_core_train_test.py scripts/picf_owm_evidence_bundle_test.py
  194 passed, 26 warnings

pytest -q src/openpi/picf scripts/picf_owm_evidence_bundle_test.py scripts/verify_picf_owm_contract_test.py scripts/picf_core_train_test.py scripts/picf_loss_audit_test.py scripts/picf_replay_windows_test.py scripts/picf_resume_train_test.py
  328 passed, 1 skipped, 28 warnings

pytest -q
  blocked during collection by environment dependency issues outside PICF:
    - src/openpi/policies/policy_test.py imports lerobot/datasets/pandas and cannot resolve pandas.core.arrays.sparse.SparseArray
    - src/openpi/training/data_loader_test.py imports lerobot/datasets/pandas and cannot resolve pandas.core.arrays.sparse.SparseArray
    - scripts/train_test.py imports wandb and cannot resolve wandb_watchdog.observers.polling
```

## Remaining Guarded Items

The final graph objects and forward paths are present. The following remain guarded by zero/default-small training weights and should only be increased after diagnostics are stable:

1. `lambda_slot_jepa`
2. `lambda_support_pred`
3. `lambda_binding_consistency`
4. `lambda_cross_modal_align`
5. `lambda_ordinal_relation`
6. `lambda_innovation_calib`

This is not a reduced deployment. It is the full graph with guarded optimization pressure.

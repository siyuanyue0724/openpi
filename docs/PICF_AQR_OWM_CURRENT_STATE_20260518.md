# PICF-AQR-OWM Current State And Cleanup Audit

Date: 2026-05-18

This document is the current cleanup/audit checkpoint for the live
PICF-AQR-OWM/MVTrack codebase. It does not replace
`src/openpi/picf/README_v2.2.md`; it is the compact operator-facing state page
that explains which modules are production-maintained, which are guarded, and
which are legacy/archive candidates.

## Verdict

The codebase is production-guarded, but not minimal.

The maintained default path is clean enough to train because high-risk branches
are gated or zero-weight by default. The repository still contains many
diagnostic, legacy, and ablation modules from the May 2026 repair sequence.
Those modules should not all be deleted at once: several are still referenced by
verifiers, experiment notes, or compatibility paths. The correct cleanup is
therefore classification plus archived entry points, not blind removal.

## Current Source Of Truth

- Primary live entry: `src/openpi/picf/README_v2.2.md`
- This cleanup status: `docs/PICF_AQR_OWM_CURRENT_STATE_20260518.md`
- Issue ledger: `docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md`
- Loss audit: `docs/PICF_AQR_OWM_LOSS_AUDIT_20260517_TEMP.md`
- SAM status: rejected/archived. Historical notes live under
  `docs/archive/picf_aqr_owm_202605/sam_rejected_20260519/`.
- AR active-anchor proposal audit: `docs/PICF_AQR_OWM_AR_ANCHOR_PROPOSAL_AUDIT_20260518_TEMP.md`
- Historical index: `docs/archive/picf_aqr_owm_202605/README.md`

Old `*_TEMP.md` files remain in place because many README links and audit
scripts point to exact paths. Treat them as historical evidence unless they are
explicitly referenced by `README_v2.2.md` or this document.

## Maintained Production Path

These modules should stay in the live path.

| Area | Status | Reason |
| --- | --- | --- |
| AQR typed memory router | Keep | Central measurement-routing contract. Replaces legacy MAPG/VL router defaults. |
| PI0.5 action path | Keep | Action generator remains unchanged in role; PICF supplies corrected belief/prefix context. |
| PaliGemma semantic/image support | Keep | First-class typed semantic/image evidence. Current diagnostics indicate trainable PG is preferable to frozen PG for action/semantic alignment. |
| V-JEPA temporal support | Keep | Static/wrist temporal token evidence is first-class; dense tokens are not replaced by proposals. |
| Sonata point support | Keep | Geometry evidence and contact/object grounding substrate. |
| AnyTouch/tactile support | Keep | Contact-side evidence; must remain soft spatial evidence, not hard object truth. |
| Same-role active/context/reserve routing | Keep | Separates active object files from context/reserve capacity; raw overlap alone is not the production health criterion. |
| Posterior owner active gate | Keep | Prevents inactive/reserve files from entering action as if they were active objects. |
| Posterior file competition | Keep | Current no-object/file competition mechanism that prevents many same-role files from all owning one observation. |
| Posterior birth competition/transport | Keep | Prevents many inactive files from being born on the same owner. |
| Cache residual gate | Keep | `evidence_cache_read_weight` is now a real residual scale, not a softmax constant. |
| Pairwise binding subspace | Keep | Structural binding term inspired by same-object probing; not an extra online loss. |
| Anchor overlays and diagnostics | Keep | Required to detect object-binding failures that scalar losses hide. |

## Disabled Prototype: AR Active-Anchor Proposal / VCAP

The current maintained path remains the fixed-capacity AQR query bank plus
active/context/reserve routing and posterior file/birth competition. A
disabled-by-default VCAP runtime prototype is now present and documented in
`docs/PICF_AQR_OWM_AR_ANCHOR_PROPOSAL_AUDIT_20260518_TEMP.md`.

The candidate is acceptable only as:

```text
variable-cardinality measurement proposal
  -> padded AQR read
  -> posterior matching/correction
  -> active/context/reserve action exposure
```

It is not acceptable as:

```text
generated anchors replace posterior truth
generated anchors hard-prune dense typed memory
count-loss-only pressure decides object count
proposal index becomes persistent object id
```

This is not a current production default. It is a guarded runtime prototype:
enabling it changes only active query initialization, preserves dense typed
memory, and still requires the unchanged AQR/posterior stack to decide belief.

Second-pass audit note: the candidate is rejected unless it preserves posterior
authority, padded AQR tensor contracts, dense typed memory, and the current
active/context/reserve plus file/birth competition layers. It is not a license
to add a count-only anchor loss or to turn proposal order into persistent
identity.

The implemented plan is `VCAP`: a disabled-by-default variable-cardinality
active proposal layer with config/contracts, padded AQR integration,
coverage/duplicate/no-object diagnostics, transition-loss hooks,
action-gradient guard, metrics, and executable contract audits delivered
together.  A stop-token-only or count-loss-only implementation remains
classified as an invalid half-measure.

## Guarded Or Data-Dependent Modules

These modules should stay available, but they are not proof of production
behavior unless their data or loss weights are active.

| Area | Default | Rule |
| --- | --- | --- |
| Tracklet typed memory | Enabled in config, no-op without sidecar tensors | Keep. It is the right long-term temporal correspondence channel. Require sidecar coverage/quality checks before production claims. |
| Generic proposal sidecar memory | Disabled | Keep only for inspected contact/task/tracklet-aware sidecars. Blind automatic SAM is rejected and archived. |
| Offline IsSameObject probe | Offline only | Keep as audit. It validates the binding subspace; it must not become an online training loss without a separate design review. |
| `slot_jepa` | Loss weight 0 | Keep hook; do not enable until matched, masked targets are empirically stable. |
| `support_pred` | Loss weight 0 | Keep hook; do not enable as production pressure until support targets are calibrated. |
| `binding_consistency` | Loss weight 0 | Keep hook; index-style assumptions remain risky under slot permutation. |
| `aqr_denoising` | Loss weight 0 | Keep hook; training-only auxiliary, not current production pressure. |
| Ordinal/relation state | Diagnostic/weak | Keep diagnostic; not a rank-supervised fourth-object solver. |

## Legacy / Off-By-Default Modules

These should remain off unless an explicit ablation is being run.

| Area | Current handling | Cleanup recommendation |
| --- | --- | --- |
| Legacy local refinement | `legacy_local_refinement_opt_in=False`, `local_refinement_enabled=False`, weight 0 | Future removal candidate. It repeatedly failed to solve the core overlap/action interaction and can confuse the graph story. |
| Blind automatic SAM proposal memory | rejected; legacy override required | Do not use in current training. Historical reproduction requires archived scripts and `--allow-legacy-blind-sam-sidecar`. |
| Role-wise/local candidate competition variants | Historical experiments | Keep only as TEMP evidence unless revalidated by current metrics. |
| Deterministic coverage seed variants | Historical experiments | Do not restore as production default; they were scaffold-like and did not solve semantic object ownership. |
| Ad-hoc root run scripts | Archived to `scripts/experiments/picf_aqr_owm_202605_archive/` | Do not keep transient cloud launch scripts in repo root. |

## Dataflow Follow-Through

Current training flow:

```text
CALVIN sample
  -> optional sidecar loader
  -> PicfObservation
  -> typed token fields
     - PaliGemma text/image
     - V-JEPA static/wrist temporal
     - Sonata point
     - AnyTouch tactile
     - previous posterior
     - cache
     - optional tracklet/proposal
  -> AQR graph read
  -> owner/active/context/reserve routing
  -> posterior update and file competition
  -> action prefix / PI0.5 action loss
  -> guarded auxiliary losses
```

Mathematical invariant:

```math
measurement\ quality \rightarrow posterior\ belief \rightarrow action\ exposure
```

Do not collapse these three values into one "confidence" scalar. Anchor graph
confidence is measurement quality. Posterior alpha is belief activity. Action
exposure is active/context/reserve routing.

## Loss Health Interpretation

Current scalar losses are not equally authoritative.

| Metric | Interpretation |
| --- | --- |
| `loss_action_default_equiv` | Comparable action-loss view for historical 4-22 ablation comparisons. |
| `loss_action_active7` | Active local training loss scale; not directly comparable to old default loss. |
| `loss_anchor_pv` | Weak anchor-to-PV alignment pressure. Short-run increases can indicate semantic/geometry owner mismatch, not necessarily pure optimization failure. |
| Raw same-role overlap | Diagnostic only. High raw overlap is tolerable if active/core overlap remains low and overlays are plausible. |
| Active same-role overlap | Production health metric. Sustained high values indicate active object files are duplicating owners. |
| Object-core overlap | Better than raw visual overlap for object ownership; use with overlay evidence. |
| `posterior_recycle_rate` | Should not saturate high or collapse to zero without stable active slots. Interpret with stable-slot fraction and overlays. |

The concentrated current issue list is maintained in
`docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md` under
`2026-05-18 Concentrated Current Issue Summary`.  The short version is:

```text
The current P0 issue is not one missing code hook.  It is rebound of binding
health after initially healthy steps:
  - loss_anchor_pv rising while loss_pv_weak falls;
  - raw same-role support/object-core overlap returning high;
  - active same-role support/object-core overlap sometimes rising;
  - posterior identity/swap/recycle remaining unstable;
  - overlays needing active_only verification of task-object ownership.

Historical attempts show that action-off, stronger support competition, local
refinement, blind SAM, posterior file competition, birth transport, and
downstream gating each answer a different subproblem.  The remaining A7 work
must judge the whole bundle rather than a single loss.
```

## Current Open Boundaries

These are not "unfixed bugs"; they are either data-dependent or require long-run
behavior evidence.

1. Tracklet sidecars: code path exists, but production claims require generated
   sidecars and coverage/quality checks.
   The 2026-05-18 A5 generation failure was traced to corrupted compressed
   tracklet `.npz` files during resume. `scripts/picf_tracklet_sidecar_precompute.py`
   now treats `zlib.error` like `BadZipFile`: the corrupt file is removed and
   regenerated instead of killing the whole shard.
   2026-05-18 sidecar pause decision: the generated tracklet files are readable
   and visualization samples show valid static/gripper points, but the only
   full-sidecar training diagnostic so far proved dataflow, not behavior
   improvement. The closest run,
   `picf_a7_full_sidecar_anchoronly_diag300_20260517`, had nonzero
   `owm_tracklet_tokens` but also residual proposal tokens and was
   anchor-only/no-action/frozen-PaliGemma, so it is not a fair proof that
   tracklet sidecars outperform the no-sidecar trainable-PaliGemma recipe.
   Current recommendation: pause dataset-scale sidecar generation, keep the
   code path and partial artifacts, and only resume when running a matched
   A/B test:
     A. no sidecar;
     B. tracklet-only sidecar with proposal memory off;
     C. prompted/contact-reranked proposal sidecar if that data exists.
   All three must share trainable scope, action settings, PaliGemma trainability,
   unroll/burnin, learning rate, and checkpoint source.
2. SAM/proposal sidecars: blind SAM was demoted; prompted/contact/reranked
   proposals are future sidecar work, not current default.
3. Fourth-object/ordinal grounding: no hard rank labels exist. Current ordinal
   path is diagnostic/weak only.
4. Long-run acceptance: short diagnostics can reject bad configurations, but
   only a 30k-style run plus overlays/CALVIN metrics can prove sustained action
   and binding behavior.

## Code Cleanliness Conclusion

The live training defaults are reasonably clean. The repository layout is not.

Keep now:

- `config.py`, `contracts.py`, `pipeline.py`, `training.py`, `picf_core_train.py`
  current production switches and guarded hooks.
- Audit scripts that enforce dataflow invariants.
- Sidecar precompute/load scripts, because tracklets are a planned data upgrade.

Do not remove now:

- Tracklet/proposal fields and loader paths.
- IsSameObject probe code.
- Legacy MAPG/VL-router compatibility paths, unless a dedicated deprecation
  branch also updates tests and docs.

Candidate future removals:

- Legacy local refinement runtime branch.
- Stale TEMP docs after all references are migrated.
- Historical ablation-specific script options that no longer have tests or
  current README coverage.

## Sidecar Pause / Restart Notes

As of 2026-05-18 A5 had these tracklet-generation tmux sessions still alive:

```text
tracklet_resume_shard0_20260517
tracklet_resume_shard1_fix_20260518
tracklet_resume_shard4_20260517
tracklet_resume_shard6_20260517
tracklet_resume_shard7_fix_20260518
```

Observed state:

```text
GPU usage: 0 MiB / 0%.
Completed manifests: shards 2, 3, 5 plus a partial/global manifest.
Generated training sidecars: about 592k episode files at the inspection time.
Shards 0 and 4 were still producing saved frames.
Shards 1, 6, and 7 had progressed but were saving 0 frames under the current
proposal-required filter, which suggests sparse seed/proposal coverage rather
than a useful high-quality data stream.
```

Safe pause command on A5:

```bash
tmux kill-session -t tracklet_resume_shard0_20260517 || true
tmux kill-session -t tracklet_resume_shard1_fix_20260518 || true
tmux kill-session -t tracklet_resume_shard4_20260517 || true
tmux kill-session -t tracklet_resume_shard6_20260517 || true
tmux kill-session -t tracklet_resume_shard7_fix_20260518 || true
```

Do not delete the partial sidecar roots unless space pressure requires it:

```text
/mnt/picf_sidecars/tracklets_samseed_stride16_w15_phase0_20260517
/mnt/picf_sidecar_previews/tracklets_20260518
```

Restart rule:

```text
Only resume generation for a matched sidecar-only ablation, and prefer
tracklet-only training with proposal memory disabled. Blind SAM proposal memory
should remain off by default because its boxes were visually noisy and did not
establish task-object posterior binding.
```

## Verification Commands

Use this command block after cleanup edits:

```bash
python -m py_compile \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  scripts/picf_core_train.py \
  scripts/serve_picf_policy.py \
  scripts/verify_picf_owm_contract.py

python scripts/verify_picf_owm_contract.py
python scripts/picf_owm_strict_diagnose.py --fail-on-fail
python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
```

If optional dependencies such as `transformers` are missing, full
`pipeline_test` / `training_test` collection can still fail at import time.
That is an environment limitation, not evidence that this cleanup changed the
runtime contract.

## Verification Results From This Cleanup Pass

Executed on 2026-05-18 from `/home/siyuanyue/Documents/openpi`.

| Check | Result |
| --- | --- |
| `python -m py_compile` on core config/contracts/pipeline/training and train/serve/verifier scripts | PASS |
| `uv run python -m py_compile` on the same files | PASS |
| `bash -n scripts/experiments/picf_aqr_owm_202605_archive/*.sh` | PASS |
| `python scripts/verify_picf_owm_contract.py` | PASS, 48/48 |
| `python scripts/picf_owm_strict_diagnose.py --fail-on-fail` | PASS with expected WARNs for missing runtime metrics/CALVIN artifact inputs |
| `python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail` | PASS |
| `python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail` | PASS, 9/9 |
| `PYTHONPATH=. python scripts/picf_anchor_run_diagnostic_report_test.py` | PASS |
| Legacy blind-SAM audit | Archived; no longer part of the current verification surface. |
| `python scripts/picf_owm_dataflow_trace.py --fail-on-fail` | PASS |
| `python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail` | PASS |
| `uv run python scripts/picf_posterior_file_competition_audit.py --fail-on-fail` | PASS |
| `python scripts/picf_posterior_birth_transport_audit.py --fail-on-fail` | PASS |
| `uv run` VCAP enabled-path tensor smoke | PASS; padded query shape preserved, hard/soft active probabilities produced, unexplained evidence finalized |

One direct `python scripts/picf_posterior_file_competition_audit.py` invocation
failed because the bare shell interpreter did not have `torch`; the same audit
passed under `uv run`, which is the expected project runtime. This is recorded
as an environment issue, not a code/dataflow failure.

## A7 VCAP Guarded 30k Launch: 2026-05-18 04:39 CST

Remote:

```text
A7 host: 36.139.225.68:28060
session: picf_a7_vcap30k_20260518
log: /mnt/picf_run_logs/picf_a7_vcap_guarded_cotrain_u2b1_30000_20260518.log
ckpt dir: /mnt/checkpoints/picf_core/picf_core/picf_a7_vcap_guarded_cotrain_u2b1_30000_20260518
```

Training contract:

```text
num_steps=30000
world_size=2
effective_global_batch=2
unroll_steps=2
burnin_steps=1
burnin_mode=state_only
save_interval=2500
keep_last_checkpoints=3
log_interval=50
anchor_overlay_interval=100
anchor_overlay_dump_signatures=True
```

Backbone contract:

```text
Sonata: frozen
V-JEPA: frozen
AnyTouch: frozen
PaliGemma: trainable
PICF/connectors/action path: trainable
```

VCAP contract for this run:

```text
vcap_enabled=True
vcap_max_active=12
vcap_min_active=1
vcap_stop_threshold=0.50
vcap_action_grad_scale=0.0
lambda_vcap_unexplained=0.01
lambda_vcap_duplicate=0.01
lambda_vcap_count=0.0001
lambda_vcap_continuity=0.001
```

Pre-launch issue found and fixed:

```text
First launch failed at global_step=1 during backward with an autograd in-place
version error on a [16] tensor. Root cause was VCAP active-prior integration
using slice writes on tensors participating in downstream graph construction.

Fix:
  - active_hard min/max guards now use torch.where instead of slice assignment.
  - proposal active additions to anchor_scores/anchor_conf now use padded
    functional tensors instead of in-place slice updates.
  - VCAP assignment uses torch.eye rather than fill_diagonal_.
```

Verification after the fix:

```text
remote py_compile src/openpi/picf/core/pipeline.py: PASS
remote verify_picf_owm_contract.py: PASS
remote same-command 2-step backward smoke: PASS
```

Observed at launch:

```text
30k tmux session alive.
GPU memory/utilization active on both A100s.
Startup log confirms PaliGemma trainable and Sonata/V-JEPA/AnyTouch frozen.
Run passed the first real backward steps after the VCAP in-place fix.
```

Tail command:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/picf_run_logs/picf_a7_vcap_guarded_cotrain_u2b1_30000_20260518.log
```

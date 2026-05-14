# PICF v2.2

Date: 2026-05-06
Repo: `/home/siyuanyue/Documents/openpi`
Status: current local v2.2 architecture record after the one-shot
action/control contract rewrite, the frozen-perception bring-up, the
VL-router supervised grounding rollout, the MAPG-v0 evidence pass, and the
AQR-MAPG direct-final graph replacement.

2026-05-09 update: the maintained direct-final graph path is now **AQR-MAPG**
and is the default PICF graph path (`aqr_mapg_enabled=True`,
`mapg_enabled=False`, `vl_anchor_router_enabled=False`). AQR-MAPG replaces
MAPG-v0 candidate-prior graph construction with learned physical/task anchor
queries over typed support memory, while reusing the same
observation/task/posterior/control graph consumer contract and the same PI0.5
action path. Use `--no-aqr-mapg-enabled` only for explicit ablations or legacy
compatibility tests.

2026-05-10 OWM audit update: use
[`docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md)
as the canonical deployment contract for the object-addressable predictive
belief-state router. The legacy `focus_pv` loss has been removed because its
attention source did not contain real visual-token rows. The active
geometry-consistent repair is `lambda_mapg_cycle=0.02`, which directly aligns
`graph.visual_priors` and `graph.point_priors` through
`projective_compatibility`, plus the low-weight anti-collapse guard
`lambda_mapg_support_diversity=0.01`.

2026-05-10 strict cleanup update: AQR state-only burn-in now uses the AQR
measurement graph when `aqr_mapg_enabled=True`, so burn-in and suffix posterior
updates no longer use different graph builders. The stale
`aqr_temporal_memory_tokens` knob and the misleading
`posterior_address_drift_mean` acceptance metric have been removed. The unused
`ordinal_confidence_threshold` knob was also removed; ordinal/relation remains
a prompt-gated diagnostic until a real rank target is implemented. Placeholder
losses for cross-modal confidence balancing, ordinal score spread, and
innovation-to-one calibration were removed; only `slot_jepa`, `support_pred`,
and `binding_consistency` remain as guarded OWM training losses.

2026-05-10 default-profile update: the CLI now defaults to the latest
PICF-AQR-OWM profile, so a normal PICF training command no longer needs a
separate AQR flag bundle. The default semantic mode is `paligemma`; the
PaliGemma heatmap head remains disabled; PaliGemma image support and explicit
V-JEPA temporal tokens are enabled; evidence cache read uses a small
posterior-grounded weight (`0.05`); and high-risk predictive/identity losses
stay zero-weight until their diagnostics justify activation. All train-time
loss defaults are sourced from `PicfTransitionLossConfig`, not duplicated in
the CLI parser.

Serving compatibility is intentionally conservative: checkpoints whose
metadata predates the OWM default and records `semantic_mode=zero` are not
silently promoted into `aqr_mapg_enabled=True` at serve time. New training uses
the OWM default; old zero-semantic checkpoints keep their recorded graph path.

2026-05-10 MVTrack runtime-c update: use
[`docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md)
as the canonical `PICF-AQR-OWM-MVTrack` contract and runtime wiring record.
This is not a replacement for the maintained v26 baseline; it is the complete
v2 architecture plan plus the deployed code pass for static+wrist V-JEPA typed
memory, optional tracklet/proposal episode-field threading, support-signature
identity binding, posterior-address-first cache retrieval, innovation-gated
address inertia based on current measurement mismatch rather than predictive
cache residuals, archived legacy latent local refinement, matched predictive losses,
permutation-tolerant binding consistency, gated weak ordinal diagnostics, and
training-only support denoising. It explicitly separates code-level runtime
completion from behavior-level CALVIN/video acceptance; code-level runtime completion
does not imply CALVIN/video behavior completion. SAM/DINO proposal generation is
intentionally not part of this maintained pass; proposal tensors are consumed
only if an upstream source provides them.

2026-05-13 experiment-gate update: the live anchor diagnosis is tracked in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).
The A5 offline same-object probe shows a strong geometry-driven weak
same-object signal (`combined_auc=0.893`) but weak exported visual/point/support
signature separability (`AUC≈0.52`) and high same-role duplicate-candidate
fraction (`0.827`). A7 is healthier than the earlier 0.99-overlap collapse run,
but as of step 970 stable slot coverage remains low (`stable_slot_fraction≈0.11`).
The next experiment gate is therefore read-only coverage/competition diagnosis
and token-level IsSameObject probing, not stronger cache, address inertia, JEPA,
action pressure, or reintroduced hand-coded role/coverage heuristics. This is a
probe-level result, not behavior acceptance.

2026-05-13 signature-probe update: commit `c4ac7b3` adds read-only
`anchor_debug` export of observation/posterior support and binding signatures,
plus signature-aware same-object probe metrics. This does not change train or
serve behavior unless anchor debug export is enabled. The active A5 diagnostic
is recorded in the experiment report and tests whether the deployed
`binding_signature_proj` subspace carries same-object information before any
new loss or identity-inertia change is considered.

2026-05-13 deployment update: A7 reached the planned 1050-step endpoint with
controlled support overlap (`aqr_same_role_support_overlap_max≈0.294`) but low
stable-slot coverage (`posterior_stable_slot_fraction≈0.122`). It is therefore
archived as partial non-collapse, not accepted as a long-run line. A5 and A7 are
now both assigned to read-only signature-level same-object probes:
A5 base `/mnt/calvin_eval_logs/picf_a5_signature_probe_c4ac7b3_retry_20260513_012306`;
A7 base `/mnt/calvin_eval_logs/picf_a7_signature_probe_c4ac7b3_now_20260513_013143`.
The 12-hour gate is documented in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-13 A5 signature-probe result: after a Python-3.8 compatibility fix in
the probe script, A5 reports `binding_signature_cos_auc=0.976` but
`duplicate_candidate_fraction_within_frame=0.848`. This answers the first
object-binding question: the projected pairwise binding subspace is decodable.
The remaining failure is assignment/coverage reuse of candidates, not absence
of same-object signal. A7 endpoint signature probing is the next discriminator.

2026-05-13 A7 signature-probe result: A7 agrees with A5. The A7 endpoint reports
`binding_signature_cos_auc=0.964` and
`duplicate_candidate_fraction_within_frame=0.854`. This rules out the hypothesis
that A7 destroyed the pairwise binding subspace. The remaining bottleneck is
that AQR assignment/local candidate selection does not distribute same-role
slots over distinct candidates even though a same-object binding signal is
available. The next repair must audit binding-signature scale and local top-k
candidate selection before any new loss, stronger cache/address inertia, or
slot-JEPA/support-prediction pressure is enabled.

2026-05-13 root-cause recycle update: the fresh A5/A7 local-signature causal
check invalidates stronger local reranking as the next default. A5
`local_refinement_binding_weight=0.0` reached step 150 with
`aqr_same_role_support_overlap_max≈0.289`, `local_jaccard≈0.129`, and healthy
effective anchors, while A7 `local_refinement_binding_weight=0.25` reached
`posterior_recycle_rate≈0.995`, near-zero address update, and lower effective
anchor count by step 125. This points to recycle-gate scale sensitivity, not
absence of a pairwise binding subspace. The maintained repair is
`recycle_normalize_residual_summary=True`: normalize the dustbin residual
summary before `recycle_head` so reset probability depends on evidence
direction/context, not unbounded residual magnitude. This is a belief-filter
trust-gate fix, not a new loss or heuristic ownership rule. A7 is assigned to
the first fresh normalized-recycle run; A5 remains the no-rerank control until
its current 300-step check finishes. Details and tail commands are in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-13 A7 normalized-recycle completion: the fresh 300-step diagnostic
reached `posterior_recycle_rate=0.540`, `posterior_recycle_logit_mean=0.161`,
`posterior_address_update_rate_mean=0.0196`,
`aqr_effective_anchor_count=23.25`, and
`aqr_same_role_support_overlap_max=0.388` on the final row. The tail-5 averages
were `recycle_rate=0.537`, `address_update=0.0197`, and
`same_role_support_overlap=0.495`. This is sufficient to archive the
recycle-saturation failure as fixed by normalized recycle inputs. It does not
close behavior acceptance or fine-instance grounding. The 300-step checkpoint
payload was deleted to free `/mnt`; metrics, args, and train logs are retained
in the experiment report paths.

2026-05-13 next-stage attribution: do not reintroduce the previously rejected
hard role competition or deterministic coverage-seeded local proposal. The next
two-hour A5/A7 matrix isolates the existing local refinement residual instead:
A7 disables local refinement, while A5 keeps the same normalized-recycle
contract with `local_refinement_weight=0.05`. This tests whether the current
typed-memory local refiner is necessary, too strong, or not the active
bottleneck. The plan is recorded in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-13 local-refinement attribution midpoint: both A5 and A7 reached
step 175. A7 local-off has `loss_total=0.771`,
`loss_action_default_equiv=0.082`, `same_role_support_overlap=0.563`,
`posterior_recycle_rate=0.515`, and no gradient clipping. A5 local-light has
slightly lower `loss_total=0.765` and `loss_anchor_pv=2.333`, but higher
`posterior_recycle_rate=0.540`, higher identity switch, lower address update,
and one fixed-threshold clipping event from `preclip_grad_norm=9.48`. This
does not yet decide the attribution question. It does show that local
refinement is not required for early non-collapse and may add gradient/recycle
pressure. Both runs should continue to step 300 before finalizing the next
maintained profile.

2026-05-13 local-refinement archive decision: local refinement is no longer a
production-default branch. It is now a legacy ablation path that requires all
of the following to be non-default: `--legacy-local-refinement-opt-in`,
`--local-refinement-enabled`, a positive `--local-refinement-topk`, and a
positive `--local-refinement-weight`. The default maintained profile uses
`legacy_local_refinement_opt_in=False`, `local_refinement_enabled=False`,
`local_refinement_topk=0`, and `local_refinement_weight=0.0`. The rationale is
mathematical rather than cosmetic: normalized recycle input is the root
belief-filter repair, while local top-k reread adds a second residual evidence
path whose short probes showed extra recycle/gradient pressure without being
necessary for early non-collapse. The code remains only so archived experiments
can be reproduced; future cleanup may remove it entirely if long-run evidence
does not justify keeping the branch.

The completed 300-step A5/A7 attribution confirms this cleanup direction.
A5 local-light achieved slightly better short-window alignment
(`loss_total=0.723`, `same_role_support_overlap=0.369`) but had worse
belief-filter trust (`posterior_recycle_rate=0.536`,
`posterior_address_update_rate=0.019`) and one earlier clipping event. A7
local-off achieved nearly identical action scale (`loss_action_default_equiv=0.066`)
with cleaner recycle/address dynamics (`posterior_recycle_rate=0.503`,
`posterior_address_update_rate=0.022`) and no clipping. Therefore local
refinement remains reproducible but is archived out of the production default.

2026-05-13 recycle-normalization closure: the next two-machine check isolates
the trust-gate normalization family rather than adding modules. A7 runs the
production candidate `recycle_residual_norm_mode=layernorm`; A5 runs the
conservative ablation `recycle_residual_norm_mode=rmsnorm`. Both keep local
refinement archived/off, action-prefix stop-gradient on, and high-risk
predictive/denoising losses at zero. Quantile normalization is intentionally
not used in forward because it is a distribution-level rank transform that can
collapse extreme evidence and introduce batch/history dependence; it remains
only an offline diagnostic option.

2026-05-13 recycle-normalization closure result: both 300-step closure runs
completed cleanly. A7 LayerNorm ended with
`loss_action_default_equiv=0.0653`,
`aqr_same_role_support_overlap_max=0.3167`,
`posterior_recycle_rate=0.5056`,
`posterior_address_update_rate_mean=0.0213`, stable-slot switch `0.0`, and no
gradient clipping; its tail-5 recycle/overlap were `0.5104` and `0.4731`. A5
RMSNorm ended with `loss_action_default_equiv=0.0651`,
`aqr_same_role_support_overlap_max=0.2863`,
`posterior_recycle_rate=0.5187`,
`posterior_address_update_rate_mean=0.0207`, stable-slot switch `0.0`, and no
gradient clipping; its tail-5 recycle/overlap were `0.5212` and `0.4422`.
RMSNorm is a healthy ablation but does not decisively improve identity
stability. The maintained default remains
`recycle_residual_norm_mode=layernorm`, with local refinement archived/off and
high-risk predictive/denoising losses still zero. This closes the diagnosed
recycle/address failure chain enough to move to a longer guarded production run;
it does not replace CALVIN/video/anchor-overlay behavior acceptance.

2026-05-13 30k guarded long-run launch contract: the next maintained training
run is a fresh 30000-step production-candidate run, not a direct resume from
any April/May diagnostic checkpoint. The maintained fast/stable recurrent
profile is `burnin_steps=4`, `burnin_mode=state_only`, `unroll_steps=1`.
Do not replace it with direct `unroll_steps=2` for this acceptance run: the
recent diagnostics found that the burn-in state path provides the necessary
posterior context while keeping the suffix transition fast enough for 2x40GB.
If "warmup" refers to the optimizer schedule, keep the established
`warmup_steps=100`; if it refers to recurrent burn-in, do not use 1 for the
main run. A7 is the production candidate with frozen Sonata/V-JEPA/AnyTouch,
trainable PICF/PI0.5/PaliGemma, `semantic_lr_scale=0.25`,
`picf_action_prefix_stopgrad=True`, `recycle_residual_norm_mode=layernorm`,
legacy local refinement archived/off, cache residual read at `0.05`, and
slot-JEPA/support-pred/binding-consistency/denoising losses at zero. A5 is a
conservative long-test control with the same contract but lower semantic LR
(`semantic_lr_scale=0.1`) to check whether PaliGemma cotrain pressure affects
anchor/recycle health. Both runs use `save_interval=2500` and
`keep_last_checkpoints=3`.

## Quick Navigation

- [`README.md`](/home/siyuanyue/Documents/openpi/README.md)
  Repo-level entry point.
- [`src/openpi/picf/README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
  PICF entry pointer that routes readers to the current and archived PICF docs.
- [`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
  Current live PICF v2.2 architecture and deployment record.
- [`src/openpi/picf/README_PI05_PARITY_AUDIT.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_PI05_PARITY_AUDIT.md)
  Local code-level audit comparing reference PI0.5 / PI0.5+Sonata dataflow
  against current PICF enabled and PI0.5-only ablated modes.
- [`src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md)
  Design record for the 2x40GB frozen-perception profile and geometry-safe
  augmentation policy. Use this before freezing V-JEPA/Sonata/AnyTouch or
  enabling any train-time augmentation in full PICF.
  The live CLI now exposes `--perception-finetune-mode auto|full|frozen`,
  `--visual-feature-mode auto|hierarchical|final`,
  `--picf-augmentation-mode off|photometric|multimodal_geometry`, and
  `--picf-photometric-strength conservative|reference`; the default remains
  `auto/off`.
- [`src/openpi/picf/README_VL_GUIDED_ANCHOR_ROUTER.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_VL_GUIDED_ANCHOR_ROUTER.md)
  Staged plan and implementation record for a PaliGemma-guided 2D-to-3D anchor
  prior router. The current code keeps the router disabled by default, but when
  explicitly enabled it builds VL grounding state and consumes it through
  role-aware observation-anchor soft seeds/biases, task-readout gated point
  prior fusion, and posterior-binding soft overlap bias. Heatmap/keypose,
  point-consistency, and anchor-diversity supervision terms are implemented and
  logged as default-zero `--lambda-vl-*` loss knobs; they are not active unless
  explicitly enabled. The live implementation now records the
  top-border scene-anchor diagnostic fix: world-frame point coordinates are
  split from model-frame point coordinates, scene/object slots use a
  projective candidate mask, PaliGemma heatmaps are mapped through
  `PaliGemmaViewTransform` instead of naive resize when transform metadata is
  available, and CALVIN debug export includes VL heatmap and point-prior
  summaries. The live trainer
  exposes this through `--vl-anchor-router-enabled` plus the `--vl-*` gate,
  radius, temperature, visible-mass, and bias-clip controls; ablated PI0.5 mode
  forces the router off.
  The current long-run-safe contract is recorded there: VL task/interaction
  lift uses a strict scene candidate mask with no global fallback, fallback
  global rows are coverage-only, and role-0 observation anchors keep their
  local/proprio/tactile seed path without static-camera VL seed or point-bias.
  Its Section 6.1 records the maintained `30000` step / `5000` checkpoint /
  `unroll_steps=2` VL-router launch template.
  Use its Current Mathematical Guardrails and Verification Commands sections
  before enabling any router stage beyond the default-off substrate.
- [`src/openpi/picf/README_MAPG_PICF.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_MAPG_PICF.md)
  Current live implementation record for MAPG-PICF: modality-optional anchor
  prior graph routing over PaliGemma, V-JEPA, Sonata, AnyTouch, and posterior
  supports. This is now the graph-level layer above the point-centric
  VL-guided router. It records the shared PaliGemma grounding builder,
  confidence-gated finite-message-passing math, role-constrained
  Sinkhorn-style slot assignment, direct tactile/posterior-to-visual support,
  PaliGemma-to-visual grounding that stays live without point support, explicit
  visual-native task readout, masked graph losses, unified MAPG anchor
  separation over assignment/support/embedding/geometry objects, state objects,
  live dataflow into observation anchors / task readout / posterior binding /
  conditioned control, CLI flags, diagnostics, verification commands, and the maintained
  `30000` step / `5000` checkpoint / `unroll_steps=2` MAPG launch template.
- [`docs/AQR_MAPG_DIRECT_FINAL_DEPLOYMENT_README.md`](/home/siyuanyue/Documents/openpi/docs/AQR_MAPG_DIRECT_FINAL_DEPLOYMENT_README.md)
  Direct-final replacement contract for the graph layer. The live code now
  implements the maintained AQR-MAPG contract for this training line: learned
  physical/task anchor queries, typed visual/point/tactile/posterior support
  reads, PaliGemma semantic conditioning for task queries, production-default
  PaliGemma heatmap/grounding disabled, PaliGemma image-token support enabled
  as a visual-semantic bridge into the V-JEPA grid, support-level Sinkhorn
  normalization, row-specific downstream slot assignment, point-optional graph
  fallback guards, and default-off CLI flags. This is not a partial MAPG-v0
  deployment; legacy `--mapg-enabled` candidate-prior graph construction must
  stay disabled for the direct-final AQR run.
- [`docs/AQR_MAPG_HANDOFF_README.md`](/home/siyuanyue/Documents/openpi/docs/AQR_MAPG_HANDOFF_README.md)
  Short handoff index for a new engineer or researcher. It lists the
  authoritative README reading order, current AQR-MAPG design, PaliGemma
  contract, trainable/frozen profile, cloud log locations, main code entry
  points, and known limitations such as fine local refinement for tiny adjacent
  objects.
- [`docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md)
  Final PICF-AQR-OWM architecture and concrete deployment contract. This is the
  required handoff document for turning the current AQR/PICF code into the
  full object-addressable predictive belief-state router: it audits the current
  code file-by-file, defines the final dataflow, lists the exact contract/state
  extensions, specifies implementation changes for V-JEPA temporal support,
  PaliGemma image support, posterior address/content, evidence cache,
  slot-level JEPA, support prediction, ordinal grounding, and diagnostics,
  resolves the proposal point-by-point as adopted/guarded/rejected, includes a
  file-by-file code audit/deployment map, and states the final Definition of
  Done. It records the direct-to-final OWM target with hard guards and now
  requires `python scripts/verify_picf_owm_contract.py` plus the OWM evidence
  bundle for strict README-to-code diagnosis. It is not a conceptual appendix;
  it is the engineering deployment blueprint for the final architecture.
- [`docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md)
  Architecture contract and runtime-c wiring record for PICF-AQR-OWM-MVTrack:
  static+wrist V-JEPA typed multiview support, support-signature identity
  binding, posterior-address-first cache retrieval, optional tracklet/proposal
  data ingestion, local refinement, training-only support denoising, matched
  predictive losses, and gated weak ordinal targets. It
  records the math, code touchpoints, paper support, verification gates, and
  CALVIN/video acceptance boundary before behavior-level completion can be
  claimed.
- [`docs/PICF_AQR_OWM_RECYCLE_DIAGNOSIS_PLAN_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_RECYCLE_DIAGNOSIS_PLAN_TEMP.md)
  Active 2026-05-11 diagnosis plan for the staged anchor/action runs. It
  records the R0/S1/S2/S3/S4/S5/D1 experiment matrix, explains why
  `posterior_recycle_rate=1.0` is an unresolved posterior-identity issue even
  when same-role support overlap improves, and lists the recycle logits,
  dustbin, support-mass, and staged-action ablations that must be run before
  predictive OWM auxiliary losses are enabled.
- [`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md)
  Live 2026-05-11 experiment report for the current A5/A7 cloud runs. It
  records the active A7 30k launch contract, the completed A5 prefix-stopgrad
  diagnosis, historical direct-action/recycle ablations, loss-scale
  interpretation, the paper-derived object-binding audit, and the 2026-05-12
  burn-in/support-collapse matrix. Use this document to distinguish code-level
  repairs and short-run evidence from still-pending behavior acceptance, and to
  avoid treating scalar action-loss improvement as anchor-health proof when
  same-role support overlap remains high.
- [`docs/PICF_AQR_OWM_DEPLOYMENT_STATUS_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_DEPLOYMENT_STATUS_TEMP.md)
  Temporary live deployment ledger for the OWM implementation. Use this while
  reviewing the current branch because it records which final README contract
  items are already wired into code, which loss hooks remain guarded, and which
  tests have passed.
- [`docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md)
  Script-generated recursive dataflow audit. Use this to verify the exact
  observation/carry -> typed evidence -> AQR -> posterior -> prediction/cache
  -> action formulas and code evidence.
- [`docs/PICF_AQR_OWM_REMOTE_CALVIN_AUDIT_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_REMOTE_CALVIN_AUDIT_TEMP.md)
  Remote audit of the older `8fdb16f` CALVIN run. It records why that checkpoint
  remains a failing anchor-quality baseline and why it cannot be used as proof
  that the current checkout is empirically fixed.
- [`src/openpi/picf/README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
  Historical pre-v2.2 record.
- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
  Compact executable contract for the live code.
- [`docs/CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
  Training, serving, and CALVIN validation workflow.
  The current canonical full PICF long-run training command is recorded in
  [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch).
  The frozen-perception profile and experimental burn-in / suffix-gradient
  `state_only` speed path are recorded in
  [`Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile).
  The current cloud-tested 20-sequence video evaluation recipes, including the
  full PICF `step=7500` run and the maintained PI0.5-only ablation run, are
  recorded in
  [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation).
  Anchor / task-readout attention diagnostics and visual predictive-cache
  comparison output are recorded in
  [`Section 6.1.3`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#613-anchor-task-readout-and-predictive-cache-diagnostics).

## 0. Current Audit Snapshot

This file should currently be read as a **local correctness / architecture
record first**, and only secondarily as a historical record of earlier cloud
bring-up attempts.

As of the latest local audit pass:

- the v2.2 physical / task-readout / conditioned-control / PI0 action contract
  is internally consistent
- the current final graph path is AQR-MAPG, not MAPG-v0:
  - `aqr_mapg_enabled=True` is the default and builds graph anchors from
    learned physical/task queries over typed support memory
  - `mapg_enabled=False` and `vl_anchor_router_enabled=False` are the default
    legacy-router state
  - PaliGemma semantic tokens still condition task queries
  - PaliGemma image tokens help task queries localize by cross-attention and
    resize-with-pad projection onto the V-JEPA visual grid
  - PaliGemma heatmaps/grounding are production-disabled by default and only
    exist under explicit `--aqr-pg-grounding-enabled`; they affect AQR only when
    `--aqr-pg-bias-weight > 0`
  - graph consumers remain observation anchors, task readout, posterior binding,
    and conditioned control, so the PI0.5 action path stays unchanged
- the runtime surface now has two explicit modes:
  - `picf_mode=enabled`: full v2.2 PICF path
  - `picf_mode=ablated`: PI0.5-only ablation path with PICF branches disabled
- future-target supervision is now explicit stop-gradient teacher supervision
- shared middle frames inside a training window are now reused as detached
  future targets instead of being redundantly rebuilt
- the live training policy remains:
  - one canonical recurrent physical carry
  - one canonical conditioned control state `C_t`
  - one final PI0.5 action path
- no current local regression evidence shows semantic leakage into posterior or
  innovation

2026-05-09 AQR deployment audit:

- the direct-final training mode is now the default PICF profile:
  - `aqr_mapg_enabled=True`
  - `mapg_enabled=False`
  - `vl_anchor_router_enabled=False`
  - `aqr_pg_grounding_enabled=False`
  - `aqr_pg_image_support_enabled=True`
  - `aqr_pg_image_support_weight=0.35`
  - `aqr_pg_bias_weight=0.0`
  - `semantic_mode=paligemma`
  - `aqr_vjepa_temporal_mode=last_two_tokens`
  - `evidence_cache_read_weight=0.05`
    - this is a true cache residual scale:
      `q <- q + 0.05 * (ReadCache(q) - q)`, not a softmax-bias-only
      switch
    - cache read skips the newest posterior cache row because
      `previous.posterior.tokens` already has a dedicated AQR posterior reader;
      cache contributes older role-compatible episodic context rather than
      duplicating t-1 posterior evidence
  - guarded OWM losses remain `lambda_slot_jepa=0`,
    `lambda_support_pred=0`, `lambda_binding_consistency=0`, and
    `lambda_aqr_denoising=0`
- equivalent CLI flags still exist for explicit overrides and ablations, but
  they are no longer required for the latest OWM default profile
- recommended large-run perception profile remains explicit:
  - formal V-JEPA temporal OWM training requires `--use-foundation-backbones`
    or an equivalent `visual_mode=encoder` launch; stub visual mode is only a
    light regression/ablation path and will not produce temporal V-JEPA priors
  - `--perception-finetune-mode frozen`
  - current cloud run cadence: `--num-train-steps 30000`,
    `--save-interval 2500`, `--unroll-steps 2`
- frozen perception means V-JEPA, Sonata, and AnyTouch backbone/pretrain
  parameters are frozen; PICF adapters, AQR query/router modules, graph
  consumers, PaliGemma semantic path, posterior/readout/control layers, and the
  PI0.5 action-side trainable path remain trainable unless separately frozen by
  the normal trainer.
- AQR now participates in the same point-optional runtime contract as the graph
  path itself: first-step point-cloud hard failures and hold reasons check
  `mapg_enabled or aqr_mapg_enabled`, so AQR does not silently inherit the old
  point-mandatory PICF guard.
- the training startup log prints an explicit AQR-MAPG contract line, so the
  cloud run can be audited from logs without inferring AQR state from the raw
  command line alone.
- the legacy MAPG-v0 builder remains available only as an explicit comparison
  path. It is not enabled in the final AQR training profile.
- the current local verification is syntax and graph-path verification, not a
  claim that CALVIN behavior has already passed. The required behavioral
  acceptance evidence starts at the first `5000`-step checkpoint: 20-sequence
  CALVIN evaluation, raw heatmap exports, heatmap-over-RGB overlays, anchor
  health videos, and JSON statistics.
- no current local regression evidence shows dual control semantics reappearing

Latest fully local verification evidence:

- local syntax / diff hygiene:
  - `python -m py_compile scripts/picf_core_train.py scripts/picf_core_train_test.py src/openpi/picf/policy.py src/openpi/picf/core/pipeline.py`
  - `git diff --check`
- remote targeted burn-in tests:
  - `pytest -q scripts/picf_core_train_test.py -k "burnin or state_only or postclip"`
  - `5 passed`
- `python scripts/verify_picf_contract.py` -> static checks, documentation
  checks, targeted invariance regressions, core regression suite, and smoke
  training check all pass
  - latest remote verifier evidence: `231 passed` in the core regression suite
- `python scripts/verify_picf_owm_contract.py` -> strict final OWM
  README-to-code contract verifier for temporal V-JEPA, PG priors, posterior
  address/content, evidence cache causality, no-leakage teacher targets, OWM
  debug metrics, and evidence-bundle coverage.

Latest 2x40GB frozen-perception burn-in smoke evidence:

- run name:
  `picf_v22_stateonly_burnin8_fixed_smoke_20260430_r3`
- command shape:
  `--unroll-steps 1 --burnin-steps 8 --burnin-mode state_only`
- result:
  - ordinary `step=10` completed
  - final `step=11` completed
  - full checkpoint saved at
    `/mnt/checkpoints/picf_core/picf_core/picf_v22_stateonly_burnin8_fixed_smoke_20260430_r3/11`
- observed speed:
  - about `0.055-0.061 steps/sec`
  - about `16.4-18.3 s/step`

Current selected 2x40GB long-run bring-up evidence:

- run name:
  `picf_v22_frozen2x40_photometric_burnin4_unroll1_30000_ckpt5000_20260430_r1`
- command shape:
  `--unroll-steps 1 --burnin-steps 4 --burnin-mode state_only`
- result:
  - reached early ordinary training steps
  - launch log printed `effective_window_steps=5`
  - launch log printed frozen Sonata / V-JEPA / AnyTouch and trainable
    PaliGemma
- observed early speed:
  - roughly `11.7-14.3 s/step`

Performance note:

- the current main open issue is still throughput, not mathematical contract
  correctness
- do not read this file as claiming that a full local or cloud `30000`-step run
  has already completed on the current code unless that evidence is recorded
  explicitly

### 0.1 Current Training / Model Summary

Use this subsection as the fast operator summary. The detailed command blocks
remain in
[`docs/CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md).

Current full PICF model contract:

- physical core:
  - language-free observation anchors
  - language-free physical posterior `W_t`
  - world-only innovation from previous `K_{t-1}^{phys}`
- semantic/control layer:
  - PaliGemma / PI0.5 semantic tokens remain native-width and trainable unless
    explicitly frozen by a future experiment
  - semantic enters PICF through current-step task readout, not by raw semantic
    prefix injection into the physical posterior or innovation
  - one canonical conditioned control state `C_t`
  - `C_t^{pi}` is a PI0.5 prefix view of `C_t`, not a second control state
- final action:
  - only PI0.5 flow matching / sampler is the final action path
  - the old PICF-local direct 7D action head is not the live action generator
- prediction:
  - `K_t^{phys}` is computed after the teacher/executed action is known
  - `K_t^{cond}` is token-level conditioned future cache built from physical
    prediction tokens plus future-condition tokens

Current training profiles:

| Profile | Use Case | Main Settings | Status |
| --- | --- | --- | --- |
| 6x40GB full PICF | Full cotrain when enough GPUs are available | `picf_mode=enabled`, `nproc_per_node=6`, `accum_steps=1`, `unroll_steps=2`, `action_horizon=16`, `save_interval=2500` | Same v2.2 objective, larger global batch than 4x40GB |
| 4x40GB full PICF | Standard all-backbone full-train reference | `picf_mode=enabled`, `fsdp_full_shard`, all foundation backbones trainable, `save_interval=2500` | Valid full-train profile, memory tight |
| 2x40GB anchor-only large-batch probe | Isolate whether AQR/MVTrack anchors can converge under frozen evidence | `use_foundation_backbones`, `perception_finetune_mode=frozen`, `picf_trainable_scope=anchor_only`, `unroll_steps=1`, `burnin_steps=0`, high-risk OWM losses `0`, `accum_steps` swept upward | Diagnostic only; acceptance is anchor health (`same_role_overlap`, effective anchor count, support entropy, identity switch/recycle), not final action quality |
| 2x40GB frozen-perception PICF | Cost-controlled full PICF without full perception cotrain | `perception_finetune_mode=frozen`, `unroll_steps=3`, `action_horizon=16`, `semantic_max_length=256` primary / `200` fallback, conservative photometric augmentation, `persistent_anchors=8`, `observation_anchors=16`, `visual_real_grid=64`, `save_interval=5000` | Full-BPTT quality reference; slower, about `24-27 s/step` in early 2x40GB probes before the 8/16 anchor + 64x64 visual-real update |
| 2x40GB frozen-perception fast PICF | Throughput-first full-PICF probe | `perception_finetune_mode=frozen`, `unroll_steps=1`, `semantic_gradient_checkpointing=False`, `action_horizon=16`, `semantic_max_length=256` primary / `200` fallback, `persistent_anchors=8`, `observation_anchors=16`, `visual_real_grid=64` | Reached about `8.8-9.3 s/step` before the 64x64 visual-real update; weakest recurrent credit-assignment profile |
| 2x40GB selected state-only burn-in | Current sub-15s 30000-step full-PICF run | `unroll_steps=1`, `burnin_steps=4`, `burnin_mode=state_only`, `semantic_gradient_checkpointing=False`, `persistent_anchors=8`, `observation_anchors=16`, `visual_real_grid=64`, `save_interval=5000` | Four no-grad recurrent updates plus one trainable suffix transition; preserves PICF architecture while reducing runtime |
| PI0.5-only ablation | Test the PI0.5 action path without PICF branches | `picf_mode=ablated`, `extra_prefix_tokens=None`, PICF core frozen | Maintained ablation profile, not full PICF |

2026-05-10 remote anchor-only runtime audit:

- Environment: 2x A100-PCIE-40GB, clean `Posterior_VLA` checkout synchronized
  through the Mainland GitHub mirror and launched exclusively with `uv run
  --no-sync`.
- Code revision audited: `c7dc6c2`.
- Smoke sweep: `accum_steps=1`, `4`, and `8` all reached checkpoint save under
  `--picf-trainable-scope anchor_only`, `--perception-finetune-mode frozen`,
  `--use-foundation-backbones`, FSDP full-shard, high-risk OWM losses at `0`.
- Selected probe: `accum_steps=8`, effective global batch `16`, 500 optimizer
  steps, `save_interval=100`, `log_interval=10`.
- First logged optimizer step (`step=10`) confirms the real foundation path and
  anchor-only contract are active: `visual=encoder(finetune_mode=frozen)`,
  `semantic=paligemma(trainable=False)`, `trainable_numel=82410222`,
  `total_numel=4088451325`, `windows_per_sec=0.1877`,
  `aqr_effective_anchor_count=23.24`, `aqr_same_role_support_overlap_max=0.865`,
  `posterior_identity_switch_rate=0.0`, `posterior_recycle_rate=0.995`.
- Interpretation: this proves the diagnostic path runs through real
  foundation-backbone AQR/MVTrack wiring with the intended frozen/trainable
  partition. It is not a convergence claim. The high early recycle rate is
  expected to be judged across the 100/500-step checkpoints together with
  support overlap, support entropy, and anchor videos.

2026-05-11 staged anchor/action recycle diagnosis:

- Source: [`docs/PICF_AQR_OWM_RECYCLE_DIAGNOSIS_PLAN_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_RECYCLE_DIAGNOSIS_PLAN_TEMP.md).
- R0 anchor-only warmup
  `picf_mvtrack_anchor_noaction_supportdiv_acc4_lr3e4_probe300_20260510_225402`
  reached step `300` with `loss_total=0.7145`, `loss_action=0`,
  `aqr_same_role_support_overlap_max=0.3756`, last20 overlap mean `0.4815`,
  `aqr_effective_anchor_count~=23.6`, and `posterior_recycle_rate=0.3837`.
  This proves the anchor-only objective can form non-collapsed supports.
- Direct action cotrain controls were structurally unhealthy even while scalar
  losses fell: R1 last20 overlap mean `0.9579`, R2 last20 overlap mean
  `0.9964`, both with `posterior_recycle_rate=1.0`. Do not use direct
  cotrain scalar loss descent as evidence of healthy anchor identity.
- The best staged candidate so far is S2
  `picf_s2_stage300_a05_cache005_sd005_den0_900_20260511_0155`: step `900`
  had `loss_total=0.7239`, `loss_action=0.0174` under action lambdas `0.5`,
  `loss_action_active7=0.3147`, last20 overlap mean `0.4551`, and
  `aqr_effective_anchor_count~=23.4`. Its default-weight-equivalent action loss
  is about `0.0696`, so it is comparable to old v22 action runs around the
  early `300-1000` step regime, not to old `20k+` step convergence.
- S2 is structurally much better than direct cotrain, but it still has
  `posterior_recycle_rate=1.0`. This is unresolved because
  `posterior_recycle_rate` is `recycle_gate.mean()` in `pipeline.py`, and high
  recycle means the posterior content path is mostly reset from residual
  evidence rather than temporally carried.
- Until recycle is understood, keep `lambda_slot_jepa=0`,
  `lambda_support_pred=0`, and `lambda_binding_consistency=0`. The staged
  action runs log `loss_slot_jepa` in the thousands; even a small nonzero
  weight could dominate the objective before identity continuity is healthy.
- Next required diagnostics are debug-only recycle instrumentation
  (`posterior_recycle_logit_*`, dustbin mass, raw/final support mass,
  residual-summary norm, role-wise recycle, address update rate), a staged
  action `0.25` run from R0, and an anchor-only continuation from R0. Do not add
  a recycle penalty or enable predictive auxiliaries before those diagnostics.
- 2026-05-11 diagnostic instrumentation is now code-level deployed and verified:
  `posterior_recycle_logit_*`, `posterior_recycle_gate_*`,
  `posterior_recycle_rate_effector/scene`, raw/final support mass, raw/final
  dustbin mass, prior variance/alpha, residual-summary norm, identity
  innovation risk, address-update rate, and local same-role overlap are carried
  through `PicfPosteriorAnchorState`, pipeline debug, trainer metrics, evidence
  bundle, strict diagnose, and MVTrack deep audit. These are observability-only
  fields; they do not change posterior update math or training losses.
- Action-loss logging now carries both the optimized weighted term and the
  4-22-comparable default-weight term: `loss_action` is still the actual
  objective contribution, while `loss_action_default_equiv` maps the action
  loss back to default `lambda_action_pos/rot/gripper=2.0` scale. Use
  `loss_action_default_equiv` and `loss_action_active7` when comparing staged
  low-action probes against the 2026-04-22 ablation baseline.
- 2026-05-11 follow-up instrumentation adds diagnostic-only controls
  `--picf-action-detach-from-anchor`, `--freeze-recycle-path`, and
  `--recycle-logit-clamp`, plus token-aware local refinement diagnostics
  `aqr_same_role_local_true_overlap_*`, `aqr_same_role_local_jaccard_*`, and
  `aqr_local_source_mass_*`. These are for causal attribution of action-gradient
  recycle saturation and local-overlap false positives; they are not new default
  training objectives.
- 2026-05-11 recycle attribution result: direct action gradients into PICF
  posterior/control prefix drive recycle saturation, while full action-loss
  detachment keeps recycle/address healthy. `--freeze-recycle-path` and
  `--recycle-logit-clamp=6` were insufficient, and both position-only and
  rotation-only action losses triggered recycle saturation. The correct cotrain
  repair is therefore a bridge-level stop-gradient:
  `--picf-action-prefix-stopgrad`. This lets PI0.5 action-flow loss train the
  action side normally while stopping the gradient at
  `conditioned_control.pi_prefix_tokens`, so action cannot use posterior recycle
  as a shortcut. This is not a permanent no-action regime; it is the safe
  staged-cotrain bridge before selectively re-opening PICF action gradients.
- The object-binding extraction path now follows the same pairwise/quadratic
  principle used by recent ViT object-binding probes: support distributions are
  converted into a projected `binding_signature` subspace, and posterior binding
  uses `hidden + geometry + support-overlap + binding-subspace + gated-address`.
  This is a structural binding term, not a new high-risk loss. It is intended to
  protect same-object slot identity without requiring dataset relabeling.
- 2026-05-11 cloud report update, historical snapshot:
  [`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md)
  is the current live experiment ledger. This paragraph records the now-closed
  2026-05-11 diagnostic state before the 2026-05-12 burn-in/support-collapse
  matrix replaced it. A5 had no active tmux and its completed
  `picf_a5_prefixstopgrad_a025_bindsub_diag300_20260511_95ea69b` diagnosis
  ended at step `600` with `posterior_recycle_rate=0.000469`,
  `posterior_recycle_logit_mean=-11.6767`, and
  `aqr_same_role_support_overlap_max=0.3027`. A7 is running
  `picf_a7_30k_prefixstopgrad_all_a1_semtrain_fast_from450_20260511_95ea69b`
  on runtime `95ea69b` with PaliGemma trainable, Sonata/V-JEPA/AnyTouch frozen,
  `accum_steps=1`, `unroll_steps=2`, and progress enabled. At early step `460`,
  A7 shows `posterior_recycle_rate=0.00838`,
  `posterior_recycle_logit_mean=-47.61`, `loss_action_default_equiv=0.0772`,
  and `aqr_temporal_view_mass_1=0.556`, while `owm_tracklet_tokens=0` and
  `owm_proposal_tokens=0` confirm tracklet/proposal evidence is still inactive
  in the current CALVIN training dataflow. A later live check reached step
  `520` with `loss_action_default_equiv=0.0682` but a recycle spike
  `posterior_recycle_rate=0.4627`; this keeps A7 in pending acceptance. Do not
  claim long-run stability from the earlier healthy samples alone.
- 2026-05-11 validation addendum: the experiment report also records the latest
  local and idle-card checks. Local checks passed `py_compile`, OWM verifier
  `31/31`, strict diagnose, dataflow trace, MVTrack deep audit, scripts pytest
  `4 passed`, targeted pipeline pytest `10 passed`, and targeted training pytest
  `24 passed`. A5 also completed a two-step full trainer runtime smoke with
  FSDP, PaliGemma trainable, Sonata/V-JEPA/AnyTouch frozen, and
  `--picf-action-prefix-stopgrad`. That smoke proves the runtime entry and
  metrics execute on GPU; it is not convergence or behavior acceptance.
- 2026-05-12 A5 follow-up diagnosis revised:
  [`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md)
  now records that A5 E1 already failed decisively by the post-500 samples:
  `posterior_recycle_rate=1.0`, `posterior_address_update_rate_mean=0.0`, and
  `aqr_same_role_support_overlap_max≈0.99995` under all-scope `unroll_steps=2`
  with PaliGemma cotrain. This is enough evidence to stop the 900-step E1 and
  replace it with a `500new` matrix: because the resume starts near trainer step
  450, the revised A5 runs use `num_train_steps=950` to obtain about 500 new
  optimizer steps. M1 sets `semantic_lr_scale=1e-6` to isolate semantic cotrain update pressure,
  M2 keeps PaliGemma cotrain but uses `burnin_steps=4`
  `state_only` identity inertia, M3 adds tiny `1e-4` guarded predictive hooks
  on top of M2, and M4 tests reduced `semantic_lr_scale=0.1` under direct
  `unroll_steps=2`. The aborted non-`500new` launch with `num_train_steps=500`, the aborted
  `m1b` launch without a semantic LR override, and the invalid `m1c` launch with
  `semantic_lr_scale=0.0` are not used for conclusions. The first would only add
  about 50 steps after resume; the second did not isolate PaliGemma update
  pressure because the foundation profile defaults semantic parameters to
  trainable; the third failed argument validation because LR scales must be > 0.
  PaliGemma cotrain remains a likely requirement for final action adaptation;
  freezing or near-freezing it is a diagnostic isolation, not the preferred
  production recipe unless the cotrain paths fail. These runs are diagnostic;
  they do not resolve tracklet/proposal, ordinal, or the offline IsSameObject
  audit by themselves.
- 2026-05-12 A7 reallocation:
  the old A7 direct `unroll_steps=2` long run is no longer treated as an
  acceptance run because the latest inspected tail still had high global
  same-role support overlap (`aqr_same_role_support_overlap_max` tail mean
  about `0.957`) despite action loss decreasing and recycle mostly suppressed.
  A7 now runs
  `picf_a7_burnin4_semtrain_aux0_1000new_20260512_95ea69b`, a medium-horizon
  stress test of the strongest current candidate: PaliGemma cotrain remains
  enabled, OWM predictive aux losses remain `0`, and identity inertia is
  provided by `burnin_steps=4`, `burnin_mode=state_only`, `unroll_steps=1`.
  The purpose is to decide whether the production long run should move from
  direct `unroll_steps=2` to burnin4/state-only before any predictive aux
  warmup.
- 2026-05-12 burn-in/support-collapse matrix supersedes the earlier A5/A7
  active-run notes. The live matrix currently runs four bounded diagnostics from
  the shared `model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b` resume:
  A5 compares `burnin_steps=1` vs `burnin_steps=2` under all-scope PaliGemma
  cotrain, and A7 compares the same `burnin_steps=2`/strong-diversity setting
  under all-scope vs `anchor_only`. The matrix is explicitly diagnostic: it
  tests whether short state-only burn-in prevents recycle saturation and whether
  stronger diversity prevents same-role support collapse. Do not infer
  production readiness from `loss_action_default_equiv` alone. The acceptance
  metrics are `posterior_recycle_rate`, `posterior_address_update_rate_mean`,
  `aqr_same_role_support_overlap_max`,
  `aqr_same_role_local_true_overlap_max`, `posterior_identity_switch_rate`,
  `loss_anchor_pv`, and `loss_mapg_routing`.
- 2026-05-12 mid-run matrix update: the first two all-scope diagnostics already
  exposed their failure modes before step 950. A5 `burnin_steps=1` kept
  `posterior_recycle_rate≈0` but still reached
  `aqr_same_role_support_overlap_max≈0.999`; A7 `burnin_steps=2` with strong
  support/geometry diversity reached `posterior_recycle_rate=1.0` and
  `aqr_same_role_support_overlap_max≈0.995`. These runs should be archived and
  replaced by their queued counterfactuals: A5 `burnin_steps=2` normal-diversity
  and A7 `anchor_only` strong-diversity. If A7 anchor-only is healthy while
  all-scope failed, action/all-scope cotrain pressure is the cause. If
  anchor-only also collapses, the current support-diversity loss form is
  insufficient and the next code change should be a direct same-role overlap-max
  or role-wise assignment-competition objective.
- 2026-05-12 second-stage update: A5 `burnin_steps=2` also kept recycle off but
  still showed `aqr_same_role_support_overlap_max≈0.982`, while A7
  `anchor_only` with strong diversity still showed
  `aqr_same_role_support_overlap_max≈0.999`. This means the issue is not just
  insufficient burn-in and not just all-scope/PaliGemma cotrain. The next
  counterfactuals isolate the anti-collapse objective itself: one
  `anchor_only/no-action/support-only` run and one `anchor_only/no-action` run
  with anchor/PV retained. If both still collapse, ordinary support-diversity
  is not enough and the next implementation should directly optimize the
  same-role overlap health metric or add role-wise assignment competition.
- 2026-05-12 third-stage launch: the next two diagnostic runs are now the
  canonical branch point. A5 is `anchor_only/no-action/support-only` with
  `lambda_mapg_support_diversity=1.0`, `lambda_mapg_geometry_diversity=0.1`, and
  all action/PV/cycle/predictive losses disabled. A7 is `anchor_only/no-action`
  with anchor/PV retained (`lambda_anchor_pv=0.25`, `lambda_pv_weak=0.05`,
  `lambda_mapg_cycle=0.05`) plus the same strong diversity weights. The target
  is not policy quality; it is isolating whether same-role support collapse is a
  loss-formulation issue, PV/alignment conflict, or action-gradient conflict.
  See `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md` for the detailed
  decision table.
- 2026-05-12 third-stage audit: both no-action runs completed. A5 briefly
  reached `aqr_same_role_support_overlap_max≈0.525` at step 460 but rebounded to
  `≈0.991` by step 750; A7 with anchor/PV retained ended at `≈0.9998`. Recycle
  stayed near zero and effective anchors stayed around 23, so the failure is not
  dead anchors or recycle saturation. The current conclusion is that ordinary
  support-diversity is a useful but insufficient proxy; the next implementation
  should directly optimize same-role visual overlap tails and soft local
  candidate-set competition before action/PV cotrain is reintroduced.
- 2026-05-12 health-aligned support patch: `loss_mapg_support_diversity` now
  includes the runtime health metric's normalized same-role visual-overlap tail
  and a differentiable local-candidate reuse penalty using
  `local_token_indices/local_priors`. This is deliberately not a new module or
  new loss family; it aligns the existing anti-collapse objective with the
  observed failure. The first validation remains no-action isolation before
  PV/action are reintroduced.
- 2026-05-12 finite-gradient hardening: the local-candidate reuse penalty now
  uses `sqrt(clamp(mass_product, min=eps))` rather than `sqrt(clamp(..., min=0))`.
  This fixes a real boundary-gradient issue exposed by the first supportfix
  remote run, where exact-zero sparse local priors produced non-finite gradients
  despite finite forward losses. The corresponding test backpropagates through
  exact-zero local priors and asserts finite gradients. The restarted A5/A7
  supportfix runs are the active acceptance tests. Step 480 already shows the
  intended causal split: A5 support-only remains below the old 0.99 collapse
  zone, while A7 with anchor/PV pressure drifts back toward high local-candidate
  reuse. Step 500 is stricter: A5 support-only rebounds to
  `aqr_same_role_support_overlap_max≈0.982` with
  `aqr_same_role_local_jaccard_max≈0.988`, proving the remaining issue is local
  candidate-set reuse inside AQR/local refinement, not action or PV alone.
  Step 520 is decisive: A5 reaches `overlap≈0.999` and `local_jaccard=1.0`,
  and A7 also reaches `overlap≈0.954` with `local_jaccard=1.0`,
  so the next repair must move anti-collapse pressure into role-wise local
  candidate selection/ownership before support aggregation. See
  `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`.
- 2026-05-12 rejected local-candidate heuristics cleanup: role-wise soft local
  candidate competition and deterministic coverage-seeded local proposal were
  both rejected by A5/A7 evidence. They reduced neither the high local Jaccard
  tail nor the same-role support collapse once AQR rows had already become
  nearly identical. These paths have been removed from production config, CLI,
  debug logging, verifier contracts, and runtime tests. They remain documented
  only as historical experiments in
  `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`.
- 2026-05-12 coverage-seed result: both A5 and A7 completed to step 750 and
  rejected the coverage-seeded proposal. A5 ended with
  `aqr_same_role_support_overlap_max≈0.99936`,
  `aqr_same_role_local_jaccard_max=1.0`, and `posterior_recycle_rate=1.0`;
  A7 ended with `aqr_same_role_support_overlap_max≈0.99933`,
  `aqr_same_role_local_jaccard_max=1.0`, but `posterior_recycle_rate=0.0`.
  This proves the remaining collapse is not just action/PV pressure, role
  competition, deterministic coverage priors, or recycle saturation. The next
  one-hour diagnostic is a pairwise binding-subspace ablation inspired by
  object-binding probes: A5 disables `bind_*signature/address` terms, while A7
  emphasizes support/binding signatures with weak address inertia. See the
  experiment report for exact overrides and acceptance gates.
- 2026-05-12 pairwise binding-subspace result: the diagnostic was stopped at
  step 500 because both branches failed the local-candidate reuse gate. A5
  binding-off reached `support_overlap≈0.994` and `local_jaccard≈0.9997`; A7
  binding-strong reached `support_overlap≈0.999` and `local_jaccard=1.0`.
  This rejects further weight sweeps of the current binding-signature path.
  Keep the moderate binding-signature prior as an architecture-aligned low-cost
  term, but do not reintroduce strong binding weights or removed local-candidate
  heuristics. The next clean test is an offline IsSameObject token probe or real
  tracklet/proposal dataflow.
- 2026-05-12 cleaned two-branch diagnostic deployment: after the cleanup at
  `145654a` and the experiment plan at `7bff430`, the next live test is not a
  new mechanism. It is a causal training-pressure split:
  `clean staged cotrain` keeps `picf_action_prefix_stopgrad` during warmup,
  while `direct cotrain control` removes that stop-gradient from the beginning.
  Both use the same 450-step model-only resume, foundation backbones,
  trainable PaliGemma when memory allows, moderate binding signatures,
  conservative support/geometry diversity, cache read weight `0.05`, and
  zero predictive auxiliary lambdas. The acceptance gates are
  `aqr_same_role_local_jaccard_max`, `aqr_same_role_support_overlap_max`,
  `posterior_recycle_rate`, finite gradients, and only secondarily
  `loss_action_default_equiv`. See
  `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`.
- 2026-05-12 local audit checkpoint: commit `146bb86` is the audited
  README/experiment-record deployment base. Local verification
  passed the OWM verifier, strict diagnosis, recursive dataflow trace,
  MVTrack deep audit, key-file `py_compile`, targeted PICF
  `pipeline`/`training`/script regressions, broader script regression
  (`162 passed` after updating stale test loss stubs), broader non-core PICF
  regression (`31 passed`), and full core `pipeline`/`training` regression
  (`101 passed`). The final combined local audit regression passed
  `232 passed`. This is a code/dataflow audit, not a replacement for live A5/A7
  metrics or CALVIN/video behavior evidence. The detailed command ledger and
  boundary conditions are recorded in
  `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`.
- 2026-05-12 stricter multi-layer local audit: after the later
  `98fe770` audit note, the local audit was repeated over the full PICF/script
  test inventory instead of only the targeted subsets. The stricter pass
  verified README routing, stale/removed knobs, explicit no-op guards,
  MVTrack invariants, key-file and full PICF/script `py_compile`, verifier,
  strict diagnose, recursive dataflow trace, and MVTrack deep audit. The full
  PICF/script test set contains 45 pytest files. `scripts/train_test.py` is
  currently blocked at collection by the local `wandb`/`wandb_watchdog`
  installation (`No module named wandb_watchdog.observers.polling`), which is
  a base training-entry environment issue rather than a PICF/MVTrack failure.
  The remaining 44 PICF/script test files pass after updating the CALVIN eval
  dummy policy test to match the current debug-argument and `close()` lifecycle:
  `370 passed, 3 skipped`. This is the current strongest local code/dataflow
  audit result, but it is still not behavior acceptance; A5/A7 metrics,
  CALVIN evals, videos, and anchor-health overlays remain the behavior gate.
  The same experiment report records the post-audit live A5/A7 status:
  both runs are alive and action/alignment losses are decreasing, but neither
  is accepted yet because local candidate reuse and posterior identity-switch
  diagnostics remain too high.
- 2026-05-12 move-on gate: the A5/A7 clean/direct cotrain pair is stopped at
  the step-600 diagnostic point and superseded by the local-refinement
  isolation matrix recorded in
  `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`: A5 disables local
  refinement; A7 keeps it but reduces top-k and residual weight. This is a
  causal diagnostic for high same-role local-candidate reuse, not a new
  mechanism or a production long-run.
- 2026-05-12 local-refinement isolation checkpoint: the step-600+ evidence
  is sufficient to move to the next diagnostic. A5 local-off reaches
  `aqr_same_role_support_overlap_max=0.7294` at step 750 with local priors
  disabled by construction, while A7 local top-k 8 / residual 0.10 reaches
  `aqr_same_role_support_overlap_max=0.4103` and
  `aqr_same_role_local_jaccard_max=0.4853` by step 750. This confirms that
  local refinement was an overlap amplifier. It does not yet prove identity
  health, because both branches still report raw
  `posterior_identity_switch_rate≈0.83..0.88`. The next step is therefore a
  debug-only stable-slot identity audit, not another scalar identity loss:
  mask low-confidence/recycled/ambiguous slots and log binding margins before
  deciding whether assignment math or only the raw switch metric is failing.
  The local patch adds these metrics without changing training behavior:
  `posterior_identity_switch_rate_stable`,
  `posterior_identity_switch_rate_nonrecycled`,
  `posterior_identity_switch_rate_recycled`,
  `posterior_stable_slot_fraction`, and
  `posterior_binding_top1_margin_*`. Use the stable metric as the next
  acceptance discriminator; keep raw switch as an alarm. A5 has started
  `picf_a5_stableid_localk8w01_burnin4_650new_20260512_253c9be` on commit
  `253c9be` to collect these metrics under local top-k 8 / residual 0.10.
- 2026-05-12 stable-identity result: the A5 stable-id run completed to
  step 650. The final row has `posterior_identity_switch_rate=0.6722` but
  `posterior_identity_switch_rate_stable=0.0`; the last-10 mean is
  `raw_switch=0.7722`, `stable_switch=0.0025`. Same-role overlap remains
  controlled (`last=0.4313`, last-10 mean `0.4491`) and stable slots have
  high binding margin (`posterior_binding_top1_margin_stable_mean≈0.96`).
  The current bottleneck is therefore not proven identity collapse; it is low
  stable-slot coverage (`posterior_stable_slot_fraction≈0.12`). The next
  experiments should increase stable-slot coverage and then test controlled
  action pressure. Do not add a raw identity-switch loss or open predictive
  OWM losses yet.
- 2026-05-12 three-line follow-up deployment: A5 is now assigned to CALVIN
  anchor/prediction visualization from the stable-id checkpoint, while A7 runs
  the stable-coverage continuation from the local top-k 8 / residual 0.10
  branch. This is paired with a paper-grounded review of 2025+ object-binding
  and VLA tracking work. The immediate hypothesis is that stable slots are
  reliable when they exist, but coverage is too low; the correct next fix is
  support/signature same-object evidence and real tracklet/proposal dataflow,
  not a raw identity-switch loss or early predictive auxiliary pressure. See
  `docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md` section
  `2026-05-12 Three-Line Follow-Up`.
- 2026-05-12 preliminary anchor-overlay reading: A5 CALVIN rollout debug is now
  producing anchor overlays and JSON. The first `push blue block right` frames
  show slot 0 stable at one region while same-role slots 1..7 are nearly
  identical in pixel/xyz coordinates. This means stable-switch near zero is not
  sufficient as acceptance: same-role slots can be stable while sharing the same
  candidate. The next gate is stable coverage plus visual/3D differentiation,
  not raw identity-switch suppression. The immediate test order is A7
  stable-coverage continuation, A5 overlay reading, offline IsSameObject probe
  if coverage fails, then support-signature audit or real tracklet/proposal
  dataflow depending on whether same-object evidence is decodable.
- 2026-05-13 stable-coverage midpoint: A7 reaches step 800 with
  `stable_identity_switch=0.0` and controlled overlap
  (`same_role_support_overlap≈0.52`), but
  `posterior_stable_slot_fraction` remains fixed at `≈0.111`. This confirms the
  current bottleneck is coverage/differentiation, not stable-slot switching. If
  the 1050-step endpoint does not raise stable coverage, the next experiment is
  an offline IsSameObject probe, not another action/JEPA/raw-switch loss sweep.
- 2026-05-13 completed A5 overlay plus A7 step-880 update: A5 CALVIN
  visualization finished with behavior success `0/2`, but its anchor overlays
  are diagnostically useful. The effector-like role-0 slot tracks a distinct
  region, while scene slots 1..7 repeatedly share the same pixel/XYZ candidate
  across both `push_blue_block_right` and `open_drawer`. A7 at step 880 remains
  better than the 0.99 overlap-collapse runs
  (`same_role_support_overlap≈0.35`, `local_true_overlap≈0.035`) and stable
  slots have high binding margin (`≈0.96`), but stable coverage is still only
  `≈0.117`. The current gate is unchanged: finish A7 to 1050, then run the
  offline IsSameObject probe if stable coverage does not rise materially. Do
  not add action/JEPA/raw-switch penalties as a shortcut.
- 2026-05-12 storage cleanup policy: `/mnt` May-2026 numeric checkpoint
  subdirectories are disposable once their logs and JSON metrics are preserved.
  Keep the April 4-22 ablation baseline, the April full-PICF baseline, the
  current active A5/A7 run directories, and the shared
  `model_only_resume_a5_prefixstopgrad_450_for_all_95ea69b` resume checkpoint.
  The cleanup is intentionally storage hygiene only; it is not a change to the
  mathematical training contract.

Default recommendation:

- use full-BPTT `unroll_steps=3` when the priority is strongest recurrent
  credit assignment and runtime is acceptable
- use `unroll_steps=1`, `burnin_steps=4`, `burnin_mode=state_only` for the
  current sub-15s 2x40GB long run; this is the selected speed/quality
  compromise under the current rental budget
- use the fast `unroll_steps=1` profile only when the immediate target is
  `~10 s/step` throughput; it keeps the PICF architecture enabled but does not
  train multi-step recurrent BPTT like `unroll_steps=3`
- use `state_only` burn-in only when explicitly testing speed/context tradeoffs
- do not treat `state_only` as equivalent to full-BPTT over
  `burnin_steps + unroll_steps`; it trains suffix losses on a longer recurrent
  context but does not backpropagate credit through the burn-in transitions
- do not disable FSDP on the 2x40GB trainable-PaliGemma profile; direct DDP was
  tested and OOMed in the PaliGemma/Gemma forward MLP
- use `semantic_max_length=256` first on frozen-perception 2x40GB runs; use
  `200` only as a memory fallback or when intentionally matching a PI0.5
  prompt-length parity run
- keep `action_horizon=16` for CALVIN PI0.5/PICF action training unless a
  separate horizon sweep is intended
## 1. Document Map

This file is the **current local v2.2 architecture record and implementation
audit**. The one-shot action/control refactor described below has been deployed
locally; the detailed planning sections are retained because they explain the
implemented contract rewrite and the reasoning behind it.

Relevant documents:

1. `/home/siyuanyue/Documents/openpi/README.md`
   Repo-level entry point. Use this if you are opening the repository cold and
   want the broad project context before diving into PICF-specific docs.

2. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md`
   This file. Current v2.2 architecture record. It records:
   - what the current live code implements
   - what changed in the v2.2 contract rewrite
   - the canonical object/state decomposition
   - file-by-file implementation scope and rationale
   - migration, testing, and rollout gates used to validate the patch

3. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md`
   Historical v2.1 deployment record from before the v2.2 refactor landed.

4. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_PI05_PARITY_AUDIT.md`
   Detailed local audit of reference PI0.5 / PI0.5+Sonata dataflow versus the
   current PICF enabled and PI0.5-only ablated paths. Use it when interpreting
   ablation quality, CALVIN loss comparability, normalization, preprocessing,
   and parity claims.

5. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`
   Design record for the planned `picf_frozen_perception_2x40` profile. It
   documents which perception modules may be frozen without changing the PICF
   architecture, why full PI0.5 RGB geometry augmentation cannot be blindly
   copied into full PICF, and which augmentation modes are safe to make
   configurable.

6. `/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md`
   Current concise executable contract enforced by regression tests. It is the
   compact version of the live v2.2 semantics described in this file.

7. `/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md`
   Current training / validation / rollout / serving workflow document.

8. `/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md`
   Directory entry / pointer file. It points at both the current live
   record (`README_v2.2.md`) and the archived v2.1 record so neither is lost.

Maintained/current docs:

- `src/openpi/picf/README_v2.2.md`
- `src/openpi/picf/README_PI05_PARITY_AUDIT.md`
- `src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md`
- `PICF_FORMAL_CONTRACT.md`
- `docs/CALVIN_VALIDATION_README.md`
- `src/openpi/picf/README.md` as entry pointer

Historical/archive docs:

- `src/openpi/picf/README_v2.1.md`

### 1.1 Read Order

If you are opening the repo cold, use this order:

1. [`/home/siyuanyue/Documents/openpi/README.md`](/home/siyuanyue/Documents/openpi/README.md)
   Repo-level entry point.
2. [`src/openpi/picf/README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)
   PICF directory entry pointer.
3. [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
   Current live architecture record and implementation audit.
4. [`README_FROZEN_PERCEPTION_AUGMENTATION.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md)
   Read this before using frozen V-JEPA/Sonata/AnyTouch profiles or enabling
   train-time augmentation in full PICF.
5. [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
   Concise executable contract for the current code.
6. [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
   Runtime / training / rollout workflow.
   For the current canonical full PICF long-run launch, jump directly to
   [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch).
   For the 2x40GB frozen-perception profile and state-only burn-in speed path,
   jump directly to
   [`Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile).
   For the current validated cloud rollout/video test recipes, including
   full PICF `step=7500` serving/evaluation, single-rollout GPU usage, and
   `/tmp` to `/mnt` artifact mirroring, jump directly to
   [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation).
   For the current 2x40GB ablated training definition and the explicit
   `2500 current optimizer steps ~= 5000 historical no-Sonata PI0.5 steps`
   comparison rule, use the ablated long-run profile in
   [`Section 3`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#3-trainer-smoke-validation).
7. [`README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
   Historical pre-v2.2 record only.

### 1.2 Temporary Audit Companions

For this local v2.2 rollout, the primary temporary audit documents are:

1. [`/tmp/picf_v22_temp_current_dataflow.md`](/tmp/picf_v22_temp_current_dataflow.md)
   Recursive current-code dataflow audit for the live trainer/policy/core/loss path, including the one-step lookahead loss wiring.
2. [`/tmp/picf_v22_temp_theory_reconciliation.md`](/tmp/picf_v22_temp_theory_reconciliation.md)
   Theory-side reconciliation of the current implementation, including detached future-target reuse on internal window frames.
3. [`/tmp/picf_v22_bug_optimization_register_20260421.md`](/tmp/picf_v22_bug_optimization_register_20260421.md)
   Explicit bug / optimization register for the current audit pass.

Supporting temporary audit documents for the same rollout:

4. [`/tmp/picf_v22_current_code_dataflow_20260420.md`](/tmp/picf_v22_current_code_dataflow_20260420.md)
   Prior recursive current-code dataflow audit for the initial 2026-04-20 pass.
5. [`/tmp/picf_v22_audit_report_20260420.md`](/tmp/picf_v22_audit_report_20260420.md)
   Prior audit conclusion file: cloud/runtime evidence, real findings, non-findings, and next exact-optimization targets.
6. [`/tmp/picf_v22_mathematical_spec_20260420.md`](/tmp/picf_v22_mathematical_spec_20260420.md)
   Refreshed v2.2 mathematical specification with the canonical recurrent-carry contract.
7. [`/tmp/picf_v22_design_reconciliation_20260420.md`](/tmp/picf_v22_design_reconciliation_20260420.md)
   Explicit list of design mismatches / unnecessary glue, plus the fixes landed during the audit.
8. [`/tmp/picf_v22_memory_audit_20260420.md`](/tmp/picf_v22_memory_audit_20260420.md)
   Quantitative 4x40GB A100 memory audit and backbone-contribution ranking.
9. [`/tmp/picf_v22_speed_audit_20260420.md`](/tmp/picf_v22_speed_audit_20260420.md)
   Performance-specific historical speed audit for the earlier exact-memory bring-up passes.
10. [`/tmp/picf_v22_readme_sync_20260420.md`](/tmp/picf_v22_readme_sync_20260420.md)
   README synchronization audit for the current live deployment profile, observability modes, and GitHub handoff scope.

Historical temp audits from the earlier 2026-04-18 pass are retained only as
archive context:

11. [`/tmp/picf_v22_current_code_dataflow_20260418.md`](/tmp/picf_v22_current_code_dataflow_20260418.md)
12. [`/tmp/picf_v22_mathematical_spec_20260418.md`](/tmp/picf_v22_mathematical_spec_20260418.md)
13. [`/tmp/picf_v22_final_reconciliation_20260418.md`](/tmp/picf_v22_final_reconciliation_20260418.md)

These `/tmp` documents are audit artifacts, not persistent maintained docs.

### 1.3 Section Guide

Use the sections in this file as follows:

- Section 2: scope
- Section 3: recursive audit of the current live implementation
- Section 4: canonical v2.2 object/state shape
- Section 5: non-negotiable invariants
- Section 6: corrections to the earlier external proposal
- Section 7: detailed end-to-end v2.2 dataflow
- Section 8: file-by-file implementation map
- Section 9: checkpoint and compatibility migration
- Section 10: validation matrix
- Section 11: rollout gate record
- Section 12: forbidden regressions
- Section 13: definition of done
- Section 14: final recommendation

### 1.4 Navigation Summary

When verifying the local v2.2 codebase, use this navigation split:

- architecture and rationale:
  - [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)
- concise executable rules:
  - [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
- runtime / training / rollout workflow:
  - [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
  - current canonical full PICF long-run launch:
    [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch)
  - 2x40GB frozen-perception full PICF and state-only burn-in speed path:
    [`Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile)
  - current cloud 20-sequence CALVIN video evaluation:
    [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation)
    This section records that one CALVIN rollout uses one policy-server GPU plus
    one EGL/evaluator GPU; it does not automatically consume all GPUs the way
    FSDP training does.
- historical pre-v2.2 reference only:
  - [`README_v2.1.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.1.md)
- temporary deep audits for this local rollout:
  - [`docs/PICF_AQR_OWM_MVTRACK_DEEP_AUDIT_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_MVTRACK_DEEP_AUDIT_TEMP.md)
    Generated strict MVTrack runtime-c audit covering cache dataclass flow,
    residual cache math, typed local refinement, guarded predictive losses,
    weak ordinal no-posterior-rewrite, and documentation overclaim boundaries.
  - [`/tmp/picf_v22_temp_current_dataflow.md`](/tmp/picf_v22_temp_current_dataflow.md)
  - [`/tmp/picf_v22_temp_theory_reconciliation.md`](/tmp/picf_v22_temp_theory_reconciliation.md)
  - [`/tmp/picf_v22_bug_optimization_register_20260421.md`](/tmp/picf_v22_bug_optimization_register_20260421.md)
  - supporting:
    [`/tmp/picf_v22_current_code_dataflow_20260420.md`](/tmp/picf_v22_current_code_dataflow_20260420.md),
    [`/tmp/picf_v22_audit_report_20260420.md`](/tmp/picf_v22_audit_report_20260420.md),
    [`/tmp/picf_v22_mathematical_spec_20260420.md`](/tmp/picf_v22_mathematical_spec_20260420.md),
    [`/tmp/picf_v22_design_reconciliation_20260420.md`](/tmp/picf_v22_design_reconciliation_20260420.md),
    [`/tmp/picf_v22_memory_audit_20260420.md`](/tmp/picf_v22_memory_audit_20260420.md),
    [`/tmp/picf_v22_speed_audit_20260420.md`](/tmp/picf_v22_speed_audit_20260420.md),
    [`/tmp/picf_v22_readme_sync_20260420.md`](/tmp/picf_v22_readme_sync_20260420.md)
  - archived 2026-04-18 temp audits only when historical comparison matters

## 2. Scope

The goal of v2.2 is **not** to redesign the physical core. The goal is to
preserve the already-correct physical world-state machinery and perform a
single, coherent refactor of the action/control integration layer.

The intended one-shot result is:

- one exported policy object
- one canonical conditioned control state `C_t`
- one final action path, always through PI0.5 flow matching / sampling
- semantic used for task-relevant current-step readout, not for physical world
  state update
- world-only predictive basis kept clean for next-step innovation

### 2.1 Runtime Modes

The live trainer / serve stack now has two explicit runtime modes.

`picf_mode=enabled`

- canonical v2.2 path
- builds and trains the PICF recurrent/task-readout/conditioned-control/future
  branches
- PI0.5 action generation consumes `conditioned_control.pi_prefix_tokens`

`picf_mode=ablated`

- PI0.5-only ablation path for parity checks against the main-branch PI0.5
  baseline
- PICF recurrent/control/future losses are disabled
- trainer uses only PI0.5 flow loss and serve uses only the PI0.5 sampler
- PICF core parameters are frozen and excluded from the optimizer
- PICF-only point/visual/tactile backbone branches are normalized back to the
  stub/off path so the ablation does not pay for unused foundation modules
- PI0.5 is called with `extra_prefix_tokens=None`
- checkpoint save/load in this mode is semantic-only by construction:
  `model.pt` stores only the PI0.5 semantic subtree, while the frozen lazy
  PICF core is deliberately omitted instead of being force-materialized for
  serialization
- `optimizer_checkpoint_mode=auto` now resolves to model-only checkpoints in
  this mode; use `--optimizer-checkpoint-mode full` only when an ablated run
  truly needs optimizer-state resume
- if serving overrides an enabled checkpoint with `--picf-mode ablated`, the
  runtime args are re-normalized and re-validated before model/source
  construction so enabled-mode tactile/visual branch assumptions do not leak
  into the ablation path

The architectural contract described in the rest of this document remains the
canonical contract for `picf_mode=enabled`. The ablated mode is a deliberate
control experiment, not a second canonical PICF semantics.

### 2.2 Shared Trainer-Shell Contract

The runtime mode switch does **not** create a second trainer shell.

Inside `scripts/picf_core_train.py`, the following operational knobs are shared
by both `picf_mode=enabled` and `picf_mode=ablated`:

- `--save-interval`
- `--log-interval`
- `--accum-steps`
- `--unroll-steps`
- `--action-horizon`
- progress-bar cadence
- `metrics.jsonl` write cadence
- `_CalvinTransitionSource` window sampling contract

What changes with runtime mode is narrower:

- the per-window semantic/PICF execution path inside `PicfPi05Policy` and
  `_PicfWindowTrainer`
- the default checkpoint payload policy under
  `--optimizer-checkpoint-mode auto`

Operational rule:

- if you change `--save-interval 2500` or `--log-interval 100` for an ablation
  run, the same flags work the same way for `picf_mode=enabled`
- if you change `--action-horizon` or `--unroll-steps`, the same parser and
  window source own those flags for both modes
- do **not** infer from this that `picf_mode=ablated` is operationally
  identical to the preserved `pi0.5_sonata` trainer; it only means both PICF
  runtime modes share one maintained training shell

v2.2 is therefore treated as:

- contract rewrite
- interface unification
- task-readout refactor

and **not** as a cosmetic cleanup.

## 3. Current Live Code Audit

This section records what was already correct in current code and what the
v2.2 rewrite changed.

### 3.1 Physical Core: Keep

The following pieces were already the right mathematical object and were kept
structurally unchanged:

- `PicfFullCore._build_observation_anchors(...)`
- `PicfFullCore._posterior_update(...)`
- `PicfFullCore._innovation(...)`
- the split between `physical_prediction_cache` and `prediction_cache`

Observed current properties from code:

- observation-anchor construction is task-agnostic
- posterior update is language-free
- innovation reads only `previous.predictive.physical_prediction_cache`
- the only recurrent physical state family remains the posterior anchor state

Relevant code:

- `src/openpi/picf/core/pipeline.py`
- `src/openpi/picf/core/contracts.py`
- `PICF_FORMAL_CONTRACT.md`

### 3.2 Current-Step Private Dense Memory: Already Present

Current code already has:

```python
@dataclasses.dataclass(frozen=True)
class _StepDenseMemory:
    point_payload: torch.Tensor
    visual_payload: torch.Tensor
    tactile_group_tokens: tuple[torch.Tensor, ...]
```

2026-05-14 object-core ownership deployment: A5 latest reached step200 with
raw same-role support overlap `0.9064`, active overlap `0.6260`, effective
anchor count `17.15`, recycle `0.8201`, and address update `0.0070`. This
rejects the current anchor-only row as a final solution. The failure is now
localized more narrowly: posterior co-location is delayed, but same-role
object files can still reread the same broad support. A7 remains useful as an
older cotrain control, but it is not the maintained candidate.

The maintained repair is a full object-core measurement prior, not another
outer loss. The previous ownership prior acted on visual and temporal tokens;
the new default also adds a label-free point ownership prior before AQR point
attention:

```text
aqr_ownership_point_prior_weight = 0.35
aqr_ownership_point_prior_sigma_m = 0.04
```

For each role-local object-file row `j`, candidate point tokens are selected by
the existing role mask (`local` for effector, visible global scene points for
scene/object). Farthest-point centers are sampled from that candidate set and
converted into a soft ownership prior:

```math
O_{j,n}
=
Normalize_n\left(
  (1-\epsilon)
  \exp\left[-\frac{\|x_n-c_j\|^2}{2\sigma_p^2}\right]
  +
  \epsilon / N_r
\right)
```

AQR receives the centered log-prior:

```math
b^{point}_{j,n}
=
\lambda_p
\left(
  \log(O_{j,n})
  -
  \frac{1}{N_r}\sum_m \log(O_{j,m})
\right)
```

This is not a category-specific detector and not a hand-labeled object mask. It
is the point-cloud analogue of DINO/query initialization: seed distinct object
hypotheses from encoder evidence, then let attention, support signatures, and
posterior correction learn the object files. It also follows the 2025
IsSameObject object-binding result: binding should be represented as a
pairwise low-dimensional relation that guides attention, not as a late action
penalty (`https://arxiv.org/abs/2510.24709`). Spatial-trace VLA work also
supports using persistent spatial evidence as a routing scaffold rather than
hoping action gradients alone discover stable object ownership
(`https://arxiv.org/abs/2508.09032`).

The active-slot filter is now aligned with the same object-core semantics. A
slot is considered redundant only when it overlaps in every available
object-core support space. With available overlaps `R^v, R^p, R^t, R^pg`, the
filter uses:

```math
R^{core}_{i,j}
=
\exp\left(
  \frac{1}{|\mathcal M|}
  \sum_{m\in\mathcal M}
  \log\left(\max(R^m_{i,j}, \varepsilon)\right)
\right)
```

This avoids demoting two object files that share diffuse visual support but
have distinct point/temporal evidence. Raw visual overlap remains logged for
diagnosis, but the active-capacity decision now uses the multi-modal
object-core overlap. New debug keys:

```text
aqr_same_role_object_core_overlap_max
aqr_same_role_object_core_overlap_mean
aqr_active_same_role_object_core_overlap_max
aqr_active_same_role_object_core_overlap_mean
```

Acceptance for the next A5 deployment is stricter than “loss goes down”:

```text
1. active object-core overlap should stay below the visual-only collapse band.
2. effective anchor count should remain high past step150/200.
3. posterior recycle must not climb simply because object files share support.
4. anchor overlays must show role-1 posterior object files no longer collapse
   to one physical centroid.
```

This matters because v2.2 needs semantic-conditioned readout over:

- current public multimodal memory
- current private dense payloads

That substrate already exists. v2.2 reuses it rather than replacing it.

### 3.3 Attention Primitives: Already Sufficient

Current core already includes:

- `CrossAttentionRead`
- `GatedCrossAttentionRead`
- `LazyCrossAttentionRead`

This is enough to implement:

- semantic-conditioned task queries
- public token read
- private visual reread
- private tactile reread
- private point reread

without introducing a fourth attention subsystem.

### 3.4 Historical Seam 1: Action Path Used To Be Glue

The pre-v2.2 trainer/serve path used to be split across:

- `core.step(...)`
- PI0.5 wrapper calls in trainer / serve
- `refresh_predictive_state_for_action(...)`

That seam is now resolved. Current live action integration is:

- trainer:
  - `PicfPi05Policy.forward_train_transition(...)`
- serving:
  - `PicfPi05Policy.act(...)`

The unified policy now owns:

- semantic encoding
- `observe_step(...)`
- PI0.5 flow loss / sampler call
- executed-action finalization through `finalize_with_action(...)`

### 3.5 Historical Seam 2: Dual Control Semantics Used To Exist

Pre-v2.2 `_predictive_state(...)` used to construct two parallel control routes:

1. `action_condition_prefix -> action_condition_tokens`
2. `control_prefix -> control_tokens -> control_query_state -> pooled_state`

That seam is now resolved. Current live code builds one canonical conditioned
control state `C_t` and derives:

- `conditioned_control.tokens`
- `conditioned_control.query_state`
- `conditioned_control.pi_prefix_tokens`
- `conditioned_control.future_condition_tokens`

Compatibility fields such as `predictive.action_condition_tokens` and
`predictive.control_query_state` survive only as views/debug aliases of the
canonical conditioned-control outputs.

### 3.6 Historical Seam 3: Raw Semantic Prefix Used To Enter Core Control/Future

Pre-v2.2 `_predictive_state(...)` directly injected the raw semantic prefix
into:

- `control_world`
- `predictive_semantic_world`

That route has now been removed from the core contract. Current live behavior
is:

- raw semantic prefix stays native in PI0.5 semantic/action generation
- semantic enters the core only through current-step task readout
- task readout contributes to the unique conditioned state `C_t`
- conditioned future is built from token-level physical predictive tokens plus
  `C_t^{future}`, not from raw semantic-prefix injection

### 3.7 Important Current Code Fact: `fused_tokens` Is Not Full Multimodal Public Memory

Current `_build_token_field(...)` constructs:

```python
all_tokens = torch.cat([point_tokens, tactile_tokens_active, context_tokens], dim=0)
```

and then runs `token_fusion` over `all_tokens` to produce `fused_tokens`.

Current `visual_tokens` exist in `PicfTokenFieldState`, but they are **not**
part of `fused_tokens`.

Therefore:

- `token_field.fused_tokens` is not a full public multimodal memory
- it is a fused point / tactile-active / context memory
- visual public routing is currently handled separately by native-first visual
  reread before public fused read

This matters because the external proposal said "read full fused tokens". That
was corrected in v2.2: task-readout public memory is not defined as
`fused_tokens` alone.

### 3.8 Current State Dataclasses: Exact Audit

Current `src/openpi/picf/core/contracts.py` exposes these load-bearing state
containers.

`PicfTokenFieldState`

- `point_tokens`
- `visual_tokens`
- `tactile_tokens`
- `context_tokens`
- `fused_tokens`
- `point_positions`
- `modality_ids`
- alignment embeddings for point / visual / tactile
- tactile per-step routing metadata:
  - `tactile_tokens_all`
  - `tactile_tokens_active`
  - `tactile_group_ids`
  - `tactile_contact_prob`
  - `tactile_anchor_mask`
  - `tactile_normals_world`
  - `tactile_contact_score`
  - `tactile_contact_score_ema`
- `fusion_attention_mean`
- `projective_geometry`

`PicfObservationAnchorState`

- `seed_indices`
- `tokens`
- `point_weights`
- `routing_mass_point`
- `routing_mass_visual`
- `routing_support_point`
- `routing_support_visual`
- `routing_gate_point`
- `routing_gate_visual`
- geometry summaries `x, S, a`
- tactile routing extensions already exist:
  - `routing_mass_tactile`
  - `routing_support_tactile`
  - `routing_gate_tactile`

`PicfPosteriorAnchorState`

- recurrent core: `h, c`
- belief state: `mu, Sigma`
- geometry state: `x, S, a`
- activity / contact / support:
  - `alpha`
  - `contact_prob`
  - `support_mass`
  - `recycle_gate`
- binding / evidence:
  - `binding`
  - `evidence_tokens`
- exported posterior tokens:
  - `tokens`
  - `global_post`

`PicfPredictiveState`

- semantic side:
  - `semantic_tokens`
  - `innovation_token`
  - `innovation_norm`
  - `availability`
- current dual control path residues:
  - `control_tokens`
  - `action_condition_tokens`
  - `control_query_state`
  - `pooled_state`
- action fields:
  - `action`
  - `action_chunk`
  - `executed_action`
- physical predictive side:
  - `physical_global_pred`
  - `physical_prediction_cache`
- conditioned predictive side:
  - `predictive_query_state`
  - `global_pred`
  - `prediction_cache`

`PicfCoreState`

- `runtime_meta`
- `G_t`
- `token_field`
- `observation_anchors`
- `posterior`
- `predictive`
- `control`
- `last_prompt`

This audit matters because v2.2 is not just adding new objects. It also removes
the independent semantic meaning of the current control-related fields in
`PicfPredictiveState`.

### 3.9 Current `PicfFullCore.__init__`: Exact Module Inventory

Current `PicfFullCore` already contains most of the building blocks required by
the v2.2 patch. The important ones are:

Physical token construction:

- `point_token_proj`
- `visual_token_proj`
- `tactile_token_proj`
- `point_align_proj`
- `visual_align_proj`
- `tactile_align_proj`
- `token_fusion`

Observation-anchor and posterior update:

- `obs_reader`
- `obs_self`
- `anchor_seed_proj`
- `anchor_reader`
- `vote_heads`
- `post_write_proj`
- `post_lstm`
- `posterior_token_proj`
- `posterior_self`
- `posterior_pool`

Native reread paths already live in current code:

- `visual_native_reread`
- `tactile_native_reread`
- `point_native_reread`
- `tactile_group_route_queries`
- `tactile_route_reread`

Innovation and target heads:

- latent query banks:
  - `visual_latent_queries`
  - `tactile_latent_queries`
  - `point_latent_queries`
- prediction heads:
  - `visual_latent_head`
  - `visual_real_head`
  - `tactile_real_head`
  - `point_real_head`
- error encoders:
  - `visual_error_encoder`
  - `visual_real_error_encoder`
  - `tactile_error_encoder`
  - `point_error_encoder`
- `innovation_proj`
- `innovation_token_proj`

Current control / future modules:

- `posterior_to_control_proj`
- `global_post_to_control_proj`
- `innovation_to_control_proj`
- `proprio_to_control_proj`
- `control_role_embedding`
- `control_query_tokens`
- `control_world`
- `control_state_proj`
- `physical_pred_to_conditioned_proj`
- `predictive_conditioned_role_embedding`
- `predictive_query_tokens`
- `predictive_world`
- `predictive_semantic_world`
- `predictive_state_proj`

This is why v2.2 is a refactor, not a rewrite. Most required primitives
already existed and were reused.

### 3.10 Current `_build_token_field(...)`: Recursive Flow

Current `_build_token_field(...)` already does more than a simple concatenation.
Its real stages are:

1. initialize empty tensors for all modalities and all tactile routing fields
2. build projective geometry from:
   - point positions
   - current visual grid shape
3. construct point tokens:
   - point backbone features
   - RGB colors
   - point positional encoding
   - projection features
4. construct visual tokens:
   - flattened current V-JEPA map
   - grid features
   - camera pose features
   - ray features
5. construct tactile per-sensor base tokens:
   - pooled tactile feature
   - bundle global feature
   - world position
   - world rotation
6. run tactile contact/hysteresis logic
7. choose active tactile groups
8. expand each active tactile group into multiple public routing proposals using:
   - `tactile_group_route_queries`
   - `tactile_route_reread`
9. build context tokens:
   - proprio context
   - previous action context
   - timing context
   - contact context
10. run `token_fusion` **only** over:
    - point tokens
    - tactile active proposal tokens
    - context tokens
11. export `PicfTokenFieldState`
12. export `_StepDenseMemory`

Consequences for v2.2:

- current tactile group proposal routing already exists and is reused
- current visual public tokens exist separately from fused tokens
- current `fused_tokens` must not be mistaken for full multimodal public
  memory

### 3.11 Current `_build_observation_anchors(...)`: Recursive Flow

Current observation-anchor construction is already two-stage:

1. initialize learned / point-seeded queries
2. do native visual reread first:
   - `queries, visual_weights = self.visual_native_reread(...)`
3. then do public fused read:
   - `queries, attn_public = self.obs_reader(...)`
4. run `obs_self`
5. derive routing masses:
   - point routing from `attn_public`
   - visual routing from native reread
   - tactile routing by aggregating proposal-token masses back to tactile group
     ids
6. derive anchor geometry summaries

This is a strong base for v2.2. It confirms that:

- physical observation anchors already do native-first visual competition
- tactile ownership already exists at group level in the physical path
- semantic does not belong here

### 3.12 Current `_posterior_update(...)`: Recursive Flow

Current posterior update is already a multi-source evidence fusion block:

1. construct current prior from previous state
2. compute observation-anchor binding logits
3. apply dustbin sinkhorn
4. compute recycle / residual summary path
5. build anchor reader query from prior hidden + latent + geometry
6. read observation-anchor evidence via `anchor_reader`
7. native visual reread:
   - `binding_cond @ routing_mass_visual`
   - gather top-k visual payload candidates
   - `visual_native_reread`
8. native tactile reread:
   - `binding_cond @ routing_mass_tactile`
   - gather winning tactile groups
   - `tactile_native_reread`
9. fuse measurement evidence
10. update geometry summaries
11. run vote heads
12. do precision fusion
13. update posterior recurrent state
14. emit posterior tokens and global summary

This means v2.2 does not re-invent dense reread. It already exists in
the physical posterior path and remains untouched.

### 3.13 Current `_current_targets(...)` and `_innovation(...)`

Current targets are already denser than the original coarse version:

- visual latent target from native V-JEPA payload probes
- visual real target from RGB downsample
- tactile latent target from dense tactile group tokens
- tactile real target from:
  - tactile latent
  - tactile map
  - tactile auxiliaries
- point latent target from point payload probes
- point real target from:
  - point latent
  - occupancy

Current `_innovation(...)` then:

1. reads only `previous.predictive.physical_prediction_cache`
2. compares branchwise against current targets
3. standardizes residuals
4. encodes per-branch residual features
5. concatenates branch features + availability
6. produces one innovation token and branch norms

This confirms:

- innovation is already correctly world-only
- v2.2 preserves the current physical innovation semantics exactly

### 3.14 Current `_predictive_state(...)`: Post-v2.2 Role

Current `_predictive_state(...)` is no longer the place where control semantics
are invented. It now has a narrower role:

1. accepts already-built `conditioned_control`
2. resolves executed action / action chunk
3. builds the physical predictive basis
4. builds the conditioned future cache from:
   - physical predictive token sequence
   - future-condition tokens
5. emits predictive state fields plus compatibility/debug aliases

The control-semantics split was moved out of `_predictive_state(...)` into:

- `observe_step(...)`
- `_build_task_readout(...)`
- `_build_conditioned_control_state(...)`
- `finalize_with_action(...)`

### 3.15 Current `refresh_predictive_state_for_action(...)` and `step(...)`

`refresh_predictive_state_for_action(...)` now acts as a compatibility bridge
around the new observe/finalize split. It reconstructs the minimal observed
state needed to re-run predictive finalization with the externally supplied
action chunk.

Current `step(...)` is also a compatibility wrapper only. The canonical live
path is:

- `observe_step(...)`
- PI0.5 action generation / teacher-forced action resolution
- `finalize_with_action(...)`

Trainer and serve no longer depend on `step(...)` as the primary action API.

### 3.16 Current Wrapper / PI0.5 Audit

Current `src/openpi/picf/paligemma/wrapper.py` already restored the PI0.5 stack:

- `PaliGemmaWithExpertModel`
- `gemma_expert`
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`
- state injection back into prompt tokenization
- flow loss:
  - `x_t`
  - `u_t`
  - `v_t`
  - denoised target recovery as `x_t - t * v_t`
- denoise sampler using cached PI0.5 prefix state

Current wrapper contract already supports:

- `encode_observation(...)`
- `supports_pi0_action_generation()`
- `compute_action_flow_loss(...)`
- `sample_action_chunk(...)`

Therefore v2.2 does not need to redesign PI0.5 integration. It only needs to
replace the source of `extra_prefix_tokens`.

### 3.17 Current Trainer Audit

Current `scripts/picf_core_train.py` still sequences training manually:

1. semantic encode current observation
2. `core.step(...)`
3. if PI0.5 available:
   - `compute_action_flow_loss(...)`
   - pass `output.state.predictive.action_condition_tokens` as extra prefix
   - override action fields in state
4. call `compute_transition_loss(...)`

Current trainer also already contains important runtime logic that v2.2
preserves:

- compat checkpoint loader
- shape mismatch filtering
- allowlists for missing/unexpected keys
- DDP guards
- invalid first-step rejection sampling
- gradient clipping / logging

So the trainer is not simplified blindly. Only the action/control glue moved
behind `PicfPi05Policy`.

### 3.18 Current Serve Audit

Current `scripts/serve_picf_policy.py` sequences inference manually:

1. encode semantic observation
2. `core.step(..., action_future=None)`
3. sample PI0.5 action chunk using
   `output.state.predictive.action_condition_tokens`
4. call `core.refresh_predictive_state_for_action(...)`
5. export first action

This confirms exactly what the v2.2 exported policy now unifies.

### 3.19 Current Verifier and Tests Audit

Current `scripts/verify_picf_contract.py` is still aligned to the current live
semantic-prefix-primary core semantics. It currently asserts things such as:

- control prefix uses full semantic prefix directly
- conditioned future uses full semantic prefix directly
- task-sidecar path is absent

So verifier changes are not optional in v2.2; they are part of the contract
rewrite.

Current tests already cover:

- physical posterior / innovation invariants
- native visual reread
- tactile group routing / winner read
- serve-time predictive refresh
- wrapper PI0.5 action generation
- trainer loss override behavior

This is useful because v2.2 can extend an existing test base rather than
starting from zero.

### 3.20 Current Default Configuration Snapshot

The current live defaults that v2.2 treats as baseline, not as part of
the refactor, are:

`PicfCoreConfig`

- `persistent_anchors = 8`
- `observation_anchors = 16`
- `effector_persistent_anchors = 1`
- `effector_observation_anchors = 1`
- `global_scene_point_cap = 1024`
- `visual_real_grid = 64`
- `task_local_queries = 8`
- `task_effector_queries = 1`
- `hidden_dim = 512`
- `posterior_hidden_dim = 512`
- `latent_dim = 112`
- `innovation_dim = 512`
- `control_dim = 512`
- `semantic_dim = 2048`
- `semantic_cross_dim = 2048`
- `future_hidden_dim = 512`

Anchor-role default:

- of the `8` recurrent posterior slots, the first `1` remains effector/contact
  and the remaining `7` are scene/object slots
- of the `16` observation anchors, the first `1` is sampled from the local
  effector/contact point pool and the remaining `15` are sampled from the global
  scene point pool
- no hard background slot is reserved in the live default. Background is handled
  through scene/object slots plus task global/instruction tokens. A dedicated
  background role should only be added after a measured diagnostic shows stable
  non-object background evidence that should be recurrently maintained; adding
  it blindly would reduce object capacity from the default `7` scene slots.

Visual-real target:

- `visual_real_grid = 64` means the future RGB supervision target is a
  `64x64` downsample of the current/future static RGB frame
- this replaces the historical `4x4` diagnostic target, which was useful for
  cache plumbing but too coarse to inspect meaningful image prediction
- direct `256x256` linear prediction is intentionally not the 2x40GB default:
  it would make `visual_real_dim = 196608` and create a large output/error-head
  parameter and activation footprint. Use a decoder-style visual head before
  promoting `256x256` to a standard profile.
- `attention_heads = 8`
- `tactile_group_proposals = 2`
- `visual_reread_topk = 32`
- `tactile_reread_groups = 2`

`PaliGemmaSemanticConfig`

- `pi05 = True`
- `action_dim = 32`
- `action_horizon = 16`
- `denoise_steps = 10`
- `inject_state_into_prompt = True`
- `prompt_state_normalization = quantile` for CALVIN-aligned training
- `prompt_state_norm_stats_path = same CALVIN norm_stats.json used by action normalization`

Important boundary:

- prompt-state normalization is applied only on the semantic prompt-tokenization
  path before state discretization
- prompt-state tokenization uses the live CALVIN `robot_obs` / `proprio`
  dimensionality, matching the reference transform order where
  `TokenizePrompt(...)` runs before `PadStatesAndActions(...)`
- the zero padding to `action_dim = 32` is only for the model state/action tensor
  contract, not for the text prompt's discretized state string
- raw `robot_obs` / `proprio` stay untouched for the PICF physical core
- this preserves the PICF physical boundary while matching the reference
  `pi0.5_sonata` preprocessing contract, where `Normalize(norm_stats)` happens
  before `TokenizePrompt(...)`

Current training/runtime assumptions relevant to v2.2:

- trainer already expects compat loading and non-strict migration paths
- trainer already has DDP-specific gradient-checkpointing guards
- serving already assumes normalized-core / unnormalized-environment action
  contract

These are operating assumptions to preserve during the refactor, not knobs to
revisit inside the same patch.

## 4. Canonical v2.2 Shape

The current v2.2 system has exactly these canonical objects:

- physical recurrent state: `W_t`
- conditioned current-step control state: `C_t`
- world-only predictive basis: `K_t^{phys}`
- conditioned predictive cache: `K_t^{cond}`
- final action path: `a_t ~ PI0.5(S_t, C_t^{pi})`

High-level flow:

```text
observation_t
-> token_field_t + dense_memory_t
-> observation_anchors_t
-> posterior_t = W_t
-> targets_t
-> innovation_t from previous K_{t-1}^{phys}
-> semantic tokens S_t
-> task readout R_t over [public_read_memory_t, dense_memory_t]
-> conditioned control state C_t
-> PI0.5 flow loss / sampler using C_t^{pi}
-> executed_action a_t
-> K_t^{phys} = physical predictive basis from [W_t, a_t, proprio]
-> K_t^{cond} = conditioned future from [K_t^{phys}, C_t]
```

## 5. Non-Negotiable v2.2 Invariants

These are hard constraints, not suggestions.

### 5.1 Physical Core Remains Language-Free

Semantic must not enter:

- `_build_observation_anchors(...)`
- `_posterior_update(...)`
- `_innovation(...)`

### 5.2 Innovation Base Remains World-Only

Next-step innovation may read only:

- `previous.predictive.physical_prediction_cache`

It must not read:

- `previous.predictive.prediction_cache`
- `previous.conditioned_control`
- `previous.task_readout`
- semantic-conditioned state of any kind

### 5.3 `C_t` Is the Only Canonical Conditioned Control State

After the refactor, these must no longer exist as independent control
semantics:

- `action_condition_tokens`
- `control_query_state`
- `pooled_state`

If any compatibility alias survives, it may only be a view/debug snapshot of
the canonical `conditioned_control`.

### 5.4 `C_t^{pi}` Is an Interface View, Not a Second Control State

`C_t^{pi}` is only the prefix representation exported to PI0.5.

It is derived from `C_t` and is not itself a second control-path object.

### 5.5 `task_readout` Is Current-Step Only

`task_readout` must not become:

- recurrent memory
- task posterior
- next-step prior input
- next-step innovation base

It exists only to build `C_t` and conditioned future context.

### 5.6 `K_t^{phys}` Must Be Computed After Executed Action Is Known

Training:

- use teacher-forced first action

Serving:

- use sampled first action from PI0.5 chunk

Then compute:

```text
K_t^{phys} = P_phys(W_t, a_t^{exec}, proprio_t)
```

Never predict `K_t^{phys}` from a placeholder action and then overwrite the
executed action later.

### 5.7 Serving Must Fail Fast Without PI0.5 Action Generation

Formal deployed serving must not silently fall back to placeholder action.

The only valid deployed action path is:

- `PicfPi05Policy.act(...)`

## 6. Implemented Corrections to the External Proposal

The external proposal was directionally correct. These corrections were applied
in the same patch.

### 6.1 Public Read Memory Must Preserve Existing Public Fusion

Because current `fused_tokens` excludes `visual_tokens`, v2.2 defines a new
explicit public read memory. That public memory does not bypass the existing
`token_fusion` result.

The corrected v2.2 definition is:

```text
public_read_memory =
[
  fused_tokens,
  visual_tokens,
]
```

Single-line contract form:

```text
public_read_memory = [fused_tokens, visual_tokens]
```

where:

- `fused_tokens` preserves the current fused point/tactile/context public field
- `visual_tokens` preserves the separate public visual branch

`task readout` reads this corrected public memory and then rereads private
dense memory. It does not regress to raw pre-fusion
`point_tokens/tactile_tokens_active/context_tokens` as its only public memory.

### 6.1A Point-Pool Role Split For Effector vs Object Anchors

The original anchor design reserves a small number of slots for the
end-effector/contact region and leaves the rest to global scene/object evidence.
The maintained implementation now makes that contract explicit instead of
letting all anchors compete inside the same 10 cm gripper crop.

Current point-token construction:

```text
local_effector_context
  = points inside crop_radius_m around TCP/tactile focus centers

global_scene_context
  = farthest-point sample over the full current point cloud
    capped by global_scene_point_cap

point_context
  = [local_effector_context, global_scene_context]

point_pool_ids
  = 0 for local_effector_context
  = 1 for global_scene_context
```

Important detail:

- `global_scene_context` is sampled from the whole frame, not from
  `not local_effector_context`
- this is intentional because, once the gripper reaches the target object, the
  target can be inside the 10 cm crop and still must remain visible to object
  anchors through its global-scene copy
- local physical/contact targets continue to use the local crop; adding the
  global point pool does not change the gripper-local occupancy target

Current role allocation:

```text
observation anchors:
  first effector_observation_anchors slots -> role 0 effector/contact
  remaining observation anchors            -> role 1 scene/object

posterior anchors:
  first effector_persistent_anchors slots -> role 0 effector/contact
  remaining persistent anchors            -> role 1 scene/object

task-local queries:
  first task_effector_queries slots -> role 0 effector/contact
  remaining local task slots        -> role 1 scene/object
```

Role-aware read masks:

- effector-role observation/task slots read local point-pool tokens
- scene/object-role observation/task slots read global point-pool tokens when
  global tokens are present
- scene/object-role task slots do not read tactile proposals directly
- global and instruction task slots remain unrestricted control/context slots
- visual public memory remains available to both role families

Role-aware posterior binding:

- effector posterior anchors bind only effector observation anchors
- scene/object posterior anchors bind only scene/object observation anchors
- this prevents recurrent object state from being overwritten by gripper-local
  observations just because the gripper crop is dense

This role split is still language-free in the physical core. Semantic tokens do
not enter observation-anchor construction or posterior binding; semantic enters
only later through task readout and conditioned control.

CLI knobs:

- `--effector-persistent-anchors`
- `--effector-observation-anchors`
- `--task-effector-queries`
- `--global-scene-point-cap`

### 6.2 Contract Rewrite Must Be Explicit

Current formal contract is semantic-prefix-primary in core control/future.

v2.2 replaces that with:

- PI0.5 keeps raw semantic stream
- core gets semantic influence only through `task_readout`
- `C_t` becomes the only conditioned-control object

This is a contract rewrite, not a minor cleanup.

### 6.3 `C_t` Must Keep Token-Level Richness

`C_t` must not collapse too early to one pooled vector.

It preserves at minimum:

- `base_tokens`
- `task_tokens`
- `tokens` (post-control-trunk sequence)

and only then derive:

- `pi_prefix_tokens`
- `future_condition_tokens`

### 6.4 Instruction Query Count Should Not Stay at 1

If raw semantic prefix is removed as a direct core control/future input, one
instruction token is too aggressive a bottleneck.

The current live default is:

- `task_instruction_queries = 2`

This is the conservative default. It keeps instruction semantics richer without
exploding complexity.

### 6.5 Compat Loader Migration Must Ship in the Same Patch

Current trainer uses explicit compat filters for:

- allowed missing keys
- allowed unexpected keys
- shape-mismatch filtering before relaxed load

Any v2.2 patch that changes:

- control role embeddings
- control/predictive query tokens
- new task-readout modules
- removal of old semantic-prefix-primary modules

updates compat loading in the same patch. Otherwise warm-start from current
checkpoints will fail abruptly.

### 6.6 DDP Safety Guards Stay in Trainer/Runtime Layer

`PicfPi05Policy` unifies action/state interfaces. It does **not** absorb
runtime/DDP-specific guards that are already working in the training script.

Those include:

- gradient-checkpointing restrictions under DDP
- pre-DDP rejection sampling for invalid first-step windows
- compact startup logging
- current compat checkpoint loader behavior
- post-sampler predictive refresh timing

## 7. Detailed v2.2 Dataflow

### 7.1 Observation and Physical World Update

Unchanged:

```text
observation_t
-> point / visual / tactile feature extraction
-> token_field_t + dense_memory_t
-> observation_anchors_t
-> posterior_t = W_t
-> targets_t
-> innovation_t from previous.K_{t-1}^{phys}
```

### 7.2 Semantic-Conditioned Task Readout

New:

```text
semantic tokens S_t
-> condition task queries
-> read public_read_memory_t
-> private reread over:
   - dense_memory.visual_payload
   - dense_memory.tactile_group_tokens
   - dense_memory.point_payload
-> task_local_tokens
-> task_global_token
-> instruction_tokens
-> geometry summaries (x, S, a)
```

This stage is current-step only.

### 7.3 Unique Conditioned Control State

New:

```text
base_tokens =
[
  posterior.tokens,
  posterior.global_post,
  innovation_token,
  proprio_token,
]

task_tokens =
[
  task_local_tokens,
  task_global_token,
  instruction_tokens,
]

C_t = control_world([base_tokens, task_tokens, conditioned_control_queries])
```

Then derive:

```text
C_t^{pi} = pi_prefix_reader(C_t.tokens)
C_t^{future} = future_condition_reader(C_t.tokens)
```

### 7.4 Unique Final Action Path

Training:

```text
flow_loss = semantic_encoder.compute_action_flow_loss(
    semantic_features,
    extra_prefix_tokens=C_t^{pi},
    action_chunk_target=teacher_chunk,
)
```

Serving:

```text
sampled_chunk = semantic_encoder.sample_action_chunk(
    semantic_features,
    extra_prefix_tokens=C_t^{pi},
)
```

No second action path may remain.

### 7.5 Physical Predictive Basis and Conditioned Future

```text
K_t^{phys} = P_phys(W_t, a_t^{exec}, proprio_t)
K_t^{cond} = P_cond(K_t^{phys}, C_t^{future})
```

This preserves:

- world-only predictive basis for innovation
- conditioned future for semantic/task-aware forecasting

## 8. File-by-File v2.2 Implementation Record

### 8.1 `src/openpi/picf/core/config.py`

Keep unchanged:

- `hidden_dim = 512`
- `posterior_hidden_dim = 512`
- `innovation_dim = 512`
- `control_dim = 512`
- `future_hidden_dim = 512`
- `semantic_dim = 2048`
- `attention_heads = 8`

Rationale: width churn is not part of this refactor.

Current file defines:

- `task_local_queries: int = 8`
- `task_global_queries: int = 1`
- `task_instruction_queries: int = 2`
- `task_self_layers: int = 1`
- `conditioned_control_queries: int = 4`
- `pi_prefix_queries: int = 4`
- `conditioned_future_queries: int = 2`
- `task_visual_reread_topk: int = 32`
- `task_tactile_reread_groups: int = 2`
- `task_point_reread_topk: int = 32`
- `require_pi0_action_generator: bool = True`

Reserved / compatibility-only field:

- `task_query_rounds: int = 2`
  - retained in `PicfCoreConfig` for a not-yet-implemented iterative task-readout variant
  - not currently consumed by the live v2.2 core/trainer path

Mark as deprecated compatibility-only:

- `predictive_semantic_reads`
- `control_semantic_reads`
- `predictive_semantic_dropout_prob`
- `semantic_prefix_dropout_prob`

### 8.2 `src/openpi/picf/core/contracts.py`

Add:

`PicfTaskReadoutState`

- conditioned semantic queries
- local tokens
- global token
- instruction tokens
- point weights
- geometry summaries `x, S, a`
- public/private attention diagnostics

`PicfConditionedControlState`

- base tokens
- task tokens
- unified control tokens
- query state
- pi prefix tokens
- future condition tokens

Current `PicfPredictiveState` semantics were reworked as follows:

- remove independent control-semantics status from:
  - `control_tokens`
  - `action_condition_tokens`
  - `control_query_state`
  - `pooled_state`
- retain only real predictive/action outputs:
  - semantic tokens
  - innovation token / norm
  - availability
  - action
  - action_chunk
  - executed_action
  - physical_global_pred
  - physical_prediction_cache
  - global_pred
  - prediction_cache

Current `PicfCoreState` was extended to:

- add `task_readout`
- add `conditioned_control`

Current-to-target field mapping must be written explicitly in code comments and
compat migration notes:

- current `predictive.action_condition_tokens`
  -> target `conditioned_control.pi_prefix_tokens`
- current `predictive.control_tokens`
  -> target `conditioned_control.tokens`
- current `predictive.control_query_state`
  -> target `conditioned_control.query_state`
- current `predictive.pooled_state`
  -> optional derived debug summary only; no longer canonical state

### 8.3 `src/openpi/picf/core/pipeline.py`

Do not structurally weaken:

- `_build_token_field`
- `_build_observation_anchors`
- `_posterior_update`
- `_current_targets`
- `_innovation`

Allowed structural fixes in these helpers must preserve the formal boundary:

- semantic still does not enter observation anchors, posterior update, or
  innovation
- physical/contact targets still use the effector-local point crop where that is
  the supervised target
- task/object selection may add explicit point-pool roles so the integration
  layer no longer forces every anchor candidate to come from the gripper crop

Current helper: `_build_public_read_memory(...)`

Construct:

```text
public_read_memory =
[
  fused_tokens,
  visual_tokens,
]
```

Do not collapse this back to `fused_tokens` alone, but also do not bypass the
existing fused public field by reverting to raw pre-fusion point/tactile/context
tokens as the only public read source.

Important correction:

- `fused_tokens` stays the current fused point/tactile/context field
- `public_read_memory` becomes the new explicit public task-readout
  memory
- v2.2 does not overload one tensor to mean both

Current helper: `_build_task_readout(...)`

Inputs:

- token_field
- public_read_memory
- dense_memory
- semantic
- proprio token

Hard rule:

- `_build_task_readout(...)` must not take `posterior` as a direct input

Flow:

1. condition learned task queries from semantic tokens
2. read public read memory
3. reread visual private payload
4. reread tactile private groups
5. reread point private payload
6. run lightweight task self stack
7. derive geometry summaries from point attention
8. output `PicfTaskReadoutState`

This helper must explicitly read:

- `token_field.fused_tokens`
- `token_field.visual_tokens`
- `dense_memory.visual_payload`
- `dense_memory.tactile_group_tokens`
- `dense_memory.point_payload`

It must not directly consume `posterior`. Posterior only merges with task
readout later inside `_build_conditioned_control_state(...)`.

It must not silently fall back to reading only observation-anchor tokens.

Implementation hard requirement:

- if `task_query_conditioner` is built from `GatedCrossAttentionRead`, it must
  not inherit the current dormant `cross_gate=0` startup behavior unchanged
- either add a nonzero `gate_init` path, or use a dedicated ungated semantic
  conditioner for this block

Current implementation status:

- `task_query_conditioner` uses `gate_init=1.0`
- semantic query conditioning is therefore active from initialization rather
  than waiting for a dormant gate to open
- private visual / point / tactile reread blocks are still lazy gated readers;
  this is intentional, but early or partial checkpoints can under-use dense
  private reread until those gates learn to open

### 8.3.1 Task-Readout Anchor Diagnostics And Interpretation

The CALVIN anchor overlay uses three visual layers and role colors:

- observation anchors: circles
  - orange = effector/contact observation slots
  - yellow = scene/object observation slots
- posterior anchors: red circles
  - purple = effector/contact posterior slots
  - red = scene/object posterior slots
- task-readout local slots: crosses
  - magenta = effector/contact task slots
  - cyan = scene/object task slots

The cyan/object crosses require careful interpretation.

Current code path:

```text
task_readout.point_weights
  = normalized point_public_attention[:task_local_queries]

task_readout.x
  = task_readout.point_weights @ token_field.point_positions

task.pixel
  = task_readout.point_weights @ projected_point_pixels
```

Therefore:

- `task.pixel` is a **point-public-attention centroid**
- it is not the same tensor as semantic attention
- it is not the same tensor as visual public attention
- it is not the same tensor as private V-JEPA / visual reread attention

Important risk:

- object-role task pixels clustering on the gripper does **not** by itself prove that
  PaliGemma language understanding failed
- it proves only that the current point-space projection of the local task
  readout is gripper-proximal
- this can happen when point tokens around the gripper dominate the point-public
  geometric projection even if semantic or visual attention still moves toward
  the task object

The maintained debug server now exports extra JSON diagnostics under:

```text
anchor_debug.task.attention
```

Fields:

- `anchor_debug.observation.role_ids`: observation-anchor roles
- `anchor_debug.posterior.role_ids`: posterior-anchor roles
- `anchor_debug.task.local_role_ids`: task-local roles
- `anchor_debug.point_cloud.pool_ids`: point-token pool ids, where `0` means
  effector-local and `1` means global-scene
- `semantic`: entropy and top-k semantic token indices for task queries
- `public`: entropy and top-k over the combined `[fused_tokens, visual_tokens]`
  public memory
- `visual_public`: top visual grid cells and projected pixels
- `point_public`: top point tokens, xyz, and projected pixels
- `tactile_public`: tactile-public attention summary
- `visual_private`, `point_private`, `tactile_private`: private reread
  attention summaries; candidate identities are local candidate slots because
  the current `PicfTaskReadoutState` does not store gather indices
- `slot_diversity`: pairwise diversity for task xyz and projected task pixels
- `near_proprio_point_mass_10cm` / `near_proprio_point_mass_20cm`: heuristic
  mass near the first three `proprio` coordinates; this should be read as an
  end-effector-proximity diagnostic, not as a formal object-label metric

Decision rules:

- if `semantic` and `visual_public` attention change with prompt while cyan
  pixels remain on the gripper, the likely issue is point-projection or
  effector/object disentanglement, not a language-input failure
- if object-role slots have `role_ids=1` but their top `point_public` entries
  mostly have `point_cloud.pool_ids=0`, the role mask has regressed
- if object-role slots have `role_ids=1` and top points have `pool_ids=1` but
  still project to the gripper, the global scene pool itself is gripper-heavy
  and point-cloud sampling/camera projection should be inspected
- if `semantic`, `visual_public`, and `point_public` are all nearly
  prompt-invariant across different instructions, the semantic conditioning
  path or prompt-state path is suspect
- if `slot_diversity` is very low and all local slots share the same top-k point
  tokens, the task slots are collapsing
- if near-proprio mass is high for most local slots, inspect whether the model
  is using the gripper as a useful manipulation reference or incorrectly
  replacing object slots with effector slots

Current audit files for the 2026-05-03 follow-through:

- `/tmp/picf_v22_anchor_role_dataflow_20260503.md`
- `/tmp/picf_v22_anchor_role_theory_20260503.md`
- `/tmp/picf_v22_anchor_role_final_report_20260503.md`

Local verification from that pass:

- `python -m py_compile src/openpi/picf/core/contracts.py src/openpi/picf/core/config.py src/openpi/picf/core/pipeline.py src/openpi/picf/frame_context.py scripts/picf_core_train.py scripts/serve_picf_policy.py scripts/calvin/evaluate_picf_policy.py`
- AST/static checks for the new point-pool fields, role masks, debug helpers,
  CLI flags, and README navigation
- README grep checks for the role/pool diagnostic contract and navigation

Known local limitation:

- the local Python environment currently fails on `import torch` before repo
  code runs because `torch.autograd.profiler_legacy` is missing; runtime tensor
  tests for this diagnostic path should be run on the cloud image where torch is
  healthy

Do not change the training architecture based only on cyan overlay videos.
First compare the JSON attention summaries across prompts such as
`open drawer`, `move slider`, and `push block`.

Current helper: `_build_conditioned_control_state(...)`

Inputs:

- posterior
- innovation token
- proprio token
- task readout

Flow:

1. project physical base tokens to semantic width
2. project task-readout tokens to semantic width
3. concatenate base + task + conditioned control query tokens
4. run a single `control_world`
5. export:
   - unified control tokens
   - query state
   - `pi_prefix_tokens`
   - `future_condition_tokens`

This helper replaces both current control routes inside `_predictive_state(...)`:

- current PI0.5 action-conditioning route
- current semantic-prefix-primary internal control route

After v2.2 there is exactly one conditioned-control route through
`control_world(...)`.

Current helper: `_build_physical_predictive_basis(...)`

Inputs:

- posterior
- executed action
- proprio token

Outputs:

- physical predictive tokens
- physical global pred
- physical prediction cache

This helper owns the world-only predictive-basis logic and must only run after
executed action is known.

Current helper: `_build_conditioned_predictive_cache(...)`

Inputs:

- physical predictive token sequence
- conditioned control future tokens

Outputs:

- conditioned `global_pred`
- conditioned `prediction_cache`

This helper becomes the only conditioned future constructor. Raw semantic prefix
must not be concatenated directly here after the refactor.

The intended token-level contract is:

```text
K_t^{cond} = P_cond(H_t^{phys_pred}, C_t^{future})
```

Do not collapse this helper to a summary-only global/cache path.

The API is restructured as:

- add `observe_step(...)`
- add `finalize_with_action(...)`
- keep `step(...)` as compatibility wrapper only

`step(...)` no longer remains the official serving/export entrypoint.

Intended split:

- `observe_step(...)`: canonical pre-action stage
- `finalize_with_action(...)`: canonical post-action stage
- `step(...)`: compatibility wrapper only

### 8.4 `src/openpi/picf/paligemma/wrapper.py`

Keep action generation logic intact:

- `compute_action_flow_loss(...)`
- `sample_action_chunk(...)`

Rename interface semantics in code/docs:

- `extra_prefix_tokens` -> conceptually `pi_action_condition_tokens`

The implementation keeps signature compatibility where needed, and internal
naming/documentation reflect PI0.5 action-conditioning semantics.

Require:

- `supports_pi0_action_generation()` fail-fast enforcement when config requires
  deployed action generation

Wrapper non-goals for v2.2:

- do not redesign PI0.5 flow equations
- do not redesign denoise scheduling
- do not redesign prompt-side state injection
- do not redesign checkpoint topology

v2.2 changes the source and semantics of action-conditioning tokens, not the
PI0.5 generator itself.

### 8.5 `src/openpi/picf/policy.py`

Current exported policy class:

`PicfPi05Policy`

Fields:

- `core: PicfFullCore`
- `semantic_encoder: PaliGemmaSemanticEncoder`

Methods:

- `forward_train_transition(...)`
- `forward_window(...)` if batching windows at policy level is useful
- `act(...)`

Engineering rule:

- the policy surface returns typed result objects, not bare dictionaries
- the training-facing result exposes:
  - `output`
  - `observed`
  - `semantic_override`
  - `flow_override`
  - `next_state` as the compact recurrent carry, not the full `PicfCoreState`
- the serving-facing result exposes:
  - `action`
  - `action_chunk`
  - `state`
  - `debug`
  - `output`

This keeps the exported interface stable and prevents trainer/serve from
silently depending on ad-hoc string keys.

Training flow:

```text
encode semantic
-> core.observe_step(...)
-> compute PI0.5 flow loss with C_t^{pi}
-> finalize core with teacher-forced executed action
-> compute transition loss using action overrides
```

Teacher-forced executed-action rule:

- prefer `current.action` when present
- otherwise use the first action from `action_chunk_target`
- never silently depend on dataset-specific implicit assumptions that both must
  always be present

Serving flow:

```text
encode semantic
-> core.observe_step(...)
-> sample PI0.5 chunk with C_t^{pi}
-> take first action
-> finalize core with sampled executed action
-> return action + next state
```

`PicfPi05Policy` is the only exported semantic+core action surface used by:

- trainer
- serving
- rollout/eval entrypoints

### 8.6 `scripts/picf_core_train.py`

The trainer no longer treats this manual distributed glue sequence as the
canonical action path:

- `core.step(...)`
- `semantic_encoder.compute_action_flow_loss(...)`
- state action mutation
- `compute_transition_loss(...)`

It now uses:

- construct `PicfPi05Policy`
- call `policy.forward_train_transition(...)`

Loss-side supervision rule:

- `compute_transition_loss(...)` must build future targets from the next
  observation as stop-gradient teacher signals
- `extract_future_targets(...)` therefore wraps `core.extract_targets(...)` in
  `torch.no_grad()`
- next-observation targets are supervision values, not a second trainable branch
  of the same transition graph
- when `unroll_steps > 1`, the window trainer now uses a one-step lookahead:
  the already-computed `observed.current_targets` from transition `t+1` are
  detached and reused as the future supervision targets for transition `t`
- therefore shared middle frames inside one training window are not rebuilt
  twice on the loss side; only the final frame in the window still needs an
  explicit `extract_future_targets(...)` pass

Historical trainer glue removed as first-class action integration:

- direct canonical dependence on `core.step(...)`
- direct canonical dependence on `semantic_encoder.compute_action_flow_loss(...)`
- direct use of `output.state.predictive.action_condition_tokens`
- post-hoc mutation of predictive action fields as the normal training path

Retain:

- current DDP safety guards
- current compat checkpoint loader
- current invalid-window rejection behavior
- current grad clipping / logging infrastructure

These are already validated runtime guards and remain in trainer/runtime
scope after policy unification.

Additional trainer runtime rule:

- in multi-rank runs, `scripts/picf_core_train.py` must not silently inherit
  `TORCH_DISTRIBUTED_DEBUG=DETAIL` as the default runtime mode
- DDP startup now defaults `TORCH_DISTRIBUTED_DEBUG` to `INFO`
- if `TORCH_DISTRIBUTED_DEBUG=DETAIL` is present under DDP, the trainer
  fails fast at startup unless
  `OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL=1` is explicitly set
- this guard exists because DETAIL-level TCPStore/NCCL trace traffic can
  destabilize standalone multi-rank bring-up and produce misleading
  `Broken pipe` heartbeat noise even while training continues
- DDP launch also fails fast if `LOCAL_RANK` is missing

Current standard long-run launch profile:

- `--num-train-steps 30000`
- `--save-interval 2500`
- `--grad-clip-mode percentile`
- `--grad-clip-percentile 75`
- `--grad-clip-window 100`
- `--visual-activation-checkpointing`
- `--semantic-gradient-checkpointing`
- `--window-activation-checkpointing`
- `--keep-last-checkpoints N` prunes old numeric step checkpoint directories
  after each successful save while leaving non-step diagnostics and metadata
  intact. Use `--save-interval 2500 --keep-last-checkpoints 3` for long
  30000-step runs when disk pressure matters.
- `--diagnostic-interval 0`
- `--training-strategy fsdp_full_shard` for the current 4x40GB A100 FSDP investigation profile
- `--optimizer-sharding none` on that FSDP path; `zero1` remains a DDP-only fallback and is not sufficient for all-backbone v2.2 finetuning
- `--perception-finetune-mode auto|full|frozen` is now the canonical
  operator-facing backbone trainability contract. `full` preserves the default
  all-backbone profile; `frozen` freezes Sonata, V-JEPA, and AnyTouch while
  leaving the PI0.5 semantic/action stack and PICF heads trainable.
- `--picf-trainable-scope all|anchor_only` is the separate PICF trainability
  contract. `all` is the maintained default. `anchor_only` is a diagnostic
  large-batch probe: it freezes perception, semantic, PI0.5 action/control, and
  predictive heads after lazy materialization, then leaves only typed-evidence
  token adapters, AQR/MVTrack readers, observation-anchor adapters, posterior
  binding/address, and support/cache/local evidence modules trainable. It also
  forces semantic runtime into no-grad/inference mode and disables window-level
  activation checkpointing, because the profile is meant to maximize safe
  throughput for anchor diagnostics rather than train the policy stack. Optional
  MVTrack adapters such as tracklet/proposal token projectors are explicitly
  materialized during trainer warmup even when the current dataset lacks those
  modalities, so FSDP/DDP still audits a stable trainable parameter contract.
  On FSDP full-shard, fully frozen root modules are expanded to their
  parameters and combined with frozen root-managed parameters through the
  single `ignored_states` API for this scope; this keeps `use_orig_params=False`
  flat-parameter handles uniform in `requires_grad` while preserving the strict
  anchor allowlist.
  Use it to test whether anchors can separate and stabilize; do not treat it as
  final policy training.
- `--visual-finetune-mode full|frozen` remains a lower-level visual-backbone
  compatibility knob; prefer the top-level perception switch for maintained
  launch profiles.
- semantic FSDP wrapping now uses a two-level exact contract: directly called PI0/PaliGemma runtime hot leaves (`embed_tokens`, per-layer `q/k/v/o` projections, per-layer `mlp`, and PI0 action/time projections) are wrapped first as explicit nested leaves, and the remaining semantic root still uses `ignored_states` for the minority float32 stabilizer parameters. The SigLIP vision tower and multimodal projector currently remain under the outer semantic root because their current image-path implementations are not yet nested-FSDP-safe under the present view-alias constraints.
- FSDP full-shard on this profile should use flat-parameter mode (`use_orig_params=False`) together with `backward_prefetch=BACKWARD_POST` and `limit_all_gathers=True`; the goal is to reduce parameter-view residency and backward all-gather overlap peaks without changing model math
- standard 4x40GB FSDP sharding is now recursive for large uniform-dtype subtrees with a 512MiB parameter-storage budget per boundary; this lets point/visual/tactile backbone wrappers and safe internal stacks split into smaller shards instead of wrapping an entire uniform subtree as one flat unit
- safe core stacks are now explicit FSDP child boundaries on this profile: `token_fusion`, `obs_self`, `posterior_self`, `task_self`, `predictive_world`, `predictive_semantic_world`, and `control_world`; the trainer now reattaches those wrapped children back onto `core` before the root wrap, so root FSDP only carries the remaining lighter core/projection parameters instead of one monolithic core shard
- the root FSDP boundary now explicitly ignores fully frozen backbone subtrees
  instead of flattening mixed `requires_grad` parameter sets. This is required
  for `perception_finetune_mode=frozen` and is the mature contract for any
  future frozen backbone mode, because `use_orig_params=False` root flattening
  is only valid over uniform trainability.
- transformer-stack entry now materializes every incoming activation once (`x = x.clone()`) before attention. This is mathematically exact and is now part of the 4x40GB contract because many PICF call sites batch tokens via `tokens[None, :]`, while FSDP can also hand stacks storage-sharing tensors whose aliasing is not reliably visible through `_base`; a single stack-entry clone is the clean boundary that prevents autograd multi-view alias failures inside residual attention blocks
- FSDP grad-norm measurement and percentile clipping on this profile must use an explicit global L2 reduction over local gradient shards instead of `FullyShardedDataParallel.clip_grad_norm_`; the semantic stack intentionally carries both bf16 bulk weights and minority float32 stabilizer parameters, so the stock helper's uniform-dtype assumption is not a valid contract here
- semantic gradient checkpointing remains enabled on that FSDP path; after routing PI0 flow-loss calls through module `forward(op, ...)` and collapsing the semantic stack to one FSDP boundary, non-reentrant checkpoint recomputation is again the correct memory-saving path rather than a forbidden custom-method re-entry
- the training stack still supports checkpointing the full `_PicfWindowTrainer.forward(...)` window body during training. That remains an exact fallback for extra peak-memory reduction, and the checkpoint input is still a standalone dummy leaf on the active CUDA device rather than a view into any FSDP flat parameter, so recompute keeps exact training math without feeding full-parameter gradients back into local shard metadata. It is now an explicit operator knob rather than something the foundation profile silently forces on every launch.
- the custom PI0/Gemma dual-branch semantic attention path now uses SDPA instead of the eager attention workspace. This is part of the standard 4x40GB profile because the eager path materializes a large attention buffer that fits at step 1 but blows up once optimizer state becomes resident; SDPA preserves the same training objective while removing that workspace peak.
- core transformer stacks (`token_fusion`, `obs_self`, `posterior_self`, `task_self`, `predictive_world`, `predictive_semantic_world`, `control_world`) now also use train-time non-reentrant activation recompute; this is part of the standard all-backbone v2.2 training path and does not alter the underlying objective
- the trainable Sonata point backbone and AnyTouch tactile encoder now also use train-time non-reentrant recompute on their main backbone forwards; this keeps all-backbone finetuning mathematically identical while shifting more of the per-rank memory burden from saved activations into recompute
- tokenwise-only projections and FFNs on the current hottest paths now support exact sequence chunking instead of monolithic execution. On the current profile this is enabled by default as `tokenwise_ff_chunk_size=64` for PICF core transformer/cross-attention FFNs and `semantic_tokenwise_chunk_size=64` as the legacy semantic compatibility knob. The live trainer now also exposes `semantic_projection_chunk_size` and `semantic_mlp_chunk_size` as finer-grained execution controls; under the standard 4x40GB full-shard profile, the balanced default is `semantic_projection_chunk_size=128` and `semantic_mlp_chunk_size=64`. This preserves the old semantic compatibility surface while giving the heavier semantic MLP path and the lighter projection path different exact-memory execution policies.
- the PI0/PaliGemma wrapper no longer adds an extra outer checkpoint around semantic forward blocks when the native language-model / vision-tower / expert-model checkpointing path is already active. This avoids redundant recompute while preserving the same gradients.
- the PI0/PICF semantic runtime now drops the unused outer causal-LM heads (`paligemma.lm_head`, `gemma_expert.lm_head`) immediately after checkpoint load. The live training path never routes through those logits heads, so removing them from the runtime graph is mathematically exact and prevents dead generation weights from inflating FSDP wrapping.
- standard multi-rank `FSDP full_shard` training now also standardizes `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; after the backbone recompute cut, the dominant remaining failure mode was allocator fragmentation (`reserved but unallocated` growing much larger than free memory), so v2.2 now treats expandable segments as part of the clean startup contract rather than a post-hoc workaround
- standard FSDP full-finetune startup no longer serializes checkpoint construction rank-by-rank; each rank builds in parallel, because the serialized path turned one large checkpoint load into a multi-stage startup stall without changing training semantics
- standard FSDP full-finetune startup now also stages the local PI0/PaliGemma checkpoint from shared `/mnt` storage into a node-local cache before rank-local `safe_open(...)/load_state_dict(...)`; this preserves training math while removing the shared-filesystem page-wait stall that appeared when four ranks loaded the same multi-GB semantic checkpoint concurrently. The default cache root is `~/.cache/openpi/pi0_checkpoints`, `OPENPI_STAGE_PI0_CHECKPOINT=auto` stages only `/mnt/...` sources, and `OPENPI_LOCAL_CHECKPOINT_CACHE_DIR` overrides the cache location when needed
- V-JEPA mixed precision on CUDA now uses one safe autocast contract for both frozen and trainable modes. The encoder stays in native fp32 weights and the forward path enters autocast when `visual_dtype` is `float16`/`bfloat16`, avoiding frozen-path conv bias dtype mismatches without changing the feature contract.
- window training now carries a canonical recurrent-carry object instead of forwarding the full `PicfCoreState` into `previous`. The carry contains only the fields that the next step actually consumes: `runtime_meta`, tactile contact hysteresis state, `posterior`, `predictive.executed_action`, and `predictive.physical_prediction_cache`. This is mathematically exact for the current v2.2 recurrence contract and removes non-recurrent semantic/control/task-readout state from the cross-step training graph.

These values are the current operational training defaults for v2.2 runs even
if historical baseline commands in older docs still show `--save-interval 5000`.

Important status note:

- this section records the implemented 4x40GB FSDP training contract and the
  code paths that now exist in `scripts/picf_core_train.py`
- the maintained v2.2 README should be read as the current operator/developer
  contract for the 4x40GB full-train path, not as a promise that every listed
  runtime knob is part of the default launch profile
- this file records:
  - what is implemented in code
  - which runtime measures are standard defaults
  - which runtime measures are explicit operator fallbacks
- it still must not overclaim beyond current evidence:
  - this file does **not** claim that `step 2500` or full `30000` completion has
    already been observed unless a later audit explicitly records that fact

### 8.6.1 Current Standard 4x40GB Deployment Profile

The current standard 4x40GB training profile is:

- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `accum_steps=1`
- `num_train_steps=30000`
- `save_interval=2500`
- `grad_clip_mode=percentile`
- `grad_clip_percentile=75`
- `grad_clip_window=100`
- `use_foundation_backbones=True`
- `use_tactile=True`
- `perception_finetune_mode=full`
- `visual_finetune_mode=full`
- `visual_trainable=True`
- `tactile_trainable=True`
- `point_backbone_trainable=True`
- `semantic_trainable=True`
- `window_activation_checkpointing=False` by default
- `semantic_gradient_checkpointing=True`
- `tokenwise_ff_chunk_size=64`
- `semantic_tokenwise_chunk_size=64`
- `semantic_projection_chunk_size=128`
- `semantic_mlp_chunk_size=64`

This is a **full-train** profile.

It does **not** freeze:

- V-JEPA
- AnyTouch
- Sonata
- PaliGemma

It also does **not** rely on:

- LoRA
- CPU offload
- watchdog restart logic

### 8.6.1Z Current 6x40GB Full-PICF Extension

The same full-PICF contract can be launched on a 6x40GB A100 node by changing
only the distributed launch width:

- `--nproc_per_node=6`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`

The current 6-GPU long-run profile keeps the same v2.2 semantics:

- `picf_mode=enabled`
- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `accum_steps=1`
- `unroll_steps=2`
- `action_horizon=16`
- `num_train_steps=30000`
- `save_interval=2500`
- `log_interval=100`
- `window_activation_checkpointing=False`
- all foundation backbones trainable
- action normalization uses CALVIN `norm_stats.json`
- prompt-state normalization inherits the same CALVIN `norm_stats.json`

Important comparison note:

- the 4x40GB profile has `effective_global_batch=4`
- the 6x40GB extension has `effective_global_batch=6`
- this is intentional when all 6 GPUs are used with `accum_steps=1`
- loss curves from 4-GPU and 6-GPU runs should therefore be compared as
  same-objective but different-global-batch runs, not as bitwise-identical
  optimizer trajectories

The 6-GPU extension does not change:

- PI0.5 flow-matching action objective
- PICF observe/finalize dataflow
- physical posterior / innovation boundary
- task-readout / conditioned-control contract
- checkpoint cadence

### 8.6.1F Historical 2x40GB Frozen-Perception Full-BPTT Reference

This subsection records the earlier full-BPTT 2x40GB A100 frozen-perception
reference. It is **not** the current VL-router long-run launch profile. The
current live long-run contract is documented in
[`README_VL_GUIDED_ANCHOR_ROUTER.md` Section 6.1](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_VL_GUIDED_ANCHOR_ROUTER.md)
and uses `training_strategy=fsdp_full_shard`, `unroll_steps=2`,
`burnin_steps=0`, `tactile_mode=encoder`, frozen Sonata/V-JEPA/AnyTouch
feature extractors, trainable PaliGemma/PI0.5 semantic-action modules, and one
effector slot by default.

The historical reference was intended for cost-controlled runs where the PICF
architecture, PI0.5 action path, recurrent carry, task readout, conditioned
control, and future supervision all stay enabled, while the heavy perception
encoders are treated as fixed feature extractors.

Historical full-BPTT reference operator profile:

- `world_size=2`
- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `picf_mode=enabled`
- `accum_steps=1`
- `unroll_steps=3`
- `burnin_steps=0`
- `action_horizon=16`
- `num_train_steps=30000`
- `save_interval=5000`
- `log_interval=100`
- `semantic_max_length=256` primary (`200` only as memory/parity fallback)
- `visual_real_grid=64`
- `persistent_anchors=8`
- `observation_anchors=16`
- `perception_finetune_mode=frozen`
- `point_backbone=sonata`
- `visual_mode=encoder`
- `visual_feature_mode=hierarchical`
- `tactile_mode=encoder`
- `picf_augmentation_mode=photometric`
- `picf_photometric_strength=conservative`
- `semantic_gradient_checkpointing=True`
- `visual_activation_checkpointing=True`
- `window_activation_checkpointing=False`

Trainability contract:

- Sonata point backbone is frozen.
- V-JEPA visual backbone is frozen.
- AnyTouch tactile backbone is frozen.
- PaliGemma / PI0.5 semantic-action stack remains trainable in this historical
  reference and in the current VL-router long-run unless an explicit frozen-
  semantic ablation is launched. The current VL-router long-run freezes the
  Sonata, V-JEPA, and AnyTouch pretrained feature extractors, while the
  PaliGemma/PI0.5 semantic-action modules plus PICF/action-side adapters and
  prediction heads remain trainable.
- PICF task-readout, conditioned-control, posterior/predictive/future heads,
  and auxiliary losses remain active.

This is **not** the PI0.5-only ablation. It keeps `picf_mode=enabled`.

Geometry-safe augmentation contract:

- the enabled augmentation is conservative photometric augmentation only
- it does not apply crop / rotation / spatial warps
- this is intentional because full PICF uses image, point/depth, tactile, and
  geometric readout streams together; spatially warping RGB alone would break
  cross-modal geometry unless the same transform were applied coherently to all
  aligned modalities
- the augmentation is therefore safe for frozen-perception feature robustness
  without changing the physical anchor / posterior / innovation contract

Current 2x40GB evidence:

- an initial `unroll_steps=2` smoke run on two A100 40GB GPUs reached
  `step >= 10`
- `unroll_steps=3` was smoke-tested and remains the slower full-BPTT quality
  reference
- the selected 2x40GB long run now uses `burnin_steps=4`,
  `burnin_mode=state_only`, and `unroll_steps=1` with `save_interval=5000`
- `unroll_steps=8` was tested and failed with CUDA OOM during `step=1`
  backward
- `unroll_steps=4` was tested and failed with CUDA OOM during `step=2`
  backward
- do not run `unroll_steps=4` or `8` on this 2x40GB profile without changing
  memory strategy
- early `unroll_steps=3` speed was approximately `24-27 s/step`
- sampled inter-step memory can look much lower, but probe peaks reached the
  40GB limit; treat `unroll_steps=3` as the maximum validated setting, not as a
  roomy configuration
- if future sample variability causes OOM, fall back to `unroll_steps=2`
- do not increase `accum_steps` or `action_horizon` on 2x40GB without a fresh
  smoke test

Current selected sub-15s long-run profile:

- `world_size=2`
- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `picf_mode=enabled`
- `accum_steps=1`
- `unroll_steps=1`
- `burnin_steps=4`
- `burnin_mode=state_only`
- `effective_window_steps=5`
- `action_horizon=16`
- `num_train_steps=30000`
- `save_interval=5000`
- `log_interval=100`
- `semantic_max_length=256` primary (`200` only as memory/parity fallback)
- `visual_real_grid=64`
- `persistent_anchors=8`
- `observation_anchors=16`
- `perception_finetune_mode=frozen`
- `picf_augmentation_mode=photometric`
- `picf_photometric_strength=conservative`
- `semantic_gradient_checkpointing=False`
- `visual_activation_checkpointing=True`
- `window_activation_checkpointing=False`

This selected profile is the current runtime compromise for the 2x40GB node:

- it gives the trainable suffix transition four previous recurrent physical
  state updates
- it keeps PaliGemma/PI0.5 and all PICF adapters/control/future heads trainable
- it freezes only Sonata, V-JEPA, and AnyTouch
- it is not full-BPTT through five transitions
- observed bring-up speed was roughly `11.7-14.3 s/step`, depending on early
  step and logging conditions
- use CALVIN evaluation to compare quality against the slower full-BPTT
  `unroll_steps=3` reference

Experimental burn-in / suffix-gradient mode:

- the trainer also supports `--burnin-steps N`
- burn-in has two explicit execution modes:
  - `--burnin-mode full`: original full no-grad PICF policy forward
  - `--burnin-mode state_only`: recurrent-carry-only burn-in
- when `burnin_steps > 0`, `--unroll-steps K` means **K trainable suffix
  transitions**, not total sampled transitions
- the effective sampled window length is:

```text
effective_window_steps = burnin_steps + unroll_steps
```

- burn-in transitions run under `torch.no_grad()`
- `burnin_mode=full` still executes the full PICF observe/action-finalize
  state update, using teacher actions to advance the recurrent carry
- `burnin_mode=state_only` computes only the objects retained by
  `make_recurrent_carry(...)`:
  - runtime freshness metadata
  - tactile contact carry
  - physical posterior anchors
  - executed teacher action
  - world-only `physical_prediction_cache`
- `state_only` deliberately skips semantic task readout, conditioned control
  `C_t`, PI0.5 action flow loss, and conditioned future cache during burn-in
  because these are current-step control views and are not stored in the
  recurrent carry
- burn-in transitions do **not** receive PI0.5 flow loss, transition loss, or
  gradient
- only the suffix `unroll_steps` transitions are included in `loss_total` and
  metric denominators
- this is standard RNN-style truncated BPTT with burn-in: it exposes the suffix
  loss to a longer recurrent context without multiplying backward memory and
  time by the full context length
- it is not identical to full-BPTT `unroll_steps = burnin_steps + K`, because
  credit assignment does not flow through the burn-in transitions
- it is only valid for `picf_mode=enabled`; `picf_mode=ablated` has no PICF
  recurrent carry to burn in, and the trainer rejects that combination

Current status on 2x40GB:

- `state_only` is implemented, unit-tested, and has passed a clean 2026-04-30
  2x40GB cloud smoke with
  `--unroll-steps 1 --burnin-steps 8 --burnin-mode state_only`
- the fixed smoke run
  `picf_v22_stateonly_burnin8_fixed_smoke_20260430_r3` reached ordinary
  `step=10`, completed final `step=11`, and saved a full checkpoint at
  `.../picf_v22_stateonly_burnin8_fixed_smoke_20260430_r3/11`
- observed speed was about `0.055-0.061 steps/sec` (`16.4-18.3 s/step`) with
  eight no-grad recurrent burn-in transitions and one trainable suffix
  transition
- the previous rank-liveness issue was traced to main-rank-only final visual
  diagnostics and duplicate post-clip FSDP gradient scanning; final diagnostics
  are now interval-only, and logging reuses the already computed pre/post clip
  norm instead of launching a second FSDP grad scan
- the selected current 2x40GB long-run uses `burnin_steps=4` because it is the
  only tested profile in this family that keeps full PICF enabled while staying
  near the sub-15s target
- full-BPTT `unroll_steps=3` remains the stronger recurrent-credit reference
  when time budget allows
- before using `state_only` for a real run, compare `burnin_steps=4/8/12`
  against full-BPTT
  `unroll_steps=3` for both speed and CALVIN quality
- do not promote `burnin_steps=16` until smaller burn-in settings have passed
  speed and CALVIN checks; it covers more context but can waste forward time

Operator note:

- this profile is the recommended way to approximate JEPA-style frozen
  perception while preserving the v2.2 PICF control/action architecture
- if future backbones replace Sonata / V-JEPA / AnyTouch, the same contract
  should be kept at the interface level: frozen feature extractors feed the
  same PICF task-readout and PI0.5 action path, rather than adding
  backbone-specific glue into the recurrent physical core

### 8.6.1A Current 2x40GB PI0.5-Only Ablation Profile

The current operator-validated PI0.5-only ablation launch profile is:

- `training_strategy=fsdp_full_shard`
- `optimizer_sharding=none`
- `world_size=2`
- `accum_steps=1`
- `unroll_steps=2`
- `action_horizon=16`
- `num_train_steps=30000`
- `save_interval=2500`
- `log_interval=100`
- `lr=2e-4`
- `min_lr=2e-5`
- `warmup_steps=600`
- `grad_clip_mode=percentile`
- `grad_clip_percentile=75`
- `grad_clip_window=100`
- `picf_mode=ablated`
- `semantic_mode=paligemma`
- `semantic_trainable=True`
- `semantic_max_length=256`
- `action_normalization=quantile`
- `prompt_state_normalization=inherit`
- `window_activation_checkpointing=False`
- `semantic_gradient_checkpointing=True`
- `semantic_tokenwise_chunk_size=64`
- `semantic_projection_chunk_size=128`
- `semantic_mlp_chunk_size=64`

This profile is **not** the canonical full PICF training profile above.

It is an operational ablation profile used to answer a narrower question:

- if PICF recurrent/control/future branches are disabled, can the current
  repository train the native PI0.5 semantic action path cleanly under the
  PICF trainer/runtime shell?

The current semantics of this run shape are:

- each rank samples `1` training window per optimizer step
- with `unroll_steps=2`, each sampled window contains `3` frames and produces
  `2` action-only transitions
- with `world_size=2` and `accum_steps=1`, one optimizer step therefore covers:
  - `2` sampled windows globally
  - `4` action-only transition objectives globally

Current training-definition bugfixes that now apply globally to both
`picf_mode=enabled` and `picf_mode=ablated`:

- `action_horizon` has been restored to `16` by default
  - this re-aligns the maintained training contract with the historical CALVIN
    PI0.5 chunk contract
- `_CalvinTransitionSource` no longer samples uniformly over all valid window
  starts
  - it now samples segments uniformly first, then samples a valid start within
    the chosen segment
  - this matches the historical CALVIN dataset semantics more closely and is the
    maintained sampling contract for both full PICF and PI0.5-only ablations
- prompt-state tokenization now reuses the shared CALVIN `norm_stats.json`
  contract instead of clipping raw `robot_obs`
  - `state` is normalized before prompt discretization, matching the reference
    PI0.5 preprocessing path
  - state is **not** padded to `action_dim = 32` before prompt tokenization;
    reference PI0.5 tokenizes the normalized live state first and pads only
    after tokenization
  - raw physical-core `robot_obs` remains unnormalized

Interpretation rule:

- this is the current **operational** ablation profile because it preserves the
  present PICF window trainer shape while disabling PICF semantics
- it is **not** identical to the main-branch PI0.5 training definition
- it is also **not** identical to the historical `pi0.5_sonata` model path,
  because `picf_mode=ablated` does not instantiate the live PICF point / visual /
  tactile backbones:
  - Sonata point feature extractor is not built
  - V-JEPA visual encoder is replaced by the null visual encoder
  - AnyTouch tactile encoder is replaced by the null tactile encoder
  - the PICF core is frozen and only the PI0.5 semantic/action path remains live
- it is also **not** identical to the preserved historical 2-GPU PI0.5 runtime
  shell:
  - the maintained ablation launch uses `training_strategy=fsdp_full_shard`
  - the preserved historical PI0.5 CALVIN baselines were run through the direct
    DDP trainer in `scripts/train_pytorch.py`
- it is also **not** identical to the preserved PI0.5 CALVIN prompt budget:
  - the maintained ablation launch uses `semantic_max_length=256`
  - the historical CALVIN PI0.5 configs use `max_token_len=200`
- it is also **not** identical to the preserved PI0.5 CALVIN optimizer regime:
  - the maintained ablation launch uses `lr=2e-4`, `min_lr=2e-5`,
    `warmup_steps=600`
  - the preserved cloud `pi05_calvin_nosonata/abc_train_nosonata_full_ddp2`
    run used `warmup=10000`, `peak_lr=5e-5`, `end_lr=5e-5`
  - the generic codebase `CosineDecaySchedule` default is
    `peak_lr=2.5e-5`, `decay_lr=2.5e-6`, `warmup_steps=1000`
  - these are not the same reference, so optimizer-parity claims should cite
    which historical baseline they mean
- if the goal is exact training-definition parity with the official/main-branch
  PI0.5 stack, the cleaner baseline is `picf_mode=ablated` with
  `unroll_steps=1`, `semantic_max_length=200`, and the exact optimizer regime of
  the PI0.5 reference being compared
- if the goal is exact `pi0.5_sonata` parity, the current `picf_mode=ablated`
  profile is not sufficient by itself because it does not preserve the old
  Sonata prefix-injection path
- if the goal is "same trainer shell, same loader shape, same optimizer loop,
  but no PICF semantics", then the current `unroll_steps=2` ablation profile is
  a legitimate control experiment

Current cloud launch command for this profile:

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=/root/openpi_posterior_vla_clean/src
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name picf_v22_ablated_pi05_30000_ckpt2500_print100 \
  --overwrite \
  --device cuda \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --accum-steps 1 \
  --unroll-steps 2 \
  --action-horizon 16 \
  --num-train-steps 30000 \
  --save-interval 2500 \
  --log-interval 100 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 600 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --wandb-mode disabled \
  --no-wandb \
  --picf-mode ablated \
  --semantic-mode paligemma \
  --semantic-trainable \
  --semantic-max-length 256 \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --action-normalization quantile \
  --action-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --prompt-state-normalization inherit \
  --prompt-state-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json
```

Recommended live monitoring commands:

```bash
tail -f /mnt/checkpoints/picf_core/debug/picf_v22_ablated_pi05_30000_ckpt2500_print100_*.log
```

```bash
watch -n 2 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

Operational checkpoint rule:

- on this profile, `save_interval=2500` is the maintained default because it is
  frequent enough for early checkpoint inspection without changing the training
  objective or runtime mode

Shared-shell reminder:

- the launch block above is an ablation launch, but the shell-level controls it
  demonstrates are still owned by the shared `scripts/picf_core_train.py`
  parser/runtime
- in practice, `--save-interval`, `--log-interval`, `--accum-steps`,
  `--unroll-steps`, and `--action-horizon` can be applied to
  `picf_mode=enabled` as well
- what changes between modes is the inner policy/core path and the default
  checkpoint payload under `--optimizer-checkpoint-mode auto`, not the outer
  training-loop cadence semantics

### 8.6.2 Current Exact-Memory Runtime Measures

The current 4x40GB full-train path relies on the following mathematically exact
runtime measures:

1. tokenwise exact chunking on the hot PICF core FFN/cross-attention FFN paths
2. tokenwise exact chunking on the hot PI0/Gemma tokenwise projection/MLP paths
3. nested semantic hot-leaf FSDP wrapping
4. dead outer semantic generation heads dropped after checkpoint load
5. recursive FSDP subtree splitting on large uniform-dtype subtrees
6. explicit safe core-stack FSDP child boundaries
7. global-L2 shard-aware grad norm / percentile clipping
8. optional full-window activation checkpointing during training when the
   operator explicitly enables it
9. semantic gradient checkpointing
10. train-time recompute on the core transformer stacks
11. train-time recompute on the Sonata / AnyTouch backbone forwards
12. SDPA on the custom PI0/Gemma dual-branch attention path
13. allocator contract `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
14. node-local staging of the shared PI0/PaliGemma checkpoint
15. compact recurrent-carry instead of forwarding the full `PicfCoreState`
16. suppression of redundant outer wrapper checkpointing when the native
    PI0/PaliGemma layer checkpointing path is already active

The current safe nested semantic hot-leaf set is:

- `embed_tokens`
- per-layer `self_attn.q_proj`
- per-layer `self_attn.k_proj`
- per-layer `self_attn.v_proj`
- per-layer `self_attn.o_proj`
- per-layer `mlp`
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`

The following remain under the outer semantic root because they are not yet
nested-FSDP-safe under the current image-path alias constraints:

- `vision_tower`
- `multi_modal_projector`

### 8.6.3 Current Throughput Diagnosis

The current exact-memory contract above is mathematically correct, but the
latest audits now point to a specific throughput problem:

- the dominant slowdown is in semantic execution fragmentation
- it is not primarily in `task_readout`, conditioned control, or the physical
  finalize path

Two concrete execution facts matter:

1. nested semantic FSDP is currently very fine-grained
   - the live runtime-hot semantic set expands to `185` nested leaves:
     `1 + 18*5 + 18*5 + 4`
   - this preserves exact training math, but it also means the custom
     dual-branch semantic path can trigger many small FSDP gather / reshard
     events
2. semantic tokenwise chunking is currently one blunt knob
   - under the standard 4x40GB profile, `semantic_tokenwise_chunk_size=64`
     remains the compatibility default
   - the live trainer now resolves that compatibility knob into:
     - `semantic_projection_chunk_size`
     - `semantic_mlp_chunk_size`
   - under the current balanced full-shard default, those resolve to:
     - `semantic_projection_chunk_size=128`
     - `semantic_mlp_chunk_size=64`
   - if an operator explicitly sets the split knobs, those explicit values win
   - for rough live sequence scales near `784` tokens, this implies about:
     - `7` chunks per projection-family tokenwise op
     - `13` chunks per MLP-family tokenwise op
   - the old single-knob arithmetic therefore overestimates projection-side
     launch count once the split controls are active

Current engineering conclusion:

- the exact-memory profile is currently clean but throughput-expensive
- direct training-side prefix-KV reuse must **not** be assumed to be exact under
  the current contract, because the PI0 semantic path still appends
  `extra_prefix_tokens` into a bidirectional prefix-LM block; any future prefix
  runtime reuse therefore needs an explicit capability contract from the
  semantic backbone rather than a hard-coded PaliGemma-specific shortcut
- the next optimization pass should stay mathematically exact and target:
  - coarser semantic FSDP execution blocks
  - separated chunk controls for semantic projections vs semantic MLPs

This section is intentionally a diagnosis, not a claim that the throughput
problem has already been solved.

### 8.6.4 Operator Display / Observability Modes

Two display modes are currently useful and both preserve the same optimization
math:

1. Standard long run
   - keep the progress bar enabled
   - use `--log-interval 100`
   - this is the normal operator-facing mode for 30000-step training
2. Early observability verification
   - optionally use `--no-progress`
   - use `--log-interval 10`
   - this exists only to prove early training progress quickly, for example
     when the operator explicitly wants direct evidence that the job crossed
     `step 10`

Important clarification:

- `--no-progress` changes only what gets rendered to the terminal
- `--log-interval` changes only how often the metrics JSON line is printed
- neither changes training math

### 8.6.5 Current Cloud Launch Templates

The executable launch commands are maintained in
[`docs/CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
to avoid duplicated shell blocks drifting out of sync.

Use:

- current canonical 4x40GB full-PICF long-run launch:
  [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch)
- current 2x40GB frozen-perception full-PICF launch and state-only burn-in
  speed path:
  [`Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile)
- current cloud CALVIN video evaluation:
  [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation)

The current early-observability verification template is the same command with:

- `--log-interval 10`
- optionally `--no-progress`

Cloud detach rule:

- long runs on rented cloud machines should be launched in a detached session,
  not as a plain SSH-attached foreground/background command
- this is an operational rule only; it does not change the training graph,
  losses, optimizer, checkpoint cadence, or model math
- the observed failure mode from a plain SSH-attached `torchrun` launch is
  `torch.distributed.elastic.multiprocessing.api.SignalException: ... signal: 1`,
  i.e. an external SIGHUP delivered to the elastic launcher after the shell or
  SSH session exits
- the maintained clean launch pattern is to write the exact `torchrun` command
  into a script and start it with `setsid + nohup + stdin redirected from
  /dev/null`

Example detached launch skeleton:

```bash
RUN=/mnt/checkpoints/picf_core/debug/run_<exp_name>.sh
LOG=/mnt/checkpoints/picf_core/debug/<exp_name>.log

chmod +x "$RUN"
nohup setsid "$RUN" </dev/null > "$LOG" 2>&1 &
```

After reconnecting, verify the launcher has no controlling TTY:

```bash
ps -o pid,ppid,sid,tty,etime,stat,cmd -C torchrun
```

The expected healthy state is `PPID=1` or otherwise independent from the
interactive SSH shell, and `TTY=?`.

### 8.6.6 Operationally Important Knobs

The current operator-facing knobs that matter most are:

Display / logging:

- `--log-interval`
- `--save-interval`
- `--no-progress`
- `--diagnostic-interval`

Training envelope:

- `--training-strategy`
- `--optimizer-sharding`
- `--accum-steps`
- `--num-train-steps`
- `--grad-clip-mode`
- `--grad-clip-percentile`
- `--grad-clip-window`

Backbone trainability:

- `--use-foundation-backbones`
- `--use-tactile`
- `--perception-finetune-mode`
- `--visual-finetune-mode`

Exact-memory controls:

- `--semantic-gradient-checkpointing`
- `--window-activation-checkpointing`
- `--tokenwise-ff-chunk-size`
- `--semantic-tokenwise-chunk-size`

Interpretation rule:

- if a run must preserve the current standard 4x40GB full-train contract, do not
  change the exact-memory controls casually; those are part of the current fit
  proof, not decorative micro-optimizations

### 8.6.7 GitHub Handoff Scope

For this v2.2 rollout, the GitHub commit scope should include:

- code implementing the exact-memory training contract
- `README_v2.2.md`
- `PICF_FORMAL_CONTRACT.md`
- `docs/CALVIN_VALIDATION_README.md`
- test and verifier updates

It should **not** include:

- `/tmp/...` audit documents
- `/tmp` cloud launch helper scripts
- transient cloud logs
- transient checkpoints

The `/tmp` audits are intentionally local operator artifacts. They are used to
derive the maintained README and contract docs, not to replace them in version
control.

### 8.7 `scripts/serve_picf_policy.py`

Serving no longer treats this manual sequence as the deployed action path:

- `core.step(...)`
- `sample_action_chunk(...)`
- `refresh_predictive_state_for_action(...)`

Serving now uses:

- `policy.act(...)`

Historical serving glue removed from the deployed path:

- `core.step(..., action_future=None)` as the public action API
- direct `sample_action_chunk(...)` call from the serve script
- direct `refresh_predictive_state_for_action(...)` call from the serve script

Serving must fail fast if:

- semantic encoder missing
- PI0.5 action generation unsupported

Serving also accepts an explicit runtime override:

- `python scripts/serve_picf_policy.py --checkpoint ... --picf-mode enabled`
- `python scripts/serve_picf_policy.py --checkpoint ... --picf-mode ablated`

If `--picf-mode` is omitted, serving uses the checkpoint's saved runtime mode.

### 8.8 `scripts/verify_picf_contract.py`

The verifier was migrated away from semantic-prefix-primary inside core and now
checks the v2.2 semantics directly.

Removed checks that asserted:

- raw semantic prefix remains primary input to core control path
- raw semantic prefix remains primary input to conditioned future path
- dual control semantics are expected

Added checks that assert:

1. semantic does not enter observation anchors
2. semantic does not enter posterior update
3. innovation reads only previous physical prediction cache
4. `_build_task_readout(...)` exists
5. task readout consumes public read memory and `_StepDenseMemory`
6. exactly one conditioned control-state builder exists
7. PI0.5 action generation consumes only `conditioned_control.pi_prefix_tokens`
8. conditioned future depends on `K_phys` and `C_t`
9. raw semantic prefix is no longer a direct core control/future trunk input
10. serving/export requires PI0.5 action generator

Also add negative checks:

11. `action_condition_tokens` is no longer a canonical independent control
    semantics
12. `control_query_state` is no longer a second control semantics
13. `task_readout` is not stored as recurrent world state
14. public task-readout memory includes `visual_tokens`
15. only one conditioned-control route through `control_world(...)` remains

## 9. Checkpoint and Compatibility Record

This is a breaking structural patch. Compatibility is explicit.

### 9.1 Reuse Existing Weights Where Possible

Warm-start:

- point / visual / tactile backbones
- token builders
- observation anchor reader
- posterior update stack
- innovation stack
- predictive world stack
- prediction heads
- PI0.5 semantic/action generator
- `control_world`
- `predictive_semantic_world`

### 9.2 Reinitialize New v2.2 Modules

New modules to initialize:

- task query tokens
- task query conditioner
- task public reader
- task private reread readers
- task self stack
- task geometry projection
- `pi_prefix_query_tokens`
- `pi_prefix_reader`
- new conditioned-control projections
- new role embeddings
- new dataclass-carried interface heads

### 9.3 Compat Loader Migration Is Part of v2.2

The patch updates:

- `_COMPAT_ALLOWED_MISSING_KEYS`
- `_COMPAT_ALLOWED_UNEXPECTED_KEYS`
- any shape-mismatch whitelist assumptions

This is required for safe warm-start from current checkpoints.

Because current trainer already uses:

- `_COMPAT_ALLOWED_MISSING_KEYS`
- `_COMPAT_ALLOWED_UNEXPECTED_KEYS`
- shape mismatch filtering before relaxed load

the patch updates those explicitly. Do not assume generic `strict=False`
loading semantics.

## 10. Validation Matrix

### 10.1 Mathematical Boundary Tests

1. `test_semantic_does_not_change_physical_posterior`
2. `test_semantic_does_not_change_physical_prediction_basis_when_action_fixed`
3. `test_previous_conditioned_state_does_not_change_next_innovation`

### 10.2 Task-Readout Structure Tests

4. `test_task_readout_reads_public_read_memory_and_private_dense_memory`
5. `test_task_readout_changes_with_prompt_but_physical_core_does_not`
6. `test_only_one_conditioned_control_state_exists`
7. `test_pi_prefix_tokens_are_the_only_action_conditioning_tokens`

These tests explicitly cover:

- `visual_tokens` being part of public read memory
- `_StepDenseMemory.visual_payload`
- `_StepDenseMemory.tactile_group_tokens`
- `_StepDenseMemory.point_payload`
- no writes from task-readout outputs into recurrent posterior state

### 10.3 Exported Policy Tests

8. `test_policy_act_matches_manual_observe_sample_finalize_sequence`
9. `test_policy_fails_fast_without_pi05_action_generator`
10. `test_conditioned_future_depends_only_on_kphys_and_Ct`

Add parity tests before deleting old glue:

- trainer parity: old manual glue vs `PicfPi05Policy.forward_train_transition(...)`
- serve parity: old manual glue vs `PicfPi05Policy.act(...)`

### 10.4 Loader / Compat Tests

11. shape-changed role embedding migration test
12. task-readout missing keys allowed during compat warm-start
13. removed semantic-prefix-primary control/future keys allowed as unexpected

### 10.5 Existing Test Files To Extend

Primary extension targets:

- `src/openpi/picf/core/pipeline_test.py`
- `src/openpi/picf/paligemma/wrapper_test.py`
- `scripts/picf_core_train_test.py`
- `scripts/serve_picf_policy_test.py`

Add new test files only if these become unreadable.

## 11. Rollout Gate Record

Completed local gates:

- `py_compile` on modified core / wrapper / policy / scripts
- unit tests
- `python scripts/verify_picf_contract.py`
- local smoke through the contract verifier

Remaining runtime-stage gates:

- single-GPU minimal train on target hardware
- 2-GPU / multi-GPU DDP minimal train
- warm-start and partial-reinit short-run A/B on cloud hardware
- cloud long run

## 12. Explicit "Do Not Do" List

Do not:

1. inject semantic into observation-anchor construction
2. inject semantic into posterior update
3. let next innovation read conditioned future or `C_t`
4. keep a direct trainable 7D core action head as deployed action path
5. keep raw semantic prefix as a separate direct core control/future input
6. leave Route A / Route B dual control semantics alive
7. turn task readout into recurrent state
8. silently fall back to placeholder action in serving/export

## 13. Local Completion Criteria

Current local v2.2 satisfies the structural completion criteria when all of the
following are true:

- physical core math remains unchanged in the protected world-only portions
- `PicfPi05Policy` is the only exported action API
- there is one canonical `conditioned_control` state `C_t`
- PI0.5 is the only final action path
- raw semantic prefix no longer directly enters core control/future trunks
- task readout exists and reads both explicit public memory and private dense
  memory
- compat loader migration lands with the structural patch
- contract verifier is rewritten to the new semantics
- train / serve both use the unified policy path
- new tests for boundaries, structure, compat, and export behavior pass

## 14. Final Local Verdict

This refactor was worth doing in one patch because current code already had the
hard substrate in place:

- physical core math is already right
- private dense memory already exists
- native V-JEPA and tactile dense reread are already active
- PI0.5 flow/sampler path is already restored
- current trainer/serve already prove the pre-action / action / post-action
  timing is workable

Therefore the v2.2 patch does **not** redesign the physical core. It performs
one clean integration-layer rewrite:

1. add `task readout`
2. collapse dual control semantics into one canonical `C_t`
3. introduce `PicfPi05Policy`
4. remove raw semantic prefix as a direct core control/future mainline

That is the correct one-shot target.

## 15. MVTrack Experiment Ledger

The active MVTrack / AQR-OWM training and diagnostic ledger is maintained here:

```text
docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md
```

Current 2026-05-13 status:

```text
Code-level verifier:
  31/31 PASS after adding the guarded signature-guided local candidate repair.

Local audit:
  py_compile, strict diagnose, dataflow trace, MVTrack deep audit, and targeted
  verifier/evidence-bundle pytest pass locally. Runtime-artifact WARNs are
  expected until a metrics/eval path is supplied.

Latest local re-audit:
  repeated after commit 8cce28a with the same pass results:
  py_compile PASS, verify_picf_owm_contract 31/31 PASS, strict diagnose PASS,
  dataflow trace PASS, MVTrack deep audit PASS, targeted pytest 4 passed, and
  git diff --check PASS.

Representational diagnosis:
  A5/A7 same-object probes show high binding_signature separability
  (AUC ~= 0.96-0.98), but high duplicate candidate fraction (~0.85).

Active next test:
  use the existing binding_signature subspace inside local refinement top-k
  reranking via --local-refinement-binding-weight.

Remote runs:
  A5: picf_a5_siglocal025_burnin4_from750_to1650_20260513_08fbf31
  A7: picf_a7_siglocal050_burnin4_from750_to1650_20260513_08fbf31

Live remote status from the same re-audit:
  both jobs remain active in tmux. A5 had reached step 800; A7 had reached
  step 775. Neither showed tracebacks, NaN/Inf, or idle GPUs. Tracklet/proposal
  tokens remain zero in this CALVIN dataflow test, so the run evaluates
  signature-guided local candidate use only.

Latest remote stop diagnosis:
  A5/A7 later stopped before the planned 1650-step target because /mnt was
  full. GPUs are idle and tmux sessions are gone. There is no clear train-log
  traceback or NaN/Inf signal. A5 has usable partial metrics through step 1225
  and a 1200 checkpoint; A7 has usable partial metrics through step 1200 but
  left tmp_1200 during checkpoint write. These runs are interrupted diagnostics,
  not completed acceptance runs. Do not launch more /mnt-writing experiments
  until old checkpoints/eval artifacts are explicitly pruned.

Storage cleanup:
  May-era checkpoint payloads and eval heavy artifacts were pruned while keeping
  args/metrics/log records. /mnt changed from full to about 409G free. The
  interrupted A5/A7 checkpoint payloads are no longer available for resume, but
  their metrics/logs remain as diagnostic records. See the experiment report for
  the exact cleanup policy and caveats.

Next causal check:
  the initial attempt to run the paired test from the preserved April full-PICF
  10000-step checkpoint was invalidated before training because the current
  MVTrack/AQR architecture has many new reader/query/signature parameters and a
  widened visual prediction head. The strict FSDP compatibility guard correctly
  refused a broad migration. This is a checkpoint-architecture mismatch, not a
  local-rerank training result.

  The active replacement is a 2-3 hour fresh paired experiment from the same
  current MVTrack args and seed. A5 disables local signature reranking
  (`local_refinement_binding_weight=0.0`); A7 enables the moderate setting
  (`0.25`). `resume_checkpoint=null`, guarded predictive losses remain zero,
  and all other settings match. This tests whether the projected same-object
  subspace helps early local candidate separation without adding a new loss or
  changing the posterior/action contract.

Important guard:
  this is not a new loss, not a hard cross-anchor ownership rule, and not a
  claim that ordinal/fourth-object grounding is solved. It is a structural
  readout test for the already-decoded same-object subspace.
```

For the exact mathematical derivation, experiment matrix, commands, and
acceptance criteria, read the "2026-05-13 10-Hour Plan: Signature-Guided Local
Candidate Repair" section in the experiment report above.

## 2026-05-13 Training Anchor Overlay Diagnostic

The live trainer now supports a low-frequency static-camera anchor overlay:

```bash
--anchor-overlay-interval 100 \
--anchor-overlay-max-anchors 64
```

When enabled on the main rank, the trainer reuses the real training forward for
that optimizer step and writes:

```text
<run_dir>/anchor_overlays/step_000100.png
<run_dir>/anchor_overlays/step_000100.json
```

The PNG draws graph anchors as squares and posterior anchors as circles on the
main static RGB image. The JSON keeps projected and non-projected anchors with
world coordinates, pixel coordinates when visible, role id, confidence, support
mass, recycle gate, geometry validity, and address-update rate.

This diagnostic is intentionally not a model change. It adds no loss, runs no
extra forward pass, and does not advance V-JEPA/tactile buffers outside the
actual training step. Its purpose is to separate two different failure
hypotheses:

```text
H_support:
  aqr_same_role_support_overlap_max is high because visual support rows reuse
  the same evidence, but 3D posterior anchors may still be spatially separated.

H_physical:
  same-role support overlap is high because actual 3D anchors/posterior slots
  project into the same static-camera region.
```

Acceptance still requires metrics plus overlay inspection. A falling action
loss alone is not sufficient if overlays show same-role posterior anchors
occupying the same visible region or if JSON shows invisible/off-camera anchors
absorbing most roles.

### 2026-05-13 A7 Unroll=2 Overlay Counterfactual

The active A5 overlay run tests the fast candidate:

```text
burnin_steps=4
burnin_mode=state_only
unroll_steps=1
effective_window_steps=5
```

A7 now runs the direct counterfactual with the same staged warmup/cotrain
recipe and the same anchor-overlay diagnostic, but with:

```text
burnin_steps=4
burnin_mode=state_only
unroll_steps=2
effective_window_steps=6
```

This A7 run is not a direct speed test. A pure direct profile:

```text
burnin_steps=0
unroll_steps=2
effective_window_steps=2
```

can be faster than `burnin_steps=4, unroll_steps=1`, because it processes fewer
recurrent transitions per optimizer step. The current A7 run deliberately keeps
`burnin_steps=4` fixed and changes only the trainable suffix from 1 to 2. It is
therefore a state-distribution-controlled test of recurrent suffix credit, not
a claim that unroll2 is slower in general.

It may provide stronger recurrent credit per optimizer step because two
trainable suffix transitions receive gradient:

```math
L_{unroll2}=\frac{1}{2}(L_t+L_{t+1})
```

The acceptance question is whether this extra suffix transition improves
anchor health, not just scalar action fitting. Promote it only if
`aqr_same_role_support_overlap_max`, `posterior_recycle_rate`, address-update
rate, and the overlay PNG/JSON are all healthier than A5. If action loss falls
but overlays show physical co-location, the unroll=2 recipe is not accepted as
a binding fix.

Active A7 run ids:

```text
warmup:
  picf_a7_overlay_unroll2_warm300_20260513_b9ad838
cotrain:
  picf_a7_overlay_unroll2_cotrain_from300_to900_20260513_b9ad838
tmux:
  picf_a7_overlay_unroll2_warmcotrain
```

Tail:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warmcotrain_20260513_b9ad838.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_warm300_20260513_b9ad838/metrics.jsonl
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_overlay_unroll2_cotrain_from300_to900_20260513_b9ad838/metrics.jsonl
```

### 2026-05-13 Task-Pressure Warmup Clarification

The pure action-off overlay warmup is now archived as a negative control. It
showed that recycle normalization can recover while same-role supports still
collapse back into the 0.95-0.99 overlap band. The current hypothesis is that
anchor warmup should include low-weight task pressure, not full cotrain and not
zero-action pretraining:

```text
trainable scope:
  anchor_only

frozen:
  Sonata / V-JEPA / AnyTouch

disabled:
  slot-JEPA, support-pred, binding-consistency, AQR denoising, local refinement

enabled:
  lambda_action_pos/rot/gripper = 0.25
  no_picf_action_prefix_stopgrad
```

This is a symmetry-breaking test. Without any task gradient, same-role anchors
can read the same high-confidence evidence and still satisfy weak evidence
objectives. A low action gradient tests whether action relevance can separate
otherwise similar supports without letting the policy head dominate the belief
router.

Active task-pressure warmup runs:

```text
A5:
  picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838
  burnin_steps=4, unroll_steps=1

A7:
  picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838
  burnin_steps=4, unroll_steps=2
```

Tail:

```bash
# A5
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a5_taskwarm_a025_unroll1_300_20260513_b9ad838/anchor_overlays

# A7
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838.train_tmux.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a7_taskwarm_a025_unroll2_300_20260513_b9ad838/anchor_overlays
```

See
`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`, section
`2026-05-13 23:45 Task-Pressure Warmup Restart`, for the full objective,
acceptance criteria, and failure interpretation.

Remote environment detail is also recorded there: the active A5/A7 worktrees
use `uv sync --project . --frozen --no-install-package av` because `av` is a
transitive `lerobot` video I/O dependency that is not used by the PICF CALVIN
training path and requires unavailable FFmpeg development libraries.
They inherit `/root/openpi/.venv/lib/python3.11/site-packages` after `src` in
`PYTHONPATH` to reuse the server's existing Sonata `spconv`/`torch_scatter`
runtime without changing the active source checkout.

### 2026-05-14 Assignment-Level Ownership Prior

The A5/A7 task-pressure warmups are now archived as a negative result for
schedule-only fixes:

```text
A5 step 125:
  same_role_overlap=0.9960
  recycle_rate=0.5223

A7 step 100:
  same_role_overlap=0.9374
  recycle_rate=0.6398
```

This means the remaining bottleneck is not simply action warmup length or
unroll length. The mathematical issue is that same-role AQR support rows can be
identical before Sinkhorn/diversity losses see them:

```math
\ell_{j,:}=\ell_{k,:}
\Rightarrow
\operatorname{Sinkhorn}(W)_{j,:}
=
\operatorname{Sinkhorn}(W)_{k,:}
```

The maintained fix is an assignment-level ownership prior, not another
auxiliary loss:

```text
aqr_ownership_prior_enabled = true
aqr_ownership_prior_weight = 0.35
aqr_ownership_temporal_prior_weight = 0.20
aqr_ownership_prior_uniform_mix = 0.05
```

It adds a low-amplitude, role-local coverage prior to visual and temporal AQR
support logits before support reads. This breaks exact same-role symmetry while
leaving current evidence dominant. It is the coherent partner to the existing
projected binding-signature subspace inspired by the IsSameObject/object-binding
probe literature: first seed distinct object ownership, then stabilize it with
binding signatures and posterior correction.

See
`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`, section
`Final Read: Schedule-Only Task Pressure Fails Ownership`, for the failed
warmup evidence, derivation, implementation files, and acceptance criteria.

### 2026-05-14 Capacity-Aware Active/Dustbin Slot Repair

The ownership-prior runs fixed initial same-role symmetry but failed after
step100/125 because the fixed physical slot set still forced every slot to
compete for a scene with fewer useful object supports. The maintained repair is
therefore capacity-aware assignment, not a stronger action warmup, stronger
cache, slot-JEPA pressure, or another downstream overlap loss.

New production-default controls:

```text
aqr_active_slot_filter_enabled = true
aqr_active_slot_min_per_role = 1
aqr_active_slot_max_per_role = 4
aqr_active_slot_min_confidence = 0.05
aqr_active_slot_overlap_threshold = 0.75
```

Semantics:

```text
active slot:
  a high-confidence, role-local support owner allowed to participate in
  observation/task assignment and active same-role diversity.

inactive/dustbin slot:
  a redundant same-role candidate retained as a recurrent/query carrier but
  excluded from assignment pressure when an active same-role set exists.
```

This is the slot-capacity analogue of the `no-object`/dustbin mechanism used by
set-prediction detectors, adapted to PICF's belief filter: extra slots should
not be forced to explain the same object. It is also consistent with recent
object-centric slot work on variable/effective slot count: fixed slots are a
capacity budget, not a promise that every slot corresponds to a distinct
physical object.

The raw `aqr_same_role_support_overlap_max` remains logged for diagnosis, but
the acceptance metric for this repair is now:

```text
aqr_active_same_role_support_overlap_max
aqr_active_same_role_support_overlap_mean
aqr_active_anchor_count
aqr_inactive_anchor_fraction
aqr_active_anchor_count_role_{0,1,2,3}
```

Do not treat high raw same-role overlap alone as failure after this change:
inactive/dustbin duplicates may intentionally overlap. Failure is active
same-role overlap returning to the old `0.95-0.99` band, active count collapsing
to one slot, or action/recycle health degrading while active overlap improves.

Local verification for this change:

```text
python scripts/verify_picf_owm_contract.py:
  PASS, including active/dustbin capacity invariant.

python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS.

python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS.

python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS.

uv run --no-sync pytest -q src/openpi/picf/core/pipeline_test.py \
  -k "ownership_prior or active_slot_filter or slot_assignment_ignores":
  4 passed.

uv run --no-sync pytest -q src/openpi/picf/core/training_test.py \
  -k support_diversity:
  4 passed.
```

The next 10-hour A5/A7 matrix is recorded in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).
It is launched by
[`scripts/run_picf_active_slot_matrix.sh`](/home/siyuanyue/Documents/openpi/scripts/run_picf_active_slot_matrix.sh).
Its purpose is to test whether the active-object subset stays separated under
task pressure and cotrain, not to claim CALVIN behavior acceptance.

2026-05-14 03:40 runtime gate: the A5 and A7 active-slot matrices are both
running normally in tmux with live GPU load and advancing metrics. The first A5
anchor-only branch shows low active overlap but over-prunes to about four
active anchors with high recycle; it is an isolation result, not production
success. The first A7 cotrain branch keeps a plausible active-anchor count and
competitive action loss, but recycle is still high at step 200; it is a warning
row, not acceptance. The queued A5/A7 branches are therefore required to
separate unroll/burn-in shape, action-pressure scale, prefix-stopgrad, and
active-capacity threshold. The acceptance gate is active overlap plus active
capacity plus non-saturated recycle, not raw overlap alone and not action loss
alone.

2026-05-14 04:55 one-hour gate: runtime remains normal. A5 completed the
u1/b4 anchor-only branch and entered u2/b1; u2/b1 improves active capacity but
pushes recycle near zero, so it is still diagnostic only. A7 full-action
prefix-stopgrad reached step350 with strong action improvement and active
overlap below the provisional gate, but recycle is saturated near one. The
remaining live question is now posterior identity/recycle dynamics under
cotrain, not whether the active/dustbin support filter can separate an active
subset. Continue the queued lower-action/no-prefix and max6-capacity branches.

2026-05-14 06:00 post-sleep gate: local verifier, strict diagnose, dataflow
trace, MVTrack deep audit, and Python compilation pass. A7 full-action
prefix-stopgrad reached step550 with `loss_action_default_equiv=0.0618`,
dual-view temporal mass active, and plausible active capacity
(`aqr_active_anchor_count=13.0`), but it is still not accepted because
`posterior_recycle_rate=0.9976`, recycle gates have near-zero variance, and
address update is effectively starved. A5 u1/b4 and u2/b1 overlays confirm
physical same-role duplicate anchors in the raw graph/posterior: by step600
several same-role graph pairs are separated by less than one pixel, and one
posterior role pair is exactly co-located. The active-slot filter is therefore
a useful discriminator/demotion mechanism, not a full identity solution. The
remaining blocker is the joint object-count plus posterior identity-continuity
dynamics; continue the queued A7 lower-action/no-prefix and max6 branches
before making the next architecture change. Detailed numbers and the overlay
audit are in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-14 06:16 stability gate: A7 automatically completed the full-action
prefix branch and entered the queued lower-action/no-prefix branch, so the
10-hour matrix driver is working as intended. The completed A7 branch is a
clear negative for posterior identity health (`posterior_recycle_rate=0.9979`,
`posterior_address_update_rate_mean=7.9e-5`) despite good action loss
(`loss_action_default_equiv=0.0676`). A5 max6 reached step100 with the best
early structural row so far: raw overlap `0.2327`, active overlap `0.2168`,
active count `20.0`, recycle `0.2623`, and nonzero address update. This does
not prove production readiness because it is anchor-only, but it strongly
supports testing larger active capacity together with controlled action
pressure. Continue A5 max6 to 200/300/600 and A7 lower-action/no-prefix to
50/100/200 before changing code.

2026-05-14 06:22 unattended-watch gate: both remote matrices are still live and
GPU-active. A5 is running `picf_a5_activecap_anchor_u2b1_max6_a025_600_ac273a2`
and has reached step150 with active overlap `0.4762`, active count `18.69`,
recycle `0.1822`, and nonzero address update `0.0288`; this remains the best
structural candidate but must be checked again at step200/300 because raw
overlap rose to `0.6595`. A7 is running
`picf_a7_activecap_cotrain_flow_u2b1_a025_450_ac273a2` with `scope=all`,
`unroll=2`, `burnin=1`, action scale `0.25`, and PaliGemma trainable; it has not
yet emitted its first metrics row. Ten-hour remote watch logs are active at
`/mnt/picf_run_logs/picf_a5_activecap_watch10h.log` and
`/mnt/picf_run_logs/picf_a7_activecap_watch10h.log`. Do not patch architecture
before A5 step200/300 and A7 flow step50/100, because those branches are the
causal tests for active capacity versus action-pressure/recycle dynamics.

2026-05-14 06:29 tens-step gate: the remote port map was rechecked
(`qgE72e:28060` for A7, `ZWWQO6:29776` for A5). A7 lower-action/no-prefix has
run at least 35 progress-bar steps and is GPU-active; it has not emitted the
first step50 metrics row yet. A5 max6 reached step200: active overlap is still
barely below the provisional gate (`0.5733`), but raw same-role overlap has
returned to `0.9959`, active count dropped to `12.16`, and the step200 overlay
shows seven role-1 posterior slots exactly co-located at pixel `[105.2, 114.1]`.
This confirms a real posterior identity/binding collapse after candidate
generation, not just a metric artifact. Continue to A5 step300 and A7 step50/100
before changing code; active-slot demotion is useful but is not yet a full cure.

2026-05-14 06:40 causal gate: A7 lower-action/no-prefix reached step50 with a
healthy first structural row: raw overlap `0.3631`, active overlap `0.1821`,
active count `14.0`, recycle `0.6349`, and address update `0.0165`. This is not
acceptance yet, but it is materially better than the rejected full-action prefix
row. A5 max6 reached step300 and is now rejected as a standalone anchor-only
cure: raw overlap `0.9996`, active count `6.46`, recycle `0.0037`, and overlay
JSON shows seven role-1 posterior slots exactly co-located at pixel
`[107.2, 120.3]`. The current causal interpretation is: active capacity helps
early, but capacity alone does not fix posterior identity; the A7 flow branch is
now the decisive test for controlled action pressure without prefix-stopgrad.

2026-05-14 07:45 one-hour audit: A5 completed and is now idle; its max6 branch
ended as a negative anchor-only result (`raw_overlap=0.9997`,
`active_count=7.5`, `recycle=0.0004`). A7 flow is still running and reached
step200; it improves recycle versus the rejected full-action prefix branch
(`0.6148` instead of near `0.998`) but still fails the structural gate:
`raw_overlap=0.9912`, `active_overlap=0.5573`, `active_count=10.1`. The A7
step200 overlay shows the same physical failure pattern as A5: graph candidates
remain spatially spread, but seven role-1 posterior slots are exactly co-located
at pixel `[90.2, 108.4]`. Current conclusion: this is no longer an active-cap
or action-scale-only problem; the remaining root cause is posterior
binding/correction coalescing same-role candidates into one physical state.

2026-05-14 08:51 requested follow-up: the remote matrix is still running
normally. A7 flow reached structured step400 and progress-bar step407/450 with
both GPUs active; A5 completed and is idle. A7 flow still fails structurally:
`loss_action_default_equiv=0.0738`, but
`aqr_same_role_support_overlap_max=0.9866`,
`aqr_active_same_role_support_overlap_max=0.5990`,
`posterior_recycle_rate=0.8593`,
`posterior_identity_switch_rate=0.8256`, and
`posterior_stable_slot_fraction=0.11`. The step400 overlay confirms the failure
is physical, not a metric artifact: graph role anchors remain somewhat spread,
but seven visible role-1 posterior slots are exactly co-located at pixel
`[66.84, 116.62]`. This further rejects active-capacity, action-scale, and
unroll-only explanations. The next architecture change must target posterior
assignment/correction with same-role occupancy control; adding another outer
support-diversity loss would be a patch, not a cure. Full numbers and the
mathematical interpretation are in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-14 posterior object-file birth update: the step400 A7 overlay exposed a
lower-level symmetry issue. AQR graph queries had coverage/type identities, but
persistent posterior object files still used the legacy symmetric birth
contract: `posterior_slot_identity_std=0`,
`task_slot_identity_std=0`, and
`posterior_bootstrap_from_observation=False`. That makes same-role recurrent
slots exactly permutation-symmetric before identity evidence exists, so a shared
posterior residual/recycle path can update multiple scene slots into the same
state. The production default is now changed to
`posterior_slot_identity_std=0.02`,
`task_slot_identity_std=0.02`, and
`posterior_bootstrap_from_observation=True`, with CLI/logging support in
[`scripts/picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)
and verifier coverage in
[`scripts/verify_picf_owm_contract.py`](/home/siyuanyue/Documents/openpi/scripts/verify_picf_owm_contract.py).
This is a minimal object-file birth prior, not a new loss or an action-side
patch. The next A5/A7 causal runs should test whether nonzero posterior identity
seeds plus first-step geometry bootstrap prevent same-pixel posterior
co-location under cotrain.

The corresponding deployment entrypoint is
[`scripts/run_picf_posterior_birth_matrix.sh`](/home/siyuanyue/Documents/openpi/scripts/run_picf_posterior_birth_matrix.sh).
It runs A5 anchor/cotrain isolation and A7 cotrain stress rows with the same
`unroll=2`, `burnin=1`, local-refinement-off, PaliGemma-cotrain profile used by
the latest matrix. This isolates posterior object-file birth from unrelated
module changes.

2026-05-14 slot-local recycle/reset update: the one-hour posterior-birth matrix
showed that identity seeding and first-step geometry bootstrap are necessary
but not sufficient. A5 step400 and A7 step150 still produced exact same-pixel
posterior co-location for seven role-1 object files. The graph-stage anchors
were not the sole failure source; the posterior recycle/reset path was still
able to erase separation because recycled slots shared one global dustbin
residual. The maintained default now uses each posterior slot's own raw
measurement mixture for recycle/reset (`posterior_slotwise_recycle_residual =
True`) and only falls back to the dustbin residual when that slot has no
support. This is the belief-filter-consistent repair: recycle is an object-file
trust decision, not a scene-global reset. The causal A5/A7 matrix entrypoint is
still
[`scripts/run_picf_posterior_birth_matrix.sh`](/home/siyuanyue/Documents/openpi/scripts/run_picf_posterior_birth_matrix.sh),
but run names now include the current git short SHA instead of the old fixed
suffix. Detailed numbers, math, and acceptance gates are recorded in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-14 posterior occupancy prior update: the slot-local recycle run fixed
the recycle/reset symmetry but still failed the overlay gate. A5 step100 had
healthy graph support separation, but seven role-1 posterior files projected to
the same pixel-level centroid. The failure is now localized to posterior
measurement association: fixed-row Sinkhorn gives every same-role posterior row
measurement mass, and without a coverage prior weak logits produce identical
broad mixtures.

The maintained runtime now adds a label-free same-role posterior occupancy prior
inside `_posterior_update`. For each role, current observation-anchor hypotheses
are farthest-point sampled and used as per-object-file measurement coverage
centers. The bias is row-centered and clipped, so it breaks symmetry without
becoming a hard pseudo-label or an action-side loss:

```text
posterior_occupancy_prior_enabled = True
posterior_occupancy_prior_weight = 1.0
posterior_occupancy_prior_sigma_m = 0.04
posterior_occupancy_prior_clip = 4.0
```

This is the next accepted test candidate, not a completed behavior claim. The
next A5/A7 matrix must show that posterior role-1 overlay points are no longer
co-located at step100/200 before action loss is treated as meaningful. Full
math, metrics, and acceptance gates are recorded in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md).

2026-05-14 follow-up: the first `1dceaef` A5 step100 run showed the posterior
occupancy prior improved the failure but did not fully solve it: posterior
role-1 pairwise pixel mean rose from the old near-zero value to roughly `0.68px`,
while graph role-1 candidates were themselves only moderately separated. The
root issue is therefore earlier than posterior assignment: observation anchors
must retain their seed coverage geometry and cannot all reread the same broad
point-cloud mixture. The maintained candidate now also enables:

```text
observation_anchor_seed_point_mix = 0.35
```

This mixes each valid seed-point one-hot prior back into its observation-anchor
point weights after graph/readout fusion. It is a measurement construction prior,
not an auxiliary loss or action-side patch.

2026-05-14 `07bdf66` runtime gate: the seed-coverage candidate is running on
both cloud machines. A5 anchor-only reached step200 and gives a split result.
The old posterior exact co-location failure is fixed at step100/200: role-1
posterior pairwise pixel mean is `25.59px` at step100 and `18.38px` at step200,
instead of the old near-zero values. However, A5 is not accepted as a stable
anchor-only solution because raw same-role support overlap rebounds to `0.9523`,
active overlap reaches `0.6029`, and effective anchor count falls to `13.32`.
This means seed coverage plus posterior occupancy repairs physical object-file
co-location, but anchor-only pressure can still reuse same-role support and
demote redundant candidates. The production-relevant discriminator is now A7
cotrain step100/200: if A7 keeps posterior separation without raw overlap
collapse under task/action/semantic pressure, the candidate can proceed; if A7
also rebounds, the remaining issue is same-role measurement competition rather
than action loss or unroll length.

2026-05-14 A7 step100 update: the production-relevant cotrain branch is
healthier than A5 anchor-only. A7 reaches `loss_action_default_equiv=0.0829`,
`aqr_same_role_support_overlap_max=0.6559`, active overlap `0.4281`,
effective anchor count `19.42`, and posterior role-1 pairwise pixel mean
`22.21px`. This is not final acceptance because raw overlap is above the
preferred `0.60` early gate, but it is materially different from the old
`0.95-0.99` collapse and from the A5 anchor-only rebound. Continue A7 to
step200 before any new code change; A5 is now only a negative-control record
showing that anchor-only pressure still demotes/reuses same-role candidates.

2026-05-14 root-cause routing update: A5 step250/300 showed the remaining
failure is not posterior exact co-location. It is same-role support reuse:
multiple physical object files can read the same visual/point/temporal evidence
and only become distinct after posterior correction, which is too late for a
stable object-file measurement model. The maintained repair is
`aqr_same_role_support_competition_enabled=True`. For each same-role physical
object-file group, the AQR support rows are transformed by a role-local
competition step before graph priors are consumed:

```text
E_jn^(0) = P_jn
E_jn^(k+1) = Normalize_n(E_jn^k / sum_{l in same_role(j)} E_ln^k)
P'_j = Normalize_n((1-lambda) P_j + lambda E_j^(K))
```

This is a measurement-routing invariant, not a new loss and not a heuristic
action patch. It cannot invent object evidence: identical rows remain
identical, so it depends on the existing ownership prior, seed-point coverage,
geometry, and pairwise binding-signature subspace to provide weak differences.
It only prevents same-role object files from all keeping the same support when
weak row-specific evidence already exists. This follows the object-binding
paper's IsSameObject lesson: binding is a pairwise low-dimensional relation
that guides attention, so the correct integration point is the support-routing
subspace rather than an action-side penalty. It is also consistent with recent
object-centric manipulation work that treats persistent slots as a belief-state
interface, not as a late auxiliary classifier. See:
`https://arxiv.org/abs/2510.24709`, `https://arxiv.org/abs/2511.06754`, and
`https://arxiv.org/abs/2601.20381`.

The first A5 support-competition run on `24c6cf7` reached step100 with
`aqr_same_role_support_overlap_max=0.6874`, active overlap `0.5120`,
effective anchor count `18.18`, and posterior role-1 mean pixel distance
`21.70px`. This is not final acceptance because raw overlap is still above the
preferred early gate, but it is no longer the old A5 `0.95+` rebound at the same
stage. Continue this run to step200 before deciding whether to change upstream
token/probe extraction.

2026-05-14 follow-up gate: the same A5 support-competition anchor-only run
reached step300 and is rejected as a standalone fix. The role-local competition
is mathematically valid and helped early, but it did not make anchor-only
pressure self-sufficient:

```text
step50:
  raw overlap    = 0.2179
  active overlap = 0.2035
  effective K    = 19.49

step100:
  raw overlap    = 0.6874
  active overlap = 0.5120
  effective K    = 18.18

step200:
  raw overlap    = 0.9963
  active overlap = 0.5776
  effective K    = 9.05

step300:
  raw overlap    = 0.9999
  active overlap = 0.1867
  effective K    = 4.63
  recycle rate   = 0.0537
```

This is a useful negative result. The remaining failure is no longer recycle
saturation or exact posterior co-location: recycle is low by step300. The
failure is that anchor-only optimization can still reuse the same same-role
support and demote redundant object files until only a small active set remains.
Active filtering prevents duplicate slots from contaminating assignment, but it
does not prove all object files remain useful. Do not add another loss penalty
on top of this. The next diagnostic must test production-like cotrain pressure:
if cotrain keeps `effective_anchor_count` healthy while action decreases, then
anchor-only is the wrong acceptance environment; if cotrain also rebounds, the
root cause moves upstream to weak object-specific evidence in token/probe
extraction.

The follow-up iteration is now an A5 cotrain-only matrix on commit `c119321`:

```text
tmux:
  picf_a5_cotrain_iter_c119321

row 1:
  picf_a5_birth_cotrain_u2b1_a025_450_c119321_support_comp_cotrain
  scope=all, unroll=2, burnin=1, action_weight=0.25

row 2:
  picf_a5_birth_cotrain_u2b1_a05_450_c119321_support_comp_cotrain
  scope=all, unroll=2, burnin=1, action_weight=0.50
```

This is the correct next diagnostic because it tests whether task/action/
semantic gradients give object files enough utility to avoid the anchor-only
support-reuse collapse. The startup gate passed: both A100s reached about
39.6GB and 100% utilization, and the progress bar entered real training.

A7 should be repurposed from its older `07bdf66` seed-coverage run to the
current support-competition discriminator. The older A7 row reached step300
with action improving but raw same-role support overlap back at `0.9624`,
active overlap `0.5882`, and effective anchor count `13.66`. That is enough to
reject the old candidate; finishing to 600 would mostly measure an outdated
configuration. The maintained fast A7 profile is:

```bash
bash scripts/run_picf_posterior_birth_matrix.sh a7_fast
```

It runs two 300-step all-scope cotrain rows at action weights `0.25` and `0.50`
with `unroll=2`, `burnin=1`, active max-per-role `6`, and the current same-role
support competition. The purpose is not to chase a lower action loss in
isolation; it is to test whether current measurement routing keeps
`effective_anchor_count` healthy while action decreases under production-like
cotrain pressure.

2026-05-14 A5 stale-run gate: the live A5 cotrain run is still on commit
`c119321`, so its healthy step50/100 rows cannot validate the maintained
posterior object-file birth / occupancy / seed-coverage repair. The run is
useful only as a support-competition cotrain early signal:
`loss_action_default_equiv` fell to `0.0869`, raw same-role overlap stayed at
`0.2062`, and effective anchor count stayed at `19.46` at step100. The current
deployment plan is to stop this stale A5 row, sync A5 to the latest branch, and
run
[`scripts/run_picf_posterior_birth_matrix.sh`](/home/siyuanyue/Documents/openpi/scripts/run_picf_posterior_birth_matrix.sh)
profile `a5`.

The maintained mathematical contract being tested is not a hard-coded object
detector. It is a label-free belief-filter birth and measurement prior:
nonzero posterior object-file identity seed, first-step observation bootstrap,
slot-local recycle residual, same-role posterior occupancy prior, observation
seed-point coverage, and same-role support competition. This matches the
object-agent tokenization / DINO-style query initialization lesson: seed
distinct object hypotheses from encoder evidence, then let learned attention,
support signatures, and posterior correction own the state. Raw PaliGemma
attention heatmaps are explicitly not used as object truth; if the maintained
candidate fails, the next escalation is an annealable learned object-core
continuation prior with robust-normalized soft proposal evidence, not a
category-specific hand rule. The detailed plan and gates are in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md),
section `2026-05-14 A5 Stale-Run Gate And Object-File Birth Plan`.

2026-05-14 A5 latest runtime gate: A5 has been synced to `4da418e` and is
running `picf_a5_posterior_birth_4da418e` with the latest maintained posterior
birth/occupancy/seed-coverage profile. Startup confirms
`posterior_occupancy_prior_enabled=True`,
`observation_anchor_seed_point_mix=0.35`,
`posterior_slotwise_recycle_residual=True`, local refinement off, and all
guarded OWM auxiliary losses at zero. The first step50 row is structurally
healthy: raw same-role overlap `0.2081`, active overlap `0.1938`, effective
anchor count `19.58`, active anchor count `19.97`, recycle `0.4809`, and
address update `0.0211`. This proves the latest run is live and early-healthy,
but it is not acceptance; previous rejected branches failed at step150-300. The
next gates are step100/200/300 plus anchor overlays. The live log is:

```bash
tail -f /mnt/picf_run_logs/picf_a5_birth_anchor_u2b1_a025_450_4da418e_a5_latest.log
```

2026-05-14 live status update: A5 latest has reached step150. The repair still
delays collapse, but the trend is not yet accepted: step100 had raw overlap
`0.4865`, active overlap `0.3340`, and effective anchors `19.45`; step150 rose
to raw overlap `0.8298`, active overlap `0.6111`, effective anchors `18.07`,
recycle `0.7589`, and address update `0.0096`. This is materially better than
the old 0.99/K≈4-8 collapse, but it is a warning row, not success. A7 fast
cotrain on the older `92f064c` support-competition commit is healthier at
step100 (`raw=0.3208`, `active=0.2612`, `K=19.46`,
`loss_action_default_equiv=0.1021`), which supports testing production-like
cotrain after A5 anchor-only. The decisive gates remain A5 step200/300 and the
following A5 cotrain row.

2026-05-14 step200 update: A5 latest reached raw overlap `0.9064`, active
overlap `0.6260`, effective anchors `17.15`, recycle `0.8201`, and address
update `0.0070`. This rejects the anchor-only row as a final solution and
triggers the object-core ownership repair described earlier in this README and
in
[`docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md):
point ownership prior before AQR point attention plus multi-modal object-core
overlap for active-slot filtering. The next maintained A5 run must be launched
from the new commit; A7 remains only an older support-competition control.

Operational logging requirement: object-core runs are only valid after
`metrics.jsonl` records both visual-only overlap and object-core overlap. The
required keys are `aqr_same_role_object_core_overlap_max`,
`aqr_active_same_role_object_core_overlap_max`,
`aqr_ownership_point_prior_weight`, and
`aqr_ownership_point_prior_sigma_m`. A run that only logs visual overlap cannot
decide whether the repair is separating physical object cores or merely
renaming the old visual-support collapse.

2026-05-14 valid object-core step50: commit `674de2f` logs the required
object-core metrics. A5 anchor-only step50 has visual raw overlap `0.2190`,
active visual overlap `0.2127`, raw object-core overlap `0.1977`, active
object-core overlap `0.1977`, effective anchors `19.65`, recycle `0.3366`, and
address update `0.0265`. This is an early healthy row, not final acceptance; the
decisive gates are still step100/200/300 plus overlays.

2026-05-14 one-hour object-core result: both A5 and A7 reproduced the same
late collapse pattern. A5 anchor-only reached step450; raw same-role visual
overlap returned to `0.9972`, active visual overlap to `0.9931`, raw object-core
overlap to `0.9524`, and `loss_mapg_support_diversity` rose from `0.3274` to
`0.9333`. A7 cotrain reached step200; raw visual overlap was `0.9907` and
active visual overlap `0.9810`. Therefore the object-core prior is useful but
not sufficient. The decisive new diagnosis is not a code-path break: it is loss
budget starvation. In the rejected A5 row, `loss_alignment_raw` rose to
`1.44-1.54`, but `loss_alignment` was capped at exactly `0.0125` because the
alignment group budget was tied to `aux_budget_alignment_ratio=0.05` and the
generic action-loss floor `0.25`. The structure loss was detecting collapse but
was not allowed to exert enough gradient to prevent it.

The maintained next test therefore separates the latent binding invariant from
the generic auxiliary action budget:

```text
aux_budget_alignment_floor = 2.0
aux_budget_alignment_ratio = 1.0
lambda_mapg_support_diversity = 0.25
lambda_mapg_geometry_diversity = 0.05
```

This is not a new module. It is a target-function repair: object-file birth,
ownership, and support diversity are the measurement model that makes posterior
belief meaningful, so they must not be starved during bootstrap. The next paired
diagnostic profiles are:

```bash
bash scripts/run_picf_posterior_birth_matrix.sh a5_structure_budget
bash scripts/run_picf_posterior_birth_matrix.sh a7_structure_budget
```

Acceptance now requires:

```text
step150/200 active visual overlap < 0.75
step150/200 active object-core overlap < 0.50
alignment_budget_scale not pinned near zero
loss_mapg_support_diversity not monotonically rising
anchor overlay role-1/2/3 pixel std not collapsing to single-digit pixels
```

2026-05-14 one-hour structure-budget result: the budget repair is a real
target-function fix, but it is not a final solution by itself. A5
`a5_structure_budget` completed 450 steps with `alignment_budget_scale ~= 1.0`,
so the previous starvation bug is gone. Early rows were healthy
(`step50 active visual overlap = 0.1914`, active object-core overlap =
`0.1691`), but the later rows still drifted upward:

```text
A5 structure-budget, step450:
  aqr_same_role_support_overlap_max              = 0.8712
  aqr_active_same_role_support_overlap_max       = 0.7683
  aqr_same_role_object_core_overlap_max          = 0.7155
  aqr_active_same_role_object_core_overlap_max   = 0.5999
  loss_mapg_support_diversity                    = 0.4075
  loss_alignment_raw                             = 0.9953
  alignment_budget_scale                         = 0.9977
```

This is a large improvement over the rejected object-core run
(`active visual overlap = 0.9931` at step450), but it misses the strict gate
(`active visual < 0.75`, active object-core < 0.50`). Overlay statistics also
show partial geometric concentration rather than a clean multi-object spread.
The diagnosis is therefore refined: the old failure was partly budget
starvation, but not only budget starvation. Same-role supports still need a
more object-conditional assignment objective, not just a larger diversity
coefficient.

A7 `a7_structure_budget` stayed healthier than A5 but did not pass the final
structure gate by step300:

```text
A7 structure-budget cotrain, step300:
  loss_action_default_equiv                      = 0.0709
  aqr_same_role_support_overlap_max              = 0.6876
  aqr_active_same_role_support_overlap_max       = 0.5681
  aqr_same_role_object_core_overlap_max          = 0.6006
  aqr_active_same_role_object_core_overlap_max   = 0.5798
  loss_mapg_support_diversity                    = 0.3125
  alignment_budget_scale                         = 0.9928
```

This is much better than the rejected collapse band and keeps action moving
down, but it misses the final object-core gate. The useful distinction is now
clear: A7's task-conditioned cotrain with prefix stop-gradient is not destroying
anchors in this profile and supplies a useful selection signal, but scalar
budget/support-diversity changes alone are not a complete cure. The next repair
must change the object-conditional assignment energy itself rather than only
increasing the weight of the existing diversity term.

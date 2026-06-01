# PICF Entry

This file is only a directory entry.

Current PICF documents:

1. [`README_v2.2.md`](./README_v2.2.md)
   The current local v2.2 architecture record and implementation audit for the
   live codebase, including the explicit effector/object anchor role split,
   global scene point-pool contract, and PICF-AQR-OWM/MVTrack deployment links.
   Current 2026-05-19 default: tracklet/contact-motion sidecars remain enabled;
   Blind automatic SAM is rejected and archived. Generic `proposal_*` memory
   remains an explicit sidecar contract only; use inspected contact/task/
   tracklet-guided sidecars, not blind SAM, when enabling
   `--proposal-memory-enabled`. The current post-SAM-archive training recipe is
   documented in `README_v2.2.md` under `Live long-run recipe`.
   The latest local full gate is
   [`docs/PICF_AQR_OWM_LOCAL_FULL_AUDIT_20260522_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_LOCAL_FULL_AUDIT_20260522_TEMP.md);
   it is the current code-level audit record before behavior-level 30k/CALVIN
   acceptance.
   The current mathematical/documentation navigation layer is
   [`docs/PICF_AQR_OWM_MATH_CONSISTENCY_AND_DOC_INDEX_20260522_TEMP.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_MATH_CONSISTENCY_AND_DOC_INDEX_20260522_TEMP.md);
   read it before reopening older raw-overlap, SAM, or slot-paper threads.
2. [`PICF_AQR_OWM_CURRENT_STATE_20260518.md`](/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_CURRENT_STATE_20260518.md)
   Compact current-state cleanup report: production-maintained modules,
   guarded/data-dependent modules, legacy/off-by-default modules, code
   cleanliness verdict, and verification commands.
   It now also points to the AR active-anchor proposal audit, which is a
   vNext capacity-allocation design candidate rather than a current training
   default.
3. [`README_PI05_PARITY_AUDIT.md`](./README_PI05_PARITY_AUDIT.md)
   Code-level comparison of reference PI0.5 / PI0.5+Sonata dataflow against
   current PICF enabled and PI0.5-only ablated modes.
4. [`README_FROZEN_PERCEPTION_AUGMENTATION.md`](./README_FROZEN_PERCEPTION_AUGMENTATION.md)
   Design record for the 2x40GB frozen-perception profile and geometry-safe
   augmentation policy.
5. [`README_VL_GUIDED_ANCHOR_ROUTER.md`](./README_VL_GUIDED_ANCHOR_ROUTER.md)
   Current staged implementation record for the default-off point-centric
   PaliGemma-guided 2D-to-3D anchor prior router.
6. [`README_MAPG_PICF.md`](./README_MAPG_PICF.md)
   Current live implementation record for MAPG-PICF, the modality-optional
   anchor prior graph over PaliGemma, V-JEPA, Sonata, AnyTouch, and posterior
   supports. Use this for the MAPG math, code dataflow, graph losses, CLI,
   tests, diagnostics, and the full MAPG launch template.
7. [`README_v2.1.md`](./README_v2.1.md)
   Historical v2.1 deployment record retained for reference.
8. [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
   The executable live-code contract enforced by regression tests.
9. [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
   The runtime, training, rollout, and validation workflow.
   Use [`Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch)
   for the current canonical full PICF long-run training command, and
   [`Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile)
   for the 2x40GB frozen-perception profile and experimental state-only
   burn-in speed path.
   Use
   [`Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation)
   for the current cloud-tested CALVIN video evaluation recipe.
   Use
   [`Section 6.1.3`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#613-anchor-task-readout-and-predictive-cache-diagnostics)
   for anchor overlays, task-readout attention JSONL, and compact predictive
   cache diagnostics.

Use `README_v2.2.md` for the current live system,
`PICF_AQR_OWM_CURRENT_STATE_20260518.md` for the compact cleanup/module
disposition verdict,
`README_PI05_PARITY_AUDIT.md` for PI0.5/PICF dataflow comparisons,
`README_FROZEN_PERCEPTION_AUGMENTATION.md` for frozen-perception and
augmentation design, `README_VL_GUIDED_ANCHOR_ROUTER.md` for the lower-level
point-centric VL router substrate, `README_MAPG_PICF.md` for the current live
modality-optional graph-level anchor implementation,
`PICF_FORMAL_CONTRACT.md` for the concise executable contract, and
`README_v2.1.md` only as the archived pre-v2.2 deployment record.

Fast operator path:

1. Start with
   [`README_v2.2.md#01-current-training--model-summary`](./README_v2.2.md#01-current-training--model-summary)
   for the current model/training summary.
   The current switch matrix and active A7 long-run ledger are now recorded near
   the top of [`README_v2.2.md`](./README_v2.2.md). Keep `README_v2.2.md` as the
   stable live path during the active run; do not rename it unless all links and
   experiment notes are migrated together.
2. Use
   [`CALVIN_VALIDATION_README.md Section 5.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51-current-canonical-full-picf-long-run-launch)
   for the canonical full-PICF 4x40GB launch.
3. Use
   [`CALVIN_VALIDATION_README.md Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile)
   for the 2x40GB frozen-perception launch and the experimental
   `state_only` burn-in speed path.
4. Use
   [`CALVIN_VALIDATION_README.md Section 6.1`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#61-current-cloud-calvin-video-evaluation)
   for cloud CALVIN video evaluation and `/mnt` artifact handling.
5. Use
   [`CALVIN_VALIDATION_README.md Section 6.1.3`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#613-anchor-task-readout-and-predictive-cache-diagnostics)
   when evaluating whether cyan task overlays indicate real slot collapse,
   point-projection bias, or semantic/visual attention failure.
6. Use
   [`README_MAPG_PICF.md`](./README_MAPG_PICF.md)
   for the live MAPG-PICF graph implementation, including `--mapg-enabled`,
   graph losses, and the 30000-step / 5000-checkpoint launch template.

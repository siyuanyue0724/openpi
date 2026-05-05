# PICF Entry

This file is only a directory entry.

Current PICF documents:

1. [`README_v2.2.md`](./README_v2.2.md)
   The current local v2.2 architecture record and implementation audit for the
   live codebase, including the explicit effector/object anchor role split and
   global scene point-pool contract.
2. [`README_PI05_PARITY_AUDIT.md`](./README_PI05_PARITY_AUDIT.md)
   Code-level comparison of reference PI0.5 / PI0.5+Sonata dataflow against
   current PICF enabled and PI0.5-only ablated modes.
3. [`README_FROZEN_PERCEPTION_AUGMENTATION.md`](./README_FROZEN_PERCEPTION_AUGMENTATION.md)
   Design record for the 2x40GB frozen-perception profile and geometry-safe
   augmentation policy.
4. [`README_VL_GUIDED_ANCHOR_ROUTER.md`](./README_VL_GUIDED_ANCHOR_ROUTER.md)
   Current staged implementation record for the default-off point-centric
   PaliGemma-guided 2D-to-3D anchor prior router.
5. [`README_MAPG_PICF.md`](./README_MAPG_PICF.md)
   Architecture and implementation contract for MAPG-PICF, the proposed
   modality-optional anchor prior graph over PaliGemma, V-JEPA, Sonata,
   AnyTouch, and posterior supports.
6. [`README_v2.1.md`](./README_v2.1.md)
   Historical v2.1 deployment record retained for reference.
7. [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
   The executable live-code contract enforced by regression tests.
8. [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)
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
`README_PI05_PARITY_AUDIT.md` for PI0.5/PICF dataflow comparisons,
`README_FROZEN_PERCEPTION_AUGMENTATION.md` for frozen-perception and
augmentation design, `README_VL_GUIDED_ANCHOR_ROUTER.md` for the current
point-centric VL router substrate, `README_MAPG_PICF.md` for the proposed
modality-optional graph-level anchor design,
`PICF_FORMAL_CONTRACT.md` for the concise executable contract, and
`README_v2.1.md` only as the archived pre-v2.2 deployment record.

Fast operator path:

1. Start with
   [`README_v2.2.md#01-current-training--model-summary`](./README_v2.2.md#01-current-training--model-summary)
   for the current model/training summary.
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
   before designing any graph-level anchor upgrade that should remain robust to
   missing pointcloud, weak tactile evidence, or RGB-only datasets.

# PICF-AQR-OWM Experiment Note - Task Owner, SAM, Tracklets - 2026-05-17

Status: active diagnostic in progress; code repairs documented, behavior
acceptance pending.

## Scope

This note records the cleanup state after the task-owner proposal repair and the
decision to promote dataset-scale SAM/proposal and tracklet sidecar generation
to scheduled work.

The controlling issue ledger is:

```text
docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md
```

The task-owner math/dataflow audit is:

```text
docs/PICF_AQR_OWM_TASK_OWNER_DATAFLOW_MATH_20260517_TEMP.md
```

## Current Run

```text
run:
  picf_a7_task_owner_bias_samseg0_diag300_20260517

purpose:
  Test whether task-query visual support can route class-agnostic SAM/proposal
  objectness toward the task-owned physical object without hard labels.

remote metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_task_owner_bias_samseg0_diag300_20260517/metrics.jsonl

remote overlays:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_task_owner_bias_samseg0_diag300_20260517/anchor_overlays/
```

## Current Repair Run

```text
run:
  picf_a7_birth_transport_sam_phase0_anchoronly_diag300_20260517

purpose:
  Test the posterior birth/no-object transport repair after the step300 SAM
  diagnostic showed controlled active duplicates but reserve/inactive files
  being recreated from the same dustbin residual.

remote metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_birth_transport_sam_phase0_anchoronly_diag300_20260517/metrics.jsonl

remote overlays:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_birth_transport_sam_phase0_anchoronly_diag300_20260517/anchor_overlays/

local/remote launch checks:
  py_compile: PASS
  verify_picf_owm_contract.py: PASS
  picf_posterior_birth_transport_audit.py: PASS
  startup printed Posterior birth transport contract.
```

Acceptance focus:

```text
posterior_file_competition_birth_count bounded;
posterior_recycle_rate does not climb together with active overlap;
aqr_active_same_role_support_overlap_max does not steadily rise;
loss_anchor_pv does not trend upward;
proposal tokens remain nonzero;
tracklet tokens are explicitly absent or nonzero.
```

Step 50 observation:

```text
loss_total                                  0.4620
loss_anchor_pv                              0.7133
loss_pv_weak                                6.0086
loss_slot_jepa                              1.0715
aqr_same_role_support_overlap_max           0.1999
aqr_active_same_role_support_overlap_max    0.0213
posterior_file_competition_duplicate_max    0.9732
posterior_file_competition_active_dup_max   0.0000
posterior_file_competition_birth_count      0.7700
posterior_file_competition_birth_share_max  0.1443
posterior_recycle_rate                      0.0499
posterior_inactive_file_recycle_rate        0.0000
owm_proposal_tokens                         1.92 average / 32 in selected overlay
owm_tracklet_tokens                         0.0
```

Interpretation:

```text
The repair is behaving as designed at step50: birth_count is bounded, inactive
file recycle is zero, active duplicate overlap is zero, and active support
overlap is low.  Raw duplicate overlap remains high because reserve/context
files are still highly correlated; that is not the action-visible acceptance
metric, but it remains a warning to track through step100/300.
```

Step 100 observation:

```text
loss_total                                  0.4773
loss_anchor_pv                              0.8837
loss_pv_weak                                4.1812
loss_slot_jepa                              0.9685
aqr_same_role_support_overlap_max           0.9230
aqr_active_same_role_support_overlap_max    0.1889
posterior_file_competition_duplicate_max    0.9889
posterior_file_competition_active_dup_max   0.0000
posterior_file_competition_birth_count      0.0150
posterior_file_competition_birth_share_max  0.0010
posterior_recycle_rate                      0.0047
posterior_inactive_file_recycle_rate        0.0000
posterior_identity_switch_rate              0.6061
owm_proposal_tokens                         1.44 average
owm_tracklet_tokens                         0.0
```

Interpretation:

```text
The dustbin-broadcast repair is doing its specific job: birth transport is now
nearly off after initial assignment, inactive recycle remains zero, and active
duplicate overlap remains zero.  The remaining degradation is not the same root
cause as the step300 failure.  The new issue is upstream AQR support/routing
pressure: raw support overlap returns high and active support overlap starts
increasing, while anchor_pv worsens.  That points to object-support assignment
pressure, not posterior birth transport.
```

## Step Trend

```text
step                                      50          100         150         200
loss_total                               0.476377    0.529843    0.505773    0.562951
loss_anchor_pv                           0.737828    1.133378    1.185894    1.187573
loss_pv_weak                             6.056437    4.277687    3.873690    3.306262
loss_aqr_denoising                       1.771745    2.138244    2.006886    1.930972
loss_mapg_cycle                          0.367694    0.410990    0.436437    n/a
loss_mapg_routing                        0.677847    0.722459    0.708319    0.733738
loss_mapg_support_diversity              0.306764    0.249809    0.136392    n/a
task_owner_visual_prior_entropy          0.665582    0.415869    0.336349    n/a
task_owner_visual_prior_max              0.034154    0.189592    0.208159    n/a
task_owner_proposal_score_nonzero        0.291520    0.294486    0.210545    n/a
active_same_role_support_max             0.012676    0.046688    0.050520    0.347965
active_same_role_object_core_max         0.064381    0.173627    0.062917    0.025162
raw_same_role_support_max                0.157804    0.587146    0.928805    0.999196
raw_same_role_object_core_max            0.094022    0.507334    0.801448    n/a
active_duplicate_overlap                 0.000000    0.000000    0.000000    0.000000
posterior_identity_switch_rate           0.687778    0.710556    0.738889    0.736667
posterior_recycle_rate                   0.106766    0.122773    0.051220    0.015970
loss_action_default_equiv                0.182817    0.204859    0.177226    0.180460
```

Interpretation:

```text
1. task_owner_visual_prior is alive and sharpens over time.
2. proposal score is nonzero, so SAM/proposal sidecar is being consumed.
3. active duplicate overlap remains controlled.
4. active object-core overlap recovered at step150 and remains low at step200.
5. raw/reserve overlap still worsens to nearly 1.0; this is not the active acceptance metric,
   but it means reserve/context capacity remains correlated.
6. identity_switch remains high; this is not closed.
7. loss_anchor_pv does not yet recover, so target-object behavior is not
   behavior-accepted.
```

## Offline IsSameObject Probe

Command class:

```bash
python scripts/picf_owm_same_object_probe.py \
  --anchor-overlays <run>/anchor_overlays \
  --overlay-source posterior \
  --quadratic-probe diag_quadratic,low_rank_quadratic \
  --quadratic-probe-feature binding_signature \
  --output <run>/same_object_probe_posterior_step150.json
```

Result:

```text
binding_signature_cos_auc       = 0.943820
combined_auc                    = 1.000000
duplicate_candidate_fraction    = 0.238095
decision                        = binding_subspace_decodable_but_assignment_duplicates_candidates
```

Interpretation:

```text
The binding subspace is not empty.  The remaining problem is not "no
same-object signal"; it is that assignment/lifecycle still leaves duplicate
candidate owners in the current short diagnostic.  This should be re-run at
step300 and after sidecar-enabled experiments.
```

## Archived Completed Repairs

These are considered engineering-closed and should stay only as regression
guards:

```text
1. dense AnyTouch 768-D to hidden_dim projection.
2. soft tactile evidence below hard contact threshold.
3. no false tactile evidence from all-zero top-k.
4. cache read residual scaling.
5. skip latest posterior cache row to avoid double-counting.
6. state_only burn-in uses AQR measurement graph.
7. active duplicate overlap is separated from raw reserve overlap.
8. task-owner proposal dataflow is code-live.
```

## Open Items and Plans

### O1. Task-Owned Object Behavior

Status:

```text
closed-code / open-train / watch
```

Plan:

```text
1. Finish the current 300-step diagnostic.
2. Inspect active-only and with-gray overlays.
3. Rerun IsSameObject probe.
4. Accept only if active object-core overlap stays controlled and duplicate
   candidate fraction falls or remains low.
```

### O2. Dataset-Scale SAM/Proposal Sidecars

Status:

```text
running-prep / open-data / scheduled
```

Plan:

```text
1. Generate SAM/SAM2 proposals offline for every available RGB view.
2. Store proposal_centers_xy, proposal_boxes_xyxy, objectness, view_ids,
   source_ids per episode/frame.
3. Keep SAM out of online training.
4. Validate with owm_proposal_tokens > 0 and nonzero task-owner proposal score.
```

Operational launch plan:

```text
remote:
  A7 / ssh -p 28060 root@36.139.225.68

repo:
  /root/openpi_posterior_vla_clean

calvin root:
  /mnt/calvin_data/task_ABC_D

output root:
  /mnt/picf_sidecars/sam_proposals_vitb_full_20260517

preview root:
  /mnt/picf_sidecars/sam_proposals_vitb_full_20260517_previews

script:
  scripts/archive/picf_sam_proposal_precompute_legacy.py

parallelism:
  two segment shards, one per visible A100 GPU.

restart safety:
  --skip-existing is enabled by default.
```

Acceptance for the first generated frames:

```text
1. .npz files exist under output_root/training.
2. proposal_objectness has nonzero length.
3. proposal_centers_xy is [N,2], proposal_boxes_xyxy is [N,4].
4. proposal_view_ids contains static and/or gripper view ids as requested.
5. preview PNGs show plausible boxes and are not empty.
```

Operational result on A7:

```text
2026-05-17 04:25 CST:
  Smoke sidecar generation passed after correcting two operator issues:
    - non-interactive shell did not expose uv in PATH;
    - CALVIN root is a directory, so the sidecar command must use --backend dir.

  The sidecar script is intentionally run with system python plus:
    PYTHONPATH=/tmp/picf_sam_code/segment-anything:/root/openpi_posterior_vla_clean/src

  Smoke output:
    output_root=/mnt/picf_sidecars/sam_proposals_vitb_smoke_20260517
    preview_root=/mnt/picf_sidecars/sam_proposals_vitb_smoke_20260517_previews
    shards=2
    frames=6 total
    proposal_count=282 total
    per-frame proposal count=44..49
    proposal_view_ids include both static(0) and gripper(1)
    objectness range is approximately 0.90..1.00

  Preview inspection:
    static preview contains plausible object/background boxes;
    gripper preview contains gripper-view boxes;
    the boxes are diagnostic proposals only, not posterior truth.

2026-05-17 04:29 CST:
  Full dual-A100 generation launched:
    shard0 session=sam_sidecar_vitb_shard0_20260517 on CUDA_VISIBLE_DEVICES=0
    shard1 session=sam_sidecar_vitb_shard1_20260517 on CUDA_VISIBLE_DEVICES=1
    output_root=/mnt/picf_sidecars/sam_proposals_vitb_full_20260517
    preview generation disabled for full run to avoid filling /mnt.

  Full run command uses:
    --backend dir
    --views both
    --sam-model-type vit_b
    --points-per-side 16
    --max-proposals-per-view 32
    --log-every 25
    --skip-existing

  Restart invariant:
    sidecar npz paths are independent per frame;
    sharded manifests are written as manifest_training_shardXXX-of-YYY.json;
    --skip-existing makes the run restartable without regenerating completed frames.

  Early full-run verification:
    tmux sessions are alive:
      sam_sidecar_vitb_shard0_20260517
      sam_sidecar_vitb_shard1_20260517
    GPU utilization is active on both A100s.
    Early full output count exceeded 180 npz files.
    First inspected full npz files contain:
      proposal_centers_xy [N,2]
      proposal_boxes_xyxy [N,4]
      proposal_objectness [N]
      proposal_view_ids [N] with both static(0) and gripper(1)
      proposal_source_ids [N] with SAM source id

  Dataset scale:
    training split has 17,870 segments and 1,053,873 frames.
    This makes complete SAM sidecar generation a long offline job even on two
    A100s.  The run is still useful because outputs are restartable and usable
    incrementally, but full coverage should be treated as offline data
    production rather than an interactive smoke test.

12-hour bounded production revision:

```text
The full per-frame run was projected at roughly 10 days.  That is too slow for
the current operator budget, so the active production job was changed without
changing the sidecar semantics:

  output_root=/mnt/picf_sidecars/sam_proposals_vitb_stride8_p8_12h_20260517
  frame_stride=8
  frame_offset=0
  points_per_side=8
  max_proposals_per_view=32
  workers=4
  GPUs=2 A100s, two SAM workers per GPU

Mathematical rationale:
  Let z_t be the dense PICF observation tokens and p_t be frozen SAM proposal
  evidence.  Proposals are additive typed evidence:
      z'_t = z_t union p_t
  They are never hard labels and never replace dense V-JEPA/PG/point/tactile
  tokens.  The 12-hour job changes only the sampling measure over frames:
      t in each episode -> t where (local_frame_index mod 8) = 0
  It does not change the sidecar contract or the posterior update semantics.

Why this is coherent:
  - Segment-level sharding preserves episode locality per worker.
  - Frame-stride coverage gives broad task/scene coverage instead of a small
    contiguous prefix.
  - Lower points_per_side reduces SAM prompt density, but still produces
    objectness boxes with predicted-IoU/stability quality scores.
  - Missing proposal sidecars remain a no-op in training, so partial coverage is
    safe and can be expanded later.

This is a bounded sidecar production pass, not a new training objective.
```

Final active 12-hour schedule:

```text
The stride=8 run still projected close to or beyond the 12-hour budget after
measured throughput.  The active run is therefore phase-0 hierarchical coverage:

  output_root=/mnt/picf_sidecars/sam_proposals_vitb_stride16_p8_phase0_20260517
  frame_stride=16
  frame_offset=0
  points_per_side=8
  workers=4
  sessions:
    sam_fast_stride16_p8_phase0_shard0_20260517
    sam_fast_stride16_p8_phase0_shard1_20260517
    sam_fast_stride16_p8_phase0_shard2_20260517
    sam_fast_stride16_p8_phase0_shard3_20260517

This is not a different model.  It is the first phase of a hierarchical sidecar
measure:
  phase 0: stride=16, offset=0
  phase 1: stride=16, offset=8
  phase 2: stride=16, offsets=4 and 12
  phase 3: dense or task-priority slices

The sequence converges toward dense coverage without changing proposal schema
or retraining semantics.

2026-05-17 12:24 CST strict production audit:
  status=running
  npz_count≈60,296
  active_workers=4
  sampled_files=159
  bad_fields=0
  sampled frames with both static and gripper proposals=159/159
  proposals per frame min/median/max=25/32/43
  objectness min/median/max≈0.891/0.961/1.0
  normalized box area min/median/max≈0.00020/0.0234/0.976

Strict conclusion:
  Continue generation.  The sidecars are structurally valid, multi-view, and
  schema-compatible.  This remains frozen proposal evidence, not posterior
  truth and not a behavior acceptance result.
```

### O3. Dataset-Scale Tracklet Sidecars

Status:

```text
open-data / scheduled
```

Plan:

```text
1. Generate static and wrist RGB tracklets offline.
2. Save tracklet_xy, velocity, visibility, confidence, ids, view_ids, age.
3. Filter by visibility/confidence/forward-backward consistency.
4. Use as optional typed memory.  Missing tracklets must remain a safe no-op.
5. Validate with owm_tracklet_tokens > 0 and aqr_tracklet_support_max > 0.
```

Implementation status:

```text
2026-05-17:
  Added scripts/picf_tracklet_sidecar_precompute.py.

Design:
  - Reads existing SAM/proposal sidecars as high-quality objectness seeds.
  - At each keyframe, at least sam_seed_fraction of seeds are requested from
    SAM proposal centers when available.
  - Remaining seeds are low-confidence generic grid seeds.  This prevents the
    tracker from inheriting only SAM's proposal bias and preserves background /
    context coverage.
  - Tracks only short windows with KLT optical flow.
  - Writes existing PICF fields:
      tracklet_xy
      tracklet_velocity
      tracklet_visibility
      tracklet_confidence
      tracklet_ids
      tracklet_view_ids
      tracklet_age
  - Merges proposal_* from proposal_root into output frames when they are truly
    proposal frames.  It does not copy keyframe proposals onto non-keyframe
    images, because that would be false geometry.

Smoke test on A7:
  proposal_root=/mnt/picf_sidecars/sam_proposals_vitb_stride16_p8_phase0_20260517
  output_root=/mnt/picf_sidecars/tracklets_samseed_smoke_20260517
  segment_id=45
  keyframe_stride=16
  window_forward=15
  seeds_per_view=32
  sam_seed_fraction=0.5

Smoke result:
  saved_frames=64
  tracklet_count=3601
  keyframe sidecar contains both proposal_* and tracklet_*
  intermediate sidecars contain tracklet_* only, as intended
  tracklet_view_ids include both static(0) and gripper(1)

Acceptance:
  The generation code is ready for phased production after SAM phase0 has
  enough coverage.  It remains an offline data sidecar; missing tracklets are
  still safe no-ops in training.

A5 validation and production start:

```text
2026-05-17:
  A5 connection updated:
    ssh -p 28718 root@px-cloud2.matpool.com

  Environment:
    two A100-40GB GPUs visible and idle;
    cv2, numpy, torch available;
    CALVIN root and SAM sidecars mounted under /mnt.

  Requirement change:
    Data quality is preferred over quantity.  The generator therefore added:
      --require-proposal-keyframe
    Production mode skips keyframes without SAM proposal sidecars instead of
    falling back to grid-only tracks.

  A5 smoke:
    segment_indices=32,33,47,63
    keyframe_stride=16
    window_forward=15
    seeds_per_view=32
    sam_seed_fraction=0.5
    require_proposal_keyframe=true

    saved_frames=239
    tracklet_count=14,315
    proposal frames=27
    all saved frames have tracklet_view_ids containing both static(0) and gripper(1)
    per-frame tracklets: min=47, median=61, max=64
    confidence: min≈0.151, median≈0.430, max=1.0

  Production started on A5:
    output_root=/mnt/picf_sidecars/tracklets_samseed_stride16_w15_phase0_20260517
    workers=8 CPU processes
    sessions=tracklet_samseed_phase0_shard0..7_20260517
    proposal_root=/mnt/picf_sidecars/sam_proposals_vitb_stride16_p8_phase0_20260517
    require_proposal_keyframe=true

  2026-05-17 12:24 CST strict production audit:
    status=running
    npz_count≈30,463
    active_workers=8
    sampled_files=139
    bad_fields=0
    per-frame tracklets: min=45, median=59, max=64
    static+gripper tracklets present in 139/139 sampled frames
    duplicate track ids in sampled frames=0
    confidence: min≈0.152, median≈0.437, max=1.0
    age: min=0.0, median≈0.467, max=1.0
    normalized speed: median≈0.00071, max≈0.191
    proposal-bearing sampled frames=19/139

  This tracklet sidecar is same-contract data production, not a new model loss.

  Strict conclusion:
    Continue generation.  The produced fields are structurally valid and
    consistent with the MVTrack typed-memory contract.  Do not yet claim final
    behavior acceptance: the generator is KLT-window evidence, not CoTracker/TAPIR
    ground truth, and the first production pass started before SAM phase0 fully
    completed.  After SAM phase0 finishes, rerun the same tracklet command with
    --skip-existing-tracklets to fill keyframes that were skipped by
    --require-proposal-keyframe.
```

2026-05-17 14:45 CST sidecar/training gate update:

```text
SAM phase0 on A7:
  output_root=/mnt/picf_sidecars/sam_proposals_vitb_stride16_p8_phase0_20260517
  status=complete
  npz_count=66,928
  sampled_files=159
  bad_fields=0
  sampled frames with both static and gripper proposals=159/159
  proposals per frame min/median/max=25/32/43
  decision:
    The proposal sidecar is accepted for proposal-memory diagnostics.

Tracklet phase0 on A5:
  output_root=/mnt/picf_sidecars/tracklets_samseed_stride16_w15_phase0_20260517
  status=partial
  npz_count≈333,593
  quality on generated files remains structurally valid, but production is not
  complete because two shards hit dataset read errors:
    shard0: EOFError: No data left in file
    shard4: zipfile.BadZipFile: File is not a zip file
  decision:
    Do not use this tracklet root as a full training sidecar yet.  Keep it as
    partial evidence/debug data until the bad CALVIN records are skipped and the
    missing shards are backfilled.

A7 proposal-memory frozen diagnostic:
  purpose:
    Validate that completed SAM proposal sidecars can enter the AQR proposal
    memory path without activating tracklet sidecars.
  first launch failure:
    tactile dense reread crashed with a 512-vs-768 key width mismatch.
  root cause:
    AnyTouch patch tokens leave AnyTouch at native width, but PICF stores dense
    tactile memory after `tactile_patch_token_proj`, i.e. in PICF hidden_dim
    space.  The train-time lazy warmup incorrectly initialized tactile reread
    projections at native AnyTouch width.
  fix:
    `scripts/picf_core_train.py::_infer_tactile_dense_dim()` now returns
    `core.config.hidden_dim`, matching the actual dense-memory dataflow.
  acceptance:
    local py_compile passed; remote py_compile passed; A7 500-step tmux
    diagnostic relaunched with proposal_memory_enabled and perception frozen.

  Step-50 first structural read:
    loss_total=0.7959
    loss_alignment=0.7209
    loss_anchor_pv=0.7533
    loss_mapg_cycle=0.3695
    loss_mapg_support_diversity=0.2900
    loss_mapg_routing=0.6795
    loss_action_weight_scale=0.0
    owm_proposal_tokens=3.62
    owm_proposal_valid_fraction=0.11
    owm_tracklet_tokens=0.0
    aqr_same_role_support_overlap_max=0.1733
    aqr_active_same_role_support_overlap_max=0.0184
    aqr_effective_anchor_count=5.33
    aqr_active_anchor_count=5.56
    posterior_recycle_rate=0.0556
    posterior_identity_switch_rate=0.7278
    posterior_file_competition_duplicate_overlap_max=0.9985
    grad_clip_applied=true
    preclip_grad_norm=9.72

    Overlay inspection:
      prompt="grasp the red block and turn it left"
      with_gray and active_only overlays were generated.
      Active graph anchors are concentrated around task-relevant block/gripper
      regions and active same-role overlap is low.  However, task anchors still
      collapse to one pixel cluster and several posterior files duplicate the
      same projected point.  This is acceptable for a frozen proposal-memory
      smoke/diagnostic start, but it is not final behavioral acceptance.

  Step-100/150 trend:
    step50:
      loss_total=0.7959
      loss_anchor_pv=0.7533
      loss_pv_weak=6.2616
      aqr_same_role_support_overlap_max=0.1733
      aqr_active_same_role_support_overlap_max=0.0184
      posterior_recycle_rate=0.0556
      posterior_identity_switch_rate=0.7278
      posterior_file_competition_duplicate_overlap_max=0.9985
    step100:
      loss_total=0.9344
      loss_anchor_pv=0.9978
      loss_pv_weak=5.4799
      aqr_same_role_support_overlap_max=0.6564
      aqr_active_same_role_support_overlap_max=0.0831
      posterior_recycle_rate=0.1211
      posterior_identity_switch_rate=0.7306
      posterior_file_competition_duplicate_overlap_max=1.0000
    step150:
      loss_total=0.9232
      loss_anchor_pv=1.1255
      loss_pv_weak=4.7910
      aqr_same_role_support_overlap_max=0.9693
      aqr_active_same_role_support_overlap_max=0.1685
      posterior_recycle_rate=0.2437
      posterior_identity_switch_rate=0.7672
      posterior_file_competition_duplicate_overlap_max=1.0000
    step200:
      loss_total=0.9816
      loss_anchor_pv=1.1766
      loss_pv_weak=4.4893
      loss_slot_jepa=2.5151
      aqr_same_role_support_overlap_max=0.9980
      aqr_active_same_role_support_overlap_max=0.2588
      aqr_same_role_object_core_overlap_max=0.8712
      aqr_active_same_role_object_core_overlap_max=0.0295
      posterior_recycle_rate=0.2849
      posterior_identity_switch_rate=0.7511
      posterior_file_competition_duplicate_overlap_max=1.0000
      posterior_file_competition_active_duplicate_overlap_max=0.0000
    step250:
      loss_total=1.0327
      loss_anchor_pv=1.1764
      loss_pv_weak=4.1960
      loss_slot_jepa=4.0184
      aqr_same_role_support_overlap_max=0.9994
      aqr_active_same_role_support_overlap_max=0.3831
      aqr_same_role_object_core_overlap_max=0.9305
      aqr_active_same_role_object_core_overlap_max=0.0147
      posterior_recycle_rate=0.2777
      posterior_identity_switch_rate=0.7650
      posterior_file_competition_duplicate_overlap_max=1.0000
      posterior_file_competition_active_duplicate_overlap_max=0.0000
    step300:
      loss_total=0.9723
      loss_anchor_pv=1.1787
      loss_pv_weak=4.0849
      loss_slot_jepa=4.7354
      aqr_same_role_support_overlap_max=0.9996
      aqr_active_same_role_support_overlap_max=0.4216
      aqr_same_role_object_core_overlap_max=0.9322
      aqr_active_same_role_object_core_overlap_max=0.0230
      posterior_recycle_rate=0.2743
      posterior_identity_switch_rate=0.7761
      posterior_file_competition_duplicate_overlap_max=1.0000
      posterior_file_competition_active_duplicate_overlap_max=0.0000
      posterior_file_competition_active_count=3.845
      owm_proposal_tokens=4.395
      owm_tracklet_tokens=0.000

    Interpretation:
      Proposal sidecar and task-owner path are active, and active-only support
      overlap remains below the full collapse metric.  However, full same-role
      overlap reaches near 1.0 by step200, active same-role overlap rises from
      0.0184 at step50 to 0.4216 at step300, anchor_pv worsens, recycle rises,
      and the slot-JEPA diagnostic drifts upward.  This means the active filter
      protects direct duplicate overlap, but the underlying candidate/posterior
      lifecycle is not solved.  SAM proposal memory is active, but with only
      about 3-5 proposal tokens per batch it is insufficient by itself to
      stabilize file ownership.  Treat this run as a diagnostic negative for
      final long-run acceptance unless later steps reverse these trends.
```

### O4. Ordinal / Fourth-Object Grounding

Status:

```text
open-data
```

Plan:

```text
Do not claim solved.  Current ordinal state is diagnostic.  True closure needs
rank labels or a validated weak selected-slot source after target-object
selection is stable.
```

### O5. Full Action Cotrain

Status:

```text
open-train
```

Plan:

```text
Run only after the structural diagnostic is accepted.  The current maintained
long-run contract is the post-blind-SAM-default-off behavior-acceptance profile:

  purpose:
    30k action cotrain / behavior acceptance, not anchor-only diagnosis.

  trainability:
    freeze pretrained perception feature extractors:
      V-JEPA
      Sonata
      AnyTouch
    keep trainable:
      PaliGemma / PI0.5 semantic-action stack
      PICF AQR/posterior/task/control adapters
      action-side and prediction heads

  recurrent window:
    --unroll-steps 2
    --burnin-steps 1
    --burnin-mode state_only

  action:
    --lambda-action-pos 0.50
    --lambda-action-rot 0.50
    --lambda-action-gripper 0.50
    --picf-action-prefix-stopgrad

  guarded OWM hooks:
    --lambda-slot-jepa 0.0
    --lambda-support-pred 0.0
    --lambda-binding-consistency 0.0
    --lambda-aqr-denoising 0.0

  sidecars:
    tracklet memory may remain enabled when a vetted tracklet sidecar root is
    supplied; missing tracklets are a safe no-op.
    blind SAM/proposal memory is production-default off:
      no --proposal-memory-enabled
      proposal read/point/task-owner proposal weights remain 0.0.
    prompted/reranked/contact-guided proposal sidecars are future ablations,
    not part of the current production acceptance run.

  cadence:
    --num-train-steps 30000
    --log-interval 50
    --anchor-overlay-interval 50
    --save-interval 2500
    --keep-last-checkpoints 3
    --progress
```

## Current Decision

```text
Do not declare "all issues solved" yet.

The code-level repairs are substantially complete for the current architecture.
The remaining questions are behavior/data acceptance:
  - target-object selection under task pressure;
  - duplicate candidate reduction;
  - tracklet sidecar activation;
  - broader SAM sidecar coverage;
  - eventual action cotrain.
```

### 2026-05-17 sparse proposal overlay diagnosis

Observation:

```text
The active overlays for the current A7 birth-transport diagnostic do not show a
clear posterior object file on the red block.  In particular:

  step150 active_only:
    active posterior pixels around gripper/drawer edge, not red block center.
  step200 active_only:
    active posterior again remains near gripper/drawer edge; red block has no
    clear posterior circle.

Therefore the earlier reading "anchors are near the task object" was too weak
and should not be treated as acceptance.
```

Box overlay root cause:

```text
The same run only drew SAM/proposal boxes at step50.  It did not draw boxes at
step100/150/200 because the saved overlay sample had no exact sidecar proposal:

  step50  step_id=1572920 exact sidecar exists -> proposals=32
  step100 step_id=1572925 exact sidecar absent -> proposals=0
  step150 step_id=1572917 exact sidecar absent -> proposals=0
  step200 step_id=1572925 exact sidecar absent -> proposals=0

This does not contradict interval metrics such as owm_proposal_tokens > 0,
because those are averaged across the training interval and can include other
windows/ranks.
```

Local repair:

```text
Added explicit age-aware nearest proposal sidecar fallback:

  --mvtrack-sidecar-proposal-nearest-max-gap
  --proposal-age-decay-steps

The fallback keeps the main dataset immutable, records proposal_age in the
observation/core state/overlay JSON, and decays proposal objectness:

  objectness_eff = objectness * exp(-proposal_age / proposal_age_decay_steps)

This is weak temporal evidence, not a hard current-frame label.
```

Verification:

```text
python -m py_compile:
  scripts/picf_core_train.py
  scripts/serve_picf_policy.py
  scripts/verify_picf_owm_contract.py
  scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py
  src/openpi/picf/contracts.py
  src/openpi/picf/core/contracts.py
  src/openpi/picf/core/config.py
  src/openpi/picf/core/pipeline.py
  src/openpi/picf/replay/calvin_replay.py

python scripts/verify_picf_owm_contract.py:
  PASS includes sparse_proposal_sidecars_are_age_aware_not_silent_missing

python scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py --skip-external-code --fail-on-fail:
  15/15 PASS
```

Next diagnostic:

```text
Deploy the age-aware nearest sidecar fallback to A7 and rerun the 300-step
anchor-only diagnostic with:

  --mvtrack-sidecar-proposal-nearest-max-gap 8
  --proposal-age-decay-steps 8.0

Acceptance is visual and scalar:
  boxes must appear on non-exact sidecar overlay frames with /a{gap} labels;
  active posterior object files should move onto the task red block rather than
  only gripper/drawer edges;
  posterior birth/inactive-recycle must remain fixed;
  if red-block binding still fails, the next target is task-conditioned support
  assignment calibration rather than sidecar coverage.
```

### 2026-05-17 proposal-to-point bridge diagnosis

The sparse-sidecar fallback diagnostic reached step100 and confirmed:

```text
proposals=32
nearest proposal to red block ~= (106,156) px
red block visual center ~= (107,157) px
nearest active anchors to red block ~= 20/25/30 px away
closest graph row to red block ~= 6 px away but inactive/context
```

This means the remaining failure is not missing SAM boxes.  The dataflow gap is:
proposal support can be present without moving `anchor_x`, because `anchor_x`
was still computed from point support only.

Implemented local repair:

```text
proposal_priors
  -> _proposal_priors_to_point_priors()
  -> bounded mix into point_priors
  -> anchor_x = point_priors @ point_positions

task_owner_proposal_score
  -> _task_owner_proposal_to_point_priors()
  -> bounded mix into task-owner physical point_priors
  -> anchor_x = point_priors @ point_positions

task_owner_proposal_score / task_owner_visual_prior
  -> _task_owner_anchor_score()
  -> added to active-slot anchor_scores for physical task-object rows
```

New debug:

```text
aqr_proposal_point_bridge_entropy_mean
aqr_proposal_point_bridge_max
aqr_task_owner_point_bridge_entropy_mean
aqr_task_owner_point_bridge_max
aqr_task_owner_point_bridge_nonzero_fraction
aqr_task_owner_anchor_score_max
aqr_task_owner_anchor_score_mean
aqr_task_owner_anchor_score_nonzero_fraction
```

Acceptance:

```text
The next A7 diagnostic must show nonzero proposal-point and task-owner-point
bridge metrics, and at least one active physical/posterior object file near the
red-block proposal.  If not, the remaining failure is semantic task-owner
calibration or active-file selection, not proposal coverage or 2D-to-3D
transport.
```

### 2026-05-17 task-owner point bridge step50 result

The A7 v2 diagnostic reached step50 and no longer reproduces the earlier
"proposal present but active geometry far away" failure:

```text
nearest proposal to red block ~= 1.4 px
nearest active graph anchor to red block ~= 11.2 px
nearest active posterior anchor to red block ~= 11.9 px
aqr_proposal_point_bridge_max ~= 0.041
aqr_task_owner_point_bridge_max ~= 0.017
aqr_task_owner_anchor_score_max ~= 0.193
aqr_same_role_support_overlap_max ~= 0.174
posterior_file_competition_active_duplicate_overlap_max = 0
```

Interpretation:

```text
The direct task-owner proposal-to-point bridge is not merely a logging change:
it has altered the overlay geometry and pulled active object files into the
red-block neighborhood.  Continue to step100/300 before declaring the issue
closed, because semantic ownership can still drift under continued updates.
```

Step100 regression:

```text
nearest active anchor to red block ~= 24 px
nearest active posterior to red block ~= 41 px
aqr_task_owner_point_bridge_max ~= 0.020
aqr_task_owner_proposal_score_max = 1.0
loss_anchor_pv ~= 1.18
loss_aqr_denoising ~= 2.03
aqr_same_role_support_overlap_max ~= 0.69
```

Interpretation:

```text
The bridge is correctly connected but too weak/transient as an observation
likelihood.  It can pull the task object at step50, then the anchor/PV/support
objectives pull active files away by step100.  The next repair should keep the
same math family and calibrate task-owner proposal-point evidence as persistent
measurement support, not add an unrelated auxiliary module.
```

Implemented follow-up repair:

```text
task_owner_proposal_score
  -> proposal-to-point matrix
  -> centered log likelihood over projected points
  -> added to AQR point-reader logits for eligible physical scene rows
  -> point_priors / q / anchor_x all receive the same owner measurement
```

New knob:

```text
task_owner_proposal_point_bias_weight = 0.75
```

V3 step100 result:

```text
nearest active graph to red block ~= 14 px
nearest active task to red block ~= 15 px
nearest active posterior to red block ~= 24 px
aqr_task_owner_point_bridge_max ~= 0.026
loss_anchor_pv ~= 1.06
aqr_same_role_support_overlap_max ~= 0.688
posterior_file_competition_active_duplicate_overlap_max = 0
```

Interpretation:

```text
V3 materially improves v2 step100, but it does not fully close the issue.  The
red-block proposal now keeps active graph/task evidence near the object edge,
while posterior ownership still lags and same-role support overlap still rises.
This should be treated as a partial repair pending step300/long-run validation,
not as a completed issue.
```

### 2026-05-17 blind SAM default-off decision

The later full-sidecar frozen-anchor diagnostic
`picf_a7_full_sidecar_anchoronly_diag300_20260517` invalidated the idea that
blind automatic SAM proposals should remain production-default evidence.

Observed:

```text
step50:
  owm_tracklet_tokens ~= 58.24
  owm_proposal_tokens ~= 1.92
  aqr_same_role_support_overlap_max ~= 0.216
  aqr_active_same_role_support_overlap_max ~= 0.056

step100:
  owm_tracklet_tokens ~= 57.42
  owm_proposal_tokens ~= 1.44
  aqr_task_owner_point_bridge_max ~= 0.00069
  aqr_same_role_support_overlap_max ~= 0.448

step150:
  owm_tracklet_tokens ~= 57.76
  owm_proposal_tokens ~= 1.44
  aqr_task_owner_point_bridge_max ~= 0.00091
  aqr_same_role_support_overlap_max ~= 0.962
```

The specific saved red-block overlays at step100/150 contained no proposal or
tracklet records, while the averaged metrics were nonzero.  Therefore the
sidecar loader works, but blind SAM does not provide reliable per-frame
task-object ownership.  It can add high-quality mask fragments for walls,
drawer sides, or robot protrusions without making the named object posterior
more correct.

Decision:

```text
Production defaults now disable blind proposal influence:
  proposal_memory_enabled = False
  proposal_read_weight = 0.0
  proposal_point_bridge_weight = 0.0
  task_owner_proposal_bias_weight = 0.0
  task_owner_proposal_point_bias_weight = 0.0
  task_owner_proposal_point_bridge_weight = 0.0

Retain:
  sidecar schema and loader
  proposal overlay diagnostics
  proposal precompute scripts
  opt-in proposal AQR path for prompted/reranked proposal ablations
  tracklet sidecars
```

This is not deleting the sidecar architecture.  It is separating a good
evidence-transport mechanism from a noisy blind proposal source.

### 2026-05-18 sidecar-precondition frozen-PaliGemma action diagnostic

Purpose:

```text
Run a short action-enabled diagnostic before tracklet sidecar generation
finishes.  The run freezes PaliGemma in addition to V-JEPA/Sonata/AnyTouch, so
it isolates whether the current AQR/posterior/action-side adapters can train
without semantic-backbone cotrain.
```

This is not the maintained production recipe.  The production long-run keeps
PaliGemma trainable because prior action cotrain evidence suggests semantic
adaptation is useful.  This diagnostic answers a narrower question:

```text
If sidecars are absent and PaliGemma is frozen, does action pressure alone
damage or preserve the repaired active posterior/anchor structure?
```

Launch invariant:

```text
Do not pass --use-foundation-backbones in this diagnostic.  That helper sets
semantic_trainable=True by design.  Instead pass the foundation modes and
checkpoint paths explicitly:
  --point-backbone sonata
  --visual-mode encoder
  --visual-feature-mode hierarchical
  --tactile-mode encoder
  --use-tactile
  --semantic-mode paligemma
  --perception-finetune-mode frozen
  no --semantic-trainable
```

Proposal/SAM invariant:

```text
Blind SAM/proposal memory stays disabled:
  --no-proposal-memory-enabled
  --proposal-read-weight 0.0
  --proposal-point-bridge-weight 0.0
  --task-owner-proposal-bias-weight 0.0
  --task-owner-proposal-point-bias-weight 0.0
  --task-owner-proposal-point-bridge-weight 0.0

This is important because the A7 remote checkout may still have the older
proposal_memory_enabled=True default.  The command must override that default
explicitly until the remote is resynced.
```

Acceptance checks:

```text
step50/100/150/200/300:
  loss_action_default_equiv decreases or stays within short-run noise;
  loss_anchor_pv does not monotonically worsen;
  aqr_active_same_role_support_overlap_max stays below the collapse regime;
  posterior_recycle_rate does not saturate;
  overlays keep at least one active posterior/anchor near the task-object
  neighborhood.
```

Interpretation:

```text
  Pass:
  PaliGemma freezing is not an immediate blocker for short action adaptation.

Fail while semantic-trainable runs remain healthier:
  semantic cotrain is structurally needed for the current CALVIN action path;
  this does not invalidate AQR, but it means the production recipe should keep
  PaliGemma trainable rather than waiting for sidecars to compensate.
```

First launch finding:

```text
FSDP full-shard is not the right wrapper for this frozen-PaliGemma diagnostic:
when PaliGemma is frozen, the mixed float32/bfloat16 trainable surface can trip
FSDP flat-parameter dtype uniformity.  The diagnostic is intentionally short,
so run it single-GPU with training_strategy=ddp instead.

The first single-GPU attempt exposed a separate valid dataflow bug: frozen
AnyTouch dense patch tokens are produced under torch.inference_mode(), then
enter trainable tactile_patch_token_proj.  PyTorch cannot save inference
tensors for backward.  The correct repair is to clone the frozen encoder output
into a normal detached tensor before the trainable PICF projection; this keeps
the two-stage math unchanged:

  frozen tactile encoder: x_tactile = stopgrad(E_tactile(obs))
  trainable PICF adapter: z_tactile = W_tactile clone(x_tactile)

This is not disabling tactile evidence and not adding a new loss.  It restores
the intended frozen-backbone/ trainable-adapter contract.
```

Relaunch status:

```text
run:
  picf_a7_frozen_pg_action_diag300_20260518
  single GPU0, training_strategy=ddp, world_size=1
  PaliGemma frozen, V-JEPA/Sonata/AnyTouch frozen
  proposal_memory_enabled=False
  no mvtrack_sidecar_root

startup contract confirmed:
  Backbone contract:
    point=sonata(trainable=False)
    visual=encoder(finetune_mode=frozen trainable=False)
    tactile=encoder(trainable=False)
    semantic=paligemma(trainable=False)
```

Step50 first metrics:

```text
loss_total = 0.7791
loss_action_default_equiv = 0.1391
loss_action_active7 = 0.4779
loss_anchor_pv = 0.6983
loss_mapg_cycle = 0.3632
loss_mapg_support_diversity = 0.2770
aqr_active_same_role_support_overlap_max = 0.0720
aqr_same_role_support_overlap_max = 0.1942
posterior_recycle_rate = 0.0579
posterior_stable_slot_fraction = 0.4556
owm_tracklet_tokens = 0
owm_proposal_tokens = 0
```

Interpretation:

```text
The frozen-PaliGemma action diagnostic is now technically valid and has not
immediately reproduced the previous action-induced overlap collapse.  The
active same-role support overlap is low at step50, recycle is not saturated,
and proposal/tracklet evidence is correctly absent.  This is only the first
log interval; continue to step100/150/300 before deciding whether frozen
PaliGemma is adequate or whether semantic cotrain is still required.
```

Step200 update:

```text
step  loss_action_default_equiv  loss_anchor_pv  active_overlap  raw_overlap  recycle  stable_slot
  50  0.1391                     0.6983          0.0720          0.1942       0.0579   0.4556
 100  0.1371                     0.7147          0.0839          0.7136       0.0147   0.3978
 150  0.1217                     0.8340          0.2752          0.9795       0.0022   0.3567
 200  0.1125                     1.1226          0.3531          0.9907       0.0016   0.3500
```

Interpretation:

```text
This diagnostic gives a split answer.  Freezing PaliGemma does not prevent
action loss from improving: loss_action_default_equiv falls from 0.139 to
0.112 by step200.  But the anchor/posterior structure degrades after step100:
raw same-role overlap returns to ~0.99, active overlap rises above the healthy
<0.25 gate by step200, loss_anchor_pv climbs, and recycle becomes almost zero.

Therefore the sidecar-precondition answer is:
  - action-side adapters can learn with frozen PaliGemma;
  - frozen PaliGemma is not sufficient to maintain the repaired active-anchor
    structure under continued action pressure;
  - production cotrain should keep PaliGemma trainable unless a later sidecar
    or task-object evidence run proves otherwise.
```

Step300 frozen-PaliGemma closeout:

```text
step  loss_action_default_equiv  loss_anchor_pv  active_overlap  raw_overlap  recycle  stable_slot  preclip_grad_norm
  50  0.1391                     0.6983          0.0720          0.1942       0.0579   0.4556       29.65
 100  0.1371                     0.7147          0.0839          0.7136       0.0147   0.3978       32.81
 150  0.1217                     0.8340          0.2752          0.9795       0.0022   0.3567       179.65
 200  0.1125                     1.1226          0.3531          0.9907       0.0016   0.3500       106.10
 250  0.1099                     1.1682          0.2687          0.9792       0.0015   0.3589       41.68
 300  0.1071                     1.1772          0.2845          0.9820       0.0014   0.3722       555.54
```

Mathematical interpretation:

```text
The frozen-PaliGemma run proves that action-side trainable adapters can reduce
the action objective under a fixed semantic backbone, but it also proves that
fixed semantics are not enough to keep the AQR/posterior assignment manifold
stable under action pressure.  The key pattern is not the action loss itself;
it is the coupled drift:

  action improves,
  loss_anchor_pv worsens,
  raw same-role overlap returns to the collapse regime,
  recycle is almost zero.

In belief-filter terms, the action objective is improving the downstream
control decoder while the measurement-to-slot assignment kernel loses
semantic adaptability.  That is not an acceptable production tradeoff unless
sidecar evidence later supplies an independent task-object signal.
```

Follow-up trainable-PaliGemma control run:

```text
run:
  picf_a7_trainable_pg_action_nosidecar_diag300_20260518

purpose:
  same no-sidecar action diagnostic, but with PaliGemma trainable, to isolate
  whether semantic cotrain stabilizes task-object assignment.

confirmed startup:
  world_size=2
  training_strategy=fsdp_full_shard
  semantic=paligemma(trainable=True)
  point=sonata(trainable=False)
  visual=encoder(finetune_mode=frozen trainable=False)
  tactile=encoder(trainable=False)
  proposal_memory_enabled=False
  mvtrack_sidecar_root=None
  unroll_steps=2
  burnin_steps=1

status:
  launched on A7 at 2026-05-18 01:40 local server time.
  First metrics are expected at step50; until then only startup/progress has
  been verified.
```

Step50 control metrics:

```text
run                         act_def  act7    total   anchor_pv  active_overlap  raw_overlap  recycle  stable_slot  preclip_grad_norm
frozen-PG no-sidecar          0.1391  0.4779  0.7791  0.6983     0.0720          0.1942       0.0579   0.4556       29.65
trainable-PG no-sidecar       0.1205  0.4427  0.8182  0.7065     0.0390          0.2132       0.0647   0.4511        4.59
```

Initial interpretation:

```text
The first comparable point favors trainable PaliGemma.  The action objective
is lower than the frozen diagnostic at the same step, active overlap is lower,
recycle is not suppressed, and the preclip gradient norm is much smaller.

This is not yet sufficient to finalize the diagnosis, because the frozen run
only began to structurally degrade after step100.  Continue the trainable run
through at least step150/300.  The decisive test is whether the trainable run
avoids the frozen pattern:

  action improves while loss_anchor_pv rises,
  raw same-role overlap returns to ~0.98,
  active overlap rises above ~0.25,
  recycle collapses toward zero.
```

## 2026-05-22 A7 Action-Aware Quality-Gate Decay Long Run

This section records the current maintained A7 production-style long run after
the object-file / sidecar / active-owner repairs.  The run keeps Sonata,
AnyTouch, and V-JEPA frozen, leaves PaliGemma and PICF connectors trainable,
uses `unroll_steps=2` with `burnin_steps=1`, writes anchor overlays every 100
steps, saves every 500 steps, and keeps the latest 3 checkpoints.

The first branch was:

```text
run:
  picf_a7_actionaware_qgdecay_from500_long30k_20260522

resume:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_defaultsync_action2_from0_long30k_20260522/500

decay:
  OBJECT_SCAFFOLD_DECAY_MODE=cosine
  OBJECT_SCAFFOLD_DECAY_START_STEP=500
  OBJECT_SCAFFOLD_DECAY_END_STEP=2000
  OBJECT_SCAFFOLD_DECAY_FLOOR=0.10
  ACTION_LOSS_WEIGHT=2.0
```

Step1000 gate:

```text
loss_total                                      0.1925
loss_action_default_equiv                      0.0515
loss_total_minus_action                        0.1410
action fraction                                26.8%
object_scaffold_decay_scale                    0.5514
loss_anchor_object_pull                        0.3914
loss_anchor_pv                                 0.6829
loss_object_explanation_point                  4.4165
aqr_active_same_role_support_overlap_max       0.1092
aqr_active_same_role_object_core_overlap_max   0.0047
aqr_downstream_same_role_support_overlap_max   0.1339
posterior_file_competition_active_duplicate    0.0000
posterior_identity_switch_rate                 0.2222
posterior_recycle_rate                         0.0473
grad_norm                                      0.7847
grad_clip_applied                              false
```

Gate decision:

```text
Do not hard-jump action weight at step1000.

Reason:
  active/core/downstream ownership metrics are healthy, but the action
  fraction is still low because the object scaffold remains too large.  A hard
  action-weight jump would confound two effects: stronger action pressure and
  abrupt removal of the weak object teacher.  The mathematically cleaner move
  is to keep ACTION_LOSS_WEIGHT=2.0 and accelerate the cosine scaffold decay so
  action receives a larger share of the same total budget.
```

The active replacement branch is:

```text
run:
  picf_a7_actionaware_qgdecay_fast1300_from1000_long30k_20260522

resume:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_qgdecay_from500_long30k_20260522/1000

decay:
  OBJECT_SCAFFOLD_DECAY_MODE=cosine
  OBJECT_SCAFFOLD_DECAY_START_STEP=500
  OBJECT_SCAFFOLD_DECAY_END_STEP=1300
  OBJECT_SCAFFOLD_DECAY_FLOOR=0.10
  ACTION_LOSS_WEIGHT=2.0
```

Step1050/1100 validation:

```text
step                                      1050       1100
loss_total                               0.1184     0.1120
loss_action_default_equiv                0.0431     0.0450
loss_action_active7                      0.1924     0.2015
loss_total_minus_action                  0.0753     0.0670
action fraction                          36.4%      40.2%
object_scaffold_decay_scale              0.3015     0.2331
loss_anchor_object_pull                  0.3355     0.3571
loss_anchor_pv                           0.6704     0.6281
loss_object_explanation_point            4.2462     5.1647
active same-role support overlap max     0.0999     0.0974
active same-role object-core overlap max  0.0039     0.0045
downstream same-role support overlap max  0.1320     0.1335
raw same-role support overlap max         1.0000     1.0000
active duplicate overlap max             0.0000     0.0000
posterior identity switch rate           0.1989     0.2439
posterior recycle rate                   0.1032     0.0979
grad_norm                                0.4897     0.7703
grad_clip_applied                        false      false
```

Interpretation:

```text
The fast-decay branch is stable through step1100.

The intended effect occurred:
  action share rose from 26.8% at step1000 to 40.2% at step1100 without a hard
  action-weight discontinuity.

The structural acceptance metrics remain healthy:
  active support overlap stays around 0.10, active core overlap remains near
  zero, downstream overlap stays near 0.13, active duplicate overlap remains
  exactly zero, and gradients do not spike.

The raw same-role overlap remains near 1.0, but this is currently reserve /
inactive telemetry rather than the active acceptance metric.  Do not optimize
against raw overlap directly unless active/core/downstream metrics also fail.

The only watch items are:
  loss_object_explanation_point is noisy and rose at step1100;
  identity_switch_rate increased from 0.20 to 0.24;
  action loss did not monotonically improve between 1050 and 1100, although it
  remains better than the step1000 gate and comparable to the old 4-22
  5k-step level.
```

Next gate:

```text
Continue the fast1300 branch unless one of these occurs:
  active same-role support overlap > 0.25 for two consecutive log points;
  downstream same-role support overlap > 0.25 for two consecutive log points;
  active duplicate overlap > 0;
  repeated grad clipping;
  recycle collapses toward zero while identity_switch_rate rises;
  action loss remains flat after the scaffold reaches its 0.10 floor.

If the next gate fails, resume from step1000 and use a slower end step
(`OBJECT_SCAFFOLD_DECAY_END_STEP=1400` or `1500`) rather than increasing a new
auxiliary loss.
```

### 2026-05-22 literature matrix and step-1500 action gate

Maintained documents:

```text
docs/PICF_AQR_OWM_SLOT_VLA_PAPER_MATRIX_20260522.md
docs/PICF_AQR_OWM_ACTION_DOMINANT_WEIGHT_AUDIT_20260522.md
```

The slot/VLA paper matrix now tracks 38 papers/systems across:

```text
object binding / IsSameObject probes
MetaSlot / QASA / slot merging / temporal slot consistency
object-centric robotics and manipulation
JEPA / V-JEPA / predictive VLA
action-dominant VLA recipes
tactile / contact-rich VLA
```

Step1500 gate from `picf_a7_actionaware_qgdecay_fast1300_from1000_long30k_20260522`:

```text
loss_total                               0.0658
loss_action_default_equiv                0.0416
loss_total_minus_action                  0.0242
action fraction                          63.3%
object_scaffold_decay_scale              0.10
active same-role support overlap max     0.0500
downstream same-role support overlap max  0.0781
active duplicate overlap max             0.0000
posterior identity switch rate           0.2172
posterior recycle rate                   0.1024
loss_slot_jepa                           3002.5 diagnostic only; lambda 0
```

Decision:

```text
The action lambdas are already at the traditional PICF/PI0 scale:
  lambda_action_pos = lambda_action_rot = lambda_action_gripper = 2.0

Do not raise action above the legacy/default scale as the first move.
The paper-consistent action-dominant continuation is to lower the weak
object-scaffold floor from 0.10 to 0.03 after active ownership is healthy.
This should move action from roughly 63% of the total to about 74-78% without
changing action-gradient units.
```

Planned continuation:

```text
EXP=picf_a7_actionaware_qgfloor003_from1500_long30k_20260522
RESUME_CHECKPOINT=/mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_qgdecay_fast1300_from1000_long30k_20260522/1500
ACTION_LOSS_WEIGHT=2.0
OBJECT_SCAFFOLD_DECAY_MODE=cosine
OBJECT_SCAFFOLD_DECAY_START_STEP=500
OBJECT_SCAFFOLD_DECAY_END_STEP=1500
OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
SAVE_INTERVAL=500
KEEP_LAST_CHECKPOINTS=3
LOG_INTERVAL=50
ANCHOR_OVERLAY_INTERVAL=100
```

### 2026-05-26 two-timescale cotrain gate

Maintained document:

```text
docs/PICF_AQR_OWM_COTRAIN_TWO_TIMESCALE_FINAL_20260526.md
```

Evidence entering this gate:

```text
A7 same-timescale cotrain:
  stopped at step 7318.
  latest logged step 7300:
    loss_action_default_equiv ~= 0.04763
    loss_total_minus_action   ~= 0.01079
    active overlap            ~= 0.105
    downstream overlap        ~= 0.138
  interpretation:
    active structure was not collapsing, but action remained on a moving-prefix
    plateau/rebound band.

PC1 policy-only causal probe:
  `picf_trainable_scope=policy_only`, structural losses 0.
  step 6050 -> 6300:
    loss_action_default_equiv 0.03512 -> 0.03316
    best observed             0.02890 at step6200
  interpretation:
    freezing `core.*` while keeping PICF forward enabled supports the
    moving-prefix root-cause hypothesis.
```

Decision:

```text
Do not promote policy_only to production.
Keep cotrain, but split optimizer timescales:
  semantic/action path learns normally;
  PICF core remains trainable but uses a slow LR scale.
```

New launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_twotimescale_cotrain_from_pc1_6000_30k_20260526.sh
```

Run contract:

```text
EXP=picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526
RESUME_CHECKPOINT=/mnt/checkpoints/picf_core/picf_core/picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526/6000
PICF_TRAINABLE_SCOPE=all
ACTION_LOSS_WEIGHT=2.0
SEMANTIC_LR_SCALE=0.35
PICF_CORE_LR_SCALE=0.05
POLICY_HEAD_LR_SCALE=1.0
LR=7e-5
MIN_LR=2e-5
OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
SAVE_INTERVAL=500
KEEP_LAST_CHECKPOINTS=3
LOG_INTERVAL=50
ANCHOR_OVERLAY_INTERVAL=100
```

First acceptance:

```text
step100:
  verify optimizer group scales in log.

step300:
  compare action trend against A7 same-timescale run and PC1 policy-only.

step500:
  continue only if action does not rebound while active/downstream overlap and
  posterior lifecycle metrics remain healthy.
```

Launch verification on A7:

```text
remote:
  ssh -p 28060 root@36.139.225.68

repo:
  /root/openpi_twotime_cotrain_20260526

tmux:
  picf_a7_twotime_cotrain_20260526

log:
  /mnt/picf_run_logs/picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526.log

metrics:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526/metrics.jsonl
```

Verified runtime args from `args.json` after launch:

```text
resume_checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
    picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526/6000

num_train_steps                         30000
save_interval                           500
keep_last_checkpoints                   3
picf_trainable_scope                    all
semantic_trainable                      true
semantic_lr_scale                       0.35
picf_core_lr_scale                      0.05
policy_head_lr_scale                    1.0
lambda_action_pos/rot/gripper           2.0 / 2.0 / 2.0
optimizer_checkpoint_mode               model_only
training_strategy                       fsdp_full_shard
optimizer_sharding                      none
action_prefix_norm_mode                 rmsnorm
mvtrack_sidecar_root:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520

lambda_slot_jepa                        0.0
lambda_support_pred                     0.0
lambda_binding_consistency              0.0
lambda_aqr_denoising                    0.0
aqr_role_layout                         object_only
effector_persistent_anchors             0
task_effector_queries                   0
tactile_attach_to_object_owner          true
```

Local and remote code verification before launch:

```text
local:
  python -m py_compile scripts/picf_core_train.py scripts/picf_core_train_test.py
  pytest -q scripts/picf_core_train_test.py -k 'optimizer or trainable_scope or normalize_train_args'
  python scripts/verify_picf_owm_contract.py
  python scripts/picf_owm_dataflow_trace.py --fail-on-fail
  python scripts/picf_owm_strict_diagnose.py --fail-on-fail

remote:
  script/document SHA matched local for:
    scripts/picf_core_train.py
    scripts/picf_core_train_test.py
    run_a7_twotimescale_cotrain_from_pc1_6000_30k_20260526.sh
    docs/PICF_AQR_OWM_COTRAIN_TWO_TIMESCALE_FINAL_20260526.md

  checkpoint path exists.
  sidecar segment file exists.
  old A7 tmux was stopped before the new launch.
```

Startup observation:

```text
13:24:32:
  resumed from checkpoint step 6000.

first progress:
  step 6001:
    loss ~= 0.0098
    lr   ~= 6.53e-05
    wall ~= 26.5 sec/step

  step 6002:
    loss ~= 0.0384
    lr   ~= 6.52e-05

GPU:
  two A100-40GB active, ~40GB each, 100% utilization during first steps.

Interpretation:
  run is live and using the intended heavy full cotrain graph, not a
  lightweight anchor-only or policy-only diagnostic.  Do not judge loss trend
  until the first log interval writes step6050 metrics.
```

First metric gate at step 6050:

```text
current two-timescale cotrain:
  loss_total                         0.046562
  loss_action_default_equiv          0.035607
  loss_action_active7                0.161482
  loss_total_minus_action            0.010955
  loss_anchor_object_pull            0.249496
  loss_object_explanation_point      7.25229
  loss_object_explanation_contact    0.315857
  loss_mapg_cycle                    0.385852
  loss_mapg_support_diversity        0.114093
  active same-role support overlap   0.089886
  downstream same-role support       0.087471
  raw same-role support overlap      1.000000
  posterior_recycle_rate             0.127679
  posterior_identity_switch_rate     0.168333
  pi_prefix_post_rms_mean/max        1.000000 / 1.000000
  grad_norm                          0.429182
  loss_slot_jepa                     0.643886  (telemetry; lambda 0)
```

Comparison against the previous same-timescale A7 action-polish run in the
same 6000-6500 window:

```text
old same-timescale mean action_default_equiv   0.045545
current step6050 action_default_equiv          0.035607
relative change                                -21.8%

old same-timescale mean loss_total             0.056617
current step6050 loss_total                    0.046562
relative change                                -17.8%

old active overlap mean                        0.105998
current active overlap                         0.089886

old downstream overlap mean                    0.113896
current downstream overlap                     0.087471

old identity switch mean                       0.186717
current identity switch                        0.168333
```

Comparison against the PC1 policy-only causal probe:

```text
PC1 policy-only step6050 action_default_equiv  0.035121
current step6050 action_default_equiv          0.035607

Interpretation:
  current two-timescale cotrain is very close to policy-only action quality at
  the first gate, while still keeping PICF core trainable and preserving
  structural losses.  This is the intended behavior.  It is not yet proof of
  long-run stability; the next gates are step6100, step6300, step6500, and
  step7000.
```

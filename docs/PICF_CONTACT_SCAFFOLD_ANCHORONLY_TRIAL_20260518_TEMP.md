# PICF Contact/Task Proposal Scaffold Anchor-Only Trial

Date: 2026-05-18

Status: deployment plan for a 1000-step diagnostic, not production acceptance.

## Goal

Run a short controlled training probe where PaliGemma/semantic parameters and
the PI0.5 action/control heads are frozen, while the PICF anchor router,
posterior binding/address path, and typed support adapters remain trainable.
If the launch keeps action lambdas nonzero, action loss still provides a
diagnostic gradient into the trainable PICF path; only the action/control head
weights are frozen.

The trial asks one narrow question:

```text
Given a weak but task-relevant current-frame contact/object proposal sidecar,
can the anchor/posterior machinery learn healthier task-object support without
changing the semantic or action heads?
```

This is intentionally not a final policy run. It is a binding dataflow
diagnostic.

## External Paper-Code Audit

Reference code checked into `temp/external_repos/` for model comparison:

```text
temp/external_repos/SlotLifter
temp/external_repos/MetaSlot
```

The relevant shared structure is not a specific loss copied verbatim; it is the
slot dataflow:

```math
K = W_k f_i,\quad V = W_v f_i,\quad Q = W_q s_j
```

```math
A_{j,i} = softmax_j(Q_j K_i / \sqrt d)
```

```math
\tilde A_{j,i} = A_{j,i} / \left(\sum_i A_{j,i}+\epsilon\right)
```

```math
u_j = \sum_i \tilde A_{j,i} V_i,\quad
s_j^+ = GRU(u_j, s_j)
```

SlotLifter then decodes point/ray evidence with an explicit empty/background
slot. MetaSlot adds prototype/codebook style slot regularization. These designs
support two PICF requirements:

```text
1. object evidence must be typed and competitive, not dense all-to-all;
2. background/inactive evidence must have a safe sink and must not become task
   object truth.
```

PICF does not directly transplant either repo because CALVIN is a two-view
RGB-D manipulation setting with posterior/action state, not many-view NeRF
reconstruction. The imported principle is the competitive slot update and
empty/background separation.

## Dataflow

The new generator is:

```text
scripts/picf_contact_motion_sidecar_precompute.py
```

It writes the already-supported MVTrack sidecar keys:

```text
proposal_centers_xy
proposal_boxes_xyxy
proposal_objectness
proposal_view_ids
proposal_source_ids
```

The trainer already reads them through:

```text
scripts/picf_core_train.py::_read_mvtrack_sidecar_fields
scripts/picf_core_train.py::_CalvinTransitionSource._load_frame
src/openpi/picf/core/pipeline.py::_build_token_field
```

Then they enter:

```text
PicfPseudoProposalState
-> AQR proposal reader / task-owner proposal score
-> graph.proposal_priors / proposal_point_priors
-> observation anchors / posterior signatures
```

No CALVIN `.npz` frame is mutated.

## Mathematical Contract

For each frame, the sidecar emits a proposal measurement:

```math
z_t^{prop}=(c_t, b_t, q_t, source)
```

where `c_t` is a normalized static-view center, `b_t` is a normalized static
box, and `q_t in [0,1]` is objectness.  A proposal box is allowed to be stronger
than a generic visual prior, but only as a calibrated measurement likelihood:
it must not become a hard posterior label.

The complete sidecar now also stores sparse soft-mask evidence:

```text
proposal_mask_xy
proposal_mask_weights
proposal_mask_offsets
```

This is the preferred proposal-to-point bridge.  It avoids the row-index
problem: generated sidecars and training point tokens may use different
sampling/FPS subsets, so sidecars store normalized support sample coordinates
rather than training-time point row ids.

Default proposal scoring is current-frame causal:

```math
score_i =
  \begin{cases}
    0.70\,color_i + 0.18\,contact_i + 0.08\,wrist_i + 0.04\,sat_i,
      & \text{colored object task}\\
    0.98\,contact_i^2 + 0.02\,wrist_i,
      & \text{actuator/effect task}
  \end{cases}
```

For actuator/effect tasks, color words are not used unless the prompt explicitly
names a colored object. This prevents the known failure mode where `push the
button` binds to the lamp/LED effect rather than the button actuator.

The proposal is soft evidence:

```math
q'_j = q_j + \lambda_{prop}(ReadProp(q_j, M_{prop}) - q_j)
```

The projected point likelihood is:

```math
P(i \mid b_k) \propto
\sigma((x_i-x^0_k)/\tau)
\cdot\sigma((x^1_k-x_i)/\tau)
\cdot\sigma((y_i-y^0_k)/\tau)
\cdot\sigma((y^1_k-y_i)/\tau)
v_i
```

When sparse mask samples are present, the runtime uses the stronger mask
likelihood instead of the box fallback:

```math
P(i \mid m_k) \propto
\frac{
  \sum_{s \in \mathcal S_k} w_s
  \exp(-\|u_i-u_s\|^2/(2\tau_m^2))
}{
  \sum_{s \in \mathcal S_k} w_s + \epsilon
}
v_i
```

and the runtime multiplies it by a soft box quality prior:

```math
q_k =
objectness_k
\cdot shape\_quality(b_k)
\cdot exp(-age_k/T).
```

This is the same mathematical role as reference boxes in Deformable DETR/DINO
and point/box prompts in SAM-style segmentation: the box is a strong spatial
reference, not object truth.  It narrows attention to plausible tokens while
still allowing dense V-JEPA/PG/point evidence and posterior correction to reject
the proposal.

The 2026-05-18 step-400 audit showed that a single event-level green proposal
box can be too broad.  Therefore the generator is now component-based:

```text
high-score task/contact pixels
-> connected components in static image space
-> up to three compact proposal boxes
-> per-proposal objectness
```

This replaces the rejected one-box percentile envelope that could cover a whole
drawer/event region and dilute support.

It is not a label and does not overwrite posterior state:

```text
proposal sidecar -> typed support prior
not:
proposal sidecar -> posterior truth
```

The default generator does not use future frames. A future segment end-effector
path can be enabled only with `--use-segment-ee-path` for visual diagnosis; it
is off for this training trial.

## Trial Configuration

Use:

```text
--picf-trainable-scope anchor_only
--perception-finetune-mode frozen
--proposal-memory-enabled
--mvtrack-sidecar-root <generated contact sidecar root>
--calvin-segment-indices <generated segment list>
```

`anchor_only` freezes:

```text
PaliGemma / semantic stack
PI0.5 action/control heads
V-JEPA / Sonata / AnyTouch pretrained backbones
predictive heads
```

and trains:

```text
AQR router
observation-anchor adapters
posterior binding/address
support/cache/proposal/tracklet typed evidence path
```

High-risk predictive objectives remain off:

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
```

## Acceptance Checks

This trial is useful only if logs show:

```text
picf_trainable_scope=anchor_only
semantic_trainable=False
visual/tactile/point trainable=False
owm_proposal_tokens > 0
owm_tracklet_tokens >= 0
aqr_proposal_support_max nonzero
aqr_task_owner_proposal_score_nonzero_fraction nonzero
```

Anchor health should improve without declaring behavior success:

```text
aqr_same_role_support_overlap_max should not return to persistent 0.98-1.00
posterior_recycle_rate should not saturate
loss_anchor_pv should not monotonically diverge
loss_action_default_equiv is diagnostic only because action head is frozen
```

## Rejection Conditions

Stop or mark the trial failed if:

```text
proposal sidecar exists but owm_proposal_tokens remains 0
proposal support becomes the only active evidence source
anchor overlays bind to lamp/effect for button/switch actuator tasks
same-role overlap still saturates despite proposal support
action loss drops while posterior identities visibly ignore task objects
```

## Run Script

Canonical script:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_contact_scaffold_anchoronly_1000_20260518.sh
```

Expected remote sidecar root:

```text
/mnt/picf_sidecars/contact_motion_causal_1000_20260518
```

Updated compact-proposal generator flags:

```text
--top-fraction 0.015
--box-pad-px 4.0
--max-proposals-per-frame 3
--component-radius-px 10.0
--component-min-points 6
--box-percentile-low 12.0
--box-percentile-high 88.0
--mask-samples-per-proposal 96
```

These flags keep proposal boxes useful as strong spatial references while
avoiding the single oversized event-box failure observed in the earlier
step-400 overlay.

## A7 Launch Record

Launched on A7 at 2026-05-18 evening:

```text
tmux session:
  picf_a7_contact_anchor1000_20260518

master log:
  /mnt/picf_run_logs/picf_a7_contact_scaffold_anchoronly_1000_20260518.master.log

training log:
  /mnt/picf_run_logs/picf_a7_contact_scaffold_anchoronly_1000_20260518.log
```

Generated sidecar data before training:

```text
proposal frames:
  1000 / 1000

used CALVIN segment ids:
  0..17

tracklet frames:
  994

tracklet observations:
  83260

proposal source id:
  8

training sidecar mode:
  causal current-frame default, no segment-future EE path
```

Startup contract observed:

```text
world_size = 1
picf_trainable_scope = anchor_only
trainable_numel = 81,300,462
total_numel = 4,082,230,269
semantic_trainable = False
visual/tactile/point trainable = False
proposal_memory_enabled = True
tracklet_memory_enabled = True
mvtrack_sidecar_root = /mnt/picf_sidecars/contact_motion_causal_1000_20260518
```

Early runtime:

```text
first optimizer steps entered normally
speed about 6.5-7.1 sec/step on one A100
GPU0 memory about 20.9GB
GPU1 unused
```

## Step-50 Runtime Evidence

The first structured metric record confirms the intended dataflow is active:

```text
owm_proposal_tokens = 0.96
owm_proposal_valid_fraction = 0.96
owm_tracklet_tokens = 81.27
owm_tracklet_valid_fraction ~= 1.0
aqr_proposal_support_max = 0.96
aqr_task_owner_proposal_score_nonzero_fraction = 0.83
aqr_task_owner_proposal_score_max = 0.83
aqr_tracklet_support_entropy_mean = 0.9393
aqr_tracklet_support_max = 0.0776
```

The first anchor-health metrics are substantially better than the previously
observed raw-collapse runs:

```text
aqr_active_same_role_support_overlap_max = 0.0464
aqr_same_role_support_overlap_max = 0.1766
aqr_active_anchor_count = 6.93
aqr_effective_anchor_count = 6.67
posterior_recycle_rate = 0.0519
posterior_active_file_recycle_rate = 0.0575
posterior_file_competition_active_duplicate_overlap_max = 0.0
```

This is not yet a pass. The same step also shows two values that must be
watched through step 100/200 overlays:

```text
posterior_identity_switch_rate = 0.6722
posterior_identity_switch_rate_stable = 0.5068
posterior_file_competition_duplicate_overlap_max = 0.8881
```

Interpretation:

```text
1. Proposal and tracklet evidence are not no-op.
2. Active support competition is behaving much better than the old collapse
   pattern.
3. Raw posterior duplicate risk still exists, but the active duplicate gate is
   suppressing it at step 50.
4. Identity switches are still too high to call the binding solved before
   overlay inspection and later metric records.
```

The action metrics are diagnostic only because action/control is frozen:

```text
loss_action_default_equiv = 0.1840
loss_action_active7 = 0.6576
loss_action_weight_scale = 0.25
```

They should not be compared directly to full co-train runs as an acceptance
criterion for this anchor-only diagnostic.

## Full Follow-Through Audit

The sidecar generator writes only this payload:

```text
proposal_centers_xy
proposal_boxes_xyxy
proposal_objectness
proposal_view_ids
proposal_source_ids
```

The train-side path is:

```text
scripts/picf_core_train.py::_read_mvtrack_sidecar_fields
-> scripts/picf_core_train.py::_CalvinTransitionSource._load_frame
-> PicfObservation.proposal_*
-> src/openpi/picf/core/pipeline.py::_build_token_field
-> PicfPseudoProposalState
-> AQR proposal reader
-> graph.proposal_priors
-> proposal-to-point bridge
-> task-owner proposal score/bias
-> observation anchors and posterior support signatures
```

The tracklet sidecar path is separate:

```text
scripts/picf_tracklet_sidecar_precompute.py
-> PicfObservation.tracklet_*
-> PicfTrackletSupportState
-> AQR tracklet reader
-> graph.tracklet_priors
-> posterior tracklet_signature
```

The proposal sidecar is therefore typed evidence. It is not a hard label, it
does not overwrite posterior state, and it does not feed future observations
into the current online path.

## Paper-Code Comparison

The external references kept under `temp/external_repos` are used as design
audits:

```text
SlotLifter:
  competitive object slots plus explicit empty/background slot in a 3D
  reconstruction pipeline.

MetaSlot:
  prototype/codebook-guided slots with masked duplicate handling.
```

The common mathematical pattern is:

```math
A_{j,i} =
softmax_j(q_j^\top k_i / \sqrt d + b_{j,i})
```

with a slot update:

```math
s_j^+ = Update(s_j, \sum_i A_{j,i} v_i).
```

PICF keeps this principle but does not copy the reconstruction objective:

```text
SlotLifter/MetaSlot optimize object-centric reconstruction/segmentation.
PICF optimizes a posterior belief router for manipulation.
```

The design choice is deliberate: proposal/tracklet sidecars act as weak
scaffolds that bias measurement routing, while the posterior remains the
authoritative belief state.

## Open Acceptance Boundary

This trial can answer whether current-frame task/contact proposals and
proposal-seeded tracklets stabilize anchor binding. It cannot answer:

```text
1. whether full action co-training is solved;
2. whether ordinal/fourth-object grounding is solved;
3. whether proposal generation is sufficient for the full CALVIN distribution;
4. whether long-horizon identity stability holds beyond this 1000-step probe.
```

Those require follow-up with overlay inspection and a longer controlled run.

## Step-100 Overlay/Metric Review

Step 100 produced the expected overlay artifacts:

```text
anchor_overlays/step_000100__go_towards_the_pink_block_in_the_drawer_and_pick_it_up__active_only.png
anchor_overlays/step_000100__go_towards_the_pink_block_in_the_drawer_and_pick_it_up__with_gray.png
anchor_overlays/step_000100__go_towards_the_pink_block_in_the_drawer_and_pick_it_up__sidecar_proposals.png
```

Note: this A7 run was launched before the 2026-05-18 overlay rename, so
existing artifacts from that live process may still use the legacy filename
`__sam_proposals.png`. That is a filename-only legacy artifact; the sidecar
source is contact-motion `source_id=8`, not blind SAM.

The sidecar proposal for this prompt is broad but task-aligned:

```text
prompt:
  go towards the pink block in the drawer and pick it up

proposal center:
  x=0.436, y=0.742

proposal box:
  [0.302, 0.462, 0.643, 0.846]

objectness:
  0.584
```

The active overlay shows active graph/posterior anchors concentrated around the
drawer opening, gripper, and visible colored blocks. This is much better than
the historical failure where many same-role anchors collapsed onto one
unrelated region, but it is not perfect object binding yet.

Step-100 single-window diagnostics:

```text
aqr_active_same_role_support_overlap_max = 0.0265
aqr_same_role_support_overlap_max = 0.3490
posterior_recycle_rate = 0.0207
posterior_identity_switch_rate = 0.4444
```

Step-100 rolling metrics:

```text
aqr_active_same_role_support_overlap_max = 0.1038
aqr_same_role_support_overlap_max = 0.2998
posterior_recycle_rate = 0.0040
posterior_identity_switch_rate = 0.6478
loss_anchor_pv = 1.1455
loss_total = 1.0302
preclip_grad_norm = 114.75
```

Interpretation:

```text
Positive:
  active same-role support overlap remains far below the old 0.95-1.00
  collapse regime.
  recycle is low, and active duplicate overlap remains suppressed.
  proposal and tracklet evidence remain nonzero.

Negative / watch:
  identity switch is still high.
  loss_anchor_pv increased from step 50 to step 100.
  preclip grad norm spiked and is being clipped.
  task query anchors in the diagnostic JSON still project to nearly identical
  task-readout positions; this is acceptable only if physical posterior files
  stay diverse and task-readout duplicates do not feed back as object truth.
```

Conclusion at step 100:

```text
The trial is stable enough to continue. It has not yet proven solved binding.
The next hard checks are step 200 trend and whether overlays keep task-object
anchors on the relevant object/drawer region rather than drifting to gripper or
background effect regions.
```

## 2026-05-18 Slot-Paper Audit Correction

The maintained slot/object-centric theory reference for this trial is now:

```text
temp/audits_20260518/slot_object_centric_2025_2026_math_followthrough.md
```

Key correction:

```text
SAM is not the architecture reference for PICF binding.  Blind SAM remains a
rejected historical proposal source.  The maintained references are MetaSlot,
SlotContrast, and QASA-style adaptive slot principles: evidence competition,
duplicate demotion, adaptive active/context/reserve capacity, background
capacity, and guarded temporal consistency.
```

Run acceptance is therefore not judged by raw reserve overlap alone:

```text
Raw overlap:
  may remain high because reserve/context/dustbin rows still exist.

Active/core overlap:
  must remain bounded because active object files are the slots that feed the
  downstream object/action path.

Proposal masks:
  must be present in sidecar npz and read as soft measurement evidence, not
  as hard labels.
```

The old single-card `picf_a7_contact_mask_anchor1000_20260518` attempt failed
before training because the remote code was not fully synced:

```text
AttributeError:
  PicfTransitionLossConfig has no attribute lambda_vcap_unexplained
```

That failure is a synchronization/contract issue, not a sidecar-quality issue.
The mask sidecar itself was generated successfully under:

```text
/mnt/picf_sidecars/contact_motion_mask_1000_20260518
```

The next valid trial must use the synced local config/training contract and run
with:

```text
NPROC_PER_NODE=2
FORCE_SIDECAR=0
FORCE_TRACKLETS=0
SIDECAR_ROOT=/mnt/picf_sidecars/contact_motion_mask_1000_20260518
```

## 2026-05-18 Dual-GPU Mask-Sidecar Restart

Launched on A7:

```text
session:
  picf_a7_contact_mask_anchor1000_dual_20260518

sidecar:
  /mnt/picf_sidecars/contact_motion_mask_1000_20260518

contract:
  NPROC_PER_NODE=2
  trainable_scope=anchor_only
  semantic=paligemma(trainable=False)
  action/control head frozen by anchor_only scope
  proposal_memory_enabled=True
  tracklet_memory_enabled=True
  num_train_steps=1000
```

Startup confirmed:

```text
world_size=2
effective_global_batch=2
unroll_steps=2
burnin_steps=1
active_slot_max_per_role=2
same_role_support_competition_weight=0.85
mvtrack_sidecar_root=/mnt/picf_sidecars/contact_motion_mask_1000_20260518
```

Step-50 early metrics:

```text
loss_total = 0.8759
loss_anchor_pv = 0.7450
loss_pv_weak = 6.3079
loss_mapg_graph = 0.1186
loss_mapg_cycle = 0.3584
loss_mapg_support_diversity = 0.3804
loss_pt = 0.3306
loss_action_default_equiv = 0.1572  # diagnostic only in anchor_only

owm_proposal_tokens = 1.64
owm_tracklet_tokens = 79.90
aqr_proposal_support_max = 0.9284
aqr_proposal_point_bridge_max = 0.0373

aqr_same_role_support_overlap_max = 0.1872
aqr_active_same_role_support_overlap_max = 0.0536
aqr_active_same_role_object_core_overlap_max = 0.1068

posterior_recycle_rate = 0.0770
posterior_stable_slot_fraction = 0.5206
posterior_identity_switch_rate = 0.6778
preclip_grad_norm = 21.19
grad_clip_applied = true
```

Interpretation:

```text
Positive:
  mask proposal and tracklet evidence are live;
  active same-role overlap is far from the old collapse regime;
  raw same-role overlap is also low at step 50;
  posterior recycle is not saturated.

Watch:
  identity switch is still high at early training;
  grad clipping is active;
  overlay evidence is not available until step 100.
```

Do not declare solved from step 50. The hard visual gate remains step 100
overlay plus step 100/200 trend.

## 2026-05-18 Step-100 Diagnosis: Proposal Selected, Anchor Not Transported

Step-100 dual-GPU metrics showed the key failure mode more precisely:

```text
owm_proposal_tokens = 1.46
owm_tracklet_tokens = 80.675
aqr_proposal_support_max = 0.8833
aqr_task_owner_proposal_score_max = 0.7000
aqr_task_owner_proposal_selected_count = 0.745

aqr_proposal_point_bridge_max = 0.0271
aqr_task_owner_point_bridge_max = 0.0223

aqr_active_same_role_support_overlap_max = 0.0771
posterior_recycle_rate = 0.0082
posterior_identity_switch_rate = 0.6733
```

Interpretation:

```text
The proposal/mask evidence is not absent.  It is selected and read as proposal
support, but the transport from selected proposal to physical point prior is
too weak to reliably birth or move a physical anchor into the green task-object
region.  This explains overlays where the task proposal box/mask is correct
but the active anchor remains outside it.
```

This is not the same as the old raw overlap collapse:

```text
active overlap:
  healthy enough for early diagnosis.

proposal point transport:
  too weak to determine geometry.
```

## 2026-05-18 Reference-Anchor Transport Fix

The repair is the new `proposal_anchor_seed_*` contract.  It is the
Deformable-DETR/DINO-style reference-query invariant translated into PICF:

```text
task/contact proposal score
  -> top proposal masks
  -> proposal-to-point matrix
  -> bounded physical measurement rows
  -> normal AQR competition
  -> posterior correction
```

Important negative constraints:

```text
It does not hard-write posterior.
It does not delete dense V-JEPA/point/proposal tokens.
It does not turn sidecar masks into labels.
It does not replace active/context/reserve routing.
It does not replace the PI0.5 action path.
```

The next diagnostic run should enable:

```text
--proposal-anchor-seed-enabled
--proposal-anchor-seed-rows 2
--proposal-anchor-seed-weight 0.75
--proposal-anchor-seed-token-weight 0.20
--proposal-anchor-seed-score-floor 0.01
--proposal-anchor-seed-point-topk 96
--proposal-anchor-seed-point-power 1.75
```

Acceptance keys:

```text
aqr_proposal_anchor_seed_row_count > 0
aqr_proposal_anchor_seed_point_max meaningfully above old bridge max
aqr_proposal_anchor_seed_assignment_max near selected proposal confidence
active task anchors enter the green task-object proposal by overlay review
active same-role object-core overlap stays bounded
posterior recycle does not saturate
```

## 2026-05-18 A7 Refseed Run: Step 50/100 Check

Run:

```text
picf_a7_contact_refseed_anchor1000_dual_20260518
NPROC_PER_NODE=2
SIDECAR_ROOT=/mnt/picf_sidecars/contact_motion_mask_1000_20260518
trainable_scope=anchor_only
proposal_anchor_seed_enabled=True
```

Startup contract passed:

```text
world_size=2
effective_global_batch=2
unroll_steps=2
burnin_steps=1
proposal_anchor_seed rows=2 weight=0.75 token_weight=0.20
```

Step 50:

```text
aqr_proposal_anchor_seed_row_count = 0.920
aqr_proposal_anchor_seed_assignment_max = 0.800
aqr_proposal_anchor_seed_point_max = 0.0503
aqr_proposal_anchor_seed_entropy_mean = 0.433

aqr_proposal_point_bridge_max = 0.0383
aqr_task_owner_point_bridge_max = 0.0292

aqr_active_same_role_support_overlap_max = 0.0422
aqr_active_same_role_object_core_overlap_max = 0.164
posterior_recycle_rate = 0.0533
posterior_identity_switch_rate = 0.639
```

Step 100:

```text
aqr_proposal_anchor_seed_row_count = 0.995
aqr_proposal_anchor_seed_assignment_max = 0.910
aqr_proposal_anchor_seed_point_max = 0.0438
aqr_proposal_anchor_seed_entropy_mean = 0.517

aqr_proposal_point_bridge_max = 0.0304
aqr_task_owner_point_bridge_max = 0.0262

aqr_active_same_role_support_overlap_max = 0.0775
aqr_active_same_role_object_core_overlap_max = 0.272
aqr_same_role_support_overlap_max = 0.250
posterior_recycle_rate = 0.0222
posterior_identity_switch_rate = 0.617
```

Interpretation:

```text
The new reference-anchor path is live and stronger than the old point bridge.
It has not caused the old active same-role support collapse or recycle
saturation by step 100.

The remaining issue is identity continuity: posterior_identity_switch_rate is
still high, so this run should not be interpreted as a solved identity result.
The next visual gate is the step-100 overlay:
  step_000100__go_towards_the_pink_block_in_the_drawer_and_pick_it_up__*.png
under the run's anchor_overlays directory.
```

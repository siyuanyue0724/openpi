# PICF-AQR-OWM SAM Proposal Quality Repair - 2026-05-17

Status: runtime calibration deployed; blind SAM proposal influence is now
production-default off. Full task-object proposal acceptance still requires
prompted/seeded sidecar generation.

This note records the repair for the observed SAM proposal failure mode:
automatic SAM boxes include useful task objects such as the red block, but also
wall panels, robot protrusions, drawer sides, and other fragments.  Those boxes
are valid high-recall segmentation proposals, but they are not object labels and
must not become hard posterior truth.

## Failure Mode

Blind SAM automatic masks optimize image-mask quality, not manipulation-object
ownership.  The sidecar currently stores:

```text
proposal_centers_xy
proposal_boxes_xyxy
proposal_objectness = sqrt(predicted_iou * stability_score)
proposal_view_ids
proposal_source_ids
proposal_age
```

This is correct as frozen proposal evidence, but insufficient as task-object
evidence.  A high-quality wall patch can have high SAM objectness, while being
irrelevant to `push red block right`.  If proposal objectness alone feeds AQR
proposal attention or proposal-to-point bridging, the graph can transport the
wrong box into 3D support.

## Paper-Code Alignment

The local paper-code audit reads:

```text
/tmp/picf_sam_code/segment-anything/segment_anything/automatic_mask_generator.py
/tmp/picf_sam_code/segment-anything/segment_anything/predictor.py
/tmp/picf_sam_code/sam2/sam2/...
/tmp/picf_sam_code/openmask3d/...
/tmp/picf_sam_code/open3dis/...
```

The important architecture facts are:

```text
SAM AutomaticMaskGenerator:
  returns masks, boxes, predicted_iou, stability_score, and area.
  It does not know the robot task.

SAM/SAM2 predictor:
  supports point/box prompts after image embedding.
  Prompted masks are the right future path for contact/action/language seeded
  sidecars, but they require prompt sources.

OpenMask3D / Open3DIS style flows:
  treat 2D masks as proposals, then fuse them with point clouds/depth-aware
  mappings and learned/open-vocabulary features.  They do not treat every 2D
  automatic mask as final object identity.
```

Therefore PICF should preserve SAM as high-recall proposal memory and apply
task/contact/geometry calibration before proposal evidence influences object
files.

## Mathematical Contract

Let a proposal `k` have box `B_k`, SAM quality `o_k`, static-view task visual
prior `v_i` over visual grid tokens, and soft projected-point membership
`M_{k,p}`.

Old task-owner proposal score:

```math
s_k
=
\frac{\sum_i v_i 1[x_i \in B_k]}{\sqrt{\operatorname{coverage}(B_k)}}
\cdot o_k^\gamma
```

This over-trusts automatic SAM fragments.  The maintained repair introduces a
soft proposal geometry quality:

```math
q_k
=
g_{\text{area-low}}(a_k)
g_{\text{area-high}}(a_k)
g_{\text{aspect}}(r_k)
```

with sigmoid gates around configurable area and aspect thresholds.  Then:

```math
s'_k
=
s_k q_k
```

and only the top task-owner proposals above a relative floor are allowed to
steer task-owner proposal and point evidence:

```math
\tilde{s}_k
=
\operatorname{TopKFloor}\left(\frac{s'_k}{\max_l s'_l}\right)
```

The generic context proposal read also uses:

```math
o'_k = o_k q_k^{\eta}
```

so background fragments can still exist as weak context proposals, but do not
receive the same routing strength as plausible object boxes.

The proposal-to-point bridge remains weak:

```math
P_{\text{owner-point}}(p)
=
\sum_k \tilde{s}_k M_{k,p}
```

and enters AQR as a point-reader likelihood bias and bounded prior mixture.  It
does not overwrite dense V-JEPA tokens, point tokens, posterior state, or action
truth.

## Implemented Dataflow

```text
sidecar proposal_* arrays
  -> PicfObservation
  -> PicfPseudoProposalState
  -> _proposal_shape_quality()
  -> proposal reader base bias: objectness * quality^eta
  -> _proposal_scores_from_visual_prior()
  -> TopK/Floor task-owner proposal score
  -> _task_owner_proposal_point_bias()
  -> _task_owner_proposal_to_point_priors()
  -> AQR graph proposal/point priors
  -> posterior correction
```

Dense visual/temporal/point tokens remain in the graph:

```text
SAM proposals are additive typed evidence.
They are not a replacement for V-JEPA, PaliGemma image tokens, point tokens, or
posterior belief.
```

## New Guards

Config:

```text
proposal_shape_quality_enabled = True
proposal_shape_area_min = 0.002
proposal_shape_area_max = 0.35
proposal_shape_aspect_min = 0.20
proposal_context_quality_power = 0.50
task_owner_proposal_topk = 4
task_owner_proposal_score_floor = 0.05
```

Debug:

```text
aqr_proposal_shape_quality_mean
aqr_proposal_shape_quality_max
aqr_proposal_shape_quality_nonzero_fraction
aqr_task_owner_proposal_score_entropy
aqr_task_owner_proposal_selected_count
aqr_task_owner_proposal_shape_quality_mean
```

Scripts:

```text
scripts/picf_sam_proposal_dataflow_audit.py
scripts/verify_picf_owm_contract.py
```

## What This Does Not Claim

This repair does not claim that blind SAM solved object grounding.  It only
prevents low-value automatic proposals from being over-trusted by task-owner
proposal and point bridges.

The fully prompted path remains a sidecar-generation task:

```text
language/PG task prior + action endpoint + tactile/contact point + tracklet
  -> prompted SAM/SAM2 candidate masks
  -> proposal_* sidecars with source_ids and quality
```

That path is data-generation work, not a safe runtime inference to fake without
contact/object labels.

## Acceptance Criteria

Short diagnostic:

```text
owm_proposal_tokens > 0
aqr_proposal_shape_quality_max > 0
aqr_task_owner_proposal_selected_count <= task_owner_proposal_topk
aqr_task_owner_proposal_score_entropy lower than blind dense proposal score
aqr_task_owner_point_bridge_max nonzero when a task proposal exists
active object anchors visually approach task object boxes
```

Failure indicators:

```text
task-owner selected proposals are still wall/robot fragments
aqr_task_owner_proposal_selected_count is large
task-owner proposal score stays high on thin/giant fragments
active anchors ignore the task object despite correct SAM box coverage
```

## A7 Runtime Result

Run:

```text
picf_a7_sam_quality_anchoronly_diag300_20260517
```

Step 50 and step 100 confirm the new runtime dataflow:

```text
step50:
  owm_proposal_tokens = 31.02
  aqr_proposal_shape_quality_mean = 0.8704
  aqr_task_owner_proposal_selected_count = 3.895
  aqr_task_owner_proposal_score_entropy = 0.3483
  aqr_task_owner_point_bridge_max = 0.0251
  aqr_active_same_role_support_overlap_max = 0.0207

step100:
  owm_proposal_tokens = 31.27
  aqr_proposal_shape_quality_mean = 0.8745
  aqr_task_owner_proposal_selected_count = 3.815
  aqr_task_owner_proposal_score_entropy = 0.3390
  aqr_task_owner_point_bridge_max = 0.0351
  aqr_active_same_role_support_overlap_max = 0.0975
```

Overlay inspection:

```text
step_000100__pick_up_the_red_block_lying_in_the_drawer__sam_proposals.png
step_000100__pick_up_the_red_block_lying_in_the_drawer__active_only.png
step_000100__pick_up_the_red_block_lying_in_the_drawer__with_gray.png
```

The repair is therefore verified as a calibration/dataflow fix, not as complete
semantic owner resolution.  The red block appears in the scene, but active
anchors still concentrate mostly around the gripper/drawer region at step100.
That is the expected limit of blind automatic masks: they can propose object-
like regions, but they do not know which region is the task object.

Follow-up full-sidecar diagnostic:

```text
picf_a7_full_sidecar_anchoronly_diag300_20260517
```

This run combined repaired proposal sidecars and tracklet sidecars while
freezing PaliGemma, V-JEPA, Sonata, and AnyTouch. It confirmed that both
sidecar paths can enter AQR:

```text
step100:
  owm_tracklet_tokens = 57.42
  owm_proposal_tokens = 1.44
  aqr_task_owner_proposal_selected_count = 0.170
  aqr_task_owner_point_bridge_max = 0.00069
  aqr_same_role_support_overlap_max = 0.4477

step150:
  owm_tracklet_tokens = 57.76
  owm_proposal_tokens = 1.44
  aqr_task_owner_proposal_selected_count = 0.130
  aqr_task_owner_point_bridge_max = 0.00091
  aqr_same_role_support_overlap_max = 0.9617
```

The important negative result is that blind SAM proposal evidence remained too
weak for task ownership and raw overlap worsened.  Therefore production defaults
now disable blind proposal memory and all proposal-to-point/task-owner proposal
weights:

```text
proposal_memory_enabled = False
proposal_read_weight = 0.0
proposal_point_bridge_weight = 0.0
task_owner_proposal_bias_weight = 0.0
task_owner_proposal_point_bias_weight = 0.0
task_owner_proposal_point_bridge_weight = 0.0
```

The proposal code path remains available for explicit ablations and for future
prompted/reranked proposal sidecars.  The sidecar mechanism itself is retained
because tracklets and future prompted masks are external evidence transport, not
the source of the blind-SAM noise.

## Next Full Sidecar Upgrade

The coherent next step is prompted/reranked proposal generation:

```text
language/PG task prior:
  target nouns and referring phrases, for example "red block".

action/gripper trajectory:
  current/future approach endpoint as offline weak prompt evidence, never as
  current-policy input leakage.

tactile/contact:
  calibrated fingertip/contact point when available.

tracklet continuity:
  repeated visible points/masks across frames.

SAM/SAM2:
  point/box-prompted masks, or automatic masks reranked by the seed evidence
  above.
```

This should write source-aware `proposal_*` sidecars.  Runtime AQR should keep
the current rule: proposals are weak typed evidence; no proposal is hard
posterior truth.

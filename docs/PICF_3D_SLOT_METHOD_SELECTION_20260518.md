# PICF 3D Slot Method Selection - 2026-05-18

Status: current method-selection ledger for replacing failed heuristic
motion-owner previews with a paper-backed 3D object-centric slot route.

This file is intentionally narrow.  It does not promote SAM boxes, optical
flow colors, or hand-written motion-owner clustering as the core PICF object
binding method.

## 1. CALVIN Reality Check

CALVIN is not a SlotLifter-style many-view NeRF setting.

```text
Available in our PICF/CALVIN path:
  static RGB-D
  gripper RGB-D when present
  robot_obs / proprio
  actions
  language
  camera calibration for static and gripper-depth projection into a world frame

Not available as a default SlotLifter assumption:
  many posed source views of one static scene
  hundreds of views per scene
  NeRF-style scene-level reconstruction target
  persistent multi-view static-scene supervision
```

The local code confirms this boundary:

```text
src/openpi/picf/pointcloud_picf.py:
  builds static and optional gripper point clouds from depth and camera
  calibration.  Gripper camera can be resolved with robot_obs when `E_T_C`
  is present.

src/openpi/picf/paligemma/wrapper.py:
  can feed static and gripper images as VLM views.

src/openpi/picf/core/pipeline.py:
  can consume typed visual, point, tactile, tracklet, and proposal memories.
```

Therefore the correct slot target is not "run SlotLifter as-is".  The correct
target is:

```text
few-view RGB-D / point-cloud object-slot lifting for robot manipulation,
with optional SlotLifter-derived point-slot math.
```

## 2. What Was Deleted

The previous self-written motion-owner prototype is removed and must not be
treated as a reusable baseline.

```text
deleted local:
  scripts/picf_motion_owner_preview.py
  docs/PICF_MOTION_OWNER_POINTCLOUD_PREVIEW_TEMP.md

deleted remote:
  /mnt/picf_motion_owner_previews

removed as primary reference code:
  TAPNet/TAPIR clone
  CoTracker clone
  SAM2 clone
```

Reason:

```text
Those tools can be useful sidecars, but they are not native 3D object-centric
slot methods.  The attempted preview also produced wrong background/object
coloring, so preserving it would create false confidence.
```

## 3. Candidate Selection Criteria

A candidate is acceptable only if it satisfies most of these constraints:

```text
1. Native 3D / RGB-D / point-cloud compatibility.
2. Object-centric slot or object-file representation.
3. Does not require dense object mask labels as a hard assumption.
4. Produces point/ray/token-to-object assignment, not only image boxes.
5. Can be used as typed evidence under posterior authority.
6. Does not prune dense V-JEPA tokens or corrupt V-JEPA world-model structure.
7. Can be confidence-gated and later faded if learned binding becomes stable.
8. Has paper/code evidence rather than being an ad-hoc heuristic.
```

## 4. Paper-Code Candidates

| Candidate | Fit For CALVIN | Code Status | Verdict |
|---|---:|---:|---|
| SlotLifter | medium | official code cloned on A7 | Use as mathematical reference for point-slot lifting, not as direct training recipe. |
| 3D block-slot / robot manipulation slot papers | high conceptually | code uncertain | Use as architecture reference if code is unavailable. |
| STORM / task-aware robotic slots | medium-high | code uncertain | Use as robotics slot objective reference if available. |
| VideoSAUR / SAVi / OCLF | medium | cloned | Use for slot attention mechanics and video grouping only; not point-cloud native. |
| SAM / SAM2 | low as core | removed as primary | Proposal sidecar only, default off. |
| TAPIR / CoTracker | low as core | removed as primary | Tracklet sidecar only, not object slot solver. |

## 5. SlotLifter Dataflow We Should Reuse

SlotLifter's official code implements the important mechanism we want:

```text
images + source cameras
 -> multi-view image encoder
 -> per-view feature maps
 -> sample 3D points along rays
 -> project each 3D point into source views
 -> aggregate multi-view features at each sampled point
 -> SlotAttention over image features to form slots
 -> JointDecoder maps every 3D sampled point to empty slot + object slots
 -> render/aggregate point-slot weights as object-centric 3D evidence
```

The key code facts from the cloned reference:

```text
model/slot_attn.py:
  SlotAttention normalizes features, projects q/k/v, softmaxes over slots,
  and updates slots with GRU + MLP.

model/slot_attn.py:
  JointDecoder appends an empty/background slot and computes point-slot
  mapping w = softmax(q(point)^T k(slot)).

model/renderer.py:
  renderer samples 3D points along rays, computes point features, then obtains
  point-slot weights before rendering instance maps.

model/nerf.py + model/projection.py:
  sampled 3D points are projected into source cameras and fused from
  multi-view image features.
```

This is the right mathematical object:

```math
p(k \mid x_i, z_i)
=
\operatorname{softmax}_k(
  \phi_p(x_i, z_i)^T \phi_s(s_k)
)
```

where `x_i` is a 3D point/ray sample, `z_i` is its lifted feature, and `s_k`
is an object slot.

## 6. Why SlotLifter Is Not Directly Deployed

Direct deployment would be wrong for CALVIN because:

```text
1. It assumes many posed views of a scene; CALVIN has static + wrist RGB-D.
2. It optimizes radiance-field reconstruction; PICF optimizes manipulation
   belief state and action.
3. It samples many rays and 3D points per scene; doing this online would be
   too slow for the current PICF training loop.
4. Its slots are scene/object reconstruction slots; PICF needs posterior
   object files under task, contact, and action pressure.
```

So the correct transfer is structural, not literal:

```text
Reuse:
  point/ray -> lifted feature -> slot assignment
  empty/background slot
  competition over object slots
  point-slot weights as object evidence

Do not reuse blindly:
  full NeRF rendering objective
  many-view training assumption
  scene-level reconstruction loop
```

## 7. Recommended PICF Module

Name:

```text
PICF Object3D Slot Lifter
```

Inputs:

```text
static point cloud:
  xyz, rgb, normals, view_id=static

gripper point cloud:
  xyz, rgb, normals, view_id=gripper, if calibrated and present

optional dense features:
  V-JEPA projected/static tokens
  PaliGemma image-language support
  tactile/contact packet
  action/proprio context
```

Core computation:

```math
z_i =
W[
  xyz_i,\ rgb_i,\ n_i,\ view_i,\ \phi_{vjepa}(i),\ \phi_{pg}(i)
]
```

```math
s_k^{(0)} =
\operatorname{SlotInit}(task,\ proprio,\ contact,\ learned\_seeds)
```

```math
a_{k,i}^{(r)}
=
\operatorname{softmax}_k
\left(
  q(s_k^{(r)})^T k(z_i)
  + b_{\text{geom}}(s_k, x_i)
  + b_{\text{contact}}(k,i)
  + b_{\text{view}}(k,i)
\right)
```

```math
s_k^{(r+1)}
=
\operatorname{GRU}
\left(
  s_k^{(r)},
  \sum_i
  \frac{a_{k,i}^{(r)}}{\sum_j a_{k,j}^{(r)}+\epsilon}
  v(z_i)
\right)
```

Output:

```text
object3d_slot.tokens:
  object slot embeddings

object3d_slot.point_priors:
  [slot, point] assignment weights

object3d_slot.centers:
  weighted 3D centroids

object3d_slot.covariances:
  weighted spatial uncertainty

object3d_slot.objectness:
  non-background support mass / entropy / contact-likelihood score

object3d_slot.background:
  explicit empty/background slot assignment
```

PICF integration:

```text
object3d_slot becomes typed evidence M_object3d.
It can bias anchor_x, point_priors, support signatures, and posterior birth.
It must not hard-overwrite posterior.
```

## 8. AnyTouch And Sonata Handling

AnyTouch should not be assigned to objects by wrist image alone.

Correct rule:

```text
If tactile sensor pose/contact point is available:
  assign tactile evidence to nearest compatible 3D object slot using the
  tactile sensor/contact coordinate in world space.

If only approximate finger/sensor frame is available:
  treat it as a noisy contact prior with high covariance.

If no contact geometry is available:
  keep tactile as typed global/contact evidence, not an object label.
```

Sonata / point-cloud features:

```text
Use the same object3d_slot.point_priors to aggregate point features per object.
Do not force AnyTouch or Sonata to invent object identity independently.
The 3D object slot assignment is the shared alignment layer.
```

## 9. Background And Gray Slots

The system must keep background evidence without letting background steal the
task object.

The correct decomposition is:

```text
object slots:
  compete for compact, supported, task/contact-relevant object evidence.

background/empty slot:
  absorbs walls, table, robot fixtures, and unexplained points.

context slots:
  can summarize useful static scene context but enter action with low weight.
```

This is directly aligned with SlotLifter's explicit empty slot in
`JointDecoder`.  It is more principled than drawing all unused anchors as gray
objects.

## 10. Why This Is Not A Patch

The previous failure was:

```text
fixed overcomplete learned anchors
 -> repeated candidates
 -> weak or noisy owner evidence
 -> anchors fail to bind the task object
```

The proposed repair changes the measurement model:

```text
raw point/ray evidence first forms object-centric 3D assignments,
then PICF anchors/posterior read those object assignments.
```

That is not a post-hoc penalty.  It changes the latent variable structure from:

```math
p(anchor_j \mid token_i)
```

to:

```math
p(object_k \mid point_i),\quad
p(posterior\ file_j \mid object_k, task, contact, history)
```

This matches the belief-state principle:

```text
separate measurement grouping from posterior file identity.
```

## 11. Deployment Plan

Phase A: paper-code audit complete

```text
SlotLifter cloned:
  /mnt/picf_slot_reference_code/SlotLifter

Rejected heuristics cleaned:
  motion-owner preview removed
```

Phase B: local PICF contract

```text
Add Object3DSlotSupportState.
Add object3d slot sidecar fields to PicfObservation only after the generator
contract is clear.
Add strict verifier checks:
  dense V-JEPA tokens are preserved
  object3d evidence is typed support only
  background slot exists
  no SAM hard labels
  no optical-flow hard labels
  no posterior overwrite
```

Phase C: generator

```text
Implement a minimal SlotLifter-derived 3D point-slot module:
  input point cloud + projected features
  output point-slot priors and centroids

Start as offline sidecar over CALVIN frames/windows.
Use confidence gates and previews before training.
```

Phase D: training integration

```text
Use object3d priors as:
  point-prior bias
  posterior birth candidate
  support signature source
  tactile/contact assignment bridge

Keep it weak at first:
  no hard object labels
  no direct action overwrite
  fade scaffold if learned PICF binding stabilizes
```

## 12. Current Conclusion

For CALVIN, the most suitable next method is:

```text
SlotLifter-inspired few-view RGB-D point-slot lifting,
not SlotLifter as-is and not SAM/flow tracking as core.
```

This is the closest paper-backed route to the user's requirement:

```text
latest/advanced slot-style method,
native 3D or point-cloud compatible,
mathematically aligned with PICF posterior belief state,
not a hand-written heuristic.
```

## 13. 2026-05-18 Input-Level Deployment Example

An input-level example has been generated on A7:

```text
/mnt/picf_object3d_slot_examples/20260518_calvin_rgbd_input
```

Selected frame:

```text
episode_1572904
language: pick up the red block lying in the drawer
```

Artifacts:

```text
rgb_static.png
depth_static.png
rgb_gripper.png
depth_gripper.png
pointcloud_rgb_views.png
pointcloud_view_views.png
object3d_slot_input_sample.npz
report.json
```

The generated input package contains:

```text
static RGB-D points: 2500
gripper RGB-D points: 1764
merged points: 4264
fields:
  xyz_world
  xyz_norm
  rgb
  view_ids
  center
  scale
  frame_id
  text
```

Interpretation:

```text
This is a dataflow and geometry example for the proposed Object3D slot lifter.
It does not claim object segmentation and does not produce fake slot labels.
```

The example verifies the minimum viable CALVIN input contract:

```text
1. static RGB-D can be lifted to world points.
2. gripper RGB-D can be lifted through robot_obs and E_T_C.
3. the two views can be merged into one point-token table with view_ids.
4. the result can be normalized for a future point-slot module without
   deleting dense RGB-D evidence.
```

The input example verified the point-token geometry only.  It did not verify
object grouping.

## 14. 2026-05-18 SlotLifter-Style Diagnostic Video

A first SlotLifter-style point-slot module has been implemented locally:

```text
src/openpi/picf/object3d_slot_lifter.py
src/openpi/picf/object3d_slot_lifter_test.py
scripts/picf_object3d_slot_video.py
```

Local checks:

```text
python -m py_compile src/openpi/picf/object3d_slot_lifter.py scripts/picf_object3d_slot_video.py
uv run pytest -q src/openpi/picf/object3d_slot_lifter_test.py
  2 passed
```

Remote diagnostic artifacts on A7:

```text
/mnt/picf_object3d_slot_examples/20260518_object3d_slot_video_redblock_v2

object3d_slot_empty_aware.gif
object3d_slot_object_only.gif
frame_000_empty_aware_panel.png
frame_000_object_only_panel.png
frame_016_empty_aware_panel.png
frame_016_object_only_panel.png
frame_031_empty_aware_panel.png
frame_031_object_only_panel.png
report.json
```

The diagnostic used:

```text
task: pick up the red block lying in the drawer
frames: episode_1572904..1572935
slots: 8 object slots + explicit empty/background slot
feature input: normalized xyz + RGB + view_id + radius
diagnostic optimization: short reconstruction overfit only
```

Observed result:

```text
recon_loss:
  0.3340 -> about 0.0491

background_mean_final:
  about 0.929

objectness_mean:
  about 0.0065 per object slot
```

Image inspection:

```text
empty-aware view:
  most static points are assigned to gray/background.

object-only SlotAttention view:
  most static scene points collapse into one dominant blue object slot.
  the red block receives only weak/local colored points, not a reliable object
  slot.
```

Conclusion:

```text
This first point-slot module is a valid code/dataflow diagnostic, but it is not
a deployable object segmentation solver.  It shows that a SlotLifter-style
empty slot plus reconstruction-only objective is under-constrained in CALVIN:
the empty/background branch can explain most points, and object slots do not
automatically bind task objects.
```

Mathematical reason:

```math
\min_{\theta}
\sum_i
\left\|
  \hat z_i(\sum_k p_\theta(k \mid x_i,z_i)s_k)
  -
  z_i
\right\|^2
```

without a radiance-field rendering constraint, multi-view consistency over many
views, mask/objectness supervision, or task/contact-conditioned object evidence
has a degenerate solution:

```math
p(empty \mid x_i,z_i) \approx 1
```

or a single broad object slot explains most points.  That degeneracy is exactly
what the diagnostic video shows.

Therefore, the next acceptable version must not simply add this module to PICF.
It must add at least one of the following principled grouping constraints:

```text
1. task/contact-conditioned object proposal from language and gripper/contact;
2. temporal 3D correspondence / tracklet consistency over RGB-D points;
3. multi-view consistency where real camera coverage supports it;
4. offline object proposal sidecar with strict confidence filtering;
5. a production objective that prevents empty/background from explaining all
   task-relevant points.
```

Current status:

```text
Object3D slot code exists for diagnostics.
It is not wired into AQR/PICF training.
It must remain off by default until the grouping constraint above is added and
passes visual inspection.
```

## 15. 2026-05-18 Foreground-Enhanced Diagnostic Repair

The first diagnostic proved that reconstruction-only point slots collapse.  The
next diagnostic therefore added the minimal missing constraint that is
mathematically consistent with the robotics setting:

```text
task/contact/color foreground salience is used only to weight the diagnostic
objective and visualization.
```

It does not create labels and does not wire Object3D slots into training.  It
implements the same high-level principle as object-centric rendering papers:
background/empty capacity must not explain all task-relevant evidence.

Code:

```text
scripts/picf_object3d_slot_video.py
```

Remote artifacts:

```text
/mnt/picf_object3d_slot_examples/20260518_object3d_slot_video_redblock_v4

object3d_slot_empty_aware.gif
object3d_slot_object_only.gif
object3d_slot_foreground_enhanced.gif
frame_000_foreground_enhanced_panel.png
frame_016_foreground_enhanced_panel.png
frame_031_foreground_enhanced_panel.png
report.json
```

The stricter v5 foreground visualization is now preferred for inspection:

```text
/mnt/picf_object3d_slot_examples/20260518_object3d_slot_video_redblock_v5

object3d_slot_empty_aware.gif
object3d_slot_object_only.gif
object3d_slot_foreground_enhanced.gif
frame_000_foreground_enhanced_panel.png
frame_016_foreground_enhanced_panel.png
frame_031_foreground_enhanced_panel.png
report.json
```

2026-05-18 v6 adds the clearest human-inspection video:

```text
/mnt/picf_object3d_slot_examples/20260518_object3d_slot_video_redblock_v6

object3d_slot_empty_aware.gif
object3d_slot_foreground_enhanced.gif
object3d_slot_background_faded_objects_strong.gif
frame_000_background_faded_objects_strong_panel.png
frame_016_background_faded_objects_strong_panel.png
frame_031_background_faded_objects_strong_panel.png
report.json
```

Use `object3d_slot_background_faded_objects_strong.gif` first when visually
checking whether the task object is visible: the RGB background is desaturated
and faded, while foreground slot points are drawn with stronger colors.  This is
closer to SlotLifter's rendered instance-map inspection mode than the raw point
overlay, but remains a diagnostic visualization rather than supervision.

Observed metrics:

```text
v4 background_mean_final:
  about 1.37e-05

v5 background_mean_final:
  about 6.7e-07

v5 foreground_mean:
  about 0.105

v5 objectness_mean:
  two dominant slots near 0.59 and 0.33, one small slot near 0.03,
  remaining slots near zero

v5 last diagnostic loss:
  about 0.0383
```

Visual result:

```text
foreground-enhanced view:
  v4 suppressed empty collapse but still colored too much table/drawer wood.
  v5 uses stricter chroma/saturation gates and a display threshold, so the
  background is substantially faded and the red block / gripper-contact region
  is much easier to inspect.

background-faded object view:
  v6 is the clearest diagnostic view.  It fades background RGB and strengthens
  high-foreground slot colors.  It is meant for human inspection of object
  visibility and slot-color consistency.

remaining problem:
  this is still not a robust object-slot solver.  The highlighted evidence is
  a foreground diagnostic prior, not learned instance identity.  It cannot be
  promoted to production training supervision without a stronger task/contact/
  temporal grouping contract.
```

Strict conclusion:

```text
The repair fixes the visualization and the empty-background diagnostic
collapse.  It does not fully solve object binding.  The next real solver must
combine point-slot competition with at least one reliable grouping source:
contact trajectory, dense temporal correspondence, or validated object
proposal evidence.
```

Remaining before true deployment:

```text
1. add task/contact/temporal constraints to point-slot assignment;
2. generate assignment previews that clearly bind task objects, not just
   background;
3. add strict tests that object3d_slot is typed evidence only and cannot
   overwrite posterior;
4. only then wire it into AQR point/support priors.
```

## 12. CALVIN Segment / Completion-Aware Slicing

The multi-task preview exposed a real diagnostic error: for button/drawer
segments, the fixed middle frame of `auto_lang_ann.npy` can miss the actual
press/pull event.  That makes a correct object-slot method look wrong because
the inspected frame is not the frame where the task object is being contacted
or changed.

Official CALVIN evaluation does not judge success from the middle of a
language interval.  In the official evaluator, a task oracle compares the
environment/task state before and after rollout; the relevant call path is:

```text
calvin_agent/evaluation/evaluate_policy.py:
  start_info = env.get_info()
  ...
  current_info = env.get_info()
  current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
```

Therefore the correct offline inspection rule is:

```text
language interval:
  weak segment boundary only

scene_obs delta:
  completion/state-change candidate when available

end-effector/contact/foreground score:
  contact candidate and best visual-inspection frame

fixed midpoint:
  never sufficient for button/drawer/switch diagnosis
```

The maintained diagnostic script is:

```bash
python scripts/picf_calvin_segment_preview.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --output-dir /mnt/picf_object3d_slot_examples/20260518_segment_previews_v1 \
  --prompts "red block" "blue block" "pink block" drawer button switch \
  --max-frames 80
```

It saves, per task:

```text
segment_strip.png
000_start_frame_*.png
... q25/mid/q75/end ...
*_best_contact_frame_*.png
*_best_scene_delta_frame_*.png
segment_report.json
```

The next Object3D SlotLifter diagnostic must be run on `best_contact` or
`best_scene_delta` candidates, not blindly on the middle frame.  If the preview
shows the task was not actually attempted in the chosen interval, the interval
must be rejected before judging the object-slot result.

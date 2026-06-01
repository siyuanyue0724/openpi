# PICF-AQR-OWM Scalability And CALVIN Repair Plan - 2026-05-23

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/picf_aqr_owm_202605/README.md
```

This note records the current token-budget decision, the extensibility contract
for future full fine-tuning, and the next CALVIN repair gates.  It is not a new
architecture proposal by itself.

## 1. Token Budget Decision

Production/default:

```text
semantic_max_length = 256
```

Parity probe:

```text
semantic_max_length = 200
```

Why:

```text
PI0.5 CALVIN reference configs use max_token_len=200.
PICF's maintained PaliGemma wrapper default is 256.
```

The 2026-05-23 A7 probe
`picf_a7_pi05tok200_frozen_policy_100_20260523` confirmed that `200` is safe
but not a material runtime fix:

```text
step  steps/s   total   obj_pull  active_ov  downstream_ov  raw_ov
25    0.0802    0.160   0.335     0.0077     0.0281         0.470
50    0.0810    0.131   0.280     0.0038     0.0278         0.432
75    0.0774    0.123   0.226     0.0098     0.0542         0.455
100   0.0819    0.213   0.454     0.0053     0.1038         0.572
```

Conclusion:

```text
Use 256 for maintained production and large-data fine-tuning.
Use 200 only for strict PI0.5 parity, memory pressure, or controlled ablation.
Do not claim 200 solves runtime; the measured bottleneck is not prompt length.
```

## 2. Runtime Bottleneck Boundary

The current serving/training timing work shows the main runtime pressure is:

```text
1. V-JEPA visual_maps / multiview temporal encoding
2. dense typed memory / token_field construction
3. point/tracklet/proposal routing
4. PI0.5 semantic/action path
```

Therefore the scalable optimization order is:

```text
1. deterministic frozen V-JEPA feature cache
2. optional deterministic Sonata feature cache if profiling says it matters
3. optional deterministic AnyTouch feature cache if tactile becomes visible
4. low-frequency PICF belief update for deployment, not for training truth
5. gated dense-context injection for background/peripheral evidence
```

Do **not** optimize by deleting dense evidence or forcing background into
object slots.  That would improve speed by weakening the reason PICF exists.

## 3. Full Fine-Tuning Scalability Contract

The maintained large-run contract is:

```text
frozen:
  V-JEPA pretrained visual encoder
  Sonata pretrained point encoder
  AnyTouch pretrained tactile encoder

trainable:
  PaliGemma / PI0.5 semantic-action stack
  PICF AQR / posterior / owner / context adapters
  action-side heads
```

Future full fine-tuning is allowed only as a staged experiment:

```text
stage 0:
  frozen-pretrain production baseline, semantic_max_length=256

stage 1:
  unfreeze PaliGemma and PICF adapters only

stage 2:
  optionally unfreeze late V-JEPA adapter/projection layers, not the frozen
  V-JEPA backbone, unless a fresh compute budget and eval gate are assigned

stage 3:
  only after CALVIN/video success, test selective Sonata/AnyTouch adapters
```

Scaling rules:

```text
If a backbone is trainable, its frozen feature cache is invalid.
If PaliGemma is trainable, do not cache semantic embeddings.
If sidecar masks are noisy, keep them weak and quality-gated; never promote
them to hard object truth.
If action loss plateaus, adjust weak object scaffold pressure before increasing
action above the traditional 2.0 scale.
```

## 4. Dense Context / Background Extensibility

Current rule:

```text
active object rows:
  high-priority object/control evidence

context rows:
  low-priority structured scene evidence with attention bias, not zeroed tokens

reserve/dustbin rows:
  lifecycle capacity and unexplained evidence, not ordinary object files

dense V-JEPA/point/temporal tokens:
  retained as typed memory; not pruned by sidecar masks
```

Future enhancement:

```math
c_t = \operatorname{GatedCrossAttn}(q_{\text{control}}, Z_{\text{dense}})
```

where `Z_dense` contains residual V-JEPA/static/wrist/temporal context and the
gate is conditioned by instruction, active object files, and posterior state.
This is the clean route to PI0.5-style action speed/quality without treating
background as object slots.

This should be a separate experiment because it changes the action-prefix
interface.  It should not be mixed into token-budget or CALVIN-eval debugging.

## 5. CALVIN Repair Queue

Known issues from the current CALVIN/eval attempts:

```text
1. Inference is much slower than PI0.5.
   Root cause: PICF observe_step, especially V-JEPA visual_maps and dense typed
   memory.  200-token prompt budget does not address this.

2. Early CALVIN rollouts did not produce success.
   Root cause not yet isolated: could be action plateau, slow control loop,
   checkpoint quality, residual nonfinite actions, or task-object binding errors.

3. Nonfinite action rows were observed in earlier eval logs.
   Root cause must be checked in the safe server/eval path before declaring a
   checkpoint behavior-failed.

4. Overlay correctness remains a behavior gate.
   Active owner must land on the intended task/contact object, not just any
   changed object.
```

Next repair sequence:

```text
Gate A:
  run a short safe CALVIN eval from the best current checkpoint with timing
  breakdown enabled for only a small number of steps.

Gate B:
  inspect nonfinite action count, per-step inference time, and first 20
  rollout outcomes.

Gate C:
  if inference is the blocker, implement V-JEPA frozen feature cache first.

Gate D:
  if action quality is the blocker, compare current action_default_equiv
  against 4-22 ablation and run a controlled action-weight/scaffold-floor gate.

Gate E:
  if owner binding is the blocker, use anchor overlays plus sidecar/mask_active
  views; do not revive blind SAM.
```

## 6. Maintained Commands

Token-budget probe:

```bash
scripts/experiments/picf_aqr_owm_202605_active/run_a7_pi05_token_budget_frozen_policy_100_20260523.sh
```

Maintained frozen-policy slot validation:

```bash
scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_qualitytarget_frozen_policy_300_20260521.sh
```

Maintained action-aware long-run family:

```bash
scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_defaultsync_long30k_ckpt500_20260521.sh
scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_qgdecay_from500_long30k_20260522.sh
```

CALVIN inference latency gate:

```bash
scripts/experiments/picf_aqr_owm_202605_active/run_a7_calvin_inference_latency_gate_20260523.sh
```

## 7. CALVIN Inference Latency Gate - 2026-05-23

Question:

```text
Can PICF inference run close enough to PI0.5 that deployment remains practical?
```

Target interpretation:

```text
strong target:
  PICF median policy latency <= 2.0x PI0.5-only ablated median latency

weak target:
  PICF median policy latency <= 3.0x PI0.5-only ablated median latency
```

This means PICF runs at roughly one-half to one-third of the PI0.5-only speed.
It does not mean PICF must match PI0.5 latency exactly.

Measurement contract:

```text
same websocket server path
same CALVIN evaluator wrapper
same action safety sanitizer
same cloud machine
debug overlays disabled
prediction debug disabled
OPENPI_PICF_TIMING_BREAKDOWN=1
```

Current A7 launcher:

```bash
cd /root/openpi_slot_quality_ea2c5f2
bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_calvin_inference_latency_gate_20260523.sh
```

Optional deployment-speed profile:

```bash
PICF_OBSERVE_INTERVAL=4 \
bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_calvin_inference_latency_gate_20260523.sh
```

`PICF_OBSERVE_INTERVAL > 1` is inference-only.  It keeps dense V-JEPA/point/
tactile tokens intact on PICF update steps, but reuses the last PICF control
prefix on intermediate control steps while still sampling a fresh PI0.5 action
from the current semantic observation.  This follows the deployment principle
that a belief router can run at lower frequency than the motor control loop.
It is not a training shortcut and should be evaluated separately from the
every-step PICF correctness profile.

Default checkpoints:

```text
PI0.5-only baseline:
  /mnt/checkpoints/picf_core/picf_core/picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2/20000

PICF enabled:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_qgfloor003_from1500_long30k_20260522/3000
```

Output:

```text
/mnt/checkpoints/picf_core/eval/picf_inference_latency_gate_20260523/
  pi05_ablated/action_safety.jsonl
  pi05_ablated/timing_summary.json
  picf_enabled/action_safety.jsonl
  picf_enabled/timing_summary.json
  latency_ratio.json
```

The first timing keys to read are:

```text
infer_ms
policy.checkpoint_policy_total_ms
policy.policy_policy_act_total_ms
policy.policy_semantic_encode_ms
policy.policy_picf_observe_ms
policy.policy_action_sample_ms
policy.policy_picf_finalize_ms
policy.policy_picf_observe_visual_maps_ms
policy.policy_picf_observe_point_features_ms
policy.policy_picf_observe_token_field_ms
policy.policy_picf_observe_anchor_graph_ms
policy.policy_picf_observe_object_explanation_ms
policy.policy_picf_observe_posterior_update_ms
```

Interpretation:

```text
If policy_action_sample dominates:
  the remaining bottleneck is PI0.5/PaliGemma action sampling, not PICF.

If policy_picf_observe_visual_maps dominates:
  optimize frozen V-JEPA inference first; online deployment needs incremental
  or lower-frequency visual-map updates rather than a dataset feature cache.

If point_features dominates:
  profile Sonata/point feature construction and consider a deterministic
  frozen point-feature cache or lower-frequency point updates.

If token_field / anchor_graph / object_explanation dominates:
  optimize dense typed-memory routing, not the semantic token budget.
```

Rejected shortcuts:

```text
do not disable dense evidence just to pass latency
do not force background into object slots
do not run latency with anchor/prediction debug enabled
do not claim training-side V-JEPA dataset cache directly solves online inference
```

The only acceptable deployment optimization is to preserve the belief-state
contract while reducing redundant frozen-backbone computation or amortizing
updates across control steps.

Initial A7 result with every-step PICF observation:

```text
run:
  picf_inference_latency_gate_20260523_r3

PI0.5-only ablated baseline:
  records=302
  infer_ms p50=268.22
  policy_action_sample_ms p50=216.92
  policy_semantic_encode_ms p50=49.95

PICF enabled, observe_interval=1, partial first rollout:
  records=35
  infer_ms p50=1570.22
  policy_picf_observe_ms p50=1266.15
  policy_picf_observe_visual_maps_ms p50=757.65
  policy_picf_observe_token_field_ms p50=251.24
  policy_action_sample_ms p50=228.26

ratio:
  1570.22 / 268.22 = 5.85x slower
```

Conclusion:

```text
Every-step PICF observation is not deployable at the requested 1/2-1/3 PI0.5
speed target on this checkpoint/profile.
```

The bottleneck is not PI0.5 action sampling; it is PICF observation,
especially V-JEPA visual map encoding plus dense token-field construction.
Therefore the next valid deployment profile is `PICF_OBSERVE_INTERVAL=4`, not
another prompt-token or action-weight change.

Initial A7 result with `PICF_OBSERVE_INTERVAL=4`:

```text
run:
  picf_inference_latency_gate_20260523_r4_observe4

PI0.5-only ablated baseline:
  records=551
  infer_ms mean=344.26
  infer_ms p50=373.34
  infer_ms p95=442.11
  policy_action_sample_ms p50=211.42

PICF enabled, observe_interval=4, completed first rollout:
  records=360
  infer_ms mean=648.55
  infer_ms p50=360.85
  infer_ms p95=1684.48
  policy_picf_observe_reused mean=0.75
  policy_picf_observe_ms p50=0.00
  policy_picf_observe_visual_maps_ms p50=785.82 on update steps
  policy_picf_observe_token_field_ms p50=241.64 on update steps
  policy_action_sample_ms p50=223.23

ratio:
  mean 648.55 / 344.26 = 1.88x slower
  p50 360.85 / 373.34 = 0.97x
  p95 1684.48 / 442.11 = 3.81x
```

Reading:

```text
Median and mean meet the requested one-half/one-third PI0.5 speed target.
The p95 still has a heavy every-fourth-step spike from full PICF observation.
```

Behavior caveat:

```text
This r4 run is a latency gate, not a CALVIN acceptance pass.

The PI0.5-only baseline completed the first two subtasks in this one-sequence
probe before failing a later subtask.  The PICF-enabled checkpoint failed the
first open_drawer rollout, and eval.log reported non-finite raw action values
after step 344 that were caught by the action sanitizer.

Therefore the speed profile is usable as a deployment-latency direction, but
the checkpoint itself is not behavior-accepted by this gate.
```

Deployment conclusion:

```text
PICF can be made practical on average by amortizing belief updates.
It is not yet hard-real-time smooth.  If p95 matters, the next speed work is
async PICF observation or incremental V-JEPA visual-map updates, not deleting
object slots or reducing semantic_max_length.
```

Feature-cache TODO:

```text
docs/PICF_AQR_OWM_FROZEN_FEATURE_CACHE_TODO_20260522.md
```

CALVIN validation guide:

```text
docs/CALVIN_VALIDATION_README.md
```

## 8. CALVIN Action Video Gate - 2000/3000 Checkpoints - 2026-05-23

Question:

```text
Do the current action-aware PICF checkpoints produce plausible CALVIN actions,
not just acceptable latency?
```

Launcher:

```bash
cd /root/openpi_slot_quality_ea2c5f2
NUM_SEQUENCES=1 \
PICF_OBSERVE_INTERVAL=1 \
SAVE_ANCHORS=0 \
bash scripts/experiments/picf_aqr_owm_202605_active/run_a7_calvin_ckpt2000_3000_video_eval_20260523.sh
```

Important measurement choice:

```text
PICF_OBSERVE_INTERVAL=1
```

This is intentional.  The run tests checkpoint behavior with full PICF observe
on every control step.  It should not be confused with the deployment-speed
profile from Section 7, where observe is amortized.

Anchor debug is disabled:

```text
SAVE_ANCHORS=0
```

Reason: anchor_debug JSONL became very large and blocked the action/video gate.
The correct artifacts for this gate are the CALVIN rollout video,
`action_safety.jsonl`, and `eval.log`.  Anchor videos should be generated in a
separate short diagnostic run.

Output:

```text
/mnt/checkpoints/picf_core/eval/picf_a7_actionaware_ckpt2000_3000_video_eval_20260523/
```

Checkpoint 2000 result:

```text
video:
  ckpt2000_observe1/videos/open_drawer_1779539417551.mp4

eval:
  open_drawer: 0 / 1
  average successful sequence length: 0.0

action_safety:
  records=360
  nonfinite_actions=146
  clip_changed=0
  first nonfinite action observed at step=214

latency:
  infer_ms_mean=1619.90
  infer_ms_p50=1590.27
```

Checkpoint 3000 result:

```text
video:
  ckpt3000_observe1/videos/open_drawer_1779540139772.mp4

eval:
  open_drawer: 0 / 1
  average successful sequence length: 0.0

action_safety:
  records=360
  nonfinite_actions=293
  clip_changed=0
  first nonfinite action observed at step=67

latency:
  infer_ms_mean=1416.91
  infer_ms_p50=1402.32
```

Conclusion:

```text
The 2000 and 3000 checkpoints are not CALVIN-behavior accepted.
Both fail the first open_drawer rollout and both produce nonfinite raw actions
that are only made safe by the action sanitizer.  The 3000 checkpoint is worse
than 2000 by nonfinite-action onset and count.
```

Immediate implication:

```text
Do not use the 3000 checkpoint as a behavior-quality reference just because its
latency profile was measurable.  The next CALVIN repair step should isolate the
source of nonfinite raw actions in the policy output path before launching a
longer success-rate run.
```

## 9. Inference Recurrent Safety Guard - 2026-05-23

Root-cause boundary:

```text
The failing CALVIN evals were finite for many control steps and then switched
abruptly to persistent nonfinite raw actions.  This is a closed-loop recurrent
pollution pattern, not a checkpoint-load, normalizer, video-export, or one-step
action-sampler-only failure.
```

Observed path:

```text
policy.act
  -> core.observe_step(previous state)
  -> observed.conditioned_control.pi_prefix_tokens
  -> semantic_encoder.sample_action_chunk(extra_prefix_tokens=prefix)
  -> core.finalize_with_action(action_future=action_chunk)
  -> server stores output.state as the next recurrent state
```

Accepted repair:

```text
src/openpi/picf/policy.py
```

The repair is inference-only:

```text
1. sanitize/clip PICF pi_prefix_tokens before PI0.5 action sampling;
2. if the sampled action_chunk is nonfinite, reuse the previous finite chunk
   with the same shape, or fall back to zeros if no finite chunk exists;
3. sanitize nonfinite recurrent state tensors before returning output.state,
   so the next observe_step is not seeded with NaN posterior/action state;
4. reset the finite-action fallback cache on reset_scaffold.
```

The defaults are intentionally conservative and environment-overridable:

```text
OPENPI_PICF_INFERENCE_PREFIX_VALUE_CLIP=50
OPENPI_PICF_INFERENCE_PREFIX_MAX_RMS=8
OPENPI_PICF_INFERENCE_STATE_VALUE_CLIP=10000
```

Mathematical interpretation:

```math
\tilde p_t =
\operatorname{clip}_{v,r}\left(\operatorname{nan\_to\_num}(p_t)\right)

\tilde a_t =
\begin{cases}
a_t, & a_t \in \mathbb{R}^{H\times d_a}\\
\tilde a_{t-1}, & a_t \notin \mathbb{R}^{H\times d_a}
                 \land \operatorname{shape}(\tilde a_{t-1})=\operatorname{shape}(a_t)\\
0, & \text{otherwise}
\end{cases}
```

This does **not** change the training loss, teacher forcing, or PICF belief
update objective.  It is the standard deployment-side recurrent-state guard:
closed-loop inference must be finite even when the learned recurrent state
temporarily leaves the training distribution.

Debug fields emitted by `policy.act`:

```text
inference_prefix_nonfinite
inference_prefix_value_clipped
inference_prefix_rms_clipped
inference_prefix_max_abs
inference_prefix_max_rms
inference_action_chunk_nonfinite
inference_action_chunk_fallback_last
inference_action_chunk_fallback_zero
inference_state_sanitized
inference_state_nonfinite_tensors
inference_state_clipped_tensors
```

Acceptance rule:

```text
A future CALVIN run may still fail behaviorally, but it should no longer enter
an irreversible NaN action state.  If the fallback counters fire often, that is
a model-quality failure to diagnose; it is not an excuse to hide the event.
```

## 10. Unroll / Burn-In Guidance After The NaN Gate

Do not try to fix the CALVIN NaN failure by making training unroll extremely
long.  That is the wrong control knob.

Reason:

```text
Longer full BPTT improves credit assignment over a short horizon, but it does
not guarantee finite closed-loop deployment over 360 CALVIN control steps.
The correct pattern is bounded recurrent state + truncated BPTT + no-grad long
rollout audits.
```

Maintained settings:

```text
fast large-run profile:
  unroll_steps=2
  burnin_steps=1
  burnin_mode=state_only

quality diagnostic profile:
  unroll_steps=3
  burnin_steps=1-2
  burnin_mode=state_only

upper bound for 2xA100-40GB without redesign:
  unroll_steps=3
```

Rejected default:

```text
Do not use unroll_steps=4/8 as the normal 2x40GB profile.
Prior local/cloud records already observed 4/8-step profiles hitting OOM or
becoming too slow.  If revisited, they need lower token/grid capacity or a
separate memory-budget experiment.
```

Inference/deployment audit:

```text
Keep training TBPTT short enough to be feasible, then run no-grad closed-loop
rollouts of 128-360 steps with action-safety logging.  This directly tests the
failure mode that produced CALVIN NaNs.
```

If using `PICF_OBSERVE_INTERVAL=4` for deployment-speed experiments, add a
future stale-prefix robustness augmentation only after the finite-state guard
passes.  Do not mix stale-prefix augmentation into the immediate NaN repair.

## 11. Verification

Local verification for this document/update:

```bash
bash -n \
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh \
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_pi05_token_budget_frozen_policy_100_20260523.sh

PYTHONPATH=src uv run python -m py_compile scripts/picf_core_train.py
python -m py_compile src/openpi/picf/policy.py src/openpi/picf/policy_test.py scripts/serve_picf_policy.py
PYTHONPATH=src uv run python -m pytest src/openpi/picf/policy_test.py -q
```

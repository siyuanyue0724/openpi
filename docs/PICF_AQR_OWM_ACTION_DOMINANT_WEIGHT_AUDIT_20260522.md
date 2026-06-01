# PICF-AQR-OWM Action-Dominant Weight Audit, Paper Survey, and Step-1500 Gate

Date: 2026-05-22

## Decision Summary

At the 1500-step gate, the active run has already reached the traditional PI0/PICF action scale:

```text
lambda_action_pos     = 2.0
lambda_action_rot     = 2.0
lambda_action_gripper = 2.0
```

Therefore, making action dominant should not be implemented by increasing the action lambdas above the legacy/default scale.  The mathematically cleaner intervention is to lower the weak object-scaffold floor after the owner/active-object metrics are healthy.

Current step1500:

```text
loss_action_default_equiv = 0.04164
loss_total_minus_action   = 0.02419
action fraction           = 63.3%
object_scaffold_scale     = 0.10
active_overlap            = 0.0500
downstream_overlap        = 0.0781
active_duplicate          = 0.0
```

If the scaffold-controlled part of the auxiliary budget is reduced from floor 0.10 to 0.03, the expected action fraction becomes approximately:

```text
A = 0.04164
B_fixed ~= 0.009-0.012
B_scaffold ~= 0.012-0.015 at floor 0.10
B_scaffold_new ~= 0.3 * B_scaffold
A / (A + B_fixed + B_scaffold_new) ~= 0.74-0.78
```

This matches modern VLA practice: action imitation / flow matching remains the dominant training signal, while object-centric or predictive terms are weak shaping signals unless they are the main pretraining objective.

## Maintained Run Change

Use this continuation from the step1500 checkpoint:

```text
SOURCE_EXP=picf_a7_actionaware_qgdecay_fast1300_from1000_long30k_20260522
SOURCE_STEP=1500
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

Do not increase `ACTION_LOSS_WEIGHT` above 2.0 unless the floor-0.03 branch still plateaus after 300-500 additional steps with active metrics healthy.

## Why Not Action Weight > 2.0 Now?

The action loss is already at the legacy/default-equivalent lambda scale.  Raising it above 2.0 would be a new non-traditional recipe.  It may still be useful as an ablation, but it is not the first production move because it changes action-gradient magnitude rather than simply letting action dominate by retiring the scaffold.

The model currently has healthy active-object metrics; the problem is not belief collapse.  The cleanest next test is reducing non-action scaffold pressure.

## Weight Mathematics and Deployment Rule

Use the following decomposition when deciding future action/scaffold weights:

```text
L_total
  = L_action
  + L_fixed_aux
  + s(t) * L_object_scaffold

where:
  L_action          = default-equivalent PI0/PICF action objective
  L_fixed_aux       = small non-decayed routing/quality/stability terms
  L_object_scaffold = weak sidecar/contact/object-owner shaping terms
  s(t)              = object_scaffold_decay_scale
```

The action fraction is diagnostic, not itself the optimized objective:

```text
rho_action(t) = L_action / L_total
```

For action-aware long runs, the healthy range is:

```text
early bootstrap:
  rho_action can be 25-50% while owner/object files are being stabilized.

post-bootstrap:
  rho_action should move toward 70-85% once active ownership metrics are
  healthy.

late training:
  rho_action may naturally fall back toward 60-75% if action loss drops faster
  than the bounded auxiliary floor.  That is not a failure by itself.
```

The important invariant is not "action fraction must always increase."  It is:

```text
action should continue to improve while non-action remains bounded and
active/downstream identity metrics stay healthy.
```

### Why `floor=0.03` Is the Current Maintained Setting

At step1500 of the fast-decay branch:

```text
L_action ~= 0.0416
L_total_minus_action ~= 0.0242
s(t) = 0.10
rho_action ~= 63.3%
```

Reducing only the scaffold floor from `0.10` to `0.03` changes the auxiliary
budget without changing the action unit:

```text
L_object_scaffold_new ~= 0.3 * L_object_scaffold_old
expected rho_action ~= 74-78%
```

Observed continuation:

```text
step   L_total  L_action  L_total-L_action  rho_action  active_ov  down_ov
1550   0.0462   0.0357    0.0105            77.3%       0.060      0.171
1600   0.0451   0.0353    0.0098            78.2%       0.025      0.106
1650   0.0396   0.0290    0.0107            73.1%       0.084      0.155
1700   0.0371   0.0269    0.0103            72.3%       0.075      0.109
```

This validates the setting.  The action objective became dominant and kept
falling, while the non-action budget remained near `0.010` and active duplicate
overlap stayed at `0`.

## Future Deployment Guidance

Use this policy for future long runs:

```text
Default production recipe:
  lambda_action_pos = 2.0
  lambda_action_rot = 2.0
  lambda_action_gripper = 2.0
  OBJECT_SCAFFOLD_DECAY_FLOOR = 0.03

Do not raise action above 2.0 unless all are true:
  action_default_equiv is flat for several 50-step gates;
  active/downstream overlap remains healthy;
  active_duplicate remains 0;
  non-action budget remains too high relative to action;
  the floor-0.03 branch has passed at least one checkpoint gate.

Do not lower scaffold below 0.03 unless:
  sidecar/object-owner metrics stay healthy;
  recycle/identity metrics are stable;
  overlays confirm object ownership remains correct.

Do not re-enable raw predictive losses unless:
  target scale is normalized;
  matching is permutation-invariant;
  future targets are detached;
  slot-JEPA/support-pred raw diagnostics no longer explode.
```

Gate thresholds for the current branch:

```text
continue:
  loss_action_default_equiv is decreasing or flat within noise;
  L_total - L_action stays near 0.010-0.015;
  active same-role support overlap < 0.25;
  downstream same-role support overlap < 0.25;
  active duplicate overlap = 0;
  grad_clip_applied is false or rare.

pause and inspect:
  recycle_rate collapses toward 0 while identity_switch rises;
  active/downstream overlap exceeds 0.25 for two consecutive logs;
  action stalls while non-action rises;
  overlays show active owner moving off the task object.

new ablation only:
  action remains flat after the floor-0.03 branch reaches the next checkpoint
  and all structural metrics remain healthy.  In that case, test action weight
  above 2.0 as an explicit ablation, not as the maintained default.
```

## Theory Link to Paper Matrix

The paper matrix in
`docs/PICF_AQR_OWM_SLOT_VLA_PAPER_MATRIX_20260522.md` supports this weighting
policy:

```text
PI0 / OpenVLA / OpenVLA-OFT / VLANeXt:
  action imitation or flow/action-token modeling is the production objective.

MetaSlot / QASA / slot merging:
  object slots need quality/duplicate control, but count/slot pressure should
  not directly fight reconstruction/action fidelity.

V-JEPA / VLA-JEPA / JEPA-VLA:
  predictive latents are useful as context or detached targets; raw predictive
  losses should not dominate the policy path.

TLA / VLA-Touch / OmniVTLA / OmniVTA:
  tactile/contact evidence should refine and ground manipulation, not become
  a competing always-on objective.
```

Therefore the maintained PICF loss philosophy is:

```text
bootstrap object ownership;
decay weak object scaffold;
preserve dense context;
let action dominate the long-run policy objective.
```

## Paper Survey: 24 Papers / Systems Reviewed

### Action-dominant VLA / policy papers

1. RT-1: Robotics Transformer for Real-World Control at Scale, 2022.  Uses scalable robot trajectory imitation as the primary objective.  Relevance: establishes action prediction as core supervision for generalist robot policies.
   URL: https://arxiv.org/abs/2212.06817

2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, 2023.  Treats actions as tokens in a VLM-style objective.  Relevance: task semantics can come from VLM pretraining, but robot finetuning is action-token oriented.
   URL: https://arxiv.org/abs/2307.15818

3. OpenVLA: An Open-Source Vision-Language-Action Model, 2024.  Uses action-token prediction as the fine-tuning target.  Relevance: no large auxiliary suite is required for action learning; auxiliary structure must not dominate.
   URL: https://arxiv.org/abs/2406.09246

4. Octo: An Open-Source Generalist Robot Policy, 2024.  Generalist policy trained on 800k robot trajectories with language/goal conditioning.  Relevance: broad robot policy pretraining remains behavior/action centered.
   URL: https://arxiv.org/abs/2405.12213

5. PI0: A Vision-Language-Action Flow Model for General Robot Control, 2024.  Builds a flow-matching action expert on a pretrained VLM.  Relevance: our `loss_action_default_equiv` is directly aligned with this action-flow paradigm.
   URL: https://arxiv.org/abs/2410.24164

6. RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation, 2024.  Uses diffusion action modeling at scale.  Relevance: action generation is the central training signal; multimodal conditioning supports it.
   URL: https://arxiv.org/abs/2410.07864

7. FAST: Efficient Action Tokenization for Vision-Language-Action Models, 2025.  Improves action tokenization efficiency.  Relevance: action representation efficiency matters more than adding heavy auxiliary losses.
   URL: https://arxiv.org/abs/2501.09747

8. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success / OpenVLA-OFT, 2025.  Studies action decoding, representations, and objectives for OpenVLA fine-tuning.  Relevance: supports optimizing action objective and decoding recipe rather than letting auxiliary losses dominate.
   URL: https://arxiv.org/abs/2502.19645

9. DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control, 2025.  Adds a diffusion action expert to a VLM.  Relevance: again separates semantic backbone from action expert while keeping action learning central.
   URL: https://arxiv.org/abs/2502.05855

10. PI0.5: a Vision-Language-Action Model with Open-World Generalization, 2025.  Co-trains on heterogeneous robot and semantic data.  Relevance: heterogeneous auxiliary data helps generalization, but the robot policy still needs action-aligned training pressure.
    URL: https://arxiv.org/abs/2504.16054

11. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics, 2025.  Efficient VLA for lower-cost training/inference.  Relevance: supports simple, efficient action-centric recipes and warns against unnecessary complexity.
    URL: https://arxiv.org/abs/2506.01844

12. SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model, 2025.  Adds 3D spatial encodings and adaptive action grids.  Relevance: spatial/object information is valuable when it improves action grounding, not as an independent objective that overwhelms behavior learning.
    URL: https://arxiv.org/abs/2501.15830

13. OG-VLA: 3D-Aware Vision Language Action Model via Orthographic Image Generation, 2025.  Integrates 3D-aware perception with VLA control.  Relevance: supports structured perception as an action-supporting prior.
    URL: https://arxiv.org/abs/2506.01196

14. WorldVLA: Towards Autoregressive Action World Model, 2025.  Explores action-world modeling.  Relevance: world modeling helps when tied to action prediction; uncontrolled predictive losses can distract.
    URL: https://arxiv.org/abs/2506.21539

15. VLANeXt: Recipes for Building Strong VLA Models, 2026.  Systematically studies VLA design choices.  Relevance: strongly supports controlled ablations and action-modeling recipes instead of ad hoc loss accumulation.
    URL: https://arxiv.org/abs/2602.18532

16. AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models, 2026.  Proposes a standalone action expert with long-lived memory.  Relevance: reinforces that high-frequency action generation is its own core module, not an auxiliary afterthought.
    URL: https://arxiv.org/abs/2603.10126

17. Green-VLA: Staged Vision-Language-Action Model for Generalist Robots, 2026.  Uses staged curriculum and action alignment.  Relevance: supports our staged scaffold-decay approach: bootstrap first, then action-aligned training.
    URL: https://arxiv.org/abs/2602.00919

18. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA, 2025.  Uses soft prompts for heterogeneous data/embodiments.  Relevance: supports data-source/embodiment conditioning as low-parameter modulation rather than large auxiliary objectives.
    URL: https://arxiv.org/abs/2510.10274

19. LingBot-VLA: A Pragmatic VLA Foundation Model, 2026.  Uses real-world dual-arm data and VLA/action modules.  Relevance: supports action-dominant learning under heterogeneous real data and careful data pipelines.
    URL: https://arxiv.org/abs/2601.18692

20. MolmoAct2: Action Reasoning Models for Real-world Deployment, 2026.  Open action reasoning architecture with updated action tokenization/continuous action prediction.  Relevance: supports action reasoning/action tokenizer design as central in deployment-oriented VLA systems.
    URL: https://huggingface.co/papers/2605.02881

### JEPA / predictive representation papers

21. V-JEPA 2: Self-Supervised Video Models, 2025.  Dense video representation and prediction.  Relevance: supports V-JEPA as frozen/typed temporal context, but not as an uncontrolled active loss in the policy path.
    URL: https://arxiv.org/abs/2506.09985

22. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model, 2026.  Uses leakage-free future latent targets.  Relevance: predictive losses should remain detached and guarded; our exploding raw slot-JEPA should not be enabled until normalized/calibrated.
    URL: https://arxiv.org/abs/2602.10098

23. JEPA-VLA: Video Predictive Embedding is Needed for VLA Models, 2026.  Conditions action prediction on predictive video embeddings.  Relevance: supports injecting V-JEPA-style context, but not letting raw predictive MSE dominate action training.
    URL: https://arxiv.org/abs/2602.11832

### Object-centric / slot / tactile papers

24. Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?, 2025.  Shows IsSameObject-like binding can be probed in pretrained ViTs.  Relevance: supports our binding subspace and same-object diagnostics; does not imply adding a raw identity loss.
    URL: https://arxiv.org/abs/2510.24709

25. MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning, 2025.  Variable effective slot count.  Relevance: supports our variable active/context/reserve slot strategy over fixed all-slots-active usage.
    URL: https://arxiv.org/abs/2505.20772

26. SlotVLA: Towards Modeling Object-Relation Representations in Robotic Manipulation, 2025.  Uses object-relation representations for manipulation.  Relevance: supports object-centric routing, but object structure must serve manipulation and action prediction.
    URL: https://arxiv.org/abs/2511.06754

27. Object-Centric World Model for Language-Guided Manipulation, 2025.  Slot-attention language-guided world model.  Relevance: supports object-centric predictive structure, but as world-model structure rather than raw uncontrolled action cotrain loss.
    URL: https://arxiv.org/abs/2503.06170

28. OA-WAM: Object-Addressable World Action Model, 2026.  Persistent address/content slots.  Relevance: supports object-addressable belief, but address must be gated by current evidence and not hard-locked.
    URL: https://arxiv.org/abs/2605.06481

29. TLA: Tactile-Language-Action Model for Contact-Rich Manipulation, 2025.  Uses tactile-language-action data for contact-rich tasks.  Relevance: supports tactile evidence as contact-conditioned information, not always-on background tokens.
    URL: https://arxiv.org/abs/2503.08548

30. OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing, 2025.  Semantic tactile alignment.  Relevance: supports our gated tactile-to-object binding design.
    URL: https://arxiv.org/abs/2508.08706

31. VLA-Touch: Enhancing VLA Models with Dual-Level Tactile Feedback, 2025.  Uses tactile feedback for high-level planning and diffusion controller refinement.  Relevance: tactile should refine actions/contact, not compete as an independent dense loss.
    URL: https://arxiv.org/abs/2507.17294

32. OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Manipulation, 2026.  Two-stream visuo-tactile world model plus contact-aware action policy.  Relevance: supports contact-aware fusion and world modeling but still centers control execution.
    URL: https://arxiv.org/abs/2603.19201

## Practical Conclusion for PICF-AQR-OWM

The literature supports this curriculum:

```text
1. Use object/slot/sidecar evidence as a bootstrap teacher.
2. When active object ownership is healthy, decay the object scaffold sharply.
3. Keep traditional PI0 action scale as the main long-run objective.
4. Keep raw predictive losses disabled unless normalized/calibrated.
5. Do not let raw reserve/context overlap drive optimization; monitor active metrics.
```

For the current branch, this means:

```text
from step1500:
  keep lambda_action_pos/rot/gripper = 2.0
  reduce object_scaffold_decay_floor = 0.03
  keep lambda_slot_jepa/support_pred/binding_consistency/aqr_denoising = 0
  keep anchor overlays and active-object gates
```

## 2026-05-24 Fresh-Optimizer Action-Polish Promotion

Evidence:

```text
A7 continuous run:
  picf_a7_actionprefix_rmsnorm_long30k_20260524
  saved checkpoint: .../picf_a7_actionprefix_rmsnorm_long30k_20260524/2000
  stopped after step2000 checkpoint was safely written.

A5 model-only/fresh-optimizer probe:
  picf_a5_optreset_from1000_action2_probe1500_20260524
  checkpoint: .../picf_a5_optreset_from1000_action2_probe1500_20260524/1500
```

Observed contrast:

```text
A7 continuous:
  action stayed in the 0.049~0.056 band through step2100.
  active overlap stayed healthy, but action did not break the plateau.
  raw slot_jepa telemetry exploded; lambda_slot_jepa remains 0.

A5 fresh optimizer:
  action improved rapidly to 0.041 at step1200.
  finished at 0.0466 by step1500 with total=0.0585.
  active/downstream overlaps stayed controlled.
  raw slot_jepa stayed low relative to A7.
```

Decision:

```text
Promote the A5 step1500 model-only checkpoint into an A7 production continuation.

Do not add new structural losses in this step.  The causal test is:
  model weights: A5 step1500
  optimizer: fresh
  action pressure: normal dominant scale
  LR: action-polish scale, not bootstrap scale
  sidecar/object scaffold: weak floor
  predictive losses: still disabled
```

Maintained launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
run_a7_from_a5_1500_freshopt_actionpolish_30k_20260524.sh
```

Key launch contract:

```text
RESUME_CHECKPOINT =
  /mnt/checkpoints/picf_core/picf_core/
  picf_a5_optreset_from1000_action2_probe1500_20260524/1500

ACTION_LOSS_WEIGHT = 2.0
LR                 = 5e-5
MIN_LR             = 2e-5
WARMUP_STEPS       = 20
SAVE_INTERVAL      = 500
KEEP_LAST_CHECKPOINTS = 3
LOG_INTERVAL       = 50
ANCHOR_OVERLAY_INTERVAL = 100
```

Acceptance:

```text
must improve:
  loss_action_default_equiv should beat the A7 continuous 0.05~0.056 band;
  loss_total should not rise because of extra scaffold pressure;
  loss_anchor_pv should not monotonically worsen;
  active/downstream overlaps should remain controlled.

must stay guarded:
  lambda_slot_jepa/support_pred/binding_consistency/aqr_denoising remain 0;
  raw slot_jepa is telemetry only until normalized/matched.
```

## 2026-05-25 Step-2500 Continuity Archive

Run:

```text
picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524
resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a5_optreset_from1000_action2_probe1500_20260524/1500
optimizer:
  model-only resume with fresh optimizer state for this A7 continuation
```

Important boundary:

```text
Within this A7 continuation, there has been one fresh optimizer reset at the
step-1500 handoff.  Do not interrupt the run for a mid-stream CALVIN eval unless
losses show real collapse, because another stop/resume would make the optimizer
trajectory harder to compare.
```

Step-2500 metrics:

```text
loss_total                         0.04962
loss_action_default_equiv          0.03631
loss_action_active7                0.16499
loss_slot_jepa_raw                 6.09
loss_anchor_object_pull            0.4165
loss_anchor_pv                     0.5316
loss_aqr_denoising                 1.1204
loss_mapg_routing                  0.4316
aqr_active_same_role_overlap_max   0.1300
aqr_downstream_same_role_overlap   0.1289
posterior_identity_switch_rate     0.2106
posterior_recycle_rate             0.1284
grad_norm                          3.49
```

Action comparison against the archived 2026-04-22 PI0.5-only ablation:

```text
4-22 step3000   loss_action ~= 0.0496
4-22 step5000   loss_action ~= 0.0451
4-22 step15100  loss_action ~= 0.0311

current step2500 loss_action_default_equiv = 0.0363
```

Interpretation:

```text
The current run is already clearly past the old 4-22 5k action-loss level and
is moving toward, but has not reached, the old 15k level.  This is the strongest
reason to preserve the continuous optimizer trajectory.
```

Health reading:

```text
healthy:
  action remains in the 0.03x band after the fresh-optimizer handoff;
  active/downstream overlap remains far below the historical 0.9+ collapse band;
  recycle is stable around 0.13;
  anchor_pv is stable near its local low;
  raw slot_jepa has retreated from the earlier spike and remains disabled.

watch:
  loss_anchor_object_pull is sample/teacher-quality sensitive and bounced to
  0.416 at step2500;
  active/downstream overlap and identity_switch are local high points at
  step2500, but still not collapse-level;
  raw same-role overlap remains 1.0 because reserve/inactive capacity is still
  counted in that diagnostic.
```

Decision:

```text
Continue training.  Do not run CALVIN eval on the active training GPUs at
step2500, because the run is still improving and a mid-stream interruption
would confound optimizer-state analysis.

Next decision points:
  step3000:
    if loss_action_default_equiv remains in 0.03x and active/downstream overlap
    remains controlled, continue.

  step3500 or first clear plateau:
    consider a small CALVIN eval on the latest checkpoint or on a separate idle
    machine.  Do not reset optimizer purely for eval convenience.
```

## 2026-05-25 Step-2500 Fresh-Optimizer Causal Probe

Reason:

```text
After the step-2400 local low, the A7 continuation moved back toward the
0.048-0.051 action band while active/downstream overlap, recycle, and gradient
health remained controlled.  This suggests an optimizer/schedule phase issue
rather than renewed anchor collapse.
```

Probe contract:

```text
resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524/2500

new run:
  picf_pc1_from_a7_2500_freshopt_lr5e5_30k_20260525

critical setting:
  num_train_steps = 30000
```

The total step count must remain `30000`, not a short diagnostic value such as
`3500`, because changing the total horizon changes the cosine/LR schedule and
invalidates the comparison.  The run is a long-scheduler causal probe that can
be stopped after the relevant intermediate checkpoints/metrics have been
observed.

Decision logic:

```text
If the fresh optimizer from step2500 immediately returns to the A7 continuous
0.05 band, the rebound is checkpoint/model/data-state driven.

If it drops back toward the 0.03x band, stale optimizer state or phase-specific
optimizer momentum is causal.

If it first drops and then rebounds again after a short window, the system is
showing a repeatable action-polish window followed by LR/schedule mismatch; the
next probe should lower LR or add an explicit phase reset schedule rather than
changing anchor structure.
```

## 2026-05-25 Low-LR Fresh-Optimizer Control

Reason:

```text
The PC1 5e-5 fresh-optimizer probe from the A7 step2500 checkpoint released
action loss from the A7 continuous 0.045-0.053 band into the 0.031-0.039 band.
However, its active/downstream support overlap and anchor-object-pull variance
were higher than the A7 continuous run.  This isolates optimizer-state reset as
useful, but it does not yet isolate whether 5e-5 is the best phase-boundary LR.
```

Control contract:

```text
resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524/2500

new run:
  picf_pc1_from_a7_2500_freshopt_lr2e5_30k_20260525

constant:
  same model checkpoint
  same sidecar data
  same slot/OEML/object-owner recipe
  same action weight = 2.0
  same num_train_steps = 30000
  fresh optimizer/scheduler state

only intended variable:
  LR 5e-5 / min 2e-5  ->  LR 2e-5 / min 8e-6
```

Interpretation:

```text
If LR=2e-5 keeps action in the 0.03x band while reducing overlap and
anchor-object-pull variance, the maintained recipe should use a low-LR
fresh-optimizer phase boundary.

If LR=2e-5 is cleaner but action stays materially worse than 5e-5, the recipe
should use a short 5e-5 action-release window followed by a low-LR polish phase.

If LR=2e-5 also rebounds to the A7 continuous 0.04x-0.05x band, the main cause
is not LR magnitude but optimizer phase/state or checkpoint/data curriculum.
```

Operational note:

```text
A7 must continue uninterrupted as the continuous-optimizer reference.  The
separate A5 host is unavailable at the time of this note, so PC1 is reused after
archiving the 5e-5 probe tail.  This is safe because the 5e-5 causal probe
already produced the relevant early action-release evidence through step3300.
```

5e-5 archive snapshot before PC1 reuse:

```text
step2700 loss_action_default_equiv=0.03155
step3100 loss_action_default_equiv=0.03366
step3250 loss_action_default_equiv=0.03362
step3300 loss_action_default_equiv=0.03456

step3300 active/downstream overlap = 0.111 / 0.127
step3300 anchor_object_pull        = 0.363
step3300 posterior_recycle_rate    = 0.126
```

This confirms that fresh optimizer state is useful for action.  The remaining
question is whether the higher LR is also injecting avoidable structure noise.

Execution correction:

```text
The strict step2500 LR=2e-5 paired control cannot be launched after the A7
training job advanced past the retention window.  `keep_last_checkpoints=3`
removed the step2500 model directory; the shared checkpoint tree now retains
3500/4000/4500 for the A7 run.

Therefore the executable control is changed to:

  picf_pc1_from_a7_4500_freshopt_lr2e5_30k_20260525

It is not a strict pair for the step2500 LR=5e-5 PC1 run.  Its role is to test
the current A7 plateau state directly: can a low-LR fresh optimizer from the
latest stable A7 model release action while preserving the cleaner A7 structure?
```

Launch:

```text
host: PC1 / px-cloud1.matpool.com:26620
tmux: picf_pc1_from_a7_4500_freshopt_lr2e5_30k_20260525
script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_pc1_from_a7_4500_freshopt_lr2e5_30k_20260525.sh

watch:
  step4550 first metric
  step4600 first 100-step decision point
```

Step4550 first metric:

```text
loss_action_default_equiv          0.03921
loss_total                         0.05079
loss_anchor_object_pull            0.26905
loss_anchor_pv                     0.51394
active/downstream overlap          0.075 / 0.079
posterior_recycle_rate             0.130
posterior_identity_switch_rate     0.186
grad_norm                          0.297
lr                                 1.93e-5
```

Reading:

```text
This is already below the A7 continuous step4550 action value (`0.04262`) while
keeping active/downstream overlap closer to A7 than to the noisier PC1 5e-5
probe.  The low-LR reset is therefore a plausible phase-boundary recipe.  Wait
for step4600 before making it the maintained answer.
```

Step4600 / 100-step check:

```text
PC1 low-LR reset from A7 step4500:
  loss_action_default_equiv        0.04111
  loss_total                       0.05245
  loss_anchor_object_pull          0.25680
  loss_anchor_pv                   0.51212
  active/downstream overlap        0.069 / 0.069
  posterior_recycle_rate           0.134
  grad_norm                        2.32

A7 continuous reference:
  step4600 loss_action_default_equiv = 0.04075
  step4650 loss_action_default_equiv = 0.04053
  step4650 active/downstream overlap = 0.051 / 0.067
```

Reading:

```text
The low-LR fresh optimizer is structurally safe, but it does not beat simply
continuing A7 over the same local window.  The earlier PC1 5e-5 release was real
evidence that optimizer phase matters, but this current-phase low-LR reset is
not yet evidence for replacing the maintained A7 continuation.  Keep A7 running
as the main line.  Use PC1 low-LR only as a control unless it later separates.
```

Step4800 follow-up:

```text
PC1 low-LR reset:
  step4650 action                   0.03872
  step4700 action                   0.03223
  step4750 action                   0.03818
  step4800 action                   0.03656
  step4800 active/downstream overlap 0.065 / 0.074
  step4800 anchor_object_pull        0.2688
  step4800 slot_jepa                 1.4078

A7 continuous reference:
  step4650 action                   0.04053
  step4700 action                   0.04570
  step4750 action                   0.04348
  step4800 action                   0.05163
  step4850 action                   0.04892
```

Updated reading:

```text
The step4600 snapshot was too early.  By step4700-4800, the PC1 low-LR fresh
optimizer run has separated from A7 on action while keeping active/downstream
overlap in the same safe band.  This makes low-LR fresh optimizer reset a real
candidate phase-boundary recipe, not just a safety control.  Continue PC1 to at
least the next 500-step checkpoint before promoting it, because we still need
to know whether the improvement persists beyond the first action-release
window.
```

Step5000 comparison:

```text
A7 continuous:
  latest checked step                5050
  action window 4800-5050            0.05163 -> 0.04393
  local max/min in same window       0.05354 / 0.04263
  active/downstream overlap          0.040 / 0.052 at step5050
  slot_jepa                          2.67 at step5050

PC1 low-LR fresh optimizer:
  latest checked step                5000
  action window 4700-5000            0.03223 -> 0.03947
  local max/min in same window       0.03948 / 0.03223
  active/downstream overlap          0.089 / 0.094 at step5000
  slot_jepa                          1.92 at step5000
```

Against the archived 2026-04-22 PI0.5-only ablation:

```text
4-22 step3000   action ~= 0.0496
4-22 step5000   action ~= 0.0451
4-22 step15100  action ~= 0.0311
4-22 step17500  action ~= 0.0266
4-22 step20000  action ~= 0.0214
```

Interpretation:

```text
At the same 5k-scale step count, both A7 and PC1 are competitive with or better
than the old 4-22 step5000 action level.  PC1 is materially better in the recent
mean and reached a local low near the old 15k level, but it has not sustained
old-15k/20k action loss yet.  Therefore the correct claim is early optimization
advantage plus cleaner object-router structure, not final behavior superiority.
```

2026-05-26 PC1 low-LR stop and mid-LR phase-boundary replacement:

```text
Stopped:
  picf_pc1_from_a7_4500_freshopt_lr2e5_30k_20260525

Reason:
  The 2e-5 current-phase control proved structurally safe and briefly reached
  the 0.03x action band, but it rebounded toward ~0.04 and became an underpowered
  continuation rather than a decisive maintained line.

Kept running:
  picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524

New probe:
  picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526

Resume:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524/5500

Recipe:
  fresh optimizer / model-only resume
  LR=3.5e-5, MIN_LR=1.4e-5, WARMUP_STEPS=20
  ACTION_LOSS_WEIGHT=2.0
  OBJECT_SCAFFOLD_DECAY_MODE=cosine
  OBJECT_SCAFFOLD_DECAY_END_STEP=1500
  OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
  SAVE_INTERVAL=1000
  KEEP_LAST_CHECKPOINTS=5

Why this is the next causal probe:
  A7 5e-5 keeps the action path trainable but has a wider 0.04x rebound band.
  PC1 2e-5 is safe but may be too slow to keep adapting after the optimizer
  reset release.  The 3.5e-5 run asks whether a middle LR can preserve PC1's
  cleaner action band without losing A7's optimization momentum.

Acceptance:
  Compare at 5600/5700/6000-equivalent windows against the still-running A7
  reference.  Promote only if the new PC1 line keeps action below the A7 recent
  mean while active/downstream overlap stays in the safe band.
```

2026-05-26 PC1 mid-LR result and next causal experiment:

```text
Run:
  picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526

Observed:
  best local action at step6400:
    loss_action_default_equiv ~= 0.02853
  rebound window:
    step6550 ~= 0.04335
    step6600 ~= 0.03994
    step6650 ~= 0.04284
    step6700 ~= 0.04155
    step6750 ~= 0.04359
    step6800 ~= 0.04138
    step6850 ~= 0.04351
  mean step6650-6850:
    ~= 0.04257

Interpretation:
  Fresh optimizer plus middle LR can release action to a strong local band, but
  it does not remove the rebound.  Because the maintained recipe uses
  picf-action-prefix-stopgrad, the rebound is unlikely to be caused by action
  gradients directly corrupting PICF.  The sharper hypothesis is that PICF
  belief/prefix values keep moving under structure/object losses while the
  policy head is trying to fit them.

Next causal probe:
  picf_pc1_freezepicf_policyonly_from_pc1_6000_action2_30k_20260526

Contract:
  resume model weights from PC1-midLR step6000 when available
  reset optimizer/scheduler
  freeze all core.* PICF parameters with picf_trainable_scope=policy_only
  keep PICF forward evidence active so the policy still sees the same prefix
  train only non-core semantic/action policy path
  set structural/object scaffold losses to 0
  keep ACTION_LOSS_WEIGHT=2.0 and LR=3.5e-5 for comparability

Decision rule:
  If action remains stable after the usual 300-500 step rebound window, the
  moving-prefix hypothesis is confirmed and the next production fix should be a
  staged/prefix-stability schedule rather than another overlap penalty.
  If action still rebounds, root cause shifts to policy optimizer/data order or
  semantic/action capacity rather than PICF drift.
```

# PICF-AQR-OWM Slot / VLA / Object-Binding Paper Matrix

Date: 2026-05-22

Purpose: maintain a paper-backed design map for the current PICF-AQR-OWM
belief router.  This document is separate from the action-weight gate because
it answers a different question:

```text
Which recent slot/object-binding/VLA/tactile mechanisms should be imported
into PICF, which should remain guarded, and which should be rejected because
they conflict with a multimodal control belief router?
```

This is not a generic bibliography.  Each paper is mapped to an implementation
decision in PICF.

## Current Design Thesis

PICF is not a pure image reconstruction object-centric learner.  It is a
posterior belief router for robot control:

```text
dense typed evidence
  -> active object files + reserve/context files
  -> posterior correction / identity binding
  -> PI0.5 action path
```

Therefore the maintained architecture should follow four constraints:

```text
1. Active object files should explain task/contact/motion-relevant evidence.
2. Dense background and low-confidence context should not be forced into
   object slots; it should remain available as residual context.
3. Weak object scaffolds should bootstrap binding, then decay so action
   imitation / flow matching dominates long-run training.
4. Predictive/object losses must be calibrated and detached; raw future-token
   losses must stay guarded until normalized.
```

## Papers Reviewed

### A. Object Binding / Slot Count / Duplicate Control

1. Does Object Binding Naturally Emerge in Large Pretrained Vision
   Transformers?  NeurIPS 2025 Spotlight.
   URL: https://arxiv.org/abs/2510.24709

   Mechanism: pairwise IsSameObject is decoded from patch embeddings with a
   quadratic / low-dimensional similarity probe.  The useful signal is a
   same-object subspace, not raw attention alone.

   PICF decision: keep `binding_signature_proj`,
   support-weighted binding signatures, and pairwise binding-subspace terms.
   Do not replace the belief filter with raw visual cosine binding.  Add
   IsSameObject-style offline probes when tracklet/mask artifacts are mature.

2. MetaSlot: Break Through the Fixed Number of Slots in Object-Centric
   Learning.  NeurIPS 2025.
   URL: https://arxiv.org/abs/2505.20772

   Mechanism: variable effective slot count by prototype codebook, duplicate
   masking, and attenuated slot noise.

   PICF decision: adopt the principle, not literal visual VQ posterior truth.
   PICF uses active/context/reserve selection, slot quality, duplicate
   suppression, and posterior file competition because identity can be tactile,
   geometric, linguistic, or temporal, not only visual-prototype based.

3. QASA: Quality-Guided K-Adaptive Slot Attention for Unsupervised
   Object-Centric Learning.  2026.
   URL: https://arxiv.org/abs/2601.12936

   Mechanism: separate slot-quality selection from reconstruction to avoid a
   conflict between minimizing slot count and preserving reconstruction
   fidelity.

   PICF decision: keep active-quality gates and quality-ordered duplicate
   suppression.  Do not add a raw "use fewer slots" loss directly against
   action training; it creates the same conflict QASA warns about.

4. When Slots Compete: Slot Merging in Object-Centric Learning.  2026.
   URL: https://arxiv.org/abs/2603.11246

   Mechanism: overlapping slots are merged during training using overlap
   statistics and barycentric updates.

   PICF decision: implement the analogue as active/support duplicate
   suppression and posterior file competition, not hard deletion of dense
   tokens.  Raw reserve overlap may be high; action-visible active/downstream
   overlap is the acceptance metric.

5. Temporally Consistent Object-Centric Learning by Contrasting Slots.  CVPR
   2025.
   URL: https://openaccess.thecvf.com/content/CVPR2025/html/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.html

   Mechanism: contrast slots across time to stabilize video object identity.

   PICF decision: keep temporal identity metrics and tracklet/motion sidecar
   as the correct supervision source.  Do not enable index-aligned predictive
   losses until matching/normalization is stable.

6. Learning Object-Centric Representations Based on Slots in Real World
   Scenarios.  2025.
   URL: https://arxiv.org/abs/2509.24652

   Mechanism: slot-based conditioning on pretrained models while preserving
   visual priors.

   PICF decision: validates our frozen-backbone + lightweight slot/belief
   adapters policy.  Avoid full backbone retraining during structural
   diagnosis.

7. Object-Centric Learning with Slot Attention.  NeurIPS 2020 baseline.
   URL: https://arxiv.org/abs/2006.15055

   Mechanism: iterative slot-feature competition and object-style grouping.

   PICF decision: PICF uses the competition principle, but must not force all
   dense tokens into object files because robot policies also need global
   layout, background affordances, and residual context.

8. SAVi / Slot Attention for Video.  2021 baseline.
   URL: https://arxiv.org/abs/2111.12594

   Mechanism: recurrent object slots with temporal conditioning.

   PICF decision: posterior files and tracklet memory are the control-oriented
   analogue.  This supports state carry-over and burn-in/unroll, but not raw
   pixel reconstruction as the primary objective.

### B. Object-Centric Robotics / Manipulation

9. SlotVLA: Towards Modeling of Object-Relation Representations in Robotic
   Manipulation.  2025.
   URL: https://arxiv.org/abs/2511.06754

   Mechanism: object-relation representations improve manipulation action
   decoding.

   PICF decision: supports object files and relation-aware action prefix.  It
   does not justify letting object auxiliary losses dominate the PI0.5 action
   path.

10. STORM: Slot-based Task-aware Object-centric Representation for robotic
    Manipulation.  2026.
    URL: https://arxiv.org/abs/2601.20381

    Mechanism: freeze visual foundation models, stabilize semantic slots, then
    adapt jointly with policy.

    PICF decision: strong support for our staged recipe:
    frozen V-JEPA/Sonata/AnyTouch, trainable PaliGemma/action adapters in
    production, and scaffold decay after active-object metrics are healthy.

11. Object-Centric World Model for Language-Guided Manipulation.  2025.
    URL: https://arxiv.org/abs/2503.06170

    Mechanism: language-guided slot world model predicts future states in
    compact object-centric latent space.

    PICF decision: supports slot-JEPA/support prediction as a future guarded
    hook.  Current raw `loss_slot_jepa` must remain disabled until normalized
    and matched; exploding raw diagnostics are not acceptable training
    pressure.

12. FOCUS: Object-centric world models for robotic manipulation.  2025.
    URL: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1585386/full

    Mechanism: object-centric world models can improve manipulation learning.

    PICF decision: supports object-centric predictive state, but the policy
    still needs action acceptance metrics.  CALVIN/video evidence remains the
    final gate.

13. SPOT: SE(3) Pose Trajectory Diffusion for Object-Centric Manipulation.
    2025.
    URL: https://research.nvidia.com/publication/2025-05_spot-se3-pose-trajectory-diffusion-object-centric-manipulation

    Mechanism: object-centric action generation around SE(3) object pose.

    PICF decision: reinforces geometry/object-owner transport.  It does not
    require hard segmentation labels when the dataset lacks object IDs.

14. Object-Centric World Models for Causality-Aware Reinforcement Learning.
    2025.
    URL: https://arxiv.org/abs/2511.14262

    Mechanism: object-centric Transformer world model plus causal policy/value
    attention.

    PICF decision: useful future direction for relation/causal metrics, but
    not a reason to add RL/causal losses during the current imitation gate.

15. Multi-Slot Attention with State Guidance for Egocentric Robotic
    Manipulation.  2026.
    URL: https://www.mdpi.com/2079-9292/15/7/1365

    Mechanism: proprioceptive state guides slot attention under egocentric
    robot views.

    PICF decision: supports wrist/proprio/contact-conditioned slot routing.
    Our tactile/contact owner binding should attach to object files, not form a
    permanent gripper object file that competes with manipulated objects.

16. Vision-based manipulation from single human video with open-world object
    graphs / ORION.  2026.
    URL: https://link.springer.com/article/10.1007/s10514-026-10253-8

    Mechanism: object graphs model states and relations of task-relevant
    objects under open-world variation.

    PICF decision: relation/context graph is valuable, but should be
    action-visible as structured context rather than all background forced into
    active object slots.

### C. Object-Addressable / Predictive VLA / JEPA

17. OA-WAM: Object-Addressable World Action Model for Robust Robot
    Manipulation.  2026.
    URL: https://arxiv.org/abs/2605.06481

    Mechanism: persistent object addresses plus time-varying content.

    PICF decision: supports `slot_address` and address-aware cache, but address
    must remain evidence-gated.  Hard address lock-in is unsafe when weak
    sidecar/mask evidence is noisy.

18. V-JEPA 2: Self-Supervised Video Models.  2025.
    URL: https://arxiv.org/abs/2506.09985

    Mechanism: dense predictive video representation.

    PICF decision: use V-JEPA as frozen dense typed context and future cache
    target.  Do not prune/destroy dense V-JEPA tokens; object files are a
    routing layer, not a replacement for dense representation.

19. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model.
    2026.
    URL: https://arxiv.org/abs/2602.10098

    Mechanism: leakage-free latent world-model targets for VLA.

    PICF decision: future latent targets must be detached from current action
    path.  This supports guarded matched predictive losses, not index-aligned
    or unnormalized raw MSE.

20. JEPA-VLA: Video Predictive Embedding is Needed for VLA Models.  2026.
    URL: https://arxiv.org/abs/2602.11832

    Mechanism: predictive video embeddings improve VLA state estimation.

    PICF decision: supports V-JEPA dense context injection and future
    gated-cross-attention work.  It does not justify letting predictive losses
    dominate the action objective.

21. VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language.
    2026.
    URL: https://huggingface.co/papers/2512.10942

    Mechanism: embedding-space prediction for efficient multimodal
    representation.

    PICF decision: validates latent/embedding prediction over pixel
    reconstruction for our control router.

22. WorldVLA: Towards Autoregressive Action World Model.  2025.
    URL: https://arxiv.org/abs/2506.21539

    Mechanism: action-world modeling.

    PICF decision: future action-world modeling should be action-aligned and
    leakage-free; current long-run should not enable unstable raw slot-JEPA.

### D. Action-Dominant VLA Recipes

23. PI0: A Vision-Language-Action Flow Model for General Robot Control.  2024.
    URL: https://arxiv.org/abs/2410.24164

    Mechanism: flow-matching action expert on a pretrained VLM.

    PICF decision: action loss is the main production objective.  Object losses
    are scaffold/shaping terms, not final control targets.

24. PI0.5: A Vision-Language-Action Model with Open-World Generalization.
    2025.
    URL: https://arxiv.org/abs/2504.16054

    Mechanism: heterogeneous co-training for open-world generalization.

    PICF decision: supports our modality-missing slot design and keeping
    PaliGemma trainable in production action co-training.

25. OpenVLA: An Open-Source Vision-Language-Action Model.  2024.
    URL: https://arxiv.org/abs/2406.09246

    Mechanism: VLM-to-action fine-tuning.

    PICF decision: action objectives are directly comparable only when
    logging `loss_action_default_equiv`; auxiliary-heavy totals are not
    behavior-equivalent.

26. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success /
    OpenVLA-OFT.  2025.
    URL: https://arxiv.org/abs/2502.19645

    Mechanism: systematic VLA finetuning recipe improves speed and success.

    PICF decision: supports controlled recipes and action-scale comparability.
    Do not infer success from auxiliary loss decrease alone.

27. VLANeXt: Recipes for Building Strong VLA Models.  2026.
    URL: https://arxiv.org/abs/2602.18532

    Mechanism: systematic VLA ablations distilled into design recipes.

    PICF decision: validates the current practice of smoke gates, ablations,
    and avoiding uncontrolled auxiliary accumulation.

28. AR-VLA: True Autoregressive Action Expert for Vision-Language-Action
    Models.  2026.
    URL: https://arxiv.org/abs/2603.10126

    Mechanism: standalone high-frequency action expert with memory.

    PICF decision: action generation should be treated as a first-class module.
    PICF object files should condition action, not replace action optimization.

29. DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General
    Robot Control.  2025.
    URL: https://arxiv.org/abs/2502.05855

    Mechanism: VLM plus diffusion action expert.

    PICF decision: supports separating semantic backbone, object belief
    routing, and action head while training the action path strongly.

30. FAST: Efficient Action Tokenization for Vision-Language-Action Models.
    2025.
    URL: https://arxiv.org/abs/2501.09747

    Mechanism: action tokenization efficiency.

    PICF decision: if action plateaus, inspect action representation and
    learning rate before adding object losses.

31. SpatialVLA: Exploring Spatial Representations for VLA Models.  2025.
    URL: https://arxiv.org/abs/2501.15830

    Mechanism: spatial representations improve action grounding.

    PICF decision: supports projective/point geometry and owner transport.
    Spatial priors must remain calibrated and not overrule task semantics.

32. VQ-VLA: Improving VLA Models via Scaling Vector-Quantized Action
    Tokenizers.  ICCV 2025.
    URL: https://www.openaccess.thecvf.com/content/ICCV2025/papers/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.pdf

    Mechanism: improves action tokenizers via residual VQ-VAE.

    PICF decision: future action-tokenization work is orthogonal to object
    router repair.  Do not conflate action-tokenizer gains with slot quality.

### E. Tactile / Contact-Rich Multimodal Policies

33. TLA: Tactile-Language-Action Model for Contact-Rich Manipulation.  2025.
    URL: https://arxiv.org/abs/2503.08548

    Mechanism: sequential tactile feedback grounded with language for contact
    policy generation.

    PICF decision: tactile tokens should be contact-gated and bound to the
    manipulated object owner, not kept as an always-active independent gripper
    slot.

34. VLA-Touch: Enhancing VLA Models with Dual-Level Tactile Feedback.  2025.
    URL: https://arxiv.org/abs/2507.17294

    Mechanism: high-level tactile semantic feedback plus low-level tactile
    control refinement.

    PICF decision: supports tactile as refinement/conditioner.  It should not
    create a separate role that steals object evidence from the active object.

35. OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned
    Tactile Sensing.  2025.
    URL: https://arxiv.org/abs/2508.08706

    Mechanism: semantic tactile alignment in VLA.

    PICF decision: supports the current tactile-to-object binding direction
    and future semantic tactile probes.

36. OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Manipulation.
    2026.
    URL: https://arxiv.org/abs/2603.19201

    Mechanism: self-supervised tactile encoder, visuo-tactile world model,
    contact-aware fusion policy, and reflexive controller.

    PICF decision: supports contact-aware fusion and guarded tactile world
    modeling.  Current training should keep tactile as evidence/correction
    until tactile predictive losses are calibrated.

37. HapticVLA: Contact-Rich Manipulation via VLA without Inference-Time
    Tactile Sensing.  2026.
    URL: https://arxiv.org/abs/2603.15257

    Mechanism: distills contact/tactile knowledge so inference can rely less
    on tactile sensing.

    PICF decision: long-term useful for data-missing robustness.  Near-term,
    do not require tactile availability for every dataset; treat tactile as an
    optional typed evidence branch.

38. Visuo-Tactile World Models.  2026.
    URL: https://arxiv.org/abs/2602.06001

    Mechanism: joint visual/tactile world model for manipulation.

    PICF decision: supports multimodal predictive belief, but only after
    action and active-object binding are stable.

## Implementation Mapping

### Already Represented in Current PICF

```text
pairwise binding subspace:
  binding_signature_proj
  support-weighted binding signatures
  low-rank/quadratic binding terms

adaptive effective slot count:
  active/context/reserve files
  active-quality gates
  no-object/dustbin behavior
  context duplicate suppression

duplicate control:
  active support dedup
  context support dedup
  posterior file competition
  active/downstream overlap metrics

weak object scaffolding:
  sidecar/contact/motion owner candidates
  object core/mask/point compactness
  scaffold decay

dense context preservation:
  background/reserve context is retained outside active object files
  dense V-JEPA/point/PaliGemma evidence is not pruned by sidecar masks

tactile policy:
  tactile is contact-conditioned evidence
  tactile should bind to manipulated object owner, not permanently to a
  separate gripper role
```

### Guarded / Future

```text
IsSameObject offline probe:
  needs mature sidecar/tracklet artifacts for weak labels.

matched slot-JEPA / support prediction:
  keep lambda 0 until normalized and detached target scale is fixed.

full multimodal address prototype:
  only after visual/tactile/point/language identities can be calibrated
  jointly; a visual-only VQ codebook is too brittle for missing-modality data.

gated dense context cross-attention:
  promising JEPA-VLA/Flamingo-style direction, but should be introduced as
  context injection, not as replacement for object files.
```

### Rejected for Current Production Runs

```text
blind SAM proposals:
  empirically noisy in CALVIN task-object binding; archived.

hard sidecar mask truth:
  sidecar artifacts are weak teachers; masks may be partial/noisy.

full RGB reconstruction decoder:
  likely turns control belief router into image autoencoder and competes with
  PI0.5 action training.

raw slot count penalty:
  QASA shows count penalties can conflict with reconstruction/assignment.
  PICF uses quality selection and downstream active gating instead.

raw unnormalized slot-JEPA:
  diagnostic only; observed scale explosion makes it unsafe as loss.
```

## Actionable Conclusion

The current PICF direction is paper-consistent if the long-run recipe is:

```text
1. Keep active object files as sparse task/contact/motion owners.
2. Keep dense background/context as residual typed context, not forced slots.
3. Keep action loss at traditional/default scale.
4. Decay weak object scaffold after active ownership is healthy.
5. Monitor active/downstream overlaps instead of raw reserve overlap alone.
6. Keep predictive/object losses guarded unless normalized and matched.
```

The next architecture work should not be another generic diversity loss.  The
paper-backed next candidates are:

```text
A. gated dense-context cross-attention for residual V-JEPA/context tokens;
B. offline IsSameObject probe using sidecar/tracklet/posterior weak labels;
C. normalized matched latent prediction, with detached targets and scale
   calibration.
```

The current production priority remains the step-1500 action-dominant
continuation documented in:

```text
docs/PICF_AQR_OWM_ACTION_DOMINANT_WEIGHT_AUDIT_20260522.md
```

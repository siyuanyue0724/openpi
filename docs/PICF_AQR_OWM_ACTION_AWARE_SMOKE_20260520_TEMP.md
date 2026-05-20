# PICF-AQR-OWM Action-Aware Smoke Validation

Date: 2026-05-20

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

## Purpose

`picf_a7_active_support_dedup_300_20260520` closed the short structural
failure where context/downstream rows reused the same full support as an active
owner.  The next question is no longer whether the slot dataflow can localize
the owner under frozen policy pressure.  It is whether the same repaired slot
contract survives the first action-aware training pressure.

Run:

```text
picf_a7_actionaware_after_dedup_smoke300_20260520
```

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_after_dedup_smoke300_20260520.sh
```

## Mathematical Contract

The previous repair establishes this invariant:

```math
\max_{a \in A_t}
\operatorname{overlap}(s_i, s_a)
< \tau
\quad
\Rightarrow
i \text{ may receive downstream owner weight.}
```

If a context row duplicates an already-active owner, it remains available as
dense typed context but must not become a second action-visible owner.

The action-aware smoke keeps this invariant unchanged and adds the production
training pressure:

```text
frozen:
  V-JEPA visual encoder
  Sonata point encoder
  AnyTouch tactile encoder

trainable:
  PaliGemma / semantic-action stack
  PICF AQR/posterior/task/control adapters
  action-side heads

memory/distribution:
  training_strategy=fsdp_full_shard
  optimizer_sharding=none
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The action loss is intentionally routed with `--picf-action-prefix-stopgrad`.
This lets PI0.5/action heads adapt to PICF belief tokens without letting early
action gradients overwrite the freshly stabilized slot-binding subspace.
The base frozen-policy launcher may run under DDP when PaliGemma is frozen, but
this action-aware wrapper must use FSDP because trainable PaliGemma/action
pressure otherwise duplicates the large semantic stack on each 40GB card and
can OOM before the first metric row.

## Paper Rationale

Recent slot/VLA work supports this staged gate:

```text
SlotVLA:
  object/relation slots are useful for manipulation, but object slots need
  stable temporal/object evidence before action decoding is trusted.

STORM:
  task-aware slots should be adapted in stages around frozen foundation
  features; direct end-to-end pressure can degenerate slots.

OCRA / visuo-tactile object-centric work:
  tactile/contact evidence should attach to the manipulated object owner, not
  to a gripper-only role.

When Slots Compete:
  overlapping slots should be merged/demoted as duplicate explanations instead
  of all being treated as distinct objects.
```

PICF implements these invariants in belief-filter form rather than by copying
a reconstruction-first object-discovery model.  Full reconstruction decoders,
hard visual VQ truth, blind SAM labels, and online weak IsSameObject losses
remain rejected for the current production path.

## Acceptance Gate

Check at steps 100, 200 and 300:

```text
loss_action_default_equiv:
  should not stall or explode.

aqr_active_same_role_support_overlap_max:
  should remain in the low active-owner band; sustained high values mean
  action/semantic pressure broke object ownership.

aqr_downstream_same_role_support_overlap_max:
  should not rebound into the old 0.76+ failure band.

loss_object_explanation_point:
  should remain bounded; it must not reproduce the old monotonic rise.

posterior_recycle_rate:
  should not saturate.

anchor overlays:
  active owner should stay near the sidecar/contact mask, while gray reserve
  rows may remain as low-priority context.
```

If this smoke fails while the frozen-policy structural run passed, the next
repair target is action/semantic co-training pressure, not slot evidence
plumbing.

If this smoke passes through 300 steps, the next gate is a longer 30000-step
run with the same production trainability contract.

## 2026-05-20 OOM Guard

An initial launch with the action-aware wrapper accidentally inherited the
frozen-policy DDP distribution default.  It failed during the first forward
before any metrics were produced:

```text
failure:
  CUDA out of memory in _projective_attention_bias / projective_bias_head

root cause:
  DDP replicated a trainable PaliGemma/semantic-action stack on each 40GB card.

fix:
  keep the slot/OEML/sidecar contract unchanged;
  set TRAINING_STRATEGY=fsdp_full_shard in the action-aware wrapper;
  keep OPTIMIZER_SHARDING=none;
  set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
```

This is not a slot-math failure.  It is the expected memory contract difference
between frozen-policy structural validation and trainable semantic/action
smoke validation.

## 2026-05-20 Step 50 Check

The FSDP relaunch reached step 50 and produced the first metric row:

```text
run:
  picf_a7_actionaware_after_dedup_smoke300_20260520

runtime contract:
  world_size=2
  training_strategy=fsdp_full_shard
  trainable_scope=all
  semantic=paligemma(trainable=True)
  visual/point/tactile pretrained modules frozen
  unroll_steps=2
  burnin_steps=1
  action_prefix_stopgrad=true

step50:
  loss_total                         0.18335
  loss_alignment                     0.15730
  loss_action_default_equiv          0.10420
  loss_action_active7                0.42364
  loss_anchor_object_pull            0.30180
  loss_anchor_pv                     0.68910
  loss_object_explanation_point      2.18409
  aqr_active_same_role_support_max   0.00051
  aqr_context_same_role_support_max  0.24863
  aqr_downstream_same_role_support   0.32976
  aqr_same_role_support_overlap_max  0.46830
  posterior_recycle_rate             0.07891
  posterior_active_file_swap_rate    0.00000
  grad_norm                          0.28956
```

Interpretation:

```text
positive:
  active owner separation is still intact under PaliGemma/action pressure;
  downstream overlap stays far below the old 0.76+ failure band;
  object point loss is bounded and close to the frozen-policy structural run;
  no gradient clipping or OOM after the FSDP relaunch;
  overlay places the active owner on the sidecar/switch mask.

not yet accepted:
  this is only step50;
  action loss is early and not yet a trend;
  stable identity metrics need step100/200/300 before deciding whether action
  co-training perturbs posterior file continuity.
```

## 2026-05-20 Step 100 Check

The same FSDP run reached step 100 without OOM, NaN, or gradient clipping.

```text
step50 -> step100:
  loss_total                         0.18335 -> 0.14945
  loss_alignment                     0.15730 -> 0.13623
  loss_action_default_equiv          0.10420 -> 0.05287
  loss_action_active7                0.42364 -> 0.23818
  loss_anchor_object_pull            0.30180 -> 0.25064
  loss_anchor_pv                     0.68910 -> 0.62389
  loss_object_explanation_point      2.18409 -> 2.00278
  loss_aqr_denoising                 0.39203 -> 0.96112
  loss_binding_consistency           0.87420 -> 0.86616
  loss_pv_weak                       6.37567 -> 6.38149
  grad_norm                          0.28956 -> 0.18183
  aqr_active_same_role_support_max   0.00051 -> 0.02327
  aqr_context_same_role_support_max  0.24863 -> 0.41246
  aqr_downstream_same_role_support   0.32976 -> 0.47420
  aqr_same_role_support_overlap_max  0.46830 -> 0.90969
  posterior_recycle_rate             0.07891 -> 0.08867
  posterior_identity_switch_rate     0.13000 -> 0.10278
  posterior_identity_switch_stable   0.29750 -> 0.20417
  active_duplicate_overlap_max       0.00000 -> 0.00000
  duplicate_overlap_max              0.88554 -> 0.98132
  demoted_mass_mean                  0.62034 -> 0.18843
  object_owner_active_dist_mean_m    0.00019 -> 0.00022
```

Interpretation:

```text
positive:
  action-equivalent loss improved by roughly 2x from step50;
  active owner support overlap remains low and far from the old collapse band;
  active duplicate posterior overlap remains zero;
  object owner geometry remains on the sidecar/contact mask;
  recycle and identity-switch metrics improved instead of saturating;
  anchor_object_pull, anchor_pv, and object_explanation_point all improved.

watch:
  raw same-role support overlap rose to 0.91, but this is not the old active
  owner failure because active overlap is still 0.023 and active duplicate
  overlap is still zero;
  context/downstream overlap increased but remains below the earlier 0.76+
  failure band;
  loss_aqr_denoising rebounded, but its training weight is zero in this smoke,
  so this is a diagnostic reading, not an active optimization pressure.
```

Overlay inspection:

```text
step_000100__push_the_switch_downwards__mask_active.png
step_000100__push_the_switch_downwards__active_only.png
step_000100__push_the_switch_downwards__with_gray.png
```

The active owner remains on the green sidecar/mask region.  Gray reserve/context
rows are still visible in the full overlay, but the action-visible active path
does not duplicate the owner.  Continue this run to step 200 unless step 150
or intermediate progress shows NaN/OOM.

## 2026-05-20 Step 150 Check

Step 150 switches the sampled overlay task to `open_the_drawer`, which is a
harder geometry case than the step50/100 switch example.

```text
step50 -> step100 -> step150:
  loss_total                         0.18335 -> 0.14945 -> 0.16718
  loss_alignment                     0.15730 -> 0.13623 -> 0.15763
  loss_action_default_equiv          0.10420 -> 0.05287 -> 0.03822
  loss_action_active7                0.42364 -> 0.23818 -> 0.17263
  loss_anchor_object_pull            0.30180 -> 0.25064 -> 0.28515
  loss_anchor_pv                     0.68910 -> 0.62389 -> 0.58060
  loss_object_explanation_point      2.18409 -> 2.00278 -> 2.46495
  loss_aqr_denoising                 0.39203 -> 0.96112 -> 0.85312
  loss_binding_consistency           0.87420 -> 0.86616 -> 0.83624
  loss_pv_weak                       6.37567 -> 6.38149 -> 6.38750
  grad_norm                          0.28956 -> 0.18183 -> 3.14013
  aqr_active_same_role_support_max   0.00051 -> 0.02327 -> 0.05250
  aqr_context_same_role_support_max  0.24863 -> 0.41246 -> 0.37929
  aqr_downstream_same_role_support   0.32976 -> 0.47420 -> 0.48380
  aqr_same_role_support_overlap_max  0.46830 -> 0.90969 -> 0.98781
  posterior_recycle_rate             0.07891 -> 0.08867 -> 0.09617
  posterior_identity_switch_rate     0.13000 -> 0.10278 -> 0.07167
  posterior_identity_switch_stable   0.29750 -> 0.20417 -> 0.18917
  active_duplicate_overlap_max       0.00000 -> 0.00000 -> 0.00000
  object_owner_active_dist_mean_m    0.00019 -> 0.00022 -> 0.00026
```

Interpretation:

```text
positive:
  action continues to improve;
  active duplicate overlap remains zero;
  active support overlap remains low enough for the action-visible owner path;
  identity switch and stable identity switch both improve;
  anchor_pv and binding consistency improve.

risk:
  total/alignment rebounded relative to step100;
  object_explanation_point rebounded on drawer;
  raw overlap is again near 1.0, which means reserve/context rows still share
  broad support even though they are demoted from the active owner path;
  grad_norm jumped to 3.14 but stayed below the fixed clip threshold 5.0.
```

Overlay inspection:

```text
step_000150__open_the_drawer__mask_active.png
step_000150__open_the_drawer__active_only.png
step_000150__open_the_drawer__with_gray.png
```

The active owner remains on the drawer sidecar/mask region.  The full overlay
still contains many gray reserve/context rows around the manipulation area.
This is not the old failure where the active object owner misses the object,
but it keeps the reserve/context overlap issue open for step200/300.

## 2026-05-20 Step 200 Check

Step 200 switches to `slide_the_door_to_the_left`.

```text
step50 -> step100 -> step150 -> step200:
  loss_total                         0.18335 -> 0.14945 -> 0.16718 -> 0.25149
  loss_alignment                     0.15730 -> 0.13623 -> 0.15763 -> 0.24424
  loss_action_default_equiv          0.10420 -> 0.05287 -> 0.03822 -> 0.02899
  loss_action_active7                0.42364 -> 0.23818 -> 0.17263 -> 0.13036
  loss_anchor_object_pull            0.30180 -> 0.25064 -> 0.28515 -> 0.51948
  loss_anchor_pv                     0.68910 -> 0.62389 -> 0.58060 -> 0.64953
  loss_object_explanation_point      2.18409 -> 2.00278 -> 2.46495 -> 2.66752
  loss_aqr_denoising                 0.39203 -> 0.96112 -> 0.85312 -> 1.11906
  loss_binding_consistency           0.87420 -> 0.86616 -> 0.83624 -> 0.84142
  loss_pv_weak                       6.37567 -> 6.38149 -> 6.38750 -> 6.39339
  grad_norm                          0.28956 -> 0.18183 -> 3.14013 -> 0.08858
  aqr_active_same_role_support_max   0.00051 -> 0.02327 -> 0.05250 -> 0.02604
  aqr_context_same_role_support_max  0.24863 -> 0.41246 -> 0.37929 -> 0.29917
  aqr_downstream_same_role_support   0.32976 -> 0.47420 -> 0.48380 -> 0.37725
  aqr_same_role_support_overlap_max  0.46830 -> 0.90969 -> 0.98781 -> 0.57750
  posterior_recycle_rate             0.07891 -> 0.08867 -> 0.09617 -> 0.03684
  posterior_identity_switch_rate     0.13000 -> 0.10278 -> 0.07167 -> 0.11389
  posterior_identity_switch_stable   0.29750 -> 0.20417 -> 0.18917 -> 0.31000
  active_duplicate_overlap_max       0.00000 -> 0.00000 -> 0.00000 -> 0.00000
  object_owner_active_dist_mean_m    0.00019 -> 0.00022 -> 0.00026 -> 0.00024
```

Interpretation:

```text
strong positives:
  action-equivalent loss continues to improve monotonically;
  active owner overlap stays low;
  active duplicate overlap remains zero;
  context/downstream overlap improves versus step150;
  raw support overlap drops back from the step150 near-1.0 reserve/context peak;
  recycle rate falls instead of saturating.

remaining risk:
  total/alignment worsen because object-side auxiliaries are noisier on the
  door/handle task;
  anchor_object_pull and object_explanation_point worsen versus step100/150;
  stable identity switch rebounds at step200, so posterior file continuity is
  not accepted yet.
```

Overlay inspection:

```text
step_000200__slide_the_door_to_the_left__mask_active.png
step_000200__slide_the_door_to_the_left__active_only.png
step_000200__slide_the_door_to_the_left__with_gray.png
```

The active owner remains inside the green sidecar/mask region.  The higher
object losses are therefore not caused by an obvious active-owner miss in the
saved overlay; they are more likely task/window-dependent sidecar and point
explanation variance.  Continue to step300 for the short-run decision.

## 2026-05-20 Step 250 Interim Check

Step 250 partially reverses the step200 auxiliary rebound.

```text
step200 -> step250:
  loss_total                         0.25149 -> 0.19957
  loss_alignment                     0.24424 -> 0.19096
  loss_action_default_equiv          0.02899 -> 0.03443
  loss_action_active7                0.13036 -> 0.15335
  loss_anchor_object_pull            0.51948 -> 0.37972
  loss_anchor_pv                     0.64953 -> 0.67670
  loss_object_explanation_point      2.66752 -> 2.46504
  loss_aqr_denoising                 1.11906 -> 1.24180
  loss_binding_consistency           0.84142 -> 0.84568
  aqr_active_same_role_support_max   0.02604 -> 0.00024
  aqr_context_same_role_support_max  0.29917 -> 0.22761
  aqr_downstream_same_role_support   0.37725 -> 0.29126
  aqr_same_role_support_overlap_max  0.57750 -> 0.38530
  posterior_recycle_rate             0.03684 -> 0.06897
  posterior_identity_switch_rate     0.11389 -> 0.10056
  posterior_identity_switch_stable   0.31000 -> 0.26500
  active_duplicate_overlap_max       0.00000 -> 0.00000
```

Interpretation:

```text
positive:
  the step200 reserve/context rebound did not persist;
  active-owner overlap is again effectively zero;
  downstream overlap is the lowest since step50;
  object pull and object point losses improve versus step200.

watch:
  action is slightly worse than step200 but still much better than step50;
  anchor_pv and denoising diagnostics are not improving monotonically;
final step300 is still required before promoting this to a longer run.
```

## 2026-05-20 Step 300 Final Check

The 300-step FSDP smoke completed and saved checkpoint `300`.

```text
step50 -> step100 -> step150 -> step200 -> step250 -> step300:
  loss_total
    0.18335 -> 0.14945 -> 0.16718 -> 0.25149 -> 0.19957 -> 0.20609
  loss_alignment
    0.15730 -> 0.13623 -> 0.15763 -> 0.24424 -> 0.19096 -> 0.19795
  loss_action_default_equiv
    0.10420 -> 0.05287 -> 0.03822 -> 0.02899 -> 0.03443 -> 0.03255
  loss_action_active7
    0.42364 -> 0.23818 -> 0.17263 -> 0.13036 -> 0.15335 -> 0.14559
  loss_anchor_object_pull
    0.30180 -> 0.25064 -> 0.28515 -> 0.51948 -> 0.37972 -> 0.41526
  loss_anchor_pv
    0.68910 -> 0.62389 -> 0.58060 -> 0.64953 -> 0.67670 -> 0.67010
  loss_object_explanation_point
    2.18409 -> 2.00278 -> 2.46495 -> 2.66752 -> 2.46504 -> 2.19150
  loss_aqr_denoising
    0.39203 -> 0.96112 -> 0.85312 -> 1.11906 -> 1.24180 -> 1.32045
  loss_binding_consistency
    0.87420 -> 0.86616 -> 0.83624 -> 0.84142 -> 0.84568 -> 0.83391
  loss_pv_weak
    6.37567 -> 6.38149 -> 6.38750 -> 6.39339 -> 6.39534 -> 6.39466
  aqr_active_same_role_support_max
    0.00051 -> 0.02327 -> 0.05250 -> 0.02604 -> 0.00024 -> 0.00105
  aqr_context_same_role_support_max
    0.24863 -> 0.41246 -> 0.37929 -> 0.29917 -> 0.22761 -> 0.24725
  aqr_downstream_same_role_support
    0.32976 -> 0.47420 -> 0.48380 -> 0.37725 -> 0.29126 -> 0.27396
  aqr_same_role_support_overlap_max
    0.46830 -> 0.90969 -> 0.98781 -> 0.57750 -> 0.38530 -> 0.37281
  posterior_recycle_rate
    0.07891 -> 0.08867 -> 0.09617 -> 0.03684 -> 0.06897 -> 0.08178
  posterior_identity_switch_rate
    0.13000 -> 0.10278 -> 0.07167 -> 0.11389 -> 0.10056 -> 0.10556
  posterior_identity_switch_stable
    0.29750 -> 0.20417 -> 0.18917 -> 0.31000 -> 0.26500 -> 0.28083
  active_duplicate_overlap_max
    0.00000 -> 0.00000 -> 0.00000 -> 0.00000 -> 0.00000 -> 0.00000
```

Final short-run verdict:

```text
passes this gate:
  trainable PaliGemma/action pressure does not break active-owner dedup;
  active duplicate overlap remains zero at every logged step;
  active same-role support overlap returns to near zero by step300;
  downstream support overlap ends lower than step50;
  action-equivalent loss improves substantially versus step50;
  recycle does not saturate;
  saved overlays keep the active owner on the sidecar/contact mask.

does not prove yet:
  object auxiliary losses are not monotonic and remain task-window dependent;
  inactive/reserve posterior duplicate overlap remains high by design/demotion,
  so raw overlap alone is not an acceptance metric;
  stable identity switch is improved versus step50 at step100/150 but rebounds
  later, so posterior file continuity still needs a longer run;
  this smoke does not replace CALVIN/video acceptance.
```

Conclusion:

```text
The repaired active-support/action-aware contract is valid enough to proceed to
the next longer co-training gate.  The next run should keep this exact slot
contract and use the production FSDP profile.  Do not add another slot module
based only on raw same-role overlap, because the active path and overlays do
not show the old failure.
```

## 2026-05-20 30K Follow-Up Launch Contract

The maintained long-run wrapper is:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_ownerdirect_long30k_20260520.sh
```

It changes only run length and experiment naming relative to the passed
action-aware smoke:

```text
num_train_steps=30000
save_interval=2500
keep_last_checkpoints=3
anchor_overlay_interval=50
log_interval=50
```

It deliberately keeps:

```text
PaliGemma/action trainable;
V-JEPA/Sonata/AnyTouch pretrained modules frozen;
slot_jepa/support_pred/binding_consistency/aqr_denoising disabled;
the same sidecar/OEML/owner-direct transport contract.
```

Do not mix the dense-background-context follow-up into this run.  That is a
separate architecture experiment documented in
`docs/PICF_AQR_OWM_30K_VALIDATION_AND_DENSE_CONTEXT_PLAN_20260520_TEMP.md`.

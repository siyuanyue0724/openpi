# PICF-AQR-OWM Owner-Gate Recycle Diagnosis

Status: temp diagnosis for the 2026-05-15 A7 `owner_gate_cotrain_u2b1_a05` long-run.

This note separates two failure modes that were previously conflated:

1. **AQR support ownership collapse**: multiple same-role graph anchors attend to the same visual/object-core evidence.
2. **Posterior object-file instability**: persistent posterior slots recycle/reset or switch identity even when active graph supports are separated.

The current evidence says the owner-active gate improves the first problem but has not yet proven the second problem is fixed.

## Current Run Evidence

Run:

```text
picf_a7_owner_gate_cotrain_u2b1_a05_30000_20260515
```

Compared against the previous A7 long run:

```text
picf_a7_dustbin_router_cotrain_u2b1_a05_30000_d163d18_long30k
```

Step 50 to 100 trend in the owner-gate run:

```text
loss_action_default_equiv:              0.1271 -> 0.0820
loss_action_active7:                    0.4471 -> 0.3527
loss_alignment:                         1.2336 -> 1.1194

aqr_active_same_role_support_overlap:   0.0430 -> 0.0603
aqr_same_role_support_overlap:          0.1012 -> 0.1437
aqr_active_same_role_object_core:        0.0523 -> 0.1138
aqr_same_role_object_core:               0.0613 -> 0.1805

aqr_active_anchor_count:                 8.465 -> 8.095
aqr_effective_anchor_count:              8.303 -> 7.898

posterior_recycle_rate:                 0.4491 -> 0.6045
posterior_identity_switch_rate:          0.6489 -> 0.6544
posterior_residual_summary_norm:         380.7 -> 669.5
posterior_dustbin_mass_raw:              7.543 -> 7.941
posterior_address_update_rate_mean:      0.0222 -> 0.0158
```

The improvement is real on action/alignment and active support. The unresolved issue is also real: posterior recycle and residual scale worsen while active support remains separated.

## Code Follow-Through

The owner-active gate is connected to posterior binding.

```text
_observation_owner_active_from_graph(...)
  graph.anchor_active + graph_assignment
  -> obs.owner_active

_posterior_update(...)
  bind_logits = _binding_logits(...)
  owner_bias = _posterior_owner_active_binding_bias(obs_anchors)
  bind_logits += owner_bias
  binding_raw = _sinkhorn_dustbin(bind_logits)
```

Inactive owner rows receive a large negative binding bias before Sinkhorn, so they should not directly update persistent object files. This fixes the measurement eligibility leak.

However, posterior recycle is a separate trust/reset channel:

```text
support_raw = binding_raw[:-1]
dustbin_raw = binding_raw[-1]
slot_residual_summary = cond(support_raw) @ obs_tokens
recycle_residual_summary = layernorm(slot_residual_summary)
recycle = sigmoid(recycle_head([h_prior, support_mass_raw, var_prior, recycle_residual_summary, alpha_prior]))

bar_h  = (1 - recycle) h_prior  + recycle res_h
bar_c  = (1 - recycle) c_prior  + recycle res_c
bar_mu = (1 - recycle) mu_prior + recycle res_mu
```

Therefore:

```text
owner gate controls which observation rows may bind as object evidence;
recycle controls whether the posterior object file trusts its prior or resets from residual evidence.
```

Those are mathematically distinct.

## Mathematical Diagnosis

Let `i` index observation anchors and `j` index persistent posterior slots.

Owner gate changes the binding logits:

```math
L_{j,i}' = L_{j,i} + b_i,
```

where:

```math
b_i = 0       if owner_active_i >= threshold,
b_i = -M      otherwise.
```

The posterior support assignment is:

```math
B = SinkhornDustbin(L').
```

This can make active support overlap healthy:

```math
overlap(B_{active}) \ll 1.
```

But the recycle gate is:

```math
r_j = \sigma f_\theta(h_j^-, m_j^{raw}, v_j^-, LN(z_j^{res}), \alpha_j^-).
```

The posterior state update is:

```math
\bar{s}_j = (1-r_j)s_j^- + r_j s_j^{res}.
```

So even if support ownership is correct, the object file can still become unstable when `r_j` grows. High recycle then suppresses address update:

```math
\rho_j \propto support_j (1-r_j) exp(-innovation_risk),
```

which matches the observed step100 drop:

```text
posterior_address_update_rate_mean: 0.0222 -> 0.0158
```

This explains the apparently contradictory metrics:

```text
active support still separated,
but posterior identity/recycle worsens.
```

## Literature Cross-Check

Object-binding in large ViTs: Li et al. (NeurIPS 2025) report that IsSameObject is decoded by a quadratic similarity probe and is encoded in a low-dimensional subspace over object features. This supports the current `binding_signature_proj` direction: binding should be pairwise/subspace-based, not just hidden cosine or a scalar diversity penalty.

Slot Contrast (CVPR 2025) emphasizes that object-centric video methods need temporally consistent slots and use previous slots as current initialization. It directly supports treating posterior slots as object files whose identity should persist unless evidence is genuinely incompatible.

MetaSlot (NeurIPS 2025) identifies a fixed-slot-count failure: one object can be represented as multiple slots when object count varies. It supports capacity-aware active/reserve selection, but it also implies that duplicate slots must be handled as inactive/reserve candidates rather than persistent object files.

SlotVLA (ICRA 2026) supports object-relation slots for manipulation, but it uses richer object-centric annotations and temporal tracking. It does not prove that CALVIN without extra instance labels can solve ordinal/fine-instance binding purely through action loss.

## Current Root Cause Hypothesis

The current issue is not ordinary same-role support collapse. It is:

```text
posterior trust/recycle instability after owner-gated measurement routing.
```

The likely mechanism is:

1. Active AQR owners are selected correctly enough to keep active support overlap low.
2. Reserve/inactive rows are pushed away from persistent binding and into dustbin.
3. The recycle head still interprets slot-local residual summaries and raw support conditions as reset evidence.
4. Action/alignment gradients lower action loss but increase posterior residual pressure.
5. Higher recycle reduces address updates, preventing stable identity consolidation.

This is why action can improve while recycle worsens.

## What Would Be a Root Repair

Do not add a stronger support diversity loss first. That would target the already-improved layer.

The next repair should make recycle a calibrated object-file trust decision:

```math
r_j^{eff}
=
r_j^{raw}
\cdot
(1 - stopgrad(T_j)),
```

with:

```math
T_j =
owner\_support_j
\cdot
binding\_margin_j
\cdot
\alpha_j^-
\cdot
exp(-\lambda \, innovation\_risk_j).
```

Interpretation:

```text
If a slot has a confident active owner, high binding margin, and reasonable prior existence,
do not recycle it aggressively.

If a slot has no owner, low margin, low support, or high innovation,
allow recycle/reset.
```

This is not a cosmetic clamp. It follows the belief-filter interpretation:

```text
recycle/reset is birth-death or object-file trust,
not a generic residual shortcut for action optimization.
```

Acceptance conditions for this repair:

```text
active same-role support overlap max     < 0.15 early, < 0.30 tolerated
raw same-role support overlap max        must not trend toward 0.5+
posterior_recycle_rate                   should not stay > 0.70 after warmup
posterior_address_update_rate_mean       should not collapse toward 0
posterior_identity_switch_rate_stable    should trend down or remain bounded
loss_action_default_equiv                should keep falling without recycle saturation
```

## Step150/200 Check To Run Before Editing

The previous dustbin-router run began showing stronger overlap after step150/200. The owner-gate run must be checked at the same points.

If step150/200 shows:

```text
active overlap stays low but recycle stays high:
  fix recycle trust gating.

active overlap also rises:
  inspect owner_active threshold/active-slot capacity and object-core overlap.

  action falls while recycle and raw dustbin rise:
  action pressure is using posterior reset as a shortcut; apply recycle trust gating
  and consider lower action weight ramp.
```

## Step150 Update: Exact Failure Mode

The step150 row is decisive enough to refine the diagnosis:

```text
loss_action_default_equiv:
  previous dustbin long run: 0.0807
  owner-gate run:            0.1010

active same-role support overlap:
  previous: 0.1572
  owner:    0.2111

raw same-role support overlap:
  previous: 0.2419
  owner:    0.4672

active object-core overlap:
  previous: 0.2572
  owner:    0.3336

raw object-core overlap:
  previous: 0.3552
  owner:    0.4967

posterior_recycle_rate:
  previous: 0.4715
  owner:    0.2514

posterior_dustbin_mass_raw:
  previous: 1.8803
  owner:    9.1413

posterior_support_mass_raw_mean:
  previous: 1.7650
  owner:    0.8573
```

Therefore the current failure is not "recycle remains too high". Recycle has
fallen by step150. The failure is:

```text
hard top-1 owner gating makes too many compatible observation rows become
dustbin residual; raw support mass collapses, raw dustbin mass explodes, and
active/raw support overlaps rise again.
```

The code-level cause is in the first owner-gate implementation:

```text
for each active graph column:
  choose one unique observation row;
  set only that row owner_active=1;
  all other compatible rows receive owner_active=0.
```

This is too hard. It treats "not the best unique owner for this active graph
column" as "not object evidence". In a belief filter this is mathematically
wrong: duplicate or secondary rows should have lower measurement reliability,
not become unexplained novel residual.

## Root Repair Deployed Locally

The corrected interpretation is:

```math
owner_i =
\max\left(
  \sum_{a \in active} A_{i,a},
  \mathbf{1}[i \text{ is the selected unique peak owner}]
\right).
```

So `owner_active` is now a continuous measurement reliability score. The
clearest active owners still receive score `1.0`, but secondary compatible rows
retain their active-assignment mass instead of being forced to zero.

Posterior recycle/birth redistribution is also reliability-gated:

```math
dustbin^{trusted}_i = dustbin_i \cdot owner_i.
```

and the recycle-share redistribution uses `dustbin^{trusted}`, not all dustbin
rows. This means:

```text
inactive/reserve duplicate rows do not update object files;
inactive/reserve duplicate rows also do not re-enter object files through
recycle_share * dustbin;
only owner-compatible residual can act as birth/reset evidence.
```

This is the mathematically consistent object-file semantics:

```text
owner score = measurement reliability,
dustbin = unexplained reliable evidence,
recycle = birth/death or trust decision,
not a path for duplicate reserve rows to reset every slot.
```

## Restart Deployment: Soft Owner Reliability

Local and A7 verification before restart:

```text
python -m py_compile:
  pipeline / pipeline_test / owner audit / verifier / trainer PASS

PYTHONPATH=src python scripts/verify_picf_owm_contract.py:
  34/34 PASS

PYTHONPATH=src python scripts/picf_owm_owner_gate_followthrough_audit.py --fail-on-fail:
  12/12 PASS

PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail:
  PASS

PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail:
  PASS

PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS

uv run pytest -q \
  src/openpi/picf/core/pipeline_test.py::test_observation_owner_active_returns_soft_measurement_reliability \
  src/openpi/picf/core/pipeline_test.py::test_posterior_owner_active_binding_bias_masks_reserve_rows:
  2 passed
```

A7 restart:

```text
session:
  picf_a7_owner_soft_long30k

run:
  picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515

log:
  /mnt/picf_run_logs/picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515.log

metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515/metrics.jsonl

run environment:
  code directory = /root/openpi_posterior_vla_clean
  python env = /root/openpi/.venv
  PYTHONPATH = /root/openpi_posterior_vla_clean/src
```

The environment detail is intentional. `/root/openpi_posterior_vla_clean/.venv`
does not contain `transformers`, while `/root/openpi/.venv` is the environment
used by the successful earlier A7 runs and contains `transformers==4.53.2`.

Tail commands:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t picf_a7_owner_soft_long30k
tail -f /mnt/picf_run_logs/picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515/metrics.jsonl
ls -lh /mnt/checkpoints/picf_core/picf_core/picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515/anchor_overlays | tail
```

First acceptance rows to inspect:

```text
step50 / step100 / step150:
  loss_action_default_equiv
  loss_action_active7
  loss_anchor_pv
  loss_mapg_cycle
  aqr_active_same_role_support_overlap_max
  aqr_same_role_support_overlap_max
  posterior_dustbin_mass_raw
  posterior_support_mass_raw_mean
  posterior_recycle_rate
  posterior_identity_switch_rate_stable
  posterior_owner_active_score_mean
  posterior_owner_active_eligible_fraction
```

Expected effect if the root diagnosis is correct:

```text
posterior_dustbin_mass_raw should not repeat the step150 spike to ~9.
posterior_support_mass_raw_mean should recover above the hard-gate run's ~0.86.
active and raw support overlap should not rise as sharply as the hard-gate run.
action loss may be slightly slower than the old dustbin run, but should not be
paid for by posterior dustbin/recycle corruption.
```

## 2026-05-15 Soft Owner Run: Step50-750 Check

Run:

```text
picf_a7_owner_soft_reliable_cotrain_u2b1_a05_30000_20260515
```

Runtime status:

```text
tmux:
  picf_a7_owner_soft_long30k still running

latest checked row:
  step750

speed:
  about 0.043 steps/sec, 23.3 sec/step

anchor overlays:
  step_000100.png/json through step_000700.png/json produced
```

Key metric trajectory:

```text
step:                           50      100     150     200     250     300     400     500     600     700     750
loss_action_default_equiv:      0.128   0.0846  0.0811  0.0715  0.0668  0.0688  0.0713  0.0639  0.0688  0.0629  0.0588
loss_action_active7:            0.449   0.359   0.359   0.324   0.298   0.298   0.318   0.289   0.311   0.285   0.266
loss_alignment:                 1.229   1.120   1.103   1.057   1.082   1.128   1.228   1.152   1.405   1.519   1.565
loss_anchor_pv:                 2.773   2.719   2.732   2.501   2.607   2.647   3.227   3.155   3.973   4.700   4.652
loss_mapg_cycle:                0.365   0.369   0.379   0.407   0.442   0.476   0.485   0.478   0.485   0.487   0.486

active support overlap max:     0.044   0.067   0.083   0.163   0.362   0.618   0.519   0.576   0.480   0.368   0.395
raw support overlap max:        0.105   0.201   0.197   0.435   0.749   0.991   0.987   0.998   0.999   1.000   1.000
active object-core overlap max: 0.067   0.092   0.138   0.195   0.295   0.200   0.135   0.168   0.111   0.051   0.044
raw object-core overlap max:    0.073   0.104   0.156   0.234   0.520   0.793   0.844   0.865   0.898   0.963   0.972

posterior_dustbin_mass_raw:     1.888   1.901   1.889   1.887   1.884   1.880   1.881   1.880   1.881   1.880   1.880
posterior_support_mass_raw:     1.764   1.762   1.764   1.764   1.765   1.765   1.765   1.765   1.765   1.765   1.765
posterior_recycle_rate:         0.237   0.330   0.815   0.596   0.375   0.607   0.968   0.808   0.0367  0.138   0.366
stable identity switch:         0.242   0.287   0.247   0.229   0.185   0.153   0.152   0.141   0.172   0.151   0.144
owner score mean:               1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000
owner eligible fraction:        1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000
```

Strict interpretation:

```text
The soft-owner patch fixed the specific step150 dustbin spike:
  previous hard-owner step150 dustbin_raw ~= 9.14
  current soft-owner step150 dustbin_raw ~= 1.89

It also restored raw support mass:
  previous hard-owner step150 support_raw_mean ~= 0.857
  current soft-owner step150 support_raw_mean ~= 1.764

However, it over-corrected the owner gate:
  posterior_owner_active_score_mean = 1.0
  posterior_owner_active_eligible_fraction = 1.0

This proves that using the post-mask/post-renormalization `graph_assignment`
as a soft owner reliability is mathematically insufficient. `_mapg_slot_assignment`
first restricts candidates to active graph anchors and then renormalizes each
row, so the sum over active columns is almost always one. The resulting
owner_active no longer distinguishes active object owners from reserve rows.
```

Anchor overlay reading:

```text
step100:
  active graph anchors are reasonably separated.
  role1 min active distance ~= 19.7 px,
  role3 min active distance ~= 39.7 px.

step300:
  some active same-role graph anchors become close.
  role2 min active distance ~= 7.6 px,
  role3 min active distance ~= 12.7 px.

step600:
  active capacity expands to 12 graph anchors, but same-role active distances
  include close pairs:
    role1 min ~= 11.1 px,
    role3 min ~= 9.4 px.

step700:
  active graph anchors recover separation:
    role2 min ~= 65.5 px,
    active overlap max ~= 0.368.
```

So this is not a simple "all anchors physically collapsed forever" failure.
It is an unstable owner/reserve selection problem: active owners can recover,
but raw/reserve candidates still converge to duplicated support and posterior
trust remains oscillatory.

Mathematical conclusion:

```text
The current owner score is computed after row normalization:

  A_post[i, active].sum() = 1

so:

  owner_i = max(A_post[i, active].sum(), unique_peak_i) = 1

for essentially every observation row. This repairs the hard-gate dustbin
misclassification, but removes the owner/reserve distinction.
```

The correct next repair must compute owner reliability from a pre-renormalized
or non-normalized objectness/novelty score:

```math
owner_i =
\sigma\left(
  w_s \, support\_mass_i
  + w_m \, margin_i
  + w_g \, geometry\_valid_i
  + w_n \, novelty_i
  - w_d \, duplicate_i
\right)
```

where:

```text
support_mass:
  how much typed evidence the row actually explains before active-only
  renormalization.

margin:
  winner-vs-runner-up graph owner confidence before row renormalization.

geometry_valid:
  whether the row has a valid physical/point estimate.

novelty:
  distance from already selected same-role active owners in geometry and
  binding-signature space.

duplicate:
  same-role high-overlap with a stronger selected owner.
```

This is the same structural lesson as MetaSlot / DIAS / dynamic-query DETR:
when capacity exceeds object count, duplicate hypotheses must be retired or
kept as no-object reserve, not merely normalized into object owners.

Operational decision:

```text
Do not accept this soft-owner run as 30k-ready.

It is useful evidence:
  hard owner gate caused the dustbin/support collapse;
  post-normalization soft owner removes the dustbin bug but degenerates into
  no owner gate;
  the remaining root problem is pre-normalization owner scoring / reserve
  duplicate retirement.

Continue only if more curve evidence is desired. The next code repair should
target owner reliability before `_mapg_slot_assignment` row normalization or
explicit reserve retirement after active owner selection.
```

## 2026-05-15 step750 repair: margin + novelty owner reliability

The step750 row made the failure non-ambiguous:

```text
loss_action_default_equiv ~= 0.0588
posterior_dustbin_mass_raw ~= 1.88
posterior_support_mass_raw_mean ~= 1.76
posterior_owner_active_score_mean = 1.0
posterior_owner_active_eligible_fraction = 1.0
raw same-role support overlap max ~= 1.0
active same-role support overlap max ~= 0.395
```

So the hard-gate dustbin spike was fixed, but the soft owner score became a
no-op. The exact mathematical bug was:

```math
A_{post}[i,k] \ge 0,\quad \sum_{k\in active(role(i))} A_{post}[i,k]=1
```

Therefore:

```math
owner_i=\sum_{k\in active}A_{post}[i,k]=1
```

for almost every row. This is not an empirical guess; it follows directly from
the row-stochastic assignment contract.

The deployed repair changes the owner variable from row-sum mass to a
measurement-eligibility estimator:

```math
p_i=\max_{k\in\mathcal A} A_{ik}
```

```math
m_i={p_i-p_i^{(2)} \over \max(p_i,\epsilon)}
```

```math
d_i=\max_{u\in U, role(u)=role(i)}
\left[
  \exp\left(-{\|x_i-x_u\|^2\over 2\sigma^2}\right),
  \langle b_i,b_u\rangle_+,
  \cos(point_i,point_u)_+
\right]
```

```math
owner_i =
\begin{cases}
1, & i\in U\\
p_i m_i (1-d_i), & i\notin U
\end{cases}
```

where `U` is the greedy one-row-per-active-graph-owner peak set. This is the
fixed-capacity object-file interpretation:

```text
unique peaks:
  persistent object-file measurements.

high-margin, novel secondary rows:
  may remain eligible if they explain distinct evidence.

same-role duplicates near a stronger owner:
  reserve/dustbin rows; they should not update persistent posterior slots.
```

This follows the 2025 object-binding / OCL literature direction rather than
adding a scalar patch:

```text
Does Object Binding Naturally Emerge in Large Pretrained ViTs:
  same-object evidence is pairwise/quadratic and can live in a low-dimensional
  binding subspace; therefore owner novelty should use projected binding
  signatures, not raw hidden cosine only.

Temporally Consistent Object-Centric Learning by Contrasting Slots:
  object files need temporal consistency and contrast against other slots;
  duplicate same-role rows must not all be treated as valid object identities.

MetaSlot:
  when fixed slot capacity exceeds object count, duplicate slots should be
  pruned/dead/reserve rather than all kept as live object slots.
```

Local verification after the repair:

```text
python -m py_compile:
  pipeline.py / pipeline_test.py / owner audit PASS

PYTHONPATH=src python scripts/picf_owm_owner_gate_followthrough_audit.py --fail-on-fail:
  12/12 PASS

uv run pytest -q \
  src/openpi/picf/core/pipeline_test.py::test_observation_owner_active_uses_margin_and_novelty_not_row_sum \
  src/openpi/picf/core/pipeline_test.py::test_posterior_owner_active_binding_bias_masks_reserve_rows:
  2 passed
```

Restart acceptance for the next 30k run:

```text
posterior_owner_active_eligible_fraction:
  must be < 1.0 by step50/100 and should remain well below 1.0.

posterior_dustbin_mass_raw:
  must not repeat the hard-gate spike near 9.

raw same-role support overlap:
  should not monotonically return to 1.0.

active same-role support overlap:
  preferred < 0.50, warning > 0.70.

loss_action_default_equiv:
  should continue decreasing, but action decrease alone is not acceptance.
```

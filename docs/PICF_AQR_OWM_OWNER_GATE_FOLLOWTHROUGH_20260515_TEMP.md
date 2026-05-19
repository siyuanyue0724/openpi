# PICF-AQR-OWM Owner/Reserve Follow-Through Audit

Date: 2026-05-15

Status: TEMP strict audit record for the owner/reserve posterior gate that
restarts the A7 30000-step behavior run.

Canonical entry: [`src/openpi/picf/README_v2.2.md`](../src/openpi/picf/README_v2.2.md)

Dedicated audit script:

```bash
PYTHONPATH=src python scripts/picf_owm_owner_gate_followthrough_audit.py --fail-on-fail
```

## 1. Failure Being Repaired

The rejected A7 run
`picf_a7_dustbin_router_cotrain_u2b1_a05_30000_d163d18_long30k`
showed a split failure:

```text
active graph anchors:
  visually separated at step100/200/300

raw support pool:
  aqr_same_role_support_overlap_max ~= 0.985 at step300

posterior object files:
  scene slots still physically clustered; role1 min pair distance about 4.69 px
```

This means the active graph filter was not the full causal repair. It selected
which graph anchors are object owners, but the posterior still received
duplicate observation rows as normal object-file measurements.

## 2. Mathematical Diagnosis

The belief update should be:

```math
b_t(s)
\propto
p(o_t \mid s_t, owner(o_t)=1)
\int p(s_t \mid s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds_{t-1}.
```

The invalid behavior was closer to:

```math
b_t(s)
\propto
p(o_t^{owner}, o_t^{reserve} \mid s_t)
\int p(s_t \mid s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds_{t-1}.
```

That second form lets reserve hypotheses update persistent object slots. It is
mathematically inconsistent with object files: reserve rows may explain excess
evidence or capacity, but they are not object identities.

The repair is:

```math
owner_i =
\operatorname{GreedyUniqueMatch}(A_{obs\rightarrow graph}, active_{graph})_i
```

followed by:

```math
\ell^{post}_{j,i}
\leftarrow
\ell^{post}_{j,i}
+
\begin{cases}
0, & owner_i \ge \tau_{owner} \\
B_{reserve}, & owner_i < \tau_{owner}
\end{cases}
```

with:

```text
tau_owner = 0.25
B_reserve = -1e4
```

This is not an auxiliary loss. It is a measurement-eligibility constraint inside
the posterior belief filter.

## 3. Why Naive Assignment Mass Is Wrong

The tempting shortcut:

```math
owner = A_{obs\rightarrow graph} active_{graph}
```

is not valid here. `_mapg_slot_assignment` masks inactive graph columns and then
renormalizes each observation row. After that operation, almost every row can
receive owner mass close to one even when the row is a duplicate of another
active owner. The repaired logic must select a unique observation row per active
graph owner.

The implemented invariant is:

```text
for each active graph owner, select one unused observation row with the largest
assignment mass; if a role would otherwise have no owner row, keep one best row
as a non-empty fallback.
```

## 4. Data Follow-Through

The strict dataflow is:

```text
PicfAnchorPriorGraphState.anchor_active
  -> _observation_owner_active_from_graph(...)
  -> PicfObservationAnchorState.owner_active
  -> _posterior_owner_active_binding_bias(...)
  -> _posterior_update(... bind_logits += owner_bias ...)
  -> Sinkhorn-dustbin posterior binding
  -> posterior object files
  -> debug metrics
  -> training metrics/evidence bundle/README acceptance
```

Code references:

```text
src/openpi/picf/core/config.py
  posterior_owner_active_gate_enabled = True
  posterior_owner_active_min = 0.25
  posterior_owner_active_bias = -1e4

src/openpi/picf/core/contracts.py
  PicfObservationAnchorState.owner_active

src/openpi/picf/core/pipeline.py
  _observation_owner_active_from_graph(...)
  _posterior_owner_active_binding_bias(...)
  _posterior_update(... owner_bias ...)
  posterior_owner_active_* debug metrics

scripts/picf_core_train.py
  CLI flags, startup log, OWM metrics

scripts/picf_owm_evidence_bundle.py
  reviewer handoff keys
```

## 5. Cross-Layer Interaction Checks

The owner gate is accepted only if all of these interactions hold:

```text
Graph layer:
  active slots can still exist as capacity-limited object owners.

Observation layer:
  active ownership is converted to observation-row eligibility, not graph-column
  eligibility.

Posterior layer:
  reserve rows receive a large negative binding bias before softer VL/MAPG/
  occupancy priors and before Sinkhorn-dustbin normalization.

Action layer:
  PI0.5 action path remains unchanged; the repair changes which measurements
  update posterior object files, not the action generator.

Diagnostics layer:
  posterior_owner_active_eligible_fraction and owner score metrics must be
  emitted; action loss alone is not acceptance.

Documentation layer:
  README_v2.2 and experiment report must state that this is a measurement
  eligibility repair, not a sub-token/ordinal grounding proof.
```

## 6. Paper Grounding

The repair follows a shared 2025 direction, without importing another model as a
large module:

```text
Object Binding in pretrained ViTs:
  pairwise same-object structure is extractable but fragile; object identity
  cannot be assumed from saliency alone.

MetaSlot:
  fixed slot counts create duplicate-slot failure when object count is lower
  than capacity; duplicate/removable slot handling is part of the object-centric
  contract.

DIAS / Slot Attention with Re-Initialization and Self-Distillation:
  redundant slots can compete with informative slots and should be treated as
  reserve/reinitialized capacity rather than stable object files.
```

The PICF-specific conclusion is:

```text
fixed-capacity anchors need explicit owner/reserve semantics at posterior
measurement time.
```

## 7. Script Evidence

Required local script suite after this repair:

```bash
python -m py_compile \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/paligemma/wrapper.py \
  scripts/picf_core_train.py \
  scripts/verify_picf_owm_contract.py \
  scripts/picf_owm_professor_grade_audit.py \
  scripts/picf_owm_owner_gate_followthrough_audit.py

PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_owner_gate_followthrough_audit.py --fail-on-fail

PYTHONPATH=src uv run pytest -q \
  src/openpi/picf/core/pipeline_test.py::test_observation_owner_active_returns_soft_measurement_reliability \
  src/openpi/picf/core/pipeline_test.py::test_posterior_owner_active_binding_bias_masks_reserve_rows \
  src/openpi/picf/paligemma/wrapper_test.py::test_build_paligemma_with_expert_omits_chunk_kwargs_for_legacy_constructor
```

## 8. Runtime Acceptance

The restarted A7 long run is accepted only if by the first stable window:

```text
posterior_owner_active_gate_enabled = 1
posterior_owner_active_eligible_fraction < 1
aqr_active_same_role_support_overlap_max remains controlled
posterior overlays do not collapse scene object slots to a few pixels
loss_action_default_equiv is compared on the 4-22 ablation scale
```

If action loss improves but owner metrics show reserve leakage, the run is
rejected. The acceptance target is the belief router plus action path, not
action loss alone.

## 9. Known Limits

This repair does not create missing object evidence. It does not solve
sub-token ordinal grounding such as "the fourth chopstick" without enough
observable evidence or rank supervision. It also does not turn optional
tracklet/proposal fields into real data if the dataset does not provide them.

Those are separate scientific limits. The owner/reserve gate repairs a specific
posterior measurement leak discovered by the A7 long-run overlays.

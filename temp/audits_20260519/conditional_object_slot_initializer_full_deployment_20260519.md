# Conditional Object-Slot Initializer Deployment

Date: 2026-05-19

Status: implementation deployed locally; remote 1000-step anchor-only probe is
the acceptance gate.

## Decision

The current failure is not missing sidecar evidence.  The contact/task sidecar
can identify a task-object mask, and owner transport can write the accepted
measurement into a posterior file.  The remaining failure is earlier:

```text
learned AQR query -> dense point reader -> sidecar/object pull
```

can read the wrong dense point region before the mask/object reference has
conditioned the object row.  Once the point read is wrong, late pull losses and
posterior precision fusion are trying to undo an already wrong measurement.

The repair is therefore not another auxiliary loss.  It is a conditional
object-slot initializer:

```text
task/contact mask sidecar
  -> candidate/background assignment
  -> bounded proposal-anchor seed
  -> query token interpolation + point-reader logit bias
  -> dense point/V-JEPA/tactile reads
  -> object candidate owner transport
  -> posterior file update
```

This is the PICF-compatible analogue of modern object-centric initializers:
condition the slot before iterative attention, keep dense memory available, and
make the proposal/mask a measurement prior rather than truth.

## Paper Alignment

The relevant paper-code/math principles are:

- MetaSlot, 2025: fixed slot counts duplicate or split objects; object-centric
  routing needs adaptive object-quality and competition rather than all slots
  owning all evidence.  Source: https://arxiv.org/abs/2505.20772
- Object Binding in pretrained ViTs, 2025: same-object information exists in
  a subspace but is not guaranteed by raw token cosine; binding needs an
  explicit pairwise/compatibility mechanism.  Source:
  https://arxiv.org/abs/2510.24709
- QASA, 2026: quality-guided K-adaptive slot attention reinforces that slot
  existence/quality should gate object ownership instead of forcing a fixed
  set of active slots.  Source: https://arxiv.org/abs/2601.12936
- STORM and SlotVLA, 2026: manipulation benefits from task-aware object-centric
  slots; dense VLA tokens alone are not a reliable object-binding mechanism.
  Sources: https://arxiv.org/abs/2601.20381 and
  https://arxiv.org/abs/2511.06754
- OCRA and visuo-tactile object-centric work, 2026: tactile/contact should
  attach to the contacted object representation, not become an independent
  gripper-owned object.  Source: https://arxiv.org/abs/2603.14401

PICF should not wholesale transplant a standalone MetaSlot or QASA decoder:
those methods target image-centric object discovery, while PICF is a
recurrent typed belief filter whose posterior update, PI0.5 action path,
V-JEPA temporal memory, point cloud, tactile memory, and missing-modality
contract must remain intact.  The transferable mechanism is conditional
initialization plus competition/background residual, not their entire
reconstruction decoder.

## Math Contract

Let a sidecar/object candidate mask induce a point support distribution:

```math
m_{p,i} = p(x_i \mid c_p)
```

where `c_p` is an inspected task/contact candidate and `x_i` is a point token.
Let current slot rows be `j`, with eligible object rows selected by role and
query type.  Candidate assignment remains soft:

```math
A_{j,p} =
\operatorname{softmax}_{j \cup bg}
\left(
  \frac{s_{j,p}}{\tau},
  \log \pi_{bg} - \lambda_q q_p
\right)
```

The proposal-anchor seed transported to point support is:

```math
\rho_{j,i} =
\frac{\sum_p A_{j,p} m_{p,i}}
     {\sum_{i'} \sum_p A_{j,p}m_{p,i'} + \epsilon}
```

The new pre-reader conditioning is:

```math
b^{point}_{j,i}
\leftarrow
b^{point}_{j,i}

\lambda_{seed} r_j
\left(
\log(\rho_{j,i}+\epsilon)
-
\frac{1}{N}\sum_{i'}\log(\rho_{j,i'}+\epsilon)
\right)
```

and query initialization is:

```math
q_j
\leftarrow
q_j + \lambda_q r_j(\bar z_j^{proposal}-q_j)
```

where `r_j` is the bounded seed reliability.  This preserves the dense memory
reader:

```math
\operatorname{Attn}(q_j, K^{point}, V^{point}; b^{point}_{j,*})
```

The seed is not a hard label:

- no point token is removed;
- no posterior file is overwritten directly;
- the background residual still absorbs untrusted candidates;
- sidecar-free datasets fall back to the normal dense readers;
- tactile remains object-owner evidence when contact exists.

## Why This Fixes The Observed Failure

The failing overlays showed that role-1/orange rows often stayed away from the
green sidecar/mask region, while blue/effector or reserve rows could occupy the
changed object.  With only late pull:

```text
wrong query -> wrong point support -> weak object pull -> wrong posterior
```

the gradient has to change the query after the reader has already built a
wrong support distribution.  The conditional initializer changes the causal
order:

```text
task/contact candidate -> conditioned query/bias -> point support on object
```

This is a structural binding change, not a scalar loss weight patch.

## Implementation Follow-Through

Files:

- `src/openpi/picf/core/config.py`
- `src/openpi/picf/core/pipeline.py`
- `scripts/picf_core_train.py`
- `scripts/picf_object_candidate_slot_binding_audit.py`
- `run_a7_conditional_slot_initializer_anchor1000_20260519.sh`

New/updated runtime contracts:

```text
proposal_anchor_seed_enabled:
  Explicitly enable inspected proposal/mask candidate seeding.

proposal_anchor_seed_pre_reader_enabled:
  Apply query interpolation and point-reader logit bias before point attention.

proposal_anchor_seed_weight:
  Bounded point logit-bias strength. Default 0.85 when seed path is enabled.

proposal_anchor_seed_token_weight:
  Bounded proposal-token query interpolation. Default 0.35 when seed path is enabled.

object_candidate_point_mix:
  Increased to 0.80 so accepted candidate point support can actually dominate
  the object row in the anchor-only probe.

object_candidate_owner_point_mix:
  Kept at 1.0 for accepted owner geometry. This is still measurement fusion,
  not posterior truth overwrite.
```

Acceptance checks:

```text
py_compile:
  config.py, pipeline.py, training.py, picf_core_train.py, audit scripts

object-candidate audit:
  proposal_anchor_seed_conditions_point_reader_before_attention must PASS

verify_picf_owm_contract:
  must pass before remote launch

remote anchor-only probe:
  step 100 should show role-1 active anchor near sidecar/mask support
  loss_anchor_object_pull should decrease
  aqr_proposal_anchor_seed_row_count should be nonzero on sidecar frames
  blue/effector role must be absent in object_only probe
```

## Rejected Alternatives

Blind SAM is not revived.  Its proposals were empirically too noisy in CALVIN
for task-object binding.  The generic `proposal_*` schema remains valid only
for inspected contact/task/tracklet-aware sidecars.

Removing dense V-JEPA/point tokens is rejected.  It would destroy the typed
memory contract and reduce PICF to a proposal detector.  The seed only biases
attention; dense memory remains available.

Adding another late object-pull loss alone is rejected.  The current failure is
causal ordering: the slot has already read the wrong region.

Wholesale image-OCL decoder transplantation is rejected for production PICF.
The correct transplant is the object-centric initializer and competition
principle, not an incompatible reconstruction decoder.


# PICF-AQR-OWM Tri-State Context Routing Follow-Through

Status: deployed as code-level diagnostic repair on 2026-05-16. Behavior
acceptance still requires the restarted A7 diagnostic run.

## 1. Why The Binary Gate Was Rejected

The previous action-visible repair used:

```math
g_i \in \{0,1\},\qquad y_i = g_i\,\phi(a_i)
```

where `a_i` is an AQR graph anchor and `y_i` is the graph-prefix token sent to
the PI0.5 control prefix. This correctly prevented duplicate inactive files
from masquerading as object evidence. It was still too coarse:

```text
inactive real context object == duplicate/no-object reserve
```

That equality is mathematically wrong. CALVIN scenes can contain real non-target
objects such as buttons, lamp, drawer handles, table edges, or distractor blocks.
They should not compete as the task-owner posterior file, but they also should
not be made indistinguishable from empty reserve capacity.

Slot Attention / DIAS-style systems avoid this by allowing both foreground slots
and background/no-object capacity. The transferable principle is not a pasted
reconstruction loss; it is a capacity semantics:

```text
object slots:
  explain current object evidence.

context/background slots:
  preserve scene evidence without becoming the current target object.

empty/reserve slots:
  absorb excess fixed-slot capacity and should not drive actions.
```

## 2. Correct Routing Semantics

PICF now uses a tri-state downstream weight:

```math
w_i =
\begin{cases}
1, & i\in A \\
\lambda_c, & i\in C \\
0, & i\in R
\end{cases}
```

where:

```text
A:
  active target/contact object anchors selected by role-local evidence,
  support diversity, and geometry duplicate checks.

C:
  inactive but real context anchors. They have enough evidence score and
  confidence, and are not duplicates of active anchors.

R:
  duplicate/no-object reserve anchors. They remain lifecycle/diagnostic
  capacity and are not action-prefix object evidence.
```

The action graph prefix is:

```math
y_i = w_i\,(\phi(a_i)+e_{graph})
```

with default:

```text
lambda_c = aqr_context_slot_weight = 0.15
```

This gate is applied only to graph-prefix object evidence. It does not delete
or consume typed memories:

```text
V-JEPA tokens, PG tokens, point tokens, tactile tokens, task/global context,
and semantic PI0.5 tokens remain readable upstream.
```

Therefore background information is not lost. The repair only prevents reserve
files from becoming full object-prefix rows.

## 3. Dataflow

```text
typed memory tokens
  -> AQR readers
  -> visual/temporal/PG/point/tactile/posterior/cache priors
  -> _aqr_active_slot_mask
       chooses distinct active target/contact object anchors
  -> _aqr_downstream_slot_weights
       active: 1.0
       context: aqr_context_slot_weight
       reserve: 0.0
  -> PicfAnchorPriorGraphState.anchor_downstream_weight
  -> _build_conditioned_control_state
       graph_tokens = mapg_to_control_proj(anchor_tokens)
       graph_tokens = graph_tokens + graph_role_embedding
       graph_tokens = graph_tokens * anchor_downstream_weight
  -> PI0.5 control prefix
```

Posterior file competition remains separate:

```text
posterior.file_competition_active
  -> _posterior_file_active_gate
  -> inactive persistent files excluded from posterior prefix/cache write
```

The graph gate decides how AQR observation anchors enter action context. The
posterior gate decides whether persistent files update and persist as current
objects.

## 4. Context Candidate Definition

For non-active anchor `i`, context eligibility is:

```math
conf_i \ge \tau_{conf}
```

```math
score_i = peak(s^v_i) + conf_i\,rawscore_i \ge \tau_{score}
```

and:

```math
\max_{j\in A} overlap(i,j) < \tau_{dup}
```

The overlap is object-core overlap over visual support, temporal support, PG
support, point support, and geometry duplicate proximity. This means context
anchors must be real enough to carry evidence and different enough not to be an
active duplicate.

## 5. Why This Is Not A New Loss

No new supervision signal is introduced. The repair changes routing semantics:

```text
capacity semantics:
  active / context / reserve

not:
  another auxiliary objective
```

It is therefore aligned with the current POMDP belief-filter contract:

```math
b_t = Correction(Prediction(b_{t-1}, a_{t-1}), Evidence(o_t))
```

Context anchors provide weak scene evidence to the action prefix. They do not
overwrite posterior truth and do not create pseudo-label pressure.

## 6. Verification

Static/math audit:

```bash
python scripts/picf_action_visible_reserve_gate_audit.py --fail-on-fail
```

Contract verifier:

```bash
python scripts/verify_picf_owm_contract.py
```

Targeted runtime tests:

```bash
uv run pytest -q src/openpi/picf/core/pipeline_test.py \
  -k "posterior_inactive_files or posterior_file_competition or posterior_lifecycle or binding_signature"
```

Runtime acceptance metrics for the next A7 diagnostic:

```text
aqr_active_same_role_support_overlap_max
aqr_context_anchor_count
aqr_context_downstream_weight_mean
aqr_reserve_anchor_fraction
posterior_identity_switch_rate
posterior_active_file_potential_swap_rate
loss_action_default_equiv
anchor overlay active_only / with_gray
```

Expected healthy behavior:

```text
active overlap:
  remains low.

context count:
  nonzero when scenes have real secondary objects.

reserve fraction:
  nonzero because fixed slot capacity exceeds current object count.

action loss:
  should not regress relative to the previous downstream-gated run.
```

Failure modes:

```text
context count = 0 always:
  gate is too strict; real scene objects are still treated as reserve.

reserve fraction = 0 always:
  every file is action-visible again; duplicate capacity has leaked back.

active overlap rises:
  the repair did not preserve object-owner separation.

identity switch remains high:
  target/contact-owner continuity remains the next root issue.
```

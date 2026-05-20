# PICF-AQR-OWM Context Attention-Bias Follow-Through

Date: 2026-05-20

Status: implemented locally; remote A7 validation in progress.

## Problem

The 2026-05-19 comprehensive frozen-policy run was not an anchor-only probe: it
kept the PICF slot/router/posterior/OEML stack trainable while freezing large
pretrained modules, PaliGemma, and the action head.  Its key failure mode was
not dense-token loss.  The immediate structural issue was that the run disabled
context slots:

```text
--no-aqr-context-slot-enabled
```

With `aqr_active_anchor_count ~= 1.3`, only one or two active object slots could
enter the control graph.  Real but non-target scene objects had no low-priority
path into the PI0.5 control prefix.  The old implementation also multiplied
graph token embeddings by downstream weight:

```math
g_i' = w_i g_i
```

This is mathematically too destructive for context/reserve states.  It changes
the token content norm instead of expressing reliability as a routing prior.

The first A7 attention-bias validation exposed an additional follow-through
issue at step 50: context rows were selected, but their downstream weight was
almost zero:

```text
aqr_context_anchor_count ~= 15.9
aqr_context_downstream_weight_mean ~= 7e-5
```

The cause was a semantic mismatch.  `_aqr_downstream_slot_weights()` selected
context rows using deterministic confidence/support/duplicate filters, then
multiplied those rows by `slot_quality.context_weight`.  The learned
slot-quality head is intentionally conservative early in training, so it
suppressed the whole context path.  That is not the intended math: slot-quality
is an object-truth gate for active posterior owner rows; context is weak scene
evidence that should remain visible after deterministic filtering.

## Paper-Code Follow-Through

The local paper-code references checked for this repair are:

```text
temp/external_repos/MetaSlot/object_centric_bench/model/metaslot.py
temp/paper_code_20260518/slotcontrast/slotcontrast/losses.py
temp/paper_code_20260518/slot-attention-video/savi/modules/attention.py
temp/paper_code_20260518/slot-attention-video/savi/modules/decoders.py
```

Primary papers used for the design boundary:

```text
MetaSlot, 2025:
  https://arxiv.org/abs/2505.20772
  Adaptive slot count and aggregate/deduplicate refinement.

QASA, 2026:
  https://arxiv.org/abs/2601.12936
  Quality-guided adaptive slot selection.

STORM, 2026:
  https://arxiv.org/abs/2601.20381
  Task-aware object-centric slots for manipulation on frozen visual foundation
  features.

Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?,
NeurIPS 2025:
  https://arxiv.org/abs/2510.24709
  IsSameObject-style pairwise binding exists as a latent subspace/probe, so
  binding should be encouraged in support/attention space rather than imposed as
  a brittle hard ID.
```

The common mechanism across the relevant slot literature is not "zero out weak
object tokens."  The recurring principles are:

1. Fixed capacity is allowed, but duplicate/no-object slots must have a
   reliability/no-object mechanism.
2. Context/background is not the same state as duplicate/no-object capacity.
3. Competition should happen in assignment/attention space, not by destroying
   slot embeddings.
4. Temporal/object consistency losses are useful only after the assignment
   channel is stable.

MetaSlot motivates adaptive active capacity and deduplication.  SAVi/Slot
Attention motivate slot-to-input attention rather than hard token deletion.
SlotContrast motivates consistency over object slots, not dense feature
destruction.  STORM-style slot-centric control motivates a separate object-slot
interface before the action model.

## Adopted Math

PICF keeps the fixed AQR query bank:

```math
Q = \{q_i\}_{i=1}^{K}
```

The selector assigns each graph row to one of three downstream states:

```math
s_i \in \{\text{reserve},\text{context},\text{active}\}
```

with reliability

```math
w_i =
\begin{cases}
1, & s_i=\text{active} \\
\lambda_c, & s_i=\text{context} \\
0, & s_i=\text{reserve}
\end{cases}
```

Active rows can still use learned slot-quality:

```math
w_i^{active}=q_i^{active}.
```

Context rows use deterministic evidence filtering by default:

```math
w_i^{context}
=
\lambda_c \mathbf{1}
\left[
conf_i \ge c_{min},
score_i \ge s_{min},
overlap(i,\text{active}) < \tau
\right].
```

The learned context-quality multiplier is an opt-in ablation only.  This avoids
using an early, noisy no-object head to hide dense context from the controller.

The maintained control graph now keeps the graph embedding intact:

```math
g_i = W_g h_i + e_{\text{graph}} + e_{s_i}
```

and expresses reliability as attention bias:

```math
\operatorname{Attn}(x, G)
=
\operatorname{softmax}
\left(
\frac{xK_G^\top}{\sqrt d}
+ \log(\max(w_i,\epsilon))
\right)V_G.
```

This is applied in three places:

```text
1. control_world self-attention columns for graph tokens.
2. pi_prefix_reader source columns for graph tokens.
3. future_condition_reader source columns for graph tokens.
```

This differs from the legacy scaling:

```math
g_i' = w_i g_i
```

The legacy version makes context embeddings low-norm and reserve embeddings
zero-norm before LayerNorm/attention.  It is harder to interpret and can erase
information that should remain available as weak context.  The new version keeps
content available but routes it with explicit reliability.

## Dataflow

```text
AQR typed evidence
  -> anchor priors / support signatures / object candidates
  -> active/context/reserve downstream weights
  -> mapg_to_control_proj(anchor_tokens)
  -> graph role embedding + graph state embedding
  -> control_world(attn_bias=log downstream weight)
  -> pi_prefix_reader(attn_bias=log downstream weight)
  -> future_condition_reader(attn_bias=log downstream weight)
```

Dense memories are not pruned:

```text
M_vjepa, M_point, M_tactile, M_tracklet, M_proposal
```

remain readable upstream.  This repair only controls how already-routed graph
rows enter the action-control interface.

## Code Changes

Config:

```text
aqr_control_graph_attention_bias_enabled = True
aqr_control_graph_token_scaling_enabled = False
aqr_control_graph_state_embedding_enabled = True
aqr_control_graph_bias_min = 1e-4
aqr_context_slot_quality_gate_enabled = False
```

Pipeline:

```text
mapg_control_state_embedding: nn.Embedding(3, semantic_dim)
_build_conditioned_control_state(...):
  state id 0 reserve, 1 context, 2 active
  graph token scaling is legacy opt-in only
  graph-column attention bias is passed to control_world
  same bias is passed to PI prefix and future readers
```

Training script:

```text
--aqr-context-slot-enabled
--aqr-context-slot-weight 0.12
--aqr-context-slot-min-confidence 0.03
--aqr-context-slot-min-score 0.005
--no-aqr-context-slot-quality-gate-enabled
--aqr-control-graph-attention-bias-enabled
--no-aqr-control-graph-token-scaling-enabled
--aqr-control-graph-state-embedding-enabled
```

## What This Does Not Claim

This repair does not claim that:

```text
1. object masks are perfect;
2. SAM should be re-enabled;
3. slot-JEPA/support-pred can be enabled;
4. full reconstruction decoding is needed now;
5. VQ prototype truth should replace PICF posterior identity.
```

Those are separate choices.  Under noisy sidecar masks and missing-modality
scaling, full prototype truth and dense reconstruction objectives would likely
over-constrain the belief router.  PICF keeps them as future guarded aux
directions, not maintained defaults.

## Acceptance

Short validation should compare against the 2026-05-19 comprehensive
frozen-policy run:

```text
aqr_context_anchor_count should be nonzero.
aqr_context_downstream_weight_mean should be near 0.12.
aqr_active_same_role_support_overlap_max should remain low.
loss_mapg_support_diversity should not regress.
loss_anchor_object_pull and loss_object_explanation_point should improve or at
least not worsen.
loss_action_default_equiv remains diagnostic only when action weights are zero.
```

The critical negative check:

```text
graph token norms must not be zeroed by downstream weights.
context downstream weights must not be zeroed by slot-quality when deterministic
context filtering accepts the row.
```

The local unit test
`test_control_graph_downstream_weights_are_attention_bias_not_token_scaling`
checks that the control world and downstream readers receive attention bias and
that reserve graph tokens remain nonzero when token scaling is disabled.

The local unit test
`test_context_slot_downstream_weight_is_not_zeroed_by_slot_quality` checks the
step-50 failure directly: accepted context rows keep the configured context
weight even when `slot_quality.context_weight=0`, unless learned context gating
is explicitly enabled.

# PICF-AQR-OWM G18 All VLA Methods Follow-Through

Date: 2026-06-04

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Status:

```text
strict follow-through complete for the requested VLA method list;
new GPU experiment launched only if the method is non-duplicate and can change
the deployment decision.
```

This document supersedes ad-hoc "try another optimizer / try another sampler"
requests for the current CALVIN action-plateau problem.  It audits every
requested method against:

```text
1. theory follow-through: what objective or gradient estimator it changes;
2. code follow-through: exact current file/function/flag path;
3. experiment follow-through: prior run, active run, or reason not to rerun;
4. deployment decision: keep default, keep diagnostic, reject, or future branch.
```

Related documents:

```text
docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
docs/PICF_AQR_OWM_G17_COMPLETE_VLA_METHOD_GATE_20260604.md
docs/PICF_AQR_OWM_ONE_HOUR_E21_DIAGNOSTIC_20260604.md
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
docs/PICF_AQR_OWM_5_6_CARD_FSDP_E21_DECISION_20260603.md
```

## External Literature Contract

The requested methods are consistent with recent VLA/multi-task training
practice, but the literature does not imply that every module should be turned
on by default.

```text
VLA Foundry / ABot-M0:
  use controlled data/task mixing, gradient accumulation, and normalization for
  heterogeneous VLA training.

PiKE:
  dynamic data mixing is useful when measured task progress justifies changing
  q_b(t); it is not a free replacement for a broken action readout.

pi0.5 / Knowledge Insulation:
  separate semantic VLM knowledge from noisy continuous action gradients; use a
  controlled motor-representation objective or bridge.

OpenVLA-OFT:
  continuous action chunks and simple stable action objectives are a strong
  baseline for action fine-tuning.

GR00T N1:
  dual-system VLM reasoning plus action module is a valid large VLA pattern,
  but it does not mandate MoE or full architecture replacement for this CALVIN
  gate.

AdaMoE/FedVLA:
  action-specialized experts are justified only after task-conflict evidence;
  do not MoE the whole backbone as a first-line patch.
```

References recorded for traceability:

```text
VLA Foundry: https://arxiv.org/abs/2604.19728
ABot-M0: https://arxiv.org/abs/2602.11236
PiKE: https://arxiv.org/abs/2502.06244
OpenVLA-OFT: https://arxiv.org/abs/2502.19645
Knowledge Insulation: https://arxiv.org/abs/2505.23705
pi0.5: https://arxiv.org/abs/2504.16054
GR00T N1: https://arxiv.org/abs/2503.14734
FedVLA: https://arxiv.org/abs/2508.02190
AdaMoE-VLA: https://charleshen1412.github.io/AdaMoE-VLA/
```

## Core Mathematical Diagnosis

The desired multi-task objective is:

```text
L = sum_b q_b L_b
G = sum_b q_b grad L_b
```

where `b` is a task/data/modality bucket.  A small physical batch is valid only
if an optimizer step estimates `G`, not a single random task gradient.

The maintained estimator is:

```text
g_hat = sum_{b in B_step} (q_b / n_b) grad L_b
```

where `n_b` is the number of selected micro-windows from bucket `b` in the
current logical optimizer update.  This estimator is what VLA Foundry-style
batch balancing and ABot-M0-style task balancing require at code level.

Important implication:

```text
If g_hat is already correct and action still plateaus, repeating sampler-only
variants cannot be the final fix.  The next non-duplicate target is the action
representation/readout boundary.
```

## Requested Method Matrix

| Method | Mathematical target | Code follow-through | Experiment follow-through | Decision |
| --- | --- | --- | --- | --- |
| Task-uniform logical batch | approximate `sum_b q_b grad L_b` with equal task-family mass | `scripts/picf_core_train.py:_compute_bucket_sampling_weights`, `_bucket_sequence_for_logical_step`, `_logical_batch_loss_scales`, `_CalvinTransitionSource.balanced_bucket_slot_index` | G12 local tests and live K4/K8 gates proved distinct bucket coverage and `q_b/n_b` scaling | Keep mandatory; not a missing fix |
| Gradient accumulation as logical batch | increase logical task coverage without requiring physical huge batch | `ACCUM_STEPS`, `logical_batch_task_count`, optimizer step after logical group | 2x/5x/6x K4/K8/K12 attempts; E21 diagnostic compared hardware/window hypotheses | Keep, but large logical batch alone did not reproduce stable 0.02 action |
| Explicit VLA-Foundry-style ratios | fixed target mixture `q_b` | `--calvin-bucket-weight-spec`, `_parse_bucket_weight_spec` | G12-RATIO improved some structure terms but did not fix action descent | Keep knob, not default |
| Temperature sampling | `q_b proportional N_b^alpha` | `--calvin-bucket-sampling-mode temperature`, `--calvin-bucket-temperature-alpha` | G12-TEMP alpha=0.5 failed fast action-slope gate and produced clipped-gradient warning | Keep knob for new datasets, not current fix |
| Trajectory-proportional sampling | empirical trajectory-frequency mixture | `--calvin-bucket-sampling-mode trajectory` | Local smoke/code path valid; rejected as default because long-tail coverage weakens | Diagnostic only |
| Per-bucket logical loss normalization | each bucket contributes target mass independent of duplicates | `_logical_batch_loss_scales` | Local smoke verified duplicate bucket example `[1/6, 1/3, 1/6, 1/3]`; tests cover DDP compensation | Keep mandatory |
| Per-modality/action normalization | avoid hidden scale dominance | action/PICF/sidecar/tactile/visual losses are separately weighted/logged; action default-equivalent metric | A3/G12 action target-scale audits; current CALVIN is single embodiment | Keep existing; generic multi-dataset normalizer is future infra |
| Per-bucket action EMA scale | bounded action backward equalization across buckets | `_logical_action_bucket_scale`, `_update_logical_action_bucket_ema` | Implemented; used as controlled normalization, not a magic fix | Keep off/on by recipe, log always when enabled |
| Dynamic PiKE-style mixing | bounded adaptive `q_b(t)` based on lag/progress | `_dynamic_bucket_sampling_weights` | G12-DYN dataflow real; action not fixed | Keep infrastructure, off by default |
| Gradient cosine diagnosis | measure inter-task conflict before surgery | `scripts/picf_bucket_gradient_cosine_probe.py`, G12 probes | Used to justify scoped PCGrad/CAGrad trials | Keep diagnostic |
| Scoped PCGrad | remove negative task-gradient components in small trainable groups | `_logical_batch_autograd_grads`, `_pcgrad_project_and_sum`, `--logical-batch-gradient-surgery pcgrad` | G12-PCG real dataflow; structure OK, action failed | Keep diagnostic/off |
| Scoped CAGrad | conflict-averse simplex update in small trainable groups | `_cagrad_project_and_sum` | G12-CAGrad real dataflow; not action fix | Keep diagnostic/off |
| Whole-model PCGrad/CAGrad | full-model multi-task gradient surgery | intentionally not deployed | Rejected by cost and current stop-gradient boundary; would require full per-task full-model grads | Do not run unless architecture changes |
| VLM/action boundary | prevent noisy continuous action gradients from rewriting semantic trunk/PICF core directly | action context stopgrad, semantic trainable scopes, optimizer groups | G11-G17 action bridge/boundary matrix | Keep; current bridge not proven helpful |
| Suffix gated cross-attention action context | let action suffix read bounded PICF context | `src/openpi/picf/paligemma/wrapper.py:_apply_action_context_adapter` | G17 fixed-window normal-vs-noPICF: no measurable positive action-loss benefit | Keep as optional, not convergence cure |
| Action expert router/residual | action-only expert residual, not whole VLM MoE | `_apply_action_expert_router`, router flags, compatibility loader | Implemented; older ckpts may lack params; not justified as default by G17 | Evidence-triggered only |
| Continuous action chunks | stable chunk-level continuous control objective | `compute_action_flow_loss`, `loss_action_default_equiv` | Current PICF action path | Keep mandatory |
| L1/Huber/SmoothL1 variants | robust scalar action objective | `--semantic-action-flow-loss` | Prior gates showed robust variants did not resolve plateau alone | Keep diagnostic |
| FAST/action-token auxiliary | motor representation CE objective insulated from continuous action expert | OpenPI native `TokenizeFASTInputs`, `ExtractFASTActions`, `Pi0FAST`; PICF path lacks this auxiliary | G17 audit: missing from PICF core training | True future implementation branch |
| Modality-specific projectors/adapters | align modalities before shared/action layers | current PICF typed visual/point/tactile/proprio/support projections and PaliGemma/action adapters | Implemented enough for CALVIN; future multi-dataset needs generic registry | Keep |
| Embodiment-specific adapters/heads | avoid robot/action-space conflict | not required in CALVIN single embodiment | No valid CALVIN gate | Future large-data branch |
| System2/System1 split | high-level plan/subtask separate from low-level action | no reliable CALVIN subtask labels in this gate | Not a fast action-loss repair | Future long-horizon branch |
| Action expert MoE | task-aware experts for persistent conflict | router exists, full MoE not default | Literature supports after conflict evidence; current G12/G17 evidence does not justify as first next step | Future evidence-triggered branch |

## Experiment Part

### E18-0: Static and unit-level contract gate

Checklist:

```text
[x] run py_compile on touched action/probe/train files
[x] confirm bucket sampler unit tests exist
[x] confirm action context adapter/router tests exist
[x] confirm native FAST tests exist in OpenPI path
[x] identify PICF FAST auxiliary gap
```

Acceptance:

```text
No training launch is allowed if a requested method lacks a code path and would
therefore be a fake experiment.
```

### E18-1: Historical experiment de-duplication gate

Checklist:

```text
[x] task_uniform: covered by G12
[x] temperature: covered by G12-TEMP
[x] explicit ratio: covered by G12-RATIO
[x] dynamic PiKE: covered by G12-DYN
[x] scoped PCGrad: covered by G12-PCG
[x] scoped CAGrad: covered by G12-CAGrad
[x] action bridge: covered by G17-B fixed-window causal probe
[x] E21 hardware/window hypothesis: covered by one-hour E21 diagnostic
```

Decision:

```text
Do not rerun these as isolated fixes.  They have code and behavior evidence.
```

### E18-2: Remote 2xA100 status gate

Checklist:

```text
[x] verify remote machine is reachable
[x] verify GPUs are idle or identify active jobs
[x] verify G17 logs and summaries exist
[x] do not launch duplicate sampler/bridge experiment
```

Acceptance:

```text
Remote result must only be used to confirm completed G17 artifacts or to run a
new non-duplicate branch.
```

### E18-3: Next non-duplicate branch

The only current branch that is both literature-supported and not already
tested in PICF is:

```text
PICF-side FAST/action-token representation auxiliary.
```

Required implementation, if selected:

```text
1. generate FAST action tokens y = FASTTokenizer(a_{1:H});
2. add a PICF/PaliGemma action-token prediction head or compatible adapter;
3. train CE only as a representation auxiliary:
      L_fast = CE(p_theta(y_i | y_<i, semantic/PICF/state condition), mask_i)
4. keep continuous flow action loss and default-equivalent metric unchanged;
5. log:
      loss_action_fast_ce
      loss_action_flow
      loss_action_default_equiv
      lambda_action_fast_ce
6. preserve Knowledge-Insulation boundary:
      continuous action gradients do not freely update PICF core/VLM trunk.
```

Do not launch a fake FAST experiment until these code paths exist.

## Deployment Part

Current maintained deployment recipe:

```text
calvin_balanced_bucket_sampler = true
calvin_bucket_sampling_mode = task_uniform
calvin_bucket_sample_without_replacement = true
logical_batch_bucket_normalization = true
logical_batch_log_bucket_metrics = true
continuous action chunk flow = on
PICF/action stop-gradient boundary = on
dynamic mixing = off by default
PCGrad/CAGrad = off by default
action expert router = off unless capacity branch is selected
SAM = archived/off
sidecar/object-mask = weak auxiliary only, never hard label
```

Do not claim:

```text
1. "all VLA paper methods are experimentally successful";
2. "large batch alone fixes the action plateau";
3. "current PICF action bridge is proven beneficial";
4. "FAST/action-token Knowledge-Insulation auxiliary is already tested in PICF".
```

Allowed claim:

```text
The sampler/mixing/normalization/gradient-surgery family is implemented,
tested, and insufficient as the sole action-convergence fix.  The next
non-duplicate implementation branch is PICF-side action-token representation
supervision or a stricter action-expert capacity/boundary redesign.
```

## G18 Outcome Log

```text
2026-06-04:
  G17 exact-window bridge probe already completed:
    normal_suffix:
      first10_action_default_equiv = 0.05230241254903376
      last10_action_default_equiv  = 0.041024536034092306
    no_picf_action:
      first10_action_default_equiv = 0.05133453262969852
      last10_action_default_equiv  = 0.04082355755381286
  Result:
    bounded suffix PICF action context did not beat no-PICF action conditioning
    on identical windows.

  G17 FAST audit already completed:
    OpenPI native FAST exists.
    PICF core training lacks a deployed FAST/action-token auxiliary.

  G18 decision:
    No duplicate sampler/PCGrad/action-bridge GPU test is justified.
    New work should implement the missing FAST/action-token auxiliary or a
    deliberate action-expert capacity redesign, then run a fixed-window gate.

  Local script/unit verification:
    python3 -m py_compile passed for:
      scripts/picf_core_train.py
      scripts/picf_action_bridge_capacity_probe.py
      src/openpi/picf/paligemma/wrapper.py
      src/openpi/picf/core/training.py
      src/openpi/transforms/_base.py
      src/openpi/models/pi0_fast.py

    pytest passed:
      scripts/picf_core_train_test.py -k "bucket or pcgrad or logical_batch or gradient_surgery"
        10 passed
      src/openpi/picf/paligemma/wrapper_test.py -k "action_context_adapter or action_expert_router"
        5 passed
      src/openpi/models/tokenizer_test.py -k "FAST or fast"
        1 passed

    FAST model-level note:
      src/openpi/models/model_test.py -k "FAST or fast" initialized native
      Pi0FAST and hit local JAX GPU OOM while allocating ~4.5 GiB during model
      construction.  This is not counted as a PICF failure and not counted as
      a PICF FAST auxiliary pass.  It only confirms that native Pi0FAST is a
      heavyweight separate model path; PICF still needs its own lightweight
      FAST/action-token auxiliary before behavior testing is valid.

  Remote 2xA100 status:
    host = qe93X5
    gpu0 = A100-PCIE-40GB, 0 MiB used
    gpu1 = A100-PCIE-40GB, 0 MiB used
    tmux = no active sessions
    G17 logs and summaries exist under:
      /mnt/picf_fixed_window_probes/g17_bridge_20260604
```

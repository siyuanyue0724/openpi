# PICF-AQR-OWM G26-B Checkpoint Cleanup and Archive

Date: 2026-06-04

Entry point:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_G26_FULL_VLA_METHOD_AUDIT_AND_NEXT_GATE_20260604.md
```

## 1. Scope

This archive records the June-04 checkpoint cleanup and the current G26-B
long-run status.  It is a storage and experiment-ledger operation, not a model
change.

Remote:

```text
host: px-cloud1.matpool.com:26120
repo: /root/openpi_g25_20260604
active tmux: picf_g26b_tokenaux_long30k_k4_fresh_20260604
active log: /mnt/picf_run_logs/picf_g26b_tokenaux_long30k_k4_fresh_20260604.log
cleanup manifest: /mnt/picf_run_logs/ckpt_cleanup_20260604_manifest.txt
```

## 2. Cleanup Policy

The cleanup deletes checkpoint directories only.  It keeps logs and local docs
because experimental conclusions should remain reproducible from records even
when low-value weights are removed.

Preserve:

```text
April baselines:
  picf_v21_*
  picf_v22_*

legacy zero/semantic prefix anchors:
  picf_zero1_*
  picf_semantic_prefix_*

known useful May/June anchors:
  picf_a7_actionprefix_ema_from6800_action2_30k_20260527
  picf_a7_e14g_fullopt_from9600_30k_20260531
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601
  picf_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601

G22/G24/G25/G26 key gates:
  picf_g22_readout_aux_k4_from11000_to11300_20260604
  picf_g24_allmethod_k4_readout_from11000_to11300_20260604
  picf_g25_flowresidual_c2_from11000_to11300_20260604
  picf_g26b_tokenaux_fresh_k4_300_20260604
  picf_g26b_tokenaux_imbalance_k4_200_20260604
  picf_g26b_tokenaux_long30k_k4_fresh_20260604
```

Delete:

```text
old A5/A7 short diagnostics
SAM/sidecar/anchor-only smoke ckpts that are already documented
sanity/2-step/model-only residues
old F/ZKY/G10-G16 method-diagnostic ckpt directories whose logs are retained
old 5x/6x repro ckpt directories whose logs are retained
```

Remote result:

```text
deleted checkpoint directories: 97
kept checkpoint directories:    18
```

The exact retained/deleted list is in:

```text
/mnt/picf_run_logs/ckpt_cleanup_20260604_manifest.txt
```

## 3. Current G26-B Long Run

Launch contract:

```text
session: picf_g26b_tokenaux_long30k_k4_fresh_20260604
steps: 30000
save interval: 500
max kept checkpoints: 3
world size: 2
accum steps: 2
effective global batch: 4
unroll steps: 2
burnin steps: 1
bucket sampling: task_uniform
logical batch task count: 4
```

Trainable/frozen split:

```text
frozen pretrains:
  Sonata
  V-JEPA
  AnyTouch

trainable:
  PaliGemma action head / adapter scope
  PICF connector/action-context modules required by G26-B
```

Runtime after cleanup:

```text
tmux alive: yes
latest checked step: 330
loss_action_default_equiv: 0.07850
loss_total: 0.30117
loss_total_minus_action: 0.07119
pi_context_token_aux_loss: 0.72975
pi_context_flow_gain_mse_delta: 0.01186
active same-role support overlap max: 0.01418
context same-role support overlap max: 0.02557
raw same-role support overlap max: 0.46802
steps/sec: 0.04208
```

## 4. Action Trend Comparison Rule

Use `loss_action_default_equiv` for PI0.5/4-22 comparable action MSE.  Do not
compare:

```text
raw loss_action
resume-only train-stream local lows
fixed-window probes
fresh production-style runs
```

as if they were identical distributions.

Current G26-B fresh early trend:

```text
step100: 0.1165
step200: 0.0766
step300: 0.0654
step330: 0.0785
```

This is slower than the best older short-window/resume branches that reached
the `0.03-0.05` band around 500-1000 steps, but those branches were not always
fresh production-style runs.  The active acceptance line is:

```text
step500 <= 0.060:
  continue as healthy.

step500 in 0.060..0.070:
  continue to step700, then decide.

step500 > 0.070:
  stop and treat G26-B fresh as not reproducing the old PICF action-compression
  speed.
```

## 5. Current Interpretation

The cleanup does not change the current scientific conclusion:

```text
G26-B has fixed the earlier direct action-readout causality gap better than
G22/G25, because token auxiliary and flow residual metrics are live and
positive.

However, early fresh-run action compression is slower than the older best
short-window/resume traces.  This must be judged at step500/700/1000 rather
than accepted from step300 alone.
```

No new branch should be launched just because raw same-role overlap is around
`0.45`; active/context overlap is the downstream-relevant metric in the current
contract and remains low so far.


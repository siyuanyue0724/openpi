# PICF-AQR-OWM One-Hour E21 Diagnostic

Date: 2026-06-04

Status: complete.

## Purpose

This diagnostic answers one narrow question:

```text
Why did historical E21-style runs show fast action descent, while the current
K12 6-card logical-batch run stayed around 0.04-0.05 action loss?
```

Do not use this diagnostic as a new training recipe.  It is a no-update,
fixed-window probe against the same current checkpoint, using three different
window sets.

## Inputs

Remote machine:

```text
ssh -p 26420 root@px-cloud1.matpool.com
repo: /root/openpi_6card_e21_955b3ed
venv: /root/openpi/.venv/bin/python
```

Current checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_6x40g_k12_acc2_from11000_11100_20260603/11100
```

Window sets:

```text
/mnt/picf_onehour_diag_20260604/old_resume_skip2400_first32.jsonl
/mnt/picf_onehour_diag_20260604/old_stratified_4perbucket.jsonl
/mnt/picf_onehour_diag_20260604/current_k12_trace_first32.jsonl
```

Each file contains 32 explicit `segment/start_step` windows.

## Tests

Old exact-window probe:

```text
tmux: picf_diag_old_exact32_20260604
input: old_resume_skip2400_first32.jsonl
output:
  /mnt/picf_onehour_diag_20260604/probes/old_exact32_records.jsonl
  /mnt/picf_onehour_diag_20260604/probes/old_exact32_summary.json
```

Old balanced stratified probe:

```text
tmux: picf_diag_stratified32_20260604
input: old_stratified_4perbucket.jsonl
output:
  /mnt/picf_onehour_diag_20260604/probes/old_stratified_4perbucket_records.jsonl
  /mnt/picf_onehour_diag_20260604/probes/old_stratified_4perbucket_summary.json
```

Current K12-trace probe:

```text
tmux: picf_diag_current32_20260604
input: current_k12_trace_first32.jsonl
output:
  /mnt/picf_onehour_diag_20260604/probes/current_trace32_records.jsonl
  /mnt/picf_onehour_diag_20260604/probes/current_trace32_summary.json
```

## Interpretation Rules

If the current checkpoint is low on `old_exact32` but high on
`old_stratified_4perbucket` and `current_trace32`, then the E21-low action rows
were primarily a window/distribution effect.  The current action path is not
proven broken, but the production trainer needs better task-balanced coverage
or curriculum.

If the current checkpoint is high on `old_exact32` too, then the current
checkpoint/action bridge is weaker than the old E21-compatible path even on
the easier historical windows.  That points to model/training-path mismatch,
not just sampling.

If `old_stratified_4perbucket` is close to historical old-stratified action
loss but `current_trace32` is worse, then current K12 trace composition is the
blocker.  Inspect bucket mix and hard buckets before changing model structure.

If all three probes are high, do not repeat K12 hardware experiments.  The next
step should be action-readout / bridge / loss-scale diagnosis, because simply
covering more windows is not enough.

## Relation To Existing Notes

This diagnostic is compatible with:

```text
docs/PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md
docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
docs/PICF_AQR_OWM_5_6_CARD_FSDP_E21_DECISION_20260603.md
```

Critical historical fact from the May-30 audit:

```text
old train-stream step20000 row: about 0.021
old exact-source replay around step20000: about 0.022 on that exact row
old exact-source replay over 19901..20000: about 0.034
old balanced stratified replay: about 0.040
```

Therefore, do not compare a current balanced/fixed-window probe directly to a
single historical live train-stream row.

## 2026-06-04 Results

All three probes completed on the 6-card machine.  The first attempt at the old
stratified set failed because one historical window was out of the current
data-source valid range:

```text
segment=4736 start_step=683786 valid=[683765,683786)
```

The stratified set was rebuilt by filtering against the current
`_CalvinTransitionSource` boundary contract.  The repaired set keeps 32
windows: 8 buckets times 4 valid windows.

Action loss summary:

```text
current_trace32:
  loss_action_default_equiv mean = 0.0379988893
  min / max = 0.0040301587 / 0.0773780271

old_exact32:
  loss_action_default_equiv mean = 0.0433369576
  min / max = 0.0091737639 / 0.0957021117

old_stratified_valid32:
  loss_action_default_equiv mean = 0.0423149307
  min / max = 0.0030706599 / 0.0848062932
```

Structure summary:

```text
current_trace32:
  active overlap mean = 0.078125
  downstream overlap mean = 0.078125
  raw same-role overlap mean ~= 1.0

old_exact32:
  active overlap mean = 0.359375
  downstream overlap mean = 0.328125
  raw same-role overlap mean = 1.0

old_stratified_valid32:
  active overlap mean = 0.3125
  downstream overlap mean = 0.296875
  raw same-role overlap mean = 1.0
```

Bucket action means:

```text
current_trace32:
  block_lift 0.037015
  block_other 0.036018
  block_push 0.030120
  drawer 0.038658
  other 0.033888
  slider 0.035421
  switch_button_light 0.053964

old_stratified_valid32:
  block+drawer 0.050518
  block+drawer+grasp 0.049775
  block+grasp 0.033276
  block+push 0.042238
  block+slider 0.019263
  drawer 0.039658
  slider 0.051290
  switch+light+turn 0.052502
```

## Conclusion

The E21-fast result is not reproduced by simply matching the K12 window-count
condition on the current production checkpoint.  The current checkpoint is
not low on historical exact windows; it is actually slightly better on the
current K12 trace windows than on the old exact windows.

This rejects the simple hypothesis:

```text
more cards + K12 logical batch alone => E21-style 0.02 action loss
```

It also rejects using the historical single train-stream `0.02` row as a direct
target for current fixed-window probes.  On comparable fixed-window probes,
the current checkpoint sits near the old balanced-stratified regime
(`0.038-0.043`), not near the live-row `0.02`.

The remaining production question is not hardware reproduction.  It is how to
move balanced/fixed-window action quality below the `0.04` region on hard task
buckets without overfitting a narrow online window.  The next tests should
focus on action readout / bridge / loss-scale and bucket difficulty, not
another blind K12/large-card run.

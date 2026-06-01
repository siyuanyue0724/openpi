# PICF 2026-04-22 Ablation Baseline Archive - 2026-05-30

Canonical comparison scalar:

```text
loss_action_default_equiv
```

For the archived 2026-04-22 PI0.5-only ablation, this is equivalent to the
action-flow loss because PICF branches are disabled. For current PICF runs, this
is the only action scalar that should be compared against the old ablation.

## Why This Archive Exists

The old `0.02` action-loss rows and the current `0.04` live train-stream rows
are not automatically comparable. They are often different sampled windows from
different deterministic/replay streams. The fixed-window probes added on
2026-05-29 show that old low train-stream rows are not the same as old
checkpoint quality on identical windows.

Future claims must state which baseline is being used:

```text
train-stream row:
  useful for online training monitoring only;
  can be dominated by sampled-window difficulty.

fixed-window probe:
  apples-to-apples checkpoint comparison;
  required for stop/continue decisions.
```

## Archived 4-22 Train-Stream Levels

These are historical training-log/action-stream values recorded in prior
experiment notes. They are useful for intuition, but they are not fixed-window
validation numbers.

```text
step      archived action level
3000      ~0.0496
5000      ~0.0451
15100     ~0.0311
17500     ~0.0266
19300     ~0.01839
20000     ~0.02137
```

The `19300 -> 20000` change shows why single train-stream points should not be
treated as a monotonic validation curve.

## Direct A7 Log Parse - 2026-05-30 Audit

The visible A7 debug logs for
`picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2` were parsed directly
with the `tqdm` row pattern:

```text
loss=<value>, lr=<value>, step=<step>
```

Parsed coverage:

```text
logs parsed: 9
parsed steps: 12564
visible range: step 1 .. 20155
```

Representative exact single rows:

```text
step      exact train-row loss
100       0.112500
200       0.032300
300       0.110200
500       0.044500
1000      0.044800
2000      0.081900
2500      0.151900
3000      0.047600
5000      0.056200
12501     0.021900
15000     0.121500
15100     0.041400
17500     0.121400
19300     0.014300
19949     0.019300
20155     0.025200
```

Local window means around selected steps are more informative than exact single
rows:

```text
center    rows  mean      min       max
2500      101   0.049567  0.005900  0.211100
3000      101   0.046307  0.003600  0.145500
5000       61   0.039613  0.004800  0.134600
12500      50   0.031816  0.002700  0.103500
15000     101   0.030104  0.002000  0.126300
17500     101   0.023074  0.002000  0.121400
```

This audit reinforces two points:

```text
1. The 4-22 train-stream can hit very low single rows such as 0.014-0.019.
2. The same log also contains high rows at nearby late steps, so a single row is
   not a checkpoint-quality estimate.
```

## Archived 4-22 Fixed-Window Levels

The same accepted 64 CALVIN training windows were evaluated on archived 4-22
checkpoints:

```text
step      fixed-window mean action
7500      0.058717
10000     0.053813
20000     0.051038
```

This is the current canonical apples-to-apples baseline for comparing the
corrected PICF branch.

## 25000-Step Status

The inspected A7 storage contains:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2/
    5000
    7500
    10000
    12500
    15000
    17500
    20000
```

No `25000` checkpoint or fixed-window summary was present during the 2026-05-30
audit, and the directly parsed visible logs only reach step `20155`. Therefore
`4-22 step25000` must not be quoted as a verified baseline unless that
checkpoint/log is recovered and evaluated on the same fixed windows.

## Current Interpretation Rule

For current live PICF runs:

```text
loss_action_default_equiv around 0.04:
  worse than the old train-stream 0.02 low points;
  not enough by itself to prove current checkpoint regression.

fixed-window action >= old step20000 fixed 0.051:
  current action quality has not beaten the archived 4-22 20k checkpoint.

fixed-window action < old step20000 fixed 0.051:
  current checkpoint has crossed the old 20k action-quality gate on the same
  windows, subject to CALVIN/video behavior.
```

The production gate remains:

```text
1. fixed-window action comparison,
2. active/downstream anchor health,
3. CALVIN/video behavior,
4. inference latency.
```

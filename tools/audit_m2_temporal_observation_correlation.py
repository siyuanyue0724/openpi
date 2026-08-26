#!/usr/bin/env python3
"""Audit temporal correlation in frozen M2 geometry observations.

The M2 uncertainty report is marginal: it calibrates one observation at a
time. A recurrent Bayesian filter additionally needs the cross-covariance
between its prior and current observation. This tool reconstructs consecutive
identity-aligned pairs without loading images or a model and reports whether a
zero-transition prior and current neural observation provide independent
errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = "picf.m2-temporal-observation-correlation.v1"
_FLOAT_FIELDS = (
    "predicted_mean_normalized",
    "target_normalized",
    "residual_normalized",
)
_TEXT_FIELDS = ("model_arm", "split", "group_kind", "identity_key", "axis")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_row(value: object, *, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"coordinate row {line_number} must be a JSON object")
    row = dict(value)
    for field in _TEXT_FIELDS:
        item = row.get(field)
        if not isinstance(item, str) or not item:
            raise ValueError(f"coordinate row {line_number} has invalid {field}")
    for field in _FLOAT_FIELDS:
        item = row.get(field)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"coordinate row {line_number} has invalid {field}")
        row[field] = float(item)
    for field in ("global_index", "group_index"):
        item = row.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"coordinate row {line_number} has invalid {field}")
    residual = row["predicted_mean_normalized"] - row["target_normalized"]
    if not math.isclose(
        residual,
        row["residual_normalized"],
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"coordinate row {line_number} has inconsistent residual")
    return row


def load_coordinate_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="ascii") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"coordinate row {line_number} is not valid JSON") from error
                rows.append(_validated_row(value, line_number=line_number))
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read coordinate rows: {source}") from error
    if not rows:
        raise ValueError("temporal correlation audit requires coordinate rows")
    return rows


def consecutive_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_arm: str = "actual",
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return adjacent frames within one split, stream, identity and axis."""

    if not isinstance(model_arm, str) or not model_arm:
        raise ValueError("model_arm must be a nonempty string")
    grouped: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    selected_count = 0
    for row in rows:
        if row["model_arm"] != model_arm:
            continue
        selected_count += 1
        key = (
            row["split"],
            row["group_kind"],
            row["group_index"],
            row["identity_key"],
            row["axis"],
        )
        grouped[key].append(row)
    if selected_count == 0:
        raise ValueError(f"coordinate rows contain no model_arm={model_arm!r}")

    pairs = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["global_index"]))
        indices = [int(row["global_index"]) for row in ordered]
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate identity-axis frame in group {key!r}")
        pairs.extend(
            (left, right)
            for left, right in zip(ordered, ordered[1:], strict=False)
            if int(right["global_index"]) == int(left["global_index"]) + 1
        )
    if not pairs:
        raise ValueError("temporal correlation audit found no consecutive identity pairs")
    return pairs


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.var(left) == 0.0 or np.var(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _axis_summary(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, float | int | None]:
    previous_residual = np.asarray(
        [float(left["residual_normalized"]) for left, _right in pairs],
        dtype=np.float64,
    )
    current_residual = np.asarray(
        [float(right["residual_normalized"]) for _left, right in pairs],
        dtype=np.float64,
    )
    previous_prediction = np.asarray(
        [float(left["predicted_mean_normalized"]) for left, _right in pairs],
        dtype=np.float64,
    )
    current_prediction = np.asarray(
        [float(right["predicted_mean_normalized"]) for _left, right in pairs],
        dtype=np.float64,
    )
    previous_target = np.asarray(
        [float(left["target_normalized"]) for left, _right in pairs],
        dtype=np.float64,
    )
    current_target = np.asarray(
        [float(right["target_normalized"]) for _left, right in pairs],
        dtype=np.float64,
    )

    zero_transition_error = previous_prediction - current_target
    current_innovation = current_prediction - previous_prediction
    denominator = float(np.dot(current_innovation, current_innovation))
    gain = None
    fused_mse = None
    improvement = None
    prior_mse = float(np.mean(np.square(zero_transition_error)))
    if denominator > np.finfo(np.float64).tiny:
        gain = float(-np.dot(zero_transition_error, current_innovation) / denominator)
        fused_error = zero_transition_error + gain * current_innovation
        fused_mse = float(np.mean(np.square(fused_error)))
        if prior_mse > 0.0:
            improvement = float((prior_mse - fused_mse) / prior_mse)

    return {
        "pair_count": len(pairs),
        "residual_lag_pearson": _pearson(previous_residual, current_residual),
        "current_observation_mse": float(np.mean(np.square(current_residual))),
        "zero_transition_prior_mse": prior_mse,
        "target_delta_mse": float(np.mean(np.square(current_target - previous_target))),
        "optimal_current_innovation_gain": gain,
        "optimal_linear_fused_mse": fused_mse,
        "optimal_linear_relative_mse_improvement": improvement,
    }


def summarize_temporal_correlation(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_arm: str = "actual",
) -> dict[str, object]:
    pairs = consecutive_pairs(rows, model_arm=model_arm)
    by_split_axis: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for pair in pairs:
        left, right = pair
        if left["split"] != right["split"] or left["axis"] != right["axis"]:
            raise RuntimeError("consecutive pair crossed a split or axis boundary")
        by_split_axis[(str(right["split"]), str(right["axis"]))].append(pair)

    splits: dict[str, dict[str, object]] = {}
    for (split, axis), axis_pairs in sorted(by_split_axis.items()):
        split_summary = splits.setdefault(split, {"pair_count": 0, "axes": {}})
        split_summary["pair_count"] = int(split_summary["pair_count"]) + len(axis_pairs)
        axes = split_summary["axes"]
        if not isinstance(axes, dict):  # pragma: no cover - local invariant
            raise RuntimeError("split axis summary is malformed")
        axes[axis] = _axis_summary(axis_pairs)
    correlations = [
        axis_summary["residual_lag_pearson"]
        for split_summary in splits.values()
        for axis_summary in split_summary["axes"].values()
        if axis_summary["residual_lag_pearson"] is not None
    ]
    return {
        "schema": _SCHEMA,
        "model_arm": model_arm,
        "pair_count": len(pairs),
        "splits": splits,
        "interpretation": {
            "maximum_absolute_residual_lag_pearson": (
                max(abs(float(value)) for value in correlations) if correlations else None
            ),
            "criterion": (
                "consecutive residual correlation must be measured before treating recurrent "
                "neural observations as conditionally independent; the report records evidence "
                "and does not embed a benchmark-specific acceptance threshold"
            ),
        },
    }


def build_report(path: str | Path, *, model_arm: str = "actual") -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    report = summarize_temporal_correlation(load_coordinate_rows(source), model_arm=model_arm)
    return {
        **report,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-arm", default="actual")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args.coordinate_rows, model_arm=args.model_arm)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

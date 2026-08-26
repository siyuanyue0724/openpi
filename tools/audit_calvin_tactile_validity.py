#!/usr/bin/env python3
"""Measure CALVIN tactile observability without inventing contact labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from picf_next.contracts import ContractError

AUDIT_SCHEMA = "picf-next.calvin-tactile-validity-audit/v1"
DEFAULT_THRESHOLDS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def _validate_episode_ranges(ranges: np.ndarray) -> np.ndarray:
    values = np.asarray(ranges)
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in "iu":
        raise ContractError("ep_start_end_ids must be an integer N-by-2 array")
    if values.shape[0] == 0:
        raise ContractError("ep_start_end_ids must contain at least one episode")
    if np.any(values[:, 0] < 0) or np.any(values[:, 1] < values[:, 0]):
        raise ContractError("episode ranges must be nonnegative inclusive intervals")
    if values.shape[0] > 1 and np.any(values[1:, 0] <= values[:-1, 1]):
        raise ContractError("episode ranges must be strictly ordered and non-overlapping")
    return values.astype(np.int64, copy=False)


def deterministic_sample_steps(
    episode_ranges: np.ndarray,
    *,
    sample_count: int,
    include_steps: Iterable[int] = (),
) -> tuple[int, ...]:
    """Uniformly sample positions over episode support and add requested steps."""

    ranges = _validate_episode_ranges(episode_ranges)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    lengths = ranges[:, 1] - ranges[:, 0] + 1
    cumulative = np.cumsum(lengths, dtype=np.int64)
    total = int(cumulative[-1])
    count = min(sample_count, total)
    positions = np.linspace(0, total - 1, num=count, dtype=np.int64)
    episode_ids = np.searchsorted(cumulative, positions, side="right")
    previous = np.where(episode_ids == 0, 0, cumulative[episode_ids - 1])
    sampled = ranges[episode_ids, 0] + positions - previous
    selected = {int(step) for step in sampled}
    for raw_step in include_steps:
        step = int(raw_step)
        episode_id = int(np.searchsorted(ranges[:, 1], step, side="left"))
        if episode_id >= ranges.shape[0] or not ranges[episode_id, 0] <= step:
            raise ContractError(f"requested tactile audit step is outside episode support: {step}")
        selected.add(step)
    return tuple(sorted(selected))


def _quantile_record(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, QUANTILES)
    return {
        f"q{quantile:g}": float(value) for quantile, value in zip(QUANTILES, quantiles, strict=True)
    }


def _representative_steps(mask: np.ndarray, steps: np.ndarray, *, limit: int = 3) -> list[int]:
    matches = np.flatnonzero(mask)
    if matches.size <= limit:
        chosen = matches
    else:
        chosen = matches[np.linspace(0, matches.size - 1, num=limit, dtype=np.int64)]
    return [int(steps[index]) for index in chosen]


def summarize_tactile_frames(
    frames: Iterable[np.ndarray],
    *,
    frame_steps: Iterable[int] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    signed_minima: list[list[float]] | None = None
    signed_maxima: list[list[float]] | None = None
    absolute_maxima: list[list[float]] | None = None
    signed_means: list[list[float]] | None = None
    absolute_means: list[list[float]] | None = None
    nonzero_fractions: list[list[float]] | None = None
    strongest_steps: list[int] | None = None
    observed_steps: list[int] = []
    shape: tuple[int, int, int] | None = None
    frame_count = 0
    step_iterator = iter(frame_steps) if frame_steps is not None else None
    for raw_frame in frames:
        if step_iterator is None:
            step = frame_count
        else:
            try:
                step = int(next(step_iterator))
            except StopIteration as exc:
                raise ContractError("tactile frames outnumber their frame steps") from exc
        observed_steps.append(step)
        frame = np.asarray(raw_frame)
        if frame.ndim != 3 or frame.shape[2] <= 0:
            raise ContractError("depth_tactile must be an H-by-W-by-sensor array")
        if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
            raise ContractError("depth_tactile must contain finite numeric measurements")
        if shape is None:
            shape = tuple(int(value) for value in frame.shape)
            signed_minima = [[] for _ in range(frame.shape[2])]
            signed_maxima = [[] for _ in range(frame.shape[2])]
            absolute_maxima = [[] for _ in range(frame.shape[2])]
            signed_means = [[] for _ in range(frame.shape[2])]
            absolute_means = [[] for _ in range(frame.shape[2])]
            nonzero_fractions = [[] for _ in range(frame.shape[2])]
            strongest_steps = [step for _ in range(frame.shape[2])]
        elif frame.shape != shape:
            raise ContractError("depth_tactile shape changed within one audit")
        assert signed_minima is not None and signed_maxima is not None
        assert absolute_maxima is not None and signed_means is not None
        assert absolute_means is not None and nonzero_fractions is not None
        assert strongest_steps is not None
        for sensor_id in range(frame.shape[2]):
            sensor = frame[..., sensor_id].astype(np.float64, copy=False)
            absolute = np.abs(sensor)
            signed_minima[sensor_id].append(float(sensor.min()))
            signed_maxima[sensor_id].append(float(sensor.max()))
            absolute_maxima[sensor_id].append(float(absolute.max()))
            signed_means[sensor_id].append(float(sensor.mean()))
            absolute_means[sensor_id].append(float(absolute.mean()))
            nonzero_fractions[sensor_id].append(float(np.count_nonzero(sensor) / sensor.size))
            if absolute_maxima[sensor_id][-1] >= max(absolute_maxima[sensor_id]):
                strongest_steps[sensor_id] = step
        frame_count += 1
    if frame_count == 0 or shape is None:
        raise ContractError("tactile audit received no frames")
    if step_iterator is not None:
        try:
            next(step_iterator)
        except StopIteration:
            pass
        else:
            raise ContractError("tactile frame steps outnumber the audited frames")

    sensor_records: list[dict[str, object]] = []
    assert signed_minima is not None and signed_maxima is not None
    assert absolute_maxima is not None and signed_means is not None
    assert absolute_means is not None and nonzero_fractions is not None
    assert strongest_steps is not None
    step_values = np.asarray(observed_steps, dtype=np.int64)
    boundaries = sorted(
        {
            float(threshold)
            for threshold in thresholds
            if math.isfinite(threshold) and threshold > 0.0
        }
    )
    for sensor_id, (
        sensor_minima,
        sensor_maxima,
        sensor_absolute_maxima,
        sensor_signed_means,
        sensor_absolute_means,
        sensor_nonzero,
    ) in enumerate(
        zip(
            signed_minima,
            signed_maxima,
            absolute_maxima,
            signed_means,
            absolute_means,
            nonzero_fractions,
            strict=True,
        )
    ):
        min_values = np.asarray(sensor_minima, dtype=np.float64)
        max_values = np.asarray(sensor_maxima, dtype=np.float64)
        absolute_max_values = np.asarray(sensor_absolute_maxima, dtype=np.float64)
        mean_values = np.asarray(sensor_signed_means, dtype=np.float64)
        absolute_mean_values = np.asarray(sensor_absolute_means, dtype=np.float64)
        nonzero_values = np.asarray(sensor_nonzero, dtype=np.float64)
        representatives = {
            "exact_zero": _representative_steps(absolute_max_values == 0.0, step_values)
        }
        lower = 0.0
        for upper in boundaries:
            representatives[f"({lower:.9g},{upper:.9g}]"] = _representative_steps(
                (absolute_max_values > lower) & (absolute_max_values <= upper),
                step_values,
            )
            lower = upper
        representatives[f">{lower:.9g}"] = _representative_steps(
            absolute_max_values > lower, step_values
        )
        sensor_records.append(
            {
                "sensor_index": sensor_id,
                "exact_zero_frames": int(np.count_nonzero(absolute_max_values == 0.0)),
                "strongest_absolute_deformation_step": strongest_steps[sensor_id],
                "representative_steps_by_absolute_max_band": representatives,
                "frame_signed_min": _quantile_record(min_values),
                "frame_signed_max": _quantile_record(max_values),
                "frame_absolute_max": _quantile_record(absolute_max_values),
                "frame_signed_mean": _quantile_record(mean_values),
                "frame_absolute_mean": _quantile_record(absolute_mean_values),
                "frame_nonzero_fraction": _quantile_record(nonzero_values),
                "frames_above_absolute_max_threshold": {
                    format(threshold, ".9g"): int(np.count_nonzero(absolute_max_values > threshold))
                    for threshold in thresholds
                },
            }
        )
    return {
        "frame_count": frame_count,
        "depth_tactile_shape": list(shape),
        "sensors": sensor_records,
    }


def build_report(
    split_root: Path,
    *,
    sample_count: int,
    include_steps: Iterable[int] = (),
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    split_root = split_root.resolve()
    ranges_path = split_root / "ep_start_end_ids.npy"
    if not ranges_path.is_file():
        raise FileNotFoundError(ranges_path)
    ranges = _validate_episode_ranges(np.load(ranges_path, allow_pickle=False))
    steps = deterministic_sample_steps(
        ranges,
        sample_count=sample_count,
        include_steps=include_steps,
    )

    def frames() -> Iterable[np.ndarray]:
        for step in steps:
            path = split_root / f"episode_{step:07d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                if "depth_tactile" not in payload.files:
                    raise ContractError(f"CALVIN frame lacks depth_tactile: {path}")
                yield np.asarray(payload["depth_tactile"])

    summary = summarize_tactile_frames(frames(), frame_steps=steps, thresholds=thresholds)
    step_bytes = np.asarray(steps, dtype="<i8").tobytes()
    total_frames = int(np.sum(ranges[:, 1] - ranges[:, 0] + 1))
    return {
        "schema": AUDIT_SCHEMA,
        "split_root": str(split_root),
        "episode_count": int(ranges.shape[0]),
        "episode_frame_count": total_frames,
        "requested_uniform_sample_count": sample_count,
        "sampled_steps_count": len(steps),
        "sampled_steps_sha256": hashlib.sha256(step_bytes).hexdigest(),
        "sampled_step_min": min(steps),
        "sampled_step_max": max(steps),
        "explicit_include_steps": sorted({int(step) for step in include_steps}),
        "thresholds_are_diagnostics_not_contact_labels": True,
        "measurement_summary": summary,
    }


def _positive_threshold(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("threshold must be finite and nonnegative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=2_000)
    parser.add_argument("--include-step", type=int, action="append", default=[])
    parser.add_argument("--threshold", type=_positive_threshold, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    thresholds = tuple(args.threshold) if args.threshold else DEFAULT_THRESHOLDS
    report = build_report(
        args.split_root,
        sample_count=args.sample_count,
        include_steps=args.include_step,
        thresholds=thresholds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

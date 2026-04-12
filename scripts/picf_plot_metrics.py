from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(int(window), 1)
    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        count = min(idx + 1, window)
        out.append(running / float(count))
    return out


def _load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _extract_series(records: list[dict[str, object]], field: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for idx, record in enumerate(records):
        value = _safe_float(record.get(field))
        if value is None:
            continue
        step = _safe_float(record.get("step"))
        xs.append(step if step is not None else float(idx + 1))
        ys.append(value)
    return xs, ys


def _plot_group(
    ax: plt.Axes,
    *,
    records: list[dict[str, object]],
    fields: list[str],
    smoothing_window: int,
    raw_alpha: float,
    title: str,
) -> None:
    plotted = False
    for field in fields:
        xs, ys = _extract_series(records, field)
        if not ys:
            continue
        plotted = True
        smoothed = _rolling_mean(ys, smoothing_window)
        ax.plot(xs, ys, alpha=raw_alpha, linewidth=1.0)
        ax.plot(xs, smoothed, linewidth=2.0, label=field)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PICF metrics.jsonl to PNG.")
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=50)
    parser.add_argument("--tail", type=int, default=0, help="Only plot the last N metric records. 0 means all.")
    parser.add_argument("--raw-alpha", type=float, default=0.18)
    args = parser.parse_args()

    records = _load_records(args.metrics_path)
    if not records:
        raise SystemExit(f"No metrics records found at: {args.metrics_path}")
    if args.tail and args.tail > 0:
        records = records[-args.tail :]

    output = args.output
    if output is None:
        output = args.metrics_path.with_name("metrics_trend.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    _plot_group(
        axes[0],
        records=records,
        fields=["loss_total", "loss_action", "loss_alignment", "loss_pt"],
        smoothing_window=args.smoothing_window,
        raw_alpha=args.raw_alpha,
        title=f"Core Loss Trend (rolling={args.smoothing_window})",
    )
    _plot_group(
        axes[1],
        records=records,
        fields=[
            "loss_action_pos",
            "loss_action_rot",
            "loss_action_gripper",
            "loss_visual_real",
            "loss_visual_latent",
            "loss_tactile_real",
            "loss_point_real",
            "loss_anchor_pv",
            "loss_focus_pv",
            "loss_pv_weak",
            "loss_semantic_future_aux",
        ],
        smoothing_window=args.smoothing_window,
        raw_alpha=args.raw_alpha,
        title="Detailed Loss Components",
    )
    _plot_group(
        axes[2],
        records=records,
        fields=[
            "tactile_active_rate",
            "tactile_contact_prob_mean",
            "projective_candidate_density",
            "steps_per_sec",
        ],
        smoothing_window=args.smoothing_window,
        raw_alpha=args.raw_alpha,
        title="Tactile / Runtime Diagnostics",
    )

    step_last = records[-1].get("step", "?")
    fig.suptitle(f"PICF training metrics through step={step_last}", fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"saved_plot={output}")
    print(f"records={len(records)}")


if __name__ == "__main__":
    main()

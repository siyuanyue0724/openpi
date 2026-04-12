from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _sparkline(values: list[float], *, width: int) -> str:
    if not values:
        return ""
    glyphs = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        step = len(values) / float(width)
        sampled = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
    else:
        sampled = values
    vmin = min(sampled)
    vmax = max(sampled)
    if math.isclose(vmin, vmax):
        return glyphs[0] * len(sampled)
    out = []
    for value in sampled:
        ratio = (value - vmin) / max(vmax - vmin, 1e-12)
        idx = min(int(round(ratio * (len(glyphs) - 1))), len(glyphs) - 1)
        out.append(glyphs[idx])
    return "".join(out)


def _summarize_field(name: str, records: list[dict[str, object]], *, window: int, spark_width: int) -> str | None:
    series = [_safe_float(record.get(name)) for record in records]
    clean = [value for value in series if value is not None]
    if not clean:
        return None
    recent = clean[-window:]
    last = recent[-1]
    avg = sum(recent) / float(len(recent))
    min_v = min(recent)
    max_v = max(recent)
    trend = _sparkline(clean, width=spark_width)
    return (
        f"{name}: last={last:.4f} avg{len(recent)}={avg:.4f} "
        f"min={min_v:.4f} max={max_v:.4f} trend={trend}"
    )


def _read_new_records(path: Path, *, offset: int) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
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
        offset = handle.tell()
    return records, offset


def _render(records: list[dict[str, object]], *, fields: list[str], window: int, spark_width: int) -> str:
    if not records:
        return "No metrics records yet."
    latest = records[-1]
    step = latest.get("step", "?")
    lines = [f"step={step} records={len(records)}"]
    for field in fields:
        summary = _summarize_field(field, records, window=window, spark_width=spark_width)
        if summary is not None:
            lines.append(summary)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight PICF metrics.jsonl watcher.")
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--fields", nargs="+", default=[
        "loss_total",
        "loss_action",
        "loss_pt",
        "tactile_active_rate",
        "tactile_contact_prob_mean",
    ])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--spark-width", type=int, default=24)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--clear-screen", action="store_true")
    args = parser.parse_args()

    path = args.metrics_path
    records: list[dict[str, object]] = []
    offset = 0

    while True:
        if path.exists():
            new_records, offset = _read_new_records(path, offset=offset)
            if new_records:
                records.extend(new_records)
            output = _render(records, fields=args.fields, window=max(args.window, 1), spark_width=max(args.spark_width, 4))
        else:
            output = f"Waiting for metrics file: {path}"

        if args.clear_screen:
            print("\033[2J\033[H", end="")
        print(output, flush=True)

        if not args.follow:
            break
        time.sleep(max(args.interval, 0.2))


if __name__ == "__main__":
    main()

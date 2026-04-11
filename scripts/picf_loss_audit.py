from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


_LOSS_KEYS = (
    "loss_total",
    "loss_action",
    "loss_visual_latent",
    "loss_visual_real",
    "loss_tactile_real",
    "loss_point_real",
    "loss_semantic_future_aux",
    "loss_alignment",
    "loss_pt",
    "loss_anchor_pv",
    "loss_pv_weak",
    "loss_focus_pv",
)


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.startswith("{"):
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if "loss_total" not in payload:
            continue
        rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PICF weighted loss contributions from JSONL-like logs.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--tail", type=int, default=300)
    args = parser.parse_args()

    rows = _load_rows(Path(args.log).expanduser())
    if not rows:
        raise SystemExit(f"No JSON loss rows found in {args.log!r}.")
    if args.tail > 0:
        rows = rows[-int(args.tail) :]

    means = {key: mean(float(row.get(key, 0.0)) for row in rows) for key in _LOSS_KEYS}
    total = max(means["loss_total"], 1e-9)
    ratios = {key: (value / total) for key, value in means.items() if key != "loss_total"}

    print(
        json.dumps(
            {
                "num_rows": len(rows),
                "means": means,
                "ratios_to_total": ratios,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

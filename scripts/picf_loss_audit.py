from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


_LOSS_KEYS = (
    "loss_total",
    "loss_action",
    "loss_action_pos",
    "loss_action_rot",
    "loss_action_gripper",
    "loss_visual_latent",
    "loss_visual_real",
    "loss_tactile_real",
    "loss_point_real",
    "loss_semantic_future_aux",
    "loss_alignment",
    "loss_pt",
    "loss_anchor_pv",
    "loss_pv_weak",
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
    parser.add_argument("--action-pos-weight", type=float, default=None)
    parser.add_argument("--action-rot-weight", type=float, default=None)
    parser.add_argument("--action-gripper-weight", type=float, default=None)
    args = parser.parse_args()

    rows = _load_rows(Path(args.log).expanduser())
    if not rows:
        raise SystemExit(f"No JSON loss rows found in {args.log!r}.")
    if args.tail > 0:
        rows = rows[-int(args.tail) :]

    means = {key: mean(float(row.get(key, 0.0)) for row in rows) for key in _LOSS_KEYS}
    total = max(means["loss_total"], 1e-9)
    ratios = {key: (value / total) for key, value in means.items() if key != "loss_total"}
    payload: dict[str, object] = {
        "num_rows": len(rows),
        "means": means,
        "ratios_to_total": ratios,
    }

    if (
        args.action_pos_weight is not None
        and args.action_rot_weight is not None
        and args.action_gripper_weight is not None
        and all(
            any(key in row for row in rows)
            for key in ("loss_action_pos", "loss_action_rot", "loss_action_gripper", "loss_action", "loss_total")
        )
    ):
        alt_action_values = [
            (float(args.action_pos_weight) * float(row.get("loss_action_pos", 0.0)))
            + (float(args.action_rot_weight) * float(row.get("loss_action_rot", 0.0)))
            + (float(args.action_gripper_weight) * float(row.get("loss_action_gripper", 0.0)))
            for row in rows
        ]
        other_values = [
            float(row.get("loss_total", 0.0)) - float(row.get("loss_action", 0.0))
            for row in rows
        ]
        alt_action_mean = mean(alt_action_values)
        alt_total_mean = mean(other_values) + alt_action_mean
        payload["counterfactual_action_weights"] = {
            "lambda_action_pos": float(args.action_pos_weight),
            "lambda_action_rot": float(args.action_rot_weight),
            "lambda_action_gripper": float(args.action_gripper_weight),
            "mean_loss_action": alt_action_mean,
            "mean_loss_total": alt_total_mean,
            "loss_action_ratio_to_total": alt_action_mean / max(alt_total_mean, 1e-9),
        }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

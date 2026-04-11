from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _load_json(path: str | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _load_metric_rows(paths: list[str]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue
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


def _check(status: str, name: str, detail: str, *, value: float | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"status": status, "name": name, "detail": detail}
    if value is not None:
        payload["value"] = value
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PICF tactile deployment artifacts and run metrics.")
    parser.add_argument("--contact-stats", default=None)
    parser.add_argument("--fingertip-calibration", default=None)
    parser.add_argument("--metrics", action="append", default=[])
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    contact_stats = _load_json(args.contact_stats)
    fingertip = _load_json(args.fingertip_calibration)
    metric_rows = _load_metric_rows(list(args.metrics))

    if contact_stats is not None:
        tau_on = float(contact_stats.get("tau_on", 0.0))
        tau_off = float(contact_stats.get("tau_off", 0.0))
        active_rate = float(contact_stats.get("active_rate_tau_on", 0.0))
        neg_active_rate = contact_stats.get("negative_active_rate_tau_on")
        neg_pool = int(contact_stats.get("negative_pool_size", 0))
        if tau_on > tau_off:
            checks.append(_check("pass", "contact_threshold_order", f"tau_on={tau_on:.6f} > tau_off={tau_off:.6f}", value=tau_on - tau_off))
        else:
            checks.append(_check("fail", "contact_threshold_order", f"tau_on={tau_on:.6f} <= tau_off={tau_off:.6f}", value=tau_on - tau_off))
        if neg_pool >= 16:
            checks.append(_check("pass", "negative_pool_size", f"negative_pool_size={neg_pool}", value=float(neg_pool)))
        elif neg_pool > 0:
            checks.append(_check("warn", "negative_pool_size", f"negative_pool_size={neg_pool} is usable but shallow", value=float(neg_pool)))
        else:
            checks.append(_check("fail", "negative_pool_size", "negative_pool_size=0", value=float(neg_pool)))
        if neg_active_rate is not None:
            neg_active_f = float(neg_active_rate)
            if neg_active_f <= 0.02:
                checks.append(_check("pass", "negative_tail_rate", f"negative_active_rate_tau_on={neg_active_f:.6f}", value=neg_active_f))
            elif neg_active_f <= 0.05:
                checks.append(_check("warn", "negative_tail_rate", f"negative_active_rate_tau_on={neg_active_f:.6f} is higher than ideal", value=neg_active_f))
            else:
                checks.append(_check("fail", "negative_tail_rate", f"negative_active_rate_tau_on={neg_active_f:.6f} is too high", value=neg_active_f))
        if 0.0 < active_rate < 0.95:
            checks.append(_check("pass", "observed_contact_rate", f"active_rate_tau_on={active_rate:.6f}", value=active_rate))
        else:
            checks.append(_check("fail", "observed_contact_rate", f"active_rate_tau_on={active_rate:.6f} is degenerate", value=active_rate))

    if fingertip is not None:
        d_nn = fingertip.get("d_nn_trimmed_mean")
        front_ratio = fingertip.get("front_ratio")
        radius = fingertip.get("recommended_pt_bag_radius_m")
        if d_nn is not None:
            d_nn_f = float(d_nn)
            checks.append(
                _check(
                    "pass" if d_nn_f < 0.03 else "fail",
                    "fingertip_nn_distance",
                    f"d_nn_trimmed_mean={d_nn_f:.6f}",
                    value=d_nn_f,
                )
            )
        if front_ratio is not None:
            front_f = float(front_ratio)
            if front_f >= 0.6:
                status = "pass"
                detail = f"front_ratio={front_f:.3f}"
            elif front_f >= 0.5:
                status = "warn"
                detail = f"front_ratio={front_f:.3f}; usable but below ideal 0.6"
            else:
                status = "fail"
                detail = f"front_ratio={front_f:.3f} < 0.5"
            checks.append(_check(status, "fingertip_front_ratio", detail, value=front_f))
        if radius is not None:
            radius_f = float(radius)
            status = "pass" if 0.035 <= radius_f <= 0.055 else "fail"
            checks.append(_check(status, "recommended_bag_radius", f"recommended_pt_bag_radius_m={radius_f:.6f}", value=radius_f))

    if metric_rows:
        tactile_prob_rows = [float(row.get("tactile_contact_prob_mean", 0.0)) for row in metric_rows if "tactile_contact_prob_mean" in row]
        tactile_active_rows = [float(row.get("tactile_active_rate", 0.0)) for row in metric_rows if "tactile_active_rate" in row]
        loss_pt_rows = [float(row.get("loss_pt", 0.0)) for row in metric_rows if "loss_pt" in row]
        loss_action_rows = [float(row.get("loss_action", 0.0)) for row in metric_rows if "loss_action" in row]
        loss_total_rows = [float(row.get("loss_total", 0.0)) for row in metric_rows if "loss_total" in row]

        if tactile_prob_rows:
            mean_prob = mean(tactile_prob_rows)
            status = "pass" if 0.05 <= mean_prob <= 0.95 else "fail"
            checks.append(_check(status, "train_contact_prob_mean", f"mean tactile_contact_prob_mean={mean_prob:.6f}", value=mean_prob))
        if tactile_active_rows:
            mean_active = mean(tactile_active_rows)
            status = "pass" if 0.05 <= mean_active <= 0.95 else "fail"
            checks.append(_check(status, "train_active_rate", f"mean tactile_active_rate={mean_active:.6f}", value=mean_active))
        if loss_pt_rows:
            nonzero_rate = mean(1.0 if abs(value) > 1e-8 else 0.0 for value in loss_pt_rows)
            status = "pass" if nonzero_rate > 0.1 else "fail"
            checks.append(_check(status, "train_loss_pt_nonzero_rate", f"loss_pt_nonzero_rate={nonzero_rate:.6f}", value=nonzero_rate))
        if loss_action_rows and loss_total_rows:
            action_ratio = mean(loss_action_rows) / max(mean(loss_total_rows), 1e-9)
            if action_ratio <= 0.9:
                status = "pass"
                detail = f"loss_action/loss_total={action_ratio:.6f}"
            elif action_ratio <= 0.95:
                status = "warn"
                detail = f"loss_action/loss_total={action_ratio:.6f}; action already dominates"
            else:
                status = "fail"
                detail = f"loss_action/loss_total={action_ratio:.6f}; auxiliaries are likely over-suppressed"
            checks.append(_check(status, "train_action_dominance", detail, value=action_ratio))

    overall = "pass"
    if any(check["status"] == "fail" for check in checks):
        overall = "fail"
    elif any(check["status"] == "warn" for check in checks):
        overall = "warn"

    print(json.dumps({"overall_status": overall, "checks": checks}, ensure_ascii=False, indent=2))
    if overall == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.anytouch.wrapper import AnyTouch2TactileEncoder
from openpi.picf.geometry import normalize_vectors
from openpi.picf.geometry import transform_points
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame
from openpi.training.calvin_dataset import CalvinLangSegmentDataset


def _split_tactile_rgb(rgb_tactile: np.ndarray, sensor_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    rgb = np.asarray(rgb_tactile)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 * len(sensor_names):
        raise ValueError(
            f"Expected rgb_tactile [H,W,{3 * len(sensor_names)}], got {tuple(rgb.shape)} "
            f"for sensors={sensor_names}."
        )
    return {
        sensor_name: rgb[..., 3 * index : 3 * (index + 1)]
        for index, sensor_name in enumerate(sensor_names)
    }


def _valid_center_mask(image: np.ndarray, fraction: float = 0.8) -> np.ndarray:
    height, width = image.shape[:2]
    keep_h = max(1, int(round(height * float(fraction))))
    keep_w = max(1, int(round(width * float(fraction))))
    y0 = max(0, (height - keep_h) // 2)
    x0 = max(0, (width - keep_w) // 2)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + keep_h, x0 : x0 + keep_w] = True
    return mask


def _masked_mean_abs_delta(current: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    current_f = np.asarray(current, dtype=np.float32) / 255.0
    reference_f = np.asarray(reference, dtype=np.float32) / 255.0
    delta = np.abs(current_f - reference_f).mean(axis=-1)
    return float(delta[mask].mean())


def _robust_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(values))
    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = max(q3 - q1, 1e-8)
    return {
        "median": median,
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "q999": float(np.quantile(values, 0.999)),
        "iqr": float(iqr),
    }


def _zscore(values: np.ndarray, *, median: float, iqr: float) -> np.ndarray:
    return (values - float(median)) / ((float(iqr) / 1.349) + 1e-8)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _trimmed_mean(values: np.ndarray, trim: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if values.size == 0:
        return float("inf")
    cut = int(math.floor(values.size * float(trim)))
    if cut * 2 >= values.size:
        return float(values.mean())
    return float(values[cut : values.size - cut].mean())


def _local_sensor_centers(
    *,
    width: float,
    u_open_local: np.ndarray,
    o_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = o_local + (0.5 * float(width) * u_open_local)
    right = o_local - (0.5 * float(width) * u_open_local)
    return left.astype(np.float32), right.astype(np.float32)


@dataclass(frozen=True)
class _FrameRecord:
    step_id: int
    robot_obs: np.ndarray
    rgb_static: np.ndarray
    depth_static: np.ndarray
    rgb_gripper: np.ndarray | None
    depth_gripper: np.ndarray | None
    tactile_rgb_by_sensor: dict[str, np.ndarray]


@dataclass(frozen=True)
class _FingertipSearchFrame:
    record: _FrameRecord
    G_t: np.ndarray
    xyz_world: np.ndarray


def _load_sampled_records(
    *,
    root: str,
    split: str,
    backend: str,
    sensor_names: tuple[str, ...],
    sample_stride: int,
    max_frames: int | None,
) -> list[_FrameRecord]:
    dataset = CalvinLangSegmentDataset(
        root=root,
        split=split,
        action_horizon=1,
        backend=backend,
        use_wrist_rgb=True,
        sample_within_segment=False,
    )
    reader = dataset.reader
    step_ids: list[int] = []
    for segment in dataset.segments:
        step_ids.extend(range(segment.start, segment.end))
    step_ids = step_ids[:: max(int(sample_stride), 1)]
    if max_frames is not None:
        step_ids = step_ids[: int(max_frames)]
    records: list[_FrameRecord] = []
    for step_id in step_ids:
        frame = reader.read_npz(
            step_id,
            keys=[
                "rgb_static",
                "depth_static",
                "rgb_gripper",
                "depth_gripper",
                "rgb_tactile",
                "robot_obs",
            ],
        )
        records.append(
            _FrameRecord(
                step_id=int(step_id),
                robot_obs=np.asarray(frame["robot_obs"], dtype=np.float32),
                rgb_static=np.asarray(frame["rgb_static"]),
                depth_static=np.asarray(frame["depth_static"], dtype=np.float32),
                rgb_gripper=None if frame.get("rgb_gripper") is None else np.asarray(frame["rgb_gripper"]),
                depth_gripper=None if frame.get("depth_gripper") is None else np.asarray(frame["depth_gripper"], dtype=np.float32),
                tactile_rgb_by_sensor=_split_tactile_rgb(frame["rgb_tactile"], sensor_names),
            )
        )
    reader.close()
    return records


def _search_grids() -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    ]
    x_grid = np.arange(-0.02, 0.0201, 0.005, dtype=np.float32)
    y_grid = np.arange(-0.02, 0.0201, 0.005, dtype=np.float32)
    z_grid = np.arange(-0.03, 0.0301, 0.005, dtype=np.float32)
    return axes, x_grid, y_grid, z_grid


def _precompute_fingertip_search_frames(
    *,
    calvin_root: str,
    records: list[_FrameRecord],
    top_indices: np.ndarray,
    point_stride: int,
    point_max_points: int,
    point_crop_radius_m: float,
    offset_norm_max: float,
) -> list[_FingertipSearchFrame]:
    if top_indices.size == 0:
        return []
    builder = CalvinDepthToPicfPointCloud(
        calvin_root,
        stride=point_stride,
        max_points=max(int(point_max_points), 4096),
        min_peripheral_points=0,
    )
    local_frame = EndEffectorLocalFrame()
    selected = [records[int(index)] for index in top_indices]
    max_width = max(float(record.robot_obs[6]) for record in selected)
    support_radius_m = float(point_crop_radius_m) + float(offset_norm_max) + (0.5 * max_width)
    cached: list[_FingertipSearchFrame] = []
    for record in selected:
        G_t = local_frame.make_transform(record.robot_obs)
        tcp_center_world = np.asarray(G_t[:3, 3], dtype=np.float32)
        point_set = builder(
            {
                "rgb_static": record.rgb_static,
                "depth_static": record.depth_static,
                "rgb_gripper": record.rgb_gripper,
                "depth_gripper": record.depth_gripper,
                "robot_obs": record.robot_obs,
                "focus_center_world": tcp_center_world,
                "focus_radius_m": support_radius_m,
            }
        )
        cached.append(
            _FingertipSearchFrame(
                record=record,
                G_t=G_t,
                xyz_world=np.asarray(point_set.xyz_world, dtype=np.float32),
            )
        )
    return cached


def _compute_global_backgrounds(
    *,
    records: list[_FrameRecord],
    sensor_names: tuple[str, ...],
    retain_fraction: float,
) -> dict[str, np.ndarray]:
    masks = {sensor_name: _valid_center_mask(records[0].tactile_rgb_by_sensor[sensor_name]) for sensor_name in sensor_names}
    frames_by_sensor = {
        sensor_name: np.stack([record.tactile_rgb_by_sensor[sensor_name].astype(np.float32) for record in records], axis=0)
        for sensor_name in sensor_names
    }
    rough_backgrounds = {
        sensor_name: np.median(frames_by_sensor[sensor_name], axis=0).astype(np.uint8)
        for sensor_name in sensor_names
    }
    refined_backgrounds: dict[str, np.ndarray] = {}
    for sensor_name in sensor_names:
        mask = masks[sensor_name]
        frames = frames_by_sensor[sensor_name]
        scores = np.asarray(
            [_masked_mean_abs_delta(frame, rough_backgrounds[sensor_name], mask) for frame in frames],
            dtype=np.float32,
        )
        keep = max(1, int(math.ceil(scores.shape[0] * float(retain_fraction))))
        indices = np.argsort(scores)[:keep]
        refined_backgrounds[sensor_name] = np.median(frames[indices], axis=0).astype(np.uint8)
    return refined_backgrounds


def _compute_latent_backgrounds_and_scores(
    *,
    encoder: AnyTouch2TactileEncoder | None,
    records: list[_FrameRecord],
    backgrounds: dict[str, np.ndarray],
    sensor_names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray] | None, list[dict[str, float]], list[dict[str, float]], dict[str, np.ndarray] | None]:
    masks = {sensor_name: _valid_center_mask(records[0].tactile_rgb_by_sensor[sensor_name]) for sensor_name in sensor_names}
    rgb_scores_per_record: list[dict[str, float]] = []
    for record in records:
        rgb_scores_per_record.append(
            {
                sensor_name: _masked_mean_abs_delta(
                    record.tactile_rgb_by_sensor[sensor_name],
                    backgrounds[sensor_name],
                    masks[sensor_name],
                )
                for sensor_name in sensor_names
            }
        )
    if encoder is None:
        return None, rgb_scores_per_record, [{sensor_name: 0.0 for sensor_name in sensor_names} for _ in records], None

    pooled_rows: list[dict[str, np.ndarray]] = []
    for record in records:
        bundle = encoder.encode_sensor_clips(
            clips_by_sensor={sensor_name: np.repeat(record.tactile_rgb_by_sensor[sensor_name][None, ...], encoder.config.num_frames, axis=0) for sensor_name in sensor_names},
            backgrounds_by_sensor=backgrounds,
            poses_by_sensor={sensor_name: np.eye(4, dtype=np.float32) for sensor_name in sensor_names},
        )
        assert bundle is not None
        pooled_row: dict[str, np.ndarray] = {}
        for sensor_name in sensor_names:
            sensor = bundle.sensors[sensor_name]
            pooled = sensor.pooled_feature.detach().cpu().numpy().astype(np.float32)
            pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
            pooled_row[sensor_name] = pooled
        pooled_rows.append(pooled_row)

    rgb_mean_scores = np.asarray(
        [
            np.mean([float(row[sensor_name]) for sensor_name in sensor_names], dtype=np.float64)
            for row in rgb_scores_per_record
        ],
        dtype=np.float32,
    )
    neg_count = max(1, int(math.ceil(rgb_mean_scores.shape[0] * 0.05)))
    neg_indices = np.argsort(rgb_mean_scores)[:neg_count]
    z_bg = {}
    for sensor_name in sensor_names:
        negatives = np.stack([pooled_rows[int(index)][sensor_name] for index in neg_indices], axis=0)
        z_bg[sensor_name] = normalize_vectors(np.mean(negatives, axis=0, keepdims=True))[0]

    final_rows: list[dict[str, float]] = []
    for pooled_row in pooled_rows:
        row: dict[str, float] = {}
        for sensor_name in sensor_names:
            pooled = pooled_row[sensor_name]
            row[sensor_name] = float(1.0 - np.dot(pooled, z_bg[sensor_name]))
        final_rows.append(row)
    return z_bg, rgb_scores_per_record, final_rows, {"negative_rgb_mean_scores": rgb_mean_scores, "negative_indices": neg_indices.astype(np.int64)}


def _calibrate_fingertips(
    *,
    calvin_root: str,
    records: list[_FrameRecord],
    combined_scores: np.ndarray,
    top_fraction: float,
    point_stride: int,
    point_max_points: int,
    point_crop_radius_m: float,
    front_radius_m: float,
    front_slack_m: float,
) -> dict[str, object]:
    count = max(1, int(math.ceil(combined_scores.shape[0] * float(top_fraction))))
    top_indices = np.argsort(combined_scores)[-count:]
    axes, x_grid, y_grid, z_grid = _search_grids()
    offset_norm_max = float(
        np.sqrt(
            float(np.max(np.abs(x_grid))) ** 2
            + float(np.max(np.abs(y_grid))) ** 2
            + float(np.max(np.abs(z_grid))) ** 2
        )
    )
    search_frames = _precompute_fingertip_search_frames(
        calvin_root=calvin_root,
        records=records,
        top_indices=top_indices,
        point_stride=point_stride,
        point_max_points=point_max_points,
        point_crop_radius_m=point_crop_radius_m,
        offset_norm_max=offset_norm_max,
    )
    best: dict[str, object] | None = None
    for axis in axes:
        for ox in x_grid:
            for oy in y_grid:
                for oz in z_grid:
                    o_local = np.array([ox, oy, oz], dtype=np.float32)
                    dist_terms: list[float] = []
                    front_terms: list[float] = []
                    for search_frame in search_frames:
                        record = search_frame.record
                        G_t = search_frame.G_t
                        left_local, right_local = _local_sensor_centers(
                            width=float(record.robot_obs[6]),
                            u_open_local=axis,
                            o_local=o_local,
                        )
                        normals_local = np.stack([axis, -axis], axis=0).astype(np.float32)
                        centers_world = transform_points(np.stack([left_local, right_local], axis=0), G_t)
                        normals_world = normalize_vectors((G_t[:3, :3] @ normals_local.T).T)
                        xyz = search_frame.xyz_world
                        if xyz.shape[0] == 0:
                            dist_terms.extend([point_crop_radius_m, point_crop_radius_m])
                            front_terms.extend([0.0, 0.0])
                            continue
                        for sensor_index in range(2):
                            diffs = xyz - centers_world[sensor_index][None, :]
                            dists = np.linalg.norm(diffs, axis=1)
                            dist_terms.append(float(dists.min()))
                            in_radius = dists <= float(front_radius_m)
                            if not np.any(in_radius):
                                front_terms.append(0.0)
                                continue
                            front = np.dot(diffs[in_radius], normals_world[sensor_index]) >= -float(front_slack_m)
                            front_terms.append(float(front.mean()))
                    trimmed = _trimmed_mean(np.asarray(dist_terms), trim=0.2)
                    front_ratio = float(np.mean(front_terms)) if front_terms else 0.0
                    objective = trimmed + (0.02 * max(0.0, 0.6 - front_ratio))
                    if best is None or objective < float(best["objective"]):
                        best = {
                            "objective": float(objective),
                            "u_open_local": axis.tolist(),
                            "o_local": o_local.tolist(),
                            "d_nn_trimmed_mean": float(trimmed),
                            "front_ratio": float(front_ratio),
                            "evaluated_frames": int(len(top_indices)),
                        }
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CALVIN tactile backgrounds, contact stats, and fingertip calibration.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--backend", choices=["dir", "zip"], default="dir")
    parser.add_argument("--sensor-names", default="digit,gelsight_mini")
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--background-retain-fraction", type=float, default=0.10)
    parser.add_argument("--contact-top-fraction", type=float, default=0.05)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--point-max-points", type=int, default=1024)
    parser.add_argument("--point-crop-radius-m", type=float, default=0.10)
    parser.add_argument("--front-radius-m", type=float, default=0.05)
    parser.add_argument("--front-slack-m", type=float, default=0.008)
    parser.add_argument("--skip-fingertip-calibration", action="store_true")
    parser.add_argument("--anytouch-checkpoint-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sensor_names = tuple(part.strip() for part in str(args.sensor_names).split(",") if part.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_sampled_records(
        root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        sensor_names=sensor_names,
        sample_stride=args.sample_stride,
        max_frames=args.max_frames,
    )
    if not records:
        raise RuntimeError("No CALVIN frames loaded for tactile calibration.")

    backgrounds = _compute_global_backgrounds(
        records=records,
        sensor_names=sensor_names,
        retain_fraction=args.background_retain_fraction,
    )

    encoder = None
    if args.anytouch_checkpoint_path:
        encoder = AnyTouch2TactileEncoder(
            AnyTouchConfig(
                checkpoint_path=args.anytouch_checkpoint_path,
                device=args.device,
                dtype="float32",
                allow_random_init=False,
                require_background=True,
            )
        )

    z_bg, rgb_scores, latent_scores, calibration_aux = _compute_latent_backgrounds_and_scores(
        encoder=encoder,
        records=records,
        backgrounds=backgrounds,
        sensor_names=sensor_names,
    )

    negative_indices = (
        np.asarray(calibration_aux["negative_indices"], dtype=np.int64)
        if calibration_aux is not None and "negative_indices" in calibration_aux
        else np.arange(len(records), dtype=np.int64)
    )
    per_sensor_rgb_stats = {
        sensor_name: _robust_stats(
            np.asarray([rgb_scores[int(index)][sensor_name] for index in negative_indices], dtype=np.float32)
        )
        for sensor_name in sensor_names
    }
    per_sensor_latent_stats = (
        {
            sensor_name: _robust_stats(
                np.asarray([latent_scores[int(index)][sensor_name] for index in negative_indices], dtype=np.float32)
            )
            for sensor_name in sensor_names
        }
        if encoder is not None
        else {}
    )

    score_rows: list[dict[str, float]] = []
    combined_scores: list[float] = []
    for index, record in enumerate(records):
        row: dict[str, float] = {"step_id": float(record.step_id)}
        combined_per_sensor = []
        for sensor_name in sensor_names:
            rgb_value = float(rgb_scores[index][sensor_name])
            latent_value = float(latent_scores[index][sensor_name]) if latent_scores is not None else 0.0
            rgb_score = _zscore(
                np.asarray([rgb_value], dtype=np.float32),
                median=per_sensor_rgb_stats[sensor_name]["median"],
                iqr=per_sensor_rgb_stats[sensor_name]["iqr"],
            )[0]
            if encoder is not None:
                latent_score = _zscore(
                    np.asarray([latent_value], dtype=np.float32),
                    median=per_sensor_latent_stats[sensor_name]["median"],
                    iqr=per_sensor_latent_stats[sensor_name]["iqr"],
                )[0]
                combined = float(0.5 * (rgb_score + latent_score))
            else:
                latent_score = 0.0
                combined = float(rgb_score)
            row[f"{sensor_name}_rgb_residual"] = rgb_value
            row[f"{sensor_name}_latent_residual"] = latent_value
            row[f"{sensor_name}_rgb_score"] = float(rgb_score)
            row[f"{sensor_name}_latent_score"] = float(latent_score)
            row[f"{sensor_name}_score"] = combined
            combined_per_sensor.append(combined)
        row["combined_score"] = float(np.mean(combined_per_sensor))
        score_rows.append(row)
        combined_scores.append(float(row["combined_score"]))
    combined_scores_np = np.asarray(combined_scores, dtype=np.float32)
    negative_combined_scores = np.asarray(
        [score_rows[int(index)]["combined_score"] for index in negative_indices],
        dtype=np.float32,
    )
    negative_score_stats = _robust_stats(negative_combined_scores)
    score_stats_all = _robust_stats(combined_scores_np)
    tau_off = max(float(negative_score_stats["q99"]), float(score_stats_all["q75"]))
    tau_on = max(float(negative_score_stats["q999"]), float(score_stats_all["q90"]))
    score_stats = {
        "threshold_source": "hybrid_negative_tail_plus_global_operating_band",
        "negative_pool": negative_score_stats,
        "all_frames": score_stats_all,
    }
    tau_mid = 0.5 * (tau_on + tau_off)
    temperature = max(0.5 * (tau_on - tau_off), 1e-3)
    contact_prob = _sigmoid((combined_scores_np - tau_mid) / temperature)
    active_on = combined_scores_np > tau_on

    calibration_payload: dict[str, object]
    if args.skip_fingertip_calibration:
        calibration_payload = {
            "sensor_names": list(sensor_names),
            "u_open_local": None,
            "o_local": None,
            "left_normal_sign": 1.0,
            "right_normal_sign": -1.0,
            "d_nn_trimmed_mean": None,
            "front_ratio": None,
            "recommended_pt_bag_radius_m": None,
            "recommended_pt_bag_sigma_m": None,
            "pending_geometry_calibration": True,
        }
    else:
        calibration = _calibrate_fingertips(
            calvin_root=args.calvin_root,
            records=records,
            combined_scores=combined_scores_np,
            top_fraction=args.contact_top_fraction,
            point_stride=args.point_stride,
            point_max_points=args.point_max_points,
            point_crop_radius_m=args.point_crop_radius_m,
            front_radius_m=args.front_radius_m,
            front_slack_m=args.front_slack_m,
        )
        d_nn_median = float(calibration["d_nn_trimmed_mean"])
        recommended_radius = float(np.clip((2.0 * d_nn_median) + 0.015, 0.035, 0.055))
        calibration_payload = {
            "sensor_names": list(sensor_names),
            "u_open_local": calibration["u_open_local"],
            "o_local": calibration["o_local"],
            "left_normal_sign": 1.0,
            "right_normal_sign": -1.0,
            "d_nn_trimmed_mean": calibration["d_nn_trimmed_mean"],
            "front_ratio": calibration["front_ratio"],
            "recommended_pt_bag_radius_m": recommended_radius,
            "recommended_pt_bag_sigma_m": recommended_radius / 3.0,
            "pending_geometry_calibration": False,
        }

    np.savez(output_dir / "tactile_backgrounds.npz", **backgrounds)
    stats_payload = {
        "sensor_names": list(sensor_names),
        "sampled_frames": int(len(records)),
        "negative_pool_size": int(negative_indices.shape[0]),
        "score_mode": "rgb_latent" if encoder is not None else "rgb_only",
        "score_stats": score_stats,
        "tau_off": float(tau_off),
        "tau_on": float(tau_on),
        "tau_mid": float(tau_mid),
        "temperature": float(temperature),
        "active_rate_tau_on": float(np.mean(active_on)),
        "negative_active_rate_tau_on": float(np.mean(negative_combined_scores > tau_on)),
        "contact_prob_mean": float(np.mean(contact_prob)),
        "contact_prob_q90": float(np.quantile(contact_prob, 0.90)),
        "per_sensor_rgb_stats": per_sensor_rgb_stats,
        "per_sensor_latent_stats": per_sensor_latent_stats,
        "top_frames": [
            {
                "step_id": int(records[int(idx)].step_id),
                "score": float(combined_scores_np[int(idx)]),
                "gripper_width": float(records[int(idx)].robot_obs[6]),
            }
            for idx in np.argsort(combined_scores_np)[-10:][::-1]
        ],
    }
    (output_dir / "tactile_contact_stats.json").write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (output_dir / "tactile_fingertip_calibration.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_dir),
        "sampled_frames": int(len(records)),
        "tau_off": float(tau_off),
        "tau_on": float(tau_on),
        "active_rate_tau_on": float(np.mean(active_on)),
        "u_open_local": calibration_payload["u_open_local"],
        "o_local": calibration_payload["o_local"],
        "recommended_pt_bag_radius_m": calibration_payload["recommended_pt_bag_radius_m"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

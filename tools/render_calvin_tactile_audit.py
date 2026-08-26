#!/usr/bin/env python3
"""Render task-labelled CALVIN tactile validity panels for human review."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_tactile import CALVIN_TACTILE_STREAM_NAMES

SENSOR_NAMES = CALVIN_TACTILE_STREAM_NAMES


def _task_labels(annotation_path: Path, steps: tuple[int, ...]) -> dict[int, tuple[str, str]]:
    payload = np.load(annotation_path, allow_pickle=True).item()
    try:
        intervals = payload["info"]["indx"]
        task_keys = payload["language"]["task"]
        descriptions = payload["language"]["ann"]
    except (KeyError, TypeError) as exc:
        raise ContractError("CALVIN language annotations violate their released schema") from exc
    if not (len(intervals) == len(task_keys) == len(descriptions)):
        raise ContractError("CALVIN language annotation columns do not align")
    result = {step: ("unannotated", "no language segment covers this frame") for step in steps}
    for interval, task_key, description in zip(intervals, task_keys, descriptions, strict=True):
        start, end = (int(value) for value in interval)
        for step in steps:
            if start <= step <= end:
                result[step] = (str(task_key), str(description))
    return result


def split_tactile_rgb(rgb_tactile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb_tactile)
    if image.ndim != 3 or image.shape[2] != 6 or image.dtype != np.uint8:
        raise ContractError("CALVIN rgb_tactile must be H-by-W-by-6 uint8")
    return image[..., :3], image[..., 3:]


def _load_frame(split_root: Path, step: int) -> dict[str, np.ndarray]:
    path = split_root / f"episode_{step:07d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"depth_tactile", "rgb_gripper", "rgb_static", "rgb_tactile"}
        if not required <= set(payload.files):
            raise ContractError(f"CALVIN tactile review frame is incomplete: {path}")
        return {key: np.array(payload[key], copy=True) for key in required}


def _render_step(
    *,
    frame: dict[str, np.ndarray],
    step: int,
    task_key: str,
    description: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    tactile_rgb = split_tactile_rgb(frame["rgb_tactile"])
    depth = np.asarray(frame["depth_tactile"], dtype=np.float32)
    if depth.ndim != 3 or depth.shape[2] != 2 or not np.isfinite(depth).all():
        raise ContractError("CALVIN depth_tactile must be finite H-by-W-by-2")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(frame["rgb_static"])
    axes[0, 0].set_title("static RGB")
    axes[1, 0].imshow(frame["rgb_gripper"])
    axes[1, 0].set_title("wrist RGB")
    for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
        axes[0, sensor_index + 1].imshow(tactile_rgb[sensor_index])
        axes[0, sensor_index + 1].set_title(f"{sensor_name} RGB")
        sensor_depth = depth[..., sensor_index]
        absolute_max = float(np.abs(sensor_depth).max())
        color_limit = max(absolute_max, 1e-6)
        heatmap = axes[1, sensor_index + 1].imshow(
            sensor_depth,
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axes[1, sensor_index + 1].set_title(
            f"{sensor_name} signed deformation\nabs-max={absolute_max:.8f}"
        )
        figure.colorbar(heatmap, ax=axes[1, sensor_index + 1], fraction=0.046, pad=0.04)
    for axis in axes.flat:
        axis.axis("off")
    title = f"step={step} | task={task_key}\n{textwrap.fill(description, width=100)}"
    figure.suptitle(title, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def render_review(
    split_root: Path,
    *,
    annotation_path: Path,
    steps: tuple[int, ...],
    output_dir: Path,
) -> dict[str, object]:
    if not steps or len(set(steps)) != len(steps) or any(step < 0 for step in steps):
        raise ContractError("tactile review steps must be unique nonnegative indices")
    labels = _task_labels(annotation_path, steps)
    records: list[dict[str, object]] = []
    for step in steps:
        frame = _load_frame(split_root, step)
        task_key, description = labels[step]
        path = output_dir / f"step_{step:07d}_{task_key}.png"
        _render_step(
            frame=frame,
            step=step,
            task_key=task_key,
            description=description,
            output=path,
        )
        depth = np.asarray(frame["depth_tactile"], dtype=np.float32)
        records.append(
            {
                "step": step,
                "task_key": task_key,
                "description": description,
                "sensor_absolute_max": [
                    float(np.abs(depth[..., sensor]).max()) for sensor in range(2)
                ],
                "panel": str(path.resolve()),
            }
        )
    report = {
        "schema": "picf-next.calvin-tactile-visual-review/v1",
        "split_root": str(split_root.resolve()),
        "language_annotations": str(annotation_path.resolve()),
        "records": records,
    }
    manifest_path = output_dir / "review_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--step", required=True, action="append", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = render_review(
        args.split_root.resolve(),
        annotation_path=args.annotations.resolve(),
        steps=tuple(args.step),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

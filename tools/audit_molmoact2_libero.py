#!/usr/bin/env python3
"""Audit immutable MolmoAct2 LIBERO rows and render reviewed contact sheets.

This tool is deliberately outside the training data path. Task strings, frame
indices, and actions are rendered only for human auditing; they are never
converted into PICF runtime inputs or object targets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATASET_ID = "allenai/MolmoAct2-LIBERO-Dataset"
DATASET_REVISION = "fe3ead447f44c0ea950396360b304cc2fb6be8f8"
EXPECTED_HASHES = {
    "README.md": "9ff5165e508459728419888aa5b491fd8ce2f96fb38b06975e9d9c884e423dc1",
    "meta/info.json": "7d673ed015598a54ad0a6ef0de173064025cd8c5234d9fbe6eb5712f75926f36",
    "meta/stats.json": "59e99cd8051da2e0c87fba2adf3abdb60e9c93431c885bde0818843aba3e2d65",
    "meta/tasks.parquet": "6fcb2ffd27cc0ffed657d2afe675a1561049bbbec3c0a0212a948b11b2a13917",
    "meta/episodes/chunk-000/file-000.parquet": (
        "646df66ee3a469a82545563bd7c5d784e971b5cb974b62af022fffd17c29c34c"
    ),
    "data/chunk-000/file-000.parquet": (
        "a3503407cafca8304e55760652f8b29e8e1f6cb53375f856bcc5758054bbba67"
    ),
}
CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _decode_image(payload: dict[str, Any]) -> Image.Image:
    raw = payload.get("bytes")
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("embedded image bytes are missing")
    with Image.open(io.BytesIO(raw)) as image:
        decoded = image.convert("RGB")
    if decoded.size != (256, 256):
        raise ValueError(f"unexpected image size {decoded.size}")
    return decoded


def _format_vector(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _phase_indices(start: int, end: int) -> tuple[tuple[str, int], ...]:
    if not start < end:
        raise ValueError(f"episode must contain at least two rows: [{start}, {end})")
    return (
        ("start", start),
        ("middle", start + (end - start - 1) // 2),
        ("end", end - 1),
    )


def _render_panel(
    *,
    episode_index: int,
    task: str,
    rows: dict[int, dict[str, Any]],
    phases: tuple[tuple[str, int], ...],
) -> Image.Image:
    tile = 256
    header_height = 174
    label_height = 68
    canvas = Image.new(
        "RGB",
        (3 * tile, header_height + 2 * (tile + label_height)),
        color=(10, 10, 10),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    body_font = _load_font(13)
    small_font = _load_font(11)
    draw.text((10, 8), f"MolmoAct2 LIBERO public audit | episode {episode_index}", font=title_font)
    wrapped_task = textwrap.wrap(f"task: {task}", width=91)
    for line_index, line in enumerate(wrapped_task[:3]):
        draw.text((10, 36 + 18 * line_index), line, font=body_font, fill=(245, 245, 245))
    draw.text(
        (10, 98),
        "Raw embedded frames: no crop, no rotation, no task-conditioned selection",
        font=small_font,
        fill=(190, 210, 255),
    )
    draw.text(
        (10, 118),
        f"dataset={DATASET_ID}  revision={DATASET_REVISION[:12]}",
        font=small_font,
        fill=(190, 190, 190),
    )
    draw.text(
        (10, 138),
        "Action convention (official mixture): delta end-effector pose, horizon=10, execute=10",
        font=small_font,
        fill=(190, 190, 190),
    )

    camera_labels = ("external camera", "wrist camera")
    for column, (phase, absolute_index) in enumerate(phases):
        row = rows[absolute_index]
        frame_index = int(row["frame_index"])
        action = [float(value) for value in row["action"]]
        for camera_row, (camera_key, camera_label) in enumerate(
            zip(CAMERA_KEYS, camera_labels, strict=True)
        ):
            x = column * tile
            y = header_height + camera_row * (tile + label_height)
            canvas.paste(_decode_image(row[camera_key]), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(230, 230, 230))
            label_y = y + tile
            draw.rectangle((x, label_y, x + tile, label_y + label_height), fill=(18, 18, 18))
            draw.text(
                (x + 5, label_y + 3),
                f"{phase} | {camera_label} | frame={frame_index}",
                font=small_font,
                fill=(255, 255, 255),
            )
            draw.text(
                (x + 5, label_y + 22),
                "dxyz=" + _format_vector(action[:3], precision=2),
                font=small_font,
                fill=(205, 225, 255),
            )
            draw.text(
                (x + 5, label_y + 40),
                "drot+grip=" + _format_vector(action[3:], precision=2),
                font=small_font,
                fill=(205, 225, 255),
            )
    return canvas


def _pearson_xyz_action_state_delta(rows: list[dict[str, Any]]) -> list[float | None]:
    actions = np.asarray([row["action"][:3] for row in rows[:-1]], dtype=np.float64)
    states = np.asarray([row["observation.state"][:3] for row in rows], dtype=np.float64)
    deltas = states[1:] - states[:-1]
    correlations: list[float | None] = []
    for dimension in range(3):
        if np.std(actions[:, dimension]) <= 1e-12 or np.std(deltas[:, dimension]) <= 1e-12:
            correlations.append(None)
            continue
        correlations.append(float(np.corrcoef(actions[:, dimension], deltas[:, dimension])[0, 1]))
    return correlations


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode", action="append", type=int)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes_to_audit = tuple(args.episode or (0, 1, 2))

    hashes: dict[str, str] = {}
    for relative_path, expected_hash in EXPECTED_HASHES.items():
        actual_hash = _sha256(root / relative_path)
        _require(actual_hash == expected_hash, f"hash mismatch for {relative_path}")
        hashes[relative_path] = actual_hash

    info = json.loads((root / "meta/info.json").read_text())
    _require(info["fps"] == 10, "unexpected dataset FPS")
    _require(info["total_episodes"] == 1693, "unexpected episode count")
    _require(info["total_frames"] == 273465, "unexpected frame count")
    _require(info["total_tasks"] == 40, "unexpected task count")
    _require(info.get("video_path") is None, "audit expects images embedded in parquet")

    tasks_table = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks_table}
    _require(len(task_by_index) == 40, "task indices are not unique and complete")

    episode_table = pq.read_table(
        root / "meta/episodes/chunk-000/file-000.parquet",
        columns=[
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "tasks",
            "length",
        ],
    ).to_pylist()
    episode_by_index = {int(row["episode_index"]): row for row in episode_table}

    parquet_cache: dict[Path, Any] = {}
    records: list[dict[str, Any]] = []
    for episode_index in episodes_to_audit:
        metadata = episode_by_index.get(episode_index)
        _require(metadata is not None, f"episode {episode_index} is absent from metadata")
        chunk_index = int(metadata["data/chunk_index"])
        file_index = int(metadata["data/file_index"])
        data_path = root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        _require(data_path.is_file(), f"missing local data shard {data_path}")
        if data_path not in parquet_cache:
            parquet_cache[data_path] = pq.read_table(data_path).to_pylist()
        shard_rows = parquet_cache[data_path]
        start = int(metadata["dataset_from_index"])
        end = int(metadata["dataset_to_index"])
        expected_length = int(metadata["length"])
        episode_rows = [row for row in shard_rows if int(row["episode_index"]) == episode_index]
        _require(len(episode_rows) == expected_length == end - start, "episode length mismatch")
        episode_rows.sort(key=lambda row: int(row["index"]))
        indices = [int(row["index"]) for row in episode_rows]
        frames = [int(row["frame_index"]) for row in episode_rows]
        timestamps = np.asarray([float(row["timestamp"]) for row in episode_rows])
        task_indices = {int(row["task_index"]) for row in episode_rows}
        _require(indices == list(range(start, end)), "global frame indices are not contiguous")
        _require(frames == list(range(expected_length)), "episode frame indices are not contiguous")
        _require(
            np.allclose(timestamps, np.arange(expected_length) / 10.0, atol=2e-6),
            "timestamps do not match 10 Hz",
        )
        _require(len(task_indices) == 1, "episode contains multiple task indices")
        task_index = next(iter(task_indices))
        task = task_by_index[task_index]
        _require(metadata["tasks"] == [task], "episode task text disagrees with task table")
        for row in episode_rows:
            state = np.asarray(row["observation.state"], dtype=np.float64)
            action = np.asarray(row["action"], dtype=np.float64)
            _require(state.shape == (8,) and np.isfinite(state).all(), "invalid state")
            _require(action.shape == (7,) and np.isfinite(action).all(), "invalid action")

        phases = _phase_indices(start, end)
        rows_by_index = {int(row["index"]): row for row in episode_rows}
        panel = _render_panel(
            episode_index=episode_index,
            task=task,
            rows=rows_by_index,
            phases=phases,
        )
        panel_name = f"episode_{episode_index:04d}.png"
        panel.save(output / panel_name)
        records.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "task": task,
                "length": expected_length,
                "global_index_range": [start, end],
                "phase_frames": [
                    {
                        "phase": phase,
                        "global_index": absolute_index,
                        "frame_index": int(rows_by_index[absolute_index]["frame_index"]),
                        "timestamp": float(rows_by_index[absolute_index]["timestamp"]),
                        "state": [
                            float(value)
                            for value in rows_by_index[absolute_index]["observation.state"]
                        ],
                        "action": [
                            float(value) for value in rows_by_index[absolute_index]["action"]
                        ],
                    }
                    for phase, absolute_index in phases
                ],
                "xyz_action_next_state_delta_correlation": (
                    _pearson_xyz_action_state_delta(episode_rows)
                ),
                "panel": panel_name,
            }
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "format": "LeRobot v3; RGB images embedded in parquet",
        "official_training_contract": {
            "setup_type": "single franka robotic arm in libero",
            "control_mode": "delta end-effector pose",
            "action_horizon": 10,
            "n_action_steps": 10,
            "camera_keys": list(CAMERA_KEYS),
            "state_key": "observation.state",
            "state_semantics_from_official_processor": (
                "eef_position_xyz(3) + eef_axis_angle_xyz(3) + gripper_qpos(2)"
            ),
            "action_key": "action",
        },
        "metadata_caveat": (
            "meta/info.json names observation.state as xyz + quaternion + one gripper, "
            "but the official LiberoProcessorStep and sample values establish xyz + "
            "axis-angle + two gripper positions. Host adapters must use the processor "
            "semantics rather than the misleading feature names."
        ),
        "scope_boundary": (
            "Human/data-interface audit only. Task text and actions shown in panels are not "
            "PICF runtime inputs or object-target selectors."
        ),
        "hashes": hashes,
        "episodes": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "episodes": len(records)}, indent=2))


if __name__ == "__main__":
    main()

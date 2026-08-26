#!/usr/bin/env python3
"""Audit every immutable MolmoAct2 LIBERO shard and render all-task evidence.

The audit has two independent layers:

1. byte integrity against the exact Hugging Face revision tree; and
2. executable row semantics against the typed PICF-Next data boundary.

Images are not decoded exhaustively because byte-level hashes already cover
every payload.  Instead, start/middle/end frames from one deterministic episode
per task and both cameras are decoded into reviewable panels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from picf_next.data.robot_record import (
    MOLMOACT2_LIBERO_CAMERA_KEYS,
    MOLMOACT2_LIBERO_DATASET_ID,
    MOLMOACT2_LIBERO_REVISION,
    decode_molmoact2_libero_row,
    validate_molmoact2_libero_metadata,
)

EXPECTED_FILE_COUNT = 385
EXPECTED_TOTAL_BYTES = 34_935_776_578
EXPECTED_DATA_SHARDS = 379
EXPECTED_EPISODES = 1693
EXPECTED_FRAMES = 273465
EXPECTED_TASKS = 40
EXPECTED_LOCATOR_MISMATCHES = 373
EXPECTED_EPISODE_TASK_STAT_MISMATCHES = 1314
EXPECTED_VARIABLE_LIST_SHARDS = {"file-146.parquet", "file-309.parquet"}
NUMERIC_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def load_revision_tree(root: Path) -> dict[str, dict[str, Any]]:
    path = root / ".cache" / "huggingface" / "trees" / f"{MOLMOACT2_LIBERO_REVISION}.json"
    _require(path.is_file(), f"missing immutable Hugging Face tree: {path}")
    document = json.loads(path.read_text())
    _require(document.get("format_version") == 1, "unknown Hugging Face tree format")
    files = document.get("files")
    _require(isinstance(files, dict), "revision tree has no file map")
    return files


def audit_revision_files(
    root: Path,
    files: Mapping[str, Mapping[str, Any]],
    *,
    verify_hashes: bool,
) -> tuple[list[dict[str, Any]], str]:
    """Validate exact paths/sizes and optionally every immutable content hash."""

    _require(len(files) == EXPECTED_FILE_COUNT, "revision file count changed")
    _require(
        sum(int(item["size"]) for item in files.values()) == EXPECTED_TOTAL_BYTES,
        "revision byte count changed",
    )
    _require(
        sum(relative.startswith("data/") for relative in files) == EXPECTED_DATA_SHARDS,
        "data shard count changed",
    )
    records: list[dict[str, Any]] = []
    for relative, expected in sorted(files.items()):
        path = root / relative
        _require(path.is_file(), f"missing revision file {relative}")
        size = path.stat().st_size
        _require(size == int(expected["size"]), f"size mismatch for {relative}")
        algorithm = "sha256" if "lfs_sha256" in expected else "git-blob-sha1"
        expected_hash = str(expected.get("lfs_sha256") or expected["blob_id"])
        actual_hash = None
        if verify_hashes:
            actual_hash = _sha256(path) if algorithm == "sha256" else _git_blob_sha1(path)
            _require(actual_hash == expected_hash, f"hash mismatch for {relative}")
        records.append(
            {
                "path": relative,
                "size": size,
                "hash_algorithm": algorithm,
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return records, hashlib.sha256(canonical).hexdigest()


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _decode_image(payload: Mapping[str, Any]) -> Image.Image:
    raw = payload.get("bytes")
    _require(isinstance(raw, bytes) and bool(raw), "embedded image bytes are missing")
    with Image.open(io.BytesIO(raw)) as image:
        decoded = image.convert("RGB")
    _require(decoded.size == (256, 256), f"unexpected image size {decoded.size}")
    return decoded


def _phase_global_indices(episode: Mapping[str, Any]) -> tuple[int, int, int]:
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    _require(end > start, "empty episode")
    return start, start + (end - start - 1) // 2, end - 1


def _validate_numeric_shard(
    table: Any,
    episodes: Mapping[int, Mapping[str, Any]] | list[Mapping[str, Any]],
    *,
    fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Validate one shard by row identity, independent of its claimed locator."""

    episode_by_index = (
        dict(episodes)
        if isinstance(episodes, Mapping)
        else {int(item["episode_index"]): item for item in episodes}
    )
    _require(table.num_rows > 0, "data shard is empty")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    _require(state.shape == (table.num_rows, 8), "state matrix has the wrong shape")
    _require(action.shape == (table.num_rows, 7), "action matrix has the wrong shape")
    _require(np.isfinite(state).all(), "state matrix contains NaN or infinity")
    _require(np.isfinite(action).all(), "action matrix contains NaN or infinity")

    episode_index = table["episode_index"].to_numpy(zero_copy_only=False)
    frame_index = table["frame_index"].to_numpy(zero_copy_only=False)
    task_index = table["task_index"].to_numpy(zero_copy_only=False)
    global_index = table["index"].to_numpy(zero_copy_only=False)
    timestamp = table["timestamp"].to_numpy(zero_copy_only=False)
    change = np.concatenate(([True], episode_index[1:] != episode_index[:-1]))
    starts = np.flatnonzero(change)
    ends = np.concatenate((starts[1:], [table.num_rows]))
    observed_episodes = [int(episode_index[start]) for start in starts]
    _require(len(set(observed_episodes)) == len(observed_episodes), "episode rows are split")

    for observed_episode, start, end in zip(observed_episodes, starts, ends, strict=True):
        _require(observed_episode in episode_by_index, "row episode is absent from metadata")
        episode = episode_by_index[observed_episode]
        length = int(episode["length"])
        _require(end - start == length, "episode row count differs from metadata")
        expected_frame = np.arange(length, dtype=np.int64)
        expected_global = np.arange(
            int(episode["dataset_from_index"]),
            int(episode["dataset_to_index"]),
            dtype=np.int64,
        )
        expected_task = int(episode["task_index"])
        _require(
            np.all(episode_index[start:end] == observed_episode),
            "episode order mismatch",
        )
        _require(np.array_equal(frame_index[start:end], expected_frame), "frame order mismatch")
        _require(
            np.all(task_index[start:end] == expected_task),
            "task order mismatch",
        )
        _require(
            np.array_equal(global_index[start:end], expected_global),
            "global index mismatch",
        )
        _require(
            np.allclose(
                timestamp[start:end],
                expected_frame / float(fps),
                atol=2e-6,
                rtol=0.0,
            ),
            "timestamp differs from frame/fps",
        )
    return state, action, task_index, observed_episodes


def _schema_storage_variant(schema: Any, canonical_schema: Any) -> str:
    """Classify the two known Arrow storage encodings without relaxing semantics."""

    import pyarrow as pa

    _require(schema.names == canonical_schema.names, "parquet column names drifted")
    for name in schema.names:
        if name in {"observation.state", "action"}:
            continue
        _require(
            schema.field(name).equals(canonical_schema.field(name), check_metadata=False),
            f"parquet field drifted: {name}",
        )
    state_type = schema.field("observation.state").type
    action_type = schema.field("action").type
    state_fixed = (
        pa.types.is_fixed_size_list(state_type)
        and state_type.list_size == 8
        and state_type.value_type == pa.float32()
    )
    action_fixed = (
        pa.types.is_fixed_size_list(action_type)
        and action_type.list_size == 7
        and action_type.value_type == pa.float32()
    )
    if state_fixed and action_fixed:
        return "fixed_size_list"
    state_variable = pa.types.is_list(state_type) and state_type.value_type == pa.float32()
    action_variable = pa.types.is_list(action_type) and action_type.value_type == pa.float32()
    if state_variable and action_variable:
        return "variable_list"
    raise ValueError(
        f"unsupported state/action parquet storage types: state={state_type}, action={action_type}"
    )


def build_episode_locator_overlay(
    episodes: list[Mapping[str, Any]],
    actual_file_by_episode: Mapping[int, int],
) -> list[dict[str, int | bool]]:
    """Record the immutable claimed locator and audited physical locator."""

    expected = {int(item["episode_index"]) for item in episodes}
    _require(set(actual_file_by_episode) == expected, "actual episode locator is incomplete")
    records = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        claimed_chunk = int(episode["data/chunk_index"])
        claimed_file = int(episode["data/file_index"])
        actual_file = int(actual_file_by_episode[episode_index])
        records.append(
            {
                "episode_index": episode_index,
                "claimed_chunk_index": claimed_chunk,
                "claimed_file_index": claimed_file,
                "actual_chunk_index": 0,
                "actual_file_index": actual_file,
                "mismatch": claimed_chunk != 0 or claimed_file != actual_file,
            }
        )
    return records


def _render_task_panel(
    task_record: Mapping[str, Any], rows: Mapping[int, Mapping[str, Any]]
) -> Image.Image:
    tile = 256
    header = 96
    label = 35
    canvas = Image.new("RGB", (3 * tile, header + 2 * (tile + label)), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(17)
    body_font = _load_font(12)
    task = str(task_record["task"])
    draw.text(
        (8, 7),
        f"task {task_record['task_index']:02d} | episode {task_record['episode_index']}",
        font=title_font,
        fill="white",
    )
    words = task.split()
    line = ""
    lines: list[str] = []
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 86 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for line_index, text in enumerate(lines[:3]):
        draw.text((8, 32 + 17 * line_index), text, font=body_font, fill=(235, 235, 235))

    phases = ("start", "middle", "end")
    cameras = ("external", "wrist")
    for column, global_index in enumerate(task_record["phase_global_indices"]):
        row = rows[int(global_index)]
        camera_pairs = zip(MOLMOACT2_LIBERO_CAMERA_KEYS, cameras, strict=True)
        for camera_row, (key, camera) in enumerate(camera_pairs):
            image = _decode_image(row[key])
            x = column * tile
            y = header + camera_row * (tile + label)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(230, 230, 230))
            draw.rectangle((x, y + tile, x + tile, y + tile + label), fill=(20, 20, 20))
            draw.text(
                (x + 5, y + tile + 8),
                f"{phases[column]} | {camera} | frame {row['frame_index']}",
                font=body_font,
                fill="white",
            )
    return canvas


def _render_overview(panels: Iterable[tuple[Mapping[str, Any], Image.Image]]) -> Image.Image:
    entries = list(panels)
    columns = 4
    thumb_width = 480
    thumb_height = 551
    rows = math.ceil(len(entries) / columns)
    canvas = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (5, 5, 5))
    for index, (_, panel) in enumerate(entries):
        thumbnail = panel.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(
            thumbnail,
            ((index % columns) * thumb_width, (index // columns) * thumb_height),
        )
    return canvas


def _task_representatives(
    episodes: list[Mapping[str, Any]],
    task_by_index: Mapping[int, str],
    actual_file_by_episode: Mapping[int, int],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[int(episode["task_index"])].append(episode)
    _require(set(grouped) == set(task_by_index), "some tasks have no episodes")
    records: list[dict[str, Any]] = []
    for task_index in sorted(grouped):
        candidates = sorted(grouped[task_index], key=lambda item: int(item["episode_index"]))
        episode = candidates[(len(candidates) - 1) // 2]
        records.append(
            {
                "task_index": task_index,
                "task": task_by_index[task_index],
                "episode_index": int(episode["episode_index"]),
                "claimed_file_index": int(episode["data/file_index"]),
                "file_index": int(actual_file_by_episode[int(episode["episode_index"])]),
                "length": int(episode["length"]),
                "phase_global_indices": list(_phase_global_indices(episode)),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="development-only: a gate report requires complete hashes",
    )
    args = parser.parse_args()

    import pyarrow.parquet as pq

    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta" / "info.json").read_text())
    validate_molmoact2_libero_metadata(info)

    tree = load_revision_tree(root)
    file_records, file_manifest_sha256 = audit_revision_files(
        root, tree, verify_hashes=not args.skip_hashes
    )

    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    _require(len(task_by_index) == EXPECTED_TASKS, "task table is incomplete or duplicated")
    task_index_by_text = {text: task_index for task_index, text in task_by_index.items()}
    _require(len(task_index_by_text) == EXPECTED_TASKS, "task texts are not unique")
    episode_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pq.read_table(episode_path).to_pylist()
    episodes.sort(key=lambda item: int(item["episode_index"]))
    _require(len(episodes) == EXPECTED_EPISODES, "episode table count changed")
    _require(
        [int(item["episode_index"]) for item in episodes] == list(range(EXPECTED_EPISODES)),
        "episode indices are not contiguous",
    )
    cursor = 0
    episode_task_stat_mismatches = 0
    for episode in episodes:
        task_texts = episode["tasks"]
        _require(
            isinstance(task_texts, list) and len(task_texts) == 1,
            "episode must contain exactly one task",
        )
        task_text = str(task_texts[0])
        _require(task_text in task_index_by_text, "episode task is absent from task table")
        episode["task_index"] = task_index_by_text[task_text]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        length = int(episode["length"])
        task_index = int(episode["task_index"])
        _require(start == cursor and end - start == length, "episode global ranges have a gap")
        _require(task_text == task_by_index[task_index], "episode task text mismatch")
        if episode.get("stats/task_index/min") != [task_index] or episode.get(
            "stats/task_index/max"
        ) != [task_index]:
            episode_task_stat_mismatches += 1
        _require(int(episode["data/chunk_index"]) == 0, "unexpected data chunk")
        cursor = end
    _require(cursor == EXPECTED_FRAMES, "episode ranges do not cover every frame")
    _require(
        episode_task_stat_mismatches == EXPECTED_EPISODE_TASK_STAT_MISMATCHES,
        "episode-level task summary mismatch count changed",
    )

    episode_by_index = {int(item["episode_index"]): item for item in episodes}

    state_min = np.full(8, np.inf, dtype=np.float64)
    state_max = np.full(8, -np.inf, dtype=np.float64)
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)
    task_frame_counts = np.zeros(EXPECTED_TASKS, dtype=np.int64)
    canonical_schema = None
    schema_variants: dict[str, list[str]] = defaultdict(list)
    row_count = 0
    actual_file_by_episode: dict[int, int] = {}
    for file_index in range(EXPECTED_DATA_SHARDS):
        path = root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        parquet = pq.ParquetFile(path)
        current_schema = parquet.schema_arrow
        if canonical_schema is None:
            canonical_schema = current_schema
        variant = _schema_storage_variant(current_schema, canonical_schema)
        schema_variants[variant].append(path.name)
        table = parquet.read(columns=list(NUMERIC_COLUMNS))
        state, action, task_index, observed_episodes = _validate_numeric_shard(
            table, episode_by_index, fps=int(info["fps"])
        )
        for episode_index in observed_episodes:
            _require(
                episode_index not in actual_file_by_episode,
                "episode occurs in more than one data shard",
            )
            actual_file_by_episode[episode_index] = file_index
        state_min = np.minimum(state_min, state.min(axis=0))
        state_max = np.maximum(state_max, state.max(axis=0))
        action_min = np.minimum(action_min, action.min(axis=0))
        action_max = np.maximum(action_max, action.max(axis=0))
        task_frame_counts += np.bincount(task_index, minlength=EXPECTED_TASKS)
        row_count += table.num_rows
    _require(row_count == EXPECTED_FRAMES, "numeric audit did not cover every frame")
    _require(
        set(schema_variants["variable_list"]) == EXPECTED_VARIABLE_LIST_SHARDS,
        "known variable-list shard set changed",
    )
    _require(
        len(schema_variants["fixed_size_list"])
        == EXPECTED_DATA_SHARDS - len(EXPECTED_VARIABLE_LIST_SHARDS),
        "fixed-size-list shard count changed",
    )
    locator_overlay = build_episode_locator_overlay(episodes, actual_file_by_episode)
    locator_mismatches = [item for item in locator_overlay if item["mismatch"]]
    _require(
        len(locator_mismatches) == EXPECTED_LOCATOR_MISMATCHES,
        "upstream episode locator mismatch count changed",
    )
    locator_overlay_path = output / "episode_file_locator_overlay.json"
    locator_overlay_path.write_text(json.dumps(locator_overlay, indent=2) + "\n")
    locator_overlay_sha256 = hashlib.sha256(
        json.dumps(locator_overlay, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    representative_records = _task_representatives(
        episodes,
        task_by_index,
        actual_file_by_episode,
    )
    representatives_by_file: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in representative_records:
        representatives_by_file[int(record["file_index"])].append(record)
    rendered: list[tuple[Mapping[str, Any], Image.Image]] = []
    for file_index, records in sorted(representatives_by_file.items()):
        path = root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        needed = {int(index) for record in records for index in record["phase_global_indices"]}
        full_rows = pq.read_table(path).to_pylist()
        selected = {int(row["index"]): row for row in full_rows if int(row["index"]) in needed}
        _require(set(selected) == needed, f"visual rows missing from {path.name}")
        for record in records:
            rows = {int(index): selected[int(index)] for index in record["phase_global_indices"]}
            episode_length = int(record["length"])
            task = str(record["task"])
            for row in rows.values():
                typed = decode_molmoact2_libero_row(row, task=task, episode_length=episode_length)
                for camera in typed.cameras:
                    _decode_image({"bytes": camera.encoded_bytes, "path": camera.source_path})
            panel = _render_task_panel(record, rows)
            panel_name = (
                f"task_{record['task_index']:02d}_episode_{record['episode_index']:04d}.png"
            )
            panel.save(output / panel_name)
            record["panel"] = panel_name
            rendered.append((record, panel))

    rendered.sort(key=lambda item: int(item[0]["task_index"]))
    overview_name = "all_40_tasks_start_middle_end_both_cameras.png"
    _render_overview(rendered).save(output / overview_name)

    episode_counts = np.bincount(
        np.asarray([int(item["task_index"]) for item in episodes]), minlength=EXPECTED_TASKS
    )
    task_summary = [
        {
            "task_index": task_index,
            "task": task_by_index[task_index],
            "episodes": int(episode_counts[task_index]),
            "frames": int(task_frame_counts[task_index]),
        }
        for task_index in range(EXPECTED_TASKS)
    ]
    report = {
        "status": (
            "PASS_WITH_UPSTREAM_LOCATOR_BUG_MITIGATED"
            if not args.skip_hashes
            else "DEVELOPMENT_ONLY_HASHES_SKIPPED"
        ),
        "dataset_id": MOLMOACT2_LIBERO_DATASET_ID,
        "dataset_revision": MOLMOACT2_LIBERO_REVISION,
        "tree": {
            "files": EXPECTED_FILE_COUNT,
            "data_shards": EXPECTED_DATA_SHARDS,
            "bytes": EXPECTED_TOTAL_BYTES,
            "hashes_verified": not args.skip_hashes,
            "canonical_file_manifest_sha256": file_manifest_sha256,
            "file_records": file_records,
        },
        "rows": {
            "episodes": len(episodes),
            "frames": row_count,
            "tasks": len(task_by_index),
            "fps": int(info["fps"]),
            "episode_task_stat_mismatches": episode_task_stat_mismatches,
            "state_shape": [8],
            "action_shape": [7],
            "state_min": state_min.tolist(),
            "state_max": state_max.tolist(),
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
            "canonical_schema": str(canonical_schema),
            "schema_storage_variants": dict(schema_variants),
        },
        "typed_contract": {
            "version": "molmoact2-libero-transition/v1",
            "state": "eef XYZ(3) + axis-angle(3) + gripper qpos(2)",
            "action": "LIBERO normalized delta end-effector command(6) + binary gripper(1)",
            "delta_t_s": 0.1,
            "runtime_target_fields": [],
        },
        "upstream_episode_locator": {
            "status": "BUG_CONFIRMED",
            "mismatched_episodes": len(locator_mismatches),
            "total_episodes": len(episodes),
            "overlay": locator_overlay_path.name,
            "overlay_sha256": locator_overlay_sha256,
            "raw_files_modified": False,
            "training_mitigation": (
                "pinned LingBot patch delegates LeRobot v3 loading to "
                "load_nested_dataset, which scans every shard once and filters by episode_index"
            ),
        },
        "task_summary": task_summary,
        "visual_sample": {
            "rule": "lower median episode per task; start/middle/end; both cameras",
            "decoded_images": EXPECTED_TASKS * 3 * 2,
            "representatives": representative_records,
            "overview": overview_name,
            "review_status": "PENDING_REVIEW",
        },
    }
    report_path = output / "full_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "frames": row_count,
                "decoded_images": EXPECTED_TASKS * 3 * 2,
                "overview": str(output / overview_name),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

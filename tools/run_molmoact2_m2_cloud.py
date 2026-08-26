#!/usr/bin/env python3
"""Run the fail-closed MolmoAct2 M2 current-frame representation gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from picf_next.hosts.molmoact2 import (
        MolmoAct2ImagePatchSpan,
        MolmoAct2VisionPatchLayout,
    )
    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.data.calvin import CALVIN_HOST_IMAGE_KEYS  # noqa: E402
from picf_next.data.calvin_loss_targets import (  # noqa: E402
    CalvinSourceFrameLossTargetRequest,
    CalvinStatefulLossTargetRequest,
)
from picf_next.geometry import PhysicalGeometryContract  # noqa: E402
from picf_next.models.evidence import ModalityTokenSpan, NativeTokenBank  # noqa: E402
from picf_next.models.set_loss import ObjectSetCriterion, ObjectSetTarget  # noqa: E402
from picf_next.training.molmoact2_m2 import (  # noqa: E402
    M2_GATE,
    MolmoAct2M2Recipe,
    load_molmoact2_m2_recipe,
    m2_recipe_report,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_M2_MACHINE_REPORTS = (
    "launch_manifest.json",
    "environment.json",
    "split_manifest.json",
    "feature_cache/manifest.json",
    "task_intervention.json",
    "batch_plan.json",
    "training_report.json",
    "evaluation_report.json",
    "visual_artifacts.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_representation.json",
    )
    parser.add_argument("--m1-run", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument(
        "--sidecar-artifact-root",
        type=Path,
        default=Path("/mnt/picf-next/artifacts/calvin_loss_sidecars"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/picf-next/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _emit_progress(stage: str, **payload: Any) -> None:
    print(
        json.dumps(
            {
                "event": "picf-next.molmoact2-m2-progress.v1",
                "stage": stage,
                **payload,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("ascii"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _clean_git_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("M2 source revision is not one full Git commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_ROOT,
        text=True,
    )
    if dirty:
        raise RuntimeError("M2 requires one clean committed source tree")
    return revision


def _is_under_mnt(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == Path("/mnt") or Path("/mnt") in resolved.parents


def _run_id(value: str | None) -> str:
    resolved = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _RUN_ID.fullmatch(resolved):
        raise ValueError(f"invalid M2 run id: {resolved!r}")
    return resolved


def validate_prior_m1(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    decision_path = run_dir / "gate_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("M2 requires an immutable accepted M1 decision")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "picf-next.molmoact2-m1-gate-decision.v1"
        or decision.get("status") != "PASS"
        or decision.get("gate") != "M1_typed_full_manifest"
        or decision.get("later_gates_authorized") != [M2_GATE]
    ):
        raise ValueError("prior M1 decision does not authorize exactly M2")
    hashes = decision.get("required_report_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("prior M1 decision omitted immutable report hashes")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("prior M1 report hash mapping is malformed")
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"prior M1 report hash changed: {relative}")
    launch = json.loads((run_dir / "launch_manifest.json").read_text())
    prior_m0 = launch.get("prior_m0")
    if not isinstance(prior_m0, dict):
        raise ValueError("prior M1 launch omitted its accepted M0 binding")
    m0_run = Path(str(prior_m0.get("run_dir", ""))).resolve()
    m0_raw = m0_run / "m0_raw_report.json"
    m0_decision = m0_run / "gate_decision.json"
    if (
        not m0_raw.is_file()
        or not m0_decision.is_file()
        or _sha256(m0_raw) != prior_m0.get("raw_report_sha256")
        or _sha256(m0_decision) != prior_m0.get("gate_decision_sha256")
    ):
        raise ValueError("prior M1 no longer resolves its exact accepted M0")
    return {
        "run_dir": str(run_dir),
        "gate_decision_sha256": _sha256(decision_path),
        "visual_review_sha256": _sha256(run_dir / "visual_review.json"),
        "m0_run_dir": str(m0_run),
        "m0_raw_report": json.loads(m0_raw.read_text()),
        "m0_raw_report_sha256": _sha256(m0_raw),
    }


def _validate_devices() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("M2 requires exactly two visible CUDA devices")
    resources = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        memory_gib = properties.total_memory / 2**30
        if "A100" not in name or memory_gib < 39.0:
            raise RuntimeError(
                f"M2 expected A100-40G at cuda:{index}, observed {name!r} with {memory_gib:.2f} GiB"
            )
        resources.append(
            {
                "device": f"cuda:{index}",
                "name": name,
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return resources


def materialize_persistent_sidecars(artifact_root: Path) -> dict[str, Any]:
    """Restore ignored loss-only shards as verified regular files."""

    artifact_root = artifact_root.expanduser().resolve()
    if not _is_under_mnt(artifact_root):
        raise RuntimeError("CALVIN sidecar artifacts must persist under /mnt")
    restored: list[dict[str, Any]] = []
    for name in ("calvin_physical_supervision_v2", "calvin_geometry_training_v4"):
        source = artifact_root / name
        destination = _ROOT / "data" / name
        tracked_manifest = destination / "manifest.json"
        persistent_manifest = source / "manifest.json"
        if not tracked_manifest.is_file() or not persistent_manifest.is_file():
            raise FileNotFoundError(f"CALVIN persistent sidecar manifest is absent: {name}")
        if tracked_manifest.read_bytes() != persistent_manifest.read_bytes():
            raise ValueError(f"CALVIN persistent sidecar manifest differs from Git: {name}")
        manifest = json.loads(tracked_manifest.read_text())
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"CALVIN sidecar manifest contains no shards: {name}")
        expected_names = {"manifest.json"}
        for shard in shards:
            relative = shard.get("path")
            expected = shard.get("sha256")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(expected, str)
            ):
                raise ValueError(f"CALVIN sidecar shard metadata is unsafe: {name}")
            expected_names.add(relative)
            source_path = source / relative
            destination_path = destination / relative
            if not source_path.is_file() or _sha256(source_path) != expected:
                raise ValueError(f"CALVIN persistent sidecar shard changed: {source_path}")
            materialization = "existing_verified_regular_file"
            if destination_path.is_symlink():
                if destination_path.resolve() != source_path.resolve():
                    raise ValueError(
                        f"CALVIN materialized sidecar symlink has an unexpected target: "
                        f"{destination_path}"
                    )
                destination_path.unlink()
            if destination_path.exists():
                if not destination_path.is_file() or _sha256(destination_path) != expected:
                    raise ValueError(
                        f"CALVIN materialized sidecar differs from persistent source: "
                        f"{destination_path}"
                    )
            else:
                temporary = destination_path.with_name(
                    f".{destination_path.name}.tmp-{os.getpid()}"
                )
                if temporary.exists():
                    raise FileExistsError(temporary)
                try:
                    with source_path.open("rb") as source_handle, temporary.open("xb") as output:
                        for block in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if _sha256(temporary) != expected:
                        raise ValueError(
                            f"CALVIN copied sidecar failed post-copy hash verification: {relative}"
                        )
                    os.replace(temporary, destination_path)
                    _fsync_directory(destination_path.parent)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                materialization = "copied_from_persistent_storage"
            restored.append(
                {
                    "sidecar": name,
                    "path": relative,
                    "sha256": expected,
                    "persistent_source": str(source_path),
                    "materialized_path": str(destination_path),
                    "materialization": materialization,
                }
            )
        extras = sorted(
            path.name
            for path in source.iterdir()
            if path.is_file() and path.name not in expected_names and path.suffix == ".npz"
        )
        if extras:
            raise ValueError(f"CALVIN persistent sidecar contains undeclared shards: {extras}")
    return {
        "schema": "picf-next.calvin-persistent-sidecar-materialization.v1",
        "artifact_root": str(artifact_root),
        "restored": restored,
    }


def _validate_split_contract(assets: Any, recipe: MolmoAct2M2Recipe) -> dict[str, Any]:
    segments = {segment.index: segment for segment in assets.index.segments}
    declared = set(recipe.splits.learned_segments) | set(
        recipe.splits.excluded_overlap_control_segments
    )
    if set(segments) != declared:
        raise ValueError(
            f"M2 split must account for every CALVIN language segment: "
            f"dataset={sorted(segments)}, declared={sorted(declared)}"
        )
    learned = [segments[index] for index in recipe.splits.learned_segments]
    for left_index, left in enumerate(learned):
        for right in learned[left_index + 1 :]:
            if max(left.start, right.start) < min(left.end, right.end):
                raise ValueError(
                    f"M2 learned splits overlap in source frames: {left.index} and {right.index}"
                )
    overlap_controls: list[dict[str, Any]] = []
    for control_index in recipe.splits.excluded_overlap_control_segments:
        control = segments[control_index]
        overlaps = [
            candidate
            for candidate in learned
            if max(control.start, candidate.start) < min(control.end, candidate.end)
        ]
        if len(overlaps) != 1 or overlaps[0].task_key != control.task_key:
            raise ValueError("M2 excluded overlap controls do not map uniquely to one learned task")
        overlap_controls.append(
            {
                "control_segment": control.index,
                "learned_segment": overlaps[0].index,
                "task_key": control.task_key,
                "control_instruction": control.instruction,
                "learned_instruction": overlaps[0].instruction,
                "intersection_start_end_exclusive": [
                    max(control.start, overlaps[0].start),
                    min(control.end, overlaps[0].end),
                ],
            }
        )
    split_rows = []
    for index in recipe.splits.learned_segments:
        segment = segments[index]
        split_rows.append(
            {
                "split": recipe.splits.split_name(index),
                "segment_index": index,
                "task_key": segment.task_key,
                "instruction": segment.instruction,
                "start": segment.start,
                "end_exclusive": segment.end,
                "transition_count": segment.transition_count,
            }
        )
    counts = {
        name: sum(row["transition_count"] for row in split_rows if row["split"] == name)
        for name in ("train", "validation", "heldout")
    }
    return {
        "schema": "picf-next.molmoact2-m2-split.v1",
        "strategy": recipe.splits.strategy,
        "rows": split_rows,
        "transition_counts": counts,
        "overlap_controls": overlap_controls,
        "learned_source_ranges_disjoint": True,
    }


def _layout_row_payload(row: Sequence[MolmoAct2ImagePatchSpan]) -> list[dict[str, Any]]:
    return [
        {
            "image_key": span.image_key,
            "start": span.start,
            "stop": span.stop,
            "image_num_crops": span.image_num_crops,
            "patches_per_crop": span.patches_per_crop,
            "image_grid": list(span.image_grid),
            "image_token_pooling": [list(values) for values in span.image_token_pooling],
        }
        for span in row
    ]


def _layout_from_payload(
    payload: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    token_count: int,
) -> MolmoAct2VisionPatchLayout:
    from picf_next.hosts.molmoact2 import (
        MolmoAct2ImagePatchSpan,
        MolmoAct2VisionPatchLayout,
    )

    row = tuple(
        MolmoAct2ImagePatchSpan(
            image_key=str(value["image_key"]),
            start=int(value["start"]),
            stop=int(value["stop"]),
            image_num_crops=int(value["image_num_crops"]),
            patches_per_crop=int(value["patches_per_crop"]),
            image_grid=tuple(int(item) for item in value["image_grid"]),
            image_token_pooling=tuple(
                tuple(int(item) for item in support) for support in value["image_token_pooling"]
            ),
        )
        for value in payload
    )
    return MolmoAct2VisionPatchLayout(
        rows=tuple(row for _ in range(batch_size)),
        tokens_per_row=token_count,
        semantic_image_keys=True,
    )


def _move_inputs(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def _regular_cpu_copy(value: Any, *, dtype: Any | None = None) -> Any:
    """Copy one inference tensor into ordinary CPU storage for later training."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("M2 feature cache accepts only tensors")
    resolved_dtype = value.dtype if dtype is None else dtype
    copied = torch.empty(value.shape, dtype=resolved_dtype, device="cpu")
    copied.copy_(value, non_blocking=False)
    if copied.is_inference():
        raise RuntimeError("M2 feature cache retained PyTorch inference-tensor semantics")
    return copied.contiguous()


def _selected_samples(assets: Any, recipe: MolmoAct2M2Recipe) -> list[Any]:
    rows = []
    selected = set(recipe.splits.learned_segments)
    for manifest in assets.dataset.episode_manifest:
        if manifest.segment_index not in selected:
            continue
        rows.extend(assets.dataset.by_key(key) for key in manifest.sample_keys)
    rows.sort(key=lambda sample: (sample.record.task_index, sample.transition_index))
    return rows


def _extract_feature_cache(
    *,
    run_dir: Path,
    recipe: MolmoAct2M2Recipe,
    foundation: Any,
    assets: Any,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation
    from picf_next.hosts.molmoact2_calvin_processor import CalvinMolmoAct2ProcessorBridge
    from picf_next.hosts.molmoact2_training import (
        calvin_visible_object_target_request,
        molmoact2_host_observation_view,
    )
    from picf_next.training.molmoact2_calvin import build_molmoact2_policy_config

    cache_dir = run_dir / "feature_cache"
    cache_dir.mkdir()
    policy_config = build_molmoact2_policy_config(
        foundation,
        checkpoint_path=checkpoint_dir,
    )
    device = torch.device("cuda:0")
    policy = MolmoAct2Policy(policy_config).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    stats = __import__(
        "picf_next.data.calvin_normalization",
        fromlist=["official_molmoact2_dataset_stats"],
    ).official_molmoact2_dataset_stats(assets.normalization_payload)
    processor = CalvinMolmoAct2ProcessorBridge.from_official_config(
        policy.config,
        dataset_stats=stats,
    )

    samples = _selected_samples(assets, recipe)
    records: list[dict[str, Any]] = []
    canonical_layout: list[dict[str, Any]] | None = None
    pending_tokens = []
    pending_valid = []
    pending_records = []
    shard_rows = recipe.cache.shard_rows
    shard_index = 0
    shards: list[dict[str, Any]] = []
    extraction_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def flush() -> None:
        nonlocal shard_index
        if not pending_tokens:
            return
        path = cache_dir / f"features-{shard_index:05d}.pt"
        tokens = torch.cat(pending_tokens, dim=0).contiguous()
        valid = torch.cat(pending_valid, dim=0).contiguous()
        _write_torch_atomic(path, {"tokens": tokens, "valid": valid})
        for row_index, record in enumerate(pending_records):
            record["shard"] = path.name
            record["row"] = row_index
            records.append(record)
        shards.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "rows": len(pending_records),
                "bytes": path.stat().st_size,
            }
        )
        _emit_progress(
            "feature_cache_shard",
            shard=path.name,
            shard_rows=len(pending_records),
            completed_rows=len(records),
            total_rows=len(samples),
        )
        pending_tokens.clear()
        pending_valid.clear()
        pending_records.clear()
        shard_index += 1

    batch_size = recipe.cache.extraction_batch_size
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        evidence = tuple((sample.picf_evidence_frame,) for sample in batch)
        views = tuple(molmoact2_host_observation_view(sample.record) for sample in batch)
        observation_inputs = _move_inputs(
            processor.build_observation_inputs(evidence, views),
            device,
        )
        with torch.inference_mode():
            prepared = prepare_molmoact2_lerobot_observation(policy, observation_inputs)
        bank = prepared.vision_patch_bank
        layout = prepared.vision_patch_layout
        if bank is None or layout is None:
            raise RuntimeError("M2 observation produced no dense Molmo patch bank or layout")
        if (
            bank.modality != recipe.cache.modality
            or bank.tokens.shape[1:] != (recipe.cache.token_count, recipe.cache.token_dim)
            or bank.valid.shape != bank.tokens.shape[:2]
        ):
            raise RuntimeError("M2 native Molmo feature contract changed")
        layout_rows = [_layout_row_payload(row) for row in layout.rows]
        for row in layout_rows:
            if canonical_layout is None:
                canonical_layout = row
            elif row != canonical_layout:
                raise RuntimeError("M2 processor patch layout changed across CALVIN rows")
        cpu_tokens = _regular_cpu_copy(bank.tokens, dtype=torch.bfloat16)
        cpu_valid = _regular_cpu_copy(bank.valid)
        for batch_index, sample in enumerate(batch):
            request = calvin_visible_object_target_request(sample)
            pending_tokens.append(cpu_tokens[batch_index : batch_index + 1])
            pending_valid.append(cpu_valid[batch_index : batch_index + 1])
            pending_records.append(
                {
                    "sample_key": sample.sample_key,
                    "split": recipe.splits.split_name(sample.record.task_index),
                    "segment_index": sample.record.task_index,
                    "transition_index": sample.transition_index,
                    "global_index": sample.record.global_index,
                    "task_key": sample.host_sample.task_key,
                    "instruction": sample.record.task,
                    "source_sensor_sha256": [list(item) for item in request.source_sensor_sha256],
                }
            )
            if len(pending_records) == shard_rows:
                flush()
        del prepared, bank, observation_inputs
    flush()
    if canonical_layout is None:
        raise RuntimeError("M2 selected no features")
    if len(records) != len(samples) or len({row["sample_key"] for row in records}) != len(records):
        raise RuntimeError("M2 feature cache is not one-to-one with selected samples")

    task_intervention = _task_intervention_probe(
        policy=policy,
        processor=processor,
        assets=assets,
        recipe=recipe,
        device=device,
    )
    manifest = {
        "schema": "picf-next.molmoact2-m2-feature-cache.v1",
        "gate": M2_GATE,
        "checkpoint_id": foundation.host.checkpoint_id,
        "checkpoint_revision": foundation.host.checkpoint_revision,
        "foundation_recipe_sha256": foundation.recipe_sha256,
        "modality": recipe.cache.modality,
        "dtype": recipe.cache.dtype,
        "token_shape": [recipe.cache.token_count, recipe.cache.token_dim],
        "processor_layout": canonical_layout,
        "processor_layout_sha256": _canonical_sha256(canonical_layout),
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "shards": shards,
        "sample_count": len(records),
        "model_input_fields": ["tokens", "valid"],
        "loss_target_fields_in_feature_shards": [],
        "elapsed_s": time.perf_counter() - extraction_started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _write_json_atomic(cache_dir / "manifest.json", manifest)
    _emit_progress(
        "feature_cache_complete",
        sample_count=len(records),
        shard_count=len(shards),
        elapsed_s=manifest["elapsed_s"],
        task_intervention_maximum_absolute_error=task_intervention["maximum_absolute_error"],
    )
    del policy
    torch.cuda.empty_cache()
    return manifest, task_intervention


def _task_intervention_probe(
    *,
    policy: Any,
    processor: Any,
    assets: Any,
    recipe: MolmoAct2M2Recipe,
    device: Any,
) -> dict[str, Any]:
    import torch

    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation
    from picf_next.hosts.molmoact2_training import (
        calvin_visible_object_target_request,
        molmoact2_host_observation_view,
    )

    segments = {segment.index: segment for segment in assets.index.segments}
    pairs = []
    maximum_error = 0.0
    exact = True
    for control_index in recipe.splits.excluded_overlap_control_segments:
        control = segments[control_index]
        candidates = [
            segments[index]
            for index in recipe.splits.learned_segments
            if segments[index].task_key == control.task_key
            and max(segments[index].start, control.start) < min(segments[index].end, control.end)
        ]
        if len(candidates) != 1:
            raise RuntimeError("task intervention control no longer has one learned overlap")
        learned = candidates[0]
        start = max(control.start, learned.start)
        stop = min(control.end, learned.end)
        indices = sorted({start, (start + stop) // 2, stop - 1})
        for global_index in indices:
            left = assets.index.stateful_transition_sample(
                learned.index,
                global_index,
                action_horizon=assets.dataset.action_horizon,
            )
            right = assets.index.stateful_transition_sample(
                control.index,
                global_index,
                action_horizon=assets.dataset.action_horizon,
            )
            left_hashes = calvin_visible_object_target_request(left).source_sensor_sha256
            right_hashes = calvin_visible_object_target_request(right).source_sensor_sha256
            if left_hashes != right_hashes or left.record.task == right.record.task:
                raise RuntimeError("task intervention pair does not isolate language")
            evidence = ((left.picf_evidence_frame,), (right.picf_evidence_frame,))
            views = (
                molmoact2_host_observation_view(left.record),
                molmoact2_host_observation_view(right.record),
            )
            inputs = _move_inputs(processor.build_observation_inputs(evidence, views), device)
            with torch.inference_mode():
                prepared = prepare_molmoact2_lerobot_observation(policy, inputs)
            bank = prepared.vision_patch_bank
            if bank is None:
                raise RuntimeError("task intervention produced no dense feature bank")
            difference = (bank.tokens[0].float() - bank.tokens[1].float()).abs()
            error = float(difference.max().item())
            row_exact = bool(torch.equal(bank.tokens[0], bank.tokens[1]))
            maximum_error = max(maximum_error, error)
            exact = exact and row_exact
            pairs.append(
                {
                    "global_index": global_index,
                    "learned_segment": learned.index,
                    "control_segment": control.index,
                    "learned_instruction": left.record.task,
                    "control_instruction": right.record.task,
                    "source_sensor_sha256": [list(item) for item in left_hashes],
                    "dense_features_exact": row_exact,
                    "maximum_absolute_error": error,
                }
            )
    return {
        "schema": "picf-next.molmoact2-m2-task-intervention.v1",
        "pair_count": len(pairs),
        "pairs": pairs,
        "all_dense_features_exact": exact,
        "maximum_absolute_error": maximum_error,
        "task_text_enters_trainable_m2_graph": False,
    }


def _load_cache(
    cache_dir: Path,
    recipe: MolmoAct2M2Recipe,
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any, dict[str, Any]]]]:
    import torch

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if (
        manifest.get("schema") != "picf-next.molmoact2-m2-feature-cache.v1"
        or manifest.get("modality") != recipe.cache.modality
        or manifest.get("token_shape") != [recipe.cache.token_count, recipe.cache.token_dim]
        or manifest.get("loss_target_fields_in_feature_shards") != []
    ):
        raise ValueError("M2 feature cache manifest changed")
    records = manifest.get("records")
    shards = manifest.get("shards")
    if (
        not isinstance(records, list)
        or not isinstance(shards, list)
        or manifest.get("records_sha256") != _canonical_sha256(records)
    ):
        raise ValueError("M2 feature cache record manifest is invalid")
    record_by_location = {(row["shard"], row["row"]): row for row in records}
    loaded: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for shard in shards:
        path = cache_dir / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"M2 feature shard hash changed: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) != {"tokens", "valid"}:
            raise ValueError("M2 feature shard contains non-observation fields")
        tokens = payload["tokens"]
        valid = payload["valid"]
        if (
            tokens.dtype != torch.bfloat16
            or tokens.shape[1:] != (recipe.cache.token_count, recipe.cache.token_dim)
            or valid.dtype != torch.bool
            or valid.shape != tokens.shape[:2]
            or tokens.shape[0] != shard["rows"]
        ):
            raise ValueError("M2 feature shard tensor contract changed")
        for row_index in range(tokens.shape[0]):
            record = record_by_location.get((path.name, row_index))
            if record is None or record["sample_key"] in loaded:
                raise ValueError("M2 feature shard locations are not one-to-one")
            loaded[record["sample_key"]] = (
                tokens[row_index],
                valid[row_index],
                record,
            )
    if len(loaded) != len(records):
        raise ValueError("M2 feature cache omitted records")
    return manifest, loaded


def _request_from_record(record: Mapping[str, Any]) -> Any:
    common = {
        "sample_key": str(record["sample_key"]),
        "source_global_index": int(record["global_index"]),
        "augmentation_seed": 0,
        "source_sensor_sha256": tuple(
            (str(name), str(digest)) for name, digest in record["source_sensor_sha256"]
        ),
    }
    contract = record.get("target_request_contract", "language_segment")
    if contract == "language_segment":
        return CalvinStatefulLossTargetRequest(
            segment_index=int(record["segment_index"]),
            **common,
        )
    if contract == "source_frame":
        return CalvinSourceFrameLossTargetRequest(**common)
    raise ValueError(f"unsupported M2 target request contract: {contract!r}")


def _build_targets(
    *,
    target_builder: CalvinVisibleObjectTargetBuilder,
    records: Sequence[Mapping[str, Any]],
    token_valid: Any,
    target_dtype: Any,
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
) -> tuple[ObjectSetTarget, ...]:
    import torch

    from picf_next.hosts.molmoact2_training import CalvinStatefulLossTargetLayout

    layout = CalvinStatefulLossTargetLayout(
        token_valid=token_valid.detach().clone(),
        spans=(ModalityTokenSpan("molmo_vision_patch", 0, token_count),),
        target_dtype=torch.float32,
        rollout_input_dtype=target_dtype,
        vision_patch_layout=_layout_from_payload(
            layout_payload,
            batch_size=len(records),
            token_count=token_count,
        ),
    )
    requests = tuple(_request_from_record(record) for record in records)
    contracts = {record.get("target_request_contract", "language_segment") for record in records}
    if contracts == {"language_segment"}:
        result = target_builder(requests, layout)
    elif contracts == {"source_frame"}:
        result = target_builder.source_frames(requests, layout)
    else:
        raise ValueError("one M2 target batch cannot mix locator contracts")
    if result.set_targets is None or len(result.set_targets) != len(records):
        raise RuntimeError("M2 current-frame target builder omitted set targets")
    return result.set_targets


def _keys_for_split(
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    split: str,
) -> list[str]:
    keys = [key for key, (_tokens, _valid, record) in cache.items() if record["split"] == split]
    return sorted(
        keys,
        key=lambda key: (
            _record_group_identity(cache[key][2])[1],
            int(cache[key][2]["global_index"]),
        ),
    )


def _record_group_identity(record: Mapping[str, Any]) -> tuple[str, int]:
    """Return the one declared grouping axis for language or source rows."""

    names = tuple(name for name in ("segment_index", "source_block_index") if name in record)
    if len(names) != 1:
        raise ValueError("M2 cache record must declare exactly one nonnegative group identity")
    value = record[names[0]]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("M2 cache record must declare exactly one nonnegative group identity")
    return names[0], value


def _epoch_order(keys: Sequence[str], *, seed: int, epoch: int) -> list[str]:
    return sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{seed}:{epoch}:{key}".encode()).digest(),
    )


def _batch_plan(keys: Sequence[str], recipe: MolmoAct2M2Recipe) -> list[list[str]]:
    if not keys:
        raise ValueError("M2 batch plan requires at least one training sample")
    plan: list[list[str]] = []
    epoch = 0
    cursor: list[str] = []
    required = recipe.optimization.steps * recipe.optimization.batch_size
    while len(cursor) < required:
        cursor.extend(_epoch_order(keys, seed=recipe.optimization.seed, epoch=epoch))
        epoch += 1
    for start in range(0, required, recipe.optimization.batch_size):
        plan.append(cursor[start : start + recipe.optimization.batch_size])
    return plan


def _derangement(keys: Sequence[str], *, seed: int) -> dict[str, str]:
    ordered = _epoch_order(keys, seed=seed, epoch=1_000_003)
    if len(ordered) < 2:
        raise ValueError("label-shuffle control requires at least two training samples")
    rotated = ordered[1:] + ordered[:1]
    mapping = dict(zip(ordered, rotated, strict=True))
    if any(key == value for key, value in mapping.items()):
        raise RuntimeError("M2 label-shuffle mapping contains a fixed point")
    return mapping


def _stack_batch(
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    *,
    device: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    import torch

    tokens = torch.stack([cache[key][0] for key in keys]).to(device, non_blocking=True)
    valid = torch.stack([cache[key][1] for key in keys]).to(device, non_blocking=True)
    records = [dict(cache[key][2]) for key in keys]
    return tokens, valid, records


def _native_bank(tokens: Any, valid: Any) -> tuple[NativeTokenBank, ...]:
    return (
        NativeTokenBank(
            modality="molmo_vision_patch",
            tokens=tokens,
            valid=valid,
            encoder_contract="allenai/MolmoAct2.native-prepool-patches.e432d85",
        ),
    )


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().to(device="cpu").clone() for name, value in model.state_dict().items()
    }


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(value.shape)))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _learning_rate_multiplier(step: int, recipe: MolmoAct2M2Recipe) -> float:
    warmup = recipe.optimization.warmup_steps
    total = recipe.optimization.steps
    if step <= warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _safe_ratio_improvement(reference: float, actual: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(actual) or reference <= 0.0:
        return float("-inf")
    return (reference - actual) / reference


def _geometry_metric_payload(
    *,
    contract: PhysicalGeometryContract,
    model_chart_absolute_by_axis: Sequence[float],
    supervised_coordinate_count_by_axis: Sequence[int],
) -> dict[str, Any]:
    """Report geometry errors without conflating model-chart and physical units."""

    if (
        len(model_chart_absolute_by_axis) != contract.dimension
        or len(supervised_coordinate_count_by_axis) != contract.dimension
    ):
        raise ValueError("geometry metric accumulators differ from their contract")
    rows: list[dict[str, Any]] = []
    total_model_chart_absolute = 0.0
    total_physical_absolute = 0.0
    total_count = 0
    for axis, unit, scale, absolute, count in zip(
        contract.axes,
        contract.units,
        contract.normalization_scale,
        model_chart_absolute_by_axis,
        supervised_coordinate_count_by_axis,
        strict=True,
    ):
        if (
            isinstance(absolute, bool)
            or not isinstance(absolute, int | float)
            or not math.isfinite(float(absolute))
            or float(absolute) < 0.0
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError("geometry metric accumulators must be finite and nonnegative")
        model_chart_absolute = float(absolute)
        physical_absolute = model_chart_absolute * scale
        rows.append(
            {
                "axis": axis,
                "unit": unit,
                "normalization_scale": scale,
                "supervised_coordinate_count": count,
                "mae_model_chart": model_chart_absolute / count if count else None,
                "mae_physical": physical_absolute / count if count else None,
            }
        )
        total_model_chart_absolute += model_chart_absolute
        total_physical_absolute += physical_absolute
        total_count += count

    common_physical_unit = contract.units[0] if len(set(contract.units)) == 1 else None
    return {
        "geometry_contract_sha256": contract.fingerprint,
        "geometry_mae_model_chart": (
            total_model_chart_absolute / total_count if total_count else None
        ),
        "geometry_mae_physical": (
            total_physical_absolute / total_count
            if total_count and common_physical_unit is not None
            else None
        ),
        "geometry_mae_physical_unit": common_physical_unit,
        "geometry_mae_by_axis": rows,
    }


def _validation_selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Select representation quality without mixing unlike loss units."""

    geometry_mae = float(metrics["geometry_mae_physical"])
    if not math.isfinite(geometry_mae):
        geometry_mae = float("inf")
    return (
        float(metrics["mean_object_dice"]),
        float(metrics["ownership_accuracy"]),
        float(metrics["exact_count_accuracy"]),
        -float(metrics["count_mae"]),
        -geometry_mae,
    )


def _evaluate(
    *,
    model: Any,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    target_builder: CalvinVisibleObjectTargetBuilder,
    criterion: ObjectSetCriterion,
    layout_payload: Sequence[Mapping[str, Any]],
    recipe: MolmoAct2M2Recipe,
    device: Any,
    include_per_sample: bool = False,
) -> dict[str, Any]:
    import torch
    from scipy.stats import spearmanr

    model.eval()
    query_count = model.discovery.config.num_queries
    losses: dict[str, list[float]] = {}
    target_counts: list[int] = []
    target_inventory_counts: list[int] = []
    predicted_counts: list[int] = []
    ownership_correct = 0
    ownership_total = 0
    balanced_ownership: list[float] = []
    context_correct = 0
    context_total = 0
    object_dice_numerator = 0.0
    object_dice_denominator = 0.0
    geometry_contract = target_builder.sidecar.geometry_contract
    geometry_absolute_by_axis = [0.0] * geometry_contract.dimension
    geometry_coordinate_count_by_axis = [0] * geometry_contract.dimension
    geometry_squared_by_object: list[float] = []
    variance_by_object: list[float] = []
    duplicate_pair_dice: list[float] = []
    fragmented_objects = 0
    supervised_objects = 0
    per_sample: list[dict[str, Any]] = []

    for start in range(0, len(keys), recipe.optimization.batch_size):
        batch_keys = keys[start : start + recipe.optimization.batch_size]
        tokens, valid, records = _stack_batch(cache, batch_keys, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(_native_bank(tokens, valid))
        targets = _build_targets(
            target_builder=target_builder,
            records=records,
            token_valid=output.projection.token_valid,
            target_dtype=output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        result = criterion(output.discovery, targets)
        for name, value in result.losses.items():
            losses.setdefault(name, []).append(float(value.detach().float().item()))

        for batch_index, (target, match, key) in enumerate(
            zip(targets, result.matches, batch_keys, strict=True)
        ):
            prediction = output.discovery
            existence = prediction.existence[batch_index].float()
            active = existence > 0.5
            target_inventory_count = target.num_objects
            target_count = target_inventory_count
            predicted_count = int(active.sum().item())
            target_counts.append(target_count)
            target_inventory_counts.append(target_inventory_count)
            predicted_counts.append(predicted_count)
            valid_rows = target.supervision_valid
            raw_category = prediction.ownership[batch_index, valid_rows].float().argmax(dim=-1)
            category_map = torch.full(
                (query_count + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            category_map[-1] = target_inventory_count
            if match.prediction_indices.numel():
                category_map[match.prediction_indices] = match.target_indices
            mapped_prediction = category_map[raw_category]
            expected = target.ownership[valid_rows].float().argmax(dim=-1)
            correct = mapped_prediction == expected
            ownership_correct += int(correct.sum().item())
            ownership_total += int(correct.numel())
            context = expected == target_inventory_count
            context_correct += int((correct & context).sum().item())
            context_total += int(context.sum().item())

            target_probability = target.ownership[valid_rows].float()
            category_mass = target_probability.sum(dim=0)
            category_recall = torch.stack(
                [
                    (
                        target_probability[:, category]
                        * (mapped_prediction == category).to(torch.float32)
                    ).sum()
                    / category_mass[category].clamp_min(1e-6)
                    for category in range(target_inventory_count + 1)
                ]
            )
            active_category = category_mass > 0.0
            balanced_ownership.append(float(category_recall[active_category].mean().item()))

            sample_dice: list[float] = []
            if match.prediction_indices.numel():
                predicted = prediction.ownership[batch_index, valid_rows][
                    :, match.prediction_indices
                ].float()
                expected_masks = target.ownership[valid_rows][:, match.target_indices].float()
                numerator = 2.0 * (predicted * expected_masks).sum(dim=0)
                denominator = predicted.sum(dim=0) + expected_masks.sum(dim=0)
                dice = (numerator + 1.0) / (denominator + 1.0)
                sample_dice = [float(value) for value in dice.tolist()]
                object_dice_numerator += sum(sample_dice)
                object_dice_denominator += len(sample_dice)

                geometry_supervised = target.geometry_supervised
                if target.geometry is not None:
                    predicted_geometry = prediction.geometry_mean[
                        batch_index, match.prediction_indices
                    ].float()
                    target_geometry = target.geometry[match.target_indices].float()
                    residual = predicted_geometry - target_geometry
                    if geometry_supervised is None:
                        selected = torch.ones_like(residual, dtype=torch.bool)
                    else:
                        selected = geometry_supervised[match.target_indices]
                    selected_absolute = residual.abs().masked_fill(~selected, 0.0)
                    absolute_by_axis = selected_absolute.sum(dim=0).tolist()
                    count_by_axis = selected.sum(dim=0).tolist()
                    for axis_index, (absolute, count) in enumerate(
                        zip(absolute_by_axis, count_by_axis, strict=True)
                    ):
                        geometry_absolute_by_axis[axis_index] += float(absolute)
                        geometry_coordinate_count_by_axis[axis_index] += int(count)
                    for row_index in range(residual.shape[0]):
                        row_selected = selected[row_index]
                        if row_selected.any():
                            geometry_squared_by_object.append(
                                float(residual[row_index, row_selected].square().mean().item())
                            )
                            variance_by_object.append(
                                float(
                                    prediction.geometry_variance[
                                        batch_index,
                                        match.prediction_indices[row_index],
                                    ][row_selected]
                                    .float()
                                    .mean()
                                    .item()
                                )
                            )

            supervised = target.ownership[valid_rows, :-1].float()
            predicted_all = prediction.ownership[batch_index, valid_rows, :-1].float()
            for target_index in range(target_inventory_count):
                expected_mask = supervised[:, target_index]
                intersection = (predicted_all * expected_mask[:, None]).sum(dim=0)
                union = (predicted_all.sum(dim=0) + expected_mask.sum() - intersection).clamp_min(
                    1e-6
                )
                plausible = ((intersection / union) >= 0.25) & active
                fragmented_objects += max(int(plausible.sum().item()) - 1, 0)
                supervised_objects += 1

            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            if active_indices.numel() > 1:
                masks = predicted_all[:, active_indices]
                for left in range(masks.shape[1]):
                    for right in range(left + 1, masks.shape[1]):
                        numerator = 2.0 * (masks[:, left] * masks[:, right]).sum()
                        denominator = masks[:, left].sum() + masks[:, right].sum()
                        duplicate_pair_dice.append(
                            float(((numerator + 1.0) / (denominator + 1.0)).item())
                        )
            if include_per_sample:
                record = cache[key][2]
                group_name, group_index = _record_group_identity(record)
                sample_metrics = {
                    "sample_key": key,
                    "group_kind": group_name,
                    "group_index": group_index,
                    "global_index": record["global_index"],
                    "task_key": record["task_key"],
                    "target_object_count": target_count,
                    "target_inventory_object_count": target_inventory_count,
                    "predicted_object_count": predicted_count,
                    "exact_count": predicted_count == target_count,
                    "mean_object_dice": (
                        sum(sample_dice) / len(sample_dice) if sample_dice else 0.0
                    ),
                    "token_ownership_accuracy": (
                        float(correct.float().mean().item()) if correct.numel() else 0.0
                    ),
                    "ownership_accuracy": balanced_ownership[-1],
                }
                sample_metrics[group_name] = group_index
                per_sample.append(sample_metrics)
        del tokens, valid, output

    if not target_counts or ownership_total <= 0:
        raise RuntimeError("M2 evaluation produced no supervised evidence")
    count_errors = [
        abs(left - right) for left, right in zip(predicted_counts, target_counts, strict=True)
    ]
    uncertainty_spearman: float | None = None
    if (
        len(geometry_squared_by_object) >= 2
        and len(set(geometry_squared_by_object)) > 1
        and len(set(variance_by_object)) > 1
    ):
        statistic = spearmanr(geometry_squared_by_object, variance_by_object).statistic
        if math.isfinite(float(statistic)):
            uncertainty_spearman = float(statistic)
    geometry_metrics = _geometry_metric_payload(
        contract=geometry_contract,
        model_chart_absolute_by_axis=geometry_absolute_by_axis,
        supervised_coordinate_count_by_axis=geometry_coordinate_count_by_axis,
    )
    metrics: dict[str, Any] = {
        "sample_count": len(keys),
        "target_object_count_mean": sum(target_counts) / len(target_counts),
        "target_inventory_object_count_mean": sum(target_inventory_counts)
        / len(target_inventory_counts),
        "predicted_object_count_mean": sum(predicted_counts) / len(predicted_counts),
        "count_mae": sum(count_errors) / len(count_errors),
        "exact_count_accuracy": sum(error == 0 for error in count_errors) / len(count_errors),
        "ownership_accuracy": sum(balanced_ownership) / len(balanced_ownership),
        "balanced_ownership_accuracy": sum(balanced_ownership) / len(balanced_ownership),
        "token_ownership_accuracy": ownership_correct / ownership_total,
        "context_accuracy": context_correct / max(context_total, 1),
        "mean_object_dice": object_dice_numerator / max(object_dice_denominator, 1e-6),
        # Deprecated compatibility alias. New consumers must choose an explicit chart.
        "geometry_mae": geometry_metrics["geometry_mae_model_chart"],
        **geometry_metrics,
        "uncertainty_error_spearman": uncertainty_spearman,
        "mean_active_queries": sum(predicted_counts) / len(predicted_counts),
        "mean_unused_query_capacity": query_count - sum(predicted_counts) / len(predicted_counts),
        "query_utilization": sum(predicted_counts) / (len(predicted_counts) * query_count),
        "fragmentation_excess_per_object": fragmented_objects / max(supervised_objects, 1),
        "maximum_active_query_pair_dice": max(duplicate_pair_dice, default=0.0),
        "nonfinite_metric_count": 0,
        "losses": {name: sum(values) / len(values) for name, values in sorted(losses.items())},
    }
    if include_per_sample:
        metrics["per_sample"] = per_sample
    for value in metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            metrics["nonfinite_metric_count"] += 1
    return metrics


def _all_context_baseline(
    *,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    target_builder: CalvinVisibleObjectTargetBuilder,
    layout_payload: Sequence[Mapping[str, Any]],
    recipe: MolmoAct2M2Recipe,
    device: Any,
) -> dict[str, Any]:
    import torch

    target_counts = []
    target_inventory_counts = []
    correct = 0
    total = 0
    balanced = []
    for start in range(0, len(keys), recipe.optimization.batch_size):
        batch_keys = keys[start : start + recipe.optimization.batch_size]
        _tokens, valid, records = _stack_batch(cache, batch_keys, device=device)
        targets = _build_targets(
            target_builder=target_builder,
            records=records,
            token_valid=valid,
            target_dtype=torch.bfloat16,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        for target in targets:
            target_counts.append(target.num_objects)
            target_inventory_counts.append(target.num_objects)
            supervised = target.supervision_valid
            expected = target.ownership[supervised].argmax(dim=-1)
            correct += int((expected == target.num_objects).sum().item())
            total += int(expected.numel())
            category_mass = target.ownership[supervised].float().sum(dim=0)
            active = category_mass > 0.0
            context_recall = torch.zeros_like(category_mass)
            context_recall[-1] = 1.0
            balanced.append(float(context_recall[active].mean().item()))
    return {
        "sample_count": len(keys),
        "count_mae": sum(target_counts) / len(target_counts),
        "exact_count_accuracy": sum(value == 0 for value in target_counts) / len(target_counts),
        "target_object_count_mean": sum(target_counts) / len(target_counts),
        "target_inventory_object_count_mean": sum(target_inventory_counts)
        / len(target_inventory_counts),
        "ownership_accuracy": sum(balanced) / len(balanced),
        "balanced_ownership_accuracy": sum(balanced) / len(balanced),
        "token_ownership_accuracy": correct / total,
        "mean_object_dice": 0.0,
        "geometry_mae": None,
        "geometry_contract_sha256": target_builder.sidecar.geometry_contract.fingerprint,
        "geometry_mae_model_chart": None,
        "geometry_mae_physical": None,
        "geometry_mae_physical_unit": (
            target_builder.sidecar.geometry_contract.units[0]
            if len(set(target_builder.sidecar.geometry_contract.units)) == 1
            else None
        ),
        "geometry_mae_by_axis": [],
        "mean_active_queries": 0.0,
    }


def _query_permutation_probe(
    *,
    model: Any,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    recipe: MolmoAct2M2Recipe,
    device: Any,
) -> dict[str, Any]:
    import torch

    selected = list(keys[: min(len(keys), recipe.optimization.batch_size)])
    tokens, valid, _records = _stack_batch(cache, selected, device=device)
    permuted = copy.deepcopy(model).to(device).eval()
    query_count = model.discovery.config.num_queries
    permutation = torch.arange(query_count - 1, -1, -1, device=device)
    with torch.no_grad():
        permuted.discovery.query_embeddings.copy_(model.discovery.query_embeddings[permutation])

    def field_errors(actual: Any, expected: Any) -> dict[str, float]:
        return {
            "query_features": float(
                (actual.query_features - expected.query_features[:, permutation]).abs().max().item()
            ),
            "address_mean": float(
                (actual.address_mean - expected.address_mean[:, permutation]).abs().max().item()
            ),
            "content_mean": float(
                (actual.content_mean - expected.content_mean[:, permutation]).abs().max().item()
            ),
            "geometry_mean": float(
                (actual.geometry_mean - expected.geometry_mean[:, permutation]).abs().max().item()
            ),
            "geometry_variance": float(
                (actual.geometry_variance - expected.geometry_variance[:, permutation])
                .abs()
                .max()
                .item()
            ),
            "existence_logits": float(
                (actual.existence_logits - expected.existence_logits[:, permutation])
                .abs()
                .max()
                .item()
            ),
            "ownership_objects": float(
                (actual.ownership[..., :-1] - expected.ownership[..., :-1][..., permutation])
                .abs()
                .max()
                .item()
            ),
            "ownership_context": float(
                (actual.context_ownership - expected.context_ownership).abs().max().item()
            ),
        }

    with torch.inference_mode():
        expected_float32 = model(_native_bank(tokens.float(), valid)).discovery
        actual_float32 = permuted(_native_bank(tokens.float(), valid)).discovery
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected_bfloat16 = model(_native_bank(tokens, valid)).discovery
        actual_bfloat16 = permuted(_native_bank(tokens, valid)).discovery
    errors = field_errors(actual_float32, expected_float32)
    runtime_errors = field_errors(actual_bfloat16, expected_bfloat16)
    return {
        "schema": "picf-next.molmoact2-m2-query-permutation.v1",
        "sample_keys": selected,
        "permutation": permutation.cpu().tolist(),
        "field_maximum_absolute_error": errors,
        "maximum_absolute_error": max(errors.values()),
        "runtime_bfloat16_field_maximum_absolute_error": runtime_errors,
        "runtime_bfloat16_maximum_absolute_error": max(runtime_errors.values()),
    }


def _train_models(
    *,
    run_dir: Path,
    recipe: MolmoAct2M2Recipe,
    foundation: Any,
    assets: Any,
    cache_manifest: Mapping[str, Any],
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    import torch

    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder

    train_keys = _keys_for_split(cache, "train")
    validation_keys = _keys_for_split(cache, "validation")
    heldout_keys = _keys_for_split(cache, "heldout")
    if not train_keys or not validation_keys or not heldout_keys:
        raise ValueError("M2 training requires nonempty train, validation and heldout splits")
    plan = _batch_plan(train_keys, recipe)
    shuffle = _derangement(train_keys, seed=recipe.optimization.seed)
    _write_json_atomic(
        run_dir / "batch_plan.json",
        {
            "schema": "picf-next.molmoact2-m2-batch-plan.v1",
            "algorithm": "sha256-epoch-sort.v1",
            "seed": recipe.optimization.seed,
            "batches": plan,
            "batches_sha256": _canonical_sha256(plan),
            "label_shuffle": shuffle,
            "label_shuffle_sha256": _canonical_sha256(shuffle),
        },
    )

    torch.manual_seed(recipe.optimization.seed)
    actual = foundation.core_config.build_current_frame()
    control = copy.deepcopy(actual)
    initial_state = _state_dict_cpu(actual)
    actual_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    actual.to(actual_device)
    control.to(control_device)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    actual_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(actual_device)
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(control_device)
    layout_payload = cache_manifest["processor_layout"]

    model_stage_started = time.perf_counter()
    random_metrics = _evaluate(
        model=actual,
        cache=cache,
        keys=heldout_keys,
        target_builder=target_builder,
        criterion=actual_criterion,
        layout_payload=layout_payload,
        recipe=recipe,
        device=actual_device,
    )
    all_context = _all_context_baseline(
        cache=cache,
        keys=heldout_keys,
        target_builder=target_builder,
        layout_payload=layout_payload,
        recipe=recipe,
        device=actual_device,
    )

    actual_optimizer = torch.optim.AdamW(
        actual.parameters(),
        lr=recipe.optimization.learning_rate,
        weight_decay=recipe.optimization.weight_decay,
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=recipe.optimization.learning_rate,
        weight_decay=recipe.optimization.weight_decay,
    )
    best_validation_loss = float("inf")
    best_validation_key: tuple[float, ...] | None = None
    best_step = 0
    best_state: dict[str, Any] | None = None
    paired_control_state: dict[str, Any] | None = None
    metrics_rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(actual_device)
    torch.cuda.reset_peak_memory_stats(control_device)
    torch.cuda.synchronize(actual_device)
    torch.cuda.synchronize(control_device)
    optimization_started = time.perf_counter()
    _emit_progress(
        "optimization_started",
        steps=recipe.optimization.steps,
        batch_size=recipe.optimization.batch_size,
        train_samples=len(train_keys),
        validation_samples=len(validation_keys),
        heldout_samples=len(heldout_keys),
    )

    for step, batch_keys in enumerate(plan, start=1):
        actual.train()
        control.train()
        actual_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        actual_tokens, actual_valid, actual_records = _stack_batch(
            cache,
            batch_keys,
            device=actual_device,
        )
        control_tokens, control_valid, _control_records = _stack_batch(
            cache,
            batch_keys,
            device=control_device,
        )
        shuffled_records = [dict(cache[shuffle[key]][2]) for key in batch_keys]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual_output = actual(_native_bank(actual_tokens, actual_valid))
            control_output = control(_native_bank(control_tokens, control_valid))
        actual_targets = _build_targets(
            target_builder=target_builder,
            records=actual_records,
            token_valid=actual_output.projection.token_valid,
            target_dtype=actual_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        control_targets = _build_targets(
            target_builder=target_builder,
            records=shuffled_records,
            token_valid=control_output.projection.token_valid,
            target_dtype=control_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        actual_result = actual_criterion(actual_output.discovery, actual_targets)
        control_result = control_criterion(control_output.discovery, control_targets)
        actual_result.total.backward()
        control_result.total.backward()
        actual_grad_norm = torch.nn.utils.clip_grad_norm_(
            actual.parameters(),
            recipe.optimization.gradient_clip_norm,
        )
        control_grad_norm = torch.nn.utils.clip_grad_norm_(
            control.parameters(),
            recipe.optimization.gradient_clip_norm,
        )
        if not torch.isfinite(actual_grad_norm) or not torch.isfinite(control_grad_norm):
            raise FloatingPointError("M2 gradient norm became non-finite")
        multiplier = _learning_rate_multiplier(step, recipe)
        for optimizer in (actual_optimizer, control_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = recipe.optimization.learning_rate * multiplier
            optimizer.step()
        row = {
            "step": step,
            "learning_rate": recipe.optimization.learning_rate * multiplier,
            "actual_loss": float(actual_result.total.detach().float().item()),
            "label_shuffle_loss": float(control_result.total.detach().float().item()),
            "actual_gradient_norm": float(actual_grad_norm.detach().float().item()),
            "label_shuffle_gradient_norm": float(control_grad_norm.detach().float().item()),
        }
        for name, value in actual_result.losses.items():
            row[f"actual_{name}"] = float(value.detach().float().item())
        for name, value in control_result.losses.items():
            row[f"label_shuffle_{name}"] = float(value.detach().float().item())

        if step % recipe.optimization.validation_interval == 0:
            validation = _evaluate(
                model=actual,
                cache=cache,
                keys=validation_keys,
                target_builder=target_builder,
                criterion=actual_criterion,
                layout_payload=layout_payload,
                recipe=recipe,
                device=actual_device,
            )
            validation_loss = float(validation["losses"]["loss_total"])
            row["validation_loss_total"] = validation_loss
            row["validation_ownership_accuracy"] = validation["ownership_accuracy"]
            row["validation_token_ownership_accuracy"] = validation["token_ownership_accuracy"]
            row["validation_mean_object_dice"] = validation["mean_object_dice"]
            row["validation_count_mae"] = validation["count_mae"]
            validation_key = _validation_selection_key(validation)
            row["validation_selection_key"] = list(validation_key)
            if best_validation_key is None or validation_key > best_validation_key:
                best_validation_key = validation_key
                best_validation_loss = validation_loss
                best_step = step
                best_state = _state_dict_cpu(actual)
                paired_control_state = _state_dict_cpu(control)
            _emit_progress(
                "validation",
                step=step,
                actual_loss=row["actual_loss"],
                label_shuffle_loss=row["label_shuffle_loss"],
                validation_loss_total=validation_loss,
                validation_ownership_accuracy=validation["ownership_accuracy"],
                validation_token_ownership_accuracy=validation["token_ownership_accuracy"],
                validation_mean_object_dice=validation["mean_object_dice"],
                validation_count_mae=validation["count_mae"],
                best_validation_step=best_step,
            )
        metrics_rows.append(row)

    torch.cuda.synchronize(actual_device)
    torch.cuda.synchronize(control_device)
    optimization_elapsed = time.perf_counter() - optimization_started
    optimization_peak = {
        "cuda:0": int(torch.cuda.max_memory_allocated(actual_device)),
        "cuda:1": int(torch.cuda.max_memory_allocated(control_device)),
    }
    if (
        best_state is None
        or paired_control_state is None
        or best_validation_key is None
        or best_step <= 0
    ):
        raise RuntimeError("M2 never selected a validation checkpoint")
    actual.load_state_dict(best_state, strict=True)
    control.load_state_dict(paired_control_state, strict=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    actual_path = checkpoint_dir / "current_frame_best.pt"
    control_path = checkpoint_dir / "label_shuffle_paired_best_step.pt"
    _write_torch_atomic(actual_path, {"model": best_state})
    control_state = _state_dict_cpu(control)
    _write_torch_atomic(control_path, {"model": control_state})

    evaluation_started = time.perf_counter()
    actual_train = _evaluate(
        model=actual,
        cache=cache,
        keys=train_keys,
        target_builder=target_builder,
        criterion=actual_criterion,
        layout_payload=layout_payload,
        recipe=recipe,
        device=actual_device,
    )
    actual_validation = _evaluate(
        model=actual,
        cache=cache,
        keys=validation_keys,
        target_builder=target_builder,
        criterion=actual_criterion,
        layout_payload=layout_payload,
        recipe=recipe,
        device=actual_device,
    )
    actual_heldout = _evaluate(
        model=actual,
        cache=cache,
        keys=heldout_keys,
        target_builder=target_builder,
        criterion=actual_criterion,
        layout_payload=layout_payload,
        recipe=recipe,
        device=actual_device,
        include_per_sample=True,
    )
    control_heldout = _evaluate(
        model=control,
        cache=cache,
        keys=heldout_keys,
        target_builder=target_builder,
        criterion=control_criterion,
        layout_payload=layout_payload,
        recipe=recipe,
        device=control_device,
    )
    query_permutation = _query_permutation_probe(
        model=actual,
        cache=cache,
        keys=heldout_keys,
        recipe=recipe,
        device=actual_device,
    )
    torch.cuda.synchronize(actual_device)
    torch.cuda.synchronize(control_device)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    model_stage_elapsed = time.perf_counter() - model_stage_started
    _emit_progress(
        "optimization_complete",
        best_validation_step=best_step,
        optimization_elapsed_s=optimization_elapsed,
        seconds_per_joint_actual_and_control_step=(
            optimization_elapsed / recipe.optimization.steps
        ),
        heldout_ownership_accuracy=actual_heldout["ownership_accuracy"],
        heldout_mean_object_dice=actual_heldout["mean_object_dice"],
        heldout_exact_count_accuracy=actual_heldout["exact_count_accuracy"],
    )
    training_report = {
        "schema": "picf-next.molmoact2-m2-training.v1",
        "gate": M2_GATE,
        "steps": recipe.optimization.steps,
        "batch_size": recipe.optimization.batch_size,
        "trainable_parameter_names": sorted(name for name, _parameter in actual.named_parameters()),
        "forbidden_parameter_prefixes_present": any(
            name.startswith(("posterior_filter.", "policy.", "action"))
            for name, _parameter in actual.named_parameters()
        ),
        "initial_state_sha256": _state_dict_sha256(initial_state),
        "best_state_sha256": _state_dict_sha256(best_state),
        "label_shuffle_state_sha256": _state_dict_sha256(control_state),
        "best_validation_step": best_step,
        "label_shuffle_checkpoint_step": best_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_selection_key": list(best_validation_key),
        "optimization_elapsed_s": optimization_elapsed,
        "post_training_evaluation_elapsed_s": evaluation_elapsed,
        "model_stage_elapsed_s": model_stage_elapsed,
        "seconds_per_joint_actual_and_control_step": (
            optimization_elapsed / recipe.optimization.steps
        ),
        "optimization_cuda_peak_allocated_bytes": optimization_peak,
        "checkpoints": {
            "current_frame_best.pt": _sha256(actual_path),
            "label_shuffle_paired_best_step.pt": _sha256(control_path),
        },
        "metrics": metrics_rows,
    }
    evaluation = {
        "schema": "picf-next.molmoact2-m2-evaluation.v1",
        "gate": M2_GATE,
        "random_initialization": random_metrics,
        "all_context": all_context,
        "label_shuffle": control_heldout,
        "actual": {
            "train": actual_train,
            "validation": actual_validation,
            "heldout": actual_heldout,
        },
        "query_permutation": query_permutation,
    }
    return training_report, evaluation, actual


def _color(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 212],
            [0, 128, 128],
            [220, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
        ],
        dtype=np.uint8,
    )
    return palette[index % len(palette)]


def _overlay(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    object_count: int,
    unknown_label: int | None = None,
) -> np.ndarray:
    from PIL import Image

    height, width = image.shape[:2]
    resized = np.asarray(
        Image.fromarray(labels.astype(np.uint8)).resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
    )
    result = image.astype(np.float32).copy()
    if unknown_label is not None:
        unknown = resized == unknown_label
        if unknown.any():
            result[unknown] = 0.35 * result[unknown] + 0.65 * np.asarray(
                [128, 128, 128],
                dtype=np.float32,
            )
    for object_index in range(object_count):
        selected = resized == object_index
        if selected.any():
            result[selected] = 0.48 * result[selected] + 0.52 * _color(object_index)
    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_anchor_points(
    image: Any,
    probability: np.ndarray,
    *,
    width: int,
    height: int,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    rows = int(round(math.sqrt(probability.shape[0])))
    if rows * rows != probability.shape[0]:
        return
    columns = rows
    for object_index in range(probability.shape[1]):
        mass = probability[:, object_index].reshape(rows, columns)
        total = float(mass.sum())
        if total <= 1e-5:
            continue
        yy, xx = np.mgrid[0:rows, 0:columns]
        x = float((mass * (xx + 0.5)).sum() / total) / columns * width
        y = float((mass * (yy + 0.5)).sum() / total) / rows * height
        radius = 6
        color = tuple(int(value) for value in _color(object_index))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=2,
        )


def _sensor_rgb(sample: Any, host_key: str) -> np.ndarray:
    source_by_host = {
        CALVIN_HOST_IMAGE_KEYS[0]: "observation.images.rgb_static",
        CALVIN_HOST_IMAGE_KEYS[1]: "observation.images.rgb_gripper",
    }
    if hasattr(sample, "record"):
        key = source_by_host[host_key]
        arrays = {
            observation.key: observation.value for observation in sample.record.array_observations
        }
        value = np.asarray(arrays[key])
    elif hasattr(sample, "images"):
        value = np.asarray(sample.images[host_key])
    else:
        raise TypeError("M2 visual source has neither a transition record nor source images")
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("CALVIN RGB source changed")
    return value


def _render_visuals(
    *,
    run_dir: Path,
    model: Any,
    assets: Any,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    cache_manifest: Mapping[str, Any],
    foundation: Any,
    recipe: MolmoAct2M2Recipe,
    visual_splits: Sequence[str] = ("train", "validation", "heldout"),
    expected_segments: set[int] | None = None,
    gate: str = M2_GATE,
) -> dict[str, Any]:
    import torch
    from PIL import Image, ImageDraw

    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder

    visuals = run_dir / "visuals"
    visuals.mkdir()
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to("cuda:0")
    model.eval()
    visual_splits = tuple(visual_splits)
    if not visual_splits or len(set(visual_splits)) != len(visual_splits):
        raise ValueError("visual splits must be nonempty and unique")
    if expected_segments is None:
        expected_segments = set(recipe.splits.learned_segments)
    chosen: list[str] = []
    for split in visual_splits:
        keys = _keys_for_split(cache, split)
        by_segment: dict[int, list[str]] = {}
        for key in keys:
            record = cache[key][2]
            _group_kind, group_index = _record_group_identity(record)
            by_segment.setdefault(group_index, []).append(key)
        for segment_keys in by_segment.values():
            positions = sorted({0, len(segment_keys) // 2, len(segment_keys) - 1})
            chosen.extend(segment_keys[position] for position in positions)
    chosen = list(dict.fromkeys(chosen))
    artifacts = []
    layout_payload = cache_manifest["processor_layout"]

    for key in chosen:
        tokens, valid, records = _stack_batch(cache, [key], device=torch.device("cuda:0"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(_native_bank(tokens, valid))
        targets = _build_targets(
            target_builder=target_builder,
            records=records,
            token_valid=output.projection.token_valid,
            target_dtype=output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        result = criterion(output.discovery, targets)
        target = targets[0]
        match = result.matches[0]
        prediction = output.discovery
        object_count = target.num_objects
        existence = prediction.existence[0].float()
        active_indices = torch.nonzero(existence > 0.5, as_tuple=False).flatten()
        predicted_count = int(active_indices.numel())
        remapped = torch.zeros(
            recipe.cache.token_count,
            object_count + 1,
            device="cuda:0",
            dtype=torch.float32,
        )
        remapped[:, -1] = prediction.context_ownership[0].float()
        if match.prediction_indices.numel():
            remapped.index_copy_(
                1,
                match.target_indices,
                prediction.ownership[0].index_select(1, match.prediction_indices).float(),
            )
        raw_prediction = torch.cat(
            (
                prediction.ownership[0].index_select(1, active_indices).float(),
                prediction.context_ownership[0, :, None].float(),
            ),
            dim=1,
        )
        target_probability = target.ownership.float().cpu().numpy()
        target_supervised = target.supervision_valid.cpu().numpy()
        predicted_probability = remapped.cpu().numpy()
        raw_prediction_probability = raw_prediction.cpu().numpy()
        record = cache[key][2]
        if record.get("target_request_contract", "language_segment") == "source_frame":
            sample = assets.index.molmoact2_source_observation(int(record["global_index"]))
        else:
            sample = assets.dataset.by_key(key)
        rows = []
        for span_payload in layout_payload:
            image_key = span_payload["image_key"]
            start = int(span_payload["start"])
            stop = int(span_payload["stop"])
            source = _sensor_rgb(sample, image_key)
            target_labels = target_probability[start:stop].argmax(axis=-1).reshape(27, 27)
            target_known = target_supervised[start:stop].reshape(27, 27)
            predicted_labels = predicted_probability[start:stop].argmax(axis=-1).reshape(27, 27)
            raw_prediction_labels = (
                raw_prediction_probability[start:stop].argmax(axis=-1).reshape(27, 27)
            )
            target_labels[target_labels == object_count] = 255
            target_labels[~target_known] = 254
            predicted_labels[predicted_labels == object_count] = 255
            raw_prediction_labels[raw_prediction_labels == predicted_count] = 255
            panels = []
            for title, array, probability in (
                ("source RGB", source, None),
                (
                    "loss-only target",
                    _overlay(
                        source,
                        target_labels,
                        object_count=object_count,
                        unknown_label=254,
                    ),
                    target_probability[start:stop, :-1],
                ),
                (
                    "matched ownership (existence ignored)",
                    _overlay(source, predicted_labels, object_count=object_count),
                    predicted_probability[start:stop, :-1],
                ),
                (
                    "active queries (raw)",
                    _overlay(source, raw_prediction_labels, object_count=predicted_count),
                    raw_prediction_probability[start:stop, :-1],
                ),
            ):
                panel = Image.fromarray(array).resize((378, 378), Image.Resampling.NEAREST)
                if probability is not None:
                    _draw_anchor_points(panel, probability, width=378, height=378)
                canvas = Image.new("RGB", (378, 408), "white")
                canvas.paste(panel, (0, 30))
                ImageDraw.Draw(canvas).text((8, 8), f"{image_key}: {title}", fill="black")
                panels.append(canvas)
            row_width = 378 * len(panels)
            row = Image.new("RGB", (row_width, 408), "white")
            for index, panel in enumerate(panels):
                row.paste(panel, (378 * index, 0))
            rows.append(row)
        header = 88
        legend = 44 + 18 * min(max(object_count, predicted_count), 12)
        canvas_width = max(row.width for row in rows)
        canvas = Image.new(
            "RGB",
            (canvas_width, header + 408 * len(rows) + legend),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        group_kind, group_index = _record_group_identity(record)
        group_label = f"{group_kind.removesuffix('_index')}={group_index}"
        draw.text(
            (10, 8),
            (
                f"{record['split']} | {group_label} | "
                f"step={record['global_index']} | task={record['instruction']}"
            ),
            fill="black",
        )
        draw.text(
            (10, 34),
            (
                f"target objects={object_count}, predicted active={predicted_count}, "
                f"set loss={float(result.total.float().item()):.4f}"
            ),
            fill="black",
        )
        draw.text(
            (10, 58),
            (
                "Circles are ownership-mass centroids; gray target patches are unknown; "
                "targets never enter the forward path."
            ),
            fill="black",
        )
        for row_index, row in enumerate(rows):
            canvas.paste(row, (0, header + 408 * row_index))
        legend_y = header + 408 * len(rows) + 8
        for object_index, name in enumerate((target.temporal_identity_keys or ())[:12]):
            color = tuple(int(value) for value in _color(object_index))
            draw.rectangle((10, legend_y, 26, legend_y + 12), fill=color)
            draw.text((34, legend_y - 2), f"{object_index}: {name}", fill="black")
            if object_index < predicted_count:
                query_index = int(active_indices[object_index].item())
                probability = float(existence[query_index].item())
                draw.rectangle(
                    (canvas_width // 2, legend_y, canvas_width // 2 + 16, legend_y + 12),
                    fill=color,
                )
                draw.text(
                    (canvas_width // 2 + 24, legend_y - 2),
                    f"raw color {object_index}: query={query_index}, existence={probability:.3f}",
                    fill="black",
                )
            legend_y += 18
        for raw_index in range(min(object_count, 12), min(predicted_count, 12)):
            query_index = int(active_indices[raw_index].item())
            probability = float(existence[query_index].item())
            color = tuple(int(value) for value in _color(raw_index))
            draw.rectangle(
                (canvas_width // 2, legend_y, canvas_width // 2 + 16, legend_y + 12),
                fill=color,
            )
            draw.text(
                (canvas_width // 2 + 24, legend_y - 2),
                f"raw color {raw_index}: query={query_index}, existence={probability:.3f}",
                fill="black",
            )
            legend_y += 18
        safe_task = re.sub(r"[^a-z0-9]+", "_", str(record["task_key"]).lower()).strip("_")
        filename = (
            f"{record['split']}_group{group_index:02d}_"
            f"step{int(record['global_index']):07d}_{safe_task}.png"
        )
        path = visuals / filename
        canvas.save(path)
        artifact = {
            "path": f"visuals/{filename}",
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "sample_key": key,
            "split": record["split"],
            "group_kind": group_kind,
            "group_index": group_index,
            "global_index": record["global_index"],
            "task_key": record["task_key"],
            "instruction": record["instruction"],
            "target_object_count": object_count,
            "predicted_active_query_count": predicted_count,
            "active_query_indices": [int(value) for value in active_indices.cpu().tolist()],
            "active_query_existence": [float(existence[index].item()) for index in active_indices],
        }
        artifact[group_kind] = group_index
        artifacts.append(artifact)
        _emit_progress(
            "visual_artifact",
            completed=len(artifacts),
            total=len(chosen),
            path=f"visuals/{filename}",
            split=record["split"],
            group_index=group_index,
            global_index=record["global_index"],
        )
    manifest = {
        "schema": "picf-next.molmoact2-m2-visual-artifacts.v1",
        "gate": gate,
        "artifacts": artifacts,
        "artifacts_sha256": _canonical_sha256(artifacts),
        "all_splits_present": set(row["split"] for row in artifacts) == set(visual_splits),
        "all_learned_segments_present": set(row["group_index"] for row in artifacts)
        == expected_segments,
        "camera_views_per_artifact": len(layout_payload),
    }
    return manifest


def _evaluate_acceptance(
    *,
    recipe: MolmoAct2M2Recipe,
    evaluation: Mapping[str, Any],
    task_intervention: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    actual = evaluation["actual"]["heldout"]
    random = evaluation["random_initialization"]
    context = evaluation["all_context"]
    shuffled = evaluation["label_shuffle"]
    query = evaluation["query_permutation"]
    acceptance = recipe.acceptance
    spearman = actual["uncertainty_error_spearman"]
    checks = {
        "trainable_scope_is_current_frame_only": training["forbidden_parameter_prefixes_present"]
        is False,
        "heldout_ownership_accuracy": actual["ownership_accuracy"]
        >= acceptance.minimum_ownership_accuracy,
        "heldout_mean_object_dice": actual["mean_object_dice"]
        >= acceptance.minimum_mean_object_dice,
        "heldout_exact_count_accuracy": actual["exact_count_accuracy"]
        >= acceptance.minimum_heldout_exact_count_accuracy,
        "ownership_beats_all_context": (
            actual["ownership_accuracy"] - context["ownership_accuracy"]
            >= acceptance.minimum_ownership_accuracy_improvement_vs_all_context
        ),
        "dice_beats_random": (
            actual["mean_object_dice"] - random["mean_object_dice"]
            >= acceptance.minimum_random_dice_margin
        ),
        "count_mae_beats_random": _safe_ratio_improvement(
            random["count_mae"],
            actual["count_mae"],
        )
        >= acceptance.minimum_count_mae_improvement_fraction_vs_random,
        "geometry_mae_beats_random": _safe_ratio_improvement(
            random["geometry_mae_physical"],
            actual["geometry_mae_physical"],
        )
        >= acceptance.minimum_geometry_mae_improvement_fraction_vs_random,
        "ownership_beats_label_shuffle": (
            actual["ownership_accuracy"] - shuffled["ownership_accuracy"]
            >= acceptance.minimum_label_shuffle_ownership_accuracy_margin
        ),
        "dice_beats_label_shuffle": (
            actual["mean_object_dice"] - shuffled["mean_object_dice"]
            >= acceptance.minimum_label_shuffle_dice_margin
        ),
        "uncertainty_ranks_errors": (
            spearman is not None and spearman >= acceptance.minimum_uncertainty_error_spearman
        ),
        "query_permutation_equivariance": query["maximum_absolute_error"]
        <= acceptance.maximum_query_permutation_error,
        "task_intervention_exact": task_intervention["maximum_absolute_error"]
        <= acceptance.maximum_task_intervention_feature_error
        and task_intervention["all_dense_features_exact"] is True,
        "no_nonfinite_metrics": actual["nonfinite_metric_count"] == 0,
    }
    return {
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "status": "PASS_PENDING_VISUAL_REVIEW" if all(checks.values()) else "FAIL",
    }


def validate_m2_machine_decision(run_dir: Path) -> dict[str, Any]:
    decision_path = run_dir / "machine_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("M2 machine decision is absent")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "picf-next.molmoact2-m2-machine-decision.v1"
        or decision.get("gate") != M2_GATE
        or decision.get("status") not in {"PASS_PENDING_VISUAL_REVIEW", "FAIL"}
    ):
        raise ValueError("M2 machine decision identity or status changed")
    hashes = decision.get("required_report_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_M2_MACHINE_REPORTS):
        raise ValueError("M2 machine decision report set changed")
    for relative, expected in hashes.items():
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"M2 machine report hash changed: {relative}")
    return decision


def main() -> None:
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets
    from tools.train_molmoact2_calvin_picf import _validate_training_checkpoint

    args = _parse_args()
    recipe = load_molmoact2_m2_recipe(args.config)
    foundation = recipe.load_foundation(_ROOT)
    prior_m1 = validate_prior_m1(args.m1_run)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    dataset_split_root = args.dataset_split_root.expanduser().resolve()
    sidecar_artifact_root = args.sidecar_artifact_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not _is_under_mnt(run_root):
        raise RuntimeError("M2 run root must persist under /mnt")
    run_dir = run_root / "molmoact2" / M2_GATE / _run_id(args.run_id)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite M2 run: {run_dir}")

    code_revision = _clean_git_revision()
    static_report = m2_recipe_report(recipe, repository_root=_ROOT)
    if args.dry_run:
        print(json.dumps(static_report, indent=2, sort_keys=True))
        return

    resources = _validate_devices()
    _validate_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        m0_report=prior_m1.pop("m0_raw_report"),
        checkpoint_id=foundation.host.checkpoint_id,
        checkpoint_revision=foundation.host.checkpoint_revision,
    )
    sidecar_materialization = materialize_persistent_sidecars(sidecar_artifact_root)
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=dataset_split_root,
    )
    split_report = _validate_split_contract(assets, recipe)
    _emit_progress(
        "preflight_complete",
        code_revision=code_revision,
        checkpoint_dir=str(checkpoint_dir),
        dataset_samples=len(assets.dataset),
        split_transition_counts=split_report["transition_counts"],
        restored_sidecar_shards=len(sidecar_materialization["restored"]),
    )
    run_dir.mkdir(parents=True)
    launch = {
        "schema": "picf-next.molmoact2-m2-launch.v1",
        "gate": M2_GATE,
        "run_dir": str(run_dir),
        "code_revision": code_revision,
        "config": str(args.config.resolve()),
        "config_file_sha256": _sha256(args.config.resolve()),
        "m2_recipe_sha256": recipe.recipe_sha256,
        "foundation_recipe_sha256": foundation.recipe_sha256,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_split_root": str(dataset_split_root),
        "sidecar_artifact_root": str(sidecar_artifact_root),
        "sidecar_materialization": sidecar_materialization,
        "prior_m1": prior_m1,
        "worktree_clean": True,
    }
    _write_json_atomic(run_dir / "launch_manifest.json", launch)
    _write_json_atomic(
        run_dir / "environment.json",
        {
            "schema": "picf-next.molmoact2-m2-environment.v1",
            "resources": resources,
            "python": sys.version,
            "torch": __import__("torch").__version__,
        },
    )
    _write_json_atomic(run_dir / "split_manifest.json", split_report)

    cache_manifest, task_intervention = _extract_feature_cache(
        run_dir=run_dir,
        recipe=recipe,
        foundation=foundation,
        assets=assets,
        checkpoint_dir=checkpoint_dir,
    )
    _write_json_atomic(run_dir / "task_intervention.json", task_intervention)
    cache_manifest, cache = _load_cache(run_dir / "feature_cache", recipe)
    training, evaluation, actual_model = _train_models(
        run_dir=run_dir,
        recipe=recipe,
        foundation=foundation,
        assets=assets,
        cache_manifest=cache_manifest,
        cache=cache,
    )
    _write_json_atomic(run_dir / "training_report.json", training)
    _write_json_atomic(run_dir / "evaluation_report.json", evaluation)
    visuals = _render_visuals(
        run_dir=run_dir,
        model=actual_model,
        assets=assets,
        cache=cache,
        cache_manifest=cache_manifest,
        foundation=foundation,
        recipe=recipe,
    )
    _write_json_atomic(run_dir / "visual_artifacts.json", visuals)
    acceptance = _evaluate_acceptance(
        recipe=recipe,
        evaluation=evaluation,
        task_intervention=task_intervention,
        training=training,
    )
    report_hashes = {relative: _sha256(run_dir / relative) for relative in _M2_MACHINE_REPORTS}
    decision = {
        "schema": "picf-next.molmoact2-m2-machine-decision.v1",
        "gate": M2_GATE,
        "status": acceptance["status"],
        "checks": acceptance["checks"],
        "failed_checks": acceptance["failed_checks"],
        "required_report_sha256": report_hashes,
        "later_gates_authorized": [],
    }
    _write_json_atomic(run_dir / "machine_decision.json", decision)
    validate_m2_machine_decision(run_dir)
    _emit_progress(
        "machine_decision",
        run_dir=str(run_dir),
        status=decision["status"],
        failed_checks=decision["failed_checks"],
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

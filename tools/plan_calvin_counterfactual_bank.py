#!/usr/bin/env python3
"""Plan a task-independent, split-isolated CALVIN object-removal pair bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_counterfactual_plan import (
    CALVIN_COUNTERFACTUAL_PARTITIONS,
    CalvinCounterfactualCandidate,
    CalvinCounterfactualPlanConfig,
    build_calvin_counterfactual_pair_plan,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_simulator_geometry import (
    load_calvin_scene_ranges,
    scene_for_global_index,
)
from picf_next.data.dataset_manifest import load_dataset_file_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-sidecar-root", required=True, type=Path)
    parser.add_argument("--foundation-m2-recipe", required=True, type=Path)
    parser.add_argument("--foundation-m2-recipe-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-pairs-per-identity", type=int, default=4)
    parser.add_argument("--validation-pairs-per-train-identity", type=int, default=2)
    parser.add_argument("--heldout-pairs-per-identity", type=int, default=4)
    parser.add_argument("--heldout-identities-per-family", type=int, default=1)
    parser.add_argument("--minimum-total-visible-pixels", type=int, default=128)
    parser.add_argument("--minimum-same-identity-frame-gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_segments(path: Path, *, expected_sha256: str) -> dict[str, tuple[int, ...]]:
    source = path.resolve()
    if _sha256(source) != expected_sha256:
        raise ContractError("foundation M2 recipe hash changed before pair planning")
    try:
        raw = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("foundation M2 recipe is not valid JSON") from error
    splits = raw.get("splits") if isinstance(raw, dict) else None
    if not isinstance(splits, dict):
        raise ContractError("foundation M2 recipe has no split contract")
    names = {
        "train": "train_segments",
        "validation": "validation_segments",
        "heldout": "heldout_segments",
    }
    output: dict[str, tuple[int, ...]] = {}
    for partition, field in names.items():
        values = splits.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ContractError(f"foundation M2 {field} is invalid")
        output[partition] = tuple(values)
    flattened = [value for values in output.values() for value in values]
    if len(set(flattened)) != len(flattened):
        raise ContractError("foundation M2 split segments overlap")
    return output


def _collect_candidates(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    source_segments: dict[str, tuple[int, ...]],
) -> tuple[CalvinCounterfactualCandidate, ...]:
    if set(source_segments) != set(CALVIN_COUNTERFACTUAL_PARTITIONS):
        raise ContractError("counterfactual source segment contract is incomplete")
    scene_ranges = load_calvin_scene_ranges(
        index.split_root,
        dataset_manifest=index.dataset_manifest,
    )
    segment_partition = {
        segment_index: partition
        for partition, segment_indices in source_segments.items()
        for segment_index in segment_indices
    }
    candidates = []
    seen_frames: dict[int, str] = {}
    for segment_index, partition in sorted(segment_partition.items()):
        if not 0 <= segment_index < len(index.segments):
            raise ContractError("counterfactual plan references an unknown language segment")
        segment = index.segments[segment_index]
        for global_index in range(segment.start, segment.end):
            prior_partition = seen_frames.setdefault(global_index, partition)
            if prior_partition != partition:
                raise ContractError("counterfactual source partitions overlap in frame coordinates")
            physical = sidecar.source_frame(global_index)
            counts: dict[str, dict[int, int]] = {}
            for camera in physical.cameras:
                counts[camera.camera_name] = {
                    owner: int(((camera.owner_index == owner) & camera.owner_supervised).sum())
                    for owner in range(1, len(physical.identity_keys) + 1)
                }
            if set(counts) != {"static", "gripper"}:
                raise ContractError("CALVIN counterfactual planning requires both cameras")
            scene = scene_for_global_index(scene_ranges, global_index)
            for owner, identity_key in enumerate(physical.identity_keys, 1):
                static_pixels = counts["static"][owner]
                gripper_pixels = counts["gripper"][owner]
                if static_pixels + gripper_pixels <= 0:
                    continue
                candidates.append(
                    CalvinCounterfactualCandidate(
                        global_index=global_index,
                        segment_index=segment_index,
                        source_partition=partition,
                        scene=scene,
                        identity_key=identity_key,
                        static_visible_pixels=static_pixels,
                        gripper_visible_pixels=gripper_pixels,
                        task_key=segment.task_key,
                        instruction=segment.instruction,
                    )
                )
    return tuple(candidates)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pair plan: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = _parse_args()
    split_root = args.split_root.resolve()
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=True,
        dataset_manifest=dataset_manifest,
    )
    source_sidecar_root = args.source_sidecar_root.resolve()
    sidecar = CalvinPhysicalSupervisionSidecar(
        source_sidecar_root,
        index,
        verify_hashes=True,
        cache_shards=24,
    )
    source_segments = _source_segments(
        args.foundation_m2_recipe,
        expected_sha256=args.foundation_m2_recipe_sha256,
    )
    candidates = _collect_candidates(
        index=index,
        sidecar=sidecar,
        source_segments=source_segments,
    )
    payload = build_calvin_counterfactual_pair_plan(
        candidates,
        config=CalvinCounterfactualPlanConfig(
            train_pairs_per_identity=args.train_pairs_per_identity,
            validation_pairs_per_train_identity=(args.validation_pairs_per_train_identity),
            heldout_pairs_per_identity=args.heldout_pairs_per_identity,
            heldout_identities_per_family=args.heldout_identities_per_family,
            minimum_total_visible_pixels=args.minimum_total_visible_pixels,
            minimum_same_identity_frame_gap=args.minimum_same_identity_frame_gap,
            seed=args.seed,
        ),
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        source_sidecar_manifest_sha256=_sha256(source_sidecar_root / "manifest.json"),
        foundation_m2_recipe_sha256=args.foundation_m2_recipe_sha256,
        source_segments=source_segments,
    )
    _write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output.resolve()),
                "audit": payload["audit"],
                "identity_partition": payload["identity_partition"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

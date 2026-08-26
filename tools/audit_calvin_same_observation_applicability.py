#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Audit physically true same-observation CALVIN grounding groups."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="CALVIN same-observation applicability audit",
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinLanguageSegment,
)
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_SOURCE_COMMIT,
    calvin_source_state_sha256,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_simulator_geometry import (
    CalvinSceneRange,
    build_calvin_geometry_environment,
    close_calvin_geometry_environment,
    load_calvin_scene_ranges,
    restore_calvin_archived_state,
    scene_for_global_index,
)
from picf_next.data.calvin_task_applicability import (
    CALVIN_OFFICIAL_ANNOTATIONS_SHA256,
    CALVIN_OFFICIAL_TASKS_SHA256,
    CalvinSameObservationGroup,
    build_same_observation_group,
    calvin_state_applicable_tasks,
    calvin_visible_supervised_identity_support,
    extract_calvin_task_applicability_state,
    load_official_calvin_annotations,
    verify_official_calvin_task_config,
)
from picf_next.data.dataset_manifest import (
    file_sha256,
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit

AUDIT_SCHEMA = "picf-next.calvin-same-observation-applicability-audit.v2"
SELECTION_ALGORITHM = "partition-segment-reset-scene-balanced-sha256-coprime.v2"
_REPRESENTATION_PARTITIONS = ("training", "validation", "heldout")
_COLORS = (
    (231, 76, 60),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
)


@dataclass(frozen=True, slots=True)
class _AcceptedFrame:
    scene: str
    group: CalvinSameObservationGroup
    stateful_reset_binding: _StatefulResetBinding
    visible_support: tuple[dict[str, object], ...]
    applicable_tasks: tuple[dict[str, str], ...]
    source_sensor_sha256: tuple[tuple[str, str], ...]

    @property
    def facets(self) -> frozenset[str]:
        variants = self.group.variants
        return frozenset(
            {
                f"scene:{self.scene}",
                *(f"task:{item.task_key}" for item in variants),
                *(f"target:{item.target_identity_key}" for item in variants),
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            **self.group.as_dict(),
            "applicable_tasks": list(self.applicable_tasks),
            "raw_visible_supervised_support": list(self.visible_support),
            "scene": self.scene,
            "source_sensor_sha256": dict(self.source_sensor_sha256),
            "stateful_reset_binding": self.stateful_reset_binding.as_dict(),
            "token_grid_measurability": "pending-host-native-projection",
        }


@dataclass(frozen=True, slots=True)
class _StatefulResetCandidate:
    scene: str
    source_global_index: int
    language_segment_index: int
    source_episode_index: int


@dataclass(frozen=True, slots=True)
class _StatefulResetBinding:
    language_segment_index: int
    source_episode_index: int
    source_instruction_sha256: str
    source_task_key: str
    stateful_episode_key: str
    stateful_sample_key: str
    transition_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "language_segment_index": self.language_segment_index,
            "source_episode_index": self.source_episode_index,
            "source_instruction_sha256": self.source_instruction_sha256,
            "source_task_key": self.source_task_key,
            "stateful_episode_key": self.stateful_episode_key,
            "stateful_sample_key": self.stateful_sample_key,
            "transition_index": self.transition_index,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash_order(*parts: object) -> bytes:
    digest = hashlib.sha256(b"picf-next.calvin-same-observation-audit-order.v1\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _coprime_ordinals(population: int, count: int, *, seed: int, scene: str) -> tuple[int, ...]:
    if (
        isinstance(population, bool)
        or isinstance(count, bool)
        or not isinstance(population, int)
        or not isinstance(count, int)
        or population <= 0
        or not 0 <= count <= population
    ):
        raise ContractError("CALVIN deterministic sample request is invalid")
    if count == 0:
        return ()
    start = int.from_bytes(_hash_order(seed, scene, "start"), "big") % population
    if population == 1:
        return (0,)
    stride = int.from_bytes(_hash_order(seed, scene, "stride"), "big") % population
    stride = stride or 1
    while math.gcd(stride, population) != 1:
        stride = stride + 1 if stride + 1 < population else 1
    return tuple((start + rank * stride) % population for rank in range(count))


def _partition_reset_candidate_inventory(
    segments: tuple[CalvinLanguageSegment, ...],
    scene_ranges: tuple[CalvinSceneRange, ...],
    *,
    admitted_segment_indices: tuple[int, ...],
    admitted_source_episode_indices: tuple[int, ...],
) -> tuple[_StatefulResetCandidate, ...]:
    admitted_segments = frozenset(admitted_segment_indices)
    admitted_sources = frozenset(admitted_source_episode_indices)
    if not admitted_segments or not admitted_sources:
        raise ContractError("CALVIN reset pilot requires a nonempty frozen partition")

    candidates_by_global_index: dict[int, list[_StatefulResetCandidate]] = {}
    observed_segments: set[int] = set()
    observed_sources: set[int] = set()
    for segment in segments:
        segment_index = int(segment.index)
        source_episode_index = int(segment.episode_index)
        if segment_index not in admitted_segments:
            continue
        if source_episode_index not in admitted_sources:
            raise ContractError("frozen partition segment belongs to another source episode")
        observed_segments.add(segment_index)
        observed_sources.add(source_episode_index)
        scene = scene_for_global_index(scene_ranges, int(segment.start))
        candidate = _StatefulResetCandidate(
            scene=scene,
            source_global_index=int(segment.start),
            language_segment_index=segment_index,
            source_episode_index=source_episode_index,
        )
        candidates_by_global_index.setdefault(candidate.source_global_index, []).append(candidate)
    if observed_segments != admitted_segments or observed_sources != admitted_sources:
        raise ContractError("frozen partition coordinates differ from CALVIN metadata")

    # Overlapping language annotations can share a source reset frame. One
    # observation must map to one immutable stream address, so choose the
    # lowest segment index without consulting targets or pixels.
    return tuple(
        min(values, key=lambda item: item.language_segment_index)
        for _global_index, values in sorted(candidates_by_global_index.items())
    )


def stratified_partition_reset_candidates(
    segments: tuple[CalvinLanguageSegment, ...],
    scene_ranges: tuple[CalvinSceneRange, ...],
    *,
    admitted_segment_indices: tuple[int, ...],
    admitted_source_episode_indices: tuple[int, ...],
    sample_count: int,
    seed: int,
) -> tuple[_StatefulResetCandidate, ...]:
    """Select reset observations already admitted by one frozen split partition."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ContractError("CALVIN reset pilot sample count and seed are invalid")
    candidates = _partition_reset_candidate_inventory(
        segments,
        scene_ranges,
        admitted_segment_indices=admitted_segment_indices,
        admitted_source_episode_indices=admitted_source_episode_indices,
    )
    by_scene: dict[str, tuple[_StatefulResetCandidate, ...]] = {}
    for scene_range in scene_ranges:
        values = tuple(item for item in candidates if item.scene == scene_range.scene)
        if values:
            by_scene[scene_range.scene] = values
    if not by_scene or sample_count > len(candidates):
        raise ContractError("CALVIN reset pilot request exceeds frozen partition coverage")

    base = sample_count // len(by_scene)
    allocation = {scene: min(base, len(values)) for scene, values in by_scene.items()}
    remaining = sample_count - sum(allocation.values())
    while remaining:
        available = tuple(
            scene for scene, values in by_scene.items() if allocation[scene] < len(values)
        )
        if not available:
            raise RuntimeError("CALVIN reset pilot allocation lost available candidates")
        winner = min(
            available,
            key=lambda scene: (
                allocation[scene],
                _hash_order(seed, scene, allocation[scene], "remainder"),
                scene,
            ),
        )
        allocation[winner] += 1
        remaining -= 1

    selected: list[_StatefulResetCandidate] = []
    for scene, values in sorted(by_scene.items()):
        requested = allocation[scene]
        selected.extend(
            values[ordinal]
            for ordinal in _coprime_ordinals(
                len(values),
                requested,
                seed=seed,
                scene=scene,
            )
        )
    if len(selected) != sample_count or len({item.source_global_index for item in selected}) != len(
        selected
    ):
        raise RuntimeError("CALVIN reset pilot lost or duplicated a source observation")
    return tuple(sorted(selected, key=lambda item: item.source_global_index))


def _verify_source_binding(
    frame_arrays: dict[str, np.ndarray],
    physical: CalvinPhysicalSupervisionFrame,
) -> tuple[tuple[str, str], ...]:
    specs = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    if set(specs) != {camera.camera_name for camera in physical.cameras}:
        raise ContractError("CALVIN physical and source camera inventories differ")
    hashes = []
    for camera in physical.cameras:
        spec = specs[camera.camera_name]
        rgb_field = str(spec["source_rgb_field"])
        depth_field = str(spec["source_depth_field"])
        rgb_digest = source_array_sha256(rgb_field, frame_arrays[rgb_field])
        depth_digest = source_array_sha256(depth_field, frame_arrays[depth_field])
        if rgb_digest != camera.source_rgb_sha256 or depth_digest != camera.source_depth_sha256:
            raise ContractError("CALVIN physical owner labels are bound to another source frame")
        hashes.extend(((rgb_field, rgb_digest), (depth_field, depth_digest)))
    return tuple(sorted(hashes))


def _support_dicts(
    physical: CalvinPhysicalSupervisionFrame,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "camera_pixel_counts": dict(item.camera_pixel_counts),
            "identity_key": item.identity_key,
            "total_pixel_count": item.total_pixel_count,
        }
        for item in calvin_visible_supervised_identity_support(physical)
    )


def _selected_target_support(
    record: _AcceptedFrame,
) -> tuple[dict[str, object], ...]:
    targets = {item.target_identity_key for item in record.group.variants}
    selected = tuple(item for item in record.visible_support if item["identity_key"] in targets)
    if {str(item["identity_key"]) for item in selected} != targets:
        raise RuntimeError("accepted CALVIN group lost visible target support")
    return selected


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_group_visual(
    *,
    record: _AcceptedFrame,
    frame_arrays: dict[str, np.ndarray],
    physical: CalvinPhysicalSupervisionFrame,
) -> bytes:
    """Render one task-labelled, two-camera owner audit sheet."""

    target_order = tuple(item.target_identity_key for item in record.group.variants)
    target_color = {
        target: _COLORS[index % len(_COLORS)] for index, target in enumerate(target_order)
    }
    spec_by_camera = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    camera_panels = []
    for camera in physical.cameras:
        spec = spec_by_camera[camera.camera_name]
        rgb_field = str(spec["source_rgb_field"])
        image = Image.fromarray(frame_arrays[rgb_field]).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for target in target_order:
            owner = physical.identity_keys.index(target) + 1
            mask_array = (camera.owner_index == owner) & camera.owner_supervised
            if not mask_array.any():
                continue
            mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
            color = target_color[target]
            fill = Image.new("RGBA", image.size, (*color, 96))
            overlay.alpha_composite(Image.composite(fill, Image.new("RGBA", image.size), mask))
            bounds = mask.getbbox()
            if bounds is not None:
                overlay_draw.rectangle(bounds, outline=(*color, 255), width=2)
        image = Image.alpha_composite(image, overlay).convert("RGB")
        target_height = 420
        scale = target_height / image.height
        panel = image.resize(
            (round(image.width * scale), target_height),
            resample=Image.Resampling.NEAREST,
        )
        labelled = Image.new("RGB", (panel.width, target_height + 30), "white")
        labelled.paste(panel, (0, 30))
        ImageDraw.Draw(labelled).text(
            (8, 6),
            camera.camera_name,
            fill="black",
            font=_font(16),
        )
        camera_panels.append(labelled)

    text_lines = [
        (
            f"step={record.group.source_global_index} scene={record.scene} "
            f"source_episode={record.stateful_reset_binding.source_episode_index} "
            f"segment={record.stateful_reset_binding.language_segment_index} "
            f"transition=0 state={record.group.source_state_sha256[:12]}"
        )
    ]
    support_by_target: dict[str, int] = {}
    for item in _selected_target_support(record):
        identity_key = item["identity_key"]
        total_pixel_count = item["total_pixel_count"]
        if (
            not isinstance(identity_key, str)
            or isinstance(total_pixel_count, bool)
            or not isinstance(total_pixel_count, int)
        ):
            raise RuntimeError("accepted CALVIN target support changed type")
        support_by_target[identity_key] = total_pixel_count
    for rank, variant in enumerate(record.group.variants):
        text_lines.append(
            f"{rank + 1}. {variant.task_key} -> {variant.target_identity_key} "
            f"pixels={support_by_target[variant.target_identity_key]} | {variant.instruction}"
        )
        text_lines.append(f"   proof={variant.proof}")
    wrapped = [
        line
        for raw in text_lines
        for line in textwrap.wrap(raw, width=105, subsequent_indent="   ") or [""]
    ]
    text_height = 14 + len(wrapped) * 21
    width = sum(panel.width for panel in camera_panels)
    height = max(panel.height for panel in camera_panels) + text_height
    canvas = Image.new("RGB", (width, height), "white")
    cursor = 0
    for panel in camera_panels:
        canvas.paste(panel, (cursor, 0))
        cursor += panel.width
    draw = ImageDraw.Draw(canvas)
    y = max(panel.height for panel in camera_panels) + 7
    for line in wrapped:
        draw.text((8, y), line, fill="black", font=_font(15))
        y += 21
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _select_visual_groups(
    records: tuple[_AcceptedFrame, ...],
    limit: int,
) -> tuple[_AcceptedFrame, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ContractError("CALVIN visual audit count must be non-negative")
    remaining = list(records)
    selected = []
    covered: set[str] = set()
    while remaining and len(selected) < limit:
        winner = min(
            remaining,
            key=lambda item: (
                -len(item.facets - covered),
                _hash_order(item.group.source_state_sha256, "visual"),
                item.group.source_global_index,
            ),
        )
        selected.append(winner)
        covered.update(winner.facets)
        remaining.remove(winner)
    return tuple(selected)


def _histogram(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _representation_partition_coordinates(
    representation_split: RepresentationTrialSplit,
    partition: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if partition == "training":
        return (
            representation_split.training_segment_indices,
            representation_split.training_source_episode_indices,
        )
    if partition == "validation":
        records = representation_split.validation_segments
    elif partition == "heldout":
        records = representation_split.heldout_segments
    else:
        raise ContractError("unknown representation split partition")
    return (
        tuple(sorted(item.segment_index for item in records)),
        tuple(sorted({item.source_episode_index for item in records})),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--representation-split", required=True, type=Path)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument(
        "--representation-partition",
        required=True,
        choices=_REPRESENTATION_PARTITIONS,
    )
    parser.add_argument("--physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--official-annotations", required=True, type=Path)
    parser.add_argument("--official-task-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=20260731)
    parser.add_argument("--maximum-variants", type=int, default=4)
    parser.add_argument("--visual-count", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--global-index", action="append", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.visual_output_dir is not None and (
        args.visual_output_dir.exists() or args.visual_output_dir.is_symlink()
    ):
        raise FileExistsError(args.visual_output_dir)
    if (
        args.maximum_variants < 2
        or args.visual_count < 0
        or args.progress_every <= 0
        or args.sample_count <= 0
        or args.selection_seed < 0
    ):
        raise ContractError("CALVIN audit numeric arguments are invalid")

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    split_root = args.split_root.resolve()
    runtime_binding = validate_dataset_runtime_binding(
        manifest,
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=split_root.name,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    scene_ranges = load_calvin_scene_ranges(split_root, dataset_manifest=manifest)
    representation_split_path = args.representation_split.resolve()
    representation_split_file_sha256 = file_sha256(representation_split_path)
    if representation_split_file_sha256 != args.representation_split_sha256:
        raise ContractError("representation split differs from its expected SHA-256")
    try:
        representation_split = RepresentationTrialSplit.load(representation_split_path)
    except ValueError as error:
        raise ContractError("representation split failed its content contract") from error
    if (
        representation_split.dataset_id != manifest.dataset_id
        or representation_split.dataset_revision != manifest.dataset_revision
        or representation_split.dataset_manifest_sha256 != manifest.tree_sha256
    ):
        raise ContractError("representation split belongs to another CALVIN dataset")
    (
        partition_segment_indices,
        partition_source_episode_indices,
    ) = _representation_partition_coordinates(
        representation_split,
        args.representation_partition,
    )
    candidate_inventory = _partition_reset_candidate_inventory(
        index.segments,
        scene_ranges,
        admitted_segment_indices=partition_segment_indices,
        admitted_source_episode_indices=partition_source_episode_indices,
    )
    candidate_by_global_index = {item.source_global_index: item for item in candidate_inventory}
    if args.global_index:
        indices = tuple(sorted(set(args.global_index)))
        if len(indices) != len(args.global_index):
            raise ContractError("explicit CALVIN audit indices must be unique")
        if any(global_index not in candidate_by_global_index for global_index in indices):
            raise ContractError(
                "explicit CALVIN audit index is not a frozen partition-segment reset"
            )
        selected_candidates = tuple(candidate_by_global_index[value] for value in indices)
        selection = {
            "algorithm": "explicit-frozen-partition-segment-resets.v2",
            "candidate_count": len(candidate_inventory),
            "representation_partition": args.representation_partition,
            "sample_count": len(indices),
            "seed": None,
        }
    else:
        selected_candidates = stratified_partition_reset_candidates(
            index.segments,
            scene_ranges,
            admitted_segment_indices=partition_segment_indices,
            admitted_source_episode_indices=partition_source_episode_indices,
            sample_count=args.sample_count,
            seed=args.selection_seed,
        )
        indices = tuple(item.source_global_index for item in selected_candidates)
        selection = {
            "algorithm": SELECTION_ALGORITHM,
            "candidate_count": len(candidate_inventory),
            "representation_partition": args.representation_partition,
            "sample_count": len(indices),
            "seed": args.selection_seed,
        }
    selected_by_global_index = {item.source_global_index: item for item in selected_candidates}

    annotations = load_official_calvin_annotations(args.official_annotations)
    verify_official_calvin_task_config(args.official_task_config)
    sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root, index)
    required_fields = tuple(
        sorted(
            {
                "robot_obs",
                "scene_obs",
                *(str(spec["source_rgb_field"]) for spec in CALVIN_CAMERA_SPECS),
                *(str(spec["source_depth_field"]) for spec in CALVIN_CAMERA_SPECS),
            }
        )
    )
    segments_by_index = {int(segment.index): segment for segment in index.segments}
    if args.representation_partition == "validation":
        frozen_partition_records = representation_split.validation_segments
    elif args.representation_partition == "heldout":
        frozen_partition_records = representation_split.heldout_segments
    else:
        frozen_partition_records = ()
    for frozen in frozen_partition_records:
        observed = segments_by_index.get(frozen.segment_index)
        if observed is None or (
            observed.task_key,
            int(observed.episode_index),
            int(observed.start),
            int(observed.end),
        ) != (
            frozen.task_key,
            frozen.source_episode_index,
            frozen.source_start,
            frozen.source_end,
        ):
            raise ContractError("representation evaluation partition differs from CALVIN metadata")
    admitted_source_episodes = frozenset(partition_source_episode_indices)
    all_partition_source_episodes = frozenset(
        {
            *representation_split.training_source_episode_indices,
            *representation_split.evaluation_source_episode_indices,
        }
    )
    excluded_source_episodes = all_partition_source_episodes - admitted_source_episodes
    accepted: list[_AcceptedFrame] = []
    rejected = []
    for scene in sorted(
        {
            scene_for_global_index(scene_ranges, source_global_index)
            for source_global_index in indices
        }
    ):
        environment = build_calvin_geometry_environment(
            args.calvin_env_root,
            scene=scene,
            include_cameras=False,
        )
        try:
            scene_indices = tuple(
                global_index
                for global_index in indices
                if scene_for_global_index(scene_ranges, global_index) == scene
            )
            for scene_rank, global_index in enumerate(scene_indices, start=1):
                candidate = selected_by_global_index[global_index]
                segment = segments_by_index[candidate.language_segment_index]
                if (
                    int(segment.start) != global_index
                    or int(segment.episode_index) != candidate.source_episode_index
                    or candidate.source_episode_index not in admitted_source_episodes
                    or candidate.source_episode_index in excluded_source_episodes
                ):
                    raise ContractError(
                        "selected CALVIN reset is not isolated to its split partition"
                    )
                stateful_sample = index.stateful_transition_sample(
                    candidate.language_segment_index,
                    global_index,
                    action_horizon=1,
                )
                if (
                    int(stateful_sample.transition_index) != 0
                    or stateful_sample.host_sample.task_key != segment.task_key
                    or stateful_sample.record.task != segment.instruction
                ):
                    raise ContractError(
                        "selected CALVIN reset differs from the stateful training contract"
                    )
                stateful_reset_binding = _StatefulResetBinding(
                    language_segment_index=candidate.language_segment_index,
                    source_episode_index=candidate.source_episode_index,
                    source_instruction_sha256=hashlib.sha256(
                        stateful_sample.record.task.encode("utf-8")
                    ).hexdigest(),
                    source_task_key=stateful_sample.host_sample.task_key,
                    stateful_episode_key=stateful_sample.episode_key,
                    stateful_sample_key=stateful_sample.sample_key,
                    transition_index=int(stateful_sample.transition_index),
                )
                arrays = dict(
                    index.validated_source_frame_arrays(
                        global_index,
                        fields=required_fields,
                    )
                )
                state_sha256 = calvin_source_state_sha256(
                    arrays["scene_obs"],
                    arrays["robot_obs"],
                )
                if sidecar.source_state_sha256(global_index) != state_sha256:
                    raise ContractError("CALVIN physical labels are bound to another source state")
                physical = sidecar.source_frame(global_index)
                sensor_hashes = _verify_source_binding(arrays, physical)
                restore_calvin_archived_state(
                    environment,
                    scene_obs=arrays["scene_obs"],
                    robot_obs=arrays["robot_obs"],
                )
                applicable = calvin_state_applicable_tasks(
                    extract_calvin_task_applicability_state(environment)
                )
                visible_support = _support_dicts(physical)
                group = build_same_observation_group(
                    source_global_index=global_index,
                    source_state_sha256=state_sha256,
                    visible_identity_keys=tuple(
                        str(item["identity_key"]) for item in visible_support
                    ),
                    applicable_tasks=applicable,
                    annotations=annotations,
                    maximum_variants=args.maximum_variants,
                )
                applicable_dicts = tuple(
                    {
                        "proof": item.proof,
                        "target_identity_key": item.target_identity_key,
                        "task_key": item.task_key,
                    }
                    for item in applicable
                )
                if group is None:
                    visible_targets = {
                        item.target_identity_key
                        for item in applicable
                        if item.target_identity_key
                        in {str(value["identity_key"]) for value in visible_support}
                    }
                    rejected.append(
                        {
                            "applicable_task_count": len(applicable),
                            "reason": "fewer-than-two-distinct-raw-visible-applicable-targets",
                            "scene": scene,
                            "source_global_index": global_index,
                            "source_state_sha256": state_sha256,
                            "stateful_reset_binding": stateful_reset_binding.as_dict(),
                            "visible_applicable_target_count": len(visible_targets),
                        }
                    )
                else:
                    accepted.append(
                        _AcceptedFrame(
                            scene=scene,
                            group=group,
                            stateful_reset_binding=stateful_reset_binding,
                            visible_support=visible_support,
                            applicable_tasks=applicable_dicts,
                            source_sensor_sha256=sensor_hashes,
                        )
                    )
                if scene_rank % args.progress_every == 0 or scene_rank == len(scene_indices):
                    print(
                        json.dumps(
                            {
                                "accepted": len(accepted),
                                "processed": len(accepted) + len(rejected),
                                "scene": scene,
                                "scene_processed": scene_rank,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            close_calvin_geometry_environment(environment)

    accepted_tuple = tuple(sorted(accepted, key=lambda item: item.group.source_global_index))
    visual_artifacts = []
    visual_records = _select_visual_groups(accepted_tuple, args.visual_count)
    if visual_records:
        if args.visual_output_dir is None:
            raise ContractError("visual audit records require --visual-output-dir")
        args.visual_output_dir.mkdir(parents=True, exist_ok=False)
        for record in visual_records:
            global_index = record.group.source_global_index
            arrays = dict(
                index.validated_source_frame_arrays(
                    global_index,
                    fields=required_fields,
                )
            )
            physical = sidecar.source_frame(global_index)
            png = render_group_visual(
                record=record,
                frame_arrays=arrays,
                physical=physical,
            )
            tasks = "__".join(item.task_key for item in record.group.variants)
            name = f"step_{global_index:07d}__{tasks}.png"
            destination = args.visual_output_dir / name
            write_bytes_durable_exclusive(destination, png)
            visual_artifacts.append(
                {
                    "file": name,
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "scene": record.scene,
                    "source_global_index": global_index,
                    "task_keys": [item.task_key for item in record.group.variants],
                }
            )

    task_values = [
        variant.task_key for record in accepted_tuple for variant in record.group.variants
    ]
    target_values = [
        variant.target_identity_key
        for record in accepted_tuple
        for variant in record.group.variants
    ]
    report_content: dict[str, Any] = {
        "acceptance_scope": {
            "raw_owner_visibility_proven": True,
            "representation_partition_isolation_proven": True,
            "source_state_and_sensor_hash_binding_proven": True,
            "stateful_reset_addressability_proven": True,
            "token_grid_measurability_proven": False,
            "training_authorized": False,
        },
        "accepted_group_count": len(accepted_tuple),
        "accepted_groups": [record.as_dict() for record in accepted_tuple],
        "calvin_env_source_commit": CALVIN_ENV_SOURCE_COMMIT,
        "calvin_source_commit": CALVIN_SOURCE_COMMIT,
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_manifest_file_sha256": file_sha256(args.dataset_manifest),
            "dataset_revision": manifest.dataset_revision,
            "dataset_tree_sha256": manifest.tree_sha256,
            "runtime_binding": runtime_binding,
            "split_name": manifest.split_name,
        },
        "leakage_contract": {
            "model_input_contains_applicability_proof": False,
            "model_input_contains_identity_or_owner": False,
            "model_input_contains_representation_split_metadata": False,
            "model_input_contains_simulator_state": False,
            "model_input_contains_stateful_binding": False,
            "model_input_contains_task_key": False,
            "model_input_contains_target": False,
            "model_input_contains_complete_natural_instruction": True,
        },
        "official_annotations_sha256": CALVIN_OFFICIAL_ANNOTATIONS_SHA256,
        "official_task_config_sha256": CALVIN_OFFICIAL_TASKS_SHA256,
        "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
        "representation_split": {
            "artifact_sha256": representation_split.artifact_sha256,
            "comparison_id": representation_split.comparison_id,
            "file_sha256": representation_split_file_sha256,
            "partition": args.representation_partition,
            "partition_segment_count": len(partition_segment_indices),
            "partition_source_episode_count": len(partition_source_episode_indices),
            "schema": representation_split.schema,
            "stream_plan_sha256": representation_split.stream_plan_sha256,
        },
        "rejected_frame_count": len(rejected),
        "rejected_frames": sorted(rejected, key=lambda item: item["source_global_index"]),
        "schema": AUDIT_SCHEMA,
        "selection": selection,
        "summary": {
            "accepted_fraction": (len(accepted_tuple) / len(indices) if indices else 0.0),
            "scene_histogram": _histogram([record.scene for record in accepted_tuple]),
            "source_episode_histogram": _histogram(
                [
                    str(record.stateful_reset_binding.source_episode_index)
                    for record in accepted_tuple
                ]
            ),
            "target_histogram": _histogram(target_values),
            "task_histogram": _histogram(task_values),
            "unique_source_state_count": len(
                {record.group.source_state_sha256 for record in accepted_tuple}
            ),
        },
        "visual_artifacts": visual_artifacts,
    }
    artifact_sha256 = hashlib.sha256(_canonical_json_bytes(report_content)).hexdigest()
    report = {**report_content, "artifact_sha256": artifact_sha256}
    payload = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    write_bytes_durable_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "accepted_group_count": len(accepted_tuple),
                "artifact_sha256": artifact_sha256,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "output": str(args.output.resolve()),
                "rejected_frame_count": len(rejected),
                "representation_partition": args.representation_partition,
                "task_histogram": _histogram(task_values),
                "visual_count": len(visual_artifacts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

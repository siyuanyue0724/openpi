#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Gate same-observation CALVIN variants on the exact Qwen token grid."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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
    entrypoint_name="CALVIN same-observation token-grid audit",
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_SOURCE_COMMIT,
)
from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_task_applicability import (
    CALVIN_OFFICIAL_ANNOTATIONS_SHA256,
    CALVIN_OFFICIAL_TASKS_SHA256,
    CalvinSameObservationGroup,
    CalvinSameObservationVariant,
)
from picf_next.data.calvin_token_grid_support import (
    CalvinTokenGridIdentitySupport,
    project_calvin_token_grid_identity_support,
)
from picf_next.data.dataset_manifest import (
    file_sha256,
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.data.lingbot_calvin_projection import (
    load_lingbot_calvin_projection_contract,
    projection_payload_sha256,
)
from picf_next.data.qwen3vl_raster import project_qwen3vl_segmentation
from picf_next.data.raster_targets import regular_grid_pixel_boxes
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
)
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit

APPLICABILITY_AUDIT_SCHEMA = "picf-next.calvin-same-observation-applicability-audit.v2"
TOKEN_GRID_AUDIT_SCHEMA = "picf-next.calvin-same-observation-token-grid-audit.v3"
_MAXIMUM_REPORT_BYTES = 32 * 1024 * 1024
_SHA256_LENGTH = 64
_EXPECTED_LEAKAGE_CONTRACT = {
    "model_input_contains_applicability_proof": False,
    "model_input_contains_complete_natural_instruction": True,
    "model_input_contains_identity_or_owner": False,
    "model_input_contains_representation_split_metadata": False,
    "model_input_contains_simulator_state": False,
    "model_input_contains_stateful_binding": False,
    "model_input_contains_target": False,
    "model_input_contains_task_key": False,
}
_EXPECTED_APPLICABILITY_SCOPE = {
    "raw_owner_visibility_proven": True,
    "representation_partition_isolation_proven": True,
    "source_state_and_sensor_hash_binding_proven": True,
    "stateful_reset_addressability_proven": True,
    "token_grid_measurability_proven": False,
    "training_authorized": False,
}
_REPORT_FIELDS = {
    "acceptance_scope",
    "accepted_group_count",
    "accepted_groups",
    "artifact_sha256",
    "calvin_env_source_commit",
    "calvin_source_commit",
    "dataset",
    "leakage_contract",
    "official_annotations_sha256",
    "official_task_config_sha256",
    "physical_sidecar_manifest_sha256",
    "representation_split",
    "rejected_frame_count",
    "rejected_frames",
    "schema",
    "selection",
    "summary",
    "visual_artifacts",
}
_GROUP_FIELDS = {
    "applicable_tasks",
    "model_input_contains_simulator_state_or_identity",
    "raw_visible_supervised_support",
    "scene",
    "schema",
    "source_global_index",
    "source_sensor_sha256",
    "source_state_sha256",
    "stateful_reset_binding",
    "token_grid_measurability",
    "variants",
}
_STATEFUL_RESET_BINDING_FIELDS = {
    "language_segment_index",
    "source_episode_index",
    "source_instruction_sha256",
    "source_task_key",
    "stateful_episode_key",
    "stateful_sample_key",
    "transition_index",
}
_REPRESENTATION_SPLIT_BINDING_FIELDS = {
    "artifact_sha256",
    "comparison_id",
    "file_sha256",
    "partition",
    "partition_segment_count",
    "partition_source_episode_count",
    "schema",
    "stream_plan_sha256",
}
_REPRESENTATION_PARTITIONS = ("training", "validation", "heldout")
_VARIANT_FIELDS = {
    "instruction",
    "instruction_sha256",
    "proof",
    "target_identity_key",
    "task_key",
}


@dataclass(frozen=True, slots=True)
class _InputGroup:
    scene: str
    group: CalvinSameObservationGroup
    source_sensor_sha256: tuple[tuple[str, str], ...]
    stateful_reset_binding: _InputResetBinding


@dataclass(frozen=True, slots=True)
class _InputResetBinding:
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


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256")
    return value


def _bound_json(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    expected = _sha256(expected_sha256, name="applicability report expected SHA-256")
    resolved = path.expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise ContractError("applicability report must be one real file")
    if resolved.stat().st_size > _MAXIMUM_REPORT_BYTES:
        raise ContractError("applicability report exceeds the maximum size")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError("applicability report differs from its expected SHA-256")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("applicability report is not valid JSON") from error
    if not isinstance(decoded, dict) or set(decoded) != _REPORT_FIELDS:
        raise ContractError("applicability report fields differ from the frozen schema")
    artifact_sha256 = _sha256(
        decoded["artifact_sha256"],
        name="applicability artifact SHA-256",
    )
    content = {key: value for key, value in decoded.items() if key != "artifact_sha256"}
    if hashlib.sha256(_canonical_json_bytes(content)).hexdigest() != artifact_sha256:
        raise ContractError("applicability artifact digest is invalid")
    return decoded


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be non-empty text")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _parse_groups(report: dict[str, Any]) -> tuple[_InputGroup, ...]:
    groups = report["accepted_groups"]
    count = report["accepted_group_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(groups, list)
        or len(groups) != count
    ):
        raise ContractError("applicability accepted-group count is invalid")
    output = []
    seen_indices: set[int] = set()
    seen_states: set[str] = set()
    for index, raw in enumerate(groups):
        if not isinstance(raw, dict) or set(raw) != _GROUP_FIELDS:
            raise ContractError(f"applicability group {index} must be a mapping")
        source_global_index = _nonnegative_int(
            raw.get("source_global_index"),
            name=f"applicability group {index} source index",
        )
        source_state_sha256 = _sha256(
            raw.get("source_state_sha256"),
            name=f"applicability group {index} source-state SHA-256",
        )
        if source_global_index in seen_indices or source_state_sha256 in seen_states:
            raise ContractError("applicability groups repeat a source observation")
        if (
            raw.get("schema") != "picf-next.calvin-task-applicability.v1"
            or raw.get("model_input_contains_simulator_state_or_identity") is not False
            or raw.get("token_grid_measurability") != "pending-host-native-projection"
        ):
            raise ContractError("applicability group contract changed")
        variants = raw.get("variants")
        if not isinstance(variants, list):
            raise ContractError("applicability group variants must be a list")
        parsed_variants = []
        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, dict) or set(variant) != _VARIANT_FIELDS:
                raise ContractError("applicability variant must be a mapping")
            parsed_variants.append(
                CalvinSameObservationVariant(
                    task_key=_text(
                        variant.get("task_key"),
                        name=f"applicability variant {variant_index} task",
                    ),
                    instruction=_text(
                        variant.get("instruction"),
                        name=f"applicability variant {variant_index} instruction",
                    ),
                    instruction_sha256=_sha256(
                        variant.get("instruction_sha256"),
                        name=f"applicability variant {variant_index} instruction SHA-256",
                    ),
                    target_identity_key=_text(
                        variant.get("target_identity_key"),
                        name=f"applicability variant {variant_index} target",
                    ),
                    proof=_text(
                        variant.get("proof"),
                        name=f"applicability variant {variant_index} proof",
                    ),
                )
            )
        group = CalvinSameObservationGroup(
            source_global_index=source_global_index,
            source_state_sha256=source_state_sha256,
            variants=tuple(parsed_variants),
        )
        raw_reset = raw.get("stateful_reset_binding")
        if not isinstance(raw_reset, dict) or set(raw_reset) != _STATEFUL_RESET_BINDING_FIELDS:
            raise ContractError("applicability stateful-reset binding changed")
        reset = _InputResetBinding(
            language_segment_index=_nonnegative_int(
                raw_reset.get("language_segment_index"),
                name=f"applicability group {index} language segment",
            ),
            source_episode_index=_nonnegative_int(
                raw_reset.get("source_episode_index"),
                name=f"applicability group {index} source episode",
            ),
            source_instruction_sha256=_sha256(
                raw_reset.get("source_instruction_sha256"),
                name=f"applicability group {index} source instruction SHA-256",
            ),
            source_task_key=_text(
                raw_reset.get("source_task_key"),
                name=f"applicability group {index} source task",
            ),
            stateful_episode_key=_text(
                raw_reset.get("stateful_episode_key"),
                name=f"applicability group {index} stateful episode",
            ),
            stateful_sample_key=_text(
                raw_reset.get("stateful_sample_key"),
                name=f"applicability group {index} stateful sample",
            ),
            transition_index=_nonnegative_int(
                raw_reset.get("transition_index"),
                name=f"applicability group {index} transition",
            ),
        )
        expected_episode_key = f"calvin-language-segment-{reset.language_segment_index:08d}"
        expected_sample_key = (
            f"{expected_episode_key}/transition-00000000-frame-{source_global_index:08d}"
        )
        if (
            reset.transition_index != 0
            or reset.stateful_episode_key != expected_episode_key
            or reset.stateful_sample_key != expected_sample_key
        ):
            raise ContractError("applicability group is not an exact stateful reset")
        sensor_hashes = raw.get("source_sensor_sha256")
        expected_sensor_fields = {str(spec["source_rgb_field"]) for spec in CALVIN_CAMERA_SPECS} | {
            str(spec["source_depth_field"]) for spec in CALVIN_CAMERA_SPECS
        }
        if not isinstance(sensor_hashes, dict) or set(sensor_hashes) != expected_sensor_fields:
            raise ContractError("applicability source-sensor hashes changed")
        normalized_hashes = tuple(
            sorted(
                (
                    field,
                    _sha256(
                        sensor_hashes[field],
                        name=f"applicability {field} SHA-256",
                    ),
                )
                for field in expected_sensor_fields
            )
        )
        scene = _text(raw.get("scene"), name=f"applicability group {index} scene")
        seen_indices.add(source_global_index)
        seen_states.add(source_state_sha256)
        output.append(
            _InputGroup(
                scene=scene,
                group=group,
                source_sensor_sha256=normalized_hashes,
                stateful_reset_binding=reset,
            )
        )
    if tuple(value.group.source_global_index for value in output) != tuple(
        sorted(value.group.source_global_index for value in output)
    ):
        raise ContractError("applicability groups must be source-index sorted")
    return tuple(output)


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


def _validate_report_identity(
    report: dict[str, Any],
    *,
    dataset_manifest_sha256: str,
    sidecar_manifest_sha256: str,
    representation_split: RepresentationTrialSplit,
    representation_split_file_sha256: str,
    expected_representation_partition: str,
) -> tuple[_InputGroup, ...]:
    if (
        report["schema"] != APPLICABILITY_AUDIT_SCHEMA
        or report["calvin_source_commit"] != CALVIN_SOURCE_COMMIT
        or report["calvin_env_source_commit"] != CALVIN_ENV_SOURCE_COMMIT
        or report["official_annotations_sha256"] != CALVIN_OFFICIAL_ANNOTATIONS_SHA256
        or report["official_task_config_sha256"] != CALVIN_OFFICIAL_TASKS_SHA256
        or report["physical_sidecar_manifest_sha256"] != sidecar_manifest_sha256
        or report["leakage_contract"] != _EXPECTED_LEAKAGE_CONTRACT
        or report["acceptance_scope"] != _EXPECTED_APPLICABILITY_SCOPE
    ):
        raise ContractError("applicability report did not pass its frozen identity contract")
    dataset = report["dataset"]
    if (
        not isinstance(dataset, dict)
        or dataset.get("dataset_manifest_file_sha256") != dataset_manifest_sha256
        or dataset.get("split_name") != "training"
    ):
        raise ContractError("applicability report belongs to another dataset")
    reported_split = report["representation_split"]
    if (
        not isinstance(reported_split, dict)
        or set(reported_split) != _REPRESENTATION_SPLIT_BINDING_FIELDS
        or reported_split["artifact_sha256"] != representation_split.artifact_sha256
        or reported_split["comparison_id"] != representation_split.comparison_id
        or reported_split["file_sha256"] != representation_split_file_sha256
        or reported_split["partition"] != expected_representation_partition
        or reported_split["schema"] != representation_split.schema
        or reported_split["stream_plan_sha256"] != representation_split.stream_plan_sha256
    ):
        raise ContractError("applicability report belongs to another representation split")
    partition_segments, partition_sources = _representation_partition_coordinates(
        representation_split,
        expected_representation_partition,
    )
    if reported_split["partition_segment_count"] != len(partition_segments) or reported_split[
        "partition_source_episode_count"
    ] != len(partition_sources):
        raise ContractError("applicability representation partition cardinality changed")
    groups = _parse_groups(report)
    admitted_segments = frozenset(partition_segments)
    admitted_sources = frozenset(partition_sources)
    all_sources = frozenset(
        {
            *representation_split.training_source_episode_indices,
            *representation_split.evaluation_source_episode_indices,
        }
    )
    excluded_sources = all_sources - admitted_sources
    for group in groups:
        reset = group.stateful_reset_binding
        if (
            reset.language_segment_index not in admitted_segments
            or reset.source_episode_index not in admitted_sources
            or reset.source_episode_index in excluded_sources
        ):
            raise ContractError(
                "applicability reset is outside its source-disjoint split partition"
            )
    return groups


def _verify_group_source(
    *,
    source: _InputGroup,
    frame: CalvinPhysicalSupervisionFrame,
    sidecar: CalvinPhysicalSupervisionSidecar,
) -> None:
    global_index = source.group.source_global_index
    if sidecar.source_state_sha256(global_index) != source.group.source_state_sha256:
        raise ContractError("token-grid frame differs from the applicability source state")
    reported = dict(source.source_sensor_sha256)
    cameras = {camera.camera_name: camera for camera in frame.cameras}
    for spec in CALVIN_CAMERA_SPECS:
        camera_name = str(spec["camera_name"])
        camera = cameras.get(camera_name)
        if camera is None:
            raise ContractError("token-grid frame omitted one physical camera")
        if (
            reported[str(spec["source_rgb_field"])] != camera.source_rgb_sha256
            or reported[str(spec["source_depth_field"])] != camera.source_depth_sha256
        ):
            raise ContractError("token-grid frame differs from the applicability sensors")
    missing = {variant.target_identity_key for variant in source.group.variants}.difference(
        frame.identity_keys
    )
    if missing:
        raise ContractError("applicability target is absent from the physical inventory")


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _target_overlay(
    *,
    rgb: np.ndarray,
    frame: CalvinPhysicalSupervisionFrame,
    camera_name: str,
    target_identity_key: str,
    projection: dict[str, Any],
) -> Image.Image:
    camera = next(value for value in frame.cameras if value.camera_name == camera_name)
    owner_id = frame.identity_keys.index(target_identity_key) + 1
    view = projection["views"][camera_name]
    projected = project_qwen3vl_segmentation(
        camera.owner_index,
        instance_ids=tuple(range(1, len(frame.identity_keys) + 1)),
        image_grid_thw=np.asarray(view["image_grid_thw"], dtype=np.int64),
        patch_size=int(projection["patch_size"]),
        merge_size=int(projection["merge_size"]),
        pixel_supervised=camera.owner_supervised,
        minimum_supervised_fraction=0.0,
    ).merged
    probability = np.zeros(projected.supervised.shape, dtype=np.float32)
    if owner_id in projected.instance_ids:
        probability = projected.object_probability[:, projected.instance_ids.index(owner_id)]
    rows, columns = (int(value) for value in view["merged_grid_hw"])
    boxes = regular_grid_pixel_boxes(
        height=rgb.shape[0],
        width=rgb.shape[1],
        rows=rows,
        columns=columns,
    )
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for token_index, (y0, x0, y1, x1) in enumerate(boxes.tolist()):
        mass = float(probability[token_index])
        outline = (220, 220, 220, 150)
        if mass > 0:
            draw.rectangle(
                (x0, y0, x1 - 1, y1 - 1),
                fill=(231, 76, 60, round(96 * mass)),
                outline=(231, 76, 60, 240),
                width=3,
            )
        else:
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=outline)
    return image


def _render_group(
    *,
    source: _InputGroup,
    frame: CalvinPhysicalSupervisionFrame,
    arrays: dict[str, np.ndarray],
    projection: dict[str, Any],
    supports: dict[str, CalvinTokenGridIdentitySupport],
) -> bytes:
    width = 900
    camera_width = width // 2
    camera_height = camera_width
    title_height = 60
    variant_height = camera_height + 108
    canvas = Image.new(
        "RGB",
        (width, title_height + variant_height * len(source.group.variants)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 7),
        (
            f"step={source.group.source_global_index} scene={source.scene} "
            "Qwen merged token target support"
        ),
        fill="black",
        font=_font(17),
    )
    for variant_index, variant in enumerate(source.group.variants):
        support = supports[variant.target_identity_key]
        top = title_height + variant_index * variant_height
        for camera_index, camera_name in enumerate(("static", "gripper")):
            field = f"rgb_{camera_name}"
            overlay = _target_overlay(
                rgb=arrays[field],
                frame=frame,
                camera_name=camera_name,
                target_identity_key=variant.target_identity_key,
                projection=projection,
            ).resize((camera_width, camera_height), resample=Image.Resampling.NEAREST)
            canvas.paste(overlay, (camera_index * camera_width, top))
            draw.text(
                (camera_index * camera_width + 6, top + 5),
                camera_name,
                fill="white",
                stroke_width=2,
                stroke_fill="black",
                font=_font(16),
            )
        status = "KEEP" if support.object_row_addressable else "DROP"
        lines = textwrap.wrap(
            (
                f"{status} task={variant.task_key} target={variant.target_identity_key} "
                f"mass={support.target_mass:.6g} positive_tokens={support.positive_token_count} "
                f"object_winners={support.strict_object_winner_token_count} | "
                f"{variant.instruction}"
            ),
            width=95,
        )
        for line_index, line in enumerate(lines[:4]):
            draw.text(
                (8, top + camera_height + 5 + 21 * line_index),
                line,
                fill="black",
                font=_font(14),
            )
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _histogram(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument(
        "--representation-partition",
        required=True,
        choices=_REPRESENTATION_PARTITIONS,
    )
    parser.add_argument("--applicability-report", type=Path, required=True)
    parser.add_argument("--applicability-report-sha256", required=True)
    parser.add_argument("--training-projection-contract", type=Path, required=True)
    parser.add_argument("--training-projection-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    visual_output = args.visual_output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if visual_output.exists() or visual_output.is_symlink():
        raise FileExistsError(visual_output)

    dataset_manifest_path = args.dataset_manifest.resolve()
    dataset_manifest_sha256 = file_sha256(dataset_manifest_path)
    manifest = load_dataset_file_manifest(dataset_manifest_path)
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
    projection = load_lingbot_calvin_projection_contract(
        args.training_projection_contract.resolve(),
        expected_sha256=args.training_projection_contract_sha256,
        expected_dataset_manifest_sha256=dataset_manifest_sha256,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root.resolve(), index)
    applicability_report = _bound_json(
        args.applicability_report,
        expected_sha256=args.applicability_report_sha256,
    )
    source_groups = _validate_report_identity(
        applicability_report,
        dataset_manifest_sha256=dataset_manifest_sha256,
        sidecar_manifest_sha256=sidecar.manifest_sha256,
        representation_split=representation_split,
        representation_split_file_sha256=representation_split_file_sha256,
        expected_representation_partition=args.representation_partition,
    )
    supervision_policy = build_known_pixel_token_supervision_policy()
    minimum_supervised_fraction = float.fromhex(
        str(supervision_policy["minimum_observed_fraction_hex"])
    )

    visual_output.mkdir(parents=True, exist_ok=False)
    eligible_group_records = []
    rejected_group_records = []
    eligible_visual_records = []
    rejected_visual_records = []
    retained_tasks: list[str] = []
    retained_targets: list[str] = []
    addressable_tasks: list[str] = []
    addressable_targets: list[str] = []
    for source in source_groups:
        global_index = source.group.source_global_index
        index.source_episode(global_index)
        frame = sidecar.source_frame(global_index)
        _verify_group_source(source=source, frame=frame, sidecar=sidecar)
        supports = project_calvin_token_grid_identity_support(
            frame,
            projection=projection,
            minimum_supervised_fraction=minimum_supervised_fraction,
        )
        support_by_identity = {value.identity_key: value for value in supports}
        variant_records = []
        retained = []
        for variant in source.group.variants:
            support = support_by_identity[variant.target_identity_key]
            keep = support.object_row_addressable
            if keep:
                retained.append(variant)
                addressable_tasks.append(variant.task_key)
                addressable_targets.append(variant.target_identity_key)
            variant_records.append(
                {
                    "fixed_x_diagnostic_eligible": keep,
                    "instruction": variant.instruction,
                    "instruction_sha256": variant.instruction_sha256,
                    "proof": variant.proof,
                    "support": support.as_dict(),
                    "target_identity_key": variant.target_identity_key,
                    "task_key": variant.task_key,
                }
            )
        group_eligible = len(retained) >= 2
        group_record = {
            "fixed_x_group_eligible": group_eligible,
            "retained_target_identity_keys": [value.target_identity_key for value in retained],
            "retained_task_keys": [value.task_key for value in retained],
            "scene": source.scene,
            "source_global_index": global_index,
            "source_sensor_sha256": dict(source.source_sensor_sha256),
            "source_state_sha256": source.group.source_state_sha256,
            "stateful_reset_binding": source.stateful_reset_binding.as_dict(),
            "variants": variant_records,
        }
        arrays = dict(
            index.validated_source_frame_arrays(
                global_index,
                fields=("rgb_gripper", "rgb_static"),
            )
        )
        png = _render_group(
            source=source,
            frame=frame,
            arrays=arrays,
            projection=projection,
            supports=support_by_identity,
        )
        task_slug = "__".join(value.task_key for value in source.group.variants)
        filename = f"step_{global_index:07d}__{task_slug}.png"
        write_bytes_durable_exclusive(visual_output / filename, png)
        visual_record = {
            "file": filename,
            "png_sha256": hashlib.sha256(png).hexdigest(),
            "source_global_index": global_index,
        }
        if group_eligible:
            eligible_group_records.append(group_record)
            eligible_visual_records.append(visual_record)
            retained_tasks.extend(value.task_key for value in retained)
            retained_targets.extend(value.target_identity_key for value in retained)
        else:
            rejected_group_records.append(group_record)
            rejected_visual_records.append(visual_record)
        print(
            json.dumps(
                {
                    "fixed_x_group_eligible": group_eligible,
                    "processed": (len(eligible_group_records) + len(rejected_group_records)),
                    "source_global_index": global_index,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    addressable_task_set = set(addressable_tasks)
    addressable_target_set = set(addressable_targets)
    retained_task_set = set(retained_tasks)
    retained_target_set = set(retained_targets)
    complete_addressable_coverage = (
        retained_task_set == addressable_task_set and retained_target_set == addressable_target_set
    )
    status = "PASS" if eligible_group_records and complete_addressable_coverage else "FAIL"
    source_variant_count = sum(len(value.group.variants) for value in source_groups)
    report_content = {
        "acceptance_scope": {
            "fixed_x_evaluation_bank_authorized": (
                status == "PASS" and args.representation_partition != "training"
            ),
            "fixed_x_partition_artifact_authorized": status == "PASS",
            "fixed_x_training_stream_plan_authorized": (
                status == "PASS" and args.representation_partition == "training"
            ),
            "raw_owner_visibility_proven": True,
            "representation_partition_isolation_proven": True,
            "source_state_and_sensor_hash_binding_proven": True,
            "stateful_reset_addressability_proven": True,
            "token_grid_measurability_proven_for_retained_variants": status == "PASS",
            "training_authorized": False,
        },
        "applicability_artifact_sha256": applicability_report["artifact_sha256"],
        "applicability_report_sha256": args.applicability_report_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_runtime_binding": runtime_binding,
        "group_count": len(eligible_group_records),
        "groups": eligible_group_records,
        "leakage_contract": _EXPECTED_LEAKAGE_CONTRACT,
        "measurement_contract": {
            "absolute_pixel_or_probability_threshold": None,
            "context_is_not_an_object_row": True,
            "fixed_x_retention_rule": (
                "target-owner-mass-strictly-exceeds-every-other-physical-object-in-"
                "at-least-one-supervised-merged-token"
            ),
            "model_input": False,
            "projection": "exact-pinned-qwen3vl-patch-and-spatial-merger-addresses",
            "target_measure": "known-owner-mass-conditioned-within-token",
        },
        "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
        "representation_split": applicability_report["representation_split"],
        "rejected_groups": rejected_group_records,
        "rejected_visual_artifacts": rejected_visual_records,
        "schema": TOKEN_GRID_AUDIT_SCHEMA,
        "source_group_count": len(source_groups),
        "status": status,
        "summary": {
            "addressable_target_histogram": _histogram(addressable_targets),
            "addressable_task_histogram": _histogram(addressable_tasks),
            "addressable_variant_count": len(addressable_tasks),
            "dropped_variant_count": source_variant_count - len(retained_tasks),
            "eligible_group_count": len(eligible_group_records),
            "ineligible_group_count": len(rejected_group_records),
            "retained_target_histogram": _histogram(retained_targets),
            "retained_task_histogram": _histogram(retained_tasks),
            "retained_variant_count": len(retained_tasks),
            "source_variant_count": source_variant_count,
            "stranded_addressable_variant_count": (len(addressable_tasks) - len(retained_tasks)),
        },
        "training_projection_contract_sha256": (args.training_projection_contract_sha256),
        "training_projection_payload_sha256": projection_payload_sha256(projection),
        "training_supervision_policy": supervision_policy,
        "training_supervision_policy_sha256": token_supervision_policy_sha256(supervision_policy),
        "visual_artifacts": eligible_visual_records,
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
    write_bytes_durable_exclusive(output, payload)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "eligible_group_count": len(eligible_group_records),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "group_count": len(eligible_group_records),
                "rejected_group_count": len(rejected_group_records),
                "output": str(output),
                "representation_partition": args.representation_partition,
                "status": status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

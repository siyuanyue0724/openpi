"""Read-only visual evidence for LingBot-native object relations.

The renderer consumes tensors that already left the model forward.  Source RGB,
loss-only labels and Hungarian assignments are used only to explain the output;
none of them can update or route the persistent posterior.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from picf_next.artifact_io import publish_prepared_file_durable_exclusive
from picf_next.data.lingbot_calvin_projection import LINGBOT_CALVIN_CAMERA_SLOTS
from picf_next.lingbot_native.calvin_entity_set import (
    PhysicalCALVINFrameTargetBundle,
    physical_frame_predictions_from_relation,
)
from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.entity_set_objective import PhysicalSetLoss
from picf_next.lingbot_native.modalities import CALVIN_VIDEOMT_MASK_LAYOUT, NO_RELATION_TARGET
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.supervision import (
    assignment_binding_start_phase,
    assignment_binding_valid_at_phase,
    assignment_row_to_track_at_phase,
)

NATIVE_VISUAL_AUDIT_SCHEMA = "picf-next.lingbot-native-relation-visual.v5"
TASK_INDEPENDENT_ENTITY_VISUAL_SCHEMA = "picf-next.lingbot-task-independent-entity-visual.v2"

FACTORIZED_ANCHOR_SURFACE = "task_object_probability.max(row)"
LEGACY_ANCHOR_SURFACE = "ownership_or_support_times_task_relevance.max(row)"

_COLORS = np.asarray(
    (
        (230, 57, 70),
        (29, 185, 204),
        (255, 190, 11),
        (131, 56, 236),
        (56, 176, 0),
        (251, 133, 0),
        (0, 119, 182),
        (247, 37, 133),
        (106, 76, 147),
        (67, 170, 139),
        (249, 65, 68),
        (87, 117, 144),
        (144, 190, 109),
        (244, 162, 97),
        (38, 70, 83),
        (255, 0, 110),
    ),
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str, *, maximum: int = 72) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or "unnamed")[:maximum]


def _source_rgb(value: object) -> np.ndarray:
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("native visual source must be one CHW RGB tensor")
    array = value.detach().cpu().permute(1, 2, 0).float().numpy()
    if not np.isfinite(array).all():
        raise ValueError("native visual source contains NaN or infinity")
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _resize_grid(values: np.ndarray, *, width: int, height: int, nearest: bool) -> np.ndarray:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("native visual grid must be one finite matrix")
    image = Image.fromarray(values.astype(np.float32))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(image.resize((width, height), resample=resample), dtype=np.float32)


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    expanded = _resize_grid(heat, width=rgb.shape[1], height=rgb.shape[0], nearest=False)
    expanded = np.clip(expanded, 0.0, 1.0)
    color = np.stack((np.ones_like(expanded), 0.75 * expanded, np.zeros_like(expanded)), axis=-1)
    alpha = (0.72 * expanded)[..., None]
    return np.clip(rgb * (1.0 - alpha) + 255.0 * color * alpha, 0, 255).astype(np.uint8)


def _categorical_overlay(
    rgb: np.ndarray,
    probability: np.ndarray,
    *,
    category_colors: np.ndarray,
) -> np.ndarray:
    if probability.ndim != 3 or probability.shape[-1] != len(category_colors):
        raise ValueError("native categorical visual has invalid category geometry")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("native categorical visual probabilities must lie in [0,1]")
    confidence = probability.max(axis=-1)
    winner = probability.argmax(axis=-1)
    winner_full = _resize_grid(
        winner.astype(np.float32),
        width=rgb.shape[1],
        height=rgb.shape[0],
        nearest=True,
    ).astype(np.int64)
    confidence_full = _resize_grid(
        confidence,
        width=rgb.shape[1],
        height=rgb.shape[0],
        nearest=False,
    )
    color = category_colors[winner_full]
    alpha = (0.68 * np.clip(confidence_full, 0.0, 1.0))[..., None]
    return np.clip(rgb * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _target_colors_in_row_gauge(
    row_to_track: Sequence[int],
    *,
    track_count: int,
    row_colors: np.ndarray,
) -> np.ndarray:
    """Color loss-only tracks by their assigned posterior row.

    Capacity-censored or otherwise unassigned targets remain neutral gray so
    the audit never invents a row correspondence that the loss did not use.
    """

    if isinstance(track_count, bool) or not isinstance(track_count, int) or track_count < 0:
        raise ValueError("visual target track count must be a non-negative integer")
    if row_colors.shape != (len(row_to_track), 3):
        raise ValueError("visual row colors differ from the row assignment")
    colors = np.full((track_count, 3), 160.0, dtype=np.float32)
    assigned = np.zeros(track_count, dtype=np.bool_)
    for row_index, track_index in enumerate(row_to_track):
        if track_index < 0:
            continue
        if track_index >= track_count:
            raise ValueError("visual row assignment references an absent target track")
        if assigned[track_index]:
            raise ValueError("visual target track is assigned to multiple rows")
        colors[track_index] = row_colors[row_index]
        assigned[track_index] = True
    return colors


def _panel(rgb: np.ndarray, title: str) -> Image.Image:
    top = 24
    output = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + top), "white")
    output.paste(Image.fromarray(rgb), (0, top))
    ImageDraw.Draw(output).text((6, 5), title, fill="black")
    return output


def _atomic_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"native visual artifact already exists: {path}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"native visual staging artifact already exists: {temporary}")
    try:
        image.save(temporary, format="PNG", optimize=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        publish_prepared_file_durable_exclusive(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def matched_row_soft_iou(
    *,
    objective: NativeCALVINObjectiveResult,
    structural_sensor_valid: torch.Tensor,
    batch_index: int,
) -> list[float | None]:
    """Measure visible physical agreement in the fixed loss-side row gauge."""

    predictions = objective.predictions
    targets = objective.targets
    row_to_track = objective.assignment.row_to_track[batch_index]
    source_binding_valid = assignment_binding_valid_at_phase(
        objective.assignment,
        targets,
        source_phase=1,
    )[batch_index]
    values: list[float | None] = []
    for row_index, track_index in enumerate(row_to_track.detach().cpu().tolist()):
        if track_index < 0 or not bool(source_binding_valid[row_index].item()):
            values.append(None)
            continue
        valid = (
            targets.mask_valid[batch_index, 0, track_index] & structural_sensor_valid[batch_index]
        )
        target = targets.masks[batch_index, 0, track_index].float()
        target_mass = target[valid].sum()
        if not valid.any() or not bool(target_mass > 0):
            values.append(None)
            continue
        prediction = (
            predictions.ownership[batch_index, 0, :, row_index]
            if targets.exclusive_ownership
            else predictions.support_logits[batch_index, 0, :, row_index].sigmoid()
        ).float()
        weight = (
            targets.token_observed_fraction[batch_index, 0]
            if targets.exclusive_ownership
            else torch.ones_like(target)
        ).float()
        effective_weight = weight[valid]
        expected = target[valid]
        measured = prediction[valid]
        intersection = (effective_weight * measured * expected).sum()
        union = (effective_weight * (measured + expected - measured * expected)).sum()
        if not bool(union > 0):
            raise RuntimeError("visible matched row has zero soft-IoU union")
        values.append(round(float((intersection / union).detach().cpu()), 7))
    return values


def matched_physical_row_soft_iou(
    *,
    relation: PhysicalRelationOutput,
    target_bundle: PhysicalCALVINFrameTargetBundle,
    set_loss: PhysicalSetLoss,
    batch_index: int,
) -> list[float | None]:
    """Measure prompt-free entity agreement in the current loss-side row gauge."""

    if not isinstance(relation, PhysicalRelationOutput):
        raise TypeError("physical visual IoU requires a task-independent relation")
    if not isinstance(target_bundle, PhysicalCALVINFrameTargetBundle):
        raise TypeError("physical visual IoU requires a CALVIN physical target bundle")
    if not isinstance(set_loss, PhysicalSetLoss):
        raise TypeError("physical visual IoU requires one physical set loss")
    targets = target_bundle.targets
    predictions = physical_frame_predictions_from_relation(relation)
    batch, _tokens, rows = predictions.support_logits.shape
    if not 0 <= batch_index < batch:
        raise IndexError("physical visual batch index is outside the relation batch")
    row_to_track = set_loss.assignment.row_to_track
    if row_to_track.shape != (batch, rows):
        raise ValueError("physical visual assignment differs from the relation rows")

    values: list[float | None] = []
    for row_index, track_index in enumerate(row_to_track[batch_index].tolist()):
        if track_index < 0:
            values.append(None)
            continue
        if track_index >= targets.masks.shape[1]:
            raise ValueError("physical visual assignment references an absent target track")
        valid = targets.mask_valid[batch_index, track_index] & predictions.sensor_valid[batch_index]
        expected = targets.masks[batch_index, track_index].float()
        if not valid.any() or not bool(expected[valid].sum() > 0):
            values.append(None)
            continue
        measured = (
            predictions.ownership_log_probability[batch_index, :, row_index].exp()
            if targets.exclusive_ownership
            else predictions.support_logits[batch_index, :, row_index].sigmoid()
        ).float()
        weight = (
            targets.token_observed_fraction[batch_index] * targets.token_measure[batch_index]
            if targets.exclusive_ownership
            else torch.ones_like(expected)
        ).float()
        effective_weight = weight[valid]
        expected = expected[valid]
        measured = measured[valid]
        intersection = (effective_weight * measured * expected).sum()
        union = (effective_weight * (measured + expected - measured * expected)).sum()
        if not bool(union > 0):
            raise RuntimeError("physical visual matched row has zero soft-IoU union")
        values.append(round(float((intersection / union).detach().cpu()), 7))
    return values


def render_task_independent_entity_visuals(
    *,
    output_root: Path,
    global_step: int,
    input_weight_global_step: int | None = None,
    weight_boundary: str = "pre_update_forward",
    rank: int,
    host_items: Sequence[Mapping[str, Any]],
    model_inputs: Mapping[str, Any],
    relation: PhysicalRelationOutput,
    target_bundle: PhysicalCALVINFrameTargetBundle,
    set_loss: PhysicalSetLoss,
    sample_keys: Sequence[str],
    merge_size: int,
) -> list[dict[str, Any]]:
    """Render source, prompt-free ownership, context and loss-only identities."""

    integers = (global_step, rank, merge_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("physical visual step, rank and merge size must be integers")
    if global_step < 0 or rank < 0 or merge_size <= 0:
        raise ValueError("physical visual controls are outside their valid ranges")
    if input_weight_global_step is None:
        if global_step == 0:
            raise ValueError("physical visual step zero requires an explicit weight boundary")
        resolved_input_weight_global_step = global_step - 1
    elif (
        isinstance(input_weight_global_step, bool)
        or not isinstance(input_weight_global_step, int)
        or input_weight_global_step < 0
    ):
        raise ValueError("physical visual input-weight step must be a non-negative integer")
    else:
        resolved_input_weight_global_step = input_weight_global_step
    if not isinstance(weight_boundary, str) or not weight_boundary:
        raise ValueError("physical visual weight boundary must be a nonempty string")
    if not isinstance(relation, PhysicalRelationOutput):
        raise TypeError("physical visual requires a task-independent relation")
    if not isinstance(target_bundle, PhysicalCALVINFrameTargetBundle):
        raise TypeError("physical visual requires a CALVIN physical target bundle")
    if not isinstance(set_loss, PhysicalSetLoss):
        raise TypeError("physical visual requires one physical set loss")
    batch = len(host_items)
    if batch == 0 or len(sample_keys) != batch:
        raise ValueError("physical visual host items and sample keys must share a nonempty batch")
    if relation.support_logits.shape[0] != batch:
        raise ValueError("physical visual relation differs from the host batch")
    targets = target_bundle.targets
    if targets.masks.shape[0] != batch or len(target_bundle.identity_keys_by_batch) != batch:
        raise ValueError("physical visual targets differ from the host batch")
    rows = relation.support_logits.shape[-1]
    assignment = set_loss.assignment.row_to_track
    if assignment.shape != (batch, rows):
        raise ValueError("physical visual assignment differs from the relation rows")
    supervised_surfaces = tuple(
        surface
        for surface in relation.relation_surfaces
        if surface.target_kind != NO_RELATION_TARGET
    )
    surface_offsets: dict[str, int] = {}
    target_offset = relation.support_logits.shape[1]
    for surface in supervised_surfaces:
        surface_offsets[surface.name] = target_offset
        target_offset += surface.support_logits.shape[1]
    if target_offset != targets.masks.shape[-1]:
        raise ValueError("physical visual relation surfaces differ from the target token axis")

    grids = model_inputs.get("image_grid_thw")
    image_valid = model_inputs.get("img_masks")
    expected_views = len(LINGBOT_CALVIN_CAMERA_SLOTS)
    if (
        not isinstance(grids, torch.Tensor)
        or grids.shape != (batch, expected_views, 3)
        or grids.dtype != torch.long
        or not isinstance(image_valid, torch.Tensor)
        or image_valid.shape != (batch, expected_views)
        or image_valid.dtype != torch.bool
    ):
        raise ValueError("physical visual Qwen view geometry differs from the frozen interface")
    expected_valid = torch.tensor(
        [slot.valid for slot in LINGBOT_CALVIN_CAMERA_SLOTS],
        dtype=torch.bool,
        device=image_valid.device,
    ).expand(batch, -1)
    if not torch.equal(image_valid, expected_valid):
        raise ValueError("physical visual Qwen view validity differs from the frozen interface")
    for slot_index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS):
        if slot.valid:
            continue
        source_index = next(
            (
                index
                for index, candidate in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS)
                if candidate.valid
                and candidate.projection_camera_name == slot.projection_camera_name
            ),
            None,
        )
        if source_index is None or not torch.equal(grids[:, slot_index], grids[:, source_index]):
            raise ValueError("physical visual padded Qwen grid differs from its source view")

    artifacts: list[dict[str, Any]] = []
    for batch_index, (host_item, sample_key) in enumerate(
        zip(host_items, sample_keys, strict=True)
    ):
        task = host_item.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("physical visual host item has no task annotation")
        identity_keys = target_bundle.identity_keys_by_batch[batch_index]
        row_to_track = assignment[batch_index].detach().cpu().tolist()
        if any(track >= len(identity_keys) for track in row_to_track):
            raise ValueError("physical visual assignment differs from target identities")
        valid_target_tracks = targets.track_valid[batch_index].nonzero().flatten()
        expected_target_tracks = torch.arange(
            len(identity_keys),
            device=valid_target_tracks.device,
        )
        if not torch.equal(valid_target_tracks, expected_target_tracks):
            raise ValueError("physical visual target identities differ from valid target tracks")

        row_colors = _COLORS[np.arange(rows) % len(_COLORS)]
        track_colors = _target_colors_in_row_gauge(
            row_to_track,
            track_count=len(identity_keys),
            row_colors=row_colors,
        )
        sensor_positions = relation.structural_valid[batch_index].nonzero().flatten()
        sensor_offset = 0
        rows_of_panels: list[Image.Image] = []
        view_metadata: list[dict[str, Any]] = []
        surface_view_metadata: list[dict[str, Any]] = []
        for view_index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS):
            if not bool(image_valid[batch_index, view_index].item()):
                continue
            view_name = slot.physical_camera_name
            if view_name is None:
                raise RuntimeError("an invalid CALVIN camera slot reached physical visualization")
            source_field = f"observation.images.{slot.runtime_camera_name}"
            rgb = _source_rgb(host_item.get(source_field))
            grid = grids[batch_index, view_index].detach().cpu().tolist()
            if grid[0] != 1 or grid[1] % merge_size or grid[2] % merge_size:
                raise ValueError("physical visual Qwen grid is not a divisible single frame")
            merged_h, merged_w = grid[1] // merge_size, grid[2] // merge_size
            count = merged_h * merged_w
            positions = sensor_positions[sensor_offset : sensor_offset + count]
            if positions.numel() != count:
                raise ValueError("physical visual sensor partition ended inside one view")
            ownership = relation.ownership[batch_index, positions, :-1]
            context = relation.context_probability[batch_index, positions]
            target = targets.masks[
                batch_index,
                : len(identity_keys),
                positions,
            ].transpose(0, 1)
            panels = (
                _panel(rgb, f"{view_name}: source"),
                _panel(
                    _categorical_overlay(
                        rgb,
                        ownership.detach().float().cpu().numpy().reshape(merged_h, merged_w, rows),
                        category_colors=row_colors,
                    ),
                    f"{view_name}: prompt-free rows",
                ),
                _panel(
                    _heat_overlay(
                        rgb,
                        context.detach().float().cpu().numpy().reshape(merged_h, merged_w),
                    ),
                    f"{view_name}: context probability",
                ),
                _panel(
                    _categorical_overlay(
                        rgb,
                        target.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(merged_h, merged_w, len(identity_keys)),
                        category_colors=track_colors,
                    ),
                    f"{view_name}: loss-only identities",
                ),
            )
            row_image = Image.new(
                "RGB",
                (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
                "white",
            )
            x = 0
            for panel in panels:
                row_image.paste(panel, (x, 0))
                x += panel.width
            rows_of_panels.append(row_image)
            view_metadata.append(
                {
                    "name": view_name,
                    "merged_grid": [merged_h, merged_w],
                    "source_shape": list(rgb.shape),
                    "token_count": count,
                }
            )
            for surface in supervised_surfaces:
                donor_panels: tuple[Image.Image, ...] = ()
                donor_top_queries: list[dict[str, Any]] = []
                if surface.name == "vjepa":
                    if (
                        surface.geometry_kind != "image_grid"
                        or surface.layout != "vjepa21.calvin.static-gripper.24x24.v1"
                        or surface.support_logits.shape[1] != 2 * 24 * 24
                    ):
                        raise ValueError("physical visual received an invalid V-JEPA surface")
                    native_view_index = {"static": 0, "gripper": 1}.get(view_name)
                    if native_view_index is None:
                        raise ValueError("V-JEPA physical visual received an unknown camera view")
                    native_height = native_width = 24
                    native_label = "V-JEPA"
                    native_start = native_view_index * native_height * native_width
                    native_stop = native_start + native_height * native_width
                elif surface.name == "videomt_masks":
                    if view_name != "static":
                        continue
                    if (
                        surface.geometry_kind != "image_grid"
                        or surface.layout != CALVIN_VIDEOMT_MASK_LAYOUT
                        or surface.grid_shape != (120, 120)
                        or surface.support_logits.shape[1] != 120 * 120
                    ):
                        raise ValueError("physical visual received an invalid VidEoMT surface")
                    native_height, native_width = surface.grid_shape
                    native_label = "VidEoMT"
                    native_start = 0
                    native_stop = native_height * native_width
                    decomposition = (
                        surface.donor_query_probability,
                        surface.donor_context_probability,
                        surface.contextual_query_ownership,
                        surface.query_valid,
                        surface.canonical_query_ids,
                    )
                    if any(value is not None for value in decomposition):
                        if any(value is None for value in decomposition):
                            raise ValueError(
                                "VidEoMT visual received a partial query decomposition"
                            )
                        query_valid = surface.query_valid[batch_index]
                        valid_queries = query_valid.nonzero().flatten()
                        if valid_queries.numel() == 0:
                            raise ValueError("VidEoMT visual has no valid object queries")
                        donor_query = surface.donor_query_probability[
                            batch_index,
                            native_start:native_stop,
                        ]
                        query_mass = donor_query.sum(dim=0)
                        top_count = min(len(_COLORS), valid_queries.numel())
                        selected = valid_queries[
                            torch.topk(query_mass[valid_queries], k=top_count).indices
                        ]
                        selected_probability = donor_query[:, selected]
                        donor_panels = (
                            _panel(
                                _heat_overlay(
                                    rgb,
                                    (1.0 - surface.donor_context_probability[
                                        batch_index,
                                        native_start:native_stop,
                                    ])
                                    .detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .reshape(native_height, native_width),
                                ),
                                "static: VidEoMT donor object mass",
                            ),
                            _panel(
                                _categorical_overlay(
                                    rgb,
                                    selected_probability.detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .reshape(native_height, native_width, top_count),
                                    category_colors=_COLORS[:top_count],
                                ),
                                "static: VidEoMT top donor queries",
                            ),
                        )
                        query_ownership = surface.contextual_query_ownership[batch_index]
                        canonical_query_ids = surface.canonical_query_ids[batch_index]
                        for color_index, query_index in enumerate(selected.tolist()):
                            donor_top_queries.append(
                                {
                                    "color_rgb": _COLORS[color_index].astype(int).tolist(),
                                    "query_index": int(query_index),
                                    "canonical_query_id": int(
                                        canonical_query_ids[query_index].item()
                                    ),
                                    "pixel_mass": float(query_mass[query_index].item()),
                                    "row_probability": query_ownership[
                                        query_index, :-1
                                    ].detach().float().cpu().tolist(),
                                    "context_probability": float(
                                        query_ownership[query_index, -1].item()
                                    ),
                                }
                            )
                else:
                    raise ValueError("physical visual received an unsupported supervised surface")
                native_valid = surface.sensor_valid[batch_index, native_start:native_stop]
                if native_valid.any() and not native_valid.all():
                    raise ValueError("physical visual native view is only partially available")
                full_start = surface_offsets[surface.name] + native_start
                full_stop = surface_offsets[surface.name] + native_stop
                native_ownership = surface.ownership[
                    batch_index,
                    native_start:native_stop,
                    :-1,
                ]
                native_context = surface.context_probability[
                    batch_index,
                    native_start:native_stop,
                ]
                native_target = targets.masks[
                    batch_index,
                    : len(identity_keys),
                    full_start:full_stop,
                ].transpose(0, 1)
                native_panels = (
                    _panel(rgb, f"{view_name}: source"),
                    *donor_panels,
                    _panel(
                        _categorical_overlay(
                            rgb,
                            native_ownership.detach()
                            .float()
                            .cpu()
                            .numpy()
                            .reshape(native_height, native_width, rows),
                            category_colors=row_colors,
                        ),
                        f"{view_name}: {native_label} posterior rows",
                    ),
                    _panel(
                        _heat_overlay(
                            rgb,
                            native_context.detach()
                            .float()
                            .cpu()
                            .numpy()
                            .reshape(native_height, native_width),
                        ),
                        f"{view_name}: {native_label} context",
                    ),
                    _panel(
                        _categorical_overlay(
                            rgb,
                            native_target.detach()
                            .float()
                            .cpu()
                            .numpy()
                            .reshape(native_height, native_width, len(identity_keys)),
                            category_colors=track_colors,
                        ),
                        f"{view_name}: {native_label} loss-only identities",
                    ),
                )
                native_row = Image.new(
                    "RGB",
                    (
                        sum(panel.width for panel in native_panels),
                        max(panel.height for panel in native_panels),
                    ),
                    "white",
                )
                native_x = 0
                for panel in native_panels:
                    native_row.paste(panel, (native_x, 0))
                    native_x += panel.width
                rows_of_panels.append(native_row)
                surface_metadata: dict[str, Any] = {
                    "name": surface.name,
                    "view": view_name,
                    "layout": surface.layout,
                    "native_grid": [native_height, native_width],
                    "token_count": native_height * native_width,
                    "available": bool(native_valid.all().item()),
                }
                if donor_top_queries:
                    surface_metadata["donor_top_queries"] = donor_top_queries
                surface_view_metadata.append(surface_metadata)
            sensor_offset += count
        if sensor_offset != sensor_positions.numel():
            raise ValueError("physical visual views did not consume every structural sensor token")
        if not rows_of_panels:
            raise ValueError("physical visual sample has no valid camera view")

        matched_soft_iou = matched_physical_row_soft_iou(
            relation=relation,
            target_bundle=target_bundle,
            set_loss=set_loss,
            batch_index=batch_index,
        )
        existence = relation.existence[batch_index].detach().float().cpu().tolist()
        header_height = 72
        legend_height = 20 * max(1, rows) + 12
        canvas_width = max(row.width for row in rows_of_panels)
        canvas_height = header_height + sum(row.height for row in rows_of_panels) + legend_height
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 7),
            (
                f"update={global_step} input_weights={resolved_input_weight_global_step} "
                f"rank={rank} sample={sample_key}"
            ),
            fill="black",
        )
        draw.text((8, 27), f"task={task}", fill="black")
        draw.text(
            (8, 47),
            "task is audit metadata only; entity objective and posterior are prompt-free",
            fill="black",
        )
        y = header_height
        for row in rows_of_panels:
            canvas.paste(row, (0, y))
            y += row.height
        for row_index, track_index in enumerate(row_to_track):
            color = tuple(int(value) for value in row_colors[row_index])
            label = "unmatched" if track_index < 0 else identity_keys[track_index]
            iou = matched_soft_iou[row_index]
            iou_label = "n/a" if iou is None else f"{iou:.3f}"
            draw.rectangle((8, y + 3, 22, y + 17), fill=color)
            draw.text(
                (28, y + 3),
                (
                    f"row {row_index:02d} -> {label}; "
                    f"exist={existence[row_index]:.3f}; soft_iou={iou_label}"
                ),
                fill="black",
            )
            y += 20

        relative = Path("entity_visuals") / f"step_{global_step:08d}" / f"rank_{rank}"
        identity_digest = hashlib.sha256(f"{sample_key}\0{task}".encode()).hexdigest()[:12]
        filename = (
            f"{_slug(str(sample_key), maximum=48)}__task_{_slug(task, maximum=72)}"
            f"__{identity_digest}.png"
        )
        path = output_root / relative / filename
        _atomic_png(canvas, path)
        artifacts.append(
            {
                "schema": TASK_INDEPENDENT_ENTITY_VISUAL_SCHEMA,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "global_step": global_step,
                "input_weight_global_step": resolved_input_weight_global_step,
                "weight_boundary": weight_boundary,
                "rank": rank,
                "batch_index": batch_index,
                "sample_key": str(sample_key),
                "task": task,
                "identity_keys": list(identity_keys),
                "row_to_track": row_to_track,
                "row_existence": [round(float(value), 7) for value in existence],
                "row_matched_soft_iou": matched_soft_iou,
                "anchor_surface": (
                    "shared_rows_qwen_plus_native_surfaces"
                    if supervised_surfaces
                    else "task_independent_categorical_ownership"
                ),
                "target_color_mode": "assigned_row_or_unassigned_gray_v1",
                "views": view_metadata,
                "relation_surfaces": surface_view_metadata,
                "task_used_by_entity_objective": False,
                "loss_only_labels_visible_to_model": False,
            }
        )
    return artifacts


def render_native_relation_visuals(
    *,
    output_root: Path,
    global_step: int,
    input_weight_global_step: int | None = None,
    weight_boundary: str = "pre_update_forward",
    rank: int,
    host_items: Sequence[Mapping[str, Any]],
    model_inputs: Mapping[str, Any],
    objective: NativeCALVINObjectiveResult,
    structural_sensor_valid: torch.Tensor,
    sample_keys: Sequence[str],
    merge_size: int,
) -> list[dict[str, Any]]:
    """Render current-model relation evidence with task-labelled file names."""

    integers = (global_step, rank, merge_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("native visual step, rank and merge size must be integers")
    if global_step < 0 or rank < 0 or merge_size <= 0:
        raise ValueError("native visual step, rank and merge size are outside their range")
    if input_weight_global_step is None:
        if global_step == 0:
            raise ValueError("native visual step zero requires an explicit weight boundary")
        resolved_input_weight_global_step = global_step - 1
    elif (
        isinstance(input_weight_global_step, bool)
        or not isinstance(input_weight_global_step, int)
        or input_weight_global_step < 0
    ):
        raise ValueError("native visual input-weight step must be a non-negative integer")
    else:
        resolved_input_weight_global_step = input_weight_global_step
    if not isinstance(weight_boundary, str) or not weight_boundary:
        raise ValueError("native visual weight boundary must be a nonempty string")
    batch = len(host_items)
    if batch == 0 or len(sample_keys) != batch:
        raise ValueError("native visual host items and sample keys must share a nonempty batch")
    grids = model_inputs.get("image_grid_thw")
    image_valid = model_inputs.get("img_masks")
    expected_views = len(LINGBOT_CALVIN_CAMERA_SLOTS)
    if (
        not isinstance(grids, torch.Tensor)
        or grids.shape != (batch, expected_views, 3)
        or grids.dtype != torch.long
        or not isinstance(image_valid, torch.Tensor)
        or image_valid.shape != (batch, expected_views)
        or image_valid.dtype != torch.bool
    ):
        raise ValueError("native visual Qwen view geometry differs from the frozen interface")
    expected_valid = torch.tensor(
        [slot.valid for slot in LINGBOT_CALVIN_CAMERA_SLOTS],
        dtype=torch.bool,
        device=image_valid.device,
    ).expand(batch, -1)
    if not torch.equal(image_valid, expected_valid):
        raise ValueError("native visual Qwen view validity differs from the frozen interface")
    for slot_index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS):
        if slot.valid:
            continue
        source_index = next(
            (
                index
                for index, candidate in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS)
                if candidate.valid
                and candidate.projection_camera_name == slot.projection_camera_name
            ),
            None,
        )
        if source_index is None or not torch.equal(
            grids[:, slot_index],
            grids[:, source_index],
        ):
            raise ValueError("native visual padded Qwen grid differs from its source view")
    predictions = objective.predictions
    targets = objective.targets
    assignment = objective.assignment.row_to_track
    binding_start_phase = assignment_binding_start_phase(
        objective.assignment,
        targets,
    )
    visual_source_time = 0
    visual_source_phase = 1
    source_binding_valid = assignment_binding_valid_at_phase(
        objective.assignment,
        targets,
        source_phase=visual_source_phase,
    )
    source_assignment = assignment_row_to_track_at_phase(
        objective.assignment,
        targets,
        source_phase=visual_source_phase,
    )
    factorized_task_relation = any(
        term.name == "set/task_row" for term in objective.structural_terms
    )
    if factorized_task_relation and predictions.task_object_probability is None:
        raise RuntimeError(
            "factorized task relation omitted the exact task-object visualization field"
        )
    anchor_surface = (
        FACTORIZED_ANCHOR_SURFACE if factorized_task_relation else LEGACY_ANCHOR_SURFACE
    )
    if predictions.support_logits.shape[0] != batch or predictions.support_logits.shape[1] < 1:
        raise ValueError("native visual predictions do not contain the current batch")
    if (
        structural_sensor_valid.shape
        != (predictions.support_logits.shape[0], predictions.support_logits.shape[2])
        or structural_sensor_valid.dtype != torch.bool
    ):
        raise ValueError("native visual structural sensor validity differs from predictions")
    if assignment.shape[0] != batch or len(objective.track_identity_keys_by_batch) != batch:
        raise ValueError("native visual assignment does not match the current batch")

    artifacts: list[dict[str, Any]] = []
    for batch_index, (host_item, sample_key) in enumerate(
        zip(host_items, sample_keys, strict=True)
    ):
        task = host_item.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("native visual host item has no task annotation")
        row_to_track = assignment[batch_index].detach().cpu().tolist()
        source_row_to_track = source_assignment[batch_index].detach().cpu().tolist()
        row_binding_start_phase = binding_start_phase[batch_index].detach().cpu().tolist()
        row_source_binding_valid = source_binding_valid[batch_index].detach().cpu().tolist()
        identity_keys = objective.track_identity_keys_by_batch[batch_index]
        # Model ownership is a row-index prediction.  Its colors must not depend
        # on a future loss-side identity assignment.
        row_colors = _COLORS[np.arange(len(row_to_track)) % len(_COLORS)]
        track_colors = _COLORS[np.arange(len(identity_keys)) % len(_COLORS)]
        valid_target_tracks = targets.track_valid[batch_index].nonzero().flatten()
        expected_target_tracks = torch.arange(
            len(identity_keys),
            device=valid_target_tracks.device,
        )
        if not torch.equal(valid_target_tracks, expected_target_tracks):
            raise ValueError("native visual target identities differ from valid target tracks")
        sensor_positions = structural_sensor_valid[batch_index].nonzero().flatten()
        sensor_offset = 0
        rows_of_panels: list[Image.Image] = []
        view_metadata: list[dict[str, Any]] = []
        for view_index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS):
            if not bool(image_valid[batch_index, view_index].item()):
                continue
            view_name = slot.physical_camera_name
            if view_name is None:
                raise RuntimeError("an invalid CALVIN camera slot reached native visualization")
            source_field = f"observation.images.{slot.runtime_camera_name}"
            rgb = _source_rgb(host_item.get(source_field))
            grid = grids[batch_index, view_index].detach().cpu().tolist()
            if grid[0] != 1 or grid[1] % merge_size or grid[2] % merge_size:
                raise ValueError("native visual Qwen grid is not a divisible single frame")
            merged_h, merged_w = grid[1] // merge_size, grid[2] // merge_size
            count = merged_h * merged_w
            positions = sensor_positions[sensor_offset : sensor_offset + count]
            if positions.numel() != count:
                raise ValueError("native visual sensor partition ended inside one view")

            if predictions.task_row_probability_by_time is not None:
                task_relevance = predictions.task_row_probability_by_time[
                    batch_index,
                    visual_source_time,
                ]
            else:
                if predictions.support_logits.shape[1] != 1:
                    raise ValueError(
                        "legacy multi-frame visual has no causal per-time task probability"
                    )
                task_relevance = predictions.task_relevance_logits[batch_index].sigmoid()
            ownership = predictions.ownership[batch_index, 0, positions, :-1]
            if predictions.task_object_probability is not None:
                task_support = predictions.task_object_probability[
                    batch_index,
                    0,
                    positions,
                ]
                task_anchor = task_support.max(dim=-1).values
            else:
                task_support = (
                    ownership
                    if targets.exclusive_ownership
                    else predictions.support_logits[batch_index, 0, positions].sigmoid()
                )
                task_anchor = (task_support * task_relevance.unsqueeze(0)).max(dim=-1).values
            dense_task = predictions.dense_task_grounding_logits[
                batch_index,
                0,
                positions,
            ].sigmoid()
            target = targets.masks[
                batch_index,
                0,
                : len(identity_keys),
                positions,
            ].transpose(0, 1)
            target_panel = (
                _panel(rgb, f"{view_name}: loss-only target (no object)")
                if not identity_keys
                else _panel(
                    _categorical_overlay(
                        rgb,
                        target.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(
                            merged_h,
                            merged_w,
                            len(identity_keys),
                        ),
                        category_colors=track_colors,
                    ),
                    f"{view_name}: loss-only target",
                )
            )
            panels = (
                _panel(rgb, f"{view_name}: source"),
                _panel(
                    _categorical_overlay(
                        rgb,
                        ownership.detach().float().cpu().numpy().reshape(merged_h, merged_w, -1),
                        category_colors=row_colors,
                    ),
                    f"{view_name}: model ownership",
                ),
                _panel(
                    _heat_overlay(
                        rgb,
                        task_anchor.detach().float().cpu().numpy().reshape(merged_h, merged_w),
                    ),
                    f"{view_name}: row task anchor",
                ),
                _panel(
                    _heat_overlay(
                        rgb,
                        dense_task.detach().float().cpu().numpy().reshape(merged_h, merged_w),
                    ),
                    f"{view_name}: dense task",
                ),
                target_panel,
            )
            width = sum(panel.width for panel in panels)
            row_image = Image.new("RGB", (width, max(panel.height for panel in panels)), "white")
            x = 0
            for panel in panels:
                row_image.paste(panel, (x, 0))
                x += panel.width
            rows_of_panels.append(row_image)
            view_metadata.append(
                {
                    "name": view_name,
                    "merged_grid": [merged_h, merged_w],
                    "source_shape": list(rgb.shape),
                    "token_count": count,
                }
            )
            sensor_offset += count
        if sensor_offset != sensor_positions.numel():
            raise ValueError("native visual views did not consume every structural sensor token")
        if not rows_of_panels:
            raise ValueError("native visual sample has no valid camera view")

        header_height = 54
        legend_height = 20 * max(1, len(row_to_track)) + 12
        canvas_width = max(row.width for row in rows_of_panels)
        canvas_height = header_height + sum(row.height for row in rows_of_panels) + legend_height
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 7),
            (
                f"update={global_step} input_weights={resolved_input_weight_global_step} "
                f"rank={rank} sample={sample_key}"
            ),
            fill="black",
        )
        draw.text((8, 27), f"task={task}", fill="black")
        y = header_height
        for row in rows_of_panels:
            canvas.paste(row, (0, y))
            y += row.height
        existence = predictions.existence_logits[batch_index, 0].sigmoid().detach().cpu().tolist()
        if predictions.task_row_probability_by_time is not None:
            relevance_tensor = predictions.task_row_probability_by_time[
                batch_index,
                visual_source_time,
            ]
        else:
            if predictions.support_logits.shape[1] != 1:
                raise ValueError(
                    "legacy multi-frame legend has no causal per-time task probability"
                )
            relevance_tensor = predictions.task_relevance_logits[batch_index].sigmoid()
        relevance = relevance_tensor.detach().cpu().tolist()
        matched_soft_iou = matched_row_soft_iou(
            objective=objective,
            structural_sensor_valid=structural_sensor_valid,
            batch_index=batch_index,
        )
        for row_index, track_index in enumerate(row_to_track):
            color = tuple(int(value) for value in row_colors[row_index])
            if track_index < 0:
                label = "unmatched"
            elif not row_source_binding_valid[row_index]:
                label = f"pending@phase{row_binding_start_phase[row_index]}"
            else:
                label = identity_keys[track_index]
            iou = matched_soft_iou[row_index]
            iou_label = "n/a" if iou is None else f"{iou:.3f}"
            draw.rectangle((8, y + 3, 22, y + 17), fill=color)
            draw.text(
                (28, y + 3),
                (
                    f"row {row_index:02d} -> {label}; "
                    f"exist={existence[row_index]:.3f}; "
                    f"task_prob={relevance[row_index]:.3f}; " + f"soft_iou={iou_label}"
                ),
                fill="black",
            )
            y += 20

        relative = Path("visuals") / f"step_{global_step:08d}" / f"rank_{rank}"
        identity_digest = hashlib.sha256(f"{sample_key}\0{task}".encode()).hexdigest()[:12]
        filename = (
            f"{_slug(str(sample_key), maximum=48)}__task_{_slug(task, maximum=72)}"
            f"__{identity_digest}.png"
        )
        path = output_root / relative / filename
        _atomic_png(canvas, path)
        artifacts.append(
            {
                "schema": NATIVE_VISUAL_AUDIT_SCHEMA,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "global_step": global_step,
                "input_weight_global_step": resolved_input_weight_global_step,
                "weight_boundary": weight_boundary,
                "rank": rank,
                "batch_index": batch_index,
                "sample_key": str(sample_key),
                "task": task,
                "identity_keys": list(identity_keys),
                "source_time": visual_source_time,
                "source_side": "posterior",
                "source_phase": visual_source_phase,
                "binding_start_phase": row_binding_start_phase,
                "source_binding_valid": row_source_binding_valid,
                "row_to_track": source_row_to_track,
                "sequence_row_to_track": row_to_track,
                "row_existence": [round(float(value), 7) for value in existence],
                "row_task_relevance": [round(float(value), 7) for value in relevance],
                "row_matched_soft_iou": matched_soft_iou,
                "anchor_surface": anchor_surface,
                "views": view_metadata,
                "loss_only_labels_visible_to_model": False,
            }
        )
    return artifacts

"""Loss-only CALVIN targets aligned to LingBot's exact Qwen visual prefix.

Physical labels are resolved after the official forward.  They validate the
source sensor hashes and Qwen address geometry, but never enter model inputs,
row state, attention masks, or prediction queries.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import numpy as np
import torch

from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_target_request import NativeCALVINStructuralTargetRequest
from picf_next.data.lingbot_calvin_projection import LINGBOT_CALVIN_CAMERA_SLOTS
from picf_next.data.qwen3vl_raster import project_qwen3vl_segmentation
from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_MASK_LAYOUT,
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
    NO_RELATION_TARGET,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
)

TaskIdentityResolver = Callable[[str], tuple[str, ...] | None]

_CALVIN_VJEPA21_LAYOUT = "vjepa21.calvin.static-gripper.24x24.v1"
_CALVIN_VJEPA21_GRID_SIZE = 24
_CALVIN_VJEPA21_PATCH_SIZE = 16
_CALVIN_VJEPA21_VIEW_NAMES = ("static", "gripper")
_CALVIN_VIDEOMT_CANONICAL_GRID = (120, 120)
_CALVIN_VIDEOMT_PATCH_SIZE = 4


@dataclass(frozen=True, slots=True)
class NativeCALVINSequenceTargetBundle:
    """Structural tensors and their exact loss-side source-track ordering."""

    targets: NativeSequenceTargets
    identity_keys_by_batch: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.targets, NativeSequenceTargets):
            raise TypeError("native CALVIN target bundle requires sequence targets")
        batch, _time, tracks, _tokens = self.targets.masks.shape
        if (
            not isinstance(self.identity_keys_by_batch, tuple)
            or len(self.identity_keys_by_batch) != batch
        ):
            raise ValueError("native CALVIN target identities must match the target batch")
        for batch_index, identity_keys in enumerate(self.identity_keys_by_batch):
            if (
                not isinstance(identity_keys, tuple)
                or not identity_keys
                or len(identity_keys) > tracks
                or len(set(identity_keys)) != len(identity_keys)
                or any(not isinstance(key, str) or not key for key in identity_keys)
            ):
                raise ValueError("native CALVIN target identities must be non-empty and unique")
            expected_valid = torch.arange(
                tracks,
                device=self.targets.track_valid.device,
            ) < len(identity_keys)
            if not torch.equal(self.targets.track_valid[batch_index], expected_valid):
                raise ValueError("native CALVIN target identities differ from track validity")


def stack_native_sequence_predictions(
    relations: Sequence[RelationOutput],
) -> NativeSequencePredictions:
    """Stack a local branch; task retrieval is read from its final state."""

    if not relations or any(not isinstance(value, RelationOutput) for value in relations):
        raise TypeError("native sequence predictions require non-empty relation outputs")
    first = relations[0]
    batch, tokens, rows = first.support_logits.shape
    factorized_fields = (
        "task_object_log_probability",
        "task_object_probability",
        "task_event_distribution",
        "task_row_probability",
    )
    factorized_available = tuple(getattr(first, field) is not None for field in factorized_fields)
    ownership_log_available = first.ownership_log_probability is not None
    if any(factorized_available) and not all(factorized_available):
        raise ValueError("native relation has an incomplete factorized relation contract")
    for relation in relations:
        if relation.support_logits.shape != (batch, tokens, rows):
            raise ValueError("native relation support shapes differ across time")
        if relation.ownership.shape != (batch, tokens, rows + 1):
            raise ValueError("native relation ownership shapes differ across time")
        if (relation.ownership_log_probability is not None) != ownership_log_available:
            raise ValueError("native ownership log-probability contracts differ across time")
        if ownership_log_available and cast(
            torch.Tensor,
            relation.ownership_log_probability,
        ).shape != (batch, tokens, rows + 1):
            raise ValueError("native ownership log-probability shapes differ across time")
        if relation.existence_logits.shape != (batch, rows):
            raise ValueError("native relation existence shapes differ across time")
        if relation.task_relevance_logits.shape != (batch, rows):
            raise ValueError("native relation task shapes differ across time")
        if relation.dense_task_grounding_logits.shape != (batch, tokens):
            raise ValueError("native relation dense task shapes differ across time")
        relation_factorized_available = tuple(
            getattr(relation, field) is not None for field in factorized_fields
        )
        if relation_factorized_available != factorized_available:
            raise ValueError("native factorized relation contracts differ across time")
        if all(factorized_available):
            task_object_log_probability = cast(
                torch.Tensor,
                relation.task_object_log_probability,
            )
            task_object_probability = cast(torch.Tensor, relation.task_object_probability)
            task_event_distribution = cast(torch.Tensor, relation.task_event_distribution)
            task_row_probability = cast(torch.Tensor, relation.task_row_probability)
            if (
                task_object_log_probability.shape != (batch, tokens, rows)
                or task_object_probability.shape != (batch, tokens, rows)
                or task_event_distribution.shape != (batch, tokens, rows + 1)
                or task_row_probability.shape != (batch, rows)
            ):
                raise ValueError("native factorized relation shapes differ across time")
    return NativeSequencePredictions(
        support_logits=torch.stack(tuple(value.support_logits for value in relations), dim=1),
        ownership=torch.stack(tuple(value.ownership for value in relations), dim=1),
        existence_logits=torch.stack(tuple(value.existence_logits for value in relations), dim=1),
        task_relevance_logits=relations[-1].task_relevance_logits,
        dense_task_grounding_logits=torch.stack(
            tuple(value.dense_task_grounding_logits for value in relations),
            dim=1,
        ),
        ownership_log_probability=(
            torch.stack(
                tuple(cast(torch.Tensor, value.ownership_log_probability) for value in relations),
                dim=1,
            )
            if ownership_log_available
            else None
        ),
        task_object_log_probability=(
            torch.stack(
                tuple(cast(torch.Tensor, value.task_object_log_probability) for value in relations),
                dim=1,
            )
            if all(factorized_available)
            else None
        ),
        task_object_probability=(
            torch.stack(
                tuple(cast(torch.Tensor, value.task_object_probability) for value in relations),
                dim=1,
            )
            if all(factorized_available)
            else None
        ),
        task_event_distribution=(
            torch.stack(
                tuple(cast(torch.Tensor, value.task_event_distribution) for value in relations),
                dim=1,
            )
            if all(factorized_available)
            else None
        ),
        task_row_probability=(
            relations[-1].task_row_probability if all(factorized_available) else None
        ),
        task_row_probability_by_time=(
            torch.stack(
                tuple(cast(torch.Tensor, value.task_row_probability) for value in relations),
                dim=1,
            )
            if all(factorized_available)
            else None
        ),
    )


def _validate_source_frame(
    frame: CalvinPhysicalSupervisionFrame,
    request: NativeCALVINStructuralTargetRequest,
) -> dict[str, Any]:
    if not isinstance(frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("CALVIN physical sidecar returned an invalid frame")
    if not frame.identity_keys or len(set(frame.identity_keys)) != len(frame.identity_keys):
        raise ValueError("CALVIN physical frame identities must be non-empty and unique")
    camera_by_name = {camera.camera_name: camera for camera in frame.cameras}
    expected_physical_cameras = {
        slot.physical_camera_name
        for slot in LINGBOT_CALVIN_CAMERA_SLOTS
        if slot.physical_camera_name is not None
    }
    if len(camera_by_name) != len(frame.cameras) or set(camera_by_name) != (
        expected_physical_cameras
    ):
        raise ValueError("CALVIN physical cameras differ from the frozen LingBot views")
    hashes = request.source_sensor_hash_by_field
    spec_by_name = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    for camera_name, camera in camera_by_name.items():
        spec = spec_by_name[camera_name]
        if camera.source_rgb_sha256 != hashes[str(spec["source_rgb_field"])]:
            raise ValueError(f"CALVIN {camera_name} RGB supervision source hash differs")
        if camera.source_depth_sha256 != hashes[str(spec["source_depth_field"])]:
            raise ValueError(f"CALVIN {camera_name} depth supervision source hash differs")
    return camera_by_name


def _capacity_censor(
    identity_keys: tuple[str, ...],
    *,
    capacity: int,
    seed: int | None,
) -> np.ndarray:
    censored = np.zeros(len(identity_keys), dtype=np.bool_)
    if len(identity_keys) <= capacity:
        return censored
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("overflowing native object sets require one uint64 capacity seed")
    prefix = struct.pack("<Q", seed)
    ranked = sorted(
        range(len(identity_keys)),
        key=lambda index: hashlib.sha256(prefix + identity_keys[index].encode("utf-8")).digest(),
    )
    censored[np.asarray(ranked[capacity:], dtype=np.int64)] = True
    return censored


def _validate_qwen_frame_inputs(
    model_inputs: Mapping[str, Any],
    relation: RelationOutput | PhysicalRelationOutput,
    *,
    batch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(model_inputs, Mapping):
        raise TypeError("native CALVIN target projection requires official model inputs")
    images = model_inputs.get("images")
    image_valid = model_inputs.get("img_masks")
    grids = model_inputs.get("image_grid_thw")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("official Qwen images must be [batch,views,patches,width]")
    expected_views = len(LINGBOT_CALVIN_CAMERA_SLOTS)
    if images.shape[:2] != (batch_size, expected_views):
        raise ValueError("official Qwen image views differ from the frozen CALVIN mapping")
    if (
        not isinstance(image_valid, torch.Tensor)
        or image_valid.shape != (batch_size, expected_views)
        or image_valid.dtype != torch.bool
    ):
        raise ValueError("official Qwen image validity differs from the CALVIN views")
    if (
        not isinstance(grids, torch.Tensor)
        or grids.shape != (batch_size, expected_views, 3)
        or grids.dtype != torch.long
    ):
        raise ValueError("official Qwen image grids differ from the CALVIN views")
    if any(value.device != relation.sensor_valid.device for value in (images, image_valid, grids)):
        raise ValueError("Qwen frame inputs and relation outputs must share one device")
    expected_valid = torch.tensor(
        [slot.valid for slot in LINGBOT_CALVIN_CAMERA_SLOTS],
        dtype=torch.bool,
        device=image_valid.device,
    ).expand(batch_size, -1)
    if not torch.equal(image_valid, expected_valid):
        raise ValueError("official Qwen image validity differs from the frozen CALVIN mapping")
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
            raise ValueError("official Qwen padded image grid differs from its source view")
    raw_counts = grids.prod(dim=-1)
    merge_unit = merge_size**2
    if (raw_counts != images.shape[2]).any() or (raw_counts % merge_unit).any():
        raise ValueError("Qwen packed patches differ from the declared merger geometry")
    expected_sensor_count = ((raw_counts // merge_unit) * image_valid).sum(dim=1)
    observed_sensor_count = relation.structural_valid.sum(dim=1)
    if not torch.equal(expected_sensor_count, observed_sensor_count):
        raise ValueError("Qwen visual-token count differs from LingBot sensor roles")
    return images, image_valid, grids


def _supervised_relation_surfaces(
    relation: RelationOutput | PhysicalRelationOutput,
) -> tuple[Any, ...]:
    if not isinstance(relation, PhysicalRelationOutput):
        return ()
    return tuple(
        surface
        for surface in relation.relation_surfaces
        if surface.target_kind != NO_RELATION_TARGET
    )


def _validate_supervised_surface_sequence(
    relations: Sequence[RelationOutput | PhysicalRelationOutput],
) -> tuple[tuple[str, str, str, int, tuple[int, int] | None], ...]:
    first = _supervised_relation_surfaces(relations[0])
    layout = tuple(
        (
            surface.name,
            surface.target_kind,
            surface.layout,
            surface.support_logits.shape[1],
            surface.grid_shape,
        )
        for surface in first
    )
    for relation in relations:
        surfaces = _supervised_relation_surfaces(relation)
        observed = tuple(
            (
                surface.name,
                surface.target_kind,
                surface.layout,
                surface.support_logits.shape[1],
                surface.grid_shape,
            )
            for surface in surfaces
        )
        if observed != layout:
            raise ValueError("supervised native relation surfaces differ across target time")
        for surface in surfaces:
            if surface.support_logits.shape[0] != relation.support_logits.shape[0]:
                raise ValueError("native relation surface batch differs from its host relation")
            if surface.support_logits.shape[2] != relation.support_logits.shape[2]:
                raise ValueError("native relation surface row capacity differs from its host")
            if surface.target_kind not in {
                CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
                CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
            }:
                raise ValueError("CALVIN target builder received an unsupported native surface")
            if surface.target_kind == CALVIN_VJEPA21_VISIBLE_OWNER_TARGET:
                if (
                    surface.name != "vjepa"
                    or surface.geometry_kind != "image_grid"
                    or surface.layout != _CALVIN_VJEPA21_LAYOUT
                    or surface.support_logits.shape[1]
                    != len(_CALVIN_VJEPA21_VIEW_NAMES) * _CALVIN_VJEPA21_GRID_SIZE**2
                ):
                    raise ValueError("CALVIN V-JEPA relation surface violates its frozen layout")
            elif (
                surface.name != "videomt_masks"
                or surface.geometry_kind != "image_grid"
                or surface.layout != CALVIN_VIDEOMT_MASK_LAYOUT
                or surface.grid_shape != _CALVIN_VIDEOMT_CANONICAL_GRID
                or surface.support_logits.shape[1] != 120 * 120
            ):
                raise ValueError(
                    "CALVIN VidEoMT sidecar targets require the canonical aligned grid"
                )
    return layout


def build_native_calvin_sequence_target_bundle(
    *,
    requests_by_time: Sequence[Sequence[NativeCALVINStructuralTargetRequest]],
    model_inputs_by_time: Sequence[Mapping[str, Any]],
    relations: Sequence[RelationOutput | PhysicalRelationOutput],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver | None,
    patch_size: int,
    merge_size: int,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
) -> NativeCALVINSequenceTargetBundle:
    """Build exact full-prefix targets and preserve source-track identities."""

    if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
        raise TypeError("native CALVIN targets require a verified physical sidecar")
    if task_identity_resolver is not None and not callable(task_identity_resolver):
        raise TypeError("native CALVIN task identity resolver must be callable or absent")
    if not requests_by_time or not (
        len(requests_by_time) == len(model_inputs_by_time) == len(relations)
    ):
        raise ValueError("native CALVIN target sequence inputs must be equal and non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (capacity, patch_size, merge_size)
    ):
        raise ValueError("native CALVIN capacity and Qwen geometry must be positive integers")
    if (
        isinstance(minimum_supervised_fraction, bool)
        or not isinstance(minimum_supervised_fraction, Real)
        or not np.isfinite(minimum_supervised_fraction)
        or not 0.0 <= minimum_supervised_fraction <= 1.0
    ):
        raise ValueError("minimum supervised fraction must lie in [0,1]")

    batch_size = len(requests_by_time[0])
    if batch_size == 0 or any(len(requests) != batch_size for requests in requests_by_time):
        raise ValueError("native CALVIN target time slices must share a non-empty batch")
    if any(
        not isinstance(request, NativeCALVINStructuralTargetRequest)
        for requests in requests_by_time
        for request in requests
    ):
        raise TypeError("native CALVIN target sequence contains an invalid request")
    if capacity_seeds is None:
        resolved_capacity_seeds: tuple[int | None, ...] = (None,) * batch_size
    else:
        resolved_capacity_seeds = tuple(capacity_seeds)
        if len(resolved_capacity_seeds) != batch_size:
            raise ValueError("capacity seeds must provide one value per sequence")

    time_count = len(relations)
    prefix_tokens = relations[0].support_logits.shape[1]
    supervised_surface_layout = _validate_supervised_surface_sequence(relations)
    target_tokens = prefix_tokens + sum(item[3] for item in supervised_surface_layout)
    device = relations[0].support_logits.device
    frame_records: list[list[tuple[CalvinPhysicalSupervisionFrame, dict[str, Any]]]] = [
        [] for _ in range(batch_size)
    ]
    for time_index, (requests, model_inputs, relation) in enumerate(
        zip(requests_by_time, model_inputs_by_time, relations, strict=True)
    ):
        if relation.support_logits.shape[:2] != (batch_size, prefix_tokens):
            raise ValueError("native relation prefix shapes differ across target time")
        _images, image_valid, grids = _validate_qwen_frame_inputs(
            model_inputs,
            relation,
            batch_size=batch_size,
            merge_size=merge_size,
        )
        for batch_index, request in enumerate(requests):
            if time_index and (
                request.episode_key != requests_by_time[0][batch_index].episode_key
                or request.task_key != requests_by_time[0][batch_index].task_key
            ):
                raise ValueError("one native target branch must retain episode and task identity")
            frame = physical_sidecar(request.segment_index, request.source_global_index)
            camera_by_name = _validate_source_frame(frame, request)
            frame_records[batch_index].append((frame, camera_by_name))
        del image_valid, grids

    identity_keys_by_batch: list[tuple[str, ...]] = []
    for records in frame_records:
        ordered: list[str] = []
        seen: set[str] = set()
        for frame, _cameras in records:
            for identity_key in frame.identity_keys:
                if identity_key not in seen:
                    seen.add(identity_key)
                    ordered.append(identity_key)
        if not ordered:
            raise ValueError("native CALVIN target branch has no physical inventory")
        identity_keys_by_batch.append(tuple(ordered))
    track_count = max(len(keys) for keys in identity_keys_by_batch)

    masks = torch.zeros(batch_size, time_count, track_count, target_tokens, device=device)
    mask_valid = torch.zeros_like(masks, dtype=torch.bool)
    existence = torch.zeros(batch_size, time_count, track_count, device=device)
    existence_valid = torch.zeros_like(existence, dtype=torch.bool)
    task_relevance = torch.zeros(batch_size, track_count, device=device)
    task_valid = torch.zeros_like(task_relevance, dtype=torch.bool)
    track_valid = torch.zeros(batch_size, track_count, dtype=torch.bool, device=device)
    capacity_censored = torch.zeros_like(track_valid)
    token_observed_fraction = torch.zeros(
        batch_size, time_count, target_tokens, dtype=torch.float32, device=device
    )
    token_measure_weight = torch.zeros_like(token_observed_fraction)
    inventory_exhaustive = torch.ones(batch_size, time_count, dtype=torch.bool, device=device)

    for batch_index, identity_keys in enumerate(identity_keys_by_batch):
        identity_to_track = {key: index for index, key in enumerate(identity_keys)}
        valid_tracks = len(identity_keys)
        track_valid[batch_index, :valid_tracks] = True
        censored = _capacity_censor(
            identity_keys,
            capacity=capacity,
            seed=resolved_capacity_seeds[batch_index],
        )
        capacity_censored[batch_index, :valid_tracks] = torch.from_numpy(censored).to(device)

        resolved_task = (
            None
            if task_identity_resolver is None
            else task_identity_resolver(requests_by_time[0][batch_index].task_key)
        )
        if resolved_task is not None:
            if (
                not isinstance(resolved_task, tuple)
                or not resolved_task
                or any(not isinstance(key, str) or not key for key in resolved_task)
                or len(set(resolved_task)) != len(resolved_task)
            ):
                raise ValueError("task identity resolver returned an invalid exact target")
            missing = sorted(set(resolved_task) - set(identity_keys))
            if missing:
                raise ValueError(f"exact CALVIN task targets are absent from inventory: {missing}")
            task_valid[batch_index, :valid_tracks] = True
            for key in resolved_task:
                task_relevance[batch_index, identity_to_track[key]] = 1.0

        for time_index, (frame, camera_by_name) in enumerate(frame_records[batch_index]):
            present = {identity_to_track[key] for key in frame.identity_keys}
            existence_valid[batch_index, time_index, :valid_tracks] = True
            if present:
                existence[batch_index, time_index, list(present)] = 1.0

            relation = relations[time_index]
            sensor_positions = relation.structural_valid[batch_index].nonzero().flatten()
            image_valid = model_inputs_by_time[time_index]["img_masks"][batch_index]
            grids = model_inputs_by_time[time_index]["image_grid_thw"][batch_index]
            sensor_offset = 0
            for view_index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS):
                if not bool(image_valid[view_index].item()):
                    continue
                camera_name = slot.physical_camera_name
                if camera_name is None:
                    raise RuntimeError("an invalid CALVIN camera slot reached target projection")
                grid = grids[view_index].detach().cpu().numpy()
                projected = project_qwen3vl_segmentation(
                    camera_by_name[camera_name].owner_index,
                    instance_ids=tuple(range(1, len(frame.identity_keys) + 1)),
                    image_grid_thw=grid,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    pixel_supervised=camera_by_name[camera_name].owner_supervised,
                    minimum_supervised_fraction=float(minimum_supervised_fraction),
                ).merged
                view_tokens = projected.supervised.shape[0]
                positions = sensor_positions[sensor_offset : sensor_offset + view_tokens]
                if positions.numel() != view_tokens:
                    raise RuntimeError("Qwen sensor-role partition ended inside a CALVIN view")
                supervised = torch.from_numpy(projected.supervised).to(device)
                observed_fraction = torch.from_numpy(projected.observed_fraction).to(device)
                token_observed_fraction[
                    batch_index,
                    time_index,
                    positions,
                ] = observed_fraction
                token_measure_weight[
                    batch_index,
                    time_index,
                    positions,
                ] = 1.0 / view_tokens
                mask_valid[
                    batch_index,
                    time_index,
                    :valid_tracks,
                    positions,
                ] = supervised.unsqueeze(0)
                for probability_column, owner_index in enumerate(projected.instance_ids):
                    identity_key = frame.identity_keys[owner_index - 1]
                    track_index = identity_to_track[identity_key]
                    probability = torch.from_numpy(
                        projected.object_probability[:, probability_column]
                    ).to(device)
                    masks[batch_index, time_index, track_index, positions] = probability
                sensor_offset += view_tokens
            if sensor_offset != sensor_positions.numel():
                raise RuntimeError("CALVIN view targets did not consume every Qwen sensor token")

            target_offset = prefix_tokens
            for surface in _supervised_relation_surfaces(relation):
                surface_count = surface.support_logits.shape[1]
                surface_valid = surface.sensor_valid[batch_index]
                if surface.target_kind == CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET:
                    if surface.canonical_token_ids is not None:
                        raise ValueError(
                            "the dense VidEoMT pixel surface must not expose token ids"
                        )
                    if surface.grid_shape is None:
                        raise ValueError("the dense VidEoMT surface omitted its runtime grid")
                    grid_height, grid_width = surface.grid_shape
                    if surface_valid.shape != (grid_height * grid_width,):
                        raise ValueError("CALVIN VidEoMT pixel validity has an invalid shape")
                    projected = project_qwen3vl_segmentation(
                        camera_by_name["static"].owner_index,
                        instance_ids=tuple(range(1, len(frame.identity_keys) + 1)),
                        image_grid_thw=np.asarray(
                            [1, grid_height, grid_width],
                            dtype=np.int64,
                        ),
                        patch_size=_CALVIN_VIDEOMT_PATCH_SIZE,
                        merge_size=1,
                        pixel_supervised=camera_by_name["static"].owner_supervised,
                        minimum_supervised_fraction=float(minimum_supervised_fraction),
                    ).merged
                    positions = torch.arange(
                        target_offset,
                        target_offset + surface_count,
                        dtype=torch.long,
                        device=device,
                    )
                    valid_tokens = torch.from_numpy(projected.supervised).to(device) & surface_valid
                    observed_fraction = torch.from_numpy(projected.observed_fraction).to(device)
                    observed_fraction = observed_fraction * surface_valid.to(
                        observed_fraction.dtype
                    )
                    token_observed_fraction[batch_index, time_index, positions] = (
                        observed_fraction
                    )
                    token_measure_weight[batch_index, time_index, positions] = (
                        1.0 / surface_count
                    )
                    mask_valid[
                        batch_index,
                        time_index,
                        :valid_tracks,
                        positions,
                    ] = valid_tokens.unsqueeze(0)
                    for probability_column, owner_index in enumerate(projected.instance_ids):
                        identity_key = frame.identity_keys[owner_index - 1]
                        track_index = identity_to_track[identity_key]
                        probability = torch.from_numpy(
                            projected.object_probability[:, probability_column]
                        ).to(device)
                        masks[
                            batch_index,
                            time_index,
                            track_index,
                            positions,
                        ] = probability * surface_valid.to(probability.dtype)
                    target_offset += surface_count
                    continue

                canonical_ids = surface.canonical_token_ids
                if canonical_ids is None:
                    raise ValueError("supervised V-JEPA surface omitted canonical token ids")
                observed_ids = canonical_ids[batch_index].masked_select(surface_valid)
                expected_ids = torch.arange(
                    observed_ids.numel(),
                    dtype=torch.long,
                    device=observed_ids.device,
                )
                if not torch.equal(observed_ids, expected_ids):
                    raise ValueError("supervised V-JEPA rows are not in canonical encoder order")

                view_token_count = _CALVIN_VJEPA21_GRID_SIZE**2
                for view_index, camera_name in enumerate(_CALVIN_VJEPA21_VIEW_NAMES):
                    local_start = view_index * view_token_count
                    local_stop = local_start + view_token_count
                    view_valid = surface_valid[local_start:local_stop]
                    if view_valid.any() and not view_valid.all():
                        raise ValueError("one V-JEPA image grid is partially available")
                    projected = project_qwen3vl_segmentation(
                        camera_by_name[camera_name].owner_index,
                        instance_ids=tuple(range(1, len(frame.identity_keys) + 1)),
                        image_grid_thw=np.asarray(
                            [1, _CALVIN_VJEPA21_GRID_SIZE, _CALVIN_VJEPA21_GRID_SIZE],
                            dtype=np.int64,
                        ),
                        patch_size=_CALVIN_VJEPA21_PATCH_SIZE,
                        merge_size=1,
                        pixel_supervised=camera_by_name[camera_name].owner_supervised,
                        minimum_supervised_fraction=float(minimum_supervised_fraction),
                    ).merged
                    positions = torch.arange(
                        target_offset + local_start,
                        target_offset + local_stop,
                        dtype=torch.long,
                        device=device,
                    )
                    valid_tokens = torch.from_numpy(projected.supervised).to(device) & view_valid
                    observed_fraction = torch.from_numpy(projected.observed_fraction).to(device)
                    observed_fraction = observed_fraction * view_valid.to(observed_fraction.dtype)
                    token_observed_fraction[
                        batch_index,
                        time_index,
                        positions,
                    ] = observed_fraction
                    token_measure_weight[
                        batch_index,
                        time_index,
                        positions,
                    ] = 1.0 / view_token_count
                    mask_valid[
                        batch_index,
                        time_index,
                        :valid_tracks,
                        positions,
                    ] = valid_tokens.unsqueeze(0)
                    for probability_column, owner_index in enumerate(projected.instance_ids):
                        identity_key = frame.identity_keys[owner_index - 1]
                        track_index = identity_to_track[identity_key]
                        probability = torch.from_numpy(
                            projected.object_probability[:, probability_column]
                        ).to(device)
                        masks[
                            batch_index,
                            time_index,
                            track_index,
                            positions,
                        ] = probability * view_valid.to(probability.dtype)
                target_offset += surface_count
            if target_offset != target_tokens:
                raise RuntimeError("native relation targets did not consume every surface token")

    return NativeCALVINSequenceTargetBundle(
        targets=NativeSequenceTargets(
            masks=masks,
            mask_valid=mask_valid,
            existence=existence,
            existence_valid=existence_valid,
            task_relevance=task_relevance,
            task_valid=task_valid,
            track_valid=track_valid,
            capacity_censored=capacity_censored,
            token_observed_fraction=token_observed_fraction,
            inventory_exhaustive=inventory_exhaustive,
            token_measure_weight=token_measure_weight,
            exclusive_ownership=True,
        ),
        identity_keys_by_batch=tuple(identity_keys_by_batch),
    )

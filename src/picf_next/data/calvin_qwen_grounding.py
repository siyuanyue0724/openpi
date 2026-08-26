"""Loss-only CALVIN records for Qwen's native grounding objective."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
)
from picf_next.data.qwen3vl_raster import project_qwen3vl_segmentation
from picf_next.eval.calvin_task_relevance import calvin_exact_task_loss_identities

QWEN_GROUNDING_REQUEST = (
    "Task: {instruction}\n"
    "Locate the physical object that must be directly manipulated to execute this task. "
    'Return its object label and bounding box as a JSON list with keys "label" and "bbox_2d".'
)
QWEN_SCENE_GROUNDING_REQUEST = (
    "Locate every visible instance that belongs to the following categories: "
    '"{categories}". Return each object label and bounding box as a JSON list with keys '
    '"label" and "bbox_2d" in the same category order; skip categories with no visible '
    "instance."
)
QWEN3VL_COORDINATE_SCALE = 1000
QWEN3VL_PATCH_SIZE = 16
QWEN3VL_SPATIAL_MERGE_SIZE = 2

# Dataset-side names for Qwen3-VL's native grounding schema. These labels are
# assistant-only supervision; physical identity keys never enter model inputs.
_QWEN_GROUNDING_LABEL_BY_IDENTITY = {
    "movable/block_blue": "blue block",
    "movable/block_pink": "pink block",
    "movable/block_red": "red block",
    "part/table/button_link": "push button",
    "part/table/drawer_link": "drawer",
    "part/table/led_link": "LED indicator",
    "part/table/light_link": "light bulb",
    "part/table/plank_link": "slider surface",
    "part/table/slide_link": "sliding door",
    "part/table/switch_link": "light switch",
}
CALVIN_QWEN_SCENE_IDENTITY_ORDER = tuple(_QWEN_GROUNDING_LABEL_BY_IDENTITY)


class _CameraSpec(TypedDict):
    camera_name: str
    host_image_key: str
    height: int
    width: int
    source_rgb_field: str


def _camera_spec(camera_name: str) -> _CameraSpec:
    matches = tuple(spec for spec in CALVIN_CAMERA_SPECS if spec["camera_name"] == camera_name)
    if len(matches) != 1:
        raise ContractError(f"unknown CALVIN grounding camera: {camera_name!r}")
    return cast(_CameraSpec, matches[0])


def _immutable_rgb_copy(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    result = np.ascontiguousarray(image).copy()
    result.setflags(write=False)
    return result


def tight_visible_owner_bbox(owner_support: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Return a tight half-open bbox around depth-verified visible owner pixels."""

    support = np.asarray(owner_support)
    if support.dtype != np.bool_ or support.ndim != 2:
        raise ContractError("visible owner support must be a two-dimensional boolean array")
    rows, columns = np.nonzero(support)
    if rows.size == 0:
        raise ContractError("visible owner support cannot be empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def qwen3vl_normalized_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Map a half-open pixel box to Qwen3-VL's relative 1000x1000 grid."""

    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ContractError("Qwen3-VL normalization requires positive image dimensions")
    if (
        not isinstance(bbox_xyxy, tuple)
        or len(bbox_xyxy) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox_xyxy)
    ):
        raise ContractError("Qwen3-VL normalization requires a four-integer bbox")
    x_min, y_min, x_max, y_max = bbox_xyxy
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ContractError("Qwen3-VL normalization bbox lies outside its source image")
    normalized_values = tuple(
        int(round(value / extent * QWEN3VL_COORDINATE_SCALE))
        for value, extent in zip(
            bbox_xyxy,
            (width, height, width, height),
            strict=True,
        )
    )
    normalized = (
        normalized_values[0],
        normalized_values[1],
        normalized_values[2],
        normalized_values[3],
    )
    nx_min, ny_min, nx_max, ny_max = normalized
    if not (
        0 <= nx_min < nx_max <= QWEN3VL_COORDINATE_SCALE
        and 0 <= ny_min < ny_max <= QWEN3VL_COORDINATE_SCALE
    ):
        raise ContractError("Qwen3-VL normalization collapsed a positive-area bbox")
    return normalized


def qwen_grounding_label(identity_key: str) -> str:
    """Return one reviewed assistant-only natural-language physical label."""

    if not isinstance(identity_key, str) or not identity_key:
        raise ContractError("CALVIN grounding identity key must be nonempty text")
    try:
        return _QWEN_GROUNDING_LABEL_BY_IDENTITY[identity_key]
    except KeyError as error:
        raise ContractError(
            "CALVIN grounding target lacks a reviewed natural-language label"
        ) from error


def minimum_projected_target_mass_for_raw_patch(*, merge_size: int) -> float:
    """Return one raw patch's exact mass in the merged-token measure."""

    if isinstance(merge_size, bool) or not isinstance(merge_size, int) or merge_size <= 0:
        raise ContractError("Qwen scene merge size must be a positive integer")
    return 1.0 / (merge_size**2)


@dataclass(frozen=True, slots=True)
class CalvinQwenGroundingRecord:
    """One auditable label-side target with one ordinary deploy-visible image."""

    global_index: int
    task_key: str
    instruction: str
    target_identity_key: str
    camera_name: str
    host_image_key: str
    bbox_xyxy: tuple[int, int, int, int]
    image: NDArray[np.uint8]
    source_rgb_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_index, bool)
            or not isinstance(self.global_index, int)
            or self.global_index < 0
        ):
            raise ContractError("CALVIN grounding source index must be non-negative")
        text_fields = (
            self.task_key,
            self.instruction,
            self.target_identity_key,
            self.camera_name,
            self.host_image_key,
        )
        if any(not isinstance(value, str) or not value for value in text_fields):
            raise ContractError("CALVIN grounding identity and text fields must be nonempty")
        exact_targets = calvin_exact_task_loss_identities(self.task_key)
        if exact_targets != (self.target_identity_key,):
            raise ContractError("CALVIN grounding record differs from the exact task target")
        spec = _camera_spec(self.camera_name)
        if self.host_image_key != spec["host_image_key"]:
            raise ContractError("CALVIN grounding host image key differs from its camera")
        height = int(spec["height"])
        width = int(spec["width"])
        if (
            not isinstance(self.image, np.ndarray)
            or self.image.dtype != np.uint8
            or self.image.shape != (height, width, 3)
            or self.image.flags.writeable
        ):
            raise ContractError("CALVIN grounding image must be immutable HWC uint8 RGB")
        if (
            not isinstance(self.bbox_xyxy, tuple)
            or len(self.bbox_xyxy) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in self.bbox_xyxy
            )
        ):
            raise ContractError("CALVIN grounding bbox must contain four integers")
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ContractError("CALVIN grounding bbox lies outside the source image")
        expected_digest = source_array_sha256(str(spec["source_rgb_field"]), self.image)
        if self.source_rgb_sha256 != expected_digest:
            raise ContractError("CALVIN grounding image differs from its physical sidecar")

    @property
    def qwen_bbox_xyxy(self) -> tuple[int, int, int, int]:
        spec = _camera_spec(self.camera_name)
        return qwen3vl_normalized_bbox(
            self.bbox_xyxy,
            width=int(spec["width"]),
            height=int(spec["height"]),
        )

    @property
    def qwen_label(self) -> str:
        return qwen_grounding_label(self.target_identity_key)

    @property
    def assistant_text(self) -> str:
        return json.dumps(
            [{"label": self.qwen_label, "bbox_2d": list(self.qwen_bbox_xyxy)}],
            ensure_ascii=True,
            separators=(",", ":"),
        )

    @property
    def grounding_request(self) -> str:
        return QWEN_GROUNDING_REQUEST.format(instruction=self.instruction)

    def qwen_user_messages(self, image_value: object | None = None) -> list[dict[str, Any]]:
        """Return the deploy-visible Qwen request without a teacher-forced answer."""

        visible_image = self.image if image_value is None else image_value
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": visible_image},
                    {"type": "text", "text": self.grounding_request},
                ],
            }
        ]

    def qwen_messages(self, image_value: object | None = None) -> list[dict[str, Any]]:
        """Return only deploy-visible image/text and the assistant answer."""

        return [
            *self.qwen_user_messages(image_value=image_value),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.assistant_text}],
            },
        ]


def build_calvin_qwen_grounding_records(
    *,
    global_index: int,
    task_key: str,
    instruction: str,
    observation_images: Mapping[str, NDArray[np.uint8]],
    physical_frame: CalvinPhysicalSupervisionFrame,
) -> tuple[CalvinQwenGroundingRecord, ...]:
    """Build one native-Qwen record per camera with visible exact target support."""

    if not isinstance(physical_frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("CALVIN grounding requires a physical supervision frame")
    if not isinstance(instruction, str) or not instruction:
        raise ContractError("CALVIN grounding requires the complete natural instruction")
    target_identities = calvin_exact_task_loss_identities(task_key)
    if target_identities is None:
        return ()
    if len(target_identities) != 1:
        raise ContractError("native Qwen grounding currently requires one exact action target")
    target_identity = target_identities[0]
    try:
        target_owner_id = physical_frame.identity_keys.index(target_identity) + 1
    except ValueError as error:
        raise ContractError("exact task target is absent from the physical sidecar") from error

    records = []
    for camera in physical_frame.cameras:
        if camera.host_image_key not in observation_images:
            raise ContractError("CALVIN grounding observation omits a sidecar camera")
        image = np.asarray(observation_images[camera.host_image_key])
        spec = _camera_spec(camera.camera_name)
        expected_shape = (int(spec["height"]), int(spec["width"]), 3)
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ContractError("CALVIN grounding source image shape or dtype is invalid")
        digest = source_array_sha256(str(spec["source_rgb_field"]), image)
        if digest != camera.source_rgb_sha256:
            raise ContractError("CALVIN grounding source image hash differs from sidecar")
        visible = (camera.owner_index == target_owner_id) & camera.owner_supervised
        if not np.any(visible):
            continue
        records.append(
            CalvinQwenGroundingRecord(
                global_index=global_index,
                task_key=task_key,
                instruction=instruction,
                target_identity_key=target_identity,
                camera_name=camera.camera_name,
                host_image_key=camera.host_image_key,
                bbox_xyxy=tight_visible_owner_bbox(visible),
                image=_immutable_rgb_copy(image),
                source_rgb_sha256=digest,
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class CalvinQwenSceneObject:
    """One observable physical object in an assistant-only scene answer."""

    identity_key: str
    bbox_xyxy: tuple[int, int, int, int]
    visible_owner_pixels: int
    projected_target_mass: float
    positive_visual_token_count: int

    def __post_init__(self) -> None:
        qwen_grounding_label(self.identity_key)
        if (
            not isinstance(self.bbox_xyxy, tuple)
            or len(self.bbox_xyxy) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in self.bbox_xyxy
            )
        ):
            raise ContractError("CALVIN scene object bbox must contain four integers")
        if (
            isinstance(self.visible_owner_pixels, bool)
            or not isinstance(self.visible_owner_pixels, int)
            or self.visible_owner_pixels <= 0
        ):
            raise ContractError("CALVIN scene object visible pixel count must be positive")
        if (
            isinstance(self.projected_target_mass, bool)
            or not isinstance(self.projected_target_mass, int | float)
            or not math.isfinite(float(self.projected_target_mass))
            or self.projected_target_mass < 0.0
        ):
            raise ContractError("CALVIN scene projected target mass must be finite and nonnegative")
        if (
            isinstance(self.positive_visual_token_count, bool)
            or not isinstance(self.positive_visual_token_count, int)
            or self.positive_visual_token_count < 0
        ):
            raise ContractError("CALVIN scene positive visual-token count must be nonnegative")
        if (self.projected_target_mass > 0.0) != (self.positive_visual_token_count > 0):
            raise ContractError("CALVIN scene projected mass and token support disagree")


@dataclass(frozen=True, slots=True)
class CalvinQwenSceneGroundingRecord:
    """One Qwen-native multi-object answer with no task or target input."""

    global_index: int
    camera_name: str
    host_image_key: str
    category_identity_order: tuple[str, ...]
    objects: tuple[CalvinQwenSceneObject, ...]
    subpatch_objects: tuple[CalvinQwenSceneObject, ...]
    absent_identity_keys: tuple[str, ...]
    minimum_projected_target_mass: float
    visual_lattice: int
    image_grid_thw: tuple[int, int, int]
    patch_size: int
    merge_size: int
    image: NDArray[np.uint8]
    source_rgb_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_index, bool)
            or not isinstance(self.global_index, int)
            or self.global_index < 0
        ):
            raise ContractError("CALVIN scene source index must be non-negative")
        if not isinstance(self.camera_name, str) or not isinstance(self.host_image_key, str):
            raise ContractError("CALVIN scene camera fields must be text")
        if set(self.category_identity_order) != set(CALVIN_QWEN_SCENE_IDENTITY_ORDER) or len(
            self.category_identity_order
        ) != len(CALVIN_QWEN_SCENE_IDENTITY_ORDER):
            raise ContractError("CALVIN scene categories must permute the reviewed inventory")
        object_keys = tuple(item.identity_key for item in self.objects)
        if len(set(object_keys)) != len(object_keys):
            raise ContractError("CALVIN scene answer repeats one object identity")
        expected_object_order = tuple(
            key for key in self.category_identity_order if key in set(object_keys)
        )
        if object_keys != expected_object_order:
            raise ContractError("CALVIN scene objects differ from requested category order")
        subpatch_keys = tuple(item.identity_key for item in self.subpatch_objects)
        if len(set(subpatch_keys)) != len(subpatch_keys):
            raise ContractError("CALVIN scene subpatch evidence repeats one identity")
        partitions = (set(object_keys), set(subpatch_keys), set(self.absent_identity_keys))
        if any(
            left & right
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        ):
            raise ContractError("CALVIN scene visibility partitions overlap")
        if set().union(*partitions) != set(self.category_identity_order):
            raise ContractError("CALVIN scene visibility partitions omit an identity")
        for values in (subpatch_keys, self.absent_identity_keys):
            expected = tuple(key for key in self.category_identity_order if key in set(values))
            if values != expected:
                raise ContractError("CALVIN scene visibility partition order changed")
        expected_minimum = minimum_projected_target_mass_for_raw_patch(merge_size=self.merge_size)
        if not math.isclose(
            self.minimum_projected_target_mass,
            expected_minimum,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContractError("CALVIN scene observability threshold differs from one raw patch")
        if (
            isinstance(self.visual_lattice, bool)
            or not isinstance(self.visual_lattice, int)
            or self.visual_lattice <= 0
            or self.patch_size != QWEN3VL_PATCH_SIZE
            or self.merge_size != QWEN3VL_SPATIAL_MERGE_SIZE
            or self.image_grid_thw
            != (
                1,
                self.visual_lattice * self.merge_size,
                self.visual_lattice * self.merge_size,
            )
        ):
            raise ContractError("CALVIN scene Qwen token geometry differs from its lattice")
        spec = _camera_spec(self.camera_name)
        if self.host_image_key != spec["host_image_key"]:
            raise ContractError("CALVIN scene host image key differs from its camera")
        height = int(spec["height"])
        width = int(spec["width"])
        if (
            not isinstance(self.image, np.ndarray)
            or self.image.dtype != np.uint8
            or self.image.shape != (height, width, 3)
            or self.image.flags.writeable
        ):
            raise ContractError("CALVIN scene image must be immutable HWC uint8 RGB")
        for item in self.objects:
            x_min, y_min, x_max, y_max = item.bbox_xyxy
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise ContractError("CALVIN scene object bbox lies outside the source image")
            if item.projected_target_mass + 1e-12 < self.minimum_projected_target_mass:
                raise ContractError("CALVIN scene object has subpatch projected support")
        for item in self.subpatch_objects:
            x_min, y_min, x_max, y_max = item.bbox_xyxy
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise ContractError("CALVIN scene subpatch bbox lies outside the source image")
            if item.projected_target_mass + 1e-12 >= self.minimum_projected_target_mass:
                raise ContractError("CALVIN scene subpatch evidence exceeds its threshold")
        expected_digest = source_array_sha256(str(spec["source_rgb_field"]), self.image)
        if self.source_rgb_sha256 != expected_digest:
            raise ContractError("CALVIN scene image differs from its physical sidecar")

    @property
    def assistant_text(self) -> str:
        payload = [
            {
                "label": qwen_grounding_label(item.identity_key),
                "bbox_2d": list(self.qwen_bbox_for_object(item)),
            }
            for item in self.objects
        ]
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def qwen_bbox_for_object(
        self,
        item: CalvinQwenSceneObject,
    ) -> tuple[int, int, int, int]:
        if item not in self.objects:
            raise ContractError("CALVIN scene Qwen bbox requires one supervised object")
        spec = _camera_spec(self.camera_name)
        return qwen3vl_normalized_bbox(
            item.bbox_xyxy,
            width=int(spec["width"]),
            height=int(spec["height"]),
        )

    @property
    def subpatch_visible_identity_keys(self) -> tuple[str, ...]:
        return tuple(item.identity_key for item in self.subpatch_objects)

    @property
    def grounding_request(self) -> str:
        categories = ", ".join(
            qwen_grounding_label(identity_key) for identity_key in self.category_identity_order
        )
        return QWEN_SCENE_GROUNDING_REQUEST.format(categories=categories)

    def qwen_user_messages(self, image_value: object | None = None) -> list[dict[str, Any]]:
        visible_image = self.image if image_value is None else image_value
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": visible_image},
                    {"type": "text", "text": self.grounding_request},
                ],
            }
        ]

    def qwen_messages(self, image_value: object | None = None) -> list[dict[str, Any]]:
        return [
            *self.qwen_user_messages(image_value=image_value),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.assistant_text}],
            },
        ]


def build_calvin_qwen_scene_grounding_record(
    *,
    global_index: int,
    camera_name: str,
    image: NDArray[np.uint8],
    physical_frame: CalvinPhysicalSupervisionFrame,
    category_identity_order: tuple[str, ...],
    visual_lattice: int,
) -> CalvinQwenSceneGroundingRecord:
    """Build one task-independent visible-object list from ordinary RGB."""

    if not isinstance(physical_frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("CALVIN scene grounding requires one physical frame")
    if isinstance(global_index, bool) or not isinstance(global_index, int) or global_index < 0:
        raise ContractError("CALVIN scene source index must be non-negative")
    if set(physical_frame.identity_keys) != set(CALVIN_QWEN_SCENE_IDENTITY_ORDER):
        raise ContractError("CALVIN scene physical inventory differs from reviewed identities")
    cameras = tuple(
        camera for camera in physical_frame.cameras if camera.camera_name == camera_name
    )
    if len(cameras) != 1:
        raise ContractError("CALVIN scene camera is absent or ambiguous")
    camera = cameras[0]
    spec = _camera_spec(camera_name)
    source_image = np.asarray(image)
    expected_shape = (int(spec["height"]), int(spec["width"]), 3)
    if source_image.dtype != np.uint8 or source_image.shape != expected_shape:
        raise ContractError("CALVIN scene source image shape or dtype is invalid")
    digest = source_array_sha256(str(spec["source_rgb_field"]), source_image)
    if camera.source_rgb_sha256 != digest:
        raise ContractError("CALVIN scene image differs from physical sidecar")
    if (
        isinstance(visual_lattice, bool)
        or not isinstance(visual_lattice, int)
        or visual_lattice <= 0
    ):
        raise ContractError("CALVIN scene visual lattice must be a positive integer")
    merge_size = QWEN3VL_SPATIAL_MERGE_SIZE
    image_grid_thw = (1, visual_lattice * merge_size, visual_lattice * merge_size)
    minimum_target_mass = minimum_projected_target_mass_for_raw_patch(merge_size=merge_size)
    owner_ids = tuple(range(1, len(physical_frame.identity_keys) + 1))
    projected = project_qwen3vl_segmentation(
        camera.owner_index,
        instance_ids=owner_ids,
        image_grid_thw=np.asarray(image_grid_thw, dtype=np.int64),
        patch_size=QWEN3VL_PATCH_SIZE,
        merge_size=merge_size,
        pixel_supervised=camera.owner_supervised,
        minimum_supervised_fraction=0.0,
    )
    owner_id_by_key = {
        identity_key: owner_id
        for owner_id, identity_key in enumerate(physical_frame.identity_keys, start=1)
    }
    objects = []
    subpatch = []
    absent = []
    for identity_key in category_identity_order:
        if identity_key not in owner_id_by_key:
            raise ContractError("CALVIN scene category is absent from physical inventory")
        owner_id = owner_id_by_key[identity_key]
        support = (camera.owner_index == owner_id) & camera.owner_supervised
        visible_pixels = int(np.count_nonzero(support))
        if visible_pixels == 0:
            absent.append(identity_key)
            continue
        if owner_id not in projected.merged.instance_ids:
            subpatch.append(
                CalvinQwenSceneObject(
                    identity_key=identity_key,
                    bbox_xyxy=tight_visible_owner_bbox(support),
                    visible_owner_pixels=visible_pixels,
                    projected_target_mass=0.0,
                    positive_visual_token_count=0,
                )
            )
            continue
        owner_column = projected.merged.instance_ids.index(owner_id)
        token_support = projected.merged.object_probability[:, owner_column]
        positive_tokens = projected.merged.supervised & (token_support > 0)
        projected_target_mass = float(
            np.sum(
                projected.merged.observed_fraction * token_support,
                dtype=np.float64,
            )
        )
        if projected_target_mass + 1e-12 < minimum_target_mass:
            subpatch.append(
                CalvinQwenSceneObject(
                    identity_key=identity_key,
                    bbox_xyxy=tight_visible_owner_bbox(support),
                    visible_owner_pixels=visible_pixels,
                    projected_target_mass=projected_target_mass,
                    positive_visual_token_count=int(np.count_nonzero(positive_tokens)),
                )
            )
            continue
        objects.append(
            CalvinQwenSceneObject(
                identity_key=identity_key,
                bbox_xyxy=tight_visible_owner_bbox(support),
                visible_owner_pixels=visible_pixels,
                projected_target_mass=projected_target_mass,
                positive_visual_token_count=int(np.count_nonzero(positive_tokens)),
            )
        )
    return CalvinQwenSceneGroundingRecord(
        global_index=global_index,
        camera_name=camera_name,
        host_image_key=str(spec["host_image_key"]),
        category_identity_order=category_identity_order,
        objects=tuple(objects),
        subpatch_objects=tuple(subpatch),
        absent_identity_keys=tuple(absent),
        minimum_projected_target_mass=minimum_target_mass,
        visual_lattice=visual_lattice,
        image_grid_thw=image_grid_thw,
        patch_size=QWEN3VL_PATCH_SIZE,
        merge_size=merge_size,
        image=_immutable_rgb_copy(source_image),
        source_rgb_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class CalvinQwenGroundingDistractor:
    """A same-image wrong physical box used only for conditional-NLL evaluation."""

    distractor_identity_key: str
    candidate_record: CalvinQwenGroundingRecord

    def __post_init__(self) -> None:
        if (
            not isinstance(self.distractor_identity_key, str)
            or not self.distractor_identity_key
            or self.distractor_identity_key == self.candidate_record.target_identity_key
        ):
            raise ContractError("CALVIN grounding distractor identity is invalid")


def build_calvin_qwen_grounding_distractors(
    record: CalvinQwenGroundingRecord,
    physical_frame: CalvinPhysicalSupervisionFrame,
) -> tuple[CalvinQwenGroundingDistractor, ...]:
    """Build same-image boxes for other visible physical identities."""

    if not isinstance(record, CalvinQwenGroundingRecord):
        raise TypeError("CALVIN grounding distractors require a grounding record")
    if not isinstance(physical_frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("CALVIN grounding distractors require a physical frame")
    if record.target_identity_key not in physical_frame.identity_keys:
        raise ContractError("grounding target is absent from the physical frame")
    cameras = tuple(
        camera for camera in physical_frame.cameras if camera.camera_name == record.camera_name
    )
    if len(cameras) != 1:
        raise ContractError("grounding camera is absent or ambiguous in physical frame")
    camera = cameras[0]
    distractors = []
    for owner_id, identity_key in enumerate(physical_frame.identity_keys, start=1):
        if identity_key == record.target_identity_key:
            continue
        visible = (camera.owner_index == owner_id) & camera.owner_supervised
        if not np.any(visible):
            continue
        distractors.append(
            CalvinQwenGroundingDistractor(
                distractor_identity_key=identity_key,
                candidate_record=replace(
                    record,
                    bbox_xyxy=tight_visible_owner_bbox(visible),
                ),
            )
        )
    return tuple(distractors)

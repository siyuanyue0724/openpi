"""Typed shared-Qwen native grounding boundary for LingBot co-training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenGroundingRecord,
    CalvinQwenSceneGroundingRecord,
    build_calvin_qwen_scene_grounding_record,
)
from picf_next.data.public_native_vl import NativeVLInstructionRecord

NATIVE_VL_IGNORE_INDEX = -100
QWEN3VL_ASSISTANT_HEADER_TOKEN_ID = 77091
QWEN3VL_END_OF_MESSAGE_TOKEN_ID = 151645

_MODEL_KWARG_NAMES = frozenset(
    {
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "labels",
        "pixel_values",
        "position_ids",
    }
)
_GENERATION_MODEL_KWARG_NAMES = _MODEL_KWARG_NAMES - {"labels", "position_ids"}
_FORBIDDEN_MODEL_KWARG_PARTS = (
    "bbox",
    "box",
    "depth",
    "identity",
    "mask_owner",
    "owner",
    "scene_obs",
    "task_key",
)

_NATIVE_QWEN_PREFIX = "model.qwenvl_with_expert.qwenvl."
_NATIVE_QWEN_LANGUAGE_PREFIX = f"{_NATIVE_QWEN_PREFIX}model.language_model."
_NATIVE_QWEN_MERGER_PREFIX = f"{_NATIVE_QWEN_PREFIX}model.visual.merger."

NativeVLRecord = (
    CalvinQwenGroundingRecord | CalvinQwenSceneGroundingRecord | NativeVLInstructionRecord
)


def _require_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ContractError(f"native VL {name} must be a tensor")
    return value


@dataclass(frozen=True, slots=True)
class NativeVLGroundingBatch:
    """Qwen-native tensors with an explicit assistant-only label proof."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    assistant_token_mask: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor
    position_ids: Tensor | None = None

    def __post_init__(self) -> None:
        input_ids = _require_tensor(self.input_ids, "input_ids")
        attention = _require_tensor(self.attention_mask, "attention_mask")
        labels = _require_tensor(self.labels, "labels")
        assistant = _require_tensor(self.assistant_token_mask, "assistant token mask")
        pixels = _require_tensor(self.pixel_values, "pixel values")
        grid = _require_tensor(self.image_grid_thw, "image grid")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ContractError("native VL input IDs must be int64[batch,sequence]")
        if labels.shape != input_ids.shape or labels.dtype != torch.long:
            raise ContractError("native VL labels must align with input IDs")
        if attention.shape != input_ids.shape or attention.dtype not in (
            torch.bool,
            torch.int32,
            torch.int64,
        ):
            raise ContractError("native VL attention mask must align with input IDs")
        if assistant.shape != input_ids.shape or assistant.dtype != torch.bool:
            raise ContractError("native VL assistant mask must be boolean and token-aligned")
        supervised = labels != NATIVE_VL_IGNORE_INDEX
        if not torch.equal(supervised, assistant):
            raise ContractError("native VL labels must supervise exactly assistant tokens")
        if not bool(torch.all(assistant.any(dim=1))):
            raise ContractError("every native VL record requires assistant supervision")
        if not bool(torch.all(attention[assistant].bool())):
            raise ContractError("native VL cannot supervise padding tokens")
        if not torch.equal(labels[assistant], input_ids[assistant]):
            raise ContractError("native VL assistant labels must equal their input token IDs")
        if not pixels.is_floating_point() or pixels.ndim < 2 or pixels.numel() == 0:
            raise ContractError("native VL pixel values must be a nonempty floating tensor")
        if (
            grid.ndim != 2
            or grid.shape[1] != 3
            or grid.dtype
            not in (
                torch.int32,
                torch.int64,
            )
        ):
            raise ContractError("native VL image grid must be integer[num_images,3]")
        if grid.shape[0] <= 0 or not bool(torch.all(grid > 0)):
            raise ContractError("native VL image grid dimensions must be positive")
        if self.position_ids is not None:
            positions = _require_tensor(self.position_ids, "position IDs")
            if positions.dtype not in (torch.int32, torch.int64):
                raise ContractError("native VL position IDs must be integral")
            if positions.shape[-2:] != input_ids.shape:
                raise ContractError("native VL position IDs must end in [batch,sequence]")

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def supervised_token_count(self) -> int:
        return int(self.assistant_token_mask.sum().item())

    def model_kwargs(self) -> dict[str, Tensor]:
        """Expose only ordinary Qwen tensors; label-side metadata has no path."""

        kwargs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }
        if self.position_ids is not None:
            kwargs["position_ids"] = self.position_ids
        validate_native_vl_model_kwargs(kwargs)
        return kwargs

    def to(self, device: torch.device, *, pixel_dtype: torch.dtype) -> NativeVLGroundingBatch:
        """Move ordinary Qwen inputs while preserving integral and boolean dtypes."""

        if not isinstance(device, torch.device):
            raise TypeError("native VL destination must be a torch device")
        if not pixel_dtype.is_floating_point:
            raise TypeError("native VL pixel dtype must be floating point")
        return NativeVLGroundingBatch(
            input_ids=self.input_ids.to(device=device),
            attention_mask=self.attention_mask.to(device=device),
            labels=self.labels.to(device=device),
            assistant_token_mask=self.assistant_token_mask.to(device=device),
            pixel_values=self.pixel_values.to(device=device, dtype=pixel_dtype),
            image_grid_thw=self.image_grid_thw.to(device=device),
            position_ids=(
                None if self.position_ids is None else self.position_ids.to(device=device)
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeVLGenerationBatch:
    """Prompt-only Qwen tensors for released-weight free-generation diagnosis."""

    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor

    def __post_init__(self) -> None:
        input_ids = _require_tensor(self.input_ids, "generation input IDs")
        attention = _require_tensor(self.attention_mask, "generation attention mask")
        pixels = _require_tensor(self.pixel_values, "generation pixel values")
        grid = _require_tensor(self.image_grid_thw, "generation image grid")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
            raise ContractError("native VL generation input IDs must be int64[1,sequence]")
        if attention.shape != input_ids.shape or attention.dtype not in (
            torch.bool,
            torch.int32,
            torch.int64,
        ):
            raise ContractError("native VL generation attention mask must align with input IDs")
        if not pixels.is_floating_point() or pixels.ndim < 2 or pixels.numel() == 0:
            raise ContractError("native VL generation pixels must be a nonempty floating tensor")
        if (
            grid.ndim != 2
            or grid.shape[1] != 3
            or grid.dtype not in (torch.int32, torch.int64)
            or grid.shape[0] <= 0
            or not bool(torch.all(grid > 0))
        ):
            raise ContractError("native VL generation image grid must be positive integer[N,3]")

    @property
    def prompt_token_count(self) -> int:
        return int(self.input_ids.shape[1])

    def model_kwargs(self) -> dict[str, Tensor]:
        kwargs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }
        validate_native_vl_generation_model_kwargs(kwargs)
        return kwargs

    def to(self, device: torch.device, *, pixel_dtype: torch.dtype) -> NativeVLGenerationBatch:
        if not isinstance(device, torch.device):
            raise TypeError("native VL generation destination must be a torch device")
        if not pixel_dtype.is_floating_point:
            raise TypeError("native VL generation pixel dtype must be floating point")
        return NativeVLGenerationBatch(
            input_ids=self.input_ids.to(device=device),
            attention_mask=self.attention_mask.to(device=device),
            pixel_values=self.pixel_values.to(device=device, dtype=pixel_dtype),
            image_grid_thw=self.image_grid_thw.to(device=device),
        )


@dataclass(frozen=True, slots=True)
class NativeVLGeneratedGrounding:
    """Structured grounding recovered from one deploy-visible Qwen answer."""

    bbox_qwen_xyxy: tuple[int, int, int, int] | None
    schema_valid: bool
    generated_label: str | None = None
    label_present: bool = False
    label_schema_valid: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schema_valid, bool):
            raise ContractError("native VL grounding schema flag must be boolean")
        if not isinstance(self.label_present, bool) or not isinstance(
            self.label_schema_valid, bool
        ):
            raise ContractError("native VL grounding label flags must be boolean")
        if self.schema_valid and self.bbox_qwen_xyxy is None:
            raise ContractError("schema-valid native VL grounding must contain one bbox")
        if self.generated_label is not None and (
            not isinstance(self.generated_label, str)
            or not self.generated_label
            or self.generated_label != self.generated_label.strip()
        ):
            raise ContractError("native VL generated label must be nonempty stripped text")
        if self.generated_label is not None and not self.label_present:
            raise ContractError("native VL generated label requires a present label field")
        expected_label_schema_valid = (
            self.schema_valid and self.label_present and self.generated_label is not None
        )
        if self.label_schema_valid != expected_label_schema_valid:
            raise ContractError("schema-valid native VL label requires one strict nonempty label")


@dataclass(frozen=True, slots=True)
class NativeVLGeneratedSceneObject:
    """One strict label-addressable object parsed from Qwen's JSON list."""

    label: str
    bbox_qwen_xyxy: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label or self.label != self.label.strip():
            raise ContractError("native VL scene label must be nonempty stripped text")
        if _qwen_bbox_from_value({"bbox_2d": list(self.bbox_qwen_xyxy)}) != (self.bbox_qwen_xyxy):
            raise ContractError("native VL scene bbox is invalid")


@dataclass(frozen=True, slots=True)
class NativeVLGeneratedSceneGrounding:
    """Strict ordered multi-object parse; set matching is performed by label."""

    objects: tuple[NativeVLGeneratedSceneObject, ...]
    schema_valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.schema_valid, bool):
            raise ContractError("native VL scene schema flag must be boolean")
        if not self.schema_valid and self.objects:
            raise ContractError("schema-invalid native VL scene answer cannot expose objects")
        normalized_labels = tuple(" ".join(item.label.casefold().split()) for item in self.objects)
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ContractError("native VL scene answer repeats one normalized label")


@dataclass(frozen=True, slots=True)
class NativeVLGroundingTrainableScope:
    """Exact Qwen-official adaptation boundary, with no auxiliary learned head."""

    parameter_count: int
    trainable_numel: int
    schema_sha256: str
    parameter_descriptors: tuple[tuple[str, tuple[int, ...], str, int], ...]

    def __post_init__(self) -> None:
        if self.parameter_count != len(self.parameter_descriptors) or self.parameter_count <= 0:
            raise ContractError("native VL trainable scope has an invalid parameter count")
        if self.trainable_numel != sum(item[3] for item in self.parameter_descriptors):
            raise ContractError("native VL trainable scope has an invalid element count")
        if len(self.schema_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.schema_sha256
        ):
            raise ContractError("native VL trainable scope has an invalid schema digest")
        names = tuple(item[0] for item in self.parameter_descriptors)
        if names != tuple(sorted(set(names))):
            raise ContractError("native VL trainable parameter names must be sorted and unique")
        if any(
            not (
                name.startswith(_NATIVE_QWEN_LANGUAGE_PREFIX)
                or name.startswith(_NATIVE_QWEN_MERGER_PREFIX)
            )
            for name in names
        ):
            raise ContractError("native VL trainable scope escapes Qwen language/merger")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self.parameter_descriptors)

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter_count": self.parameter_count,
            "trainable_numel": self.trainable_numel,
            "schema_sha256": self.schema_sha256,
            "parameters": [
                {
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype,
                    "numel": numel,
                }
                for name, shape, dtype, numel in self.parameter_descriptors
            ],
        }


def _qwen_grounding_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


def _qwen_bbox_from_value(value: object) -> tuple[int, int, int, int] | None:
    grounding = _qwen_grounding_mapping(value)
    if grounding is None:
        return None
    raw_bbox = grounding.get("bbox_2d")
    if (
        not isinstance(raw_bbox, list)
        or len(raw_bbox) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_bbox)
    ):
        return None
    x0, y0, x1, y1 = cast(tuple[int, int, int, int], tuple(raw_bbox))
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        return None
    return x0, y0, x1, y1


def _qwen_label_from_value(value: object) -> tuple[bool, str | None]:
    grounding = _qwen_grounding_mapping(value)
    if grounding is None or "label" not in grounding:
        return False, None
    raw_label = grounding["label"]
    if not isinstance(raw_label, str):
        return True, None
    label = raw_label.strip()
    return True, label or None


def _strip_native_vl_answer(text: str) -> str:
    payload = text.strip()
    if payload.endswith("<|im_end|>"):
        payload = payload[: -len("<|im_end|>")].rstrip()
    lines = payload.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_native_vl_grounding_answer(text: str) -> NativeVLGeneratedGrounding:
    """Parse official Qwen JSON, retaining geometry from a malformed outer list."""

    if not isinstance(text, str):
        raise TypeError("native VL grounding answer must be text")
    payload = _strip_native_vl_answer(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    bbox = _qwen_bbox_from_value(parsed)
    label_present, generated_label = _qwen_label_from_value(parsed)
    if bbox is not None:
        return NativeVLGeneratedGrounding(
            bbox_qwen_xyxy=bbox,
            schema_valid=True,
            generated_label=generated_label,
            label_present=label_present,
            label_schema_valid=generated_label is not None,
        )

    recovered = None
    object_start = payload.find("{")
    if object_start >= 0:
        try:
            recovered, _ = json.JSONDecoder().raw_decode(payload[object_start:])
        except json.JSONDecodeError:
            recovered = None
        bbox = _qwen_bbox_from_value(recovered)
    label_present, generated_label = _qwen_label_from_value(recovered)
    return NativeVLGeneratedGrounding(
        bbox_qwen_xyxy=bbox,
        schema_valid=False,
        generated_label=generated_label,
        label_present=label_present,
        label_schema_valid=False,
    )


def parse_native_vl_scene_grounding_answer(text: str) -> NativeVLGeneratedSceneGrounding:
    """Parse one complete Qwen multi-target list without positional recovery."""

    if not isinstance(text, str):
        raise TypeError("native VL scene grounding answer must be text")
    payload = _strip_native_vl_answer(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return NativeVLGeneratedSceneGrounding(objects=(), schema_valid=False)
    if not isinstance(parsed, list):
        return NativeVLGeneratedSceneGrounding(objects=(), schema_valid=False)
    objects = []
    normalized_labels = set()
    for value in parsed:
        if not isinstance(value, Mapping) or set(value) != {"label", "bbox_2d"}:
            return NativeVLGeneratedSceneGrounding(objects=(), schema_valid=False)
        bbox = _qwen_bbox_from_value(value)
        label_present, label = _qwen_label_from_value(value)
        if bbox is None or not label_present or label is None:
            return NativeVLGeneratedSceneGrounding(objects=(), schema_valid=False)
        normalized = " ".join(label.casefold().split())
        if normalized in normalized_labels:
            return NativeVLGeneratedSceneGrounding(objects=(), schema_valid=False)
        normalized_labels.add(normalized)
        objects.append(NativeVLGeneratedSceneObject(label=label, bbox_qwen_xyxy=bbox))
    return NativeVLGeneratedSceneGrounding(objects=tuple(objects), schema_valid=True)


def qwen_grounding_bbox_iou(
    predicted: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> float:
    """Return continuous IoU in Qwen's normalized 0..1000 coordinate system."""

    for name, bbox in (("predicted", predicted), ("target", target)):
        if _qwen_bbox_from_value({"bbox_2d": list(bbox)}) != bbox:
            raise ContractError(f"native VL {name} bbox is invalid")
    x0 = max(predicted[0], target[0])
    y0 = max(predicted[1], target[1])
    x1 = min(predicted[2], target[2])
    y1 = min(predicted[3], target[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    predicted_area = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    return intersection / (predicted_area + target_area - intersection)


def qwen_target_center_in_bbox(
    predicted: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> bool:
    """Measure target-object selection without inventing an IoU pass threshold."""

    qwen_grounding_bbox_iou(predicted, target)
    target_x = (target[0] + target[2]) / 2.0
    target_y = (target[1] + target[3]) / 2.0
    return predicted[0] <= target_x <= predicted[2] and predicted[1] <= target_y <= predicted[3]


def _require_native_vl_record(record: object) -> NativeVLRecord:
    if not isinstance(
        record,
        (
            CalvinQwenGroundingRecord,
            CalvinQwenSceneGroundingRecord,
            NativeVLInstructionRecord,
        ),
    ):
        raise TypeError("native VL preprocessing requires an approved instruction record")
    return record


def _writable_processor_image(record: NativeVLRecord) -> Any:
    processor_image = record.image.copy()
    if not processor_image.flags.writeable:
        raise ContractError("native VL processor image copy must be writable")
    return processor_image


def build_native_vl_grounding_batch(
    record: NativeVLRecord,
    processor: Any,
) -> NativeVLGroundingBatch:
    """Copy Qwen3-VL's official single-record assistant-label preprocessing."""

    record = _require_native_vl_record(record)
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        raise TypeError("native VL processor exposes no Qwen chat template")
    processor_image = _writable_processor_image(record)
    raw_result = apply_template(
        record.qwen_messages(image_value=processor_image),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not isinstance(raw_result, Mapping):
        raise ContractError("Qwen chat template returned no tensor mapping")
    result = cast(Mapping[str, object], raw_result)
    required = {"attention_mask", "image_grid_thw", "input_ids", "pixel_values"}
    if required - set(result.keys()):
        raise ContractError("Qwen chat template omitted native grounding tensors")
    input_ids = _require_tensor(result["input_ids"], "input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ContractError("native grounding preprocessing requires one Qwen record")
    labels = torch.full_like(input_ids, NATIVE_VL_IGNORE_INDEX)
    tokens = input_ids[0].tolist()
    assistant_headers = tuple(
        index for index, token in enumerate(tokens) if token == QWEN3VL_ASSISTANT_HEADER_TOKEN_ID
    )
    if len(assistant_headers) != 1:
        raise ContractError("Qwen grounding record must contain one assistant header")
    answer_start = assistant_headers[0] + 2
    try:
        answer_end = tokens.index(QWEN3VL_END_OF_MESSAGE_TOKEN_ID, answer_start)
    except ValueError as error:
        raise ContractError("Qwen grounding answer has no end-of-message token") from error
    label_stop = min(answer_end + 2, input_ids.shape[1])
    if answer_start >= label_stop:
        raise ContractError("Qwen grounding assistant span is empty")
    labels[0, answer_start:label_stop] = input_ids[0, answer_start:label_stop]
    assistant_mask = labels != NATIVE_VL_IGNORE_INDEX
    position_ids = result.get("position_ids")
    if position_ids is not None and not isinstance(position_ids, Tensor):
        raise ContractError("Qwen chat template returned invalid position IDs")
    return NativeVLGroundingBatch(
        input_ids=input_ids,
        attention_mask=_require_tensor(result["attention_mask"], "attention_mask"),
        labels=labels,
        assistant_token_mask=assistant_mask,
        pixel_values=_require_tensor(result["pixel_values"], "pixel values"),
        image_grid_thw=_require_tensor(result["image_grid_thw"], "image grid"),
        position_ids=position_ids,
    )


def build_native_vl_generation_batch(
    record: NativeVLRecord,
    processor: Any,
) -> NativeVLGenerationBatch:
    """Build Qwen's official prompt-only tensors for deterministic generation."""

    record = _require_native_vl_record(record)
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        raise TypeError("native VL processor exposes no Qwen chat template")
    user_messages = record.qwen_user_messages(image_value=_writable_processor_image(record))
    if (
        not isinstance(user_messages, list)
        or not user_messages
        or any(
            not isinstance(message, Mapping) or message.get("role") != "user"
            for message in user_messages
        )
    ):
        raise ContractError("native VL generation input may contain only user messages")
    raw_result = apply_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not isinstance(raw_result, Mapping):
        raise ContractError("Qwen generation template returned no tensor mapping")
    result = cast(Mapping[str, object], raw_result)
    if _GENERATION_MODEL_KWARG_NAMES - set(result):
        raise ContractError("Qwen generation template omitted native tensors")
    batch = NativeVLGenerationBatch(
        input_ids=_require_tensor(result["input_ids"], "generation input IDs"),
        attention_mask=_require_tensor(result["attention_mask"], "generation attention mask"),
        pixel_values=_require_tensor(result["pixel_values"], "generation pixel values"),
        image_grid_thw=_require_tensor(result["image_grid_thw"], "generation image grid"),
    )
    tokenizer = getattr(processor, "tokenizer", None)
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        decoded = decode(batch.input_ids[0], skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise ContractError("Qwen generation prompt decode returned non-text")
        if "assistant" not in decoded:
            raise ContractError("Qwen generation prompt omitted the assistant boundary")
    return batch


def validate_native_vl_model_kwargs(kwargs: dict[str, Tensor]) -> None:
    """Reject privileged supervision or a drifted Qwen call surface."""

    if set(kwargs) - _MODEL_KWARG_NAMES:
        raise ContractError("native VL model kwargs differ from the approved Qwen surface")
    required = _MODEL_KWARG_NAMES - {"position_ids"}
    if set(kwargs) < required:
        raise ContractError("native VL model kwargs omit required Qwen tensors")
    normalized = tuple(name.lower() for name in kwargs)
    if any(part in name for name in normalized for part in _FORBIDDEN_MODEL_KWARG_PARTS):
        raise ContractError("native VL model kwargs contain privileged supervision")
    if any(not isinstance(value, Tensor) for value in kwargs.values()):
        raise ContractError("native VL model kwargs must contain only tensors")


def validate_native_vl_generation_model_kwargs(kwargs: dict[str, Tensor]) -> None:
    """Reject labels and privileged metadata from free-generation diagnosis."""

    if set(kwargs) != _GENERATION_MODEL_KWARG_NAMES:
        raise ContractError("native VL generation kwargs differ from the approved Qwen surface")
    normalized = tuple(name.lower() for name in kwargs)
    if any(part in name for name in normalized for part in _FORBIDDEN_MODEL_KWARG_PARTS):
        raise ContractError("native VL generation kwargs contain privileged supervision")
    if any(not isinstance(value, Tensor) for value in kwargs.values()):
        raise ContractError("native VL generation kwargs must contain only tensors")


NATIVE_VL_GENERATION_MAX_NEW_TOKENS = 1024


def generate_native_vl_answer(
    model: nn.Module,
    batch: NativeVLGenerationBatch,
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> str:
    """Run deterministic native-Qwen generation without teacher-answer inputs."""

    if not isinstance(batch, NativeVLGenerationBatch):
        raise TypeError("native VL answer generation requires a typed prompt batch")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= NATIVE_VL_GENERATION_MAX_NEW_TOKENS
    ):
        raise ContractError(
            f"native VL generation length must lie in [1,{NATIVE_VL_GENERATION_MAX_NEW_TOKENS}]"
        )
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise ContractError("native Qwen model does not expose generation")
    generated = cast(
        Any,
        generate(
            **batch.model_kwargs(),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        ),
    )
    if not isinstance(generated, Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
        raise ContractError("Qwen grounding generation returned malformed token IDs")
    prompt_tokens = batch.prompt_token_count
    if generated.shape[1] <= prompt_tokens:
        raise ContractError("Qwen grounding generation returned no assistant tokens")
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise ContractError("Qwen grounding tokenizer exposes no decode method")
    decoded = decode(generated[0, prompt_tokens:], skip_special_tokens=False)
    if not isinstance(decoded, str):
        raise ContractError("Qwen grounding tokenizer decode returned non-text")
    return decoded


def register_native_vl_fsdp_forward_method(policy: nn.Module) -> None:
    """Register the non-forward Qwen loss path on a sharded root policy."""

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    if not isinstance(policy, FSDPModule):
        raise RuntimeError("native VL policy must be the root FSDP2 unit")
    if not callable(getattr(policy, "picf_native_vl_forward", None)):
        raise TypeError("LingBot policy lacks the audited native VL root method")
    register_fsdp_forward_method(policy, "picf_native_vl_forward")


def run_native_vl_grounding_forward(policy: nn.Module, batch: NativeVLGroundingBatch) -> Tensor:
    """Execute and validate one native Qwen grounding loss."""

    if not isinstance(batch, NativeVLGroundingBatch):
        raise TypeError("native VL forward requires a typed grounding batch")
    forward = getattr(policy, "picf_native_vl_forward", None)
    if not callable(forward):
        raise TypeError("LingBot policy exposes no native VL forward")
    loss = forward(**batch.model_kwargs())
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise ContractError("native VL forward must return one scalar loss")
    if not bool(torch.isfinite(loss)):
        raise ContractError("native VL loss is non-finite")
    if torch.is_grad_enabled() and not loss.requires_grad:
        raise ContractError("native VL training loss is detached from Qwen")
    return loss


def compose_native_vl_vla_loss(
    robot_loss: Tensor,
    grounding_loss: Tensor | None,
    *,
    grounding_weight: float,
) -> Tensor:
    """Compose an explicitly weighted VL factor without hidden policy defaults."""

    if not isinstance(robot_loss, Tensor) or robot_loss.ndim != 0:
        raise ContractError("robot objective must be a scalar tensor")
    if not bool(torch.isfinite(robot_loss)):
        raise ContractError("robot objective is non-finite")
    if not isinstance(grounding_weight, float) or not 0.0 < grounding_weight <= 1.0:
        raise ContractError("native VL loss weight must lie in (0,1]")
    if grounding_loss is None:
        return robot_loss
    if not isinstance(grounding_loss, Tensor) or grounding_loss.ndim != 0:
        raise ContractError("grounding objective must be a scalar tensor")
    if not bool(torch.isfinite(grounding_loss)):
        raise ContractError("grounding objective is non-finite")
    return robot_loss + grounding_weight * grounding_loss


def native_vl_stream_factor(
    step_index: int,
    *,
    robot_steps: int,
    grounding_steps: int,
) -> str:
    """Return an explicitly preregistered robot/native-VL stream factor."""

    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ContractError("native VL stream step index must be non-negative")
    for name, value in (("robot steps", robot_steps), ("grounding steps", grounding_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"native VL {name} must be a positive integer")
    cycle = robot_steps + grounding_steps
    return "robot" if step_index % cycle < robot_steps else "grounding"


def _describe_native_vl_grounding_trainable_scope(
    policy: nn.Module,
) -> NativeVLGroundingTrainableScope:
    descriptors = tuple(
        sorted(
            (
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                parameter.numel(),
            )
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        )
    )
    serialized = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "numel": numel,
        }
        for name, shape, dtype, numel in descriptors
    ]
    digest = hashlib.sha256(
        json.dumps(
            serialized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return NativeVLGroundingTrainableScope(
        parameter_count=len(descriptors),
        trainable_numel=sum(item[3] for item in descriptors),
        schema_sha256=digest,
        parameter_descriptors=descriptors,
    )


def configure_native_vl_grounding_trainable_scope(
    policy: nn.Module,
) -> NativeVLGroundingTrainableScope:
    """Mirror Qwen3-VL SFT: train its language model and visual merger only."""

    if not isinstance(policy, nn.Module):
        raise TypeError("native VL trainable scope requires one policy module")
    try:
        host = cast(Any, policy)
        qwen = host.model.qwenvl_with_expert.qwenvl
        language_model = qwen.model.language_model
        visual_merger = qwen.model.visual.merger
    except AttributeError as error:
        raise ContractError("LingBot policy lacks Qwen language/merger modules") from error
    selected_ids = {
        id(parameter)
        for module in (language_model, visual_merger)
        for parameter in module.parameters()
    }
    if not selected_ids:
        raise ContractError("Qwen language/merger trainable selection is empty")
    for parameter in policy.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    observed_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if observed_ids != selected_ids:
        raise ContractError("Qwen language/merger parameters are not uniquely installed")
    validate_tied_qwen_lm_head(policy)
    return _describe_native_vl_grounding_trainable_scope(policy)


def verify_native_vl_grounding_trainable_scope(
    policy: nn.Module,
    *,
    expected: NativeVLGroundingTrainableScope,
) -> NativeVLGroundingTrainableScope:
    """Prove distributed wrapping preserved the exact frozen-host boundary."""

    if not isinstance(expected, NativeVLGroundingTrainableScope):
        raise TypeError("native VL scope verification requires a typed expected scope")
    observed = _describe_native_vl_grounding_trainable_scope(policy)
    if observed != expected:
        raise ContractError("distributed wrapping changed the native VL trainable scope")
    validate_tied_qwen_lm_head(policy)
    return observed


def _materialize_fixed_observation_native_vl_records_by_camera(
    *,
    index: Any,
    sidecar: Any,
    group: Any,
    variant: Any,
) -> dict[str, CalvinQwenGroundingRecord]:
    """Materialize all measurable cameras for one audited source variant."""

    from picf_next.data.calvin_qwen_grounding import build_calvin_qwen_grounding_records
    from picf_next.lingbot_native.fixed_observation import (
        FixedObservationGroup,
        FixedObservationVariant,
        validate_fixed_observation_group_source_index,
    )

    if not isinstance(group, FixedObservationGroup):
        raise TypeError("native VL materialization requires one audited group")
    if not isinstance(variant, FixedObservationVariant) or variant not in group.variants:
        raise ContractError("native VL materialization requires one audited variant")
    validate_fixed_observation_group_source_index(index, group)
    global_index = group.source_global_index
    arrays = dict(
        index.validated_source_frame_arrays(
            global_index,
            fields=("rgb_gripper", "rgb_static"),
        )
    )
    observation_images = {
        "observation.images.image": arrays["rgb_static"],
        "observation.images.wrist_image": arrays["rgb_gripper"],
    }
    physical = sidecar.source_frame(global_index)
    records = build_calvin_qwen_grounding_records(
        global_index=global_index,
        task_key=variant.task_key,
        instruction=variant.instruction,
        observation_images=observation_images,
        physical_frame=physical,
    )
    if any(record.target_identity_key != variant.target_identity_key for record in records):
        raise ContractError("native VL target differs from its audited variant")
    cameras = {record.camera_name: record for record in records}
    if len(cameras) != len(records):
        raise ContractError("native VL records repeat one camera")
    return cameras


def materialize_fixed_observation_native_vl_record(
    *,
    index: Any,
    sidecar: Any,
    group: Any,
    variant: Any,
    expected_camera_name: str,
) -> CalvinQwenGroundingRecord:
    """Materialize one source-bound native Qwen record for a planned camera."""

    if not isinstance(expected_camera_name, str) or not expected_camera_name:
        raise ContractError("native VL materialization requires one planned camera")
    cameras = _materialize_fixed_observation_native_vl_records_by_camera(
        index=index,
        sidecar=sidecar,
        group=group,
        variant=variant,
    )
    try:
        return cameras[expected_camera_name]
    except KeyError as error:
        raise ContractError("native VL planned camera is not measurably visible") from error


def materialize_fixed_observation_native_vl_records(
    *,
    index: Any,
    sidecar: Any,
    group: Any,
    variants: tuple[Any, Any],
    expected_camera_name: str | None = None,
) -> tuple[CalvinQwenGroundingRecord, CalvinQwenGroundingRecord]:
    """Materialize two byte-identical-image Qwen records from one audited pair."""

    from picf_next.lingbot_native.fixed_observation import (
        FixedObservationGroup,
        FixedObservationVariant,
    )

    if not isinstance(group, FixedObservationGroup):
        raise TypeError("native VL fixed-X materialization requires one audited group")
    if (
        not isinstance(variants, tuple)
        or len(variants) != 2
        or any(not isinstance(item, FixedObservationVariant) for item in variants)
        or any(item not in group.variants for item in variants)
    ):
        raise ContractError("native VL fixed-X materialization requires two audited variants")
    by_variant = tuple(
        _materialize_fixed_observation_native_vl_records_by_camera(
            index=index,
            sidecar=sidecar,
            group=group,
            variant=variant,
        )
        for variant in variants
    )
    common = set(by_variant[0]).intersection(by_variant[1])
    if not common:
        raise ContractError("native VL fixed-X pair has no common visible camera")
    if expected_camera_name is None:
        camera_name = "static" if "static" in common else sorted(common)[0]
    elif not isinstance(expected_camera_name, str) or expected_camera_name not in common:
        raise ContractError("native VL planned camera is not commonly visible")
    else:
        camera_name = expected_camera_name
    pair = (by_variant[0][camera_name], by_variant[1][camera_name])
    if pair[0].source_rgb_sha256 != pair[1].source_rgb_sha256 or not np.array_equal(
        pair[0].image, pair[1].image
    ):
        raise ContractError("native VL fixed-X variants do not share byte-identical pixels")
    return pair


def build_counterfactual_scene_grounding_records(
    target_records: tuple[CalvinQwenGroundingRecord, CalvinQwenGroundingRecord],
    physical_frame: Any,
    *,
    visual_lattice: int,
) -> tuple[CalvinQwenSceneGroundingRecord, CalvinQwenSceneGroundingRecord]:
    """Build canonical/reverse scene lists for one byte-identical image pair."""

    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionFrame,
    )

    if (
        not isinstance(target_records, tuple)
        or len(target_records) != 2
        or any(not isinstance(item, CalvinQwenGroundingRecord) for item in target_records)
    ):
        raise TypeError("counterfactual scene grounding requires two target records")
    if not isinstance(physical_frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("counterfactual scene grounding requires one physical frame")
    first, second = target_records
    if (
        first.global_index != second.global_index
        or first.camera_name != second.camera_name
        or first.source_rgb_sha256 != second.source_rgb_sha256
        or not np.array_equal(first.image, second.image)
    ):
        raise ContractError("counterfactual scene records require byte-identical images")
    orders = (
        CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER)),
    )
    records = tuple(
        build_calvin_qwen_scene_grounding_record(
            global_index=record.global_index,
            camera_name=record.camera_name,
            image=record.image,
            physical_frame=physical_frame,
            category_identity_order=order,
            visual_lattice=visual_lattice,
        )
        for record, order in zip(target_records, orders, strict=True)
    )
    object_maps = tuple(
        {
            item.identity_key: (
                item.bbox_xyxy,
                item.visible_owner_pixels,
                item.projected_target_mass,
                item.positive_visual_token_count,
            )
            for item in record.objects
        }
        for record in records
    )
    if object_maps[0] != object_maps[1]:
        raise ContractError("counterfactual scene orders changed the supervised object set")
    subpatch_maps = tuple(
        {
            item.identity_key: (
                item.bbox_xyxy,
                item.visible_owner_pixels,
                item.projected_target_mass,
                item.positive_visual_token_count,
            )
            for item in record.subpatch_objects
        }
        for record in records
    )
    if subpatch_maps[0] != subpatch_maps[1] or set(records[0].absent_identity_keys) != set(
        records[1].absent_identity_keys
    ):
        raise ContractError("counterfactual scene orders changed visibility partitions")
    return cast(
        tuple[CalvinQwenSceneGroundingRecord, CalvinQwenSceneGroundingRecord],
        records,
    )


def validate_tied_qwen_lm_head(policy: nn.Module) -> str:
    """Prove that grounding uses the shared Qwen embedding parameter."""

    try:
        host_policy = cast(Any, policy)
        host_model = host_policy.model
        qwen = host_model.qwenvl_with_expert.qwenvl
        embedding = qwen.model.language_model.embed_tokens.weight
        lm_head = qwen.lm_head.weight
    except AttributeError as error:
        raise ContractError("LingBot policy lacks the native Qwen LM-head path") from error
    if lm_head is not embedding:
        raise ContractError("native Qwen LM head is not tied to the shared token embedding")
    matches = tuple(name for name, parameter in policy.named_parameters() if parameter is embedding)
    if len(matches) != 1:
        raise ContractError("tied Qwen embedding must have one canonical parameter name")
    return matches[0]


def retie_and_validate_native_qwen_lm_head(policy: nn.Module) -> str:
    """Use Qwen's native tie hook and prove the resulting parameter alias."""

    try:
        host_policy = cast(Any, policy)
        qwen = host_policy.model.qwenvl_with_expert.qwenvl
        tie_weights = qwen.tie_weights
    except AttributeError as error:
        raise ContractError("LingBot policy lacks Qwen's native weight-tie hook") from error
    if not callable(tie_weights):
        raise ContractError("Qwen native weight-tie hook is not callable")
    tie_weights()
    return validate_tied_qwen_lm_head(policy)


def validate_native_vl_optimizer_membership(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> str:
    """Prove that the tied shared-Qwen parameter belongs to the active optimizer."""

    canonical_name = validate_tied_qwen_lm_head(policy)
    parameter = dict(policy.named_parameters())[canonical_name]
    optimizer_ids = {
        id(value)
        for group in optimizer.param_groups
        for value in group.get("params", ())
        if isinstance(value, nn.Parameter)
    }
    if id(parameter) not in optimizer_ids:
        raise ContractError("shared Qwen grounding parameter is absent from the optimizer")
    return canonical_name

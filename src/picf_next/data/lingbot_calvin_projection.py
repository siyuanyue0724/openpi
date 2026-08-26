"""Hash-bound LingBot Qwen projection geometry for CALVIN visual audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS

LINGBOT_CALVIN_PROJECTION_SCHEMA = "picf-next.lingbot-calvin-qwen-projection.v1"


@dataclass(frozen=True, slots=True)
class LingBotCALVINCameraSlot:
    """One released LingBot camera slot and its CALVIN supervision source."""

    runtime_camera_name: str
    physical_camera_name: str | None
    projection_camera_name: str
    valid: bool


# LingBot's released FeatureTransform always emits these three slots in this
# order. CALVIN has no right-wrist camera, so the official transform pads that
# slot from the first (top/static) image geometry and marks it invalid.
LINGBOT_CALVIN_CAMERA_SLOTS = (
    LingBotCALVINCameraSlot("camera_top", "static", "static", True),
    LingBotCALVINCameraSlot("camera_wrist_left", "gripper", "gripper", True),
    LingBotCALVINCameraSlot("camera_wrist_right", None, "static", False),
)

_MAXIMUM_CONTRACT_BYTES = 1024 * 1024
_SHA256_LENGTH = 64
_GIT_COMMIT_LENGTH = 40
_CONTRACT_FIELDS = {
    "schema",
    "status",
    "runtime_input",
    "processor_id",
    "processor_revision",
    "processor_assets_sha256",
    "processor_config_sha256",
    "processor_preprocessor_config_sha256",
    "dataset_manifest_sha256",
    "dataset_tree_sha256",
    "source_frame_count",
    "sample_global_indices",
    "patch_size",
    "merge_size",
    "temporal_patch_size",
    "views",
    "transformers_version",
}
_VIEW_FIELDS = {
    "source_field",
    "source_shape",
    "image_grid_thw",
    "merged_grid_hw",
    "raw_patch_count",
    "merged_token_count",
    "pixel_values_shape",
    "source_rgb_sha256",
}


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, *, name: str, fields: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    return result


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _git_commit(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != _GIT_COMMIT_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ContractError(f"{name} must be one lowercase full Git commit")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _int_list(
    value: object,
    *,
    name: str,
    length: int | None = None,
    positive: bool = False,
) -> list[int]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise ContractError(f"{name} must be an integer list of the required length")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or (positive and item <= 0):
            raise ContractError(f"{name} contains an invalid integer")
        result.append(item)
    return result


def processor_assets_sha256(assets: object) -> str:
    """Return one canonical digest for an exact processor asset manifest."""

    if not isinstance(assets, Sequence) or isinstance(assets, str | bytes) or not assets:
        raise ContractError("processor assets must be one non-empty sequence")
    normalized = []
    for index, raw in enumerate(assets):
        item = _exact(
            raw,
            name=f"processor asset {index}",
            fields={"path", "bytes", "sha256"},
        )
        path = _text(item["path"], name=f"processor asset {index} path")
        if path.startswith("/") or "\\" in path or path in {".", ".."} or ".." in path.split("/"):
            raise ContractError("processor asset path must be normalized and relative")
        normalized.append(
            {
                "path": path,
                "bytes": _positive_int(
                    item["bytes"],
                    name=f"processor asset {index} bytes",
                ),
                "sha256": _sha256(
                    item["sha256"],
                    name=f"processor asset {index} SHA-256",
                ),
            }
        )
    if [item["path"] for item in normalized] != sorted({item["path"] for item in normalized}):
        raise ContractError("processor assets must be unique and path-sorted")
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(b"picf-next.processor-assets.v1\0" + payload).hexdigest()


def projection_payload_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical identity of a validated projection payload."""

    validated = validate_lingbot_calvin_projection_payload(payload)
    encoded = json.dumps(
        validated,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(b"picf-next.lingbot-calvin-projection.v1\0" + encoded).hexdigest()


def validate_lingbot_calvin_projection_payload(
    value: object,
    *,
    expected_dataset_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one official-processor measurement without importing Transformers."""

    contract = _exact(
        value,
        name="LingBot CALVIN projection contract",
        fields=_CONTRACT_FIELDS,
    )
    if (
        contract["schema"] != LINGBOT_CALVIN_PROJECTION_SCHEMA
        or contract["status"] != "PASS"
        or contract["runtime_input"] is not False
    ):
        raise ContractError("LingBot CALVIN projection contract did not pass")
    dataset_manifest_sha256 = _sha256(
        contract["dataset_manifest_sha256"],
        name="projection dataset manifest SHA-256",
    )
    if expected_dataset_manifest_sha256 is not None and dataset_manifest_sha256 != _sha256(
        expected_dataset_manifest_sha256,
        name="expected projection dataset manifest SHA-256",
    ):
        raise ContractError("LingBot projection belongs to another dataset manifest")
    source_frame_count = _positive_int(
        contract["source_frame_count"],
        name="projection source frame count",
    )
    sample_global_indices = _int_list(
        contract["sample_global_indices"],
        name="projection sample global indices",
    )
    if (
        not sample_global_indices
        or sample_global_indices != sorted(set(sample_global_indices))
        or sample_global_indices[0] < 0
        or sample_global_indices[-1] >= source_frame_count
    ):
        raise ContractError("projection sample indices are not unique in-range source frames")
    patch_size = _positive_int(contract["patch_size"], name="projection patch size")
    merge_size = _positive_int(contract["merge_size"], name="projection merge size")
    temporal_patch_size = _positive_int(
        contract["temporal_patch_size"],
        name="projection temporal patch size",
    )
    expected_views = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    views = _mapping(contract["views"], name="projection views")
    if set(views) != set(expected_views):
        raise ContractError("projection views differ from the CALVIN camera contract")
    normalized_views: dict[str, dict[str, object]] = {}
    for camera_name, spec in expected_views.items():
        view = _exact(
            views[camera_name],
            name=f"projection view {camera_name}",
            fields=_VIEW_FIELDS,
        )
        source_field = _text(
            view["source_field"],
            name=f"projection view {camera_name} source field",
        )
        if source_field != spec["source_rgb_field"]:
            raise ContractError("projection source field differs from the CALVIN camera")
        source_shape = _int_list(
            view["source_shape"],
            name=f"projection view {camera_name} source shape",
            length=3,
            positive=True,
        )
        if source_shape != [int(spec["height"]), int(spec["width"]), 3]:
            raise ContractError("projection source shape differs from the CALVIN camera")
        image_grid = _int_list(
            view["image_grid_thw"],
            name=f"projection view {camera_name} image grid",
            length=3,
            positive=True,
        )
        if image_grid[0] != 1 or image_grid[1] % merge_size or image_grid[2] % merge_size:
            raise ContractError("projection image grid is not one divisible image")
        merged_grid = _int_list(
            view["merged_grid_hw"],
            name=f"projection view {camera_name} merged grid",
            length=2,
            positive=True,
        )
        expected_merged = [
            image_grid[1] // merge_size,
            image_grid[2] // merge_size,
        ]
        if merged_grid != expected_merged:
            raise ContractError("projection merged grid differs from Qwen geometry")
        raw_patch_count = _positive_int(
            view["raw_patch_count"],
            name=f"projection view {camera_name} raw patch count",
        )
        merged_token_count = _positive_int(
            view["merged_token_count"],
            name=f"projection view {camera_name} merged token count",
        )
        if raw_patch_count != image_grid[0] * image_grid[1] * image_grid[2]:
            raise ContractError("projection raw patch count differs from Qwen grid")
        if merged_token_count != merged_grid[0] * merged_grid[1]:
            raise ContractError("projection merged token count differs from Qwen grid")
        pixel_values_shape = _int_list(
            view["pixel_values_shape"],
            name=f"projection view {camera_name} pixel-values shape",
            length=2,
            positive=True,
        )
        expected_width = temporal_patch_size * patch_size * patch_size * 3
        if pixel_values_shape != [raw_patch_count, expected_width]:
            raise ContractError("projection pixel-values shape differs from Qwen patches")
        source_hashes = view["source_rgb_sha256"]
        if not isinstance(source_hashes, list) or len(source_hashes) != len(sample_global_indices):
            raise ContractError("projection source hashes differ from sampled frames")
        normalized_hashes = [
            _sha256(
                digest,
                name=f"projection view {camera_name} source RGB SHA-256",
            )
            for digest in source_hashes
        ]
        normalized_views[camera_name] = {
            "source_field": source_field,
            "source_shape": source_shape,
            "image_grid_thw": image_grid,
            "merged_grid_hw": merged_grid,
            "raw_patch_count": raw_patch_count,
            "merged_token_count": merged_token_count,
            "pixel_values_shape": pixel_values_shape,
            "source_rgb_sha256": normalized_hashes,
        }
    return {
        "schema": LINGBOT_CALVIN_PROJECTION_SCHEMA,
        "status": "PASS",
        "runtime_input": False,
        "processor_id": _text(contract["processor_id"], name="projection processor ID"),
        "processor_revision": _git_commit(
            contract["processor_revision"],
            name="projection processor revision",
        ),
        "processor_assets_sha256": _sha256(
            contract["processor_assets_sha256"],
            name="projection processor assets SHA-256",
        ),
        "processor_config_sha256": _sha256(
            contract["processor_config_sha256"],
            name="projection processor config SHA-256",
        ),
        "processor_preprocessor_config_sha256": _sha256(
            contract["processor_preprocessor_config_sha256"],
            name="projection preprocessor config SHA-256",
        ),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_tree_sha256": _sha256(
            contract["dataset_tree_sha256"],
            name="projection dataset tree SHA-256",
        ),
        "source_frame_count": source_frame_count,
        "sample_global_indices": sample_global_indices,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "views": normalized_views,
        "transformers_version": _text(
            contract["transformers_version"],
            name="projection Transformers version",
        ),
    }


def load_lingbot_calvin_projection_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_sha256: str,
) -> dict[str, Any]:
    """Load one immutable official-processor projection measurement."""

    resolved = Path(path).expanduser().absolute()
    expected = _sha256(expected_sha256, name="projection contract expected SHA-256")
    if resolved.is_symlink() or not resolved.is_file():
        raise ContractError("projection contract must be one real file")
    if resolved.stat().st_size > _MAXIMUM_CONTRACT_BYTES:
        raise ContractError("projection contract exceeds the maximum size")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError("projection contract differs from its expected SHA-256")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("projection contract is not valid JSON") from error
    return validate_lingbot_calvin_projection_payload(
        value,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
    )

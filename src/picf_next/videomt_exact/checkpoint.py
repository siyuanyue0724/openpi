"""Integrity and mechanical key conversion for the released VidEoMT checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

PUBLISHED_CHECKPOINT_BYTES = 1_264_120_741
PUBLISHED_CHECKPOINT_SHA256 = "2cfa7a2df68e6f21f29bea3be571b1f63d0f94c90b7b528a67267eb84317c04f"
PUBLISHED_MODEL_TENSORS = 437
PUBLISHED_BACKBONE_TENSORS = 436
PUBLISHED_MODEL_NUMEL = 315_987_030

COMPLETE_DISTRIBUTED_CHECKPOINT_SCHEMA = (
    "picf-next.videomt-complete-distributed-checkpoint/v1"
)
COMPLETE_TRAINING_REPORT_SCHEMA = "picf-next.videomt-complete-distributed-calvin/v1"
ADAPTED_MODEL_CHECKPOINT_SCHEMA = "picf-next.videomt-adapted-model/v1"
ADAPTED_MODEL_TENSORS = 436
ADAPTED_MODEL_NUMEL = 315_986_989

VIDEOMT_BACKBONE_PREFIX = "backbone."
DINO_BACKBONE_PREFIX = "backbone.encoder.backbone."

# This is the released DINOv3-L/16 architecture. Every non-default value is
# supported by both the published checkpoint shapes and Meta's DINOv3 factory.
DINO_V3_L_CONFIG: dict[str, Any] = {
    "architectures": ["DINOv3ViTModel"],
    "model_type": "dinov3_vit",
    "patch_size": 16,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "hidden_act": "gelu",
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-5,
    "rope_theta": 100.0,
    "image_size": 224,
    "num_channels": 3,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "layerscale_value": 1e-5,
    "drop_path_rate": 0.0,
    "use_gated_mlp": False,
    "num_register_tokens": 4,
    "pos_embed_shift": None,
    "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0,
    "torch_dtype": "float32",
    "transformers_version": "4.56.1",
}


@dataclass(frozen=True, slots=True)
class PublishedCheckpointReceipt:
    """Auditable facts read from one released checkpoint."""

    path: str
    byte_count: int
    sha256: str
    top_level_keys: tuple[str, ...]
    tensor_count: int
    backbone_tensor_count: int
    model_numel: int
    criterion_tensor_count: int

    @property
    def matches_release(self) -> bool:
        return (
            self.byte_count == PUBLISHED_CHECKPOINT_BYTES
            and self.sha256 == PUBLISHED_CHECKPOINT_SHA256
            and self.tensor_count == PUBLISHED_MODEL_TENSORS
            and self.backbone_tensor_count == PUBLISHED_BACKBONE_TENSORS
            and self.model_numel == PUBLISHED_MODEL_NUMEL
            and self.criterion_tensor_count == 1
        )


@dataclass(frozen=True, slots=True)
class AdaptedCheckpointReceipt:
    """Authenticated facts for one complete CALVIN-adapted donor state."""

    path: str
    byte_count: int
    sha256: str
    source_checkpoint_sha256: str
    source_report_sha256: str
    global_step: int
    split_plan_sha256: str
    implementation_sha256: str
    dataset_manifest_sha256: str
    physical_sidecar_manifest_sha256: str
    released_checkpoint_sha256: str
    tensor_count: int
    model_numel: int

    @property
    def is_complete(self) -> bool:
        return (
            self.released_checkpoint_sha256 == PUBLISHED_CHECKPOINT_SHA256
            and self.tensor_count == ADAPTED_MODEL_TENSORS
            and self.model_numel == ADAPTED_MODEL_NUMEL
        )


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def adapted_videomt_model_state(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    require_complete: bool = True,
) -> tuple[AdaptedCheckpointReceipt, Mapping[str, torch.Tensor]]:
    """Load a tensor-only adapted artifact and verify its full-model provenance."""

    expected_sha256 = _require_sha256("expected adapted checkpoint SHA-256", expected_sha256)
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("adapted VidEoMT checkpoint must not be a symlink")
    checkpoint = candidate.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError("adapted VidEoMT checkpoint SHA-256 differs")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        ADAPTED_MODEL_CHECKPOINT_SCHEMA
    ):
        raise ValueError("adapted VidEoMT checkpoint schema differs")
    source = payload.get("source")
    model = payload.get("model")
    if not isinstance(source, Mapping):
        raise TypeError("adapted VidEoMT checkpoint has no source provenance")
    if source.get("checkpoint_schema") != COMPLETE_DISTRIBUTED_CHECKPOINT_SCHEMA:
        raise ValueError("adapted VidEoMT source checkpoint schema differs")
    if source.get("report_schema") != COMPLETE_TRAINING_REPORT_SCHEMA:
        raise ValueError("adapted VidEoMT source report schema differs")
    if not isinstance(model, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in model.items()
    ):
        raise TypeError("adapted VidEoMT model state must contain only named tensors")
    global_step = source.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("adapted VidEoMT source step is invalid")

    receipt = AdaptedCheckpointReceipt(
        path=str(checkpoint),
        byte_count=checkpoint.stat().st_size,
        sha256=checkpoint_sha256,
        source_checkpoint_sha256=_require_sha256(
            "source checkpoint SHA-256", source.get("checkpoint_sha256")
        ),
        source_report_sha256=_require_sha256(
            "source report SHA-256", source.get("report_sha256")
        ),
        global_step=global_step,
        split_plan_sha256=_require_sha256(
            "source split plan SHA-256", source.get("split_plan_sha256")
        ),
        implementation_sha256=_require_sha256(
            "source implementation SHA-256", source.get("implementation_sha256")
        ),
        dataset_manifest_sha256=_require_sha256(
            "source dataset manifest SHA-256", source.get("dataset_manifest_sha256")
        ),
        physical_sidecar_manifest_sha256=_require_sha256(
            "source physical sidecar manifest SHA-256",
            source.get("physical_sidecar_manifest_sha256"),
        ),
        released_checkpoint_sha256=_require_sha256(
            "source released checkpoint SHA-256",
            source.get("released_checkpoint_sha256"),
        ),
        tensor_count=len(model),
        model_numel=sum(value.numel() for value in model.values()),
    )
    if require_complete and not receipt.is_complete:
        raise ValueError(f"adapted VidEoMT state is not the complete donor: {receipt}")
    return receipt, model


def _load_model_state(path: Path) -> tuple[tuple[str, ...], Mapping[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("published VidEoMT checkpoint must be a mapping")
    top_level_keys = tuple(str(key) for key in payload)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise KeyError("published VidEoMT checkpoint has no model mapping")
    if any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in model.items()
    ):
        raise TypeError("published VidEoMT model state must contain only named tensors")
    return top_level_keys, model


def inspect_published_checkpoint(
    path: str | os.PathLike[str],
    *,
    require_release_match: bool = True,
) -> PublishedCheckpointReceipt:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    top_level_keys, state = _load_model_state(checkpoint)
    receipt = PublishedCheckpointReceipt(
        path=str(checkpoint),
        byte_count=checkpoint.stat().st_size,
        sha256=sha256_file(checkpoint),
        top_level_keys=top_level_keys,
        tensor_count=len(state),
        backbone_tensor_count=sum(key.startswith(VIDEOMT_BACKBONE_PREFIX) for key in state),
        model_numel=sum(value.numel() for value in state.values()),
        criterion_tensor_count=sum(key.startswith("criterion.") for key in state),
    )
    if require_release_match and not receipt.matches_release:
        raise ValueError(f"checkpoint does not match the published VidEoMT release: {receipt}")
    return receipt


def published_videomt_backbone_state(
    path: str | os.PathLike[str],
    *,
    require_release_match: bool = True,
) -> dict[str, torch.Tensor]:
    if require_release_match:
        inspect_published_checkpoint(path)
    _top_level_keys, state = _load_model_state(Path(path).expanduser().resolve())
    backbone = {
        key.removeprefix(VIDEOMT_BACKBONE_PREFIX): value
        for key, value in state.items()
        if key.startswith(VIDEOMT_BACKBONE_PREFIX)
    }
    if len(backbone) != PUBLISHED_BACKBONE_TENSORS:
        raise ValueError(f"expected {PUBLISHED_BACKBONE_TENSORS} VidEoMT backbone tensors")
    return backbone


def hf_dinov3_state_from_published(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Reverse only the naming mutation performed by upstream ``ViT``.

    VidEoMT loads a Hugging Face DINOv3 model and then aliases ``embeddings``
    to ``patch_embed`` and ``layer`` to ``blocks`` before training. Its released
    state therefore uses the aliased names. This function restores the original
    constructor names; it does not alter a trained tensor value.
    """

    converted: dict[str, torch.Tensor] = {}
    source_by_target: dict[str, str] = {}
    for source_key, value in state.items():
        if not source_key.startswith(DINO_BACKBONE_PREFIX):
            continue
        local_key = source_key.removeprefix(DINO_BACKBONE_PREFIX)
        if local_key.startswith("patch_embed."):
            target_key = "embeddings." + local_key.removeprefix("patch_embed.")
        elif local_key.startswith("blocks."):
            target_key = "layer." + local_key.removeprefix("blocks.")
        else:
            target_key = local_key
        if target_key in converted:
            raise ValueError(f"checkpoint conversion collision for {target_key}")
        converted[target_key] = value
        source_by_target[target_key] = source_key

    mask_key = "embeddings.mask_token"
    if mask_key in converted:
        raise ValueError("published checkpoint unexpectedly contains the deleted DINO mask token")
    converted[mask_key] = torch.zeros(1, 1, DINO_V3_L_CONFIG["hidden_size"], dtype=torch.float32)
    source_by_target[mask_key] = "constructor_zero_only; deleted by upstream before VidEoMT forward"
    return converted, source_by_target


def build_local_dinov3_bundle(
    checkpoint_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Build the local HF constructor bundle required by unmodified upstream code."""

    from safetensors.torch import save_file

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    receipt = inspect_published_checkpoint(checkpoint)
    _top_level_keys, published_state = _load_model_state(checkpoint)
    converted, source_by_target = hf_dinov3_state_from_published(published_state)

    if destination.exists():
        if not force:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        save_file(converted, temporary / "model.safetensors", metadata={"format": "pt"})
        (temporary / "config.json").write_text(
            json.dumps(DINO_V3_L_CONFIG, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        conversion_receipt = {
            "published_checkpoint": asdict(receipt),
            "config": DINO_V3_L_CONFIG,
            "converted_tensor_count": len(converted),
            "converted_numel": sum(value.numel() for value in converted.values()),
            "source_by_target": source_by_target,
        }
        (temporary / "conversion_receipt.json").write_text(
            json.dumps(conversion_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination

"""Frozen Hugging Face V-JEPA2 encoder boundary with no token reduction.

The core package intentionally does not depend on PyTorch or Transformers.
Those dependencies are imported only by ``from_pretrained`` so data-contract
tests remain lightweight and host environments can pin their own accelerator
stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError, DenseEvidence

VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"


def _strict_checkpoint_revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("V-JEPA2 checkpoint revision must be one exact lowercase commit SHA")
    return value


def _strict_positive_config_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"V-JEPA2 config {name} must be a positive integer")
    return value


def _require_positive_divisor(name: str, value: int, size: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or value <= 0
        or size <= 0
        or size % value
    ):
        raise ContractError(f"{name}={value} must divide size={size}")


def vjepa2_dense_geometry(
    *,
    frame_count: int,
    image_height: int,
    image_width: int,
    tubelet_size: int,
    patch_size: int,
) -> NDArray[np.float32]:
    """Return normalized ``(time, y, x)`` centers in Conv3D flatten order."""

    _require_positive_divisor("tubelet_size", tubelet_size, frame_count)
    _require_positive_divisor("patch_size", patch_size, image_height)
    _require_positive_divisor("patch_size", patch_size, image_width)
    temporal = frame_count // tubelet_size
    rows = image_height // patch_size
    columns = image_width // patch_size
    time = (np.arange(temporal, dtype=np.float32) + 0.5) / temporal
    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    x = (np.arange(columns, dtype=np.float32) + 0.5) / columns
    grid = np.stack(np.meshgrid(time, y, x, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid.astype(np.float32, copy=False)
    grid.setflags(write=False)
    return grid


def vjepa2_dense_timestamps(
    frame_timestamps_s: Sequence[float],
    *,
    tubelet_size: int,
    patches_per_frame: int,
) -> NDArray[np.float32]:
    """Map each output token to its tubelet-center observation timestamp."""

    timestamps = np.asarray(frame_timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.isfinite(timestamps).all():
        raise ContractError("V-JEPA2 frame timestamps must be one finite non-empty vector")
    if timestamps.size % tubelet_size:
        raise ContractError("V-JEPA2 frame count must be divisible by tubelet size")
    if timestamps.size > 1 and not (np.diff(timestamps) > 0.0).all():
        raise ContractError("V-JEPA2 frame timestamps must be strictly increasing")
    if (
        not isinstance(patches_per_frame, int)
        or isinstance(patches_per_frame, bool)
        or patches_per_frame <= 0
    ):
        raise ContractError("V-JEPA2 patches_per_frame must be positive")
    tubelet_times = timestamps.reshape(-1, tubelet_size).mean(axis=1)
    output = np.repeat(tubelet_times, patches_per_frame).astype(np.float32)
    output.setflags(write=False)
    return output


def vjepa2_context_only_role(token_count: int) -> NDArray[np.bool_]:
    """Declare clip-entangled V-JEPA rows as non-correcting action context."""

    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
        raise ContractError("V-JEPA2 token count must be a nonnegative integer")
    output = np.zeros(token_count, dtype=np.bool_)
    output.setflags(write=False)
    return output


@dataclass(slots=True)
class Vjepa2DenseEncoder:
    """Official V-JEPA2 encoder-only inference wrapped as ``DenseEvidence``.

    This is an offline/frozen evidence extractor. It preserves every encoder
    patch token. End-to-end fine-tuning must use a tensor-native host adapter
    rather than converting through this NumPy contract.
    """

    model: Any
    processor: Any
    torch: Any
    model_id: str
    checkpoint_revision: str
    device: str
    encoder_contract: str
    frames_per_clip: int
    image_size: int
    tubelet_size: int
    patch_size: int
    hidden_size: int

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = VJEPA2_MODEL_ID,
        *,
        checkpoint_revision: str = VJEPA2_MODEL_REVISION,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> Vjepa2DenseEncoder:
        """Load the official Transformers implementation and freeze it."""

        try:
            import torch

            transformers = import_module("transformers")
        except ImportError as exc:  # pragma: no cover - depends on host environment
            raise RuntimeError("V-JEPA2 extraction requires torch and transformers") from exc
        AutoModel = transformers.AutoModel
        AutoVideoProcessor = transformers.AutoVideoProcessor

        if not isinstance(model_id, str) or not model_id:
            raise ValueError("V-JEPA2 model id must be a nonempty string")
        checkpoint_revision = _strict_checkpoint_revision(checkpoint_revision)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif not isinstance(device, str) or not device:
            raise ValueError("V-JEPA2 device must be a nonempty string")
        if not isinstance(local_files_only, bool):
            raise ValueError("V-JEPA2 local_files_only must be boolean")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("a CUDA V-JEPA2 device was requested but CUDA is unavailable")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        processor = AutoVideoProcessor.from_pretrained(
            model_id,
            revision=checkpoint_revision,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=checkpoint_revision,
            local_files_only=local_files_only,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        config = model.config
        resolved_revision = getattr(config, "_commit_hash", None)
        if resolved_revision != checkpoint_revision:
            raise RuntimeError(
                "V-JEPA2 resolved checkpoint differs from the requested immutable revision"
            )
        frames_per_clip = _strict_positive_config_int("frames_per_clip", config.frames_per_clip)
        image_size = _strict_positive_config_int("image_size", config.image_size)
        tubelet_size = _strict_positive_config_int("tubelet_size", config.tubelet_size)
        patch_size = _strict_positive_config_int("patch_size", config.patch_size)
        hidden_size = _strict_positive_config_int("hidden_size", config.hidden_size)
        contract = (
            f"{model_id}@{checkpoint_revision}/encoder-last-hidden-state/"
            f"fpc{frames_per_clip}-image{image_size}-tubelet{tubelet_size}-patch{patch_size}/v2"
        )
        return cls(
            model=model,
            processor=processor,
            torch=torch,
            model_id=model_id,
            checkpoint_revision=checkpoint_revision,
            device=device,
            encoder_contract=contract,
            frames_per_clip=frames_per_clip,
            image_size=image_size,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )

    def encode_clip(
        self,
        frames: Sequence[NDArray[np.uint8]],
        frame_timestamps_s: Sequence[float],
        *,
        require_pretrained_frame_count: bool = True,
    ) -> DenseEvidence:
        """Encode one chronological RGB clip without pooling or token selection."""

        if len(frames) != len(frame_timestamps_s):
            raise ContractError("V-JEPA2 frames and timestamps must have equal lengths")
        if require_pretrained_frame_count and len(frames) != self.frames_per_clip:
            raise ContractError(
                f"V-JEPA2 checkpoint expects {self.frames_per_clip} frames, got {len(frames)}"
            )
        if not frames or len(frames) % self.tubelet_size:
            raise ContractError("V-JEPA2 clip must contain complete non-empty tubelets")
        for index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ContractError(f"V-JEPA2 frame {index} must be H-by-W-by-3 uint8")

        processed = self.processor(videos=[list(frames)], return_tensors="pt")
        pixel_values = processed["pixel_values_videos"]
        if pixel_values.ndim != 5 or pixel_values.shape[0] != 1:
            raise ContractError("V-JEPA2 processor returned an unexpected video tensor")
        processed_frames = int(pixel_values.shape[1])
        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
        if processed_frames != len(frames):
            raise ContractError("V-JEPA2 processor changed the temporal sample count")

        with self.torch.inference_mode():
            output = self.model(
                pixel_values_videos=pixel_values.to(self.device),
                skip_predictor=True,
            ).last_hidden_state
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != self.hidden_size:
            raise ContractError("V-JEPA2 model returned an unexpected dense-token tensor")

        rows = height // self.patch_size
        columns = width // self.patch_size
        temporal = processed_frames // self.tubelet_size
        expected_tokens = temporal * rows * columns
        if output.shape[1] != expected_tokens:
            raise ContractError(
                f"V-JEPA2 emitted {output.shape[1]} tokens, expected {expected_tokens}"
            )
        tokens = output[0].detach().to(dtype=self.torch.float32, device="cpu").numpy()
        if not np.isfinite(tokens).all():
            raise ContractError("V-JEPA2 output contains NaN or infinity")
        tokens.setflags(write=False)
        geometry = vjepa2_dense_geometry(
            frame_count=processed_frames,
            image_height=height,
            image_width=width,
            tubelet_size=self.tubelet_size,
            patch_size=self.patch_size,
        )
        timestamps = vjepa2_dense_timestamps(
            frame_timestamps_s,
            tubelet_size=self.tubelet_size,
            patches_per_frame=rows * columns,
        )
        confidence = np.ones(expected_tokens, dtype=np.float32)
        confidence.setflags(write=False)
        # Final V-JEPA tokens jointly attend to the complete causal clip. They
        # are therefore correlated with history already summarized by PICF's
        # prior and must not be multiplied in as an independent measurement.
        # The full bank remains available to the action expert as motion
        # context; current-frame encoders alone define the filter likelihood.
        current_measurement_valid = vjepa2_context_only_role(expected_tokens)
        return DenseEvidence(
            modality="vjepa",
            encoder_contract=self.encoder_contract,
            tokens=tokens,
            available=True,
            timestamps=timestamps,
            confidence=confidence,
            geometry=geometry,
            current_measurement_valid=current_measurement_valid,
        )

"""Pinned AnyTouch2 dense tactile evidence without semantic side decisions."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.content_addressing import canonical_mapping_sha256
from picf_next.contracts import ContractError, DenseEvidence
from picf_next.full_modal_assets import FullModalAssetManifest

ANYTOUCH2_SOURCE_COMMIT = "82c5677d9cf0176d97a1fe04745f63cd02dd6f54"
ANYTOUCH2_SENSOR_IDS = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
}
ANYTOUCH2_TOKEN_WIDTH = 768
ANYTOUCH2_TOKENS_PER_SENSOR = 398
ANYTOUCH2_GEOMETRY_WIDTH = 18


def _readonly(value: NDArray[np.generic]) -> NDArray[np.generic]:
    value.setflags(write=False)
    return value


def _sensor_id(name: str) -> int:
    try:
        return ANYTOUCH2_SENSOR_IDS[name]
    except KeyError as exc:
        raise ContractError(f"unapproved AnyTouch2 sensor {name!r}") from exc


@dataclass(frozen=True, slots=True)
class AnyTouch2DenseConfig:
    frame_count: int = 4
    stride: int = 2
    image_size: int = 224
    patch_size: int = 16
    offset: float = 130.0 / 255.0
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for value, name in (
            (self.frame_count, "frame count"),
            (self.stride, "stride"),
            (self.image_size, "image size"),
            (self.patch_size, "patch size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"AnyTouch2 {name} must be a positive integer")
        if self.frame_count % self.stride or self.image_size % self.patch_size:
            raise ValueError("AnyTouch2 dimensions must form complete spatiotemporal patches")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("AnyTouch2 dtype is unsupported")
        if self.patch_token_count + 6 != ANYTOUCH2_TOKENS_PER_SENSOR:
            raise ValueError("AnyTouch2 token geometry differs from the pinned checkpoint")

    @property
    def spatial_grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def temporal_grid(self) -> int:
        return self.frame_count // self.stride

    @property
    def patch_token_count(self) -> int:
        return self.temporal_grid * self.spatial_grid**2


def preprocess_anytouch2_clip(
    clip: NDArray[np.generic],
    background_rgb: NDArray[np.generic],
    config: AnyTouch2DenseConfig,
    *,
    torch: Any,
) -> Any:
    frames = np.asarray(clip)
    background = np.asarray(background_rgb)
    if frames.ndim != 4 or frames.shape[0] != config.frame_count or frames.shape[-1] != 3:
        raise ContractError(
            f"AnyTouch2 clip must be [{config.frame_count},H,W,3], got {frames.shape}"
        )
    if background.shape != frames.shape[1:]:
        raise ContractError("AnyTouch2 background must match one tactile RGB frame")
    clip_tensor = torch.as_tensor(np.array(frames, copy=True), dtype=torch.float32)
    background_tensor = torch.as_tensor(np.array(background, copy=True), dtype=torch.float32)
    if float(clip_tensor.max().item()) > 1.0:
        clip_tensor = clip_tensor / 255.0
    if float(background_tensor.max().item()) > 1.0:
        background_tensor = background_tensor / 255.0
    clip_tensor = (clip_tensor - background_tensor.unsqueeze(0) + config.offset).clamp(0.0, 1.0)
    clip_tensor = clip_tensor.permute(0, 3, 1, 2).contiguous()
    clip_tensor = torch.nn.functional.interpolate(
        clip_tensor,
        size=(config.image_size, config.image_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = clip_tensor.new_tensor(config.mean)[None, :, None, None]
    std = clip_tensor.new_tensor(config.std)[None, :, None, None]
    return (clip_tensor - mean) / std


def anytouch2_token_metadata(
    *,
    sensor_id: int,
    sensor_poses_world: Sequence[NDArray[np.generic]] | NDArray[np.generic],
    frame_timestamps_s: Sequence[float],
    config: AnyTouch2DenseConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Build physical metadata for `[CLS, five sensor, patch...]` rows."""

    if sensor_id not in ANYTOUCH2_SENSOR_IDS.values():
        raise ContractError("AnyTouch2 sensor id is outside the released registry")
    poses = np.asarray(sensor_poses_world, dtype=np.float32)
    if poses.shape != (config.frame_count, 4, 4) or not np.isfinite(poses).all():
        raise ContractError("AnyTouch2 sensor poses must be one finite transform per frame")
    timestamps = np.asarray(frame_timestamps_s, dtype=np.float64)
    if timestamps.shape != (config.frame_count,) or not np.isfinite(timestamps).all():
        raise ContractError("AnyTouch2 frame timestamps violate the four-frame contract")
    if (timestamps < 0.0).any() or (np.diff(timestamps) < 0.0).any():
        raise ContractError("AnyTouch2 timestamps must be chronological and nonnegative")
    # A tubelet becomes deploy-visible only when its final source frame arrives.
    # Endpoint time keeps the final tubelet eligible as a current measurement;
    # a midpoint timestamp would contradict DenseEvidence.current_measurement_valid.
    temporal_times = timestamps.reshape(config.temporal_grid, config.stride)[:, -1]
    temporal_poses = poses.reshape(config.temporal_grid, config.stride, 4, 4)[:, -1]
    newest = float(temporal_times[-1])
    special_count = 6
    output_timestamps = np.concatenate(
        (
            np.full(special_count, newest, dtype=np.float32),
            np.repeat(temporal_times, config.spatial_grid**2).astype(np.float32),
        )
    )
    current = np.concatenate(
        (
            np.ones(special_count, dtype=np.bool_),
            np.repeat(
                np.arange(config.temporal_grid) == config.temporal_grid - 1,
                config.spatial_grid**2,
            ),
        )
    )

    geometry = np.zeros((ANYTOUCH2_TOKENS_PER_SENSOR, ANYTOUCH2_GEOMETRY_WIDTH), dtype=np.float32)
    geometry[:, 5] = float(sensor_id) / 19.0
    geometry[:6, 6:] = poses[-1, :3].reshape(1, 12)
    geometry[0, 0] = 1.0
    geometry[0, 3] = 1.0
    geometry[1:6, 0] = 1.0
    geometry[1:6, 4] = 1.0
    time = (np.arange(config.temporal_grid, dtype=np.float32) + 0.5) / config.temporal_grid
    center = (np.arange(config.spatial_grid, dtype=np.float32) + 0.5) / config.spatial_grid
    patch_geometry = np.stack(np.meshgrid(time, center, center, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    geometry[6:, :3] = patch_geometry
    geometry[6:, 6:] = np.repeat(
        temporal_poses[:, None, :3, :],
        config.spatial_grid**2,
        axis=1,
    ).reshape(-1, 12)
    return (
        _readonly(geometry),
        _readonly(output_timestamps.astype(np.float32, copy=False)),
        _readonly(current.astype(np.bool_, copy=False)),
    )


def _unwrap_checkpoint(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("unsupported AnyTouch2 checkpoint format")
    if raw and all(isinstance(key, str) for key in raw):
        if all(hasattr(value, "shape") for value in raw.values()):
            return raw
        for key in ("state_dict", "model"):
            value = raw.get(key)
            if isinstance(value, dict) and value and all(isinstance(item, str) for item in value):
                return value
    raise RuntimeError("unsupported AnyTouch2 checkpoint state dictionary")


def _encoder_state(raw_state: dict[str, Any], model: Any) -> dict[str, Any]:
    selected = {
        key.replace("touch_mae_model.", ""): value
        for key, value in raw_state.items()
        if "touch_mae_model" in key and "decoder" not in key and "mask_token" not in key
    }
    if not selected:
        raise RuntimeError("AnyTouch2 checkpoint contains no encoder weights")
    complete = model.state_dict()
    unexpected = sorted(set(selected).difference(complete))
    if unexpected:
        raise RuntimeError(f"AnyTouch2 encoder checkpoint has unexpected keys: {unexpected[:8]}")
    for key, value in selected.items():
        if tuple(value.shape) != tuple(complete[key].shape):
            raise RuntimeError(f"AnyTouch2 encoder checkpoint shape changed for {key!r}")
        complete[key] = value
    return complete


@dataclass(slots=True)
class AnyTouch2DenseEncoder:
    """Frozen tactile encoder; validity is supplied by measured data, not this model."""

    model: Any
    torch: Any
    device: str
    dtype: Any
    config: AnyTouch2DenseConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    encoder_contract: str

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        device: str | None = None,
        config: AnyTouch2DenseConfig | None = None,
        verify_asset: bool = True,
    ) -> AnyTouch2DenseEncoder:
        try:
            import torch
            from transformers import AutoConfig

            from picf_next.encoders.vendor.anytouch2.tactile_mae import TactileVideoMAE
        except ImportError as exc:  # pragma: no cover - accelerator environment
            raise RuntimeError("AnyTouch2 requires torch and transformers") from exc
        resolved = config or AnyTouch2DenseConfig()
        manifest = FullModalAssetManifest.load(manifest_path, verify_files=verify_asset)
        asset = manifest.asset("anytouch")
        if asset.upstream_commit != ANYTOUCH2_SOURCE_COMMIT:
            raise RuntimeError("AnyTouch2 source identity differs from the production adapter")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("AnyTouch2 CUDA was requested but is unavailable")
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[resolved.dtype]
        config_path = Path(__file__).resolve().parent / "vendor" / "anytouch2" / "CLIP-B-16"
        hf_config = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
        # AnyTouch2 was released against Transformers 4.31, where CLIP used the
        # eager attention path implicitly. Newer Transformers requires the
        # implementation to be materialized on the nested vision config.
        hf_config.vision_config._attn_implementation = "eager"
        model = TactileVideoMAE(
            hf_config,
            num_frames=resolved.frame_count,
            stride=resolved.stride,
            mask_ratio=0.0,
        )
        raw = torch.load(asset.persistent_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_encoder_state(_unwrap_checkpoint(raw), model), strict=True)
        model.requires_grad_(False)
        model.eval()
        model.to(device=device, dtype=dtype)
        numerical_contract = canonical_mapping_sha256(
            "picf-next.anytouch2-numerical-contract/v1",
            {
                "dtype": resolved.dtype,
                "frame_count": resolved.frame_count,
                "image_size": resolved.image_size,
                "mean": resolved.mean,
                "offset": resolved.offset,
                "patch_size": resolved.patch_size,
                "sensor_ids": ANYTOUCH2_SENSOR_IDS,
                "std": resolved.std,
                "stride": resolved.stride,
            },
        )
        contract = (
            f"anytouch2-4frames@{asset.sha256}/"
            f"official-probe-dense398x768-explicit-validity@{numerical_contract}/v2"
        )
        return cls(
            model=model,
            torch=torch,
            device=device,
            dtype=dtype,
            config=resolved,
            checkpoint_path=asset.persistent_path,
            checkpoint_sha256=asset.sha256,
            encoder_contract=contract,
        )

    def missing_evidence(self) -> DenseEvidence:
        return DenseEvidence(
            modality="anytouch",
            encoder_contract=self.encoder_contract,
            tokens=_readonly(np.empty((0, ANYTOUCH2_TOKEN_WIDTH), dtype=np.float32)),
            available=False,
            timestamps=_readonly(np.empty(0, dtype=np.float32)),
            confidence=_readonly(np.empty(0, dtype=np.float32)),
            geometry=_readonly(np.empty((0, ANYTOUCH2_GEOMETRY_WIDTH), dtype=np.float32)),
            group_ids=_readonly(np.empty(0, dtype=np.int64)),
            current_measurement_valid=_readonly(np.empty(0, dtype=np.bool_)),
        )

    def encode_active_sensors(
        self,
        *,
        clips_by_sensor: Mapping[str, NDArray[np.generic]],
        sensor_types_by_stream: Mapping[str, str],
        backgrounds_by_sensor: Mapping[str, NDArray[np.generic]],
        poses_world_by_sensor: Mapping[str, Sequence[NDArray[np.generic]] | NDArray[np.generic]],
        timestamps_by_sensor: Mapping[str, Sequence[float]],
    ) -> DenseEvidence:
        """Encode only caller-validated measured contacts; never infer availability."""

        names = tuple(sorted(clips_by_sensor))
        if not names:
            return self.missing_evidence()
        expected_names = set(names)
        for mapping, label in (
            (sensor_types_by_stream, "hardware type"),
            (backgrounds_by_sensor, "background"),
            (poses_world_by_sensor, "pose"),
            (timestamps_by_sensor, "timestamp"),
        ):
            if set(mapping) != expected_names:
                raise ContractError(f"AnyTouch2 {label} sensors differ from active clips")
        inputs: list[Any] = []
        sensor_ids: list[int] = []
        metadata: list[NDArray[np.float32]] = []
        timestamps: list[NDArray[np.float32]] = []
        current: list[NDArray[np.bool_]] = []
        for name in names:
            sensor_id = _sensor_id(sensor_types_by_stream[name])
            inputs.append(
                preprocess_anytouch2_clip(
                    clips_by_sensor[name],
                    backgrounds_by_sensor[name],
                    self.config,
                    torch=self.torch,
                )
            )
            sensor_ids.append(sensor_id)
            geometry, sensor_timestamps, sensor_current = anytouch2_token_metadata(
                sensor_id=sensor_id,
                sensor_poses_world=poses_world_by_sensor[name],
                frame_timestamps_s=timestamps_by_sensor[name],
                config=self.config,
            )
            metadata.append(geometry)
            timestamps.append(sensor_timestamps)
            current.append(sensor_current)
        newest = np.asarray([value.max() for value in timestamps], dtype=np.float32)
        if not np.allclose(newest, newest[0], rtol=0.0, atol=1e-7):
            raise ContractError("active AnyTouch2 sensors must be time-synchronized")
        batch = self.torch.stack(inputs, dim=0).to(device=self.device, dtype=self.dtype)
        ids = self.torch.tensor(sensor_ids, device=self.device, dtype=self.torch.long)
        context = self.torch.inference_mode()
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.startswith("cuda") and self.dtype != self.torch.float32
            else contextlib.nullcontext()
        )
        with context, autocast:
            output = self.model(batch, ids, probe=True)
        expected_shape = (len(names), ANYTOUCH2_TOKENS_PER_SENSOR, ANYTOUCH2_TOKEN_WIDTH)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "AnyTouch2 output violates the frozen dense contract: "
                f"actual={tuple(output.shape)} expected={expected_shape}"
            )
        tokens = (
            output.detach()
            .to(device="cpu", dtype=self.torch.float32)
            .numpy()
            .reshape(-1, ANYTOUCH2_TOKEN_WIDTH)
        )
        count = tokens.shape[0]
        return DenseEvidence(
            modality="anytouch",
            encoder_contract=self.encoder_contract,
            tokens=_readonly(tokens.astype(np.float32, copy=False)),
            available=True,
            timestamps=_readonly(np.concatenate(timestamps).astype(np.float32, copy=False)),
            confidence=_readonly(np.ones(count, dtype=np.float32)),
            geometry=_readonly(np.concatenate(metadata).astype(np.float32, copy=False)),
            group_ids=_readonly(
                np.repeat(np.arange(len(names), dtype=np.int64), ANYTOUCH2_TOKENS_PER_SENSOR)
            ),
            current_measurement_valid=_readonly(
                np.concatenate(current).astype(np.bool_, copy=False)
            ),
        )

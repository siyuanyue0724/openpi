"""Pinned SpatialLM-adapted Sonata dense geometry evidence producer."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.content_addressing import canonical_mapping_sha256
from picf_next.contracts import ContractError, DenseEvidence
from picf_next.full_modal_assets import FullModalAssetManifest

SPATIALLM_SOURCE_COMMIT = "8913c44d84a450c53e9340b13317f8cf7144a738"
SONATA_ARCHITECTURE_COMMIT = "18c09ff8d713494f78a8213792262b910977a65d"
SPATIALLM_SONATA_TOKEN_WIDTH = 512
SPATIALLM_SONATA_NATIVE_GEOMETRY_WIDTH = 3
SPATIALLM_SONATA_FULL_GEOMETRY_WIDTH = 4


def _readonly(value: NDArray[np.generic]) -> NDArray[np.generic]:
    value.setflags(write=False)
    return value


def normalize_sonata_colors(colors: NDArray[np.generic]) -> NDArray[np.float32]:
    value = np.asarray(colors, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ContractError("SpatialLM/Sonata colors must be finite N-by-3")
    if value.size and float(value.max()) > 1.0:
        value = value / 255.0
    return _readonly(np.clip(value, 0.0, 1.0).astype(np.float32, copy=False))


def sonata_grid_coordinates(
    xyz_world: NDArray[np.generic], *, voxel_size_m: float
) -> NDArray[np.int32]:
    xyz = np.asarray(xyz_world, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ContractError("SpatialLM/Sonata points must be finite N-by-3")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ContractError("SpatialLM/Sonata voxel size must be positive")
    if not xyz.shape[0]:
        return _readonly(np.empty((0, 3), dtype=np.int32))
    grid = np.floor((xyz - xyz.min(axis=0, keepdims=True)) / voxel_size_m).astype(np.int32)
    grid -= grid.min(axis=0, keepdims=True)
    return _readonly(grid)


@dataclass(frozen=True, slots=True)
class SpatialLMSonataConfig:
    voxel_size_m: float = 0.01
    stage_name: str = "enc4"
    # ``enc4`` is Sonata's native sparse bottleneck. Full-resolution
    # restoration is retained only as a parity oracle because it duplicates
    # each pooled feature over thousands of source points.
    return_full_resolution: bool = False
    maximum_points: int = 4096
    dtype: str = "float32"
    enable_flash: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("SpatialLM/Sonata voxel size must be positive")
        if self.stage_name != "enc4":
            raise ValueError("the parity SpatialLM/Sonata contract requires enc4")
        if not isinstance(self.return_full_resolution, bool):
            raise TypeError("SpatialLM/Sonata restoration mode must be boolean")
        if (
            isinstance(self.maximum_points, bool)
            or not isinstance(self.maximum_points, int)
            or self.maximum_points <= 0
        ):
            raise ValueError("SpatialLM/Sonata maximum points must be positive")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("SpatialLM/Sonata output dtype is unsupported")

    @property
    def geometry_width(self) -> int:
        return (
            SPATIALLM_SONATA_FULL_GEOMETRY_WIDTH
            if self.return_full_resolution
            else SPATIALLM_SONATA_NATIVE_GEOMETRY_WIDTH
        )


def _unwrap_state_dict(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        if raw and all(
            isinstance(key, str) and hasattr(value, "shape") for key, value in raw.items()
        ):
            return raw
        for key in ("state_dict", "model", "sonata", "encoder"):
            value = raw.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(
                    isinstance(item, str) and hasattr(tensor, "shape")
                    for item, tensor in value.items()
                )
            ):
                return value
    raise RuntimeError("unsupported SpatialLM/Sonata checkpoint format")


def _complete_strict_state(model: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    expected = model.state_dict()
    unexpected = sorted(set(checkpoint).difference(expected))
    if unexpected:
        raise RuntimeError(f"SpatialLM/Sonata checkpoint has unexpected keys: {unexpected[:8]}")
    missing = sorted(set(expected).difference(checkpoint))
    illegal_missing = [key for key in missing if "mask_token" not in key]
    if illegal_missing:
        raise RuntimeError(
            f"SpatialLM/Sonata checkpoint is missing encoder keys: {illegal_missing[:8]}"
        )
    for key, value in checkpoint.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise RuntimeError(f"SpatialLM/Sonata checkpoint shape changed for {key!r}")
        expected[key] = value
    return expected


def _restore_full_resolution_features(point: Any) -> Any:
    features = point.feat
    cursor = point
    while "pooling_parent" in cursor and "pooling_inverse" in cursor:
        inverse = cursor["pooling_inverse"].long()
        if inverse.numel():
            minimum = int(inverse.min().item())
            maximum = int(inverse.max().item())
            if minimum < 0 or maximum >= int(features.shape[0]):
                raise RuntimeError("SpatialLM/Sonata pooling inverse is out of bounds")
        features = features[inverse]
        cursor = cursor["pooling_parent"]
    return features


@dataclass(slots=True)
class SpatialLMSonataDenseEncoder:
    """Frozen scene encoder; point relevance is decided only by the shared host."""

    model: Any
    point_type: Any
    torch: Any
    device: str
    output_dtype: Any
    config: SpatialLMSonataConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    encoder_contract: str

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        device: str | None = None,
        config: SpatialLMSonataConfig | None = None,
        verify_asset: bool = True,
    ) -> SpatialLMSonataDenseEncoder:
        try:
            import torch

            from picf_next.encoders.vendor.spatiallm_sonata import model as sonata_module
        except ImportError as exc:  # pragma: no cover - accelerator environment
            raise RuntimeError(
                "SpatialLM/Sonata requires torch, addict, spconv and torch-scatter"
            ) from exc
        resolved = config or SpatialLMSonataConfig()
        manifest = FullModalAssetManifest.load(manifest_path, verify_files=verify_asset)
        asset = manifest.asset("sonata")
        if asset.upstream_commit != SPATIALLM_SOURCE_COMMIT:
            raise RuntimeError("SpatialLM source identity differs from the production adapter")
        if asset.architecture_upstream_commit != SONATA_ARCHITECTURE_COMMIT:
            raise RuntimeError("Sonata architecture identity differs from the production adapter")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("the production SpatialLM/Sonata adapter requires CUDA")
        raw = torch.load(asset.persistent_path, map_location="cpu", weights_only=False)
        state = _unwrap_state_dict(raw)
        embedding_key = "embedding.stem.linear.weight"
        if embedding_key not in state or tuple(state[embedding_key].shape)[1] != 6:
            raise RuntimeError("SpatialLM/Sonata checkpoint is not the xyz+rgb encoder")
        enable_fourier = "input_proj.weight" in state
        flash_available = sonata_module.flash_attn is not None
        flash_enabled = resolved.enable_flash and flash_available
        model = sonata_module.Sonata(
            in_channels=6,
            order=("z", "z-trans"),
            shuffle_orders=False,
            enable_flash=flash_enabled,
            enable_fourier_encode=enable_fourier,
        )
        model.load_state_dict(_complete_strict_state(model, state), strict=True)
        model.requires_grad_(False)
        model.eval()
        model.to(device=device, dtype=torch.float32)
        output_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[resolved.dtype]
        numerical_contract = canonical_mapping_sha256(
            "picf-next.spatiallm-sonata-numerical-contract/v1",
            {
                "dtype": resolved.dtype,
                "enable_flash": flash_enabled,
                "enable_fourier": enable_fourier,
                "maximum_points": resolved.maximum_points,
                "return_full_resolution": resolved.return_full_resolution,
                "stage_name": resolved.stage_name,
                "voxel_size_m": resolved.voxel_size_m,
            },
        )
        resolution = "fullres" if resolved.return_full_resolution else "native"
        contract = (
            f"spatiallm-sonata@{asset.sha256}/"
            f"xyzrgb-{resolved.stage_name}-{resolution}-w{SPATIALLM_SONATA_TOKEN_WIDTH}"
            f"@{numerical_contract}/v3"
        )
        return cls(
            model=model,
            point_type=sonata_module.Point,
            torch=torch,
            device=device,
            output_dtype=output_dtype,
            config=resolved,
            checkpoint_path=asset.persistent_path,
            checkpoint_sha256=asset.sha256,
            encoder_contract=contract,
        )

    def missing_evidence(self) -> DenseEvidence:
        return DenseEvidence(
            modality="sonata",
            encoder_contract=self.encoder_contract,
            tokens=_readonly(np.empty((0, SPATIALLM_SONATA_TOKEN_WIDTH), dtype=np.float32)),
            available=False,
            timestamps=_readonly(np.empty(0, dtype=np.float32)),
            confidence=_readonly(np.empty(0, dtype=np.float32)),
            geometry=_readonly(np.empty((0, self.config.geometry_width), dtype=np.float32)),
            group_ids=(
                _readonly(np.empty(0, dtype=np.int64))
                if self.config.return_full_resolution
                else None
            ),
            current_measurement_valid=_readonly(np.empty(0, dtype=np.bool_)),
        )

    def _encode_stage(self, sample: dict[str, Any]) -> Any:
        point = self.point_type(sample)
        point = self.model.embedding(point)
        point.serialization(order=self.model.order, shuffle_orders=False)
        point.sparsify()
        for name, module in self.model.enc.named_children():
            point = module(point)
            if name == self.config.stage_name:
                return point
        raise RuntimeError("SpatialLM/Sonata enc4 stage is absent")

    def encode_points(
        self,
        *,
        xyz_world: NDArray[np.generic],
        colors: NDArray[np.generic],
        view_ids: NDArray[np.generic],
        timestamp_s: float,
    ) -> DenseEvidence:
        xyz = np.asarray(xyz_world, dtype=np.float32)
        color = normalize_sonata_colors(colors)
        views = np.asarray(view_ids)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ContractError("SpatialLM/Sonata points must be finite N-by-3")
        if color.shape != xyz.shape:
            raise ContractError("SpatialLM/Sonata colors must align with points")
        if views.shape != (xyz.shape[0],) or not np.issubdtype(views.dtype, np.integer):
            raise ContractError("SpatialLM/Sonata view ids must align with points")
        if ((views < 0) | (views > 1)).any():
            raise ContractError("SpatialLM/Sonata view ids must be static=0 or wrist=1")
        if not np.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ContractError("SpatialLM/Sonata timestamp must be finite and nonnegative")
        count = xyz.shape[0]
        if count == 0:
            return self.missing_evidence()
        if count > self.config.maximum_points:
            raise ContractError("SpatialLM/Sonata input exceeds its frozen point budget")
        grid = sonata_grid_coordinates(xyz, voxel_size_m=self.config.voxel_size_m)
        features = np.concatenate((xyz, color), axis=1).astype(np.float32, copy=False)
        sample = {
            "coord": self.torch.from_numpy(np.array(xyz, copy=True)).to(
                self.device, self.torch.float32
            ),
            "grid_coord": self.torch.from_numpy(np.array(grid, copy=True)).to(
                self.device, self.torch.int32
            ),
            "color": self.torch.from_numpy(np.array(color, copy=True)).to(
                self.device, self.torch.float32
            ),
            "feat": self.torch.from_numpy(features).to(self.device, self.torch.float32),
            "grid_size": float(self.config.voxel_size_m),
            "batch": self.torch.zeros(count, device=self.device, dtype=self.torch.int64),
            "offset": self.torch.tensor([count], device=self.device, dtype=self.torch.int64),
        }
        with self.torch.inference_mode(), contextlib.nullcontext():
            stage = self._encode_stage(sample)
        if self.config.return_full_resolution:
            encoded = _restore_full_resolution_features(stage)
            encoded_geometry = xyz
            group_ids: NDArray[np.int64] | None = views.astype(np.int64, copy=False)
            geometry = np.concatenate(
                (encoded_geometry, views.astype(np.float32, copy=False)[:, None]), axis=1
            )
        else:
            encoded = stage.feat
            encoded_geometry = (
                stage.coord.detach().to(device="cpu", dtype=self.torch.float32).numpy()
            )
            group_ids = None
            geometry = encoded_geometry
        output_count = int(encoded.shape[0])
        if tuple(encoded.shape) != (output_count, SPATIALLM_SONATA_TOKEN_WIDTH):
            raise RuntimeError(
                "SpatialLM/Sonata output violates the frozen dense contract: "
                f"actual={tuple(encoded.shape)} "
                f"expected={('N', SPATIALLM_SONATA_TOKEN_WIDTH)}"
            )
        if (
            geometry.shape != (output_count, self.config.geometry_width)
            or not np.isfinite(geometry).all()
        ):
            raise RuntimeError("SpatialLM/Sonata output geometry violates its frozen contract")
        tokens = encoded.detach().to(device="cpu", dtype=self.torch.float32).numpy()
        return DenseEvidence(
            modality="sonata",
            encoder_contract=self.encoder_contract,
            tokens=_readonly(tokens.astype(np.float32, copy=False)),
            available=True,
            timestamps=_readonly(np.full(output_count, timestamp_s, dtype=np.float32)),
            confidence=_readonly(np.ones(output_count, dtype=np.float32)),
            geometry=_readonly(geometry.astype(np.float32, copy=False)),
            group_ids=(None if group_ids is None else _readonly(group_ids)),
            current_measurement_valid=_readonly(np.ones(output_count, dtype=np.bool_)),
        )

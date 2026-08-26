"""Pinned V-JEPA2.1 ViT-B/384 dense temporal evidence producer."""

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
from picf_next.data.causal_video import CausalVideoClip, left_pad_causal_video_clip
from picf_next.full_modal_assets import FullModalAssetManifest

VJEPA21_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA21_CALVIN_VIEW_NAMES = ("static", "gripper")
VJEPA21_CALVIN_GEOMETRY_WIDTH = 2 + len(VJEPA21_CALVIN_VIEW_NAMES)
_STATE_PREFIXES = (
    "module.",
    "backbone.",
    "target_encoder.",
    "encoder.",
    "ema_encoder.",
    "model.",
)


def _readonly(value: NDArray[np.generic]) -> NDArray[np.generic]:
    value.setflags(write=False)
    return value


def vjepa21_current_grid_geometry(grid_size: int = 24) -> NDArray[np.float32]:
    """Return normalized `(y, x)` centers in encoder flatten order."""

    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
        raise ContractError("V-JEPA2.1 grid size must be a positive integer")
    centers = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    geometry = np.stack(np.meshgrid(centers, centers, indexing="ij"), axis=-1).reshape(-1, 2)
    return _readonly(geometry.astype(np.float32, copy=False))


def vjepa21_current_timestamp(
    frame_timestamps_s: Sequence[float], *, tubelet_size: int = 2
) -> float:
    timestamps = np.asarray(frame_timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < tubelet_size:
        raise ContractError("V-JEPA2.1 timestamps must cover the final tubelet")
    if not np.isfinite(timestamps).all() or (timestamps < 0.0).any():
        raise ContractError("V-JEPA2.1 timestamps must be finite and nonnegative")
    if timestamps.size > 1 and not (np.diff(timestamps) >= 0.0).all():
        raise ContractError("V-JEPA2.1 timestamps must be chronological")
    return float(timestamps[-tubelet_size:].mean())


def combine_vjepa21_calvin_views(
    evidence_by_view: Mapping[str, DenseEvidence],
) -> DenseEvidence:
    """Combine complete static/wrist grids with explicit non-semantic view coordinates."""

    if set(evidence_by_view) != set(VJEPA21_CALVIN_VIEW_NAMES):
        raise ContractError("CALVIN V-JEPA2.1 evidence must contain static and gripper views")
    ordered = tuple(evidence_by_view[name] for name in VJEPA21_CALVIN_VIEW_NAMES)
    contracts = {evidence.encoder_contract for evidence in ordered}
    if len(contracts) != 1:
        raise ContractError("CALVIN V-JEPA2.1 views use different encoder contracts")
    expected = Vjepa21DenseConfig()
    for evidence in ordered:
        if (
            evidence.modality != "vjepa"
            or not evidence.available
            or evidence.token_count != expected.token_count
            or evidence.tokens.shape[1] != expected.token_width
            or evidence.geometry is None
            or evidence.geometry.shape != (evidence.token_count, 2)
            or evidence.group_ids is not None
        ):
            raise ContractError("one CALVIN V-JEPA2.1 view violates the dense grid contract")
    geometry_rows: list[NDArray[np.float32]] = []
    for view_index, evidence in enumerate(ordered):
        one_hot = np.zeros((evidence.token_count, len(ordered)), dtype=np.float32)
        one_hot[:, view_index] = 1.0
        geometry_rows.append(
            np.concatenate((evidence.geometry, one_hot), axis=1).astype(np.float32, copy=False)
        )
    base_contract = next(iter(contracts))
    encoder_contract = vjepa21_calvin_encoder_contract(base_contract)
    return DenseEvidence(
        modality="vjepa",
        encoder_contract=encoder_contract,
        tokens=_readonly(
            np.concatenate([evidence.tokens for evidence in ordered], axis=0).astype(
                np.float32, copy=False
            )
        ),
        available=True,
        timestamps=_readonly(
            np.concatenate([evidence.timestamps for evidence in ordered]).astype(
                np.float32, copy=False
            )
        ),
        confidence=_readonly(
            np.concatenate([evidence.confidence for evidence in ordered]).astype(
                np.float32, copy=False
            )
        ),
        geometry=_readonly(np.concatenate(geometry_rows, axis=0)),
        current_measurement_valid=_readonly(
            np.concatenate(
                [evidence.effective_current_measurement_valid for evidence in ordered]
            ).astype(np.bool_, copy=False)
        ),
    )


def vjepa21_calvin_encoder_contract(base_contract: str) -> str:
    """Bind the frozen encoder to the complete ordered CALVIN view composition."""

    if not isinstance(base_contract, str) or not base_contract:
        raise ContractError("V-JEPA2.1 base encoder contract must be nonempty text")
    return f"{base_contract}/calvin-static-gripper-onehot/v1"


@dataclass(frozen=True, slots=True)
class Vjepa21DenseConfig:
    image_size: int = 384
    frame_count: int = 64
    patch_size: int = 16
    tubelet_size: int = 2
    # The final target-encoder layer is the released model's native dense
    # representation. Four-layer concatenation remains available as an oracle,
    # but is too wide to be the production LingBot input contract.
    feature_mode: str = "final"
    dtype: str = "bfloat16"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        for value, name in (
            (self.image_size, "image size"),
            (self.frame_count, "frame count"),
            (self.patch_size, "patch size"),
            (self.tubelet_size, "tubelet size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"V-JEPA2.1 {name} must be a positive integer")
        if self.image_size % self.patch_size or self.frame_count % self.tubelet_size:
            raise ValueError("V-JEPA2.1 input dimensions must form complete patches")
        if self.feature_mode not in {"final", "hierarchical"}:
            raise ValueError("V-JEPA2.1 feature mode must be final or hierarchical")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("V-JEPA2.1 dtype is unsupported")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def token_width(self) -> int:
        return 768 * (4 if self.feature_mode == "hierarchical" else 1)

    @property
    def token_count(self) -> int:
        return self.grid_size**2


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise RuntimeError("V-JEPA2.1 checkpoint contains a non-text state key")
        current = key
        changed = True
        while changed:
            changed = False
            for prefix in _STATE_PREFIXES:
                if current.startswith(prefix):
                    current = current[len(prefix) :]
                    changed = True
        cleaned[current] = value
    return cleaned


def _extract_state_dict(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("ema_encoder", "target_encoder", "encoder", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return _clean_state_dict(value)
        if payload and all(isinstance(key, str) for key in payload):
            return _clean_state_dict(payload)
    raise RuntimeError("unsupported V-JEPA2.1 checkpoint format")


@dataclass(slots=True)
class Vjepa21DenseEncoder:
    """Frozen exact-weight producer; object semantics remain in LingBot."""

    model: Any
    torch: Any
    device: str
    config: Vjepa21DenseConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    encoder_contract: str

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        device: str | None = None,
        config: Vjepa21DenseConfig | None = None,
        verify_asset: bool = True,
    ) -> Vjepa21DenseEncoder:
        try:
            import torch

            from picf_next.encoders.vendor.vjepa21 import vision_transformer
        except ImportError as exc:  # pragma: no cover - accelerator environment
            raise RuntimeError("V-JEPA2.1 requires torch and timm") from exc
        resolved = config or Vjepa21DenseConfig()
        manifest = FullModalAssetManifest.load(manifest_path, verify_files=verify_asset)
        asset = manifest.asset("vjepa")
        if asset.upstream_commit != VJEPA21_SOURCE_COMMIT:
            raise RuntimeError("V-JEPA2.1 source identity differs from the production adapter")
        if resolved.image_size != 384 or resolved.frame_count != 64:
            raise ValueError("the pinned V-JEPA2.1 parity checkpoint requires 64 frames at 384")
        if resolved.patch_size != 16 or resolved.tubelet_size != 2:
            raise ValueError("the pinned V-JEPA2.1 parity patch contract changed")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("V-JEPA2.1 CUDA was requested but is unavailable")
        model = vision_transformer.vit_base(
            patch_size=resolved.patch_size,
            img_size=(resolved.image_size, resolved.image_size),
            num_frames=resolved.frame_count,
            tubelet_size=resolved.tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            use_activation_checkpointing=False,
        )
        payload = torch.load(asset.persistent_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_extract_state_dict(payload), strict=True)
        model.return_hierarchical = resolved.feature_mode == "hierarchical"
        model.requires_grad_(False)
        model.eval()
        model.to(device=device, dtype=torch.float32)
        numerical_contract = canonical_mapping_sha256(
            "picf-next.vjepa21-numerical-contract/v1",
            {
                "dtype": resolved.dtype,
                "feature_mode": resolved.feature_mode,
                "frame_count": resolved.frame_count,
                "image_size": resolved.image_size,
                "mean": resolved.mean,
                "patch_size": resolved.patch_size,
                "std": resolved.std,
                "tubelet_size": resolved.tubelet_size,
            },
        )
        contract = (
            f"vjepa2.1-vitb-384@{asset.sha256}/"
            f"causal-leftpad-current-grid-{resolved.feature_mode}-w{resolved.token_width}"
            f"@{numerical_contract}/v2"
        )
        return cls(
            model=model,
            torch=torch,
            device=device,
            config=resolved,
            checkpoint_path=asset.persistent_path,
            checkpoint_sha256=asset.sha256,
            encoder_contract=contract,
        )

    def _preprocess(self, frames: Sequence[NDArray[np.uint8]]) -> Any:
        if len(frames) != self.config.frame_count:
            raise ContractError(
                f"V-JEPA2.1 requires {self.config.frame_count} frames, got {len(frames)}"
            )
        arrays: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ContractError(f"V-JEPA2.1 frame {index} must be H-by-W-by-3 uint8")
            arrays.append(array)
        tensor = self.torch.from_numpy(np.stack(arrays, axis=0)).to(self.torch.float32) / 255.0
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = self.torch.nn.functional.interpolate(
            tensor,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = tensor.new_tensor(self.config.mean).view(1, 3, 1, 1)
        std = tensor.new_tensor(self.config.std).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

    def encode_clip(
        self,
        frames: Sequence[NDArray[np.uint8]],
        frame_timestamps_s: Sequence[float],
    ) -> DenseEvidence:
        return self.encode_clips((frames,), (frame_timestamps_s,))[0]

    def encode_clips(
        self,
        frames: Sequence[Sequence[NDArray[np.uint8]]],
        frame_timestamps_s: Sequence[Sequence[float]],
    ) -> tuple[DenseEvidence, ...]:
        """Encode independent fixed clips in one numerically identical model call."""

        if not frames or len(frames) != len(frame_timestamps_s):
            raise ContractError("V-JEPA2.1 batch requires aligned nonempty clips")
        for clip_frames, clip_timestamps in zip(frames, frame_timestamps_s, strict=True):
            if len(clip_frames) != len(clip_timestamps):
                raise ContractError("V-JEPA2.1 frames and timestamps must align")
        video = self.torch.cat([self._preprocess(value) for value in frames], dim=0).to(
            device=self.device
        )
        if self.config.dtype == "bfloat16":
            autocast_dtype = self.torch.bfloat16
        elif self.config.dtype == "float16":
            autocast_dtype = self.torch.float16
        else:
            autocast_dtype = self.torch.float32
        use_autocast = self.device.startswith("cuda") and autocast_dtype != self.torch.float32
        context = (
            self.torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with self.torch.inference_mode(), context:
            output = self.model(video, training=False)
        expected_all = (
            self.config.frame_count // self.config.tubelet_size
        ) * self.config.token_count
        batch_size = len(frames)
        if tuple(output.shape) != (batch_size, expected_all, self.config.token_width):
            raise RuntimeError(
                "V-JEPA2.1 output violates the frozen dense contract: "
                f"actual={tuple(output.shape)} "
                f"expected={(batch_size, expected_all, self.config.token_width)}"
            )
        current_batch = output.reshape(
            batch_size,
            self.config.frame_count // self.config.tubelet_size,
            self.config.token_count,
            self.config.token_width,
        )[:, -1]
        tokens_batch = current_batch.detach().to(device="cpu", dtype=self.torch.float32).numpy()
        count = self.config.token_count
        geometry = vjepa21_current_grid_geometry(self.config.grid_size)
        return tuple(
            DenseEvidence(
                modality="vjepa",
                encoder_contract=self.encoder_contract,
                tokens=_readonly(tokens.astype(np.float32, copy=False)),
                available=True,
                timestamps=_readonly(
                    np.full(
                        count,
                        vjepa21_current_timestamp(
                            timestamps,
                            tubelet_size=self.config.tubelet_size,
                        ),
                        dtype=np.float32,
                    )
                ),
                confidence=_readonly(np.ones(count, dtype=np.float32)),
                geometry=geometry,
                current_measurement_valid=_readonly(np.ones(count, dtype=np.bool_)),
            )
            for tokens, timestamps in zip(
                tokens_batch,
                frame_timestamps_s,
                strict=True,
            )
        )

    def encode_causal_clip(self, clip: CausalVideoClip) -> DenseEvidence:
        """Encode a deploy-visible prefix with explicit old-contract left padding."""

        return self.encode_causal_clips((clip,))[0]

    def encode_causal_clips(
        self,
        clips: Sequence[CausalVideoClip],
    ) -> tuple[DenseEvidence, ...]:
        """Left-pad then encode independent deploy-visible prefixes as one batch."""

        if not clips:
            raise ContractError("V-JEPA2.1 causal batch must be nonempty")
        fixed = tuple(
            left_pad_causal_video_clip(clip, frame_count=self.config.frame_count) for clip in clips
        )
        return self.encode_clips(
            tuple(value.images for value in fixed),
            tuple(value.frame_timestamps_s for value in fixed),
        )

"""Exact released V-JEPA2-AC donor boundary and CALVIN causal controls.

The encoder, predictor, attention mask and preprocessing come from the pinned
Meta source. This wrapper only binds an immutable checkpoint, reproduces the
official notebook/training equations, and exposes falsifiable control errors.
It does not create PICF object rows or claim policy improvement.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin_vjepa2_ac import (
    VJEPA2_AC_FRAME_COUNT,
    CalvinVjepa2AcClip,
)

VJEPA2_AC_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA2_AC_CHECKPOINT_FILENAME = "vjepa2-ac-vitg.pt"
VJEPA2_AC_CHECKPOINT_BYTES = 11_760_743_310
VJEPA2_AC_HUB_CAPACITY_FRAMES = 64
VJEPA2_AC_TOKEN_WIDTH = 1408
VJEPA2_AC_TOKENS_PER_FRAME = 256
VJEPA2_AC_AUTO_STEPS = 2
VJEPA2_AC_LOSS_EXPONENT = 1.0


def _require_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError("V-JEPA2-AC checkpoint SHA-256 is invalid")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_complete_state_load(incompatible: Any, *, component: str) -> None:
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise ContractError(
            f"V-JEPA2-AC {component} checkpoint coverage changed: "
            f"missing={missing}, unexpected={unexpected}"
        )


def vjepa2_ac_control_actions(
    realized_motion: NDArray[np.float32],
    *,
    seed: int,
) -> dict[str, NDArray[np.float32]]:
    """Build deterministic actual/zero/reversed/shuffled action interventions."""

    actions = np.asarray(realized_motion)
    if (
        actions.shape != (VJEPA2_AC_FRAME_COUNT - 1, 7)
        or actions.dtype != np.float32
        or not np.isfinite(actions).all()
    ):
        raise ContractError("V-JEPA2-AC controls require finite float32 [7, 7] motion")
    if isinstance(seed, bool | np.bool_) or not isinstance(seed, int) or seed < 0:
        raise ContractError("V-JEPA2-AC control seed must be a non-negative integer")
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(actions.shape[0])
    if np.array_equal(permutation, np.arange(actions.shape[0])):
        permutation = np.roll(permutation, 1)

    def frozen(value: NDArray[np.float32]) -> NDArray[np.float32]:
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(contiguous.shape)

    return {
        "actual": frozen(actions),
        "zero": frozen(np.zeros_like(actions)),
        "reversed": frozen(actions[::-1]),
        "shuffled": frozen(actions[permutation]),
    }


@dataclass(frozen=True, slots=True)
class Vjepa2AcDonorConfig:
    """Frozen official Hub evaluation entrypoint for the released checkpoint."""

    image_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    sampled_frames: int = 8
    model_capacity_frames: int = VJEPA2_AC_HUB_CAPACITY_FRAMES
    normalize_representations: bool = True
    autoregressive_steps: int = VJEPA2_AC_AUTO_STEPS
    autocast_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        expected = (256, 16, 2, 8, VJEPA2_AC_HUB_CAPACITY_FRAMES, True, 2)
        actual = (
            self.image_size,
            self.patch_size,
            self.tubelet_size,
            self.sampled_frames,
            self.model_capacity_frames,
            self.normalize_representations,
            self.autoregressive_steps,
        )
        if actual != expected:
            raise ContractError("V-JEPA2-AC donor config differs from the official Hub gate")
        if self.autocast_dtype not in {"float32", "bfloat16"}:
            raise ContractError("V-JEPA2-AC autocast dtype is unsupported")


@dataclass(frozen=True, slots=True)
class Vjepa2AcPrediction:
    teacher_forced: Any
    autoregressive: Any
    teacher_forced_loss: Any
    autoregressive_loss: Any


@dataclass(slots=True)
class Vjepa2AcDonor:
    """Frozen source-faithful ViT-g encoder and 24-layer AC predictor."""

    encoder: Any
    predictor: Any
    transform: Any
    torch: Any
    device: str
    checkpoint_path: Path
    checkpoint_sha256: str
    config: Vjepa2AcDonorConfig

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        checkpoint_sha256: str,
        device: str | None = None,
        config: Vjepa2AcDonorConfig | None = None,
    ) -> Vjepa2AcDonor:
        path = Path(checkpoint_path).expanduser().resolve()
        expected_hash = _require_sha256(checkpoint_sha256)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != VJEPA2_AC_CHECKPOINT_BYTES:
            raise ContractError("V-JEPA2-AC checkpoint byte size differs from the published asset")
        if _sha256_file(path) != expected_hash:
            raise ContractError("V-JEPA2-AC checkpoint hash mismatch")
        try:
            import torch

            from picf_next._vendor.vjepa2_ac.hub import backbones
            from picf_next._vendor.vjepa2_ac_app.transforms import make_transforms
        except ImportError as exc:  # pragma: no cover - accelerator environment
            raise RuntimeError("V-JEPA2-AC requires torch, torchvision, timm and einops") from exc

        resolved = config or Vjepa2AcDonorConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("V-JEPA2-AC CUDA was requested but is unavailable")

        encoder, predictor = backbones.vjepa2_ac_vit_giant(
            pretrained=False,
            num_frames=resolved.model_capacity_frames,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("encoder"), dict):
            raise ContractError("V-JEPA2-AC checkpoint omitted the released encoder state")
        if not isinstance(payload.get("predictor"), dict):
            raise ContractError("V-JEPA2-AC checkpoint omitted the released predictor state")

        def clean(state_dict: dict[str, Any]) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for key, value in state_dict.items():
                if not isinstance(key, str):
                    raise ContractError("V-JEPA2-AC checkpoint contains a non-text state key")
                output[key.replace("module.", "").replace("backbone.", "")] = value
            return output

        encoder_coverage = encoder.load_state_dict(clean(payload["encoder"]), strict=False)
        predictor_coverage = predictor.load_state_dict(clean(payload["predictor"]), strict=True)
        _require_complete_state_load(encoder_coverage, component="encoder")
        _require_complete_state_load(predictor_coverage, component="predictor")
        del payload
        encoder.requires_grad_(False).eval().to(device)
        predictor.requires_grad_(False).eval().to(device)
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=resolved.image_size,
        )
        return cls(
            encoder=encoder,
            predictor=predictor,
            transform=transform,
            torch=torch,
            device=device,
            checkpoint_path=path,
            checkpoint_sha256=expected_hash,
            config=resolved,
        )

    def _autocast(self) -> Any:
        if not self.device.startswith("cuda") or self.config.autocast_dtype == "float32":
            return contextlib.nullcontext()
        return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)

    def encode_target(self, clip: CalvinVjepa2AcClip) -> Any:
        """Run the official per-frame duplicated-tubelet target encoder path."""

        if not isinstance(clip, CalvinVjepa2AcClip):
            raise TypeError("V-JEPA2-AC target encoding requires a typed CALVIN clip")
        video = self.transform(clip.images).unsqueeze(0).to(self.device)
        batch_size, _channels, frame_count, _height, _width = video.size()
        if frame_count != self.config.sampled_frames:
            raise RuntimeError("V-JEPA2-AC transform changed the eight-frame donor sequence")
        with self.torch.inference_mode(), self._autocast():
            target_input = (
                video.permute(0, 2, 1, 3, 4)
                .flatten(0, 1)
                .unsqueeze(2)
                .repeat(1, 1, 2, 1, 1)
            )
            target = self.encoder(target_input)
            target = target.view(batch_size, frame_count, -1, target.size(-1)).flatten(1, 2)
            if self.config.normalize_representations:
                target = self.torch.nn.functional.layer_norm(target, (target.size(-1),))
        expected_shape = (
            batch_size,
            frame_count * VJEPA2_AC_TOKENS_PER_FRAME,
            VJEPA2_AC_TOKEN_WIDTH,
        )
        if tuple(target.shape) != expected_shape:
            raise RuntimeError(
                f"V-JEPA2-AC target shape changed: actual={tuple(target.shape)} "
                f"expected={expected_shape}"
            )
        return target.detach()

    def predict(
        self,
        target: Any,
        *,
        realized_motion: NDArray[np.float32],
        states: NDArray,
    ) -> Vjepa2AcPrediction:
        """Reproduce the released teacher-forced and two-step AR equations."""

        if tuple(target.shape[1:]) != (
            self.config.sampled_frames * VJEPA2_AC_TOKENS_PER_FRAME,
            VJEPA2_AC_TOKEN_WIDTH,
        ):
            raise ContractError("V-JEPA2-AC target tensor violates the dense latent ABI")
        actions_array = np.asarray(realized_motion)
        states_array = np.asarray(states)
        if actions_array.shape != (self.config.sampled_frames - 1, 7):
            raise ContractError("V-JEPA2-AC realized motion shape changed")
        if states_array.shape != (self.config.sampled_frames, 7):
            raise ContractError("V-JEPA2-AC state shape changed")
        actions = self.torch.from_numpy(np.ascontiguousarray(actions_array)).unsqueeze(0).to(
            device=self.device,
            dtype=self.torch.float32,
        )
        state_tensor = self.torch.from_numpy(np.ascontiguousarray(states_array)).unsqueeze(0).to(
            device=self.device,
            dtype=self.torch.float32,
        )
        tokens = VJEPA2_AC_TOKENS_PER_FRAME

        def step(context: Any, action_value: Any, state_value: Any) -> Any:
            prediction = self.predictor(context, action_value, state_value)
            if self.config.normalize_representations:
                prediction = self.torch.nn.functional.layer_norm(
                    prediction,
                    (prediction.size(-1),),
                )
            return prediction

        with self.torch.inference_mode(), self._autocast():
            teacher_forced = step(target[:, :-tokens], actions, state_tensor[:, :-1])
            autoregressive_context = self.torch.cat(
                [target[:, :tokens], teacher_forced[:, :tokens]],
                dim=1,
            )
            for rollout_index in range(1, self.config.autoregressive_steps):
                prefix = rollout_index + 1
                next_frame = step(
                    autoregressive_context,
                    actions[:, :prefix],
                    state_tensor[:, :prefix],
                )[:, -tokens:]
                autoregressive_context = self.torch.cat(
                    [autoregressive_context, next_frame],
                    dim=1,
                )
            autoregressive = autoregressive_context[:, tokens:]

            def loss(prediction: Any) -> Any:
                future = target[:, tokens : prediction.size(1) + tokens]
                return self.torch.mean(self.torch.abs(prediction - future))

            teacher_forced_loss = loss(teacher_forced)
            autoregressive_loss = loss(autoregressive)
        return Vjepa2AcPrediction(
            teacher_forced=teacher_forced,
            autoregressive=autoregressive,
            teacher_forced_loss=teacher_forced_loss,
            autoregressive_loss=autoregressive_loss,
        )

    def evaluate_controls(
        self,
        clip: CalvinVjepa2AcClip,
        *,
        seed: int,
    ) -> dict[str, dict[str, float]]:
        """Score actual motion against deterministic interventions on one fixed target."""

        target = self.encode_target(clip)
        controls = vjepa2_ac_control_actions(clip.realized_motion, seed=seed)
        report: dict[str, dict[str, float]] = {}
        for name, actions in controls.items():
            prediction = self.predict(target, realized_motion=actions, states=clip.states)
            report[name] = {
                "teacher_forced_l1": float(prediction.teacher_forced_loss),
                "autoregressive_l1": float(prediction.autoregressive_loss),
            }
        return report

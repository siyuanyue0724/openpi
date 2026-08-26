from __future__ import annotations

import importlib.util
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from picf_next.lingbot_native.wsa_da3_loss import (
    WSA_DA3_QUERY_DIM,
    WSA_DA3_TEACHER_LAYERS,
    WSADA3TeacherTargets,
)

WSA_DA3_TEACHER_RUNTIME_SCHEMA = "picf-next.wsa-da3-online-teacher/v1"
WSA_DA3_CAMERA_KEYS = (
    "observation.images.camera_top",
    "observation.images.camera_wrist_left",
)
WSA_DA3_PROCESS_RESOLUTION = 504
_OFFICIAL_DA3_TEACHER_MODULE = "picf_next_upstream_wsa_da3_teacher"


@dataclass(frozen=True, slots=True)
class WSADA3TeacherReceipt:
    schema: str
    batch_size: int
    valid_view_count: int
    future_source_global_indices: tuple[int, ...]
    teacher_forward_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        if self.schema != WSA_DA3_TEACHER_RUNTIME_SCHEMA:
            raise ValueError("WSA DA3 teacher receipt has the wrong schema")
        if self.batch_size <= 0:
            raise ValueError("WSA DA3 teacher batch size must be positive")
        if not 0 < self.valid_view_count <= 2 * self.batch_size:
            raise ValueError("WSA DA3 teacher valid-view count is invalid")
        if len(self.future_source_global_indices) != self.batch_size:
            raise ValueError("WSA DA3 teacher source indices differ from its batch")
        if self.teacher_forward_seconds < 0.0 or self.total_seconds < 0.0:
            raise ValueError("WSA DA3 teacher timings must be non-negative")


def _load_official_da3_teacher_module(wsa_source_root: Path) -> ModuleType:
    source_file = (
        wsa_source_root
        / "src"
        / "lerobot"
        / "policies"
        / "WSA_Base"
        / "da3_teacher.py"
    )
    if not source_file.is_file():
        raise FileNotFoundError(
            f"official WSA DA3 teacher source is missing: {source_file}"
        )
    cached = sys.modules.get(_OFFICIAL_DA3_TEACHER_MODULE)
    if cached is not None:
        cached_file = Path(str(getattr(cached, "__file__", ""))).resolve()
        if cached_file != source_file.resolve():
            raise RuntimeError("another WSA DA3 teacher source is already loaded")
        return cached
    spec = importlib.util.spec_from_file_location(
        _OFFICIAL_DA3_TEACHER_MODULE,
        source_file,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official WSA DA3 teacher from {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_OFFICIAL_DA3_TEACHER_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_OFFICIAL_DA3_TEACHER_MODULE, None)
        raise
    return module


def _camera_chw(value: Any, *, name: str) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim != 3:
        raise ValueError(f"WSA DA3 camera {name} must be rank-three")
    if image.shape[0] == 3:
        chw = image
    elif image.shape[-1] == 3:
        chw = image.permute(2, 0, 1)
    else:
        raise ValueError(f"WSA DA3 camera {name} must have three RGB channels")
    if not chw.dtype.is_floating_point and chw.dtype is not torch.uint8:
        raise TypeError(f"WSA DA3 camera {name} must be uint8 or floating point")
    chw = chw.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(chw).all():
        raise ValueError(f"WSA DA3 camera {name} contains non-finite values")
    return F.interpolate(
        chw.unsqueeze(0),
        size=(WSA_DA3_PROCESS_RESOLUTION, WSA_DA3_PROCESS_RESOLUTION),
        mode="bilinear",
        align_corners=False,
    )[0]


def prepare_official_wsa_da3_future_views(
    future_observations: Sequence[Mapping[str, Any]],
    *,
    camera_keys: Sequence[str] = WSA_DA3_CAMERA_KEYS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build WSA's two-view future tensor without changing its teacher math."""

    if not future_observations:
        raise ValueError("WSA DA3 future observation batch cannot be empty")
    resolved_camera_keys = tuple(camera_keys)
    if (
        len(resolved_camera_keys) != 2
        or len(set(resolved_camera_keys)) != 2
        or any(not isinstance(key, str) or not key for key in resolved_camera_keys)
    ):
        raise ValueError("WSA DA3 requires exactly two distinct camera keys")
    batch_views: list[torch.Tensor] = []
    validity: list[tuple[bool, bool]] = []
    for batch_index, observation in enumerate(future_observations):
        views: list[torch.Tensor | None] = []
        valid: list[bool] = []
        for key in resolved_camera_keys:
            value = observation.get(key)
            present = value is not None
            valid.append(present)
            views.append(
                None
                if not present
                else _camera_chw(value, name=f"{batch_index}:{key}")
            )
        if not any(valid):
            raise ValueError("WSA DA3 sample has no valid future camera view")
        primary = next(view for view in views if view is not None)
        batch_views.append(
            torch.stack(
                tuple(primary if view is None else view for view in views),
                dim=0,
            )
        )
        validity.append((valid[0], valid[1]))
    return (
        torch.stack(batch_views, dim=0),
        torch.tensor(validity, dtype=torch.bool),
    )


class OnlineWSADA3TeacherRuntime:
    """Stage the unchanged official WSA DA3 teacher around one batch only."""

    def __init__(
        self,
        teacher: nn.Module,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not isinstance(teacher, nn.Module):
            raise TypeError("WSA DA3 teacher must be a torch module")
        if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("WSA DA3 target dtype is unsupported")
        self.teacher = teacher.to(device="cpu")
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.target_dtype = target_dtype

    @classmethod
    def from_official_source(
        cls,
        *,
        wsa_source_root: Path,
        da3_model_dir: Path,
        da3_source_root: Path,
        target_dtype: torch.dtype = torch.bfloat16,
        teacher_factory: Callable[..., nn.Module] | None = None,
    ) -> OnlineWSADA3TeacherRuntime:
        if not da3_model_dir.is_dir():
            raise FileNotFoundError(f"DA3 model directory is missing: {da3_model_dir}")
        if not da3_source_root.is_dir():
            raise FileNotFoundError(f"DA3 source directory is missing: {da3_source_root}")
        if teacher_factory is None:
            module = _load_official_da3_teacher_module(wsa_source_root.resolve())
            teacher_factory = getattr(module, "DA3BackboneTeacher", None)
        if not callable(teacher_factory):
            raise TypeError("official WSA source did not expose DA3BackboneTeacher")
        teacher = teacher_factory(
            model_path_or_name=str(da3_model_dir.resolve()),
            process_res=WSA_DA3_PROCESS_RESOLUTION,
            dtype=torch.bfloat16,
            teacher_layers=WSA_DA3_TEACHER_LAYERS,
            code_root=str(da3_source_root.resolve()),
        )
        feature_dim = getattr(teacher, "feature_dim", WSA_DA3_QUERY_DIM)
        teacher_layers = tuple(
            getattr(teacher, "teacher_layers", WSA_DA3_TEACHER_LAYERS)
        )
        if feature_dim != WSA_DA3_QUERY_DIM:
            raise ValueError("official WSA DA3 teacher feature width differs")
        if teacher_layers != WSA_DA3_TEACHER_LAYERS:
            raise ValueError("official WSA DA3 teacher layers differ")
        return cls(teacher, target_dtype=target_dtype)

    def build_targets(
        self,
        *,
        future_observations: Sequence[Mapping[str, Any]],
        future_source_global_indices: Sequence[int],
        device: torch.device | str,
        camera_keys: Sequence[str] = WSA_DA3_CAMERA_KEYS,
    ) -> tuple[WSADA3TeacherTargets, WSADA3TeacherReceipt]:
        started = time.perf_counter()
        indices = tuple(int(index) for index in future_source_global_indices)
        if len(indices) != len(future_observations) or any(index < 0 for index in indices):
            raise ValueError("WSA DA3 future source indices differ from its batch")
        images, view_valid = prepare_official_wsa_da3_future_views(
            future_observations,
            camera_keys=camera_keys,
        )
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("WSA DA3 requested CUDA but CUDA is unavailable")

        teacher_started = time.perf_counter()
        self.teacher.to(device=target_device)
        self.teacher.eval()
        try:
            with torch.inference_mode():
                teacher_layers = self.teacher(images.to(device=target_device))
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            teacher_seconds = time.perf_counter() - teacher_started
            host_layers = tuple(
                layer.detach().to(device="cpu", dtype=self.target_dtype)
                for layer in teacher_layers
            )
            del teacher_layers
        finally:
            self.teacher.to(device="cpu")
            if target_device.type == "cuda":
                torch.cuda.empty_cache()

        targets = WSADA3TeacherTargets(
            layers=tuple(layer.to(device=target_device) for layer in host_layers),
            view_valid=view_valid.to(device=target_device),
        )
        targets.validate()
        receipt = WSADA3TeacherReceipt(
            schema=WSA_DA3_TEACHER_RUNTIME_SCHEMA,
            batch_size=len(future_observations),
            valid_view_count=int(view_valid.sum().item()),
            future_source_global_indices=indices,
            teacher_forward_seconds=teacher_seconds,
            total_seconds=time.perf_counter() - started,
        )
        return targets, receipt

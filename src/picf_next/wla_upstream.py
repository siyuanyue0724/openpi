from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

from picf_next.wla_source_tree_identity import verify_pinned_wla_source_tree


WLA_COMMIT: Final = "155ac94eaca8b3d1ae0789ae298fc55e37936081"
WLA_CRITICAL_FILES: Final[dict[str, str]] = {
    "models/model.py": "73f2c65f0b6450aeb5f5a4f5687fe0e0afefd0c0200f3fddeb29869658ac7c2f",
    "models/action_model/action_model.py": (
        "89c620c9e09d18e09d958eb1dc4fe5877d8290531d1b5a4acaab87465e3ec9d5"
    ),
    "models/action_model/cross_attention_dit.py": (
        "216a94f32172fe30d5bea07309b7402f9ef543fc115e73be82585ae475edbe74"
    ),
    "configs/libero_all_image_action.yaml": (
        "af7623ea09aa3596da6c9c1a07e5fe7c4995830e8342eaf0aa731a614888584f"
    ),
    "train.py": "e5972641ae13dd4efd0faa5643b5afd0afe702eec09025775e90a5d83ab235ed",
    "models/wla.py": "17dd50632deee8bf6cd037b875b324d61fd0e49a09481090e75e8b36f51f18d8",
    "dataset.py": "7d7e6ba8c5b0115bde1d1d792bcca556848182ec0c6c105ad7bf19d60083ad3f",
    "utils/transforms.py": "6f7bfaddd0a379948642d7d970f13b235b2ac73d23039d2361bf56696daf8133",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WLASourceReceipt:
    root: Path
    commit: str
    files: tuple[tuple[str, str], ...]
    tree_file_count: int | None = None
    tree_sha256: str | None = None
    tree_receipt_sha256: str | None = None


def verify_wla_source(root: Path | str) -> WLASourceReceipt:
    """Verify the untouched WLA donor before any of its modules are imported."""

    source_root = Path(root).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("WLA source root must be a directory")
    observed: list[tuple[str, str]] = []
    for relative, expected_digest in WLA_CRITICAL_FILES.items():
        candidate = source_root / relative
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(source_root):
            raise ValueError(f"WLA source path escapes the pinned root: {relative}")
        if not resolved.is_file():
            raise ValueError(f"WLA source path is not a regular file: {relative}")
        digest = _sha256_file(resolved)
        if digest != expected_digest:
            raise ValueError(
                f"WLA source digest mismatch for {relative}: "
                f"expected {expected_digest}, observed {digest}"
            )
        observed.append((relative, digest))
    return WLASourceReceipt(
        root=source_root,
        commit=WLA_COMMIT,
        files=tuple(observed),
    )


@dataclass(frozen=True)
class WLAActionSymbols:
    source: WLASourceReceipt
    action_head: type


@dataclass(frozen=True)
class WLAWorldSymbols:
    source: WLASourceReceipt
    mllm_in_context: type
    wla: type
    qwen2_encoder: type
    rms_norm: type
    qwen2_config: type
    sana_transformer: type
    autoencoder: type
    flow_scheduler: type
    density_sampler: object
    loss_weighting: object
    resize_with_pad: object


def load_wla_action_symbols(root: Path | str) -> WLAActionSymbols:
    """Import the exact upstream action class from a hash-verified checkout.

    WLA uses the generic top-level package name ``models``. The ADR-224 trainer
    runs in a dedicated process, so this loader rejects an existing unrelated
    package instead of silently binding the upstream relative imports to it.
    """

    tree_receipt = verify_pinned_wla_source_tree(root)
    receipt = verify_wla_source(root)
    receipt = WLASourceReceipt(
        root=receipt.root,
        commit=receipt.commit,
        files=receipt.files,
        tree_file_count=tree_receipt.file_count,
        tree_sha256=tree_receipt.tree_sha256,
        tree_receipt_sha256=tree_receipt.receipt_sha256,
    )
    existing = sys.modules.get("models")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        existing_path = None if existing_file is None else Path(existing_file).resolve()
        if existing_path is None or not existing_path.is_relative_to(receipt.root):
            raise RuntimeError(
                "cannot import pinned WLA source because an unrelated top-level "
                "'models' package is already loaded"
            )

    root_text = str(receipt.root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        module: ModuleType = importlib.import_module("models.action_model.action_model")
    except Exception:
        if inserted:
            sys.path.remove(root_text)
        raise
    action_head = getattr(module, "LayerwiseFlowmatchingActionHead", None)
    if not isinstance(action_head, type):
        raise RuntimeError("pinned WLA action module lost its published action-head class")
    return WLAActionSymbols(
        source=receipt,
        action_head=action_head,
    )


def load_wla_world_symbols(root: Path | str) -> WLAWorldSymbols:
    """Import WLA's exact world classes and objective primitives."""

    action = load_wla_action_symbols(root)
    model_module: ModuleType = importlib.import_module("models.model")
    wla_module: ModuleType = importlib.import_module("models.wla")
    transform_path = action.source.root / "utils/transforms.py"
    transform_spec = importlib.util.spec_from_file_location(
        "picf_pinned_wla_transforms",
        transform_path,
    )
    if transform_spec is None or transform_spec.loader is None:
        raise RuntimeError("cannot load the pinned WLA target-image transform")
    transform_module = importlib.util.module_from_spec(transform_spec)
    transform_spec.loader.exec_module(transform_module)
    values = {
        "mllm_in_context": getattr(model_module, "MLLMInContext", None),
        "wla": getattr(wla_module, "WLA", None),
        "qwen2_encoder": getattr(model_module, "Qwen2Encoder", None),
        "rms_norm": getattr(model_module, "RMSNorm", None),
        "qwen2_config": getattr(model_module, "Qwen2Config", None),
        "sana_transformer": getattr(model_module, "SanaTransformer2DModel", None),
        "autoencoder": getattr(wla_module, "AutoencoderDC", None),
        "flow_scheduler": getattr(wla_module, "FlowMatchEulerDiscreteScheduler", None),
        "density_sampler": getattr(wla_module, "compute_density_for_timestep_sampling", None),
        "loss_weighting": getattr(wla_module, "compute_loss_weighting_for_sd3", None),
        "resize_with_pad": getattr(transform_module, "resize_with_pad", None),
    }
    function_names = {"density_sampler", "loss_weighting", "resize_with_pad"}
    if any(
        not isinstance(values[name], type)
        for name in values
        if name not in function_names
    ):
        raise RuntimeError("pinned WLA world source lost one or more published classes")
    if any(
        not callable(values[name])
        for name in ("density_sampler", "loss_weighting", "resize_with_pad")
    ):
        raise RuntimeError("pinned WLA world source lost its published objective primitives")
    return WLAWorldSymbols(source=action.source, **values)

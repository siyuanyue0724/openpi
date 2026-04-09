from .config import VjepaVisualConfig
from .history import VisualClipBuffer

__all__ = [
    "VisualClipBuffer",
    "Vjepa2VisualEncoder",
    "VjepaFeatureMap",
    "VjepaVisualConfig",
    "vjepa_runtime_available",
]


def __getattr__(name: str):
    if name in {"Vjepa2VisualEncoder", "VjepaFeatureMap", "vjepa_runtime_available"}:
        from . import wrapper as _wrapper

        return getattr(_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

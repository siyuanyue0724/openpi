from .config import PosteriorConfig
from .contracts import PointExpertState
from .contracts import PosteriorDebugMetrics
from .contracts import PosteriorState
from .contracts import VisualExpertState

__all__ = [
    "PointExpertState",
    "PointOnlyPosteriorPipeline",
    "PointVisualPosteriorPipeline",
    "PosteriorConfig",
    "PosteriorDebugMetrics",
    "PosteriorState",
    "VisualExpertState",
]


def __getattr__(name: str):
    if name == "PointOnlyPosteriorPipeline":
        from .pipeline import PointOnlyPosteriorPipeline

        return PointOnlyPosteriorPipeline
    if name == "PointVisualPosteriorPipeline":
        from .pipeline_visual import PointVisualPosteriorPipeline

        return PointVisualPosteriorPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

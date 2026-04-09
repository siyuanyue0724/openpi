from openpi.picf.sonata.config import SonataPointConfig

__all__ = [
    "SonataPointConfig",
    "SonataPointFeatureExtractor",
    "SonataPointFeatures",
    "sonata_runtime_available",
]


def __getattr__(name: str):
    if name in {"SonataPointFeatureExtractor", "SonataPointFeatures", "sonata_runtime_available"}:
        from openpi.picf.sonata import wrapper as _wrapper

        return getattr(_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

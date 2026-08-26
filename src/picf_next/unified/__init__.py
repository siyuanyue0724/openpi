"""Unified PICF probabilistic token-graph primitives.

The historical discovery/filter/action-sidecar stack remains available for
controls.  This package implements the owner-approved unified architecture from
ADR 65 without importing it from :mod:`picf_next` at package import time, so the
NumPy-only core can still be used without PyTorch.
"""

from picf_next.unified.state import GeometrySchema, UnifiedBeliefState

__all__ = ["GeometrySchema", "UnifiedBeliefState"]

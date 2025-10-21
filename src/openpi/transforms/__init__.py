# openpi/transforms/__init__.py
"""
Package initializer for openpi.transforms.

- Re-export all Pi0.5 core transforms from ._base (unchanged semantics)
- Export Sonata / point-cloud transforms as separate submodules (no inline duplicates)
- Export ValidatePointCloud (strict fail-fast) from a small standalone file.
"""

# 1) Pi0.5 核心语义（保持不变）
from ._base import *  # noqa: F401,F403

# 为避免 _base 中历史遗留的 Sonata 内联类在命名空间造成困惑，
# 先显式删除这些同名符号；随后由子模块提供权威实现。
for _name in ("DecodeLiberoDepth", "DepthToPointCloud", "LiberoInputsKeepExtras", "ValidatePointCloud"):
    if _name in globals():
        try:
            del globals()[_name]
        except Exception:
            pass

# 2) Sonata / 点云
from .decode_libero_depth import DecodeLiberoDepth  # noqa: F401
from .depth_to_pointcloud import DepthToPointCloud  # noqa: F401
from .pointcloud_attach import (  # noqa: F401
    AttachPointCloudToObservation,
    AttachPointCloudToLiberoObservation,
)
from .libero_keep_extras import LiberoInputsKeepExtras  # noqa: F401
from .state_adapters import SplitStateToDroid          # noqa: F401
from .inject_prompt import InjectPromptFromTaskIndex   # noqa: F401

# 3) 严格校验（fail-fast）
from .validate_pointcloud import ValidatePointCloud    # noqa: F401

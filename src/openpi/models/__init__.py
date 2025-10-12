from enum import Enum

# 新增: 定义点云编码后端类型枚举（Scenescript 或 Sonata）
class PointBackboneType(Enum):
    SCENESCRIPT = "scenescript"
    SONATA = "sonata"

# 新增: 定义点云特征投影器类型枚举（线性或 MLP）
class ProjectorType(Enum):
    LINEAR = "linear"
    MLP = "mlp"

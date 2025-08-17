# configs/pi0_fast_sonata.py
from openpi.training.config import TrainConfig, SimpleDataConfig, AssetsConfig, DataConfig, ModelType, ValidatePointCloud
from openpi.models.pi0_fast import Pi0FASTConfig, PointBackboneType, ProjectorType
import openpi.policies.libero_policy as libero_policy
import openpi.transforms as T

# 使用 LIBERO-10 (LeRobot + mask/depth) + SONATA 点云
config = TrainConfig(
    name="pi0_fast_sonata_libero10",
    model=Pi0FASTConfig(
        action_dim=7,                 # ★ 数据集动作是 7 维
        action_horizon=10,
        point_backbone_type=PointBackboneType.SONATA,
        projector_type=ProjectorType.LINEAR,
        point_feat_dim=6,            # ★ extras=xyz(3)+rgb(3)=6；总列=3(grid)+6=9
    ),
    data=SimpleDataConfig(
        # A) 走 HF 缓存（最省事）
        repo_id="binhng/libero_10_lerobot_mask_depth",
        # B) 若必须走本地，可把 repo_id 改成你的本地绝对路径；或采用下文“可选小补丁”支持 repo_root
        assets=AssetsConfig(asset_id="libero_10_lerobot_mask_depth"),
        # === transforms ===
        # 1) 数据集→推理接口 键名对齐（只做重命名/重排）
        base_config=DataConfig(
            prompt_from_task=True,
            repack_transforms=T.Group(inputs=[
                T.RepackTransform({
                    # 图像（LiberoInputs 预期的键）
                    "observation/image":        "observation.images.image",
                    "observation/wrist_image":  "observation.images.wrist_image",
                    "observation/state":        "observation.state",
                    # 动作：把 action → actions，匹配默认 action_sequence_keys=("actions",)
                    "actions":                  "action",
                    # 深度：供 DecodeLiberoDepth 使用
                    "depth/front/raw":          "observation.images.image_depth",
                    "depth/wrist/raw":          "observation.images.wrist_depth",
                })
            ]),
        ),
        # 2) 深度解码 → 点云 → 校验 → Libero 输入/输出
        data_transforms=lambda model: T.Group(
            inputs=[
                # 深度解码（自动识别 CHW / HWC、三通道相同 or 24-bit 打包）
                T.DecodeLiberoDepth(
                    src_keys=["depth/front/raw", "depth/wrist/raw"],
                    dst_keys=["depth/front/decoded", "depth/wrist/decoded"],
                    scale=None,   # 先相对尺度；如确认毫米→米，改 0.001
                ),
                # 深度 → 点云（无内参：相对尺度；有内参：米制）
                T.DepthToPointCloud(
                    depth_map={"front": "depth/front/decoded", "wrist": "depth/wrist/decoded"},
                    # 颜色从 repack 后的键取（LiberoInputs 也会用到这些键）
                    rgb_map={"front": "observation/image", "wrist": "observation/wrist_image"},
                    intrinsics=None,  # 如果你有 fx,fy,cx,cy，可填 dict 得到米制点云
                    stride=4,
                    out_key="pointcloud",
                ),
                # 早失败校验（与模型 point_feat_dim=6 对齐）
                ValidatePointCloud(key="pointcloud", feat_dim=getattr(model, "point_feat_dim", 6), min_points=1),
                # Libero 输入规范化（PI0_FAST 分支）
                libero_policy.LiberoInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST),
            ],
            outputs=[libero_policy.LiberoOutputs()],
        ),
        # 模型 transforms 仍走默认（ResizeImages, TokenizeFASTInputs 等由 ModelTransformFactory 提供）
    ),
)

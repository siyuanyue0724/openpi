from openpi.training.config import TrainConfig, SimpleDataConfig, AssetsConfig, DataConfig, ModelType
from openpi.models.pi0_fast import Pi0FASTConfig, PointBackboneType, ProjectorType
import openpi.policies.droid_policy as droid_policy
import openpi.transforms as _transforms

# 定义 Pi0FAST + Sonata 点云编码 模型的训练配置，注意，如果要改回libero，dim等超参数需要做对应修改，具体参考config.py里面的
config = TrainConfig(
    name="pi0_fast_sonata",
    model=Pi0FASTConfig(
        action_dim=8,
        action_horizon=10,
        point_backbone_type=PointBackboneType.SONATA,
        projector_type=ProjectorType.LINEAR,
        point_feat_dim=9,
    ),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    # 如果需要初始化预训练权重，可在此指定 CheckpointWeightLoader
    # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    # 可能需要改成lora，后面再看
)

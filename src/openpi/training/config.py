"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import numpy as np

import openpi.models.model as _model
from openpi.models.tokenizer import (FAST_POINT_START_ID as POINT_START_ID, FAST_POINT_END_ID as POINT_END_ID, POINT_START as POINT_START_TOKEN, POINT_END as POINT_END_TOKEN)
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

import openpi.models.pi0_fast_sonata as pi0_fast_sonata

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

# 记录一下映射（此处仅日志，不做任何自动推断/退化）
logging.info(
    "[config] SpatialLM fence ids (PaliGemma vocab tail): %s=%d, %s=%d",
    POINT_START_TOKEN, POINT_START_ID, POINT_END_TOKEN, POINT_END_ID
)

# 若 transforms 中已有 MapPrompt，优先使用；否则提供一个极简 fallback。
try:
    MapPrompt = _transforms.MapPrompt  # type: ignore[attr-defined]
except AttributeError:
    @dataclasses.dataclass(frozen=True)
    class MapPrompt:  # fallback：尽量与 Group 调用方式兼容（就地改 prompt）
        fn: Any
        def __call__(self, batch: dict) -> dict:
            # 适配标量字符串或 list[str]
            prompt = batch.get("prompt", None)
            if isinstance(prompt, (list, tuple)):
                batch["prompt"] = [self.fn(p) for p in prompt]  # type: ignore[call-arg]
            else:
                batch["prompt"] = self.fn(prompt)               # type: ignore[call-arg]
            return batch

# 仅生产用：必须“恰好一对”窗口标记，且顺序正确；否则抛错（不做自动补齐/退化）
@dataclasses.dataclass(frozen=True)
class RequirePointWindow:
    def __call__(self, batch: dict) -> dict:
        prompt = batch.get("prompt", "")
        def _ok(s: str | None) -> bool:
            s = (s or "")
            c_start = s.count(POINT_START_TOKEN)
            c_end   = s.count(POINT_END_TOKEN)
            if c_start != 1 or c_end != 1:
                return False
            i_start = s.find(POINT_START_TOKEN)
            i_end   = s.find(POINT_END_TOKEN)
            return i_start != -1 and i_end != -1 and i_start < i_end
        if isinstance(prompt, (list, tuple)):
            if not all(_ok(p) for p in prompt):
                raise ValueError(
                    "Prompt 必须恰好包含一次且顺序正确的 <|point_start|><|point_end|> 窗口标记。"
                )
        else:
            if not _ok(prompt):
                raise ValueError(
                    "Prompt 必须恰好包含一次且顺序正确的 <|point_start|><|point_end|> 窗口标记。"
                )
        return batch


# 严格校验：点云必须存在且满足形状/类型/数值约束（dbg 也走真严模式）
@dataclasses.dataclass(frozen=True)
class ValidatePointCloud:
    """
    校验单路点云：
      observation.point_clouds[key]        : [N, 3 + feat_dim] float32   （样本内点数 N）
      observation.point_cloud_masks[key]   : bool 或 [1]                 （帧级掩码，每样本一个）
    校验不过直接抛错（不做任何自动修复/退化）。
    """
    key: str = "pointcloud"
    feat_dim: int = 6
    min_points: int = 1
    allow_mask_all_false: bool = False
    def __call__(self, batch: dict) -> dict:
        # 兼容两种放置位置：顶层（规范）或 legacy 的 observation 下
        pcs = None
        pms = None
        if "point_clouds" in batch:
            pcs = batch["point_clouds"]
            pms = batch.get("point_cloud_masks", None)
        elif "observation" in batch and isinstance(batch["observation"], dict):
            pcs = batch["observation"].get("point_clouds", None)
            pms = batch["observation"].get("point_cloud_masks", None)
        if pcs is None or self.key not in pcs:
            raise KeyError(f"missing point_clouds['{self.key}']")

        x = np.asarray(pcs[self.key])
        # 允许单样本 2D [N, F]，或已带 batch 维度的 3D [B, N, F]（要求 B==1）
        if x.ndim == 3:
            if x.shape[0] != 1:
                raise ValueError(f"pointcloud first dim must be 1 if batched; got B={x.shape[0]}")
            x = x[0]
        if x.ndim != 2 or x.shape[1] != 3 + self.feat_dim:
            raise ValueError(f"pointcloud shape must be [N, {3 + self.feat_dim}], got {tuple(x.shape)}")
        if x.dtype != np.float32:
            raise TypeError(f"pointcloud dtype must be float32, got {x.dtype}")
        if not np.isfinite(x).all():
            raise ValueError("pointcloud contains NaN/Inf")
        if x.shape[0] < self.min_points:
            raise ValueError(f"pointcloud must have at least {self.min_points} points, got {x.shape[0]}")
        if pms is None or self.key not in pms:
            raise KeyError(f"missing point_cloud_masks['{self.key}']")
        m = np.asarray(pms[self.key])
        if m.dtype != np.bool_:
            raise TypeError(f"mask dtype must be bool, got {m.dtype}")
        # 帧级掩码：标量 bool、[1] 或 [B]（要求 B==1）
        if m.ndim == 0:
            valid = bool(m)
        elif m.ndim == 1 and m.shape[0] == 1:
            valid = bool(m[0])
        elif m.ndim == 1 and m.shape[0] > 1:
            # 已带 batch：要求 B==1（与上面的点云检查一致）
            if m.shape[0] != 1:
                raise ValueError(f"mask is batched but B={m.shape[0]} (expected 1)")
            valid = bool(m[0])
        else:
            raise ValueError(f"mask must be a frame-level bool (scalar or [1]), got shape {tuple(m.shape)}")
        if not self.allow_mask_all_false and not valid:
            raise ValueError("frame-level mask is False")
        return batch

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                # 复用同一个 tokenizer，避免重复加载 AutoProcessor
                _fast_tok = _tokenizer.FASTTokenizer(model_config.max_token_len)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_fast_tok),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _fast_tok,
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
        TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    # ------------------------------------------------------------------
    # Inference: Pi0FAST-Sonata  (真实 DROID / 资产齐全版本)
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi0_fast_sonata",
        model=pi0_fast_sonata.Pi0FASTSonataConfig(   # ← 换成新 Config
            action_dim=8,
            action_horizon=10,
            point_feat_dim=6,
            projector_type=pi0_fast.ProjectorType.LINEAR,
            # 必填：用于原位插入点 token 的 special ids
            point_start_id=POINT_START_ID,
            point_end_id=POINT_END_ID,
        ),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[
                    # 在生产配置里也做严格点云校验（早失败）
                    ValidatePointCloud(
                        key="pointcloud",
                        feat_dim=getattr(model, "point_feat_dim", 6),
                        min_points=1,
                        allow_mask_all_false=False,
                    ),
                    droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST),
                ],
                outputs=[droid_policy.DroidOutputs()],
            ),
            # 生产配置：严格要求已有窗口标记（缺则报错，不自动补齐）
            model_transforms=lambda model: (  # 复用同一个 tokenizer 实例
                (lambda _tk=_tokenizer.FASTTokenizer(model.max_token_len): _transforms.Group(
                    inputs=[
                        RequirePointWindow(),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tk),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tk,
                            action_horizon=model.action_horizon,
                            action_dim=model.action_dim,
                        )
                    ],
                ))()
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        # 可选: 如有预训练基模型，可在此指定权重加载路径以初始化模型参数
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://your_base_checkpoint/params"),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        name="pi0_fast_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # libero10 config, 用于在18G的libero10数据集上测试
    #
    TrainConfig(
        name="pi0_fast_sonata_libero10",
        model=pi0_fast_sonata.Pi0FASTSonataConfig(
            paligemma_variant="gemma_2b_lora",   # ← LoRA 变体（关键）
            action_dim=7,
            action_horizon=10,
            max_token_len=180,                   # 建议：收敛几乎不受影响，显存大幅降低
            point_feat_dim=6,                    # Sonata 严格 6 通道（xyz+rgb），此处必须为 6
            projector_type=pi0_fast.ProjectorType.LINEAR,
            point_backbone_type=pi0_fast.PointBackboneType.SONATA,
            # 供 tokenizer 在 Prompt 里识别点窗标记；和 dbg 一致
            point_start_id=POINT_START_ID,
            point_end_id=POINT_END_ID,

            # ★★★ 关键：开启 “all” 端到端训练模式（Sonata encoder + projector 可训练）
            # 若只想训 projector：改为 "projector"；若全冻结 Sonata：改 "frozen"
            sonata_train_mode="all",

            # （可选）严格要求 CUDA，可在需要时打开（当前保持默认 False 以兼容开发环境）
            # require_cuda=True,
            # （可选）如每帧点 token 很多且显存允许，可增大点 token 容量上限（与 enc_patch_size 无关）
            # point_token_cap=1024,
        ),

        data=SimpleDataConfig(
            # 用 HF id（最省事）。要用本地，就把 repo_id 则改成本地绝对路径。
            # repo_id="binhng/libero_10_lerobot_mask_depth",
            # assets=AssetsConfig(asset_id="libero_10_lerobot_mask_depth"),
            # https://huggingface.co/datasets/binhng/libero_10_lerobot_mask_depth

            # ----------------------------
            # 路径选一：服务器版本（两处要一起改）
            # repo_id="/mnt/libero_10_lerobot_mask_depth",
            # assets=AssetsConfig(assets_dir="/mnt", asset_id="libero_10_lerobot_mask_depth"),
            # ----------------------------

            # 当前（本地）版本：确保 repo_id 与 tasks_jsonl 前缀一致（关键）
            repo_id="/home/siyuanyue/Documents/openpi/src/dataset/libero_10_lerobot_mask_depth",
            assets=AssetsConfig(
                assets_dir="/home/siyuanyue/Documents/openpi/src/dataset",  # 指向数据集根
                asset_id="libero_10_lerobot_mask_depth",                    # 子目录名
            ),

            # ① repack（键名对齐）→ 把数据集键重命名成策略/模型侧的常用键
            base_config=DataConfig(
                # 该数据集没有文本列 `task`，不能依赖 prompt_from_task；改为 False
                # prompt 由下游的 InjectPromptFromTaskIndex 从 meta/tasks.jsonl 注入
                prompt_from_task=False,
                # 这里必须写“数据集真实列名”，HF 列是 'action' 而不是 'actions'！
                # 读取后会在 repack 阶段把 'action' 重命名为 'actions'，供下游 transforms / 模型使用
                action_sequence_keys=("action",),
                repack_transforms=_transforms.Group(inputs=[
                    _transforms.RepackTransform({
                        # 目标键（左）= LiberoInputs 期望；源键（右）= 数据集真实列名（注意：该数据集使用点号扁平键）
                        "observation/image":       "observation.images.image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state":       "observation.state",

                        # 动作：把数据集的 'action' 规范化为 'actions'
                        "actions":                  "action",

                        # 深度：供 DecodeLiberoDepth 使用（数据集真实列在 observation.images.* 下）
                        "depth/front/raw":          "observation.images.image_depth",
                        "depth/wrist/raw":          "observation.images.wrist_depth",

                        # 透传任务索引，供 InjectPromptFromTaskIndex 使用
                        "task_index":               "task_index",
                    }),
                ]),
            ),

            # ② 深度解码 → 点云 → 校验 → Libero 输入/输出（顺序不要改）
            data_transforms=lambda model: (
                _transforms.Group(
                    inputs=[
                        # 先根据 task_index 注入文本指令（从 meta/tasks.jsonl 读取）
                        _transforms.InjectPromptFromTaskIndex(
                            # 路径选一：服务器版本
                            # tasks_jsonl="/mnt/libero_10_lerobot_mask_depth/meta/tasks.jsonl",
                            # 本地版本（与 repo_id 前缀保持一致 —— 关键）
                            tasks_jsonl="/home/siyuanyue/Documents/openpi/src/dataset/libero_10_lerobot_mask_depth/meta/tasks.jsonl",
                            #===============================================================================
                            task_index_key="task_index",
                            dst_key="prompt",
                            strict=True,
                        ),
                        _transforms.DecodeLiberoDepth(
                            src_keys=["depth/front/raw", "depth/wrist/raw"],
                            dst_keys=["depth/front/decoded", "depth/wrist/decoded"],
                            scale=None,   # 先相对尺度；若深度单位为毫米，可设 0.001 得到米制
                        ),
                        _transforms.DepthToPointCloud(
                            depth_map={"front": "depth/front/decoded", "wrist": "depth/wrist/decoded"},
                            rgb_map={"front": "observation/image", "wrist": "observation/wrist_image"},
                            intrinsics=None,  # 如有 (fx,fy,cx,cy) 可填 dict，得到米制点云
                            stride=4,
                            out_key="pointcloud",
                        ),

                        # —— 关键点 1：先把点云挂到“字典版 observation”，并校验 —— #
                        _transforms.AttachPointCloudToObservation(key="pointcloud"),
                        ValidatePointCloud(
                            key="pointcloud",
                            feat_dim=getattr(model, "point_feat_dim", 6),
                            min_points=1,
                        ),

                        # —— 关键点 2：构造“对象版 Observation”并保留/回挂点云 —— #
                        _transforms.LiberoInputsKeepExtras(
                            action_dim=model.action_dim,
                            model_type=ModelType.PI0_FAST,
                        ),

                        # 训练用 delta（前6维关节）
                        _transforms.DeltaActions(_transforms.make_bool_mask(6, -1)),
                    ],
                    outputs=[
                        libero_policy.LiberoOutputs(),
                        # 推理端把 delta → absolute（前6维关节）
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(6, -1)),

                        # 关键：显式“锚定”点云到结构里，避免批处理时被裁掉
                        _transforms.RepackTransform({
                            "observation/point_clouds/pointcloud": "observation/point_clouds/pointcloud",
                            "observation/point_cloud_masks/pointcloud": "observation/point_cloud_masks/pointcloud",
                        }),
                    ],
                )
            ),

            # ★ 模型 transforms：复用 FAST tokenizer，但在 tokenize 之前“保证点窗标记存在且恰好一对”
            #   - 若 tasks.jsonl 中 prompt 已含窗口标记，MapPrompt 不会重复添加；
            #   - RequirePointWindow 在训练/推理均严格校验，缺失或顺序错误会早失败。
            # 模型 transforms：只做 prompt/resize/tokenize，不再在这里回挂点云
            model_transforms=lambda model: (
                (lambda _tk=_tokenizer.FASTTokenizer(model.max_token_len): _transforms.Group(
                    inputs=[
                        # 若 prompt 未包含 <|point_start|><|point_end|>，在末尾追加一对（与 dbg 一致）
                        MapPrompt(lambda s: s if (s and (POINT_START_TOKEN in s and POINT_END_TOKEN in s))
                                else (f"{(s or '').strip()} {POINT_START_TOKEN}{POINT_END_TOKEN}").strip()),
                        RequirePointWindow(),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tk),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tk,
                            action_horizon=model.action_horizon,
                            action_dim=model.action_dim,
                        )
                    ],
                ))()
            ),
        ),

        # 加载 pi0‑FAST 基座权重：避免 PaliGemma 随机初始化（训练更稳、更快收敛）
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),

        # ★ 与“all”模式一致：只冻结 LoRA 以外的基座；同时 **解冻** Sonata encoder + projector
        #   注意：get_freeze_filter 需要读到 config.sonata_train_mode，故这里也显式传入 "all"
        freeze_filter=pi0_fast_sonata.Pi0FASTSonataConfig(
            paligemma_variant="gemma_2b_lora",
            point_feat_dim=6,
            sonata_train_mode="all",    # ← 与上面的 model 保持一致
        ).get_freeze_filter(),

        # LoRA 训练关闭 EMA（省显存、和 dbg 一致）
        ema_decay=None,

        log_interval=1,
        num_workers=0,
    ),
    #
    # agibotworld方案（最小改动、无任务表依赖版）
    #
    TrainConfig(
        name="pi0_fast_sonata_agibotworld",

        # —— 模型：沿用 Sonata + LoRA 方案（dim/horizon 按需改）——
        model=pi0_fast_sonata.Pi0FASTSonataConfig(
            paligemma_variant="gemma_2b_lora",
            action_dim=7,              # ★ 若你是 6 关节 + 1 指，留 7；否则改成你的动作维度
            action_horizon=10,         # ★ 每段动作长度，按你需要的 chunk 长度改
            max_token_len=180,
            point_feat_dim=6,          # 固定 6（xyz+rgb）
            projector_type=pi0_fast.ProjectorType.LINEAR,
            point_backbone_type=pi0_fast.PointBackboneType.SONATA,
            point_start_id=POINT_START_ID,
            point_end_id=POINT_END_ID,
            # 端到端训练 Sonata（建议）
            sonata_train_mode="all",
            # 没有点云就会报错（更早暴露问题）；若需暂时绕过，把它改成 False
            require_pointcloud=True,
        ),

            # ----------------------------
            # 路径选一：服务器版本（两处要一起改）
            # repo_id="/mnt/agiworldbot_task_390",
            # assets=AssetsConfig(assets_dir="/mnt", asset_id="agiworldbot_task_390"),
            # ----------------------------
            # 本地版本（默认启用）
            repo_id="/home/siyuanyue/Documents/openpi/src/dataset/agiworldbot_task_390",
            assets=AssetsConfig(
                assets_dir="/home/siyuanyue/Documents/openpi/src/dataset",
                asset_id="agiworldbot_task_390",
            ),

            # ① repack（把数据集真实键名映射成管线常用键）
            base_config=DataConfig(
                prompt_from_task=False,           # 不再依赖 tasks.jsonl；下方会用 MapPrompt 兜底
                action_sequence_keys=("action",), # 若你的列名是 'actions' 就写成 ("actions",)
                repack_transforms=_transforms.Group(inputs=[
                    _transforms.RepackTransform({
                        # ====== 按你的相机键名改（与数据集真实列对齐）======
                        # front 视角：用 top_head（480×640，与深度同分辨率）
                        "observation/image":       "observation.images.top_head",
                        # wrist 视角：用 hand_left（或 hand_right）
                        "observation/wrist_image": "observation.images.hand_left",

                        # 机器人状态（若无就删掉这行，同时确保下游 transform 不依赖它）
                        "observation/state":       "observation.state",

                        # 动作：把真实列名映射为 'actions'
                        "actions":                 "action",

                        # ====== 深度键（你的数据只有 cam_top_depth）======
                        # 把 cam_top_depth 当作“raw”，交给 DecodeLiberoDepth 去 squeeze/cast
                        "depth/front/raw":         "observation.images.cam_top_depth",

                        # ⚠️ 去掉 task_index 的强制映射，避免数据里没有该列时直接报错
                        # "task_index":              "task_index",
                    }),
                ]),
            ),

            # ② 数据变换：深度→点云→校验→组装 Libero 输入/输出（无任务表依赖）
            data_transforms=lambda model: (
                _transforms.Group(
                    inputs=[
                        # 先把 raw (1,H,W 或 uint16) 解码成 HxW float32
                        _transforms.DecodeLiberoDepth(
                            src_keys=["depth/front/raw"],
                            dst_keys=["depth/front/decoded"],
                            scale=None,  # ✅ 最稳妥：先不假设单位；确认是毫米后再改成 0.001
                        ),
                        # 用 front 深度 + front RGB 生成点云
                        _transforms.DepthToPointCloud(
                            depth_map={"front": "depth/front/decoded"},          # 仅 front
                            rgb_map={"front": "observation/image"},              # 与之配对的 RGB
                            intrinsics=None,  # ✅ 若你有 (fx,fy,cx,cy) 可填 dict 得到米制；暂无就先相对值
                            stride=4,
                            out_key="pointcloud",
                        ),

                        # 把点云“挂”到 Observation 新接口上，并做严格校验
                        _transforms.AttachPointCloudToObservation(key="pointcloud"),
                        ValidatePointCloud(
                            key="pointcloud",
                            feat_dim=getattr(model, "point_feat_dim", 6),
                            min_points=1,
                            allow_mask_all_false=False,
                        ),

                        # 组装为 Libero 风格输入（包含 images/state/prompt；并保持额外字段）
                        _transforms.LiberoInputsKeepExtras(
                            action_dim=model.action_dim,
                            model_type=ModelType.PI0_FAST,
                        ),

                        # 训练端使用 delta（如你的动作本来就是 delta，可移除）
                        _transforms.DeltaActions(_transforms.make_bool_mask(6, -1)),
                    ],
                    outputs=[
                        libero_policy.LiberoOutputs(),
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(6, -1)),
                        _transforms.RepackTransform({
                            # 显式锚定点云到 Observation 里，避免批处理时被裁掉
                            "observation/point_clouds/pointcloud": "observation/point_clouds/pointcloud",
                            "observation/point_cloud_masks/pointcloud": "observation/point_cloud_masks/pointcloud",
                        }),
                    ],
                )
            ),

            # ③ 模型变换：保证 prompt 含 <|point_start|><|point_end|>，再 Tokenize FAST
            model_transforms=lambda model: (
                (lambda _tk=_tokenizer.FASTTokenizer(model.max_token_len): _transforms.Group(
                    inputs=[
                        # 没有任务表：自动补一对点窗标记
                        MapPrompt(lambda s: s if (s and (POINT_START_TOKEN in s and POINT_END_TOKEN in s))
                                else (f"{(s or '').strip()} {POINT_START_TOKEN}{POINT_END_TOKEN}").strip() ),
                        RequirePointWindow(),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tk),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tk,
                            action_horizon=model.action_horizon,
                            action_dim=model.action_dim,
                        )
                    ],
                ))()
            ),
        ),

        # 载入 pi0‑FAST 基座，LoRA 训练建议关闭 EMA（与既有配置一致）
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        freeze_filter=pi0_fast_sonata.Pi0FASTSonataConfig(
            paligemma_variant="gemma_2b_lora",
            point_feat_dim=6,
            sonata_train_mode="all",
        ).get_freeze_filter(),
        ema_decay=None,

        # 其他超参按需改
        batch_size=2,
        log_interval=50,
        save_interval=5000,
        num_workers=0,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    # ------------------------------------------------------------------
    # Debug / smoke‑test: Pi0FAST + SONATA + Dummy 点云（LoRA 版）
    # 完全复用旧版 *_low_mem_finetune* 的写法
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi0_fast_sonata_dbg",
        exp_name="smoke_pc",
        # dbg 路线：显式使用 dummy_point 以避免 CLI 要求 --data.repo-id

        # ① LoRA‑variant 模型（行内写一次）
        model=pi0_fast_sonata.Pi0FASTSonataConfig(
            paligemma_variant="gemma_2b_lora",
            point_feat_dim=6,
            point_start_id=POINT_START_ID,
            point_end_id=POINT_END_ID,
            # dbg 也走严格模式：要求必须提供点云
            require_pointcloud=True,
        ),

        # dbg：严格校验点云，不再注入空点云
        data=SimpleDataConfig(
            repo_id="dummy_point",                  # ★ 关键：设定 repo_id，避免 CLI 要求 --data.repo-id
            # assets 仍可复用 dummy_point 以屏蔽 norm_stats 需求
            assets=AssetsConfig(asset_id="dummy_point"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[
                    ValidatePointCloud(
                        key="pointcloud",
                        feat_dim=getattr(model, "point_feat_dim", 6),
                        min_points=1,
                        allow_mask_all_false=False,  # 必须至少有一个有效点
                    ),
                ],
                outputs=[],
            ),
            model_transforms=lambda model: (
                (lambda _tk=_tokenizer.FASTTokenizer(model.max_token_len): _transforms.Group(
                    inputs=[
                        RequirePointWindow(),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tk),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tk,
                            action_horizon=model.action_horizon,
                            action_dim=model.action_dim,
                        )
                    ],
                ))()
            ),
            base_config=DataConfig(prompt_from_task=False),
        ),
        batch_size=2,
        num_train_steps=4,
        wandb_enabled=False,
        overwrite=True,

        # ② 加载 Pi0‑FAST 基座权重（与旧例同源）
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),

        # ③ 仅训练 LoRA + projector，其余权重冻结（行内再写一次）
        freeze_filter=pi0_fast_sonata.Pi0FASTSonataConfig(
            paligemma_variant="gemma_2b_lora",
            point_feat_dim=6,
        ).get_freeze_filter(),

        # ④ LoRA 训练关闭 EMA，保持一致
        ema_decay=None,
    )
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.models.tokenizer import POINT_START, POINT_END
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

import dataclasses

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])
    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(self, dataset: IterableDataset, transforms: Sequence[_transforms.DataTransformFn], *, is_batched: bool = False):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched
    def __iter__(self):
        for sample in self._dataset:
            if not self._is_batched:
                yield self._transform(sample)
                continue

            # 1) 只从“有batch维的 array 叶子”里推断 batch size
            bs = None
            for leaf in jax.tree_util.tree_leaves(sample):
                if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1:
                    bs = int(leaf.shape[0]); break
            if bs is None:
                raise ValueError("IterableTransformedDataset: cannot infer batch size from sample.")

            # 2) 拆成单样本：array 取第0维；list/tuple 长度为B则取第i个；其余（标量/字符串等）视为样本无关
            def _take_i(x, i):
                if hasattr(x, "shape") and getattr(x, "ndim", 0) >= 1 and x.shape[0] > i:
                    return x[i]
                if isinstance(x, (list, tuple)) and len(x) > i:
                    return x[i]
                return x
            singles = [
                # 只把 str/bytes 视作叶子，允许对 dict/list/tuple 递归
                jax.tree_map(lambda x, i=i: _take_i(x, i), sample,
                             is_leaf=lambda z: isinstance(z, (str, bytes)))
                for i in range(bs)
            ]

            # 3) 逐样本做 transform
            singles_t = [self._transform(s) for s in singles]

            # 4) 合回批：形状一致→stack；若仅轴0变长→pad后stack（仅数值/布尔）；否则→list
            def _stack_or_pad(*xs):
                arrays = []
                # 能转 ndarray 就优先按数组规则处理；否则退化为 list
                try:
                    for x in xs:
                        arrays.append(np.asarray(x))
                except Exception:
                    return list(xs)
                kinds = [a.dtype.kind for a in arrays]
                if not all(k in "biufc" for k in kinds):
                    return list(xs)
                shapes = [a.shape for a in arrays]
                if all(s == shapes[0] for s in shapes):
                    return np.stack(arrays, axis=0)
                ndims = {a.ndim for a in arrays}
                if len(ndims) == 1 and next(iter(ndims)) >= 1:
                    trailing = [a.shape[1:] for a in arrays]
                    if all(t == trailing[0] for t in trailing):
                        maxn = max(a.shape[0] for a in arrays)
                        out_shape = (len(arrays), maxn) + trailing[0]
                        out = np.zeros(out_shape, dtype=np.result_type(*[a.dtype for a in arrays]))
                        for i, a in enumerate(arrays):
                            out[i, : a.shape[0], ...] = a
                        return out
                # 非数组或不兼容 → 若值全等保留一个，否则 list
                try:
                    all_eq = all((x == xs[0]) for x in xs)
                except Exception:
                    all_eq = False
                return xs[0] if all_eq else list(xs)

            # 保持递归到 dict/list/tuple；只把 str/bytes 当叶子
            yield jax.tree_map(
                _stack_or_pad, *singles_t,
                is_leaf=lambda z: isinstance(z, (str, bytes))
            )
    def __len__(self) -> int:
        return len(self._dataset)

class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()
    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())
        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)
        observation = jax.tree_map(make_from_spec, self._observation_spec)
        action = jax.tree_map(make_from_spec, self._action_spec)
        return {
            **observation.to_dict(),
            "actions": action,
        }
    def __len__(self) -> int:
        return self._num_samples

class DummyPointDataset(Dataset):
    """
    A dummy dataset that generates synthetic observations with dummy point cloud data and random actions.
    Used for debugging the Pi0FASTSonata model's point cloud processing.
    """
    def __init__(self, model_config: _model.BaseModelConfig, length: int = 2):
        self.model_config = model_config
        self.length = length  # number of samples to generate
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, idx: SupportsIndex) -> dict:
        # ---------- 生成单样本，不带 batch 维 ----------
        obs_batched = self.model_config.fake_obs(batch_size=1)
        act_batched = self.model_config.fake_act(batch_size=1)
        obs = jax.tree_map(lambda x: x[0], obs_batched)
        act = jax.tree_map(lambda x: x[0], act_batched)

        # ---------- 生成符合“新接口”的点云 ----------
        # 形状: [N, 3 + feat_dim]；惯例: [:,0:3]=grid_coord(占位可用0), [:,3:6]=xyz, [:,6:]=extras
        num_points = 100
        feat_dim = int(getattr(self.model_config, "point_feat_dim", 6))
        coords_xyz = np.random.rand(num_points, 3).astype(np.float32) * 5.0  # xyz in [0,5)
        grid_coord = np.zeros((num_points, 3), dtype=np.float32)             # 占位网格坐标
        # extras 维度 = feat_dim - 3（因为 3 维被 xyz 占用）
        num_extras = max(feat_dim - 3, 0)
        extras = np.zeros((num_points, num_extras), dtype=np.float32)
        if num_extras > 0:
            # 用 RGB 作为 extras（如恰好 3 维时）
            rand_rgb = np.random.rand(num_points, 3).astype(np.float32)
            extras[:, :min(3, num_extras)] = rand_rgb[:, :min(3, num_extras)]
        # 拼接成 [N, 3 + feat_dim]
        pc_feat = np.concatenate([coords_xyz, extras], axis=1).astype(np.float32)  # [N, feat_dim]
        pointcloud = np.concatenate([grid_coord, pc_feat], axis=1)                 # [N, 3 + feat_dim]

        # ---------- prompt：恰好一对窗口标记 ----------
        prompt = f"debug window {POINT_START} {POINT_END}"

        # ---------- 输出 dict（顶层即新接口） ----------
        sample = {**obs.to_dict(), "actions": act}
        sample["prompt"] = prompt
        sample["point_clouds"] = {"pointcloud": pointcloud}
        sample["point_cloud_masks"] = {"pointcloud": np.bool_(True)}
        # 不再下发 legacy 的 pointcloud_data，保持新接口单一来源
        sample.pop("pointcloud_data", None)
        return sample

def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset object according to data_config. Supports custom identifiers for special datasets."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("DataConfig.repo_id is not set; cannot create dataset.")
    if repo_id == "fake":
        # Purely random data for debugging (no point cloud)
        return FakeDataset(model_config, num_samples=1024)
    if repo_id == "dummy_point":
        # Dummy dataset with random point cloud data for testing point cloud pipeline
        return DummyPointDataset(model_config, length=2)
    # Otherwise, treat repo_id as a LeRobot dataset identifier (load real dataset)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    # If prompts should be generated from task info, apply that transform
    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    return dataset

def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # Currently only support DROID RLDS datasets
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
    )

def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    """Apply data transforms (repacking, normalization, etc.) to a static dataset."""
    norm_stats = {}
    # Skip normalization stats for fake or dummy datasets
    if data_config.repo_id not in ("fake", "dummy_point") and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run `scripts/compute_norm_stats.py --config-name=<your-config>` to generate them."
            )
        norm_stats = data_config.norm_stats
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )

def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Apply data transforms to an iterable dataset (e.g., RLDS)."""
    norm_stats = {}
    if data_config.repo_id not in ("fake", "dummy_point") and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )

def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training based on the provided configuration."""
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )

def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a PyTorch-like data loader for a static dataset (non-RLDS)."""
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    # If using fake or dummy dataset, force skip_norm_stats to True (no normalization needed)
    skip_norm_stats = skip_norm_stats or data_config.repo_id in ("fake", "dummy_point")
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )
    return DataLoaderImpl(data_config, data_loader)

def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for RLDS (streaming) datasets."""
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)
    data_loader = RLDSDataLoader(dataset, sharding=sharding, num_batches=num_batches)
    return DataLoaderImpl(data_config, data_loader)

class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a data loader backed by torch.utils.data.DataLoader for random access datasets."""
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than dataset size ({len(dataset)}).")
        if sharding is None:
            # Default to data-parallel sharding across available devices
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding
        self._num_batches = num_batches
        # 保证本地 batch 能均分到本地设备
        ndev = max(1, jax.local_device_count())
        if local_batch_size % ndev != 0:
            raise ValueError(
                f"local_batch_size ({local_batch_size}) must be divisible by number of local devices ({ndev})."
            )

        mp_ctx = None
        if num_workers > 0:
            mp_ctx = multiprocessing.get_context("spawn")
        generator = torch.Generator()
        generator.manual_seed(seed)
        # Use PyTorch DataLoader for actual data loading
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_ctx,
            persistent_workers=(num_workers > 0),
            collate_fn=self._collate_fn,
            worker_init_fn=self._worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @staticmethod
    def _collate_fn(items):
        """Stack when shapes match; if only axis-0 is ragged, pad-to-max then stack (numeric/bool only);
        otherwise collect as list (e.g., strings)."""
        def _stack_or_pad(*xs):
            # 1) 全部能转为 ndarray 吗？
            arrays = []
            try:
                for x in xs:
                    arrays.append(np.asarray(x))
            except Exception:
                # 非数组（如 str/None/对象）→ 列表返回，避免报错
                return list(xs)

            # 1.5) 仅对数值/布尔数组做 pad/stack；字符串、对象等直接 list
            kinds = [a.dtype.kind for a in arrays]
            if not all(k in "biufc" for k in kinds):  # b:bool i:int u:uint f:float c:complex
                return list(xs)

            # 2) 形状完全一致 → 直接 stack
            shapes = [a.shape for a in arrays]
            if all(s == shapes[0] for s in shapes):
                return np.stack(arrays, axis=0)

            # 3) 若仅第 0 维不同，且“后续维度完全一致” → 沿轴 0 零填充到本批最大长度
            ndims = {a.ndim for a in arrays}
            if len(ndims) == 1 and next(iter(ndims)) >= 1:
                trailing = [a.shape[1:] for a in arrays]
                if all(t == trailing[0] for t in trailing):
                    maxn = max(a.shape[0] for a in arrays)
                    out_shape = (len(arrays), maxn) + trailing[0]
                    out = np.zeros(out_shape, dtype=np.result_type(*[a.dtype for a in arrays]))
                    for i, a in enumerate(arrays):
                        out[i, : a.shape[0], ...] = a
                    return out

            # 4) 其余情况 → 列表返回（如维度本身不同或 trailing 维不同）
            return list(xs)

        return jax.tree_map(_stack_or_pad, *items)

    @staticmethod
    def _worker_init_fn(worker_id: int):
        """Worker init: avoid preallocating GPU memory in forked processes."""
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def __iter__(self):
        num_yielded = 0
        # Loop indefinitely or until num_batches is reached
        while True:
            for batch in self._data_loader:
                if self._num_batches is not None and num_yielded >= self._num_batches:
                    return
                num_yielded += 1
                # 只对 array 叶子做 device_put 到指定 sharding；其他类型原样返回
                def _to_sharded(x):
                    if hasattr(x, "shape"):
                        try:
                            dt = np.asarray(x).dtype
                            if dt.kind in "biufc":
                                return jax.device_put(x, self._sharding)
                        except Exception:
                            pass
                    return x
                yield jax.tree_map(_to_sharded, batch)

class RLDSDataLoader:
    """Wrapper for an iterable RLDS dataset to provide the same interface as TorchDataLoader."""
    def __init__(
        self,
        dataset: IterableDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._dataset = dataset
        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_yielded = 0
        ndev = max(1, jax.local_device_count())
        while True:
            for batch in self._dataset:
                if self._num_batches is not None and num_yielded >= self._num_batches:
                    return
                num_yielded += 1
                # 一次性检查：能找到 array 叶子就推断 B，并校验能均分到本地设备
                B = None
                for leaf in jax.tree_util.tree_leaves(batch):
                    if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1:
                        B = int(leaf.shape[0]); break
                if B is not None and B % ndev != 0:
                    raise ValueError(
                        f"RLDS batch size ({B}) must be divisible by number of local devices ({ndev})."
                    )
                # 与 TorchDataLoader 保持一致：仅对 array 叶子 device_put 到给定 sharding
                def _to_sharded(x):
                    if hasattr(x, "shape"):
                        try:
                            dt = np.asarray(x).dtype
                            if dt.kind in "biufc":
                                return jax.device_put(x, self._sharding)
                        except Exception:
                            pass
                    return x
                yield jax.tree_map(_to_sharded, batch)

class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
    def data_config(self) -> _config.DataConfig:
        return self._data_config
    def __iter__(self):
        for batch in self._data_loader:
            # Convert batch dict to Observation object and separate actions
            yield _model.Observation.from_dict(batch), batch["actions"]
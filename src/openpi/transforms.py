from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


 # === Sonata / Point Cloud transforms (ported from intergrate_test_backup) ===
 # 这些是简单的可调用 dataclass；隐式满足 DataTransformFn 协议。
 
 # --- DecodeLiberoDepth ---
 from dataclasses import dataclass
 from typing import List, Optional, Tuple, Any
 import numpy as np
 
 def _to_numpy(x: Any) -> np.ndarray:
     try:
         return np.asarray(x)
     except Exception:
         try:
             import torch
             if isinstance(x, torch.Tensor):
                 return x.detach().cpu().numpy()
         except Exception:
             pass
         return np.array(x)
 
 def _as_hwc(x: np.ndarray) -> np.ndarray:
     # 支持 HWC / CHW -> HWC
     if x.ndim == 3 and x.shape[0] in (1,3):
         return np.transpose(x, (1,2,0))
     return x
 
 def _float_to_u8(a: np.ndarray) -> np.ndarray:
     # 将 [0,1] 或 [0,255] 的浮点域近似映射回 uint8（避免因浮点误差误判“通道全等”）
     a = np.asarray(a, dtype=np.float32)
     if a.max() <= 1.0:
         a = a * 255.0
     return np.clip(np.round(a), 0, 255).astype(np.uint8)
 
 @dataclass(frozen=True)
 class DecodeLiberoDepth:
     """
     支持三种输入：
       1) 单通道 uint16/float (H,W,1) / (1,H,W) / (H,W)
       2) 三通道“灰度复制”（C==3 && 三通道近似全等）
       3) 三通道 24-bit 打包 (R,G,B) -> (R*65536 + G*256 + B)
     输出 float32 深度（可选缩放与裁剪），写入 dst_keys。
     """
     src_keys: List[str]
     dst_keys: List[str]
     scale: Optional[float] = None
     clip_min: float = 0.0
     clip_max: Optional[float] = None
     identical_eps: float = 1e-5
     def __call__(self, sample: dict) -> dict:
         assert len(self.src_keys) == len(self.dst_keys)
         for s, d in zip(self.src_keys, self.dst_keys):
             arr = _as_hwc(_to_numpy(sample[s]))
             if arr.ndim == 2:
                 depth = arr.astype(np.float32)
             elif arr.ndim == 3 and arr.shape[-1] == 1:
                 depth = arr[..., 0].astype(np.float32)
             elif arr.ndim == 3 and arr.shape[-1] == 3:
                 ch0, ch1, ch2 = arr[..., 0], arr[..., 1], arr[..., 2]
                 if np.allclose(ch0, ch1, atol=self.identical_eps) and np.allclose(ch0, ch2, atol=self.identical_eps):
                     depth = ch0.astype(np.float32)
                 else:
                     a_u8 = _float_to_u8(arr)
                     r, g, b = a_u8[..., 0].astype(np.uint32), a_u8[..., 1].astype(np.uint32), a_u8[..., 2].astype(np.uint32)
                     depth = (r << 16) + (g << 8) + b
                     depth = depth.astype(np.float32)
             else:
                 raise TypeError(f"Unsupported depth array shape: {arr.shape}")
             if self.scale is not None:
                 depth = depth * float(self.scale)
             depth = np.clip(depth, self.clip_min, self.clip_max) if self.clip_max is not None else np.clip(depth, self.clip_min, None)
             sample[d] = depth.astype(np.float32)
         return sample
 
 # --- DepthToPointCloud ---
 @dataclass(frozen=True)
 class DepthToPointCloud:
     """
     深度 -> 点云（可带相机内参，输出 [grid(3)|xyz(3)|extras(C)] 共 3+C 列；grid 填 0 以满足 PI0 契约）
     - depth_map: dict 名到深度图键
     - color_map: dict 名到 RGB 图键（可选，缺省则 extras 置零）
     - intrinsics: {name: (fx,fy,cx,cy)} 或 None（None 走相对尺度）
     - stride: 下采样步长
     - out_key: 写入 point_clouds[out_key] / point_cloud_masks[out_key]
     """
     depth_map: dict
     color_map: Optional[dict] = None
     intrinsics: Optional[dict] = None
     stride: int = 4
     out_key: str = "pointcloud"
     max_points: int = 32768
     def __call__(self, sample: dict) -> dict:
         pts_all, rgb_all = [], []
         for name, dkey in self.depth_map.items():
             depth = _to_numpy(sample[dkey])
             if self.stride > 1:
                 depth = depth[:: self.stride, :: self.stride]
             H, W = depth.shape[:2]
             yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
             Z = depth.astype(np.float32)
             if self.intrinsics and name in self.intrinsics:
                 fx, fy, cx, cy = self.intrinsics[name]
                 X = (xx - cx) * Z / fx
                 Y = (yy - cy) * Z / fy
             else:
                 X = (xx - (W - 1) / 2.0).astype(np.float32)
                 Y = (yy - (H - 1) / 2.0).astype(np.float32)
             P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
             if self.color_map and name in self.color_map:
                 img = _as_hwc(_to_numpy(sample[self.color_map[name]])).astype(np.float32)
                 if img.max() > 1.5: img = img / 255.0
                 C = img.reshape(-1, 3)
             else:
                 C = np.zeros((P.shape[0], 3), dtype=np.float32)
             pts_all.append(P); rgb_all.append(C)
         if not pts_all:
             return sample
         P = np.concatenate(pts_all, axis=0)
         C = np.concatenate(rgb_all, axis=0)
         if P.shape[0] > self.max_points:
             idx = np.random.choice(P.shape[0], self.max_points, replace=False)
             P, C = P[idx], C[idx]
         grid = np.zeros_like(P, dtype=np.float32)
         pc = np.concatenate([grid, P.astype(np.float32), C.astype(np.float32)], axis=1)
         sample.setdefault("point_clouds", {})[self.out_key] = pc
         sample.setdefault("point_cloud_masks", {})[self.out_key] = np.bool_(True)
         return sample
 
 # --- ValidatePointCloud (strict fail-fast) ---
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
         pcs = pms = None
         if "point_clouds" in batch:
             pcs = batch["point_clouds"]; pms = batch.get("point_cloud_masks", None)
         elif "observation" in batch and isinstance(batch["observation"], dict):
             pcs = batch["observation"].get("point_clouds", None)
             pms = batch["observation"].get("point_cloud_masks", None)
         if pcs is None or self.key not in pcs:
             raise KeyError(f"missing point_clouds['{self.key}']")
         x = np.asarray(pcs[self.key])
         if x.ndim == 3 and x.shape[0] == 1: x = x[0]
         if x.ndim != 2 or x.shape[1] != 3 + self.feat_dim:
             raise ValueError(f"pointcloud shape must be [N,{3 + self.feat_dim}], got {tuple(x.shape)}")
         if x.dtype != np.float32:
             raise TypeError(f"pointcloud dtype must be float32, got {x.dtype}")
         if not np.isfinite(x).all():
             raise ValueError("pointcloud contains NaN/Inf")
         if x.shape[0] < self.min_points:
             raise ValueError(f"pointcloud must have at least {self.min_points} points, got {x.shape[0]}")
         if pms is None or self.key not in pms:
             raise KeyError(f"missing point_cloud_masks['{self.key}']")
         m = np.asarray(pms[self.key])
         if m.shape == () or m.shape == (1,):
             valid = bool(m.reshape(-1)[0])
         elif m.ndim == 2 and m.shape[0] == 1:
             valid = bool(m[0])
         else:
             raise ValueError(f"mask must be a frame-level bool (scalar or [1]), got shape {tuple(m.shape)}")
         if not self.allow_mask_all_false and not valid:
             raise ValueError("frame-level mask is False")
         return batch
 
 # --- LiberoInputsKeepExtras (preserve extras through LiberoInputs) ---
 @dataclasses.dataclass(frozen=True)
 class LiberoInputsKeepExtras:
     """
     包一层 `openpi.policies.libero_policy.LiberoInputs`，并保留/回附点云字段，
     防止它被下游丢弃。仅在你用 Libero 的默认输入转换且需要点云时使用。
     """
     action_dim: int
     model_type: "_model.ModelType"
     def __call__(self, batch: dict) -> dict:
         pcs = batch.get("point_clouds", None)
         pms = batch.get("point_cloud_masks", None)
         from openpi.policies.libero_policy import LiberoInputs  # 延迟导入，避免循环
         wrapped = LiberoInputs(model_type=self.model_type)
         out = wrapped(batch)
         if isinstance(out, dict):
             if isinstance(pcs, dict) and "pointcloud" in pcs:
                 out.setdefault("point_clouds", {})["pointcloud"] = pcs["pointcloud"]
             if isinstance(pms, dict) and "pointcloud" in pms:
                 out.setdefault("point_cloud_masks", {})["pointcloud"] = pms["pointcloud"]
         return out
import abc
from collections.abc import Sequence
import dataclasses
from dataclasses import field
import enum
import logging
import pathlib
from typing import Any, Generic, TypeAlias, TypeVar

# ---- Optional dependencies (augmax, Flax, JAX, Orbax) with safe fallbacks ---
class _MissingModule:
    def __init__(self, name: str):
        self.__name = name
    def __getattr__(self, _):
        raise ImportError(
            f"Optional dependency '{self.__name}' is required for this code path but is not installed."
        )

# augmax is only used in JAX preprocessing path
try:
    import augmax  # noqa: F401
except Exception:
    augmax = _MissingModule("augmax")  # type: ignore

# Flax (nnx / struct / traverse_util)
try:
    from flax import nnx  # noqa: F401
except Exception:
    class _NNXModule:  # minimal placeholder for class base
        pass
    def _nnx_fail(*_args, **_kwargs):
        raise ImportError("flax.nnx is required for this code path.")
    class _NNXShim:
        Module = _NNXModule
        eval_shape = staticmethod(_nnx_fail)
        split = staticmethod(_nnx_fail)
        merge = staticmethod(_nnx_fail)
    nnx = _NNXShim()  # type: ignore

try:
    from flax import struct as _flax_struct  # noqa: F401
    struct = _flax_struct
except Exception:
    # fall back to stdlib dataclass so class definitions still succeed
    class _StructShim:
        dataclass = dataclasses.dataclass
    struct = _StructShim()  # type: ignore

try:
    from flax import traverse_util  # noqa: F401
except Exception:
    traverse_util = _MissingModule("flax.traverse_util")  # type: ignore

# JAX (optional): provide jnp fallback and minimal placeholders
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except Exception:
    import numpy as jnp  # type: ignore
    _HAS_JAX = False
    class _JaxArray:  # placeholder for typing
        pass
    class _JaxShapeDtypeStruct:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
    class _JaxSharding:
        class Sharding: ...
        class Mesh:
            def __init__(self, *args, **kwargs): ...
        class NamedSharding:
            def __init__(self, *args, **kwargs): ...
        class PartitionSpec:
            def __init__(self, *args, **kwargs): ...
    class _JaxTyping:
        ArrayLike = object
    class _JaxTree:
        @staticmethod
        def map(*_a, **_k):  # raised on use
            raise ImportError("JAX is required for this code path.")
    class _JaxRandom:
        @staticmethod
        def split(*_a, **_k):
            raise ImportError("JAX is required for this code path.")
        @staticmethod
        def key(*_a, **_k):
            raise ImportError("JAX is required for this code path.")
    class _JaxStub:
        Array = _JaxArray
        ShapeDtypeStruct = _JaxShapeDtypeStruct
        sharding = _JaxSharding
        typing = _JaxTyping
        tree = _JaxTree
        random = _JaxRandom
        @staticmethod
        def vmap(*_a, **_k):
            raise ImportError("JAX is required for this code path.")
        @staticmethod
        def devices():
            raise ImportError("JAX is required for this code path.")
        @staticmethod
        def default_device(_dev):
            raise ImportError("JAX is required for this code path.")
    jax = _JaxStub()  # type: ignore


import numpy as np
try:
    import orbax.checkpoint as ocp  # noqa: F401
except Exception:
    ocp = _MissingModule("orbax.checkpoint")  # type: ignore
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# 注意：Pylance 不允许在 TypeVar 约束中使用“变量”。为兼容 Pylance，这里改为不带约束的 TypeVar。
# 运行时精确性由 jaxtyping 的运行时检查（at.Float/at.Bool 等）保证；不影响实际行为。
ArrayT = TypeVar("ArrayT")

class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Legacy point cloud (兼容旧管线)
    pointcloud_data: dict[str, ArrayT] | None = None
    # New point cloud (SpatialLM 对齐，帧级 mask)
    point_clouds: dict[str, ArrayT] = field(default_factory=dict)
    point_cloud_masks: dict[str, ArrayT] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """Preserve backends: Torch stays Torch; NumPy -> JAX; JAX stays JAX."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # images
        imgs = {}
        for k, img in data["image"].items():
            if isinstance(img, torch.Tensor):
                if img.dtype == torch.uint8:
                    img = img.to(torch.float32) / 255.0 * 2.0 - 1.0
                # 统一 Observation 为 BHWC；若输入是 NCHW（B,C,H,W），则转为 NHWC（B,H,W,C）
                if img.ndim == 4 and (img.shape[1] in (1, 3, 4)) and (img.shape[-1] not in (1, 3, 4)):
                    img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
                imgs[k] = img
            else:
                # 同时处理 np.uint8 与 jnp.uint8
                if hasattr(img, "dtype") and img.dtype in (np.uint8, jnp.uint8):
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                imgs[k] = jnp.asarray(img)
        data["image"] = imgs
        # image_mask -> bool (preserve backend)
        def _to_bool_mask(x):
            if isinstance(x, torch.Tensor):
                return x.to(torch.bool)
            x = jnp.asarray(x); return x if x.dtype == jnp.bool_ else (x != 0)
        data["image_mask"] = {k: _to_bool_mask(v) for k,v in data["image_mask"].items()}
        # state & token_* (preserve backend)
        def _keep(x): return x if isinstance(x, torch.Tensor) else jnp.asarray(x)
        data["state"] = _keep(data["state"])
        for fld in ("tokenized_prompt","tokenized_prompt_mask","token_ar_mask","token_loss_mask"):
            if fld in data and data[fld] is not None:
                data[fld] = _keep(data[fld])
        # new point clouds (preserve backend; mask -> bool; auto-synthesize)
        new_pc, new_pm = {}, {}
        use_new = ("point_clouds" in data) and (data["point_clouds"] is not None)
        if use_new:
            if not isinstance(data["point_clouds"], dict):
                raise TypeError("point_clouds must be a dict[str, array].")
            new_pc = {k: (v if isinstance(v, torch.Tensor) else jnp.asarray(v))
                      for k, v in data["point_clouds"].items()}
            if "pointcloud" in new_pc:
                arr = new_pc["pointcloud"]
                if getattr(arr, "ndim", None) != 3:
                    raise ValueError(f"point_clouds['pointcloud'] must be 3D [B,M,3+C], got {getattr(arr,'shape',None)}.")
                if arr.shape[-1] < 6:
                    logger.warning("pointcloud last dim %d (<6); expect [3 grid | 3 xyz | extras].", arr.shape[-1])
        if "point_cloud_masks" in data and data["point_cloud_masks"] is not None:
            if not isinstance(data["point_cloud_masks"], dict):
                raise TypeError("point_cloud_masks must be a dict[str, array].")
            new_pm = {k: (v.to(torch.bool) if isinstance(v, torch.Tensor) else jnp.asarray(v, dtype=jnp.bool_))
                      for k,v in data["point_cloud_masks"].items()}
        if new_pc and not new_pm:
            for k,v in new_pc.items():
                new_pm[k] = (torch.ones((v.shape[0],), dtype=torch.bool, device=v.device)
                             if isinstance(v, torch.Tensor) else jnp.ones((v.shape[0],), dtype=jnp.bool_))
        elif new_pc and new_pm:
            for k,v in new_pc.items():
                if k not in new_pm:
                    new_pm[k] = (torch.ones((v.shape[0],), dtype=torch.bool, device=v.device)
                                 if isinstance(v, torch.Tensor) else jnp.ones((v.shape[0],), dtype=jnp.bool_))
        # Ensure every provided mask matches [B].
        if new_pc and new_pm:
            for k, v in new_pc.items():
                if k in new_pm:
                    pm = new_pm[k]
                    expected = (v.shape[0],)
                    if tuple(pm.shape) != expected:
                        raise ValueError(
                            f"point_cloud_masks['{k}'] must have shape {expected}, got {tuple(pm.shape)}."
                        )
        if new_pc and new_pm and set(new_pc.keys()) != set(new_pm.keys()):
            logger.warning("point_clouds keys %s != point_cloud_masks keys %s.", sorted(new_pc), sorted(new_pm))
        # legacy pointcloud_data
        pc = None
        if use_new and "pointcloud_data" in data and data["pointcloud_data"] is not None:
            logger.warning("Both 'point_clouds' (new) and 'pointcloud_data' (legacy) provided; using NEW.")
        if (not use_new) and "pointcloud_data" in data and data["pointcloud_data"] is not None:
            if not isinstance(data["pointcloud_data"], dict):
                raise TypeError("pointcloud_data must be a dict[str, array].")
            pc = {k: (v if isinstance(v, torch.Tensor) else jnp.asarray(v)) for k,v in data["pointcloud_data"].items()}
            required = {"coord","batch"}; missing = required - pc.keys()
            if missing: raise ValueError(f"pointcloud_data missing {sorted(missing)}; have {sorted(pc.keys())}")
            if pc["coord"].ndim < 2 or pc["coord"].shape[-1] != 3:
                raise ValueError(f"pointcloud_data['coord'] must be (...,N,3); got {pc['coord'].shape}")
            n = pc["coord"].shape[-2]
            if pc["batch"].shape[-1] != n: raise ValueError("batch length != coord N")
            if "feat" in pc and pc["feat"].shape[-2] != n: raise ValueError("feat N != coord N")
            if "grid_size" in pc and pc["grid_size"].shape[-1] != 3: raise ValueError("grid_size last dim != 3")
        return cls(
            images=data["image"], image_masks=data["image_mask"], state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            pointcloud_data=pc, point_clouds=new_pc if use_new else {}, point_cloud_masks=new_pm if use_new else {},
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        if result.get("pointcloud_data") is None: result.pop("pointcloud_data", None)
        if not result.get("point_clouds"): result.pop("point_clouds", None)
        if not result.get("point_cloud_masks"): result.pop("point_cloud_masks", None)
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions: TypeAlias = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    if train and rng is None:
        raise ValueError("rng must be provided when `train=True` for image augmentations.")
    # JAX-only guard: state must not be a torch.Tensor
    if isinstance(observation.state, torch.Tensor):
        raise TypeError(
            "preprocess_observation expects JAX/NumPy state. "
            "Got torch.Tensor; run state preprocessing in the PyTorch pipeline instead."
        )

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        # This function is JAX-only; Torch images should be preprocessed in the PyTorch pipeline.
        if isinstance(image, torch.Tensor):
            raise TypeError(
                "preprocess_observation expects JAX/NumPy images (BHWC). "
                "Got torch.Tensor; run image preprocessing in the PyTorch pipeline instead."
            )
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            # Lazy import; this path is JAX-only
            from openpi.shared import image_tools as _image_tools
            image = _image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
    # new point cloud pass-through
    out_point_clouds = dict(getattr(observation, "point_clouds", {}) or {})
    out_point_cloud_masks = dict(getattr(observation, "point_cloud_masks", {}) or {})
    # JAX-only guard: point clouds and masks must not be torch.Tensors
    for k, arr in out_point_clouds.items():
        if isinstance(arr, torch.Tensor):
            raise TypeError(
                f"preprocess_observation expects JAX/NumPy point clouds; got torch.Tensor for key '{k}'. "
                "Run point-cloud preprocessing in the PyTorch pipeline instead."
            )
    for k, m in out_point_cloud_masks.items():
        if isinstance(m, torch.Tensor):
            raise TypeError(
                f"preprocess_observation expects JAX/NumPy point cloud masks; got torch.Tensor for key '{k}'. "
                "Run point-cloud preprocessing in the PyTorch pipeline instead."
            )
    for k, arr in out_point_clouds.items():
        if k not in out_point_cloud_masks:
            out_point_cloud_masks[k] = jnp.ones((arr.shape[0],), dtype=jnp.bool_)

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        pointcloud_data=getattr(observation, "pointcloud_data", None),
        point_clouds=out_point_clouds,
        point_cloud_masks=out_point_cloud_masks,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    # Pylance 友好：避免在类型表达式中引用运行期变量（jax.Array / jnp.dtype / jax.sharding.Sharding）
    restore_type: type | None = None,
    dtype: np.dtype | None = None,
    sharding: Any | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    # 保持原逻辑：默认用 JAX Array；若无 JAX 则退回到 numpy.ndarray
    if restore_type is None:
        restore_type = jax.Array if _HAS_JAX else np.ndarray
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

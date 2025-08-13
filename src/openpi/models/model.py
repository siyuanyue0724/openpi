import abc
from collections.abc import Sequence
import dataclasses
from dataclasses import field
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"


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
#     # (New, optional) Point cloud for Sonata-style encoders:
#     "point_clouds": {
#         "pointcloud": float32[*b, m, 3 + c]   # [:,:,0:3]=grid_coord(int as float ok upstream); [:,:,3:6]=xyz; [:,:,6:]=extras
#     },
#     "point_cloud_masks": {
#         "pointcloud": bool[*b]                # True if this sample provides a point cloud
#     },
#     # (Legacy, optional) Point cloud dict kept for backward compat:
#     "pointcloud_data": {coord(N,3), feat(N,F), batch(N), offset(B), grid_coord(N,3), ...}
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

    # ----------------------------------------------------------------------
    # Legacy point cloud field (kept for backward-compatibility with older
    # pipelines that already build Sonata-compatible dicts).
    # Expected keys typically: 'coord','feat','batch','offset','grid_coord',...
    # ----------------------------------------------------------------------
    pointcloud_data: dict[str, ArrayT] | None = None

    # ----------------------------------------------------------------------
    # New point cloud interface (SpatialLM-aligned, frame-level masking).
    # point_clouds["pointcloud"]: [B, M, 3 + C] float32
    # point_cloud_masks["pointcloud"]: [B] bool
    # ----------------------------------------------------------------------
    point_clouds: dict[str, ArrayT] = field(default_factory=dict)
    point_cloud_masks: dict[str, ArrayT] = field(default_factory=dict)


    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # -----------------------------  images  -----------------------------
        # 如果是 uint8 → 先归一化；最后统一转 jax.Array
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                img = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            else:
                img = data["image"][key]
            data["image"][key] = jnp.asarray(img)

        # image_mask / state / prompt 等也全部转成 jax.Array，保持 dtype 不变
        data["image_mask"] = {
            k: jnp.asarray(v) for k, v in data["image_mask"].items()
        }
        data["state"] = jnp.asarray(data["state"])
        for fld in ("tokenized_prompt", "tokenized_prompt_mask",
                    "token_ar_mask", "token_loss_mask"):
            if fld in data and data[fld] is not None:
                data[fld] = jnp.asarray(data[fld])
    
        # --------------------------------------------------------------------
        # New point cloud API (SpatialLM-aligned): point_clouds + masks
        # --------------------------------------------------------------------
        new_pc: dict[str, ArrayT] = {}
        new_pm: dict[str, ArrayT] = {}
        use_new_api = ("point_clouds" in data) and (data["point_clouds"] is not None)
        if "point_clouds" in data and data["point_clouds"] is not None:
            if not isinstance(data["point_clouds"], dict):
                raise TypeError("point_clouds must be a dict[str, array].")
            # to jax.Array
            new_pc = {k: jnp.asarray(v) for k, v in data["point_clouds"].items()}
            # lightweight sanity checks for the common key "pointcloud"
            if "pointcloud" in new_pc:
                arr = new_pc["pointcloud"]
                if arr.ndim != 3:
                    raise ValueError(f"point_clouds['pointcloud'] must be 3D [B,M,3+C], got shape {arr.shape}.")
                if arr.shape[-1] < 6:
                    logger.warning(
                        "point_clouds['pointcloud'] last dim is %d (<6). Expected [3 grid | 3 xyz | extras].",
                        arr.shape[-1],
                    )
        if "point_cloud_masks" in data and data["point_cloud_masks"] is not None:
            if not isinstance(data["point_cloud_masks"], dict):
                raise TypeError("point_cloud_masks must be a dict[str, array].")
        # If new API is present but masks are missing, synthesize all-True masks per key.
        if new_pc and not new_pm:
            new_pm = {k: jnp.ones((v.shape[0],), dtype=jnp.bool_) for k, v in new_pc.items()}
        elif new_pc and new_pm:
            # Ensure every PC key has a mask; if missing, fill with True.
            for k, v in new_pc.items():
                if k not in new_pm:
                    new_pm[k] = jnp.ones((v.shape[0],), dtype=jnp.bool_)
        # If both dicts exist, ensure key sets match (soft check)
        if new_pc and new_pm:
            if set(new_pc.keys()) != set(new_pm.keys()):
                logger.warning(
                    "point_clouds keys %s != point_cloud_masks keys %s; proceeding but this is likely a bug.",
                    sorted(new_pc.keys()), sorted(new_pm.keys()),
                )

        # --------------------------------------------------------------------
        # point cloud sanity‑check  🚦
        # --------------------------------------------------------------------
        pc = None
        # If both new API and legacy are present, prefer the NEW API, warn once.
        if use_new_api and "pointcloud_data" in data and data["pointcloud_data"] is not None:
            logger.warning(
                "Both 'point_clouds' (new API) and 'pointcloud_data' (legacy) provided; "
                "using NEW API and ignoring legacy pointcloud_data."
            )
        # Legacy path only if new API not present.
        if not use_new_api and "pointcloud_data" in data and data["pointcloud_data"] is not None:
            pc = {k: jnp.asarray(v) for k, v in data["pointcloud_data"].items()}

            # 必须包含 coord(N×3) 与 batch(N)；其它键可选
            required = {"coord", "batch"}
            missing  = required - pc.keys()
            if missing:
                raise ValueError(
                    f"pointcloud_data 缺少必须字段 {sorted(missing)}；"
                    f"目前仅看到 {sorted(pc.keys())}"
                )

            # coord: (..., N, 3)
            if pc["coord"].ndim < 2 or pc["coord"].shape[-1] != 3:
                raise ValueError(
                    f"pointcloud_data['coord'] 形状须为 (..., N, 3)，实际 {pc['coord'].shape}"
                )
            n_pts = pc["coord"].shape[-2]

            # batch: (..., N)
            if pc["batch"].shape[-1] != n_pts:
                raise ValueError(
                    f"pointcloud_data['batch'] 长度 {pc['batch'].shape[-1]} "
                    f"与 coord 点数 {n_pts} 不一致"
                )

            # feat (可选): (..., N, F)
            if "feat" in pc and pc["feat"].shape[-2] != n_pts:
                raise ValueError(
                    f"pointcloud_data['feat'] 点数 {pc['feat'].shape[-2]} "
                    f"与 coord 点数 {n_pts} 不一致"
                )
            # grid_size (可选): (..., 3)
            if "grid_size" in pc and pc["grid_size"].shape[-1] != 3:
                raise ValueError(
                    f"pointcloud_data['grid_size'] 最后维须为 3，实际 {pc['grid_size'].shape}"
                )

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            pointcloud_data=pc,  # legacy field (may be None)
            # new API (default to empty dicts if not present)
            point_clouds=new_pc if use_new_api else {},
            point_cloud_masks=new_pm if use_new_api else {},
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        # 若无 legacy 点云则移除字段，避免写出 None
        if result.get("pointcloud_data") is None:
            result.pop("pointcloud_data", None)
        # 若新接口为空 dict，也移除，避免产生空对象
        if not result.get("point_clouds"):
            result.pop("point_clouds", None)
        if not result.get("point_cloud_masks"):
            result.pop("point_cloud_masks", None)
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


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

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            if rng is None:
                raise ValueError("rng must be provided when `train=True` for image augmentations.")
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

    # 透传新接口点云（原样，不做任何修复/体素化/偏移）
    out_point_clouds = dict(getattr(observation, "point_clouds", {}) or {})
    out_point_cloud_masks = dict(getattr(observation, "point_cloud_masks", {}) or {})
    # If upstream provided point_clouds but forgot masks, synthesize all-True masks.
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
        pointcloud_data=observation.pointcloud_data,  # legacy unchanged
        point_clouds=out_point_clouds,               # new API透传
        point_cloud_masks=out_point_cloud_masks,     # new API透传
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

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        def _fill(x):
            # booleans -> True; everything else -> 0
            if x.dtype == jnp.bool_:
                return jnp.ones(x.shape, x.dtype)
            return jnp.zeros(x.shape, x.dtype)
        return jax.tree.map(_fill, observation_spec)

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
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
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
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

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

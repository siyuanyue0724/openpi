# File: openpi/models/pi0_fast_sonata.py

import dataclasses
import jax
import jax.numpy as jnp
import torch
from flax import nnx
from openpi.shared import download         # 仅 download 在 shared 包
from openpi.shared import sonata_encoder
from openpi.shared import nnx_bridge       # 我们自建的桥接模块

from openpi.models import gemma as _gemma
from openpi.models import siglip as _siglip
import openpi.models.model as _model  # for BaseModel and Observation

# Use the Sonata class from the sonata_encoder module
Sonata = sonata_encoder.Sonata

@dataclasses.dataclass(frozen=True)
class Pi0FASTSonataConfig(_model.BaseModelConfig):
    """Config for Pi0FASTSonata model (Pi0FAST with Sonata point cloud encoder)."""
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Inherit default action_dim, action_horizon, max_token_len from Pi0FAST (32, 32, 250)

    @property
    def model_type(self) -> _model.ModelType:
        # Reuse PI0_FAST type or define a new one if needed
        return _model.ModelType.PI0_FAST

    def create(self, rng: jax.random.KeyArray) -> "Pi0FASTSonata":
        """Instantiate a Pi0FASTSonata model with random key `rng`."""
        return Pi0FASTSonata(self, rngs=nnx.Rngs(rng))

class Pi0FASTSonata(_model.BaseModel):
    """
    Pi0FASTSonata model: Extends Pi0FAST to incorporate a Sonata point cloud encoder.
    The Sonata encoder is loaded with pretrained weights and integrated as an 
    additional modality in the model (self.PaliGemma.point).
    """
    def __init__(self, config: Pi0FASTSonataConfig, rngs: nnx.Rngs):
        # Initialize base model (set up action_dim, horizon, etc.)
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # Configure and initialize LLM (Gemma) and image (SigLIP) encoders
        paligemma_cfg = _gemma.get_config(config.paligemma_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_cfg,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_cfg.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # Use a dummy image from fake_obs to initialize image encoder dimensions
        dummy_image = next(iter(config.fake_obs(batch_size=1).images.values()))
        img.lazy_init(dummy_image, train=False, rngs=rngs)
        # Prepare Sonata point cloud encoder with pretrained weights
        # 1. Download pretrained Sonata checkpoint (if not already cached)
        weights_url = "https://huggingface.co/facebook/sonata/resolve/main/pretrain-sonata-v1m1-0-base.pth"
        checkpoint_path = download.maybe_download(weights_url)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # 2. Load weights into Sonata model
        point_model = Sonata()  # uses default in_channels=6 (coords+feat)
        point_model.load_state_dict(state_dict)
        point_model.eval()  # set to inference mode
        # 3. Wrap the Sonata model as an NNX module for JAX integration
        point = nnx_bridge.ToNNX(point_model)
        # Create a minimal dummy point cloud input to initialize the Sonata module
        dummy_data = {
            "coord": jax.numpy.zeros((1, 3), dtype=jax.numpy.float32),
            "feat": jax.numpy.zeros((1, 6), dtype=jax.numpy.float32),
            "batch": jax.numpy.zeros((1,), dtype=jax.numpy.int32),
            "grid_size": jax.numpy.ones((1, 3), dtype=jax.numpy.int32),
        }
        point.lazy_init(dummy_data, train=False, rngs=rngs)
        # Combine LLM, image, and point encoders into one dict
        self.PaliGemma = nnx.Dict(llm=llm, img=img, point=point)

    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Embed all input modalities (images, point cloud, text tokens) into token sequences.
        Returns:
            token_embeddings (jax.Array) – concatenated token features [B, N, emb_dim]
            input_mask (jax.Array[bool]) – mask indicating valid tokens [B, N]
            ar_mask (jax.Array[int]) – autoregressive mask for tokens [B, N] (0 = no autoregressive dependency).
        """
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # 1. Image tokens (use image encoder)
        for cam_name in obs.images:
            img_tokens, _ = self.PaliGemma.img(obs.images[cam_name], train=False)
            token_embeddings.append(img_tokens)
            # Image mask: 1 for each token, same length as img_tokens per batch
            mask = jnp.broadcast_to(obs.image_masks[cam_name][:, None], (obs.image_masks[cam_name].shape[0], img_tokens.shape[1]))
            input_mask.append(mask)
            # Image tokens attend among themselves (non-AR) -> ar_mask = 0
            ar_mask.append(jnp.zeros_like(mask))
        # 2. Point cloud tokens (use Sonata encoder) – if dummy point data is present
        if hasattr(obs, "pointcloud_data"):
            # Run Sonata encoder on the dummy point cloud data to get context tokens
            point_tokens = self.PaliGemma.point(obs.pointcloud_data, train=False)
            token_embeddings.append(point_tokens)
            # Point tokens mask (all valid, non-padding)
            pt_mask = jnp.ones((point_tokens.shape[0], point_tokens.shape[1]), dtype=bool)
            input_mask.append(pt_mask)
            # Point tokens are treated like image tokens (no causal dependency among themselves)
            ar_mask.append(jnp.zeros_like(pt_mask))
        # 3. Text tokens (tokenized prompt via LLM embedding)
        assert obs.tokenized_prompt is not None and obs.tokenized_prompt_mask is not None and obs.token_ar_mask is not None, \
               "Tokenized prompt and masks are required."
        txt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(txt_tokens)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)
        # Concatenate all token sequences and masks along sequence dimension
        tokens = jnp.concatenate(token_embeddings, axis=1)
        mask = jnp.concatenate(input_mask, axis=1)
        ar = jnp.concatenate(ar_mask, axis=1)
        return tokens, mask, ar

    def compute_loss(
        self, rng: jax.Array, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> jax.Array:
        """Compute loss as in Pi0FAST, extended to handle point cloud data."""
        observation = _model.preprocess_observation(None, observation, train=train, 
                                                   image_keys=list(observation.images.keys()))
        tokens, mask, ar = self.embed_inputs(observation)
        attn_mask = _model.make_attn_mask(mask, ar)
        # Next-token prediction as in diffusion (prefix+suffix)
        targets = jax.nn.one_hot(observation.tokenized_prompt[:, 1:], self.PaliGemma.llm.module.vocab_size)
        # Forward through LLM for prefix + suffix (excluding last token)
        pre_logits, _, _ = self.PaliGemma.llm(embedded_prefix=tokens[:, :-1], mask=attn_mask[:, :-1, :-1], return_prelogits=True)
        # Decode only the new suffix tokens to compute logits
        logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1]:])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Compute cross-entropy loss on prompt tokens (masked)
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_nll = -jnp.sum(targets * log_probs, axis=-1)  # negative log-likelihood per token
        # Mean loss per sequence (only over actual target tokens)
        seq_loss = jnp.sum(token_nll * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)
        return seq_loss  # shape (batch_size,) loss per sequence

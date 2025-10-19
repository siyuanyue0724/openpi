import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import os
from openpi.models.sonata_encoder import Sonata


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        # ---- Sonata configuration (default: enabled; mode 'all') ----
        self.enable_sonata = bool(getattr(config, "enable_sonata", True))
        default_mode = os.environ.get("OPENPI_SONATA_MODE", "all")
        self.sonata_mode = str(getattr(config, "sonata_mode", default_mode))
        if self.sonata_mode not in ("off", "projector", "all"):
            raise RuntimeError(f"Invalid sonata_mode: {self.sonata_mode}")
        self.require_cuda  = bool(getattr(config, "require_cuda", True))
        self.sonata_ckpt   = getattr(config, "sonata_ckpt_path", None)
        self.point_start_id = getattr(config, "point_start_id", None)
        self.point_end_id   = getattr(config, "point_end_id", None)
        self.point_token_cap = int(getattr(config, "point_token_cap", 0) or 0)
        self.point_feat_dim  = int(getattr(config, "point_feat_dim", 6) or 6)
        # Lazily constructed encoder and projector
        self.sonata: Sonata | None = None
        self.pc_projector: nn.Linear | None = None
        # Cache for current batch (raw features, before projection)
        self._pt_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None


    def _ensure_sonata_ready(self, device, dtype):
        if not self.enable_sonata or self.sonata_mode == "off":
            raise RuntimeError("Sonata is disabled but encoder was requested.")
        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Sonata, but CUDA is not available.")
        if self.sonata is None:
            self.sonata = Sonata(in_channels=self.point_feat_dim)
            if self.sonata_ckpt:
                state = torch.load(self.sonata_ckpt, map_location="cpu")
                missing, unexpected = self.sonata.load_state_dict(state, strict=True)
                if missing or unexpected:
                    raise RuntimeError(f"Sonata ckpt mismatch: missing={missing}, unexpected={unexpected}")
            self.sonata.to(device=device, dtype=dtype).eval()

    def set_point_cache(self, pt_feat_raw: torch.Tensor, pt_mask: torch.Tensor) -> None:
        if pt_feat_raw.ndim != 3 or pt_mask.ndim != 2 or pt_feat_raw.shape[:2] != pt_mask.shape[:2]:
            raise RuntimeError(f"Invalid pt shapes: {pt_feat_raw.shape=}, {pt_mask.shape=}")
        self._pt_cache = (pt_feat_raw, pt_mask)

    @torch.no_grad()
    def torch_sonata_encode_batch(self, observation, train: bool = False):
        if not self.enable_sonata or self.sonata_mode == "off":
            raise RuntimeError("Sonata is disabled (enable_sonata=False or sonata_mode=off).")
        if "point_clouds" not in observation.__dict__ or not observation.point_clouds:
            raise RuntimeError("Sonata enabled but observation.point_clouds is missing or empty.")
        if "pointcloud" not in observation.point_clouds:
            raise RuntimeError("Sonata enabled but observation.point_clouds['pointcloud'] is missing.")

        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        self._ensure_sonata_ready(device, dtype)

        pcs   = observation.point_clouds["pointcloud"]
        pmask = observation.point_cloud_masks.get("pointcloud", None)
        if not isinstance(pcs, torch.Tensor):
            pcs = torch.as_tensor(pcs)
        pcs = pcs.to(device=device, dtype=dtype)
        B, M, D = pcs.shape
        if D < 6:
            raise RuntimeError(f"pointcloud last dim {D} < 6; expect [3 grid | 3 xyz | extras].")
        if pmask is None:
            pmask = torch.ones((B,), dtype=torch.bool, device=device)
        elif not isinstance(pmask, torch.Tensor):
            pmask = torch.as_tensor(pmask, dtype=torch.bool, device=device)
        else:
            pmask = pmask.to(device=device, dtype=torch.bool)
        if torch.isnan(pcs).any() or torch.isinf(pcs).any():
            raise RuntimeError("Point cloud contains NaN/Inf.")
        grid = pcs[..., :3]
        if (grid < 0).any():
            raise RuntimeError("Point grid (first 3 dims) must be non-negative.")
        feat = pcs[..., 3:]
        if feat.shape[-1] != self.point_feat_dim:
            raise RuntimeError(f"feat_dim mismatch: got {feat.shape[-1]}, expect config.point_feat_dim={self.point_feat_dim}")

        cap = int(self.point_token_cap)
        if cap <= 0:
            raise RuntimeError("point_token_cap must be > 0 when Sonata is enabled.")

        pt_list = []
        mask_list = []
        for b in range(B):
            if not pmask[b]:
                raise RuntimeError("Sonata enabled but point_cloud_masks['pointcloud'][b] is False.")
            arr = pcs[b]
            g = arr[:, :3].to(torch.int64)
            f = arr[:, 3:].to(dtype=torch.float32)
            c = f[:, :3].to(dtype=torch.float32)
            pad = (g == 0).all(dim=1) & (c == 0).all(dim=1) & (f == 0).all(dim=1)
            g = g[~pad]; c = c[~pad]; f = f[~pad]
            n = g.shape[0]
            if n == 0:
                raise RuntimeError("Empty point cloud after removing paddings.")
            sample = {"coord": c, "feat": f, "grid_coord": g,
                      "batch": torch.zeros((n,), dtype=torch.int64, device=device),
                      "offset": torch.tensor([n], dtype=torch.int64, device=device)}
            out_any = self.sonata(sample)
            enc = out_any if isinstance(out_any, torch.Tensor) else getattr(out_any, "feat", None)
            if enc is None:
                raise RuntimeError("Sonata.forward must return Tensor or object with .feat")
            real = int(enc.size(0))
            if real > cap:
                raise RuntimeError(f"[Sonata] token_len={real} exceeds point_token_cap={cap}.")
            if real < cap:
                pad_zeros = torch.zeros(cap - real, enc.size(1), dtype=enc.dtype, device=enc.device)
                enc = torch.cat([enc, pad_zeros], dim=0)
            elif real >= int(0.95 * cap):
                logging.getLogger("openpi").warning("[Sonata] token_len=%d is close to cap (%d).", real, cap)
            m = torch.zeros((cap,), dtype=torch.bool, device=device)
            m[:real] = True
            pt_list.append(enc.to(dtype=dtype))
            mask_list.append(m)
        pt_feat_raw = torch.stack(pt_list, dim=0)
        pt_mask     = torch.stack(mask_list, dim=0)
        self._pt_cache = (pt_feat_raw, pt_mask)
        return self._pt_cache

    @staticmethod
    def _insert_points_torch(
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        token_ids: torch.Tensor,
        pt_emb: torch.Tensor,
        pt_mask: torch.Tensor,
        start_id: int,
        end_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, E = text_emb.shape
        if token_ids.dtype != torch.long:
            raise RuntimeError("token_ids must be torch.long for embedding/indexing.")
        starts = (token_ids == start_id).nonzero(as_tuple=False)
        ends   = (token_ids == end_id).nonzero(as_tuple=False)
        s_pos = torch.full((B,), -1, device=token_ids.device, dtype=torch.long)
        e_pos = torch.full((B,), -1, device=token_ids.device, dtype=torch.long)
        for b in range(B):
            s_idx = starts[starts[:,0]==b][:,1]
            e_idx = ends[ends[:,0]==b][:,1]
            if s_idx.numel() != 1 or e_idx.numel() != 1:
                raise RuntimeError("Each sample must contain exactly one pair of point window tokens.")
            s_pos[b] = s_idx.item()
            e_pos[b] = e_idx.item()
            if not (0 <= s_pos[b] < e_pos[b] < S):
                raise RuntimeError(f"Invalid window positions: start={s_pos[b]}, end={e_pos[b]}, S={S}")
        fused_embs, fused_masks = [], []
        for b in range(B):
            s = s_pos[b].item(); e = e_pos[b].item()
            k = int(pt_mask[b].sum().item())
            if k == 0:
                raise RuntimeError("Empty point token sequence for a sample with a point window.")
            left_emb  = text_emb[b, :s+1, :]
            left_m    = text_mask[b, :s+1]
            mid_emb   = pt_emb[b, :k, :]
            mid_m     = pt_mask[b, :k]
            right_emb = text_emb[b, e:, :]
            right_m   = text_mask[b, e:]
            fused_embs.append(torch.cat([left_emb, mid_emb, right_emb], dim=0))
            fused_masks.append(torch.cat([left_m,   mid_m,   right_m  ], dim=0))
        L = max(x.shape[0] for x in fused_embs)
        out_emb = text_emb.new_zeros((B, L, E))
        out_msk = text_mask.new_zeros((B, L), dtype=torch.bool)
        for b in range(B):
            n = fused_embs[b].shape[0]
            out_emb[b, :n, :] = fused_embs[b]
            out_msk[b, :n]    = fused_masks[b]
        return out_emb, out_msk


    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        # --- Sonata point cloud insertion (default enabled) ---
        if self.enable_sonata:
            if self.point_start_id is None or self.point_end_id is None:
                raise RuntimeError("point_start_id/point_end_id must be set in config when Sonata is enabled.")
            if self.sonata_mode == "all":
                if self._pt_cache is None:
                    raise RuntimeError("Sonata mode=all requires torch_sonata_encode_batch() to be called before embed_prefix.")
                pt_feat_raw, pt_mask = self._pt_cache
                lang_emb_dim = lang_emb.shape[-1]
                if self.pc_projector is None:
                    self.pc_projector = nn.Linear(pt_feat_raw.shape[-1], lang_emb_dim, bias=False).to(device=lang_emb.device, dtype=lang_emb.dtype)
                pt_emb = self.pc_projector(pt_feat_raw.to(dtype=lang_emb.dtype, device=lang_emb.device))
                lang_emb, lang_masks = self._insert_points_torch(
                    lang_emb, lang_masks, lang_tokens, pt_emb, pt_mask, int(self.point_start_id), int(self.point_end_id)
                )

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        # --- Sonata encode (default enabled): compute pt cache for this batch ---
        if self.enable_sonata and self.sonata_mode in ("projector","all"):
            self.torch_sonata_encode_batch(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        if self.enable_sonata and self.sonata_mode in ("projector","all"):
            self.torch_sonata_encode_batch(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

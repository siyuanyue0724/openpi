"""Paper-faithful FLARE future-token alignment for the LingBot action expert."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

FLARE_GENERIC_TARGET_SCHEMA = "picf-next.flare-generic-siglip2.v1"
FLARE_SIGLIP2_MODEL_ID = "google/siglip2-large-patch16-256"
FLARE_SIGLIP2_REVISION = "787800c8990e6f058423089178e718139608408c"
FLARE_TARGET_VIEW_ORDER = ("rgb_static", "rgb_gripper")
FUTURE_LATENT_OBJECTIVE_SCALES = (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class FutureLatentAlignmentConfig:
    """Exact ADR-209 mapping of the successful generic-target FLARE arm."""

    future_token_count: int = 128
    action_hidden_width: int = 768
    target_width: int = 1024
    capture_layer_index: int = 26
    action_layer_count: int = 36
    loss_weight: float = 0.2
    future_token_init_std: float = 0.02
    target_offset_source_frames: int = 16
    tokens_per_view: int = 64
    view_order: tuple[str, ...] = FLARE_TARGET_VIEW_ORDER
    schema: str = FLARE_GENERIC_TARGET_SCHEMA

    def __post_init__(self) -> None:
        integer_fields = (
            self.future_token_count,
            self.action_hidden_width,
            self.target_width,
            self.capture_layer_index,
            self.action_layer_count,
            self.target_offset_source_frames,
            self.tokens_per_view,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("FLARE dimensions and layer identities must be integers")
        if (
            self.future_token_count <= 0
            or self.action_hidden_width <= 0
            or self.target_width <= 0
            or self.action_layer_count <= 0
            or not 0 <= self.capture_layer_index < self.action_layer_count
            or self.target_offset_source_frames <= 0
            or self.tokens_per_view <= 0
        ):
            raise ValueError("FLARE dimensions, offset, and layer identity must be positive")
        if self.future_token_count != self.tokens_per_view * len(self.view_order):
            raise ValueError("FLARE token count must retain every token from every target view")
        if tuple(self.view_order) != FLARE_TARGET_VIEW_ORDER:
            raise ValueError("ADR-209 requires fixed static-then-gripper target view order")
        if not 0.0 <= self.loss_weight <= 1.0:
            raise ValueError("FLARE loss weight must lie in [0,1]")
        if self.future_token_init_std <= 0:
            raise ValueError(
                "FLARE future-token initialization standard deviation must be positive"
            )
        if self.schema != FLARE_GENERIC_TARGET_SCHEMA:
            raise ValueError("unknown FLARE alignment schema")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_adr209_complete(self) -> None:
        """Reject resource-motivated reductions in a production ADR-209 arm."""

        expected = FutureLatentAlignmentConfig()
        if self != expected:
            raise ValueError(
                "production ADR-209 requires the complete frozen FLARE generic-target contract"
            )


@dataclass(frozen=True, slots=True)
class FutureLatentTargetBatch:
    """Detached, manifest-bound SigLIP2 targets for one factual policy batch."""

    tokens: torch.Tensor
    sample_keys: tuple[str, ...]
    source_global_indices: tuple[int, ...]
    future_global_indices: tuple[int, ...]
    manifest_sha256: str
    config_digest: str

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or self.tokens.dtype != torch.float32:
            raise ValueError("FLARE cache targets must be FP32 [batch,tokens,width]")
        if self.tokens.requires_grad:
            raise ValueError("FLARE target cache must be detached")
        if not torch.isfinite(self.tokens).all():
            raise ValueError("FLARE target cache contains NaN or infinity")
        batch = self.tokens.shape[0]
        if not (
            len(self.sample_keys)
            == len(self.source_global_indices)
            == len(self.future_global_indices)
            == batch
        ):
            raise ValueError("FLARE target identities differ from the target batch axis")
        if any(not isinstance(key, str) or not key for key in self.sample_keys):
            raise ValueError("FLARE target sample keys must be non-empty strings")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*self.source_global_indices, *self.future_global_indices)
        ):
            raise ValueError("FLARE source identities must be non-negative integers")
        if any(
            future - source != FutureLatentAlignmentConfig().target_offset_source_frames
            for source, future in zip(
                self.source_global_indices,
                self.future_global_indices,
                strict=True,
            )
        ):
            raise ValueError("FLARE targets must be exactly 16 raw source frames in the future")
        for name, digest in (
            ("manifest", self.manifest_sha256),
            ("config", self.config_digest),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"FLARE {name} digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FutureLatentAlignmentResult:
    """One attached FLARE objective kept separate from official LingBot loss."""

    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    mean_cosine: torch.Tensor
    prediction: torch.Tensor
    target_manifest_sha256: str
    capture_layer_index: int
    action_layer_count: int
    future_token_count: int

    def __post_init__(self) -> None:
        scalars = (self.raw_loss, self.weighted_loss, self.mean_cosine)
        if any(value.ndim != 0 or not value.is_floating_point() for value in scalars):
            raise ValueError("FLARE losses and cosine must be floating scalars")
        if any(not torch.isfinite(value) for value in scalars):
            raise ValueError("FLARE result contains NaN or infinity")
        if self.prediction.ndim != 3 or not self.prediction.is_floating_point():
            raise ValueError("FLARE prediction must be [batch,tokens,width]")
        if self.prediction.shape[1] != self.future_token_count:
            raise ValueError("FLARE prediction token count differs from its receipt")
        if not torch.isfinite(self.prediction).all():
            raise ValueError("FLARE prediction contains NaN or infinity")
        if not 0 <= self.capture_layer_index < self.action_layer_count:
            raise ValueError("FLARE result has an invalid capture layer")


def future_latent_objective_contribution(
    result: FutureLatentAlignmentResult,
    *,
    scale: float,
) -> torch.Tensor:
    """Apply the preregistered candidate/control coefficient to one intact arm."""

    if not isinstance(result, FutureLatentAlignmentResult):
        raise TypeError("future-latent objective requires its typed result")
    if isinstance(scale, bool) or scale not in FUTURE_LATENT_OBJECTIVE_SCALES:
        raise ValueError("future-latent objective scale must be exactly zero or one")
    return result.weighted_loss * scale


class FutureLatentForwardContext:
    """Single-use capture object passed through one LingBot policy forward."""

    __slots__ = (
        "_captured_hidden",
        "_config",
        "_finalized",
        "_native_suffix_count",
        "_result",
        "_target",
        "_total_suffix_count",
    )

    def __init__(
        self,
        *,
        config: FutureLatentAlignmentConfig,
        target: FutureLatentTargetBatch,
    ) -> None:
        if target.config_digest != config.digest:
            raise ValueError("FLARE target cache and model configuration differ")
        if target.tokens.shape[1:] != (config.future_token_count, config.target_width):
            raise ValueError("FLARE target shape differs from the configured complete arm")
        self._config = config
        self._target = target
        self._native_suffix_count: int | None = None
        self._total_suffix_count: int | None = None
        self._captured_hidden: torch.Tensor | None = None
        self._result: FutureLatentAlignmentResult | None = None
        self._finalized = False

    @property
    def target(self) -> FutureLatentTargetBatch:
        return self._target

    def bind_suffix(self, *, native_suffix_count: int, total_suffix_count: int) -> None:
        if self._native_suffix_count is not None or self._total_suffix_count is not None:
            raise RuntimeError("FLARE forward context was bound to a suffix more than once")
        if (
            isinstance(native_suffix_count, bool)
            or isinstance(total_suffix_count, bool)
            or not isinstance(native_suffix_count, int)
            or not isinstance(total_suffix_count, int)
            or native_suffix_count <= 1
            or total_suffix_count != native_suffix_count + self._config.future_token_count
        ):
            raise ValueError("FLARE suffix does not preserve state, actions, and all future tokens")
        self._native_suffix_count = native_suffix_count
        self._total_suffix_count = total_suffix_count

    def record_action_hidden(
        self,
        *,
        action_hidden: torch.Tensor,
        layer_index: int,
        layer_count: int,
    ) -> None:
        if self._native_suffix_count is None or self._total_suffix_count is None:
            raise RuntimeError("FLARE capture occurred before suffix binding")
        if layer_count != self._config.action_layer_count:
            raise ValueError("LingBot action depth differs from the frozen FLARE mapping")
        if not 0 <= layer_index < layer_count:
            raise ValueError("FLARE capture received an invalid action layer identity")
        expected = (
            self._target.tokens.shape[0],
            self._total_suffix_count,
            self._config.action_hidden_width,
        )
        if action_hidden.shape != expected or not action_hidden.is_floating_point():
            raise ValueError("LingBot action hidden surface differs from the FLARE contract")
        if layer_index != self._config.capture_layer_index:
            return
        if self._captured_hidden is not None:
            raise RuntimeError("FLARE capture layer executed more than once in one forward")
        self._captured_hidden = action_hidden[:, self._native_suffix_count :]
        if self._captured_hidden.shape[1] != self._config.future_token_count:
            raise RuntimeError("FLARE capture dropped future tokens")

    def finalize(
        self,
        alignment: LingBotFutureLatentAlignment,
        *,
        require_grad: bool,
    ) -> FutureLatentAlignmentResult:
        if self._finalized:
            raise RuntimeError("FLARE forward context may be finalized only once")
        if self._captured_hidden is None:
            raise RuntimeError("LingBot omitted the registered FLARE capture layer")
        if alignment.config != self._config:
            raise ValueError("FLARE context was finalized by a different alignment module")
        decoder_weight = alignment.embedding_decoder[0].weight
        decoder_input = self._captured_hidden.to(
            device=decoder_weight.device,
            dtype=decoder_weight.dtype,
        )
        prediction = alignment.embedding_decoder(decoder_input)
        prediction_fp32 = prediction.float()
        target = self._target.tokens.to(
            device=prediction.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        cosine = F.cosine_similarity(prediction_fp32, target, dim=-1, eps=1e-8)
        raw_loss = (1.0 - cosine).mean()
        weighted_loss = raw_loss * self._config.loss_weight
        if require_grad and (
            not prediction.requires_grad
            or not raw_loss.requires_grad
            or not weighted_loss.requires_grad
        ):
            raise RuntimeError("FLARE loss detached from the LingBot action expert")
        result = FutureLatentAlignmentResult(
            raw_loss=raw_loss,
            weighted_loss=weighted_loss,
            mean_cosine=cosine.mean(),
            prediction=prediction,
            target_manifest_sha256=self._target.manifest_sha256,
            capture_layer_index=self._config.capture_layer_index,
            action_layer_count=self._config.action_layer_count,
            future_token_count=self._config.future_token_count,
        )
        self._result = result
        self._finalized = True
        return result

    def finalized_result(self, *, require_grad: bool) -> FutureLatentAlignmentResult:
        """Return the result computed inside the still-unsharded policy root."""

        if not self._finalized or self._result is None:
            raise RuntimeError("LingBot policy omitted in-forward FLARE finalization")
        if require_grad and (
            not self._result.prediction.requires_grad
            or not self._result.raw_loss.requires_grad
            or not self._result.weighted_loss.requires_grad
        ):
            raise RuntimeError("finalized FLARE loss detached from the LingBot action expert")
        return self._result


class LingBotFutureLatentAlignment(nn.Module):
    """Learned FLARE tokens and the paper-required two-layer decoder."""

    def __init__(self, config: FutureLatentAlignmentConfig | None = None) -> None:
        super().__init__()
        self.config = FutureLatentAlignmentConfig() if config is None else config
        self.future_tokens = nn.Embedding(
            self.config.future_token_count,
            self.config.action_hidden_width,
        )
        nn.init.normal_(
            self.future_tokens.weight,
            mean=0.0,
            std=self.config.future_token_init_std,
        )
        self.embedding_decoder = nn.Sequential(
            nn.Linear(self.config.action_hidden_width, self.config.action_hidden_width),
            nn.SiLU(),
            nn.Linear(self.config.action_hidden_width, self.config.target_width),
        )

    def new_forward_context(
        self,
        target: FutureLatentTargetBatch,
    ) -> FutureLatentForwardContext:
        return FutureLatentForwardContext(config=self.config, target=target)

    def append_future_tokens(
        self,
        *,
        suffix_embeddings: torch.Tensor,
        suffix_valid: torch.Tensor,
        suffix_blocks: torch.Tensor,
        context: FutureLatentForwardContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            suffix_embeddings.ndim != 3
            or suffix_embeddings.shape[-1] != self.config.action_hidden_width
            or suffix_valid.shape != suffix_embeddings.shape[:2]
            or suffix_blocks.shape != suffix_embeddings.shape[:2]
            or suffix_valid.dtype != torch.bool
            or suffix_blocks.dtype != torch.bool
        ):
            raise ValueError("LingBot suffix differs from the FLARE insertion contract")
        batch, native_suffix_count, _width = suffix_embeddings.shape
        future = self.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
        future = future.to(device=suffix_embeddings.device, dtype=suffix_embeddings.dtype)
        valid = torch.ones(
            (batch, self.config.future_token_count),
            dtype=torch.bool,
            device=suffix_valid.device,
        )
        # False continues LingBot's bidirectional action block; no new causal block is made.
        blocks = torch.zeros_like(valid, device=suffix_blocks.device)
        total_suffix_count = native_suffix_count + self.config.future_token_count
        if context is not None:
            if not isinstance(context, FutureLatentForwardContext):
                raise TypeError("FLARE training requires a typed single-forward context")
            context.bind_suffix(
                native_suffix_count=native_suffix_count,
                total_suffix_count=total_suffix_count,
            )
        return (
            torch.cat((suffix_embeddings, future), dim=1),
            torch.cat((suffix_valid, valid), dim=1),
            torch.cat((suffix_blocks, blocks), dim=1),
        )


def _unwrapped_policy(policy: nn.Module) -> nn.Module:
    root = policy
    seen: set[int] = set()
    while isinstance(getattr(root, "module", None), nn.Module):
        if id(root) in seen:
            raise ValueError("LingBot policy wrappers contain a module cycle")
        seen.add(id(root))
        root = root.module
    return root


def installed_future_latent_alignment(policy: nn.Module) -> LingBotFutureLatentAlignment | None:
    root = _unwrapped_policy(policy)
    flow = getattr(root, "model", None)
    alignment = getattr(flow, "picf_future_latent_alignment", None)
    if alignment is None:
        return None
    if not isinstance(alignment, LingBotFutureLatentAlignment):
        raise TypeError("LingBot exposes an incompatible future-latent alignment module")
    return alignment


def install_lingbot_future_latent_alignment(
    policy: nn.Module,
    alignment: LingBotFutureLatentAlignment,
    *,
    require_adr209_complete: bool = True,
) -> None:
    """Attach FLARE through the audited LingBot hook before FSDP2 wrapping."""

    if not isinstance(alignment, LingBotFutureLatentAlignment):
        raise TypeError("LingBot FLARE installation requires its typed alignment module")
    if require_adr209_complete:
        alignment.config.assert_adr209_complete()
    root = _unwrapped_policy(policy)
    if installed_future_latent_alignment(root) is not None:
        raise RuntimeError("LingBot already has a future-latent alignment module")
    setter = getattr(root, "set_picf_future_latent_alignment", None)
    if not callable(setter):
        raise TypeError("LingBot policy lacks the audited FLARE registration hook")
    setter(alignment)
    if installed_future_latent_alignment(root) is not alignment:
        raise RuntimeError("LingBot did not retain the exact FLARE module instance")

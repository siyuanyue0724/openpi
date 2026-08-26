"""Typed runtime around the byte-identical VidEoMT-DINOv3 backbone source."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from picf_next._vendor.videomt.modeling.backbone.videomt import VidEoMT_CLASS
from picf_next.videomt_exact.checkpoint import (
    AdaptedCheckpointReceipt,
    adapted_videomt_model_state,
    published_videomt_backbone_state,
)

VIDEOMT_DINOV3_L_QUERIES = 200
VIDEOMT_DINOV3_L_WIDTH = 1024
VIDEOMT_DINOV3_L_CLASSES = 40
VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS = (20, 21, 22, 23)
VIDEOMT_PIXEL_MEAN_255 = (123.675, 116.280, 103.530)
VIDEOMT_PIXEL_STD_255 = (58.395, 57.120, 57.375)


@dataclass(frozen=True, slots=True)
class ExactVidEoMTConfig:
    """Released architecture values plus immutable local asset locations."""

    checkpoint_path: Path
    local_dinov3_bundle: Path
    adapted_checkpoint_path: Path | None = None
    adapted_checkpoint_sha256: str | None = None
    constructor_image_size: int = 640
    num_frames: int = 5
    num_queries: int = VIDEOMT_DINOV3_L_QUERIES
    num_classes: int = VIDEOMT_DINOV3_L_CLASSES
    segmenter_blocks: tuple[int, ...] = VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS
    fused_qkv: bool = False
    norm_queries: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if not (self.local_dinov3_bundle / "config.json").is_file():
            raise FileNotFoundError(self.local_dinov3_bundle / "config.json")
        if not (self.local_dinov3_bundle / "model.safetensors").is_file():
            raise FileNotFoundError(self.local_dinov3_bundle / "model.safetensors")
        if (self.adapted_checkpoint_path is None) != (
            self.adapted_checkpoint_sha256 is None
        ):
            raise ValueError("adapted VidEoMT path and SHA-256 must be supplied together")
        if self.adapted_checkpoint_path is not None and not (
            self.adapted_checkpoint_path.is_file()
        ):
            raise FileNotFoundError(self.adapted_checkpoint_path)
        if self.num_queries != VIDEOMT_DINOV3_L_QUERIES:
            raise ValueError("exact upstream reproduction requires 200 object queries")
        if self.num_classes != VIDEOMT_DINOV3_L_CLASSES:
            raise ValueError("released YouTube-VIS 2019 checkpoint requires 40 classes")
        if self.segmenter_blocks != VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS:
            raise ValueError("exact upstream reproduction requires DINOv3 blocks 20-23")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")


@dataclass(frozen=True, slots=True)
class ExactVidEoMTOutput:
    """All released outputs plus the official propagated query state."""

    class_logits: torch.Tensor
    mask_logits: torch.Tensor
    query_embeddings: torch.Tensor
    propagated_queries: torch.Tensor
    auxiliary_outputs: tuple[dict[str, torch.Tensor], ...]
    prediction_query_surface: torch.Tensor | None = None
    latest_mask_embeddings: torch.Tensor | None = None
    latest_mask_features: torch.Tensor | None = None
    latest_segmenter_input_tokens: torch.Tensor | None = None
    latest_position_cos: torch.Tensor | None = None
    latest_position_sin: torch.Tensor | None = None
    latest_patch_grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.class_logits.ndim != 4:
            raise ValueError("class logits must have shape [batch, time, query, class]")
        if self.mask_logits.ndim != 5:
            raise ValueError("mask logits must have shape [batch, query, time, height, width]")
        batch, time, queries, classes = self.class_logits.shape
        mask_batch, mask_queries, mask_time, _height, _width = self.mask_logits.shape
        if (
            (mask_batch, mask_time, mask_queries) != (batch, time, queries)
            or self.query_embeddings.shape != (batch, time, queries, VIDEOMT_DINOV3_L_WIDTH)
            or self.propagated_queries.shape != (batch, queries, VIDEOMT_DINOV3_L_WIDTH)
            or queries != VIDEOMT_DINOV3_L_QUERIES
            or classes != VIDEOMT_DINOV3_L_CLASSES + 1
        ):
            raise ValueError("VidEoMT output axes do not match the released architecture")
        tensors = (
            self.class_logits,
            self.mask_logits,
            self.query_embeddings,
            self.propagated_queries,
        )
        if any(
            not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors
        ):
            raise ValueError("VidEoMT outputs must be finite floating tensors")
        if self.prediction_query_surface is not None:
            if self.prediction_query_surface.shape != (
                batch * time,
                queries,
                VIDEOMT_DINOV3_L_WIDTH,
            ):
                raise ValueError("VidEoMT prediction-query surface has invalid axes")
            if (
                self.prediction_query_surface.device != self.query_embeddings.device
                or self.prediction_query_surface.dtype != self.query_embeddings.dtype
                or not self.prediction_query_surface.is_floating_point()
                or not torch.isfinite(self.prediction_query_surface).all()
            ):
                raise ValueError(
                    "VidEoMT prediction-query surface must match the released query output"
                )
        if (self.latest_mask_embeddings is None) != (self.latest_mask_features is None):
            raise ValueError("VidEoMT mask decoder outputs must be present together")
        if self.latest_mask_embeddings is not None:
            if self.latest_mask_embeddings.shape != (
                batch,
                queries,
                VIDEOMT_DINOV3_L_WIDTH,
            ):
                raise ValueError("VidEoMT mask embeddings have invalid axes")
            if self.latest_mask_features.shape != (
                batch,
                VIDEOMT_DINOV3_L_WIDTH,
                self.mask_logits.shape[-2],
                self.mask_logits.shape[-1],
            ):
                raise ValueError("VidEoMT dense mask features have invalid axes")
            decoder_outputs = (self.latest_mask_embeddings, self.latest_mask_features)
            if any(
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or value.device != self.mask_logits.device
                or value.dtype != self.mask_logits.dtype
                for value in decoder_outputs
            ):
                raise ValueError(
                    "VidEoMT mask decoder outputs must be finite and match mask logits"
                )
        refinement = (
            self.latest_segmenter_input_tokens,
            self.latest_position_cos,
            self.latest_position_sin,
            self.latest_patch_grid_shape,
        )
        if any(value is not None for value in refinement):
            if any(value is None for value in refinement):
                raise ValueError("VidEoMT refinement boundary must be present atomically")
            segmenter_input = self.latest_segmenter_input_tokens
            position_cos = self.latest_position_cos
            position_sin = self.latest_position_sin
            patch_grid = self.latest_patch_grid_shape
            if (
                segmenter_input is None
                or position_cos is None
                or position_sin is None
                or patch_grid is None
            ):
                raise RuntimeError("validated VidEoMT refinement boundary disappeared")
            patch_height, patch_width = patch_grid
            patch_count = patch_height * patch_width
            if (
                patch_height <= 0
                or patch_width <= 0
                or segmenter_input.shape
                != (batch, queries + 5 + patch_count, VIDEOMT_DINOV3_L_WIDTH)
                or position_cos.shape[-2:] != (patch_count, 64)
                or position_sin.shape != position_cos.shape
            ):
                raise ValueError("VidEoMT refinement boundary has invalid axes")
            if any(
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or value.device != self.mask_logits.device
                or value.dtype != self.mask_logits.dtype
                for value in (segmenter_input, position_cos, position_sin)
            ):
                raise ValueError("VidEoMT refinement boundary must match released outputs")


def merge_exact_videomt_causal_outputs(
    outputs: Sequence[ExactVidEoMTOutput],
) -> ExactVidEoMTOutput:
    """Reassemble sequential one-frame executions into the released video ABI.

    This is an execution-schedule transform only. It retains every query, final
    prediction and auxiliary prediction, and rejects any source output key that
    cannot be merged without interpretation.
    """

    values = tuple(outputs)
    if not values:
        raise ValueError("causal VidEoMT output sequence must be non-empty")
    first = values[0]
    batch = first.class_logits.shape[0]
    queries = first.class_logits.shape[2]
    classes = first.class_logits.shape[3]
    mask_shape = first.mask_logits.shape[-2:]
    aux_count = len(first.auxiliary_outputs)
    for value in values:
        if (
            value.class_logits.shape != (batch, 1, queries, classes)
            or value.mask_logits.shape != (batch, queries, 1, *mask_shape)
            or value.query_embeddings.shape != (batch, 1, queries, VIDEOMT_DINOV3_L_WIDTH)
            or value.propagated_queries.shape != (batch, queries, VIDEOMT_DINOV3_L_WIDTH)
            or len(value.auxiliary_outputs) != aux_count
        ):
            raise ValueError("causal VidEoMT frame outputs do not share the released ABI")

    auxiliary: list[dict[str, torch.Tensor]] = []
    for layer in range(aux_count):
        layer_values = tuple(value.auxiliary_outputs[layer] for value in values)
        if any(set(item) != {"pred_logits", "pred_masks"} for item in layer_values):
            raise ValueError("causal VidEoMT auxiliary output inventory changed")
        if any(
            item["pred_logits"].shape != (batch, 1, queries, classes)
            or item["pred_masks"].shape != (batch, queries, 1, *mask_shape)
            for item in layer_values
        ):
            raise ValueError("causal VidEoMT auxiliary output axes changed")
        auxiliary.append(
            {
                "pred_logits": torch.cat(
                    tuple(item["pred_logits"] for item in layer_values),
                    dim=1,
                ),
                "pred_masks": torch.cat(
                    tuple(item["pred_masks"] for item in layer_values),
                    dim=2,
                ),
            }
        )

    latest = values[-1]
    return ExactVidEoMTOutput(
        class_logits=torch.cat(tuple(value.class_logits for value in values), dim=1),
        mask_logits=torch.cat(tuple(value.mask_logits for value in values), dim=2),
        query_embeddings=torch.cat(
            tuple(value.query_embeddings for value in values),
            dim=1,
        ),
        propagated_queries=latest.propagated_queries,
        auxiliary_outputs=tuple(auxiliary),
        latest_mask_embeddings=latest.latest_mask_embeddings,
        latest_mask_features=latest.latest_mask_features,
        latest_segmenter_input_tokens=latest.latest_segmenter_input_tokens,
        latest_position_cos=latest.latest_position_cos,
        latest_position_sin=latest.latest_position_sin,
        latest_patch_grid_shape=latest.latest_patch_grid_shape,
    )


@dataclass(frozen=True, slots=True)
class ExactVidEoMTCausalSequenceOutput:
    """Complete source output plus each causal frame boundary and state."""

    merged: ExactVidEoMTOutput
    per_frame: tuple[ExactVidEoMTOutput, ...]
    propagated_queries_by_frame: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        time = self.merged.class_logits.shape[1]
        if (
            time <= 0
            or len(self.per_frame) != time
            or len(self.propagated_queries_by_frame) != time
        ):
            raise ValueError("causal VidEoMT sequence lengths differ")
        expected = self.merged.propagated_queries.shape
        if any(value.class_logits.shape[1] != 1 for value in self.per_frame):
            raise ValueError("causal VidEoMT per-frame outputs must contain one frame")
        if any(
            state.shape != expected
            or state.device != self.merged.propagated_queries.device
            or state.dtype != self.merged.propagated_queries.dtype
            or not torch.isfinite(state).all()
            for state in self.propagated_queries_by_frame
        ):
            raise ValueError("causal VidEoMT propagated states changed ABI")
        if not torch.equal(
            self.propagated_queries_by_frame[-1],
            self.merged.propagated_queries,
        ):
            raise ValueError("causal VidEoMT merged output lost the final propagated state")


class ExactVidEoMTRuntime(nn.Module):
    """Stateful wrapper that leaves the released model graph unchanged."""

    def __init__(
        self,
        config: ExactVidEoMTConfig,
        model: VidEoMT_CLASS,
        *,
        adapted_checkpoint_receipt: AdaptedCheckpointReceipt | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.adapted_checkpoint_receipt = adapted_checkpoint_receipt
        self._query_state_binding_unsharded = False

    @property
    def propagated_queries(self) -> torch.Tensor | None:
        return self.model.last_query_embed

    def reset_state(self) -> None:
        self.model._clear_memory()
        if self._query_state_binding_unsharded:
            reshard = getattr(self.model, "reshard", None)
            if not callable(reshard):
                raise RuntimeError("FSDP2 source lost reshard() after explicit query binding")
            reshard()
            self._query_state_binding_unsharded = False

    def _unshard_query_state_parameter(self) -> None:
        """Materialize root-owned query weights before a state-only operation."""

        unshard = getattr(self.model, "unshard", None)
        if unshard is None:
            return
        if not callable(unshard) or not callable(getattr(self.model, "reshard", None)):
            raise RuntimeError("FSDP2 source exposes an incomplete unshard/reshard interface")
        if self._query_state_binding_unsharded:
            raise RuntimeError("VidEoMT query state was bound twice before a source forward")
        unshard(async_op=False)
        self._query_state_binding_unsharded = True

    def bind_mixed_propagated_queries(
        self,
        previous_queries: torch.Tensor | None,
        *,
        reset: torch.Tensor,
    ) -> torch.Tensor:
        """Bind one exact initial query state for a mixed reset/resume batch.

        The released model exposes one scalar ``resume`` flag for the complete
        batch.  Its reset branch seeds a sample with ``q.weight`` and its resume
        branch seeds it with ``last_query_embed``.  Selecting between those two
        released tensors per sample is therefore the exact batched form of the
        upstream branches; it adds no state transition or learned component.
        """

        if reset.ndim != 1 or reset.dtype != torch.bool or not reset.numel():
            raise ValueError("VidEoMT reset mask must be non-empty boolean [batch]")
        self._unshard_query_state_parameter()
        try:
            learned_queries = getattr(getattr(self.model, "q", None), "weight", None)
            if not isinstance(learned_queries, torch.Tensor) or learned_queries.shape != (
                VIDEOMT_DINOV3_L_QUERIES,
                VIDEOMT_DINOV3_L_WIDTH,
            ):
                raise RuntimeError("released VidEoMT learned-query parameter changed ABI")
            if reset.device != learned_queries.device:
                raise ValueError("VidEoMT reset mask and learned queries must share one device")
            batch = reset.shape[0]
            cold = learned_queries.unsqueeze(0).expand(batch, -1, -1)
            if previous_queries is None:
                if not reset.all():
                    raise ValueError("resumed VidEoMT samples require propagated query state")
                initial = cold
            else:
                if (
                    previous_queries.shape != cold.shape
                    or previous_queries.device != cold.device
                    or previous_queries.dtype != cold.dtype
                    or not previous_queries.is_floating_point()
                    or not torch.isfinite(previous_queries).all()
                ):
                    raise ValueError("cached VidEoMT queries differ from the released state ABI")
                initial = torch.where(reset[:, None, None], cold, previous_queries)
            self.model.last_query_embed = initial
            return initial
        except BaseException:
            self.reset_state()
            raise

    def restore_propagated_queries(self, queries: torch.Tensor) -> None:
        """Restore an exact frame boundary without detaching its graph."""

        if (
            queries.ndim != 3
            or queries.shape[1:] != (
                VIDEOMT_DINOV3_L_QUERIES,
                VIDEOMT_DINOV3_L_WIDTH,
            )
            or not queries.is_floating_point()
            or not torch.isfinite(queries).all()
        ):
            raise ValueError("VidEoMT propagated queries have an invalid state ABI")
        # FSDP2 keeps the idle parameter shard in FP32 but materializes the
        # mixed-precision parameter used by the source forward on unshard.  The
        # recurrent state must be checked against that compute parameter, not
        # against the idle shard dtype.
        self._unshard_query_state_parameter()
        try:
            learned_queries = getattr(getattr(self.model, "q", None), "weight", None)
            if not isinstance(learned_queries, torch.Tensor):
                raise RuntimeError("released VidEoMT learned-query parameter is unavailable")
            if queries.device != learned_queries.device or queries.dtype != learned_queries.dtype:
                raise ValueError(
                    "VidEoMT propagated state and compute queries must share placement"
                )
            self.model.last_query_embed = queries
        except BaseException:
            self.reset_state()
            raise
        if self._query_state_binding_unsharded:
            reshard = getattr(self.model, "reshard", None)
            if not callable(reshard):
                self.reset_state()
                raise RuntimeError("FSDP2 source lost reshard() after state restoration")
            reshard()
            self._query_state_binding_unsharded = False

    def forward(
        self,
        normalized_padded_rgb: torch.Tensor,
        *,
        resume: bool = False,
    ) -> ExactVidEoMTOutput:
        if normalized_padded_rgb.ndim != 4 or normalized_padded_rgb.shape[1] != 3:
            raise ValueError("VidEoMT input must have shape [time, 3, height, width]")
        if normalized_padded_rgb.shape[-2] % 16 or normalized_padded_rgb.shape[-1] % 16:
            raise ValueError("VidEoMT input height and width must be divisible by patch size 16")
        if (
            not normalized_padded_rgb.is_floating_point()
            or not torch.isfinite(normalized_padded_rgb).all()
        ):
            raise ValueError("VidEoMT input must be finite floating RGB")

        final_query_input: torch.Tensor | None = None
        final_mask_embeddings: torch.Tensor | None = None
        final_mask_features: torch.Tensor | None = None
        final_segmenter_input: torch.Tensor | None = None
        position_cos: torch.Tensor | None = None
        position_sin: torch.Tensor | None = None

        def capture_prediction_queries(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            nonlocal final_query_input
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise RuntimeError("released VidEoMT class head received an unexpected input")
            # The released final prediction is the last _predict call. Replacing
            # this reference on auxiliary calls avoids retaining extra graphs.
            final_query_input = inputs[0]

        def capture_mask_embeddings(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            nonlocal final_mask_embeddings
            if output.ndim != 3:
                raise RuntimeError("released VidEoMT mask head produced unexpected axes")
            final_mask_embeddings = output

        def capture_mask_features(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            nonlocal final_mask_features
            if output.ndim != 4:
                raise RuntimeError("released VidEoMT upscale path produced unexpected axes")
            final_mask_features = output

        def capture_segmenter_input(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            nonlocal final_segmenter_input
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise RuntimeError("released segmenter block received unexpected input axes")
            final_segmenter_input = inputs[0]

        def capture_position_embeddings(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            nonlocal position_cos, position_sin
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or output[0].shape != output[1].shape
            ):
                raise RuntimeError("released DINOv3 RoPE produced an unexpected output")
            position_cos, position_sin = output

        capture_handle = self.model.class_head.register_forward_pre_hook(capture_prediction_queries)
        mask_embedding_handle = self.model.mask_head.register_forward_hook(capture_mask_embeddings)
        mask_feature_handle = self.model.upscale.register_forward_hook(capture_mask_features)
        backbone = getattr(getattr(self.model, "encoder", None), "backbone", None)
        refinement_boundary_available = (
            backbone is not None
            and hasattr(backbone, "blocks")
            and len(backbone.blocks) > self.config.segmenter_blocks[0]
            and hasattr(backbone.blocks[self.config.segmenter_blocks[0]], "norm1")
            and hasattr(backbone, "rope_embeddings")
        )
        segmenter_input_handle = None
        position_handle = None
        if refinement_boundary_available:
            segmenter_input_handle = backbone.blocks[
                self.config.segmenter_blocks[0]
            ].norm1.register_forward_pre_hook(capture_segmenter_input)
            position_handle = backbone.rope_embeddings.register_forward_hook(
                capture_position_embeddings
            )
        try:
            outputs = self.model(normalized_padded_rgb, resume=resume)
        except BaseException:
            self.reset_state()
            raise
        finally:
            capture_handle.remove()
            mask_embedding_handle.remove()
            mask_feature_handle.remove()
            if segmenter_input_handle is not None:
                segmenter_input_handle.remove()
            if position_handle is not None:
                position_handle.remove()
        self._query_state_binding_unsharded = False
        propagated = self.model.last_query_embed
        if propagated is None:
            raise RuntimeError("released VidEoMT forward did not produce propagated queries")
        if final_query_input is None:
            raise RuntimeError("released VidEoMT forward did not expose final prediction queries")
        if final_mask_embeddings is None or final_mask_features is None:
            raise RuntimeError("released VidEoMT forward did not expose final mask decoder outputs")
        if refinement_boundary_available and (
            final_segmenter_input is None or position_cos is None or position_sin is None
        ):
            raise RuntimeError("released VidEoMT forward did not expose its refinement boundary")
        batch, time, queries, _classes = outputs["pred_logits"].shape
        expected_query_shape = (batch * time, queries, VIDEOMT_DINOV3_L_WIDTH)
        if final_query_input.shape != expected_query_shape:
            raise RuntimeError(
                "released final prediction query shape differs from output axes: "
                f"{tuple(final_query_input.shape)} != {expected_query_shape}"
            )
        query_embeddings = final_query_input.reshape(
            batch,
            time,
            queries,
            VIDEOMT_DINOV3_L_WIDTH,
        )
        expected_mask_embedding_shape = (
            batch * time,
            queries,
            VIDEOMT_DINOV3_L_WIDTH,
        )
        if final_mask_embeddings.shape != expected_mask_embedding_shape:
            raise RuntimeError(
                "released final mask embedding shape differs from output axes: "
                f"{tuple(final_mask_embeddings.shape)} != {expected_mask_embedding_shape}"
            )
        expected_mask_feature_shape = (
            batch * time,
            VIDEOMT_DINOV3_L_WIDTH,
            outputs["pred_masks"].shape[-2],
            outputs["pred_masks"].shape[-1],
        )
        if final_mask_features.shape != expected_mask_feature_shape:
            raise RuntimeError(
                "released final mask feature shape differs from output axes: "
                f"{tuple(final_mask_features.shape)} != {expected_mask_feature_shape}"
            )
        latest_mask_embeddings = final_mask_embeddings.reshape(
            batch,
            time,
            queries,
            VIDEOMT_DINOV3_L_WIDTH,
        )[:, -1].clone()
        latest_mask_features = final_mask_features.reshape(
            batch,
            time,
            VIDEOMT_DINOV3_L_WIDTH,
            outputs["pred_masks"].shape[-2],
            outputs["pred_masks"].shape[-1],
        )[:, -1].clone()
        patch_grid_shape: tuple[int, int] | None = None
        if refinement_boundary_available:
            patch_height = outputs["pred_masks"].shape[-2] // 4
            patch_width = outputs["pred_masks"].shape[-1] // 4
            if (
                patch_height * 4 != outputs["pred_masks"].shape[-2]
                or patch_width * 4 != outputs["pred_masks"].shape[-1]
            ):
                raise RuntimeError("released upscaler geometry is not four times the patch grid")
            patch_grid_shape = (patch_height, patch_width)
        auxiliary = tuple(outputs.get("aux_outputs", ()))
        return ExactVidEoMTOutput(
            class_logits=outputs["pred_logits"],
            mask_logits=outputs["pred_masks"],
            query_embeddings=query_embeddings,
            propagated_queries=propagated,
            auxiliary_outputs=auxiliary,
            prediction_query_surface=final_query_input,
            latest_mask_embeddings=latest_mask_embeddings,
            latest_mask_features=latest_mask_features,
            latest_segmenter_input_tokens=(
                final_segmenter_input.clone() if final_segmenter_input is not None else None
            ),
            latest_position_cos=position_cos.clone() if position_cos is not None else None,
            latest_position_sin=position_sin.clone() if position_sin is not None else None,
            latest_patch_grid_shape=patch_grid_shape,
        )

    def forward_causal_sequence(
        self,
        normalized_padded_rgb: torch.Tensor,
        *,
        resume: bool = False,
    ) -> ExactVidEoMTCausalSequenceOutput:
        """Execute the unchanged source one causal frame at a time.

        The released training implementation uses ``model.num_frames`` only to
        recover the batch axis from a flattened video tensor. Temporarily using
        one frame per call exposes its native ``resume`` state without changing
        a parameter, layer, query, prediction head or objective. Dropout and
        stochastic depth are both zero in the pinned DINOv3-L configuration.
        """

        if normalized_padded_rgb.ndim == 4:
            if normalized_padded_rgb.shape[0] <= 0:
                raise ValueError("causal VidEoMT input must contain one or more RGB frames")
            frames = normalized_padded_rgb.unsqueeze(0)
        elif normalized_padded_rgb.ndim == 5:
            if min(normalized_padded_rgb.shape[:2]) <= 0:
                raise ValueError("causal VidEoMT batch and time axes must be non-empty")
            frames = normalized_padded_rgb
        else:
            raise ValueError(
                "causal VidEoMT input must be [time,3,height,width] or "
                "[batch,time,3,height,width]"
            )
        if frames.shape[2] != 3:
            raise ValueError("causal VidEoMT input must contain three RGB channels")
        original_num_frames = getattr(self.model, "num_frames", None)
        if (
            isinstance(original_num_frames, bool)
            or not isinstance(original_num_frames, int)
            or original_num_frames <= 0
        ):
            raise TypeError("released VidEoMT model has no valid num_frames contract")
        configured_num_frames = getattr(self.config, "num_frames", original_num_frames)
        if original_num_frames != configured_num_frames:
            raise RuntimeError("released VidEoMT runtime and model frame contracts differ")

        frame_outputs: list[ExactVidEoMTOutput] = []
        propagated_states: list[torch.Tensor] = []
        self.model.num_frames = 1
        try:
            for frame_index in range(frames.shape[1]):
                output = self.forward(
                    frames[:, frame_index],
                    resume=resume or frame_index > 0,
                )
                frame_outputs.append(output)
                # The source replaces rather than mutates this tensor at the next
                # frame. Clone the boundary so its exact graph remains explicit.
                propagated_states.append(output.propagated_queries.clone())
        finally:
            self.model.num_frames = original_num_frames
        per_frame = tuple(frame_outputs)
        merged = merge_exact_videomt_causal_outputs(per_frame)
        return ExactVidEoMTCausalSequenceOutput(
            merged=merged,
            per_frame=per_frame,
            propagated_queries_by_frame=tuple(propagated_states),
        )


def normalize_rgb_255(rgb_255: torch.Tensor) -> torch.Tensor:
    """Apply the released VidEoMT RGB normalization without guessing input scale."""

    if rgb_255.ndim != 4 or rgb_255.shape[1] != 3:
        raise ValueError("RGB input must have shape [time, 3, height, width]")
    if rgb_255.dtype == torch.uint8:
        rgb = rgb_255.to(torch.float32)
    elif rgb_255.is_floating_point():
        rgb = rgb_255
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("floating RGB must explicitly use the [0, 255] range")
    else:
        raise TypeError("RGB input must be uint8 or floating point")
    mean = rgb.new_tensor(VIDEOMT_PIXEL_MEAN_255).view(1, 3, 1, 1)
    std = rgb.new_tensor(VIDEOMT_PIXEL_STD_255).view(1, 3, 1, 1)
    return (rgb - mean) / std


def load_exact_videomt(
    config: ExactVidEoMTConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    assign: bool = True,
) -> ExactVidEoMTRuntime:
    """Instantiate the released graph and require a strict checkpoint load."""

    model = VidEoMT_CLASS(
        img_size=config.constructor_image_size,
        num_classes=config.num_classes,
        name=str(config.local_dinov3_bundle),
        num_frames=config.num_frames,
        num_q=config.num_queries,
        segmenter_blocks=config.segmenter_blocks,
        fused_qkv=config.fused_qkv,
        norm_queries=config.norm_queries,
    )
    state = published_videomt_backbone_state(config.checkpoint_path)
    incompatible = model.load_state_dict(state, strict=True, assign=assign)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict VidEoMT load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    adapted_receipt: AdaptedCheckpointReceipt | None = None
    if config.adapted_checkpoint_path is not None:
        if config.adapted_checkpoint_sha256 is None:
            raise RuntimeError("adapted VidEoMT checkpoint SHA-256 is absent")
        adapted_receipt, adapted_state = adapted_videomt_model_state(
            config.adapted_checkpoint_path,
            expected_sha256=config.adapted_checkpoint_sha256,
        )
        incompatible = model.load_state_dict(adapted_state, strict=True, assign=assign)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "strict adapted VidEoMT load returned incompatible keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    model.eval()
    model.to(device=torch.device(device), dtype=dtype)
    return ExactVidEoMTRuntime(
        config,
        model,
        adapted_checkpoint_receipt=adapted_receipt,
    )

"""Frozen VidEoMT query/patch refinement for LingBot posterior rows.

The module copies the released final DINOv3 blocks and prediction heads.  It
does not assign semantics or identities: LingBot posterior rows are additional
queries, while the complete released query bank and patch stream remain in the
attention context.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from picf_next._vendor.videomt.modeling.backbone.videomt import VidEoMT_CLASS
from picf_next.videomt_exact.runtime import (
    VIDEOMT_DINOV3_L_QUERIES,
    VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS,
    VIDEOMT_DINOV3_L_WIDTH,
)

PREPEND_PROJECTED_ROWS = "prepend_projected_rows"
REPLACE_WITH_RELEASED_PROPAGATION = "replace_with_released_propagation"
POSTERIOR_QUERY_INTEGRATIONS = (
    PREPEND_PROJECTED_ROWS,
    REPLACE_WITH_RELEASED_PROPAGATION,
)


@dataclass(frozen=True, slots=True)
class PosteriorRowRefinementOutput:
    """Final source-head prediction for posterior rows in frozen eval mode."""

    support_logits: torch.Tensor
    class_logits: torch.Tensor
    auxiliary_support_logits: tuple[torch.Tensor, ...] = ()
    auxiliary_class_logits: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if self.support_logits.ndim != 3:
            raise ValueError("refined support logits must be [batch,pixel,row]")
        batch, _pixels, rows = self.support_logits.shape
        if self.class_logits.shape != (batch, rows, 41):
            raise ValueError("refined class logits differ from the released 41-way head")
        if self.auxiliary_support_logits or self.auxiliary_class_logits:
            raise ValueError("frozen evaluation refinement must not emit training-only auxiliaries")
        values = (
            self.support_logits,
            self.class_logits,
            *self.auxiliary_support_logits,
            *self.auxiliary_class_logits,
        )
        if any(
            not value.is_floating_point()
            or not torch.isfinite(value).all()
            or value.device != self.support_logits.device
            for value in values
        ):
            raise ValueError("refinement outputs must be finite floating tensors on one device")


class FrozenVidEoMTPosteriorRowRefiner(nn.Module):
    """Run posterior rows through the released final query/patch block stack."""

    def __init__(
        self,
        source: VidEoMT_CLASS,
        *,
        query_integration: str = PREPEND_PROJECTED_ROWS,
    ) -> None:
        super().__init__()
        if not isinstance(source, VidEoMT_CLASS):
            raise TypeError("posterior refinement requires the released VidEoMT model")
        if not source.is_v3:
            raise ValueError("the selected posterior refinement requires released DINOv3")
        if tuple(source.segmenter_blocks) != VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS:
            raise ValueError("posterior refinement requires released blocks 20-23")
        if source.num_q != VIDEOMT_DINOV3_L_QUERIES or source.embed_dim != VIDEOMT_DINOV3_L_WIDTH:
            raise ValueError("posterior refinement requires the released query ABI")
        if query_integration not in POSTERIOR_QUERY_INTEGRATIONS:
            raise ValueError("unknown posterior-query integration")

        start = source.segmenter_blocks[0]
        stop = source.segmenter_blocks[-1] + 1
        self.blocks = nn.ModuleList(deepcopy(source.encoder.backbone.blocks[start:stop]))
        self.norm = deepcopy(source.encoder.backbone.norm)
        self.class_head = deepcopy(source.class_head)
        self.mask_head = deepcopy(source.mask_head)
        self.upscale = deepcopy(source.upscale)
        self.query_prior = deepcopy(source.q)
        self.query_updater = deepcopy(source.query_updater)
        self.source_query_count = int(source.num_q)
        self.prefix_token_count = int(source.encoder.backbone.num_prefix_tokens)
        self.norm_queries = bool(source.norm_queries)
        self.query_integration = query_integration
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenVidEoMTPosteriorRowRefiner:
        super().train(False)
        return self

    @staticmethod
    def _block_forward(
        x: torch.Tensor,
        block: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Byte-for-byte algebra of VidEoMT._block_forward's DINOv3 branch.
        attn_out = block.attention(
            block.norm1(x),
            attention_mask=None,
            position_embeddings=position_embeddings,
        )[0]
        x = x + block.drop_path(block.layer_scale1(attn_out))
        x = x + block.drop_path(block.layer_scale2(block.mlp(block.norm2(x))))
        return x

    def _predict(
        self,
        x: torch.Tensor,
        *,
        row_count: int,
        patch_grid_height: int,
        patch_grid_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x[:, :row_count]
        patch_start = self.source_query_count + self.prefix_token_count
        if self.query_integration == PREPEND_PROJECTED_ROWS:
            patch_start += row_count
        patches = x[:, patch_start:]
        expected_patches = patch_grid_height * patch_grid_width
        if patches.shape[1] != expected_patches:
            raise ValueError("refinement patch count differs from its declared grid")
        patches = patches.transpose(1, 2).reshape(
            patches.shape[0],
            VIDEOMT_DINOV3_L_WIDTH,
            patch_grid_height,
            patch_grid_width,
        )
        mask_logits = torch.einsum(
            "bkc,bchw->bkhw",
            self.mask_head(rows),
            self.upscale(patches),
        )
        support_logits = mask_logits.flatten(2).transpose(1, 2)
        return support_logits, self.class_head(rows)

    def _compose_source_input(
        self,
        decoded_rows: torch.Tensor,
        segmenter_input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.query_integration == PREPEND_PROJECTED_ROWS:
            return torch.cat((decoded_rows, segmenter_input_tokens), dim=1)
        row_count = decoded_rows.shape[1]
        if row_count > self.source_query_count:
            raise ValueError("posterior rows exceed the released source query bank")
        # Exact VidEoMT temporal query update: U(new_query) + learned query.
        # Replacing source rows preserves the released token count and RoPE ABI.
        conditioned_rows = (
            self.query_updater(decoded_rows) + self.query_prior.weight[None, :row_count]
        )
        return torch.cat((conditioned_rows, segmenter_input_tokens[:, row_count:]), dim=1)

    def _refine_and_predict(
        self,
        posterior_rows: torch.Tensor,
        semantic_projection_weight: torch.Tensor,
        segmenter_input_tokens: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        *,
        patch_grid_height: int,
        patch_grid_width: int,
        checkpoint_blocks: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded_rows = torch.matmul(posterior_rows, semantic_projection_weight)
        x = self._compose_source_input(decoded_rows, segmenter_input_tokens)
        position_embeddings = (position_cos, position_sin)
        for block in self.blocks:
            if checkpoint_blocks:

                def exact_source_block(
                    value: torch.Tensor,
                    cos: torch.Tensor,
                    sin: torch.Tensor,
                    *,
                    source_block: nn.Module = block,
                ) -> torch.Tensor:
                    return self._block_forward(value, source_block, (cos, sin))

                x = checkpoint(
                    exact_source_block,
                    x,
                    position_cos,
                    position_sin,
                    use_reentrant=False,
                    determinism_check="default",
                )
            else:
                x = self._block_forward(x, block, position_embeddings)
        return self._predict(
            self.norm(x),
            row_count=posterior_rows.shape[1],
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
        )

    def forward(
        self,
        *,
        posterior_rows: torch.Tensor,
        semantic_projection_weight: torch.Tensor,
        segmenter_input_tokens: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        patch_grid_height: int,
        patch_grid_width: int,
    ) -> PosteriorRowRefinementOutput:
        if posterior_rows.ndim != 3:
            raise ValueError("posterior rows must be [batch,row,host]")
        batch, row_count, host_width = posterior_rows.shape
        if semantic_projection_weight.shape != (host_width, VIDEOMT_DINOV3_L_WIDTH):
            raise ValueError("semantic projection differs from the tied donor width")
        expected_tokens = (
            self.source_query_count + self.prefix_token_count + patch_grid_height * patch_grid_width
        )
        if segmenter_input_tokens.shape != (batch, expected_tokens, VIDEOMT_DINOV3_L_WIDTH):
            raise ValueError("captured segmenter input has invalid axes")
        expected_rope = (patch_grid_height * patch_grid_width, 64)
        if position_cos.shape[-2:] != expected_rope or position_sin.shape != position_cos.shape:
            raise ValueError("captured DINOv3 RoPE has invalid axes")
        values = (
            posterior_rows,
            semantic_projection_weight,
            segmenter_input_tokens,
            position_cos,
            position_sin,
        )
        parameter = next(self.parameters())
        if any(value.device != parameter.device for value in values):
            raise ValueError("refinement inputs and source weights must share one device")
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("refinement inputs contain NaN or infinity")

        differentiable = torch.is_grad_enabled() and (
            posterior_rows.requires_grad or semantic_projection_weight.requires_grad
        )
        # Released transformer implementations checkpoint at block boundaries.
        # Keeping the same boundary here avoids simultaneously rematerializing
        # all four large query/patch attention graphs during backward.
        support, classes = self._refine_and_predict(
            posterior_rows,
            semantic_projection_weight,
            segmenter_input_tokens,
            position_cos,
            position_sin,
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
            checkpoint_blocks=differentiable,
        )
        return PosteriorRowRefinementOutput(
            support_logits=support,
            class_logits=classes,
        )

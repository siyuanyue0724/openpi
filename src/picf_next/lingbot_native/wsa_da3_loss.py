from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

WSA_LARGE_MODEL_SHA256 = "40aba8c2712002c576adc11bdea228c7bf4bff8f33f12ea06f1df94ec8a009a7"
WSA_LIBERO_CONFIG_SHA256 = "1e1b40785124c64dce987baaa4566d236d083f4557492d109cc8571b35cb01c8"
WSA_DA3_TEACHER_LAYERS = (11, 15, 19, 23)
WSA_FUTURE_QUERY_LAYERS = (17, 23, 29, 35)
WSA_DA3_LAYER_WEIGHTS = (1.0, 1.2, 1.4, 1.6)
WSA_DA3_NUM_VIEWS = 2
WSA_DA3_TOKENS_PER_VIEW = 1296
WSA_DA3_QUERY_DIM = 2048


@dataclass(frozen=True)
class WSADA3TeacherTargets:
    layers: tuple[torch.Tensor, ...]
    view_valid: torch.Tensor

    def validate(self) -> None:
        if len(self.layers) != len(WSA_DA3_TEACHER_LAYERS):
            raise ValueError("WSA requires all four released DA3 teacher layers")
        if self.view_valid.ndim != 2 or self.view_valid.shape[1] != WSA_DA3_NUM_VIEWS:
            raise ValueError("WSA DA3 view-valid mask must have shape [B,2]")
        if self.view_valid.dtype is not torch.bool:
            raise TypeError("WSA DA3 view-valid mask must be boolean")
        expected = (
            self.view_valid.shape[0],
            WSA_DA3_NUM_VIEWS * WSA_DA3_TOKENS_PER_VIEW,
            WSA_DA3_QUERY_DIM,
        )
        for index, layer in enumerate(self.layers):
            if tuple(layer.shape) != expected:
                raise ValueError(
                    f"WSA DA3 teacher layer {index} must have shape {expected}, "
                    f"got {tuple(layer.shape)}"
                )


def build_wsa_da3_token_mask(
    view_valid: torch.Tensor,
    *,
    target_length: int,
) -> torch.Tensor:
    if view_valid.ndim != 2 or view_valid.shape[1] != WSA_DA3_NUM_VIEWS:
        raise ValueError("WSA DA3 view-valid mask must have shape [B,2]")
    token_mask = view_valid.unsqueeze(-1).expand(
        -1,
        -1,
        WSA_DA3_TOKENS_PER_VIEW,
    ).reshape(view_valid.shape[0], -1)
    if token_mask.shape[1] == target_length:
        return token_mask
    token_mask = token_mask[:, None, :].to(dtype=torch.float32)
    token_mask = F.interpolate(token_mask, size=target_length, mode="nearest")
    return token_mask[:, 0, :].to(dtype=torch.bool)


def compute_official_wsa_da3_loss(
    predicted_queries: tuple[torch.Tensor, ...],
    targets: WSADA3TeacherTargets,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """WSA-Large `_compute_future_3d_loss`, with a frozen teacher cache."""
    targets.validate()
    if len(predicted_queries) != len(WSA_FUTURE_QUERY_LAYERS):
        raise ValueError("WSA requires all four future-query readouts")
    token_mask = build_wsa_da3_token_mask(
        targets.view_valid,
        target_length=targets.layers[0].shape[1],
    )
    total_loss = predicted_queries[0].new_zeros((), dtype=torch.float32)
    logs: dict[str, torch.Tensor] = {}
    for pred, target, weight, teacher_layer, query_layer in zip(
        predicted_queries,
        targets.layers,
        WSA_DA3_LAYER_WEIGHTS,
        WSA_DA3_TEACHER_LAYERS,
        WSA_FUTURE_QUERY_LAYERS,
        strict=True,
    ):
        if pred.shape != target.shape:
            raise ValueError(
                f"WSA projected/teacher shape mismatch: {tuple(pred.shape)} != "
                f"{tuple(target.shape)}"
            )
        pred_valid = pred[token_mask]
        target_valid = target.to(device=pred.device, dtype=pred.dtype)[token_mask]
        if pred_valid.numel() == 0:
            continue
        pred_norm = F.normalize(pred_valid.float(), p=2, dim=-1)
        target_norm = F.normalize(target_valid.detach().float(), p=2, dim=-1)
        cosine_loss = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()
        pred_ln = F.layer_norm(pred_valid.float(), normalized_shape=(pred_valid.shape[-1],))
        target_ln = F.layer_norm(
            target_valid.detach().float(),
            normalized_shape=(target_valid.shape[-1],),
        )
        mse_loss = F.mse_loss(pred_ln, target_ln)
        layer_loss = (cosine_loss + mse_loss) * float(weight)
        total_loss = total_loss + layer_loss
        logs[f"loss_3d_q{query_layer}_t{teacher_layer}"] = layer_loss.detach()
    total_loss = total_loss / len(predicted_queries)
    return total_loss, logs

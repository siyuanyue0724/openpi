from __future__ import annotations

import torch


def replace_oov_image_tokens(input_ids: torch.Tensor, *, image_token_id: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image_token_mask = input_ids == int(image_token_id)
    if int(image_token_id) < int(vocab_size):
        return input_ids, image_token_mask
    return torch.where(image_token_mask, torch.zeros_like(input_ids), input_ids), image_token_mask


def merge_image_features_dense(
    *,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    image_token_mask = input_ids == int(image_token_id)
    image_tokens_per_sample = image_token_mask.sum(dim=1)
    expected = int(image_features.shape[1])
    if not torch.all(image_tokens_per_sample == expected):
        raise ValueError(
            "Number of images does not match number of special image tokens in the input text. "
            f"Got per-sample counts={image_tokens_per_sample.tolist()} but expected {expected} tokens."
        )
    gather_index = image_token_mask.to(dtype=torch.long).cumsum(dim=1) - 1
    gather_index = torch.clamp(gather_index, min=0, max=max(expected - 1, 0))
    gather_index = gather_index.unsqueeze(-1).expand(-1, -1, image_features.shape[-1])
    dense_image_features = torch.gather(image_features, dim=1, index=gather_index)
    return torch.where(image_token_mask.unsqueeze(-1), dense_image_features, inputs_embeds)

# ruff: noqa

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        if m.numel() > 0:
            min_index = int(m.min().item())
            max_index = int(m.max().item())
            if min_index < 0 or max_index >= int(x.size(1)):
                raise RuntimeError(
                    "V-JEPA mask index out of bounds: "
                    f"valid=[0,{int(x.size(1)) - 1}] got min={min_index} max={max_index} "
                    f"for x.shape={tuple(x.shape)} mask.shape={tuple(m.shape)}"
                )
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)

from __future__ import annotations

import inspect

import pytest
import torch

from picf_next.lingbot_native.source_mask import (
    QwenWholeViewOmission,
    apply_qwen_packed_patch_mask,
    qwen_mask_query_addresses,
    qwen_patch_merger_dependency_map,
    qwen_source_masked_model_inputs,
    qwen_whole_view_omitted_model_inputs,
    sample_qwen_packed_patch_mask,
    sample_qwen_whole_view_omission,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.arange(2 * 2 * 16 * 3, dtype=torch.float32).reshape(2, 2, 16, 3)
    valid = torch.tensor([[True, False], [True, True]])
    grid = torch.tensor([[[1, 4, 4], [1, 4, 4]], [[1, 4, 4], [1, 4, 4]]])
    return images, valid, grid


def test_qwen_dependency_matches_official_consecutive_spatial_merger_groups() -> None:
    dependency = qwen_patch_merger_dependency_map(
        raw_patch_count=16,
        spatial_merge_size=2,
    )
    expected = torch.zeros(4, 16, dtype=torch.bool)
    for index in range(4):
        expected[index, index * 4 : (index + 1) * 4] = True
    assert torch.equal(dependency.output_depends_on_raw, expected)


def test_qwen_source_mask_is_deterministic_label_free_and_missing_factor_clean() -> None:
    images, valid, grid = _inputs()
    first = sample_qwen_packed_patch_mask(
        images=images,
        image_valid=valid,
        image_grid_thw=grid,
        spatial_merge_size=2,
        probability=0.5,
        seed=91,
        eligible_view_indices=(0,),
    )
    second = sample_qwen_packed_patch_mask(
        images=images,
        image_valid=valid,
        image_grid_thw=grid,
        spatial_merge_size=2,
        probability=0.5,
        seed=91,
        eligible_view_indices=(0,),
    )
    assert first.digest == second.digest
    assert torch.equal(first.merged_target_mask, second.merged_target_mask)
    assert torch.equal(
        first.raw_target_mask,
        first.merged_target_mask.repeat_interleave(4, dim=-1),
    )
    assert not first.merged_target_mask[~valid].any()
    assert first.query_count == 2
    assert torch.equal(first.query_valid, valid[:, :1].expand(-1, 2))
    assert first.merged_target_mask[0, 0].sum() == 2
    assert first.merged_target_mask[1, 0].sum() == 2
    assert not first.merged_target_mask[:, 1].any()
    addresses = qwen_mask_query_addresses(first, dtype=torch.float32)
    assert addresses.shape == (2, 2, 2)
    assert ((addresses > -1) & (addresses < 1)).all()
    assert set(inspect.signature(sample_qwen_packed_patch_mask).parameters) == {
        "images",
        "image_valid",
        "image_grid_thw",
        "spatial_merge_size",
        "probability",
        "seed",
        "eligible_view_indices",
    }


def test_qwen_source_branch_changes_only_selected_raw_patches_and_no_other_field() -> None:
    images, valid, grid = _inputs()
    plan = sample_qwen_packed_patch_mask(
        images=images,
        image_valid=valid,
        image_grid_thw=grid,
        spatial_merge_size=2,
        probability=1.0,
        seed=3,
        eligible_view_indices=(0,),
    )
    masked = apply_qwen_packed_patch_mask(images, plan, replacement=-7.0)
    assert (masked[plan.raw_target_mask] == -7).all()
    assert torch.equal(masked[~plan.raw_target_mask], images[~plan.raw_target_mask])
    state = torch.randn(2, 5)
    model_inputs = {"images": images, "state": state}
    branch = qwen_source_masked_model_inputs(model_inputs, plan)
    assert branch["state"] is state
    assert branch["images"] is not images


def test_qwen_source_mask_rejects_grid_or_raw_order_mismatch() -> None:
    images, valid, grid = _inputs()
    with pytest.raises(ValueError, match="patch axis"):
        sample_qwen_packed_patch_mask(
            images=images[:, :, :-1],
            image_valid=valid,
            image_grid_thw=grid,
            spatial_merge_size=2,
            probability=0.5,
            seed=1,
            eligible_view_indices=(0,),
        )
    with pytest.raises(ValueError, match="divisible"):
        qwen_patch_merger_dependency_map(raw_patch_count=15, spatial_merge_size=2)


def test_qwen_whole_view_omission_is_deterministic_and_content_independent() -> None:
    images, _valid, grid = _inputs()
    valid = torch.ones(2, 2, dtype=torch.bool)
    first = sample_qwen_whole_view_omission(
        images=images,
        image_valid=valid,
        image_grid_thw=grid,
        seed=17,
        eligible_view_indices=(0, 1),
    )
    second = sample_qwen_whole_view_omission(
        images=images.flip(-1),
        image_valid=valid,
        image_grid_thw=grid,
        seed=17,
        eligible_view_indices=(0, 1),
    )

    assert first.digest == second.digest
    assert first.omitted_view_index == second.omitted_view_index
    assert torch.equal(first.source_valid, torch.ones(2, dtype=torch.bool))
    assert not first.source_image_valid[:, first.omitted_view_index].any()
    assert set(inspect.signature(sample_qwen_whole_view_omission).parameters) == {
        "images",
        "image_valid",
        "image_grid_thw",
        "seed",
        "eligible_view_indices",
    }


def test_qwen_whole_view_omission_uses_only_official_missing_view_fields() -> None:
    images, _valid, grid = _inputs()
    valid = torch.ones(2, 2, dtype=torch.bool)
    plan = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=grid,
        image_valid=valid,
        seed=5,
    )
    state = torch.randn(2, 5)
    language = torch.ones(2, 3, dtype=torch.long)
    model_inputs = {
        "images": images,
        "img_masks": valid,
        "image_grid_thw": grid,
        "state": state,
        "lang_tokens": language,
    }
    omitted = qwen_whole_view_omitted_model_inputs(model_inputs, plan)

    assert omitted["images"] is not images
    assert (omitted["images"][:, 0] == -1).all()
    assert torch.equal(omitted["images"][:, 1], images[:, 1])
    assert torch.equal(omitted["img_masks"], torch.tensor([[False, True], [False, True]]))
    assert omitted["image_grid_thw"] is grid
    assert omitted["state"] is state
    assert omitted["lang_tokens"] is language


def test_qwen_whole_view_omission_rejects_the_only_available_source_view() -> None:
    images, _valid, grid = _inputs()
    valid = torch.tensor([[True, False], [True, True]])
    with pytest.raises(ValueError, match="no eligible"):
        sample_qwen_whole_view_omission(
            images=images,
            image_valid=valid,
            image_grid_thw=grid,
            seed=9,
            eligible_view_indices=(0,),
        )

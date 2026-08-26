"""Lightweight, prediction-free MolmoAct2 processor-layout contracts."""

from __future__ import annotations

from dataclasses import dataclass

MOLMO_VISION_PATCH_MODALITY = "molmo_vision_patch"


@dataclass(frozen=True, slots=True)
class MolmoAct2ImagePatchSpan:
    """One processor image's exact interval in a dense vision-token row."""

    image_key: str
    start: int
    stop: int
    image_num_crops: int
    patches_per_crop: int
    image_grid: tuple[int, int, int, int]
    image_token_pooling: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.image_key, str) or not self.image_key:
            raise ValueError("MolmoAct2 image patch span requires a nonempty image key")
        integers = (self.start, self.stop, self.image_num_crops, self.patches_per_crop)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
            raise TypeError("MolmoAct2 image patch span dimensions must be integers")
        if (
            self.start < 0
            or self.stop <= self.start
            or self.image_num_crops <= 0
            or self.patches_per_crop <= 0
            or self.stop - self.start != self.image_num_crops * self.patches_per_crop
        ):
            raise ValueError("MolmoAct2 image patch span has inconsistent dimensions")
        if (
            len(self.image_grid) != 4
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in self.image_grid
            )
            or min(self.image_grid) < 0
            or self.image_grid[0] <= 0
            or self.image_grid[1] <= 0
        ):
            raise ValueError("MolmoAct2 image grid is invalid")
        pooled_count = self.image_grid[0] * self.image_grid[1] + (
            self.image_grid[2] * self.image_grid[3]
        )
        if len(self.image_token_pooling) != pooled_count or not self.image_token_pooling:
            raise ValueError("MolmoAct2 pooling rows disagree with the image grid")
        support_width = len(self.image_token_pooling[0])
        if support_width <= 0 or any(
            len(row) != support_width
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < -1
                or value >= self.stop - self.start
                for value in row
            )
            for row in self.image_token_pooling
        ):
            raise ValueError("MolmoAct2 image pooling support is invalid")


@dataclass(frozen=True, slots=True)
class MolmoAct2VisionPatchLayout:
    """Prediction-free camera/patch layout for each dense-bank batch row."""

    rows: tuple[tuple[MolmoAct2ImagePatchSpan, ...], ...]
    tokens_per_row: int
    semantic_image_keys: bool
    contract: str = "molmoact2.resize-dense-patch-layout.v1"

    def __post_init__(self) -> None:
        if not self.rows or any(not row for row in self.rows):
            raise ValueError("MolmoAct2 vision layout requires images in every batch row")
        if (
            not isinstance(self.tokens_per_row, int)
            or isinstance(self.tokens_per_row, bool)
            or self.tokens_per_row <= 0
        ):
            raise ValueError("MolmoAct2 vision layout token count must be positive")
        if not isinstance(self.semantic_image_keys, bool):
            raise TypeError("MolmoAct2 semantic-image-key flag must be boolean")
        if self.contract != "molmoact2.resize-dense-patch-layout.v1":
            raise ValueError("unsupported MolmoAct2 vision patch layout contract")
        expected_keys = tuple(span.image_key for span in self.rows[0])
        if len(set(expected_keys)) != len(expected_keys):
            raise ValueError("MolmoAct2 image keys must be unique")
        for row in self.rows:
            if tuple(span.image_key for span in row) != expected_keys:
                raise ValueError("MolmoAct2 camera ordering must be identical across the batch")
            cursor = 0
            for span in row:
                if span.start != cursor:
                    raise ValueError("MolmoAct2 image patch spans must be contiguous")
                cursor = span.stop
            if cursor > self.tokens_per_row:
                raise ValueError("MolmoAct2 image patch spans exceed the dense bank")

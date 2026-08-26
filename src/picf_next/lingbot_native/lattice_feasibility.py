"""Fail-closed metrics for the released-weight native-lattice feasibility test."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, cast

from picf_next.contracts import ContractError

LATTICE_FEASIBILITY_SCHEMA = "picf-next.lingbot-native-lattice-feasibility.v1"
LATTICE_BASELINE = 8
LATTICE_CANDIDATE = 12
LATTICE_REQUIRED_SAMPLE_COUNT = 12
LATTICE_MINIMUM_ELIGIBLE_SAMPLES = 10
LATTICE_MINIMUM_PURITY_RATIO = 1.15
LATTICE_MINIMUM_SELF_IOU_RATIO = 1.10
LATTICE_MINIMUM_MEAN_AUC_DELTA = 0.02
LATTICE_MAXIMUM_MEAN_ACTION_DELTA = 0.01
LATTICE_MAXIMUM_MEDIAN_ACTION_DELTA = 0.005
LATTICE_MAXIMUM_INFERENCE_ALLOCATED_BYTES = 18 * 1024**3
LATTICE_SELECTION_SEED = "20260729"
LATTICE_TASK_KEYS = (
    "lift_pink_block_drawer",
    "lift_blue_block_table",
    "push_red_block_left",
    "rotate_blue_block_right",
    "turn_on_led",
    "turn_off_led",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "open_drawer",
    "close_drawer",
    "move_slider_left",
    "move_slider_right",
)
LATTICE_VISUAL_GRID_CACHE_FIELDS = (
    "pos_embeds",
    "position_embeddings",
    "cu_seqlens",
    "visual_split_sizes",
    "visual_max_seqlen",
)
LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS = (
    "position_embeddings",
    "cu_seqlens",
    "visual_split_sizes",
    "visual_max_seqlen",
)
LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS = ("pos_embeds",)


def native_lattice_shortest_edge(
    lattice: int,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
) -> int:
    """Return Qwen's area-valued shortest-edge setting for one merged lattice."""

    for name, value in (
        ("lattice", lattice),
        ("patch_size", patch_size),
        ("merge_size", merge_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    pixels = lattice * patch_size * merge_size
    return pixels * pixels


def configure_native_processor_lattice(
    processor: object,
    lattice: int,
) -> dict[str, object]:
    """Set Qwen's official dynamic-resolution area for one merged lattice."""

    image_processor = cast(Any, getattr(processor, "image_processor", None))
    size = getattr(image_processor, "size", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not isinstance(size, Mapping):
        raise RuntimeError("Qwen processor has no dynamic-resolution mapping")
    if patch_size != 16 or merge_size != 2:
        raise RuntimeError("Qwen processor patch/merge geometry differs from 16x16/2x2")
    longest_edge = size.get("longest_edge")
    if isinstance(longest_edge, bool) or not isinstance(longest_edge, int) or longest_edge <= 0:
        raise RuntimeError("Qwen processor maximum image area is invalid")
    shortest_edge = native_lattice_shortest_edge(
        lattice,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    image_processor.size = {
        "shortest_edge": shortest_edge,
        "longest_edge": longest_edge,
    }
    expected = {
        "shortest_edge": shortest_edge,
        "longest_edge": longest_edge,
    }
    if dict(image_processor.size) != expected:
        raise RuntimeError("Qwen processor rejected the requested native lattice")
    return {
        "lattice": lattice,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "pixels_per_edge": lattice * patch_size * merge_size,
        "shortest_edge_area": shortest_edge,
        "longest_edge_area": longest_edge,
    }


def native_processor_area_budget_contract(
    lattice: int,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
) -> dict[str, object]:
    """Return the canonical official-Qwen area and merged-token budget."""

    image_area = native_lattice_shortest_edge(
        lattice,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    return {
        "lattice": lattice,
        "mode": "official_qwen_aspect_ratio_preserving_area_budget",
        "patch_size": patch_size,
        "merge_size": merge_size,
        "target_image_area": image_area,
        "maximum_raw_patch_tokens": lattice * lattice * merge_size * merge_size,
        "maximum_merged_visual_tokens": lattice * lattice,
    }


def native_processor_expected_grid(
    *,
    image_height: int,
    image_width: int,
    lattice: int,
    patch_size: int = 16,
    merge_size: int = 2,
) -> list[list[int]]:
    """Reproduce official Qwen smart-resize geometry for one still image."""

    for name, value in (
        ("image_height", image_height),
        ("image_width", image_width),
        ("lattice", lattice),
        ("patch_size", patch_size),
        ("merge_size", merge_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max(image_height, image_width) / min(image_height, image_width) > 200:
        raise ValueError("Qwen image aspect ratio must not exceed 200")

    factor = patch_size * merge_size
    target_area = native_lattice_shortest_edge(
        lattice,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    resized_height = round(image_height / factor) * factor
    resized_width = round(image_width / factor) * factor
    if resized_height * resized_width > target_area:
        beta = math.sqrt((image_height * image_width) / target_area)
        resized_height = max(
            factor,
            math.floor(image_height / beta / factor) * factor,
        )
        resized_width = max(
            factor,
            math.floor(image_width / beta / factor) * factor,
        )
    elif resized_height * resized_width < target_area:
        beta = math.sqrt(target_area / (image_height * image_width))
        resized_height = math.ceil(image_height * beta / factor) * factor
        resized_width = math.ceil(image_width * beta / factor) * factor
    return [[1, resized_height // patch_size, resized_width // patch_size]]


def configure_native_processor_area_budget(
    processor: object,
    lattice: int,
) -> dict[str, object]:
    """Bound official Qwen smart-resize while preserving each image's aspect ratio."""

    image_processor = cast(Any, getattr(processor, "image_processor", None))
    size = getattr(image_processor, "size", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not isinstance(size, Mapping):
        raise RuntimeError("Qwen processor has no dynamic-resolution mapping")
    if patch_size != 16 or merge_size != 2:
        raise RuntimeError("Qwen processor patch/merge geometry differs from 16x16/2x2")
    if any(
        isinstance(size.get(name), bool) or not isinstance(size.get(name), int) or size[name] <= 0
        for name in ("shortest_edge", "longest_edge")
    ):
        raise RuntimeError("Qwen processor dynamic-resolution area is invalid")
    contract = native_processor_area_budget_contract(
        lattice,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    image_area = contract["target_image_area"]
    expected = {
        "shortest_edge": image_area,
        "longest_edge": image_area,
    }
    image_processor.size = expected
    if dict(image_processor.size) != expected:
        raise RuntimeError("Qwen processor rejected the requested image-area budget")
    return contract


def validate_native_processor_grid_budget(
    image_grid_thw: object,
    *,
    lattice: int,
    merge_size: int = 2,
) -> dict[str, object]:
    """Validate one official Qwen grid against a merged-token lattice budget."""

    for name, value in (("lattice", lattice), ("merge_size", merge_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(image_grid_thw, Sequence)
        or isinstance(image_grid_thw, str | bytes)
        or len(image_grid_thw) != 1
    ):
        raise RuntimeError("Qwen image grid budget requires exactly one image")
    raw_grid = image_grid_thw[0]
    if (
        not isinstance(raw_grid, Sequence)
        or isinstance(raw_grid, str | bytes)
        or len(raw_grid) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw_grid
        )
    ):
        raise RuntimeError("Qwen image grid budget received a malformed THW grid")
    temporal, height, width = raw_grid
    if temporal != 1:
        raise RuntimeError("Qwen image grid budget supports exactly one temporal patch")
    if height % merge_size or width % merge_size:
        raise RuntimeError("Qwen image grid is not divisible by its spatial merge size")
    raw_patch_tokens = temporal * height * width
    merged_visual_tokens = temporal * (height // merge_size) * (width // merge_size)
    maximum_raw_patch_tokens = lattice * lattice * merge_size * merge_size
    maximum_merged_visual_tokens = lattice * lattice
    if (
        raw_patch_tokens > maximum_raw_patch_tokens
        or merged_visual_tokens > maximum_merged_visual_tokens
    ):
        raise RuntimeError("Qwen image grid exceeds its declared visual-token budget")
    return {
        "image_grid_thw": [[temporal, height, width]],
        "raw_patch_tokens": raw_patch_tokens,
        "merged_visual_tokens": merged_visual_tokens,
        "maximum_raw_patch_tokens": maximum_raw_patch_tokens,
        "maximum_merged_visual_tokens": maximum_merged_visual_tokens,
    }


def validate_native_processor_record_grid(
    image_grid_thw: object,
    *,
    image_height: int,
    image_width: int,
    lattice: int,
    merge_size: int = 2,
) -> dict[str, object]:
    """Bind an official Qwen grid to one source image and its token budget."""

    budget = validate_native_processor_grid_budget(
        image_grid_thw,
        lattice=lattice,
        merge_size=merge_size,
    )
    expected_grid = native_processor_expected_grid(
        image_height=image_height,
        image_width=image_width,
        lattice=lattice,
        merge_size=merge_size,
    )
    if budget["image_grid_thw"] != expected_grid:
        raise RuntimeError("Qwen image grid differs from official smart-resize geometry")
    return budget


def reset_native_visual_grid_cache(host: object) -> dict[str, object]:
    """Reset LingBot's official shape-dependent Qwen visual cache."""

    config = getattr(host, "config", None)
    if getattr(config, "precompute_grid_thw", None) is not True:
        raise RuntimeError("native lattice requires LingBot's official grid precompute path")
    missing = [name for name in LATTICE_VISUAL_GRID_CACHE_FIELDS if not hasattr(host, name)]
    if missing:
        raise RuntimeError(f"LingBot visual-grid cache fields are absent: {missing}")
    nonempty_before = [
        name for name in LATTICE_VISUAL_GRID_CACHE_FIELDS if getattr(host, name) is not None
    ]
    for name in LATTICE_VISUAL_GRID_CACHE_FIELDS:
        setattr(host, name, None)
    if not all(getattr(host, name) is None for name in LATTICE_VISUAL_GRID_CACHE_FIELDS):
        raise RuntimeError("LingBot visual-grid cache reset was incomplete")
    return {
        "precompute_grid_thw": True,
        "fields": list(LATTICE_VISUAL_GRID_CACHE_FIELDS),
        "none_by_design": list(LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS),
        "nonempty_before": nonempty_before,
        "all_none_after": True,
    }


def require_native_visual_grid_cache_populated(host: object) -> list[str]:
    """Require the exact official Qwen visual-cache population contract."""

    populated = [
        name for name in LATTICE_VISUAL_GRID_CACHE_FIELDS if getattr(host, name, None) is not None
    ]
    none_by_design = [
        name for name in LATTICE_VISUAL_GRID_CACHE_FIELDS if getattr(host, name, None) is None
    ]
    if populated != list(LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS) or none_by_design != list(
        LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS
    ):
        raise RuntimeError(
            "LingBot visual-grid cache differs from the official Qwen3-VL return contract: "
            f"populated={populated}, none={none_by_design}"
        )
    return populated


def select_lattice_segment_indices(segments: Sequence[object]) -> tuple[int, ...]:
    """Select the frozen transition-zero task bank without reading observations."""

    by_task: dict[str, list[int]] = {task: [] for task in LATTICE_TASK_KEYS}
    for segment in segments:
        task_key = getattr(segment, "task_key", None)
        index = getattr(segment, "index", None)
        if task_key not in by_task:
            continue
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ContractError("CALVIN segment index is invalid")
        by_task[task_key].append(index)
    missing = sorted(task for task, indices in by_task.items() if not indices)
    if missing:
        raise ContractError(f"lattice task bank is absent from CALVIN: {missing}")
    selected = []
    for task_key in LATTICE_TASK_KEYS:
        selected.append(
            min(
                by_task[task_key],
                key=lambda index: (
                    hashlib.sha256(
                        f"{LATTICE_SELECTION_SEED}\0{task_key}\0{index}".encode()
                    ).digest(),
                    index,
                ),
            )
        )
    if len(set(selected)) != LATTICE_REQUIRED_SAMPLE_COUNT:
        raise ContractError("lattice task bank selected duplicate CALVIN segments")
    return tuple(selected)


def _finite_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{name} must be one finite real value")
    return float(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_vector(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{name} must be one nonempty list")
    return tuple(_finite_float(item, name=f"{name} item") for item in value)


def fractional_token_metrics(
    scores: Sequence[float],
    target_mass: Sequence[float],
) -> dict[str, object]:
    """Measure aliasing and task-token ranking with fractional target occupancy."""

    if (
        not isinstance(scores, Sequence)
        or isinstance(scores, str | bytes)
        or not isinstance(target_mass, Sequence)
        or isinstance(target_mass, str | bytes)
    ):
        raise TypeError("fractional token metrics require two sequences")
    score_values = tuple(_finite_float(value, name="task-token score") for value in scores)
    mass_values = tuple(_finite_float(value, name="target token mass") for value in target_mass)
    if not score_values or len(score_values) != len(mass_values):
        raise ValueError("task-token scores and target masses must be equal and nonempty")
    if any(value < 0.0 or value > 1.0 for value in mass_values):
        raise ValueError("target token mass must lie in [0,1]")

    target_total = math.fsum(mass_values)
    target_square = math.fsum(value * value for value in mass_values)
    background = tuple(1.0 - value for value in mass_values)
    background_total = math.fsum(background)
    if target_total == 0.0:
        return {
            "eligible": False,
            "token_count": len(score_values),
            "target_mass_total": 0.0,
            "target_area_fraction": 0.0,
            "nonzero_target_tokens": 0,
            "peak_target_occupancy": 0.0,
            "purity": None,
            "self_soft_iou": None,
            "effective_support": None,
            "fractional_weighted_auc": None,
            "target_background_logit_margin": None,
            "top_ten_percent_target_recall": None,
        }
    if background_total <= 0.0 or target_square <= 0.0:
        raise ValueError("eligible target metrics require target and background support")

    weighted_pairs = 0.0
    pair_total = target_total * background_total
    for left_score, left_mass in zip(score_values, mass_values, strict=True):
        if left_mass == 0.0:
            continue
        for right_score, right_mass in zip(score_values, background, strict=True):
            weight = left_mass * right_mass
            if left_score > right_score:
                weighted_pairs += weight
            elif left_score == right_score:
                weighted_pairs += 0.5 * weight

    target_mean = (
        math.fsum(score * mass for score, mass in zip(score_values, mass_values, strict=True))
        / target_total
    )
    background_mean = (
        math.fsum(score * mass for score, mass in zip(score_values, background, strict=True))
        / background_total
    )
    top_count = max(1, math.ceil(0.1 * len(score_values)))
    top_indices = sorted(
        range(len(score_values)),
        key=lambda index: (-score_values[index], index),
    )[:top_count]
    top_recall = math.fsum(mass_values[index] for index in top_indices) / target_total
    self_union = 2.0 * target_total - target_square
    return {
        "eligible": True,
        "token_count": len(score_values),
        "target_mass_total": target_total,
        "target_area_fraction": target_total / len(score_values),
        "nonzero_target_tokens": sum(value > 0.0 for value in mass_values),
        "peak_target_occupancy": max(mass_values),
        "purity": target_square / target_total,
        "self_soft_iou": target_square / self_union,
        "effective_support": target_total * target_total / target_square,
        "fractional_weighted_auc": weighted_pairs / pair_total,
        "target_background_logit_margin": target_mean - background_mean,
        "top_ten_percent_target_recall": top_recall,
    }


def _sample_metrics(value: Mapping[str, Any], *, arm: str, index: int) -> dict[str, object]:
    scores = _finite_vector(
        value.get("dense_task_logits"),
        name=f"{arm} sample {index} dense task logits",
    )
    target_mass = _finite_vector(
        value.get("target_mass"),
        name=f"{arm} sample {index} target mass",
    )
    expected = fractional_token_metrics(scores, target_mass)
    if value.get("metrics") != expected:
        raise ContractError(f"{arm} sample {index} persisted metrics differ from raw evidence")
    if value.get("eligible") is not expected["eligible"]:
        raise ContractError(f"{arm} sample {index} eligibility differs from raw evidence")
    return expected


def _sample_identity(value: Mapping[str, Any], *, arm: str, index: int) -> tuple[object, ...]:
    sample_key = value.get("sample_key")
    task_key = value.get("task_key")
    task = value.get("task")
    source_global_index = value.get("source_global_index")
    segment_index = value.get("segment_index")
    transition_index = value.get("transition_index")
    augmentation_seed = value.get("augmentation_seed")
    flow_seed = value.get("flow_seed")
    identities = value.get("target_identity_keys")
    image_sha256 = value.get("image_sha256")
    if (
        not isinstance(sample_key, str)
        or not sample_key
        or not isinstance(task_key, str)
        or not task_key
        or not isinstance(task, str)
        or not task
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (
                source_global_index,
                segment_index,
                transition_index,
                augmentation_seed,
                flow_seed,
            )
        )
        or transition_index != 0
        or not isinstance(identities, list)
        or not identities
        or any(not isinstance(item, str) or not item for item in identities)
        or len(set(identities)) != len(identities)
        or not isinstance(image_sha256, Mapping)
        or not image_sha256
    ):
        raise ContractError(f"{arm} sample {index} identity is malformed")
    hashes = tuple(
        (str(name), _sha256(digest, name=f"{arm} sample {index} image digest"))
        for name, digest in sorted(image_sha256.items())
    )
    return (
        sample_key,
        task_key,
        task,
        source_global_index,
        segment_index,
        transition_index,
        augmentation_seed,
        flow_seed,
        tuple(identities),
        hashes,
    )


def _nonimage_shapes(value: Mapping[str, Any], *, arm: str, index: int) -> dict[str, object]:
    shapes = value.get("input_shapes")
    if not isinstance(shapes, Mapping):
        raise ContractError(f"{arm} sample {index} input shapes are absent")
    result = {
        str(name): shape
        for name, shape in sorted(shapes.items())
        if name not in {"images", "image_grid_thw"}
    }
    for name, shape in result.items():
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in shape
            )
        ):
            raise ContractError(f"{arm} sample {index} input shape {name} is invalid")
    return result


def _validate_grid(
    value: Mapping[str, Any],
    *,
    arm: str,
    index: int,
    lattice: int,
) -> bool:
    grids = value.get("image_grid_thw")
    image_valid = value.get("image_valid")
    if (
        not isinstance(grids, list)
        or len(grids) != 3
        or not isinstance(image_valid, list)
        or image_valid != [True, True, False]
    ):
        raise ContractError(f"{arm} sample {index} image-grid contract is malformed")
    expected = [1, lattice * 2, lattice * 2]
    if any(grid != expected for grid in grids):
        raise ContractError(f"{arm} sample {index} image grid differs from declared lattice")
    scores = value.get("dense_task_logits")
    return isinstance(scores, list) and len(scores) == 2 * lattice * lattice


def _validate_processor_contract(value: Mapping[str, Any], *, lattice: int) -> bool:
    processor = value.get("processor")
    if not isinstance(processor, Mapping):
        raise ContractError("lattice arm has no processor contract")
    longest_edge = processor.get("longest_edge_area")
    expected = {
        "lattice": lattice,
        "patch_size": 16,
        "merge_size": 2,
        "pixels_per_edge": lattice * 32,
        "shortest_edge_area": native_lattice_shortest_edge(lattice),
        "longest_edge_area": longest_edge,
    }
    return (
        isinstance(longest_edge, int)
        and not isinstance(longest_edge, bool)
        and longest_edge >= expected["shortest_edge_area"]
        and dict(processor) == expected
    )


def _validate_grid_cache_contract(
    value: Mapping[str, Any],
    *,
    candidate: bool,
) -> bool:
    cache = value.get("visual_grid_cache_invalidation")
    if not isinstance(cache, Mapping):
        raise ContractError("lattice arm has no visual-grid cache contract")
    expected_fields = list(LATTICE_VISUAL_GRID_CACHE_FIELDS)
    expected_populated = list(LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS)
    nonempty_before = cache.get("nonempty_before")
    populated_after_arm = cache.get("populated_after_arm")
    return bool(
        cache.get("precompute_grid_thw") is True
        and cache.get("fields") == expected_fields
        and cache.get("none_by_design") == list(LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS)
        and cache.get("all_none_after") is True
        and isinstance(nonempty_before, list)
        and all(name in expected_fields for name in nonempty_before)
        and len(set(nonempty_before)) == len(nonempty_before)
        and nonempty_before == (expected_populated if candidate else [])
        and populated_after_arm == expected_populated
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ContractError("lattice decision cannot aggregate an empty vector")
    return float(statistics.median(values))


def lattice_feasibility_decision(report: Mapping[str, Any]) -> dict[str, object]:
    """Recompute paired aggregates and all frozen lattice gates."""

    if not isinstance(report, Mapping):
        raise ContractError("lattice feasibility report must be a mapping")
    if report.get("schema") != LATTICE_FEASIBILITY_SCHEMA:
        raise ContractError("lattice feasibility report schema changed")
    if report.get("baseline_lattice") != LATTICE_BASELINE:
        raise ContractError("lattice feasibility baseline changed")
    if report.get("candidate_lattice") != LATTICE_CANDIDATE:
        raise ContractError("lattice feasibility candidate changed")
    arms = report.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"8", "12"}:
        raise ContractError("lattice feasibility report must contain exactly two arms")
    baseline = arms["8"]
    candidate = arms["12"]
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ContractError("lattice feasibility arms must be mappings")
    if baseline.get("lattice") != 8 or candidate.get("lattice") != 12:
        raise ContractError("lattice feasibility arm identities changed")
    processor_exact = _validate_processor_contract(
        baseline,
        lattice=LATTICE_BASELINE,
    ) and _validate_processor_contract(
        candidate,
        lattice=LATTICE_CANDIDATE,
    )
    cache_exact = _validate_grid_cache_contract(
        baseline,
        candidate=False,
    ) and _validate_grid_cache_contract(
        candidate,
        candidate=True,
    )
    baseline_samples = baseline.get("samples")
    candidate_samples = candidate.get("samples")
    if (
        not isinstance(baseline_samples, list)
        or not isinstance(candidate_samples, list)
        or len(baseline_samples) != LATTICE_REQUIRED_SAMPLE_COUNT
        or len(candidate_samples) != LATTICE_REQUIRED_SAMPLE_COUNT
    ):
        raise ContractError("lattice feasibility sample count changed")

    contract_identical = True
    geometry_exact = True
    all_finite = True
    purity_ratios: list[float] = []
    self_iou_ratios: list[float] = []
    auc_deltas: list[float] = []
    action_deltas: list[float] = []
    baseline_actions: list[float] = []
    candidate_actions: list[float] = []
    candidate_peak_allocated_values: list[int] = []
    for index, (raw_baseline, raw_candidate) in enumerate(
        zip(baseline_samples, candidate_samples, strict=True)
    ):
        if not isinstance(raw_baseline, Mapping) or not isinstance(raw_candidate, Mapping):
            raise ContractError("lattice feasibility samples must be mappings")
        contract_identical &= _sample_identity(
            raw_baseline,
            arm="8",
            index=index,
        ) == _sample_identity(raw_candidate, arm="12", index=index)
        contract_identical &= _nonimage_shapes(
            raw_baseline,
            arm="8",
            index=index,
        ) == _nonimage_shapes(raw_candidate, arm="12", index=index)
        geometry_exact &= _validate_grid(
            raw_baseline,
            arm="8",
            index=index,
            lattice=8,
        )
        geometry_exact &= _validate_grid(
            raw_candidate,
            arm="12",
            index=index,
            lattice=12,
        )
        baseline_metrics = _sample_metrics(raw_baseline, arm="8", index=index)
        candidate_metrics = _sample_metrics(raw_candidate, arm="12", index=index)
        baseline_action = _finite_float(
            raw_baseline.get("official_action_loss"),
            name=f"8 sample {index} action loss",
        )
        candidate_action = _finite_float(
            raw_candidate.get("official_action_loss"),
            name=f"12 sample {index} action loss",
        )
        baseline_actions.append(baseline_action)
        candidate_actions.append(candidate_action)
        action_deltas.append(candidate_action - baseline_action)
        for raw, arm in ((raw_baseline, "8"), (raw_candidate, "12")):
            _finite_float(raw.get("forward_seconds"), name=f"{arm} sample forward seconds")
            peak_allocated = _positive_int(
                raw.get("peak_cuda_allocated_bytes"),
                name=f"{arm} sample peak allocated bytes",
            )
            if arm == "12":
                candidate_peak_allocated_values.append(peak_allocated)
            _positive_int(
                raw.get("peak_cuda_reserved_bytes"),
                name=f"{arm} sample peak reserved bytes",
            )
        baseline_eligible = baseline_metrics.get("eligible")
        candidate_eligible = candidate_metrics.get("eligible")
        if not isinstance(baseline_eligible, bool) or not isinstance(candidate_eligible, bool):
            raise ContractError("lattice metric eligibility must be boolean")
        if baseline_eligible and candidate_eligible:
            baseline_purity = _finite_float(
                baseline_metrics.get("purity"),
                name="baseline lattice purity",
            )
            baseline_self_iou = _finite_float(
                baseline_metrics.get("self_soft_iou"),
                name="baseline lattice self soft IoU",
            )
            candidate_purity = _finite_float(
                candidate_metrics.get("purity"),
                name="candidate lattice purity",
            )
            candidate_self_iou = _finite_float(
                candidate_metrics.get("self_soft_iou"),
                name="candidate lattice self soft IoU",
            )
            baseline_auc = _finite_float(
                baseline_metrics.get("fractional_weighted_auc"),
                name="baseline lattice weighted AUC",
            )
            candidate_auc = _finite_float(
                candidate_metrics.get("fractional_weighted_auc"),
                name="candidate lattice weighted AUC",
            )
            purity_ratios.append(candidate_purity / baseline_purity)
            self_iou_ratios.append(candidate_self_iou / baseline_self_iou)
            auc_deltas.append(candidate_auc - baseline_auc)

    eligible_count = len(auc_deltas)
    mean_auc_delta = math.fsum(auc_deltas) / eligible_count if eligible_count else None
    improved_auc_count = sum(value >= 0.0 for value in auc_deltas)
    required_auc_improvements = math.ceil((2.0 / 3.0) * eligible_count)
    mean_action_delta = (
        math.fsum(candidate_actions) - math.fsum(baseline_actions)
    ) / LATTICE_REQUIRED_SAMPLE_COUNT
    median_action_delta = _median(action_deltas)
    candidate_peak_allocated = max(candidate_peak_allocated_values)
    aggregates = {
        "eligible_sample_count": eligible_count,
        "median_purity_ratio": _median(purity_ratios) if purity_ratios else None,
        "median_self_soft_iou_ratio": _median(self_iou_ratios) if self_iou_ratios else None,
        "mean_fractional_weighted_auc_delta": mean_auc_delta,
        "nonnegative_auc_delta_count": improved_auc_count,
        "required_nonnegative_auc_delta_count": required_auc_improvements,
        "mean_baseline_action_loss": math.fsum(baseline_actions) / LATTICE_REQUIRED_SAMPLE_COUNT,
        "mean_candidate_action_loss": math.fsum(candidate_actions) / LATTICE_REQUIRED_SAMPLE_COUNT,
        "mean_action_loss_delta": mean_action_delta,
        "median_action_loss_delta": median_action_delta,
        "candidate_peak_cuda_allocated_bytes": candidate_peak_allocated,
    }
    gates = {
        "paired_contract_identical": contract_identical,
        "native_grid_exact": geometry_exact and processor_exact and cache_exact,
        "all_outputs_finite": all_finite,
        "minimum_eligible_samples": eligible_count >= LATTICE_MINIMUM_ELIGIBLE_SAMPLES,
        "raster_purity_improves": bool(
            purity_ratios and _median(purity_ratios) >= LATTICE_MINIMUM_PURITY_RATIO
        ),
        "raster_self_iou_improves": bool(
            self_iou_ratios and _median(self_iou_ratios) >= LATTICE_MINIMUM_SELF_IOU_RATIO
        ),
        "task_token_auc_improves": bool(
            eligible_count >= LATTICE_MINIMUM_ELIGIBLE_SAMPLES
            and mean_auc_delta is not None
            and mean_auc_delta >= LATTICE_MINIMUM_MEAN_AUC_DELTA
            and improved_auc_count >= required_auc_improvements
        ),
        "released_action_path_preserved": bool(
            mean_action_delta <= LATTICE_MAXIMUM_MEAN_ACTION_DELTA
            and median_action_delta <= LATTICE_MAXIMUM_MEDIAN_ACTION_DELTA
        ),
        "inference_memory_within_envelope": (
            candidate_peak_allocated <= LATTICE_MAXIMUM_INFERENCE_ALLOCATED_BYTES
        ),
        "loss_only_supervision": bool(
            report.get("loss_only_supervision") is True
            and report.get("target_resolution_happened_after_forward") is True
            and report.get("optimizer_created") is False
            and report.get("checkpoint_mutated") is False
            and report.get("same_parameter_objects_across_arms") is True
            and report.get("target_or_mask_fields_in_model_inputs") == []
        ),
    }
    return {"aggregates": aggregates, "gates": gates}


def validate_lattice_feasibility_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate persisted lattice evidence and recompute the scientific decision."""

    decision = lattice_feasibility_decision(value)
    if value.get("aggregates") != decision["aggregates"]:
        raise ContractError("lattice persisted aggregates differ from recomputation")
    if value.get("gates") != decision["gates"]:
        raise ContractError("lattice persisted gates differ from recomputation")
    gates = decision.get("gates")
    if not isinstance(gates, Mapping) or any(
        not isinstance(name, str) or not isinstance(passed, bool) for name, passed in gates.items()
    ):
        raise ContractError("lattice recomputation produced malformed gates")
    failures = sorted(name for name, passed in gates.items() if not passed)
    if value.get("failures") != failures:
        raise ContractError("lattice persisted failures differ from recomputation")
    status = "PASS" if not failures else "FAIL"
    if value.get("status") != status:
        raise ContractError("lattice persisted status differs from frozen gates")
    return dict(value)

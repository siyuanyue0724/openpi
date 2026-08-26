"""Strict contract for the MolmoAct2 current-frame representation gate.

M2 isolates the observation model:

    frozen native Molmo patches -> shared projection -> unordered discovery

The temporal filter, previous action, action expert and task text are outside
the trainable graph.  This file contains only immutable experiment parsing and
split validation; cloud execution remains in ``tools/run_molmoact2_m2_cloud.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from picf_next.training.recipe import PICFTrainingRecipe, load_training_recipe

M2_RECIPE_SCHEMA = "picf-next.molmoact2-m2-representation.v1"
M2_GATE = "M2_representation_smoke"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        missing = sorted(fields - set(payload))
        unknown = sorted(set(payload) - fields)
        raise ValueError(f"{name} fields differ; missing={missing}, unknown={unknown}")
    return payload


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return text


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be one finite number")
    converted = float(value)
    if not math.isfinite(converted) or (minimum is not None and converted < minimum):
        raise ValueError(f"{name} is outside its finite range")
    return converted


def _probability(value: object, name: str) -> float:
    converted = _number(value, name, minimum=0.0)
    if converted > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return converted


def _segments(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty segment list")
    result = tuple(_nonnegative_int(item, f"{name}[{index}]") for index, item in enumerate(value))
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class M2SplitRecipe:
    strategy: str
    train_segments: tuple[int, ...]
    validation_segments: tuple[int, ...]
    heldout_segments: tuple[int, ...]
    excluded_overlap_control_segments: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.strategy != "disjoint-language-segments-and-source-ranges.v1":
            raise ValueError("unsupported M2 split strategy")
        groups = (
            set(self.train_segments),
            set(self.validation_segments),
            set(self.heldout_segments),
            set(self.excluded_overlap_control_segments),
        )
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1 :]):
            raise ValueError("M2 segment groups must be disjoint")

    @property
    def learned_segments(self) -> tuple[int, ...]:
        return self.train_segments + self.validation_segments + self.heldout_segments

    def split_name(self, segment_index: int) -> str:
        groups = {
            "train": self.train_segments,
            "validation": self.validation_segments,
            "heldout": self.heldout_segments,
        }
        matches = tuple(name for name, values in groups.items() if segment_index in values)
        if len(matches) != 1:
            raise KeyError(f"segment {segment_index} is not in exactly one learned M2 split")
        return matches[0]


@dataclass(frozen=True, slots=True)
class M2FeatureCacheRecipe:
    modality: str
    token_count: int
    token_dim: int
    dtype: str
    extraction_batch_size: int
    shard_rows: int

    def __post_init__(self) -> None:
        if self.modality != "molmo_vision_patch":
            raise ValueError("M2 must use the audited native Molmo vision patch bank")
        _positive_int(self.token_count, "cache.token_count")
        _positive_int(self.token_dim, "cache.token_dim")
        if self.dtype != "bfloat16":
            raise ValueError("M2 frozen feature cache must retain native bfloat16 values")
        _positive_int(self.extraction_batch_size, "cache.extraction_batch_size")
        _positive_int(self.shard_rows, "cache.shard_rows")


@dataclass(frozen=True, slots=True)
class M2OptimizationRecipe:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    warmup_steps: int
    validation_interval: int
    seed: int

    def __post_init__(self) -> None:
        if not 20 <= self.steps <= 200:
            raise ValueError("M2 is restricted to 20-200 optimizer steps")
        _positive_int(self.batch_size, "optimization.batch_size")
        _number(self.learning_rate, "optimization.learning_rate", minimum=1e-12)
        _number(self.weight_decay, "optimization.weight_decay", minimum=0.0)
        _number(self.gradient_clip_norm, "optimization.gradient_clip_norm", minimum=1e-12)
        _nonnegative_int(self.warmup_steps, "optimization.warmup_steps")
        _positive_int(self.validation_interval, "optimization.validation_interval")
        _nonnegative_int(self.seed, "optimization.seed")
        if self.warmup_steps >= self.steps:
            raise ValueError("M2 warmup must end before the final optimizer step")
        if self.steps % self.validation_interval:
            raise ValueError("M2 validation interval must divide optimizer steps exactly")


@dataclass(frozen=True, slots=True)
class M2AcceptanceRecipe:
    minimum_count_mae_improvement_fraction_vs_random: float
    minimum_geometry_mae_improvement_fraction_vs_random: float
    minimum_heldout_exact_count_accuracy: float
    minimum_label_shuffle_dice_margin: float
    minimum_label_shuffle_ownership_accuracy_margin: float
    minimum_mean_object_dice: float
    minimum_ownership_accuracy: float
    minimum_ownership_accuracy_improvement_vs_all_context: float
    minimum_random_dice_margin: float
    minimum_uncertainty_error_spearman: float
    maximum_query_permutation_error: float
    maximum_task_intervention_feature_error: float

    def __post_init__(self) -> None:
        for name in (
            "minimum_count_mae_improvement_fraction_vs_random",
            "minimum_geometry_mae_improvement_fraction_vs_random",
            "minimum_heldout_exact_count_accuracy",
            "minimum_label_shuffle_dice_margin",
            "minimum_label_shuffle_ownership_accuracy_margin",
            "minimum_mean_object_dice",
            "minimum_ownership_accuracy",
            "minimum_ownership_accuracy_improvement_vs_all_context",
            "minimum_random_dice_margin",
        ):
            _probability(getattr(self, name), f"acceptance.{name}")
        if not -1.0 <= self.minimum_uncertainty_error_spearman <= 1.0:
            raise ValueError("minimum uncertainty Spearman must lie in [-1, 1]")
        _number(
            self.maximum_query_permutation_error,
            "acceptance.maximum_query_permutation_error",
            minimum=0.0,
        )
        _number(
            self.maximum_task_intervention_feature_error,
            "acceptance.maximum_task_intervention_feature_error",
            minimum=0.0,
        )


@dataclass(frozen=True, slots=True)
class MolmoAct2M2Recipe:
    foundation_recipe_path: str
    foundation_recipe_sha256: str
    splits: M2SplitRecipe
    cache: M2FeatureCacheRecipe
    optimization: M2OptimizationRecipe
    acceptance: M2AcceptanceRecipe
    schema: str = M2_RECIPE_SCHEMA
    gate: str = M2_GATE

    def __post_init__(self) -> None:
        if self.schema != M2_RECIPE_SCHEMA or self.gate != M2_GATE:
            raise ValueError("M2 schema or gate identity changed")
        _relative_path(self.foundation_recipe_path, "foundation_recipe_path")
        _sha256(self.foundation_recipe_sha256, "foundation_recipe_sha256")

    @property
    def recipe_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def load_foundation(self, repository_root: str | Path) -> PICFTrainingRecipe:
        root = Path(repository_root).resolve()
        path = (root / self.foundation_recipe_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError("M2 foundation recipe is absent or escaped the repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.foundation_recipe_sha256:
            raise ValueError("M2 foundation recipe SHA-256 changed")
        recipe = load_training_recipe(path)
        if recipe.host.name != "MolmoAct2":
            raise ValueError("M2 foundation host changed")
        if recipe.core_config.runtime_validation != "full":
            raise ValueError("M2 requires full tensor validation")
        if set(recipe.core_config.dense_token_dims) != {self.cache.modality}:
            raise ValueError("M2 foundation modality set changed")
        if recipe.core_config.dense_token_dims[self.cache.modality] != self.cache.token_dim:
            raise ValueError("M2 native feature width differs from the foundation")
        if recipe.core_config.discovery.num_queries <= 1:
            raise ValueError("M2 requires a nontrivial unordered query set")
        return recipe

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "gate": self.gate,
            "foundation_recipe_path": self.foundation_recipe_path,
            "foundation_recipe_sha256": self.foundation_recipe_sha256,
            "splits": {
                "strategy": self.splits.strategy,
                "train_segments": list(self.splits.train_segments),
                "validation_segments": list(self.splits.validation_segments),
                "heldout_segments": list(self.splits.heldout_segments),
                "excluded_overlap_control_segments": list(
                    self.splits.excluded_overlap_control_segments
                ),
            },
            "cache": {
                "modality": self.cache.modality,
                "token_count": self.cache.token_count,
                "token_dim": self.cache.token_dim,
                "dtype": self.cache.dtype,
                "extraction_batch_size": self.cache.extraction_batch_size,
                "shard_rows": self.cache.shard_rows,
            },
            "optimization": {
                "steps": self.optimization.steps,
                "batch_size": self.optimization.batch_size,
                "learning_rate": self.optimization.learning_rate,
                "weight_decay": self.optimization.weight_decay,
                "gradient_clip_norm": self.optimization.gradient_clip_norm,
                "warmup_steps": self.optimization.warmup_steps,
                "validation_interval": self.optimization.validation_interval,
                "seed": self.optimization.seed,
            },
            "acceptance": {
                name: getattr(self.acceptance, name)
                for name in self.acceptance.__dataclass_fields__
            },
        }


def load_molmoact2_m2_recipe(path: str | Path) -> MolmoAct2M2Recipe:
    source = Path(path)
    try:
        raw = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("M2 recipe cannot be read as JSON") from error
    payload = _exact(
        raw,
        "M2 recipe",
        {
            "schema",
            "gate",
            "foundation_recipe_path",
            "foundation_recipe_sha256",
            "splits",
            "cache",
            "optimization",
            "acceptance",
        },
    )
    split = _exact(
        payload["splits"],
        "splits",
        {
            "strategy",
            "train_segments",
            "validation_segments",
            "heldout_segments",
            "excluded_overlap_control_segments",
        },
    )
    cache = _exact(
        payload["cache"],
        "cache",
        {
            "modality",
            "token_count",
            "token_dim",
            "dtype",
            "extraction_batch_size",
            "shard_rows",
        },
    )
    optimization = _exact(
        payload["optimization"],
        "optimization",
        {
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "warmup_steps",
            "validation_interval",
            "seed",
        },
    )
    acceptance_names = {
        "minimum_count_mae_improvement_fraction_vs_random",
        "minimum_geometry_mae_improvement_fraction_vs_random",
        "minimum_heldout_exact_count_accuracy",
        "minimum_label_shuffle_dice_margin",
        "minimum_label_shuffle_ownership_accuracy_margin",
        "minimum_mean_object_dice",
        "minimum_ownership_accuracy",
        "minimum_ownership_accuracy_improvement_vs_all_context",
        "minimum_random_dice_margin",
        "minimum_uncertainty_error_spearman",
        "maximum_query_permutation_error",
        "maximum_task_intervention_feature_error",
    }
    acceptance = _exact(payload["acceptance"], "acceptance", acceptance_names)
    return MolmoAct2M2Recipe(
        schema=_text(payload["schema"], "schema"),
        gate=_text(payload["gate"], "gate"),
        foundation_recipe_path=_relative_path(
            payload["foundation_recipe_path"],
            "foundation_recipe_path",
        ),
        foundation_recipe_sha256=_sha256(
            payload["foundation_recipe_sha256"],
            "foundation_recipe_sha256",
        ),
        splits=M2SplitRecipe(
            strategy=_text(split["strategy"], "splits.strategy"),
            train_segments=_segments(split["train_segments"], "splits.train_segments"),
            validation_segments=_segments(
                split["validation_segments"],
                "splits.validation_segments",
            ),
            heldout_segments=_segments(split["heldout_segments"], "splits.heldout_segments"),
            excluded_overlap_control_segments=_segments(
                split["excluded_overlap_control_segments"],
                "splits.excluded_overlap_control_segments",
            ),
        ),
        cache=M2FeatureCacheRecipe(
            modality=_text(cache["modality"], "cache.modality"),
            token_count=_positive_int(cache["token_count"], "cache.token_count"),
            token_dim=_positive_int(cache["token_dim"], "cache.token_dim"),
            dtype=_text(cache["dtype"], "cache.dtype"),
            extraction_batch_size=_positive_int(
                cache["extraction_batch_size"],
                "cache.extraction_batch_size",
            ),
            shard_rows=_positive_int(cache["shard_rows"], "cache.shard_rows"),
        ),
        optimization=M2OptimizationRecipe(
            steps=_positive_int(optimization["steps"], "optimization.steps"),
            batch_size=_positive_int(optimization["batch_size"], "optimization.batch_size"),
            learning_rate=_number(
                optimization["learning_rate"],
                "optimization.learning_rate",
                minimum=1e-12,
            ),
            weight_decay=_number(
                optimization["weight_decay"],
                "optimization.weight_decay",
                minimum=0.0,
            ),
            gradient_clip_norm=_number(
                optimization["gradient_clip_norm"],
                "optimization.gradient_clip_norm",
                minimum=1e-12,
            ),
            warmup_steps=_nonnegative_int(
                optimization["warmup_steps"],
                "optimization.warmup_steps",
            ),
            validation_interval=_positive_int(
                optimization["validation_interval"],
                "optimization.validation_interval",
            ),
            seed=_nonnegative_int(optimization["seed"], "optimization.seed"),
        ),
        acceptance=M2AcceptanceRecipe(
            **{
                name: _number(
                    acceptance[name],
                    f"acceptance.{name}",
                )
                for name in acceptance_names
            }
        ),
    )


def m2_recipe_report(
    recipe: MolmoAct2M2Recipe,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    foundation = recipe.load_foundation(repository_root)
    return {
        "schema": M2_RECIPE_SCHEMA,
        "gate": M2_GATE,
        "recipe_sha256": recipe.recipe_sha256,
        "foundation_recipe_sha256": foundation.recipe_sha256,
        "foundation_authorization_stage": foundation.authorization.stage,
        "trainable_runtime_modules": ["projector", "discovery"],
        "forbidden_runtime_modules": [
            "action_expert",
            "action_adapter",
            "posterior_filter",
            "task_text",
            "previous_executed_action",
        ],
        "split_segments": {
            "train": list(recipe.splits.train_segments),
            "validation": list(recipe.splits.validation_segments),
            "heldout": list(recipe.splits.heldout_segments),
            "excluded_overlap_controls": list(recipe.splits.excluded_overlap_control_segments),
        },
        "optimizer_steps": recipe.optimization.steps,
        "long_training_authorized": False,
    }

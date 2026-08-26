"""Frozen Stage-B recipe for stationary temporal-core calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch

from picf_next.models.objective import PICFObjectiveConfig
from picf_next.training.molmoact2_m2_source_coverage import (
    MolmoAct2M2SourceCoverageRecipe,
    load_molmoact2_m2_source_coverage_recipe,
)
from picf_next.training.recipe import PICFTrainingRecipe
from picf_next.training.stationary_temporal import STATIONARY_TEMPORAL_EXECUTION_CONTRACT

STATIONARY_STAGE_SCHEMA = "picf-next.stationary-temporal-stage-b.v1"
STATIONARY_STAGE_NAME = "M3_stationary_temporal_calibration"
STATIONARY_STAGE_SCHEDULER = "linear-warmup-cosine-decay.v1"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        raise ValueError(f"{name} fields differ from the frozen Stage-B schema")
    return payload


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return text


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    output = float(value)
    if not math.isfinite(output) or output < 0.0 or (positive and output <= 0.0):
        raise ValueError(f"{name} has an invalid numeric value")
    return output


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{name} must be repository-relative")
    return text


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class StationaryStageClipRecipe:
    prefix_lengths: tuple[int, ...]
    train_length: int
    seed: int

    def __post_init__(self) -> None:
        if self.prefix_lengths != (0, 8, 32, 128):
            raise ValueError("Stage-B prefix lengths are preregistered as (0, 8, 32, 128)")
        if self.train_length != 2:
            raise ValueError("Stage-B train suffix is preregistered as two frames")
        _nonnegative_integer(self.seed, "Stage-B clip seed")


@dataclass(frozen=True, slots=True)
class StationaryStageOptimizerRecipe:
    optimizer_steps: int
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    gradient_clip_norm: float
    warmup_steps: int
    minimum_learning_rate: float
    scheduler: str = STATIONARY_STAGE_SCHEDULER

    def __post_init__(self) -> None:
        if self.optimizer_steps != 200:
            raise ValueError("Stage-B calibration is bounded to exactly 200 optimizer steps")
        _finite(self.learning_rate, "Stage-B learning rate", positive=True)
        _finite(self.weight_decay, "Stage-B weight decay")
        if len(self.betas) != 2 or any(not 0.0 <= value < 1.0 for value in self.betas):
            raise ValueError("Stage-B AdamW betas must lie in [0, 1)")
        _finite(self.eps, "Stage-B AdamW epsilon", positive=True)
        _finite(self.gradient_clip_norm, "Stage-B gradient clip", positive=True)
        _nonnegative_integer(self.warmup_steps, "Stage-B warmup steps")
        if not 0 < self.warmup_steps < self.optimizer_steps:
            raise ValueError("Stage-B warmup must end inside the bounded probe")
        _finite(self.minimum_learning_rate, "Stage-B minimum learning rate", positive=True)
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("Stage-B minimum learning rate must be below its peak")
        if self.scheduler != STATIONARY_STAGE_SCHEDULER:
            raise ValueError("unsupported Stage-B scheduler")


@dataclass(frozen=True, slots=True)
class StationaryStageDistributedRecipe:
    world_size: int
    per_rank_batch_size: int

    def __post_init__(self) -> None:
        if self.world_size != 2 or self.per_rank_batch_size != 1:
            raise ValueError("Stage-B deployment is preregistered as 2 ranks x 1 clip")


@dataclass(frozen=True, slots=True)
class StationaryTemporalStageRecipe:
    source_coverage_recipe_path: str
    source_coverage_recipe_sha256: str
    clip: StationaryStageClipRecipe
    optimizer: StationaryStageOptimizerRecipe
    distributed: StationaryStageDistributedRecipe
    stage: str = STATIONARY_STAGE_NAME
    execution_contract: str = STATIONARY_TEMPORAL_EXECUTION_CONTRACT
    schema: str = STATIONARY_STAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATIONARY_STAGE_SCHEMA or self.stage != STATIONARY_STAGE_NAME:
            raise ValueError("Stage-B schema or stage identity changed")
        if self.execution_contract != STATIONARY_TEMPORAL_EXECUTION_CONTRACT:
            raise ValueError("Stage-B stationary execution contract changed")
        _relative_path(self.source_coverage_recipe_path, "source coverage recipe path")
        _sha256(self.source_coverage_recipe_sha256, "source coverage recipe sha256")

    @property
    def recipe_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def load_source_coverage(
        self,
        repository_root: str | Path,
    ) -> MolmoAct2M2SourceCoverageRecipe:
        root = Path(repository_root).resolve()
        path = (root / self.source_coverage_recipe_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError("Stage-B source-coverage recipe is absent or escaped")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.source_coverage_recipe_sha256:
            raise ValueError("Stage-B source-coverage recipe SHA-256 changed")
        return load_molmoact2_m2_source_coverage_recipe(path)

    def load_foundation(self, repository_root: str | Path) -> PICFTrainingRecipe:
        source = self.load_source_coverage(repository_root)
        foundation = source.load_base_m2(repository_root).load_foundation(repository_root)
        if foundation.authorization.stage != "M3_structural_probe":
            raise ValueError("Stage-B foundation must be the bounded M3 structural recipe")
        if foundation.objective_config.action_weight <= 0.0:
            raise ValueError("Stage-B derivation expected the joint historical foundation")
        if foundation.geometry_overshooting.horizons != (1, 2):
            raise ValueError("Stage-B foundation overshooting horizons changed")
        if not source.split.train_ranges:
            raise ValueError("Stage-B source coverage has no train range")
        return foundation

    def structural_foundation(self, repository_root: str | Path) -> PICFTrainingRecipe:
        """Remove action credit without changing the core or structural criteria."""

        foundation = self.load_foundation(repository_root)
        structural_config = replace(
            foundation.objective_config,
            action_weight=0.0,
        )
        if not isinstance(structural_config, PICFObjectiveConfig):
            raise RuntimeError("Stage-B objective derivation changed type")
        structural = replace(foundation, objective_config=structural_config)
        if structural.objective_config.action_weight != 0.0 or any(
            left != right
            for left, right in (
                (structural.objective_config.set_weight, foundation.objective_config.set_weight),
                (
                    structural.objective_config.dynamics_weight,
                    foundation.objective_config.dynamics_weight,
                ),
                (
                    structural.objective_config.binding_weight,
                    foundation.objective_config.binding_weight,
                ),
            )
        ):
            raise RuntimeError("Stage-B derivation changed a structural objective")
        return structural

    def build_optimizer_and_scheduler(
        self,
        module: torch.nn.Module,
    ) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
        parameters = tuple(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
        if not parameters:
            raise ValueError("Stage-B module has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.optimizer.learning_rate,
            betas=self.optimizer.betas,
            eps=self.optimizer.eps,
            weight_decay=self.optimizer.weight_decay,
        )
        minimum_ratio = self.optimizer.minimum_learning_rate / self.optimizer.learning_rate

        def multiplier(step: int) -> float:
            if step < self.optimizer.warmup_steps:
                return float(step + 1) / float(self.optimizer.warmup_steps)
            progress = (step - self.optimizer.warmup_steps) / float(
                self.optimizer.optimizer_steps - self.optimizer.warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "execution_contract": self.execution_contract,
            "source_coverage_recipe_path": self.source_coverage_recipe_path,
            "source_coverage_recipe_sha256": self.source_coverage_recipe_sha256,
            "clip": {
                "prefix_lengths": list(self.clip.prefix_lengths),
                "train_length": self.clip.train_length,
                "seed": self.clip.seed,
            },
            "optimizer": {
                "optimizer_steps": self.optimizer.optimizer_steps,
                "learning_rate": self.optimizer.learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "betas": list(self.optimizer.betas),
                "eps": self.optimizer.eps,
                "gradient_clip_norm": self.optimizer.gradient_clip_norm,
                "warmup_steps": self.optimizer.warmup_steps,
                "minimum_learning_rate": self.optimizer.minimum_learning_rate,
                "scheduler": self.optimizer.scheduler,
            },
            "distributed": {
                "world_size": self.distributed.world_size,
                "per_rank_batch_size": self.distributed.per_rank_batch_size,
            },
        }


def load_stationary_temporal_stage_recipe(
    path: str | Path,
) -> StationaryTemporalStageRecipe:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-B recipe cannot be read as JSON") from exc
    payload = _exact(
        raw,
        "Stage-B recipe",
        {
            "schema",
            "stage",
            "execution_contract",
            "source_coverage_recipe_path",
            "source_coverage_recipe_sha256",
            "clip",
            "optimizer",
            "distributed",
        },
    )
    clip = _exact(payload["clip"], "Stage-B clip", {"prefix_lengths", "train_length", "seed"})
    raw_prefix = clip["prefix_lengths"]
    if not isinstance(raw_prefix, list):
        raise ValueError("Stage-B prefix lengths must be a list")
    optimizer = _exact(
        payload["optimizer"],
        "Stage-B optimizer",
        {
            "optimizer_steps",
            "learning_rate",
            "weight_decay",
            "betas",
            "eps",
            "gradient_clip_norm",
            "warmup_steps",
            "minimum_learning_rate",
            "scheduler",
        },
    )
    raw_betas = optimizer["betas"]
    if not isinstance(raw_betas, list) or len(raw_betas) != 2:
        raise ValueError("Stage-B optimizer betas must be a two-item list")
    distributed = _exact(
        payload["distributed"],
        "Stage-B distributed",
        {"world_size", "per_rank_batch_size"},
    )
    return StationaryTemporalStageRecipe(
        schema=_text(payload["schema"], "Stage-B schema"),
        stage=_text(payload["stage"], "Stage-B stage"),
        execution_contract=_text(payload["execution_contract"], "Stage-B execution contract"),
        source_coverage_recipe_path=_relative_path(
            payload["source_coverage_recipe_path"],
            "source coverage recipe path",
        ),
        source_coverage_recipe_sha256=_sha256(
            payload["source_coverage_recipe_sha256"],
            "source coverage recipe sha256",
        ),
        clip=StationaryStageClipRecipe(
            prefix_lengths=tuple(
                _nonnegative_integer(value, "Stage-B prefix length") for value in raw_prefix
            ),
            train_length=_positive_integer(clip["train_length"], "Stage-B train length"),
            seed=_nonnegative_integer(clip["seed"], "Stage-B clip seed"),
        ),
        optimizer=StationaryStageOptimizerRecipe(
            optimizer_steps=_positive_integer(
                optimizer["optimizer_steps"],
                "Stage-B optimizer steps",
            ),
            learning_rate=_finite(optimizer["learning_rate"], "Stage-B lr", positive=True),
            weight_decay=_finite(optimizer["weight_decay"], "Stage-B weight decay"),
            betas=(
                _finite(raw_betas[0], "Stage-B beta"),
                _finite(raw_betas[1], "Stage-B beta"),
            ),
            eps=_finite(optimizer["eps"], "Stage-B epsilon", positive=True),
            gradient_clip_norm=_finite(
                optimizer["gradient_clip_norm"],
                "Stage-B gradient clip",
                positive=True,
            ),
            warmup_steps=_nonnegative_integer(
                optimizer["warmup_steps"],
                "Stage-B warmup steps",
            ),
            minimum_learning_rate=_finite(
                optimizer["minimum_learning_rate"],
                "Stage-B minimum lr",
                positive=True,
            ),
            scheduler=_text(optimizer["scheduler"], "Stage-B scheduler"),
        ),
        distributed=StationaryStageDistributedRecipe(
            world_size=_positive_integer(distributed["world_size"], "Stage-B world size"),
            per_rank_batch_size=_positive_integer(
                distributed["per_rank_batch_size"],
                "Stage-B per-rank batch size",
            ),
        ),
    )

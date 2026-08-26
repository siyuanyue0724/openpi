"""Contract for current-frame counterfactual measurement calibration.

The stage does not add a model head or a second objective.  It supplies the
existing unordered current-frame set likelihood with paired factual and exact
object-removal observations, while replaying natural observations to constrain
regression.  Synthetic removals are measurement interventions only and must
never supervise temporal lifecycle events.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

COUNTERFACTUAL_MEASUREMENT_SCHEMA = "picf-next.molmoact2-m2-counterfactual-measurement.v1"
COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2 = "picf-next.molmoact2-m2-counterfactual-measurement.v2"
COUNTERFACTUAL_MEASUREMENT_GATE = "M2_counterfactual_measurement"
FIXED_MARGIN_DECISION_RULE = "fixed-margin-v2"
OCCAM_COMPLETE_SET_DECISION_RULE = "occam-complete-set-v1"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name)
    if set(result) != fields:
        raise ValueError(
            f"{name} fields differ; missing={sorted(fields - set(result))}, "
            f"unknown={sorted(set(result) - fields)}"
        )
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _relative_path(value: object, name: str) -> str:
    result = _text(value, name)
    path = Path(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return result


def _sha256(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _probability(value: object, name: str) -> float:
    result = _number(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class CounterfactualMeasurementOptimization:
    steps: int
    pair_count_per_step: int
    natural_count_per_step: int
    natural_replay_pool_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    warmup_steps: int
    seed: int

    def __post_init__(self) -> None:
        _integer(self.steps, "optimization.steps", minimum=1)
        _integer(self.pair_count_per_step, "optimization.pair_count_per_step", minimum=1)
        _integer(
            self.natural_count_per_step,
            "optimization.natural_count_per_step",
            minimum=1,
        )
        _integer(
            self.natural_replay_pool_size,
            "optimization.natural_replay_pool_size",
            minimum=self.natural_count_per_step,
        )
        _number(self.learning_rate, "optimization.learning_rate", minimum=1e-12)
        _number(self.weight_decay, "optimization.weight_decay")
        _number(
            self.gradient_clip_norm,
            "optimization.gradient_clip_norm",
            minimum=1e-12,
        )
        _integer(self.warmup_steps, "optimization.warmup_steps")
        _integer(self.seed, "optimization.seed")
        if self.warmup_steps >= self.steps:
            raise ValueError("counterfactual warmup must end before the last step")


@dataclass(frozen=True, slots=True)
class CounterfactualMeasurementAcceptance:
    minimum_pairs: int
    minimum_distinct_identities: int
    maximum_removed_unmatched_existence: float
    minimum_factual_target_existence: float
    minimum_factual_target_soft_dice: float
    minimum_removed_loss_improvement_fraction: float
    maximum_natural_replay_loss_regression_fraction: float
    maximum_natural_count_mae_regression: float
    minimum_control_removed_existence_margin: float

    def __post_init__(self) -> None:
        _integer(self.minimum_pairs, "acceptance.minimum_pairs", minimum=2)
        _integer(
            self.minimum_distinct_identities,
            "acceptance.minimum_distinct_identities",
            minimum=2,
        )
        for name in (
            "maximum_removed_unmatched_existence",
            "minimum_factual_target_existence",
            "minimum_factual_target_soft_dice",
            "minimum_removed_loss_improvement_fraction",
            "maximum_natural_replay_loss_regression_fraction",
            "minimum_control_removed_existence_margin",
        ):
            _probability(getattr(self, name), f"acceptance.{name}")
        _number(
            self.maximum_natural_count_mae_regression,
            "acceptance.maximum_natural_count_mae_regression",
        )
        if self.minimum_distinct_identities > self.minimum_pairs:
            raise ValueError("minimum identities cannot exceed minimum pairs")


@dataclass(frozen=True, slots=True)
class CounterfactualMeasurementOccamAcceptance:
    """Operational gates for the v2 complete-set candidate decision."""

    minimum_pairs: int
    minimum_distinct_identities: int
    maximum_removed_unmatched_existence: float
    minimum_factual_target_existence: float
    minimum_factual_target_soft_dice: float
    maximum_removed_set_loss_regression_fraction: float
    maximum_natural_replay_loss_regression_fraction: float
    maximum_natural_count_mae_regression: float

    def __post_init__(self) -> None:
        _integer(self.minimum_pairs, "acceptance.minimum_pairs", minimum=2)
        _integer(
            self.minimum_distinct_identities,
            "acceptance.minimum_distinct_identities",
            minimum=2,
        )
        for name in (
            "maximum_removed_unmatched_existence",
            "minimum_factual_target_existence",
            "minimum_factual_target_soft_dice",
            "maximum_removed_set_loss_regression_fraction",
            "maximum_natural_replay_loss_regression_fraction",
        ):
            _probability(getattr(self, name), f"acceptance.{name}")
        _number(
            self.maximum_natural_count_mae_regression,
            "acceptance.maximum_natural_count_mae_regression",
        )
        if self.minimum_distinct_identities > self.minimum_pairs:
            raise ValueError("minimum identities cannot exceed minimum pairs")


@dataclass(frozen=True, slots=True)
class CounterfactualMeasurementRecipe:
    foundation_m2_recipe_path: str
    foundation_m2_recipe_sha256: str
    optimization: CounterfactualMeasurementOptimization
    acceptance: CounterfactualMeasurementAcceptance | CounterfactualMeasurementOccamAcceptance
    schema: str = COUNTERFACTUAL_MEASUREMENT_SCHEMA
    gate: str = COUNTERFACTUAL_MEASUREMENT_GATE
    decision_rule: str = FIXED_MARGIN_DECISION_RULE

    def __post_init__(self) -> None:
        if self.gate != COUNTERFACTUAL_MEASUREMENT_GATE:
            raise ValueError("counterfactual measurement schema or gate changed")
        expected_rule = {
            COUNTERFACTUAL_MEASUREMENT_SCHEMA: FIXED_MARGIN_DECISION_RULE,
            COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2: OCCAM_COMPLETE_SET_DECISION_RULE,
        }.get(self.schema)
        if expected_rule is None or self.decision_rule != expected_rule:
            raise ValueError("counterfactual measurement schema and decision rule disagree")
        expected_acceptance = (
            CounterfactualMeasurementOccamAcceptance
            if self.schema == COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2
            else CounterfactualMeasurementAcceptance
        )
        if not isinstance(self.acceptance, expected_acceptance):
            raise ValueError("counterfactual schema and acceptance contract disagree")
        _relative_path(self.foundation_m2_recipe_path, "foundation_m2_recipe_path")
        _sha256(self.foundation_m2_recipe_sha256, "foundation_m2_recipe_sha256")

    @property
    def recipe_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()

    def foundation_m2_path(self, repository_root: str | Path) -> Path:
        root = Path(repository_root).resolve()
        path = (root / self.foundation_m2_recipe_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError("counterfactual foundation M2 recipe is absent")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.foundation_m2_recipe_sha256:
            raise ValueError("counterfactual foundation M2 recipe hash changed")
        return path

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "gate": self.gate,
            "foundation_m2_recipe_path": self.foundation_m2_recipe_path,
            "foundation_m2_recipe_sha256": self.foundation_m2_recipe_sha256,
            "optimization": {
                name: getattr(self.optimization, name)
                for name in self.optimization.__dataclass_fields__
            },
            "acceptance": {
                name: getattr(self.acceptance, name)
                for name in self.acceptance.__dataclass_fields__
            },
        }
        if self.schema == COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2:
            payload["decision_rule"] = self.decision_rule
        return payload


def load_counterfactual_measurement_recipe(
    path: str | Path,
) -> CounterfactualMeasurementRecipe:
    source = Path(path)
    try:
        raw = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("counterfactual measurement recipe is not valid JSON") from error
    raw_mapping = _mapping(raw, "counterfactual measurement recipe")
    schema = _text(raw_mapping.get("schema"), "schema")
    recipe_fields = {
        "schema",
        "gate",
        "foundation_m2_recipe_path",
        "foundation_m2_recipe_sha256",
        "optimization",
        "acceptance",
    }
    if schema == COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2:
        recipe_fields.add("decision_rule")
    elif schema != COUNTERFACTUAL_MEASUREMENT_SCHEMA:
        raise ValueError("counterfactual measurement schema changed")
    payload = _exact(raw_mapping, "counterfactual measurement recipe", recipe_fields)
    optimization = _exact(
        payload["optimization"],
        "optimization",
        {
            "steps",
            "pair_count_per_step",
            "natural_count_per_step",
            "natural_replay_pool_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "warmup_steps",
            "seed",
        },
    )
    acceptance_fields = (
        {
            "minimum_pairs",
            "minimum_distinct_identities",
            "maximum_removed_unmatched_existence",
            "minimum_factual_target_existence",
            "minimum_factual_target_soft_dice",
            "maximum_removed_set_loss_regression_fraction",
            "maximum_natural_replay_loss_regression_fraction",
            "maximum_natural_count_mae_regression",
        }
        if schema == COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2
        else {
            "minimum_pairs",
            "minimum_distinct_identities",
            "maximum_removed_unmatched_existence",
            "minimum_factual_target_existence",
            "minimum_factual_target_soft_dice",
            "minimum_removed_loss_improvement_fraction",
            "maximum_natural_replay_loss_regression_fraction",
            "maximum_natural_count_mae_regression",
            "minimum_control_removed_existence_margin",
        }
    )
    acceptance = _exact(
        payload["acceptance"],
        "acceptance",
        acceptance_fields,
    )
    parsed_acceptance: (
        CounterfactualMeasurementAcceptance | CounterfactualMeasurementOccamAcceptance
    )
    if schema == COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2:
        parsed_acceptance = CounterfactualMeasurementOccamAcceptance(
            minimum_pairs=_integer(
                acceptance["minimum_pairs"],
                "acceptance.minimum_pairs",
                minimum=2,
            ),
            minimum_distinct_identities=_integer(
                acceptance["minimum_distinct_identities"],
                "acceptance.minimum_distinct_identities",
                minimum=2,
            ),
            maximum_removed_unmatched_existence=_probability(
                acceptance["maximum_removed_unmatched_existence"],
                "acceptance.maximum_removed_unmatched_existence",
            ),
            minimum_factual_target_existence=_probability(
                acceptance["minimum_factual_target_existence"],
                "acceptance.minimum_factual_target_existence",
            ),
            minimum_factual_target_soft_dice=_probability(
                acceptance["minimum_factual_target_soft_dice"],
                "acceptance.minimum_factual_target_soft_dice",
            ),
            maximum_removed_set_loss_regression_fraction=_probability(
                acceptance["maximum_removed_set_loss_regression_fraction"],
                "acceptance.maximum_removed_set_loss_regression_fraction",
            ),
            maximum_natural_replay_loss_regression_fraction=_probability(
                acceptance["maximum_natural_replay_loss_regression_fraction"],
                "acceptance.maximum_natural_replay_loss_regression_fraction",
            ),
            maximum_natural_count_mae_regression=_number(
                acceptance["maximum_natural_count_mae_regression"],
                "acceptance.maximum_natural_count_mae_regression",
            ),
        )
    else:
        parsed_acceptance = CounterfactualMeasurementAcceptance(
            minimum_pairs=_integer(
                acceptance["minimum_pairs"],
                "acceptance.minimum_pairs",
                minimum=2,
            ),
            minimum_distinct_identities=_integer(
                acceptance["minimum_distinct_identities"],
                "acceptance.minimum_distinct_identities",
                minimum=2,
            ),
            maximum_removed_unmatched_existence=_probability(
                acceptance["maximum_removed_unmatched_existence"],
                "acceptance.maximum_removed_unmatched_existence",
            ),
            minimum_factual_target_existence=_probability(
                acceptance["minimum_factual_target_existence"],
                "acceptance.minimum_factual_target_existence",
            ),
            minimum_factual_target_soft_dice=_probability(
                acceptance["minimum_factual_target_soft_dice"],
                "acceptance.minimum_factual_target_soft_dice",
            ),
            minimum_removed_loss_improvement_fraction=_probability(
                acceptance["minimum_removed_loss_improvement_fraction"],
                "acceptance.minimum_removed_loss_improvement_fraction",
            ),
            maximum_natural_replay_loss_regression_fraction=_probability(
                acceptance["maximum_natural_replay_loss_regression_fraction"],
                "acceptance.maximum_natural_replay_loss_regression_fraction",
            ),
            maximum_natural_count_mae_regression=_number(
                acceptance["maximum_natural_count_mae_regression"],
                "acceptance.maximum_natural_count_mae_regression",
            ),
            minimum_control_removed_existence_margin=_probability(
                acceptance["minimum_control_removed_existence_margin"],
                "acceptance.minimum_control_removed_existence_margin",
            ),
        )
    return CounterfactualMeasurementRecipe(
        schema=schema,
        gate=_text(payload["gate"], "gate"),
        decision_rule=(
            _text(payload["decision_rule"], "decision_rule")
            if schema == COUNTERFACTUAL_MEASUREMENT_SCHEMA_V2
            else FIXED_MARGIN_DECISION_RULE
        ),
        foundation_m2_recipe_path=_relative_path(
            payload["foundation_m2_recipe_path"],
            "foundation_m2_recipe_path",
        ),
        foundation_m2_recipe_sha256=_sha256(
            payload["foundation_m2_recipe_sha256"],
            "foundation_m2_recipe_sha256",
        ),
        optimization=CounterfactualMeasurementOptimization(
            steps=_integer(optimization["steps"], "optimization.steps", minimum=1),
            pair_count_per_step=_integer(
                optimization["pair_count_per_step"],
                "optimization.pair_count_per_step",
                minimum=1,
            ),
            natural_count_per_step=_integer(
                optimization["natural_count_per_step"],
                "optimization.natural_count_per_step",
                minimum=1,
            ),
            natural_replay_pool_size=_integer(
                optimization["natural_replay_pool_size"],
                "optimization.natural_replay_pool_size",
                minimum=1,
            ),
            learning_rate=_number(
                optimization["learning_rate"],
                "optimization.learning_rate",
                minimum=1e-12,
            ),
            weight_decay=_number(
                optimization["weight_decay"],
                "optimization.weight_decay",
            ),
            gradient_clip_norm=_number(
                optimization["gradient_clip_norm"],
                "optimization.gradient_clip_norm",
                minimum=1e-12,
            ),
            warmup_steps=_integer(
                optimization["warmup_steps"],
                "optimization.warmup_steps",
            ),
            seed=_integer(optimization["seed"], "optimization.seed"),
        ),
        acceptance=parsed_acceptance,
    )


def deterministic_cycle(
    values: tuple[str, ...],
    *,
    count: int,
    seed: int,
    step: int,
) -> tuple[str, ...]:
    """Select a deterministic, epoch-balanced slice without RNG state."""

    if not values or count <= 0 or step <= 0:
        raise ValueError("deterministic cycle requires values, count and a positive step")
    ordered: list[str] = []
    epoch = ((step - 1) * count) // len(values)
    while len(ordered) < count:
        permutation = sorted(
            values,
            key=lambda value: hashlib.sha256(f"{seed}:{epoch}:{value}".encode("ascii")).digest(),
        )
        offset = ((step - 1) * count) % len(values) if not ordered else 0
        ordered.extend(permutation[offset:])
        epoch += 1
    return tuple(ordered[:count])


def deterministic_cycle_exposure_counts(
    values: tuple[str, ...],
    *,
    count: int,
    seed: int,
    steps: int,
) -> dict[str, int]:
    """Enumerate the exact per-item exposure implied by a cycle schedule."""

    if steps <= 0:
        raise ValueError("deterministic cycle exposure requires positive steps")
    exposure = {value: 0 for value in values}
    if len(exposure) != len(values):
        raise ValueError("deterministic cycle values must be unique")
    for step in range(1, steps + 1):
        for value in deterministic_cycle(values, count=count, seed=seed, step=step):
            exposure[value] += 1
    return exposure


def _formal_partition_acceptance(
    *,
    recipe: CounterfactualMeasurementRecipe,
    prior: Mapping[str, object],
    actual: Mapping[str, object],
    control: Mapping[str, object],
) -> tuple[dict[str, bool], dict[str, float]]:
    threshold = recipe.acceptance
    if not isinstance(threshold, CounterfactualMeasurementAcceptance):
        raise ValueError("fixed-margin partition acceptance requires a v1 recipe")
    actual_rows = _mapping(actual["pairs"], "actual pair rows")
    rows = tuple(_mapping(value, f"actual pair row {key}") for key, value in actual_rows.items())
    prior_removed = _number(prior["mean_removed_loss"], "prior mean removed loss")
    actual_removed = _number(actual["mean_removed_loss"], "actual mean removed loss")
    removed_improvement = (
        (prior_removed - actual_removed) / prior_removed if prior_removed > 0.0 else float("-inf")
    )
    control_margin = _probability(
        control["mean_removed_maximum_unmatched_existence"],
        "control mean removed maximum unmatched existence",
    ) - _probability(
        actual["mean_removed_maximum_unmatched_existence"],
        "actual mean removed maximum unmatched existence",
    )

    def branch(row: Mapping[str, object], name: str) -> Mapping[str, object]:
        return _mapping(row[name], f"pair row {name}")

    checks = {
        "factual_targets_preserved": all(
            _probability(
                branch(row, "factual")["target_existence"],
                "factual target existence",
            )
            >= threshold.minimum_factual_target_existence
            and _probability(
                branch(row, "factual")["target_soft_dice"],
                "factual target soft dice",
            )
            >= threshold.minimum_factual_target_soft_dice
            for row in rows
        ),
        "removed_targets_rejected": all(
            _probability(
                branch(row, "removed")["maximum_unmatched_existence"],
                "removed maximum unmatched existence",
            )
            <= threshold.maximum_removed_unmatched_existence
            for row in rows
        ),
        "removed_set_cardinality_exact": all(
            _integer(branch(row, "removed")["active_count"], "removed active count")
            == _integer(branch(row, "removed")["target_count"], "removed target count")
            for row in rows
        ),
        "removed_loss_improves": (
            removed_improvement >= threshold.minimum_removed_loss_improvement_fraction
        ),
        "counterfactual_signal_beats_factual_only_control": (
            control_margin >= threshold.minimum_control_removed_existence_margin
        ),
    }
    return checks, {
        "removed_loss_improvement_fraction": removed_improvement,
        "factual_only_control_removed_existence_margin": control_margin,
    }


def formal_counterfactual_measurement_acceptance(
    *,
    recipe: CounterfactualMeasurementRecipe,
    prior_pairs: Mapping[str, Mapping[str, object]],
    actual_pairs: Mapping[str, Mapping[str, object]],
    control_pairs: Mapping[str, Mapping[str, object]],
    prior_natural: Mapping[str, object],
    actual_natural: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate unseen-source and unseen-identity pairs without fitting either."""

    threshold = recipe.acceptance
    if not isinstance(threshold, CounterfactualMeasurementAcceptance):
        raise ValueError("fixed-margin acceptance requires a v1 recipe")
    partition_checks = {}
    partition_metrics = {}
    checks: dict[str, bool] = {}
    for partition in ("validation", "heldout"):
        current_checks, current_metrics = _formal_partition_acceptance(
            recipe=recipe,
            prior=prior_pairs[partition],
            actual=actual_pairs[partition],
            control=control_pairs[partition],
        )
        partition_checks[partition] = current_checks
        partition_metrics[partition] = current_metrics
        checks.update({f"{partition}_{name}": passed for name, passed in current_checks.items()})
    prior_losses = _mapping(prior_natural["losses"], "prior natural losses")
    actual_losses = _mapping(actual_natural["losses"], "actual natural losses")
    prior_natural_loss = _number(prior_losses["loss_total"], "prior natural total loss")
    actual_natural_loss = _number(actual_losses["loss_total"], "actual natural total loss")
    natural_regression = (
        (actual_natural_loss - prior_natural_loss) / prior_natural_loss
        if prior_natural_loss > 0.0
        else float("inf")
    )
    checks.update(
        {
            "natural_loss_preserved": natural_regression
            <= threshold.maximum_natural_replay_loss_regression_fraction,
            "natural_count_preserved": (
                _number(actual_natural["count_mae"], "actual natural count MAE")
                - _number(prior_natural["count_mae"], "prior natural count MAE")
                <= threshold.maximum_natural_count_mae_regression
            ),
        }
    )
    passed = all(checks.values())
    return {
        "status": "PASS_COUNTERFACTUAL_MEASUREMENT" if passed else "FAIL",
        "checks": checks,
        "failed_checks": sorted(name for name, result in checks.items() if not result),
        "partition_checks": partition_checks,
        "partition_metrics": partition_metrics,
        "natural_replay_loss_regression_fraction": natural_regression,
        "later_gates_authorized": (["M3_stationary_temporal_revalidation"] if passed else []),
    }


def _complete_pair_partition_acceptance(
    *,
    recipe: CounterfactualMeasurementRecipe,
    prior: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[
    dict[str, bool],
    dict[str, float | int],
    dict[str, dict[str, bool]],
]:
    """Evaluate one candidate as a complete finite measurement-set model."""

    threshold = recipe.acceptance
    if not isinstance(threshold, CounterfactualMeasurementOccamAcceptance):
        raise ValueError("complete-set acceptance requires a v2 recipe")
    prior_rows = _mapping(prior["pairs"], "prior pair rows")
    candidate_rows = _mapping(candidate["pairs"], "candidate pair rows")
    if set(prior_rows) != set(candidate_rows):
        raise ValueError("candidate and prior pair identities differ")

    def branch(row: Mapping[str, object], name: str) -> Mapping[str, object]:
        return _mapping(row[name], f"pair row {name}")

    pair_checks: dict[str, dict[str, bool]] = {}
    identities: set[str] = set()
    factual_target_success = []
    factual_cardinality_success = []
    removed_target_success = []
    removed_cardinality_success = []
    for pair_id, raw_row in candidate_rows.items():
        row = _mapping(raw_row, f"candidate pair row {pair_id}")
        prior_row = _mapping(prior_rows[pair_id], f"prior pair row {pair_id}")
        identity = _text(row["target_identity_key"], "target_identity_key")
        if identity != _text(prior_row["target_identity_key"], "prior target_identity_key"):
            raise ValueError("candidate and prior pair target identities differ")
        identities.add(identity)
        factual = branch(row, "factual")
        removed = branch(row, "removed")
        target_ok = (
            _probability(factual["target_existence"], "factual target existence")
            >= threshold.minimum_factual_target_existence
            and _probability(factual["target_soft_dice"], "factual target soft dice")
            >= threshold.minimum_factual_target_soft_dice
        )
        factual_count_ok = _integer(factual["active_count"], "factual active count") == _integer(
            factual["target_count"], "factual target count"
        )
        removed_target_ok = (
            _probability(
                removed["maximum_unmatched_existence"],
                "removed maximum unmatched existence",
            )
            <= threshold.maximum_removed_unmatched_existence
        )
        removed_count_ok = _integer(removed["active_count"], "removed active count") == _integer(
            removed["target_count"], "removed target count"
        )
        factual_target_success.append(target_ok)
        factual_cardinality_success.append(factual_count_ok)
        removed_target_success.append(removed_target_ok)
        removed_cardinality_success.append(removed_count_ok)
        pair_checks[pair_id] = {
            "factual_target_preserved": target_ok,
            "factual_set_cardinality_exact": factual_count_ok,
            "removed_target_rejected": removed_target_ok,
            "removed_set_cardinality_exact": removed_count_ok,
        }

    prior_removed_loss = _number(prior["mean_removed_loss"], "prior mean removed loss")
    candidate_removed_loss = _number(candidate["mean_removed_loss"], "candidate mean removed loss")
    removed_loss_regression = (
        (candidate_removed_loss - prior_removed_loss) / prior_removed_loss
        if prior_removed_loss > 0.0
        else float("inf")
    )
    checks = {
        "pair_count_sufficient": len(pair_checks) >= threshold.minimum_pairs,
        "identity_count_sufficient": (len(identities) >= threshold.minimum_distinct_identities),
        "factual_targets_preserved": all(factual_target_success),
        "factual_set_cardinality_exact": all(factual_cardinality_success),
        "removed_targets_rejected": all(removed_target_success),
        "removed_set_cardinality_exact": all(removed_cardinality_success),
        "removed_set_loss_preserved": (
            removed_loss_regression <= threshold.maximum_removed_set_loss_regression_fraction
        ),
    }
    complete_pair_success = sum(all(values.values()) for values in pair_checks.values())
    pair_count = len(pair_checks)
    return (
        checks,
        {
            "pair_count": pair_count,
            "identity_count": len(identities),
            "complete_pair_success_count": complete_pair_success,
            "complete_pair_success_rate": (
                complete_pair_success / pair_count if pair_count else 0.0
            ),
            "mean_removed_loss": candidate_removed_loss,
            "removed_loss_regression_fraction": removed_loss_regression,
            "mean_removed_maximum_unmatched_existence": _probability(
                candidate["mean_removed_maximum_unmatched_existence"],
                "candidate mean removed maximum unmatched existence",
            ),
        },
        pair_checks,
    )


def _natural_candidate_acceptance(
    *,
    recipe: CounterfactualMeasurementRecipe,
    prior: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[dict[str, bool], dict[str, float]]:
    threshold = recipe.acceptance
    if not isinstance(threshold, CounterfactualMeasurementOccamAcceptance):
        raise ValueError("Occam natural acceptance requires a v2 recipe")
    prior_losses = _mapping(prior["losses"], "prior natural losses")
    candidate_losses = _mapping(candidate["losses"], "candidate natural losses")
    prior_loss = _number(prior_losses["loss_total"], "prior natural total loss")
    candidate_loss = _number(candidate_losses["loss_total"], "candidate natural total loss")
    loss_regression = (
        (candidate_loss - prior_loss) / prior_loss if prior_loss > 0.0 else float("inf")
    )
    count_regression = _number(candidate["count_mae"], "candidate natural count MAE") - _number(
        prior["count_mae"], "prior natural count MAE"
    )
    return {
        "natural_loss_preserved": (
            loss_regression <= threshold.maximum_natural_replay_loss_regression_fraction
        ),
        "natural_count_preserved": (
            count_regression <= threshold.maximum_natural_count_mae_regression
        ),
    }, {
        "natural_loss_regression_fraction": loss_regression,
        "natural_count_mae_regression": count_regression,
    }


def formal_counterfactual_measurement_occam_acceptance(
    *,
    recipe: CounterfactualMeasurementRecipe,
    prior_pairs: Mapping[str, Mapping[str, object]],
    actual_pairs: Mapping[str, Mapping[str, object]],
    control_pairs: Mapping[str, Mapping[str, object]],
    prior_natural: Mapping[str, object],
    actual_natural: Mapping[str, object],
    control_natural: Mapping[str, object],
) -> dict[str, object]:
    """Select the simplest complete-set candidate that passes every gate.

    The factual-only arm wins whenever it is already operationally sufficient.
    Counterfactual calibration is selected only when it passes all the same
    gates and corrects independently failing control pairs across identities.
    This makes floor-saturated correct controls non-inferiority cases instead
    of demanding an impossible fixed probability decrease.
    """

    if recipe.decision_rule != OCCAM_COMPLETE_SET_DECISION_RULE:
        raise ValueError("Occam acceptance requires its versioned decision rule")
    partitions = ("validation", "heldout")
    candidates = {
        "counterfactual": (actual_pairs, actual_natural),
        "factual_only_control": (control_pairs, control_natural),
    }
    candidate_checks: dict[str, dict[str, object]] = {}
    candidate_metrics: dict[str, dict[str, object]] = {}
    pair_checks_by_candidate: dict[str, dict[str, dict[str, dict[str, bool]]]] = {}
    for candidate_name, (pair_partitions, natural) in candidates.items():
        partition_checks: dict[str, dict[str, bool]] = {}
        partition_metrics: dict[str, dict[str, float | int]] = {}
        pair_checks_by_candidate[candidate_name] = {}
        for partition in partitions:
            checks, metrics, pair_checks = _complete_pair_partition_acceptance(
                recipe=recipe,
                prior=prior_pairs[partition],
                candidate=pair_partitions[partition],
            )
            partition_checks[partition] = checks
            partition_metrics[partition] = metrics
            pair_checks_by_candidate[candidate_name][partition] = pair_checks
        natural_checks, natural_metrics = _natural_candidate_acceptance(
            recipe=recipe,
            prior=prior_natural,
            candidate=natural,
        )
        all_checks = {
            f"{partition}_{name}": passed
            for partition, checks in partition_checks.items()
            for name, passed in checks.items()
        }
        all_checks.update(natural_checks)
        candidate_checks[candidate_name] = {
            "all_pass": all(all_checks.values()),
            "checks": all_checks,
            "partition_checks": partition_checks,
            "natural_checks": natural_checks,
        }
        candidate_metrics[candidate_name] = {
            "partition_metrics": partition_metrics,
            "natural_metrics": natural_metrics,
        }

    corrected_pairs: list[dict[str, str]] = []
    corrected_identities: set[str] = set()
    for partition in partitions:
        actual_pair_checks = pair_checks_by_candidate["counterfactual"][partition]
        control_pair_checks = pair_checks_by_candidate["factual_only_control"][partition]
        actual_rows = _mapping(actual_pairs[partition]["pairs"], "actual pair rows")
        for pair_id, actual_checks in actual_pair_checks.items():
            control_checks = control_pair_checks[pair_id]
            actual_complete = all(actual_checks.values())
            control_factual_ok = (
                control_checks["factual_target_preserved"]
                and control_checks["factual_set_cardinality_exact"]
            )
            failed_removed_modes = sorted(
                name
                for name in (
                    "removed_target_rejected",
                    "removed_set_cardinality_exact",
                )
                if not control_checks[name]
            )
            if not actual_complete or not control_factual_ok or not failed_removed_modes:
                continue
            row = _mapping(actual_rows[pair_id], f"actual pair row {pair_id}")
            identity = _text(row["target_identity_key"], "target_identity_key")
            corrected_pairs.append(
                {
                    "partition": partition,
                    "pair_id": pair_id,
                    "target_identity_key": identity,
                    "corrected_failure_modes": ",".join(failed_removed_modes),
                }
            )
            corrected_identities.add(identity)
    added_value = len(corrected_identities) >= recipe.acceptance.minimum_distinct_identities
    control_pass = bool(candidate_checks["factual_only_control"]["all_pass"])
    actual_pass = bool(candidate_checks["counterfactual"]["all_pass"])
    if control_pass:
        status = "PASS_FACTUAL_BASELINE"
        selected_candidate: str | None = "factual_only_control"
    elif actual_pass and added_value:
        status = "PASS_COUNTERFACTUAL_MEASUREMENT"
        selected_candidate = "counterfactual"
    else:
        status = "FAIL"
        selected_candidate = None

    failed_checks: list[str] = []
    if status == "FAIL":
        for candidate_name, report in candidate_checks.items():
            checks = _mapping(report["checks"], f"{candidate_name} checks")
            failed_checks.extend(
                f"{candidate_name}_{name}" for name, passed in checks.items() if not passed
            )
        if not added_value:
            failed_checks.append("counterfactual_corrects_control_failures_across_identities")
    return {
        "status": status,
        "decision_rule": recipe.decision_rule,
        "selected_candidate": selected_candidate,
        "candidate_checks": candidate_checks,
        "candidate_metrics": candidate_metrics,
        "corrected_control_pairs": corrected_pairs,
        "corrected_control_identity_count": len(corrected_identities),
        "counterfactual_added_value": added_value,
        "failed_checks": sorted(failed_checks),
        "later_gates_authorized": (
            ["M3_stationary_temporal_revalidation"] if selected_candidate is not None else []
        ),
    }

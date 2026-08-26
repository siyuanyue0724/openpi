"""Pre-registered all-source data-coverage audit for the MolmoAct2 M2 gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from picf_next.training.molmoact2_m2 import (
    MolmoAct2M2Recipe,
    load_molmoact2_m2_recipe,
)

M2_SOURCE_COVERAGE_SCHEMA = "picf-next.molmoact2-m2-source-coverage.v1"
M2_SOURCE_COVERAGE_GATE = "M2_representation_source_coverage_root_cause"
M2_SOURCE_NEUTRAL_TASK_CONTRACT = "official-processor-task-field-absent.v1"
M2_SOURCE_SPLIT_STRATEGY = "disjoint-contiguous-source-ranges-with-guard.v1"
M2_SOURCE_COMPARISON_SCOPE = (
    "composite-corrected-data-pipeline-readiness-not-single-variable-ablation.v1"
)
M2_SOURCE_EXTERNAL_ACCEPTANCE = "base-m2-applicable-thresholds-without-reselection.v1"


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return text


def _range(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{name} must be one integer [start, end_exclusive] pair")
    start, stop = value
    if start < 0 or stop <= start:
        raise ValueError(f"{name} must be one nonempty nonnegative half-open range")
    return start, stop


def _ranges(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be one nonempty range list")
    ranges = tuple(_range(item, f"{name}[{index}]") for index, item in enumerate(value))
    if ranges != tuple(sorted(ranges)) or any(
        left[1] > right[0] for left, right in zip(ranges, ranges[1:], strict=False)
    ):
        raise ValueError(f"{name} ranges must be sorted and non-overlapping")
    return ranges


@dataclass(frozen=True, slots=True)
class M2SourceCoverageSplit:
    source_episode: tuple[int, int]
    train_ranges: tuple[tuple[int, int], ...]
    validation_ranges: tuple[tuple[int, int], ...]
    heldout_ranges: tuple[tuple[int, int], ...]
    guard_ranges: tuple[tuple[int, int], ...]
    minimum_guard_frames: int
    strategy: str = M2_SOURCE_SPLIT_STRATEGY

    def __post_init__(self) -> None:
        if self.strategy != M2_SOURCE_SPLIT_STRATEGY:
            raise ValueError("unsupported M2 all-source split strategy")
        if (
            not isinstance(self.minimum_guard_frames, int)
            or isinstance(self.minimum_guard_frames, bool)
            or self.minimum_guard_frames <= 0
        ):
            raise ValueError("M2 all-source minimum guard must be positive")
        named_ranges = (
            ("train", self.train_ranges),
            ("validation", self.validation_ranges),
            ("heldout", self.heldout_ranges),
            ("guard", self.guard_ranges),
        )
        flattened = sorted(
            (start, stop, name) for name, ranges in named_ranges for start, stop in ranges
        )
        if any(left[1] != right[0] for left, right in zip(flattened, flattened[1:], strict=False)):
            raise ValueError("M2 all-source ranges must partition the source episode exactly")
        if (
            not flattened
            or flattened[0][0] != self.source_episode[0]
            or flattened[-1][1] != self.source_episode[1]
        ):
            raise ValueError("M2 all-source ranges do not cover the declared source episode")
        if (
            flattened[0][2] == "guard"
            or flattened[-1][2] == "guard"
            or any(
                (left[2] == "guard") == (right[2] == "guard")
                for left, right in zip(flattened, flattened[1:], strict=False)
            )
        ):
            raise ValueError(
                "M2 all-source learned ranges must be separated by exactly one guard range"
            )
        if any(stop - start < self.minimum_guard_frames for start, stop in self.guard_ranges):
            raise ValueError("M2 all-source guard range is shorter than its declared minimum")

    @property
    def learned_ranges(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (name, start, stop)
            for name, ranges in (
                ("train", self.train_ranges),
                ("validation", self.validation_ranges),
                ("heldout", self.heldout_ranges),
            )
            for start, stop in ranges
        )

    def split_name(self, global_index: int) -> str:
        matches = tuple(
            name for name, start, stop in self.learned_ranges if start <= global_index < stop
        )
        if len(matches) != 1:
            raise KeyError(f"source frame {global_index} is not in exactly one learned split")
        return matches[0]


@dataclass(frozen=True, slots=True)
class M2SourceExternalValidation:
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    physical_sidecar_name: str
    physical_sidecar_manifest_sha256: str
    target_probe_path: str
    target_probe_sha256: str
    source_episode: tuple[int, int]
    acceptance_policy: str = M2_SOURCE_EXTERNAL_ACCEPTANCE

    def __post_init__(self) -> None:
        _relative_path(self.dataset_manifest_path, "external_validation.dataset_manifest_path")
        _sha256(
            self.dataset_manifest_sha256,
            "external_validation.dataset_manifest_sha256",
        )
        if (
            not self.physical_sidecar_name
            or "/" in self.physical_sidecar_name
            or self.physical_sidecar_name in {".", ".."}
        ):
            raise ValueError("external-validation sidecar name must be one path component")
        _sha256(
            self.physical_sidecar_manifest_sha256,
            "external_validation.physical_sidecar_manifest_sha256",
        )
        _relative_path(self.target_probe_path, "external_validation.target_probe_path")
        _sha256(self.target_probe_sha256, "external_validation.target_probe_sha256")
        if self.acceptance_policy != M2_SOURCE_EXTERNAL_ACCEPTANCE:
            raise ValueError("unsupported M2 source external-validation acceptance policy")

    @property
    def frame_count(self) -> int:
        return self.source_episode[1] - self.source_episode[0]

    def load_dataset_manifest(self, repository_root: str | Path) -> dict[str, Any]:
        root = Path(repository_root).resolve()
        path = (root / self.dataset_manifest_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError("external-validation dataset manifest is absent or escaped")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.dataset_manifest_sha256:
            raise ValueError("external-validation dataset manifest SHA-256 changed")
        try:
            manifest = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("external-validation dataset manifest is invalid JSON") from error
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if (
            not isinstance(files, list)
            or manifest.get("split_name") != "validation"
            or manifest.get("file_count") != len(files)
        ):
            raise ValueError("external-validation dataset manifest identity is invalid")
        episode_indices = []
        for record in files:
            relative = record.get("path") if isinstance(record, dict) else None
            if (
                isinstance(relative, str)
                and relative.startswith("episode_")
                and relative.endswith(".npz")
            ):
                stem = relative.removeprefix("episode_").removesuffix(".npz")
                if len(stem) != 7 or not stem.isdigit():
                    raise ValueError("external-validation episode filename is malformed")
                episode_indices.append(int(stem))
        expected = list(range(*self.source_episode))
        if episode_indices != expected:
            raise ValueError("external-validation manifest differs from its source episode")
        return manifest

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_manifest_path": self.dataset_manifest_path,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "physical_sidecar_name": self.physical_sidecar_name,
            "physical_sidecar_manifest_sha256": self.physical_sidecar_manifest_sha256,
            "target_probe_path": self.target_probe_path,
            "target_probe_sha256": self.target_probe_sha256,
            "source_episode": list(self.source_episode),
            "acceptance_policy": self.acceptance_policy,
        }


def _load_target_probe(
    *,
    repository_root: str | Path,
    relative_path: str,
    expected_sha256: str,
    expected_frame_count: int,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError("M2 source-coverage target probe is absent or escaped")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("M2 source-coverage target-probe SHA-256 changed")
    try:
        report = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("M2 source-coverage target probe is invalid JSON") from error
    if (
        not isinstance(report, dict)
        or report.get("schema") != "picf-next.calvin-molmoact2-target-probe.v3"
        or report.get("status") != "PASS"
        or report.get("coverage") != "all_source_frames"
        or report.get("runtime_target_leakage") is not False
        or report.get("frame_count") != expected_frame_count
        or report.get("objects_without_supervised_ownership_mass") != 0
    ):
        raise ValueError("M2 source-coverage target probe did not pass its exact contract")
    return report


@dataclass(frozen=True, slots=True)
class MolmoAct2M2SourceCoverageRecipe:
    base_m2_recipe_path: str
    base_m2_recipe_sha256: str
    physical_sidecar_name: str
    physical_sidecar_manifest_sha256: str
    target_probe_path: str
    target_probe_sha256: str
    split: M2SourceCoverageSplit
    external_validation: M2SourceExternalValidation
    neutral_task_contract: str = M2_SOURCE_NEUTRAL_TASK_CONTRACT
    historical_comparison_scope: str = M2_SOURCE_COMPARISON_SCOPE
    schema: str = M2_SOURCE_COVERAGE_SCHEMA
    gate: str = M2_SOURCE_COVERAGE_GATE

    def __post_init__(self) -> None:
        if self.schema != M2_SOURCE_COVERAGE_SCHEMA or self.gate != M2_SOURCE_COVERAGE_GATE:
            raise ValueError("M2 source-coverage schema or gate changed")
        if self.neutral_task_contract != M2_SOURCE_NEUTRAL_TASK_CONTRACT:
            raise ValueError("M2 source-coverage neutral-task contract changed")
        if self.historical_comparison_scope != M2_SOURCE_COMPARISON_SCOPE:
            raise ValueError("M2 source-coverage historical comparison scope changed")
        _relative_path(self.base_m2_recipe_path, "base_m2_recipe_path")
        _sha256(self.base_m2_recipe_sha256, "base_m2_recipe_sha256")
        if (
            not self.physical_sidecar_name
            or "/" in self.physical_sidecar_name
            or self.physical_sidecar_name in {".", ".."}
        ):
            raise ValueError("M2 source-coverage sidecar name must be one path component")
        _sha256(
            self.physical_sidecar_manifest_sha256,
            "physical_sidecar_manifest_sha256",
        )
        _relative_path(self.target_probe_path, "target_probe_path")
        _sha256(self.target_probe_sha256, "target_probe_sha256")

    @property
    def recipe_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def load_base_m2(self, repository_root: str | Path) -> MolmoAct2M2Recipe:
        root = Path(repository_root).resolve()
        path = (root / self.base_m2_recipe_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError("base M2 recipe is absent or escaped the repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.base_m2_recipe_sha256:
            raise ValueError("base M2 recipe SHA-256 changed")
        return load_molmoact2_m2_recipe(path)

    def load_target_probe(self, repository_root: str | Path) -> dict[str, Any]:
        return _load_target_probe(
            repository_root=repository_root,
            relative_path=self.target_probe_path,
            expected_sha256=self.target_probe_sha256,
            expected_frame_count=self.source_frame_count,
        )

    def load_external_target_probe(self, repository_root: str | Path) -> dict[str, Any]:
        return _load_target_probe(
            repository_root=repository_root,
            relative_path=self.external_validation.target_probe_path,
            expected_sha256=self.external_validation.target_probe_sha256,
            expected_frame_count=self.external_validation.frame_count,
        )

    @property
    def source_frame_count(self) -> int:
        return self.split.source_episode[1] - self.split.source_episode[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "gate": self.gate,
            "base_m2_recipe_path": self.base_m2_recipe_path,
            "base_m2_recipe_sha256": self.base_m2_recipe_sha256,
            "physical_sidecar_name": self.physical_sidecar_name,
            "physical_sidecar_manifest_sha256": self.physical_sidecar_manifest_sha256,
            "target_probe_path": self.target_probe_path,
            "target_probe_sha256": self.target_probe_sha256,
            "neutral_task_contract": self.neutral_task_contract,
            "historical_comparison_scope": self.historical_comparison_scope,
            "external_validation": self.external_validation.to_dict(),
            "split": {
                "strategy": self.split.strategy,
                "source_episode": list(self.split.source_episode),
                "train_ranges": [list(value) for value in self.split.train_ranges],
                "validation_ranges": [list(value) for value in self.split.validation_ranges],
                "heldout_ranges": [list(value) for value in self.split.heldout_ranges],
                "guard_ranges": [list(value) for value in self.split.guard_ranges],
                "minimum_guard_frames": self.split.minimum_guard_frames,
            },
        }


def load_molmoact2_m2_source_coverage_recipe(
    path: str | Path,
) -> MolmoAct2M2SourceCoverageRecipe:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("M2 source-coverage recipe cannot be read as JSON") from error
    expected = {
        "schema",
        "gate",
        "base_m2_recipe_path",
        "base_m2_recipe_sha256",
        "physical_sidecar_name",
        "physical_sidecar_manifest_sha256",
        "target_probe_path",
        "target_probe_sha256",
        "neutral_task_contract",
        "historical_comparison_scope",
        "external_validation",
        "split",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("M2 source-coverage recipe fields differ from schema")
    split = raw["split"]
    split_fields = {
        "strategy",
        "source_episode",
        "train_ranges",
        "validation_ranges",
        "heldout_ranges",
        "guard_ranges",
        "minimum_guard_frames",
    }
    if not isinstance(split, dict) or set(split) != split_fields:
        raise ValueError("M2 source-coverage split fields differ from schema")
    external = raw["external_validation"]
    external_fields = {
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "physical_sidecar_name",
        "physical_sidecar_manifest_sha256",
        "target_probe_path",
        "target_probe_sha256",
        "source_episode",
        "acceptance_policy",
    }
    if not isinstance(external, dict) or set(external) != external_fields:
        raise ValueError("M2 source external-validation fields differ from schema")
    minimum_guard = split["minimum_guard_frames"]
    if not isinstance(minimum_guard, int) or isinstance(minimum_guard, bool) or minimum_guard <= 0:
        raise ValueError("minimum_guard_frames must be positive")
    return MolmoAct2M2SourceCoverageRecipe(
        schema=_text(raw["schema"], "schema"),
        gate=_text(raw["gate"], "gate"),
        base_m2_recipe_path=_relative_path(
            raw["base_m2_recipe_path"],
            "base_m2_recipe_path",
        ),
        base_m2_recipe_sha256=_sha256(
            raw["base_m2_recipe_sha256"],
            "base_m2_recipe_sha256",
        ),
        physical_sidecar_name=_text(
            raw["physical_sidecar_name"],
            "physical_sidecar_name",
        ),
        physical_sidecar_manifest_sha256=_sha256(
            raw["physical_sidecar_manifest_sha256"],
            "physical_sidecar_manifest_sha256",
        ),
        target_probe_path=_relative_path(raw["target_probe_path"], "target_probe_path"),
        target_probe_sha256=_sha256(
            raw["target_probe_sha256"],
            "target_probe_sha256",
        ),
        neutral_task_contract=_text(
            raw["neutral_task_contract"],
            "neutral_task_contract",
        ),
        historical_comparison_scope=_text(
            raw["historical_comparison_scope"],
            "historical_comparison_scope",
        ),
        external_validation=M2SourceExternalValidation(
            dataset_manifest_path=_relative_path(
                external["dataset_manifest_path"],
                "external_validation.dataset_manifest_path",
            ),
            dataset_manifest_sha256=_sha256(
                external["dataset_manifest_sha256"],
                "external_validation.dataset_manifest_sha256",
            ),
            physical_sidecar_name=_text(
                external["physical_sidecar_name"],
                "external_validation.physical_sidecar_name",
            ),
            physical_sidecar_manifest_sha256=_sha256(
                external["physical_sidecar_manifest_sha256"],
                "external_validation.physical_sidecar_manifest_sha256",
            ),
            target_probe_path=_relative_path(
                external["target_probe_path"],
                "external_validation.target_probe_path",
            ),
            target_probe_sha256=_sha256(
                external["target_probe_sha256"],
                "external_validation.target_probe_sha256",
            ),
            source_episode=_range(
                external["source_episode"],
                "external_validation.source_episode",
            ),
            acceptance_policy=_text(
                external["acceptance_policy"],
                "external_validation.acceptance_policy",
            ),
        ),
        split=M2SourceCoverageSplit(
            strategy=_text(split["strategy"], "split.strategy"),
            source_episode=_range(split["source_episode"], "split.source_episode"),
            train_ranges=_ranges(split["train_ranges"], "split.train_ranges"),
            validation_ranges=_ranges(
                split["validation_ranges"],
                "split.validation_ranges",
            ),
            heldout_ranges=_ranges(split["heldout_ranges"], "split.heldout_ranges"),
            guard_ranges=_ranges(split["guard_ranges"], "split.guard_ranges"),
            minimum_guard_frames=minimum_guard,
        ),
    )


def m2_source_coverage_report(
    recipe: MolmoAct2M2SourceCoverageRecipe,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    base = recipe.load_base_m2(repository_root)
    target_probe = recipe.load_target_probe(repository_root)
    external_probe = recipe.load_external_target_probe(repository_root)
    external_manifest = recipe.external_validation.load_dataset_manifest(repository_root)
    counts = {
        name: sum(
            stop - start for split, start, stop in recipe.split.learned_ranges if split == name
        )
        for name in ("train", "validation", "heldout")
    }
    return {
        "schema": recipe.schema,
        "gate": recipe.gate,
        "recipe_sha256": recipe.recipe_sha256,
        "base_m2_recipe_sha256": base.recipe_sha256,
        "split_frame_counts": counts,
        "guard_frame_count": sum(stop - start for start, stop in recipe.split.guard_ranges),
        "target_probe_frame_count": target_probe["frame_count"],
        "external_validation": {
            "frame_count": external_probe["frame_count"],
            "dataset_tree_sha256": external_manifest["tree_sha256"],
            "acceptance_policy": recipe.external_validation.acceptance_policy,
            "thresholds": {
                "minimum_exact_count_accuracy": (
                    base.acceptance.minimum_heldout_exact_count_accuracy
                ),
                "minimum_mean_object_dice": base.acceptance.minimum_mean_object_dice,
                "minimum_ownership_accuracy": base.acceptance.minimum_ownership_accuracy,
                "minimum_random_dice_margin": base.acceptance.minimum_random_dice_margin,
                "minimum_ownership_accuracy_improvement_vs_all_context": (
                    base.acceptance.minimum_ownership_accuracy_improvement_vs_all_context
                ),
                "minimum_count_mae_improvement_fraction_vs_random": (
                    base.acceptance.minimum_count_mae_improvement_fraction_vs_random
                ),
                "minimum_geometry_mae_improvement_fraction_vs_random": (
                    base.acceptance.minimum_geometry_mae_improvement_fraction_vs_random
                ),
            },
            "checkpoint_reselection_authorized": False,
            "threshold_reselection_authorized": False,
        },
        "candidate_under_test": (
            "task-independent all-source coverage with corrected token-measurable v3 targets"
        ),
        "historical_comparison_scope": recipe.historical_comparison_scope,
        "declared_differences_vs_historical_sparse_m2": [
            "all-source rather than language-frame coverage",
            "depth-consistent unknown-pixel supervision",
            "token-measurable rather than raw-pixel object inventory",
            "contiguous source-time splits with explicit guard ranges",
        ],
        "single_variable_source_coverage_attribution_authorized": False,
        "unchanged_trainable_runtime_modules": ["projector", "discovery"],
        "long_training_authorized": False,
    }

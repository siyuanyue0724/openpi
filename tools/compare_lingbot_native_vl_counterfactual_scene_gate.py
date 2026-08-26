#!/usr/bin/env python3
"""Apply the preregistered ADR-127 candidate-vs-control scientific gate."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    qwen_grounding_label,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    CALVIN_GROUNDING_FAMILIES,
    native_vl_calvin_task_family,
    native_vl_fixed_x_pair_geometry_metrics,
    native_vl_fixed_x_partition_summary,
    normalize_native_vl_answer,
)
from picf_next.lingbot_native.native_vl_scene_metrics import (
    native_vl_scene_bank_summary,
    normalize_scene_label,
)
from picf_next.lingbot_native.runtime_provenance import (
    ADR127_LINGBOT_RUNTIME_PYTHON_TREE,
    revision_bound_python_source_tree_contract,
)
from picf_next.lingbot_native.vl_cotraining import (
    parse_native_vl_grounding_answer,
    parse_native_vl_scene_grounding_answer,
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
)

TRAINING_SCHEMA = "picf-next.lingbot-native-vl-grounding-adaptation.v9"
FIXED_X_SCHEMA = "picf-next.lingbot-native-vl-fixed-x-g0.v8"
ADR125_FIXED_X_SCHEMA = "picf-next.lingbot-native-vl-fixed-x-g0.v3"
ADR126_FIXED_LABEL_SCHEMA = "picf-next.lingbot-restored-qwen-grounding-g0.v1"
FIXED_LABEL_SCHEMA = "picf-next.lingbot-restored-qwen-grounding-g0.v2"
OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-counterfactual-scene-gate.v4"

ADR125_REFERENCE_FIXED_X_SHA256 = "82572e4f43c03d99b5cb1dc9038dc1410b2196f0966fc25f8513cdbb3b2513ef"
ADR126_REFERENCE_FIXED_LABEL_SHA256 = (
    "094209f8516fe59b6e4f24ca617badb7410cdfeb050a0d64c3bcbcff29c51844"
)
ADR127_INITIAL_QWEN_REVISION = "0196dc7bb23f3c742616147c3254d0e4f1207787"
ADR127_MAX_STEPS = 64
ADR127_SCHEDULE_TOTAL_STEPS = 432
ADR127_SCENE_PAIR_COUNT = 32
ADR127_TARGET_PAIR_COUNT = 136
ADR127_TARGET_VARIANT_COUNT = 272
ADR127_FIXED_LABEL_COUNT = 16
ADR127_CLEAN_COLLISION_GLOBAL_INDEX = 1_513_384
ADR127_GRADIENT_AUDIT_STEPS = (0, 21, 42, 63)
ADR127_GRADIENT_OBJECTIVES = ("target", "scene", "public")
ADR127_GRADIENT_WEIGHTS = {"target": 0.5, "scene": 0.5, "public": 0.1}
ADR127_PIXEL_CAUSAL_IDENTITIES = (
    "movable/block_blue",
    "movable/block_pink",
    "movable/block_red",
    "part/table/button_link",
    "part/table/slide_link",
    "part/table/switch_link",
)

TARGET_THRESHOLDS = {
    "center_hit_count_exclusive": 212,
    "bidirectional_own_only_count_exclusive": 76,
    "mean_iou_exclusive": 0.553421,
    "exact_label_count_inclusive": 249,
}
FAMILY_MINIMUM_HITS = {
    "block": 107,
    "drawer": 37,
    "slider": 25,
    "led": 34,
    "lightbulb": 12,
}
PUBLIC_NLL_CEILINGS = {"referring": 0.670161, "vqa": 0.345908}
FIXED_LABEL_MEAN_FLOORS = {
    "mean_token_margin": 0.155940,
    "mean_sequence_margin": 4.912097,
}

_TRAINING_COMMON_FIELDS = (
    "cuda_allocator",
    "dataset_manifest_sha256",
    "fsdp2_placement",
    "initial_qwen",
    "native_vl_patch_sha256",
    "observation_mode",
    "optimizer",
    "optimizer_state_parameter_count",
    "physical_sidecar_manifest_sha256",
    "picf_code_revision",
    "processor_lattices",
    "processor_snapshot_size",
    "public_vl_retention",
    "runtime_python_trees",
    "seed",
    "source_commit",
    "teacher_prune",
    "trainable_scope",
    "training_plan",
    "world_size",
)
_FIXED_X_COMMON_FIELDS = (
    "checkpoint_model_file_sha256",
    "dataset_manifest_sha256",
    "eligible_item_count",
    "evaluation_plan_artifact_sha256",
    "evaluation_plan_file_sha256",
    "excluded_items",
    "item_limit_per_partition",
    "max_new_tokens",
    "native_vl_patch_sha256",
    "partition",
    "picf_code_revision",
    "physical_sidecar_manifest_sha256",
    "processor_lattice",
    "runtime_python_trees",
    "seed",
    "selected_item_count",
    "source_commit",
)
_TARGET_GROUP_BINDING_FIELDS = (
    "ordinal",
    "partition",
    "source_global_index",
    "source_rgb_sha256",
    "source_state_sha256",
)
_TARGET_VARIANT_BINDING_FIELDS = (
    "camera_name",
    "grounding_request",
    "grounding_request_sha256",
    "instruction",
    "instruction_sha256",
    "target_answer",
    "target_answer_sha256",
    "target_bbox_qwen_xyxy",
    "target_identity_key",
    "target_label",
    "task_key",
)
_PUBLIC_BINDING_FIELDS = (
    "family",
    "image_height",
    "image_rgb_sha256",
    "image_grid_thw",
    "image_width",
    "record_id",
    "record_sha256",
    "source_row_index",
    "source_subindex",
    "supervised_token_count",
    "target_answer",
    "target_answer_sha256",
    "user_text",
    "user_text_sha256",
)
_FIXED_LABEL_BINDING_FIELDS = (
    "bbox_xyxy",
    "camera_name",
    "global_index",
    "host_image_key",
    "instruction",
    "source_rgb_sha256",
    "target_identity_key",
    "task_key",
)
_FIXED_LABEL_COMMON_FIELDS = (
    "dataset_manifest_sha256",
    "native_vl_patch_sha256",
    "record_count",
    "source_commit",
    "uniform_vocabulary_nll",
    "visual_sha256",
    "vocabulary_size",
)
_PUBLIC_TRAINING_RANK_BINDING_FIELDS = (
    "family",
    "grid_budget",
    "image_height",
    "image_rgb_sha256",
    "image_grid_thw",
    "image_width",
    "loss_weight",
    "rank",
    "record_id",
    "record_sha256",
    "source_row_index",
    "source_subindex",
    "supervised_token_count",
    "target_answer_sha256",
    "user_text",
    "user_text_sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adr125-reference-fixed-x-report", required=True, type=Path)
    parser.add_argument("--adr126-reference-fixed-label-report", required=True, type=Path)
    for arm in ("control", "candidate"):
        for kind in ("training", "fixed-x", "fixed-label"):
            parser.add_argument(f"--{arm}-{kind}-report", required=True, type=Path)
            parser.add_argument(f"--{arm}-{kind}-report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"ADR-127 comparison {name} must be one lowercase SHA-256")
    return value


def _require_revision(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"ADR-127 comparison {name} must be one lowercase Git revision")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"ADR-127 comparison {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _list(value: object, *, name: str, length: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" of length {length}"
        raise ContractError(f"ADR-127 comparison {name} must be a list{suffix}")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"ADR-127 comparison {name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"ADR-127 comparison {name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or (minimum is not None and output < minimum):
        raise ContractError(f"ADR-127 comparison {name} is outside its finite domain")
    return output


def _text(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\0" in value or (not allow_empty and not value):
        raise ContractError(f"ADR-127 comparison {name} must be valid text")
    return value


def _text_digest(row: Mapping[str, Any], field: str) -> str:
    value = _text(row.get(field), name=field)
    digest = _require_sha256(row.get(f"{field}_sha256"), name=f"{field} digest")
    if not hmac.compare_digest(hashlib.sha256(value.encode()).hexdigest(), digest):
        raise ContractError(f"ADR-127 comparison {field} digest changed")
    return value


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ContractError("ADR-127 comparison value is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _read_regular_input(path: Path) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical.anchor, directory_flags)
        for part in lexical.parent.parts[1:]:
            try:
                child = os.open(part, directory_flags | nofollow, dir_fd=descriptor)
            finally:
                os.close(descriptor)
            descriptor = child
    except OSError as error:
        raise ContractError("ADR-127 comparison input path traverses a symlink") from error
    try:
        try:
            file_descriptor = os.open(
                lexical.name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
                dir_fd=descriptor,
            )
        except OSError as error:
            raise ContractError("ADR-127 comparison input is not a readable real file") from error
    finally:
        os.close(descriptor)
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("ADR-127 comparison input is not a regular file")
        blocks = []
        while block := os.read(file_descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(file_descriptor)
    finally:
        os.close(file_descriptor)
    payload = b"".join(blocks)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(
        payload
    ) != after.st_size:
        raise ContractError("ADR-127 comparison input changed while being read")
    return payload


def _load_report(
    path: Path,
    *,
    expected_sha256: str,
    schemas: str | Sequence[str],
    require_pass: bool = False,
) -> tuple[dict[str, Any], str]:
    expected = _require_sha256(expected_sha256, name="input report digest")
    payload = _read_regular_input(path)
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ContractError("ADR-127 comparison input report digest changed")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ContractError(f"ADR-127 comparison JSON has duplicate key: {key}")
            output[key] = value
        return output

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError("ADR-127 comparison input is invalid strict JSON") from error
    accepted = (schemas,) if isinstance(schemas, str) else tuple(schemas)
    if not isinstance(value, dict) or value.get("schema") not in accepted:
        raise ContractError("ADR-127 comparison input schema changed")
    if require_pass and value.get("status") != "PASS":
        raise ContractError("ADR-127 comparison training input did not pass execution")
    return cast(dict[str, Any], value), actual


def _model_hashes(value: object, *, name: str) -> dict[str, str]:
    mapping = _mapping(value, name=name)
    if not mapping:
        raise ContractError(f"ADR-127 comparison {name} is empty")
    output = {}
    for path, digest in mapping.items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ContractError(f"ADR-127 comparison {name} has an unsafe path")
        output[path] = _require_sha256(digest, name=f"{name} {path}")
    return dict(sorted(output.items()))


def _runtime_python_trees(value: object, *, name: str) -> dict[str, dict[str, object]]:
    trees = _mapping(value, name=f"{name} runtime Python trees")
    if set(trees) != {"lingbot", "picf"}:
        raise ContractError(f"ADR-127 {name} runtime Python tree set changed")
    output = {}
    for tree_name, raw_contract in trees.items():
        contract = _mapping(raw_contract, name=f"{name} {tree_name} runtime Python tree")
        if set(contract) != {"file_count", "tree_sha256"}:
            raise ContractError(f"ADR-127 {name} runtime Python tree contract changed")
        output[tree_name] = {
            "file_count": _integer(
                contract.get("file_count"),
                name=f"{name} {tree_name} runtime file count",
                minimum=1,
            ),
            "tree_sha256": _require_sha256(
                contract.get("tree_sha256"), name=f"{name} {tree_name} runtime tree"
            ),
        }
    return dict(sorted(output.items()))


def _validate_report_runtime_trees(
    report: Mapping[str, Any],
    *,
    expected: Mapping[str, Mapping[str, object]],
    name: str,
) -> None:
    actual = _runtime_python_trees(report.get("runtime_python_trees"), name=name)
    if actual != expected:
        raise ContractError(f"ADR-127 {name} runtime Python trees differ from trusted checkout")


def _evaluation_qwen_hashes(report: Mapping[str, Any], *, name: str) -> dict[str, str]:
    restore = _mapping(report.get("qwen_restore"), name=f"{name} Qwen restore")
    return _model_hashes(restore.get("model_file_sha256"), name=f"{name} Qwen weights")


def _check(*, actual: object, passed: bool, relation: str, threshold: object) -> dict[str, object]:
    return {"actual": actual, "passed": bool(passed), "relation": relation, "threshold": threshold}


def _training_rank_binding(row: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(
        row.get(field)
        for field in (
            "rank",
            "camera_name",
            "global_index",
            "image_grid_thw",
            "source_rgb_sha256",
            "visual_lattice",
        )
    )


def _training_public_binding(step: Mapping[str, Any]) -> tuple[str, str]:
    section = _mapping(step.get("public_vl_retention"), name="training public factor")
    ranks = _list(section.get("ranks"), name="training public ranks", length=2)
    output = []
    for expected_rank, raw_row in enumerate(ranks):
        row = _mapping(raw_row, name="training public rank")
        if row.get("rank") != expected_rank:
            raise ContractError("ADR-127 public training rank order changed")
        output.append(
            _canonical_sha256([row.get(field) for field in _PUBLIC_TRAINING_RANK_BINDING_FIELDS])
        )
    return cast(tuple[str, str], tuple(output))


def _gradient_objective_binding(row: Mapping[str, Any], *, objective: str) -> tuple[object, ...]:
    if objective in ("target", "scene"):
        fields = (
            "assistant_text_sha256",
            "camera_name",
            "global_index",
            "image_grid_thw",
            "rank",
            "record_type",
            "source_rgb_sha256",
            "supervised_token_count",
            "user_text_sha256",
            "visual_lattice",
        )
    elif objective == "public":
        fields = (
            "assistant_text_sha256",
            "image_rgb_sha256",
            "image_grid_thw",
            "rank",
            "record_id",
            "record_sha256",
            "supervised_token_count",
            "user_text_sha256",
        )
    else:
        raise ContractError("ADR-127 gradient objective binding is unsupported")
    return tuple(row.get(field) for field in fields)


def _expected_gradient_rank(
    candidate: Mapping[str, Any],
    *,
    step: int,
    objective: str,
    rank: int,
) -> Mapping[str, Any]:
    steps = _list(candidate.get("step_reports"), name="candidate gradient-bound steps", length=64)
    step_row = _mapping(steps[step], name="candidate gradient-bound step")
    if objective == "public":
        public = _mapping(step_row.get("public_vl_retention"), name="gradient-bound public")
        ranks = _list(public.get("ranks"), name="gradient-bound public ranks", length=2)
        return _mapping(ranks[rank], name="gradient-bound public rank")
    batches = _list(step_row.get("microbatches"), name="gradient-bound batches", length=1)
    batch = _mapping(batches[0], name="gradient-bound batch")
    factors = _list(batch.get("factors"), name="gradient-bound factors", length=2)
    factor = next(
        (
            _mapping(item, name="gradient-bound factor")
            for item in factors
            if _mapping(item, name="gradient-bound factor").get("factor_name") == objective
        ),
        None,
    )
    if factor is None:
        raise ContractError("ADR-127 gradient objective has no matching training factor")
    ranks = _list(factor.get("ranks"), name="gradient-bound factor ranks", length=2)
    return _mapping(ranks[rank], name="gradient-bound factor rank")


def _validate_global_gradient_gram(summary: Mapping[str, Any], *, name: str) -> dict[str, object]:
    element_count = _integer(
        summary.get("element_count"), name=f"{name} gradient element count", minimum=1
    )
    parameter_tensor_count = _integer(
        summary.get("parameter_tensor_count"),
        name=f"{name} gradient parameter tensor count",
        minimum=1,
    )
    global_summary = summary
    squared = _mapping(
        global_summary.get("gradient_squared_norms"),
        name="global gradient squared norms",
    )
    dots = _mapping(global_summary.get("pairwise_dot_products"), name="global gradient dots")
    if set(squared) != set(ADR127_GRADIENT_OBJECTIVES) or set(dots) != {
        "target__scene",
        "target__public",
        "scene__public",
    }:
        raise ContractError("ADR-127 global gradient Gram matrix changed")
    values = {
        name: _finite(squared[name], name=f"{name} gradient squared norm", minimum=0.0)
        for name in ADR127_GRADIENT_OBJECTIVES
    }
    norms = _mapping(global_summary.get("gradient_norms"), name="global gradient norms")
    if set(norms) != set(ADR127_GRADIENT_OBJECTIVES):
        raise ContractError("ADR-127 global gradient norm surface changed")
    for name, squared_norm in values.items():
        actual = _finite(norms[name], name=f"{name} gradient norm", minimum=0.0)
        if not math.isclose(actual, math.sqrt(squared_norm), rel_tol=1e-12, abs_tol=1e-12):
            raise ContractError("ADR-127 gradient norm was not Gram-recomputed")
    pair_values = {
        name: _finite(value, name=f"{name} gradient dot") for name, value in dots.items()
    }
    pair_objectives = {
        "target__scene": ("target", "scene"),
        "target__public": ("target", "public"),
        "scene__public": ("scene", "public"),
    }
    for pair_name, (left, right) in pair_objectives.items():
        product = values[left] * values[right]
        dot_squared = pair_values[pair_name] ** 2
        tolerance = 1e-8 * max(product, dot_squared, 1e-300)
        if dot_squared > product + tolerance:
            raise ContractError(f"ADR-127 {name} gradient Gram violates Cauchy-Schwarz")
    target_squared = values["target"]
    scene_squared = values["scene"]
    public_squared = values["public"]
    target_scene = pair_values["target__scene"]
    target_public = pair_values["target__public"]
    scene_public = pair_values["scene__public"]
    determinant_terms = (
        target_squared * scene_squared * public_squared,
        2.0 * target_scene * target_public * scene_public,
        -target_squared * scene_public**2,
        -scene_squared * target_public**2,
        -public_squared * target_scene**2,
    )
    determinant = math.fsum(determinant_terms)
    determinant_tolerance = 1e-8 * max(*(abs(value) for value in determinant_terms), 1e-300)
    if determinant < -determinant_tolerance:
        raise ContractError(f"ADR-127 {name} gradient Gram is not positive semidefinite")
    cosines = _mapping(global_summary.get("pairwise_cosines"), name="global gradient cosines")
    if set(cosines) != set(pair_values):
        raise ContractError("ADR-127 global gradient cosine surface changed")
    for pair_name, (left, right) in pair_objectives.items():
        denominator = math.sqrt(values[left] * values[right])
        expected = (
            None
            if denominator == 0.0
            else max(-1.0, min(1.0, pair_values[pair_name] / denominator))
        )
        actual = cosines[pair_name]
        if expected is None:
            if actual is not None:
                raise ContractError("ADR-127 zero-norm gradient cosine changed")
        elif not math.isclose(
            _finite(actual, name=f"{pair_name} gradient cosine"),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ContractError("ADR-127 gradient cosine was not Gram-recomputed")
    expected_directional = {
        "target": 0.5 * values["target"] + 0.5 * target_scene + 0.1 * target_public,
        "scene": 0.5 * target_scene + 0.5 * values["scene"] + 0.1 * scene_public,
        "public": 0.5 * target_public + 0.5 * scene_public + 0.1 * values["public"],
    }
    directional = _mapping(
        global_summary.get("mixed_gradient_directional_inner_products"),
        name="global gradient directional products",
    )
    descends = _mapping(
        global_summary.get("mixed_gradient_descends"),
        name="global gradient descent directions",
    )
    if set(directional) != set(expected_directional) or set(descends) != set(expected_directional):
        raise ContractError("ADR-127 global gradient directional surface changed")
    for name, expected in expected_directional.items():
        actual = _finite(directional[name], name=f"{name} directional product")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ContractError("ADR-127 gradient directional product was not Gram-recomputed")
        if descends[name] is not (expected > 0.0):
            raise ContractError("ADR-127 gradient direction flag was not Gram-recomputed")
    mixed_squared = math.fsum(
        (
            0.25 * values["target"],
            0.25 * values["scene"],
            0.01 * values["public"],
            0.5 * target_scene,
            0.1 * target_public,
            0.1 * scene_public,
        )
    )
    tolerance = 1e-12 * max(
        abs(mixed_squared),
        values["target"],
        values["scene"],
        values["public"],
        abs(target_scene),
        abs(target_public),
        abs(scene_public),
        1e-300,
    )
    if mixed_squared < -tolerance:
        raise ContractError("ADR-127 global mixed gradient norm is invalid")
    expected_norm = math.sqrt(max(mixed_squared, 0.0))
    actual_norm = _finite(global_summary.get("mixed_gradient_norm"), name="mixed gradient norm")
    if not math.isclose(actual_norm, expected_norm, rel_tol=1e-12, abs_tol=1e-12):
        raise ContractError("ADR-127 mixed gradient norm was not Gram-recomputed")
    return {
        "element_count": element_count,
        "gradient_squared_norms": values,
        "pairwise_dot_products": pair_values,
        "parameter_tensor_count": parameter_tensor_count,
    }


def _validate_counterfactual_gradient_audit(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, object]:
    control_section = _mapping(
        control.get("counterfactual_gradient_audit"),
        name="control gradient audit",
    )
    if dict(control_section) != {"enabled": False}:
        raise ContractError("ADR-127 control must not execute the scene gradient audit")
    section = _mapping(
        candidate.get("counterfactual_gradient_audit"),
        name="candidate gradient audit",
    )
    if (
        set(section)
        != {
            "enabled",
            "objective_weights",
            "reports",
            "status",
            "step_indices",
        }
        or section.get("enabled") is not True
    ):
        raise ContractError("ADR-127 candidate gradient audit contract changed")
    if section.get("objective_weights") != ADR127_GRADIENT_WEIGHTS or section.get(
        "step_indices"
    ) != list(ADR127_GRADIENT_AUDIT_STEPS):
        raise ContractError("ADR-127 candidate gradient objectives changed")
    reports = _list(
        section.get("reports"),
        name="candidate gradient audit reports",
        length=len(ADR127_GRADIENT_AUDIT_STEPS),
    )
    recomputed_pass = True
    report_statuses = []
    for expected_step, raw_report in zip(
        ADR127_GRADIENT_AUDIT_STEPS,
        reports,
        strict=True,
    ):
        report = _mapping(raw_report, name="candidate gradient audit report")
        if report.get("completed_updates_before_audit") != expected_step:
            raise ContractError("ADR-127 candidate gradient audit step changed")
        _finite(report.get("elapsed_seconds"), name="gradient audit elapsed", minimum=0.0)
        objective_reports = _list(
            report.get("objective_rank_reports"),
            name="gradient objective reports",
            length=3,
        )
        for expected_objective, raw_objective in zip(
            ADR127_GRADIENT_OBJECTIVES,
            objective_reports,
            strict=True,
        ):
            objective = _mapping(raw_objective, name="gradient objective report")
            if objective.get("objective") != expected_objective:
                raise ContractError("ADR-127 gradient objective order changed")
            ranks = _list(objective.get("ranks"), name="gradient objective ranks", length=2)
            for expected_rank, raw_rank in enumerate(ranks):
                rank = _mapping(raw_rank, name="gradient objective rank")
                if rank.get("objective") != expected_objective or rank.get("rank") != expected_rank:
                    raise ContractError("ADR-127 gradient objective rank order changed")
                _finite(rank.get("loss"), name="gradient objective loss", minimum=0.0)
                _finite(rank.get("elapsed_seconds"), name="gradient objective elapsed", minimum=0.0)
                _integer(
                    rank.get("supervised_token_count"),
                    name="gradient objective supervised tokens",
                    minimum=1,
                )
                _require_sha256(
                    rank.get("assistant_text_sha256"),
                    name="gradient objective assistant text",
                )
                _require_sha256(
                    rank.get("user_text_sha256"),
                    name="gradient objective user text",
                )
                expected_training_rank = _expected_gradient_rank(
                    candidate,
                    step=expected_step,
                    objective=expected_objective,
                    rank=expected_rank,
                )
                if _gradient_objective_binding(
                    rank,
                    objective=expected_objective,
                ) != _gradient_objective_binding(
                    expected_training_rank,
                    objective=expected_objective,
                ):
                    raise ContractError(
                        "ADR-127 gradient audit record differs from its real training record"
                    )
        summary = _mapping(report.get("summary"), name="gradient audit summary")
        if (
            summary.get("objective_weights") != ADR127_GRADIENT_WEIGHTS
            or summary.get("parameter_count") != 404
        ):
            raise ContractError("ADR-127 gradient summary scope changed")
        global_summary = _mapping(summary.get("global"), name="global gradient summary")
        global_gram = _validate_global_gradient_gram(global_summary, name="global")
        if global_gram["element_count"] != 4_049_739_776:
            raise ContractError("ADR-127 gradient summary element coverage changed")
        if global_gram["parameter_tensor_count"] != summary.get("parameter_count"):
            raise ContractError("ADR-127 gradient summary parameter coverage changed")
        groups = _mapping(summary.get("groups"), name="grouped gradient summary")
        if not {
            "visual_merger",
            "language_embedding_tied_lm_head",
        }.issubset(groups):
            raise ContractError("ADR-127 gradient summary omits a required Qwen surface")
        grouped_grams = {
            group_name: _validate_global_gradient_gram(
                _mapping(raw_group, name=f"{group_name} gradient summary"),
                name=f"group {group_name}",
            )
            for group_name, raw_group in groups.items()
        }
        if (
            sum(cast(int, group["element_count"]) for group in grouped_grams.values())
            != global_gram["element_count"]
            or sum(cast(int, group["parameter_tensor_count"]) for group in grouped_grams.values())
            != global_gram["parameter_tensor_count"]
        ):
            raise ContractError("ADR-127 grouped gradient coverage does not sum to global")
        for field in ("gradient_squared_norms", "pairwise_dot_products"):
            global_values = cast(Mapping[str, float], global_gram[field])
            for value_name, global_value in global_values.items():
                grouped_value = math.fsum(
                    cast(Mapping[str, float], group[field])[value_name]
                    for group in grouped_grams.values()
                )
                if not math.isclose(
                    grouped_value,
                    global_value,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ContractError("ADR-127 grouped gradient moments do not sum to global")
        descends = _mapping(
            global_summary.get("mixed_gradient_descends"),
            name="global gradient descent directions",
        )
        if set(descends) != set(ADR127_GRADIENT_OBJECTIVES) or any(
            not isinstance(descends[name], bool) for name in ADR127_GRADIENT_OBJECTIVES
        ):
            raise ContractError("ADR-127 global gradient directions are malformed")
        directional = _mapping(
            global_summary.get("mixed_gradient_directional_inner_products"),
            name="global gradient directional products",
        )
        if set(directional) != set(ADR127_GRADIENT_OBJECTIVES):
            raise ContractError("ADR-127 global gradient directional products changed")
        for name in ADR127_GRADIENT_OBJECTIVES:
            value = _finite(directional[name], name=f"{name} directional product")
            if descends[name] is not (value > 0.0):
                raise ContractError("ADR-127 gradient direction flag was not recomputed")
        local_pass = descends["target"] and descends["scene"]
        expected_status = "PASS" if local_pass else "FAIL"
        if report.get("status") != expected_status:
            raise ContractError("ADR-127 per-step gradient audit status changed")
        report_statuses.append(expected_status)
        recomputed_pass = recomputed_pass and local_pass
    expected_status = "PASS" if recomputed_pass else "FAIL"
    if section.get("status") != expected_status:
        raise ContractError("ADR-127 aggregate gradient audit status changed")
    return {
        "passed": recomputed_pass,
        "report_statuses": report_statuses,
        "status": expected_status,
    }


def _validate_training_pair(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_runtime_python_trees: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if control.get("schema") != TRAINING_SCHEMA or candidate.get("schema") != TRAINING_SCHEMA:
        raise ContractError("ADR-127 training schema changed")
    if control.get("status") != "PASS" or candidate.get("status") != "PASS":
        raise ContractError("ADR-127 requires two passing training executions")
    if any(control.get(field) != candidate.get(field) for field in _TRAINING_COMMON_FIELDS):
        raise ContractError("ADR-127 training arms are not configuration matched")
    if control.get("world_size") != 2 or control.get("optimizer") != "torch.optim.AdamW":
        raise ContractError("ADR-127 training topology changed")
    if control.get("seed") != 20260801:
        raise ContractError("ADR-127 training seed changed")
    for field in (
        "dataset_manifest_sha256",
        "native_vl_patch_sha256",
        "physical_sidecar_manifest_sha256",
    ):
        _require_sha256(control.get(field), name=f"training {field}")
    for field in ("picf_code_revision", "source_commit"):
        _require_revision(control.get(field), name=f"training {field}")
    _validate_report_runtime_trees(
        control,
        expected=expected_runtime_python_trees,
        name="training",
    )
    _require_sha256(
        _mapping(control.get("training_plan"), name="training plan").get("artifact_sha256"),
        name="training plan artifact",
    )
    _require_sha256(
        _mapping(control.get("public_vl_retention"), name="training public retention").get(
            "artifact_sha256"
        ),
        name="training public-retention artifact",
    )
    scope = _mapping(control.get("trainable_scope"), name="training trainable scope")
    if (
        scope.get("parameter_count") != 404
        or scope.get("trainable_numel") != 4_049_739_776
        or control.get("optimizer_state_parameter_count") != 404
    ):
        raise ContractError("ADR-127 training trainable scope changed")
    hyper = _mapping(control.get("hyperparameters"), name="training hyperparameters")
    expected_hyper = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "learning_rate": 1e-6,
        "max_grad_norm": 1.0,
        "max_steps": ADR127_MAX_STEPS,
        "schedule": "linear-warmup-cosine-decay",
        "schedule_total_steps": ADR127_SCHEDULE_TOTAL_STEPS,
        "warmup_steps": 0,
        "weight_decay": 0.0,
    }
    if dict(hyper) != expected_hyper or candidate.get("hyperparameters") != control.get(
        "hyperparameters"
    ):
        raise ContractError("ADR-127 training hyperparameters changed")
    expected_factors = {
        "control": {
            "adr127_smoke": False,
            "mode": "target_repeat_control",
            "rank_factor_weights_before_microbatch_average": {"target": 0.5, "target_repeat": 0.5},
        },
        "candidate": {
            "adr127_smoke": False,
            "mode": "counterfactual_scene_candidate",
            "rank_factor_weights_before_microbatch_average": {"scene": 0.5, "target": 0.5},
        },
    }
    if control.get("calvin_factor_contract") != expected_factors["control"]:
        raise ContractError("ADR-127 control factor contract changed")
    if candidate.get("calvin_factor_contract") != expected_factors["candidate"]:
        raise ContractError("ADR-127 candidate factor contract changed")
    gradient_audit = _validate_counterfactual_gradient_audit(control, candidate)
    initial = _mapping(control.get("initial_qwen"), name="training initial Qwen")
    if initial.get("revision") != ADR127_INITIAL_QWEN_REVISION:
        raise ContractError("ADR-127 initial Qwen revision changed")
    initial_hashes = _model_hashes(initial.get("model_file_sha256"), name="initial Qwen")
    control_output = _model_hashes(
        control.get("candidate_model_file_sha256"), name="control output Qwen"
    )
    candidate_output = _model_hashes(
        candidate.get("candidate_model_file_sha256"), name="candidate output Qwen"
    )
    if control_output == candidate_output or initial_hashes in (control_output, candidate_output):
        raise ContractError("ADR-127 training did not produce two distinct updated arms")

    control_steps = _list(control.get("step_reports"), name="control steps", length=64)
    candidate_steps = _list(candidate.get("step_reports"), name="candidate steps", length=64)
    for step_index, (control_step, candidate_step) in enumerate(
        zip(control_steps, candidate_steps, strict=True)
    ):
        left = _mapping(control_step, name="control step")
        right = _mapping(candidate_step, name="candidate step")
        for field in (
            "optimizer_step",
            "curriculum_group_index",
            "curriculum_optimizer_step",
            "observation_mode",
            "learning_rate",
        ):
            if left.get(field) != right.get(field):
                raise ContractError(f"ADR-127 step {step_index} schedule/source binding changed")
        if (
            left.get("optimizer_step") != step_index
            or left.get("curriculum_optimizer_step") != step_index
        ):
            raise ContractError("ADR-127 training step order changed")
        if _training_public_binding(left) != _training_public_binding(right):
            raise ContractError("ADR-127 public training records differ across arms")
        for arm_name, step, expected_names in (
            ("control", left, ("target", "target_repeat")),
            ("candidate", right, ("target", "scene")),
        ):
            batches = _list(step.get("microbatches"), name=f"{arm_name} microbatches", length=1)
            batch = _mapping(batches[0], name=f"{arm_name} microbatch")
            if batch.get("visual_lattice") != 8:
                raise ContractError("ADR-127 training lattice changed")
            factors = _list(batch.get("factors"), name=f"{arm_name} factors", length=2)
            if (
                tuple(_mapping(item, name="factor").get("factor_name") for item in factors)
                != expected_names
            ):
                raise ContractError(f"ADR-127 {arm_name} factor order changed")
            for expected_name, factor in zip(expected_names, factors, strict=True):
                factor = _mapping(factor, name=f"{arm_name} factor")
                if factor.get("loss_weight") != 0.5:
                    raise ContractError("ADR-127 CALVIN factor weight changed")
                ranks = _list(factor.get("ranks"), name=f"{arm_name} factor ranks", length=2)
                for rank_index, rank in enumerate(ranks):
                    rank = _mapping(rank, name=f"{arm_name} rank")
                    if rank.get("rank") != rank_index or rank.get("factor_name") != expected_name:
                        raise ContractError("ADR-127 factor rank order changed")
                    _finite(rank.get("loss"), name="factor loss", minimum=0.0)
                    _integer(
                        rank.get("supervised_token_count"), name="supervised tokens", minimum=1
                    )
                    _require_sha256(rank.get("assistant_text_sha256"), name="assistant text")
                    _require_sha256(rank.get("source_rgb_sha256"), name="training source RGB")
                    _require_sha256(rank.get("user_text_sha256"), name="training user text")

        control_factors = cast(
            list[Any],
            cast(Mapping[str, Any], control_steps[step_index]["microbatches"][0])["factors"],
        )
        candidate_factors = cast(
            list[Any],
            cast(Mapping[str, Any], candidate_steps[step_index]["microbatches"][0])["factors"],
        )
        control_target = _list(
            _mapping(control_factors[0], name="control target").get("ranks"),
            name="control target ranks",
            length=2,
        )
        control_repeat = _list(
            _mapping(control_factors[1], name="control repeat").get("ranks"),
            name="control repeat ranks",
            length=2,
        )
        candidate_target = _list(
            _mapping(candidate_factors[0], name="candidate target").get("ranks"),
            name="candidate target ranks",
            length=2,
        )
        candidate_scene = _list(
            _mapping(candidate_factors[1], name="candidate scene").get("ranks"),
            name="candidate scene ranks",
            length=2,
        )
        for rank_index in range(2):
            target = _mapping(control_target[rank_index], name="control target rank")
            repeat = _mapping(control_repeat[rank_index], name="control repeat rank")
            other_target = _mapping(candidate_target[rank_index], name="candidate target rank")
            scene = _mapping(candidate_scene[rank_index], name="candidate scene rank")
            target_binding = _training_rank_binding(target)
            if target_binding != _training_rank_binding(
                repeat
            ) or target_binding != _training_rank_binding(other_target):
                raise ContractError("ADR-127 target/repeat/candidate source records differ")
            target_specific = (
                "instruction",
                "task_key",
                "target_identity_key",
                "assistant_text_sha256",
                "user_text_sha256",
            )
            if any(
                target.get(field) != repeat.get(field)
                or target.get(field) != other_target.get(field)
                for field in target_specific
            ):
                raise ContractError("ADR-127 target factor serialization changed across arms")
            if _training_rank_binding(scene) != target_binding:
                raise ContractError("ADR-127 scene factor does not use the target observation")
            expected_order = (
                list(CALVIN_QWEN_SCENE_IDENTITY_ORDER)
                if rank_index == 0
                else list(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
            )
            if scene.get("category_identity_order") != expected_order:
                raise ContractError("ADR-127 scene category counterfactual changed")
            objects = _list(scene.get("object_identity_keys"), name="scene objects")
            absent = _list(scene.get("absent_identity_keys"), name="scene absent objects")
            subpatch = _list(
                scene.get("subpatch_visible_identity_keys"), name="scene subpatch objects"
            )
            if len(objects) != len(set(objects)) or set(objects) | set(absent) | set(
                subpatch
            ) != set(CALVIN_QWEN_SCENE_IDENTITY_ORDER):
                raise ContractError("ADR-127 scene inventory partition changed")
        first_scene = _mapping(candidate_scene[0], name="rank-zero scene")
        second_scene = _mapping(candidate_scene[1], name="rank-one scene")
        if first_scene.get("source_rgb_sha256") != second_scene.get("source_rgb_sha256"):
            raise ContractError("ADR-127 scene rank source image changed")
        for field in (
            "object_identity_keys",
            "absent_identity_keys",
            "subpatch_visible_identity_keys",
        ):
            first_values = _list(first_scene.get(field), name=f"rank-zero {field}")
            second_values = _list(second_scene.get(field), name=f"rank-one {field}")
            if second_values != list(reversed(first_values)):
                raise ContractError("ADR-127 scene evidence did not reverse with category order")
        for step in (left, right):
            gradient = _mapping(step.get("gradient_metrics"), name="training gradients")
            if (
                gradient.get("all_finite") is not True
                or _integer(gradient.get("trainable_gradient_elements"), name="gradient coverage")
                != 4_049_739_776
                or gradient.get("frozen_gradient_elements") != 0
            ):
                raise ContractError("ADR-127 training gradient contract failed")
    return {
        "candidate_output_model_file_sha256": candidate_output,
        "control_output_model_file_sha256": control_output,
        "initial_model_file_sha256": initial_hashes,
        "counterfactual_gradient_audit": gradient_audit,
        "physical_sidecar_manifest_sha256": control.get("physical_sidecar_manifest_sha256"),
        "picf_code_revision": control.get("picf_code_revision"),
        "step_count": 64,
    }


def _target_binding(groups: list[Any]) -> tuple[str, ...]:
    bindings = []
    for raw_group in groups:
        group = _mapping(raw_group, name="target group")
        _require_sha256(group.get("source_rgb_sha256"), name="target source RGB")
        _require_sha256(group.get("source_state_sha256"), name="target source state")
        variants = _list(group.get("variants"), name="target variants", length=2)
        variant_bindings = []
        for raw_variant in variants:
            variant = _mapping(raw_variant, name="target variant")
            _text_digest(variant, "instruction")
            _text_digest(variant, "grounding_request")
            _text_digest(variant, "target_answer")
            variant_bindings.append(
                tuple(variant.get(field) for field in _TARGET_VARIANT_BINDING_FIELDS)
            )
        bindings.append(
            _canonical_sha256(
                [
                    *(group.get(field) for field in _TARGET_GROUP_BINDING_FIELDS),
                    variant_bindings,
                ]
            )
        )
    if len(bindings) != len(set(bindings)):
        raise ContractError("ADR-127 target bank has duplicate bound rows")
    return tuple(bindings)


def _target_metrics(report: Mapping[str, Any]) -> dict[str, object]:
    groups = _list(report.get("results"), name="target results", length=ADR127_TARGET_PAIR_COUNT)
    family_hits = {family: 0 for family in CALVIN_GROUNDING_FAMILIES}
    exact_labels = 0
    normalized_groups = []
    collision_score = None
    for raw_group in groups:
        group = _mapping(raw_group, name="target group")
        variants = _list(group.get("variants"), name="target variants", length=2)
        predictions = []
        targets = []
        normalized_variants = []
        for raw_variant in variants:
            variant = _mapping(raw_variant, name="target variant")
            generated_text = _text(
                variant.get("generated_text"), name="target generation", allow_empty=True
            )
            parsed = parse_native_vl_grounding_answer(generated_text)
            prediction = parsed.bbox_qwen_xyxy
            reported_prediction = variant.get("generated_bbox_qwen_xyxy")
            expected_prediction = None if prediction is None else list(prediction)
            if (
                reported_prediction != expected_prediction
                or variant.get("generated_bbox_schema_valid") is not parsed.schema_valid
            ):
                raise ContractError("ADR-127 target generation was not parsed faithfully")
            target_raw = _list(variant.get("target_bbox_qwen_xyxy"), name="target bbox", length=4)
            target = tuple(_integer(value, name="target coordinate") for value in target_raw)
            if (
                any(value > 1000 for value in target)
                or target[0] >= target[2]
                or target[1] >= target[3]
            ):
                raise ContractError("ADR-127 target bbox is outside normalized Qwen geometry")
            target_answer = parse_native_vl_grounding_answer(
                _text(variant.get("target_answer"), name="target answer")
            )
            if (
                not target_answer.schema_valid
                or target_answer.bbox_qwen_xyxy != target
                or target_answer.generated_label != variant.get("target_label")
            ):
                raise ContractError("ADR-127 target answer does not bind its label and bbox")
            generated_label = parsed.generated_label
            normalized_exact = (
                parsed.label_schema_valid
                and generated_label is not None
                and normalize_native_vl_answer(generated_label)
                == normalize_native_vl_answer(
                    _text(variant.get("target_label"), name="target label")
                )
            )
            if (
                variant.get("generated_label") != generated_label
                or variant.get("generated_label_present") is not parsed.label_present
                or variant.get("generated_label_schema_valid") is not parsed.label_schema_valid
                or variant.get("normalized_label_exact_match") is not normalized_exact
            ):
                raise ContractError("ADR-127 target semantic flags were not recomputed")
            exact_labels += int(normalized_exact)
            predictions.append(prediction)
            targets.append(cast(tuple[int, int, int, int], target))
        pair = native_vl_fixed_x_pair_geometry_metrics(
            cast(tuple[Any, Any], tuple(predictions)), cast(tuple[Any, Any], tuple(targets))
        )
        reported_pair = _mapping(group.get("pair_metrics"), name="target pair metrics")
        if dict(reported_pair) != {key: value for key, value in pair.items() if key != "variants"}:
            raise ContractError("ADR-127 target pair metrics were not recomputed")
        pair_variants = _list(pair.get("variants"), name="target pair metric variants", length=2)
        for raw_variant, metric in zip(variants, pair_variants, strict=True):
            variant = _mapping(raw_variant, name="target variant")
            metric = _mapping(metric, name="target metric")
            for field, value in metric.items():
                if variant.get(field) != value:
                    raise ContractError("ADR-127 target variant geometry was not recomputed")
            family = native_vl_calvin_task_family(
                _text(variant.get("task_key"), name="task key"),
                _text(variant.get("target_identity_key"), name="target identity"),
            )
            family_hits[family] += int(metric.get("own_target_center_hit") is True)
            normalized_variants.append({**dict(variant), **dict(metric)})
        normalized_group = {**dict(group), "variants": normalized_variants}
        normalized_groups.append(normalized_group)
        if group.get("source_global_index") == ADR127_CLEAN_COLLISION_GLOBAL_INDEX:
            identities = {variant.get("target_identity_key") for variant in variants}
            if identities == {"part/table/button_link", "part/table/switch_link"}:
                collision_score = sum(
                    metric.get("own_only_center_hit") is True for metric in pair_variants
                )
    if collision_score is None:
        raise ContractError("ADR-127 target bank omitted the clean switch/button collision")
    summary = native_vl_fixed_x_partition_summary(normalized_groups)
    if _integer(summary.get("variant_count"), name="target variant count") != 272:
        raise ContractError("ADR-127 target variant cardinality changed")
    return {
        "bidirectional_own_only_count": summary["bidirectional_own_only_center_hit_count"],
        "center_hit_count": summary["own_target_center_hit_count"],
        "clean_collision_own_only_count": collision_score,
        "exact_label_count": exact_labels,
        "family_hits": family_hits,
        "mean_iou": summary["mean_own_target_iou"],
    }


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_contains_point(bbox: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _pixel_causal_metrics(report: Mapping[str, Any]) -> dict[str, object]:
    """Select output-independent same-prompt sentinels and test pixel dependence."""

    groups = _list(report.get("results"), name="pixel-causal target results", length=136)
    by_prompt: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for raw_group in groups:
        group = _mapping(raw_group, name="pixel-causal target group")
        partition = _text(group.get("partition"), name="pixel-causal partition")
        global_index = _integer(group.get("source_global_index"), name="pixel-causal global index")
        source_rgb_sha256 = _require_sha256(
            group.get("source_rgb_sha256"), name="pixel-causal source RGB"
        )
        for raw_variant in _list(group.get("variants"), name="pixel-causal variants", length=2):
            variant = _mapping(raw_variant, name="pixel-causal variant")
            task_key = _text(variant.get("task_key"), name="pixel-causal task")
            identity_key = _text(variant.get("target_identity_key"), name="pixel-causal identity")
            camera_name = _text(variant.get("camera_name"), name="pixel-causal camera")
            instruction = _text_digest(variant, "instruction")
            instruction_sha256 = _require_sha256(
                variant.get("instruction_sha256"), name="pixel-causal instruction"
            )
            target_raw = _list(
                variant.get("target_bbox_qwen_xyxy"),
                name="pixel-causal target bbox",
                length=4,
            )
            target = cast(
                tuple[int, int, int, int],
                tuple(
                    _integer(value, name="pixel-causal target coordinate") for value in target_raw
                ),
            )
            if (
                any(value > 1000 for value in target)
                or target[0] >= target[2]
                or target[1] >= target[3]
            ):
                raise ContractError("ADR-127 pixel-causal target bbox is invalid")
            generated_text = _text(
                variant.get("generated_text"),
                name="pixel-causal generation",
                allow_empty=True,
            )
            parsed = parse_native_vl_grounding_answer(generated_text)
            expected_label = _text(variant.get("target_label"), name="pixel-causal label")
            label_exact = (
                parsed.label_schema_valid
                and parsed.generated_label is not None
                and normalize_native_vl_answer(parsed.generated_label)
                == normalize_native_vl_answer(expected_label)
            )
            key = (
                partition,
                task_key,
                identity_key,
                camera_name,
                instruction_sha256,
            )
            by_prompt[key].append(
                {
                    "camera_name": camera_name,
                    "generated_bbox_qwen_xyxy": parsed.bbox_qwen_xyxy,
                    "global_index": global_index,
                    "identity_key": identity_key,
                    "instruction": instruction,
                    "instruction_sha256": instruction_sha256,
                    "label_exact": label_exact,
                    "partition": partition,
                    "schema_valid": parsed.schema_valid,
                    "source_rgb_sha256": source_rgb_sha256,
                    "target_bbox_qwen_xyxy": target,
                    "task_key": task_key,
                }
            )

    eligible_by_identity: dict[str, list[tuple[float, tuple[object, ...], tuple[Any, Any]]]] = (
        defaultdict(list)
    )
    for prompt_key, records in by_prompt.items():
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                if (
                    left["global_index"] == right["global_index"]
                    or left["source_rgb_sha256"] == right["source_rgb_sha256"]
                ):
                    continue
                left_target = cast(tuple[int, int, int, int], left["target_bbox_qwen_xyxy"])
                right_target = cast(tuple[int, int, int, int], right["target_bbox_qwen_xyxy"])
                left_center = _bbox_center(left_target)
                right_center = _bbox_center(right_target)
                if _bbox_contains_point(left_target, right_center) or _bbox_contains_point(
                    right_target, left_center
                ):
                    continue
                ordered = tuple(
                    sorted(
                        (left, right),
                        key=lambda row: (
                            cast(int, row["global_index"]),
                            cast(str, row["source_rgb_sha256"]),
                        ),
                    )
                )
                distance_squared = (left_center[0] - right_center[0]) ** 2 + (
                    left_center[1] - right_center[1]
                ) ** 2
                binding = (
                    *prompt_key,
                    *(cast(int, row["global_index"]) for row in ordered),
                    *(cast(str, row["source_rgb_sha256"]) for row in ordered),
                    *(
                        cast(tuple[int, int, int, int], row["target_bbox_qwen_xyxy"])
                        for row in ordered
                    ),
                )
                identity_key = cast(str, left["identity_key"])
                eligible_by_identity[identity_key].append(
                    (distance_squared, binding, cast(tuple[Any, Any], ordered))
                )

    if set(eligible_by_identity) != set(ADR127_PIXEL_CAUSAL_IDENTITIES):
        raise ContractError("ADR-127 pixel-causal sentinel identity coverage changed")
    sentinels = []
    pass_count = 0
    for identity_key in ADR127_PIXEL_CAUSAL_IDENTITIES:
        choices = eligible_by_identity[identity_key]
        distance_squared, binding, pair = min(
            choices,
            key=lambda item: (-item[0], _canonical_sha256(item[1])),
        )
        endpoint_rows = []
        endpoint_passes = []
        for endpoint_index, row in enumerate(pair):
            counterpart = pair[1 - endpoint_index]
            prediction = cast(
                tuple[int, int, int, int] | None,
                row["generated_bbox_qwen_xyxy"],
            )
            target = cast(tuple[int, int, int, int], row["target_bbox_qwen_xyxy"])
            counterpart_target = cast(
                tuple[int, int, int, int], counterpart["target_bbox_qwen_xyxy"]
            )
            own_target_center_hit = prediction is not None and _bbox_contains_point(
                prediction, _bbox_center(target)
            )
            generated_center_hits_own_target = prediction is not None and _bbox_contains_point(
                target, _bbox_center(prediction)
            )
            excludes_counterpart_target_center = (
                prediction is not None
                and not _bbox_contains_point(prediction, _bbox_center(counterpart_target))
            )
            endpoint_pass = (
                row["label_exact"] is True
                and row["schema_valid"] is True
                and own_target_center_hit
                and generated_center_hits_own_target
                and excludes_counterpart_target_center
            )
            endpoint_passes.append(endpoint_pass)
            endpoint_rows.append(
                {
                    "excludes_counterpart_target_center": excludes_counterpart_target_center,
                    "generated_bbox_qwen_xyxy": (None if prediction is None else list(prediction)),
                    "generated_center_hits_own_target": generated_center_hits_own_target,
                    "global_index": row["global_index"],
                    "label_exact": row["label_exact"],
                    "own_target_center_hit": own_target_center_hit,
                    "schema_valid": row["schema_valid"],
                    "source_rgb_sha256": row["source_rgb_sha256"],
                    "target_bbox_qwen_xyxy": list(target),
                }
            )
        sentinel_pass = all(endpoint_passes)
        pass_count += int(sentinel_pass)
        sentinels.append(
            {
                "binding_sha256": _canonical_sha256(binding),
                "camera_name": pair[0]["camera_name"],
                "center_distance_squared": distance_squared,
                "eligible_pair_count": len(choices),
                "endpoints": endpoint_rows,
                "identity_key": identity_key,
                "instruction": pair[0]["instruction"],
                "partition": pair[0]["partition"],
                "sentinel_pass": sentinel_pass,
                "task_key": pair[0]["task_key"],
            }
        )
    return {
        "binding": tuple(row["binding_sha256"] for row in sentinels),
        "pass_count": pass_count,
        "sentinel_count": len(sentinels),
        "sentinels": sentinels,
    }


def _scene_metrics(report: Mapping[str, Any]) -> dict[str, object]:
    section = _mapping(report.get("scene_evaluation"), name="scene evaluation")
    if section.get("enabled") is not True or section.get("source_disjoint_scene_bank_count") != 32:
        raise ContractError("ADR-127 source-disjoint scene evaluation is incomplete")
    _require_sha256(section.get("audit_artifact_sha256"), name="scene audit artifact")
    _require_sha256(section.get("audit_file_sha256"), name="scene audit file")
    max_new_tokens = _integer(
        section.get("max_new_tokens"), name="scene maximum new tokens", minimum=1
    )
    generation_budget = _mapping(section.get("generation_budget"), name="scene generation budget")
    expected_budget_keys = {
        "configured_max_new_tokens",
        "headroom_tokens",
        "maximum_target_supervised_tokens",
        "minimum_target_supervised_tokens",
        "target_record_count",
    }
    if set(generation_budget) != expected_budget_keys:
        raise ContractError("ADR-127 scene generation budget fields are invalid")
    budget_configured = _integer(
        generation_budget.get("configured_max_new_tokens"),
        name="scene configured generation budget",
        minimum=1,
    )
    budget_minimum = _integer(
        generation_budget.get("minimum_target_supervised_tokens"),
        name="scene minimum target tokens",
        minimum=1,
    )
    budget_required = _integer(
        generation_budget.get("maximum_target_supervised_tokens"),
        name="scene maximum target tokens",
        minimum=1,
    )
    budget_headroom = _integer(
        generation_budget.get("headroom_tokens"),
        name="scene generation headroom",
        minimum=0,
    )
    budget_records = _integer(
        generation_budget.get("target_record_count"),
        name="scene target token records",
        minimum=1,
    )
    if (
        budget_configured != max_new_tokens
        or budget_records != 2 * ADR127_SCENE_PAIR_COUNT
        or budget_minimum > budget_required
        or budget_required > max_new_tokens
        or budget_headroom != max_new_tokens - budget_required
    ):
        raise ContractError("ADR-127 scene generation budget is unsound")
    rows = _list(section.get("results"), name="scene results", length=32)
    bindings = []
    for expected_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, name="scene row")
        if row.get("bank_index") != expected_index:
            raise ContractError("ADR-127 scene bank order changed")
        _require_sha256(row.get("source_rgb_sha256"), name="scene source RGB")
        metrics = _mapping(row.get("pair_metrics"), name="scene pair metrics")
        variants = _list(metrics.get("variants"), name="scene variants", length=2)
        parsed_maps = []
        for index, raw_variant in enumerate(variants):
            variant = _mapping(raw_variant, name="scene variant")
            expected_identity_order = (
                CALVIN_QWEN_SCENE_IDENTITY_ORDER
                if index == 0
                else tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
            )
            if variant.get("category_identity_order") != list(expected_identity_order):
                raise ContractError("ADR-127 scene category identity order changed")
            generated_text = _text(
                variant.get("generated_text"), name="scene generation", allow_empty=True
            )
            if hashlib.sha256(generated_text.encode()).hexdigest() != _require_sha256(
                variant.get("generated_text_sha256"), name="scene generation"
            ):
                raise ContractError("ADR-127 scene generation digest changed")
            _text_digest(variant, "grounding_request")
            target_text = _text_digest(variant, "target_answer")
            parsed = parse_native_vl_scene_grounding_answer(generated_text)
            target = parse_native_vl_scene_grounding_answer(target_text)
            if not target.schema_valid:
                raise ContractError("ADR-127 scene target answer is invalid")
            labels = [normalize_scene_label(item.label) for item in parsed.objects]
            boxes = {
                normalize_scene_label(item.label): list(item.bbox_qwen_xyxy)
                for item in parsed.objects
            }
            expected_labels = [normalize_scene_label(item.label) for item in target.objects]
            expected_boxes = {
                normalize_scene_label(item.label): item.bbox_qwen_xyxy for item in target.objects
            }
            if len(expected_labels) < 2:
                raise ContractError("ADR-127 scene evaluation is not multi-object")
            expected_set = set(expected_labels)
            predicted_set = set(labels)
            full_label_order = [
                normalize_scene_label(qwen_grounding_label(identity_key))
                for identity_key in expected_identity_order
            ]
            if expected_labels != [label for label in full_label_order if label in expected_set]:
                raise ContractError("ADR-127 scene target answer order changed")
            if (
                variant.get("schema_valid") is not parsed.schema_valid
                or variant.get("generated_label_order") != labels
                or variant.get("generated_object_count") != len(labels)
                or len(labels) != len(set(labels))
                or variant.get("expected_label_order") != expected_labels
                or variant.get("expected_object_count") != len(expected_labels)
                or variant.get("label_set_exact") is not (predicted_set == expected_set)
                or variant.get("missing_labels") != sorted(expected_set - predicted_set)
                or variant.get("extra_labels") != sorted(predicted_set - expected_set)
                or variant.get("order_exact") is not (labels == expected_labels)
                or variant.get("order_variant") != ("canonical" if index == 0 else "reverse")
            ):
                raise ContractError("ADR-127 scene parse evidence changed")
            object_rows = _list(variant.get("objects"), name="scene object rows")
            if len(object_rows) != len(expected_labels):
                raise ContractError("ADR-127 scene object metric cardinality changed")
            by_label = {
                normalize_scene_label(
                    _text(
                        _mapping(row, name="scene object row").get("label"),
                        name="scene label",
                    )
                ): _mapping(row, name="scene object row")
                for row in object_rows
            }
            if set(by_label) != expected_set:
                raise ContractError("ADR-127 scene object metric labels changed")
            for label, expected_bbox in expected_boxes.items():
                object_row = by_label[label]
                expected_identity = next(
                    identity_key
                    for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER
                    if normalize_scene_label(qwen_grounding_label(identity_key)) == label
                )
                if object_row.get("identity_key") != expected_identity or object_row.get(
                    "target_bbox_qwen_xyxy"
                ) != list(expected_bbox):
                    raise ContractError("ADR-127 scene object target binding changed")
                prediction = next(
                    (
                        item.bbox_qwen_xyxy
                        for item in parsed.objects
                        if normalize_scene_label(item.label) == label
                    ),
                    None,
                )
                if object_row.get("generated_bbox_qwen_xyxy") != (
                    None if prediction is None else list(prediction)
                ) or object_row.get("label_found") is not (prediction is not None):
                    raise ContractError("ADR-127 scene label-addressed prediction changed")
                hit = (
                    False
                    if prediction is None
                    else qwen_target_center_in_bbox(prediction, expected_bbox)
                )
                generated_center_hit = (
                    False
                    if prediction is None
                    else qwen_target_center_in_bbox(expected_bbox, prediction)
                )
                iou = (
                    0.0
                    if prediction is None
                    else qwen_grounding_bbox_iou(prediction, expected_bbox)
                )
                unexpected_center_hit_labels = []
                if prediction is not None:
                    for other_label, other_bbox in expected_boxes.items():
                        if other_label == label:
                            continue
                        ground_truth_overlaps_other_center = qwen_target_center_in_bbox(
                            expected_bbox,
                            other_bbox,
                        )
                        prediction_hits_other_center = qwen_target_center_in_bbox(
                            prediction,
                            other_bbox,
                        )
                        if prediction_hits_other_center and not ground_truth_overlaps_other_center:
                            unexpected_center_hit_labels.append(other_label)
                center_selective = hit and generated_center_hit and not unexpected_center_hit_labels
                if (
                    object_row.get("target_center_hit") is not hit
                    or object_row.get("generated_center_hit") is not generated_center_hit
                    or object_row.get("unexpected_center_hit_labels")
                    != unexpected_center_hit_labels
                    or object_row.get("center_selective") is not center_selective
                    or not math.isclose(
                        _finite(object_row.get("target_iou"), name="scene target IoU"),
                        iou,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ContractError("ADR-127 scene object geometry was not recomputed")
            parsed_maps.append(boxes)
        pair_pass = (
            all(
                _mapping(item, name="scene variant").get("expected_object_count", 0) >= 2
                for item in variants
            )
            and all(
                _mapping(item, name="scene variant").get("schema_valid") is True
                for item in variants
            )
            and all(
                _mapping(item, name="scene variant").get("label_set_exact") is True
                for item in variants
            )
            and all(
                _mapping(item, name="scene variant").get("order_exact") is True for item in variants
            )
            and all(
                all(
                    _mapping(row, name="scene object row").get("center_selective") is True
                    for row in _list(
                        _mapping(item, name="scene variant").get("objects"),
                        name="scene object rows",
                    )
                )
                for item in variants
            )
            and parsed_maps[0] == parsed_maps[1]
        )
        if (
            metrics.get("label_box_map_exact") is not (parsed_maps[0] == parsed_maps[1])
            or metrics.get("pair_pass") is not pair_pass
        ):
            raise ContractError("ADR-127 scene pair result was not recomputed")
        bindings.append(
            (
                row.get("bank_index"),
                row.get("group_index"),
                row.get("source_global_index"),
                row.get("source_rgb_sha256"),
                tuple(row.get("task_keys", [])),
                tuple(
                    tuple(_mapping(item, name="scene variant").get("category_identity_order", []))
                    for item in variants
                ),
                tuple(
                    _mapping(item, name="scene variant").get("grounding_request_sha256")
                    for item in variants
                ),
                tuple(
                    _mapping(item, name="scene variant").get("target_answer_sha256")
                    for item in variants
                ),
            )
        )
    if len(bindings) != len(set(bindings)):
        raise ContractError("ADR-127 scene bank contains duplicate rows")
    summary = native_vl_scene_bank_summary(cast(Sequence[Mapping[str, object]], rows))
    if dict(_mapping(section.get("summary"), name="scene summary")) != summary:
        raise ContractError("ADR-127 scene summary was not recomputed")
    per_identity = _mapping(summary.get("per_identity"), name="scene per-identity summary")
    if set(per_identity) != set(CALVIN_QWEN_SCENE_IDENTITY_ORDER):
        raise ContractError("ADR-127 scene bank does not evaluate every registered identity")
    if any(
        _integer(
            _mapping(value, name="scene identity summary").get("expected_count"),
            name="scene identity expected count",
            minimum=1,
        )
        < 1
        for value in per_identity.values()
    ):
        raise ContractError("ADR-127 scene identity evaluation is empty")
    return {
        "audit_artifact_sha256": section["audit_artifact_sha256"],
        "audit_file_sha256": section["audit_file_sha256"],
        "binding": tuple(bindings),
        "generation_budget": dict(generation_budget),
        **summary,
    }


def _public_metrics(report: Mapping[str, Any]) -> dict[str, object]:
    section = _mapping(report.get("public_vl_retention"), name="public evaluation")
    if section.get("enabled") is not True or section.get("heldout_limit_per_family") != 32:
        raise ContractError("ADR-127 public evaluation is incomplete")
    artifact_sha256 = _require_sha256(
        section.get("artifact_sha256"), name="public evaluation artifact"
    )
    manifest_file_sha256 = _require_sha256(
        section.get("manifest_file_sha256"), name="public evaluation manifest"
    )
    rows = _list(section.get("results"), name="public results", length=64)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    binding = []
    discrete = {}
    for raw_row in rows:
        row = _mapping(raw_row, name="public row")
        family = _text(row.get("family"), name="public family")
        if family not in PUBLIC_NLL_CEILINGS:
            raise ContractError("ADR-127 public family changed")
        _finite(row.get("mean_token_nll"), name="public NLL", minimum=0.0)
        _integer(row.get("supervised_token_count"), name="public tokens", minimum=1)
        _text_digest(row, "target_answer")
        _text_digest(row, "user_text")
        _require_sha256(row.get("record_sha256"), name="public record")
        _require_sha256(row.get("image_rgb_sha256"), name="public image")
        key = tuple(row.get(field) for field in _PUBLIC_BINDING_FIELDS)
        binding.append(_canonical_sha256(key))
        by_family[family].append(row)
        if family == "vqa":
            expected = normalize_native_vl_answer(
                _text(row.get("target_answer"), name="VQA target")
            )
            generated = normalize_native_vl_answer(
                _text(row.get("generated_text"), name="VQA generation", allow_empty=True)
            )
            flag = generated == expected
            if row.get("normalized_exact_match") is not flag:
                raise ContractError("ADR-127 VQA exact match was not recomputed")
            discrete[(family, row.get("record_id"))] = flag
        else:
            parsed = parse_native_vl_grounding_answer(
                _text(row.get("generated_text"), name="RefCOCO generation", allow_empty=True)
            )
            if (
                row.get("generated_bbox_qwen_xyxy")
                != (None if parsed.bbox_qwen_xyxy is None else list(parsed.bbox_qwen_xyxy))
                or row.get("generated_bbox_schema_valid") is not parsed.schema_valid
            ):
                raise ContractError("ADR-127 RefCOCO generation was not parsed faithfully")
            target = parse_native_vl_grounding_answer(
                _text(row.get("target_answer"), name="RefCOCO target")
            )
            if not target.schema_valid or target.bbox_qwen_xyxy is None:
                raise ContractError("ADR-127 RefCOCO target answer is invalid")
            center_hit = (
                parsed.schema_valid
                and parsed.bbox_qwen_xyxy is not None
                and qwen_target_center_in_bbox(parsed.bbox_qwen_xyxy, target.bbox_qwen_xyxy)
            )
            if row.get("target_center_hit") is not center_hit:
                raise ContractError("ADR-127 RefCOCO centre hit was not recomputed")
            discrete[(family, row.get("record_id"))] = (
                parsed.schema_valid,
                center_hit,
            )
    if len(binding) != len(set(binding)):
        raise ContractError("ADR-127 public bank contains duplicate rows")
    nll = {}
    for family, family_rows in by_family.items():
        token_count = sum(int(row["supervised_token_count"]) for row in family_rows)
        nll[family] = (
            math.fsum(
                float(row["mean_token_nll"]) * int(row["supervised_token_count"])
                for row in family_rows
            )
            / token_count
        )
    return {
        "artifact_sha256": artifact_sha256,
        "binding": tuple(binding),
        "discrete": discrete,
        "manifest_file_sha256": manifest_file_sha256,
        "token_weighted_nll": nll,
    }


def _scored_answer(answer: Mapping[str, Any], *, name: str) -> tuple[float, float]:
    mean = _finite(answer.get("mean_token_nll"), name=f"{name} mean NLL", minimum=0.0)
    tokens = _integer(answer.get("supervised_token_count"), name=f"{name} tokens", minimum=1)
    sequence = _finite(answer.get("sequence_nll"), name=f"{name} sequence NLL", minimum=0.0)
    if not math.isclose(sequence, mean * tokens, rel_tol=1e-10, abs_tol=1e-10):
        raise ContractError(f"ADR-127 {name} sequence NLL was not recomputed")
    return mean, sequence


def _fixed_label_metrics(report: Mapping[str, Any]) -> dict[str, object]:
    if (
        report.get("schema")
        not in {
            ADR126_FIXED_LABEL_SCHEMA,
            FIXED_LABEL_SCHEMA,
        }
        or report.get("record_count") != 16
    ):
        raise ContractError("ADR-127 fixed-label report contract changed")
    rows = _list(report.get("records"), name="fixed-label records", length=16)
    for field in ("dataset_manifest_sha256", "native_vl_patch_sha256"):
        _require_sha256(report.get(field), name=f"fixed-label {field}")
    _require_revision(report.get("source_commit"), name="fixed-label source commit")
    vocabulary_size = _integer(
        report.get("vocabulary_size"), name="fixed-label vocabulary size", minimum=1
    )
    uniform_nll = _finite(
        report.get("uniform_vocabulary_nll"), name="fixed-label uniform vocabulary NLL"
    )
    if not math.isclose(uniform_nll, math.log(vocabulary_size), rel_tol=1e-10, abs_tol=1e-10):
        raise ContractError("ADR-127 fixed-label uniform vocabulary NLL changed")
    _model_hashes(report.get("visual_sha256"), name="fixed-label visuals")
    bindings = []
    metrics = {}
    token_values = []
    sequence_values = []
    for raw_row in rows:
        row = _mapping(raw_row, name="fixed-label row")
        _require_sha256(row.get("source_rgb_sha256"), name="fixed-label source RGB")
        correct = _mapping(row.get("correct_answer"), name="correct answer")
        correct_mean, correct_sequence = _scored_answer(correct, name="correct answer")
        distractors = _list(row.get("distractors"), name="fixed-label distractors")
        if not distractors:
            raise ContractError("ADR-127 fixed-label row has no distractor")
        distractor_bindings = []
        mean_losses = []
        sequence_losses = []
        for raw_distractor in distractors:
            distractor = _mapping(raw_distractor, name="fixed-label distractor")
            mean, sequence = _scored_answer(distractor, name="distractor")
            identity = _text(distractor.get("distractor_identity_key"), name="distractor identity")
            distractor_bindings.append(
                (
                    identity,
                    distractor.get("assistant_text"),
                    tuple(distractor.get("bbox_xyxy", [])),
                    tuple(distractor.get("qwen_bbox_xyxy", [])),
                    distractor.get("supervised_token_count"),
                )
            )
            mean_losses.append(mean)
            sequence_losses.append(sequence)
        token_margin = min(mean_losses) - correct_mean
        sequence_margin = min(sequence_losses) - correct_sequence
        if not math.isclose(
            _finite(row.get("nll_margin"), name="token margin"),
            token_margin,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ContractError("ADR-127 fixed-label token margin was not recomputed")
        if not math.isclose(
            _finite(row.get("sequence_nll_margin"), name="sequence margin"),
            sequence_margin,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ContractError("ADR-127 fixed-label sequence margin was not recomputed")
        key = (
            row.get("global_index"),
            row.get("camera_name"),
            row.get("task_key"),
            row.get("target_identity_key"),
        )
        if key in metrics:
            raise ContractError("ADR-127 fixed-label record key is duplicated")
        metrics[key] = {"sequence_margin": sequence_margin, "token_margin": token_margin}
        bindings.append(
            (
                *tuple(row.get(field) for field in _FIXED_LABEL_BINDING_FIELDS),
                correct.get("assistant_text"),
                tuple(correct.get("bbox_xyxy", [])),
                tuple(correct.get("qwen_bbox_xyxy", [])),
                correct.get("supervised_token_count"),
                tuple(distractor_bindings),
            )
        )
        token_values.append(token_margin)
        sequence_values.append(sequence_margin)
    return {
        "binding": tuple(bindings),
        "mean_sequence_margin": math.fsum(sequence_values) / len(sequence_values),
        "mean_token_margin": math.fsum(token_values) / len(token_values),
        "metrics": metrics,
    }


def _validate_model_bindings(
    training: Mapping[str, object],
    control_fixed_x: Mapping[str, Any],
    candidate_fixed_x: Mapping[str, Any],
    control_fixed_label: Mapping[str, Any],
    candidate_fixed_label: Mapping[str, Any],
) -> None:
    pairs = (
        (training["control_output_model_file_sha256"], control_fixed_x, control_fixed_label),
        (training["candidate_output_model_file_sha256"], candidate_fixed_x, candidate_fixed_label),
    )
    for expected, fixed_x, fixed_label in pairs:
        if _evaluation_qwen_hashes(fixed_x, name="fixed-X") != expected:
            raise ContractError("ADR-127 fixed-X report is not its arm's training output")
        if _evaluation_qwen_hashes(fixed_label, name="fixed-label") != expected:
            raise ContractError("ADR-127 fixed-label report is not its arm's training output")
        expected_revision = training["picf_code_revision"]
        if fixed_x.get("picf_code_revision") != expected_revision:
            raise ContractError("ADR-127 fixed-X code revision differs from training")
        if fixed_label.get("picf_code_revision") != expected_revision:
            raise ContractError("ADR-127 fixed-label code revision differs from training")
        for report, name in ((fixed_x, "fixed-X"), (fixed_label, "fixed-label")):
            restore = _mapping(report.get("qwen_restore"), name=f"{name} Qwen restore")
            if restore.get("model_revision") != expected_revision:
                raise ContractError(f"ADR-127 {name} Qwen revision differs from training")


def _validate_fixed_x_source(
    report: Mapping[str, Any],
    *,
    expected_runtime_python_trees: Mapping[str, Mapping[str, object]],
    name: str,
) -> None:
    _model_hashes(report.get("checkpoint_model_file_sha256"), name=f"{name} base checkpoint")
    for field in (
        "dataset_manifest_sha256",
        "evaluation_plan_artifact_sha256",
        "evaluation_plan_file_sha256",
        "native_vl_patch_sha256",
        "physical_sidecar_manifest_sha256",
    ):
        _require_sha256(report.get(field), name=f"{name} {field}")
    for field in ("picf_code_revision", "source_commit"):
        _require_revision(report.get(field), name=f"{name} {field}")
    _validate_report_runtime_trees(
        report,
        expected=expected_runtime_python_trees,
        name=name,
    )


def compare_lingbot_native_vl_counterfactual_scene_gate(
    *,
    adr125_reference_fixed_x: Mapping[str, Any],
    adr126_reference_fixed_label: Mapping[str, Any],
    control_training: Mapping[str, Any],
    candidate_training: Mapping[str, Any],
    control_fixed_x: Mapping[str, Any],
    candidate_fixed_x: Mapping[str, Any],
    control_fixed_label: Mapping[str, Any],
    candidate_fixed_label: Mapping[str, Any],
    expected_runtime_python_trees: Mapping[str, Mapping[str, object]],
    input_report_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Recompute every ADR-127 threshold and return a terminal PASS or FAIL."""

    digests = {
        key: _require_sha256(value, name=f"{key} report")
        for key, value in input_report_sha256.items()
    }
    required_digest_keys = {
        "adr125_reference_fixed_x",
        "adr126_reference_fixed_label",
        "control_training",
        "candidate_training",
        "control_fixed_x",
        "candidate_fixed_x",
        "control_fixed_label",
        "candidate_fixed_label",
    }
    if set(digests) != required_digest_keys:
        raise ContractError("ADR-127 comparison input digest inventory changed")
    if (
        digests["adr125_reference_fixed_x"] != ADR125_REFERENCE_FIXED_X_SHA256
        or digests["adr126_reference_fixed_label"] != ADR126_REFERENCE_FIXED_LABEL_SHA256
    ):
        raise ContractError("ADR-127 historical reference report changed")
    if adr125_reference_fixed_x.get("schema") != ADR125_FIXED_X_SCHEMA:
        raise ContractError("ADR-127 ADR-125 reference schema changed")

    expected_runtime = _runtime_python_trees(
        expected_runtime_python_trees,
        name="trusted checkout",
    )
    if expected_runtime["lingbot"] != ADR127_LINGBOT_RUNTIME_PYTHON_TREE:
        raise ContractError("ADR-127 trusted LingBot runtime tree differs from preregistration")
    training = _validate_training_pair(
        control_training,
        candidate_training,
        expected_runtime_python_trees=expected_runtime,
    )
    if (
        control_fixed_x.get("schema") != FIXED_X_SCHEMA
        or candidate_fixed_x.get("schema") != FIXED_X_SCHEMA
    ):
        raise ContractError("ADR-127 fixed-X schema changed")
    _validate_fixed_x_source(
        control_fixed_x,
        expected_runtime_python_trees=expected_runtime,
        name="control fixed-X",
    )
    _validate_fixed_x_source(
        candidate_fixed_x,
        expected_runtime_python_trees=expected_runtime,
        name="candidate fixed-X",
    )
    if any(
        control_fixed_x.get(field) != candidate_fixed_x.get(field)
        for field in _FIXED_X_COMMON_FIELDS
    ):
        raise ContractError("ADR-127 fixed-X arms are not source matched")
    if (
        control_fixed_x.get("physical_sidecar_manifest_sha256")
        != training["physical_sidecar_manifest_sha256"]
    ):
        raise ContractError("ADR-127 evaluation sidecar differs from training")
    control_groups = _list(control_fixed_x.get("results"), name="control target bank", length=136)
    candidate_groups = _list(
        candidate_fixed_x.get("results"), name="candidate target bank", length=136
    )
    if _target_binding(control_groups) != _target_binding(candidate_groups):
        raise ContractError("ADR-127 fixed-X arms evaluated different target records")
    _validate_model_bindings(
        training,
        control_fixed_x,
        candidate_fixed_x,
        control_fixed_label,
        candidate_fixed_label,
    )

    control_target = _target_metrics(control_fixed_x)
    candidate_target = _target_metrics(candidate_fixed_x)
    candidate_center_hits = _integer(
        candidate_target["center_hit_count"], name="candidate center hits", minimum=0
    )
    candidate_bidirectional = _integer(
        candidate_target["bidirectional_own_only_count"],
        name="candidate bidirectional hits",
        minimum=0,
    )
    candidate_mean_iou = _finite(
        candidate_target["mean_iou"], name="candidate mean IoU", minimum=0.0
    )
    candidate_exact_labels = _integer(
        candidate_target["exact_label_count"], name="candidate exact labels", minimum=0
    )
    candidate_clean_collision = _integer(
        candidate_target["clean_collision_own_only_count"],
        name="candidate clean collision count",
        minimum=0,
    )
    control_clean_collision = _integer(
        control_target["clean_collision_own_only_count"],
        name="control clean collision count",
        minimum=0,
    )
    target_checks = {
        "center_hits_strictly_exceed_adr126": _check(
            actual=candidate_center_hits,
            passed=candidate_center_hits > TARGET_THRESHOLDS["center_hit_count_exclusive"],
            relation=">",
            threshold=TARGET_THRESHOLDS["center_hit_count_exclusive"],
        ),
        "bidirectional_own_only_strictly_exceeds_adr126": _check(
            actual=candidate_bidirectional,
            passed=candidate_bidirectional
            > TARGET_THRESHOLDS["bidirectional_own_only_count_exclusive"],
            relation=">",
            threshold=TARGET_THRESHOLDS["bidirectional_own_only_count_exclusive"],
        ),
        "mean_iou_strictly_exceeds_adr126": _check(
            actual=candidate_mean_iou,
            passed=candidate_mean_iou > TARGET_THRESHOLDS["mean_iou_exclusive"],
            relation=">",
            threshold=TARGET_THRESHOLDS["mean_iou_exclusive"],
        ),
        "exact_labels_meet_adr126": _check(
            actual=candidate_exact_labels,
            passed=candidate_exact_labels >= TARGET_THRESHOLDS["exact_label_count_inclusive"],
            relation=">=",
            threshold=TARGET_THRESHOLDS["exact_label_count_inclusive"],
        ),
    }
    family_checks = {
        family: _check(
            actual=cast(Mapping[str, int], candidate_target["family_hits"])[family],
            passed=cast(Mapping[str, int], candidate_target["family_hits"])[family] >= minimum,
            relation=">=",
            threshold=minimum,
        )
        for family, minimum in FAMILY_MINIMUM_HITS.items()
    }
    collision_checks = {
        "candidate_clean_collision_complete": _check(
            actual=candidate_clean_collision,
            passed=candidate_clean_collision == 2,
            relation="==",
            threshold=2,
        ),
        "candidate_beats_control_clean_collision": _check(
            actual={
                "candidate": candidate_clean_collision,
                "control": control_clean_collision,
            },
            passed=candidate_clean_collision > control_clean_collision,
            relation="candidate > control",
            threshold=None,
        ),
    }

    control_pixel_causal = _pixel_causal_metrics(control_fixed_x)
    candidate_pixel_causal = _pixel_causal_metrics(candidate_fixed_x)
    if control_pixel_causal["binding"] != candidate_pixel_causal["binding"]:
        raise ContractError("ADR-127 pixel-causal arms selected different sentinels")
    pixel_causal_checks = {
        "all_candidate_same_prompt_position_sentinels_pass": _check(
            actual=candidate_pixel_causal["pass_count"],
            passed=candidate_pixel_causal["pass_count"] == candidate_pixel_causal["sentinel_count"],
            relation="== sentinel_count",
            threshold=candidate_pixel_causal["sentinel_count"],
        )
    }

    control_scene = _scene_metrics(control_fixed_x)
    candidate_scene = _scene_metrics(candidate_fixed_x)
    if (
        control_scene["audit_artifact_sha256"] != candidate_scene["audit_artifact_sha256"]
        or control_scene["audit_file_sha256"] != candidate_scene["audit_file_sha256"]
        or control_scene["binding"] != candidate_scene["binding"]
        or control_scene["generation_budget"] != candidate_scene["generation_budget"]
    ):
        raise ContractError(
            "ADR-127 scene arms evaluated different source-disjoint banks or generation budgets"
        )
    candidate_scene_pairs = _integer(
        candidate_scene["pair_pass_count"], name="candidate scene pairs", minimum=0
    )
    control_scene_pairs = _integer(
        control_scene["pair_pass_count"], name="control scene pairs", minimum=0
    )
    scene_checks = {
        "all_candidate_generated_centers_hit_targets": _check(
            actual=candidate_scene["generated_center_hit_count"],
            passed=candidate_scene["generated_center_hit_count"]
            == candidate_scene["object_prediction_count"],
            relation="== object_prediction_count",
            threshold=candidate_scene["object_prediction_count"],
        ),
        "all_candidate_scene_objects_center_selective": _check(
            actual=candidate_scene["center_selective_count"],
            passed=candidate_scene["center_selective_count"]
            == candidate_scene["object_prediction_count"],
            relation="== object_prediction_count",
            threshold=candidate_scene["object_prediction_count"],
        ),
        "all_candidate_scene_pairs_pass": _check(
            actual=candidate_scene_pairs,
            passed=candidate_scene_pairs == ADR127_SCENE_PAIR_COUNT,
            relation="==",
            threshold=ADR127_SCENE_PAIR_COUNT,
        ),
        "all_candidate_scene_generations_schema_valid": _check(
            actual=candidate_scene["schema_valid_count"],
            passed=candidate_scene["schema_valid_count"] == 2 * ADR127_SCENE_PAIR_COUNT,
            relation="==",
            threshold=2 * ADR127_SCENE_PAIR_COUNT,
        ),
        "candidate_scene_has_no_unexpected_center_hits": _check(
            actual=candidate_scene["unexpected_center_hit_count"],
            passed=candidate_scene["unexpected_center_hit_count"] == 0,
            relation="==",
            threshold=0,
        ),
        "candidate_beats_control_scene_order_equivariance": _check(
            actual={
                "candidate": candidate_scene_pairs,
                "control": control_scene_pairs,
            },
            passed=candidate_scene_pairs > control_scene_pairs,
            relation="candidate > control",
            threshold=None,
        ),
    }

    reference_public = _public_metrics(adr125_reference_fixed_x)
    control_public = _public_metrics(control_fixed_x)
    candidate_public = _public_metrics(candidate_fixed_x)
    if (
        control_public["artifact_sha256"] != candidate_public["artifact_sha256"]
        or candidate_public["artifact_sha256"] != reference_public["artifact_sha256"]
        or control_public["manifest_file_sha256"] != candidate_public["manifest_file_sha256"]
        or candidate_public["manifest_file_sha256"] != reference_public["manifest_file_sha256"]
        or control_public["binding"] != candidate_public["binding"]
        or candidate_public["binding"] != reference_public["binding"]
    ):
        raise ContractError("ADR-127 public evaluation bank changed from ADR-125")
    public_checks = {}
    for family, ceiling in PUBLIC_NLL_CEILINGS.items():
        value = cast(Mapping[str, float], candidate_public["token_weighted_nll"])[family]
        public_checks[f"{family}_token_weighted_nll_ceiling"] = _check(
            actual=value, passed=value <= ceiling, relation="<=", threshold=ceiling
        )
    reference_discrete = cast(Mapping[object, object], reference_public["discrete"])
    candidate_discrete = cast(Mapping[object, object], candidate_public["discrete"])
    if set(candidate_discrete) != set(reference_discrete):
        raise ContractError("ADR-127 public discrete record inventory changed from ADR-125")
    discrete_regressions = []
    for key, reference_value in reference_discrete.items():
        candidate_value = candidate_discrete[key]
        if isinstance(reference_value, bool):
            if not isinstance(candidate_value, bool):
                raise ContractError("ADR-127 public discrete metric type changed")
            if reference_value and not candidate_value:
                discrete_regressions.append(str(key))
            continue
        if not isinstance(reference_value, tuple) or not isinstance(candidate_value, tuple):
            raise ContractError("ADR-127 public discrete metric type changed")
        if len(reference_value) != len(candidate_value) or not all(
            isinstance(value, bool) for value in (*reference_value, *candidate_value)
        ):
            raise ContractError("ADR-127 public discrete metric shape changed")
        if any(
            expected and not actual
            for expected, actual in zip(reference_value, candidate_value, strict=True)
        ):
            discrete_regressions.append(str(key))
    public_checks["adr125_discrete_generation_nonregression"] = _check(
        actual={"regression_count": len(discrete_regressions), "regressions": discrete_regressions},
        passed=not discrete_regressions,
        relation="==",
        threshold={"regression_count": 0},
    )

    fixed_label_reports = (
        adr126_reference_fixed_label,
        control_fixed_label,
        candidate_fixed_label,
    )
    if adr126_reference_fixed_label.get("schema") != ADR126_FIXED_LABEL_SCHEMA or any(
        report.get("schema") != FIXED_LABEL_SCHEMA for report in fixed_label_reports[1:]
    ):
        raise ContractError("ADR-127 fixed-label report schema changed")
    for report, name in (
        (control_fixed_label, "control fixed-label"),
        (candidate_fixed_label, "candidate fixed-label"),
    ):
        _require_revision(report.get("picf_code_revision"), name=f"{name} PICF code revision")
        _validate_report_runtime_trees(
            report,
            expected=expected_runtime,
            name=name,
        )
    if any(
        report.get(field) != adr126_reference_fixed_label.get(field)
        for report in fixed_label_reports[1:]
        for field in _FIXED_LABEL_COMMON_FIELDS
    ):
        raise ContractError("ADR-127 fixed-label reports are not source matched to ADR-126")
    reference_margin = _fixed_label_metrics(adr126_reference_fixed_label)
    control_margin = _fixed_label_metrics(control_fixed_label)
    candidate_margin = _fixed_label_metrics(candidate_fixed_label)
    if (
        reference_margin["binding"] != control_margin["binding"]
        or control_margin["binding"] != candidate_margin["binding"]
    ):
        raise ContractError("ADR-127 fixed-label bank changed from ADR-126")
    reference_rows = cast(
        Mapping[tuple[Any, ...], Mapping[str, float]], reference_margin["metrics"]
    )
    control_rows = cast(Mapping[tuple[Any, ...], Mapping[str, float]], control_margin["metrics"])
    candidate_rows = cast(
        Mapping[tuple[Any, ...], Mapping[str, float]], candidate_margin["metrics"]
    )
    negative_switch_keys = [
        key
        for key, value in reference_rows.items()
        if key[3] == "part/table/switch_link"
        and (value["token_margin"] <= 0.0 or value["sequence_margin"] <= 0.0)
    ]
    if not negative_switch_keys:
        raise ContractError("ADR-127 ADR-126 reference has no negative switch margin")
    unresolved = []
    for key in negative_switch_keys:
        reference = reference_rows[key]
        candidate = candidate_rows[key]
        if reference["token_margin"] <= 0.0 and candidate["token_margin"] <= 0.0:
            unresolved.append(
                {"key": list(key), "metric": "token_margin", "value": candidate["token_margin"]}
            )
        if reference["sequence_margin"] <= 0.0 and candidate["sequence_margin"] <= 0.0:
            unresolved.append(
                {
                    "key": list(key),
                    "metric": "sequence_margin",
                    "value": candidate["sequence_margin"],
                }
            )
    candidate_switch_token = math.fsum(
        candidate_rows[key]["token_margin"] for key in negative_switch_keys
    ) / len(negative_switch_keys)
    control_switch_token = math.fsum(
        control_rows[key]["token_margin"] for key in negative_switch_keys
    ) / len(negative_switch_keys)
    candidate_switch_sequence = math.fsum(
        candidate_rows[key]["sequence_margin"] for key in negative_switch_keys
    ) / len(negative_switch_keys)
    control_switch_sequence = math.fsum(
        control_rows[key]["sequence_margin"] for key in negative_switch_keys
    ) / len(negative_switch_keys)
    margin_checks = {
        "all_previously_negative_switch_margins_positive": _check(
            actual={"unresolved": unresolved},
            passed=not unresolved,
            relation="all >",
            threshold=0.0,
        ),
        "mean_token_margin_nonregression": _check(
            actual=candidate_margin["mean_token_margin"],
            passed=_finite(candidate_margin["mean_token_margin"], name="candidate token margin")
            >= FIXED_LABEL_MEAN_FLOORS["mean_token_margin"],
            relation=">=",
            threshold=FIXED_LABEL_MEAN_FLOORS["mean_token_margin"],
        ),
        "mean_sequence_margin_nonregression": _check(
            actual=candidate_margin["mean_sequence_margin"],
            passed=_finite(
                candidate_margin["mean_sequence_margin"], name="candidate sequence margin"
            )
            >= FIXED_LABEL_MEAN_FLOORS["mean_sequence_margin"],
            relation=">=",
            threshold=FIXED_LABEL_MEAN_FLOORS["mean_sequence_margin"],
        ),
        "candidate_beats_control_switch_token_margin": _check(
            actual={"candidate": candidate_switch_token, "control": control_switch_token},
            passed=candidate_switch_token > control_switch_token,
            relation="candidate > control",
            threshold=None,
        ),
        "candidate_beats_control_switch_sequence_margin": _check(
            actual={"candidate": candidate_switch_sequence, "control": control_switch_sequence},
            passed=candidate_switch_sequence > control_switch_sequence,
            relation="candidate > control",
            threshold=None,
        ),
    }
    gradient_audit = _mapping(
        training.get("counterfactual_gradient_audit"),
        name="training gradient audit binding",
    )
    gradient_checks = {
        "candidate_mixed_gradient_descends_target_and_scene": _check(
            actual={
                "report_statuses": gradient_audit.get("report_statuses"),
                "status": gradient_audit.get("status"),
            },
            passed=gradient_audit.get("passed") is True,
            relation="all target and scene directional products >",
            threshold=0.0,
        )
    }

    sections = {
        "target": {
            "checks": target_checks,
            "control": control_target,
            "candidate": candidate_target,
        },
        "family": {"checks": family_checks},
        "clean_collision": {"checks": collision_checks},
        "fixed_label": {
            "checks": margin_checks,
            "negative_switch_record_count": len(negative_switch_keys),
        },
        "gradient": {"checks": gradient_checks},
        "pixel_causality": {
            "checks": pixel_causal_checks,
            "control": control_pixel_causal,
            "candidate": candidate_pixel_causal,
            "claim_boundary": (
                "same-prompt pixel dependence on the registered CALVIN evaluation layouts; "
                "not open-world grounding generalization"
            ),
        },
        "scene": {"checks": scene_checks, "control": control_scene, "candidate": candidate_scene},
        "public": {"checks": public_checks, "candidate": candidate_public["token_weighted_nll"]},
    }
    all_checks = [
        check
        for section in sections.values()
        for check in cast(Mapping[str, Mapping[str, object]], section["checks"]).values()
    ]
    status = "PASS" if all(check["passed"] is True for check in all_checks) else "FAIL"
    return {
        "input_reports": {
            key: {
                "schema": {
                    "adr125_reference_fixed_x": ADR125_FIXED_X_SCHEMA,
                    "adr126_reference_fixed_label": ADR126_FIXED_LABEL_SCHEMA,
                    "control_training": TRAINING_SCHEMA,
                    "candidate_training": TRAINING_SCHEMA,
                    "control_fixed_x": FIXED_X_SCHEMA,
                    "candidate_fixed_x": FIXED_X_SCHEMA,
                    "control_fixed_label": FIXED_LABEL_SCHEMA,
                    "candidate_fixed_label": FIXED_LABEL_SCHEMA,
                }[key],
                "sha256": value,
            }
            for key, value in sorted(digests.items())
        },
        "schema": OUTPUT_SCHEMA,
        "sections": sections,
        "status": status,
        "training_binding": training,
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    loaded: dict[str, Mapping[str, Any]] = {}
    digests: dict[str, str] = {}
    specifications = {
        "adr125_reference_fixed_x": (
            args.adr125_reference_fixed_x_report,
            ADR125_REFERENCE_FIXED_X_SHA256,
            ADR125_FIXED_X_SCHEMA,
            False,
        ),
        "adr126_reference_fixed_label": (
            args.adr126_reference_fixed_label_report,
            ADR126_REFERENCE_FIXED_LABEL_SHA256,
            ADR126_FIXED_LABEL_SCHEMA,
            False,
        ),
    }
    for arm in ("control", "candidate"):
        for option, schema, require_pass in (
            ("training", TRAINING_SCHEMA, True),
            ("fixed_x", FIXED_X_SCHEMA, False),
            ("fixed_label", FIXED_LABEL_SCHEMA, False),
        ):
            key = f"{arm}_{option}"
            specifications[key] = (
                getattr(args, f"{arm}_{option}_report"),
                getattr(args, f"{arm}_{option}_report_sha256"),
                schema,
                require_pass,
            )
    for key, (path, digest, schema, require_pass) in specifications.items():
        loaded[key], digests[key] = _load_report(
            path,
            expected_sha256=digest,
            schemas=schema,
            require_pass=require_pass,
        )
    picf_revision = _require_revision(
        loaded["control_training"].get("picf_code_revision"),
        name="trusted PICF checkout revision",
    )
    repo_root = Path(__file__).resolve().parents[1]
    expected_runtime_python_trees = {
        "lingbot": dict(ADR127_LINGBOT_RUNTIME_PYTHON_TREE),
        "picf": revision_bound_python_source_tree_contract(
            repo_root=repo_root,
            revision=picf_revision,
            roots={"src": repo_root / "src", "tools": repo_root / "tools"},
        ),
    }
    report = compare_lingbot_native_vl_counterfactual_scene_gate(
        adr125_reference_fixed_x=loaded["adr125_reference_fixed_x"],
        adr126_reference_fixed_label=loaded["adr126_reference_fixed_label"],
        control_training=loaded["control_training"],
        candidate_training=loaded["candidate_training"],
        control_fixed_x=loaded["control_fixed_x"],
        candidate_fixed_x=loaded["candidate_fixed_x"],
        control_fixed_label=loaded["control_fixed_label"],
        candidate_fixed_label=loaded["candidate_fixed_label"],
        expected_runtime_python_trees=expected_runtime_python_trees,
        input_report_sha256=digests,
    )
    terminal_picf_tree = revision_bound_python_source_tree_contract(
        repo_root=repo_root,
        revision=picf_revision,
        roots={"src": repo_root / "src", "tools": repo_root / "tools"},
    )
    if terminal_picf_tree != expected_runtime_python_trees["picf"]:
        raise ContractError("ADR-127 comparator runtime source changed during execution")
    write_text_durable_exclusive(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

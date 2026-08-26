#!/usr/bin/env python3
"""Compare frozen public-VL held-out reports before candidate adoption."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

try:
    from tools.bootstrap_lingbot_vla2 import CHECKPOINT_ASSET_CONTRACT
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import CHECKPOINT_ASSET_CONTRACT  # type: ignore[no-redef]

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_FAMILIES,
    PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
    PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
)
from picf_next.lingbot_native.fixed_observation_evaluation import (
    FIXED_OBSERVATION_EVALUATION_PARTITIONS,
)
from picf_next.lingbot_native.lattice_feasibility import (
    native_lattice_shortest_edge,
    native_processor_area_budget_contract,
    validate_native_processor_record_grid,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    CALVIN_GROUNDING_FAMILIES,
    CALVIN_GROUNDING_FAMILY_VARIANT_COUNTS,
    native_vl_calvin_task_family,
    native_vl_fixed_x_pair_geometry_metrics,
    native_vl_fixed_x_partition_summary,
    normalize_native_vl_answer,
)
from picf_next.lingbot_native.vl_cotraining import (
    parse_native_vl_grounding_answer,
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
)

FIXED_X_SCHEMA = "picf-next.lingbot-native-vl-fixed-x-g0.v3"
TRAINING_SCHEMA = "picf-next.lingbot-native-vl-grounding-adaptation.v6"
OUTPUT_SCHEMA = "picf-next.public-native-vl-retention-comparison.v2"
ADR125_CURRICULUM_ARTIFACT_SHA256 = (
    "e229237589e1b1da8ff2908ab3070955ae8147138d47658921d6026c4ee67648"
)
ADR125_CURRICULUM_FILE_SHA256 = "0bb2d09b272edb7ead5b809fb6983630aab93fd9557d12a7d612b5922964f55d"
ADR125_PUBLIC_ARTIFACT_SHA256 = "3c247033fde2815c3d0b350a264fa940d541529cfa9bacf34bb8737730499480"
ADR125_PUBLIC_MANIFEST_FILE_SHA256 = (
    "e6ad12f1d6df8fc53e3661d9d999d5a65b2069436822c6cfc0553f63e5323252"
)
ADR125_PUBLIC_HELDOUT_LIMIT_PER_FAMILY = 32
ADR125_PUBLIC_HELDOUT_BINDING_SHA256 = (
    "798cc30452ae35fb37167320c0e739884ef1d8fee32dc4ee4811cc52301fcbaa"
)
ADR125_PUBLIC_TRAINING_BINDING_SHA256 = (
    "fc7c86dd9b9cf886589433b383525efa15d75ad850f4515e489ab266811e170d"
)
ADR125_CALVIN_DATASET_MANIFEST_SHA256 = (
    "ad9d19ed35c708263f08c5d8376cf6ef80ec3d4e0e198e32611ae3b94971b58d"
)
ADR125_CALVIN_EVALUATION_PLAN_ARTIFACT_SHA256 = (
    "c74878284c96bd87296205c667e9e5d6b3f8a6a91ef0d4443676ba1e98daca3f"
)
ADR125_CALVIN_EVALUATION_PLAN_FILE_SHA256 = (
    "a2ee04954b0b6afc9bf10e956c967dfb9ad57c4a5fb138f9e16302aa22a51ca2"
)
ADR125_CALVIN_PHYSICAL_SIDECAR_MANIFEST_SHA256 = (
    "0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4"
)
ADR125_CALVIN_RECORD_BINDING_SHA256 = (
    "66c798ca7ef0dca3a4bf3b9ec88342c2bef987eacd7e087ed9b83a6fdcb7b976"
)
ADR125_QWEN_LONGEST_EDGE_AREA = 16_777_216
ADR125_FIXED_X_SEED = 20260801
ADR125_FIXED_X_MAX_NEW_TOKENS = 64
ADR125_PICF_CODE_REVISION = "364d8403883c4fbb6eb1b30b09a78a9a49b6e90d"
ADR125_LINGBOT_SOURCE_COMMIT = "2838c1862bbec1ea47942fb61512130f635eb595"
ADR125_NATIVE_VL_PATCH_SHA256 = "0cc8667d15082432a5095b4dd0bd892e94cad682f17f654fc1dd19289ba5c166"
ADR125_INITIAL_QWEN_REVISION = "0196dc7bb23f3c742616147c3254d0e4f1207787"
ADR125_CALVIN_TRAINING_BINDING_SHA256 = (
    "1b830c5a50f1d13edaa4084538e84e272b025cf2c06da282c4cdc2efc7b4a88f"
)
ADR125_TRAINABLE_PARAMETER_COUNT = 404
ADR125_TRAINABLE_NUMEL = 4_049_739_776
ADR125_TRAINABLE_SCHEMA_SHA256 = "f3c101623317d773f084be7064d0b588ca7932e143a5335b07db1dba52035f62"
ADR125_RELEASED_CHECKPOINT_ROOT_FILE_SHA256 = {
    path: digest for path, (_, digest) in CHECKPOINT_ASSET_CONTRACT.items() if "/" not in path
}
ADR125_CALVIN_FAMILY_MINIMUM_HITS = {
    "block": 107,
    "drawer": 31,
    "slider": 23,
    "led": 16,
    "lightbulb": 6,
}
ADR125_CALVIN_MINIMUM_NONBLOCK_HITS_EXCLUSIVE = 76
ADR125_CALVIN_MINIMUM_BIDIRECTIONAL_HITS_EXCLUSIVE = 60
ADR125_CALVIN_MINIMUM_MEAN_IOU_EXCLUSIVE = 0.395628
ADR125_CALVIN_PARTITION_MINIMUMS = {
    "validation": {"center_hits": 94, "bidirectional_hits": 31},
    "heldout": {"center_hits": 89, "bidirectional_hits": 29},
}
ADR125_CALVIN_MINIMUM_GENERATED_BOXES = 271
ADR125_CALVIN_MINIMUM_SCHEMA_VALID_BOXES = 205
ADR125_TRAINING_HYPERPARAMETERS = {
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "learning_rate": 1e-6,
    "max_grad_norm": 1.0,
    "max_steps": 64,
    "schedule": "linear-warmup-cosine-decay",
    "schedule_total_steps": 432,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}
_COMMON_BINDINGS = (
    "dataset_manifest_sha256",
    "evaluation_plan_artifact_sha256",
    "evaluation_plan_file_sha256",
    "item_limit_per_partition",
    "max_new_tokens",
    "native_vl_patch_sha256",
    "partition",
    "picf_code_revision",
    "processor_lattice",
    "seed",
    "selected_item_count",
    "source_commit",
)
_RECORD_BINDINGS = (
    "family",
    "record_id",
    "record_sha256",
    "image_rgb_sha256",
    "image_height",
    "image_width",
    "source_row_index",
    "source_subindex",
    "supervised_token_count",
    "target_answer",
    "target_answer_sha256",
    "user_text",
    "user_text_sha256",
)
_PUBLIC_SCHEDULE_BINDINGS = (
    "family",
    "record_id",
    "record_sha256",
    "image_rgb_sha256",
    "image_height",
    "image_width",
    "source_row_index",
    "source_subindex",
    "supervised_token_count",
    "target_answer_sha256",
    "user_text_sha256",
)
_PUBLIC_RECORD_IDENTITY_FIELDS = (
    "family",
    "record_id",
    "source_row_index",
    "source_subindex",
)
_CALVIN_TRAINING_RANK_BINDINGS = (
    "rank",
    "global_index",
    "camera_name",
    "instruction",
    "task_key",
    "target_identity_key",
    "visual_lattice",
    "image_grid_thw",
    "supervised_token_count",
    "loss_weight",
)
_CALVIN_GROUP_BINDINGS = (
    "partition",
    "ordinal",
    "source_global_index",
    "source_state_sha256",
    "source_rgb_sha256",
)
_CALVIN_VARIANT_BINDINGS = (
    "camera_name",
    "task_key",
    "instruction",
    "instruction_sha256",
    "target_identity_key",
    "target_bbox_qwen_xyxy",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--control-report-sha256", required=True)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--candidate-training-report", required=True, type=Path)
    parser.add_argument("--candidate-training-report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"public retention comparison {name} must be one SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ContractError("public retention comparison binding is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError(f"public retention comparison {name} must be nonempty text")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"public retention comparison {name} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"public retention comparison {name} must be a nonnegative integer")
    return value


def _validate_text_sha256(row: Mapping[str, Any], *, field: str) -> str:
    text = _require_text(row.get(field), name=field)
    digest = _require_sha256(row.get(f"{field}_sha256"), name=f"{field} digest")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
        raise ContractError(f"public retention comparison {field} digest changed")
    return text


def _generated_text(row: Mapping[str, Any], *, name: str) -> str:
    value = row.get("generated_text")
    if not isinstance(value, str) or "\0" in value:
        raise ContractError(f"public retention comparison {name} must be text")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"public retention comparison {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _checkpoint_dir(report: Mapping[str, Any], *, name: str) -> str:
    value = report.get("checkpoint_dir")
    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError(f"public retention comparison {name} checkpoint path is malformed")
    return value


def _read_regular_input(path: Path) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(lexical.anchor, directory_flags)
        for part in lexical.parent.parts[1:]:
            try:
                child = os.open(
                    part,
                    directory_flags | nofollow,
                    dir_fd=parent_descriptor,
                )
            finally:
                os.close(parent_descriptor)
            parent_descriptor = child
    except OSError as error:
        raise ContractError(
            "public retention comparison input path traverses a symlink or non-directory"
        ) from error
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    try:
        try:
            descriptor = os.open(lexical.name, file_flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ContractError(
                "public retention comparison input must be a readable non-symlink file"
            ) from error
    finally:
        os.close(parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("public retention comparison input must be a regular file")
        blocks = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ContractError("public retention comparison input cannot be read") from error
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != after.st_size:
        raise ContractError("public retention comparison input changed while being read")
    return payload


def _load_report(
    path: Path,
    *,
    expected_sha256: str,
    schema: str,
    require_pass: bool,
) -> tuple[dict[str, Any], str]:
    expected = _require_sha256(expected_sha256, name="report digest")
    payload = _read_regular_input(path)
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ContractError("public retention comparison report digest changed")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(
                    f"public retention comparison JSON contains duplicate key: {key}"
                )
            result[key] = value
        return result

    def reject_constant(constant: str) -> NoReturn:
        raise ValueError(f"non-finite JSON value: {constant}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError("public retention comparison input is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ContractError("public retention comparison input schema changed")
    if require_pass and value.get("status") != "PASS":
        raise ContractError("public retention comparison requires a passing training report")
    return cast(dict[str, Any], value), actual


def _public_section(report: Mapping[str, Any]) -> Mapping[str, Any]:
    section = report.get("public_vl_retention")
    if not isinstance(section, Mapping) or section.get("enabled") is not True:
        raise ContractError("public retention comparison input omitted public evaluation")
    if section.get("heldout_limit_per_family") != ADR125_PUBLIC_HELDOUT_LIMIT_PER_FAMILY:
        raise ContractError(
            "public retention comparison held-out limit must equal the frozen 32 records"
        )
    counts = section.get("family_partition_counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(f"{family}/heldout") != PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
        for family in PUBLIC_NATIVE_VL_FAMILIES
    ):
        raise ContractError("public retention comparison held-out counts changed")
    processor = _mapping(section.get("processor"), name="public evaluation processor")
    if dict(processor) != native_processor_area_budget_contract(8):
        raise ContractError("public retention comparison public processor contract changed")
    if section.get("artifact_sha256") != ADR125_PUBLIC_ARTIFACT_SHA256:
        raise ContractError("public retention comparison public artifact changed")
    if section.get("manifest_file_sha256") != ADR125_PUBLIC_MANIFEST_FILE_SHA256:
        raise ContractError("public retention comparison public manifest changed")
    return section


def _finite_nonnegative_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"public retention comparison {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContractError(f"public retention comparison {name} must be finite and nonnegative")
    return result


def _count_metric(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"public retention comparison {name} must be an integer")
    if not 0 <= value <= PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY:
        raise ContractError(f"public retention comparison {name} is outside its valid range")
    return value


def _grid_binding(row: Mapping[str, Any], *, lattice: int) -> str:
    try:
        expected = validate_native_processor_record_grid(
            row.get("image_grid_thw"),
            image_height=_require_positive_integer(
                row.get("image_height"),
                name="public image height",
            ),
            image_width=_require_positive_integer(
                row.get("image_width"),
                name="public image width",
            ),
            lattice=lattice,
        )
    except (ContractError, RuntimeError, ValueError) as error:
        raise ContractError("public retention comparison row grid is invalid") from error
    claimed = row.get("grid_budget")
    if not isinstance(claimed, Mapping) or dict(claimed) != expected:
        raise ContractError("public retention comparison row grid budget changed")
    return json.dumps(expected, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _bound_records(
    section: Mapping[str, Any],
    *,
    lattice: int,
) -> tuple[tuple[object, ...], ...]:
    rows = section.get("results")
    if not isinstance(rows, list) or len(rows) != (
        len(PUBLIC_NATIVE_VL_FAMILIES) * PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
    ):
        raise ContractError("public retention comparison result cardinality changed")
    bindings = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("public retention comparison result row is malformed")
        binding = (
            *(row.get(field) for field in _RECORD_BINDINGS),
            _grid_binding(row, lattice=lattice),
        )
        if any(value is None for value in binding):
            raise ContractError("public retention comparison result binding is incomplete")
        _require_sha256(row.get("record_sha256"), name="public record digest")
        _require_sha256(row.get("image_rgb_sha256"), name="public image RGB digest")
        _validate_text_sha256(row, field="target_answer")
        _validate_text_sha256(row, field="user_text")
        bindings.append(binding)
    if len(bindings) != len(set(bindings)):
        raise ContractError("public retention comparison result rows are duplicated")
    schedule = [
        [row.get(field) for field in _PUBLIC_SCHEDULE_BINDINGS]
        for row in rows
        if isinstance(row, Mapping)
    ]
    if _canonical_sha256(schedule) != ADR125_PUBLIC_HELDOUT_BINDING_SHA256:
        raise ContractError("public retention comparison held-out schedule changed")
    return tuple(bindings)


def _validate_training_public_steps(
    training: Mapping[str, Any],
    *,
    lattice: int,
) -> None:
    steps = training.get("step_reports")
    if not isinstance(steps, list) or len(steps) != ADR125_TRAINING_HYPERPARAMETERS["max_steps"]:
        raise ContractError("public retention comparison training step coverage changed")
    expected_families = ("referring", "vqa")
    schedule = []
    record_ids = []
    for expected_step, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("optimizer_step") != expected_step:
            raise ContractError("public retention comparison training step order changed")
        public = _mapping(
            step.get("public_vl_retention"),
            name="training step public retention",
        )
        ranks = public.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != len(expected_families):
            raise ContractError("public retention comparison training rank coverage changed")
        for expected_rank, (family, row) in enumerate(zip(expected_families, ranks, strict=True)):
            if (
                not isinstance(row, Mapping)
                or row.get("rank") != expected_rank
                or row.get("family") != family
                or row.get("loss_weight") != PUBLIC_NATIVE_VL_RETENTION_WEIGHT
            ):
                raise ContractError("public retention comparison training rank binding changed")
            _grid_binding(row, lattice=lattice)
            _require_sha256(row.get("record_sha256"), name="training public record digest")
            _require_sha256(row.get("image_rgb_sha256"), name="training public image digest")
            _validate_text_sha256(row, field="user_text")
            _require_sha256(
                row.get("target_answer_sha256"),
                name="training target answer digest",
            )
            _require_positive_integer(
                row.get("supervised_token_count"),
                name="training supervised token count",
            )
            _finite_nonnegative_metric(row.get("loss"), name="training public loss")
            schedule.append([row.get(field) for field in _PUBLIC_SCHEDULE_BINDINGS])
            record_ids.append(row.get("record_id"))
    if len(record_ids) != len(set(record_ids)):
        raise ContractError("public retention comparison training records are duplicated")
    if _canonical_sha256(schedule) != ADR125_PUBLIC_TRAINING_BINDING_SHA256:
        raise ContractError("public retention comparison training schedule changed")


def _validate_training_calvin_steps(training: Mapping[str, Any]) -> None:
    steps = training.get("step_reports")
    if not isinstance(steps, list) or len(steps) != ADR125_TRAINING_HYPERPARAMETERS["max_steps"]:
        raise ContractError("public retention comparison CALVIN step coverage changed")
    schedule = []
    for expected_step, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("optimizer_step") != expected_step:
            raise ContractError("public retention comparison CALVIN step order changed")
        if (
            step.get("curriculum_optimizer_step") != expected_step
            or step.get("observation_mode") != "official_native_once"
        ):
            raise ContractError("public retention comparison CALVIN curriculum changed")
        _require_nonnegative_integer(
            step.get("curriculum_group_index"),
            name="CALVIN curriculum group index",
        )
        learning_rate = _finite_nonnegative_metric(
            step.get("learning_rate"),
            name="CALVIN learning rate",
        )
        if learning_rate <= 0.0:
            raise ContractError("public retention comparison CALVIN learning rate is zero")
        microbatches = step.get("microbatches")
        if not isinstance(microbatches, list) or len(microbatches) != 1:
            raise ContractError("public retention comparison CALVIN microbatch coverage changed")
        normalized_microbatches = []
        for microbatch in microbatches:
            if not isinstance(microbatch, Mapping) or microbatch.get("visual_lattice") != 8:
                raise ContractError("public retention comparison CALVIN microbatch lattice changed")
            ranks = microbatch.get("ranks")
            if not isinstance(ranks, list) or len(ranks) != 2:
                raise ContractError("public retention comparison CALVIN rank coverage changed")
            normalized_ranks = []
            for expected_rank, rank in enumerate(ranks):
                if not isinstance(rank, Mapping) or rank.get("rank") != expected_rank:
                    raise ContractError("public retention comparison CALVIN rank order changed")
                if (
                    rank.get("visual_lattice") != 8
                    or rank.get("image_grid_thw") != [[1, 16, 16]]
                    or rank.get("loss_weight") != 1.0
                ):
                    raise ContractError("public retention comparison CALVIN rank lattice changed")
                _require_nonnegative_integer(
                    rank.get("global_index"),
                    name="CALVIN training global index",
                )
                for field in ("camera_name", "instruction", "task_key", "target_identity_key"):
                    _require_text(rank.get(field), name=f"CALVIN training {field}")
                _require_positive_integer(
                    rank.get("supervised_token_count"),
                    name="CALVIN training supervised token count",
                )
                _finite_nonnegative_metric(rank.get("loss"), name="CALVIN training loss")
                normalized_ranks.append(
                    {field: rank.get(field) for field in _CALVIN_TRAINING_RANK_BINDINGS}
                )
            normalized_microbatches.append({"visual_lattice": 8, "ranks": normalized_ranks})
        gradient = _mapping(step.get("gradient_metrics"), name="CALVIN gradient metrics")
        if (
            gradient.get("all_finite") is not True
            or gradient.get("frozen_gradient_elements") != 0
            or gradient.get("trainable_gradient_elements") != ADR125_TRAINABLE_NUMEL
        ):
            raise ContractError("public retention comparison CALVIN gradient coverage changed")
        global_norm = _finite_nonnegative_metric(
            gradient.get("global_norm_before_clip"),
            name="CALVIN global gradient norm",
        )
        clip_coefficient = _finite_nonnegative_metric(
            gradient.get("clip_coefficient"),
            name="CALVIN clip coefficient",
        )
        if global_norm <= 0.0 or not 0.0 < clip_coefficient <= 1.0:
            raise ContractError("public retention comparison CALVIN gradient metric is invalid")
        expected_clip_coefficient = min(
            1.0,
            float(ADR125_TRAINING_HYPERPARAMETERS["max_grad_norm"]) / (global_norm + 1e-6),
        )
        if not math.isclose(
            clip_coefficient,
            expected_clip_coefficient,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ContractError(
                "public retention comparison CALVIN clip coefficient contradicts gradient norm"
            )
        schedule.append(
            {
                "optimizer_step": expected_step,
                "curriculum_group_index": step.get("curriculum_group_index"),
                "curriculum_optimizer_step": expected_step,
                "observation_mode": "official_native_once",
                "learning_rate": learning_rate,
                "microbatches": normalized_microbatches,
            }
        )
    if _canonical_sha256(schedule) != ADR125_CALVIN_TRAINING_BINDING_SHA256:
        raise ContractError("public retention comparison CALVIN training schedule changed")


def _positive_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"public retention comparison {name} must be a positive integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"public retention comparison {name} must be boolean")
    return value


def _recomputed_summaries(section: Mapping[str, Any]) -> dict[str, dict[str, object]]:
    rows = section.get("results")
    if not isinstance(rows, list):
        raise ContractError("public retention comparison result rows are malformed")
    summaries: dict[str, dict[str, object]] = {}
    for family in PUBLIC_NATIVE_VL_FAMILIES:
        selected = [row for row in rows if isinstance(row, Mapping) and row.get("family") == family]
        if len(selected) != PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY:
            raise ContractError("public retention comparison result family count changed")
        token_counts = [
            _positive_count(
                row.get("supervised_token_count"),
                name=f"{family} supervised token count",
            )
            for row in selected
        ]
        nlls = [
            _finite_nonnegative_metric(
                row.get("mean_token_nll"),
                name=f"{family} row NLL",
            )
            for row in selected
        ]
        supervised_tokens = sum(token_counts)
        summary: dict[str, object] = {
            "mean_record_nll": sum(nlls) / len(nlls),
            "record_count": len(selected),
            "supervised_token_count": supervised_tokens,
            "token_weighted_mean_nll": sum(
                nll * token_count for nll, token_count in zip(nlls, token_counts, strict=True)
            )
            / supervised_tokens,
        }
        if family == "referring":
            generated = []
            schema_valid = []
            target_ious = []
            target_center_hits = []
            for row in selected:
                target = parse_native_vl_grounding_answer(
                    _validate_text_sha256(row, field="target_answer")
                )
                if target.bbox_qwen_xyxy is None or not target.schema_valid:
                    raise ContractError("public retention comparison referring target is malformed")
                generated_answer = parse_native_vl_grounding_answer(
                    _generated_text(row, name="referring generated text")
                )
                prediction = generated_answer.bbox_qwen_xyxy
                reported_prediction = _qwen_bbox(
                    row.get("generated_bbox_qwen_xyxy"),
                    name="referring generated bbox",
                    optional=True,
                )
                reported_target = _qwen_bbox(
                    row.get("target_bbox_qwen_xyxy"),
                    name="referring target bbox",
                    optional=False,
                )
                if (
                    prediction != reported_prediction
                    or target.bbox_qwen_xyxy != reported_target
                    or generated_answer.schema_valid
                    != _boolean(
                        row.get("generated_bbox_schema_valid"),
                        name="referring generated bbox schema flag",
                    )
                ):
                    raise ContractError(
                        "public retention comparison referring text and bbox differ"
                    )
                target_iou = (
                    0.0
                    if prediction is None
                    else qwen_grounding_bbox_iou(prediction, target.bbox_qwen_xyxy)
                )
                target_center_hit = (
                    False
                    if prediction is None
                    else qwen_target_center_in_bbox(prediction, target.bbox_qwen_xyxy)
                )
                if (
                    row.get("target_iou") != target_iou
                    or row.get("target_center_hit") != target_center_hit
                ):
                    raise ContractError("public retention comparison referring geometry changed")
                generated.append(prediction is not None)
                schema_valid.append(generated_answer.schema_valid)
                target_ious.append(target_iou)
                target_center_hits.append(target_center_hit)
            summary.update(
                {
                    "generated_bbox_count": sum(generated),
                    "generated_bbox_schema_valid_count": sum(schema_valid),
                    "mean_target_iou": sum(target_ious) / len(target_ious),
                    "target_center_hit_count": sum(target_center_hits),
                }
            )
        else:
            normalized_exact_matches = []
            for row in selected:
                expected = normalize_native_vl_answer(
                    _validate_text_sha256(row, field="target_answer")
                )
                actual = normalize_native_vl_answer(_generated_text(row, name="VQA generated text"))
                exact_match = actual == expected
                if row.get("normalized_exact_match") != exact_match:
                    raise ContractError("public retention comparison VQA metric changed")
                normalized_exact_matches.append(exact_match)
            summary["normalized_exact_match_count"] = sum(normalized_exact_matches)
        summaries[family] = summary
    return summaries


def _validated_summaries(section: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    reported = section.get("summaries")
    if not isinstance(reported, Mapping):
        raise ContractError("public retention comparison summaries are malformed")
    recomputed = _recomputed_summaries(section)
    validated: dict[str, Mapping[str, Any]] = {}
    for family in PUBLIC_NATIVE_VL_FAMILIES:
        family_report = reported.get(family)
        if not isinstance(family_report, Mapping):
            raise ContractError("public retention comparison family summary is missing")
        if dict(family_report) != recomputed[family]:
            raise ContractError(
                f"public retention comparison {family} summary differs from its result rows"
            )
        validated[family] = family_report
    return validated


def _public_record_index(
    section: Mapping[str, Any],
    *,
    name: str,
) -> dict[tuple[object, ...], Mapping[str, Any]]:
    rows = section.get("results")
    if not isinstance(rows, list):
        raise ContractError(f"public retention comparison {name} result rows are malformed")
    result: dict[tuple[object, ...], Mapping[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise ContractError(f"public retention comparison {name} result row is malformed")
        identity = tuple(value.get(field) for field in _PUBLIC_RECORD_IDENTITY_FIELDS)
        if any(field is None for field in identity):
            raise ContractError(f"public retention comparison {name} record identity is incomplete")
        if identity in result:
            raise ContractError(f"public retention comparison {name} record identity is duplicated")
        result[identity] = cast(Mapping[str, Any], value)
    return result


def _public_generation_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, bool], list[dict[str, object]]]:
    control_rows = _public_record_index(control, name="control")
    candidate_rows = _public_record_index(candidate, name="candidate")
    if control_rows.keys() != candidate_rows.keys():
        raise ContractError("public retention comparison public record identities changed")

    referring_nonregression = True
    vqa_nonregression = True
    referring_review: list[dict[str, object]] = []
    for identity, control_row in control_rows.items():
        candidate_row = candidate_rows[identity]
        family, record_id, source_row_index, source_subindex = identity
        if family == "referring":
            control_prediction = _qwen_bbox(
                control_row.get("generated_bbox_qwen_xyxy"),
                name="control referring generated bbox",
                optional=True,
            )
            candidate_prediction = _qwen_bbox(
                candidate_row.get("generated_bbox_qwen_xyxy"),
                name="candidate referring generated bbox",
                optional=True,
            )
            control_generated = control_prediction is not None
            candidate_generated = candidate_prediction is not None
            control_schema_valid = _boolean(
                control_row.get("generated_bbox_schema_valid"),
                name="control referring generated bbox schema flag",
            )
            candidate_schema_valid = _boolean(
                candidate_row.get("generated_bbox_schema_valid"),
                name="candidate referring generated bbox schema flag",
            )
            control_center_hit = _boolean(
                control_row.get("target_center_hit"),
                name="control referring target center hit",
            )
            candidate_center_hit = _boolean(
                candidate_row.get("target_center_hit"),
                name="candidate referring target center hit",
            )
            control_iou = _finite_nonnegative_metric(
                control_row.get("target_iou"),
                name="control referring target IoU",
            )
            candidate_iou = _finite_nonnegative_metric(
                candidate_row.get("target_iou"),
                name="candidate referring target IoU",
            )
            reasons = []
            if _generated_text(
                control_row,
                name="control referring generated text",
            ) != _generated_text(
                candidate_row,
                name="candidate referring generated text",
            ):
                reasons.append("generated_text_changed")
            if control_prediction != candidate_prediction:
                reasons.append("generated_bbox_changed")
            if control_generated != candidate_generated:
                reasons.append("generated_bbox_presence_flipped")
            if control_schema_valid != candidate_schema_valid:
                reasons.append("generated_bbox_schema_valid_flipped")
            if control_center_hit != candidate_center_hit:
                reasons.append("target_center_hit_flipped")
            if candidate_iou != control_iou:
                reasons.append("target_iou_changed")
            if reasons:
                referring_review.append(
                    {
                        "family": "referring",
                        "reasons": reasons,
                        "record_id": record_id,
                        "source_row_index": source_row_index,
                        "source_subindex": source_subindex,
                    }
                )
            if (
                (control_generated and not candidate_generated)
                or (control_schema_valid and not candidate_schema_valid)
                or (control_center_hit and not candidate_center_hit)
            ):
                referring_nonregression = False
        elif family == "vqa":
            control_exact = _boolean(
                control_row.get("normalized_exact_match"),
                name="control VQA normalized exact match",
            )
            candidate_exact = _boolean(
                candidate_row.get("normalized_exact_match"),
                name="candidate VQA normalized exact match",
            )
            if control_exact and not candidate_exact:
                vqa_nonregression = False
        else:
            raise ContractError("public retention comparison public family changed")
    return (
        {
            "referring_per_record_nonregression": referring_nonregression,
            "vqa_per_record_nonregression": vqa_nonregression,
        },
        referring_review,
    )


def _normalized_model_hashes(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError(f"public retention comparison {name} hashes are missing")
    normalized = {}
    for filename, digest in value.items():
        if (
            not isinstance(filename, str)
            or not filename
            or "\0" in filename
            or Path(filename).name != filename
        ):
            raise ContractError(f"public retention comparison {name} filename is malformed")
        normalized[filename] = _require_sha256(
            digest,
            name=f"{name} file digest",
        )
    return dict(sorted(normalized.items()))


def _model_binding(report: Mapping[str, Any], *, name: str) -> dict[str, object]:
    normalized_checkpoint_hashes = _normalized_model_hashes(
        report.get("checkpoint_model_file_sha256"),
        name=f"{name} checkpoint",
    )
    if any(
        normalized_checkpoint_hashes.get(filename) != digest
        for filename, digest in ADR125_RELEASED_CHECKPOINT_ROOT_FILE_SHA256.items()
    ):
        raise ContractError(
            f"public retention comparison {name} released checkpoint identity changed"
        )
    checkpoint_weight_hashes = {
        filename: digest
        for filename, digest in normalized_checkpoint_hashes.items()
        if filename.endswith(".safetensors")
    }
    if not checkpoint_weight_hashes:
        raise ContractError(
            f"public retention comparison {name} checkpoint has no safetensor weights"
        )
    restore = report.get("qwen_restore")
    restore_hashes = None
    restore_revision = None
    if restore is not None:
        if not isinstance(restore, Mapping):
            raise ContractError(f"public retention comparison {name} Qwen restore is malformed")
        restore_hashes = _normalized_model_hashes(
            restore.get("model_file_sha256"),
            name=f"{name} Qwen restore",
        )
        raw_revision = restore.get("model_revision")
        if (
            not isinstance(raw_revision, str)
            or len(raw_revision) != 40
            or any(character not in "0123456789abcdef" for character in raw_revision)
        ):
            raise ContractError(
                f"public retention comparison {name} Qwen restore revision is malformed"
            )
        restore_revision = raw_revision
    restore_weight_hashes = (
        None
        if restore_hashes is None
        else {
            filename: digest
            for filename, digest in restore_hashes.items()
            if filename.endswith(".safetensors")
        }
    )
    if restore_hashes is not None and not restore_weight_hashes:
        raise ContractError(
            f"public retention comparison {name} Qwen restore has no safetensor weights"
        )
    effective_qwen_weight_hashes = (
        checkpoint_weight_hashes if restore_weight_hashes is None else restore_weight_hashes
    )
    return {
        "checkpoint_model_file_sha256": dict(sorted(normalized_checkpoint_hashes.items())),
        "effective_qwen_weight_file_sha256": dict(sorted(effective_qwen_weight_hashes.items())),
        "qwen_restore_model_file_sha256": (
            None if restore_hashes is None else dict(sorted(restore_hashes.items()))
        ),
        "qwen_restore_revision": restore_revision,
    }


def _calvin_processor_lattice(report: Mapping[str, Any]) -> int:
    processor = _mapping(report.get("processor_lattice"), name="CALVIN processor lattice")
    lattice = processor.get("lattice")
    if lattice not in (8, 14):
        raise ContractError("public retention comparison CALVIN lattice must be 8 or 14")
    expected = {
        "lattice": lattice,
        "longest_edge_area": ADR125_QWEN_LONGEST_EDGE_AREA,
        "merge_size": 2,
        "patch_size": 16,
        "pixels_per_edge": lattice * 16 * 2,
        "shortest_edge_area": native_lattice_shortest_edge(lattice),
    }
    if dict(processor) != expected:
        raise ContractError("public retention comparison CALVIN processor contract changed")
    return lattice


def _qwen_bbox(value: object, *, name: str, optional: bool) -> tuple[int, int, int, int] | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ContractError(f"public retention comparison {name} is malformed")
    bbox = cast(tuple[int, int, int, int], tuple(value))
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ContractError(f"public retention comparison {name} is invalid")
    return bbox


def _calvin_group_binding(group: Mapping[str, Any]) -> list[object]:
    variants = group.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ContractError("public retention comparison CALVIN pair is malformed")
    variant_bindings = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ContractError("public retention comparison CALVIN variant is malformed")
        instruction = _require_text(variant.get("instruction"), name="CALVIN instruction")
        instruction_sha256 = _require_sha256(
            variant.get("instruction_sha256"),
            name="CALVIN instruction digest",
        )
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != instruction_sha256:
            raise ContractError("public retention comparison CALVIN instruction digest changed")
        variant_bindings.append([variant.get(field) for field in _CALVIN_VARIANT_BINDINGS])
    return [*(group.get(field) for field in _CALVIN_GROUP_BINDINGS), variant_bindings]


def _validated_calvin_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("partition") != "all"
        or report.get("item_limit_per_partition") != 0
        or report.get("selected_item_count") != 136
        or report.get("eligible_item_count") != 136
        or report.get("excluded_items") != []
    ):
        raise ContractError("public retention comparison requires the complete CALVIN bank")
    if report.get("dataset_manifest_sha256") != ADR125_CALVIN_DATASET_MANIFEST_SHA256:
        raise ContractError("public retention comparison CALVIN dataset changed")
    if (
        report.get("evaluation_plan_artifact_sha256")
        != ADR125_CALVIN_EVALUATION_PLAN_ARTIFACT_SHA256
        or report.get("evaluation_plan_file_sha256") != ADR125_CALVIN_EVALUATION_PLAN_FILE_SHA256
    ):
        raise ContractError("public retention comparison CALVIN evaluation plan changed")
    if (
        report.get("physical_sidecar_manifest_sha256")
        != ADR125_CALVIN_PHYSICAL_SIDECAR_MANIFEST_SHA256
    ):
        raise ContractError("public retention comparison CALVIN sidecar changed")
    lattice = _calvin_processor_lattice(report)
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 136:
        raise ContractError("public retention comparison CALVIN pair count changed")

    expected_order = [
        (partition, ordinal)
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
        for ordinal in range(68)
    ]
    observed_order = []
    bindings = []
    normalized_groups = []
    family_hits: dict[str, int] = {family: 0 for family in CALVIN_GROUNDING_FAMILIES}
    family_counts: dict[str, int] = {family: 0 for family in CALVIN_GROUNDING_FAMILIES}
    variants_by_key: dict[tuple[str, int, int], dict[str, object]] = {}
    for group in rows:
        if not isinstance(group, Mapping):
            raise ContractError("public retention comparison CALVIN group is malformed")
        partition = group.get("partition")
        ordinal = group.get("ordinal")
        observed_order.append((partition, ordinal))
        _require_nonnegative_integer(
            group.get("source_global_index"),
            name="CALVIN source global index",
        )
        _require_sha256(group.get("source_state_sha256"), name="CALVIN source state digest")
        _require_sha256(group.get("source_rgb_sha256"), name="CALVIN source RGB digest")
        binding = _calvin_group_binding(group)
        bindings.append(binding)
        variants = group.get("variants")
        if not isinstance(variants, list):
            raise RuntimeError("validated CALVIN pair lost its variants")
        predictions = []
        targets = []
        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, Mapping):
                raise RuntimeError("validated CALVIN pair lost one variant")
            task_key = _require_text(variant.get("task_key"), name="CALVIN task key")
            target_identity = _require_text(
                variant.get("target_identity_key"),
                name="CALVIN target identity",
            )
            family = native_vl_calvin_task_family(task_key, target_identity)
            target = _qwen_bbox(
                variant.get("target_bbox_qwen_xyxy"),
                name="CALVIN target bbox",
                optional=False,
            )
            reported_prediction = _qwen_bbox(
                variant.get("generated_bbox_qwen_xyxy"),
                name="CALVIN generated bbox",
                optional=True,
            )
            if target is None:
                raise RuntimeError("validated CALVIN target bbox became absent")
            schema_valid = _boolean(
                variant.get("generated_bbox_schema_valid"),
                name="CALVIN generated bbox schema flag",
            )
            generated = parse_native_vl_grounding_answer(
                _generated_text(variant, name="CALVIN generated text")
            )
            prediction = generated.bbox_qwen_xyxy
            if prediction != reported_prediction or generated.schema_valid != schema_valid:
                raise ContractError("CALVIN generated text and bbox differ")
            predictions.append(prediction)
            targets.append(target)
            family_counts[family] += 1
            variants_by_key[(cast(str, partition), cast(int, ordinal), variant_index)] = {
                "family": family,
                "generated_bbox_qwen_xyxy": (None if prediction is None else list(prediction)),
                "generated_bbox_schema_valid": schema_valid,
                "task_key": task_key,
            }
        metrics = native_vl_fixed_x_pair_geometry_metrics(
            (predictions[0], predictions[1]),
            (targets[0], targets[1]),
        )
        reported_pair_metrics = _mapping(
            group.get("pair_metrics"),
            name="CALVIN pair metrics",
        )
        expected_pair_metrics = {key: value for key, value in metrics.items() if key != "variants"}
        if dict(reported_pair_metrics) != expected_pair_metrics:
            raise ContractError("public retention comparison CALVIN pair metrics changed")
        metric_variants = metrics.get("variants")
        if not isinstance(metric_variants, list):
            raise RuntimeError("CALVIN metric recomputation lost variant rows")
        normalized_variants = []
        for variant_index, (variant, recomputed) in enumerate(
            zip(variants, metric_variants, strict=True)
        ):
            if not isinstance(variant, Mapping) or not isinstance(recomputed, Mapping):
                raise RuntimeError("CALVIN metric recomputation produced malformed rows")
            for key, value in recomputed.items():
                if variant.get(key) != value:
                    raise ContractError("public retention comparison CALVIN variant metric changed")
            variant_key = (cast(str, partition), cast(int, ordinal), variant_index)
            family = cast(str, variants_by_key[variant_key]["family"])
            if bool(recomputed["own_target_center_hit"]):
                family_hits[family] += 1
            normalized_variants.append(
                {
                    "generated_bbox_qwen_xyxy": variant.get("generated_bbox_qwen_xyxy"),
                    "generated_bbox_schema_valid": variant.get("generated_bbox_schema_valid"),
                    **dict(recomputed),
                }
            )
            variants_by_key[(cast(str, partition), cast(int, ordinal), variant_index)].update(
                recomputed
            )
        normalized_groups.append(
            {
                "pair_metrics": expected_pair_metrics,
                "partition": partition,
                "ordinal": ordinal,
                "variants": normalized_variants,
            }
        )
    if observed_order != expected_order:
        raise ContractError("public retention comparison CALVIN pair order changed")
    if _canonical_sha256(bindings) != ADR125_CALVIN_RECORD_BINDING_SHA256:
        raise ContractError("public retention comparison CALVIN record binding changed")
    if family_counts != CALVIN_GROUNDING_FAMILY_VARIANT_COUNTS:
        raise ContractError("public retention comparison CALVIN family coverage changed")

    summaries = {
        partition: native_vl_fixed_x_partition_summary(
            [group for group in normalized_groups if group["partition"] == partition]
        )
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
    }
    if report.get("summaries") != summaries:
        raise ContractError("public retention comparison CALVIN summaries changed")
    all_variants = [variant for group in normalized_groups for variant in group["variants"]]
    return {
        "bindings": bindings,
        "family_counts": family_counts,
        "family_hits": family_hits,
        "generated_bbox_count": sum(
            variant["generated_bbox_qwen_xyxy"] is not None for variant in all_variants
        ),
        "generated_bbox_schema_valid_count": sum(
            bool(variant["generated_bbox_schema_valid"]) for variant in all_variants
        ),
        "lattice": lattice,
        "mean_own_target_iou": sum(float(variant["own_target_iou"]) for variant in all_variants)
        / len(all_variants),
        "summaries": summaries,
        "total_bidirectional_hits": sum(
            _require_nonnegative_integer(
                summary.get("bidirectional_own_only_center_hit_count"),
                name="CALVIN bidirectional hit count",
            )
            for summary in summaries.values()
        ),
        "variants_by_key": variants_by_key,
    }


def _calvin_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, object]:
    control_calvin = _validated_calvin_report(control)
    candidate_calvin = _validated_calvin_report(candidate)
    if control_calvin["bindings"] != candidate_calvin["bindings"]:
        raise ContractError("public retention comparison evaluated different CALVIN records")
    if control_calvin["lattice"] != candidate_calvin["lattice"]:
        raise ContractError("public retention comparison CALVIN lattice changed")

    family_hits = cast(dict[str, int], candidate_calvin["family_hits"])
    control_family_hits = cast(dict[str, int], control_calvin["family_hits"])
    checks: dict[str, bool] = {
        f"{family}_minimum": family_hits[family] >= minimum
        for family, minimum in ADR125_CALVIN_FAMILY_MINIMUM_HITS.items()
    }
    checks["block_exact"] = family_hits["block"] == CALVIN_GROUNDING_FAMILY_VARIANT_COUNTS["block"]
    checks["led_or_lightbulb_strictly_improves"] = (
        family_hits["led"] > control_family_hits["led"]
        or family_hits["lightbulb"] > control_family_hits["lightbulb"]
    )
    checks["nonblock_center_hits"] = (
        sum(family_hits[family] for family in CALVIN_GROUNDING_FAMILIES if family != "block")
        > ADR125_CALVIN_MINIMUM_NONBLOCK_HITS_EXCLUSIVE
    )
    checks["bidirectional_hits"] = (
        int(candidate_calvin["total_bidirectional_hits"])
        > ADR125_CALVIN_MINIMUM_BIDIRECTIONAL_HITS_EXCLUSIVE
    )
    checks["mean_own_target_iou"] = (
        float(candidate_calvin["mean_own_target_iou"]) > ADR125_CALVIN_MINIMUM_MEAN_IOU_EXCLUSIVE
    )
    summaries = cast(dict[str, dict[str, object]], candidate_calvin["summaries"])
    for partition, minimums in ADR125_CALVIN_PARTITION_MINIMUMS.items():
        summary = summaries[partition]
        checks[f"{partition}_center_hits"] = (
            _require_nonnegative_integer(
                summary.get("own_target_center_hit_count"),
                name=f"{partition} CALVIN center hit count",
            )
            >= minimums["center_hits"]
        )
        checks[f"{partition}_bidirectional_hits"] = (
            _require_nonnegative_integer(
                summary.get("bidirectional_own_only_center_hit_count"),
                name=f"{partition} CALVIN bidirectional hit count",
            )
            >= minimums["bidirectional_hits"]
        )
    checks["generated_bbox_count"] = (
        int(candidate_calvin["generated_bbox_count"]) >= ADR125_CALVIN_MINIMUM_GENERATED_BOXES
    )
    checks["generated_bbox_schema_valid_count"] = (
        int(candidate_calvin["generated_bbox_schema_valid_count"])
        >= ADR125_CALVIN_MINIMUM_SCHEMA_VALID_BOXES
    )
    variants = cast(
        dict[tuple[str, int, int], dict[str, object]],
        candidate_calvin["variants_by_key"],
    )
    validation_16 = [
        value
        for (partition, ordinal, _), value in variants.items()
        if partition == "validation" and ordinal == 16
    ]
    checks["validation_item_16_prompt_switch"] = (
        len(validation_16) == 2
        and all(bool(value["own_only_center_hit"]) for value in validation_16)
        and validation_16[0]["generated_bbox_qwen_xyxy"]
        != validation_16[1]["generated_bbox_qwen_xyxy"]
    )
    heldout_41_drawer = [
        value
        for (partition, ordinal, _), value in variants.items()
        if partition == "heldout" and ordinal == 41 and value["family"] == "drawer"
    ]
    checks["heldout_item_41_drawer_hit"] = bool(heldout_41_drawer) and all(
        bool(value["own_target_center_hit"]) for value in heldout_41_drawer
    )

    control_variants = cast(
        dict[tuple[str, int, int], dict[str, object]],
        control_calvin["variants_by_key"],
    )
    visual_review = []
    for key, candidate_variant in variants.items():
        control_variant = control_variants[key]
        reasons = []
        if (
            control_variant["generated_bbox_qwen_xyxy"]
            != candidate_variant["generated_bbox_qwen_xyxy"]
        ):
            reasons.append("prediction_bbox_changed")
        if bool(control_variant["own_target_center_hit"]) != bool(
            candidate_variant["own_target_center_hit"]
        ):
            reasons.append("own_target_center_hit_flipped")
        if bool(control_variant["own_only_center_hit"]) and not bool(
            candidate_variant["own_only_center_hit"]
        ):
            reasons.append("own_only_center_hit_regressed")
        if bool(candidate_variant["own_target_center_hit"]) and bool(
            candidate_variant["alternate_target_center_hit"]
        ):
            reasons.append("prediction_covers_both_target_centers")
        candidate_iou = _finite_nonnegative_metric(
            candidate_variant["own_target_iou"],
            name="candidate CALVIN own-target IoU",
        )
        control_iou = _finite_nonnegative_metric(
            control_variant["own_target_iou"],
            name="control CALVIN own-target IoU",
        )
        if candidate_iou < control_iou:
            reasons.append("own_target_iou_regressed")
        if reasons:
            visual_review.append(
                {
                    "partition": key[0],
                    "ordinal": key[1],
                    "variant_index": key[2],
                    "family": candidate_variant["family"],
                    "reasons": reasons,
                    "task_key": candidate_variant["task_key"],
                }
            )
    heldout_41_key = {
        "partition": "heldout",
        "ordinal": 41,
        "reason": "drawer prediction requires original-resolution broad-box review",
    }
    heldout_41_digest = _canonical_sha256(heldout_41_key)
    if not any(_canonical_sha256(item) == heldout_41_digest for item in visual_review):
        visual_review.append(heldout_41_key)
    return {
        "checks": checks,
        "control_family_hits": control_family_hits,
        "family_hits": family_hits,
        "generated_bbox_count": candidate_calvin["generated_bbox_count"],
        "generated_bbox_schema_valid_count": candidate_calvin["generated_bbox_schema_valid_count"],
        "lattice": candidate_calvin["lattice"],
        "mean_own_target_iou": candidate_calvin["mean_own_target_iou"],
        "numeric_status": "PASS" if all(checks.values()) else "FAIL",
        "summaries": summaries,
        "total_bidirectional_hits": candidate_calvin["total_bidirectional_hits"],
        "visual_review_required": visual_review,
    }


def _validate_training_binding(
    training: Mapping[str, Any],
    *,
    control_model: Mapping[str, object],
    candidate_model: Mapping[str, object],
    artifact_sha256: object,
    dataset_manifest_sha256: object,
    manifest_file_sha256: object,
    native_vl_patch_sha256: object,
    picf_code_revision: object,
    source_commit: object,
) -> dict[str, object]:
    if training.get("schema") != TRAINING_SCHEMA or training.get("status") != "PASS":
        raise ContractError("public retention comparison training report is not a passing v6 run")
    if training.get("picf_code_revision") != picf_code_revision:
        raise ContractError("public retention comparison training PICF revision changed")
    if training.get("source_commit") != source_commit:
        raise ContractError("public retention comparison training source commit changed")
    if training.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ContractError("public retention comparison training dataset manifest changed")
    if training.get("native_vl_patch_sha256") != native_vl_patch_sha256:
        raise ContractError("public retention comparison training patch changed")
    if training.get("world_size") != 2 or training.get("optimizer") != "torch.optim.AdamW":
        raise ContractError("public retention comparison training topology changed")
    if (
        training.get("fsdp2_placement") != "gpu-sharded"
        or training.get("cuda_allocator") != "expandable-segments"
    ):
        raise ContractError("public retention comparison training runtime mode changed")
    hyperparameters = _mapping(
        training.get("hyperparameters"),
        name="candidate training hyperparameters",
    )
    if dict(hyperparameters) != ADR125_TRAINING_HYPERPARAMETERS:
        raise ContractError("public retention comparison training hyperparameters changed")
    training_plan = _mapping(training.get("training_plan"), name="candidate training plan")
    expected_plan_fields = {
        "artifact_sha256": ADR125_CURRICULUM_ARTIFACT_SHA256,
        "file_sha256": ADR125_CURRICULUM_FILE_SHA256,
        "observation_mode": "official_native_once",
        "source_visual_lattices": [8, 14],
        "type": "official_native_once_curriculum",
        "visual_lattices": [8],
    }
    if any(training_plan.get(key) != value for key, value in expected_plan_fields.items()):
        raise ContractError("public retention comparison training plan changed")
    if training.get("observation_mode") != "official_native_once":
        raise ContractError("public retention comparison training observation mode changed")
    processor_lattices = _mapping(
        training.get("processor_lattices"),
        name="candidate training processor lattices",
    )
    expected_lattice = {
        "lattice": 8,
        "longest_edge_area": ADR125_QWEN_LONGEST_EDGE_AREA,
        "merge_size": 2,
        "patch_size": 16,
        "pixels_per_edge": 256,
        "shortest_edge_area": native_lattice_shortest_edge(8),
    }
    if dict(processor_lattices) != {"8": expected_lattice}:
        raise ContractError("public retention comparison training processor lattice changed")
    trainable_scope = _mapping(
        training.get("trainable_scope"),
        name="candidate training trainable scope",
    )
    if (
        trainable_scope.get("parameter_count") != ADR125_TRAINABLE_PARAMETER_COUNT
        or trainable_scope.get("trainable_numel") != ADR125_TRAINABLE_NUMEL
        or trainable_scope.get("schema_sha256") != ADR125_TRAINABLE_SCHEMA_SHA256
        or training.get("optimizer_state_parameter_count") != ADR125_TRAINABLE_PARAMETER_COUNT
    ):
        raise ContractError("public retention comparison training trainable scope changed")
    retention = _mapping(
        training.get("public_vl_retention"),
        name="candidate training public retention",
    )
    expected_retention_fields = {
        "artifact_sha256": artifact_sha256,
        "enabled": True,
        "global_loss_factors": {
            "referring": PUBLIC_NATIVE_VL_RETENTION_WEIGHT / 2,
            "vqa": PUBLIC_NATIVE_VL_RETENTION_WEIGHT / 2,
        },
        "manifest_file_sha256": manifest_file_sha256,
        "rank_loss_weight": PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
        "rank_streams": {"0": "referring", "1": "vqa"},
    }
    if any(retention.get(key) != value for key, value in expected_retention_fields.items()):
        raise ContractError("public retention comparison training artifact changed")
    expected_public_processor = native_processor_area_budget_contract(8)
    public_processor = _mapping(
        retention.get("processor"),
        name="candidate training public processor",
    )
    if dict(public_processor) != expected_public_processor:
        raise ContractError("public retention comparison training public processor changed")
    _validate_training_calvin_steps(training)
    _validate_training_public_steps(training, lattice=8)
    initial_qwen = _mapping(training.get("initial_qwen"), name="candidate training initial Qwen")
    initial_hashes = _normalized_model_hashes(
        initial_qwen.get("model_file_sha256"),
        name="candidate training initial Qwen",
    )
    candidate_hashes = _normalized_model_hashes(
        training.get("candidate_model_file_sha256"),
        name="candidate training output Qwen",
    )
    if initial_hashes != control_model.get("qwen_restore_model_file_sha256"):
        raise ContractError(
            "public retention comparison control is not the training initialization"
        )
    if candidate_hashes != candidate_model.get("qwen_restore_model_file_sha256"):
        raise ContractError("public retention comparison candidate is not the training output")
    initial_revision = initial_qwen.get("revision")
    if initial_revision != control_model.get("qwen_restore_revision"):
        raise ContractError("public retention comparison control Qwen revision changed")
    if candidate_model.get("qwen_restore_revision") != picf_code_revision:
        raise ContractError("public retention comparison candidate Qwen revision changed")
    return {
        "candidate_model_file_sha256": candidate_hashes,
        "initial_model_file_sha256": initial_hashes,
        "picf_code_revision": picf_code_revision,
    }


def compare_public_native_vl_retention_reports(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    control_report_sha256: str,
    candidate_report_sha256: str,
    candidate_training_report_sha256: str,
) -> dict[str, object]:
    """Apply the preregistered held-out NLL Pareto rule."""

    input_reports = {
        "candidate": {
            "schema": FIXED_X_SCHEMA,
            "sha256": _require_sha256(
                candidate_report_sha256,
                name="candidate report digest",
            ),
        },
        "candidate_training": {
            "schema": TRAINING_SCHEMA,
            "sha256": _require_sha256(
                candidate_training_report_sha256,
                name="candidate training report digest",
            ),
        },
        "control": {
            "schema": FIXED_X_SCHEMA,
            "sha256": _require_sha256(
                control_report_sha256,
                name="control report digest",
            ),
        },
    }

    if any(control.get(field) != candidate.get(field) for field in _COMMON_BINDINGS):
        raise ContractError("public retention comparison common evaluation binding changed")
    if (
        control.get("seed") != ADR125_FIXED_X_SEED
        or control.get("max_new_tokens") != ADR125_FIXED_X_MAX_NEW_TOKENS
        or control.get("picf_code_revision") != ADR125_PICF_CODE_REVISION
        or control.get("source_commit") != ADR125_LINGBOT_SOURCE_COMMIT
        or control.get("native_vl_patch_sha256") != ADR125_NATIVE_VL_PATCH_SHA256
    ):
        raise ContractError("public retention comparison frozen protocol changed")
    control_public = _public_section(control)
    candidate_public = _public_section(candidate)
    if control_public.get("artifact_sha256") != candidate_public.get("artifact_sha256"):
        raise ContractError("public retention comparison artifact changed")
    if _bound_records(control_public, lattice=8) != _bound_records(
        candidate_public,
        lattice=8,
    ):
        raise ContractError("public retention comparison evaluated different records")
    control_model = _model_binding(control, name="control")
    candidate_model = _model_binding(candidate, name="candidate")
    if (
        control_model["checkpoint_model_file_sha256"]
        != candidate_model["checkpoint_model_file_sha256"]
    ):
        raise ContractError("public retention comparison base checkpoint changed")
    # The producer validates the complete authoritative asset contract, while its
    # report records only root checkpoint files. The comparison binds every root
    # asset from that contract plus equality across fixed-X arms. TRAINING_SCHEMA
    # v6 has no host checkpoint manifest, so the nested depth/DINO assets cannot
    # honestly be cross-bound to the training report here.
    if (
        control_model["qwen_restore_model_file_sha256"] is None
        or candidate_model["qwen_restore_model_file_sha256"] is None
    ):
        raise ContractError("public retention comparison requires explicit Qwen restores")
    if control_model["qwen_restore_revision"] != ADR125_INITIAL_QWEN_REVISION:
        raise ContractError("public retention comparison initial Qwen revision changed")
    if (
        control_model["effective_qwen_weight_file_sha256"]
        == candidate_model["effective_qwen_weight_file_sha256"]
    ):
        raise ContractError("public retention comparison control and candidate models are equal")
    calvin = _calvin_gate(control, candidate)
    training_binding = _validate_training_binding(
        training,
        control_model=control_model,
        candidate_model=candidate_model,
        artifact_sha256=control_public.get("artifact_sha256"),
        dataset_manifest_sha256=control.get("dataset_manifest_sha256"),
        manifest_file_sha256=control_public.get("manifest_file_sha256"),
        native_vl_patch_sha256=control.get("native_vl_patch_sha256"),
        picf_code_revision=control.get("picf_code_revision"),
        source_commit=control.get("source_commit"),
    )
    control_summaries = _validated_summaries(control_public)
    candidate_summaries = _validated_summaries(candidate_public)
    per_record_generation_checks, public_generation_visual_review = _public_generation_gate(
        control_public,
        candidate_public,
    )

    families = {}
    gates = []
    for family in PUBLIC_NATIVE_VL_FAMILIES:
        control_family = control_summaries.get(family)
        candidate_family = candidate_summaries.get(family)
        if not isinstance(control_family, Mapping) or not isinstance(
            candidate_family,
            Mapping,
        ):
            raise ContractError("public retention comparison family summary is missing")
        if (
            control_family.get("record_count") != PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
            or candidate_family.get("record_count") != PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
        ):
            raise ContractError("public retention comparison family count changed")
        control_record_nll = _finite_nonnegative_metric(
            control_family.get("mean_record_nll"),
            name=f"{family} control record NLL",
        )
        candidate_record_nll = _finite_nonnegative_metric(
            candidate_family.get("mean_record_nll"),
            name=f"{family} candidate record NLL",
        )
        control_token_nll = _finite_nonnegative_metric(
            control_family.get("token_weighted_mean_nll"),
            name=f"{family} control token NLL",
        )
        candidate_token_nll = _finite_nonnegative_metric(
            candidate_family.get("token_weighted_mean_nll"),
            name=f"{family} candidate token NLL",
        )
        record_nonregression = candidate_record_nll <= control_record_nll
        token_improvement = candidate_token_nll < control_token_nll
        gates.extend((record_nonregression, token_improvement))
        generation_checks: dict[str, bool]
        generation_metrics: dict[str, dict[str, int | float]]
        if family == "referring":
            count_fields = (
                "generated_bbox_count",
                "generated_bbox_schema_valid_count",
                "target_center_hit_count",
            )
            generation_metrics = {}
            generation_checks = {}
            for field in count_fields:
                control_value = _count_metric(
                    control_family.get(field),
                    name=f"{family} control {field}",
                )
                candidate_value = _count_metric(
                    candidate_family.get(field),
                    name=f"{family} candidate {field}",
                )
                generation_metrics[field] = {
                    "candidate": candidate_value,
                    "control": control_value,
                    "delta": candidate_value - control_value,
                }
                generation_checks[f"{field}_nonregression"] = candidate_value >= control_value
            control_iou = _finite_nonnegative_metric(
                control_family.get("mean_target_iou"),
                name="referring control mean target IoU",
            )
            candidate_iou = _finite_nonnegative_metric(
                candidate_family.get("mean_target_iou"),
                name="referring candidate mean target IoU",
            )
            generation_metrics["mean_target_iou"] = {
                "candidate": candidate_iou,
                "control": control_iou,
                "delta": candidate_iou - control_iou,
            }
            generation_checks["mean_target_iou_nonregression"] = candidate_iou >= control_iou
            generation_checks["per_record_nonregression"] = per_record_generation_checks[
                "referring_per_record_nonregression"
            ]
        elif family == "vqa":
            field = "normalized_exact_match_count"
            control_value = _count_metric(
                control_family.get(field),
                name=f"{family} control {field}",
            )
            candidate_value = _count_metric(
                candidate_family.get(field),
                name=f"{family} candidate {field}",
            )
            generation_metrics = {
                field: {
                    "candidate": candidate_value,
                    "control": control_value,
                    "delta": candidate_value - control_value,
                }
            }
            generation_checks = {f"{field}_nonregression": candidate_value >= control_value}
            generation_checks["per_record_nonregression"] = per_record_generation_checks[
                "vqa_per_record_nonregression"
            ]
        else:
            raise RuntimeError("public retention comparison generation family drifted")
        generation_nonregression = all(generation_checks.values())
        gates.append(generation_nonregression)
        families[family] = {
            "candidate_mean_record_nll": candidate_record_nll,
            "candidate_minus_control_mean_record_nll": (candidate_record_nll - control_record_nll),
            "candidate_minus_control_token_weighted_mean_nll": (
                candidate_token_nll - control_token_nll
            ),
            "candidate_token_weighted_mean_nll": candidate_token_nll,
            "control_mean_record_nll": control_record_nll,
            "control_token_weighted_mean_nll": control_token_nll,
            "generation_checks": generation_checks,
            "generation_metrics": generation_metrics,
            "generation_nonregression": generation_nonregression,
            "mean_record_nll_nonregression": record_nonregression,
            "token_weighted_mean_nll_strictly_improves": token_improvement,
        }
    numeric_pass = all(gates) and calvin["numeric_status"] == "PASS"
    status = "PASS_PENDING_CALVIN_VISUAL_REVIEW" if numeric_pass else "FAIL"
    result: dict[str, object] = {
        "artifact_sha256": control_public["artifact_sha256"],
        "calvin": calvin,
        "candidate_checkpoint_dir": _checkpoint_dir(candidate, name="candidate"),
        "candidate_model": candidate_model,
        "candidate_training": training_binding,
        "control_checkpoint_dir": _checkpoint_dir(control, name="control"),
        "control_model": control_model,
        "families": families,
        "input_reports": input_reports,
        "schema": OUTPUT_SCHEMA,
        "status": status,
    }
    if public_generation_visual_review:
        result["public_generation_visual_review_required"] = public_generation_visual_review
    return result


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    control, control_digest = _load_report(
        args.control_report,
        expected_sha256=args.control_report_sha256,
        schema=FIXED_X_SCHEMA,
        require_pass=False,
    )
    candidate, candidate_digest = _load_report(
        args.candidate_report,
        expected_sha256=args.candidate_report_sha256,
        schema=FIXED_X_SCHEMA,
        require_pass=False,
    )
    training, training_digest = _load_report(
        args.candidate_training_report,
        expected_sha256=args.candidate_training_report_sha256,
        schema=TRAINING_SCHEMA,
        require_pass=True,
    )
    report = compare_public_native_vl_retention_reports(
        control,
        candidate,
        training,
        control_report_sha256=control_digest,
        candidate_report_sha256=candidate_digest,
        candidate_training_report_sha256=training_digest,
    )
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

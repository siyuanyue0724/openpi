#!/usr/bin/env python3
"""Audit M2 object cardinality without changing the trained model or gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.eval.cardinality import (  # noqa: E402
    binary_calibration_metrics,
    continuous_calibration_metrics,
    count_metrics,
    probability_distribution,
    query_usage_summary,
    select_count_threshold,
    task_usage_summary,
    threshold_sweep,
)
from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder  # noqa: E402
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_calvin import load_calvin_training_assets  # noqa: E402
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from picf_next.training.molmoact2_m2_source_coverage import (  # noqa: E402
    load_molmoact2_m2_source_coverage_recipe,
)
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _build_targets,
    _keys_for_split,
    _load_cache,
    _native_bank,
    _stack_batch,
)
from tools.run_molmoact2_m2_source_coverage_cloud import (  # noqa: E402
    _load_source_sidecar,
)

_FULL_TRAINING_CORE_PREFIX = "joint_bridge.sequence_bridge.core."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_representation.json",
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-coverage-config",
        type=Path,
        help="Hash-bound all-source recipe used to replace the base sparse sidecar.",
    )
    parser.add_argument(
        "--sidecar-artifact-root",
        type=Path,
        help="Artifact root containing the all-source sidecar named by its recipe.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checkpoint_control(checkpoint: Path) -> tuple[Path, str] | None:
    if not checkpoint.is_dir():
        return None
    control_path = checkpoint / "picf_control.json"
    try:
        control = json.loads(control_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("training checkpoint control is not valid JSON") from error
    if not isinstance(control, dict) or control.get("schema") != (
        "picf-next.checkpoint-control-manifest.v2"
    ):
        raise ValueError("unsupported training checkpoint control schema")
    state_files = control.get("state_files")
    if not isinstance(state_files, dict):
        raise ValueError("training checkpoint control has no state_files mapping")
    model_record = state_files.get("model.safetensors")
    if not isinstance(model_record, dict):
        raise ValueError("training checkpoint control has no bound model state")
    model_path = checkpoint / "model.safetensors"
    expected_size = model_record.get("size_bytes")
    expected_sha256 = model_record.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise ValueError("training checkpoint model binding is malformed")
    try:
        actual_size = model_path.stat().st_size
    except OSError as error:
        raise ValueError("training checkpoint model state is missing") from error
    if actual_size != expected_size:
        raise ValueError("training checkpoint model size differs from checkpoint control")
    actual_sha256 = _sha256(model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("training checkpoint model hash differs from checkpoint control")
    return model_path, actual_sha256


def _select_current_frame_state(
    state: Mapping[str, Any],
    *,
    expected_names: set[str],
) -> dict[str, Any]:
    direct_names = expected_names.intersection(state)
    prefixed_names = {name for name in state if name.startswith(_FULL_TRAINING_CORE_PREFIX)}
    if direct_names and prefixed_names:
        raise ValueError("checkpoint mixes direct and full-training PICF core keys")
    if prefixed_names:
        full_core = {
            name.removeprefix(_FULL_TRAINING_CORE_PREFIX): state[name] for name in prefixed_names
        }
        unexpected = set(full_core).difference(expected_names)
        if any(not name.startswith("posterior_filter.") for name in unexpected):
            raise ValueError("full-training checkpoint contains unexpected PICF core keys")
        return {name: full_core[name] for name in expected_names.intersection(full_core)}
    return dict(state)


def _load_current_frame_state(
    checkpoint: Path,
    *,
    expected_names: set[str],
) -> tuple[dict[str, Any], Path, str]:
    import torch

    checkpoint = checkpoint.expanduser().resolve()
    control_binding = _read_checkpoint_control(checkpoint)
    model_path = control_binding[0] if control_binding is not None else checkpoint
    if not model_path.is_file():
        raise ValueError("cardinality checkpoint is not a file or bound checkpoint directory")
    checkpoint_sha256 = control_binding[1] if control_binding is not None else _sha256(model_path)
    if model_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(model_path, framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            direct_names = expected_names.intersection(names)
            prefixed_names = {name for name in names if name.startswith(_FULL_TRAINING_CORE_PREFIX)}
            if direct_names and prefixed_names:
                raise ValueError("checkpoint mixes direct and full-training PICF core keys")
            if prefixed_names:
                stripped_names = {
                    name.removeprefix(_FULL_TRAINING_CORE_PREFIX) for name in prefixed_names
                }
                unexpected = stripped_names.difference(expected_names)
                if any(not name.startswith("posterior_filter.") for name in unexpected):
                    raise ValueError("full-training checkpoint contains unexpected PICF core keys")
                selected_names = {
                    f"{_FULL_TRAINING_CORE_PREFIX}{name}"
                    for name in expected_names.intersection(stripped_names)
                }
            else:
                selected_names = names
            state = {name: handle.get_tensor(name) for name in selected_names}
    else:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("cardinality checkpoint must contain a state mapping")
        state = payload["model"] if set(payload) == {"model"} else payload
        if not isinstance(state, Mapping):
            raise ValueError("cardinality checkpoint model payload must be a mapping")
    return (
        _select_current_frame_state(state, expected_names=expected_names),
        model_path,
        checkpoint_sha256,
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(left) < 2 or len(left) != len(right):
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    statistic = float(spearmanr(left, right).statistic)
    return statistic if math.isfinite(statistic) else None


def _split_audit(
    *,
    model: Any,
    criterion: ObjectSetCriterion,
    target_builder: CalvinVisibleObjectTargetBuilder,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
    batch_size: int,
    device: Any,
    calibration_bins: int,
) -> dict[str, Any]:
    import torch

    probabilities: list[float] = []
    localization_confidences: list[float] = []
    measurement_probabilities: list[float] = []
    mask_qualities: list[float] = []
    mask_coherence_scores: list[float] = []
    matched_localization_confidences: list[float] = []
    matched_soft_ious: list[float] = []
    matched_mask_qualities: list[float] = []
    matched_mask_coherence_scores: list[float] = []
    training_scores: list[float] = []
    labels: list[int] = []
    ownership_mass: list[float] = []
    dominant_fraction: list[float] = []
    per_sample: list[dict[str, Any]] = []
    model.eval()
    query_count = model.discovery.config.num_queries

    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        tokens, valid, records = _stack_batch(cache, batch_keys, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(_native_bank(tokens, valid))
        targets = _build_targets(
            target_builder=target_builder,
            records=records,
            token_valid=output.projection.token_valid,
            target_dtype=output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=token_count,
        )
        result = criterion(output.discovery, targets)
        for batch_index, (key, target, match) in enumerate(
            zip(batch_keys, targets, result.matches, strict=True)
        ):
            record = cache[key][2]
            posterior = output.discovery.existence[batch_index].float()
            localization_confidence = output.discovery.localization_confidence[batch_index].float()
            measurement_probability = output.discovery.measurement_probability[batch_index].float()
            mask_quality = output.discovery.mask_quality[batch_index].float()
            mask_coherence_score = output.discovery.mask_coherence_score[batch_index].float()
            training = output.discovery.training_existence_score[batch_index].float()
            query_label = torch.zeros(query_count, device=device, dtype=torch.long)
            query_label[match.prediction_indices] = 1
            supervised = target.supervision_valid
            ownership = output.discovery.ownership[batch_index, supervised].float()
            object_ownership = ownership[:, :-1]
            mass = object_ownership.mean(dim=0)
            winner = ownership.argmax(dim=-1)
            dominant = torch.stack(
                [(winner == index).float().mean() for index in range(query_count)]
            )
            soft_iou_by_query: list[float | None] = [None] * query_count
            if match.prediction_indices.numel():
                predicted = ownership[:, match.prediction_indices]
                expected = target.ownership[supervised][:, match.target_indices].float()
                target_mass = expected.sum(dim=0)
                intersection = (predicted * expected).sum(dim=0)
                union = predicted.sum(dim=0) + target_mass - intersection
                soft_iou = intersection / union.clamp_min(1e-6)
                quality_supervised = target_mass > 0.0
                for match_index, query_index in enumerate(match.prediction_indices.tolist()):
                    if not bool(quality_supervised[match_index].item()):
                        continue
                    iou = float(soft_iou[match_index].item())
                    soft_iou_by_query[int(query_index)] = iou
                    matched_soft_ious.append(iou)
                    matched_localization_confidences.append(
                        float(localization_confidence[query_index].item())
                    )
                    matched_mask_qualities.append(float(mask_quality[query_index].item()))
                    matched_mask_coherence_scores.append(
                        float(mask_coherence_score[query_index].item())
                    )

            posterior_values = [float(value) for value in posterior.tolist()]
            localization_confidence_values = [
                float(value) for value in localization_confidence.tolist()
            ]
            measurement_probability_values = [
                float(value) for value in measurement_probability.tolist()
            ]
            quality_values = [float(value) for value in mask_quality.tolist()]
            coherence_values = [float(value) for value in mask_coherence_score.tolist()]
            training_values = [float(value) for value in training.tolist()]
            label_values = [int(value) for value in query_label.tolist()]
            mass_values = [float(value) for value in mass.tolist()]
            dominant_values = [float(value) for value in dominant.tolist()]
            probabilities.extend(posterior_values)
            localization_confidences.extend(localization_confidence_values)
            measurement_probabilities.extend(measurement_probability_values)
            mask_qualities.extend(quality_values)
            mask_coherence_scores.extend(coherence_values)
            training_scores.extend(training_values)
            labels.extend(label_values)
            ownership_mass.extend(mass_values)
            dominant_fraction.extend(dominant_values)
            per_sample.append(
                {
                    "sample_key": key,
                    "target_request_contract": str(
                        record.get("target_request_contract", "language_segment")
                    ),
                    "segment_index": (
                        int(record["segment_index"]) if "segment_index" in record else None
                    ),
                    "source_block_index": (
                        int(record["source_block_index"])
                        if "source_block_index" in record
                        else None
                    ),
                    "global_index": int(record["global_index"]),
                    "task_key": str(record["task_key"]),
                    "target_count": target.num_objects,
                    "matched_query_indices": [
                        int(value) for value in match.prediction_indices.tolist()
                    ],
                    "existence_probability": posterior_values,
                    "localization_confidence": localization_confidence_values,
                    "measurement_probability": measurement_probability_values,
                    "mask_quality": quality_values,
                    "mask_coherence_score": coherence_values,
                    "matched_soft_iou_by_query": soft_iou_by_query,
                    "training_existence_score": training_values,
                    "mean_object_ownership_mass": mass_values,
                    "dominant_token_fraction": dominant_values,
                }
            )
        del tokens, valid, output

    sample_probabilities = [row["existence_probability"] for row in per_sample]
    sample_measurement_probabilities = [row["measurement_probability"] for row in per_sample]
    target_counts = [int(row["target_count"]) for row in per_sample]
    matched_probability = [
        probability for probability, label in zip(probabilities, labels, strict=True) if label
    ]
    unmatched_probability = [
        probability for probability, label in zip(probabilities, labels, strict=True) if not label
    ]
    matched_mass = [value for value, label in zip(ownership_mass, labels, strict=True) if label]
    unmatched_mass = [
        value for value, label in zip(ownership_mass, labels, strict=True) if not label
    ]
    matched_dominant = [
        value for value, label in zip(dominant_fraction, labels, strict=True) if label
    ]
    unmatched_dominant = [
        value for value, label in zip(dominant_fraction, labels, strict=True) if not label
    ]
    sweep = threshold_sweep(sample_probabilities, target_counts)
    return {
        "sample_count": len(per_sample),
        "query_count": query_count,
        "usage": query_usage_summary(per_sample, query_count=query_count),
        "per_task_usage": task_usage_summary(per_sample, query_count=query_count),
        "query_calibration": binary_calibration_metrics(
            probabilities,
            labels,
            bins=calibration_bins,
        ),
        "training_score_calibration": binary_calibration_metrics(
            training_scores,
            labels,
            bins=calibration_bins,
        ),
        "measurement_probability_against_match": binary_calibration_metrics(
            measurement_probabilities,
            labels,
            bins=calibration_bins,
        ),
        "localization_confidence": {
            "matched": probability_distribution(
                [
                    value
                    for value, label in zip(localization_confidences, labels, strict=True)
                    if label
                ]
            ),
            "unmatched": probability_distribution(
                [
                    value
                    for value, label in zip(localization_confidences, labels, strict=True)
                    if not label
                ]
            ),
            "soft_iou_calibration": (
                continuous_calibration_metrics(
                    matched_localization_confidences,
                    matched_soft_ious,
                    bins=calibration_bins,
                )
                if matched_soft_ious
                else None
            ),
        },
        "mask_quality": {
            "matched": probability_distribution(
                [value for value, label in zip(mask_qualities, labels, strict=True) if label]
            ),
            "unmatched": probability_distribution(
                [value for value, label in zip(mask_qualities, labels, strict=True) if not label]
            ),
        },
        "mask_coherence_score": {
            "matched": probability_distribution(
                [value for value, label in zip(mask_coherence_scores, labels, strict=True) if label]
            ),
            "unmatched": probability_distribution(
                [
                    value
                    for value, label in zip(mask_coherence_scores, labels, strict=True)
                    if not label
                ]
            ),
        },
        "quality_correctness_coupling": {
            "localization_confidence_vs_soft_iou_spearman": _spearman(
                matched_localization_confidences,
                matched_soft_ious,
            ),
            "mask_quality_vs_soft_iou_spearman": _spearman(
                matched_mask_qualities,
                matched_soft_ious,
            ),
            "mask_coherence_score_vs_soft_iou_spearman": _spearman(
                matched_mask_coherence_scores,
                matched_soft_ious,
            ),
            "matched_soft_iou": probability_distribution(matched_soft_ious),
        },
        "count_at_physical_posterior_half": count_metrics(
            sample_probabilities,
            target_counts,
            threshold=0.5,
        ),
        "measurement_probability_count_at_half_diagnostic": count_metrics(
            sample_measurement_probabilities,
            target_counts,
            threshold=0.5,
        ),
        "threshold_sweep": sweep,
        "best_count_threshold_on_this_split": select_count_threshold(sweep),
        "existence_ownership_coupling": {
            "existence_vs_mean_object_ownership_mass_spearman": _spearman(
                probabilities,
                ownership_mass,
            ),
            "existence_vs_dominant_token_fraction_spearman": _spearman(
                probabilities,
                dominant_fraction,
            ),
            "matched_mean_object_ownership_mass": probability_distribution(matched_mass),
            "unmatched_mean_object_ownership_mass": probability_distribution(unmatched_mass),
            "matched_dominant_token_fraction": probability_distribution(matched_dominant),
            "unmatched_dominant_token_fraction": probability_distribution(unmatched_dominant),
            "matched_existence_probability": probability_distribution(matched_probability),
            "unmatched_existence_probability": probability_distribution(unmatched_probability),
        },
        "per_sample": per_sample,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    import torch

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M2 cardinality audit requires CUDA")
    if args.calibration_bins <= 0:
        raise ValueError("calibration bins must be positive")
    source_coverage_requested = args.source_coverage_config is not None
    if source_coverage_requested != (args.sidecar_artifact_root is not None):
        raise ValueError(
            "source coverage config and sidecar artifact root must be supplied together"
        )
    device = torch.device(args.device)
    recipe = load_molmoact2_m2_recipe(args.config.resolve())
    foundation = recipe.load_foundation(_ROOT)
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
    )
    source_coverage_report = None
    if source_coverage_requested:
        source_recipe = load_molmoact2_m2_source_coverage_recipe(
            args.source_coverage_config.expanduser().resolve()
        )
        source_base = source_recipe.load_base_m2(_ROOT)
        if source_base.recipe_sha256 != recipe.recipe_sha256:
            raise ValueError("source-coverage recipe does not wrap the selected M2 recipe")
        assets, source_coverage_report = _load_source_sidecar(
            artifact_root=args.sidecar_artifact_root.expanduser().resolve(),
            recipe=source_recipe,
            assets=assets,
        )
        source_coverage_report = {
            **source_coverage_report,
            "config": str(args.source_coverage_config.expanduser().resolve()),
            "recipe_sha256": source_recipe.recipe_sha256,
        }
    feature_cache = args.feature_cache.expanduser().resolve()
    cache_manifest, cache = _load_cache(feature_cache, recipe)
    torch.manual_seed(recipe.optimization.seed)
    model = foundation.core_config.build_current_frame().to(device)
    checkpoint = args.checkpoint.expanduser().resolve()
    state, checkpoint_model_path, checkpoint_sha256 = _load_current_frame_state(
        checkpoint,
        expected_names=set(model.state_dict()),
    )
    incompatible = model.load_state_dict(state, strict=False)
    fresh_quality_keys = {
        "discovery.localization_confidence_head.weight",
        "discovery.localization_confidence_head.bias",
    }
    if incompatible.unexpected_keys or set(incompatible.missing_keys) not in (
        set(),
        fresh_quality_keys,
    ):
        raise ValueError("cardinality checkpoint does not map exactly onto current-frame PICF")
    if set(incompatible.missing_keys) == fresh_quality_keys:
        quality_head = model.discovery.localization_confidence_head
        if bool(torch.count_nonzero(quality_head.weight)) or bool(
            torch.count_nonzero(quality_head.bias)
        ):
            raise ValueError("fresh localization-confidence head is not neutral")
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(device)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)

    train = _split_audit(
        model=model,
        criterion=criterion,
        target_builder=target_builder,
        cache=cache,
        keys=_keys_for_split(cache, "train"),
        layout_payload=cache_manifest["processor_layout"],
        token_count=recipe.cache.token_count,
        batch_size=recipe.optimization.batch_size,
        device=device,
        calibration_bins=args.calibration_bins,
    )
    validation = _split_audit(
        model=model,
        criterion=criterion,
        target_builder=target_builder,
        cache=cache,
        keys=_keys_for_split(cache, "validation"),
        layout_payload=cache_manifest["processor_layout"],
        token_count=recipe.cache.token_count,
        batch_size=recipe.optimization.batch_size,
        device=device,
        calibration_bins=args.calibration_bins,
    )
    heldout = _split_audit(
        model=model,
        criterion=criterion,
        target_builder=target_builder,
        cache=cache,
        keys=_keys_for_split(cache, "heldout"),
        layout_payload=cache_manifest["processor_layout"],
        token_count=recipe.cache.token_count,
        batch_size=recipe.optimization.batch_size,
        device=device,
        calibration_bins=args.calibration_bins,
    )
    selected_threshold = float(validation["best_count_threshold_on_this_split"]["threshold"])
    heldout_at_validation_threshold = count_metrics(
        [row["existence_probability"] for row in heldout["per_sample"]],
        [int(row["target_count"]) for row in heldout["per_sample"]],
        threshold=selected_threshold,
    )
    report = {
        "schema": "picf-next.molmoact2-m2-cardinality-audit.v3",
        "authorizes_training_changes": False,
        "interpretation_boundary": (
            "Hungarian-matched query labels are diagnostic and model-dependent; "
            "validation-selected thresholds are not production parameters."
        ),
        "config": str(args.config.resolve()),
        "recipe_sha256": recipe.recipe_sha256,
        "feature_cache": str(feature_cache),
        "feature_cache_manifest_sha256": _sha256(feature_cache / "manifest.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_model_path": str(checkpoint_model_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_coverage": source_coverage_report,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "train": train,
        "validation": validation,
        "heldout": heldout,
        "validation_selected_threshold": selected_threshold,
        "heldout_at_validation_selected_threshold": heldout_at_validation_threshold,
    }
    output = args.output.expanduser().resolve()
    _write_json_atomic(output, report)
    summary = {
        "validation_selected_threshold": selected_threshold,
        "validation_count_at_half": validation["count_at_physical_posterior_half"],
        "heldout_count_at_half": heldout["count_at_physical_posterior_half"],
        "heldout_at_validation_selected_threshold": heldout_at_validation_threshold,
        "heldout_query_calibration": {
            name: heldout["query_calibration"][name]
            for name in (
                "brier",
                "negative_log_likelihood",
                "expected_calibration_error",
                "maximum_calibration_error",
            )
        },
        "heldout_localization_confidence": heldout["localization_confidence"],
        "heldout_quality_correctness_coupling": heldout["quality_correctness_coupling"],
        "heldout_existence_ownership_coupling": heldout["existence_ownership_coupling"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

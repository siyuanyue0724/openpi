#!/usr/bin/env python3
"""Run fixed-sample counterfactual acceptance for a Stage-B candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from picf_next.data.calvin import CALVIN_HOST_IMAGE_KEYS  # noqa: E402
from picf_next.eval.stationary_lifecycle import (  # noqa: E402
    build_stationary_lifecycle_calibration,
)
from picf_next.eval.stationary_replay import (  # noqa: E402
    STATIONARY_FIXED_REPLAY_FAIL,
    STATIONARY_FIXED_REPLAY_METRICS,
    STATIONARY_FIXED_REPLAY_PASS,
    STATIONARY_FIXED_REPLAY_SCHEMA,
    aggregate_replay_measurements,
    compare_stationary_replay_summaries,
    validate_stationary_fixed_replay,
)
from picf_next.eval.stationary_runtime import (  # noqa: E402
    build_stationary_runtime_probe,
)
from picf_next.eval.stationary_visual import (  # noqa: E402
    STATIONARY_VISUAL_ARTIFACTS_SCHEMA,
    validate_stationary_visual_artifacts,
)
from picf_next.hosts.molmoact2_layout import (  # noqa: E402
    MOLMO_VISION_PATCH_MODALITY,
)
from picf_next.training.m2_acceptance import validate_axis_calibrated_m2  # noqa: E402
from picf_next.training.stage_checkpoints import (  # noqa: E402
    load_stationary_temporal_checkpoint,
    sha256_file,
)
from picf_next.training.stationary_calvin_stage import (  # noqa: E402
    build_stationary_temporal_trainer,
    load_stationary_calvin_stage_assets,
    load_stationary_calvin_stage_definition,
)
from picf_next.training.temporal_clips import (  # noqa: E402
    build_distributed_stationary_temporal_clip_plan,
)

_PREFIX_LENGTHS = (0, 8, 32, 128)
_SPLIT_NAMES = ("validation", "heldout")
_MODEL_NAMES = ("fresh_m2", "candidate")
_SAFE_NAME = re.compile(r"[^a-z0-9]+")
_ABSOLUTE_TOLERANCE = 1e-6
_DEFAULT_SEED = 20260720
_DEFAULT_STEPS_PER_SPLIT = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json",
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--m2-report", type=Path, required=True)
    parser.add_argument("--m2-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-steps-per-split", type=int, default=_DEFAULT_STEPS_PER_SPLIT)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    encoded = (_canonical_json(payload) + "\n").encode("ascii")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _publish_visual_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Publish visual evidence only after its complete on-disk contract validates."""

    validate_stationary_visual_artifacts(manifest, evidence_root=output_dir)
    _write_json_atomic(output_dir / "visual_artifacts.json", manifest)


def _git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise ValueError("stationary audit Git revision is malformed")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("stationary audit requires a clean committed worktree")
    return revision


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    steps = args.optimizer_steps_per_split
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < len(_PREFIX_LENGTHS)
        or steps % len(_PREFIX_LENGTHS)
    ):
        raise ValueError("optimizer steps per split must be a positive multiple of four")
    if not isinstance(args.seed, int) or isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("stationary replay seed must be a non-negative integer")
    output = args.output_dir.expanduser().resolve()
    if not str(output).startswith("/mnt/"):
        raise ValueError("stationary replay evidence must persist beneath /mnt")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)


def _validate_candidate_report(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    definition: Any,
    provenance: Any,
    binding: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema",
        "status",
        "stage_recipe_sha256",
        "source_coverage_recipe_sha256",
        "foundation_recipe_sha256",
        "structural_recipe_sha256",
        "clip_plan_sha256",
        "optimizer_steps",
        "world_size",
        "prefix_lengths",
        "train_length",
        "required_future_horizon",
        "action_weight",
        "checkpoint_sha256",
        "metrics_sha256",
        "completed_optimizer_steps",
        "long_training_authorized",
    }
    if set(report) != expected_fields:
        raise ValueError("Stage-B candidate report fields changed")
    expected = {
        "schema": "picf-next.stationary-temporal-candidate-report.v1",
        "status": "CANDIDATE_REQUIRES_FIXED_CHECKPOINT_AUDIT",
        "stage_recipe_sha256": definition.stage.recipe_sha256,
        "source_coverage_recipe_sha256": definition.source_coverage.recipe_sha256,
        "foundation_recipe_sha256": definition.historical_foundation.recipe_sha256,
        "structural_recipe_sha256": definition.structural_foundation.recipe_sha256,
        "clip_plan_sha256": definition.clip_plan.plan_sha256,
        "optimizer_steps": definition.stage.optimizer.optimizer_steps,
        "world_size": definition.stage.distributed.world_size,
        "prefix_lengths": list(definition.stage.clip.prefix_lengths),
        "train_length": definition.stage.clip.train_length,
        "required_future_horizon": definition.maximum_horizon,
        "action_weight": 0.0,
        "checkpoint_sha256": checkpoint_sha256,
        "completed_optimizer_steps": definition.stage.optimizer.optimizer_steps,
        "long_training_authorized": False,
    }
    for name, value in expected.items():
        if report[name] != value:
            raise ValueError(f"Stage-B candidate report changed: {name}")
    if provenance.stage_recipe_sha256 != definition.stage.recipe_sha256:
        raise ValueError("candidate checkpoint stage recipe differs from replay")
    if provenance.source_coverage_recipe_sha256 != definition.source_coverage.recipe_sha256:
        raise ValueError("candidate checkpoint source coverage differs from replay")
    if provenance.foundation_recipe_sha256 != definition.historical_foundation.recipe_sha256:
        raise ValueError("candidate checkpoint foundation differs from replay")
    if provenance.m2_checkpoint_sha256 != binding["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint M2 initialization differs from replay")
    if provenance.feature_cache_manifest_sha256 != binding["feature_cache_manifest_sha256"]:
        raise ValueError("candidate checkpoint feature cache differs from replay")
    if provenance.clip_plan_sha256 != definition.clip_plan.plan_sha256:
        raise ValueError("candidate checkpoint training clip plan differs from report")
    if (
        provenance.optimizer_steps != definition.stage.optimizer.optimizer_steps
        or provenance.state_parameter_version != provenance.optimizer_steps
        or provenance.recurrent_state_serialized is not False
    ):
        raise ValueError("candidate checkpoint parameter-version contract changed")


def _soft_iou(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    prediction = prediction.float()[valid]
    target = target.float()[valid]
    if prediction.numel() == 0:
        raise ValueError("stationary replay object overlap has no supervised tokens")
    denominator = torch.maximum(prediction, target).sum()
    if float(denominator) <= 0.0:
        return 0.0
    return float((torch.minimum(prediction, target).sum() / denominator).cpu())


def _object_metrics(trainer: Any, output: Any, supervision: Any) -> dict[str, float]:
    final = output.train_outputs[-1]
    target = supervision.set_targets[0]
    valid = target.supervision_valid.bool()
    set_output = trainer.objective.set_criterion(final.discovery, supervision.set_targets)
    match = set_output.matches[0]
    match_by_target = {
        int(target_index): int(prediction_index)
        for prediction_index, target_index in zip(
            match.prediction_indices.detach().cpu().tolist(),
            match.target_indices.detach().cpu().tolist(),
            strict=True,
        )
    }
    discovery_iou = []
    for target_index in range(target.num_objects):
        query = match_by_target.get(target_index)
        discovery_iou.append(
            0.0
            if query is None
            else _soft_iou(
                final.discovery.ownership[0, :, query],
                target.ownership[:, target_index],
                valid,
            )
        )

    keys = target.temporal_identity_keys
    if keys is None or len(keys) != target.num_objects:
        raise ValueError("stationary replay requires exact visible physical identity keys")
    row_by_key = {
        key: row
        for row, key in enumerate(output.objective.loss_track_keys_by_row[0])
        if key is not None
    }
    posterior_iou = []
    covered = 0
    for target_index, key in enumerate(keys):
        row = row_by_key.get(key)
        if row is None or not bool(final.posterior.belief.valid[0, row]):
            posterior_iou.append(0.0)
            continue
        covered += 1
        posterior_iou.append(
            _soft_iou(
                final.posterior.ownership[0, :, row],
                target.ownership[:, target_index],
                valid,
            )
        )
    denominator = max(1, target.num_objects)
    return {
        "discovery_soft_iou": sum(discovery_iou) / denominator,
        "posterior_soft_iou": sum(posterior_iou) / denominator,
        "posterior_identity_coverage": covered / denominator,
    }


def _run_model(
    trainer: Any,
    batch: Any,
    *,
    device: torch.device,
) -> tuple[Any, dict[str, float], dict[str, int | float]]:
    trainer.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
            cache_enabled=False,
        ),
    ):
        output = trainer(
            batch.observations,
            prefix_length=batch.prefix_length,
            supervision_builder=batch.build_supervision,
            geometry_builder=batch.build_geometry_rollout,
        )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    final_supervision = batch.build_supervision(len(batch.observations) - 1)
    losses = output.objective.losses
    metrics = {
        "loss_total": float(losses["loss_total"].detach().float().cpu()),
        "loss_set": float(losses["loss_set"].detach().float().cpu()),
        "loss_dynamics": float(losses["loss_dynamics"].detach().float().cpu()),
        "loss_dynamics_survival": float(losses["loss_dynamics_survival"].detach().float().cpu()),
        "loss_dynamics_visibility": float(
            losses["loss_dynamics_visibility"].detach().float().cpu()
        ),
        "loss_binding": float(losses["loss_binding"].detach().float().cpu()),
        "assignment_conflicts_per_clip": float(
            output.prefix_assignment_conflicts
            + output.objective.diagnostics["loss_track_assignment_conflicts"]
        ),
        **_object_metrics(trainer, output, final_supervision),
    }
    if set(metrics) != set(STATIONARY_FIXED_REPLAY_METRICS) or any(
        not math.isfinite(value) for value in metrics.values()
    ):
        raise RuntimeError("stationary replay produced malformed or non-finite metrics")
    runtime = {
        "prefix_length": batch.prefix_length,
        "transition_count": len(batch.observations),
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    return output, metrics, runtime


def _color(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [0, 128, 128],
            [170, 110, 40],
            [128, 0, 0],
        ],
        dtype=np.float32,
    )
    return palette[index % len(palette)]


def _overlay(
    source: np.ndarray,
    probability: np.ndarray,
    *,
    supervised: np.ndarray | None = None,
) -> np.ndarray:
    from PIL import Image

    patch_count = probability.shape[0]
    side = math.isqrt(patch_count)
    if side * side != patch_count or probability.shape[1] < 2:
        raise ValueError("stationary visual global crop is not one square object/context grid")
    height, width = source.shape[:2]
    object_probability = np.clip(probability[:, :-1].astype(np.float32), 0.0, 1.0)
    object_mass = np.clip(object_probability.sum(axis=-1), 0.0, 1.0)
    colors = np.stack([_color(index) for index in range(object_probability.shape[1])])
    weighted_color = object_probability @ colors
    weighted_color = np.divide(
        weighted_color,
        object_mass[:, None],
        out=np.zeros_like(weighted_color),
        where=object_mass[:, None] > 0.0,
    )
    tint = np.asarray(
        Image.fromarray(
            weighted_color.reshape(side, side, 3).astype(np.uint8),
        ).resize((width, height), Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    alpha = np.asarray(
        Image.fromarray(object_mass.reshape(side, side)).resize(
            (width, height), Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    )[..., None]
    result = source.astype(np.float32).copy()
    result = (1.0 - 0.65 * alpha) * result + 0.65 * alpha * tint
    if supervised is not None:
        known = supervised.reshape(side, side).astype(np.uint8) * 255
        known = (
            np.asarray(Image.fromarray(known).resize((width, height), Image.Resampling.NEAREST)) > 0
        )
        result[~known] = 0.35 * result[~known] + 0.65 * np.asarray(
            [128, 128, 128], dtype=np.float32
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_centroids(image: Any, probability: np.ndarray) -> None:
    from PIL import ImageDraw

    side = math.isqrt(probability.shape[0])
    if side * side != probability.shape[0]:
        raise ValueError("stationary visual centroid grid is not square")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    yy, xx = np.mgrid[0:side, 0:side]
    for column in range(probability.shape[1] - 1):
        mass = probability[:, column].reshape(side, side)
        total = float(mass.sum())
        if total <= 1e-5:
            continue
        x = float((mass * (xx + 0.5)).sum() / total) / side * width
        y = float((mass * (yy + 0.5)).sum() / total) / side * height
        color = tuple(int(value) for value in _color(column))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2)


def _panel(title: str, array: np.ndarray, probability: np.ndarray | None = None) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(array).resize((260, 260), Image.Resampling.NEAREST)
    if probability is not None:
        _draw_centroids(image, probability)
    canvas = Image.new("RGB", (260, 300), "white")
    canvas.paste(image, (0, 40))
    ImageDraw.Draw(canvas).text((6, 7), title, fill="black")
    return canvas


def _aligned_probabilities(
    trainer: Any,
    output: Any,
    supervision: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
    final = output.train_outputs[-1]
    target = supervision.set_targets[0]
    lifecycle = supervision.lifecycle_targets[0]
    if lifecycle is None:
        raise ValueError("stationary visual requires a complete lifecycle inventory")
    keys = tuple(lifecycle.alive_identity_keys)
    visible_keys = target.temporal_identity_keys
    if visible_keys is None:
        raise ValueError("stationary visual requires current physical identity keys")
    if len(set(keys)) != len(keys) or not set(visible_keys).issubset(keys):
        raise ValueError("stationary visual lifecycle and current identity inventories disagree")
    visibility = lifecycle.visibility
    visibility_supervised = lifecycle.visibility_supervised
    if (
        visibility is None
        or visibility_supervised is None
        or visibility.shape != (len(keys),)
        or visibility_supervised.shape != (len(keys),)
    ):
        raise ValueError("stationary visual lifecycle visibility is incomplete")

    raw_target = target.ownership.detach().float().cpu().numpy()
    aligned_target = np.zeros((raw_target.shape[0], len(keys) + 1), dtype=raw_target.dtype)
    key_to_index = {key: index for index, key in enumerate(keys)}
    for source_index, key in enumerate(visible_keys):
        aligned_target[:, key_to_index[key]] = raw_target[:, source_index]
    aligned_target[:, -1] = raw_target[:, -1]

    discovery = np.zeros_like(aligned_target)
    discovery[:, -1] = final.discovery.context_ownership[0].detach().float().cpu().numpy()
    match = trainer.objective.set_criterion(final.discovery, supervision.set_targets).matches[0]
    query_by_key: dict[str, int] = {}
    for query, target_index in zip(
        match.prediction_indices.detach().cpu().tolist(),
        match.target_indices.detach().cpu().tolist(),
        strict=True,
    ):
        key = visible_keys[target_index]
        query_by_key[key] = query
        discovery[:, key_to_index[key]] = (
            final.discovery.ownership[0, :, query].detach().float().cpu().numpy()
        )

    posterior = np.zeros_like(discovery)
    row_by_key = {
        key: row
        for row, key in enumerate(output.objective.loss_track_keys_by_row[0])
        if key is not None
    }
    raw = final.posterior.ownership[0].detach().float().cpu().numpy()
    for target_index, key in enumerate(keys):
        row = row_by_key.get(key)
        if row is not None and bool(final.posterior.belief.valid[0, row]):
            posterior[:, target_index] = raw[:, row]
    posterior[:, -1] = np.clip(1.0 - posterior[:, :-1].sum(axis=-1), 0.0, 1.0)

    diagnostics = []
    for target_index, key in enumerate(keys):
        query = query_by_key.get(key)
        row = row_by_key.get(key)
        visibility_is_supervised = bool(visibility_supervised[target_index].detach().cpu().item())
        diagnostic: dict[str, object] = {
            "identity_key": key,
            "target_currently_measurable": key in visible_keys,
            "target_visibility": (
                float(visibility[target_index].detach().float().cpu().item())
                if visibility_is_supervised
                else None
            ),
            "target_visibility_supervised": visibility_is_supervised,
            "target_ownership_mass": float(aligned_target[:, target_index].sum()),
            "discovery_existence": None,
            "discovery_localization": None,
            "discovery_measurement": None,
            "discovery_ownership_mass": 0.0,
            "posterior_association": 0.0,
            "posterior_existence": None,
            "posterior_visibility": None,
            "posterior_ownership_mass": 0.0,
            "posterior_map_present": False,
        }
        if query is not None:
            diagnostic.update(
                {
                    "discovery_existence": float(
                        final.discovery.existence[0, query].detach().float().cpu().item()
                    ),
                    "discovery_localization": float(
                        final.discovery.localization_confidence[0, query]
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    ),
                    "discovery_measurement": float(
                        final.discovery.measurement_probability[0, query]
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    ),
                    "discovery_ownership_mass": float(
                        final.discovery.ownership[0, :, query].detach().float().sum().cpu().item()
                    ),
                }
            )
        if row is not None and bool(final.posterior.belief.valid[0, row]):
            diagnostic.update(
                {
                    "posterior_association": float(
                        final.posterior.match_probability[0, row]
                        .detach()
                        .float()
                        .sum()
                        .cpu()
                        .item()
                    ),
                    "posterior_existence": float(
                        final.posterior.belief.existence[0, row].detach().float().cpu().item()
                    ),
                    "posterior_visibility": float(
                        final.posterior.belief.visibility[0, row].detach().float().cpu().item()
                    ),
                    "posterior_ownership_mass": float(raw[:, row].sum()),
                    "posterior_map_present": bool(final.posterior.map_present[0, row]),
                }
            )
        diagnostics.append(diagnostic)
    return aligned_target, discovery, posterior, tuple(diagnostics)


def _terminal_lifecycle_history(
    batch: Any,
    diagnostics: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object]]:
    """Summarize post-forward measurement history for terminal identities."""

    final_frame = len(batch.observations) - 1
    if final_frame < 0:
        raise ValueError("stationary visual lifecycle history requires a nonempty clip")
    keys = tuple(str(diagnostic["identity_key"]) for diagnostic in diagnostics)
    if len(set(keys)) != len(keys):
        raise ValueError("stationary visual terminal identities must be unique")
    histories: dict[str, list[tuple[int, int, bool]]] = {key: [] for key in keys}
    for frame_index in range(len(batch.observations)):
        supervision = batch.build_supervision(frame_index)
        lifecycle = supervision.lifecycle_targets[0]
        target = supervision.set_targets[0]
        if lifecycle is None or target.temporal_identity_keys is None:
            raise ValueError("stationary visual history requires physical lifecycle identities")
        alive = set(lifecycle.alive_identity_keys)
        measurable = set(target.temporal_identity_keys)
        if not measurable.issubset(alive):
            raise ValueError("stationary visual measurable identities must be physically alive")
        global_index = batch.source_indices_by_frame[frame_index][0]
        for key in keys:
            if key in alive:
                histories[key].append((frame_index, global_index, key in measurable))

    result: dict[str, dict[str, object]] = {}
    for diagnostic in diagnostics:
        key = str(diagnostic["identity_key"])
        history = histories[key]
        if not history or history[-1][0] != final_frame:
            raise ValueError("stationary visual terminal identity is absent from final inventory")
        currently_measurable = bool(diagnostic["target_currently_measurable"])
        if history[-1][2] != currently_measurable:
            raise ValueError("stationary visual terminal measurability differs from history")
        prior_measurable = [row for row in history if row[0] < final_frame and row[2]]
        measurable_rows = [row for row in history if row[2]]
        terminal_unmeasurable_frames = 0
        for _frame_index, _global_index, measured in reversed(history):
            if measured:
                break
            terminal_unmeasurable_frames += 1
        retained = diagnostic["posterior_existence"] is not None
        result[key] = {
            "ever_measurable_before_final": bool(prior_measurable),
            "last_measurable_global_index": (measurable_rows[-1][1] if measurable_rows else None),
            "terminal_unmeasurable_frames": terminal_unmeasurable_frames,
            "seen_then_unmeasurable": bool(prior_measurable) and not currently_measurable,
            "candidate_posterior_identity_retained": retained,
            "candidate_posterior_map_present": bool(diagnostic["posterior_map_present"]),
            "candidate_posterior_existence": diagnostic["posterior_existence"],
        }
    return result


def _visual_history_score(artifact: dict[str, object]) -> tuple[int, int, int]:
    """Prefer the longest seen occlusion, retaining failure evidence if present."""

    lifecycle_targets = artifact.get("lifecycle_targets")
    if not isinstance(lifecycle_targets, list):
        raise ValueError("stationary visual artifact omitted lifecycle targets")
    seen = [
        target
        for target in lifecycle_targets
        if isinstance(target, dict) and target.get("seen_then_unmeasurable") is True
    ]
    if not seen:
        return (0, 0, 0)
    longest = max(int(target["terminal_unmeasurable_frames"]) for target in seen)
    retained = any(
        int(target["terminal_unmeasurable_frames"]) == longest
        and target.get("candidate_posterior_identity_retained") is True
        for target in seen
    )
    return (1, longest, int(retained))


def _attach_camera_ownership_masses(
    diagnostics: tuple[dict[str, object], ...],
    *,
    target: np.ndarray,
    discovery: np.ndarray,
    posterior: np.ndarray,
    image_spans: tuple[Any, ...],
    vision_start: int,
) -> tuple[dict[str, object], ...]:
    """Attach exact all-crop ownership mass for each physical camera source."""

    if target.shape != discovery.shape or target.shape != posterior.shape:
        raise ValueError("stationary visual aligned ownership shapes differ")
    if target.shape[1] != len(diagnostics) + 1:
        raise ValueError("stationary visual diagnostics differ from aligned identities")
    enriched = []
    for identity_index, diagnostic in enumerate(diagnostics):
        masses: dict[str, dict[str, float]] = {}
        for image_span in image_spans:
            start = vision_start + image_span.start
            stop = vision_start + image_span.stop
            if start < 0 or stop > target.shape[0] or stop <= start:
                raise ValueError("stationary visual camera span escaped aligned ownership")
            masses[image_span.image_key] = {
                "target": float(target[start:stop, identity_index].sum()),
                "discovery": float(discovery[start:stop, identity_index].sum()),
                "posterior": float(posterior[start:stop, identity_index].sum()),
            }
        enriched.append({**diagnostic, "camera_ownership_mass": masses})
    return tuple(enriched)


def _format_diagnostic_probability(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_visual_diagnostics(label: str, diagnostic: Mapping[str, object]) -> str:
    first = (
        f"{label}: D(ex={_format_diagnostic_probability(diagnostic['discovery_existence'])}, "
        f"loc={_format_diagnostic_probability(diagnostic['discovery_localization'])}, "
        f"meas={_format_diagnostic_probability(diagnostic['discovery_measurement'])}, "
        f"own={float(diagnostic['discovery_ownership_mass']):.3f}) "
        f"P(map={int(bool(diagnostic['posterior_map_present']))}, "
        f"assoc={float(diagnostic['posterior_association']):.3f}, "
        f"ex={_format_diagnostic_probability(diagnostic['posterior_existence'])}, "
        f"vis={_format_diagnostic_probability(diagnostic['posterior_visibility'])}, "
        f"own={float(diagnostic['posterior_ownership_mass']):.3f})"
    )
    masses = diagnostic.get("camera_ownership_mass")
    if not isinstance(masses, Mapping):
        return first
    short_names = {
        "observation.images.image": "ext",
        "observation.images.wrist_image": "wrist",
    }
    source_parts = []
    for camera, raw in masses.items():
        if not isinstance(raw, Mapping):
            raise ValueError("stationary visual camera ownership diagnostic is malformed")
        source_parts.append(
            f"{short_names.get(str(camera), str(camera))}:"
            f"D={float(raw['discovery']):.2f},P={float(raw['posterior']):.2f}"
        )
    return f"{first}\n{label} per-camera own " + "; ".join(source_parts)


def _task_labels(index: Any, global_index: int) -> list[dict[str, object]]:
    return [
        {
            "segment_index": segment.index,
            "task_key": segment.task_key,
            "instruction": segment.instruction,
        }
        for segment in index.segments
        if segment.start <= global_index <= segment.end
    ]


def _render_comparison(
    *,
    path: Path,
    assets: Any,
    batch: Any,
    fresh_trainer: Any,
    fresh_output: Any,
    candidate_trainer: Any,
    candidate_output: Any,
    split_name: str,
    optimizer_step: int,
    rank: int,
) -> dict[str, object]:
    from PIL import Image, ImageDraw

    frame_index = len(batch.observations) - 1
    global_index = batch.source_indices_by_frame[frame_index][0]
    source = assets.index.molmoact2_source_observation(global_index)
    supervision = batch.build_supervision(frame_index)
    target = supervision.set_targets[0]
    target_supervised = target.supervision_valid.detach().cpu().numpy()
    target_probability, fresh_discovery, fresh_posterior, fresh_diagnostics = (
        _aligned_probabilities(fresh_trainer, fresh_output, supervision)
    )
    candidate_target, candidate_discovery, candidate_posterior, candidate_diagnostics = (
        _aligned_probabilities(candidate_trainer, candidate_output, supervision)
    )
    if not np.array_equal(target_probability, candidate_target):
        raise ValueError("stationary visual target alignment changed between models")
    layout = batch.layouts[frame_index].vision_patch_layout
    if layout is None or len(layout.rows) != 1:
        raise ValueError("stationary visual requires one explicit processor layout row")
    vision_spans = tuple(
        span
        for span in batch.layouts[frame_index].spans
        if span.modality == MOLMO_VISION_PATCH_MODALITY
    )
    if len(vision_spans) != 1:
        raise ValueError("stationary visual requires one dense vision-token span")
    vision_start = vision_spans[0].start
    image_spans = layout.rows[0]
    fresh_diagnostics = _attach_camera_ownership_masses(
        fresh_diagnostics,
        target=target_probability,
        discovery=fresh_discovery,
        posterior=fresh_posterior,
        image_spans=image_spans,
        vision_start=vision_start,
    )
    candidate_diagnostics = _attach_camera_ownership_masses(
        candidate_diagnostics,
        target=target_probability,
        discovery=candidate_discovery,
        posterior=candidate_posterior,
        image_spans=image_spans,
        vision_start=vision_start,
    )
    candidate_history = _terminal_lifecycle_history(batch, candidate_diagnostics)
    rows = []
    for image_span in image_spans:
        if image_span.image_key not in CALVIN_HOST_IMAGE_KEYS:
            raise ValueError("stationary visual encountered an unknown camera key")
        start = vision_start + image_span.start
        stop = start + image_span.patches_per_crop
        if stop > vision_start + image_span.stop:
            raise ValueError("stationary visual global crop exceeds its processor span")
        source_rgb = np.asarray(source.images[image_span.image_key])
        target_local = target_probability[start:stop]
        supervised_local = target_supervised[start:stop]
        fresh_discovery_local = fresh_discovery[start:stop]
        candidate_discovery_local = candidate_discovery[start:stop]
        fresh_posterior_local = fresh_posterior[start:stop]
        candidate_posterior_local = candidate_posterior[start:stop]
        panels = (
            _panel(f"{image_span.image_key}: source", source_rgb),
            _panel(
                "loss-only physical target",
                _overlay(source_rgb, target_local, supervised=supervised_local),
                target_local,
            ),
            _panel(
                "fresh M2 discovery",
                _overlay(source_rgb, fresh_discovery_local),
                fresh_discovery_local,
            ),
            _panel(
                "candidate discovery",
                _overlay(source_rgb, candidate_discovery_local),
                candidate_discovery_local,
            ),
            _panel(
                "fresh persistent posterior",
                _overlay(source_rgb, fresh_posterior_local),
                fresh_posterior_local,
            ),
            _panel(
                "candidate persistent posterior",
                _overlay(source_rgb, candidate_posterior_local),
                candidate_posterior_local,
            ),
        )
        row = Image.new("RGB", (260 * len(panels), 300), "white")
        for panel_index, panel in enumerate(panels):
            row.paste(panel, (260 * panel_index, 0))
        rows.append(row)

    tasks = _task_labels(assets.index, global_index)
    task_text = (
        " | ".join(f"{item['task_key']}: {item['instruction']}" for item in tasks)
        or "no CALVIN language segment covers this source frame"
    )
    lifecycle = supervision.lifecycle_targets[0]
    if lifecycle is None:
        raise ValueError("stationary visual requires lifecycle targets")
    identity_keys = tuple(lifecycle.alive_identity_keys)
    if (
        tuple(item["identity_key"] for item in fresh_diagnostics) != identity_keys
        or tuple(item["identity_key"] for item in candidate_diagnostics) != identity_keys
    ):
        raise ValueError("stationary visual diagnostics differ from physical identity order")
    header_height = 125
    legend_height = 82 * len(identity_keys) + 20
    canvas = Image.new(
        "RGB",
        (1560, header_height + 300 * len(rows) + legend_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        (
            f"split={split_name} step={optimizer_step + 1} rank={rank} "
            f"prefix={batch.prefix_length} source_frame={global_index}"
        ),
        fill="black",
    )
    draw.text((8, 32), task_text[:240], fill="black")
    draw.text(
        (8, 56),
        "Task text is audit-only. Masks/IDs are post-forward loss-only and never model inputs.",
        fill="black",
    )
    draw.text(
        (8, 80),
        "Overlays use each camera's global processor crop; training/evaluation uses every token.",
        fill="black",
    )
    draw.text(
        (8, 104),
        "D=discovery; P=posterior; own=ownership mass; assoc=any-camera current-match mass; "
        "vis=P(exists and visible in >=1 camera).",
        fill="black",
    )
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, header_height + 300 * row_index))
    legend_y = header_height + 300 * len(rows) + 7
    for index, key in enumerate(identity_keys):
        color = tuple(int(value) for value in _color(index))
        draw.rectangle((8, legend_y, 24, legend_y + 12), fill=color)
        diagnostic = candidate_diagnostics[index]
        target_visibility = diagnostic["target_visibility"]
        target_visibility_text = (
            "?" if target_visibility is None else str(int(float(target_visibility) >= 0.5))
        )
        draw.text(
            (32, legend_y - 2),
            (
                f"physical object {index}: {key}; target measurable(any-camera)="
                f"{target_visibility_text}; current measurable="
                f"{int(bool(diagnostic['target_currently_measurable']))}"
            ),
            fill="black",
        )
        draw.multiline_text(
            (32, legend_y + 14),
            _format_visual_diagnostics("fresh", fresh_diagnostics[index]),
            fill="black",
            spacing=2,
        )
        draw.multiline_text(
            (32, legend_y + 44),
            _format_visual_diagnostics("candidate", candidate_diagnostics[index]),
            fill="black",
            spacing=2,
        )
        legend_y += 82
    canvas.save(path)
    return {
        "bytes": path.stat().st_size,
        "cameras": list(CALVIN_HOST_IMAGE_KEYS),
        "global_index": global_index,
        "optimizer_step": optimizer_step,
        "panels": [
            "source",
            "loss_only_target",
            "fresh_m2_discovery",
            "candidate_discovery",
            "fresh_m2_persistent_posterior",
            "candidate_persistent_posterior",
        ],
        "path": str(path.relative_to(path.parents[1])),
        "prefix_length": batch.prefix_length,
        "rank": rank,
        "sha256": sha256_file(path),
        "split": split_name,
        "tasks": tasks,
        "lifecycle_targets": [
            {
                "identity_key": str(diagnostic["identity_key"]),
                "currently_measurable": bool(diagnostic["target_currently_measurable"]),
                "conditional_detection_target": diagnostic["target_visibility"],
                "conditional_detection_supervised": bool(
                    diagnostic["target_visibility_supervised"]
                ),
                **candidate_history[str(diagnostic["identity_key"])],
            }
            for diagnostic in candidate_diagnostics
        ],
    }


def _build_plans(definition: Any, *, steps: int, seed: int) -> dict[str, Any]:
    source_ranges = {
        "validation": definition.source_coverage.split.validation_ranges,
        "heldout": definition.source_coverage.split.heldout_ranges,
    }
    return {
        split_name: build_distributed_stationary_temporal_clip_plan(
            source_ranges=source_ranges[split_name],
            prefix_lengths=_PREFIX_LENGTHS,
            train_length=definition.stage.clip.train_length,
            required_future_horizon=definition.maximum_horizon,
            optimizer_steps=steps,
            world_size=definition.stage.distributed.world_size,
            seed=seed + split_index,
        )
        for split_index, split_name in enumerate(_SPLIT_NAMES)
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    revision = _git_revision(_ROOT)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stationary candidate replay requires one CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    definition = load_stationary_calvin_stage_definition(
        args.stage_recipe,
        repository_root=_ROOT,
    )
    binding = validate_axis_calibrated_m2(
        report_path=args.m2_report,
        checkpoint_path=args.m2_checkpoint,
    )
    assets = load_stationary_calvin_stage_assets(
        definition,
        repository_root=_ROOT,
        split_root=args.split_root,
        feature_cache_root=args.feature_cache_root,
        feature_cache_manifest_sha256=binding["feature_cache_manifest_sha256"],
        physical_sidecar_root=args.physical_sidecar_root,
    )

    torch.manual_seed(definition.stage.clip.seed)
    fresh_trainer = build_stationary_temporal_trainer(
        definition,
        m2_checkpoint_path=args.m2_checkpoint,
        m2_checkpoint_sha256=binding["checkpoint_sha256"],
        device=device,
    )
    torch.manual_seed(definition.stage.clip.seed)
    candidate_trainer = build_stationary_temporal_trainer(
        definition,
        m2_checkpoint_path=args.m2_checkpoint,
        m2_checkpoint_sha256=binding["checkpoint_sha256"],
        device=device,
    )
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
    checkpoint_sha256 = sha256_file(candidate_checkpoint)
    provenance = load_stationary_temporal_checkpoint(
        candidate_trainer.core,
        candidate_trainer.objective,
        candidate_checkpoint,
        expected_sha256=checkpoint_sha256,
    )
    candidate_report_path = args.candidate_report.expanduser().resolve()
    candidate_report = _read_json(candidate_report_path, "Stage-B candidate report")
    _validate_candidate_report(
        candidate_report,
        checkpoint_sha256=checkpoint_sha256,
        definition=definition,
        provenance=provenance,
        binding=binding,
    )

    output_dir = args.output_dir.expanduser().resolve()
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True)
    plans = _build_plans(
        definition,
        steps=args.optimizer_steps_per_split,
        seed=args.seed,
    )
    measurements: list[dict[str, object]] = []
    metric_rows: dict[str, dict[str, list[dict[str, float]]]] = {
        split_name: {model_name: [] for model_name in _MODEL_NAMES} for split_name in _SPLIT_NAMES
    }
    visual_artifacts_by_coordinate: dict[tuple[str, int, int], dict[str, object]] = {}
    runtime_measurements: list[dict[str, object]] = []

    for split_name in _SPLIT_NAMES:
        plan = plans[split_name]
        for optimizer_step in range(plan.optimizer_steps):
            for rank in range(plan.world_size):
                clip = plan.clip(optimizer_step, rank)
                batch = assets.batch_builder.build(
                    (clip,),
                    device=device,
                    dtype=torch.bfloat16,
                )
                fresh_output, fresh_metrics, fresh_runtime = _run_model(
                    fresh_trainer, batch, device=device
                )
                candidate_output, candidate_metrics, candidate_runtime = _run_model(
                    candidate_trainer, batch, device=device
                )
                for model_name, metrics, runtime in (
                    ("fresh_m2", fresh_metrics, fresh_runtime),
                    ("candidate", candidate_metrics, candidate_runtime),
                ):
                    metric_rows[split_name][model_name].append(metrics)
                    measurements.append(
                        {
                            "clip": clip.to_dict(),
                            "metrics": metrics,
                            "model": model_name,
                            "optimizer_step": optimizer_step,
                            "rank": rank,
                            "split": split_name,
                        }
                    )
                    runtime_measurements.append(
                        {
                            "model": model_name,
                            "split": split_name,
                            "optimizer_step": optimizer_step,
                            "rank": rank,
                            **runtime,
                        }
                    )
                render_key = (split_name, clip.prefix_length, rank)
                final_index = clip.stop_global_index - 1
                labels = _task_labels(assets.index, final_index)
                task_key = labels[0]["task_key"] if labels else "task_independent"
                safe_task = _SAFE_NAME.sub("_", str(task_key).lower()).strip("_")
                filename = (
                    f"{split_name}_prefix{clip.prefix_length:03d}_rank{rank}_"
                    f"step{final_index:07d}_{safe_task}.png"
                )
                candidate_artifact = _render_comparison(
                    path=visuals_dir / filename,
                    assets=assets,
                    batch=batch,
                    fresh_trainer=fresh_trainer,
                    fresh_output=fresh_output,
                    candidate_trainer=candidate_trainer,
                    candidate_output=candidate_output,
                    split_name=split_name,
                    optimizer_step=optimizer_step,
                    rank=rank,
                )
                previous_artifact = visual_artifacts_by_coordinate.get(render_key)
                if previous_artifact is None or _visual_history_score(
                    candidate_artifact
                ) > _visual_history_score(previous_artifact):
                    if previous_artifact is not None:
                        (output_dir / str(previous_artifact["path"])).unlink()
                    visual_artifacts_by_coordinate[render_key] = candidate_artifact
                else:
                    (output_dir / str(candidate_artifact["path"])).unlink()
                del batch, fresh_output, candidate_output

    visual_artifacts = [
        visual_artifacts_by_coordinate[(split_name, prefix_length, rank)]
        for split_name in _SPLIT_NAMES
        for prefix_length in _PREFIX_LENGTHS
        for rank in range(definition.stage.distributed.world_size)
    ]

    split_reports: dict[str, object] = {}
    checks: dict[str, bool] = {}
    for split_name in _SPLIT_NAMES:
        summaries = {
            model_name: aggregate_replay_measurements(metric_rows[split_name][model_name])
            for model_name in _MODEL_NAMES
        }
        comparisons = compare_stationary_replay_summaries(
            fresh_m2=summaries["fresh_m2"],
            candidate=summaries["candidate"],
            absolute_tolerance=_ABSOLUTE_TOLERANCE,
        )
        checks.update({f"{split_name}_{name}": passed for name, passed in comparisons.items()})
        split_reports[split_name] = {
            "clip_count": len(metric_rows[split_name]["candidate"]),
            "models": summaries,
            "comparisons": comparisons,
        }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    status = STATIONARY_FIXED_REPLAY_PASS if not failed_checks else STATIONARY_FIXED_REPLAY_FAIL
    report = {
        "schema": STATIONARY_FIXED_REPLAY_SCHEMA,
        "status": status,
        "protocol": {
            "comparison": "same-frozen-clips-fresh-m2-vs-stage-b-candidate.v1",
            "observation_inputs": "task-independent-cached-native-token-bank",
            "target_use": "post-forward-loss-and-evaluation-only",
            "split_names": list(_SPLIT_NAMES),
            "prefix_lengths": list(_PREFIX_LENGTHS),
            "train_length": definition.stage.clip.train_length,
            "world_size": definition.stage.distributed.world_size,
            "optimizer_steps_per_split": args.optimizer_steps_per_split,
            "seed": args.seed,
        },
        "bindings": {
            "audit_code_revision": revision,
            "candidate_checkpoint_sha256": checkpoint_sha256,
            "candidate_code_revision": provenance.code_revision,
            "candidate_report_sha256": sha256_file(candidate_report_path),
            "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
            "feature_cache_manifest_sha256": provenance.feature_cache_manifest_sha256,
            "foundation_recipe_sha256": provenance.foundation_recipe_sha256,
            "m2_checkpoint_sha256": provenance.m2_checkpoint_sha256,
            "m2_report_sha256": sha256_file(args.m2_report.expanduser().resolve()),
            "physical_sidecar_manifest_sha256": provenance.physical_sidecar_manifest_sha256,
            "source_coverage_recipe_sha256": provenance.source_coverage_recipe_sha256,
            "stage_recipe_sha256": provenance.stage_recipe_sha256,
        },
        "plans": {
            split_name: {
                "plan_sha256": plans[split_name].plan_sha256,
                "source_ranges": [list(value) for value in plans[split_name].source_ranges],
            }
            for split_name in _SPLIT_NAMES
        },
        "thresholds": {
            "absolute_tolerance": _ABSOLUTE_TOLERANCE,
            "lower_is_better": [
                "loss_total",
                "loss_set",
                "loss_dynamics",
                "loss_dynamics_survival",
                "loss_dynamics_visibility",
                "loss_binding",
                "assignment_conflicts_per_clip",
            ],
            "higher_is_better": [
                "discovery_soft_iou",
                "posterior_soft_iou",
                "posterior_identity_coverage",
            ],
        },
        "splits": split_reports,
        "checks": checks,
        "failed_checks": failed_checks,
        "measurements": measurements,
        "long_training_authorized": False,
    }
    validate_stationary_fixed_replay(report)
    report_path = output_dir / "fixed_checkpoint_replay.json"
    _write_json_atomic(report_path, report)
    report_sha256 = sha256_file(report_path)

    lifecycle_calibration = build_stationary_lifecycle_calibration(
        report,
        fixed_replay_sha256=report_sha256,
    )
    _write_json_atomic(output_dir / "lifecycle_calibration.json", lifecycle_calibration)

    properties = torch.cuda.get_device_properties(device)
    runtime_probe = build_stationary_runtime_probe(
        report,
        fixed_replay_sha256=report_sha256,
        candidate_recurrent_state_serialized=provenance.recurrent_state_serialized,
        device_name=properties.name,
        total_memory_bytes=properties.total_memory,
        measurements=runtime_measurements,
    )
    _write_json_atomic(output_dir / "runtime_probe.json", runtime_probe)

    visual_manifest = {
        "schema": STATIONARY_VISUAL_ARTIFACTS_SCHEMA,
        "status": "PENDING_HUMAN_REVIEW",
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "fixed_checkpoint_replay_sha256": report_sha256,
        "artifact_count": len(visual_artifacts),
        "required_split_prefix_rank_coverage": [
            {"split": split_name, "prefix_length": prefix_length, "rank": rank}
            for split_name in _SPLIT_NAMES
            for prefix_length in _PREFIX_LENGTHS
            for rank in range(definition.stage.distributed.world_size)
        ],
        "artifacts": visual_artifacts,
        "artifacts_sha256": _sha256_bytes(_canonical_json(visual_artifacts).encode("ascii")),
        "mask_or_identity_visible_to_model": False,
        "task_text_visible_to_stationary_model": False,
    }
    _publish_visual_manifest(output_dir, visual_manifest)
    print(_canonical_json({"report": str(report_path), "status": status}))


if __name__ == "__main__":
    main()

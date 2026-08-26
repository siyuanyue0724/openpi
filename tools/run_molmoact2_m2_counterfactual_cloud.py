#!/usr/bin/env python3
"""Run paired exact-removal calibration of the MolmoAct2 current-frame set model.

This is a mechanism gate, not an authorization for temporal or action training.
GPU 0 receives natural replay plus factual/removed pairs.  GPU 1 receives the
same natural replay and compute budget but factual observations only.  Both use
the unchanged M2 current-frame model and ObjectSetCriterion.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
for _path in (_ROOT, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.data.calvin_counterfactual_plan import (  # noqa: E402
    CALVIN_COUNTERFACTUAL_PARTITIONS,
    CalvinCounterfactualPairPlan,
    load_calvin_counterfactual_pair_plan,
)
from picf_next.data.calvin_physical_supervision_schema import (  # noqa: E402
    CALVIN_CAMERA_SPECS,
    calvin_camera_name_from_host_image_key,
)
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.eval.calvin_same_renderer_removal import (  # noqa: E402
    CalvinSameRendererRemoval,
    CalvinSameRendererRemovalStore,
)
from picf_next.eval.object_set_assignment import (  # noqa: E402
    object_set_assignment_diagnostics,
)
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    CalvinStatefulLossTargetLayout,
    CalvinVisibleObjectTargetBuilder,
    molmoact2_host_observation_view,
)
from picf_next.models.evidence import ModalityTokenSpan  # noqa: E402
from picf_next.models.set_loss import ObjectSetCriterion, ObjectSetTarget  # noqa: E402
from picf_next.training.counterfactual_measurement import (  # noqa: E402
    COUNTERFACTUAL_MEASUREMENT_GATE,
    OCCAM_COMPLETE_SET_DECISION_RULE,
    CounterfactualMeasurementRecipe,
    deterministic_cycle,
    deterministic_cycle_exposure_counts,
    formal_counterfactual_measurement_acceptance,
    formal_counterfactual_measurement_occam_acceptance,
    load_counterfactual_measurement_recipe,
)
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from picf_next.training.stage_checkpoints import (  # noqa: E402
    load_picf_current_frame_checkpoint,
)
from tools import run_molmoact2_m2_cloud as m2  # noqa: E402

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(_ROOT / "configs/training/molmoact2_calvin_m2_counterfactual_smoke.json"),
    )
    parser.add_argument("--m2-run", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-split-root", required=True, type=Path)
    parser.add_argument("--removal-dir", required=True, type=Path)
    parser.add_argument("--pair-plan", type=Path)
    parser.add_argument("--pair-plan-sha256")
    parser.add_argument(
        "--source-sidecar-root",
        type=Path,
        default=Path(
            "/mnt/picf-next/artifacts/calvin_loss_sidecars/"
            "calvin_physical_supervision_all_source_training_v3_depth_consistent"
        ),
    )
    parser.add_argument(
        "--sidecar-artifact-root",
        type=Path,
        default=Path("/mnt/picf-next/artifacts/calvin_loss_sidecars"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/picf-next/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_id(value: str | None) -> str:
    resolved = value or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not _RUN_ID.fullmatch(resolved):
        raise ValueError(f"invalid counterfactual run id: {resolved!r}")
    return resolved


def _record_hash(seed: int, record: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(f"{seed}:{record['sample_key']}".encode("ascii")).digest()


def _select_natural_records(
    manifest: Mapping[str, Any],
    *,
    split: str,
    count: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("natural M2 cache has no record list")
    candidates = [dict(record) for record in records if record.get("split") == split]
    if len(candidates) < count:
        raise ValueError(f"natural M2 cache has too few {split} records")
    by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_group[m2._record_group_identity(record)].append(record)
    selected: list[dict[str, Any]] = []
    per_group = max(1, math.ceil(count / len(by_group)))
    for group in sorted(by_group):
        ordered = sorted(by_group[group], key=lambda row: _record_hash(seed, row))
        selected.extend(ordered[:per_group])
    selected = sorted(selected, key=lambda row: _record_hash(seed + 1, row))[:count]
    if len(selected) != count or len({row["sample_key"] for row in selected}) != count:
        raise RuntimeError("natural replay selection is not one-to-one")
    return tuple(selected)


def _load_natural_subset(
    cache_dir: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Any, Any, dict[str, Any]]]:
    import torch

    shard_specs = {
        str(shard["path"]): dict(shard)
        for shard in manifest.get("shards", [])
        if isinstance(shard, Mapping)
    }
    wanted: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        wanted[str(record["shard"])].append(record)
    loaded: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for shard_name, rows in sorted(wanted.items()):
        spec = shard_specs.get(shard_name)
        path = cache_dir / shard_name
        if spec is None or not path.is_file() or m2._sha256(path) != spec.get("sha256"):
            raise ValueError(f"natural replay shard changed: {shard_name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) != {"tokens", "valid"}:
            raise ValueError("natural replay feature shard contains target fields")
        for record in rows:
            row = int(record["row"])
            key = str(record["sample_key"])
            if key in loaded or not 0 <= row < payload["tokens"].shape[0]:
                raise ValueError("natural replay row mapping is invalid")
            loaded[key] = (
                payload["tokens"][row].contiguous(),
                payload["valid"][row].contiguous(),
                dict(record),
            )
    if len(loaded) != len(records):
        raise RuntimeError("natural replay subset omitted records")
    return loaded


def _source_hashes(frame: Any) -> tuple[tuple[str, str], ...]:
    spec_by_camera = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    values: list[tuple[str, str]] = []
    for camera in frame.cameras:
        spec = spec_by_camera[camera.camera_name]
        values.extend(
            (
                (str(spec["source_depth_field"]), camera.source_depth_sha256),
                (str(spec["source_rgb_field"]), camera.source_rgb_sha256),
            )
        )
    result = tuple(sorted(values))
    if len(result) != 4 or len({name for name, _digest in result}) != 4:
        raise RuntimeError("counterfactual branch source hashes are incomplete")
    return result


def _pair_identity(pair: CalvinSameRendererRemoval) -> str:
    if len(pair.target_identity_keys) != 1:
        raise ValueError("counterfactual measurement pairs must have one target identity")
    return pair.target_identity_keys[0]


def _pair_global_index(pair: CalvinSameRendererRemoval) -> int:
    contract = pair.artifact_contract.get("pair")
    value = contract.get("source_global_index") if isinstance(contract, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("counterfactual measurement pair has no valid source index")
    return value


def _sample_for_global_index(
    assets: Any,
    global_index: int,
    *,
    segment_index: int | None = None,
) -> Any:
    if segment_index is None:
        matches = tuple(
            segment
            for segment in assets.index.segments
            if segment.start <= global_index < segment.end
        )
        if not matches:
            raise ValueError(f"counterfactual frame {global_index} has no language transition")
        segment = sorted(matches, key=lambda value: value.index)[0]
    else:
        if not 0 <= segment_index < len(assets.index.segments):
            raise ValueError("counterfactual pair plan references an unknown segment")
        segment = assets.index.segments[segment_index]
        if not segment.start <= global_index < segment.end:
            raise ValueError("counterfactual pair plan frame lies outside its segment")
    return assets.index.stateful_transition_sample(
        segment.index,
        global_index,
        action_horizon=assets.dataset.action_horizon,
    )


def _materialize_pairs(
    *,
    assets: Any,
    store: CalvinSameRendererRemovalStore,
    source_sidecar: CalvinPhysicalSupervisionSidecar,
    recipe: CounterfactualMeasurementRecipe,
    pair_plan: CalvinCounterfactualPairPlan | None = None,
) -> tuple[
    dict[str, CalvinSameRendererRemoval],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, tuple[str, ...]],
]:
    pairs: dict[str, CalvinSameRendererRemoval] = {}
    samples: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    pair_ids_by_partition: dict[str, list[str]] = {
        partition: [] for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
    }
    request_by_key = (
        {request.key: request for request in pair_plan.requests} if pair_plan is not None else {}
    )
    if pair_plan is not None and set(store.keys) != set(request_by_key):
        raise ValueError("counterfactual removal bank differs from its pair plan")
    for global_index, identity_key in store.keys:
        pair_id = f"{global_index:07d}:{identity_key}"
        plan_request = request_by_key.get((global_index, identity_key))
        sample = _sample_for_global_index(
            assets,
            global_index,
            segment_index=(None if plan_request is None else plan_request.source_segment_index),
        )
        pair = store(
            sample.picf_evidence_frame,
            global_index=global_index,
            target_identity_keys=(identity_key,),
            physical_frame=source_sidecar.source_frame(global_index),
        )
        if (
            pair is None
            or pair.factual_physical_frame is None
            or pair.removed_physical_frame is None
        ):
            raise RuntimeError("counterfactual store omitted verified measurement frames")
        pairs[pair_id] = pair
        samples[pair_id] = sample
        if plan_request is None:
            for partition in CALVIN_COUNTERFACTUAL_PARTITIONS:
                pair_ids_by_partition[partition].append(pair_id)
        else:
            if (
                sample.host_sample.task_key != plan_request.task_key
                or sample.record.task != plan_request.instruction
            ):
                raise ValueError("counterfactual pair plan language metadata changed")
            pair_ids_by_partition[plan_request.partition].append(pair_id)
        for branch, physical in (
            ("factual", pair.factual_physical_frame),
            ("removed", pair.removed_physical_frame),
        ):
            key = f"counterfactual/{pair_id}/{branch}"
            records[key] = {
                "sample_key": key,
                "pair_id": pair_id,
                "branch": branch,
                "global_index": global_index,
                "target_identity_key": identity_key,
                "task_key": sample.host_sample.task_key,
                "instruction": sample.record.task,
                "pair_partition": ("smoke" if plan_request is None else plan_request.partition),
                "source_sensor_sha256": [list(item) for item in _source_hashes(physical)],
                "target_request_contract": "counterfactual_measurement",
            }
    identities = {_pair_identity(pair) for pair in pairs.values()}
    if (
        len(pairs) < recipe.acceptance.minimum_pairs
        or len(identities) < recipe.acceptance.minimum_distinct_identities
    ):
        raise ValueError("counterfactual store is too small or identity-poor for this recipe")
    if pair_plan is not None:
        for partition in CALVIN_COUNTERFACTUAL_PARTITIONS:
            partition_pairs = pair_ids_by_partition[partition]
            partition_identities = {_pair_identity(pairs[pair_id]) for pair_id in partition_pairs}
            if (
                len(partition_pairs) < recipe.acceptance.minimum_pairs
                or len(partition_identities) < recipe.acceptance.minimum_distinct_identities
            ):
                raise ValueError(
                    f"counterfactual {partition} partition is too small or identity-poor"
                )
    return (
        pairs,
        samples,
        records,
        {
            partition: tuple(sorted(pair_ids))
            for partition, pair_ids in pair_ids_by_partition.items()
        },
    )


def _extract_pair_features(
    *,
    run_dir: Path,
    foundation: Any,
    checkpoint_dir: Path,
    assets: Any,
    pairs: Mapping[str, CalvinSameRendererRemoval],
    samples: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    expected_layout: Sequence[Mapping[str, Any]],
    extraction_batch_size: int,
) -> dict[str, tuple[Any, Any, dict[str, Any]]]:
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation
    from picf_next.hosts.molmoact2_calvin_processor import CalvinMolmoAct2ProcessorBridge
    from picf_next.training.molmoact2_calvin import build_molmoact2_policy_config

    policy_config = build_molmoact2_policy_config(foundation, checkpoint_path=checkpoint_dir)
    device = torch.device("cuda:0")
    policy = MolmoAct2Policy(policy_config).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    stats = __import__(
        "picf_next.data.calvin_normalization",
        fromlist=["official_molmoact2_dataset_stats"],
    ).official_molmoact2_dataset_stats(assets.normalization_payload)
    processor = CalvinMolmoAct2ProcessorBridge.from_official_config(
        policy.config,
        dataset_stats=stats,
    )

    if extraction_batch_size <= 0:
        raise ValueError("counterfactual extraction batch size must be positive")
    keys = sorted(records)
    token_chunks = []
    valid_chunks = []
    for start in range(0, len(keys), extraction_batch_size):
        chunk_keys = keys[start : start + extraction_batch_size]
        evidence_rows = []
        views = []
        for key in chunk_keys:
            record = records[key]
            pair = pairs[str(record["pair_id"])]
            evidence = (
                pair.factual_evidence_frame
                if record["branch"] == "factual"
                else pair.evidence_frame
            )
            evidence_rows.append((evidence,))
            views.append(molmoact2_host_observation_view(samples[str(record["pair_id"])].record))
        inputs = m2._move_inputs(
            processor.build_observation_inputs(tuple(evidence_rows), tuple(views)),
            device,
        )
        with torch.inference_mode():
            prepared = prepare_molmoact2_lerobot_observation(policy, inputs)
        bank = prepared.vision_patch_bank
        layout = prepared.vision_patch_layout
        if bank is None or layout is None:
            raise RuntimeError("counterfactual extraction produced no native Molmo patch bank")
        actual_layout = [m2._layout_row_payload(row) for row in layout.rows]
        if any(row != list(expected_layout) for row in actual_layout):
            raise RuntimeError("counterfactual and natural Molmo patch layouts differ")
        token_chunks.append(m2._regular_cpu_copy(bank.tokens, dtype=torch.bfloat16))
        valid_chunks.append(m2._regular_cpu_copy(bank.valid))
        m2._emit_progress(
            "counterfactual_feature_extraction",
            rows_complete=min(start + len(chunk_keys), len(keys)),
            rows_total=len(keys),
        )
        del prepared, bank, layout, inputs
    tokens = torch.cat(token_chunks, dim=0)
    valid = torch.cat(valid_chunks, dim=0)
    cache = {
        key: (tokens[index], valid[index], dict(records[key])) for index, key in enumerate(keys)
    }
    feature_path = run_dir / "counterfactual_features.pt"
    m2._write_torch_atomic(feature_path, {"tokens": tokens, "valid": valid})
    manifest_records = [dict(records[key], row=index) for index, key in enumerate(keys)]
    m2._write_json_atomic(
        run_dir / "counterfactual_feature_manifest.json",
        {
            "schema": "picf-next.counterfactual-measurement-features.v1",
            "records": manifest_records,
            "records_sha256": m2._canonical_sha256(manifest_records),
            "feature_file": feature_path.name,
            "feature_file_sha256": m2._sha256(feature_path),
            "model_input_fields": ["tokens", "valid"],
            "loss_target_fields_in_feature_file": [],
            "layout": list(expected_layout),
            "extraction_batch_size": extraction_batch_size,
        },
    )
    del policy, processor
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def _measurement_target(
    *,
    builder: CalvinVisibleObjectTargetBuilder,
    physical: Any,
    record: Mapping[str, Any],
    token_valid: Any,
    target_dtype: Any,
    layout_payload: Sequence[Mapping[str, Any]],
) -> ObjectSetTarget:
    import torch

    layout = CalvinStatefulLossTargetLayout(
        token_valid=token_valid.detach().clone(),
        spans=(ModalityTokenSpan("molmo_vision_patch", 0, token_valid.shape[1]),),
        target_dtype=torch.float32,
        rollout_input_dtype=target_dtype,
        vision_patch_layout=m2._layout_from_payload(
            layout_payload,
            batch_size=1,
            token_count=token_valid.shape[1],
        ),
    )
    return builder.measurement_frames(
        (physical,),
        (tuple((str(name), str(digest)) for name, digest in record["source_sensor_sha256"]),),
        layout,
    )[0]


def _targets(
    *,
    builder: CalvinVisibleObjectTargetBuilder,
    records: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, CalvinSameRendererRemoval],
    token_valid: Any,
    target_dtype: Any,
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
) -> tuple[ObjectSetTarget, ...]:
    output = []
    for index, record in enumerate(records):
        one_valid = token_valid[index : index + 1]
        if record.get("target_request_contract") == "counterfactual_measurement":
            pair = pairs[str(record["pair_id"])]
            physical = (
                pair.factual_physical_frame
                if record["branch"] == "factual"
                else pair.removed_physical_frame
            )
            if physical is None:
                raise RuntimeError("counterfactual target lost its physical frame")
            output.append(
                _measurement_target(
                    builder=builder,
                    physical=physical,
                    record=record,
                    token_valid=one_valid,
                    target_dtype=target_dtype,
                    layout_payload=layout_payload,
                )
            )
        else:
            output.extend(
                m2._build_targets(
                    target_builder=builder,
                    records=(record,),
                    token_valid=one_valid,
                    target_dtype=target_dtype,
                    layout_payload=layout_payload,
                    token_count=token_count,
                )
            )
    return tuple(output)


def _load_model(
    foundation: Any,
    checkpoint: Path,
    device: Any,
    *,
    expected_sha256: str,
) -> Any:
    model = foundation.core_config.build_current_frame()
    load_picf_current_frame_checkpoint(
        model,
        checkpoint,
        expected_sha256=expected_sha256,
    )
    return model.to(device)


def _lr_multiplier(step: int, recipe: CounterfactualMeasurementRecipe) -> float:
    warmup = recipe.optimization.warmup_steps
    total = recipe.optimization.steps
    if step <= warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _pair_metrics(
    *,
    model: Any,
    device: Any,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    pairs: Mapping[str, CalvinSameRendererRemoval],
    builder: CalvinVisibleObjectTargetBuilder,
    criterion: ObjectSetCriterion,
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    rows: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for pair_id, pair in sorted(pairs.items()):
            branch_rows: dict[str, Any] = {}
            for branch in ("factual", "removed"):
                key = f"counterfactual/{pair_id}/{branch}"
                tokens, valid, records = m2._stack_batch(cache, (key,), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(m2._native_bank(tokens, valid))
                targets = _targets(
                    builder=builder,
                    records=records,
                    pairs=pairs,
                    token_valid=output.projection.token_valid,
                    target_dtype=output.discovery.ownership.dtype,
                    layout_payload=layout_payload,
                    token_count=token_count,
                )
                result = criterion(output.discovery, targets)
                target = targets[0]
                match = result.matches[0]
                existence = output.discovery.existence[0].float()
                matched = set(int(value) for value in match.prediction_indices.tolist())
                unmatched = [index for index in range(existence.numel()) if index not in matched]
                payload: dict[str, Any] = {
                    "loss_total": float(result.total.float().item()),
                    "target_count": target.num_objects,
                    "active_count": int((existence > 0.5).sum().item()),
                    "maximum_unmatched_existence": (
                        max(float(existence[index].item()) for index in unmatched)
                        if unmatched
                        else 0.0
                    ),
                    **object_set_assignment_diagnostics(
                        output.discovery,
                        target,
                        match,
                        batch_index=0,
                    ),
                }
                if branch == "factual":
                    identities = target.temporal_identity_keys or ()
                    target_index = identities.index(_pair_identity(pair))
                    locations = torch.nonzero(
                        match.target_indices == target_index,
                        as_tuple=False,
                    ).flatten()
                    if locations.numel() != 1:
                        raise RuntimeError("factual target has no unique Hungarian match")
                    query = int(match.prediction_indices[int(locations.item())].item())
                    supervised = target.supervision_valid
                    prediction = output.discovery.ownership[0, supervised, query].float()
                    truth = target.ownership[supervised, target_index].float()
                    dice = (2.0 * (prediction * truth).sum()) / (
                        prediction.sum() + truth.sum() + 1e-8
                    )
                    payload.update(
                        {
                            "target_query": query,
                            "target_existence": float(existence[query].item()),
                            "target_soft_dice": float(dice.item()),
                        }
                    )
                branch_rows[branch] = payload
            rows[pair_id] = {
                "global_index": _pair_global_index(pair),
                "target_identity_key": _pair_identity(pair),
                **branch_rows,
            }
    return {
        "pairs": rows,
        "mean_removed_loss": float(
            np.mean([row["removed"]["loss_total"] for row in rows.values()])
        ),
        "mean_removed_maximum_unmatched_existence": float(
            np.mean([row["removed"]["maximum_unmatched_existence"] for row in rows.values()])
        ),
    }


def _train(
    *,
    run_dir: Path,
    recipe: CounterfactualMeasurementRecipe,
    foundation: Any,
    initial_checkpoint: Path,
    initial_checkpoint_sha256: str,
    pair_cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    natural_cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    pairs: Mapping[str, CalvinSameRendererRemoval],
    train_pair_ids: Sequence[str],
    builder: CalvinVisibleObjectTargetBuilder,
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    actual_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    actual = _load_model(
        foundation,
        initial_checkpoint,
        actual_device,
        expected_sha256=initial_checkpoint_sha256,
    )
    control = _load_model(
        foundation,
        initial_checkpoint,
        control_device,
        expected_sha256=initial_checkpoint_sha256,
    )
    actual_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(actual_device)
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(control_device)
    actual_optimizer = torch.optim.AdamW(
        actual.parameters(),
        lr=recipe.optimization.learning_rate,
        weight_decay=recipe.optimization.weight_decay,
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=recipe.optimization.learning_rate,
        weight_decay=recipe.optimization.weight_decay,
    )
    merged = {**natural_cache, **pair_cache}
    natural_keys = tuple(sorted(natural_cache))
    pair_ids = tuple(sorted(train_pair_ids))
    if not pair_ids or any(pair_id not in pairs for pair_id in pair_ids):
        raise ValueError("counterfactual training pair partition is empty or invalid")
    metrics = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(actual_device)
    torch.cuda.reset_peak_memory_stats(control_device)
    for step in range(1, recipe.optimization.steps + 1):
        selected_pairs = deterministic_cycle(
            pair_ids,
            count=recipe.optimization.pair_count_per_step,
            seed=recipe.optimization.seed + 101,
            step=step,
        )
        selected_natural = deterministic_cycle(
            natural_keys,
            count=recipe.optimization.natural_count_per_step,
            seed=recipe.optimization.seed,
            step=step,
        )
        actual_keys = (
            tuple(
                key
                for pair_id in selected_pairs
                for key in (
                    f"counterfactual/{pair_id}/factual",
                    f"counterfactual/{pair_id}/removed",
                )
            )
            + selected_natural
        )
        control_keys = (
            tuple(
                f"counterfactual/{pair_id}/factual"
                for pair_id in selected_pairs
                for _repeat in range(2)
            )
            + selected_natural
        )
        actual.train()
        control.train()
        actual_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        actual_tokens, actual_valid, actual_records = m2._stack_batch(
            merged,
            actual_keys,
            device=actual_device,
        )
        control_tokens, control_valid, control_records = m2._stack_batch(
            merged,
            control_keys,
            device=control_device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual_output = actual(m2._native_bank(actual_tokens, actual_valid))
            control_output = control(m2._native_bank(control_tokens, control_valid))
        actual_targets = _targets(
            builder=builder,
            records=actual_records,
            pairs=pairs,
            token_valid=actual_output.projection.token_valid,
            target_dtype=actual_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=token_count,
        )
        control_targets = _targets(
            builder=builder,
            records=control_records,
            pairs=pairs,
            token_valid=control_output.projection.token_valid,
            target_dtype=control_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=token_count,
        )
        actual_result = actual_criterion(actual_output.discovery, actual_targets)
        control_result = control_criterion(control_output.discovery, control_targets)
        actual_result.total.backward()
        control_result.total.backward()
        actual_norm = torch.nn.utils.clip_grad_norm_(
            actual.parameters(), recipe.optimization.gradient_clip_norm
        )
        control_norm = torch.nn.utils.clip_grad_norm_(
            control.parameters(), recipe.optimization.gradient_clip_norm
        )
        if not torch.isfinite(actual_norm) or not torch.isfinite(control_norm):
            raise FloatingPointError("counterfactual calibration gradient became non-finite")
        multiplier = _lr_multiplier(step, recipe)
        for optimizer in (actual_optimizer, control_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = recipe.optimization.learning_rate * multiplier
            optimizer.step()
        row = {
            "step": step,
            "learning_rate": recipe.optimization.learning_rate * multiplier,
            "actual_loss": float(actual_result.total.detach().float().item()),
            "factual_only_control_loss": float(control_result.total.detach().float().item()),
            "actual_gradient_norm": float(actual_norm.detach().float().item()),
            "factual_only_control_gradient_norm": float(control_norm.detach().float().item()),
            "actual_keys": list(actual_keys),
            "control_keys": list(control_keys),
        }
        metrics.append(row)
        m2._emit_progress(
            "counterfactual_optimization",
            step=step,
            total_steps=recipe.optimization.steps,
            actual_loss=row["actual_loss"],
            factual_only_control_loss=row["factual_only_control_loss"],
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    actual_state = m2._state_dict_cpu(actual)
    control_state = m2._state_dict_cpu(control)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    actual_path = checkpoint_dir / "current_frame_counterfactual.pt"
    control_path = checkpoint_dir / "current_frame_factual_only_control.pt"
    m2._write_torch_atomic(actual_path, {"model": actual_state})
    m2._write_torch_atomic(control_path, {"model": control_state})
    return (
        actual,
        control,
        {
            "steps": recipe.optimization.steps,
            "elapsed_s": elapsed,
            "seconds_per_joint_actual_and_control_step": elapsed / recipe.optimization.steps,
            "metrics": metrics,
            "actual_state_sha256": m2._state_dict_sha256(actual_state),
            "control_state_sha256": m2._state_dict_sha256(control_state),
            "checkpoints": {
                actual_path.name: m2._sha256(actual_path),
                control_path.name: m2._sha256(control_path),
            },
            "peak_allocated_bytes": {
                "cuda:0": int(torch.cuda.max_memory_allocated(actual_device)),
                "cuda:1": int(torch.cuda.max_memory_allocated(control_device)),
            },
        },
    )


def _natural_metrics(
    *,
    model: Any,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    builder: CalvinVisibleObjectTargetBuilder,
    criterion: ObjectSetCriterion,
    layout_payload: Sequence[Mapping[str, Any]],
    m2_recipe: Any,
    device: Any,
) -> dict[str, Any]:
    return m2._evaluate(
        model=model,
        cache=cache,
        keys=tuple(sorted(cache)),
        target_builder=builder,
        criterion=criterion,
        layout_payload=layout_payload,
        recipe=m2_recipe,
        device=device,
    )


def _pair_visuals(
    *,
    run_dir: Path,
    models: Mapping[str, tuple[Any, Any]],
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    pairs: Mapping[str, CalvinSameRendererRemoval],
    builder: CalvinVisibleObjectTargetBuilder,
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
    criterion_config: Any,
) -> dict[str, Any]:
    import torch
    from PIL import Image, ImageDraw

    visual_dir = run_dir / "visuals"
    visual_dir.mkdir()
    artifacts = []
    source_key = {
        "observation.images.rgb_static": "static",
        "observation.images.rgb_gripper": "gripper",
    }
    for pair_id, pair in sorted(pairs.items()):
        row_images = []
        for branch in ("factual", "removed"):
            key = f"counterfactual/{pair_id}/{branch}"
            evidence = pair.factual_evidence_frame if branch == "factual" else pair.evidence_frame
            rgb_by_camera = {
                source_key[item.key]: np.asarray(item.value)
                for item in evidence.sensor_observations
                if item.key in source_key
            }
            prediction_by_model: dict[str, Any] = {}
            target: ObjectSetTarget | None = None
            for name, (model, device) in models.items():
                model.eval()
                tokens, valid, records = m2._stack_batch(cache, (key,), device=device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(m2._native_bank(tokens, valid))
                current_targets = _targets(
                    builder=builder,
                    records=records,
                    pairs=pairs,
                    token_valid=output.projection.token_valid,
                    target_dtype=output.discovery.ownership.dtype,
                    layout_payload=layout_payload,
                    token_count=token_count,
                )
                target = current_targets[0]
                active = torch.nonzero(
                    output.discovery.existence[0].float() > 0.5,
                    as_tuple=False,
                ).flatten()
                probability = torch.cat(
                    (
                        output.discovery.ownership[0].index_select(1, active).float(),
                        output.discovery.context_ownership[0, :, None].float(),
                    ),
                    dim=1,
                )
                prediction_by_model[name] = (
                    probability.cpu().numpy(),
                    int(active.numel()),
                )
            if target is None:
                raise RuntimeError("counterfactual visual target was not built")
            target_probability = target.ownership.float().cpu().numpy()
            target_known = target.supervision_valid.cpu().numpy()
            for span in layout_payload:
                camera = calvin_camera_name_from_host_image_key(str(span["image_key"]))
                source = rgb_by_camera[camera]
                start, stop = int(span["start"]), int(span["stop"])
                target_labels = target_probability[start:stop].argmax(-1).reshape(27, 27)
                target_labels[target_labels == target.num_objects] = 255
                target_labels[~target_known[start:stop].reshape(27, 27)] = 254
                panels = [
                    ("source", source),
                    (
                        "target",
                        m2._overlay(
                            source,
                            target_labels,
                            object_count=target.num_objects,
                            unknown_label=254,
                        ),
                    ),
                ]
                for name in ("prior", "actual", "factual_only_control"):
                    probability, count = prediction_by_model[name]
                    labels = probability[start:stop].argmax(-1).reshape(27, 27)
                    labels[labels == count] = 255
                    panels.append((name, m2._overlay(source, labels, object_count=count)))
                width = 270 * len(panels)
                row = Image.new("RGB", (width, 300), "white")
                draw = ImageDraw.Draw(row)
                for index, (title, array) in enumerate(panels):
                    panel = Image.fromarray(array).resize((270, 270), Image.Resampling.NEAREST)
                    row.paste(panel, (index * 270, 30))
                    draw.text((index * 270 + 6, 7), f"{branch}/{camera}: {title}", fill="black")
                row_images.append(row)
        header = 60
        canvas = Image.new(
            "RGB",
            (max(image.width for image in row_images), header + 300 * len(row_images)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            f"frame={_pair_global_index(pair)} target={_pair_identity(pair)}",
            fill="black",
        )
        draw.text(
            (8, 30),
            "factual/removed use exact same renderer; colors are active unordered queries",
            fill="black",
        )
        for index, image in enumerate(row_images):
            canvas.paste(image, (0, header + index * 300))
        safe_identity = _pair_identity(pair).replace("/", "-")
        path = visual_dir / f"frame{_pair_global_index(pair):07d}_{safe_identity}.png"
        canvas.save(path)
        artifacts.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": m2._sha256(path),
                "global_index": _pair_global_index(pair),
                "target_identity_key": _pair_identity(pair),
            }
        )
    return {
        "schema": "picf-next.counterfactual-measurement-visuals.v1",
        "artifacts": artifacts,
        "artifacts_sha256": m2._canonical_sha256(artifacts),
        "criterion_config": str(criterion_config),
    }


def _acceptance(
    *,
    recipe: CounterfactualMeasurementRecipe,
    prior_pair: Mapping[str, Any],
    actual_pair: Mapping[str, Any],
    control_pair: Mapping[str, Any],
    prior_natural: Mapping[str, Any],
    actual_natural: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = recipe.acceptance
    actual_rows = list(actual_pair["pairs"].values())
    prior_removed = float(prior_pair["mean_removed_loss"])
    actual_removed = float(actual_pair["mean_removed_loss"])
    removed_improvement = (
        (prior_removed - actual_removed) / prior_removed if prior_removed > 0.0 else float("-inf")
    )
    prior_natural_loss = float(prior_natural["losses"]["loss_total"])
    actual_natural_loss = float(actual_natural["losses"]["loss_total"])
    natural_regression = (
        (actual_natural_loss - prior_natural_loss) / prior_natural_loss
        if prior_natural_loss > 0.0
        else float("inf")
    )
    control_margin = float(control_pair["mean_removed_maximum_unmatched_existence"]) - float(
        actual_pair["mean_removed_maximum_unmatched_existence"]
    )
    checks = {
        "factual_targets_preserved": all(
            row["factual"]["target_existence"] >= threshold.minimum_factual_target_existence
            and row["factual"]["target_soft_dice"] >= threshold.minimum_factual_target_soft_dice
            for row in actual_rows
        ),
        "removed_targets_rejected": all(
            row["removed"]["maximum_unmatched_existence"]
            <= threshold.maximum_removed_unmatched_existence
            for row in actual_rows
        ),
        "removed_set_cardinality_exact": all(
            row["removed"]["active_count"] == row["removed"]["target_count"] for row in actual_rows
        ),
        "removed_loss_improves": removed_improvement
        >= threshold.minimum_removed_loss_improvement_fraction,
        "natural_loss_preserved": natural_regression
        <= threshold.maximum_natural_replay_loss_regression_fraction,
        "natural_count_preserved": (
            float(actual_natural["count_mae"]) - float(prior_natural["count_mae"])
            <= threshold.maximum_natural_count_mae_regression
        ),
        "counterfactual_signal_beats_factual_only_control": control_margin
        >= threshold.minimum_control_removed_existence_margin,
    }
    return {
        "status": "PASS_MECHANISM_SMOKE_ONLY" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "removed_loss_improvement_fraction": removed_improvement,
        "natural_replay_loss_regression_fraction": natural_regression,
        "factual_only_control_removed_existence_margin": control_margin,
        "later_gates_authorized": [],
    }


def main() -> None:
    import torch

    from picf_next.training.molmoact2_calvin import load_calvin_training_assets
    from tools.train_molmoact2_calvin_picf import _validate_training_checkpoint

    args = _parse_args()
    recipe = load_counterfactual_measurement_recipe(args.config)
    m2_recipe = load_molmoact2_m2_recipe(recipe.foundation_m2_path(_ROOT))
    foundation = m2_recipe.load_foundation(_ROOT)
    static = {
        "schema": recipe.schema,
        "gate": recipe.gate,
        "recipe_sha256": recipe.recipe_sha256,
        "foundation_m2_recipe_sha256": m2_recipe.recipe_sha256,
        "objective": "natural replay + factual/removed ObjectSetCriterion",
        "new_model_heads": 0,
        "lifecycle_supervision_from_removal": False,
        "later_gates_authorized_before_decision": [],
    }
    if recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE:
        static["decision_rule"] = recipe.decision_rule
    if args.dry_run:
        print(json.dumps(static, indent=2, sort_keys=True))
        return

    resources = m2._validate_devices()
    revision = m2._clean_git_revision()
    run_root = args.run_root.expanduser().resolve()
    if not m2._is_under_mnt(run_root):
        raise RuntimeError("counterfactual run root must persist under /mnt")
    run_dir = run_root / "molmoact2" / COUNTERFACTUAL_MEASUREMENT_GATE / _run_id(args.run_id)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    initial_checkpoint = args.initial_checkpoint.expanduser().resolve()
    m2_run = args.m2_run.expanduser().resolve()
    natural_manifest_path = m2_run / "feature_cache/manifest.json"
    if not initial_checkpoint.is_file() or not natural_manifest_path.is_file():
        raise FileNotFoundError("counterfactual calibration prerequisites are absent")
    initial_checkpoint_sha256 = m2._sha256(initial_checkpoint)
    checkpoint_probe = foundation.core_config.build_current_frame()
    initial_checkpoint_load = load_picf_current_frame_checkpoint(
        checkpoint_probe,
        initial_checkpoint,
        expected_sha256=initial_checkpoint_sha256,
    )
    del checkpoint_probe
    if (
        recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE
        and initial_checkpoint_load["fresh_keys"]
    ):
        raise ValueError("complete-set calibration requires a fully initialized checkpoint")
    m2.materialize_persistent_sidecars(args.sidecar_artifact_root)
    m2_launch = json.loads((m2_run / "launch_manifest.json").read_text())
    prior_m0 = json.loads(
        Path(m2_launch["prior_m1"]["m0_run_dir"]).joinpath("m0_raw_report.json").read_text()
    )
    _validate_training_checkpoint(
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        m0_report=prior_m0,
        checkpoint_id=foundation.host.checkpoint_id,
        checkpoint_revision=foundation.host.checkpoint_revision,
    )
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
    )
    source_sidecar_root = args.source_sidecar_root.expanduser().resolve()
    source_sidecar = CalvinPhysicalSupervisionSidecar(
        source_sidecar_root,
        assets.index,
        verify_hashes=True,
        cache_shards=24,
    )
    expected_source_sidecar_hash = m2_launch.get("source_sidecar", {}).get("manifest_sha256")
    actual_source_sidecar_hash = m2._sha256(source_sidecar_root / "manifest.json")
    if expected_source_sidecar_hash != actual_source_sidecar_hash:
        raise ValueError("counterfactual source sidecar differs from the natural M2 cache")
    store = CalvinSameRendererRemovalStore(
        args.removal_dir.expanduser().resolve(),
        dataset_id=assets.index.dataset_id,
        dataset_revision=assets.index.dataset_revision,
    )
    if (args.pair_plan is None) != (args.pair_plan_sha256 is None):
        raise ValueError("counterfactual pair plan path and hash must be supplied together")
    pair_plan = (
        None
        if args.pair_plan is None
        else load_calvin_counterfactual_pair_plan(
            args.pair_plan.expanduser().resolve(),
            expected_sha256=args.pair_plan_sha256,
        )
    )
    if pair_plan is None:
        if store.pair_plan_sha256 is not None:
            raise ValueError("planned counterfactual bank requires its exact pair plan")
    else:
        if (
            pair_plan.dataset_id != assets.index.dataset_id
            or pair_plan.dataset_revision != assets.index.dataset_revision
            or pair_plan.split_name != assets.index.split_root.name
            or pair_plan.foundation_m2_recipe_sha256 != recipe.foundation_m2_recipe_sha256
            or pair_plan.source_sidecar_manifest_sha256 != actual_source_sidecar_hash
            or store.pair_plan_sha256 != pair_plan.file_sha256
            or store.source_sidecar_manifest_sha256 != actual_source_sidecar_hash
        ):
            raise ValueError("counterfactual pair plan provenance differs from this run")
    pairs, samples, pair_records, pair_ids_by_partition = _materialize_pairs(
        assets=assets,
        store=store,
        source_sidecar=source_sidecar,
        recipe=recipe,
        pair_plan=pair_plan,
    )
    if pair_plan is not None:
        if (
            recipe.optimization.natural_count_per_step
            <= 2 * recipe.optimization.pair_count_per_step
        ):
            raise ValueError("formal counterfactual calibration requires dominant natural replay")
        planned_pair_exposures = recipe.optimization.steps * recipe.optimization.pair_count_per_step
        if planned_pair_exposures < 4 * len(pair_ids_by_partition["train"]):
            raise ValueError(
                "formal counterfactual calibration gives each train pair fewer than four "
                "balanced exposures"
            )
    natural_manifest = json.loads(natural_manifest_path.read_text())
    if natural_manifest.get("loss_target_fields_in_feature_shards") != []:
        raise ValueError("natural M2 cache contains loss-target leakage")
    layout_payload = natural_manifest["processor_layout"]
    train_records = _select_natural_records(
        natural_manifest,
        split="train",
        count=recipe.optimization.natural_replay_pool_size,
        seed=recipe.optimization.seed,
    )
    balanced_exposure = None
    if recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE:
        pair_exposure = deterministic_cycle_exposure_counts(
            tuple(sorted(pair_ids_by_partition["train"])),
            count=recipe.optimization.pair_count_per_step,
            seed=recipe.optimization.seed + 101,
            steps=recipe.optimization.steps,
        )
        natural_exposure = deterministic_cycle_exposure_counts(
            tuple(sorted(str(record["sample_key"]) for record in train_records)),
            count=recipe.optimization.natural_count_per_step,
            seed=recipe.optimization.seed,
            steps=recipe.optimization.steps,
        )
        pair_counts = set(pair_exposure.values())
        natural_counts = set(natural_exposure.values())
        if len(pair_counts) != 1 or len(natural_counts) != 1 or pair_counts != natural_counts:
            raise ValueError(
                "complete-set calibration requires equal exact exposure for every pair "
                "and natural replay row"
            )
        balanced_exposure = {
            "pair_item_count": len(pair_exposure),
            "natural_item_count": len(natural_exposure),
            "exposures_per_item": next(iter(pair_counts)),
            "pair_exposure_sha256": m2._canonical_sha256(pair_exposure),
            "natural_exposure_sha256": m2._canonical_sha256(natural_exposure),
        }
    evaluation_records = _select_natural_records(
        natural_manifest,
        split="heldout",
        count=min(32, recipe.optimization.natural_replay_pool_size),
        seed=recipe.optimization.seed + 10_000,
    )
    natural_cache_dir = natural_manifest_path.parent
    natural_train_cache = _load_natural_subset(
        natural_cache_dir,
        natural_manifest,
        train_records,
    )
    natural_evaluation_cache = _load_natural_subset(
        natural_cache_dir,
        natural_manifest,
        evaluation_records,
    )
    run_dir.mkdir(parents=True)
    m2._write_json_atomic(
        run_dir / "launch_manifest.json",
        {
            **static,
            "run_dir": str(run_dir),
            "code_revision": revision,
            "config": str(args.config.resolve()),
            "config_file_sha256": m2._sha256(args.config.resolve()),
            "initial_checkpoint": str(initial_checkpoint),
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "initial_checkpoint_load": initial_checkpoint_load,
            "natural_feature_manifest": str(natural_manifest_path),
            "natural_feature_manifest_sha256": m2._sha256(natural_manifest_path),
            "removal_store": str(store.root),
            "removal_store_summary_sha256": store.summary_sha256,
            "pair_plan": None if pair_plan is None else str(pair_plan.path),
            "pair_plan_sha256": (None if pair_plan is None else pair_plan.file_sha256),
            "source_sidecar_root": str(source_sidecar_root),
            "source_sidecar_manifest_sha256": actual_source_sidecar_hash,
            "pair_ids": sorted(pairs),
            "pair_ids_by_partition": {
                partition: list(pair_ids_by_partition[partition])
                for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
            },
            "balanced_exposure": balanced_exposure,
            "worktree_clean": True,
        },
    )
    m2._write_json_atomic(
        run_dir / "environment.json",
        {
            "schema": "picf-next.counterfactual-measurement-environment.v1",
            "resources": resources,
            "python": sys.version,
            "torch": torch.__version__,
        },
    )
    pair_cache = _extract_pair_features(
        run_dir=run_dir,
        foundation=foundation,
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        assets=assets,
        pairs=pairs,
        samples=samples,
        records=pair_records,
        expected_layout=layout_payload,
        extraction_batch_size=m2_recipe.cache.extraction_batch_size,
    )
    builder = CalvinVisibleObjectTargetBuilder(source_sidecar)
    pair_groups = (
        {"smoke": pairs}
        if pair_plan is None
        else {
            partition: {pair_id: pairs[pair_id] for pair_id in pair_ids_by_partition[partition]}
            for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
        }
    )
    prior = _load_model(
        foundation,
        initial_checkpoint,
        torch.device("cuda:0"),
        expected_sha256=initial_checkpoint_sha256,
    )
    prior_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to("cuda:0")
    prior_pairs = {
        partition: _pair_metrics(
            model=prior,
            device=torch.device("cuda:0"),
            cache=pair_cache,
            pairs=partition_pairs,
            builder=builder,
            criterion=prior_criterion,
            layout_payload=layout_payload,
            token_count=m2_recipe.cache.token_count,
        )
        for partition, partition_pairs in pair_groups.items()
    }
    prior_natural = _natural_metrics(
        model=prior,
        cache=natural_evaluation_cache,
        builder=builder,
        criterion=prior_criterion,
        layout_payload=layout_payload,
        m2_recipe=m2_recipe,
        device=torch.device("cuda:0"),
    )
    del prior, prior_criterion
    torch.cuda.empty_cache()
    actual, control, training = _train(
        run_dir=run_dir,
        recipe=recipe,
        foundation=foundation,
        initial_checkpoint=initial_checkpoint,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        pair_cache=pair_cache,
        natural_cache=natural_train_cache,
        pairs=pairs,
        train_pair_ids=pair_ids_by_partition["train"],
        builder=builder,
        layout_payload=layout_payload,
        token_count=m2_recipe.cache.token_count,
    )
    actual_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to("cuda:0")
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to("cuda:1")
    actual_pairs = {
        partition: _pair_metrics(
            model=actual,
            device=torch.device("cuda:0"),
            cache=pair_cache,
            pairs=partition_pairs,
            builder=builder,
            criterion=actual_criterion,
            layout_payload=layout_payload,
            token_count=m2_recipe.cache.token_count,
        )
        for partition, partition_pairs in pair_groups.items()
    }
    control_pairs = {
        partition: _pair_metrics(
            model=control,
            device=torch.device("cuda:1"),
            cache=pair_cache,
            pairs=partition_pairs,
            builder=builder,
            criterion=control_criterion,
            layout_payload=layout_payload,
            token_count=m2_recipe.cache.token_count,
        )
        for partition, partition_pairs in pair_groups.items()
    }
    actual_natural = _natural_metrics(
        model=actual,
        cache=natural_evaluation_cache,
        builder=builder,
        criterion=actual_criterion,
        layout_payload=layout_payload,
        m2_recipe=m2_recipe,
        device=torch.device("cuda:0"),
    )
    control_natural = None
    if recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE:
        control_natural = _natural_metrics(
            model=control,
            cache=natural_evaluation_cache,
            builder=builder,
            criterion=control_criterion,
            layout_payload=layout_payload,
            m2_recipe=m2_recipe,
            device=torch.device("cuda:1"),
        )
    if pair_plan is None:
        report = {
            "schema": "picf-next.counterfactual-measurement-evaluation.v1",
            "prior": {
                "pairs": prior_pairs["smoke"],
                "natural_heldout_subset": prior_natural,
            },
            "actual": {
                "pairs": actual_pairs["smoke"],
                "natural_heldout_subset": actual_natural,
            },
            "factual_only_control": {"pairs": control_pairs["smoke"]},
        }
    elif recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE:
        if control_natural is None:
            raise RuntimeError("Occam decision requires control natural evaluation")
        report = {
            "schema": "picf-next.counterfactual-measurement-evaluation.v3",
            "prior": {
                "pair_partitions": prior_pairs,
                "natural_heldout_subset": prior_natural,
            },
            "actual": {
                "pair_partitions": actual_pairs,
                "natural_heldout_subset": actual_natural,
            },
            "factual_only_control": {
                "pair_partitions": control_pairs,
                "natural_heldout_subset": control_natural,
            },
        }
    else:
        report = {
            "schema": "picf-next.counterfactual-measurement-evaluation.v2",
            "prior": {
                "pair_partitions": prior_pairs,
                "natural_heldout_subset": prior_natural,
            },
            "actual": {
                "pair_partitions": actual_pairs,
                "natural_heldout_subset": actual_natural,
            },
            "factual_only_control": {"pair_partitions": control_pairs},
        }
    m2._write_json_atomic(run_dir / "training_report.json", training)
    m2._write_json_atomic(run_dir / "evaluation_report.json", report)
    prior_visual = _load_model(
        foundation,
        initial_checkpoint,
        torch.device("cuda:0"),
        expected_sha256=initial_checkpoint_sha256,
    )
    visuals = _pair_visuals(
        run_dir=run_dir,
        models={
            "prior": (prior_visual, torch.device("cuda:0")),
            "actual": (actual, torch.device("cuda:0")),
            "factual_only_control": (control, torch.device("cuda:1")),
        },
        cache=pair_cache,
        pairs=pairs,
        builder=builder,
        layout_payload=layout_payload,
        token_count=m2_recipe.cache.token_count,
        criterion_config=foundation.set_loss_config,
    )
    m2._write_json_atomic(run_dir / "visual_artifacts.json", visuals)
    if pair_plan is None:
        acceptance = _acceptance(
            recipe=recipe,
            prior_pair=prior_pairs["smoke"],
            actual_pair=actual_pairs["smoke"],
            control_pair=control_pairs["smoke"],
            prior_natural=prior_natural,
            actual_natural=actual_natural,
        )
    elif recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE:
        if control_natural is None:
            raise RuntimeError("Occam decision requires control natural evaluation")
        acceptance = formal_counterfactual_measurement_occam_acceptance(
            recipe=recipe,
            prior_pairs=prior_pairs,
            actual_pairs=actual_pairs,
            control_pairs=control_pairs,
            prior_natural=prior_natural,
            actual_natural=actual_natural,
            control_natural=control_natural,
        )
    else:
        acceptance = formal_counterfactual_measurement_acceptance(
            recipe=recipe,
            prior_pairs=prior_pairs,
            actual_pairs=actual_pairs,
            control_pairs=control_pairs,
            prior_natural=prior_natural,
            actual_natural=actual_natural,
        )
    decision = {
        "schema": (
            "picf-next.counterfactual-measurement-decision.v1"
            if pair_plan is None
            else (
                "picf-next.counterfactual-measurement-decision.v3"
                if recipe.decision_rule == OCCAM_COMPLETE_SET_DECISION_RULE
                else "picf-next.counterfactual-measurement-decision.v2"
            )
        ),
        "gate": COUNTERFACTUAL_MEASUREMENT_GATE,
        **acceptance,
        "required_report_sha256": {
            name: m2._sha256(run_dir / name)
            for name in (
                "launch_manifest.json",
                "environment.json",
                "counterfactual_feature_manifest.json",
                "training_report.json",
                "evaluation_report.json",
                "visual_artifacts.json",
            )
        },
    }
    selected_candidate = decision.get("selected_candidate")
    if selected_candidate is not None:
        selected_name = {
            "counterfactual": "current_frame_counterfactual.pt",
            "factual_only_control": "current_frame_factual_only_control.pt",
        }[selected_candidate]
        decision["selected_checkpoint"] = {
            "path": f"checkpoints/{selected_name}",
            "sha256": training["checkpoints"][selected_name],
        }
    m2._write_json_atomic(run_dir / "machine_decision.json", decision)
    m2._emit_progress(
        "counterfactual_machine_decision",
        run_dir=str(run_dir),
        status=decision["status"],
        failed_checks=decision["failed_checks"],
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

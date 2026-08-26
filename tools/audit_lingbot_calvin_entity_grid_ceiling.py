#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Audit the exact entity-mask ceiling of LingBot's merged visual token grid.

The audit replays the frozen, source-disjoint ADR-138 evaluation frames through
the released LingBot processor only. It never loads policy weights or predicts
an entity. Instead, it projects the verified loss-side physical masks onto the
exact Qwen token lattice and computes the best possible soft-IoU on that
lattice. This separates representation-resolution limits from optimization
failure without consuming a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.official_config import official_lingbot_data_config

try:
    from tools.bootstrap_lingbot_vla2 import (
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        load_lingbot_training_config,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        load_lingbot_training_config,
    )


GRID_CEILING_REPORT_SCHEMA = "picf-next.lingbot-calvin-entity-grid-ceiling.v2"
_SUPPORTED_VISUAL_LATTICES = (8, 12)
_AREA_STRATA = (
    ("lt_2_percent", 0.0, 0.02),
    ("2_to_5_percent", 0.02, 0.05),
    ("ge_5_percent", 0.05, None),
)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_NATIVE_SOURCE") or (
        root / CHECKOUT_RELATIVE_PATH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=root / "configs/lingbot/calvin_robot.yaml",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=root / "configs/lingbot/calvin_data.json",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--dataset-split", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--physical-sidecar-root", type=Path)
    parser.add_argument("--physical-sidecar-manifest", type=Path)
    parser.add_argument("--physical-sidecar-manifest-sha256")
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument("--evaluation-plan", type=Path)
    parser.add_argument("--evaluation-plan-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument(
        "--visual-lattice",
        type=int,
        choices=_SUPPORTED_VISUAL_LATTICES,
        default=8,
    )
    parser.add_argument("--minimum-supervised-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _validate_args(args: argparse.Namespace) -> None:
    required_files = (
        args.patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
        args.physical_sidecar_manifest,
        args.representation_split,
        args.evaluation_plan,
    )
    required_directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.physical_sidecar_root,
    )
    if any(value is None or not Path(value).is_file() for value in required_files):
        raise FileNotFoundError("one or more grid-ceiling input files are absent")
    if any(value is None or not Path(value).is_dir() for value in required_directories):
        raise FileNotFoundError("one or more grid-ceiling input directories are absent")
    for name in (
        "physical_sidecar_manifest_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
    ):
        _require_sha256(name, getattr(args, name))
    if _sha256(args.physical_sidecar_manifest) != args.physical_sidecar_manifest_sha256:
        raise ValueError("physical sidecar manifest SHA-256 differs")
    if _sha256(args.representation_split) != args.representation_split_sha256:
        raise ValueError("representation split file SHA-256 differs")
    if _sha256(args.evaluation_plan) != args.evaluation_plan_sha256:
        raise ValueError("evaluation plan file SHA-256 differs")
    if isinstance(args.capacity, bool) or not isinstance(args.capacity, int) or args.capacity <= 0:
        raise ValueError("grid-ceiling capacity must be positive")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("grid-ceiling seed must be non-negative")
    fraction = args.minimum_supervised_fraction
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("minimum supervised fraction must lie in [0,1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)


def _evaluation_replay_seed(plan_sha256: str, sample_key: str) -> int:
    _require_sha256("evaluation plan artifact SHA-256", plan_sha256)
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("grid-ceiling sample key must be nonempty")
    return int.from_bytes(
        hashlib.sha256(f"{plan_sha256}\0{sample_key}".encode("ascii")).digest()[:8],
        "big",
    )


def _tensor_mapping_sha256(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"grid-ceiling model input is not a tensor: {name}")
        local = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(json.dumps(list(local.shape), separators=(",", ":")).encode())
        digest.update(local.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _area_stratum(area_fraction: float) -> str:
    for name, lower, upper in _AREA_STRATA:
        if area_fraction >= lower and (upper is None or area_fraction < upper):
            return name
    raise ValueError("grid-ceiling area fraction lies outside [0,1]")


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _summarize(samples: list[dict[str, Any]], *, partition: str) -> dict[str, Any]:
    selected = [sample for sample in samples if sample["partition"] == partition]
    rows = [row for sample in selected for row in sample["rows"]]
    if not selected or not rows:
        raise ValueError(f"grid-ceiling partition is empty: {partition}")

    def summarize_rows(chosen: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "entity_count": len(chosen),
            "mean_area_fraction": _mean([float(row["area_fraction"]) for row in chosen]),
            "mean_soft_iou_ceiling": _mean(
                [float(row["soft_iou_ceiling"]) for row in chosen]
            ),
            "minimum_soft_iou_ceiling": (
                min(float(row["soft_iou_ceiling"]) for row in chosen) if chosen else None
            ),
        }

    return {
        "sample_count": len(selected),
        "task_count": len({sample["task_key"] for sample in selected}),
        **summarize_rows(rows),
        "area_strata": {
            name: summarize_rows([row for row in rows if row["area_stratum"] == name])
            for name, _, _ in _AREA_STRATA
        },
    }


def _dummy_relation(*, visual_tokens: int, capacity: int) -> Any:
    from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput

    support = torch.zeros(1, visual_tokens, capacity, dtype=torch.float32)
    ownership = torch.full(
        (1, visual_tokens, capacity + 1),
        1 / (capacity + 1),
        dtype=torch.float32,
    )
    existence = torch.zeros(1, capacity, dtype=torch.float32)
    valid = torch.ones(1, visual_tokens, dtype=torch.bool)
    return PhysicalRelationOutput(
        support_logits=support,
        visible_support=support.sigmoid(),
        ownership=ownership,
        ownership_log_probability=ownership.log(),
        existence=existence.sigmoid(),
        existence_logits=existence,
        row_embeddings=torch.zeros(1, capacity, 1),
        relation_temperature=torch.ones(1),
        sensor_valid=valid,
        structural_sensor_valid=valid,
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(root=root, checkout=args.source_checkout, check_apply=True)
    prepared_source = validate_prepared_native_source(
        checkout=args.source_checkout,
        patch_path=args.patch,
    )
    if prepared_source.get("patched_source_sha256") != patch_report.get(
        "patched_source_sha256"
    ):
        raise RuntimeError("grid-ceiling LingBot source differs from immutable patch replay")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from transformers import AutoConfig

    from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.calvin import (
        build_native_calvin_replay_batch,
        collate_native_calvin_training_batch,
    )
    from picf_next.lingbot_native.calvin_entity_set import (
        build_task_independent_calvin_targets,
    )
    from picf_next.lingbot_native.entity_evaluation_plan import (
        ENTITY_EVALUATION_PARTITIONS,
        EntityEvaluationPlan,
        build_entity_evaluation_plan,
    )
    from picf_next.lingbot_native.entity_set_evaluation import (
        maximum_token_grid_soft_iou,
    )
    from picf_next.lingbot_native.lattice_feasibility import (
        configure_native_processor_lattice,
    )
    from picf_next.lingbot_native.representation_split import RepresentationTrialSplit

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    norm_stats = json.loads(args.norm_stats.read_text())
    validate_lingbot_calvin_norm_stats(norm_stats)
    source = norm_stats["source"]
    if (
        source["dataset_id"] != manifest.dataset_id
        or source["dataset_revision"] != manifest.dataset_revision
        or source["dataset_tree_sha256"] != manifest.tree_sha256
        or manifest.split_name != args.dataset_split.name
    ):
        raise ValueError("grid-ceiling CALVIN manifest and normalization differ")
    binding = validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=source["dataset_id"],
        dataset_revision=source["dataset_revision"],
        split_name=args.dataset_split.name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split.resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.physical_sidecar_root,
        index,
        manifest_path=args.physical_sidecar_manifest,
        expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
    )
    representation_split = RepresentationTrialSplit.load(args.representation_split)
    evaluation_plan = EntityEvaluationPlan.load(args.evaluation_plan)
    if evaluation_plan.representation_split_sha256 != representation_split.artifact_sha256:
        raise ValueError("grid-ceiling plan and representation split differ")

    training = load_lingbot_training_config(args.training_config)
    merged, _ = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=1,
    )
    qwen_config = AutoConfig.from_pretrained(  # nosec B615
        args.processor_dir,
        revision=QWEN_PROCESSOR_REVISION,
        local_files_only=True,
    )
    config = LingbotVLAV2Config(**merged)
    for key, value in merged.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    _merge_qwen_config(config, qwen_config)
    config.tokenizer_path = str(args.processor_dir.resolve())
    patch_size = int(qwen_config.vision_config.patch_size)
    merge_size = int(qwen_config.vision_config.spatial_merge_size)
    dataset = CalvinStatefulTransitionDataset(
        index,
        action_horizon=int(config.chunk_size),
    )
    if build_entity_evaluation_plan(representation_split, dataset) != evaluation_plan:
        raise ValueError("grid-ceiling evaluation plan is not reproducible from source")
    processor = build_processor(str(args.processor_dir.resolve()))
    processor_lattice = configure_native_processor_lattice(processor, args.visual_lattice)
    feature_transform = FeatureTransform(
        str(args.robot_config.resolve()),
        official_lingbot_data_config(json.loads(args.data_config.read_text())),
        config,
        processor,
        chunk_size=int(config.chunk_size),
        norm_stats_path=str(args.norm_stats.resolve()),
        use_depth_align=False,
        image_augment=False,
        use_future_image=False,
    )

    samples: list[dict[str, Any]] = []
    identity_rows: defaultdict[str, list[float]] = defaultdict(list)
    for item in evaluation_plan.items:
        replay_seed = _evaluation_replay_seed(
            evaluation_plan.artifact_sha256,
            item.sample_key,
        )
        planned = build_native_calvin_replay_batch(
            dataset,
            sample_key=item.sample_key,
            lane_id=item.rank,
            episode_instance_id=f"grid-ceiling/{item.partition}/{item.ordinal}",
            optimizer_step=0,
            replay_seed=replay_seed,
            device="cpu",
            dtype=torch.float32,
        )
        collated = collate_native_calvin_training_batch(
            planned.training,
            feature_transform=feature_transform,
            collator=VLADataCollatorWithPacking(),
            augmentation_seeds=planned.augmentation_seeds,
            source_digest=planned.source_digest,
        )
        grids = collated.model_inputs["image_grid_thw"]
        image_valid = collated.model_inputs["img_masks"].bool()
        raw_counts = grids.prod(dim=-1)
        visual_tokens = int(((raw_counts // (merge_size**2)) * image_valid).sum().item())
        expected_visual_tokens = 2 * args.visual_lattice**2
        if visual_tokens != expected_visual_tokens:
            raise RuntimeError(
                "grid-ceiling visual token count differs from the declared two-view lattice"
            )
        relation = _dummy_relation(visual_tokens=visual_tokens, capacity=args.capacity)
        target_bundle = build_task_independent_calvin_targets(
            requests_by_time=(collated.structural_target_requests,),
            model_inputs_by_time=(collated.model_inputs,),
            relations=(relation,),
            physical_sidecar=sidecar,
            capacity=args.capacity,
            patch_size=patch_size,
            merge_size=merge_size,
            minimum_supervised_fraction=args.minimum_supervised_fraction,
            capacity_seeds=planned.augmentation_seeds,
        )[0]
        targets = target_bundle.targets
        observed = targets.token_observed_fraction[0]
        denominator = observed.sum()
        if denominator <= 0:
            raise RuntimeError("grid-ceiling sample has no observed visual token")
        eligible = (
            targets.track_valid[0]
            & ~targets.capacity_censored[0]
            & targets.existence_valid[0]
            & (targets.existence[0] > 0)
        )
        rows: list[dict[str, Any]] = []
        for track_tensor in eligible.nonzero().flatten():
            track_index = int(track_tensor.item())
            target = targets.masks[0, track_index].float()
            weight = targets.mask_valid[0, track_index].float() * observed.float()
            target_mass = (target * weight).sum()
            if target_mass <= 0:
                continue
            area_fraction = float((target_mass / denominator).item())
            ceiling = float(maximum_token_grid_soft_iou(target, weight).item())
            identity_key = target_bundle.identity_keys_by_batch[0][track_index]
            identity_rows[identity_key].append(ceiling)
            rows.append(
                {
                    "track_index": track_index,
                    "identity_key": identity_key,
                    "area_fraction": area_fraction,
                    "area_stratum": _area_stratum(area_fraction),
                    "soft_iou_ceiling": ceiling,
                }
            )
        if not rows:
            raise RuntimeError("grid-ceiling sample has no eligible visible physical entity")
        samples.append(
            {
                "partition": item.partition,
                "ordinal": item.ordinal,
                "task_key": item.task_key,
                "sample_key": item.sample_key,
                "source_digest": planned.source_digest,
                "model_inputs_sha256": _tensor_mapping_sha256(dict(collated.model_inputs)),
                "visual_token_count": visual_tokens,
                "rows": rows,
            }
        )

    report = {
        "schema": GRID_CEILING_REPORT_SCHEMA,
        "status": "PASS",
        "dataset_manifest_file_sha256": _sha256(args.dataset_manifest),
        "dataset_runtime_binding": binding,
        "normalization_file_sha256": _sha256(args.norm_stats),
        "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
        "representation_split_file_sha256": args.representation_split_sha256,
        "representation_split_artifact_sha256": representation_split.artifact_sha256,
        "evaluation_plan_file_sha256": args.evaluation_plan_sha256,
        "evaluation_plan_artifact_sha256": evaluation_plan.artifact_sha256,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "visual_lattice": args.visual_lattice,
        "processor_lattice": processor_lattice,
        "capacity": args.capacity,
        "minimum_supervised_fraction": args.minimum_supervised_fraction,
        "sample_count": len(samples),
        "implementation_files": {
            "tools/audit_lingbot_calvin_entity_grid_ceiling.py": _sha256(
                root / "tools/audit_lingbot_calvin_entity_grid_ceiling.py"
            ),
            "src/picf_next/lingbot_native/calvin_entity_set.py": _sha256(
                root / "src/picf_next/lingbot_native/calvin_entity_set.py"
            ),
            "src/picf_next/lingbot_native/entity_set_evaluation.py": _sha256(
                root / "src/picf_next/lingbot_native/entity_set_evaluation.py"
            ),
            "src/picf_next/lingbot_native/lattice_feasibility.py": _sha256(
                root / "src/picf_next/lingbot_native/lattice_feasibility.py"
            ),
        },
        "summaries": {
            partition: _summarize(samples, partition=partition)
            for partition in ENTITY_EVALUATION_PARTITIONS
        },
        "identity_mean_soft_iou_ceiling": {
            identity: _mean(values) for identity, values in sorted(identity_rows.items())
        },
        "samples": samples,
    }
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

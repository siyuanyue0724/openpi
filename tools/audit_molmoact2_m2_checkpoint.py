#!/usr/bin/env python3
"""Re-evaluate one M2 checkpoint without training or authorizing later gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder  # noqa: E402
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_calvin import load_calvin_training_assets  # noqa: E402
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _evaluate,
    _keys_for_split,
    _load_cache,
    _write_json_atomic,
)


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
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> dict[str, object]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        text=True,
    ).strip()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=_ROOT,
    )
    audited_paths = (
        "src/picf_next/hosts/molmoact2_training.py",
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_checkpoint.py",
        "tools/run_molmoact2_m2_cloud.py",
    )
    return {
        "base_revision": revision,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "audited_file_sha256": {relative: _sha256(_ROOT / relative) for relative in audited_paths},
    }


def main() -> None:
    import torch

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M2 checkpoint audit requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("M2 checkpoint audit requires a CUDA device")

    config = args.config.expanduser().resolve()
    feature_cache = args.feature_cache.expanduser().resolve()
    dataset_split_root = args.dataset_split_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    recipe = load_molmoact2_m2_recipe(config)
    foundation = recipe.load_foundation(_ROOT)
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=dataset_split_root,
    )
    cache_manifest, cache = _load_cache(feature_cache, recipe)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"] if set(payload) == {"model"} else payload
    torch.manual_seed(recipe.optimization.seed)
    model = foundation.core_config.build_current_frame().to(device)
    model.load_state_dict(state, strict=True)
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(device)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)

    evaluation = {
        split: _evaluate(
            model=model,
            cache=cache,
            keys=_keys_for_split(cache, split),
            target_builder=target_builder,
            criterion=criterion,
            layout_payload=cache_manifest["processor_layout"],
            recipe=recipe,
            device=device,
            include_per_sample=True,
        )
        for split in ("validation", "heldout")
    }
    report = {
        "schema": "picf-next.molmoact2-m2-checkpoint-audit.v1",
        "authorizes_later_gates": False,
        "interpretation_boundary": (
            "This is a read-only recomputation under the current metric contract. "
            "It neither selects a checkpoint nor authorizes a training configuration."
        ),
        "source": _source_identity(),
        "config": str(config),
        "recipe_sha256": recipe.recipe_sha256,
        "feature_cache": str(feature_cache),
        "feature_cache_manifest_sha256": _sha256(feature_cache / "manifest.json"),
        "dataset_split_root": str(dataset_split_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "evaluation": evaluation,
    }
    _write_json_atomic(output, report)
    print(
        json.dumps(
            {
                split: {
                    name: metrics[name]
                    for name in (
                        "mean_object_dice",
                        "balanced_ownership_accuracy",
                        "count_mae",
                        "exact_count_accuracy",
                        "geometry_mae_model_chart",
                        "geometry_mae_physical",
                        "geometry_mae_physical_unit",
                    )
                }
                for split, metrics in evaluation.items()
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare positive-support CALVIN measurements with the removed one-token rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json",
    )
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache-manifest-sha256", required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-shards", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def legacy_one_token_fixed_point(
    ownership: np.ndarray,
    supervised: np.ndarray,
    *,
    minimum_mass: float = 1.0,
) -> np.ndarray:
    """Reproduce the removed resolution gate without retaining it in production."""

    ownership = np.asarray(ownership, dtype=np.float64)
    supervised = np.asarray(supervised, dtype=np.bool_)
    if ownership.ndim != 2 or ownership.shape[1] < 1:
        raise ValueError("ownership must be token-by-object-plus-context")
    if supervised.shape != (ownership.shape[0],):
        raise ValueError("supervision must align with ownership tokens")
    if not np.isfinite(ownership).all() or (ownership < 0.0).any():
        raise ValueError("ownership must be finite and nonnegative")
    if (
        isinstance(minimum_mass, bool | np.bool_)
        or not np.isfinite(minimum_mass)
        or minimum_mass <= 0.0
    ):
        raise ValueError("minimum mass must be finite and positive")

    objects = ownership[:, :-1]
    remaining = objects[supervised].sum(axis=0) >= minimum_mass
    while remaining.any():
        unresolved = (objects[:, ~remaining] > 0.0).any(axis=1)
        selective_supervised = supervised & ~unresolved
        mass = objects[selective_supervised].sum(axis=0)
        next_remaining = remaining & (mass >= minimum_mass)
        if np.array_equal(next_remaining, remaining):
            break
        remaining = next_remaining
    return remaining


def main() -> None:
    import torch

    from picf_next.hosts.molmoact2_training import (
        CalvinStatefulLossTargetLayout,
        CalvinVisibleObjectTargetBuilder,
    )
    from picf_next.models.evidence import ModalityTokenSpan
    from picf_next.training.stationary_calvin_stage import (
        load_stationary_calvin_stage_assets,
        load_stationary_calvin_stage_definition,
    )

    args = _parse_args()
    if args.cache_shards <= 0:
        raise ValueError("cache shards must be positive")
    stage_config = args.stage_config.expanduser().resolve()
    definition = load_stationary_calvin_stage_definition(
        stage_config,
        repository_root=_ROOT,
    )
    assets = load_stationary_calvin_stage_assets(
        definition,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
        feature_cache_root=args.feature_cache_root.expanduser().resolve(),
        feature_cache_manifest_sha256=args.feature_cache_manifest_sha256,
        physical_sidecar_root=args.physical_sidecar_root.expanduser().resolve(),
        cache_shards=args.cache_shards,
    )
    cache = assets.feature_cache
    layout = CalvinStatefulLossTargetLayout(
        token_valid=torch.ones((1, cache.token_count), dtype=torch.bool),
        spans=(ModalityTokenSpan(cache.modality, 0, cache.token_count),),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.float32,
        vision_patch_layout=cache.vision_layout(1),
    )
    builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)

    split_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "frames": 0,
            "positive_support_observations": 0,
            "one_token_observations": 0,
            "dropped_observations": 0,
        }
    )
    identity_stats: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "positive_support": 0,
            "one_token": 0,
            "dropped": 0,
            "dropped_mass": [],
            "examples": [],
        }
    )
    all_mass: list[float] = []

    for count, global_index in enumerate(sorted(cache.records), start=1):
        targets = builder.source_frames((cache.target_request(global_index),), layout)
        if targets.set_targets is None or len(targets.set_targets) != 1:
            raise RuntimeError("measurement audit requires exactly one set target")
        target = targets.set_targets[0]
        keys = tuple(target.temporal_identity_keys or ())
        ownership = target.ownership.detach().double().cpu().numpy()
        supervised = target.supervision_valid.detach().cpu().numpy()
        mass = ownership[supervised, :-1].sum(axis=0)
        if mass.shape != (len(keys),) or (mass <= 0.0).any():
            raise RuntimeError("positive-support target contains an invalid object column")
        legacy_keep = legacy_one_token_fixed_point(ownership, supervised)
        if legacy_keep.shape != (len(keys),):
            raise RuntimeError("legacy gate result differs from target cardinality")

        split = cache.records[global_index].split
        split_stats[split]["frames"] += 1
        split_stats[split]["positive_support_observations"] += len(keys)
        split_stats[split]["one_token_observations"] += int(legacy_keep.sum())
        split_stats[split]["dropped_observations"] += int((~legacy_keep).sum())
        for object_index, (identity, object_mass) in enumerate(
            zip(keys, mass.tolist(), strict=True)
        ):
            row = identity_stats[identity]
            row["positive_support"] += 1
            all_mass.append(float(object_mass))
            if legacy_keep[object_index]:
                row["one_token"] += 1
            else:
                row["dropped"] += 1
                row["dropped_mass"].append(float(object_mass))
                if len(row["examples"]) < 12:
                    row["examples"].append(
                        {
                            "global_index": global_index,
                            "split": split,
                            "projected_supervised_mass": float(object_mass),
                        }
                    )
        if count % 200 == 0:
            print(f"progress {count}/{len(cache.records)}", file=sys.stderr, flush=True)

    by_identity = {}
    for identity, values in sorted(identity_stats.items()):
        positive = int(values["positive_support"])
        dropped = int(values["dropped"])
        by_identity[identity] = {
            "positive_support": positive,
            "one_token": int(values["one_token"]),
            "dropped": dropped,
            "drop_fraction_of_positive_support": dropped / positive,
            "dropped_mass": _distribution(values["dropped_mass"]),
            "examples": values["examples"],
        }

    positive_total = sum(values["positive_support_observations"] for values in split_stats.values())
    dropped_total = sum(values["dropped_observations"] for values in split_stats.values())
    report = {
        "schema": "picf-next.calvin-measurement-contract-compare.v1",
        "status": "PASS",
        "code_revision": _revision(),
        "stage_config": {
            "path": str(stage_config),
            "sha256": _sha256(stage_config),
        },
        "frame_count": len(cache.records),
        "contracts": {
            "positive_support": (
                "projected supervised object mass > 0; restored production contract"
            ),
            "one_token": (
                "projected supervised object mass >= 1 plus greatest-fixed-point pruning; "
                "removed unaccepted contract"
            ),
        },
        "by_split": dict(sorted(split_stats.items())),
        "overall": {
            "positive_support_observations": positive_total,
            "one_token_observations": sum(
                values["one_token_observations"] for values in split_stats.values()
            ),
            "dropped_observations": dropped_total,
            "dropped_fraction": dropped_total / positive_total,
            "positive_support_mass": _distribution(all_mass),
        },
        "by_identity": by_identity,
        "runtime_target_leakage": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

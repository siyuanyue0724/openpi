#!/usr/bin/env python3
"""Measure action-independent shortcuts in an immutable FLARE target cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from picf_next.lingbot_native.future_latent_alignment import FLARE_TARGET_VIEW_ORDER
from picf_next.lingbot_native.future_latent_cache import FutureLatentTargetCache

REPORT_SCHEMA = "picf-next.adr209-flare-target-identifiability/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-trained-raw-loss", type=float)
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _episode_key(record: Mapping[str, Any]) -> str:
    sample_key = record.get("sample_key")
    if not isinstance(sample_key, str) or "/frame-" not in sample_key:
        raise ValueError("FLARE cache sample key lacks its episode/frame identity")
    return sample_key.rsplit("/frame-", 1)[0]


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("cosine distribution must be one non-empty finite vector")
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "p05": torch.quantile(values, 0.05).item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p95": torch.quantile(values, 0.95).item(),
    }


def summarize_normalized_targets(
    normalized_targets: torch.Tensor,
    *,
    current_future_pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Return exact cosine shortcut statistics for [sample,token,width] targets."""

    if (
        normalized_targets.ndim != 3
        or not normalized_targets.is_floating_point()
        or normalized_targets.shape[0] < 2
        or normalized_targets.shape[1] < 2
        or normalized_targets.shape[2] < 2
        or not torch.isfinite(normalized_targets).all()
    ):
        raise ValueError("normalized FLARE targets must be finite [sample,token,width]")
    norms = normalized_targets.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError("FLARE targets must be unit-normalized before shortcut analysis")
    sample_count = normalized_targets.shape[0]
    position_mean = normalized_targets.mean(dim=0)
    position_concentration = position_mean.norm(dim=-1)
    fixed_cosine = position_concentration.mean()
    offset = max(1, sample_count // 2 - 1)
    unrelated_cosines = (
        normalized_targets * normalized_targets.roll(offset, dims=0)
    ).sum(dim=-1).mean(dim=-1)
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "token_count": normalized_targets.shape[1],
        "width": normalized_targets.shape[2],
        "optimal_sample_independent_position_template": {
            "mean_cosine": fixed_cosine.item(),
            "raw_cosine_loss": (1.0 - fixed_cosine).item(),
            "mean_unit_vector_variance": (
                1.0 - position_concentration.square().mean()
            ).item(),
            "derivation": (
                "c_j=E[y_j]/||E[y_j]|| maximizes E[c_j^T y_j] among "
                "sample-independent unit predictors"
            ),
        },
        "deterministic_cross_sample_offset": offset,
        "cross_sample_same_position_cosine": _distribution(unrelated_cosines),
    }
    if current_future_pairs:
        pair_values = []
        for current_row, future_row in current_future_pairs:
            if not (
                0 <= current_row < sample_count
                and 0 <= future_row < sample_count
                and current_row != future_row
            ):
                raise ValueError("current/future pair points outside the target cache")
            pair_values.append(
                (
                    normalized_targets[current_row]
                    * normalized_targets[future_row]
                ).sum(dim=-1).mean()
            )
        result["cached_current_to_t_plus_h_cosine"] = _distribution(
            torch.stack(pair_values)
        )
    else:
        result["cached_current_to_t_plus_h_cosine"] = None
    return result


def _load_cache(cache_root: Path) -> tuple[dict[str, Any], torch.Tensor]:
    verified = FutureLatentTargetCache(cache_root, verify_shards=True)
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("manifest_sha256") != verified.manifest_sha256:
        raise ValueError("verified FLARE cache and manifest identities differ")
    tensors = []
    for shard in manifest["shards"]:
        path = cache_root / shard["path"]
        value = load_file(str(path), device="cpu")
        if set(value) != {"targets"}:
            raise ValueError(f"FLARE shard has an unexpected tensor table: {path}")
        tensors.append(value["targets"])
    targets = torch.cat(tensors, dim=0)
    if targets.shape[0] != len(manifest["records"]):
        raise ValueError("FLARE manifest and concatenated target count differ")
    return manifest, targets


def _temporal_pairs(records: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    by_future: dict[tuple[str, int], int] = {}
    for record_index, record in enumerate(records):
        key = (_episode_key(record), int(record["future_global_index"]))
        if key in by_future:
            raise ValueError("FLARE cache repeats one episode/future-frame identity")
        by_future[key] = record_index
    pairs = []
    for record_index, record in enumerate(records):
        prior = by_future.get((_episode_key(record), int(record["source_global_index"])))
        if prior is not None:
            pairs.append((prior, record_index))
    return pairs


def _adjacent_pairs(records: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    by_source = {
        (_episode_key(record), int(record["source_global_index"])): record_index
        for record_index, record in enumerate(records)
    }
    result = []
    for record_index, record in enumerate(records):
        next_row = by_source.get(
            (_episode_key(record), int(record["source_global_index"]) + 1)
        )
        if next_row is not None:
            result.append((record_index, next_row))
    return result


def _paired_cosine(
    normalized_targets: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, float | int]:
    if not pairs:
        raise ValueError("FLARE cache has no same-episode adjacent target pairs")
    values = torch.stack(
        [
            (
                normalized_targets[left] * normalized_targets[right]
            ).sum(dim=-1).mean()
            for left, right in pairs
        ]
    )
    return _distribution(values)


def audit_cache(
    cache_root: Path,
    *,
    observed_trained_raw_loss: float | None,
) -> dict[str, Any]:
    manifest, targets = _load_cache(cache_root)
    if targets.shape[1] % len(FLARE_TARGET_VIEW_ORDER) != 0:
        raise ValueError("FLARE target tokens cannot be divided across the frozen views")
    normalized = torch.nn.functional.normalize(targets.float(), dim=-1)
    records = manifest["records"]
    current_future_pairs = _temporal_pairs(records)
    tokens_per_view = targets.shape[1] // len(FLARE_TARGET_VIEW_ORDER)
    views: dict[str, Any] = {}
    for index, view in enumerate(FLARE_TARGET_VIEW_ORDER):
        start = index * tokens_per_view
        stop = start + tokens_per_view
        views[view] = summarize_normalized_targets(
            normalized[:, start:stop],
            current_future_pairs=current_future_pairs,
        )
    complete = summarize_normalized_targets(
        normalized,
        current_future_pairs=current_future_pairs,
    )
    adjacent_pairs = _adjacent_pairs(records)
    complete["same_episode_adjacent_target_cosine"] = _paired_cosine(
        normalized,
        adjacent_pairs,
    )
    fixed_loss = float(
        complete["optimal_sample_independent_position_template"]["raw_cosine_loss"]
    )
    observed = None
    if observed_trained_raw_loss is not None:
        if not math.isfinite(observed_trained_raw_loss) or observed_trained_raw_loss < 0:
            raise ValueError("observed trained raw loss must be finite and non-negative")
        observed = {
            "raw_loss": observed_trained_raw_loss,
            "fixed_template_loss_ratio": observed_trained_raw_loss / fixed_loss,
            "fixed_template_has_lower_loss": fixed_loss < observed_trained_raw_loss,
        }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "cache_root": str(cache_root.resolve()),
        "cache_manifest_sha256": manifest["manifest_sha256"],
        "target_shape": list(targets.shape),
        "target_dtype": str(targets.dtype).removeprefix("torch."),
        "target_norm": {
            "mean": targets.norm(dim=-1).mean().item(),
            "standard_deviation": targets.norm(dim=-1).std().item(),
        },
        "current_future_pair_count": len(current_future_pairs),
        "same_episode_adjacent_pair_count": len(adjacent_pairs),
        "complete_target": complete,
        "views": views,
        "observed_candidate_endpoint": observed,
        "interpretation_boundary": {
            "establishes": (
                "whether this frozen target admits sample-independent or current-frame "
                "cosine shortcuts"
            ),
            "does_not_establish": (
                "which shortcut a trained policy uses; that requires action and target "
                "interventions"
            ),
        },
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    args = _parse_args()
    report = audit_cache(
        args.cache_root,
        observed_trained_raw_loss=args.observed_trained_raw_loss,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(args.output)


if __name__ == "__main__":
    main()

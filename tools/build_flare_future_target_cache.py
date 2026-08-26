#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build exact frozen SigLIP2 `t+16` targets for a FLARE stream prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="FLARE future-target cache builder",
)

import torch

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.calvin import build_native_calvin_physical_episode_domain
from picf_next.lingbot_native.future_latent_alignment import FutureLatentAlignmentConfig
from picf_next.lingbot_native.future_latent_cache import (
    FrozenSiglip2FutureEncoder,
    FutureLatentCacheContract,
    build_calvin_future_latent_records,
    future_latent_source_keys_digest,
    teacher_asset_digests,
    write_future_latent_target_cache,
)
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from picf_next.training.control import load_frozen_episode_stream_plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")


def _validate_file_digest(path: Path, expected: str, name: str) -> str:
    _require_sha256(expected, name)
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} file SHA-256 differs")
    return observed


def _validate_teacher_file_manifest(
    *,
    model_root: Path,
    manifest: Path,
    expected_sha256: str,
) -> dict[str, object]:
    manifest_sha256 = _validate_file_digest(
        manifest,
        expected_sha256,
        "teacher file manifest",
    )
    rows: list[tuple[str, str]] = []
    for line_number, raw in enumerate(manifest.read_text("utf-8").splitlines(), start=1):
        if not raw:
            continue
        try:
            digest, relative = raw.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"teacher file manifest line {line_number} is malformed"
            ) from error
        relative = relative.removeprefix("*").strip()
        _require_sha256(digest, f"teacher file line {line_number}")
        path = PurePosixPath(relative)
        if path.is_absolute() or path.as_posix() != relative or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError("teacher file manifest contains an unsafe relative path")
        rows.append((digest, relative))
    if not rows or len({relative for _digest, relative in rows}) != len(rows):
        raise ValueError("teacher file manifest must contain unique non-empty paths")
    for digest, relative in rows:
        path = model_root / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"teacher file differs from manifest: {relative}")
    return {
        "file_count": len(rows),
        "manifest_sha256": manifest_sha256,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--stream-plan", type=Path, required=True)
    parser.add_argument("--stream-plan-sha256", required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-files-manifest", type=Path)
    parser.add_argument("--teacher-files-manifest-sha256")
    parser.add_argument("--training-prefix-steps", type=int, required=True)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--records-per-shard", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if args.build_report.exists() or args.build_report.is_symlink():
        raise FileExistsError(args.build_report)
    if args.training_prefix_steps <= 0:
        raise ValueError("training prefix steps must be positive")
    if args.encoder_batch_size <= 0 or args.records_per_shard <= 0:
        raise ValueError("FLARE cache batch and shard sizes must be positive")
    if (args.teacher_files_manifest is None) != (
        args.teacher_files_manifest_sha256 is None
    ):
        raise ValueError("teacher file manifest and digest must be supplied together")

    stream_file_sha256 = _validate_file_digest(
        args.stream_plan,
        args.stream_plan_sha256,
        "stream plan",
    )
    representation_split_sha256 = _validate_file_digest(
        args.representation_split,
        args.representation_split_sha256,
        "representation split",
    )
    teacher_file_receipt = None
    if args.teacher_files_manifest is not None:
        teacher_file_receipt = _validate_teacher_file_manifest(
            model_root=args.teacher_root,
            manifest=args.teacher_files_manifest,
            expected_sha256=args.teacher_files_manifest_sha256,
        )

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.resolve().name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    representation_split = RepresentationTrialSplit.load(args.representation_split)
    config = FutureLatentAlignmentConfig()
    config.assert_adr209_complete()
    stream_plan = load_frozen_episode_stream_plan(
        args.stream_plan,
        episodes=build_native_calvin_physical_episode_domain(
            dataset,
            excluded_source_episode_indices=(
                representation_split.stream_domain_excluded_source_episode_indices
            ),
            minimum_future_source_frames=config.target_offset_source_frames,
        ),
    )
    if args.training_prefix_steps > stream_plan.total_steps:
        raise ValueError("FLARE cache prefix exceeds the frozen stream")

    visited_keys = {
        transition.sample.sample_key
        for optimizer_step in range(args.training_prefix_steps)
        for transition in stream_plan.global_batch(optimizer_step).transitions
    }
    if not visited_keys:
        raise RuntimeError("FLARE stream prefix contains no training sample")
    identities_by_key: dict[str, tuple[str, int, int]] = {}
    for sample_key in visited_keys:
        source_index = dataset.source_global_index_by_key(sample_key)
        future_index = dataset.future_source_global_indices_by_key(
            sample_key,
            count=config.target_offset_source_frames,
        )[-1]
        identities_by_key[sample_key] = (sample_key, source_index, future_index)
    identities = tuple(sorted(identities_by_key.values()))
    sample_keys = tuple(identity[0] for identity in identities)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.compute_dtype]
    encoder = FrozenSiglip2FutureEncoder.from_pretrained(
        args.teacher_root,
        device=torch.device(args.device),
        compute_dtype=dtype,
    )
    records = build_calvin_future_latent_records(
        dataset,
        sample_keys=sample_keys,
        encoder=encoder,
        batch_size=args.encoder_batch_size,
    )
    observed_identities = tuple(
        (record.sample_key, record.source_global_index, record.future_global_index)
        for record in records
    )
    if observed_identities != identities:
        raise RuntimeError("FLARE cache encoder changed source record order or identity")
    assets = teacher_asset_digests(args.teacher_root)
    contract = FutureLatentCacheContract(
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=manifest.split_name,
        dataset_tree_sha256=manifest.tree_sha256,
        stream_plan_sha256=stream_plan.plan_sha256,
        stream_plan_file_sha256=stream_file_sha256,
        representation_split_sha256=representation_split_sha256,
        source_keys_sha256=future_latent_source_keys_digest(identities),
        expected_record_count=len(records),
        training_prefix_steps=args.training_prefix_steps,
        alignment_config_digest=config.digest,
        **assets,
    )
    cache_manifest_sha256 = write_future_latent_target_cache(
        args.output_root,
        contract=contract,
        records=records,
        records_per_shard=args.records_per_shard,
    )
    report = {
        "alignment_config": config.__dict__ if hasattr(config, "__dict__") else {
            name: getattr(config, name) for name in config.__dataclass_fields__
        },
        "cache_manifest_sha256": cache_manifest_sha256,
        "contract": contract.to_dict(),
        "encoder_batch_size": args.encoder_batch_size,
        "output_root": str(args.output_root.resolve()),
        "records_per_shard": args.records_per_shard,
        "schema": "picf-next.flare-future-target-cache-build.v1",
        "teacher_file_receipt": teacher_file_receipt,
        "unique_record_count": len(records),
        "visit_count": args.training_prefix_steps * stream_plan.global_batch_size,
    }
    write_text_durable_exclusive(
        args.build_report,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

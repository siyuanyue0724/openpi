#!/usr/bin/env python3
"""Fail-closed provenance audit for the exact LingBot plus VidEoMT transplant."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "picf-next.full-transplant-sources.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CRITERION_UPSTREAM_IMPORT = "from videomt.utils.misc import is_dist_avail_and_initialized"
CRITERION_VENDORED_IMPORT = (
    "from picf_next._vendor.videomt.utils.misc import is_dist_avail_and_initialized"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _resolve_inside(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to the repository")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label} escapes the repository root")
    return resolved


def _require_file_hash(path: Path, expected: object, *, label: str) -> None:
    digest = _require_sha256(expected, label=f"{label}.sha256")
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 {actual} differs from {digest}")


def _run(command: list[str], *, cwd: Path | None = None) -> bytes:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout


def _git_object(repository: Path, commit: str, relative_path: str) -> bytes:
    try:
        return _run(["git", "show", f"{commit}:{relative_path}"], cwd=repository)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Git object {commit}:{relative_path} is unavailable in {repository}"
        ) from error


def _validate_assets(root: Path, assets: object, *, label: str) -> dict[str, int]:
    rows = _require_sequence(assets, label=f"{label}.assets")
    seen: set[str] = set()
    total_bytes = 0
    for index, item in enumerate(rows):
        row = _require_mapping(item, label=f"{label}.assets[{index}]")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError(f"{label} has an invalid or duplicate asset path {relative!r}")
        expected_bytes = row.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"{label}.{relative}.bytes must be positive")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"{label} asset is absent: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{label} asset {relative} has {actual_bytes} bytes, expected {expected_bytes}"
            )
        _require_file_hash(path, row.get("sha256"), label=f"{label}.{relative}")
        seen.add(relative)
        total_bytes += actual_bytes
    if not rows:
        raise ValueError(f"{label} must declare at least one asset")
    return {"asset_count": len(rows), "byte_count": total_bytes}


def _validate_lingbot_manifest(donor: Mapping[str, Any], *, root: Path) -> dict[str, int]:
    _require_commit(donor.get("source_commit"), label="lingbot.source_commit")
    _require_commit(donor.get("audited_public_head"), label="lingbot.audited_public_head")
    source_files = _require_mapping(donor.get("source_files"), label="lingbot.source_files")
    patched_files = _require_mapping(
        donor.get("patched_source_files"), label="lingbot.patched_source_files"
    )
    if set(patched_files) - set(source_files):
        raise ValueError("every patched LingBot path must have an audited upstream source")
    for path, digest in source_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("LingBot source paths must be nonempty strings")
        _require_sha256(digest, label=f"lingbot.source_files.{path}")
    for path, digest in patched_files.items():
        _require_sha256(digest, label=f"lingbot.patched_source_files.{path}")

    core = _require_sequence(
        donor.get("public_head_identical_core_files"),
        label="lingbot.public_head_identical_core_files",
    )
    if not core or not all(isinstance(path, str) and path in source_files for path in core):
        raise ValueError("LingBot public-head core paths must reference audited sources")

    overlays = _require_sequence(donor.get("ordered_overlays"), label="lingbot.overlays")
    if len(overlays) != 4:
        raise ValueError("LingBot exact runtime requires the four frozen ordered overlays")
    for index, item in enumerate(overlays):
        overlay = _require_mapping(item, label=f"lingbot.overlays[{index}]")
        path = _resolve_inside(root, overlay.get("path"), label="LingBot overlay path")
        _require_file_hash(path, overlay.get("sha256"), label=f"lingbot overlay {index}")

    checkpoint = _require_mapping(donor.get("checkpoint"), label="lingbot.checkpoint")
    processor = _require_mapping(donor.get("processor"), label="lingbot.processor")
    _require_commit(checkpoint.get("revision"), label="lingbot.checkpoint.revision")
    _require_commit(processor.get("revision"), label="lingbot.processor.revision")
    for label, block in (("checkpoint", checkpoint), ("processor", processor)):
        assets = _require_sequence(block.get("assets"), label=f"lingbot.{label}.assets")
        if not assets:
            raise ValueError(f"LingBot {label} must declare exact assets")
        seen: set[str] = set()
        for index, item in enumerate(assets):
            row = _require_mapping(item, label=f"lingbot.{label}.assets[{index}]")
            path = row.get("path")
            if not isinstance(path, str) or not path or path in seen:
                raise ValueError(f"LingBot {label} has an invalid asset path")
            if not isinstance(row.get("bytes"), int) or row["bytes"] <= 0:
                raise ValueError(f"LingBot {label}.{path}.bytes must be positive")
            _require_sha256(row.get("sha256"), label=f"lingbot.{label}.{path}")
            seen.add(path)
    return {
        "source_file_count": len(source_files),
        "patched_source_file_count": len(patched_files),
        "overlay_count": len(overlays),
        "checkpoint_asset_count": len(checkpoint["assets"]),
        "processor_asset_count": len(processor["assets"]),
    }


def _validate_videomt_manifest(donor: Mapping[str, Any], *, root: Path) -> dict[str, int]:
    _require_commit(donor.get("source_commit"), label="videomt.source_commit")
    snapshot = _resolve_inside(root, donor.get("snapshot_root"), label="videomt.snapshot_root")
    vendor = _resolve_inside(root, donor.get("vendor_root"), label="videomt.vendor_root")
    rows = _require_sequence(donor.get("source_files"), label="videomt.source_files")
    if not rows:
        raise ValueError("VidEoMT must declare normative source files")
    adapters: dict[str, int] = {}
    seen_sources: set[str] = set()
    seen_vendors: set[str] = set()
    for index, item in enumerate(rows):
        row = _require_mapping(item, label=f"videomt.source_files[{index}]")
        source_relative = row.get("source")
        if not isinstance(source_relative, str) or not source_relative:
            raise ValueError("VidEoMT source path must be nonempty")
        if source_relative in seen_sources:
            raise ValueError(f"duplicate VidEoMT source path {source_relative}")
        source_path = snapshot / source_relative
        expected = row.get("sha256")
        _require_file_hash(source_path, expected, label=f"videomt source {source_relative}")
        adapter = row.get("adapter")
        if adapter not in {"byte-identical", "single-import-rewrite", "normative-reference"}:
            raise ValueError(f"unsupported VidEoMT adapter {adapter!r}")
        adapters[adapter] = adapters.get(adapter, 0) + 1
        vendor_relative = row.get("vendor")
        if adapter == "normative-reference":
            if vendor_relative is not None:
                raise ValueError("normative-reference source must not name a vendored file")
        else:
            if not isinstance(vendor_relative, str) or not vendor_relative:
                raise ValueError("vendored VidEoMT source must name its destination")
            if vendor_relative in seen_vendors:
                raise ValueError(f"duplicate VidEoMT vendor path {vendor_relative}")
            vendor_path = vendor / vendor_relative
            if not vendor_path.is_file():
                raise ValueError(f"vendored VidEoMT source is absent: {vendor_path}")
            if adapter == "byte-identical":
                if vendor_path.read_bytes() != source_path.read_bytes():
                    raise ValueError(f"VidEoMT vendor {vendor_relative} is not byte-identical")
            else:
                upstream = source_path.read_text(encoding="utf-8")
                adapted = vendor_path.read_text(encoding="utf-8")
                if adapted.count(CRITERION_VENDORED_IMPORT) != 1:
                    raise ValueError("VidEoMT criterion must contain exactly one approved import")
                normalized = adapted.replace(CRITERION_VENDORED_IMPORT, CRITERION_UPSTREAM_IMPORT)
                if normalized.rstrip("\n") != upstream.rstrip("\n"):
                    raise ValueError("VidEoMT criterion differs beyond the approved import rewrite")
            seen_vendors.add(vendor_relative)
        seen_sources.add(source_relative)

    checkpoint = _require_mapping(donor.get("checkpoint"), label="videomt.checkpoint")
    _require_commit(
        checkpoint.get("repository_commit"), label="videomt.checkpoint.repository_commit"
    )
    _require_sha256(checkpoint.get("sha256"), label="videomt.checkpoint.sha256")
    for field in ("bytes", "model_tensors", "backbone_tensors", "model_numel"):
        if not isinstance(checkpoint.get(field), int) or checkpoint[field] <= 0:
            raise ValueError(f"videomt.checkpoint.{field} must be positive")
    return {"source_file_count": len(rows), **adapters}


def _validate_dinov3_manifest(donor: Mapping[str, Any]) -> dict[str, int]:
    _require_commit(donor.get("source_commit"), label="dinov3.source_commit")
    if not isinstance(donor.get("released_pretrain_blob_bytes"), int):
        raise ValueError("dinov3.released_pretrain_blob_bytes must be an integer")
    bundle = _require_mapping(donor.get("runtime_bundle"), label="dinov3.runtime_bundle")
    if bundle.get("derivation") != "mechanical-key-conversion-from-released-videomt-checkpoint":
        raise ValueError("DINOv3 runtime bundle must use the frozen mechanical conversion")
    _require_sha256(bundle.get("config_sha256"), label="dinov3.runtime_bundle.config_sha256")
    _require_sha256(
        bundle.get("source_checkpoint_sha256"),
        label="dinov3.runtime_bundle.source_checkpoint_sha256",
    )
    if not isinstance(bundle.get("config_bytes"), int) or bundle["config_bytes"] <= 0:
        raise ValueError("dinov3.runtime_bundle.config_bytes must be positive")
    return {"runtime_bundle_contracts": 1}


def _verify_lingbot_git_source(
    donor: Mapping[str, Any], *, repository: Path, root: Path
) -> dict[str, int]:
    if not (repository / ".git").exists():
        raise ValueError(f"LingBot source repository is absent: {repository}")
    source_commit = str(donor["source_commit"])
    public_head = str(donor["audited_public_head"])
    source_files = _require_mapping(donor["source_files"], label="lingbot.source_files")
    for path, expected in source_files.items():
        actual = _sha256_bytes(_git_object(repository, source_commit, str(path)))
        if actual != expected:
            raise ValueError(f"LingBot upstream source {path} differs at {source_commit}")
    for path in donor["public_head_identical_core_files"]:
        public_digest = _sha256_bytes(_git_object(repository, public_head, str(path)))
        if public_digest != source_files[path]:
            raise ValueError(f"LingBot public-head core file drifted: {path}")

    with tempfile.TemporaryDirectory(prefix="picf-full-transplant-lingbot-") as temporary:
        exported = Path(temporary)
        archive = _run(["git", "archive", "--format=tar", source_commit], cwd=repository)
        archive_path = exported / "source.tar"
        archive_path.write_bytes(archive)
        source_root = exported / "source"
        source_root.mkdir()
        with tarfile.open(archive_path, mode="r:") as stream:
            stream.extractall(source_root, filter="data")
        for index, item in enumerate(donor["ordered_overlays"]):
            overlay = _require_mapping(item, label=f"lingbot.overlays[{index}]")
            overlay_path = _resolve_inside(root, overlay["path"], label="LingBot overlay path")
            try:
                _run(["git", "apply", "--check", str(overlay_path)], cwd=source_root)
                _run(["git", "apply", str(overlay_path)], cwd=source_root)
            except subprocess.CalledProcessError as error:
                raise ValueError(f"LingBot overlay {index} cannot be replayed in order") from error
        for path, expected in donor["patched_source_files"].items():
            _require_file_hash(
                source_root / path,
                expected,
                label=f"replayed LingBot source {path}",
            )
    return {
        "upstream_source_files_verified": len(source_files),
        "public_head_core_files_verified": len(donor["public_head_identical_core_files"]),
        "patched_source_files_replayed": len(donor["patched_source_files"]),
    }


def _verify_prepared_lingbot(donor: Mapping[str, Any], *, checkout: Path) -> dict[str, int]:
    if not (checkout / ".git").exists():
        raise ValueError(f"prepared LingBot checkout is absent: {checkout}")
    head = _run(["git", "rev-parse", "HEAD"], cwd=checkout).decode().strip()
    if head != donor["source_commit"]:
        raise ValueError(f"prepared LingBot HEAD {head} differs from frozen source")
    for path, expected in donor["patched_source_files"].items():
        _require_file_hash(checkout / path, expected, label=f"prepared LingBot source {path}")
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=checkout
    ).decode()
    dirty = {line[3:] for line in status.splitlines() if line}
    expected_dirty = set(donor["patched_source_files"])
    if dirty != expected_dirty:
        raise ValueError(f"prepared LingBot checkout has unrelated paths: {sorted(dirty)}")
    return {"prepared_source_files_verified": len(expected_dirty)}


def _verify_videomt_checkpoint(
    donor: Mapping[str, Any], *, checkpoint_path: Path
) -> dict[str, int]:
    contract = _require_mapping(donor["checkpoint"], label="videomt.checkpoint")
    if not checkpoint_path.is_file():
        raise ValueError(f"VidEoMT checkpoint is absent: {checkpoint_path}")
    if checkpoint_path.stat().st_size != contract["bytes"]:
        raise ValueError("VidEoMT checkpoint byte count differs from release")
    _require_file_hash(checkpoint_path, contract["sha256"], label="released VidEoMT checkpoint")
    from picf_next.videomt_exact.checkpoint import inspect_published_checkpoint

    receipt = inspect_published_checkpoint(checkpoint_path, require_release_match=True)
    return {
        "tensor_count": receipt.tensor_count,
        "backbone_tensor_count": receipt.backbone_tensor_count,
        "model_numel": receipt.model_numel,
    }


def _verify_dinov3_bundle(
    donor: Mapping[str, Any], *, bundle_path: Path, checkpoint_path: Path
) -> dict[str, int]:
    config_path = bundle_path / "config.json"
    weights_path = bundle_path / "model.safetensors"
    receipt_path = bundle_path / "conversion_receipt.json"
    for path in (config_path, weights_path, receipt_path):
        if not path.is_file():
            raise ValueError(f"DINOv3 runtime bundle is incomplete: {path}")
    contract = _require_mapping(donor["runtime_bundle"], label="dinov3.runtime_bundle")
    if config_path.stat().st_size != contract["config_bytes"]:
        raise ValueError("DINOv3 runtime config byte count differs")
    _require_file_hash(config_path, contract["config_sha256"], label="DINOv3 runtime config")
    receipt = _require_mapping(
        json.loads(receipt_path.read_text(encoding="utf-8")),
        label="DINOv3 conversion receipt",
    )
    published = _require_mapping(
        receipt.get("published_checkpoint"), label="DINOv3 receipt checkpoint"
    )
    if published.get("sha256") != contract["source_checkpoint_sha256"]:
        raise ValueError("DINOv3 bundle receipt names the wrong source checkpoint")

    from safetensors.torch import load_file

    from picf_next.videomt_exact.checkpoint import (
        _load_model_state,
        hf_dinov3_state_from_published,
    )

    _keys, published_state = _load_model_state(checkpoint_path)
    expected_state, expected_sources = hf_dinov3_state_from_published(published_state)
    actual_state = load_file(weights_path, device="cpu")
    if set(actual_state) != set(expected_state):
        raise ValueError("DINOv3 runtime bundle tensor keys differ from exact conversion")
    import torch

    for name, expected in expected_state.items():
        if not torch.equal(actual_state[name], expected):
            raise ValueError(f"DINOv3 runtime tensor differs from exact conversion: {name}")
    if receipt.get("source_by_target") != expected_sources:
        raise ValueError("DINOv3 conversion source map differs from exact conversion")
    if receipt.get("converted_tensor_count") != len(expected_state):
        raise ValueError("DINOv3 conversion receipt tensor count differs")
    return {
        "tensor_count": len(actual_state),
        "model_bytes": weights_path.stat().st_size,
    }


def load_and_validate_manifest(
    manifest_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), label="full transplant manifest"
    )
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported full transplant manifest schema")
    architecture = _resolve_inside(
        root, manifest.get("architecture_contract"), label="architecture_contract"
    )
    if not architecture.is_file():
        raise ValueError(f"architecture contract is absent: {architecture}")
    policy = _require_mapping(manifest.get("policy"), label="policy")
    required_policy = {
        "exact_upstream_must_precede_adaptation",
        "scientific_adapters_require_named_arms",
        "silent_simplification_forbidden",
        "strict_runtime_requires_all_external_assets",
    }
    if set(policy) != required_policy or any(policy[name] is not True for name in required_policy):
        raise ValueError("full transplant policy must retain all four fail-closed requirements")
    donors = _require_mapping(manifest.get("donors"), label="donors")
    if set(donors) != {"lingbot_vla2", "videomt", "dinov3"}:
        raise ValueError("full transplant donor set must be exactly LingBot, VidEoMT and DINOv3")
    summary = {
        "lingbot": _validate_lingbot_manifest(
            _require_mapping(donors["lingbot_vla2"], label="lingbot_vla2"), root=root
        ),
        "videomt": _validate_videomt_manifest(
            _require_mapping(donors["videomt"], label="videomt"), root=root
        ),
        "dinov3": _validate_dinov3_manifest(_require_mapping(donors["dinov3"], label="dinov3")),
    }
    return dict(manifest), summary


def audit_full_transplant_contract(
    manifest_path: Path,
    *,
    root: Path,
    lingbot_repository: Path | None = None,
    prepared_lingbot_checkout: Path | None = None,
    lingbot_checkpoint_dir: Path | None = None,
    processor_dir: Path | None = None,
    videomt_checkpoint: Path | None = None,
    dinov3_bundle: Path | None = None,
    strict_runtime: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = root.resolve()
    manifest, repository_summary = load_and_validate_manifest(manifest_path, root=root)
    donors = manifest["donors"]
    external: dict[str, Any] = {}

    if strict_runtime:
        required = {
            "lingbot_repository": lingbot_repository,
            "prepared_lingbot_checkout": prepared_lingbot_checkout,
            "lingbot_checkpoint_dir": lingbot_checkpoint_dir,
            "processor_dir": processor_dir,
            "videomt_checkpoint": videomt_checkpoint,
            "dinov3_bundle": dinov3_bundle,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"strict runtime audit requires: {', '.join(missing)}")

    lingbot = donors["lingbot_vla2"]
    if lingbot_repository is not None:
        external["lingbot_source"] = _verify_lingbot_git_source(
            lingbot,
            repository=lingbot_repository.resolve(),
            root=root,
        )
    if prepared_lingbot_checkout is not None:
        external["prepared_lingbot"] = _verify_prepared_lingbot(
            lingbot, checkout=prepared_lingbot_checkout.resolve()
        )
    if lingbot_checkpoint_dir is not None:
        external["lingbot_checkpoint"] = _validate_assets(
            lingbot_checkpoint_dir.resolve(),
            lingbot["checkpoint"]["assets"],
            label="LingBot checkpoint",
        )
    if processor_dir is not None:
        external["lingbot_processor"] = _validate_assets(
            processor_dir.resolve(),
            lingbot["processor"]["assets"],
            label="LingBot processor",
        )
    if videomt_checkpoint is not None:
        external["videomt_checkpoint"] = _verify_videomt_checkpoint(
            donors["videomt"], checkpoint_path=videomt_checkpoint.resolve()
        )
    if dinov3_bundle is not None:
        if videomt_checkpoint is None:
            raise ValueError("DINOv3 bundle audit requires the source VidEoMT checkpoint")
        external["dinov3_bundle"] = _verify_dinov3_bundle(
            donors["dinov3"],
            bundle_path=dinov3_bundle.resolve(),
            checkpoint_path=videomt_checkpoint.resolve(),
        )

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "picf-next.full-transplant-audit.v1",
        "status": "passed",
        "strict_runtime": strict_runtime,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": _sha256_bytes(canonical),
        "repository": repository_summary,
        "external": external,
        "external_checks_complete": strict_runtime,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("references/full_transplant_sources.json"),
    )
    parser.add_argument("--lingbot-repository", type=Path)
    parser.add_argument("--prepared-lingbot-checkout", type=Path)
    parser.add_argument("--lingbot-checkpoint-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--videomt-checkpoint", type=Path)
    parser.add_argument("--dinov3-bundle", type=Path)
    parser.add_argument("--strict-runtime", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    report = audit_full_transplant_contract(
        manifest,
        root=root,
        lingbot_repository=args.lingbot_repository,
        prepared_lingbot_checkout=args.prepared_lingbot_checkout,
        lingbot_checkpoint_dir=args.lingbot_checkpoint_dir,
        processor_dir=args.processor_dir,
        videomt_checkpoint=args.videomt_checkpoint,
        dinov3_bundle=args.dinov3_bundle,
        strict_runtime=args.strict_runtime,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        destination = args.json_out.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        shutil.move(temporary, destination)
    print(payload, end="")


if __name__ == "__main__":
    main()

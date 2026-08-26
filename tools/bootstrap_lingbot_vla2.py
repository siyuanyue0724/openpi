#!/usr/bin/env python3
"""Prepare the exact patched LingBot V2 host and optional pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path

try:
    from tools.verify_lingbot_vla2_patch import detect_patch_state
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from verify_lingbot_vla2_patch import detect_patch_state

LINGBOT_SOURCE_URL = "https://github.com/Robbyant/lingbot-vla-v2.git"
LINGBOT_SOURCE_COMMIT = "69729b4ef24c63ec25e750915491635f4753be1d"
LINGBOT_CHECKPOINT_ID = "robbyant/lingbot-vla-v2-6b"
LINGBOT_CHECKPOINT_REVISION = "11c703bf6a5c1f45b3b69168482da11fdbba53d7"
QWEN_PROCESSOR_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_PROCESSOR_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
LINGBOT_MODEL_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py")
LINGBOT_DATA_SOURCE = Path("lingbotvla/data/vla_data/base_dataset.py")
LINGBOT_DATA_UTILS_SOURCE = Path("lingbotvla/data/vla_data/utils.py")
LINGBOT_PATCHED_SOURCES = (
    LINGBOT_MODEL_SOURCE,
    LINGBOT_DATA_SOURCE,
    LINGBOT_DATA_UTILS_SOURCE,
)
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "depth/model.pt",
    "dino_video/config.yaml",
    "dino_video/teacher_step_10000.pth",
    *(f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)),
)
REQUIRED_PROCESSOR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
)

# Exact file contracts reported by the immutable Hugging Face revisions above.
# LFS objects use their content SHA-256; regular Git blobs were downloaded from
# the same immutable revisions and hashed as served content.
CHECKPOINT_ASSET_CONTRACT: dict[str, tuple[int, str]] = {
    "config.json": (
        31,
        "7c1fddcf59379627bfafda7b5923092f370fa28f15d2132ee305a037335dfb80",
    ),
    "depth/model.pt": (
        1_316_220_456,
        "d70c5191eab853d436763b35d40ff99d13534b4bcd43e4d02823656968159e5b",
    ),
    "dino_video/config.yaml": (
        1_388,
        "333befe36960b094f2604084802639ab4396c1fa51e51f8d1cb160e29522ac08",
    ),
    "dino_video/teacher_step_10000.pth": (
        1_401_509_792,
        "086285efd8d65bc66e96b363807c4010ee5c790b7452b765edaa23837a63705b",
    ),
    "model-00001-of-00006.safetensors": (
        4_987_151_072,
        "4afb52b06a13df8b738a156ae5c8196d3bfe6b3ca931cecbd701e44cb9674e45",
    ),
    "model-00002-of-00006.safetensors": (
        4_985_113_408,
        "ec131afa26a340db94c0dba8ec00e990be5f3d842ce6532070f0c8e26a067501",
    ),
    "model-00003-of-00006.safetensors": (
        4_928_593_216,
        "7dccb068ca66c11fa514476d64661eeead56e2e80e1c0572c9ea82aa0d9ecf27",
    ),
    "model-00004-of-00006.safetensors": (
        4_990_740_540,
        "1c2cb78066b69ae11255db851df95071e2cd69a3a4bc5020b3fd3b13b17819fb",
    ),
    "model-00005-of-00006.safetensors": (
        4_990_095_864,
        "8fe36bf1f4f617869954bdfa1ad12e16abd8b0235a5fa99c88d571d4cddf4a17",
    ),
    "model-00006-of-00006.safetensors": (
        622_195_024,
        "3cf613d592dad64b1e2a1b1bb34a6f556a7fa05eacdb9972d73e0ea4882555a0",
    ),
    "model.safetensors.index.json": (
        207_389,
        "5a753ec331c51925d064e1e76e921a7eb3dca9770d2a6a91dac5e0e4162d676a",
    ),
    "preprocessor_config.json": (
        782,
        "93585062a80db5e8ca038efc7726a3e6411d9db948472d81d63c6303993be8c5",
    ),
    "tokenizer.json": (
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": (
        5_472,
        "cf43a5bf1a49ee69ecced02f419b169e72559034dcf15af47cf775bd253830f0",
    ),
}

PROCESSOR_ASSET_CONTRACT: dict[str, tuple[int, str]] = {
    "chat_template.json": (
        5_502,
        "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    ),
    "config.json": (
        1_505,
        "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
    ),
    "merges.txt": (
        1_671_839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    "preprocessor_config.json": (
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "tokenizer.json": (
        7_032_403,
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    ),
    "tokenizer_config.json": (
        10_868,
        "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    ),
    "video_preprocessor_config.json": (
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dirty_paths(checkout: Path) -> set[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    return {line[3:] for line in output.splitlines() if line}


def prepare_source(
    *,
    checkout: Path,
    patch_path: Path,
    source_url: str = LINGBOT_SOURCE_URL,
) -> dict[str, str]:
    """Clone if needed and enter the one accepted patched source state."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    if not patch_path.is_file():
        raise ValueError(f"LingBot adapter patch is absent: {patch_path}")
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", source_url, str(checkout)])
        _run(["git", "checkout", "--detach", LINGBOT_SOURCE_COMMIT], cwd=checkout)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")

    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_SOURCE_COMMIT}")
    state = detect_patch_state(checkout, patch_path)
    dirty = _dirty_paths(checkout)
    expected_dirty = (
        set() if state == "baseline" else {str(path) for path in LINGBOT_PATCHED_SOURCES}
    )
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    if state == "baseline":
        _run(["git", "apply", str(patch_path)], cwd=checkout)
    if detect_patch_state(checkout, patch_path) != "applied":
        raise RuntimeError("LingBot adapter patch did not reach the exact applied state")
    if _dirty_paths(checkout) != {str(path) for path in LINGBOT_PATCHED_SOURCES}:
        raise RuntimeError("LingBot patch modified an unexpected source path")
    for relative_path in LINGBOT_PATCHED_SOURCES:
        source_path = checkout / relative_path
        compile(source_path.read_text(), str(source_path), "exec")
    return {
        "source_commit": actual,
        "patch_state": "applied",
        "checkout": str(checkout),
    }


def checkpoint_download_command(*, hf_command: str, checkpoint_dir: Path) -> list[str]:
    return [
        hf_command,
        "download",
        LINGBOT_CHECKPOINT_ID,
        "--revision",
        LINGBOT_CHECKPOINT_REVISION,
        "--local-dir",
        str(checkpoint_dir.resolve()),
    ]


def processor_download_command(*, hf_command: str, processor_dir: Path) -> list[str]:
    """Download only the exact Qwen config/processor assets, not base weights."""

    return [
        hf_command,
        "download",
        QWEN_PROCESSOR_ID,
        "--revision",
        QWEN_PROCESSOR_REVISION,
        "--local-dir",
        str(processor_dir.resolve()),
        "--exclude",
        "*.safetensors",
    ]


def asset_contract_manifest(
    contract: Mapping[str, tuple[int, str]],
) -> list[dict[str, object]]:
    return [
        {"path": path, "bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(contract.items())
    ]


def _validate_exact_assets(
    root: Path,
    *,
    required: tuple[str, ...],
    contract: Mapping[str, tuple[int, str]],
    label: str,
) -> list[dict[str, object]]:
    if set(required) != set(contract):
        raise RuntimeError(f"{label} required-file and digest contracts differ")
    root = root.resolve()
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"{label} is incomplete: {missing}")

    manifest = asset_contract_manifest(contract)
    expected_by_path = {item["path"]: item for item in manifest}
    for relative in sorted(required):
        expected_bytes, expected_sha256 = contract[relative]
        path = root / relative
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"{label} asset is not a regular file: {relative}")
                if metadata.st_size != expected_bytes:
                    raise ValueError(f"{label} asset size differs: {relative}")
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        except FileNotFoundError as error:
            raise ValueError(f"{label} changed while it was being validated: {relative}") from error
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{label} asset digest differs: {relative}")
        if expected_by_path[relative] != {
            "path": relative,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }:
            raise RuntimeError(f"{label} manifest construction changed")
    return manifest


def validate_checkpoint(checkpoint_dir: Path) -> dict[str, object]:
    checkpoint_dir = checkpoint_dir.resolve()
    assets = _validate_exact_assets(
        checkpoint_dir,
        required=tuple(REQUIRED_CHECKPOINT_FILES),
        contract=CHECKPOINT_ASSET_CONTRACT,
        label="LingBot checkpoint",
    )
    return {
        "checkpoint_id": LINGBOT_CHECKPOINT_ID,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "checkpoint_dir": str(checkpoint_dir),
        "required_files": len(REQUIRED_CHECKPOINT_FILES),
        "checkpoint_assets": assets,
    }


def validate_processor(processor_dir: Path) -> dict[str, object]:
    processor_dir = processor_dir.resolve()
    assets = _validate_exact_assets(
        processor_dir,
        required=REQUIRED_PROCESSOR_FILES,
        contract=PROCESSOR_ASSET_CONTRACT,
        label="Qwen processor/config snapshot",
    )
    return {
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_dir": str(processor_dir),
        "required_processor_files": len(REQUIRED_PROCESSOR_FILES),
        "processor_assets": assets,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--checkout",
        type=Path,
        default=root / "references/source_checkouts/lingbot-vla-v2",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=root / "references/patches/lingbot_vla2_action_layer_adapter.patch",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--download-processor", action="store_true")
    parser.add_argument("--hf-command", default="hf")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result: dict[str, object] = prepare_source(
        checkout=args.checkout,
        patch_path=args.patch,
    )
    if args.download_checkpoint:
        if args.checkpoint_dir is None:
            raise ValueError("--download-checkpoint requires --checkpoint-dir")
        subprocess.run(
            checkpoint_download_command(
                hf_command=args.hf_command,
                checkpoint_dir=args.checkpoint_dir,
            ),
            check=True,
        )
    if args.download_processor:
        if args.processor_dir is None:
            raise ValueError("--download-processor requires --processor-dir")
        subprocess.run(
            processor_download_command(
                hf_command=args.hf_command,
                processor_dir=args.processor_dir,
            ),
            check=True,
        )
    if args.checkpoint_dir is not None:
        result.update(validate_checkpoint(args.checkpoint_dir))
    if args.processor_dir is not None:
        result.update(validate_processor(args.processor_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

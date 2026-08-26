#!/usr/bin/env python3
"""Prepare the exact MolmoAct2 source, LeRobot trainer and checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

try:
    from tools.verify_molmoact2_lerobot_patch import detect_patch_state
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from verify_molmoact2_lerobot_patch import detect_patch_state

MOLMO_SOURCE_URL = "https://github.com/allenai/molmoact2.git"
MOLMO_SOURCE_COMMIT = "c2282820f9b188b60e66ea1636b3efd81c45cbb4"
MOLMO_LEROBOT_URL = "https://github.com/allenai/lerobot.git"
MOLMO_LEROBOT_COMMIT = "80633827176a0203064cb141383664fba024e050"
MOLMO_CHECKPOINT_ID = "allenai/MolmoAct2"
MOLMO_CHECKPOINT_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"
MOLMO_LEROBOT_LOCK_SHA256 = "f79437aeed6ac8f6fd83ff1a250136df040ef5e10657df7e280e0f409c21d8a6"
MOLMO_LEROBOT_MODEL_SOURCE = Path("src/lerobot/policies/molmoact2/modeling_molmoact2.py")
MOLMO_LEROBOT_PATCH = (
    Path(__file__).resolve().parents[1]
    / "references/patches/molmoact2_lerobot_action_layer_adapter.patch"
)

REQUIRED_CHECKPOINT_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration_molmoact2.py",
    "generation_config.json",
    "image_processing_molmoact2.py",
    "inference.py",
    "model.safetensors.index.json",
    "modeling_molmoact2.py",
    "norm_stats.json",
    "processing_molmoact2.py",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_processing_molmoact2.py",
    *(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
)
METADATA_SHA256 = {
    "chat_template.jinja": "ec66862fda57dc1aef90a21b2a0f64c18a280e2483206449bd251a73b8993101",
    "config.json": "12bb27d584b3b91ebf86b6f85327a3f934f239b5fcdf5f89298e91d7c9516112",
    "configuration_molmoact2.py": (
        "ccc1b24db7ab42b30705414b14de9601b1d82fba6f6d85393b0e9cbe1052fac4"
    ),
    "generation_config.json": ("52ed940eef55d9a7e41adc8cac3408b01e26727ef828309c962f0ae5890cd281"),
    "image_processing_molmoact2.py": (
        "bb623a013e7f897ffb938990bd02374b61dabcabd87f9f19d59111ec88ecf675"
    ),
    "inference.py": "14a328aa9c8094fe5bf6d99aa98d49d0337485ff48f828cc1944bb66318cc67d",
    "model.safetensors.index.json": (
        "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06"
    ),
    "modeling_molmoact2.py": ("3e0090b8cc045d59911c0a579815ebd17ce0423f9fdd03be887412df5c7b8b49"),
    "processing_molmoact2.py": ("61ec88c04cf70844dc98ab518d332911ec32cf5f24f052b7413ddd5d61483109"),
    "processor_config.json": ("bd5ad3ddc456b7534005e075a5c3cb0644da5e4b1aa77f1d74268a9e4c7f071a"),
    "norm_stats.json": ("0cb25a612fdd18c0615f0207d7efcc317c269efea4f05aceb28f011644b9555b"),
    "tokenizer.json": "d5395aefc9b1b7f0385d8c86a2f1775e5af81bdfbf9f2d97827ea37921d9f862",
    "tokenizer_config.json": ("0ffb85ef66d5b354f53a341ff01edc5137c324981bf37928b86bcdee2b265415"),
    "video_processing_molmoact2.py": (
        "8da721aa638da65aaa35e458920037c16b81dca344a623bb08db2b73f9ce9992"
    ),
}
WEIGHT_SHARD_SHA256 = {
    "model-00001-of-00005.safetensors": (
        "512674fc34842123fd4405fc72143bf8d48ee71165b0dc59e666132ca9447dc9"
    ),
    "model-00002-of-00005.safetensors": (
        "d3c335f3291604d25e2092e0a22441752949745050c894d18537394f8f66ac84"
    ),
    "model-00003-of-00005.safetensors": (
        "198e7aeebcd24150db62e0af096d81c7054b57491edff01179f71b1ef8f2e2fe"
    ),
    "model-00004-of-00005.safetensors": (
        "a81faa0f56099dd27590c1088e73b0a84e9fad71a322a90b89eb31dfd283d278"
    ),
    "model-00005-of-00005.safetensors": (
        "6b2eee6db4ad12f8b78fc3b0143aa4bd2510f477cdb2e736c355c41d26850afe"
    ),
}
EXPECTED_INDEX = {
    "total_parameters": 5_485_309_424,
    "total_size": 21_768_785_088,
    "tensor_keys": 1_295,
    "shards": 5,
}


def _run(command: list[str], *, cwd: Path | None = None, skip_lfs_smudge: bool = False) -> str:
    environment = os.environ.copy()
    if skip_lfs_smudge:
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_checkpoint_asset(checkpoint_dir: Path, name: str) -> Path:
    path = checkpoint_dir / name
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"MolmoAct2 checkpoint is incomplete: {name}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"MolmoAct2 checkpoint assets must be regular non-symlink files: {name}")
    return path


def _prepare_clean_checkout(*, checkout: Path, url: str, commit: str) -> str:
    checkout = checkout.absolute()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", url, str(checkout)])
        _run(["git", "fetch", url, commit], cwd=checkout)
        _run(
            ["git", "checkout", "--detach", commit],
            cwd=checkout,
            skip_lfs_smudge=True,
        )
    if not (checkout / ".git").exists():
        raise ValueError(f"source checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != commit:
        raise ValueError(f"source checkout {actual} differs from frozen {commit}")
    dirty = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=checkout)
    if dirty:
        raise ValueError(f"source checkout is dirty: {dirty.splitlines()[:10]}")
    return actual


def _dirty_paths(checkout: Path) -> set[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    return {line[3:] for line in output.splitlines() if line}


def _prepare_patched_trainer(
    *,
    checkout: Path,
    url: str,
    patch_path: Path,
) -> str:
    checkout = checkout.absolute()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", url, str(checkout)])
        _run(["git", "fetch", url, MOLMO_LEROBOT_COMMIT], cwd=checkout)
        _run(
            ["git", "checkout", "--detach", MOLMO_LEROBOT_COMMIT],
            cwd=checkout,
            skip_lfs_smudge=True,
        )
    if not (checkout / ".git").exists():
        raise ValueError(f"trainer checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != MOLMO_LEROBOT_COMMIT:
        raise ValueError(f"trainer checkout {actual} differs from frozen {MOLMO_LEROBOT_COMMIT}")
    if not patch_path.is_file():
        raise ValueError(f"MolmoAct2 LeRobot adapter patch is absent: {patch_path}")
    state = detect_patch_state(checkout, patch_path)
    expected_dirty = set() if state == "baseline" else {str(MOLMO_LEROBOT_MODEL_SOURCE)}
    dirty = _dirty_paths(checkout)
    if dirty != expected_dirty:
        raise ValueError(f"trainer checkout has unrelated changes: {sorted(dirty)}")
    if state == "baseline":
        _run(["git", "apply", str(patch_path)], cwd=checkout)
    if detect_patch_state(checkout, patch_path) != "applied":
        raise RuntimeError("MolmoAct2 LeRobot patch did not reach exact applied state")
    if _dirty_paths(checkout) != {str(MOLMO_LEROBOT_MODEL_SOURCE)}:
        raise RuntimeError("MolmoAct2 patch modified an unexpected source path")
    source = checkout / MOLMO_LEROBOT_MODEL_SOURCE
    compile(source.read_text(), str(source), "exec")
    return actual


def prepare_sources(
    *,
    source_checkout: Path,
    lerobot_checkout: Path,
    source_url: str = MOLMO_SOURCE_URL,
    lerobot_url: str = MOLMO_LEROBOT_URL,
    patch_path: Path = MOLMO_LEROBOT_PATCH,
) -> dict[str, str]:
    """Prepare both immutable Git repositories and verify the parent gitlink."""

    parent = _prepare_clean_checkout(
        checkout=source_checkout,
        url=source_url,
        commit=MOLMO_SOURCE_COMMIT,
    )
    gitlink = _run(["git", "ls-tree", "HEAD", "lerobot"], cwd=source_checkout)
    expected_gitlink = f"160000 commit {MOLMO_LEROBOT_COMMIT}\tlerobot"
    if gitlink != expected_gitlink:
        raise ValueError(f"MolmoAct2 LeRobot gitlink changed: {gitlink!r}")
    trainer = _prepare_patched_trainer(
        checkout=lerobot_checkout,
        url=lerobot_url,
        patch_path=patch_path,
    )
    lock_path = lerobot_checkout / "uv.lock"
    if not lock_path.is_file() or _sha256(lock_path) != MOLMO_LEROBOT_LOCK_SHA256:
        raise ValueError("MolmoAct2 LeRobot uv.lock differs from the frozen contract")
    return {
        "source_commit": parent,
        "source_checkout": str(source_checkout.absolute()),
        "lerobot_commit": trainer,
        "lerobot_checkout": str(lerobot_checkout.absolute()),
        "lerobot_lock_sha256": MOLMO_LEROBOT_LOCK_SHA256,
        "lerobot_patch_state": "applied",
    }


def checkpoint_download_command(*, hf_command: str, checkpoint_dir: Path) -> list[str]:
    return [
        hf_command,
        "download",
        MOLMO_CHECKPOINT_ID,
        "--revision",
        MOLMO_CHECKPOINT_REVISION,
        "--local-dir",
        str(checkpoint_dir.absolute()),
    ]


def runtime_sync_command(*, uv_command: str = "uv") -> list[str]:
    return [
        uv_command,
        "sync",
        "--frozen",
        "--extra",
        "molmoact2",
        "--extra",
        "training",
    ]


def picf_install_command(
    *,
    repo_root: Path,
    lerobot_checkout: Path,
    uv_command: str = "uv",
) -> list[str]:
    """Install this frozen source tree without re-resolving the host lock."""

    return [
        uv_command,
        "pip",
        "install",
        "--python",
        str(lerobot_checkout.absolute() / ".venv/bin/python"),
        "--no-deps",
        "scipy==1.17.1",
        "--editable",
        str(repo_root.absolute()),
    ]


def validate_checkpoint(
    checkpoint_dir: Path,
    *,
    validate_weight_shards: bool = True,
) -> dict[str, object]:
    checkpoint_dir = checkpoint_dir.absolute()
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError("MolmoAct2 checkpoint directory must be an existing non-symlink directory")
    assets = {
        name: _require_regular_checkpoint_asset(checkpoint_dir, name)
        for name in REQUIRED_CHECKPOINT_FILES
    }
    metadata_hashes = {name: _sha256(assets[name]) for name in METADATA_SHA256}
    bad_hashes = {
        name: actual for name, actual in metadata_hashes.items() if actual != METADATA_SHA256[name]
    }
    if bad_hashes:
        raise ValueError(f"MolmoAct2 checkpoint metadata hashes changed: {bad_hashes}")
    index = json.loads(assets["model.safetensors.index.json"].read_text())
    observed = {
        "total_parameters": index.get("metadata", {}).get("total_parameters"),
        "total_size": index.get("metadata", {}).get("total_size"),
        "tensor_keys": len(index.get("weight_map", {})),
        "shards": len(set(index.get("weight_map", {}).values())),
    }
    if observed != EXPECTED_INDEX:
        raise ValueError(f"MolmoAct2 checkpoint index changed: {observed}")
    weight_hashes = None
    if validate_weight_shards:
        weight_hashes = {name: _sha256(assets[name]) for name in WEIGHT_SHARD_SHA256}
        changed_weights = {
            name: actual
            for name, actual in weight_hashes.items()
            if actual != WEIGHT_SHARD_SHA256[name]
        }
        if changed_weights:
            raise ValueError(f"MolmoAct2 checkpoint weight hashes changed: {changed_weights}")
    return {
        "checkpoint_id": MOLMO_CHECKPOINT_ID,
        "checkpoint_revision": MOLMO_CHECKPOINT_REVISION,
        "checkpoint_dir": str(checkpoint_dir),
        **observed,
        "metadata_sha256": METADATA_SHA256,
        "weight_shard_sha256": weight_hashes,
    }


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=root / "references/source_checkouts/molmoact2-cloud",
    )
    parser.add_argument(
        "--lerobot-checkout",
        type=Path,
        default=root / "references/source_checkouts/molmoact2-lerobot-cloud",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--sync-runtime", action="store_true")
    parser.add_argument("--install-picf", action="store_true")
    parser.add_argument("--picf-root", type=Path, default=root)
    parser.add_argument("--hf-command", default="hf")
    parser.add_argument("--uv-command", default="uv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result: dict[str, object] = prepare_sources(
        source_checkout=args.source_checkout,
        lerobot_checkout=args.lerobot_checkout,
    )
    if args.sync_runtime:
        subprocess.run(
            runtime_sync_command(uv_command=args.uv_command),
            cwd=args.lerobot_checkout,
            check=True,
        )
    if args.install_picf:
        subprocess.run(
            picf_install_command(
                repo_root=args.picf_root,
                lerobot_checkout=args.lerobot_checkout,
                uv_command=args.uv_command,
            ),
            check=True,
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
    if args.checkpoint_dir is not None:
        result["checkpoint"] = validate_checkpoint(args.checkpoint_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

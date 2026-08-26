#!/usr/bin/env python3
"""Prepare the exact unified LingBot VLA2 source and pinned model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_SOURCE_COMMIT,
        LINGBOT_SOURCE_URL,
        checkpoint_download_command,
        processor_download_command,
        validate_checkpoint,
        validate_processor,
    )
    from tools.verify_lingbot_vla2_patch import detect_patch_state
    from tools.verify_lingbot_vla2_unified_patch import (
        DATA_PATCH_RELATIVE_PATH,
        GRAPH_PATCH_RELATIVE_PATH,
        PATCHED_SOURCES,
        expected_patched_source_hashes,
        verify_unified_patches,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_SOURCE_COMMIT,
        LINGBOT_SOURCE_URL,
        checkpoint_download_command,
        processor_download_command,
        validate_checkpoint,
        validate_processor,
    )
    from verify_lingbot_vla2_patch import detect_patch_state  # type: ignore[no-redef]
    from verify_lingbot_vla2_unified_patch import (  # type: ignore[no-redef]
        DATA_PATCH_RELATIVE_PATH,
        GRAPH_PATCH_RELATIVE_PATH,
        PATCHED_SOURCES,
        expected_patched_source_hashes,
        verify_unified_patches,
    )


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


def prepare_unified_source(
    *,
    checkout: Path,
    data_patch: Path,
    graph_patch: Path,
    source_url: str = LINGBOT_SOURCE_URL,
) -> dict[str, object]:
    """Clone if needed and enter the one accepted unified source state."""

    checkout = checkout.resolve()
    patches = (data_patch.resolve(), graph_patch.resolve())
    if any(not patch.is_file() for patch in patches):
        raise ValueError("one or more unified LingBot patches are absent")
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", source_url, str(checkout)])
        _run(["git", "checkout", "--detach", LINGBOT_SOURCE_COMMIT], cwd=checkout)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_SOURCE_COMMIT}")

    states = tuple(detect_patch_state(checkout, patch) for patch in patches)
    if len(set(states)) != 1:
        raise ValueError(f"LingBot unified patches are partially applied: {states}")
    state = states[0]
    expected_dirty = set() if state == "baseline" else {str(path) for path in PATCHED_SOURCES}
    dirty = _dirty_paths(checkout)
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    if state == "baseline":
        for patch in patches:
            _run(["git", "apply", str(patch)], cwd=checkout)
    final_states = tuple(detect_patch_state(checkout, patch) for patch in patches)
    if final_states != ("applied", "applied"):
        raise RuntimeError(f"LingBot unified patches did not apply exactly: {final_states}")
    if _dirty_paths(checkout) != {str(path) for path in PATCHED_SOURCES}:
        raise RuntimeError("LingBot unified patches modified unexpected source paths")
    for relative_path in PATCHED_SOURCES:
        source_path = checkout / relative_path
        compile(source_path.read_text(), str(source_path), "exec")
    expected_hashes = expected_patched_source_hashes(
        checkout=checkout,
        patches=patches,
    )
    actual_hashes = {
        str(relative_path): hashlib.sha256((checkout / relative_path).read_bytes()).hexdigest()
        for relative_path in PATCHED_SOURCES
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError("prepared LingBot source digests differ from the replayed patches")
    model_text = (checkout / PATCHED_SOURCES[-1]).read_text()
    if "set_unified_belief_graph" not in model_text or "action_layer_adapter" in model_text:
        raise RuntimeError("LingBot source does not expose only the unified graph hook")
    return {
        "source_commit": actual,
        "patch_states": list(final_states),
        "checkout": str(checkout),
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
        "patched_source_sha256": actual_hashes,
    }


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout",
        type=Path,
        default=root / "references/source_checkouts/lingbot-vla-v2-unified",
    )
    parser.add_argument("--data-patch", type=Path, default=root / DATA_PATCH_RELATIVE_PATH)
    parser.add_argument("--graph-patch", type=Path, default=root / GRAPH_PATCH_RELATIVE_PATH)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--download-processor", action="store_true")
    parser.add_argument("--hf-command", default="hf")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    result = prepare_unified_source(
        checkout=args.checkout,
        data_patch=args.data_patch,
        graph_patch=args.graph_patch,
    )
    replay = verify_unified_patches(root=root, checkout=args.checkout)
    if replay.get("patched_source_sha256") != result["patched_source_sha256"]:
        raise RuntimeError("prepared LingBot source differs from independent patch replay")
    result["patch_replay"] = replay
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

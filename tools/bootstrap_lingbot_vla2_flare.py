#!/usr/bin/env python3
"""Replay and validate the ADR-209 FLARE overlay on exact native LingBot source."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tools.bootstrap_lingbot_vla2_native import (
    CHECKOUT_RELATIVE_PATH,
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MODEL_SOURCE,
    PATCH_RELATIVE_PATH,
    PATCHED_MODEL_SHA256,
    PATCHED_MUON_SHA256,
    PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
    PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
    PATCHED_PARALLEL_SHA256,
    PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256,
    PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256,
    PATCHED_SOURCES,
    _dirty_paths,
    _export_commit,
    _patch_paths,
    _purge_generated_python_bytecode,
    _run,
    _sha256,
    _validate_patch,
    _validate_patched_sources,
)

FLARE_PATCH_RELATIVE_PATH = Path(
    "references/patches/lingbot_vla2_flare_generic_target.patch"
)
FLARE_PATCH_SHA256 = "97482f8559b96af0a2ff6dad4115433d37cba97246ce8d71dc328b4df6ffd8ab"
FLARE_PATCHED_MODEL_SHA256 = (
    "e5851b5f9b409befe8b0994564816d7ad480e27c653820651dfd739470d6425c"
)

_REQUIRED_FLARE_FRAGMENTS = (
    "picf_future_latent_context=None",
    "picf_future_latent_context.record_action_hidden(",
    "def set_picf_future_latent_alignment(",
    "def _append_picf_future_latent_suffix(",
    "alignment.append_future_tokens(",
    "def _native_action_hidden(",
    "native_action_suffix_count = suffix_embs.shape[1]",
    "action_query_count=(",
    "suffix_out = self._native_action_hidden(suffix_out)",
    "and self.picf_future_latent_alignment is None",
    "picf_future_latent_context.finalize(",
    "require_grad=torch.is_grad_enabled()",
)


def _validate_flare_patch(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"LingBot FLARE patch is absent: {path}")
    if _sha256(path) != FLARE_PATCH_SHA256:
        raise ValueError("LingBot FLARE patch digest differs from the approved artifact")
    text = path.read_text()
    if _patch_paths(text) != {MODEL_SOURCE}:
        raise ValueError("LingBot FLARE patch modifies a source outside the policy model")
    missing = [fragment for fragment in _REQUIRED_FLARE_FRAGMENTS if fragment not in text]
    if missing:
        raise ValueError(f"LingBot FLARE patch omits required mechanisms: {missing}")


def _validate_flare_model(path: Path) -> str:
    text = path.read_text()
    missing = [fragment for fragment in _REQUIRED_FLARE_FRAGMENTS if fragment not in text]
    if missing:
        raise ValueError(f"LingBot model omits required FLARE mechanisms: {missing}")
    if text.count("self._append_picf_future_latent_suffix(") != 2:
        raise ValueError("FLARE future tokens must execute in both train and inference paths")
    if text.count("self._native_action_hidden(") != 2:
        raise ValueError("FLARE must preserve the native action slice in train and inference")
    digest = _sha256(path)
    if digest != FLARE_PATCHED_MODEL_SHA256:
        raise ValueError("LingBot FLARE model digest differs from the approved replay")
    return digest


def verify_flare_overlay(
    *,
    root: Path,
    checkout: Path | None = None,
) -> dict[str, object]:
    """Rebuild native plus FLARE from the immutable upstream Git object."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    flare_patch = root / FLARE_PATCH_RELATIVE_PATH
    _validate_patch(native_patch)
    _validate_flare_patch(flare_patch)
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError("LingBot checkout differs from the immutable source commit")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-flare-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for overlay in (native_patch, flare_patch):
            _run(["git", "apply", "--check", str(overlay)], cwd=exported)
            _run(["git", "apply", str(overlay)], cwd=exported)
        model_sha256 = _validate_flare_model(exported / MODEL_SOURCE)
    return {
        "commit": actual,
        "flare_patch": str(FLARE_PATCH_RELATIVE_PATH),
        "flare_patch_sha256": FLARE_PATCH_SHA256,
        "model_sha256": model_sha256,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "verification_source": "immutable_commit_archive_plus_native_plus_flare",
    }


def prepare_flare_overlay(
    *,
    root: Path,
    checkout: Path,
    require_muon_collective_hotfix: bool = False,
    require_frozen_vision_offload: bool = False,
    require_trainable_vision_offload: bool = False,
) -> dict[str, object]:
    """Apply only the exact FLARE overlay to an already native-patched checkout."""

    root = root.resolve()
    checkout = checkout.resolve()
    flare_patch = root / FLARE_PATCH_RELATIVE_PATH
    _validate_flare_patch(flare_patch)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError("LingBot checkout differs from the immutable source commit")
    if require_frozen_vision_offload and require_trainable_vision_offload:
        raise ValueError("FLARE source cannot require frozen and trainable vision offload")
    if (
        require_frozen_vision_offload or require_trainable_vision_offload
    ) and not require_muon_collective_hotfix:
        raise ValueError("selective vision offload requires the approved Muon hotfix chain")
    _purge_generated_python_bytecode(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    dirty = _dirty_paths(checkout)
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    model_path = checkout / MODEL_SOURCE
    digest = _sha256(model_path)
    if digest == PATCHED_MODEL_SHA256:
        _run(["git", "apply", "--check", str(flare_patch)], cwd=checkout)
        _run(["git", "apply", str(flare_patch)], cwd=checkout)
    elif digest != FLARE_PATCHED_MODEL_SHA256:
        raise ValueError("LingBot checkout has an unknown policy-model source state")
    return validate_prepared_flare_overlay(
        root=root,
        checkout=checkout,
        require_muon_collective_hotfix=require_muon_collective_hotfix,
        require_frozen_vision_offload=require_frozen_vision_offload,
        require_trainable_vision_offload=require_trainable_vision_offload,
    )


def validate_prepared_flare_overlay(
    *,
    root: Path,
    checkout: Path,
    require_muon_collective_hotfix: bool = False,
    require_frozen_vision_offload: bool = False,
    require_trainable_vision_offload: bool = False,
) -> dict[str, object]:
    """Validate an already prepared source tree without mutating it."""

    root = root.resolve()
    checkout = checkout.resolve()
    flare_patch = root / FLARE_PATCH_RELATIVE_PATH
    _validate_flare_patch(flare_patch)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError("LingBot checkout differs from the immutable source commit")
    if require_frozen_vision_offload and require_trainable_vision_offload:
        raise ValueError("FLARE source cannot require frozen and trainable vision offload")
    if (
        require_frozen_vision_offload or require_trainable_vision_offload
    ) and not require_muon_collective_hotfix:
        raise ValueError("selective vision offload requires the approved Muon hotfix chain")
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    dirty = _dirty_paths(checkout)
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    model_path = checkout / MODEL_SOURCE
    if _sha256(model_path) != FLARE_PATCHED_MODEL_SHA256:
        raise ValueError("LingBot checkout does not contain the exact FLARE model source")
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(flare_patch)],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if reverse.returncode != 0:
        raise ValueError("LingBot checkout does not contain the exact FLARE overlay")
    model_sha256 = _validate_flare_model(model_path)
    patched_source_sha256 = _validate_patched_sources(
        checkout,
        expected_muon_sha256=(
            PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256
            if require_trainable_vision_offload
            else PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256
            if require_muon_collective_hotfix
            else PATCHED_MUON_SHA256
        ),
        require_muon_collective_hotfix=require_muon_collective_hotfix,
        require_muon_mixed_device_hotfix=require_trainable_vision_offload,
        expected_parallel_sha256=(
            PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256
            if require_trainable_vision_offload
            else PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256
            if require_frozen_vision_offload
            else PATCHED_PARALLEL_SHA256
        ),
        require_frozen_vision_offload=require_frozen_vision_offload,
        require_trainable_vision_offload=require_trainable_vision_offload,
        expected_model_sha256=FLARE_PATCHED_MODEL_SHA256,
        additional_required_model_fragments=_REQUIRED_FLARE_FRAGMENTS,
    )
    replay = verify_flare_overlay(root=root, checkout=checkout)
    if model_sha256 != replay["model_sha256"]:
        raise ValueError("prepared LingBot FLARE source differs from immutable replay")
    return {
        **replay,
        "checkout": str(checkout),
        "patch_state": "native_plus_exact_flare_generic_target",
        "patched_source_sha256": patched_source_sha256,
        "require_frozen_vision_offload": require_frozen_vision_offload,
        "require_trainable_vision_offload": require_trainable_vision_offload,
        "require_muon_collective_hotfix": require_muon_collective_hotfix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--require-muon-collective-hotfix", action="store_true")
    parser.add_argument("--require-frozen-vision-offload", action="store_true")
    parser.add_argument("--require-trainable-vision-offload", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout
    if args.prepare:
        if checkout is None:
            raise ValueError("--prepare requires --checkout")
        report = prepare_flare_overlay(
            root=args.root,
            checkout=checkout,
            require_muon_collective_hotfix=args.require_muon_collective_hotfix,
            require_frozen_vision_offload=args.require_frozen_vision_offload,
            require_trainable_vision_offload=args.require_trainable_vision_offload,
        )
    else:
        report = verify_flare_overlay(root=args.root, checkout=checkout)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

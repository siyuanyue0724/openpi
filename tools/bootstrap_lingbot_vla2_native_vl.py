#!/usr/bin/env python3
"""Verify and prepare the ADR-124 native Qwen grounding overlay."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

try:
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        MODEL_SOURCE,
        PATCH_RELATIVE_PATH,
        PATCHED_SOURCES,
        _dirty_paths,
        _export_commit,
        _purge_generated_python_bytecode,
        _run,
        _validate_patched_sources,
        detect_native_patch_state,
        prepare_native_source,
        verify_native_patch,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        MODEL_SOURCE,
        PATCH_RELATIVE_PATH,
        PATCHED_SOURCES,
        _dirty_paths,
        _export_commit,
        _purge_generated_python_bytecode,
        _run,
        _validate_patched_sources,
        detect_native_patch_state,
        prepare_native_source,
        verify_native_patch,
    )

NATIVE_VL_PATCH_RELATIVE_PATH = Path("references/patches/lingbot_vla2_native_vl_grounding.patch")
NATIVE_VL_PATCH_SHA256 = "0cc8667d15082432a5095b4dd0bd892e94cad682f17f654fc1dd19289ba5c166"
NATIVE_VL_PATCHED_MODEL_SHA256 = "83edf1a1205d5b6297d3e057c2a986f9aa3ed3c7fa66578320434430f11f8fe3"

_REQUIRED_NATIVE_VL_FRAGMENTS = (
    "def picf_native_vl_forward(",
    "native vision-language loss requires the tied Qwen LM head",
    "self.model.qwenvl_with_expert.qwenvl(",
    "labels=labels",
    "pixel_values=pixel_values",
    "image_grid_thw=image_grid_thw",
    "use_cache=False",
    "return output.loss",
)
_FORBIDDEN_NATIVE_VL_FRAGMENTS = (
    "semantic_scorer",
    "teacher_model",
    "student_model",
    "lifecycle",
    "owner_index",
    "owner_supervised",
    "target_identity",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patch_paths(patch_text: str) -> set[Path]:
    return {
        Path(line.removeprefix("+++ b/"))
        for line in patch_text.splitlines()
        if line.startswith("+++ b/")
    }


def _validate_native_vl_patch(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"native VL overlay is absent: {path}")
    if _sha256(path) != NATIVE_VL_PATCH_SHA256:
        raise ValueError("native VL overlay digest differs from the approved artifact")
    text = path.read_text()
    if _patch_paths(text) != {MODEL_SOURCE}:
        raise ValueError("native VL overlay may modify only the LingBot V2 model source")
    missing = [fragment for fragment in _REQUIRED_NATIVE_VL_FRAGMENTS if fragment not in text]
    if missing:
        raise ValueError(f"native VL overlay omits required fragments: {missing}")
    forbidden = [fragment for fragment in _FORBIDDEN_NATIVE_VL_FRAGMENTS if fragment in text]
    if forbidden:
        raise ValueError(f"native VL overlay contains forbidden private modules: {forbidden}")


def _validate_native_vl_model(path: Path) -> str:
    source = path.read_text()
    compile(source, str(path), "exec")
    if source.count("def picf_native_vl_forward(") != 1:
        raise ValueError("native VL root method must have exactly one implementation")
    missing = [fragment for fragment in _REQUIRED_NATIVE_VL_FRAGMENTS if fragment not in source]
    if missing:
        raise ValueError(f"patched LingBot source omits native VL fragments: {missing}")
    forbidden = [fragment for fragment in _FORBIDDEN_NATIVE_VL_FRAGMENTS if fragment in source]
    if forbidden:
        raise ValueError(f"patched LingBot source contains forbidden private modules: {forbidden}")
    digest = _sha256(path)
    if digest != NATIVE_VL_PATCHED_MODEL_SHA256:
        raise ValueError("native VL model source digest differs from the approved replay")
    return digest


def detect_native_vl_patch_state(checkout: Path, patch_path: Path) -> str:
    """Return whether the second overlay is absent or exactly applied."""

    forward = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    reverse = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    if forward.returncode == 0 and reverse.returncode != 0:
        return "baseline"
    if reverse.returncode == 0 and forward.returncode != 0:
        return "applied"
    raise ValueError("LingBot checkout is neither native-VL baseline nor applied state")


def verify_native_vl_patch(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay both immutable overlays from the pinned official Git object."""

    root = root.resolve()
    patch_path = root / NATIVE_VL_PATCH_RELATIVE_PATH
    _validate_native_vl_patch(patch_path)
    verify_native_patch(root=root, check_apply=False)
    result: dict[str, object] = {
        "apply_checked": False,
        "base_patch": str(PATCH_RELATIVE_PATH),
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_vl_patch": str(NATIVE_VL_PATCH_RELATIVE_PATH),
        "native_vl_patch_sha256": NATIVE_VL_PATCH_SHA256,
        "patched_source": str(MODEL_SOURCE),
    }
    if not check_apply:
        return result
    source_checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (source_checkout / ".git").exists():
        raise ValueError(f"pinned LingBot checkout is absent: {source_checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=source_checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError("LingBot checkout differs from the pinned source commit")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-native-vl-") as temporary:
        exported = Path(temporary)
        _export_commit(source_checkout, exported)
        _run(["git", "apply", str(root / PATCH_RELATIVE_PATH)], cwd=exported)
        _validate_patched_sources(exported)
        _run(["git", "apply", str(patch_path)], cwd=exported)
        model_digest = _validate_native_vl_model(exported / MODEL_SOURCE)
    result.update(
        {
            "apply_checked": True,
            "patched_model_sha256": model_digest,
            "verification_source": "immutable_commit_archive",
        }
    )
    return result


def prepare_native_vl_source(
    *,
    root: Path,
    checkout: Path,
    source_url: str,
) -> dict[str, object]:
    """Prepare the exact base patch plus ADR-124 overlay idempotently."""

    root = root.resolve()
    checkout = checkout.resolve()
    base_patch = root / PATCH_RELATIVE_PATH
    overlay = root / NATIVE_VL_PATCH_RELATIVE_PATH
    _validate_native_vl_patch(overlay)
    if not checkout.exists():
        prepare_native_source(
            checkout=checkout,
            patch_path=base_patch,
            source_url=source_url,
        )
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    if _run(["git", "rev-parse", "HEAD"], cwd=checkout) != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError("LingBot checkout differs from the pinned source commit")
    _purge_generated_python_bytecode(checkout)
    overlay_state = detect_native_vl_patch_state(checkout, overlay)
    if overlay_state == "baseline":
        if detect_native_patch_state(checkout, base_patch) != "applied":
            raise ValueError("native VL overlay requires the exact base PICF patch")
        verify_native_patch(root=root, checkout=checkout, check_apply=True)
        _run(["git", "apply", str(overlay)], cwd=checkout)
    if detect_native_vl_patch_state(checkout, overlay) != "applied":
        raise RuntimeError("native VL overlay did not reach its exact applied state")
    if _dirty_paths(checkout) != {str(path) for path in PATCHED_SOURCES}:
        raise ValueError("native VL checkout contains unrelated source changes")
    model_digest = _validate_native_vl_model(checkout / MODEL_SOURCE)
    verify_native_vl_patch(root=root, checkout=checkout, check_apply=True)
    return {
        "checkout": str(checkout),
        "native_vl_patch_sha256": NATIVE_VL_PATCH_SHA256,
        "native_vl_patch_state": "applied",
        "patched_model_sha256": model_digest,
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
    }


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--checkout", type=Path, default=root / CHECKOUT_RELATIVE_PATH)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--source-url",
        default="https://github.com/Robbyant/lingbot-vla-v2.git",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = (
        prepare_native_vl_source(
            root=args.root,
            checkout=args.checkout,
            source_url=args.source_url,
        )
        if args.prepare
        else verify_native_vl_patch(
            root=args.root,
            checkout=args.checkout,
            check_apply=True,
        )
    )
    import json

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

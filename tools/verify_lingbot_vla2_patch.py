#!/usr/bin/env python3
"""Verify the pinned LingBot V2 adapter patch without modifying its checkout."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

LINGBOT_COMMIT = "69729b4ef24c63ec25e750915491635f4753be1d"
PATCH_RELATIVE_PATH = Path("references/patches/lingbot_vla2_action_layer_adapter.patch")
CHECKOUT_RELATIVE_PATH = Path("references/source_checkouts/lingbot-vla-v2")

_REQUIRED_PATCH_FRAGMENTS = (
    "class QwenvlWithExpertV2Model",
    'if self.config.attention_implementation == "flex_cached"',
    "vlm_config._attn_implementation = hf_attention",
    "self.config.qwen_expert_config._attn_implementation = hf_attention",
    "self.action_layer_adapter = None",
    "def set_action_layer_adapter",
    "out_emb = self.action_layer_adapter(",
    "action_layer_context=action_layer_context",
    "class FlowMatchingV2",
    "and action_layer_context is None",
    "class LingbotVlaV2Policy",
    "from lerobot.datasets.utils import hf_transform_to_torch, load_nested_dataset",
    "hf_dataset = load_nested_dataset(",
    "episodes=self.episodes",
    "def _as_feature_mapping(value)",
    "joint_info = _as_feature_mapping(s)",
    "for k, v in _as_feature_mapping(d).items()",
)

_REQUIRED_UPSTREAM_FRAGMENTS = (
    "class QwenvlWithExpertV2Model",
    "class FlowMatchingV2",
    "def sample_actions(",
    "def predict_velocity(",
    "class LingbotVlaV2Policy",
    "def load_hf_dataset(self, features=None)",
)


def detect_patch_state(checkout: Path, patch_path: Path) -> str:
    """Return baseline/applied only for the exact reversible patch states."""

    forward = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "apply",
            "--reverse",
            "--check",
            str(patch_path),
        ],
        capture_output=True,
        text=True,
    )
    if forward.returncode == 0 and reverse.returncode != 0:
        return "baseline"
    if reverse.returncode == 0 and forward.returncode != 0:
        return "applied"
    raise ValueError("LingBot checkout is neither exact baseline nor exact patched state")


def _export_commit(checkout: Path, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", LINGBOT_COMMIT],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def verify_patch(
    *,
    root: Path,
    check_apply: bool,
    required_state: str = "baseline",
) -> dict[str, object]:
    if required_state not in {"baseline", "applied", "either"}:
        raise ValueError(f"unsupported patch state {required_state!r}")
    patch_path = root / PATCH_RELATIVE_PATH
    checkout = root / CHECKOUT_RELATIVE_PATH
    if not patch_path.is_file():
        raise ValueError(f"LingBot patch is absent: {patch_path}")
    patch_text = patch_path.read_text()
    missing = [fragment for fragment in _REQUIRED_PATCH_FRAGMENTS if fragment not in patch_text]
    if missing:
        raise ValueError(f"LingBot patch omits required training/inference fragments: {missing}")

    result: dict[str, object] = {
        "commit": LINGBOT_COMMIT,
        "patch": str(PATCH_RELATIVE_PATH),
        "apply_checked": False,
        "patch_state": "unchecked",
    }
    if check_apply:
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != LINGBOT_COMMIT:
            raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_COMMIT}")
        if required_state == "either":
            with tempfile.TemporaryDirectory(prefix="picf-lingbot-sidecar-") as temporary:
                source_root = Path(temporary)
                _export_commit(checkout, source_root)
                source_paths = (
                    source_root / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py",
                    source_root / "lingbotvla/data/vla_data/base_dataset.py",
                    source_root / "lingbotvla/data/vla_data/utils.py",
                )
                source_text = "\n".join(path.read_text() for path in source_paths)
                missing_upstream = [
                    fragment
                    for fragment in _REQUIRED_UPSTREAM_FRAGMENTS
                    if fragment not in source_text
                ]
                if missing_upstream:
                    raise ValueError(
                        f"pinned LingBot V2 source omits required symbols: {missing_upstream}"
                    )
                subprocess.run(
                    ["git", "apply", "--check", str(patch_path)],
                    cwd=source_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            patch_state = "baseline"
            result["verification_source"] = "immutable_commit_archive"
        else:
            source_paths = (
                checkout / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py",
                checkout / "lingbotvla/data/vla_data/base_dataset.py",
                checkout / "lingbotvla/data/vla_data/utils.py",
            )
            source_text = "\n".join(path.read_text() for path in source_paths)
            missing_upstream = [
                fragment for fragment in _REQUIRED_UPSTREAM_FRAGMENTS if fragment not in source_text
            ]
            if missing_upstream:
                raise ValueError(
                    f"pinned LingBot V2 source omits required symbols: {missing_upstream}"
                )
            patch_state = detect_patch_state(checkout, patch_path)
            if patch_state != required_state:
                raise ValueError(f"LingBot patch state is {patch_state}, required {required_state}")
            result["verification_source"] = "working_checkout"
        result["apply_checked"] = True
        result["patch_state"] = patch_state
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-apply", action="store_true")
    parser.add_argument(
        "--require-state",
        choices=("baseline", "applied", "either"),
        default="baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            verify_patch(
                root=root,
                check_apply=args.check_apply,
                required_state=args.require_state,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

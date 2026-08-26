#!/usr/bin/env python3
"""Verify the pinned MolmoAct2 LeRobot action-layer adapter patch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MOLMO_LEROBOT_COMMIT = "80633827176a0203064cb141383664fba024e050"
PATCH_RELATIVE_PATH = Path("references/patches/molmoact2_lerobot_action_layer_adapter.patch")
CHECKOUT_RELATIVE_PATH = Path("references/source_checkouts/molmoact2-lerobot")
MODEL_RELATIVE_PATH = Path("src/lerobot/policies/molmoact2/modeling_molmoact2.py")

_REQUIRED_PATCH_FRAGMENTS = (
    "class MolmoAct2Policy",
    "self.action_layer_adapter: torch.nn.Module | None = None",
    "def set_action_layer_adapter",
    "action_layer_adapter_params: list[Tensor] = []",
    'if name.startswith("action_layer_adapter.")',
    "action_layer_context: Any | None = None",
    "flow_timesteps: Tensor | None = None",
    "flow_noise: Tensor | None = None",
    "timesteps=flow_timesteps",
    "noise=flow_noise",
    "self.action_layer_adapter.apply_training_layer(",
    "self.action_layer_adapter.forward_with_context(",
    'or "inputs_embeds" in model_inputs',
    "action_condition_input_ids: Tensor | None = None",
    "precomputed MolmoAct2 inputs_embeds require action_condition_input_ids",
    "resolved_action_condition_input_ids",
    "trajectory_dtype = action_expert.action_embed.weight.dtype",
    "dtype=trajectory_dtype",
    "device=device, dtype=torch.float32",
    "PICF action-layer context and condition IDs are only defined for",
)


def detect_patch_state(checkout: Path, patch_path: Path) -> str:
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
    raise ValueError("MolmoAct2 checkout is neither exact baseline nor exact patched state")


def verify_patch(
    *,
    root: Path,
    check_apply: bool,
    required_state: str = "either",
    checkout: Path | None = None,
    patch_path: Path | None = None,
) -> dict[str, object]:
    if required_state not in {"baseline", "applied", "either"}:
        raise ValueError(f"unsupported patch state {required_state!r}")
    patch_path = patch_path or root / PATCH_RELATIVE_PATH
    checkout = checkout or root / CHECKOUT_RELATIVE_PATH
    if not patch_path.is_file():
        raise ValueError(f"MolmoAct2 LeRobot patch is absent: {patch_path}")
    patch_text = patch_path.read_text()
    missing = [fragment for fragment in _REQUIRED_PATCH_FRAGMENTS if fragment not in patch_text]
    if missing:
        raise ValueError(f"MolmoAct2 patch omits required fragments: {missing}")

    result: dict[str, object] = {
        "commit": MOLMO_LEROBOT_COMMIT,
        "checkout": str(checkout),
        "patch": str(patch_path),
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
        if actual != MOLMO_LEROBOT_COMMIT:
            raise ValueError(f"MolmoAct2 LeRobot checkout {actual} differs from pinned commit")
        state = detect_patch_state(checkout, patch_path)
        if required_state != "either" and state != required_state:
            raise ValueError(f"MolmoAct2 patch state is {state}, required {required_state}")
        compile(
            (checkout / MODEL_RELATIVE_PATH).read_text(),
            str(checkout / MODEL_RELATIVE_PATH),
            "exec",
        )
        result["apply_checked"] = True
        result["patch_state"] = state
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-apply", action="store_true")
    parser.add_argument(
        "--require-state",
        choices=("baseline", "applied", "either"),
        default="either",
    )
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkout = args.checkout
    if checkout is not None and not checkout.is_absolute():
        checkout = root / checkout
    patch_path = args.patch
    if patch_path is not None and not patch_path.is_absolute():
        patch_path = root / patch_path
    print(
        json.dumps(
            verify_patch(
                root=root,
                check_apply=args.check_apply,
                required_state=args.require_state,
                checkout=checkout,
                patch_path=patch_path,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

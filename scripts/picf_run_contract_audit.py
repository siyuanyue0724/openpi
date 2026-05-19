#!/usr/bin/env python3
"""Classify a PICF training command against the experiment contract.

This is read-only and intentionally conservative.  It prevents conflating
anchor-only probes with comprehensive frozen-policy validation or formal
co-training runs.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def _tokens_from_text(text: str) -> list[str]:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        if stripped_line.endswith("\\"):
            stripped_line = stripped_line[:-1].strip()
        cleaned_lines.append(stripped_line)
    stripped = " ".join(cleaned_lines)
    try:
        return [token.strip() for token in shlex.split(stripped)]
    except ValueError:
        return [token.strip() for token in stripped.split()]


def _arg_value(tokens: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for idx, token in enumerate(tokens):
        if token == name and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return default


def _flag(tokens: list[str], name: str) -> bool:
    return name in tokens


def _float_arg(tokens: list[str], name: str, default: float = 0.0) -> float:
    value = _arg_value(tokens, name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def classify_command(text: str) -> dict[str, Any]:
    tokens = _tokens_from_text(text)
    scope = str(_arg_value(tokens, "--picf-trainable-scope", "all")).replace("-", "_")
    perception = str(_arg_value(tokens, "--perception-finetune-mode", "auto")).replace("-", "_")
    semantic_trainable = _flag(tokens, "--semantic-trainable")
    action_lambdas = {
        "pos": _float_arg(tokens, "--lambda-action-pos", 2.0),
        "rot": _float_arg(tokens, "--lambda-action-rot", 2.0),
        "gripper": _float_arg(tokens, "--lambda-action-gripper", 2.0),
    }
    action_enabled = any(value != 0.0 for value in action_lambdas.values())
    sidecar_root = _arg_value(tokens, "--mvtrack-sidecar-root")
    has_sidecar = sidecar_root is not None and str(sidecar_root).lower() not in {"", "none", "null"}
    has_sam_like_sidecar = bool(sidecar_root) and any(
        needle in str(sidecar_root).lower()
        for needle in ("sam_proposal", "sam-proposal", "sam_proposals", "blind_sam", "segment-anything")
    )
    object_pull = _float_arg(tokens, "--lambda-anchor-object-pull", 0.0)
    slot_quality = _float_arg(tokens, "--lambda-slot-quality", 0.0)
    aqr_denoising = _float_arg(tokens, "--lambda-aqr-denoising", 0.0)
    slot_jepa = _float_arg(tokens, "--lambda-slot-jepa", 0.0)
    support_pred = _float_arg(tokens, "--lambda-support-pred", 0.0)
    binding_consistency = _float_arg(tokens, "--lambda-binding-consistency", 0.0)
    object_explanation = {
        "feature": _float_arg(tokens, "--lambda-object-explanation-feature", 0.0),
        "point": _float_arg(tokens, "--lambda-object-explanation-point", 0.0),
        "contact": _float_arg(tokens, "--lambda-object-explanation-contact", 0.0),
        "duplicate": _float_arg(tokens, "--lambda-object-explanation-duplicate", 0.0),
        "background": _float_arg(tokens, "--lambda-object-explanation-background", 0.0),
    }

    if scope == "anchor_only":
        run_class = "anchor_capability_probe"
        reason = (
            "picf_trainable_scope=anchor_only freezes semantic/action/control and trains only "
            "the PICF anchor/router/posterior support subset."
        )
    elif perception == "frozen" and not semantic_trainable and not action_enabled:
        run_class = "slot_comprehensive_frozen_policy_validation"
        reason = (
            "scope=all with frozen perception, frozen PaliGemma, and zero action losses leaves "
            "the broader PICF slot/router/posterior/OEML path trainable without policy pressure."
        )
    elif perception == "frozen" and semantic_trainable and action_enabled:
        run_class = "formal_frozen_pretrain_cotrain"
        reason = (
            "pretrained perception backbones are frozen while PaliGemma/action/PICF are co-trained."
        )
    else:
        run_class = "custom_or_ambiguous"
        reason = "The command does not match a documented PICF v2.2 experiment class."

    warnings: list[str] = []
    if has_sam_like_sidecar:
        warnings.append("sidecar root looks like legacy blind SAM; this is rejected unless explicitly reproducing legacy.")
    if run_class == "slot_comprehensive_frozen_policy_validation" and scope != "all":
        warnings.append("comprehensive frozen-policy validation requires picf_trainable_scope=all.")
    if run_class == "anchor_capability_probe" and (aqr_denoising or slot_jepa or support_pred or binding_consistency):
        warnings.append("anchor capability probes should not enable predictive/denoising identity losses.")
    if has_sidecar and object_pull == 0.0 and slot_quality == 0.0 and not any(object_explanation.values()):
        warnings.append("sidecar evidence is present but no object pull, slot-quality, or OEML loss is enabled.")

    return {
        "run_class": run_class,
        "reason": reason,
        "picf_trainable_scope": scope,
        "perception_finetune_mode": perception,
        "semantic_trainable": semantic_trainable,
        "action_enabled": action_enabled,
        "action_lambdas": action_lambdas,
        "has_sidecar": has_sidecar,
        "sidecar_root": sidecar_root,
        "has_sam_like_sidecar": has_sam_like_sidecar,
        "loss_switches": {
            "lambda_anchor_object_pull": object_pull,
            "lambda_slot_quality": slot_quality,
            "lambda_aqr_denoising": aqr_denoising,
            "lambda_slot_jepa": slot_jepa,
            "lambda_support_pred": support_pred,
            "lambda_binding_consistency": binding_consistency,
            "lambda_object_explanation": object_explanation,
        },
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", help="PICF training command string to classify.")
    parser.add_argument("--command-file", type=Path, help="File containing the command string.")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    if bool(args.command) == bool(args.command_file):
        raise SystemExit("provide exactly one of --command or --command-file")
    text = args.command if args.command is not None else args.command_file.read_text(encoding="utf-8", errors="ignore")
    result = classify_command(text)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_warning and result["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

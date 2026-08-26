"""Shared trainable-scope contracts for LingBot PICF and matched baselines."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TRAINABLE_SCOPE_FULL_HOST = "full-host"
TRAINABLE_SCOPE_FROZEN_VISION_HOST = "frozen-vision-host"
TRAINABLE_SCOPES = (
    TRAINABLE_SCOPE_FULL_HOST,
    TRAINABLE_SCOPE_FROZEN_VISION_HOST,
)


def lingbot_trainable_scope_receipt(policy: Any, *, scope: str) -> dict[str, object]:
    """Prove a LingBot visual freeze without removing its forward computation."""

    if scope not in TRAINABLE_SCOPES:
        raise ValueError("LingBot trainable scope is unsupported")
    named_parameters = tuple(policy.named_parameters())
    if not named_parameters:
        raise RuntimeError("LingBot trainable-scope audit found no parameters")
    visual_prefix = "model.qwenvl_with_expert.qwenvl.model.visual."
    visual_parameters = tuple(
        (name, parameter) for name, parameter in named_parameters if name.startswith(visual_prefix)
    )
    if not visual_parameters:
        raise RuntimeError("LingBot trainable-scope audit found no visual parameters")

    expected_visual_trainable = scope == TRAINABLE_SCOPE_FULL_HOST
    mismatched = tuple(
        name
        for name, parameter in visual_parameters
        if bool(parameter.requires_grad) != expected_visual_trainable
    )
    if mismatched:
        raise RuntimeError(
            f"LingBot visual parameters violate the declared trainable scope: {mismatched[:3]}"
        )

    trainable = tuple(
        (name, parameter) for name, parameter in named_parameters if parameter.requires_grad
    )
    identity = tuple(
        (name, int(parameter.numel()), bool(parameter.requires_grad))
        for name, parameter in named_parameters
    )
    visual_numel = sum(int(parameter.numel()) for _, parameter in visual_parameters)
    trainable_visual_numel = sum(
        int(parameter.numel()) for _, parameter in visual_parameters if parameter.requires_grad
    )
    scope_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "picf-next.lingbot-trainable-scope/v1",
        "scope": scope,
        "forward_model_complete": True,
        "visual_forward_enabled": True,
        "visual_parameter_count": len(visual_parameters),
        "visual_numel": visual_numel,
        "trainable_visual_numel": trainable_visual_numel,
        "parameter_count": len(named_parameters),
        "total_numel": sum(int(parameter.numel()) for _, parameter in named_parameters),
        "trainable_parameter_count": len(trainable),
        "trainable_numel": sum(int(parameter.numel()) for _, parameter in trainable),
        "scope_sha256": scope_sha256,
    }

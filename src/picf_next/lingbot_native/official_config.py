"""Typed boundary adapter for LingBot's released data-config parser."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

_FEATURE_DECLARATION_FIELDS = ("joints", "norm_type")


def _mapping_declaration(value: object, *, field: str, index: int) -> dict[str, Any]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"LingBot {field}[{index}] is not a literal mapping") from error
    if not isinstance(parsed, Mapping) or len(parsed) != 1:
        raise TypeError(f"LingBot {field}[{index}] must contain exactly one mapping entry")
    name, declaration = next(iter(parsed.items()))
    if not isinstance(name, str) or not name:
        raise TypeError(f"LingBot {field}[{index}] has an invalid feature name")
    if field == "joints":
        if isinstance(declaration, bool) or not isinstance(declaration, int) or declaration < 0:
            raise TypeError(f"LingBot joints[{index}] width must be a non-negative integer")
    elif not isinstance(declaration, str) or not declaration:
        raise TypeError(f"LingBot norm_type[{index}] must name one normalization mode")
    return {name: declaration}


def official_lingbot_data_config(config: Mapping[str, Any]) -> SimpleNamespace:
    """Convert structured config into the exact released ``FeatureTransform`` ABI.

    LingBot's pinned ADR-74 source calls ``ast.literal_eval`` on every joint and
    normalization declaration, even though its published YAML represents those
    declarations as mappings. PICF keeps its own config structured and performs
    this serialization once at the host boundary instead of patching model code.
    """

    if not isinstance(config, Mapping):
        raise TypeError("LingBot data config must be a mapping")
    converted = deepcopy(dict(config))
    declared_names: dict[str, tuple[str, ...]] = {}
    for field in _FEATURE_DECLARATION_FIELDS:
        values = converted.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise TypeError(f"LingBot data config {field} must be a non-empty sequence")
        mappings = tuple(
            _mapping_declaration(value, field=field, index=index)
            for index, value in enumerate(values)
        )
        names = tuple(next(iter(mapping)) for mapping in mappings)
        if len(set(names)) != len(names):
            raise ValueError(f"LingBot data config {field} contains duplicate features")
        declared_names[field] = names
        converted[field] = [repr(mapping) for mapping in mappings]

    cameras = converted.get("cameras")
    if (
        not isinstance(cameras, Sequence)
        or isinstance(cameras, (str, bytes))
        or not cameras
        or any(not isinstance(camera, str) or not camera for camera in cameras)
    ):
        raise TypeError("LingBot data config cameras must be non-empty feature names")
    unknown_norms = set(declared_names["norm_type"]) - set(declared_names["joints"])
    if unknown_norms:
        raise ValueError(
            f"LingBot normalization references undeclared joints: {sorted(unknown_norms)}"
        )
    return SimpleNamespace(**converted)

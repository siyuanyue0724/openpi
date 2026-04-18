from __future__ import annotations

import contextlib
from typing import Any

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
except Exception:  # pragma: no cover - optional distributed dependency
    _FSDP = None


def _iter_fsdp_method_roots(module: Any) -> list[Any]:
    if _FSDP is None:
        return []
    if isinstance(module, _FSDP):
        return [module]
    if not hasattr(module, "children"):
        return []

    roots: list[Any] = []

    def _collect(current: Any, *, has_fsdp_ancestor: bool) -> None:
        for child in current.children():
            is_fsdp = isinstance(child, _FSDP)
            if is_fsdp and not has_fsdp_ancestor:
                roots.append(child)
                continue
            _collect(child, has_fsdp_ancestor=has_fsdp_ancestor or is_fsdp)

    _collect(module, has_fsdp_ancestor=False)
    return roots


def call_fsdp_method(module: Any, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
    fsdp_roots = _iter_fsdp_method_roots(module)
    if not fsdp_roots:
        return getattr(module, method_name)(*args, **kwargs)

    target = module.module if _FSDP is not None and isinstance(module, _FSDP) else module
    with contextlib.ExitStack() as stack:
        for wrapped in fsdp_roots:
            stack.enter_context(_FSDP.summon_full_params(wrapped, recurse=True, writeback=False))
        return getattr(target, method_name)(*args, **kwargs)

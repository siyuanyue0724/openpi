from __future__ import annotations

from typing import Any

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
except Exception:  # pragma: no cover - optional distributed dependency
    _FSDP = None


def call_fsdp_method(module: Any, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
    if _FSDP is not None and isinstance(module, _FSDP):
        with _FSDP.summon_full_params(module, recurse=True, writeback=False):
            return getattr(module.module, method_name)(*args, **kwargs)
    return getattr(module, method_name)(*args, **kwargs)

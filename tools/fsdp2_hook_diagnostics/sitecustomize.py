"""Optional FSDP2 hook-state diagnostics for released-weight cloud probes.

Python imports ``sitecustomize`` during interpreter startup when this directory
is prepended to ``PYTHONPATH``. The instrumentation remains inert unless
``PICF_FSDP2_HOOK_DIAGNOSTICS=1`` is set.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stderr.flush()


if os.environ.get("PICF_FSDP2_HOOK_DIAGNOSTICS") == "1":
    import torch
    import torch.distributed.fsdp as fsdp
    from torch import nn
    from torch.distributed.tensor import DTensor

    _original_fully_shard = fsdp.fully_shard
    _ordinal = 0

    def _diagnostic_fully_shard(module, *args, **kwargs):
        global _ordinal
        result = _original_fully_shard(module, *args, **kwargs)
        if isinstance(module, nn.Linear):
            ordinal = _ordinal
            _ordinal += 1

            def _inspect_after_fsdp_pre_hook(current, unused_args):
                weight = current.weight
                if not isinstance(weight, DTensor):
                    return
                state = _original_fully_shard.state(current)
                parameter_group = state._fsdp_param_group
                _emit(
                    {
                        "event": "fsdp2_linear_remained_dtensor_after_pre_hook",
                        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
                        "module_class": current.__class__.__name__,
                        "module_state": str(state._training_state),
                        "ordinal": ordinal,
                        "parameter_group_state": (
                            str(parameter_group._training_state)
                            if parameter_group is not None
                            else None
                        ),
                        "weight_global_shape": list(weight.shape),
                        "weight_local_shape": list(weight.to_local().shape),
                    }
                )

            module.register_forward_pre_hook(_inspect_after_fsdp_pre_hook)
        return result

    fsdp.fully_shard = _diagnostic_fully_shard
    torch.distributed.fsdp.fully_shard = _diagnostic_fully_shard
    _emit(
        {
            "event": "fsdp2_hook_diagnostics_enabled",
            "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
            "pid": os.getpid(),
        }
    )

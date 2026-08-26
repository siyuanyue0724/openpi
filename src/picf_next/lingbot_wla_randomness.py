from __future__ import annotations

import hashlib
from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from typing import Any

import torch


def paired_wla_seed(model_inputs: Mapping[str, Any]) -> int:
    """Derive common-random-number entropy from the frozen CALVIN flow draw.

    The tensors are not substituted into WLA.  They only identify the frozen
    sample/rank draw so the untouched WLA action and world objectives sample
    the same distributions in matched experimental arms.
    """

    digest = hashlib.sha256(b"picf-next.adr224-wla-common-random-numbers/v1\0")
    for name in ("noise", "time"):
        value = model_inputs.get(name)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"paired WLA randomness requires floating tensor {name!r}")
        detached = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tuple(detached.shape)).encode("ascii") + b"\0")
        digest.update(detached.numpy().tobytes(order="C"))
    # torch.manual_seed accepts signed/unsigned 64-bit values, but retaining
    # 63 bits avoids backend-specific signed conversion at the boundary.
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def paired_wla_inference_seed(noise: torch.Tensor) -> int:
    """Bind exact WLA inference randomness to a frozen paired-evaluation draw."""

    if not isinstance(noise, torch.Tensor) or not noise.is_floating_point():
        raise TypeError("paired WLA inference requires a floating noise tensor")
    detached = noise.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if detached.ndim != 3 or not torch.isfinite(detached).all():
        raise ValueError("paired WLA inference noise must be finite [batch,horizon,width]")
    digest = hashlib.sha256(b"picf-next.adr224-wla-inference-common-random-numbers/v1\0")
    digest.update(str(tuple(detached.shape)).encode("ascii") + b"\0")
    digest.update(detached.numpy().tobytes(order="C"))
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def paired_wla_rng(
    model_inputs: Mapping[str, Any],
    *,
    device: torch.device,
) -> Iterator[int]:
    """Run exact upstream stochastic code under a restorable paired RNG state."""

    if device.type != "cuda" or device.index is None:
        raise ValueError("complete WLA training requires one indexed CUDA device")
    seed = paired_wla_seed(model_inputs)
    with torch.random.fork_rng(devices=[device.index], enabled=True):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        yield seed


@contextmanager
def paired_wla_inference_rng(
    noise: torch.Tensor,
    *,
    device: torch.device,
) -> Iterator[int]:
    """Seed WLA's untouched internal sampler while restoring caller RNG state."""

    if device.type != "cuda" or device.index is None:
        raise ValueError("complete WLA inference requires one indexed CUDA device")
    seed = paired_wla_inference_seed(noise)
    with torch.random.fork_rng(devices=[device.index], enabled=True):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        yield seed

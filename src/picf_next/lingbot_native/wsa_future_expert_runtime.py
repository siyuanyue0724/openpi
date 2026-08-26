from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from picf_next.lingbot_native.wsa_full_depth_adaptation import WSA_COMMIT

_PACKAGE_PATHS = (
    ("lerobot", "lerobot"),
    ("lerobot.policies", "lerobot/policies"),
    ("lerobot.policies.WSA_Base", "lerobot/policies/WSA_Base"),
    ("lerobot.policies.WSA_Large", "lerobot/policies/WSA_Large"),
    ("lerobot.policies.WSA_Large.core", "lerobot/policies/WSA_Large/core"),
    ("lerobot.policies.WSA_Large.core.models", "lerobot/policies/WSA_Large/core/models"),
    (
        "lerobot.policies.WSA_Large.core.models.wan22",
        "lerobot/policies/WSA_Large/core/models/wan22",
    ),
)

WSA_FSDP_QKV_METHOD = "adr218_wsa_build_attention_io"
WSA_FSDP_POST_METHOD = "adr218_wsa_apply_post"


def _wsa_block_build_attention_io(
    block: nn.Module,
    tokens: torch.Tensor,
    time_modulation: torch.Tensor,
    freqs: torch.Tensor,
    *,
    mot_type: type[nn.Module],
    rope_apply: Any,
) -> tuple[torch.Tensor, ...]:
    """Run the released WSA attention-input equations inside one block call."""

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        mot_type._split_modulation(block, time_modulation)
    )
    normalized = block.norm1(tokens)
    attention_input = normalized * (1 + scale_msa) + shift_msa
    query = block.self_attn.norm_q(block.self_attn.q(attention_input))
    key = block.self_attn.norm_k(block.self_attn.k(attention_input))
    value = block.self_attn.v(attention_input)
    query = rope_apply(query, freqs, block.num_heads)
    key = rope_apply(key, freqs, block.num_heads)
    shape = (*query.shape[:-1], block.num_heads, block.attn_head_dim)
    return (
        query.view(shape),
        key.view(shape),
        value.view(shape),
        tokens,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
    )


def _wsa_block_apply_post(
    block: nn.Module,
    attention_output: torch.Tensor,
    residual: torch.Tensor,
    gate_msa: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    *,
    mot_type: type[nn.Module],
) -> torch.Tensor:
    """Run the released WSA gate/residual/FFN equations inside one block call."""

    return mot_type._apply_expert_post_block(
        block=block,
        residual_x=residual,
        mixed_attn_out=attention_output,
        gate_msa=gate_msa,
        shift_mlp=shift_mlp,
        scale_mlp=scale_mlp,
        gate_mlp=gate_mlp,
        context_payload=None,
    )


def install_pinned_wsa_namespace(source_root: Path) -> None:
    """Expose only pinned WSA package paths without importing unrelated policies."""
    src = source_root.resolve() / "src"
    for name, relative_path in _PACKAGE_PATHS:
        path = src / relative_path
        if not path.is_dir():
            raise FileNotFoundError(f"Pinned WSA package path is missing: {path}")
        current = sys.modules.get(name)
        if current is not None:
            current_paths = tuple(
                Path(item).resolve() for item in getattr(current, "__path__", ())
            )
            if path in current_paths:
                continue
            if name not in {"lerobot", "lerobot.policies"}:
                raise RuntimeError(f"Python module {name} is already bound to {current_paths}")
            if not hasattr(current, "__path__"):
                raise RuntimeError(f"Python module {name} is not a package")
            current.__path__.append(str(path))
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


@dataclass
class WSAFutureRuntime:
    tokens: torch.Tensor
    freqs: torch.Tensor
    time_modulation: torch.Tensor
    query_layer_indices: tuple[int, ...]
    collected: dict[int, torch.Tensor] = field(default_factory=dict)
    pending_post: dict[str, Any] | None = None


class WSAFutureExpertRuntime(nn.Module):
    """Execution-only adapter around the unmodified WSA Future3DExpert."""

    def __init__(self, expert: nn.Module, *, mot_type: type[nn.Module], rope_apply: Any):
        super().__init__()
        self.expert = expert
        self._mot_type = mot_type
        self._rope_apply = rope_apply
        self._validate_expert()
        self._install_staged_block_methods()

    def _install_staged_block_methods(self) -> None:
        """Expose both halves of released MoT blocks as FSDP-callable methods."""

        for block in self.expert.blocks:
            for name, implementation in (
                (WSA_FSDP_QKV_METHOD, _wsa_block_build_attention_io),
                (WSA_FSDP_POST_METHOD, _wsa_block_apply_post),
            ):
                if hasattr(block, name):
                    raise RuntimeError(f"WSA Future3D block already exposes {name}")
                setattr(block, name, MethodType(implementation, block))

    def _validate_expert(self) -> None:
        expected = {
            "hidden_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 32,
            "attn_head_dim": 128,
            "num_layers": 36,
            "num_query_tokens": 432,
            "da3_num_views": 2,
            "da3_tokens_per_view": 1296,
            "da3_query_dim": 2048,
            "query_layer_indices": (17, 23, 29, 35),
            "query_mode": "slot_noise",
        }
        for name, value in expected.items():
            actual = getattr(self.expert, name)
            if actual != value:
                raise ValueError(f"WSA Future3D {name} differs: expected {value}, got {actual}")
        if len(self.expert.blocks) != 36:
            raise ValueError("WSA Future3D block list differs from the 36-layer contract")

    @classmethod
    def from_adapted_checkpoint(
        cls,
        *,
        source_root: Path,
        checkpoint: Path,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> WSAFutureExpertRuntime:
        install_pinned_wsa_namespace(source_root)
        from lerobot.policies.WSA_Large.core.models.wan22.future_3d_expert import (  # noqa: PLC0415
            Future3DExpert,
        )
        from lerobot.policies.WSA_Large.core.models.wan22.mot import MoT  # noqa: PLC0415
        from lerobot.policies.WSA_Large.core.models.wan22.wan_video_dit import (  # noqa: PLC0415
            rope_apply,
        )

        expert = Future3DExpert(
            hidden_dim=768,
            ffn_dim=3072,
            num_heads=32,
            attn_head_dim=128,
            num_layers=36,
            num_query_tokens=432,
            da3_num_views=2,
            da3_tokens_per_view=1296,
            da3_query_dim=2048,
            query_layer_indices=(17, 23, 29, 35),
            query_mode="slot_noise",
            query_noise_scale=0.5,
            query_noise_min_sigma=0.0,
            query_noise_max_sigma=0.5,
            query_sigma_source="constant",
            slot_pos_scale=0.5,
            use_gradient_checkpointing=True,
        )
        state = load_file(str(checkpoint), device="cpu")
        incompatible = expert.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict WSA load returned incompatible keys: {incompatible}")
        del state
        expert = expert.to(device=device, dtype=dtype)
        return cls(expert, mot_type=MoT, rope_apply=rope_apply)

    def prepare(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        timestep: torch.Tensor | None = None,
        query_noise_sigma: torch.Tensor | None = None,
    ) -> WSAFutureRuntime:
        # LingBot deliberately keeps some nested experts in FP32 while its
        # root/VLM surfaces are BF16. FSDP can cast tensor arguments, but the
        # ``torch.dtype`` argument consumed by WSA's original ``pre_dit`` is
        # metadata and is not transformed by FSDP. Derive that metadata from
        # the expert itself so WSA creates queries and timesteps in the dtype
        # of the parameters that consume them.
        expert_dtype = self.expert.time_embedding[0].weight.dtype
        if timestep is not None:
            timestep = timestep.to(device=device, dtype=expert_dtype)
        if query_noise_sigma is not None:
            query_noise_sigma = query_noise_sigma.to(device=device, dtype=expert_dtype)
        prepared = self.expert.pre_dit(
            batch_size=batch_size,
            device=device,
            dtype=expert_dtype,
            timestep=timestep,
            query_noise_sigma=query_noise_sigma,
        )
        return WSAFutureRuntime(
            tokens=prepared["tokens"],
            freqs=prepared["freqs"],
            time_modulation=prepared["t_mod"],
            query_layer_indices=tuple(self.expert.query_layer_indices),
        )

    def compute_layer_qkv(
        self,
        runtime: WSAFutureRuntime,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if runtime.pending_post is not None:
            raise RuntimeError("Previous WSA Future3D layer has not consumed its attention output")
        block = self.expert.blocks[layer_index]
        build_attention_io = getattr(block, WSA_FSDP_QKV_METHOD)
        (
            query,
            key,
            value,
            residual,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = build_attention_io(
            runtime.tokens,
            runtime.time_modulation,
            runtime.freqs,
            mot_type=self._mot_type,
            rope_apply=self._rope_apply,
        )
        runtime.pending_post = {
            "block": block,
            "residual": residual,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return query, key, value

    def apply_layer_attention(
        self,
        runtime: WSAFutureRuntime,
        *,
        layer_index: int,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        pending = runtime.pending_post
        if pending is None:
            raise RuntimeError("WSA Future3D attention output arrived before Q/K/V")
        block = pending["block"]
        if block is not self.expert.blocks[layer_index]:
            raise RuntimeError("WSA Future3D layer output order changed")
        attention_output = attention_output.to(dtype=block.self_attn.o.weight.dtype)
        apply_post = getattr(block, WSA_FSDP_POST_METHOD)

        def _post_fn(
            mixed: torch.Tensor,
            residual: torch.Tensor,
            gate_msa: torch.Tensor,
            shift_mlp: torch.Tensor,
            scale_mlp: torch.Tensor,
            gate_mlp: torch.Tensor,
        ) -> torch.Tensor:
            return apply_post(
                mixed,
                residual,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                mot_type=self._mot_type,
            )

        post_inputs = (
            attention_output,
            pending["residual"],
            pending["gate_msa"],
            pending["shift_mlp"],
            pending["scale_mlp"],
            pending["gate_mlp"],
        )
        if self.expert.use_gradient_checkpointing and self.training:
            runtime.tokens = checkpoint(
                _post_fn,
                *post_inputs,
                use_reentrant=False,
            )
        else:
            runtime.tokens = _post_fn(*post_inputs)
        runtime.pending_post = None
        if layer_index in runtime.query_layer_indices:
            runtime.collected[layer_index] = runtime.tokens
        return runtime.tokens

    def project_targets(self, runtime: WSAFutureRuntime) -> tuple[torch.Tensor, ...]:
        if runtime.pending_post is not None:
            raise RuntimeError("Cannot decode WSA Future3D with an unfinished block")
        missing = tuple(
            index for index in runtime.query_layer_indices if index not in runtime.collected
        )
        if missing:
            raise RuntimeError(f"WSA Future3D did not execute required readout layers: {missing}")
        layer_tokens = tuple(runtime.collected[index] for index in runtime.query_layer_indices)
        return tuple(self.expert.project_query_layers(layer_tokens))

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "picf-next.adr218-wsa-future-runtime.v1",
            "wsa_commit": WSA_COMMIT,
            "parameter_count": sum(parameter.numel() for parameter in self.expert.parameters()),
            "layer_count": len(self.expert.blocks),
            "future_slots": self.expert.num_query_tokens,
            "query_layers": list(self.expert.query_layer_indices),
        }

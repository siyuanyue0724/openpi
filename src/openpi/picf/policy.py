from __future__ import annotations

import dataclasses
import inspect
import os
import time
import types
from typing import Any

import numpy as np
import torch

from openpi.picf.contracts import PicfObservation
from openpi.picf.core import PicfCoreOutput
from openpi.picf.core import PicfCoreState
from openpi.picf.core import PicfFullCore
from openpi.picf.core import PicfPreviousState
from openpi.picf.fsdp_utils import call_module_forward_or_method


def _timing_breakdown_enabled() -> bool:
    return os.environ.get("OPENPI_PICF_TIMING_BREAKDOWN", "").strip().lower() in {"1", "true", "yes", "on"}


def _sync_cuda_for_timing() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timing_start(enabled: bool) -> float:
    if enabled:
        _sync_cuda_for_timing()
        return time.perf_counter()
    return 0.0


def _timing_record(timing: dict[str, float], name: str, start: float, enabled: bool) -> None:
    if not enabled:
        return
    _sync_cuda_for_timing()
    timing[f"{name}_ms"] = (time.perf_counter() - start) * 1000.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _tensor_has_nonfinite(tensor: Any) -> bool:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return False
    return not bool(torch.isfinite(tensor).all().item())


def _tensor_exceeds_abs(tensor: Any, value_clip: float) -> bool:
    if (
        value_clip <= 0
        or not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or tensor.numel() == 0
    ):
        return False
    return float(torch.amax(torch.abs(tensor.detach())).item()) > value_clip


def _stabilize_prefix_tokens_for_inference(
    tokens: torch.Tensor,
    *,
    value_clip: float,
    max_rms: float,
    safety: dict[str, float],
) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor) or not tokens.is_floating_point():
        return tokens
    out = tokens
    had_nonfinite = _tensor_has_nonfinite(out)
    if had_nonfinite:
        out = torch.nan_to_num(out, nan=0.0, posinf=value_clip, neginf=-value_clip)
        safety["inference_prefix_nonfinite"] = 1.0
    else:
        safety.setdefault("inference_prefix_nonfinite", 0.0)
    if value_clip > 0:
        max_abs = float(torch.amax(torch.abs(out.detach())).item()) if out.numel() else 0.0
        safety["inference_prefix_max_abs"] = max_abs
        if max_abs > value_clip:
            out = out.clamp(min=-value_clip, max=value_clip)
            safety["inference_prefix_value_clipped"] = 1.0
        else:
            safety.setdefault("inference_prefix_value_clipped", 0.0)
    if max_rms > 0 and out.numel():
        rms = torch.sqrt(torch.mean(out.detach().to(dtype=torch.float32).square(), dim=-1, keepdim=True) + 1e-12)
        max_seen_rms = float(torch.amax(rms).item())
        safety["inference_prefix_max_rms"] = max_seen_rms
        scale = torch.clamp(max_rms / rms.to(device=out.device, dtype=out.dtype), max=1.0)
        if max_seen_rms > max_rms:
            out = out * scale
            safety["inference_prefix_rms_clipped"] = 1.0
        else:
            safety.setdefault("inference_prefix_rms_clipped", 0.0)
    return out


def _sanitize_floating_tensor_for_inference(
    tensor: torch.Tensor,
    *,
    value_clip: float,
    safety: dict[str, float],
) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    out = tensor
    changed = False
    if _tensor_has_nonfinite(out):
        out = torch.nan_to_num(out, nan=0.0, posinf=value_clip, neginf=-value_clip)
        changed = True
        safety["inference_state_nonfinite_tensors"] = safety.get("inference_state_nonfinite_tensors", 0.0) + 1.0
    if value_clip > 0 and out.numel():
        max_abs = float(torch.amax(torch.abs(out.detach())).item())
        if max_abs > value_clip:
            out = out.clamp(min=-value_clip, max=value_clip)
            changed = True
            safety["inference_state_clipped_tensors"] = safety.get("inference_state_clipped_tensors", 0.0) + 1.0
    if changed:
        safety["inference_state_sanitized"] = 1.0
    return out


def _sanitize_tree_for_inference(obj: Any, *, value_clip: float, safety: dict[str, float]) -> Any:
    if isinstance(obj, torch.Tensor):
        return _sanitize_floating_tensor_for_inference(obj, value_clip=value_clip, safety=safety)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        replacements: dict[str, Any] = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            sanitized = _sanitize_tree_for_inference(value, value_clip=value_clip, safety=safety)
            if sanitized is not value:
                replacements[field.name] = sanitized
        return dataclasses.replace(obj, **replacements) if replacements else obj
    if isinstance(obj, types.SimpleNamespace):
        replacements = {}
        changed = False
        for key, value in vars(obj).items():
            sanitized = _sanitize_tree_for_inference(value, value_clip=value_clip, safety=safety)
            replacements[key] = sanitized
            changed = changed or sanitized is not value
        return types.SimpleNamespace(**replacements) if changed else obj
    if isinstance(obj, dict):
        changed = False
        out = {}
        for key, value in obj.items():
            sanitized = _sanitize_tree_for_inference(value, value_clip=value_clip, safety=safety)
            out[key] = sanitized
            changed = changed or sanitized is not value
        return out if changed else obj
    if isinstance(obj, tuple):
        values = tuple(_sanitize_tree_for_inference(value, value_clip=value_clip, safety=safety) for value in obj)
        return values if any(sanitized is not value for sanitized, value in zip(values, obj, strict=True)) else obj
    if isinstance(obj, list):
        values = [_sanitize_tree_for_inference(value, value_clip=value_clip, safety=safety) for value in obj]
        return values if any(sanitized is not value for sanitized, value in zip(values, obj, strict=True)) else obj
    return obj


def _get_nested_attr(obj: Any, path: tuple[str, ...]) -> Any | None:
    current = obj
    for name in path:
        if current is None or not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _state_needs_inference_sanitize(state: Any, *, value_clip: float) -> bool:
    for path in (
        ("predictive", "action"),
        ("predictive", "action_chunk"),
        ("posterior", "tokens"),
        ("posterior", "slot_address"),
        ("posterior", "slot_content"),
        ("conditioned_control", "pi_prefix_tokens"),
    ):
        tensor = _get_nested_attr(state, path)
        if _tensor_has_nonfinite(tensor) or _tensor_exceeds_abs(tensor, value_clip):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class PicfPolicyTrainResult:
    output: PicfCoreOutput | None
    observed: Any | None
    semantic_override: Any | None
    flow_override: dict[str, torch.Tensor] | None
    next_state: PicfPreviousState | None


@dataclasses.dataclass(frozen=True)
class PicfPolicyActResult:
    action: torch.Tensor
    action_chunk: torch.Tensor | None
    state: PicfCoreState | None
    debug: dict[str, float]
    output: PicfCoreOutput | None


class PicfPi05Policy:
    def __init__(
        self,
        *,
        core: PicfFullCore,
        semantic_encoder: torch.nn.Module | None,
        picf_enabled: bool = True,
        inference_observe_interval: int = 1,
    ) -> None:
        self.core = core
        self.semantic_encoder = semantic_encoder
        self.picf_enabled = bool(picf_enabled)
        self.inference_observe_interval = max(1, int(inference_observe_interval))
        self._cached_inference_observed: Any | None = None
        self._cached_inference_step = 0
        self._last_finite_action_chunk: torch.Tensor | None = None
        self._last_inference_safety: dict[str, float] = {}

    def _supports_action_generation(self) -> bool:
        return bool(
            self.semantic_encoder is not None
            and bool(getattr(self.semantic_encoder, "supports_pi0_action_generation", lambda: False)())
        )

    def _requires_action_generation(self) -> bool:
        return bool(getattr(getattr(self.core, "config", None), "require_pi0_action_generator", False))

    def _require_action_generation(self) -> None:
        if not self._requires_action_generation():
            return
        if not self._supports_action_generation():
            raise RuntimeError(
                "PICF v2.2 requires PI0.5 action generation. "
                "No semantic action generator is available for this policy."
            )

    def _legacy_action_condition_tokens(self, output: PicfCoreOutput) -> Any:
        predictive = getattr(output.state, "predictive", None)
        prefix = None if predictive is None else getattr(predictive, "action_condition_tokens", None)
        if prefix is None:
            raise RuntimeError(
                "Legacy PICF core fallback requires predictive.action_condition_tokens "
                "to drive PI0.5 action generation."
            )
        return prefix

    def _action_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if bool(getattr(getattr(self.core, "config", None), "action_prefix_stopgrad", False)):
            return tokens.detach()
        return tokens

    def _picf_action_condition_enabled(self) -> bool:
        return bool(getattr(getattr(self.core, "config", None), "picf_action_condition_enabled", True))

    @staticmethod
    def _action_condition_enabled_metric(reference: torch.Tensor, enabled: bool) -> dict[str, torch.Tensor]:
        zero = reference.reshape(-1)[0].detach() * 0.0
        return {"picf_action_condition_enabled": zero + (1.0 if enabled else 0.0)}

    @staticmethod
    def _prefix_rms_normalized(tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or not tokens.is_floating_point() or tokens.numel() == 0:
            return tokens
        rms = torch.sqrt(torch.mean(tokens.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + 1.0e-12)
        return tokens / torch.clamp(rms.to(device=tokens.device, dtype=tokens.dtype), min=1.0e-6)

    def _training_action_prefix_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = getattr(self.core, "config", None)
        mode = str(getattr(cfg, "action_prefix_teacher_mode", "off")).strip().lower().replace("-", "_")
        if mode in {"", "none", "off", "disabled"}:
            return self._action_prefix_tokens(tokens), {}
        if mode != "ema":
            raise ValueError(f"Unsupported action_prefix_teacher_mode={mode!r}")
        teacher_buffer = getattr(self.core, "action_prefix_teacher_tokens", None)
        initialized = getattr(self.core, "action_prefix_teacher_initialized", None)
        if not isinstance(teacher_buffer, torch.Tensor) or not isinstance(initialized, torch.Tensor):
            return self._action_prefix_tokens(tokens), {}
        if not isinstance(tokens, torch.Tensor) or not tokens.is_floating_point() or tokens.numel() == 0:
            return self._action_prefix_tokens(tokens), {}

        online = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        online_for_teacher = online.detach()
        if online_for_teacher.ndim == 3:
            online_for_teacher = online_for_teacher.mean(dim=0)
        if online_for_teacher.shape != teacher_buffer.shape:
            return self._action_prefix_tokens(tokens), {}

        with torch.no_grad():
            if float(initialized.detach().item()) <= 0.5:
                teacher_buffer.copy_(online_for_teacher.to(device=teacher_buffer.device, dtype=teacher_buffer.dtype))
                initialized.fill_(1.0)
            teacher = teacher_buffer.to(device=online.device, dtype=online.dtype)

        online_norm = self._prefix_rms_normalized(online)
        teacher_norm = self._prefix_rms_normalized(teacher.detach())
        trust_raw = torch.mean((online_norm - teacher_norm) ** 2)
        trust_weight = max(float(getattr(cfg, "lambda_action_prefix_trust", 0.0)), 0.0)
        trust_loss = trust_raw * trust_weight
        delta_rms = torch.sqrt(torch.mean((online.detach().to(dtype=torch.float32) - teacher.detach().to(dtype=torch.float32)) ** 2) + 1.0e-12)
        online_flat = online.detach().to(dtype=torch.float32).reshape(-1)
        teacher_flat = teacher.detach().to(dtype=torch.float32).reshape(-1)
        cos = torch.sum(online_flat * teacher_flat) / torch.clamp(
            torch.linalg.norm(online_flat) * torch.linalg.norm(teacher_flat),
            min=1.0e-12,
        )

        blend = max(0.0, min(float(getattr(cfg, "action_prefix_teacher_blend", 1.0)), 1.0))
        prefix = (blend * teacher) + ((1.0 - blend) * online)
        prefix = self._action_prefix_tokens(prefix)

        decay = max(0.0, min(float(getattr(cfg, "action_prefix_teacher_ema_decay", 0.99)), 0.99999))
        with torch.no_grad():
            teacher_buffer.mul_(decay).add_(
                online_for_teacher.to(device=teacher_buffer.device, dtype=teacher_buffer.dtype),
                alpha=1.0 - decay,
            )

        zero_like = trust_loss.detach() * 0.0
        return prefix, {
            "picf_action_prefix_trust_loss": trust_loss,
            "picf_action_prefix_trust_raw": trust_raw.detach(),
            "picf_action_prefix_delta_rms": delta_rms.detach().to(device=trust_loss.device, dtype=trust_loss.dtype),
            "picf_action_prefix_cos_to_teacher": cos.detach().to(device=trust_loss.device, dtype=trust_loss.dtype),
            "picf_action_prefix_teacher_blend": zero_like + blend,
            "picf_action_prefix_teacher_ema_decay": zero_like + decay,
        }

    def _action_context_tokens(
        self,
        conditioned_control: Any,
        *,
        safety: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        cfg = getattr(self.core, "config", None)
        max_tokens = int(max(getattr(cfg, "action_context_tokens", 0), 0))
        tokens = getattr(conditioned_control, "tokens", None)
        if max_tokens <= 0 or not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
            return None, {}
        if tokens.ndim != 2 or not tokens.is_floating_point():
            return None, {}

        context = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        include_queries = bool(getattr(cfg, "action_context_include_query_tokens", False))
        query_count = int(max(getattr(cfg, "conditioned_control_queries", 0), 0))
        if not include_queries and query_count > 0 and context.shape[0] > query_count:
            context = context[:-query_count]
        context = context[: min(max_tokens, int(context.shape[0]))]
        if context.numel() == 0:
            return None, {}

        mode = str(getattr(cfg, "action_context_norm_mode", "rmsnorm")).strip().lower().replace("-", "_")
        eps = 1.0e-6
        target = max(float(getattr(cfg, "action_context_rms_target", 1.0)), eps)
        post_rms = None
        if mode in {"rmsnorm", "rms_norm", "rmscap", "rms_cap"}:
            rms = torch.sqrt(torch.mean(context.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            scale = target / torch.clamp(rms.to(device=context.device, dtype=context.dtype), min=eps)
            if mode in {"rmscap", "rms_cap"}:
                scale = torch.clamp(scale, max=1.0)
            context = context * scale
            post_rms = torch.sqrt(torch.mean(context.to(dtype=torch.float32).square(), dim=-1))
        elif mode in {"layernorm", "layer_norm"}:
            context = torch.nn.functional.layer_norm(context, (context.shape[-1],))
            post_rms = torch.sqrt(torch.mean(context.to(dtype=torch.float32).square(), dim=-1))
        elif mode not in {"", "none", "off", "disabled"}:
            raise ValueError(f"Unsupported action_context_norm_mode={mode!r}")

        gate_value = max(0.0, min(float(getattr(cfg, "action_context_output_gate", 0.25)), 1.0))
        context = context * gate_value
        if bool(getattr(cfg, "action_context_stopgrad", True)):
            context = context.detach()

        metrics_device = context.device
        metrics_dtype = context.dtype
        zero = context.reshape(-1)[0].detach() * 0.0
        metrics = {
            "picf_action_context_token_count": zero + float(context.shape[0]),
            "picf_action_context_gate": zero + gate_value,
        }
        if post_rms is not None and post_rms.numel() > 0:
            metrics["picf_action_context_post_rms_mean"] = post_rms.detach().mean().to(
                device=metrics_device,
                dtype=metrics_dtype,
            )
        if safety is not None:
            safety["inference_action_context_token_count"] = float(context.shape[0])
            safety["inference_action_context_gate"] = float(gate_value)
            if post_rms is not None and post_rms.numel() > 0:
                safety["inference_action_context_post_rms_mean"] = float(post_rms.detach().mean().item())
        return context, metrics

    @staticmethod
    def _fuse_action_context_into_prefix(
        prefix: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse bounded dense context into existing PI prefix tokens.

        Directly appending context tokens changes the PI0.5 prefix length and
        therefore shifts the action suffix position ids.  This adapter keeps the
        prefix length fixed: each prefix token reads the bounded context through
        a parameter-free attention residual, then the output is RMS-capped to
        the original prefix scale.
        """
        if (
            not isinstance(prefix, torch.Tensor)
            or not isinstance(context, torch.Tensor)
            or prefix.ndim != 2
            or context.ndim != 2
            or prefix.numel() == 0
            or context.numel() == 0
            or prefix.shape[-1] != context.shape[-1]
        ):
            return prefix, {}

        eps = 1.0e-6
        prefix_for_score = PicfPi05Policy._prefix_rms_normalized(prefix)
        context_for_score = PicfPi05Policy._prefix_rms_normalized(context.to(device=prefix.device, dtype=prefix.dtype))
        scale = float(prefix.shape[-1]) ** -0.5
        logits = torch.matmul(prefix_for_score.to(dtype=torch.float32), context_for_score.to(dtype=torch.float32).T) * scale
        attn = torch.softmax(logits, dim=-1).to(device=prefix.device, dtype=prefix.dtype)
        residual = torch.matmul(attn, context.to(device=prefix.device, dtype=prefix.dtype))
        fused = prefix + residual

        prefix_rms = torch.sqrt(torch.mean(prefix.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
        fused_rms = torch.sqrt(torch.mean(fused.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
        # Cap only upward scale drift.  This preserves the action interface
        # distribution while allowing context to rotate the prefix direction.
        cap = torch.clamp(prefix_rms / torch.clamp(fused_rms, min=eps), max=1.0)
        fused = fused * cap.to(device=fused.device, dtype=fused.dtype)

        entropy = -(attn.to(dtype=torch.float32) * torch.log(torch.clamp(attn.to(dtype=torch.float32), min=1.0e-12))).sum(
            dim=-1
        )
        zero = fused.reshape(-1)[0].detach() * 0.0
        metrics = {
            "picf_action_context_fused_prefix_token_count": zero + float(prefix.shape[0]),
            "picf_action_context_attention_entropy_mean": entropy.detach().mean().to(device=fused.device, dtype=fused.dtype),
            "picf_action_context_fused_post_rms_mean": torch.sqrt(
                torch.mean(fused.detach().to(dtype=torch.float32).square(), dim=-1)
            )
            .mean()
            .to(device=fused.device, dtype=fused.dtype),
        }
        return fused, metrics

    def _training_action_condition_tokens(self, conditioned_control: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prefix, metrics = self._training_action_prefix_tokens(conditioned_control.pi_prefix_tokens)
        metrics.update(self._action_condition_enabled_metric(prefix, True))
        context, context_metrics = self._action_context_tokens(conditioned_control)
        if context is not None:
            metrics.update(context_metrics)
            mode = str(
                getattr(getattr(self.core, "config", None), "action_context_integration", "append")
            ).strip().lower().replace("-", "_")
            if mode in {"prefix_fusion", "prefix_attention", "fuse_prefix", "fused_prefix"}:
                prefix, fusion_metrics = self._fuse_action_context_into_prefix(prefix, context)
                metrics.update(fusion_metrics)
            elif mode in {"suffix_cross_attention", "action_cross_attention", "action_adapter", "gated_cross_attention"}:
                # This mode intentionally does not append to the PI prefix.
                # The returned tensor is consumed by the semantic action-side
                # adapter as context, preserving native prefix/suffix positions.
                prefix = torch.cat([prefix, context.to(device=prefix.device, dtype=prefix.dtype)], dim=0)
            elif mode in {"append", "concat", "concatenate"}:
                prefix = torch.cat([prefix, context.to(device=prefix.device, dtype=prefix.dtype)], dim=0)
            else:
                raise ValueError(f"Unsupported action_context_integration={mode!r}")
            zero = prefix.reshape(-1)[0].detach() * 0.0
            metrics["picf_action_condition_token_count"] = zero + float(prefix.shape[0])
        return prefix, metrics

    def _uses_action_side_context_adapter(self) -> bool:
        mode = str(
            getattr(getattr(self.core, "config", None), "action_context_integration", "append")
        ).strip().lower().replace("-", "_")
        return mode in {"suffix_cross_attention", "action_cross_attention", "action_adapter", "gated_cross_attention"}

    def _inference_action_condition_tokens(self, conditioned_control: Any, safety: dict[str, float]) -> torch.Tensor:
        prefix = self._inference_action_prefix_tokens(conditioned_control.pi_prefix_tokens, safety)
        context, _metrics = self._action_context_tokens(conditioned_control, safety=safety)
        if context is not None:
            mode = str(
                getattr(getattr(self.core, "config", None), "action_context_integration", "append")
            ).strip().lower().replace("-", "_")
            if mode in {"prefix_fusion", "prefix_attention", "fuse_prefix", "fused_prefix"}:
                prefix, _fusion_metrics = self._fuse_action_context_into_prefix(prefix, context)
            elif mode in {"suffix_cross_attention", "action_cross_attention", "action_adapter", "gated_cross_attention"}:
                # Return action-side adapter context, not extra PI prefix tokens.
                prefix = torch.cat([prefix, context.to(device=prefix.device, dtype=prefix.dtype)], dim=0)
            elif mode in {"append", "concat", "concatenate"}:
                prefix = torch.cat([prefix, context.to(device=prefix.device, dtype=prefix.dtype)], dim=0)
            else:
                raise ValueError(f"Unsupported action_context_integration={mode!r}")
        safety["inference_action_condition_token_count"] = float(prefix.shape[0])
        return prefix

    def _inference_action_prefix_tokens(self, tokens: torch.Tensor, safety: dict[str, float]) -> torch.Tensor:
        """Bound PICF inference prefixes before they enter the PI0.5 sampler.

        Training intentionally keeps using `_action_prefix_tokens`; this guard is for closed-loop
        deployment only, where one non-finite recurrent state can poison all subsequent actions.
        """
        return _stabilize_prefix_tokens_for_inference(
            tokens,
            value_clip=_env_float("OPENPI_PICF_INFERENCE_PREFIX_VALUE_CLIP", 50.0),
            max_rms=_env_float("OPENPI_PICF_INFERENCE_PREFIX_MAX_RMS", 8.0),
            safety=safety,
        )

    def _safe_inference_action_chunk(self, action_chunk: Any, safety: dict[str, float]) -> torch.Tensor:
        chunk = torch.as_tensor(action_chunk)
        if chunk.is_floating_point() and _tensor_has_nonfinite(chunk):
            fallback = self._last_finite_action_chunk
            safety["inference_action_chunk_nonfinite"] = 1.0
            if fallback is not None and tuple(fallback.shape) == tuple(chunk.shape):
                safety["inference_action_chunk_fallback_last"] = 1.0
                return fallback.to(device=chunk.device, dtype=chunk.dtype).clone()
            safety["inference_action_chunk_fallback_zero"] = 1.0
            return torch.zeros_like(chunk)
        safety.setdefault("inference_action_chunk_nonfinite", 0.0)
        if chunk.is_floating_point():
            self._last_finite_action_chunk = chunk.detach().clone()
        return chunk

    def _sanitize_output_state_for_inference(
        self,
        output: PicfCoreOutput,
        safety: dict[str, float],
    ) -> PicfCoreOutput:
        state = getattr(output, "state", None)
        safety.setdefault("inference_state_sanitized", 0.0)
        value_clip = _env_float("OPENPI_PICF_INFERENCE_STATE_VALUE_CLIP", 1.0e4)
        if state is None or not _state_needs_inference_sanitize(state, value_clip=value_clip):
            return output
        sanitized_state = _sanitize_tree_for_inference(
            state,
            value_clip=value_clip,
            safety=safety,
        )
        if dataclasses.is_dataclass(output) and not isinstance(output, type):
            return dataclasses.replace(output, state=sanitized_state)
        output.state = sanitized_state
        return output

    def encode_semantic(self, observation: PicfObservation) -> Any | None:
        if self.semantic_encoder is None:
            return None
        return call_module_forward_or_method(self.semantic_encoder, "encode_observation", "encode_observation", observation)

    def recurrent_state(self, state: PicfCoreState | None) -> PicfPreviousState | None:
        if state is None:
            return None
        if hasattr(self.core, "make_recurrent_carry"):
            return self.core.make_recurrent_carry(state)
        return state

    def burnin_recurrent_transition(
        self,
        current: PicfObservation,
        *,
        previous: PicfPreviousState | None = None,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
    ) -> PicfPreviousState:
        if not self.picf_enabled:
            raise RuntimeError("PICF recurrent burn-in requires picf_enabled=True.")
        if not hasattr(self.core, "recurrent_burnin_step"):
            fallback = self.forward_train_transition(
                current,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                action_chunk_target=None,
            )
            if fallback.next_state is None:
                raise RuntimeError("PICF burn-in fallback did not produce a recurrent state.")
            return fallback.next_state
        return self.core.recurrent_burnin_step(
            current,
            previous=previous,
            point_features_override=point_features_override,
            visual_map_override=visual_map_override,
            action_future=current.action_chunk if current.action_chunk is not None else current.action,
        )

    def _pi05_only_train_transition(
        self,
        current: PicfObservation,
        *,
        semantic_override: Any | None,
        action_chunk_target: torch.Tensor | np.ndarray | None,
    ) -> PicfPolicyTrainResult:
        self._require_action_generation()
        teacher_action = self._teacher_forced_action_future(
            current,
            action_chunk_target=action_chunk_target,
        )
        if teacher_action is None:
            raise RuntimeError(
                "PI0.5-only ablation training requires a teacher-forced action or action chunk target."
            )
        flow_override = call_module_forward_or_method(
            self.semantic_encoder,
            "compute_action_flow_loss",
            "compute_action_flow_loss",
            semantic_override,
            extra_prefix_tokens=None,
            action_chunk_target=teacher_action,
        )
        return PicfPolicyTrainResult(
            output=None,
            observed=None,
            semantic_override=semantic_override,
            flow_override=flow_override,
            next_state=None,
        )

    @staticmethod
    def _action_from_sampled_chunk(action_chunk: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.as_tensor(action_chunk)
        if chunk.ndim == 1:
            return chunk[:7], chunk
        return chunk[0, :7], chunk

    def _legacy_core_step(
        self,
        observation: PicfObservation,
        *,
        previous: PicfCoreState | None,
        point_features_override: torch.Tensor | np.ndarray | None,
        visual_map_override: torch.Tensor | np.ndarray | None,
        semantic_override: Any | None,
        action_future: torch.Tensor | np.ndarray | None,
    ) -> PicfCoreOutput:
        step = self.core.step
        signature = inspect.signature(step)
        kwargs: dict[str, Any] = {}
        for name, value in (
            ("previous", previous),
            ("point_features_override", point_features_override),
            ("visual_map_override", visual_map_override),
            ("semantic_override", semantic_override),
            ("action_future", action_future),
        ):
            if name in signature.parameters:
                kwargs[name] = value
        return step(observation, **kwargs)

    def _teacher_forced_action_future(
        self,
        observation: PicfObservation,
        *,
        action_chunk_target: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor | np.ndarray | None:
        if action_chunk_target is not None:
            return action_chunk_target
        if observation.action is not None:
            return observation.action
        return None

    def forward_train_transition(
        self,
        current: PicfObservation,
        *,
        previous: PicfCoreState | None = None,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        semantic_override: Any | None = None,
        action_chunk_target: torch.Tensor | np.ndarray | None = None,
    ) -> PicfPolicyTrainResult:
        if semantic_override is None and self.semantic_encoder is not None:
            semantic_override = self.encode_semantic(current)
        if not self.picf_enabled:
            return self._pi05_only_train_transition(
                current,
                semantic_override=semantic_override,
                action_chunk_target=action_chunk_target,
            )
        if not hasattr(self.core, "observe_step"):
            output = self._legacy_core_step(
                current,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                semantic_override=semantic_override,
                action_future=current.action,
            )
            flow_override: dict[str, torch.Tensor] | None = None
            if action_chunk_target is not None:
                self._require_action_generation()
                if self._supports_action_generation():
                    prefix_tokens, prefix_metrics = self._training_action_prefix_tokens(
                        self._legacy_action_condition_tokens(output)
                    )
                    flow_override = call_module_forward_or_method(
                        self.semantic_encoder,
                        "compute_action_flow_loss",
                        "compute_action_flow_loss",
                        semantic_override,
                        extra_prefix_tokens=prefix_tokens,
                        action_chunk_target=action_chunk_target,
                    )
                    flow_override.update(prefix_metrics)
                    output.state.predictive.action = flow_override["predicted_action"]
                    output.state.predictive.action_chunk = flow_override["predicted_chunk"]
                    for source_key, debug_key in (
                        ("picf_action_prefix_trust_loss", "pi_prefix_teacher_trust_loss"),
                        ("picf_action_prefix_trust_raw", "pi_prefix_teacher_trust_raw"),
                        ("picf_action_prefix_delta_rms", "pi_prefix_teacher_delta_rms"),
                        ("picf_action_prefix_cos_to_teacher", "pi_prefix_teacher_cos_to_teacher"),
                        ("picf_action_prefix_teacher_blend", "pi_prefix_teacher_blend"),
                        ("picf_action_prefix_teacher_ema_decay", "pi_prefix_teacher_ema_decay"),
                        ("action_flow_objective_mode_id", "pi_action_flow_objective_mode_id"),
                        ("action_flow_time_mean", "pi_action_flow_time_mean"),
                        ("picf_action_expert_router_enabled", "pi_action_expert_router_enabled"),
                        ("picf_action_expert_router_gate", "pi_action_expert_router_gate"),
                        ("picf_action_expert_router_entropy_mean", "pi_action_expert_router_entropy_mean"),
                        ("picf_action_expert_router_top_weight_mean", "pi_action_expert_router_top_weight_mean"),
                        ("picf_action_expert_router_residual_rms_mean", "pi_action_expert_router_residual_rms_mean"),
                    ):
                        value = flow_override.get(source_key)
                        if isinstance(value, torch.Tensor) and value.numel() > 0:
                            output.debug[debug_key] = float(
                                value.detach().to(dtype=torch.float32).reshape(-1).mean().item()
                            )
                    output.debug["pi_prefix_teacher_mode_enabled"] = (
                        1.0 if "picf_action_prefix_trust_loss" in flow_override else 0.0
                    )
            return PicfPolicyTrainResult(
                output=output,
                observed=None,
                semantic_override=semantic_override,
                flow_override=flow_override,
                next_state=self.recurrent_state(output.state),
            )
        observed = self.core.observe_step(
            current,
            previous=previous,
            point_features_override=point_features_override,
            visual_map_override=visual_map_override,
            semantic_override=semantic_override,
        )
        flow_override: dict[str, torch.Tensor] | None = None
        if action_chunk_target is not None:
            self._require_action_generation()
            if self._picf_action_condition_enabled():
                prefix_tokens, prefix_metrics = self._training_action_condition_tokens(observed.conditioned_control)
                use_action_adapter = self._uses_action_side_context_adapter()
            else:
                prefix_tokens = None
                use_action_adapter = False
                prefix_metrics = self._action_condition_enabled_metric(
                    observed.conditioned_control.pi_prefix_tokens,
                    False,
                )
            flow_kwargs: dict[str, Any] = {
                "extra_prefix_tokens": None if use_action_adapter else prefix_tokens,
                "action_chunk_target": action_chunk_target,
            }
            if use_action_adapter:
                flow_kwargs["extra_action_context_tokens"] = prefix_tokens
            flow_override = call_module_forward_or_method(
                self.semantic_encoder,
                "compute_action_flow_loss",
                "compute_action_flow_loss",
                semantic_override,
                **flow_kwargs,
            )
            flow_override.update(prefix_metrics)
        output = self.core.finalize_with_action(
            current,
            observed,
            action_future=self._teacher_forced_action_future(
                current,
                action_chunk_target=action_chunk_target,
            ),
        )
        if flow_override is not None:
            output.state.predictive.action = flow_override["predicted_action"]
            output.state.predictive.action_chunk = flow_override["predicted_chunk"]
            for source_key, debug_key in (
                ("picf_action_prefix_trust_loss", "pi_prefix_teacher_trust_loss"),
                ("picf_action_prefix_trust_raw", "pi_prefix_teacher_trust_raw"),
                ("picf_action_prefix_delta_rms", "pi_prefix_teacher_delta_rms"),
                ("picf_action_prefix_cos_to_teacher", "pi_prefix_teacher_cos_to_teacher"),
                ("picf_action_prefix_teacher_blend", "pi_prefix_teacher_blend"),
                ("picf_action_prefix_teacher_ema_decay", "pi_prefix_teacher_ema_decay"),
                ("picf_action_context_token_count", "pi_context_token_count"),
                ("picf_action_context_gate", "pi_context_gate"),
                ("picf_action_context_post_rms_mean", "pi_context_post_rms_mean"),
                ("picf_action_context_fused_prefix_token_count", "pi_context_fused_prefix_token_count"),
                ("picf_action_context_attention_entropy_mean", "pi_context_attention_entropy_mean"),
                ("picf_action_context_fused_post_rms_mean", "pi_context_fused_post_rms_mean"),
                ("picf_action_condition_token_count", "pi_action_condition_token_count"),
                ("picf_action_context_adapter_token_count", "pi_context_adapter_token_count"),
                ("picf_action_context_adapter_gate", "pi_context_adapter_gate"),
                ("picf_action_context_adapter_attention_entropy_mean", "pi_context_adapter_attention_entropy_mean"),
                ("picf_action_context_adapter_residual_rms_mean", "pi_context_adapter_residual_rms_mean"),
                ("picf_action_context_probe_mode_id", "pi_context_probe_mode_id"),
                ("picf_action_context_probe_delta_rms_mean", "pi_context_probe_delta_rms_mean"),
                ("picf_action_context_probe_post_rms_mean", "pi_context_probe_post_rms_mean"),
                ("picf_action_condition_enabled", "pi_action_condition_enabled"),
                ("picf_action_prefix_probe_mode_id", "pi_prefix_probe_mode_id"),
                ("picf_action_prefix_probe_delta_rms_mean", "pi_prefix_probe_delta_rms_mean"),
                ("picf_action_prefix_probe_post_rms_mean", "pi_prefix_probe_post_rms_mean"),
                ("action_flow_objective_mode_id", "pi_action_flow_objective_mode_id"),
                ("action_flow_time_mean", "pi_action_flow_time_mean"),
                ("picf_action_expert_router_enabled", "pi_action_expert_router_enabled"),
                ("picf_action_expert_router_gate", "pi_action_expert_router_gate"),
                ("picf_action_expert_router_entropy_mean", "pi_action_expert_router_entropy_mean"),
                ("picf_action_expert_router_top_weight_mean", "pi_action_expert_router_top_weight_mean"),
                ("picf_action_expert_router_residual_rms_mean", "pi_action_expert_router_residual_rms_mean"),
            ):
                value = flow_override.get(source_key)
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    output.debug[debug_key] = float(value.detach().to(dtype=torch.float32).reshape(-1).mean().item())
            output.debug["pi_prefix_teacher_mode_enabled"] = (
                1.0 if "picf_action_prefix_trust_loss" in flow_override else 0.0
            )
        return PicfPolicyTrainResult(
            output=output,
            observed=observed,
            semantic_override=semantic_override,
            flow_override=flow_override,
            next_state=self.recurrent_state(output.state),
        )

    @torch.no_grad()
    def act(
        self,
        observation: PicfObservation,
        *,
        previous: PicfCoreState | None = None,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        semantic_override: Any | None = None,
    ) -> PicfPolicyActResult:
        timing_enabled = _timing_breakdown_enabled()
        timing: dict[str, float] = {}
        safety: dict[str, float] = {}
        total_start = _timing_start(timing_enabled)
        if semantic_override is None:
            stage_start = _timing_start(timing_enabled)
            semantic_override = self.encode_semantic(observation)
            _timing_record(timing, "semantic_encode", stage_start, timing_enabled)
        semantic_timing = getattr(self.semantic_encoder, "last_runtime_timing", None)
        if timing_enabled and isinstance(semantic_timing, dict):
            timing.update({f"semantic_{key}": float(value) for key, value in semantic_timing.items()})
        if not self.picf_enabled:
            self._require_action_generation()
            stage_start = _timing_start(timing_enabled)
            action_chunk = call_module_forward_or_method(
                self.semantic_encoder,
                "sample_action_chunk",
                "sample_action_chunk",
                semantic_override,
                extra_prefix_tokens=None,
            )
            _timing_record(timing, "action_sample", stage_start, timing_enabled)
            semantic_timing = getattr(self.semantic_encoder, "last_runtime_timing", None)
            if timing_enabled and isinstance(semantic_timing, dict):
                timing.update({f"action_{key}": float(value) for key, value in semantic_timing.items()})
            stage_start = _timing_start(timing_enabled)
            action, normalized_chunk = self._action_from_sampled_chunk(action_chunk)
            _timing_record(timing, "action_extract", stage_start, timing_enabled)
            _timing_record(timing, "policy_act_total", total_start, timing_enabled)
            debug = {"picf_enabled": 0.0}
            if timing_enabled:
                debug["timing"] = timing
            return PicfPolicyActResult(
                action=action,
                action_chunk=normalized_chunk,
                state=None,
                debug=debug,
                output=None,
            )
        if not hasattr(self.core, "observe_step"):
            self._require_action_generation()
            stage_start = _timing_start(timing_enabled)
            output = self._legacy_core_step(
                observation,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                semantic_override=semantic_override,
                action_future=None,
            )
            _timing_record(timing, "legacy_core_step", stage_start, timing_enabled)
            if self._supports_action_generation():
                stage_start = _timing_start(timing_enabled)
                prefix_tokens = self._inference_action_prefix_tokens(
                    self._legacy_action_condition_tokens(output),
                    safety,
                )
                action_chunk = call_module_forward_or_method(
                    self.semantic_encoder,
                    "sample_action_chunk",
                    "sample_action_chunk",
                    semantic_override,
                    extra_prefix_tokens=prefix_tokens,
                )
                action_chunk = self._safe_inference_action_chunk(action_chunk, safety)
                _timing_record(timing, "action_sample", stage_start, timing_enabled)
                semantic_timing = getattr(self.semantic_encoder, "last_runtime_timing", None)
                if timing_enabled and isinstance(semantic_timing, dict):
                    timing.update({f"action_{key}": float(value) for key, value in semantic_timing.items()})
                if not hasattr(self.core, "refresh_predictive_state_for_action"):
                    raise RuntimeError(
                        "Legacy PICF core fallback requires refresh_predictive_state_for_action "
                        "to finalize PI0.5 sampled actions."
                    )
                stage_start = _timing_start(timing_enabled)
                output.state.predictive = self.core.refresh_predictive_state_for_action(
                    observation,
                    output.state,
                    action_future=action_chunk,
                )
                _timing_record(timing, "legacy_refresh_predictive", stage_start, timing_enabled)
            output = self._sanitize_output_state_for_inference(output, safety)
            _timing_record(timing, "policy_act_total", total_start, timing_enabled)
            debug = dict(output.debug)
            debug.update(safety)
            self._last_inference_safety = safety
            if timing_enabled:
                debug["timing"] = timing
            return PicfPolicyActResult(
                action=output.state.predictive.action,
                action_chunk=getattr(output.state.predictive, "action_chunk", None),
                state=output.state,
                debug=debug,
                output=output,
            )
        self._require_action_generation()
        if bool(getattr(observation, "reset_scaffold", False)):
            self._cached_inference_observed = None
            self._cached_inference_step = 0
            self._last_finite_action_chunk = None
        should_observe = (
            self.inference_observe_interval <= 1
            or self._cached_inference_observed is None
            or (self._cached_inference_step % self.inference_observe_interval) == 0
        )
        if should_observe:
            stage_start = _timing_start(timing_enabled)
            observed = self.core.observe_step(
                observation,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                semantic_override=semantic_override,
            )
            _timing_record(timing, "picf_observe", stage_start, timing_enabled)
            observe_timing = getattr(self.core, "_last_observe_timing", None)
            if timing_enabled and isinstance(observe_timing, dict):
                timing.update({f"picf_observe_{key}": float(value) for key, value in observe_timing.items()})
            self._cached_inference_observed = observed
            timing["picf_observe_reused"] = 0.0
        else:
            observed = self._cached_inference_observed
            timing["picf_observe_ms"] = 0.0
            timing["picf_observe_reused"] = 1.0
        self._cached_inference_step += 1
        stage_start = _timing_start(timing_enabled)
        if self._picf_action_condition_enabled():
            prefix_tokens = self._inference_action_condition_tokens(observed.conditioned_control, safety)
            use_action_adapter = self._uses_action_side_context_adapter()
            safety["inference_picf_action_condition_enabled"] = 1.0
        else:
            prefix_tokens = None
            use_action_adapter = False
            safety["inference_picf_action_condition_enabled"] = 0.0
        sample_kwargs: dict[str, Any] = {
            "extra_prefix_tokens": None if use_action_adapter else prefix_tokens,
        }
        if use_action_adapter:
            sample_kwargs["extra_action_context_tokens"] = prefix_tokens
        action_chunk = call_module_forward_or_method(
            self.semantic_encoder,
            "sample_action_chunk",
            "sample_action_chunk",
            semantic_override,
            **sample_kwargs,
        )
        action_chunk = self._safe_inference_action_chunk(action_chunk, safety)
        _timing_record(timing, "action_sample", stage_start, timing_enabled)
        semantic_timing = getattr(self.semantic_encoder, "last_runtime_timing", None)
        if timing_enabled and isinstance(semantic_timing, dict):
            timing.update({f"action_{key}": float(value) for key, value in semantic_timing.items()})
        stage_start = _timing_start(timing_enabled)
        output = self.core.finalize_with_action(observation, observed, action_future=action_chunk)
        output = self._sanitize_output_state_for_inference(output, safety)
        _timing_record(timing, "picf_finalize", stage_start, timing_enabled)
        _timing_record(timing, "policy_act_total", total_start, timing_enabled)
        debug = dict(output.debug)
        debug.update(safety)
        self._last_inference_safety = safety
        if timing_enabled:
            debug["timing"] = timing
        return PicfPolicyActResult(
            action=output.state.predictive.action,
            action_chunk=output.state.predictive.action_chunk,
            state=output.state,
            debug=debug,
            output=output,
        )

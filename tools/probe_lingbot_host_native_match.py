"""Diagnostic-only fixed-batch probe for host-native prompt/object matching.

The tiny Transformer below is not a production component.  It exercises the
same role mask, MATCH tokens, posterior boundary and scalar readout used by the
released LingBot host so local CI can reject an unlearnable topology before
loading multi-billion-parameter weights.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)

HOST_NATIVE_MATCH_OVERFIT_SCHEMA = "picf-next.host-native-match-overfit.v1"


@dataclass(frozen=True, slots=True)
class HostNativeMatchOverfitReport:
    """Numerical evidence from one deterministic local expressivity probe."""

    optimizer_updates: int
    initial_loss: float
    final_loss: float
    factual_winners: tuple[int, int]
    swapped_prompt_winners: tuple[int, int]
    posterior_prompt_max_abs: float
    layer_gradient_norms: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.optimizer_updates, bool)
            or not isinstance(self.optimizer_updates, int)
            or self.optimizer_updates <= 0
        ):
            raise ValueError("match overfit updates must be positive")
        values = (
            self.initial_loss,
            self.final_loss,
            self.posterior_prompt_max_abs,
            *self.layer_gradient_norms,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values
        ):
            raise ValueError("match overfit measurements must be finite and non-negative")
        if self.factual_winners != (0, 1) or self.swapped_prompt_winners != (1, 0):
            raise ValueError("host-native MATCH failed the prompt/winner reversal gate")
        if self.final_loss >= self.initial_loss * 0.1:
            raise ValueError("host-native MATCH fixed-batch loss did not fall by at least 90%")
        if self.posterior_prompt_max_abs != 0.0:
            raise ValueError("task prompt wrote into the persistent posterior")
        if not self.layer_gradient_norms or any(value <= 0 for value in self.layer_gradient_norms):
            raise ValueError("task matching did not reach every shared host layer")

    def as_dict(self) -> dict[str, object]:
        return {
            "factual_winners": list(self.factual_winners),
            "final_loss": self.final_loss,
            "initial_loss": self.initial_loss,
            "layer_gradient_norms": list(self.layer_gradient_norms),
            "optimizer_updates": self.optimizer_updates,
            "posterior_prompt_max_abs": self.posterior_prompt_max_abs,
            "schema": HOST_NATIVE_MATCH_OVERFIT_SCHEMA,
            "status": "PASS",
            "swapped_prompt_winners": list(self.swapped_prompt_winners),
        }


class _DiagnosticSharedHost(nn.Module):
    def __init__(self, *, width: int, layer_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=4 * width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layer_count)
        )
        self.final_norm = nn.LayerNorm(width)

    def forward(self, hidden: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        if allowed.ndim != 3 or not torch.equal(allowed, allowed[:1].expand_as(allowed)):
            raise ValueError("diagnostic shared host requires one common role mask")
        output = hidden
        blocked = ~allowed[0]
        for layer in self.layers:
            output = layer(output, src_mask=blocked)
        return self.final_norm(output)


def _controls(batch_size: int) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.15, -0.2]]]).expand(batch_size, -1, -1).clone(),
        field_valid=torch.ones(batch_size, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        delta_time=torch.full((batch_size, 1), 0.1),
        reset=torch.zeros(batch_size, 1, dtype=torch.bool),
        acknowledged=torch.ones(batch_size, 1, dtype=torch.bool),
    )


def _context(batch_size: int) -> LingBotNativeContext:
    return LingBotNativeContext(
        controls=_controls(batch_size),
        native_roles=torch.tensor(
            [
                [
                    int(NativeRole.SENSOR),
                    int(NativeRole.SENSOR),
                    int(NativeRole.LANGUAGE),
                    int(NativeRole.LANGUAGE),
                ]
            ]
        ).expand(batch_size, -1),
        native_valid=torch.ones(batch_size, 4, dtype=torch.bool),
        instruction_last_index=torch.full((batch_size,), 3, dtype=torch.long),
    )


def _fixed_prefix(*, width: int, swap_prompts: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(177)
    sensors = torch.randn(1, 2, width, generator=generator).expand(2, -1, -1).clone()
    prompts = torch.randn(2, 2, width, generator=generator)
    if swap_prompts:
        prompts = prompts.flip(0)
    return torch.cat((sensors, prompts), dim=1)


def _forward(
    graph: LingBotNativeGraph,
    host: _DiagnosticSharedHost,
    *,
    swap_prompts: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = _fixed_prefix(width=graph.config.host_width, swap_prompts=swap_prompts)
    batch_size, token_count, _ = prefix.shape
    context = _context(batch_size)
    prepared, allowed, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, None],
        attention_mask=torch.ones(batch_size, token_count, token_count, dtype=torch.bool),
        position_ids=(
            torch.arange(token_count)
            .reshape(1, 1, token_count)
            .expand(3, batch_size, token_count)
            .clone()
        ),
        visual_pos_masks=torch.tensor([[True, True, False, False]]).expand(batch_size, -1),
        context=context,
    )
    hidden = host(prepared[0], allowed)
    graph.finalize_joint_outputs(outputs_embeds=[hidden, None], runtime=runtime)
    if context.posterior_state is None or context.relation_output is None:
        raise RuntimeError("host-native MATCH diagnostic forward did not finalize")
    return (
        context.relation_output.task_relevance_logits,
        context.posterior_state.rows,
    )


def posterior_prompt_effect_max_abs(
    factual_posterior: torch.Tensor,
    swapped_posterior: torch.Tensor,
) -> float:
    """Measure a prompt intervention while holding each batch slot fixed."""

    if (
        factual_posterior.ndim != 3
        or swapped_posterior.shape != factual_posterior.shape
        or not factual_posterior.is_floating_point()
        or swapped_posterior.dtype != factual_posterior.dtype
        or swapped_posterior.device != factual_posterior.device
    ):
        raise ValueError("prompt-effect posterior tensors are incompatible")
    return float((factual_posterior - swapped_posterior).abs().max().detach())


def run_host_native_match_overfit_probe(
    *,
    seed: int = 20260730,
    optimizer_updates: int = 160,
) -> HostNativeMatchOverfitReport:
    """Fit opposite prompt-conditioned winners under one identical observation."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("match overfit seed must be a non-negative integer")
    if (
        isinstance(optimizer_updates, bool)
        or not isinstance(optimizer_updates, int)
        or optimizer_updates <= 0
    ):
        raise ValueError("match overfit updates must be positive")
    torch.manual_seed(seed)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=16,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=1,
        )
    )
    host = _DiagnosticSharedHost(width=16, layer_count=3)
    parameters = tuple(graph.parameters()) + tuple(host.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3.0e-3, weight_decay=0.0)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    initial_loss = 0.0
    for update in range(optimizer_updates):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _forward(graph, host, swap_prompts=False)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        if update == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    factual_logits, factual_posterior = _forward(graph, host, swap_prompts=False)
    final_loss_tensor = F.binary_cross_entropy_with_logits(factual_logits, targets)
    final_loss_tensor.backward()
    gradient_norms = tuple(
        float(
            torch.linalg.vector_norm(
                torch.cat(
                    tuple(
                        parameter.grad.detach().reshape(-1)
                        for parameter in layer.parameters()
                        if parameter.grad is not None
                    )
                )
            )
        )
        for layer in host.layers
    )
    with torch.no_grad():
        swapped_logits, swapped_posterior = _forward(graph, host, swap_prompts=True)
    posterior_prompt_max_abs = posterior_prompt_effect_max_abs(
        factual_posterior,
        swapped_posterior,
    )
    if factual_logits.shape != (2, 2) or swapped_logits.shape != (2, 2):
        raise RuntimeError("host-native MATCH probe produced an unexpected logit shape")
    return HostNativeMatchOverfitReport(
        optimizer_updates=optimizer_updates,
        initial_loss=initial_loss,
        final_loss=float(final_loss_tensor.detach()),
        factual_winners=(
            int(factual_logits[0].argmax().item()),
            int(factual_logits[1].argmax().item()),
        ),
        swapped_prompt_winners=(
            int(swapped_logits[0].argmax().item()),
            int(swapped_logits[1].argmax().item()),
        ),
        posterior_prompt_max_abs=posterior_prompt_max_abs,
        layer_gradient_norms=gradient_norms,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--optimizer-updates", type=int, default=160)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_host_native_match_overfit_probe(
        seed=args.seed,
        optimizer_updates=args.optimizer_updates,
    ).as_dict()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="ascii")


if __name__ == "__main__":
    main()

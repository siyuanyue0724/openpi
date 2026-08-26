#!/usr/bin/env python3
"""Benchmark PICF core validation and detached-state overhead.

This is a reproducible synthetic core benchmark. It is not released-weight VLA
throughput evidence and must not be used to authorize a long training run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import replace

import torch

from picf_next.geometry import PhysicalGeometryContract
from picf_next.models.core import PICFCoreConfig
from picf_next.models.discovery import ObjectDiscoveryConfig, ObjectDiscoveryOutput
from picf_next.models.dynamics_loss import (
    ObjectGeometryOvershootingCriterion,
    ObjectGeometryRolloutTarget,
)
from picf_next.models.evidence import ModalityProjectionSpec, NativeTokenBank
from picf_next.models.temporal import (
    ObjectBeliefBatch,
    TemporalFilterConfig,
)

BENCHMARK_GEOMETRY = PhysicalGeometryContract(
    name="picf.synthetic-workspace-position.v1",
    quantity="object_position",
    reference_frame="synthetic_robot_base",
    axes=("x", "y", "z"),
    units=("m", "m", "m"),
    normalization_offset=(0.0, 0.0, 0.0),
    normalization_scale=(1.0, 1.0, 1.0),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--train-iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--vjepa-tokens", type=int, default=196)
    parser.add_argument("--touch-tokens", type=int, default=32)
    parser.add_argument("--sonata-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": statistics.quantiles(values, n=20)[18],
    }


def _time_interleaved(
    arms: tuple[tuple[str, Callable[[], None]], ...],
    *,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        for _name, call in arms:
            call()
    _synchronize(device)
    values = {name: [] for name, _call in arms}
    for iteration in range(iterations):
        ordered = arms if iteration % 2 == 0 else tuple(reversed(arms))
        for name, call in ordered:
            _synchronize(device)
            start = time.perf_counter_ns()
            call()
            _synchronize(device)
            values[name].append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {name: _summary_ms(samples) for name, samples in values.items()}


def _config(*, runtime_validation: str) -> PICFCoreConfig:
    return PICFCoreConfig(
        modality_specs=(
            ModalityProjectionSpec("vjepa", token_dim=256, geometry_dim=3),
            ModalityProjectionSpec(
                "anytouch",
                token_dim=128,
                require_single_active_group=True,
            ),
            ModalityProjectionSpec("sonata", token_dim=192, geometry_dim=3),
        ),
        binding_dim=256,
        discovery=ObjectDiscoveryConfig(
            input_dim=256,
            hidden_dim=256,
            num_queries=16,
            num_layers=2,
            num_heads=8,
            address_dim=64,
            content_dim=128,
            geometry_dim=3,
            geometry_contract=BENCHMARK_GEOMETRY,
            initial_variance=0.1,
        ),
        temporal=TemporalFilterConfig(
            address_dim=64,
            content_dim=128,
            geometry_dim=3,
            geometry_contract=BENCHMARK_GEOMETRY,
            action_dim=32,
            reference_delta_t_s=0.05,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
        ),
        posterior_capacity=16,
        runtime_validation=runtime_validation,
    )


def _bank(
    name: str,
    *,
    batch_size: int,
    token_count: int,
    token_dim: int,
    geometry_dim: int,
    generator: torch.Generator,
    device: torch.device,
    grouped: bool = False,
) -> NativeTokenBank:
    valid = torch.ones(batch_size, token_count, dtype=torch.bool, device=device)
    geometry = (
        torch.randn(
            batch_size,
            token_count,
            geometry_dim,
            generator=generator,
            device=device,
        )
        if geometry_dim
        else None
    )
    group_id = None
    if grouped:
        group_id = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, token_count)
    return NativeTokenBank(
        modality=name,
        tokens=torch.randn(
            batch_size,
            token_count,
            token_dim,
            generator=generator,
            device=device,
        ),
        valid=valid,
        geometry=geometry,
        group_id=group_id,
    )


def _occupied_belief(discovery: ObjectDiscoveryOutput) -> ObjectBeliefBatch:
    """Materialize a valid synthetic posterior independent of birth calibration."""

    batch_size, capacity = discovery.existence_logits.shape
    return ObjectBeliefBatch(
        address_mean=discovery.address_mean.detach().clone(),
        content_mean=discovery.content_mean.detach().clone(),
        geometry_mean=discovery.geometry_mean.detach().clone(),
        geometry_covariance_diag=discovery.geometry_variance.detach().clone(),
        existence_logits=torch.full_like(discovery.existence_logits, 4.0),
        visibility_given_existence_logits=torch.full_like(discovery.existence_logits, 4.0),
        measurement_age_s=torch.zeros_like(discovery.existence_logits),
        valid=torch.ones(
            batch_size,
            capacity,
            device=discovery.existence_logits.device,
            dtype=torch.bool,
        ),
        age=torch.zeros(
            batch_size,
            capacity,
            device=discovery.existence_logits.device,
            dtype=torch.long,
        ),
    )


def _synthetic_geometry_rollout(
    belief: ObjectBeliefBatch,
    *,
    action_dim: int,
    horizon: int,
    generator: torch.Generator,
) -> tuple[tuple[tuple[str | None, ...], ...], ObjectGeometryRolloutTarget]:
    """Build a detached, row-aligned physical target for timing only."""

    if horizon <= 0:
        raise ValueError("rollout benchmark horizon must be positive")
    batch_size, capacity = belief.valid.shape
    valid_cpu = belief.valid.detach().cpu()
    row_keys = tuple(
        tuple(
            f"sample-{batch_index}/row-{row_index}"
            if bool(valid_cpu[batch_index, row_index])
            else None
            for row_index in range(capacity)
        )
        for batch_index in range(batch_size)
    )
    if not any(key is not None for sample in row_keys for key in sample):
        raise RuntimeError("synthetic rollout benchmark requires at least one occupied row")
    target_keys = tuple(tuple(sample for _ in range(horizon)) for sample in row_keys)
    supervised = (
        belief.valid.detach()
        .unsqueeze(1)
        .unsqueeze(-1)
        .expand(
            -1,
            horizon,
            -1,
            belief.geometry_mean.shape[-1],
        )
    )
    horizon_offset = torch.arange(
        1,
        horizon + 1,
        device=belief.geometry_mean.device,
        dtype=belief.geometry_mean.dtype,
    ).view(1, horizon, 1, 1)
    geometry = belief.geometry_mean.detach().unsqueeze(1) + 0.001 * horizon_offset
    geometry = torch.where(supervised, geometry, torch.zeros_like(geometry))
    variance = torch.where(
        supervised,
        torch.full_like(geometry, 0.001),
        torch.zeros_like(geometry),
    )
    target = ObjectGeometryRolloutTarget(
        executed_actions=torch.randn(
            batch_size,
            horizon,
            action_dim,
            generator=generator,
            device=belief.geometry_mean.device,
            dtype=belief.geometry_mean.dtype,
        ),
        delta_t_s=torch.full(
            (batch_size, horizon),
            0.05,
            device=belief.geometry_mean.device,
            dtype=belief.geometry_mean.dtype,
        ),
        step_valid=torch.ones(
            batch_size,
            horizon,
            device=belief.geometry_mean.device,
            dtype=torch.bool,
        ),
        identity_keys=target_keys,
        geometry=geometry,
        geometry_variance=variance,
        geometry_supervised=supervised,
        geometry_contract=BENCHMARK_GEOMETRY,
    )
    return row_keys, target


def main() -> None:
    args = _parse_args()
    positive = {
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "train_iterations": args.train_iterations,
        "warmup": args.warmup,
    }
    if (
        any(value <= 0 for value in positive.values())
        or args.iterations < 2
        or (not args.skip_train and args.train_iterations < 2)
    ):
        raise ValueError("batch/iteration sizes must be positive and measured iterations >= 2")
    token_counts = (args.vjepa_tokens, args.touch_tokens, args.sonata_tokens)
    if any(value < 0 for value in token_counts) or not any(token_counts):
        raise ValueError("token counts must be nonnegative and not all zero")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")

    full_config = _config(runtime_validation="full")
    metadata_config = replace(full_config, runtime_validation="metadata")
    full = full_config.build().eval().to(device)
    metadata = metadata_config.build().eval().to(device)
    metadata.load_state_dict(full.state_dict())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    banks = (
        _bank(
            "vjepa",
            batch_size=args.batch_size,
            token_count=args.vjepa_tokens,
            token_dim=256,
            geometry_dim=3,
            generator=generator,
            device=device,
        ),
        _bank(
            "anytouch",
            batch_size=args.batch_size,
            token_count=args.touch_tokens,
            token_dim=128,
            geometry_dim=0,
            generator=generator,
            device=device,
            grouped=True,
        ),
        _bank(
            "sonata",
            batch_size=args.batch_size,
            token_count=args.sonata_tokens,
            token_dim=192,
            geometry_dim=3,
            generator=generator,
            device=device,
        ),
    )
    action = torch.zeros(args.batch_size, full_config.temporal.action_dim, device=device)
    delta_t = torch.full(
        (args.batch_size,),
        full_config.temporal.reference_delta_t_s,
        device=device,
    )
    empty = full_config.empty_belief(batch_size=args.batch_size, device=device)

    with torch.no_grad():
        full_output = full(banks, empty, action, delta_t)
        metadata_output = metadata(banks, empty, action, delta_t)
        checked_outputs = {
            "action_value": (
                metadata_output.action_bank.value,
                full_output.action_bank.value,
            ),
            "binding_features": (
                metadata_output.projection.binding_features,
                full_output.projection.binding_features,
            ),
            "discovery_ownership": (
                metadata_output.discovery.ownership,
                full_output.discovery.ownership,
            ),
            "posterior_state": (
                metadata_output.posterior.belief.state_mean,
                full_output.posterior.belief.state_mean,
            ),
        }
        parity_max_abs = {}
        for name, (actual, expected) in checked_outputs.items():
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            parity_max_abs[name] = (actual - expected).abs().max().item()
        if not torch.equal(
            metadata_output.posterior.event_type,
            full_output.posterior.event_type,
        ):
            raise AssertionError("runtime validation modes changed discrete lifecycle events")
        carried = _occupied_belief(full_output.discovery)

        inference = _time_interleaved(
            (
                ("full", lambda: full(banks, carried, action, delta_t)),
                ("metadata", lambda: metadata(banks, carried, action, delta_t)),
            ),
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
        )

    training = None
    if not args.skip_train:
        train_model = metadata_config.build().to(device)
        train_model.load_state_dict(full.state_dict())
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=0.0)
        overshooting = ObjectGeometryOvershootingCriterion()
        with torch.no_grad():
            rollout_start = train_model(banks, carried, action, delta_t).posterior.belief
        rollout_row_keys, rollout_target = _synthetic_geometry_rollout(
            rollout_start,
            action_dim=full_config.temporal.action_dim,
            horizon=2,
            generator=generator,
        )

        def train_step(prior: ObjectBeliefBatch, *, include_overshooting: bool = False) -> None:
            optimizer.zero_grad(set_to_none=True)
            output = train_model(banks, prior, action, delta_t)
            loss = output.action_bank.value.square().mean()
            loss = loss + output.discovery.ownership.square().mean()
            if include_overshooting:
                if not torch.equal(output.posterior.belief.valid, rollout_start.valid):
                    raise RuntimeError("timed rollout changed the fixed posterior row inventory")
                loss = (
                    loss
                    + overshooting(
                        train_model.posterior_filter.transition,
                        output.posterior.belief,
                        rollout_row_keys,
                        rollout_target,
                    ).loss
                )
            loss.backward()
            optimizer.step()

        training = _time_interleaved(
            (
                ("reset", lambda: train_step(empty)),
                ("detached_carry", lambda: train_step(carried)),
                (
                    "detached_carry_h2_overshooting",
                    lambda: train_step(carried, include_overshooting=True),
                ),
            ),
            iterations=args.train_iterations,
            warmup=args.warmup,
            device=device,
        )
        training["detached_carry_over_reset_median"] = (
            training["detached_carry"]["median"] / training["reset"]["median"]
        )
        training["h2_overshooting_over_detached_carry_median"] = (
            training["detached_carry_h2_overshooting"]["median"]
            / training["detached_carry"]["median"]
        )

    report = {
        "batch_size": args.batch_size,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "full_over_metadata_inference_median": (
            inference["full"]["median"] / inference["metadata"]["median"]
        ),
        "inference_ms": inference,
        "runtime_mode_parity_max_abs": parity_max_abs,
        "runtime_modes_numerically_equivalent": True,
        "schema": "picf-next.core-runtime-benchmark.v1",
        "token_counts": {
            "anytouch": args.touch_tokens,
            "sonata": args.sonata_tokens,
            "vjepa": args.vjepa_tokens,
        },
        "torch_version": torch.__version__,
        "training_ms": training,
        "warning": (
            "synthetic PICF core only; not released-weight VLA, data-pipeline, "
            "two-A100, or long-train throughput evidence"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

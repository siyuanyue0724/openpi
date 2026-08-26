"""Deterministic long-horizon acceptance probe for the persistent object filter.

The synthetic labels in this module are evaluation-only. They are never passed
to the filter. The probe exercises the production lifecycle MAP and variational
belief update over query permutations, geometric crossings, clutter and occlusion.
It is a mechanics/calibration test, not evidence of learned visual discovery or
closed-loop robot capability.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import torch

from picf_next.geometry import PhysicalGeometryContract
from picf_next.models.discovery import ObjectDiscoveryOutput, ObjectExistenceCalibration
from picf_next.models.filter import PersistentObjectFilter
from picf_next.models.temporal import TemporalFilterConfig, empty_object_belief

_OBJECT_COUNT = 3
_CLUTTER_LABEL = -1
_PROBE_GEOMETRY = PhysicalGeometryContract(
    name="picf.posterior-probe-position.v1",
    quantity="object_position",
    reference_frame="synthetic_world",
    axes=("x", "y"),
    units=("m", "m"),
    normalization_offset=(0.0, 0.0),
    normalization_scale=(1.0, 1.0),
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _probe_config() -> TemporalFilterConfig:
    # Low detectability makes 20-frame misses survivable and simultaneously
    # requires the calibrated spherical identity likelihood to recover a real
    # reappearance over birth/clutter. Production detectability remains learned
    # and starts at the independent TemporalFilterConfig default of 0.85.
    return TemporalFilterConfig(
        address_dim=4,
        content_dim=3,
        geometry_dim=2,
        geometry_contract=_PROBE_GEOMETRY,
        action_dim=2,
        reference_delta_t_s=0.1,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        initial_process_variance=1e-4,
        initial_survival_probability=0.9995,
        initial_detection_probability=0.08,
    )


def _constant_transition(model: PersistentObjectFilter) -> None:
    """Make the learned predictor an exact constant-motion-free calibration fixture."""

    config = model.config
    with torch.no_grad():
        for parameter in model.transition.parameters():
            parameter.zero_()
        model.transition.process_variance_head.bias.fill_(
            _inverse_softplus(config.initial_process_variance - config.minimum_variance)
        )
        model.transition.survival_head.bias.fill_(_logit(config.initial_survival_probability))
        for head in (
            model.transition.detectability_if_detected_head,
            model.transition.detectability_if_missed_head,
        ):
            head.bias.fill_(_logit(config.initial_detection_probability))


def _occlusion_windows(steps: int) -> dict[int, tuple[int, int]]:
    width = max(3, min(20, steps // 40))
    starts = (steps // 5, steps // 2, 4 * steps // 5)
    return {
        object_index: (start, min(start + width, steps - 1))
        for object_index, start in enumerate(starts)
    }


def _object_state(
    step: int,
    steps: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = step / max(steps - 1, 1)
    addresses = torch.eye(4, dtype=torch.float32)[:_OBJECT_COUNT]
    content = torch.tensor(
        [
            [1.0, 0.0, 0.2 * phase],
            [0.0, 1.0, -0.2 * phase],
            [0.0, 0.0, 1.0 + 0.1 * math.sin(4.0 * math.pi * phase)],
        ],
        dtype=torch.float32,
    )
    geometry = torch.tensor(
        [
            [-1.0 + 2.0 * phase, -0.2],
            [1.0 - 2.0 * phase, -0.2],
            [0.25 * math.sin(2.0 * math.pi * phase), 0.5],
        ],
        dtype=torch.float32,
    )
    return addresses.to(dtype), content.to(dtype), geometry.to(dtype)


def _frame_discovery(
    *,
    step: int,
    steps: int,
    windows: Mapping[int, tuple[int, int]],
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[ObjectDiscoveryOutput, tuple[int, ...]]:
    addresses, content, geometry = _object_state(step, steps, dtype=dtype)
    visible = [
        object_index
        for object_index in range(_OBJECT_COUNT)
        if not (windows[object_index][0] <= step < windows[object_index][1])
    ]
    labels = visible + [_CLUTTER_LABEL]
    permutation = torch.randperm(len(labels), generator=generator).tolist()
    labels = [labels[index] for index in permutation]

    frame_addresses = []
    frame_content = []
    frame_geometry = []
    existence_logits = []
    for label in labels:
        if label == _CLUTTER_LABEL:
            frame_addresses.append(
                torch.tensor([-1.0, -1.0, 0.0, 0.0], dtype=dtype).div(math.sqrt(2.0))
            )
            frame_content.append(torch.tensor([-2.0, -2.0, -2.0], dtype=dtype))
            frame_geometry.append(torch.tensor([3.0, 3.0], dtype=dtype))
            existence_logits.append(_logit(0.01))
        else:
            frame_addresses.append(addresses[label])
            frame_content.append(content[label])
            noise = (0.002 * torch.randn(2, generator=generator)).to(dtype)
            frame_geometry.append(geometry[label] + noise)
            existence_logits.append(_logit(0.999))

    query_count = len(labels)
    token_count = query_count + 1
    ownership_logits = torch.full(
        (1, token_count, query_count + 1),
        -12.0,
        dtype=dtype,
    )
    for query_index in range(query_count):
        ownership_logits[0, query_index, query_index] = 12.0
    ownership_logits[0, -1, -1] = 12.0
    token_valid = torch.ones(1, token_count, dtype=torch.bool)
    return (
        ObjectDiscoveryOutput(
            query_features=torch.zeros(1, query_count, 8, dtype=dtype),
            address_mean=torch.stack(frame_addresses).unsqueeze(0),
            content_mean=torch.stack(frame_content).unsqueeze(0),
            geometry_mean=torch.stack(frame_geometry).unsqueeze(0),
            geometry_variance=torch.full((1, query_count, 2), 0.01, dtype=dtype),
            geometry_contract=_PROBE_GEOMETRY,
            existence_logits=torch.tensor([existence_logits], dtype=dtype),
            localization_confidence_logits=torch.full(
                (1, query_count),
                12.0,
                dtype=dtype,
            ),
            ownership_logits=ownership_logits,
            ownership=torch.softmax(ownership_logits, dim=-1),
            token_valid=token_valid,
            token_group_id=torch.full((1, token_count), -1, dtype=torch.long),
            evidence_available=torch.ones(1, dtype=torch.bool),
            existence_calibration=ObjectExistenceCalibration(),
        ),
        tuple(labels),
    )


def run_long_horizon_posterior_probe(
    *,
    steps: int = 1000,
    seed: int = 20260715,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Run and score a deterministic no-gradient posterior sequence."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 100:
        raise ValueError("steps must be an integer of at least 100")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("posterior probe dtype must be float32 or bfloat16")

    config = _probe_config()
    model = PersistentObjectFilter(config).to(dtype=dtype).eval()
    _constant_transition(model)
    belief = empty_object_belief(config, batch_size=1, capacity=5, dtype=dtype)
    generator = torch.Generator().manual_seed(seed)
    windows = _occlusion_windows(steps)
    initial_rows: dict[int, int] = {}
    initialization_errors = 0
    identity_failures = 0
    unexpected_map_births = 0
    support_births_after_initialization = 0
    map_cardinality_violations = 0
    posterior_support_violations = 0
    max_posterior_support = 0
    max_tentative_ownership_leak = 0.0
    incorrect_row_modes = 0
    incorrect_column_modes = 0
    visible_null_modes = 0
    invalid_persistent_frames = 0
    max_address_drift = 0.0
    max_current_content_residual = 0.0
    max_content_equation_error = 0.0
    max_ownership_normalization_error = 0.0
    max_association_convergence_residual = 0.0
    query_orders: set[tuple[int, ...]] = set()
    finite = True
    covariance_records = {
        object_index: {
            "before": None,
            "peak": 0.0,
            "after": None,
            "miss_events": 0,
        }
        for object_index in range(_OBJECT_COUNT)
    }

    start_ns = time.perf_counter_ns()
    with torch.no_grad():
        for step in range(steps):
            for object_index, (start, _end) in windows.items():
                if step == start and object_index in initial_rows:
                    row = initial_rows[object_index]
                    covariance_records[object_index]["before"] = float(
                        belief.geometry_covariance_diag[0, row].mean()
                    )

            discovery, labels = _frame_discovery(
                step=step,
                steps=steps,
                windows=windows,
                generator=generator,
                dtype=dtype,
            )
            query_orders.add(labels)
            output = model(
                belief,
                discovery,
                torch.zeros(1, config.action_dim, dtype=dtype),
                torch.full((1,), config.reference_delta_t_s, dtype=dtype),
            )
            belief = output.belief

            max_association_convergence_residual = max(
                max_association_convergence_residual,
                float(output.association_convergence_residual),
            )
            max_ownership_normalization_error = max(
                max_ownership_normalization_error,
                float((output.ownership.float().sum(dim=-1) - 1.0).abs().max()),
            )
            posterior_support = int(belief.valid.sum())
            map_cardinality = int(output.map_present.sum())
            max_posterior_support = max(max_posterior_support, posterior_support)
            map_cardinality_violations += int(map_cardinality != _OBJECT_COUNT)
            posterior_support_violations += int(
                posterior_support < _OBJECT_COUNT or posterior_support > belief.valid.shape[1]
            )
            tentative_ownership = output.ownership[..., :-1].masked_select(
                (~output.map_present).unsqueeze(1)
            )
            if tentative_ownership.numel() > 0:
                max_tentative_ownership_leak = max(
                    max_tentative_ownership_leak,
                    float(tentative_ownership.abs().max()),
                )
            if step > 0:
                support_births_after_initialization += int(output.born.sum())
                unexpected_map_births += int((output.born & output.map_present).sum())

            finite = finite and all(
                bool(torch.isfinite(tensor).all())
                for tensor in (
                    belief.state_mean,
                    belief.geometry_covariance_diag,
                    belief.existence_logits,
                    belief.visibility_given_existence_logits,
                    output.innovation,
                    output.ownership,
                )
            )
            if step == 0:
                used_rows: set[int] = set()
                for query_index, label in enumerate(labels):
                    if label == _CLUTTER_LABEL:
                        continue
                    distance = (
                        (belief.address_mean[0] - discovery.address_mean[0, query_index])
                        .abs()
                        .amax(dim=-1)
                    )
                    distance = distance.masked_fill(~belief.valid[0], torch.inf)
                    row = int(distance.argmin())
                    address_tolerance = max(1e-6, torch.finfo(dtype).eps)
                    if float(distance[row]) > address_tolerance or row in used_rows:
                        initialization_errors += 1
                        continue
                    initial_rows[label] = row
                    used_rows.add(row)

            predicted = output.prior_prediction.belief
            if step > 0:
                probability_floor = torch.finfo(torch.float32).eps
                predicted_existence = predicted.existence.float()
                predicted_detection = torch.sigmoid(
                    predicted.visibility_given_existence_logits.float()
                )
                no_detection_mass = (1.0 - predicted_existence * predicted_detection).clamp_min(
                    probability_floor
                )
                alive_given_no_detection = (
                    predicted_existence * (1.0 - predicted_detection) / no_detection_mass
                )
                posterior_existence = (
                    output.match_probability.float().sum(dim=-1)
                    + output.null_probability.float() * alive_given_no_detection
                ).clamp_min(probability_floor)
                null_weight = (
                    output.null_probability.float() * alive_given_no_detection / posterior_existence
                )
                match_weight = output.match_probability.float() / posterior_existence.unsqueeze(-1)
                expected_content = (
                    null_weight.unsqueeze(-1) * predicted.content_mean.float()
                    + (
                        match_weight.unsqueeze(-1) * discovery.content_mean.float().unsqueeze(1)
                    ).sum(dim=2)
                ).to(dtype)
                retained_existing = predicted.valid & belief.valid & ~output.born
                if retained_existing.any():
                    max_content_equation_error = max(
                        max_content_equation_error,
                        float(
                            (
                                belief.content_mean[retained_existing]
                                - expected_content[retained_existing]
                            )
                            .abs()
                            .max()
                        ),
                    )

            for query_index, label in enumerate(labels):
                if label == _CLUTTER_LABEL or label not in initial_rows:
                    continue
                row = initial_rows[label]
                if not bool(belief.valid[0, row]):
                    identity_failures += 1
                    continue
                max_current_content_residual = max(
                    max_current_content_residual,
                    float(
                        (belief.content_mean[0, row] - discovery.content_mean[0, query_index])
                        .abs()
                        .max()
                    ),
                )
                if step == 0:
                    continue
                true_probability = output.match_probability[0, row, query_index]
                row_competitors = output.match_probability[0, row].clone()
                row_competitors[query_index] = -1.0
                column_competitors = output.match_probability[0, :, query_index].clone()
                column_competitors[row] = -1.0
                comparison_tolerance = 32.0 * torch.finfo(torch.float32).eps
                incorrect_row_modes += int(
                    float(true_probability + comparison_tolerance) < float(row_competitors.max())
                )
                incorrect_column_modes += int(
                    float(true_probability + comparison_tolerance) < float(column_competitors.max())
                )
                visible_null_modes += int(
                    float(true_probability + comparison_tolerance)
                    < float(output.null_probability[0, row])
                )

            for object_index, row in initial_rows.items():
                start, end = windows[object_index]
                hidden = start <= step < end
                invalid_persistent_frames += int(hidden and not bool(belief.valid[0, row]))
                if hidden and bool(belief.valid[0, row]):
                    covariance_records[object_index]["peak"] = max(
                        float(covariance_records[object_index]["peak"]),
                        float(belief.geometry_covariance_diag[0, row].mean()),
                    )
                    covariance_records[object_index]["miss_events"] = int(
                        covariance_records[object_index]["miss_events"]
                    ) + int(
                        output.null_probability[0, row] > output.match_probability[0, row].sum()
                    )
                if step == end and bool(belief.valid[0, row]):
                    covariance_records[object_index]["after"] = float(
                        belief.geometry_covariance_diag[0, row].mean()
                    )
                expected_address = torch.eye(config.address_dim)[object_index]
                if bool(belief.valid[0, row]):
                    max_address_drift = max(
                        max_address_drift,
                        float((belief.address_mean[0, row] - expected_address).abs().max()),
                    )

    elapsed_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000.0
    occlusion_checks = []
    for object_index in range(_OBJECT_COUNT):
        start, end = windows[object_index]
        record = covariance_records[object_index]
        before = record["before"]
        peak = float(record["peak"])
        after = record["after"]
        passed = (
            isinstance(before, float)
            and isinstance(after, float)
            and peak > before
            and after < peak
            and int(record["miss_events"]) == end - start
        )
        occlusion_checks.append(
            {
                "after_variance": after,
                "before_variance": before,
                "end_step_exclusive": end,
                "expected_miss_events": end - start,
                "miss_events": int(record["miss_events"]),
                "object_index": object_index,
                "passed": passed,
                "peak_variance": peak,
                "start_step": start,
            }
        )

    numerical_tolerance = max(1e-6, 4.0 * torch.finfo(dtype).eps)
    acceptance = {
        "all_objects_initialized": len(initial_rows) == _OBJECT_COUNT,
        "initialization_is_one_to_one": initialization_errors == 0,
        "no_map_birth_after_initialization": unexpected_map_births == 0,
        "map_cardinality_is_exact": map_cardinality_violations == 0,
        "posterior_support_is_bounded": posterior_support_violations == 0,
        "tentative_ownership_does_not_leak": (max_tentative_ownership_leak <= numerical_tolerance),
        "content_matches_pda_equation": max_content_equation_error <= numerical_tolerance,
        "finite": finite,
        "identity_address_immutable": max_address_drift <= 1e-7,
        "identity_rows_persist": identity_failures == 0,
        "visible_edge_is_row_mode": incorrect_row_modes == 0,
        "visible_edge_is_column_mode": incorrect_column_modes == 0,
        "visible_edge_beats_null": visible_null_modes == 0,
        "no_persistent_row_loss_during_occlusion": invalid_persistent_frames == 0,
        "occlusion_covariance_calibrates": all(item["passed"] for item in occlusion_checks),
        "ownership_is_conservative": (max_ownership_normalization_error <= numerical_tolerance),
        "association_transport_converges": max_association_convergence_residual <= 1e-3,
    }
    return {
        "acceptance": acceptance,
        "dtype": str(dtype).removeprefix("torch."),
        "elapsed_seconds": elapsed_s,
        # Backward-compatible field name: cardinality now means the extracted
        # MAP object set, not every retained Bernoulli support component.
        "cardinality_violations": map_cardinality_violations,
        "identity_failures": identity_failures,
        "initialization_errors": initialization_errors,
        "incorrect_column_modes": incorrect_column_modes,
        "incorrect_row_modes": incorrect_row_modes,
        "invalid_persistent_frames": invalid_persistent_frames,
        "max_address_drift": max_address_drift,
        "max_association_convergence_residual": max_association_convergence_residual,
        "max_content_equation_error": max_content_equation_error,
        "max_current_content_residual": max_current_content_residual,
        "max_ownership_normalization_error": max_ownership_normalization_error,
        "max_posterior_support": max_posterior_support,
        "max_tentative_ownership_leak": max_tentative_ownership_leak,
        "milliseconds_per_transition": 1000.0 * elapsed_s / steps,
        "occlusions": occlusion_checks,
        "passed": all(acceptance.values()),
        "query_order_count": len(query_orders),
        "posterior_support_violations": posterior_support_violations,
        "schema": "picf-next.long-horizon-posterior-probe.v4",
        "seed": seed,
        "steps": steps,
        "support_births_after_initialization": support_births_after_initialization,
        "unexpected_births": unexpected_map_births,
        "visible_null_modes": visible_null_modes,
        "warning": (
            "controlled filter mechanics only; not learned discovery, released-weight VLA, "
            "CALVIN closed-loop, or two-A100 evidence"
        ),
    }

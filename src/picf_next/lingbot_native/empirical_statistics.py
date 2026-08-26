"""Hash-bound paired hierarchical statistics for LingBot-native gates.

The model-specific evaluators emit one scalar observation per paired episode.
This module owns only the preregistered aggregation and decision procedure: it
never reads model inputs, labels, or hidden states and is not part of inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

EMPIRICAL_OBSERVATIONS_SCHEMA = "picf-next.lingbot-native-empirical-observations.v3"
EMPIRICAL_EVALUATION_PLAN_SCHEMA = "picf-next.lingbot-native-evaluation-plan.v3"
EMPIRICAL_REPORT_SCHEMAS = {
    "G2": "picf-next.lingbot-native-g2-object-evaluation.v4",
    "G3": "picf-next.lingbot-native-g3-temporal-evaluation.v4",
    "G4": "picf-next.lingbot-native-g4-cross-modal.v4",
    "G5": "picf-next.lingbot-native-g5-action-curves.v4",
    "G6": "picf-next.lingbot-native-g6-closed-loop.v4",
}

EMPIRICAL_REQUIRED_ARMS = {
    "G2": ("M", "C"),
    "G3": ("O", "C"),
    "G4": ("C",),
    "G5": ("A", "H", "M", "O", "C"),
    "G6": ("A", "O", "C"),
}

# ``difference`` means candidate minus reference. ``value`` is a directly
# measured non-negative error/rate. Labels are frozen to prevent a producer
# from silently reversing a comparison after seeing results.
EMPIRICAL_COMPARISON_SPECS = {
    "G2": {
        "object_mask_vs_chance": ("lower_gt", "difference", "C", "chance"),
        "object_mask_C_vs_M": ("lower_gt", "difference", "C", "M"),
        "existence_calibration_error": ("upper_lt", "value", "C", None),
        "task_grounding_C_vs_M": ("lower_gt", "difference", "C", "M"),
        "dense_task_grounding_C_vs_M": ("lower_gt", "difference", "C", "M"),
        "row_collapse_rate": ("upper_lt", "value", "C", None),
    },
    "G3": {
        "identity_stability_C_vs_O": ("lower_gt", "difference", "C", "O"),
        "occlusion_recovery_C_vs_O": ("lower_gt", "difference", "C", "O"),
        "long_age_calibration_error": ("upper_lt", "value", "C", None),
        "reidentification_C_vs_O": ("lower_gt", "difference", "C", "O"),
    },
    "G4": {
        "binding_vs_within_scene_hard_negative": (
            "lower_gt",
            "difference",
            "same_entity",
            "within_scene_hard_negative",
        ),
        "missing_modality_noninferiority": (
            "lower_ge",
            "difference",
            "missing_modality",
            "all_available_modalities",
        ),
        "corrupt_modality_noninferiority": (
            "lower_ge",
            "difference",
            "corrupt_modality",
            "all_available_modalities",
        ),
    },
    "G5": {
        "action_C_vs_A_noninferiority": ("lower_ge", "difference", "C", "A"),
        "action_C_vs_H": ("lower_gt", "difference", "C", "H"),
        "action_C_vs_M": ("lower_gt", "difference", "C", "M"),
        "action_C_vs_O": ("lower_gt", "difference", "C", "O"),
        "row_intervention_effect": (
            "lower_gt",
            "difference",
            "C_factual",
            "C_row_intervened",
        ),
        "action_convergence_regression": (
            "upper_le",
            "difference",
            "C_convergence_time",
            "A_convergence_time",
        ),
    },
    "G6": {
        "calvin_success_C_vs_A": ("lower_gt", "difference", "C", "A"),
        "calvin_success_C_vs_O": ("lower_gt", "difference", "C", "O"),
        "calvin_recovery_C_vs_O": ("lower_gt", "difference", "C", "O"),
        "row_intervention_closed_loop_effect": (
            "lower_gt",
            "difference",
            "C_factual",
            "C_row_intervened",
        ),
    },
}
EMPIRICAL_COMPARISON_RULES = {
    gate: {name: spec[0] for name, spec in comparisons.items()}
    for gate, comparisons in EMPIRICAL_COMPARISON_SPECS.items()
}

EMPIRICAL_REQUIRED_CHECKS = {
    "G2": {
        "heldout_split_only",
        "object_and_no_object_scored",
        "dense_fallback_scored_separately",
        "task_grounding_scored",
        "no_row_collapse",
    },
    "G3": {
        "registered_occlusion_lengths_covered",
        "state_ages_1_8_32_64_128_covered",
        "identity_switches_scored",
        "existence_calibration_scored",
        "reidentification_scored",
    },
    "G4": {
        "independent_target_support_only",
        "random_current_grid_excluded_from_binding_claim",
        "within_scene_hard_negatives_scored",
        "missing_modality_control_passed",
        "corrupt_modality_control_passed",
    },
    "G5": {
        "complete_curves_retained",
        "auc_and_time_to_threshold_scored",
        "matched_data_order_optimizer_and_noise",
        "raw_history_compute_memory_latency_matched",
        "gradient_conflicts_reported",
        "row_interventions_scored",
    },
    "G6": {
        "paired_calvin_sequences",
        "task_and_episode_units_retained",
        "recovery_trials_scored",
        "row_interventions_scored",
        "reset_and_session_isolation_passed",
    },
}

PROTOCOL_FIELDS = {
    "criteria_path",
    "criteria_sha256",
    "dataset_manifest_path",
    "dataset_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "evaluation_plan_path",
    "evaluation_plan_sha256",
    "preregistered_before_evaluation",
}
DESIGN_FIELDS = {
    "arms",
    "paired_seed_count",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence_level",
    "top_level_unit",
    "nested_units",
    "aggregation",
    "frames_treated_as_independent",
}
COMPARISON_FIELDS = {
    "name",
    "estimate",
    "ci_lower",
    "ci_upper",
    "acceptance_bound",
    "bound_rule",
    "sample_count",
    "paired",
    "passed",
}
OBSERVATION_REFERENCE_FIELDS = {"schema", "path", "sha256", "record_count"}
EMPIRICAL_REPORT_FIELDS = {
    "schema",
    "status",
    "gate",
    "subject",
    "protocol",
    "design",
    "observations",
    "comparisons",
    "checks",
    "failures",
    "long_training_authorized",
}
_RAW_FIELDS = {
    "schema",
    "gate",
    "subject",
    "protocol",
    "design",
    "check_evidence",
    "producer",
    "records",
}
_RECORD_FIELDS = {
    "comparison",
    "seed",
    "task",
    "episode",
    "candidate_label",
    "candidate",
    "reference_label",
    "reference",
}
EMPIRICAL_EVALUATION_PLAN_FIELDS = {
    "schema",
    "gate",
    "design",
    "metric_config",
    "acceptance_bounds",
    "required_checks",
}
_CHECK_EVIDENCE_FIELDS = {"path", "sha256"}
_SUBJECT_FIELDS = {
    "input_full_report_sha256",
    "saved_global_step",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
}


def _exact_dict(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen schema")
    return cast(dict[str, Any], value)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_hashed_file(path_value: object, digest_value: object, *, name: str) -> Path:
    path = Path(path_value) if isinstance(path_value, str) else None
    digest = _sha256(digest_value, name=f"{name} sha256")
    if (
        path is None
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or _file_sha256(path) != digest
    ):
        raise ValueError(f"{name} differs from its hash-bound real file")
    return path


def _validate_protocol(value: object, *, gate: str) -> dict[str, Any]:
    protocol = _exact_dict(value, name=f"{gate} protocol", fields=PROTOCOL_FIELDS)
    if protocol["preregistered_before_evaluation"] is not True:
        raise ValueError(f"{gate} evaluation was not preregistered")
    paths = [
        _real_hashed_file(
            protocol[f"{stem}_path"],
            protocol[f"{stem}_sha256"],
            name=f"{gate} {stem.replace('_', ' ')}",
        )
        for stem in ("criteria", "dataset_manifest", "split_manifest", "evaluation_plan")
    ]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{gate} protocol artifacts must be distinct files")
    return protocol


def _validate_subject(value: object, *, gate: str) -> dict[str, Any]:
    subject = _exact_dict(value, name=f"{gate} subject", fields=_SUBJECT_FIELDS)
    for field in _SUBJECT_FIELDS - {"saved_global_step"}:
        _sha256(subject[field], name=f"{gate} subject {field}")
    step = subject["saved_global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"{gate} subject saved step must be positive")
    return subject


def _validate_design(value: object, *, gate: str) -> dict[str, Any]:
    design = _exact_dict(value, name=f"{gate} design", fields=DESIGN_FIELDS)
    if design["arms"] != list(EMPIRICAL_REQUIRED_ARMS[gate]):
        raise ValueError(f"{gate} comparison arms differ from the frozen design")
    seeds = design["paired_seed_count"]
    replicates = design["bootstrap_replicates"]
    bootstrap_seed = design["bootstrap_seed"]
    confidence = _finite(design["confidence_level"], name=f"{gate} confidence level")
    if (
        isinstance(seeds, bool)
        or not isinstance(seeds, int)
        or seeds < 5
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1_000
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError(f"{gate} paired-seed or bootstrap design is invalid")
    if not math.isclose(confidence, 0.95, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{gate} confidence level must remain 0.95")
    if (
        design["top_level_unit"] != "seed"
        or design["nested_units"] != ["task", "episode"]
        or design["aggregation"] != "equal_seed_task_episode_mean"
        or design["frames_treated_as_independent"] is not False
    ):
        raise ValueError(f"{gate} confidence intervals are not paired hierarchical intervals")
    return design


def validate_empirical_metric_config(value: object, *, gate: str) -> dict[str, Any]:
    """Validate gate-specific metric choices frozen before released-weight evaluation."""

    if gate not in EMPIRICAL_COMPARISON_SPECS or not isinstance(value, dict):
        raise ValueError(f"{gate} metric configuration is malformed")
    expected = {"action_loss_threshold"} if gate == "G5" else set()
    if set(value) != expected:
        raise ValueError(f"{gate} metric configuration differs from the frozen schema")
    if gate == "G5":
        threshold = _finite(
            value["action_loss_threshold"],
            name="G5 action loss threshold",
        )
        if threshold <= 0:
            raise ValueError("G5 action loss threshold must be positive")
        return {"action_loss_threshold": threshold}
    return {}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def _hierarchical_mean(grouped: dict[int, dict[str, list[float]]]) -> float:
    seed_means = []
    for tasks in grouped.values():
        seed_means.append(statistics.fmean(statistics.fmean(values) for values in tasks.values()))
    return statistics.fmean(seed_means)


def _hierarchical_interval(
    grouped: dict[int, dict[str, list[float]]],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    estimate = _hierarchical_mean(grouped)
    seeds = sorted(grouped)
    rng = random.Random(bootstrap_seed)
    samples: list[float] = []
    for _ in range(replicates):
        seed_values: list[float] = []
        for selected_seed in rng.choices(seeds, k=len(seeds)):
            tasks = grouped[selected_seed]
            task_names = sorted(tasks)
            task_values: list[float] = []
            for selected_task in rng.choices(task_names, k=len(task_names)):
                episodes = tasks[selected_task]
                task_values.append(statistics.fmean(rng.choices(episodes, k=len(episodes))))
            seed_values.append(statistics.fmean(task_values))
        samples.append(statistics.fmean(seed_values))
    tail = (1.0 - confidence_level) / 2.0
    return estimate, _percentile(samples, tail), _percentile(samples, 1.0 - tail)


def _comparison_passes(*, lower: float, upper: float, bound: float, rule: str) -> bool:
    if rule == "lower_gt":
        return lower > bound
    if rule == "lower_ge":
        return lower >= bound
    if rule == "upper_lt":
        return upper < bound
    if rule == "upper_le":
        return upper <= bound
    raise ValueError("comparison bound rule is unsupported")


def build_empirical_gate_report_from_observations(
    observations_path: Path,
    *,
    report_schema: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute one complete G2-G6 report from immutable episode observations."""

    path = observations_path.resolve()
    if observations_path.is_symlink() or not observations_path.is_file():
        raise ValueError("empirical observations must be one real file")
    payload = observations_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, name="empirical observations"
    ):
        raise ValueError("empirical observations differ from their expected digest")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("empirical observations are not valid JSON") from error
    raw = _exact_dict(raw, name="empirical observations", fields=_RAW_FIELDS)
    gate = raw["gate"]
    if gate not in EMPIRICAL_COMPARISON_SPECS:
        raise ValueError("empirical observation gate is unsupported")
    if raw["schema"] != EMPIRICAL_OBSERVATIONS_SCHEMA:
        raise ValueError("empirical observation schema changed")
    if report_schema != EMPIRICAL_REPORT_SCHEMAS[gate]:
        raise ValueError("empirical report schema differs from the gate")

    subject = _validate_subject(raw["subject"], gate=gate)
    protocol = _validate_protocol(raw["protocol"], gate=gate)
    design = _validate_design(raw["design"], gate=gate)
    plan_path = Path(cast(str, protocol["evaluation_plan_path"]))
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("empirical evaluation plan is not valid JSON") from error
    plan = _exact_dict(
        plan,
        name=f"{gate} evaluation plan",
        fields=EMPIRICAL_EVALUATION_PLAN_FIELDS,
    )
    if (
        plan["schema"] != EMPIRICAL_EVALUATION_PLAN_SCHEMA
        or plan["gate"] != gate
        or plan["design"] != design
        or plan["required_checks"] != sorted(EMPIRICAL_REQUIRED_CHECKS[gate])
    ):
        raise ValueError(f"{gate} evaluation plan differs from the observation design")
    validate_empirical_metric_config(plan["metric_config"], gate=gate)
    bounds = _exact_dict(
        plan["acceptance_bounds"],
        name=f"{gate} acceptance bounds",
        fields=set(EMPIRICAL_COMPARISON_SPECS[gate]),
    )
    bounds = {
        name: _finite(value, name=f"{gate} {name} acceptance bound")
        for name, value in bounds.items()
    }

    # Import lazily to preserve a one-way runtime dependency: producers use the
    # frozen statistical schemas, while the report builder only invokes a
    # producer when validating detached evidence.
    from picf_next.lingbot_native.empirical_producers import (
        build_empirical_observations_from_producer,
        validate_producer_reference,
    )

    producer_path, producer_sha256 = validate_producer_reference(raw["producer"])
    if producer_path == path or producer_path in {
        Path(cast(str, protocol[f"{stem}_path"])).resolve()
        for stem in ("criteria", "dataset_manifest", "split_manifest", "evaluation_plan")
    }:
        raise ValueError(f"{gate} producer must be distinct from protocol and observations")
    recomputed = build_empirical_observations_from_producer(
        producer_path,
        expected_sha256=producer_sha256,
    )
    if recomputed != raw:
        raise ValueError(f"{gate} observations were not recomputed from the hash-bound producer")

    check_evidence = _exact_dict(
        raw["check_evidence"],
        name=f"{gate} check evidence",
        fields=EMPIRICAL_REQUIRED_CHECKS[gate],
    )
    protocol_paths = {
        Path(cast(str, protocol[f"{stem}_path"])).resolve()
        for stem in ("criteria", "dataset_manifest", "split_manifest", "evaluation_plan")
    }
    check_payloads: dict[tuple[Path, str], dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for check_name, raw_reference in check_evidence.items():
        reference = _exact_dict(
            raw_reference,
            name=f"{gate} {check_name} check evidence",
            fields=_CHECK_EVIDENCE_FIELDS,
        )
        check_path = _real_hashed_file(
            reference["path"],
            reference["sha256"],
            name=f"{gate} {check_name} check evidence",
        )
        if check_path.resolve() in protocol_paths or check_path.resolve() == path:
            raise ValueError(
                f"{gate} check evidence must be distinct from protocol and observations"
            )
        cache_key = (check_path.resolve(), cast(str, reference["sha256"]))
        artifact = check_payloads.get(cache_key)
        if artifact is None:
            try:
                decoded = json.loads(check_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{gate} check evidence is not valid JSON") from error
            if not isinstance(decoded, dict) or not isinstance(decoded.get("checks"), dict):
                raise ValueError(f"{gate} check evidence has no check mapping")
            artifact = cast(dict[str, Any], decoded)
            check_payloads[cache_key] = artifact
        passed = artifact["checks"].get(check_name)
        if not isinstance(passed, bool):
            raise ValueError(f"{gate} {check_name} is absent from its evaluator artifact")
        checks[check_name] = passed
    records = raw["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{gate} empirical observations are empty")

    grouped: dict[str, dict[int, dict[str, list[float]]]] = {
        name: defaultdict(lambda: defaultdict(list)) for name in EMPIRICAL_COMPARISON_SPECS[gate]
    }
    observed_units: dict[str, set[tuple[int, str, str]]] = {
        name: set() for name in EMPIRICAL_COMPARISON_SPECS[gate]
    }
    for raw_record in records:
        record = _exact_dict(raw_record, name=f"{gate} observation", fields=_RECORD_FIELDS)
        comparison = record["comparison"]
        if not isinstance(comparison, str) or comparison not in EMPIRICAL_COMPARISON_SPECS[gate]:
            raise ValueError(f"{gate} observation names an unregistered comparison")
        rule, mode, candidate_label, reference_label = EMPIRICAL_COMPARISON_SPECS[gate][comparison]
        del rule
        if (
            record["candidate_label"] != candidate_label
            or record["reference_label"] != reference_label
        ):
            raise ValueError(f"{gate} {comparison} reverses or relabels a frozen comparison")
        seed = record["seed"]
        task = record["task"]
        episode = record["episode"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or not isinstance(task, str)
            or not task.strip()
            or not isinstance(episode, str)
            or not episode.strip()
        ):
            raise ValueError(f"{gate} observation hierarchy is malformed")
        unit = (seed, task, episode)
        if unit in observed_units[comparison]:
            raise ValueError(f"{gate} {comparison} duplicates one seed/task/episode unit")
        observed_units[comparison].add(unit)
        candidate = _finite(record["candidate"], name=f"{gate} {comparison} candidate")
        if mode == "difference":
            reference = _finite(record["reference"], name=f"{gate} {comparison} reference")
            effect = candidate - reference
        else:
            if record["reference"] is not None:
                raise ValueError(f"{gate} {comparison} absolute metric has a reference value")
            if not 0 <= candidate <= 1:
                raise ValueError(f"{gate} {comparison} rate or error lies outside [0, 1]")
            effect = candidate
        grouped[comparison][seed][task].append(effect)

    expected_seed_count = cast(int, design["paired_seed_count"])
    seed_sets = []
    comparisons = []
    for index, (name, (rule, _mode, _candidate, _reference)) in enumerate(
        EMPIRICAL_COMPARISON_SPECS[gate].items()
    ):
        comparison_groups = dict(grouped[name])
        seeds = set(comparison_groups)
        if len(seeds) != expected_seed_count:
            raise ValueError(f"{gate} {name} does not cover every paired seed")
        seed_layouts = [
            {task: len(episodes) for task, episodes in sorted(comparison_groups[seed].items())}
            for seed in sorted(seeds)
        ]
        if any(layout != seed_layouts[0] for layout in seed_layouts[1:]):
            raise ValueError(f"{gate} {name} uses an unbalanced task/episode plan across seeds")
        seed_sets.append(seeds)
        derived_seed = int.from_bytes(
            hashlib.sha256(f"{design['bootstrap_seed']}:{gate}:{index}:{name}".encode()).digest()[
                :8
            ],
            "big",
        )
        estimate, lower, upper = _hierarchical_interval(
            comparison_groups,
            replicates=cast(int, design["bootstrap_replicates"]),
            bootstrap_seed=derived_seed,
            confidence_level=cast(float, design["confidence_level"]),
        )
        bound = bounds[name]
        comparisons.append(
            {
                "name": name,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "acceptance_bound": bound,
                "bound_rule": rule,
                "sample_count": len(observed_units[name]),
                "paired": True,
                "passed": _comparison_passes(
                    lower=lower,
                    upper=upper,
                    bound=bound,
                    rule=rule,
                ),
            }
        )
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError(f"{gate} comparisons do not share the same paired seeds")

    failures = [
        f"comparison:{comparison['name']}" for comparison in comparisons if not comparison["passed"]
    ]
    failures.extend(f"check:{name}" for name, passed in checks.items() if not passed)
    failures.sort()
    return {
        "schema": report_schema,
        "status": "PASS" if not failures else "FAIL",
        "gate": gate,
        "subject": subject,
        "protocol": protocol,
        "design": design,
        "observations": {
            "schema": EMPIRICAL_OBSERVATIONS_SCHEMA,
            "path": str(path),
            "sha256": digest,
            "record_count": len(records),
        },
        "comparisons": comparisons,
        "checks": checks,
        "failures": failures,
        "long_training_authorized": False,
    }

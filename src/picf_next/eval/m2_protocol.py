"""Host-neutral source-split and paired-intervention utilities for M2 audits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _epoch_order(keys: Sequence[str], *, seed: int, epoch: int) -> list[str]:
    return sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{seed}:{epoch}:{key}".encode()).digest(),
    )


def paired_count_support_plan(
    *,
    base_keys: Sequence[str],
    treatment_supplement: Sequence[str],
    control_supplement: Sequence[str],
    seed: int,
    steps: int,
    batch_size: int,
) -> tuple[list[list[str]], list[list[str]], dict[str, Any]]:
    """Build paired batches whose arms differ only at supplement positions."""

    if (
        not base_keys
        or not treatment_supplement
        or len(treatment_supplement) != len(control_supplement)
        or len(set(base_keys)) != len(base_keys)
        or len(set(treatment_supplement)) != len(treatment_supplement)
        or len(set(control_supplement)) != len(control_supplement)
        or set(base_keys) & set(treatment_supplement)
        or set(base_keys) & set(control_supplement)
        or set(treatment_supplement) & set(control_supplement)
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps <= 0
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("paired count-support key sets or dimensions are invalid")

    abstract = [f"base:{key}" for key in base_keys] + [
        f"supplement:{index:08d}" for index in range(len(treatment_supplement))
    ]
    required = steps * batch_size
    ordered: list[str] = []
    epoch = 0
    while len(ordered) < required:
        ordered.extend(_epoch_order(abstract, seed=seed, epoch=epoch))
        epoch += 1

    treatment: list[list[str]] = []
    control: list[list[str]] = []
    abstract_batches: list[list[str]] = []
    for start in range(0, required, batch_size):
        batch = ordered[start : start + batch_size]
        treatment_batch = []
        control_batch = []
        for identifier in batch:
            kind, payload = identifier.split(":", 1)
            if kind == "base":
                treatment_batch.append(payload)
                control_batch.append(payload)
            elif kind == "supplement":
                index = int(payload)
                treatment_batch.append(treatment_supplement[index])
                control_batch.append(control_supplement[index])
            else:
                raise RuntimeError("unknown abstract count-support sample")
        abstract_batches.append(batch)
        treatment.append(treatment_batch)
        control.append(control_batch)

    return (
        treatment,
        control,
        {
            "schema": "picf-next.molmoact2-m2-paired-count-support-plan.v1",
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "base_sample_count": len(base_keys),
            "supplement_sample_count": len(treatment_supplement),
            "abstract_batches": abstract_batches,
            "abstract_batches_sha256": _canonical_sha256(abstract_batches),
            "treatment_batches_sha256": _canonical_sha256(treatment),
            "control_batches_sha256": _canonical_sha256(control),
        },
    )


def low_count_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate the preregistered seven/eight-object evaluation stratum."""

    selected = [row for row in rows if int(row["target_object_count"]) <= 8]
    if not selected:
        raise ValueError("count-support evaluation contains no seven/eight-object samples")
    return {
        "sample_count": len(selected),
        "count_mae": sum(
            abs(int(row["predicted_object_count"]) - int(row["target_object_count"]))
            for row in selected
        )
        / len(selected),
        "exact_count_accuracy": sum(bool(row["exact_count"]) for row in selected) / len(selected),
        "mean_object_dice": sum(float(row["mean_object_dice"]) for row in selected) / len(selected),
        "ownership_accuracy": sum(float(row["ownership_accuracy"]) for row in selected)
        / len(selected),
    }


def language_samples(dataset: Any) -> list[Any]:
    """Return unique CALVIN language transitions in source-stable order."""

    samples = [dataset[index] for index in range(len(dataset))]
    samples.sort(key=lambda sample: (sample.record.task_index, sample.transition_index))
    if not samples or len({sample.sample_key for sample in samples}) != len(samples):
        raise RuntimeError("external validation language samples are empty or non-unique")
    return samples


def unique_source_keys(
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
) -> list[str]:
    """Choose one stable language annotation for every physical source frame."""

    first_by_global: dict[int, str] = {}
    for key, (_tokens, _valid, record) in sorted(
        cache.items(),
        key=lambda item: (
            int(item[1][2]["segment_index"]),
            int(item[1][2]["transition_index"]),
        ),
    ):
        first_by_global.setdefault(int(record["global_index"]), key)
    return sorted(first_by_global.values(), key=lambda key: int(cache[key][2]["global_index"]))


def task_intervention_check(
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Prove that task text does not alter same-source dense observations."""

    import torch

    keys_by_global: dict[int, list[str]] = defaultdict(list)
    for key, (_tokens, _valid, record) in cache.items():
        keys_by_global[int(record["global_index"])].append(key)
    pair_count = 0
    maximum_error = 0.0
    all_exact = True
    for keys in keys_by_global.values():
        if len(keys) < 2:
            continue
        reference_tokens, reference_valid, reference_record = cache[keys[0]]
        for key in keys[1:]:
            tokens, valid, record = cache[key]
            if reference_record["source_sensor_sha256"] != record["source_sensor_sha256"]:
                raise RuntimeError("overlapping validation annotations changed source sensors")
            difference = (reference_tokens.float() - tokens.float()).abs()
            maximum_error = max(maximum_error, float(difference.max().item()))
            exact = torch.equal(reference_tokens, tokens) and torch.equal(reference_valid, valid)
            all_exact = all_exact and exact
            pair_count += 1
    if pair_count <= 0 or not all_exact:
        raise RuntimeError("external validation task intervention is absent or non-exact")
    return {
        "schema": "picf-next.molmoact2-m2-external-task-intervention.v1",
        "pair_count": pair_count,
        "all_dense_features_exact": all_exact,
        "maximum_absolute_error": maximum_error,
        "task_text_enters_trainable_m2_graph": False,
    }


def group_by_target_count(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate hard-count and representation metrics by target cardinality."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["target_object_count"])].append(row)
    return {
        str(count): {
            "sample_count": len(group),
            "predicted_count_mean": sum(float(row["predicted_object_count"]) for row in group)
            / len(group),
            "count_mae": sum(
                abs(float(row["predicted_object_count"]) - float(row["target_object_count"]))
                for row in group
            )
            / len(group),
            "exact_count_accuracy": sum(bool(row["exact_count"]) for row in group) / len(group),
            "mean_object_dice": sum(float(row["mean_object_dice"]) for row in group) / len(group),
            "ownership_accuracy": sum(float(row["ownership_accuracy"]) for row in group)
            / len(group),
        }
        for count, group in sorted(grouped.items())
    }

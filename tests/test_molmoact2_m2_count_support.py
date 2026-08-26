from __future__ import annotations

import pytest

pytest.importorskip("torch")

from picf_next.eval.m2_protocol import (  # noqa: E402
    low_count_metrics,
    paired_count_support_plan,
)


def test_paired_count_support_plan_changes_only_supplement_slots() -> None:
    treatment, control, report = paired_count_support_plan(
        base_keys=("base-a", "base-b"),
        treatment_supplement=("low-a", "low-b"),
        control_supplement=("high-a", "high-b"),
        seed=17,
        steps=5,
        batch_size=2,
    )

    assert len(treatment) == len(control) == 5
    assert report["base_sample_count"] == 2
    assert report["supplement_sample_count"] == 2
    for abstract_batch, treatment_batch, control_batch in zip(
        report["abstract_batches"],
        treatment,
        control,
        strict=True,
    ):
        for abstract, treatment_key, control_key in zip(
            abstract_batch,
            treatment_batch,
            control_batch,
            strict=True,
        ):
            if abstract.startswith("base:"):
                assert treatment_key == control_key == abstract.removeprefix("base:")
            else:
                index = int(abstract.removeprefix("supplement:"))
                assert treatment_key == ("low-a", "low-b")[index]
                assert control_key == ("high-a", "high-b")[index]

    repeated = paired_count_support_plan(
        base_keys=("base-a", "base-b"),
        treatment_supplement=("low-a", "low-b"),
        control_supplement=("high-a", "high-b"),
        seed=17,
        steps=5,
        batch_size=2,
    )
    assert repeated == (treatment, control, report)


def test_paired_count_support_plan_rejects_source_overlap() -> None:
    with pytest.raises(ValueError, match="invalid"):
        paired_count_support_plan(
            base_keys=("base",),
            treatment_supplement=("same",),
            control_supplement=("same",),
            seed=1,
            steps=1,
            batch_size=1,
        )


def test_low_count_metrics_exclude_nine_and_ten_object_rows() -> None:
    metrics = low_count_metrics(
        (
            {
                "target_object_count": 7,
                "predicted_object_count": 8,
                "exact_count": False,
                "mean_object_dice": 0.6,
                "ownership_accuracy": 0.7,
            },
            {
                "target_object_count": 8,
                "predicted_object_count": 8,
                "exact_count": True,
                "mean_object_dice": 0.8,
                "ownership_accuracy": 0.9,
            },
            {
                "target_object_count": 9,
                "predicted_object_count": 2,
                "exact_count": False,
                "mean_object_dice": 0.0,
                "ownership_accuracy": 0.0,
            },
        )
    )

    assert metrics == {
        "sample_count": 2,
        "count_mae": 0.5,
        "exact_count_accuracy": 0.5,
        "mean_object_dice": pytest.approx(0.7),
        "ownership_accuracy": pytest.approx(0.8),
    }

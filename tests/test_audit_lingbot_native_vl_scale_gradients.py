from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from tools.audit_lingbot_native_vl_scale_gradients import (
    ADR125_RETENTION_GRADIENT_STEP_INDICES,
    _parse_step_indices,
    _retention_gate_status,
    _retention_step_summary,
    _step_summary,
    _validate_retention_step_indices,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools/audit_lingbot_native_vl_scale_gradients.py"


def test_parse_step_indices_accepts_one_frozen_sorted_set() -> None:
    assert _parse_step_indices("0,3,31") == (0, 3, 31)


@pytest.mark.parametrize("value", ("", "1,1", "2,1", "-1", "x"))
def test_parse_step_indices_rejects_ambiguous_sequences(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_step_indices(value)


def test_retention_gradient_indices_are_frozen_before_execution() -> None:
    _validate_retention_step_indices(ADR125_RETENTION_GRADIENT_STEP_INDICES)
    with pytest.raises(ContractError, match="differ from ADR-125"):
        _validate_retention_step_indices((0, 1, 2, 3))


def test_retention_gradient_audit_uses_the_same_bounded_processor_contract() -> None:
    source = _TOOL.read_text()
    processor = source.index("retention_processor = build_processor")
    configure = source.index("configure_native_processor_area_budget(", processor)
    preprocess = source.index("retention_batch = build_native_vl_grounding_batch(")
    validate = source.index("retention_grid_budget = validate_native_processor_record_grid(")
    transfer = source.index("retention_batch = retention_batch.to(", validate)
    forward = source.index("retention_loss = run_native_vl_grounding_forward(")

    assert processor < configure < preprocess < validate < transfer < forward
    assert '"processor": retention_processor_contract' in source


def test_step_summary_preserves_directional_failures() -> None:
    def surface(cosine: float, *, descends8: bool = True) -> dict[str, object]:
        return {
            "cosine": cosine,
            "mean_gradient_descends_lattice8": descends8,
            "mean_gradient_descends_lattice14": True,
            "parameter_tensor_negative_dot_mass_fraction": 0.25,
        }

    report = _step_summary(
        [
            {"alignment": {"global": surface(0.5), "groups": {"visual_merger": surface(-0.2)}}},
            {
                "alignment": {
                    "global": surface(-0.1),
                    "groups": {"visual_merger": surface(-0.4, descends8=False)},
                }
            },
        ]
    )
    global_report = report["global"]
    merger_report = report["visual_merger"]
    assert isinstance(global_report, dict)
    assert isinstance(merger_report, dict)
    assert global_report["negative_cosine_step_count"] == 1
    assert merger_report["negative_cosine_step_count"] == 2
    assert merger_report["lattice8_mean_descent_failure_count"] == 1


def test_retention_step_summary_preserves_weighted_directional_failures() -> None:
    def surface(
        cosine: float,
        *,
        descends_first: bool = True,
        descends_second: bool = True,
    ) -> dict[str, object]:
        return {
            "cosine": cosine,
            "mixed_gradient_descends_first_objective": descends_first,
            "mixed_gradient_descends_second_objective": descends_second,
            "parameter_tensor_negative_dot_mass_fraction": 0.5,
        }

    report = _retention_step_summary(
        [
            {
                "public_vl_retention": {
                    "alignment": {
                        "global": surface(0.2),
                        "groups": {"visual_merger": surface(-0.3, descends_second=False)},
                    }
                }
            },
            {
                "public_vl_retention": {
                    "alignment": {
                        "global": surface(-0.1, descends_first=False),
                        "groups": {"visual_merger": surface(-0.4)},
                    }
                }
            },
        ]
    )

    assert report["global"]["first_objective_descent_failure_count"] == 1
    assert report["global"]["negative_cosine_step_count"] == 1
    assert report["visual_merger"]["second_objective_descent_failure_count"] == 1
    assert _retention_gate_status(report) == "FAIL"

    report["global"]["first_objective_descent_failure_count"] = 0
    report["global"]["second_objective_descent_failure_count"] = 0
    assert _retention_gate_status(report) == "PASS"

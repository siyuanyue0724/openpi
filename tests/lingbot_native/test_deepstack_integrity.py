from __future__ import annotations

import copy
import types

import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.lingbot_native.deepstack_integrity import (
    DEEPSTACK_INTEGRITY_SCHEMA,
    deepstack_integrity_gates,
    tensor_difference_summary,
    tensor_numeric_summary,
    validate_deepstack_integrity_report,
)
from tools.probe_lingbot_deepstack_integrity import _DeepStackTrace

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _difference(*, bitwise_equal: bool, delta_rms: float, relative_rms: float) -> dict:
    return {
        "bitwise_equal": bitwise_equal,
        "all_finite": True,
        "delta_rms": delta_rms,
        "relative_rms": relative_rms,
        "max_abs": delta_rms,
        "reference_sha256": _DIGEST_A,
        "candidate_sha256": _DIGEST_A if bitwise_equal else _DIGEST_B,
    }


def _report() -> dict:
    runs = {}
    for run_name, mode in (
        ("normal", "normal"),
        ("normal_repeat", "normal"),
        ("zeroed", "zeroed"),
    ):
        injections = []
        for layer in range(2):
            injections.append(
                {
                    "layer_index": layer,
                    "visual_position_count": 8,
                    "feature": {
                        "sha256": str(layer) * 64,
                        "all_finite": True,
                        "rms": 0.5,
                        "std": 0.25,
                        "nonzero_fraction": 1.0,
                    },
                    "visual_delta_rms": 0.0 if mode == "zeroed" else 0.5,
                    "visual_expected_max_abs_error": None if mode == "zeroed" else 0.0,
                    "nonvisual_max_abs_delta": 0.0,
                }
            )
        runs[run_name] = {
            "mode": mode,
            "injections": injections,
        }
    value = {
        "schema": DEEPSTACK_INTEGRITY_SCHEMA,
        "expected_deepstack_count": 2,
        "runs": runs,
        "comparisons": {
            "normal_repeat": {
                "posterior_rows": _difference(
                    bitwise_equal=True,
                    delta_rms=0.0,
                    relative_rms=0.0,
                ),
                "relation_ownership": _difference(
                    bitwise_equal=True,
                    delta_rms=0.0,
                    relative_rms=0.0,
                ),
            },
            "normal_zeroed": {
                "posterior_rows": _difference(
                    bitwise_equal=False,
                    delta_rms=0.01,
                    relative_rms=0.02,
                ),
                "relation_ownership": _difference(
                    bitwise_equal=False,
                    delta_rms=0.001,
                    relative_rms=0.01,
                ),
            },
        },
    }
    value["gates"] = deepstack_integrity_gates(value)
    value["failures"] = sorted(name for name, passed in value["gates"].items() if not passed)
    value["status"] = "PASS" if not value["failures"] else "FAIL"
    return value


def test_tensor_statistics_and_paired_difference_preserve_exact_identity() -> None:
    tensor = torch.tensor([[1.0, -2.0], [0.0, 3.0]], dtype=torch.bfloat16)
    summary = tensor_numeric_summary(tensor)
    identical = tensor_difference_summary(tensor, tensor.clone())
    changed = tensor_difference_summary(tensor, tensor + 1)

    assert summary["shape"] == [2, 2]
    assert summary["all_finite"] is True
    assert summary["nonzero_fraction"] == 0.75
    assert identical["bitwise_equal"] is True
    assert identical["delta_rms"] == 0.0
    assert changed["bitwise_equal"] is False
    assert changed["delta_rms"] == pytest.approx(1.0)


def test_deepstack_report_requires_nonzero_features_and_downstream_effect() -> None:
    accepted = _report()
    assert validate_deepstack_integrity_report(accepted)["status"] == "PASS"

    zero_feature = copy.deepcopy(accepted)
    zero_feature["runs"]["normal"]["injections"][0]["feature"]["rms"] = 0.0
    zero_feature["gates"] = deepstack_integrity_gates(zero_feature)
    zero_feature["failures"] = sorted(
        name for name, passed in zero_feature["gates"].items() if not passed
    )
    zero_feature["status"] = "FAIL"
    assert (
        validate_deepstack_integrity_report(zero_feature)["gates"]["feature_tensors_finite_nonzero"]
        is False
    )

    no_effect = copy.deepcopy(accepted)
    no_effect["comparisons"]["normal_zeroed"]["posterior_rows"] = _difference(
        bitwise_equal=True,
        delta_rms=0.0,
        relative_rms=0.0,
    )
    no_effect["comparisons"]["normal_zeroed"]["relation_ownership"] = _difference(
        bitwise_equal=True,
        delta_rms=0.0,
        relative_rms=0.0,
    )
    no_effect["gates"] = deepstack_integrity_gates(no_effect)
    no_effect["failures"] = sorted(
        name for name, passed in no_effect["gates"].items() if not passed
    )
    no_effect["status"] = "FAIL"
    assert (
        validate_deepstack_integrity_report(no_effect)["gates"]["deepstack_reaches_picf_posterior"]
        is False
    )


def test_deepstack_report_rejects_persisted_decision_tampering() -> None:
    tampered = _report()
    tampered["gates"]["normal_repeat_deterministic"] = False
    with pytest.raises(ContractError, match="persisted gates"):
        validate_deepstack_integrity_report(tampered)


class _FakeDeepStackHost:
    def _apply_deepstack(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states[visual_pos_masks, :] = hidden_states[
            visual_pos_masks, :
        ] + deepstack_visual_embeds[layer_idx].to(hidden_states)
        return hidden_states


def test_runtime_trace_observes_exact_visual_injection_and_zero_ablation() -> None:
    host = _FakeDeepStackHost()
    mask = torch.tensor([[True, False, True, False]])
    feature = [torch.full((2, 3), 0.5)]

    normal_hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    with _DeepStackTrace(torch_module=torch, module=host, mode="normal") as trace:
        host._apply_deepstack(normal_hidden, 0, mask, feature)
    assert trace.injections[0]["visual_delta_rms"] == pytest.approx(0.5)
    assert trace.injections[0]["visual_expected_max_abs_error"] == 0.0
    assert trace.injections[0]["nonvisual_max_abs_delta"] == 0.0

    zero_hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    before = zero_hidden.clone()
    with _DeepStackTrace(torch_module=torch, module=host, mode="zeroed") as trace:
        host._apply_deepstack(zero_hidden, 0, mask, feature)
    assert torch.equal(zero_hidden, before)
    assert trace.injections[0]["visual_delta_rms"] == 0.0

    assert isinstance(host._apply_deepstack, types.MethodType)

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.adr175_validation import (
    ADR175_AMBIGUOUS_TASKS,
    ADR175_ARM_REPORT_SCHEMA,
    ADR175_ARMS,
    ADR175_EXACT_TASK_TARGETS,
    ADR175_MILESTONES,
    ADR175_TOTAL_STEPS,
    ADR175ArmReport,
    ADR175ValidationResult,
    canonical_json_bytes,
    canonical_sha256,
    seal_adr175_arm_report,
    validate_adr175_matched_three_arm,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CLI = _REPOSITORY_ROOT / "tools" / "validate_adr175_matched_three_arm.py"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _shared_contract() -> dict[str, object]:
    return {
        "broad_support_contract_sha256": _digest("broad-support"),
        "broad_support_contract_file_sha256": _digest("broad-support-file"),
        "matched_arm_input_sha256": _digest("matched-arm-input"),
        "dataset_contract_sha256": _digest("dataset"),
        "physical_sidecar_manifest_sha256": _digest("physical-sidecar"),
        "stream_plan_sha256": _digest("stream"),
        "representation_split_sha256": _digest("split"),
        "evaluation_plan_sha256": _digest("evaluation"),
        "shared_initialization_sha256": _digest("shared-initialization"),
        "shared_optimizer_contract_sha256": _digest("shared-optimizer"),
        "source_commit": "2838c1862bbec1ea47942fb61512130f635eb595",
        "source_patch_sha256": _digest("source-patch"),
        "patched_source_sha256": _digest("patched-source"),
        "implementation_sha256": _digest("implementation"),
        "checkpoint_contract_sha256": _digest("checkpoint-contract"),
        "processor_contract_sha256": _digest("processor-contract"),
        "objective_sha256": _digest("objective"),
        "vision_geometry_sha256": _digest("vision-geometry"),
        "runtime_contract_sha256": _digest("runtime-contract"),
        "total_steps": ADR175_TOTAL_STEPS,
    }


def _step_receipts() -> list[dict[str, object]]:
    return [
        {
            "global_step": step,
            "execution_input_sha256": _digest(f"execution-input-{step}"),
            "sample_sha256": _digest(f"sample-{step}"),
            "action_target_sha256": _digest(f"action-{step}"),
            "noise_sha256": _digest(f"noise-{step}"),
            "time_sha256": _digest(f"time-{step}"),
            "prompt_sha256": _digest(f"prompt-{step}"),
        }
        for step in range(1, ADR175_TOTAL_STEPS + 1)
    ]


def _milestones(arm: str) -> list[dict[str, object]]:
    action_scale = {"lbot": 1.0, "physical-set": 1.01, "native-attention": 0.98}[arm]
    entity_offset = {"lbot": 0.0, "physical-set": 0.08, "native-attention": 0.12}[arm]
    action_curve = (1.0, 0.82, 0.64, 0.50, 0.40)
    entity_curve = (0.20, 0.28, 0.34, 0.41, 0.50)
    return [
        {
            "global_step": step,
            "posterior_adoption": None
            if arm == "lbot"
            else {
                "validation": min(0.95, 0.20 + 0.0002 * step + entity_offset),
                "heldout": min(0.95, 0.18 + 0.00018 * step + entity_offset),
            },
            "conditional_selectivity": None
            if arm == "lbot"
            else {
                "validation": -0.02 + 0.00008 * step + entity_offset,
                "heldout": -0.03 + 0.00007 * step + entity_offset,
            },
            "action_loss": {
                "validation": action_curve[index] * action_scale,
                "heldout": action_curve[index] * action_scale * 1.01,
            },
            "entity_set_score": None
            if arm == "lbot"
            else {
                "validation": min(0.99, entity_curve[index] + entity_offset),
                "heldout": min(0.99, entity_curve[index] - 0.02 + entity_offset),
            },
        }
        for index, step in enumerate(ADR175_MILESTONES)
    ]


def _exact_strata(arm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (task_key, target_identity_keys) in enumerate(ADR175_EXACT_TASK_TARGETS):
        identities = list(target_identity_keys)
        if arm == "physical-set":
            validation_score = 0.44
            heldout_score = 0.41
        else:
            positive = index < 23
            validation_score = 0.50 if positive else 0.43
            heldout_score = 0.47 if positive else 0.40
        rows.append(
            {
                "stratum_id": canonical_sha256(
                    {"task_key": task_key, "target_identity_keys": identities}
                ),
                "task_key": task_key,
                "target_identity_keys": identities,
                "validation_score": validation_score,
                "heldout_score": heldout_score,
                "validation_censored": False,
                "heldout_censored": False,
                "validation_sample_count": 1,
                "heldout_sample_count": 2,
                "validation_observable_sample_count": 1,
                "heldout_observable_sample_count": 2,
                "observability_receipt_sha256": _digest(f"observability-{task_key}"),
            }
        )
    return sorted(rows, key=lambda row: str(row["stratum_id"]))


def _bootstrap(arm: str) -> dict[str, object] | None:
    if arm != "native-attention":
        return None
    return {
        "cluster_unit": "source_episode",
        "cluster_count": 41,
        "confidence_level": 0.95,
        "resampling_scheme": "paired_global_source_episode_bayesian",
        "replicates": 10_000,
        "seed": 20260816,
        "reference_arm": "physical-set",
        "candidate_arm": "native-attention",
        "raw_estimate": 0.12,
        "raw_lower_bound": 0.03,
        "normalized_estimate": 0.16,
        "normalized_lower_bound": 0.04,
    }


def _unsigned_report(arm: str) -> dict[str, object]:
    is_lbot = arm == "lbot"
    return {
        "schema": ADR175_ARM_REPORT_SCHEMA,
        "status": "COMPLETE",
        "arm": arm,
        "raw_report_file_sha256": _digest(f"raw-report-{arm}"),
        "evaluation_evidence_sha256": _digest(f"evaluation-evidence-{arm}"),
        "picf_treatment_contract_sha256": (None if is_lbot else _digest("picf-treatment-contract")),
        "shared_contract": _shared_contract(),
        "picf_graph_sha256": None if is_lbot else _digest("picf-graph"),
        "picf_initialization_sha256": None if is_lbot else _digest("picf-init"),
        "exact_observability_sha256": (None if is_lbot else _digest("exact-observability")),
        "ambiguous_target_validity": [
            {"task_key": task_key, "target_valid": False} for task_key in ADR175_AMBIGUOUS_TASKS
        ],
        "step_receipts": _step_receipts(),
        "milestones": _milestones(arm),
        "exact_strata": None if is_lbot else _exact_strata(arm),
        "heldout_selectivity_bootstrap": _bootstrap(arm),
    }


def _valid_reports() -> list[dict[str, object]]:
    return [seal_adr175_arm_report(_unsigned_report(arm)) for arm in ADR175_ARMS]


def _reseal(report: dict[str, object]) -> dict[str, object]:
    unsigned = copy.deepcopy(report)
    unsigned.pop("artifact_sha256")
    return seal_adr175_arm_report(unsigned)


def _resign_without_semantic_validation(report: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(report)
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = canonical_sha256(unsigned)
    return value


def test_arm_report_roundtrip_and_unsigned_sealing_are_canonical() -> None:
    value = _valid_reports()[2]
    report = ADR175ArmReport.from_dict(value)

    assert report.to_dict() == value
    assert ADR175ArmReport.from_dict(report.to_dict()) == report
    assert canonical_sha256(report.to_unsigned_dict()) == report.artifact_sha256


def test_arm_report_rejects_unresigned_artifact_tamper() -> None:
    value = _valid_reports()[2]
    value["milestones"][4]["conditional_selectivity"]["heldout"] = 0.99  # type: ignore[index]

    with pytest.raises(ContractError, match="artifact SHA-256 changed"):
        ADR175ArmReport.from_dict(value)


def test_validator_accepts_complete_matched_three_arm_evidence() -> None:
    result = validate_adr175_matched_three_arm(_valid_reports())

    assert result.status == "PASS"
    assert [item.arm for item in result.arm_report_sha256] == list(ADR175_ARMS)
    assert all(gate.passed for gate in result.gates)
    assert ADR175ValidationResult.from_dict(result.to_dict()) == result
    support = next(gate for gate in result.gates if gate.name == "exact_strata_joint_support")
    assert json.loads(support.evidence_json)["jointly_positive_count"] == 23


def test_validation_result_rejects_artifact_tamper() -> None:
    result = validate_adr175_matched_three_arm(_valid_reports()).to_dict()
    result["gates"][0]["evidence"]["arms"].reverse()  # type: ignore[index]

    with pytest.raises(ContractError, match="artifact SHA-256 changed"):
        ADR175ValidationResult.from_dict(result)


def test_validator_rejects_inexact_arm_set_after_valid_resigning() -> None:
    reports = _valid_reports()
    reports[2] = copy.deepcopy(reports[1])

    with pytest.raises(ContractError, match="arm set must be exactly"):
        validate_adr175_matched_three_arm(reports)


@pytest.mark.parametrize(
    "field",
    sorted(field for field in _shared_contract() if field != "total_steps"),
)
def test_validator_rejects_every_shared_contract_identity_drift(field: str) -> None:
    reports = _valid_reports()
    reports[1]["shared_contract"][field] = _digest(f"other-{field}")  # type: ignore[index]
    reports[1] = _reseal(reports[1])

    with pytest.raises(ContractError, match="shared optimizer contract differs"):
        validate_adr175_matched_three_arm(reports)


def test_arm_report_rejects_missing_per_step_receipt() -> None:
    report = _valid_reports()[1]
    report["step_receipts"].pop(874)  # type: ignore[union-attr]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="cover every update 1..2000"):
        ADR175ArmReport.from_dict(report)


@pytest.mark.parametrize(
    "receipt_field",
    [
        "sample_sha256",
        "action_target_sha256",
        "noise_sha256",
        "time_sha256",
        "prompt_sha256",
    ],
)
def test_validator_rejects_each_per_step_receipt_drift(receipt_field: str) -> None:
    reports = _valid_reports()
    reports[1]["step_receipts"][731][receipt_field] = _digest(f"changed-{receipt_field}")  # type: ignore[index]
    reports[1] = _reseal(reports[1])

    with pytest.raises(ContractError, match="per-step sample/action/noise/time/prompt"):
        validate_adr175_matched_three_arm(reports)


@pytest.mark.parametrize("field", ["picf_graph_sha256", "picf_initialization_sha256"])
def test_validator_rejects_picf_graph_or_initialization_drift(field: str) -> None:
    reports = _valid_reports()
    reports[2][field] = _digest(f"changed-{field}")
    reports[2] = _reseal(reports[2])

    with pytest.raises(ContractError, match="PICF (graph|initialization) digests differ"):
        validate_adr175_matched_three_arm(reports)


def test_arm_report_rejects_resigned_ambiguous_target_valid_true() -> None:
    report = _valid_reports()[2]
    report["ambiguous_target_validity"][0]["target_valid"] = True  # type: ignore[index]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="target_valid=false"):
        ADR175ArmReport.from_dict(report)


def test_arm_report_rejects_resigned_missing_milestone() -> None:
    report = _valid_reports()[2]
    report["milestones"].pop(2)  # type: ignore[union-attr]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="milestones must be exactly"):
        ADR175ArmReport.from_dict(report)


def test_arm_report_rejects_conflated_or_missing_selectivity_channel() -> None:
    report = _valid_reports()[2]
    report["milestones"][2].pop("conditional_selectivity")  # type: ignore[index]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="fields differ from schema"):
        ADR175ArmReport.from_dict(report)


@pytest.mark.parametrize(
    "field",
    ["posterior_adoption", "conditional_selectivity", "entity_set_score"],
)
def test_lbot_rejects_nonnull_picf_only_milestone_metrics(field: str) -> None:
    reports = _valid_reports()
    report = reports[0]
    report["milestones"][1][field] = copy.deepcopy(reports[1]["milestones"][1][field])  # type: ignore[index]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="LBOT must publish null"):
        ADR175ArmReport.from_dict(report)


@pytest.mark.parametrize(
    "field",
    ["posterior_adoption", "conditional_selectivity", "entity_set_score"],
)
def test_treatment_rejects_null_picf_only_milestone_metrics(field: str) -> None:
    report = _valid_reports()[1]
    report["milestones"][1][field] = None  # type: ignore[index]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="require all PICF-only metrics"):
        ADR175ArmReport.from_dict(report)


def test_lbot_rejects_nonnull_exact_strata_or_bootstrap() -> None:
    reports = _valid_reports()
    for field, value, message in (
        ("exact_strata", reports[1]["exact_strata"], "LBOT exact_strata"),
        (
            "heldout_selectivity_bootstrap",
            reports[2]["heldout_selectivity_bootstrap"],
            "LBOT heldout selectivity bootstrap",
        ),
    ):
        report = copy.deepcopy(reports[0])
        report[field] = copy.deepcopy(value)
        report = _resign_without_semantic_validation(report)
        with pytest.raises(ContractError, match=message):
            ADR175ArmReport.from_dict(report)


def test_physical_set_requires_null_bootstrap_and_native_requires_bootstrap() -> None:
    reports = _valid_reports()
    physical = copy.deepcopy(reports[1])
    physical["heldout_selectivity_bootstrap"] = copy.deepcopy(
        reports[2]["heldout_selectivity_bootstrap"]
    )
    physical = _resign_without_semantic_validation(physical)
    with pytest.raises(ContractError, match="physical-set bootstrap must be null"):
        ADR175ArmReport.from_dict(physical)

    native = copy.deepcopy(reports[2])
    native["heldout_selectivity_bootstrap"] = None
    native = _resign_without_semantic_validation(native)
    with pytest.raises(ContractError, match="native-attention requires"):
        ADR175ArmReport.from_dict(native)


def test_validator_rejects_resigned_nonprotocol_exact_stratum_inventory() -> None:
    reports = _valid_reports()
    row = reports[1]["exact_strata"][0]  # type: ignore[index]
    row["task_key"] = "changed_exact_task"
    row["stratum_id"] = canonical_sha256(
        {
            "task_key": row["task_key"],
            "target_identity_keys": row["target_identity_keys"],
        }
    )
    reports[1]["exact_strata"] = sorted(  # type: ignore[index]
        reports[1]["exact_strata"],
        key=lambda item: item["stratum_id"],  # type: ignore[index]
    )
    reports[1] = _resign_without_semantic_validation(reports[1])

    with pytest.raises(ContractError, match="absent from the frozen CALVIN task protocol"):
        validate_adr175_matched_three_arm(reports)


def test_validator_rejects_only_21_jointly_positive_exact_strata() -> None:
    reports = _valid_reports()
    positive = 0
    for row in reports[2]["exact_strata"]:  # type: ignore[union-attr]
        if positive < 21:
            row["validation_score"] = 0.50
            row["heldout_score"] = 0.47
            positive += 1
        else:
            row["validation_score"] = 0.39
            row["heldout_score"] = 0.37
    reports[2] = _reseal(reports[2])

    with pytest.raises(ContractError, match="below 22/29"):
        validate_adr175_matched_three_arm(reports)


def test_validator_accepts_exactly_22_jointly_positive_exact_strata() -> None:
    reports = _valid_reports()
    positive = 0
    for row in reports[2]["exact_strata"]:  # type: ignore[union-attr]
        is_positive = positive < 22
        row["validation_score"] = 0.50 if is_positive else 0.39
        row["heldout_score"] = 0.47 if is_positive else 0.37
        positive += int(is_positive)
    reports[2] = _reseal(reports[2])

    result = validate_adr175_matched_three_arm(reports)
    support = next(gate for gate in result.gates if gate.name == "exact_strata_joint_support")
    assert json.loads(support.evidence_json)["jointly_positive_count"] == 22


def test_validator_never_counts_a_censored_exact_stratum_as_positive() -> None:
    reports = _valid_reports()
    positive = 0
    for row in reports[2]["exact_strata"]:  # type: ignore[union-attr]
        is_positive = positive < 22
        row["validation_score"] = 0.50 if is_positive else 0.39
        row["heldout_score"] = 0.47 if is_positive else 0.37
        positive += int(is_positive)
    for arm_index in (1, 2):
        row = reports[arm_index]["exact_strata"][0]  # type: ignore[index]
        row["validation_censored"] = True
        row["validation_observable_sample_count"] = 0
        row["validation_score"] = 0.0
        reports[arm_index] = _reseal(reports[arm_index])

    with pytest.raises(ContractError, match="below 22/29"):
        validate_adr175_matched_three_arm(reports)


def test_arm_report_rejects_exact_score_outside_unit_interval() -> None:
    report = _valid_reports()[2]
    report["exact_strata"][0]["validation_score"] = 1.01  # type: ignore[index]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="must be at most 1.0"):
        ADR175ArmReport.from_dict(report)


@pytest.mark.parametrize("field", ["raw_lower_bound", "normalized_lower_bound"])
def test_validator_rejects_nonpositive_heldout_bootstrap_lower_bound(field: str) -> None:
    reports = _valid_reports()
    reports[2]["heldout_selectivity_bootstrap"][field] = 0.0  # type: ignore[index]
    reports[2] = _reseal(reports[2])

    with pytest.raises(ContractError, match="must both be positive"):
        validate_adr175_matched_three_arm(reports)


def test_validator_rejects_action_auc_more_than_two_percent_worse() -> None:
    reports = _valid_reports()
    for milestone in reports[1]["milestones"]:  # type: ignore[union-attr]
        milestone["action_loss"]["validation"] *= 1.03
        milestone["action_loss"]["heldout"] *= 1.03
    reports[1] = _reseal(reports[1])

    with pytest.raises(ContractError, match="more than 2% worse"):
        validate_adr175_matched_three_arm(reports)


def test_validator_accepts_action_auc_exactly_two_percent_worse() -> None:
    reports = _valid_reports()
    for baseline, treatment in zip(
        reports[0]["milestones"],  # type: ignore[arg-type]
        reports[1]["milestones"],  # type: ignore[arg-type]
        strict=True,
    ):
        treatment["action_loss"]["validation"] = (  # type: ignore[index]
            baseline["action_loss"]["validation"] * 1.02  # type: ignore[index]
        )
        treatment["action_loss"]["heldout"] = (  # type: ignore[index]
            baseline["action_loss"]["heldout"] * 1.02  # type: ignore[index]
        )
    reports[1] = _reseal(reports[1])

    assert validate_adr175_matched_three_arm(reports).status == "PASS"


def test_arm_report_rejects_bootstrap_contrast_identity_drift() -> None:
    report = _valid_reports()[2]
    report["heldout_selectivity_bootstrap"]["reference_arm"] = "lbot"  # type: ignore[index]
    report = _resign_without_semantic_validation(report)

    with pytest.raises(ContractError, match="must compare native-attention against physical-set"):
        ADR175ArmReport.from_dict(report)


@pytest.mark.parametrize("arm_index", [1, 2])
def test_validator_rejects_treatment_entity_set_without_step0_improvement(
    arm_index: int,
) -> None:
    reports = _valid_reports()
    initial = reports[arm_index]["milestones"][0]["entity_set_score"]  # type: ignore[index]
    reports[arm_index]["milestones"][-1]["entity_set_score"]["heldout"] = initial[  # type: ignore[index]
        "heldout"
    ]
    reports[arm_index] = _reseal(reports[arm_index])

    with pytest.raises(ContractError, match="did not improve from step 0"):
        validate_adr175_matched_three_arm(reports)


def test_validator_rejects_native_entity_set_inferiority_to_physical_set() -> None:
    reports = _valid_reports()
    physical_final = reports[1]["milestones"][-1]["entity_set_score"]  # type: ignore[index]
    reports[2]["milestones"][-1]["entity_set_score"]["heldout"] = (  # type: ignore[index]
        physical_final["heldout"] - 0.01
    )
    reports[2] = _reseal(reports[2])

    with pytest.raises(ContractError, match="inferior to physical-set"):
        validate_adr175_matched_three_arm(reports)


def test_validator_accepts_native_entity_set_equal_to_physical_set() -> None:
    reports = _valid_reports()
    physical_final = reports[1]["milestones"][-1]["entity_set_score"]  # type: ignore[index]
    reports[2]["milestones"][-1]["entity_set_score"] = copy.deepcopy(physical_final)  # type: ignore[index]
    reports[2] = _reseal(reports[2])

    assert validate_adr175_matched_three_arm(reports).status == "PASS"


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(report) + b"\n")


def _run_cli(
    reports: list[dict[str, object]],
    output: Path,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    paths = {arm: tmp_path / f"{arm}.json" for arm in ADR175_ARMS}
    for arm, report in zip(ADR175_ARMS, reports, strict=True):
        _write_report(paths[arm], report)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--lbot-report",
            str(paths["lbot"]),
            "--physical-set-report",
            str(paths["physical-set"]),
            "--native-attention-report",
            str(paths["native-attention"]),
            "--output",
            str(output),
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_atomically_writes_only_a_validated_result(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"
    completed = _run_cli(_valid_reports(), output, tmp_path)

    assert completed.returncode == 0, completed.stderr
    result = ADR175ValidationResult.from_dict(json.loads(output.read_bytes()))
    assert result.status == "PASS"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_cli_failure_preserves_existing_output_and_cleans_temporary_files(tmp_path: Path) -> None:
    reports = _valid_reports()
    reports[2]["step_receipts"][0]["sample_sha256"] = _digest("unsigned-tamper")  # type: ignore[index]
    output = tmp_path / "validation.json"
    sentinel = b"preexisting-result-must-survive\n"
    output.write_bytes(sentinel)

    completed = _run_cli(reports, output, tmp_path)

    assert completed.returncode == 2
    assert "validation failed" in completed.stderr
    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_cli_rejects_output_symlink_without_touching_target(tmp_path: Path) -> None:
    reports = _valid_reports()
    target = tmp_path / "target.json"
    target.write_bytes(b"target-must-survive\n")
    output = tmp_path / "validation.json"
    output.symlink_to(target)

    completed = _run_cli(reports, output, tmp_path)

    assert completed.returncode == 2
    assert "must not be a symlink" in completed.stderr
    assert target.read_bytes() == b"target-must-survive\n"
    assert output.is_symlink()

#!/usr/bin/env python3
"""Open ADR-157 labels only after action receipts and score G2 offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_target_request import NativeCALVINStructuralTargetRequest
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.lingbot_native.calvin_entity_set import (
    build_task_independent_calvin_targets,
)
from picf_next.lingbot_native.entity_training import (
    TaskIndependentEntityObjectiveConfig,
    compose_task_independent_entity_objective,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _relation(value: dict[str, Any]) -> PhysicalRelationOutput:
    return PhysicalRelationOutput(**value)


def _identity_to_row(
    *,
    relation: PhysicalRelationOutput,
    request: NativeCALVINStructuralTargetRequest,
    model_inputs: dict[str, torch.Tensor],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity_seed: int,
    patch_size: int,
    merge_size: int,
) -> tuple[dict[str, int], dict[int, float]]:
    bundle = build_task_independent_calvin_targets(
        requests_by_time=((request,),),
        model_inputs_by_time=(model_inputs,),
        relations=(relation,),
        physical_sidecar=physical_sidecar,
        capacity=relation.support_logits.shape[-1],
        patch_size=patch_size,
        merge_size=merge_size,
        capacity_seeds=(capacity_seed,),
    )[0]
    objective = compose_task_independent_entity_objective(
        official_policy_loss=None,
        relations=(relation,),
        targets=(bundle.targets,),
        config=TaskIndependentEntityObjectiveConfig(
            action_weight=0.0,
            entity_weight=1.0,
            predictive_weight=0.0,
        ),
    )
    row_to_track = objective.frame_losses[0].assignment.row_to_track[0]
    identities = bundle.identity_keys_by_batch[0]
    identity_to_row = {}
    existence_by_row = {}
    for row, track in enumerate(row_to_track.tolist()):
        existence_by_row[row] = float(relation.existence[0, row].float().item())
        if track >= 0:
            if track >= len(identities):
                raise RuntimeError("ADR-157 G2 assignment references an absent identity")
            identity_to_row[identities[track]] = row
    return identity_to_row, existence_by_row


def _action_table(
    execution: dict[str, Any], actions: dict[str, torch.Tensor]
) -> dict[tuple[str, str], torch.Tensor]:
    table = {}
    for receipt in execution["receipts"]:
        key = receipt["action_key"]
        action = actions.get(key)
        if not isinstance(action, torch.Tensor):
            raise ValueError("ADR-157 G2 action payload is incomplete")
        table[(receipt["prompt_name"], receipt["arm_name"])] = action.float()
    if len(table) != execution["receipt_count"]:
        raise ValueError("ADR-157 G2 action receipts are duplicate or incomplete")
    return table


def _normalized_effect(changed: torch.Tensor, factual: torch.Tensor) -> float:
    if changed.shape != factual.shape:
        raise ValueError("ADR-157 G2 action shapes differ")
    denominator = factual.norm().clamp_min(1e-8)
    return float(((changed - factual).norm() / denominator).item())


def _score_item(
    *,
    item: dict[str, Any],
    label_item: dict[str, Any],
    evidence_root: Path,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    patch_size: int,
    merge_size: int,
) -> dict[str, Any]:
    rank_root = evidence_root / f"rank_{item['execution_rank']}"
    execution_path = rank_root / "execution.json"
    offline_receipt_path = rank_root / "offline_receipt.json"
    execution = _load_json(execution_path)
    offline_receipt = _load_json(offline_receipt_path)
    if (
        execution.get("status") != "ACTION_RECEIPTS_SEALED"
        or execution.get("item_id") != item["item_id"]
        or offline_receipt.get("status") != "SEALED_AFTER_ACTIONS"
        or offline_receipt.get("execution_file_sha256") != _sha256(execution_path)
    ):
        raise ValueError("ADR-157 G2 sealed rank receipts differ")
    action_path = rank_root / execution["actions_file"]
    offline_path = rank_root / offline_receipt["offline_inputs_file"]
    if (
        _sha256(action_path) != execution["actions_file_sha256"]
        or _sha256(offline_path) != offline_receipt["offline_inputs_file_sha256"]
    ):
        raise ValueError("ADR-157 G2 sealed tensor artifact changed")
    actions = torch.load(action_path, map_location="cpu", weights_only=True)
    offline = torch.load(offline_path, map_location="cpu", weights_only=True)
    if offline["item_id"] != item["item_id"]:
        raise ValueError("ADR-157 G2 offline payload names another item")
    relations = tuple(_relation(value) for value in offline["relation_by_prompt"])
    request = NativeCALVINStructuralTargetRequest(**offline["structural_target_request"])
    model_inputs = offline["model_inputs"]
    gauges = tuple(
        _identity_to_row(
            relation=relation,
            request=replace_task_key(request, prompt["task_key"]),
            model_inputs=model_inputs,
            physical_sidecar=physical_sidecar,
            capacity_seed=offline["capacity_seed"],
            patch_size=patch_size,
            merge_size=merge_size,
        )
        for relation, prompt in zip(relations, item["prompts"], strict=True)
    )
    action_table = _action_table(execution, actions)
    prompt_labels = label_item["prompts"]
    if [value["name"] for value in prompt_labels] != [value["name"] for value in item["prompts"]]:
        raise ValueError("ADR-157 G2 prompt labels differ from execution names")
    canonical_gauge, canonical_existence = gauges[0]
    target_identities = [value["target_identity_key"] for value in prompt_labels]
    if any(identity not in canonical_gauge for identity in target_identities):
        raise ValueError("ADR-157 G2 target identity has no canonical posterior row")
    target_rows = [canonical_gauge[identity] for identity in target_identities]
    if target_rows[0] == target_rows[1]:
        raise ValueError("ADR-157 G2 distinct targets collapsed to one row")

    prompt_reports = []
    row_effects_by_prompt: dict[str, list[float]] = {}
    for prompt_index, prompt in enumerate(item["prompts"]):
        name = prompt["name"]
        factual = action_table[(name, "factual")]
        repeated = action_table[(name, "factual-repeat")]
        repeat_floor = _normalized_effect(repeated, factual)
        row_effects = [
            _normalized_effect(action_table[(name, f"remove-row-{row}")], factual)
            for row in range(execution["capacity"])
        ]
        row_effects_by_prompt[name] = row_effects
        target_row = target_rows[prompt_index]
        distractor_row = target_rows[1 - prompt_index]
        target_effect = row_effects[target_row]
        distractor_effect = row_effects[distractor_row]
        controls = [
            effect
            for row, effect in enumerate(row_effects)
            if row not in {target_row, distractor_row}
        ]
        control_median = float(torch.tensor(controls).median().item())
        prompt_reports.append(
            {
                "control_median_effect": control_median,
                "distractor_effect": distractor_effect,
                "distractor_identity": target_identities[1 - prompt_index],
                "distractor_row": distractor_row,
                "existence_gap": abs(
                    canonical_existence[target_row] - canonical_existence[distractor_row]
                ),
                "mass_log_gap": abs(
                    math.log(prompt_labels[prompt_index]["target_mass"])
                    - math.log(prompt_labels[1 - prompt_index]["target_mass"])
                ),
                "prompt_name": name,
                "repeat_floor": repeat_floor,
                "row_effects": row_effects,
                "target_effect": target_effect,
                "target_identity": target_identities[prompt_index],
                "target_minus_distractor": target_effect - distractor_effect,
                "target_minus_distractor_over_floor": (
                    (target_effect - distractor_effect) / max(repeat_floor, 1e-12)
                ),
                "target_row": target_row,
            }
        )
    prompt_a, prompt_b = (item["prompts"][0]["name"], item["prompts"][1]["name"])
    effects_a = row_effects_by_prompt[prompt_a]
    effects_b = row_effects_by_prompt[prompt_b]
    prompt_switch_did = (effects_a[target_rows[0]] - effects_a[target_rows[1]]) - (
        effects_b[target_rows[0]] - effects_b[target_rows[1]]
    )
    second_gauge = gauges[1][0]
    return {
        "canonical_identity_to_row": canonical_gauge,
        "item_id": item["item_id"],
        "partition": item["partition"],
        "posterior_prompt_relative_l2": execution["posterior_prompt_relative_l2"],
        "prompt_reports": prompt_reports,
        "prompt_switch_difference_in_differences": prompt_switch_did,
        "prompt_switch_positive": prompt_switch_did > 0,
        "row_gauge_stable_for_targets": all(
            second_gauge.get(identity) == canonical_gauge[identity]
            for identity in target_identities
        ),
        "sample_key": item["sample_key"],
    }


def replace_task_key(
    request: NativeCALVINStructuralTargetRequest,
    task_key: str,
) -> NativeCALVINStructuralTargetRequest:
    return NativeCALVINStructuralTargetRequest(
        sample_key=request.sample_key,
        episode_key=request.episode_key,
        task_key=task_key,
        segment_index=request.segment_index,
        source_global_index=request.source_global_index,
        source_sensor_sha256=request.source_sensor_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--offline-labels", type=Path, required=True)
    parser.add_argument("--offline-labels-sha256", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--merge-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if _sha256(args.offline_labels) != args.offline_labels_sha256:
        raise ValueError("ADR-157 G2 offline-label file digest changed")
    execution = _load_json(args.execution_contract)
    labels = _load_json(args.offline_labels)
    aggregate = _load_json(args.evidence_root / "aggregate.json")
    if (
        aggregate.get("status") != "ACTION_RECEIPTS_SEALED"
        or aggregate.get("execution_contract_file_sha256") != _sha256(args.execution_contract)
        or labels.get("source_execution_sha256")
        != hashlib.sha256(_canonical_bytes(execution)).hexdigest()
    ):
        raise ValueError("ADR-157 G2 execution, labels, and sealed aggregate differ")
    label_by_item = {item["item_id"]: item for item in labels["items"]}
    if set(label_by_item) != {item["item_id"] for item in execution["items"]}:
        raise ValueError("ADR-157 G2 offline labels do not cover the execution contract")

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        args.dataset_split.resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.physical_sidecar_root,
        index,
        manifest_path=args.physical_sidecar_manifest,
        expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
    )
    reports = [
        _score_item(
            item=item,
            label_item=label_by_item[item["item_id"]],
            evidence_root=args.evidence_root,
            physical_sidecar=sidecar,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
        )
        for item in execution["items"]
    ]
    validation = [item for item in reports if item["partition"] == "validation"]
    heldout = [item for item in reports if item["partition"] == "heldout"]
    result = {
        "execution_contract_file_sha256": _sha256(args.execution_contract),
        "offline_labels_file_sha256": args.offline_labels_sha256,
        "item_reports": reports,
        "mean_prompt_switch_difference_in_differences": float(
            torch.tensor(
                [item["prompt_switch_difference_in_differences"] for item in reports]
            ).mean()
        ),
        "positive_prompt_report_fraction": sum(
            report["target_minus_distractor"] > 0
            for item in reports
            for report in item["prompt_reports"]
        )
        / (2 * len(reports)),
        "schema": "picf-next.adr157-g2-offline-score/v1",
        "status": "MEASURED",
        "summary_by_partition": {
            "heldout": {
                "item_count": len(heldout),
                "prompt_switch_positive_count": sum(
                    item["prompt_switch_positive"] for item in heldout
                ),
            },
            "validation": {
                "item_count": len(validation),
                "prompt_switch_positive_count": sum(
                    item["prompt_switch_positive"] for item in validation
                ),
            },
        },
    }
    write_bytes_durable_exclusive(args.output, _canonical_bytes(result) + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

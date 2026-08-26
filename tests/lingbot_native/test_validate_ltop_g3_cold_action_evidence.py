from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

import tools.validate_ltop_g3_cold_action_evidence as validator_module
from tools.validate_ltop_g3_cold_action_evidence import (
    OUTPUT_SCHEMA,
    ValidationInputError,
    main,
    validate_ltop_g3_cold_action_evidence,
)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _refresh_prompt_score(score: dict[str, object]) -> None:
    factual_target = list(score["factual_target_effect_rms"])
    factual_distractor = list(score["factual_distractor_effect_rms"])
    blocked_target = list(score["blocked_target_effect_rms"])
    blocked_distractor = list(score["blocked_distractor_effect_rms"])
    replay_floor = list(score["replay_floor_rms"])
    factual_delta = [
        target - distractor
        for target, distractor in zip(factual_target, factual_distractor, strict=True)
    ]
    blocked_delta = [
        target - distractor
        for target, distractor in zip(blocked_target, blocked_distractor, strict=True)
    ]
    did = [factual - blocked for factual, blocked in zip(factual_delta, blocked_delta, strict=True)]
    score["factual_target_minus_distractor"] = factual_delta
    score["blocked_target_minus_distractor"] = blocked_delta
    score["blocked_path_difference_in_differences"] = did
    score["mean_factual_target_minus_distractor"] = _mean(factual_delta)
    score["mean_blocked_path_difference_in_differences"] = _mean(did)
    score["positive_factual_count"] = sum(
        value > floor for value, floor in zip(factual_delta, replay_floor, strict=True)
    )
    score["positive_blocked_path_did_count"] = sum(
        value > floor for value, floor in zip(did, replay_floor, strict=True)
    )


def _prompt_score(
    prompt_name: str,
    sample_key: str,
    *,
    factual_delta: float = 0.4,
    blocked_delta: float = 0.1,
) -> dict[str, object]:
    score: dict[str, object] = {
        "prompt_name": prompt_name,
        "sample_keys": [sample_key],
        "replay_floor_rms": [0.0],
        "factual_target_effect_rms": [0.5 + factual_delta],
        "factual_distractor_effect_rms": [0.5],
        "blocked_target_effect_rms": [0.2 + blocked_delta],
        "blocked_distractor_effect_rms": [0.2],
    }
    _refresh_prompt_score(score)
    return score


def _refresh_scene_score(scene: dict[str, object]) -> None:
    prompts = list(scene["prompts"])
    scores = [prompt["score"] for prompt in prompts]
    scene["score"] = {
        "prompt_name": f"{scene['item_id']}/aggregate",
        "sample_keys": [scene["sample_key"] for _prompt in prompts],
        "replay_floor_rms": [value for score in scores for value in score["replay_floor_rms"]],
        "mean_factual_target_minus_distractor": _mean(
            [score["mean_factual_target_minus_distractor"] for score in scores]
        ),
        "mean_blocked_path_difference_in_differences": _mean(
            [score["mean_blocked_path_difference_in_differences"] for score in scores]
        ),
        "positive_factual_count": sum(score["positive_factual_count"] for score in scores),
        "positive_blocked_path_did_count": sum(
            score["positive_blocked_path_did_count"] for score in scores
        ),
    }


def _scene(partition: str, *, rank: int, scene_index: int) -> dict[str, object]:
    ordinal = (0 if partition == "validation" else 8) + rank * 4 + scene_index
    item_id = f"{partition}-{ordinal:04d}"
    sample_key = f"calvin/{item_id}"
    prompt_sample_key = f"ltop-g3/{item_id}"
    first_identity = f"object/{item_id}/first"
    second_identity = f"object/{item_id}/second"
    first_row = 2 * scene_index
    second_row = first_row + 1
    scene: dict[str, object] = {
        "item_id": item_id,
        "sample_key": sample_key,
        "prompts": [
            {
                "prompt_name": f"{item_id}/prompt-0",
                "target_identity": first_identity,
                "matched_distractor_identity": second_identity,
                "target_row": first_row,
                "matched_distractor_row": second_row,
                "score": _prompt_score(f"{item_id}/prompt-0", prompt_sample_key),
            },
            {
                "prompt_name": f"{item_id}/prompt-1",
                "target_identity": second_identity,
                "matched_distractor_identity": first_identity,
                "target_row": second_row,
                "matched_distractor_row": first_row,
                "score": _prompt_score(f"{item_id}/prompt-1", prompt_sample_key),
            },
        ],
    }
    _refresh_scene_score(scene)
    return scene


def _report(information_set: str) -> dict[str, object]:
    return {
        "schema": "picf-next.ltop-g3-evaluation-phase.v1",
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "phase": "evaluation",
        "world_size": 2,
        "steps": 128,
        "eval_every": 32,
        "capacity": 16,
        "evaluation_action_information_set": information_set,
        "thresholds": {
            "bitwise_factual_replay": True,
            "action_loss_improvement_ratio_maximum": 0.95,
            "mean_factual_target_minus_distractor_strictly_positive": True,
            "mean_blocked_path_did_strictly_positive": True,
            "positive_sample_fraction_minimum": 0.625,
        },
        "rank_reports": [
            {
                "rank": rank,
                "history": [
                    {
                        "step": 128,
                        **{
                            partition: {
                                "scenes": [
                                    _scene(partition, rank=rank, scene_index=scene_index)
                                    for scene_index in range(4)
                                ],
                                "max_replay_floor_rms": 0.0,
                            }
                            for partition in ("validation", "heldout")
                        },
                    }
                ],
            }
            for rank in range(2)
        ],
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=True, sort_keys=True) + "\n", encoding="ascii")


def _write_reports(
    tmp_path: Path,
    *,
    factual: dict[str, object] | None = None,
    mediator: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    factual_path = tmp_path / "factual.json"
    mediator_path = tmp_path / "mediator.json"
    _write(factual_path, _report("factual") if factual is None else factual)
    _write(
        mediator_path,
        _report("mediator-required") if mediator is None else mediator,
    )
    return factual_path, mediator_path


def _first_scene(report: dict[str, object]) -> dict[str, object]:
    return report["rank_reports"][0]["history"][0]["validation"]["scenes"][0]


def _tamper_prompt_did(report: dict[str, object]) -> None:
    score = _first_scene(report)["prompts"][0]["score"]
    score["blocked_path_difference_in_differences"][0] += 0.25


def _tamper_scene_mean(report: dict[str, object]) -> None:
    _first_scene(report)["score"]["mean_factual_target_minus_distractor"] += 0.25


def _tamper_count_upper_bound(report: dict[str, object]) -> None:
    _first_scene(report)["prompts"][0]["score"]["positive_factual_count"] = 2


def _tamper_crossed_rows(report: dict[str, object]) -> None:
    _first_scene(report)["prompts"][1]["matched_distractor_row"] += 1


def _tamper_nonfinite(report: dict[str, object]) -> None:
    _first_scene(report)["prompts"][0]["score"]["factual_target_effect_rms"][0] = float("nan")


def _tamper_duplicate_sample(report: dict[str, object]) -> None:
    scenes = report["rank_reports"][0]["history"][0]["validation"]["scenes"]
    scenes[1]["sample_key"] = scenes[0]["sample_key"]
    scenes[1]["score"]["sample_keys"] = [scenes[0]["sample_key"]] * 2


def test_valid_reports_recompute_every_level_and_write_one_independent_json(
    tmp_path: Path,
) -> None:
    factual_path, mediator_path = _write_reports(tmp_path)
    factual_before = factual_path.read_bytes()
    mediator_before = mediator_path.read_bytes()

    result = validate_ltop_g3_cold_action_evidence(
        factual_report_path=factual_path,
        mediator_required_report_path=mediator_path,
    )

    assert result["schema"] == OUTPUT_SCHEMA
    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["cross_report"]["layout_match"] is True
    for label in ("factual", "mediator_required"):
        for partition in ("validation", "heldout"):
            summary = result["reports"][label]["partitions"][partition]
            assert summary["scene_count"] == 8
            assert summary["prompt_count"] == 16
            assert summary["sample_count"] == 16
            assert summary["minimum_positive_count"] == 10
            assert summary["positive_factual_count"] == 16
            assert summary["positive_blocked_path_did_count"] == 16
            assert summary["status"] == "PASS"

    output = tmp_path / "independent-validation.json"
    assert (
        main(
            [
                "--factual-report",
                str(factual_path),
                "--mediator-required-report",
                str(mediator_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="ascii"))["status"] == "PASS"
    assert factual_path.read_bytes() == factual_before
    assert mediator_path.read_bytes() == mediator_before


def test_atomic_output_publish_leaves_no_same_directory_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "independent-validation.json"
    payload = '{"status":"PASS"}\n'
    real_replace = validator_module.os.replace
    observed_temp: list[Path] = []

    def observe_atomic_replace(source: Path, destination: Path) -> None:
        source_path = Path(source)
        assert source_path.parent == tmp_path
        assert source_path.read_text(encoding="ascii") == payload
        assert not Path(destination).exists()
        observed_temp.append(source_path)
        real_replace(source, destination)

    monkeypatch.setattr(validator_module.os, "replace", observe_atomic_replace)

    validator_module._write_exclusive(output, payload)

    assert output.read_text(encoding="ascii") == payload
    assert len(observed_temp) == 1
    assert not observed_temp[0].exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_output_publish_failure_removes_temp_and_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "independent-validation.json"

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(validator_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        validator_module._write_exclusive(output, '{"status":"PASS"}\n')

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_output_publish_never_replaces_preexisting_final(tmp_path: Path) -> None:
    output = tmp_path / "independent-validation.json"
    output.write_text("original\n", encoding="ascii")

    with pytest.raises(ValidationInputError, match="output already exists"):
        validator_module._write_exclusive(output, '{"status":"PASS"}\n')

    assert output.read_text(encoding="ascii") == "original\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_output_publish_does_not_remove_another_writer_lock(tmp_path: Path) -> None:
    output = tmp_path / "independent-validation.json"
    lock = tmp_path / f".{output.name}.publish.lock"
    lock.mkdir()

    with pytest.raises(FileExistsError):
        validator_module._write_exclusive(output, '{"status":"PASS"}\n')

    assert lock.is_dir()
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_input_report_reached_through_symlinked_parent_is_rejected(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    report_root.mkdir()
    factual_path, mediator_path = _write_reports(report_root)
    symlinked_root = tmp_path / "reports-link"
    symlinked_root.symlink_to(report_root, target_is_directory=True)

    with pytest.raises(ValidationInputError, match="symbolic-link component"):
        validate_ltop_g3_cold_action_evidence(
            factual_report_path=symlinked_root / factual_path.name,
            mediator_required_report_path=mediator_path,
        )


def test_symlinked_output_parent_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    factual_path, mediator_path = _write_reports(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    symlinked_root = tmp_path / "output-link"
    symlinked_root.symlink_to(output_root, target_is_directory=True)

    assert (
        main(
            [
                "--factual-report",
                str(factual_path),
                "--mediator-required-report",
                str(mediator_path),
                "--output",
                str(symlinked_root / "independent-validation.json"),
            ]
        )
        == 2
    )
    assert "output parent contains a symbolic-link component" in capsys.readouterr().err
    assert not (output_root / "independent-validation.json").exists()


@pytest.mark.parametrize(
    ("tamper", "expected_failure"),
    [
        (_tamper_prompt_did, "serialized blocked_path_difference_in_differences"),
        (_tamper_scene_mean, "serialized mean_factual_target_minus_distractor"),
        (_tamper_count_upper_bound, "exceeds sample_count"),
        (_tamper_crossed_rows, "crossed prompt target/distractor rows do not swap"),
        (_tamper_nonfinite, "must be finite"),
        (_tamper_duplicate_sample, "duplicate scene sample_key"),
    ],
)
def test_tampered_factual_evidence_fails_closed(
    tmp_path: Path,
    tamper: Callable[[dict[str, object]], None],
    expected_failure: str,
) -> None:
    factual = _report("factual")
    tamper(factual)
    factual_path, mediator_path = _write_reports(tmp_path, factual=factual)

    result = validate_ltop_g3_cold_action_evidence(
        factual_report_path=factual_path,
        mediator_required_report_path=mediator_path,
    )

    assert result["status"] == "FAIL"
    assert any(expected_failure in failure for failure in result["failures"])


def test_cross_report_layout_tamper_fails_after_both_reports_validate(tmp_path: Path) -> None:
    mediator = _report("mediator-required")
    scene = _first_scene(mediator)
    changed_identity = "object/relabelled/first"
    scene["prompts"][0]["target_identity"] = changed_identity
    scene["prompts"][1]["matched_distractor_identity"] = changed_identity
    factual_path, mediator_path = _write_reports(tmp_path, mediator=mediator)

    result = validate_ltop_g3_cold_action_evidence(
        factual_report_path=factual_path,
        mediator_required_report_path=mediator_path,
    )

    assert result["reports"]["factual"]["recomputed_status"] == "PASS"
    assert result["reports"]["mediator_required"]["recomputed_status"] == "PASS"
    assert result["status"] == "FAIL"
    assert result["failures"] == [
        "factual and mediator-required scene/sample/target layouts differ"
    ]


def test_recomputed_partition_gate_rejects_fewer_than_ten_positive_samples(
    tmp_path: Path,
) -> None:
    factual = _report("factual")
    changed = 0
    for rank in factual["rank_reports"]:
        for scene in rank["history"][0]["validation"]["scenes"]:
            for prompt in scene["prompts"]:
                if changed >= 7:
                    break
                score = prompt["score"]
                score["factual_target_effect_rms"] = [0.3]
                score["factual_distractor_effect_rms"] = [0.5]
                score["blocked_target_effect_rms"] = [0.15]
                score["blocked_distractor_effect_rms"] = [0.2]
                _refresh_prompt_score(score)
                changed += 1
            _refresh_scene_score(scene)
    factual_path, mediator_path = _write_reports(tmp_path, factual=factual)

    result = validate_ltop_g3_cold_action_evidence(
        factual_report_path=factual_path,
        mediator_required_report_path=mediator_path,
    )

    validation = result["reports"]["factual"]["partitions"]["validation"]
    assert validation["positive_factual_count"] == 9
    assert validation["positive_blocked_path_did_count"] == 9
    assert validation["minimum_positive_count"] == 10
    assert validation["mean_factual_target_minus_distractor"] > 0.0
    assert validation["mean_blocked_path_difference_in_differences"] > 0.0
    assert validation["status"] == "FAIL"
    assert any("positive_factual_count_minimum" in failure for failure in result["failures"])
    assert any(
        "positive_blocked_path_did_count_minimum" in failure for failure in result["failures"]
    )

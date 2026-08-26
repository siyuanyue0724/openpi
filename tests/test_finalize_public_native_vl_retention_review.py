from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from picf_next.contracts import ContractError
from tests.test_compare_public_native_vl_retention_reports import (
    _bind_synthetic_frozen_banks_for_tests,
    _report,
    _set_referring_prediction,
    _training_report,
)
from tools import finalize_public_native_vl_retention_review as finalizer
from tools.compare_public_native_vl_retention_reports import (
    FIXED_X_SCHEMA,
    TRAINING_SCHEMA,
    compare_public_native_vl_retention_reports,
)
from tools.finalize_public_native_vl_retention_review import (
    COMPARISON_SCHEMA,
    PENDING_STATUS,
    REVIEW_SCHEMA,
    REVIEWER_TRUST_BLOCKER,
    finalize_public_native_vl_retention_review,
)

_PUBLIC_REVIEW_FIELD = "public_generation_visual_review_required"


@pytest.fixture(autouse=True)
def _bind_frozen_test_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind_synthetic_frozen_banks_for_tests(monkeypatch)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(payload)


def _write_json(path: Path, value: object, *, compact: bool = False) -> str:
    if compact:
        text = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    else:
        text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _write_panel(directory: Path, relative: str, *, seed: int) -> dict[str, str]:
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(
        "RGB",
        (8, 6),
        color=((seed * 31) % 256, (seed * 67) % 256, (seed * 97) % 256),
    )
    image.save(path, format="PNG")
    return {"file": relative, "sha256": _sha256_bytes(path.read_bytes())}


def _calvin_group(report: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return next(
        group
        for group in report["results"]
        if group["partition"] == item["partition"] and group["ordinal"] == item["ordinal"]
    )


def _public_row(report: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return next(
        row
        for row in report["public_vl_retention"]["results"]
        if row["family"] == item["family"]
        and row["record_id"] == item["record_id"]
        and row["source_row_index"] == item["source_row_index"]
        and row["source_subindex"] == item["source_subindex"]
    )


def _attach_required_panels(
    *,
    comparison: dict[str, Any],
    report: dict[str, Any],
    report_dir: Path,
    role_seed: int,
) -> None:
    seen_calvin: set[tuple[str, int]] = set()
    for index, item in enumerate(comparison["calvin"]["visual_review_required"]):
        key = item["partition"], item["ordinal"]
        if key in seen_calvin:
            continue
        seen_calvin.add(key)
        group = _calvin_group(report, item)
        relative = f"visuals/{key[0]}-{key[1]:03d}.png"
        group["visual"] = _write_panel(
            report_dir,
            relative,
            seed=role_seed + index,
        )
    for index, item in enumerate(comparison.get(_PUBLIC_REVIEW_FIELD, [])):
        row = _public_row(report, item)
        relative = f"public_retention_visuals/{item['record_id']}.png"
        row["visual"] = _write_panel(
            report_dir,
            relative,
            seed=role_seed + 100 + index,
        )


def _panel_binding(
    *,
    report: dict[str, Any],
    report_dir: Path,
    role: str,
    item: dict[str, Any],
    public: bool,
) -> dict[str, int | str]:
    sample = _public_row(report, item) if public else _calvin_group(report, item)
    visual = sample["visual"]
    sample_evidence = {key: value for key, value in sample.items() if key != "visual"}
    path = report_dir / visual["file"]
    with Image.open(path) as image:
        image.load()
        image_format = image.format
        width, height = image.size
    return {
        "image_format": image_format,
        "image_height": height,
        "image_width": width,
        "path": visual["file"],
        "report": role,
        "sample_sha256": _canonical_sha256(sample_evidence),
        "sha256": visual["sha256"],
    }


def _decision(
    *,
    fixture: dict[str, Any],
    item: dict[str, Any],
    public: bool,
) -> dict[str, object]:
    panels = [
        _panel_binding(
            report=fixture[f"{role}_report"],
            report_dir=fixture[f"{role}_dir"],
            role=role,
            item=item,
            public=public,
        )
        for role in ("control", "candidate")
    ]
    common = {
        "decision": "accept",
        "finding": "decoded panel and report evidence agree",
        "panels": panels,
        "required_visual_review_item_sha256": _canonical_sha256(item),
    }
    if public:
        return {
            **common,
            "family": item["family"],
            "record_id": item["record_id"],
            "source_row_index": item["source_row_index"],
            "source_subindex": item["source_subindex"],
        }
    return {
        **common,
        "family": item.get("family"),
        "ordinal": item["ordinal"],
        "partition": item["partition"],
        "task_key": item.get("task_key"),
        "variant_index": item.get("variant_index"),
    }


def _review_decisions(fixture: dict[str, Any]) -> list[dict[str, object]]:
    decisions = [
        _decision(fixture=fixture, item=item, public=False)
        for item in fixture["comparison"]["calvin"]["visual_review_required"]
    ]
    decisions.extend(
        _decision(fixture=fixture, item=item, public=True)
        for item in fixture["comparison"].get(_PUBLIC_REVIEW_FIELD, [])
    )
    return decisions


def _compare_exact(fixture: dict[str, Any]) -> dict[str, Any]:
    return compare_public_native_vl_retention_reports(
        fixture["control_report"],
        fixture["candidate_report"],
        fixture["training_report"],
        control_report_sha256=fixture["control_sha256"],
        candidate_report_sha256=fixture["candidate_sha256"],
        candidate_training_report_sha256=fixture["training_sha256"],
    )


def _write_review(fixture: dict[str, Any]) -> None:
    _write_json(fixture["review_path"], fixture["review"])


def _refresh_comparison(fixture: dict[str, Any], *, refresh_decisions: bool) -> None:
    fixture["comparison"] = _compare_exact(fixture)
    fixture["comparison_sha256"] = _write_json(
        fixture["comparison_path"],
        fixture["comparison"],
    )
    fixture["review"]["comparison_report_sha256"] = fixture["comparison_sha256"]
    fixture["review"]["candidate_report_sha256"] = fixture["candidate_sha256"]
    fixture["review"]["control_report_sha256"] = fixture["control_sha256"]
    fixture["review"]["candidate_training_report_sha256"] = fixture["training_sha256"]
    if refresh_decisions:
        fixture["review"]["decisions"] = _review_decisions(fixture)
    _write_review(fixture)


def _fixture(tmp_path: Path, *, public_change: bool = False) -> dict[str, Any]:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["checkpoint_dir"] = "/candidate-checkpoint"
    if public_change:
        _set_referring_prediction(
            candidate,
            index=20,
            prediction=[100, 100, 200, 200],
        )
    training = _training_report(control, candidate)
    provisional = compare_public_native_vl_retention_reports(
        control,
        candidate,
        training,
        control_report_sha256="1" * 64,
        candidate_report_sha256="2" * 64,
        candidate_training_report_sha256="3" * 64,
    )
    assert provisional["status"] == PENDING_STATUS

    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    _attach_required_panels(
        comparison=provisional,
        report=control,
        report_dir=control_dir,
        role_seed=10,
    )
    _attach_required_panels(
        comparison=provisional,
        report=candidate,
        report_dir=candidate_dir,
        role_seed=20,
    )
    fixture: dict[str, Any] = {
        "candidate_dir": candidate_dir,
        "candidate_report": candidate,
        "candidate_report_path": candidate_dir / "report.json",
        "comparison_path": tmp_path / "comparison.json",
        "control_dir": control_dir,
        "control_report": control,
        "control_report_path": control_dir / "report.json",
        "output_path": tmp_path / "finalized.json",
        "review_path": tmp_path / "visual-review.json",
        "training_report": training,
        "training_report_path": tmp_path / "training.json",
    }
    fixture["control_sha256"] = _write_json(fixture["control_report_path"], control)
    fixture["candidate_sha256"] = _write_json(fixture["candidate_report_path"], candidate)
    fixture["training_sha256"] = _write_json(fixture["training_report_path"], training)
    fixture["comparison"] = _compare_exact(fixture)
    fixture["comparison_sha256"] = _write_json(
        fixture["comparison_path"],
        fixture["comparison"],
    )
    fixture["review"] = {
        "candidate_report": str(fixture["candidate_report_path"].resolve()),
        "candidate_report_sha256": fixture["candidate_sha256"],
        "candidate_training_report": str(fixture["training_report_path"].resolve()),
        "candidate_training_report_sha256": fixture["training_sha256"],
        "comparison_report": str(fixture["comparison_path"].resolve()),
        "comparison_report_sha256": fixture["comparison_sha256"],
        "control_report": str(fixture["control_report_path"].resolve()),
        "control_report_sha256": fixture["control_sha256"],
        "decisions": _review_decisions(fixture),
        "reviewer": "independent-reviewer",
        "schema": REVIEW_SCHEMA,
        "status": "PASS",
    }
    _write_review(fixture)
    return fixture


def _finalize(fixture: dict[str, Any]) -> dict[str, object]:
    return finalize_public_native_vl_retention_review(
        comparison_report_path=fixture["comparison_path"],
        comparison_report_sha256=fixture["comparison_sha256"],
        control_report_path=fixture["control_report_path"],
        control_report_sha256=fixture["control_sha256"],
        candidate_report_path=fixture["candidate_report_path"],
        candidate_report_sha256=fixture["candidate_sha256"],
        candidate_training_report_path=fixture["training_report_path"],
        candidate_training_report_sha256=fixture["training_sha256"],
        visual_review_path=fixture["review_path"],
        output_path=fixture["output_path"],
    )


def test_authentic_decodable_evidence_reaches_unsigned_reviewer_blocker(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, public_change=True)

    with pytest.raises(ContractError, match="reviewer identity is unsigned"):
        _finalize(fixture)

    assert not fixture["output_path"].exists()
    assert REVIEWER_TRUST_BLOCKER.startswith("reviewer identity is unsigned")
    assert fixture["comparison"]["schema"] == COMPARISON_SCHEMA
    assert fixture["comparison"]["input_reports"] == {
        "candidate": {"schema": FIXED_X_SCHEMA, "sha256": fixture["candidate_sha256"]},
        "candidate_training": {
            "schema": TRAINING_SCHEMA,
            "sha256": fixture["training_sha256"],
        },
        "control": {"schema": FIXED_X_SCHEMA, "sha256": fixture["control_sha256"]},
    }


def test_finalizer_rejects_handcrafted_boolean_pass_comparison(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["comparison"] = {
        "calvin": {
            "checks": {"forged": True},
            "numeric_status": "PASS",
            "visual_review_required": deepcopy(
                fixture["comparison"]["calvin"]["visual_review_required"]
            ),
        },
        "families": {
            family: {
                "generation_checks": {"forged": True},
                "generation_metrics": {"forged": {"candidate": 1}},
                "generation_nonregression": True,
                "mean_record_nll_nonregression": True,
                "token_weighted_mean_nll_strictly_improves": True,
            }
            for family in ("referring", "vqa")
        },
        "schema": COMPARISON_SCHEMA,
        "status": PENDING_STATUS,
    }
    fixture["comparison_sha256"] = _write_json(
        fixture["comparison_path"],
        fixture["comparison"],
    )
    fixture["review"]["comparison_report_sha256"] = fixture["comparison_sha256"]
    _write_review(fixture)

    with pytest.raises(ContractError, match="exact authentic recomputation"):
        _finalize(fixture)


def test_finalizer_rejects_substituted_training_report_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["training_sha256"] = _write_json(
        fixture["training_report_path"],
        fixture["training_report"],
        compact=True,
    )
    fixture["review"]["candidate_training_report_sha256"] = fixture["training_sha256"]
    _write_review(fixture)

    with pytest.raises(ContractError, match="exact authentic recomputation"):
        _finalize(fixture)


def test_panel_review_binds_generated_text_prediction_and_metrics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    item = fixture["comparison"]["calvin"]["visual_review_required"][0]
    group = _calvin_group(fixture["candidate_report"], item)
    variant_index = item.get("variant_index")
    if variant_index is None:
        variant_index = 0
    variant = group["variants"][variant_index]
    bbox = variant["generated_bbox_qwen_xyxy"]
    variant["generated_text"] = json.dumps({"bbox_2d": bbox}, separators=(",", ":"))
    fixture["candidate_sha256"] = _write_json(
        fixture["candidate_report_path"],
        fixture["candidate_report"],
    )
    _refresh_comparison(fixture, refresh_decisions=False)

    with pytest.raises(ContractError, match="panels differ from fixed-X declared artifacts"):
        _finalize(fixture)


def test_finalizer_decodes_image_instead_of_trusting_extension_or_magic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    item = fixture["comparison"]["calvin"]["visual_review_required"][0]
    group = _calvin_group(fixture["candidate_report"], item)
    panel_path = fixture["candidate_dir"] / group["visual"]["file"]
    invalid = b"\x89PNG\r\n\x1a\nnot-an-image"
    panel_path.write_bytes(invalid)
    group["visual"]["sha256"] = _sha256_bytes(invalid)
    fixture["candidate_sha256"] = _write_json(
        fixture["candidate_report_path"],
        fixture["candidate_report"],
    )
    _refresh_comparison(fixture, refresh_decisions=False)

    with pytest.raises(ContractError, match="not a decodable image"):
        _finalize(fixture)


def test_finalizer_rejects_panel_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    item = fixture["comparison"]["calvin"]["visual_review_required"][0]
    group = _calvin_group(fixture["candidate_report"], item)
    panel_path = fixture["candidate_dir"] / group["visual"]["file"]
    outside = tmp_path / "outside.png"
    outside.write_bytes(panel_path.read_bytes())
    panel_path.unlink()
    panel_path.symlink_to(outside)

    with pytest.raises(ContractError, match="must not be a symlink"):
        _finalize(fixture)


def test_finalizer_final_rehash_detects_post_validation_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    item = fixture["comparison"]["calvin"]["visual_review_required"][0]
    group = _calvin_group(fixture["candidate_report"], item)
    panel_path = fixture["candidate_dir"] / group["visual"]["file"]
    original_validate = finalizer._validate_review

    def mutate_after_review(*args: Any, **kwargs: Any) -> Any:
        result = original_validate(*args, **kwargs)
        panel_path.write_bytes(panel_path.read_bytes() + b"post-validation mutation")
        return result

    monkeypatch.setattr(finalizer, "_validate_review", mutate_after_review)

    with pytest.raises(ContractError, match="changed after validation"):
        _finalize(fixture)


def test_finalizer_rejects_symlinked_report_parent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "evidence")
    linked_parent = tmp_path / "linked-control"
    linked_parent.symlink_to(fixture["control_dir"], target_is_directory=True)
    fixture["control_report_path"] = linked_parent / "report.json"

    with pytest.raises(ContractError, match="traverses a symlink"):
        _finalize(fixture)


def test_finalizer_requires_every_exact_accept_decision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["review"]["decisions"].pop()
    _write_review(fixture)

    with pytest.raises(ContractError, match="decide every required item"):
        _finalize(fixture)


def test_finalizer_rejects_duplicate_json_keys_and_wrong_bound_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["review_path"].write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate key"):
        _finalize(fixture)

    fixture = _fixture(tmp_path / "wrong-digest")
    fixture["comparison_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="comparison report SHA-256"):
        _finalize(fixture)


def test_finalizer_refuses_existing_output_before_evidence_work(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["output_path"].write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _finalize(fixture)

    assert fixture["output_path"].read_text(encoding="utf-8") == "do not overwrite"

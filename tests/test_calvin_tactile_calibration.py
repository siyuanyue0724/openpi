from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_SOURCE_COMMIT,
    CALVIN_TACTILE_SOURCE_FILES_SHA256,
)
from picf_next.data.calvin_tactile_calibration import (
    CALVIN_TACTILE_BACKGROUND_ARCHIVE_SCHEMA,
    CALVIN_TACTILE_CALIBRATION_SCHEMA,
    CalvinTactileCalibrationSample,
    build_calvin_tactile_background_calibration,
    canonical_calibration_receipt_sha256,
    load_calvin_tactile_backgrounds,
    tactile_background_sha256,
)
from picf_next.data.dataset_manifest import file_sha256


def _sample(
    step: int,
    *,
    left_rgb: int,
    right_rgb: int,
    left_deformation: float,
    right_deformation: float,
) -> CalvinTactileCalibrationSample:
    rgb = np.empty((160, 120, 6), dtype=np.uint8)
    rgb[..., :3] = left_rgb
    rgb[..., 3:] = right_rgb
    deformation = np.empty((160, 120, 2), dtype=np.float32)
    deformation[..., 0] = left_deformation
    deformation[..., 1] = right_deformation
    rgb.setflags(write=False)
    deformation.setflags(write=False)
    return CalvinTactileCalibrationSample(
        source_global_index=step,
        source_file_sha256=hashlib.sha256(f"frame-{step}".encode()).hexdigest(),
        rgb=rgb,
        deformation_m=deformation,
    )


def test_background_calibration_selects_sensors_independently_and_uses_median() -> None:
    samples = tuple(
        _sample(
            step,
            left_rgb=255 if step == 3 else 20,
            right_rgb=40,
            left_deformation=0.0 if step < 18 else 2e-3,
            right_deformation=2e-3 if step < 2 else 0.0,
        )
        for step in range(20)
    )

    calibration = build_calvin_tactile_background_calibration(
        samples,
        background_noise_ceiling_m=1e-6,
        validity_thresholds_m={"left_digit": 1e-4, "right_digit": 1e-4},
        minimum_candidates_per_stream=8,
        maximum_selected_per_stream=16,
    )

    assert calibration.candidate_steps_by_stream["left_digit"] == tuple(range(18))
    assert calibration.candidate_steps_by_stream["right_digit"] == tuple(range(2, 20))
    assert len(calibration.selected_steps_by_stream["left_digit"]) == 16
    assert len(calibration.selected_steps_by_stream["right_digit"]) == 16
    assert np.all(calibration.backgrounds_by_stream["left_digit"] == 20.0)
    assert np.all(calibration.backgrounds_by_stream["right_digit"] == 40.0)
    assert not calibration.backgrounds_by_stream["left_digit"].flags.writeable
    assert (
        tactile_background_sha256(calibration.backgrounds_by_stream["left_digit"])
        == calibration.background_sha256_by_stream["left_digit"]
    )


def test_background_calibration_is_deterministic_and_rejects_invalid_support() -> None:
    samples = tuple(
        _sample(
            step,
            left_rgb=step,
            right_rgb=step + 1,
            left_deformation=0.0,
            right_deformation=0.0,
        )
        for step in range(20)
    )
    kwargs = {
        "background_noise_ceiling_m": 1e-6,
        "validity_thresholds_m": {"left_digit": 1e-4, "right_digit": 1e-4},
        "minimum_candidates_per_stream": 8,
        "maximum_selected_per_stream": 8,
    }

    first = build_calvin_tactile_background_calibration(samples, **kwargs)
    second = build_calvin_tactile_background_calibration(samples, **kwargs)

    assert first.receipt_payload() == second.receipt_payload()
    assert first.selected_steps_by_stream == second.selected_steps_by_stream
    with pytest.raises(ContractError, match="validity thresholds"):
        build_calvin_tactile_background_calibration(
            samples,
            background_noise_ceiling_m=1e-6,
            validity_thresholds_m={"left_digit": 1e-7, "right_digit": 1e-4},
            minimum_candidates_per_stream=8,
            maximum_selected_per_stream=8,
        )
    with pytest.raises(ContractError, match="quiet frames"):
        build_calvin_tactile_background_calibration(
            samples[:4],
            **kwargs,
        )


def test_background_calibration_rejects_duplicate_or_unsorted_sources() -> None:
    first = _sample(
        1,
        left_rgb=1,
        right_rgb=2,
        left_deformation=0.0,
        right_deformation=0.0,
    )
    duplicate = _sample(
        1,
        left_rgb=3,
        right_rgb=4,
        left_deformation=0.0,
        right_deformation=0.0,
    )
    with pytest.raises(ContractError, match="source-unique and sorted"):
        build_calvin_tactile_background_calibration(
            (first, duplicate),
            background_noise_ceiling_m=1e-6,
            validity_thresholds_m={"left_digit": 1e-4, "right_digit": 1e-4},
            minimum_candidates_per_stream=1,
            maximum_selected_per_stream=1,
        )


def _published_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    samples = tuple(
        _sample(
            step,
            left_rgb=20,
            right_rgb=40,
            left_deformation=0.0,
            right_deformation=0.0,
        )
        for step in range(8)
    )
    calibration = build_calvin_tactile_background_calibration(
        samples,
        background_noise_ceiling_m=1e-6,
        validity_thresholds_m={"left_digit": 1e-4, "right_digit": 1e-4},
        minimum_candidates_per_stream=8,
        maximum_selected_per_stream=8,
    )
    archive = tmp_path / "backgrounds.npz"
    np.savez_compressed(
        archive,
        schema=np.asarray(CALVIN_TACTILE_BACKGROUND_ARCHIVE_SCHEMA),
        left_digit=calibration.backgrounds_by_stream["left_digit"],
        right_digit=calibration.backgrounds_by_stream["right_digit"],
        left_digit_selected_steps=np.asarray(
            calibration.selected_steps_by_stream["left_digit"], dtype=np.int64
        ),
        right_digit_selected_steps=np.asarray(
            calibration.selected_steps_by_stream["right_digit"], dtype=np.int64
        ),
    )
    tree_sha256 = hashlib.sha256(b"tree").hexdigest()
    receipt_payload = {
        "schema": CALVIN_TACTILE_CALIBRATION_SCHEMA,
        "dataset": {
            "dataset_id": "fixture",
            "dataset_revision": "fixture-revision",
            "file_count": 10,
            "manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
            "split_name": "training",
            "tree_sha256": tree_sha256,
        },
        "sampling": {
            "sample_count": 8,
            "sampled_steps_sha256": hashlib.sha256(b"steps").hexdigest(),
            "tactile_audit_sha256": hashlib.sha256(b"audit").hexdigest(),
            "visual_review_manifest_sha256": hashlib.sha256(b"review").hexdigest(),
        },
        "official_calvin_source": {
            "commit": CALVIN_TACTILE_SOURCE_COMMIT,
            "files_sha256": CALVIN_TACTILE_SOURCE_FILES_SHA256,
        },
        "calibration": calibration.receipt_payload(),
        "archive": {"path": str(archive.resolve()), "sha256": file_sha256(archive)},
    }
    receipt_payload["receipt_payload_sha256"] = canonical_calibration_receipt_sha256(
        receipt_payload
    )
    receipt = tmp_path / "backgrounds.receipt.json"
    receipt.write_text(json.dumps(receipt_payload, sort_keys=True), encoding="ascii")
    return archive, receipt, file_sha256(receipt), tree_sha256


def test_calibration_loader_authenticates_receipt_archive_and_dataset_tree(
    tmp_path: Path,
) -> None:
    archive, receipt, receipt_sha256, tree_sha256 = _published_fixture(tmp_path)

    loaded = load_calvin_tactile_backgrounds(
        archive,
        receipt,
        receipt_sha256=receipt_sha256,
        dataset_tree_sha256=tree_sha256,
    )

    assert loaded.archive_sha256 == file_sha256(archive)
    assert loaded.receipt_sha256 == receipt_sha256
    assert loaded.validity_thresholds_m == {
        "left_digit": 1e-4,
        "right_digit": 1e-4,
    }
    assert np.all(loaded.backgrounds_by_stream["left_digit"] == 20.0)
    assert np.all(loaded.backgrounds_by_stream["right_digit"] == 40.0)
    with pytest.raises(ContractError, match="another dataset tree"):
        load_calvin_tactile_backgrounds(
            archive,
            receipt,
            receipt_sha256=receipt_sha256,
            dataset_tree_sha256=hashlib.sha256(b"wrong-tree").hexdigest(),
        )


def test_calibration_loader_rejects_archive_or_receipt_tampering(tmp_path: Path) -> None:
    archive, receipt, receipt_sha256, tree_sha256 = _published_fixture(tmp_path)
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ContractError, match="content hash mismatch"):
        load_calvin_tactile_backgrounds(
            archive,
            receipt,
            receipt_sha256=receipt_sha256,
            dataset_tree_sha256=tree_sha256,
        )

    archive, receipt, receipt_sha256, tree_sha256 = _published_fixture(tmp_path / "second")
    with receipt.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ContractError, match="content hash mismatch"):
        load_calvin_tactile_backgrounds(
            archive,
            receipt,
            receipt_sha256=receipt_sha256,
            dataset_tree_sha256=tree_sha256,
        )

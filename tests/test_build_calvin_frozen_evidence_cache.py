from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tools import build_calvin_frozen_evidence_cache as tool


def _args(**updates):
    values = {
        "asset_manifest": "assets.json",
        "camera_calibration": None,
        "device": "cuda",
        "modality": "vjepa",
        "point_budget": 4096,
        "point_pixel_stride": 2,
        "tactile_calibration_archive": None,
        "tactile_calibration_receipt": None,
        "tactile_calibration_receipt_sha256": None,
        "token_dtype": "float16",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_builder_constructs_only_the_requested_frozen_encoder(monkeypatch) -> None:
    calls = []

    def vjepa_from_manifest(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(
        tool.Vjepa21DenseEncoder, "from_manifest", staticmethod(vjepa_from_manifest)
    )
    monkeypatch.setattr(tool, "CalvinVjepa21EvidenceBuilder", lambda **kwargs: kwargs)

    result = tool._builder(_args(), object(), coverage_plan_sha256="c" * 64)

    assert set(result) == {
        "coverage_plan_sha256",
        "dataset",
        "encoder",
        "token_dtype",
    }
    assert calls == [(("assets.json",), {"device": "cuda", "verify_asset": True})]


def test_sonata_and_anytouch_require_their_physical_calibrations() -> None:
    with pytest.raises(ValueError, match="camera-calibration"):
        tool._builder(
            _args(modality="sonata"),
            object(),
            coverage_plan_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="tactile-calibration-archive"):
        tool._builder(
            _args(modality="anytouch"),
            object(),
            coverage_plan_sha256="c" * 64,
        )


def test_parser_defaults_to_small_resumable_shards_and_lossless_source_contract() -> None:
    parser = tool._parser()
    parsed = parser.parse_args(
        [
            "--dataset-root",
            "/dataset",
            "--dataset-manifest",
            "/manifest.json",
            "--asset-manifest",
            "/assets.json",
            "--modality",
            "vjepa",
            "--output-root",
            "/cache",
            "--coverage-plan",
            "/coverage.json",
            "--coverage-plan-sha256",
            "c" * 64,
        ]
    )

    assert parsed.shard_rows == 64
    assert parsed.encoder_batch_size == 1
    assert parsed.partition_count == 1
    assert parsed.partition_index == 0
    assert parsed.coverage_plan.as_posix() == "/coverage.json"
    assert parsed.coverage_plan_sha256 == "c" * 64
    assert parsed.token_dtype == "float16"
    assert parsed.verify_all_dataset_files is False


@pytest.mark.parametrize(
    ("count", "index", "expected"),
    ((4, 0, (0, 2)), (4, 1, (2, 5)), (4, 2, (5, 7)), (4, 3, (7, 10))),
)
def test_partition_bounds_are_contiguous_and_cover_every_record(
    count: int,
    index: int,
    expected: tuple[int, int],
) -> None:
    assert tool._partition_bounds(10, count, index) == expected


@pytest.mark.parametrize(("count", "index"), ((0, 0), (11, 0), (4, -1), (4, 4)))
def test_partition_bounds_reject_invalid_domains(count: int, index: int) -> None:
    with pytest.raises(ValueError, match="partition"):
        tool._partition_bounds(10, count, index)

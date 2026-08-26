from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from picf_next.data.calvin_geometry_schema import (
    CALVIN_M2_TRAIN_GEOMETRY_OFFSET,
    CALVIN_M2_TRAIN_GEOMETRY_SCALE,
    CALVIN_M2_TRAIN_VISIBLE_GEOMETRY_ROWS,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
)
from picf_next.geometry import PhysicalGeometryContract
from tools.transcode_calvin_geometry_chart import _transform_geometry

_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = _ROOT / "evidence/calvin_m2_geometry_chart/report.json"


def test_declared_chart_is_exactly_the_float32_train_visible_evidence() -> None:
    report = json.loads(_EVIDENCE.read_text())
    train = report["raw_geometry_statistics_m"]["train"]
    normalized = report["destination_normalized_statistics"]["train"]

    assert report["all_feature_sensor_hashes_match_physical_sidecar"] is True
    assert train["sample_count"] == 192
    assert train["object_rows"] == CALVIN_M2_TRAIN_VISIBLE_GEOMETRY_ROWS
    np.testing.assert_array_equal(
        [axis["mean"] for axis in train["axis"]],
        CALVIN_M2_TRAIN_GEOMETRY_OFFSET,
    )
    np.testing.assert_array_equal(
        [axis["standard_deviation"] for axis in train["axis"]],
        CALVIN_M2_TRAIN_GEOMETRY_SCALE,
    )
    np.testing.assert_allclose(
        [axis["mean"] for axis in normalized["axis"]],
        np.zeros(3),
        atol=3e-16,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [axis["variance_about_mean"] for axis in normalized["axis"]],
        np.ones(3),
        atol=1e-14,
        rtol=0.0,
    )


def test_geometry_chart_transcode_preserves_raw_values_and_variances() -> None:
    source = PhysicalGeometryContract(
        name="test.source",
        quantity="point",
        reference_frame="base",
        axes=("x", "y", "z"),
        units=("m", "m", "m"),
        normalization_offset=(1.0, 2.0, 3.0),
        normalization_scale=(2.0, 4.0, 8.0),
    )
    geometry = np.asarray(
        [[0.5, -0.25, 0.125], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    variance = np.asarray(
        [[0.25, 0.5, 0.75], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    supervised = np.asarray(
        [[True, True, True], [False, False, False]],
        dtype=np.bool_,
    )

    transformed, transformed_variance, error = _transform_geometry(
        geometry,
        variance,
        supervised,
        source=source,
        destination=CALVIN_OBJECT_GEOMETRY_CONTRACT,
    )

    raw = geometry[0].astype(np.float64) * np.asarray(source.normalization_scale) + np.asarray(
        source.normalization_offset
    )
    recovered = np.asarray(
        CALVIN_OBJECT_GEOMETRY_CONTRACT.denormalize_values(
            tuple(float(value) for value in transformed[0])
        )
    )
    raw_variance = variance[0].astype(np.float64) * np.square(source.normalization_scale)
    recovered_variance = np.asarray(
        CALVIN_OBJECT_GEOMETRY_CONTRACT.denormalize_variance(
            tuple(float(value) for value in transformed_variance[0])
        )
    )

    np.testing.assert_allclose(recovered, raw, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(recovered_variance, raw_variance, atol=2e-6, rtol=0.0)
    assert error <= 1e-6
    assert not transformed[1].any()
    assert not transformed_variance[1].any()


def test_bfloat16_rounded_statistics_do_not_define_the_geometry_chart() -> None:
    report = json.loads(_EVIDENCE.read_text())
    raw_means = np.asarray(
        [axis["mean"] for axis in report["raw_geometry_statistics_m"]["train"]["axis"]],
        dtype=np.float64,
    )
    legacy_bfloat16_means = np.asarray(
        [0.37967314512375894, 0.4591287317291782, 0.21797112003585217],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(raw_means, CALVIN_M2_TRAIN_GEOMETRY_OFFSET)
    assert float(np.max(np.abs(raw_means - legacy_bfloat16_means))) > 2.9e-4

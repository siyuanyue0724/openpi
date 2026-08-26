from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    source_array_sha256,
)
from tools import audit_calvin_physical_supervision as audit_tool
from tools.audit_calvin_physical_supervision import (
    _distribution,
    _full_tail_selections,
    _FullTailScan,
    _known_visible_owner_indices,
    _owner_overlay,
    _panel,
    _recomputed_manifest_summary,
    _scan_full_tail,
    _select_extreme_indices,
    _token_overlay,
    _validate_projection_source_samples,
    _validate_recomputed_manifest_summary,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools/audit_calvin_physical_supervision.py"


def test_physical_audit_directory_publication_is_atomic_and_cleans_failures(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"

    def prepare_success(partial: Path) -> dict[str, object]:
        partial.mkdir()
        (partial / "report.json").write_text('{"status":"PASS"}\n')
        return {"status": "PASS"}

    assert audit_tool._publish_audit_directory(output, prepare_success) == {"status": "PASS"}
    assert (output / "report.json").read_text() == '{"status":"PASS"}\n'
    assert not tuple(tmp_path.glob(".*.publish-lock"))
    assert not tuple(tmp_path.glob(".*.partial-*"))

    failed_output = tmp_path / "failed"

    def prepare_failure(partial: Path) -> dict[str, object]:
        partial.mkdir()
        (partial / "partial.json").write_text("incomplete\n")
        raise RuntimeError("injected audit failure")

    with pytest.raises(RuntimeError, match="injected audit failure"):
        audit_tool._publish_audit_directory(failed_output, prepare_failure)
    assert not failed_output.exists()
    assert not tuple(tmp_path.glob(".*.partial-*"))


def _projection(dataset_manifest_sha256: str = "a" * 64) -> dict[str, object]:
    def view(source_field: str, shape: list[int], digit: str) -> dict[str, object]:
        return {
            "source_field": source_field,
            "source_shape": shape,
            "image_grid_thw": [1, 16, 16],
            "merged_grid_hw": [8, 8],
            "raw_patch_count": 256,
            "merged_token_count": 64,
            "pixel_values_shape": [256, 1536],
            "source_rgb_sha256": [digit * 64] * 3,
        }

    return {
        "schema": "picf-next.lingbot-calvin-qwen-projection.v1",
        "status": "PASS",
        "runtime_input": False,
        "processor_id": "Qwen/Qwen3-VL-4B-Instruct",
        "processor_revision": "b" * 40,
        "processor_assets_sha256": "c" * 64,
        "processor_config_sha256": "d" * 64,
        "processor_preprocessor_config_sha256": "e" * 64,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_tree_sha256": "f" * 64,
        "source_frame_count": 10,
        "sample_global_indices": [0, 5, 9],
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "views": {
            "static": view("rgb_static", [200, 200, 3], "1"),
            "gripper": view("rgb_gripper", [84, 84, 3], "2"),
        },
        "transformers_version": "5.0.0",
    }


def test_external_sidecar_audit_view_requires_expected_hash() -> None:
    source = _TOOL.read_text(encoding="utf-8")

    assert '"--sidecar-manifest-sha256"' in source
    assert "external sidecar manifest requires its immutable SHA-256" in source
    assert "expected_sha256=sidecar_manifest_sha256" in source


def test_visual_owner_audit_excludes_unknown_pixels() -> None:
    rgb = np.zeros((27, 27, 3), dtype=np.uint8)
    owner = np.ones((27, 27), dtype=np.uint8)
    supervised = np.ones((27, 27), dtype=np.bool_)
    supervised[:, :13] = False

    overlay = _owner_overlay(rgb, owner, supervised, ("object/one",))

    assert _known_visible_owner_indices(owner, supervised) == (1,)
    assert not np.array_equal(overlay[10, 5], overlay[10, 20])
    assert np.any(np.all(overlay[:, :13] == np.asarray((255, 0, 255)), axis=-1))
    assert not np.any(np.all(overlay[:, 13:] == np.asarray((255, 0, 255)), axis=-1))


def test_visual_owner_audit_does_not_claim_unknown_only_identity() -> None:
    owner = np.ones((27, 27), dtype=np.uint8)
    supervised = np.zeros((27, 27), dtype=np.bool_)

    assert _known_visible_owner_indices(owner, supervised) == ()


def test_token_audit_uses_training_pixel_supervision() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    owner = np.ones((200, 200), dtype=np.uint8)
    known = np.ones((200, 200), dtype=np.bool_)
    unknown = np.zeros((200, 200), dtype=np.bool_)

    known_overlay = _token_overlay(
        rgb,
        owner,
        known,
        ("object/one",),
        projection=_projection(),
        camera_name="static",
    )
    unknown_overlay = _token_overlay(
        rgb,
        owner,
        unknown,
        ("object/one",),
        projection=_projection(),
        camera_name="static",
    )

    assert not np.array_equal(known_overlay, unknown_overlay)


def test_token_audit_visualizes_minority_object_mass_below_context() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    context_owner = np.zeros((200, 200), dtype=np.uint8)
    minority_owner = context_owner.copy()
    minority_owner[:25, :4] = 1
    supervised = np.ones((200, 200), dtype=np.bool_)

    context_overlay = _token_overlay(
        rgb,
        context_owner,
        supervised,
        ("object/one",),
        projection=_projection(),
        camera_name="static",
    )
    minority_overlay = _token_overlay(
        rgb,
        minority_owner,
        supervised,
        ("object/one",),
        projection=_projection(),
        camera_name="static",
    )

    assert not np.array_equal(minority_overlay[:25, :25], context_overlay[:25, :25])
    assert np.array_equal(minority_overlay[:25, 10:25], context_overlay[:25, 10:25])


def test_projection_audit_reopens_every_measured_source_image() -> None:
    projection = _projection()
    images = {
        "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
    }
    views = cast(dict[str, dict[str, Any]], projection["views"])
    for view in views.values():
        field = cast(str, view["source_field"])
        view["source_rgb_sha256"] = [source_array_sha256(field, images[field])] * 3

    class Index:
        @staticmethod
        def validated_source_frame_arrays(
            global_index: int,
            *,
            fields: tuple[str, ...],
        ) -> dict[str, np.ndarray]:
            assert global_index in {0, 5, 9}
            return {field: images[field] for field in fields}

    _validate_projection_source_samples(Index(), projection)  # type: ignore[arg-type]
    cast(list[str], views["static"]["source_rgb_sha256"])[1] = "0" * 64
    with pytest.raises(ContractError, match="source image differs"):
        _validate_projection_source_samples(Index(), projection)  # type: ignore[arg-type]


def test_visual_panel_expands_header_instead_of_truncating_audit_context() -> None:
    image = np.zeros((27, 27, 3), dtype=np.uint8)
    short = _panel(
        static=(image, image, image),
        gripper=(image, image, image),
        title="short",
        legend="short",
    )
    long = _panel(
        static=(image, image, image),
        gripper=(image, image, image),
        title="task=" + "long_task_name " * 20,
        legend="reasons=" + "metric_tail " * 80,
    )

    assert long.width == short.width
    assert long.height > short.height


def test_distribution_reports_missing_values_without_fabricating_quantiles() -> None:
    measured = _distribution(np.asarray([0.25, np.nan, 0.75], dtype=np.float64))
    missing = _distribution(np.asarray([np.nan, np.nan], dtype=np.float64))

    assert measured["count"] == 3
    assert measured["finite_count"] == 2
    assert measured["missing_count"] == 1
    assert measured["minimum"] == pytest.approx(0.25)
    assert measured["p50"] == pytest.approx(0.5)
    assert measured["maximum"] == pytest.approx(0.75)
    assert missing["finite_count"] == 0
    assert missing["minimum"] is None
    assert missing["maximum"] is None


def test_extreme_selection_is_value_then_global_index_deterministic() -> None:
    indices = np.asarray([10, 20, 30, 40], dtype=np.int64)
    values = np.asarray([0.5, np.nan, 0.5, 0.25], dtype=np.float64)

    assert _select_extreme_indices(indices, values, count=2, direction="high") == (10, 30)
    assert _select_extreme_indices(indices, values, count=2, direction="low") == (40, 10)


def _camera_arrays() -> dict[str, np.ndarray]:
    owner = np.asarray(
        [
            [[1, 1], [1, 1]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    supervised = np.asarray(
        [
            [[True, False], [True, False]],
            [[True, True], [True, True]],
        ],
        dtype=np.bool_,
    )
    return {
        "owner_index": owner,
        "owner_supervised": supervised,
        "rgb_mae": np.asarray([0.2, 0.1], dtype=np.float32),
        "depth_mae_m": np.asarray([0.02, 0.01], dtype=np.float32),
        "depth_p95_m": np.asarray([0.03, 0.015], dtype=np.float32),
        "depth_consistent_fraction": supervised.mean(axis=(1, 2)).astype(np.float32),
    }


def test_full_tail_scan_uses_canonical_validated_arrays_and_exact_coverage() -> None:
    loaded = SimpleNamespace(
        global_indices=np.asarray([2, 5], dtype=np.int64),
        camera_arrays={
            "static": _camera_arrays(),
            "gripper": _camera_arrays(),
        },
    )
    shard = SimpleNamespace(
        first_global_index=2,
        last_global_index=5,
        frame_count=2,
    )

    class FakeSidecar:
        coverage = CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
        shards = (shard,)
        cleared = False

        def _load_shard(self, shard_index: int) -> SimpleNamespace:
            assert shard_index == 0
            return loaded

        def clear_cache(self) -> None:
            self.cleared = True

    sidecar = FakeSidecar()
    scan = _scan_full_tail(sidecar, {"frame_count": 2})  # type: ignore[arg-type]

    assert sidecar.cleared is True
    np.testing.assert_array_equal(scan.global_indices, np.asarray([2, 5], dtype=np.int64))
    np.testing.assert_allclose(
        scan.series["static"]["known_owner_retention"],
        np.asarray([0.5, np.nan]),
        equal_nan=True,
    )
    assert scan.distributions["static"]["known_owner_retention"]["missing_count"] == 1
    assert scan.recomputed_manifest_summary == _recomputed_manifest_summary(scan.series)


def test_recomputed_manifest_summary_rejects_any_numeric_drift() -> None:
    camera = {
        "rgb_mae": np.asarray([0.1, 0.2]),
        "depth_mae_m": np.asarray([0.01, 0.02]),
        "depth_p95_m": np.asarray([0.015, 0.03]),
        "known_pixel_fraction": np.asarray([0.5, 1.0]),
    }
    measured = _recomputed_manifest_summary(
        {
            "static": camera,
            "gripper": camera,
        }
    )

    assert all(
        error == 0.0
        for error in _validate_recomputed_manifest_summary(dict(measured), measured).values()
    )
    drifted = dict(measured)
    drifted["maximum_static_rgb_mae"] += 1e-6
    with pytest.raises(ContractError, match="full-tail summary mismatch"):
        _validate_recomputed_manifest_summary(drifted, measured)


def test_full_tail_selection_unions_metric_time_and_task_strata() -> None:
    indices = np.arange(10, dtype=np.int64)
    series = {
        camera: {
            metric: np.linspace(0.0, 1.0, num=10, dtype=np.float64)
            for metric in (
                "rgb_mae",
                "depth_mae_m",
                "depth_p95_m",
                "known_pixel_fraction",
                "raw_object_pixel_fraction",
                "known_object_pixel_fraction",
                "known_owner_retention",
            )
        }
        for camera in ("static", "gripper")
    }
    scan = _FullTailScan(
        global_indices=indices,
        series=series,
        distributions={},
        recomputed_manifest_summary={},
    )
    index = SimpleNamespace(
        segments=(
            SimpleNamespace(start=2, end=4, task_key="move_blue", instruction="move blue"),
            SimpleNamespace(start=6, end=8, task_key="toggle_led", instruction="toggle led"),
        )
    )

    selections = _full_tail_selections(
        index,  # type: ignore[arg-type]
        scan,
        tail_per_metric=1,
        temporal_strata=3,
    )

    assert any("metric_tail:static:rgb_mae:high" in reasons for reasons in selections.values())
    assert any("temporal_stratum:001" in reasons for reasons in selections.values())
    assert "task_stratum:move_blue" in selections[3]
    assert "task_stratum:toggle_led" in selections[7]


def test_full_tail_report_uses_all_source_accessor_and_declares_no_training_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_indices = np.asarray([0], dtype=np.int64)
    series = {
        camera: {
            "rgb_mae": np.asarray([0.1]),
            "depth_mae_m": np.asarray([0.01]),
            "depth_p95_m": np.asarray([0.02]),
            "known_pixel_fraction": np.asarray([1.0]),
            "raw_object_pixel_fraction": np.asarray([1.0]),
            "known_object_pixel_fraction": np.asarray([1.0]),
            "known_owner_retention": np.asarray([1.0]),
        }
        for camera in ("static", "gripper")
    }
    scan = _FullTailScan(
        global_indices=global_indices,
        series=series,
        distributions={
            camera: {metric: _distribution(values) for metric, values in camera_series.items()}
            for camera, camera_series in series.items()
        },
        recomputed_manifest_summary=_recomputed_manifest_summary(series),
    )
    monkeypatch.setattr(audit_tool, "_scan_full_tail", lambda _sidecar, _manifest: scan)

    sidecar_root = tmp_path / "sidecar"
    sidecar_root.mkdir()
    manifest = {"calibration_summary": scan.recomputed_manifest_summary}
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (sidecar_root / "manifest.json").write_text('{"legacy_identity":true}\n')
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text("{}\n")
    dataset_manifest_sha256 = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()

    sources = {}
    cameras = []
    for camera_name, size in (("static", 200), ("gripper", 84)):
        owner = np.ones((size, size), dtype=np.uint8)
        supervised = np.ones((size, size), dtype=np.bool_)
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        depth = np.ones((size, size), dtype=np.float32)
        sources[f"rgb_{camera_name}"] = rgb
        sources[f"depth_{camera_name}"] = depth
        cameras.append(
            SimpleNamespace(
                camera_name=camera_name,
                owner_index=owner,
                owner_supervised=supervised,
                source_rgb_sha256=source_array_sha256(f"rgb_{camera_name}", rgb),
                source_depth_sha256=source_array_sha256(f"depth_{camera_name}", depth),
                rgb_mae=0.1,
                depth_mae_m=0.01,
                depth_p95_m=0.02,
            )
        )
    physical = SimpleNamespace(identity_keys=("object/one",), cameras=tuple(cameras))
    source_calls = []

    class FakeSidecar:
        root = sidecar_root
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        coverage = CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES

        def source_frame(self, global_index: int) -> SimpleNamespace:
            source_calls.append(global_index)
            return physical

    index = SimpleNamespace(
        segments=(
            SimpleNamespace(
                index=0,
                start=0,
                end=1,
                task_key="move_blue",
                instruction="move the blue block",
            ),
        ),
        validated_source_frame_arrays=lambda _global_index, fields: {
            field: sources[field] for field in fields
        },
    )
    output = tmp_path / "audit"

    audit_tool._run_full_tail_audit(
        index=index,  # type: ignore[arg-type]
        sidecar=FakeSidecar(),  # type: ignore[arg-type]
        sidecar_manifest_bytes=manifest_bytes,
        dataset_manifest_path=dataset_manifest,
        projection=_projection(dataset_manifest_sha256),
        projection_contract_sha256="9" * 64,
        output=output,
        tail_per_metric=1,
        temporal_strata=1,
    )

    report = json.loads((output / "audit_manifest.json").read_text())
    assert source_calls == [0]
    assert report["runtime_input"] is False
    assert report["task_used_for_owner_selection"] is False
    assert report["task_used_for_audit_selection"] is True
    assert report["selection_affects_training"] is False
    assert report["full_shard_schema_validation"] is True
    assert report["sidecar_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert report["format"] == "picf-next.calvin-physical-supervision-audit.v5"
    assert report["training_projection"]["views"]["static"]["merged_grid_hw"] == [8, 8]
    assert report["training_projection_contract_sha256"] == "9" * 64
    assert (
        report["training_supervision_policy"]["unknown_pixel_semantics"]
        == "zero-loss-mass-never-context"
    )
    assert (
        report["records"][0]["cameras"]["static"]["training_token_nonzero_observed_fraction"] == 1.0
    )
    assert report["records"][0]["cameras"]["static"]["training_token_mean_observed_fraction"] == 1.0
    assert report["record_count"] == 1
    assert (output / report["records"][0]["panel"]).is_file()

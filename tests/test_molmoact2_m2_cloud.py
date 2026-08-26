from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tools.run_molmoact2_m2_cloud as m2_runner  # noqa: E402
from picf_next.data.calvin_loss_targets import (  # noqa: E402
    CalvinSourceFrameLossTargetRequest,
)
from picf_next.models.core import PICFCoreConfig  # noqa: E402
from picf_next.models.discovery import ObjectDiscoveryConfig  # noqa: E402
from picf_next.models.evidence import ModalityProjectionSpec, NativeTokenBank  # noqa: E402
from picf_next.models.temporal import TemporalFilterConfig  # noqa: E402
from picf_next.training.molmoact2_m2 import (  # noqa: E402
    M2_GATE,
    M2_RECIPE_SCHEMA,
    load_molmoact2_m2_recipe,
    m2_recipe_report,
)
from tests.geometry_contract import synthetic_geometry_contract  # noqa: E402
from tools.finalize_molmoact2_m2 import (  # noqa: E402
    finalize_m2,
    validate_m2_visual_review,
)
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _M2_MACHINE_REPORTS,
    _record_group_identity,
    _regular_cpu_copy,
    _sha256,
    _validate_split_contract,
    materialize_persistent_sidecars,
    validate_m2_machine_decision,
    validate_prior_m1,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs/training/molmoact2_calvin_m2_representation.json"


def test_m2_recipe_is_strict_and_authorizes_only_current_frame_training(tmp_path: Path) -> None:
    recipe = load_molmoact2_m2_recipe(_CONFIG)
    report = m2_recipe_report(recipe, repository_root=_ROOT)

    assert recipe.schema == M2_RECIPE_SCHEMA
    assert recipe.gate == M2_GATE
    assert report["trainable_runtime_modules"] == ["projector", "discovery"]
    assert "posterior_filter" in report["forbidden_runtime_modules"]
    assert "action_expert" in report["forbidden_runtime_modules"]
    assert report["long_training_authorized"] is False
    assert set(recipe.splits.learned_segments).isdisjoint(
        recipe.splits.excluded_overlap_control_segments
    )
    foundation = recipe.load_foundation(_ROOT)
    assert foundation.set_loss_config.existence_weight == 2.0
    assert foundation.set_loss_config.ownership_ce_weight == 5.0
    assert foundation.set_loss_config.ownership_dice_weight == 5.0
    assert foundation.set_loss_config.localization_confidence_weight == 1.0

    changed = json.loads(_CONFIG.read_text())
    changed["optimization"]["steps"] = 201
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="20-200"):
        load_molmoact2_m2_recipe(path)


def test_m2_foundation_hash_fails_closed(tmp_path: Path) -> None:
    changed = json.loads(_CONFIG.read_text())
    changed["foundation_recipe_sha256"] = "0" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    recipe = load_molmoact2_m2_recipe(path)

    with pytest.raises(ValueError, match="SHA-256"):
        recipe.load_foundation(_ROOT)


def test_current_frame_checkpoint_keys_are_direct_full_core_prefixes() -> None:
    geometry = synthetic_geometry_contract(2)
    config = PICFCoreConfig(
        modality_specs=(ModalityProjectionSpec("vision", 5),),
        binding_dim=7,
        discovery=ObjectDiscoveryConfig(
            input_dim=7,
            hidden_dim=12,
            num_queries=3,
            num_layers=1,
            num_heads=3,
            address_dim=4,
            content_dim=6,
            geometry_dim=2,
            geometry_contract=geometry,
            initial_variance=0.1,
        ),
        temporal=TemporalFilterConfig(
            address_dim=4,
            content_dim=6,
            geometry_dim=2,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=12,
            num_layers=1,
            num_heads=3,
            geometry_contract=geometry,
        ),
        posterior_capacity=4,
    )
    current = config.build_current_frame()
    full = config.build()
    current_keys = set(current.state_dict())
    full_keys = set(full.state_dict())

    assert current_keys
    assert current_keys < full_keys
    assert all(key.startswith(("projector.", "discovery.")) for key in current_keys)
    incompatible = full.load_state_dict(current.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("posterior_filter.") for key in incompatible.missing_keys)

    bank = NativeTokenBank(
        modality="vision",
        tokens=torch.randn(2, 4, 5),
        valid=torch.ones(2, 4, dtype=torch.bool),
    )
    output = current((bank,))
    assert output.discovery.ownership.shape == (2, 4, 4)


def test_m2_feature_cache_copy_removes_inference_tensor_semantics() -> None:
    with torch.inference_mode():
        source = torch.randn(2, 3, dtype=torch.float32)
    assert source.is_inference()

    copied = _regular_cpu_copy(source, dtype=torch.bfloat16)

    assert copied.dtype == torch.bfloat16
    assert copied.device.type == "cpu"
    assert copied.is_contiguous()
    assert not copied.is_inference()


def test_m2_source_frame_record_builds_task_independent_target_request() -> None:
    request = m2_runner._request_from_record(
        {
            "sample_key": "source-frame-0000012",
            "global_index": 12,
            "target_request_contract": "source_frame",
            "source_sensor_sha256": [
                ["depth_gripper", "1" * 64],
                ["depth_static", "2" * 64],
                ["rgb_gripper", "3" * 64],
                ["rgb_static", "4" * 64],
            ],
        }
    )

    assert isinstance(request, CalvinSourceFrameLossTargetRequest)
    assert request.source_global_index == 12
    assert not hasattr(request, "segment_index")


def test_m2_cache_group_identity_supports_exactly_one_row_contract() -> None:
    assert _record_group_identity({"segment_index": 7}) == ("segment_index", 7)
    assert _record_group_identity({"source_block_index": 2}) == (
        "source_block_index",
        2,
    )
    with pytest.raises(ValueError, match="exactly one"):
        _record_group_identity({})
    with pytest.raises(ValueError, match="exactly one"):
        _record_group_identity({"segment_index": 1, "source_block_index": 2})
    with pytest.raises(ValueError, match="exactly one"):
        _record_group_identity({"segment_index": True})
    recipe = load_molmoact2_m2_recipe(_CONFIG)
    with pytest.raises(ValueError, match="training sample"):
        m2_runner._batch_plan([], recipe)


def test_m2_visual_overlay_distinguishes_unknown_from_context() -> None:
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    labels = np.asarray([[254, 255], [0, 255]], dtype=np.uint8)

    overlay = m2_runner._overlay(
        image,
        labels,
        object_count=1,
        unknown_label=254,
    )

    assert not np.array_equal(overlay[0, 0], image[0, 0])
    np.testing.assert_array_equal(overlay[0, 1], image[0, 1])
    assert not np.array_equal(overlay[1, 0], image[1, 0])


def test_validation_checkpoint_selection_prioritizes_representation_quality() -> None:
    stronger_representation = {
        "mean_object_dice": 0.40,
        "ownership_accuracy": 0.70,
        "exact_count_accuracy": 0.30,
        "count_mae": 1.0,
        "geometry_mae_physical": 0.05,
        "losses": {"loss_total": 4.0},
    }
    lower_loss_but_worse_representation = {
        "mean_object_dice": 0.20,
        "ownership_accuracy": 0.90,
        "exact_count_accuracy": 0.90,
        "count_mae": 0.1,
        "geometry_mae_physical": 0.01,
        "losses": {"loss_total": -20.0},
    }

    assert m2_runner._validation_selection_key(
        stronger_representation
    ) > m2_runner._validation_selection_key(lower_loss_but_worse_representation)


def test_geometry_metrics_keep_model_chart_and_physical_units_distinct() -> None:
    contract = synthetic_geometry_contract(2)
    contract = type(contract)(
        name=contract.name,
        quantity=contract.quantity,
        reference_frame=contract.reference_frame,
        axes=contract.axes,
        units=("m", "m"),
        normalization_offset=(10.0, -5.0),
        normalization_scale=(0.5, 2.0),
    )

    metrics = m2_runner._geometry_metric_payload(
        contract=contract,
        model_chart_absolute_by_axis=(4.0, 6.0),
        supervised_coordinate_count_by_axis=(2, 3),
    )

    assert metrics["geometry_mae_model_chart"] == pytest.approx(2.0)
    assert metrics["geometry_mae_physical"] == pytest.approx(2.8)
    assert metrics["geometry_mae_physical_unit"] == "m"
    assert metrics["geometry_mae_by_axis"] == [
        {
            "axis": contract.axes[0],
            "unit": "m",
            "normalization_scale": 0.5,
            "supervised_coordinate_count": 2,
            "mae_model_chart": 2.0,
            "mae_physical": 1.0,
        },
        {
            "axis": contract.axes[1],
            "unit": "m",
            "normalization_scale": 2.0,
            "supervised_coordinate_count": 3,
            "mae_model_chart": 2.0,
            "mae_physical": 4.0,
        },
    ]


def test_geometry_metrics_do_not_average_incompatible_physical_units() -> None:
    contract = synthetic_geometry_contract(2)
    contract = type(contract)(
        name=contract.name,
        quantity=contract.quantity,
        reference_frame=contract.reference_frame,
        axes=contract.axes,
        units=("m", "rad"),
        normalization_offset=(0.0, 0.0),
        normalization_scale=(1.0, 1.0),
    )

    metrics = m2_runner._geometry_metric_payload(
        contract=contract,
        model_chart_absolute_by_axis=(1.0, 1.0),
        supervised_coordinate_count_by_axis=(1, 1),
    )

    assert metrics["geometry_mae_model_chart"] == 1.0
    assert metrics["geometry_mae_physical"] is None
    assert metrics["geometry_mae_physical_unit"] is None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_m2_persistent_sidecars_materialize_as_verified_regular_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    artifact_root = tmp_path / "persistent"
    names = ("calvin_physical_supervision_v2", "calvin_geometry_training_v4")
    for index, name in enumerate(names):
        destination = repository / "data" / name
        source = artifact_root / name
        destination.mkdir(parents=True)
        source.mkdir(parents=True)
        shard_name = "part00000_shard000000.npz"
        source_shard = source / shard_name
        source_shard.write_bytes(f"sidecar-{index}".encode())
        manifest = {
            "schema": f"fixture-{index}",
            "shards": [{"path": shard_name, "sha256": _sha256(source_shard)}],
        }
        _write_json(destination / "manifest.json", manifest)
        _write_json(source / "manifest.json", manifest)
        (destination / shard_name).symlink_to(source_shard)

    monkeypatch.setattr(m2_runner, "_ROOT", repository)
    monkeypatch.setattr(m2_runner, "_is_under_mnt", lambda _path: True)
    first = materialize_persistent_sidecars(artifact_root)

    assert len(first["restored"]) == 2
    assert {row["materialization"] for row in first["restored"]} == {
        "copied_from_persistent_storage"
    }
    for name in names:
        materialized = repository / "data" / name / "part00000_shard000000.npz"
        assert materialized.is_file()
        assert not materialized.is_symlink()

    second = materialize_persistent_sidecars(artifact_root)
    assert {row["materialization"] for row in second["restored"]} == {
        "existing_verified_regular_file"
    }


def _segment(
    index: int,
    start: int,
    end: int,
    task_key: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        start=start,
        end=end,
        task_key=task_key,
        instruction=f"instruction-{index}",
        transition_count=end - start,
    )


def test_m2_split_contract_uses_half_open_source_ranges() -> None:
    recipe = load_molmoact2_m2_recipe(_CONFIG)
    segments = [
        _segment(0, 0, 10, "validation"),
        _segment(1, 10, 20, "heldout-1"),
        _segment(2, 20, 30, "heldout-2"),
        _segment(3, 30, 40, "overlap-a"),
        _segment(4, 40, 50, "train-4"),
        _segment(5, 50, 60, "overlap-b"),
        _segment(6, 35, 38, "overlap-a"),
        _segment(7, 60, 70, "heldout-7"),
        _segment(8, 55, 58, "overlap-b"),
    ]
    assets = SimpleNamespace(index=SimpleNamespace(segments=segments))

    report = _validate_split_contract(assets, recipe)

    assert report["learned_source_ranges_disjoint"] is True
    assert report["transition_counts"] == {"train": 30, "validation": 10, "heldout": 30}
    assert report["overlap_controls"][0]["intersection_start_end_exclusive"] == [35, 38]

    segments[1] = _segment(1, 9, 20, "heldout-1")
    with pytest.raises(ValueError, match="overlap in source frames"):
        _validate_split_contract(assets, recipe)


def _m2_machine_fixture(run_dir: Path) -> dict:
    visual = run_dir / "visuals" / "sample.png"
    visual.parent.mkdir(parents=True)
    visual.write_bytes(b"png")
    artifacts = {
        "schema": "picf-next.molmoact2-m2-visual-artifacts.v1",
        "gate": M2_GATE,
        "artifacts": [
            {
                "path": "visuals/sample.png",
                "sha256": _sha256(visual),
                "bytes": visual.stat().st_size,
            }
        ],
        "artifacts_sha256": m2_runner._canonical_sha256(
            [
                {
                    "path": "visuals/sample.png",
                    "sha256": _sha256(visual),
                    "bytes": visual.stat().st_size,
                }
            ]
        ),
        "all_splits_present": True,
        "all_learned_segments_present": True,
        "camera_views_per_artifact": 2,
    }
    _write_json(run_dir / "visual_artifacts.json", artifacts)
    for relative in _M2_MACHINE_REPORTS:
        path = run_dir / relative
        if not path.exists():
            _write_json(path, {"fixture": relative})
    hashes = {relative: _sha256(run_dir / relative) for relative in _M2_MACHINE_REPORTS}
    machine = {
        "schema": "picf-next.molmoact2-m2-machine-decision.v1",
        "gate": M2_GATE,
        "status": "PASS_PENDING_VISUAL_REVIEW",
        "checks": {"fixture": True},
        "failed_checks": [],
        "required_report_sha256": hashes,
        "later_gates_authorized": [],
    }
    _write_json(run_dir / "machine_decision.json", machine)
    return machine


def _visual_review(run_dir: Path) -> dict:
    return {
        "schema": "picf-next.molmoact2-m2-visual-review.v1",
        "status": "PASS",
        "gate": M2_GATE,
        "run_dir": str(run_dir.resolve()),
        "machine_decision_sha256": _sha256(run_dir / "machine_decision.json"),
        "visual_artifacts_sha256": _sha256(run_dir / "visual_artifacts.json"),
        "inspected_files": ["visuals/sample.png"],
        "reviewer": "test",
        "findings": ["ownership remains on the physical object in both camera views"],
        "physical_object_ownership_accepted": True,
        "multi_camera_accepted": True,
        "occlusion_cases_accepted": True,
        "fragmentation_accepted": True,
    }


def test_m2_machine_and_visual_finalizers_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _m2_machine_fixture(run_dir)
    validate_m2_machine_decision(run_dir)
    review = _visual_review(run_dir)
    validate_m2_visual_review(review, run_dir=run_dir)
    review_path = tmp_path / "review.json"
    _write_json(review_path, review)
    monkeypatch.setattr(
        "tools.finalize_molmoact2_m2._is_under_mnt",
        lambda _path: True,
    )
    decision = finalize_m2(run_dir=run_dir, visual_review_path=review_path)
    assert decision["status"] == "PASS"
    assert decision["later_gates_authorized"] == ["M3_structural_probe"]


def test_m2_visual_review_cannot_skip_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _m2_machine_fixture(run_dir)
    review = _visual_review(run_dir)
    review["inspected_files"] = []
    with pytest.raises(ValueError, match="every artifact"):
        validate_m2_visual_review(review, run_dir=run_dir)


def test_m2_visual_review_requires_complete_two_camera_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _m2_machine_fixture(run_dir)
    manifest_path = run_dir / "visual_artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["all_learned_segments_present"] = False
    _write_json(manifest_path, manifest)
    review = _visual_review(run_dir)

    with pytest.raises(ValueError, match="coverage"):
        validate_m2_visual_review(review, run_dir=run_dir)


def test_prior_m1_must_authorize_exactly_m2(tmp_path: Path) -> None:
    run_dir = tmp_path / "m1"
    run_dir.mkdir()
    _write_json(
        run_dir / "gate_decision.json",
        {
            "schema": "picf-next.molmoact2-m1-gate-decision.v1",
            "status": "PASS",
            "gate": "M1_typed_full_manifest",
            "required_report_sha256": {},
            "later_gates_authorized": [],
        },
    )
    with pytest.raises(ValueError, match="authorize exactly M2"):
        validate_prior_m1(run_dir)

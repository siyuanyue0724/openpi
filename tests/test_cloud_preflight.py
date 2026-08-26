from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tools.preflight_cloud import (
    _system_memory_gib,
    _validate_molmo_import_origins,
    runtime_preflight,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
MOLMO_CONFIG = ROOT / "configs/cloud/2xa100_40g_gates.json"
MOLMO_M4_CONFIG = ROOT / "configs/cloud/2xa100_40g_m4_action_adoption.json"
LINGBOT_CONFIG = ROOT / "configs/cloud/2xa100_40g_lingbot_ceiling.json"


@pytest.mark.parametrize(
    "entrypoint",
    [
        "tools/preflight_cloud.py",
        "tools/run_molmoact2_m0_cloud.py",
        "tools/smoke_molmoact2_lerobot_full_weight.py",
        "tools/train_molmoact2_calvin_picf.py",
    ],
)
def test_cloud_entrypoints_prefer_their_checkout_source(
    entrypoint: str,
    tmp_path: Path,
) -> None:
    fake_package = tmp_path / "picf_next"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("SOURCE = 'stale-environment'\n")
    command = (
        "import pathlib, runpy; "
        f"runpy.run_path({entrypoint!r}, run_name='deployment_import_probe'); "
        "import picf_next; "
        "print(pathlib.Path(picf_next.__file__).resolve())"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert Path(result.stdout.strip()).is_relative_to((ROOT / "src").resolve())


def _config(path: Path = MOLMO_CONFIG) -> dict:
    return json.loads(path.read_text())


def test_system_memory_uses_the_smallest_cgroup_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_memory_gib = 768
    page_size = 4096
    monkeypatch.setattr(
        "tools.preflight_cloud.os.sysconf",
        lambda name: page_size if name == "SC_PAGE_SIZE" else host_memory_gib * 2**30 // page_size,
    )
    unlimited = tmp_path / "memory.max"
    unlimited.write_text("max\n")
    limited = tmp_path / "memory.limit_in_bytes"
    limited.write_text(f"{180 * 2**30}\n")

    assert _system_memory_gib(cgroup_limit_paths=(unlimited, limited)) == 180.0


def test_system_memory_falls_back_to_host_for_missing_or_invalid_cgroup_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_memory_gib = 128
    page_size = 4096
    monkeypatch.setattr(
        "tools.preflight_cloud.os.sysconf",
        lambda name: page_size if name == "SC_PAGE_SIZE" else host_memory_gib * 2**30 // page_size,
    )
    invalid = tmp_path / "memory.max"
    invalid.write_text("not-a-limit\n")

    assert _system_memory_gib(cgroup_limit_paths=(tmp_path / "missing", invalid)) == host_memory_gib


@pytest.mark.parametrize("path", [MOLMO_CONFIG, MOLMO_M4_CONFIG, LINGBOT_CONFIG])
def test_frozen_two_a100_profiles_are_static_valid(path: Path) -> None:
    validate_config(_config(path))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("gpu_count", 1, "exactly two"),
        ("gpu_name_contains", "H100", "A100"),
        ("minimum_gpu_memory_gib", 20.0, "40 GB-class"),
        ("minimum_system_memory_gib", 32.0, "host memory"),
        ("minimum_free_storage_gib", 50.0, "storage"),
    ],
)
def test_hardware_downgrades_fail_closed(field: str, bad_value: object, message: str) -> None:
    config = _config()
    config["hardware"][field] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_runtime_targets_or_dense_token_pruning_fail_closed() -> None:
    config = _config()
    config["training_contract"]["object_runtime_targets_forbidden"] = False
    with pytest.raises(ValueError, match="object_runtime_targets_forbidden"):
        validate_config(config)

    config = _config()
    config["training_contract"]["native_dense_tokens_pruned"] = True
    with pytest.raises(ValueError, match="native_dense_tokens_pruned"):
        validate_config(config)

    config = _config()
    config["training_contract"]["molmo_native_prepool_same_forward_local_contract_ready"] = False
    with pytest.raises(ValueError, match="same-forward local contract"):
        validate_config(config)

    config = _config()
    config["training_contract"]["picf_dense_bank_contract"] = (
        "explicit_external_native_banks_until_M2"
    )
    with pytest.raises(ValueError, match="dense-bank contract"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("temporal_training_mode", "fixed_window"),
        ("gradient_transitions", 2),
        ("detached_context_frames", 2),
    ],
)
def test_production_temporal_path_is_single_transition(
    field: str,
    bad_value: object,
) -> None:
    config = _config()
    config["training_contract"][field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_superseded_fixed_window_temporal_smoke_is_rejected() -> None:
    config = _config()
    config["training_contract"]["diagnostic_temporal_smoke"] = {
        "authorized_gate": "M3_temporal_posterior",
        "gradient_transitions": 2,
        "detached_context_frames": 0,
        "require_temporal_positive_pairs": True,
    }
    with pytest.raises(ValueError, match="unrecognized fields"):
        validate_config(config)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("state_factorization",), "single_recurrent_vector"),
        (("loss_target_resolution",), "before_forward"),
        (("geometry_overshooting", "production_transition_shared"), False),
        (("geometry_overshooting", "future_image_encoding"), True),
        (("geometry_overshooting", "initial_horizons"), [1, 2, 4, 8]),
        (("geometry_overshooting", "episode_boundary_padding"), "repeat_last"),
        (("geometry_overshooting", "update_schedule", "fraction_denominator"), 4),
    ],
)
def test_picf_objective_semantics_are_frozen(path: tuple[str, ...], bad_value: object) -> None:
    config = _config()
    contract = config["training_contract"]["picf_objective_contract"]
    cursor = contract
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = bad_value

    with pytest.raises(ValueError, match="picf_objective_contract"):
        validate_config(config)


def test_cloud_schema_and_training_fields_fail_closed() -> None:
    config = _config()
    config["schema"] = "picf-next.cloud-gates.v1"
    with pytest.raises(ValueError, match="unsupported cloud gate schema"):
        validate_config(config)

    config = _config()
    config["training_contract"]["silent_experimental_toggle"] = True
    with pytest.raises(ValueError, match="unrecognized fields"):
        validate_config(config)

    config = _config()
    config["silent_top_level_toggle"] = True
    with pytest.raises(ValueError, match="top-level fields"):
        validate_config(config)


def test_cloud_profile_binds_the_exact_non_long_training_recipe() -> None:
    config = _config()
    config["training_recipe"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical SHA-256"):
        validate_config(config)

    config = _config()
    config["training_recipe"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="escapes the repository"):
        validate_config(config)


def test_m4_profile_is_action_only_and_enters_only_after_stationary_acceptance() -> None:
    config = _config(MOLMO_M4_CONFIG)
    assert config["initial_authorization"] == ["M4_action_adoption"]
    assert config["training_contract"]["picf_objective_contract"]["families"] == ["action"]
    validate_config(config)

    wrong_recipe = deepcopy(config)
    wrong_recipe["training_recipe"] = deepcopy(_config()["training_recipe"])
    with pytest.raises(ValueError, match="stage differs"):
        validate_config(wrong_recipe)

    wrong_gate = deepcopy(config)
    wrong_gate["initial_authorization"] = ["M0_full_weight_parity"]
    with pytest.raises(ValueError, match="entry authorization"):
        validate_config(wrong_gate)

    structural_leak = deepcopy(config)
    structural_leak["training_contract"]["picf_objective_contract"]["families"].append("set")
    with pytest.raises(ValueError, match="picf_objective_contract"):
        validate_config(structural_leak)


def test_only_first_profile_gate_is_initially_authorized() -> None:
    config = deepcopy(_config())
    config["initial_authorization"].append("M1_typed_full_manifest")
    with pytest.raises(ValueError, match="entry authorization"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("sample_plan_algorithm", "random-sampler"),
        ("checkpoint_backend", "model-only"),
        ("atomic_checkpoint_publication", False),
        ("matched_abc_common_plan_required", False),
        ("explicit_flow_randomness_required", False),
    ],
)
def test_molmo_training_control_cannot_be_downgraded(field: str, bad_value: object) -> None:
    config = _config()
    config["training_contract"][field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_config_stores_secret_names_not_values() -> None:
    config = _config()
    config["optional_secret_env"][0] = "HF_TOKEN=leaked"
    with pytest.raises(ValueError, match="not secret values"):
        validate_config(config)


def test_host_runtime_and_path_contracts_are_immutable() -> None:
    config = _config()
    config["host"]["checkpoint_revision"] = "main"
    with pytest.raises(ValueError, match="source/checkpoint"):
        validate_config(config)

    config = _config()
    del config["paths"]["dataset_root_env"]
    with pytest.raises(ValueError, match="path environment"):
        validate_config(config)

    config = _config()
    config["runtime"]["datasets"] = "3.6.0"
    with pytest.raises(ValueError, match="runtime contract"):
        validate_config(config)


def test_runtime_preflight_emits_nonsecret_molmo_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(name=f"NVIDIA A100-SXM4-{index}", total_memory=40 * 2**30)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("tools.preflight_cloud._system_memory_gib", lambda: 128.0)
    monkeypatch.setattr(
        "tools.preflight_cloud._runtime_versions",
        lambda config: {
            key: value for key, value in config["runtime"].items() if not key.endswith("_sha256")
        },
    )
    monkeypatch.setattr(
        "tools.preflight_cloud._git_head",
        lambda path: (
            config["host"]["training_source_commit"]
            if "lerobot" in path.name
            else config["host"]["source_commit"]
        ),
    )
    monkeypatch.setattr(
        "tools.preflight_cloud._sha256",
        lambda path: config["runtime"]["uv_lock_sha256"],
    )
    monkeypatch.setattr(
        "tools.preflight_cloud._validate_molmo_import_origins",
        lambda source, trainer: {"olmo": str(source), "lerobot": str(trainer)},
    )
    monkeypatch.setattr(
        "tools.bootstrap_molmoact2.validate_checkpoint",
        lambda path, **_kwargs: {"checkpoint_dir": str(path)},
    )
    monkeypatch.setattr(
        "tools.verify_molmoact2_lerobot_patch.detect_patch_state",
        lambda checkout, patch: "applied",
    )
    monkeypatch.setattr(
        "tools.preflight_cloud.shutil.disk_usage",
        lambda path: SimpleNamespace(free=300 * 2**30),
    )

    checkpoint_root = tmp_path / "checkpoints"
    dataset_root = tmp_path / "datasets"
    run_root = tmp_path / "runs"
    checkpoint_root.mkdir()
    dataset_root.mkdir()
    monkeypatch.setenv("PICF_CHECKPOINT_DIR", str(checkpoint_root))
    monkeypatch.setenv("PICF_DATASET_DIR", str(dataset_root))
    monkeypatch.setenv("PICF_RUN_DIR", str(run_root))
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")

    report = runtime_preflight(config, root=ROOT)

    assert report["status"] == "PASS"
    assert len(report["devices"]) == 2
    assert report["authorized_gates"] == ["M0_full_weight_parity"]
    assert report["training_source_head"] == config["host"]["training_source_commit"]
    assert set(report["runtime_import_origins"]) == {"olmo", "lerobot"}
    assert report["checkpoint"]["checkpoint_dir"].endswith("molmoact2")
    assert "must-not-appear" not in json.dumps(report)
    assert run_root.is_dir()


def test_molmo_runtime_import_origins_must_match_frozen_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    trainer = tmp_path / "trainer"
    olmo = source / "experiments/olmo/hf_model/modeling_molmoact2.py"
    lerobot = trainer / "src/lerobot/policies/molmoact2/modeling_molmoact2.py"
    wrong = tmp_path / "installed/lerobot/policies/molmoact2/modeling_molmoact2.py"
    for path in (olmo, lerobot, wrong):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    origins = {
        "olmo.hf_model.modeling_molmoact2": olmo,
        "lerobot.policies.molmoact2.modeling_molmoact2": lerobot,
    }
    monkeypatch.setattr(
        "tools.preflight_cloud.importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=str(origins[name])),
    )
    report = _validate_molmo_import_origins(source, trainer)
    assert report["olmo.hf_model.modeling_molmoact2"] == str(olmo.resolve())

    origins["lerobot.policies.molmoact2.modeling_molmoact2"] = wrong
    with pytest.raises(RuntimeError, match="resolves to"):
        _validate_molmo_import_origins(source, trainer)

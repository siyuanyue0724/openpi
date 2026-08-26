from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.build_calvin_dataset_manifest as calvin_manifest_tool
from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import (
    DATASET_RUNTIME_VERIFICATION_MODE,
    DatasetFileManifest,
    build_dataset_file_manifest,
    content_identified_dataset_manifest,
    load_dataset_file_manifest,
    read_verified_dataset_file,
    validate_dataset_files,
    validate_dataset_runtime_binding,
    validate_dataset_runtime_binding_report,
)


def _manifest(tmp_path: Path) -> tuple[Path, DatasetFileManifest]:
    split = tmp_path / "training"
    (split / "nested").mkdir(parents=True)
    (split / "a.bin").write_bytes(b"alpha")
    (split / "nested" / "b.bin").write_bytes(b"beta")
    manifest = build_dataset_file_manifest(
        split,
        dataset_id="fixture",
        dataset_revision="sha256:fixture",
        split_name="training",
        relative_paths=("nested/b.bin", "a.bin"),
    )
    return split, manifest


def test_dataset_manifest_roundtrips_and_validates_every_byte(tmp_path: Path) -> None:
    split, manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True))

    loaded = load_dataset_file_manifest(path)
    report = validate_dataset_files(
        loaded,
        split,
        dataset_id="fixture",
        dataset_revision="sha256:fixture",
        split_name="training",
    )

    assert loaded == manifest
    assert tuple(record.path for record in loaded.files) == ("a.bin", "nested/b.bin")
    assert loaded.record_for("a.bin").path == "a.bin"
    assert loaded.record_for("nested/b.bin").path == "nested/b.bin"
    with pytest.raises(ContractError, match="absent from the frozen manifest"):
        loaded.record_for("missing.bin")
    assert report["dataset_file_count"] == 2
    assert report["dataset_total_size_bytes"] == 9
    assert report["dataset_tree_sha256"] == manifest.tree_sha256
    assert (
        read_verified_dataset_file(
            loaded,
            split,
            "nested/b.bin",
            maximum_bytes=4,
        )
        == b"beta"
    )
    with pytest.raises(ContractError, match="byte limit"):
        read_verified_dataset_file(
            loaded,
            split,
            "nested/b.bin",
            maximum_bytes=3,
        )


def test_content_identity_excludes_declared_identity_but_preserves_file_bytes(
    tmp_path: Path,
) -> None:
    _split, manifest = _manifest(tmp_path)

    identified = content_identified_dataset_manifest(
        manifest,
        dataset_id="calvin.cs.uni-freiburg.de/task_ABC_D",
    )

    assert identified.files is manifest.files
    assert identified.content_sha256 == manifest.content_sha256
    assert identified.dataset_revision == f"sha256:{manifest.content_sha256}"
    assert identified.dataset_id == "calvin.cs.uni-freiburg.de/task_ABC_D"
    assert identified.tree_sha256 != manifest.tree_sha256
    assert DatasetFileManifest.from_dict(identified.to_dict()) == identified

    with pytest.raises(ContractError, match="dataset_id"):
        content_identified_dataset_manifest(manifest, dataset_id="")
    with pytest.raises(TypeError, match="DatasetFileManifest"):
        content_identified_dataset_manifest(object(), dataset_id="dataset")  # type: ignore[arg-type]


def test_parallel_dataset_manifest_is_identical_and_bounded(tmp_path: Path) -> None:
    split, sequential = _manifest(tmp_path)
    progress: list[tuple[int, int]] = []
    parallel = build_dataset_file_manifest(
        split,
        dataset_id="fixture",
        dataset_revision="sha256:fixture",
        split_name="training",
        relative_paths=("nested/b.bin", "a.bin"),
        maximum_workers=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert parallel == sequential
    assert progress == [(2, 2)]
    validation = validate_dataset_files(
        parallel,
        split,
        dataset_id="fixture",
        dataset_revision="sha256:fixture",
        split_name="training",
        maximum_workers=2,
    )
    assert validation["dataset_files_verified"] is True
    with pytest.raises(TypeError, match="maximum_workers"):
        build_dataset_file_manifest(
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
            relative_paths=("a.bin",),
            maximum_workers=False,
        )
    with pytest.raises(TypeError, match="maximum_workers"):
        validate_dataset_files(
            parallel,
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
            maximum_workers=False,
        )


def test_dataset_manifest_rejects_source_and_manifest_tampering(tmp_path: Path) -> None:
    split, manifest = _manifest(tmp_path)
    (split / "a.bin").write_bytes(b"ALPHA")
    with pytest.raises(ContractError, match="differs from frozen manifest"):
        validate_dataset_files(
            manifest,
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
        )
    payload = manifest.to_dict()
    payload["files"][0]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ContractError, match="tree SHA-256 changed"):
        DatasetFileManifest.from_dict(payload)

    unsorted = manifest.to_dict()
    unsorted["files"] = list(reversed(unsorted["files"]))  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="unique and sorted"):
        DatasetFileManifest.from_dict(unsorted)

    duplicate = manifest.to_dict()
    duplicate["files"] = [duplicate["files"][0], duplicate["files"][0]]  # type: ignore[index]
    duplicate["file_count"] = 2
    duplicate["total_size_bytes"] = 10
    with pytest.raises(ContractError, match="unique and sorted"):
        DatasetFileManifest.from_dict(duplicate)


def test_runtime_binding_is_explicit_and_runtime_reads_remain_fail_closed(
    tmp_path: Path,
) -> None:
    split, manifest = _manifest(tmp_path)
    binding = validate_dataset_runtime_binding(
        manifest,
        split,
        dataset_id="fixture",
        dataset_revision="sha256:fixture",
        split_name="training",
    )
    probe_digest = hashlib.sha256(b"picf-next.dataset-runtime-probes.v1\0")
    for record in (manifest.files[0], manifest.files[-1]):
        encoded_path = record.path.encode("utf-8")
        probe_digest.update(len(encoded_path).to_bytes(8, "big"))
        probe_digest.update(encoded_path)
        probe_digest.update(bytes.fromhex(record.sha256))

    assert binding == {
        "dataset_file_count": 2,
        "dataset_total_size_bytes": 9,
        "dataset_tree_sha256": manifest.tree_sha256,
        "dataset_manifest_self_consistent": True,
        "dataset_full_tree_rescanned": False,
        "dataset_runtime_verified_read_required": True,
        "dataset_runtime_probe_file_count": 2,
        "dataset_runtime_probe_sha256": probe_digest.hexdigest(),
        "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
    }
    wrong_probe_count = dict(binding)
    wrong_probe_count["dataset_runtime_probe_file_count"] = 1
    with pytest.raises(ContractError, match="probe count"):
        validate_dataset_runtime_binding_report(wrong_probe_count)

    (split / "a.bin").write_bytes(b"changed")
    with pytest.raises(ContractError, match="differs from frozen manifest|byte limit"):
        validate_dataset_runtime_binding(
            manifest,
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
        )
    with pytest.raises(ContractError, match="differs from frozen manifest|byte limit"):
        read_verified_dataset_file(
            manifest,
            split,
            "a.bin",
            maximum_bytes=16,
        )


def test_dataset_manifest_rejects_boolean_file_size(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    payload = manifest.to_dict()
    files = payload["files"]
    assert isinstance(files, list)
    record = files[0]
    assert isinstance(record, dict)
    record["size_bytes"] = True

    with pytest.raises(ContractError, match="file size must be a nonnegative integer"):
        DatasetFileManifest.from_dict(payload)


def test_dataset_manifest_rejects_identity_path_and_symlink_drift(tmp_path: Path) -> None:
    split, manifest = _manifest(tmp_path)
    with pytest.raises(ContractError, match="identity differs"):
        validate_dataset_files(
            manifest,
            split,
            dataset_id="wrong",
            dataset_revision="sha256:fixture",
            split_name="training",
            verify_hashes=False,
        )
    with pytest.raises(ContractError, match="normalized and relative"):
        build_dataset_file_manifest(
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
            relative_paths=("../a.bin",),
        )

    (split / "link.bin").symlink_to(split / "a.bin")
    with pytest.raises(ContractError, match="must not use symlinks"):
        build_dataset_file_manifest(
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
            relative_paths=("link.bin",),
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"outside")
    (split / "linked-directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError, match="must not use symlinks"):
        build_dataset_file_manifest(
            split,
            dataset_id="fixture",
            dataset_revision="sha256:fixture",
            split_name="training",
            relative_paths=("linked-directory/payload.bin",),
        )


def test_calvin_manifest_inventory_does_not_decode_pickle_metadata(tmp_path: Path) -> None:
    split = tmp_path / "training"
    (split / ".hydra").mkdir(parents=True)
    (split / "lang_annotations").mkdir()
    for relative in (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
        "scene_info.npy",
    ):
        (split / relative).write_bytes(b"not safe to decode")
    (split / "episode_0000002.npz").write_bytes(b"two")
    (split / "episode_0000001.npz").write_bytes(b"one")

    assert calvin_manifest_tool._calvin_source_relative_paths(split) == (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
        "scene_info.npy",
        "episode_0000001.npz",
        "episode_0000002.npz",
    )

    (split / "episode_bad.npz").write_bytes(b"bad")
    with pytest.raises(ContractError, match="filename is not canonical"):
        calvin_manifest_tool._calvin_source_relative_paths(split)


def test_calvin_manifest_cli_publishes_a_compact_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    split = tmp_path / "training"
    (split / ".hydra").mkdir(parents=True)
    (split / "lang_annotations").mkdir()
    for relative in (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
        "scene_info.npy",
    ):
        (split / relative).write_bytes(relative.encode("ascii"))
    (split / "episode_0000000.npz").write_bytes(b"frame")
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_calvin_dataset_manifest.py",
            "--split-root",
            str(split),
            "--output",
            str(output),
            "--dataset-id",
            "fixture/calvin",
            "--dataset-revision",
            "sha256:fixture-source",
        ],
    )

    calvin_manifest_tool.main()

    summary = json.loads(capsys.readouterr().out)
    manifest = load_dataset_file_manifest(output)
    assert summary == {
        "file_count": 6,
        "output": str(output.resolve()),
        "total_size_bytes": manifest.total_size_bytes,
        "tree_sha256": manifest.tree_sha256,
    }
    assert manifest.dataset_id == "fixture/calvin"
    assert manifest.dataset_revision == "sha256:fixture-source"


def test_calvin_manifest_uses_shared_durable_exclusive_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "manifest.json"
    payload = {"schema": "fixture", "value": 3}
    published: list[tuple[Path, bytes]] = []
    original = calvin_manifest_tool.write_bytes_durable_exclusive

    def tracked(path: Path, encoded: bytes) -> Path:
        published.append((path, encoded))
        return original(path, encoded)

    monkeypatch.setattr(calvin_manifest_tool, "write_bytes_durable_exclusive", tracked)
    calvin_manifest_tool._atomic_write_json(output, payload)

    assert published == [
        (
            output,
            json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n",
        )
    ]
    assert json.loads(output.read_text(encoding="ascii")) == payload
    with pytest.raises(FileExistsError):
        calvin_manifest_tool._atomic_write_json(output, payload)


@pytest.mark.parametrize(
    "relative_path",
    [
        "tools/build_calvin_normalization.py",
        "tools/build_calvin_physical_supervision.py",
        "tools/audit_calvin_physical_supervision.py",
        "tools/build_lingbot_calvin_current_grid_cache.py",
        "tools/build_lingbot_calvin_predictive_cache.py",
        "tools/build_lingbot_representation_split.py",
        "tools/audit_lingbot_dino_teacher_causality.py",
        "tools/audit_lingbot_predictive_temporal_targets.py",
        "tools/run_lingbot_vla2_native_g0.py",
        "tools/run_lingbot_vla2_native_full.py",
    ],
)
def test_native_calvin_consumers_use_verified_reads_without_full_tree_rescans(
    relative_path: str,
) -> None:
    source = (Path(__file__).parents[1] / relative_path).read_text(encoding="utf-8")

    assert "validate_dataset_runtime_binding(" in source
    assert "validate_dataset_files(" not in source
    assert "verify_files=False" in source

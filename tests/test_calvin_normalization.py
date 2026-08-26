from __future__ import annotations

import copy
import json
import sys
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import picf_next.data.calvin_normalization as normalization_module
import tools.build_calvin_normalization as calvin_normalization_tool
import tools.build_lingbot_calvin_norm_stats as lingbot_normalization_tool
from picf_next.contracts import ContractError
from picf_next.data.calvin import CALVIN_ACTION_AXES, CALVIN_STATE_AXES, CalvinDatasetIndex
from picf_next.data.calvin_normalization import (
    build_calvin_normalization_artifact,
    load_calvin_normalization_artifact,
    official_lingbot_calvin_norm_stats,
    official_molmoact2_dataset_stats,
    validate_calvin_normalization_artifact,
    validate_lingbot_calvin_norm_stats,
    write_calvin_normalization_artifact,
)
from tests.test_calvin_data import _split_manifest, _write_split


def _artifact(tmp_path: Path) -> dict[str, object]:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    return build_calvin_normalization_artifact(index)


def test_calvin_normalization_matches_the_ordered_language_training_manifest(
    tmp_path: Path,
) -> None:
    payload = _artifact(tmp_path)

    assert payload["sample_count"] == 8
    assert payload["unique_source_frame_count"] == 7
    assert isinstance(payload["dataset_tree_sha256"], str)
    assert len(payload["dataset_tree_sha256"]) == 64
    assert payload["state"]["axes"] == list(CALVIN_STATE_AXES)
    assert payload["action"]["axes"] == list(CALVIN_ACTION_AXES)
    assert payload["state"]["normalize_mask"][6] is False
    assert payload["state"]["normalize_mask"][14] is False
    assert payload["action"]["normalize_mask"][6] is False
    np.testing.assert_allclose(payload["action"]["min"][0], 0.0)
    np.testing.assert_allclose(payload["action"]["max"][0], 0.3)
    np.testing.assert_allclose(
        payload["action"]["std"][0],
        np.std(np.asarray([0.0, 0.05, 0.1, 0.15, 0.15, 0.2, 0.25, 0.3])),
    )

    stats = official_molmoact2_dataset_stats(payload)
    assert stats["observation.state"]["q01"].shape == (15,)
    assert stats["action"]["q99"].shape == (7,)
    assert stats["action"]["mask"].dtype == np.bool_


def test_calvin_normalization_reads_unique_frames_once_and_parallel_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    manifest = _split_manifest(split)
    sequential_index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=manifest,
    )
    expected = build_calvin_normalization_artifact(sequential_index, maximum_workers=1)

    parallel_index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=manifest,
    )
    original = parallel_index.state_and_action
    calls: list[int] = []
    calls_lock = threading.Lock()

    def tracked(global_index: int) -> tuple[np.ndarray, np.ndarray]:
        with calls_lock:
            calls.append(global_index)
        return original(global_index)

    monkeypatch.setattr(parallel_index, "state_and_action", tracked)
    progress: list[tuple[int, int]] = []
    actual = build_calvin_normalization_artifact(
        parallel_index,
        maximum_workers=4,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert actual == expected
    assert Counter(calls) == Counter({global_index: 1 for global_index in range(10, 17)})
    assert progress == [(7, 7)]


@pytest.mark.parametrize("maximum_workers", [0, -1, True, 1.5, "4"])
def test_calvin_normalization_rejects_invalid_worker_count(
    tmp_path: Path,
    maximum_workers: object,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )

    with pytest.raises(TypeError, match="positive integer"):
        build_calvin_normalization_artifact(
            index,
            maximum_workers=maximum_workers,  # type: ignore[arg-type]
        )


def test_calvin_normalization_roundtrip_is_atomic_and_hash_checked(tmp_path: Path) -> None:
    payload = _artifact(tmp_path)
    output = tmp_path / "stats" / "calvin.json"

    assert write_calvin_normalization_artifact(payload, output) == output
    assert load_calvin_normalization_artifact(output) == payload
    with pytest.raises(FileExistsError):
        write_calvin_normalization_artifact(payload, output)

    changed = json.loads(output.read_text())
    changed["action"]["mean"][0] += 0.001
    with pytest.raises(ContractError, match="artifact SHA-256"):
        validate_calvin_normalization_artifact(changed)


def test_calvin_normalization_uses_shared_durable_exclusive_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact(tmp_path)
    output = tmp_path / "stats" / "calvin.json"
    published: list[tuple[Path, bytes]] = []
    original = normalization_module.write_bytes_durable_exclusive

    def tracked(path: Path, encoded: bytes) -> Path:
        published.append((path, encoded))
        return original(path, encoded)

    monkeypatch.setattr(normalization_module, "write_bytes_durable_exclusive", tracked)
    assert write_calvin_normalization_artifact(payload, output) == output
    assert published == [
        (
            output,
            json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n",
        )
    ]


def test_calvin_normalization_propagates_publication_failure_and_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact(tmp_path)
    output = tmp_path / "stats" / "calvin.json"

    def fail_publication(path: Path, encoded: bytes) -> Path:
        del path, encoded
        raise OSError("injected shared publication failure")

    monkeypatch.setattr(normalization_module, "write_bytes_durable_exclusive", fail_publication)
    with pytest.raises(OSError, match="shared publication"):
        write_calvin_normalization_artifact(payload, output)
    assert not output.exists()

    monkeypatch.undo()
    target = tmp_path / "target.json"
    target.write_text("do not replace\n", encoding="ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_calvin_normalization_artifact(payload, output)
    assert target.read_text(encoding="ascii") == "do not replace\n"


def test_lingbot_normalization_translation_preserves_source_identity_and_axes(
    tmp_path: Path,
) -> None:
    source = _artifact(tmp_path)
    dataset_tree_sha256 = source["dataset_tree_sha256"]
    assert isinstance(dataset_tree_sha256, str)
    translated = official_lingbot_calvin_norm_stats(
        source,
        dataset_tree_sha256=dataset_tree_sha256,
    )
    validate_lingbot_calvin_norm_stats(translated)
    assert translated["count"] == source["sample_count"]
    assert translated["source"]["artifact_sha256"] == source["artifact_sha256"]
    assert translated["source"]["dataset_tree_sha256"] == dataset_tree_sha256
    assert translated["source"]["unique_source_frame_count"] == source["unique_source_frame_count"]
    stats = translated["norm_stats"]
    assert set(stats) == {
        "action.effector.position",
        "action.end.position",
        "observation.state.arm.position",
        "observation.state.effector.position",
        "observation.state.end.position",
    }
    np.testing.assert_allclose(
        stats["observation.state.arm.position"]["q01"],
        source["state"]["q01"][7:14],
    )
    np.testing.assert_allclose(
        stats["action.end.position"]["q99"],
        source["action"]["q99"][:6],
    )

    changed = copy.deepcopy(translated)
    changed["norm_stats"]["action.end.position"]["mean"][0] += 0.01
    with pytest.raises(ContractError, match="artifact SHA-256"):
        validate_lingbot_calvin_norm_stats(changed)

    wrong_source = copy.deepcopy(translated)
    wrong_source["source"]["schema"] = "wrong"
    with pytest.raises(ContractError, match="source schema"):
        validate_lingbot_calvin_norm_stats(wrong_source)

    with pytest.raises(ContractError, match="dataset tree"):
        official_lingbot_calvin_norm_stats(
            source,
            dataset_tree_sha256="not-a-digest",
        )


def test_calvin_to_lingbot_normalization_cli_pipeline_is_tree_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    manifest = _split_manifest(split)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="ascii",
    )
    calvin_output = tmp_path / "calvin-normalization.json"
    lingbot_output = tmp_path / "lingbot-normalization.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_calvin_normalization.py",
            "--split-root",
            str(split),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(calvin_output),
            "--dataset-id",
            manifest.dataset_id,
            "--dataset-revision",
            manifest.dataset_revision,
            "--workers",
            "2",
        ],
    )
    calvin_normalization_tool.main()
    calvin_stdout = json.loads(capsys.readouterr().out)
    calvin_payload = load_calvin_normalization_artifact(calvin_output)
    assert calvin_stdout == calvin_payload
    assert calvin_payload["dataset_tree_sha256"] == manifest.tree_sha256

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lingbot_calvin_norm_stats.py",
            "--calvin-normalization",
            str(calvin_output),
            "--dataset-manifest",
            str(manifest_path),
            "--output",
            str(lingbot_output),
        ],
    )
    lingbot_normalization_tool.main()
    lingbot_payload = json.loads(lingbot_output.read_text(encoding="ascii"))
    assert json.loads(capsys.readouterr().out) == lingbot_payload
    validate_lingbot_calvin_norm_stats(lingbot_payload)
    assert lingbot_payload["source"]["dataset_tree_sha256"] == manifest.tree_sha256
    assert lingbot_payload["source"]["artifact_sha256"] == calvin_payload["artifact_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.__setitem__("unknown", 1), "fields differ"),
        (
            lambda payload: payload["state"].__setitem__("axes", ["wrong"] * 15),
            "axes differ",
        ),
        (
            lambda payload: payload["action"].__setitem__("normalize_mask", [True] * 7),
            "gripper semantics",
        ),
        (
            lambda payload: payload["action"].__setitem__("q01", [2.0] * 7),
            "quantiles are not ordered",
        ),
    ],
)
def test_calvin_normalization_rejects_contract_drift(tmp_path: Path, mutate, message: str) -> None:
    payload = copy.deepcopy(_artifact(tmp_path))
    mutate(payload)
    with pytest.raises(ContractError, match=message):
        validate_calvin_normalization_artifact(payload)

# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS
from picf_next.data.molmoact2_source_cache import MolmoAct2SourceFeatureCache


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def _layout() -> list[dict[str, object]]:
    return [
        {
            "image_key": "observation.images.image",
            "start": 0,
            "stop": 2,
            "image_num_crops": 1,
            "patches_per_crop": 2,
            "image_grid": [1, 1, 0, 0],
            "image_token_pooling": [[0, 1]],
        },
        {
            "image_key": "observation.images.wrist_image",
            "start": 2,
            "stop": 4,
            "image_num_crops": 1,
            "patches_per_crop": 2,
            "image_grid": [1, 1, 0, 0],
            "image_token_pooling": [[0, 1]],
        },
    ]


def _sensor_hashes(seed: str) -> list[list[str]]:
    names = sorted(
        str(spec[field])
        for spec in CALVIN_CAMERA_SPECS
        for field in ("source_rgb_field", "source_depth_field")
    )
    return [[name, hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()] for name in names]


def _write_cache(
    root: Path,
    *,
    shard_payloads: tuple[dict[str, torch.Tensor], ...] | None = None,
    mutate_manifest=None,
) -> tuple[str, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    if shard_payloads is None:
        shard_payloads = tuple(
            {
                "tokens": torch.full((1, 4, 3), value, dtype=torch.bfloat16),
                "valid": torch.ones(1, 4, dtype=torch.bool),
            }
            for value in (1.0, 2.0)
        )
    shards = []
    records = []
    for shard_index, payload in enumerate(shard_payloads):
        name = f"features-{shard_index:05d}.pt"
        path = root / name
        torch.save(payload, path)
        content = path.read_bytes()
        rows = int(payload["tokens"].shape[0])
        shards.append({"path": name, "sha256": _sha(content), "rows": rows, "bytes": len(content)})
        for row in range(rows):
            global_index = 100 + len(records)
            records.append(
                {
                    "sample_key": f"source-frame-{global_index:07d}",
                    "split": "train",
                    "source_block_index": 0,
                    "global_index": global_index,
                    "task_key": "task-independent-source-frame",
                    "instruction": "task field absent",
                    "target_request_contract": "source_frame",
                    "source_sensor_sha256": _sensor_hashes(str(global_index)),
                    "shard": name,
                    "row": row,
                }
            )
    layout = _layout()
    manifest: dict[str, object] = {
        "schema": "picf-next.molmoact2-m2-feature-cache.v1",
        "gate": "M2_representation_source_coverage_root_cause",
        "checkpoint_id": "allenai/MolmoAct2",
        "checkpoint_revision": "e" * 40,
        "foundation_recipe_sha256": "a" * 64,
        "source_coverage_recipe_sha256": "b" * 64,
        "modality": "molmo_vision_patch",
        "dtype": "bfloat16",
        "token_shape": [4, 3],
        "processor_layout": layout,
        "processor_layout_sha256": _canonical_sha(layout),
        "records": records,
        "records_sha256": _canonical_sha(records),
        "shards": shards,
        "sample_count": len(records),
        "model_input_fields": ["tokens", "valid"],
        "loss_target_fields_in_feature_shards": [],
        "task_field_supplied": False,
        "elapsed_s": 1.0,
        "cuda_peak_allocated_bytes": 1,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    payload = json.dumps(manifest, sort_keys=True).encode()
    (root / "manifest.json").write_bytes(payload)
    return _sha(payload), manifest


def _load(root: Path, manifest_sha256: str, *, capacity: int = 1):
    return MolmoAct2SourceFeatureCache.load(
        root,
        manifest_sha256=manifest_sha256,
        expected_modality="molmo_vision_patch",
        expected_token_count=4,
        expected_token_dim=3,
        expected_checkpoint_id="allenai/MolmoAct2",
        expected_checkpoint_revision="e" * 40,
        memory_capacity=capacity,
    )


def test_source_cache_is_lazy_bounded_and_exposes_only_observation_bank(tmp_path: Path) -> None:
    manifest_sha, _manifest = _write_cache(tmp_path)
    cache = _load(tmp_path, manifest_sha)

    assert not cache._loaded
    first = cache.native_bank((100,), device="cpu")
    assert first.modality == "molmo_vision_patch"
    assert first.tokens.shape == (1, 4, 3)
    assert first.tokens.dtype == torch.bfloat16
    assert torch.equal(first.valid, torch.ones(1, 4, dtype=torch.bool))
    assert len(cache._loaded) == 1

    second = cache.native_bank((101,), device="cpu", dtype=torch.float32)
    assert second.tokens.dtype == torch.float32
    assert torch.equal(second.tokens, torch.full((1, 4, 3), 2.0))
    assert tuple(cache._loaded) == ("features-00001.pt",)

    request = cache.target_request(100)
    assert request.sample_key == "source-frame-0000100"
    assert request.source_global_index == 100
    layout = cache.vision_layout(2)
    assert len(layout.rows) == 2
    assert layout.tokens_per_row == 4


def test_source_cache_rejects_shard_tampering_and_target_leakage(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    manifest_sha, manifest = _write_cache(clean)
    cache = _load(clean, manifest_sha)
    first_shard = clean / str(manifest["shards"][0]["path"])
    first_shard.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="size or file type|content hash"):
        cache.native_bank((100,), device="cpu")

    leaked = tmp_path / "leaked"
    payload = {
        "tokens": torch.ones(1, 4, 3, dtype=torch.bfloat16),
        "valid": torch.ones(1, 4, dtype=torch.bool),
        "ownership": torch.ones(1, 4),
    }
    leak_sha, _manifest = _write_cache(leaked, shard_payloads=(payload,))
    with pytest.raises(ContractError, match="non-observation fields"):
        _load(leaked, leak_sha).native_bank((100,), device="cpu")


def test_source_cache_rejects_non_bijective_records_and_bad_tensor_shape(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"

    def duplicate_location(manifest: dict[str, object]) -> None:
        records = manifest["records"]
        assert isinstance(records, list)
        records[1]["shard"] = records[0]["shard"]
        records[1]["row"] = records[0]["row"]
        manifest["records_sha256"] = _canonical_sha(records)

    duplicate_sha, _manifest = _write_cache(duplicate, mutate_manifest=duplicate_location)
    with pytest.raises(ContractError, match="locations must be unique"):
        _load(duplicate, duplicate_sha)

    malformed = tmp_path / "malformed"
    bad_payload = {
        "tokens": torch.ones(1, 5, 3, dtype=torch.bfloat16),
        "valid": torch.ones(1, 5, dtype=torch.bool),
    }
    malformed_sha, _manifest = _write_cache(malformed, shard_payloads=(bad_payload,))
    with pytest.raises(ContractError, match="tensor contract"):
        _load(malformed, malformed_sha).native_bank((100,), device="cpu")


def test_source_cache_rejects_task_input_and_manifest_hash_drift(tmp_path: Path) -> None:
    def add_task(manifest: dict[str, object]) -> None:
        manifest["task_field_supplied"] = True

    manifest_sha, _manifest = _write_cache(tmp_path, mutate_manifest=add_task)
    with pytest.raises(ContractError, match="supplied a task field"):
        _load(tmp_path, manifest_sha)
    with pytest.raises(ContractError, match="content hash mismatch"):
        _load(tmp_path, "f" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("checkpoint_revision", "f" * 40, "checkpoint identity"),),
)
def test_source_cache_rejects_producer_identity_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        manifest[field] = value

    manifest_sha, _manifest = _write_cache(tmp_path, mutate_manifest=mutate)
    with pytest.raises(ContractError, match=message):
        _load(tmp_path, manifest_sha)


def test_source_cache_preserves_but_does_not_overbind_consumer_recipe_provenance(
    tmp_path: Path,
) -> None:
    def change_downstream_provenance(manifest: dict[str, object]) -> None:
        manifest["foundation_recipe_sha256"] = "f" * 64
        manifest["source_coverage_recipe_sha256"] = "e" * 64

    manifest_sha, _manifest = _write_cache(
        tmp_path,
        mutate_manifest=change_downstream_provenance,
    )
    cache = _load(tmp_path, manifest_sha)

    assert cache.foundation_recipe_sha256 == "f" * 64
    assert cache.source_coverage_recipe_sha256 == "e" * 64

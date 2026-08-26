from __future__ import annotations

import hashlib
import json
import stat
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import NativeVLInstructionRecord, native_vl_rgb_sha256
from picf_next.lingbot_native.public_vl_evidence import (
    PUBLIC_VL_SCHEDULE_FIELDS,
    public_vl_schedule_sha256,
    text_sha256,
)
from picf_next.lingbot_native.vl_cotraining import (
    QWEN3VL_ASSISTANT_HEADER_TOKEN_ID,
    QWEN3VL_END_OF_MESSAGE_TOKEN_ID,
)
from tools import audit_public_native_vl_processor_budget as audit
from tools.bootstrap_lingbot_vla2 import (
    PROCESSOR_ASSET_CONTRACT,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
    REQUIRED_PROCESSOR_FILES,
    asset_contract_manifest,
)


@dataclass
class _Record:
    record_id: str
    record_sha256: str
    family: str
    partition: str
    source_key: str
    source_row_index: int
    source_subindex: int
    priority_sha256: str
    user_text: str
    assistant_text: str
    image_file: str
    image_file_sha256: str
    image_rgb_sha256: str
    width: int
    height: int
    image: np.ndarray[Any, np.dtype[np.uint8]]


class _Manifest:
    def __init__(self, *, artifact_sha256: str = audit.ADR125_PUBLIC_ARTIFACT_SHA256) -> None:
        records: list[_Record] = []
        index = 0
        for family in ("referring", "vqa"):
            for partition, count in (("heldout", 32), ("train", 64)):
                for local_index in range(count):
                    image = np.asarray([[[index, 0, 0]]], dtype=np.uint8)
                    image.setflags(write=False)
                    records.append(
                        _Record(
                            record_id=f"{family}-{partition}-{local_index:03d}",
                            record_sha256=hashlib.sha256(f"record-{index}".encode()).hexdigest(),
                            family=family,
                            partition=partition,
                            source_key=family,
                            source_row_index=index,
                            source_subindex=0,
                            priority_sha256=hashlib.sha256(
                                f"priority-{family}-{partition}-{local_index}".encode()
                            ).hexdigest(),
                            user_text=f"question {index}",
                            assistant_text=f"answer {index}",
                            image_file=f"images/{index:03d}.png",
                            image_file_sha256=hashlib.sha256(f"file-{index}".encode()).hexdigest(),
                            image_rgb_sha256=native_vl_rgb_sha256(image),
                            width=1,
                            height=1,
                            image=image,
                        )
                    )
                    index += 1
        self.records = tuple(records)
        self.artifact_sha256 = artifact_sha256
        self.family_partition_counts = dict(audit.EXPECTED_COUNTS)
        self.materialized_record_ids: list[str] = []
        self.training_lookups: list[tuple[int, int]] = []
        self.sources = {
            family: SimpleNamespace(
                dataset_id=f"dataset/{family}",
                dataset_revision="b" * 40,
                source_file=f"{family}.parquet",
                source_file_sha256=hashlib.sha256(family.encode()).hexdigest(),
            )
            for family in ("referring", "vqa")
        }

    def records_for(self, family: str, partition: str) -> tuple[_Record, ...]:
        return tuple(
            record
            for record in self.records
            if record.family == family and record.partition == partition
        )

    def training_record_for_rank(self, *, optimizer_step: int, rank: int) -> _Record:
        self.training_lookups.append((optimizer_step, rank))
        family = "referring" if rank == 0 else "vqa"
        return self.records_for(family, "train")[optimizer_step]

    def materialize_record(self, record: _Record, *, artifact_root: Path) -> object:
        assert artifact_root.is_absolute()
        self.materialized_record_ids.append(record.record_id)
        return NativeVLInstructionRecord(
            record_id=record.record_id,
            family=record.family,  # type: ignore[arg-type]
            user_text=record.user_text,
            assistant_text=record.assistant_text,
            image=record.image,
        )


class _Processor:
    def __init__(
        self,
        *,
        grid: tuple[int, int, int] = (1, 16, 16),
        pixel_patch_count: int = 256,
        answer_token: int = 123,
        pixel_value: float = 1.0,
        extra_tensor_name: str | None = None,
        include_position_ids: bool = False,
    ) -> None:
        self.grid = grid
        self.pixel_patch_count = pixel_patch_count
        self.answer_token = answer_token
        self.pixel_value = pixel_value
        self.extra_tensor_name = extra_tensor_name
        self.include_position_ids = include_position_ids
        self.call_count = 0
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 16_777_216, "longest_edge": 16_777_216},
            patch_size=16,
            merge_size=2,
        )

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        self.call_count += 1
        input_ids = torch.tensor(
            [
                [
                    QWEN3VL_ASSISTANT_HEADER_TOKEN_ID,
                    0,
                    self.answer_token,
                    QWEN3VL_END_OF_MESSAGE_TOKEN_ID,
                    0,
                ]
            ],
            dtype=torch.long,
        )
        result = {
            "attention_mask": torch.ones_like(input_ids),
            "image_grid_thw": torch.tensor([self.grid], dtype=torch.long),
            "input_ids": input_ids,
            "pixel_values": torch.full(
                (self.pixel_patch_count, 4), self.pixel_value, dtype=torch.float32
            ),
        }
        if self.include_position_ids:
            result["position_ids"] = torch.arange(input_ids.shape[1]).reshape(1, 1, -1)
        if self.extra_tensor_name is not None:
            result[self.extra_tensor_name] = torch.ones((1,), dtype=torch.long)
        return result


def _dependency_environment(*, mutation: str = "") -> dict[str, object]:
    packages = []
    for index, (distribution, import_name) in enumerate(audit.DEPENDENCY_IMPORTS):
        if distribution == "transformers":
            version = audit.ADR125_TRANSFORMERS_VERSION
        elif distribution == "torch":
            version = str(torch.__version__).split("+", 1)[0]
        else:
            version = f"{index}.0"
        packages.append(
            {
                "distribution": distribution,
                "distribution_root": f"/runtime/site-packages/{distribution}",
                "generated_script_count": 0,
                "import_file": f"{import_name.replace('.', '/')}/__init__.py",
                "import_file_sha256": hashlib.sha256(
                    f"import-{distribution}-{mutation}".encode()
                ).hexdigest(),
                "import_file_size": index + 11,
                "import_name": import_name,
                "installed_file_count": index + 1,
                "installed_files_sha256": hashlib.sha256(
                    f"files-{distribution}-{mutation}".encode()
                ).hexdigest(),
                "record_sha256": hashlib.sha256(
                    f"record-{distribution}-{mutation}".encode()
                ).hexdigest(),
                "version": version,
            }
        )
    python_runtime = audit._python_runtime_contract()
    if mutation:
        python_content = dict(python_runtime)
        python_content.pop("sha256")
        python_content["executable_sha256"] = hashlib.sha256(
            f"python-{mutation}".encode()
        ).hexdigest()
        python_runtime = {
            **python_content,
            "sha256": hashlib.sha256(audit._canonical_bytes(python_content)).hexdigest(),
        }
    content = {"packages": packages, "python_runtime": python_runtime}
    return {**content, "sha256": hashlib.sha256(audit._canonical_bytes(content)).hexdigest()}


def test_dependency_contract_covers_pinned_processor_cpu_runtime_closure() -> None:
    assert set(audit.DEPENDENCY_DISTRIBUTIONS) == {
        "Jinja2",
        "MarkupSafe",
        "Pillow",
        "PyYAML",
        "certifi",
        "charset-normalizer",
        "filelock",
        "fsspec",
        "huggingface-hub",
        "idna",
        "mpmath",
        "networkx",
        "numpy",
        "packaging",
        "regex",
        "requests",
        "safetensors",
        "setuptools",
        "sympy",
        "tokenizers",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "typing-extensions",
        "urllib3",
    }


def _processor_identity(path: Path) -> dict[str, object]:
    inventory = [
        [name, size, digest] for name, (size, digest) in sorted(PROCESSOR_ASSET_CONTRACT.items())
    ]
    return {
        "dependency_environment": _dependency_environment(),
        "processor_assets": asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        "processor_dir": str(path.resolve()),
        "processor_file_inventory": inventory,
        "processor_file_inventory_sha256": hashlib.sha256(
            audit._canonical_bytes(inventory)
        ).hexdigest(),
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_load_protocol": audit.PROCESSOR_SNAPSHOT_PROTOCOL,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "required_processor_files": len(REQUIRED_PROCESSOR_FILES),
    }


def _processor_asset_identity(path: Path) -> dict[str, object]:
    identity = _processor_identity(path)
    identity.pop("dependency_environment")
    return identity


def _internal_report(
    tmp_path: Path,
    *,
    manifest: _Manifest | None = None,
    processor: _Processor | None = None,
    processor_identity: dict[str, object] | None = None,
) -> tuple[dict[str, object], _Manifest, _Processor]:
    active_manifest = _Manifest() if manifest is None else manifest
    active_processor = _Processor() if processor is None else processor
    _prepare_artifact_files(tmp_path, active_manifest)
    identity = _processor_identity(tmp_path) if processor_identity is None else processor_identity
    report = audit.build_public_native_vl_processor_budget_audit(
        manifest=active_manifest,  # type: ignore[arg-type]
        artifact_root=tmp_path.resolve(),
        manifest_file_sha256=audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
        processor=active_processor,
        processor_identity=identity,
        processor_dir=tmp_path,
    )
    return report, active_manifest, active_processor


def _prepare_artifact_files(root: Path, manifest: _Manifest) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for record in manifest.records:
        image = root / record.image_file
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"file-{record.source_row_index}".encode())


def _production_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "public-artifact"
    root.mkdir()
    manifest = root / audit.ADR125_PUBLIC_MANIFEST_FILE
    manifest.write_text("{}", encoding="ascii")
    processor = tmp_path / "processor"
    processor.mkdir()
    for relative in REQUIRED_PROCESSOR_FILES:
        path = processor / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    return root.resolve(), manifest.resolve(), processor.resolve()


def _allow_cpu_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(audit.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(audit.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(audit.torch.version, "cuda", None)
    monkeypatch.setattr(audit.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(audit.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(audit.torch.xpu, "is_available", lambda: False)


def _production_runtime() -> dict[str, object]:
    return {
        "cuda_visible_devices": "",
        "python_runtime_sha256": audit._python_runtime_contract()["sha256"],
        "torch_cuda_device_count": 0,
        "torch_cuda_initialized": False,
        "torch_cuda_version": None,
        "torch_hip_version": None,
        "torch_mps_available": False,
        "torch_version": str(torch.__version__),
        "torch_xpu_available": False,
    }


def test_production_cpu_gate_uses_observable_runtime_facts_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(audit.torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ContractError, match="CUDA_VISIBLE_DEVICES empty"):
        audit._require_production_cpu_environment()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ContractError, match="CUDA_VISIBLE_DEVICES empty"):
        audit._require_production_cpu_environment()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(audit.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ContractError, match="device_count 0"):
        audit._require_production_cpu_environment()

    monkeypatch.setattr(audit.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(audit.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ContractError, match="CPU-only Torch build"):
        audit._require_production_cpu_environment()

    monkeypatch.setattr(audit.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(audit.torch.version, "cuda", "12.8")
    with pytest.raises(ContractError, match="CPU-only Torch build"):
        audit._require_production_cpu_environment()

    monkeypatch.setattr(audit.torch.version, "cuda", None)
    assert audit._require_production_cpu_environment() == _production_runtime()


def test_production_runtime_is_bound_to_dependency_python_and_torch() -> None:
    environment = _dependency_environment()
    runtime = _production_runtime()
    processor_contract = {"dependency_environment": environment}
    assert audit._validated_runtime_dependency_pair(
        processor_contract=processor_contract,
        production_runtime=runtime,
    ) == (runtime, environment)

    with pytest.raises(ContractError, match="Python runtime differs"):
        audit._validated_runtime_dependency_pair(
            processor_contract=processor_contract,
            production_runtime={**runtime, "python_runtime_sha256": "0" * 64},
        )

    drifted_environment = deepcopy(environment)
    packages = cast(list[dict[str, object]], drifted_environment["packages"])
    torch_package = next(row for row in packages if row["distribution"] == "torch")
    torch_package["version"] = "2.7.0"
    dependency_content = {
        "packages": packages,
        "python_runtime": drifted_environment["python_runtime"],
    }
    drifted_environment["sha256"] = hashlib.sha256(
        audit._canonical_bytes(dependency_content)
    ).hexdigest()
    with pytest.raises(ContractError, match="Torch import differs"):
        audit._validated_runtime_dependency_pair(
            processor_contract={"dependency_environment": drifted_environment},
            production_runtime=runtime,
        )


def test_pinned_processor_disables_remote_code_and_binds_dependency_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_dir = (tmp_path / "processor").resolve()
    processor_dir.mkdir()
    snapshot_dir = (tmp_path / "private-snapshot").resolve()
    snapshot_dir.mkdir()
    processor = object()
    calls: list[tuple[Path, dict[str, object]]] = []

    def load(path: Path, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return processor

    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = SimpleNamespace(from_pretrained=load)  # type: ignore[attr-defined]
    transformers_module.__version__ = audit.ADR125_TRANSFORMERS_VERSION
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(
        audit,
        "_processor_asset_contract_from_disk",
        _processor_asset_identity,
    )
    monkeypatch.setattr(
        audit,
        "_make_private_processor_snapshot",
        lambda path: (snapshot_dir, _processor_asset_identity(path)),
    )
    removed: list[Path] = []
    monkeypatch.setattr(audit, "_remove_private_processor_snapshot", removed.append)
    monkeypatch.setattr(audit, "_dependency_environment_contract", _dependency_environment)
    with audit._load_pinned_processor(processor_dir) as (loaded, identity):
        assert loaded is processor

    assert calls == [
        (
            snapshot_dir,
            {
                "local_files_only": True,
                "padding_side": "right",
                "revision": QWEN_PROCESSOR_REVISION,
                "trust_remote_code": False,
            },
        )
    ]
    assert identity["dependency_environment"] == _dependency_environment()
    assert identity["processor_dir"] == str(processor_dir)
    assert removed == [snapshot_dir]

    transformers_module.__version__ = "4.57.4"
    with (
        pytest.raises(ContractError, match="transformers=="),
        audit._load_pinned_processor(processor_dir),
    ):
        pass


def test_processor_snapshot_has_complete_inventory_and_immutable_verified_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        "processor_config.json": b'{"processor_class":"Qwen3VLProcessor"}\n',
        "tokenizer/tokenizer.json": b'{"version":"1.0"}\n',
    }
    contract = {
        name: (len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()
    }
    monkeypatch.setattr(audit, "REQUIRED_PROCESSOR_FILES", tuple(payloads))
    monkeypatch.setattr(audit, "PROCESSOR_ASSET_CONTRACT", contract)
    source = tmp_path / "processor-source"
    source.mkdir()
    for name, payload in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    source = source.resolve()

    snapshot, identity = audit._make_private_processor_snapshot(source)
    try:
        assert snapshot != source
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o500
        assert identity["processor_dir"] == str(source)
        assert identity["processor_load_protocol"] == audit.PROCESSOR_SNAPSHOT_PROTOCOL
        assert identity["processor_file_inventory"] == [
            [name, size, digest] for name, (size, digest) in sorted(contract.items())
        ]
        for name, payload in payloads.items():
            copied = snapshot / name
            assert copied.read_bytes() == payload
            assert stat.S_IMODE(copied.stat().st_mode) == 0o400
        snapshot_identity = audit._processor_asset_contract_from_disk(snapshot)
        assert audit._processor_contract_without_path(snapshot_identity) == (
            audit._processor_contract_without_path(identity)
        )
    finally:
        audit._remove_private_processor_snapshot(snapshot)
    assert not snapshot.exists()

    unexpected = source / "unreviewed.json"
    unexpected.write_text("{}\n", encoding="ascii")
    with pytest.raises(ContractError, match="complete file inventory"):
        audit._verified_processor_payloads(source)
    unexpected.unlink()

    expected = source / "processor_config.json"
    external = tmp_path / "external-processor-config.json"
    external.write_bytes(payloads["processor_config.json"])
    expected.unlink()
    expected.symlink_to(external)
    with pytest.raises(ContractError, match="symlink or special file"):
        audit._verified_processor_payloads(source)


def test_snapshot_cleanup_does_not_follow_replaced_directory_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"verified\n"
    contract = {"nested/asset.json": (len(payload), hashlib.sha256(payload).hexdigest())}
    monkeypatch.setattr(audit, "REQUIRED_PROCESSOR_FILES", ("nested/asset.json",))
    monkeypatch.setattr(audit, "PROCESSOR_ASSET_CONTRACT", contract)
    source = tmp_path / "processor-source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/asset.json").write_bytes(payload)
    snapshot, _ = audit._make_private_processor_snapshot(source.resolve())
    external = tmp_path / "external-directory"
    external.mkdir()
    external.chmod(0o500)
    external_mode = stat.S_IMODE(external.stat().st_mode)

    snapshot.chmod(0o700)
    nested = snapshot / "nested"
    nested.chmod(0o700)
    (nested / "asset.json").unlink()
    nested.rmdir()
    nested.symlink_to(external, target_is_directory=True)
    audit._remove_private_processor_snapshot(snapshot)

    assert not snapshot.exists()
    assert external.is_dir()
    assert stat.S_IMODE(external.stat().st_mode) == external_mode


def test_internal_processor_evidence_covers_all_records_and_both_schedules(
    tmp_path: Path,
) -> None:
    report, manifest, processor = _internal_report(tmp_path)
    report_view = cast(dict[str, Any], report)

    assert report["schema"] == audit.INTERNAL_SCHEMA
    assert report["status"] == audit.INTERNAL_STATUS
    assert report["publication_authorized"] is False
    assert "artifact_sha256" not in report
    assert len(manifest.materialized_record_ids) == 192
    assert len(set(manifest.materialized_record_ids)) == 192
    assert processor.call_count == 192
    assert report["summary"] == {
        "family_partition_counts": dict(sorted(audit.EXPECTED_COUNTS.items())),
        "merged_visual_token_maximum": 64,
        "merged_visual_token_minimum": 64,
        "raw_patch_token_maximum": 256,
        "raw_patch_token_minimum": 256,
        "record_count": 192,
        "sequence_length_maximum": 5,
        "sequence_length_minimum": 5,
        "supervised_token_maximum": 3,
        "supervised_token_minimum": 3,
    }

    records = report["records"]
    assert isinstance(records, list)
    first = records[0]
    assert first["sequence_length"] == 5
    assert first["supervised_token_count"] == 3
    assert first["pixel_patch_count"] == 256
    assert first["pixel_values_shape"] == [256, 4]
    assert first["image_grid_thw"] == [[1, 16, 16]]
    assert first["user_text_sha256"] == text_sha256(manifest.records[0].user_text)
    assert first["target_answer_sha256"] == text_sha256(manifest.records[0].assistant_text)
    for field in (
        "assistant_mask_sha256",
        "attention_mask_sha256",
        "image_grid_thw_sha256",
        "input_ids_sha256",
        "labels_sha256",
        "pixel_values_sha256",
        "record_evidence_sha256",
    ):
        assert len(first[field]) == 64
    assert first["position_ids_sha256"] is None
    assert set(first["processor_output_tensors"]) == {
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    }
    assert set(first["semantic_tensors"]) == set(audit.SEMANTIC_TENSOR_NAMES)
    assert first["semantic_tensors"]["position_ids"] == {"present": False}
    assert first["semantic_tensors"]["pixel_values"]["sha256"] == first["pixel_values_sha256"]
    assert len(report_view["record_aggregate_sha256"]) == 64
    assert len(report_view["tensor_aggregates"]["semantic"]["sha256"]) == 64
    assert len(report_view["tensor_aggregates"]["processor_output"]["sha256"]) == 64
    assert report["production_evidence"] == audit._normalized_production_evidence(report)

    schedules = report["schedules"]
    assert isinstance(schedules, dict)
    assert schedules["field_order"] == list(PUBLIC_VL_SCHEDULE_FIELDS)
    heldout = schedules["heldout"]
    training = schedules["training"]
    assert isinstance(heldout, dict)
    assert isinstance(training, dict)
    assert heldout["record_count"] == 64
    assert training["record_count"] == 128
    assert len(heldout["rows"]) == 64
    assert len(training["rows"]) == 128
    assert heldout["sha256"] == public_vl_schedule_sha256(heldout["rows"])
    assert training["sha256"] == public_vl_schedule_sha256(training["rows"])
    record_id_index = PUBLIC_VL_SCHEDULE_FIELDS.index("record_id")
    assert training["rows"][0][record_id_index] == "referring-train-000"
    assert training["rows"][1][record_id_index] == "vqa-train-000"
    assert manifest.training_lookups == [(step, rank) for step in range(64) for rank in range(2)]

    with pytest.raises(ContractError, match="CAPTURE or authorized PASS"):
        audit._publish_report(tmp_path / "forbidden.json", report)


def test_first_real_asset_entry_publishes_capture_but_never_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, manifest_path, processor_dir = _production_paths(tmp_path)
    _allow_cpu_cli(monkeypatch)
    manifest = _Manifest()
    _prepare_artifact_files(root, manifest)

    def load_manifest(**kwargs: object) -> object:
        assert kwargs == {
            "artifact_root": root,
            "manifest_file_sha256": audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
            "manifest_path": manifest_path,
            "max_steps": 64,
        }
        return manifest

    @contextmanager
    def load_processor(path: Path) -> Any:
        assert path == processor_dir
        yield _Processor(), _processor_identity(path)

    monkeypatch.setattr(audit, "PublicNativeVLRetentionManifest", _Manifest)
    monkeypatch.setattr(audit, "load_frozen_public_native_vl_retention_gate", load_manifest)
    monkeypatch.setattr(audit, "_load_pinned_processor", load_processor)
    monkeypatch.setattr(
        audit,
        "_sha256_regular_file",
        lambda path: (
            audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256
            if path == manifest_path
            else pytest.fail("unexpected manifest path")
        ),
    )
    monkeypatch.setattr(audit, "_processor_asset_contract_from_disk", _processor_asset_identity)
    monkeypatch.setattr(audit, "_dependency_environment_contract", _dependency_environment)
    output = tmp_path / "report.json"
    audit.main(
        [
            "--root",
            str(root),
            "--manifest",
            str(manifest_path),
            "--manifest-sha256",
            audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
            "--processor",
            str(processor_dir),
            "--output",
            str(output),
            "--mode",
            "capture",
        ]
    )

    report = json.loads(output.read_text(encoding="ascii"))
    assert report["schema"] == audit.CAPTURE_SCHEMA
    assert report["status"] == audit.CAPTURE_STATUS
    assert report["publication_authorized"] is False
    assert report["public_native_vl_contract"]["artifact_sha256"] == (
        audit.ADR125_PUBLIC_ARTIFACT_SHA256
    )
    assert report["public_native_vl_contract"]["manifest_file_sha256"] == (
        audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256
    )
    assert report["processor_contract"]["processor_assets"] == asset_contract_manifest(
        PROCESSOR_ASSET_CONTRACT
    )
    assert report["processor_contract"]["dependency_environment"] == _dependency_environment()
    assert report["path_contract"] == {
        "artifact_root": str(root),
        "manifest_path": str(manifest_path),
        "processor_dir": str(processor_dir),
    }
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["status"] == audit.CAPTURE_STATUS
    assert stdout["record_count"] == 192
    assert stdout["production_evidence"] == report["production_evidence"]

    with pytest.raises(FileExistsError):
        audit._publish_report(output, report)
    tampered = deepcopy(report)
    tampered["records"][0]["input_ids_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="artifact digest changed"):
        audit._publish_report(tmp_path / "tampered.json", tampered)


def test_production_entry_rejects_wrong_manifest_sha_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, processor_dir = _production_paths(tmp_path)
    _allow_cpu_cli(monkeypatch)
    with pytest.raises(ContractError, match="manifest SHA-256 changed"):
        audit.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                "0" * 64,
                "--processor",
                str(processor_dir),
                "--output",
                str(tmp_path / "wrong-sha.json"),
                "--mode",
                "capture",
            ]
        )

    monkeypatch.setattr(
        audit,
        "load_frozen_public_native_vl_retention_gate",
        lambda **_kwargs: _Manifest(artifact_sha256="0" * 64),
    )
    with pytest.raises(ContractError, match="artifact SHA-256 changed"):
        audit.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
                "--processor",
                str(processor_dir),
                "--output",
                str(tmp_path / "wrong-artifact.json"),
                "--mode",
                "capture",
            ]
        )


def test_production_paths_reject_aliases_wrong_location_and_missing_assets(tmp_path: Path) -> None:
    root, manifest_path, processor_dir = _production_paths(tmp_path)
    wrong_manifest = root / "other.json"
    wrong_manifest.write_text("{}", encoding="ascii")
    with pytest.raises(ContractError, match="frozen artifact manifest"):
        audit._resolve_production_paths(
            artifact_root=root,
            manifest_path=wrong_manifest,
            processor_dir=processor_dir,
        )

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ContractError, match="real directory"):
        audit._resolve_production_paths(
            artifact_root=linked_root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
        )

    (processor_dir / REQUIRED_PROCESSOR_FILES[0]).unlink()
    with pytest.raises(ContractError, match="processor asset"):
        audit._resolve_production_paths(
            artifact_root=root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
        )


def test_artifact_snapshot_rejects_symlink_even_when_bytes_match(tmp_path: Path) -> None:
    manifest = _Manifest()
    root = tmp_path.resolve()
    _prepare_artifact_files(root, manifest)
    first = root / manifest.records[0].image_file
    external = root / "external-copy.png"
    external.write_bytes(first.read_bytes())
    first.unlink()
    first.symlink_to(external)

    with pytest.raises(ContractError, match="symbolic-link component"):
        audit._artifact_image_snapshot(manifest, root)  # type: ignore[arg-type]


def test_authorization_rechecks_manifest_and_processor_after_all_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, processor_dir = _production_paths(tmp_path)
    manifest = _Manifest()
    _prepare_artifact_files(root, manifest)
    internal = audit.build_public_native_vl_processor_budget_audit(
        manifest=manifest,  # type: ignore[arg-type]
        artifact_root=root,
        manifest_file_sha256=audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
        processor=_Processor(),
        processor_identity=_processor_identity(processor_dir),
        processor_dir=processor_dir,
    )
    monkeypatch.setattr(audit, "PublicNativeVLRetentionManifest", _Manifest)
    tool_sha256 = hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    with pytest.raises(ContractError, match="manifest changed"):
        audit._validate_adr125_internal_report(
            internal=internal,
            manifest=manifest,  # type: ignore[arg-type]
            artifact_root=root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
            tool_sha256=tool_sha256,
        )

    monkeypatch.setattr(
        audit,
        "_sha256_regular_file",
        lambda _path: audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
    )
    monkeypatch.setattr(
        audit,
        "_processor_asset_contract_from_disk",
        lambda _path: {**_processor_asset_identity(processor_dir), "processor_revision": "0" * 40},
    )
    with pytest.raises(ContractError, match="processor assets changed"):
        audit._validate_adr125_internal_report(
            internal=internal,
            manifest=manifest,  # type: ignore[arg-type]
            artifact_root=root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
            tool_sha256=tool_sha256,
        )


def test_processor_evidence_fails_closed_on_cardinality_grid_patch_and_non_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _Manifest()
    manifest.records = manifest.records[:-1]
    with pytest.raises(ContractError, match="exactly 192"):
        _internal_report(tmp_path, manifest=manifest)

    with pytest.raises(RuntimeError, match="smart-resize geometry"):
        _internal_report(tmp_path, processor=_Processor(grid=(1, 14, 16)))

    with pytest.raises(ContractError, match="pixel patch count"):
        _internal_report(tmp_path, processor=_Processor(pixel_patch_count=255))

    with monkeypatch.context() as cuda_patch:
        cuda_patch.setattr(audit.torch.cuda, "is_initialized", lambda: True)
        report, _, _ = _internal_report(tmp_path / "cuda-already-initialized")
        assert report["status"] == audit.INTERNAL_STATUS

    original = audit.build_native_vl_grounding_batch

    def non_cpu_batch(record: Any, processor: Any) -> object:
        batch = original(record, processor)
        return SimpleNamespace(
            input_ids=torch.empty(batch.input_ids.shape, dtype=torch.long, device="meta"),
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            assistant_token_mask=batch.assistant_token_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            position_ids=batch.position_ids,
        )

    monkeypatch.setattr(audit, "build_native_vl_grounding_batch", non_cpu_batch)
    with pytest.raises(ContractError, match="input_ids must remain on CPU"):
        _internal_report(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("processor_revision", "0" * 40),
        ("processor_assets", []),
        ("processor_id", "different/model"),
    ],
)
def test_processor_identity_mutations_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    identity = _processor_identity(tmp_path)
    identity[field] = value
    with pytest.raises(ContractError, match="pinned contract"):
        _internal_report(tmp_path, processor_identity=identity)


def test_duplicate_record_and_tensor_digest_mutations_are_observable(tmp_path: Path) -> None:
    manifest = _Manifest()
    manifest.records[96].image_rgb_sha256 = manifest.records[0].image_rgb_sha256
    with pytest.raises(ContractError, match="duplicate"):
        _internal_report(tmp_path, manifest=manifest)

    control, _, _ = _internal_report(tmp_path / "control")
    candidate, _, _ = _internal_report(
        tmp_path / "candidate",
        processor=_Processor(answer_token=124),
    )
    control_records = control["records"]
    candidate_records = candidate["records"]
    assert isinstance(control_records, list)
    assert isinstance(candidate_records, list)
    control_first = control_records[0]
    candidate_first = candidate_records[0]
    assert isinstance(control_first, dict)
    assert isinstance(candidate_first, dict)
    assert control_first["input_ids_sha256"] != candidate_first["input_ids_sha256"]
    assert control_first["labels_sha256"] != candidate_first["labels_sha256"]
    assert control_first["record_evidence_sha256"] != candidate_first["record_evidence_sha256"]


def test_pixel_position_and_dependency_mutations_change_production_evidence(tmp_path: Path) -> None:
    control, _, _ = _internal_report(tmp_path / "control")
    pixels, _, _ = _internal_report(
        tmp_path / "pixels",
        processor=_Processor(pixel_value=1.25),
    )
    positions, _, _ = _internal_report(
        tmp_path / "positions",
        processor=_Processor(include_position_ids=True),
    )
    dependency_identity = _processor_identity(tmp_path / "dependency")
    dependency_identity["dependency_environment"] = _dependency_environment(mutation="drift")
    dependency, _, _ = _internal_report(
        tmp_path / "dependency",
        processor_identity=dependency_identity,
    )
    control_view = cast(dict[str, Any], control)
    pixels_view = cast(dict[str, Any], pixels)
    positions_view = cast(dict[str, Any], positions)

    assert (
        control_view["records"][0]["pixel_values_sha256"]
        != pixels_view["records"][0]["pixel_values_sha256"]
    )
    assert (
        control_view["tensor_aggregates"]["semantic"]["sha256"]
        != pixels_view["tensor_aggregates"]["semantic"]["sha256"]
    )
    assert positions_view["records"][0]["position_ids_sha256"] is not None
    assert positions_view["records"][0]["semantic_tensors"]["position_ids"]["present"] is True
    assert (
        control_view["processor_contract"]["dependency_environment"]["sha256"]
        != cast(dict[str, Any], dependency)["processor_contract"]["dependency_environment"][
            "sha256"
        ]
    )


def test_unknown_processor_tensor_and_dependency_api_drift_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="output key set changed"):
        _internal_report(tmp_path / "unknown", processor=_Processor(extra_tensor_name="new_ids"))

    class _NestedProcessor(_Processor):
        def apply_chat_template(self, *args: object, **kwargs: object) -> dict[str, object]:
            result: dict[str, object] = super().apply_chat_template(*args, **kwargs)
            result["metadata"] = {"nested": torch.ones(1)}
            return result

    with pytest.raises(ContractError, match="output key set changed"):
        _internal_report(tmp_path / "nested", processor=_NestedProcessor())

    class _NonTensorRequiredOutput(_Processor):
        def apply_chat_template(self, *args: object, **kwargs: object) -> dict[str, object]:
            result: dict[str, object] = super().apply_chat_template(*args, **kwargs)
            result["pixel_values"] = {"nested": torch.ones(1)}
            return result

    with pytest.raises(ContractError, match="pixel_values must remain on CPU"):
        _internal_report(tmp_path / "required-nested", processor=_NonTensorRequiredOutput())

    class _DriftingMapping(dict[str, object]):
        def items(self) -> Any:
            yield from super().items()
            yield "metadata", torch.ones(1)

    class _DriftingProcessor(_Processor):
        def apply_chat_template(self, *args: object, **kwargs: object) -> dict[str, object]:
            return _DriftingMapping(super().apply_chat_template(*args, **kwargs))

    with pytest.raises(ContractError, match="mapping changed during inspection"):
        _internal_report(tmp_path / "mapping-drift", processor=_DriftingProcessor())

    malformed = _dependency_environment()
    packages = malformed["packages"]
    assert isinstance(packages, list)
    packages[0]["new_runtime_field"] = "drift"
    content = {"packages": packages, "python_runtime": malformed["python_runtime"]}
    malformed["sha256"] = hashlib.sha256(audit._canonical_bytes(content)).hexdigest()
    identity = _processor_identity(tmp_path / "dependency-api")
    identity["dependency_environment"] = malformed
    with pytest.raises(ContractError, match="package contract fields changed"):
        _internal_report(tmp_path / "dependency-api", processor_identity=identity)


def test_distribution_evidence_hashes_actual_code_and_record_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "venv"
    site_packages = environment / "lib/python3.12/site-packages"
    import_name = "processor_evidence_example"
    code_relative = Path(f"{import_name}/__init__.py")
    bytecode_relative = Path(
        f"{import_name}/__pycache__/__init__.{sys.implementation.cache_tag}.pyc"
    )
    record_relative = Path("example-1.0.dist-info/RECORD")
    script_relative = Path("../../../bin/example")
    code = site_packages / code_relative
    bytecode = site_packages / bytecode_relative
    script = environment / "bin/example"
    record = site_packages / record_relative
    for path, content in (
        (code, b"VALUE = 1\n"),
        (bytecode, b"generated-bytecode-v1"),
        (script, b"#!/bin/sh\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def record_hash(payload: bytes) -> str:
        encoded = audit.base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode()
        return encoded.rstrip("=")

    def write_record() -> None:
        code_payload = code.read_bytes()
        script_payload = script.read_bytes()
        payload = (
            f"{code_relative.as_posix()},sha256={record_hash(code_payload)},{len(code_payload)}\n"
            f"{bytecode_relative.as_posix()},,\n"
            f"{script_relative.as_posix()},sha256={record_hash(script_payload)},{len(script_payload)}\n"
            f"{record_relative.as_posix()},,\n"
        ).encode()
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_bytes(payload)

    write_record()
    distribution = SimpleNamespace(
        files=[record_relative],
        locate_file=lambda relative: site_packages / relative,
        version="1.0",
    )
    monkeypatch.setattr(audit.importlib.metadata, "distribution", lambda _name: distribution)
    monkeypatch.syspath_prepend(str(site_packages))
    sys.modules.pop(import_name, None)
    try:
        control = audit._distribution_evidence("example", import_name=import_name)
        assert control["installed_file_count"] == 4
        assert control["generated_script_count"] == 1
        assert control["import_name"] == import_name
        assert control["import_file"] == code_relative.as_posix()
        assert control["import_file_sha256"] == hashlib.sha256(code.read_bytes()).hexdigest()
        assert control["record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
        bytecode.write_bytes(b"generated-bytecode-v2")
        changed_bytecode = audit._distribution_evidence("example", import_name=import_name)
        assert control["record_sha256"] == changed_bytecode["record_sha256"]
        assert control["installed_files_sha256"] != changed_bytecode["installed_files_sha256"]
        code.write_bytes(b"VALUE = 2\n")
        with pytest.raises(ContractError, match="differ from RECORD"):
            audit._distribution_evidence("example", import_name=import_name)
        write_record()
        mutated = audit._distribution_evidence("example", import_name=import_name)
        assert control["installed_files_sha256"] != mutated["installed_files_sha256"]
        assert control["record_sha256"] != mutated["record_sha256"]

        imported_module = sys.modules[import_name]

        def mutate_entry_during_import(_name: str) -> object:
            code.write_bytes(b"VALUE = 3\n")
            return imported_module

        monkeypatch.setattr(audit.importlib, "import_module", mutate_entry_during_import)
        with pytest.raises(ContractError, match="entry file changed during import"):
            audit._distribution_evidence("example", import_name=import_name)
        code.write_bytes(b"VALUE = 2\n")

        shadow = tmp_path / "shadow" / import_name / "__init__.py"
        shadow.parent.mkdir(parents=True)
        shadow.write_bytes(code.read_bytes())
        shadow_module = ModuleType(import_name)
        shadow_module.__file__ = str(shadow)
        shadow_module.__spec__ = SimpleNamespace(origin=str(shadow))  # type: ignore[assignment]
        monkeypatch.setattr(audit.importlib, "import_module", lambda _name: shadow_module)
        with pytest.raises(ContractError, match="outside its recorded distribution"):
            audit._distribution_evidence("example", import_name=import_name)
    finally:
        sys.modules.pop(import_name, None)


def test_distribution_evidence_rejects_unhashed_non_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    import_name = "unhashed_source_example"
    code_relative = Path(f"{import_name}/__init__.py")
    record_relative = Path("example-1.0.dist-info/RECORD")
    code = site_packages / code_relative
    record = site_packages / record_relative
    code.parent.mkdir(parents=True)
    code.write_bytes(b"VALUE = 1\n")
    record.parent.mkdir(parents=True)
    record.write_text(
        f"{code_relative.as_posix()},,\n{record_relative.as_posix()},,\n",
        encoding="utf-8",
    )
    distribution = SimpleNamespace(
        files=[record_relative],
        locate_file=lambda relative: site_packages / relative,
        version="1.0",
    )
    monkeypatch.setattr(audit.importlib.metadata, "distribution", lambda _name: distribution)
    monkeypatch.syspath_prepend(str(site_packages))
    with pytest.raises(ContractError, match="only generated bytecode"):
        audit._distribution_evidence("example", import_name=import_name)


def test_real_installed_torch_280_record_and_generated_scripts_are_hashable() -> None:
    evidence = audit._distribution_evidence("torch", import_name="torch")
    assert str(evidence["version"]).split("+", maxsplit=1)[0] == "2.8.0"
    assert isinstance(evidence["installed_file_count"], int)
    assert evidence["installed_file_count"] > 0
    assert isinstance(evidence["generated_script_count"], int)
    assert evidence["generated_script_count"] >= 2
    assert len(str(evidence["record_sha256"])) == 64
    assert len(str(evidence["installed_files_sha256"])) == 64
    assert evidence["import_name"] == "torch"
    assert len(str(evidence["import_file_sha256"])) == 64


def _approval_for_capture(
    capture: dict[str, Any],
    capture_file_sha256: str,
    rationale_identity: dict[str, object],
) -> dict[str, object]:
    processor_contract = capture["processor_contract"]
    return {
        "authenticity": {"status": audit.APPROVAL_AUTHENTICITY_STATUS},
        "capture_report": {
            "artifact_sha256": capture["artifact_sha256"],
            "file_sha256": capture_file_sha256,
        },
        "decision": audit.APPROVAL_DECISION,
        "dependency_environment": processor_contract["dependency_environment"],
        "production_evidence": capture["production_evidence"],
        "production_evidence_sha256": capture["production_evidence_sha256"],
        "review": {
            "rationale_file_sha256": rationale_identity["file_sha256"],
            "rationale_size_bytes": rationale_identity["size_bytes"],
            "reviewed_at_utc": "2026-08-02T00:00:00Z",
            "reviewer": "test-reviewer",
        },
        "schema": audit.APPROVAL_SCHEMA,
        "source_contract": capture["source_contract"],
    }


def test_capture_requires_separate_immutable_approval_and_full_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, processor_dir = _production_paths(tmp_path)
    manifest = _Manifest()
    _prepare_artifact_files(root, manifest)
    internal = audit.build_public_native_vl_processor_budget_audit(
        manifest=manifest,  # type: ignore[arg-type]
        artifact_root=root,
        manifest_file_sha256=audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
        processor=_Processor(),
        processor_identity=_processor_identity(processor_dir),
        processor_dir=processor_dir,
    )
    monkeypatch.setattr(audit, "PublicNativeVLRetentionManifest", _Manifest)
    monkeypatch.setattr(
        audit,
        "_sha256_regular_file",
        lambda _path: audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
    )
    monkeypatch.setattr(audit, "_processor_asset_contract_from_disk", _processor_asset_identity)
    monkeypatch.setattr(audit, "_dependency_environment_contract", _dependency_environment)
    tool_sha256 = hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    runtime = _production_runtime()
    monkeypatch.setattr(audit, "_require_production_cpu_environment", lambda: runtime)

    first_image = root / manifest.records[0].image_file
    original_image = first_image.read_bytes()
    first_image.write_bytes(b"mutated-after-processing")
    with pytest.raises(ContractError, match="artifact image changed"):
        audit._capture_adr125_report(
            internal=internal,
            manifest=manifest,  # type: ignore[arg-type]
            artifact_root=root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
            production_runtime=runtime,
            tool_sha256=tool_sha256,
        )
    first_image.write_bytes(original_image)

    capture = audit._capture_adr125_report(
        internal=internal,
        manifest=manifest,  # type: ignore[arg-type]
        artifact_root=root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        production_runtime=runtime,
        tool_sha256=tool_sha256,
    )
    assert capture["status"] == audit.CAPTURE_STATUS
    assert capture["publication_authorized"] is False
    forged_pass = deepcopy(capture)
    forged_pass.update(
        {
            "publication_authorized": True,
            "schema": audit.SCHEMA,
            "status": "PASS",
        }
    )
    forged_content = dict(forged_pass)
    forged_content.pop("artifact_sha256", None)
    forged_pass["artifact_sha256"] = hashlib.sha256(
        audit._canonical_bytes(forged_content)
    ).hexdigest()
    with pytest.raises(ContractError, match="top-level fields changed"):
        audit._publish_report(tmp_path / "forged-pass.json", forged_pass)

    capture_with_unknown_field = deepcopy(capture)
    capture_with_unknown_field["new_api_field"] = "drift"
    unknown_content = dict(capture_with_unknown_field)
    unknown_content.pop("artifact_sha256", None)
    capture_with_unknown_field["artifact_sha256"] = hashlib.sha256(
        audit._canonical_bytes(unknown_content)
    ).hexdigest()
    with pytest.raises(ContractError, match="top-level fields changed"):
        audit._publish_report(tmp_path / "capture-api-drift.json", capture_with_unknown_field)

    monkeypatch.setattr(
        audit,
        "_dependency_environment_contract",
        lambda: _dependency_environment(mutation="during-audit"),
    )
    with pytest.raises(ContractError, match="dependency environment changed"):
        audit._capture_adr125_report(
            internal=internal,
            manifest=manifest,  # type: ignore[arg-type]
            artifact_root=root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
            production_runtime=runtime,
            tool_sha256=tool_sha256,
        )
    monkeypatch.setattr(audit, "_dependency_environment_contract", _dependency_environment)
    capture_path = tmp_path / "capture.json"
    first_image.write_bytes(b"mutated-after-finalization")
    with pytest.raises(ContractError, match="image changed before report write"):
        audit._publish_report(tmp_path / "image-mutated.json", capture)
    first_image.write_bytes(original_image)
    capture_publish = audit._publish_report(capture_path, capture)
    monkeypatch.setattr(
        audit,
        "_require_production_cpu_environment",
        lambda: {**runtime, "python_runtime_sha256": "0" * 64},
    )
    with pytest.raises(ContractError, match="CPU runtime changed"):
        audit._publish_report(tmp_path / "runtime-mutated.json", capture)
    monkeypatch.setattr(audit, "_require_production_cpu_environment", lambda: runtime)
    capture_view = cast(dict[str, Any], capture)
    rationale_path = (tmp_path / "approval-rationale.txt").resolve()
    rationale_path.write_text(
        "Independent reviewer inspected the complete processor capture and dependency evidence.\n",
        encoding="utf-8",
    )
    rationale_identity = audit._read_rationale_identity(rationale_path)
    approval = _approval_for_capture(
        capture_view,
        str(capture_publish["file_sha256"]),
        rationale_identity,
    )
    approval_path = tmp_path / "approval.json"
    approval_path.write_bytes(audit._canonical_bytes(approval) + b"\n")

    altered_approval = deepcopy(approval)
    altered_production = cast(dict[str, Any], altered_approval["production_evidence"])
    altered_summary = cast(dict[str, Any], altered_production["summary"])
    altered_summary["record_count"] = 191
    altered_path = tmp_path / "altered-approval.json"
    altered_path.write_bytes(audit._canonical_bytes(altered_approval) + b"\n")
    with pytest.raises(ContractError, match="complete capture evidence"):
        audit._load_authorization_inputs(
            capture_report_path=capture_path,
            approval_contract_path=altered_path,
            rationale_file_path=rationale_path,
        )

    authorization_inputs = audit._load_authorization_inputs(
        capture_report_path=capture_path,
        approval_contract_path=approval_path,
        rationale_file_path=rationale_path,
    )
    assert authorization_inputs["rationale_identity"] == rationale_identity
    with pytest.raises(ContractError, match="no pinned approval signature trust root"):
        audit._authorize_adr125_report(
            internal=internal,
            manifest=manifest,  # type: ignore[arg-type]
            artifact_root=root,
            manifest_path=manifest_path,
            processor_dir=processor_dir,
            production_runtime=runtime,
            tool_sha256=tool_sha256,
            authorization_inputs=authorization_inputs,
        )

    monkeypatch.setattr(audit, "_require_production_cpu_environment", lambda: runtime)
    monkeypatch.setattr(
        audit,
        "_load_pinned_processor",
        lambda _path: pytest.fail("AUTHORIZE reached processor replay without a trust root"),
    )
    with pytest.raises(ContractError, match="no pinned approval signature trust root"):
        audit.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                audit.ADR125_PUBLIC_MANIFEST_FILE_SHA256,
                "--processor",
                str(processor_dir),
                "--output",
                str(tmp_path / "authorize-must-not-exist.json"),
                "--mode",
                "authorize",
                "--capture-report",
                str(capture_path),
                "--approval-contract",
                str(approval_path),
                "--rationale-file",
                str(rationale_path),
            ]
        )

    internal_content = dict(internal)
    evidence_sha256 = str(internal_content.pop("evidence_sha256"))
    forged_authorized = audit._final_report(
        internal_content=internal_content,
        evidence_sha256=evidence_sha256,
        artifact_root=root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        production_runtime=runtime,
        tool_sha256=tool_sha256,
        authorized=True,
        authorization_inputs=authorization_inputs,
    )
    with pytest.raises(ContractError, match="no pinned approval signature trust root"):
        audit._publish_report(tmp_path / "forged-authorized.json", forged_authorized)

    rationale_path.write_text("Rationale changed after approval.\n", encoding="utf-8")
    with pytest.raises(ContractError, match="exact rationale file"):
        audit._revalidate_authorization_inputs(authorization_inputs)
    rationale_path.write_text(
        "Independent reviewer inspected the complete processor capture and dependency evidence.\n",
        encoding="utf-8",
    )

    approval["decision"] = "REJECT"
    approval_path.write_bytes(audit._canonical_bytes(approval) + b"\n")
    with pytest.raises(ContractError, match="not an explicit approval"):
        audit._revalidate_authorization_inputs(authorization_inputs)


def test_approval_schema_file_is_bound_by_source_contract() -> None:
    schema_path = Path(audit._REPOSITORY_ROOT) / audit.APPROVAL_SCHEMA_RELATIVE_PATH
    schema = json.loads(schema_path.read_text(encoding="ascii"))
    assert schema["$id"] == audit.APPROVAL_SCHEMA
    assert schema["properties"]["authenticity"]["properties"]["status"] == {
        "const": audit.APPROVAL_AUTHENTICITY_STATUS
    }
    assert set(schema["properties"]["review"]["required"]) == {
        "rationale_file_sha256",
        "rationale_size_bytes",
        "reviewed_at_utc",
        "reviewer",
    }
    source = audit._repository_source_contract()
    rows = source["files"]
    assert isinstance(rows, list)
    assert any(row[0] == audit.APPROVAL_SCHEMA_RELATIVE_PATH.as_posix() for row in rows)

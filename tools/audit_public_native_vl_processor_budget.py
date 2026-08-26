#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Audit every frozen public native-VL image through the pinned Qwen processor."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="public native VL processor-budget audit",
)

import torch

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_FAMILIES,
    PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
    PUBLIC_NATIVE_VL_MANIFEST_SCHEMA,
    PUBLIC_NATIVE_VL_PARTITIONS,
    PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY,
    NativeVLInstructionRecord,
    PublicNativeVLManifestRecord,
    PublicNativeVLRetentionManifest,
    load_frozen_public_native_vl_retention_gate,
    native_vl_rgb_sha256,
)
from picf_next.lingbot_native.lattice_feasibility import (
    configure_native_processor_area_budget,
    validate_native_processor_record_grid,
)
from picf_next.lingbot_native.public_vl_evidence import (
    PUBLIC_VL_SCHEDULE_FIELDS,
    cpu_tensor_evidence,
    cpu_tensor_sha256,
    public_vl_schedule_row,
    public_vl_schedule_sha256,
    require_cpu_tensor,
    tensor_evidence_aggregate_sha256,
    text_sha256,
)
from picf_next.lingbot_native.vl_cotraining import (
    NativeVLGroundingBatch,
    build_native_vl_grounding_batch,
)
from tools.bootstrap_lingbot_vla2 import (
    PROCESSOR_ASSET_CONTRACT,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
    REQUIRED_PROCESSOR_FILES,
    asset_contract_manifest,
)

SCHEMA = "picf-next.public-native-vl-processor-budget-audit.v3"
CAPTURE_SCHEMA = "picf-next.public-native-vl-processor-budget-capture.v1"
INTERNAL_SCHEMA = "picf-next.public-native-vl-processor-budget-internal-evidence.v2"
APPROVAL_SCHEMA = "picf-next.public-native-vl-processor-approval.v1"
INTERNAL_STATUS = "INTERNAL_EVIDENCE_ONLY"
CAPTURE_STATUS = "CAPTURE_EVIDENCE_ONLY"
APPROVAL_DECISION = "APPROVE"
APPROVAL_AUTHENTICITY_STATUS = "UNCONFIGURED_FAIL_CLOSED"
APPROVAL_SCHEMA_RELATIVE_PATH = Path(
    "configs/evidence/adr125_public_native_vl_processor_approval.schema.json"
)
ADR125_PUBLIC_MANIFEST_FILE = "public_native_vl_retention_manifest.json"
ADR125_PUBLIC_MANIFEST_FILE_SHA256 = (
    "e6ad12f1d6df8fc53e3661d9d999d5a65b2069436822c6cfc0553f63e5323252"
)
ADR125_PUBLIC_ARTIFACT_SHA256 = "3c247033fde2815c3d0b350a264fa940d541529cfa9bacf34bb8737730499480"
ADR125_TRANSFORMERS_VERSION = "4.57.3"
AUDITED_LATTICE = 8
PROCESSOR_SNAPSHOT_PROTOCOL = "verified-private-read-only-snapshot.v1"
MAXIMUM_RATIONALE_BYTES = 1024 * 1024
DEPENDENCY_IMPORTS = (
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("tokenizers", "tokenizers"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("huggingface-hub", "huggingface_hub"),
    ("Jinja2", "jinja2"),
    ("regex", "regex"),
    ("packaging", "packaging"),
    ("requests", "requests"),
    ("PyYAML", "yaml"),
    ("safetensors", "safetensors"),
    ("tqdm", "tqdm"),
    ("filelock", "filelock"),
    ("fsspec", "fsspec"),
    ("typing-extensions", "typing_extensions"),
    ("setuptools", "setuptools"),
    ("sympy", "sympy"),
    ("mpmath", "mpmath"),
    ("networkx", "networkx"),
    ("MarkupSafe", "markupsafe"),
    ("charset-normalizer", "charset_normalizer"),
    ("idna", "idna"),
    ("urllib3", "urllib3"),
    ("certifi", "certifi"),
)
DEPENDENCY_DISTRIBUTIONS = tuple(item[0] for item in DEPENDENCY_IMPORTS)
PROCESSOR_OUTPUT_TENSOR_NAMES = (
    "attention_mask",
    "image_grid_thw",
    "input_ids",
    "pixel_values",
    "position_ids",
)
SEMANTIC_TENSOR_NAMES = (
    "attention_mask",
    "image_grid_thw",
    "input_ids",
    "label_mask",
    "labels",
    "pixel_values",
    "position_ids",
)
EXPECTED_COUNTS = {
    f"{family}/{partition}": (
        PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY
        if partition == "train"
        else PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
    )
    for family in PUBLIC_NATIVE_VL_FAMILIES
    for partition in PUBLIC_NATIVE_VL_PARTITIONS
}
EXPECTED_RECORD_COUNT = sum(EXPECTED_COUNTS.values())
CAPTURE_REPORT_FIELDS = frozenset(
    {
        "artifact_image_snapshot",
        "artifact_sha256",
        "device_contract",
        "internal_evidence_sha256",
        "path_contract",
        "processor_contract",
        "production_evidence",
        "production_evidence_sha256",
        "production_runtime",
        "public_native_vl_contract",
        "publication_authorized",
        "record_aggregate_sha256",
        "records",
        "records_sha256",
        "schedules",
        "schema",
        "source_contract",
        "status",
        "summary",
        "tensor_aggregates",
        "tool_sha256",
    }
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("processor-budget audit is not canonical JSON") from error


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256")
    return value


def _require_unique(values: Sequence[object], *, name: str) -> None:
    try:
        unique_count = len(set(values))
    except TypeError as error:
        raise ContractError(f"public native VL {name} identity is not hashable") from error
    if unique_count != len(values):
        raise ContractError(f"public native VL {name} contains duplicate records")


def _open_directory_descriptor(path: Path, *, description: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ContractError("processor evidence requires O_NOFOLLOW and O_DIRECTORY support")
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ContractError(f"{description} must be one canonical absolute directory")
    flags = os.O_RDONLY | nofollow | directory
    flags |= getattr(os, "O_CLOEXEC", 0)
    current: int | None = None
    try:
        current = os.open(path.anchor, flags)
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        metadata = os.fstat(current)
    except OSError as error:
        if current is not None:
            os.close(current)
        raise ContractError(f"{description} cannot be opened without following symlinks") from error
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(current)
        raise ContractError(f"{description} must be a directory")
    return current


def _open_regular_descriptor(
    path: Path,
    *,
    description: str,
    ownership_root: Path | None = None,
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ContractError("processor evidence requires O_NOFOLLOW and O_DIRECTORY support")
    root = path.parent if ownership_root is None else ownership_root
    if (
        not path.is_absolute()
        or not root.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or any(part in {".", ".."} for part in root.parts)
        or not _is_beneath(path, root)
    ):
        raise ContractError(f"{description} escaped its canonical ownership root")
    relative = path.relative_to(root)
    if not relative.parts:
        raise ContractError(f"{description} must name a file below its ownership root")

    directory_flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    current = _open_directory_descriptor(root, description=f"{description} ownership root")
    descriptor: int | None = None
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(relative.parts[-1], file_flags, dir_fd=current)
        metadata = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ContractError(f"{description} cannot be opened without following symlinks") from error
    finally:
        os.close(current)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ContractError(f"{description} must be a regular file")
    return descriptor


def _sha256_file_content(
    path: Path,
    *,
    description: str,
    ownership_root: Path | None = None,
) -> tuple[int, str]:
    descriptor = _open_regular_descriptor(
        path,
        description=description,
        ownership_root=ownership_root,
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ContractError(f"{description} cannot be hashed") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ContractError(f"{description} changed while being hashed")
    return after.st_size, digest.hexdigest()


def _read_regular_file(
    path: Path,
    *,
    description: str,
    maximum_bytes: int,
    ownership_root: Path | None = None,
) -> tuple[bytes, str]:
    descriptor = _open_regular_descriptor(
        path,
        description=description,
        ownership_root=ownership_root,
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            before = os.fstat(stream.fileno())
            if before.st_size > maximum_bytes:
                raise ContractError(f"{description} exceeds its byte limit")
            payload = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ContractError(f"{description} cannot be read") from error
    if len(payload) > maximum_bytes:
        raise ContractError(f"{description} exceeds its byte limit")
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise ContractError(f"{description} changed while being read")
    return payload, hashlib.sha256(payload).hexdigest()


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_no_symlink_components(path: Path, *, root: Path, description: str) -> None:
    if not _is_beneath(path, root):
        raise ContractError(f"{description} escaped its ownership root")
    current = root
    if current.is_symlink():
        raise ContractError(f"{description} ownership root must not be a symlink")
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{description} contains a symbolic-link component")


def _decode_record_sha256(value: str, *, description: str) -> str:
    if not value.startswith("sha256=") or value.count("=") != 1:
        raise ContractError(f"{description} RECORD hash is not SHA-256")
    encoded = value.removeprefix("sha256=")
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, binascii.Error) as error:
        raise ContractError(f"{description} RECORD hash is malformed") from error
    if len(raw) != hashlib.sha256().digest_size:
        raise ContractError(f"{description} RECORD hash length changed")
    return raw.hex()


def _unhashed_bytecode_source(
    relative_text: str,
    *,
    distribution_root: Path,
    record_rows_by_path: Mapping[str, tuple[str, str]],
) -> str:
    """Validate the narrow PEP 376 exception for generated ``.pyc`` files."""

    relative = PurePosixPath(relative_text)
    if relative.suffix != ".pyc" or ".." in relative.parts:
        raise ContractError("only generated bytecode may omit its RECORD hash and size")
    bytecode_path = distribution_root.joinpath(*relative.parts)
    try:
        source_path = Path(importlib.util.source_from_cache(str(bytecode_path)))
    except (NotImplementedError, ValueError) as error:
        raise ContractError("unhashed RECORD bytecode path is not PEP 3147 canonical") from error
    if not _is_beneath(source_path, distribution_root):
        raise ContractError("unhashed RECORD bytecode source escaped its distribution root")
    source_relative = source_path.relative_to(distribution_root).as_posix()
    source_record = record_rows_by_path.get(source_relative)
    if source_record is None or not all(source_record):
        raise ContractError("unhashed RECORD bytecode lacks one hash-verified source file")
    return source_relative


def _distribution_record_path(distribution: importlib.metadata.Distribution) -> Path:
    files = distribution.files
    if not files:
        raise ContractError("dependency exposes no installed file manifest")
    candidates = [item for item in files if str(item).endswith(".dist-info/RECORD")]
    if len(candidates) != 1 or ".." in PurePosixPath(str(candidates[0])).parts:
        raise ContractError("dependency exposes no unique owned RECORD file")
    return Path(str(distribution.locate_file(candidates[0])))


def _resolve_record_owned_path(
    distribution: importlib.metadata.Distribution,
    record_path: str,
    *,
    distribution_root: Path,
) -> tuple[Path, str, Path]:
    if not record_path or "\\" in record_path or "\0" in record_path:
        raise ContractError("dependency RECORD path is not canonical POSIX text")
    relative = PurePosixPath(record_path)
    if relative.is_absolute() or relative.as_posix() != record_path or "." in relative.parts:
        raise ContractError("dependency RECORD path is not canonical and relative")
    candidate = Path(os.path.abspath(str(distribution.locate_file(record_path))))
    if ".." not in relative.parts:
        if not _is_beneath(candidate, distribution_root):
            raise ContractError("dependency package file escaped its distribution root")
        _require_no_symlink_components(
            candidate,
            root=distribution_root,
            description="dependency package file",
        )
        return candidate, "distribution_file", distribution_root

    parts = relative.parts
    parent_count = 0
    for part in parts:
        if part != "..":
            break
        parent_count += 1
    suffix = parts[parent_count:]
    installation_root = distribution_root
    for _ in range(parent_count):
        parent = installation_root.parent
        if parent == installation_root:
            raise ContractError("dependency traversal escaped its installation prefix")
        installation_root = parent
    scripts_root = installation_root / "bin"
    if (
        parent_count == 0
        or installation_root == Path(installation_root.anchor)
        or len(suffix) != 2
        or suffix[0] != "bin"
        or suffix[1] in {"", ".", ".."}
        or candidate.parent != scripts_root
    ):
        raise ContractError("dependency traversal is not an owned generated script")
    _require_no_symlink_components(
        candidate,
        root=installation_root,
        description="dependency generated script",
    )
    return candidate, "generated_script", installation_root


def _active_import_roots() -> tuple[Path, ...]:
    roots: set[Path] = set()
    for value in sys.path:
        if not value:
            continue
        try:
            candidate = Path(value).resolve(strict=True)
        except OSError:
            continue
        if candidate.is_dir():
            roots.add(candidate)
    if not roots:
        raise ContractError("active Python environment exposes no import roots")
    return tuple(sorted(roots))


def _distribution_evidence(
    distribution_name: str,
    *,
    import_name: str | None = None,
) -> dict[str, object]:
    """Bind one installed distribution's version, RECORD, and actual file bytes."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError(f"required dependency is not installed: {distribution_name}") from error
    distribution_root = Path(str(distribution.locate_file(""))).resolve(strict=True)
    if not any(
        distribution_root == root or _is_beneath(distribution_root, root)
        for root in _active_import_roots()
    ):
        raise ContractError(f"dependency is outside the active environment: {distribution_name}")
    record_path = Path(os.path.abspath(_distribution_record_path(distribution)))
    _require_no_symlink_components(
        record_path,
        root=distribution_root,
        description=f"{distribution_name} RECORD",
    )
    record_payload, record_digest = _read_regular_file(
        record_path,
        description=f"{distribution_name} RECORD",
        maximum_bytes=64 * 1024 * 1024,
        ownership_root=distribution_root,
    )
    try:
        text = record_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"dependency RECORD is not UTF-8: {distribution_name}") from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        record_rows = list(reader)
    except csv.Error as error:
        raise ContractError(f"dependency RECORD CSV is malformed: {distribution_name}") from error
    record_rows_by_path: dict[str, tuple[str, str]] = {}
    for row in record_rows:
        if len(row) != 3:
            raise ContractError(f"dependency RECORD row width changed: {distribution_name}")
        relative_text, recorded_hash, recorded_size = row
        if relative_text in record_rows_by_path:
            raise ContractError(f"dependency RECORD contains duplicate paths: {distribution_name}")
        record_rows_by_path[relative_text] = (recorded_hash, recorded_size)

    rows: list[list[object]] = []
    installed_files: dict[Path, tuple[int, str]] = {}
    record_self_count = 0
    for relative_text, recorded_hash, recorded_size in record_rows:
        path, ownership, ownership_root = _resolve_record_owned_path(
            distribution,
            relative_text,
            distribution_root=distribution_root,
        )
        size, digest = _sha256_file_content(
            path,
            description=f"{distribution_name} installed file {relative_text}",
            ownership_root=ownership_root,
        )
        is_record = path == record_path
        if is_record:
            record_self_count += 1
            if recorded_hash or recorded_size:
                raise ContractError(
                    f"dependency RECORD must leave its own hash empty: {distribution_name}"
                )
        elif not recorded_hash and not recorded_size:
            _unhashed_bytecode_source(
                relative_text,
                distribution_root=distribution_root,
                record_rows_by_path=record_rows_by_path,
            )
        else:
            expected_hash = _decode_record_sha256(
                recorded_hash,
                description=f"{distribution_name} {relative_text}",
            )
            try:
                expected_size = int(recorded_size)
            except ValueError as error:
                raise ContractError(
                    f"dependency RECORD size is malformed: {distribution_name}"
                ) from error
            if (
                str(expected_size) != recorded_size
                or expected_size != size
                or expected_hash != digest
            ):
                raise ContractError(
                    f"dependency installed bytes differ from RECORD: {relative_text}"
                )
        rows.append([relative_text, ownership, size, digest])
        installed_files[path] = (size, digest)
    if not rows or record_self_count != 1:
        raise ContractError(
            f"dependency RECORD is empty or lacks one self-entry: {distribution_name}"
        )
    module_name = distribution_name if import_name is None else import_name
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ContractError(f"dependency import failed: {module_name}") from error
    module_file_value = getattr(module, "__file__", None)
    if not isinstance(module_file_value, str) or not module_file_value or "\0" in module_file_value:
        raise ContractError(f"dependency import exposes no source file: {module_name}")
    module_file = Path(os.path.abspath(module_file_value))
    module_spec = getattr(module, "__spec__", None)
    module_origin = getattr(module_spec, "origin", None)
    if (
        not isinstance(module_origin, str)
        or not module_origin
        or "\0" in module_origin
        or Path(os.path.abspath(module_origin)) != module_file
    ):
        raise ContractError(f"dependency import origin is malformed: {module_name}")
    if not _is_beneath(module_file, distribution_root):
        raise ContractError(
            f"imported dependency is outside its recorded distribution: {distribution_name}"
        )
    _require_no_symlink_components(
        module_file,
        root=distribution_root,
        description=f"{distribution_name} imported module",
    )
    imported_identity = installed_files.get(module_file)
    if imported_identity is None:
        raise ContractError(f"imported dependency file is absent from RECORD: {distribution_name}")
    imported_size, imported_digest = imported_identity
    if (
        _sha256_file_content(
            module_file,
            description=f"{distribution_name} imported module after import",
            ownership_root=distribution_root,
        )
        != imported_identity
    ):
        raise ContractError(f"dependency entry file changed during import: {distribution_name}")
    return {
        "distribution": distribution_name,
        "distribution_root": str(distribution_root),
        "generated_script_count": sum(row[1] == "generated_script" for row in rows),
        "import_file": module_file.relative_to(distribution_root).as_posix(),
        "import_file_sha256": imported_digest,
        "import_file_size": imported_size,
        "import_name": module_name,
        "installed_file_count": len(rows),
        "installed_files_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
        "record_sha256": record_digest,
        "version": distribution.version,
    }


def _python_runtime_contract() -> dict[str, object]:
    executable = Path(sys.executable).resolve(strict=True)
    executable_size, executable_sha256 = _sha256_file_content(
        executable,
        description="processor evidence Python executable",
        ownership_root=executable.parent,
    )
    content = {
        "byte_order": sys.byteorder,
        "executable_sha256": executable_sha256,
        "executable_size": executable_size,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_version": platform.python_version(),
        "system": platform.system(),
    }
    return {**content, "sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest()}


def _dependency_environment_contract() -> dict[str, object]:
    packages = [
        _distribution_evidence(distribution, import_name=import_name)
        for distribution, import_name in DEPENDENCY_IMPORTS
    ]
    content = {"packages": packages, "python_runtime": _python_runtime_contract()}
    return {
        **content,
        "sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
    }


def _repository_source_contract() -> dict[str, object]:
    root = Path(_REPOSITORY_ROOT).resolve(strict=True)
    candidates = {
        *(path for parent in (root / "src", root / "tools") for path in parent.rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
        root / APPROVAL_SCHEMA_RELATIVE_PATH,
    }
    rows: list[list[object]] = []
    for path in sorted(candidates):
        candidate = Path(os.path.abspath(path))
        _require_no_symlink_components(
            candidate,
            root=root,
            description="processor evidence source contract file",
        )
        size, digest = _sha256_file_content(
            candidate,
            description=f"processor evidence source {candidate.relative_to(root).as_posix()}",
            ownership_root=root,
        )
        rows.append([candidate.relative_to(root).as_posix(), size, digest])
    if not rows:
        raise ContractError("processor evidence source contract is empty")
    return {
        "file_count": len(rows),
        "files": rows,
        "sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _expected_processor_tree() -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for relative_text in REQUIRED_PROCESSOR_FILES:
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ContractError("Qwen processor asset contract path is not canonical")
        files.add(relative_text)
        for depth in range(1, len(relative.parts)):
            directories.add(PurePosixPath(*relative.parts[:depth]).as_posix())
    return files, directories


def _processor_tree_from_disk(root: Path) -> tuple[set[str], set[str]]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ContractError("processor evidence requires no-follow directory traversal")
    files: set[str] = set()
    directories: set[str] = set()
    flags = os.O_RDONLY | nofollow | directory_flag | getattr(os, "O_CLOEXEC", 0)
    root_descriptor = _open_directory_descriptor(root, description="Qwen processor root")

    def visit(descriptor: int, prefix: tuple[str, ...]) -> None:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise ContractError("Qwen processor directory cannot be inventoried") from error
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\0" in name:
                raise ContractError("Qwen processor directory contains a noncanonical entry")
            relative_parts = (*prefix, name)
            relative_text = PurePosixPath(*relative_parts).as_posix()
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ContractError("Qwen processor entry cannot be inspected") from error
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative_text)
                try:
                    child = os.open(name, flags, dir_fd=descriptor)
                except OSError as error:
                    raise ContractError(
                        "Qwen processor directory changed during inventory"
                    ) from error
                try:
                    visit(child, relative_parts)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative_text)
            else:
                raise ContractError("Qwen processor tree contains a symlink or special file")

    try:
        visit(root_descriptor, ())
    finally:
        os.close(root_descriptor)
    return files, directories


def _verified_processor_payloads(
    processor_dir: Path,
) -> tuple[dict[str, object], dict[str, bytes]]:
    if set(REQUIRED_PROCESSOR_FILES) != set(PROCESSOR_ASSET_CONTRACT):
        raise ContractError("Qwen processor required-file and digest contracts differ")
    root = processor_dir.resolve(strict=True)
    if root != processor_dir or processor_dir.is_symlink():
        raise ContractError("Qwen processor root must be one resolved real directory")
    expected_files, expected_directories = _expected_processor_tree()
    actual_files, actual_directories = _processor_tree_from_disk(root)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ContractError(
            "Qwen processor complete file inventory differs from the pinned revision"
        )
    rows: list[list[object]] = []
    payloads: dict[str, bytes] = {}
    for relative_text, (expected_size, expected_digest) in sorted(PROCESSOR_ASSET_CONTRACT.items()):
        relative = PurePosixPath(relative_text)
        payload, digest = _read_regular_file(
            root.joinpath(*relative.parts),
            description=f"Qwen processor asset {relative_text}",
            maximum_bytes=expected_size,
            ownership_root=root,
        )
        if len(payload) != expected_size or digest != expected_digest:
            raise ContractError(f"Qwen processor asset differs: {relative_text}")
        payloads[relative_text] = payload
        rows.append([relative_text, len(payload), digest])
    inventory_sha256 = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    return {
        "processor_assets": asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        "processor_dir": str(root),
        "processor_file_inventory": rows,
        "processor_file_inventory_sha256": inventory_sha256,
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_load_protocol": PROCESSOR_SNAPSHOT_PROTOCOL,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "required_processor_files": len(REQUIRED_PROCESSOR_FILES),
    }, payloads


def _processor_asset_contract_from_disk(processor_dir: Path) -> dict[str, object]:
    contract, _ = _verified_processor_payloads(processor_dir)
    return contract


def _make_private_processor_snapshot(
    processor_dir: Path,
) -> tuple[Path, dict[str, object]]:
    source_contract, payloads = _verified_processor_payloads(processor_dir)
    snapshot = Path(tempfile.mkdtemp(prefix="picf-qwen-processor-snapshot-")).resolve(strict=True)
    try:
        for relative_text, payload in sorted(payloads.items()):
            relative = PurePosixPath(relative_text)
            target = snapshot.joinpath(*relative.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            target.chmod(0o400)
        for directory in sorted(
            (path for path in snapshot.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        snapshot.chmod(0o500)
        snapshot_contract, _ = _verified_processor_payloads(snapshot)
        comparable_fields = set(source_contract) - {"processor_dir"}
        if {key: source_contract[key] for key in comparable_fields} != {
            key: snapshot_contract[key] for key in comparable_fields
        }:
            raise ContractError("private processor snapshot differs from its verified source")
    except BaseException:
        _remove_private_processor_snapshot(snapshot, ignore_errors=True)
        raise
    return snapshot, source_contract


def _make_snapshot_directories_owner_writable(snapshot: Path) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ContractError("processor snapshot cleanup requires no-follow directory traversal")
    flags = os.O_RDONLY | nofollow | directory_flag | getattr(os, "O_CLOEXEC", 0)
    root_descriptor = _open_directory_descriptor(
        snapshot,
        description="private processor snapshot",
    )

    def visit(descriptor: int) -> None:
        os.fchmod(descriptor, 0o700)
        for name in os.listdir(descriptor):
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                continue
            try:
                child = os.open(name, flags, dir_fd=descriptor)
            except OSError:
                continue
            try:
                visit(child)
            finally:
                os.close(child)

    try:
        visit(root_descriptor)
    finally:
        os.close(root_descriptor)


def _remove_private_processor_snapshot(snapshot: Path, *, ignore_errors: bool = False) -> None:
    try:
        metadata = os.lstat(snapshot)
    except FileNotFoundError:
        return
    try:
        if stat.S_ISLNK(metadata.st_mode):
            snapshot.unlink()
            return
        if not stat.S_ISDIR(metadata.st_mode):
            raise ContractError("private processor snapshot path is no longer a directory")
        _make_snapshot_directories_owner_writable(snapshot)
        shutil.rmtree(snapshot)
    except (OSError, ContractError):
        if not ignore_errors:
            raise


def _validated_python_runtime(value: object) -> dict[str, object]:
    fields = {
        "byte_order",
        "executable_sha256",
        "executable_size",
        "implementation",
        "machine",
        "python_cache_tag",
        "python_version",
        "sha256",
        "system",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError("processor Python runtime contract fields changed")
    executable_size = value.get("executable_size")
    if (
        value.get("byte_order") not in {"little", "big"}
        or isinstance(executable_size, bool)
        or not isinstance(executable_size, int)
        or executable_size <= 0
    ):
        raise ContractError("processor Python runtime identity changed")
    for field in (
        "implementation",
        "machine",
        "python_cache_tag",
        "python_version",
        "system",
    ):
        item = value.get(field)
        if not isinstance(item, str) or not item or "\0" in item:
            raise ContractError("processor Python runtime text identity changed")
    content = {field: value[field] for field in sorted(fields - {"sha256"})}
    content["executable_sha256"] = _require_sha256(
        value.get("executable_sha256"),
        name="processor Python executable SHA-256",
    )
    expected_digest = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    if value.get("sha256") != expected_digest:
        raise ContractError("processor Python runtime digest changed")
    return {**content, "sha256": expected_digest}


def _validated_dependency_environment(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "packages",
        "python_runtime",
        "sha256",
    }:
        raise ContractError("processor dependency environment contract fields changed")
    packages = value.get("packages")
    if not isinstance(packages, list) or len(packages) != len(DEPENDENCY_DISTRIBUTIONS):
        raise ContractError("processor dependency environment package set changed")
    normalized: list[dict[str, object]] = []
    for (expected_name, expected_import), package in zip(
        DEPENDENCY_IMPORTS,
        packages,
        strict=True,
    ):
        if not isinstance(package, Mapping) or set(package) != {
            "distribution",
            "distribution_root",
            "generated_script_count",
            "import_file",
            "import_file_sha256",
            "import_file_size",
            "import_name",
            "installed_file_count",
            "installed_files_sha256",
            "record_sha256",
            "version",
        }:
            raise ContractError("processor dependency package contract fields changed")
        count = package.get("installed_file_count")
        script_count = package.get("generated_script_count")
        import_file_size = package.get("import_file_size")
        version = package.get("version")
        distribution_root = package.get("distribution_root")
        import_file = package.get("import_file")
        if (
            package.get("distribution") != expected_name
            or package.get("import_name") != expected_import
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(script_count, bool)
            or not isinstance(script_count, int)
            or not 0 <= script_count <= count
            or isinstance(import_file_size, bool)
            or not isinstance(import_file_size, int)
            or import_file_size <= 0
            or not isinstance(version, str)
            or not version
            or not isinstance(distribution_root, str)
            or not Path(distribution_root).is_absolute()
            or not isinstance(import_file, str)
            or not import_file
            or PurePosixPath(import_file).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(import_file).parts)
        ):
            raise ContractError("processor dependency package identity changed")
        normalized.append(
            {
                "distribution": expected_name,
                "distribution_root": distribution_root,
                "generated_script_count": script_count,
                "import_file": import_file,
                "import_file_sha256": _require_sha256(
                    package.get("import_file_sha256"),
                    name=f"{expected_name} imported-file SHA-256",
                ),
                "import_file_size": import_file_size,
                "import_name": expected_import,
                "installed_file_count": count,
                "installed_files_sha256": _require_sha256(
                    package.get("installed_files_sha256"),
                    name=f"{expected_name} installed-files SHA-256",
                ),
                "record_sha256": _require_sha256(
                    package.get("record_sha256"),
                    name=f"{expected_name} RECORD SHA-256",
                ),
                "version": version,
            }
        )
    if normalized[0]["version"] != ADR125_TRANSFORMERS_VERSION:
        raise ContractError("processor transformers version differs from the pinned contract")
    python_runtime = _validated_python_runtime(value.get("python_runtime"))
    content = {"packages": normalized, "python_runtime": python_runtime}
    expected_digest = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    if value.get("sha256") != expected_digest:
        raise ContractError("processor dependency environment digest changed")
    return {**content, "sha256": expected_digest}


class _ProcessorOutputAuditProxy:
    """Capture exactly the semantic tensors returned by the real processor."""

    def __init__(self, processor: object) -> None:
        self._processor = processor
        self._pending: dict[str, dict[str, object]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def apply_chat_template(self, *args: object, **kwargs: object) -> object:
        if self._pending is not None:
            raise ContractError("processor tensor evidence was not consumed exactly once")
        apply_template = getattr(self._processor, "apply_chat_template", None)
        if not callable(apply_template):
            raise TypeError("native VL processor exposes no Qwen chat template")
        result = apply_template(*args, **kwargs)
        if not isinstance(result, Mapping):
            raise ContractError("Qwen chat template returned no tensor mapping")
        keys = tuple(result.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ContractError("Qwen processor returned a non-text output key")
        key_set = set(keys)
        allowed = set(PROCESSOR_OUTPUT_TENSOR_NAMES)
        required = allowed - {"position_ids"}
        if key_set not in (required, allowed):
            raise ContractError(f"Qwen processor output key set changed: {sorted(key_set)}")
        captured: dict[str, dict[str, object]] = {}
        for key, value in result.items():
            if not isinstance(key, str) or key not in key_set or key not in allowed:
                raise ContractError("Qwen processor output mapping changed during inspection")
            captured[key] = cpu_tensor_evidence(
                require_cpu_tensor(value, name=f"processor output {key}")
            )
        if set(captured) != key_set:
            raise ContractError("Qwen processor output mapping changed during inspection")
        self._pending = captured
        return result

    def consume(self) -> dict[str, dict[str, object]]:
        if self._pending is None:
            raise ContractError("Qwen processor tensor evidence was not captured")
        result = self._pending
        self._pending = None
        return result


def _validate_manifest_record_set(
    manifest: PublicNativeVLRetentionManifest,
) -> tuple[PublicNativeVLManifestRecord, ...]:
    records = tuple(manifest.records)
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ContractError("public native VL processor audit requires exactly 192 records")
    if tuple(record.record_id for record in records) != tuple(
        sorted(record.record_id for record in records)
    ):
        raise ContractError("public native VL processor audit record order changed")

    counts = Counter(f"{record.family}/{record.partition}" for record in records)
    if dict(sorted(counts.items())) != dict(sorted(EXPECTED_COUNTS.items())):
        raise ContractError("public native VL processor audit family/partition coverage changed")
    if dict(sorted(manifest.family_partition_counts.items())) != dict(
        sorted(EXPECTED_COUNTS.items())
    ):
        raise ContractError("public native VL processor audit manifest counts changed")

    _require_unique([record.record_id for record in records], name="record ID set")
    _require_unique([record.record_sha256 for record in records], name="record SHA-256 set")
    _require_unique(
        [
            (record.source_key, record.source_row_index, record.source_subindex)
            for record in records
        ],
        name="source-location set",
    )
    _require_unique([record.image_file for record in records], name="image-path set")
    _require_unique([record.image_file_sha256 for record in records], name="image-file SHA-256 set")
    _require_unique([record.image_rgb_sha256 for record in records], name="image-RGB SHA-256 set")

    if set(manifest.sources) != set(PUBLIC_NATIVE_VL_FAMILIES):
        raise ContractError("public native VL processor audit source families changed")
    for record in records:
        if record.family not in PUBLIC_NATIVE_VL_FAMILIES:
            raise ContractError("public native VL processor audit found an unsupported family")
        if record.partition not in PUBLIC_NATIVE_VL_PARTITIONS:
            raise ContractError("public native VL processor audit found an unsupported partition")
        if record.source_key != record.family or record.source_key not in manifest.sources:
            raise ContractError("public native VL processor audit record/source binding changed")
        for name, value in (
            ("record SHA-256", record.record_sha256),
            ("source priority SHA-256", record.priority_sha256),
            ("image file SHA-256", record.image_file_sha256),
            ("image RGB SHA-256", record.image_rgb_sha256),
        ):
            _require_sha256(value, name=name)
        source = manifest.sources[record.source_key]
        _require_sha256(source.source_file_sha256, name="source file SHA-256")
    return records


def _artifact_image_snapshot(
    manifest: PublicNativeVLRetentionManifest,
    artifact_root: Path,
) -> dict[str, object]:
    root = artifact_root.resolve(strict=True)
    if artifact_root.is_symlink() or root != artifact_root:
        raise ContractError("public native VL artifact root must be one resolved real directory")
    rows: list[list[object]] = []
    for descriptor in manifest.records:
        relative = PurePosixPath(descriptor.image_file)
        if (
            relative.is_absolute()
            or relative.as_posix() != descriptor.image_file
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ContractError("public native VL image path changed after manifest validation")
        candidate = root.joinpath(*relative.parts)
        _require_no_symlink_components(
            candidate,
            root=root,
            description=f"public native VL image {descriptor.record_id}",
        )
        size, digest = _sha256_file_content(
            candidate,
            description=f"public native VL image {descriptor.record_id}",
            ownership_root=root,
        )
        if digest != descriptor.image_file_sha256:
            raise ContractError("public native VL artifact image changed after processing")
        rows.append([descriptor.record_id, descriptor.image_file, size, digest])
    if len(rows) != EXPECTED_RECORD_COUNT:
        raise ContractError("public native VL artifact image snapshot is incomplete")
    return {
        "image_count": len(rows),
        "sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _validated_processor_contract(
    value: Mapping[str, object],
    *,
    processor_dir: Path,
) -> dict[str, object]:
    required_fields = {
        "dependency_environment",
        "processor_assets",
        "processor_dir",
        "processor_file_inventory",
        "processor_file_inventory_sha256",
        "processor_id",
        "processor_load_protocol",
        "processor_revision",
        "required_processor_files",
    }
    if set(value) != required_fields:
        raise ContractError("Qwen processor identity contract fields changed")
    expected_assets = asset_contract_manifest(PROCESSOR_ASSET_CONTRACT)
    dependency_environment = _validated_dependency_environment(value.get("dependency_environment"))
    if (
        value["processor_id"] != QWEN_PROCESSOR_ID
        or value["processor_revision"] != QWEN_PROCESSOR_REVISION
        or value["processor_dir"] != str(processor_dir.expanduser().resolve())
        or value["required_processor_files"] != len(REQUIRED_PROCESSOR_FILES)
        or value["processor_assets"] != expected_assets
        or value["processor_load_protocol"] != PROCESSOR_SNAPSHOT_PROTOCOL
    ):
        raise ContractError("Qwen processor identity differs from the pinned contract")
    expected_inventory = [
        [name, size, digest] for name, (size, digest) in sorted(PROCESSOR_ASSET_CONTRACT.items())
    ]
    expected_inventory_sha256 = hashlib.sha256(_canonical_bytes(expected_inventory)).hexdigest()
    if (
        value["processor_file_inventory"] != expected_inventory
        or value["processor_file_inventory_sha256"] != expected_inventory_sha256
    ):
        raise ContractError("Qwen processor complete inventory differs from the pinned contract")
    return {
        "dependency_environment": dependency_environment,
        "processor_assets": expected_assets,
        "processor_dir": str(processor_dir.expanduser().resolve()),
        "processor_file_inventory": expected_inventory,
        "processor_file_inventory_sha256": expected_inventory_sha256,
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_load_protocol": PROCESSOR_SNAPSHOT_PROTOCOL,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "required_processor_files": len(REQUIRED_PROCESSOR_FILES),
    }


def _require_cpu_batch(batch: NativeVLGroundingBatch) -> None:
    tensor_names = (
        "input_ids",
        "attention_mask",
        "labels",
        "assistant_token_mask",
        "pixel_values",
        "image_grid_thw",
    )
    for name in tensor_names:
        require_cpu_tensor(getattr(batch, name), name=name)
    if batch.position_ids is not None:
        require_cpu_tensor(batch.position_ids, name="position_ids")


def _semantic_tensor_contract(
    batch: NativeVLGroundingBatch,
    processor_tensors: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    values: dict[str, torch.Tensor | None] = {
        "attention_mask": batch.attention_mask,
        "image_grid_thw": batch.image_grid_thw,
        "input_ids": batch.input_ids,
        "label_mask": batch.assistant_token_mask,
        "labels": batch.labels,
        "pixel_values": batch.pixel_values,
        "position_ids": batch.position_ids,
    }
    result: dict[str, dict[str, object]] = {}
    for name in SEMANTIC_TENSOR_NAMES:
        tensor = values[name]
        if tensor is None:
            result[name] = {"present": False}
            continue
        evidence = cpu_tensor_evidence(require_cpu_tensor(tensor, name=name))
        result[name] = {**evidence, "present": True}
        returned = processor_tensors.get(name)
        if name not in {"label_mask", "labels"} and returned != evidence:
            raise ContractError(f"processor-returned {name} changed before batch construction")
    if ("position_ids" in processor_tensors) != (batch.position_ids is not None):
        raise ContractError("processor position_ids presence changed before batch construction")
    return result


def _record_evidence(
    *,
    manifest: PublicNativeVLRetentionManifest,
    descriptor: PublicNativeVLManifestRecord,
    artifact_root: Path,
    processor: _ProcessorOutputAuditProxy,
    merge_size: int,
) -> dict[str, object]:
    record = manifest.materialize_record(descriptor, artifact_root=artifact_root)
    if not isinstance(record, NativeVLInstructionRecord):
        raise ContractError("public native VL processor audit materialized an invalid record")
    if (
        record.record_id != descriptor.record_id
        or record.family != descriptor.family
        or record.user_text != descriptor.user_text
        or record.assistant_text != descriptor.assistant_text
        or tuple(record.image.shape) != (descriptor.height, descriptor.width, 3)
        or native_vl_rgb_sha256(record.image) != descriptor.image_rgb_sha256
    ):
        raise ContractError("public native VL processor audit materialization binding changed")

    batch = build_native_vl_grounding_batch(record, processor)
    processor_tensors = processor.consume()
    _require_cpu_batch(batch)
    semantic_tensors = _semantic_tensor_contract(batch, processor_tensors)
    grid = batch.image_grid_thw.detach().tolist()
    budget = validate_native_processor_record_grid(
        grid,
        image_height=descriptor.height,
        image_width=descriptor.width,
        lattice=AUDITED_LATTICE,
        merge_size=merge_size,
    )
    pixel_values_shape = list(batch.pixel_values.shape)
    if not pixel_values_shape or pixel_values_shape[0] != budget["raw_patch_tokens"]:
        raise ContractError("public native VL pixel patch count differs from its Qwen grid")
    sequence_length = int(batch.input_ids.shape[1])
    supervised_token_count = batch.supervised_token_count
    if sequence_length <= 0 or not 0 < supervised_token_count <= sequence_length:
        raise ContractError("public native VL supervised token budget is invalid")
    source = manifest.sources[descriptor.source_key]
    content = {
        "assistant_mask_sha256": cpu_tensor_sha256(batch.assistant_token_mask),
        "attention_mask_sha256": cpu_tensor_sha256(batch.attention_mask),
        "family": descriptor.family,
        "grid_budget": {
            "maximum_merged_visual_tokens": budget["maximum_merged_visual_tokens"],
            "maximum_raw_patch_tokens": budget["maximum_raw_patch_tokens"],
            "merged_visual_tokens": budget["merged_visual_tokens"],
            "raw_patch_tokens": budget["raw_patch_tokens"],
        },
        "image": {
            "height": descriptor.height,
            "image_file": descriptor.image_file,
            "image_file_sha256": descriptor.image_file_sha256,
            "image_rgb_sha256": descriptor.image_rgb_sha256,
            "width": descriptor.width,
        },
        "image_height": descriptor.height,
        "image_grid_thw": budget["image_grid_thw"],
        "image_grid_thw_sha256": cpu_tensor_sha256(batch.image_grid_thw),
        "image_rgb_sha256": descriptor.image_rgb_sha256,
        "image_width": descriptor.width,
        "input_ids_sha256": cpu_tensor_sha256(batch.input_ids),
        "labels_sha256": cpu_tensor_sha256(batch.labels),
        "partition": descriptor.partition,
        "pixel_patch_count": pixel_values_shape[0],
        "pixel_values_sha256": cpu_tensor_sha256(batch.pixel_values),
        "pixel_values_shape": pixel_values_shape,
        "position_ids_sha256": (
            None if batch.position_ids is None else cpu_tensor_sha256(batch.position_ids)
        ),
        "processor_output_tensors": processor_tensors,
        "record_id": descriptor.record_id,
        "record_sha256": descriptor.record_sha256,
        "sequence_length": sequence_length,
        "source": {
            "dataset_id": source.dataset_id,
            "dataset_revision": source.dataset_revision,
            "priority_sha256": descriptor.priority_sha256,
            "source_file": source.source_file,
            "source_file_sha256": source.source_file_sha256,
            "source_key": descriptor.source_key,
            "source_row_index": descriptor.source_row_index,
            "source_subindex": descriptor.source_subindex,
        },
        "source_row_index": descriptor.source_row_index,
        "source_subindex": descriptor.source_subindex,
        "supervised_token_count": supervised_token_count,
        "semantic_tensors": semantic_tensors,
        "target_answer_sha256": text_sha256(descriptor.assistant_text),
        "user_text_sha256": text_sha256(descriptor.user_text),
    }
    return {
        **content,
        "record_evidence_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
    }


def _schedule_row(evidence: Mapping[str, object]) -> list[object]:
    return public_vl_schedule_row(
        family=evidence.get("family"),
        record_id=evidence.get("record_id"),
        record_sha256=evidence.get("record_sha256"),
        image_rgb_sha256=evidence.get("image_rgb_sha256"),
        image_height=evidence.get("image_height"),
        image_width=evidence.get("image_width"),
        source_row_index=evidence.get("source_row_index"),
        source_subindex=evidence.get("source_subindex"),
        supervised_token_count=evidence.get("supervised_token_count"),
        target_answer_sha256=evidence.get("target_answer_sha256"),
        user_text_sha256=evidence.get("user_text_sha256"),
    )


def _build_schedule_contract(
    manifest: PublicNativeVLRetentionManifest,
    evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_record_id = {item.get("record_id"): item for item in evidence}
    if len(by_record_id) != EXPECTED_RECORD_COUNT or None in by_record_id:
        raise ContractError("public native VL evidence record lookup is incomplete")

    heldout_rows = [
        _schedule_row(by_record_id[descriptor.record_id])
        for family in PUBLIC_NATIVE_VL_FAMILIES
        for descriptor in manifest.records_for(family, "heldout")
    ]
    training_rows = [
        _schedule_row(
            by_record_id[
                manifest.training_record_for_rank(
                    optimizer_step=optimizer_step,
                    rank=rank,
                ).record_id
            ]
        )
        for optimizer_step in range(PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY)
        for rank in range(len(PUBLIC_NATIVE_VL_FAMILIES))
    ]
    expected_heldout = len(PUBLIC_NATIVE_VL_FAMILIES) * PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
    expected_training = len(PUBLIC_NATIVE_VL_FAMILIES) * PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY
    if (
        len(heldout_rows) != expected_heldout
        or len(training_rows) != expected_training
        or len({row[1] for row in heldout_rows}) != expected_heldout
        or len({row[1] for row in training_rows}) != expected_training
    ):
        raise ContractError("public native VL schedule coverage is incomplete")
    return {
        "field_order": list(PUBLIC_VL_SCHEDULE_FIELDS),
        "heldout": {
            "ordering": "family_then_manifest_priority",
            "record_count": len(heldout_rows),
            "rows": heldout_rows,
            "sha256": public_vl_schedule_sha256(heldout_rows),
        },
        "training": {
            "optimizer_steps": PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY,
            "ordering": "optimizer_step_major_then_rank_0_1",
            "ranks_per_step": len(PUBLIC_NATIVE_VL_FAMILIES),
            "record_count": len(training_rows),
            "rows": training_rows,
            "sha256": public_vl_schedule_sha256(training_rows),
        },
    }


def _tensor_aggregate_contract(
    evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    semantic_per_tensor: dict[str, str] = {}
    for name in SEMANTIC_TENSOR_NAMES:
        rows: list[list[object]] = []
        for record in evidence:
            tensors = record.get("semantic_tensors")
            if not isinstance(tensors, Mapping) or not isinstance(tensors.get(name), Mapping):
                raise ContractError("semantic tensor evidence is incomplete")
            item = tensors[name]
            rows.append(
                [
                    record.get("record_id"),
                    item.get("present"),
                    item.get("dtype"),
                    item.get("shape"),
                    item.get("numel"),
                    item.get("sha256"),
                ]
            )
        semantic_per_tensor[name] = tensor_evidence_aggregate_sha256(rows)

    processor_rows: list[list[object]] = []
    for record in evidence:
        tensors = record.get("processor_output_tensors")
        if not isinstance(tensors, Mapping):
            raise ContractError("processor-output tensor evidence is incomplete")
        processor_rows.append(
            [
                record.get("record_id"),
                [
                    [name, tensors[name]]
                    for name in PROCESSOR_OUTPUT_TENSOR_NAMES
                    if name in tensors
                ],
            ]
        )
    semantic_content = {
        "field_order": list(SEMANTIC_TENSOR_NAMES),
        "per_tensor_sha256": semantic_per_tensor,
    }
    processor_content = {
        "allowed_field_order": list(PROCESSOR_OUTPUT_TENSOR_NAMES),
        "rows_sha256": tensor_evidence_aggregate_sha256(processor_rows),
    }
    return {
        "processor_output": {
            **processor_content,
            "sha256": hashlib.sha256(_canonical_bytes(processor_content)).hexdigest(),
        },
        "semantic": {
            **semantic_content,
            "sha256": hashlib.sha256(_canonical_bytes(semantic_content)).hexdigest(),
        },
    }


def build_public_native_vl_processor_budget_audit(
    *,
    manifest: PublicNativeVLRetentionManifest,
    artifact_root: Path,
    manifest_file_sha256: str,
    processor: object,
    processor_identity: Mapping[str, object],
    processor_dir: Path,
) -> dict[str, object]:
    """Build internal evidence; this function never authorizes a publishable PASS."""

    manifest_digest = _require_sha256(
        manifest_file_sha256,
        name="public native VL manifest file SHA-256",
    )
    records = _validate_manifest_record_set(manifest)
    identity = _validated_processor_contract(processor_identity, processor_dir=processor_dir)
    audited_processor = _ProcessorOutputAuditProxy(processor)
    area_budget = configure_native_processor_area_budget(audited_processor, lattice=AUDITED_LATTICE)
    merge_size = area_budget.get("merge_size")
    if isinstance(merge_size, bool) or not isinstance(merge_size, int) or merge_size <= 0:
        raise ContractError("Qwen processor area-budget merge size is invalid")

    evidence = []
    for descriptor in records:
        evidence.append(
            _record_evidence(
                manifest=manifest,
                descriptor=descriptor,
                artifact_root=artifact_root,
                processor=audited_processor,
                merge_size=merge_size,
            )
        )
    if tuple(item["record_id"] for item in evidence) != tuple(
        descriptor.record_id for descriptor in records
    ):
        raise RuntimeError("public native VL processor audit omitted or reordered records")

    merged_tokens = [
        int(item["grid_budget"]["merged_visual_tokens"])  # type: ignore[index]
        for item in evidence
    ]
    raw_tokens = [
        int(item["grid_budget"]["raw_patch_tokens"])  # type: ignore[index]
        for item in evidence
    ]
    sequence_lengths = [int(item["sequence_length"]) for item in evidence]
    supervised_tokens = [int(item["supervised_token_count"]) for item in evidence]
    summary = {
        "family_partition_counts": dict(sorted(EXPECTED_COUNTS.items())),
        "merged_visual_token_maximum": max(merged_tokens),
        "merged_visual_token_minimum": min(merged_tokens),
        "raw_patch_token_maximum": max(raw_tokens),
        "raw_patch_token_minimum": min(raw_tokens),
        "record_count": len(evidence),
        "sequence_length_maximum": max(sequence_lengths),
        "sequence_length_minimum": min(sequence_lengths),
        "supervised_token_maximum": max(supervised_tokens),
        "supervised_token_minimum": min(supervised_tokens),
    }
    schedules = _build_schedule_contract(manifest, evidence)
    tensor_aggregates = _tensor_aggregate_contract(evidence)
    artifact_image_snapshot = _artifact_image_snapshot(manifest, artifact_root)
    record_aggregate_sha256 = hashlib.sha256(
        _canonical_bytes([item["record_evidence_sha256"] for item in evidence])
    ).hexdigest()
    records_sha256 = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
    dependency_environment = identity["dependency_environment"]
    if not isinstance(dependency_environment, Mapping):
        raise ContractError("processor dependency environment disappeared")
    processor_contract = {**identity, "area_budget": area_budget}
    public_contract = {
        "artifact_sha256": manifest.artifact_sha256,
        "family_partition_counts": dict(sorted(manifest.family_partition_counts.items())),
        "manifest_file_sha256": manifest_digest,
        "manifest_schema": PUBLIC_NATIVE_VL_MANIFEST_SCHEMA,
        "record_count": len(records),
    }
    production_evidence = {
        "artifact_image_snapshot": artifact_image_snapshot,
        "processor_contract": processor_contract,
        "public_native_vl_contract": public_contract,
        "record_aggregate_sha256": record_aggregate_sha256,
        "records_sha256": records_sha256,
        "schedules": schedules,
        "summary": summary,
        "tensor_aggregates": tensor_aggregates,
    }
    production_evidence_sha256 = hashlib.sha256(_canonical_bytes(production_evidence)).hexdigest()
    source_contract = _repository_source_contract()
    tool_sha256 = _sha256_file_content(
        Path(__file__).resolve(strict=True),
        description="processor evidence tool",
        ownership_root=Path(_REPOSITORY_ROOT).resolve(strict=True),
    )[1]
    content = {
        "artifact_image_snapshot": artifact_image_snapshot,
        "device_contract": {
            "observed_processor_tensor_device": "cpu",
            "torch_global_cuda_initialization_not_asserted": True,
        },
        "publication_authorized": False,
        "processor_contract": processor_contract,
        "production_evidence": production_evidence,
        "production_evidence_sha256": production_evidence_sha256,
        "public_native_vl_contract": public_contract,
        "records": evidence,
        "record_aggregate_sha256": record_aggregate_sha256,
        "records_sha256": records_sha256,
        "schedules": schedules,
        "schema": INTERNAL_SCHEMA,
        "status": INTERNAL_STATUS,
        "summary": summary,
        "source_contract": source_contract,
        "tensor_aggregates": tensor_aggregates,
        "tool_sha256": tool_sha256,
    }
    return {
        **content,
        "evidence_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
    }


def _processor_contract_without_path(value: Mapping[str, object]) -> dict[str, object]:
    return {key: value[key] for key in value if key != "processor_dir"}


@contextmanager
def _load_pinned_processor(
    processor_dir: Path,
) -> Iterator[tuple[object, dict[str, object]]]:
    from transformers import AutoProcessor, __version__ as transformers_version

    if transformers_version != ADR125_TRANSFORMERS_VERSION:
        raise ContractError(
            "ADR-125 processor audit requires "
            f"transformers=={ADR125_TRANSFORMERS_VERSION}, got {transformers_version}"
        )
    root = processor_dir.expanduser().resolve(strict=True)
    snapshot, source_contract = _make_private_processor_snapshot(root)
    try:
        # QWEN_PROCESSOR_REVISION is an exact commit and this load is local-only.
        processor = AutoProcessor.from_pretrained(  # nosec B615
            snapshot,
            padding_side="right",
            revision=QWEN_PROCESSOR_REVISION,
            trust_remote_code=False,
            local_files_only=True,
        )
        loaded_contract = _processor_asset_contract_from_disk(snapshot)
        if _processor_contract_without_path(loaded_contract) != _processor_contract_without_path(
            source_contract
        ):
            raise ContractError("Qwen processor snapshot changed while being loaded")
        dependency_environment = _dependency_environment_contract()
        packages = dependency_environment["packages"]
        if not isinstance(packages, list) or packages[0].get("version") != transformers_version:
            raise ContractError("transformers import and distribution versions differ")
        yield processor, {**source_contract, "dependency_environment": dependency_environment}
        final_contract = _processor_asset_contract_from_disk(snapshot)
        if _processor_contract_without_path(final_contract) != _processor_contract_without_path(
            source_contract
        ):
            raise ContractError("Qwen processor snapshot changed during evidence capture")
    finally:
        _remove_private_processor_snapshot(snapshot)


def _resolve_production_paths(
    *,
    artifact_root: Path,
    manifest_path: Path,
    processor_dir: Path,
) -> tuple[Path, Path, Path]:
    named = {
        "artifact root": artifact_root.expanduser(),
        "manifest": manifest_path.expanduser(),
        "processor": processor_dir.expanduser(),
    }
    if any(not path.is_absolute() for path in named.values()):
        raise ContractError("ADR-125 processor audit paths must be absolute")
    root_input = named["artifact root"]
    manifest_input = named["manifest"]
    processor_input = named["processor"]
    if root_input.is_symlink() or not root_input.is_dir():
        raise ContractError("ADR-125 public artifact root must be one real directory")
    if manifest_input.is_symlink() or not manifest_input.is_file():
        raise ContractError("ADR-125 public manifest must be one real file")
    if processor_input.is_symlink() or not processor_input.is_dir():
        raise ContractError("ADR-125 processor root must be one real directory")
    root = root_input.resolve(strict=True)
    manifest = manifest_input.resolve(strict=True)
    processor = processor_input.resolve(strict=True)
    if manifest != root / ADR125_PUBLIC_MANIFEST_FILE:
        raise ContractError("ADR-125 manifest path is not the frozen artifact manifest")
    for relative in REQUIRED_PROCESSOR_FILES:
        asset = processor / relative
        if asset.is_symlink() or not asset.is_file():
            raise ContractError(f"ADR-125 processor asset is not one real file: {relative}")
    return root, manifest, processor


def _sha256_regular_file(path: Path) -> str:
    return _sha256_file_content(
        path,
        description="ADR-125 public manifest",
    )[1]


def _validate_adr125_internal_report(
    *,
    internal: Mapping[str, object],
    manifest: PublicNativeVLRetentionManifest,
    artifact_root: Path,
    manifest_path: Path,
    processor_dir: Path,
    tool_sha256: str,
) -> tuple[dict[str, object], str]:
    resolved_paths = _resolve_production_paths(
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
    )
    if resolved_paths != (artifact_root, manifest_path, processor_dir):
        raise ContractError("ADR-125 authorization received unresolved paths")
    if _sha256_regular_file(manifest_path) != ADR125_PUBLIC_MANIFEST_FILE_SHA256:
        raise ContractError("ADR-125 manifest changed before report authorization")
    try:
        final_processor_identity = _processor_asset_contract_from_disk(processor_dir)
    except (OSError, RuntimeError, ValueError) as error:
        raise ContractError("ADR-125 processor assets cannot be revalidated") from error
    if final_processor_identity != {
        "processor_assets": asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        "processor_dir": str(processor_dir),
        "processor_file_inventory": [
            [name, size, digest]
            for name, (size, digest) in sorted(PROCESSOR_ASSET_CONTRACT.items())
        ],
        "processor_file_inventory_sha256": hashlib.sha256(
            _canonical_bytes(
                [
                    [name, size, digest]
                    for name, (size, digest) in sorted(PROCESSOR_ASSET_CONTRACT.items())
                ]
            )
        ).hexdigest(),
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_load_protocol": PROCESSOR_SNAPSHOT_PROTOCOL,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "required_processor_files": len(REQUIRED_PROCESSOR_FILES),
    }:
        raise ContractError("ADR-125 processor assets changed before report authorization")
    if not isinstance(manifest, PublicNativeVLRetentionManifest):
        raise ContractError("ADR-125 production authorization requires the typed frozen manifest")
    if (
        manifest.artifact_sha256 != ADR125_PUBLIC_ARTIFACT_SHA256
        or internal.get("schema") != INTERNAL_SCHEMA
        or internal.get("status") != INTERNAL_STATUS
        or internal.get("publication_authorized") is not False
    ):
        raise ContractError("ADR-125 internal evidence does not bind the frozen artifact")
    evidence_sha256 = _require_sha256(
        internal.get("evidence_sha256"),
        name="internal evidence SHA-256",
    )
    internal_content = dict(internal)
    internal_content.pop("evidence_sha256", None)
    if hashlib.sha256(_canonical_bytes(internal_content)).hexdigest() != evidence_sha256:
        raise ContractError("ADR-125 internal evidence digest changed")
    public_contract = internal.get("public_native_vl_contract")
    processor_contract = internal.get("processor_contract")
    if not isinstance(public_contract, Mapping) or not isinstance(processor_contract, Mapping):
        raise ContractError("ADR-125 internal evidence contracts are malformed")
    if (
        public_contract.get("artifact_sha256") != ADR125_PUBLIC_ARTIFACT_SHA256
        or public_contract.get("manifest_file_sha256") != ADR125_PUBLIC_MANIFEST_FILE_SHA256
        or public_contract.get("record_count") != EXPECTED_RECORD_COUNT
        or processor_contract.get("processor_dir") != str(processor_dir)
        or processor_contract.get("processor_assets")
        != asset_contract_manifest(PROCESSOR_ASSET_CONTRACT)
        or processor_contract.get("processor_revision") != QWEN_PROCESSOR_REVISION
    ):
        raise ContractError("ADR-125 internal evidence identity changed")
    if internal.get("artifact_image_snapshot") != _artifact_image_snapshot(
        manifest,
        artifact_root,
    ):
        raise ContractError("ADR-125 artifact images changed before report publication")
    captured_dependency_environment = _validated_dependency_environment(
        processor_contract.get("dependency_environment")
    )
    if _dependency_environment_contract() != captured_dependency_environment:
        raise ContractError("ADR-125 dependency environment changed during processor audit")
    source_contract = internal.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise ContractError("ADR-125 source contract is malformed")
    if _repository_source_contract() != source_contract:
        raise ContractError("ADR-125 executable source changed during processor audit")
    current_tool_sha256 = _sha256_file_content(
        Path(__file__).resolve(strict=True),
        description="processor evidence tool",
        ownership_root=Path(_REPOSITORY_ROOT).resolve(strict=True),
    )[1]
    if tool_sha256 != current_tool_sha256 or internal.get("tool_sha256") != tool_sha256:
        raise ContractError("ADR-125 processor audit tool changed during execution")
    _normalized_production_evidence(internal)
    return internal_content, evidence_sha256


def _normalized_production_evidence(internal: Mapping[str, object]) -> dict[str, object]:
    required = {
        "artifact_image_snapshot": internal.get("artifact_image_snapshot"),
        "processor_contract": internal.get("processor_contract"),
        "public_native_vl_contract": internal.get("public_native_vl_contract"),
        "record_aggregate_sha256": internal.get("record_aggregate_sha256"),
        "records_sha256": internal.get("records_sha256"),
        "schedules": internal.get("schedules"),
        "summary": internal.get("summary"),
        "tensor_aggregates": internal.get("tensor_aggregates"),
    }
    for name in ("record_aggregate_sha256", "records_sha256"):
        _require_sha256(required[name], name=name)
    if any(
        not isinstance(required[name], Mapping)
        for name in required
        if name
        not in {
            "record_aggregate_sha256",
            "records_sha256",
        }
    ):
        raise ContractError("ADR-125 normalized production evidence is malformed")
    if internal.get("production_evidence") != required:
        raise ContractError("ADR-125 declared production evidence differs from complete evidence")
    expected_digest = hashlib.sha256(_canonical_bytes(required)).hexdigest()
    if internal.get("production_evidence_sha256") != expected_digest:
        raise ContractError("ADR-125 normalized production evidence digest changed")
    return required


def _validated_report_records(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    records = report.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise ContractError("processor evidence report does not contain exactly 192 records")
    normalized: list[Mapping[str, object]] = []
    record_ids: list[str] = []
    evidence_digests: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ContractError("processor evidence report contains a malformed record")
        digest = _require_sha256(
            record.get("record_evidence_sha256"),
            name="record evidence SHA-256",
        )
        content = dict(record)
        content.pop("record_evidence_sha256", None)
        if hashlib.sha256(_canonical_bytes(content)).hexdigest() != digest:
            raise ContractError("processor evidence record digest changed")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id or "\0" in record_id:
            raise ContractError("processor evidence record ID is malformed")
        normalized.append(record)
        record_ids.append(record_id)
        evidence_digests.append(digest)
    _require_unique(record_ids, name="report record ID set")
    if record_ids != sorted(record_ids):
        raise ContractError("processor evidence report record ordering changed")
    if report.get("records_sha256") != hashlib.sha256(_canonical_bytes(records)).hexdigest():
        raise ContractError("processor evidence complete record digest changed")
    if (
        report.get("record_aggregate_sha256")
        != hashlib.sha256(_canonical_bytes(evidence_digests)).hexdigest()
    ):
        raise ContractError("processor evidence record aggregate changed")
    return normalized


def _artifact_image_snapshot_from_report(
    report: Mapping[str, object],
    *,
    artifact_root: Path,
) -> dict[str, object]:
    records = _validated_report_records(report)
    rows: list[list[object]] = []
    for record in records:
        image = record.get("image")
        if not isinstance(image, Mapping) or set(image) != {
            "height",
            "image_file",
            "image_file_sha256",
            "image_rgb_sha256",
            "width",
        }:
            raise ContractError("processor evidence report image contract changed")
        record_id = record.get("record_id")
        image_file = image.get("image_file")
        if not isinstance(record_id, str) or not isinstance(image_file, str):
            raise ContractError("processor evidence report image identity is malformed")
        expected_digest = _require_sha256(
            image.get("image_file_sha256"),
            name="report image file SHA-256",
        )
        relative = PurePosixPath(image_file)
        if (
            relative.is_absolute()
            or relative.as_posix() != image_file
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ContractError("processor evidence report image path is not canonical")
        size, digest = _sha256_file_content(
            artifact_root.joinpath(*relative.parts),
            description=f"processor evidence publication image {record_id}",
            ownership_root=artifact_root,
        )
        if digest != expected_digest:
            raise ContractError("public native VL artifact image changed before report write")
        rows.append([record_id, image_file, size, digest])
    return {
        "image_count": len(rows),
        "sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _read_json_contract(path: Path, *, description: str) -> tuple[dict[str, object], str]:
    if not path.is_absolute():
        raise ContractError(f"{description} path must be absolute")
    payload, digest = _read_regular_file(
        path,
        description=description,
        maximum_bytes=64 * 1024 * 1024,
    )
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{description} must be canonical ASCII JSON") from error
    if not isinstance(value, dict) or _canonical_bytes(value) + b"\n" != payload:
        raise ContractError(f"{description} must use canonical serialized JSON")
    return value, digest


def _validate_capture_report(
    report: Mapping[str, object],
    *,
    file_sha256: str,
) -> dict[str, object]:
    if (
        set(report) != CAPTURE_REPORT_FIELDS
        or report.get("schema") != CAPTURE_SCHEMA
        or report.get("status") != CAPTURE_STATUS
        or report.get("publication_authorized") is not False
    ):
        raise ContractError("authorization input is not an evidence-only CAPTURE report")
    artifact_sha256 = _require_sha256(
        report.get("artifact_sha256"),
        name="capture artifact SHA-256",
    )
    content = dict(report)
    content.pop("artifact_sha256", None)
    if hashlib.sha256(_canonical_bytes(content)).hexdigest() != artifact_sha256:
        raise ContractError("capture report artifact digest changed")
    _require_sha256(file_sha256, name="capture file SHA-256")
    _normalized_production_evidence(report)
    _validated_report_records(report)
    if not isinstance(report.get("source_contract"), Mapping):
        raise ContractError("capture report source contract is malformed")
    _validated_runtime_dependency_pair(
        processor_contract=report.get("processor_contract"),
        production_runtime=report.get("production_runtime"),
    )
    return dict(report)


def _validated_production_runtime(value: object) -> dict[str, object]:
    fields = {
        "cuda_visible_devices",
        "python_runtime_sha256",
        "torch_cuda_device_count",
        "torch_cuda_initialized",
        "torch_cuda_version",
        "torch_hip_version",
        "torch_mps_available",
        "torch_version",
        "torch_xpu_available",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError("processor production CPU runtime fields changed")
    if (
        value.get("cuda_visible_devices") != ""
        or value.get("torch_cuda_device_count") != 0
        or value.get("torch_cuda_initialized") is not False
        or value.get("torch_cuda_version") is not None
        or value.get("torch_hip_version") is not None
        or value.get("torch_mps_available") is not False
        or value.get("torch_xpu_available") is not False
    ):
        raise ContractError("processor production evidence is not from a CPU-only runtime")
    torch_version = value.get("torch_version")
    if not isinstance(torch_version, str) or torch_version.split("+", 1)[0] != "2.8.0":
        raise ContractError("processor production Torch version differs from 2.8.0 CPU")
    _require_sha256(value.get("python_runtime_sha256"), name="production Python runtime SHA-256")
    return dict(value)


def _validated_runtime_dependency_pair(
    *,
    processor_contract: object,
    production_runtime: object,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(processor_contract, Mapping):
        raise ContractError("processor runtime has no dependency environment")
    dependency_environment = _validated_dependency_environment(
        processor_contract.get("dependency_environment")
    )
    runtime = _validated_production_runtime(production_runtime)
    python_runtime = dependency_environment["python_runtime"]
    packages = dependency_environment["packages"]
    if not isinstance(python_runtime, Mapping) or not isinstance(packages, list):
        raise ContractError("processor dependency runtime identity is malformed")
    if runtime["python_runtime_sha256"] != python_runtime.get("sha256"):
        raise ContractError("processor Python runtime differs from its dependency environment")
    torch_packages = [
        package
        for package in packages
        if isinstance(package, Mapping) and package.get("distribution") == "torch"
    ]
    if len(torch_packages) != 1:
        raise ContractError("processor dependency environment has no unique Torch distribution")
    distribution_version = torch_packages[0].get("version")
    runtime_version = runtime["torch_version"]
    if (
        not isinstance(distribution_version, str)
        or not isinstance(runtime_version, str)
        or distribution_version.split("+", 1)[0] != runtime_version.split("+", 1)[0]
    ):
        raise ContractError("processor Torch import differs from its installed distribution")
    return runtime, dependency_environment


def _validate_approval_contract(
    approval: Mapping[str, object],
    *,
    capture: Mapping[str, object],
    capture_file_sha256: str,
    rationale_identity: Mapping[str, object],
) -> dict[str, object]:
    required_fields = {
        "authenticity",
        "capture_report",
        "decision",
        "dependency_environment",
        "production_evidence",
        "production_evidence_sha256",
        "review",
        "schema",
        "source_contract",
    }
    if set(approval) != required_fields or approval.get("schema") != APPROVAL_SCHEMA:
        raise ContractError("processor approval contract fields or schema changed")
    if approval.get("decision") != APPROVAL_DECISION:
        raise ContractError("processor approval contract is not an explicit approval")
    if approval.get("authenticity") != {"status": APPROVAL_AUTHENTICITY_STATUS}:
        raise ContractError("processor approval authenticity status changed")
    capture_identity = approval.get("capture_report")
    if not isinstance(capture_identity, Mapping) or set(capture_identity) != {
        "artifact_sha256",
        "file_sha256",
    }:
        raise ContractError("processor approval capture identity is malformed")
    if capture_identity != {
        "artifact_sha256": capture.get("artifact_sha256"),
        "file_sha256": capture_file_sha256,
    }:
        raise ContractError("processor approval does not bind the exact capture report")
    production_evidence = _normalized_production_evidence(capture)
    production_evidence_sha256 = hashlib.sha256(_canonical_bytes(production_evidence)).hexdigest()
    if (
        approval.get("production_evidence") != production_evidence
        or approval.get("production_evidence_sha256") != production_evidence_sha256
        or approval.get("source_contract") != capture.get("source_contract")
    ):
        raise ContractError("processor approval does not bind complete capture evidence")
    processor_contract = production_evidence.get("processor_contract")
    if not isinstance(processor_contract, Mapping):
        raise ContractError("capture processor contract is malformed")
    dependency_environment = processor_contract.get("dependency_environment")
    if approval.get("dependency_environment") != dependency_environment:
        raise ContractError("processor approval dependency environment differs from capture")
    review = approval.get("review")
    if not isinstance(review, Mapping) or set(review) != {
        "rationale_file_sha256",
        "rationale_size_bytes",
        "reviewed_at_utc",
        "reviewer",
    }:
        raise ContractError("processor approval review identity is malformed")
    if {
        "file_sha256": review.get("rationale_file_sha256"),
        "size_bytes": review.get("rationale_size_bytes"),
    } != rationale_identity:
        raise ContractError("processor approval does not bind the exact rationale file")
    for field in ("reviewed_at_utc", "reviewer"):
        value = review.get(field)
        if not isinstance(value, str) or not value or "\0" in value:
            raise ContractError(f"processor approval {field} is invalid")
    return dict(approval)


def _read_rationale_identity(path: Path) -> dict[str, object]:
    if not path.is_absolute():
        raise ContractError("processor approval rationale path must be absolute")
    payload, digest = _read_regular_file(
        path,
        description="processor approval rationale",
        maximum_bytes=MAXIMUM_RATIONALE_BYTES,
    )
    if not payload or not payload.endswith(b"\n") or b"\0" in payload:
        raise ContractError("processor approval rationale must be nonempty newline-terminated text")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("processor approval rationale must be UTF-8") from error
    if not text.strip():
        raise ContractError("processor approval rationale must contain substantive text")
    return {"file_sha256": digest, "size_bytes": len(payload)}


def _verify_approval_authenticity(_inputs: Mapping[str, object]) -> None:
    raise ContractError(
        "ADR-125 AUTHORIZE is fail-closed: no pinned approval signature trust root is configured"
    )


def _load_authorization_inputs(
    *,
    capture_report_path: Path,
    approval_contract_path: Path,
    rationale_file_path: Path,
) -> dict[str, object]:
    capture, capture_file_sha256 = _read_json_contract(
        capture_report_path,
        description="processor CAPTURE report",
    )
    capture = _validate_capture_report(capture, file_sha256=capture_file_sha256)
    approval, approval_file_sha256 = _read_json_contract(
        approval_contract_path,
        description="processor approval contract",
    )
    rationale_identity = _read_rationale_identity(rationale_file_path)
    approval = _validate_approval_contract(
        approval,
        capture=capture,
        capture_file_sha256=capture_file_sha256,
        rationale_identity=rationale_identity,
    )
    return {
        "approval": approval,
        "approval_contract_path": str(approval_contract_path),
        "approval_file_sha256": approval_file_sha256,
        "capture": capture,
        "capture_file_sha256": capture_file_sha256,
        "capture_report_path": str(capture_report_path),
        "rationale_file_path": str(rationale_file_path),
        "rationale_identity": rationale_identity,
    }


def _revalidate_authorization_inputs(inputs: Mapping[str, object]) -> None:
    capture_path = inputs.get("capture_report_path")
    approval_path = inputs.get("approval_contract_path")
    rationale_path = inputs.get("rationale_file_path")
    if (
        not isinstance(capture_path, str)
        or not isinstance(approval_path, str)
        or not isinstance(rationale_path, str)
    ):
        raise ContractError("authorization input paths are malformed")
    current = _load_authorization_inputs(
        capture_report_path=Path(capture_path),
        approval_contract_path=Path(approval_path),
        rationale_file_path=Path(rationale_path),
    )
    if current != inputs:
        raise ContractError("capture or approval contract changed during authorization replay")


def _authorization_report_binding(inputs: Mapping[str, object]) -> dict[str, object]:
    approval = inputs.get("approval")
    capture = inputs.get("capture")
    if not isinstance(approval, Mapping) or not isinstance(capture, Mapping):
        raise ContractError("authorized report inputs are malformed")
    return {
        "approval_contract_path": inputs.get("approval_contract_path"),
        "approval_file_sha256": inputs.get("approval_file_sha256"),
        "capture_artifact_sha256": capture.get("artifact_sha256"),
        "capture_file_sha256": inputs.get("capture_file_sha256"),
        "capture_report_path": inputs.get("capture_report_path"),
        "rationale_file_path": inputs.get("rationale_file_path"),
        "rationale_identity": inputs.get("rationale_identity"),
        "review": approval.get("review"),
    }


def _revalidate_report_external_state(report: Mapping[str, object]) -> None:
    captured_runtime, _ = _validated_runtime_dependency_pair(
        processor_contract=report.get("processor_contract"),
        production_runtime=report.get("production_runtime"),
    )
    path_contract = report.get("path_contract")
    if not isinstance(path_contract, Mapping) or set(path_contract) != {
        "artifact_root",
        "manifest_path",
        "processor_dir",
    }:
        raise ContractError("processor report path contract changed")
    path_values = [
        path_contract.get("artifact_root"),
        path_contract.get("manifest_path"),
        path_contract.get("processor_dir"),
    ]
    if any(not isinstance(value, str) or not value or "\0" in value for value in path_values):
        raise ContractError("processor report path contract is malformed")
    artifact_root, manifest_path, processor_dir = _resolve_production_paths(
        artifact_root=Path(str(path_values[0])),
        manifest_path=Path(str(path_values[1])),
        processor_dir=Path(str(path_values[2])),
    )
    if [str(artifact_root), str(manifest_path), str(processor_dir)] != path_values:
        raise ContractError("processor report path contract is not fully resolved")
    if _sha256_regular_file(manifest_path) != ADR125_PUBLIC_MANIFEST_FILE_SHA256:
        raise ContractError("ADR-125 manifest changed immediately before report write")

    processor_contract = report.get("processor_contract")
    if not isinstance(processor_contract, Mapping):
        raise ContractError("processor report dependency contract is malformed")
    current_processor = _processor_asset_contract_from_disk(processor_dir)
    expected_processor = {
        key: processor_contract.get(key)
        for key in (
            "processor_assets",
            "processor_dir",
            "processor_file_inventory",
            "processor_file_inventory_sha256",
            "processor_id",
            "processor_load_protocol",
            "processor_revision",
            "required_processor_files",
        )
    }
    if current_processor != expected_processor:
        raise ContractError("Qwen processor assets changed immediately before report write")
    captured_dependencies = _validated_dependency_environment(
        processor_contract.get("dependency_environment")
    )
    if _dependency_environment_contract() != captured_dependencies:
        raise ContractError("processor dependencies changed immediately before report write")
    snapshot = _artifact_image_snapshot_from_report(report, artifact_root=artifact_root)
    if report.get("artifact_image_snapshot") != snapshot:
        raise ContractError("artifact image aggregate changed immediately before report write")
    if _require_production_cpu_environment() != captured_runtime:
        raise ContractError("processor production CPU runtime changed before report write")


def _final_report(
    *,
    internal_content: Mapping[str, object],
    evidence_sha256: str,
    artifact_root: Path,
    manifest_path: Path,
    processor_dir: Path,
    production_runtime: Mapping[str, object],
    tool_sha256: str,
    authorized: bool,
    authorization_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    runtime, _ = _validated_runtime_dependency_pair(
        processor_contract=internal_content.get("processor_contract"),
        production_runtime=production_runtime,
    )
    content = dict(internal_content)
    content.update(
        {
            "internal_evidence_sha256": evidence_sha256,
            "path_contract": {
                "artifact_root": str(artifact_root),
                "manifest_path": str(manifest_path),
                "processor_dir": str(processor_dir),
            },
            "publication_authorized": authorized,
            "production_runtime": runtime,
            "schema": SCHEMA if authorized else CAPTURE_SCHEMA,
            "status": "PASS" if authorized else CAPTURE_STATUS,
            "tool_sha256": tool_sha256,
        }
    )
    if authorized:
        if authorization_inputs is None:
            raise ContractError("authorized report requires immutable authorization inputs")
        content["authorization_contract"] = _authorization_report_binding(authorization_inputs)
    elif authorization_inputs is not None:
        raise ContractError("CAPTURE report cannot consume an approval contract")
    return {
        **content,
        "artifact_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
    }


def _capture_adr125_report(
    *,
    internal: Mapping[str, object],
    manifest: PublicNativeVLRetentionManifest,
    artifact_root: Path,
    manifest_path: Path,
    processor_dir: Path,
    production_runtime: Mapping[str, object],
    tool_sha256: str,
) -> dict[str, object]:
    internal_content, evidence_sha256 = _validate_adr125_internal_report(
        internal=internal,
        manifest=manifest,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        tool_sha256=tool_sha256,
    )
    return _final_report(
        internal_content=internal_content,
        evidence_sha256=evidence_sha256,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        production_runtime=production_runtime,
        tool_sha256=tool_sha256,
        authorized=False,
    )


def _authorize_adr125_report(
    *,
    internal: Mapping[str, object],
    manifest: PublicNativeVLRetentionManifest,
    artifact_root: Path,
    manifest_path: Path,
    processor_dir: Path,
    production_runtime: Mapping[str, object],
    tool_sha256: str,
    authorization_inputs: Mapping[str, object],
) -> dict[str, object]:
    _verify_approval_authenticity(authorization_inputs)
    internal_content, evidence_sha256 = _validate_adr125_internal_report(
        internal=internal,
        manifest=manifest,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        tool_sha256=tool_sha256,
    )
    _revalidate_authorization_inputs(authorization_inputs)
    capture = authorization_inputs.get("capture")
    if not isinstance(capture, Mapping):
        raise ContractError("authorization capture input is malformed")
    if _normalized_production_evidence(internal) != _normalized_production_evidence(capture):
        raise ContractError("current 192-record replay differs from the approved capture")
    if internal.get("source_contract") != capture.get("source_contract"):
        raise ContractError("current executable source differs from the approved capture")
    if production_runtime != capture.get("production_runtime"):
        raise ContractError("current CPU runtime contract differs from the approved capture")
    replayed_capture = _final_report(
        internal_content=internal_content,
        evidence_sha256=evidence_sha256,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        production_runtime=production_runtime,
        tool_sha256=tool_sha256,
        authorized=False,
    )
    if replayed_capture != capture:
        raise ContractError("current replay does not reproduce the complete approved capture")

    return _final_report(
        internal_content=internal_content,
        evidence_sha256=evidence_sha256,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        processor_dir=processor_dir,
        production_runtime=production_runtime,
        tool_sha256=tool_sha256,
        authorized=True,
        authorization_inputs=authorization_inputs,
    )


def _publish_report(output: Path, report: Mapping[str, object]) -> dict[str, object]:
    status = report.get("status")
    authorized = report.get("publication_authorized")
    if not (
        (report.get("schema") == SCHEMA and status == "PASS" and authorized is True)
        or (
            report.get("schema") == CAPTURE_SCHEMA
            and status == CAPTURE_STATUS
            and authorized is False
        )
    ):
        raise ContractError("only an ADR-125 CAPTURE or authorized PASS report may be published")
    expected_fields = (
        CAPTURE_REPORT_FIELDS | {"authorization_contract"}
        if status == "PASS"
        else CAPTURE_REPORT_FIELDS
    )
    if set(report) != expected_fields:
        raise ContractError("processor evidence report top-level fields changed")
    authorization_contract = report.get("authorization_contract")
    if status == "PASS":
        if not isinstance(authorization_contract, Mapping) or set(authorization_contract) != {
            "approval_contract_path",
            "approval_file_sha256",
            "capture_artifact_sha256",
            "capture_file_sha256",
            "capture_report_path",
            "rationale_file_path",
            "rationale_identity",
            "review",
        }:
            raise ContractError("authorized PASS report has no complete approval binding")
        for field in (
            "approval_file_sha256",
            "capture_artifact_sha256",
            "capture_file_sha256",
        ):
            _require_sha256(authorization_contract.get(field), name=field)
    elif authorization_contract is not None:
        raise ContractError("CAPTURE report must not contain an authorization binding")
    artifact_sha256 = _require_sha256(
        report.get("artifact_sha256"),
        name="processor evidence report artifact SHA-256",
    )
    content = dict(report)
    content.pop("artifact_sha256", None)
    if hashlib.sha256(_canonical_bytes(content)).hexdigest() != artifact_sha256:
        raise ContractError("ADR-125 processor evidence report artifact digest changed")
    _normalized_production_evidence(report)
    _validated_report_records(report)
    _revalidate_report_external_state(report)
    if status == "PASS":
        if not isinstance(authorization_contract, Mapping):
            raise ContractError("authorized PASS report has no approval contract")
        capture_path = authorization_contract.get("capture_report_path")
        approval_path = authorization_contract.get("approval_contract_path")
        rationale_path = authorization_contract.get("rationale_file_path")
        if (
            not isinstance(capture_path, str)
            or not isinstance(approval_path, str)
            or not isinstance(rationale_path, str)
        ):
            raise ContractError("authorized PASS approval paths are malformed")
        current_inputs = _load_authorization_inputs(
            capture_report_path=Path(capture_path),
            approval_contract_path=Path(approval_path),
            rationale_file_path=Path(rationale_path),
        )
        _verify_approval_authenticity(current_inputs)
        if authorization_contract != _authorization_report_binding(current_inputs):
            raise ContractError("authorized PASS binding differs from immutable approval inputs")
        capture = current_inputs.get("capture")
        if not isinstance(capture, Mapping):
            raise ContractError("authorized PASS capture input is malformed")
        if (
            _normalized_production_evidence(report) != _normalized_production_evidence(capture)
            or report.get("source_contract") != capture.get("source_contract")
            or report.get("production_runtime") != capture.get("production_runtime")
        ):
            raise ContractError("authorized PASS differs from its reviewed capture")
        capture_projection = dict(report)
        capture_projection.pop("artifact_sha256", None)
        capture_projection.pop("authorization_contract", None)
        capture_projection.update(
            {
                "publication_authorized": False,
                "schema": CAPTURE_SCHEMA,
                "status": CAPTURE_STATUS,
            }
        )
        projected_capture = {
            **capture_projection,
            "artifact_sha256": hashlib.sha256(_canonical_bytes(capture_projection)).hexdigest(),
        }
        if projected_capture != capture:
            raise ContractError("authorized PASS is not an exact transform of its reviewed capture")
    current_tool_sha256 = _sha256_file_content(
        Path(__file__).resolve(strict=True),
        description="processor evidence tool",
        ownership_root=Path(_REPOSITORY_ROOT).resolve(strict=True),
    )[1]
    if report.get("tool_sha256") != current_tool_sha256:
        raise ContractError("ADR-125 authorized report tool digest changed")
    if report.get("source_contract") != _repository_source_contract():
        raise ContractError("ADR-125 executable source changed before report write")
    payload = _canonical_bytes(dict(report)) + b"\n"
    write_bytes_durable_exclusive(output, payload)
    return {
        "artifact_sha256": report["artifact_sha256"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "output": str(output.expanduser().absolute()),
        "production_evidence": report["production_evidence"],
        "record_count": report["summary"]["record_count"],  # type: ignore[index]
        "status": report["status"],
    }


def _require_production_cpu_environment() -> dict[str, object]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != "":
        raise ContractError("processor production evidence requires CUDA_VISIBLE_DEVICES empty")
    device_count = torch.cuda.device_count()
    if isinstance(device_count, bool) or device_count != 0:
        raise ContractError("processor production evidence requires torch CUDA device_count 0")
    cuda_initialized = torch.cuda.is_initialized()
    torch_version = getattr(torch, "version", None)
    cuda_version = getattr(torch_version, "cuda", None)
    hip_version = getattr(torch_version, "hip", None)
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    xpu_backend = getattr(torch, "xpu", None)
    xpu_available = bool(xpu_backend is not None and xpu_backend.is_available())
    if (
        cuda_initialized
        or cuda_version is not None
        or hip_version is not None
        or mps_available
        or xpu_available
    ):
        raise ContractError(
            "processor production evidence requires a CPU-only Torch build "
            "with no accelerator state"
        )
    runtime = _python_runtime_contract()
    result = {
        "cuda_visible_devices": visible_devices,
        "python_runtime_sha256": runtime["sha256"],
        "torch_cuda_device_count": device_count,
        "torch_cuda_initialized": cuda_initialized,
        "torch_cuda_version": cuda_version,
        "torch_hip_version": hip_version,
        "torch_mps_available": mps_available,
        "torch_version": str(torch.__version__),
        "torch_xpu_available": xpu_available,
    }
    return _validated_production_runtime(result)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("capture", "authorize"), required=True)
    parser.add_argument("--capture-report", type=Path)
    parser.add_argument("--approval-contract", type=Path)
    parser.add_argument("--rationale-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    production_runtime = _require_production_cpu_environment()
    if args.manifest_sha256 != ADR125_PUBLIC_MANIFEST_FILE_SHA256:
        raise ContractError("ADR-125 processor audit manifest SHA-256 changed")
    if args.mode == "capture":
        if (
            args.capture_report is not None
            or args.approval_contract is not None
            or args.rationale_file is not None
        ):
            raise ContractError("CAPTURE mode cannot consume prior evidence or approval")
        authorization_inputs = None
    else:
        if (
            args.capture_report is None
            or args.approval_contract is None
            or args.rationale_file is None
        ):
            raise ContractError(
                "AUTHORIZE mode requires capture report, approval contract, and rationale file"
            )
        authorization_inputs = _load_authorization_inputs(
            capture_report_path=args.capture_report,
            approval_contract_path=args.approval_contract,
            rationale_file_path=args.rationale_file,
        )
        _verify_approval_authenticity(authorization_inputs)
    tool_sha256 = _sha256_file_content(
        Path(__file__).resolve(strict=True),
        description="processor evidence tool",
        ownership_root=Path(_REPOSITORY_ROOT).resolve(strict=True),
    )[1]
    artifact_root, manifest_path, processor_dir = _resolve_production_paths(
        artifact_root=args.root,
        manifest_path=args.manifest,
        processor_dir=args.processor,
    )
    manifest = load_frozen_public_native_vl_retention_gate(
        manifest_path=manifest_path,
        manifest_file_sha256=ADR125_PUBLIC_MANIFEST_FILE_SHA256,
        artifact_root=artifact_root,
        max_steps=PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY,
    )
    if manifest.artifact_sha256 != ADR125_PUBLIC_ARTIFACT_SHA256:
        raise ContractError("ADR-125 public artifact SHA-256 changed")
    with _load_pinned_processor(processor_dir) as (processor, processor_identity):
        internal = build_public_native_vl_processor_budget_audit(
            manifest=manifest,
            artifact_root=artifact_root,
            manifest_file_sha256=ADR125_PUBLIC_MANIFEST_FILE_SHA256,
            processor=processor,
            processor_identity=processor_identity,
            processor_dir=processor_dir,
        )
        if args.mode == "capture":
            report = _capture_adr125_report(
                internal=internal,
                manifest=manifest,
                artifact_root=artifact_root,
                manifest_path=manifest_path,
                processor_dir=processor_dir,
                production_runtime=production_runtime,
                tool_sha256=tool_sha256,
            )
        else:
            if authorization_inputs is None:
                raise ContractError("AUTHORIZE mode lost validated inputs")
            report = _authorize_adr125_report(
                internal=internal,
                manifest=manifest,
                artifact_root=artifact_root,
                manifest_path=manifest_path,
                processor_dir=processor_dir,
                production_runtime=production_runtime,
                tool_sha256=tool_sha256,
                authorization_inputs=authorization_inputs,
            )
    print(json.dumps(_publish_report(args.output, report), sort_keys=True))


if __name__ == "__main__":
    main()

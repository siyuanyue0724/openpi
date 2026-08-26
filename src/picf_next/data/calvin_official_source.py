"""Contracts for rebinding verified CALVIN artifacts to the official source identity."""

from __future__ import annotations

from collections.abc import Mapping

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import DatasetFileManifest

CALVIN_OFFICIAL_DATASET_ID = "calvin.cs.uni-freiburg.de/task_ABC_D"
CALVIN_OFFICIAL_SOURCE_RECEIPT_SCHEMA = "picf-next.calvin-official-source-receipt.v1"
CALVIN_OFFICIAL_MANIFEST_NAME = "calvin-training-files.json"
CALVIN_OFFICIAL_SOURCE_VERIFICATION_MODE = "full-local-sha256-and-official-zip-crc32.v1"
CALVIN_OFFICIAL_ARCHIVE_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
CALVIN_OFFICIAL_ARCHIVE_CONTENT_LENGTH = 555_309_812_705
CALVIN_OFFICIAL_ARCHIVE_LAST_MODIFIED = "Thu, 15 Sep 2022 17:47:47 GMT"
CALVIN_OFFICIAL_ARCHIVE_ETAG = '"814b0b4be1-5e8bad7e824c0"'
CALVIN_OFFICIAL_ARCHIVE_TAIL_SIZE_BYTES = 262_144
CALVIN_OFFICIAL_ARCHIVE_TAIL_SHA256 = (
    "7a2d372d53fa4f7be52f784be249bd853dbf4a54d8a76a38efc9d32239aebdf9"
)
CALVIN_OFFICIAL_CENTRAL_DIRECTORY_OFFSET = 555_080_601_096
CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SIZE = 229_211_511
CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SHA256 = (
    "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
)
CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT = 1_894_126
CALVIN_OFFICIAL_TRAINING_PREFIX = "task_ABC_D/training/"
CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES = (
    ".hydra/config.yaml",
    ".hydra/hydra.yaml",
    ".hydra/overrides.yaml",
    "lang_BERT/auto_lang_ann.npy",
    "lang_all-distilroberta-v1/auto_lang_ann.npy",
    "lang_all-mpnet-base-v2/auto_lang_ann.npy",
    "lang_clip_resnet50/auto_lang_ann.npy",
    "lang_huggingface_distilroberta/auto_lang_ann.npy",
    "lang_huggingface_mpnet/auto_lang_ann.npy",
    "lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy",
    "statistics.yaml",
)
CALVIN_OFFICIAL_PUBLISHER_AUTHENTICITY = (
    "official-http-endpoint-metadata-and-zip-crc32;no-publisher-signature"
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def _same_http_date(value: object, expected: str) -> bool:
    return isinstance(value, str) and value in {expected, expected.replace(",", "")}


def validate_calvin_content_identity_migration(
    source: DatasetFileManifest,
    target: DatasetFileManifest,
) -> None:
    """Require an identity-only migration over exactly the same selected bytes."""

    if not isinstance(source, DatasetFileManifest) or not isinstance(target, DatasetFileManifest):
        raise TypeError("CALVIN identity migration requires dataset manifests")
    if source.split_name != target.split_name or source.files != target.files:
        raise ContractError("CALVIN identity migration changed the selected source bytes")
    if source.content_sha256 != target.content_sha256:
        raise ContractError("CALVIN identity migration changed the content digest")
    if target.dataset_id != CALVIN_OFFICIAL_DATASET_ID:
        raise ContractError("CALVIN target manifest does not use the official source identity")
    if target.dataset_revision != f"sha256:{target.content_sha256}":
        raise ContractError("CALVIN target manifest revision is not content-derived")


def validate_calvin_official_source_receipt(
    receipt: Mapping[str, object],
    *,
    source_manifest: DatasetFileManifest,
    source_manifest_sha256: str,
    target_manifest: DatasetFileManifest,
    target_manifest_sha256: str,
) -> None:
    """Validate the exact source-receipt claims required by derived artifacts."""

    if receipt.get("schema") != CALVIN_OFFICIAL_SOURCE_RECEIPT_SCHEMA:
        raise ContractError("CALVIN source receipt schema changed")
    archive = _mapping(receipt.get("official_archive"), "source receipt official archive")
    if (
        archive.get("url") != CALVIN_OFFICIAL_ARCHIVE_URL
        or archive.get("transport") != "http"
        or archive.get("content_length") != CALVIN_OFFICIAL_ARCHIVE_CONTENT_LENGTH
        or not _same_http_date(archive.get("last_modified"), CALVIN_OFFICIAL_ARCHIVE_LAST_MODIFIED)
        or archive.get("etag") != CALVIN_OFFICIAL_ARCHIVE_ETAG
        or archive.get("tail_size_bytes") != CALVIN_OFFICIAL_ARCHIVE_TAIL_SIZE_BYTES
        or archive.get("tail_sha256") != CALVIN_OFFICIAL_ARCHIVE_TAIL_SHA256
        or archive.get("central_directory_offset") != CALVIN_OFFICIAL_CENTRAL_DIRECTORY_OFFSET
        or archive.get("central_directory_size") != CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SIZE
        or archive.get("central_directory_sha256") != CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SHA256
        or archive.get("entry_count") != CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT
        or archive.get("zip64") is not True
        or archive.get("publisher_authenticity") != CALVIN_OFFICIAL_PUBLISHER_AUTHENTICITY
    ):
        raise ContractError("CALVIN source receipt official archive binding differs")
    inventory = _mapping(
        receipt.get("official_training_inventory"),
        "source receipt official training inventory",
    )
    if (
        inventory.get("archive_prefix") != CALVIN_OFFICIAL_TRAINING_PREFIX
        or inventory.get("archive_entry_count") != CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT
        or inventory.get("file_count")
        != len(target_manifest.files) + len(CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES)
        or inventory.get("excluded_non_runtime_files")
        != list(CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES)
    ):
        raise ContractError("CALVIN source receipt official training inventory differs")
    source = _mapping(receipt.get("source_manifest"), "source receipt manifest")
    migrated = _mapping(receipt.get("migrated_manifest"), "source migrated manifest")
    verified = _mapping(receipt.get("verified_content"), "source verified content")
    if (
        source.get("file_sha256") != source_manifest_sha256
        or source.get("tree_sha256") != source_manifest.tree_sha256
        or source.get("declared_dataset_id") != source_manifest.dataset_id
        or source.get("declared_dataset_revision") != source_manifest.dataset_revision
        or migrated.get("file_name") != CALVIN_OFFICIAL_MANIFEST_NAME
        or migrated.get("file_sha256") != target_manifest_sha256
        or migrated.get("tree_sha256") != target_manifest.tree_sha256
    ):
        raise ContractError("CALVIN source receipt manifest bindings differ")
    if (
        verified.get("dataset_id") != target_manifest.dataset_id
        or verified.get("dataset_revision") != target_manifest.dataset_revision
        or verified.get("content_sha256") != target_manifest.content_sha256
        or verified.get("split_name") != target_manifest.split_name
        or verified.get("file_count") != len(target_manifest.files)
        or verified.get("total_size_bytes") != target_manifest.total_size_bytes
        or verified.get("verification_mode") != CALVIN_OFFICIAL_SOURCE_VERIFICATION_MODE
        or verified.get("all_manifest_sha256_matches") is not True
        or verified.get("all_official_crc32_matches") is not True
        or verified.get("official_inventory_exact_after_declared_exclusions") is not True
    ):
        raise ContractError("CALVIN source receipt has not closed content verification")
    if receipt.get("training_authorized") is not False:
        raise ContractError("CALVIN source receipt must not authorize model training")

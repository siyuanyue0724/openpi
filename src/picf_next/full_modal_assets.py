"""Fail-closed asset identities for the production full-modal path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from picf_next.contracts import ContractError

FULL_MODAL_ASSET_SCHEMA = "picf-next.full-modal-assets.v1"
PRODUCTION_MODALITIES = ("anytouch", "sonata", "vjepa")
PRODUCTION_ROLES = {
    "anytouch": "frozen_dense_tactile_encoder",
    "sonata": "frozen_dense_geometry_encoder",
    "vjepa": "frozen_causal_temporal_encoder",
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name)
    if set(result) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _commit(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one full lowercase Git commit")
    return result


def _absolute_path(value: object, name: str) -> Path:
    result = _text(value, name)
    pure = PurePosixPath(result)
    if not pure.is_absolute() or "\\" in result or "\0" in result or ".." in pure.parts:
        raise ContractError(f"{name} must be a normalized absolute POSIX path")
    return Path(result)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FullModalAsset:
    modality: str
    model: str
    persistent_path: Path
    legacy_source_path: Path
    size_bytes: int
    sha256: str
    upstream_url: str
    upstream_commit: str
    production_role: str
    architecture_upstream_url: str | None = None
    architecture_upstream_commit: str | None = None

    def verify_file(self) -> None:
        if not self.persistent_path.is_file():
            raise ContractError(f"missing production {self.modality} checkpoint")
        observed_size = self.persistent_path.stat().st_size
        if observed_size != self.size_bytes:
            raise ContractError(
                f"production {self.modality} checkpoint size changed: "
                f"expected={self.size_bytes} observed={observed_size}"
            )
        observed_sha256 = sha256_file(self.persistent_path)
        if observed_sha256 != self.sha256:
            raise ContractError(f"production {self.modality} checkpoint SHA-256 changed")


@dataclass(frozen=True, slots=True)
class FullModalAssetManifest:
    path: Path
    assets: tuple[FullModalAsset, ...]

    @classmethod
    def load(cls, path: str | Path, *, verify_files: bool = False) -> FullModalAssetManifest:
        manifest_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("full-modal asset manifest is not readable canonical JSON") from exc
        root = _exact(
            payload,
            "full-modal asset manifest",
            {
                "assets",
                "created_at",
                "legacy_code",
                "legacy_source",
                "quarantined",
                "schema",
                "selection_policy",
            },
        )
        if root["schema"] != FULL_MODAL_ASSET_SCHEMA:
            raise ContractError("full-modal asset manifest schema changed")
        _text(root["created_at"], "manifest creation time")
        policy = _exact(
            root["selection_policy"],
            "asset selection policy",
            {
                "allow_directory_scan",
                "allow_unlisted_checkpoint",
                "require_exact_sha256",
                "require_strict_state_dict_load",
            },
        )
        if policy != {
            "allow_directory_scan": False,
            "allow_unlisted_checkpoint": False,
            "require_exact_sha256": True,
            "require_strict_state_dict_load": True,
        }:
            raise ContractError("full-modal asset selection policy is not fail-closed")
        cls._validate_legacy_source(root["legacy_source"])
        cls._validate_legacy_code(root["legacy_code"])
        cls._validate_quarantine(root["quarantined"])
        raw_assets = root["assets"]
        if not isinstance(raw_assets, list):
            raise ContractError("full-modal assets must be a list")
        assets = tuple(
            cls._parse_asset(value, index=index) for index, value in enumerate(raw_assets)
        )
        modalities = tuple(asset.modality for asset in assets)
        if modalities != PRODUCTION_MODALITIES:
            raise ContractError("full-modal assets must contain the exact canonical modality order")
        persistent_paths = tuple(asset.persistent_path for asset in assets)
        if len(set(persistent_paths)) != len(persistent_paths):
            raise ContractError("full-modal assets contain duplicate persistent paths")
        manifest = cls(path=manifest_path, assets=assets)
        if verify_files:
            for asset in manifest.assets:
                asset.verify_file()
        return manifest

    @staticmethod
    def _parse_asset(value: object, *, index: int) -> FullModalAsset:
        mapping = _mapping(value, f"asset[{index}]")
        common = {
            "legacy_source_path",
            "modality",
            "model",
            "persistent_path",
            "production_role",
            "sha256",
            "size_bytes",
            "upstream_commit",
            "upstream_url",
        }
        spatial = common | {"architecture_upstream_commit", "architecture_upstream_url"}
        if set(mapping) not in (common, spatial):
            raise ContractError(f"asset[{index}] fields differ from the frozen schema")
        modality = _text(mapping["modality"], f"asset[{index}] modality")
        if modality not in PRODUCTION_ROLES:
            raise ContractError(f"asset[{index}] has an unapproved modality")
        role = _text(mapping["production_role"], f"asset[{index}] production role")
        if role != PRODUCTION_ROLES[modality]:
            raise ContractError(f"asset[{index}] production role changed")
        size = mapping["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ContractError(f"asset[{index}] size must be a positive integer")
        architecture_url = None
        architecture_commit = None
        if set(mapping) == spatial:
            if modality != "sonata":
                raise ContractError("only the SpatialLM/Sonata asset has an architecture source")
            architecture_url = _text(
                mapping["architecture_upstream_url"], "Sonata architecture upstream URL"
            )
            architecture_commit = _commit(
                mapping["architecture_upstream_commit"], "Sonata architecture upstream commit"
            )
        elif modality == "sonata":
            raise ContractError("SpatialLM/Sonata asset lost its architecture source")
        return FullModalAsset(
            modality=modality,
            model=_text(mapping["model"], f"asset[{index}] model"),
            persistent_path=_absolute_path(
                mapping["persistent_path"], f"asset[{index}] persistent path"
            ),
            legacy_source_path=_absolute_path(
                mapping["legacy_source_path"], f"asset[{index}] legacy source path"
            ),
            size_bytes=size,
            sha256=_sha256(mapping["sha256"], f"asset[{index}] SHA-256"),
            upstream_url=_text(mapping["upstream_url"], f"asset[{index}] upstream URL"),
            upstream_commit=_commit(mapping["upstream_commit"], f"asset[{index}] upstream commit"),
            production_role=role,
            architecture_upstream_url=architecture_url,
            architecture_upstream_commit=architecture_commit,
        )

    @staticmethod
    def _validate_legacy_source(value: object) -> None:
        source = _exact(
            value,
            "legacy source",
            {"dirty_worktree_read_only", "git_commit", "worktree"},
        )
        _absolute_path(source["worktree"], "legacy source worktree")
        _commit(source["git_commit"], "legacy source commit")
        if source["dirty_worktree_read_only"] is not True:
            raise ContractError("legacy dirty worktree must remain read-only")

    @staticmethod
    def _validate_legacy_code(value: object) -> None:
        code = _exact(value, "legacy code", {"anytouch", "sonata", "vjepa"})
        for modality in PRODUCTION_MODALITIES:
            item = _mapping(code[modality], f"legacy {modality} code")
            files = item.get("files")
            if not isinstance(files, Mapping) or not files:
                raise ContractError(f"legacy {modality} code file hashes are missing")
            for source_path, digest in files.items():
                if not isinstance(source_path, str) or not source_path:
                    raise ContractError(f"legacy {modality} source path is invalid")
                _sha256(digest, f"legacy {modality} source SHA-256")
            for key, digest in item.items():
                if key != "files":
                    _sha256(digest, f"legacy {modality} {key}")

    @staticmethod
    def _validate_quarantine(value: object) -> None:
        if not isinstance(value, list) or not value:
            raise ContractError("full-modal quarantine list must be nonempty")
        paths: list[Path] = []
        for index, item in enumerate(value):
            record = _exact(item, f"quarantine[{index}]", {"path", "reason"})
            paths.append(_absolute_path(record["path"], f"quarantine[{index}] path"))
            _text(record["reason"], f"quarantine[{index}] reason")
        if len(set(paths)) != len(paths):
            raise ContractError("full-modal quarantine contains duplicate paths")

    def asset(self, modality: str) -> FullModalAsset:
        if modality not in PRODUCTION_MODALITIES:
            raise ContractError(f"unknown production modality {modality!r}")
        return self.assets[PRODUCTION_MODALITIES.index(modality)]

"""Atomic Accelerate resume state for Stage B, deliberately without posterior state."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from picf_next.training.accelerate_runner import distributed_main_process_call
from picf_next.training.stage_checkpoints import sha256_file

STATIONARY_ACCELERATE_IDENTITY_SCHEMA = "picf-next.stationary-accelerate-identity.v1"
STATIONARY_ACCELERATE_CONTROL_SCHEMA = "picf-next.stationary-accelerate-control.v1"


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _git_sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("stationary checkpoint code revision must be one Git SHA")
    return value


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class StationaryAccelerateCheckpointIdentity:
    stage_recipe_sha256: str
    source_coverage_recipe_sha256: str
    foundation_recipe_sha256: str
    m2_checkpoint_sha256: str
    feature_cache_manifest_sha256: str
    dataset_manifest_sha256: str
    physical_sidecar_manifest_sha256: str
    clip_plan_sha256: str
    code_revision: str
    world_size: int
    total_steps: int
    recurrent_state_serialized: bool = False
    schema: str = STATIONARY_ACCELERATE_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATIONARY_ACCELERATE_IDENTITY_SCHEMA:
            raise ValueError("stationary Accelerate identity schema changed")
        for name in (
            "stage_recipe_sha256",
            "source_coverage_recipe_sha256",
            "foundation_recipe_sha256",
            "m2_checkpoint_sha256",
            "feature_cache_manifest_sha256",
            "dataset_manifest_sha256",
            "physical_sidecar_manifest_sha256",
            "clip_plan_sha256",
        ):
            _sha256(getattr(self, name), name)
        _git_sha(self.code_revision)
        _positive_integer(self.world_size, "stationary checkpoint world size")
        _positive_integer(self.total_steps, "stationary checkpoint total steps")
        if self.recurrent_state_serialized is not False:
            raise ValueError("stationary Accelerate state cannot serialize recurrent state")

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stage_recipe_sha256": self.stage_recipe_sha256,
            "source_coverage_recipe_sha256": self.source_coverage_recipe_sha256,
            "foundation_recipe_sha256": self.foundation_recipe_sha256,
            "m2_checkpoint_sha256": self.m2_checkpoint_sha256,
            "feature_cache_manifest_sha256": self.feature_cache_manifest_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "physical_sidecar_manifest_sha256": self.physical_sidecar_manifest_sha256,
            "clip_plan_sha256": self.clip_plan_sha256,
            "code_revision": self.code_revision,
            "world_size": self.world_size,
            "total_steps": self.total_steps,
            "recurrent_state_serialized": self.recurrent_state_serialized,
        }


def _state_inventory(root: Path, control_name: str) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == control_name:
            continue
        if path.is_symlink():
            raise ValueError(f"stationary checkpoint contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"stationary checkpoint contains a non-regular file: {relative}")
        if relative.startswith("picf_rank_state_"):
            raise ValueError("stationary checkpoint attempted to serialize posterior rank state")
        inventory[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not inventory:
        raise ValueError("stationary checkpoint contains no Accelerate state")
    return inventory


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_control(
    root: Path,
    *,
    identity: StationaryAccelerateCheckpointIdentity,
    completed_steps: int,
) -> None:
    completed_steps = _positive_integer(completed_steps, "completed stationary steps")
    if completed_steps > identity.total_steps:
        raise ValueError("stationary checkpoint progress exceeds its bounded stage")
    control = root / "stationary_control.json"
    if control.exists() or control.is_symlink():
        raise FileExistsError(control)
    payload = {
        "schema": STATIONARY_ACCELERATE_CONTROL_SCHEMA,
        "identity": identity.to_dict(),
        "identity_sha256": identity.identity_sha256,
        "completed_steps": completed_steps,
        "recurrent_state_serialized": False,
        "state_files": _state_inventory(root, control.name),
    }
    with control.open("xb") as stream:
        stream.write(_canonical_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_stationary_accelerate_checkpoint(
    checkpoint_dir: str | Path,
    *,
    identity: StationaryAccelerateCheckpointIdentity,
) -> int:
    root = Path(checkpoint_dir).resolve()
    control = root / "stationary_control.json"
    try:
        payload = json.loads(control.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("stationary checkpoint control is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "identity",
        "identity_sha256",
        "completed_steps",
        "recurrent_state_serialized",
        "state_files",
    }:
        raise ValueError("stationary checkpoint control fields changed")
    if (
        payload["schema"] != STATIONARY_ACCELERATE_CONTROL_SCHEMA
        or payload["identity"] != identity.to_dict()
        or payload["identity_sha256"] != identity.identity_sha256
        or payload["recurrent_state_serialized"] is not False
    ):
        raise ValueError("stationary checkpoint identity differs from the active run")
    completed = _positive_integer(payload["completed_steps"], "completed stationary steps")
    if completed > identity.total_steps:
        raise ValueError("stationary checkpoint progress exceeds the active run")
    expected_inventory = payload["state_files"]
    if not isinstance(expected_inventory, dict) or not expected_inventory:
        raise ValueError("stationary checkpoint control has no state inventory")
    if _state_inventory(root, control.name) != expected_inventory:
        raise ValueError("stationary checkpoint state files are missing, added, or corrupt")
    return completed


def save_stationary_accelerate_checkpoint(
    *,
    accelerator: Any,
    checkpoint_dir: str | Path,
    identity: StationaryAccelerateCheckpointIdentity,
    completed_steps: int,
) -> Path:
    if int(accelerator.num_processes) != identity.world_size:
        raise ValueError("stationary checkpoint world size differs from Accelerator")
    if not bool(accelerator.sync_gradients):
        raise RuntimeError("stationary checkpoint requires an optimizer boundary")
    final = Path(checkpoint_dir).resolve()
    staging = final.with_name(f".{final.name}.incomplete")
    accelerator.wait_for_everyone()

    def create_staging() -> None:
        if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
            raise FileExistsError(final)
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()

    distributed_main_process_call(
        accelerator,
        label="stationary checkpoint staging",
        action=create_staging,
    )
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=str(staging), safe_serialization=True)
    accelerator.wait_for_everyone()

    def publish() -> None:
        _write_control(staging, identity=identity, completed_steps=completed_steps)
        _fsync_tree(staging)
        os.replace(staging, final)
        parent_descriptor = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    distributed_main_process_call(
        accelerator,
        label="stationary checkpoint publication",
        action=publish,
    )
    accelerator.wait_for_everyone()
    return final


def load_stationary_accelerate_checkpoint(
    *,
    accelerator: Any,
    checkpoint_dir: str | Path,
    identity: StationaryAccelerateCheckpointIdentity,
) -> int:
    if int(accelerator.num_processes) != identity.world_size:
        raise ValueError("stationary resume world size differs from Accelerator")
    completed = distributed_main_process_call(
        accelerator,
        label="stationary checkpoint validation",
        action=lambda: validate_stationary_accelerate_checkpoint(
            checkpoint_dir,
            identity=identity,
        ),
    )
    accelerator.wait_for_everyone()
    accelerator.load_state(str(Path(checkpoint_dir).resolve()))
    accelerator.wait_for_everyone()
    return int(completed)

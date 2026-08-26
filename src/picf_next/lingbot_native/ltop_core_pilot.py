"""Frozen contracts for the post-G3 LTOP exact-cache core pilot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from picf_next.artifact_io import directory_tree_sha256
from picf_next.lingbot_native.ltop_g3_mediator_acceptance import (
    ACTION_INFORMATION_SET_POLICY as MEDIATOR_ACTION_INFORMATION_SET_POLICY,
)
from picf_next.lingbot_native.ltop_g3_mediator_acceptance import (
    MEDIATOR_ACCEPTANCE_SCHEMA,
    MODEL_ONLY_CHECKPOINT_FORMAT,
    MODEL_TREE_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA,
    compose_ltop_g3_mediator_acceptance,
)
from picf_next.lingbot_native.ltop_g3_source_aligned_acceptance import (
    ACTION_SUPERVISION_SCHEMA as SOURCE_ALIGNED_ACTION_SUPERVISION_SCHEMA,
)
from picf_next.lingbot_native.ltop_g3_source_aligned_acceptance import (
    EXPECTED_ACTION_SUPERVISION as SOURCE_ALIGNED_ACTION_SUPERVISION,
)
from picf_next.lingbot_native.ltop_g3_source_aligned_acceptance import (
    SOURCE_ACTION_SCHEDULE_SCHEMA,
    SOURCE_ALIGNED_ACCEPTANCE_SCHEMA,
    compose_ltop_g3_source_aligned_acceptance,
)
from picf_next.lingbot_native.ltop_g3_source_aligned_acceptance import (
    TRAINING_CHECKPOINT_SCHEMA as SOURCE_ALIGNED_TRAINING_CHECKPOINT_SCHEMA,
)

LTOP_CORE_PILOT_SCHEMA: Final = "picf-next.ltop-core-pilot.v1"
LTOP_CORE_PILOT_G3_SCHEMA: Final = "picf-next.ltop-g3-production-action-mediation.v1"
LTOP_CORE_PILOT_WORLD_SIZE: Final = 2
LTOP_CORE_PILOT_TOTAL_STEPS: Final = 2_000
LTOP_CORE_PILOT_METRICS_EVERY: Final = 100
LTOP_CORE_PILOT_DIAGNOSTICS_EVERY: Final = 250
LTOP_CORE_PILOT_CHECKPOINT_STEP: Final = 2_000
LTOP_CORE_LONG_TOTAL_STEPS: Final = 30_000
LTOP_CORE_LONG_METRICS_EVERY: Final = 100
LTOP_CORE_LONG_DIAGNOSTICS_EVERY: Final = 250
LTOP_CORE_LONG_CHECKPOINT_EVERY: Final = 2_000
LTOP_CORE_PILOT_MODES: Final = ("smoke", "restart-smoke", "pilot", "long")
LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY: Final = "rank-step-counterbalanced-50-50"
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")


class LTOPCorePilotArm(str, Enum):
    """The path-matched treatment and control used by the core pilot."""

    FACTUAL = "ltop-ec-factual"
    BLOCKED = "ltop-ec-blocked"


@dataclass(frozen=True, slots=True)
class LTOPCorePilotCadence:
    """Registered cadence; no early checkpoint is permitted."""

    total_steps: int = LTOP_CORE_PILOT_TOTAL_STEPS
    metrics_every: int = LTOP_CORE_PILOT_METRICS_EVERY
    diagnostics_every: int = LTOP_CORE_PILOT_DIAGNOSTICS_EVERY
    checkpoint_step: int = LTOP_CORE_PILOT_CHECKPOINT_STEP

    def __post_init__(self) -> None:
        actual = (
            self.total_steps,
            self.metrics_every,
            self.diagnostics_every,
            self.checkpoint_step,
        )
        expected = (
            LTOP_CORE_PILOT_TOTAL_STEPS,
            LTOP_CORE_PILOT_METRICS_EVERY,
            LTOP_CORE_PILOT_DIAGNOSTICS_EVERY,
            LTOP_CORE_PILOT_CHECKPOINT_STEP,
        )
        if actual != expected:
            raise ValueError("the production LTOP core-pilot cadence is frozen at 2k/100/250/2k")

    def metrics_due(self, step: int) -> bool:
        return step > 0 and step % self.metrics_every == 0

    def diagnostics_due(self, step: int) -> bool:
        return step > 0 and step % self.diagnostics_every == 0

    def checkpoint_due(self, step: int) -> bool:
        return step == self.checkpoint_step


@dataclass(frozen=True, slots=True)
class LTOPCorePilotSmokeCadence:
    """Two-step I/O smoke; it makes no scientific claim and is discarded."""

    total_steps: int = 2
    metrics_every: int = 2
    diagnostics_every: int = 2
    checkpoint_step: int = 2

    def __post_init__(self) -> None:
        if (
            self.total_steps,
            self.metrics_every,
            self.diagnostics_every,
            self.checkpoint_step,
        ) != (2, 2, 2, 2):
            raise ValueError("the LTOP core-pilot engineering smoke is frozen at 2/2/2/2")

    def metrics_due(self, step: int) -> bool:
        return step == 2

    def diagnostics_due(self, step: int) -> bool:
        return step == 2

    def checkpoint_due(self, step: int) -> bool:
        return step == 2


@dataclass(frozen=True, slots=True)
class LTOPCoreRestartSmokeCadence:
    """Four-step engineering path with a real cold resume at step two."""

    total_steps: int = 4
    metrics_every: int = 2
    diagnostics_every: int = 2
    checkpoint_every: int = 2

    def __post_init__(self) -> None:
        if (
            self.total_steps,
            self.metrics_every,
            self.diagnostics_every,
            self.checkpoint_every,
        ) != (4, 2, 2, 2):
            raise ValueError("the LTOP restart smoke is frozen at 4/2/2/2")

    @property
    def checkpoint_step(self) -> int:
        return self.total_steps

    def metrics_due(self, step: int) -> bool:
        return step > 0 and step % self.metrics_every == 0

    def diagnostics_due(self, step: int) -> bool:
        return step > 0 and step % self.diagnostics_every == 0

    def checkpoint_due(self, step: int) -> bool:
        return step > 0 and step % self.checkpoint_every == 0


@dataclass(frozen=True, slots=True)
class LTOPCoreLongCadence:
    """Registered 30k maximum budget with resumable 2k boundaries."""

    total_steps: int = LTOP_CORE_LONG_TOTAL_STEPS
    metrics_every: int = LTOP_CORE_LONG_METRICS_EVERY
    diagnostics_every: int = LTOP_CORE_LONG_DIAGNOSTICS_EVERY
    checkpoint_every: int = LTOP_CORE_LONG_CHECKPOINT_EVERY

    def __post_init__(self) -> None:
        actual = (
            self.total_steps,
            self.metrics_every,
            self.diagnostics_every,
            self.checkpoint_every,
        )
        expected = (
            LTOP_CORE_LONG_TOTAL_STEPS,
            LTOP_CORE_LONG_METRICS_EVERY,
            LTOP_CORE_LONG_DIAGNOSTICS_EVERY,
            LTOP_CORE_LONG_CHECKPOINT_EVERY,
        )
        if actual != expected:
            raise ValueError("the LTOP long cadence is frozen at 30k/100/250/2k")

    @property
    def checkpoint_step(self) -> int:
        """Return the terminal boundary for callers shared with the 2k pilot."""

        return self.total_steps

    def metrics_due(self, step: int) -> bool:
        return step > 0 and step % self.metrics_every == 0

    def diagnostics_due(self, step: int) -> bool:
        return step > 0 and step % self.diagnostics_every == 0

    def checkpoint_due(self, step: int) -> bool:
        return step > 0 and step % self.checkpoint_every == 0


@dataclass(frozen=True, slots=True)
class AcceptedG3Gate:
    """Immutable evidence that authorizes the separate 2k pilot."""

    path: Path
    file_sha256: str
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AcceptedG3MediatorGate:
    """Immutable complete G3 evidence that authorizes long training."""

    path: Path
    file_sha256: str
    report: dict[str, Any]
    checkpoint_path: Path
    training_final_model_local_state_sha256_by_rank: tuple[str, str]
    checkpoint_model_tree_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_accepted_g3_gate(path: Path) -> AcceptedG3Gate:
    """Reject any non-final, failed, or non-registered G3 report."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"G3 PASS report is absent or not a regular file: {path}")
    report = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(report, dict):
        raise TypeError("G3 report must be a JSON object")
    if report.get("schema") != LTOP_CORE_PILOT_G3_SCHEMA:
        raise ValueError("G3 report schema is not the registered production action gate")
    if report.get("status") != "PASS" or report.get("failures") != []:
        raise ValueError("LTOP core pilot requires a failure-free G3 PASS report")
    if report.get("mode") != "gate":
        raise ValueError("LTOP core pilot cannot be authorized by a G3 smoke report")
    if report.get("steps") != 128 or report.get("eval_every") != 32:
        raise ValueError("G3 report does not use the registered 128/32 schedule")
    if report.get("world_size") != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("G3 report does not use the registered two-GPU topology")
    return AcceptedG3Gate(path=path.resolve(), file_sha256=_sha256(path), report=report)


def load_accepted_g3_mediator_gate(path: Path) -> AcceptedG3MediatorGate:
    """Load the complete mediator G3 ABI and revalidate live evidence receipts."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"mediator G3 acceptance is absent or not a regular file: {path}")
    report = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(report, dict):
        raise TypeError("mediator G3 acceptance must be a JSON object")
    if report.get("schema") != MEDIATOR_ACCEPTANCE_SCHEMA:
        raise ValueError("mediator G3 acceptance schema differs from the registered ABI")
    if report.get("status") != "PASS" or report.get("failures") != []:
        raise ValueError("long training requires a failure-free mediator G3 acceptance")
    if report.get("world_size") != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("mediator G3 acceptance uses another distributed topology")

    training_contract = report.get("training_contract")
    if not isinstance(training_contract, dict):
        raise TypeError("mediator G3 acceptance omits its training contract")
    expected_training_contract = {
        "mode": "mediator-trial",
        "steps": 256,
        "eval_every": 32,
        "action_information_set_policy": MEDIATOR_ACTION_INFORMATION_SET_POLICY,
    }
    for name, expected in expected_training_contract.items():
        if training_contract.get(name) != expected:
            raise ValueError(f"mediator G3 training contract changed {name}")
    schedule_sha256 = training_contract.get("schedule_sha256")
    if not isinstance(schedule_sha256, str) or not _SHA256_PATTERN.fullmatch(schedule_sha256):
        raise ValueError("mediator G3 training schedule SHA-256 is malformed")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("mediator G3 acceptance omits its checkpoint receipt")
    if checkpoint.get("format") != MODEL_ONLY_CHECKPOINT_FORMAT:
        raise ValueError("mediator G3 checkpoint format differs from the registered ABI")
    if checkpoint.get("optimizer_saved") is not False:
        raise ValueError("mediator G3 initialization checkpoint must be model-only")
    checkpoint_value = checkpoint.get("path")
    if not isinstance(checkpoint_value, str) or not Path(checkpoint_value).is_absolute():
        raise ValueError("mediator G3 checkpoint path must be absolute")
    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise FileNotFoundError("mediator G3 checkpoint is absent or not a real directory")
    checkpoint_manifest = checkpoint_path / "ltop_g3_training_checkpoint.json"
    if checkpoint_manifest.is_symlink() or not checkpoint_manifest.is_file():
        raise FileNotFoundError("mediator G3 checkpoint omits its immutable manifest")
    manifest = json.loads(checkpoint_manifest.read_text(encoding="ascii"))
    checkpoint_identity = report.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, dict):
        raise TypeError("mediator G3 acceptance omits checkpoint identity")
    expected_manifest_sha256 = checkpoint_identity.get("manifest_sha256")
    if (
        not isinstance(expected_manifest_sha256, str)
        or not _SHA256_PATTERN.fullmatch(expected_manifest_sha256)
        or _sha256(checkpoint_manifest) != expected_manifest_sha256
        or checkpoint.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("mediator G3 checkpoint manifest SHA-256 differs")
    if (
        checkpoint_identity.get("model_tree_schema") != MODEL_TREE_SCHEMA
        or checkpoint.get("model_tree_schema") != MODEL_TREE_SCHEMA
    ):
        raise ValueError("mediator G3 checkpoint model-tree schema differs")
    model_dir = checkpoint_path / "model"
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise FileNotFoundError("mediator G3 checkpoint omits its DCP model directory")
    model_names = {value.name for value in model_dir.iterdir() if value.is_file()}
    if ".metadata" not in model_names or not any(
        value.endswith(".distcp") for value in model_names
    ):
        raise ValueError("mediator G3 checkpoint omits its DCP model payload")
    observed_model_tree_sha256 = directory_tree_sha256(
        model_dir,
        schema=MODEL_TREE_SCHEMA,
    )
    expected_model_tree_sha256 = checkpoint_identity.get("model_tree_sha256")
    if (
        not isinstance(expected_model_tree_sha256, str)
        or not _SHA256_PATTERN.fullmatch(expected_model_tree_sha256)
        or observed_model_tree_sha256 != expected_model_tree_sha256
        or checkpoint.get("model_tree_sha256") != expected_model_tree_sha256
    ):
        raise ValueError("mediator G3 checkpoint model tree changed after acceptance")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != TRAINING_CHECKPOINT_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("optimizer_saved") is not False
        or manifest.get("format") != MODEL_ONLY_CHECKPOINT_FORMAT
        or manifest.get("world_size") != LTOP_CORE_PILOT_WORLD_SIZE
        or manifest.get("global_step") != 256
        or manifest.get("action_information_set_schedule_sha256") != schedule_sha256
        or manifest.get("model_tree_schema") != MODEL_TREE_SCHEMA
        or manifest.get("model_tree_sha256") != expected_model_tree_sha256
    ):
        raise ValueError("mediator G3 checkpoint manifest differs from acceptance")

    digests = report.get("training_final_model_local_state_sha256_by_rank")
    if (
        not isinstance(digests, list)
        or len(digests) != LTOP_CORE_PILOT_WORLD_SIZE
        or any(
            not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value) for value in digests
        )
    ):
        raise ValueError("mediator G3 per-rank training terminal model digests are malformed")
    if (
        checkpoint.get("training_final_model_local_state_sha256_by_rank") != digests
        or manifest.get("training_final_model_local_state_sha256_by_rank") != digests
    ):
        raise ValueError("mediator G3 checkpoint terminal model digests differ")

    evidence = report.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "training_report",
        "arm_validation",
        "cold_action_evaluation",
        "cold_retention",
    }:
        raise ValueError("mediator G3 acceptance evidence set is incomplete")
    evidence_paths: dict[str, Path] = {}
    for name, receipt in evidence.items():
        if not isinstance(receipt, dict):
            raise TypeError(f"mediator G3 evidence receipt is malformed: {name}")
        evidence_path_value = receipt.get("path")
        expected_sha256 = receipt.get("sha256")
        if (
            not isinstance(evidence_path_value, str)
            or not Path(evidence_path_value).is_absolute()
            or not isinstance(expected_sha256, str)
            or not _SHA256_PATTERN.fullmatch(expected_sha256)
        ):
            raise ValueError(f"mediator G3 evidence receipt is malformed: {name}")
        evidence_path = Path(evidence_path_value)
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise FileNotFoundError(f"mediator G3 evidence disappeared: {name}")
        if _sha256(evidence_path) != expected_sha256:
            raise ValueError(f"mediator G3 evidence changed after acceptance: {name}")
        evidence_paths[name] = evidence_path
    recomposed = compose_ltop_g3_mediator_acceptance(
        training_path=evidence_paths["training_report"],
        arm_validation_path=evidence_paths["arm_validation"],
        action_evaluation_path=evidence_paths["cold_action_evaluation"],
        retention_path=evidence_paths["cold_retention"],
    )
    if recomposed != report:
        raise ValueError("mediator G3 acceptance differs from semantic recomposition")

    return AcceptedG3MediatorGate(
        path=path.resolve(),
        file_sha256=_sha256(path),
        report=report,
        checkpoint_path=checkpoint_path.resolve(),
        training_final_model_local_state_sha256_by_rank=(digests[0], digests[1]),
        checkpoint_model_tree_sha256=expected_model_tree_sha256,
    )


def load_accepted_g3_source_aligned_gate(path: Path) -> AcceptedG3MediatorGate:
    """Load ADR170 and semantically recompose all live source-aligned evidence."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"source-aligned G3 acceptance is absent or not a regular file: {path}"
        )
    report = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(report, dict):
        raise TypeError("source-aligned G3 acceptance must be a JSON object")
    if report.get("schema") != SOURCE_ALIGNED_ACCEPTANCE_SCHEMA:
        raise ValueError("source-aligned G3 acceptance schema differs from the ADR170 ABI")
    if report.get("status") != "PASS" or report.get("failures") != []:
        raise ValueError("long training requires a failure-free source-aligned G3 acceptance")
    if report.get("world_size") != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("source-aligned G3 acceptance uses another distributed topology")

    training_contract = report.get("training_contract")
    if not isinstance(training_contract, dict):
        raise TypeError("source-aligned G3 acceptance omits its training contract")
    expected_training_contract = {
        "mode": "mediator-trial",
        "steps": 256,
        "eval_every": 32,
        "schedule_schema": SOURCE_ACTION_SCHEDULE_SCHEMA,
        "action_supervision": SOURCE_ALIGNED_ACTION_SUPERVISION,
    }
    for name, expected in expected_training_contract.items():
        if training_contract.get(name) != expected:
            raise ValueError(f"source-aligned G3 training contract changed {name}")
    schedule_sha256 = training_contract.get("schedule_sha256")
    if not isinstance(schedule_sha256, str) or not _SHA256_PATTERN.fullmatch(schedule_sha256):
        raise ValueError("source-aligned G3 training schedule SHA-256 is malformed")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("source-aligned G3 acceptance omits its checkpoint receipt")
    if checkpoint.get("format") != MODEL_ONLY_CHECKPOINT_FORMAT:
        raise ValueError("source-aligned G3 checkpoint format differs")
    if checkpoint.get("optimizer_saved") is not False:
        raise ValueError("source-aligned G3 initialization checkpoint must be model-only")
    if checkpoint.get("action_supervision_schema") != SOURCE_ALIGNED_ACTION_SUPERVISION_SCHEMA:
        raise ValueError("source-aligned G3 checkpoint supervision schema differs")
    checkpoint_value = checkpoint.get("path")
    if not isinstance(checkpoint_value, str) or not Path(checkpoint_value).is_absolute():
        raise ValueError("source-aligned G3 checkpoint path must be absolute")
    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise FileNotFoundError(
            "source-aligned G3 checkpoint is absent or not a real directory"
        )
    checkpoint_manifest = checkpoint_path / "ltop_g3_training_checkpoint.json"
    if checkpoint_manifest.is_symlink() or not checkpoint_manifest.is_file():
        raise FileNotFoundError("source-aligned G3 checkpoint omits its immutable manifest")
    manifest = json.loads(checkpoint_manifest.read_text(encoding="ascii"))
    checkpoint_identity = report.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, dict):
        raise TypeError("source-aligned G3 acceptance omits checkpoint identity")
    expected_manifest_sha256 = checkpoint_identity.get("manifest_sha256")
    if (
        not isinstance(expected_manifest_sha256, str)
        or not _SHA256_PATTERN.fullmatch(expected_manifest_sha256)
        or _sha256(checkpoint_manifest) != expected_manifest_sha256
        or checkpoint.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("source-aligned G3 checkpoint manifest SHA-256 differs")
    if checkpoint_identity.get("manifest_schema") != SOURCE_ALIGNED_TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("source-aligned G3 checkpoint manifest schema differs")
    if (
        checkpoint_identity.get("model_tree_schema") != MODEL_TREE_SCHEMA
        or checkpoint.get("model_tree_schema") != MODEL_TREE_SCHEMA
    ):
        raise ValueError("source-aligned G3 checkpoint model-tree schema differs")
    model_dir = checkpoint_path / "model"
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise FileNotFoundError("source-aligned G3 checkpoint omits its DCP model directory")
    model_names = {value.name for value in model_dir.iterdir() if value.is_file()}
    if ".metadata" not in model_names or not any(
        value.endswith(".distcp") for value in model_names
    ):
        raise ValueError("source-aligned G3 checkpoint omits its DCP model payload")
    observed_model_tree_sha256 = directory_tree_sha256(model_dir, schema=MODEL_TREE_SCHEMA)
    expected_model_tree_sha256 = checkpoint_identity.get("model_tree_sha256")
    if (
        not isinstance(expected_model_tree_sha256, str)
        or not _SHA256_PATTERN.fullmatch(expected_model_tree_sha256)
        or observed_model_tree_sha256 != expected_model_tree_sha256
        or checkpoint.get("model_tree_sha256") != expected_model_tree_sha256
    ):
        raise ValueError("source-aligned G3 checkpoint model tree changed after acceptance")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SOURCE_ALIGNED_TRAINING_CHECKPOINT_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("optimizer_saved") is not False
        or manifest.get("format") != MODEL_ONLY_CHECKPOINT_FORMAT
        or manifest.get("world_size") != LTOP_CORE_PILOT_WORLD_SIZE
        or manifest.get("global_step") != 256
        or manifest.get("action_supervision_schema")
        != SOURCE_ALIGNED_ACTION_SUPERVISION_SCHEMA
        or manifest.get("action_information_set_schedule_sha256") != schedule_sha256
        or manifest.get("model_tree_schema") != MODEL_TREE_SCHEMA
        or manifest.get("model_tree_sha256") != expected_model_tree_sha256
    ):
        raise ValueError("source-aligned G3 checkpoint manifest differs from acceptance")

    digests = report.get("training_final_model_local_state_sha256_by_rank")
    if (
        not isinstance(digests, list)
        or len(digests) != LTOP_CORE_PILOT_WORLD_SIZE
        or any(
            not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value)
            for value in digests
        )
    ):
        raise ValueError("source-aligned G3 per-rank terminal model digests are malformed")
    if (
        checkpoint.get("training_final_model_local_state_sha256_by_rank") != digests
        or manifest.get("training_final_model_local_state_sha256_by_rank") != digests
    ):
        raise ValueError("source-aligned G3 checkpoint terminal model digests differ")

    evidence = report.get("evidence")
    expected_evidence = {
        "training_report",
        "arm_validation",
        "cold_action_factual",
        "cold_action_mediator_required",
        "cold_retention",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
        raise ValueError("source-aligned G3 acceptance evidence set is incomplete")
    evidence_paths: dict[str, Path] = {}
    for name, receipt in evidence.items():
        if not isinstance(receipt, dict):
            raise TypeError(f"source-aligned G3 evidence receipt is malformed: {name}")
        evidence_path_value = receipt.get("path")
        expected_sha256 = receipt.get("sha256")
        if (
            not isinstance(evidence_path_value, str)
            or not Path(evidence_path_value).is_absolute()
            or not isinstance(expected_sha256, str)
            or not _SHA256_PATTERN.fullmatch(expected_sha256)
        ):
            raise ValueError(f"source-aligned G3 evidence receipt is malformed: {name}")
        evidence_path = Path(evidence_path_value)
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise FileNotFoundError(f"source-aligned G3 evidence disappeared: {name}")
        if _sha256(evidence_path) != expected_sha256:
            raise ValueError(f"source-aligned G3 evidence changed after acceptance: {name}")
        evidence_paths[name] = evidence_path
    recomposed = compose_ltop_g3_source_aligned_acceptance(
        training_path=evidence_paths["training_report"],
        arm_validation_path=evidence_paths["arm_validation"],
        factual_action_path=evidence_paths["cold_action_factual"],
        mediator_action_path=evidence_paths["cold_action_mediator_required"],
        retention_path=evidence_paths["cold_retention"],
    )
    if recomposed != report:
        raise ValueError("source-aligned G3 acceptance differs from semantic recomposition")

    return AcceptedG3MediatorGate(
        path=path.resolve(),
        file_sha256=_sha256(path),
        report=report,
        checkpoint_path=checkpoint_path.resolve(),
        training_final_model_local_state_sha256_by_rank=(digests[0], digests[1]),
        checkpoint_model_tree_sha256=expected_model_tree_sha256,
    )


def matched_arm_contract(
    arm: LTOPCorePilotArm,
    *,
    start_state: str = "same-accepted-g2b-model-only-checkpoint",
) -> dict[str, Any]:
    """Describe the sole registered difference between the paired arms."""

    if not isinstance(arm, LTOPCorePilotArm):
        raise TypeError("core-pilot arm must be typed")
    if not isinstance(start_state, str) or not start_state:
        raise ValueError("core-pilot start-state description cannot be empty")
    return {
        "arm": arm.value,
        "start_state": start_state,
        "optimizer_state": "fresh-zero-state",
        "forward_abi": "lingbot-native-exact-cache-prefix-suffix",
        "training_objective": (
            "released-action-moe+task-free-physical-set+applicable-task-address"
        ),
        "object_read_action_intervention": (
            "factual" if arm is LTOPCorePilotArm.FACTUAL else "blocked"
        ),
        "only_permitted_pair_difference": "typed-object-read-to-action-edge",
    }

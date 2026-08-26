"""Label-free, two-stage frozen-posterior action diagnostics.

This module is deliberately diagnostic-only.  It defines a strict boundary
between a factual correction pass and an action-only readout without changing
the released training or deployment paths.  A real-host adapter may implement
the two protocols below only if it can preserve these fail-closed contracts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.state import NativeLayerwisePosteriorState


class FrozenPosteriorVisibility(str, Enum):
    """The only admitted posterior-to-action visibility contracts."""

    DIRECT_ONLY = "direct-only"
    LANGUAGE_MEDIATED = "language-mediated"
    BOTH = "both"


class DiagnosticInformationNode(str, Enum):
    """Abstract information sources used to audit the action-only boundary."""

    CURRENT_SCENE = "current-scene"
    DENSE_MODALITY = "dense-modality"
    EXTERNAL_TRACE = "external-trace"
    PRIOR = "prior"
    POSTERIOR = "posterior"
    LANGUAGE = "language"
    CONTROL = "control"
    PROPRIOCEPTION = "proprioception"
    HOST_AUX = "host-aux"
    MATCH = "match"
    ACTION = "action"


_FORBIDDEN_ACTION_SOURCES = (
    DiagnosticInformationNode.CURRENT_SCENE,
    DiagnosticInformationNode.DENSE_MODALITY,
    DiagnosticInformationNode.EXTERNAL_TRACE,
    DiagnosticInformationNode.PRIOR,
    DiagnosticInformationNode.HOST_AUX,
    DiagnosticInformationNode.MATCH,
)


def parse_frozen_posterior_visibility(value: str) -> FrozenPosteriorVisibility:
    """Parse one exact contract spelling; aliases and normalization fail closed."""

    if type(value) is not str:
        raise TypeError("frozen-posterior visibility must be a string")
    try:
        return FrozenPosteriorVisibility(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in FrozenPosteriorVisibility)
        raise ValueError(
            f"unknown frozen-posterior visibility {value!r}; expected {allowed}"
        ) from error


def frozen_posterior_visibility_edges(
    visibility: FrozenPosteriorVisibility,
) -> frozenset[tuple[DiagnosticInformationNode, DiagnosticInformationNode]]:
    """Return source-to-sink edges for the isolated action-only phase."""

    if not isinstance(visibility, FrozenPosteriorVisibility):
        raise TypeError("visibility must be a parsed FrozenPosteriorVisibility")
    edges = {
        (DiagnosticInformationNode.LANGUAGE, DiagnosticInformationNode.ACTION),
        (DiagnosticInformationNode.CONTROL, DiagnosticInformationNode.ACTION),
        (DiagnosticInformationNode.PROPRIOCEPTION, DiagnosticInformationNode.ACTION),
    }
    if visibility in {FrozenPosteriorVisibility.DIRECT_ONLY, FrozenPosteriorVisibility.BOTH}:
        edges.add((DiagnosticInformationNode.POSTERIOR, DiagnosticInformationNode.ACTION))
    if visibility in {
        FrozenPosteriorVisibility.LANGUAGE_MEDIATED,
        FrozenPosteriorVisibility.BOTH,
    }:
        edges.add((DiagnosticInformationNode.POSTERIOR, DiagnosticInformationNode.LANGUAGE))
    return frozenset(edges)


def _is_reachable(
    edges: frozenset[tuple[DiagnosticInformationNode, DiagnosticInformationNode]],
    source: DiagnosticInformationNode,
    sink: DiagnosticInformationNode,
) -> bool:
    frontier = {source}
    visited: set[DiagnosticInformationNode] = set()
    while frontier:
        current = frontier.pop()
        if current == sink:
            return True
        if current in visited:
            continue
        visited.add(current)
        frontier.update(target for origin, target in edges if origin == current)
    return False


@dataclass(frozen=True, slots=True)
class FrozenPosteriorVisibilityAudit:
    visibility: FrozenPosteriorVisibility
    direct_posterior_path: bool
    language_mediated_posterior_path: bool
    forbidden_sources_reaching_action: tuple[DiagnosticInformationNode, ...]


def audit_frozen_posterior_visibility(
    visibility: FrozenPosteriorVisibility,
) -> FrozenPosteriorVisibilityAudit:
    """Prove the declared action-stage graph is closed to undeclared sources."""

    edges = frozen_posterior_visibility_edges(visibility)
    direct = (
        DiagnosticInformationNode.POSTERIOR,
        DiagnosticInformationNode.ACTION,
    ) in edges
    mediated = (
        DiagnosticInformationNode.POSTERIOR,
        DiagnosticInformationNode.LANGUAGE,
    ) in edges and (
        DiagnosticInformationNode.LANGUAGE,
        DiagnosticInformationNode.ACTION,
    ) in edges
    forbidden = tuple(
        source
        for source in _FORBIDDEN_ACTION_SOURCES
        if _is_reachable(edges, source, DiagnosticInformationNode.ACTION)
    )
    expected = {
        FrozenPosteriorVisibility.DIRECT_ONLY: (True, False),
        FrozenPosteriorVisibility.LANGUAGE_MEDIATED: (False, True),
        FrozenPosteriorVisibility.BOTH: (True, True),
    }[visibility]
    if (direct, mediated) != expected or forbidden:
        raise RuntimeError("frozen-posterior visibility contract is not causally closed")
    return FrozenPosteriorVisibilityAudit(
        visibility=visibility,
        direct_posterior_path=direct,
        language_mediated_posterior_path=mediated,
        forbidden_sources_reaching_action=forbidden,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor digest requires a torch tensor")
    materialized = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(materialized.dtype).encode("ascii"))
    digest.update(repr(tuple(materialized.shape)).encode("ascii"))
    digest.update(materialized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenPosteriorShapeContract:
    num_layers: int
    capacity: int
    host_width: int

    def __post_init__(self) -> None:
        for name in ("num_layers", "capacity", "host_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FrozenPosteriorSnapshot:
    """Detached factual correction output, including every declared host layer."""

    state: NativeLayerwisePosteriorState
    shape_contract: FrozenPosteriorShapeContract
    provenance_id: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, NativeLayerwisePosteriorState):
            raise TypeError("snapshot requires a NativeLayerwisePosteriorState")
        if not isinstance(self.shape_contract, FrozenPosteriorShapeContract):
            raise TypeError("snapshot requires a FrozenPosteriorShapeContract")
        if self.state.layer_rows.requires_grad:
            raise ValueError("frozen posterior snapshot cannot require gradients")
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("snapshot provenance_id must be nonempty")
        expected_shape = (
            self.shape_contract.num_layers,
            self.shape_contract.capacity,
            self.shape_contract.host_width,
        )
        if self.state.layer_rows.shape[1:] != expected_shape:
            raise ValueError(
                "factual correction did not return the complete declared layerwise state"
            )
        if self.tensor_sha256 != _tensor_sha256(self.state.layer_rows):
            raise ValueError("frozen posterior snapshot digest does not match its tensor")

    def assert_intact(self) -> None:
        if self.tensor_sha256 != _tensor_sha256(self.state.layer_rows):
            raise RuntimeError("frozen posterior snapshot was mutated")


FactualCorrection = Callable[[], NativeLayerwisePosteriorState]


def capture_factual_posterior_snapshot(
    correction: FactualCorrection,
    *,
    shape_contract: FrozenPosteriorShapeContract,
    provenance_id: str,
) -> FrozenPosteriorSnapshot:
    """Run factual correction once and detach a complete layerwise snapshot."""

    if not callable(correction):
        raise TypeError("factual correction must be callable")
    with torch.inference_mode():
        state = correction()
        if not isinstance(state, NativeLayerwisePosteriorState):
            raise TypeError("factual correction must return NativeLayerwisePosteriorState")
        frozen_rows = state.layer_rows.detach().clone()
    frozen = NativeLayerwisePosteriorState(frozen_rows)
    return FrozenPosteriorSnapshot(
        state=frozen,
        shape_contract=shape_contract,
        provenance_id=provenance_id,
        tensor_sha256=_tensor_sha256(frozen_rows),
    )


@dataclass(frozen=True, slots=True)
class LanguagePromptBatch:
    """Deploy-visible language only; no object identity or target-row side channel."""

    token_ids: torch.Tensor
    token_valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.token_ids.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("language token_ids must be integer [batch,tokens]")
        if self.token_valid.shape != self.token_ids.shape or self.token_valid.dtype != torch.bool:
            raise ValueError("language token_valid must be boolean and match token_ids")
        if self.token_valid.device != self.token_ids.device:
            raise ValueError("language tensors must share one device")
        if not self.token_valid.any(dim=1).all():
            raise ValueError("every prompt requires at least one valid language token")


@dataclass(frozen=True, slots=True)
class LabelFreePromptVariant:
    name: str
    language: LanguagePromptBatch

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("prompt variant name must be nonempty")
        if not isinstance(self.language, LanguagePromptBatch):
            raise TypeError("prompt variant requires a LanguagePromptBatch")


@dataclass(frozen=True, slots=True)
class FrozenInferenceNoise:
    values: torch.Tensor
    tensor_sha256: str

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or not self.values.is_floating_point():
            raise ValueError("inference noise must be floating [batch,horizon,action_dim]")
        if not torch.isfinite(self.values).all() or self.values.requires_grad:
            raise ValueError("inference noise must be finite and gradient-free")
        if self.tensor_sha256 != _tensor_sha256(self.values):
            raise ValueError("inference noise digest does not match its tensor")

    @classmethod
    def capture(cls, value: torch.Tensor) -> FrozenInferenceNoise:
        if not isinstance(value, torch.Tensor):
            raise TypeError("inference noise must be a torch tensor")
        frozen = value.detach().clone()
        return cls(values=frozen, tensor_sha256=_tensor_sha256(frozen))

    def assert_intact(self) -> None:
        if self.tensor_sha256 != _tensor_sha256(self.values):
            raise RuntimeError("frozen inference noise was mutated")


class FrozenPosteriorInterventionKind(str, Enum):
    FACTUAL = "factual"
    VISIBILITY_REMOVAL = "visibility-removal"
    MOMENT_MATCHED_DONOR = "moment-matched-donor"
    CONSISTENT_PERMUTATION = "consistent-permutation"


@dataclass(frozen=True, slots=True)
class FrozenPosteriorArm:
    """Offline arm metadata; only state and visibility enter the policy call."""

    name: str
    kind: FrozenPosteriorInterventionKind
    state: NativeLayerwisePosteriorState
    row_visible: torch.Tensor
    row_index: int | None = None
    permutation: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("posterior arm name must be nonempty")
        if not isinstance(self.kind, FrozenPosteriorInterventionKind):
            raise TypeError("posterior arm kind must be a parsed intervention kind")
        if not isinstance(self.state, NativeLayerwisePosteriorState):
            raise TypeError("posterior arm requires a NativeLayerwisePosteriorState")
        if self.state.layer_rows.requires_grad:
            raise ValueError("posterior arm state must be gradient-free")
        expected = (self.state.batch_size, self.state.capacity)
        if self.row_visible.shape != expected or self.row_visible.dtype != torch.bool:
            raise ValueError("posterior row visibility must be boolean [batch,capacity]")
        if self.row_visible.device != self.state.layer_rows.device:
            raise ValueError("posterior state and visibility must share one device")
        if self.row_index is not None and not 0 <= self.row_index < self.state.capacity:
            raise ValueError("posterior arm row_index is out of range")
        if self.permutation is not None and tuple(sorted(self.permutation)) != tuple(
            range(self.state.capacity)
        ):
            raise ValueError("posterior arm permutation must contain every row exactly once")


def factual_frozen_posterior_arm(snapshot: FrozenPosteriorSnapshot) -> FrozenPosteriorArm:
    snapshot.assert_intact()
    return FrozenPosteriorArm(
        name="factual",
        kind=FrozenPosteriorInterventionKind.FACTUAL,
        state=NativeLayerwisePosteriorState(snapshot.state.layer_rows.detach().clone()),
        row_visible=torch.ones(
            snapshot.state.batch_size,
            snapshot.state.capacity,
            dtype=torch.bool,
            device=snapshot.state.layer_rows.device,
        ),
    )


def label_blind_visibility_removal_arms(
    snapshot: FrozenPosteriorSnapshot,
) -> tuple[FrozenPosteriorArm, ...]:
    """Enumerate every row; no target identity is accepted by this interface."""

    snapshot.assert_intact()
    arms = []
    for row_index in range(snapshot.state.capacity):
        visible = torch.ones(
            snapshot.state.batch_size,
            snapshot.state.capacity,
            dtype=torch.bool,
            device=snapshot.state.layer_rows.device,
        )
        visible[:, row_index] = False
        arms.append(
            FrozenPosteriorArm(
                name=f"remove-row-{row_index}",
                kind=FrozenPosteriorInterventionKind.VISIBILITY_REMOVAL,
                state=NativeLayerwisePosteriorState(snapshot.state.layer_rows.detach().clone()),
                row_visible=visible,
                row_index=row_index,
            )
        )
    return tuple(arms)


def _moment_match_row(
    donor: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    donor_work = donor.float()
    target_work = target.float()
    donor_centered = donor_work - donor_work.mean(dim=-1, keepdim=True)
    target_mean = target_work.mean(dim=-1, keepdim=True)
    target_centered = target_work - target_mean
    donor_rms = donor_centered.square().mean(dim=-1, keepdim=True).sqrt()
    target_rms = target_centered.square().mean(dim=-1, keepdim=True).sqrt()
    impossible = (donor_rms <= epsilon) & (target_rms > epsilon)
    if impossible.any():
        raise ValueError("degenerate donor row cannot match a non-degenerate factual row")
    scaled = donor_centered / donor_rms.clamp_min(epsilon) * target_rms + target_mean
    matched = torch.where((target_rms <= epsilon).expand_as(scaled), target_mean, scaled)
    return matched.to(target.dtype)


def label_blind_moment_matched_donor_arms(
    snapshot: FrozenPosteriorSnapshot,
    donor: FrozenPosteriorSnapshot,
    *,
    epsilon: float = 1e-6,
) -> tuple[FrozenPosteriorArm, ...]:
    """Replace every row in turn with a cross-provenance, per-layer matched donor."""

    snapshot.assert_intact()
    donor.assert_intact()
    if snapshot.provenance_id == donor.provenance_id:
        raise ValueError("moment-matched donor must have different provenance")
    if snapshot.state.layer_rows.shape != donor.state.layer_rows.shape:
        raise ValueError("factual and donor posterior snapshots must have identical shapes")
    if snapshot.state.layer_rows.device != donor.state.layer_rows.device:
        raise ValueError("factual and donor posterior snapshots must share one device")
    if snapshot.state.layer_rows.dtype != donor.state.layer_rows.dtype:
        raise ValueError("factual and donor posterior snapshots must share one dtype")
    if not isinstance(epsilon, float) or not 0 < epsilon < 1:
        raise ValueError("moment-match epsilon must be a float strictly between zero and one")

    arms = []
    factual_rows = snapshot.state.layer_rows
    donor_rows = donor.state.layer_rows
    for row_index in range(snapshot.state.capacity):
        replaced = factual_rows.detach().clone()
        replaced[:, :, row_index] = _moment_match_row(
            donor_rows[:, :, row_index],
            factual_rows[:, :, row_index],
            epsilon=epsilon,
        )
        arms.append(
            FrozenPosteriorArm(
                name=f"donor-row-{row_index}",
                kind=FrozenPosteriorInterventionKind.MOMENT_MATCHED_DONOR,
                state=NativeLayerwisePosteriorState(replaced),
                row_visible=torch.ones(
                    snapshot.state.batch_size,
                    snapshot.state.capacity,
                    dtype=torch.bool,
                    device=replaced.device,
                ),
                row_index=row_index,
            )
        )
    return tuple(arms)


def consistent_row_permutation_arm(
    snapshot: FrozenPosteriorSnapshot,
    permutation: Sequence[int],
) -> FrozenPosteriorArm:
    """Move row content with one identical permutation at every host layer."""

    snapshot.assert_intact()
    values = tuple(permutation)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("row permutation entries must be integers")
    indices = torch.tensor(values, dtype=torch.long, device=snapshot.state.layer_rows.device)
    moved = snapshot.state.permute_rows(indices).detached()
    return FrozenPosteriorArm(
        name="permute-rows-" + "-".join(str(value) for value in values),
        kind=FrozenPosteriorInterventionKind.CONSISTENT_PERMUTATION,
        state=NativeLayerwisePosteriorState(moved.layer_rows.clone()),
        row_visible=torch.ones(
            moved.batch_size,
            moved.capacity,
            dtype=torch.bool,
            device=moved.layer_rows.device,
        ),
        permutation=values,
    )


@dataclass(frozen=True, slots=True)
class FrozenPosteriorActionRequest:
    """The complete and intentionally narrow policy-forward diagnostic ABI."""

    language: LanguagePromptBatch
    controls: ExecutedControlBatch
    proprioception: torch.Tensor
    posterior: NativeLayerwisePosteriorState
    posterior_row_visible: torch.Tensor
    inference_noise: torch.Tensor
    visibility: FrozenPosteriorVisibility

    def __post_init__(self) -> None:
        audit_frozen_posterior_visibility(self.visibility)
        if not isinstance(self.language, LanguagePromptBatch):
            raise TypeError("action-only request requires a LanguagePromptBatch")
        if not isinstance(self.controls, ExecutedControlBatch):
            raise TypeError("action-only request requires an ExecutedControlBatch")
        if not isinstance(self.posterior, NativeLayerwisePosteriorState):
            raise TypeError("action-only request requires NativeLayerwisePosteriorState")
        batch = self.posterior.batch_size
        device = self.posterior.layer_rows.device
        if self.posterior.layer_rows.requires_grad:
            raise ValueError("action-only posterior must be gradient-free")
        if self.language.token_ids.shape[0] != batch or self.language.token_ids.device != device:
            raise ValueError("language and posterior must share batch and device")
        if self.controls.batch_size != batch or self.controls.values.device != device:
            raise ValueError("controls and posterior must share batch and device")
        if (
            self.proprioception.ndim < 2
            or self.proprioception.shape[0] != batch
            or not self.proprioception.is_floating_point()
            or not torch.isfinite(self.proprioception).all()
            or self.proprioception.requires_grad
            or self.proprioception.device != device
        ):
            raise ValueError(
                "proprioception must be finite floating point with matching batch/device"
            )
        expected_visibility = (batch, self.posterior.capacity)
        if (
            self.posterior_row_visible.shape != expected_visibility
            or self.posterior_row_visible.dtype != torch.bool
            or self.posterior_row_visible.device != device
        ):
            raise ValueError("posterior_row_visible must be boolean [batch,capacity]")
        if (
            self.inference_noise.ndim != 3
            or self.inference_noise.shape[0] != batch
            or not self.inference_noise.is_floating_point()
            or not torch.isfinite(self.inference_noise).all()
            or self.inference_noise.requires_grad
            or self.inference_noise.device != device
        ):
            raise ValueError("inference noise must be finite gradient-free [batch,horizon,width]")


class FrozenPosteriorActionReadout(Protocol):
    """Diagnostic adapter for the real or stub action-only LingBot readout."""

    def __call__(self, request: FrozenPosteriorActionRequest, /) -> torch.Tensor: ...


def _controls_sha256(controls: ExecutedControlBatch) -> str:
    digest = hashlib.sha256()
    for value in (
        controls.values,
        controls.field_valid,
        controls.token_valid,
        controls.delta_time,
        controls.reset,
        controls.acknowledged,
    ):
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _action_request_sha256(request: FrozenPosteriorActionRequest) -> str:
    digest = hashlib.sha256()
    for value in (
        request.language.token_ids,
        request.language.token_valid,
        request.controls.values,
        request.controls.field_valid,
        request.controls.token_valid,
        request.controls.delta_time,
        request.controls.reset,
        request.controls.acknowledged,
        request.proprioception,
        request.posterior.layer_rows,
        request.posterior_row_visible,
        request.inference_noise,
    ):
        digest.update(_tensor_sha256(value).encode("ascii"))
    digest.update(request.visibility.value.encode("ascii"))
    return digest.hexdigest()


def _clone_controls(controls: ExecutedControlBatch) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=controls.values.detach().clone(),
        field_valid=controls.field_valid.clone(),
        token_valid=controls.token_valid.clone(),
        delta_time=controls.delta_time.detach().clone(),
        reset=controls.reset.clone(),
        acknowledged=controls.acknowledged.clone(),
    )


@dataclass(frozen=True, slots=True)
class FrozenPosteriorActionReceipt:
    prompt_name: str
    arm_name: str
    visibility: FrozenPosteriorVisibility
    action: torch.Tensor
    posterior_sha256: str
    row_visibility_sha256: str
    inference_noise_sha256: str
    request_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenPosteriorDiagnosticResult:
    factual_snapshot_sha256: str
    inference_noise_sha256: str
    visibility_audits: tuple[FrozenPosteriorVisibilityAudit, ...]
    receipts: tuple[FrozenPosteriorActionReceipt, ...]


@dataclass(frozen=True, slots=True)
class OfflinePromptTargetRows:
    """Evaluator-only row identities, never admitted by the action request."""

    prompt_name: str
    row_indices: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_name, str) or not self.prompt_name:
            raise ValueError("offline prompt target name must be nonempty")
        if self.row_indices.ndim != 1 or self.row_indices.dtype != torch.long:
            raise ValueError("offline target rows must be a rank-one long tensor")


@dataclass(frozen=True, slots=True)
class OfflineRowSelectivityScore:
    prompt_name: str
    visibility: FrozenPosteriorVisibility
    target_row_indices: torch.Tensor
    row_effect_rms: torch.Tensor
    target_effect_rms: torch.Tensor
    control_effect_rms: torch.Tensor
    target_to_control_ratio: torch.Tensor
    effective_row_count: torch.Tensor


@dataclass(frozen=True, slots=True)
class OfflinePromptSwitchScore:
    prompt_a: str
    prompt_b: str
    visibility: FrozenPosteriorVisibility
    per_sample_difference_in_differences: torch.Tensor
    mean_difference_in_differences: float


def run_frozen_posterior_action_diagnostic(
    readout: FrozenPosteriorActionReadout,
    *,
    snapshot: FrozenPosteriorSnapshot,
    prompts: Sequence[LabelFreePromptVariant],
    controls: ExecutedControlBatch,
    proprioception: torch.Tensor,
    inference_noise: torch.Tensor,
    arms: Sequence[FrozenPosteriorArm],
    visibility_contracts: Sequence[FrozenPosteriorVisibility],
) -> FrozenPosteriorDiagnosticResult:
    """Run prompt-switch and posterior arms through one label-free action ABI."""

    if not callable(readout):
        raise TypeError("frozen-posterior action readout must be callable")
    prompt_items = tuple(prompts)
    arm_items = tuple(arms)
    visibility_items = tuple(visibility_contracts)
    if not prompt_items or len({item.name for item in prompt_items}) != len(prompt_items):
        raise ValueError("diagnostic prompts must be nonempty and uniquely named")
    if not arm_items or len({item.name for item in arm_items}) != len(arm_items):
        raise ValueError("diagnostic posterior arms must be nonempty and uniquely named")
    if not visibility_items or len(set(visibility_items)) != len(visibility_items):
        raise ValueError("diagnostic visibility contracts must be nonempty and unique")
    visibility_audits = tuple(audit_frozen_posterior_visibility(item) for item in visibility_items)
    snapshot.assert_intact()
    frozen_noise = FrozenInferenceNoise.capture(inference_noise)
    controls_digest = _controls_sha256(controls)
    proprio_digest = _tensor_sha256(proprioception)
    receipts = []

    for prompt in prompt_items:
        if prompt.language.token_ids.shape[0] != snapshot.state.batch_size:
            raise ValueError("prompt and frozen posterior batch sizes differ")
        for arm in arm_items:
            if arm.state.layer_rows.shape != snapshot.state.layer_rows.shape:
                raise ValueError("posterior arm shape differs from factual snapshot")
            for visibility in visibility_items:
                snapshot.assert_intact()
                frozen_noise.assert_intact()
                if controls_digest != _controls_sha256(controls):
                    raise RuntimeError("shared diagnostic controls were mutated")
                if proprio_digest != _tensor_sha256(proprioception):
                    raise RuntimeError("shared diagnostic proprioception was mutated")
                request = FrozenPosteriorActionRequest(
                    language=LanguagePromptBatch(
                        token_ids=prompt.language.token_ids.clone(),
                        token_valid=prompt.language.token_valid.clone(),
                    ),
                    controls=_clone_controls(controls),
                    proprioception=proprioception.detach().clone(),
                    posterior=NativeLayerwisePosteriorState(arm.state.layer_rows.detach().clone()),
                    posterior_row_visible=arm.row_visible.clone(),
                    inference_noise=frozen_noise.values.clone(),
                    visibility=visibility,
                )
                request_digest = _action_request_sha256(request)
                with torch.inference_mode():
                    action = readout(request)
                if request_digest != _action_request_sha256(request):
                    raise RuntimeError("action-only diagnostic adapter mutated its request")
                if (
                    not isinstance(action, torch.Tensor)
                    or action.shape != frozen_noise.values.shape
                    or not action.is_floating_point()
                    or not torch.isfinite(action).all()
                    or action.device != frozen_noise.values.device
                    or action.requires_grad
                ):
                    raise ValueError(
                        "action-only readout must return finite actions matching noise shape"
                    )
                receipts.append(
                    FrozenPosteriorActionReceipt(
                        prompt_name=prompt.name,
                        arm_name=arm.name,
                        visibility=visibility,
                        action=action.detach().clone(),
                        posterior_sha256=_tensor_sha256(request.posterior.layer_rows),
                        row_visibility_sha256=_tensor_sha256(request.posterior_row_visible),
                        inference_noise_sha256=_tensor_sha256(request.inference_noise),
                        request_sha256=request_digest,
                    )
                )

    snapshot.assert_intact()
    frozen_noise.assert_intact()
    return FrozenPosteriorDiagnosticResult(
        factual_snapshot_sha256=snapshot.tensor_sha256,
        inference_noise_sha256=frozen_noise.tensor_sha256,
        visibility_audits=visibility_audits,
        receipts=tuple(receipts),
    )


def _diagnostic_receipt_table(
    result: FrozenPosteriorDiagnosticResult,
) -> dict[tuple[str, str, FrozenPosteriorVisibility], FrozenPosteriorActionReceipt]:
    table = {}
    for receipt in result.receipts:
        key = (receipt.prompt_name, receipt.arm_name, receipt.visibility)
        if key in table:
            raise ValueError("diagnostic result contains a duplicate action receipt")
        table[key] = receipt
    return table


def score_offline_row_selectivity(
    result: FrozenPosteriorDiagnosticResult,
    *,
    factual_arm_name: str,
    row_arms: Sequence[FrozenPosteriorArm],
    targets: Sequence[OfflinePromptTargetRows],
    visibility: FrozenPosteriorVisibility,
    epsilon: float = 1e-8,
) -> tuple[OfflineRowSelectivityScore, ...]:
    """Score all-row effects only after every label-free forward has completed."""

    if not isinstance(factual_arm_name, str) or not factual_arm_name:
        raise ValueError("factual arm name must be nonempty")
    if not isinstance(visibility, FrozenPosteriorVisibility):
        raise TypeError("offline scoring requires a parsed visibility contract")
    if not isinstance(epsilon, float) or not 0 < epsilon < 1:
        raise ValueError("offline scoring epsilon must be a float between zero and one")
    arm_items = tuple(row_arms)
    target_items = tuple(targets)
    if not arm_items or not target_items:
        raise ValueError("offline scoring requires row arms and target metadata")
    if len({arm.name for arm in arm_items}) != len(arm_items):
        raise ValueError("offline row arms must be uniquely named")
    if any(not isinstance(target, OfflinePromptTargetRows) for target in target_items):
        raise TypeError("offline target metadata must use OfflinePromptTargetRows")
    capacity = arm_items[0].state.capacity
    arm_by_row = {}
    for arm in arm_items:
        if arm.state.capacity != capacity or arm.row_index is None:
            raise ValueError("offline row arms must declare one common capacity and row index")
        if arm.row_index in arm_by_row:
            raise ValueError("offline row arms contain a duplicate row index")
        arm_by_row[arm.row_index] = arm
    if tuple(sorted(arm_by_row)) != tuple(range(capacity)) or capacity < 2:
        raise ValueError("offline row arms must cover every row and provide at least two rows")
    if len({target.prompt_name for target in target_items}) != len(target_items):
        raise ValueError("offline prompt target metadata must be uniquely named")

    receipt_table = _diagnostic_receipt_table(result)
    scores = []
    for target in target_items:
        factual_key = (target.prompt_name, factual_arm_name, visibility)
        if factual_key not in receipt_table:
            raise ValueError("offline scoring cannot find the factual action receipt")
        factual_action = receipt_table[factual_key].action.float()
        effects = []
        for row_index in range(capacity):
            arm_key = (target.prompt_name, arm_by_row[row_index].name, visibility)
            if arm_key not in receipt_table:
                raise ValueError("offline scoring cannot find one row intervention receipt")
            changed = receipt_table[arm_key].action.float()
            if changed.shape != factual_action.shape or changed.device != factual_action.device:
                raise ValueError("offline action receipts disagree on shape or device")
            effects.append((changed - factual_action).square().mean(dim=(1, 2)).sqrt())
        row_effect = torch.stack(effects, dim=1)
        target_rows = target.row_indices.to(device=row_effect.device)
        if target_rows.shape != (row_effect.shape[0],):
            raise ValueError("offline target row batch differs from action receipts")
        if ((target_rows < 0) | (target_rows >= capacity)).any():
            raise ValueError("offline target row lies outside posterior capacity")
        target_effect = row_effect.gather(1, target_rows[:, None]).squeeze(1)
        control_mask = torch.ones_like(row_effect, dtype=torch.bool)
        control_mask.scatter_(1, target_rows[:, None], False)
        control_effect = (
            torch.where(
                control_mask,
                row_effect,
                torch.full_like(row_effect, torch.nan),
            )
            .nanmedian(dim=1)
            .values
        )
        positive = row_effect.clamp_min(0)
        positive_sum = positive.sum(dim=1)
        effective_rows = torch.where(
            positive_sum > epsilon,
            positive_sum.square() / positive.square().sum(dim=1).clamp_min(epsilon),
            torch.zeros_like(positive_sum),
        )
        scores.append(
            OfflineRowSelectivityScore(
                prompt_name=target.prompt_name,
                visibility=visibility,
                target_row_indices=target_rows.detach().clone(),
                row_effect_rms=row_effect.detach().clone(),
                target_effect_rms=target_effect.detach().clone(),
                control_effect_rms=control_effect.detach().clone(),
                target_to_control_ratio=(target_effect / control_effect.clamp_min(epsilon))
                .detach()
                .clone(),
                effective_row_count=effective_rows.detach().clone(),
            )
        )
    return tuple(scores)


def score_offline_prompt_switch(
    scores: Sequence[OfflineRowSelectivityScore],
    *,
    prompt_a: str,
    prompt_b: str,
) -> OfflinePromptSwitchScore:
    """Compute task-target row DID without invoking another policy forward."""

    score_items = tuple(scores)
    if prompt_a == prompt_b or not prompt_a or not prompt_b:
        raise ValueError("prompt-switch scoring requires two distinct prompt names")
    if any(not isinstance(score, OfflineRowSelectivityScore) for score in score_items):
        raise TypeError("prompt-switch scoring requires OfflineRowSelectivityScore values")
    table = {score.prompt_name: score for score in score_items}
    if len(table) != len(score_items) or prompt_a not in table or prompt_b not in table:
        raise ValueError("prompt-switch scores are missing or ambiguously named")
    score_a = table[prompt_a]
    score_b = table[prompt_b]
    if score_a.visibility is not score_b.visibility:
        raise ValueError("prompt-switch scores must use one visibility contract")
    if score_a.row_effect_rms.shape != score_b.row_effect_rms.shape:
        raise ValueError("prompt-switch row-effect shapes differ")
    rows_a = score_a.target_row_indices.to(score_a.row_effect_rms.device)
    rows_b = score_b.target_row_indices.to(score_a.row_effect_rms.device)
    if rows_a.shape != rows_b.shape or not (rows_a != rows_b).all():
        raise ValueError("prompt-switch targets must name different rows for every sample")
    effect_a_a = score_a.row_effect_rms.gather(1, rows_a[:, None]).squeeze(1)
    effect_a_b = score_a.row_effect_rms.gather(1, rows_b[:, None]).squeeze(1)
    effect_b_a = score_b.row_effect_rms.gather(1, rows_a[:, None]).squeeze(1)
    effect_b_b = score_b.row_effect_rms.gather(1, rows_b[:, None]).squeeze(1)
    did = (effect_a_a - effect_a_b) - (effect_b_a - effect_b_b)
    return OfflinePromptSwitchScore(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        visibility=score_a.visibility,
        per_sample_difference_in_differences=did.detach().clone(),
        mean_difference_in_differences=float(did.mean().item()),
    )

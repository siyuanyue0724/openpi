"""Cached V-JEPA2 causal clips as read-only Molmo action context."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from picf_next.data.calvin import CalvinPICFEvidenceFrame
from picf_next.data.causal_video import build_calvin_causal_video_clip
from picf_next.data.multimodal import ModalityBatchSpec, collate_dense_evidence
from picf_next.data.vjepa2_cache import VJEPA2_CONTEXT_SENSORS, Vjepa2FeatureCache
from picf_next.models.evidence import NativeTokenBank


class _CausalEvidenceRequest(Protocol):
    sample_key: str
    augmentation_seed: int
    evidence_prefix: tuple[CalvinPICFEvidenceFrame, ...]


class CalvinVjepa2CachedContextBuilder:
    """Resolve hash-verified frozen video tokens without posterior correction."""

    def __init__(
        self,
        cache: Vjepa2FeatureCache,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(cache, Vjepa2FeatureCache):
            raise TypeError("V-JEPA2 context builder requires a Vjepa2FeatureCache")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("V-JEPA2 action context requires a floating tensor dtype")
        self.cache = cache
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_specs = tuple(
            ModalityBatchSpec(
                name=modality,
                encoder_contract=cache.encoder_contract,
                token_dim=cache.hidden_size,
                geometry_dim=3,
            )
            for _sensor_key, modality in VJEPA2_CONTEXT_SENSORS
        )

    @property
    def token_dims(self) -> dict[str, int]:
        return {spec.name: spec.token_dim for spec in self.batch_specs}

    @property
    def maximum_source_frames(self) -> int:
        return self.cache.maximum_frames

    def __call__(self, requests: Sequence[_CausalEvidenceRequest]) -> tuple[NativeTokenBank, ...]:
        if not requests:
            raise ValueError("V-JEPA2 context builder requires at least one request")
        samples = []
        for request in requests:
            if not isinstance(request.sample_key, str) or not request.sample_key:
                raise ValueError("V-JEPA2 request sample key cannot be empty")
            if (
                not isinstance(request.augmentation_seed, int)
                or isinstance(request.augmentation_seed, bool)
                or request.augmentation_seed < 0
            ):
                raise ValueError("V-JEPA2 request augmentation seed is invalid")
            if not isinstance(request.evidence_prefix, tuple) or not request.evidence_prefix:
                raise TypeError("V-JEPA2 request requires one causal evidence prefix")
            clips = {
                sensor_key: build_calvin_causal_video_clip(
                    request.evidence_prefix,
                    sensor_key=sensor_key,
                    maximum_frames=self.cache.maximum_frames,
                    tubelet_size=self.cache.tubelet_size,
                )
                for sensor_key, _modality in VJEPA2_CONTEXT_SENSORS
            }
            samples.append(self.cache.evidence_for(request.sample_key, clips))
        banks = collate_dense_evidence(
            samples,
            self.batch_specs,
            device=self.device,
            dtype=self.dtype,
        )
        if any(
            bank.current_measurement_valid is None or bool(bank.current_measurement_valid.any())
            for bank in banks
        ):
            raise RuntimeError("cached V-JEPA2 context attempted to update the posterior")
        return banks

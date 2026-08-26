"""Strict ADR-74 LingBot-native PICF production path.

This namespace is intentionally separate from :mod:`picf_next.unified`, which
retains the historical fixed-fusion implementation as comparison arm D.
"""

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.dense_modalities import (
    NativeDenseModalityBinding,
    dense_modality_bindings_sha256,
    native_modalities_from_dense_evidence,
)
from picf_next.lingbot_native.graph import NativeRole, NativeTokenLayout
from picf_next.lingbot_native.host import (
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    NATIVE_VIDEOMT_QUERY_COUNT,
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    ObjectReadActionIntervention,
    install_lingbot_native_graph,
)
from picf_next.lingbot_native.physical_relations import NativeObjectQueryPosteriorOutput
from picf_next.lingbot_native.modalities import (
    NativeObjectQuerySpatialRelation,
    NativeObjectQuerySpatialSpec,
    NativeModalityBatch,
    NativeModalityOmissionPlan,
    NativeModalitySpec,
    NativeModalityStream,
    NativeRelationSurfaceSpec,
    sample_native_modality_omission,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
    TokenizerDependencyMap,
)
from picf_next.lingbot_native.relations import RelationOutput, SharedRelationReadout
from picf_next.lingbot_native.runtime import LingBotNativePolicyRuntime, NativePolicyStep
from picf_next.lingbot_native.session import (
    NativeObservationBatch,
    NativeSessionConfig,
    NativeSessionManager,
)
from picf_next.lingbot_native.source_mask import (
    QwenPackedPatchMask,
    QwenWholeViewOmission,
    apply_qwen_packed_patch_mask,
    qwen_patch_merger_dependency_map,
    qwen_source_masked_model_inputs,
    qwen_whole_view_omitted_model_inputs,
    sample_qwen_packed_patch_mask,
    sample_qwen_whole_view_omission,
)
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativePosteriorState,
    NativeVidEoMTPairedPosteriorState,
)
from picf_next.lingbot_native.temporal import (
    NativeLaneConfig,
    NativeTrainingLaneBank,
    TemporalCostProfile,
    TemporalEstimatorConfig,
    TemporalWorkload,
)
from picf_next.lingbot_native.training import (
    NativeLocalBPTTResult,
    NativeLocalBPTTStep,
    NativeOmittedModalityPrediction,
    NativePolicyForwardResult,
    NativeSourceMaskedPrediction,
    NativeTrainingLaneCoordinator,
    run_native_local_bptt,
    run_native_omitted_image_view_training_forward,
    run_native_omitted_modality_training_forward,
    run_native_policy_relation_training_forward,
    run_native_policy_training_forward,
    run_native_relation_local_bptt,
    run_native_source_masked_training_forward,
)

__all__ = [
    "ExecutedControlBatch",
    "LingBotNativeContext",
    "LingBotNativeGraph",
    "LingBotNativeGraphConfig",
    "LingBotNativePolicyRuntime",
    "LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR",
    "NATIVE_VIDEOMT_QUERY_COUNT",
    "NATIVE_VIDEOMT_QUERY_POSTERIOR",
    "ObjectReadActionIntervention",
    "NativeLaneConfig",
    "NativeLocalBPTTResult",
    "NativeLocalBPTTStep",
    "NativeLayerwisePosteriorState",
    "NativeModalityBatch",
    "NativeDenseModalityBinding",
    "NativeModalityOmissionPlan",
    "NativeModalitySpec",
    "NativeModalityStream",
    "NativeObjectQuerySpatialRelation",
    "NativeObjectQuerySpatialSpec",
    "NativeObjectQueryPosteriorOutput",
    "NativeRelationSurfaceSpec",
    "dense_modality_bindings_sha256",
    "native_modalities_from_dense_evidence",
    "NativeOmittedModalityPrediction",
    "NativePolicyForwardResult",
    "NativePolicyStep",
    "NativePosteriorState",
    "NativeVidEoMTPairedPosteriorState",
    "NativePredictionRequest",
    "NativeSourceMaskedPrediction",
    "NativeTrainingLaneBank",
    "NativeTrainingLaneCoordinator",
    "NativeObservationBatch",
    "NativeRole",
    "NativeSessionConfig",
    "NativeSessionManager",
    "NativeTokenLayout",
    "PredictionEvidence",
    "PredictionSource",
    "QwenPackedPatchMask",
    "QwenWholeViewOmission",
    "RelationOutput",
    "SharedRelationReadout",
    "TokenizerDependencyMap",
    "TemporalCostProfile",
    "TemporalEstimatorConfig",
    "TemporalWorkload",
    "apply_qwen_packed_patch_mask",
    "install_lingbot_native_graph",
    "qwen_patch_merger_dependency_map",
    "qwen_source_masked_model_inputs",
    "qwen_whole_view_omitted_model_inputs",
    "run_native_local_bptt",
    "run_native_omitted_image_view_training_forward",
    "run_native_omitted_modality_training_forward",
    "run_native_policy_relation_training_forward",
    "run_native_policy_training_forward",
    "run_native_relation_local_bptt",
    "run_native_source_masked_training_forward",
    "sample_native_modality_omission",
    "sample_qwen_packed_patch_mask",
    "sample_qwen_whole_view_omission",
]

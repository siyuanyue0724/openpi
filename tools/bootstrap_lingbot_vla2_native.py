#!/usr/bin/env python3
"""Verify and prepare the exact ADR-74 LingBot-native source overlay."""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_ID,
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_ID,
        QWEN_PROCESSOR_REVISION,
        checkpoint_download_command,
        processor_download_command,
        validate_checkpoint,
        validate_processor,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_ID,
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_ID,
        QWEN_PROCESSOR_REVISION,
        checkpoint_download_command,
        processor_download_command,
        validate_checkpoint,
        validate_processor,
    )

LINGBOT_NATIVE_SOURCE_URL = "https://github.com/Robbyant/lingbot-vla-v2.git"
LINGBOT_NATIVE_SOURCE_COMMIT = "2838c1862bbec1ea47942fb61512130f635eb595"
UTILS3D_SOURCE_URL = "https://github.com/EasternJournalist/utils3d.git"
UTILS3D_SOURCE_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"
PATCH_RELATIVE_PATH = Path("references/patches/lingbot_vla2_picf_native.patch")
MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH = Path(
    "references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch"
)
MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH = Path(
    "references/patches/lingbot_vla2_muon_mixed_device_megabatch.patch"
)
SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH = Path(
    "references/patches/lingbot_vla2_selective_frozen_vision_offload.patch"
)
FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH = Path(
    "references/patches/lingbot_vla2_frozen_visual_root_offload.patch"
)
SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH = Path(
    "references/patches/lingbot_vla2_selective_trainable_vision_offload.patch"
)
SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH = Path(
    "adr221/patches/lingbot_fsdp2_selective_class_cpu_offload.patch"
)
SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH = Path(
    "adr221/patches/lingbot_fsdp2_selective_class_after_trainable_vision_offload.patch"
)
VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH = Path(
    "adr224/patches/lingbot_fsdp2_vlm_selective_class_after_trainable_vision_offload.patch"
)
CHECKOUT_RELATIVE_PATH = Path("references/source_checkouts/lingbot-vla-v2-adr74")
UTILS3D_CHECKOUT_RELATIVE_PATH = Path("references/source_checkouts/utils3d-3fab839f")
MODEL_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py")
ACTION_DECODER_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py")
TEXT_DECODER_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py")
QWEN25_TEXT_DECODER_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/qwenvl_in_vla.py")
PARALLEL_SOURCE = Path("lingbotvla/distributed/torch_parallelize.py")
CHECKPOINTER_SOURCE = Path("lingbotvla/checkpoint/checkpointer.py")
MUON_SOURCE = Path("lingbotvla/optim/muon.py")
PATCHED_SOURCES = (
    CHECKPOINTER_SOURCE,
    PARALLEL_SOURCE,
    MODEL_SOURCE,
    ACTION_DECODER_SOURCE,
    TEXT_DECODER_SOURCE,
    QWEN25_TEXT_DECODER_SOURCE,
    MUON_SOURCE,
)
LINGBOT_DEPTH_SOURCE = Path("lingbotvla/models/vla/vision_models/lingbot-depth")
MOGE_SOURCE = Path("lingbotvla/models/vla/vision_models/MoGe")
PATCH_SHA256 = "3879f68206c1ed2c842a5d2ab7bd96dfb9a488d0ca68c118a531e2464c130c24"
MUON_COLLECTIVE_HOTFIX_SHA256 = (
    "cf1cf8ab41baea50d41e2030977e78fae2ff9742b7c5a7c70c6c40838dcbeab3"
)
MUON_MIXED_DEVICE_MEGABATCH_SHA256 = (
    "3b59e07a41617f627f67e013e11952b25a95f69bdb7bd4e8441880402fc32f56"
)
SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256 = (
    "783a7cf21cfd439e418f14cc2e6a0770d264268895dd9570800537bc908e9090"
)
FROZEN_VISUAL_ROOT_OFFLOAD_SHA256 = (
    "cb734d999459bfa065f619108e64aa3fa780650442efdaef24b740d622fcbf70"
)
SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256 = (
    "e0ca8b0587ebe6e38b16b4ff83a0298d59918904e5c8839ea33dd20015359fde"
)
SELECTIVE_CLASS_CPU_OFFLOAD_SHA256 = (
    "3ba693ac4ac2158bf60756aaee067fbd368ae6e2770ab340838fa5b63bb226fa"
)
SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256 = (
    "7035b239f6c94ae58ac7bd66969ce9b2a5b1676ce105ed95e36e8597f11734d4"
)
VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256 = (
    "7634367ee5dbfe08161c405a25a4e44014d2fdf3a9bc6ecb6cef5331840a93c9"
)
PATCHED_CHECKPOINTER_SHA256 = "ac7bd7e4bcbf6d92d095800a80d44d4340c28caae86a8743f5c77b08f9be316e"
PATCHED_MODEL_SHA256 = "c4fc62f391404dab155a388d48a9f285fffd65409b0c1c55e88edf1332c35937"
PATCHED_PARALLEL_SHA256 = "04e902bcee08ae6e0a571d315db5afdf10483988a9beaa6c278899bc33d6d288"
PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256 = (
    "de3cd1cb59cd25d36b74771fff1b23dd318ac2d54f6639d2ef71518ccace4e47"
)
PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256 = (
    "f0836993c3dcd1b43dfe03c8c78f2360b56f2d6fe22b5b3bf54355474d263def"
)
PATCHED_PARALLEL_WITH_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256 = (
    "d3233e5ec507a75c50778e10edb355e89d4d082b34ad989d9caccab51ce63c6c"
)
PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256 = (
    "d7d3ba3ced4ff53d82f34a67c6541afa0f5b011a3acd33fb3d3e89cfee8b7f3f"
)
PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_VLM_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256 = (
    "a02db3cfa6fcad6a4704e3f6efdda29d530f8aa1c473cfc889831ee71e86e635"
)
PATCHED_MUON_SHA256 = "1ac4053236ab0707cbd6b57e7fd1279e1ea39d98566c01e24641cf9e024bae26"
PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256 = (
    "cc3cf6de187eacaddb0b5a964f64ab305693b28529a0e5e9f46e149ffc4a81e4"
)
PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256 = (
    "98a1de761c2f8f8e1c041d3c0e4b053b2aef2df3b84a7c34fc1c5973c10553b5"
)
PATCHED_ACTION_DECODER_SHA256 = "b100e1975ed87e5a23564bb7ad390667b8fc42bce38a8b928ede067d534d875e"
PATCHED_TEXT_DECODER_SHA256 = "e1961fbb369d1c405ac0e67c16652d901bfad5e143abc9d9ae679b014c3fb649"
PATCHED_QWEN25_TEXT_DECODER_SHA256 = (
    "bc26161e8698ca858f2ecfc6428c2e9b353ebfdbe33e2319934071eddfd26271"
)
LINGBOT_NATIVE_REQUIREMENTS_SHA256 = (
    "4bea8eca2e5e81107332947fe38d9a2787bc6a8fe4d3f875fa7e3d028f48993d"
)
LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256 = (
    "95333d7165ab18c2d31c62a0d07c93e8b20f024fa0c8488fe7f386038592400d"
)
LINGBOT_NATIVE_RUNTIME_EXTRAS = {
    "flash-attn": "2.8.3",
    "lerobot": "0.4.2",
    "numpydantic": "1.9.0",
}
LINGBOT_NATIVE_AUDIT_TOOLS = {
    "iniconfig": "2.3.0",
    "pluggy": "1.6.0",
    "pygments": "2.20.0",
    "pytest": "9.1.1",
    "ruff": "0.15.21",
}
CALVIN_SOURCE_COMMIT = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_SOURCE_COMMIT = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CALVIN_ENV_REQUIREMENTS_SHA256 = "b6b3761ddb6491d9e03a2b6f4b61a38b0ee1bb499be888f5c24c8f96479fce13"
CALVIN_ENV_SETUP_SHA256 = "25468602eab059c3446b5d9c7e2ac83677a28c326ac3ac35f9eb87f891085488"
# Exact additions not already frozen by LingBot's immutable requirement files.
# opencv-python-headless is intentionally provided by LingBot instead of the
# GUI-enabled opencv-python entry in CALVIN's otherwise unpinned requirements.
CALVIN_OFFLINE_RUNTIME_EXTRAS = {
    "antlr4-python3-runtime": "4.9.3",
    "cloudpickle": "3.1.2",
    "colorlog": "6.12.0",
    "gitpython": "3.1.55",
    "gym": "0.26.2",
    "gym-notices": "0.1.0",
    "hydra-colorlog": "1.2.0",
    "hydra-core": "1.3.2",
    "llvmlite": "0.44.0",
    "numba": "0.61.2",
    "numpy-quaternion": "2024.0.13",
    "pybullet": "3.2.7",
}

_REQUIRED_UPSTREAM_FRAGMENTS = (
    "class QwenvlWithExpertV2Model",
    "class FlowMatchingV2",
    "def embed_prefix(",
    "def sample_actions(",
    "def predict_velocity(",
    "class LingbotVlaV2Policy",
)
_REQUIRED_MODEL_PATCH_FRAGMENTS = (
    "def embed_language_and_special_tokens(",
    "torch.cat((flat_language_tokens, special_tokens))",
    "(cfg.vision_start_token_id, cfg.vision_end_token_id)",
    "self.picf_native_graph = None",
    "def set_picf_native_graph",
    "def _bind_picf_native_prefix",
    "picf_native_context=None",
    "self.picf_native_graph.prepare_joint_inputs(",
    "self.picf_native_graph.layerwise_qk_address_bias(",
    "self.picf_native_graph.layerwise_memory_inputs(",
    "self.picf_native_graph.record_layerwise_posterior(",
    "layer_attention_mask = torch.cat(",
    "_key_value_len = key_states.shape[1]",
    "self.picf_native_graph.requires_intermediate_relation(",
    "self.picf_native_graph.record_intermediate_relation(",
    "self.picf_native_graph.finalize_joint_outputs(",
    "picf_native_context.bind_native_prefix(",
    "native_valid=prefix_pad_masks.bool()",
    "visual_sensor_mask=visual_pos_masks.bool()",
    "host_current_mask=host_current_mask",
    "host_future_mask=host_future_mask",
    'current_query_names = {"current_depth"}',
    'future_query_names = {"future_video_cls", "future_video", "future_depth"}',
    "unexpected_query_names = set(query_spans)",
    "PICF refuses unclassified official query spans",
    "visual_boundary_mask = (",
    "visual_boundary_mask[:, language_start:] = False",
    "visual_boundary_mask=visual_boundary_mask",
    "native_past_key_values",
    "expanded_past_key_values",
    "compact_lingbot_action_cache",
    "picf_action_attention_callback=None",
    "layer_index=layer_idx",
    "layer_count=num_layers",
    "suffix_count=suffix_len if picf_action_attention_callback",
    "action_layout = compact_cache.action_attention_layout",
    "def _picf_cached_training_forward(",
    "Train action through the same exact-native cache ABI used at inference.",
    "native_outputs_embeds, suffix_out, router_logits_list",
    "past_key_values = compact_cache.past_key_values",
    "prefix_pad_masks = compact_cache.valid",
    "prefix_position_ids = compact_cache.position_ids",
    "prefix_position_pad_masks = compact_cache.position_valid",
    "prefix_position_pad_masks = prefix_pad_masks",
    "prefix_position_pad_masks=prefix_position_pad_masks",
    "position and visibility prefix masks must have identical shape",
    "and picf_native_context is None",
    "native_prefix_len=native_prefix_len",
    "def picf_native_prior_forward(",
    "def picf_native_observation_forward(",
    "def picf_native_frozen_posterior_action_forward(",
    "run_registered_lingbot_frozen_posterior_action(self, request)",
    "inputs_embeds=[prefix_embs, None]",
    "use_cache=False",
    "compute_alignment_losses=True",
    "self.config.align_params != {} and compute_alignment_losses",
    "compute_alignment_losses=compute_alignment_losses",
    "picf_native_context.root_output_tensors()",
)
_REQUIRED_PARALLEL_PATCH_FRAGMENTS = (
    "CPUOffloadPolicy(pin_memory=False)",
    'user_fsdp_kwargs = dict(kwargs.pop("fsdp_kwargs", {}))',
    'user_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
    "enable_shared_embedding_offload = kwargs.pop(",
    "full-model and shared-embedding FSDP2 CPU offload are mutually exclusive",
    'shared_embedding_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
    "model._lingbot_fsdp2_selective_cpu_modules = selective_cpu_modules",
    "def _collect_vlm_fsdp_layers",
    'layers: dict[str, list[tuple[str, nn.Module]]] = {"text": [], "vision": []}',
    "model._lingbot_vlm_fsdp2_topology = topology",
    "sharding only the root would create a large, non-overlapped AllGather",
)
_REQUIRED_DECODER_DTYPE_PATCH_FRAGMENTS = (
    "parent-owned norm",
    "param_dtype = self.input_layernorm.weight.dtype",
    "qk_input_bias: Optional[torch.Tensor] = None",
    "qk_hidden_states = hidden_states + qk_input_bias.to(hidden_states.dtype)",
    "self.self_attn.q_proj(qk_hidden_states)",
    "self.self_attn.k_proj(qk_hidden_states)",
    "self.self_attn.v_proj(hidden_states)",
)
_REQUIRED_CHECKPOINTER_PATCH_FRAGMENTS = (
    "StateDictOptions",
    "options=StateDictOptions(strict=False)",
    "prune_synthetic_optimizer_state_from_dcp_template",
    "optimizer_reader.read_metadata().state_dict_metadata",
    "planner=DefaultLoadPlanner(allow_partial_load=True)",
)
_REQUIRED_PATCH_FRAGMENTS = (
    *_REQUIRED_CHECKPOINTER_PATCH_FRAGMENTS,
    *_REQUIRED_MODEL_PATCH_FRAGMENTS,
    *_REQUIRED_PARALLEL_PATCH_FRAGMENTS,
    *_REQUIRED_DECODER_DTYPE_PATCH_FRAGMENTS,
)


def _validate_prefix_binding_reachability(source_text: str) -> None:
    if source_text.count("def _bind_picf_native_prefix(") != 1:
        raise ValueError("native prefix binding must have exactly one implementation")
    if source_text.count("self._bind_picf_native_prefix(") != 3:
        raise ValueError(
            "native prefix binding must be called by training, observation and inference"
        )
    if source_text.count("def picf_native_observation_forward(") != 2:
        raise ValueError("observation-only forward requires flow and policy root implementations")
    prefix_only_calls = source_text.count("inputs_embeds=[prefix_embs, None]")
    if prefix_only_calls not in (4, 5):
        raise ValueError(
            "observation plus native/PICF training and inference must use all prefix-only paths"
        )


_FORBIDDEN_PATCH_FRAGMENTS = (
    "action_layer_adapter",
    "set_action_layer_adapter",
    "unified_belief_graph",
    "lifecycle",
    "semantic_scorer",
)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patch_paths(patch_text: str) -> set[Path]:
    return {
        Path(line.removeprefix("+++ b/"))
        for line in patch_text.splitlines()
        if line.startswith("+++ b/")
    }


def _validate_patch(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot-native patch is absent: {patch_path}")
    if _sha256(patch_path) != PATCH_SHA256:
        raise ValueError("LingBot-native patch digest differs from the approved artifact")
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != set(PATCHED_SOURCES):
        raise ValueError("LingBot-native patch modifies undeclared source paths")
    missing = [fragment for fragment in _REQUIRED_PATCH_FRAGMENTS if fragment not in patch_text]
    if missing:
        raise ValueError(f"LingBot-native patch omits required fragments: {missing}")
    forbidden = [fragment for fragment in _FORBIDDEN_PATCH_FRAGMENTS if fragment in patch_text]
    if forbidden:
        raise ValueError(f"LingBot-native patch contains forbidden legacy fragments: {forbidden}")
    _validate_prefix_binding_reachability(patch_text)


def _validate_muon_collective_hotfix(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot Muon collective hotfix is absent: {patch_path}")
    if _sha256(patch_path) != MUON_COLLECTIVE_HOTFIX_SHA256:
        raise ValueError("LingBot Muon collective hotfix digest differs from the approved artifact")
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {MUON_SOURCE}:
        raise ValueError("LingBot Muon collective hotfix modifies undeclared source paths")
    required = (
        "other_params: List[Tuple[Tensor, str, bool]]",
        "if not has_grad and kind != _KIND_MOE_GATHER_3D",
        "other_params.append((p, kind, has_grad))",
        "for p, kind, has_grad in other_params",
        "update_local = torch.zeros_like(p_local)",
        "if not has_grad:",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(f"LingBot Muon collective hotfix omits required fragments: {missing}")


def _validate_muon_mixed_device_megabatch(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot Muon mixed-device patch is absent: {patch_path}")
    if _sha256(patch_path) != MUON_MIXED_DEVICE_MEGABATCH_SHA256:
        raise ValueError(
            "LingBot Muon mixed-device patch digest differs from the approved artifact"
        )
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {MUON_SOURCE}:
        raise ValueError("LingBot Muon mixed-device patch modifies undeclared source paths")
    required = (
        "local_parameter = p.to_local() if isinstance(p, DTensor) else p",
        "key = (global_shape, str(p.dtype), local_parameter.device.type)",
        'if stacked_local.device.type == "cpu":',
        'device=torch.device("cuda", torch.cuda.current_device())',
        "ortho_local.to(device=p_local.device, dtype=p_local.dtype)",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(
            f"LingBot Muon mixed-device patch omits required fragments: {missing}"
        )


def _validate_selective_frozen_vision_offload(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot selective frozen-vision patch is absent: {patch_path}")
    if _sha256(patch_path) != SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256:
        raise ValueError(
            "LingBot selective frozen-vision patch digest differs from the approved artifact"
        )
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {PARALLEL_SOURCE}:
        raise ValueError("LingBot selective frozen-vision patch modifies undeclared sources")
    required = (
        "enable_frozen_vision_offload = kwargs.pop(",
        "full-model and selective FSDP2 CPU offload are mutually exclusive",
        'if kind == "vision" and enable_frozen_vision_offload:',
        "selective vision offload requires every vision block to be frozen",
        'kind_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
        "fully_shard(layer, **kind_fsdp_kwargs)",
        "selective_cpu_modules.extend(paths)",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(
            f"LingBot selective frozen-vision patch omits required fragments: {missing}"
        )


def _validate_frozen_visual_root_offload(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot frozen visual-root patch is absent: {patch_path}")
    if _sha256(patch_path) != FROZEN_VISUAL_ROOT_OFFLOAD_SHA256:
        raise ValueError(
            "LingBot frozen visual-root patch digest differs from the approved artifact"
        )
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {PARALLEL_SOURCE}:
        raise ValueError("LingBot frozen visual-root patch modifies undeclared sources")
    required = (
        "visual_root = vlm.model.visual",
        "selective vision offload requires the complete visual root to be frozen",
        "fully_shard(visual_root, **kind_fsdp_kwargs)",
        "selective_cpu_modules.append(visual_root_path)",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(
            f"LingBot frozen visual-root patch omits required fragments: {missing}"
        )


def _validate_selective_trainable_vision_offload(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot selective trainable-vision patch is absent: {patch_path}")
    if _sha256(patch_path) != SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256:
        raise ValueError(
            "LingBot selective trainable-vision patch digest differs from the approved artifact"
        )
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {PARALLEL_SOURCE}:
        raise ValueError("LingBot selective trainable-vision patch modifies undeclared sources")
    required = (
        "enable_trainable_vision_offload = kwargs.pop(",
        "frozen and trainable selective vision offload are mutually exclusive",
        "enable_vision_offload = (",
        'if kind == "vision" and enable_vision_offload:',
        "if enable_frozen_vision_offload and any(",
        "vision_offload_mode = (",
        "FSDP2 did not augment the complete visual root.",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(
            f"LingBot selective trainable-vision patch omits required fragments: {missing}"
        )


def _validate_selective_class_cpu_offload(
    patch_path: Path,
    *,
    expected_digest: str = SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
) -> None:
    if not patch_path.is_file():
        raise ValueError(f"LingBot selective-class CPU offload patch is absent: {patch_path}")
    if _sha256(patch_path) != expected_digest:
        raise ValueError(
            "LingBot selective-class CPU offload patch digest differs from the "
            "approved artifact"
        )
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {PARALLEL_SOURCE}:
        raise ValueError(
            "LingBot selective-class CPU offload patch modifies undeclared sources"
        )
    required = (
        'kwargs.pop("selective_cpu_module_classes", ())',
        "selective_cpu_module_classes must contain nonempty class names",
        "full-model and selective-class FSDP2 CPU offload are mutually exclusive",
        "selective_class_offload_policy = CPUOffloadPolicy(pin_memory=False)",
        "module.__class__.__name__ in selective_cpu_module_classes",
        'module_fsdp_kwargs["offload_policy"] = (',
        "model._lingbot_fsdp2_selective_cpu_module_classes = (",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(
            f"LingBot selective-class CPU offload patch omits required fragments: {missing}"
        )


def _validate_vlm_selective_class_cpu_offload(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise ValueError(
            f"LingBot VLM selective-class CPU offload patch is absent: {patch_path}"
        )
    if _sha256(patch_path) != VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256:
        raise ValueError(
            "LingBot VLM selective-class CPU offload patch digest differs from the "
            "approved artifact"
        )
    patch_text = patch_path.read_text()
    if _patch_paths(patch_text) != {PARALLEL_SOURCE}:
        raise ValueError(
            "LingBot VLM selective-class CPU offload patch modifies undeclared sources"
        )
    required = (
        "layer_fsdp_kwargs = kind_fsdp_kwargs",
        "if layer.__class__.__name__ in selective_cpu_module_classes",
        "layer_fsdp_kwargs = dict(kind_fsdp_kwargs)",
        'layer_fsdp_kwargs["offload_policy"] = (',
        "selective_class_offload_policy",
        "fully_shard(layer, **layer_fsdp_kwargs)",
    )
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(
            "LingBot VLM selective-class CPU offload patch omits required fragments: "
            f"{missing}"
        )


def detect_native_patch_state(checkout: Path, patch_path: Path) -> str:
    forward = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "apply",
            "--reverse",
            "--check",
            str(patch_path),
        ],
        capture_output=True,
        text=True,
    )
    if forward.returncode == 0 and reverse.returncode != 0:
        return "baseline"
    if reverse.returncode == 0 and forward.returncode != 0:
        return "applied"
    raise ValueError("LingBot checkout is neither exact baseline nor exact native-patched state")


def _dirty_paths(checkout: Path) -> set[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    return {line[3:] for line in output.splitlines() if line}


def _purge_generated_python_bytecode(checkout: Path) -> None:
    """Remove import-created bytecode without accepting any unbound executable.

    Distributed launchers may validate the same external checkout concurrently.
    Missing paths are therefore idempotent success, while every path that still
    exists is inspected with ``lstat`` before removal so symlinks and unrelated
    artifacts remain fail-closed.
    """

    caches = sorted(
        (path for path in checkout.rglob("__pycache__") if path.is_dir() or path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for cache in caches:
        for _attempt in range(8):
            try:
                cache_mode = cache.lstat().st_mode
            except FileNotFoundError:
                break
            if stat.S_ISLNK(cache_mode):
                raise ValueError(f"LingBot checkout contains a symlinked bytecode cache: {cache}")
            if not stat.S_ISDIR(cache_mode):
                raise ValueError(f"LingBot bytecode cache is not a directory: {cache}")
            try:
                children = list(cache.iterdir())
            except FileNotFoundError:
                break
            for child in children:
                try:
                    child_mode = child.lstat().st_mode
                except FileNotFoundError:
                    continue
                if (
                    stat.S_ISLNK(child_mode)
                    or not stat.S_ISREG(child_mode)
                    or child.suffix not in {".pyc", ".pyo"}
                ):
                    raise ValueError(
                        f"LingBot bytecode cache contains an unknown artifact: {child}"
                    )
                try:
                    child.unlink()
                except FileNotFoundError:
                    continue
            try:
                cache.rmdir()
            except FileNotFoundError:
                break
            except OSError as error:
                if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                    continue
                raise
            else:
                break
        else:
            raise RuntimeError(f"LingBot bytecode cache remained populated during cleanup: {cache}")


def _export_commit(checkout: Path, destination: Path) -> None:
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "archive",
            "--format=tar",
            LINGBOT_NATIVE_SOURCE_COMMIT,
        ],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def _validate_patched_model_source(
    source_path: Path,
    *,
    expected_digest: str = PATCHED_MODEL_SHA256,
    additional_required_fragments: tuple[str, ...] = (),
) -> str:
    source_text = source_path.read_text()
    compile(source_text, str(source_path), "exec")
    missing = [
        fragment for fragment in _REQUIRED_MODEL_PATCH_FRAGMENTS if fragment not in source_text
    ]
    if missing:
        raise ValueError(f"patched LingBot source omits native hooks: {missing}")
    additional_missing = [
        fragment for fragment in additional_required_fragments if fragment not in source_text
    ]
    if additional_missing:
        raise ValueError(
            f"patched LingBot source omits required extension hooks: {additional_missing}"
        )
    forbidden = [fragment for fragment in _FORBIDDEN_PATCH_FRAGMENTS if fragment in source_text]
    if forbidden:
        raise ValueError(f"patched LingBot source contains legacy hooks: {forbidden}")
    if "def embed_special_token(" in source_text:
        raise ValueError("patched LingBot source retains duplicate shared embedding lookups")
    _validate_prefix_binding_reachability(source_text)
    digest = _sha256(source_path)
    if digest != expected_digest:
        raise ValueError("patched LingBot model source digest differs from approved replay")
    return digest


def _validate_patched_parallel_source(
    source_path: Path,
    *,
    expected_digest: str = PATCHED_PARALLEL_SHA256,
    require_frozen_vision_offload: bool = False,
    require_trainable_vision_offload: bool = False,
    require_selective_class_cpu_offload: bool = False,
    require_vlm_selective_class_cpu_offload: bool = False,
) -> str:
    source_text = source_path.read_text()
    compile(source_text, str(source_path), "exec")
    required = (
        "CPUOffloadPolicy(pin_memory=False)",
        'user_fsdp_kwargs = dict(kwargs.pop("fsdp_kwargs", {}))',
        'user_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
        "enable_shared_embedding_offload = kwargs.pop(",
        'shared_embedding_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
        "model._lingbot_fsdp2_selective_cpu_modules =",
        '"enable_fsdp_offload and fsdp_kwargs.offload_policy are mutually exclusive"',
        "**user_fsdp_kwargs",
        "def _collect_vlm_fsdp_layers",
        "model._lingbot_vlm_fsdp2_topology = topology",
        'for kind in ("text", "vision")',
        'if not hasattr(layer, "reshard") or not hasattr(layer, "unshard")',
    )
    missing = [fragment for fragment in required if fragment not in source_text]
    if missing:
        raise ValueError(f"patched LingBot parallel source omits FSDP2 offload: {missing}")
    if require_frozen_vision_offload and require_trainable_vision_offload:
        raise ValueError("parallel source cannot require both vision offload modes")
    if require_vlm_selective_class_cpu_offload and not (
        require_trainable_vision_offload and require_selective_class_cpu_offload
    ):
        raise ValueError(
            "VLM selective-class offload requires trainable-vision and "
            "selective-class offload"
        )
    if require_selective_class_cpu_offload:
        selective_class_required = (
            'kwargs.pop("selective_cpu_module_classes", ())',
            "selective_cpu_module_classes must contain nonempty class names",
            "full-model and selective-class FSDP2 CPU offload are mutually exclusive",
            "selective_class_offload_policy = CPUOffloadPolicy(pin_memory=False)",
            "module.__class__.__name__ in selective_cpu_module_classes",
            'module_fsdp_kwargs["offload_policy"] = (',
            "selective_class_offload_policy",
            "model._lingbot_fsdp2_selective_cpu_module_classes = (",
        )
        missing = [
            fragment for fragment in selective_class_required if fragment not in source_text
        ]
        if missing:
            raise ValueError(
                f"patched LingBot parallel source omits selective-class offload: {missing}"
            )
        if source_text.count(
            "module.__class__.__name__ in selective_cpu_module_classes"
        ) != 1:
            raise ValueError(
                "selective-class CPU offload has an unexpected FSDP dispatch count"
            )
        if (
            not require_frozen_vision_offload
            and not require_trainable_vision_offload
            and source_text.count("fully_shard(layer, **mp_fsdp_kwargs)") != 1
        ):
            raise ValueError("Qwen-VL FSDP2 blocks must retain one unified sharding loop")
    if require_vlm_selective_class_cpu_offload:
        vlm_selective_class_required = (
            "layer_fsdp_kwargs = kind_fsdp_kwargs",
            "if layer.__class__.__name__ in selective_cpu_module_classes",
            "layer_fsdp_kwargs = dict(kind_fsdp_kwargs)",
            'layer_fsdp_kwargs["offload_policy"] = (',
            "fully_shard(layer, **layer_fsdp_kwargs)",
        )
        missing = [
            fragment
            for fragment in vlm_selective_class_required
            if fragment not in source_text
        ]
        if missing:
            raise ValueError(
                "patched LingBot parallel source omits VLM selective-class offload: "
                f"{missing}"
            )
        if source_text.count("fully_shard(layer, **layer_fsdp_kwargs)") != 1:
            raise ValueError(
                "Qwen-VL blocks require one VLM selective-class sharding loop"
            )
    if require_frozen_vision_offload:
        frozen_vision_required = (
            "enable_frozen_vision_offload = kwargs.pop(",
            "full-model and selective FSDP2 CPU offload are mutually exclusive",
            'if kind == "vision" and enable_frozen_vision_offload:',
            "selective vision offload requires every vision block to be frozen",
            'kind_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
            "fully_shard(layer, **kind_fsdp_kwargs)",
            "visual_root = vlm.model.visual",
            "selective vision offload requires the complete visual root to be frozen",
            "fully_shard(visual_root, **kind_fsdp_kwargs)",
            "selective_cpu_modules.append(visual_root_path)",
            "model._lingbot_fsdp2_selective_cpu_modules = tuple(selective_cpu_modules)",
        )
        missing = [
            fragment for fragment in frozen_vision_required if fragment not in source_text
        ]
        if missing:
            raise ValueError(
                f"patched LingBot parallel source omits frozen-vision offload: {missing}"
            )
        if source_text.count("fully_shard(layer, **kind_fsdp_kwargs)") != 1:
            raise ValueError("Qwen-VL blocks require one selective text/vision sharding loop")
    if require_trainable_vision_offload:
        trainable_vision_required = (
            "enable_frozen_vision_offload = kwargs.pop(",
            "enable_trainable_vision_offload = kwargs.pop(",
            "frozen and trainable selective vision offload are mutually exclusive",
            "enable_vision_offload = (",
            "full-model and selective FSDP2 CPU offload are mutually exclusive",
            'if kind == "vision" and enable_vision_offload:',
            "if enable_frozen_vision_offload and any(",
            'kind_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
            (
                "fully_shard(layer, **layer_fsdp_kwargs)"
                if require_vlm_selective_class_cpu_offload
                else "fully_shard(layer, **kind_fsdp_kwargs)"
            ),
            "visual_root = vlm.model.visual",
            "vision_offload_mode = (",
            "fully_shard(visual_root, **kind_fsdp_kwargs)",
            "selective_cpu_modules.append(visual_root_path)",
            "model._lingbot_fsdp2_selective_cpu_modules = tuple(selective_cpu_modules)",
        )
        missing = [
            fragment for fragment in trainable_vision_required if fragment not in source_text
        ]
        if missing:
            raise ValueError(
                f"patched LingBot parallel source omits trainable-vision offload: {missing}"
            )
        trainable_loop = (
            "fully_shard(layer, **layer_fsdp_kwargs)"
            if require_vlm_selective_class_cpu_offload
            else "fully_shard(layer, **kind_fsdp_kwargs)"
        )
        if source_text.count(trainable_loop) != 1:
            raise ValueError("Qwen-VL blocks require one selective text/vision sharding loop")
    if not require_frozen_vision_offload and not require_trainable_vision_offload:
        if (
            "full-model and shared-embedding FSDP2 CPU offload are mutually exclusive"
            not in source_text
        ):
            raise ValueError("patched LingBot parallel source changed its base offload contract")
        if source_text.count("fully_shard(layer, **mp_fsdp_kwargs)") != 1:
            raise ValueError("Qwen-VL FSDP2 blocks must use one unified text/vision sharding loop")
    if source_text.count("**user_fsdp_kwargs") != 2:
        raise ValueError("FSDP2 offload policy must cover nested and root parameter groups")
    if source_text.count("fully_shard(shared_embedding, **shared_embedding_fsdp_kwargs)") != 1:
        raise ValueError("shared embedding must have exactly one selective FSDP2 group")
    digest = _sha256(source_path)
    if digest != expected_digest:
        raise ValueError("patched LingBot parallel source digest differs from approved replay")
    return digest


def _validate_patched_checkpointer_source(source_path: Path) -> str:
    source_text = source_path.read_text()
    compile(source_text, str(source_path), "exec")
    missing = [
        fragment
        for fragment in _REQUIRED_CHECKPOINTER_PATCH_FRAGMENTS
        if fragment not in source_text
    ]
    if missing:
        raise ValueError(f"patched LingBot checkpointer omits DCP closure: {missing}")
    if "Skip loading Optimizer" in source_text:
        raise ValueError("patched LingBot checkpointer may not silently skip optimizer restore")
    digest = _sha256(source_path)
    if digest != PATCHED_CHECKPOINTER_SHA256:
        raise ValueError("patched LingBot checkpointer digest differs from approved replay")
    return digest


def _validate_patched_decoder_source(source_path: Path, expected_digest: str) -> str:
    source_text = source_path.read_text()
    compile(source_text, str(source_path), "exec")
    if source_text.count("param_dtype = self.input_layernorm.weight.dtype") != 1:
        raise ValueError(
            f"patched LingBot decoder does not bind runtime dtype to its call boundary: "
            f"{source_path}"
        )
    if "param_dtype = self.self_attn.q_proj.weight.dtype" in source_text:
        raise ValueError(
            f"patched LingBot decoder still reads dtype from a nested FSDP unit: {source_path}"
        )
    if "self.self_attn.o_proj.weight.dtype" in source_text:
        raise ValueError(
            f"patched LingBot decoder still reads output dtype from a nested FSDP unit: "
            f"{source_path}"
        )
    digest = _sha256(source_path)
    if digest != expected_digest:
        raise ValueError("patched LingBot decoder digest differs from approved replay")
    return digest


def _validate_patched_sources(
    root: Path,
    *,
    expected_muon_sha256: str = PATCHED_MUON_SHA256,
    require_muon_collective_hotfix: bool = False,
    require_muon_mixed_device_hotfix: bool = False,
    expected_parallel_sha256: str = PATCHED_PARALLEL_SHA256,
    require_frozen_vision_offload: bool = False,
    require_trainable_vision_offload: bool = False,
    require_selective_class_cpu_offload: bool = False,
    require_vlm_selective_class_cpu_offload: bool = False,
    expected_model_sha256: str = PATCHED_MODEL_SHA256,
    additional_required_model_fragments: tuple[str, ...] = (),
) -> dict[str, str]:
    muon_source = root / MUON_SOURCE
    muon_text = muon_source.read_text()
    if "_MEGABATCH_MAX_GROUP_SIZE = 8" not in muon_text:
        raise ValueError("patched LingBot Muon does not bound two-rank optimizer batching")
    if require_muon_mixed_device_hotfix and not require_muon_collective_hotfix:
        raise ValueError("Muon mixed-device support requires collective alignment first")
    if require_muon_collective_hotfix:
        required = (
            "other_params: List[Tuple[Tensor, str, bool]]",
            "if not has_grad and kind != _KIND_MOE_GATHER_3D",
            "other_params.append((p, kind, has_grad))",
            "for p, kind, has_grad in other_params",
            "update_local = torch.zeros_like(p_local)",
        )
        missing = [fragment for fragment in required if fragment not in muon_text]
        if missing:
            raise ValueError(
                f"patched LingBot Muon omits collective-alignment closure: {missing}"
            )
    if require_muon_mixed_device_hotfix:
        required = (
            "local_parameter = p.to_local() if isinstance(p, DTensor) else p",
            "key = (global_shape, str(p.dtype), local_parameter.device.type)",
            'if stacked_local.device.type == "cpu":',
            'device=torch.device("cuda", torch.cuda.current_device())',
            "ortho_local.to(device=p_local.device, dtype=p_local.dtype)",
        )
        missing = [fragment for fragment in required if fragment not in muon_text]
        if missing:
            raise ValueError(
                f"patched LingBot Muon omits mixed-device closure: {missing}"
            )
    muon_digest = _sha256(muon_source)
    if muon_digest != expected_muon_sha256:
        raise ValueError("patched LingBot Muon digest differs from approved replay")
    return {
        str(CHECKPOINTER_SOURCE): _validate_patched_checkpointer_source(root / CHECKPOINTER_SOURCE),
        str(PARALLEL_SOURCE): _validate_patched_parallel_source(
            root / PARALLEL_SOURCE,
            expected_digest=expected_parallel_sha256,
            require_frozen_vision_offload=require_frozen_vision_offload,
            require_trainable_vision_offload=require_trainable_vision_offload,
            require_selective_class_cpu_offload=require_selective_class_cpu_offload,
            require_vlm_selective_class_cpu_offload=(
                require_vlm_selective_class_cpu_offload
            ),
        ),
        str(MODEL_SOURCE): _validate_patched_model_source(
            root / MODEL_SOURCE,
            expected_digest=expected_model_sha256,
            additional_required_fragments=additional_required_model_fragments,
        ),
        str(ACTION_DECODER_SOURCE): _validate_patched_decoder_source(
            root / ACTION_DECODER_SOURCE,
            PATCHED_ACTION_DECODER_SHA256,
        ),
        str(TEXT_DECODER_SOURCE): _validate_patched_decoder_source(
            root / TEXT_DECODER_SOURCE,
            PATCHED_TEXT_DECODER_SHA256,
        ),
        str(QWEN25_TEXT_DECODER_SOURCE): _validate_patched_decoder_source(
            root / QWEN25_TEXT_DECODER_SOURCE,
            PATCHED_QWEN25_TEXT_DECODER_SHA256,
        ),
        str(MUON_SOURCE): muon_digest,
    }


def verify_muon_collective_hotfix(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay the approved Muon fix on top of the immutable native patch."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    hotfix = root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH
    _validate_patch(native_patch)
    _validate_muon_collective_hotfix(hotfix)
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix": str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH),
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-muon-hotfix-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        _run(["git", "apply", "--check", str(native_patch)], cwd=exported)
        _run(["git", "apply", str(native_patch)], cwd=exported)
        _run(["git", "apply", "--check", str(hotfix)], cwd=exported)
        _run(["git", "apply", str(hotfix)], cwd=exported)
        patched_digests = _validate_patched_sources(
            exported,
            expected_muon_sha256=PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
            require_muon_collective_hotfix=True,
        )
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive_plus_ordered_patches",
        }
    )
    return result


def verify_selective_class_cpu_offload(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay the native, Muon, and selective-class offload overlays in order."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    hotfix = root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH
    offload_patch = root / SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH
    _validate_patch(native_patch)
    _validate_muon_collective_hotfix(hotfix)
    _validate_selective_class_cpu_offload(offload_patch)
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix": str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH),
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "selective_class_cpu_offload": str(SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH),
        "selective_class_cpu_offload_sha256": SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-selective-class-offload-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for patch in (native_patch, hotfix, offload_patch):
            _run(["git", "apply", "--check", str(patch)], cwd=exported)
            _run(["git", "apply", str(patch)], cwd=exported)
        patched_digests = _validate_patched_sources(
            exported,
            expected_muon_sha256=PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
            require_muon_collective_hotfix=True,
            expected_parallel_sha256=(
                PATCHED_PARALLEL_WITH_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
            ),
            require_selective_class_cpu_offload=True,
        )
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive_plus_three_ordered_patches",
        }
    )
    return result


def verify_selective_frozen_vision_offload(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay the complete execution-only visual offload overlay chain."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    hotfix = root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH
    offload_patch = root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
    visual_root_patch = root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
    _validate_patch(native_patch)
    _validate_muon_collective_hotfix(hotfix)
    _validate_selective_frozen_vision_offload(offload_patch)
    _validate_frozen_visual_root_offload(visual_root_patch)
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix": str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH),
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "selective_frozen_vision_offload": str(
            SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload": str(FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-frozen-vision-offload-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for patch in (native_patch, hotfix, offload_patch, visual_root_patch):
            _run(["git", "apply", "--check", str(patch)], cwd=exported)
            _run(["git", "apply", str(patch)], cwd=exported)
        patched_digests = _validate_patched_sources(
            exported,
            expected_muon_sha256=PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
            require_muon_collective_hotfix=True,
            expected_parallel_sha256=(
                PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256
            ),
            require_frozen_vision_offload=True,
        )
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive_plus_four_ordered_patches",
        }
    )
    return result


def verify_selective_trainable_vision_offload(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay the complete execution-only trainable visual offload overlay chain."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    hotfix = root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH
    frozen_offload_patch = root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
    visual_root_patch = root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
    trainable_offload_patch = root / SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    mixed_device_muon_patch = root / MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH
    _validate_patch(native_patch)
    _validate_muon_collective_hotfix(hotfix)
    _validate_selective_frozen_vision_offload(frozen_offload_patch)
    _validate_frozen_visual_root_offload(visual_root_patch)
    _validate_selective_trainable_vision_offload(trainable_offload_patch)
    _validate_muon_mixed_device_megabatch(mixed_device_muon_patch)
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix": str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH),
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "muon_mixed_device_megabatch": str(MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH),
        "muon_mixed_device_megabatch_sha256": MUON_MIXED_DEVICE_MEGABATCH_SHA256,
        "selective_frozen_vision_offload": str(
            SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload": str(FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "selective_trainable_vision_offload": str(
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_trainable_vision_offload_sha256": (
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(
        prefix="picf-lingbot-trainable-vision-offload-"
    ) as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for patch in (
            native_patch,
            hotfix,
            frozen_offload_patch,
            visual_root_patch,
            trainable_offload_patch,
            mixed_device_muon_patch,
        ):
            _run(["git", "apply", "--check", str(patch)], cwd=exported)
            _run(["git", "apply", str(patch)], cwd=exported)
        patched_digests = _validate_patched_sources(
            exported,
            expected_muon_sha256=PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
            require_muon_collective_hotfix=True,
            require_muon_mixed_device_hotfix=True,
            expected_parallel_sha256=(
                PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256
            ),
            require_trainable_vision_offload=True,
        )
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive_plus_six_ordered_patches",
        }
    )
    return result


def verify_selective_trainable_vision_with_selective_class_cpu_offload(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay trainable-vision and selective-class execution overlays together."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    hotfix = root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH
    frozen_offload_patch = root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
    visual_root_patch = root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
    trainable_offload_patch = root / SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    mixed_device_muon_patch = root / MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH
    selective_class_patch = (
        root / SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    )
    _validate_patch(native_patch)
    _validate_muon_collective_hotfix(hotfix)
    _validate_selective_frozen_vision_offload(frozen_offload_patch)
    _validate_frozen_visual_root_offload(visual_root_patch)
    _validate_selective_trainable_vision_offload(trainable_offload_patch)
    _validate_muon_mixed_device_megabatch(mixed_device_muon_patch)
    _validate_selective_class_cpu_offload(
        selective_class_patch,
        expected_digest=SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256,
    )
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix": str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH),
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "muon_mixed_device_megabatch": str(MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH),
        "muon_mixed_device_megabatch_sha256": MUON_MIXED_DEVICE_MEGABATCH_SHA256,
        "selective_frozen_vision_offload": str(
            SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload": str(FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "selective_trainable_vision_offload": str(
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_trainable_vision_offload_sha256": (
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "selective_class_after_trainable_vision_offload": str(
            SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_class_after_trainable_vision_offload_sha256": (
            SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(
        prefix="picf-lingbot-trainable-vision-selective-class-offload-"
    ) as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for patch in (
            native_patch,
            hotfix,
            frozen_offload_patch,
            visual_root_patch,
            trainable_offload_patch,
            mixed_device_muon_patch,
            selective_class_patch,
        ):
            _run(["git", "apply", "--check", str(patch)], cwd=exported)
            _run(["git", "apply", str(patch)], cwd=exported)
        patched_digests = _validate_patched_sources(
            exported,
            expected_muon_sha256=PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
            require_muon_collective_hotfix=True,
            require_muon_mixed_device_hotfix=True,
            expected_parallel_sha256=(
                PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
            ),
            require_trainable_vision_offload=True,
            require_selective_class_cpu_offload=True,
        )
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive_plus_seven_ordered_patches",
        }
    )
    return result


def verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Replay the exact eight-overlay WLA placement chain from the pinned commit."""

    root = root.resolve()
    native_patch = root / PATCH_RELATIVE_PATH
    hotfix = root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH
    frozen_offload_patch = root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
    visual_root_patch = root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
    trainable_offload_patch = root / SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    mixed_device_muon_patch = root / MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH
    selective_class_patch = (
        root / SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    )
    vlm_selective_class_patch = (
        root / VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    )
    _validate_patch(native_patch)
    _validate_muon_collective_hotfix(hotfix)
    _validate_selective_frozen_vision_offload(frozen_offload_patch)
    _validate_frozen_visual_root_offload(visual_root_patch)
    _validate_selective_trainable_vision_offload(trainable_offload_patch)
    _validate_muon_mixed_device_megabatch(mixed_device_muon_patch)
    _validate_selective_class_cpu_offload(
        selective_class_patch,
        expected_digest=SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256,
    )
    _validate_vlm_selective_class_cpu_offload(vlm_selective_class_patch)
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "native_patch": str(PATCH_RELATIVE_PATH),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix": str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH),
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "muon_mixed_device_megabatch": str(MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH),
        "muon_mixed_device_megabatch_sha256": MUON_MIXED_DEVICE_MEGABATCH_SHA256,
        "selective_frozen_vision_offload": str(
            SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload": str(FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "selective_trainable_vision_offload": str(
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_trainable_vision_offload_sha256": (
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "selective_class_after_trainable_vision_offload": str(
            SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "selective_class_after_trainable_vision_offload_sha256": (
            SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "vlm_selective_class_after_trainable_vision_offload": str(
            VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
        ),
        "vlm_selective_class_after_trainable_vision_offload_sha256": (
            VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(
        prefix="picf-lingbot-trainable-vision-vlm-selective-class-offload-"
    ) as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for patch in (
            native_patch,
            hotfix,
            frozen_offload_patch,
            visual_root_patch,
            trainable_offload_patch,
            mixed_device_muon_patch,
            selective_class_patch,
            vlm_selective_class_patch,
        ):
            _run(["git", "apply", "--check", str(patch)], cwd=exported)
            _run(["git", "apply", str(patch)], cwd=exported)
        patched_digests = _validate_patched_sources(
            exported,
            expected_muon_sha256=PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
            require_muon_collective_hotfix=True,
            require_muon_mixed_device_hotfix=True,
            expected_parallel_sha256=(
                PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_VLM_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
            ),
            require_trainable_vision_offload=True,
            require_selective_class_cpu_offload=True,
            require_vlm_selective_class_cpu_offload=True,
        )
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive_plus_eight_ordered_patches",
        }
    )
    return result


def verify_native_patch(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Rebuild the overlay from the immutable official Git object."""

    root = root.resolve()
    patch_path = root / PATCH_RELATIVE_PATH
    _validate_patch(patch_path)
    result: dict[str, object] = {
        "apply_checked": False,
        "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "patch": str(PATCH_RELATIVE_PATH),
        "patch_sha256": PATCH_SHA256,
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result
    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot-native checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-native-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        upstream_text = (exported / MODEL_SOURCE).read_text()
        missing = [
            fragment for fragment in _REQUIRED_UPSTREAM_FRAGMENTS if fragment not in upstream_text
        ]
        if missing:
            raise ValueError(f"pinned LingBot source omits required symbols: {missing}")
        _run(["git", "apply", "--check", str(patch_path)], cwd=exported)
        _run(["git", "apply", str(patch_path)], cwd=exported)
        patched_digests = _validate_patched_sources(exported)
    result.update(
        {
            "apply_checked": True,
            "patched_source_sha256": patched_digests,
            "verification_source": "immutable_commit_archive",
        }
    )
    return result


def validate_prepared_native_source(
    *,
    checkout: Path,
    patch_path: Path,
) -> dict[str, object]:
    """Validate one already-patched checkout, including every dirty path.

    Python imports may leave bytecode caches in the external checkout. They are
    removed under the same strict artifact policy used during preparation before
    the Git worktree is compared with the exact patch-owned path set.
    """

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    _validate_patch(patch_path)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    if detect_native_patch_state(checkout, patch_path) != "applied":
        raise ValueError("LingBot-native checkout is not in the exact applied patch state")
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    dirty = _dirty_paths(checkout)
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    return {
        "checkout": str(checkout),
        "patch_sha256": PATCH_SHA256,
        "patch_state": "applied",
        "patched_source_sha256": _validate_patched_sources(checkout),
        "source_commit": actual,
    }


def validate_prepared_native_source_with_muon_collective_hotfix(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
) -> dict[str, object]:
    """Validate the exact native overlay plus the approved optimizer-only hotfix."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    reverse = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(hotfix_path)],
        capture_output=True,
        text=True,
    )
    if reverse.returncode != 0:
        raise ValueError("LingBot checkout does not contain the exact Muon collective hotfix")
    patched_digests = _validate_patched_sources(
        checkout,
        expected_muon_sha256=PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
        require_muon_collective_hotfix=True,
    )
    replay = verify_muon_collective_hotfix(root=patch_path.parents[2], checkout=checkout)
    if patched_digests != replay.get("patched_source_sha256"):
        raise ValueError("LingBot checkout differs from the ordered native-plus-hotfix replay")
    return {
        "checkout": str(checkout),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "patch_state": "native_applied_with_muon_collective_hotfix",
        "patched_source_sha256": patched_digests,
        "source_commit": actual,
    }


def validate_prepared_native_source_with_selective_class_cpu_offload(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
    offload_patch_path: Path,
) -> dict[str, object]:
    """Validate the exact three-overlay source used by the full WSA runtime."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    offload_patch_path = offload_patch_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    _validate_selective_class_cpu_offload(offload_patch_path)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    for label, overlay in (
        ("Muon collective hotfix", hotfix_path),
        ("selective-class CPU offload", offload_patch_path),
    ):
        reverse = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(overlay)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode != 0:
            raise ValueError(f"LingBot checkout does not contain the exact {label}")
    patched_digests = _validate_patched_sources(
        checkout,
        expected_muon_sha256=PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
        require_muon_collective_hotfix=True,
        expected_parallel_sha256=PATCHED_PARALLEL_WITH_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
        require_selective_class_cpu_offload=True,
    )
    replay = verify_selective_class_cpu_offload(
        root=patch_path.parents[2],
        checkout=checkout,
    )
    if patched_digests != replay.get("patched_source_sha256"):
        raise ValueError("LingBot checkout differs from the approved three-overlay replay")
    return {
        "checkout": str(checkout),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "selective_class_cpu_offload_sha256": SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
        "patch_state": "native_plus_muon_plus_selective_class_cpu_offload",
        "patched_source_sha256": patched_digests,
        "source_commit": actual,
    }


def validate_prepared_native_source_with_selective_frozen_vision_offload(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
    offload_patch_path: Path,
    visual_root_patch_path: Path,
) -> dict[str, object]:
    """Validate all four ordered LingBot overlays and no unrelated edits."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    offload_patch_path = offload_patch_path.resolve()
    visual_root_patch_path = visual_root_patch_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    _validate_selective_frozen_vision_offload(offload_patch_path)
    _validate_frozen_visual_root_offload(visual_root_patch_path)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    for label, overlay in (
        ("Muon collective hotfix", hotfix_path),
        ("frozen visual-root offload", visual_root_patch_path),
    ):
        reverse = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(overlay)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode != 0:
            raise ValueError(f"LingBot checkout does not contain the exact {label}")
    patched_digests = _validate_patched_sources(
        checkout,
        expected_muon_sha256=PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
        require_muon_collective_hotfix=True,
        expected_parallel_sha256=PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256,
        require_frozen_vision_offload=True,
    )
    replay = verify_selective_frozen_vision_offload(
        root=patch_path.parents[2],
        checkout=checkout,
    )
    if patched_digests != replay.get("patched_source_sha256"):
        raise ValueError("LingBot checkout differs from the approved four-overlay replay")
    return {
        "checkout": str(checkout),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "patch_state": "native_plus_muon_plus_complete_frozen_visual_root_offload",
        "patched_source_sha256": patched_digests,
        "source_commit": actual,
    }


def validate_prepared_native_source_with_selective_trainable_vision_offload(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
    frozen_offload_patch_path: Path,
    visual_root_patch_path: Path,
    trainable_offload_patch_path: Path,
    mixed_device_muon_patch_path: Path,
) -> dict[str, object]:
    """Validate all six ordered LingBot overlays and no unrelated edits."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    frozen_offload_patch_path = frozen_offload_patch_path.resolve()
    visual_root_patch_path = visual_root_patch_path.resolve()
    trainable_offload_patch_path = trainable_offload_patch_path.resolve()
    mixed_device_muon_patch_path = mixed_device_muon_patch_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    _validate_selective_frozen_vision_offload(frozen_offload_patch_path)
    _validate_frozen_visual_root_offload(visual_root_patch_path)
    _validate_selective_trainable_vision_offload(trainable_offload_patch_path)
    _validate_muon_mixed_device_megabatch(mixed_device_muon_patch_path)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    # The final mixed-device overlay changes context introduced by the earlier
    # Muon hotfix, so that intermediate patch is not independently reversible
    # from the final tree. Its exact presence is proven below by the final-file
    # digest and immutable six-overlay replay. Final overlays remain directly
    # reversible here.
    for label, overlay in (
        ("trainable visual offload", trainable_offload_patch_path),
        ("Muon mixed-device megabatch", mixed_device_muon_patch_path),
    ):
        reverse = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(overlay)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode != 0:
            raise ValueError(f"LingBot checkout does not contain the exact {label}")
    patched_digests = _validate_patched_sources(
        checkout,
        expected_muon_sha256=PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
        require_muon_collective_hotfix=True,
        require_muon_mixed_device_hotfix=True,
        expected_parallel_sha256=PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256,
        require_trainable_vision_offload=True,
    )
    replay = verify_selective_trainable_vision_offload(
        root=patch_path.parents[2],
        checkout=checkout,
    )
    if patched_digests != replay.get("patched_source_sha256"):
        raise ValueError("LingBot checkout differs from the approved six-overlay replay")
    return {
        "checkout": str(checkout),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "muon_mixed_device_megabatch_sha256": MUON_MIXED_DEVICE_MEGABATCH_SHA256,
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "selective_trainable_vision_offload_sha256": (
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "patch_state": "native_plus_muon_plus_complete_trainable_visual_root_offload",
        "patched_source_sha256": patched_digests,
        "source_commit": actual,
    }


def validate_prepared_native_source_with_trainable_vision_and_selective_class_offload(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
    frozen_offload_patch_path: Path,
    visual_root_patch_path: Path,
    trainable_offload_patch_path: Path,
    mixed_device_muon_patch_path: Path,
    selective_class_patch_path: Path,
) -> dict[str, object]:
    """Validate the exact seven-overlay source required by ADR-221 on two GPUs."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    frozen_offload_patch_path = frozen_offload_patch_path.resolve()
    visual_root_patch_path = visual_root_patch_path.resolve()
    trainable_offload_patch_path = trainable_offload_patch_path.resolve()
    mixed_device_muon_patch_path = mixed_device_muon_patch_path.resolve()
    selective_class_patch_path = selective_class_patch_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    _validate_selective_frozen_vision_offload(frozen_offload_patch_path)
    _validate_frozen_visual_root_offload(visual_root_patch_path)
    _validate_selective_trainable_vision_offload(trainable_offload_patch_path)
    _validate_muon_mixed_device_megabatch(mixed_device_muon_patch_path)
    _validate_selective_class_cpu_offload(
        selective_class_patch_path,
        expected_digest=SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256,
    )
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    for label, overlay in (
        ("selective-class after trainable-vision offload", selective_class_patch_path),
        ("Muon mixed-device megabatch", mixed_device_muon_patch_path),
    ):
        reverse = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(overlay)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode != 0:
            raise ValueError(f"LingBot checkout does not contain the exact {label}")
    patched_digests = _validate_patched_sources(
        checkout,
        expected_muon_sha256=PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
        require_muon_collective_hotfix=True,
        require_muon_mixed_device_hotfix=True,
        expected_parallel_sha256=(
            PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
        ),
        require_trainable_vision_offload=True,
        require_selective_class_cpu_offload=True,
    )
    replay = verify_selective_trainable_vision_with_selective_class_cpu_offload(
        root=patch_path.parents[2],
        checkout=checkout,
    )
    if patched_digests != replay.get("patched_source_sha256"):
        raise ValueError("LingBot checkout differs from the approved seven-overlay replay")
    return {
        "checkout": str(checkout),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "muon_mixed_device_megabatch_sha256": MUON_MIXED_DEVICE_MEGABATCH_SHA256,
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "selective_trainable_vision_offload_sha256": (
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "selective_class_after_trainable_vision_offload_sha256": (
            SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "patch_state": (
            "native_plus_muon_plus_trainable_visual_root_plus_selective_class_offload"
        ),
        "patched_source_sha256": patched_digests,
        "source_commit": actual,
    }


def validate_prepared_native_source_with_trainable_vision_and_vlm_selective_class_offload(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
    frozen_offload_patch_path: Path,
    visual_root_patch_path: Path,
    trainable_offload_patch_path: Path,
    mixed_device_muon_patch_path: Path,
    selective_class_patch_path: Path,
    vlm_selective_class_patch_path: Path,
) -> dict[str, object]:
    """Validate the exact eight-overlay source required by ADR-224 WLA."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    frozen_offload_patch_path = frozen_offload_patch_path.resolve()
    visual_root_patch_path = visual_root_patch_path.resolve()
    trainable_offload_patch_path = trainable_offload_patch_path.resolve()
    mixed_device_muon_patch_path = mixed_device_muon_patch_path.resolve()
    selective_class_patch_path = selective_class_patch_path.resolve()
    vlm_selective_class_patch_path = vlm_selective_class_patch_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    _validate_selective_frozen_vision_offload(frozen_offload_patch_path)
    _validate_frozen_visual_root_offload(visual_root_patch_path)
    _validate_selective_trainable_vision_offload(trainable_offload_patch_path)
    _validate_muon_mixed_device_megabatch(mixed_device_muon_patch_path)
    _validate_selective_class_cpu_offload(
        selective_class_patch_path,
        expected_digest=SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256,
    )
    _validate_vlm_selective_class_cpu_offload(vlm_selective_class_patch_path)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    expected_dirty = {str(path) for path in PATCHED_SOURCES}
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    for label, overlay in (
        ("VLM selective-class dispatch", vlm_selective_class_patch_path),
        ("selective-class after trainable-vision offload", selective_class_patch_path),
        ("Muon mixed-device megabatch", mixed_device_muon_patch_path),
    ):
        reverse = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(overlay)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode != 0:
            raise ValueError(f"LingBot checkout does not contain the exact {label}")
    patched_digests = _validate_patched_sources(
        checkout,
        expected_muon_sha256=PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
        require_muon_collective_hotfix=True,
        require_muon_mixed_device_hotfix=True,
        expected_parallel_sha256=(
            PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_VLM_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
        ),
        require_trainable_vision_offload=True,
        require_selective_class_cpu_offload=True,
        require_vlm_selective_class_cpu_offload=True,
    )
    replay = verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload(
        root=patch_path.parents[2],
        checkout=checkout,
    )
    if patched_digests != replay.get("patched_source_sha256"):
        raise ValueError("LingBot checkout differs from the approved eight-overlay replay")
    return {
        "checkout": str(checkout),
        "native_patch_sha256": PATCH_SHA256,
        "runtime_hotfix_sha256": MUON_COLLECTIVE_HOTFIX_SHA256,
        "muon_mixed_device_megabatch_sha256": MUON_MIXED_DEVICE_MEGABATCH_SHA256,
        "selective_frozen_vision_offload_sha256": (
            SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "frozen_visual_root_offload_sha256": FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
        "selective_trainable_vision_offload_sha256": (
            SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "selective_class_after_trainable_vision_offload_sha256": (
            SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "vlm_selective_class_after_trainable_vision_offload_sha256": (
            VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
        ),
        "patch_state": (
            "native_plus_muon_plus_trainable_visual_root_plus_vlm_selective_class_offload"
        ),
        "patched_source_sha256": patched_digests,
        "source_commit": actual,
    }


def prepare_native_source_with_muon_collective_hotfix(
    *,
    checkout: Path,
    patch_path: Path,
    hotfix_path: Path,
    source_url: str = LINGBOT_NATIVE_SOURCE_URL,
) -> dict[str, object]:
    """Prepare the ordered native overlay and optimizer-only runtime hotfix."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    hotfix_path = hotfix_path.resolve()
    _validate_patch(patch_path)
    _validate_muon_collective_hotfix(hotfix_path)
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", source_url, str(checkout)])
        _run(["git", "checkout", "--detach", LINGBOT_NATIVE_SOURCE_COMMIT], cwd=checkout)
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    dirty = _dirty_paths(checkout)
    if not dirty:
        _run(["git", "apply", str(patch_path)], cwd=checkout)
        _run(["git", "apply", str(hotfix_path)], cwd=checkout)
    elif dirty == {str(path) for path in PATCHED_SOURCES}:
        muon_digest = _sha256(checkout / MUON_SOURCE)
        if muon_digest == PATCHED_MUON_SHA256:
            _run(["git", "apply", str(hotfix_path)], cwd=checkout)
        elif muon_digest != PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256:
            raise ValueError("LingBot checkout has an unknown Muon source state")
    else:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    return validate_prepared_native_source_with_muon_collective_hotfix(
        checkout=checkout,
        patch_path=patch_path,
        hotfix_path=hotfix_path,
    )


def prepare_native_source(
    *,
    checkout: Path,
    patch_path: Path,
    source_url: str = LINGBOT_NATIVE_SOURCE_URL,
) -> dict[str, object]:
    """Clone or validate the sole accepted idempotent patched source state."""

    checkout = checkout.resolve()
    patch_path = patch_path.resolve()
    _validate_patch(patch_path)
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", source_url, str(checkout)])
        _run(
            ["git", "checkout", "--detach", LINGBOT_NATIVE_SOURCE_COMMIT],
            cwd=checkout,
        )
    if not (checkout / ".git").exists():
        raise ValueError(f"LingBot checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_NATIVE_SOURCE_COMMIT}")
    _purge_generated_python_bytecode(checkout)
    state = detect_native_patch_state(checkout, patch_path)
    expected_dirty = set() if state == "baseline" else {str(path) for path in PATCHED_SOURCES}
    dirty = _dirty_paths(checkout)
    if dirty != expected_dirty:
        raise ValueError(f"LingBot checkout has unrelated changes: {sorted(dirty)}")
    if state == "baseline":
        _run(["git", "apply", str(patch_path)], cwd=checkout)
    return validate_prepared_native_source(checkout=checkout, patch_path=patch_path)


def prepare_utils3d_source(
    *,
    checkout: Path,
    source_url: str = UTILS3D_SOURCE_URL,
) -> dict[str, str]:
    """Materialize MoGe's exact upstream ``utils3d`` dependency."""

    checkout = checkout.resolve()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", source_url, str(checkout)])
        _run(["git", "checkout", "--detach", UTILS3D_SOURCE_COMMIT], cwd=checkout)
    if not (checkout / ".git").exists():
        raise ValueError(f"utils3d checkout is not a Git repository: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != UTILS3D_SOURCE_COMMIT:
        raise ValueError(f"utils3d checkout {actual} differs from {UTILS3D_SOURCE_COMMIT}")
    dirty = _dirty_paths(checkout)
    if dirty:
        raise ValueError(f"utils3d checkout has unrelated changes: {sorted(dirty)}")
    return {
        "checkout": str(checkout),
        "source_commit": actual,
    }


def native_depth_runtime_install_commands(
    *,
    python: Path,
    source_checkout: Path,
    utils3d_checkout: Path,
) -> tuple[list[str], ...]:
    """Install the three depth packages without mutating the pinned runtime."""

    source_checkout = source_checkout.resolve()
    utils3d_checkout = utils3d_checkout.resolve()
    if _run(["git", "rev-parse", "HEAD"], cwd=utils3d_checkout) != UTILS3D_SOURCE_COMMIT:
        raise ValueError("utils3d checkout differs from MoGe's exact dependency commit")
    if _dirty_paths(utils3d_checkout):
        raise ValueError("utils3d checkout must be clean before runtime installation")
    moge_source = source_checkout / MOGE_SOURCE
    lingbot_depth_source = source_checkout / LINGBOT_DEPTH_SOURCE
    for package_source in (moge_source, lingbot_depth_source):
        if not package_source.is_dir():
            raise ValueError(f"LingBot depth package source is absent: {package_source}")
    moge_metadata = (moge_source / "pyproject.toml").read_text()
    exact_requirement = (
        f"utils3d @ git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_SOURCE_COMMIT}"
    )
    if exact_requirement not in moge_metadata:
        raise ValueError("pinned MoGe metadata no longer selects the approved utils3d commit")

    prefix = [
        str(python.absolute()),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--editable",
    ]
    return tuple(
        [*prefix, str(package_source)]
        for package_source in (utils3d_checkout, lingbot_depth_source, moge_source)
    )


def write_native_depth_path(*, python: Path, utils3d_checkout: Path) -> str:
    """Replace the broken upstream ``.pth`` target with the pinned checkout."""

    checkout = utils3d_checkout.resolve()
    if not checkout.is_dir():
        raise ValueError(f"utils3d checkout is absent: {checkout}")
    program = """
import json
import os
import sys
import sysconfig
from pathlib import Path

checkout = Path(os.environ["PICF_UTILS3D_CHECKOUT"]).resolve(strict=True)
raw_site_root = sysconfig.get_path("purelib")
if not raw_site_root:
    raise RuntimeError("selected Python exposes no purelib installation path")
prefix = Path(sys.prefix).resolve(strict=True)
site_root = Path(raw_site_root).resolve(strict=True)
try:
    site_root.relative_to(prefix)
except ValueError as error:
    raise RuntimeError(
        f"selected Python purelib lies outside its prefix: {site_root} not under {prefix}"
    ) from error
target = site_root / "stablevla_local_depth.pth"
if target.is_symlink():
    raise RuntimeError(f"refusing to replace symlinked path file: {target}")
flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(target, flags, 0o644)
try:
    os.write(descriptor, (str(checkout) + "\\n").encode("ascii"))
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(json.dumps({"path_file": str(target), "utils3d_checkout": str(checkout)}))
"""
    environment = os.environ.copy()
    environment["PICF_UTILS3D_CHECKOUT"] = str(checkout)
    completed = subprocess.run(
        [str(python.absolute()), "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = json.loads(completed.stdout)
    if payload.get("utils3d_checkout") != str(checkout):
        raise RuntimeError("selected Python wrote an unexpected utils3d path")
    path_file = Path(payload["path_file"])
    if path_file.read_text(encoding="ascii") != f"{checkout}\n":
        raise RuntimeError("selected Python did not publish the exact utils3d path")
    return str(path_file)


def native_runtime_restore_command(*, python: Path, source_checkout: Path) -> list[str]:
    """Restore both immutable requirement sets after the upstream setup."""

    requirements = source_checkout.resolve() / "requirements.txt"
    depth_requirements = source_checkout.resolve() / "requirements-depth.txt"
    if not requirements.is_file() or _sha256(requirements) != LINGBOT_NATIVE_REQUIREMENTS_SHA256:
        raise ValueError("LingBot-native requirements differ from the pinned source contract")
    if (
        not depth_requirements.is_file()
        or _sha256(depth_requirements) != LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256
    ):
        raise ValueError("LingBot-native depth requirements differ from the pinned source contract")
    return [
        str(python.absolute()),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "-r",
        str(requirements),
        "-r",
        str(depth_requirements),
    ]


def native_audit_tools_install_command(*, python: Path) -> list[str]:
    """Install the exact tools used by cloud preflight without requiring ``uv``."""

    return [
        str(python.absolute()),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        *(f"{name}=={version}" for name, version in sorted(LINGBOT_NATIVE_AUDIT_TOOLS.items())),
    ]


def validate_calvin_offline_source(calvin_env_root: Path) -> dict[str, str]:
    """Bind the offline renderer to the exact clean CALVIN source pair."""

    environment = calvin_env_root.resolve()
    parent = environment.parent.resolve()
    if not environment.is_dir() or not (parent / ".git").exists():
        raise ValueError(f"pinned CALVIN environment is absent: {environment}")
    identities = (
        ("calvin_commit", parent, CALVIN_SOURCE_COMMIT),
        ("calvin_env_commit", environment, CALVIN_ENV_SOURCE_COMMIT),
    )
    report: dict[str, str] = {
        "calvin_env_root": str(environment),
        "status": "PASS",
    }
    for name, checkout, expected in identities:
        top = Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=checkout)).resolve()
        actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
        if top != checkout or actual != expected:
            raise ValueError(f"{name} differs from the frozen source contract")
        if _dirty_paths(checkout):
            raise ValueError(f"{name} checkout is dirty")
        report[name] = actual
    requirements = environment / "requirements.txt"
    setup = environment / "setup.py"
    if (
        not requirements.is_file()
        or _sha256(requirements) != CALVIN_ENV_REQUIREMENTS_SHA256
        or not setup.is_file()
        or _sha256(setup) != CALVIN_ENV_SETUP_SHA256
    ):
        raise ValueError("CALVIN environment dependency metadata differs from the frozen source")
    report["calvin_requirements_sha256"] = _sha256(requirements)
    report["calvin_setup_sha256"] = _sha256(setup)
    return report


def calvin_offline_runtime_install_command(
    *,
    python: Path,
    calvin_env_root: Path,
) -> list[str]:
    """Install the exact supplemental runtime for offline CALVIN rendering."""

    validate_calvin_offline_source(calvin_env_root)
    return [
        str(python.absolute()),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        *(f"{name}=={version}" for name, version in sorted(CALVIN_OFFLINE_RUNTIME_EXTRAS.items())),
    ]


def picf_overlay_install_command(*, python: Path, repo_root: Path) -> list[str]:
    return [
        str(python.absolute()),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--editable",
        str(repo_root.resolve()),
    ]


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=root / CHECKOUT_RELATIVE_PATH,
    )
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--source-url", default=LINGBOT_NATIVE_SOURCE_URL)
    parser.add_argument(
        "--utils3d-checkout",
        type=Path,
        default=root / UTILS3D_CHECKOUT_RELATIVE_PATH,
    )
    parser.add_argument("--utils3d-source-url", default=UTILS3D_SOURCE_URL)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--download-processor", action="store_true")
    parser.add_argument("--hf-command", default="hf")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--repair-depth-runtime", action="store_true")
    parser.add_argument("--restore-runtime-pins", action="store_true")
    parser.add_argument("--install-audit-tools", action="store_true")
    parser.add_argument("--calvin-env-root", type=Path)
    parser.add_argument("--install-calvin-offline-runtime", action="store_true")
    parser.add_argument("--install-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    result = prepare_native_source(
        checkout=args.checkout,
        patch_path=args.patch,
        source_url=args.source_url,
    )
    result["replay"] = verify_native_patch(
        root=root,
        checkout=args.checkout,
        check_apply=True,
    )
    result["checkpoint_id"] = LINGBOT_CHECKPOINT_ID
    result["checkpoint_revision"] = LINGBOT_CHECKPOINT_REVISION
    result["processor_id"] = QWEN_PROCESSOR_ID
    result["processor_revision"] = QWEN_PROCESSOR_REVISION
    python_operations = (
        args.repair_depth_runtime,
        args.restore_runtime_pins,
        args.install_audit_tools,
        args.install_calvin_offline_runtime,
        args.install_overlay,
    )
    if any(python_operations) and args.python is None:
        raise ValueError("runtime repair, audit-tool and overlay installation require --python")
    if args.python is not None:
        if not args.python.is_file():
            raise ValueError(f"overlay Python executable is absent: {args.python}")
        runtime_command = native_runtime_restore_command(
            python=args.python,
            source_checkout=args.checkout,
        )
        audit_command = native_audit_tools_install_command(python=args.python)
        overlay_command = picf_overlay_install_command(python=args.python, repo_root=root)
        result["runtime_restore_command"] = runtime_command
        result["audit_tools_install_command"] = audit_command
        result["overlay_install_command"] = overlay_command
        if args.repair_depth_runtime:
            utils3d = prepare_utils3d_source(
                checkout=args.utils3d_checkout,
                source_url=args.utils3d_source_url,
            )
            depth_commands = native_depth_runtime_install_commands(
                python=args.python,
                source_checkout=args.checkout,
                utils3d_checkout=args.utils3d_checkout,
            )
            for command in depth_commands:
                subprocess.run(command, check=True)
            utils3d["path_file"] = write_native_depth_path(
                python=args.python,
                utils3d_checkout=args.utils3d_checkout,
            )
            result["depth_runtime"] = utils3d
            result["depth_runtime_install_commands"] = depth_commands
        else:
            result["depth_runtime"] = None
        if args.restore_runtime_pins:
            subprocess.run(runtime_command, check=True)
            result["runtime_pins_restored"] = True
        else:
            result["runtime_pins_restored"] = False
        if args.install_audit_tools:
            subprocess.run(audit_command, check=True)
            result["audit_tools_installed"] = True
        else:
            result["audit_tools_installed"] = False
        if args.install_calvin_offline_runtime:
            if args.calvin_env_root is None:
                raise ValueError("--install-calvin-offline-runtime requires --calvin-env-root")
            calvin_command = calvin_offline_runtime_install_command(
                python=args.python,
                calvin_env_root=args.calvin_env_root,
            )
            subprocess.run(calvin_command, check=True)
            result["calvin_offline_runtime_install_command"] = calvin_command
            result["calvin_offline_runtime_installed"] = True
            result["calvin_offline_source"] = validate_calvin_offline_source(args.calvin_env_root)
        else:
            result["calvin_offline_runtime_installed"] = False
        if args.install_overlay:
            subprocess.run(overlay_command, check=True)
            result["overlay_installed"] = True
        else:
            result["overlay_installed"] = False
    if args.download_checkpoint:
        if args.checkpoint_dir is None:
            raise ValueError("--download-checkpoint requires --checkpoint-dir")
        subprocess.run(
            checkpoint_download_command(
                hf_command=args.hf_command,
                checkpoint_dir=args.checkpoint_dir,
            ),
            check=True,
        )
    if args.download_processor:
        if args.processor_dir is None:
            raise ValueError("--download-processor requires --processor-dir")
        subprocess.run(
            processor_download_command(
                hf_command=args.hf_command,
                processor_dir=args.processor_dir,
            ),
            check=True,
        )
    if args.checkpoint_dir is not None:
        result.update(validate_checkpoint(args.checkpoint_dir))
    if args.processor_dir is not None:
        result.update(validate_processor(args.processor_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

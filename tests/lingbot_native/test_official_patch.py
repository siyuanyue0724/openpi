from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bootstrap_lingbot_vla2_native import (
    ACTION_DECODER_SOURCE,
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OFFLINE_RUNTIME_EXTRAS,
    CALVIN_SOURCE_COMMIT,
    CHECKOUT_RELATIVE_PATH,
    CHECKPOINTER_SOURCE,
    FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH,
    FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
    LINGBOT_DEPTH_SOURCE,
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MODEL_SOURCE,
    MOGE_SOURCE,
    MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
    MUON_COLLECTIVE_HOTFIX_SHA256,
    MUON_SOURCE,
    PARALLEL_SOURCE,
    PATCH_RELATIVE_PATH,
    PATCH_SHA256,
    PATCHED_ACTION_DECODER_SHA256,
    PATCHED_CHECKPOINTER_SHA256,
    PATCHED_MODEL_SHA256,
    PATCHED_MUON_SHA256,
    PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
    PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256,
    PATCHED_PARALLEL_SHA256,
    PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256,
    PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
    PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_VLM_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
    PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256,
    PATCHED_QWEN25_TEXT_DECODER_SHA256,
    PATCHED_SOURCES,
    PATCHED_TEXT_DECODER_SHA256,
    QWEN25_TEXT_DECODER_SOURCE,
    SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH,
    SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256,
    SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH,
    SELECTIVE_CLASS_CPU_OFFLOAD_SHA256,
    SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
    SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256,
    SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
    SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256,
    TEXT_DECODER_SOURCE,
    UTILS3D_CHECKOUT_RELATIVE_PATH,
    UTILS3D_SOURCE_COMMIT,
    VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
    VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256,
    _dirty_paths,
    _parse_args,
    _purge_generated_python_bytecode,
    calvin_offline_runtime_install_command,
    native_audit_tools_install_command,
    native_depth_runtime_install_commands,
    native_runtime_restore_command,
    picf_overlay_install_command,
    prepare_native_source,
    prepare_native_source_with_muon_collective_hotfix,
    validate_calvin_offline_source,
    validate_prepared_native_source,
    validate_prepared_native_source_with_muon_collective_hotfix,
    verify_muon_collective_hotfix,
    verify_native_patch,
    verify_selective_class_cpu_offload,
    verify_selective_frozen_vision_offload,
    verify_selective_trainable_vision_with_selective_class_cpu_offload,
    verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload,
    write_native_depth_path,
)

ROOT = Path(__file__).resolve().parents[2]


def _native_source_checkout() -> Path:
    selected = os.environ.get("PICF_LINGBOT_NATIVE_SOURCE")
    if selected is not None and selected.strip():
        return Path(selected).expanduser().resolve()
    return ROOT / CHECKOUT_RELATIVE_PATH


def test_native_source_checkout_honors_explicit_runtime_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected-source"
    monkeypatch.setenv("PICF_LINGBOT_NATIVE_SOURCE", str(selected))
    assert _native_source_checkout() == selected.resolve()


def test_native_patch_is_content_pinned_and_replayable() -> None:
    checkout = _native_source_checkout()
    check_apply = (checkout / ".git").exists()
    result = verify_native_patch(root=ROOT, checkout=checkout, check_apply=check_apply)
    assert result["commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert result["patch_sha256"] == PATCH_SHA256
    assert result["patched_sources"] == [str(path) for path in PATCHED_SOURCES]
    assert result["apply_checked"] is check_apply
    if check_apply:
        assert result["patched_source_sha256"] == {
            str(CHECKPOINTER_SOURCE): PATCHED_CHECKPOINTER_SHA256,
            str(PARALLEL_SOURCE): PATCHED_PARALLEL_SHA256,
            str(MODEL_SOURCE): PATCHED_MODEL_SHA256,
            str(ACTION_DECODER_SOURCE): PATCHED_ACTION_DECODER_SHA256,
            str(TEXT_DECODER_SOURCE): PATCHED_TEXT_DECODER_SHA256,
            str(QWEN25_TEXT_DECODER_SOURCE): PATCHED_QWEN25_TEXT_DECODER_SHA256,
            str(MUON_SOURCE): PATCHED_MUON_SHA256,
        }


def test_muon_collective_hotfix_is_content_pinned_and_narrow() -> None:
    result = verify_muon_collective_hotfix(root=ROOT, check_apply=False)
    assert result["commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert result["runtime_hotfix"] == str(MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH)
    assert result["runtime_hotfix_sha256"] == MUON_COLLECTIVE_HOTFIX_SHA256
    assert result["native_patch_sha256"] == PATCH_SHA256
    assert result["apply_checked"] is False

    patch = (ROOT / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH).read_text()
    assert patch.count("+++ b/") == 1
    assert f"+++ b/{MUON_SOURCE}" in patch
    assert "other_params: List[Tuple[Tensor, str, bool]]" in patch
    assert "if not has_grad and kind != _KIND_MOE_GATHER_3D" in patch
    assert "other_params.append((p, kind, has_grad))" in patch
    assert "update_local = torch.zeros_like(p_local)" in patch


def test_selective_frozen_vision_offload_is_content_pinned_and_narrow() -> None:
    result = verify_selective_frozen_vision_offload(root=ROOT, check_apply=False)
    assert result["commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert result["selective_frozen_vision_offload"] == str(
        SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
    )
    assert (
        result["selective_frozen_vision_offload_sha256"]
        == SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256
    )
    assert result["frozen_visual_root_offload"] == str(
        FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
    )
    assert result["frozen_visual_root_offload_sha256"] == FROZEN_VISUAL_ROOT_OFFLOAD_SHA256
    patch = (ROOT / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH).read_text()
    assert patch.count("+++ b/") == 1
    assert f"+++ b/{PARALLEL_SOURCE}" in patch
    assert "enable_frozen_vision_offload = kwargs.pop(" in patch
    assert "selective vision offload requires every vision block to be frozen" in patch
    root_patch = (ROOT / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH).read_text()
    assert root_patch.count("+++ b/") == 1
    assert f"+++ b/{PARALLEL_SOURCE}" in root_patch
    assert "fully_shard(visual_root, **kind_fsdp_kwargs)" in root_patch


def test_selective_class_cpu_offload_is_content_pinned_and_narrow() -> None:
    result = verify_selective_class_cpu_offload(root=ROOT, check_apply=False)
    assert result["commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert result["selective_class_cpu_offload"] == str(
        SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH
    )
    assert (
        result["selective_class_cpu_offload_sha256"]
        == SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
    )
    patch = (ROOT / SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH).read_text()
    assert patch.count("+++ b/") == 1
    assert f"+++ b/{PARALLEL_SOURCE}" in patch
    assert 'kwargs.pop("selective_cpu_module_classes", ())' in patch
    assert "module.__class__.__name__ in selective_cpu_module_classes" in patch
    assert "model._lingbot_fsdp2_selective_cpu_module_classes = (" in patch


def test_trainable_vision_and_selective_class_composition_is_pinned() -> None:
    result = verify_selective_trainable_vision_with_selective_class_cpu_offload(
        root=ROOT,
        check_apply=False,
    )
    assert result["selective_trainable_vision_offload"] == str(
        SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    )
    assert (
        result["selective_trainable_vision_offload_sha256"]
        == SELECTIVE_TRAINABLE_VISION_OFFLOAD_SHA256
    )
    assert result["selective_class_after_trainable_vision_offload"] == str(
        SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    )
    assert (
        result["selective_class_after_trainable_vision_offload_sha256"]
        == SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
    )
    patch = (
        ROOT / SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    ).read_text()
    assert patch.count("+++ b/") == 1
    assert f"+++ b/{PARALLEL_SOURCE}" in patch
    assert 'kwargs.pop("selective_cpu_module_classes", ())' in patch
    assert "module.__class__.__name__ in selective_cpu_module_classes" in patch


def test_wla_vlm_selective_class_composition_is_pinned() -> None:
    result = verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload(
        root=ROOT,
        check_apply=False,
    )
    assert result["vlm_selective_class_after_trainable_vision_offload"] == str(
        VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    )
    assert (
        result["vlm_selective_class_after_trainable_vision_offload_sha256"]
        == VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_SHA256
    )
    patch = (
        ROOT / VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
    ).read_text()
    assert patch.count("+++ b/") == 1
    assert f"+++ b/{PARALLEL_SOURCE}" in patch
    assert "layer_fsdp_kwargs = kind_fsdp_kwargs" in patch
    assert "if layer.__class__.__name__ in selective_cpu_module_classes" in patch
    assert "fully_shard(layer, **layer_fsdp_kwargs)" in patch


def test_native_patch_shards_qwen_text_and_vision_blocks_before_the_root() -> None:
    checkout = _native_source_checkout()
    if not checkout.exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    source_path = checkout / PARALLEL_SOURCE
    source = source_path.read_text()
    assert "def _collect_vlm_fsdp_layers" in source
    assert 'for kind in ("text", "vision")' in source
    assert "model._lingbot_vlm_fsdp2_topology = topology" in source
    assert "sharding only the root would create a large, non-overlapped AllGather" in source
    if "fully_shard(layer, **layer_fsdp_kwargs)" in source:
        assert source.count("fully_shard(layer, **layer_fsdp_kwargs)") == 1
        assert source.count("fully_shard(layer, **kind_fsdp_kwargs)") == 0
        assert source.count("fully_shard(visual_root, **kind_fsdp_kwargs)") == 1
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_VLM_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
        )
    elif (
        "enable_trainable_vision_offload = kwargs.pop(" in source
        and 'kwargs.pop("selective_cpu_module_classes", ())' in source
    ):
        assert source.count("fully_shard(layer, **kind_fsdp_kwargs)") == 1
        assert source.count("fully_shard(visual_root, **kind_fsdp_kwargs)") == 1
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == PATCHED_PARALLEL_WITH_TRAINABLE_VISION_AND_SELECTIVE_CLASS_CPU_OFFLOAD_SHA256
        )
    elif "enable_trainable_vision_offload = kwargs.pop(" in source:
        assert source.count("fully_shard(layer, **kind_fsdp_kwargs)") == 1
        assert source.count("fully_shard(visual_root, **kind_fsdp_kwargs)") == 1
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == PATCHED_PARALLEL_WITH_TRAINABLE_VISION_OFFLOAD_SHA256
        )
    elif "enable_frozen_vision_offload = kwargs.pop(" in source:
        assert source.count("fully_shard(layer, **kind_fsdp_kwargs)") == 1
        assert source.count("fully_shard(visual_root, **kind_fsdp_kwargs)") == 1
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256
        )
    else:
        assert source.count("fully_shard(layer, **mp_fsdp_kwargs)") == 1
    assert "shard_bifurcated_execution_units(" not in source
    assert "_BIFURCATED_ATTN_PROJECTION_UNITS" not in source
    assert source.count("cast_forward_inputs=False") == 1
    assert "root_fsdp_kwargs = dict(mp_fsdp_kwargs)" in source
    assert "fully_shard(model, **root_fsdp_kwargs)" in source
    assert "fully_shard(model, **mp_fsdp_kwargs)" not in source
    assert "llm_layers, llm_path = _resolve" not in source


def test_native_patch_uses_parent_owned_runtime_dtype_for_checkpoint_recompute() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    assert patch.count("+        param_dtype = self.input_layernorm.weight.dtype") == 3
    assert patch.count("-        param_dtype = self.self_attn.q_proj.weight.dtype") == 2
    assert (
        patch.count("-            if att_output.dtype != self.self_attn.o_proj.weight.dtype:") == 3
    )
    assert (
        patch.count(
            "-                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)"
        )
        == 3
    )
    assert f"+++ b/{ACTION_DECODER_SOURCE}" in patch
    assert f"+++ b/{TEXT_DECODER_SOURCE}" in patch
    assert f"+++ b/{QWEN25_TEXT_DECODER_SOURCE}" in patch


def test_native_patch_adds_addresses_to_qk_only_and_keeps_values_content_only() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    assert patch.count("+        qk_input_bias: Optional[torch.Tensor] = None,") == 2
    assert (
        patch.count(
            "+                qk_hidden_states = hidden_states + "
            "qk_input_bias.to(hidden_states.dtype)"
        )
        == 2
    )
    assert patch.count("q_proj(qk_hidden_states)") >= 2
    assert patch.count("k_proj(qk_hidden_states)") >= 2
    assert "v_proj(qk_hidden_states)" not in patch
    assert patch.count("value_state = self.self_attn.v_proj(hidden_states)") >= 2


def test_native_patch_classifies_current_and_future_aux_fail_closed() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    assert '+        current_query_names = {"current_depth"}' in patch
    assert (
        '+        future_query_names = {"future_video_cls", "future_video", '
        '"future_depth"}'
    ) in patch
    assert "+        unexpected_query_names = set(query_spans) - (" in patch
    assert '+                "PICF refuses unclassified official query spans: "' in patch
    assert "+        host_current_mask = torch.zeros_like(" in patch
    assert "+        host_future_mask = torch.zeros_like(" in patch
    assert "+        visual_boundary_mask = (" in patch
    assert "+            prefix_pad_masks.bool() & ~visual_pos_masks.bool()" in patch
    assert "+        visual_boundary_mask[:, language_start:] = False" in patch
    assert "+            visual_boundary_mask=visual_boundary_mask," in patch
    assert (
        "+                host_current_mask if name in current_query_names "
        "else host_future_mask"
    ) in patch
    assert "+        host_current_mask &= prefix_pad_masks.bool()" in patch
    assert "+        host_future_mask &= prefix_pad_masks.bool()" in patch
    assert "+            host_current_mask=host_current_mask," in patch
    assert "+            host_future_mask=host_future_mask," in patch


def test_native_patch_injects_same_layer_memory_after_official_cache_capture() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    official_cache_tail = patch.index("                 use_cache=use_cache,")
    cache = patch.index("+            layer_attention_mask = attention_mask")
    memory = patch.index(
        "+                memory_inputs = self.picf_native_graph.layerwise_memory_inputs("
    )
    prepend = patch.index("+                key_states = torch.cat((memory_k, key_states), dim=1)")
    assert official_cache_tail < cache < memory < prepend
    assert "+                        _query_len," in patch
    assert "+                        _key_value_len," in patch
    assert patch.count("self.picf_native_graph.record_layerwise_posterior(") == 1


def test_native_patch_uses_one_layerwise_callback_surface_for_all_host_modes() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    shared_start = patch.index("@@ -320,7 +344,")
    shared_stop = patch.index("     def get_attention_interface", shared_start)
    shared_forward = patch[shared_start:shared_stop]
    callbacks = (
        "self.picf_native_graph.layerwise_qk_address_bias(",
        "self.picf_native_graph.layerwise_memory_inputs(",
        "self.picf_native_graph.record_layerwise_posterior(",
    )
    for callback in callbacks:
        assert callback in shared_forward
        assert patch.count(callback) == 1

    training_start = patch.index("+    def _picf_cached_training_forward(")
    training_stop = patch.index(
        '         """Do a full Qwen3-VL inference forward and compute the action."""',
        training_start,
    )
    training_forward = patch[training_start:training_stop]
    inference_start = training_stop
    inference_stop = patch.index("+    def picf_native_prior_forward(", inference_start)
    inference_prefix = patch[inference_start:inference_stop]
    assert "+    def _picf_cached_training_forward(" in training_forward
    assert "compact_lingbot_action_cache(" in training_forward
    assert "native_past_key_values=native_past_key_values," in training_forward
    assert "expanded_past_key_values=expanded_past_key_values," in training_forward
    assert "inputs_embeds=[None, suffix_embs]," in training_forward
    assert "picf_native_context=picf_native_context," in training_forward
    assert "picf_native_context=None," in training_forward
    assert "self.qwenvl_with_expert.forward(" in inference_prefix
    assert "picf_native_context=picf_native_context," in inference_prefix


def test_native_patch_exposes_explicit_suffix_action_attention_receipt() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    assert "+        picf_action_attention_callback=None," in patch
    policy_start = patch.index("@@ -1251,6 +1765,9 @@ class LingbotVlaV2Policy")
    policy_stop = patch.index("@@ -1313,7 +1832,22 @@ class LingbotVlaV2Policy", policy_start)
    policy_forward = patch[policy_start:policy_stop]
    assert policy_forward.count("+        picf_action_attention_callback=None,") == 1
    assert (
        policy_forward.count(
            "+            picf_action_attention_callback=picf_action_attention_callback,"
        )
        == 1
    )
    assert "+                    layer_index=layer_idx," in patch
    assert "+                    layer_count=num_layers," in patch
    assert "+            suffix_count=suffix_len if picf_action_attention_callback" in patch
    assert "+            action_layout = compact_cache.action_attention_layout" in patch
    assert "+                return picf_action_attention_callback(layout=action_layout" in patch
    assert patch.count("picf_action_attention_callback(") == 2

    prior_start = patch.index("+    def picf_native_prior_forward(")
    prior_stop = patch.index("+    def picf_native_observation_forward(", prior_start)
    prior_forward = patch[prior_start:prior_stop]
    assert "self.model.qwenvl_with_expert.forward(" in prior_forward
    assert "picf_native_context=picf_native_context," in prior_forward
    assert patch.count("def picf_native_frozen_posterior_action_forward(") == 1
    assert "run_registered_lingbot_frozen_posterior_action(self, request)" in patch
    assert "record_layerwise_prior" not in patch
    assert "layerwise_action" not in patch


def test_native_patch_fuses_shared_embedding_lookups() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    assert "+    def embed_language_and_special_tokens(" in patch
    assert "+            torch.cat((flat_language_tokens, special_tokens))" in patch
    assert "+    def embed_special_token(" not in patch
    assert (
        patch.count("+                self.qwenvl_with_expert.embed_language_and_special_tokens(")
        == 1
    )
    assert (
        patch.count("+            lang_emb = self.qwenvl_with_expert.embed_language_tokens(") == 1
    )


def test_native_patch_uses_exact_native_cache_and_only_authorized_picf_rows() -> None:
    patch = (ROOT / PATCH_RELATIVE_PATH).read_text()
    assert "+        prefix_position_pad_masks = prefix_pad_masks" in patch
    assert "+        _, native_past_key_values, _ = self.qwenvl_with_expert.forward(" in patch
    assert "+            _, expanded_past_key_values, _ = self.qwenvl_with_expert.forward(" in patch
    assert (
        "+            from picf_next.lingbot_native.host import compact_lingbot_action_cache"
        in patch
    )
    assert "+            compact_cache = compact_lingbot_action_cache(" in patch
    assert "+            past_key_values = compact_cache.past_key_values" in patch
    assert "+            prefix_pad_masks = compact_cache.valid" in patch
    assert "+            prefix_position_ids = compact_cache.position_ids" in patch
    assert "+            prefix_position_pad_masks = compact_cache.position_valid" in patch
    assert "+                prefix_position_pad_masks=prefix_position_pad_masks," in patch
    assert "+        prefix_position_pad_masks=None," in patch
    assert "+            prefix_position_pad_masks = prefix_pad_masks" in patch
    assert "+            prefix_position_pad_masks," in patch
    assert "prefix_pad_masks = prefix_pad_masks & action_cache_visible" not in patch


def test_native_patch_reads_selected_depths_after_deepstack_without_detaching() -> None:
    checkout = _native_source_checkout()
    if not checkout.exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    source = (checkout / MODEL_SOURCE).read_text()
    assert source.count("self.picf_native_graph.requires_intermediate_relation(") == 1
    assert source.count("self.picf_native_graph.record_intermediate_relation(") == 1
    assert source.count("normalized_prefix=models[0].norm(prefix_hidden)") == 1
    start = source.index("self.picf_native_graph.requires_intermediate_relation(")
    stop = source.index("inputs_embeds = outputs_embeds", start)
    callback = source[start:stop]
    assert ".detach(" not in callback
    assert ".clone(" not in callback
    assert source.rindex("self._apply_deepstack(", 0, start) < start


def test_native_patch_reuses_the_official_prefix_without_an_action_suffix() -> None:
    checkout = _native_source_checkout()
    if not checkout.exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    source = (checkout / MODEL_SOURCE).read_text()
    assert source.count("def picf_native_observation_forward(") == 2
    assert source.count("inputs_embeds=[prefix_embs, None]") == 5
    assert source.count("self._bind_picf_native_prefix(") == 3
    assert "use_cache=False" in source


def test_native_patch_bounds_muon_megabatches_and_casts_only_the_local_update() -> None:
    checkout = _native_source_checkout()
    if not checkout.exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    source = (checkout / MUON_SOURCE).read_text()
    assert "_MEGABATCH_MAX_GROUP_SIZE = 8" in source
    assert "for batch_start in range(0, N, _MEGABATCH_MAX_GROUP_SIZE)" in source
    assert source.count("_step_megabatch_chunk(") == 2
    mega_batch = source[source.index("    def _step_megabatch_chunk(") :]
    compute_cast = "stacked_compute = stacked_full.to(torch.bfloat16)"
    local_cast = "p_local.add_(ortho_local.to(dtype=p_local.dtype), alpha=-adjusted_lr)"
    mixed_device_local_cast = (
        "ortho_local.to(device=p_local.device, dtype=p_local.dtype)"
    )
    assert compute_cast in mega_batch
    assert mega_batch.index(compute_cast) < mega_batch.index("del stacked_full")
    assert mega_batch.index("del stacked_full") < mega_batch.index(
        "stacked_ortho = batched_newton_schulz("
    )
    muon_digest = hashlib.sha256((checkout / MUON_SOURCE).read_bytes()).hexdigest()
    if muon_digest == PATCHED_MUON_WITH_MIXED_DEVICE_MEGABATCH_SHA256:
        assert mixed_device_local_cast in mega_batch
        assert local_cast not in mega_batch
    else:
        assert local_cast in mega_batch
        assert mixed_device_local_cast not in mega_batch
    assert "batched_newton_schulz(stacked_full," not in mega_batch


def test_native_source_preparation_is_exact_and_idempotent(tmp_path: Path) -> None:
    source = _native_source_checkout()
    if not (source / ".git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    checkout = tmp_path / "lingbot-native"
    kwargs = {
        "checkout": checkout,
        "patch_path": ROOT / PATCH_RELATIVE_PATH,
        "source_url": str(source),
    }
    first = prepare_native_source(**kwargs)
    bytecode = checkout / "lingbotvla/__pycache__/generated.cpython-312.pyc"
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"generated bytecode is not executable source")
    second = prepare_native_source(**kwargs)
    assert first == second
    assert first["source_commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert first["patched_source_sha256"] == {
        str(CHECKPOINTER_SOURCE): PATCHED_CHECKPOINTER_SHA256,
        str(PARALLEL_SOURCE): PATCHED_PARALLEL_SHA256,
        str(MODEL_SOURCE): PATCHED_MODEL_SHA256,
        str(ACTION_DECODER_SOURCE): PATCHED_ACTION_DECODER_SHA256,
        str(TEXT_DECODER_SOURCE): PATCHED_TEXT_DECODER_SHA256,
        str(QWEN25_TEXT_DECODER_SOURCE): PATCHED_QWEN25_TEXT_DECODER_SHA256,
        str(MUON_SOURCE): PATCHED_MUON_SHA256,
    }
    assert not bytecode.exists()
    assert _dirty_paths(checkout) == {str(path) for path in PATCHED_SOURCES}

    unknown = checkout / "lingbotvla/__pycache__/not-bytecode.txt"
    unknown.parent.mkdir(parents=True)
    unknown.write_text("must not be silently removed\n")
    with pytest.raises(ValueError, match="bytecode cache contains an unknown artifact"):
        prepare_native_source(**kwargs)
    unknown.unlink()
    unknown.parent.rmdir()

    with (checkout / MODEL_SOURCE).open("a") as stream:
        stream.write("\n# undeclared mutation\n")
    with pytest.raises((ValueError, RuntimeError), match="digest|exact"):
        prepare_native_source(**kwargs)


def test_native_source_with_muon_collective_hotfix_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    source = _native_source_checkout()
    if not (source / ".git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    checkout = tmp_path / "lingbot-native-muon-hotfix"
    kwargs = {
        "checkout": checkout,
        "patch_path": ROOT / PATCH_RELATIVE_PATH,
        "hotfix_path": ROOT / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
        "source_url": str(source),
    }
    first = prepare_native_source_with_muon_collective_hotfix(**kwargs)
    second = prepare_native_source_with_muon_collective_hotfix(**kwargs)
    validated = validate_prepared_native_source_with_muon_collective_hotfix(
        checkout=checkout,
        patch_path=ROOT / PATCH_RELATIVE_PATH,
        hotfix_path=ROOT / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
    )
    assert first == second == validated
    assert first["runtime_hotfix_sha256"] == MUON_COLLECTIVE_HOTFIX_SHA256
    assert first["patched_source_sha256"][str(MUON_SOURCE)] == (
        PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256
    )
    assert _dirty_paths(checkout) == {str(path) for path in PATCHED_SOURCES}


def test_prepared_native_source_rejects_every_unrelated_dirty_path(tmp_path: Path) -> None:
    source = _native_source_checkout()
    if not (source / ".git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    checkout = tmp_path / "lingbot-native"
    patch_path = ROOT / PATCH_RELATIVE_PATH
    prepare_native_source(
        checkout=checkout,
        patch_path=patch_path,
        source_url=str(source),
    )

    unrelated = checkout / "lingbotvla/unreviewed_runtime.py"
    unrelated.write_text("raise RuntimeError('unreviewed')\n")
    with pytest.raises(ValueError, match="unrelated changes"):
        validate_prepared_native_source(checkout=checkout, patch_path=patch_path)

    unrelated.unlink()
    bytecode = checkout / "lingbotvla/__pycache__/generated.cpython-312.pyc"
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"generated bytecode")
    report = validate_prepared_native_source(checkout=checkout, patch_path=patch_path)
    assert report["patch_state"] == "applied"
    assert not bytecode.exists()


def test_bytecode_cleanup_is_idempotent_when_a_peer_removes_a_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bytecode = tmp_path / "lingbotvla/__pycache__/generated.cpython-312.pyc"
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"generated bytecode")
    original_unlink = Path.unlink
    injected = False

    def peer_raced_unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if path == bytecode and not injected:
            injected = True
            original_unlink(path)
            raise FileNotFoundError(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", peer_raced_unlink)
    _purge_generated_python_bytecode(tmp_path)
    assert injected
    assert not bytecode.parent.exists()


def test_bytecode_cleanup_is_idempotent_when_a_peer_removes_a_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "lingbotvla/__pycache__"
    cache.mkdir(parents=True)
    original_rmdir = Path.rmdir
    injected = False

    def peer_raced_rmdir(path: Path) -> None:
        nonlocal injected
        if path == cache and not injected:
            injected = True
            original_rmdir(path)
            raise FileNotFoundError(path)
        original_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", peer_raced_rmdir)
    _purge_generated_python_bytecode(tmp_path)
    assert injected
    assert not cache.exists()


def test_native_patch_verifier_rejects_legacy_or_incomplete_artifacts(tmp_path: Path) -> None:
    patch = tmp_path / PATCH_RELATIVE_PATH
    patch.parent.mkdir(parents=True)
    patch.write_text("set_action_layer_adapter\n")
    with pytest.raises(ValueError, match="digest"):
        verify_native_patch(root=tmp_path, check_apply=False)


def test_native_overlay_install_command_is_explicit_and_dependency_neutral(
    tmp_path: Path,
) -> None:
    command = picf_overlay_install_command(
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path / "repo",
    )
    assert command[:4] == [
        str((tmp_path / "venv/bin/python").absolute()),
        "-m",
        "pip",
        "install",
    ]
    assert "--no-deps" in command
    assert "--editable" in command
    assert command[-1] == str((tmp_path / "repo").resolve())


def test_native_runtime_repair_and_audit_tools_use_selected_python(tmp_path: Path) -> None:
    source = _native_source_checkout()
    if not source.exists():
        pytest.skip("optional pinned LingBot requirements are absent")
    python = tmp_path / "persistent-env/bin/python"
    runtime = native_runtime_restore_command(python=python, source_checkout=source)
    assert runtime[:4] == [str(python.absolute()), "-m", "pip", "install"]
    assert "--no-deps" in runtime
    assert runtime.count("-r") == 2
    assert str((source / "requirements.txt").resolve()) in runtime
    assert runtime[-1] == str((source / "requirements-depth.txt").resolve())
    audit = native_audit_tools_install_command(python=python)
    assert audit[:4] == [str(python.absolute()), "-m", "pip", "install"]
    assert "--no-deps" in audit
    assert "iniconfig==2.3.0" in audit
    assert "pluggy==1.6.0" in audit
    assert "pygments==2.20.0" in audit
    assert "pytest==9.1.1" in audit
    assert "ruff==0.15.21" in audit


def test_calvin_offline_runtime_is_source_bound_and_exact(tmp_path: Path) -> None:
    environment = ROOT / "references/source_checkouts/calvin/calvin_env"
    if not (environment / ".git").exists():
        pytest.skip("optional pinned CALVIN source checkout is absent")
    report = validate_calvin_offline_source(environment)
    assert report["status"] == "PASS"
    assert report["calvin_commit"] == CALVIN_SOURCE_COMMIT
    assert report["calvin_env_commit"] == CALVIN_ENV_SOURCE_COMMIT
    python = tmp_path / "persistent-env/bin/python"
    command = calvin_offline_runtime_install_command(
        python=python,
        calvin_env_root=environment,
    )
    assert command[:4] == [str(python.absolute()), "-m", "pip", "install"]
    assert "--no-deps" in command
    assert {value for value in command if "==" in value} == {
        f"{name}=={version}" for name, version in CALVIN_OFFLINE_RUNTIME_EXTRAS.items()
    }
    assert "opencv-python==4.11.0.86" not in command


def test_native_depth_repair_uses_moge_pin_and_no_dependency_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "lingbot"
    moge = source / MOGE_SOURCE
    depth = source / LINGBOT_DEPTH_SOURCE
    moge.mkdir(parents=True)
    depth.mkdir(parents=True)
    (moge / "pyproject.toml").write_text(
        f"utils3d @ git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_SOURCE_COMMIT}\n"
    )
    utils3d = tmp_path / "utils3d"
    utils3d.mkdir()

    def fake_run(command: list[str], *, cwd: Path | None = None) -> str:
        del cwd
        assert command == ["git", "rev-parse", "HEAD"]
        return UTILS3D_SOURCE_COMMIT

    monkeypatch.setattr("tools.bootstrap_lingbot_vla2_native._run", fake_run)
    monkeypatch.setattr("tools.bootstrap_lingbot_vla2_native._dirty_paths", lambda _: set())
    python = tmp_path / "env/bin/python"
    commands = native_depth_runtime_install_commands(
        python=python,
        source_checkout=source,
        utils3d_checkout=utils3d,
    )
    assert tuple(command[-1] for command in commands) == (
        str(utils3d.resolve()),
        str(depth.resolve()),
        str(moge.resolve()),
    )
    for command in commands:
        assert command[:4] == [str(python.absolute()), "-m", "pip", "install"]
        assert "--no-deps" in command
        assert "--editable" in command


def test_native_depth_path_is_written_by_the_selected_python(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", str(runtime)], check=True)
    utils3d = tmp_path / "utils3d"
    utils3d.mkdir()
    path_file = Path(
        write_native_depth_path(
            python=runtime / "bin/python",
            utils3d_checkout=utils3d,
        )
    )
    assert path_file.is_file()
    assert not path_file.is_symlink()
    assert path_file.read_text(encoding="ascii") == f"{utils3d.resolve()}\n"


def test_native_bootstrap_defaults_to_the_verified_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["bootstrap_lingbot_vla2_native.py"])
    args = _parse_args()
    assert args.checkout == ROOT / CHECKOUT_RELATIVE_PATH
    assert args.utils3d_checkout == ROOT / UTILS3D_CHECKOUT_RELATIVE_PATH


def test_native_bootstrap_requires_an_explicit_python_for_overlay_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["bootstrap_lingbot_vla2_native.py", "--install-overlay"],
    )
    args = _parse_args()
    assert args.install_overlay
    assert args.python is None


@pytest.mark.parametrize(
    "flag",
    [
        "--repair-depth-runtime",
        "--restore-runtime-pins",
        "--install-audit-tools",
        "--install-calvin-offline-runtime",
    ],
)
def test_native_bootstrap_runtime_operations_require_selected_python(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["bootstrap_lingbot_vla2_native.py", flag])
    args = _parse_args()
    assert args.python is None
    assert (
        args.repair_depth_runtime
        or args.restore_runtime_pins
        or args.install_audit_tools
        or args.install_calvin_offline_runtime
    )

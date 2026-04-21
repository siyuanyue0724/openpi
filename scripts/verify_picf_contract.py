from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openpi.picf.test_utils import build_mini_calvin_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from picf_core_train_smoke import run_smoke
from openpi.picf.core.config import PicfCoreConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "core" / "pipeline.py"
README_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "README.md"
V21_README_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "README_v2.1.md"
V22_README_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "README_v2.2.md"
CALVIN_README_PATH = REPO_ROOT / "docs" / "CALVIN_VALIDATION_README.md"
FORMAL_CONTRACT_PATH = REPO_ROOT / "PICF_FORMAL_CONTRACT.md"
PALIGEMMA_WRAPPER_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "paligemma" / "wrapper.py"
TRAINER_SCRIPT_PATH = REPO_ROOT / "scripts" / "picf_core_train.py"
SERVE_SCRIPT_PATH = REPO_ROOT / "scripts" / "serve_picf_policy.py"
POLICY_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "policy.py"
CORE_CONTRACTS_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "core" / "contracts.py"
GEMMA_PYTORCH_PATH = REPO_ROOT / "src" / "openpi" / "models_pytorch" / "gemma_pytorch.py"
SONATA_WRAPPER_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "sonata" / "wrapper.py"
ANYTOUCH_WRAPPER_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "anytouch" / "wrapper.py"
VJEPA_WRAPPER_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "vjepa" / "wrapper.py"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"Function {name!r} not found in AST.")


def _node_source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise RuntimeError("Failed to recover source segment from AST.")
    return segment


def _attribute_strings(node: ast.AST) -> set[str]:
    found: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, attr: ast.Attribute) -> None:
            parts: list[str] = []
            cur: ast.AST | None = attr
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                found.add(".".join(reversed(parts)))
            self.generic_visit(attr)

    Visitor().visit(node)
    return found


def _call_order(source: str, func_source: str, call_texts: list[str]) -> CheckResult:
    positions = []
    for call in call_texts:
        index = func_source.find(call)
        if index < 0:
            return CheckResult(
                name=f"order:{' > '.join(call_texts)}",
                ok=False,
                detail=f"Missing call text {call!r}.",
            )
        positions.append(index)
    ok = positions == sorted(positions)
    return CheckResult(
        name=f"order:{' > '.join(call_texts)}",
        ok=ok,
        detail=" -> ".join(f"{call}@{pos}" for call, pos in zip(call_texts, positions)),
    )


def verify_static_contract() -> list[CheckResult]:
    source = _read(PIPELINE_PATH)
    wrapper_source = _read(PALIGEMMA_WRAPPER_PATH)
    trainer_source = _read(TRAINER_SCRIPT_PATH)
    serve_source = _read(SERVE_SCRIPT_PATH)
    policy_source = _read(POLICY_PATH)
    policy_test_source = _read(REPO_ROOT / "src" / "openpi" / "picf" / "policy_test.py")
    readme_v22_source = _read(V22_README_PATH)
    calvin_readme_source = _read(CALVIN_README_PATH)
    contracts_source = _read(CORE_CONTRACTS_PATH)
    gemma_pytorch_source = _read(GEMMA_PYTORCH_PATH)
    sonata_wrapper_source = _read(SONATA_WRAPPER_PATH)
    anytouch_wrapper_source = _read(ANYTOUCH_WRAPPER_PATH)
    vjepa_wrapper_source = _read(VJEPA_WRAPPER_PATH)
    tree = ast.parse(source)
    defaults = PicfCoreConfig()

    posterior_node = _function_node(tree, "_posterior_update")
    posterior_attrs = _attribute_strings(posterior_node)
    posterior_source = _node_source(source, posterior_node)

    innovation_node = _function_node(tree, "_innovation")
    innovation_attrs = _attribute_strings(innovation_node)
    innovation_source = _node_source(source, innovation_node)

    prev_action_node = _function_node(tree, "_previous_action")
    prev_action_source = _node_source(source, prev_action_node)

    public_memory_node = _function_node(tree, "_build_public_read_memory")
    public_memory_source = _node_source(source, public_memory_node)

    task_readout_node = _function_node(tree, "_build_task_readout")
    task_readout_source = _node_source(source, task_readout_node)
    task_readout_attrs = _attribute_strings(task_readout_node)

    conditioned_control_node = _function_node(tree, "_build_conditioned_control_state")
    conditioned_control_source = _node_source(source, conditioned_control_node)

    conditioned_predictive_node = _function_node(tree, "_build_conditioned_predictive_cache")
    conditioned_predictive_source = _node_source(source, conditioned_predictive_node)

    observe_node = _function_node(tree, "observe_step")
    observe_source = _node_source(source, observe_node)

    predictive_node = _function_node(tree, "_predictive_state")
    predictive_source = _node_source(source, predictive_node)
    current_targets_node = _function_node(tree, "_current_targets")
    current_targets_source = _node_source(source, current_targets_node)

    finalize_node = _function_node(tree, "finalize_with_action")
    finalize_source = _node_source(source, finalize_node)

    step_node = _function_node(tree, "step")
    step_source = _node_source(source, step_node)

    checks = [
        CheckResult(
            name="paligemma_wrapper_restores_pi05_expert_stack",
            ok=(
                "PaliGemmaWithExpertModel" in wrapper_source
                and "self.paligemma_with_expert = self._build_paligemma_with_expert(" in wrapper_source
                and "self.action_in_proj = nn.Linear(self.model_action_dim" in wrapper_source
                and "self.time_mlp_in = nn.Linear(" in wrapper_source
            ),
            detail="PICF semantic wrapper restores the PI0.5 expert stack and suffix action projections.",
        ),
        CheckResult(
            name="gemma_training_honors_semantic_checkpointing_flag",
            ok=(
                "Forcing gradient checkpointing to be enabled for Gemma expert model" not in gemma_pytorch_source
                and "self.gemma_expert.model.gradient_checkpointing = True" not in gemma_pytorch_source
            ),
            detail=(
                "Gemma expert training no longer force-enables gradient checkpointing; "
                "trainer/wrapper checkpointing policy remains authoritative."
            ),
        ),
        CheckResult(
            name="gemma_dual_branch_attention_uses_sdpa_not_eager_workspace",
            ok=(
                "transformers_sdpa_attention_forward(" in gemma_pytorch_source
                and "modeling_gemma.eager_attention_forward(" not in gemma_pytorch_source
            ),
            detail=(
                "The custom PI0/Gemma dual-branch attention path now uses SDPA instead of the eager "
                "attention workspace, preserving training semantics while avoiding the large attention "
                "buffer that exhausts 40GB A100s once optimizer state is resident."
            ),
        ),
        CheckResult(
            name="semantic_runtime_drops_unused_generation_heads",
            ok=(
                "self._drop_unused_generation_heads()" in wrapper_source
                and "def _drop_unused_generation_heads(self) -> None:" in wrapper_source
                and 'self.paligemma_with_expert.paligemma.lm_head = None' in wrapper_source
                and 'self.paligemma_with_expert.gemma_expert.lm_head = None' in wrapper_source
            ),
            detail=(
                "The PI0/PICF semantic runtime now drops the unused outer causal-LM heads after checkpoint load, "
                "so dead generation weights do not inflate FSDP wrapping or optimizer enumeration."
            ),
        ),
        CheckResult(
            name="fsdp_keeps_semantic_gradient_checkpointing_enabled",
            ok=(
                "args.semantic_gradient_checkpointing_disabled_for_fsdp = True" not in trainer_source
                and "Semantic contract: disabled PaliGemma gradient checkpointing for training_strategy=" not in trainer_source
                and "Semantic contract: FSDP keeps the PI0/PaliGemma stack at one shard boundary" in trainer_source
            ),
            detail=(
                "FSDP no longer force-disables semantic gradient checkpointing; "
                "the single-boundary PI0/PaliGemma contract keeps non-reentrant checkpoint recomputation live."
            ),
        ),
        CheckResult(
            name="fsdp_uses_backward_post_and_core_recompute",
            ok=(
                '"use_orig_params": False' in trainer_source
                and
                "BackwardPrefetch.BACKWARD_POST" in trainer_source
                and "activation_checkpointing=True" in source
                and "torch.utils.checkpoint.checkpoint(" in source
                and '"window_activation_checkpointing"' in trainer_source
                and "_window_outputs_to_tensor_tuple(" in trainer_source
                and "_checkpoint_dummy_input(" in trainer_source
                and "preserve_rng_state=True" in trainer_source
                and "args.diagnostic_interval = 0" in trainer_source
            ),
            detail=(
                "Standard v2.2 FSDP training now reduces backward peak memory with flat-parameter sharding "
                "(use_orig_params=False), BACKWARD_POST, train-time core transformer activation recompute, "
                "and an exact whole-window recompute fallback that uses a standalone dummy leaf input instead "
                "of a flat-parameter view so recompute does not leak full-parameter gradients back into local "
                "shard metadata."
            ),
        ),
        CheckResult(
            name="trainer_threads_v22_live_conditioned_control_knobs",
            ok=all(
                text in trainer_source
                for text in (
                    'parser.add_argument("--task-local-queries"',
                    'parser.add_argument("--task-global-queries"',
                    'parser.add_argument("--task-instruction-queries"',
                    'parser.add_argument("--task-self-layers"',
                    'parser.add_argument("--conditioned-control-queries"',
                    'parser.add_argument("--pi-prefix-queries"',
                    'parser.add_argument("--conditioned-future-queries"',
                    'parser.add_argument("--task-visual-reread-topk"',
                    'parser.add_argument("--task-tactile-reread-groups"',
                    'parser.add_argument("--task-point-reread-topk"',
                    'task_local_queries=args.task_local_queries',
                    'task_global_queries=args.task_global_queries',
                    'task_instruction_queries=args.task_instruction_queries',
                    'task_self_layers=args.task_self_layers',
                    'conditioned_control_queries=args.conditioned_control_queries',
                    'pi_prefix_queries=args.pi_prefix_queries',
                    'conditioned_future_queries=args.conditioned_future_queries',
                    'task_visual_reread_topk=int(',
                    'task_tactile_reread_groups=int(',
                    'task_point_reread_topk=int(',
                    'require_pi0_action_generator=bool(',
                )
            ),
            detail=(
                "The trainer/parser now exposes and threads the active v2.2 conditioned-control/task-readout "
                "knobs into PicfCoreConfig instead of leaving them as README-only declarations."
            ),
        ),
        CheckResult(
            name="fsdp_recursively_splits_large_uniform_subtrees_and_shards_safe_core_stacks",
            ok=all(
                text in trainer_source
                for text in (
                    "_FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES = 512 * 1024 * 1024",
                    'getattr(core, "token_fusion", None)',
                    'getattr(core, "obs_self", None)',
                    'getattr(core, "posterior_self", None)',
                    'getattr(core, "task_self", None)',
                    'getattr(core, "predictive_world", None)',
                    'getattr(core, "predictive_semantic_world", None)',
                    'getattr(core, "control_world", None)',
                    "subtree_param_bytes <= _FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES",
                    "def _assign_fsdp_wrapped_child_module(",
                    "_assign_fsdp_wrapped_child_module(model, original=child, wrapped=wrapped)",
                    "def _fsdp_root_ignored_modules(",
                    "ignored_modules",
                )
            ),
            detail=(
                "Standard 4x40GB v2.2 FSDP now recursively splits large uniform-dtype subtrees and "
                "reattaches the safe core transformer stacks as dedicated boundaries before the root wrapper, "
                "while the root FSDP boundary ignores fully frozen backbone subtrees instead of flattening "
                "mixed requires_grad parameter sets."
            ),
        ),
        CheckResult(
            name="vjepa_mixed_precision_uses_safe_autocast_for_frozen_and_trainable_paths",
            ok=(
                "def _vjepa_uses_autocast(" in vjepa_wrapper_source
                and "_trainable_vjepa_uses_autocast" not in vjepa_wrapper_source
                and "if _vjepa_uses_autocast(device=self.device, dtype=self.dtype):" in vjepa_wrapper_source
                and "use_autocast = _vjepa_uses_autocast(device=self.device, dtype=self.dtype)" in vjepa_wrapper_source
            ),
            detail=(
                "V-JEPA mixed precision now uses one CUDA autocast contract for both frozen and trainable paths, "
                "avoiding bf16/fp32 conv bias mismatches while preserving the encoder's native fp32 weights."
            ),
        ),
        CheckResult(
            name="transformer_stacks_break_batched_view_aliases_before_attention",
            ok=(
                "x = x.clone()" in source
                and "storage aliasing is not reliably exposed via" in source
                and "multi-view in-place checks from tripping" in source
            ),
            detail=(
                "Transformer stacks now materialize stack inputs once at entry so FSDP full-shard "
                "does not trip autograd multi-view alias errors inside residual attention blocks."
            ),
        ),
        CheckResult(
            name="fsdp_grad_clipping_uses_explicit_global_l2_reduction",
            ok=(
                "def _fsdp_global_grad_l2_norm(" in trainer_source
                and "dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)" in trainer_source
                and "return _fsdp_global_grad_l2_norm(model)" in trainer_source
                and "return _fsdp_clip_grad_norm_exact(model, max_norm=float(max_norm))" in trainer_source
            ),
            detail=(
                "Mixed-dtype FSDP training now computes grad norm and percentile clipping via an explicit "
                "global L2 reduction over local gradient shards instead of relying on clip_grad_norm_."
            ),
        ),
        CheckResult(
            name="parallel_rank_model_build_is_standard_path",
            ok=(
                "for build_rank in range(int(world_size))" not in trainer_source
                and "return _build_model(args, device=device)" in trainer_source
            ),
            detail=(
                "Standard v2.2 full-finetune startup now builds the stack on each rank in parallel "
                "instead of serializing checkpoint construction through rank barriers."
            ),
        ),
        CheckResult(
            name="semantic_checkpoint_startup_uses_node_local_staging_from_shared_storage",
            ok=all(
                text in wrapper_source
                for text in (
                    "def _stage_pi0_checkpoint_if_needed(",
                    'os.environ.get("OPENPI_LOCAL_CHECKPOINT_CACHE_DIR", "")',
                    'os.environ.get("OPENPI_STAGE_PI0_CHECKPOINT", "auto")',
                    'return str(resolved).startswith("/mnt/")',
                    "logging.info(",
                    "Staged PI0 checkpoint locally:",
                    "self.config = _stage_local_pi0_config(config or PaliGemmaSemanticConfig())",
                )
            ),
            detail=(
                "Standard multi-rank startup now stages the shared PI0/PaliGemma checkpoint into a node-local cache "
                "before rank-local weight loading, removing `/mnt` page-wait stalls without changing training semantics."
            ),
        ),
        CheckResult(
            name="foundation_backbones_use_train_recompute_under_fsdp",
            ok=(
                "def _apply_train_checkpoint(self, func, *args):" in sonata_wrapper_source
                and "torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False)" in sonata_wrapper_source
                and "feat = self._encode_stage_checkpointed(sample)" in sonata_wrapper_source
                and "def _apply_train_checkpoint(self, func, *args):" in anytouch_wrapper_source
                and "tokens = self._apply_train_checkpoint(lambda clip_batch: self.model(clip_batch, sensor_id_tensor, probe=True), inputs)" in anytouch_wrapper_source
            ),
            detail=(
                "Trainable Sonata and AnyTouch backbones now use non-reentrant train-time recompute, "
                "so v2.2 FSDP full-finetune reduces backbone activation peaks without changing the objective."
            ),
        ),
        CheckResult(
            name="tokenwise_exact_chunking_is_wired_for_core_and_semantic_hot_paths",
            ok=(
                "tokenwise_ff_chunk_size" in trainer_source
                and "semantic_tokenwise_chunk_size" in trainer_source
                and "semantic_projection_chunk_size" in trainer_source
                and "semantic_mlp_chunk_size" in trainer_source
                and "def _apply_tokenwise_in_chunks(" in source
                and "def _apply_tokenwise_in_chunks(" in gemma_pytorch_source
                and "tokenwise_chunk_size =" in wrapper_source
                and 'self.__dict__.get("config")' in wrapper_source
                and "chunk_size=self.projection_chunk_size" in gemma_pytorch_source
                and "chunk_size=self.mlp_chunk_size" in gemma_pytorch_source
                and "chunk_size=self.ff_chunk_size" in source
            ),
            detail=(
                "Exact tokenwise chunking is now wired into the current hottest core and PI0/Gemma "
                "token-local paths, so sequence-local projections and FFNs can lower activation peaks "
                "without changing model math."
            ),
        ),
        CheckResult(
            name="semantic_runtime_hot_leaves_are_nested_fsdp_ready",
            ok=(
                "def fsdp_runtime_leaf_module_specs(self) -> list[tuple[nn.Module, str, str]]:" in wrapper_source
                and "def _prepare_semantic_runtime_leaf_fsdp(" in trainer_source
                and '_prepare_semantic_runtime_leaf_fsdp(child, device=device)' in trainer_source
                and "module_parameter_dtype" in wrapper_source
                and "module_num_embeddings" in wrapper_source
                and "module_parameter_dtype" in gemma_pytorch_source
            ),
            detail=(
                "The semantic runtime now exposes directly called hot leaves for nested exact FSDP wrapping, "
                "and trainer/wrapper code use FSDP-safe module metadata instead of raw `.weight` attribute access."
            ),
        ),
        CheckResult(
            name="fsdp_defaults_expandable_segments_allocator_contract",
            ok=(
                'applied = "expandable_segments:True"' in trainer_source
                and "PYTORCH_CUDA_ALLOC_CONF" in trainer_source
                and "expandable_segments:True" in trainer_source
                and "all-backbone CUDA training expects" in trainer_source
            ),
            detail=(
                "FSDP full-finetune now standardizes the CUDA allocator contract on "
                "expandable_segments:True instead of leaving fragmentation behavior implicit."
            ),
        ),
        CheckResult(
            name="paligemma_tokenization_injects_state_into_prompt",
            ok="self.tokenizer.tokenize(str(prompt), state=state)" in wrapper_source,
            detail="PICF semantic tokenization injects robot state back into the PI0.5 prompt path.",
        ),
        CheckResult(
            name="paligemma_flow_training_recovers_denoised_chunk_not_velocity",
            ok=(
                "def _recover_flow_target(" in wrapper_source
                and "predicted_chunk = _recover_flow_target(x_t, v_t, time_expanded).detach()" in wrapper_source
                and '"predicted_chunk": predicted_chunk[0]' in wrapper_source
            ),
            detail="PI0.5 flow training recovers the denoised chunk estimate x_t - t * v_t instead of treating velocity as the action chunk.",
        ),
        CheckResult(
            name="posterior_update_excludes_semantic",
            ok="semantic" not in posterior_source and not any("semantic" in attr for attr in posterior_attrs),
            detail="`_posterior_update` contains no semantic references.",
        ),
        CheckResult(
            name="innovation_reads_physical_prediction_cache",
            ok="previous.predictive.physical_prediction_cache" in innovation_source,
            detail="Innovation constructor references previous physical prediction cache.",
        ),
        CheckResult(
            name="innovation_excludes_semantic_conditioned_cache",
            ok=not any(
                attr.startswith("previous.predictive.prediction_cache")
                or attr.startswith("previous.predictive.global_pred")
                or attr.startswith("previous.predictive.semantic_tokens")
                or attr.startswith("previous.predictive.predictive_query_state")
                or attr.startswith("previous.predictive.control_query_state")
                for attr in innovation_attrs
            ),
            detail="Innovation constructor does not read semantic-conditioned cache/global_pred/previous semantic fields.",
        ),
        CheckResult(
            name="task_query_conditioner_gate_is_live",
            ok="gate_init=1.0" in source and "self.task_query_conditioner = GatedCrossAttentionRead(" in source,
            detail="Task-query conditioner is initialized with a live cross gate instead of the dormant default.",
        ),
        CheckResult(
            name="public_read_memory_uses_fused_tokens_plus_visual_tokens",
            ok=(
                "token_field.fused_tokens" in public_memory_source
                and "token_field.visual_tokens" in public_memory_source
                and "token_field.point_tokens" not in public_memory_source
                and "token_field.tactile_tokens_active" not in public_memory_source
                and "token_field.context_tokens" not in public_memory_source
            ),
            detail="Public task-readout memory is defined as fused public tokens plus explicit visual tokens.",
        ),
        CheckResult(
            name="task_readout_reads_current_observation_memory_only",
            ok="posterior" not in task_readout_source and not any("posterior" in attr for attr in task_readout_attrs),
            detail="Task readout does not directly consume posterior state.",
        ),
        CheckResult(
            name="task_readout_uses_public_memory_and_private_dense_rereads",
            ok=all(
                text in task_readout_source
                for text in (
                    "public_read_memory = self._build_public_read_memory(token_field)",
                    "dense_memory.visual_payload",
                    "dense_memory.tactile_group_tokens",
                    "dense_memory.point_payload",
                )
            ),
            detail="Task readout consumes public fused/visual memory plus private dense visual/tactile/point payloads.",
        ),
        CheckResult(
            name="task_readout_derives_point_and_tactile_routing_from_fused_attention",
            ok=all(
                text in task_readout_source
                for text in (
                    "point_public_attention = fused_attention[:, :point_count]",
                    "tactile_public_attention = fused_attention[:, point_count : point_count + tactile_count]",
                )
            ),
            detail="Task readout derives point/tactile routing from fused public attention instead of rereading raw public point/tactile streams.",
        ),
        CheckResult(
            name="conditioned_control_state_is_single_canonical_route",
            ok=all(
                text in conditioned_control_source
                for text in (
                    "base_tokens = torch.cat(",
                    "task_tokens = torch.cat(",
                    "control_tokens = self.control_world(control_prefix[None, :])[0]",
                    "pi_prefix_tokens, _ = self.pi_prefix_reader(",
                    "future_condition_tokens, _ = self.future_condition_reader(",
                )
            )
            and "semantic.prefix_tokens" not in conditioned_control_source,
            detail="Control semantics are unified into a single conditioned-control trunk without raw semantic-prefix injection.",
        ),
        CheckResult(
            name="conditioned_predictive_cache_is_token_level_from_physical_pred_and_future_condition",
            ok=all(
                text in conditioned_predictive_source
                for text in (
                    "conditioned_physical_pred_tokens = self.physical_pred_to_conditioned_proj(physical_pred_tokens)",
                    "_add_role_embedding(conditioned_physical_pred_tokens, self.predictive_conditioned_role_embedding, 0)",
                    "_add_role_embedding(future_condition_tokens, self.predictive_conditioned_role_embedding, 1)",
                    "pred_tokens = self.predictive_semantic_world(pred_conditioned_tokens[None, :])[0]",
                )
            )
            and all(
                text not in conditioned_predictive_source
                for text in (
                    "semantic.prefix_tokens",
                    "semantic.tokens",
                    "semantic_override",
                    "conditioned_semantic_prefix_tokens",
                )
            ),
            detail="Conditioned future cache is built from physical predictive token sequence plus future-condition tokens only.",
        ),
        CheckResult(
            name="previous_action_prefers_executed_action",
            ok='getattr(previous.predictive, "executed_action", None)' in prev_action_source and "previous.predictive.action" in prev_action_source,
            detail="`_previous_action` uses executed_action first, action as fallback.",
        ),
        _call_order(
            source,
            observe_source,
            [
                "posterior = self._posterior_update(",
                "innovation_token, innovation_norm = self._innovation(",
                "task_readout = self._build_task_readout(",
                "conditioned_control = self._build_conditioned_control_state(",
            ],
        ),
        _call_order(
            source,
            predictive_source,
            [
                "physical_pred_tokens, physical_global_pred, physical_prediction_cache = self._build_physical_predictive_basis(",
                "predictive_query_state, global_pred, prediction_cache = self._build_conditioned_predictive_cache(",
            ],
        ),
        CheckResult(
            name="step_is_compat_wrapper_over_observe_and_finalize",
            ok=all(
                text in step_source
                for text in (
                    "observed = self.observe_step(",
                    "return self.finalize_with_action(",
                )
            ),
            detail="Core step is now only a compatibility wrapper over observe/finalize.",
        ),
        CheckResult(
            name="live_visual_native_first_competition_path",
            ok=(
                "return _to_tensor(fmap.current_map(use_last_two_mean=self.visual_config.use_last_two_mean), device=self.device, dtype=self.dtype)"
                in source
                and "all_tokens = torch.cat([point_tokens, tactile_tokens_active, context_tokens], dim=0)" in source
                and "queries, visual_weights = self.visual_native_reread(queries, dense_memory.visual_payload[None, :])" in source
                and "queries, attn_public = self.obs_reader(queries, token_field.fused_tokens[None, :])" in source
            ),
            detail=(
                "Live PICF observation competition is now visual-native-first: native V-JEPA payload is re-read "
                "before the public fused point/tactile/context path."
            ),
        ),
        CheckResult(
            name="live_visual_native_reread_is_active_in_posterior",
            ok=(
                "self.visual_native_reread = LazyCrossAttentionRead(hidden_dim, inner_dim=hidden_dim)" in source
                and "dense_memory.visual_payload" in posterior_source
                and "visual_read, _ = self.visual_native_reread(" in posterior_source
            ),
            detail="Live PICF posterior now re-reads native visual payload after anchor competition.",
        ),
        CheckResult(
            name="live_visual_innovation_uses_native_latent_probes",
            ok=(
                "self.visual_latent_queries = nn.Parameter(" in source
                and "def _visual_latent_target(" in source
                and 'targets["visual_latent"] = self._visual_latent_target(dense_memory)' in current_targets_source
            ),
            detail="Live PICF visual innovation target now uses latent probes read from native current-step V-JEPA payload.",
        ),
        CheckResult(
            name="live_anytouch_public_path_uses_group_routing_proposals",
            ok=all(
                text in source
                for text in (
                    "sensor.pooled_feature.to(device=self.device, dtype=self.dtype)",
                    "self.tactile_group_route_queries = nn.Parameter(torch.zeros((self.config.tactile_group_proposals, hidden_dim)))",
                    "route_tokens, _ = self.tactile_route_reread(route_queries, dense_group[None, :])",
                    "tactile_group_ids = torch.cat(proposal_group_ids, dim=0)",
                )
            ),
            detail=(
                "Live PICF tactile public routing now uses pooled sensor proposals plus multi-proposal group routing; "
                "ownership stays group-level via tactile_group_ids."
            ),
        ),
        CheckResult(
            name="live_tactile_group_winner_read_is_active_in_posterior",
            ok=(
                "routing_mass_tactile" in source
                and "dense_memory.tactile_group_tokens" in posterior_source
                and "tactile_read, _ = self.tactile_native_reread(" in posterior_source
            ),
            detail="Live PICF posterior now uses tactile group routing plus winner-read over full dense tactile tokens.",
        ),
        CheckResult(
            name="live_tactile_innovation_uses_dense_latent_plus_map_and_aux",
            ok=(
                "self.tactile_latent_queries = nn.Parameter(" in source
                and "def _tactile_latent_target(" in source
                and 'targets["tactile_real"] = torch.cat([tactile_latent, tactile_base, aux_full], dim=0)' in current_targets_source
            ),
            detail="Live PICF tactile innovation target now combines dense tactile latent probes with coarse tactile map and tactile auxiliaries.",
        ),
        CheckResult(
            name="live_point_innovation_uses_native_latent_probes_plus_occupancy",
            ok=(
                "self.point_latent_queries = nn.Parameter(" in source
                and "def _point_latent_target(" in source
                and 'targets["point_real"] = torch.cat([point_latent, occ.reshape(-1)], dim=0)' in current_targets_source
            ),
            detail="Live PICF point innovation target now combines native point latent probes with coarse occupancy.",
        ),
        CheckResult(
            name="conditioned_control_upprojects_physical_and_task_tokens",
            ok=all(
                text in conditioned_control_source
                for text in (
                    "control_posterior_tokens = self.posterior_to_control_proj(posterior.tokens)",
                    "control_global_post = self.global_post_to_control_proj(posterior.global_post[None, :])",
                    "control_innovation_token = self.innovation_to_control_proj(innovation_token[None, :])",
                    "control_proprio_token = self.proprio_to_control_proj(proprio_token[None, :])",
                    "task_local_tokens = self.task_to_control_proj(task_readout.local_tokens)",
                    "task_global_token = self.task_global_to_control_proj(task_readout.global_token[None, :])",
                    "self.instruction_to_control_proj(task_readout.instruction_tokens)",
                )
            ),
            detail="Conditioned control trunk consumes up-projected physical base tokens and task-readout tokens.",
        ),
        CheckResult(
            name="predictive_conditioned_upprojects_physical_pred_tokens",
            ok=(
                "conditioned_physical_pred_tokens = self.physical_pred_to_conditioned_proj(physical_pred_tokens)" in conditioned_predictive_source
                and "_add_role_embedding(conditioned_physical_pred_tokens, self.predictive_conditioned_role_embedding, 0)" in conditioned_predictive_source
                and "global_pred = self.predictive_state_proj(predictive_query_state)" in conditioned_predictive_source
            ),
            detail="Conditioned future trunk consumes up-projected physical prediction tokens and projects the semantic-width query state back to the physical cache width.",
        ),
        CheckResult(
            name="default_core_widths_are_mixed_512_phys_2048_semantic",
            ok=(
                int(defaults.hidden_dim) == 512
                and int(defaults.posterior_hidden_dim) == 512
                and int(defaults.innovation_dim) == 512
                and int(defaults.control_dim) == 512
                and int(defaults.future_hidden_dim) == 512
                and int(defaults.semantic_dim) == 2048
                and int(defaults.semantic_cross_dim) == 2048
            ),
            detail=(
                "Default core widths are "
                f"hidden={defaults.hidden_dim} posterior_hidden={defaults.posterior_hidden_dim} "
                f"innovation={defaults.innovation_dim} control={defaults.control_dim} "
                f"semantic={defaults.semantic_dim} semantic_cross={defaults.semantic_cross_dim} "
                f"future_hidden={defaults.future_hidden_dim}."
            ),
        ),
        CheckResult(
            name="semantic_prefix_projection_is_identity_in_mainline",
            ok="self.semantic_prefix_proj = nn.Identity()" in source,
            detail="The mainline semantic prefix projection is identity because semantic tokens remain native-width.",
        ),
        CheckResult(
            name="control_prefix_includes_explicit_global_post",
            ok="control_global_post = self.global_post_to_control_proj(posterior.global_post[None, :])" in conditioned_control_source,
            detail="Conditioned control tokens explicitly include posterior.global_post via up-projection.",
        ),
        CheckResult(
            name="policy_object_unifies_train_and_serve_interfaces",
            ok=(
                "class PicfPi05Policy:" in policy_source
                and "class PicfPolicyTrainResult:" in policy_source
                and "class PicfPolicyActResult:" in policy_source
                and "def forward_train_transition(" in policy_source
                and "def act(" in policy_source
            ),
            detail="PicfPi05Policy exists as the unified exported policy interface with typed result objects.",
        ),
        CheckResult(
            name="trainer_primary_action_loss_uses_policy_and_pi05_flow",
            ok=(
                "self.policy = PicfPi05Policy(" in trainer_source
                and "policy_forward = self.policy.forward_train_transition(" in trainer_source
                and "output = policy_forward.output" in trainer_source
                and "flow_override = policy_forward.flow_override" in trainer_source
            ),
            detail="PICF trainer now uses the unified policy interface for action/control integration.",
        ),
        CheckResult(
            name="trainer_uses_compact_recurrent_next_state_instead_of_full_output_state",
            ok=(
                "previous = policy_forward.next_state" in trainer_source
                and "def make_recurrent_carry(self, state: PicfCoreState)" in source
                and "class PicfRecurrentCarryState:" in contracts_source
                and "class PicfRecurrentPredictiveState:" in contracts_source
            ),
            detail=(
                "Window training now carries only the canonical recurrent state between transitions "
                "instead of forwarding the full PicfCoreState object."
            ),
        ),
        CheckResult(
            name="trainer_ddp_runtime_guard_rejects_detail_debug_by_default",
            ok=(
                "def _configure_distributed_runtime_env(" in trainer_source
                and "DDP runtime guard: TORCH_DISTRIBUTED_DEBUG=DETAIL is not allowed by default." in trainer_source
                and 'applied = "INFO"' in trainer_source
                and "OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL" in trainer_source
                and "LOCAL_RANK must be set when running under DDP" in trainer_source
            ),
            detail=(
                "PICF trainer hardens multi-rank startup by defaulting TORCH_DISTRIBUTED_DEBUG to INFO, "
                "rejecting DETAIL unless it is explicitly opted into, and failing fast when LOCAL_RANK is missing."
            ),
        ),
        CheckResult(
            name="serve_primary_action_path_uses_policy_act",
            ok=(
                "self._policy = getattr(" in serve_source
                and "PicfPi05Policy(" in serve_source
                and "act_result = self._policy.act(" in serve_source
                and "output = act_result.output" in serve_source
                and "refresh_predictive_state_for_action(" not in serve_source
            ),
            detail=(
                "PICF serving path now delegates action generation and predictive finalization to PicfPi05Policy.act."
            ),
        ),
        CheckResult(
            name="policy_act_requires_pi05_action_generator_on_new_path",
            ok="self._require_action_generation()" in policy_source and "observed = self.core.observe_step(" in policy_source,
            detail="The v2.2 policy requires a PI0.5 action generator before running the new observe/sample/finalize path.",
        ),
        CheckResult(
            name="policy_training_resolves_teacher_forced_action_future_explicitly",
            ok=(
                "def _teacher_forced_action_future(" in policy_source
                and "if action_chunk_target is not None:" in policy_source
                and "if observation.action is not None:" in policy_source
                and "action_future=self._teacher_forced_action_future(" in policy_source
            ),
            detail="Policy training path resolves teacher-forced executed action explicitly from action_chunk_target first, then observation.action.",
        ),
        CheckResult(
            name="picf_ablation_mode_is_explicit_across_policy_trainer_and_docs",
            ok=(
                '"--picf-mode"' in trainer_source
                and '"--picf-mode"' in serve_source
                and "_normalize_train_args(args)" in serve_source
                and "_validate_train_args(args)" in serve_source
                and "picf_enabled=False" in policy_test_source
                and "_forward_action_only_window(" in trainer_source
                and "extra_prefix_tokens=None" in policy_source
                and "picf_mode=ablated" in readme_v22_source
                and "--picf-mode ablated" in calvin_readme_source
            ),
            detail="The repo exposes a first-class PI0.5-only ablation mode instead of relying on implicit PICF bypasses.",
        ),
        CheckResult(
            name="core_no_longer_uses_direct_7d_action_head",
            ok=(
                "self.action_head = nn.Linear(self.config.control_dim, 7)" not in source
                and "action = self._clip_action(self.action_head(pooled_state))" not in predictive_source
                and "action, action_chunk = self._default_predictive_action(action_future)" in predictive_source
            ),
            detail="Core no longer uses a direct trainable 7D head; trainer/serve rely on the restored PI0.5 action path and core uses only a non-trainable default action placeholder when needed.",
        ),
        CheckResult(
            name="legacy_boolean_advanced_indexing_removed_from_pipeline",
            ok=all(
                text not in source
                for text in (
                    "semantic_tokens[keep]",
                    "depth_factor[valid_depth_rows]",
                    "S[valid] = S_obs[valid]",
                    "a[valid] = _extent_from_cov(S[valid], self.config)",
                )
            ),
            detail="Pipeline no longer uses the audited boolean advanced-indexing patterns on live training tensors.",
        ),
        CheckResult(
            name="legacy_boolean_advanced_indexing_removed_from_paligemma_wrapper",
            ok=all(
                text not in wrapper_source
                for text in (
                    "hidden_states[0][valid]",
                    "prefix_output[0][valid]",
                )
            ),
            detail="PaliGemma wrapper no longer slices trainable token streams with boolean advanced indexing.",
        ),
    ]
    return checks


def verify_doc_links() -> list[CheckResult]:
    checks = []
    for path in (README_PATH, V21_README_PATH, V22_README_PATH, CALVIN_README_PATH):
        text = _read(path)
        checks.append(
            CheckResult(
                name=f"doc_links_formal_contract:{path.name}",
                ok="PICF_FORMAL_CONTRACT.md" in text,
                detail=f"{path.name} references PICF_FORMAL_CONTRACT.md",
            )
        )
    readme_text = _read(README_PATH)
    v21_text = _read(V21_README_PATH)
    v22_text = _read(V22_README_PATH)
    calvin_text = _read(CALVIN_README_PATH)
    formal_text = _read(FORMAL_CONTRACT_PATH)
    checks.append(
        CheckResult(
            name="readme_entry_points_to_v22_as_current_and_v21_as_historical",
            ok=all(
                needle in readme_text
                for needle in (
                    "README_v2.2.md",
                    "current local v2.2 architecture record",
                    "README_v2.1.md",
                    "Historical v2.1 deployment record",
                )
            ),
            detail="README entry points to v2.2 as current-live and v2.1 as historical.",
        )
    )
    checks.append(
        CheckResult(
            name="readme_v21_is_marked_historical",
            ok=all(
                needle in v21_text
                for needle in (
                    "archived v2.1 deployment record",
                    "current live local architecture record",
                    "README_v2.2.md",
                )
            ),
            detail="README_v2.1 is now explicitly marked as historical context.",
        )
    )
    checks.append(
        CheckResult(
            name="readme_v22_records_contract_rewrite_scope",
            ok=all(
                needle in v22_text
                for needle in (
                    "current local v2.2 architecture record",
                    "contract rewrite",
                    "one exported policy object",
                    "one canonical conditioned control state `C_t`",
                    "one final action path",
                )
            ),
            detail="README_v2.2 records the one-shot v2.2 contract rewrite scope.",
        )
    )
    checks.append(
        CheckResult(
            name="readme_v22_records_corrected_public_read_memory_and_future_contract",
            ok=all(
                needle in v22_text
                for needle in (
                    "public_read_memory = [fused_tokens, visual_tokens]",
                    "_build_task_readout(...)",
                    "must not take `posterior` as a direct input",
                    "K_t^{cond} = P_cond(H_t^{phys_pred}, C_t^{future})",
                )
            ),
            detail="README_v2.2 captures the corrected public-read/task-readout/conditioned-future contract.",
        )
    )
    checks.append(
        CheckResult(
            name="readme_v22_marks_task_query_rounds_as_reserved_not_live_knob",
            ok=all(
                needle in v22_text
                for needle in (
                    "Reserved / compatibility-only field:",
                    "`task_query_rounds: int = 2`",
                    "not currently consumed by the live v2.2 core/trainer path",
                )
            ),
            detail="README_v2.2 does not overclaim task_query_rounds as a live implemented knob.",
        )
    )
    checks.append(
        CheckResult(
            name="readme_v22_and_calvin_doc_record_tokenwise_exact_chunking_contract",
            ok=all(
                needle in v22_text
                for needle in (
                    "tokenwise_ff_chunk_size=64",
                    "semantic_tokenwise_chunk_size=64",
                    "semantic_projection_chunk_size=128",
                    "semantic_mlp_chunk_size=64",
                )
            )
            and all(
                needle in calvin_text
                for needle in (
                    "tokenwise_ff_chunk_size=64",
                    "semantic_tokenwise_chunk_size=64",
                    "semantic_projection_chunk_size=128",
                    "semantic_mlp_chunk_size=64",
                )
            ),
            detail="README_v2.2 and CALVIN validation doc both record the exact tokenwise chunking contract, including split semantic projection/MLP controls.",
        )
    )
    checks.append(
        CheckResult(
            name="readme_v22_and_calvin_doc_record_semantic_nested_leaf_fsdp_contract",
            ok=all(
                needle in v22_text
                for needle in (
                    "directly called PI0/PaliGemma runtime hot leaves",
                    "remaining semantic root still uses `ignored_states`",
                    "SigLIP vision tower and multimodal projector currently remain under the outer semantic root",
                )
            )
            and all(
                needle in calvin_text
                for needle in (
                    "semantic FSDP path now pre-wraps the directly called PI0/PaliGemma runtime hot leaves",
                    "mixed-dtype semantic root wrapper only to the remaining parameters",
                    "SigLIP vision tower and multimodal projector currently stay under the outer semantic root",
                )
            ),
            detail="README_v2.2 and CALVIN validation doc both record the nested semantic-leaf FSDP contract.",
        )
    )
    checks.append(
        CheckResult(
            name="calvin_validation_points_to_v22_current_live_docs",
            ok=all(
                needle in calvin_text
                for needle in (
                    "README_v2.2.md",
                    "current local architecture and deployment document",
                    "README_v2.1.md` is retained only as the archived pre-v2.2 deployment record",
                )
            ),
            detail="CALVIN validation README now points to v2.2 as the current-live deployment document.",
        )
    )
    checks.append(
        CheckResult(
            name="formal_contract_exists",
            ok=FORMAL_CONTRACT_PATH.is_file(),
            detail=f"{FORMAL_CONTRACT_PATH} exists",
        )
    )
    checks.append(
        CheckResult(
            name="formal_contract_records_current_v22_control_and_future_contract",
            ok=all(
                needle in formal_text
                for needle in (
                    "current local",
                    "v2.2 PICF codebase",
                    "public read memory",
                    "one canonical conditioned control state",
                    "K_t^{cond} = P_cond(H_t^{phys_pred}, C_t^{future})",
                )
            ),
            detail="Formal contract records the current v2.2 public-read / conditioned-control / conditioned-future semantics.",
        )
    )
    return checks


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> CheckResult:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    detail = (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip()
    return CheckResult(
        name=" ".join(cmd),
        ok=proc.returncode == 0,
        detail=detail,
    )


def verify_regressions() -> list[CheckResult]:
    tests = [
        "src/openpi/picf/core/pipeline_test.py::test_language_is_late_and_does_not_change_current_posterior",
        "src/openpi/picf/core/pipeline_test.py::test_semantic_changes_do_not_pollute_physical_prediction_cache_or_next_innovation",
        "src/openpi/picf/core/pipeline_test.py::test_semantic_prefix_prompt_changes_do_not_change_physical_branch_but_do_change_control_and_future",
        "src/openpi/picf/core/pipeline_test.py::test_control_prefix_explicitly_depends_on_global_post",
        "src/openpi/picf/core/pipeline_test.py::test_control_and_future_trunks_consume_task_readout_and_not_raw_semantic_prefix",
        "src/openpi/picf/core/pipeline_test.py::test_semantic_tokens_directly_condition_control_and_semantic_future_readout",
        "src/openpi/picf/core/pipeline_test.py::test_semantic_tokens_alone_can_condition_action_without_cross_reads",
        "src/openpi/picf/core/pipeline_test.py::test_prior_and_context_use_previous_executed_action_not_previous_policy_output",
        "src/openpi/picf/core/pipeline_test.py::test_previous_semantic_conditioned_predictive_state_does_not_feed_next_prior_or_innovation",
        "src/openpi/picf/core/pipeline_test.py::test_previous_physical_prediction_cache_is_the_only_predictive_cache_allowed_to_change_next_innovation",
    ]
    return [_run([sys.executable, "-m", "pytest", "-q", *tests])]


def verify_full_core_suite() -> list[CheckResult]:
    return [
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "src/openpi/picf/core/pipeline_test.py",
                "src/openpi/picf/core/training_test.py",
                "src/openpi/picf/policy_test.py",
                "scripts/picf_core_train_test.py",
                "scripts/picf_resume_train_test.py",
                "scripts/serve_picf_policy_test.py",
                "src/openpi/picf/paligemma/wrapper_test.py",
            ]
        )
    ]


def verify_smoke() -> list[CheckResult]:
    with tempfile.TemporaryDirectory(prefix="picf-contract-smoke-") as tmp_dir:
        calvin_root = build_mini_calvin_dataset(Path(tmp_dir), make_zip=False)
        result = run_smoke(
            calvin_root=str(calvin_root),
            split="training",
            backend="dir",
            segment_index=0,
            stride=1,
            max_points=256,
            device="cpu",
            lr=1e-3,
            use_tactile=False,
            tactile_sensor_names=("digit", "gelsight_mini"),
            tactile_sensor_offsets_m=((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)),
            visual_mode="stub",
            tactile_mode="stub",
            point_backbone="rgb",
            visual_checkpoint_path=None,
            visual_checkpoint_key=None,
            visual_model_name="vjepa2_1_vit_base_384",
            visual_dtype="float32",
            visual_img_size=384,
            visual_num_frames=4,
            visual_patch_size=16,
            visual_tubelet_size=2,
            visual_use_last_two_mean=False,
            tactile_checkpoint_path=None,
            tactile_dtype="float32",
            tactile_num_frames=4,
            tactile_stride=1,
            sonata_checkpoint_path=None,
            sonata_stage_name="base",
            sonata_dtype="float32",
        )
    ok = all(
        (
            bool(result["policy_path_used"]),
            bool(result["flow_override_used"]),
            bool(result["conditioned_control_present"]),
            bool(result["pi_prefix_tokens_present"]),
            bool(result["predictive_action_chunk_present"]),
            float(result["loss_total"]) > 0.0,
            float(result["point_grad_norm"]) > 0.0,
        )
    )
    return [
        CheckResult(
            name="picf_core_train_smoke",
            ok=ok,
            detail=str(result),
        )
    ]


def _print_results(title: str, results: list[CheckResult]) -> bool:
    print(f"\n== {title} ==")
    all_ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}")
        print(result.detail)
        print()
        all_ok = all_ok and result.ok
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PICF formal contract with static and dynamic checks.")
    parser.add_argument("--skip-full-suite", action="store_true", help="Skip the larger regression suite.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the local CPU smoke training check.")
    args = parser.parse_args()

    groups: list[tuple[str, list[CheckResult]]] = [
        ("Static Contract Checks", verify_static_contract()),
        ("Documentation Checks", verify_doc_links()),
        ("Targeted Invariance Regressions", verify_regressions()),
    ]
    if not args.skip_full_suite:
        groups.append(("Core Regression Suite", verify_full_core_suite()))
    if not args.skip_smoke:
        groups.append(("Smoke Training Check", verify_smoke()))

    ok = True
    for title, results in groups:
        ok = _print_results(title, results) and ok

    print("== Summary ==")
    if ok:
        print("PASS: PICF formal contract checks passed.")
        return 0
    print("FAIL: At least one PICF formal contract check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

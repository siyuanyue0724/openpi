#!/usr/bin/env python3
"""Verify the unified LingBot VLA2 patches against an immutable Git object."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

LINGBOT_COMMIT = "69729b4ef24c63ec25e750915491635f4753be1d"
CHECKOUT_RELATIVE_PATH = Path("references/source_checkouts/lingbot-vla-v2")
DATA_PATCH_RELATIVE_PATH = Path("references/patches/lingbot_vla2_lerobot_data_compat.patch")
GRAPH_PATCH_RELATIVE_PATH = Path("references/patches/lingbot_vla2_unified_belief_graph.patch")
MODEL_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py")
PARALLEL_SOURCE = Path("lingbotvla/distributed/torch_parallelize.py")
CHECKPOINTER_SOURCE = Path("lingbotvla/checkpoint/checkpointer.py")
ATTENTION_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/utils.py")
TEXT_LAYER_SOURCE = Path("lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py")
DATA_SOURCES = (
    Path("lingbotvla/data/vla_data/base_dataset.py"),
    Path("lingbotvla/data/vla_data/utils.py"),
)
PATCHED_SOURCES = (*DATA_SOURCES, PARALLEL_SOURCE, CHECKPOINTER_SOURCE, MODEL_SOURCE)

_REQUIRED_DATA_FRAGMENTS = (
    "from lerobot.datasets.utils import hf_transform_to_torch, load_nested_dataset",
    "hf_dataset = load_nested_dataset(",
    'if LEROBOT_DATASET_API == "v3":',
    "return super().__getitem__(idx)",
    "def _as_feature_mapping(value)",
    "joint_info = _as_feature_mapping(s)",
    "for k, v in _as_feature_mapping(d).items()",
)
_REQUIRED_GRAPH_FRAGMENTS = (
    "self.unified_belief_graph = None",
    "def set_unified_belief_graph",
    "unified_belief_context=None",
    "self.unified_belief_graph.prepare_joint_inputs(",
    "self.unified_belief_graph.observe_joint_qkv(",
    "self.unified_belief_graph.apply_relation_message(",
    "self.unified_belief_graph.after_layer(",
    "align_outputs_embeds = (",
    "outputs_embeds[:, :prefix_len]",
    "compute_alignment_losses=True",
    "self.config.align_params != {} and compute_alignment_losses",
    "compute_alignment_losses=compute_alignment_losses",
    "cache_inputs = [prefix_embs, state_emb]",
    "prefix_pad_masks = unified_belief_context.expanded_cache_valid",
    "prefix_position_ids = unified_belief_context.expanded_cache_position_ids",
    "action_cache_visible = unified_belief_context.expanded_action_cache_visible",
    "prefix_pad_masks = prefix_pad_masks & action_cache_visible",
    "state_is_cached=unified_belief_context is not None",
    "native_prefix_len=native_prefix_len",
    "suffix_embs = suffix_embs[:, 1:]",
    "and unified_belief_context is None",
    "class LingbotVlaV2Policy",
)
_REQUIRED_PARALLEL_FRAGMENTS = (
    "CPUOffloadPolicy(pin_memory=False)",
    'user_fsdp_kwargs = dict(kwargs.pop("fsdp_kwargs", {}))',
    '"enable_fsdp_offload and fsdp_kwargs.offload_policy are mutually exclusive"',
    'user_fsdp_kwargs["offload_policy"] = CPUOffloadPolicy',
)
_REQUIRED_CHECKPOINTER_FRAGMENTS = (
    "def _atomic_torch_save",
    "os.fsync(stream.fileno())",
    "planner=DefaultLoadPlanner(allow_partial_load=allow_partial_load)",
    "if not allow_partial_load:",
    "Failed to restore optimizer",
    "sync_files=True",
    "weights_only=True",
)
_REQUIRED_UPSTREAM_FRAGMENTS = (
    "class QwenvlWithExpertV2Model",
    "query_states = torch.cat(query_states, dim=1)",
    "query_states, key_states = self.apply_mrope(",
    "att_output = self.attention_interface(",
    "class FlowMatchingV2",
    "def sample_actions(",
    "class LingbotVlaV2Policy",
)
_REQUIRED_ATTENTION_CONTRACT_FRAGMENTS = (
    "Output tensor of shape [batch_size, seq_len, num_attention_heads * head_dim].",
    "att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)",
)
_REQUIRED_TEXT_LAYER_CONTRACT_FRAGMENTS = (
    "hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)",
    "out_emb = self.self_attn.o_proj(att_output[:, start:end])",
)
_FORBIDDEN_GRAPH_FRAGMENTS = (
    "action_layer_adapter",
    "set_action_layer_adapter",
    "action_layer_context",
)


def _patch_paths(patch_text: str) -> set[Path]:
    paths: set[Path] = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            paths.add(Path(line.removeprefix("+++ b/")))
    return paths


def _check_fragments(
    *,
    patch_name: str,
    patch_text: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> None:
    missing = [fragment for fragment in required if fragment not in patch_text]
    if missing:
        raise ValueError(f"{patch_name} omits required fragments: {missing}")
    present = [fragment for fragment in forbidden if fragment in patch_text]
    if present:
        raise ValueError(f"{patch_name} contains forbidden legacy fragments: {present}")


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _export_commit(checkout: Path, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", LINGBOT_COMMIT],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_patched_source_hashes(
    *,
    checkout: Path,
    patches: tuple[Path, ...],
) -> dict[str, str]:
    """Reproduce patched source from the pinned Git object and hash its bytes."""

    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout).stdout.strip()
    if actual != LINGBOT_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-source-hash-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        for patch in patches:
            _run(["git", "apply", "--check", str(patch.resolve())], cwd=exported)
            _run(["git", "apply", str(patch.resolve())], cwd=exported)
        return {
            str(relative_path): _sha256(exported / relative_path)
            for relative_path in PATCHED_SOURCES
        }


def verify_unified_patches(
    *,
    root: Path,
    checkout: Path | None = None,
    check_apply: bool = True,
) -> dict[str, object]:
    """Fail closed unless both patches reproduce the declared unified source."""

    data_patch = root / DATA_PATCH_RELATIVE_PATH
    graph_patch = root / GRAPH_PATCH_RELATIVE_PATH
    for path in (data_patch, graph_patch):
        if not path.is_file():
            raise ValueError(f"required LingBot patch is absent: {path}")
    data_text = data_patch.read_text()
    graph_text = graph_patch.read_text()
    _check_fragments(
        patch_name="LingBot data patch",
        patch_text=data_text,
        required=_REQUIRED_DATA_FRAGMENTS,
    )
    _check_fragments(
        patch_name="LingBot unified graph patch",
        patch_text=graph_text,
        required=(*_REQUIRED_GRAPH_FRAGMENTS, *_REQUIRED_PARALLEL_FRAGMENTS),
        forbidden=_FORBIDDEN_GRAPH_FRAGMENTS,
    )
    if _patch_paths(data_text) != set(DATA_SOURCES):
        raise ValueError("LingBot data patch modifies undeclared source paths")
    if _patch_paths(graph_text) != {MODEL_SOURCE, PARALLEL_SOURCE, CHECKPOINTER_SOURCE}:
        raise ValueError("LingBot unified graph patch modifies undeclared source paths")

    result: dict[str, object] = {
        "commit": LINGBOT_COMMIT,
        "data_patch": str(DATA_PATCH_RELATIVE_PATH),
        "graph_patch": str(GRAPH_PATCH_RELATIVE_PATH),
        "apply_checked": False,
        "patched_sources": [str(path) for path in PATCHED_SOURCES],
    }
    if not check_apply:
        return result

    checkout = root / CHECKOUT_RELATIVE_PATH if checkout is None else checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"pinned LingBot checkout is absent: {checkout}")
    actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout).stdout.strip()
    if actual != LINGBOT_COMMIT:
        raise ValueError(f"LingBot checkout {actual} differs from {LINGBOT_COMMIT}")

    patched_source_hashes: dict[str, str]
    with tempfile.TemporaryDirectory(prefix="picf-lingbot-unified-") as temporary:
        exported = Path(temporary)
        _export_commit(checkout, exported)
        source_text = (exported / MODEL_SOURCE).read_text()
        missing_upstream = [
            fragment for fragment in _REQUIRED_UPSTREAM_FRAGMENTS if fragment not in source_text
        ]
        if missing_upstream:
            raise ValueError(f"pinned LingBot source omits required symbols: {missing_upstream}")
        attention_text = (exported / ATTENTION_SOURCE).read_text()
        missing_attention = [
            fragment
            for fragment in _REQUIRED_ATTENTION_CONTRACT_FRAGMENTS
            if fragment not in attention_text
        ]
        if missing_attention:
            raise ValueError(
                f"pinned LingBot attention output contract changed: {missing_attention}"
            )
        text_layer = (exported / TEXT_LAYER_SOURCE).read_text()
        missing_text_layer = [
            fragment
            for fragment in _REQUIRED_TEXT_LAYER_CONTRACT_FRAGMENTS
            if fragment not in text_layer
        ]
        if missing_text_layer:
            raise ValueError(f"pinned LingBot text-layer contract changed: {missing_text_layer}")
        for patch in (data_patch, graph_patch):
            _run(["git", "apply", "--check", str(patch)], cwd=exported)
            _run(["git", "apply", str(patch)], cwd=exported)
        for relative_path in PATCHED_SOURCES:
            source_path = exported / relative_path
            compile(source_path.read_text(), str(source_path), "exec")
        patched_model = (exported / MODEL_SOURCE).read_text()
        _check_fragments(
            patch_name="patched LingBot model source",
            patch_text=patched_model,
            required=_REQUIRED_GRAPH_FRAGMENTS,
            forbidden=_FORBIDDEN_GRAPH_FRAGMENTS,
        )
        patched_parallel = (exported / PARALLEL_SOURCE).read_text()
        _check_fragments(
            patch_name="patched LingBot FSDP source",
            patch_text=patched_parallel,
            required=_REQUIRED_PARALLEL_FRAGMENTS,
        )
        patched_checkpointer = (exported / CHECKPOINTER_SOURCE).read_text()
        _check_fragments(
            patch_name="patched LingBot checkpointer source",
            patch_text=patched_checkpointer,
            required=_REQUIRED_CHECKPOINTER_FRAGMENTS,
        )
        patched_source_hashes = {
            str(relative_path): _sha256(exported / relative_path)
            for relative_path in PATCHED_SOURCES
        }

    result["apply_checked"] = True
    result["attention_output_contract"] = "[batch,tokens,query_heads*head_dim]"
    result["patched_source_sha256"] = patched_source_hashes
    return result


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--no-check-apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            verify_unified_patches(
                root=args.root,
                checkout=args.checkout,
                check_apply=not args.no_check_apply,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

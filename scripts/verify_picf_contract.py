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
HANDOFF_README_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "README_semantic_prefix_primary_mixedwidth_refactor.md"
PLAN_PATH = REPO_ROOT / "plan_readme_ray_geometry.md"
CALVIN_README_PATH = REPO_ROOT / "docs" / "CALVIN_VALIDATION_README.md"
FORMAL_CONTRACT_PATH = REPO_ROOT / "PICF_FORMAL_CONTRACT.md"
PALIGEMMA_WRAPPER_PATH = REPO_ROOT / "src" / "openpi" / "picf" / "paligemma" / "wrapper.py"


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

    predictive_node = _function_node(tree, "_predictive_state")
    predictive_source = _node_source(source, predictive_node)

    step_node = _function_node(tree, "step")
    step_source = _node_source(source, step_node)

    checks = [
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
            name="previous_action_prefers_executed_action",
            ok='getattr(previous.predictive, "executed_action", None)' in prev_action_source and "previous.predictive.action" in prev_action_source,
            detail="`_previous_action` uses executed_action first, action as fallback.",
        ),
        _call_order(
            source,
            predictive_source,
            [
                "physical_prediction_cache = self._prediction_cache_from_global(physical_global_pred)",
                "pred_tokens = self.predictive_semantic_world(",
                "prediction_cache = self._prediction_cache_from_global(global_pred)",
            ],
        ),
        _call_order(
            source,
            step_source,
            [
                "posterior = self._posterior_update(",
                "innovation_token, innovation_norm = self._innovation(",
                "predictive = self._predictive_state(",
            ],
        ),
        CheckResult(
            name="task_sidecar_removed_from_pipeline",
            ok=all(text not in source for text in ("_build_task_anchors(", "task_anchors=", "task_query_tokens")),
            detail="Task-anchor sidecar path has been removed from the core pipeline.",
        ),
        CheckResult(
            name="control_prefix_uses_full_semantic_prefix_primary",
            ok="_add_role_embedding(control_semantic_prefix_tokens, self.control_role_embedding, 3)" in predictive_source,
            detail="`_predictive_state` injects the full semantic prefix directly into the control trunk.",
        ),
        CheckResult(
            name="control_prefix_upprojects_physical_world_tokens",
            ok=all(
                text in predictive_source
                for text in (
                    "control_posterior_tokens = self.posterior_to_control_proj(posterior.tokens)",
                    "control_global_post = self.global_post_to_control_proj(posterior.global_post[None, :])",
                    "control_innovation_token = self.innovation_to_control_proj(innovation_token[None, :])",
                    "control_proprio_token = self.proprio_to_control_proj(proprio_token[None, :])",
                )
            ),
            detail="Physical posterior/global_post/innovation/proprio are up-projected into the semantic-width control trunk.",
        ),
        CheckResult(
            name="predictive_conditioned_uses_full_semantic_prefix_primary",
            ok="_add_role_embedding(conditioned_semantic_prefix_tokens, self.predictive_conditioned_role_embedding, 1)" in predictive_source,
            detail="`_predictive_state` injects the full semantic prefix directly into the conditioned future trunk.",
        ),
        CheckResult(
            name="predictive_conditioned_upprojects_physical_pred_tokens",
            ok=(
                "conditioned_physical_pred_tokens = self.physical_pred_to_conditioned_proj(physical_pred_tokens)" in predictive_source
                and "_add_role_embedding(conditioned_physical_pred_tokens, self.predictive_conditioned_role_embedding, 0)" in predictive_source
                and "global_pred = self.predictive_state_proj(predictive_query_state)" in predictive_source
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
            ok="control_global_post = self.global_post_to_control_proj(posterior.global_post[None, :])" in predictive_source,
            detail="`_predictive_state` explicitly injects posterior.global_post into control tokens via up-projection.",
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
    for path in (README_PATH, HANDOFF_README_PATH, PLAN_PATH, CALVIN_README_PATH):
        text = _read(path)
        checks.append(
            CheckResult(
                name=f"doc_links_formal_contract:{path.name}",
                ok="PICF_FORMAL_CONTRACT.md" in text,
                detail=f"{path.name} references PICF_FORMAL_CONTRACT.md",
            )
        )
    checks.append(
        CheckResult(
            name="formal_contract_exists",
            ok=FORMAL_CONTRACT_PATH.is_file(),
            detail=f"{FORMAL_CONTRACT_PATH} exists",
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
        "src/openpi/picf/core/pipeline_test.py::test_full_semantic_prefix_tokens_are_directly_included_in_control_and_future_trunks",
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
                "scripts/picf_core_train_test.py",
                "scripts/picf_resume_train_test.py",
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
    ok = float(result["loss_total"]) > 0.0 and float(result["action_grad_norm"]) > 0.0
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

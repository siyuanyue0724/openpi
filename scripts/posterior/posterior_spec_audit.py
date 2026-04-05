from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from pathlib import Path

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PosteriorState


def _module_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _load_trees(posterior_root: Path) -> dict[str, ast.AST]:
    trees: dict[str, ast.AST] = {}
    for path in sorted(posterior_root.glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        trees[path.name] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return trees


def run_spec_audit(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)
    posterior_root = repo_root / "src" / "openpi" / "picf" / "posterior"
    trees = _load_trees(posterior_root)
    imports_by_file = {name: _module_imports(tree) for name, tree in trees.items()}
    all_imports = set().union(*imports_by_file.values()) if imports_by_file else set()

    forbidden_stage_modules = (
        "openpi.picf.object",
        "openpi.picf.semantic",
        "openpi.picf.context",
        "openpi.picf.vjepa",
        "openpi.picf.anytouch",
        "openpi.picf.visual",
        "openpi.picf.tactile",
    )
    forbidden_hits = {
        name: sorted(
            module for module in modules if any(token in module.lower() for token in forbidden_stage_modules)
        )
        for name, modules in imports_by_file.items()
    }
    forbidden_hits = {name: hits for name, hits in forbidden_hits.items() if hits}

    state_fields = [field.name for field in dataclasses.fields(PosteriorState)]
    config = PosteriorConfig()

    checks = {
        "posterior_package_exists": posterior_root.is_dir(),
        "point_only_import_surface": not forbidden_hits,
        "pipeline_uses_point_fusion": "openpi.picf.posterior.fusion" in imports_by_file.get("pipeline.py", set()),
        "pipeline_no_visual_modules": not any("visual" in module.lower() for module in all_imports),
        "pipeline_no_tactile_modules": not any("tactile" in module.lower() for module in all_imports),
        "state_has_no_object_field": "object" not in state_fields,
        "state_has_no_semantic_field": "semantic" not in state_fields,
        "state_has_no_context_field": "context" not in state_fields,
        "block_variance_contract": config.dim_total == config.dim_h + config.dim_g + config.dim_c,
        "block_variance_shape_is_triplet": len(
            dataclasses.fields(__import__("openpi.picf.posterior.contracts", fromlist=["PointExpertState"]).PointExpertState)
        )
        >= 2,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "forbidden_hits": forbidden_hits,
        "state_fields": state_fields,
        "imports_by_file": {name: sorted(modules) for name, modules in imports_by_file.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Static spec-boundary audit for PICF posterior.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_spec_audit(args.repo_root)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

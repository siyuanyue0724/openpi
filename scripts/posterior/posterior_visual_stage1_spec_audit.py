from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from pathlib import Path

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


def run_visual_stage1_spec_audit(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)
    posterior_root = repo_root / "src" / "openpi" / "picf" / "posterior"
    filenames = {
        "config.py",
        "contracts.py",
        "fusion_visual.py",
        "pipeline_visual.py",
        "point_expert.py",
        "prior.py",
        "visual_expert.py",
    }
    trees = {
        path.name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(posterior_root.glob("*.py"))
        if path.name in filenames and not path.name.endswith("_test.py")
    }
    imports_by_file = {name: _module_imports(tree) for name, tree in trees.items()}
    all_imports = set().union(*imports_by_file.values()) if imports_by_file else set()
    forbidden_stage_modules = (
        "openpi.picf.object",
        "openpi.picf.semantic",
        "openpi.picf.context",
        "openpi.picf.anytouch",
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
    checks = {
        "posterior_visual_stage_exists": posterior_root.is_dir(),
        "visual_stage_forbidden_surface_clean": not forbidden_hits,
        "pipeline_imports_visual_expert": "openpi.picf.posterior.visual_expert"
        in imports_by_file.get("pipeline_visual.py", set()),
        "pipeline_imports_vjepa": any("openpi.picf.vjepa" in module for module in all_imports),
        "state_has_visual_field": "visual" in state_fields,
        "state_has_no_object_field": "object" not in state_fields,
        "state_has_no_semantic_field": "semantic" not in state_fields,
        "state_has_no_context_field": "context" not in state_fields,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "forbidden_hits": forbidden_hits,
        "state_fields": state_fields,
        "imports_by_file": {name: sorted(modules) for name, modules in imports_by_file.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Static spec audit for PICF posterior visual stage-1.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_visual_stage1_spec_audit(args.repo_root)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

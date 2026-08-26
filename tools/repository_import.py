"""Fail-closed source binding for production tools run from Git worktrees."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _module_is_inside(module: object, root: Path) -> bool:
    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        origins.append(Path(module_file).resolve())
    for value in getattr(module, "__path__", ()):
        origins.append(Path(value).resolve())
    return any(origin == root or root in origin.parents for origin in origins)


def bind_entrypoint_to_own_repository(
    entrypoint_file: str,
    *,
    entrypoint_name: str,
) -> Path:
    """Prepend and verify the repository that physically contains a CLI."""

    if not isinstance(entrypoint_name, str) or not entrypoint_name:
        raise ValueError("repository-bound entrypoint name must be non-empty")
    repository_root = Path(entrypoint_file).resolve().parents[1]
    expected_package = (repository_root / "src/picf_next").resolve()
    for import_path in (repository_root, repository_root / "src"):
        import_text = str(import_path)
        while import_text in sys.path:
            sys.path.remove(import_text)
        sys.path.insert(0, import_text)

    # A third-party distribution named ``tools`` can be cached by the failed
    # package-form import that precedes direct-script fallback. Remove only
    # that foreign namespace so subsequent imports resolve to this checkout.
    tools_root = (repository_root / "tools").resolve()
    cached_tools = sys.modules.get("tools")
    if cached_tools is not None and not _module_is_inside(cached_tools, tools_root):
        for module_name in tuple(sys.modules):
            if module_name == "tools" or module_name.startswith("tools."):
                del sys.modules[module_name]
    importlib.invalidate_caches()
    import picf_next

    package_file = getattr(picf_next, "__file__", None)
    if package_file is None or Path(package_file).resolve().parent != expected_package:
        raise RuntimeError(f"{entrypoint_name} did not import picf_next from its own checkout")
    return repository_root

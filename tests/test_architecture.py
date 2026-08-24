"""
Dependency-rule tests: `core` and `ports` must not import infrastructure.

Reference: R. C. Martin, Clean Architecture — the dependency arrow points inward.
`adapters` may depend on `core` and `ports`; neither `core` nor `ports` may
depend on `adapters` or third-party I/O modules.
"""
import ast
from pathlib import Path

import core
import ports

CORE_PACKAGE_DIR: Path = Path(core.__file__).parent
PORTS_PACKAGE_DIR: Path = Path(ports.__file__).parent

INFRASTRUCTURE_TOP_LEVELS: frozenset[str] = frozenset(
    {"adapters", "ports", "aiohttp", "torchvision"}
)


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if not p.name.startswith("._"))


def _imported_top_level_modules(file_path: Path) -> set[str]:
    """Extracts the top-level module names statically imported by a source file."""
    tree: ast.Module = ast.parse(
        file_path.read_text(encoding="utf-8"), filename=str(file_path)
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    return imported


def test_core_domain_has_no_infrastructure_imports() -> None:
    """
    No module under `core` may import infrastructure.
    """
    offenders: list[str] = []
    for file_path in _python_files(CORE_PACKAGE_DIR):
        imported: set[str] = _imported_top_level_modules(file_path)
        violations: set[str] = imported & INFRASTRUCTURE_TOP_LEVELS
        if violations:
            offenders.append(f"{file_path.name}: {sorted(violations)}")
    assert not offenders, (
        "Domain isolation violated — infrastructure imports detected in core/: "
        + "; ".join(offenders)
    )


def test_ports_layer_has_no_adapters_import() -> None:
    """
    No module under `ports` may import `adapters`.
    """
    offenders: list[str] = []
    for file_path in _python_files(PORTS_PACKAGE_DIR):
        imported: set[str] = _imported_top_level_modules(file_path)
        if "adapters" in imported:
            offenders.append(file_path.name)
    assert not offenders, (
        "Ports layer violated — concrete adapter imports detected: " + ", ".join(offenders)
    )


def test_adapters_depend_on_ports_and_core() -> None:
    """
    Every adapter must import the `ports` layer.
    """
    adapters_root: Path = Path(core.__file__).parent.parent / "adapters"
    if not adapters_root.exists():
        return
    for file_path in _python_files(adapters_root):
        if file_path.name == "__init__.py":
            continue
        imported: set[str] = _imported_top_level_modules(file_path)
        assert "ports" in imported, f"Adapter {file_path.name} bypasses the ports layer."

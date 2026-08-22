"""
LaTeX/TikZ package catalog and preamble construction.

Pure functions with no I/O and no global mutable state. Centralizes the mapping
from declared package dependencies to a self-contained compilable standalone
preamble, plus deterministic import detection for real-world sources.

Reference: cambiospaquetes.md decision D1 — the model predicts only the drawing
body; packages are resolved by catalog at compile time so the 512-token budget
stays reserved for geometry.
"""

import re
from dataclasses import dataclass

from core.exceptions import DomainError


@dataclass(frozen=True)
class PackageSpec:
    """Immutable specification of a LaTeX/TikZ package dependency."""

    name: str
    preamble_lines: tuple[str, ...]
    tikz_libraries: tuple[str, ...]
    engine: str


PACKAGE_CATALOG: dict[str, PackageSpec] = {
    "tikz": PackageSpec(
        name="tikz",
        preamble_lines=("\\usepackage{tikz}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "pgfplots": PackageSpec(
        name="pgfplots",
        preamble_lines=("\\usepackage{pgfplots}", "\\pgfplotsset{compat=1.18}"),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "circuitikz": PackageSpec(
        name="circuitikz",
        preamble_lines=("\\usepackage{circuitikz}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "tikz-cd": PackageSpec(
        name="tikz-cd",
        preamble_lines=("\\usepackage{tikz-cd}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "tikz-3dplot": PackageSpec(
        name="tikz-3dplot",
        preamble_lines=("\\usepackage{tikz-3dplot}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "amsmath": PackageSpec(
        name="amsmath",
        preamble_lines=("\\usepackage{amsmath}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "siunitx": PackageSpec(
        name="siunitx",
        preamble_lines=("\\usepackage{siunitx}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
    "xcolor": PackageSpec(
        name="xcolor",
        preamble_lines=("\\usepackage{xcolor}",),
        tikz_libraries=(),
        engine="pdflatex",
    ),
}

# Rich TikZ library bundle enabling positioned, decorated figures. Kept separate
# from the base ``tikz`` entry so the minimal F1 preamble remains untouched.
BASE_TIKZ_LIBRARIES: tuple[str, ...] = (
    "arrows.meta",
    "positioning",
    "calc",
    "patterns",
    "decorations.pathmorphing",
    "shapes.geometric",
)

# Command fingerprints mapped to their package dependency. Order is fixed so
# detection is deterministic; catalog order is preserved on return.
_COMMAND_PACKAGE_HINTS: tuple[tuple[str, str], ...] = (
    (r"\\addplot", "pgfplots"),
    (r"\\begin\{axis\}", "pgfplots"),
    (r"\\begin\{tikzcd\}", "tikz-cd"),
    (r"\\arrow\b", "tikz-cd"),
    (r"\\tdplotsetmaincoords", "tikz-3dplot"),
    (r"\\tdplotsetrotatedcoords", "tikz-3dplot"),
    (r"to\s*\[\s*(?:R|C|L|V|I|battery|short|open|ammeter|voltmeter|european)\b", "circuitikz"),
    (r"\\num\b", "siunitx"),
    (r"\\si\b", "siunitx"),
    (r"\\definecolor", "xcolor"),
    (r"\\color\b", "xcolor"),
)


def _resolve_packages(packages: tuple[str, ...]) -> list[str]:
    """
    Validates and deduplicates declared packages, prepending the ``tikz`` base.

    Raises:
        DomainError: If any declared package is absent from the catalog.
    """
    resolved: list[str] = []
    for package in packages:
        if package not in PACKAGE_CATALOG:
            raise DomainError(
                f"Unknown package '{package}'. Known packages: {tuple(PACKAGE_CATALOG)}."
            )
        if package not in resolved:
            resolved.append(package)
    if "tikz" not in resolved:
        resolved.insert(0, "tikz")
    return resolved


def build_preamble(
    packages: tuple[str, ...], tikz_libraries: tuple[str, ...] = ()
) -> str:
    """
    Builds a complete standalone document preamble from declared packages.

    The ``tikz`` package is always prepended as the base dependency when not
    explicitly declared, so every produced preamble is self-sufficient. Preamble
    lines and TikZ libraries are deduplicated while preserving first-seen order.

    Args:
        packages (tuple[str, ...]): Declared package dependencies, each present
            in ``PACKAGE_CATALOG``.
        tikz_libraries (tuple[str, ...]): Additional TikZ libraries merged with
            any catalog-declared libraries into a single ``\\usetikzlibrary`` line.

    Returns:
        str: Newline-terminated preamble ending after the package/libraries
            block. The caller appends ``\\begin{document}`` and the body.

    Raises:
        DomainError: If any package is absent from the catalog.

    Temporal complexity: O(P + L) where P is the package count and L the library count.
    """
    resolved: list[str] = _resolve_packages(packages)

    preamble_lines: list[str] = ["\\documentclass{standalone}"]
    seen_lines: set[str] = set()
    libraries: list[str] = []
    seen_libraries: set[str] = set()

    for package_name in resolved:
        spec: PackageSpec = PACKAGE_CATALOG[package_name]
        for line in spec.preamble_lines:
            if line not in seen_lines:
                seen_lines.add(line)
                preamble_lines.append(line)
        for library in spec.tikz_libraries:
            if library not in seen_libraries:
                seen_libraries.add(library)
                libraries.append(library)

    for library in tikz_libraries:
        if library not in seen_libraries:
            seen_libraries.add(library)
            libraries.append(library)

    if libraries:
        preamble_lines.append("\\usetikzlibrary{" + ",".join(libraries) + "}")

    return "\n".join(preamble_lines) + "\n"


def detect_required_packages(markup: str) -> tuple[str, ...]:
    """
    Infers required package dependencies from observed drawing commands.

    Maps command fingerprints (e.g. ``\\addplot`` -> ``pgfplots``) to package
    names. No preamble scanning is performed; use ``extract_preamble_imports``
    for sources that declare imports explicitly.

    Args:
        markup (str): Candidate TikZ markup body.

    Returns:
        tuple[str, ...]: Deduplicated package names in catalog order.

    Temporal complexity: O(H * L) where H is the hint count and L the markup length.
    """
    detected: set[str] = {
        package_name
        for pattern, package_name in _COMMAND_PACKAGE_HINTS
        if re.search(pattern, markup)
    }
    return tuple(package for package in PACKAGE_CATALOG if package in detected)


def extract_preamble_imports(raw_text: str) -> tuple[str, ...]:
    """
    Extracts declared package dependencies from a raw LaTeX document source.

    Scans ``\\usepackage`` and ``\\usetikzlibrary`` macros. Package names are
    filtered to those present in ``PACKAGE_CATALOG``; any TikZ library import
    folds into the implicit ``tikz`` base dependency.

    Args:
        raw_text (str): Raw LaTeX document source.

    Returns:
        tuple[str, ...]: Deduplicated known package names in catalog order.

    Temporal complexity: O(N) where N is the document length.
    """
    declared: set[str] = set()

    for match in re.finditer(r"\\usepackage\s*\{([^}]*)\}", raw_text):
        for name in match.group(1).split(","):
            candidate: str = name.strip()
            if candidate in PACKAGE_CATALOG:
                declared.add(candidate)

    for match in re.finditer(r"\\usetikzlibrary\s*\{([^}]*)\}", raw_text):
        for name in match.group(1).split(","):
            if name.strip():
                declared.add("tikz")

    return tuple(package for package in PACKAGE_CATALOG if package in declared)

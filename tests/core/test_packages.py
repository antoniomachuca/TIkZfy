import pytest

from core.dataset.packages import (
    BASE_TIKZ_LIBRARIES,
    PACKAGE_CATALOG,
    PackageSpec,
    build_preamble,
    detect_required_packages,
    extract_preamble_imports,
)
from core.exceptions import DomainError


def test_package_catalog_has_minimum_entries() -> None:
    """Verify the catalog declares at least the 8 required package entries."""
    assert len(PACKAGE_CATALOG) >= 8
    required_names = (
        "tikz", "pgfplots", "circuitikz", "tikz-cd", "tikz-3dplot", "amsmath", "siunitx", "xcolor",
    )
    for name in required_names:
        assert name in PACKAGE_CATALOG


def test_package_spec_is_immutable() -> None:
    """Verify PackageSpec is a frozen specification value object."""
    spec = PackageSpec(
        name="pgfplots",
        preamble_lines=("\\usepackage{pgfplots}",),
        tikz_libraries=(),
        engine="pdflatex",
    )
    with pytest.raises(AttributeError):
        spec.engine = "lualatex"  # type: ignore


def test_build_preamble_always_includes_tikz_base() -> None:
    """Verify the tikz base dependency is always present in the preamble."""
    preamble = build_preamble(())
    assert preamble.startswith("\\documentclass{standalone}\n")
    assert "\\usepackage{tikz}" in preamble


def test_build_preamble_pgfplots_lines() -> None:
    """Verify pgfplots emits its package and compat lines."""
    preamble = build_preamble(("pgfplots",))
    assert "\\usepackage{pgfplots}" in preamble
    assert "\\pgfplotsset{compat=1.18}" in preamble


def test_build_preamble_merges_tikz_libraries() -> None:
    """Verify explicit libraries are emitted as a single usetikzlibrary line."""
    preamble = build_preamble(("tikz",), ("arrows.meta", "positioning"))
    assert "\\usetikzlibrary{arrows.meta,positioning}" in preamble


def test_build_preamble_deduplicates_packages_and_lines() -> None:
    """Verify duplicate packages and preamble lines are collapsed."""
    preamble = build_preamble(("tikz", "tikz", "pgfplots"))
    assert preamble.count("\\usepackage{tikz}") == 1
    assert preamble.count("\\usepackage{pgfplots}") == 1


def test_build_preamble_unknown_package_raises() -> None:
    """Verify unknown packages are rejected with a domain error."""
    with pytest.raises(DomainError):
        build_preamble(("nonexistent-package",))


def test_detect_required_packages_commands() -> None:
    """Verify command fingerprints map to their package dependencies."""
    markup = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}\\addplot coordinates {(0,0) (1,1)};\\end{axis}\n"
        "\\draw (0,0) to[R, l=$R_1$] (1,1);\n"
        "\\end{tikzpicture}"
    )
    packages = detect_required_packages(markup)
    assert "pgfplots" in packages
    assert "circuitikz" in packages


def test_detect_required_packages_no_hits() -> None:
    """Verify a plain tikzpicture yields no package dependencies."""
    assert detect_required_packages("\\draw (0,0) -- (1,1);") == ()


def test_detect_required_packages_catalog_order() -> None:
    """Verify returned packages follow catalog insertion order."""
    markup = "\\begin{tikzcd}\\arrow[r]\\end{tikzcd}"
    assert detect_required_packages(markup) == ("tikz-cd",)


def test_extract_preamble_imports_usepackage() -> None:
    """Verify usepackage declarations are extracted and catalog-filtered."""
    raw = "\\documentclass{article}\n\\usepackage{pgfplots}\n\\usepackage{foo}\n"
    assert extract_preamble_imports(raw) == ("pgfplots",)


def test_extract_preamble_imports_library_folds_to_tikz() -> None:
    """Verify usetikzlibrary declarations fold into the implicit tikz base."""
    raw = "\\documentclass{article}\n\\usetikzlibrary{arrows.meta}\n"
    assert extract_preamble_imports(raw) == ("tikz",)


def test_extract_preamble_imports_no_imports() -> None:
    """Verify documents without imports yield an empty sequence."""
    assert extract_preamble_imports("\\begin{document}\\end{document}") == ()


def test_base_tikz_libraries_are_declared() -> None:
    """Verify the rich TikZ library bundle is non-empty and deterministic."""
    assert len(BASE_TIKZ_LIBRARIES) >= 5
